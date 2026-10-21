"""Direct PICO controller targets for SONIC's existing TELEOP encoder.

No body reconstruction, inverse arm controller, or motor commands live here.
"""

import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# Direct XRT poses use RH tracking coordinates: +X right, +Y up, -Z forward.
# This is the upstream XRoboToolkit Teleop Sample's R_HEADSET_TO_WORLD, NOT
# the Unity/body-joint conversion in pico_manager_thread_server.
XRT_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class ArmsOnlyTargets:
    """Fixed-frame mapping: recoverable target projection, latched tracking hold."""

    MAX_STEP_M = 0.10
    MAX_STEP_RAD = np.deg2rad(45)
    MAX_EXCURSION_M = 0.60

    def __init__(self, fk, grip_offsets=None, target_filter=None, mirror=False):
        self.fk = fk
        self.mirror = bool(mirror)
        self.target_filter = target_filter
        self.requested_pose = None
        self.limit_status = ["", ""]
        self.grip_offsets = np.zeros((2, 3)) if grip_offsets is None else np.asarray(grip_offsets)
        if self.grip_offsets.shape != (2, 3) or not np.all(np.isfinite(self.grip_offsets)):
            raise ValueError("Grip offsets must be two finite XYZ vectors in metres")
        self.target = None
        self.held = True
        self.reason = "Not calibrated"
        self.source_timestamp_ns = 0

    def hold(self, reason):
        if not self.held:
            logger.warning("Arms-only HOLD: %s; left stick click to reanchor", reason)
        self.held, self.reason = True, reason

    def _controllers(self, sample):
        poses = sample["controller_poses"]
        rotations = R.from_quat(poses[:, 3:])
        # Optional physical grip-to-command-point lever arm, in controller-local
        # tracking axes. Default zero: grip centre commands the robot key point.
        positions = poses[:, :3] + rotations.apply(self.grip_offsets)
        positions = positions @ self.frame.T
        rotations = R.from_matrix(self.frame @ rotations.as_matrix() @ XRT_TO_ROBOT.T)
        if self.mirror:
            from gear_sonic.utils.teleop.mirroring import mirror_controller_targets

            positions, rotations = mirror_controller_targets(positions, rotations)
        return positions, rotations

    def calibrate(self, sample, measured_q, now=None):
        now = time.monotonic() if now is None else now
        if sample is None or not 0 <= now - sample["timestamp_monotonic"] <= 0.25:
            raise ValueError("Fresh controller poses required for calibration")
        measured_q = np.asarray(measured_q, dtype=float)
        if measured_q.shape != (29,) or not np.all(np.isfinite(measured_q)):
            raise ValueError("Fresh measured body_q_measured (29 joints) required; no zero fallback")
        measured = np.asarray(self.fk(measured_q), dtype=float)
        if measured.shape != (3, 7) or not np.all(np.isfinite(measured)):
            raise ValueError("Invalid measured robot FK")
        if self.target is None:
            head = sample["headset_pose"]
            if head is None:
                raise ValueError("A valid headset pose is required for initial yaw alignment")
            forward = XRT_TO_ROBOT @ R.from_quat(head[3:]).apply([0, 0, -1])
            if np.linalg.norm(forward[:2]) < 0.2:
                raise ValueError("Look horizontally forward for initial calibration")
            yaw = np.arctan2(forward[1], forward[0])
            self.frame = R.from_euler("z", -yaw).as_matrix() @ XRT_TO_ROBOT
            self.target = measured.copy()
            self.workspace_origin = measured[:2, :3].copy()
        # During reanchor, keep the outgoing target (including torso) exactly.
        # Replacing it with measured FK would jump by the policy tracking error.
        # Still require fresh FK, and expose that residual for trial diagnostics.
        self.measured_at_calibration = measured
        self.calibration_residual_m = np.linalg.norm(self.target[:2, :3] - measured[:2, :3], axis=1)
        self.anchor = self.target.copy()
        self.controller_origin, self.controller_rotation = self._controllers(sample)
        self.last_controller_positions = self.controller_origin.copy()
        self.last_controller_rotations = self.controller_rotation
        self.requested_pose = self.target.copy()
        if self.target_filter is not None:
            self.target_filter.reset(measured_q, self.target)
        self.source_timestamp_ns = sample["timestamp_ns"]
        self.held, self.reason = False, ""
        return self.target.copy()

    def update(self, sample, now=None, measured_q=None):
        now = time.monotonic() if now is None else now
        if sample is None or not 0 <= now - sample["timestamp_monotonic"] <= 0.25:
            self.hold("Invalid or stale tracking")
        if self.target is None or self.held:
            return None if self.target is None else self.target.copy()
        if self.target_filter is not None and measured_q is None:
            self.hold("Measured robot feedback missing/stale")
            return self.target.copy()
        if sample["timestamp_ns"] == self.source_timestamp_ns:
            return self.target.copy()
        if sample["timestamp_ns"] < self.source_timestamp_ns:
            self.hold("Source clock reset")
            return self.target.copy()
        positions, rotations = self._controllers(sample)
        candidate = self.anchor.copy()
        candidate[:2, :3] += positions - self.controller_origin
        # Spatial rotation delta, followed by measured wrist orientation. This
        # calibrates each grip-to-wrist orientation without skeleton OFFSETS.
        candidate[:2, 3:] = (
            rotations * self.controller_rotation.inv() * R.from_quat(self.anchor[:2, 3:], scalar_first=True)
        ).as_quat(scalar_first=True)
        # Detect raw tracking jumps, independently of a limited/lagging target.
        # Comparing against the output caused ordinary motion at a boundary to
        # look like an ever-growing discontinuity.
        delta_m = np.linalg.norm(positions - self.last_controller_positions, axis=1)
        delta_r = (rotations * self.last_controller_rotations.inv()).magnitude()
        for i, side in enumerate(("left", "right")):
            if delta_m[i] > self.MAX_STEP_M or delta_r[i] > self.MAX_STEP_RAD:
                self.hold(
                    f"{side} tracking jump: translation={delta_m[i]:.4f}m "
                    f"rotation={np.rad2deg(delta_r[i]):.1f}deg "
                    f"limits={self.MAX_STEP_M}m/{np.rad2deg(self.MAX_STEP_RAD):.0f}deg"
                )
                return self.target.copy()
        self.last_controller_positions = positions.copy()
        self.last_controller_rotations = rotations
        self.requested_pose = candidate.copy()
        # A bounded workspace is a recoverable projection, not a latched hold.
        # Left/right limiting is independent; inward motion remains available.
        excursion = np.linalg.norm(candidate[:2, :3] - self.workspace_origin, axis=1)
        self.limit_status = ["workspace" if distance > self.MAX_EXCURSION_M else "" for distance in excursion]
        for i, distance in enumerate(excursion):
            if distance > self.MAX_EXCURSION_M:
                candidate[i, :3] = (
                    self.workspace_origin[i]
                    + (candidate[i, :3] - self.workspace_origin[i]) * self.MAX_EXCURSION_M / distance
                )
        if self.target_filter is not None:
            candidate = self.target_filter.project(candidate, self.target)
            self.limit_status = [
                ";".join(s for s in (workspace, posture) if s)
                for workspace, posture in zip(self.limit_status, self.target_filter.status)
            ]
        # Absorb motion rejected by projection, avoiding a hidden backlog of
        # unreachable commands. A reverse controller movement takes effect on
        # the very next sample, even after moving far beyond the boundary.
        self.controller_origin += self.requested_pose[:2, :3] - candidate[:2, :3]
        self.controller_rotation = (
            R.from_quat(self.anchor[:2, 3:], scalar_first=True)
            * R.from_quat(candidate[:2, 3:], scalar_first=True).inv()
            * rotations
        )
        # Preserve quaternion sign for downstream consumers too.
        flip = np.sum(candidate[:, 3:] * self.target[:, 3:], axis=1) < 0
        candidate[flip, 3:] *= -1
        self.target = candidate
        self.source_timestamp_ns = sample["timestamp_ns"]
        return self.target.copy()


def make_robot_fk():
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses

    model = instantiate_g1_robot_model()

    def fk(measured_q):
        q = model.get_configuration_from_actuated_joints(body_actuated_joint_values=measured_q)
        poses = get_g1_key_frame_poses(model, q=q)
        return np.stack(
            [
                np.r_[poses[name]["position"], poses[name]["orientation_wxyz"]]
                for name in ("left_wrist", "right_wrist", "torso")
            ]
        )

    return fk


def make_target_filter(max_gravity_nm):
    from gear_sonic.utils.teleop.arm_target_filter import G1ArmTargetFilter

    return G1ArmTargetFilter(max_gravity_nm=max_gravity_nm)


class ArmsOnlySession:
    """Only OFF and VR_3PT are reachable; all locomotion axes are ignored."""

    def __init__(self, targets):
        self.targets = targets
        self.active = False
        self.prev_start = False
        self.prev_click = False

    def step(self, sample, measured_q, buttons, left_click=False, menu=False, now=None):
        start = all(buttons)  # A+B+X+Y has priority even when tracking is invalid.
        start_edge = start and not self.prev_start
        click_edge = left_click and not self.prev_click
        self.prev_start, self.prev_click = start, left_click
        if self.active and start_edge:
            self.active = False
            return "stop", None
        action = None
        if (not self.active and start_edge) or (self.active and click_edge and not start):
            try:
                self.targets.calibrate(sample, measured_q, now)
                action = "start" if not self.active else "reanchor"
                self.active = True
                logger.info(
                    "Arms-only calibrated; target/FK residual (m): %s", self.targets.calibration_residual_m
                )
            except ValueError as exc:
                self.targets.hold(str(exc))
                logger.warning("Arms-only calibration refused: %s", exc)
        if menu:
            self.targets.hold("Left menu pressed")
        return action, self.targets.update(sample, now, measured_q) if self.active else None


class MeasuredRobotFeedback:
    """Continuously drain a conflated g1_calibration stream; never use a zero fallback."""

    def __init__(self, context, host, port):
        import zmq

        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "g1_calibration")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f"tcp://{host}:{port}")
        self.received_at = -float("inf")
        self.q = None
        self.telemetry = {}

    def get(self, now):
        import msgpack
        import zmq

        if self.socket.poll(0):
            try:
                packet = msgpack.unpackb(self.socket.recv(zmq.NOBLOCK)[len(b"g1_calibration") :], raw=False)
                age = float(packet["state_age_s"])
                if not np.isfinite(age) or not 0 <= age <= 0.05:
                    raise ValueError("Stale robot low state")
                q = np.asarray(packet["body_q_measured"], dtype=float)
                if q.shape != (29,) or not np.all(np.isfinite(q)):
                    raise ValueError("Invalid measured robot joints")
                self.q, self.received_at = q, now
                self.telemetry = packet
            except (KeyError, ValueError, TypeError, msgpack.UnpackException):
                self.q = None
                self.telemetry = {}
        if not 0 <= now - self.received_at <= 0.25:
            self.telemetry = {}
            return None
        return self.q


class ArmDiagnostics:
    """JSONL trial data plus one-second summaries, including while targets hold."""

    def __init__(self, path="", context=None):
        self.file = None
        self.last_summary = -float("inf")
        self.warned_missing_telemetry = False
        if path:
            destination = Path(path).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.file = destination.open("a", encoding="utf-8")
            self.file.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "schema": "arms-only-diagnostics-v1",
                        "context": context or {},
                        "run_started_unix_ns": time.time_ns(),
                        "units": "metres/radians/Nm/degrees-C",
                        "quaternion_order": "wxyz",
                        "frame": "robot root",
                        "arm_motor_indices": {"left": list(range(15, 22)), "right": list(range(22, 29))},
                        "temperature_order": "hardware order; winding, driver per motor",
                        "gravity_estimate": "unloaded URDF; excludes payload/contact/dynamics",
                        "joint_order_per_arm": [
                            "shoulder_pitch",
                            "shoulder_roll",
                            "shoulder_yaw",
                            "elbow",
                            "wrist_roll",
                            "wrist_pitch",
                            "wrist_yaw",
                        ],
                    }
                )
                + "\n"
            )

    @staticmethod
    def _array(packet, name, shape):
        value = np.asarray(packet.get(name, []), dtype=float)
        return value if value.shape == shape and np.all(np.isfinite(value)) else None

    def event(self, name, payload):
        if self.file is not None:
            self.file.write(json.dumps({"type": "event", "event": name, **payload}) + "\n")
            self.file.flush()

    def record(self, targets, sample, measured_q, telemetry, sent_mono, sent_unix_ns, processing_ms):
        summarize = sent_mono - self.last_summary >= 1.0
        if self.file is None and not summarize:
            return
        observation = {}
        observer = getattr(targets, "diagnostics_observer", targets.target_filter)
        if measured_q is not None and observer is not None:
            observation = observer.observe(measured_q, targets.target)
        temperatures = self._array(telemetry, "motor_temperature", (58,))
        torques = self._array(telemetry, "motor_torque", (29,))
        velocities = self._array(telemetry, "body_dq_measured", (29,))
        if (
            measured_q is not None
            and (temperatures is None or torques is None)
            and not self.warned_missing_telemetry
        ):
            logger.warning("Feedback lacks temperature/torque telemetry; rebuild the updated C++ deployment")
            self.warned_missing_telemetry = True
        arm_temperatures = None if temperatures is None else temperatures.reshape(29, 2)[15:29]
        row = {
            "type": "target",
            "source_ns": targets.source_timestamp_ns,
            "incoming_source_ns": sample["timestamp_ns"] if sample else None,
            "source_receive_mono": sample["timestamp_monotonic"] if sample else None,
            "sent_mono": sent_mono,
            "sent_unix_ns": sent_unix_ns,
            "replay_elapsed_s": getattr(targets, "replay_elapsed_s", None),
            "feedback_timestamp_unix_ns": telemetry.get("timestamp_unix_ns"),
            "feedback_state_age_s": telemetry.get("state_age_s"),
            "processing_ms": processing_ms,
            "gravity_budget_nm": targets.target_filter.max_gravity_nm if targets.target_filter else None,
            "held": targets.held,
            "hold_reason": targets.reason,
            "limits": targets.limit_status,
            "controller_pose_xyzw": sample["controller_poses"].tolist() if sample else None,
            "requested_pose": targets.requested_pose.tolist(),
            "outgoing_pose": targets.target.tolist(),
            "projection_position_error_m": np.linalg.norm(
                targets.requested_pose[:2, :3] - targets.target[:2, :3], axis=1
            ).tolist(),
            "projection_rotation_error_rad": (
                R.from_quat(targets.requested_pose[:2, 3:], scalar_first=True)
                * R.from_quat(targets.target[:2, 3:], scalar_first=True).inv()
            )
            .magnitude()
            .tolist(),
            "measured_body_q": measured_q.tolist() if measured_q is not None else None,
            "body_dq_measured": velocities.tolist() if velocities is not None else None,
            "motor_torque": torques.tolist() if torques is not None else None,
            "motor_temperature": temperatures.tolist() if temperatures is not None else None,
            "projection_metrics": targets.target_filter.solution_diagnostics if targets.target_filter else [],
            **observation,
        }
        if self.file is not None:
            self.file.write(json.dumps(row, allow_nan=False) + "\n")
        if summarize:
            self.last_summary = sent_mono
            max_temperature = float(np.max(arm_temperatures)) if arm_temperatures is not None else None
            logger.info(
                "Arms diagnostics: hold=%s limits=%s wrist_error_m=%s arm_max_temperature_C=%s processing_ms=%.2f",
                targets.held,
                targets.limit_status,
                observation.get("wrist_error_m"),
                max_temperature,
                processing_ms,
            )
            if observation:
                logger.info(
                    "Measured arm posture: elbow_rad=%s min_joint_margin_rad=%s gravity_Nm=%s",
                    [arm[3] for arm in observation["measured_arm_q"]],
                    [arm["min_joint_limit_margin_rad"] for arm in observation["measured_arm_metrics"]],
                    [arm["gravity_nm"] for arm in observation["measured_arm_metrics"]],
                )
            if max_temperature is not None and max_temperature >= 90:
                logger.warning(
                    "ARM OVERHEATING: %.1f C; end the trial using the deployment stop. "
                    "Target hold does not unload joints.",
                    max_temperature,
                )
            if self.file is not None:
                self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def run_arms_only_manager(
    args, sdk, init_hands, compute_hands, read_buttons, read_inputs, read_clicks,
    *, replay_driver=None, target_mode="filtered",
):
    """Dedicated acquisition/manager path: no SMPL startup, calibration or modes."""
    import subprocess

    import zmq

    from gear_sonic.utils.teleop.input_readers import PicoControllerReader
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        build_command_message,
        build_planner_message,
        pack_pose_message,
    )

    # Fail on unsupported SDK/model before opening the service or publisher.
    reader = (
        PicoControllerReader(sdk, clock=replay_driver.input_clock)
        if replay_driver else PicoControllerReader(sdk)
    )
    target_class = ArmsOnlyTargets
    if target_mode == "direct":
        if replay_driver is None:
            raise ValueError("Direct comparison mapping is available only for simulation replay")
        from gear_sonic.utils.teleop.demo_targets import DirectDemoTargets

        target_class = DirectDemoTargets
    elif target_mode != "filtered":
        raise ValueError("Unknown target mapping")
    targets = target_class(
        make_robot_fk(),
        [args.left_grip_offset, args.right_grip_offset],
        make_target_filter(args.arms_max_gravity_nm) if target_mode == "filtered" else None,
        mirror=getattr(args, "mirror", False),
    )
    if replay_driver and target_mode == "direct":
        targets.diagnostics_observer = make_target_filter(args.arms_max_gravity_nm)
    session = ArmsOnlySession(targets)
    solvers = init_hands()
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.bind(f"tcp://{'127.0.0.1' if replay_driver else '*'}:{args.port}")
    feedback = MeasuredRobotFeedback(context, args.zmq_feedback_host, args.zmq_feedback_port)
    last_hands = (None, None)
    previous_collection = previous_abort = False
    initialized = False
    started = False
    diagnostics = ArmDiagnostics(args.arms_only_log, getattr(replay_driver, "configuration", None))
    logging.basicConfig(level=logging.INFO)
    try:
        if replay_driver is None:
            subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
        sdk.init()
        initialized = True
        print(
            "Arms-only: enable Head + Controller, disable Body/Motion tracking. "
            "A+B+X+Y=start/stop; left stick click=reanchor; left menu=hold; Ctrl-C=stop."
        )
        waiting_since = time.monotonic()
        checked_sim_at = -float("inf")
        while True:
            tick = time.monotonic()
            measured_q = feedback.get(tick)
            if replay_driver:
                from gear_sonic.utils.teleop.pico_demo import require_local_simulation

                if tick - checked_sim_at >= 1.0:
                    require_local_simulation()
                    checked_sim_at = tick
                if replay_driver.started_at is None:
                    if measured_q is None or tick - waiting_since < 0.5:
                        if tick - waiting_since > 15:
                            raise RuntimeError(
                                "No fresh simulation joints on feedback port; enable --output-type zmq"
                            )
                        time.sleep(0.02)
                        continue
                    replay_driver.start()
                    print("Replaying recorded input; initial calibration uses simulated measured joints.")
                if not replay_driver.advance():
                    print("Recording finished; stopping simulation replay.")
                    break
                targets.replay_elapsed_s = replay_driver.elapsed_s
            # Stop controls are read regardless of pose freshness or validity.
            buttons = read_buttons()
            menu, lt, rt, lg, rg = read_inputs()
            left_click, _ = read_clicks()
            sample = reader.get_latest()
            input_now = time.monotonic()
            if replay_driver:
                buttons = replay_driver.buttons(sample, session)
                input_now = replay_driver.input_clock()
                if measured_q is None and session.active:
                    targets.hold("Measured robot feedback missing/stale")
            action, target = session.step(sample, measured_q, buttons, left_click, menu, input_now)
            if action in ("start", "reanchor"):
                diagnostics.event("calibration", {
                    "action": action,
                    "replay_elapsed_s": replay_driver.elapsed_s if replay_driver else None,
                    "source_ns": targets.source_timestamp_ns,
                    "measured_body_q": measured_q.tolist(),
                    "headset_pose_xyzw": (
                        sample["headset_pose"].tolist() if sample["headset_pose"] is not None else None
                    ),
                    "controller_poses_xyzw": sample["controller_poses"].tolist(),
                    "frame": targets.frame.tolist(), "anchor_pose": targets.anchor.tolist(),
                    "grip_offsets": targets.grip_offsets.tolist(),
                    "mirror": targets.mirror,
                    "gripper_profile": getattr(args, "gripper_profile", "legacy"),
                })
            if action == "stop":
                break
            if target is not None:
                # Retain last Dex3 command on tracking loss/menu hold, but use
                # unchanged legacy trigger/grip IK while tracking is active.
                if not targets.held:
                    from gear_sonic.utils.teleop.mirroring import route_hand_inputs

                    hand_inputs = route_hand_inputs(lt, lg, rt, rg, targets.mirror)
                    lh, rh = compute_hands(*solvers, *hand_inputs)
                    last_hands = (lh.reshape(-1).tolist(), rh.reshape(-1).tolist())
                socket.send(
                    build_planner_message(
                        0,
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        speed=-1.0,
                        height=-1.0,
                        left_hand_position=last_hands[0],
                        right_hand_position=last_hands[1],
                        vr_3pt_position=target[:, :3].ravel().tolist(),
                        vr_3pt_orientation=target[:, 3:].ravel().tolist(),
                    )
                )
                # Seed targets before requesting the existing planner/TELEOP path.
                if action == "start":
                    started = True
                    socket.send(build_command_message(start=True, stop=False, planner=True))
                sent_mono, sent_unix_ns = time.monotonic(), time.time_ns()
                diagnostics.record(
                    targets,
                    sample,
                    measured_q,
                    feedback.telemetry,
                    sent_mono,
                    sent_unix_ns,
                    (sent_mono - tick) * 1000.0,
                )
                if args.arms_only_timing:
                    print(
                        f"arms_timing source_ns={targets.source_timestamp_ns} "
                        f"read_mono={sample['timestamp_monotonic'] if sample else -1:.9f} "
                        f"sent_mono={time.monotonic():.9f} sent_unix_ns={time.time_ns()} "
                        f"hold={int(targets.held)}",
                        flush=True,
                    )
            collection = bool(buttons[0]) and lg > 0.5
            abort = bool(buttons[1]) and lg > 0.5
            if replay_driver:
                collection = abort = False
            socket.send(
                pack_pose_message(
                    {
                        "stream_mode": np.array([5 if session.active else 0], dtype=np.int32),
                        "toggle_data_collection": np.array([collection and not previous_collection], dtype=bool),
                        "toggle_data_abort": np.array([abort and not previous_abort], dtype=bool),
                    },
                    topic="manager_state",
                )
            )
            previous_collection, previous_abort = collection, abort
            time.sleep(max(0.0, 1.0 / args.target_fps - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("Stopping arms-only manager...")
    finally:
        if started:
            # ZMQ PUB is best effort. Keep the independent deployment stop control.
            for _ in range(3 if replay_driver else 1):
                socket.send(build_command_message(start=False, stop=True, planner=True))
                time.sleep(0.05)
        if initialized:
            sdk.close()
        diagnostics.close()
        feedback.socket.close(linger=0)
        socket.close(linger=100)
        context.term()
    if replay_driver and not started:
        raise RuntimeError("No replay start occurred. Record A+B+X+Y in a neutral pose, or use --start-mode auto.")
