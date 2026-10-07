"""Direct PICO controller targets for SONIC's existing TELEOP encoder.

No body reconstruction, inverse arm controller, or motor commands live here.
"""

import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# Direct XRT poses use RH tracking coordinates: +X right, +Y up, -Z forward.
# This is the upstream XRoboToolkit Teleop Sample's R_HEADSET_TO_WORLD, NOT
# the Unity/body-joint conversion in pico_manager_thread_server.
XRT_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class ArmsOnlyTargets:
    """Calibrated fixed-frame incremental mapping, with latched target hold."""

    MAX_STEP_M = 0.10
    MAX_STEP_RAD = np.deg2rad(45)
    MAX_EXCURSION_M = 0.60

    def __init__(self, fk, grip_offsets=None):
        self.fk = fk
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
        return (positions @ self.frame.T, R.from_matrix(self.frame @ rotations.as_matrix() @ XRT_TO_ROBOT.T))

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
        self.source_timestamp_ns = sample["timestamp_ns"]
        self.held, self.reason = False, ""
        return self.target.copy()

    def update(self, sample, now=None):
        now = time.monotonic() if now is None else now
        if sample is None or not 0 <= now - sample["timestamp_monotonic"] <= 0.25:
            self.hold("Invalid or stale tracking")
        if self.target is None or self.held:
            return None if self.target is None else self.target.copy()
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
        delta_m = np.linalg.norm(candidate[:2, :3] - self.target[:2, :3], axis=1)
        delta_r = (
            R.from_quat(candidate[:2, 3:], scalar_first=True)
            * R.from_quat(self.target[:2, 3:], scalar_first=True).inv()
        ).magnitude()
        excursion = np.linalg.norm(candidate[:2, :3] - self.workspace_origin, axis=1)
        if (
            np.any(delta_m > self.MAX_STEP_M)
            or np.any(delta_r > self.MAX_STEP_RAD)
            or np.any(excursion > self.MAX_EXCURSION_M)
        ):
            self.hold("Target discontinuity or workspace excursion")
            return self.target.copy()
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
        return action, self.targets.update(sample, now) if self.active else None


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
            except (KeyError, ValueError, TypeError, msgpack.UnpackException):
                self.q = None
        return self.q if 0 <= now - self.received_at <= 0.25 else None


def run_arms_only_manager(args, sdk, init_hands, compute_hands, read_buttons, read_inputs, read_clicks):
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
    reader = PicoControllerReader(sdk)
    targets = ArmsOnlyTargets(make_robot_fk(), [args.left_grip_offset, args.right_grip_offset])
    session = ArmsOnlySession(targets)
    solvers = init_hands()
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.bind(f"tcp://*:{args.port}")
    feedback = MeasuredRobotFeedback(context, args.zmq_feedback_host, args.zmq_feedback_port)
    last_hands = (None, None)
    previous_collection = previous_abort = False
    initialized = False
    started = False
    logging.basicConfig(level=logging.INFO)
    try:
        subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
        sdk.init()
        initialized = True
        print(
            "Arms-only: enable Head + Controller, disable Body/Motion tracking. "
            "A+B+X+Y=start/stop; left stick click=reanchor; left menu=hold; Ctrl-C=stop."
        )
        while True:
            tick = time.monotonic()
            # Stop controls are read regardless of pose freshness or validity.
            buttons = read_buttons()
            menu, lt, rt, lg, rg = read_inputs()
            left_click, _ = read_clicks()
            sample = reader.get_latest()
            action, target = session.step(sample, feedback.get(tick), buttons, left_click, menu, time.monotonic())
            if action == "stop":
                break
            if target is not None:
                # Retain last Dex3 command on tracking loss/menu hold, but use
                # unchanged legacy trigger/grip IK while tracking is active.
                if not targets.held:
                    lh, rh = compute_hands(*solvers, lt, lg, rt, rg)
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
            socket.send(build_command_message(start=False, stop=True, planner=True))
            time.sleep(0.05)
        if initialized:
            sdk.close()
        feedback.socket.close(linger=0)
        socket.close(linger=100)
        context.term()
