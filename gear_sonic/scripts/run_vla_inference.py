"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (SONIC latent v4 or stationary REAL_G1 planner targets)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> SONIC: initial token in POSE mode; REAL_G1: hold posture in PLANNER mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import queue
import threading
import time

import numpy as np
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.gr00t_adapter import Gr00tAdapter
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.rtc import RTCScheduler
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, pack_pose_message


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Legacy horizon hint; the actual horizon is read from the policy server."""

    rate: float = 1 / 0.4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    rtc: bool = False
    """Condition overlapping action chunks using the server's RTC implementation."""

    rtc_overlap_steps: int = 20
    """Replan when this many actions remain (expanded to cover measured latency)."""

    rtc_initial_delay: float = 0.2
    """Initial round-trip delay estimate (seconds), updated from subsequent calls."""

    rtc_safety_steps: int = 2
    """Additional frozen actions to cover control and transport jitter."""

    rtc_ramp_rate: float = 5.0
    """Exponential soft-overlap ramp strength."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "auto"
    """Infer the embodiment from the server, or check REAL_G1 / UNITREE_G1_SONIC."""

    observation_rate: float = 50.0
    """Checkpoint sampling rate (Hz): video [-20, 0] spans 0.4 s at 50 Hz."""

    sensor_timeout: float = 1.0
    """Maximum age (seconds) of camera and state observations."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Initial pose
    initial_pose_blend_duration: float = 1.0
    """Duration (seconds) for smooth interpolation to initial pose. The robot
    blends from its current motion token to the initial pose token over this
    period. Set to 0 to snap instantly (no blend)."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(f"left_hand_joints must have shape [7], got {left_hand_joints.shape}")
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(f"right_hand_joints must have shape [7], got {right_hand_joints.shape}")
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def run_policy_inference_and_process(policy, observation, adapter):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        action, _info = policy.get_action(observation)

        return adapter.process_action(action)
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    continue

                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)

                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


def _rtc_inference_worker_loop(
    inference_queue,
    result_queue,
    stop_event,
    busy_event,
    prepare_obs_fn,
    inference_fn,
):
    """Always return the request token, including failures and sensor warm-up."""
    while not stop_event.is_set():
        try:
            request = inference_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        busy_event.set()
        try:
            observation = prepare_obs_fn()
            if observation is None:
                result = (request, None, {"error": "waiting for fresh sensor history"})
            else:
                action, info = inference_fn(observation, request)
                result = (request, action, info)
        except Exception as exc:
            result = (request, None, {"error": str(exc)})
        finally:
            busy_event.clear()
        result_queue.put(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str, profile="rtc")
    return solver._get_middle_close_q_desired().astype(np.float32)


def main(config: InferenceConfig):
    pause_loop = True

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    # Isaac-GR00T PolicyClient
    from gr00t.policy.server_client import PolicyClient

    n1_policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        raise RuntimeError("PolicyServer is not reachable; start it before the VLA client")

    adapter = Gr00tAdapter(
        robot_model,
        n1_policy.get_modality_config(),
        config.embodiment_tag,
        config.observation_rate,
        config.sensor_timeout,
    )
    config.action_horizon = adapter.action_horizon
    rtc_scheduler = None
    if config.rtc:
        try:
            capabilities = n1_policy.call_endpoint("get_capabilities", requires_input=False)
        except RuntimeError as exc:
            raise RuntimeError("RTC requires the updated Isaac-GR00T server; restart that server") from exc
        if capabilities.get("rtc", {}).get("version") != 1:
            raise RuntimeError("This policy server does not support RTC version 1")
        rtc_scheduler = RTCScheduler(
            adapter.action_horizon,
            config.action_publish_rate,
            config.rtc_overlap_steps,
            config.rtc_initial_delay,
            config.rtc_safety_steps,
            config.rtc_ramp_rate,
        )
        print_green(
            f"RTC enabled: overlap={rtc_scheduler.overlap_steps}, initial freeze={rtc_scheduler.freeze_steps}"
        )
    print_green(f"Server embodiment: {adapter.embodiment_tag}")
    print(f"Video history: {adapter.video_deltas} at {config.observation_rate} Hz")
    print(f"Action horizon from server: {adapter.action_horizon}")
    if not adapter.is_latent:
        print_green("REAL_G1 stationary control: arms, hands and waist; navigation/height disabled")
        print("'i' holds the measured posture in PLANNER mode; 'p' applies VLA targets")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    camera_subscriber = ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    print_green(f"Using server embodiment: {adapter.embodiment_tag}")

    keyboard_listener = ZMQKeyboardSubscriber(port=config.keyboard_zmq_port, host=config.keyboard_zmq_host)

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False
    last_stationary_message = None

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position."""
        nonlocal last_stationary_message
        print("Moving to initial pose")
        left_hand = (
            _compute_closed_hand_joints("L") if initial_pose_left_hand_closed else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R") if initial_pose_right_hand_closed else np.zeros(7, dtype=np.float32)
        )
        if not adapter.is_latent:
            try:
                last_stationary_message = adapter.stationary_message(left_hand=left_hand, right_hand=right_hand)
                zmq_socket.send(last_stationary_message)
            except ValueError as exc:
                print(f"Cannot initialize posture: {exc}")
                return
            print_green("Holding measured posture in stationary PLANNER mode")
            return
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def maintain_stationary_posture():
        # The planner releases upper-body overrides after one second without a
        # message. Keep the last posture while paused or awaiting a new chunk.
        if not adapter.is_latent and cpp_loop_running and last_stationary_message is not None:
            zmq_socket.send(last_stationary_message)

    def blend_to_initial_pose(duration_s: float) -> bool:
        """Smoothly interpolate from the last sent motion token to the initial pose.

        Linearly blends over ``duration_s`` seconds at the action publish rate,
        sending intermediate tokens each loop iteration. Returns True if blend
        was performed, False if skipped (no previous token available).
        """
        nonlocal last_sent_motion_token
        if last_sent_motion_token is None:
            print("No previous motion token — snapping to initial pose instead.")
            publish_initial_pose()
            return False

        start_token = last_sent_motion_token.copy()
        target_token = LATENT_INITIAL_MOTION_TOKEN.copy()
        num_steps = max(1, round(config.action_publish_rate * duration_s))
        step_period = 1.0 / config.action_publish_rate

        left_hand = (
            _compute_closed_hand_joints("L") if initial_pose_left_hand_closed else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R") if initial_pose_right_hand_closed else np.zeros(7, dtype=np.float32)
        )

        print(
            f"Blending to initial pose over {duration_s:.2f}s "
            f"({num_steps} steps at {config.action_publish_rate} Hz)"
        )

        for step in range(num_steps):
            t_step_start = time.monotonic()
            alpha = (step + 1) / num_steps
            blended_token = ((1.0 - alpha) * start_token + alpha * target_token).astype(np.float32)
            zmq_message = pack_latent_action_message(
                motion_token=blended_token,
                frame_index=np.array([0], dtype=np.int64),
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
            )
            zmq_socket.send(zmq_message)
            last_sent_motion_token = blended_token.copy()

            elapsed = time.monotonic() - t_step_start
            remaining = step_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print_green("Initial pose blend complete.")
        return True

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate

    zmq_frame_counter = 0
    last_sent_motion_token: np.ndarray | None = None

    PROMPT_MSG_PREFIX = "prompt:"

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter, last_sent_motion_token

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX) :]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                if rtc_scheduler is not None:
                    rtc_scheduler.reset()
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            if rtc_scheduler is not None:
                rtc_scheduler.reset()
            target_mode = "POSE" if adapter.is_latent else "PLANNER"
            if cpp_loop_running and cpp_mode != target_mode:
                if send_cpp_control_command(start=True, planner=not adapter.is_latent):
                    print(f"Switched to {target_mode} mode")
                else:
                    print(f"Warning: Failed to switch to {target_mode} mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")

            pause_loop = True
            if config.initial_pose_blend_duration > 0 and last_sent_motion_token is not None:
                blend_to_initial_pose(config.initial_pose_blend_duration)
            else:
                publish_initial_pose()

            zmq_frame_counter = 0
            cached_action_chunk = None
            action_chunk_index = 0
            print("Cleared cached action chunk, reset frame counter")
        elif key == "p":
            if rtc_scheduler is not None:
                rtc_scheduler.reset()
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
        elif key == "k":
            if rtc_scheduler is not None:
                rtc_scheduler.reset()
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to initialize the posture, then 'p' to apply VLA actions")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}")
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(f"Initial pose right hand: {'closed' if initial_pose_right_hand_closed else 'open'}")

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    def rtc_inference(observation, request):
        action, info = n1_policy.get_action(observation, options=request.options)
        return adapter.process_action(action, retain_all=True), info

    inference_worker_thread = threading.Thread(
        target=_rtc_inference_worker_loop if config.rtc else _inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: adapter.prepare_observation(language_prompt_ref[0]),
            rtc_inference
            if config.rtc
            else lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                adapter=adapter,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()
    print("Waiting for fresh sensors and the checkpoint's video history...")
    sensors_were_ready = False

    try:
        while True:
            t_start = time.monotonic()
            # Drain sensors on every control tick, including while inference is busy.
            # ZMQ sockets stay on this thread; only immutable snapshots reach the worker.
            adapter.update(camera_subscriber.read(), state_subscriber.get_msg())
            check_keyboard_input()

            sensors_ready = adapter.sensors_ready()
            if rtc_scheduler is not None:
                if sensors_were_ready and not sensors_ready:
                    rtc_scheduler.reset()
                try:
                    request, action, info = result_queue.get_nowait()
                    if rtc_scheduler.finish(request, action, info):
                        print_green(
                            f"RTC chunk: latency={rtc_scheduler.last_latency:.3f}s, "
                            f"overlap={0 if request.prefix is None else next(iter(request.prefix.values())).shape[1]}, "
                            f"frozen={request.frozen_steps}, "
                            f"overrun={rtc_scheduler.last_overrun_steps} steps"
                        )
                    else:
                        print(f"RTC: {rtc_scheduler.last_status}")
                except queue.Empty:
                    pass
                if sensors_ready and rtc_scheduler.can_request():
                    inference_queue.put_nowait(rtc_scheduler.start_request())
                current = rtc_scheduler.current()
                cached_action_chunk, action_chunk_index = current if current else (None, 0)
            else:
                # Consume result first so last_inference_time is fresh before trigger check
                try:
                    processed_action, inference_start_time = result_queue.get_nowait()
                    inference_delay = time.monotonic() - inference_start_time
                    action_chunk_index = calculate_latency_compensated_index(
                        inference_delay, config.action_publish_rate, config.action_horizon
                    )
                    cached_action_chunk = processed_action
                    last_inference_time = time.monotonic()
                    print_green(
                        f'New action chunk (prompt: "{language_prompt_ref[0]}", latency: {inference_delay:.3f}s)'
                    )
                except queue.Empty:
                    pass

                worker_is_busy = inference_busy_event.is_set()
                should_start = should_trigger_new_inference(
                    cached_chunk_exists=(cached_action_chunk is not None),
                    inference_thread_running=worker_is_busy,
                    time_since_last_inference=(time.monotonic() - last_inference_time),
                    inference_interval=inference_interval,
                )

                if should_start:
                    try:
                        inference_queue.put_nowait(None)
                    except queue.Full:
                        pass

            sensors_were_ready = sensors_ready
            if not sensors_ready:
                _sleep_remaining(t_start, loop_period)
                continue

            if pause_loop:
                maintain_stationary_posture()
                _sleep_remaining(t_start, loop_period)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    maintain_stationary_posture()
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    current_idx = min(action_chunk_index, adapter.action_horizon - 1)
                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    if adapter.is_latent:
                        motion_token = processed_action["motion_token"][0, current_idx]
                        zmq_message = pack_latent_action_message(
                            motion_token,
                            frame_index,
                            left_hand_joints=processed_action["left_hand_joints"][0, current_idx],
                            right_hand_joints=processed_action["right_hand_joints"][0, current_idx],
                        )
                        last_sent_motion_token = motion_token.copy()
                    else:
                        zmq_message = adapter.stationary_message(processed_action, current_idx)
                        last_stationary_message = zmq_message
                    zmq_socket.send(zmq_message)
                    zmq_frame_counter += 1
                    if zmq_frame_counter % 50 == 0:
                        print_green(f"ZMQ: Sent {adapter.embodiment_tag} action - frame: {frame_index[0]}")

                if rtc_scheduler is not None:
                    rtc_scheduler.advance()
                else:
                    action_chunk_index += 1
                    if action_chunk_index >= adapter.action_horizon:
                        cached_action_chunk = None

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(context="VLA Inference Loop Missed", threshold=0.001)

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        camera_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
