"""Adapt GR00T REAL_G1 or SONIC observations and actions to the SONIC controller.

REAL_G1 uses measured wrist FK and temporal RGB history. Its joint predictions
are absolute after GR00T's action decoding. Stationary deployment sends the
upper body to the planner in IDLE mode; navigation and height are not applied.
"""

from collections import deque
import threading
import time

import numpy as np

from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.inference.vla_utils import prepare_observation_for_eval
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message

# policy_parameters.hpp: upper_body_joint_isaaclab_order_in_mujoco_index.
# The planner's 17 targets interleave the two arms; SDK body_q groups them.
PLANNER_UPPER_BODY_INDICES = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
JOINT_DIMS = {"left_arm": 7, "right_arm": 7, "left_hand": 7, "right_hand": 7, "waist": 3}


def _field(config, name):
    return config[name] if isinstance(config, dict) else getattr(config, name)


class Gr00tAdapter:
    """Keep sensor history on the control thread, build observations on the worker.

    ``observation_rate`` defines the duration of a checkpoint delta index, not
    the frequency of inference requests. E.g. [-20, 0] at 50 Hz spans 0.4 s.
    Camera frames are selected by acquisition/arrival time, so camera and policy
    frequencies need not match. Warm-up waits for the required past frame.
    """

    def __init__(
        self,
        robot_model,
        modality_configs,
        embodiment_tag="auto",
        observation_rate=50.0,
        sensor_timeout=1.0,
    ):
        if observation_rate <= 0 or sensor_timeout <= 0:
            raise ValueError("Observation rate and sensor timeout must be positive")
        self.robot_model = robot_model
        self.configs = modality_configs
        self.observation_rate = observation_rate
        self.sensor_timeout = sensor_timeout
        self.action_keys = list(_field(self.configs["action"], "modality_keys"))
        self.is_latent = "motion_token" in self.action_keys
        self.embodiment_tag = "unitree_g1_sonic" if self.is_latent else "real_g1_relative_eef_relative_joints"
        aliases = {
            "real_g1": "real_g1_relative_eef_relative_joints",
            "unitree_g1_sonic": "unitree_g1_sonic",
        }
        requested = aliases.get(embodiment_tag.lower(), embodiment_tag.lower())
        if requested not in ("auto", self.embodiment_tag):
            raise ValueError(f"Client requested {embodiment_tag}, but server schema is {self.embodiment_tag}")
        required = {"motion_token", "left_hand_joints", "right_hand_joints"} if self.is_latent else set(JOINT_DIMS)
        if not required.issubset(self.action_keys):
            raise ValueError(f"Unsupported server action schema: {self.action_keys}")
        self.video_keys = list(_field(self.configs["video"], "modality_keys"))
        self.video_deltas = list(_field(self.configs["video"], "delta_indices"))
        if not self.video_keys or not self.video_deltas or max(self.video_deltas) != 0:
            raise ValueError("Video delta indices must be nonpositive and include the current frame")
        if any(not isinstance(i, int) for i in self.video_deltas):
            raise ValueError("Video delta indices must be integers")
        if list(_field(self.configs["state"], "delta_indices")) != [0]:
            raise ValueError("Only current-state observations (delta_indices=[0]) are supported")
        if list(_field(self.configs["language"], "delta_indices")) != [0]:
            raise ValueError("Only current language prompts are supported")
        supported_states = set(JOINT_DIMS) | {
            "left_leg",
            "right_leg",
            "projected_gravity",
            "left_wrist_eef_9d",
            "right_wrist_eef_9d",
        }
        self.state_keys = list(_field(self.configs["state"], "modality_keys"))
        if set(self.state_keys) - supported_states:
            raise ValueError(f"Unsupported server state schema: {self.state_keys}")
        self.action_horizon = len(_field(self.configs["action"], "delta_indices"))
        if (
            list(_field(self.configs["action"], "delta_indices")) != list(range(self.action_horizon))
            or self.action_horizon == 0
        ):
            raise ValueError("Action delta indices must be consecutive and start at zero")
        self._lock = threading.Lock()
        self._history = {key: deque() for key in self.video_keys}
        self._camera_stamps = {}
        self._state = None
        self._state_time = 0.0
        self._facing = None
        self._reported_observation = False

    def update(self, camera_msg, state_msg, now=None):
        """Capture fresh messages independently of slow model inference."""
        now = time.monotonic() if now is None else now
        span = -min(self.video_deltas) / self.observation_rate
        with self._lock:
            if state_msg is not None:
                self._state = {
                    key: np.asarray(state_msg[key], dtype=np.float32).copy()
                    for key in ("body_q", "left_hand_q", "right_hand_q", "base_quat")
                }
                for key, size in (
                    ("body_q", 29),
                    ("left_hand_q", 7),
                    ("right_hand_q", 7),
                    ("base_quat", 4),
                ):
                    value = self._state[key]
                    if value.shape != (size,) or not np.isfinite(value).all():
                        raise ValueError(f"Invalid robot state {key}: expected finite ({size},) values")
                if np.linalg.norm(self._state["base_quat"]) < 1e-6:
                    raise ValueError("Invalid robot state base_quat: quaternion has zero norm")
                self._state_time = now
            if camera_msg is None:
                return
            for key in self.video_keys:
                source = "right_wrist" if key == "wrist_view" else key
                if source not in camera_msg.get("images", {}):
                    continue
                stamp = camera_msg.get("timestamps", {}).get(source)
                if stamp is not None and self._camera_stamps.get(key) == stamp:
                    continue  # read() may return the same cached camera message.
                frame = np.asarray(camera_msg["images"][source])
                if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                    raise ValueError(
                        f"Camera {key} must provide uint8 RGB (H, W, 3), got {frame.shape}/{frame.dtype}"
                    )
                history = self._history[key]
                if history and (
                    now - history[-1][0] > self.sensor_timeout
                    or (
                        stamp is not None
                        and self._camera_stamps.get(key) is not None
                        and stamp < self._camera_stamps[key]
                    )
                ):
                    history.clear()
                history.append((now, frame.copy()))
                self._camera_stamps[key] = stamp
                # Retain the frame just before the oldest requested timestamp.
                while len(history) > 2 and history[1][0] <= now - span:
                    history.popleft()

    def _ready(self, now):
        span = -min(self.video_deltas) / self.observation_rate
        return (
            self._state is not None
            and now - self._state_time <= self.sensor_timeout
            and all(
                history
                and now - history[-1][0] <= self.sensor_timeout
                and history[-1][0] - history[0][0] >= span - 1e-6
                for history in self._history.values()
            )
        )

    def sensors_ready(self, now=None):
        with self._lock:
            return self._ready(time.monotonic() if now is None else now)

    def _configuration(self, state):
        left_hand = state["left_hand_q"].copy()
        left_hand[5:7] = left_hand[3:5]  # Hardware finger coupling; don't mutate telemetry.
        return self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=state["body_q"],
            left_hand_actuated_joint_values=left_hand,
            right_hand_actuated_joint_values=state["right_hand_q"],
        ).astype(np.float32)

    def prepare_observation(self, prompt, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            if not self._ready(now):
                return None
            state = {key: value.copy() for key, value in self._state.items()}
            video = {}
            for key, history in self._history.items():
                current_time = history[-1][0]
                frames = [
                    min(
                        history,
                        key=lambda item: abs(item[0] - (current_time + delta / self.observation_rate)),
                    )[1]
                    for delta in self.video_deltas
                ]
                video[key] = np.stack(frames)[None]
        q = self._configuration(state)
        observation = {
            "video": video,
            "state": {},
            "language": {key: [[prompt]] for key in _field(self.configs["language"], "modality_keys")},
            "q": q[None, None],
        }
        prepare_observation_for_eval(self.robot_model, observation)
        if "projected_gravity" in self.state_keys:
            observation["state"]["projected_gravity"] = compute_projected_gravity(state["base_quat"])[None, None]
        if any(key.endswith("eef_9d") for key in self.state_keys):
            self.robot_model.cache_forward_kinematics(q, auto_clip=False)
            root = self.robot_model.frame_placement(self.robot_model.supplemental_info.root_frame_name)
            for side in ("left", "right"):
                frame = self.robot_model.supplemental_info.hand_frame_names[side]
                pose = root.inverse() * self.robot_model.frame_placement(frame)
                # GR00T Pose uses the FIRST TWO ROWS, not columns, for rot6d.
                observation["state"][f"{side}_wrist_eef_9d"] = np.concatenate(
                    (pose.translation, pose.rotation[:2].reshape(-1))
                ).astype(np.float32)[None, None]
        observation["state"] = {key: observation["state"][key] for key in self.state_keys}
        if not self._reported_observation:
            print("GR00T video shapes:", {key: value.shape for key, value in video.items()})
            print(
                "GR00T state shapes:",
                {key: value.shape for key, value in observation["state"].items()},
            )
            self._reported_observation = True
        return observation

    def process_action(self, action, retain_all=False):
        """Validate server-decoded absolute targets before publishing them."""
        action = {key.removeprefix("action."): value for key, value in action.items()}
        dimensions = (
            {"motion_token": 64, "left_hand_joints": 7, "right_hand_joints": 7} if self.is_latent else JOINT_DIMS
        )
        if retain_all and not self.is_latent:
            dimensions = dict(JOINT_DIMS)
            extra_dims = {
                "left_wrist_eef_9d": 9,
                "right_wrist_eef_9d": 9,
                "navigate_command": 3,
                "base_height_command": 1,
            }
            for key in self.action_keys:
                if key not in dimensions:
                    if key not in extra_dims:
                        raise ValueError(f"Unknown RTC action dimension for {key}")
                    dimensions[key] = extra_dims[key]
        result = {}
        for key, dim in dimensions.items():
            if key not in action:
                raise ValueError(f"Server omitted action field {key}")
            value = np.asarray(action[key], dtype=np.float32)
            if value.shape != (1, self.action_horizon, dim) or not np.isfinite(value).all():
                raise ValueError(
                    f"Invalid action {key}: expected finite (1, {self.action_horizon}, {dim}), got {value.shape}"
                )
            result[key] = value.copy()
            if not self.is_latent:
                indices = self.robot_model.get_joint_group_indices(key) if key in JOINT_DIMS else None
                if indices is not None:
                    # Prefixes must describe the same bounded physical targets
                    # that stationary_message will actually send to the planner.
                    result[key] = np.clip(
                        result[key],
                        self.robot_model.lower_joint_limits[indices] + 1e-6,
                        self.robot_model.upper_joint_limits[indices] - 1e-6,
                    ).astype(np.float32)
        if self.is_latent and np.abs(result["motion_token"]).max() > 1.25:
            raise ValueError("motion_token exceeds the SONIC action bound 1.25")
        return result

    def stationary_message(self, action=None, index=0, left_hand=None, right_hand=None):
        """Publish joints in the planner's order with IDLE/no locomotion.

        An omitted action holds the current measured posture (the 'i' key).
        Predicted EEF, navigation and base height are deliberately not applied:
        the absolute arm/waist targets define this stationary joint controller.
        """
        with self._lock:
            if self._state is None or time.monotonic() - self._state_time > self.sensor_timeout:
                raise ValueError("Waiting for fresh robot state before publishing a posture")
            state = {key: value.copy() for key, value in self._state.items()}
            if self._facing is None:
                w, x, y, z = state["base_quat"] / np.linalg.norm(state["base_quat"])
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                self._facing = [np.cos(yaw), np.sin(yaw), 0.0]
            facing = self._facing.copy()
        q = self._configuration(state)
        if action is not None:
            for key in JOINT_DIMS:
                q[self.robot_model.get_joint_group_indices(key)] = action[key][0, index]
        if left_hand is not None:
            q[self.robot_model.get_hand_actuated_joint_indices("left")] = left_hand
        if right_hand is not None:
            q[self.robot_model.get_hand_actuated_joint_indices("right")] = right_hand
        q = self.robot_model.clip_configuration(q)
        body = self.robot_model.get_body_actuated_joints(q)
        return build_planner_message(
            mode=0,
            movement=[0.0, 0.0, 0.0],
            facing=facing,
            speed=0.0,
            height=-1.0,
            upper_body_position=body[PLANNER_UPPER_BODY_INDICES],
            upper_body_velocity=np.zeros(17),
            left_hand_position=self.robot_model.get_hand_actuated_joints(q, "left"),
            right_hand_position=self.robot_model.get_hand_actuated_joints(q, "right"),
        )
