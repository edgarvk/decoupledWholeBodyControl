"""Offline regression tests: real G1 FK and decoded controller wire messages."""

import json
import time
from types import SimpleNamespace
import unittest

import numpy as np

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.inference.gr00t_adapter import JOINT_DIMS, Gr00tAdapter
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE


def configs(latent=False, deltas=None):
    def config(keys, indices):
        return SimpleNamespace(modality_keys=keys, delta_indices=indices)

    states = list(JOINT_DIMS) + (
        ["left_leg", "right_leg", "projected_gravity"] if latent else ["left_wrist_eef_9d", "right_wrist_eef_9d"]
    )
    return {
        "video": config(["ego_view"], [0] if latent else [-20, 0] if deltas is None else deltas),
        "state": config(states, [0]),
        "language": config(["annotation.human.task_description"], [0]),
        "action": config(
            ["motion_token", "left_hand_joints", "right_hand_joints"]
            if latent
            else list(JOINT_DIMS) + ["navigate_command", "base_height_command"],
            list(range(4)),
        ),
    }


def state():
    q = np.zeros(29, dtype=np.float32)
    q[15:22] = [0.23, 0.12, -0.13, 0.6, 0.16, -0.12, 0.18]
    return {
        "body_q": q,
        "left_hand_q": np.arange(7, dtype=np.float32) / 10,
        "right_hand_q": np.zeros(7, np.float32),
        "base_quat": np.array([1, 0, 0, 0], np.float32),
    }


def camera(index):
    return {
        "images": {"ego_view": np.full((8, 10, 3), index, np.uint8)},
        "timestamps": {"ego_view": index},
    }


def decode_message(message):
    topic = b"planner"
    assert message.startswith(topic)
    header = json.loads(message[len(topic) : len(topic) + HEADER_SIZE].rstrip(b"\0"))
    offset = len(topic) + HEADER_SIZE
    fields = {}
    for field in header["fields"]:
        dtype = {"i32": "<i4", "f32": "<f4"}[field["dtype"]]
        count = int(np.prod(field["shape"]))
        fields[field["name"]] = np.frombuffer(message, dtype=dtype, count=count, offset=offset).reshape(
            field["shape"]
        )
        offset += count * np.dtype(dtype).itemsize
    assert offset == len(message)
    return fields


class TestGr00tAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    def ready_adapter(self, latent=False):
        adapter = Gr00tAdapter(self.robot, configs(latent))
        start = time.monotonic()
        for i in range(13):
            adapter.update(camera(i), state(), now=start + i / 30)
        return adapter, start + 0.4

    def test_history_uses_camera_time_instead_of_inference_count(self):
        adapter = Gr00tAdapter(self.robot, configs())
        start = time.monotonic()
        adapter.update(camera(0), state(), now=start)
        self.assertIsNone(adapter.prepare_observation("raise your arms", now=start))
        for i in range(1, 13):
            adapter.update(camera(i), state(), now=start + i / 30)
        # One inference after thirteen camera captures, not thirteen inferences.
        obs = adapter.prepare_observation("raise your arms", now=start + 0.4)
        self.assertEqual(obs["video"]["ego_view"].shape, (1, 2, 8, 10, 3))
        np.testing.assert_array_equal(obs["video"]["ego_view"][0, :, 0, 0, 0], [0, 12])

    def test_cached_camera_message_does_not_make_history_or_refresh_freshness(self):
        adapter = Gr00tAdapter(self.robot, configs())
        start = time.monotonic()
        for i in range(25):
            adapter.update(camera(0), state(), now=start + i / 50)
        self.assertIsNone(adapter.prepare_observation("demo", now=start + 0.48))
        self.assertEqual(len(adapter._history["ego_view"]), 1)
        self.assertFalse(adapter.sensors_ready(now=start + 2))

    def test_wrist_fk_matches_gr00t_rotation_rows(self):
        adapter, now = self.ready_adapter()
        obs = adapter.prepare_observation("raise your arms", now=now)
        self.robot.cache_forward_kinematics(obs["q"][0, 0], auto_clip=False)
        root = self.robot.frame_placement("pelvis")
        for side in ("left", "right"):
            wrist = root.inverse() * self.robot.frame_placement(f"{side}_wrist_yaw_link")
            value = obs["state"][f"{side}_wrist_eef_9d"]
            self.assertEqual(value.shape, (1, 1, 9))
            self.assertEqual(value.dtype, np.float32)
            np.testing.assert_allclose(value[0, 0, :3], wrist.translation, atol=1e-7)
            np.testing.assert_allclose(value[0, 0, 3:], wrist.rotation[:2].ravel(), atol=1e-7)
        wrist = self.robot.frame_placement("left_wrist_yaw_link")
        self.assertFalse(np.allclose(wrist.rotation[:2].ravel(), wrist.rotation[:, :2].T.ravel()))

    def test_observation_does_not_mutate_robot_telemetry(self):
        adapter = Gr00tAdapter(self.robot, configs(True))
        measured = state()
        original = measured["left_hand_q"].copy()
        adapter.update(camera(0), measured)
        obs = adapter.prepare_observation("demo")
        np.testing.assert_array_equal(measured["left_hand_q"], original)
        actual = self.robot.get_hand_actuated_joints(obs["q"][0, 0], "left")
        np.testing.assert_array_equal(actual[5:7], original[3:5])

    def test_sonic_schema_keeps_one_frame_and_gravity(self):
        adapter, now = self.ready_adapter(True)
        obs = adapter.prepare_observation("demo", now=now)
        self.assertTrue(adapter.is_latent)
        self.assertEqual(obs["video"]["ego_view"].shape, (1, 1, 8, 10, 3))
        np.testing.assert_allclose(obs["state"]["projected_gravity"], [[[0, 0, -1]]])

    def test_rejects_mismatched_embodiment_and_nonfinite_actions(self):
        with self.assertRaisesRegex(ValueError, "server schema"):
            Gr00tAdapter(self.robot, configs(), "UNITREE_G1_SONIC")
        adapter, _ = self.ready_adapter()
        action = {key: np.zeros((1, 4, dim), np.float32) for key, dim in JOINT_DIMS.items()}
        action["left_arm"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid action left_arm"):
            adapter.process_action(action)

    def test_stationary_packet_maps_every_upper_body_joint_and_disables_navigation(self):
        adapter, _ = self.ready_adapter()
        # Different values at every motor catch left/right interleaving mistakes.
        measured = np.arange(29) / 100
        measured[16], measured[23] = 0.3, -0.3  # Inside shoulder-roll joint limits.
        q = self.robot.get_configuration_from_actuated_joints(measured)
        action = {
            key: np.tile(q[self.robot.get_joint_group_indices(key)], (1, 4, 1)).astype(np.float32)
            for key in JOINT_DIMS
        }
        action["navigate_command"] = np.full((1, 4, 3), 5.0)
        action["base_height_command"] = np.full((1, 4, 1), 0.3)
        parsed = adapter.process_action({f"action.{key}": value for key, value in action.items()})
        fields = decode_message(adapter.stationary_message(parsed, 2))
        self.assertEqual(fields["mode"].item(), 0)
        np.testing.assert_array_equal(fields["movement"], [0, 0, 0])
        self.assertEqual(fields["speed"].item(), 0)
        self.assertEqual(fields["height"].item(), -1)
        # Expected SDK indices are independently copied from the C++ wire contract.
        expected = measured[[12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]]
        np.testing.assert_allclose(fields["upper_body_position"], expected, atol=1e-7)
        self.assertEqual(fields["left_hand_joints"].shape, (7,))
        self.assertEqual(fields["right_hand_joints"].shape, (7,))

    def test_joint_targets_are_clipped_to_robot_limits(self):
        adapter, _ = self.ready_adapter()
        action = {key: np.full((1, 4, dim), 100, np.float32) for key, dim in JOINT_DIMS.items()}
        fields = decode_message(adapter.stationary_message(adapter.process_action(action)))
        self.assertTrue(np.isfinite(fields["upper_body_position"]).all())
        self.assertLess(np.max(fields["upper_body_position"]), 4)

    def test_rtc_retains_all_action_groups_and_the_actual_bounded_joint_targets(self):
        adapter, _ = self.ready_adapter()
        action = {key: np.full((1, 4, dim), 100, np.float32) for key, dim in JOINT_DIMS.items()}
        action["navigate_command"] = np.zeros((1, 4, 3), np.float32)
        action["base_height_command"] = np.zeros((1, 4, 1), np.float32)
        result = adapter.process_action(action, retain_all=True)
        self.assertEqual(set(result), set(adapter.action_keys))
        self.assertLess(result["left_arm"].max(), 4)
        # The transmitted finger values are the same bounded targets held in the RTC prefix.
        fields = decode_message(adapter.stationary_message(result))
        q = self.robot.get_configuration_from_actuated_joints(np.zeros(29))
        q[self.robot.get_joint_group_indices("left_hand")] = result["left_hand"][0, 0]
        expected = self.robot.get_hand_actuated_joints(q, "left")
        np.testing.assert_array_equal(fields["left_hand_joints"], expected)

    def test_stale_robot_state_prevents_inference(self):
        adapter, now = self.ready_adapter()
        self.assertIsNone(adapter.prepare_observation("demo", now=now + 2))

    def test_zero_quaternion_is_rejected(self):
        adapter = Gr00tAdapter(self.robot, configs())
        measured = state()
        measured["base_quat"][:] = 0
        with self.assertRaisesRegex(ValueError, "zero norm"):
            adapter.update(camera(0), measured)


if __name__ == "__main__":
    unittest.main()
