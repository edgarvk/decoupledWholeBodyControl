"""Integration options tested without vendor services, model inference or motors."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from gear_sonic.utils.teleop.arms_only import ArmsOnlyTargets
from gear_sonic.utils.teleop.hand_controls import generate_finger_data
from gear_sonic.utils.teleop.mirroring import (
    mirror_body_rotations,
    mirror_controller_targets,
    mirror_flat_vr_targets,
    mirror_pose_message_outputs,
    mirror_vr_wrist_targets,
    route_hand_inputs,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import G1GripperInverseKinematicsSolver

ROOT = Path(__file__).parents[2]


def test_reflections_are_proper_rotations_and_involutions():
    rng = np.random.default_rng(42)
    positions = rng.normal(size=(2, 3))
    rotations = R.random(2, random_state=rng)
    saved = positions.copy()
    mirrored_p, mirrored_r = mirror_controller_targets(positions, rotations)
    restored_p, restored_r = mirror_controller_targets(mirrored_p, mirrored_r)
    np.testing.assert_allclose(restored_p, positions)
    np.testing.assert_allclose(restored_r.as_matrix(), rotations.as_matrix(), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(mirrored_r.as_matrix()), [1, 1])
    np.testing.assert_array_equal(positions, saved)
    vr = np.c_[rng.normal(size=(3, 3)), R.random(3, random_state=rng).as_quat(scalar_first=True)]
    mirrored = mirror_vr_wrist_targets(vr)
    np.testing.assert_array_equal(mirrored[2], vr[2])
    restored = mirror_vr_wrist_targets(mirrored)
    np.testing.assert_allclose(restored[:, :3], vr[:, :3])
    np.testing.assert_allclose(
        R.from_quat(restored[:, 3:], scalar_first=True).as_matrix(),
        R.from_quat(vr[:, 3:], scalar_first=True).as_matrix(),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "axis,delta,expected", [(2, -0.02, [0.02, 0, 0]), (0, 0.02, [0, 0.02, 0]), (1, 0.02, [0, 0, 0.02])]
)
def test_mirror_anchors_asymmetric_robot_and_routes_operator_right_to_robot_left(axis, delta, expected):
    pose = np.array([[0.3, 0.25, 0.1, 1, 0, 0, 0], [0.4, -0.3, 0.2, 1, 0, 0, 0], [0, 0, 0.4, 1, 0, 0, 0]])
    sample = {
        "controller_poses": np.array([[0.2, 1, -0.3, 0, 0, 0, 1], [-0.2, 1, -0.3, 0, 0, 0, 1.0]]),
        "headset_pose": np.array([0, 1.6, 0, 0, 0, 0, 1.0]),
        "timestamp_ns": 1,
        "timestamp_monotonic": 10.0,
    }
    targets = ArmsOnlyTargets(lambda q: pose, mirror=True)
    np.testing.assert_allclose(targets.calibrate(sample, np.zeros(29), now=10), pose)
    moved = copy.deepcopy(sample)
    moved["timestamp_ns"] = 2
    moved["controller_poses"][1, axis] += delta
    result = targets.update(moved, now=10)
    np.testing.assert_allclose(result[0, :3] - pose[0, :3], expected, atol=1e-12)
    np.testing.assert_allclose(result[1:], pose[1:], atol=1e-12)
    np.testing.assert_array_equal(sample["controller_poses"][1, :3], [-0.2, 1, -0.3])
    assert route_hand_inputs(0.1, 0.2, 0.3, 0.4, mirror=True) == (0.3, 0.4, 0.1, 0.2)


def test_body_mirror_retains_non_arm_rotations_and_respects_torso_yaw():
    rotations = list(R.random(24, random_state=42))
    mirrored = mirror_body_rotations(rotations)
    np.testing.assert_allclose(mirrored[0].as_matrix(), rotations[0].as_matrix())
    np.testing.assert_allclose(mirrored[4].as_matrix(), rotations[4].as_matrix())
    restored = mirror_body_rotations(mirrored)
    for a, b in zip(rotations, restored):
        np.testing.assert_allclose(a.as_matrix(), b.as_matrix(), atol=1e-12)
    yaw = R.from_euler("z", 1.1)
    turned = mirror_body_rotations([yaw * r for r in rotations])
    for a, b in zip(mirrored, turned):
        np.testing.assert_allclose((yaw * a).as_matrix(), b.as_matrix(), atol=1e-12)


def test_legacy_pose_and_planner_use_same_mirror_without_changing_recorded_input():
    pose = np.c_[np.arange(9).reshape(3, 3), R.random(3, random_state=42).as_quat(scalar_first=True)]
    data = {
        "vr_position": pose[:, :3].ravel(),
        "vr_orientation": pose[:, 3:].ravel(),
        "left_trigger": 0.1,
        "right_trigger": 0.8,
        "left_grip": 0.2,
        "right_grip": 0.9,
    }
    saved = copy.deepcopy(data)
    output = mirror_pose_message_outputs(data)
    p, q = mirror_flat_vr_targets(data["vr_position"], data["vr_orientation"])
    np.testing.assert_allclose(output["vr_position"], p)
    np.testing.assert_allclose(output["vr_orientation"], q)
    assert output["left_trigger"] == 0.8 and output["right_grip"] == 0.2
    np.testing.assert_array_equal(saved["vr_position"], data["vr_position"])


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("legacy", [0, 0.7, 0.7, -1, -1.5, -1, -1.5]),
        ("rtc", [0, 0.7, 0.7, -1, -1.5, -1, -1.5]),
        ("enhanced", [0, 0.8, 0.8, -1.5, -1.8, -1.5, -1.8]),
    ],
)
def test_saved_gripper_profiles_and_right_joint_signs(profile, expected):
    left = G1GripperInverseKinematicsSolver("left", profile)
    right = G1GripperInverseKinematicsSolver("right", profile)
    np.testing.assert_allclose(left._get_middle_close_q_desired(), expected)
    np.testing.assert_allclose(
        right._get_middle_close_q_desired(), np.asarray(expected) * [1, -1, -1, -1, -1, -1, -1]
    )
    opened = left({"position": generate_finger_data("left", 0, 0, profile)})
    np.testing.assert_array_equal(opened, np.zeros(7))
    if profile == "enhanced":
        for trigger, grip, name in [(1, 0, "index"), (0, 1, "ring"), (1, 1, "middle")]:
            actual = left({"position": generate_finger_data("left", trigger, grip, profile)})
            np.testing.assert_allclose(actual, getattr(left, f"_get_{name}_close_q_desired")())
    else:
        np.testing.assert_allclose(left({"position": generate_finger_data("left", 1, 0, profile)}), expected)


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_dds_pair_selected_after_ros_from_any_directory_and_repeated_source(tmp_path, architecture):
    deploy = tmp_path / "deployment with spaces"
    scripts = deploy / "scripts"
    scripts.mkdir(parents=True)
    helper = scripts / "configure_unitree_dds.sh"
    shutil.copyfile(ROOT / "gear_sonic_deploy/scripts/configure_unitree_dds.sh", helper)
    libraries = deploy / "thirdparty/unitree_sdk2/thirdparty/lib" / architecture
    libraries.mkdir(parents=True)
    for name in ("libddsc.so.0", "libddscxx.so.0"):
        (libraries / name).write_text("test library placeholder")
    tools = tmp_path / "tools"
    tools.mkdir()
    uname = tools / "uname"
    uname.write_text(f'#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo {architecture};; esac\n')
    uname.chmod(0o755)
    env = dict(
        os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"], LD_LIBRARY_PATH="/opt/ros/jazzy/lib:/old"
    )
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            'source "$1"; export LD_LIBRARY_PATH="/new/ros:$LD_LIBRARY_PATH"; '
            'source "$1"; printf "PATH_RESULT=%s\\n" "$LD_LIBRARY_PATH"',
            "test",
            str(helper),
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert f"PATH_RESULT={libraries}:/new/ros:/opt/ros/jazzy/lib:/old" in result.stdout
    (libraries / "libddscxx.so.0").unlink()
    failed = subprocess.run(
        ["bash", "-c", 'source "$1"', "test", str(helper)], env=env, capture_output=True, text=True
    )
    assert failed.returncode != 0 and "Missing bundled" in failed.stderr
    (libraries / "libddscxx.so.0").write_text("version https://git-lfs.github.com/spec/v1\n")
    failed = subprocess.run(
        ["bash", "-c", 'source "$1"', "test", str(helper)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0 and "LFS pointer" in failed.stderr


def test_all_entry_points_expose_shared_options_without_starting_sdk():
    for module in ("pico_manager_thread_server", "pico_manager_thread_server_mirror"):
        result = subprocess.run(
            [sys.executable, "-m", f"gear_sonic.scripts.{module}", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "--mirror" in result.stdout and "--gripper-profile" in result.stdout


def test_offline_mirror_preview_returns_to_neutral_and_preserves_raw_recording(tmp_path):
    from gear_sonic.scripts.pico_demo import preview
    from gear_sonic.utils.teleop.pico_demo import DemoWriter

    path = tmp_path / "input.jsonl"
    output = tmp_path / "preview.jsonl"
    seed = tmp_path / "initial-joints.json"
    q = np.zeros(29)
    q[[15, 16, 18, 22, 23, 25]] = [0.15, 0.3, 0.65, 0.15, -0.3, 0.65]
    seed.write_text(json.dumps(q.tolist()))
    writer = DemoWriter(path, {})
    for i in range(4):
        pressed = i in (0, 3)
        packet = {
            "timeStampNs": i + 1,
            "receive_age_s": 0,
            "Head": {"pose": "0,1.6,0,0,0,0,1"},
            "Controller": {
                side: {
                    "pose": f"{0.22 if i == 1 and side == 'right' else 0.2},1,-.3,0,0,0,1",
                    "primaryButton": pressed,
                    "secondaryButton": pressed,
                }
                for side in ("left", "right")
            },
        }
        writer.sample(i * 0.02, packet, valid=True)
    writer.close(0.08)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    preview(
        SimpleNamespace(
            input=str(path),
            output=str(output),
            initial_joints=str(seed),
            start_mode="buttons",
            mapping="direct",
            mirror=True,
            gripper_profile="enhanced",
            arms_max_gravity_nm=8,
            left_grip_offset=[0, 0, 0],
            right_grip_offset=[0, 0, 0],
        )
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["mirror"] and rows[0]["gripper_profile"] == "enhanced"
    start, moved, returned = [np.asarray(row["outgoing_pose"]) for row in rows[1:]]
    np.testing.assert_allclose(moved[0, :3] - start[0, :3], [0, 0.02, 0], atol=1e-12)
    np.testing.assert_allclose(moved[1:], start[1:], atol=1e-12)
    np.testing.assert_allclose(returned, start, atol=1e-12)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
