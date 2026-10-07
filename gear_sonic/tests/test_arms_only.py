"""Synthetic tracking/FK/transport tests: never connect to a physical robot."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from gear_sonic.utils.teleop.arms_only import (
    XRT_TO_ROBOT,
    ArmsOnlySession,
    ArmsOnlyTargets,
    MeasuredRobotFeedback,
    make_robot_fk,
)
from gear_sonic.utils.teleop.input_readers import PicoControllerReader
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE, build_planner_message


def sample(stamp=1, now=10.0):
    return {
        "controller_poses": np.array([[0.2, 1.0, -0.3, 0, 0, 0, 1], [-0.2, 1.0, -0.3, 0, 0, 0, 1.0]]),
        "headset_pose": np.array([0, 1.6, 0, 0, 0, 0, 1.0]),
        "timestamp_ns": stamp,
        "timestamp_monotonic": now,
    }


def robot_pose():
    return np.array([[0.3, 0.25, 0.1, 1, 0, 0, 0], [0.3, -0.25, 0.1, 1, 0, 0, 0], [0, 0, 0.4, 1, 0, 0, 0.0]])


def mapper():
    return ArmsOnlyTargets(lambda q: robot_pose())


class FakeSDK:
    def __init__(self):
        self.packet = {
            "timeStampNs": 1,
            "receive_age_s": 0.0,
            "Controller": {side: {"pose": "0.2,1,-0.3,0,0,0,1"} for side in ("left", "right")},
            "Head": {"pose": "0,1.6,0,0,0,0,1"},
        }

    def get_controller_snapshot(self):
        return json.dumps(self.packet)

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected SDK access (including body/ankles): {name}")


def test_reader_never_reads_body_and_head_optional():
    sdk = FakeSDK()
    reader = PicoControllerReader(sdk, clock=lambda: 10.0)
    assert reader.get_latest()["controller_poses"].shape == (2, 7)
    del sdk.packet["Head"]
    assert reader.get_latest()["headset_pose"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["Controller"].pop("left"),
        lambda p: p["Controller"]["left"].update(pose="0,0,0,0,0,0,0"),
        lambda p: p["Controller"]["right"].update(pose="nan,0,0,0,0,0,1"),
        lambda p: p.update(receive_age_s=0.251),
        lambda p: p.update(appState={"focus": False}),
        lambda p: p["Controller"]["left"].update(isTracked=False),
        lambda p: p["Controller"]["left"].update(trackingState=1),
        lambda p: p["Controller"].update(left=None),
    ],
)
def test_reader_rejects_invalid_packet(mutation):
    sdk = FakeSDK()
    reader = PicoControllerReader(sdk, clock=lambda: 10.0)
    assert reader.get_latest() is not None
    mutation(sdk.packet)
    assert reader.get_latest() is None


def test_duplicate_source_timestamp_stales_even_with_new_receives():
    sdk = FakeSDK()
    clock = [10.0]
    reader = PicoControllerReader(sdk, clock=lambda: clock[0])
    assert reader.get_latest() is not None
    clock[0] += 0.3
    assert reader.get_latest() is None
    sdk.packet["timeStampNs"] += 1
    assert reader.get_latest() is not None


def test_start_and_recalibrate_continuous_with_measured_fk():
    calls = []

    def fk(q):
        calls.append(q.copy())
        pose = robot_pose()
        pose[0, 0] += q[15]
        return pose

    targets = ArmsOnlyTargets(fk)
    q = np.arange(29.0) / 100
    initial = targets.calibrate(sample(), q, now=10.0)
    np.testing.assert_allclose(initial, fk(q))
    moved = sample(2)
    moved["controller_poses"][0, 2] -= 0.03
    previous = targets.update(moved, now=10.0)
    q[15] += 0.01  # actual lags target: reanchor must not snap to actual
    moved["controller_poses"][0, 2] -= 0.12
    np.testing.assert_allclose(targets.calibrate(moved, q, now=10.0), previous)
    np.testing.assert_allclose(targets.update(moved, now=10.0), previous)
    np.testing.assert_allclose(calls[-1], q)
    assert targets.calibration_residual_m[0] == pytest.approx(0.02)


@pytest.mark.parametrize("axis, expected", [(0, [0, -0.02, 0]), (1, [0, 0, 0.02]), (2, [-0.02, 0, 0])])
def test_translation_axes(axis, expected):
    targets = mapper()
    targets.calibrate(sample(), np.zeros(29), now=10.0)
    moved = sample(2)
    moved["controller_poses"][0, axis] += 0.02
    result = targets.update(moved, now=10.0)
    np.testing.assert_allclose(result[0, :3] - robot_pose()[0, :3], expected, atol=1e-12)
    np.testing.assert_allclose(result[1:], robot_pose()[1:])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_spatial_rotation_with_nonidentity_grip_and_wrist(axis):
    reference = robot_pose()
    reference[:2, 3:] = R.from_euler("xyz", [20, 15, 10], degrees=True).as_quat(scalar_first=True)
    targets = ArmsOnlyTargets(lambda q: reference)
    start = sample()
    initial_rotation = R.from_euler("xyz", [30, -15, 5], degrees=True)
    start["controller_poses"][:, 3:] = initial_rotation.as_quat()
    targets.calibrate(start, np.zeros(29), now=10.0)
    moved = copy.deepcopy(start)
    moved["timestamp_ns"] = 2
    vector = np.eye(3)[axis] * 0.1
    delta = R.from_rotvec(vector)
    moved["controller_poses"][:, 3:] = (delta * initial_rotation).as_quat()
    result = targets.update(moved, now=10.0)
    expected = (
        XRT_TO_ROBOT
        @ delta.as_matrix()
        @ XRT_TO_ROBOT.T
        @ R.from_quat(reference[0, 3:], scalar_first=True).as_matrix()
    )
    np.testing.assert_allclose(R.from_quat(result[0, 3:], scalar_first=True).as_matrix(), expected, atol=1e-12)
    np.testing.assert_allclose(result[:, :3], reference[:, :3])


def test_head_motion_and_loss_do_not_move_targets():
    targets = mapper()
    targets.calibrate(sample(), np.zeros(29), now=10.0)
    moved = sample(2)
    moved["headset_pose"] = np.r_[[5, 6, 7], R.from_euler("xyz", [1, 2, 3]).as_quat()]
    np.testing.assert_allclose(targets.update(moved, now=10.0), robot_pose(), atol=1e-12)
    moved["headset_pose"] = None
    moved["timestamp_ns"] = 3
    np.testing.assert_allclose(targets.update(moved, now=10.0), robot_pose(), atol=1e-12)


def test_initial_head_yaw_aligns_fixed_operator_forward():
    targets = mapper()
    start = sample()
    start["headset_pose"][3:] = R.from_euler("y", np.pi / 2).as_quat()
    targets.calibrate(start, np.zeros(29), now=10.0)
    moved = copy.deepcopy(start)
    moved["timestamp_ns"] = 2
    moved["controller_poses"][0, 0] -= 0.02
    np.testing.assert_allclose(
        targets.update(moved, now=10.0)[0, :3] - robot_pose()[0, :3], [0.02, 0, 0], atol=1e-12
    )


def test_grip_lever_arm_rotates_about_controller_origin():
    targets = ArmsOnlyTargets(lambda q: robot_pose(), [[0.05, 0, 0], [0, 0, 0]])
    targets.calibrate(sample(), np.zeros(29), now=10.0)
    moved = sample(2)
    rotation = R.from_euler("y", 0.2)
    moved["controller_poses"][0, 3:] = rotation.as_quat()
    result = targets.update(moved, now=10.0)
    expected = XRT_TO_ROBOT @ (rotation.apply([0.05, 0, 0]) - [0.05, 0, 0])
    np.testing.assert_allclose(result[0, :3] - robot_pose()[0, :3], expected, atol=1e-12)


@pytest.mark.parametrize("q", [None, np.zeros(28), np.full(29, np.nan)])
def test_no_zero_fallback(q):
    targets = mapper()
    with pytest.raises(ValueError, match="measured"):
        targets.calibrate(sample(), q, now=10.0)
    assert targets.target is None


def test_stale_and_discontinuity_latch_until_reanchor():
    targets = mapper()
    targets.calibrate(sample(), np.zeros(29), now=10.0)
    np.testing.assert_allclose(targets.update(sample(), now=10.3), robot_pose())
    assert targets.held
    moved = sample(2, now=10.3)
    moved["controller_poses"][0, 0] += 0.3
    np.testing.assert_allclose(targets.update(moved, now=10.3), robot_pose())
    targets.calibrate(moved, np.zeros(29), now=10.3)
    jumped = copy.deepcopy(moved)
    jumped["timestamp_ns"] = 3
    jumped["controller_poses"][0, 0] += 0.11
    np.testing.assert_allclose(targets.update(jumped, now=10.3), robot_pose())
    assert targets.held


def test_session_start_failure_modes_hold_stop_without_tracking():
    session = ArmsOnlySession(mapper())
    all_buttons = [True] * 4
    no_buttons = [False] * 4
    assert session.step(sample(), None, all_buttons, now=10.0) == (None, None)
    assert not session.active
    session.step(sample(), None, no_buttons, now=10.0)
    action, target = session.step(sample(), np.zeros(29), all_buttons, now=10.0)
    assert action == "start"
    np.testing.assert_allclose(target, robot_pose())
    # All old mode combos stay in VR_3PT.
    for combo in ([1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]):
        assert session.step(sample(), np.zeros(29), combo, now=10.0)[0] is None
        assert session.active
    session.step(None, None, no_buttons, menu=True, now=10.3)
    assert session.targets.held
    assert session.step(None, None, all_buttons, now=10.3) == ("stop", None)


def test_real_g1_fk_at_measured_nonzero_joints():
    pytest.importorskip("pinocchio")
    fk = make_robot_fk()
    q = np.zeros(29)
    base = fk(q)
    q[15] = 0.25  # MuJoCo order: left shoulder pitch
    moved = fk(q)
    assert not np.allclose(base[0], moved[0])
    np.testing.assert_allclose(base[1:], moved[1:])
    q[12] = 0.15  # waist yaw affects both wrists and torso orientation
    turned = fk(q)
    assert not np.allclose(turned[2, 3:], moved[2, 3:])
    targets = ArmsOnlyTargets(fk)
    np.testing.assert_allclose(targets.calibrate(sample(), q, now=10.0), turned)


def test_planner_wire_contract():
    pose = robot_pose()
    message = build_planner_message(
        0, [0, 0, 0], [1, 0, 0], vr_3pt_position=pose[:, :3].ravel(), vr_3pt_orientation=pose[:, 3:].ravel()
    )
    header = json.loads(message[7 : 7 + HEADER_SIZE].rstrip(b"\0"))
    payload = message[7 + HEADER_SIZE :]
    decoded = {}
    offset = 0
    for field in header["fields"]:
        size = int(np.prod(field["shape"]))
        dtype = "<i4" if field["dtype"] == "i32" else "<f4"
        decoded[field["name"]] = np.frombuffer(payload, dtype=dtype, count=size, offset=offset)
        offset += size * 4
    assert decoded["mode"] == 0
    np.testing.assert_array_equal(decoded["movement"], [0, 0, 0])
    np.testing.assert_array_equal(decoded["facing"], [1, 0, 0])
    np.testing.assert_allclose(decoded["vr_position"], pose[:, :3].ravel())
    np.testing.assert_allclose(decoded["vr_orientation"], pose[:, 3:].ravel())
    assert "upper_body_position" not in decoded


def test_feedback_expires_and_rejects_missing_measured_state():
    import msgpack

    class Socket:
        data = None

        def poll(self, timeout):
            return self.data is not None

        def recv(self, flags):
            data, self.data = self.data, None
            return data

    feedback = MeasuredRobotFeedback.__new__(MeasuredRobotFeedback)
    feedback.socket = Socket()
    feedback.q, feedback.received_at = None, -float("inf")
    assert feedback.get(10.0) is None
    feedback.socket.data = b"g1_calibration" + msgpack.packb({"body_q_measured": [0.1] * 29, "state_age_s": 0.01})
    np.testing.assert_allclose(feedback.get(10.0), 0.1)
    assert feedback.get(10.3) is None
    feedback.socket.data = b"g1_calibration" + msgpack.packb({})
    assert feedback.get(10.31) is None


def test_manager_workflow_with_no_body_api(monkeypatch):
    """Run the actual manager loop, publishing locally to an in-memory socket."""
    import subprocess
    import time

    import zmq

    from gear_sonic.utils.teleop import arms_only

    messages = []

    class Socket:
        def setsockopt(self, *args):
            pass

        def bind(self, *args):
            pass

        def send(self, message):
            messages.append(message)

        def close(self, **kwargs):
            pass

    class Context:
        def socket(self, *args):
            return Socket()

        def term(self):
            pass

    class Feedback:
        def __init__(self, *args):
            self.socket = Socket()

        def get(self, now):
            return np.full(29, 0.1)

    class SDK(FakeSDK):
        tick = 0

        def init(self):
            pass

        def close(self):
            pass

        def get_controller_snapshot(self):
            self.packet["timeStampNs"] += 1
            # After two ticks, remove controller: hold targets and hands, but stop still works.
            if self.tick >= 2:
                self.packet["Controller"].pop("left", None)
            self.tick += 1
            return super().get_controller_snapshot()

    monkeypatch.setattr(zmq, "Context", Context)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: None)
    monkeypatch.setattr(time, "sleep", lambda *args: None)
    monkeypatch.setattr(arms_only, "MeasuredRobotFeedback", Feedback)
    monkeypatch.setattr(arms_only, "make_robot_fk", lambda: lambda q: robot_pose())
    controls = iter([[True] * 4, [False] * 4, [True, False, True, False], [True] * 4])
    hand_calls = []

    def hands(*args):
        hand_calls.append(args)
        return np.full((1, 7), 0.2), np.full((1, 7), 0.3)

    args = SimpleNamespace(
        left_grip_offset=[0, 0, 0],
        right_grip_offset=[0, 0, 0],
        port=5556,
        zmq_feedback_host="localhost",
        zmq_feedback_port=5557,
        target_fps=50,
        arms_only_timing=False,
    )
    arms_only.run_arms_only_manager(
        args,
        SDK(),
        lambda: (None, None),
        hands,
        lambda: next(controls),
        lambda: (False, 0.2, 0.3, 0.4, 0.5),
        lambda: (False, False),
    )
    planner = [m for m in messages if m.startswith(b"planner")]
    command = [m for m in messages if m.startswith(b"command")]
    assert len(planner) == 3
    assert planner[0] == planner[1] == planner[2]  # held targets and Dex3
    assert len(hand_calls) == 2
    assert len(command) == 2
    assert command[0][-3:] == bytes([1, 0, 1])
    assert command[1][-3:] == bytes([0, 1, 1])
    assert messages.index(planner[0]) < messages.index(command[0])


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], "pose"),
        (["--manager"], "legacy_manager"),
        (["--manager", "--arms-only"], "arms"),
        (["--arms-only"], "error"),
        (["--manager", "--arms-only", "--input-source", "isaac-teleop"], "error"),
        (["--manager", "--arms-only", "--vr3pt_live"], "error"),
        (["--manager", "--arms-only", "--vis_smpl"], "error"),
    ],
)
def test_cli_routes_opt_in_and_rejects_body_paths(monkeypatch, argv, expected):
    # Execute the actual CLI block in isolation from optional torch/display SDKs.
    import ast
    from pathlib import Path
    import sys

    from gear_sonic.utils.teleop import arms_only

    path = Path(__file__).parents[1] / "scripts/pico_manager_thread_server.py"
    tree = ast.parse(path.read_text())
    cli = tree.body[-1]
    assert isinstance(cli, ast.If)
    called = []
    monkeypatch.setattr(sys, "argv", [str(path), *argv])
    monkeypatch.setattr(arms_only, "run_arms_only_manager", lambda *args: called.append("arms"))
    namespace = {
        "run_pico": lambda **kwargs: called.append("pose"),
        "run_pico_manager": lambda **kwargs: called.append("legacy_manager"),
        "xrt": None,
        "init_hand_ik_solvers": None,
        "compute_hand_joints_from_inputs": None,
        "get_abxy_buttons": None,
        "get_controller_inputs": None,
        "get_axis_clicks": None,
    }
    try:
        exec(compile(ast.Module(body=cli.body, type_ignores=[]), str(path), "exec"), namespace)
    except SystemExit as exc:
        if exc.code:
            called.append("error")
    assert called == [expected]
