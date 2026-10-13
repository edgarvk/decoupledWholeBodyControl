"""Offline G1 kinematic replay: no deployment process or robot connection."""

import json
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from gear_sonic.tests.test_arms_only import robot_pose, sample
from gear_sonic.utils.teleop.arms_only import ArmDiagnostics, ArmsOnlyTargets, make_robot_fk


@pytest.fixture
def kinematics():
    pytest.importorskip("pinocchio")
    from gear_sonic.utils.teleop.arm_target_filter import G1ArmTargetFilter

    q = np.zeros(29)
    q[16], q[23], q[18], q[25] = 0.35, -0.35, 1.0, 1.0
    return G1ArmTargetFilter(), make_robot_fk(), q


def test_workspace_boundary_allows_immediate_inward_motion_without_reanchor():
    targets = ArmsOnlyTargets(lambda q: robot_pose())
    targets.calibrate(sample(), np.zeros(29), now=10.0)
    for i in range(1, 101):
        frame = sample(i + 1)
        frame["controller_poses"][0, 1] += 0.01 * i
        result = targets.update(frame, now=10.0)
        assert not targets.held
        assert np.linalg.norm(result[0, :3] - robot_pose()[0, :3]) <= 0.60000001
        np.testing.assert_allclose(result[1:], robot_pose()[1:])
    assert targets.limit_status[0] == "workspace"
    frame = sample(102)
    frame["controller_poses"][0, 1] += 0.99
    result = targets.update(frame, now=10.0)
    assert result[0, 2] == pytest.approx(robot_pose()[0, 2] + 0.59)
    assert not targets.held


def test_jump_detection_uses_raw_tracking_not_limited_targets():
    targets = ArmsOnlyTargets(lambda q: robot_pose())
    targets.calibrate(sample(), np.zeros(29), now=10.0)
    for i in range(1, 120):
        frame = sample(i + 1)
        frame["controller_poses"][0, 1] += i * 0.01
        targets.update(frame, now=10.0)
    assert not targets.held
    jump = sample(121)
    jump["controller_poses"][0, 1] += 1.4
    previous = targets.target.copy()
    np.testing.assert_allclose(targets.update(jump, now=10.0), previous)
    assert targets.held and "left tracking jump" in targets.reason
    assert "translation=" in targets.reason and "rotation=" in targets.reason


def test_offset_point_jacobian_matches_g1_finite_differences(kinematics):
    filter_, _, body_q = kinematics
    q = filter_.robot.get_configuration_from_actuated_joints(body_q)
    for arm in filter_.arms:
        filter_._frames(q, jacobians=True)
        pose = filter_._pose(arm).copy()
        jac = filter_._jacobian(arm)[:3]
        numeric = []
        for index in arm["indices"]:
            perturbed = q.copy()
            perturbed[index] += 1e-6
            filter_._frames(perturbed)
            numeric.append((filter_._pose(arm)[:3] - pose[:3]) / 1e-6)
        np.testing.assert_allclose(jac, np.array(numeric).T, atol=4e-7)


def test_filtered_initial_and_head_only_targets_stay_exact(kinematics):
    filter_, fk, q = kinematics
    targets = ArmsOnlyTargets(fk, target_filter=filter_)
    initial = targets.calibrate(sample(), q, now=10.0)
    original_solution = filter_.q.copy()
    for stamp in range(2, 12):
        frame = sample(stamp)
        frame["headset_pose"][3:] = R.from_euler("xyz", [0.5, 0.8, -0.7]).as_quat()
        frame["headset_pose"][:3] += [4, 5, 6]
        np.testing.assert_allclose(targets.update(frame, now=10.0, measured_q=q), initial, atol=1e-12)
    np.testing.assert_array_equal(filter_.q, original_solution)


def test_g1_projection_moves_toward_reachable_request_and_never_moves_other_joints(kinematics):
    filter_, fk, q = kinematics
    previous = fk(q)
    filter_.reset(q, previous)
    baseline = filter_.q.copy()
    request_q = q.copy()
    request_q[15] -= 0.04
    request_q[18] += 0.04
    requested = fk(request_q)
    original_error = np.linalg.norm(requested[0, :3] - previous[0, :3])
    for _ in range(10):
        result = filter_.project(requested, previous)
        assert np.linalg.norm(result[0, :3] - previous[0, :3]) <= 0.02000001
        angle = (
            R.from_quat(result[0, 3:], scalar_first=True) * R.from_quat(previous[0, 3:], scalar_first=True).inv()
        ).magnitude()
        assert angle <= np.deg2rad(5) + 1e-8
        np.testing.assert_allclose(result[1:], requested[1:], atol=1e-12)
        previous = result
    assert np.linalg.norm(previous[0, :3] - requested[0, :3]) < original_error
    other = np.setdiff1d(np.arange(filter_.model.nq), filter_.arms[0]["indices"])
    np.testing.assert_array_equal(filter_.q[other], baseline[other])
    metrics = filter_.solution_diagnostics[0]
    assert metrics["min_comfort_margin_rad"] >= -1e-8
    assert metrics["elbow_above_shoulder_m"] <= 0.02000001
    assert max(abs(torque) for torque in metrics["gravity_nm"]) <= filter_.max_gravity_nm


def test_extreme_position_and_orientation_are_projected_with_inward_recovery(kinematics):
    filter_, fk, q = kinematics
    targets = ArmsOnlyTargets(fk, target_filter=filter_)
    initial = targets.calibrate(sample(), q, now=10.0)
    for stamp in range(2, 103):
        frame = sample(stamp)
        frame["controller_poses"][0, 1] += 0.01 * (stamp - 1)
        frame["controller_poses"][0, 3:] = R.from_euler("y", 0.02 * (stamp - 1)).as_quat()
        previous = targets.target.copy()
        result = targets.update(frame, now=10.0, measured_q=q)
        assert not targets.held
        assert np.linalg.norm(result[0, :3] - previous[0, :3]) <= 0.02000001
        np.testing.assert_allclose(result[1:], initial[1:], atol=1e-12)
        assert filter_.solution_diagnostics[0]["elbow_above_shoulder_m"] <= 0.02000001
    high = result.copy()
    for stamp in range(103, 155):
        frame = sample(stamp)
        frame["controller_poses"][0, 1] += 1.02 - 0.01 * (stamp - 102)
        frame["controller_poses"][0, 3:] = R.from_euler("y", 2.04 - 0.02 * (stamp - 102)).as_quat()
        result = targets.update(frame, now=10.0, measured_q=q)
        assert not targets.held
    assert result[0, 2] < high[0, 2] - 0.02


def test_reanchor_keeps_projection_and_output_continuous(kinematics):
    filter_, fk, q = kinematics
    targets = ArmsOnlyTargets(fk, target_filter=filter_)
    targets.calibrate(sample(), q, now=10.0)
    frame = sample(2)
    frame["controller_poses"][0, 2] -= 0.02
    previous = targets.update(frame, now=10.0, measured_q=q)
    solution = filter_.q.copy()
    frame["controller_poses"][0, 2] += 0.3
    actual_lag = q.copy()
    actual_lag[15] += 0.03
    np.testing.assert_allclose(targets.calibrate(frame, actual_lag, now=10.0), previous)
    np.testing.assert_array_equal(filter_.q, solution)
    np.testing.assert_allclose(targets.update(frame, now=10.0, measured_q=actual_lag), previous)


def test_missing_robot_state_latches_hold_and_keeps_stop_available(kinematics):
    filter_, fk, q = kinematics
    from gear_sonic.utils.teleop.arms_only import ArmsOnlySession

    session = ArmsOnlySession(ArmsOnlyTargets(fk, target_filter=filter_))
    assert session.step(sample(), q, [True] * 4, now=10.0)[0] == "start"
    initial = session.targets.target.copy()
    _, target = session.step(sample(2), None, [False] * 4, now=10.0)
    np.testing.assert_allclose(target, initial)
    assert session.targets.held and "feedback" in session.targets.reason
    assert session.step(None, None, [True] * 4, now=10.3) == ("stop", None)


def test_diagnostics_capture_hold_actual_posture_and_temperature(tmp_path, kinematics, caplog):
    filter_, fk, q = kinematics
    targets = ArmsOnlyTargets(fk, target_filter=filter_)
    targets.calibrate(sample(), q, now=10.0)
    targets.hold("synthetic tracking loss")
    hot = np.full(58, 30.0)
    hot[2 * 25] = 92.0  # right elbow, winding temperature
    telemetry = {
        "motor_temperature": hot.tolist(),
        "motor_torque": np.full(29, 1.2).tolist(),
        "body_dq_measured": np.zeros(29).tolist(),
        "timestamp_unix_ns": 123,
    }
    path = tmp_path / "trial.jsonl"
    logger = ArmDiagnostics(path)
    logger.record(targets, None, q, telemetry, 10.3, 456, 1.5)
    logger.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    row = records[-1]
    assert row["held"] and row["hold_reason"] == "synthetic tracking loss"
    assert row["incoming_source_ns"] is None
    assert row["motor_temperature"][50] == 92.0
    assert row["motor_torque"][25] == 1.2
    assert len(row["measured_arm_q"]) == 2
    assert row["wrist_error_m"] == pytest.approx([0.0, 0.0])
    assert "ARM OVERHEATING" in caplog.text


def test_projection_work_budget_and_timing_report(kinematics):
    filter_, fk, q = kinematics
    pose = fk(q)
    filter_.reset(q, pose)
    timings = []
    for i in range(30):
        requested = pose.copy()
        requested[0, 0] += 0.003
        start = time.perf_counter()
        pose = filter_.project(requested, pose)
        timings.append((time.perf_counter() - start) * 1000)
    # No flaky time assertion: expose observed overhead for the trial report.
    print(f"G1 projection replay ms median={np.median(timings):.3f} max={max(timings):.3f}")
    assert len(filter_.solution_diagnostics) == 2
