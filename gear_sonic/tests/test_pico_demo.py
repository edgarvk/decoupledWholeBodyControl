"""Recording/replay tests: no physical robot, vendor SDK or learned policy."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.teleop.arms_only import ArmsOnlySession
from gear_sonic.utils.teleop.demo_targets import DirectDemoTargets
from gear_sonic.utils.teleop.input_readers import PicoControllerReader
from gear_sonic.utils.teleop.pico_demo import (
    DemoReplay,
    DemoWriter,
    demo_summary,
    load_demo,
    packet_controls,
    require_local_simulation,
)


def packet(stamp=1, pressed=False, left=None, menu=False, click=False):
    controllers = {}
    for side in ("left", "right"):
        controllers[side] = {
            "pose": "0.2,1,-0.3,0,0,0,1",
            "axisX": 0.7,
            "axisY": -0.4,
            "trigger": 0.6,
            "grip": 0.2,
            "primaryButton": pressed,
            "secondaryButton": pressed,
            "menuButton": menu,
            "axisClick": click,
        }
    if left:
        controllers["left"].update(left)
    return {
        "timeStampNs": stamp,
        "receive_age_s": 0.0,
        "Controller": controllers,
        "Head": {"pose": "0,1.6,0,0,0,0,1"},
        "appState": {"focus": True},
    }


def write_demo(tmp_path, frames, name="demo.jsonl"):
    path = tmp_path / name
    writer = DemoWriter(path, {"description": "synthetic input"})
    for elapsed, value in frames:
        writer.sample(elapsed, value, valid=True)
    writer.close(frames[-1][0])
    return path


@pytest.mark.parametrize("name", ["demo.jsonl", "demo.jsonl.gz"])
def test_raw_roundtrip_preserves_head_axes_validity_missing_and_duplicate_packets(tmp_path, name):
    value = packet()
    value["Controller"]["left"]["trackingState"] = 3
    value["Controller"]["left"]["isTracked"] = True
    missing = copy.deepcopy(value)
    del missing["Controller"]["right"]
    path = write_demo(tmp_path, [(0, value), (0.01, value), (0.4, missing)], name)
    _, frames, complete = load_demo(path)
    assert complete and frames[0]["packet"] == value
    assert frames[1]["packet"] == value  # repeated source timestamp is not discarded
    assert frames[2]["elapsed_s"] == 0.4 and "right" not in frames[2]["packet"]["Controller"]
    assert demo_summary(path)["distinct_source_timestamps"] == 1
    assert demo_summary(path)["max_poll_gap_s"] == pytest.approx(0.39)
    with pytest.raises(FileExistsError):
        DemoWriter(path, {})


def test_torn_final_line_recovers_but_corruption_and_old_diagnostics_rejected(tmp_path):
    path = write_demo(tmp_path, [(0, packet())])
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2]) + '\n{"type":')
    assert load_demo(path)[2] is False
    path.write_text("\n".join(lines[:2]) + '\n{"type":\n')
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_demo(path)
    path.write_text(json.dumps({"type": "metadata", "schema": "arms-only-diagnostics-v1"}) + "\n")
    with pytest.raises(ValueError, match="Expected one"):
        load_demo(path)


def test_writer_handles_nonfinite_packet_and_load_rejects_backwards_local_time(tmp_path):
    path = tmp_path / "bad.jsonl"
    writer = DemoWriter(path, {})
    writer.sample(0.2, {"value": float("nan")}, valid=True)
    writer.sample(0.1, packet())
    writer.close(0.3)
    assert '"packet":null' in path.read_text()
    with pytest.raises(ValueError, match="timeline"):
        load_demo(path)


@pytest.mark.parametrize("speed", [0.5, 1, 2])
def test_virtual_clock_preserves_stale_gap_at_each_speed(tmp_path, speed):
    path = write_demo(tmp_path, [(0, packet()), (0.5, packet(2))])
    wall = [100.0]
    replay = DemoReplay(path, speed, clock=lambda: wall[0])
    reader = PicoControllerReader(replay, clock=replay.input_clock)
    replay.start()
    assert replay.advance() and reader.get_latest() is not None
    wall[0] += 0.3 / speed
    assert replay.advance() and reader.get_latest() is None
    assert "Stale" in reader.error
    wall[0] = 100 + 0.5 / speed
    replay.advance()
    assert reader.get_latest()["timestamp_ns"] == 2
    wall[0] = 100 + 0.6 / speed
    assert replay.advance() is False


def test_source_clock_reset_remains_reset_and_no_future_pose_used(tmp_path):
    path = write_demo(tmp_path, [(0.1, packet(10)), (0.2, packet(9))])
    wall = [0.0]
    replay = DemoReplay(path, clock=lambda: wall[0])
    reader = PicoControllerReader(replay, clock=replay.input_clock)
    replay.start()
    replay.advance()
    assert reader.get_latest() is None
    wall[0] = 0.1
    replay.advance()
    assert reader.get_latest()["timestamp_ns"] == 10
    wall[0] = 0.2
    replay.advance()
    assert reader.get_latest() is None and "Source clock reset" in reader.error


def test_atomic_controls_do_not_use_missing_device_cached_buttons():
    value = packet(pressed=True)
    buttons, inputs, clicks = packet_controls(value)
    assert all(buttons) and inputs == (False, 0.6, 0.6, 0.2, 0.2) and clicks == (False, False)
    del value["Controller"]["right"]
    assert packet_controls(value)[0] == (False, False, True, True)
    assert packet_controls(value)[1][2] == 0
    assert packet_controls({"Controller": [1, 2]})[0] == (False,) * 4


def test_direct_mapping_long_reachable_loop_returns_to_calibration_without_reference_drift(tmp_path):
    initial = packet()
    path = write_demo(tmp_path, [(0, initial)])
    replay = DemoReplay(path)
    replay.packet = initial
    reader = PicoControllerReader(replay, clock=replay.input_clock)
    start = reader.get_latest()
    pose = np.array([[0.3, 0.2, 0.1, 1, 0, 0, 0], [0.3, -0.2, 0.1, 1, 0, 0, 0], [0, 0, 0.4, 1, 0, 0, 0]])
    targets = DirectDemoTargets(lambda q: pose)
    targets.calibrate(start, np.zeros(29), now=0)
    origin = targets.controller_origin.copy()
    for i in range(1, 201):
        sample = copy.deepcopy(start)
        sample["timestamp_ns"] = i + 1
        sample["timestamp_monotonic"] = i * 0.02
        # Beyond the 60 cm live workspace, reached in increments below jump limit.
        sample["controller_poses"][0, 1] += 0.8 * np.sin(np.pi * i / 200)
        result = targets.update(sample, now=i * 0.02)
        assert not targets.held
    np.testing.assert_allclose(result, pose, atol=1e-12)
    np.testing.assert_array_equal(targets.controller_origin, origin)


@pytest.mark.parametrize(
    "commands",
    [
        "",
        "/tmp/g1_deploy_onnx_ref eth0 model data --input-type zmq_manager",
        "/tmp/g1_deploy_onnx_ref lo model data --input-type keyboard",
        "/tmp/g1_deploy_onnx_ref lo model data --input-type",
        "/tmp/g1_deploy_onnx_ref lo m d --input-type zmq_manager --zmq-host 192.168.1.5",
        "/tmp/g1_deploy_onnx_ref lo m d --input-type zmq_manager\n"
        "/tmp/g1_deploy_onnx_ref eth0 m d --input-type zmq_manager",
    ],
)
def test_replay_refuses_missing_real_unknown_or_multiple_deployments(commands):
    with pytest.raises(RuntimeError):
        require_local_simulation(commands)


def test_loopback_deployment_acceptance():
    assert (
        require_local_simulation("./target/release/g1_deploy_onnx_ref lo model data --input-type zmq_manager")[1]
        == "lo"
    )


def test_record_cli_keeps_acquisition_errors_and_never_opens_command_transport(tmp_path, monkeypatch):
    import sys

    import zmq

    from gear_sonic.scripts import pico_demo as cli

    wall = [100.0]
    calls = []

    def snapshot():
        calls.append("read")
        return "broken JSON" if calls.count("read") == 2 else json.dumps(packet(calls.count("read")))

    sdk = SimpleNamespace(
        init=lambda: calls.append("init"),
        close=lambda: calls.append("close"),
        get_controller_snapshot=snapshot,
    )
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", sdk)
    monkeypatch.setattr(cli.time, "monotonic", lambda: wall[0])
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: wall.__setitem__(0, wall[0] + seconds))
    monkeypatch.setattr(cli, "code_version", lambda: {"commit": "synthetic"})
    monkeypatch.setattr(zmq, "Context", lambda: pytest.fail("Recording must not open command sockets"))
    path = tmp_path / "recorded.jsonl"
    cli.record(
        SimpleNamespace(
            output=str(path),
            poll_hz=100,
            duration=0.06,
            description="test",
            start_service=False,
        )
    )
    _, frames, complete = load_demo(path)
    assert complete and len(frames) >= 5
    assert frames[1]["packet"] is None and frames[1]["input_error"]
    assert frames[0]["packet"]["Controller"]["left"]["axisX"] == 0.7
    assert calls[0] == "init" and calls[-1] == "close"


def test_auto_start_does_not_stop_on_recorded_abxy_and_releases_start_edge(tmp_path):
    replay = DemoReplay(write_demo(tmp_path, [(0, packet(pressed=True))]), start_mode="auto")
    replay.packet = packet(pressed=True)
    reader = PicoControllerReader(replay, clock=replay.input_clock)
    sample = reader.get_latest()
    targets = DirectDemoTargets(
        lambda q: np.array([[0, 0.2, 0, 1, 0, 0, 0], [0, -0.2, 0, 1, 0, 0, 0], [0, 0, 0.4, 1, 0, 0, 0]])
    )
    session = ArmsOnlySession(targets)
    action, _ = session.step(sample, np.zeros(29), replay.buttons(sample, session), now=0)
    assert action == "start"
    assert replay.buttons(sample, session) == (False,) * 4
    action, _ = session.step(sample, np.zeros(29), replay.buttons(sample, session), now=0)
    assert action is None and session.active


def test_manager_replay_seeds_targets_before_start_and_stops_at_eof_without_vendor_service(tmp_path, monkeypatch):
    import subprocess

    import zmq

    from gear_sonic.utils.teleop import arms_only, pico_demo
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE

    path = write_demo(
        tmp_path,
        [
            (0, packet()),
            (0.02, packet(2, pressed=True)),
            (0.04, packet(3)),
            (0.06, packet(4, left={"pose": "0.2,1.02,-0.3,0,0,0,1"})),
            (0.08, packet(5)),
        ],
    )
    wall = [100.0]
    monkeypatch.setattr(arms_only.time, "monotonic", lambda: wall[0])
    monkeypatch.setattr(arms_only.time, "sleep", lambda seconds: wall.__setitem__(0, wall[0] + seconds))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Replay must not start vendor service"))
    monkeypatch.setattr(pico_demo, "require_local_simulation", lambda: ["g1_deploy_onnx_ref", "lo"])
    replay = DemoReplay(path, clock=lambda: wall[0])
    pose = np.array([[0.3, 0.2, 0.1, 1, 0, 0, 0], [0.3, -0.2, 0.1, 1, 0, 0, 0], [0, 0, 0.4, 1, 0, 0, 0]])
    monkeypatch.setattr(arms_only, "make_robot_fk", lambda: lambda q: pose)
    observer = SimpleNamespace(observe=lambda q, target: {}, solution_diagnostics=[])
    monkeypatch.setattr(arms_only, "make_target_filter", lambda value: observer)
    sent = []
    sock = SimpleNamespace(
        setsockopt=lambda *a: None,
        bind=lambda address: sent.append(address),
        send=sent.append,
        close=lambda **k: None,
    )
    context = SimpleNamespace(socket=lambda *a: sock, term=lambda: None)
    monkeypatch.setattr(zmq, "Context", lambda: context)
    monkeypatch.setattr(
        arms_only,
        "MeasuredRobotFeedback",
        lambda *a: SimpleNamespace(
            socket=sock,
            get=lambda now: np.zeros(29),
            telemetry={},
        ),
    )
    args = SimpleNamespace(
        left_grip_offset=[0, 0, 0],
        right_grip_offset=[0, 0, 0],
        arms_max_gravity_nm=8,
        port=5556,
        zmq_feedback_host="127.0.0.1",
        zmq_feedback_port=5557,
        arms_only_log=str(tmp_path / "results.jsonl"),
        arms_only_timing=False,
        target_fps=50,
    )
    arms_only.run_arms_only_manager(
        args,
        replay,
        lambda: (None, None),
        lambda *a: (np.zeros(7), np.zeros(7)),
        lambda: packet_controls(replay.packet)[0],
        lambda: packet_controls(replay.packet)[1],
        lambda: packet_controls(replay.packet)[2],
        replay_driver=replay,
        target_mode="direct",
    )
    assert sent[0] == "tcp://127.0.0.1:5556"
    messages = [m for m in sent if isinstance(m, bytes)]
    first_start = next(i for i, m in enumerate(messages) if m.startswith(b"command") and m[7 + HEADER_SIZE] == 1)
    assert messages[first_start - 1].startswith(b"planner")
    stops = [m for m in messages if m.startswith(b"command") and m[8 + HEADER_SIZE] == 1]
    assert len(stops) == 3
    events = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert any(r.get("event") == "calibration" and r["measured_body_q"] == [0] * 29 for r in events)
