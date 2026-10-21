"""Record raw PICO input once, inspect it, and replay it through SONIC in MuJoCo.

Run from the repository root: python -m gear_sonic.scripts.pico_demo --help
Recording never opens a command socket. Replay requires a local sim deployment.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from gear_sonic.utils.teleop.pico_demo import (
    DemoReplay,
    DemoWriter,
    code_version,
    demo_summary,
    packet_controls,
    require_local_simulation,
)


class CachedSnapshot:
    def __init__(self):
        self.packet = {}

    def get_controller_snapshot(self):
        return json.dumps(self.packet)


def record(args):
    import xrobotoolkit_sdk as sdk

    from gear_sonic.utils.teleop.input_readers import PicoControllerReader

    if not callable(getattr(sdk, "get_controller_snapshot", None)):
        raise RuntimeError("Rebuild the bundled XRoboToolkit binding: get_controller_snapshot is required")
    cached = CachedSnapshot()
    reader = PicoControllerReader(cached)
    writer = DemoWriter(
        args.output,
        {
            "code": code_version(),
            "poll_hz": args.poll_hz,
            "description": args.description,
        },
    )
    initialized = False
    start = time.monotonic()
    try:
        if args.start_service:
            service = Path(args.service_script)
            if not service.is_file():
                raise RuntimeError(f"XRoboToolkit service script not found: {service}")
            subprocess.Popen(["bash", str(service)])
        sdk.init()
        initialized = True
        print("Recording only: no robot commands. Hold a neutral pose and press A+B+X+Y to mark replay start.")
        print("Perform the demonstration, press A+B+X+Y again to mark stop, then Ctrl-C to finish recording.")
        last_status = start
        while not args.duration or time.monotonic() - start < args.duration:
            tick = time.monotonic()
            error = ""
            packet = None
            try:
                packet = json.loads(sdk.get_controller_snapshot())
                if not isinstance(packet, dict):
                    raise ValueError("Expected a snapshot object")
            except (ValueError, TypeError, RuntimeError) as exc:
                packet, error = None, str(exc)
            cached.packet = packet or {}
            sample = reader.get_latest()
            writer.sample(tick - start, packet, error, sample is not None, reader.error)
            if tick - last_status >= 2:
                print(
                    f"{tick - start:.1f}s / {writer.count} polls: {reader.error or 'controller poses valid'}",
                    flush=True,
                )
                last_status = tick
            time.sleep(max(0.0, 1 / args.poll_hz - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("Finishing recording...")
    finally:
        try:
            if initialized:
                sdk.close()
        finally:
            writer.close(time.monotonic() - start)
    print(json.dumps(demo_summary(args.output), indent=2))


def replay(args):
    driver = DemoReplay(args.input, args.speed, start_mode=args.start_mode)
    deployment = require_local_simulation()
    if not driver.complete:
        print("Recording has no complete footer; replaying recovered complete input rows.")
    if args.start_mode == "buttons" and not demo_summary(args.input)["start_stop_button_edges"]:
        raise ValueError("No recorded A+B+X+Y start edge; use --start-mode auto or record a start marker")
    output = Path(args.log).expanduser()
    if output.exists():
        raise FileExistsError(f"Use a new result filename: {output}")
    driver.configuration = {
        "kind": "simulation_replay",
        "input_path": driver.path,
        "input_sha256": driver.file_sha256,
        "recorded_code": driver.metadata.get("code"),
        "replay_code": code_version(),
        "mapping": args.mapping,
        "mirror": args.mirror,
        "gripper_profile": args.gripper_profile,
        "speed": args.speed,
        "target_fps": args.target_fps,
        "start_mode": args.start_mode,
        "gravity_budget_nm": args.arms_max_gravity_nm,
        "simulation_command": deployment,
        "tracking_timing": "virtual recording time; measured simulation feedback uses wall time",
    }
    # Reuse the actual manager, wire protocol, FK, filter and legacy Dex3 solver.
    # No headset SDK initialization or roboticsservice process is needed here.
    from gear_sonic.scripts.pico_manager_thread_server import (
        compute_hand_joints_from_inputs,
        init_hand_ik_solvers,
    )
    from gear_sonic.utils.teleop.arms_only import run_arms_only_manager

    manager_args = SimpleNamespace(
        mirror=args.mirror,
        gripper_profile=args.gripper_profile,
        left_grip_offset=args.left_grip_offset,
        right_grip_offset=args.right_grip_offset,
        arms_max_gravity_nm=args.arms_max_gravity_nm,
        port=5556,
        zmq_feedback_host="127.0.0.1",
        zmq_feedback_port=5557,
        arms_only_log=str(output),
        arms_only_timing=False,
        target_fps=args.target_fps,
    )
    run_arms_only_manager(
        manager_args,
        driver,
        lambda: init_hand_ik_solvers(profile=args.gripper_profile),
        compute_hand_joints_from_inputs,
        lambda: packet_controls(driver.packet)[0],
        lambda: packet_controls(driver.packet)[1],
        lambda: packet_controls(driver.packet)[2],
        replay_driver=driver,
        target_mode=args.mapping,
    )
    print(f"Simulation result log: {output}")


def preview(args):
    """Deterministic mapping-only preview. This does not run the learned policy."""
    import numpy as np

    from gear_sonic.utils.teleop.arms_only import (
        ArmsOnlySession,
        ArmsOnlyTargets,
        make_robot_fk,
        make_target_filter,
    )
    from gear_sonic.utils.teleop.demo_targets import DirectDemoTargets
    from gear_sonic.utils.teleop.input_readers import PicoControllerReader

    q = np.asarray(json.loads(Path(args.initial_joints).expanduser().read_text()), dtype=float)
    if q.shape != (29,) or not np.all(np.isfinite(q)):
        raise ValueError(
            "--initial-joints must contain a JSON array of 29 finite G1 motor-order angles in radians"
        )
    driver = DemoReplay(args.input, start_mode=args.start_mode)
    cached = CachedSnapshot()
    reader = PicoControllerReader(cached, clock=driver.input_clock)
    cls = ArmsOnlyTargets if args.mapping == "filtered" else DirectDemoTargets
    targets = cls(
        make_robot_fk(),
        [args.left_grip_offset, args.right_grip_offset],
        make_target_filter(args.arms_max_gravity_nm) if args.mapping == "filtered" else None,
        mirror=args.mirror,
    )
    session = ArmsOnlySession(targets)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("x", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "type": "metadata",
                    "schema": "pico-target-preview-v1",
                    "policy_executed": False,
                    "mapping": args.mapping,
                    "mirror": args.mirror,
                    "gripper_profile": args.gripper_profile,
                    "input_sha256": driver.file_sha256,
                    "initial_body_q": q.tolist(),
                    "code": code_version(),
                }
            )
            + "\n"
        )
        for row in driver.frames:
            driver.elapsed_s = row["elapsed_s"]
            driver.packet = cached.packet = row.get("packet") or {}
            sample = reader.get_latest()
            buttons = driver.buttons(sample, session)
            _, inputs, clicks = packet_controls(driver.packet)
            action, pose = session.step(sample, q, buttons, clicks[0], inputs[0], driver.input_clock())
            if action == "stop":
                break
            if pose is None:
                continue
            file.write(
                json.dumps(
                    {
                        "type": "target",
                        "elapsed_s": row["elapsed_s"],
                        "source_ns": targets.source_timestamp_ns,
                        "action": action,
                        "held": targets.held,
                        "hold_reason": targets.reason,
                        "limits": targets.limit_status,
                        "requested_pose": targets.requested_pose.tolist(),
                        "outgoing_pose": pose.tolist(),
                        "frame": targets.frame.tolist(),
                    }
                )
                + "\n"
            )
            count += 1
    if not count:
        raise ValueError("Preview produced no targets; check headset pose and recorded start buttons")
    print(f"{count} mapping-only targets written to {output}; no SONIC policy or robot transport used.")


def report(args):
    """Summarize simulation diagnostics without mistaking preview targets for policy output."""
    metadata, rows = [], []
    with Path(args.input).expanduser().open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row.get("type") == "metadata":
                if row.get("schema") != "arms-only-diagnostics-v1":
                    raise ValueError(
                        "Report needs a manager simulation/live diagnostic log, not a raw demo or preview"
                    )
                metadata.append(row.get("context", {}))
            elif row.get("type") == "target":
                rows.append(row)
    if not rows:
        raise ValueError("No diagnostic targets found")

    def stats(values):
        values = sorted(values)
        if not values:
            return None
        return {
            "samples": len(values),
            "median": values[len(values) // 2],
            "p95": values[int(0.95 * (len(values) - 1))],
            "max": values[-1],
        }

    result = {"runs": metadata, "samples": len(rows), "held_samples": sum(r["held"] for r in rows)}
    for arm, name in enumerate(("left", "right")):
        result[name] = {
            key: stats([r[key][arm] for r in rows if key in r])
            for key in (
                "projection_position_error_m",
                "projection_rotation_error_rad",
                "wrist_error_m",
                "wrist_error_rad",
            )
        }
        result[name]["limited_samples"] = sum(r["limits"][arm] not in ("", "tracking") for r in rows)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record", help="Record raw headset/controllers without commanding a robot")
    rec.add_argument("--output", required=True, help="New .jsonl or .jsonl.gz filename")
    rec.add_argument("--poll-hz", type=float, default=100)
    rec.add_argument("--duration", type=float, default=0, help="Seconds; 0 records until Ctrl-C")
    rec.add_argument("--description", default="")
    rec.add_argument("--start-service", action="store_true")
    rec.add_argument("--service-script", default="/opt/apps/roboticsservice/runService.sh")
    rec.set_defaults(func=record)
    inspect = commands.add_parser("inspect", help="Show recording completeness, gaps and start/stop markers")
    inspect.add_argument("--input", required=True)
    inspect.set_defaults(func=lambda args: print(json.dumps(demo_summary(args.input), indent=2)))
    rep = commands.add_parser("replay", help="Replay through the existing SONIC manager and local simulation")
    rep.add_argument("--input", required=True)
    rep.add_argument("--log", required=True, help="New diagnostic .jsonl result filename")
    rep.add_argument("--speed", type=float, default=1)
    rep.add_argument("--target-fps", type=int, default=50)
    rep.set_defaults(func=replay)
    pre = commands.add_parser("preview", help="Offline mapping-only output; does not execute SONIC")
    pre.add_argument("--input", required=True)
    pre.add_argument("--output", required=True)
    pre.add_argument("--initial-joints", required=True)
    pre.set_defaults(func=preview)
    for command in (rep, pre):
        command.add_argument("--mirror", action="store_true")
        command.add_argument("--gripper-profile", choices=("legacy", "rtc", "enhanced"), default="legacy")
        command.add_argument("--mapping", choices=("filtered", "direct"), default="filtered")
        command.add_argument("--start-mode", choices=("buttons", "auto"), default="buttons")
        command.add_argument("--arms-max-gravity-nm", type=float, default=8)
        command.add_argument("--left-grip-offset", type=float, nargs=3, default=[0, 0, 0])
        command.add_argument("--right-grip-offset", type=float, nargs=3, default=[0, 0, 0])
    summary = commands.add_parser("report", help="Summarize target and actual-wrist errors in a result log")
    summary.add_argument("--input", required=True)
    summary.set_defaults(func=report)
    args = parser.parse_args()
    if args.command == "record":
        if not math.isfinite(args.poll_hz) or not 10 <= args.poll_hz <= 500:
            parser.error("--poll-hz must be between 10 and 500")
        if not math.isfinite(args.duration) or args.duration < 0:
            parser.error("--duration must be non-negative and finite")
    if args.command in ("preview", "replay"):
        if not math.isfinite(args.arms_max_gravity_nm) or args.arms_max_gravity_nm <= 0:
            parser.error("--arms-max-gravity-nm must be positive and finite")
        if not all(math.isfinite(x) for x in args.left_grip_offset + args.right_grip_offset):
            parser.error("Grip offsets must be finite")
    if args.command == "replay" and not 1 <= args.target_fps <= 100:
        parser.error("--target-fps must be between 1 and 100")
    try:
        args.func(args)
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"pico_demo: {exc}\n")


if __name__ == "__main__":
    main()
