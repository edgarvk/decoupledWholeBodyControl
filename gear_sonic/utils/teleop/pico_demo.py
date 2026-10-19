"""Raw PICO input capture and timed playback, independent of robot control.

The SDK exposes a latest-packet snapshot, not a lossless packet subscription.
We record every poll (including repeats, errors and stale poses) so gaps are not
compressed away. Source timestamps remain unchanged during playback.
"""

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

SCHEMA = "pico-input-demo-v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def code_version():
    def git(*args):
        result = subprocess.run(
            ["git", "-c", "filter.lfs.process=", "-c", "filter.lfs.required=false", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {"commit": git("rev-parse", "HEAD"), "worktree_status": git("status", "--porcelain")}


def open_text(path, mode):
    path = Path(path).expanduser()
    return (
        gzip.open(path, mode + "t", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(mode, encoding="utf-8")
    )


class DemoWriter:
    """Append only to a new recording; flush each sample for interrupt recovery."""

    def __init__(self, path, metadata):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open_text(path, "x")
        self.count = 0
        self.write(
            {
                "type": "metadata",
                "schema": SCHEMA,
                "started_unix_ns": time.time_ns(),
                "pose_order": "x,y,z,qx,qy,qz,qw",
                "position_unit": "metres",
                "tracking_frame": "XRT: +X right, +Y up, -Z forward",
                "timing": "local monotonic elapsed time; original device timestamp preserved",
                "acquisition": "latest atomic SDK snapshot at each poll; not a lossless packet stream",
                "optical_validity": "only available when explicitly included by the headset client",
                **metadata,
            }
        )

    def write(self, row):
        self.file.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        self.file.flush()

    def sample(self, elapsed_s, packet, error="", valid=False, validation_error=""):
        row = {
            "type": "input",
            "elapsed_s": elapsed_s,
            "received_unix_ns": time.time_ns(),
            "packet": packet,
            "input_error": error,
            "valid_controllers": valid,
            "validation_error": validation_error,
        }
        try:
            json.dumps(row, allow_nan=False)
        except (TypeError, ValueError):
            row.update(packet=None, valid_controllers=False, input_error="Non-JSON or non-finite SDK snapshot")
        self.write(row)
        self.count += 1

    def close(self, elapsed_s):
        if not self.file.closed:
            try:
                self.write({"type": "end", "elapsed_s": elapsed_s, "samples": self.count})
            finally:
                self.file.close()


def load_demo(path):
    """Validate before any publisher starts; tolerate only a torn final JSON line."""
    metadata = None
    frames = []
    complete = False
    torn_tail = False
    with open_text(path, "r") as file:
        for number, line in enumerate(file, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    torn_tail = True
                    break
                raise ValueError(f"Invalid JSON at line {number}") from None
            if not isinstance(row, dict):
                raise ValueError(f"Invalid row at line {number}")
            kind = row.get("type")
            if kind == "metadata":
                if metadata is not None or frames or row.get("schema") != SCHEMA:
                    raise ValueError("Expected one pico-input-demo-v1 recording per file")
                metadata = row
            elif kind == "input":
                elapsed = row.get("elapsed_s")
                if (
                    metadata is None
                    or complete
                    or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed)
                    or elapsed < 0
                    or (frames and elapsed < frames[-1]["elapsed_s"])
                    or (row.get("packet") is not None and not isinstance(row["packet"], dict))
                ):
                    raise ValueError(f"Invalid input/timeline at line {number}")
                # Reject non-finite values rather than silently fabricating valid data.
                json.dumps(row, allow_nan=False)
                frames.append(row)
            elif kind == "end":
                if complete or metadata is None:
                    raise ValueError(f"Invalid end row at line {number}")
                if row.get("samples") != len(frames):
                    raise ValueError("Recording footer sample count does not match input rows")
                complete = True
            else:
                raise ValueError(f"Unknown row type at line {number}")
    if metadata is None or not frames:
        raise ValueError("Recording contains no input samples")
    return metadata, frames, complete and not torn_tail


def demo_summary(path):
    metadata, frames, complete = load_demo(path)
    stamps = [r["packet"].get("timeStampNs") for r in frames if r.get("packet")]
    stamps = [s for s in stamps if isinstance(s, int)]
    resets = sum(b < a for a, b in zip(stamps, stamps[1:]))
    button_edges = 0
    previous = False
    for row in frames:
        buttons = packet_controls(row.get("packet"))[0]
        pressed = all(buttons)
        button_edges += pressed and not previous
        previous = pressed
    intervals = [b["elapsed_s"] - a["elapsed_s"] for a, b in zip(frames, frames[1:])]
    return {
        "schema": metadata["schema"],
        "complete": complete,
        "samples": len(frames),
        "duration_s": frames[-1]["elapsed_s"],
        "distinct_source_timestamps": len(set(stamps)),
        "source_clock_resets": resets,
        "start_stop_button_edges": button_edges,
        "valid_controller_samples": sum(bool(r.get("valid_controllers")) for r in frames),
        "input_error_samples": sum(bool(r.get("input_error")) for r in frames),
        "max_poll_gap_s": max(intervals, default=0),
        "recorded_code": metadata.get("code"),
    }


def packet_controls(packet):
    """ABXY and analog controls from the same snapshot as the poses.

    Joystick axes are kept in the raw recording, but arms-only never uses them for
    locomotion. Missing devices do not inherit cached buttons/trigger values.
    """
    controllers = (packet or {}).get("Controller") or {}
    if not isinstance(controllers, dict):
        controllers = {}
    left = controllers.get("left") or {}
    right = controllers.get("right") or {}
    left = left if isinstance(left, dict) else {}
    right = right if isinstance(right, dict) else {}

    def button(device, key):
        return device.get(key) is True

    def analog(device, key):
        try:
            value = float(device.get(key, 0.0))
            return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    buttons = (
        button(right, "primaryButton"),
        button(right, "secondaryButton"),
        button(left, "primaryButton"),
        button(left, "secondaryButton"),
    )
    inputs = (
        button(left, "menuButton"),
        analog(left, "trigger"),
        analog(right, "trigger"),
        analog(left, "grip"),
        analog(right, "grip"),
    )
    clicks = (button(left, "axisClick"), button(right, "axisClick"))
    return buttons, inputs, clicks


class DemoReplay:
    """Retimed SDK-compatible input with a virtual clock for tracking freshness.

    Wall time controls playback speed; virtual time controls the original tracking
    gaps. No interpolation fills missing poses. Source clock resets remain resets.
    """

    def __init__(self, path, speed=1.0, clock=time.monotonic, start_mode="buttons"):
        if not math.isfinite(speed) or not 0.1 <= speed <= 2.0:
            raise ValueError("Replay speed must be between 0.1 and 2")
        if start_mode not in ("buttons", "auto"):
            raise ValueError("Unknown replay start mode")
        self.metadata, self.frames, self.complete = load_demo(path)
        self.path = str(Path(path).expanduser().resolve())
        self.speed, self.wall_clock, self.start_mode = speed, clock, start_mode
        self.started_at = None
        self.index = -1
        self.elapsed_s = 0.0
        self.packet = {}
        self.auto_started = False
        self.file_sha256 = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()

    def start(self):
        self.started_at = self.wall_clock()

    def input_clock(self):
        return self.elapsed_s

    def advance(self):
        self.elapsed_s = max(0.0, (self.wall_clock() - self.started_at) * self.speed)
        if self.elapsed_s > self.frames[-1]["elapsed_s"] + 0.02:
            return False
        while self.index + 1 < len(self.frames) and self.frames[self.index + 1]["elapsed_s"] <= self.elapsed_s:
            self.index += 1
        self.packet = copy.deepcopy(self.frames[self.index].get("packet") or {}) if self.index >= 0 else {}
        if self.index >= 0:
            original_age = self.packet.get("receive_age_s", 0.0)
            try:
                self.packet["receive_age_s"] = (
                    float(original_age) + self.elapsed_s - self.frames[self.index]["elapsed_s"]
                )
            except (ValueError, TypeError):
                self.packet["receive_age_s"] = None
        return True

    def get_controller_snapshot(self):
        return json.dumps(self.packet)

    def init(self):
        pass

    def close(self):
        pass

    def buttons(self, sample, session):
        if self.start_mode == "buttons":
            return packet_controls(self.packet)[0]
        # Auto is explicit: disregard recorded ABXY; menu/reanchor remain active.
        if not self.auto_started and sample is not None and sample["headset_pose"] is not None:
            if session.active:
                self.auto_started = True
            else:
                return (not session.prev_start,) * 4
        return (False,) * 4


def require_local_simulation(process_list=None):
    """Replay publishes only with a local loopback SONIC deployment.

    The C++ binary's first positional argument is its DDS network interface.
    Refuse real/unknown interfaces and multiple deployments before publishing.
    """
    if process_list is None:
        process_list = subprocess.check_output(["ps", "-axww", "-o", "args="], text=True)
    deployments = []
    for line in process_list.splitlines():
        try:
            argv = shlex.split(line)
        except ValueError:
            continue
        if argv and Path(argv[0]).name == "g1_deploy_onnx_ref":
            deployments.append(argv)
    if len(deployments) != 1 or len(deployments[0]) < 2 or deployments[0][1] not in ("lo", "lo0"):
        raise RuntimeError(
            "Replay needs exactly one local g1_deploy_onnx_ref on lo/lo0. "
            "Start deploy.sh ... sim; stop real deployments."
        )
    input_type = None
    if "--input-type" in deployments[0]:
        index = deployments[0].index("--input-type") + 1
        if index < len(deployments[0]):
            input_type = deployments[0][index]
    if input_type != "zmq_manager":
        raise RuntimeError("Simulation deployment must use --input-type zmq_manager")
    if "--zmq-host" in deployments[0]:
        index = deployments[0].index("--zmq-host") + 1
        if index >= len(deployments[0]) or deployments[0][index] not in ("localhost", "127.0.0.1"):
            raise RuntimeError("Simulation replay requires deployment --zmq-host localhost")
    return deployments[0]
