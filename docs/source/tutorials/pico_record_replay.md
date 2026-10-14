# Record PICO demonstrations and replay SONIC in simulation

Capture the headset and both controllers once, then reuse the same input while
changing target mapping or the policy. Recording does not require a robot,
Pinocchio model, policy checkpoint, body tracking or ankle trackers. It never
opens a robot-command socket.

The command is `python -m gear_sonic.scripts.pico_demo`, run from the repository
root in the existing `.venv_teleop`. It offers `record`, `inspect`, `replay`,
`preview`, and `report`. Files in the root `recordings/` and `logs/` directories
are ignored by Git. Keep recordings as experiment data separately from source
code and transfer them to the simulation PC as ordinary files.

## What the recording contains

One JSONL metadata row, one input row per poll, and an end row. `.jsonl.gz` uses
gzip compression. Each input row keeps the complete **atomic direct-device SDK
snapshot**: headset/controller poses, original source timestamp, receive age,
buttons, trigger/grip values, joystick axes/clicks, application state, and any
explicit tracking-validity fields supplied by the headset client. It also records
local elapsed time, wall-clock receive time and validation/acquisition errors.
Position units are metres; raw quaternion order is **xyzw**. Coordinate axes are
XRT +X right, +Y up, -Z forward. Device absence and invalid data remain observable.

Duplicate source timestamps and invalid/stale snapshots are retained. The SDK
offers the latest snapshot, not a packet subscription: polling at 100 Hz is not
a guarantee that every vendor packet or very short button press is captured.
Hold each start/stop or reanchor button combination for at least half a second.
Inspect reports sample counts, source-clock resets, maximum polling gap,
controller validity and start/stop button edges. Missing optical-validity flags
cannot be inferred from a plausible stationary pose.

The metadata records Git commit and dirty-worktree status. Commit candidate code
before experiments when you need an unambiguous source revision. Existing target
diagnostic logs are not raw demonstrations: they lack sufficient headset and
button/calibration data to reconstruct the original input completely.

## Capture demonstrations on the headset service PC

Use the existing XRoboToolkit binding exposing `get_controller_snapshot`.
For an installation already running arms-only-v2-rc1 it needs no rebuild.
In PICO enable **Head**, **Controller**, and **Send**, disable Body/Motion tracking,
and disable the “Switch w/ A Button” shortcut. Connect to the XRoboToolkit service
PC as in live teleoperation. The recorder is the only SDK client for this session;
do not run the live manager alongside it.

```bash
cd "$HOME/GR00T-WholeBodyControl"
source .venv_teleop/bin/activate

python -m gear_sonic.scripts.pico_demo record \
    --output recordings/reach-and-return.jsonl.gz \
    --description "Slow arm raises, wrist rotations, return to neutral" \
    --start-service
```

Omit `--start-service` when the service is already running. The default service
script is `/opt/apps/roboticsservice/runService.sh`; `--service-script` can change
it. `--duration 120` ends recording after two minutes; otherwise Ctrl-C ends it.
Existing recording files are never overwritten. An uncompressed recording with
a torn final JSON line can be recovered up to its last complete input row. A
truncated gzip stream may need external recovery and is not guaranteed readable.

For each clip:

1. Stand still, look horizontally forward and hold a comfortable neutral
   controller pose for two seconds.
2. Press **A+B+X+Y** for half a second, then release. This marks calibration/start
   for replay; it does not command any robot during recording.
3. Perform the movement slowly. Include stationary holds of 2–3 seconds at useful
   poses, and return to the same neutral controller pose several times.
4. Press **A+B+X+Y** again to mark the end, release, and stop recording with Ctrl-C.

Start with separate clips for axis translations, wrist rotations, and arm
raises/overhead sweeps. Exercise one arm at a time, then both. Include trigger
motions, menu holds and stick-click reanchors in a separate controls clip.
Keep operator feet still and avoid changing the headset tracking origin during
a clip. Human reach is not automatically within the G1's reachable workspace;
the simulator is where these requests are evaluated.

```bash
python -m gear_sonic.scripts.pico_demo inspect \
    --input recordings/reach-and-return.jsonl.gz
```

Expect valid controller samples and two start/stop edges for a single episode.
Use one start/stop episode per clip: replay exits at its first recorded stop.
Copy the recording to the simulation PC if that is a different machine; no
headset or running XRoboToolkit service is needed on the simulation PC.

## Execute the learned policy in MuJoCo

Use the existing Linux simulation/deployment environments and SONIC checkpoint.
This workflow calls the same `ArmsOnlySession`, mapping, filter, FK, hand solver,
planner encoder and C++ policy deployment as live teleoperation. It is not a
kinematic animation pretending to be a policy rollout.

**Terminal 1: simulator**

```bash
cd "$HOME/GR00T-WholeBodyControl"
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

**Terminal 2: SONIC deployment, explicitly in simulation**

```bash
cd "$HOME/GR00T-WholeBodyControl/gear_sonic_deploy"
./deploy.sh \
    --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq_manager --output-type zmq \
    --motor-kp-scale 4,10=1.5 --motor-kd-scale 4,10=1.5 \
    sim
```

The deployment rebuilds automatically. The arms-only-v2-rc1 C++ telemetry changes
are required, even though the recorder/replayer itself adds no C++ changes.
Wait for initialization/`WAIT_FOR_CONTROL` and fresh simulated low-state data.

**Terminal 3: replay**

```bash
cd "$HOME/GR00T-WholeBodyControl"
source .venv_teleop/bin/activate

python -m gear_sonic.scripts.pico_demo replay \
    --input recordings/reach-and-return.jsonl.gz \
    --mapping filtered \
    --log logs/reach-filtered.jsonl
```

Replay checks that exactly one local `g1_deploy_onnx_ref` is running on `lo`/`lo0`
with `--input-type zmq_manager`. It refuses real/unknown network interfaces and
multiple deployments, checks again while running, binds commands only to loopback,
and uses local feedback on port 5557. Stop real deployments before using it.
Commands use port 5556; the live manager must not occupy it.

The recording clock starts only after fresh measured simulation joints arrive
and publisher warmup. The recorded start edge calibrates controller motion to
**measured simulated** wrist/torso FK, not to invented zero joints. Calibration
events record the initial measured joints, headset/controllers, frame transform,
grip offsets and target anchors. Feedback freshness uses wall time; tracking
freshness uses the virtual recording clock. Source timestamps, resets and tracking
gaps survive replay. No pose interpolation hides missing input.

Recorded menu holds, left-stick reanchors and trigger/grip actions are honored.
Joystick axes are recorded but cannot select locomotion in arms-only. Data
collection/abort gestures are suppressed during replay. Recorded stop, end of
file and Ctrl-C send the existing deployment stop; ZMQ delivery is best effort,
so the deployment's independent **O** stop remains available.

If a clip has no start marker, explicitly use `--start-mode auto`: it calibrates
at the first usable headset/controller sample, ignores recorded ABXY start/stop,
and runs to end of file. Menu/reanchor controls still apply. This changes the
calibration point, so use the same start mode for comparisons.

`--speed 0.5` slows playback; `--speed 2` speeds it up. The default output rate is
50 Hz (`--target-fps`). Replay samples the latest recorded snapshot at each output
tick, just as live teleop samples the latest SDK packet. Very short events can be
missed at lower output rates. Speed changes alter the policy's dynamic task and
the number of mapper updates: use **speed 1 / 50 Hz** for the first comparisons.

## Compare direct mapping with the current filter

Stop and restart the simulator and deployment before each comparison, using the
same initial pose, checkpoint, gains, output rate and playback speed. The log
records the input file SHA256, recorded/replay code versions, command/checkpoint
arguments, mode and calibration. Different simulation initial poses/calibration
produce different targets even with identical controller input. Real-time DDS,
ZMQ scheduling and policy dynamics are not guaranteed bitwise deterministic.

```bash
python -m gear_sonic.scripts.pico_demo replay \
    --input recordings/reach-and-return.jsonl.gz \
    --mapping direct \
    --log logs/reach-direct.jsonl
```

`filtered` is the existing current live mapper, including its workspace projection
and reference absorption. `direct` is a **simulation-only comparison reference**:
fixed calibration with no workspace/IK projection. Tracking freshness, source
jumps, menu holds and stop controls remain enforced. It reproduces the raw
mapping intent, not every historical arms-only-v1 behavior (in particular not
the workspace freeze). These tools do not change live mapping behavior or fix
the previously diagnosed reference drift; they make it reproducible and measurable.

Each run creates a new result filename; replay refuses to append to existing
results. Result logs include requested/outgoing targets, measured simulated
wrists/joints/velocities/torque estimates, projection limits, hold reasons,
source and playback timing, and calibration events. Simulated temperatures are
not a hardware thermal validation.

```bash
python -m gear_sonic.scripts.pico_demo report --input logs/reach-filtered.jsonl
python -m gear_sonic.scripts.pico_demo report --input logs/reach-direct.jsonl
```

Compare stationary-return accuracy, projection error and actual-wrist position/
orientation error separately. Report percentiles include motion lag and feedback
timing; they are not automatically steady-state accuracy measurements. Human
demonstrations are input tests, not ground-truth robot joint trajectories or
training labels.

## Offline target preview without simulation or policy inference

For fast mapper checks, supply a JSON array of 29 finite initial motor-order joint
angles in radians. Use the measured initial joints from a calibration event in a
previous simulation result, not a zero fallback:

```bash
python - <<'PY'
import json
from pathlib import Path
for line in Path("logs/reach-filtered.jsonl").read_text().splitlines():
    row = json.loads(line)
    if row.get("event") == "calibration" and row["action"] == "start":
        Path("recordings/initial-sim-joints.json").write_text(json.dumps(row["measured_body_q"]))
        break
else:
    raise SystemExit("No simulation start calibration found")
PY

python -m gear_sonic.scripts.pico_demo preview \
    --input recordings/reach-and-return.jsonl.gz \
    --initial-joints recordings/initial-sim-joints.json \
    --mapping direct --output logs/reach-direct-preview.jsonl
```

Preview processes every captured poll in recorded order using the same mapping
classes and a fixed explicit calibration seed. It writes a distinct preview
schema and never publishes commands or executes SONIC. It therefore cannot
measure actual robot tracking, balance, contacts or thermal behavior. It is useful
for testing coordinate mapping and return-to-neutral behavior before the learned
policy replay.
