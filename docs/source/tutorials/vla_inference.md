# VLA Inference

This guide covers running a trained Isaac-GR00T VLA policy on the Unitree G1 robot
using the Sonic whole-body control stack.

## Overview

The inference pipeline consists of:

1. **Isaac-GR00T PolicyServer** — loads the VLA model and serves actions over ZMQ
2. **VLA inference client** (`run_vla_inference.py`) — reads camera + robot state,
   queries the PolicyServer, and publishes actions to the C++ control loop
3. **C++ deploy** (`gear_sonic_deploy`) — executes whole-body control on the robot
4. **Camera server** — provides camera images over ZMQ (runs as a systemd service)
5. **Data exporter** (optional) — records episodes during inference

```
┌──────────────────────┐
│  Isaac-GR00T         │
│  PolicyServer        │
│  (GPU machine)       │
└──────┬───────────────┘
       │ ZMQ REQ/REP
       ▼
┌─────────────────────┐    ZMQ TCP    ┌──────────────────────┐
│  VLA Inference      │ ◄─────────── │  Camera Server       │
│  (run_vla_inference)│              │  (on robot)          │
└────┬───────────┬────┘              └──────────────────────┘
     │           │
     │ ZMQ PUB   │ ZMQ SUB
     │ (actions) │ (state)
     ▼           ▼
┌─────────────────────┐
│  C++ Deploy         │
│  (gear_sonic_deploy)│
└─────────────────────┘
```

## Prerequisites

### 1. Isaac-GR00T PolicyServer

The PolicyServer runs on a machine with a GPU. It loads your finetuned VLA model
and serves inference over ZMQ.

Install [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) and start the server:

```bash
# On the GPU machine (from the Isaac-GR00T repo)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/your/finetuned_model \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

### Base model with REAL_G1 (stationary upper-body control)

The base `nvidia/GR00T-N1.7-3B` checkpoint supports `REAL_G1`. This is a
different observation/action schema from a finetuned `UNITREE_G1_SONIC` policy:
it needs temporal camera history and wrist poses, and returns absolute joint
targets after the server decodes the actions.

Start the **base model** server in the Isaac-GR00T repository:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag REAL_G1 \
    --device cuda:0 \
    --port 5550
```

Then launch the client from GR00T-WholeBodyControl:

```bash
python gear_sonic/scripts/launch_inference.py \
    --embodiment-tag REAL_G1 \
    --policy-port 5550 \
    --camera-host 192.168.123.164 \
    --observation-rate 50 \
    --prompt "raise your arms"
```

The client queries the server's modality configuration at startup. The default
`--embodiment-tag auto` also detects either schema; specifying `REAL_G1` checks
that the server uses the expected format. It derives the action horizon from
the server instead of assuming a fixed chunk size.

Camera capture continues while inference runs. Video indices `[-20, 0]` select
images separated by 0.4 seconds at `--observation-rate 50`; this rate defines
the checkpoint's temporal sampling interval and is independent of inference
frequency. Match it to the dataset sampling rate if using another checkpoint.
Startup waits for enough actual camera history. Wrist poses are computed from
the measured robot configuration in the pelvis frame, using GR00T's rotation
6D convention (the first two rows of the rotation matrix).

For `REAL_G1`, the client publishes **stationary planner commands**: IDLE mode,
zero movement, fixed facing, and absolute arm, hand and waist targets in the
controller's joint order. Navigation and base-height predictions are omitted.
The SONIC planner/controller manages standing balance; the VLA does not directly
command leg motors. This adapter uses the model's joint predictions rather than
its redundant wrist-pose action predictions.

The client starts paused. In the keyboard pane, `k` starts the C++ controller,
`i` holds the measured upper-body posture while staying in PLANNER mode, and
`p` enables publishing the predicted targets. For a latent SONIC policy, `i`
retains the existing initial-token and POSE-mode behavior.

The client checks tensor shapes, finite values and joint limits, and suspends
publishing when camera or state data is stale. Offline tests validate FK,
temporal sampling and the controller packet format; tracking and task success
still need evaluation in simulation and on the target robot.

### 2. Inference Environment

On the inference machine (can be the same as the PolicyServer or a separate PC):

```bash
bash install_scripts/install_inference.sh
```

This creates `.venv_inference` with the Isaac-GR00T PolicyClient and all
inference dependencies.

### 3. Camera Server

The camera server should be running as a systemd service on the robot.
See [Data Collection](data_collection.md) for camera server setup.

### 4. C++ Deploy

The `gear_sonic_deploy` binary must be built. See the main README.

### SONIC v1.1 Checkpoint

Use the `sonic_v1_1/` checkpoint when the VLA policy was trained against
the robot-heading-normalized SONIC controller. It uses a 10-frame SMPL/wrist
reference horizon and was trained with wrist-pose augmentation. It is not the
low-latency checkpoint.

```bash
python download_from_hf.py --sonic-v1-1
```

Launch the matching C++ controller:

```bash
cd gear_sonic_deploy
./deploy.sh \
    --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq_manager \
    --motor-kp-scale 4,10=1.5 \
    --motor-kd-scale 4,10=1.5 \
    real
```

This v1.1 tuning scales the left and right ankle-pitch motors (hardware
indices `4` and `10`). The resulting whole-body stability improves observed
wrist tracking; it does not directly scale the wrist motors.

Or pass the same model pair to the Python launcher:

```bash
python gear_sonic/scripts/launch_inference.py \
    --deploy-checkpoint policy/sonic_v1_1/model \
    --deploy-obs-config policy/sonic_v1_1/observation_config.yaml \
    --deploy-motor-kp-scale 4,10=1.5 \
    --deploy-motor-kd-scale 4,10=1.5 \
    --camera-host 192.168.123.164 \
    --prompt "pick up the cup"
```

### Low-Latency Teleoperation Checkpoint

The `low_latency/` checkpoint is configured for responsive whole-body
teleoperation. Its SMPL encoder uses 4 future reference frames at 50 Hz
(approximately 80 ms of reference lookahead), compared with 10 frames
(approximately 200 ms) in the default release. This is reference lookahead,
not total end-to-end system latency.

Download the deployment files from Hugging Face:

```bash
python download_from_hf.py --low-latency
```

Then launch `gear_sonic_deploy` with the low-latency model prefix and matching
observation config:

**C++ deploy:**

```bash
cd gear_sonic_deploy
./deploy.sh \
    --cp policy/low_latency/model \
    --obs-config policy/low_latency/observation_config.yaml \
    --input-type zmq_manager \
    real
```

For simulation, replace `real` with `sim`. The `--cp` value is a model prefix:
`deploy.sh` appends `_encoder.onnx` and `_decoder.onnx` internally.

**Python launcher:**

```bash
python gear_sonic/scripts/launch_inference.py \
    --deploy-checkpoint policy/low_latency/model \
    --deploy-obs-config policy/low_latency/observation_config.yaml \
    --camera-host 192.168.123.164 \
    --prompt "pick up the cup"
```

The Python launcher starts the same C++ deploy command in a tmux pane, then runs
the Python VLA inference client, keyboard publisher, and optional data exporter.

## Real-Time Chunking (RTC)

Restart the PolicyServer using the matching updated Isaac-GR00T checkout, then
enable RTC in the client. For the base model and stationary REAL_G1 adapter:

```bash
python gear_sonic/scripts/launch_inference.py \
    --embodiment-tag REAL_G1 \
    --policy-port 5550 \
    --camera-host 192.168.123.164 \
    --prompt "raise your arms" \
    --rtc
```

The client first checks the server's RTC capability/version. Enabling the flag
against an older server fails explicitly rather than silently running ordinary
asynchronous inference. The same scheduler also supports latent SONIC action
schemas; a compatible checkpoint is still required for those schemas.

The initial prediction starts at step zero after it arrives. Later requests
include the remaining absolute actions of the current chunk, including all
configured model output groups. The server converts them to its normalized
representation relative to the new measured state, freezes committed steps
and conditions the soft-overlap region using the native action-head primitive.
REAL_G1 joint limits are applied before forming the next prefix, so RTC sees
the same bounded motor targets that are sent to the controller. The server
restores frozen physical targets exactly after bfloat16 decoding.

The client anchors each response to its request's control step. While waiting,
it continues executing the previous trajectory. On arrival it skips exactly
the steps already published, and validates the server acknowledgement and
frozen targets. Prompt changes, pause/resume, initialization, stop/start and
sensor discontinuities invalidate older responses. A single worker serializes
policy requests. Exhausted chunks hold the last posture instead of being
replayed.

Defaults for the 40-step base checkpoint at 50 Hz:

| Option | Default | Meaning |
|---|---:|---|
| `--rtc-overlap-steps` | 20 | Request another chunk when this many actions remain. |
| `--rtc-initial-delay` | 0.2 | Initial round-trip delay estimate in seconds. |
| `--rtc-safety-steps` | 2 | Extra frozen steps for control/transport jitter. |
| `--rtc-ramp-rate` | 5.0 | Native exponential soft-overlap ramp strength. |

The freeze estimate adapts to the maximum of the last eight conditioned-call
latencies; the overlap grows if needed, bounded by the model horizon. Startup
model warm-up is not used as a steady-state delay estimate. Logs report
`RTC chunk: latency=..., overlap=..., frozen=..., overrun=... steps`.
Repeated overruns or holds indicate that the available prediction horizon and
inference throughput cannot sustain the configured control rate. RTC does not
make an arbitrarily slow model real-time.

Leaving out `--rtc` retains ordinary asynchronous inference. The REAL_G1
stationary restriction remains unchanged: navigation and height predictions
are not sent to the controller. Offline tests cover relative-state prefix
conversion, frozen-step preservation, delayed execution, resets and packet
mapping; physical tracking and task understanding still require robot tests.

## Action Space

The Sonic embodiment (`unitree_g1_sonic`) uses a 78-dimensional action
space: 64-dim motion token + 7-dim left hand joints + 7-dim right hand joints.

## Quick Start — tmux Launcher

The easiest way to run inference is with the all-in-one tmux launcher:

```bash
# Real robot
python gear_sonic/scripts/launch_inference.py \
    --prompt "pick up the apple" \
    --camera-host 192.168.123.164

# Simulation: base model + REAL_G1 + RTC (local MuJoCo camera)
python gear_sonic/scripts/launch_inference.py --sim \
    --embodiment-tag REAL_G1 --policy-port 5550 \
    --camera-host 127.0.0.1 --prompt "raise your arms" --rtc

# Without data recording
python gear_sonic/scripts/launch_inference.py \
    --no-data-exporter \
    --prompt "pick up the apple"
```

Simulation uses camera port **5559** by default; hardware retains **5555**.
An explicit `--camera-port` is forwarded to the simulator, VLA client and exporter.
If a camera port is occupied, the simulator reports the publishing failure instead
of silently continuing without images. Restart the simulator after fixing it.
The default scene has no manipulation objects; start with posture prompts.

The launcher creates a tmux session with four panes:

| Pane | Component | Description |
|------|-----------|-------------|
| 0 (top-left) | C++ Deploy | Whole-body controller |
| 1 (bottom-left) | Keyboard Publisher | Type keyboard commands here |
| 2 (top-right) | VLA Inference | Policy client + action loop |
| 3 (bottom-right) | Data Exporter | Records episodes (optional) |

### Keyboard Controls

Type these keys in the **Keyboard Publisher** pane (pane 1):

| Key | Action |
|-----|--------|
| `k` | Start / stop the C++ control loop |
| `i` | Blend smoothly to initial pose and switch to POSE mode |
| `p` | Pause / resume policy inference |
| `[` | Toggle left hand open/closed (initial pose) |
| `]` | Toggle right hand open/closed (initial pose) |
| `t <text>` | Change the inference prompt (e.g., `t pick up the cup`) |
| `c` | Start recording an episode (data exporter) |
| `s` | Stop recording — success (data exporter) |
| `f` | Stop recording — failure / discard (data exporter) |

### Typical Workflow

1. Wait for all panes to initialize
2. Click on **pane 0** (C++ Deploy) and press Enter to confirm deployment
3. Switch to **pane 1** (Keyboard Publisher)
4. Press `k` to start the C++ control loop (starts in PLANNER mode)
5. Press `i` to blend to the initial pose (switches to POSE mode)
   > The robot smoothly interpolates to the initial pose over 1 second. If your
   > task starts from a different pose than the default, see
   > [Customizing the Initial Pose](#customizing-the-initial-pose) below.
6. Press `p` to unpause the inference loop
7. The robot will begin executing VLA-predicted actions
8. Press `p` to pause, `k` to stop the control loop when done

## Manual Setup (Without tmux)

If you prefer to run each component in separate terminals:

### Terminal 1 — Isaac-GR00T PolicyServer (GPU machine)

```bash
# From the Isaac-GR00T repo
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/your/finetuned_model \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

### Terminal 2 — C++ Deploy

```bash
cd gear_sonic_deploy
./deploy.sh --input-type zmq_manager real
```

Low-latency variant:

```bash
python gear_sonic/scripts/launch_inference.py \
    --deploy-checkpoint policy/low_latency/model \
    --deploy-obs-config policy/low_latency/observation_config.yaml \
    --camera-host 192.168.123.164 \
    --prompt "pick up the apple"
```

Manual C++ deploy equivalent:

```bash
cd gear_sonic_deploy
./deploy.sh \
    --cp policy/low_latency/model \
    --obs-config policy/low_latency/observation_config.yaml \
    --input-type zmq_manager \
    real
```

### Terminal 3 — VLA Inference

```bash
source .venv_inference/bin/activate
python gear_sonic/scripts/run_vla_inference.py \
    --host <policy_server_ip> \
    --port 5550 \
    --embodiment-tag unitree_g1_sonic \
    --prompt "pick up the apple" \
    --camera-host 192.168.123.164
```

### Terminal 4 — Data Exporter (optional)

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the apple" \
    --camera-host 192.168.123.164
```

## Configuration Reference

### VLA Inference (`run_vla_inference.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | PolicyServer host |
| `--port` | `5550` | PolicyServer port |
| `--embodiment-tag` | `unitree_g1_sonic` | Embodiment tag |
| `--prompt` | `demo` | Language prompt |
| `--action-publish-rate` | `50` | Action publish rate (Hz) |
| `--action-horizon` | `40` | Actions per inference chunk |
| `--rate` | `2.5` | Inference rate (Hz) |
| `--camera-host` | `localhost` | Camera server host |
| `--camera-port` | `5555` | Camera server port |
| `--initial-pose-blend-duration` | `1.0` | Seconds to blend to initial pose (0 = instant snap) |
| `--verbose-timing` | `false` | Always print loop timing |

### tmux Launcher (`launch_inference.py`)

The launcher exposes all the above flags plus deploy and data exporter options.
Use `--deploy-motor-kp-scale` and `--deploy-motor-kd-scale` to forward hardware
gain specifications to the C++ controller.
Run `python gear_sonic/scripts/launch_inference.py --help` for the full list.

## Remote PolicyServer

When running the PolicyServer on a separate GPU machine:

```bash
# On the inference machine, point to the remote server
python gear_sonic/scripts/launch_inference.py \
    --policy-host <gpu_machine_ip> \
    --policy-port 5550 \
    --camera-host 192.168.123.164 \
    --prompt "pick up the apple"
```

Make sure port 5550 (or your chosen port) is accessible between the two machines.

## Latency Compensation

The inference loop automatically compensates for network and compute latency.
When a new action chunk arrives, the system calculates how many actions in the
chunk are already "stale" based on the time elapsed since inference started,
and skips to the appropriate action index. This is controlled by `--action-publish-rate`
and `--action-horizon`.

## Customizing the Initial Pose

When you press `i`, the inference client blends the robot smoothly from its
current configuration to a predefined **initial pose** encoded as a 64-dim
latent motion token. This pose should match the starting configuration your
demonstrations typically begin from.

### When to Change the Initial Pose

You should update the initial motion token if:

- Your collected demonstrations start from a pose far from the default
  (e.g., arms raised, holding an object, or a different standing stance)
- You switch to a different SONIC checkpoint (each checkpoint has its own
  latent space — the same token produces different poses across checkpoints)
- The robot is snapping to a dangerous or unstable configuration on `i` press

### Where to Change It

Edit `gear_sonic/utils/inference/initial_poses.py`:

```python
LATENT_INITIAL_MOTION_TOKEN = np.array(
    [
        # Replace with your 64-dim token
        ...
    ],
    dtype=np.float32,
)
```

### How to Find a Good Token

1. **From data collection:** Look at the first action frame of a good demonstration
   episode. The `action.motion_token` column in the parquet file at `frame_index=0`
   gives you the latent token for that pose.

2. **From the C++ deploy:** Put the robot in the desired starting pose via teleop,
   then read the most recent latent token published on the ZMQ action channel.

### Blend Duration

The blend duration controls how quickly the robot transitions to the initial pose:

```bash
# Default: 1 second smooth blend
python gear_sonic/scripts/run_vla_inference.py --initial-pose-blend-duration 1.0

# Faster blend (0.5 seconds)
python gear_sonic/scripts/run_vla_inference.py --initial-pose-blend-duration 0.5

# Instant snap (no interpolation, legacy behavior)
python gear_sonic/scripts/run_vla_inference.py --initial-pose-blend-duration 0
```

```{warning}
Setting `--initial-pose-blend-duration` too low (or to 0) can cause jerky motion,
especially if the robot's current pose is far from the initial pose. The default
1-second blend is safe for most configurations.
```
