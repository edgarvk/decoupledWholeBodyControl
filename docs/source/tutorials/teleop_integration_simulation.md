# Test the integrated teleoperation branch in simulation

Run this tutorial on the Linux simulation PC with the existing Python
environment, deployment dependencies, G1 assets, and SONIC checkpoint installed.
It tests live teleoperation and recorded input through MuJoCo and SONIC.
GR00T N1.7 inference requires a separate test with its paired server.

The examples use your personal repository at `~/decoupledWholeBodyControl`,
with its Python environment, dependencies, and policy assets already available.
This is not a fresh-install tutorial. Run all three terminals on the same PC.
Stop existing simulator and controller processes before starting. These commands
do not switch branches or modify files in a separate official repository checkout.

## 1. Select the integration branch

```bash
cd "$HOME/decoupledWholeBodyControl"
git status --short
```

If local changes appear, save them before switching branches. Otherwise:

```bash
git remote get-url decoupled >/dev/null 2>&1 ||
    git remote add decoupled https://github.com/edgarvk/decoupledWholeBodyControl.git

git remote get-url decoupled
git fetch decoupled
```

Verify that the remote URL points to `edgarvk/decoupledWholeBodyControl`.
If this is your first checkout of the integration branch:

```bash
git switch -c codex/teleop-integration --track decoupled/codex/teleop-integration
```

If the local branch already exists:

```bash
git switch codex/teleop-integration
git pull --ff-only decoupled codex/teleop-integration
```

Check the selected revision:

```bash
git log -1 --oneline
```

Use the branch for the current tutorial. The `enhanced` gripper profile name was
introduced in commit `1ec3f1d`, after the `teleop-integration-v1-rc1` tag.

## 2. Terminal 1: start MuJoCo

```bash
cd "$HOME/decoupledWholeBodyControl/gear_sonic_deploy"
source scripts/setup_env.sh

cd ..
source .venv_teleop/bin/activate
python -m gear_sonic.scripts.run_sim_loop
```

Keep this terminal running. The default scene is the tracked `scene_43dof.xml`.

## 3. Terminal 2: start the SONIC controller

```bash
cd "$HOME/decoupledWholeBodyControl/gear_sonic_deploy"

./deploy.sh \
    --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq_manager \
    --output-type zmq \
    --motor-kp-scale 4,10=1.5 \
    --motor-kd-scale 4,10=1.5 \
    sim
```

The script rebuilds automatically. Confirm that the displayed environment is
simulation and the network interface is `lo`, accept the prompt, and wait for
`Init done` / `WAIT_FOR_CONTROL`. The `--output-type zmq` option supplies the
measured feedback required for arms-only calibration.

The integration branch selects the bundled Unitree DDS libraries automatically.
If startup fails with a DDS library assertion, inspect resolution in another
terminal:

```bash
cd "$HOME/decoupledWholeBodyControl/gear_sonic_deploy"
source scripts/setup_env.sh
ldd target/release/g1_deploy_onnx_ref | grep -E 'libddsc|libddscxx'
```

Both libraries should resolve to the bundled Unitree directory rather than a
mixture of ROS and Unitree libraries. See the
[integration guide](teleop_integration.md) for the DDS checks and limitations.

## 4. Terminal 3: choose one command source

Run either live teleoperation or replay, one at a time.

### Live headset control

In the PICO application, enable **Head**, **Controller**, and **Send**. Disable
Body/Motion tracking and the “Switch w/ A Button” shortcut. Connect to this PC's
XRoboToolkit service as in your existing setup. The manager launches the service
through `/opt/apps/roboticsservice/runService.sh`.

```bash
cd "$HOME/decoupledWholeBodyControl"
source .venv_teleop/bin/activate

python -m gear_sonic.scripts.pico_manager_thread_server \
    --manager --arms-only \
    --mirror --gripper-profile enhanced \
    --arms-only-timing
```

Hold a comfortable neutral controller pose, look forward, and press
**A+B+X+Y** together to calibrate and start. Release the buttons before moving.
Press the same combination again to stop. The controller's independent **O**
stop is available in Terminal 2.

For ordinary arm routing, omit `--mirror`. For the original gripper behavior,
replace `--gripper-profile enhanced` with `--gripper-profile legacy`.

### Replay a recording

No headset or XRoboToolkit service is needed for replay. Replace the input path
with your recording filename:

```bash
cd "$HOME/decoupledWholeBodyControl"
source .venv_teleop/bin/activate

python -m gear_sonic.scripts.pico_demo replay \
    --input recordings/reach-and-return.jsonl.gz \
    --mapping filtered \
    --mirror --gripper-profile enhanced \
    --log "logs/mirrored-$(date +%Y%m%d-%H%M%S).jsonl"
```

Replay uses recorded start/stop button edges. If a clip has no start marker,
explicitly add `--start-mode auto`; use the same start mode for comparisons.
The default speed is 1 and output rate is 50 Hz. Replay exits at the first
recorded stop or end of file; Ctrl-C also attempts to stop the controller.

To compare with the unfiltered mapping, replace `--mapping filtered` with
`--mapping direct`. This is a simulation comparison mode; it does not change
the live mapper. See [recording and replay](pico_record_replay.md) for the full
workflow and log reports.

## 5. Compare results

Stop all three processes and restart the simulator and controller before each
comparison. Keep the initial posture, checkpoint, gains, recording, speed, and
start mode equal. Change one option at a time.

Check arm routing, wrist orientation, gripper gestures, return-to-neutral drift,
and movement near the head. For live control, also check calibration, hold and
reanchor, start/stop, and tracking loss. The existing posture filter can still
limit reach and absorb reference motion; integration does not resolve that drift.

Passing simulation does not validate physical motor temperatures, contacts, or
hardware shutdown behavior. The unresolved shutdown incident remains a hardware
validation blocker. See the [integration validation gates](teleop_integration.md#validation-gates).
