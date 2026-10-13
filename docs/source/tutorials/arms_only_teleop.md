# PICO arms-only teleoperation

`--manager --arms-only` maps direct PICO controller motion into SONIC's existing
VR_3PT / TELEOP encoder. No ankle trackers, body-joint packets, SMPL reconstruction,
or body calibration are used. SONIC still commands the whole robot, including
legs and waist for balance. This is stationary manipulation intent, not a guarantee
that the robot's feet, torso, or pelvis will remain motionless.

## Comfort-filter update after the first hardware trials

This revision replaces the latched 60 cm workspace freeze with a recoverable
projection and enables a G1 kinematic target filter by default in `--arms-only`.
Tracking loss/menu hold/source jumps still latch a hold and preserve stop controls.
The filter changes the Cartesian wrist targets supplied to SONIC, never sends joint
or torque commands, and never selects a different encoder.

For an installation that already built the previous arms-only version, copy the
updated Python files (including the **new** `arm_target_filter.py`) and the three
updated C++ deployment files, then run:

```bash
cd "$HOME/GR00T-WholeBodyControl/gear_sonic_deploy"
source scripts/setup_env.sh
just build
```

The existing teleop environment already contains NumPy, SciPy, and Pinocchio. This
revision does not change the XRoboToolkit binding and does not require rebuilding
it again. Rebuild the C++ deployment to enable velocity, torque-estimate, and
winding/driver temperature feedback, available before and during policy engagement.

Run the same deployment command from your earlier trial. For the first replay,
replace its final `real` with `sim` and start the MuJoCo simulator. In the streamer
terminal, from the original repository root, capture both JSONL and terminal logs:

```bash
cd "$HOME/GR00T-WholeBodyControl"
source .venv_teleop/bin/activate
mkdir -p logs
trial="$(date +%Y%m%d-%H%M%S)"
python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager --arms-only --input-source xrt --target_fps 50 \
    --zmq_feedback_host localhost --zmq_feedback_port 5557 \
    --arms-only-log "logs/arms-only-$trial.jsonl" \
    --arms-only-timing 2>&1 | tee "logs/arms-only-$trial.txt"
```

Repeat small axis translations and rotations, then slowly raise and lower one arm
through the limiting region. Confirm that `held` remains false for geometric
limiting and inward motion responds without a stick click. Verify the other arm,
head-only motion, reanchor, menu hold, lost tracking, and both stop controls. Compare
requested and outgoing wrists with actual measured FK, shoulder/elbow angles,
joint margins, torque estimates, and temperatures. End a trial on overheating;
a target hold does not unload the motors. This implementation does not add an
automatic thermal recovery or override the existing deployment stop semantics.

The JSONL log appends a metadata record per run and a target record per output
sample. It contains incoming and accepted source timestamps, source receive and
send times, requested/outgoing/actual wrists, position/rotation errors, hold/limit
reasons, measured joints and velocities, estimated motor torques, both temperature
readings, and projected/measured kinematic metrics. Missing/stale measured data is
recorded as absent rather than zero. Default terminal diagnostics update once per
second, including during holds. `processing_ms` covers input acquisition through
send, including IK; JSONL analysis/FK/write work follows send and can affect the next
cycle. Inspect consecutive `sent_mono` values for total observed scheduling delay.

## Setup and simulation startup

Use the existing Linux teleop and deployment environments from
[VR setup](../getting_started/vr_teleop_setup.md) and
[deployment installation](../getting_started/installation_deploy.md).
Rebuild **both** the bundled Python SDK and C++ deployment after this change.
The new SDK snapshot API and pre-start measured-joint feedback are required;
older binaries are not sufficient.

From the repository root, in the teleop environment:

```bash
source .venv_teleop/bin/activate
python -m pip install pybind11 cmake setuptools
cd external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64
python setup.py build_ext --inplace --force
python -m pip install . --no-build-isolation --force-reinstall
cd ../..
python -c 'import xrobotoolkit_sdk as x; assert callable(x.get_controller_snapshot)'
```

On PICO, enable **Head**, **Controller**, and data **Send**; set motion tracking
mode to **None** and disable **Body** and **Hand** tracking. Disable the application's
“Switch w/ A Button” sending shortcut so A remains available to the manager.
Connect XRoboToolkit to the service PC. No tracker pairing or body calibration is
needed. These switches are described in the
[upstream Unity client](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client).

Use three terminals for the first trial:

```bash
# Terminal 1, repository root: simulated G1 only
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

```bash
# Terminal 2, repository root: rebuild, then deploy to simulation explicitly
cd gear_sonic_deploy
source scripts/setup_env.sh
just build
./deploy.sh --input-type zmq_manager --output-type zmq sim
# Wait for Init done / WAIT_FOR_CONTROL.
```

```bash
# Terminal 3, repository root
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager --arms-only --input-source xrt --target_fps 50 \
    --zmq_feedback_host localhost --zmq_feedback_port 5557 \
    --arms-only-timing
```

If the streamer and deployment run on different computers, use
`--zmq_feedback_host <deployment-IP>` in terminal 3 and
`--zmq-host <streamer-IP>` in terminal 2. Ports remain 5556 (commands) and
5557 (feedback). `--output-type zmq` enables the required `g1_calibration` topic.
The manager launches the existing `/opt/apps/roboticsservice/runService.sh`.
No real-robot activation is part of this setup or the automated tests.

## Calibration and controls

1. Wait for fresh measured joints from deployment. Look horizontally in the
   intended forward direction and hold both controllers comfortably still.
   A prescribed body/skeleton pose is unnecessary.
2. Press **A+B+X+Y** together. Initial yaw alignment is captured from the headset;
   controller poses are anchored to FK of the **measured** 29 robot joints.
   The measured torso key point is captured at the same time. Targets are
   published before the start command. Missing/stale feedback or tracking refuses
   start with a diagnostic; no default/zero joint fallback is used.
3. Release the buttons, then move either controller. Head movement is ignored
   after the initial calibration. Joysticks and A+X/B+Y/A+B/X+Y mode combinations
   cannot select locomotion or body-tracking modes in this path.
4. **Left menu** latches a target hold. Reposition the controllers while held,
   release menu, and press **left stick click** to reanchor and resume. That click
   also recalibrates during streaming. It requires fresh measured FK but preserves
   the exact last outgoing targets, including the torso; it does not snap to the
   measured wrists when the policy has tracking error. The positional target/FK
   residual is logged. Operator yaw stays fixed until the manager is restarted.
5. **A+B+X+Y** again sends the existing stop command and exits. **Ctrl-C** and
   unexpected runtime errors also attempt a stop after engagement. Deployment's
   independent **O** stop remains available. A disconnected headset cannot deliver
   button events; use the deployment stop or Ctrl-C. ZMQ PUB stop delivery is best
   effort, as in the existing workflow.

Dex3 uses the existing trigger/grip-to-hand solver unchanged (the current finger
mapping is trigger based). During a tracking/menu hold, the last hand commands
are retained too. Normal deployment hand closure/delta limits remain in effect.
A hold keeps SONIC running with stationary intent; a stop has the existing
deployment stop semantics and must not be confused with a balance-preserving hold.

Without `--arms-only`, existing whole-body, planner, frozen-upper-body and VR_3PT
modes remain available. Arms-only rejects Isaac Teleop, SMPL/waist visualization,
standalone skeleton visualizers, and SMPL recording flags instead of silently
entering those paths. `manager_state` data-collection/abort controls remain available.

## Frames and targets

Direct XRoboToolkit tracking poses are metres and **xyzw** quaternions in a
right-handed frame (+X right, +Y up, -Z forward), following the
[upstream direct-pose conversion](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python/blob/main/xrobotoolkit_teleop/utils/geometry.py).
This differs from the reconstructed body-joint path and its `OFFSETS`.
The fixed basis conversion is:

```text
robot XYZ = [-tracking Z, -tracking X, tracking Y]
B = [[0,0,-1], [-1,0,0], [0,1,0]]
A = yaw_alignment_at_start * B
p_target = p_anchor + A * ((p_controller + R_controller * grip_offset) - p_initial_point)
R_target = (A * R_controller * R_initial_controller^-1 * A^-1) * R_anchor
```

Robot axes are +X forward, +Y left, +Z up. The calibration-derived orientation
relationship aligns each controller grip with its measured robot wrist frame;
no reconstructed-skeleton wrist rotation offsets are reused. Quaternions on the
planner wire are **wxyz**, ordered left wrist, right wrist, torso. Translation
scale is 1:1. The fixed operator frame does not follow the head or robot motion.
Targets are **root-relative**, not fixed world-space constraints on the robot.

`--left-grip-offset X Y Z` and `--right-grip-offset X Y Z` specify a physical
controller-local lever arm in metres. Both default to zero: the grip centre
commands the robot manipulation key point. They are not robot hand dimensions.
Changing these offsets changes the translational effect of controller rotation;
measure and validate any nonzero value in simulation.

The existing robot key points are:

| Target | Robot link | Local offset (m) |
|---|---|---|
| Left wrist | `left_wrist_yaw_link` | `[0.18, -0.025, 0]` |
| Right wrist | `right_wrist_yaw_link` | `[0.18, 0.025, 0]` |
| “Neck/head” | `torso_link` | `[0, 0, 0.35]` |

The third quaternion is the torso link orientation. It is **not** the headset pose
or an independent neck joint. All three references are computed with the existing
G1 Pinocchio model and shared key-frame offsets. The fixed third target is compatible
with the existing TELEOP observation contract; task precision and stationary balance
still depend on the learned policy and its operating range.

## Tracking loss, limits, and known API limitation

The native SDK now exposes an atomic snapshot of the latest Head/Controller packet,
its source timestamp and local receive age. A missing controller cannot reuse a
cached pose from an earlier packet. There is no Python pose queue or smoothing.
At 50 Hz the manager samples the latest packet every 20 ms. A stopped source timestamp,
receive age over 250 ms, invalid/non-finite pose, non-unit quaternion, lost application
focus, or backwards source clock latches a hold. Tracking recovery alone does not
resume movement; left stick click explicitly reanchors. Source-clock resets require reanchor. If XR
recentering changes the tracking axes, stop and restart the manager to capture a new
heading alignment; reanchor alone does not change the original alignment.

A raw **controller** step above 10 cm / 45 degrees still latches hold, with the
side, measured jump, and threshold logged explicitly. This is computed against the
previous source sample, so a limited output cannot create a false tracking jump.

The 60 cm radius about each initial measured wrist is now projected independently.
Rejected translation/rotation is absorbed into the controller-to-target anchor, so
blocked commands do not accumulate. Reversing the controller immediately moves the
target inward even after a long outward motion. Consequently, returning a controller
to its original physical position need not restore the original robot target after
limiting; use the existing reanchor control to reposition the operator's hands.
The operator axes and torso reference remain fixed.

The default kinematic projection uses the real G1 URDF, the existing manipulation
point offsets and Jacobians, and at most six damped IK iterations per arm. Wrist
position has greater weight than orientation, allowing difficult orientations to
relax. The redundant direction prefers lowered elbows and a comfortable bent-arm
reference. Solver joint bounds reserve 0.08 rad from existing limits and prefer at
least 0.25 rad elbow flexion. The projected elbow stays at most 2 cm above the
shoulder; an initially elevated/out-of-envelope posture is grandfathered only while
moving toward the envelope. Calibration itself does not snap the arms to a default
pose. Projected per-sample changes are capped at 2 cm and 5 degrees. These caps are
per sample, so the effective velocity cap depends on `--target_fps`.

`--arms-max-gravity-nm` defaults to **8 Nm per arm joint** in the projection. This is
an experimental **unloaded URDF comfort budget**, not a manufacturer continuous
rating or a thermal guarantee. It excludes payloads, contacts and dynamics. An
initial gravity estimate above the budget can be retained only without increasing
that joint's estimate. The actual motor torque estimate is logged separately.

The IK joint solution is internal feasibility evidence: **SONIC receives only
wrist/torso targets and can choose a different elbow configuration.** Measured
elbow height, joint-limit/comfort margins, gravity estimates, and smallest positional
Jacobian singular values are logged to reveal that difference. Neither elbow-down
posture on the actual robot, collision avoidance, global IK optimality, nor prevention
of overheating is guaranteed. IK introduces bounded computation and target projection,
so measure its processing cost and any resulting target correction/response delay.

Fresh measured robot feedback is now required throughout active filtered teleop.
Missing/stale feedback latches hold; fresh data and explicit reanchor are needed to
resume. Geometric limiting remains recoverable without reanchor.

**Stock XRoboToolkit does not publish controller optical tracking validity.**
The checked [Unity acquisition source](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/main/Assets/Scripts/TrackingData.cs)
publishes predicted controller poses and buttons without `isTracked`/`trackingState`.
The SDK's hand `isActive` belongs to optical finger tracking and cannot validate a
controller. Consequently, a device that keeps sending a plausible frozen or predicted
pose with advancing timestamps cannot be distinguished from a stationary controller.
Do not regard the implemented freshness checks as proof of optical tracking.

The smallest upstream extension is to emit controller `isTracked` and
`trackingState` from Unity `CommonUsages.isTracked` / `CommonUsages.trackingState`
(position bit 1, rotation bit 2). The atomic snapshot preserves those fields and the
reader already rejects false/incomplete values when present. This extension and
occlusion/reconnection/recentering trials remain necessary to validate optical-loss
handling for hardware use. The mode does not fabricate such validity flags.

Robot calibration feedback is a separate `g1_calibration` msgpack topic on the
existing ZMQ output port. Deployment publishes raw motor joint angles, velocities,
torque estimates and winding/driver temperatures in hardware/MuJoCo order after
initialization, before and during policy execution, only if the
low-state sample is at most 50 ms old. Python requires finite 29-joint data and a
receive age at most 250 ms. It never substitutes a zero vector. Local receive age
cannot reveal an arbitrarily delayed network packet; inspect transport timing on
the actual setup. The original `g1_debug` stream is still used for trial analysis.

## First simulation measurements

Use small independent translations along each axis and rotations about each grip;
then test head-only movement, menu hold, reanchor while the policy lags, loss of one
controller, disabled Send, clock/reset events, and both stop paths. Confirm left/right
mapping, physical axis signs, grip offsets, fixed torso targets, and absence of
intentional locomotion. Real headset coordinate/API versions must be checked here.

With `--arms-only-timing`, record `source_ns` (the packet that produced the held or
current target), `read_mono` (estimated local packet receive time), `sent_mono`,
`sent_unix_ns`, and `hold`. Record the `planner` wire targets and the deployment
`g1_debug` measured joints with receive timestamps. Recompute actual wrist poses
from measured joints using the same FK offsets. Compare orientations by relative
rotation angle and positions in the same root frame. The debug visualization's
`vr_3point_position` is rotated by the reference root quaternion; undo that rotation
before comparing it to the raw planner target.

Measure separately:

- **Command latency:** tracking/source packet to outgoing target, then deployment
  receive/observation time. Source timestamp is a device packet timestamp, not
  necessarily the camera exposure time. Synchronize clocks or estimate clock offset
  before subtracting device and PC timestamps. Monotonic clocks from different
  computers cannot be directly subtracted. Exclude hold rows from movement latency.
- **Local processing/queue delay:** packet receive to send and source/target timestamp
  increments. Expect up to one 20 ms manager sampling period at 50 Hz, plus SDK,
  network and deployment scheduling. The Python reader and robot-feedback subscriber
  retain only the latest sample; the unchanged C++ planner subscriber has HWM 3 and
  does not conflate. Inspect its observed backlog rather than assuming zero latency.
- **Robot response lag:** outgoing/received target changes to measured wrist changes,
  including policy inference, control cycle, compliance, actuator dynamics and load.
  A fast command stream does not establish fast or accurate physical tracking.

The sender log is local send completion, not a deployment acknowledgment. Instrument
`OnPlannerReceived` / policy observation sampling for end-to-end command timing.
The startup/reanchor continuity guarantee concerns outgoing targets, not zero physical
response lag or exact policy tracking. The deployment's existing initialization ramp
and policy transition behavior remain in place.

## Verification and implementation trace

```bash
python -m pytest gear_sonic/tests/test_arms_only.py gear_sonic/tests/test_arm_target_filter.py gear_sonic/tests/test_xrt_snapshot.py -q
```

The tests replay packets with no Body/Hand/ankle API, exercise the real manager loop
with fake I/O, decode the planner wire format, verify mapping/hold/stop/reanchor,
use the real G1 URDF for FK when Pinocchio is installed, and compile the native
snapshot binding against a **fake** PC service when pybind11 and a C++ compiler are
available. An isolated C++ replay also verifies that the actual pre-start publication
block emits only fresh measured joints, without requiring a start command. Tests
never link to a robot SDK service or engage motors.

The legacy dependency chain is `_init_input_source` → body availability wait →
`PicoReader` body joints → `_process_3pt_pose` pelvis/wrists/neck → `ThreePointPose`
calibration (initially zero joints) → `PlannerStreamer` → planner message.
POSE and visualization additionally perform full SMPL processing; mode transitions
can return to those paths. The opt-in CLI selects `run_arms_only_manager` before any
of those acquisitions or streamers are instantiated. It uses `PicoControllerReader`
→ `ArmsOnlyTargets` → `G1ArmTargetFilter` → the unchanged planner message builder. Deployment's
`ZMQCommandManager::OnPlannerReceived` still sets VR buffers, selects encoder 1,
and feeds `GatherVR3PointPosition/Orientation` to the same whole-body policy.

The added read-only calibration output resolves the pre-start feedback dependency;
no policy weights, locomotion controller split, arm motor controller, or leg disabling
is introduced. Full learned-policy simulation requires the Linux deployment build,
model checkpoints and simulator environment. Synthetic/FK/native-binding tests alone
do not validate balance, contact behavior, optical-loss detection, or manipulation
accuracy on physical hardware.
