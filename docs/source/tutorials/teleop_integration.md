# Teleoperation integration candidate

This branch combines the saved feature branches while preserving their histories.
It is an integration candidate, not a validated hardware release. `rc1` means
release candidate 1; further fixes produce rc2, rc3, and so on.

For launch commands and comparisons, follow the
[step-by-step simulation tutorial](teleop_integration_simulation.md).

## Source inventory

| Source | Saved commit | Contribution |
|---|---|---|
| `main` / `arms-only-v1` | `56a05ffa571ea88b2fd50675b013ad290318c7be` / `7a3747a69bd02ec0b086dbc7ac4871034b40744f` | Direct arms-only PICO mapping and measured-FK calibration |
| `arms-only-kinematics` / `arms-only-v2-rc1` | `0300f9627806b0215b2a00cd68e5a2f578fc9f02` | Posture/load projection and measured diagnostics |
| `codex/pico-record-replay` / `pico-record-replay-v1` | `4b5a5b6bf2129378be4b102249cc69bf57df718b` | Raw recording, preview and simulation replay |
| `codex/gr00t-n17-rtc` / `gr00t-n17-rtc-v1-rc1` | `e2da2be384ed9fc6b402c7cb833bbf3a678c7ab8` | GR00T N1.7 client adapter, RTC, camera startup, saved inference configuration |
| `teleoperacion_espejo+pinzav1.0` / `teleop_esp+fuerzadedosv1.0` | `b9d703d132f3ec6a62b4df562d0790844270bf52` | Mirrored teleoperation and alternate gripper joint targets |

The recorder branch already contains main and kinematics. The GR00T snapshot
branches from older upstream history but carries copies of the recorder and mirror
code. Merge results must therefore be reviewed by content as well as ancestry.

GR00T uses the paired private server repository
`https://github.com/edgarvk/Isaac-GR00T-RTC`, tag `gr00t-n17-rtc-v1-rc1`.
See [its saved baseline](vla_n17_rtc_baseline.md) for the model revision and
installation details. The unresolved shutdown/stop incident remains a hardware
validation blocker; combining sources does not resolve it.

Each controller mode has one active command source. GR00T autonomous inference
and live teleoperation must not publish commands concurrently. Historical tags
remain available for exact baseline comparisons.

## Selecting features

The integration branch starts from `pico-record-replay-v1`; its default live
teleoperation is unchanged: ordinary routing and the `legacy` hand profile.
Kinematic filtering still has the previously diagnosed reference-absorption drift.
This integration preserves that baseline for comparison rather than silently
changing its control behavior.

```bash
# Direct controllers, current posture filtering, original hand behavior:
python -m gear_sonic.scripts.pico_manager_thread_server --manager --arms-only

# Mirrored direct controllers and the enhanced gripper profile:
python -m gear_sonic.scripts.pico_manager_thread_server \
    --manager --arms-only --mirror --gripper-profile enhanced

# The same mirror option also supports the existing body/SMPL manager:
python -m gear_sonic.scripts.pico_manager_thread_server \
    --manager --mirror --gripper-profile enhanced --vis_vr3pt --vis_smpl
```

The old `pico_manager_thread_server_mirror.py` command remains a compatibility
wrapper: it selects `--mirror --gripper-profile enhanced` on the shared manager.
The profiles are joint position presets and gesture mappings, not a change to
motor torque limits or measured grip force:

- `legacy`: original recorder/teleop presets and trigger-based middle closure.
- `rtc`: the saved GR00T branch's presets; inference initialization selects these.
- `enhanced`: larger closure presets and index/ring/middle selection
  from trigger/grip/both. Existing downstream hand limits still apply.

Mirroring exchanges the operator arms and hand inputs. Physical menu, stick-click,
start/stop and collection buttons keep their roles. Direct-controller mirroring
acts on robot-frame controller deltas before calibration: the measured robot
wrists remain the starting targets even if the robot posture is asymmetric.
Robot Y is reflected, preserving forward motion and height. The historical
body/SMPL path retains the legacy X-reflection convention; do not reuse
that transform as though the two input coordinate systems were interchangeable.
Both paths use proper rotation conjugation, not an improper rotation quaternion.

Raw recordings remain unchanged. Mirror routing and hand profile are selected
when replaying, so one recording can evaluate several configurations:

```bash
python -m gear_sonic.scripts.pico_demo replay \
    --input recordings/demo.jsonl.gz --mapping filtered \
    --mirror --gripper-profile enhanced --log logs/demo-mirrored-enhanced.jsonl
```

The same options are accepted by `preview`; preview does not execute the policy
or actuate hands. Replay metadata and calibration events record the options.
Reset simulation between comparisons and use equal initial joints, speed, gains
and checkpoints. See [record/replay](pico_record_replay.md).

## DDS environment and simulation

`setup_env.sh` now selects the bundled CycloneDDS C/C++ pair after sourcing ROS,
CUDA and TensorRT. `deploy.sh` rechecks this under error propagation. The helper
is idempotent, resolves its deployment root from its own location, supports the
bundled x86_64/aarch64 directories, and refuses missing/LFS-pointer libraries.
It does not change global ROS installations or shell startup files.

On the Linux deployment PC:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
ldd target/release/g1_deploy_onnx_ref | grep -E 'libddsc|libddscxx'
```

Both libraries must resolve to `$UNITREE_DDS_LIBRARY_DIR`. Run the isolated
loopback DDS publishing test on Linux before policy rollout. Shell path tests
alone are not DDS ABI/runtime validation. The normal `deploy.sh ... sim` command
can be used; it no longer requires a manual export afterward. Re-source the
setup script if another environment has subsequently changed library precedence.

The default scene is the tracked `scene_43dof.xml`, not the absent lab-local
`scene_43dof_with_table.xml`. A task scene can be explicitly selected:

```bash
python -m gear_sonic.scripts.run_sim_loop --robot-scene /path/to/task_scene.xml
```

DDS initialization is owned by BaseSimulator rather than repeated in its wrapper.
The saved camera transport
startup checks and simulation camera port 5559 are retained; the normal onscreen
simulation remains enabled by default.

## Paired inference environment

The inference extra pins the private Isaac-GR00T-RTC server source to commit
`5ca56e7cc4f995da6d3bc0b3a97002aeb8e294ca` (the paired saved tag), replacing
the lab PC's absolute `file:///home/mobicare/Isaac-GR00T` dependency. Installing
this extra needs access to the private repository. Use a separate inference
environment; preserve the lab's working baseline environments for recovery.

The existing REAL_G1/SONIC schema adapter, temporal camera history, RTC tests,
stationary REAL_G1 routing and model-hash manifest are retained. No policy weights
or paired server source are copied into this repository.

## Validation gates

1. CPU/unit regression tests for mapping, mirror axes/rotations, profiles, recorded
   input, RTC, camera transport, calibration, hold and stop messages.
2. Linux build, actual resolved DDS libraries and loopback publishing.
3. MuJoCo replay in ordinary/mirror modes, with equal initial conditions; check
   defaults, boundaries, holds, reanchors, stop/reset and hand directions.
4. Paired GR00T closed-loop simulation, including startup/pause/stop/reset and
   stale camera/state handling.
5. Hardware validation only after the unresolved shutdown transition is understood.

The initial source-integration tests can run on macOS, but cannot establish Linux
DDS compatibility, full policy behavior, thermal behavior, contacts or hardware
shutdown safety. Keep the integration tag at release-candidate status until the
remaining gates have been completed.

## Initial integration verification (21 October 2026)

The focused macOS CPU suite covers the saved adapter/RTC/client and camera tests,
existing arms-only/filter/recording/native-snapshot tests, and the new integration
tests. It also verifies mirrored preview with the real G1 model, asymmetric
calibration continuity, return to neutral, raw-file integrity, historical gripper
presets, and DDS path ordering after repeated ROS-like environment changes.
RPC transport and the native XR service are faked where appropriate; no policy
server, physical robot or Linux DDS runtime is used by these tests.

The next required lab checks are the actual Linux DDS library/publishing check,
full MuJoCo policy replay and paired N1.7 inference. The existing target-filter
drift and shutdown incident remain separate unresolved behavior issues.

Result: **107 focused tests passed** on the final integration source tree. To
repeat them in a Python environment with the client/test dependencies and G1 assets:

```bash
python -m pytest -q \
    gear_sonic/tests/test_teleop_integration.py \
    gear_sonic/tests/test_gr00t_adapter.py \
    gear_sonic/tests/test_rtc.py \
    gear_sonic/tests/test_rtc_runner.py \
    gear_sonic/tests/test_sim_camera_startup.py \
    gear_sonic/tests/test_arms_only.py \
    gear_sonic/tests/test_arm_target_filter.py \
    gear_sonic/tests/test_xrt_snapshot.py \
    gear_sonic/tests/test_pico_demo.py
```
