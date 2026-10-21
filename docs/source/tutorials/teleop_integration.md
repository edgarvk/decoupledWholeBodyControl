# Teleoperation integration candidate

This branch combines the saved feature branches while preserving their histories.
It is an integration candidate, not a validated hardware release. `rc1` means
release candidate 1; further fixes produce rc2, rc3, and so on.

## Source inventory

| Source | Saved commit | Contribution |
|---|---|---|
| `main` / `arms-only-v1` | `56a05ffa571ea88b2fd50675b013ad290318c7be` / `7a3747a69bd02ec0b086dbc7ac4871034b40744f` | Direct arms-only PICO mapping and measured-FK calibration |
| `arms-only-kinematics` / `arms-only-v2-rc1` | `0300f9627806b0215b2a00cd68e5a2f578fc9f02` | Posture/load projection and measured diagnostics |
| `codex/pico-record-replay` / `pico-record-replay-v1` | `4b5a5b6bf2129378be4b102249cc69bf57df718b` | Raw recording, preview and simulation replay |
| `codex/gr00t-n17-rtc` / `gr00t-n17-rtc-v1-rc1` | `e2da2be` | GR00T N1.7 client adapter, RTC, camera startup, saved inference configuration |
| `teleoperacion_espejo+pinzav1.0` / `teleop_esp+fuerzadedosv1.0` | `b9d703d` | Mirrored teleoperation and alternate gripper joint targets |

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
