Saved on 19 October 2026 before comparing against newer upstream code.

Paired branch: `codex/gr00t-n17-rtc`. Paired annotated tag: `gr00t-n17-rtc-v1-rc1`.

- Client/controller: https://github.com/edgarvk/decoupledWholeBodyControl
- Server: https://github.com/edgarvk/Isaac-GR00T-RTC (private)
- Client upstream base: `b042411` from NVlabs/GR00T-WholeBodyControl.
- Server upstream base: `d2b7e75` from NVIDIA/Isaac-GR00T.
- Model: `nvidia/GR00T-N1.7-3B`, revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- Embodiment: `REAL_G1`; server port 5550; simulated camera port 5559.

This is a recovery baseline, not a validated hardware release. Earlier robot trials
showed a response to the simple “raise your arms” prompt. General manipulation
success has not been established. A damaging shutdown incident occurred; the
C++ stop transition remains unresolved. Do not interpret successful unit tests
or simulated motion as hardware safety validation. The private server repository
preserves the incident report and hashed source snapshots.

The snapshot includes temporal camera history, measured wrist FK, stationary
arm/hand/waist target conversion, absolute-action RTC prefixes, variable-length
prefix normalization using per-step checkpoint statistics, and separate simulation
camera ports with startup-failure detection. Navigation and base-height commands
remain disabled for REAL_G1. The default MuJoCo scene contains no task objects,
and its hand model carries an instability warning.

Checks at save time: 38 focused Isaac CPU tests, 24 client/adapter/RTC tests and
2 isolated camera-startup tests passed. Isaac pre-commit passed with historical
incident source copies excluded from reformatting. No new hardware run or full
3B closed-loop simulation was performed for this snapshot.

The two repositories must be restored together at the tag above: the RTC wire
interface is a local extension. A newer upstream server or client is not assumed
to implement it. A version mismatch remains a hypothesis, not an established cause.

Model weights, installed virtual environments, recordings, generated logs and
machine-specific binaries are not part of these Git snapshots. Cached weights
remain local; controller model hashes are in the adjacent JSON file. Full package
version lists, Git history bundles and uncommitted-file archives are retained at
`/home/mobicare/Isaac-GR00T/local_backups/2026-10-19-n17-rtc/` on the original machine.

The original upstream Git history is retained. The private server code repository
is not a mirror of upstream LFS storage. To restore on another machine without
asking GitHub for missing LFS binaries, clone with smudging disabled and obtain
unmodified LFS assets from the original upstream:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch gr00t-n17-rtc-v1-rc1 <saved-repository-url> <directory>
cd <directory>
git remote add upstream <original-upstream-url>
git -c lfs.url=<original-upstream-url>/info/lfs lfs pull
```

Use `https://github.com/NVIDIA/Isaac-GR00T.git` as the server upstream and
`https://github.com/NVlabs/GR00T-WholeBodyControl.git` as the client upstream.
The inference dependency currently uses the original machine's local Isaac path;
reinstall/repoint it to the paired server checkout if using a different location.

To compare newer upstream code, use separate checkouts/worktrees and fresh
virtual environments. Preserve these baseline directories and environments.
Do not reset these branches, move these tags, or update these environments for
the experiment.
