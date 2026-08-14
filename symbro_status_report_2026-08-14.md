# symbro — status report, 2026-08-14

This supersedes the 2026-08-13 report (which the user deleted from git tracking in commit `8e4d0a6`, along with the 2026-08-11 report — both were working documents, not permanent repo fixtures). This version was built from facts re-checked directly against the repo on 2026-08-14 (git log, git status, git diff --stat, file line counts, `.symbro/` and scratch-dir contents) rather than carried over from memory, per the standing lesson from the previous report cycle: a specific, checkable claim (the sbatch `set -u` fix) was wrong once already, so nothing below is asserted without a fresh check.

## 1. What symbro is

A CLI/pipeline for symmetry-broken protein cage design: RCSB query → download → geometry analysis → isolate ring subunits → RFdiffusion → ProteinMPNN → structure-prediction screening → validated designs. Developed against a local Windows checkout (`C:\Users\youruser\4_Projects\symbro`) with real execution on an HPC cluster (`your-hpc-cluster`, SLURM).

## 2. Pipeline stage status

| # | Stage | CLI command | Status |
|---|---|---|---|
| 1 | Query RCSB | `symbro query` | Wired, verified |
| 2 | Download assemblies | `symbro download` | Wired, verified |
| 3 | Geometry analysis | `symbro geometry` | Wired, verified |
| 4 | Isolate ring subunits | `symbro isolate` | Wired, verified |
| 5 | RFdiffusion | `symbro rfdiffusion` + `symbro status` | **Newly wired this cycle.** Verified via real synthetic execution (CLI-level and pipeline-level). Not yet run for real on the local checkout or against the live cluster through the new CLI path — only through the pre-CLI standalone test scripts (`test_rfdiffusion_hpc.py`), which were the proven reference this integration was modeled on. |
| 6 | ProteinMPNN | `symbro pmpnn` | **Newly wired this cycle.** Same verification status as above — synthetic-execution verified, not yet exercised through the new CLI path on real HPC. |
| 7 | Structure-prediction screening | — | Library code exists (`run_predictor.py`) but is unverified and has no CLI command. Still the next real milestone once stages 5–6 are exercised for real. |

Housekeeping: `symbro clean` (new this cycle) clears `.symbro/` checkpoints and all three scratch directories (`temporary_files/`, `temporary_subunits/`, `temporary_simulations/`) between runs, with `--keep-state`/`--keep-downloads`/`--keep-subunits`/`--keep-simulations`/`--dry-run` flags.

## 3. What changed since the 08-13 report

### 3.1 RFdiffusion + ProteinMPNN CLI integration (previously "decided but not implemented" — now done)

Added to `pipeline.py`:
- `run_rfdiffusion()` — batch-per-assembly (one job per assembly, not per ring row, not one giant job). Submits all jobs before polling any of them. Blocks and polls by default; `detach=True` returns immediately but is SLURM-only (raises `ValueError` for local/singularity backends, since those hold a live, non-optional `subprocess.Popen` that can't be resumed after the process exits). Saves `.symbro/rfdiffusion.{pkl,csv}`.
- `refresh_rfdiffusion_status()` / `run_status()` — single-pass re-poll of any non-terminal rows, used to resume after a detached submission. Backs `symbro status`.
- `run_pmpnn()` — batch-per-assembly, always blocks (ProteinMPNN has no SLURM backend — confirmed in its own module docstring: "LOCAL ONLY, on purpose"). Filters candidate designs by `top_n`/`min_plddt`/`select`, skips (does not abort on) assemblies with no usable designs, consistent with the project's fail-soft convention. Saves `.symbro/pmpnn.{pkl,csv}`.

Added to `cli.py`: `symbro rfdiffusion`, `symbro status`, `symbro pmpnn` commands wired to the above, with curated preview tables (assembly_id/symmetry_type/state/etc.) rather than dumping raw dataclass reprs — this was a real bug caught during testing (the first pass flooded the table with `RFdiffusionRun` repr text) and fixed before delivery.

Verified via two independent real-execution test suites (not just read-through):
- `test_cli_integration.py` — direct `pipeline.*` calls: block-and-poll, detach+status resume, detach+non-slurm `ValueError`, multi-assembly pmpnn, `select=` validation. All passed.
- `test_cli_layer.py` — `CliRunner`-driven end-to-end `symbro rfdiffusion`/`symbro pmpnn` invocations against monkeypatched I/O boundaries (`rfdiffusion.submit`/`poll_status`, `pmpnn.submit`/`poll_status`, `config.load_installation_config`), using real dataclasses and real code paths otherwise. All passed.

### 3.2 `symbro clean` (previously "`.symbro/` has no cleanup path" — now done)

Added `pmpnn.clear_simulations_dir()` (ProteinMPNN previously had no equivalent to `download.py`'s `clear_temp_dir()` / `isolate.py`'s `clear_temp_subunits_dir()` — this was a genuine gap, now closed) and `pipeline.clean()`, exposed as `symbro clean`. Default behavior clears everything (all six `.symbro/` checkpoints + all three scratch dirs) on the reasoning that partial clearing leaves dangling filepath references between stages; `--keep-*` flags narrow scope; `--dry-run` previews without deleting.

Verified via `test_clean.py`, four scenarios: dry-run (nothing deleted), full clean (everything gone, folders survive), re-running on an already-clean tree ("Nothing to clear."), `--keep-state` (scratch cleared, checkpoints untouched). All passed.

### 3.3 Confirmed still-fixed: cross-machine/cross-OS path portability

`paths.py`'s `_is_absolute_anywhere()` (using `ntpath.isabs()`/`posixpath.isabs()` instead of OS-native `os.path.isabs()`) and the relative-to-CWD checkpoint storage fix are both present on disk and were re-verified again this cycle by direct file inspection — no regression.

### 3.4 Confirmed still-fixed: sbatch `set -u` crash

This is the fix that was falsely reported as done in an earlier draft, then genuinely applied and corrected in the 08-13 report. Re-checked again this cycle: still present in `rfdiffusion.py`'s `_render_sbatch_script()` (wraps `setup_lines` in `set +u` / `set -u`). No regression.

## 4. Git state (freshly checked, 2026-08-14)

```
8e4d0a6  RF and Pmpnn HPC Validation, several bugs resolved. Next commit should
         integrate CLI for rfd and pmpnn.               <- user's own commit
995591b  Fix RFdiffusion chain-fusion residue overflow and sbatch nounset
         crash; add design ranking
41b4828  Fix cross-machine checkpoint path portability; surface resolved
         state dir in CLI
```

`8e4d0a6` is the user's own commit, made directly on the machine (not through me). It deleted both `symbro_status_report_*.md` files from git tracking (and from disk — confirmed, they no longer exist in the working tree). This is expected/fine: those were working handoff documents, not permanent repo content.

**Uncommitted work (all of section 3.1 and 3.2):**

```
 src/toolkit/cli.py      | 204 ++++++++++++++++++++++-
 src/toolkit/pipeline.py | 434 +++++++++++++++++++++++++++++++++++++++++++++++-
 src/toolkit/pmpnn.py    |  18 ++
 3 files changed, 650 insertions(+), 6 deletions(-)
```

None of the RFdiffusion/ProteinMPNN CLI integration or the `symbro clean` command is in git yet. This is the single most important actionable item in this report — it's a full week-equivalent of newly-verified feature work sitting only on disk.

**Standing git-lock caveat (recurring, not one-time):** every git command run through this device bridge leaves a stale `.git/index.lock` that the bridge cannot clean up itself (`rm`/`unlink` returns "Operation not permitted" under this bridge's permission model), because the bridge lacks delete permission on files it doesn't own the lifecycle of. This was mistaken once for a one-off issue; it is not — it recurs on every future git invocation made this way. It does not appear to block read-only commands (`git status`, `git diff --stat` both completed successfully with a stale lock present), but will block `git commit`. Workaround each time: move the lock into a `_to_delete/` folder before committing. The only real fix is running git directly on the machine, outside the bridge — which the user has already demonstrated is available and working (commit `8e4d0a6` was made that way). A stray `_to_delete/` folder now exists at the repo root holding old lock files; safe to delete manually.

## 5. Local checkout vs. cluster divergence (flagged previously, still unaddressed, still low-urgency)

`.symbro/candidates.pkl`, `downloaded.pkl`, `geometry.pkl`, `rings.pkl` on the local Windows checkout are all frozen at an Aug 11 run — timestamps confirm none have been touched since. They predate the path-portability fix (section 3.3) and, per earlier direct inspection of the raw pickle bytes, still carry stale absolute Windows paths internally. The real, portability-fixed validation of stages 1–4 happened on the cluster-side checkout, not this one.

This has not caused a failure and has not blocked any work — stages 5–7 development has proceeded independently using synthetic fixtures rather than these checkpoints. It remains a cheap fix whenever it becomes relevant: re-run `symbro download` → `symbro isolate` locally (or `symbro clean` first, then the full stage 1–4 sequence) to regenerate current, portable checkpoints. No local `.symbro/rfdiffusion.pkl` or `.symbro/pmpnn.pkl` exists yet — those stages have never been run against this local checkout at all, only via the standalone HPC test scripts and the synthetic test suites above.

## 6. What's built but genuinely unverified

- **Structure prediction (stage 7)**: `run_predictor.py` exists as library code with no CLI command and no real-execution test. This is the next real milestone.
- **RFdiffusion/ProteinMPNN CLI commands against live infrastructure**: verified synthetically (real code paths, monkeypatched I/O boundary only) but not yet run through `symbro rfdiffusion`/`symbro pmpnn` against the actual local installation or the actual cluster. The pre-CLI standalone scripts (`test_rfdiffusion_hpc.py`, `test_pmpnn_hpc.py`) did validate the underlying `rfdiffusion.py`/`pmpnn.py` modules for real on HPC — the new CLI layer sits on top of that proven core, but the layer itself hasn't had a live run.

## 7. Priority-ordered next steps

1. **Commit the uncommitted CLI integration + clean command work** (section 4) — this is a full week-equivalent of verified feature work with zero git history right now. Do this directly on the machine to sidestep the lock issue, or push through the bridge with the lock-relocation workaround.
2. **Run `symbro rfdiffusion` → `symbro status` → `symbro pmpnn` for real** against the local installation and/or cluster, closing the gap flagged in section 6.
3. Optionally regenerate local `.symbro/` checkpoints (section 5) via `symbro clean` + a fresh stage 1–4 run, so the local checkout matches the portability-fixed state validated on the cluster.
4. Begin stage 7 (structure-prediction screening) integration — CLI command + real-execution verification, following the same pattern used for stages 5–6.
