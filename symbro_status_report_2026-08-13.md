# SymBro — Status Report (2026-08-13, corrected)

Prepared as a handoff snapshot for continuing this project in a new conversation. Supersedes `symbro_status_report_2026-08-11.md` (still in the repo root, worth keeping for history) — this one folds in everything that changed in the two days since.

**Correction note, read before anything else:** the first version of this report claimed the sbatch `set -u`/conda-activation fix (below) had already landed in `rfdiffusion.py`. The user independently checked this report line-by-line against the actual repo and found that claim was false — `_render_sbatch_script()` still executed `setup_lines` unguarded under `nounset`. Exactly why that edit didn't make it into the committed file isn't fully reconstructable (most likely it was described as done without the corresponding `Edit`/commit actually landing, possibly lost across a session boundary) — rather than guess further, it's simply been applied now, verified by generating real sbatch scripts and running them under `bash -n` and actual execution (confirmed: an unset variable inside `setup_lines` no longer crashes the script, and an unset variable in the actual job command still fails loudly afterward, exactly as intended). Every other specific, checkable claim in the original report — `git status`'s exact file list, the hybrid-36 fix's details, the `torch` optional-dependency change, `run_predictor.py`'s placeholder string, `cli.py`'s stub message, the empty test files — held up exactly as written under the same line-by-line check. Three claims (the ProteinMPNN pLDDT/`global_score` results, `test_pmpnn_hpc.py` completing on `your-hpc-cluster`, and the existence of `downloaded.pkl.bak_manual_fix`) are cluster-side and can't be confirmed from this local checkout either way — they're carried through from the user's own direct report of that run earlier in this conversation, not independently verified against the cluster by either party. Take this as the general lesson for continuing from this document: it is well-checked, not infallible — a claim this specific and checkable was still wrong once, so anything here worth relying on for real cluster work is worth a direct check against the code, the same way this correction was found.

## Where things actually stand

SymBro's target pipeline has seven stages: **query → download → geometry → isolate → RFdiffusion → ProteinMPNN → structure-prediction self-consistency screening → validated designs**. The honest, current picture:

| # | Stage | Library code | Wired into `symbro` CLI | Verified on real data/HPC |
|---|---|---|---|---|
| 1 | query | Done | Yes (`symbro query`) | Yes (real RCSB data) |
| 2 | download | Done | Yes (`symbro download`) | Yes |
| 3 | geometry | Done | Yes (`symbro geometry`) | Yes |
| 4 | isolate | Done | Yes (`symbro isolate`) | Yes |
| 5 | RFdiffusion | Done (`rfdiffusion.py`) | **No** — library + standalone test script only | Yes, real GPU job on your-hpc-cluster |
| 6 | ProteinMPNN | Done (`pmpnn.py`) | **No** — library + standalone test script only | Yes, real cluster run, **new since 08-11** |
| 7 | Structure prediction (AlphaFold2 / Boltz / AF3) | Done (`alphafold2.py`, `boltz.py`, `af3.py`, dispatch via `structure_prediction.py`) | **No** | **No — never run against real data** |

Stages 1–4 are a genuinely finished, user-facing CLI. Stages 5–7 are real, working, increasingly HPC-proven *library* code, but none of them has a `symbro` command yet — today you'd have to write a Python script (or use the ad hoc `test_rfdiffusion_hpc.py` / `test_pmpnn_hpc.py`) to run them. That gap — not the science — is the main thing standing between where this is and "a tool a non-technical user could run."

## What changed since the 08-11 report

The 08-11 report closed with RFdiffusion's SLURM path freshly proven on a GPU job, two structural gaps flagged (state-dir cleanup, CWD-dependent path resolution), and ProteinMPNN named as the next real test. Since then:

**A second RFdiffusion bug was found and fixed, one layer deeper than the 08-11 fix.** The 08-11 session fixed RFdiffusion's contig grammar rejecting multi-chain fusion by relabeling fused chains onto one shared ID with disjoint residue-number blocks (`_FUSION_RESIDUE_BLOCK = 10_000`). The first real multi-chain job under that fix (three ~150-residue chains) pushed fused residue numbers past PDB's 9999 ceiling, which gemmi silently hybrid-36-encodes (`10001` → `"A001"`) — a string RFdiffusion's own parser then can't read as an integer. Fixed in `rfdiffusion.py`'s `_fuse_chains_for_contig()` by replacing the fixed 10,000 block with a tight, dynamically-tracked offset, plus an explicit `_MAX_FUSED_RESIDUE_NUMBER = 9999` guard that raises a clear `ValueError` before hybrid-36 overflow could happen silently again. Verified by re-running the actual offset math against a real completed job's contig string, not just by inspection.

**The sbatch `set -u`/conda-activation crash is now a real code fix, not a per-user workaround — as of this correction, not earlier.** 08-11 flagged this as "patched around via your own `installation.yaml`, won't travel to someone else's." An earlier draft of this report claimed `_render_sbatch_script()` already wrapped `setup_lines` in `set +u`/`set -u`; that was false when checked directly against the file (see the correction note at the top). It's applied now: `setup_lines` execution is wrapped in `set +u` before / `set -u` after, so any conda/module activation script that references unset variables doesn't crash the job under nounset mode, while an unset variable in the actual job command still fails loudly. Verified by rendering real scripts via the actual function (not just reading the diff) and running them: a deliberately-unset variable inside `setup_lines` no longer aborts the script; the same variable referenced after `setup_lines` still does, with a clear "unbound variable" error. Until this correction, your own `installation.yaml`'s `setup_lines` workaround was the only thing actually protecting your runs — worth confirming this landed on your machine (it has, verified by re-reading the file fresh after committing, not just trusting the write) before assuming it's covered.

**Design ranking/selection was added to `rfdiffusion.py`.** New `rank_designs(design_paths, top_n=None, min_plddt=None)` reads each design's `.trb`, reduces `plddt`'s final timestep to `mean_plddt`/`min_plddt`, and returns a sorted-best-first DataFrame — this is what lets a user pick which RFdiffusion outputs actually go to ProteinMPNN, rather than running MPNN on everything.

**ProteinMPNN ran successfully against a real RFdiffusion output for the first time.** `test_pmpnn_hpc.py` (new) rebuilds the RFdiffusion job deterministically, ranks its designs, selects via `--top-n`/`--min-plddt`/`--select`, submits ProteinMPNN, and writes ranked sequences. Confirmed running end-to-end on your-hpc-cluster: design pLDDT 0.971, top ProteinMPNN sequences at `global_score` ≈ 1.965–1.972. This is the first time stage 6 has touched real HPC output at all. ProteinMPNN itself was confirmed to need no GPU (device selection falls back to CPU automatically, verified against `dauparas/ProteinMPNN`'s own source) — `torch` is now an optional dependency (`pip install -e ".[proteinmpnn]"`, `pyproject.toml`'s `[project.optional-dependencies]`), not a base one, keeping it out of the way of anyone only running stages 1–4.

**A real, non-cosmetic cross-machine path-portability bug was found, root-caused, and fixed — twice, because the first pass missed an edge case.** This is worth understanding in some depth since it's the kind of bug that will recur if the pattern isn't recognized elsewhere:

- *Root cause:* `download.py` and `isolate.py` both build their default output directories as `os.path.join(os.getcwd(), <name>)` and store the resulting **absolute** path straight into a DataFrame column that gets pickled to `.symbro/downloaded.pkl` / `.symbro/rings.pkl`. An absolute Windows path baked in on the laptop means nothing on the Linux cluster — moving `.symbro/` plus the referenced files over silently breaks the next stage that reads that column. This is exactly what happened to `rings.pkl` (found and hand-patched by the user) and, on the next run, to `downloaded.pkl` (same root cause, one stage upstream).
- *Fix (`src/toolkit/paths.py`, new file):* `to_portable(path)` stores a path relative to the current working directory instead of absolute (falling back to absolute if it genuinely can't be made relative — e.g. a custom dir outside the project, or a different Windows drive); `resolve_path(path)` turns it back into absolute right before actual file I/O, and is a no-op on an already-absolute path for backward compatibility with existing checkpoints. Applied at the two write sites in `download.py`/`isolate.py` and every real file-I/O site (`download_structure()`, `save_structures()`, `isolate._load_single_model()`). Nothing changed in `rfdiffusion.py`/`pmpnn.py` — they already call `os.path.abspath()` on paths before use, so a relative path just resolves correctly against whichever machine's CWD is current, which is the existing convention the whole CLI already relies on (`state_dir`/`data_dir`/`output_dir` all default off `os.getcwd()`).
- *Second bug, found on the very next real test:* `resolve_path()`'s "already absolute → leave alone" check used `os.path.isabs()`, which is OS-native — on Linux it only recognizes a leading `/`, so a **Windows**-absolute path like `C:\Users\...` (baked into `downloaded.pkl` before `paths.py` existed) read as "not absolute" and got silently `os.path.join()`'d onto the Linux CWD, producing a garbage path instead of failing cleanly. Fixed by checking absoluteness under both Windows and POSIX rules (`ntpath.isabs()` or `posixpath.isabs()`) regardless of which OS the process is currently running on. Reproduced the exact failure and confirmed the fix with a synthetic test before shipping.
- *Verified end-to-end*, not just unit-tested: built a synthetic `downloaded.pkl`/`rings.pkl` pair on one directory, relocated the entire tree (`.symbro/` + `temporary_files/` + `temporary_subunits/`) to a different, differently-named root (simulating the Windows→Linux move), and confirmed every real downstream consumer found its files using only the relative paths, with zero manual patching — while an old-style absolute-path row in the same checkpoint kept working unchanged.
- *Scope confirmed, not assumed:* grepped every module for a `filepath` column. Only `download.py` and `isolate.py` ever write one into a checkpoint — `candidates.pkl` (raw RCSB metadata, no filepath) and `geometry.pkl` (traced through `rings.py`'s fixed output-column tuple and `orientation.py`'s `subset.copy()` pattern — neither ever carries one) are **not** at risk. The user's manually hand-patched `downloaded.pkl.bak_manual_fix` backup is fine to keep as a safety net but nothing further needs checking upstream.

**`cli.py` now echoes the resolved working directory and state directory** at the top of every subcommand's output (`Working from: <cwd>  (state dir: <resolved path>)`, silent on bare `--help`/`--version`). This is a partial answer to the 08-11 report's second flagged gap ("state-directory resolution is silently CWD-dependent... a `symbro status`/`symbro where` command... would make it visible") — it makes the existing CWD assumption checkable at a glance (useful in a cluster batch-job log with no one watching interactively), but it is **not** the full `symbro status`/`symbro where` command that was proposed; that's still unbuilt.

**The PyPy-vs-CPython cluster environment issue is resolved** (user's own base conda env was `mambaforge-pypy3`, which has no PyTorch support at all — confirmed via the upstream PyTorch GitHub issue; fixed by creating a genuine CPython env on the cluster).

## What's decided but still not implemented

These were explicitly discussed and resolved as decisions, but none of them exist in code yet:

- **RFdiffusion/ProteinMPNN CLI integration** — `symbro rfdiffusion`, `symbro pmpnn` (and eventually `symbro predict`) commands, wired through `pipeline.py` the same way stages 1–4 are (`run_rfdiffusion()`, `run_pmpnn()` functions, checkpointed to `.symbro/`). Decided architecture: **batch-per-assembly submission** (one job per assembly, not one giant job or one-per-ring), **block-and-poll by default** (the CLI call blocks and polls until done, matching stages 1–4's synchronous feel), with a `--detach` flag for the SLURM-backed stages plus a `symbro status` command to check on detached jobs later. None of this is built — `pipeline.py` still only has `run_query`/`run_download`/`run_geometry`/`run_isolate`, and `cli.py`'s `isolate` command literally prints "RFdiffusion + ProteinMPNN + prediction commands are coming in the next round."
- **`.symbro/` cleanup** — flagged in the 08-11 report (`pipeline.clear_state()` / `symbro clean`, mirroring `download.py`'s `clear_temp_dir()` / `isolate.py`'s `clear_temp_subunits_dir()`). Still not built.
- **`run_predictor.py`'s `MPNN_OUT_FOLDER`** is a literal placeholder path (`r"C:\path\to\your\symbro\project\..."`) the user must hand-edit before every run — the same class of footgun `paths.py` was built to eliminate elsewhere, just not yet applied here since this script sits outside the checkpoint system entirely (it deliberately re-derives its inputs from an already-completed run's output folder rather than a `.symbro/` checkpoint). Worth deciding whether this script gets folded into a real `symbro predict` command (making the placeholder moot) or gets its own small portability fix if it's going to stay standalone longer.

## What's built but genuinely unverified

- **Structure-prediction screening** (`alphafold2.py`, `boltz.py`, `af3.py`, dispatched via `structure_prediction.py`, run via `run_predictor.py`) — substantial, well-documented modules (licensing tradeoffs between the three backends are already written up in `structure_prediction.py`'s own docstring), but **none of them has ever been run against real data**. This is the same state RFdiffusion was in before 08-11 and ProteinMPNN was in before this week — "reads correctly, never tested against a live job." This is the next real milestone, not further code-writing.
- `selfconsistency.py`, `colab.py`, `symbro_rfdiffusion_colab.ipynb` — present in the repo, not reviewed or exercised in recent sessions; status unknown, worth a fresh look before relying on them.

## Repository state — action needed before continuing

`git status` on the working tree right now:

```
 M environment.yml
 M pyproject.toml
 M src/toolkit/cli.py
 M src/toolkit/download.py
 M src/toolkit/isolate.py
 M src/toolkit/rfdiffusion.py
?? src/toolkit/paths.py
?? test_pmpnn_hpc.py
?? test_rfdiffusion_hpc.py
?? requirements_backup.txt
?? symbro_status_report_2026-08-11.md
```

**Everything described above as "done" in this report is sitting uncommitted in the working tree.** The last real commit (`a3c3227`, "Replace hardcoded local path in run_predictor.py with a placeholder") predates all of it — the hybrid-36 fix, the (now actually-applied) `set -u` fix, `rank_designs()`, the whole path-portability fix, and both test scripts. Before doing more work, or definitely before continuing in a new chat session, commit this working tree (or at minimum make sure it's not lost) — a fresh session has no way to recover uncommitted local changes if something goes wrong with this machine. `requirements_backup.txt` is untracked and its purpose isn't documented anywhere obvious; worth a quick look before deciding whether it should be committed, gitignored, or deleted.

A stray `.git/index.lock` (empty, left over from a `git status` run through the device bridge, which couldn't clean it up after itself due to the bridge's write permissions) was found and cleared — moved to `_to_delete/git_index.lock_stale` in the project root rather than deleted outright, since file deletion isn't available through this path either. Safe to delete that folder by hand; `git status` runs clean now with no lock error.

## Still zero, unchanged from 08-11

- **Test suite**: `tests/test_geometry.py` and `tests/test_query.py` are still 0 bytes. Every verification done in this project so far — including the two real bugs found this week — has been through synthetic dry-runs and real HPC runs, never captured as a repeatable regression test. The hybrid-36 fix in particular has no test guarding against a regression.
- `README.md` and `CITATION.cff` are still 0 bytes (deliberately deferred).
- `examples/quickstart.py` is still 0 bytes.
- NiceGUI hasn't been started. Groundwork is deliberate, not incidental: `pipeline.py` exists specifically so a future GUI and the CLI share one orchestration layer, and `rfdiffusion.py`'s `submit()`/`poll_status()` split (non-blocking dispatch, cheap polling) was built explicitly so a `ui.timer`-driven GUI wouldn't need to block its UI thread — both by design, per their own docstrings. No layout, page structure, or interaction design exists yet beyond "single-user local tool, not hosted multi-user."

## Recommended priority order from here

1. **Commit the working tree.** Nothing else below matters if this week's work isn't safely in git first.
2. **Decide whether to build `symbro rfdiffusion`/`symbro pmpnn` now**, given the architecture is already decided (batch-per-assembly, block-and-poll + `--detach`, `symbro status`). This is the highest-leverage next step for the "user-friendly tool" goal specifically, since it's the difference between "the science works" and "someone who isn't you can run it."
3. **Run structure-prediction screening (Boltz first — cleanest licensing, MIT code and weights) against the real ProteinMPNN output this week's test just produced**, closing the loop to an actual validated design for the first time.
4. Once 2–3 exist, revisit `.symbro/` cleanup and `run_predictor.py`'s placeholder-path footgun — both are small, mechanical, and better done once the CLI shape for stages 5–7 is settled rather than twice.
5. Start a thin test suite, at minimum a regression test for the hybrid-36 contig bug and the path-portability fix — both were real, subtle, and already fixed once each; nothing currently stops either from coming back silently.
6. NiceGUI, once 1–5 are in a state you're comfortable exposing to a non-technical user.

## Quick reference — where things live

- `src/toolkit/pipeline.py` — orchestration layer, stages 1–4 only so far; this is where `run_rfdiffusion()`/`run_pmpnn()` belong when built.
- `src/toolkit/cli.py` — the `symbro` Typer app; stages 1–4 only so far.
- `src/toolkit/{download,isolate}.py` + new `src/toolkit/paths.py` — this week's portability fix.
- `src/toolkit/rfdiffusion.py`, `src/toolkit/pmpnn.py` — both HPC-verified, both not yet CLI-wired.
- `src/toolkit/{structure_prediction,alphafold2,boltz,af3,colab,selfconsistency}.py` — stage 7, built, unverified.
- `src/toolkit/config.py` + `installation.yaml` (gitignored, per-machine) — where every external tool's local paths/backend choice live.
- `test_rfdiffusion_hpc.py`, `test_pmpnn_hpc.py` (repo root, untracked) — the standalone scripts currently standing in for real CLI commands.
- `run_predictor.py` (repo root) — standalone stage-7 runner, has the placeholder-path issue noted above.
