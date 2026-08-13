# SymBro — Status Report (2026-08-11)

## Where things stand

SymBro's core scientific pipeline — query RCSB PDB → download → geometry analysis → isolate ring subunits → RFdiffusion → ProteinMPNN → structure-prediction self-consistency screening → validated designs — is roughly two-fifths built as a CLI, and for the first time this session, part of it has actually touched a real HPC cluster rather than just being designed against documentation. That's a meaningful milestone: everything before today was either working code that had never left your laptop, or careful design that had never been tested. Today closed that gap for the RFdiffusion/SLURM path specifically, and surfaced two real bugs in the process — both fixed and verified against primary sources rather than assumed.

## What's built and verified

**Phase 1 of the CLI** — `symbro query`, `download`, `geometry`, `isolate` — was already complete and tested against real RCSB data before this session, and remains untouched. `pipeline.py` orchestrates all four stages with checkpointing to `.symbro/`; `cli.py` is a thin, well-tested Typer layer on top.

**Git history** is now clean. The old `INSTALLATION.yaml` (containing your real local paths, briefly tracked early on) is confirmed gone from every commit via `git filter-repo`, verified independently rather than trusting the tool's own output — no dangling objects, `origin` intact, nothing pushed yet so nothing was ever exposed publicly.

**RFdiffusion's SLURM backend has now been exercised on your-hpc-cluster for the first time ever.** This matters because the code had sat since a commit literally titled *"Predictors and HPC unproven"* — today changed that. In order: the SLURM submission/polling plumbing itself (sbatch generation, `squeue`/`sacct` polling, job-id tracking) is now confirmed working end to end via a completed no-GPU smoke test. A real GPU job then got substantially further — CUDA visible, model checkpoint loaded, diffuser initialized — before hitting a genuine bug in how this project builds fusion contigs, not a config or cluster problem.

That bug is now fixed and independently verified: RFdiffusion's own contig grammar (confirmed directly against `RosettaCommons/RFdiffusion`'s source, not recalled from memory) cannot fuse fixed motif residues from more than one original chain letter into a single new output chain — which is exactly what fusing a ring's three subunits into one new chain requires. The fix relabels the chains being fused onto one shared id with disjoint residue-number blocks before building the contig, and was verified by running RFdiffusion's actual `ContigMap` parser against both the old (failing) and new (passing) contig shapes — not just by reasoning about the source. A GPU job with this fix is running as of this report; the earlier bugs (an OpenSSH `scp -r` incompatibility during file transfer, and a `set -u`/conda-MKL activation conflict in the generated sbatch script) are both resolved, the second via a workaround in your `installation.yaml` rather than a code change — more on that below.

## Two things worth fixing before going further

You flagged two real gaps, and both hold up on inspection:

**`.symbro/` has no cleanup path.** Every stage writes checkpoints there, but there's no equivalent of `download.py`'s `clear_temp_dir()` or `isolate.py`'s `clear_temp_subunits_dir()` for the state directory itself — a user re-running the pipeline from scratch currently has to delete it by hand. A `pipeline.clear_state()` plus a `symbro clean` command would match the pattern already established elsewhere in the codebase; small, mechanical, and not yet built.

**State-directory resolution is silently CWD-dependent, not just "hardcoded."** It's not that a specific machine's path is baked in (it isn't) — it's that `.symbro/candidates.pkl` and friends are resolved relative to wherever the command happens to be invoked *from*, every time. Run `symbro query` from one directory and `symbro geometry` from another, and the second command won't find the first's output — it'll just say no checkpoint exists, with nothing pointing at why. That's a real footgun for a CLI meant to be used by people other than its author, and it's exactly what "easily findable" is getting at. `run_predictor.py`'s `MPNN_OUT_FOLDER`, which is still a literal `C:\path\to\your\symbro\project\...` placeholder, is the same category of issue at a smaller scale — already flagged before, still unaddressed. A `symbro status` or `symbro where` command that prints the currently-resolved state directory (and whether each expected checkpoint exists in it) would make both problems visible instead of silent, on top of whatever the actual fix ends up being.

Neither is fixed yet — flagging them here rather than acting on them, since you asked to step back rather than keep building.

## Still open from before

The phase-2 CLI design question (batch-per-assembly submission, block-and-poll by default, `--detach` for SLURM only) was proposed two sessions ago and never explicitly answered — today's work deliberately tested `rfdiffusion.py` directly, bypassing the CLI, specifically so that question wouldn't need answering yet. It still does, before `symbro rfdiffusion` itself gets built. Separately, the `set -u`/MKL fix in the sbatch template was proposed as a permanent code change and never confirmed — right now it's only patched around via your own `installation.yaml`'s `setup_lines`, which works for you but won't travel to someone else's `installation.yaml` automatically. Worth deciding once the current GPU run's result is in.

Test coverage is still zero (`tests/test_geometry.py` and `test_query.py` remain 0 bytes), and it matters more now than it did last time this was flagged: today's verification of the contig-fusion fix lived entirely in a scratch sandbox and isn't captured anywhere in the repo, so the exact bug that was just found and fixed has no regression test guarding against it coming back. `README.md`/`CITATION.cff` remain deliberately deferred, unchanged.

## The road ahead, kept in view

Once the current GPU job confirms the fix, the natural next real test is ProteinMPNN against whatever RFdiffusion actually produces — `pmpnn.py` exists and is unmodified since before this project's rename, but is just as untested against a real HPC as `rfdiffusion.py` was this morning. Structure-prediction screening (`alphafold2.py`/`boltz.py`/`af3.py`) is the step after that, and is where the pipeline finally produces something to evaluate: a filtered, validated, symmetry-broken cage design. Each of those modules deserves the same treatment RFdiffusion just got — real HPC runs, not just design review — before being wired into the CLI.

NiceGUI hasn't been started, but the groundwork for it has been deliberate rather than incidental: `pipeline.py` exists specifically so the CLI and a future GUI can share one orchestration layer without duplicating logic, and `rfdiffusion.py`'s `submit()`/`poll_status()` split (non-blocking dispatch, cheap polling) was built explicitly so a NiceGUI `ui.timer` could drive it without freezing the UI thread — both docstrings say so directly. What's still entirely undecided is layout, page structure, and interaction design; nothing there yet beyond "single-user local tool, not hosted multi-user."

## A general note on progress

This session covered a lot of real ground in a short space — a from-scratch git history rewrite, first-time HPC onboarding, and a genuine, previously-undiscovered bug in scientific code, all found and fixed through actual testing rather than review alone. That's exactly the kind of progress that's easy to keep riding past the point of digesting it. Pausing here to fix the two structural gaps you just raised, decide the two open questions above, and get at least a thin layer of tests in place before extending the pipeline further, seems like the right-sized next step rather than pushing straight on to ProteinMPNN.
