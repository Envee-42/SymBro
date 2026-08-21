# Changelog

All notable changes to SymBro are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **pyproject.toml license metadata migrated to PEP 639**: `license = {
  text = "Apache-2.0" }` plus a separate `"License :: OSI Approved ::
  Apache Software License"` classifier replaced with the single SPDX
  license expression `license = "Apache-2.0"`, which current setuptools
  otherwise deprecates that combination in favor of. Requires
  `setuptools>=77` as the build-time requirement (bumped from `>=61.0`),
  since older setuptools can't parse the SPDX string form.

### Added

- **`codon.py` LICENSING docstring section**: documents DNAChisel (MIT)
  and python_codon_tables (CC0-1.0), confirmed directly against each
  project's own PyPI/GitHub metadata -- matching the LICENSING section
  every other optional-dependency backend module (af3.py, boltz.py,
  alphafold2.py) already carries. No functional change; both licenses
  are fully permissive with no conflict against this project's own
  Apache-2.0.

- **Annotated-symmetry cross-check** (`symbro geometry`, on by default):
  cross-checks each assembly's empirically detected rings against RCSB's
  own annotated symmetry (`rcsb_struct_symmetry.symbol`, now always
  fetched by `symbro query`) and drops any assembly where none of its
  expected cyclic axes were actually confirmed by detection -- a real,
  if uncommon, sign of a PDB annotation issue (wrong assembly marked
  biological, a crystallographic packing mate mistaken for the real
  ring, etc.) rather than a genuine candidate worth pursuing compute on.
  Covers plain cyclic annotations (`C2`-`C5`) directly, and Platonic
  annotations (`T`, `O`, `I`) via their own known constituent cyclic
  axes (e.g. `T` -> C3 or C2) -- this project's own target assemblies
  are Platonic by nature, so that decomposition is exactly what
  `symbro isolate` already extracts, not a guess. A warning naming the
  assembly, what was expected, and what was actually detected is always
  printed. Dihedral/helical/asymmetric annotations, and `symbro local`
  candidates (never looked up against RCSB), are left unvalidated.
  Pass `--no-validate-symmetry` to keep every detected ring regardless.

- **Incomplete-ring warning** (`symbro geometry`, on by default): flags
  -- but never drops -- any surviving row whose `axis_count` falls short
  of the most disjoint rings its own component chain count could
  possibly support (a new `component_chain_count` column on `rings.py`'s
  output, `// order` gives the theoretical max). Independent of the
  annotation cross-check above: it fires even when the expected axis
  type is correctly confirmed, since it's checking whether every
  eligible chain formed a detectable ring, not whether the right type of
  ring exists at all. A shortfall is deliberately warn-only, not
  dropped -- it usually reflects real, chain-specific structural
  disorder in that particular deposition rather than an annotation
  problem, and collapsing that into an automatic drop would be too
  aggressive a call to make without a human looking at which chains were
  left unclaimed. Pass `--no-warn-incomplete-axes` to skip it.

- **`symbro view`**: renders a structure (a direct file path, or
  `--stage downloaded`/`rings` + `--assembly-id`/`--component-id`) as a
  self-contained, interactive 3D HTML file via a vendored copy of
  3Dmol.js (`toolkit/static/3dmol/`, BSD-3-Clause -- see that directory's
  `NOTICE.md` for exact version/provenance/license). Deliberately the one
  command in this pipeline needing nothing beyond `pip install symbro`:
  no `installation.yaml` entry, no external tool, no GPU, and no network
  access either to generate the file or to open it later, since the
  entire viewer library is embedded in the output rather than
  CDN-linked -- meant to work identically for anyone who downloads
  symbro, not just on any one lab's own hardware. Scope kept lean for
  this first pass: one structure per view, colored by chain; no overlay
  of two structures or symmetry-axis drawing yet.

- **`symbro codon --max-attempts`** (`codon.optimize_sequence()`/
  `optimize_candidates()`/`run_codon()`): DNAChisel's constraint solver
  has genuine call-to-call randomness (no seed exposed anywhere in its
  API) -- on a hard case, most notably a protein with near-identical
  repeated domains (this project's own fused-ring designs' own shape,
  which fights the no-repeated-k-mers constraint), a single attempt can
  fail a meaningful fraction of the time even though the constraints ARE
  jointly satisfiable most tries -- confirmed on a real HPC-produced
  candidate at roughly 1-in-3 failure per attempt, resolving within a
  couple of retries. `--max-attempts N` (default 1, no behavior change
  unless you opt in) retries within the same command, keeping the first
  attempt whose `warnings` comes back empty -- turning "notice a
  flagged row, re-run `symbro codon` by hand, repeat" into one flag.
  Deliberately not made the default behavior -- every attempt is real
  solver work, so cost stays predictable unless explicitly requested.
  Verified against a synthetic repeated-domain protein: `--max-attempts 3`
  raised the clean-result rate from roughly 65-70% (a single attempt) to
  95% (19/20) in a real run.

### Fixed

- **Ring-detection direction/exclusivity bug** (`symbro geometry`, C3/C4/C5
  rings): for a chain set of order >= 3, the C-terminus -> N-terminus
  cycle search could accept either the true forward ring or an equally
  step-homogeneous "skip past your real neighbor" reverse reading of the
  exact same chains -- and, one level up, a higher-symmetry assembly
  (e.g. a T-symmetric 12-chain cage) could have more than one disjoint
  way to partition its chains into same-order rings, all independently
  passing detection. CV (step-to-step regularity) couldn't distinguish
  either case from the truth, and on a real, reported case (PDB 4EGG) it
  actively preferred the wrong answer both times -- first a
  reversed-direction trimer at ~70 A instead of the correct ~26-29 A
  junction, then, after fixing the direction, a cross-trimer partition at
  ~58-59 A instead. Both are now resolved the same way, per this
  project's own fusion-design goal (closest N-to-C termini get fused):
  `find_cyclic_groupings` keeps only the closest-termini direction per
  chain set, and `select_disjoint_groupings` now sorts
  closest-mean-distance-first (CV/std remain as tiebreakers) instead of
  CV-first. Verified against the real 4EGG structure: now correctly
  recovers its 4 true C3 trimers (~26-27 A junctions, matching the
  originally reported ~29 A figure) instead of the wrong-direction or
  cross-trimer readings.

- **`completed_partial` treated as terminal with no retry** (every
  compute-backend `run()` convenience wrapper -- `boltz.py`,
  `alphafold2.py`, `af3.py`, `pmpnn.py` -- plus `pipeline.run_pmpnn()`'s
  own independently-written version of the same loop): each of these
  blocked with `while status["state"] == "running": ...`, which exits
  immediately the first time `poll_status()` reports *any* non-running
  state, including `"completed_partial"`. That state is reported both
  when a job genuinely crashed/was killed after finishing only some
  outputs (no amount of waiting fixes this) AND when a job finished
  cleanly (`returncode == 0`, every output really was written) but the
  polling process's own view of the filesystem hadn't caught up to all
  of them yet -- two very different situations the old code treated
  identically, permanently and silently dropping any candidate caught in
  the second, transient case. Confirmed for real on a live HPC run
  (4V6B, boltz): 3 of 180 candidates were fully folded and on disk but
  got excluded from `predict.pkl` with no error or warning, because the
  one poll that ran landed inside that window -- caught only by manually
  cross-checking raw output files against the reported result rather
  than trusting `predict.csv`'s "0 passed."
  Fixed via a new shared helper, `toolkit/polling.py`'s
  `wait_for_terminal()`: on seeing `"completed_partial"`, re-polls up to
  `partial_retries` more times (`partial_retry_interval` seconds apart,
  defaulting to 3 retries / 5s) before accepting it as final -- a retry
  that shows more outputs than the last poll keeps going, one that shows
  the exact same (still-incomplete) count for every retry is accepted as
  genuinely partial. All five call sites above now go through this one
  helper instead of their own copy of the loop, matching this project's
  own precedent of extracting shared polling logic after duplicated
  copies of it already caused a real bug once (see the SLURM
  sbatch-polling primitives shared via `selfconsistency.py`).
  Deliberately NOT applied to RFdiffusion's own multi-row round-robin
  poller (`pipeline._poll_all_rfdiffusion()`, backing `symbro rfdiffusion`'s
  blocking wait and `symbro status`) -- its `single_pass=True` mode
  (`symbro status`) is a deliberate one-shot manual refresh where a human
  re-invoking the command already acts as the retry; folding automatic
  partial-retries into it would change that command's own documented
  "cheap, immediate" contract rather than just fix a bug. Left as a
  candidate for a follow-up if the same filesystem-lag failure mode is
  ever confirmed there too.

- **`symbro codon` crashing, and separately silently returning a
  constraint-violating sequence, on repeated-domain proteins**
  (`codon.optimize_sequence()`): found while codon-optimizing the first
  real validated designs from a live HPC run -- this project's own
  fused-ring output (several near-identical domains joined by diffused
  linkers into one chain) is exactly the shape that triggers both bugs
  below, not an edge case.
  1. `problem.optimize()` (DNAChisel's objective-improvement step) used
     to run UNCONDITIONALLY, even right after `problem.resolve_constraints()`
     had already raised `NoSolutionError`. DNAChisel requires every
     constraint to already be verified before `optimize()` can run
     (confirmed directly against its source) and raises its OWN
     `NoSolutionError` otherwise -- a second, previously-uncaught
     exception that turned a should-be-recoverable "warn and return the
     best-effort sequence" case (see this module's own docstring) into an
     unhandled crash, which `optimize_candidates()`'s per-row
     `try/except` then only surfaced as a plain "Skipped" line --
     silently dropping the candidate from `codon.pkl`/`codon.fasta`
     rather than returning it flagged.
  2. Separately, and worse: DNAChisel's `optimize()` does NOT actually
     guarantee it preserves constraint satisfaction while improving the
     `CodonOptimize` objective -- confirmed empirically (not assumed from
     docs) by reproducing it directly against a synthetic repeated-domain
     protein: `optimize()` can reintroduce a duplicate-k-mer violation
     that `resolve_constraints()` had *just* resolved, with no exception
     raised at all, leaving `warnings` empty (the documented "safe,
     common case") for a sequence that actually still violates one of
     its own constraints. Reproduced at roughly a 5-10% rate on a
     synthetic 4-near-identical-domain test protein.
  Both fixed together: `optimize()` is now only called once
  `resolve_constraints()` has actually succeeded, and its result is
  checked afterward via `problem.all_constraints_pass()` (the real,
  authoritative check DNAChisel itself exposes) -- if that fails, the
  sequence is reverted to the last confirmed constraint-satisfying one
  (from immediately before `optimize()` ran) and a warning is added,
  rather than either crashing or silently keeping the now-violating
  "optimized" result. Every `warnings`-populated case was independently
  re-verified (not just trusting DNAChisel's own report) across a real
  80-attempt run against the same synthetic repeated-domain protein:
  zero crashes, zero sequences with an undetected duplicate 15-mer.

### Documentation

- Added a "Structure-prediction backend status" section to the README
  recording real end-to-end SLURM testing results on our own HPC cluster
  for all three predictor backends: `af2` works but silently fell back
  to CPU due to a driver/CUDA mismatch; `boltz`'s earlier failure was
  root-caused to a corrupted local model cache, fix applied but not yet
  reconfirmed; `af3` is blocked specifically by this cluster's GPU
  driver (`525.60.13`) being one full CUDA release behind what its
  pinned `jax[cuda12]` build requires, not by anything in symbro's own
  config. None of these are symbro bugs -- recorded so they don't need
  re-diagnosing later.

## [1.0.0] - 2026-08-18

First stable release. SymBro now covers the full symmetry-broken protein
cage design pipeline end to end, from an RCSB search (or your own local
structure file) to a codon-optimized, orderable DNA sequence, entirely
through the `symbro` CLI.

### Added

- **Full 11-command pipeline**, each stage resuming automatically from a
  `.symbro/` checkpoint written by the previous one:
  - `symbro query` — search RCSB PDB for candidate assemblies.
  - `symbro download` — download the matching structure files.
  - `symbro local` — register your own local PDB/CIF file(s) as an
    alternative entry point to `query` + `download`.
  - `symbro geometry` — detect symmetry rings (and orientation/termini).
  - `symbro isolate` — extract each ring's PDB, ready for RFdiffusion.
  - `symbro rfdiffusion` / `symbro status` — submit RFdiffusion jobs
    (local, Singularity, or SLURM backends) and poll detached runs.
  - `symbro pmpnn` — run ProteinMPNN against each assembly's best design(s).
  - `symbro predict` — fold candidates back through a structure-prediction
    backend (`boltz`, `af2`/`alphafold2`, `af3`/`alphafold3`) and screen by
    self-consistency RMSD/pLDDT, with `--top-n`/`--max-rmsd`/`--min-plddt`
    filtering.
  - `symbro codon` — reverse-translate validated designs into
    host-codon-optimized DNA (optional `symbro[codon]` extra, via
    DNAChisel + python_codon_tables), with GC-content, homopolymer,
    hairpin, k-mer-repeat, and Golden Gate enzyme-site safety checks.
  - `symbro clean` — clear `.symbro/` checkpoints and scratch directories
    between runs (`--keep-*`/`--dry-run` flags).
- **AF3 MSA/template-free shortcut** (`--backend af3`, `run_data_pipeline`
  flag): lets AF3 run without its multi-hundred-GB genetic database
  directory when a full data pipeline isn't needed.
- **Onboarding kit**: an `examples/` directory with four worked walkthroughs
  (full pipeline, flag/narrowing reference, SLURM detach workflow, and a
  local analysis notebook for exploring a completed run's results) plus a
  `minimal_scripts/` quick-start (`launch.sh`/`launch.ps1`/`Instructions.txt`).
- **Two Colab notebooks** (`symbro_rfdiffusion_colab.ipynb`,
  `symbro_full_pipeline_colab.ipynb`) for running the GPU-bound stages on
  hosted infrastructure when no local/HPC GPU is available.
- **Shared checkpoint-join helper** (`pipeline.join_predict_with_pmpnn()`)
  used by both `symbro codon` and the analysis notebook, replacing
  duplicated join logic and fixing a real dtype-mismatch bug in the
  process (narrowed, all-non-null `predict` frames failed to merge against
  the un-narrowed `pmpnn` frame).
- Per-machine tool configuration via `installation.yaml`
  (`installation.example.yaml` as the template), supporting local,
  Singularity, and SLURM backends for RFdiffusion and ProteinMPNN.

### Fixed

- Hardcoded, machine-specific local paths removed from `run_predictor.py`
  and replaced with portable placeholders / relative paths.
- RFdiffusion SLURM submission script no longer crashes under `nounset`
  when `setup_lines` is empty.
- Cross-machine checkpoint path portability (`.symbro/` state now resolves
  correctly regardless of which machine/OS wrote it).
- `join_predict_with_pmpnn()` / `_component_key()` dtype mismatch that
  raised `ValueError` when merging a narrowed (pure float64) `predict`
  frame against an un-narrowed (object-dtype) `pmpnn` frame.

### Testing

- 132 tests across the CLI layer, pipeline stages, and each backend
  integration (RFdiffusion/ProteinMPNN HPC scripts, `predict`, `codon`,
  `local`, `af3`), run against real code paths with mocking limited to the
  true external I/O boundary (cluster submission/polling, RCSB network
  calls) per this project's own testing convention.
- RFdiffusion and ProteinMPNN's SLURM backends have been exercised for
  real against a live HPC cluster; the CLI layer on top of `predict` has
  not yet had a live end-to-end run against real infrastructure — tracked
  as the next verification milestone.

[1.0.0]: https://github.com/Envee-42/SymBro/releases/tag/v1.0.0
