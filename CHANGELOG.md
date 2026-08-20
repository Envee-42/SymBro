# Changelog

All notable changes to SymBro are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
