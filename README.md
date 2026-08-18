# symbro

A CLI/pipeline for symmetry-broken protein cage design: RCSB query → download → geometry analysis → isolate ring subunits → RFdiffusion → ProteinMPNN → structure-prediction screening → validated designs → (optionally) codon-optimized DNA, ready to order.

**New to this?** In plain terms: symbro helps you take a symmetric protein assembly — a "cage" made of several identical copies of a chain arranged around an axis or a point — and redesign one repeating "wedge" of it into something new, using AI protein-design tools, while checking computationally that the redesign is likely to actually fold correctly before you'd ever synthesize it in a lab. That starting assembly doesn't have to be naturally occurring: RCSB (the archive symbro searches) holds both naturally-evolved structures (e.g. a virus shell or an enzyme ring) and previously engineered/de novo-designed ones (e.g. a published synthetic nanocage) side by side, and symbro works with either kind the same way. You don't need to be a programmer to use it: each step below is one typed command, and symbro tells you what to run next after every one.

Built for real use on a local checkout (Windows/Linux/macOS) with heavy compute (RFdiffusion, ProteinMPNN, structure prediction) run separately — locally, in a Singularity container (a lightweight, HPC-friendly alternative to Docker), or submitted to a SLURM HPC cluster (SLURM is a common job scheduler for shared university/institute computing clusters). Whether "locally" is actually an option for you depends on which AI tool and which operating system — see the callout below the stage table before assuming everything runs on your own machine.

## What it does

Given a symmetry type you're interested in (e.g. C3, meaning three identical copies arranged around a central axis), symbro takes you from "search the PDB (the Protein Data Bank, RCSB's public archive of experimentally-solved protein structures) for candidate structures" through "here are validated designed sequences" in a sequence of small, resumable steps. Each step picks up automatically where the last one left off, via a small `.symbro/` checkpoint folder in your project directory — so you can stop, inspect the output, adjust parameters, and re-run any single stage without redoing the ones before it.

| Stage | Command | What it does |
|---|---|---|
| 1 | `symbro query` | Search RCSB PDB for candidate assemblies matching your criteria (symmetry, resolution, keywords, ...) |
| 2 | `symbro download` | Download the matching structure files |
| — | `symbro local` | *Alternative* to `query`+`download`: register your own local PDB/CIF file(s) instead of searching RCSB |
| 3 | `symbro geometry` | Detect symmetry rings in each structure (and orientation/termini secondary structure, for one chosen symmetry order) |
| 4 | `symbro isolate` | Extract each ring's own structure file, ready for RFdiffusion |
| 5 | `symbro rfdiffusion` | Submit RFdiffusion (an AI model that generates new protein backbone shapes) — one job per assembly — locally, via Singularity, or to a SLURM cluster |
| — | `symbro status` | Check on `--detach`'d RFdiffusion jobs |
| 6 | `symbro pmpnn` | Run ProteinMPNN (an AI model that fills in an amino-acid sequence for a given backbone shape) against each assembly's best RFdiffusion design(s) |
| 7 | `symbro predict` | Fold ProteinMPNN's best candidate sequence(s) back into a 3D structure (via AlphaFold2, Boltz, or AlphaFold3) and screen them by self-consistency — i.e., check the refolded shape actually matches what RFdiffusion originally designed, before you trust the sequence |
| 8 | `symbro codon` | *Optional.* Reverse-translate `predict`'s validated designs into host-codon-optimized DNA, ready to order (needs the `codon` extra: `pip install symbro[codon]`) |
| — | `symbro clean` | Clear scratch files and checkpoints between runs |

Every stage before `symbro rfdiffusion` runs on your own machine with no special hardware. `symbro rfdiffusion`, `symbro pmpnn`, and `symbro predict` are the compute-heavy AI steps — a GPU (graphics card capable of accelerating deep learning) makes these dramatically faster, and is required for most real-world-sized jobs. Each of these three offers `backend: "local"` (run right here, as a plain subprocess), `"singularity"` (a container image), or `"slurm"` (submit to an HPC cluster) — configured per-tool in `installation.yaml` (see **Configuration** below).

**A real caveat about `backend="local"`:** it's genuine local execution wherever symbro itself is running — but whether the underlying AI tool can actually install and run there depends on that tool's own upstream dependencies, not on symbro. symbro's own code (everything through `symbro isolate`, plus all the orchestration around the AI steps) is itself fully OS-agnostic pure Python — it runs identically on Windows, macOS, and Linux. The AI tools it wraps are a different story, and each has its own answer:

| Tool | Windows (native) | macOS | Linux |
|---|---|---|---|
| **RFdiffusion** | ✗ — DGL/SE(3)-Transformer has no native-Windows GPU path; needs WSL2 | ~ works via a community CPU/MPS-patched fork (no NVIDIA GPU needed, but 2-3x slower without one) — not the official install path | ✓ official, GPU-accelerated |
| **ProteinMPNN** (`pmpnn`) | ✓ CPU or GPU | ✓ CPU (falls back automatically — no CUDA/GPU required) | ✓ CPU or GPU |
| **AlphaFold2/ColabFold** (`predict --predictor af2`) | ✗ — JAX has no native-Windows GPU support; [needs WSL2](https://github.com/jax-ml/jax/discussions/5607) | ~ installable (community `localcolabfold` Apple Silicon build exists), but effectively CPU-only — Apple's own JAX/Metal GPU plugin is unmaintained | ✓ official, GPU-accelerated |
| **Boltz** (`predict --predictor boltz`) | ✓ CPU or GPU (plain PyTorch) | ~ installs and runs, but CPU-only in practice today — Apple Silicon MPS acceleration isn't merged upstream yet, and boltz's own CLI errors instead of auto-falling-back if it doesn't find a GPU, so pass an explicit CPU override | ✓ CPU or GPU |
| **AlphaFold3** (`predict --predictor af3`) | ✗ — Linux only, no exceptions | ✗ — same: Google's own install docs state "AlphaFold 3 does not support other operating systems," and it also requires an NVIDIA GPU no Mac has anyway | ✓ official, GPU-accelerated (NVIDIA Ampere+) |
| **`backend="singularity"`** | ✗ — Linux container runtimes don't run natively on Windows | ✗ — same; both need a Linux VM (e.g. Lima) to use this backend at all | ✓ native |
| **`backend="slurm"`** | ✓ — just submits to a remote Linux cluster; your own OS barely matters | ✓ | ✓ |

("✓" = works as officially supported; "~" = works via a community path or with a real caveat, not the tool's own official support; "✗" = does not work natively.)

**Practically:** if you're moving your symbro checkout from Windows to a Mac, `pmpnn` and `predict --predictor boltz` (CPU) keep working the same way. `rfdiffusion` and `predict --predictor af2` go from "blocked without WSL2" to "works, just slower, via a community-patched build" — a genuine improvement, though still not the tool's own officially-supported path. `predict --predictor af3` and `backend="singularity"` stay blocked either way. None of this is a symbro limitation specifically — it's inherited entirely from what RFdiffusion/JAX/AlphaFold3 support — but it's worth knowing before planning a local-only workflow on either OS. A SLURM cluster (`backend="slurm"`) sidesteps all of it, on any OS.

Run `symbro --help` or `symbro <command> --help` at any time for the full option list — every command explains its own options in plain language there, including ones not covered in this README.

### Advanced querying

`symbro query`'s `--symmetry`/`--resolution-min`/`--resolution-max`/`--description` options cover the common cases, but two extra flags exist for anything else, both using the same `attribute=value[:operator]` syntax (repeatable):

- `--criterion` — an extra condition sent to RCSB's own search, e.g. `--criterion "experimental_method=X-RAY DIFFRACTION"`.
- `--filter` — a condition applied locally *after* results come back, for the handful of fields RCSB's search can't filter on directly (e.g. `--filter "model_quality=70:greater_or_equal"`).

If you're not sure which one you need, start with `--criterion` — `symbro query --help` explains the difference and lists example operators (`range`, `contains_phrase`, etc.).

### Using your own structure instead of RCSB

Already have the PDB/CIF file you want to work with — your own prior design, something not (yet) in RCSB, or just a structure you'd rather not re-download? `symbro local` registers it directly, skipping `symbro query`/`symbro download` entirely:

```bash
symbro local my_cage.pdb
symbro local ring_a.pdb ring_b.pdb --assembly-id my-cage-A --assembly-id my-cage-B
```

It writes to the exact same checkpoint `symbro download` does, so every stage from `symbro geometry` onward treats a locally-registered structure identically to one fetched from RCSB — no separate code path, no missing features. Each file is copied into `temporary_files/local/` (covered by `symbro clean`'s existing `--keep-downloads` flag, same as a real download). This is an alternative to `symbro query`+`symbro download`, not something you combine with them in the same project — running `symbro local` after `symbro download` (or vice versa) overwrites whichever `downloaded.pkl` ran last, the same way re-running `symbro download` itself would.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/Envee-42/SymBro.git
cd symbro
pip install -e .
```

This includes PyTorch, since most symbro users run ProteinMPNN (CPU works fine — no CUDA, NVIDIA's GPU-computing toolkit, required). On a machine without a GPU, or where you don't want pip to resolve a multi-GB CUDA build by default, install with the CPU-only wheel index instead:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

Prefer conda? An equivalent environment is provided (CPU-only PyTorch by default — see the comments in `environment.yml` if you want a CUDA build instead):

```bash
conda env create -f environment.yml
conda activate symbro
```

RFdiffusion, ProteinMPNN itself (the `protein_mpnn_run.py` script and model weights), and the structure-prediction backends (AlphaFold2/Boltz/AlphaFold3) are still **not** installed by this package — their own installs are heavy (RFdiffusion in particular needs a CUDA-pinned environment), and most of what symbro does (query/download/geometry/isolate) never touches them. `pip install -e .` only gets you the Python library ProteinMPNN needs (PyTorch) — you still need your own clone of ProteinMPNN itself. Install and point symbro at your own copies of each as you need them — see **Configuration** below.

`symbro codon` (reverse-translating validated designs into orderable DNA — see **Quickstart**'s optional last step) needs its own small extra, since it's pure Python with no GPU/external repo involved but still only used by that one, optional final stage:

```bash
pip install -e ".[codon]"
```

No `installation.yaml` setup needed for it — unlike the AI stages, it's a plain pip dependency.

## Configuration

Machine-specific settings (where your RFdiffusion/ProteinMPNN/structure-prediction (AlphaFold2/Boltz/AlphaFold3) clones live, which Python environment to use, which execution backend) live in one gitignored file, `installation.yaml`, in your project root:

```bash
cp installation.example.yaml installation.yaml
```

Then edit it — at minimum, `repo_path`/`python_executable` (or the equivalent executable path) for whichever tools you're using. See `installation.example.yaml` for the full field-by-field reference, including SLURM-specific settings (`partition`, `time`, `gres`/`gpus`, `setup_lines` for conda/module activation, etc.) if you're submitting to an HPC cluster. Note that AlphaFold3's model weights specifically carry a non-commercial license and must be requested directly from Google — symbro never fetches or bundles them; see `af3.py`'s own module docstring and the `af3:` section of `installation.example.yaml` before using that backend.

**AF3 without the hundreds-of-GB database:** `af3.db_dir` is only required if you actually want AF3 to search real genetic/template databases for MSA hits — set `af3.run_data_pipeline: false` instead (or pass `--af3-no-data-pipeline`) and this backend needs nothing beyond `model_dir`, the same setup cost as Boltz/AlphaFold2. Confirmed directly against AF3's own source and docs, not assumed: `db_dir` is never read when the data pipeline is off, and every candidate is folded using AF3's own documented MSA/template-free mode (query sequence only) rather than left in an undefined in-between state. This is a genuine prediction-quality tradeoff, not just a setup shortcut — ProteinMPNN-designed sequences are often novel enough that a real search adds little, but it's still an explicit choice, not symbro's default.

Nothing here is required to get started — every setting can also be passed directly as a CLI flag, and `symbro query`/`download`/`geometry`/`isolate` don't need any of this at all.

## Quickstart

```bash
# 1. Find candidate structures
symbro query --symmetry C3 --resolution-max 2.5

# 2. Download them
symbro download

# 3. See what symmetry types are present, then narrow to one
symbro geometry
symbro geometry --symmetry-type C3

# 4. Extract each ring structure
symbro isolate --symmetry-type C3

# 5. Submit RFdiffusion (blocks until done; add --detach to return immediately
#    and check back later with `symbro status` — SLURM backend only)
symbro rfdiffusion

# 6. Run ProteinMPNN against the best RFdiffusion design(s)
symbro pmpnn

# 7. Fold the best sequence(s) back and screen by self-consistency
#    (picks a predictor from installation.yaml by default; override with --predictor)
symbro predict

# 8. Optional: reverse-translate the validated designs into orderable DNA
#    (needs the "codon" extra: pip install symbro[codon])
symbro codon --host e_coli

# Start fresh between runs
symbro clean
```

Every command prints where its output was saved (a `.pkl` checkpoint plus a human-readable `.csv` preview) and what to run next. The designs left in `predict.csv` after step 7 — the ones that passed the RMSD/pLDDT self-consistency thresholds — are your validated output.

**Analyzing your results:** [`examples/04_analysis_notebook`](examples/04_analysis_notebook/) has a local Jupyter notebook (no GPU needed, just `symbro` itself installed) that reads `predict.pkl` straight off disk and gives you a self-consistency RMSD-vs-pLDDT scatter, a per-assembly/symmetry-type breakdown, an adjustable shortlist, and a structure preview — a much better lens on the same numbers than eyeballing `predict.csv`.

**Codon-optimizing for gene synthesis:** step 8 above reverse-translates each validated design into DNA, codon-optimized for a host of your choice (`--host e_coli` by default; also `s_cerevisiae`, `h_sapiens`, `c_elegans`, `b_subtilis`, `d_melanogaster`, `g_gallus`, `m_musculus`), applying the standard safety checks a synthesis vendor would otherwise flag: GC content kept within a safe range both overall and in local windows, no long homopolymer runs, no problematic hairpins or repeated subsequences, and the two most common Type IIS Golden Gate cloning sites (BsaI/BsmBI) kept out of the sequence by default. It saves `.symbro/codon.{pkl,csv}` plus an orderable `.symbro/codon.fasta`. This gets you a strong starting sequence per design, not a fully automated expression-vector assembly — every sequence is still meant to be reviewed by hand before you actually order it. See `codon.py`'s own module docstring for the full scope and reasoning, including how this compares to the Baker-lab Domesticator3 tool prosculpt's own equivalent step wraps.

## Project status

Query, download, local structure registration, geometry, isolate, RFdiffusion, ProteinMPNN, structure-prediction screening (`symbro predict`, across all three backends — AlphaFold2, Boltz, and AlphaFold3), and codon optimization (`symbro codon`) are all wired up end-to-end and covered by the automated test suite described below. Real-world validation of each structure-prediction backend against a live HPC cluster run is ongoing — if you hit a rough edge specifically in `symbro predict` on your own cluster, that's the newest and least battle-tested part of the pipeline, so it's the most likely place for one. `symbro codon`'s own DNA-optimization logic is fully tested against real DNAChisel runs; what isn't independently verified is a synthesis vendor's actual acceptance of its output, since that needs a real order to confirm.

## Testing

```bash
pip install pytest
pytest
```

The test suite mocks only the actual outside-world I/O — submitting an RFdiffusion/ProteinMPNN/structure-prediction job and waiting on a GPU, and the network calls `symbro query` makes to RCSB — everything else (file parsing, checkpoint handling, CLI argument wiring, error messages) runs for real against small fixture structures, so it runs in well under a second with no GPU, cluster, or internet access required.

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Citing symbro

See [CITATION.cff](CITATION.cff).
