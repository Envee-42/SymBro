# symbro examples

Three worked examples, in increasing order of how much infrastructure they need. Each is a folder with its own `README.md` — start with `01_full_walkthrough` regardless of which of the other two you actually need, since it's the one that establishes the real, working baseline the others build on.

| Example | What it shows | What you need to actually run it |
|---|---|---|
| [`01_full_walkthrough`](01_full_walkthrough/) | `symbro query` → `download` → `geometry` → `isolate` against a real, named RCSB entry, plus the exact commands for the remaining stages (`rfdiffusion`/`pmpnn`/`predict`) | Nothing beyond `pip install -e .` for stages 1–4. RFdiffusion/ProteinMPNN/a structure-prediction backend installed and configured (see the main README's **Configuration** section) for stages 5–7. |
| [`02_flags_and_narrowing`](02_flags_and_narrowing/) | The flags that matter once your candidate set has more than one interesting thing in it: `--symmetry-type`, `--component-id`, `--assembly-id`, `--select`, `--top-n`/`--min-plddt`, `--linker-min`/`--linker-max` | Same as above — this is a flag reference with real captured output, not a new dataset. |
| [`03_slurm_detach_workflow`](03_slurm_detach_workflow/) | Submitting `symbro rfdiffusion --detach` to SLURM and checking back later with `symbro status`, instead of blocking your terminal | A working `installation.yaml` with `rfdiffusion.backend: slurm`, and an actual SLURM cluster. This one is a documented pattern, not something re-verified from this review — see that folder's README for exactly what was and wasn't checked. |

## A note on how these were built

`01` and `02`'s query/download/geometry/isolate output below is real, captured output — not hand-written — from actually running those four stages against RCSB entry **8UF0** (`T33-ml23`, a machine-learning-designed tetrahedral protein cage; see [its RCSB page](https://www.rcsb.org/structure/8UF0)). It was picked deliberately: it has a two-component C3 axis *and* a C2 axis in the same structure, so one real download exercises both the "multi-component assembly" path (`symbro geometry`'s `components > assemblies` case, explained in `symbro geometry --help`) and the plain single-component path, without needing two different downloads. It's also itself a designed structure rather than a natural one — a fitting first example given the main README's own point that symbro treats the two identically.

Exact chain labels in `symbro geometry`/`isolate`'s output (e.g. which of `A-2`/`A-7`/`A-12` gets picked to represent a ring) can vary slightly between runs — they're symmetry-mate selections, not a fixed canonical numbering — so treat the row *counts* and *structure* of the output below as the reliable part, not the literal chain names.

The RFdiffusion/ProteinMPNN/structure-prediction commands shown in `01` and the SLURM pattern in `03` are documented from `cli.py`'s actual flag definitions, not re-run here — they need a GPU and, for `03`, a real cluster, neither of which this review had access to.
