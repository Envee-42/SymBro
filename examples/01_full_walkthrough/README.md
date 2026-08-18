# 01 — full walkthrough

An end-to-end run against one real, named RCSB entry: **8UF0** (`T33-ml23`, a machine-learning-designed tetrahedral protein cage — [RCSB page](https://www.rcsb.org/structure/8UF0)). Stages 1–4 below (query → download → geometry → isolate) need nothing but `pip install -e .`; the output shown for them is real, captured by actually running each command. Stages 5–7 are shown as the exact commands to run next, but need RFdiffusion/ProteinMPNN/a structure-prediction backend installed and configured first — see the main README's **Installation**/**Configuration** sections — so they aren't re-run here.

Run everything from a fresh, empty project directory (anywhere outside the symbro checkout itself — `.symbro/`, `temporary_files/`, etc. all get created relative to wherever you run `symbro` from).

## 1–4: query → download → geometry → isolate

```bash
mkdir my_first_symbro_project && cd my_first_symbro_project

# 1. Look up this one entry directly by ID (skips the broader --symmetry/
#    --resolution-max search criteria entirely -- good for a first run,
#    or any time you already know exactly which structure you want)
symbro query --entry-id 8UF0
```
```
✓ Found 1 candidate(s).
  Saved to .symbro/candidates.pkl (preview: candidates.csv)
  Next: symbro download
```

```bash
# 2. Download it
symbro download
```
```
✓ Downloaded 1 structure(s).
  Saved to .symbro/downloaded.pkl (preview: downloaded.csv)
  Next: symbro geometry --symmetry-type <e.g. C3>
```

```bash
# 3. First pass: see what symmetry is actually present before committing to one
symbro geometry
```
```
✓ Symmetry types detected:
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ symmetry_type ┃ assemblies ┃ components ┃ total_axis_count ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ C2            │          1 │          1 │                4 │
│ C3            │          1 │          2 │                8 │
└───────────────┴────────────┴────────────┴──────────────────┘
  "components" > "assemblies" for a symmetry_type means at least one
  assembly has more than one structurally distinct component at that
  order -- e.g. a two-protein cage -- each of which will get its own
  row/file downstream.
  Saved to .symbro/geometry.pkl (preview: geometry.csv)
  Next: re-run with --symmetry-type on one of the above, e.g.
  symbro geometry --symmetry-type C2
```

8UF0's tetrahedral (T) symmetry contains both a C2 axis and a C3 axis — and its C3 axis is where the `components: 2` shows up, since `T33-ml23` is built from two structurally distinct trimeric components. We'll take the more interesting C3 path here; see `02_flags_and_narrowing` for how `--component-id` narrows a two-component case like this one down to a single component.

```bash
# 4. Narrow to C3 -- this also computes orientation + termini secondary
#    structure, which the broad first pass above doesn't
symbro geometry --symmetry-type C3
```
```
✓ 2 assembly/assemblies with C3 symmetry.
  Saved to .symbro/geometry.pkl (preview: geometry.csv)
  Next: symbro isolate
```

```bash
# 5. Extract each component's ring structure -- the files RFdiffusion needs next
symbro isolate
```
```
✓ Extracted 2 ring structure(s).
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ assembl… ┃ symmetr… ┃ compone… ┃ chain_gr… ┃ recomme… ┃ filepath  ┃ chain_r… ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ 8UF0-1   │ C3       │ 0        │ A, A-5,   │ 10, 17   │ temporar… │ A-5=B,   │
│          │          │          │ A-9       │          │           │ A-9=C    │
│ 8UF0-1   │ C3       │ 1        │ B, B-12,  │ 9, 14    │ temporar… │ B=A,     │
│          │          │          │ B-6       │          │           │ B-12=B,  │
│          │          │          │           │          │           │ B-6=C    │
└──────────┴──────────┴──────────┴───────────┴──────────┴───────────┴──────────┘
  Saved to .symbro/rings.pkl (preview: rings.csv)
  Next: symbro rfdiffusion
```

Two rows because 8UF0's C3 axis has two structurally distinct components (`component_id` 0 and 1) — `symbro isolate` extracted one ring PDB per component, each ready for RFdiffusion on its own. Check `temporary_subunits/` — you should have two real `.pdb` files there now.

## 5–7: the compute-heavy stages (commands only — need real infrastructure)

These need RFdiffusion, ProteinMPNN, and a structure-prediction backend (Boltz/AF2/AF3) actually installed, plus `installation.yaml` pointing at them (`cp installation.example.yaml installation.yaml` in your project root, then edit it — see the main README).

```bash
# 6. Submit RFdiffusion -- one job per (assembly, component) row from isolate,
#    so this submits 2 jobs for 8UF0's two C3 components. Blocks until done
#    unless you're on backend="slurm" and pass --detach (see 03_slurm_detach_workflow).
symbro rfdiffusion

# 7. Run ProteinMPNN against each assembly's best RFdiffusion design(s)
symbro pmpnn

# 8. Fold candidates back and screen by self-consistency (RMSD/pLDDT)
symbro predict

# Start fresh for your next project
symbro clean
```

A `run.sh` in this folder chains steps 1–5 (the part that's actually been verified above) so you can copy the whole thing instead of retyping it.
