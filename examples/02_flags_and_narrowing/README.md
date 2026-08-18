# 02 — flags and narrowing

Once a candidate set has more than one interesting thing in it — more than one assembly, a multi-component cage, more RFdiffusion designs than you want to fold — these are the flags that narrow things down. This reuses `01_full_walkthrough`'s same real 8UF0 download rather than fetching anything new; run `01` first (through at least `symbro download`) if you want to follow along with real output.

## `--symmetry-type` — pick one axis order out of several present

8UF0 has both a C2 axis and a C3 axis (see `01`'s README for why). A broad `symbro geometry` call shows both; `--symmetry-type` narrows to one and additionally computes orientation + termini secondary structure, which the broad pass skips. This is real, captured output:

```bash
symbro geometry --symmetry-type C2
```
```
✓ 1 assembly/assemblies with C2 symmetry.
┏━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━┓
┃ asse… ┃ symm… ┃ chai… ┃ mean… ┃ comp… ┃ axis… ┃ reco… ┃ mean… ┃ ori… ┃ term… ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━┩
│ 8UF0… │ C2    │ A,    │ 29.04 │ 0     │ 4     │ 10,   │ 78.78 │ A/A… │ A/E/… │
│       │       │ A-5   │       │       │       │ 17    │       │ A-5… │ A-5/… │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴──────┴───────┘
```

Only one component here — unlike the C3 case in `01`, which had two — because 8UF0's C2 axis isn't where its two structurally distinct components diverge.

## `--component-id` — narrow a multi-component assembly to one component

`symbro geometry --symmetry-type C3` on 8UF0 produces two rows (`component_id` 0 and 1 — see `01`). Without `--component-id`, `symbro isolate` extracts both. To get just one:

```bash
symbro isolate --component-id 0
```
```
✓ Extracted 1 ring structure(s).
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ assembl… ┃ symmetr… ┃ compone… ┃ chain_gr… ┃ recomme… ┃ filepath  ┃ chain_r… ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ 8UF0-1   │ C3       │ 0        │ A-12,     │ 10, 17   │ temporar… │ A-12=A,  │
│          │          │          │ A-2, A-7  │          │           │ A-2=B,   │
│          │          │          │           │          │           │ A-7=C    │
└──────────┴──────────┴──────────┴───────────┴──────────┴───────────┴──────────┘
```

(The specific chain letters picked — `A-12`/`A-2`/`A-7` here vs. `A`/`A-5`/`A-9` in `01`'s run — are symmetry-mate selections and can vary run to run; the row count and structure are what to rely on.)

## `--assembly-id` / `--component-id` on the later stages

`symbro rfdiffusion`, `symbro pmpnn`, and `symbro predict` all accept the same `--assembly-id`/`--component-id` pair once you have more than one candidate in flight, to run just one instead of every row in the previous stage's checkpoint:

```bash
symbro rfdiffusion --assembly-id 8UF0-1
symbro pmpnn --assembly-id 8UF0-1 --component-id 0
symbro predict --assembly-id 8UF0-1 --component-id 0
```

## `--select` — bypass ranking entirely for `symbro pmpnn`

`symbro pmpnn` normally submits its own top-N RFdiffusion designs by pLDDT (`--top-n`, `--min-plddt`). To instead hand it exact design PDB paths yourself — e.g. ones you've eyeballed in a structure viewer — use `--select` (repeatable), which requires `--assembly-id` (and `--component-id` too, for a multi-component assembly like 8UF0's C3 axis):

```bash
symbro pmpnn --assembly-id 8UF0-1 --component-id 0 \
    --select temporary_subunits/rfdiffusion_out/8UF0-1_c0_design_3.pdb \
    --select temporary_subunits/rfdiffusion_out/8UF0-1_c0_design_7.pdb
```

## `--top-n` / `--min-plddt` — the default ranking, if you don't use `--select`

```bash
# submit only the top 5 RFdiffusion designs by pLDDT, and only if they also
# clear a mean pLDDT of 80
symbro pmpnn --top-n 5 --min-plddt 80
```

The same `--top-n` idea reappears in `symbro predict` (default 3), controlling the ProteinMPNN-side shortlist that actually gets folded and screened, independent of `symbro pmpnn`'s own `--top-n`.

## `--linker-min` / `--linker-max` — override RFdiffusion's diffused linker length

By default, `symbro rfdiffusion` uses each ring's own geometry-informed recommendation (the `recommended_linker_length` column visible in `01`'s and this file's `geometry`/`isolate` output above — e.g. `10, 17` for 8UF0's component 0). Passing both flags together overrides that with one fixed range applied to every job instead:

```bash
symbro rfdiffusion --linker-min 12 --linker-max 20
```

(`cli.py` requires both or neither — passing just one fails with a clear error rather than silently doing something unintended.)

## `--predictor` — choosing a structure-prediction backend per run

`symbro predict` defaults to whatever `installation.yaml`'s `structure_prediction.default` says, but any run can override it directly:

```bash
symbro predict --predictor boltz
symbro predict --predictor af2      # or alphafold2
symbro predict --predictor af3 --af3-terms-acknowledged   # see main README re: AF3's license terms
```
