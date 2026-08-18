# 04. Analyzing pipeline results

`symbro_analysis.ipynb` — a local Jupyter notebook for making sense of a completed
run, once you have validated candidates sitting in `.symbro/predict.pkl`. Unlike the
two Colab notebooks (`src/toolkit/symbro_rfdiffusion_colab.ipynb` and
`symbro_full_pipeline_colab.ipynb`), this one doesn't run any pipeline stage itself —
it only reads checkpoints a real run already produced, so it needs no GPU.

## What you need to actually run it

- `symbro` itself installed (`pip install -e .` from the repo — the same install you
  already have to have run the CLI) plus `pandas`/`matplotlib`, both already base
  dependencies. The notebook calls `pipeline.join_predict_with_pmpnn()` directly for
  its checkpoint join rather than reimplementing it (see Section 2) -- that's the only
  reason it needs `symbro` importable rather than just unpickling files by hand.
  `py3Dmol`/`ipywidgets` are optional, for the inline structure preview only.
- A completed run: at minimum `.symbro/predict.pkl` with at least one validated
  candidate (i.e. you've already run `symbro predict` and it found something that
  passed screening).
- Run the notebook **from that same project directory** — the one containing
  `.symbro/`, not from inside this `examples/` folder. Copy `symbro_analysis.ipynb`
  there, or point Jupyter's working directory at it.

## What it shows

- A self-consistency RMSD vs. mean pLDDT scatter of your validated candidates —
  once colored by assembly, once colored by ProteinMPNN's own `global_score` (joined
  in from `.symbro/pmpnn.pkl`, since `predict.pkl` doesn't carry it directly).
- A per-assembly / per-symmetry-type breakdown table (candidate counts, best/mean
  RMSD and pLDDT).
- An adjustable re-thresholding view, in case you want a stricter shortlist than
  whatever `--max-rmsd`/`--min-plddt` you originally ran `symbro predict` with.
- A sorted shortlist table, and an export of it (CSV + structure files) to an
  `analysis_shortlist/` folder.

See the notebook's own intro cell for the metric definitions/thresholds used
(self-consistency RMSD, pLDDT, ProteinMPNN `global_score`) and — importantly — a real
limitation worth knowing before you read too much into the plots: `predict.pkl` only
ever contains candidates that *passed* screening, not the full attempted pool, so this
notebook can't show you near-misses or a true pass rate. See the notebook's footer for
what that would take to add.

## How this was verified

The notebook's checkpoint join (recovering each validated candidate's sequence/score
from `pmpnn.pkl`, including the NaN-safe `(assembly_id, component_id)` handling that
join needs) now goes through `pipeline.join_predict_with_pmpnn()` — the same function
`symbro codon` uses, covered directly by `tests/test_codon.py`'s own join tests
(including a regression test for a real dtype bug that join hit once). The notebook's
remaining logic (the plots, breakdown table, shortlist/export) was prototyped and run
for real against synthetic-but-schema-correct `.symbro/*.pkl` fixtures (matching
`pipeline.py`'s actual column shapes exactly), then the notebook itself was executed
top-to-bottom with `jupyter nbconvert --execute` against those same fixtures and its
outputs inspected — not just written and assumed to work, and re-verified after the
switch to the shared join helper. What *wasn't* verified: the actual numbers a real
RFdiffusion/ProteinMPNN/structure-predictor run would produce, since that needs a GPU
this review didn't have access to (same caveat as `01_full_walkthrough`'s
RFdiffusion/ProteinMPNN/predict commands).
