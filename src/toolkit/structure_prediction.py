"""
structure_prediction.py — one place to pick WHICH structure-prediction
backend runs ProteinMPNN candidates' self-consistency check: AlphaFold2
(via ColabFold, alphafold2.py), Boltz (boltz.py), or AlphaFold3 (af3.py).
This module IS the "the user should be able to easily select either
tool" layer — everything it does is dispatch to one of those three, each
of which is also fully usable directly and independently (import
boltz/alphafold2/af3 yourself if you only ever want one and don't want
this indirection).

QUICKSTART:

    from toolkit import pmpnn, structure_prediction

    shortlist = pmpnn.select_best_designs(sequences_df, top_n=3)

    # pick a predictor by name at the call site...
    winners = structure_prediction.run("boltz", shortlist, design_paths, config=cfg)

    # ...or once, in installation.yaml:
    #   structure_prediction:
    #     default: boltz
    winners = structure_prediction.run(None, shortlist, design_paths, config=cfg)

Which one to pick — see each module's own docstring for the full
picture, this is the short version:

  "alphafold2"  ColabFold running AlphaFold2. Fully permissive licensing
                (code Apache 2.0, weights CC BY 4.0, ColabFold + its
                MMseqs2 MSA search both MIT) — commercial-safe,
                redistributable, no extra acknowledgement needed. Uses a
                free hosted MSA search API by default.
  "boltz"       Fully permissive too (code AND weights MIT, explicitly
                "for both academic and commercial purposes" per its own
                README) — the licensing-cleanest option. Also uses a
                free hosted MSA search API by default.
  "af3"         AlphaFold3. Source code is Apache 2.0, but the model
                WEIGHTS are restricted to non-commercial use, must be
                requested directly from Google (not bundled here), and
                cannot be redistributed. submit()/run() refuse to run
                unless you pass terms_acknowledged=True. Also needs a
                much larger local database download than the other two
                unless you disable its data pipeline. See af3.py's
                module docstring before reaching for this one.

Aliases ("af2"/"colabfold" -> alphafold2, "alphafold3" -> af3) exist so
either the tool name or the model name resolves to the right module.
"""

from typing import Optional, Sequence

import pandas as pd

from toolkit import af3, alphafold2, boltz
from toolkit.config import get_tool_config

PREDICTORS = {
    "alphafold2": alphafold2,
    "af2": alphafold2,
    "colabfold": alphafold2,
    "boltz": boltz,
    "af3": af3,
    "alphafold3": af3,
}


def _resolve(predictor: Optional[str], config: Optional[dict]):
    """predictor=None falls back to config["structure_prediction"]["default"]
    — see module docstring's "or once, in installation.yaml" example.
    Raises ValueError (listing every valid name) if neither resolves to
    a known predictor, rather than silently picking one."""
    if predictor is None:
        predictor = get_tool_config(config or {}, "structure_prediction").get("default")
    if not predictor:
        raise ValueError(
            "no predictor given, and no structure_prediction.default set in your installation "
            f"config — pass predictor=... explicitly, one of {sorted(set(PREDICTORS))}."
        )
    module = PREDICTORS.get(predictor.lower())
    if module is None:
        raise ValueError(
            f"unknown predictor {predictor!r} — choose one of {sorted(set(PREDICTORS))} (see "
            f"this module's docstring for what each wraps)."
        )
    return module


def run(predictor: Optional[str], selected_df: pd.DataFrame, design_paths: Sequence[str], config: Optional[dict] = None, **kwargs) -> pd.DataFrame:
    """Dispatches to <predictor>.run(selected_df, design_paths, config=config, **kwargs)
    — see alphafold2.run()/boltz.run()/af3.run() for the shared shape
    (backend=, poll_interval=, max_rmsd=, min_plddt=), and af3.run()
    specifically for its required model_dir=/terms_acknowledged= kwargs."""
    return _resolve(predictor, config).run(selected_df, design_paths, config=config, **kwargs)


def prepare_self_consistency_job(predictor: Optional[str], selected_df: pd.DataFrame, design_paths: Sequence[str], config: Optional[dict] = None, **kwargs):
    return _resolve(predictor, config).prepare_self_consistency_job(selected_df, design_paths, **kwargs)


def submit(predictor: Optional[str], job, config: Optional[dict] = None, **kwargs):
    return _resolve(predictor, config).submit(job, config=config, **kwargs)


def poll_status(predictor: Optional[str], run_handle, config: Optional[dict] = None) -> dict:
    return _resolve(predictor, config).poll_status(run_handle)


def cancel(predictor: Optional[str], run_handle, config: Optional[dict] = None) -> None:
    return _resolve(predictor, config).cancel(run_handle)
