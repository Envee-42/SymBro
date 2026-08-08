"""
run_predictor.py — runs structural self-consistency screening (see
structure_prediction.py) against an ALREADY-COMPLETED ProteinMPNN run's
output, without re-running query/download/geometry/RFdiffusion/
ProteinMPNN from scratch.

pipeline.py is a straight-through script with no checkpointing, so
re-running it top to bottom just to add this one downstream step would
redo everything expensive (structure queries, RFdiffusion, ProteinMPNN)
for no reason. Everything this script needs is already sitting on disk
from that earlier run:

  sequences_df : reloaded straight from "{mpnn out_folder}/sequences.csv"
      — written by pipeline.py's own last few lines, nothing to
      recompute.
  design_paths : the RFdiffusion design PDBs ProteinMPNN actually read
      as input. Two ways this script can find them, tried in order:
        1. A "design_path" column right in sequences.csv, if this run's
           pipeline.py already included the small addition that writes
           one (source_pdb -> the real path, via mpnn_job.input_pdbs) —
           the robust, permanent fix, and what any FUTURE run will have.
        2. Falling back to "{mpnn out_folder}/_mpnn_inputs/pdbs/*.pdb" —
           pmpnn.py's own _stage_input_pdbs() copies ProteinMPNN's exact
           input PDBs there before every run and only clears that folder
           at the START of a fresh submit() against the SAME out_folder
           (never after a run completes) — so for a sequences.csv
           written BEFORE the design_path column existed, these staged
           copies are still the correct, exact files, just not linked
           to sequences.csv explicitly yet. This fallback breaks only if
           you've since resubmitted a DIFFERENT ProteinMPNN job reusing
           the same out_folder, which overwrites pdbs/ with that newer
           job's inputs — pipeline.py's design_path column exists
           specifically so this stops being a concern going forward.

Just edit the settings below and run — no need to touch pipeline.py or
re-run any of its earlier stages.
"""

import glob
import os

import pandas as pd

from toolkit import config, pmpnn, structure_prediction

# Wherever pipeline.py's own "mpnn_run.job.out_folder" pointed — it's
# the folder pipeline.py printed when it wrote "Wrote N sequences to
# .../sequences.csv"; normally temporary_simulations/mpnn_designs/.
MPNN_OUT_FOLDER = r"C:\Users\youruser\4_Projects\symbro\temporary_simulations\mpnn_designs"

# Which predictor to screen with — "alphafold2", "boltz", or "af3" (see
# structure_prediction.py's own module docstring for the licensing/setup
# tradeoffs). Leave as None to use installation.yaml's own
# structure_prediction.default instead of naming one here.
PREDICTOR = "boltz"

TOP_N = 3          # pmpnn.select_best_designs()'s top_n — the cheap pre-filter
MAX_RMSD = 2.0      # select_validated_designs()'s max_rmsd (Angstrom)
MIN_PLDDT = 70.0    # select_validated_designs()'s min_plddt


def _load_design_paths(sequences_df: pd.DataFrame, mpnn_out_folder: str) -> list:
    if "design_path" in sequences_df.columns:
        paths = sorted(sequences_df["design_path"].dropna().unique().tolist())
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"sequences.csv's own design_path column names {len(missing)} file(s) that "
                f"no longer exist on disk (e.g. {missing[0]!r}) — the run these designs came "
                f"from may have been cleaned up. Falling back isn't safe here since the "
                f"column exists but is stale; re-run pipeline.py's RFdiffusion/ProteinMPNN "
                f"stages if these files were genuinely deleted."
            )
        print(f"Found {len(paths)} reference design PDB(s) via sequences.csv's own design_path column.")
        return paths

    staged_dir = os.path.join(mpnn_out_folder, "_mpnn_inputs", "pdbs")
    paths = sorted(glob.glob(os.path.join(staged_dir, "*.pdb")))
    if not paths:
        raise FileNotFoundError(
            f"sequences.csv has no design_path column, and no staged input PDBs were found "
            f"under {staged_dir!r} either (pmpnn.py's own _stage_input_pdbs() writes this "
            f"folder every time ProteinMPNN actually runs) — MPNN_OUT_FOLDER is probably "
            f"pointed at the wrong run, or that folder's been cleaned up since."
        )
    print(
        f"sequences.csv predates the design_path column — recovered {len(paths)} reference "
        f"design PDB(s) from {staged_dir!r} instead (pmpnn.py's staged ProteinMPNN inputs)."
    )
    return paths


def main():
    csv_path = os.path.join(MPNN_OUT_FOLDER, "sequences.csv")
    sequences_df = pd.read_csv(csv_path)
    print(f"Reloaded {len(sequences_df)} sequences from {csv_path}")

    design_paths = _load_design_paths(sequences_df, MPNN_OUT_FOLDER)

    shortlist = pmpnn.select_best_designs(sequences_df, top_n=TOP_N)
    print(f"\nShortlist ({len(shortlist)} candidate(s), top {TOP_N} per design by ProteinMPNN's own score):")
    print(shortlist[["source_pdb", "sequence", "global_score", "rank"]])

    cfg = config.load_installation_config()
    winners = structure_prediction.run(
        PREDICTOR, shortlist, design_paths, config=cfg,
        max_rmsd=MAX_RMSD, min_plddt=MIN_PLDDT,
    )
    print(f"\n{len(winners)} candidate(s) passed self-consistency screening (max_rmsd={MAX_RMSD}, min_plddt={MIN_PLDDT}):")
    print(winners)

    out_path = os.path.join(MPNN_OUT_FOLDER, f"validated_designs_{PREDICTOR or 'default'}.csv")
    winners.to_csv(out_path, index=False)
    print(f"\nWrote {len(winners)} validated design(s) to {out_path}")


if __name__ == "__main__":
    main()
