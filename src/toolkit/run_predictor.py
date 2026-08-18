"""
run_predictor.py — runs structural self-consistency screening (see
structure_prediction.py) against an ALREADY-COMPLETED ProteinMPNN run's
output, without re-running query/download/geometry/RFdiffusion/
ProteinMPNN from scratch.

Run this FROM the project root (same convention every `symbro` command
already requires — .symbro/ and temporary_simulations/ are both resolved
relative to cwd) — no need to touch pipeline.py or re-run any of its
earlier stages.

  sequences_df : reloaded via pipeline.load_checkpoint("pmpnn") --
      .symbro/pmpnn.pkl, written by pipeline.mpnn() / `symbro pmpnn`'s
      own last line. (An earlier version of this script instead looked
      for a "sequences.csv" inside the ProteinMPNN out_folder -- that
      file is never actually written by pipeline.py/pmpnn.py; fixed here
      to reload from the real checkpoint every other stage already
      uses.) Narrow to one assembly with ASSEMBLY_ID below if the
      checkpoint has rows from more than one run.

  design_paths : the RFdiffusion design PDBs ProteinMPNN actually read
      as input, recovered from "{mpnn out_folder}/_mpnn_inputs/pdbs/*.pdb"
      -- pmpnn.py's own _stage_input_pdbs() copies ProteinMPNN's exact
      input PDBs there before every run, and only clears/overwrites that
      folder at the START of the NEXT submit() against the SAME
      out_folder (never after a run completes). pipeline.py never passes
      a per-assembly out_folder today, so if you've run `symbro pmpnn`
      against MORE THAN ONE assembly since the last `symbro clean`, only
      the LAST one's designs are still staged here -- either narrow
      ASSEMBLY_ID to that one, or re-run `symbro pmpnn --assembly-id ...`
      for whichever assembly you actually want to screen right before
      running this script.
"""

import glob
import os

from toolkit import config, pipeline, pmpnn, structure_prediction

# Wherever `symbro pmpnn` printed its designs as being written --
# normally temporary_simulations/mpnn_designs, pmpnn.py's own default
# out_folder (see prepare_mpnn_job()'s out_folder=None fallback).
MPNN_OUT_FOLDER = os.path.join("temporary_simulations", "mpnn_designs")

# Narrow the reloaded .symbro/pmpnn.pkl checkpoint to one assembly (and,
# for a multi-component assembly, one component) -- leave both None to
# use every row in the checkpoint (only safe if it's a single assembly/
# component run -- see this module's own docstring above on staged-PDB
# reuse).
ASSEMBLY_ID = None      # e.g. "4V6B-5"
COMPONENT_ID = None

# Which predictor to screen with -- "alphafold2", "boltz", or "af3" (see
# structure_prediction.py's own module docstring for the licensing/setup
# tradeoffs). Leave as None to use installation.yaml's own
# structure_prediction.default instead of naming one here.
PREDICTOR = "boltz"

TOP_N = 3            # pmpnn.select_best_designs()'s top_n -- the cheap pre-filter
MAX_RMSD = 2.0       # select_validated_designs()'s max_rmsd (Angstrom)
MIN_PLDDT = 70.0     # select_validated_designs()'s min_plddt

# AF3 only -- ignored for every other PREDICTOR value. See af3.py's own
# module docstring, and installation.yaml's af3: section, before setting
# these.
AF3_MODEL_DIR = None
AF3_DB_DIR = None
AF3_TERMS_ACKNOWLEDGED = False


def _load_design_paths(mpnn_out_folder: str) -> list:
    staged_dir = os.path.join(mpnn_out_folder, "_mpnn_inputs", "pdbs")
    paths = sorted(glob.glob(os.path.join(staged_dir, "*.pdb")))
    if not paths:
        raise FileNotFoundError(
            f"No staged input PDBs found under {staged_dir!r} (pmpnn.py's own "
            f"_stage_input_pdbs() writes this folder every time ProteinMPNN actually "
            f"runs) -- MPNN_OUT_FOLDER is probably pointed at the wrong run, or that "
            f"folder's been cleaned (`symbro clean`) since."
        )
    print(f"Found {len(paths)} reference design PDB(s) under {staged_dir!r}.")
    return paths


def main():
    sequences_df = pipeline.load_checkpoint(pipeline.PMPNN_STAGE)
    print(f"Reloaded {len(sequences_df)} sequence row(s) from .symbro/{pipeline.PMPNN_STAGE}.pkl")

    if ASSEMBLY_ID is not None:
        sequences_df = sequences_df[sequences_df["assembly_id"] == ASSEMBLY_ID]
    if COMPONENT_ID is not None:
        sequences_df = sequences_df[sequences_df["component_id"] == COMPONENT_ID]
    if sequences_df.empty:
        raise ValueError(
            f"No rows left after filtering to assembly_id={ASSEMBLY_ID!r}, "
            f"component_id={COMPONENT_ID!r} -- check .symbro/pmpnn.csv for what's actually there."
        )

    design_paths = _load_design_paths(MPNN_OUT_FOLDER)

    shortlist = pmpnn.select_best_designs(sequences_df, top_n=TOP_N)
    print(f"\nShortlist ({len(shortlist)} candidate(s), top {TOP_N} per design by ProteinMPNN's own score):")
    print(shortlist[["source_pdb", "sequence", "global_score", "rank"]])

    cfg = config.load_installation_config()
    run_kwargs = dict(max_rmsd=MAX_RMSD, min_plddt=MIN_PLDDT)
    if PREDICTOR in ("af3", "alphafold3"):
        run_kwargs.update(
            model_dir=AF3_MODEL_DIR, db_dir=AF3_DB_DIR, terms_acknowledged=AF3_TERMS_ACKNOWLEDGED,
        )

    winners = structure_prediction.run(PREDICTOR, shortlist, design_paths, config=cfg, **run_kwargs)
    print(f"\n{len(winners)} candidate(s) passed self-consistency screening "
          f"(max_rmsd={MAX_RMSD}, min_plddt={MIN_PLDDT}):")
    print(winners)

    out_path = os.path.join(MPNN_OUT_FOLDER, f"validated_designs_{PREDICTOR or 'default'}.csv")
    winners.to_csv(out_path, index=False)
    print(f"\nWrote {len(winners)} validated design(s) to {out_path}")


if __name__ == "__main__":
    main()
