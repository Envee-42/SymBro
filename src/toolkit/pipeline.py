"""
pipeline.py — the parameterized orchestration layer between the toolkit's
individual modules (query, download, geometry, isolate, ...) and any
front end that wants to drive them: cli.py today, a NiceGUI app later.

Each stage function here does three things, always in the same order:
  1. load its input from the previous stage's checkpoint, if not handed
     one directly (so `run_download()` with no arguments just picks up
     wherever `run_query()` last left off);
  2. call the real module-level function(s) that do the actual work —
     this file adds no science of its own, it only wires stages together;
  3. save its own output as a checkpoint, so the NEXT stage (or a human,
     re-running this same stage later with different arguments) can pick
     up from here without redoing everything before it.

Checkpoints live in a small state directory (".symbro/" by default, next
to wherever you're running from — override with state_dir= on any
function, or --state-dir on the CLI). Each stage writes TWO files:

  <stage>.pkl   the authoritative checkpoint, read back by later stages.
                Pickled rather than CSV on purpose: several stages carry
                real Python objects in their columns (chain_groups
                tuples, orientation vectors, chain-rename dicts) that a
                CSV round-trip would flatten into strings and quietly
                break the next stage. Pickle preserves them exactly.
  <stage>.csv   a human-readable preview of the same data, best-effort
                stringified, for opening in a text editor or Excel. The
                pipeline never reads this file back — it's for you, not
                for the next command.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import pandas as pd

# NOTE: every toolkit.* submodule import below is deliberately LOCAL to the
# function that needs it, not up here at module level. toolkit.query pulls
# in `rcsbapi`, which fetches a schema from search.rcsb.org the moment it's
# imported -- if that import happened at the top of this file, EVERY
# command (including `symbro --help`, or `symbro download` which has
# nothing to do with querying RCSB) would require internet access just to
# start. Deferring it here means only `run_query()` -- and therefore only
# `symbro query` -- ever pays that cost.

DEFAULT_STATE_DIR = ".symbro"

CANDIDATES_STAGE = "candidates"
DOWNLOADED_STAGE = "downloaded"
GEOMETRY_STAGE = "geometry"
ISOLATE_STAGE = "rings"


class StageNotFoundError(FileNotFoundError):
    """Raised when a stage function needs a previous checkpoint that doesn't exist yet."""

    def __init__(self, stage: str, state_dir: str, needed_by: str):
        self.stage = stage
        self.state_dir = state_dir
        self.needed_by = needed_by
        super().__init__(
            f"No '{stage}' checkpoint found in '{state_dir}/'. "
            f"{needed_by} needs that first — run the earlier stage, "
            f"or pass its result in directly."
        )


def _checkpoint_paths(stage: str, state_dir: str) -> tuple[str, str]:
    return (
        os.path.join(state_dir, f"{stage}.pkl"),
        os.path.join(state_dir, f"{stage}.csv"),
    )


def save_checkpoint(df: pd.DataFrame, stage: str, state_dir: str = DEFAULT_STATE_DIR) -> str:
    """Writes <stage>.pkl (authoritative) and <stage>.csv (human preview). Returns the .pkl path."""
    os.makedirs(state_dir, exist_ok=True)
    pkl_path, csv_path = _checkpoint_paths(stage, state_dir)
    df.to_pickle(pkl_path)
    try:
        df.to_csv(csv_path, index=False)
    except Exception:
        # The .csv is a convenience preview only -- never let a column that
        # doesn't stringify cleanly (rare, but possible) block the real save.
        pass
    return pkl_path


def load_checkpoint(stage: str, state_dir: str = DEFAULT_STATE_DIR, needed_by: str = "This step") -> pd.DataFrame:
    """Reads <stage>.pkl. Raises StageNotFoundError (a friendly, catchable error) if it's missing."""
    pkl_path, _ = _checkpoint_paths(stage, state_dir)
    if not os.path.exists(pkl_path):
        raise StageNotFoundError(stage, state_dir, needed_by)
    return pd.read_pickle(pkl_path)


def checkpoint_exists(stage: str, state_dir: str = DEFAULT_STATE_DIR) -> bool:
    pkl_path, _ = _checkpoint_paths(stage, state_dir)
    return os.path.exists(pkl_path)


# ----------------------------------------------------------------------
# Stage 1: query
# ----------------------------------------------------------------------

def _build_criteria(
    symmetry: Optional[Sequence[str]],
    entry_id: Optional[Sequence[str]],
    resolution_range: Optional[tuple],
    description: Optional[str],
    extra_criteria: Optional[Sequence[dict]],
) -> list:
    from toolkit import query as _query

    criteria = []
    if symmetry:
        criteria.append(_query.build_query("symmetry", list(symmetry)))
    if entry_id:
        criteria.append(_query.build_query("entry_id", list(entry_id)))
    if resolution_range:
        criteria.append(_query.build_query("resolution", tuple(resolution_range), operator="range"))
    if description:
        criteria.append(_query.build_query("description", description, operator="contains_words"))
    for c in extra_criteria or []:
        criteria.append(_query.build_query(c["attribute"], c["value"], c.get("operator", "exact_match")))
    return criteria


def run_query(
    symmetry: Optional[Sequence[str]] = None,
    entry_id: Optional[Sequence[str]] = None,
    resolution_range: Optional[tuple] = None,
    description: Optional[str] = None,
    extra_criteria: Optional[Sequence[dict]] = None,
    match: str = "and",
    fetch_fields: Optional[Sequence[str]] = None,
    return_type: str = "assembly",
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Search RCSB PDB for candidates. At least one of symmetry / entry_id /
    resolution_range / description / extra_criteria must be given, or
    every criterion in the search would be unconstrained.

    Saves the result to <state_dir>/candidates.{pkl,csv} and returns it
    (empty DataFrame if nothing matched -- not an error).
    """
    # Checked BEFORE importing toolkit.query on purpose: that import alone
    # triggers rcsbapi's network fetch of its schema, so a call that was
    # always going to fail validation shouldn't pay for a network round
    # trip first.
    if not (symmetry or entry_id or resolution_range or description or extra_criteria):
        raise ValueError(
            "No search criteria given -- provide at least one of symmetry, "
            "entry_id, resolution_range, description, or extra_criteria."
        )

    from toolkit import query as _query

    criteria = _build_criteria(symmetry, entry_id, resolution_range, description, extra_criteria)
    df = _query.query_candidates(
        search_criteria=criteria,
        fetch_fields=list(fetch_fields) if fetch_fields else None,
        mode=match,
        return_type=return_type,
    )
    save_checkpoint(df, CANDIDATES_STAGE, state_dir)
    return df


# ----------------------------------------------------------------------
# Stage 2: download
# ----------------------------------------------------------------------

def run_download(
    candidates_df: Optional[pd.DataFrame] = None,
    data_dir: Optional[str] = None,
    overwrite: bool = False,
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Downloads every candidate's structure file. If candidates_df isn't
    given, loads it from <state_dir>/candidates.pkl (i.e. run_query()'s
    last output).

    Saves the result to <state_dir>/downloaded.{pkl,csv} and returns it.
    """
    from toolkit import download as _download

    if candidates_df is None:
        candidates_df = load_checkpoint(CANDIDATES_STAGE, state_dir, needed_by="Downloading structures")
    df = _download.download_candidates(candidates_df, data_dir=data_dir, overwrite=overwrite)
    save_checkpoint(df, DOWNLOADED_STAGE, state_dir)
    return df


# ----------------------------------------------------------------------
# Stage 3: geometry (rings, always; orientation + termini SS, if symmetry_type given)
# ----------------------------------------------------------------------

def run_geometry(
    downloaded_df: Optional[pd.DataFrame] = None,
    symmetry_type: Optional[str] = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Detects symmetry rings in every downloaded structure. If symmetry_type
    is given (e.g. "C3"), also computes orientation and termini secondary
    structure for that one symmetry order and merges them in -- narrowing
    to exactly one order is required for those two steps, same contract
    the underlying geometry modules use.

    If symmetry_type is omitted, this only runs ring detection (useful as
    a first pass to see what symmetry types are even present before
    committing to one) and the returned DataFrame carries every order
    found, unmerged.

    If downloaded_df isn't given, loads it from
    <state_dir>/downloaded.pkl. Saves the result to
    <state_dir>/geometry.{pkl,csv} and returns it.
    """
    from toolkit.geometry import orientation as _orientation
    from toolkit.geometry import rings as _rings
    from toolkit.geometry import structure as _structure

    if downloaded_df is None:
        downloaded_df = load_checkpoint(DOWNLOADED_STAGE, state_dir, needed_by="Geometry analysis")

    rings_df = _rings.from_structure(downloaded_df)

    if symmetry_type is None or rings_df.empty:
        save_checkpoint(rings_df, GEOMETRY_STAGE, state_dir)
        return rings_df

    orientation_df = _orientation.from_rings(rings_df, downloaded_df, symmetry_type)
    if orientation_df.empty:
        save_checkpoint(orientation_df, GEOMETRY_STAGE, state_dir)
        return orientation_df

    termini_df = _structure.from_rings(rings_df, downloaded_df, symmetry_type)
    merged = orientation_df.merge(
        termini_df[["assembly_id", "symmetry_type", "termini_ss"]],
        on=["assembly_id", "symmetry_type"],
        how="left",
    )
    save_checkpoint(merged, GEOMETRY_STAGE, state_dir)
    return merged


def detected_symmetry_types(rings_df: pd.DataFrame) -> pd.Series:
    """Counts of assemblies found per symmetry_type -- for printing a friendly summary."""
    if rings_df.empty or "symmetry_type" not in rings_df.columns:
        return pd.Series(dtype=int)
    return rings_df.groupby("symmetry_type")["assembly_id"].nunique().sort_values(ascending=False)


# ----------------------------------------------------------------------
# Stage 4: isolate
# ----------------------------------------------------------------------

def run_isolate(
    geometry_df: Optional[pd.DataFrame] = None,
    downloaded_df: Optional[pd.DataFrame] = None,
    symmetry_type: Optional[str] = None,
    file_format: str = "pdb",
    output_dir: Optional[str] = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Extracts one ring PDB (or mmCIF) per assembly into temporary_subunits/
    (or output_dir) -- the files RFdiffusion needs next. If geometry_df /
    downloaded_df aren't given, loads them from
    <state_dir>/geometry.pkl and <state_dir>/downloaded.pkl.

    Saves the result to <state_dir>/rings.{pkl,csv} and returns it.
    """
    from toolkit import isolate as _isolate

    if geometry_df is None:
        geometry_df = load_checkpoint(GEOMETRY_STAGE, state_dir, needed_by="Isolating rings")
    if downloaded_df is None:
        downloaded_df = load_checkpoint(DOWNLOADED_STAGE, state_dir, needed_by="Isolating rings")

    df = _isolate.from_rings(
        geometry_df, downloaded_df, symmetry_type=symmetry_type,
        output_dir=output_dir, file_format=file_format,
    )
    save_checkpoint(df, ISOLATE_STAGE, state_dir)
    return df
