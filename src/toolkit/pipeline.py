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
from typing import Optional, Sequence, Tuple, Union

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
RFDIFFUSION_STAGE = "rfdiffusion"
PMPNN_STAGE = "pmpnn"

# poll_status() states that mean "this job is done, one way or another" --
# shared by both the RFdiffusion and ProteinMPNN polling loops below.
_TERMINAL_STATES = ("completed", "completed_partial", "failed")


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


# ----------------------------------------------------------------------
# Stage 5: RFdiffusion (batch-per-assembly: one job per rings.pkl row)
# ----------------------------------------------------------------------

def _poll_all_rfdiffusion(rows: list, poll_interval: int = 20, timeout: Optional[int] = None, single_pass: bool = False) -> list:
    """
    Round-robin polls every row's "run" (an RFdiffusionRun) until each
    reaches a terminal state, mutating each row dict's "state"/
    "design_paths" in place as it goes -- also returns `rows` for
    convenience.

    single_pass=True (used by refresh_rfdiffusion_status(), i.e.
    `symbro status`) checks each not-yet-terminal row exactly once and
    returns immediately, no sleeping/looping -- a cheap "what's the
    latest?" refresh rather than a blocking wait.

    Every row's "run" is sanitized (run.process forced to None) before
    this returns, regardless of how polling ended -- a live
    subprocess.Popen (only ever present for backend="local"/
    "singularity"; "slurm" runs always have process=None already, see
    rfdiffusion.py's _submit_slurm()) can't be pickled, and every caller
    of this function is about to save the result to a checkpoint. A row
    still "running" when this happens (only possible for local/
    singularity, via a timeout -- SLURM jobs stay checkable through
    slurm_job_id regardless) prints a warning: its own OS process may
    still finish on its own, but this checkpoint loses the ability to
    poll or reattach to it -- `symbro status` only works for SLURM jobs.
    """
    import time as _time
    from dataclasses import replace

    from toolkit import rfdiffusion as _rfdiffusion

    start = _time.time()
    while True:
        pending = [r for r in rows if r["state"] not in _TERMINAL_STATES]
        if not pending:
            break
        for row in pending:
            status = _rfdiffusion.poll_status(row["run"])
            row["state"] = status["state"]
            row["design_paths"] = status["design_paths"]
            if status["state"] in _TERMINAL_STATES:
                print(f"  {row['assembly_id']}: {status['state']} "
                      f"({status['designs_written']}/{status['designs_expected']} designs)")
        if single_pass or all(r["state"] in _TERMINAL_STATES for r in rows):
            break
        if timeout is not None and (_time.time() - start) > timeout:
            still_running = [r["assembly_id"] for r in rows if r["state"] not in _TERMINAL_STATES]
            print(f"Timed out after {timeout}s -- still running: {still_running}. The job(s) keep "
                  f"running either way; SLURM jobs stay checkable with `symbro status` afterward.")
            break
        _time.sleep(poll_interval)

    for row in rows:
        if row["run"].process is not None:
            if row["state"] not in _TERMINAL_STATES:
                print(f"  Warning: {row['assembly_id']}'s local/singularity job is still running "
                      f"but is no longer trackable after this command exits -- its own process may "
                      f"finish on its own, but `symbro status` can't check on it (only SLURM jobs "
                      f"support that). Re-run with a longer --timeout, or use backend='slurm' for "
                      f"jobs that need to survive detaching.")
            row["run"] = replace(row["run"], process=None)
    return rows


def run_rfdiffusion(
    rings_df: Optional[pd.DataFrame] = None,
    assembly_id: Optional[str] = None,
    linker_length: Union[int, Tuple[int, int]] = (15, 25),
    num_designs: int = 10,
    diffuser_T: int = 50,
    backend: Optional[str] = None,
    detach: bool = False,
    poll_interval: int = 20,
    timeout: Optional[int] = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Submits one RFdiffusion job per assembly -- batch-per-assembly, the
    architecture decided for this stage: every assembly gets its own
    independent job rather than one giant multi-assembly job or one job
    per ring row (rings.pkl already has exactly one ring row per
    assembly, the main grouping). Narrow to a single assembly with
    assembly_id=.

    Every job is submitted FIRST (rfdiffusion.submit() dispatches
    without blocking), and only THEN polled -- this keeps multiple SLURM
    jobs running in parallel on the cluster's own scheduler, rather than
    waiting for assembly 1's job to finish before assembly 2's is even
    queued.

    detach=True returns immediately after submission, before any
    polling, so `symbro rfdiffusion --detach` doesn't block the
    terminal -- check progress later with `symbro status` (see
    refresh_rfdiffusion_status()). Only supported when the resolved
    backend is "slurm": a "local"/"singularity" run's RFdiffusionRun
    carries a live subprocess.Popen that can't survive being
    checkpointed and resumed in a later, separate process, whereas a
    SLURM run is tracked by its own job id (slurm_job_id) independent of
    any local process handle. Raises ValueError if detach=True is
    combined with a non-"slurm" backend.

    If rings_df isn't given, loads it from <state_dir>/rings.pkl (i.e.
    `symbro isolate`'s last output). Saves the result to
    <state_dir>/rfdiffusion.{pkl,csv} and returns it -- one row per
    assembly: assembly_id, symmetry_type, chain_groups, run (a
    picklable RFdiffusionRun -- see _poll_all_rfdiffusion()'s docstring
    for why .process is always None by the time this is saved), state,
    design_paths.
    """
    from toolkit import config as _config
    from toolkit import rfdiffusion as _rfdiffusion

    if rings_df is None:
        rings_df = load_checkpoint(ISOLATE_STAGE, state_dir, needed_by="Running RFdiffusion")

    subset = rings_df
    if assembly_id is not None:
        subset = subset[subset["assembly_id"] == assembly_id]
        if subset.empty:
            raise ValueError(f"No row for assembly_id={assembly_id!r} in the rings checkpoint.")
    subset = subset.reset_index(drop=True)

    cfg = _config.load_installation_config()
    tool_cfg = _config.get_tool_config(cfg, "rfdiffusion")
    resolved_backend = backend or tool_cfg.get("backend", "local")

    if detach and resolved_backend != "slurm":
        raise ValueError(
            f"--detach only works with backend='slurm' (a SLURM job survives after this "
            f"process exits, tracked by its own job id) -- resolved backend is "
            f"{resolved_backend!r}. Either set rfdiffusion.backend: slurm in installation.yaml "
            f"(or pass backend='slurm' here), or drop --detach and let this block until done."
        )

    rows = []
    for _, row in subset.iterrows():
        short_order = _rfdiffusion.remap_chain_order(tuple(row["chain_groups"]), row["chain_rename_map"])
        job = _rfdiffusion.prepare_fusion_job(
            row["filepath"], short_order, linker_length=linker_length,
            num_designs=num_designs, diffuser_T=diffuser_T,
        )
        run = _rfdiffusion.submit(job, backend=backend, config=cfg)
        rows.append({
            "assembly_id": row["assembly_id"], "symmetry_type": row["symmetry_type"],
            "chain_groups": tuple(row["chain_groups"]), "run": run,
            "state": "submitted", "design_paths": [],
        })
        print(f"Submitted {row['assembly_id']}" + (f" (slurm job {run.slurm_job_id})" if run.slurm_job_id else ""))

    if not detach:
        rows = _poll_all_rfdiffusion(rows, poll_interval=poll_interval, timeout=timeout)

    df = pd.DataFrame(rows, columns=["assembly_id", "symmetry_type", "chain_groups", "run", "state", "design_paths"])
    save_checkpoint(df, RFDIFFUSION_STAGE, state_dir)
    return df


def refresh_rfdiffusion_status(state_dir: str = DEFAULT_STATE_DIR) -> pd.DataFrame:
    """
    Re-polls every not-yet-terminal row of <state_dir>/rfdiffusion.pkl
    (e.g. from a --detach'd run_rfdiffusion() call) exactly once and
    re-saves the checkpoint with updated state/design_paths -- rows
    already completed/completed_partial/failed are left alone, no
    wasted squeue/sacct calls. Returns the refreshed DataFrame.
    """
    df = load_checkpoint(RFDIFFUSION_STAGE, state_dir, needed_by="Checking RFdiffusion status")
    rows = df.to_dict("records")
    _poll_all_rfdiffusion(rows, single_pass=True)
    df = pd.DataFrame(rows, columns=list(df.columns))
    save_checkpoint(df, RFDIFFUSION_STAGE, state_dir)
    return df


def run_status(state_dir: str = DEFAULT_STATE_DIR) -> pd.DataFrame:
    """
    Checks on --detach'd jobs. Only RFdiffusion ever has a pending state
    to check -- ProteinMPNN has no SLURM backend, so run_pmpnn() always
    blocks-and-polls to completion itself (see pmpnn.py's own module
    docstring: "LOCAL ONLY, on purpose") and never leaves anything
    for this to refresh.
    """
    if not checkpoint_exists(RFDIFFUSION_STAGE, state_dir):
        raise StageNotFoundError(RFDIFFUSION_STAGE, state_dir, "Checking status")
    return refresh_rfdiffusion_status(state_dir)


# ----------------------------------------------------------------------
# Stage 6: ProteinMPNN (batch-per-assembly, always local + block-and-poll)
# ----------------------------------------------------------------------

_PMPNN_SEQUENCE_COLUMNS: Tuple[str, ...] = (
    "assembly_id", "source_pdb", "sequence", "is_native", "temperature",
    "sample_index", "score", "global_score", "seq_recovery",
)


def run_pmpnn(
    rfdiffusion_df: Optional[pd.DataFrame] = None,
    assembly_id: Optional[str] = None,
    top_n: int = 1,
    min_plddt: Optional[float] = None,
    select: Optional[Sequence[str]] = None,
    num_seq_per_target: int = 8,
    sampling_temp: float = 0.1,
    batch_size: int = 8,
    poll_interval: float = 5.0,
    timeout: int = 900,
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Runs ProteinMPNN against each assembly's selected RFdiffusion
    design(s) -- batch-per-assembly, one job at a time. Unlike
    RFdiffusion, ProteinMPNN has no SLURM backend (always a local,
    blocking subprocess -- see pmpnn.py's own module docstring), so
    there's no detach/status split here: this always blocks until each
    assembly's job is done before moving to the next. Narrow to one
    assembly with assembly_id=.

    For each assembly processed:
      1. re-derives that assembly's RFdiffusionJob from its own saved
         run (a pure, deterministic rebuild -- does NOT re-run
         RFdiffusion) and ranks its designs via rfdiffusion.rank_designs(),
         RFdiffusion's own per-design confidence -- this IS the "let the
         user select which RFdiffusion models to run ProteinMPNN on
         based on their scoring" feature.
      2. selects design(s): select= (explicit design PDB path(s),
         repeatable -- only valid together with assembly_id=, since an
         explicit path list can't be automatically split across
         multiple assemblies) takes priority; otherwise top_n=/
         min_plddt= filter the ranked table (top_n=1 by default: just
         the single best-scored design).
      3. submits + blocks-and-polls ProteinMPNN, collects sequences.

    An assembly is skipped (reported, not fatal to the rest of the
    batch -- same fail-soft convention as the rest of this project) if
    it has no row in the rfdiffusion checkpoint, its RFdiffusion job
    isn't completed/completed_partial yet (run `symbro status` first
    for a --detach'd job), or nothing survives top_n=/min_plddt=
    filtering.

    If rfdiffusion_df isn't given, loads it from
    <state_dir>/rfdiffusion.pkl. Saves the result to
    <state_dir>/pmpnn.{pkl,csv} and returns it -- every processed
    assembly's sequences_df concatenated together, each row tagged with
    its own assembly_id.
    """
    import time as _time

    from toolkit import config as _config
    from toolkit import pmpnn as _pmpnn
    from toolkit import rfdiffusion as _rfdiffusion

    if rfdiffusion_df is None:
        rfdiffusion_df = load_checkpoint(RFDIFFUSION_STAGE, state_dir, needed_by="Running ProteinMPNN")

    subset = rfdiffusion_df
    if assembly_id is not None:
        subset = subset[subset["assembly_id"] == assembly_id]
        if subset.empty:
            raise ValueError(f"No row for assembly_id={assembly_id!r} in the rfdiffusion checkpoint.")
    subset = subset.reset_index(drop=True)

    if select is not None and (assembly_id is None or len(subset) != 1):
        raise ValueError(
            "select= (explicit design paths) requires assembly_id= to narrow to exactly one "
            "assembly first -- an explicit path list can't be automatically split across "
            "multiple assemblies' jobs."
        )

    cfg = _config.load_installation_config()
    frames = []
    for _, row in subset.iterrows():
        aid = row["assembly_id"]
        if row["state"] not in ("completed", "completed_partial"):
            print(f"Skipped {aid}: RFdiffusion state is {row['state']!r}, not completed yet "
                  f"(run `symbro status` first if this was a --detach'd job).")
            continue
        if not row["design_paths"]:
            print(f"Skipped {aid}: RFdiffusion produced no design files.")
            continue

        job = row["run"].job
        if select is not None:
            ranked = _rfdiffusion.rank_designs(row["design_paths"])
            selected_abs = [os.path.abspath(p) for p in select]
            missing = [g for g, ga in zip(select, selected_abs) if ga not in ranked["design_path"].values]
            if missing:
                raise ValueError(f"--select path(s) not among {aid}'s own designs: {missing}")
            selected_paths = selected_abs
        else:
            ranked = _rfdiffusion.rank_designs(row["design_paths"], top_n=top_n, min_plddt=min_plddt)
            selected_paths = ranked["design_path"].tolist()
        if not selected_paths:
            print(f"Skipped {aid}: no designs left after top_n/min_plddt filtering.")
            continue

        print(f"{aid}: submitting ProteinMPNN for {len(selected_paths)} design(s): "
              f"{[os.path.basename(p) for p in selected_paths]}")
        mpnn_job = _pmpnn.prepare_mpnn_job(
            selected_paths, rf_job=job, num_seq_per_target=num_seq_per_target,
            sampling_temp=sampling_temp, batch_size=batch_size,
        )
        run = _pmpnn.submit(mpnn_job, config=cfg)

        start = _time.time()
        status = _pmpnn.poll_status(run)
        while status["state"] not in _TERMINAL_STATES:
            if _time.time() - start > timeout:
                print(f"  {aid}: timed out after {timeout}s waiting for ProteinMPNN "
                      f"(local process -- check {run.log_path} by hand).")
                break
            _time.sleep(poll_interval)
            status = _pmpnn.poll_status(run)
        print(f"  {aid}: {status['state']} "
              f"({status['sequences_written']}/{status['sequences_expected']} sequences)")

        sequences_df = _pmpnn.collect_sequences(status)
        if not sequences_df.empty:
            sequences_df.insert(0, "assembly_id", aid)
            frames.append(sequences_df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(_PMPNN_SEQUENCE_COLUMNS))
    save_checkpoint(df, PMPNN_STAGE, state_dir)
    return df


# ----------------------------------------------------------------------
# Cleanup — the "start fresh" button, between pipeline runs
# ----------------------------------------------------------------------

_ALL_STAGES: Tuple[str, ...] = (
    CANDIDATES_STAGE, DOWNLOADED_STAGE, GEOMETRY_STAGE, ISOLATE_STAGE, RFDIFFUSION_STAGE, PMPNN_STAGE,
)


def clean(
    state: bool = True, downloads: bool = True, subunits: bool = True, simulations: bool = True,
    dry_run: bool = False, state_dir: str = DEFAULT_STATE_DIR,
) -> dict:
    """
    Clears scratch files and checkpoints between pipeline runs. Each
    category defaults to True (clear everything); pass False to keep
    that one category instead.

      state        : every <stage>.pkl/.csv checkpoint under state_dir
                     (candidates, downloaded, geometry, rings,
                     rfdiffusion, pmpnn) -- state_dir itself is left in
                     place, only the checkpoint files inside it go.
      downloads    : contents of temporary_files/ (download.py's own
                     clear_temp_dir()).
      subunits     : contents of temporary_subunits/ (isolate.py's own
                     clear_temp_subunits_dir() -- this also covers
                     RFdiffusion's own designs, which nest under
                     temporary_subunits/rfdiffusion_designs/).
      simulations  : contents of temporary_simulations/ (pmpnn.py's own
                     clear_simulations_dir() -- ProteinMPNN's out_folder).

    state=True is the default specifically because clearing scratch
    files WITHOUT also clearing the checkpoint(s) that reference them
    leaves stale, dangling filepath references behind -- e.g.
    downloaded.pkl still pointing at a .cif this call just deleted, so
    the next `symbro geometry` would fail with a confusing
    FileNotFoundError instead of a clear "state is stale" message. Only
    pass state=False for a deliberately surgical clean (e.g. clearing
    rfdiffusion/pmpnn scratch output to re-run with different parameters
    without re-downloading/re-isolating) where you understand exactly
    which checkpoints you're keeping and that everything they reference
    still needs to exist.

    dry_run=True reports what WOULD be cleared without deleting
    anything -- same category flags apply, nothing on disk changes.

    Returns a dict: {"state": [stage names actually cleared],
    "temporary_files": bool, "temporary_subunits": bool,
    "temporary_simulations": bool} -- False/empty for a category that
    was requested but had nothing to clear, so the CLI can report
    accurately rather than just echoing back what was asked for.
    """
    from toolkit import download as _download
    from toolkit import isolate as _isolate
    from toolkit import pmpnn as _pmpnn

    cleared = {
        "state": [], "temporary_files": False,
        "temporary_subunits": False, "temporary_simulations": False,
    }

    if state:
        for stage in _ALL_STAGES:
            pkl_path, csv_path = _checkpoint_paths(stage, state_dir)
            existing = [p for p in (pkl_path, csv_path) if os.path.exists(p)]
            if existing:
                if not dry_run:
                    for p in existing:
                        os.remove(p)
                cleared["state"].append(stage)

    def _has_content(dir_path: str) -> bool:
        return any(not entry.startswith(".") for entry in os.listdir(dir_path))

    if downloads:
        temp_dir = _download.get_temp_dir()
        if _has_content(temp_dir):
            if not dry_run:
                _download.clear_temp_dir(temp_dir)
            cleared["temporary_files"] = True

    if subunits:
        temp_dir = _isolate.get_temp_subunits_dir()
        if _has_content(temp_dir):
            if not dry_run:
                _isolate.clear_temp_subunits_dir(temp_dir)
            cleared["temporary_subunits"] = True

    if simulations:
        temp_dir = _pmpnn.get_simulations_dir()
        if _has_content(temp_dir):
            if not dry_run:
                _pmpnn.clear_simulations_dir(temp_dir)
            cleared["temporary_simulations"] = True

    return cleared
