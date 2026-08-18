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
import re
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
PREDICT_STAGE = "predict"
CODON_STAGE = "codon"

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
    filter_criteria: Optional[Sequence[dict]] = None,
    return_type: str = "assembly",
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Search RCSB PDB for candidates. At least one of symmetry / entry_id /
    resolution_range / description / extra_criteria must be given, or
    every criterion in the search would be unconstrained.

    filter_criteria : optional list of {"attribute", "value", "operator"}
        dicts (same shape as extra_criteria), applied locally via
        query.filter_metadata() AFTER the RCSB fetch, instead of being
        sent to RCSB's own Search API. This is the only way to filter on a
        FETCH_ONLY_ATTRIBUTES field (e.g. model_quality) -- those have no
        Search API equivalent at all, so they can never appear in
        extra_criteria/search_criteria. Does NOT count toward the "at
        least one search criterion" requirement above -- it only narrows
        results a real search criterion has already fetched, so it can't
        by itself constrain an otherwise-unbounded search.

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
            "entry_id, resolution_range, description, or extra_criteria. "
            "(filter_criteria alone doesn't count -- it only narrows an "
            "already-constrained search.)"
        )

    from toolkit import query as _query

    criteria = _build_criteria(symmetry, entry_id, resolution_range, description, extra_criteria)
    # "symmetry" (RCSB's OWN annotated rcsb_struct_symmetry.symbol) is
    # always fetched, whether or not the caller asked for it or searched
    # by it: run_geometry()'s annotated-symmetry cross-check (see
    # _drop_symmetry_mismatches() below) needs it on every candidate to
    # have anything to compare its own empirical ring detection against
    # -- making this opt-in via --fetch-field would silently disable that
    # safety check for anyone who didn't think to ask for it.
    all_fetch_fields = list(fetch_fields) if fetch_fields else []
    if "symmetry" not in all_fetch_fields:
        all_fetch_fields.append("symmetry")
    df = _query.query_candidates(
        search_criteria=criteria,
        fetch_fields=all_fetch_fields,
        filter_criteria=list(filter_criteria) if filter_criteria else None,
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
# Stage 2 (alternative): local -- register your own structure file(s)
# instead of query + download
# ----------------------------------------------------------------------

def run_local(
    paths: Sequence[str],
    assembly_ids: Optional[Sequence[str]] = None,
    data_dir: Optional[str] = None,
    overwrite: bool = False,
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Registers your own local PDB/CIF file(s) as candidates -- an
    alternative to run_query() + run_download() for a structure you
    already have, rather than one you want RCSB to find for you.

    Writes to the SAME checkpoint run_download() does
    (<state_dir>/downloaded.{pkl,csv}) -- every stage from run_geometry()
    onward can't tell the difference and needs no changes to work with a
    locally-sourced structure. Calling this after `symbro query` (or
    `symbro download`) in the same project overwrites that checkpoint,
    same as re-running `symbro download` itself would -- run_local() and
    run_query()+run_download() are alternatives, not additive; mixing
    RCSB-sourced and local candidates in one project isn't supported
    today (call this once with every local path you want registered, in
    one go).

    See local.py's own module docstring for why files are copied into
    temporary_files/local/ rather than referenced in place, and what
    happens if no assembly_ids are given.
    """
    from toolkit import local as _local

    df = _local.register_local_structures(
        list(paths), assembly_ids=list(assembly_ids) if assembly_ids else None,
        data_dir=data_dir, overwrite=overwrite,
    )
    save_checkpoint(df, DOWNLOADED_STAGE, state_dir)
    return df


# ----------------------------------------------------------------------
# Stage 3: geometry (rings, always; orientation + termini SS, if symmetry_type given)
# ----------------------------------------------------------------------

_CYCLIC_SYMBOL_RE = re.compile(r"^C(\d+)$")

# Platonic point groups -> the cyclic rotation axes they're actually built
# from (proper-rotation subgroup only, which is what a homomeric protein
# cage's chains are ever related by): T (tetrahedral) has 4 C3 axes + 3 C2
# axes; O (octahedral) has 3 C4 + 4 C3 + 6 C2; I (icosahedral) has 6 C5 +
# 10 C3 + 15 C2. This project's own target assemblies ARE, by nature,
# Platonic -- geometry/rings.py's whole job is isolating their cyclic
# sub-rings for RFdiffusion (see isolate.py, and rings.py's own T:3,3
# example) -- so unlike dihedral/helical/asymmetric annotations (below),
# this decomposition isn't a guess, it's exactly what the rest of this
# pipeline already assumes and acts on.
_PLATONIC_EXPECTED_ORDERS = {"T": (3, 2), "O": (4, 3, 2), "I": (5, 3, 2)}


def _expected_cyclic_orders(annotated_symmetry, allowed_orders: Sequence[int]) -> list:
    """
    Parses an RCSB rcsb_struct_symmetry.symbol value (e.g. "C3", "T", or
    "C3, C2" -- query.extract_leaf_values' comma-join of a multi-component
    assembly's several symmetry records) into the cyclic orders symbro's
    own ring detector could, in principle, confirm or refute: a plain
    "C{n}" token maps to itself, and "T"/"O"/"I" map to every cyclic axis
    order their Platonic point group actually contains (see
    _PLATONIC_EXPECTED_ORDERS) -- finding ANY one of them is enough for
    _drop_symmetry_mismatches() to keep the assembly, same as a
    multi-component "C3, C2" annotation already works.

    Dihedral "D*", helical "H", asymmetric "C1", an order outside
    allowed_orders (e.g. "C6"), or a missing/NaN value are still simply
    dropped from the result, not guessed at -- this project doesn't
    otherwise claim to know which cyclic sub-rings THOSE point groups
    decompose into the way it does for its own Platonic target
    assemblies. Returns [] if nothing in the annotation is in scope.
    """
    if annotated_symmetry is None or (isinstance(annotated_symmetry, float) and pd.isna(annotated_symmetry)):
        return []
    allowed = set(allowed_orders)
    orders = []
    for token in str(annotated_symmetry).split(","):
        token = token.strip()
        match = _CYCLIC_SYMBOL_RE.match(token)
        if match:
            if int(match.group(1)) in allowed:
                orders.append(int(match.group(1)))
            continue
        for order in _PLATONIC_EXPECTED_ORDERS.get(token, ()):
            if order in allowed and order not in orders:
                orders.append(order)
    return orders


def _drop_symmetry_mismatches(
    rings_df: pd.DataFrame, downloaded_df: pd.DataFrame, allowed_orders: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-checks each assembly's empirically detected rings (rings_df --
    pure structural analysis, no RCSB involved) against RCSB's OWN
    annotated symmetry (downloaded_df's "symmetry" column,
    rcsb_struct_symmetry.symbol -- always fetched by run_query(), see its
    own comment). Real PDB depositions occasionally have this wrong (the
    wrong assembly definition marked biological, a crystallographic
    packing mate mistaken for a real ring, etc.) -- an assembly that
    fails this check is a strong candidate for exactly that kind of
    annotation issue, not a design worth spending RFdiffusion/ProteinMPNN
    compute chasing.

    Only assemblies whose annotation includes at least one in-scope
    cyclic order (see _expected_cyclic_orders -- plain "C{n}" tokens, or
    "T"/"O"/"I" mapped to their own known constituent axes) are checked
    at all -- dihedral/helical/asymmetric annotations, or a missing
    "symmetry" column entirely (e.g. `symbro local` candidates, never
    looked up against RCSB), are left untouched.

    An assembly is dropped -- every one of its rows, across every
    detected symmetry_type/component -- only if NONE of its expected
    cyclic orders were found anywhere in rings_df for that assembly_id.
    Finding even one (e.g. RCSB says "C3, C2", or "T" which expands to
    (3, 2), and only the C3 ring was confirmed) is enough to keep it --
    deliberately lenient, since even a genuinely correct Platonic
    assembly's OTHER axis types can fail rings.py's own N-C-register
    detection for real geometric reasons having nothing to do with
    whether the annotation is right.

    Returns (kept_df, dropped_df). dropped_df has one row per dropped
    assembly_id: assembly_id, expected (e.g. "C3", or "C2, C3" if more
    than one order was annotated), detected (comma-joined symmetry_types
    rings_df DID find for that assembly, or "none"). Empty (but
    correctly-columned) if nothing was dropped, or if there was nothing
    in scope to check.
    """
    empty_dropped = pd.DataFrame(columns=["assembly_id", "expected", "detected"])
    if rings_df.empty or downloaded_df is None or "symmetry" not in downloaded_df.columns \
            or "assembly_id" not in downloaded_df.columns:
        return rings_df, empty_dropped

    annotated = downloaded_df.drop_duplicates(subset="assembly_id", keep="first") \
        .set_index("assembly_id")["symmetry"]
    detected_by_assembly = rings_df.groupby("assembly_id")["symmetry_type"].apply(set)

    assemblies_to_drop = set()
    dropped_rows = []
    for assembly_id, annotated_symmetry in annotated.items():
        expected_orders = _expected_cyclic_orders(annotated_symmetry, allowed_orders)
        if not expected_orders:
            continue  # nothing in scope for this assembly -- leave it alone
        expected_types = {f"C{n}" for n in expected_orders}
        detected_types = detected_by_assembly.get(assembly_id, set())
        if expected_types & detected_types:
            continue  # at least one expected ring was actually confirmed
        assemblies_to_drop.add(assembly_id)
        dropped_rows.append({
            "assembly_id": assembly_id,
            "expected": ", ".join(sorted(expected_types)),
            "detected": ", ".join(sorted(detected_types)) if detected_types else "none",
        })

    if not assemblies_to_drop:
        return rings_df, empty_dropped

    kept_df = rings_df[~rings_df["assembly_id"].isin(assemblies_to_drop)].reset_index(drop=True)
    return kept_df, pd.DataFrame(dropped_rows, columns=["assembly_id", "expected", "detected"])


def run_geometry(
    downloaded_df: Optional[pd.DataFrame] = None,
    symmetry_type: Optional[str] = None,
    state_dir: str = DEFAULT_STATE_DIR,
    validate_annotated_symmetry: bool = True,
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

    validate_annotated_symmetry : if True (default), cross-checks every
        assembly's detected rings against RCSB's own annotated symmetry
        (see _drop_symmetry_mismatches()) and drops any assembly whose
        detection found none of its expected cyclic orders -- almost
        always a sign of a PDB annotation issue (wrong assembly marked
        biological, etc.) rather than a real candidate worth pursuing.
        A warning naming each dropped assembly, what was expected, and
        what was actually detected is printed either way. Pass False to
        keep every detected ring regardless of what RCSB annotated --
        e.g. if you're deliberately investigating a mismatch yourself.

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

    if validate_annotated_symmetry and not rings_df.empty:
        rings_df, dropped_df = _drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
        for _, row in dropped_df.iterrows():
            print(
                f"Warning: {row['assembly_id']} dropped -- RCSB annotates it as "
                f"{row['expected']} symmetry, but symbro's own geometry detection "
                f"found {row['detected']} instead (likely a PDB annotation issue, "
                f"not a real candidate). Re-run with validate_annotated_symmetry=False "
                f"(--no-validate-symmetry on the CLI) to keep it anyway."
            )

    if symmetry_type is None or rings_df.empty:
        save_checkpoint(rings_df, GEOMETRY_STAGE, state_dir)
        return rings_df

    orientation_df = _orientation.from_rings(rings_df, downloaded_df, symmetry_type)
    if orientation_df.empty:
        save_checkpoint(orientation_df, GEOMETRY_STAGE, state_dir)
        return orientation_df

    termini_df = _structure.from_rings(rings_df, downloaded_df, symmetry_type)
    # Merge key includes chain_groups, not just (assembly_id,
    # symmetry_type): since rings.py started emitting one row per
    # (order, identity component) rather than one row per order (see
    # rings.py's module docstring), a multi-component assembly now has
    # MULTIPLE rows sharing the same (assembly_id, symmetry_type) --
    # merging on those two columns alone would cross-join every
    # component's orientation row against every component's termini_ss
    # row instead of pairing each component with its own. chain_groups is
    # unique per row within one assembly's own symmetry_type (different
    # components/duplicates always differ in actual chain composition),
    # so adding it here restores a correct one-to-one merge.
    merged = orientation_df.merge(
        termini_df[["assembly_id", "symmetry_type", "chain_groups", "termini_ss"]],
        on=["assembly_id", "symmetry_type", "chain_groups"],
        how="left",
    )
    save_checkpoint(merged, GEOMETRY_STAGE, state_dir)
    return merged


def detected_symmetry_types(rings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary of what ring detection found, per symmetry_type -- for
    printing a friendly overview before committing to one --symmetry-type.

    Returns a DataFrame with columns:
      symmetry_type, assemblies (distinct assembly_id count), components
      (row count -- one per structurally distinct identity component
      detected per assembly; see rings.py's module docstring. Equal to
      `assemblies` for single-component/homomeric cages, greater than it
      for multi-component ones, e.g. a two-protein T:3,3 cage has 2
      components for its C3 rows), total_axis_count (sum of axis_count
      across all those rows -- i.e. total rings found, INCLUDING
      same-component redundant duplicates, e.g. all 4 copies of one
      protein's trimer in a tetrahedral cage).

    Sorted by `assemblies` descending, matching this function's previous
    (Series-returning) sort order. Empty (but correctly-columned) if
    rings_df is empty or has no symmetry_type column.
    """
    columns = ["symmetry_type", "assemblies", "components", "total_axis_count"]
    if rings_df.empty or "symmetry_type" not in rings_df.columns:
        return pd.DataFrame(columns=columns)

    grouped = rings_df.groupby("symmetry_type")
    summary = pd.DataFrame({
        "assemblies": grouped["assembly_id"].nunique(),
        "components": grouped.size(),
        "total_axis_count": (
            grouped["axis_count"].sum() if "axis_count" in rings_df.columns
            else grouped.size() * 0  # older rings_df without axis_count -- report 0, not a crash
        ),
    }).reset_index()
    return summary.sort_values("assemblies", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Stage 4: isolate
# ----------------------------------------------------------------------

def run_isolate(
    geometry_df: Optional[pd.DataFrame] = None,
    downloaded_df: Optional[pd.DataFrame] = None,
    symmetry_type: Optional[str] = None,
    component_id: Optional[int] = None,
    file_format: str = "pdb",
    output_dir: Optional[str] = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Extracts one ring PDB (or mmCIF) per (assembly, component) into
    temporary_subunits/ (or output_dir) -- the files RFdiffusion needs
    next. If geometry_df / downloaded_df aren't given, loads them from
    <state_dir>/geometry.pkl and <state_dir>/downloaded.pkl.

    A multi-component assembly (e.g. a two-protein T:3,3 cage -- see
    rings.py's module docstring) isolates EVERY component by default,
    each into its own file; pass component_id= to isolate only one
    specific component instead (e.g. to hand just one protein of a
    two-component cage to RFdiffusion for now).

    Saves the result to <state_dir>/rings.{pkl,csv} and returns it.
    """
    from toolkit import isolate as _isolate

    if geometry_df is None:
        geometry_df = load_checkpoint(GEOMETRY_STAGE, state_dir, needed_by="Isolating rings")
    if downloaded_df is None:
        downloaded_df = load_checkpoint(DOWNLOADED_STAGE, state_dir, needed_by="Isolating rings")

    df = _isolate.from_rings(
        geometry_df, downloaded_df, symmetry_type=symmetry_type, component_id=component_id,
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


# Used only when linker_length=None (auto) AND the row itself has no
# recommended_linker_length -- e.g. a rings.pkl checkpoint written by a
# rings.py that predates estimate_linker_length(). Matches the CLI's
# previous hardcoded default so behavior for old checkpoints is unchanged.
_FALLBACK_LINKER_LENGTH: Tuple[int, int] = (15, 25)


def run_rfdiffusion(
    rings_df: Optional[pd.DataFrame] = None,
    assembly_id: Optional[str] = None,
    linker_length: Optional[Union[int, Tuple[int, int]]] = None,
    num_designs: int = 10,
    diffuser_T: int = 50,
    backend: Optional[str] = None,
    detach: bool = False,
    poll_interval: int = 20,
    timeout: Optional[int] = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Submits one RFdiffusion job per (assembly, component) row --
    batch-per-row, the architecture decided for this stage: every row of
    rings.pkl gets its own independent job rather than one giant
    multi-assembly job. Since rings.py now emits one row per
    (symmetry_type, identity component) rather than one row per
    symmetry_type (see rings.py's module docstring), a multi-component
    assembly's rows -- e.g. a two-protein T:3,3 cage's two C3 rows --
    already submit as two separate jobs here without any special-casing:
    this function has always iterated every row of rings_df, never
    assumed exactly one row per assembly. Narrow to a single assembly's
    row(s) with assembly_id= (still possibly more than one row, for a
    multi-component assembly); narrow further at the isolate stage with
    component_id= if only one component's job is wanted.

    linker_length : (min_residues, max_residues) for RFdiffusion's
        diffused-linker contig segment. Defaults to None, meaning
        "auto" -- each row uses ITS OWN rings.py-computed
        recommended_linker_length (derived from that ring's own measured
        mean_distance; see rings.py's estimate_linker_length()) if
        present, falling back to a fixed (15, 25) only for a row from an
        older rings_df that predates that column. Pass an explicit
        int or (min, max) tuple here (or via the CLI's --linker-min/
        --linker-max) to override the recommendation for every row
        uniformly instead.

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
    (assembly, component): assembly_id, symmetry_type, chain_groups, run
    (a picklable RFdiffusionRun -- see _poll_all_rfdiffusion()'s
    docstring for why .process is always None by the time this is
    saved), state, design_paths.
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
        if linker_length is not None:
            row_linker_length = linker_length
        else:
            recommended = row.get("recommended_linker_length")
            row_linker_length = tuple(recommended) if isinstance(recommended, (tuple, list)) else _FALLBACK_LINKER_LENGTH
        short_order = _rfdiffusion.remap_chain_order(tuple(row["chain_groups"]), row["chain_rename_map"])
        job = _rfdiffusion.prepare_fusion_job(
            row["filepath"], short_order, linker_length=row_linker_length,
            num_designs=num_designs, diffuser_T=diffuser_T,
        )
        run = _rfdiffusion.submit(job, backend=backend, config=cfg)
        rows.append({
            "assembly_id": row["assembly_id"], "symmetry_type": row["symmetry_type"],
            "component_id": row.get("component_id"),
            "chain_groups": tuple(row["chain_groups"]), "run": run,
            "state": "submitted", "design_paths": [],
        })
        component_note = f", component {row.get('component_id')}" if row.get("component_id") is not None else ""
        print(f"Submitted {row['assembly_id']}{component_note}"
              + (f" (slurm job {run.slurm_job_id})" if run.slurm_job_id else ""))

    if not detach:
        rows = _poll_all_rfdiffusion(rows, poll_interval=poll_interval, timeout=timeout)

    df = pd.DataFrame(
        rows, columns=["assembly_id", "symmetry_type", "component_id", "chain_groups", "run", "state", "design_paths"],
    )
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
    "assembly_id", "component_id", "source_pdb", "sequence", "is_native", "temperature",
    "sample_index", "score", "global_score", "seq_recovery",
)


def run_pmpnn(
    rfdiffusion_df: Optional[pd.DataFrame] = None,
    assembly_id: Optional[str] = None,
    component_id: Optional[int] = None,
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
    Runs ProteinMPNN against each (assembly, component)'s selected
    RFdiffusion design(s) -- batch-per-row, one job at a time. Unlike
    RFdiffusion, ProteinMPNN has no SLURM backend (always a local,
    blocking subprocess -- see pmpnn.py's own module docstring), so
    there's no detach/status split here: this always blocks until each
    row's job is done before moving to the next. Narrow to one assembly
    with assembly_id= (a multi-component assembly may still have more
    than one matching row -- one per component); narrow further to one
    specific component with component_id=.

    For each row processed:
      1. re-derives that row's RFdiffusionJob from its own saved run (a
         pure, deterministic rebuild -- does NOT re-run RFdiffusion) and
         ranks its designs via rfdiffusion.rank_designs(), RFdiffusion's
         own per-design confidence -- this IS the "let the user select
         which RFdiffusion models to run ProteinMPNN on based on their
         scoring" feature.
      2. selects design(s): select= (explicit design PDB path(s),
         repeatable -- only valid together with assembly_id= AND, for a
         multi-component assembly, component_id= narrowing to exactly
         one row, since an explicit path list can't be automatically
         split across multiple jobs) takes priority; otherwise top_n=/
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
    <state_dir>/pmpnn.{pkl,csv} and returns it -- every processed row's
    sequences_df concatenated together, each row tagged with its own
    assembly_id AND component_id (None if rfdiffusion_df predates
    per-component rings -- see rings.py's module docstring). A
    multi-component assembly (e.g. a two-protein T:3,3 cage) has more
    than one row in rfdiffusion_df sharing the same assembly_id, one per
    component, each processed independently here -- component_id is what
    lets the resulting pmpnn.pkl tell those apart afterward.
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
    if component_id is not None:
        if "component_id" not in subset.columns:
            raise ValueError(
                "component_id filter was given, but the rfdiffusion checkpoint has no "
                "'component_id' column -- it predates rings.py's per-component grouping."
            )
        subset = subset[subset["component_id"] == component_id]
        if subset.empty:
            raise ValueError(f"No row for assembly_id={assembly_id!r}, component_id={component_id!r}.")
    subset = subset.reset_index(drop=True)

    if select is not None and (assembly_id is None or len(subset) != 1):
        raise ValueError(
            "select= (explicit design paths) requires assembly_id= (and, for a multi-component "
            "assembly, component_id=) to narrow to exactly one row first -- an explicit path list "
            "can't be automatically split across multiple jobs."
        )

    cfg = _config.load_installation_config()
    frames = []
    for _, row in subset.iterrows():
        aid = row["assembly_id"]
        cid = row.get("component_id")
        label = f"{aid}" + (f" (component {cid})" if cid is not None else "")
        if row["state"] not in ("completed", "completed_partial"):
            print(f"Skipped {label}: RFdiffusion state is {row['state']!r}, not completed yet "
                  f"(run `symbro status` first if this was a --detach'd job).")
            continue
        if not row["design_paths"]:
            print(f"Skipped {label}: RFdiffusion produced no design files.")
            continue

        job = row["run"].job
        if select is not None:
            ranked = _rfdiffusion.rank_designs(row["design_paths"])
            selected_abs = [os.path.abspath(p) for p in select]
            missing = [g for g, ga in zip(select, selected_abs) if ga not in ranked["design_path"].values]
            if missing:
                raise ValueError(f"--select path(s) not among {label}'s own designs: {missing}")
            selected_paths = selected_abs
        else:
            ranked = _rfdiffusion.rank_designs(row["design_paths"], top_n=top_n, min_plddt=min_plddt)
            selected_paths = ranked["design_path"].tolist()
        if not selected_paths:
            print(f"Skipped {label}: no designs left after top_n/min_plddt filtering.")
            continue

        print(f"{label}: submitting ProteinMPNN for {len(selected_paths)} design(s): "
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
                print(f"  {label}: timed out after {timeout}s waiting for ProteinMPNN "
                      f"(local process -- check {run.log_path} by hand).")
                break
            _time.sleep(poll_interval)
            status = _pmpnn.poll_status(run)
        print(f"  {label}: {status['state']} "
              f"({status['sequences_written']}/{status['sequences_expected']} sequences)")

        sequences_df = _pmpnn.collect_sequences(status)
        if not sequences_df.empty:
            sequences_df.insert(0, "assembly_id", aid)
            sequences_df.insert(1, "component_id", cid)
            frames.append(sequences_df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(_PMPNN_SEQUENCE_COLUMNS))
    save_checkpoint(df, PMPNN_STAGE, state_dir)
    return df


# ----------------------------------------------------------------------
# Stage 7: structure prediction (self-consistency screening, batch-per-
# (assembly, component), always local-call-blocks-to-completion)
# ----------------------------------------------------------------------

# assembly_id/component_id/predictor (inserted by run_predict() itself) +
# selfconsistency.collect_results()'s own column set -- kept here, same
# convention as _PMPNN_SEQUENCE_COLUMNS above, so an empty result still
# carries the real schema instead of a bare, columnless DataFrame.
_PREDICT_COLUMNS: Tuple[str, ...] = (
    "assembly_id", "component_id", "predictor", "candidate_id", "folded_path",
    "reference_path", "rmsd_to_design", "mean_plddt",
)


def run_predict(
    pmpnn_df: Optional[pd.DataFrame] = None,
    rfdiffusion_df: Optional[pd.DataFrame] = None,
    predictor: Optional[str] = None,
    assembly_id: Optional[str] = None,
    component_id: Optional[int] = None,
    top_n: int = 3,
    max_rmsd: float = 2.0,
    min_plddt: float = 70.0,
    backend: Optional[str] = None,
    af3_model_dir: Optional[str] = None,
    af3_db_dir: Optional[str] = None,
    af3_terms_acknowledged: bool = False,
    af3_run_data_pipeline: Optional[bool] = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Folds ProteinMPNN's best candidate(s) per (assembly, component) back
    with a structure-prediction backend (see structure_prediction.py:
    "boltz"/"af2"/"alphafold2"/"af3"/"alphafold3", or None for
    installation.yaml's structure_prediction.default) and screens them by
    self-consistency (CA-RMSD + pLDDT against the RFdiffusion backbone
    they were designed on) -- the pipeline's final validation step.

    Unlike run_rfdiffusion()/run_pmpnn(), there's no manual poll loop
    here: alphafold2.run()/boltz.run()/af3.run() already do
    prepare-submit-poll-collect-validate as ONE blocking call.

    design_paths (the RFdiffusion PDBs each candidate must be compared
    against) are recovered from rfdiffusion_df's own design_paths column,
    joined on (assembly_id, component_id) -- the SAME row run_pmpnn()
    itself read to build ProteinMPNN's input. Deliberately not
    reconstructed from ProteinMPNN's staged input directory the way the
    older standalone run_predictor.py script did: that directory is only
    ever valid for the MOST RECENTLY run assembly (pmpnn.py's own
    _stage_input_pdbs() clears/overwrites it at the start of every next
    submit() against the same out_folder), so it silently breaks for any
    other assembly. Reading it back off rfdiffusion_df instead works
    regardless of how many assemblies have been run since.

    If pmpnn_df/rfdiffusion_df aren't given, loads them from
    <state_dir>/pmpnn.pkl and <state_dir>/rfdiffusion.pkl. Saves the
    result to <state_dir>/predict.{pkl,csv} and returns it -- every
    validated candidate across every processed (assembly, component),
    tagged with predictor/assembly_id/component_id.

    af3_model_dir/af3_db_dir/af3_terms_acknowledged/af3_run_data_pipeline
    are ignored for every predictor except af3/alphafold3. Unlike
    terms_acknowledged (which af3.run() itself already falls back to
    installation.yaml's af3.terms_acknowledged for), model_dir/db_dir/
    run_data_pipeline have NO such fallback inside af3.run() -- its
    model_dir parameter has no default at all -- so this function
    resolves them from installation.yaml's own af3: section first.

    An (assembly, component) is skipped (reported, not fatal to the rest
    of the batch -- same fail-soft convention as run_pmpnn) if it has no
    row in the rfdiffusion checkpoint, that row has no design_paths, or
    the predictor call itself raises RuntimeError (a failed job -- see
    the printed log_path).
    """
    from toolkit import config as _config
    from toolkit import pmpnn as _pmpnn
    from toolkit import structure_prediction as _structure_prediction

    if pmpnn_df is None:
        pmpnn_df = load_checkpoint(PMPNN_STAGE, state_dir, needed_by="Running structure prediction")
    if rfdiffusion_df is None:
        rfdiffusion_df = load_checkpoint(RFDIFFUSION_STAGE, state_dir, needed_by="Running structure prediction")

    subset = pmpnn_df
    if assembly_id is not None:
        subset = subset[subset["assembly_id"] == assembly_id]
        if subset.empty:
            raise ValueError(f"No row for assembly_id={assembly_id!r} in the pmpnn checkpoint.")
    if component_id is not None:
        subset = subset[subset["component_id"] == component_id]
        if subset.empty:
            raise ValueError(f"No row for assembly_id={assembly_id!r}, component_id={component_id!r}.")

    cfg = _config.load_installation_config()
    resolved_predictor = predictor or _config.get_tool_config(cfg, "structure_prediction").get("default")

    predictor_kwargs = {}
    if resolved_predictor and resolved_predictor.lower() in ("af3", "alphafold3"):
        af3_cfg = _config.get_tool_config(cfg, "af3")
        predictor_kwargs["model_dir"] = af3_model_dir or af3_cfg.get("model_dir")
        predictor_kwargs["db_dir"] = af3_db_dir or af3_cfg.get("db_dir")
        predictor_kwargs["terms_acknowledged"] = af3_terms_acknowledged or bool(af3_cfg.get("terms_acknowledged", False))
        run_data_pipeline = af3_run_data_pipeline
        if run_data_pipeline is None:
            run_data_pipeline = af3_cfg.get("run_data_pipeline")
        if run_data_pipeline is not None:
            predictor_kwargs["run_data_pipeline"] = run_data_pipeline

    frames = []
    for (aid, cid), group_df in subset.groupby(["assembly_id", "component_id"], dropna=False, sort=False):
        label = f"{aid}" + (f" (component {cid})" if pd.notna(cid) else "")
        same_assembly = rfdiffusion_df["assembly_id"] == aid
        if "component_id" in rfdiffusion_df.columns:
            # groupby(dropna=False) hands back a float nan for a missing
            # component_id (the common case: a single-component assembly),
            # never the original None -- comparing that nan against
            # rfdiffusion_df's own component_id column with plain == is
            # always False (NaN != NaN, and NaN != None too), even when the
            # "same" None/NaN row is right there. pd.isna() on both sides
            # is what actually matches it -- confirmed against a minimal
            # single-assembly reproduction of this exact shape, which
            # matched 0 rows with == and 1 row with this fix.
            cid_match = rfdiffusion_df["component_id"].isna() if pd.isna(cid) else rfdiffusion_df["component_id"] == cid
            rf_rows = rfdiffusion_df[same_assembly & cid_match]
        else:
            rf_rows = rfdiffusion_df[same_assembly]
        if rf_rows.empty:
            print(f"Skipped {label}: no matching row in the rfdiffusion checkpoint.")
            continue
        design_paths = rf_rows.iloc[0]["design_paths"]
        if not design_paths:
            print(f"Skipped {label}: RFdiffusion row has no design_paths.")
            continue

        shortlist = _pmpnn.select_best_designs(group_df, top_n=top_n)
        if shortlist.empty:
            print(f"Skipped {label}: no candidates left after top_n filtering.")
            continue

        print(f"{label}: screening {len(shortlist)} candidate(s) with {resolved_predictor!r}.")
        try:
            winners = _structure_prediction.run(
                predictor, shortlist, design_paths, config=cfg, backend=backend,
                max_rmsd=max_rmsd, min_plddt=min_plddt, **predictor_kwargs,
            )
        except RuntimeError as exc:
            print(f"  {label}: predictor run failed: {exc}")
            continue

        print(f"  {label}: {len(winners)}/{len(shortlist)} candidate(s) passed self-consistency screening.")
        if not winners.empty:
            winners = winners.copy()
            winners.insert(0, "assembly_id", aid)
            winners.insert(1, "component_id", cid)
            winners.insert(2, "predictor", resolved_predictor)
            frames.append(winners)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(_PREDICT_COLUMNS))
    save_checkpoint(df, PREDICT_STAGE, state_dir)
    return df


def _component_key(df: pd.DataFrame) -> pd.Series:
    """A NaN-safe merge key for a component_id column -- component_id is
    None for a single-component assembly, and plain == / a plain merge on
    a float NaN column doesn't reliably match None to None (this is the
    exact join bug run_predict() itself needed a fix for once already --
    see test_predict.py). Shared here so it isn't reinvented per caller.

    Always stringifies the non-null case too (not just the "__none__"
    sentinel) -- if one side of a merge happens to be a subset with no
    NaN component_id at all (e.g. join_predict_with_pmpnn() called after
    narrowing predict_df to a single multi-component assembly), leaving
    non-null values as raw floats produces a plain float64 key column on
    that side but an object-dtype column (mixed "__none__" strings and
    floats) on the other, un-narrowed side -- pandas' merge() refuses to
    join float64 against object outright, confirmed by hitting exactly
    this with `symbro codon --assembly-id`'s own narrowing. Stringifying
    unconditionally keeps both sides' dtype identical regardless of
    whether either one happens to contain a NaN in a given call."""
    return df["component_id"].apply(lambda v: "__none__" if pd.isna(v) else str(v))


def join_predict_with_pmpnn(predict_df: pd.DataFrame, pmpnn_df: pd.DataFrame) -> pd.DataFrame:
    """
    predict.pkl (see _PREDICT_COLUMNS) doesn't carry the amino acid
    sequence that produced each validated candidate -- only
    candidate_id/folded_path/rmsd_to_design/mean_plddt. The sequence
    itself lives in pmpnn.pkl, keyed differently: candidate_id is
    sanitize_id(f"{source_pdb}_rank{rank}") (see selfconsistency.py's own
    sanitize_id()/build_reference_map()), where rank is assigned fresh
    every time run_predict() calls pmpnn.select_best_designs() -- sort
    each (assembly_id, component_id, source_pdb) group by global_score
    ascending (excluding the native-sequence readback row), rank starting
    at 1 -- and is never itself stored in pmpnn.pkl. This re-derives that
    same rank/candidate_id here to join the two checkpoints back together
    (used by run_codon() below, and by examples/04_analysis_notebook's
    own join, kept in sync with this rather than reimplemented there).

    Returns predict_df with sequence/score/global_score/seq_recovery
    columns added from the matching pmpnn_df row (NaN in those columns,
    not a raised error, if a row's originating pmpnn_df entry can't be
    found -- e.g. mismatched checkpoints from different runs).
    """
    non_native = pmpnn_df[~pmpnn_df["is_native"]].copy()
    non_native["rank"] = (
        non_native.sort_values("global_score")
        .groupby(["assembly_id", "component_id", "source_pdb"], dropna=False, sort=False)
        .cumcount() + 1
    )
    non_native["candidate_id"] = [
        re.sub(r"[^A-Za-z0-9_-]", "_", f"{src}_rank{rank}")
        for src, rank in zip(non_native["source_pdb"], non_native["rank"])
    ]
    non_native["_component_key"] = _component_key(non_native)

    result = predict_df.copy()
    result["_component_key"] = _component_key(result)
    result = result.merge(
        non_native[["assembly_id", "_component_key", "candidate_id", "sequence", "score",
                     "global_score", "seq_recovery"]],
        on=["assembly_id", "_component_key", "candidate_id"], how="left",
    )
    return result.drop(columns="_component_key")


# ----------------------------------------------------------------------
# Stage 8: codon optimization (reverse-translate validated designs into
# synthesis-ready DNA -- always local, pure computation, no external tool
# or GPU involved)
# ----------------------------------------------------------------------

_CODON_COLUMNS: Tuple[str, ...] = (
    "assembly_id", "component_id", "candidate_id", "host", "protein_sequence",
    "dna_sequence", "gc_content", "warnings",
)


def run_codon(
    predict_df: Optional[pd.DataFrame] = None,
    pmpnn_df: Optional[pd.DataFrame] = None,
    assembly_id: Optional[str] = None,
    host: Optional[str] = None,
    method: str = "use_best_codon",
    gc_min: float = 0.3,
    gc_max: float = 0.65,
    gc_window: int = 100,
    homopolymer_max: int = 5,
    avoid_hairpins: bool = True,
    avoid_repeats_kmer: Optional[int] = 15,
    avoid_enzymes: Optional[Sequence[str]] = None,
    add_stop_codon: bool = True,
    fasta_path: Optional[str] = None,
    state_dir: str = DEFAULT_STATE_DIR,
) -> pd.DataFrame:
    """
    Reverse-translates every validated candidate in predict.pkl into
    host-codon-optimized DNA (toolkit.codon, built on DNAChisel +
    python_codon_tables -- needs the "codon" extra: `pip install
    symbro[codon]`), applying the standard gene-synthesis safety
    constraints described in codon.py's own module docstring (GC content
    window, homopolymer runs, hairpins, repeated k-mers, common Golden
    Gate enzyme sites). See that docstring for exactly what this is and
    isn't meant to replace -- short version: a strong, safe-by-default
    starting sequence per candidate, meant to still be manually reviewed
    before ordering, not a fully automated expression-vector assembly.

    predict_df doesn't carry the amino acid sequence itself (see
    _PREDICT_COLUMNS) -- join_predict_with_pmpnn() recovers it from
    pmpnn_df first. If predict_df/pmpnn_df aren't given, loads them from
    <state_dir>/predict.pkl and <state_dir>/pmpnn.pkl.

    Saves the result to <state_dir>/codon.{pkl,csv} AND an orderable
    FASTA (default <state_dir>/codon.fasta, override with fasta_path=)
    with one record per successfully optimized candidate. Returns the
    same DataFrame that was saved.

    A candidate is skipped (reported, not fatal to the rest of the
    batch -- same fail-soft convention as run_pmpnn()/run_predict()) if
    its sequence couldn't be recovered from pmpnn_df (checkpoint
    mismatch) or if DNAChisel couldn't produce a valid sequence for it.
    """
    from toolkit import codon as _codon

    if predict_df is None:
        predict_df = load_checkpoint(PREDICT_STAGE, state_dir, needed_by="Codon optimization")
    if pmpnn_df is None:
        pmpnn_df = load_checkpoint(PMPNN_STAGE, state_dir, needed_by="Codon optimization")
    if predict_df.empty:
        raise ValueError(
            "predict.pkl has no validated candidates to codon-optimize -- check `symbro predict`'s "
            "own output, or loosen its --max-rmsd/--min-plddt and rerun it."
        )

    subset = predict_df
    if assembly_id is not None:
        subset = subset[subset["assembly_id"] == assembly_id]
        if subset.empty:
            raise ValueError(f"No row for assembly_id={assembly_id!r} in the predict checkpoint.")

    joined = join_predict_with_pmpnn(subset, pmpnn_df)
    df = _codon.optimize_candidates(
        joined, host=host or _codon.DEFAULT_HOST, method=method, gc_min=gc_min, gc_max=gc_max,
        gc_window=gc_window, homopolymer_max=homopolymer_max, avoid_hairpins=avoid_hairpins,
        avoid_repeats_kmer=avoid_repeats_kmer,
        avoid_enzymes=avoid_enzymes if avoid_enzymes is not None else _codon.DEFAULT_ENZYMES_AVOIDED,
        add_stop_codon=add_stop_codon,
    )
    save_checkpoint(df, CODON_STAGE, state_dir)

    if fasta_path is None:
        fasta_path = os.path.join(state_dir, "codon.fasta")
    if not df.empty:
        _codon.write_fasta(df, fasta_path)
        print(f"Wrote orderable FASTA ({len(df)} record(s)) to {fasta_path}")

    return df


# ----------------------------------------------------------------------
# Cleanup — the "start fresh" button, between pipeline runs
# ----------------------------------------------------------------------

_ALL_STAGES: Tuple[str, ...] = (
    CANDIDATES_STAGE, DOWNLOADED_STAGE, GEOMETRY_STAGE, ISOLATE_STAGE, RFDIFFUSION_STAGE, PMPNN_STAGE,
    PREDICT_STAGE, CODON_STAGE,
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
                     rfdiffusion, pmpnn, predict, codon), plus codon's own
                     extra codon.fasta (not a checkpoint file itself, but
                     exactly the kind of stale reference back to a
                     just-cleared run this same warning is about below)
                     -- state_dir itself is left in place, only the files
                     inside it go.
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

        # codon.fasta isn't a <stage>.pkl/.csv pair -- save_checkpoint() never
        # wrote it, run_codon() did, alongside codon.pkl/.csv -- but leaving
        # it behind after codon's own checkpoint is cleared is exactly the
        # stale-reference situation this function's own docstring warns
        # about, so it's cleared here too.
        fasta_path = os.path.join(state_dir, "codon.fasta")
        if os.path.exists(fasta_path):
            if not dry_run:
                os.remove(fasta_path)
            if CODON_STAGE not in cleared["state"]:
                cleared["state"].append(CODON_STAGE)

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