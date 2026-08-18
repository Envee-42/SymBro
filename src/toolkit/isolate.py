"""
isolate.py — extracts the MAIN ring grouping out of a source assembly into
its own small, standalone structure file, ready to hand to RFdiffusion,
ProteinMPNN, a folding backend, or ChimeraX (see
external_tools_architecture.md's "Chain isolation" section — this module
is that ring-extraction step).

Design, in short:

- Accepts ANY of rings.py's/orientation.py's/structure.py's ring-grain
  outputs (a distance df from detect_symmetry_rings/from_structure, an
  orientation df from orientation.from_rings, or a termini-SS df from
  structure.from_rings) interchangeably — they all share the three
  columns this module actually needs: assembly_id, symmetry_type,
  chain_groups. rings.py's raw output additionally carries
  "equivalent_groups" (every OTHER accepted grouping of the SAME
  identity component's axis — see rings.py's module docstring) —
  deliberately never read here. "the main ring grouping" always means
  chain_groups, the single tightest, disjoint-resolved grouping rings.py
  already selected as the representative one for that (order, component)
  pair; its equivalent copies are, by construction, physically redundant
  repeats of the identical junction geometry FOR THAT SAME COMPONENT, so
  isolating every one of them would just be duplicate work for
  RFdiffusion/ProteinMPNN/folding to redo on what is structurally the
  same ring.

  Since rings.py emits one row per (symmetry_type, component) rather
  than one row per symmetry_type, a multi-component assembly (e.g. a
  two-protein T:3,3 cage) already arrives here as multiple rows for the
  same symmetry_type — one per component — and this module isolates
  every row it's given (subject to the optional component_id= filter
  below), so every structurally distinct component gets its own
  extracted file automatically; nothing extra is needed to "pick up"
  a second component. component_id/recommended_linker_length are carried
  through to this module's own output (as pass-through metadata, filled
  with None if the input df predates them) purely so a caller can still
  see which component a given isolated file came from, and so
  pipeline.run_rfdiffusion can default that ring's diffused-linker length
  to its own measured geometry instead of a fixed guess.

- Output format: legacy PDB is symbro's TRUE default in practice — RFdiffusion,
  the very next stage every isolated ring feeds into, only accepts PDB
  input, and `symbro isolate`/from_rings() both default file_format to
  "pdb" accordingly (_write_ring_structure()/extract_ring_structure()/
  isolate_assembly_rings() still individually default to "cif" — call one
  of those directly if you want that instead, e.g. for a non-RFdiffusion
  destination). mmCIF remains available (file_format="cif") for the case
  that motivated offering it at all: RCSB assembly files commonly carry
  chain names longer than the legacy PDB format's 1-2 character chain-ID
  field allows — e.g. the symmetry-expanded names a large multi-copy
  assembly produces (a 24-mer ferritin's chains come back named things
  like "A-13", "A-18"). mmCIF has no such length restriction, so nothing
  needs renaming to write it, and the exact chain names rings.py/
  orientation.py/structure.py already reported are exactly what shows up
  in the extracted file too — no translation table to keep straight
  later. Writing PDB, by contrast, requires shortening any over-length
  chain name to fit, so that path always returns an explicit
  {original_name: short_name} mapping alongside the file — never a
  silent, undiscoverable rename.

- Follows download.py's "workspace-visible scratch folder" philosophy,
  under its own name: temporary_subunits/ (a sibling of download.py's
  temporary_files/, not nested inside it, since these are EXTRACTED
  PRODUCTS of a completed analysis, not raw downloads) — so isolated ring
  files are just as easy to browse, open, and clear out as the assemblies
  they came from, and open directly in ChimeraX per the project's own
  downstream workflow.

- Only ever reads structure[0] (the first model) — the same single-model
  assumption rings.py/orientation.py/structure.py already make throughout
  this project.

- Same efficiency discipline as the rest of the project: a source
  structure file is parsed with gemmi ONCE per assembly and reused for
  every ring row that assembly has (e.g. both its C2 and C3 rows), rather
  than re-parsing the file once per row.
"""

import os
import string
from typing import Dict, Optional, Sequence, Tuple, Union

import gemmi
import pandas as pd

from toolkit.paths import resolve_path, to_portable


TEMP_SUBUNITS_DIR_NAME: str = "temporary_subunits"

_REQUIRED_COLUMNS: Tuple[str, ...] = ("assembly_id", "symmetry_type", "chain_groups")
# component_id/recommended_linker_length are optional pass-through --
# present on any rings_df from the current rings.py, None if the caller
# supplies an older df that predates them (see rings.py's module
# docstring for why they exist: scoping ring exclusivity per identity
# component rather than pooling every component together).
_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "assembly_id", "symmetry_type", "component_id", "chain_groups",
    "recommended_linker_length", "filepath", "chain_rename_map",
)

_SANITIZE_CHARS: str = '/\\:*?"<>|'


# ============================================================================
# Scratch-folder management — mirrors download.py's temporary_files/ exactly
# ============================================================================

def get_temp_subunits_dir() -> str:
    """
    Returns the absolute path to the temporary_subunits/ folder, creating
    it if it doesn't exist yet — resolved relative to the current working
    directory, same convention download.py's get_temp_dir() uses, so
    isolated ring files show up right in your workspace rather than an
    OS-managed temp directory.
    """
    path = os.path.join(os.getcwd(), TEMP_SUBUNITS_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def clear_temp_subunits_dir(data_dir: Optional[str] = None) -> None:
    """Deletes everything INSIDE temporary_subunits/ (or a custom
    data_dir) but keeps the folder itself — same "easy to empty, stays
    ready to use" behavior as download.py's clear_temp_dir()."""
    import shutil

    if data_dir is None:
        data_dir = get_temp_subunits_dir()

    for entry in os.listdir(data_dir):
        if entry.startswith('.'):
            continue
        entry_path = os.path.join(data_dir, entry)
        if os.path.isfile(entry_path) or os.path.islink(entry_path):
            os.remove(entry_path)
        else:
            shutil.rmtree(entry_path)


# ============================================================================
# Naming helpers
# ============================================================================

def _sanitize(text: str) -> str:
    """Strips filesystem-unfriendly characters out of an assembly_id or
    chain name before it becomes part of a filename. RCSB IDs and chain
    names don't normally contain these, but a stray "/" would otherwise
    silently turn into an unintended subdirectory."""
    for ch in _SANITIZE_CHARS:
        text = text.replace(ch, "_")
    return text


def _ring_filename(assembly_id, symmetry_type: str, chain_group: Sequence[str], file_format: str) -> str:
    chains_part = "-".join(_sanitize(str(c)) for c in chain_group)
    stem = f"{_sanitize(str(assembly_id))}_{symmetry_type}_{chains_part}"
    ext = "cif" if file_format == "cif" else "pdb"
    return f"{stem}.{ext}"


def _short_chain_id(index: int) -> str:
    """Excel-style base-26 letter ID: 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z',
    26 -> 'AA', ... rings.py only ever produces groups up to order 5, so
    this never needs more than one letter in practice, but a group of any
    size is handled without collision."""
    letters = string.ascii_uppercase
    index += 1
    result = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = letters[remainder] + result
    return result


# ============================================================================
# Core extraction — operates on an already-parsed structure/model
# ============================================================================

def _load_single_model(filepath: str) -> Tuple[gemmi.Structure, gemmi.Model]:
    """Reads `filepath` and drops every model but the first — the same
    single-model assumption rings.py/orientation.py/structure.py make.

    `filepath` may be relative (as stored in a checkpoint's filepath
    column -- e.g. rings.pkl, or downloaded.pkl if this is reading a
    source assembly directly -- see paths.py) or absolute (older
    checkpoints, or a caller-supplied path); resolved to absolute here,
    the single point every filepath this module reads passes through.
    """
    structure = gemmi.read_structure(resolve_path(filepath))
    while len(structure) > 1:
        del structure[-1]
    return structure, structure[0]


def _write_ring_structure(
    source: gemmi.Structure, model: gemmi.Model, chain_group: Sequence[str],
    output_path: str, file_format: str = "cif",
) -> Tuple[str, Dict[str, str]]:
    """
    Builds a new, single-model structure containing ONLY the chains named
    in `chain_group` — added in chain_group's own order (the physical
    ring order rings.py/orientation.py established), not necessarily the
    source file's chain order — and writes it to output_path.

    Operates on an already gemmi.read_structure()'d `source`/`model` so
    callers extracting several ring rows from the same assembly (see
    isolate_assembly_rings below) never re-parse the file per row.

    Returns (output_path, rename_map). rename_map is {} for
    file_format="cif" (mmCIF has no chain-name length limit, so nothing
    is ever renamed). For file_format="pdb", any chain whose name doesn't
    already fit the legacy PDB chain-ID field gets reassigned a short
    A/B/C/... id (in chain_group order) and rename_map records
    {original_name: short_id} for every renamed chain — empty if every
    chain in the group already had a short enough name.

    Raises KeyError (naming the offending chains) if any chain in
    chain_group isn't present in `model` — same contract
    orientation.py's compute_ring_orientation / structure.py's
    compute_ring_termini_ss use, meaning the structure passed in doesn't
    match the one this grouping was detected from. Raises ValueError for
    an unrecognized file_format.
    """
    if file_format not in ("cif", "pdb"):
        raise ValueError(f"file_format must be 'cif' or 'pdb' — got {file_format!r}")

    missing = [name for name in chain_group if model.find_chain(name) is None]
    if missing:
        raise KeyError(
            f"chain(s) {missing} (from chain_group {tuple(chain_group)!r}) not found in the "
            f"structure — is this the same structure this grouping was detected from?"
        )

    extracted = gemmi.Structure()
    extracted.name = source.name
    extracted.cell = source.cell
    extracted.spacegroup_hm = source.spacegroup_hm

    new_model = gemmi.Model(model.num)
    for name in chain_group:
        new_model.add_chain(model.find_chain(name).clone())
    extracted.add_model(new_model)
    extracted.setup_entities()

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    rename_map: Dict[str, str] = {}
    if file_format == "pdb":
        # PDB's chain-ID field is 1 character (up to 2 in some lenient
        # readers) -- reassign every chain a short, deterministic id in
        # chain_group order rather than relying on gemmi's own
        # shorten_chain_names(), so the mapping returned here is exactly
        # and always what was applied, never an opaque internal scheme.
        for i, chain in enumerate(extracted[0]):
            short_id = _short_chain_id(i)
            if chain.name != short_id:
                rename_map[chain.name] = short_id
                chain.name = short_id
        extracted.write_pdb(output_path)
    else:
        extracted.make_mmcif_document().write_file(output_path)

    return output_path, rename_map


def extract_ring_structure(
    filepath: str, chain_group: Sequence[str], output_path: str, file_format: str = "cif",
) -> Tuple[str, Dict[str, str]]:
    """
    Single-shot convenience: parses `filepath` and extracts one
    chain_group ring straight to `output_path` in one call — handy for
    interactive/notebook use (e.g. against a single row you've already
    picked out of orientation_df), without going through the
    temporary_subunits/ batch pipeline below.

    See _write_ring_structure for file_format / rename_map / raised
    errors.
    """
    source, model = _load_single_model(filepath)
    return _write_ring_structure(source, model, chain_group, output_path, file_format=file_format)


# ============================================================================
# Per-assembly / per-batch orchestration
# ============================================================================

def _validate_ring_df(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing expected column(s): {missing}")


def isolate_assembly_rings(
    filepath: str, rings_df: pd.DataFrame, assembly_id: Optional[str] = None,
    output_dir: Optional[str] = None, file_format: str = "cif",
) -> pd.DataFrame:
    """
    Single-structure pipeline: loads the structure at `filepath` ONCE,
    then extracts every row of rings_df's chain_groups ring into its own
    file under output_dir — reusing the one parsed structure for every
    ring row, the same "parse once, reuse for every row of that assembly"
    discipline orientation.py's compute_assembly_orientations /
    structure.py's compute_assembly_termini_ss already follow.

    rings_df here is expected to already be narrowed to rows for a SINGLE
    assembly (any mix of symmetry_type is fine — e.g. both its C2 and C3
    rows) — exactly the per-assembly slice from_rings hands this function
    below. Accepts dist_df/orientation_df/termini_ss_df equally, per the
    module docstring.

    A row whose chain_group isn't actually present in this structure is
    reported and skipped (fail-soft, same convention as the rest of this
    project) rather than aborting the other rows.

    Returns a DataFrame (assembly_id, symmetry_type, component_id,
    chain_groups, recommended_linker_length, filepath, chain_rename_map)
    — one row per successfully extracted ring (component_id/
    recommended_linker_length are None if rings_df predates them).
    Empty (but correctly-columned) if nothing was extracted.
    """
    _validate_ring_df(rings_df)

    if output_dir is None:
        output_dir = get_temp_subunits_dir()

    source, model = _load_single_model(filepath)

    rows = []
    for _, row in rings_df.iterrows():
        chain_group = tuple(row["chain_groups"])
        symmetry_type = row["symmetry_type"]
        row_assembly_id = row["assembly_id"] if assembly_id is None else assembly_id

        output_path = os.path.join(output_dir, _ring_filename(row_assembly_id, symmetry_type, chain_group, file_format))
        try:
            written_path, rename_map = _write_ring_structure(source, model, chain_group, output_path, file_format=file_format)
        except KeyError as exc:
            print(f"Failed: {row_assembly_id} ({symmetry_type}) — {exc}")
            continue

        rows.append({
            "assembly_id": row_assembly_id,
            "symmetry_type": symmetry_type,
            "component_id": row.get("component_id"),  # pandas Series.get() -- None if the column is absent
            "chain_groups": chain_group,
            "recommended_linker_length": row.get("recommended_linker_length"),
            # Stored relative to the current working directory (see
            # paths.py) so rings.pkl survives a move to another machine --
            # resolved back to absolute by _load_single_model() (and, for
            # any downstream consumer, e.g. rfdiffusion.py, automatically
            # via the normal OS relative-path-resolves-against-CWD
            # behavior, since that's exactly the same convention this
            # whole pipeline already relies on for state_dir/data_dir/
            # output_dir).
            "filepath": to_portable(written_path),
            "chain_rename_map": rename_map,
        })

    return pd.DataFrame(rows, columns=list(_OUTPUT_COLUMNS))


def from_rings(
    rings_df: pd.DataFrame, structures: Union[pd.DataFrame, Dict[str, str]], symmetry_type: Optional[str] = None,
    component_id: Optional[int] = None,
    filepath_column: str = "filepath", assembly_id_column: str = "assembly_id",
    output_dir: Optional[str] = None, file_format: str = "pdb",
) -> pd.DataFrame:
    """
    Batch entry point, mirroring orientation.py's/structure.py's
    from_rings: groups rings_df by assembly_id and runs
    isolate_assembly_rings() once per assembly, so each source structure
    file is parsed exactly once regardless of how many ring rows it has.
    THIS is the function most callers want — the direct answer to "take
    my dist_df / orientation_df / termini_ss_df and isolate every main
    ring into temporary_subunits/":

        isolate.from_rings(orientation_df, downloaded_df)

    rings_df    : dist_df (rings.py), orientation_df (orientation.py), or
        termini_ss_df (structure.py) — or any DataFrame with
        assembly_id/symmetry_type/chain_groups columns. Only chain_groups
        (the main grouping) is ever isolated; a raw dist_df's
        "equivalent_groups" column, if present, is never read.
    structures  : where to find each assembly's SOURCE structure file —
        either a DataFrame with an assembly_id_column and a
        filepath_column (e.g. download.py's output), or a plain
        {assembly_id: filepath} dict.
    symmetry_type : optionally narrow to ONE symmetry type first (e.g.
        "C3") before isolating — same "one axis order at a time"
        contract orientation.py/structure.py use. Pass None (default) to
        isolate every row regardless of order — the natural choice once
        you've already filtered rings_df down to exactly the rows you
        want extracted.
    component_id : optionally narrow to ONE structurally distinct
        component first (rings.py's component_id — see its module
        docstring), e.g. to isolate only one protein of a multi-component
        cage instead of every component found. Pass None (default) to
        isolate every component — the right choice for most callers,
        since each component is a genuinely different ring that normally
        DOES need its own RFdiffusion/ProteinMPNN run. Raises ValueError
        if rings_df has no component_id column (i.e. it predates
        rings.py's per-component grouping) — there's nothing to filter on.
    output_dir  : defaults to temporary_subunits/ in the current working
        directory (get_temp_subunits_dir()).
    file_format : "pdb" (default here — RFdiffusion's own actual input
        requirement, and what `symbro isolate` always ends up passing in
        practice) or "cif" (module docstring's original mmCIF rationale
        still applies if you're headed somewhere other than RFdiffusion —
        see module docstring). Note this default deliberately differs
        from _write_ring_structure()/extract_ring_structure()/
        isolate_assembly_rings(), which still default to "cif" — pass
        file_format= explicitly if you're calling one of those directly
        and care which one you get.

    An assembly present in rings_df but missing from `structures`, or
    whose structure fails to load, is reported by assembly_id and
    skipped rather than aborting the whole batch — same fail-soft
    convention as the rest of this project.

    Returns a DataFrame (assembly_id, symmetry_type, component_id,
    chain_groups, recommended_linker_length, filepath, chain_rename_map),
    one row per successfully extracted ring. Empty (but correctly-
    columned) if nothing was extracted.
    """
    _validate_ring_df(rings_df)

    subset = rings_df
    if symmetry_type is not None:
        subset = subset[subset["symmetry_type"] == symmetry_type]
    if component_id is not None:
        if "component_id" not in subset.columns:
            raise ValueError(
                "component_id filter was given, but rings_df has no 'component_id' column -- "
                "it predates rings.py's per-component grouping (see rings.py's module docstring), "
                "so there's nothing to filter on. Re-run `symbro geometry` to regenerate it."
            )
        subset = subset[subset["component_id"] == component_id]
    subset = subset.reset_index(drop=True)

    if subset.empty:
        return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))

    if isinstance(structures, pd.DataFrame):
        filepath_by_assembly = dict(zip(structures[assembly_id_column], structures[filepath_column]))
    else:
        filepath_by_assembly = dict(structures)

    if output_dir is None:
        output_dir = get_temp_subunits_dir()

    frames = []
    for aid, group_df in subset.groupby("assembly_id", sort=False):
        filepath = filepath_by_assembly.get(aid)
        if filepath is None:
            print(f"Failed: {aid} — no filepath found in `structures`")
            continue
        try:
            frames.append(
                isolate_assembly_rings(filepath, group_df, assembly_id=aid, output_dir=output_dir, file_format=file_format)
            )
        except Exception as exc:
            print(f"Failed: {aid} — {exc}")

    if not frames:
        return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))

    return pd.concat(frames, ignore_index=True)