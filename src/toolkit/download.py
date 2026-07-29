"""
download.py — fetches candidate structure files from the PDB.

Design goal (per discussion): downloaded structures live in a
`temporary_files/` folder inside the project workspace — not an OS-managed
temp directory hidden away in /tmp somewhere. That means you can browse,
open, and inspect the files directly in your editor/file explorer while
you're working, and clear them out with one function call once you're done
with a given batch. A structure only becomes permanent once you explicitly
promote it via save_structures().
"""

import os
import gzip
import shutil
import requests
import pandas as pd


# RCSB's download URL pattern for a specific biological assembly, gzip-compressed.
# {pdb_id} must be lowercase for this endpoint to resolve correctly.
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}-assembly{assembly_num}.cif.gz"

# Name of the workspace-visible scratch folder. Kept as a module-level
# constant (rather than hardcoded inline) so if you ever want to rename it,
# there's exactly one place to change.
TEMP_DIR_NAME = "temporary_files"


def get_temp_dir():
    """
    Returns the absolute path to the temporary_files/ folder, creating it
    if it doesn't exist yet. Resolved relative to the current working
    directory — i.e. wherever you're running your script from — which is
    normally your project root, so the folder shows up right there in your
    workspace/editor rather than somewhere in the OS's hidden temp space.
    """
    path = os.path.join(os.getcwd(), TEMP_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def parse_assembly_id(assembly_id):
    """
    Splits an RCSB assembly ID (e.g. "6XYZ-1") into its two components:
    entry_id ("6XYZ") and assembly_num ("1"). This is exactly the ID format
    search_ids() returns when return_type="assembly", and exactly what
    fetch_metadata()/query_candidates() name their "assembly_id" column
    (query.py's Data API fetch always surfaces the record's own rcsb_id
    under an "<input_type>_id" column — "assembly_id" for the default
    return_type="assembly").

    rsplit("-", 1) splits from the RIGHT, at most once, so this stays
    correct even in the (currently unlikely, but not impossible) case of an
    entry_id that itself contains a hyphen — only the final "-<number>"
    segment gets peeled off as the assembly number.
    """
    entry_id, assembly_num = assembly_id.rsplit("-", 1)
    return entry_id, assembly_num


def build_download_table(assembly_ids, data_dir=None):
    """
    Takes the raw list of assembly ID strings — exactly what search_ids()
    returns, e.g. ["6XYZ-1", "6XYZ-2", ...] — and builds a DataFrame with
    everything needed to download each one: assembly_id, entry_id,
    assembly_num, filepath, and url.

    This is the piece that was missing before: nothing was actually
    splitting the combined assembly ID into its parts. Call this directly
    on your search results, or just use download_candidates() below, which
    calls this for you automatically when given a plain ID list.
    """
    if data_dir is None:
        data_dir = get_temp_dir()

    rows = []
    for assembly_id in assembly_ids:
        entry_id, assembly_num = parse_assembly_id(assembly_id)
        rows.append({
            "assembly_id": assembly_id,
            "entry_id": entry_id,
            "assembly_num": assembly_num,
            "filepath": os.path.join(data_dir, f"{entry_id}-assembly{assembly_num}.cif"),
            # RCSB's download endpoint requires the PDB code in lowercase —
            # note this is ONLY applied here, for the URL. The filepath above
            # keeps entry_id in its original case, so your files on disk stay
            # named consistently with however entry_id normally appears
            # elsewhere (e.g. in a metadata DataFrame).
            "url": RCSB_DOWNLOAD_URL.format(pdb_id=entry_id.lower(), assembly_num=assembly_num),
        })
    return pd.DataFrame(rows)


def add_download_columns(df, id_column="entry_id", assembly_column="assembly_num", combined_id_column="assembly_id", data_dir=None):
    """
    Adds 'filepath' and 'url' columns to a metadata DataFrame — WITHOUT
    dropping any of its other columns (title, weight, symmetry, whatever
    else you fetched) — so the result can be handed straight to
    distance.py's run_ring_analysis() with all of that metadata still
    attached, rather than having to re-merge it back in by hand afterward.

    Two input shapes are accepted, checked in this order:

      1. df already has separate `id_column`/`assembly_column` columns
         (e.g. hand-assembled metadata with those two fields already split
         out) — used directly, exactly as before.

      2. df has a single `combined_id_column` (default "assembly_id" —
         precisely what query.py's fetch_metadata()/query_candidates()
         name their ID column when return_type="assembly") holding
         combined IDs like "6XYZ-1". These get split into `id_column` /
         `assembly_column` via parse_assembly_id() first, so a
         query_candidates() result can be passed in here directly with no
         manual ID-wrangling step in between.

    Raises ValueError if df has neither shape — i.e. no `id_column`/
    `assembly_column` pair AND no `combined_id_column` to split.
    """
    if data_dir is None:
        data_dir = get_temp_dir()

    df = df.copy()  # avoid silently mutating the caller's original DataFrame

    if id_column not in df.columns or assembly_column not in df.columns:
        if combined_id_column not in df.columns:
            raise ValueError(
                f"df must have either '{id_column}'/'{assembly_column}' columns "
                f"(already-split entry ID + assembly number), or a combined "
                f"'{combined_id_column}' column (e.g. 'assembly_id', exactly what "
                f"query.py's fetch_metadata()/query_candidates() produce) to split "
                f"via parse_assembly_id() — df has neither."
            )
        parsed = df[combined_id_column].apply(parse_assembly_id)
        df[id_column] = parsed.apply(lambda pair: pair[0])
        df[assembly_column] = parsed.apply(lambda pair: pair[1])

    df["filepath"] = df.apply(
        lambda row: os.path.join(
            data_dir, f"{row[id_column]}-assembly{row[assembly_column]}.cif"
        ),
        axis=1,
    )
    df["url"] = df.apply(
        lambda row: RCSB_DOWNLOAD_URL.format(
            pdb_id=str(row[id_column]).lower(), assembly_num=row[assembly_column]
        ),
        axis=1,
    )
    return df


def download_structure(filepath, url, overwrite=False):
    """
    Downloads one gzip-compressed assembly file from `url` and writes the
    decompressed .cif contents to `filepath`.

    overwrite : if False (default) and filepath already exists, skips the
        download entirely — safe to re-run a download loop repeatedly
        without re-fetching everything each time.
    """
    if os.path.exists(filepath) and not overwrite:
        return filepath

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    response = requests.get(url)

    # raise_for_status() raises an exception if the server returned an error
    # (e.g. 404 for a withdrawn or renumbered entry) instead of silently
    # treating an HTML error page as if it were the file content — without
    # this, gzip.decompress below would crash confusingly or, worse, write
    # garbage to disk without any warning.
    response.raise_for_status()

    content = gzip.decompress(response.content)
    with open(filepath, "wb") as f:
        f.write(content)

    return filepath


def download_candidates(candidates, data_dir=None, id_column="entry_id", assembly_column="assembly_num", combined_id_column="assembly_id", overwrite=False):
    """
    Downloads every candidate structure into temporary_files/ (or a custom
    data_dir), and returns a DataFrame describing what was downloaded.

    candidates : any of
        - a plain list of raw assembly ID strings, exactly as returned by
          search_ids() — e.g. ["6XYZ-1", "6XYZ-2"]. Internally this gets
          split into entry_id/assembly_num via build_download_table.
        - a DataFrame straight out of query.py's fetch_metadata() /
          query_candidates() — i.e. one with a combined "assembly_id"
          column (or whatever combined_id_column names it) rather than
          separate entry_id/assembly_num columns. This is the common case
          once you're filtering candidates by metadata before downloading
          them: hand the (possibly filter_metadata()-narrowed) DataFrame
          straight in here, with no manual ID-splitting step in between —
          every other column it carries (title, weight, symmetry, ...)
          comes along for the ride in the result too.
        - a DataFrame that already has id_column/assembly_column columns
          separately (e.g. hand-built metadata) — handled the same way as
          before.
      Both DataFrame shapes are routed through add_download_columns(),
      which is what actually tells them apart (see its docstring); this
      function only decides list-vs-DataFrame.

    Returns a DataFrame with assembly_id (if built from a list, or however
    the input DataFrame named it), entry_id, assembly_num, filepath, and
    url columns, PLUS every other column the input DataFrame already had —
    filepath is what you hand to gemmi or distance.py next.
    """
    if data_dir is None:
        data_dir = get_temp_dir()

    if isinstance(candidates, pd.DataFrame):
        df = add_download_columns(
            candidates, id_column=id_column, assembly_column=assembly_column,
            combined_id_column=combined_id_column, data_dir=data_dir,
        )
    else:
        df = build_download_table(candidates, data_dir=data_dir)

    for _, row in df.iterrows():
        download_structure(row["filepath"], row["url"], overwrite=overwrite)

    return df


def save_structures(df, destination, filepath_column="filepath"):
    """
    Copies files referenced in df[filepath_column] into a permanent
    destination folder, outside temporary_files/. Use this once you've
    identified candidates worth keeping, before clearing the scratch folder.

    Returns the list of new permanent file paths.
    """
    os.makedirs(destination, exist_ok=True)
    saved_paths = []
    for filepath in df[filepath_column]:
        dest_path = os.path.join(destination, os.path.basename(filepath))
        shutil.copy2(filepath, dest_path)  # copy2 preserves file metadata (timestamps etc.)
        saved_paths.append(dest_path)
    return saved_paths


def clear_temp_dir(data_dir=None):
    """
    Deletes everything INSIDE temporary_files/ (or a custom data_dir) but
    keeps the folder itself in place — so it stays visible in your
    workspace as an empty, ready-to-use scratch folder rather than
    disappearing and needing to be recreated. This is the "easy to delete
    when the work is done" button.
    """
    if data_dir is None:
        data_dir = get_temp_dir()

    for entry in os.listdir(data_dir):
        entry_path = os.path.join(data_dir, entry)
        if entry.startswith('.'):
            continue

        if os.path.isfile(entry_path) or os.path.islink(entry_path):
            os.remove(entry_path)
        else:
            shutil.rmtree(entry_path)