"""
local.py — registers your own local PDB/CIF file(s) as pipeline
candidates, producing the same "downloaded" checkpoint shape
download.py's download_candidates() does.

Why this exists
----------------
Every stage from `symbro geometry` onward only needs a DataFrame with a
`filepath` column and an `assembly_id` column used purely as an opaque
label (see geometry/rings.py's from_structure() docstring) -- nothing
downstream actually requires the file to have come from RCSB, or for
assembly_id to look like a real RCSB one ("6XYZ-1"). This module is the
supported way to seed that checkpoint from a structure file you already
have on disk, instead of `symbro query` + `symbro download`.

Usage
-----
    from toolkit import local as _local
    df = _local.register_local_structures(["my_cage.pdb"])

or, from the CLI:

    symbro local my_cage.pdb
    symbro geometry
    symbro isolate
    ...

Each input file is COPIED into temporary_files/local/ (a subfolder of
download.py's own get_temp_dir()) rather than referenced in place --
same reasoning a real download gets stored under temporary_files/ to
begin with: paths.py's whole module exists because a checkpoint holding
a path outside the project tree doesn't survive a move to another
machine. Using download.py's own get_temp_dir() as the parent also means
`symbro clean`'s existing --keep-downloads flag already covers locally-
registered structures too, with no changes needed there.
"""

import os
import re
import shutil
from typing import Optional, Sequence

import pandas as pd

from toolkit.download import get_temp_dir
from toolkit.paths import to_portable

LOCAL_SUBDIR_NAME = "local"

# Anything that isn't alphanumeric/underscore/hyphen/dot collapses to "_"
# -- a file saved by hand off e.g. a browser download often has spaces,
# parentheses, or a "(1)" suffix in its name, none of which are safe to
# carry through into an assembly_id that ends up in filenames downstream
# (isolate.py/rfdiffusion.py both build output paths directly from it).
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _default_assembly_id(path: str, taken: set) -> str:
    """
    Derives an assembly_id from a local file's own name (its stem,
    sanitized -- see _SANITIZE_RE), deduplicated against `taken` by
    appending "-2", "-3", ... if two input files sanitize to the same
    stem (e.g. "cage.pdb" from two different folders).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = _SANITIZE_RE.sub("_", stem).strip("_.") or "structure"
    candidate = stem
    n = 1
    while candidate in taken:
        n += 1
        candidate = f"{stem}-{n}"
    taken.add(candidate)
    return candidate


def register_local_structures(
    paths: Sequence[str],
    assembly_ids: Optional[Sequence[str]] = None,
    data_dir: Optional[str] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Registers one or more local PDB/CIF files as pipeline candidates --
    an alternative to `symbro query` + `symbro download` for a structure
    you already have, rather than one you want RCSB to find for you.

    paths : local file path(s) -- .pdb or .cif. gemmi (geometry/rings.py's
        own reader) picks the parser by extension, same as it already
        does for a real RCSB download.

    assembly_ids : optional, one per path, same length as `paths` if
        given at all -- raises ValueError otherwise. Defaults to each
        file's own name (sanitized, deduplicated -- see
        _default_assembly_id): "my_cage.pdb" becomes "my_cage". Unlike a
        real RCSB assembly_id, nothing downstream requires any particular
        format here (no "<entry>-<number>" convention to match) -- pick
        whatever's meaningful to you. Raises ValueError on a duplicate
        within assembly_ids itself.

    data_dir : where copies get written. Default: temporary_files/local/.

    overwrite : if False (default) and a same-named copy already exists
        at the destination, it's left alone rather than re-copied -- same
        convention download_structure() uses for a real download.

    Raises FileNotFoundError (naming every missing path at once, not just
    the first) if any path in `paths` doesn't actually exist.

    Returns a DataFrame shaped like download_candidates()'s own output,
    minus the "url" column (nothing was fetched from anywhere): columns
    assembly_id, entry_id (same value as assembly_id here -- there's no
    separate RCSB entry-ID/assembly-number split for a local file),
    assembly_num (the literal string "local" for every row, so it's
    visibly distinguishable from a real RCSB-downloaded row in
    downloaded.csv), and filepath (portable -- see paths.py). run_query()/
    run_download()'s "-1"-suffixed downstream code, if any, is not
    involved: assembly_id is used everywhere else as an opaque label
    only, never re-split.
    """
    if assembly_ids is not None and len(assembly_ids) != len(paths):
        raise ValueError(
            f"assembly_ids must be the same length as paths if given at all "
            f"(got {len(assembly_ids)} assembly_ids for {len(paths)} paths)."
        )

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Local structure file(s) not found: {missing!r}")

    if data_dir is None:
        data_dir = os.path.join(get_temp_dir(), LOCAL_SUBDIR_NAME)
    os.makedirs(data_dir, exist_ok=True)

    taken = set()
    rows = []
    for i, src_path in enumerate(paths):
        if assembly_ids is not None:
            assembly_id = assembly_ids[i]
            if assembly_id in taken:
                raise ValueError(f"Duplicate assembly_id {assembly_id!r} in assembly_ids.")
            taken.add(assembly_id)
        else:
            assembly_id = _default_assembly_id(src_path, taken)

        ext = os.path.splitext(src_path)[1] or ".pdb"
        dest_path = os.path.join(data_dir, f"{assembly_id}{ext}")

        if overwrite or not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)
            print(f"Registered {assembly_id!r} <- {src_path}")
        else:
            print(f"Registered {assembly_id!r} <- {src_path} (destination already exists, kept as-is)")

        rows.append({
            "assembly_id": assembly_id,
            "entry_id": assembly_id,
            "assembly_num": "local",
            "filepath": to_portable(dest_path),
        })

    return pd.DataFrame(rows)
