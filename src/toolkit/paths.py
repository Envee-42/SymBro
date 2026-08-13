"""
paths.py — makes filesystem paths stored in pickled checkpoints
(.symbro/*.pkl) portable across machines.

The problem: download.py and isolate.py both build their default output
directories as os.path.join(os.getcwd(), <name>) and then store the
resulting absolute path directly into a DataFrame column that gets
pickled to disk (downloaded.pkl's/rings.pkl's "filepath" column). An
absolute path baked in on one machine (e.g. a Windows laptop) means
nothing on another (e.g. a Linux HPC login node) -- moving .symbro/ plus
the referenced files to a new machine silently breaks every downstream
stage that reads that column, exactly the bug that hit rings.pkl.

The fix relies on a convention this project already requires for the
pipeline to work AT ALL: every `symbro` command (and every stage function
in pipeline.py) resolves its default state/data/output directories off
os.getcwd(), i.e. it already assumes you run from the project root. Given
that, storing paths RELATIVE to that same root -- instead of absolute --
and resolving them back to absolute only at the point of actual file I/O,
makes checkpoints portable with no new requirement on how the CLI is
already used, and no change needed in any downstream consumer (RFdiffusion/
ProteinMPNN's own path handling already calls os.path.abspath() on
whatever they're given before using it, so a relative path just resolves
correctly against whichever machine's CWD is current at that moment).
"""

import ntpath
import os
import posixpath


def _is_absolute_anywhere(path: str) -> bool:
    """
    True if `path` is absolute under EITHER Windows or POSIX path rules --
    not just whichever OS this process happens to be running on right now.

    This matters specifically for backward compatibility: a checkpoint's
    filepath column can carry a path baked in on a DIFFERENT OS than the
    one reading it back -- e.g. a Windows-absolute path like
    "C:\\Users\\..." written before this module existed, later read on a
    Linux cluster. Plain os.path.isabs() is OS-native: on POSIX it only
    recognizes a leading "/", so it wrongly judges a perfectly absolute
    Windows path to be relative -- which then makes resolve_path()
    os.path.join() it onto the current (Linux) cwd instead of leaving an
    already-absolute path alone, corrupting it into a garbage nested path
    instead of just failing loudly on the original, recognizable one.

    ntpath/posixpath are the standard library's OS-independent path
    parsers (what os.path actually IS, aliased to whichever one matches
    the running platform) -- using both directly, regardless of platform,
    is exactly what's needed here.
    """
    return ntpath.isabs(path) or posixpath.isabs(path)


def to_portable(path: str, base: str = None) -> str:
    """
    Converts `path` to one relative to `base` (default: the current
    working directory) for storage in a checkpoint DataFrame column.

    Falls back to returning `path` unchanged (absolute) if it can't be
    expressed relative to `base` -- e.g. a custom output/data dir given
    explicitly outside the project tree, or a different drive on Windows
    (os.path.relpath raises ValueError there). That's a no-op, not a
    regression: every path is stored absolute today anyway, so this only
    ever narrows -- never widens -- the set of paths that fail to survive
    a move to another machine.
    """
    base = base or os.getcwd()
    try:
        rel = os.path.relpath(path, base)
    except ValueError:
        return path
    # A path outside `base` relpaths to something starting with ".." --
    # keep those stored absolute too, since a ".."-prefixed path is just
    # as machine-specific as an absolute one (it still depends on `base`
    # living in the same place on the next machine).
    return path if rel.startswith(os.pardir) else rel


def resolve_path(path: str, base: str = None) -> str:
    """
    Resolves `path` (as read back from a checkpoint -- relative under
    this scheme, or absolute for backward compatibility with checkpoints
    written before it existed) to an absolute path, right before it's
    actually used for file I/O.

    A no-op for an already-absolute path -- checked against BOTH Windows
    and POSIX rules (see _is_absolute_anywhere), not just whichever OS
    this process happens to be running on, so a checkpoint written on one
    OS and read back on another still gets left alone here instead of
    being silently corrupted into a bogus joined path. It won't
    necessarily FIND that file (a Windows path genuinely doesn't exist on
    Linux), but it fails loudly and recognizably instead of pointing
    somewhere wrong. Old same-OS checkpoints keep working exactly as
    before -- no migration step required.
    """
    base = base or os.getcwd()
    return path if _is_absolute_anywhere(path) else os.path.normpath(os.path.join(base, path))
