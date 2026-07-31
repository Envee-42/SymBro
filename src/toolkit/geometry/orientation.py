"""
orientation.py — relative termini orientation for symmetry-grouped chains.

Takes rings.py's detect_symmetry_rings/from_structure output (one row per
detected symmetry axis per assembly: assembly_id, symmetry_type,
chain_groups, mean_distance, std_distance, junctions, axis_count,
equivalent_groups) and, for ONE symmetry_type chosen by the caller (never
all of them at once — a user analyzing a C3 axis usually isn't asking
about the C2 axes the same assembly may also have), builds a new
DataFrame narrowed to (assembly_id, symmetry_type, chain_groups,
mean_distance) with a new orientation column appended.

Design, in short:

- Reuses termini.get_chain_ca_geometry directly. Its n_vector/c_vector are
  ALREADY exactly "a vector built from the coordinates of the last
  vector_window (default 4) resolved residues at a chain's terminus" — no
  new geometry primitive is needed here, only a new use of the existing
  one. rings.py uses this same function's n/c COORDINATES for distance;
  this module uses its n_vector/c_vector DIRECTIONS for angle.

- chain_groups tuples are treated as a ring, the same way rings.py's own
  cycle-finding treats them: for a group of k chains, there are k
  consecutive junctions (i -> i+1, wrapping i=k-1 back to 0), each compared
  as chain_i's c_vector (the direction the chain exits AT its C-terminus)
  against chain_(i+1)'s n_vector (the direction the chain would continue
  PAST its N-terminus if extrapolated). For cyclic orders (C3/C4/C5) this
  exactly matches the C->N junction topology rings.py already validated by
  step-homogeneity and contact-checking, so chain_groups' stored order IS
  the real physical ring order.

  For C2, chain_groups is only ever an alphabetically-sorted pair
  (rings.py's `tuple(sorted((a, b)))`), and rings.py's final output does
  not retain which interface type (N-N, C-C, or N-C) was actually
  accepted for that pair. The same ring rule is still applied mechanically
  here (a.c_vector -> b.n_vector, b.c_vector -> a.n_vector) rather than
  re-deriving the true accepted interface — so treat a C2 row's
  orientation as a consistent, reproducible read of the pair's geometry,
  not a guaranteed match to whichever specific interface rings.py's
  select_disjoint_groupings happened to accept.

- Orientation is expressed as the angle, in degrees (0-180), between the
  two unit vectors at a junction: arccos of their dot product. 0 degrees
  means the two termini point the same way, 180 means they point directly
  opposite.

- One row per input (assembly_id, symmetry_type) pair is preserved (same
  grain as rings.py's output): "mean_orientation" is the mean junction
  angle for that group, and "orientation_junctions" lists every individual
  (from_chain, to_chain, angle_degrees) triple that went into it — the
  same mean/detail pairing rings.py itself uses for
  mean_distance/junctions.

- rings.py's output dataframe alone isn't enough to compute any of this:
  it never retains n/c VECTORS (only the summary distances derived from
  them), so this module always re-reads the source structure file and
  re-runs get_chain_ca_geometry per assembly. Callers supply a filepath
  for each assembly_id the same way rings.py's from_structure does.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import gemmi
import numpy as np
import pandas as pd

from toolkit.geometry.termini import get_chain_ca_geometry
from toolkit.geometry.rings import ALLOWED_ORDERS


_INPUT_COLUMNS: Tuple[str, ...] = ("assembly_id", "symmetry_type", "chain_groups", "mean_distance")
_OUTPUT_COLUMNS: Tuple[str, ...] = _INPUT_COLUMNS + ("mean_orientation", "orientation_junctions")

DEFAULT_VECTOR_WINDOW: int = 4


# ============================================================================
# Shared helpers
# ============================================================================

def _angle_degrees(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> Optional[float]:
    """
    Angle in degrees between two unit vectors, via arccos of their dot
    product (clipped to [-1, 1] to absorb floating-point drift outside
    arccos's domain — v1/v2 are already unit vectors from
    termini.get_chain_ca_geometry, but norm can drift by ~1e-16 either
    side of 1.0).

    Returns None if either vector is undefined — get_chain_ca_geometry
    returns n_vector/c_vector as None when a chain has only a single
    resolved CA, since direction has no meaning for one point.
    """
    if v1 is None or v2 is None:
        return None
    cos_angle = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def select_symmetry_type(rings_df: pd.DataFrame, symmetry_type: str) -> pd.DataFrame:
    """
    Narrows rings.py's output to just the rows for ONE requested
    symmetry_type (e.g. "C3"), and to just the four columns this module
    cares about (assembly_id, symmetry_type, chain_groups, mean_distance)
    — std_distance/junctions/axis_count/equivalent_groups all describe the
    DISTANCE-based grouping decision rings.py already made and aren't
    needed again here.

    Raises ValueError for an unrecognized symmetry_type (same C2..C5
    vocabulary rings.py itself enforces via ALLOWED_ORDERS) rather than
    silently returning zero rows, since an empty result here is far more
    likely to be a typo (e.g. "c3" or "3") than a real "no C6 axes"
    answer — rings.py never produces orders outside ALLOWED_ORDERS anyway.
    """
    allowed = {f"C{o}" for o in ALLOWED_ORDERS}
    if symmetry_type not in allowed:
        raise ValueError(f"symmetry_type must be one of {sorted(allowed)} — got {symmetry_type!r}")

    missing = [c for c in _INPUT_COLUMNS if c not in rings_df.columns]
    if missing:
        raise ValueError(f"rings_df is missing expected column(s): {missing}")

    subset = rings_df.loc[rings_df["symmetry_type"] == symmetry_type, list(_INPUT_COLUMNS)]
    return subset.reset_index(drop=True)


def _empty_result(subset: pd.DataFrame) -> pd.DataFrame:
    """Adds the two orientation columns (correctly typed, zero rows) onto
    an already-empty subset — keeps every return path of this module
    correctly-columned even when nothing could be computed."""
    result = subset.copy()
    result["mean_orientation"] = pd.Series(dtype=float)
    result["orientation_junctions"] = pd.Series(dtype=object)
    return result


# ============================================================================
# Per-group orientation
# ============================================================================

def compute_ring_orientation(
    chain_geometry: Dict[str, dict], chain_group: Sequence[str],
) -> Tuple[Optional[float], List[Tuple[str, str, Optional[float]]]]:
    """
    For one chain_groups tuple (already in ring order — rings.py's own
    cycle order for C3/C4/C5, or its sorted pair for C2), walks the k
    consecutive junctions (i -> i+1, wrapping the last chain back to the
    first) and, for each, measures the angle between the FROM chain's
    c_vector (direction it exits AT its C-terminus) and the TO chain's
    n_vector (direction it would continue PAST its N-terminus).

    Returns (mean_orientation, junctions):
      - mean_orientation : mean angle in degrees across junctions with a
        defined angle, or None if none of them were defined (e.g. every
        chain in the group happens to have only one resolved CA).
      - junctions : one (from_chain, to_chain, angle_degrees) triple per
        walked step, angle_degrees rounded to 2 decimal places or None
        where it couldn't be computed — kept in the list (rather than
        dropped) so the junction count always matches len(chain_group),
        the same convention rings.py uses for its own junctions column.

    Raises KeyError (naming the offending chain) if a chain in
    chain_group isn't present in chain_geometry at all — that means the
    structure passed in here doesn't match the one rings.py originally
    detected this grouping from.
    """
    k = len(chain_group)
    junctions: List[Tuple[str, str, Optional[float]]] = []
    for i in range(k):
        from_chain = chain_group[i]
        to_chain = chain_group[(i + 1) % k]
        for name in (from_chain, to_chain):
            if name not in chain_geometry:
                raise KeyError(
                    f"chain {name!r} (from chain_group {tuple(chain_group)!r}) not found in "
                    f"chain_geometry — is this the same structure rings.py detected this grouping from?"
                )
        angle = _angle_degrees(chain_geometry[from_chain]["c_vector"], chain_geometry[to_chain]["n_vector"])
        junctions.append((from_chain, to_chain, round(angle, 2) if angle is not None else None))

    defined = [a for _, _, a in junctions if a is not None]
    mean_orientation = round(float(np.mean(defined)), 2) if defined else None
    return mean_orientation, junctions


# ============================================================================
# Per-assembly / per-batch orchestration
# ============================================================================

def compute_assembly_orientations(
    filepath: str, rings_df: pd.DataFrame, symmetry_type: str,
    assembly_id: Optional[str] = None, vector_window: int = DEFAULT_VECTOR_WINDOW,
) -> pd.DataFrame:
    """
    Single-structure pipeline: loads the structure at `filepath`, computes
    per-chain terminus vectors (termini.get_chain_ca_geometry, with the
    same vector_window rings.py's distances were built from — default 4,
    i.e. "the last 4 residues" at each terminus), then, for every row of
    rings_df matching `symmetry_type` (and `assembly_id`, if given), walks
    its chain_groups ring and appends mean_orientation /
    orientation_junctions.

    rings_df is expected to be rings.py's raw detect_symmetry_rings/
    from_structure output (or anything with its columns) for a SINGLE
    assembly — select_symmetry_type() is applied internally, so callers
    don't need to pre-filter columns or rows themselves.

    Returns a DataFrame with columns (assembly_id, symmetry_type,
    chain_groups, mean_distance, mean_orientation, orientation_junctions).
    Empty (but correctly-columned) if no rows of rings_df match
    symmetry_type (and assembly_id, if given).
    """
    subset = select_symmetry_type(rings_df, symmetry_type)
    if assembly_id is not None and "assembly_id" in subset.columns:
        subset = subset[subset["assembly_id"] == assembly_id].reset_index(drop=True)

    if subset.empty:
        return _empty_result(subset)

    structure = gemmi.read_structure(filepath)
    model = structure[0]

    chain_geometry: Dict[str, dict] = {}
    for chain in model:
        geometry = get_chain_ca_geometry(chain, vector_window=vector_window)
        if geometry is not None:
            chain_geometry[chain.name] = geometry

    mean_orientations: List[Optional[float]] = []
    junctions_col: List[List[Tuple[str, str, Optional[float]]]] = []
    for chain_group in subset["chain_groups"]:
        mean_orientation, junctions = compute_ring_orientation(chain_geometry, chain_group)
        mean_orientations.append(mean_orientation)
        junctions_col.append(junctions)

    result = subset.copy()
    result["mean_orientation"] = mean_orientations
    result["orientation_junctions"] = junctions_col
    return result


def from_rings(
    rings_df: pd.DataFrame, structures: Union[pd.DataFrame, Dict[str, str]], symmetry_type: str,
    filepath_column: str = "filepath", assembly_id_column: str = "assembly_id",
    vector_window: int = DEFAULT_VECTOR_WINDOW,
) -> pd.DataFrame:
    """
    Batch entry point, mirroring rings.py's from_structure(): runs
    compute_assembly_orientations() once per assembly appearing in
    rings_df for the chosen symmetry_type, and concatenates the results.

    rings_df    : rings.py's detect_symmetry_rings/from_structure output
        (or a superset of its columns — only assembly_id, symmetry_type,
        chain_groups, mean_distance are used; std_distance, junctions,
        axis_count, equivalent_groups are dropped).
    structures  : where to find each assembly's structure file. Either
        - a DataFrame with an assembly_id_column and a filepath_column
          (e.g. the same download.py-produced df originally passed to
          rings.from_structure), or
        - a plain {assembly_id: filepath} dict.
    symmetry_type : the SINGLE symmetry type to compute orientations for
        (e.g. "C3") — never "all of them"; call this again with a
        different value if more than one type is needed. Rows of
        rings_df for other symmetry types are ignored entirely.

    An assembly present in rings_df (for symmetry_type) but missing from
    `structures`, or whose structure fails to load/analyze, is reported by
    assembly_id and skipped — same fail-soft convention rings.py's own
    from_structure uses for a structure that fails to parse — rather than
    aborting the whole batch.

    Returns a DataFrame with columns (assembly_id, symmetry_type,
    chain_groups, mean_distance, mean_orientation, orientation_junctions),
    one row per assembly that had a detected grouping for symmetry_type.
    Empty (but correctly-columned) if nothing matched symmetry_type at all.
    """
    subset = select_symmetry_type(rings_df, symmetry_type)
    if subset.empty:
        return _empty_result(subset)

    if isinstance(structures, pd.DataFrame):
        filepath_by_assembly = dict(zip(structures[assembly_id_column], structures[filepath_column]))
    else:
        filepath_by_assembly = dict(structures)

    frames = []
    for assembly_id, group_df in subset.groupby("assembly_id", sort=False):
        filepath = filepath_by_assembly.get(assembly_id)
        if filepath is None:
            print(f"Failed: {assembly_id} — no filepath found in `structures`")
            continue
        try:
            frames.append(
                compute_assembly_orientations(
                    filepath, group_df, symmetry_type, assembly_id=assembly_id, vector_window=vector_window,
                )
            )
        except Exception as exc:
            print(f"Failed: {assembly_id} — {exc}")

    if not frames:
        return _empty_result(subset.iloc[0:0])

    result = pd.concat(frames, ignore_index=True)
    return result

import gemmi
import numpy as np
import matplotlib.pyplot as plt


def plot_chain_group_vectors(filepath, chain_group, vector_window=4, vector_length=5.0, ax=None, show=True):
    """
    Self-contained sanity-check plot for one symmetry group's termini.

    filepath    : path to the assembly structure file (the same one
        rings.py/orientation.py were run against for this assembly_id).
    chain_group : the chain names in ring order, e.g.
        orientation_df.loc[i, "chain_groups"].
    vector_window : how many resolved residues in from each terminus to
        build the direction vector from (default 4, matching termini.py's
        default -- i.e. "the last 4 residues" at each terminus).
    vector_length : how long to draw the (unit-length) direction vectors,
        in Angstroms, purely for visibility.
    ax, show    : pass an existing 3D Axes to draw into, and/or set
        show=False to skip the blocking plt.show() call.

    Recomputes N/C-terminal CA positions and direction vectors straight
    from the structure file itself (the same logic termini.py's
    get_chain_ca_geometry uses) rather than importing anything from this
    project, so the whole thing is one paste-able, dependency-free block.
    """
    structure = gemmi.read_structure(filepath)
    model = structure[0]

    chain_by_name = {chain.name: chain for chain in model}

    chain_geometry = {}
    for chain_name in chain_group:
        if chain_name not in chain_by_name:
            raise ValueError(f"chain {chain_name!r} not found in structure at {filepath!r}")

        polymer = chain_by_name[chain_name].get_polymer()
        ca_coords = []
        for res in polymer:
            ca = res.find_atom("CA", "*")
            if ca:
                ca_coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
        ca_coords = np.array(ca_coords)

        if len(ca_coords) < 2:
            raise ValueError(f"chain {chain_name!r} has fewer than 2 resolved CA atoms -- can't define a direction vector")

        window = min(vector_window, len(ca_coords) - 1)
        n_vector = ca_coords[0] - ca_coords[window]
        n_vector = n_vector / np.linalg.norm(n_vector)
        c_vector = ca_coords[-1] - ca_coords[-1 - window]
        c_vector = c_vector / np.linalg.norm(c_vector)

        chain_geometry[chain_name] = {"n": ca_coords[0], "c": ca_coords[-1], "n_vector": n_vector, "c_vector": c_vector}

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    colors = plt.cm.tab10.colors
    for i, name in enumerate(chain_group):
        geom = chain_geometry[name]
        color = colors[i % len(colors)]
        ax.scatter(*geom["n"], color=color, marker="o", s=60, label=f"{name} (N)")
        ax.scatter(*geom["c"], color=color, marker="s", s=60, label=f"{name} (C)")
        ax.quiver(*geom["n"], *geom["n_vector"], length=vector_length, color=color, linestyle="dashed", arrow_length_ratio=0.3)
        ax.quiver(*geom["c"], *geom["c_vector"], length=vector_length, color=color, arrow_length_ratio=0.3)

    k = len(chain_group)
    for i in range(k):
        from_chain, to_chain = chain_group[i], chain_group[(i + 1) % k]
        cos_angle = np.clip(np.dot(chain_geometry[from_chain]["c_vector"], chain_geometry[to_chain]["n_vector"]), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        midpoint = (chain_geometry[from_chain]["c"] + chain_geometry[to_chain]["n"]) / 2.0
        ax.text(*midpoint, f"{angle:.1f}°", fontsize=9, color="black")

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(f"Termini orientation — {', '.join(chain_group)}")
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    if show:
        plt.show()

    return ax