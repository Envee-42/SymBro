"""
structure.py — computationally light secondary-structure assignment for
N/C termini, from Cα coordinates alone.

Real DSSP classifies secondary structure from hydrogen-bonding patterns
between backbone N and O atoms, which needs a fully-modeled backbone (or
at least N/C/O placed) and an O(n^2) H-bond search over the whole
structure. That's more machinery — and more atoms — than this project
needs just to characterize a chain's two termini. Everything downstream
of termini.py only ever has Cα coordinates in hand (rings.py's step
distances, orientation.py's direction vectors), so this module stays on
that same, single, cheap representation: the resolved Cα trace.

The approach here is the classic CA-only trick (see P-SEA, Labesse et
al. 1997; also how PyMOL's `dss` estimates structure when a backbone is
incomplete): a virtual bond angle and a virtual dihedral, defined purely
from consecutive Cα positions, already carry most of the signal DSSP's
H-bond pattern is a proxy for — an alpha helix's Cα trace curls with a
tight, consistent ~91 degrees/~50 degrees bond-angle/dihedral signature
step after step; a beta strand's Cα trace zig-zags nearly flat, wide
angle, dihedral near +/-180 degrees; anything else is a turn, a bend, or
plain coil. No hydrogen bonds, no side chains, no O(n^2) search — just
vector subtraction, a dot product, and a cross product per residue.

This is an APPROXIMATION, not a DSSP reimplementation: it will disagree
with real DSSP at boundaries and on the harder-to-place states (3-10 vs
alpha, bridge vs strand). It is tuned for what this project actually
needs — a fast, dependency-free read on what's happening structurally at
a chain's very ends, immediately next to n_vector/c_vector, not a
publication-grade full-chain assignment.

Q8 vocabulary used throughout (the standard 8-state DSSP alphabet, with
"C" standing in for DSSP's blank/coil code so an empty string never ends
up sitting in a DataFrame cell):

    H — alpha helix          G — 3-10 helix         I — pi helix
    E — extended strand      B — isolated beta bridge
    T — turn                 S — bend                C — coil / loop

Design, in short:

- Reuses termini.get_chain_ca_geometry directly for the resolved Cα
  trace — no new structure-scanning code, same one-scan-per-chain
  discipline the rest of this project follows.

- A terminus is classified from a WINDOW of residues at that end (first
  `window` Cα's for N, last `window` Cα's for C — always read in
  sequence order either way), not a single residue in isolation: one Cα
  alone has no angle or dihedral to measure. `window` defaults to 4,
  matching termini.get_chain_ca_geometry's own vector_window default, so
  a chain's reported terminus geometry (n_vector/c_vector) and its
  reported terminus secondary structure are always drawn from the same
  stretch of residues.

- Two entry-point grains are provided, mirroring rings.py/orientation.py:
  a per-CHAIN grain (compute_structure_termini_ss / from_structure) for
  when you just want every chain's termini classified, and a per-RING
  grain (compute_assembly_termini_ss / from_rings) that slots directly
  onto rings.py's/orientation.py's (assembly_id, symmetry_type,
  chain_groups, ...) output, appending one new "termini_ss" column —
  the literal "df column ... appendable to another dataframe" this
  module exists to produce.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gemmi
import numpy as np
import pandas as pd

from toolkit.geometry.termini import get_chain_ca_geometry
from toolkit.geometry.rings import ALLOWED_ORDERS


DEFAULT_WINDOW: int = 4  # matches termini.get_chain_ca_geometry's vector_window default
Q8_STATES: Tuple[str, ...] = ("H", "G", "I", "E", "B", "T", "S", "C")

_RING_INPUT_COLUMNS: Tuple[str, ...] = ("assembly_id", "symmetry_type", "chain_groups")
_RING_OUTPUT_COLUMNS: Tuple[str, ...] = _RING_INPUT_COLUMNS + ("termini_ss",)
_CHAIN_OUTPUT_COLUMNS: Tuple[str, ...] = ("assembly_id", "chain", "n_terminus_ss", "c_terminus_ss")


# ============================================================================
# Vector-geometry primitives
# ============================================================================

def _vector_angle(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    """Angle in degrees (0-180) between two vectors, via arccos of their
    normalized dot product. None for a zero-length vector — direction is
    undefined for two coincident Cα coordinates, same degenerate case
    termini.py's _unit_vector guards against."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _bond_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Optional[float]:
    """Virtual bond angle (degrees) at p1, formed by Cα's p0-p1-p2."""
    return _vector_angle(p0 - p1, p2 - p1)


def _dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
    """
    Virtual dihedral angle (degrees, -180 to 180) of four consecutive Cα's
    p0-p1-p2-p3, computed the standard way: project the outer bonds (b0,
    b2) perpendicular to the central bond (b1), then take the signed angle
    between those projections via atan2(cross . unit_b1, dot).

    Returns None if the central bond (p1->p2) is degenerate (zero length)
    — there's no axis to measure a dihedral around.
    """
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    norm_b1 = np.linalg.norm(b1)
    if norm_b1 == 0:
        return None
    b1_unit = b1 / norm_b1

    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    if np.linalg.norm(v) == 0 or np.linalg.norm(w) == 0:
        return None

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_unit, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _circular_mean_and_consistency(angles_deg: Sequence[float]) -> Tuple[Optional[float], float]:
    """
    Circular mean and mean resultant length (R, in [0, 1]) of a set of
    angles — the correct way to average a dihedral, since +179 and -179
    degrees are one degree apart, not 358. R doubles as a consistency
    score: 1.0 means every angle in the window agreed exactly (a clean,
    periodic helix or strand signature), 0.0 means they point every
    which way (no structural periodicity at all).
    """
    if not angles_deg:
        return None, 0.0
    radians = np.radians(angles_deg)
    sin_mean, cos_mean = float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))
    consistency = float(np.hypot(sin_mean, cos_mean))
    mean_angle = float(np.degrees(np.arctan2(sin_mean, cos_mean)))
    return mean_angle, consistency


# ============================================================================
# Per-window classification
# ============================================================================

def _q8_from_geometry(mean_angle: float, mean_dihedral: float, consistency: float, net_curvature: Optional[float]) -> str:
    """
    Maps one terminal window's summary geometry to a single Q8 code.

    Thresholds loosely follow the virtual Cα bond-angle/dihedral windows
    P-SEA (Labesse et al., 1997) uses to read secondary structure off a
    CA-only trace: canonical alpha helix sits at roughly a 91 degree bond
    angle / 50 degree dihedral, canonical beta strand at roughly a 124
    degree bond angle / 180 degree dihedral. Both are widened here into
    Q8-relevant sub-bands (tighter dihedral -> 3-10, looser -> pi; strong
    vs. weak periodicity -> strand vs. isolated bridge) rather than only
    ever returning the two canonical states.

    mean_angle    : circular-independent mean virtual bond angle, degrees.
    mean_dihedral : circular mean virtual dihedral, degrees (-180, 180].
    consistency   : mean resultant length of the window's dihedrals,
        [0, 1] — how periodic/repeating the local geometry is.
    net_curvature : angle (degrees) between the window's entry and exit
        directions — how much the chain's overall heading changes across
        the window, independent of dihedral periodicity. None if
        undefined (degenerate endpoints), treated as "no net turn".
    """
    abs_dihedral = abs(mean_dihedral)

    # Helical family: a right-handed turn each step, tight and consistent.
    if consistency >= 0.6 and 80.0 <= mean_angle <= 112.0 and 15.0 <= mean_dihedral <= 90.0:
        if mean_dihedral <= 35.0:
            return "G"  # tighter per-residue turn -> 3-10 helix
        if mean_dihedral <= 70.0:
            return "H"  # canonical alpha helix
        return "I"      # looser per-residue turn -> pi helix

    # Extended/strand family: near-planar zig-zag, wide bond angle,
    # dihedral near +/-180 degrees.
    if abs_dihedral >= 140.0 and mean_angle >= 100.0:
        return "E" if consistency >= 0.5 else "B"

    # Turn: the window bends sharply overall without a periodic
    # helical/strand signature.
    if net_curvature is not None and net_curvature >= 70.0:
        return "T"

    # Bend: locally irregular (no consistent handedness) AND actually
    # curving by some minimum amount -- a window with low consistency but
    # essentially zero net curvature isn't bending, it's just straight
    # with no periodic signal (falls through to coil below instead).
    if consistency < 0.4 and net_curvature is not None and net_curvature >= 20.0:
        return "S"

    return "C"


def _classify_window(segment: np.ndarray) -> Optional[str]:
    """
    Classifies one contiguous stretch of Cα coordinates (already sliced
    to the terminus of interest, in sequence order) into a single Q8
    code. Needs at least 4 points — the minimum to define one virtual
    dihedral (4 atoms) alongside two virtual bond angles (3 atoms each).
    Returns None if the segment is too short, or if every bond angle in
    it turns out to be degenerate (coincident Cα coordinates) — a
    perfectly straight/collinear window (no dihedral defined, but bond
    angles still are) still gets classified; see the fallback below.
    """
    k = len(segment)
    if k < 4:
        return None

    dihedrals = [d for i in range(k - 3) if (d := _dihedral_angle(segment[i], segment[i + 1], segment[i + 2], segment[i + 3])) is not None]
    angles = [a for i in range(k - 2) if (a := _bond_angle(segment[i], segment[i + 1], segment[i + 2])) is not None]
    if not angles:
        return None

    if dihedrals:
        mean_dihedral, consistency = _circular_mean_and_consistency(dihedrals)
    else:
        # Every dihedral in the window was undefined because its central
        # bond had no measurable perpendicular component -- i.e. the
        # window is (numerically) perfectly straight/collinear. There's
        # no rotation to report, so treat it as "no torsion signal" (0
        # degrees, 0 consistency) rather than bailing out entirely: bond
        # angle and net curvature alone are still enough to tell a
        # straight extended run from a straight-line turn.
        mean_dihedral, consistency = 0.0, 0.0
    mean_angle = float(np.mean(angles))
    net_curvature = _vector_angle(segment[1] - segment[0], segment[-1] - segment[-2])

    return _q8_from_geometry(mean_angle, mean_dihedral, consistency, net_curvature)


def classify_terminus(ca_coords: np.ndarray, terminus: str, window: int = DEFAULT_WINDOW) -> Optional[str]:
    """
    Q8 secondary-structure code for one end of a chain, from its resolved
    Cα trace alone.

    ca_coords : (n_points, 3) array of resolved Cα coordinates in
        sequence order — exactly termini.get_chain_ca_geometry's
        "ca_coords".
    terminus  : "N" or "C". The window examined is the first `window`
        residues (N) or last `window` residues (C) — always taken IN
        SEQUENCE ORDER regardless of which terminus, mirroring how
        termini.py's own n_vector/c_vector windows are defined, so a
        chain's reported terminus geometry and its reported terminus
        secondary structure are always drawn from the same residues.
    window    : how many resolved residues to examine at that terminus
        (default 4, matching termini.get_chain_ca_geometry's
        vector_window default). At least 4 points are needed for even
        one virtual dihedral, so the segment actually used is
        min(max(window, 4), n_points) — the largest available window
        rather than a hard failure on a short chain, same convention
        termini.py's own vector_window follows.

    Returns one of the eight Q8 codes (see module docstring), or None if
    fewer than 4 Cα's are resolved at all — secondary structure has no
    defined meaning from 3 points or fewer.
    """
    terminus = terminus.upper()
    if terminus not in ("N", "C"):
        raise ValueError(f"terminus must be 'N' or 'C' — got {terminus!r}")

    ca_coords = np.asarray(ca_coords, dtype=float)
    n_points = len(ca_coords)
    if n_points < 4:
        return None

    size = min(max(window, 4), n_points)
    segment = ca_coords[:size] if terminus == "N" else ca_coords[-size:]
    return _classify_window(segment)


# ============================================================================
# Per-chain convenience
# ============================================================================

def get_chain_termini_ss(chain: "gemmi.Chain", window: int = DEFAULT_WINDOW) -> Dict[str, Optional[str]]:
    """
    Runs termini.get_chain_ca_geometry once for `chain` and classifies
    both of its termini from the resulting Cα trace.

    Returns {"n_terminus_ss": <Q8 code or None>, "c_terminus_ss": <Q8
    code or None>} — both None if the chain has no resolved polymer CA's
    at all (get_chain_ca_geometry returns None), matching that
    function's own "nothing resolved" convention.
    """
    geometry = get_chain_ca_geometry(chain, vector_window=window)
    if geometry is None:
        return {"n_terminus_ss": None, "c_terminus_ss": None}

    ca_coords = geometry["ca_coords"]
    return {
        "n_terminus_ss": classify_terminus(ca_coords, "N", window=window),
        "c_terminus_ss": classify_terminus(ca_coords, "C", window=window),
    }


# ============================================================================
# Per-structure orchestration — CHAIN grain
# ============================================================================

def compute_structure_termini_ss(
    filepath: str, assembly_id: Optional[str] = None,
    chain_names: Optional[Sequence[str]] = None, window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Single-structure pipeline, chain grain: loads the structure at
    `filepath` and classifies every chain's N- and C-terminus (or just
    the chains named in `chain_names`, if given).

    Returns a DataFrame with columns (assembly_id, chain, n_terminus_ss,
    c_terminus_ss) — one row per chain that has a resolved polymer with
    at least one Cα atom (chains with no resolved CA at all, e.g.
    ligand/solvent-only chains, are skipped rather than emitted as an
    all-None row). Empty (but correctly-columned) if no chain qualifies.
    """
    structure = gemmi.read_structure(filepath)
    structure.setup_entities()  # required for get_polymer() on a header-less, ATOM-only file
    model = structure[0]

    rows = []
    for chain in model:
        if chain_names is not None and chain.name not in chain_names:
            continue
        geometry = get_chain_ca_geometry(chain, vector_window=window)
        if geometry is None:
            continue
        ca_coords = geometry["ca_coords"]
        rows.append({
            "assembly_id": assembly_id,
            "chain": chain.name,
            "n_terminus_ss": classify_terminus(ca_coords, "N", window=window),
            "c_terminus_ss": classify_terminus(ca_coords, "C", window=window),
        })

    return pd.DataFrame(rows, columns=list(_CHAIN_OUTPUT_COLUMNS))


def from_structure(
    df: pd.DataFrame, filepath_column: str = "filepath", assembly_id_column: str = "assembly_id", **kwargs: Any,
) -> pd.DataFrame:
    """
    Runs compute_structure_termini_ss over every row of a
    downloaded-candidates DataFrame (download.py's download_candidates()
    output) and concatenates the results — one row per chain per
    assembly. Mirrors rings.py's from_structure exactly, including its
    fail-soft convention: a structure that fails to parse/analyze is
    reported by assembly_id and skipped rather than aborting the batch.

    assembly_id_column : used directly if present in df; otherwise falls
        back to f"{entry_id}-{assembly_num}" (download.py's convention).
    """
    frames = []
    for _, row in df.iterrows():
        assembly_id = row[assembly_id_column] if assembly_id_column in df.columns else f"{row['entry_id']}-{row['assembly_num']}"
        try:
            frames.append(compute_structure_termini_ss(row[filepath_column], assembly_id, **kwargs))
        except Exception as exc:
            print(f"Failed: {assembly_id} — {exc}")

    if not frames:
        return pd.DataFrame(columns=list(_CHAIN_OUTPUT_COLUMNS))

    result = pd.concat(frames, ignore_index=True)
    if not result.empty:
        result = result.sort_values(["assembly_id", "chain"]).reset_index(drop=True)
    return result


# ============================================================================
# Per-structure orchestration — RING grain (slots onto rings.py/orientation.py)
# ============================================================================

def _select_ring_rows(df: pd.DataFrame, symmetry_type: str, assembly_id: Optional[str] = None) -> pd.DataFrame:
    """
    Narrows a rings.py/orientation.py-shaped DataFrame to one
    symmetry_type (and, if given, one assembly_id), keeping just the
    columns this module needs (assembly_id, symmetry_type, chain_groups)
    — same "one symmetry type at a time, never all of them" contract
    orientation.py's select_symmetry_type uses, and the same reasoning:
    a caller analyzing one axis order usually isn't asking about every
    other axis order the same assembly may also have.
    """
    allowed = {f"C{o}" for o in ALLOWED_ORDERS}
    if symmetry_type not in allowed:
        raise ValueError(f"symmetry_type must be one of {sorted(allowed)} — got {symmetry_type!r}")

    missing = [c for c in _RING_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing expected column(s): {missing}")

    subset = df.loc[df["symmetry_type"] == symmetry_type, list(_RING_INPUT_COLUMNS)]
    if assembly_id is not None and "assembly_id" in subset.columns:
        subset = subset[subset["assembly_id"] == assembly_id]
    return subset.reset_index(drop=True)


def _empty_ring_result(subset: pd.DataFrame) -> pd.DataFrame:
    result = subset.copy()
    result["termini_ss"] = pd.Series(dtype=object)
    return result


def compute_ring_termini_ss(
    chain_geometry: Dict[str, dict], chain_group: Sequence[str], window: int = DEFAULT_WINDOW,
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    For one chain_groups tuple (rings.py's ring order, or C2's sorted
    pair), classifies every member chain's N- and C-terminus from
    already-computed chain_geometry (a {chain_name: get_chain_ca_geometry
    result} dict — same shape rings.py/orientation.py build internally).

    Returns one (chain_name, n_terminus_ss, c_terminus_ss) triple per
    chain in the group, in group order — the same "one row, one list
    column of per-junction/per-chain triples" convention rings.py's
    junctions and orientation.py's orientation_junctions already use.

    Raises KeyError (naming the offending chain) if a chain in
    chain_group isn't present in chain_geometry — same contract
    orientation.py's compute_ring_orientation uses, meaning the
    structure passed in doesn't match the one this grouping came from.
    """
    triples = []
    for chain_name in chain_group:
        if chain_name not in chain_geometry:
            raise KeyError(
                f"chain {chain_name!r} (from chain_group {tuple(chain_group)!r}) not found in "
                f"chain_geometry — is this the same structure this grouping was detected from?"
            )
        ca_coords = chain_geometry[chain_name]["ca_coords"]
        n_ss = classify_terminus(ca_coords, "N", window=window)
        c_ss = classify_terminus(ca_coords, "C", window=window)
        triples.append((chain_name, n_ss, c_ss))
    return triples


def compute_assembly_termini_ss(
    filepath: str, rings_df: pd.DataFrame, symmetry_type: str,
    assembly_id: Optional[str] = None, window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Single-structure pipeline, ring grain: loads the structure at
    `filepath`, computes every chain's Cα trace once
    (termini.get_chain_ca_geometry), then, for every row of rings_df
    matching `symmetry_type` (and `assembly_id`, if given), classifies
    every chain in its chain_groups ring and appends a "termini_ss"
    column. Mirrors orientation.py's compute_assembly_orientations.

    rings_df is expected to be rings.py's detect_symmetry_rings/
    from_structure output (or orientation.py's output — a superset of
    the needed columns) for a SINGLE assembly.

    Returns a DataFrame with columns (assembly_id, symmetry_type,
    chain_groups, termini_ss) — termini_ss holds a list of (chain,
    n_terminus_ss, c_terminus_ss) triples per row. Empty (but
    correctly-columned) if no rows of rings_df match symmetry_type (and
    assembly_id, if given).
    """
    subset = _select_ring_rows(rings_df, symmetry_type, assembly_id)
    if subset.empty:
        return _empty_ring_result(subset)

    structure = gemmi.read_structure(filepath)
    structure.setup_entities()  # required for get_polymer() on a header-less, ATOM-only file
    model = structure[0]

    chain_geometry: Dict[str, dict] = {}
    for chain in model:
        geometry = get_chain_ca_geometry(chain, vector_window=window)
        if geometry is not None:
            chain_geometry[chain.name] = geometry

    termini_ss_col = [compute_ring_termini_ss(chain_geometry, chain_group, window=window) for chain_group in subset["chain_groups"]]

    result = subset.copy()
    result["termini_ss"] = termini_ss_col
    return result


def from_rings(
    rings_df: pd.DataFrame, structures: Union[pd.DataFrame, Dict[str, str]], symmetry_type: str,
    filepath_column: str = "filepath", assembly_id_column: str = "assembly_id", window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Batch entry point, ring grain — mirrors orientation.py's from_rings
    exactly: runs compute_assembly_termini_ss() once per assembly
    appearing in rings_df for the chosen symmetry_type, and concatenates
    the results. THIS is the function most callers want for "add a
    termini secondary-structure column to my existing rings/orientation
    DataFrame":

        rings_df["termini_ss"] = structure.from_rings(
            rings_df, downloaded_df, "C3"
        )["termini_ss"]

    rings_df    : rings.py's/orientation.py's output (or a superset of
        its columns — only assembly_id, symmetry_type, chain_groups are
        used).
    structures  : where to find each assembly's structure file — either
        a DataFrame with an assembly_id_column and a filepath_column
        (e.g. download.py's output), or a plain {assembly_id: filepath}
        dict.
    symmetry_type : the SINGLE symmetry type to compute termini SS for
        (e.g. "C3") — call again with a different value for other types.

    An assembly present in rings_df (for symmetry_type) but missing from
    `structures`, or whose structure fails to load/analyze, is reported
    by assembly_id and skipped rather than aborting the whole batch —
    same fail-soft convention rings.py/orientation.py use.

    Returns a DataFrame with columns (assembly_id, symmetry_type,
    chain_groups, termini_ss), one row per assembly that had a detected
    grouping for symmetry_type. Empty (but correctly-columned) if nothing
    matched symmetry_type at all.
    """
    subset = _select_ring_rows(rings_df, symmetry_type)
    if subset.empty:
        return _empty_ring_result(subset)

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
                compute_assembly_termini_ss(filepath, group_df, symmetry_type, assembly_id=assembly_id, window=window)
            )
        except Exception as exc:
            print(f"Failed: {assembly_id} — {exc}")

    if not frames:
        return _empty_ring_result(subset.iloc[0:0])

    return pd.concat(frames, ignore_index=True)