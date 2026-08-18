"""
rings.py — cyclic symmetry axis detection for self-assembling protein cages.

Given a downloaded assembly (see download.py) built from candidates found
via query.py, this module answers: what rotational symmetry axes (C2, C3,
C4, C5 — the only orders Platonic/dihedral cages have) does this assembly
actually permit, which chains form each axis, and how far apart are their
termini?

Design, in short (see AnAnaS/ProSHADE/PDBePISA for the general idea this
takes inspiration from — inferring symmetry from the arrangement of
repeated subunits rather than fitting a global rotation matrix):

- Built entirely on termini.get_chain_ca_geometry's per-chain N/C-terminal
  Cα coordinates. No whole-assembly centroid or fixed rotation is ever
  computed — cage subunits pack with real local flexibility, so a rigid
  global-alignment approach would be the wrong tool.
- Only chains that are actual sequence copies of each other (>=
  identity_threshold) are ever compared for a shared axis — two chains
  from different proteins can have coincidentally close termini at an
  interface without being related by symmetry.
- A candidate grouping is judged by STEP HOMOGENEITY, not absolute
  distance: a real ring repeats the same physical junction n times, so its
  n termini-to-termini step distances should agree with each other
  (population std <= tolerance, OR coefficient of variation <=
  relative_tolerance) regardless of whether that shared distance is 8
  Angstroms or 60.
- A CONTACT CHECK then confirms the chains in a homogeneity-passing
  candidate actually touch somewhere along their backbone (min pairwise
  Cα-Cα distance <= contact_cutoff) — homogeneous termini distances alone
  can't rule out two unrelated, non-adjacent copies that merely sit at a
  similar distance apart.
- Every order this module detects — C2 included — is a head-to-tail (N-C)
  register: a rotational symmetry axis is, by definition, a repeating
  N-to-C arrangement of identical subunits around the axis. find_c2_groupings
  and find_cyclic_groupings both only ever build/check N-C edges; neither
  considers an N-N or C-C contact a candidate rotational axis, because
  that isn't the geometry a rotation axis produces (see find_c2_groupings'
  docstring for the history here — N-N/C-C used to be accepted too, which
  was a bug, not a design choice).
- Within one order, a chain may only belong to ONE accepted grouping
  (resolved by greedily accepting candidates tightest-CV-first and
  discarding anything that reuses an already-claimed chain). ACROSS
  orders there's no such restriction — a chain sitting on a C2 axis and a
  C3 axis simultaneously is exactly how a T/O/I cage is built.
- Output has one row per detected axis order PER STRUCTURALLY DISTINCT
  COMPONENT per assembly — never one row per individual grouping, but
  also never one row that silently pools multiple different proteins
  together. "Component" here means one of group_chains_by_identity's
  sequence-identity clusters: exclusivity resolution (which candidate
  becomes "chain_groups" vs. gets folded into "equivalent_groups") is
  now scoped to ONE identity cluster at a time, so a two-component cage
  (e.g. a T:3,3 architecture, two different C3-symmetric proteins) gets
  TWO rows for symmetry_type="C3" — one per protein — each carrying its
  own "component_id" (the identity cluster's index), its own tightest
  "chain_groups", and its own "equivalent_groups" containing ONLY that
  same component's redundant repeats (never another component's rings).
  This matters because every downstream stage that consumes this
  DataFrame (isolate.py, and via it rfdiffusion.py/pmpnn.py) iterates
  every row rather than assuming one row per (assembly_id,
  symmetry_type) — so a multi-component assembly is now carried all the
  way through the pipeline automatically, rather than only its
  geometrically "tightest" component surviving past detection. This is a
  behavior-preserving refactor at the level of WHICH groupings are
  accepted (candidates from different identity clusters never share
  chains, so scoping exclusivity resolution per cluster instead of
  pooling everything first accepts exactly the same set of groupings —
  only how they're bucketed into rows changes).
- Each row also carries "recommended_linker_length" — an (min_residues,
  max_residues) estimate, derived directly from that grouping's own
  measured mean_distance via estimate_linker_length() below, for how
  long a diffused linker spanning that junction should be. This feeds
  straight into RFdiffusion's contig "diffuse" segment length
  (rfdiffusion.py's build_linker_fusion_contig) so the diffusion length
  range is set from this ring's own geometry rather than a fixed guess.
"""

import math
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gemmi
import networkx as nx
import numpy as np
import pandas as pd

from toolkit.geometry.termini import get_chain_ca_geometry


ALLOWED_ORDERS: Tuple[int, ...] = (2, 3, 4, 5)
CYCLIC_ORDERS: Tuple[int, ...] = (3, 4, 5)  # C2 is undirected pairwise, handled separately

DEFAULT_TOLERANCE: float = 2.0
DEFAULT_RELATIVE_TOLERANCE: float = 0.05
DEFAULT_IDENTITY_THRESHOLD: float = 0.9
DEFAULT_CONTACT_CUTOFF: Optional[float] = 8.0
# Search-space prefilter only (keeps cycle enumeration tractable on large
# assemblies) — never the actual validity test, which is step homogeneity.
DEFAULT_MAX_CANDIDATE_DISTANCE: Optional[float] = 100.0

# estimate_linker_length()'s defaults — see that function's docstring for
# the reasoning behind each value.
DEFAULT_LINKER_MAX_REACH: float = 3.8   # Å/residue, fully-extended backbone upper bound
DEFAULT_LINKER_MIN_REACH: float = 2.0   # Å/residue, conservative reach for a relaxed/coiled linker
DEFAULT_LINKER_BUFFER: int = 2          # extra residues added at both ends for design/attachment slack

_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "assembly_id", "symmetry_type", "component_id", "chain_groups",
    "mean_distance", "std_distance", "recommended_linker_length", "junctions",
    "axis_count", "equivalent_groups", "component_chain_count",
)


# ============================================================================
# Sequence-identity clustering — only true copies are compared
# ============================================================================

def group_chains_by_identity(
    model: gemmi.Model, identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD
) -> List[List[str]]:
    """
    Clusters chain names by approximate sequence identity (difflib ratio,
    which tracks true percent identity closely for near-identical
    homo-oligomer copies without needing a real alignment). Greedy
    clustering against each cluster's first member is enough here — copies
    of one cage subunit resemble each other closely, not just a distant
    common ancestor, so representative-based clustering agrees with full
    pairwise linkage in practice.

    Chains without a resolved polymer (ligands, solvent) are dropped.
    """
    sequences: Dict[str, str] = {}
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) > 0:
            sequences[chain.name] = gemmi.one_letter_code(polymer.extract_sequence())

    representatives: List[str] = []
    clusters: List[List[str]] = []
    for name, seq in sequences.items():
        for idx, rep in enumerate(representatives):
            if seq and sequences[rep] and SequenceMatcher(None, seq, sequences[rep]).ratio() >= identity_threshold:
                clusters[idx].append(name)
                break
        else:
            representatives.append(name)
            clusters.append([name])

    return clusters


# ============================================================================
# Shared geometry / statistics helpers
# ============================================================================

def _terminal_coords(chain_geometry: Dict[str, dict], chain_names: Sequence[str], terminus: str) -> np.ndarray:
    key = "n" if terminus.upper() == "N" else "c"
    return np.array([chain_geometry[name][key] for name in chain_names])


def _step_stats(distances: Sequence[float]) -> Tuple[float, float, float]:
    """Mean, population std (this candidate's steps ARE the whole
    population, not a sample), and coefficient of variation. CV is 0.0 for
    a perfectly uniform zero-mean case, infinite otherwise (an unbounded
    relative spread that can never pass relative_tolerance)."""
    arr = np.asarray(distances, dtype=float)
    mean, std = float(arr.mean()), float(arr.std(ddof=0))
    cv = 0.0 if mean == 0.0 and std == 0.0 else (float("inf") if mean == 0.0 else std / mean)
    return mean, std, cv


def _passes_tolerance(std: float, cv: float, tolerance: Optional[float], relative_tolerance: Optional[float]) -> bool:
    """Either the absolute or the relative spread test is sufficient — an
    OR, so a long junction (large mean, small CV) and a short tight one
    (small mean, small std) are each judged on the yardstick that fits its
    own scale rather than one shared absolute cutoff."""
    return (tolerance is not None and std <= tolerance) or (relative_tolerance is not None and cv <= relative_tolerance)


def estimate_linker_length(
    distance: float,
    max_reach: float = DEFAULT_LINKER_MAX_REACH,
    min_reach: float = DEFAULT_LINKER_MIN_REACH,
    buffer: int = DEFAULT_LINKER_BUFFER,
) -> Tuple[int, int]:
    """
    Recommended (min_residues, max_residues) for a diffused linker meant
    to span a measured junction distance (Angstrom) — feeds directly into
    RFdiffusion's contig "diffuse" segment length (see rfdiffusion.py's
    build_linker_fusion_contig / build_contig_string), so a ring's own
    measured geometry sets the diffusion length range instead of a fixed
    guess.

    min_residues = ceil(distance / max_reach) + buffer: the fewest
        residues that could possibly span `distance` even held fully
        extended. max_reach defaults to 3.8 Å/residue, the standard
        fully-extended (trans) backbone Cα-Cα spacing upper bound — fewer
        residues than this literally cannot reach across the gap.
    max_residues = ceil(distance / min_reach) + buffer: an upper bound
        assuming a more relaxed, non-extended (coiled/flexible) linker
        conformation. min_reach defaults to 2.0 Å/residue, a conservative
        net per-residue reach for a loop that isn't held taut — this
        gives RFdiffusion room to sample a natural-looking backbone
        rather than being forced maximally stretched.
    buffer defaults to +2 residues at both ends, for ordinary design/
    attachment slack — the same kind of small, literature-adjacent-but-
    tunable default this project already uses elsewhere (see
    selfconsistency.py's max_rmsd=2.0/min_plddt=70.0 for the analogous
    pattern: a reasonable starting point, not an absolute).

    This is a triage HEURISTIC for picking a plausible diffusion length
    range, not a guarantee RFdiffusion will actually close the gap at
    either bound — always sanity-check resulting designs (e.g. against
    selfconsistency.py's own thresholds) rather than treating this
    interval as authoritative.

    Returns (min_residues, max_residues), both >= 1 + buffer. Raises
    ValueError for a non-positive distance/max_reach/min_reach.
    """
    if distance <= 0:
        raise ValueError(f"distance must be positive — got {distance}")
    if max_reach <= 0 or min_reach <= 0:
        raise ValueError("max_reach and min_reach must both be positive")

    min_residues = max(1, math.ceil(distance / max_reach)) + buffer
    max_residues = max(min_residues, math.ceil(distance / min_reach) + buffer)
    return min_residues, max_residues


def _canonical_cycle(cycle: Tuple[str, ...]) -> Tuple[str, ...]:
    """Rotates a cyclic chain sequence to start at its lexicographically
    smallest chain — ('A','B','C') and ('B','C','A') are the same physical
    ring and must compare equal before exclusivity resolution runs."""
    min_idx = min(range(len(cycle)), key=lambda i: cycle[i])
    return cycle[min_idx:] + cycle[:min_idx]


def _has_contact(
    chain_geometry: Dict[str, dict], chain_a: str, chain_b: str,
    cutoff: Optional[float], cache: Optional[Dict[frozenset, bool]] = None,
) -> bool:
    """
    Confirms chain_a and chain_b actually touch somewhere along their
    backbone (min pairwise Cα-Cα distance <= cutoff), not just that their
    termini happen to be a homogeneous distance apart. Two unrelated,
    non-adjacent copies of the same subunit can coincidentally satisfy the
    step-homogeneity test without ever being in contact — this catches
    that case. cutoff=None skips the check entirely.

    Results are memoized per unordered chain pair (`cache`) since the same
    edge is re-examined across many candidate rings within one order.
    """
    if cutoff is None:
        return True
    key = frozenset((chain_a, chain_b))
    if cache is not None and key in cache:
        return cache[key]

    a = chain_geometry[chain_a]["ca_coords"]
    b = chain_geometry[chain_b]["ca_coords"]
    min_dist = float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)))
    result = min_dist <= cutoff

    if cache is not None:
        cache[key] = result
    return result


# ============================================================================
# Candidate detection
# ============================================================================

def find_c2_groupings(
    chain_names: Sequence[str], chain_geometry: Dict[str, dict],
    tolerance: Optional[float] = DEFAULT_TOLERANCE, relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    contact_cutoff: Optional[float] = DEFAULT_CONTACT_CUTOFF, max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
    contact_cache: Optional[Dict[frozenset, bool]] = None,
) -> List[Dict[str, Any]]:
    """
    A two-fold axis has no larger loop to walk (exactly two chains), but a
    genuine ROTATIONAL two-fold axis is, like every other order this
    module detects, a head-to-tail (N-C) register: find_cyclic_groupings
    never considers a C-to-C or N-to-N edge for C3/C4/C5 either, because
    that isn't the geometry a rotation axis produces. So the only
    candidate checked here is N-C, taken in both directions — nc_fwd
    (chain a's N to chain b's C) and nc_rev (chain b's N to chain a's C),
    the two independent measurements a real two-fold's head-to-tail
    register produces — required to agree with each other via the same
    step-homogeneity test every other order uses.

    N-N and C-C used to also be accepted here as alternate C2 "interface
    types", on the reasoning that a two-fold axis might reasonably be
    head-to-head or tail-to-tail instead. That was a bug, not a design
    choice: an N-N or C-C contact is a single distance measurement, so its
    population std is trivially 0.0 and its CV is trivially 0.0 — which
    means it would ALWAYS out-rank a real two-measurement N-C candidate in
    select_disjoint_groupings' CV-tightest-first sort, even when a
    genuine, tolerance-and-contact-passing N-C axis was also present for
    the same pair. In practice this meant a nearby but non-rotational
    tail-to-tail or head-to-head contact would almost always be reported
    as "the" C2 axis instead of the real one. Restricting this function to
    N-C only removes the false positives and the ranking bias in one move,
    and brings C2 in line with how C3/C4/C5 were already being detected.
    """
    n_coords = _terminal_coords(chain_geometry, chain_names, "N")
    c_coords = _terminal_coords(chain_geometry, chain_names, "C")
    index_of = {name: i for i, name in enumerate(chain_names)}

    results: List[Dict[str, Any]] = []
    for a, b in combinations(chain_names, 2):
        ia, ib = index_of[a], index_of[b]
        pair = tuple(sorted((a, b)))

        nc_fwd = float(np.linalg.norm(n_coords[ia] - c_coords[ib]))
        nc_rev = float(np.linalg.norm(c_coords[ia] - n_coords[ib]))
        mean, std, cv = _step_stats([nc_fwd, nc_rev])

        if max_candidate_distance is not None and mean > max_candidate_distance:
            continue
        if not _passes_tolerance(std, cv, tolerance, relative_tolerance):
            continue
        if not _has_contact(chain_geometry, a, b, contact_cutoff, contact_cache):
            continue

        results.append({
            "chains": pair, "chain_set": frozenset(pair), "interface_type": "N-C",
            "junctions": [(a, b, nc_fwd), (b, a, nc_rev)], "mean_distance": round(mean, 2),
            "std_distance": round(std, 2), "cv": round(cv, 4) if np.isfinite(cv) else None,
        })

    return results


def find_cyclic_groupings(
    chain_names: Sequence[str], chain_geometry: Dict[str, dict], order: int,
    tolerance: Optional[float] = DEFAULT_TOLERANCE, relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    contact_cutoff: Optional[float] = DEFAULT_CONTACT_CUTOFF, max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
    contact_cache: Optional[Dict[frozenset, bool]] = None,
) -> List[Dict[str, Any]]:
    """
    Finds elementary directed cycles of length `order` (3, 4, or 5) in the
    C-terminus -> N-terminus candidate graph — chain_1(C) -> chain_2(N) ->
    chain_2(C) -> ... -> chain_1(N) — keeping only cycles whose `order`
    step distances pass the step-homogeneity test and whose every
    consecutive chain pair is in real contact. Returns every surviving
    candidate (possibly overlapping in chain membership); exclusivity is
    resolved separately by select_disjoint_groupings.
    """
    if order not in CYCLIC_ORDERS:
        raise ValueError(f"order must be one of {CYCLIC_ORDERS} — C2 is handled by find_c2_groupings")
    if len(chain_names) < order:
        return []

    c_coords = _terminal_coords(chain_geometry, chain_names, "C")
    n_coords = _terminal_coords(chain_geometry, chain_names, "N")
    dist_matrix = np.linalg.norm(c_coords[:, None, :] - n_coords[None, :, :], axis=-1)

    graph = nx.DiGraph()
    graph.add_nodes_from(chain_names)
    for i, from_chain in enumerate(chain_names):
        for j, to_chain in enumerate(chain_names):
            if i == j:
                continue
            d = float(dist_matrix[i, j])
            if max_candidate_distance is None or d <= max_candidate_distance:
                graph.add_edge(from_chain, to_chain, weight=d)

    seen: set = set()
    results: List[Dict[str, Any]] = []
    for cycle in nx.simple_cycles(graph, length_bound=order):
        if len(cycle) != order:
            continue
        canonical = _canonical_cycle(tuple(cycle))
        if canonical in seen:
            continue
        seen.add(canonical)

        junctions = [(canonical[k], canonical[(k + 1) % order], graph[canonical[k]][canonical[(k + 1) % order]]["weight"]) for k in range(order)]
        mean, std, cv = _step_stats([j[2] for j in junctions])
        if not _passes_tolerance(std, cv, tolerance, relative_tolerance):
            continue
        if not all(_has_contact(chain_geometry, a, b, contact_cutoff, contact_cache) for a, b, _ in junctions):
            continue

        results.append({
            "chains": canonical, "chain_set": frozenset(canonical), "interface_type": None,
            "junctions": junctions, "mean_distance": round(mean, 2), "std_distance": round(std, 2),
            "cv": round(cv, 4) if np.isfinite(cv) else None,
        })

    return results


def select_disjoint_groupings(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enforces the exclusivity rule for one order: sorts candidates
    tightest-first (ascending CV, then std, then mean, so ordering is
    deterministic; a degenerate cv=None sorts last), then greedily accepts
    each candidate whose chains are still entirely unclaimed. A tight,
    early-accepted grouping gets first claim on its chains, so the result
    is disjoint by construction and always favors the geometrically
    tightest reading of the data.
    """
    ordered = sorted(candidates, key=lambda c: (c["cv"] if c["cv"] is not None else float("inf"), c["std_distance"], c["mean_distance"]))

    claimed: set = set()
    accepted: List[Dict[str, Any]] = []
    for candidate in ordered:
        if claimed.isdisjoint(candidate["chain_set"]):
            accepted.append(candidate)
            claimed.update(candidate["chain_set"])
    return accepted


# ============================================================================
# Per-assembly / per-batch orchestration
# ============================================================================

def detect_symmetry_rings(
    filepath: str, assembly_id: Optional[str] = None, orders: Sequence[int] = ALLOWED_ORDERS,
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD, tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE, contact_cutoff: Optional[float] = DEFAULT_CONTACT_CUTOFF,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> pd.DataFrame:
    """
    Full single-assembly pipeline: load the structure, compute per-chain
    terminal geometry (termini.get_chain_ca_geometry), cluster chains by
    sequence identity, then per symmetry order AND per identity cluster —
    resolving disjoint exclusivity separately within each cluster, never
    pooling different clusters together — keep the tightest grouping plus
    any other accepted ones for that (order, cluster) pair.

    Scoping exclusivity per identity cluster instead of globally doesn't
    change WHICH groupings get accepted (candidates from different
    clusters never share a chain, by construction of
    group_chains_by_identity, so they could never have contended for the
    same chain in a pooled resolution either) — it only changes how
    accepted groupings are bucketed into output rows. The practical
    effect: a multi-component assembly (e.g. a two-protein T:3,3 cage)
    gets one row per component per order, each with its own
    "component_id", rather than only its single geometrically tightest
    component surviving into "chain_groups" while every OTHER component's
    rings get silently folded into "equivalent_groups" alongside genuine
    same-component duplicates (and then dropped — isolate.py's
    isolate_assembly_rings/from_rings iterate every row of this
    DataFrame but never read equivalent_groups).

    Returns a DataFrame with one row per (detected order, identity
    cluster) pair:
      assembly_id, symmetry_type ("C2".."C5"), component_id (the
      identity cluster's index — stable within one call for one
      structure, not meaningful across different assemblies/calls),
      chain_groups (tightest grouping's chain-ID tuple, for this
      component), mean_distance / std_distance (that grouping's termini
      step distances, Angstroms), recommended_linker_length
      ((min_residues, max_residues), from estimate_linker_length() on
      mean_distance), junctions (list of (from_chain, to_chain,
      distance) for that grouping), axis_count (how many disjoint
      groupings of this order were found for THIS component),
      equivalent_groups (every other accepted grouping's chain-ID tuple
      for this SAME component — empty if chain_groups was the only one
      found for it; never another component's rings), and
      component_chain_count (how many chains this identity cluster had
      usable geometry for in THIS structure — i.e. the size `usable`
      below, before any grouping/exclusivity logic runs at all. Lets a
      caller compute the maximum number of disjoint groupings of this
      order this component's own chain count could possibly support
      (component_chain_count // order) and compare it against axis_count
      to flag a component that's short some chains from a full ring
      decomposition — see pipeline.py's _warn_incomplete_axis_counts()).

    Empty (but correctly-columned) if nothing survives detection for any
    requested order.
    """
    invalid = sorted(set(orders) - set(ALLOWED_ORDERS))
    if invalid:
        raise ValueError(f"orders must be a subset of {ALLOWED_ORDERS} — got invalid {invalid}")

    structure = gemmi.read_structure(filepath)
    structure.setup_entities()  # required for get_polymer() on a header-less, ATOM-only file
    model = structure[0]

    chain_geometry: Dict[str, dict] = {}
    for chain in model:
        geometry = get_chain_ca_geometry(chain)
        if geometry is not None:
            chain_geometry[chain.name] = geometry

    identity_groups = group_chains_by_identity(model, identity_threshold)
    contact_cache: Dict[frozenset, bool] = {}

    rows = []
    for order in sorted(set(orders)):
        for component_id, group in enumerate(identity_groups):
            usable = [name for name in group if name in chain_geometry]
            if len(usable) < order:
                continue
            finder = find_c2_groupings if order == 2 else find_cyclic_groupings
            args = (usable, chain_geometry) if order == 2 else (usable, chain_geometry, order)
            candidates = finder(*args, tolerance=tolerance, relative_tolerance=relative_tolerance,
                                 contact_cutoff=contact_cutoff, max_candidate_distance=max_candidate_distance,
                                 contact_cache=contact_cache)

            accepted = select_disjoint_groupings(candidates)
            if not accepted:
                continue

            main = accepted[0]
            rows.append({
                "assembly_id": assembly_id,
                "symmetry_type": f"C{order}",
                "component_id": component_id,
                "chain_groups": main["chains"],
                "mean_distance": main["mean_distance"],
                "std_distance": main["std_distance"],
                "recommended_linker_length": estimate_linker_length(main["mean_distance"]),
                "junctions": main["junctions"],
                "axis_count": len(accepted),
                "equivalent_groups": [c["chains"] for c in accepted[1:]],
                "component_chain_count": len(usable),
            })

    return pd.DataFrame(rows, columns=list(_OUTPUT_COLUMNS))


def from_structure(
    df: pd.DataFrame, filepath_column: str = "filepath", assembly_id_column: str = "assembly_id", **kwargs: Any,
) -> pd.DataFrame:
    """
    Runs detect_symmetry_rings over every row of a downloaded-candidates
    DataFrame (download.py's download_candidates() output) and
    concatenates the results — up to len(orders) rows per assembly.

    assembly_id_column : used directly if present in df; otherwise falls
        back to f"{entry_id}-{assembly_num}" (download.py's convention).

    A structure that fails to parse or analyze is reported by assembly_id
    and skipped rather than aborting the whole batch.
    """
    frames = []
    for _, row in df.iterrows():
        assembly_id = row[assembly_id_column] if assembly_id_column in df.columns else f"{row['entry_id']}-{row['assembly_num']}"
        try:
            frames.append(detect_symmetry_rings(row[filepath_column], assembly_id, **kwargs))
        except Exception as exc:
            print(f"Failed: {assembly_id} — {exc}")

    if not frames:
        return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))

    result = pd.concat(frames, ignore_index=True)
    if not result.empty:
        result = result.sort_values(["assembly_id", "symmetry_type"]).reset_index(drop=True)
    return result