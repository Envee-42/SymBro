"""
symmetry.py — agnostic cyclic-symmetry (C2/C3/C4/C5) grouping detection for
self-assembling protein cages, built on top of termini.py's per-chain
terminal coordinates.

WHY THIS IS A SEPARATE MODULE FROM distance.py
------------------------------------------------------------------------
distance.py's ring detector was built around a single, specific job:
recover the ONE correct fusion order for a designed cyclic subunit (a
trimer, say) so it can be handed to RFdiffusion/ProteinMPNN as one
representative construct. It does that by requiring EXCLUSIVE, non-
overlapping chain assignment per ring size (a chain claimed by one C3
ring can't also be claimed by another C3 candidate), because for that
specific job, "which 3 chains make up THIS trimer copy" has one right
answer per copy.

This module's job is different: describe the FULL symmetry structure of
an assembly — every cyclic axis a chain participates in, simultaneously.
A single chain in an octahedral cage sits on a C4 axis (relating it to 3
other chains around a square face), a C3 axis (relating it to 2 other
chains in its own trimer), AND one or more C2 axes (relating it to
neighbors across edges) all at once. None of these compete with each
other — they're all physically real, simultaneous descriptions of the
same rigid body's symmetry environment. Reusing distance.py's exclusive-
assignment machinery here would silently throw away exactly the
information this module exists to report, so the graph/cycle-search
logic below is written fresh, without any "claim a chain, remove it from
the pool" step anywhere.

CORE DOMAIN RULES THIS MODULE IS BUILT AROUND
------------------------------------------------------------------------
1. MULTI-AXIS OVERLAP — a chain can belong to any number of symmetry
   axes at once. Nothing here partitions chains or removes them from
   candidate pools once used in one grouping. Every valid grouping found
   is kept; the same chain can appear in many rows of the output.

2. NO LOW HARDCODED DISTANCE CUTOFF — a real fusion/interface junction
   in a large subunit can run past 50-60 Angstroms (long terminal arms,
   big subunits) and still be completely valid, so long as it repeats
   consistently around the ring. Validity here is decided by STEP
   HOMOGENEITY (see below), not by how short the steps are in absolute
   terms. `max_candidate_distance` in this module is NOT a biological
   cutoff — it exists purely so the cycle search stays computationally
   tractable on a large assembly (a fully-connected candidate graph over
   dozens of chains makes exhaustive cycle enumeration explode
   combinatorially). It defaults generously high (see
   DEFAULT_MAX_CANDIDATE_DISTANCE) specifically so it never excludes a
   genuine long-but-uniform junction; set it to None to disable entirely
   on small structures where exhaustive search is affordable.

3. NON-DIRECTIONAL C2 — a two-fold interface can be head-to-head (C-C),
   tail-to-tail (N-N), or head-to-tail (N-C/C-N), and unlike a C3+ ring,
   there's no larger directed loop to walk: two chains are the whole
   story. find_c2_interfaces checks all of N-N, C-C, N-C, and C-N as
   plain undirected pairwise measurements, independent of the directed
   cyclic-walk machinery used for C3/C4/C5.

4. NO GLOBAL POINT-GROUP FITTING — nothing here computes a whole-
   assembly centroid, a global rotation/point-group matrix, or does any
   kind of rigid-body superposition. Real cages are locally rigid but
   globally floppy (crystal packing, minor asymmetry between chemically
   identical copies), so every measurement stays strictly LOCAL: a
   distance between two specific terminus coordinates, nothing more.

STEP HOMOGENEITY — THE MATH BEHIND RING VALIDITY
------------------------------------------------------------------------
For a candidate C_n ring (n = 3, 4, or 5), walking the directed path
chain_1(C) -> chain_2(N) -> chain_2(C) -> chain_3(N) -> ... ->
chain_n(C) -> chain_1(N) produces n step distances [d_1, ..., d_n]. In a
genuine cyclic assembly, every one of these steps is a copy of the SAME
physical interface, repeated around the ring by the assembly's own
rotational symmetry — so the n distances should agree with each other
(up to ordinary biological jitter), regardless of whether that shared
value is 8 Angstroms or 60. Two equivalent ways to test that agreement
are used, and a ring passes if EITHER holds:

    d_mean = mean(d_1, ..., d_n)
    d_std  = population standard deviation of (d_1, ..., d_n)
    CV     = d_std / d_mean                      (coefficient of variation)

    valid  = (d_std <= tolerance) OR (CV <= relative_tolerance)

d_std alone is an absolute-Angstrom yardstick (good for tight, short
junctions where even a couple of Angstroms of spread is meaningful).
CV is a scale-free yardstick (good for the long-arm case: a 60 Angstrom
step wandering by 3 Angstroms is just as internally consistent, in
relative terms, as an 8 Angstrom step wandering by 0.4). Requiring only
ONE of the two to hold — rather than both — is what lets this module
treat short and long junctions on an equal footing instead of silently
being biased toward short ones.

A C2 axis (a plain undirected two-chain interface — see find_c2_
interfaces) has exactly ONE measurement per interface type by
construction (d(N_i, N_j) is a single symmetric number, not n repeated
copies of anything), so there is no second, independent sample of "the
same interface" to check for agreement against. std/CV are reported as
0.0 for these rows for API consistency, not because the interface was
tested for homogeneity and passed — see find_c2_interfaces' docstring.

MODULE LAYOUT
------------------------------------------------------------------------
  1. Sequence-identity pre-clustering (group_chains_by_identity)
  2. Terminal-distance helpers shared by both detection paths
  3. Directed cyclic-walk detection for C3/C4/C5
     (build_directed_terminal_graph, find_cyclic_symmetry_groups)
  4. Undirected pairwise detection for C2 (find_c2_interfaces)
  5. Per-identity-group orchestrator (detect_symmetry_groupings)
  6. Global polyhedral diagnostic (summarize_axis_counts,
     polyhedral_diagnostic)
  7. Per-assembly / batch entry points (analyze_assembly_symmetry,
     run_symmetry_analysis) — these return the pandas DataFrame described
     in the module's calling convention: assembly/structure ID, symmetry
     axis type, constituent chain IDs, mean junction distance, and
     junction distance standard deviation.
"""

from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gemmi
import networkx as nx
import numpy as np
import pandas as pd

from toolkit.geometry.termini import get_chain_ca_geometry


# ============================================================================
# Domain constants
# ============================================================================

# The only ring sizes a Platonic-solid-type (T/O/I point group) cage can
# physically have. C2 is handled separately (see CYCLIC_RING_SIZES below)
# because a two-membered "ring" isn't a directed loop with independently
# repeating steps the way C3+ is — it's a single undirected interface.
ALLOWED_RING_SIZES: Tuple[int, ...] = (2, 3, 4, 5)

# Ring sizes handled by the directed C-terminus -> N-terminus cyclic-walk
# method (find_cyclic_symmetry_groups). C2 is deliberately excluded here —
# see find_c2_interfaces and rule 3 in the module docstring.
CYCLIC_RING_SIZES: Tuple[int, ...] = (3, 4, 5)

# Absolute-distance step-homogeneity gate, in Angstroms (see module
# docstring's STEP HOMOGENEITY section). A ring passes if its step
# distances' standard deviation is within this OR their coefficient of
# variation is within relative_tolerance below — either is sufficient.
DEFAULT_TOLERANCE: float = 2.0

# Scale-free step-homogeneity gate: coefficient of variation (d_std /
# d_mean), expressed as a fraction (0.05 = 5%), not a percentage. This is
# what makes a 60 Angstrom long-arm junction and an 8 Angstrom compact
# junction equally checkable for internal consistency — see module
# docstring.
DEFAULT_RELATIVE_TOLERANCE: float = 0.05

# NOT a biological cutoff — see rule 2 in the module docstring. This is a
# search-space prefilter so the directed terminal graph stays sparse
# enough for networkx's cycle enumeration to stay tractable on a large
# assembly (a fully dense graph over dozens of chains makes exhaustive
# elementary-cycle search combinatorially explode). Set generously above
# the ~50-60 Angstrom real long-arm junctions this module is meant to
# still catch, with margin. Set to None to disable entirely (only
# affordable on small chain counts, or when you already know every real
# junction is short).
DEFAULT_MAX_CANDIDATE_DISTANCE: Optional[float] = 100.0

# Minimum fractional sequence identity (0-1) for two chains to be treated
# as literal copies of the same homo-oligomeric building block — see
# group_chains_by_identity. 0.9 (90%) is generous enough to tolerate a
# handful of unresolved/mutated residues between nominally identical
# copies, while still keeping genuinely different proteins in a multi-
# component cage from ever being compared to each other.
DEFAULT_IDENTITY_THRESHOLD: float = 0.9

# Reference axis counts for the three Platonic-solid (T/O/I) point
# groups a self-assembling cage can realistically adopt, i.e. the number
# of DISTINCT rotational axes of each order the point group contains
# (identity operation excluded). These are textbook point-group facts,
# not fitted to any specific structure:
#   Tetrahedral (T), order 12  : 4 x C3 axes, 3 x C2 axes
#   Octahedral  (O), order 24  : 3 x C4 axes, 4 x C3 axes, 6 x C2 axes
#   Icosahedral (I), order 60  : 6 x C5 axes, 10 x C3 axes, 15 x C2 axes
# Used by polyhedral_diagnostic as a soft best-fit reference, not a
# strict equality target — see that function's docstring for why.
THEORETICAL_POLYHEDRAL_AXES: Dict[str, Dict[str, int]] = {
    "Tetrahedral (T)": {"C2": 3, "C3": 4},
    "Octahedral (O)": {"C2": 6, "C3": 4, "C4": 3},
    "Icosahedral (I)": {"C2": 15, "C3": 10, "C5": 6},
}


# ============================================================================
# 1. Sequence-identity pre-clustering
# ============================================================================

def _chain_sequences(model: gemmi.Model) -> Dict[str, str]:
    """
    Extracts the one-letter polymer sequence for every chain in `model`
    that has a resolved polymer. Chains with no polymer (ligand-only,
    solvent, etc.) are skipped — there's no meaningful sequence identity
    to compute for them, and they have no termini for this module's
    purposes anyway.
    """
    sequences: Dict[str, str] = {}
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) == 0:
            continue
        sequences[chain.name] = gemmi.one_letter_code(polymer.extract_sequence())
    return sequences


def _sequence_identity_ratio(seq_a: str, seq_b: str) -> float:
    """
    Approximate fractional sequence identity between two one-letter
    sequences, in [0, 1].

    Uses difflib.SequenceMatcher's ratio() rather than a full pairwise
    alignment: homo-oligomeric copies within one assembly are expected to
    be the SAME protein (identical up to unresolved residues or the
    occasional point mutation/engineering tag), not distantly related
    homologs needing a substitution-matrix alignment to compare fairly.
    For near-identical sequences (the only case this module needs to
    resolve — "is this chain a literal copy of that one, or a genuinely
    different component?"), SequenceMatcher's ratio tracks true percent
    identity closely without pulling in an extra alignment dependency.
    It is a coarser tool than a real alignment for sequences that are
    only distantly related, but that distinction doesn't matter here:
    anything below the clustering threshold is treated as "different
    component" regardless of how far below it actually is.
    """
    if not seq_a or not seq_b:
        return 0.0
    return SequenceMatcher(None, seq_a, seq_b).ratio()


def group_chains_by_identity(
    model: gemmi.Model, identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD
) -> List[List[str]]:
    """
    Groups chain names by sequence identity (>= identity_threshold, a
    fraction in [0, 1]) so that only chains which are effectively literal
    copies of the same protein are ever compared for a shared symmetry
    axis. This is the safeguard for multi-component cages: two chains
    from DIFFERENT proteins might still have terminus coordinates that
    happen to sit close together at an inter-component interface, but
    they can't be genuine copies related by a cyclic or two-fold
    symmetry operation of the SAME building block, so they're never
    allowed into the same candidate pool.

    Method: greedy single-linkage-to-representative clustering. Walk
    chains in whatever order `model` iterates them; for each chain, join
    the first existing cluster whose REPRESENTATIVE (the first chain that
    started that cluster) matches within identity_threshold, or start a
    new cluster if none does. This is deliberately simple (compares
    against one representative per cluster, not every existing member) —
    appropriate here because homo-oligomeric copies of one designed or
    natural subunit are expected to be near-identical to EACH OTHER, not
    just to some common ancestor, so representative-based clustering and
    full pairwise clustering should agree in practice for this specific
    use case.

    Returns a list of groups, each a list of chain names. Chains with no
    resolved polymer are absent from every group (see _chain_sequences).
    """
    sequences = _chain_sequences(model)

    representatives: List[str] = []  # one representative chain name per cluster
    clusters: List[List[str]] = []

    for chain_name, seq in sequences.items():
        placed = False
        for idx, rep_name in enumerate(representatives):
            if _sequence_identity_ratio(seq, sequences[rep_name]) >= identity_threshold:
                clusters[idx].append(chain_name)
                placed = True
                break
        if not placed:
            representatives.append(chain_name)
            clusters.append([chain_name])

    return clusters


# ============================================================================
# 2. Shared terminal-distance helpers
# ============================================================================

def _terminal_coords(
    chain_geometry: Dict[str, dict], chain_names: Sequence[str], terminus: str
) -> np.ndarray:
    """
    Stacks the requested terminus ("N" or "C") coordinate for every chain
    in `chain_names`, in order, into one (len(chain_names), 3) array —
    the vectorized building block both the directed (C3/C4/C5) and
    undirected (C2) distance computations are built on.
    """
    key = "n" if terminus.upper() == "N" else "c"
    return np.array([chain_geometry[name][key] for name in chain_names])


def _step_homogeneity(distances: Sequence[float]) -> Tuple[float, float, float]:
    """
    Core math for STEP HOMOGENEITY (see module docstring): given the step
    distances around one candidate ring (or, for C2, a single-element
    sequence), returns (mean, population_std, coefficient_of_variation).

    Population standard deviation (ddof=0) is used rather than the sample
    (ddof=1) version because these n values ARE the entire population of
    steps this specific candidate ring has — there's no larger population
    being sampled from, so there's no bias correction to apply.

    CV is guarded against division by zero: a mean of exactly 0.0 (chains
    with coincident termini — a geometrically degenerate case) reports
    CV as 0.0 if std is also 0 (perfectly, trivially uniform) or infinity
    otherwise (any spread around a zero mean is unbounded in relative
    terms).
    """
    arr = np.asarray(distances, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if mean == 0.0:
        cv = 0.0 if std == 0.0 else float("inf")
    else:
        cv = std / mean
    return mean, std, cv


def _passes_homogeneity(
    std: float, cv: float, tolerance: Optional[float], relative_tolerance: Optional[float]
) -> bool:
    """
    Applies the module docstring's validity rule: a ring is accepted if
    EITHER its absolute step spread (std) is within `tolerance` Angstroms
    OR its relative step spread (CV) is within `relative_tolerance`. This
    is an OR, not an AND — either yardstick clearing the bar is
    sufficient, precisely so a long, wide-radius junction (large mean,
    modest CV) and a short, tight one (small mean, modest std) are both
    reachable by whichever measure suits their own scale. Passing
    tolerance=None or relative_tolerance=None disables that specific
    check (both None would make every candidate fail, since neither
    yardstick could ever be satisfied — callers should supply at least
    one).
    """
    passes_absolute = tolerance is not None and std <= tolerance
    passes_relative = relative_tolerance is not None and cv <= relative_tolerance
    return passes_absolute or passes_relative


# ============================================================================
# 3. Directed cyclic-walk detection — C3 / C4 / C5
# ============================================================================

def build_directed_terminal_graph(
    chain_names: Sequence[str],
    chain_geometry: Dict[str, dict],
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> nx.DiGraph:
    """
    Builds the directed candidate-interface graph G = (V, E): one node
    per chain, and a directed edge i -> j wherever chain i's C-terminus
    to chain j's N-terminus distance is within max_candidate_distance (or
    every possible edge, if max_candidate_distance is None). Edge weights
    carry the raw C-to-N distance for later step-homogeneity scoring.

    max_candidate_distance is a computational-tractability prefilter, NOT
    a biological validity gate — see rule 2 in the module docstring. Ring
    ACCEPTANCE is decided entirely by find_cyclic_symmetry_groups' step-
    homogeneity check on whichever cycles this graph happens to contain,
    not by how short any individual edge is.

    Every chain in `chain_names` is added as a node even if it ends up
    with no qualifying edges at all, so downstream code can always rely
    on every input chain being present.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(chain_names)

    c_coords = _terminal_coords(chain_geometry, chain_names, "C")
    n_coords = _terminal_coords(chain_geometry, chain_names, "N")
    # Vectorized all-pairs C_i -> N_j distance matrix in one pass.
    diffs = c_coords[:, None, :] - n_coords[None, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=-1)

    n = len(chain_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(dist_matrix[i, j])
            if max_candidate_distance is None or d <= max_candidate_distance:
                graph.add_edge(chain_names[i], chain_names[j], weight=d)

    return graph


def find_cyclic_symmetry_groups(
    chain_names: Sequence[str],
    chain_geometry: Dict[str, dict],
    sizes: Sequence[int] = CYCLIC_RING_SIZES,
    tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> List[Dict[str, Any]]:
    """
    Finds every elementary directed cycle of length n in `sizes` (each n
    in {3, 4, 5}) within the directed C-to-N terminal graph, keeping only
    cycles whose n step distances pass the STEP HOMOGENEITY test (see
    module docstring and _passes_homogeneity) — i.e. every physically
    plausible ring, regardless of its absolute size, with NO exclusivity
    between candidates: a chain found in one valid ring is still fully
    eligible to appear in any other valid ring found, of the same or a
    different size (see rule 1 in the module docstring).

    Mathematically: a closed directed walk chain_1(C) -> chain_2(N) ->
    chain_2(C) -> chain_3(N) -> ... -> chain_n(C) -> chain_1(N) is
    exactly an elementary cycle of length n in this graph — cycle length
    directly IS the ring size, never guessed or tried as a candidate
    parameter ahead of time. networkx.simple_cycles (Johnson's algorithm)
    enumerates these directly, bounded by length_bound=max(sizes) so
    cycles longer than the largest requested size are never even
    constructed.

    Returns a list of dicts, each describing one accepted candidate:
      - "symmetry_type" : "C3" / "C4" / "C5"
      - "chain_order"    : chains in cyclic fusion order, e.g. (A, B, C)
                           meaning A -> B -> C -> A
      - "chains"         : the same membership as a SORTED tuple (order-
                           independent identity of the grouping, for
                           output/dedup purposes)
      - "junctions"      : list of (from_chain, to_chain, distance) for
                           every consecutive step, including the closing
                           step back to the first chain
      - "mean_distance"  : mean step distance (Angstroms), rounded to 2 dp
      - "std_distance"   : population std of the step distances, rounded
      - "cv"             : coefficient of variation (std / mean), or None
                           if undefined (mean of exactly 0 with nonzero
                           std — see _step_homogeneity)
      - "interface_type" : None (only meaningful for C2 — see
                           find_c2_interfaces)
      - "method"         : "directed_cycle"
    """
    invalid = sorted(set(sizes) - set(CYCLIC_RING_SIZES))
    if invalid:
        raise ValueError(
            f"find_cyclic_symmetry_groups only handles sizes in {CYCLIC_RING_SIZES} "
            f"(C2 is a plain undirected interface, not a directed cyclic walk — see "
            f"find_c2_interfaces) — got invalid size(s) {invalid}"
        )
    if not sizes or len(chain_names) < min(sizes):
        return []

    graph = build_directed_terminal_graph(chain_names, chain_geometry, max_candidate_distance)
    allowed_sizes = set(sizes)
    max_size = max(sizes)

    results: List[Dict[str, Any]] = []
    for cycle in nx.simple_cycles(graph, length_bound=max_size):
        size = len(cycle)
        if size not in allowed_sizes:
            continue

        edges = []
        for k in range(size):
            a, b = cycle[k], cycle[(k + 1) % size]
            edges.append((a, b, graph[a][b]["weight"]))

        distances = [e[2] for e in edges]
        mean, std, cv = _step_homogeneity(distances)

        if not _passes_homogeneity(std, cv, tolerance, relative_tolerance):
            continue

        results.append({
            "symmetry_type": f"C{size}",
            "chain_order": tuple(cycle),
            "chains": tuple(sorted(cycle)),
            "junctions": edges,
            "mean_distance": round(mean, 2),
            "std_distance": round(std, 2),
            "cv": round(cv, 4) if np.isfinite(cv) else None,
            "interface_type": None,
            "method": "directed_cycle",
        })

    return results


# ============================================================================
# 4. Undirected pairwise detection — C2
# ============================================================================

_C2_INTERFACE_TYPES: Tuple[str, ...] = ("N-N", "C-C", "N-C", "C-N")


def find_c2_interfaces(
    chain_names: Sequence[str],
    chain_geometry: Dict[str, dict],
    tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
    interface_types: Sequence[str] = _C2_INTERFACE_TYPES,
) -> List[Dict[str, Any]]:
    """
    Finds candidate C2 (two-fold) axes by checking every UNDIRECTED
    terminal-distance combination between each pair of chains — N-N
    (tail-to-tail), C-C (head-to-head), N-C, and C-N (the two head-to-
    tail arrangements) — independent of the directed cyclic-walk method
    used for C3/C4/C5 (see rule 3 in the module docstring).

    This is deliberately NOT built on top of find_cyclic_symmetry_groups
    with size=2: a directed 2-cycle would require BOTH chain_i(C) close
    to chain_j(N) AND chain_j(C) close to chain_i(N) simultaneously,
    which only captures the head-to-tail case, and even then imposes a
    directionality real C2 interfaces don't need to have. A genuine
    two-fold interface can be exactly ONE physical contact (e.g. two
    N-termini packed against each other, with the two C-termini pointing
    away in unrelated directions) — checked here as its own, independent
    undirected measurement.

    ON STEP HOMOGENEITY FOR C2: each interface type between one chain
    pair is a SINGLE measurement (d(N_i, N_j) is one number — there is no
    second, independently-repeating copy of "this same interface" the
    way a C3+ ring has n repeating steps around its loop). std and cv are
    therefore always reported as exactly 0.0 here — not because
    homogeneity was tested and found perfect, but because there is
    nothing to test variance across with a sample size of one.
    tolerance/relative_tolerance are accepted for signature symmetry with
    find_cyclic_symmetry_groups but have no effect on which C2 candidates
    are returned; the only filter actually applied is
    max_candidate_distance (a plausibility/tractability bound, same
    caveat as elsewhere — see rule 2 in the module docstring).

    Returns a list of dicts in the same shape as find_cyclic_symmetry_
    groups' output (symmetry_type is always "C2"; chain_order and chains
    both hold the sorted 2-chain tuple, since an undirected pair has no
    inherent direction to preserve; interface_type records which of
    N-N/C-C/N-C/C-N this row is; method is "undirected_interface").
    """
    del tolerance, relative_tolerance  # accepted for API symmetry only — see docstring

    coord_lookup = {
        "N": _terminal_coords(chain_geometry, chain_names, "N"),
        "C": _terminal_coords(chain_geometry, chain_names, "C"),
    }
    index_of = {name: i for i, name in enumerate(chain_names)}

    results: List[Dict[str, Any]] = []
    for a, b in combinations(chain_names, 2):
        ia, ib = index_of[a], index_of[b]
        candidate_distances = {
            "N-N": float(np.linalg.norm(coord_lookup["N"][ia] - coord_lookup["N"][ib])),
            "C-C": float(np.linalg.norm(coord_lookup["C"][ia] - coord_lookup["C"][ib])),
            "N-C": float(np.linalg.norm(coord_lookup["N"][ia] - coord_lookup["C"][ib])),
            "C-N": float(np.linalg.norm(coord_lookup["C"][ia] - coord_lookup["N"][ib])),
        }

        for itype in interface_types:
            d = candidate_distances[itype]
            if max_candidate_distance is not None and d > max_candidate_distance:
                continue

            results.append({
                "symmetry_type": "C2",
                "chain_order": tuple(sorted((a, b))),
                "chains": tuple(sorted((a, b))),
                "junctions": [(a, b, d)],
                "mean_distance": round(d, 2),
                "std_distance": 0.0,
                "cv": 0.0,
                "interface_type": itype,
                "method": "undirected_interface",
            })

    return results


# ============================================================================
# 5. Per-identity-group orchestrator
# ============================================================================

def detect_symmetry_groupings(
    chain_names: Sequence[str],
    chain_geometry: Dict[str, dict],
    ring_sizes: Sequence[int] = ALLOWED_RING_SIZES,
    tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> List[Dict[str, Any]]:
    """
    Runs both detection paths — find_cyclic_symmetry_groups for whichever
    of {3, 4, 5} appear in `ring_sizes`, and find_c2_interfaces if 2
    appears in `ring_sizes` — over ONE sequence-identity group of chains
    (see group_chains_by_identity), and returns the concatenated results.

    Both paths run against the SAME, FULL `chain_names` list — neither
    removes or reserves chains based on what the other found, and
    find_cyclic_symmetry_groups itself never removes a chain from
    contention after using it in one accepted cycle (see rule 1 in the
    module docstring). A chain can legitimately appear in the C4 rows,
    the C3 rows, AND several C2 rows this function returns, all at once.
    """
    invalid = sorted(set(ring_sizes) - set(ALLOWED_RING_SIZES))
    if invalid:
        raise ValueError(
            f"ring_sizes must be a subset of {ALLOWED_RING_SIZES} — Platonic-solid "
            f"(T/O/I point-group) cages only have 2-, 3-, 4-, and 5-fold rotational "
            f"symmetry axes, so no other ring size is physically possible — got "
            f"invalid size(s) {invalid}"
        )

    results: List[Dict[str, Any]] = []

    cyclic_sizes = tuple(s for s in ring_sizes if s in CYCLIC_RING_SIZES)
    if cyclic_sizes:
        results.extend(find_cyclic_symmetry_groups(
            chain_names, chain_geometry, sizes=cyclic_sizes,
            tolerance=tolerance, relative_tolerance=relative_tolerance,
            max_candidate_distance=max_candidate_distance,
        ))

    if 2 in ring_sizes:
        results.extend(find_c2_interfaces(
            chain_names, chain_geometry,
            tolerance=tolerance, relative_tolerance=relative_tolerance,
            max_candidate_distance=max_candidate_distance,
        ))

    return results


# ============================================================================
# 6. Global polyhedral diagnostic
# ============================================================================

def summarize_axis_counts(results_df: pd.DataFrame) -> Dict[str, int]:
    """
    Counts DISTINCT symmetry axes per type from a results DataFrame (the
    shape analyze_assembly_symmetry/run_symmetry_analysis return).

    "Distinct" here means unique (symmetry_type, chains) combinations —
    e.g. a C2 candidate that shows up under both its N-N and C-C
    interface_type (a real antiparallel two-contact dimer interface) is
    still ONE physical axis relating those two chains, not two, so it's
    only counted once. This matters for comparing against
    THEORETICAL_POLYHEDRAL_AXES, which counts physical axes, not raw
    detection rows.
    """
    if results_df.empty:
        return {}
    deduped = results_df.drop_duplicates(subset=["symmetry_type", "chains"])
    return deduped["symmetry_type"].value_counts().to_dict()


def polyhedral_diagnostic(
    results_df: pd.DataFrame, assembly_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Global diagnostic check: counts the total DISTINCT symmetry axes
    found per type (see summarize_axis_counts) and compares them against
    the theoretical axis counts of each Platonic-solid point group
    (Tetrahedral, Octahedral, Icosahedral — see
    THEORETICAL_POLYHEDRAL_AXES).

    This is intentionally a SOFT, best-fit report, not a strict
    equality check, for two reasons real structures actually run into:

      - Experimental coordinate truncation: a real downloaded structure
        can have a handful of unresolved termini, packing that pushes a
        real axis's junction just outside terminal detection settings,
        etc. — a genuinely icosahedral cage that only yields 13 of 15
        real C2 axes shouldn't be reported as "not icosahedral," it
        should be reported as "icosahedral, 13/15 C2 axes found."
      - Over-detection is possible too (e.g. a spurious coincidental
        contact passing the C2 undirected check) — found counts are
        capped at the theoretical expectation per axis type when scoring
        completeness, so a few extra spurious rows don't make a
        structure look MORE symmetric than physically possible, they
        just don't hurt its score either.

    completeness_score, per polyhedral type, starts as the mean over that
    type's own expected axis kinds (e.g. only C4/C3/C2 for Octahedral —
    never penalized for lacking C5, which Octahedral doesn't have) of
    min(found / expected, 1.0). 1.0 means every expected axis of every
    kind was found; lower scores point to which structure candidate is
    the best (if imperfect) fit, alongside a plain-language note listing
    exactly which axis kind(s) are short and by how much — the "soft
    warning" this function is meant to produce instead of a strict
    pass/fail.

    That raw coverage ratio alone can't distinguish between point groups
    whose expected axes are a SUBSET of a larger group's (Tetrahedral's
    3xC2 + 4xC3 is exactly the subset of Octahedral's 6xC2 + 4xC3 + 3xC4)
    — a genuinely octahedral cage would score a perfect 1.0 for
    Tetrahedral too, on coverage alone, despite also showing C4 axes that
    Tetrahedral's point group cannot have at all. To break that tie, any
    axis type found in the data that ISN'T one of this candidate's own
    expected kinds (a "foreign" axis — e.g. C4 axes present when scoring
    a Tetrahedral candidate) applies a soft multiplicative penalty,
    1 / (1 + foreign_axis_count), to that candidate's score. This isn't a
    hard disqualification (an isolated spurious foreign axis from noisy
    detection shouldn't zero out an otherwise-good candidate outright),
    but it reliably pushes a point group with unaccounted-for axis types
    below one that explains the full observed inventory.

    assembly_id : if given and results_df has an "assembly_id" column,
        restricts the diagnostic to that one assembly's rows first.
        Leave None to diagnose every row in results_df as one pool (only
        sensible if results_df already contains a single assembly).

    Returns a DataFrame with one row per candidate point group, sorted
    best-fit-first (highest completeness_score on top): polyhedral_type,
    expected_axes (dict), found_axes (dict), completeness_score, notes.
    """
    df = results_df
    if assembly_id is not None and "assembly_id" in df.columns:
        df = df[df["assembly_id"] == assembly_id]

    axis_counts = summarize_axis_counts(df)

    rows = []
    for poly_name, expected in THEORETICAL_POLYHEDRAL_AXES.items():
        per_type_ratio = {}
        missing_notes = []
        for sym_type, expected_count in expected.items():
            found = axis_counts.get(sym_type, 0)
            ratio = min(found / expected_count, 1.0) if expected_count else 1.0
            per_type_ratio[sym_type] = ratio
            if found < expected_count:
                missing_notes.append(f"{sym_type}: found {found}/{expected_count}")

        coverage = float(np.mean(list(per_type_ratio.values()))) if per_type_ratio else 0.0

        # Foreign-axis tie-break — see docstring: penalize (softly, not a
        # hard zero) any axis type present in the data that this point
        # group's own rotational symmetry can't produce at all.
        foreign_axis_count = sum(
            count for sym_type, count in axis_counts.items() if sym_type not in expected
        )
        penalty = 1.0 / (1.0 + foreign_axis_count)
        completeness = coverage * penalty

        notes_parts = []
        if missing_notes:
            notes_parts.append("under-accounted axes: " + "; ".join(missing_notes))
        else:
            notes_parts.append("every expected axis type fully accounted for")
        if foreign_axis_count:
            foreign_breakdown = {
                sym_type: count for sym_type, count in axis_counts.items()
                if sym_type not in expected and count > 0
            }
            notes_parts.append(
                f"{foreign_axis_count} axis(es) of a type this point group cannot "
                f"have were also found ({foreign_breakdown}), penalizing this fit"
            )
        notes = "soft warning — " + "; ".join(notes_parts) if (missing_notes or foreign_axis_count) else "complete match — " + notes_parts[0]

        rows.append({
            "polyhedral_type": poly_name,
            "expected_axes": dict(expected),
            "found_axes": {sym_type: axis_counts.get(sym_type, 0) for sym_type in expected},
            "completeness_score": round(completeness, 3),
            "notes": notes,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("completeness_score", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================================
# 7. Per-assembly and batch entry points
# ============================================================================

_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "assembly_id", "symmetry_type", "chains", "chain_order",
    "mean_distance", "std_distance", "cv", "interface_type", "method",
)


def analyze_assembly_symmetry(
    filepath: str,
    assembly_id: str,
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    ring_sizes: Sequence[int] = ALLOWED_RING_SIZES,
    tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> pd.DataFrame:
    """
    Full per-assembly symmetry-grouping pipeline: load the structure,
    compute per-chain terminal geometry via termini.get_chain_ca_geometry,
    group chains by sequence identity (group_chains_by_identity — the
    multi-component-cage safeguard), then run detect_symmetry_groupings
    independently within each identity group.

    Every identity group is handled independently: chains from different
    proteins in a multi-component cage never compete for, or get merged
    into, the same axis (see group_chains_by_identity), and different
    groups can naturally end up with entirely different symmetry_type
    axes present (e.g. one protein forming trimers, another forming
    dimers).

    Returns a pandas DataFrame with one row PER DISTINCT SYMMETRY AXIS
    FOUND (no partitioning, no exclusivity — a chain can legitimately
    appear across many rows; see rule 1 in the module docstring), columns:

      - assembly_id     : as passed in
      - symmetry_type   : "C2" / "C3" / "C4" / "C5"
      - chains          : constituent chain IDs, as a sorted tuple
      - chain_order     : constituent chain IDs in cyclic fusion order
                          (C3/C4/C5) or the same sorted pair (C2, which
                          has no inherent direction)
      - mean_distance   : mean junction distance, Angstroms
      - std_distance    : junction distance standard deviation, Angstroms
                          (see find_c2_interfaces re: always 0.0 for C2)
      - cv              : coefficient of variation (std / mean), or None
      - interface_type  : which of N-N/C-C/N-C/C-N this row is, for C2
                          rows; None for C3/C4/C5 rows
      - method          : "directed_cycle" or "undirected_interface"

    Empty (but correctly-columned) if the structure has fewer than 2
    chains with usable CA geometry, or no identity group is large enough
    to form any requested ring size.
    """
    invalid = sorted(set(ring_sizes) - set(ALLOWED_RING_SIZES))
    if invalid:
        raise ValueError(
            f"ring_sizes must be a subset of {ALLOWED_RING_SIZES} — got invalid "
            f"size(s) {invalid}"
        )

    st = gemmi.read_structure(filepath)
    model = st[0]

    chain_geometry: Dict[str, dict] = {}
    for chain in model:
        geometry = get_chain_ca_geometry(chain)
        if geometry is not None:
            chain_geometry[chain.name] = geometry

    identity_groups = group_chains_by_identity(model, identity_threshold=identity_threshold)

    rows = []
    for group in identity_groups:
        usable = [name for name in group if name in chain_geometry]
        if len(usable) < 2:
            continue

        groupings = detect_symmetry_groupings(
            usable, chain_geometry, ring_sizes=ring_sizes,
            tolerance=tolerance, relative_tolerance=relative_tolerance,
            max_candidate_distance=max_candidate_distance,
        )

        for g in groupings:
            rows.append({
                "assembly_id": assembly_id,
                "symmetry_type": g["symmetry_type"],
                "chains": g["chains"],
                "chain_order": g["chain_order"],
                "mean_distance": g["mean_distance"],
                "std_distance": g["std_distance"],
                "cv": g["cv"],
                "interface_type": g["interface_type"],
                "method": g["method"],
            })

    result_df = pd.DataFrame(rows, columns=list(_OUTPUT_COLUMNS))
    if not result_df.empty:
        result_df = result_df.sort_values(
            ["symmetry_type", "mean_distance"]
        ).reset_index(drop=True)
    return result_df


def run_symmetry_analysis(
    df: pd.DataFrame,
    filepath_column: str = "filepath",
    assembly_id_column: str = "assembly_id",
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    ring_sizes: Sequence[int] = ALLOWED_RING_SIZES,
    tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> pd.DataFrame:
    """
    Runs analyze_assembly_symmetry over every row of a downloaded-
    candidates DataFrame (i.e. what download.py's download_candidates()
    returns) and concatenates the results into one DataFrame — one row
    per distinct symmetry axis, across every assembly in the batch.

    assembly_id_column : if this column exists in df (it will, if df came
        from download_candidates()), it's used directly; otherwise falls
        back to f"{entry_id}-{assembly_num}", matching distance.py's
        convention for the same situation.

    Failures on individual structures are caught and reported by
    assembly_id rather than stopping the whole batch, same rationale as
    distance.py's process_candidates/run_ring_analysis: one malformed or
    unexpectedly-shaped structure shouldn't take down a run over hundreds
    of candidates.
    """
    frames = []
    for _, row in df.iterrows():
        if assembly_id_column in df.columns:
            assembly_id = row[assembly_id_column]
        else:
            assembly_id = f"{row['entry_id']}-{row['assembly_num']}"

        try:
            frames.append(analyze_assembly_symmetry(
                row[filepath_column], assembly_id,
                identity_threshold=identity_threshold, ring_sizes=ring_sizes,
                tolerance=tolerance, relative_tolerance=relative_tolerance,
                max_candidate_distance=max_candidate_distance,
            ))
        except Exception as e:
            print(f"Failed: {assembly_id} — {e}")

    if not frames:
        return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))

    result_df = pd.concat(frames, ignore_index=True)
    if not result_df.empty:
        result_df = result_df.sort_values(
            ["assembly_id", "symmetry_type", "mean_distance"]
        ).reset_index(drop=True)
    return result_df