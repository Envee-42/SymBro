"""
symmetry.py — cyclic symmetry (C2/C3/C4/C5) grouping detection, exclusivity
resolution, point-group inference, and per-order summarization for
self-assembling protein cage structures.

Built entirely on top of termini.py's per-chain terminal coordinates
(termini.get_chain_ca_geometry) — every distance computed here is between
two specific N/C terminal Cα coordinates. Nothing in this module ever
computes a whole-assembly centroid, a global coordinate origin, or fits a
fixed rotation/point-group matrix: cage subunits pack with real local
flexibility (crystallographic disorder, minor asymmetry between
chemically identical copies), so a rigid global-alignment approach is
exactly the wrong tool here (see rule 4 below).

CORE DOMAIN RULES THIS MODULE IS BUILT AROUND
------------------------------------------------------------------------
1. MULTI-SYMMETRY OVERLAP, WITH PER-ORDER EXCLUSIVITY — a chain can sit
   on a C2 axis AND a C3 axis AND a C4 axis all at the same time (this is
   literally how a T/O/I point-group cage is built: one subunit
   participates in several different rotational relationships at once).
   But WITHIN one order k, the accepted C_k rings must be a set of
   DISJOINT chain groups — no chain may be claimed by two different C_k
   rings simultaneously. A ring and every rotation of it (('A','B','C')
   vs ('B','C','A')) are the same physical object and are canonicalized
   to compare equal before any exclusivity logic runs (see
   _canonical_cycle).

2. STEP HOMOGENEITY OVER SHORT DISTANCE — no hardcoded low absolute
   distance cutoff. A real fusion/interface junction in a large subunit
   can exceed 50 Angstroms and still be entirely valid, provided the
   steps around the ring agree with each other. Validity is judged by
   the STEP TOLERANCE FILTER: d_std <= 2.0 Angstroms OR
   (d_std / d_mean) <= 0.05 — either condition is sufficient, so a long,
   wide junction (large mean, small CV) and a short, tight one (small
   mean, small std) are both reachable on their own terms. The one
   distance knob this module DOES use, `max_candidate_distance`, is a
   computational-tractability prefilter only (keeps the directed-cycle
   search from exploding combinatorially on a large assembly) — it
   defaults generously high specifically so it never substitutes for the
   real, homogeneity-based validity test.

3. NON-DIRECTIONAL C2 DIMERS — a two-fold interface has no larger loop to
   walk (it's exactly two chains), and can be head-to-head (C-C),
   tail-to-tail (N-N), or head-to-tail (N-C/C-N). find_c2_candidates
   checks all of these as plain UNDIRECTED measurements between a chain
   pair, independent of the directed cyclic-walk machinery used for
   C3/C4/C5.

4. RIGID BODY GEOMETRY — no global coordinate origin / center-of-mass,
   no fixed rotation matrices, anywhere in this module. Every distance is
   a local, pairwise terminus-to-terminus measurement.

MATH BEHIND RING VALIDITY (STEP HOMOGENEITY)
------------------------------------------------------------------------
For a directed candidate C_n ring (n = 3, 4, 5), walking
chain_1(C) -> chain_2(N) -> chain_2(C) -> chain_3(N) -> ... ->
chain_n(C) -> chain_1(N) produces n step distances [d_1, ..., d_n]. In a
genuine rotationally-symmetric ring, every step is a copy of the SAME
physical interface, repeated around the loop by the assembly's own
symmetry, so the n values should agree with each other (ordinary
biological jitter aside) regardless of whether their shared value is 8
Angstroms or 60. Two equivalent agreement tests are used; a ring is
valid if EITHER holds:

    d_mean = mean(d_1, ..., d_n)
    d_std  = population standard deviation of (d_1, ..., d_n)
    CV     = d_std / d_mean

    valid  = (d_std <= tolerance) OR (CV <= relative_tolerance)

For C2, an N-N or C-C interface has exactly ONE measurement (there is
only one d(N_i, N_j) — no second, independent copy of "the same
interface" to compare it against), so std/CV are trivially 0.0 for those
two interface types. The N-C / C-N interface type, however, DOES have two
independent measurements available for the same pair — d(N_i, C_j) and
d(C_i, N_j), the two possible head-to-tail directions — and those two ARE
checked against each other via the same d_std/CV test, since a genuine
reciprocal two-fold relationship should show both directions agreeing.

DISJOINT CYCLE ASSIGNMENT — HOW EXCLUSIVITY IS ACTUALLY ENFORCED
------------------------------------------------------------------------
Within one symmetry order k, several candidate rings can be found that
share chains (e.g. two different candidate C3 triples that both include
chain 'B', where only one can be the REAL trimer 'B' belongs to). This is
resolved by select_disjoint_axes: sort all of that order's homogeneity-
passing candidates by ascending Coefficient of Variation (tightest,
lowest-CV rings first — the CV is preferred over raw std here as the
sort key because it is what actually decides pass/fail for a long-arm
ring, and using the same statistic for ranking that decides validity
keeps "best" and "valid" consistent), then walk the sorted list and greedily
accept each candidate whose chain set is still entirely unclaimed,
marking its chains as used. A candidate that shares even one chain with
an already-accepted (and therefore tighter-or-equal) candidate is
rejected. This guarantees the final accepted set for order k is disjoint
by construction, and that ties always favor the geometrically tightest
reading of the data.

POINT-GROUP INFERENCE AND THE IMPOSSIBILITY GUARDRAIL
------------------------------------------------------------------------
A cage's total chain count (within one sequence-identity group — see
group_chains_by_identity) is a strong, purely combinatorial signal for
which point group it can be: a Tetrahedral (T) point-group cage built
from one homo-oligomeric building block has exactly 12 copies of that
chain, Octahedral (O) has 24, Icosahedral (I) has 60, and a Dihedral D_n
cage has 2n. This falls directly out of group order (T has rotational
order 12, O has 24, I has 60) combined with each copy of the subunit
occupying exactly one position in the point group's orbit.

That same combinatorial fact fixes the MAXIMUM number of DISJOINT C_k
rings a point group of n_chains total copies can possibly support:
floor(n_chains / k) (you cannot disjointly partition n_chains chains into
more than that many k-sized groups). This is exactly why the problem
statement's reference maximums (T: 6 C2 / 4 C3; O: 12 C2 / 8 C3 / 6 C4;
I: 30 C2 / 20 C3 / 12 C5) all equal n_chains // k for their respective
group — they are not an independent rule to separately enforce, they are
a consequence of exclusivity (rule 1) that this module gets for free once
disjoint selection is applied. What IS independently enforced is WHICH
orders are allowed to exist at all for a given point group (T has no C4
or C5 axis; O has no C5 axis; I has no C4 axis) — see
IMPOSSIBLE_ORDERS_BY_GROUP. Any accepted ring of a disallowed order is
pruned entirely for that identity group before results are reported.

Because chain count alone can be ambiguous (e.g. 12 chains is consistent
with both T and a hypothetical D6), the inference step breaks ties using
the ACTUAL disjoint axis counts already found: whichever candidate point
group's own allowed orders best explain the observed counts (a coverage
score), penalized if orders that point group cannot have were
nonetheless found in the data, wins. See infer_point_group.

If no reference point group's expected chain count matches this identity
group's chain count at all, point_group is reported as "Unknown" and NO
guardrail is applied — an unrecognized chain count isn't reason to
discard real geometric findings, it just means this module can't
confidently name the point group.

OUTPUT AGGREGATION
------------------------------------------------------------------------
analyze_assembly_symmetry emits exactly ONE row per detected symmetry
order (C2, C3, C4, C5) per structure — never one row per individual ring
— specifically to keep the output readable for a cage with many
symmetric copies of the same ring. Because a real point-group cage can
have more than one sequence-identity group (a multi-component assembly),
candidates from every identity group are pooled per order before this
row is built: the globally lowest-CV accepted ring across all groups
becomes that order's "Main Grouping", every other accepted (and
therefore, by construction, chain-disjoint) ring of that order becomes
an "Equivalent Axis Grouping", and the reported point group is whichever
identity group the main grouping came from (noted in-line if groups
disagree — see analyze_assembly_symmetry's docstring).
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

# The only cyclic symmetry orders this module detects. Platonic-solid
# (T/O/I) and dihedral (D_n) cages only have 2-, 3-, 4-, and 5-fold
# rotational axes to look for in the first place.
ALLOWED_ORDERS: Tuple[int, ...] = (2, 3, 4, 5)

# Orders handled by the directed C-terminus -> N-terminus cyclic-walk
# method (find_cyclic_ring_candidates). C2 is handled separately by
# find_c2_candidates — see rule 3 in the module docstring.
CYCLIC_ORDERS: Tuple[int, ...] = (3, 4, 5)

# Step Tolerance Filter (see module docstring's MATH section): a ring is
# valid if its step distances' population standard deviation is within
# this many Angstroms, OR its coefficient of variation is within
# DEFAULT_RELATIVE_TOLERANCE below. Either condition alone is sufficient.
DEFAULT_TOLERANCE: float = 2.0
DEFAULT_RELATIVE_TOLERANCE: float = 0.05

# NOT a biological validity rule (see rule 2) — a search-space prefilter
# only, so build_directed_terminal_graph doesn't have to construct and
# enumerate cycles over a fully dense graph on a large assembly. Set well
# above the ~50-60 Angstrom long-arm junctions this module must still be
# able to catch, with margin. None disables it entirely (only affordable
# for small chain counts).
DEFAULT_MAX_CANDIDATE_DISTANCE: Optional[float] = 100.0

# Minimum fractional sequence identity (0-1) for two chains to be treated
# as literal copies of the same homo-oligomeric building block.
DEFAULT_IDENTITY_THRESHOLD: float = 0.9

# Total chain count expected for a point-group cage built from ONE
# homo-oligomeric building block, per Platonic solid point group's
# rotational order. A Dihedral D_n cage (2n chains, n itself one of
# ALLOWED_ORDERS) is handled generically in _reference_point_groups
# rather than being pre-listed here, since n can vary.
_PLATONIC_REFERENCE_CHAIN_COUNTS: Dict[str, int] = {"T": 12, "O": 24, "I": 60}

# Which orders each Platonic point group's rotational symmetry actually
# contains. T has no 4-fold or 5-fold axis at all; O has no 5-fold axis;
# I has no 4-fold axis. This is the IMPOSSIBILITY GUARDRAIL reference —
# any accepted ring of an order not listed here for the inferred group is
# pruned entirely, regardless of how tight/valid it looked in isolation.
_PLATONIC_ALLOWED_ORDERS: Dict[str, Tuple[int, ...]] = {
    "T": (2, 3),
    "O": (2, 3, 4),
    "I": (2, 3, 5),
}


# ============================================================================
# 1. Sequence-identity pre-clustering
# ============================================================================

def _chain_sequences(model: gemmi.Model) -> Dict[str, str]:
    """
    One-letter polymer sequence for every chain in `model` with a
    resolved polymer. Chains with no polymer (ligand/solvent-only) are
    skipped — they have no meaningful sequence identity, and no termini
    for this module's purposes.
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
    Approximate fractional sequence identity in [0, 1], via
    difflib.SequenceMatcher.ratio(). Homo-oligomeric copies within one
    assembly are expected to be the SAME protein (identical up to
    unresolved residues or an occasional engineered mutation/tag), not
    distant homologs needing a substitution-matrix alignment to compare
    fairly — for that near-identical regime, ratio() tracks true percent
    identity closely without an extra alignment dependency. It is a
    coarser tool than a real alignment for genuinely distant sequences,
    but that distinction doesn't matter here: anything below the
    clustering threshold is simply treated as "a different component,"
    regardless of how far below it falls.
    """
    if not seq_a or not seq_b:
        return 0.0
    return SequenceMatcher(None, seq_a, seq_b).ratio()


def group_chains_by_identity(
    model: gemmi.Model, identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD
) -> List[List[str]]:
    """
    Groups chain names by sequence identity (>= identity_threshold) so
    that only chains which are effectively literal copies of the same
    protein are ever compared for a shared symmetry axis — the safeguard
    for multi-component cages, where two chains from DIFFERENT proteins
    might have terminus coordinates that happen to sit close together at
    an inter-component interface, but can never be genuine copies related
    by a cyclic/two-fold symmetry operation of the same building block.

    Method: greedy clustering against a single representative per
    cluster (the chain that started it) rather than full pairwise
    linkage — appropriate here because homo-oligomeric copies of one
    subunit are expected to closely resemble EACH OTHER, not just share
    a distant common ancestor, so representative-based and full pairwise
    clustering should agree in practice for this use case.

    Returns a list of groups, each a list of chain names. Chains with no
    resolved polymer are absent from every group.
    """
    sequences = _chain_sequences(model)

    representative_names: List[str] = []
    clusters: List[List[str]] = []

    for chain_name, seq in sequences.items():
        placed = False
        for idx, rep_name in enumerate(representative_names):
            if _sequence_identity_ratio(seq, sequences[rep_name]) >= identity_threshold:
                clusters[idx].append(chain_name)
                placed = True
                break
        if not placed:
            representative_names.append(chain_name)
            clusters.append([chain_name])

    return clusters


# ============================================================================
# 2. Shared geometry / statistics helpers
# ============================================================================

def _terminal_coords(
    chain_geometry: Dict[str, dict], chain_names: Sequence[str], terminus: str
) -> np.ndarray:
    """
    Stacks the requested terminus ("N" or "C") coordinate for every chain
    in `chain_names`, in order, into one (len(chain_names), 3) array.
    Purely a local per-chain lookup — never touches a whole-assembly
    centroid or origin (see rule 4 in the module docstring).
    """
    key = "n" if terminus.upper() == "N" else "c"
    return np.array([chain_geometry[name][key] for name in chain_names])


def _step_homogeneity(distances: Sequence[float]) -> Tuple[float, float, float]:
    """
    Core STEP HOMOGENEITY math (see module docstring): given the step
    distances around one candidate ring (or a 1- or 2-element sequence
    for a C2 candidate), returns (mean, population_std, coefficient_of_
    variation).

    Population standard deviation (ddof=0) is used because these values
    ARE the complete set of steps this one candidate ring has — there is
    no larger population being sampled from, so no bias correction
    applies.

    CV divides by mean and is guarded against a zero mean (coincident
    termini, a geometrically degenerate edge case): CV is reported as 0.0
    if std is also exactly 0 (perfectly, trivially uniform), or infinity
    otherwise (any spread around a zero mean is unbounded in relative
    terms, so it can never pass the relative_tolerance test).
    """
    arr = np.asarray(distances, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if mean == 0.0:
        cv = 0.0 if std == 0.0 else float("inf")
    else:
        cv = std / mean
    return mean, std, cv


def _passes_step_tolerance(
    std: float, cv: float, tolerance: Optional[float], relative_tolerance: Optional[float]
) -> bool:
    """
    The STEP TOLERANCE FILTER (module docstring): a candidate ring is
    valid if EITHER its absolute step spread (std) is within `tolerance`
    Angstroms OR its relative step spread (CV) is within
    `relative_tolerance` — an OR, not an AND, so a long, wide-radius
    junction (large mean, modest CV, possibly large std) and a short,
    tight one (small mean, modest std) are each reachable on the
    yardstick that actually suits their own scale. Passing either
    parameter as None disables that specific check.
    """
    passes_absolute = tolerance is not None and std <= tolerance
    passes_relative = relative_tolerance is not None and cv <= relative_tolerance
    return passes_absolute or passes_relative


def _canonical_cycle(cycle: Tuple[str, ...]) -> Tuple[str, ...]:
    """
    Canonicalizes a cyclic chain-ID sequence under ROTATION so that
    ('A', 'B', 'C') and ('B', 'C', 'A') — the same physical ring, walked
    starting from a different chain — compare and hash equal. Rotates the
    tuple so it begins at its lexicographically smallest chain ID; this
    is well-defined (a unique canonical form) as long as chain IDs within
    one ring are themselves unique, which they always are (a ring can't
    revisit the same chain — see find_cyclic_ring_candidates, which only
    searches ELEMENTARY / simple cycles).

    Direction is deliberately preserved (this canonicalizes rotations
    only, not reflections): chain_1(C) -> chain_2(N) -> ... is a
    genuinely different directed fusion path from its reverse, not
    merely a relabeling of the same one.
    """
    if not cycle:
        return cycle
    min_idx = min(range(len(cycle)), key=lambda i: cycle[i])
    return cycle[min_idx:] + cycle[:min_idx]


# ============================================================================
# 3. Directed cyclic-walk detection — C3 / C4 / C5
# ============================================================================

def build_directed_terminal_graph(
    chain_names: Sequence[str],
    chain_geometry: Dict[str, dict],
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> nx.DiGraph:
    """
    Builds the directed candidate-interface graph: one node per chain,
    and a directed edge i -> j wherever chain i's C-terminus to chain j's
    N-terminus distance is within max_candidate_distance (or every
    possible edge, if max_candidate_distance is None). Edge weights carry
    the raw C-to-N distance for the step-homogeneity test performed by
    find_cyclic_ring_candidates.

    max_candidate_distance is a computational-tractability prefilter,
    NOT a biological validity gate (rule 2 in the module docstring): ring
    ACCEPTANCE is decided entirely by step homogeneity on whatever cycles
    this graph happens to contain, never by how short any one edge is.

    Every chain in `chain_names` is added as a node even with no
    qualifying edges, so downstream code can always rely on every input
    chain being present.
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


def find_cyclic_ring_candidates(
    chain_names: Sequence[str],
    chain_geometry: Dict[str, dict],
    order: int,
    tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> List[Dict[str, Any]]:
    """
    Finds every ELEMENTARY (simple, non-self-intersecting) directed cycle
    of exactly length `order` in the directed C-to-N terminal graph,
    keeping only those whose step distances pass the STEP TOLERANCE
    FILTER (see _passes_step_tolerance). `order` must be one of
    CYCLIC_ORDERS (3, 4, or 5) — C2 has no directed loop to walk and is
    handled separately by find_c2_candidates.

    This function does NOT resolve exclusivity — it returns every
    homogeneity-passing candidate, including candidates that overlap in
    chain membership. Overlap resolution is select_disjoint_axes' job,
    run once per order over this function's output (see module
    docstring's DISJOINT CYCLE ASSIGNMENT section and rule 1).

    Cycles are canonicalized (_canonical_cycle) and de-duplicated by that
    canonical form before being returned, so a ring found starting from
    two different chains is reported exactly once.

    Each returned dict:
      - "order"          : `order` (3, 4, or 5)
      - "chains"         : chain IDs in canonical cyclic fusion order,
                            e.g. ("A", "B", "C") meaning A -> B -> C -> A
      - "chain_set"       : frozenset(chains) — the O(1) membership-
                            overlap check select_disjoint_axes needs
      - "junctions"      : list of (from_chain, to_chain, distance) for
                            every consecutive step, closing the loop
      - "mean_distance"  : mean step distance (Angstroms)
      - "std_distance"   : population std of the step distances
      - "cv"             : coefficient of variation, or None if
                            undefined (see _step_homogeneity)
      - "interface_type" : None (only meaningful for C2 candidates)
    """
    if order not in CYCLIC_ORDERS:
        raise ValueError(
            f"find_cyclic_ring_candidates only handles orders in {CYCLIC_ORDERS} — "
            f"C2 is a plain undirected interface, not a directed cyclic walk (see "
            f"find_c2_candidates) — got order={order}"
        )
    if len(chain_names) < order:
        return []

    graph = build_directed_terminal_graph(chain_names, chain_geometry, max_candidate_distance)

    seen_canonical: set = set()
    results: List[Dict[str, Any]] = []
    for cycle in nx.simple_cycles(graph, length_bound=order):
        if len(cycle) != order:
            continue

        canonical = _canonical_cycle(tuple(cycle))
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)

        edges = []
        for k in range(order):
            a, b = canonical[k], canonical[(k + 1) % order]
            edges.append((a, b, graph[a][b]["weight"]))

        distances = [e[2] for e in edges]
        mean, std, cv = _step_homogeneity(distances)

        if not _passes_step_tolerance(std, cv, tolerance, relative_tolerance):
            continue

        results.append({
            "order": order,
            "chains": canonical,
            "chain_set": frozenset(canonical),
            "junctions": edges,
            "mean_distance": round(mean, 2),
            "std_distance": round(std, 2),
            "cv": round(cv, 4) if np.isfinite(cv) else None,
            "interface_type": None,
        })

    return results


# ============================================================================
# 4. Undirected pairwise detection — C2
# ============================================================================

def find_c2_candidates(
    chain_names: Sequence[str],
    chain_geometry: Dict[str, dict],
    tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> List[Dict[str, Any]]:
    """
    Finds candidate C2 (two-fold) axes by checking every UNDIRECTED
    terminal-distance combination between each chain pair — N-N
    (tail-to-tail), C-C (head-to-head), and N-C (head-to-tail, evaluated
    in both possible directions) — independent of the directed cyclic-
    walk method used for C3/C4/C5 (rule 3 in the module docstring).

    For N-N and C-C, there is exactly ONE measurement between a given
    pair (d(N_i, N_j) is a single number, not n repeating copies of
    anything), so std/cv are reported as 0.0 for those rows — trivially,
    not because homogeneity was tested and passed with a sample size of
    one.

    For N-C, there ARE two independent measurements per pair — d(N_i,
    C_j) and d(C_i, N_j), the two possible head-to-tail directions — and
    the SAME step-homogeneity test used for C3/C4/C5 is applied across
    exactly those two values, since a genuine reciprocal two-fold
    relationship should show both directions agreeing with each other.

    Every (pair, interface_type) combination that passes the STEP
    TOLERANCE FILTER (or is trivially homogeneous, for N-N/C-C) is kept
    as its own candidate — including, potentially, more than one
    interface type for the same pair. This is deliberate: it lets
    select_disjoint_axes (run once, over the pooled C2 candidates from
    this function) pick whichever interface type is tightest for a given
    pair via its normal ascending-CV sort, with no special-casing needed
    here for "the" interface type of a pair.

    Returns a list of dicts in the same shape as find_cyclic_ring_
    candidates' output (order is always 2; "chains" holds the sorted
    2-chain tuple, since an undirected pair has no inherent cyclic
    direction to preserve; "interface_type" records which of
    N-N/C-C/N-C this row is).
    """
    n_coords = _terminal_coords(chain_geometry, chain_names, "N")
    c_coords = _terminal_coords(chain_geometry, chain_names, "C")
    index_of = {name: i for i, name in enumerate(chain_names)}

    results: List[Dict[str, Any]] = []
    for a, b in combinations(chain_names, 2):
        ia, ib = index_of[a], index_of[b]
        pair = tuple(sorted((a, b)))
        pair_set = frozenset(pair)

        # N-N and C-C: one measurement each, trivially homogeneous.
        for itype, d in (
            ("N-N", float(np.linalg.norm(n_coords[ia] - n_coords[ib]))),
            ("C-C", float(np.linalg.norm(c_coords[ia] - c_coords[ib]))),
        ):
            if max_candidate_distance is not None and d > max_candidate_distance:
                continue
            results.append({
                "order": 2, "chains": pair, "chain_set": pair_set,
                "junctions": [(a, b, d)],
                "mean_distance": round(d, 2), "std_distance": 0.0, "cv": 0.0,
                "interface_type": itype,
            })

        # N-C: two genuinely independent reciprocal measurements, tested
        # against each other exactly like a C3+ ring's steps.
        nc_forward = float(np.linalg.norm(n_coords[ia] - c_coords[ib]))  # N_a -> C_b
        nc_reverse = float(np.linalg.norm(c_coords[ia] - n_coords[ib]))  # C_a -> N_b
        if max_candidate_distance is None or (
            nc_forward <= max_candidate_distance and nc_reverse <= max_candidate_distance
        ):
            mean, std, cv = _step_homogeneity([nc_forward, nc_reverse])
            if _passes_step_tolerance(std, cv, tolerance, relative_tolerance):
                results.append({
                    "order": 2, "chains": pair, "chain_set": pair_set,
                    "junctions": [(a, b, nc_forward), (b, a, nc_reverse)],
                    "mean_distance": round(mean, 2), "std_distance": round(std, 2),
                    "cv": round(cv, 4) if np.isfinite(cv) else None,
                    "interface_type": "N-C",
                })

    return results


# ============================================================================
# 5. Disjoint-set exclusivity resolution (rule 1 / Algorithmic Req. #2)
# ============================================================================

def select_disjoint_axes(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Resolves the EXCLUSIVITY RULE for one symmetry order: given every
    homogeneity-passing candidate ring of that order (possibly with
    overlapping chain membership — e.g. two different candidate C3
    triples both including chain 'B'), returns the subset that is
    mutually chain-DISJOINT, preferring the geometrically tightest
    reading of the data whenever candidates conflict.

    Method (greedy, tightest-first): sort every candidate ascending by
    coefficient of variation (a candidate with cv=None — the degenerate
    zero-mean case from _step_homogeneity — sorts last, treated as the
    least trustworthy), breaking ties by std then by mean distance so the
    ordering is fully deterministic. Walk the sorted list once; accept a
    candidate if its chain_set is disjoint from every chain already
    claimed by a PREVIOUSLY accepted (and therefore tighter-or-equal)
    candidate, and mark its chains as claimed. Reject it otherwise.

    Why CV (not raw distance) is the sort key: CV is exactly the
    statistic the STEP TOLERANCE FILTER itself uses to judge relative
    consistency, so "tightest" and "most clearly valid" mean the same
    thing here — a long-arm ring with a large mean distance but a small
    CV is preferred over a short ring with a larger CV, consistent with
    rule 2's requirement that validity never be biased toward short
    absolute distances.

    Because acceptance only ever checks against PREVIOUSLY accepted
    candidates (not all candidates), a tight, early-accepted ring gets
    first claim on its own chains before a looser, overlapping
    alternative ever has a chance to contest them. The result is, by
    construction, a set of chain-disjoint rings — the DISJOINT CYCLE
    ASSIGNMENT this module's exclusivity rule requires.
    """
    def sort_key(c: Dict[str, Any]) -> Tuple[float, float, float]:
        cv = c["cv"] if c["cv"] is not None else float("inf")
        return (cv, c["std_distance"], c["mean_distance"])

    ordered = sorted(candidates, key=sort_key)

    claimed: set = set()
    accepted: List[Dict[str, Any]] = []
    for candidate in ordered:
        if claimed.isdisjoint(candidate["chain_set"]):
            accepted.append(candidate)
            claimed.update(candidate["chain_set"])

    return accepted


# ============================================================================
# 6. Point-group inference and the impossibility guardrail
# ============================================================================

def _reference_point_groups(n_chains: int) -> Dict[str, Tuple[int, ...]]:
    """
    Every point-group label whose theoretical total chain count (for one
    homo-oligomeric building block) equals `n_chains` exactly, mapped to
    the rotational orders that point group's symmetry actually contains.

    Platonic solids (T/O/I) have one fixed expected chain count each (12,
    24, 60 — see _PLATONIC_REFERENCE_CHAIN_COUNTS), fixed by their
    rotational group order. A Dihedral D_n cage has 2n chains for
    whichever n is its principal axis order; since this module can only
    ever detect orders in ALLOWED_ORDERS (2-5), only D_n for n in
    ALLOWED_ORDERS is offered as a candidate here — D_n's own rotational
    symmetry contains its principal Cn axis plus n perpendicular C2 axes
    (for n >= 3; D2 has three mutually perpendicular C2 axes and no
    higher single principal axis), which is why allowed orders are {2, n}
    (or just {2} when n == 2).

    More than one label can come back for the same n_chains (e.g. 12
    chains matches both T and a hypothetical D6 whose only detectable
    order within ALLOWED_ORDERS is 2) — infer_point_group breaks that tie
    using the actual observed axis counts.
    """
    candidates: Dict[str, Tuple[int, ...]] = {}

    for label, expected_chains in _PLATONIC_REFERENCE_CHAIN_COUNTS.items():
        if n_chains == expected_chains:
            candidates[label] = _PLATONIC_ALLOWED_ORDERS[label]

    for n in ALLOWED_ORDERS:
        if n_chains == 2 * n:
            allowed = (2,) if n == 2 else tuple(sorted({2, n}))
            candidates[f"D{n}"] = allowed

    return candidates


def _score_point_group_fit(
    axis_counts: Dict[int, int], allowed_orders: Tuple[int, ...], n_chains: int
) -> float:
    """
    Scores how well one candidate point group explains the OBSERVED,
    already-disjoint-resolved axis counts (see select_disjoint_axes) for
    one identity group.

    coverage = mean, over this candidate's own allowed_orders, of
    min(found[k] / (n_chains // k), 1.0) — how completely each order this
    point group IS supposed to have was actually detected (capped at 1.0
    so over-detection can't inflate the score past a perfect match).

    A raw coverage score alone can't distinguish a point group from one
    whose allowed orders are a SUBSET of a larger point group's (e.g. T's
    {2, 3} is a subset of O's {2, 3, 4}) — a genuinely octahedral
    12-chain-per-face-class... actually a genuinely octahedral assembly's
    C2/C3 counts could equally "complete" a Tetrahedral reading if T were
    offered as a candidate at the same chain count, despite T being
    unable to have the C4 axes that were also found. So any axis order
    found in the data that ISN'T one of this candidate's allowed_orders
    (a "foreign" order) applies a soft multiplicative penalty,
    1 / (1 + foreign_count), rather than a hard disqualification (an
    isolated spurious foreign axis from noisy detection shouldn't zero
    out an otherwise-good candidate outright, but a point group that
    fails to explain part of the real data should still rank below one
    that explains all of it).
    """
    if not allowed_orders:
        return 0.0

    per_order_ratio = []
    for k in allowed_orders:
        expected_max = n_chains // k
        found = axis_counts.get(k, 0)
        per_order_ratio.append(min(found / expected_max, 1.0) if expected_max else 0.0)
    coverage = float(np.mean(per_order_ratio))

    foreign_count = sum(count for k, count in axis_counts.items() if k not in allowed_orders)
    penalty = 1.0 / (1.0 + foreign_count)

    return coverage * penalty


def infer_point_group(
    n_chains: int, axis_counts: Dict[int, int]
) -> Tuple[str, Tuple[int, ...]]:
    """
    Infers the best-fit point group label for one sequence-identity
    group, from its total chain count and its OBSERVED, already
    disjoint-resolved axis counts per order (see select_disjoint_axes) —
    the combinatorial GLOBAL POINT GROUP inference step (Algorithmic
    Requirement #3).

    Candidates are restricted to labels whose theoretical chain count
    matches n_chains exactly (see _reference_point_groups) — chain count
    is a hard combinatorial fact (a point group of rotational order m
    needs exactly m copies of a subunit occupying one orbit position
    each), not a fuzzy signal, so a chain count that doesn't match ANY
    reference means this module cannot confidently name a point group at
    all. In that case, ("Unknown", ALLOWED_ORDERS) is returned — every
    order stays permitted, since an unrecognized chain count is not
    grounds to prune real geometric findings.

    When more than one reference label matches n_chains (e.g. 12 chains
    matching both T and D6), each is scored by _score_point_group_fit
    against the observed axis_counts, and the highest-scoring label wins.

    Returns (point_group_label, allowed_orders_for_that_group) — the
    latter is exactly what apply_point_group_guardrail needs to prune
    disallowed orders.
    """
    candidates = _reference_point_groups(n_chains)
    if not candidates:
        return "Unknown", ALLOWED_ORDERS

    if len(candidates) == 1:
        (label, allowed_orders), = candidates.items()
        return label, allowed_orders

    scored = [
        (label, allowed_orders, _score_point_group_fit(axis_counts, allowed_orders, n_chains))
        for label, allowed_orders in candidates.items()
    ]
    scored.sort(key=lambda t: t[2], reverse=True)
    best_label, best_allowed_orders, _ = scored[0]
    return best_label, best_allowed_orders


def apply_point_group_guardrail(
    axes_by_order: Dict[int, List[Dict[str, Any]]], allowed_orders: Tuple[int, ...]
) -> Dict[int, List[Dict[str, Any]]]:
    """
    IMPOSSIBILITY GUARDRAIL: given this identity group's disjoint-
    resolved candidates per order (axes_by_order — the output of running
    select_disjoint_axes per order) and the orders the inferred point
    group is actually allowed to have (from infer_point_group), drops
    EVERY candidate of any order not in allowed_orders — entirely, not
    partially — regardless of how tight or individually valid those
    candidates looked. A Tetrahedral-inferred group, for instance, cannot
    have a real C4 or C5 axis by group theory (see
    _PLATONIC_ALLOWED_ORDERS), so any C4/C5 rings found for it are
    necessarily spurious (coincidental geometry, not a genuine symmetry
    element) and are pruned here before being reported.

    Note that the MAXIMUM count per allowed order is never separately
    enforced here — select_disjoint_axes' exclusivity already guarantees
    at most n_chains // k disjoint rings of order k can ever be accepted
    in the first place (see module docstring's POINT-GROUP INFERENCE
    section), so there is nothing left for this function to additionally
    cap; its only job is which orders exist AT ALL for this point group.
    """
    return {
        order: candidates
        for order, candidates in axes_by_order.items()
        if order in allowed_orders
    }


# ============================================================================
# 7. Per-identity-group orchestration
# ============================================================================

def detect_symmetry_groupings(
    chain_names: Sequence[str],
    chain_geometry: Dict[str, dict],
    orders: Sequence[int] = ALLOWED_ORDERS,
    tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> Tuple[str, Dict[int, List[Dict[str, Any]]]]:
    """
    Full per-identity-group pipeline, over ONE sequence-identity group of
    chains (see group_chains_by_identity):

      1. Raw detection — find_c2_candidates (order 2) and
         find_cyclic_ring_candidates (orders 3/4/5, per `orders`),
         independently, each already filtered by the STEP TOLERANCE
         FILTER but NOT yet exclusivity-resolved.
      2. Per-order exclusivity — select_disjoint_axes, run separately
         for each order, so a chain claimed by one C3 ring cannot also
         appear in a different C3 ring, while remaining completely free
         to also appear in a C2 and/or C4 ring (rule 1: multi-symmetry
         overlap is fine ACROSS orders, never WITHIN one).
      3. Point-group inference — infer_point_group, from this group's
         chain count and the disjoint-resolved axis counts just produced.
      4. Impossibility guardrail — apply_point_group_guardrail prunes any
         order the inferred point group cannot physically have.

    Returns (point_group_label, axes_by_order) where axes_by_order maps
    order -> list of accepted (disjoint, guardrail-surviving) candidate
    dicts for that order (an order with no surviving candidates is
    simply absent from the dict, not present with an empty list, so
    callers can use `order in axes_by_order` directly).
    """
    invalid = sorted(set(orders) - set(ALLOWED_ORDERS))
    if invalid:
        raise ValueError(f"orders must be a subset of {ALLOWED_ORDERS} — got invalid {invalid}")

    raw_by_order: Dict[int, List[Dict[str, Any]]] = {}
    if 2 in orders:
        raw_by_order[2] = find_c2_candidates(
            chain_names, chain_geometry, tolerance=tolerance,
            relative_tolerance=relative_tolerance, max_candidate_distance=max_candidate_distance,
        )
    for k in orders:
        if k in CYCLIC_ORDERS:
            raw_by_order[k] = find_cyclic_ring_candidates(
                chain_names, chain_geometry, order=k, tolerance=tolerance,
                relative_tolerance=relative_tolerance, max_candidate_distance=max_candidate_distance,
            )

    # Resolve exclusivity independently per order BEFORE inference, since
    # inference/guardrail decisions should be based on genuine, disjoint
    # axis counts (what will actually be reported), not on raw,
    # potentially-overlapping candidate counts.
    disjoint_by_order = {
        order: select_disjoint_axes(candidates)
        for order, candidates in raw_by_order.items()
    }

    axis_counts = {order: len(candidates) for order, candidates in disjoint_by_order.items()}
    point_group, allowed_orders = infer_point_group(len(chain_names), axis_counts)

    final_axes_by_order = apply_point_group_guardrail(disjoint_by_order, allowed_orders)
    # Drop empty entries so `order in axes_by_order` is a clean presence check.
    final_axes_by_order = {k: v for k, v in final_axes_by_order.items() if v}

    return point_group, final_axes_by_order


# ============================================================================
# 8. Output aggregation — ONE row per symmetry order per structure
# ============================================================================

_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "assembly_id", "point_group", "symmetry_type", "main_chains",
    "main_mean_distance", "main_std_distance", "total_axis_count",
    "equivalent_axis_groupings",
)


def _rows_from_group_results(
    assembly_id: str, point_group: str, axes_by_order: Dict[int, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Converts one identity group's detect_symmetry_groupings output into
    output rows — one per order present. Kept separate from
    analyze_assembly_symmetry so multi-group pooling (see that function)
    has plain per-group row lists to merge, rather than needing to
    re-derive rows after pooling.

    Candidates within an order arrive from select_disjoint_axes already
    sorted tightest-first (ascending CV), so accepted[0] IS the Main
    Grouping (lowest CV) and accepted[1:] are the Equivalent Axis
    Groupings, with no re-sorting needed here.
    """
    rows = []
    for order, accepted in axes_by_order.items():
        main = accepted[0]
        rows.append({
            "assembly_id": assembly_id,
            "point_group": point_group,
            "symmetry_type": f"C{order}",
            "main_chains": main["chains"],
            "main_mean_distance": main["mean_distance"],
            "main_std_distance": main["std_distance"],
            "total_axis_count": len(accepted),
            "equivalent_axis_groupings": [c["chains"] for c in accepted[1:]],
            "_cv": main["cv"] if main["cv"] is not None else float("inf"),  # merge key only
        })
    return rows


def _merge_rows_across_groups(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pools per-identity-group rows (see _rows_from_group_results) down to
    exactly ONE row per symmetry_type per assembly (Output Aggregation
    requirement #4), for the (typical, but not universal) case where a
    structure has more than one sequence-identity group and more than one
    group produced a row for the same order.

    Because different identity groups' chains are, by construction,
    disjoint from each other (group_chains_by_identity partitions all
    chains), rings from different groups never share a chain — pooling
    them for the same order is always safe with no additional exclusivity
    check needed here.

    Merge rule: for a given symmetry_type, the row whose "main" candidate
    has the globally lowest CV becomes the pooled Main Grouping; every
    other group's main AND every other group's own equivalent groupings
    are folded into one combined Equivalent Axis Groupings list; total_
    axis_count sums across groups. The reported point_group is whichever
    group the winning Main Grouping came from — if groups disagree on
    point group, this is noted in the merged row's point_group string
    rather than silently picked, so a genuine multi-component,
    multi-point-group cage isn't misreported as uniform.
    """
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_type.setdefault(row["symmetry_type"], []).append(row)

    merged_rows = []
    for symmetry_type, rows in by_type.items():
        rows_sorted = sorted(rows, key=lambda r: r["_cv"])
        winner = rows_sorted[0]

        other_point_groups = {r["point_group"] for r in rows_sorted[1:]} - {winner["point_group"]}
        point_group = winner["point_group"]
        if other_point_groups:
            point_group = f"{point_group} (other groups: {', '.join(sorted(other_point_groups))})"

        equivalents = list(winner["equivalent_axis_groupings"])
        for r in rows_sorted[1:]:
            equivalents.append(r["main_chains"])
            equivalents.extend(r["equivalent_axis_groupings"])

        merged_rows.append({
            "assembly_id": winner["assembly_id"],
            "point_group": point_group,
            "symmetry_type": symmetry_type,
            "main_chains": winner["main_chains"],
            "main_mean_distance": winner["main_mean_distance"],
            "main_std_distance": winner["main_std_distance"],
            "total_axis_count": sum(r["total_axis_count"] for r in rows_sorted),
            "equivalent_axis_groupings": equivalents,
        })

    return merged_rows


def analyze_assembly_symmetry(
    filepath: str,
    assembly_id: str,
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    orders: Sequence[int] = ALLOWED_ORDERS,
    tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> pd.DataFrame:
    """
    Full per-assembly symmetry pipeline: load the structure, compute
    per-chain terminal geometry via termini.get_chain_ca_geometry, group
    chains by sequence identity (group_chains_by_identity), then run
    detect_symmetry_groupings independently per identity group (raw
    detection -> per-order exclusivity -> point-group inference ->
    impossibility guardrail), and finally pool every group's rows down to
    exactly one row per symmetry order for this assembly (see module
    docstring's OUTPUT AGGREGATION section and _merge_rows_across_groups).

    Returns a pandas DataFrame with one row per DETECTED SYMMETRY ORDER
    (never one row per individual ring), columns:

      - assembly_id                : as passed in
      - point_group                : inferred label ("T", "O", "I",
                                      "D2".."D5", or "Unknown"); if more
                                      than one identity group contributed
                                      to this order and they disagree, the
                                      other group(s)' labels are appended
                                      in parentheses rather than hidden
      - symmetry_type               : "C2" / "C3" / "C4" / "C5"
      - main_chains                 : the lowest-CV accepted ring/pair for
                                      this order, as a tuple of chain IDs
      - main_mean_distance          : that ring's mean junction distance
                                      (Angstroms)
      - main_std_distance           : that ring's junction distance
                                      standard deviation (Angstroms)
      - total_axis_count            : how many disjoint, guardrail-
                                      surviving axes of this order were
                                      found in total (main + equivalents)
      - equivalent_axis_groupings   : list of chain-ID tuples for every
                                      OTHER accepted axis of this same
                                      order (empty list if main_chains is
                                      the only one found)

    Empty (but correctly-columned) if the structure has fewer than 2
    chains with usable CA geometry, or no order in `orders` survived
    detection + exclusivity + the guardrail for any identity group.
    """
    invalid = sorted(set(orders) - set(ALLOWED_ORDERS))
    if invalid:
        raise ValueError(f"orders must be a subset of {ALLOWED_ORDERS} — got invalid {invalid}")

    st = gemmi.read_structure(filepath)
    model = st[0]

    chain_geometry: Dict[str, dict] = {}
    for chain in model:
        geometry = get_chain_ca_geometry(chain)
        if geometry is not None:
            chain_geometry[chain.name] = geometry

    identity_groups = group_chains_by_identity(model, identity_threshold=identity_threshold)

    all_rows: List[Dict[str, Any]] = []
    for group in identity_groups:
        usable = [name for name in group if name in chain_geometry]
        if len(usable) < 2:
            continue

        point_group, axes_by_order = detect_symmetry_groupings(
            usable, chain_geometry, orders=orders, tolerance=tolerance,
            relative_tolerance=relative_tolerance, max_candidate_distance=max_candidate_distance,
        )
        all_rows.extend(_rows_from_group_results(assembly_id, point_group, axes_by_order))

    merged_rows = _merge_rows_across_groups(all_rows)

    result_df = pd.DataFrame(merged_rows, columns=list(_OUTPUT_COLUMNS))
    if not result_df.empty:
        result_df = result_df.sort_values("symmetry_type").reset_index(drop=True)
    return result_df


def run_symmetry_analysis(
    df: pd.DataFrame,
    filepath_column: str = "filepath",
    assembly_id_column: str = "assembly_id",
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    orders: Sequence[int] = ALLOWED_ORDERS,
    tolerance: Optional[float] = DEFAULT_TOLERANCE,
    relative_tolerance: Optional[float] = DEFAULT_RELATIVE_TOLERANCE,
    max_candidate_distance: Optional[float] = DEFAULT_MAX_CANDIDATE_DISTANCE,
) -> pd.DataFrame:
    """
    Runs analyze_assembly_symmetry over every row of a downloaded-
    candidates DataFrame (i.e. what download.py's download_candidates()
    returns) and concatenates the results into one DataFrame — up to 4
    rows (one per detected order) per assembly in the batch.

    assembly_id_column : if this column exists in df, it's used
        directly; otherwise falls back to f"{entry_id}-{assembly_num}",
        matching distance.py's convention for the same situation.

    Failures on individual structures are caught and reported by
    assembly_id rather than stopping the whole batch — one malformed or
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
                identity_threshold=identity_threshold, orders=orders,
                tolerance=tolerance, relative_tolerance=relative_tolerance,
                max_candidate_distance=max_candidate_distance,
            ))
        except Exception as e:
            print(f"Failed: {assembly_id} — {e}")

    if not frames:
        return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))

    result_df = pd.concat(frames, ignore_index=True)
    if not result_df.empty:
        result_df = result_df.sort_values(["assembly_id", "symmetry_type"]).reset_index(drop=True)
    return result_df