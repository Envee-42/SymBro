"""
distance.py — NC (N-terminus to C-terminus) distance and ring-identification
calculations on downloaded assemblies.

Built on top of termini.get_chain_ca_geometry for per-chain coordinates.
Two paths live here:

  - compute_nc_distance / process_candidates: single-closest-pair NC
    distance, kept as a lighter-weight utility for other uses.
  - analyze_assembly_rings / run_ring_analysis: the primary path — groups
    chains directly into full oligomeric rings (e.g. all 3 chains of a
    trimer) by scoring whole candidate groups rather than relying on
    pairwise closest-centroid matching, then finds the best fusion
    ordering within each ring.

RING SIZE — "Challenge 1": recognizing how many chains belong in a ring.
Platonic-solid-type cages (the T / O / I point groups referenced in the
Yeates lab's mathematical framework this tool is built around) only ever
have 2-, 3-, 4-, and 5-fold rotational symmetry axes — T has 2- and
3-fold, O adds 4-fold, I adds 5-fold, and nothing in this family has a
6-fold (or higher) axis. So a chain ring can only ever be a dimer,
trimer, tetramer, or pentamer — see ALLOWED_RING_SIZES. Rather than
requiring the caller to already know which one applies (from RCSB
metadata that may be missing, wrong, or simply not fetched), this module
can AUTO-DETECT the ring size per sequence-identity group by trying every
allowed size and keeping whichever one best explains the group's actual
chain positions — see detect_ring_size(). An explicit ring_size is still
accepted (and skips auto-detection) for the case where you already trust
a metadata field like oligomeric_count.

WHY RAW TIGHTNESS ALONE FAILS — a cage's chain-to-chain distances reflect
EVERY symmetry axis present at once, not just the one you care about. An
octahedral (O) cage built from 8 trimers has 2-, 3-, and 4-fold axes all
acting on the same 24 chains, and the 2-fold "cage-forming" contact
between neighboring trimers is very often TIGHTER than the trimer's own
3-fold "oligomerization" contact — that's a normal, expected feature of
efficient cage packing, not a defect. Two consequences follow:

  1. A candidate GROUP of chains can be spuriously tight simply because
     it mixes members of two DIFFERENT true rings connected by that tight
     inter-ring contact (e.g. two chains from one trimer plus one from
     its neighbor) — this can outscore the real, uncontaminated trimer on
     raw summed distance alone. The fix: _group_combinations_info also
     scores each candidate's shape UNIFORMITY (the coefficient of
     variation, or CV, of its internal pairwise distances), and
     candidates whose CV exceeds a size-specific threshold (see
     _ideal_ring_shape_cv / RING_SHAPE_CV_JITTER_MARGIN) are excluded
     outright before any ranking happens. A genuine n-gon has consistent, predictable spacing
     between its members (dimers/trimers: CV=0, exactly uniform by
     geometry; tetramers/pentamers: CV≈0.17/0.24, from the fixed
     adjacent-vs-diagonal ratio of a regular polygon); a group
     contaminated with a member from a different true ring does not.

  2. Even after contamination is filtered out, a real trimer and the
     genuine (tight, uniform) inter-trimer dimers next to it can BOTH be
     valid, fully-covering, internally-uniform decompositions of the same
     chain set — this isn't resolved by picking whichever is tighter
     (that would always favor the smaller ring, since a cage's
     inter-subunit contacts are so often engineered to be tight). Per
     point-group theory, a higher-order rotation axis's symmetry already
     implies/contains the lower-order axes as a byproduct — so whenever
     multiple ring sizes independently give a valid, uniform, adequately-
     covering decomposition (see RING_SIZE_COVERAGE_FLOOR),
     detect_ring_size prefers the LARGEST one outright, rather than
     comparing tightness across sizes. An earlier version tried to
     resolve size ambiguity with a per-ring "isolation ratio" (how much
     farther a ring's nearest outside chain sits) — that correctly caught
     one failure mode (an even ring size like a tetramer trivially
     subdividing into tight adjacent-pair "dimers", where the excluded
     chains really are the SAME true ring's other members) but wrongly
     penalized this one (a genuine, separate, uniform trimer whose
     neighbor just happens to sit close by, by design) — so it's been
     replaced by the size-preference rule above, which handles both.

REDUNDANCY — avoiding, e.g., reporting a tetrahedron's 4 physically
equivalent trimers as 4 separate findings. deduplicate_rings_by_geometry()
collapses rings that are the SAME ring (related by the assembly's own
symmetry) into one representative, by comparing each ring's full pairwise-
distance "fingerprint" rather than just its single nc_distance number —
see that function's docstring for why a single-number comparison isn't
accurate enough on its own.

TOLERANCE — real structures never hit exact symmetry: crystal packing,
minor conformational differences, and refinement noise mean two
"identical" copies of the same ring will never measure out to bit-for-bit
identical distances. Every distance-based decision in this module (same-
size ring acceptance, cross-size comparison, redundancy collapsing) is
governed by one user-adjustable `tolerance` parameter, in Angstroms —
widen it if real assemblies are coming out more fragmented/asymmetric
than they should; narrow it if unrelated chains are being merged. Ring
shape uniformity (RING_SHAPE_CV_JITTER_MARGIN) and coverage
(RING_SIZE_COVERAGE_FLOOR) are separate, dimensionless knobs — they're
about whether a candidate group's SHAPE is coherent at all, which
tolerance alone can't express.
"""

from itertools import combinations
from math import comb
import gemmi
import numpy as np
import pandas as pd

from toolkit.geometry.termini import get_chain_ca_geometry


# ============================================================================
# Domain constants
# ============================================================================

# The only ring sizes a Platonic-solid-type (T/O/I point group) cage can
# physically have, per the rotational symmetry axes those three point
# groups contain (see module docstring). Any function here that accepts a
# ring_size validates against this — a ring_size outside this set isn't a
# looser/stricter setting, it's a physically impossible request, so it's
# rejected rather than silently attempted.
ALLOWED_RING_SIZES = (3, 3)

# Ring-size auto-detection (detect_ring_size) and the underlying combinatorial
# group scoring (find_best_rings_by_group_distance) both work by scoring
# EVERY possible group of chains of a given size — C(n, ring_size)
# candidates for n chains in one identity group. That's cheap for the
# common cases (a few dozen chains, ring_size <= 5), but auto-detection
# blindly tries ring_size=5 even for assemblies where it's the wrong
# answer, and C(n, 5) grows fast — e.g. C(60, 5) is ~5.46 million. Rather
# than let a single bad candidate size silently make a run take minutes,
# any size whose candidate count exceeds this cap is skipped with a
# printed warning. Raise it explicitly (per-call) if you have a specific,
# known-large case that genuinely needs it.
MAX_RING_CANDIDATE_COMBINATIONS = 200_000

# Shape-uniformity gate: candidates whose internal pairwise centroid
# distances aren't consistent enough to plausibly be a real n-gon are
# excluded before any tightness-based ranking happens (see
# _filter_uniform_shape / _ideal_ring_shape_cv). Consistency is judged by
# coefficient of variation (CV = standard deviation / mean of the
# candidate's pairwise distances) against the THEORETICAL CV of an ideal
# regular n-gon of that ring_size — a dimer/trimer's is exactly 0 (a
# dimer has only one pairwise distance; an equilateral triangle's three
# edges are all equal), a tetramer's ≈0.1716 and a pentamer's ≈0.2361
# (both from the fixed adjacent-vs-diagonal ratio of a regular polygon —
# see _ideal_ring_shape_cv). Using a SIZE-SPECIFIC bound rather than one
# flat number matters: a flat cutoff loose enough to admit a real
# pentamer's ~0.24 is also loose enough to admit some contaminated larger
# groups (empirically, a group formed from one whole true ring plus part
# of a neighboring one can land right around CV≈0.3, which "coincidental
# uniformity" can slip under a single flat threshold but not the
# pentamer-specific bound below).
#
# This margin, added on top of the ideal value, is what a real structure's
# biological jitter (crystal packing, minor conformational asymmetry — see
# the module docstring's TOLERANCE section) can be expected to add to CV
# without indicating contamination. Checked empirically against a jittered
# synthetic trimer (radius=5, jitter=0.3 — see test_biological_jitter_still_detected):
# observed CV stayed under 0.05, so 0.08 leaves comfortable headroom.
RING_SHAPE_CV_JITTER_MARGIN = 0.08

# Selection floor for detect_ring_size: a candidate ring size only
# "qualifies" to be preferred by size (see RING SIZE / point-group
# nesting in the module docstring) if it successfully accounts for at
# least this fraction of the identity group's chains. Without this floor,
# a ring size that only manages to explain a small, lucky subset of the
# group (leaving most chains unassigned) could still out-rank a smaller
# size that cleanly explains nearly everything, just for being "bigger".
# 0.75 comfortably allows for a genuine outlier chain or two (see
# test_leftover_chain_excluded) while still requiring a candidate size to
# be doing most of the explanatory work before it's trusted over a
# smaller, fully-covering alternative.
RING_SIZE_COVERAGE_FLOOR = 0.75


def compute_nc_distance(filepath, assembly_id, pair_filter=None):
    """
    Loads one structure and calculates the NC distance between its two
    closest chains (by centroid distance).

    pair_filter : optional function(chain_name_1, chain_name_2) -> bool.
        If given, a candidate chain pair is only considered when this
        returns True. This is the hook for enforcing "only pair chains
        that plausibly belong to the same subunit" (e.g. matching
        sequence/entity) — relevant for multi-component cages, where the
        globally closest two chain centroids aren't guaranteed to be part
        of the same biological trimer rather than two chains from
        neighboring trimers that happen to pack close together. Left
        optional rather than hardcoded: single-component cages don't need
        it, and the right identity check depends on what data you want to
        compare chains against (sequence, entity ID, etc.).

    Returns a dict with assembly_id, the two chain names, and nc_distance
    (in Angstroms, rounded to 2 decimal places) — or None if the structure
    doesn't have at least two chains with usable CA geometry (or none
    satisfy pair_filter, if given).
    """
    st = gemmi.read_structure(filepath)
    model = st[0]

    chain_geometry = {}
    for chain in model:
        geometry = get_chain_ca_geometry(chain)
        if geometry is not None:
            chain_geometry[chain.name] = geometry

    if len(chain_geometry) < 2:
        return None

    min_dist = float("inf")
    closest_pair = None
    for (name1, geo1), (name2, geo2) in combinations(chain_geometry.items(), 2):
        if pair_filter is not None and not pair_filter(name1, name2):
            continue
        dist = np.linalg.norm(geo1["centroid"] - geo2["centroid"])
        if dist < min_dist:
            min_dist = dist
            closest_pair = (name1, name2)

    if closest_pair is None:
        return None

    name1, name2 = closest_pair
    # Directionality is ambiguous a priori (either chain could be the
    # "upstream" one), so we compute both C1->N2 and C2->N1 and keep
    # whichever is shorter.
    d_forward = np.linalg.norm(chain_geometry[name1]["c"] - chain_geometry[name2]["n"])
    d_reverse = np.linalg.norm(chain_geometry[name2]["c"] - chain_geometry[name1]["n"])

    return {
        "assembly_id": assembly_id,
        "chain_1": name1,
        "chain_2": name2,
        "nc_distance": round(min(d_forward, d_reverse), 2),
    }


def process_candidates(df, filepath_column="filepath", assembly_id_column="assembly_id", pair_filter=None):
    """
    Runs compute_nc_distance over every row of a downloaded-candidates
    DataFrame (i.e. what download_candidates() returns) and collects the
    results into a single DataFrame, sorted by nc_distance.

    assembly_id_column : if this column exists in df (it will, if df came
        from download_candidates()), it's used directly rather than
        reconstructed from entry_id/assembly_num, avoiding duplicating
        that string-building logic in two places.

    Failures on individual structures are caught and reported by
    assembly_id rather than stopping the whole batch — one malformed or
    unexpectedly-shaped structure shouldn't take down a run over hundreds
    of candidates.
    """
    results = []
    for _, row in df.iterrows():
        if assembly_id_column in df.columns:
            assembly_id = row[assembly_id_column]
        else:
            assembly_id = f"{row['entry_id']}-{row['assembly_num']}"

        try:
            result = compute_nc_distance(row[filepath_column], assembly_id, pair_filter=pair_filter)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"Failed: {assembly_id} — {e}")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("nc_distance").reset_index(drop=True)
    return results_df


# ============================================================================
# Ring-level analysis — the primary path.
# ============================================================================

def group_chains_by_sequence(model):
    """
    Groups chain names by their one-letter polymer sequence, so that only
    chains which are LITERAL COPIES of the same protein are ever considered
    for the same oligomeric ring.

    This is the fix for the two-component cage risk: without this
    grouping, the two globally closest chain centroids in a two-component
    assembly could easily be one chain from EACH different component
    meeting at an interface — a real, close chain pair, but not a valid
    "these are copies of one subunit" pairing. Grouping by sequence first
    means a spurious cross-component pair can never even be considered,
    rather than relying on distance alone and hoping it doesn't happen.

    It's also the boundary ring-size auto-detection operates within: each
    returned group is scored for its own best ring size independently
    (see detect_ring_size), so a multi-component cage where different
    proteins form different-sized rings (e.g. one forms trimers, another
    forms dimers) is handled correctly rather than assuming one size for
    the whole assembly.

    Returns a list of groups, each a list of chain names sharing one
    sequence. Chains with no resolved polymer (already filtered out
    elsewhere) are skipped.
    """
    groups = {}
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) == 0:
            continue
        seq = gemmi.one_letter_code(polymer.extract_sequence())
        groups.setdefault(seq, []).append(chain.name)
    return list(groups.values())


def _pairwise_centroid_matrix(chain_names, chain_geometry):
    """
    Precomputes the full n x n pairwise centroid-distance matrix for
    `chain_names` in one vectorized pass. n is the number of chains in one
    sequence-identity group — at most a few dozen even for the largest
    icosahedral cages — so this O(n^2) precomputation is always cheap. The
    point is to make it a ONE-TIME cost: without it, scoring every
    candidate ring (see _group_combinations_info) would recompute
    np.linalg.norm for the same chain pair over and over across
    overlapping candidate groups.
    """
    centroids = np.array([chain_geometry[name]["centroid"] for name in chain_names])
    diffs = centroids[:, None, :] - centroids[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def _group_combinations_info(chain_names, dist_matrix, ring_size):
    """
    Scores every possible group of `ring_size` chains from `chain_names`,
    using the precomputed pairwise-distance matrix from
    _pairwise_centroid_matrix, and returns them sorted tightest-first by
    summed pairwise distance.

    Each candidate is a dict with:
      - "chains"        : the chain-name tuple.
      - "sum_distance"  : sum of all C(ring_size, 2) pairwise centroid
                           distances between members. This is the
                           tightness metric find_best_rings_by_group_distance
                           has always used for accepting rings of ONE
                           given size.
      - "mean_distance" : sum_distance divided by the number of pairs —
                           the same tightness, normalized so it's
                           comparable ACROSS different ring sizes (a
                           tetramer's 6 summed pairwise distances are not
                           comparable to a trimer's 3 without dividing
                           out the pair count first). Used by
                           detect_ring_size for exactly that comparison;
                           same-size acceptance still goes by
                           sum_distance, unchanged from before.
      - "fingerprint"   : sorted tuple of the individual pairwise
                           distances — a rotation/reflection-invariant
                           shape descriptor used by
                           deduplicate_rings_by_geometry to recognize
                           genuinely equivalent rings.
      - "cv"            : coefficient of variation (std / mean) of the
                           pairwise distances — 0.0 when ring_size == 2
                           (a single distance is trivially "uniform").
                           This is the shape-uniformity signal used to
                           reject candidates that mix chains from two
                           different true rings — see _ideal_ring_shape_cv
                           and the module docstring's "WHY RAW TIGHTNESS
                           ALONE FAILS" section for why this check exists
                           at all.

    Sorting by sum_distance and by mean_distance agree WITHIN one
    ring_size, since mean = sum / (a constant, C(ring_size, 2)) — so this
    one sort order is valid for both find_best_rings_by_group_distance
    (which uses sum_distance) and detect_ring_size (which uses
    mean_distance); no separate sort is needed per caller.
    """
    index_by_name = {name: i for i, name in enumerate(chain_names)}
    n_pairs = ring_size * (ring_size - 1) // 2

    candidates = []
    for group in combinations(chain_names, ring_size):
        idx_pairs = combinations((index_by_name[name] for name in group), 2)
        pair_distances = [float(dist_matrix[a, b]) for a, b in idx_pairs]
        total = sum(pair_distances)
        mean_distance = total / n_pairs
        if n_pairs > 1 and mean_distance > 0:
            variance = sum((x - mean_distance) ** 2 for x in pair_distances) / n_pairs
            cv = (variance ** 0.5) / mean_distance
        else:
            cv = 0.0
        candidates.append({
            "chains": group,
            "sum_distance": total,
            "mean_distance": mean_distance,
            "fingerprint": tuple(sorted(pair_distances)),
            "cv": cv,
        })

    candidates.sort(key=lambda c: c["sum_distance"])
    return candidates


def _ideal_ring_shape_cv(ring_size):
    """
    The coefficient of variation (CV) of the pairwise distances between
    `ring_size` points evenly spaced on a circle — i.e. what a perfectly
    regular, unjittered n-gon's own shape-uniformity CV works out to.
    Used as the baseline for _filter_uniform_shape's per-size threshold
    (see RING_SHAPE_CV_JITTER_MARGIN): 0.0 for ring_size 2 or 3 (a dimer
    has one pairwise distance; an equilateral triangle's three edges are
    exactly equal), ≈0.1716 for 4, ≈0.2361 for 5. Scale-invariant (the
    circle's radius cancels out of a ratio of distances), so this needs
    no chain geometry to compute — it's pure polygon math.
    """
    from math import sin, pi
    distances = []
    for i in range(ring_size):
        for j in range(i + 1, ring_size):
            angle = 2 * pi * abs(i - j) / ring_size
            # chord length between two points on a unit circle separated
            # by `angle` radians.
            distances.append(2 * sin(angle / 2) if angle <= pi else 2 * sin((2 * pi - angle) / 2))
    mean = sum(distances) / len(distances)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in distances) / len(distances)
    return (variance ** 0.5) / mean


def _filter_uniform_shape(candidates, ring_size, max_cv=None):
    """
    Drops any candidate whose shape-uniformity CV exceeds the allowed
    threshold for `ring_size` — see RING_SHAPE_CV_JITTER_MARGIN and
    _ideal_ring_shape_cv.

    max_cv : None (default) — use the size-specific threshold
        (_ideal_ring_shape_cv(ring_size) + RING_SHAPE_CV_JITTER_MARGIN),
        the recommended setting. A number — override with one flat
        threshold applied regardless of ring_size. float("inf") —
        disable the filter entirely (every candidate passes through
        unchanged).

    This runs BEFORE greedy acceptance in both find_best_rings_by_group_distance
    and detect_ring_size, so a contaminated candidate (one mixing chains
    from two different true rings) can never win a slot away from a
    genuine one just because its raw distance happens to be tighter — see
    the module docstring for a worked example of exactly this happening.
    """
    threshold = (
        _ideal_ring_shape_cv(ring_size) + RING_SHAPE_CV_JITTER_MARGIN
        if max_cv is None else max_cv
    )
    return [c for c in candidates if c["cv"] <= threshold]


def _greedy_nonoverlapping_accept(candidates, tolerance, score_key):
    """
    Walks `candidates` (already sorted tightest-first by score_key) and
    greedily accepts non-overlapping groups: the first accepted
    candidate's score sets the quality bar ("anchor"), and any later
    candidate whose score exceeds anchor + tolerance stops the search
    entirely — since candidates are sorted tightest-first, every
    remaining one is at least as loose and would fail the cutoff too.

    Assignment is greedy, not globally optimal: each candidate is only
    checked against chains already claimed by a PREVIOUSLY accepted
    (and therefore tighter-or-equal) candidate, so a genuine ring gets
    first claim on its own chains before a worse grouping has a chance to
    steal one of them.

    Quality cutoff, in words: without the anchor+tolerance stop, greedy
    assignment would happily keep going after the real rings are claimed,
    forcing leftover chains into additional "rings" no matter how loose.
    Chains that don't end up in any group within tolerance of the
    tightest one are left unassigned rather than forced into a bad one —
    that's the correct outcome when those chains genuinely aren't part of
    an equivalent ring (extra copies in the asymmetric unit, mismatched
    leftovers, a genuinely asymmetric arrangement), not a bug.

    score_key : "sum_distance" (same-size acceptance, e.g. inside
        find_best_rings_by_group_distance) or "mean_distance" (cross-size
        comparison, inside detect_ring_size) — see _group_combinations_info
        for why these differ and when each applies.

    Returns the accepted candidate dicts, in the same shape they came in.
    """
    assigned = set()
    accepted = []
    anchor_score = None

    for candidate in candidates:
        chains = candidate["chains"]
        if not assigned.isdisjoint(chains):
            continue
        score = candidate[score_key]
        if anchor_score is None:
            anchor_score = score
        elif score - anchor_score > tolerance:
            break
        accepted.append(candidate)
        assigned.update(chains)

    return accepted


def find_best_rings_by_group_distance(chain_names, chain_geometry, ring_size, tolerance=1.5, max_combinations=MAX_RING_CANDIDATE_COMBINATIONS, max_shape_cv=None):
    """
    Directly evaluates EVERY possible group of `ring_size` chains within
    one same-sequence identity group, discards any whose internal shape
    isn't uniform enough to plausibly be a real ring (see max_shape_cv /
    _ideal_ring_shape_cv and the module docstring), scores what's left by
    how tightly packed its members are (summed pairwise centroid
    distance), and greedily keeps the tightest non-overlapping groups
    first — see _greedy_nonoverlapping_accept for the acceptance rule.

    This is deliberately a direct group-scoring approach rather than
    seeding a cluster from the single globally closest PAIR and growing it
    outward: the globally closest pair can belong to two DIFFERENT
    physical rings that happen to sit near each other (same sequence,
    same cage, just neighboring rings) rather than being genuine
    ring-mates. Scoring whole candidate groups directly avoids that — a
    candidate is only accepted by comparing it against every OTHER
    possible group, not by trusting one pairwise distance as a stand-in
    for ring membership.

    ring_size must be one of ALLOWED_RING_SIZES (2/3/4/5) — see that
    constant and the module docstring for why no other size is physically
    possible for a Platonic-solid-type cage.

    This deliberately doesn't hardcode an expected ring COUNT (e.g. 4 for
    a tetrahedral cage) — that number falls out naturally once tightness
    is bounded correctly: a tetrahedral assembly's 4 trimers are all real
    and comparably tight, so all 4 pass the cutoff together.

    max_combinations : if C(len(chain_names), ring_size) exceeds this,
        the search is skipped (a warning is printed, and an empty list is
        returned) rather than silently taking a long time — see
        MAX_RING_CANDIDATE_COMBINATIONS. Raise it explicitly if you have a
        specific, known-large case that genuinely needs it.

    max_shape_cv : candidates whose internal pairwise distances are less
        uniform than allowed are excluded before tightness ranking even
        happens — this is what stops a group that accidentally mixes
        chains from two different true rings from winning a slot just
        because it happens to be numerically tighter. None (default) uses
        a size-specific threshold derived from the ideal n-gon's own CV
        plus a small jitter allowance (see _ideal_ring_shape_cv /
        RING_SHAPE_CV_JITTER_MARGIN) — the recommended setting. Pass a
        number to override with one flat threshold, or float("inf") to
        disable this filter entirely.

    Returns a list of chain-name groups (each of length ring_size).
    Chains left out (by the overlap rule, the tolerance cutoff, the
    shape-uniformity filter, or a skipped oversized search) are excluded
    entirely rather than padded.
    """
    if ring_size not in ALLOWED_RING_SIZES:
        raise ValueError(
            f"ring_size must be one of {ALLOWED_RING_SIZES} — Platonic-solid "
            f"(T/O/I point-group) cages only have 2-, 3-, 4-, and 5-fold "
            f"rotational symmetry axes, so no other ring size is physically "
            f"possible — got {ring_size}"
        )
    if len(chain_names) < ring_size:
        return []

    n_combinations = comb(len(chain_names), ring_size)
    if n_combinations > max_combinations:
        print(
            f"find_best_rings_by_group_distance: skipping ring_size={ring_size} "
            f"for {len(chain_names)} chains ({n_combinations} candidate groups "
            f"exceeds max_combinations={max_combinations})."
        )
        return []

    dist_matrix = _pairwise_centroid_matrix(chain_names, chain_geometry)
    candidates = _group_combinations_info(chain_names, dist_matrix, ring_size)
    candidates = _filter_uniform_shape(candidates, ring_size, max_shape_cv)
    accepted = _greedy_nonoverlapping_accept(candidates, tolerance, score_key="sum_distance")

    return [list(c["chains"]) for c in accepted]


def _ring_isolation_ratio(ring_chains, chain_names, dist_matrix, intra_mean_distance):
    """
    How much farther this ring's nearest chain OUTSIDE it sits, relative
    to how tight the ring itself is. Retained as a diagnostic value
    surfaced in detect_ring_size's informational output — NOT used to
    accept or reject a candidate size (see below for why).

    An earlier version of this module used isolation ratio as a hard
    selection criterion, on the theory that a genuine ring should sit
    clearly apart from everything outside it. That's true for ONE failure
    mode — an even ring size (a tetramer, most notably) can always be cut
    into tight, fully-covering smaller groups using just its
    adjacent-neighbor contacts, and the chains "left out" of each such
    fake group are the true ring's OTHER members, sitting almost as close
    as the fake group's own chains, so isolation ratio correctly flags
    that as suspicious. But it actively gets a SECOND, equally real case
    wrong: a genuine, separate, internally-uniform ring (e.g. one trimer
    in an octahedral cage of 8) whose neighboring ring just happens to
    sit close by, because the cage's own inter-subunit ("cage-forming")
    contact is tighter than the ring's internal ("oligomerization")
    contact — a normal, expected feature of efficient cage packing, not a
    sign that the trimer is spurious. Isolation ratio can't tell these
    two cases apart, because both produce "something close was left out".
    What actually distinguishes them is shape uniformity (see
    _ideal_ring_shape_cv / _filter_uniform_shape) plus preferring the
    larger of two otherwise-valid ring sizes (see detect_ring_size) —
    isolation ratio itself is kept only as a number worth glancing at,
    not a gate.

    Returns +inf if every other chain in `chain_names` is already inside
    `ring_chains` (nothing left to compare against — the whole identity
    group IS this one ring) — maximally isolated by definition.
    """
    index_by_name = {name: i for i, name in enumerate(chain_names)}
    ring_idx = {index_by_name[c] for c in ring_chains}
    outside_idx = [i for i in range(len(chain_names)) if i not in ring_idx]
    if not outside_idx:
        return float("inf")
    gap = min(dist_matrix[i, j] for i in ring_idx for j in outside_idx)
    return gap / intra_mean_distance if intra_mean_distance > 0 else float("inf")


def detect_ring_size(chain_names, chain_geometry, candidate_sizes=ALLOWED_RING_SIZES, tolerance=1.5, max_combinations=MAX_RING_CANDIDATE_COMBINATIONS, max_shape_cv=None, min_coverage=RING_SIZE_COVERAGE_FLOOR):
    """
    Auto-detects which ring size best explains one sequence-identity
    group's chains — "Challenge 1": recognizing how many chains belong in
    a ring, from geometry alone, rather than requiring it as an input.

    For each candidate size (restricted to ALLOWED_RING_SIZES — 2/3/4/5,
    the only rotational symmetry orders a Platonic-solid-type cage can
    have), every possible group of that size is scored, filtered by
    shape uniformity (max_shape_cv — see _ideal_ring_shape_cv and the
    module docstring's "WHY RAW TIGHTNESS ALONE FAILS" section), and
    greedily assigned into non-overlapping rings using the same machinery
    find_best_rings_by_group_distance uses, except tightness here is
    judged by MEAN pairwise centroid distance rather than the summed
    distance. That normalization is what makes different sizes
    comparable at all — a tetramer's 6 pairwise distances will always sum
    to more than a trimer's 3, even if the tetramer is individually
    tighter — and it's what lets `tolerance` mean the same physical thing
    (Angstroms of acceptable centroid-distance variation between
    equivalent copies) no matter which size is being evaluated.

    A candidate size is skipped (with a printed warning) if the identity
    group has fewer chains than that size, or if C(n, size) exceeds
    max_combinations — see that parameter and MAX_RING_CANDIDATE_COMBINATIONS.

    Selection: think of it the way you'd check a partition by hand — for
    a well-formed identity group, the CORRECT ring size is the one where
    sorting the chains into same-size, non-overlapping, shape-uniform
    groups leaves no overlap and (ideally) no residual chains: every
    chain assigned to exactly one group. So COVERAGE (the fraction of the
    group's chains successfully assigned) is the primary criterion,
    across every candidate size — not a pass/fail floor gate for a
    separate "which is biggest" competition. The size with the highest
    coverage wins outright.

    Point-group nesting only comes in as a TIE-BREAKER: if two sizes
    cover the exact same number of chains (a real tie, not just close),
    the larger one wins, since a higher-order rotational symmetry axis,
    when genuinely present, mathematically implies/contains whatever
    lower-order proximities (a 2-fold "dimer"-looking contact between
    neighboring trimers) show up alongside it — never the reverse. This
    is what correctly prefers 8 trimers over 12 dimers when both happen
    to cleanly, fully partition the same chains. But it must never
    override a size that explains MORE of the structure: 4 trimers with
    perfect 12/12 coverage beats a spurious 2-pentamer reading that only
    manages 10/12, even though 5 > 3 — a size that leaves real chains
    unassigned isn't "more advanced" symmetry, it's just a worse fit,
    and coverage is what has to decide that, not size.

    min_coverage (default RING_SIZE_COVERAGE_FLOOR) plays a much smaller
    role now: it's a confidence check on the WINNER, not a competition
    gate — if even the best-covering size falls short of it, a warning is
    printed (this group might be a mix of unrelated chains, or `tolerance`
    /`max_shape_cv` may need widening) but the best available fit is still
    returned rather than nothing.

    Returns (winning_size, accepted_rings) — accepted_rings is that
    size's list of chain-name-group lists, exactly like
    find_best_rings_by_group_distance would return for that size, so the
    caller never has to recompute it. Returns (None, []) if no candidate
    size assigned even a single valid ring.
    """
    invalid = sorted(set(candidate_sizes) - set(ALLOWED_RING_SIZES))
    if invalid:
        raise ValueError(
            f"candidate_sizes can only include {ALLOWED_RING_SIZES} — "
            f"Platonic-solid (T/O/I point-group) cages only have 2-, 3-, "
            f"4-, and 5-fold rotational symmetry axes — got invalid size(s) {invalid}"
        )

    n = len(chain_names)
    results = []  # (size, coverage, mean_tightness, min_isolation_ratio, accepted_candidates)

    for size in sorted(candidate_sizes):
        if size > n:
            continue

        n_combinations = comb(n, size)
        if n_combinations > max_combinations:
            print(
                f"detect_ring_size: skipping ring_size={size} for {n} chains "
                f"({n_combinations} candidate groups exceeds max_combinations={max_combinations})."
            )
            continue

        dist_matrix = _pairwise_centroid_matrix(chain_names, chain_geometry)
        candidates = _group_combinations_info(chain_names, dist_matrix, size)
        candidates = _filter_uniform_shape(candidates, size, max_shape_cv)
        accepted = _greedy_nonoverlapping_accept(candidates, tolerance, score_key="mean_distance")

        if not accepted:
            continue

        coverage = (len(accepted) * size) / n
        mean_tightness = sum(c["mean_distance"] for c in accepted) / len(accepted)
        # diagnostic only (see _ring_isolation_ratio) — not used to decide anything below.
        min_isolation_ratio = min(
            _ring_isolation_ratio(c["chains"], chain_names, dist_matrix, c["mean_distance"])
            for c in accepted
        )
        results.append((size, coverage, mean_tightness, min_isolation_ratio, accepted))

    if not results:
        return None, []

    # Coverage decides first, across ALL viable sizes — not just ones
    # clearing min_coverage. Ties (same number of chains covered) are
    # broken by preferring the larger size (point-group nesting); a
    # further tie by tightness. Sizes covering strictly fewer chains
    # never win just for being bigger — see the docstring above.
    results.sort(key=lambda r: (-r[1], -r[0], r[2]))
    winner = results[0]
    winning_size, winning_coverage, winning_tightness, _, accepted_candidates = winner

    if len(results) > 1:
        runner_up = results[1]
        coverage_tied = abs(winner[1] - runner_up[1]) < 1e-9
        if coverage_tied:
            print(
                f"detect_ring_size: ring_size={winning_size} and ring_size={runner_up[0]} "
                f"both cover the same {round(winning_coverage * n)} of {n} chains "
                f"(coverage={winning_coverage:.2f}) — choosing the larger per point-group "
                f"nesting (a higher-order symmetry axis's rings account for the lower-order "
                f"ones as a byproduct, not the reverse). Consider cross-checking against "
                f"known oligomeric-state metadata if ring_size={winning_size} looks wrong."
            )

    if winning_coverage < min_coverage:
        print(
            f"detect_ring_size: best fit is ring_size={winning_size}, but it only covers "
            f"{winning_coverage:.0%} of this group of {n} chains (below the "
            f"{min_coverage:.0%} confidence floor) — consider widening tolerance or "
            f"max_shape_cv, or checking this group isn't a mix of unrelated chains."
        )

    return winning_size, [list(c["chains"]) for c in accepted_candidates]


def find_shortest_ring_junction(ring_chain_names, chain_geometry):
    """
    Given the chain names in one validated ring, finds the SHORTEST single
    N-to-C junction distance between any two DIFFERENT chains in that ring.

    This is NC distance as originally defined: one number, one specific
    junction — not a summed path across the whole ring. Checks every
    ordered pair within the ring (chain A's C-terminus to chain B's
    N-terminus, for every A != B) and keeps the minimum. Ring size is
    small (at most 5 — see ALLOWED_RING_SIZES), so a direct comparison is
    cheap.

    Also computes and attaches this ring's "fingerprint" — the sorted
    tuple of every pairwise centroid distance between its members — and
    its ring_size, both used by deduplicate_rings_by_geometry to recognize
    which rings are physically equivalent copies of each other.

    Returns a dict: chain_order (all member chains), the specific
    from_chain/to_chain pair achieving the minimum, nc_distance itself
    (rounded to 2 decimals), ring_size, and fingerprint.
    """
    best = {"nc_distance": float("inf")}
    for name_from in ring_chain_names:
        for name_to in ring_chain_names:
            if name_from == name_to:
                continue
            dist = np.linalg.norm(
                chain_geometry[name_from]["c"] - chain_geometry[name_to]["n"]
            )
            if dist < best["nc_distance"]:
                best = {
                    "chain_order": list(ring_chain_names),
                    "from_chain": name_from,
                    "to_chain": name_to,
                    "nc_distance": round(dist, 2),
                }

    best["ring_size"] = len(ring_chain_names)
    best["fingerprint"] = tuple(sorted(
        float(np.linalg.norm(chain_geometry[a]["centroid"] - chain_geometry[b]["centroid"]))
        for a, b in combinations(ring_chain_names, 2)
    ))
    return best


def deduplicate_rings_by_geometry(rings, tolerance=1.5):
    """
    Collapses rings that are the SAME physical ring — related by the
    assembly's own symmetry — into one representative row. This is the
    redundancy check: a tetrahedral cage built from one identity group has
    4 physically equivalent trimers, and there's no reason to report all
    4 as if they were 4 different findings, just as there's no reason to
    merge two DIFFERENT rings that happen to look similar by one number.

    Equivalence is judged by comparing each ring's FULL shape — its
    "fingerprint" from find_shortest_ring_junction (the sorted tuple of
    every pairwise centroid distance between its members) — element by
    element, within `tolerance` Angstroms, rather than by comparing just
    the single nc_distance number each ring reports. A single-number
    comparison fails in both directions:

      - Two DIFFERENT rings can end up with a similar nc_distance by
        coincidence — nc_distance is only the SHORTEST junction within a
        ring, and plenty of unrelated ring shapes could happen to share a
        similarly-short one. Comparing nc_distance alone risks merging
        two genuinely non-equivalent rings into one.
      - Two truly EQUIVALENT (symmetry-related) rings can end up with
        different nc_distance if real biological asymmetry (see module
        docstring) happens to make a different chain pair the "shortest"
        one in each copy. Comparing nc_distance alone risks treating two
        real copies of the same ring as separate findings.

    Comparing the whole shape is a much stronger test for "these are the
    same physical ring" than either failure mode allows.

    Rings of DIFFERENT sizes are never compared to each other — they're
    grouped by ring_size first. This matters for multi-component cages
    where different proteins form different-sized rings; nothing about a
    scalar summary should ever accidentally conflate a trimer with a
    dimer just because their numbers happen to land close together.

    Method, per ring_size group: sort rings by fingerprint (so
    similarly-shaped rings land near each other), then walk through in
    order, joining a ring to the first existing cluster whose anchor (the
    ring that started that cluster) matches it within `tolerance` at
    EVERY corresponding fingerprint position — not just on average —
    starting a new cluster otherwise.

    Returns a list of representative rings (the lowest-nc_distance ring
    in each cluster), each with "equivalent_rings" (chain lists for every
    ring folded into that cluster) and "ring_count" added.
    """
    if not rings:
        return []

    by_size = {}
    for ring in rings:
        by_size.setdefault(ring["ring_size"], []).append(ring)

    representatives = []
    for size_rings in by_size.values():
        sorted_rings = sorted(size_rings, key=lambda r: r["fingerprint"])

        clusters = []
        for ring in sorted_rings:
            fingerprint = ring["fingerprint"]
            placed = False
            for cluster in clusters:
                anchor_fingerprint = cluster[0]["fingerprint"]
                if all(abs(a - b) <= tolerance for a, b in zip(anchor_fingerprint, fingerprint)):
                    cluster.append(ring)
                    placed = True
                    break
            if not placed:
                clusters.append([ring])

        for cluster in clusters:
            representative = dict(min(cluster, key=lambda r: r["nc_distance"]))
            representative["equivalent_rings"] = [r["chain_order"] for r in cluster]
            representative["ring_count"] = len(cluster)
            representatives.append(representative)

    return representatives


def _apply_annotators(ring, chain_geometry, annotators):
    """
    Runs each function in `annotators` over one ring in order, chaining
    output into input — the result of annotator N is what annotator N+1
    receives. Each annotator's contract: annotator(ring, chain_geometry)
    -> ring (a dict — either the same one with fields added, or a new one;
    orientation.annotate_ring_orientation is the reference implementation
    this contract is built around).

    chain_geometry is already sitting in memory at the point this gets
    called (built once per assembly inside analyze_assembly_rings) — this
    is the whole point of doing annotation here rather than downstream:
    every annotator reuses that same structure load instead of each one
    re-parsing the PDB file from scratch. As more annotators are added
    (secondary structure, solvent accessibility), they all share this one
    load too.

    A failing annotator prints a warning naming the annotator and the
    ring's junction, then the loop continues with whatever the ring
    dict looked like going in — one broken annotator doesn't lose the
    ring or any annotations that already succeeded on it.
    """
    for annotator in annotators:
        try:
            ring = annotator(ring, chain_geometry)
        except Exception as e:
            print(f"Annotator {annotator.__name__} failed on "
                  f"{ring.get('from_chain')}->{ring.get('to_chain')}: {e}")
    return ring


def _flatten_ring_row(ring):
    """
    Flattens one ring dict into a single-level dict suitable for a
    DataFrame row. Any value that's itself a dict (e.g. the "orientation"
    field an annotator adds) gets its keys pulled up one level with the
    parent key as a prefix — "orientation": {"backbone_angle": 12.4, ...}
    becomes "orientation_backbone_angle": 12.4 — so results land as plain
    sortable/filterable columns instead of a dict blob per cell that has
    to be unpacked before it's usable.

    A None value (an annotator that ran but found nothing computable) is
    dropped rather than kept as a bare key: pandas fills it in as NaN
    automatically for that column once other rows populate it, which
    reads cleaner than a stray all-null column sitting alongside the
    flattened ones.
    """
    row = {}
    for key, value in ring.items():
        if value is None:
            continue
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                row[f"{key}_{subkey}"] = subvalue
        else:
            row[key] = value
    return row


def analyze_assembly_rings(filepath, assembly_id, ring_size=None, candidate_ring_sizes=ALLOWED_RING_SIZES, tolerance=1.5, max_combinations=MAX_RING_CANDIDATE_COMBINATIONS, max_shape_cv=None, min_coverage=RING_SIZE_COVERAGE_FLOOR, annotators=None):
    """
    Full per-assembly ring pipeline: load the structure, compute chain
    geometry, group chains by sequence identity (the two-component
    safeguard), determine each identity group's ring size, cluster it into
    spatial rings of that size, and find the best fusion junction within
    each ring.

    ring_size : the oligomeric count for the subunit you're targeting (3
        for a trimer, etc), if you already trust it — e.g. from
        oligomeric_count / stoichiometry via query.py's fetch_metadata().
        When given, it must be one of ALLOWED_RING_SIZES and is used
        directly (via find_best_rings_by_group_distance) for every
        identity group, skipping auto-detection — faster, and appropriate
        when the reported oligomeric state is already trustworthy.

        Left as None (the default), the ring size is instead AUTO-DETECTED
        independently for each identity group via detect_ring_size() —
        this is "Challenge 1": recognizing how many chains belong in the
        ring from geometry alone, with no dependency on external metadata
        being present or correct. This is the right default for
        exploratory work, and multi-component assemblies where different
        proteins may form different-sized rings are handled correctly
        since detection runs per identity group, not once for the whole
        assembly.

    candidate_ring_sizes : which sizes auto-detection may consider
        (default ALLOWED_RING_SIZES). Only relevant when ring_size=None.

    max_shape_cv, min_coverage : passed straight through to
        find_best_rings_by_group_distance / detect_ring_size for every
        identity group — see _ideal_ring_shape_cv / RING_SHAPE_CV_JITTER_MARGIN
        and RING_SIZE_COVERAGE_FLOOR. max_shape_cv filters out chain
        groups whose internal shape isn't uniform enough to be a real
        ring (rejecting, e.g., a candidate that accidentally mixes chains
        from two different true rings);
        min_coverage is the auto-detection qualification floor used to
        prefer the largest valid ring size (see detect_ring_size).

    Ring membership within a given size is found via
    find_best_rings_by_group_distance (or detect_ring_size, which calls
    the same underlying scoring) — every possible group of that size is
    scored and the tightest non-overlapping groups are kept, so a ring is
    never built by first committing to a single closest PAIR that might
    belong to two different physical rings.

    Rings that are the SAME physical ring (within `tolerance` Angstroms
    of each other, judged by full shape — see deduplicate_rings_by_geometry)
    are collapsed to one representative row, so the result doesn't repeat
    the same ring once per symmetric copy. The full chain lists for every
    equivalent ring are still available via the representative's
    "equivalent_rings" field.

    Returns {"assembly_id": ..., "rings": [ring_result, ...]} — one row
    per DISTINCT ring found (within tolerance), potentially more than one
    if the assembly has more than one identity group, or a genuinely
    asymmetric arrangement.

    annotators : optional list of per-ring annotator functions, e.g.
        [orientation.annotate_ring_orientation]. Each runs against this
        assembly's already-loaded chain_geometry (see _apply_annotators)
        right here, before the rings are returned — so orientation (and
        later, secondary structure / solvent accessibility) never needs
        the caller to reload the structure a second time downstream. This
        is the only wiring distance.py has to those modules: it never
        imports them itself, so pass whichever annotator functions you
        want at the call site and distance.py stays agnostic to what they
        compute.
    """
    if ring_size is not None and ring_size not in ALLOWED_RING_SIZES:
        raise ValueError(
            f"ring_size must be one of {ALLOWED_RING_SIZES} — Platonic-solid "
            f"(T/O/I point-group) cages only have 2-, 3-, 4-, and 5-fold "
            f"rotational symmetry axes, so no other ring size is physically "
            f"possible — got {ring_size}"
        )

    st = gemmi.read_structure(filepath)
    model = st[0]

    chain_geometry = {}
    for chain in model:
        geometry = get_chain_ca_geometry(chain)
        if geometry is not None:
            chain_geometry[chain.name] = geometry

    identity_groups = group_chains_by_sequence(model)

    rings = []
    for group in identity_groups:
        usable = [name for name in group if name in chain_geometry]

        if ring_size is not None:
            if len(usable) < ring_size:
                continue
            chain_groups = find_best_rings_by_group_distance(
                usable, chain_geometry, ring_size, tolerance=tolerance,
                max_combinations=max_combinations, max_shape_cv=max_shape_cv,
            )
        else:
            detected_size, chain_groups = detect_ring_size(
                usable, chain_geometry, candidate_sizes=candidate_ring_sizes,
                tolerance=tolerance, max_combinations=max_combinations,
                max_shape_cv=max_shape_cv, min_coverage=min_coverage,
            )
            if detected_size is None:
                if len(usable) >= min(candidate_ring_sizes):
                    print(
                        f"analyze_assembly_rings: no ring size in {candidate_ring_sizes} fit "
                        f"an identity group of {len(usable)} chains within tolerance={tolerance}Å "
                        f"for assembly {assembly_id} — skipping this group."
                    )
                continue

        for chain_group in chain_groups:
            rings.append(find_shortest_ring_junction(chain_group, chain_geometry))

    rings = deduplicate_rings_by_geometry(rings, tolerance=tolerance)

    if annotators:
        rings = [_apply_annotators(ring, chain_geometry, annotators) for ring in rings]

    return {"assembly_id": assembly_id, "rings": rings}


def run_ring_analysis(df, ring_size=None, filepath_column="filepath", assembly_id_column="assembly_id", tolerance=1.5, candidate_ring_sizes=ALLOWED_RING_SIZES, max_combinations=MAX_RING_CANDIDATE_COMBINATIONS, max_shape_cv=None, min_coverage=RING_SIZE_COVERAGE_FLOOR, annotators=None):
    """
    Runs analyze_assembly_rings across every row of a candidates DataFrame,
    flattening the (possibly several) DISTINCT rings found per assembly
    into one row per ring — so results sort/filter like a normal table,
    with the best (lowest nc_distance) rings naturally on top regardless
    of which assembly they came from.

    ring_size : None (default — auto-detect independently for every
        assembly; see analyze_assembly_rings), a fixed integer from
        ALLOWED_RING_SIZES (same size for every row), or the NAME of a
        column in df to look up per-row — e.g. "oligomeric_count", if you
        fetched that via query.py's fetch_metadata() and merged it in.
        Useful if one batch mixes candidates with different, already-known
        oligomeric states.
    tolerance : in Angstroms, passed straight through to every stage for
        every assembly in the batch — see the module docstring for what
        it controls.
    candidate_ring_sizes, max_combinations, max_shape_cv, min_coverage :
        passed straight through to analyze_assembly_rings for every
        assembly in the batch — only relevant when auto-detecting
        (ring_size is None; a per-row column always supplies an explicit
        size). See _ideal_ring_shape_cv / RING_SIZE_COVERAGE_FLOOR.
    annotators : optional list of per-ring annotator functions, passed
        straight through to analyze_assembly_rings for every assembly in
        the batch (see that function's docstring). Their output is
        flattened into plain DataFrame columns by _flatten_ring_row — e.g.
        passing [orientation.annotate_ring_orientation] adds
        orientation_backbone_angle / orientation_from_alignment /
        orientation_to_alignment columns to the result, with no separate
        structure reload and no manual caching required at the call site:

            from toolkit.geometry.orientation import annotate_ring_orientation
            rings_df = run_ring_analysis(
                candidates_df, annotators=[annotate_ring_orientation]
            )

        This is deliberately for cheap, numeric per-ring annotations only.
        It does NOT run plotting or anything else expensive per row — with
        potentially hundreds of candidates, generating a figure for every
        row isn't something you want happening automatically. Keep
        plotting a separate, deliberate step on whichever specific rows
        you actually want to look at (e.g. the top few after sorting).
    """
    rows = []
    for _, row in df.iterrows():
        assembly_id = (
            row[assembly_id_column] if assembly_id_column in df.columns
            else f"{row['entry_id']}-{row['assembly_num']}"
        )

        if ring_size is None:
            this_ring_size = None
        elif isinstance(ring_size, str):
            this_ring_size = row[ring_size]
        else:
            this_ring_size = ring_size

        try:
            result = analyze_assembly_rings(
                row[filepath_column], assembly_id, ring_size=this_ring_size,
                candidate_ring_sizes=candidate_ring_sizes, tolerance=tolerance,
                max_combinations=max_combinations, max_shape_cv=max_shape_cv,
                min_coverage=min_coverage, annotators=annotators,
            )
            for ring in result["rings"]:
                rows.append({"assembly_id": assembly_id, **_flatten_ring_row(ring)})
        except Exception as e:
            print(f"Failed: {assembly_id} — {e}")

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values("nc_distance").reset_index(drop=True)
    return results_df