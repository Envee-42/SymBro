"""
distance.py — NC (N-terminus to C-terminus) distance and ring-identification
calculations on downloaded assemblies.

Built on top of termini.get_chain_ca_geometry for per-chain coordinates.
Two paths live here:

  - compute_nc_distance / process_candidates: single-closest-pair NC
    distance, kept as a lighter-weight utility for other uses.
  - analyze_assembly_rings / run_ring_analysis: the current primary path —
    groups chains directly into full oligomeric rings (e.g. all 3 chains
    of a trimer) by scoring whole candidate groups rather than relying on
    pairwise closest-centroid matching, then finds the best fusion
    ordering within each ring. Direct group scoring avoids a real failure
    mode that pairwise matching had: the globally closest PAIR of chains
    can belong to two different physical trimers rather than being
    genuine ring-mates.
"""

from itertools import combinations
import gemmi
import numpy as np
import pandas as pd

from .termini import get_chain_ca_geometry


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
        compare chains against (sequence, entity ID, etc.) — worth
        building out once you're ready to work through multi-component
        candidates specifically.

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
    # whichever is shorter — same logic as the original, just carried over.
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
        from download_candidates() on a raw ID list — see download.py),
        it's used directly rather than reconstructed from entry_id/
        assembly_num, avoiding duplicating that string-building logic in
        two places.

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
# Ring-level analysis — the current primary path.
#
# Groups chains directly into full oligomeric subunits (e.g. all 3 chains
# of a trimer) by scoring whole candidate groups rather than relying on
# pairwise closest-centroid matching, then finds the best fusion ordering
# within each ring.
#
# compute_nc_distance / process_candidates above (single-closest-pair NC
# distance) are kept as a lighter-weight utility for other uses, but are no
# longer the primary route into ring identification — direct group scoring
# turned out to avoid a real failure mode that pairwise matching had: the
# globally closest PAIR of chains can belong to two different physical
# trimers rather than being genuine ring-mates.
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

    Returns a list of groups, each a list of chain names sharing one
    sequence. Chains with no resolved polymer (already filtered out
    elsewhere) are skipped.
    """
    groups = {}
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) == 0:
            continue
        # NOTE: unverified against a live gemmi install — if this line
        # errors, the method name may differ slightly in your gemmi
        # version; paste the error and we'll correct it.
        seq = gemmi.one_letter_code(polymer.extract_sequence())
        groups.setdefault(seq, []).append(chain.name)
    return list(groups.values())


def find_best_rings_by_group_distance(chain_names, chain_geometry, ring_size, tolerance=1.5):
    """
    Directly evaluates EVERY possible group of `ring_size` chains within
    one same-sequence identity group, scores each by how tightly packed
    its members are, and greedily keeps the tightest non-overlapping
    groups first.

    This replaces an earlier pair-then-grow approach that seeded a cluster
    from the single globally closest PAIR and grew it outward — which
    turned out to have a real failure mode: the closest overall pair can
    belong to two DIFFERENT physical trimers that happen to sit near each
    other (same sequence, same cage, just neighboring rings), rather than
    being genuine ring-mates. That approach committed to a pair before
    ever checking whether the trimer it implied was actually the tightest
    one available. Scoring whole groups directly avoids that: a candidate
    group is only accepted by comparing it against every OTHER possible
    group, not by trusting one pairwise distance as a stand-in for ring
    membership.

    Scoring: each candidate group's score is the SUM of all pairwise
    centroid distances between its members (a group of 3 has 3 such
    pairs). Using the sum rather than, say, just the single closest pair's
    distance rewards groups where every member is close to every OTHER
    member — a real trimer should be uniformly compact, not just contain
    one tight pair plus a straggler that happens to be nearby.

    Assignment: candidate groups are sorted tightest-first, then accepted
    greedily as long as none of their chains have already been claimed by
    a previously-accepted (tighter) group. This is a greedy approximation,
    not a guaranteed globally-optimal partition — but since it always
    prioritizes the tightest available group at each step, a genuine
    trimer (which should score much lower than any spurious cross-ring
    grouping) gets first claim on its own chains before a worse grouping
    has a chance to steal one of them.

    Quality cutoff: without a floor, greedy assignment happily keeps going
    after the real rings are claimed — it'll force whatever chains are
    LEFT OVER into additional "rings" too, no matter how loose, right down
    to extra copies in the asymmetric unit, mismatched leftover chains, or
    genuinely non-equivalent groupings, none of which are real cage
    subunits. To prevent that: the first accepted group's score becomes an
    anchor, and any later candidate whose score exceeds anchor + tolerance
    stops the search entirely (candidates are sorted tightest-first, so
    once one candidate fails the cutoff, every remaining one is at least
    as loose and would fail too). Chains that don't end up in any group
    within tolerance of the tightest ring are left unassigned rather than
    forced into a bad one — that's the correct outcome, not a bug, when
    those chains genuinely aren't part of an equivalent ring.

    This deliberately doesn't hardcode an expected ring COUNT (e.g. 4 for
    a tetrahedral cage) — that number is a property of point-group
    symmetry and would need separate logic per symmetry type (8 for
    octahedral, 20 for icosahedral, etc.), and gets it for free once
    tightness is bounded correctly: a tetrahedral assembly's 4 trimers are
    all real and comparably tight, so all 4 pass the cutoff together.

    tolerance : same units as the centroid-distance sum being scored
    (Angstroms), and conceptually the same tolerance later used by
    deduplicate_rings_by_geometry on nc_distance — but it's a genuinely
    different metric (summed pairwise centroid distances here vs. a
    single junction's NC distance there), so treat the shared default as
    a reasonable starting heuristic, not a guarantee the two should always
    match. Widen it if real assemblies with more geometric asymmetry
    between equivalent rings start getting split up unnecessarily.

    Complexity note: for m chains this evaluates C(m, ring_size)
    combinations — e.g. 220 for 12 chains at ring_size=3, still only in
    the tens of thousands even for a 60-chain icosahedral cage. Cheap
    enough not to worry about.

    Returns a list of chain-name groups (each of length ring_size). Chains
    left out (either by the overlap rule or the tolerance cutoff above)
    are excluded entirely rather than padded — worth checking
    len(groups) * ring_size against len(chain_names) if you want to
    confirm how many were dropped and why.
    """
    candidates = []
    for group in combinations(chain_names, ring_size):
        score = sum(
            np.linalg.norm(chain_geometry[a]["centroid"] - chain_geometry[b]["centroid"])
            for a, b in combinations(group, 2)
        )
        candidates.append((score, group))

    candidates.sort(key=lambda item: item[0])  # tightest groups first

    assigned = set()
    rings = []
    anchor_score = None
    for score, group in candidates:
        if not assigned.isdisjoint(group):
            continue  # a member of this group was already claimed by a tighter ring
        if anchor_score is None:
            anchor_score = score  # first accepted group sets the quality bar
        elif score - anchor_score > tolerance:
            break  # this and every remaining (looser) candidate fail the cutoff
        rings.append(list(group))
        assigned.update(group)

    return rings


def find_shortest_ring_junction(ring_chain_names, chain_geometry):
    """
    Given the chain names in one validated ring, finds the SHORTEST single
    N-to-C junction distance between any two DIFFERENT chains in that ring.

    This is NC distance as originally defined: one number, one specific
    junction — not a summed path across the whole ring. (An earlier version
    of this function instead brute-forced the best full ORDERING of all
    ring members to minimize a total summed distance across every
    junction — that's a different question, fusion-path length rather than
    "what's the shortest junction available," and isn't what's needed here.)

    Checks every ordered pair within the ring (chain A's C-terminus to
    chain B's N-terminus, for every A != B) and keeps the minimum. Ring
    size is small, so a direct comparison is cheap — no permutation search
    needed.

    Returns a dict: chain_order (all member chains, for reference), the
    specific from_chain/to_chain pair achieving the minimum, and
    nc_distance itself (rounded to 2 decimals).
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
    return best


def deduplicate_rings_by_geometry(rings, tolerance=1.5):
    """
    Collapses rings whose nc_distance falls within `tolerance` Angstroms of
    each other into a single representative row, rather than requiring an
    exact match.

    Real structures rarely produce EXACTLY identical geometry across
    symmetric copies — small deviations from perfect symmetry (crystal
    packing effects, minor conformational differences between otherwise-
    equivalent chains) mean the previous exact-rounding approach could
    miss rings that are clearly "the same" geometry in any meaningful
    sense. Tolerance-based grouping catches those without over-merging
    genuinely different rings.

    Method: sort all rings by nc_distance, then walk through in order.
    Each group is anchored to the nc_distance of the FIRST ring that
    started it; a ring joins the current group only if its nc_distance is
    within `tolerance` of that anchor — not just of its immediate
    predecessor. Anchoring to the group's start (rather than chaining
    neighbor-to-neighbor) keeps each group's total spread bounded by
    `tolerance`; without that, a run of rings each within tolerance of the
    next could drift arbitrarily far from where the group started and
    merge geometries that aren't really equivalent.

    Returns a list of representative rings (the lowest-nc_distance ring in
    each group), each with "equivalent_rings" (chain lists for every ring
    folded into that group) and "ring_count" added.
    """
    if not rings:
        return []

    sorted_rings = sorted(rings, key=lambda r: r["nc_distance"])

    groups = []
    current_group = [sorted_rings[0]]
    anchor = sorted_rings[0]["nc_distance"]

    for ring in sorted_rings[1:]:
        if ring["nc_distance"] - anchor <= tolerance:
            current_group.append(ring)
        else:
            groups.append(current_group)
            current_group = [ring]
            anchor = ring["nc_distance"]
    groups.append(current_group)

    representatives = []
    for group in groups:
        representative = dict(group[0])
        representative["equivalent_rings"] = [r["chain_order"] for r in group]
        representative["ring_count"] = len(group)
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

    A None value (an annotator that ran but found nothing computable —
    see annotate_ring_orientation) is dropped rather than kept as a bare
    key: pandas fills it in as NaN automatically for that column once
    other rows populate it, which reads cleaner than a stray all-null
    "orientation" column sitting alongside the flattened ones.
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


def analyze_assembly_rings(filepath, assembly_id, ring_size, tolerance=1.5, annotators=None):
    """
    Full per-assembly ring pipeline: load the structure, compute chain
    geometry, group chains by sequence identity (the two-component
    safeguard), cluster each identity group into spatial rings of
    ring_size chains, and find the best fusion ordering within each ring.

    ring_size : the expected oligomeric count for the subunit you're
        targeting (3 for a trimer, etc). Pass this in from metadata you
        already have (oligomeric_count / stoichiometry from fetch_metadata)
        rather than trying to infer it structurally — inferring ring size
        purely from geometry is a much harder, less reliable problem than
        just using the number RCSB already reports.

    Ring membership is found directly via find_best_rings_by_group_distance
    (see that function's docstring) — every possible group of ring_size
    chains is scored and the tightest non-overlapping groups are kept, so
    a ring is never built by first committing to a single closest PAIR
    that might turn out to belong to two different physical trimers.

    Geometrically similar rings (within `tolerance` Angstroms of each
    other in nc_distance — common in symmetric cages, e.g. all 4 trimers
    of a tetrahedron) are collapsed to one representative row via
    deduplicate_rings_by_geometry, so the result doesn't repeat the same
    number 4 times over. The full chain lists for every equivalent ring
    are still available via the representative's "equivalent_rings" field.

    Returns {"assembly_id": ..., "rings": [ring_result, ...]} — one row
    per DISTINCT ring geometry found (within tolerance), potentially still
    more than one if the assembly's rings aren't all equivalent (e.g. a
    genuinely asymmetric arrangement, or more than one identity group).

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
        if len(usable) < ring_size:
            continue
        for chain_group in find_best_rings_by_group_distance(usable, chain_geometry, ring_size, tolerance=tolerance):
            rings.append(find_shortest_ring_junction(chain_group, chain_geometry))

    rings = deduplicate_rings_by_geometry(rings, tolerance=tolerance)

    if annotators:
        rings = [_apply_annotators(ring, chain_geometry, annotators) for ring in rings]

    return {"assembly_id": assembly_id, "rings": rings}


def run_ring_analysis(df, ring_size, filepath_column="filepath", assembly_id_column="assembly_id", tolerance=1.5, annotators=None):
    """
    Runs analyze_assembly_rings across every row of a candidates DataFrame,
    flattening the (possibly several) DISTINCT rings found per assembly
    into one row per ring — so results sort/filter like a normal table,
    with the best (lowest nc_distance) rings naturally on top regardless
    of which assembly they came from. Rings within `tolerance` Angstroms
    of each other (typical for symmetric cages) have already been
    collapsed to one row by analyze_assembly_rings before reaching here.

    ring_size : either a fixed integer (same ring size for every row), or
        the NAME of a column in df to look up per-row — e.g. "oligomeric_count",
        if you fetched that via fetch_metadata and merged it in. Useful if
        one batch mixes candidates with different oligomeric states.
    tolerance : passed straight through to deduplicate_rings_by_geometry
        for every assembly in the batch — see that function for what it
        controls.
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
                candidates_df, ring_size=3, annotators=[annotate_ring_orientation]
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
        this_ring_size = row[ring_size] if isinstance(ring_size, str) else ring_size

        try:
            result = analyze_assembly_rings(
                row[filepath_column], assembly_id, ring_size=this_ring_size,
                tolerance=tolerance, annotators=annotators,
            )
            for ring in result["rings"]:
                rows.append({"assembly_id": assembly_id, **_flatten_ring_row(ring)})
        except Exception as e:
            print(f"Failed: {assembly_id} — {e}")

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values("nc_distance").reset_index(drop=True)
    return results_df