"""
distance.py — NC (N-terminus to C-terminus) distance and ring-identification
calculations on downloaded assemblies.

Built on top of termini.get_chain_ca_geometry for per-chain coordinates.
Two paths live here:

  - compute_nc_distance / process_candidates: single-closest-pair NC
    distance, kept as a lighter-weight utility for other uses.
  - analyze_assembly_rings / run_ring_analysis: the primary path — finds
    complete oligomeric rings (e.g. all 3 chains of a trimer) AND their
    correct chain-to-chain fusion order in one step, via the directed
    terminal-cycle method described below, then reports each ring's best
    (and full) fusion junction(s).

ORIGIN — this module's ring-detection logic began as a direct
generalization of one manual, exploratory notebook analysis (Phase_2.ipynb)
that correctly identified all 8 trimers of a real downloaded cage (PDB
3vcd, 24 chains) by eye: compute every chain's centroid, sort all pairwise
centroid distances, notice the sharp gap between "same ring" (~34Å) and
"different ring" (~100Å+) distances, and print every chain trio whose
worst internal distance stayed under a hand-picked cutoff (35.0Å). That
worked, but only because it was tuned by eye for one specific structure,
assumed the ring size (3) was already known, assumed there'd be no overlap
between candidate trios to resolve, and offered no way to tell a genuinely
clean call apart from a lucky one.

HISTORY (superseded approaches, kept as design record):

  1. Atomic-contact-based detection — built a graph from actual Cα–Cα
     atomic contacts, on the theory that real interface SIZE, not just
     centroid proximity, should be the most reliable signal for which
     chains belong to the same designed ring. Worked on synthetic
     geometry but failed on real downloaded structures (frequently
     returned dimers instead of trimers) — real "identical" copies of one
     ring aren't perfectly identical at the atomic level (crystallographic
     disorder, minor conformational asymmetry, different residues
     resolved in different copies), so noise disproportionately breaks
     larger rings while leaving spurious dimers intact. Abandoned.

  2. Centroid-distance combinatorial group-scoring — the previous version
     of this module. For each candidate ring_size in turn, it scored
     EVERY possible group of that size within a sequence-identity group
     (C(n, size) candidates) by summed/mean pairwise CENTROID distance,
     filtered by a hand-derived shape-uniformity threshold (coefficient
     of variation vs. an ideal regular n-gon) to reject groups that mixed
     chains from two different true rings, then had to arbitrate BETWEEN
     candidate sizes when more than one produced a fully-covering,
     uniform reading of the same chains (coverage-first, with point-group
     nesting as a tie-break). This worked, but at real cost: it required
     trying every ring_size explicitly (an actual list of "candidate
     sizes to attempt", not true size-agnosticism), its combinatorial
     cost grew as C(n, size) per size attempted (capped by
     MAX_RING_CANDIDATE_COMBINATIONS to avoid pathological blowups), and
     a large fraction of its logic (CV filtering, coverage floors,
     nesting tie-breaks) existed purely to compensate for centroid
     distance being a coarse, orientation-blind signal: two chains can
     have close centroids for reasons that have nothing to do with being
     genuine ring-mates (e.g. a neighboring ring's cage-forming contact).
     Superseded by the directed terminal-cycle method below.

REDESIGN — size-agnostic directed terminal cycles (current approach):
this module now builds ring membership AND ring order directly from
N-to-C terminal proximity, rather than from whole-chain centroid
proximity, and rather than trying candidate ring sizes one at a time.

The physical insight: in a cyclic C_n assembly meant to be fused into one
continuous polypeptide, the chains are arranged head-to-tail in a single
rotational direction. If chain A's C-terminus sits near chain B's
N-terminus, then by the assembly's own rotational symmetry, B's
C-terminus must sit near some OTHER chain's N-terminus (call it C), and so
on, until the path closes back on A. That closed path — not centroid
clustering — is what actually defines a ring, and IS the fusion order,
for free, with no separate step needed to find "the best junction" among
an already-identified but unordered group of chains.

Mechanically:

  1. EXTRACT TERMINI — for every chain, take its N-terminal and
     C-terminal CA coordinates from termini.get_chain_ca_geometry
     (already computed once per chain elsewhere in the pipeline).

  2. BUILD A DIRECTED INTERFACE MATRIX — d_term(i -> j) = || C_i - N_j ||
     for every ordered pair i != j. This is NOT symmetric: C_i -> N_j and
     C_j -> N_i are unrelated numbers, because a cyclic interface has a
     direction (see _directed_terminal_matrix).

  3. THRESHOLD IT INTO A GRAPH — a directed edge i -> j exists only where
     d_term(i -> j) <= terminal_threshold (tau), a distance in Angstroms
     tight enough that the two termini are plausibly close enough to link
     (see build_terminal_graph and DEFAULT_TERMINAL_THRESHOLD).

  4. FIND ELEMENTARY CYCLES — run cycle-finding (networkx's simple_cycles,
     an implementation of Johnson's algorithm, bounded to the longest
     physically possible ring — see find_ring_cycles) on that graph. Every
     closed, non-self-intersecting cycle k1 -> k2 -> ... -> kn -> k1 found
     is a candidate ring, and its LENGTH is automatically that ring's
     size — no candidate size ever has to be proposed, tried, or ranked
     against alternatives: a genuine C2/C3/C4/C5 ring shows up as exactly
     that, a 2/3/4/5-node cycle, because that's what it structurally is.
     A candidate is also required to have internally CONSISTENT junction
     distances (see JUNCTION UNIFORMITY below) before it's kept at all.

  5. RESOLVE OVERLAP — a noisy or densely-linked graph can produce more
     candidate cycles than there are real rings (e.g. two cycles sharing a
     chain, or one real ring plus a spurious alternate path through the
     same nodes). _greedy_nonoverlapping_cycles walks every candidate
     tightest-first and accepts EVERY one that doesn't reuse an
     already-claimed chain — full stop. It does not additionally require
     a non-overlapping candidate to be within some tolerance of the very
     first (tightest) ring accepted (see BUGFIX below for why that
     earlier design was actively harmful, not just overly cautious).

What this eliminates from the old approach, and why it's not needed
anymore: there is no more shape-uniformity CV filter, because a spurious
group can no longer sneak through on raw distance alone — surviving as a
candidate now requires an entire consistent, closed, DIRECTED loop across
every one of its members, a far stronger simultaneous constraint than
"these centroids happen to be close together". There is no more
combinatorial candidate-count cap (MAX_RING_CANDIDATE_COMBINATIONS),
because cost is now driven by the actual (typically sparse — most chains
have only one or two termini within tau of them) terminal-contact graph,
not by C(n, size) over every chain in an identity group. Ambiguity
between candidate sizes, on the other hand, is NOT eliminated — an
earlier cut of this redesign tried to resolve it by picking one "winning"
size per group (first by pooling every size together and ranking by raw
tightness, then by coverage when that proved wrong — see git history /
prior revisions of this docstring). Real testing on real cages proved
picking a single winner IS the wrong framing entirely: see TOTAL-COVERAGE
AXIS SELECTION below for why every size that geometrically, fully
accounts for the group is now reported side by side, with no size ever
declared "the" answer by this module alone.

BUGFIX — dropped rings from an over-eager tolerance cutoff: the first cut
of this redesign kept the old module's "anchor + tolerance, stop once
exceeded" acceptance rule (see TOLERANCE below), just applied globally
across pooled candidate cycles instead of per ring_size. On a real
downloaded cage (8FNV-1) this produced exactly 3 of 4 real trimers: the
4th was a completely disjoint, valid cycle over exactly the chains left
over once the other 3 were claimed — not spurious, not overlapping
anything — but its mean junction distance happened to be more than
`tolerance` looser than the very first (tightest) ring accepted, so the
anchor+tolerance cutoff stopped searching before it was ever reached.
That cutoff was never actually protecting anything: the only thing that
can make a candidate cycle wrong is reusing chains an already-accepted,
tighter cycle claimed first, and disjoint chain sets already rule that
out on their own. A real cage's several rings are not guaranteed to be
equally tight to begin with — different local packing, resolution, or
how many residues happened to resolve per copy can all make one ring's
junctions measurably looser than another's without either being fake.
Conflating that expected between-ring variation with genuine noise is
what silently dropped a ring that was, by the time it mattered, the only
candidate left standing for its chains. Fixed by dropping the
tolerance-gated stop entirely — see RESOLVE OVERLAP above and
_greedy_nonoverlapping_cycles.

JUNCTION UNIFORMITY — filtering implausible cycles before they're ever
ranked. Raw distance and closed-loop topology alone (steps 1-5 above)
still leave one gap: a cycle can be entirely real in the sense of every
individual link clearing terminal_threshold, non-overlapping with
anything else, and yet still be the WRONG cycle, because the closest
available terminus by raw distance isn't always the true ring-mate's
(see LIMITATIONS below — observed on a real downloaded cage, 1WPB-2,
with unusually long, winding chains). The fix — suggested directly by
comparing what a genuine ring's geometry has to look like against what
an accidental one doesn't — is to also check each candidate's junctions
against EACH OTHER, not just against terminal_threshold individually: in
a real C_n ring, every consecutive junction is the SAME physical
interface, repeated around the ring by the assembly's own rotational
symmetry, so their Cα-Cα distances should track each other closely,
give or take ordinary biological jitter. A cycle stitched together from
coincidentally-close-but-non-corresponding termini has no such symmetry
forcing its links to agree — so requiring that they agree (within
max_junction_spread — see DEFAULT_JUNCTION_UNIFORMITY_TOLERANCE) rejects
a class of wrong-but-otherwise-plausible cycles that distance and
topology alone can't tell apart from real ones. This is find_ring_cycles'
job, applied per candidate before find_ring_cycles even returns it, so it
is a per-candidate structural check — unlike the BUGFIX above, it never
compares one candidate against another, and so can't reintroduce that
same failure mode (a genuinely internally-consistent ring is never
penalized just because some other ring elsewhere happens to be tighter
or looser).

INTERFACE CONTACT SANITY CHECK — a second, independent per-candidate
filter, also applied before greedy acceptance, targeting the specific
1WPB-2 failure mode JUNCTION UNIFORMITY alone didn't fully catch:
unusually long, winding chains where a chain's termini can genuinely
reach across to a chain it doesn't otherwise pack against anywhere. The
obvious first idea — require the two chains' minimum CA-CA distance
(over ALL their residues, not just termini) to fall under some cutoff —
turns out to be tautological and was rejected before being implemented:
the ring's own junction is itself a CA-to-CA pair (the from_chain
C-terminus and to_chain N-terminus), so the minimum inter-chain CA-CA
distance can never exceed the junction distance that already got the
candidate this far. A minimum-distance check could never actually reject
anything post-junction; it would just re-confirm what terminal_threshold
already guaranteed. The non-tautological version — and what's actually
implemented (_interchain_contact_count / _filter_has_interface_contact)
— is a COUNT: how many INDEPENDENT residue pairs (not just the one
already-known-close termini pair) fall within contact_threshold of each
other. A genuine subunit interface has more than one contact; a chain
whose termini happen to reach across to an unrelated neighbor, with
nothing else about either chain nearby, does not. A candidate is dropped
if any of its own junction pairs falls short of min_contact_pairs — see
DEFAULT_CONTACT_THRESHOLD / DEFAULT_MIN_CONTACT_PAIRS. Like JUNCTION
UNIFORMITY, this is a property of one candidate in isolation, never a
comparison against other candidates, so it can't reintroduce the BUGFIX
failure mode above. It does require the full per-residue CA trace,
which termini.get_chain_ca_geometry now also returns ("ca_coords") for
exactly this purpose — see that function's docstring; a chain_geometry
built by an older version without that field skips this check with a
printed explanation rather than erroring.

TOTAL-COVERAGE AXIS SELECTION — replaces an earlier "pick one winning
size" design (coverage-first, tightness-first — see git history) after
real testing on real downloaded cages showed picking a single winner was
the wrong framing. The observed failure that prompted this: on 3VCD (8
real trimers expected), the method above found only 2 correct trimers
plus several erroneous dimers. Why: a T/O/I point-group cage's own
rotational symmetry is NOT limited to the n-fold axis its designed
subunit oligomerizes around. An octahedral cage built from trimers, for
instance, also has real 2-fold (and 4-fold) axes relating chains ACROSS
different trimer copies — that IS the cage's architecture, not noise. A
pair of chains related by one of these other axes can have reciprocally
close termini (C_i near N_j AND C_j near N_i — a genuine closed 2-cycle),
uniform junction distances (trivially, with only 2 links), and a real
multi-residue interface (cage-forming contacts between neighboring
trimers are often substantial, sometimes tighter than the trimer's own
internal spacing — this exact scenario is what the module's own
ORIGIN/HISTORY notes originally called the "octahedral cage bug" against
the old centroid method, recurring here against the new terminal-cycle
method for the same underlying reason under a different mechanism). No
per-candidate test — distance, uniformity, or contact — can tell such a
pair apart from a genuine ring-forming junction, because by every one of
those measures, it IS real.

The fix is to stop trying to pick a winner at all. detect_rings, when
auto-detecting, runs EVERY candidate size's non-overlapping acceptance
independently against the WHOLE identity group, and keeps a size's
result ONLY if it accounts for every single chain in the group — zero
residual. A size that leaves even one chain unassigned is discarded
ENTIRELY for that group (not partially reported): a coincidental,
non-ring-forming pairing only ever explains the specific chains it
happens to relate, never the whole group, so it fails this test and is
dropped regardless of how tight or uniform or contact-validated it
looked in isolation. A size that DOES achieve total coverage is a
genuine, complete symmetry axis of the assembly by construction — and
since a real point-group cage can genuinely have more than one such axis
simultaneously (the octahedral cage's 3-fold AND 2-fold axes are both
"real" and can both, independently, fully partition the same 24 chains),
detect_rings does not pick between them: it returns the UNION of every
size that achieves total coverage, each candidate still tagged with its
own "ring_size", so the caller (or a human) decides which axis is the
biologically designed subunit — typically the one matching a trusted
oligomeric_count from metadata, or the smallest one, or whichever a
downstream step (RFdiffusion input selection, etc.) actually needs.

This trades one failure mode for a different, more honest one: a size
that's genuinely correct but only PARTIALLY detected (some real rings
found, a few chains missed — e.g. due to terminal_threshold /
max_junction_spread / contact settings being slightly too strict for a
specific structure) is now discarded in full rather than reported with
its gaps, since "leaves residual chains" and "is the wrong size" are not
distinguished by this rule — see 1WPB-2 under LIMITATIONS below for a
real example, and the printed per-size summary detect_rings emits for
diagnosing which case you're in. The safest way to avoid this trade-off
entirely, when available, remains passing an explicit ring_size from
trusted metadata (oligomeric_count / stoichiometry via query.py's
fetch_metadata()) — that path never requires total coverage, since
there's no size ambiguity left to resolve.

REDUNDANCY — a cage assembled from several physically equivalent copies
of one ring (e.g. a tetrahedron's 4 trimers) still produces one distinct
cycle PER physical ring — cycle-finding naturally can't merge them, since
they involve entirely different chains. But nothing stops those several
equivalent rings from being reported as separate findings when the
pipeline's actual purpose (design one representative construct per unique
ring, not once per symmetric copy) only wants one. deduplicate_rings_by_
geometry() still does this collapsing, unchanged in method from before:
compare each ring's full pairwise-centroid-distance "fingerprint" (not
just its single nc_distance number) within `tolerance`, and merge.

TOLERANCE vs. TERMINAL_THRESHOLD — two different knobs, and it matters
which one you reach for:

  - terminal_threshold (tau) is a HARD, structural gate — "is this
    specific C-to-N contact even physically plausible as a fusion
    junction at all?" It controls which edges exist in the graph, and
    therefore which cycles can even be found in the first place. Too
    tight and real, slightly-asymmetric ring junctions get missed
    entirely (a ring silently vanishes); too loose and spurious
    non-adjacent contacts start appearing as edges (more candidate
    cycles for the greedy step to have to discriminate between).

  - tolerance, in Angstroms, no longer has anything to do with whether a
    candidate ring gets accepted (see BUGFIX above) — it now governs
    exactly one thing: how far apart two rings' fingerprints may be and
    still be treated as the same physical ring by
    deduplicate_rings_by_geometry. Real structures never hit exact
    symmetry (crystal packing, minor conformational differences,
    refinement noise), so this absorbs that biological jitter when
    collapsing symmetric copies; it has no effect on which edges/cycles
    exist, or on which non-overlapping cycles get accepted as rings.

LIMITATIONS — this method is only as good as the assumption it's built
on: that an engineered cyclic ring's true adjacent-chain junction really
is the closest C-to-N contact its termini have available. If a chain's
true ring-mate's N-terminus is NOT its single closest N-terminus overall
— e.g. some other, unrelated chain's N-terminus happens to sit even
closer by coincidence — tightest-first greedy acceptance (see
_greedy_nonoverlapping_cycles) will prefer that tighter but biologically
wrong edge, and the true ring's cycle can fail to close at all, or close
around the wrong chain. This is a narrower failure mode than the old
centroid-based method's equivalent risk (two whole CHAINS coincidentally
sitting close together, for reasons unrelated to their specific termini)
— a false edge here requires an actual terminus-level coincidence, not
just a whole-chain one — but it isn't impossible. A ring silently missing
from the results is worth a manual sanity check (e.g. against a trusted
oligomeric_count from metadata) rather than trusted blindly on a novel
structure, same as it would have been under the old method.

A second, distinct limitation comes directly from TOTAL-COVERAGE AXIS
SELECTION above: when auto-detecting, a candidate size that's genuinely
correct but only PARTIALLY found (a real ring size, with a few real
copies missed due to terminal_threshold / max_junction_spread / contact
settings being a little too strict for this specific structure) is
INDISTINGUISHABLE, by this module's own total-coverage rule, from a size
that's simply wrong — both leave residual chains, and both get discarded
entirely. This means a size can vanish from the results not because it
isn't real, but because detection under-counted it by even one ring. See
1WPB-2 below for a concrete case, and detect_rings' printed per-size
summary for how to tell the two apart on a specific structure (a size
discarded with only 1-2 chains left over, close to a plausible ring's
worth, is far more likely under-detected than genuinely wrong).

Observed in practice on real downloaded cages, updated as each round of
mitigations was tried against real data (not synthetic geometry) rather
than left as a one-time note:

  - 1WPB-2 (geometrically unusual — long, winding chains rather than
    compact globular subunits), FIRST observation: the ring that got
    accepted was internally consistent but not the biologically intended
    trio — a winding chain's closest terminus by raw distance isn't
    guaranteed to be its true ring-mate's. AFTER adding JUNCTION
    UNIFORMITY and INTERFACE CONTACT SANITY CHECK: the WRONG-chains
    failure mode was gone — the correct chains were being identified —
    but only 5 of the real 8 rings were found; the other 3 were missing
    entirely, not misassigned. Under the CURRENT total-coverage rule,
    this specific result (5 of 8, 3 chains left over) would now be
    discarded ENTIRELY rather than returned partially — see the second
    LIMITATIONS paragraph above. This remains an open, unresolved
    detection-sensitivity problem as of this writing — the honest fix
    requires calibrating against 1WPB-2's own actual numbers (which
    junctions/contacts the 3 missing rings have, and how far short of the
    defaults they fall), not a blind default change with no real
    structure to check it against. See detect_rings' printed per-size
    summary for a starting point, and try rerunning with
    max_junction_spread=None and/or contact_threshold=None to see
    whether either recovers total coverage — or pass an explicit
    ring_size=3 for this structure, which bypasses the total-coverage
    requirement entirely and returns whatever it finds, gaps included.

  - 3VCD (8 real trimers expected): only 2 real trimers plus several
    erroneous dimers were found, under the old single-winner design. This
    was NOT a distance, uniformity, or contact problem — it was the
    cage's own additional symmetry axes (see TOTAL-COVERAGE AXIS
    SELECTION above) being read as a competition to resolve rather than
    as multiple simultaneously-real answers to report. Under the CURRENT
    design, whichever size(s) among 2/3/4/5 achieve total coverage of
    this 24-chain group are ALL returned, tagged by ring_size — the real
    3-fold trimer axis should now appear complete (8 rings, 24/24 chains)
    alongside any other size that also happens to achieve total coverage,
    rather than one being silently preferred over the other.

If terminal_threshold / max_junction_spread / contact_threshold /
min_contact_pairs tuning still leaves real rings missing after checking
per the 1WPB-2 note above, the next untried lead is directional:
n_vector/c_vector (see termini.get_chain_ca_geometry) describe each
terminus's local backbone direction, and a genuine fusion junction should
have the upstream chain's C-terminus pointing roughly toward the
downstream chain's N-terminus, not just sitting near it. Scoring
candidate edges by that directional consistency — the same information
orientation.py already uses for a DIFFERENT purpose (judging linker
difficulty for an already-chosen junction) — remains a plausible
follow-up, not yet implemented.
"""

from itertools import combinations

import gemmi
import networkx as nx
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
ALLOWED_RING_SIZES = (2, 3, 4, 5)

# Default tau: the maximum C_i -> N_j distance (Angstroms) for a directed
# terminal contact to be considered a plausible fusion junction at all,
# and therefore to exist as an edge in the terminal graph at all (see
# build_terminal_graph). Chosen at the permissive end of the ~10-15Å range
# real linked/closely-interacting termini fall in: it's cheaper to let a
# few extra, non-adjacent contacts appear as edges (the tightness-ranked,
# non-overlapping cycle acceptance in detect_rings sorts those out) than
# to silently miss a real ring because one of its junctions runs a little
# wide of a stricter cutoff. Narrow this per-call if a specific structure
# is producing spurious extra edges/cycles; widen it if a real ring isn't
# being found at all.
DEFAULT_TERMINAL_THRESHOLD = 15.0

# Default junction-uniformity gate: the maximum allowed spread (loosest
# junction distance minus tightest, in Angstroms) within a single
# candidate ring before it's excluded outright — see find_ring_cycles and
# the module docstring's JUNCTION UNIFORMITY section. In a genuine C_n
# ring every consecutive junction is a copy of the SAME physical
# interface, related by the ring's own rotational symmetry, so their
# distances should track each other closely; a candidate stitched
# together from coincidentally-close-but-non-corresponding termini has no
# such symmetry forcing that agreement. 5.0Å is generous relative to the
# ~1-2Å of spread ordinary biological jitter (crystal packing, minor
# conformational asymmetry) produces between genuinely equivalent
# junctions, while still catching a candidate whose links plainly don't
# belong to one consistent interface. Set to None to disable this filter
# entirely and fall back to distance/threshold alone, if it turns out to
# reject a real ring on a specific structure.
DEFAULT_JUNCTION_UNIFORMITY_TOLERANCE = 5.0

# Interface-contact sanity check (see the module docstring's INTERFACE
# CONTACT SANITY CHECK section): two chains linked by an accepted ring
# junction must show a real, multi-residue interface — not just the one
# coincidentally-close termini pair that got the junction accepted in the
# first place. DEFAULT_CONTACT_THRESHOLD, in Angstroms, is the Cα-Cα
# cutoff for counting a residue PAIR (one from each chain) as "in
# contact" — 8Å is a standard, generous convention for a Cα-based contact
# definition (heavy-atom/side-chain contacts sit closer, ~4-5Å, but Cα
# positions for genuinely interacting residues commonly land out to
# ~8-10Å). DEFAULT_MIN_CONTACT_PAIRS is how many such pairs must exist
# between the two chains before they're accepted as a real interface — 1
# is not enough, since the ring's own junction (the termini pair itself)
# already guarantees at least that one; requiring MORE than that is what
# actually distinguishes "these chains have a real interface" from "these
# two specific CA atoms happen to be close, and nothing else about the
# chains is." 3 is a light bar — a real, if small, interface footprint —
# not an attempt to precisely quantify a genuine binding interface's size.
DEFAULT_CONTACT_THRESHOLD = 8.0
DEFAULT_MIN_CONTACT_PAIRS = 3


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

    This is a deliberately lightweight, single-pair utility — for the
    primary ring-membership + ring-order pipeline, see analyze_assembly_
    rings / run_ring_analysis instead.

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
    grouping, two chains from DIFFERENT components meeting at an interface
    could in principle still have a directed terminal contact within
    tau by coincidence — grouping by sequence first means a spurious
    cross-component pair can never even be considered, rather than relying
    on the terminal-distance graph alone and hoping it doesn't happen.

    It's also the boundary ring detection operates within: each returned
    group gets its own independent terminal graph and cycle search (see
    detect_rings), so a multi-component cage where different proteins form
    different-sized rings (e.g. one forms trimers, another forms dimers)
    is handled correctly rather than assuming one size — or even one
    graph — for the whole assembly.

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


def _directed_terminal_matrix(chain_names, chain_geometry):
    """
    Step 2 of the module docstring's algorithm: the full n x n directed
    terminal-distance matrix for `chain_names`, computed in one vectorized
    pass. Entry [i, j] is d_term(i -> j) = || C_i - N_j || — chain i's
    C-terminus to chain j's N-terminus. The diagonal (i == j, a chain's
    distance to its own termini) is meaningless for a fusion junction and
    is set to +inf so it can never pass a threshold check.

    This is intentionally NOT symmetric — d_term(i -> j) and d_term(j -> i)
    are unrelated numbers in general, since a cyclic interface has a
    direction (see module docstring). Callers that need the reverse
    direction read matrix[j, i], not matrix[i, j] transposed-and-relabeled.
    """
    n = len(chain_names)
    c_coords = np.array([chain_geometry[name]["c"] for name in chain_names])
    n_coords = np.array([chain_geometry[name]["n"] for name in chain_names])
    diffs = c_coords[:, None, :] - n_coords[None, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dist_matrix, np.inf)
    return dist_matrix


def build_terminal_graph(chain_names, chain_geometry, threshold=DEFAULT_TERMINAL_THRESHOLD):
    """
    Step 3: builds the directed interface graph G = (V, E) — one node per
    chain, and a directed edge i -> j wherever d_term(i -> j) <= threshold
    (see _directed_terminal_matrix). Edge weights carry the actual
    distance, for downstream tightness scoring (find_ring_cycles).

    threshold : tau, in Angstroms (see DEFAULT_TERMINAL_THRESHOLD for the
        default and the reasoning behind it).

    Returns a networkx.DiGraph with every chain in `chain_names` present
    as a node even if it ends up with no edges at all (e.g. a chain with
    no terminus close enough to any other chain's — plausible for a
    genuinely non-cyclic or monomeric entry in the identity group), so
    downstream code can always rely on every input chain being a node.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(chain_names)

    dist_matrix = _directed_terminal_matrix(chain_names, chain_geometry)
    n = len(chain_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dist_matrix[i, j]
            if d <= threshold:
                graph.add_edge(chain_names[i], chain_names[j], weight=float(d))

    return graph


def find_ring_cycles(graph, allowed_sizes=ALLOWED_RING_SIZES,
                      max_junction_spread=DEFAULT_JUNCTION_UNIFORMITY_TOLERANCE):
    """
    Step 4: finds every elementary (simple, non-self-intersecting) directed
    cycle in `graph` whose length is one of `allowed_sizes`, using
    networkx's simple_cycles — Johnson's algorithm, bounded by
    length_bound=max(allowed_sizes) so it never wastes time discovering
    cycles longer than the longest physically possible ring.

    This is the crux of size-agnosticism: no candidate ring size is ever
    proposed or tried — a cycle's own length (how many chains it took to
    close the loop back on itself) directly IS the ring size, for
    whichever sizes happen to be geometrically present in `graph`, all in
    one pass.

    max_junction_spread : a candidate cycle is excluded outright if its
        loosest junction minus its tightest junction exceeds this many
        Angstroms — see DEFAULT_JUNCTION_UNIFORMITY_TOLERANCE and the
        module docstring's JUNCTION UNIFORMITY section for the physical
        reasoning (every junction in a genuine C_n ring is the same
        interface repeated by symmetry, so their distances should agree
        with each other, not just each individually clear the
        terminal_threshold gate). None disables this filter — every
        elementary cycle found is kept regardless of internal spread,
        the behavior before this filter existed.

    Each returned candidate is a dict:
      - "chains"        : tuple of chain names in cyclic fusion order,
                           e.g. ("A", "B", "C") meaning A -> B -> C -> A.
      - "edges"          : list of (from_chain, to_chain, nc_distance)
                            for each consecutive link in the cycle,
                            including the closing link back to the start.
      - "ring_size"      : len(chains) — 2/3/4/5, read directly off the
                            cycle, never assumed going in.
      - "mean_distance"  : average of the ring_size junction distances —
                            the tightness metric used to rank and
                            compare candidates, including candidates of
                            DIFFERENT sizes (dividing out the link count
                            is what makes a trimer's 3 links comparable to
                            a pentamer's 5 — see the old module's
                            equivalent reasoning for mean vs. sum).
      - "min_distance"   : the single tightest junction in the ring.
      - "max_distance"   : the single loosest (weakest) junction in the
                            ring — useful for flagging rings where the
                            members aren't all equally well-linked, even
                            if the ring as a whole is accepted.
      - "distance_spread": max_distance - min_distance — the uniformity
                            signal max_junction_spread filters on, kept on
                            every surviving candidate for inspection even
                            when the filter is disabled.

    Sorted tightest-first by mean_distance, so callers (in particular
    _greedy_nonoverlapping_cycles) can walk it in acceptance order
    directly.
    """
    if not allowed_sizes:
        return []

    max_size = max(allowed_sizes)
    allowed = set(allowed_sizes)

    candidates = []
    for cycle in nx.simple_cycles(graph, length_bound=max_size):
        size = len(cycle)
        if size not in allowed:
            continue

        edges = []
        for k in range(size):
            a, b = cycle[k], cycle[(k + 1) % size]
            edges.append((a, b, graph[a][b]["weight"]))

        distances = [e[2] for e in edges]
        min_distance = min(distances)
        max_distance = max(distances)
        spread = max_distance - min_distance

        if max_junction_spread is not None and spread > max_junction_spread:
            continue

        candidates.append({
            "chains": tuple(cycle),
            "edges": edges,
            "ring_size": size,
            "mean_distance": sum(distances) / size,
            "min_distance": min_distance,
            "max_distance": max_distance,
            "distance_spread": spread,
        })

    candidates.sort(key=lambda c: c["mean_distance"])
    return candidates


def _has_ca_coords(chain_names, chain_geometry):
    """
    True if every chain in `chain_names` has a "ca_coords" entry in
    chain_geometry — the full per-residue CA trace termini.
    get_chain_ca_geometry started returning alongside n/c/centroid/
    n_vector/c_vector, specifically so the interface-contact sanity check
    below (_interchain_contact_count / _filter_has_interface_contact)
    doesn't need to re-parse the structure. False if chain_geometry was
    built by an older termini.py that predates that field — callers use
    this to skip the contact check gracefully (with an explanation)
    rather than crashing on a missing key.
    """
    return all(chain_geometry.get(name, {}).get("ca_coords") is not None for name in chain_names)


def _interchain_contact_count(chain_a, chain_b, chain_geometry, contact_threshold):
    """
    Counts how many (residue_in_chain_a, residue_in_chain_b) CA-CA pairs
    fall within `contact_threshold` Angstroms of each other — the full
    per-residue interface footprint between two chains, not just their
    single closest approach.

    This is deliberately a COUNT, not a minimum distance. A minimum
    inter-chain CA-CA distance is not a useful signal here: the two
    chains being checked are exactly the two chains an accepted ring
    junction already links, and that junction's own from_chain C-terminus
    / to_chain N-terminus ARE two of the CA atoms being compared — so the
    minimum is already guaranteed to be at most the junction distance
    that got the ring accepted in the first place. A minimum-distance
    check could never fail for an accepted ring; it would be pure
    overhead with no discriminating power. Counting how many INDEPENDENT
    residue pairs are close, by contrast, can't be satisfied by that one
    already-known-close termini pair alone — see _filter_has_interface_
    contact and DEFAULT_MIN_CONTACT_PAIRS.

    Vectorized over the full (n_a, 3) x (n_b, 3) coordinate arrays — cheap
    for typical chain lengths, and only ever run on the handful of
    candidate rings that already survived the terminal-threshold and
    junction-uniformity filters, not on every possible chain pair.
    """
    coords_a = chain_geometry[chain_a]["ca_coords"]
    coords_b = chain_geometry[chain_b]["ca_coords"]
    diffs = coords_a[:, None, :] - coords_b[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return int(np.count_nonzero(dists <= contact_threshold))


def _filter_has_interface_contact(candidates, chain_geometry, contact_threshold, min_contact_pairs):
    """
    Drops any candidate cycle where at least one of its own junctions'
    two chains fail to show a real, multi-residue interface (see
    _interchain_contact_count) — the mitigation for the failure mode
    observed on 1WPB-2 (long, winding chains whose termini can reach
    across to a chain they don't actually pack against anywhere else).

    Every consecutive pair in the cycle (the same pairs the ring's own
    junctions already connect) is checked independently — not every
    pairwise combination of ring members — since those are exactly the
    pairs the cycle is claiming are genuine ring-mates; a tetramer or
    pentamer's non-adjacent ("diagonal") members are never required to
    directly contact each other.

    A candidate that survives has its per-edge contact counts stashed
    under "min_contact_pairs_found" (the smallest count across its
    junctions) so _cycle_to_ring_result can surface it on the final ring
    result without recomputing it against a possibly different threshold
    later.

    contact_threshold / min_contact_pairs : either being None disables
    this filter entirely (every candidate passes through unchanged) — see
    DEFAULT_CONTACT_THRESHOLD / DEFAULT_MIN_CONTACT_PAIRS.
    """
    if contact_threshold is None or min_contact_pairs is None:
        return candidates

    kept = []
    for candidate in candidates:
        counts = [
            _interchain_contact_count(a, b, chain_geometry, contact_threshold)
            for a, b, _ in candidate["edges"]
        ]
        if min(counts) >= min_contact_pairs:
            candidate["min_contact_pairs_found"] = min(counts)
            kept.append(candidate)
    return kept


def _greedy_nonoverlapping_cycles(cycles):
    """
    Step 5: walks `cycles` (already sorted tightest-first by
    mean_distance — see find_ring_cycles) and accepts EVERY candidate
    that doesn't reuse a chain an earlier (and therefore tighter-or-equal)
    accepted candidate already claimed. That is the entire rule — there is
    no additional "stop once things get too much looser than the first
    ring found" cutoff (see the module docstring's BUGFIX section for why
    an earlier version of this had exactly that cutoff, and why it was
    actively wrong: it conflated ordinary between-ring tightness variation
    in a real structure with genuine noise, and could silently drop a
    real, fully disjoint ring for no reason other than being found later
    in a looser part of the sorted list).

    Assignment is greedy, not globally optimal: each candidate is only
    checked against chains already claimed by a PREVIOUSLY accepted (and
    therefore tighter-or-equal) candidate, so a genuine, tight ring gets
    first claim on its own chains before a looser, overlapping alternative
    has a chance to steal one of them.

    Called once PER CANDIDATE RING SIZE by detect_rings — this only
    resolves overlap WITHIN that one size's own candidates. Whether that
    size's result is trustworthy at all (does it account for every chain
    in the group, with none left over?) is detect_rings' job, not this
    one's — see the module docstring's TOTAL-COVERAGE AXIS SELECTION
    section.

    Returns the accepted candidate dicts (in the same shape they came
    in) — every candidate that did NOT share a chain with an
    already-accepted one.
    """
    assigned = set()
    accepted = []

    for candidate in cycles:
        chains = candidate["chains"]
        if assigned.isdisjoint(chains):
            accepted.append(candidate)
            assigned.update(chains)

    return accepted


def detect_rings(chain_names, chain_geometry, ring_size=None, candidate_ring_sizes=ALLOWED_RING_SIZES,
                  terminal_threshold=DEFAULT_TERMINAL_THRESHOLD,
                  max_junction_spread=DEFAULT_JUNCTION_UNIFORMITY_TOLERANCE,
                  contact_threshold=DEFAULT_CONTACT_THRESHOLD, min_contact_pairs=DEFAULT_MIN_CONTACT_PAIRS):
    """
    Auto-detects every ring findable within one sequence-identity group of
    chains, via the directed-terminal-cycle method (see module docstring):
    build the directed C-to-N interface graph, find every elementary
    cycle whose length is a physically allowed ring size AND whose own
    junction distances are internally consistent (max_junction_spread —
    see JUNCTION UNIFORMITY) AND whose linked chains show a real
    multi-residue interface (contact_threshold / min_contact_pairs — see
    INTERFACE CONTACT SANITY CHECK).

    When auto-detecting (ring_size=None), EVERY size in candidate_ring_
    sizes is tried independently against the WHOLE group: for each size,
    the tightest-first non-overlapping candidates (_greedy_nonoverlapping_
    cycles) are accepted, and that size's result is kept ONLY IF it
    accounts for every single chain in the group — no residue left over.
    A size that leaves even one chain uncovered is discarded ENTIRELY for
    this group (not partially reported). See the module docstring's
    TOTAL-COVERAGE AXIS SELECTION section for the full reasoning: a real,
    designed cyclic subunit's own symmetry axis has to account for the
    WHOLE identity group by definition (every copy of it), while a size
    that only explains SOME of the chains is either the wrong size, or
    reading a coincidental (non-ring-forming) contact as if it were one.

    Because a real T/O/I point-group cage genuinely CAN have more than
    one size simultaneously achieve total coverage of the same group
    (e.g. an octahedral cage's own 2-fold axes, relating chains across
    different trimers, alongside its 3-fold axis within each trimer —
    both are real, both can independently partition every chain), this
    function does NOT pick a single winning size when auto-detecting. It
    returns the UNION of every candidate size that achieves total
    coverage, each candidate still tagged with its own "ring_size" — the
    caller (or a human) decides which axis is the biologically intended
    subunit, e.g. against a trusted oligomeric_count from metadata.

    ring_size : if given, restricts to cycles of exactly this size (must
        be one of ALLOWED_RING_SIZES) and skips the total-coverage
        requirement entirely — every surviving candidate of that size is
        accepted tightest-first, same as before, and leftover chains are
        fine (e.g. genuine crystallographic extras). This remains the
        SAFEST option whenever you already trust the subunit's oligomeric
        state (e.g. oligomeric_count / stoichiometry via query.py's
        fetch_metadata()), since it sidesteps the multi-axis ambiguity
        above entirely rather than resolving it structurally.
        None (default) auto-detects across candidate_ring_sizes.
    candidate_ring_sizes : which sizes may be considered when ring_size is
        None (default ALLOWED_RING_SIZES).
    terminal_threshold : tau, passed to build_terminal_graph — see
        DEFAULT_TERMINAL_THRESHOLD.
    max_junction_spread : passed to find_ring_cycles — see
        DEFAULT_JUNCTION_UNIFORMITY_TOLERANCE. Set to None to disable
        this filter and accept any cycle within terminal_threshold
        regardless of how much its own junction distances vary.
    contact_threshold, min_contact_pairs : the interface-contact sanity
        check (see _filter_has_interface_contact and the module
        docstring's INTERFACE CONTACT SANITY CHECK section) — every
        candidate's own junctions must show at least min_contact_pairs
        independent CA-CA pairs within contact_threshold Å between the
        two chains, beyond the one termini pair that got the junction
        candidate in the first place. Either being None disables this
        check. Silently skipped (with a printed note) if chain_geometry
        lacks the "ca_coords" field this check needs — see termini.
        get_chain_ca_geometry.

    Returns a flat list of accepted ring-cycle dicts (see
    find_ring_cycles) — possibly of MIXED ring_size when auto-detecting
    (see above; check each candidate's own "ring_size" field to tell them
    apart). Pass each to _cycle_to_ring_result for the fully annotated,
    DataFrame-ready form, or just use analyze_assembly_rings, which does
    this for you.

    Returns [] if the group has too few chains, if no cycle at all
    survived every filter, or (when auto-detecting) if NO candidate size
    managed to account for every chain in the group — see the printed
    per-size summary for which sizes were found but discarded, and why
    (also see the module docstring's LIMITATIONS section: a genuinely
    correct size that's merely under-detected by a chain or two is
    discarded the same as a genuinely wrong one under this rule).
    """
    sizes = (ring_size,) if ring_size is not None else tuple(candidate_ring_sizes)
    invalid = sorted(set(sizes) - set(ALLOWED_RING_SIZES))
    if invalid:
        raise ValueError(
            f"ring size(s) must be one of {ALLOWED_RING_SIZES} — Platonic-solid "
            f"(T/O/I point-group) cages only have 2-, 3-, 4-, and 5-fold "
            f"rotational symmetry axes, so no other ring size is physically "
            f"possible — got invalid size(s) {invalid}"
        )

    if len(chain_names) < min(sizes):
        return []

    graph = build_terminal_graph(chain_names, chain_geometry, threshold=terminal_threshold)
    cycles = find_ring_cycles(graph, allowed_sizes=sizes, max_junction_spread=max_junction_spread)

    if contact_threshold is not None and min_contact_pairs is not None:
        if _has_ca_coords(chain_names, chain_geometry):
            cycles = _filter_has_interface_contact(cycles, chain_geometry, contact_threshold, min_contact_pairs)
        else:
            print(
                "detect_rings: chain_geometry has no 'ca_coords' field (built by a "
                "termini.py older than the interface-contact sanity check) — skipping "
                "that check for this group. Update termini.get_chain_ca_geometry to "
                "enable it, or pass contact_threshold=None to silence this message."
            )

    if not cycles:
        return []

    if ring_size is not None:
        # Explicit size: no total-coverage requirement, no competition
        # against other sizes — accept tightest-first, same as before.
        return _greedy_nonoverlapping_cycles(cycles)

    # Auto-detect: run EACH candidate size's own non-overlapping
    # acceptance independently against the WHOLE group, and keep a
    # size's result only if it leaves no chain uncovered — see this
    # function's docstring and the module docstring's TOTAL-COVERAGE
    # AXIS SELECTION section. Sizes are reported side by side, not
    # competed against each other: more than one can genuinely be a
    # real, total symmetry axis of the same assembly at once.
    n_chains = len(chain_names)
    by_size = {}
    for c in cycles:
        by_size.setdefault(c["ring_size"], []).append(c)

    all_accepted = []
    summary = []
    for size in sorted(by_size):
        accepted_s = _greedy_nonoverlapping_cycles(by_size[size])
        covered = {chain for cyc in accepted_s for chain in cyc["chains"]}
        if len(covered) == n_chains:
            all_accepted.extend(accepted_s)
            summary.append(f"size={size}: VALID — {len(accepted_s)} ring(s), covers all {n_chains} chains")
        else:
            summary.append(
                f"size={size}: discarded — {len(accepted_s)} ring(s) found but "
                f"{n_chains - len(covered)} of {n_chains} chains left uncovered "
                f"(not a total symmetry axis for this group)"
            )

    print(f"detect_rings: per-size axis check for this {n_chains}-chain group — " + "; ".join(summary))

    return all_accepted


def find_shortest_ring_junction(ring_chain_names, chain_geometry):
    """
    Given the chain names in one ring (from whatever source — detect_rings,
    or already known some other way, e.g. trusted metadata), finds the
    SHORTEST single N-to-C junction distance between any two DIFFERENT
    chains in that ring, by brute-force checking every ordered pair.

    This is kept as a standalone utility for the case where you already
    have ring MEMBERSHIP from elsewhere and just want its best candidate
    fusion pair, without needing to know or trust any particular chain
    ORDER. The primary pipeline (analyze_assembly_rings) doesn't need
    this brute-force scan — detect_rings' cycles already carry the true,
    directed fusion order for every consecutive pair, which is strictly
    more informative — but this remains useful on its own, e.g. against a
    ring membership list sourced from metadata rather than geometry.

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


def _cycle_to_ring_result(cycle, chain_geometry):
    """
    Converts one accepted cycle (from detect_rings / find_ring_cycles)
    into the fully annotated ring-result dict analyze_assembly_rings
    returns, carrying strictly more information than the old module's
    find_shortest_ring_junction-based result did — the whole ordered
    fusion path is known now, not just its single tightest link.

    Fields:
      - chain_order  : chains in cyclic fusion order (cycle["chains"]).
      - ring_size    : as detected — len(chain_order).
      - junctions    : every consecutive directed link in the ring, in
                        order, as {"from_chain", "to_chain", "nc_distance"}
                        dicts — the complete fusion path, including the
                        closing link back to the first chain.
      - from_chain, to_chain, nc_distance : the single TIGHTEST junction
                        in the ring, kept at the top level for backward
                        compatibility with consumers (e.g.
                        orientation.annotate_ring_orientation) built
                        around "one representative junction per ring",
                        and because it's still a reasonable answer to
                        "what's this ring's best/most confident single
                        fusion candidate".
      - mean_nc_distance, weakest_junction_distance : overall ring
                        tightness, and its single loosest link — useful
                        for flagging a ring where fusion may be uneven
                        across its junctions even though the ring as a
                        whole was accepted.
      - junction_spread : weakest_junction_distance minus this ring's
                        tightest junction (nc_distance) — the same
                        uniformity signal find_ring_cycles' max_junction_
                        spread filters candidates on (see the module
                        docstring's JUNCTION UNIFORMITY section), kept
                        here so a surviving ring's own consistency is
                        still visible even when the filter passed it (or
                        was disabled).
      - min_contact_pairs : the fewest independent CA-CA contact pairs
                        (see _interchain_contact_count) found across this
                        ring's own junctions, when the interface-contact
                        sanity check ran (see detect_rings' contact_
                        threshold / min_contact_pairs) — None if that
                        check was disabled or skipped (e.g. chain_geometry
                        has no "ca_coords"). Every accepted ring already
                        cleared min_contact_pairs when the check ran; this
                        is how much margin it had, not a pass/fail flag.
      - fingerprint  : sorted tuple of pairwise centroid distances between
                        members — unchanged in meaning from before, still
                        what deduplicate_rings_by_geometry compares to
                        recognize physically equivalent rings. Present on
                        this dict for that comparison to use, but
                        analyze_assembly_rings drops it from its returned
                        rings once deduplication is done — it's an
                        internal comparison key, not user-facing output.
    """
    edges = cycle["edges"]
    tightest = min(edges, key=lambda e: e[2])
    weakest = max(edges, key=lambda e: e[2])
    chain_order = list(cycle["chains"])

    return {
        "chain_order": chain_order,
        "ring_size": cycle["ring_size"],
        "junctions": [
            {"from_chain": a, "to_chain": b, "nc_distance": round(d, 2)}
            for a, b, d in edges
        ],
        "from_chain": tightest[0],
        "to_chain": tightest[1],
        "nc_distance": round(tightest[2], 2),
        "mean_nc_distance": round(cycle["mean_distance"], 2),
        "weakest_junction_distance": round(weakest[2], 2),
        "junction_spread": round(weakest[2] - tightest[2], 2),
        "min_contact_pairs": cycle.get("min_contact_pairs_found"),
        "fingerprint": tuple(sorted(
            float(np.linalg.norm(chain_geometry[a]["centroid"] - chain_geometry[b]["centroid"]))
            for a, b in combinations(chain_order, 2)
        )),
    }


def deduplicate_rings_by_geometry(rings, tolerance=1.5):
    """
    Collapses rings that are the SAME physical ring — related by the
    assembly's own symmetry — into one representative row. This is the
    redundancy check: a tetrahedral cage built from one identity group has
    4 physically equivalent trimers, and there's no reason to report all
    4 as if they were 4 different findings, just as there's no reason to
    merge two DIFFERENT rings that happen to look similar by one number.

    Equivalence is judged by comparing each ring's FULL shape — its
    "fingerprint" from _cycle_to_ring_result / find_shortest_ring_junction
    (the sorted tuple of every pairwise centroid distance between its
    members) — element by element, within `tolerance` Angstroms, rather
    than by comparing just the single nc_distance number each ring
    reports. A single-number comparison fails in both directions:

      - Two DIFFERENT rings can end up with a similar nc_distance by
        coincidence — nc_distance is only the SHORTEST junction within a
        ring, and plenty of unrelated ring shapes could happen to share a
        similarly-short one. Comparing nc_distance alone risks merging
        two genuinely non-equivalent rings into one.
      - Two truly EQUIVALENT (symmetry-related) rings can end up with
        different nc_distance if real biological asymmetry (see module
        docstring) happens to make a different chain pair the "tightest"
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


def _format_junctions(junctions):
    """Renders the "junctions" list ([{"from_chain", "to_chain",
    "nc_distance"}, ...] — see _cycle_to_ring_result) as one compact,
    fully-readable string: "A->B: 2.51Å; B->C: 2.48Å; C->A: 2.60Å". A raw
    list-of-dicts is what a DataFrame cell held before this — pandas has
    no special rendering for that, so it either printed the Python repr
    truncated mid-value or hid it behind "...", with no way to actually
    read the individual Angstrom numbers off the table. This is plain
    text specifically so every junction distance in the ring is visible
    directly in the DataFrame, not just accessible by drilling into a
    nested object."""
    return "; ".join(f"{j['from_chain']}->{j['to_chain']}: {j['nc_distance']}Å" for j in junctions)


def _format_equivalent_rings(equivalent_rings):
    """Renders "equivalent_rings" (added by deduplicate_rings_by_geometry
    — a list of chain-order lists, one per physical copy folded into this
    representative) as one readable string: "[A,B,C]; [D,E,F]" — same
    rationale as _format_junctions."""
    return "; ".join(f"[{','.join(chains)}]" for chains in equivalent_rings)


# Ring-dict fields that are lists needing their own readable string
# rendering (see _format_junctions / _format_equivalent_rings) rather
# than the generic dict-flattening or raw pass-through _flatten_ring_row
# otherwise applies.
_LIST_FIELD_FORMATTERS = {
    "junctions": _format_junctions,
    "equivalent_rings": _format_equivalent_rings,
}


def _flatten_ring_row(ring):
    """
    Flattens one ring dict into a single-level dict suitable for a
    DataFrame row. Any value that's itself a dict (e.g. the "orientation"
    field an annotator adds) gets its keys pulled up one level with the
    parent key as a prefix — "orientation": {"backbone_angle": 12.4, ...}
    becomes "orientation_backbone_angle": 12.4 — so results land as plain
    sortable/filterable columns instead of a dict blob per cell that has
    to be unpacked before it's usable.

    A list value with a registered formatter (_LIST_FIELD_FORMATTERS —
    currently "junctions" and "equivalent_rings") is rendered as a single
    readable string rather than kept as a raw Python list: a DataFrame
    cell holding a list of dicts has no useful display of its own, and
    the whole point of this field existing is to show every junction's
    actual Angstrom distance, not hide it behind an unreadable object.
    Any OTHER list-valued field (there are none built into this module
    today, but a future annotator could add one) is passed through
    unchanged rather than guessed at.

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
        elif isinstance(value, list) and key in _LIST_FIELD_FORMATTERS:
            row[key] = _LIST_FIELD_FORMATTERS[key](value)
        else:
            row[key] = value
    return row


def analyze_assembly_rings(filepath, assembly_id, ring_size=None, candidate_ring_sizes=ALLOWED_RING_SIZES,
                            terminal_threshold=DEFAULT_TERMINAL_THRESHOLD,
                            max_junction_spread=DEFAULT_JUNCTION_UNIFORMITY_TOLERANCE,
                            contact_threshold=DEFAULT_CONTACT_THRESHOLD,
                            min_contact_pairs=DEFAULT_MIN_CONTACT_PAIRS,
                            tolerance=1.5, annotators=None):
    """
    Full per-assembly ring pipeline: load the structure, compute chain
    geometry, group chains by sequence identity (the two-component
    safeguard — chains from DIFFERENT components/sequences are never
    considered for the same ring; see group_chains_by_sequence), and —
    per identity group — find every ring AND its correct fusion order via
    the directed terminal-cycle method (see module docstring and
    detect_rings).

    ring_size : the oligomeric count for the subunit you're targeting (3
        for a trimer, etc), if you already trust it — e.g. from
        oligomeric_count / stoichiometry via query.py's fetch_metadata().
        When given, it must be one of ALLOWED_RING_SIZES and restricts
        detect_rings to cycles of exactly that length for every identity
        group, with no total-coverage requirement (leftover chains are
        fine — see the "leftover" warning below).

        Left as None (the default), EVERY size in candidate_ring_sizes is
        tried, independently, against each identity group, and ALL sizes
        that fully account for that group's chains (no residue left
        over) are returned side by side — not just one "winning" size.
        See detect_rings and the module docstring's TOTAL-COVERAGE AXIS
        SELECTION section: a real point-group cage can genuinely have
        more than one true, total symmetry axis (e.g. an octahedral
        cage's 3-fold AND 2-fold axes), and this module no longer guesses
        which one is "the" designed subunit — every ring in the output
        carries its own "ring_size" field so you (or a downstream step)
        can filter to the one you actually want, e.g. by matching a
        trusted oligomeric_count from metadata.

    candidate_ring_sizes : which sizes may be considered (default
        ALLOWED_RING_SIZES). Only relevant when ring_size=None.

    terminal_threshold : tau, in Angstroms — passed straight through to
        build_terminal_graph for every identity group. See
        DEFAULT_TERMINAL_THRESHOLD.

    max_junction_spread : in Angstroms — passed straight through to
        find_ring_cycles for every identity group. See
        DEFAULT_JUNCTION_UNIFORMITY_TOLERANCE and the module docstring's
        JUNCTION UNIFORMITY section; None disables this filter.

    contact_threshold, min_contact_pairs : passed straight through to
        detect_rings for every identity group — see
        DEFAULT_CONTACT_THRESHOLD / DEFAULT_MIN_CONTACT_PAIRS and the
        module docstring's INTERFACE CONTACT SANITY CHECK section. Either
        being None disables this check.

    tolerance : in Angstroms — passed straight through to
        deduplicate_rings_by_geometry only (ring ACCEPTANCE no longer
        uses a tolerance cutoff at all — see the module docstring's
        BUGFIX and TOLERANCE vs. TERMINAL_THRESHOLD sections).

    Rings that are the SAME physical ring (within `tolerance` Angstroms
    of each other, judged by full shape — see deduplicate_rings_by_geometry)
    are collapsed to one representative row, so the result doesn't repeat
    the same ring once per symmetric copy. Rings of DIFFERENT ring_size
    are never collapsed into each other (see deduplicate_rings_by_geometry).
    The full chain lists for every equivalent ring are still available via
    the representative's "equivalent_rings" field.

    Returns {"assembly_id": ..., "rings": [ring_result, ...]} — one row
    per DISTINCT ring found (within tolerance), across every identity
    group AND every ring_size that achieved total coverage for its group
    when auto-detecting — check each row's "ring_size" to tell candidate
    axes apart. Each ring_result is the _cycle_to_ring_result dict (see
    that function), MINUS "fingerprint" — used only internally by
    deduplicate_rings_by_geometry to recognize physically equivalent
    rings, and dropped once that job is done, since it's not something a
    user picking a subunit to build with needs to see in the output.

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
    sizes = (ring_size,) if ring_size is not None else tuple(candidate_ring_sizes)
    invalid = sorted(set(sizes) - set(ALLOWED_RING_SIZES))
    if invalid:
        raise ValueError(
            f"ring size(s) must be one of {ALLOWED_RING_SIZES} — Platonic-solid "
            f"(T/O/I point-group) cages only have 2-, 3-, 4-, and 5-fold "
            f"rotational symmetry axes, so no other ring size is physically "
            f"possible — got invalid size(s) {invalid}"
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
        if len(usable) < min(sizes):
            continue

        accepted = detect_rings(
            usable, chain_geometry, ring_size=ring_size, candidate_ring_sizes=candidate_ring_sizes,
            terminal_threshold=terminal_threshold, max_junction_spread=max_junction_spread,
            contact_threshold=contact_threshold, min_contact_pairs=min_contact_pairs,
        )

        if not accepted:
            print(
                f"analyze_assembly_rings: no ring of size {list(sizes)} found within "
                f"terminal_threshold={terminal_threshold}Å for an identity group of "
                f"{len(usable)} chains in assembly {assembly_id} — skipping this group. "
                f"(When auto-detecting, this also happens if candidate sizes were FOUND but "
                f"none achieved total coverage of the group — see detect_rings' printed "
                f"per-size summary above.)"
            )
            continue

        # Only meaningful when ring_size was given explicitly: detect_rings'
        # auto-detect path (ring_size=None) never returns a size whose
        # accepted rings leave chains uncovered (that's the total-coverage
        # requirement — see detect_rings), so `leftover` is always empty
        # here in auto mode. With an explicit ring_size, no such
        # requirement applies, so leftover chains are worth flagging.
        assigned = {c for cycle in accepted for c in cycle["chains"]}
        leftover = [name for name in usable if name not in assigned]
        if len(leftover) >= min(sizes):
            print(
                f"analyze_assembly_rings: {len(leftover)} chains in assembly {assembly_id} "
                f"were left unassigned after accepting {len(accepted)} ring(s) — {leftover}. "
                f"That's enough chains to plausibly form another ring; if one is expected "
                f"here, check whether their termini actually fall within "
                f"terminal_threshold={terminal_threshold}Å of each other at all (a real but "
                f"unusually spaced junction can need a larger terminal_threshold to be found "
                f"as an edge in the first place — see the module docstring)."
            )

        for cycle in accepted:
            ring = _cycle_to_ring_result(cycle, chain_geometry)
            rings.append(ring)

    rings = deduplicate_rings_by_geometry(rings, tolerance=tolerance)

    # "fingerprint" (sorted pairwise centroid distances) exists purely so
    # deduplicate_rings_by_geometry can recognize physically equivalent
    # rings just above — it's an internal comparison key, not a number a
    # user would ever want to read off the output table, so it's dropped
    # here once its one job is done.
    for ring in rings:
        ring.pop("fingerprint", None)

    if annotators:
        rings = [_apply_annotators(ring, chain_geometry, annotators) for ring in rings]

    return {"assembly_id": assembly_id, "rings": rings}


def run_ring_analysis(df, ring_size=None, filepath_column="filepath", assembly_id_column="assembly_id",
                       candidate_ring_sizes=ALLOWED_RING_SIZES, terminal_threshold=DEFAULT_TERMINAL_THRESHOLD,
                       max_junction_spread=DEFAULT_JUNCTION_UNIFORMITY_TOLERANCE,
                       contact_threshold=DEFAULT_CONTACT_THRESHOLD,
                       min_contact_pairs=DEFAULT_MIN_CONTACT_PAIRS,
                       tolerance=1.5, annotators=None):
    """
    Runs analyze_assembly_rings across every row of a candidates DataFrame,
    flattening the (possibly several) DISTINCT rings found per assembly
    into one row per ring — so results sort/filter like a normal table,
    with the best (lowest nc_distance) rings naturally on top regardless
    of which assembly they came from.

    ring_size : None (default — try every size in candidate_ring_sizes
        independently for every assembly and keep ALL sizes that achieve
        total coverage of their identity group, side by side; see
        analyze_assembly_rings and detect_rings' TOTAL-COVERAGE AXIS
        SELECTION), a fixed integer from ALLOWED_RING_SIZES (same size
        for every row), or the NAME of a column in df to look up per-row
        — e.g. "oligomeric_count", if you fetched that via query.py's
        fetch_metadata() and merged it in. Useful if one batch mixes
        candidates with different, already-known oligomeric states. When
        auto-detecting, filter the resulting DataFrame's "ring_size"
        column afterward if you only want one particular axis per
        assembly.
    candidate_ring_sizes : passed straight through to analyze_assembly_
        rings for every assembly in the batch — only relevant when
        auto-detecting (ring_size is None; a per-row column always
        supplies an explicit size).
    terminal_threshold : tau, in Angstroms — passed straight through to
        every assembly in the batch. See DEFAULT_TERMINAL_THRESHOLD.
    max_junction_spread : in Angstroms, passed straight through to every
        assembly in the batch. See DEFAULT_JUNCTION_UNIFORMITY_TOLERANCE;
        None disables the filter.
    contact_threshold, min_contact_pairs : passed straight through to
        every assembly in the batch. See DEFAULT_CONTACT_THRESHOLD /
        DEFAULT_MIN_CONTACT_PAIRS; either being None disables the check.
    tolerance : in Angstroms, passed straight through to analyze_assembly_
        rings for every assembly in the batch, where it's used only for
        deduplicate_rings_by_geometry (ring acceptance itself has no
        tolerance cutoff — see the module docstring's BUGFIX section).
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
                candidate_ring_sizes=candidate_ring_sizes, terminal_threshold=terminal_threshold,
                max_junction_spread=max_junction_spread, contact_threshold=contact_threshold,
                min_contact_pairs=min_contact_pairs,
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