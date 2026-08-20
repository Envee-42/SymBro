"""
test_rings_selection.py -- direct unit coverage for the two closest-
termini-wins fixes in rings.py: _closest_direction_per_chain_set (within
one chain SET, prefers the shorter of two possible cyclic directions) and
select_disjoint_groupings's mean_distance-first sort (across DIFFERENT
chain sets competing for the same chains, prefers the shorter-junction
grouping over a tighter-CV-but-farther one). Both were added/changed in
response to a real, reported bug on PDB entry 4EGG: a directed-graph
cycle search over every C->N edge in range finds both the true forward
ring and a "skip past your real neighbor" reverse reading of the exact
same chains, and -- one level up -- can also find an entirely different,
still-self-consistent partition of the same chain pool. CV alone can't
tell either apart from the truth; distance can, per this project's own
fusion-design goal (closest termini get fused). Real code paths
throughout -- no mocking, since neither function crosses an external I/O
boundary.
"""
import numpy as np
import pytest

from toolkit.geometry import rings


# ----------------------------------------------------------------------
# _closest_direction_per_chain_set -- within one chain set, direction only
# ----------------------------------------------------------------------

def _candidate(chains, mean_distance):
    return {"chains": chains, "chain_set": frozenset(chains), "mean_distance": mean_distance}


def test_closest_direction_keeps_shorter_of_two_directions_for_same_chain_set():
    forward = _candidate(("A", "B", "C"), 10.0)
    reverse = _candidate(("A", "C", "B"), 30.0)
    kept = rings._closest_direction_per_chain_set([forward, reverse])
    assert kept == [forward]


def test_closest_direction_order_independent():
    """Same two candidates, reverse input order -- same winner either way,
    since the comparison is per chain_set, not first-seen-wins."""
    forward = _candidate(("A", "B", "C"), 10.0)
    reverse = _candidate(("A", "C", "B"), 30.0)
    kept = rings._closest_direction_per_chain_set([reverse, forward])
    assert kept == [forward]


def test_closest_direction_leaves_distinct_chain_sets_untouched():
    """The common case: a chain set with only one passing candidate is
    never touched, and unrelated chain sets don't interact."""
    one = _candidate(("A", "B", "C"), 10.0)
    other = _candidate(("D", "E", "F"), 12.0)
    kept = rings._closest_direction_per_chain_set([one, other])
    assert {c["chains"] for c in kept} == {("A", "B", "C"), ("D", "E", "F")}


def test_find_cyclic_groupings_drops_reverse_direction_real_geometry():
    """End-to-end through find_cyclic_groupings (not just the dedup helper
    in isolation): a 3-chain ring laid out on a line so the forward
    C->N->C->N->C->N traversal is a tight 10 A step and the reverse
    traversal of the SAME three chains is an equally homogeneous (std=0)
    but 3x longer 30 A step. Only the 10 A direction should survive.
    """
    geometry = {
        "A": {"n": np.array([50.0, 0, 0]), "c": np.array([0.0, 0, 0]), "ca_coords": np.array([[0.0, 0, 0]])},
        "B": {"n": np.array([10.0, 0, 0]), "c": np.array([20.0, 0, 0]), "ca_coords": np.array([[10.0, 0, 0]])},
        "C": {"n": np.array([30.0, 0, 0]), "c": np.array([40.0, 0, 0]), "ca_coords": np.array([[30.0, 0, 0]])},
    }
    candidates = rings.find_cyclic_groupings(["A", "B", "C"], geometry, 3, contact_cutoff=None)
    assert len(candidates) == 1
    assert candidates[0]["chains"] == ("A", "B", "C")
    assert candidates[0]["mean_distance"] == 10.0


# ----------------------------------------------------------------------
# select_disjoint_groupings -- across different chain sets, distance-first
# ----------------------------------------------------------------------

def _grouping(chains, mean_distance, std_distance, cv):
    return {
        "chains": chains, "chain_set": frozenset(chains), "mean_distance": mean_distance,
        "std_distance": std_distance, "cv": cv,
    }


def test_select_disjoint_groupings_prefers_closer_junction_over_tighter_cv():
    """Mirrors the real 4EGG shape after the direction fix: a correct,
    physically-closer trimer (25.79 A, cv=0.0066) competes for chain 'F'
    with a farther, cross-trimer candidate (58.40 A) that happens to have
    a tighter CV (0.0031). Distance-first sorting must pick the correct
    trimer, not the tighter-CV one.
    """
    correct = _grouping(("D", "E", "F"), 25.79, 0.17, 0.0066)
    cross = _grouping(("A", "F", "B"), 58.40, 0.18, 0.0031)  # tighter CV, but shares chain F
    accepted = rings.select_disjoint_groupings([cross, correct])
    assert accepted == [correct]


def test_select_disjoint_groupings_accepts_all_when_chains_disjoint():
    a = _grouping(("D", "E", "F"), 25.79, 0.17, 0.0066)
    b = _grouping(("A", "C", "B"), 26.79, 1.65, 0.0614)
    accepted = rings.select_disjoint_groupings([b, a])
    # both disjoint (no shared chains) -- both accepted, closer one first
    assert [g["chains"] for g in accepted] == [("D", "E", "F"), ("A", "C", "B")]


def test_select_disjoint_groupings_std_and_cv_still_break_distance_ties():
    tighter = _grouping(("A", "B", "C"), 20.0, 0.05, 0.0025)
    looser = _grouping(("D", "E", "F"), 20.0, 0.5, 0.025)
    accepted = rings.select_disjoint_groupings([looser, tighter])
    # equal mean_distance, disjoint chains -- order doesn't affect
    # acceptance here (both fit), but confirms std/cv tiebreak doesn't
    # error out and ordering is deterministic when distance ties.
    assert {g["chains"] for g in accepted} == {("A", "B", "C"), ("D", "E", "F")}
    assert accepted[0]["chains"] == ("A", "B", "C")  # tighter std/cv sorts first on a tie


def test_select_disjoint_groupings_handles_none_cv_without_crashing():
    with_cv = _grouping(("A", "B", "C"), 10.0, 0.0, 0.0)
    none_cv = _grouping(("D", "E", "F"), 10.0, 0.0, None)
    accepted = rings.select_disjoint_groupings([none_cv, with_cv])
    assert {g["chains"] for g in accepted} == {("A", "B", "C"), ("D", "E", "F")}
