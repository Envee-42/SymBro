"""
orientation.py — relative terminal orientation between fusion-candidate
termini.

Built on the n_vector / c_vector fields in chain_geometry (see
termini.get_chain_ca_geometry): local backbone direction vectors
extrapolated past each terminus, from a short window of nearby CA
coordinates. Used to judge whether two termini picked for a fusion
junction (typically from_chain/to_chain out of a distance.py ring result)
are already "aimed" at each other — a short, simple linker likely
suffices — or pointing away from each other, which usually means a
longer or more rigid linker is needed to bridge the gap.
"""

import numpy as np


def _angle_between(v1, v2):
    """Angle in degrees between two vectors, 0-180."""
    cos_angle = np.clip(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0
    )
    return np.degrees(np.arccos(cos_angle))


def compute_relative_orientation(chain_geometry, from_chain, to_chain):
    """
    Computes the relative orientation of a fusion junction: from_chain's
    C-terminus to to_chain's N-terminus.

    Returns three angles (degrees, 0-180):
      - backbone_angle : angle between from_chain's c_vector and
        to_chain's n_vector directly. How parallel (near 0) or
        antiparallel (near 180) the two termini's local trajectories are
        to each other, independent of where they actually sit in space.
      - from_alignment : angle between from_chain's c_vector and the
        displacement vector pointing from from_chain's C-terminus to
        to_chain's N-terminus. Small = from_chain is already heading
        toward the junction point; large = it's heading away, and the
        linker will have to double back on itself to reach the target.
      - to_alignment : the mirror of from_alignment on to_chain's side —
        angle between to_chain's n_vector and the displacement vector
        pointing from to_chain's N-terminus back to from_chain's
        C-terminus.

    Low values on all three generally mark the easiest junctions: the two
    termini already point at each other and along the same line. A low
    backbone_angle but high alignment values means the two termini are
    parallel to each other but offset to the side — same idea, worth
    checking case by case once you're picking real candidates.

    Returns None if either chain is missing from chain_geometry, or if
    either lacks a usable vector (e.g. a single-residue chain).
    """
    from_geo = chain_geometry.get(from_chain)
    to_geo = chain_geometry.get(to_chain)

    if from_geo is None or to_geo is None:
        return None
    if from_geo.get("c_vector") is None or to_geo.get("n_vector") is None:
        return None

    displacement = to_geo["n"] - from_geo["c"]
    disp_norm = np.linalg.norm(displacement)
    if disp_norm == 0:
        return None  # coincident points — angle to displacement is undefined

    return {
        "from_chain": from_chain,
        "to_chain": to_chain,
        "backbone_angle": round(_angle_between(from_geo["c_vector"], to_geo["n_vector"]), 1),
        "from_alignment": round(_angle_between(from_geo["c_vector"], displacement), 1),
        "to_alignment": round(_angle_between(to_geo["n_vector"], -displacement), 1),
    }


def annotate_ring_orientation(ring, chain_geometry):
    """
    Convenience wrapper for a distance.py ring result (from
    find_shortest_ring_junction / analyze_assembly_rings, which already
    carries from_chain/to_chain for the chosen junction): computes
    compute_relative_orientation for that specific junction and returns a
    new dict with the ring's existing fields plus "orientation" added.

    Leaves the input ring unmodified. "orientation" is None if the
    vectors aren't available for this pair (see compute_relative_orientation).
    """
    orientation = compute_relative_orientation(
        chain_geometry, ring["from_chain"], ring["to_chain"]
    )
    return {**ring, "orientation": orientation}


def plot_ring_orientation(chain_geometry, ring_chain_names, from_chain=None, to_chain=None, vector_length=5.0):
    """
    Quick 3D sanity-check plot of a ring's terminus positions and their
    n_vector/c_vector directions — matplotlib, static. A development tool
    for eyeballing whether the vectors look geometrically sane while
    you're building this out, NOT the production visualization.

    Once the NiceGUI interface exists, this should be swapped for an
    interactive molecular viewer (py3Dmol or NGL) embedded in the app —
    those can render the actual structure alongside the vectors and let
    the user rotate/select chains. matplotlib's 3D backend can't do
    interactive picking and doesn't scale well past a handful of chains,
    so it's a debugging aid here, not the long-term answer.

    ring_chain_names : chain names belonging to one ring (e.g. a
        distance.py ring result's "chain_order").
    from_chain, to_chain : optional — if given, draws a dashed line
        between that junction's C-terminus and N-terminus points so the
        specific pair being evaluated is visually obvious rather than
        left implicit among all the ring's termini.
    vector_length : Angstroms to draw the (unit-length) vectors at, purely
        for visibility — doesn't affect the underlying data.

    Returns the matplotlib Figure (caller decides whether to show/save it
    — keeps this usable both interactively and in a script/notebook).
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10.colors
    for i, name in enumerate(ring_chain_names):
        geo = chain_geometry.get(name)
        if geo is None:
            continue
        color = colors[i % len(colors)]

        ax.scatter(*geo["n"], color=color, marker="o", s=40, label=f"{name} N-term")
        ax.scatter(*geo["c"], color=color, marker="s", s=40, label=f"{name} C-term")

        if geo.get("n_vector") is not None:
            ax.quiver(*geo["n"], *(geo["n_vector"] * vector_length), color=color, linestyle="dotted")
        if geo.get("c_vector") is not None:
            ax.quiver(*geo["c"], *(geo["c_vector"] * vector_length), color=color)

    if from_chain is not None and to_chain is not None:
        c_pt = chain_geometry[from_chain]["c"]
        n_pt = chain_geometry[to_chain]["n"]
        ax.plot(*zip(c_pt, n_pt), color="black", linestyle="--", linewidth=1.5, label="junction")

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title("Ring terminus orientation")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig
