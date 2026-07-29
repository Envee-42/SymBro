"""
termini.py — per-chain terminus and CA coordinate extraction.

Single shared entry point for pulling N-terminal CA, C-terminal CA,
centroid, and local terminus-direction vectors out of a chain. Everything
downstream (NC distance, ring finding, relative terminal orientation, and
later secondary structure / solvent accessibility) is built on top of
this one function, so the structure only ever gets scanned once per
chain.
"""

import numpy as np


def _unit_vector(v):
    """Normalizes a vector to unit length. Returns None for a zero vector
    (degenerate case — e.g. two coincident CA coordinates)."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return None
    return v / norm


def get_chain_ca_geometry(chain, vector_window=4):
    """
    Single pass over one chain's polymer: collects every alpha-carbon (CA)
    coordinate in residue order, then derives the N-terminal CA, C-terminal
    CA, centroid, and local terminus-direction vectors from that one
    collected list.

    This replaces the original three-pass approach (a forward scan for the
    N-terminal CA, a REVERSED scan for the C-terminal CA, then a third scan
    to collect coordinates for the centroid) with a single scan — each
    residue's CA atom is looked up exactly once instead of up to three
    times. Since residues in a polymer are already stored in sequence
    order, the first resolved CA IS the N-terminal one and the last
    resolved CA IS the C-terminal one — no need to scan twice from opposite
    ends to find them separately.

    vector_window : how many resolved residues in from each terminus to
        use for the direction vector (default 4). n_vector is the unit
        vector from the CA at position `vector_window` to the N-terminal
        CA — i.e. the direction the chain would continue if extrapolated
        PAST the N-terminus. c_vector is the mirror image at the other
        end: from the CA `vector_window` positions before the C-terminal
        CA, to the C-terminal CA itself — the direction the chain is
        exiting AT the C-terminus. Both are local (a handful of residues),
        not whole-chain, so they capture the terminus's own trajectory
        rather than the chain's overall fold. If the chain has fewer than
        vector_window+1 resolved CAs, the largest available window is used
        instead of failing outright.

    Returns None if the chain has no polymer residues at all, or none with
    a resolved CA atom (e.g. a ligand-only chain, or a polymer with fully
    missing backbone density). n_vector/c_vector within the returned dict
    are None (rather than the whole result) if there's only a single
    resolved CA — direction is undefined with just one point.
    """
    polymer = chain.get_polymer()
    if len(polymer) == 0:
        return None

    ca_coords = []
    for res in polymer:
        ca = res.find_atom("CA", "*")
        if ca:
            ca_coords.append([ca.pos.x, ca.pos.y, ca.pos.z])

    if not ca_coords:
        return None

    ca_coords = np.array(ca_coords)
    n_points = len(ca_coords)

    if n_points > 1:
        n_window = min(vector_window, n_points - 1)
        c_window = min(vector_window, n_points - 1)
        n_vector = _unit_vector(ca_coords[0] - ca_coords[n_window])
        c_vector = _unit_vector(ca_coords[-1] - ca_coords[-1 - c_window])
    else:
        n_vector = None
        c_vector = None

    return {
        "n": ca_coords[0],           # first resolved CA in sequence = N-terminus
        "c": ca_coords[-1],          # last resolved CA in sequence = C-terminus
        "centroid": ca_coords.mean(axis=0),
        "n_vector": n_vector,        # unit vector, direction extrapolated past N-term
        "c_vector": c_vector,        # unit vector, direction exiting at C-term
    }