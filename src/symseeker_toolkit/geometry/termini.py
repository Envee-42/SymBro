"""
termini.py — per-chain terminus and CA coordinate extraction.

Single shared entry point for pulling N-terminal CA, C-terminal CA, and
centroid coordinates out of a chain. Everything downstream (NC distance,
ring finding, and later relative terminal orientation / secondary
structure / solvent accessibility) is built on top of this one function,
so the structure only ever gets scanned once per chain.
"""

import numpy as np


def get_chain_ca_geometry(chain):
    """
    Single pass over one chain's polymer: collects every alpha-carbon (CA)
    coordinate in residue order, then derives the N-terminal CA, C-terminal
    CA, and centroid from that one collected list.

    This replaces the original three-pass approach (a forward scan for the
    N-terminal CA, a REVERSED scan for the C-terminal CA, then a third scan
    to collect coordinates for the centroid) with a single scan — each
    residue's CA atom is looked up exactly once instead of up to three
    times. Since residues in a polymer are already stored in sequence
    order, the first resolved CA IS the N-terminal one and the last
    resolved CA IS the C-terminal one — no need to scan twice from opposite
    ends to find them separately.

    Returns None if the chain has no polymer residues at all, or none with
    a resolved CA atom (e.g. a ligand-only chain, or a polymer with fully
    missing backbone density).
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
    return {
        "n": ca_coords[0],           # first resolved CA in sequence = N-terminus
        "c": ca_coords[-1],          # last resolved CA in sequence = C-terminus
        "centroid": ca_coords.mean(axis=0),
    }
