"""
Shared fixtures for the CLI/pipeline test suite.

Builds a real, minimal 2-chain PDB with gemmi (not a hand-written string —
less error-prone, and it's exactly how the rest of this project builds
structures) so prepare_fusion_job()/remap_chain_order() and
rank_designs()/collect_sequences() run against REAL files through their
REAL code, matching the project's own stated testing convention (mock only
at the submit()/poll_status() I/O boundary, real code paths otherwise).
"""
import os
import pickle
import shutil

import gemmi
import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------------
# Isolated working directory -- every test runs with cwd = a fresh tmp_path,
# matching the project's own "everything resolves relative to os.getcwd()"
# convention (see paths.py / pipeline.py's own module docstrings).
# ----------------------------------------------------------------------
@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _build_ring_structure(path, chain_ids=("A", "B"), n_residues=6):
    """A tiny, real, gemmi-readable 2-chain polymer PDB -- short chain ids
    (already <=1 char) so no isolate.py-style renaming is needed."""
    structure = gemmi.Structure()
    structure.name = "ring"
    model = gemmi.Model("1")

    for c_idx, chain_id in enumerate(chain_ids):
        chain = gemmi.Chain(chain_id)
        for i in range(n_residues):
            res = gemmi.Residue()
            res.name = "ALA"
            res.seqid = gemmi.SeqId(i + 1, " ")
            res.het_flag = "A"
            for atom_name, offset in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                atom.pos = gemmi.Position(c_idx * 20.0 + i * 3.8, offset, 0.0)
                res.add_atom(atom)
            chain.add_residue(res)
        model.add_chain(chain)

    structure.add_model(model)
    structure.setup_entities()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    structure.write_pdb(path)
    return path


@pytest.fixture
def ring_pdb(project_dir):
    path = os.path.join(str(project_dir), "temporary_subunits", "TEST-1_C2_A-B.pdb")
    return _build_ring_structure(path)


def rings_df(assembly_ids, ring_pdb_path):
    """A rings.pkl-shaped DataFrame -- one row per assembly_id, all
    pointing at the same fixture ring PDB (fine: prepare_fusion_job()
    only reads it, never mutates it)."""
    from toolkit.paths import to_portable

    return pd.DataFrame([
        {
            "assembly_id": aid, "symmetry_type": "C2",
            "chain_groups": ("A", "B"),
            "filepath": to_portable(ring_pdb_path),
            "chain_rename_map": {},
        }
        for aid in assembly_ids
    ])


def write_fake_rfdiffusion_design(output_prefix, index, mean_plddt=0.9):
    """Writes a fake "<prefix>_<index>.pdb" (copy of nothing real -- just
    needs to exist and be non-empty) + matching ".trb" (a real pickle
    with a "plddt" array, exactly the shape _load_design_score() reads)
    -- enough for rank_designs() to run for real against it."""
    pdb_path = f"{output_prefix}_{index}.pdb"
    trb_path = f"{output_prefix}_{index}.trb"
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    with open(pdb_path, "w") as f:
        f.write("REMARK fake design fixture\nEND\n")
    plddt = np.full((1, 10), mean_plddt, dtype=float)
    with open(trb_path, "wb") as f:
        pickle.dump({"plddt": plddt, "device": "cpu", "time": 0.0, "config": {}}, f)
    return pdb_path


def write_fake_pmpnn_output(out_folder, basename, n_seqs=2):
    """Writes a fake "<out_folder>/seqs/<basename>.fa" in ProteinMPNN's
    own real output format -- exercises collect_sequences()/
    parse_fasta_output() (real regex parsing) against real files."""
    seqs_dir = os.path.join(out_folder, "seqs")
    os.makedirs(seqs_dir, exist_ok=True)
    fa_path = os.path.join(seqs_dir, f"{basename}.fa")
    lines = [
        ">native, score=1.234, global_score=1.234, fixed_chains=[], designed_chains=['A'], "
        "model_name=v_48_020",
        "AAAAAAAAAA",
    ]
    for i in range(n_seqs):
        lines.append(
            f">T=0.1, sample={i}, score={0.5 + i * 0.01:.3f}, global_score={0.6 + i * 0.01:.3f}, "
            f"seq_recovery={0.4 + i * 0.01:.3f}"
        )
        lines.append("AAAAGGGGAA")
    with open(fa_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return fa_path
