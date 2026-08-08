"""
selfconsistency.py — shared machinery behind every structure-prediction
"self-consistency" backend (alphafold2.py, boltz.py, af3.py): does the
folded prediction a candidate sequence produces actually match the
backbone it was designed on?

This module holds ONLY the parts that are genuinely identical across all
three backends — candidate-id bookkeeping, Kabsch RMSD, pLDDT extraction,
result validation, and SLURM submission plumbing. Everything backend-
specific (how to invoke the tool, what input format it wants, what its
output is named) stays in that backend's own file. See each backend's
own module docstring for why it exists as a separate file rather than
being folded into this one: each wraps a DIFFERENT underlying tool with
its own license terms, its own install footprint, and its own CLI/output
conventions — the "user picks one, they're interchangeable" property
comes from ALL THREE calling into this shared core, not from sharing a
class hierarchy.

Why pLDDT is read from the structure file's own B-factor column, not
from each tool's own separately-versioned confidence JSON: this is a
long-standing, tool-independent PDB/mmCIF convention (AlphaFold2,
AlphaFold3, and Boltz all write per-atom pLDDT into the B-factor field —
confirmed directly for AF2/ColabFold against real output; AF3 and Boltz
follow the same ecosystem convention in their own docs) — reading it
this way means this module never has to track three different JSON
schemas (which, confirmed directly against each tool's own docs, are
NOT the same shape or even the same key names across AF2/AF3/Boltz) and
works identically regardless of which backend produced the file. One
caveat, flagged honestly rather than silently assumed: AF2/AF3's
B-factor pLDDT is confirmed 0-100. Boltz's *separate* JSON confidence
score is documented as combining plddt with ipTM in one weighted
formula ("0.8*plddt + 0.2*ipTM" — confirmed against Boltz's own FAQ),
which only makes dimensional sense if plddt there is on ipTM's 0-1
scale, not 0-100 — but that's the JSON value, not necessarily the
B-factor column's own scale, which this module reads instead and which
hasn't been directly confirmed against a real Boltz output file. The
RELATIVE ordering (which candidate is more/less confident) is reliable
either way; only the absolute default threshold (mean_plddt >= 70, an
AF2/AF3-shaped number) might need adjusting for Boltz specifically —
see boltz.py's own docstring.

RMSD comparison, in short (see alphafold2.py's original version of this
note for the full reasoning): a ProteinMPNN-redesigned sequence has the
EXACT SAME LENGTH and residue ORDER as the RFdiffusion backbone it was
designed on, so comparing a folded candidate against its own reference
design is a direct position-for-position Kabsch superposition — no
sliding-window search needed, unlike pmpnn.py's own fixed-position
detection (which has to search, because it's matching against a
DIFFERENT source file's different chain layout).
"""

import os
import re
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np
import pandas as pd

from toolkit.rfdiffusion import _SBATCH_JOB_ID_RE, _render_sbatch_script, _slurm_job_returncode


# ============================================================================
# Candidate bookkeeping — shared by every backend's prepare_*_job()
# ============================================================================

def sanitize_id(name: str) -> str:
    """
    Every backend derives output file names from a candidate id (a FASTA
    record id, or a per-candidate YAML/JSON job name) — kept conservative
    here (anything other than alnum/underscore/hyphen becomes "_") so
    matching a candidate's id back to its own output file is never
    ambiguous, regardless of what characters source_pdb/rank happen to
    contain, and regardless of which backend is doing the matching.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def build_reference_map(selected_df: pd.DataFrame, design_paths: Sequence[str]) -> Dict[str, dict]:
    """
    Turns pmpnn.select_best_designs()'s output + the RFdiffusion design
    PDBs ProteinMPNN actually read as input into
    {candidate_id: {"sequence": ..., "reference_pdb": ..., "source_pdb": ..., "rank": ...}},
    keyed by the SAME sanitize_id(f"{source_pdb}_rank{rank}") convention
    every backend uses — this is the one piece of job-prep logic that's
    genuinely identical across alphafold2.py/boltz.py/af3.py (only how
    the resulting sequences get WRITTEN — FASTA, YAML, JSON — differs),
    so it's built once here instead of three times.

    Raises ValueError (naming every offending row) for: missing required
    columns, an empty selected_df, a source_pdb with no match in
    design_paths, or a duplicate (source_pdb, rank) pair.
    """
    required_cols = {"source_pdb", "sequence", "rank"}
    missing = required_cols - set(selected_df.columns)
    if missing:
        raise ValueError(
            f"selected_df is missing column(s) {sorted(missing)} — expected the shape "
            f"pmpnn.select_best_designs() returns."
        )
    if selected_df.empty:
        raise ValueError("selected_df has no rows — nothing to fold.")

    design_by_basename = {os.path.splitext(os.path.basename(p))[0]: p for p in design_paths}
    missing_refs = sorted(set(selected_df["source_pdb"]) - set(design_by_basename))
    if missing_refs:
        raise ValueError(
            f"no reference design PDB found in design_paths for source_pdb {missing_refs} "
            f"— every row's source_pdb must match one of design_paths' basenames."
        )

    reference_map = {}
    for _, row in selected_df.iterrows():
        candidate_id = sanitize_id(f"{row['source_pdb']}_rank{int(row['rank'])}")
        if candidate_id in reference_map:
            raise ValueError(
                f"selected_df produces a duplicate candidate id {candidate_id!r} after "
                f"sanitizing — check for duplicate (source_pdb, rank) pairs."
            )
        reference_map[candidate_id] = {
            "sequence": row["sequence"],
            "reference_pdb": design_by_basename[row["source_pdb"]],
            "source_pdb": row["source_pdb"],
            "rank": int(row["rank"]),
        }
    return reference_map


# ============================================================================
# SLURM submission — reused by every backend's own _submit_slurm()
# ============================================================================

def submit_via_slurm(
    inner_argv: List[str], out_dir: str, log_path: str, sbatch_script_path: str,
    job_name: str, partition: Optional[str] = None, account: Optional[str] = None,
    time: str = "04:00:00", gres: Optional[str] = None, gpus: Optional[int] = None,
    cpus_per_task: int = 4, mem: str = "16G", setup_lines: Sequence[str] = (),
    extra_sbatch_directives: Sequence[str] = (), sbatch_executable: str = "sbatch",
) -> Tuple[str, str]:
    """
    Renders an sbatch script wrapping `inner_argv` and submits it — thin
    shared wrapper around rfdiffusion.py's own
    _render_sbatch_script()/_SBATCH_JOB_ID_RE (reused directly, not
    reimplemented — see rfdiffusion.py's module docstring for why
    duplicating scheduler-polling logic is a real maintenance risk this
    project has already been bitten by once). Every backend's own
    _submit_slurm() calls this with its own inner_argv (built by its own
    build_command()/_build_singularity_argv()) and its own log/script
    paths.

    Returns (slurm_job_id, sbatch_script_path). Raises RuntimeError
    (naming the script, exit code, stdout, and stderr) if sbatch itself
    fails to submit.
    """
    os.makedirs(out_dir, exist_ok=True)
    script_content = _render_sbatch_script(
        inner_argv, job_name=job_name, log_path=log_path, partition=partition, account=account,
        time=time, gres=gres, gpus=gpus, cpus_per_task=cpus_per_task, mem=mem,
        setup_lines=setup_lines, extra_sbatch_directives=extra_sbatch_directives, cd_to=None,
    )
    with open(sbatch_script_path, "w") as f:
        f.write(script_content)

    result = subprocess.run([sbatch_executable, sbatch_script_path], capture_output=True, text=True)
    match = _SBATCH_JOB_ID_RE.search(result.stdout or "")
    if result.returncode != 0 or not match:
        raise RuntimeError(
            f"sbatch failed to submit {sbatch_script_path!r} (exit {result.returncode}) — "
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
    return match.group(1), sbatch_script_path


def slurm_returncode(slurm_job_id: str) -> Optional[int]:
    """Re-exports rfdiffusion.py's own _slurm_job_returncode() — every
    backend's poll_status() calls this the same way rfdiffusion.py's
    own poll_status() does."""
    return _slurm_job_returncode(slurm_job_id)


def terminate_or_cancel(process: Optional[subprocess.Popen], slurm_job_id: Optional[str]) -> None:
    """Shared cancel() body: scancel for a slurm run, process.terminate()
    otherwise — same convention rfdiffusion.py/alphafold2.py/boltz.py/
    af3.py all use."""
    if slurm_job_id is not None:
        subprocess.run(["scancel", slurm_job_id], capture_output=True)
    elif process is not None and process.poll() is None:
        process.terminate()


# ============================================================================
# Results: fold-vs-design RMSD + pLDDT
# ============================================================================

def ca_coords(structure_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a structure's (PDB OR mmCIF — gemmi picks the parser off the
    extension, so this works unchanged for ColabFold's .pdb and Boltz's/
    AF3's .cif output) CA coordinates AND per-residue B-factor (pLDDT for
    a predicted structure) in residue order, as parallel arrays.
    setup_entities() is required first for get_polymer() to see anything
    on a header-less, ATOM-only file — the same gemmi quirk already
    documented and fixed throughout pmpnn.py/rfdiffusion.py.
    """
    structure = gemmi.read_structure(structure_path)
    structure.setup_entities()
    model = structure[0]
    coords, bfactors = [], []
    for chain in model:
        for res in chain.get_polymer():
            atom = res.find_atom("CA", "*")
            if atom is not None:
                coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
                bfactors.append(atom.b_iso)
    return np.array(coords), np.array(bfactors)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Optimal rigid-body (Kabsch) superposition RMSD between two
    equal-length, residue-order-matched CA coordinate arrays."""
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)
    H = P_centered.T @ Q_centered
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    P_aligned = (R @ P_centered.T).T
    return float(np.sqrt(np.mean(np.sum((P_aligned - Q_centered) ** 2, axis=1))))


def collect_results(folded_paths: Dict[str, str], reference_map: Dict[str, dict]) -> pd.DataFrame:
    """
    Fully backend-agnostic: for every candidate with a folded structure
    written so far, computes rmsd_to_design (whole-chain CA RMSD,
    Angstrom, via Kabsch superposition against the ORIGINAL RFdiffusion
    design PDB) and mean_plddt (mean per-residue pLDDT off the predicted
    structure's own B-factor column). Every backend's own
    collect_results() is a thin wrapper that locates folded_paths (its
    own output-naming convention) and calls this.

    folded_paths : {candidate_id: predicted structure path} — from
        poll_status()'s own "folded_paths" (or equivalent) key.
    reference_map : build_reference_map()'s own return value (or
        anything with the same {"reference_pdb": ..., ...} shape per
        candidate_id).

    Raises ValueError (naming the candidate) if a folded structure's CA
    count doesn't match its reference's — a mismatch means something
    upstream is wrong, and silently RMSD-ing mismatched lengths would
    just produce a meaningless number.
    """
    rows = []
    for candidate_id, folded_path in folded_paths.items():
        reference_path = reference_map[candidate_id]["reference_pdb"]
        pred_coords, pred_bfactors = ca_coords(folded_path)
        ref_coords, _ref_bfactors = ca_coords(reference_path)

        if len(pred_coords) != len(ref_coords):
            raise ValueError(
                f"candidate {candidate_id!r}: folded structure has {len(pred_coords)} CA atoms "
                f"but its reference design {reference_path!r} has {len(ref_coords)} — expected "
                f"an exact match (ProteinMPNN never changes chain length). Check that "
                f"{folded_path!r} is really this candidate's own prediction."
            )

        rows.append({
            "candidate_id": candidate_id,
            "folded_path": folded_path,
            "reference_path": reference_path,
            "rmsd_to_design": kabsch_rmsd(pred_coords, ref_coords),
            "mean_plddt": float(np.mean(pred_bfactors)) if len(pred_bfactors) else None,
        })

    return pd.DataFrame(
        rows, columns=["candidate_id", "folded_path", "reference_path", "rmsd_to_design", "mean_plddt"],
    )


def select_validated_designs(results_df: pd.DataFrame, max_rmsd: float = 2.0, min_plddt: float = 70.0) -> pd.DataFrame:
    """
    Final filter, applied AFTER collect_results(): keeps only candidates
    whose predicted structure actually looks like the intended design —
    low RMSD to the RFdiffusion backbone (default 2.0 A, a common
    self-consistency cutoff in the RFdiffusion/dl_binder_design
    literature) AND high enough confidence (default mean pLDDT >= 70,
    AlphaFold's own "confident" threshold — see this module's docstring
    for the caveat on Boltz's B-factor scale specifically) that the RMSD
    number itself is trustworthy. Returns a copy sorted best (lowest
    RMSD) first. Neither threshold is a universal constant — tighten for
    a more conservative final shortlist, or loosen if nothing passes and
    you want to see what came closest.
    """
    required_cols = {"rmsd_to_design", "mean_plddt"}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(
            f"results_df is missing column(s) {sorted(missing)} — expected the shape "
            f"collect_results() returns."
        )
    passed = results_df[(results_df["rmsd_to_design"] <= max_rmsd) & (results_df["mean_plddt"] >= min_plddt)]
    return passed.sort_values("rmsd_to_design").reset_index(drop=True)
