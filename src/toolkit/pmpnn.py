"""
proteinmpnn.py — runs ProteinMPNN locally against RFdiffusion's output,
however that output arrives, and writes sequences into
temporary_simulations/ (see download.py's temporary_files/ and colab.py's
own temporary_simulations/ for the same "workspace-visible scratch
folder" convention this follows).

QUICKSTART — the one call most people need:

    from toolkit import proteinmpnn

    # from a Colab results.zip:
    sequences_df = proteinmpnn.run("results.zip", repo_path="/path/to/ProteinMPNN")

    # from an HPC/SLURM RFdiffusion run's output folder:
    sequences_df = proteinmpnn.run("path/to/hpc_output_folder", repo_path="/path/to/ProteinMPNN")

    # from a local/singularity RFdiffusion run you already have a handle for:
    sequences_df = proteinmpnn.run(rf_run, repo_path="/path/to/ProteinMPNN")

run() blocks until ProteinMPNN finishes and hands back a DataFrame of
designed sequences. If you need this to be non-blocking instead (e.g.
driving a NiceGUI progress panel), use prepare_mpnn_job() / submit() /
poll_status() / collect_sequences() directly — run() is just those four
calls wired together for the common case.

WHAT "source" CAN BE — prepare_mpnn_job()/run() accept, interchangeably:
  - a path to a .zip file          (a Colab results.zip)
  - a path to a folder of PDBs     (an HPC/SLURM output folder, or an
                                     already-unzipped Colab download)
  - a path to one .pdb file, or a list of .pdb paths
  - an rfdiffusion.RFdiffusionRun handle (local/singularity backend)
  - a dict shaped like rfdiffusion.poll_status()'s/
    colab.import_colab_results()'s return value
_resolve_input_pdbs() (below) is the one place that tells these apart —
everything downstream never needs to know or care which shape it started
as.

Command syntax below (batch-mode jsonl parsing, the chain_id_jsonl /
fixed_positions_jsonl shapes, --sampling_temp being PER-TEMPERATURE and
num_seq_per_target being a per-temperature count rather than a total, the
num_seq_per_target // batch_size floor-division gotcha) was verified
directly against dauparas/ProteinMPNN's protein_mpnn_run.py,
helper_scripts/parse_multiple_chains.py, and
helper_scripts/assign_fixed_chains.py source — not recalled from memory.

Design, in short:

- LOCAL ONLY, on purpose: this always runs ProteinMPNN as a local
  subprocess. No Singularity/SLURM EXECUTION backend for ProteinMPNN
  itself exists yet — that's a separate question from accepting inputs
  that came from an HPC/SLURM-run RFdiffusion job, which this module
  already does (see "WHAT source CAN BE" above). ProteinMPNN's own
  install is light (plain PyTorch, no compiled extras), so there's been
  no need for the container path RFdiffusion needed from day one. If a
  remote execution backend is ever wanted, submit()'s shape (a job in,
  a non-blocking run handle out) is already set up so adding one
  wouldn't change how poll_status()/cancel() are called — same pattern
  rfdiffusion.py's own backend= parameter follows.

- Every design PDB — regardless of whether it arrived via a zip, a
  folder, or a run handle — is treated as ONE batch: parse_multiple_
  chains.py parses all of them in one pass, and protein_mpnn_run.py
  loads the model once and designs sequences for every backbone in one
  process, rather than once per backbone.

- ZIP/FOLDER INGESTION IS TOP-LEVEL ONLY, NEVER RECURSIVE — this is
  deliberate, not an oversight. RFdiffusion writes a "traj/" subfolder
  alongside its real per-design output whenever trajectory recording is
  on (confirmed against scripts/run_inference.py): per-step diffusion
  snapshots named like "design_0_pX0_traj.pdb" and
  "design_0_Xt-1_traj.pdb", sitting right next to the real
  "design_0.pdb". A recursive glob would silently feed those
  half-denoised trajectory frames to ProteinMPNN as if they were
  finished designs. Scanning only the top level of wherever the real
  designs live is both simpler and exactly matches where RFdiffusion
  actually puts its finished output.

- FIXED vs DIFFUSED residues: prepare_mpnn_job() does NOT default to
  "every residue is designable". RFdiffusion's own scripts/run_inference.py
  DOES write a clean binary fixed(1.0)/diffused(0.0) B-factor mask when it
  writes a design (confirmed directly against RosettaCommons/RFdiffusion's
  current writepdb()/run_inference.py source) — but that convention turned
  out NOT to be a reliable signal to read back here: a real Colab-produced
  design PDB from this project's own pipeline was checked directly
  (residue-by-residue, via gemmi) and its B-factor column was a smooth,
  non-binary gradient (roughly 0.89-1.00 everywhere, never actually 0.0) —
  not the clean split the RFdiffusion source implies. Rather than trust an
  assumption that demonstrably didn't hold against real output (whatever
  the exact cause — an older/forked RFdiffusion commit, a Colab-side
  re-write, or something else), fixed-position detection here instead
  matches COORDINATES, which is robust regardless of B-factor quirks:

  fixed_positions_from_contig_match() reads the ORIGINAL per-chain
  segments straight off the RFdiffusionJob that produced a design (its
  .input_pdb and .contigs — the exact chains/residue-ranges
  build_linker_fusion_contig() marked "fixed", parsed back out of the
  contig string) and, for each design, Kabsch-superposes each segment's
  CA coordinates against a sliding window of the design's own CA
  coordinates to find where RFdiffusion placed it. This works because a
  "fixed" segment is copied into the output VERBATIM (same internal
  geometry, just rigidly rotated/translated along with the rest of the
  design) — confirmed against this project's own real design output:
  every segment's best-matching window comes back at ~0.1-0.3 Å RMSD,
  and the leftover un-matched stretch between segments lands exactly at
  the configured linker_length range. This also sidesteps needing
  OUTPUT residue numbering to match INPUT numbering, and needs no
  RFdiffusion-version-specific assumption about metadata columns at all.

  colab.import_colab_results() and rfdiffusion.poll_status() both carry
  their originating RFdiffusionJob through in their returned dict's
  "job" key specifically so prepare_mpnn_job() can find it automatically
  — pass rf_job=... explicitly only if you built `source` some other way
  (e.g. a bare folder/zip path with no job attached). prepare_mpnn_job()
  calls fixed_positions_from_contig_match() automatically (via
  auto_fix_from_rfdiffusion=True) unless you pass fixed_positions
  yourself, so "keep the native segments, design only what RFdiffusion
  actually generated" is the default. If no RFdiffusionJob can be found
  anywhere, this prints a warning and leaves every residue designable
  (same fail-soft behavior as before) rather than raising.

  detect_fixed_residues() / fixed_positions_from_rfdiffusion() (the
  original B-factor-reading pair) are kept below for a "local"/
  "singularity" backend run against a KNOWN-good RFdiffusion install
  where you've confirmed the binary B-factor convention actually holds
  — just not used by default anymore.

  NOTE: every gemmi.read_structure() call in this module is followed by
  structure.setup_entities() — required for chain.get_polymer() to see
  anything at all on a header-less, ATOM-only PDB (no SEQRES/entity
  records), which is exactly what RFdiffusion/ProteinMPNN write and pass
  around. Without it, get_polymer() silently returns empty and every
  residue-lookup here would look like "not found" — confirmed as the
  root cause of a real ValueError from build_fixed_positions_dict()
  rejecting perfectly valid, already-verified fixed positions.

- --sampling_temp is a per-TEMPERATURE value, and num_seq_per_target is
  the sequence count PER temperature, not a total (protein_mpnn_run.py's
  own inner loop is `for temp in temperatures: for j in
  range(num_seq_per_target // batch_size): ...`) — num_seq_per_target //
  batch_size floor-divides, so a mismatched pair (e.g.
  num_seq_per_target=10, batch_size=8) would silently return only 8, not
  10. ProteinMPNNJob.__post_init__ validates batch_size evenly divides
  num_seq_per_target up front rather than letting that happen quietly.

- submit() launches ProteinMPNN via subprocess.Popen (never
  subprocess.run) for the actual model run, returning a ProteinMPNNRun
  handle immediately. (The one exception: staging the jsonl via
  parse_multiple_chains.py IS run synchronously first, since it's a
  fast, model-free, single-pass text/coordinate extraction the main
  run's jsonl_path depends on existing — negligible next to the
  GPU-bound step that follows.) poll_status() counts
  "{out_folder}/seqs/<pdb_basename>.fa" files written so far —
  ProteinMPNN, like RFdiffusion, writes no separate completion marker.

- collect_sequences() parses ProteinMPNN's own FASTA header convention
  into a tidy DataFrame, one row per generated sequence.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gemmi
import numpy as np
import pandas as pd

from toolkit.config import get_tool_config


# ============================================================================
# Scratch folder — everything this module writes lands under here
# ============================================================================

# Same "temporary_X/" convention download.py's TEMP_DIR_NAME
# (temporary_files/) and colab.py's TEMP_SIMULATIONS_DIR_NAME already
# established — kept as its own local constant/helper here rather than
# imported from colab.py, matching this project's existing pattern of
# each module owning its own scratch-folder constant rather than
# centralizing it (download.py and colab.py don't share one either).
TEMP_SIMULATIONS_DIR_NAME = "temporary_simulations"


def get_simulations_dir() -> str:
    """
    Returns the absolute path to the temporary_simulations/ folder,
    creating it if it doesn't exist yet. Mirrors download.py's
    get_temp_dir() / colab.py's get_simulations_dir() exactly: resolved
    relative to the current working directory (normally your project
    root) so it shows up right there in your workspace/editor. This is
    where ProteinMPNN's own out_folder lands by default, and where a zip
    passed to prepare_mpnn_job()/run() gets extracted.
    """
    path = os.path.join(os.getcwd(), TEMP_SIMULATIONS_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def clear_simulations_dir(data_dir: Optional[str] = None) -> None:
    """Deletes everything INSIDE temporary_simulations/ (or a custom
    data_dir) but keeps the folder itself -- same "easy to empty, stays
    ready to use" behavior as download.py's clear_temp_dir() / isolate.py's
    clear_temp_subunits_dir()."""
    if data_dir is None:
        data_dir = get_simulations_dir()

    for entry in os.listdir(data_dir):
        if entry.startswith('.'):
            continue
        entry_path = os.path.join(data_dir, entry)
        if os.path.isfile(entry_path) or os.path.islink(entry_path):
            os.remove(entry_path)
        else:
            shutil.rmtree(entry_path)


# ============================================================================
# Turning "whatever RFdiffusion produced" into a plain list of .pdb paths
# ============================================================================

def _pdbs_in_folder(folder: str) -> List[str]:
    """
    Top-level-only *.pdb scan of `folder` — see the module docstring's
    "ZIP/FOLDER INGESTION IS TOP-LEVEL ONLY" section for why this never
    recurses into a "traj/" subfolder.

    If `folder` itself has no top-level PDBs but has exactly one
    subfolder (other than "traj"), descends into THAT one subfolder
    once — this is what lets an unzipped Colab bundle work whether its
    real layout is "<folder>/design_0.pdb" or
    "<folder>/output/design_0.pdb" (RFdiffusion's own bundle convention
    nests output under "output/" — see colab.py's BUNDLE_OUTPUT_PREFIX)
    without the caller needing to know which.
    """
    found = sorted(glob.glob(os.path.join(folder, "*.pdb")))
    if found:
        return found

    subdirs = [
        os.path.join(folder, name) for name in os.listdir(folder)
        if name != "traj" and os.path.isdir(os.path.join(folder, name))
    ]
    if len(subdirs) == 1:
        nested = sorted(glob.glob(os.path.join(subdirs[0], "*.pdb")))
        if nested:
            return nested

    raise FileNotFoundError(
        f"no .pdb files found directly inside {folder!r} (one level of nesting was also "
        f"checked) — make sure this is RFdiffusion's own output folder (or its zip), not "
        f"renamed or reorganized, and not just a 'traj/' trajectory folder."
    )


def _resolve_input_pdbs(source) -> List[str]:
    """
    Normalizes ANY of the shapes this module accepts as "RFdiffusion
    output" (see module docstring's "WHAT source CAN BE") into a plain
    list of local .pdb file paths.
    """
    from toolkit.rfdiffusion import RFdiffusionRun, poll_status as rfdiffusion_poll_status

    if isinstance(source, RFdiffusionRun):
        source = rfdiffusion_poll_status(source)

    if isinstance(source, dict):
        if "design_paths" not in source:
            raise TypeError(
                f"dict source is missing a 'design_paths' key — expected the shape "
                f"rfdiffusion.poll_status()/colab.import_colab_results() return, got keys "
                f"{list(source)}"
            )
        design_paths = list(source["design_paths"])
        if not design_paths:
            raise ValueError(f"no design_paths to work with — state={source.get('state')!r}")
        return design_paths

    if isinstance(source, (list, tuple)):
        if not source:
            raise ValueError("source list is empty — nothing to design sequences for")
        return list(source)

    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(
                f"source={source!r} does not exist — pass the real path to a results.zip, "
                f"an output folder, or a .pdb file. (A bare relative filename only resolves "
                f"if it's sitting in the current working directory — pass an absolute path "
                f"if you're not sure.)"
            )
        if os.path.isfile(source):
            if source.endswith(".zip"):
                extract_dir = os.path.join(
                    get_simulations_dir(), "_extracted", os.path.splitext(os.path.basename(source))[0],
                )
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(source) as zf:
                    zf.extractall(extract_dir)
                return _pdbs_in_folder(extract_dir)
            if source.endswith(".pdb"):
                return [source]
            raise ValueError(
                f"source file {source!r} is neither a .zip nor a .pdb — pass one of those, "
                f"a folder, or a list of .pdb paths instead."
            )
        return _pdbs_in_folder(source)  # a directory

    raise TypeError(
        f"source must be a .zip path, a folder path, a .pdb path, a list of .pdb paths, an "
        f"rfdiffusion.RFdiffusionRun handle, or a poll_status()-shaped dict — got {type(source)!r}"
    )


# ============================================================================
# FIXED vs DIFFUSED residue detection (see module docstring)
# ============================================================================

def detect_fixed_residues(pdb_path: str, fixed_bfactor: float = 1.0, tolerance: float = 0.01) -> Dict[str, List[int]]:
    """
    Reads an RFdiffusion output PDB and returns {chain: [resnum, ...]}
    for every residue RFdiffusion flagged as FIXED — i.e. copied
    verbatim from the original input structure, as opposed to newly
    diffused. See the module docstring's "FIXED vs DIFFUSED residues"
    section for how this B-factor signal is confirmed to work.

    A residue counts as "fixed" if its own atoms' B-factor is within
    `tolerance` of `fixed_bfactor` (default 1.0) — read from that
    residue's first atom, since every atom of one residue shares the
    same B-factor by construction. Returns {} (not an error) for a
    fully de-novo / unconditional design with no fixed segments at all.
    """
    structure = gemmi.read_structure(pdb_path)
    structure.setup_entities()  # see _read_chain_names()'s comment for why
    model = structure[0]

    fixed: Dict[str, List[int]] = {}
    for chain in model:
        resnums = [
            res.seqid.num for res in chain.get_polymer()
            if len(res) > 0 and abs(res[0].b_iso - fixed_bfactor) <= tolerance
        ]
        if resnums:
            fixed[chain.name] = resnums
    return fixed


def fixed_positions_from_rfdiffusion(input_pdbs: Sequence[str], **kwargs) -> Dict[str, Dict[str, List[int]]]:
    """
    Batch entry point: runs detect_fixed_residues() over every input PDB
    and returns {pdb_basename: {chain: [resnum, ...]}} — exactly the
    shape ProteinMPNNJob.fixed_positions / build_fixed_positions_dict()
    expect. NOT called automatically by prepare_mpnn_job() anymore (see
    module docstring's "FIXED vs DIFFUSED residues" section for why) —
    fixed_positions_from_contig_match() below is the default. Use this
    directly only against a "local"/"singularity" RFdiffusion install
    you've confirmed actually writes the binary B-factor convention.
    """
    result: Dict[str, Dict[str, List[int]]] = {}
    for pdb_path in input_pdbs:
        basename = os.path.splitext(os.path.basename(pdb_path))[0]
        chain_positions = detect_fixed_residues(pdb_path, **kwargs)
        if chain_positions:
            result[basename] = chain_positions
    return result


# ============================================================================
# FIXED vs DIFFUSED residue detection, take two: coordinate matching
# ============================================================================
#
# See the module docstring's "FIXED vs DIFFUSED residues" section for why
# this replaced the B-factor approach above as prepare_mpnn_job()'s
# default. Short version: a "fixed" segment is copied into RFdiffusion's
# output VERBATIM — same internal geometry, just carried along with
# whatever rigid rotation/translation the rest of the design got — so
# Kabsch-superposing each original chain segment's CA coordinates against
# a sliding window of the design's own CA coordinates reliably finds
# exactly where RFdiffusion placed it, with no dependency on metadata
# columns that turned out not to hold in practice.

_CONTIG_FIXED_TOKEN_RE = re.compile(r"^([A-Za-z]+)(\d+)-(\d+)$")


def _fixed_segments_from_contig(contigs: str) -> List[Tuple[str, int, int]]:
    """
    Parses an RFdiffusion contig string (e.g.
    "[A1-159/10-15/B1-159/10-15/C1-158]", exactly what
    rfdiffusion.build_linker_fusion_contig()/build_contig_string()
    produce) into an ORDERED list of (chain, start, end) FIXED segments
    only. Bare "min-max" diffuse/linker tokens (no leading chain letter)
    are skipped on purpose — RFdiffusion resamples each linker's actual
    realized length per design, so the contig string alone can't say how
    long a given design's linker really came out; that's recovered
    per-design by fixed_positions_from_contig_match() below instead.

    Only understands the linear fixed/diffused/fixed/... shape
    build_linker_fusion_contig() generates — a "/0 " chain-break token
    (see rfdiffusion.build_contig_string()) is not meaningful for a
    fusion job and isn't handled here.
    """
    inner = contigs.strip().lstrip("[").rstrip("]")
    tokens = inner.replace("/0 ", "/").split("/")
    segments = []
    for token in tokens:
        match = _CONTIG_FIXED_TOKEN_RE.match(token.strip())
        if match:
            chain, start, end = match.group(1), int(match.group(2)), int(match.group(3))
            segments.append((chain, start, end))
    return segments


def _residue_ca_coords(residues) -> List[Optional[Tuple[float, float, float]]]:
    coords = []
    for res in residues:
        atom = res.find_atom("CA", "*")
        coords.append(None if atom is None else (atom.pos.x, atom.pos.y, atom.pos.z))
    return coords


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    RMSD between two Nx3 coordinate arrays after the optimal rigid-body
    superposition (rotation + translation, no reflection) that minimizes
    it — i.e. "how close are these two point sets to being the SAME
    shape, ignoring where in space each one sits." This is exactly what's
    needed to recognize a segment RFdiffusion copied verbatim even though
    the whole design ends up rigidly re-placed somewhere else in space.
    """
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)
    H = P_centered.T @ Q_centered
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    P_aligned = (R @ P_centered.T).T
    return float(np.sqrt(np.mean(np.sum((P_aligned - Q_centered) ** 2, axis=1))))


def fixed_positions_from_contig_match(
    rf_job, design_paths: Sequence[str], rmsd_tolerance: float = 1.0,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Batch entry point matching detect_fixed_residues()'s shape/purpose,
    but by geometry instead of B-factor: for each design in
    design_paths, locates every "fixed" segment named in
    rf_job.contigs (read against rf_job.input_pdb — the same ring PDB
    the RFdiffusion job actually ran on) by Kabsch-matching its CA
    coordinates against the design's own, in contig order. Returns
    {pdb_basename: {chain: [resnum, ...]}} — exactly what
    ProteinMPNNJob.fixed_positions / build_fixed_positions_dict() expect.

    A segment whose best match exceeds rmsd_tolerance Angstroms (default
    1.0 — comfortably above the ~0.1-0.3 A typically seen for a genuine
    verbatim-copied match, and comfortably below the several-Angstrom
    RMSD a wrong window gives) is treated as unlocatable: a warning is
    printed naming the design and segment, and that design is skipped
    entirely (no partial/wrong fixed_positions entry for it) rather than
    guessing — same fail-soft-but-not-silently-wrong spirit as
    process_ring_dataframe()'s missing-file handling in rfdiffusion.py.
    """
    segments = _fixed_segments_from_contig(rf_job.contigs)
    if not segments:
        return {}

    input_structure = gemmi.read_structure(rf_job.input_pdb)
    input_structure.setup_entities()
    input_model = input_structure[0]

    result: Dict[str, Dict[str, List[int]]] = {}
    for pdb_path in design_paths:
        basename = os.path.splitext(os.path.basename(pdb_path))[0]

        design_structure = gemmi.read_structure(pdb_path)
        design_structure.setup_entities()
        design_model = design_structure[0]
        out_chain = next((c for c in design_model if len(c.get_polymer()) > 0), None)
        if out_chain is None:
            print(f"Warning: {basename!r} has no polymer chain — skipping fixed-position detection for it.")
            continue

        out_residues = list(out_chain.get_polymer())
        out_ca = _residue_ca_coords(out_residues)

        cursor = 0
        fixed_resnums: List[int] = []
        design_ok = True
        for chain_name, start, end in segments:
            in_chain = input_model.find_chain(chain_name)
            if in_chain is None:
                raise ValueError(
                    f"contig references chain {chain_name!r}, not found in "
                    f"rf_job.input_pdb={rf_job.input_pdb!r} — available chains: "
                    f"{[c.name for c in input_model]}"
                )
            seg_residues = [r for r in in_chain.get_polymer() if start <= r.seqid.num <= end]
            seg_ca = _residue_ca_coords(seg_residues)
            if not seg_ca or any(c is None for c in seg_ca):
                raise ValueError(
                    f"chain {chain_name!r} residues {start}-{end} (from rf_job.contigs) "
                    f"are missing a CA atom somewhere in {rf_job.input_pdb!r} — cannot match."
                )
            seg_arr = np.array(seg_ca)
            seg_len = len(seg_arr)

            best = None
            for window_start in range(cursor, len(out_ca) - seg_len + 1):
                window = out_ca[window_start:window_start + seg_len]
                if any(w is None for w in window):
                    continue
                rmsd = _kabsch_rmsd(seg_arr, np.array(window))
                if best is None or rmsd < best[0]:
                    best = (rmsd, window_start)

            if best is None or best[0] > rmsd_tolerance:
                found = f"best RMSD {best[0]:.2f} A" if best else "no candidate window at all"
                print(
                    f"Warning: could not confidently locate fixed segment {chain_name}{start}-{end} "
                    f"inside {basename!r} ({found}, tolerance is {rmsd_tolerance} A) — skipping "
                    f"fixed-position detection for this design; every residue will be designable."
                )
                design_ok = False
                break

            _rmsd, window_start = best
            fixed_resnums.extend(out_residues[i].seqid.num for i in range(window_start, window_start + seg_len))
            cursor = window_start + seg_len

        if design_ok and fixed_resnums:
            result[basename] = {out_chain.name: fixed_resnums}

    return result


def _extract_rf_job(source):
    """
    Best-effort pull of the originating RFdiffusionJob out of `source` —
    an RFdiffusionRun handle (source.job), or a dict that carries one
    under a "job" key (colab.import_colab_results() and
    rfdiffusion.poll_status() both do). Returns None if source doesn't
    carry one (e.g. a bare .zip/folder/.pdb-list source) — pass rf_job=
    explicitly to prepare_mpnn_job() in that case.
    """
    from toolkit.rfdiffusion import RFdiffusionRun

    if isinstance(source, RFdiffusionRun):
        return source.job
    if isinstance(source, dict):
        return source.get("job")
    return None


# ============================================================================
# Job spec
# ============================================================================

@dataclass
class ProteinMPNNJob:
    """A fully-specified, ready-to-submit ProteinMPNN batch inference call."""

    input_pdbs: List[str]
    out_folder: str
    num_seq_per_target: int = 8
    sampling_temp: Union[float, Sequence[float]] = 0.1
    batch_size: int = 8
    seed: int = 37
    # {chain_name, ...} — applied to EVERY structure in input_pdbs: every
    # chain named here is designed, every other chain each PDB actually
    # has is fixed. None (the default) leaves every chain in every
    # structure designable (subject to fixed_positions, below).
    designed_chains: Optional[Sequence[str]] = None
    # {pdb_basename: {chain: [resnum, ...]}} — residues to keep at their
    # native identity even within an otherwise-designed chain. A PDB not
    # mentioned here has no fixed positions at all. Auto-populated by
    # prepare_mpnn_job() from RFdiffusion's own B-factor markup unless
    # you pass this explicitly.
    fixed_positions: Optional[Dict[str, Dict[str, Sequence[int]]]] = None
    model_name: str = "v_48_020"
    ca_only: bool = False
    use_soluble_model: bool = False
    save_score: bool = False
    backbone_noise: float = 0.0
    omit_aas: str = "X"
    # Escape hatch: any other protein_mpnn_run.py flag this module
    # doesn't model explicitly, passed straight through as --key=value.
    # Keys should NOT include the leading "--".
    extra_overrides: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.input_pdbs:
            raise ValueError("input_pdbs must be non-empty")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive — got {self.batch_size}")
        if self.num_seq_per_target % self.batch_size != 0:
            actual = (self.num_seq_per_target // self.batch_size) * self.batch_size
            raise ValueError(
                f"num_seq_per_target={self.num_seq_per_target} is not evenly divisible by "
                f"batch_size={self.batch_size} — ProteinMPNN floor-divides internally, so "
                f"this combination would silently generate only {actual} sequences per "
                f"temperature instead of {self.num_seq_per_target}. Pick a batch_size that "
                f"divides evenly instead."
            )
        basenames = [os.path.splitext(os.path.basename(p))[0] for p in self.input_pdbs]
        dupes = sorted({b for b in basenames if basenames.count(b) > 1})
        if dupes:
            raise ValueError(
                f"input_pdbs has duplicate basename(s) {dupes} — ProteinMPNN names each "
                f"output FASTA after its input PDB's basename, so duplicates would "
                f"silently overwrite each other's results. Rename the source files or "
                f"pass unique paths."
            )


def prepare_mpnn_job(
    source, out_folder: Optional[str] = None,
    designed_chains: Optional[Sequence[str]] = None,
    fixed_positions: Optional[Dict[str, Dict[str, Sequence[int]]]] = None,
    auto_fix_from_rfdiffusion: bool = True, rf_job=None,
    num_seq_per_target: int = 8, sampling_temp: Union[float, Sequence[float]] = 0.1,
    batch_size: int = 8, seed: int = 37, **job_kwargs,
) -> ProteinMPNNJob:
    """
    Builds a submit()-ready ProteinMPNNJob from `source` — see the module
    docstring's "WHAT source CAN BE" for every accepted shape (a
    results.zip, an HPC output folder, a .pdb path, a list of .pdb
    paths, an RFdiffusionRun handle, or a poll_status()-shaped dict).

    out_folder defaults to "mpnn_designs" INSIDE get_simulations_dir()
    (temporary_simulations/mpnn_designs/) — every sequence this module
    writes lands under temporary_simulations/ unless you override this.

    auto_fix_from_rfdiffusion : if True (the default) AND fixed_positions
        isn't given explicitly, finds each design's native/fixed segments
        by matching coordinates back against the RFdiffusionJob that
        produced it (see fixed_positions_from_contig_match() and the
        module docstring's "FIXED vs DIFFUSED residues" section) and uses
        THAT — so a fusion design's native, kept-fixed segments keep
        their real sequence by default, and ProteinMPNN only actually
        designs the linkers RFdiffusion generated between them. Pass
        fixed_positions explicitly (even {} to force "nothing fixed") to
        override, or set auto_fix_from_rfdiffusion=False to fall back to
        "every residue in every designed chain is designable".
    rf_job : the RFdiffusionJob to match designs against. Only needed if
        `source` doesn't already carry one — an RFdiffusionRun handle, or
        a dict from colab.import_colab_results()/rfdiffusion.poll_status()
        (both include a "job" key), all supply this automatically via
        _extract_rf_job(). Required explicitly for a bare .zip/folder/
        .pdb-list source; if none is found at all, a warning is printed
        and every residue is left designable rather than raising.
    """
    design_paths = _resolve_input_pdbs(source)

    if out_folder is None:
        out_folder = os.path.join(get_simulations_dir(), "mpnn_designs")

    if fixed_positions is None and auto_fix_from_rfdiffusion:
        rf_job = rf_job or _extract_rf_job(source)
        if rf_job is not None:
            fixed_positions = fixed_positions_from_contig_match(rf_job, design_paths)
        else:
            print(
                "Warning: auto_fix_from_rfdiffusion=True but no RFdiffusionJob was "
                "available to match fixed segments against (source didn't carry one) "
                "— every residue will be treated as designable. Pass rf_job=<your "
                "RFdiffusionJob> explicitly, or fixed_positions yourself, to keep the "
                "ring's native sequence fixed."
            )

    return ProteinMPNNJob(
        input_pdbs=design_paths, out_folder=out_folder,
        num_seq_per_target=num_seq_per_target, sampling_temp=sampling_temp,
        batch_size=batch_size, seed=seed, designed_chains=designed_chains,
        fixed_positions=fixed_positions or None, **job_kwargs,
    )


# ============================================================================
# chain_id_jsonl / fixed_positions_jsonl construction
# ============================================================================

def _read_chain_names(pdb_path: str) -> List[str]:
    # setup_entities() is required for get_polymer() to see anything at all
    # on a header-less, ATOM-only PDB (no SEQRES/entity records) — exactly
    # what RFdiffusion/ProteinMPNN write and pass around. Without it,
    # get_polymer() silently returns empty, and every chain looks like it
    # has zero residues. Same fix as fixed_positions_from_contig_match()
    # and build_fixed_positions_dict() below need for the same reason.
    structure = gemmi.read_structure(pdb_path)
    structure.setup_entities()
    model = structure[0]
    return [chain.name for chain in model if len(chain.get_polymer()) > 0]


def build_chain_id_dict(input_pdbs: Sequence[str], designed_chains: Sequence[str]) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Builds {pdb_basename: (designed_chain_list, fixed_chain_list)} — the
    exact shape helper_scripts/assign_fixed_chains.py itself produces,
    built directly against each PDB's REAL chain names (via gemmi)
    rather than typed by hand.

    designed_chains is applied globally to every input PDB: every chain
    named here is designed in EVERY structure, and every chain a given
    structure actually has that ISN'T named here is fixed for that
    structure. Raises ValueError (naming every offending PDB) if any
    requested chain isn't actually present in that PDB.
    """
    result: Dict[str, Tuple[List[str], List[str]]] = {}
    missing = []
    for pdb_path in input_pdbs:
        basename = os.path.splitext(os.path.basename(pdb_path))[0]
        all_chains = _read_chain_names(pdb_path)
        not_found = [c for c in designed_chains if c not in all_chains]
        if not_found:
            missing.append(f"{basename}: requested {not_found}, has {all_chains}")
            continue
        fixed = [c for c in all_chains if c not in designed_chains]
        result[basename] = (list(designed_chains), fixed)

    if missing:
        raise ValueError(f"designed_chains reference chain(s) not present in the structure — {missing}")
    return result


def build_fixed_positions_dict(
    input_pdbs: Sequence[str], fixed_positions: Dict[str, Dict[str, Sequence[int]]],
) -> Dict[str, Dict[str, List[int]]]:
    """
    Validates a {pdb_basename: {chain: [resnum, ...]}} fixed-positions
    spec against each named PDB's ACTUAL resolved residues (gemmi) — a
    typo'd chain or out-of-range residue number fails here, not silently
    inside a ProteinMPNN run.

    Only PDBs actually named in `fixed_positions` are touched; any input
    PDB not mentioned gets no fixed_positions_jsonl entry, matching
    protein_mpnn_run.py's own "missing means nothing is fixed" default.
    """
    pdb_by_basename = {os.path.splitext(os.path.basename(p))[0]: p for p in input_pdbs}
    bad = []
    result: Dict[str, Dict[str, List[int]]] = {}
    for basename, chain_positions in fixed_positions.items():
        pdb_path = pdb_by_basename.get(basename)
        if pdb_path is None:
            bad.append(f"{basename!r}: not among this job's input_pdbs (have {sorted(pdb_by_basename)})")
            continue

        # setup_entities() -- see _read_chain_names()'s comment above for why
        # this is required before get_polymer() will return anything at all
        # against a header-less design PDB. Without it, EVERY residue number
        # looks "not found" (resolved comes back empty), which is exactly
        # what was rejecting perfectly valid fixed_positions here before.
        structure = gemmi.read_structure(pdb_path)
        structure.setup_entities()
        model = structure[0]
        entry: Dict[str, List[int]] = {}
        for chain_name, resnums in chain_positions.items():
            gemmi_chain = model.find_chain(chain_name)
            resolved = {res.seqid.num for res in gemmi_chain.get_polymer()} if gemmi_chain is not None else set()
            not_found = [r for r in resnums if r not in resolved]
            if gemmi_chain is None or not_found:
                bad.append(f"{basename}/{chain_name}: {not_found if gemmi_chain is not None else 'chain not found'}")
                continue
            entry[chain_name] = list(resnums)
        result[basename] = entry

    if bad:
        raise ValueError(f"invalid fixed_positions entries — {bad}")
    return result


def _write_single_line_json(payload: dict, path: str) -> None:
    """protein_mpnn_run.py's own loader for chain_id_jsonl/
    fixed_positions_jsonl keeps only the LAST line of the file, so these
    must always be written as exactly one json.dumps(...) line —
    matching assign_fixed_chains.py's/make_fixed_positions_dict.py's own
    output convention exactly."""
    with open(path, "w") as f:
        f.write(json.dumps(payload) + "\n")


# ============================================================================
# Command construction
# ============================================================================

def _sampling_temp_str(sampling_temp: Union[float, Sequence[float]]) -> str:
    temps = [sampling_temp] if isinstance(sampling_temp, (int, float)) else list(sampling_temp)
    if not temps:
        raise ValueError("sampling_temp must have at least one value")
    return " ".join(str(float(t)) for t in temps)


def _posix(path: str) -> str:
    """
    Converts a Windows-style path to forward-slash form, for embedding into
    a ProteinMPNN CLI argument specifically — NOT for our own file I/O
    (os.makedirs/open/shutil already handle backslash paths correctly and
    keep using them elsewhere in this module; only strings handed to
    ProteinMPNN's own scripts go through this).

    Confirmed directly (not assumed) against a real failure: ProteinMPNN's
    own source repeatedly derives a file's "name" by hand-splitting on "/"
    (e.g. `filename.rfind("/")`) with no guard for "not found" — this is
    the same bug class as the path_to_model_weights issue documented in
    _run_local() below, just triggered during PDB parsing instead: on
    Windows a path has no "/" in it at all, so that split silently returns
    the WRONG slice (the whole path) instead of raising, producing e.g.
    "...mpnn_designs//seqs/C:\\Users\\...\\pdbs\\4V6B-5_ring_0.fa" — a path
    that can never be opened. Windows itself accepts forward slashes for
    every file I/O operation Python performs just as well as backslashes,
    so normalizing to "/" before handing a path to ProteinMPNN's own CLI
    neutralizes this whole bug class at the source, rather than patching
    each new spot it turns up in. No-op on Linux/macOS (os.sep is already
    "/").
    """
    return path.replace(os.sep, "/") if os.sep != "/" else path


def build_parse_command(
    pdbs_dir: str, jsonl_path: str, ca_only: bool = False,
    python_executable: str = "python", parse_script_path: str = "helper_scripts/parse_multiple_chains.py",
) -> List[str]:
    """argv for helper_scripts/parse_multiple_chains.py — turns a folder of
    plain PDBs into the single .jsonl ProteinMPNN's batch mode reads.

    input_path/output_path are normalized via _posix() — see that function's
    docstring: parse_multiple_chains.py derives each PDB's "name" from
    input_path by hand-splitting on "/", which silently breaks on a
    Windows backslash path.
    """
    argv = [
        python_executable, parse_script_path,
        f"--input_path={_posix(pdbs_dir)}", f"--output_path={_posix(jsonl_path)}",
    ]
    if ca_only:
        argv.append("--ca_only")
    return argv


def build_mpnn_command(
    job: ProteinMPNNJob, jsonl_path: str, chain_id_jsonl: Optional[str] = None,
    fixed_positions_jsonl: Optional[str] = None, python_executable: str = "python",
    script_path: str = "protein_mpnn_run.py",
) -> List[str]:
    """
    Builds the argv LIST (never a shell string) for invoking
    protein_mpnn_run.py directly — --sampling_temp's value is
    space-separated (e.g. "0.1 0.2"), and passing it as one argv element
    via "--sampling_temp=0.1 0.2" sidesteps any shell-quoting question
    entirely, since subprocess.Popen given a list never touches a shell.
    """
    argv = [
        python_executable, script_path,
        f"--jsonl_path={_posix(jsonl_path)}",
        f"--out_folder={_posix(job.out_folder)}",
        f"--num_seq_per_target={job.num_seq_per_target}",
        f"--sampling_temp={_sampling_temp_str(job.sampling_temp)}",
        f"--batch_size={job.batch_size}",
        f"--seed={job.seed}",
        f"--model_name={job.model_name}",
        f"--backbone_noise={job.backbone_noise}",
        f"--omit_AAs={job.omit_aas}",
    ]
    if job.ca_only:
        argv.append("--ca_only")
    if job.use_soluble_model:
        argv.append("--use_soluble_model")
    if job.save_score:
        argv.append("--save_score=1")
    if chain_id_jsonl:
        argv.append(f"--chain_id_jsonl={_posix(chain_id_jsonl)}")
    if fixed_positions_jsonl:
        argv.append(f"--fixed_positions_jsonl={_posix(fixed_positions_jsonl)}")
    for key, value in job.extra_overrides.items():
        argv.append(f"--{key}={value}")
    return argv


# ============================================================================
# Non-blocking dispatch
# ============================================================================

@dataclass
class ProteinMPNNRun:
    """Handle to a dispatched (possibly still-running) ProteinMPNN job."""

    job: ProteinMPNNJob
    process: subprocess.Popen
    command: List[str]
    log_path: str
    # The (already-completed-by-the-time-you-see-this) parse_multiple_chains.py
    # call that produced this run's jsonl_path — kept for debugging, not polled.
    parse_command: List[str]
    parse_log_path: str


def _job_work_dir(job: ProteinMPNNJob) -> str:
    return os.path.join(job.out_folder, "_mpnn_inputs")


def _stage_input_pdbs(job: ProteinMPNNJob) -> str:
    """
    Copies job.input_pdbs into a fresh "pdbs/" folder so
    parse_multiple_chains.py's folder-glob (every "*.pdb" file in
    whatever directory it's pointed at) picks up EXACTLY these files —
    never whatever else happens to already be sitting alongside an
    RFdiffusion output_prefix (.trb/.log/prior-run leftovers).
    """
    pdbs_dir = os.path.join(_job_work_dir(job), "pdbs")
    if os.path.exists(pdbs_dir):
        shutil.rmtree(pdbs_dir)
    os.makedirs(pdbs_dir)
    for src in job.input_pdbs:
        if not os.path.exists(src):
            raise FileNotFoundError(f"input PDB not found: {src!r}")
        shutil.copy(src, os.path.join(pdbs_dir, os.path.basename(src)))
    return pdbs_dir


def _clear_stale_outputs(job: ProteinMPNNJob) -> None:
    """
    Removes any "{out_folder}/seqs/<basename>.fa" already on disk for
    THIS job's own input_pdbs before starting a new run — otherwise a
    resubmit against the same out_folder would have poll_status() see
    OLD .fa files and report "completed" before the new process has
    written anything. Only this job's own basenames are touched.
    """
    seqs_dir = os.path.join(job.out_folder, "seqs")
    if not os.path.isdir(seqs_dir):
        return
    for pdb_path in job.input_pdbs:
        basename = os.path.splitext(os.path.basename(pdb_path))[0]
        stale = os.path.join(seqs_dir, f"{basename}.fa")
        if os.path.exists(stale):
            os.remove(stale)


def _clean_pdb_name(name: str) -> str:
    """
    parse_multiple_chains.py derives each entry's "name" by hand-slicing
    its own matched file path on the LAST "/" it can find
    (`fi = biounit.rfind("/"); name = biounit[(fi+1):-4]`) — confirmed
    directly against its source. _posix() (above) normalizes every path WE
    pass in to forward slashes, but that turns out not to be enough on its
    own: parse_multiple_chains.py finds its PDB files via
    `glob.glob(folder_with_pdbs_path + '*.pdb')`, and Python's glob
    module re-joins the folder and each matched filename using the OS-
    native separator for that FINAL join specifically — a real backslash
    on Windows, no matter what separator style the folder path itself
    used. So `rfind("/")` finds the wrong (an earlier, still-forward-
    slash) separator instead, and "name" ends up as something like
    "pdbs\\4V6B-5_ring_0" — a leftover parent-folder fragment glued on —
    instead of just "4V6B-5_ring_0" (confirmed against a real failure:
    protein_mpnn_run.py then can't open
    ".../seqs/pdbs\\4V6B-5_ring_0.fa", since no "pdbs" folder exists
    under seqs/). This re-derives the true basename ourselves from
    whatever ProteinMPNN's own script produced, normalizing BOTH
    separator styles rather than assuming only one survived — correct
    regardless of exactly how the upstream slicing broke, and a no-op on
    a name that was already clean (e.g. on Linux/macOS, where this bug
    never triggers in the first place).
    """
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def _fix_parsed_jsonl_names(jsonl_path: str) -> None:
    """
    Rewrites parsed_pdbs.jsonl in place immediately after
    parse_multiple_chains.py produces it, correcting every entry's
    "name" field via _clean_pdb_name() — see that function's docstring
    for why this is necessary on Windows. Every downstream consumer of
    "name" then sees a correct, consistent value: protein_mpnn_run.py's
    own ali_file/score-file naming, AND our own chain_id_jsonl/
    fixed_positions_jsonl keys (build_chain_id_dict/
    build_fixed_positions_dict key by the exact same "basename without
    extension" convention _clean_pdb_name() produces, so they stay in
    sync with whatever protein_mpnn_run.py ends up reading as "name").
    """
    with open(jsonl_path, "r") as f:
        lines = [line for line in f if line.strip()]

    fixed_lines = []
    for line in lines:
        entry = json.loads(line)
        if "name" in entry:
            entry["name"] = _clean_pdb_name(entry["name"])
        fixed_lines.append(json.dumps(entry))

    with open(jsonl_path, "w") as f:
        f.write("\n".join(fixed_lines) + "\n")


def submit(
    job: ProteinMPNNJob, repo_path: Optional[str] = None, python_executable: Optional[str] = None,
    config: Optional[dict] = None,
) -> ProteinMPNNRun:
    """
    Dispatches `job` locally via subprocess.Popen WITHOUT blocking on the
    actual model run — returns a ProteinMPNNRun handle immediately so a
    NiceGUI request handler (or run(), below) can poll it via
    poll_status() instead of blocking on a multi-minute run.

    repo_path : where your local ProteinMPNN clone lives (the folder
        containing protein_mpnn_run.py and helper_scripts/). Pass it
        directly, or set it once via
        config={"proteinmpnn": {"repo_path": "..."}} (or wire the same
        key into config.py's installation config) so you don't have to
        repeat it on every call.
    python_executable : defaults to "python" (or
        config["proteinmpnn"]["python_executable"]) — point this at a
        specific interpreter/conda env if ProteinMPNN needs one.

    Always runs locally — see module docstring for why a remote/SLURM
    EXECUTION backend for ProteinMPNN itself isn't built yet, and how it
    would slot in later without changing this function's shape.
    """
    tool_config = get_tool_config(config or {}, "proteinmpnn")
    repo_path = repo_path if repo_path is not None else tool_config.get("repo_path")
    python_executable = python_executable or tool_config.get("python_executable", "python")
    script_path = tool_config.get("script_path", "protein_mpnn_run.py")
    parse_script_path = tool_config.get("parse_script_path", "helper_scripts/parse_multiple_chains.py")

    return _run_local(
        job, script_path=script_path, parse_script_path=parse_script_path,
        python_executable=python_executable, cwd=repo_path,
    )


def _conda_env_lib_dir(python_executable: str) -> Optional[str]:
    """
    Best-effort guess at a conda/venv environment's own lib/ directory
    from its python executable's path (<env_root>/bin/python ->
    <env_root>/lib) -- mirrors rfdiffusion.py's function of the same
    name; duplicated here rather than imported since it's a small, self-
    contained helper and this module already stands alone from
    rfdiffusion.py everywhere else. See that copy's docstring for the
    full rationale: a compiled extension inside the env (here: torch)
    can need a newer libstdc++ than whatever the CALLING shell's own
    LD_LIBRARY_PATH already points at, and unlike rfdiffusion.py's SLURM
    backend, ProteinMPNN is ALWAYS invoked via a bare subprocess (see
    module docstring: "LOCAL ONLY") with no shell in between -- so
    `conda activate` never runs and $CONDA_PREFIX is never set, meaning
    this is the ONLY way to fix this for ProteinMPNN, not just the most
    convenient one.

    Returns None if python_executable doesn't look like a
    <env_root>/bin/python layout, or that env_root has no lib/ directory.
    """
    bin_dir = os.path.dirname(os.path.abspath(python_executable))
    if os.path.basename(bin_dir) != "bin":
        return None
    lib_dir = os.path.join(os.path.dirname(bin_dir), "lib")
    return lib_dir if os.path.isdir(lib_dir) else None


def _local_subprocess_env(python_executable: str) -> Optional[dict]:
    """Environment dict for a locally-invoked python_executable subprocess
    -- see rfdiffusion.py's function of the same name. None (a no-op --
    subprocess.Popen/run's own default is to inherit the parent's
    environment unchanged) if _conda_env_lib_dir() found nothing to add."""
    lib_dir = _conda_env_lib_dir(python_executable)
    if lib_dir is None:
        return None
    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    return env


def _run_local(
    job: ProteinMPNNJob, script_path: str = "protein_mpnn_run.py",
    parse_script_path: str = "helper_scripts/parse_multiple_chains.py",
    python_executable: str = "python", cwd: Optional[str] = None,
) -> ProteinMPNNRun:
    # cwd is normally the ProteinMPNN repo itself (protein_mpnn_run.py does
    # its own relative imports assuming that's where it's run from) — so
    # every path handed to the child process as an argv token MUST be
    # absolute, or it resolves against the REPO's directory instead of
    # wherever the caller actually meant. job.out_folder feeds every other
    # path built below (staging dir, jsonl, log, chain_id/fixed_positions
    # jsonl), so resolving it here, once, up front — via dataclasses.replace
    # so the RETURNED ProteinMPNNRun.job carries the same absolute path
    # poll_status() will later glob against — fixes all of them at once.
    # Mirrors rfdiffusion.py's own _submit_singularity handling of exactly
    # this issue for its container backend.
    #
    # Separately: protein_mpnn_run.py's OWN auto-detection of where its model
    # weights live is a confirmed Windows bug, not a SymBro bug — its
    # source does `file_path.rfind("/")` against os.path.realpath(__file__)
    # with no `if k != -1` guard. On Windows, realpath returns a backslash
    # path with no "/" in it at all, so rfind returns -1, and
    # `file_path[:-1]` silently chops just the LAST CHARACTER off the
    # filename ("...protein_mpnn_run.py" -> "...protein_mpnn_run.p") instead
    # of the intended "drop the filename, keep the directory" — then appends
    # "/vanilla_model_weights/..." onto THAT, producing a path that can never
    # exist (confirmed against a real run's proteinmpnn.log: FileNotFoundError
    # on "...protein_mpnn_run.p/vanilla_model_weights/v_48_020.pt"). Passing
    # --path_to_model_weights explicitly bypasses that broken auto-detection
    # entirely (protein_mpnn_run.py only runs it when this arg is absent), so
    # it's always supplied here from job/cwd — which subfolder depends on
    # ca_only/use_soluble_model, same convention protein_mpnn_run.py itself
    # uses. An explicit job.extra_overrides["path_to_model_weights"] (e.g. a
    # custom-trained checkpoint elsewhere) always wins over this default.
    extra_overrides = dict(job.extra_overrides)
    if cwd and "path_to_model_weights" not in extra_overrides:
        weights_subdir = (
            "ca_model_weights" if job.ca_only
            else "soluble_model_weights" if job.use_soluble_model
            else "vanilla_model_weights"
        )
        extra_overrides["path_to_model_weights"] = os.path.join(cwd, weights_subdir)

    job = replace(
        job, out_folder=os.path.abspath(job.out_folder), extra_overrides=extra_overrides,
    )

    os.makedirs(job.out_folder, exist_ok=True)
    _clear_stale_outputs(job)
    pdbs_dir = _stage_input_pdbs(job)
    jsonl_path = os.path.join(_job_work_dir(job), "parsed_pdbs.jsonl")

    # Parsing is fast, model-free coordinate/sequence extraction (no GPU) —
    # unlike the MPNN run itself, blocking on it here is negligible, and
    # it must finish (jsonl_path must exist) before the main run can read it.
    parse_argv = build_parse_command(
        pdbs_dir, jsonl_path, ca_only=job.ca_only,
        python_executable=python_executable, parse_script_path=parse_script_path,
    )
    parse_log_path = os.path.join(_job_work_dir(job), "parse.log")
    local_env = _local_subprocess_env(python_executable)
    with open(parse_log_path, "w") as log_file:
        parse_result = subprocess.run(
            parse_argv, stdout=log_file, stderr=subprocess.STDOUT, cwd=cwd, env=local_env,
        )
    if parse_result.returncode != 0 or not os.path.exists(jsonl_path):
        raise RuntimeError(
            f"parse_multiple_chains.py failed (exit {parse_result.returncode}) — see "
            f"{parse_log_path} for details. Command: {' '.join(parse_argv)}"
        )
    _fix_parsed_jsonl_names(jsonl_path)

    chain_id_jsonl = None
    if job.designed_chains:
        chain_id_dict = build_chain_id_dict(job.input_pdbs, job.designed_chains)
        chain_id_jsonl = os.path.join(_job_work_dir(job), "chain_id.jsonl")
        _write_single_line_json(chain_id_dict, chain_id_jsonl)

    fixed_positions_jsonl = None
    if job.fixed_positions:
        fixed_positions_dict = build_fixed_positions_dict(job.input_pdbs, job.fixed_positions)
        fixed_positions_jsonl = os.path.join(_job_work_dir(job), "fixed_positions.jsonl")
        _write_single_line_json(fixed_positions_dict, fixed_positions_jsonl)

    mpnn_argv = build_mpnn_command(
        job, jsonl_path, chain_id_jsonl=chain_id_jsonl, fixed_positions_jsonl=fixed_positions_jsonl,
        python_executable=python_executable, script_path=script_path,
    )

    # subprocess.Popen (never subprocess.run/check_call, which block until
    # the child exits) so this returns immediately. env=local_env (same
    # dict computed above for the parse step) fixes the GLIBCXX-shadowing
    # class of bug described in _conda_env_lib_dir()'s docstring.
    log_path = os.path.join(job.out_folder, "proteinmpnn.log")
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(mpnn_argv, stdout=log_file, stderr=subprocess.STDOUT, cwd=cwd, env=local_env)

    return ProteinMPNNRun(
        job=job, process=process, command=mpnn_argv, log_path=log_path,
        parse_command=parse_argv, parse_log_path=parse_log_path,
    )


def poll_status(run: ProteinMPNNRun) -> dict:
    """
    Cheap, non-blocking status check — safe to call from a NiceGUI
    ui.timer every second or two. ProteinMPNN writes no completion/
    summary file, so progress is read by counting
    "{out_folder}/seqs/<pdb_basename>.fa" files that have appeared on
    disk, one per input backbone.

    Returns a dict:
        state              : "running" | "completed" | "completed_partial" | "failed"
        returncode         : None while still running, else the process's exit code
        sequences_written  : how many *.fa files exist so far (one per input PDB)
        sequences_expected : len(job.input_pdbs)
        fasta_paths        : sorted list of the *.fa paths found so far
        log_path           : where ProteinMPNN's own stdout/stderr is being captured
    """
    returncode = run.process.poll()

    seqs_dir = os.path.join(run.job.out_folder, "seqs")
    expected_basenames = {os.path.splitext(os.path.basename(p))[0] for p in run.job.input_pdbs}
    fasta_paths = sorted(
        p for p in glob.glob(os.path.join(seqs_dir, "*.fa"))
        if os.path.splitext(os.path.basename(p))[0] in expected_basenames
    )

    if returncode is None:
        state = "running"
    elif returncode != 0:
        state = "failed"
    elif len(fasta_paths) >= len(expected_basenames):
        state = "completed"
    else:
        state = "completed_partial"

    return {
        "state": state,
        "returncode": returncode,
        "sequences_written": len(fasta_paths),
        "sequences_expected": len(expected_basenames),
        "fasta_paths": fasta_paths,
        "log_path": run.log_path,
    }


def cancel(run: ProteinMPNNRun) -> None:
    """Terminates a still-running job — a no-op if it already finished."""
    if run.process.poll() is None:
        run.process.terminate()


# ============================================================================
# Output parsing
# ============================================================================

_SEQUENCE_COLUMNS: Tuple[str, ...] = (
    "source_pdb", "sequence", "is_native", "temperature", "sample_index",
    "score", "global_score", "seq_recovery",
)

# ">T={temp}, sample={n}, score={s}, global_score={g}, seq_recovery={r}" —
# confirmed verbatim against protein_mpnn_run.py's own f.write(...) call
# for every GENERATED sequence record.
_GENERATED_HEADER_RE = re.compile(
    r"T=(?P<temp>[-\d.eE]+),\s*sample=(?P<sample>\d+),\s*score=(?P<score>[-\d.eE]+),\s*"
    r"global_score=(?P<global_score>[-\d.eE]+),\s*seq_recovery=(?P<seq_recovery>[-\d.eE]+)"
)
# ">{name}, score={s}, global_score={g}, fixed_chains=[...], designed_chains=[...], ..." —
# the ONE native-sequence readback record written first in every .fa file.
_NATIVE_HEADER_RE = re.compile(r"score=(?P<score>[-\d.eE]+),\s*global_score=(?P<global_score>[-\d.eE]+)")


def parse_fasta_output(fa_path: str) -> pd.DataFrame:
    """
    Parses one ProteinMPNN "seqs/<basename>.fa" file into a tidy
    DataFrame. The file's first record is always ProteinMPNN's own
    native-sequence readback (no "T=" in its header); every record after
    that is one generated sequence.

    Returns columns: source_pdb, sequence, is_native, temperature,
    sample_index, score, global_score, seq_recovery — temperature/
    sample_index/seq_recovery are None on the native-readback row.
    """
    source_pdb = os.path.splitext(os.path.basename(fa_path))[0]
    with open(fa_path) as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    rows = []
    for i in range(0, len(lines) - 1, 2):
        header, sequence = lines[i].lstrip(">"), lines[i + 1]
        match = _GENERATED_HEADER_RE.search(header)
        if match:
            rows.append({
                "source_pdb": source_pdb, "sequence": sequence, "is_native": False,
                "temperature": float(match["temp"]), "sample_index": int(match["sample"]),
                "score": float(match["score"]), "global_score": float(match["global_score"]),
                "seq_recovery": float(match["seq_recovery"]),
            })
        else:
            native_match = _NATIVE_HEADER_RE.search(header)
            rows.append({
                "source_pdb": source_pdb, "sequence": sequence, "is_native": True,
                "temperature": None, "sample_index": None,
                "score": float(native_match["score"]) if native_match else None,
                "global_score": float(native_match["global_score"]) if native_match else None,
                "seq_recovery": None,
            })

    return pd.DataFrame(rows, columns=list(_SEQUENCE_COLUMNS))


def collect_sequences(run_or_status: Union[ProteinMPNNRun, dict]) -> pd.DataFrame:
    """
    Takes either a ProteinMPNNRun handle (poll_status() is called
    internally) or an already-fetched poll_status() dict, and
    concatenates parse_fasta_output() over every *.fa file written so
    far — safe to call against a "completed_partial" run.

    Returns an empty (but correctly-columned) DataFrame if nothing has
    been written yet.
    """
    status = poll_status(run_or_status) if isinstance(run_or_status, ProteinMPNNRun) else run_or_status
    frames = [parse_fasta_output(p) for p in status.get("fasta_paths", [])]
    if not frames:
        return pd.DataFrame(columns=list(_SEQUENCE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# Design selection — cheap pre-filter before structural validation
# ============================================================================

def select_best_designs(
    sequences_df: pd.DataFrame, top_n: int = 3, per_source: bool = True,
    metric: str = "global_score", exclude_native: bool = True,
) -> pd.DataFrame:
    """
    Ranks ProteinMPNN's own candidates by ProteinMPNN's own confidence in
    them — a free, zero-extra-compute pre-filter meant to run BEFORE
    anything expensive (structural self-consistency screening — see
    alphafold.py's fold-and-compare-RMSD pipeline) on the full candidate
    pool, cutting it down to a manageable shortlist first.

    Why not multiple sequence alignment / a consensus sequence: MSA
    consensus makes sense across evolutionarily related sequences, where
    conservation at a position reflects real selective pressure. These
    candidates aren't that — each is an independent sample from
    ProteinMPNN's own generative model, conditioned on a backbone
    RFdiffusion itself already varied per design (diffused linker
    LENGTH is resampled per design — see
    fixed_positions_from_contig_match()'s own docstring), so candidates
    routinely aren't even the same length and can't be aligned
    column-for-column without introducing gaps. Worse, a per-position
    majority vote across independent samples can stitch together
    residues no single ProteinMPNN pass ever validated TOGETHER as one
    coherent sequence — undermining exactly the "scored as a whole"
    property fixed-backbone design is supposed to give you. Ranking by
    the model's own score sidesteps both problems entirely, using a
    signal ProteinMPNN already computed.

    metric : "global_score" (default) or "score" — both parsed straight
        out of ProteinMPNN's own FASTA header by parse_fasta_output(),
        nothing computed here, just sorted/filtered. Lower is better in
        both cases (negative log-likelihood under the model — more
        plausible for this structure). "global_score" (whole-sequence)
        is the better default over "score" (computed per DESIGNED
        position only), since each design's fixed/designed split
        differs — exactly this project's case, where linker length
        varies per design — making "score" not directly comparable
        across designs the way "global_score" is. seq_recovery is
        deliberately not offered as a metric here: for a genuinely NEW
        linker, high recovery of some reference sequence isn't a design
        goal at all.

    per_source : if True (the default), ranks and takes the top_n WITHIN
        each source_pdb group separately — the best top_n linker
        sequences PER RFdiffusion design — rather than collapsing to one
        global top_n that could all come from a single design and leave
        every other design's shortlist empty.

    exclude_native : drops ProteinMPNN's own native-sequence readback row
        (is_native=True — see parse_fasta_output()) before ranking; that
        row isn't a generated candidate.

    Returns a copy of sequences_df filtered to the selected rows, best
    first, with an added "rank" column (1 = best; restarts at 1 within
    each source_pdb group when per_source=True).
    """
    if metric not in ("score", "global_score"):
        raise ValueError(f"metric must be 'score' or 'global_score' — got {metric!r}")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1 — got {top_n}")

    required_cols = {"source_pdb", "is_native", metric}
    missing = required_cols - set(sequences_df.columns)
    if missing:
        raise ValueError(
            f"sequences_df is missing column(s) {sorted(missing)} — expected the shape "
            f"collect_sequences()/parse_fasta_output() return."
        )

    df = sequences_df[~sequences_df["is_native"]] if exclude_native else sequences_df
    if df.empty:
        return df.assign(rank=pd.Series(dtype=int))

    if per_source:
        selected = (
            df.sort_values(metric)
            .groupby("source_pdb", sort=False, group_keys=False)
            .head(top_n)
            .copy()
        )
        selected["rank"] = selected.groupby("source_pdb").cumcount() + 1
        sort_cols = ["source_pdb", "rank"]
    else:
        selected = df.sort_values(metric).head(top_n).copy()
        selected["rank"] = range(1, len(selected) + 1)
        sort_cols = ["rank"]

    return selected.sort_values(sort_cols).reset_index(drop=True)


# ============================================================================
# The one-call convenience wrapper (see QUICKSTART at the top of this file)
# ============================================================================

def run(
    source, repo_path: Optional[str] = None, out_folder: Optional[str] = None,
    designed_chains: Optional[Sequence[str]] = None,
    fixed_positions: Optional[Dict[str, Dict[str, Sequence[int]]]] = None,
    auto_fix_from_rfdiffusion: bool = True,
    num_seq_per_target: int = 8, sampling_temp: Union[float, Sequence[float]] = 0.1,
    batch_size: int = 8, seed: int = 37, python_executable: Optional[str] = None,
    config: Optional[dict] = None, poll_interval: float = 2.0, **job_kwargs,
) -> pd.DataFrame:
    """
    The simplest way to use this module end to end: prepare_mpnn_job() +
    submit() + poll until done + collect_sequences(), in one blocking
    call. See the QUICKSTART at the top of this file for examples.

    Blocks the calling thread until ProteinMPNN finishes — use
    prepare_mpnn_job()/submit()/poll_status() directly instead if you
    need this to be non-blocking (e.g. driving a NiceGUI progress panel;
    both routes share the exact same underlying job/run objects).

    Raises RuntimeError if the run fails (non-zero exit) rather than
    returning a result — check `log_path` (mentioned in the error) for
    ProteinMPNN's own stdout/stderr.
    """
    job = prepare_mpnn_job(
        source, out_folder=out_folder, designed_chains=designed_chains,
        fixed_positions=fixed_positions, auto_fix_from_rfdiffusion=auto_fix_from_rfdiffusion,
        num_seq_per_target=num_seq_per_target, sampling_temp=sampling_temp,
        batch_size=batch_size, seed=seed, **job_kwargs,
    )
    run_handle = submit(job, repo_path=repo_path, python_executable=python_executable, config=config)

    status = poll_status(run_handle)
    while status["state"] == "running":
        time.sleep(poll_interval)
        status = poll_status(run_handle)

    if status["state"] == "failed":
        raise RuntimeError(
            f"ProteinMPNN failed (exit {status['returncode']}) — see {status['log_path']} "
            f"for its own stdout/stderr."
        )

    return collect_sequences(status)


# ============================================================================
# MSA generation — deferred (see module docstring)
# ============================================================================

def generate_msa(sequence: str, **kwargs):
    """
    NOT YET IMPLEMENTED.

    Per project discussion, MSA generation for downstream AlphaFold2
    validation of ProteinMPNN-designed sequences is deliberately out of
    scope for this pass — the immediate goal was getting ProteinMPNN
    itself generating sequences locally. Left as a defined, importable
    entry point so calling code has a stable name to call once this IS
    built.

    Candidates to evaluate when this gets picked up: the free ColabFold
    MMseqs2 search API (api.colabfold.com, no local database/install
    needed), versus a local MMseqs2/HHblits install against a real
    sequence database (a better fit once the HPC/SLURM backend work
    happens).

    collect_sequences() already gives you the designed sequences
    themselves; MSA / AlphaFold2 validation is a separate, not-yet-built
    step downstream of that.
    """
    raise NotImplementedError(
        "MSA generation is deferred — see generate_msa()'s docstring. "
        "collect_sequences() already gives you ProteinMPNN's designed sequences; "
        "MSA/AlphaFold2 validation is a separate, not-yet-built step."
    )
