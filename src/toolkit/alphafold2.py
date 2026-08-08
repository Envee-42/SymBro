"""
alphafold2.py — structural self-consistency screening for ProteinMPNN-
designed sequences via ColabFold (AlphaFold2 + automatic MSA search):
folds each candidate back and checks whether the PREDICTED structure
actually matches the backbone it was designed for (low CA-RMSD, high
pLDDT). See pmpnn.select_best_designs() for the cheap upstream pre-filter
(ProteinMPNN's own score) this is meant to run after, and
selfconsistency.py for the RMSD/pLDDT/results machinery every predictor
backend (this file, boltz.py, af3.py) shares.

QUICKSTART:

    from toolkit import pmpnn, alphafold2

    sequences_df = pmpnn.collect_sequences(mpnn_run)
    shortlist = pmpnn.select_best_designs(sequences_df, top_n=3)
    winners = alphafold2.run(shortlist, design_paths, config=cfg)

LICENSING — confirmed directly against each project's own license files,
not recalled from memory: ColabFold (sokrypton/ColabFold) is MIT.
AlphaFold2's own source is Apache 2.0, and — the piece that actually
matters here — its TRAINED PARAMETERS were relicensed to CC BY 4.0 in
2022 (see google-deepmind/alphafold's own README/LICENSE): permissive,
commercial use is fine, attribution required. MMseqs2 (ColabFold's MSA
search engine) is MIT. Nothing in this backend's dependency chain
restricts commercial or redistributable use — this is the "safe by
default" predictor. Compare af3.py's module docstring, whose weights
carry real restrictions this one doesn't.

Why ColabFold specifically, not the ESMFold public API: see this
module's git history / the original alphafold.py — ESMFold's free REST
API was evaluated and rejected as actively broken (an incomplete SSL
certificate chain) with its repo archived.

Backends ("local"/"singularity"/"slurm") mirror rfdiffusion.py's
exactly, on purpose — same config shape, same submit(job, backend=...,
config=...) call convention. See rfdiffusion.py's module docstring for
why "slurm" wraps rather than replaces the other two, and
selfconsistency.py's for why the SLURM plumbing itself is imported
rather than reimplemented here.
"""

import glob
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from toolkit.config import get_tool_config
from toolkit.selfconsistency import (
    build_reference_map, collect_results, select_validated_designs,
    submit_via_slurm, terminate_or_cancel,
)

# Re-exported so callers that only import alphafold2 still get the full
# public surface (select_validated_designs/collect_results are otherwise
# identical across every backend — see selfconsistency.py).
__all__ = [
    "AlphaFold2Job", "AlphaFold2Run", "prepare_self_consistency_job", "build_command",
    "submit", "poll_status", "cancel", "collect_results", "select_validated_designs", "run",
]


# ============================================================================
# Job spec
# ============================================================================

@dataclass
class AlphaFold2Job:
    """A fully-specified, ready-to-submit ColabFold batch fold call."""

    fasta_path: str
    out_dir: str
    reference_map: Dict[str, dict]  # selfconsistency.build_reference_map()'s own return shape
    num_models: int = 1
    num_recycles: Optional[int] = None
    use_templates: bool = False
    extra_flags: List[str] = field(default_factory=list)


def prepare_self_consistency_job(
    selected_df: pd.DataFrame, design_paths: Sequence[str], out_dir: Optional[str] = None,
    num_models: int = 1, num_recycles: Optional[int] = None, use_templates: bool = False,
    extra_flags: Optional[Sequence[str]] = None,
) -> AlphaFold2Job:
    """
    Builds a submit()-ready AlphaFold2Job straight from
    pmpnn.select_best_designs()'s own output shape (see
    selfconsistency.build_reference_map() for the candidate-id/reference
    matching logic every backend shares). out_dir defaults to a
    "self_consistency" folder next to the first reference design PDB —
    same workspace-visible-scratch-folder philosophy as every other
    module here.
    """
    reference_map = build_reference_map(selected_df, design_paths)

    if out_dir is None:
        first_ref = next(iter(reference_map.values()))["reference_pdb"]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(first_ref)) or ".", "self_consistency")
    os.makedirs(out_dir, exist_ok=True)

    fasta_path = os.path.join(out_dir, "candidates.fasta")
    with open(fasta_path, "w") as f:
        for candidate_id, entry in reference_map.items():
            f.write(f">{candidate_id}\n{entry['sequence']}\n")

    return AlphaFold2Job(
        fasta_path=fasta_path, out_dir=out_dir, reference_map=reference_map,
        num_models=num_models, num_recycles=num_recycles, use_templates=use_templates,
        extra_flags=list(extra_flags or []),
    )


def build_command(job: AlphaFold2Job, colabfold_executable: str = "colabfold_batch") -> List[str]:
    """argv list (never a shell string) for colabfold_batch — flag names
    confirmed against ColabFold's own README/CLI help."""
    argv = [colabfold_executable, job.fasta_path, job.out_dir, "--num-models", str(job.num_models)]
    if job.num_recycles is not None:
        argv += ["--num-recycle", str(job.num_recycles)]
    if job.use_templates:
        argv.append("--templates")
    argv += list(job.extra_flags)
    return argv


# ============================================================================
# Non-blocking dispatch
# ============================================================================

@dataclass
class AlphaFold2Run:
    """Handle to a dispatched (possibly still-running) ColabFold job."""

    job: AlphaFold2Job
    process: Optional[subprocess.Popen]
    command: List[str]
    log_path: str
    slurm_job_id: Optional[str] = None
    sbatch_script_path: Optional[str] = None


def submit(job: AlphaFold2Job, backend: Optional[str] = None, config: Optional[dict] = None, **backend_kwargs) -> AlphaFold2Run:
    """Dispatches `job` without blocking — see rfdiffusion.submit()'s
    docstring, which this mirrors exactly (backend resolution order:
    explicit backend= > config["alphafold2"]["backend"] > "local")."""
    tool_config = get_tool_config(config or {}, "alphafold2")
    if backend is None:
        backend = tool_config.get("backend", "local")

    if backend == "local":
        defaults = {"colabfold_executable": tool_config.get("colabfold_executable", "colabfold_batch")}
        return _submit_local(job, **{**defaults, **backend_kwargs})

    if backend == "singularity":
        defaults = {
            "image": tool_config.get("singularity_image"),
            "executable": tool_config.get("singularity_executable", "singularity"),
            "bind_paths": tool_config.get("bind_paths", []),
            "use_gpu": tool_config.get("use_gpu", True),
        }
        merged = {**defaults, **backend_kwargs}
        if not merged.get("image"):
            raise ValueError(
                "backend='singularity' needs an 'image' (the .sif path, e.g. ColabFold's own "
                "published image) — pass image=... directly, or set "
                "alphafold2.singularity_image in your installation config."
            )
        return _submit_singularity(job, **merged)

    if backend == "slurm":
        slurm_config = dict(tool_config.get("slurm") or {})
        inner_backend = tool_config.get("inner_backend", "local")
        defaults = {
            "inner_backend": inner_backend,
            "colabfold_executable": tool_config.get("colabfold_executable", "colabfold_batch"),
            "image": tool_config.get("singularity_image"),
            "singularity_executable": tool_config.get("singularity_executable", "singularity"),
            "bind_paths": tool_config.get("bind_paths", []),
            "use_gpu": tool_config.get("use_gpu", True),
            "partition": slurm_config.get("partition"),
            "account": slurm_config.get("account"),
            "time": slurm_config.get("time", "04:00:00"),
            "gres": slurm_config.get("gres"),
            "gpus": slurm_config.get("gpus"),
            "cpus_per_task": slurm_config.get("cpus_per_task", 4),
            "mem": slurm_config.get("mem", "16G"),
            "job_name": slurm_config.get("job_name", "alphafold2"),
            "setup_lines": slurm_config.get("setup_lines", []),
            "extra_sbatch_directives": slurm_config.get("extra_sbatch_directives", []),
            "sbatch_executable": slurm_config.get("sbatch_executable", "sbatch"),
        }
        merged = {**defaults, **backend_kwargs}
        if merged["inner_backend"] == "singularity" and not merged.get("image"):
            raise ValueError(
                "backend='slurm' with inner_backend='singularity' needs an 'image' — pass "
                "image=... directly, or set alphafold2.singularity_image in your installation "
                "config."
            )
        return _submit_slurm(job, **merged)

    raise NotImplementedError(
        f"backend {backend!r} is not implemented yet — 'local', 'singularity', and 'slurm' are "
        f"available today."
    )


def _submit_local(job: AlphaFold2Job, colabfold_executable: str = "colabfold_batch") -> AlphaFold2Run:
    argv = build_command(job, colabfold_executable=colabfold_executable)
    os.makedirs(job.out_dir, exist_ok=True)
    log_path = os.path.join(job.out_dir, "colabfold.log")
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)
    return AlphaFold2Run(job=job, process=process, command=argv, log_path=log_path)


def _build_singularity_argv(
    job: AlphaFold2Job, image: str, executable: str = "singularity",
    bind_paths: Sequence[str] = (), use_gpu: bool = True,
) -> List[str]:
    """Confirmed against a real production ColabFold/Singularity HPC
    deployment (bartongroup/Colabfold_batch_installer): "singularity
    exec --nv -B host:container <image> colabfold_batch <fasta>
    <out_dir> ..." — exec, not run (ColabFold's image has no entrypoint
    that already knows to invoke colabfold_batch, unlike prosculpt's
    RFdiffusion image)."""
    fasta_dir = os.path.dirname(os.path.abspath(job.fasta_path))
    out_dir_abs = os.path.abspath(job.out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    mount_dirs = sorted({fasta_dir, out_dir_abs})

    argv = [executable, "exec"]
    if use_gpu:
        argv.append("--nv")
    for d in mount_dirs:
        argv += ["-B", f"{d}:{d}"]
    for extra in bind_paths:
        argv += ["-B", extra]
    argv.append(image)
    argv += build_command(job, colabfold_executable="colabfold_batch")
    return argv


def _submit_singularity(
    job: AlphaFold2Job, image: str, executable: str = "singularity",
    bind_paths: Sequence[str] = (), use_gpu: bool = True,
) -> AlphaFold2Run:
    argv = _build_singularity_argv(job, image, executable=executable, bind_paths=bind_paths, use_gpu=use_gpu)
    os.makedirs(job.out_dir, exist_ok=True)
    log_path = os.path.join(job.out_dir, "colabfold.log")
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)
    return AlphaFold2Run(job=job, process=process, command=argv, log_path=log_path)


def _submit_slurm(
    job: AlphaFold2Job, inner_backend: str = "local", colabfold_executable: str = "colabfold_batch",
    image: Optional[str] = None, singularity_executable: str = "singularity",
    bind_paths: Sequence[str] = (), use_gpu: bool = True,
    partition: Optional[str] = None, account: Optional[str] = None, time: str = "04:00:00",
    gres: Optional[str] = None, gpus: Optional[int] = None, cpus_per_task: int = 4, mem: str = "16G",
    job_name: str = "alphafold2", setup_lines: Sequence[str] = (),
    extra_sbatch_directives: Sequence[str] = (), sbatch_executable: str = "sbatch",
) -> AlphaFold2Run:
    os.makedirs(job.out_dir, exist_ok=True)

    if inner_backend == "local":
        inner_argv = build_command(job, colabfold_executable=colabfold_executable)
    elif inner_backend == "singularity":
        if not image:
            raise ValueError("_submit_slurm(inner_backend='singularity') needs image=... (the .sif path)")
        inner_argv = _build_singularity_argv(job, image, executable=singularity_executable, bind_paths=bind_paths, use_gpu=use_gpu)
    else:
        raise ValueError(f"inner_backend must be 'local' or 'singularity' — got {inner_backend!r}")

    log_path = os.path.join(job.out_dir, "colabfold.log")
    sbatch_script_path = os.path.join(job.out_dir, "colabfold.sbatch.sh")
    slurm_job_id, sbatch_script_path = submit_via_slurm(
        inner_argv, out_dir=job.out_dir, log_path=log_path, sbatch_script_path=sbatch_script_path,
        job_name=job_name, partition=partition, account=account, time=time, gres=gres, gpus=gpus,
        cpus_per_task=cpus_per_task, mem=mem, setup_lines=setup_lines,
        extra_sbatch_directives=extra_sbatch_directives, sbatch_executable=sbatch_executable,
    )
    return AlphaFold2Run(
        job=job, process=None, command=inner_argv, log_path=log_path,
        slurm_job_id=slurm_job_id, sbatch_script_path=sbatch_script_path,
    )


_RANK1_RE = re.compile(r"rank_0*1(?!\d)", re.IGNORECASE)


def _find_top_rank_pdb(out_dir: str, candidate_id: str) -> Optional[str]:
    """ColabFold's output filename schema has changed across versions
    ("{id}_unrelaxed_rank_1_model_1.pdb" vs
    "{id}_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb") — matches
    EITHER via a "rank_1"/"rank_001" token (never "rank_2"/"rank_10").
    Prefers "_relaxed_" over "_unrelaxed_" when both exist."""
    matches = [
        p for p in glob.glob(os.path.join(out_dir, f"{candidate_id}_*.pdb"))
        if _RANK1_RE.search(os.path.basename(p))
    ]
    if not matches:
        return None
    relaxed = sorted(p for p in matches if "_relaxed_" in os.path.basename(p) and "_unrelaxed_" not in os.path.basename(p))
    return relaxed[0] if relaxed else sorted(matches)[0]


def poll_status(run: AlphaFold2Run) -> dict:
    """Cheap, non-blocking status check — same "count output files on
    disk" convention every module here uses. Returns state/returncode/
    candidates_folded/candidates_expected/folded_paths/log_path."""
    candidate_ids = list(run.job.reference_map)
    folded_paths = {cid: p for cid in candidate_ids if (p := _find_top_rank_pdb(run.job.out_dir, cid))}

    if run.slurm_job_id is not None:
        from toolkit.selfconsistency import slurm_returncode
        returncode = slurm_returncode(run.slurm_job_id)
        if returncode is None and len(folded_paths) >= len(candidate_ids):
            returncode = 0
    else:
        returncode = run.process.poll()

    if returncode is None:
        state = "running"
    elif returncode != 0:
        state = "failed"
    elif len(folded_paths) >= len(candidate_ids):
        state = "completed"
    else:
        state = "completed_partial"

    return {
        "state": state, "returncode": returncode, "candidates_folded": len(folded_paths),
        "candidates_expected": len(candidate_ids), "folded_paths": folded_paths, "log_path": run.log_path,
    }


def cancel(run: AlphaFold2Run) -> None:
    """Terminates a still-running job — a no-op if it already finished."""
    terminate_or_cancel(run.process, run.slurm_job_id)


# ============================================================================
# The one-call convenience wrapper
# ============================================================================

def run(
    selected_df: pd.DataFrame, design_paths: Sequence[str], backend: Optional[str] = None,
    config: Optional[dict] = None, poll_interval: float = 5.0,
    max_rmsd: float = 2.0, min_plddt: float = 70.0, **job_kwargs,
) -> pd.DataFrame:
    """prepare_self_consistency_job() + submit() + poll until done +
    collect_results() + select_validated_designs(), in one blocking
    call. Raises RuntimeError on a failed run (see log_path)."""
    job = prepare_self_consistency_job(selected_df, design_paths, **job_kwargs)
    run_handle = submit(job, backend=backend, config=config)

    status = poll_status(run_handle)
    while status["state"] == "running":
        time.sleep(poll_interval)
        status = poll_status(run_handle)

    if status["state"] == "failed":
        raise RuntimeError(
            f"colabfold_batch failed (exit {status['returncode']}) — see {status['log_path']} "
            f"for its own stdout/stderr."
        )

    results_df = collect_results(status["folded_paths"], job.reference_map)
    return select_validated_designs(results_df, max_rmsd=max_rmsd, min_plddt=min_plddt)
