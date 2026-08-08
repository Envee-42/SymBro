"""
af3.py — structural self-consistency screening for ProteinMPNN-designed
sequences via AlphaFold3 (google-deepmind/alphafold3): folds each
candidate back and checks whether the PREDICTED structure actually
matches the backbone it was designed for (low CA-RMSD, high pLDDT).
Same QUICKSTART/role as alphafold2.py/boltz.py — see those modules'
docstrings for the overall pattern, and selfconsistency.py for the
RMSD/pLDDT/results machinery all three backends share.

    from toolkit import pmpnn, af3

    shortlist = pmpnn.select_best_designs(sequences_df, top_n=3)
    winners = af3.run(
        shortlist, design_paths, model_dir="/path/to/af3/models",
        db_dir="/path/to/af3/public_databases", terms_acknowledged=True,
        config=cfg,
    )

LICENSING — read this before using this backend. Confirmed directly
against google-deepmind/alphafold3's own README/WEIGHTS_TERMS_OF_USE.md,
not recalled from memory:

  - The SOURCE CODE is Apache 2.0 — no restriction there.
  - The MODEL WEIGHTS are NOT: they're available only by requesting them
    directly from Google (a form, approval at Google's sole discretion),
    the terms restrict use to NON-COMMERCIAL purposes, prohibit
    redistributing the weights to anyone else, and prohibit using AF3
    outputs to train a competing structure-prediction model. The free
    web version (AlphaFold Server) carries the same non-commercial
    restriction and isn't meant for bulk/programmatic access either.

This module NEVER fetches, bundles, or redistributes AF3 weights —
model_dir must point at weights the USER separately obtained directly
from Google under those terms (same "orchestration only" pattern this
whole project already uses for RFdiffusion/ProteinMPNN/ColabFold/Boltz,
none of which this project bundles either — but AF3's weights terms are
meaningfully more restrictive than any of those, which is why this
module adds an extra guard the others don't need: submit() raises
unless you explicitly pass terms_acknowledged=True (or set
af3.terms_acknowledged: true in your installation config), so nobody
runs an AF3 job without having actually seen this notice first. If
symseeker is heading toward a commercial or bulk-served use, this
backend specifically is very likely NOT usable for that under Google's
current terms — alphafold2.py (CC BY 4.0 weights) and boltz.py (MIT,
code and weights) don't have that problem. Get real legal review before
leaning on this reasoning for anything beyond individual academic use;
this is my own reading of the published terms, not legal advice.

SCOPE — this module only builds single-chain PROTEIN jobs (matching
what ProteinMPNN actually produces here). AF3 itself also models
nucleic acids, ligands, and ions, none of which this project's pipeline
currently needs — see WHAT source CAN BE below if that ever changes.

INPUT/OUTPUT — confirmed against google-deepmind/alphafold3's own
docs/input.md and docs/output.md: run_alphafold.py accepts either a
single --json_path or a directory of JSON files via --input_dir (used
here, one JSON per candidate, matching alphafold2.py's/boltz.py's own
batch-directory approach). Each JSON's minimal schema is
{"name", "sequences": [{"protein": {"id", "sequence"}}], "modelSeeds",
"dialect": "alphafold3", "version": 1}. Output lands at
"{output_dir}/{name.lower()}/{name.lower()}_model.cif", so candidate
ids (already filesystem-sanitized by selfconsistency.sanitize_id) are
lowercased again here to match AF3's own folder-naming convention
before looking for output.

INFRASTRUCTURE, honestly flagged: unlike ColabFold/Boltz (which default
to a free hosted MSA search API), AF3's own data pipeline needs a large
local genetic-database download (hundreds of GB) unless you disable it
(run_data_pipeline=False) and supply pre-computed MSAs some other way —
this project doesn't build that "bring your own MSA" path today, so a
real "local"/"singularity" run needs db_dir set up in full. This is a
materially heavier local install than either other backend.
"""

import glob
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from toolkit.config import get_tool_config
from toolkit.selfconsistency import (
    build_reference_map, collect_results, select_validated_designs,
    slurm_returncode, submit_via_slurm, terminate_or_cancel,
)

__all__ = [
    "AF3Job", "AF3Run", "prepare_self_consistency_job", "build_command",
    "submit", "poll_status", "cancel", "collect_results", "select_validated_designs", "run",
]

WEIGHTS_TERMS_URL = "https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md"


def _require_terms_acknowledged(terms_acknowledged: bool) -> None:
    if not terms_acknowledged:
        raise ValueError(
            "af3.submit() needs an explicit terms_acknowledged=True before it will run — "
            "AlphaFold3's model weights are restricted to non-commercial use, must be obtained "
            "directly from Google (not bundled or redistributed by this project), and cannot be "
            f"redistributed further. Read {WEIGHTS_TERMS_URL} before setting this, or set "
            "af3.terms_acknowledged: true once in your installation config if you've already "
            "reviewed it."
        )


# ============================================================================
# Job spec
# ============================================================================

@dataclass
class AF3Job:
    """A fully-specified, ready-to-submit AlphaFold3 batch fold call."""

    input_dir: str
    out_dir: str
    reference_map: Dict[str, dict]
    model_dir: str
    db_dir: Optional[str] = None
    run_data_pipeline: bool = True
    run_inference: bool = True
    extra_flags: List[str] = field(default_factory=list)


def _write_candidate_json(path: str, candidate_id: str, sequence: str) -> None:
    payload = {
        "name": candidate_id,
        "sequences": [{"protein": {"id": ["A"], "sequence": sequence}}],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def prepare_self_consistency_job(
    selected_df: pd.DataFrame, design_paths: Sequence[str], model_dir: str,
    out_dir: Optional[str] = None, db_dir: Optional[str] = None,
    run_data_pipeline: bool = True, run_inference: bool = True,
    extra_flags: Optional[Sequence[str]] = None,
) -> AF3Job:
    """
    Builds a submit()-ready AF3Job straight from
    pmpnn.select_best_designs()'s own output shape (see
    selfconsistency.build_reference_map()). Writes one JSON file per
    candidate into "{out_dir}/inputs/" — the directory run_alphafold.py's
    own --input_dir batch mode expects (see module docstring).

    model_dir : REQUIRED, no default — must point at AF3 weights YOU
        separately obtained directly from Google under their terms (see
        module docstring). This module never fetches or assumes a
        default location, on purpose.
    db_dir : the genetic/template database directory run_data_pipeline
        needs. Required unless run_data_pipeline=False (e.g. you're
        supplying pre-computed MSAs some other way outside this module).
    """
    if run_data_pipeline and not db_dir:
        raise ValueError(
            "db_dir is required when run_data_pipeline=True (the default) — AF3's own data "
            "pipeline needs its genetic/template database directory. Pass db_dir=..., or set "
            "run_data_pipeline=False if you're supplying MSAs some other way."
        )

    reference_map = build_reference_map(selected_df, design_paths)

    if out_dir is None:
        first_ref = next(iter(reference_map.values()))["reference_pdb"]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(first_ref)) or ".", "self_consistency")
    input_dir = os.path.join(out_dir, "inputs")
    os.makedirs(input_dir, exist_ok=True)

    for candidate_id, entry in reference_map.items():
        _write_candidate_json(os.path.join(input_dir, f"{candidate_id}.json"), candidate_id, entry["sequence"])

    return AF3Job(
        input_dir=input_dir, out_dir=out_dir, reference_map=reference_map, model_dir=model_dir,
        db_dir=db_dir, run_data_pipeline=run_data_pipeline, run_inference=run_inference,
        extra_flags=list(extra_flags or []),
    )


def build_command(job: AF3Job, python_executable: str = "python", script_path: str = "run_alphafold.py") -> List[str]:
    """argv list for run_alphafold.py — flags (--input_dir, --model_dir,
    --output_dir, --db_dir, --run_data_pipeline, --run_inference)
    confirmed against google-deepmind/alphafold3's own README/docs."""
    argv = [
        python_executable, script_path,
        f"--input_dir={job.input_dir}", f"--model_dir={job.model_dir}", f"--output_dir={job.out_dir}",
        f"--run_data_pipeline={'true' if job.run_data_pipeline else 'false'}",
        f"--run_inference={'true' if job.run_inference else 'false'}",
    ]
    if job.db_dir:
        argv.append(f"--db_dir={job.db_dir}")
    argv += list(job.extra_flags)
    return argv


# ============================================================================
# Non-blocking dispatch
# ============================================================================

@dataclass
class AF3Run:
    """Handle to a dispatched (possibly still-running) AlphaFold3 job."""

    job: AF3Job
    process: Optional[subprocess.Popen]
    command: List[str]
    log_path: str
    slurm_job_id: Optional[str] = None
    sbatch_script_path: Optional[str] = None


def submit(
    job: AF3Job, backend: Optional[str] = None, config: Optional[dict] = None,
    terms_acknowledged: bool = False, **backend_kwargs,
) -> AF3Run:
    """
    Dispatches `job` without blocking — same backend resolution order as
    every other module here: explicit backend= > config["af3"]["backend"]
    > "local".

    Raises ValueError unless terms_acknowledged=True (or
    config["af3"]["terms_acknowledged"]: true) — see module docstring
    and _require_terms_acknowledged().
    """
    tool_config = get_tool_config(config or {}, "af3")
    _require_terms_acknowledged(terms_acknowledged or bool(tool_config.get("terms_acknowledged", False)))

    if backend is None:
        backend = tool_config.get("backend", "local")

    if backend == "local":
        defaults = {
            "python_executable": tool_config.get("python_executable", "python"),
            "script_path": tool_config.get("script_path", "run_alphafold.py"),
        }
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
                "backend='singularity' needs an 'image' (the .sif path, e.g. built from AF3's "
                "own published Dockerfile) — pass image=... directly, or set "
                "af3.singularity_image in your installation config."
            )
        return _submit_singularity(job, **merged)

    if backend == "slurm":
        slurm_config = dict(tool_config.get("slurm") or {})
        inner_backend = tool_config.get("inner_backend", "local")
        defaults = {
            "inner_backend": inner_backend,
            "python_executable": tool_config.get("python_executable", "python"),
            "script_path": tool_config.get("script_path", "run_alphafold.py"),
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
            "job_name": slurm_config.get("job_name", "af3"),
            "setup_lines": slurm_config.get("setup_lines", []),
            "extra_sbatch_directives": slurm_config.get("extra_sbatch_directives", []),
            "sbatch_executable": slurm_config.get("sbatch_executable", "sbatch"),
        }
        merged = {**defaults, **backend_kwargs}
        if merged["inner_backend"] == "singularity" and not merged.get("image"):
            raise ValueError(
                "backend='slurm' with inner_backend='singularity' needs an 'image' — pass "
                "image=... directly, or set af3.singularity_image in your installation config."
            )
        return _submit_slurm(job, **merged)

    raise NotImplementedError(
        f"backend {backend!r} is not implemented yet — 'local', 'singularity', and 'slurm' are "
        f"available today."
    )


def _submit_local(job: AF3Job, python_executable: str = "python", script_path: str = "run_alphafold.py") -> AF3Run:
    argv = build_command(job, python_executable=python_executable, script_path=script_path)
    os.makedirs(job.out_dir, exist_ok=True)
    log_path = os.path.join(job.out_dir, "af3.log")
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)
    return AF3Run(job=job, process=process, command=argv, log_path=log_path)


def _build_singularity_argv(
    job: AF3Job, image: str, executable: str = "singularity",
    bind_paths: Sequence[str] = (), use_gpu: bool = True,
) -> List[str]:
    """Same "singularity exec --nv -B host:container <image> <command>"
    shape as alphafold2.py's/boltz.py's own _build_singularity_argv() —
    AF3's own README shows a Docker invocation (`docker run ... python
    run_alphafold.py ...`, i.e. no baked-in entrypoint), so the command
    is named explicitly here too, and model_dir/db_dir/out_dir/input_dir
    all need their own bind mounts since none of them typically live
    inside the fasta/out_dir tree the way ColabFold's/Boltz's single
    working directory does."""
    dirs_to_mount = {job.input_dir, os.path.abspath(job.out_dir), job.model_dir}
    if job.db_dir:
        dirs_to_mount.add(job.db_dir)
    os.makedirs(job.out_dir, exist_ok=True)

    argv = [executable, "exec"]
    if use_gpu:
        argv.append("--nv")
    for d in sorted(os.path.abspath(d) for d in dirs_to_mount):
        argv += ["-B", f"{d}:{d}"]
    for extra in bind_paths:
        argv += ["-B", extra]
    argv.append(image)
    argv += build_command(job, python_executable="python", script_path="run_alphafold.py")
    return argv


def _submit_singularity(
    job: AF3Job, image: str, executable: str = "singularity",
    bind_paths: Sequence[str] = (), use_gpu: bool = True,
) -> AF3Run:
    argv = _build_singularity_argv(job, image, executable=executable, bind_paths=bind_paths, use_gpu=use_gpu)
    os.makedirs(job.out_dir, exist_ok=True)
    log_path = os.path.join(job.out_dir, "af3.log")
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)
    return AF3Run(job=job, process=process, command=argv, log_path=log_path)


def _submit_slurm(
    job: AF3Job, inner_backend: str = "local", python_executable: str = "python",
    script_path: str = "run_alphafold.py", image: Optional[str] = None,
    singularity_executable: str = "singularity", bind_paths: Sequence[str] = (), use_gpu: bool = True,
    partition: Optional[str] = None, account: Optional[str] = None, time: str = "04:00:00",
    gres: Optional[str] = None, gpus: Optional[int] = None, cpus_per_task: int = 4, mem: str = "16G",
    job_name: str = "af3", setup_lines: Sequence[str] = (),
    extra_sbatch_directives: Sequence[str] = (), sbatch_executable: str = "sbatch",
) -> AF3Run:
    os.makedirs(job.out_dir, exist_ok=True)

    if inner_backend == "local":
        inner_argv = build_command(job, python_executable=python_executable, script_path=script_path)
    elif inner_backend == "singularity":
        if not image:
            raise ValueError("_submit_slurm(inner_backend='singularity') needs image=... (the .sif path)")
        inner_argv = _build_singularity_argv(job, image, executable=singularity_executable, bind_paths=bind_paths, use_gpu=use_gpu)
    else:
        raise ValueError(f"inner_backend must be 'local' or 'singularity' — got {inner_backend!r}")

    log_path = os.path.join(job.out_dir, "af3.log")
    sbatch_script_path = os.path.join(job.out_dir, "af3.sbatch.sh")
    slurm_job_id, sbatch_script_path = submit_via_slurm(
        inner_argv, out_dir=job.out_dir, log_path=log_path, sbatch_script_path=sbatch_script_path,
        job_name=job_name, partition=partition, account=account, time=time, gres=gres, gpus=gpus,
        cpus_per_task=cpus_per_task, mem=mem, setup_lines=setup_lines,
        extra_sbatch_directives=extra_sbatch_directives, sbatch_executable=sbatch_executable,
    )
    return AF3Run(
        job=job, process=None, command=inner_argv, log_path=log_path,
        slurm_job_id=slurm_job_id, sbatch_script_path=sbatch_script_path,
    )


def _find_model_cif(out_dir: str, candidate_id: str) -> Optional[str]:
    """AF3's own naming, confirmed against docs/output.md: the top-ranked
    structure lands at "{output_dir}/{name.lower()}/{name.lower()}_model.cif"
    — AF3 lowercases the job "name" for its output folder, so
    candidate_id is lowercased again here to match."""
    lowered = candidate_id.lower()
    path = os.path.join(out_dir, lowered, f"{lowered}_model.cif")
    return path if os.path.exists(path) else None


def poll_status(run: AF3Run) -> dict:
    """Cheap, non-blocking status check — same "count output files on
    disk" convention every module here uses."""
    candidate_ids = list(run.job.reference_map)
    folded_paths = {cid: p for cid in candidate_ids if (p := _find_model_cif(run.job.out_dir, cid))}

    if run.slurm_job_id is not None:
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


def cancel(run: AF3Run) -> None:
    """Terminates a still-running job — a no-op if it already finished."""
    terminate_or_cancel(run.process, run.slurm_job_id)


# ============================================================================
# The one-call convenience wrapper
# ============================================================================

def run(
    selected_df: pd.DataFrame, design_paths: Sequence[str], model_dir: str,
    terms_acknowledged: bool = False, backend: Optional[str] = None, config: Optional[dict] = None,
    poll_interval: float = 5.0, max_rmsd: float = 2.0, min_plddt: float = 70.0, **job_kwargs,
) -> pd.DataFrame:
    """prepare_self_consistency_job() + submit() + poll until done +
    collect_results() + select_validated_designs(), in one blocking
    call. Raises ValueError immediately if terms_acknowledged isn't set
    (see module docstring) — before writing any input files — and
    RuntimeError on a failed run (see log_path)."""
    tool_config = get_tool_config(config or {}, "af3")
    _require_terms_acknowledged(terms_acknowledged or bool(tool_config.get("terms_acknowledged", False)))

    job = prepare_self_consistency_job(selected_df, design_paths, model_dir=model_dir, **job_kwargs)
    run_handle = submit(job, backend=backend, config=config, terms_acknowledged=True)

    status = poll_status(run_handle)
    while status["state"] == "running":
        time.sleep(poll_interval)
        status = poll_status(run_handle)

    if status["state"] == "failed":
        raise RuntimeError(
            f"run_alphafold.py failed (exit {status['returncode']}) — see {status['log_path']} "
            f"for its own stdout/stderr."
        )

    results_df = collect_results(status["folded_paths"], job.reference_map)
    return select_validated_designs(results_df, max_rmsd=max_rmsd, min_plddt=min_plddt)
