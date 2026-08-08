"""
boltz.py — structural self-consistency screening for ProteinMPNN-
designed sequences via Boltz (jwohlwend/boltz): folds each candidate
back and checks whether the PREDICTED structure actually matches the
backbone it was designed for (low CA-RMSD, high pLDDT). Same
QUICKSTART/role as alphafold2.py — see that module's docstring for the
overall pattern, and selfconsistency.py for the RMSD/pLDDT/results
machinery this backend shares with alphafold2.py and af3.py.

    from toolkit import pmpnn, boltz

    shortlist = pmpnn.select_best_designs(sequences_df, top_n=3)
    winners = boltz.run(shortlist, design_paths, config=cfg)

LICENSING — confirmed directly against jwohlwend/boltz's own README, not
recalled from memory: "Our model and code are released under MIT
License, and can be freely used for both academic and commercial
purposes." Both code AND weights, no separate weights license to track
down — the cleanest of the three backends license-wise (compare
af3.py's docstring). Its optional hosted MSA search
(--use_msa_server) talks to the same MMseqs2-based service ColabFold
uses (also MIT — see alphafold2.py's docstring).

INPUT FORMAT — Boltz takes a directory of per-target YAML files (or a
single YAML/FASTA file; YAML is the current, non-deprecated format —
confirmed against boltz's own docs/prediction.md, which explicitly
marks FASTA input "deprecated"). One YAML file is written per candidate
here, e.g.:

    version: 1
    sequences:
      - protein:
          id: A
          sequence: MKT...

with no "msa" key — --use_msa_server (passed by default here) tells
Boltz to fetch one automatically, which its own docs confirm is a valid
way to omit that key entirely.

CONFIDENCE SCALE CAVEAT — read directly off the predicted structure's
own B-factor column (ca_coords() in selfconsistency.py), the same way
as every other backend here, rather than off Boltz's own confidence
JSON — see selfconsistency.py's module docstring for why, including the
honest caveat that this hasn't been directly confirmed against a real
Boltz output file (only inferred from Boltz's own confidence-formula
documentation). If mean_plddt values from this backend look
systematically off from alphafold2.py's/af3.py's (e.g. everything in
the 0-1 range instead of 0-100), that's the thing to check first —
select_validated_designs()'s default min_plddt=70.0 assumes the
AF2/AF3-shaped 0-100 scale.
"""

import glob
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
    "BoltzJob", "BoltzRun", "prepare_self_consistency_job", "build_command",
    "submit", "poll_status", "cancel", "collect_results", "select_validated_designs", "run",
]


# ============================================================================
# Job spec
# ============================================================================

@dataclass
class BoltzJob:
    """A fully-specified, ready-to-submit Boltz batch fold call."""

    input_dir: str
    out_dir: str
    reference_map: Dict[str, dict]
    use_msa_server: bool = True
    recycling_steps: Optional[int] = None
    diffusion_samples: Optional[int] = None
    sampling_steps: Optional[int] = None
    accelerator: Optional[str] = None
    devices: Optional[int] = None
    extra_flags: List[str] = field(default_factory=list)


def _write_candidate_yaml(path: str, sequence: str) -> None:
    # Hand-written, not a yaml.dump() call: the schema is fixed and tiny
    # (confirmed against boltz's own docs/prediction.md example), and
    # avoiding a PyYAML dependency purely for a 4-line, no-special-
    # characters file keeps this module's own dependency footprint
    # minimal. sequence is a plain amino-acid string (no quoting needed
    # in YAML's block scalar form).
    with open(path, "w") as f:
        f.write("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: " + sequence + "\n")


def prepare_self_consistency_job(
    selected_df: pd.DataFrame, design_paths: Sequence[str], out_dir: Optional[str] = None,
    use_msa_server: bool = True, recycling_steps: Optional[int] = None,
    diffusion_samples: Optional[int] = None, sampling_steps: Optional[int] = None,
    accelerator: Optional[str] = None, devices: Optional[int] = None,
    extra_flags: Optional[Sequence[str]] = None,
) -> BoltzJob:
    """
    Builds a submit()-ready BoltzJob straight from
    pmpnn.select_best_designs()'s own output shape (see
    selfconsistency.build_reference_map()). Writes one YAML file per
    candidate into "{out_dir}/inputs/" — the directory Boltz's own
    batched-prediction mode expects (see module docstring).

    out_dir defaults to a "self_consistency" folder next to the first
    reference design PDB — same convention every module here uses.

    accelerator/devices map straight to boltz predict's own
    --accelerator [gpu,cpu,tpu] / --devices flags. Left as None by
    default (i.e. not passed at all — boltz picks its own default,
    which auto-selects a GPU backend). Pass accelerator="cpu" explicitly
    on a machine with no CUDA-enabled torch install — boltz predict
    otherwise crashes with pytorch_lightning's own
    "No supported gpu backend found!" MisconfigurationException rather
    than falling back to CPU on its own.
    """
    reference_map = build_reference_map(selected_df, design_paths)

    if out_dir is None:
        first_ref = next(iter(reference_map.values()))["reference_pdb"]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(first_ref)) or ".", "self_consistency")
    input_dir = os.path.join(out_dir, "inputs")
    os.makedirs(input_dir, exist_ok=True)

    for candidate_id, entry in reference_map.items():
        _write_candidate_yaml(os.path.join(input_dir, f"{candidate_id}.yaml"), entry["sequence"])

    return BoltzJob(
        input_dir=input_dir, out_dir=out_dir, reference_map=reference_map,
        use_msa_server=use_msa_server, recycling_steps=recycling_steps,
        diffusion_samples=diffusion_samples, sampling_steps=sampling_steps,
        accelerator=accelerator, devices=devices,
        extra_flags=list(extra_flags or []),
    )


def build_command(job: BoltzJob, boltz_executable: str = "boltz") -> List[str]:
    """argv list for `boltz predict` — flag names (--out_dir,
    --use_msa_server, --recycling_steps, --diffusion_samples,
    --sampling_steps, --accelerator, --devices) confirmed against
    boltz's own docs/prediction.md."""
    argv = [boltz_executable, "predict", job.input_dir, "--out_dir", job.out_dir]
    if job.use_msa_server:
        argv.append("--use_msa_server")
    if job.recycling_steps is not None:
        argv += ["--recycling_steps", str(job.recycling_steps)]
    if job.diffusion_samples is not None:
        argv += ["--diffusion_samples", str(job.diffusion_samples)]
    if job.sampling_steps is not None:
        argv += ["--sampling_steps", str(job.sampling_steps)]
    if job.accelerator is not None:
        argv += ["--accelerator", job.accelerator]
    if job.devices is not None:
        argv += ["--devices", str(job.devices)]
    argv += list(job.extra_flags)
    return argv


# ============================================================================
# Non-blocking dispatch
# ============================================================================

@dataclass
class BoltzRun:
    """Handle to a dispatched (possibly still-running) Boltz job."""

    job: BoltzJob
    process: Optional[subprocess.Popen]
    command: List[str]
    log_path: str
    slurm_job_id: Optional[str] = None
    sbatch_script_path: Optional[str] = None


def submit(job: BoltzJob, backend: Optional[str] = None, config: Optional[dict] = None, **backend_kwargs) -> BoltzRun:
    """Dispatches `job` without blocking — same backend resolution order
    as every other module here: explicit backend= > config["boltz"]
    ["backend"] > "local"."""
    tool_config = get_tool_config(config or {}, "boltz")
    if backend is None:
        backend = tool_config.get("backend", "local")

    if backend == "local":
        defaults = {"boltz_executable": tool_config.get("boltz_executable", "boltz")}
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
                "backend='singularity' needs an 'image' (the .sif path) — pass image=... "
                "directly, or set boltz.singularity_image in your installation config."
            )
        return _submit_singularity(job, **merged)

    if backend == "slurm":
        slurm_config = dict(tool_config.get("slurm") or {})
        inner_backend = tool_config.get("inner_backend", "local")
        defaults = {
            "inner_backend": inner_backend,
            "boltz_executable": tool_config.get("boltz_executable", "boltz"),
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
            "job_name": slurm_config.get("job_name", "boltz"),
            "setup_lines": slurm_config.get("setup_lines", []),
            "extra_sbatch_directives": slurm_config.get("extra_sbatch_directives", []),
            "sbatch_executable": slurm_config.get("sbatch_executable", "sbatch"),
        }
        merged = {**defaults, **backend_kwargs}
        if merged["inner_backend"] == "singularity" and not merged.get("image"):
            raise ValueError(
                "backend='slurm' with inner_backend='singularity' needs an 'image' — pass "
                "image=... directly, or set boltz.singularity_image in your installation config."
            )
        return _submit_slurm(job, **merged)

    raise NotImplementedError(
        f"backend {backend!r} is not implemented yet — 'local', 'singularity', and 'slurm' are "
        f"available today."
    )


def _submit_local(job: BoltzJob, boltz_executable: str = "boltz") -> BoltzRun:
    argv = build_command(job, boltz_executable=boltz_executable)
    os.makedirs(job.out_dir, exist_ok=True)
    log_path = os.path.join(job.out_dir, "boltz.log")
    with open(log_path, "w") as log_file:
        # Log the actual resolved command line FIRST, before boltz's own
        # output — otherwise there's no on-disk way to confirm whether a
        # flag like --accelerator actually made it onto the command
        # boltz was invoked with, short of re-deriving it from the
        # BoltzJob object by hand.
        log_file.write("$ " + subprocess.list2cmdline(argv) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)
    return BoltzRun(job=job, process=process, command=argv, log_path=log_path)


def _build_singularity_argv(
    job: BoltzJob, image: str, executable: str = "singularity",
    bind_paths: Sequence[str] = (), use_gpu: bool = True,
) -> List[str]:
    """Same "singularity exec --nv -B host:container <image> <command>"
    shape as alphafold2.py's own _build_singularity_argv() — Boltz's
    published container, like ColabFold's, has no entrypoint that
    already knows to invoke `boltz predict`, so it's named explicitly."""
    input_dir_abs = os.path.abspath(job.input_dir)
    out_dir_abs = os.path.abspath(job.out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    mount_dirs = sorted({input_dir_abs, out_dir_abs})

    argv = [executable, "exec"]
    if use_gpu:
        argv.append("--nv")
    for d in mount_dirs:
        argv += ["-B", f"{d}:{d}"]
    for extra in bind_paths:
        argv += ["-B", extra]
    argv.append(image)
    argv += build_command(job, boltz_executable="boltz")
    return argv


def _submit_singularity(
    job: BoltzJob, image: str, executable: str = "singularity",
    bind_paths: Sequence[str] = (), use_gpu: bool = True,
) -> BoltzRun:
    argv = _build_singularity_argv(job, image, executable=executable, bind_paths=bind_paths, use_gpu=use_gpu)
    os.makedirs(job.out_dir, exist_ok=True)
    log_path = os.path.join(job.out_dir, "boltz.log")
    with open(log_path, "w") as log_file:
        log_file.write("$ " + subprocess.list2cmdline(argv) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)
    return BoltzRun(job=job, process=process, command=argv, log_path=log_path)


def _submit_slurm(
    job: BoltzJob, inner_backend: str = "local", boltz_executable: str = "boltz",
    image: Optional[str] = None, singularity_executable: str = "singularity",
    bind_paths: Sequence[str] = (), use_gpu: bool = True,
    partition: Optional[str] = None, account: Optional[str] = None, time: str = "04:00:00",
    gres: Optional[str] = None, gpus: Optional[int] = None, cpus_per_task: int = 4, mem: str = "16G",
    job_name: str = "boltz", setup_lines: Sequence[str] = (),
    extra_sbatch_directives: Sequence[str] = (), sbatch_executable: str = "sbatch",
) -> BoltzRun:
    os.makedirs(job.out_dir, exist_ok=True)

    if inner_backend == "local":
        inner_argv = build_command(job, boltz_executable=boltz_executable)
    elif inner_backend == "singularity":
        if not image:
            raise ValueError("_submit_slurm(inner_backend='singularity') needs image=... (the .sif path)")
        inner_argv = _build_singularity_argv(job, image, executable=singularity_executable, bind_paths=bind_paths, use_gpu=use_gpu)
    else:
        raise ValueError(f"inner_backend must be 'local' or 'singularity' — got {inner_backend!r}")

    log_path = os.path.join(job.out_dir, "boltz.log")
    sbatch_script_path = os.path.join(job.out_dir, "boltz.sbatch.sh")
    slurm_job_id, sbatch_script_path = submit_via_slurm(
        inner_argv, out_dir=job.out_dir, log_path=log_path, sbatch_script_path=sbatch_script_path,
        job_name=job_name, partition=partition, account=account, time=time, gres=gres, gpus=gpus,
        cpus_per_task=cpus_per_task, mem=mem, setup_lines=setup_lines,
        extra_sbatch_directives=extra_sbatch_directives, sbatch_executable=sbatch_executable,
    )
    return BoltzRun(
        job=job, process=None, command=inner_argv, log_path=log_path,
        slurm_job_id=slurm_job_id, sbatch_script_path=sbatch_script_path,
    )


def _find_model_cif(out_dir: str, candidate_id: str) -> Optional[str]:
    """
    Boltz's own naming, confirmed against its docs/prediction.md:
    "{out_dir}/predictions/{input_stem}/{input_stem}_model_0.cif" —
    "_model_0" is the default (highest-confidence, when
    diffusion_samples=1) sample; a run with diffusion_samples > 1 would
    also write "_model_1.cif" etc., but "_model_0" is confirmed to
    always exist and be Boltz's own default/first-ranked pick, so it's
    what's read here regardless of diffusion_samples.
    """
    path = os.path.join(out_dir, "predictions", candidate_id, f"{candidate_id}_model_0.cif")
    return path if os.path.exists(path) else None


def poll_status(run: BoltzRun) -> dict:
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


def cancel(run: BoltzRun) -> None:
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
            f"boltz predict failed (exit {status['returncode']}) — see {status['log_path']} "
            f"for its own stdout/stderr."
        )

    results_df = collect_results(status["folded_paths"], job.reference_map)
    return select_validated_designs(results_df, max_rmsd=max_rmsd, min_plddt=min_plddt)
