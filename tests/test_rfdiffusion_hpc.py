"""
test_rfdiffusion_hpc.py — standalone smoke test for the SLURM backend of
toolkit/rfdiffusion.py, run directly on an HPC login/submit node (wherever
`sbatch`/`squeue`/`sacct` are actually on PATH).

Why standalone, not through `symbro`: rfdiffusion.py's submit()/poll_status()
have never been run against a real cluster (see the "HPC unproven" note in
this project's own commit history) and there is no `symbro rfdiffusion`
command yet either. This script exercises exactly the same code path
production use will (submit() -> poll_status(), reading the same
installation.yaml via config.load_installation_config()), just without any
CLI/pipeline.py wrapping around it -- so if something about the sbatch
script, the squeue/sacct polling, or a config field doesn't match THIS
cluster's conventions, that surfaces here directly instead of being buried
under a CLI layer.

Portable by construction: nothing about your specific cluster is hardcoded
below. Every site-specific value (partition, account, GPU flag, module
loads/conda activate lines, RFdiffusion's repo path and python env) comes
from installation.yaml -- so this same script works unmodified on anyone
else's HPC too, as long as their installation.yaml describes their own
site. This is deliberate: the toolkit itself has to work on any HPC once
it's public, not just this one.

Before running, on the HPC:
  1. `pip install -e .` this project there (or otherwise make `toolkit`
     importable) -- and make sure you have a ring PDB already, i.e.
     `symbro isolate` has run to completion and .symbro/rings.pkl exists.
     (If your query/download/geometry/isolate run happened on a DIFFERENT
     machine, just copy that machine's .symbro/ folder and the ring PDB(s)
     it points at over to the HPC -- nothing in this script needs the
     earlier stages to have run ON the HPC itself.)
  2. Fill in installation.yaml's `rfdiffusion:` section for THIS site --
     see installation.example.yaml for the full field-by-field reference.
     For inner_backend="local" (a conda/pip env with RFdiffusion installed
     directly, no container), you need at minimum:

         rfdiffusion:
           backend: slurm
           repo_path: /path/to/your/RFdiffusion          # has scripts/run_inference.py
           python_executable: /path/to/envs/rfdiffusion/bin/python
           slurm:
             partition: your_gpu_partition
             # account: your_allocation                   # omit if not needed
             time: "00:30:00"                              # short, this is a smoke test
             gres: "gpu:1"                                 # or use `gpus: 1` instead --
                                                             # whichever your cluster expects
             cpus_per_task: 4
             mem: "16G"
             setup_lines:
               - "module load cuda/12.1"                   # whatever YOUR cluster needs
               - "source activate rfdiffusion-env"

     installation.yaml is already gitignored -- your real values never get
     committed.

Usage (from the project root, i.e. next to installation.yaml):
    python test_rfdiffusion_hpc.py
    python test_rfdiffusion_hpc.py --num-designs 2 --diffuser-t 15
    python test_rfdiffusion_hpc.py --assembly-id 4V6B --poll-interval 30

--diffuser-t defaults LOW (15, vs. the real default of 50) on purpose for
this smoke test -- it makes the run fast and cheap, but the resulting
design(s) are NOT representative of real design quality. Once this proves
the plumbing works, drop --diffuser-t entirely (or pass 50) for anything
you actually intend to use.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd


def _load_ring_row(state_dir: str, assembly_id: str | None):
    """Reads .symbro/rings.pkl (the `symbro isolate` checkpoint) and picks
    one row to submit -- either the one matching --assembly-id, or the
    first row if not given."""
    import os

    rings_path = os.path.join(state_dir, "rings.pkl")
    if not os.path.exists(rings_path):
        sys.exit(
            f"No checkpoint at {rings_path!r}. Run `symbro isolate` (with "
            f"--file-format pdb, the default) first -- or point --state-dir "
            f"at wherever that ran, or copy its .symbro/ folder + the ring "
            f"PDB(s) it references over to this machine."
        )

    df = pd.read_pickle(rings_path)
    if df.empty:
        sys.exit(f"{rings_path!r} exists but is empty -- nothing to submit.")

    if assembly_id is not None:
        df = df[df["assembly_id"] == assembly_id]
        if df.empty:
            sys.exit(f"No row for assembly_id={assembly_id!r} in {rings_path!r}.")

    row = df.iloc[0]
    print(f"Using assembly_id={row['assembly_id']!r}, symmetry_type={row['symmetry_type']!r}, "
          f"chain_groups={tuple(row['chain_groups'])!r}")
    print(f"Ring PDB: {row['filepath']}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", default=".symbro", help="Where rings.pkl lives (default: .symbro).")
    parser.add_argument("--assembly-id", default=None, help="Which assembly's ring to submit (default: first row).")
    parser.add_argument("--linker-min", type=int, default=15, help="Diffused linker min length (residues).")
    parser.add_argument("--linker-max", type=int, default=25, help="Diffused linker max length (residues).")
    parser.add_argument("--num-designs", type=int, default=1, help="How many designs this job should produce.")
    parser.add_argument("--diffuser-t", type=int, default=15,
                         help="RFdiffusion's diffuser.T -- LOW by default for a fast smoke test "
                              "(real runs should use 50, RFdiffusion's own default).")
    parser.add_argument("--poll-interval", type=int, default=20, help="Seconds between status checks.")
    parser.add_argument("--timeout", type=int, default=1800,
                         help="Give up waiting after this many seconds (default 30 min) -- the job "
                              "itself keeps running on the cluster either way, this just stops the "
                              "script from blocking forever; check squeue/sacct by hand if it fires.")
    args = parser.parse_args()

    from toolkit import config, rfdiffusion

    row = _load_ring_row(args.state_dir, args.assembly_id)

    cfg = config.load_installation_config()
    tool_cfg = config.get_tool_config(cfg, "rfdiffusion")
    if not tool_cfg:
        sys.exit(
            "No [rfdiffusion] section found in installation.yaml -- see this script's own "
            "module docstring (top of the file) for what to fill in before running."
        )
    print(f"Loaded rfdiffusion config: backend={tool_cfg.get('backend')!r}, "
          f"repo_path={tool_cfg.get('repo_path')!r}")

    short_order = rfdiffusion.remap_chain_order(tuple(row["chain_groups"]), row["chain_rename_map"])
    print(f"Chain order (short ids): {short_order}")

    job = rfdiffusion.prepare_fusion_job(
        row["filepath"], short_order,
        linker_length=(args.linker_min, args.linker_max),
        num_designs=args.num_designs, diffuser_T=args.diffuser_t,
    )
    print(f"Built job: contigs={job.contigs!r}, output_prefix={job.output_prefix!r}")

    print("\nSubmitting to SLURM...")
    run = rfdiffusion.submit(job, backend="slurm", config=cfg)
    print(f"Submitted -- slurm_job_id={run.slurm_job_id}")
    print(f"  sbatch script: {run.sbatch_script_path}")
    print(f"  log file:      {run.log_path}")

    print(f"\nPolling every {args.poll_interval}s (timeout {args.timeout}s)...")
    start = time.time()
    while True:
        status = rfdiffusion.poll_status(run)
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>5}s] state={status['state']} "
              f"designs={status['designs_written']}/{status['designs_expected']}")

        if status["state"] in ("completed", "completed_partial", "failed"):
            break
        if elapsed > args.timeout:
            print(f"\nTimed out after {args.timeout}s waiting -- the job is still running on the "
                  f"cluster (job id {run.slurm_job_id}). Check `squeue -j {run.slurm_job_id}` or "
                  f"`sacct -j {run.slurm_job_id}` by hand, and {run.log_path} for its output.")
            sys.exit(2)

        time.sleep(args.poll_interval)

    print(f"\nFinal state: {status['state']} (returncode={status['returncode']})")
    print(f"Log: {status['log_path']}")
    if status["design_paths"]:
        print("Design(s) written:")
        for p in status["design_paths"]:
            print(f"  {p}")

    if status["state"] == "failed":
        print(f"\n--- last lines of {status['log_path']} ---")
        try:
            with open(status["log_path"]) as f:
                lines = f.readlines()
            print("".join(lines[-30:]))
        except OSError as exc:
            print(f"(couldn't read log: {exc})")
        sys.exit(1)

    print("\nSmoke test passed: submission, sbatch generation, and squeue/sacct polling all "
          "worked end to end on this cluster.")


if __name__ == "__main__":
    main()