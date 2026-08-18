# 03 — SLURM `--detach` / `symbro status` workflow

**What this is:** a documented pattern, built directly from `cli.py`'s and `installation.example.yaml`'s actual field definitions — not something re-run and verified here, since that needs a real SLURM cluster with RFdiffusion installed, which this review didn't have access to. Treat the commands as accurate to the code; treat "does the whole thing work end to end on a live cluster" as still open, same as the main README's own **Project status** section already says about the compute-heavy stages generally.

## Why you'd want this

By default, `symbro rfdiffusion` blocks your terminal until every submitted job finishes — fine for a quick local/Singularity run, less fine for a SLURM submission that might sit in a queue for a while. `--detach` submits and returns immediately instead; you check back later with `symbro status`. `--detach` only works with `backend="slurm"` — `cli.py` doesn't reject the combination client-side, but a non-SLURM backend has nothing to detach from.

## 1. Configure the SLURM backend once

```bash
cp installation.example.yaml installation.yaml
```

Edit the `rfdiffusion:` section's `slurm:` block — every field is site-specific, ask your cluster's docs/admins for the actual values:

```yaml
rfdiffusion:
  backend: slurm
  repo_path: /path/to/RFdiffusion
  python_executable: /path/to/envs/rfdiffusion/bin/python
  inner_backend: singularity   # or "local" -- see the comment above this
                                # key in installation.example.yaml for which
                                # one your cluster actually needs
  # singularity_image: /path/to/rfdiffusion.sif   # if inner_backend: singularity

  slurm:
    partition: gpu
    time: "04:00:00"
    gres: "gpu:1"               # or gpus: 1 -- whichever your cluster expects
    cpus_per_task: 4
    mem: "16G"
    job_name: rfdiffusion
    setup_lines:
      - "module load cuda/12.1"
      - "source activate rfdiffusion-env"
```

## 2. Submit and detach

```bash
symbro rfdiffusion --detach
```

Returns immediately once the sbatch job(s) are submitted, rather than waiting for them to finish. The checkpoint (`.symbro/rfdiffusion.pkl`/`.csv`) is written right away too, with each row's SLURM job ID and a `state` that isn't `completed` yet.

## 3. Check back later

```bash
symbro status
```

Refreshes every tracked job's state and reprints a summary — `state` lands on `completed`, `completed_partial`, or `failed` per job once SLURM finishes it; anything else still counts as running. Safe to run as many times as you want, e.g. from a cron job or just periodically by hand.

```
✓ N RFdiffusion job(s) tracked, M still running.
...
  All jobs done. Next: symbro pmpnn
```

(That "All jobs done" line only prints once every tracked job has actually left the running state.)

## 4. Continue once everything's done

```bash
symbro pmpnn
symbro predict
```

Neither of these has its own `--detach` — `pmpnn` is always a local process that doesn't survive the command exiting (see `symbro pmpnn --help`), and `predict`'s three backends each have their own backend setting but no separate detach flag today.

## A minimal `launch.sh`

If you're kicking this off from a login node and want one script to point a `cron`/scheduler at, or just to avoid retyping the path each time (the same idea as prosculpt's own `Minimal_Prosculpt_scripts/launch.sh`):

```bash
./submit.sh /path/to/your/project
```

— see `submit.sh` in this folder. It only wraps step 2 above (submit + detach); run `symbro status`/`pmpnn`/`predict` yourself once you're ready to check on it, the same as steps 3–4.
