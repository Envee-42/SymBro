"""
test_pmpnn_hpc.py -- standalone smoke test for toolkit/pmpnn.py, run
against RFdiffusion designs that have already completed (see
test_rfdiffusion_hpc.py). Same philosophy as that script: exercise
exactly the production code path directly (prepare_mpnn_job() ->
submit() -> poll_status() -> collect_sequences()), reading the same
installation.yaml via config.load_installation_config(), with no
CLI/pipeline.py wrapping -- there's no `symbro pmpnn` command yet either.

IMPORTANT -- unlike RFdiffusion, ProteinMPNN has NO SLURM backend
(pmpnn.py's own module docstring: "LOCAL ONLY, on purpose"). submit()
always runs it as a local subprocess wherever THIS SCRIPT is running --
there is no sbatch job to survive a dropped connection the way
RFdiffusion's did. Run this interactively on a node your site allows
this on (ProteinMPNN is far lighter than RFdiffusion, so the login node
may be fine -- check your cluster's own usage policy first) and keep the
session open, or wrap the invocation in your own nohup/tmux/screen if
you need it to survive a disconnect; this script does not do that for
you.

WHICH RFdiffusion design(s) get submitted, and why scoring matters here:
RFdiffusion's own SLURM job may have produced more than one design, and
running ProteinMPNN (and everything downstream of it) against a design
RFdiffusion itself was unconfident about wastes real compute for no
benefit. So this script:
  1. Re-derives the SAME RFdiffusionJob the original RFdiffusion run
     used (same ring PDB + chain order + linker length -- a pure,
     deterministic rebuild that does NOT re-run RFdiffusion itself),
     specifically so fixed_positions_from_contig_match() can correctly
     tell ProteinMPNN which residues are RFdiffusion's native, kept-
     fixed motif vs. its newly-diffused linkers.
  2. Globs that job's own design_paths and scores every one via
     rfdiffusion.rank_designs() -- RFdiffusion's own final-timestep mean
     pLDDT (see that function's docstring for exactly what it measures,
     and why "motif RMSD" is deliberately NOT used: RFdiffusion never
     saves that value anywhere, only logs it to stdout).
  3. Prints the FULL ranked table before submitting anything -- that
     table IS the "let the user select by scoring" feature.
     --top-n/--min-plddt/--select below are three different ways to act
     on what the table already shows, not a replacement for showing it.

Before running, on the HPC:
  1. `pip install -e .` this project there, and have already run
     test_rfdiffusion_hpc.py to completion for the SAME --assembly-id /
     --linker-min / --linker-max you pass here (defaults match that
     script's own defaults) -- this script never runs RFdiffusion
     itself, only re-derives where its designs already are.
  2. Fill in installation.yaml's `proteinmpnn:` section -- see
     installation.example.yaml for the full reference. At minimum:

        proteinmpnn:
          repo_path: /path/to/your/ProteinMPNN
          python_executable: /path/to/envs/proteinmpnn/bin/python

     installation.yaml is already gitignored -- your real values never
     get committed.

Usage (from the project root, i.e. next to installation.yaml):
    python test_pmpnn_hpc.py
    python test_pmpnn_hpc.py --assembly-id 4V6B --top-n 2
    python test_pmpnn_hpc.py --min-plddt 0.9 --num-seq-per-target 16
    python test_pmpnn_hpc.py --select temporary_subunits/rfdiffusion_designs/4V6B-1_C3_AH-AP-AU_0.pdb
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import pandas as pd


def _load_ring_row(state_dir: str, assembly_id: str | None):
    """Reads .symbro/rings.pkl (the `symbro isolate` checkpoint) and picks
    one row to work from -- either the one matching --assembly-id, or the
    first row if not given. Identical to test_rfdiffusion_hpc.py's own
    helper of the same name -- duplicated rather than shared, matching
    this project's "each smoke-test script is fully standalone" convention."""
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
        sys.exit(f"{rings_path!r} exists but is empty -- nothing to work from.")

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
    parser.add_argument("--assembly-id", default=None, help="Which assembly's ring to work from (default: first row).")
    parser.add_argument("--linker-min", type=int, default=15,
                         help="Must match the RFdiffusion run's own --linker-min -- needed to rebuild "
                              "the identical job spec (contig/input_pdb) its designs were made from.")
    parser.add_argument("--linker-max", type=int, default=25, help="Must match the RFdiffusion run's own --linker-max.")
    parser.add_argument("--top-n", type=int, default=1,
                         help="Submit only the top N designs by RFdiffusion's own pLDDT score "
                              "(default: 1, the single best-scored design). The full ranked table is "
                              "always printed regardless, so you can see what was passed over.")
    parser.add_argument("--min-plddt", type=float, default=None,
                         help="Also require at least this mean pLDDT -- combined with --top-n if both are given.")
    parser.add_argument("--select", action="append", default=None,
                         help="Explicit design PDB path to submit (repeatable) -- bypasses --top-n/"
                              "--min-plddt entirely and submits exactly the path(s) given. Relative or "
                              "absolute; relative paths are resolved against your current directory.")
    parser.add_argument("--num-seq-per-target", type=int, default=8, help="ProteinMPNN sequences per design.")
    parser.add_argument("--sampling-temp", type=float, default=0.1, help="ProteinMPNN sampling temperature.")
    parser.add_argument("--batch-size", type=int, default=8, help="Must evenly divide --num-seq-per-target.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between status checks.")
    parser.add_argument("--timeout", type=int, default=900,
                         help="Give up waiting after this many seconds (default 15 min -- ProteinMPNN is "
                              "much faster than RFdiffusion's GPU job). Unlike RFdiffusion's SLURM path, "
                              "this IS a local process: if this script dies, so does ProteinMPNN.")
    args = parser.parse_args()

    from toolkit import config, pmpnn, rfdiffusion

    row = _load_ring_row(args.state_dir, args.assembly_id)
    short_order = rfdiffusion.remap_chain_order(tuple(row["chain_groups"]), row["chain_rename_map"])

    # Pure, deterministic rebuild -- does NOT re-run RFdiffusion, just
    # reconstructs the same job spec (contigs/input_pdb/output_prefix) so
    # rank_designs() can find the same designs and
    # fixed_positions_from_contig_match() can correctly interpret them.
    job = rfdiffusion.prepare_fusion_job(
        row["filepath"], short_order, linker_length=(args.linker_min, args.linker_max),
    )
    print(f"Rebuilt RFdiffusion job spec (output_prefix={job.output_prefix!r}) -- not re-running RFdiffusion.")

    # Same filter poll_status() uses, and for the same reason: a bare
    # "{output_prefix}_*.pdb" glob also matches "{output_prefix}_fused_input.pdb"
    # (prepare_fusion_job()'s own relabeled-input file, written right next to the
    # real designs) -- the "_<digits>.pdb" suffix is what actually distinguishes
    # a real RFdiffusion design from that. Caught by testing this against a real
    # job.output_prefix rather than assuming the naive glob was safe.
    _design_pattern = re.compile(re.escape(job.output_prefix) + r"_\d+\.pdb$")
    design_paths = sorted(p for p in glob.glob(f"{job.output_prefix}_*.pdb") if _design_pattern.search(p))
    if not design_paths:
        sys.exit(
            f"No RFdiffusion designs found at {job.output_prefix!r}_*.pdb -- run "
            f"test_rfdiffusion_hpc.py first (with matching --assembly-id/--linker-min/--linker-max), "
            f"or check --state-dir/--assembly-id here match that run."
        )

    ranked = rfdiffusion.rank_designs(design_paths)
    print(f"\n{len(ranked)} RFdiffusion design(s) found, scored by final-timestep mean pLDDT:")
    print(ranked[["design_path", "mean_plddt", "min_plddt"]].to_string(index=False))

    if args.select:
        # ranked["design_path"] is always absolute (job.output_prefix is built from
        # os.path.abspath(ring_pdb_path) inside prepare_fusion_job()) -- resolve each
        # --select value the same way so a relative path (as this script's own
        # docstring example gives) actually matches instead of silently failing.
        selected_abs = [os.path.abspath(p) for p in args.select]
        missing = [
            given for given, given_abs in zip(args.select, selected_abs)
            if given_abs not in ranked["design_path"].values
        ]
        if missing:
            sys.exit(f"--select path(s) not among this job's own designs: {missing}")
        selected_paths = selected_abs
    else:
        selected = rfdiffusion.rank_designs(design_paths, top_n=args.top_n, min_plddt=args.min_plddt)
        selected_paths = selected["design_path"].tolist()

    if not selected_paths:
        sys.exit("No designs left after --top-n/--min-plddt filtering -- loosen the threshold and try again.")
    print(f"\nSelected {len(selected_paths)} design(s) for ProteinMPNN: "
          f"{[os.path.basename(p) for p in selected_paths]}")

    cfg = config.load_installation_config()
    tool_cfg = config.get_tool_config(cfg, "proteinmpnn")
    if not tool_cfg:
        sys.exit(
            "No [proteinmpnn] section found in installation.yaml -- see this script's own "
            "module docstring (top of the file) for what to fill in before running."
        )
    print(f"Loaded proteinmpnn config: repo_path={tool_cfg.get('repo_path')!r}")

    mpnn_job = pmpnn.prepare_mpnn_job(
        selected_paths, rf_job=job,
        num_seq_per_target=args.num_seq_per_target, sampling_temp=args.sampling_temp,
        batch_size=args.batch_size,
    )
    n_fixed = len(mpnn_job.fixed_positions or {})
    print(f"Built ProteinMPNN job: out_folder={mpnn_job.out_folder!r}, "
          f"fixed-position data found for {n_fixed}/{len(selected_paths)} design(s)")
    if n_fixed < len(selected_paths):
        print(
            "  Warning: at least one selected design has no detected fixed positions -- "
            "ProteinMPNN would treat it as fully designable, including RFdiffusion's native "
            "motif residues. Check the warning(s) above from fixed_positions_from_contig_match()."
        )

    print("\nSubmitting to ProteinMPNN (runs locally on this machine, not via sbatch)...")
    run = pmpnn.submit(mpnn_job, config=cfg)
    print(f"  log file: {run.log_path}")

    print(f"\nPolling every {args.poll_interval}s (timeout {args.timeout}s)...")
    start = time.time()
    while True:
        status = pmpnn.poll_status(run)
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>5}s] state={status['state']} "
              f"sequences={status['sequences_written']}/{status['sequences_expected']}")

        if status["state"] in ("completed", "completed_partial", "failed"):
            break
        if elapsed > args.timeout:
            print(f"\nTimed out after {args.timeout}s waiting -- check {run.log_path} by hand. Unlike "
                  f"RFdiffusion's SLURM job, this is a local process: it does NOT keep running if this "
                  f"script's own process has been killed, so a timeout here may mean the job is gone, "
                  f"not just slow.")
            sys.exit(2)

        time.sleep(args.poll_interval)

    print(f"\nFinal state: {status['state']} (returncode={status['returncode']})")
    if status["state"] == "failed":
        print(f"\n--- last lines of {status['log_path']} ---")
        try:
            with open(status["log_path"]) as f:
                lines = f.readlines()
            print("".join(lines[-30:]))
        except OSError as exc:
            print(f"(couldn't read log: {exc})")
        sys.exit(1)

    sequences_df = pmpnn.collect_sequences(status)
    out_csv = os.path.join(mpnn_job.out_folder, "sequences.csv")
    sequences_df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(sequences_df)} sequence row(s) (including native readbacks) to {out_csv}")

    shortlist = pmpnn.select_best_designs(sequences_df, top_n=3)
    if not shortlist.empty:
        print(f"\nTop candidate(s) by ProteinMPNN's own global_score (lower = more plausible):")
        print(shortlist[["source_pdb", "global_score", "rank"]].to_string(index=False))

    print("\nSmoke test passed: RFdiffusion design scoring/selection, fixed-position detection, "
          "ProteinMPNN submission, and sequence parsing all worked end to end.")


if __name__ == "__main__":
    main()
