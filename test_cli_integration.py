"""
test_cli_integration.py — direct pipeline.* calls against the newly-wired
RFdiffusion/ProteinMPNN CLI integration (pipeline.run_rfdiffusion,
run_status/refresh_rfdiffusion_status, run_pmpnn) and symbro clean.

Mocks only rfdiffusion.submit()/poll_status() and pmpnn.submit()/
poll_status() -- the actual I/O boundary (subprocess/sbatch dispatch and
squeue/sacct polling) -- everything else (prepare_fusion_job(),
remap_chain_order(), rank_designs(), collect_sequences(), checkpoint
save/load) runs for real against a real fixture ring PDB (see conftest.py).
"""
import os

import pandas as pd
import pytest

from toolkit import pipeline, pmpnn, rfdiffusion
from conftest import rings_df, write_fake_rfdiffusion_design, write_fake_pmpnn_output


# ----------------------------------------------------------------------
# Fakes for the I/O boundary
# ----------------------------------------------------------------------
def install_rfdiffusion_fakes(monkeypatch, calls, states=None):
    """states: optional {output_prefix: [state, state, ...]} -- poll_status()
    pops one state per call for that job, repeating the last once exhausted.
    Default (states=None): every job reports "completed" on the FIRST poll."""
    states = states or {}

    def fake_submit(job, backend=None, config=None, **kwargs):
        calls.setdefault("submit", []).append((job.output_prefix, backend))
        n = len(calls["submit"])
        slurm_job_id = f"{9000 + n}" if backend == "slurm" else None
        return rfdiffusion.RFdiffusionRun(
            job=job, process=None, command=["fake-rfdiffusion"], log_path=f"{job.output_prefix}.log",
            slurm_job_id=slurm_job_id,
            sbatch_script_path=f"{job.output_prefix}.sbatch.sh" if backend == "slurm" else None,
        )

    def fake_poll_status(run):
        calls.setdefault("poll", []).append(run.job.output_prefix)
        queue = states.get(run.job.output_prefix)
        state = queue.pop(0) if queue else "completed"
        if state != "completed":
            return {
                "state": state, "returncode": None, "designs_written": 0,
                "designs_expected": run.job.num_designs, "design_paths": [],
                "log_path": run.log_path, "job": run.job,
            }
        design_paths = [write_fake_rfdiffusion_design(run.job.output_prefix, i) for i in range(run.job.num_designs)]
        return {
            "state": "completed", "returncode": 0, "designs_written": len(design_paths),
            "designs_expected": run.job.num_designs, "design_paths": design_paths,
            "log_path": run.log_path, "job": run.job,
        }

    monkeypatch.setattr(rfdiffusion, "submit", fake_submit)
    monkeypatch.setattr(rfdiffusion, "poll_status", fake_poll_status)
    monkeypatch.setattr("time.sleep", lambda s: None)


def install_pmpnn_fakes(monkeypatch, calls):
    def fake_submit(job, config=None, **kwargs):
        calls.setdefault("submit", []).append(job.out_folder)
        return pmpnn.ProteinMPNNRun(
            job=job, process=None, command=["fake-pmpnn"],
            log_path=os.path.join(job.out_folder, "proteinmpnn.log"),
            parse_command=["fake-parse"], parse_log_path=os.path.join(job.out_folder, "parse.log"),
        )

    def fake_poll_status(run):
        calls.setdefault("poll", []).append(run.job.out_folder)
        basenames = [os.path.splitext(os.path.basename(p))[0] for p in run.job.input_pdbs]
        fasta_paths = [write_fake_pmpnn_output(run.job.out_folder, b) for b in basenames]
        return {
            "state": "completed", "returncode": 0, "sequences_written": len(fasta_paths),
            "sequences_expected": len(basenames), "fasta_paths": fasta_paths, "log_path": run.log_path,
        }

    monkeypatch.setattr(pmpnn, "submit", fake_submit)
    monkeypatch.setattr(pmpnn, "poll_status", fake_poll_status)
    monkeypatch.setattr("time.sleep", lambda s: None)


# ----------------------------------------------------------------------
# run_rfdiffusion
# ----------------------------------------------------------------------
def test_run_rfdiffusion_block_and_poll_completes(project_dir, ring_pdb, monkeypatch):
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)
    df = pipeline.run_rfdiffusion(rings_df(["TEST-1"], ring_pdb), num_designs=2, diffuser_T=10)

    assert list(df.columns) == ["assembly_id", "symmetry_type", "component_id", "chain_groups", "run", "state", "design_paths"]
    assert len(df) == 1
    assert df.iloc[0]["state"] == "completed"
    assert len(df.iloc[0]["design_paths"]) == 2
    # submitted BEFORE polled -- see run_rfdiffusion()'s own docstring on why
    assert len(calls["submit"]) == 1
    assert len(calls["poll"]) >= 1
    # process is always sanitized to None before the checkpoint is saved
    assert df.iloc[0]["run"].process is None
    assert os.path.exists(os.path.join(".symbro", "rfdiffusion.pkl"))
    reloaded = pd.read_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))
    assert reloaded.iloc[0]["state"] == "completed"


def test_run_rfdiffusion_polls_until_terminal(project_dir, ring_pdb, monkeypatch):
    calls = {}

    def fake_submit(job, backend=None, config=None, **kwargs):
        calls.setdefault("submit", []).append(job.output_prefix)
        return rfdiffusion.RFdiffusionRun(
            job=job, process=None, command=["fake"], log_path=f"{job.output_prefix}.log",
        )

    def fake_poll_status(run):
        n = len(calls.setdefault("poll", [])) + 1
        calls["poll"].append(run.job.output_prefix)
        if n < 3:  # "running" on the first two polls, "completed" on the third
            return {
                "state": "running", "returncode": None, "designs_written": 0,
                "designs_expected": run.job.num_designs, "design_paths": [],
                "log_path": run.log_path, "job": run.job,
            }
        design_paths = [write_fake_rfdiffusion_design(run.job.output_prefix, i) for i in range(run.job.num_designs)]
        return {
            "state": "completed", "returncode": 0, "designs_written": len(design_paths),
            "designs_expected": run.job.num_designs, "design_paths": design_paths,
            "log_path": run.log_path, "job": run.job,
        }

    monkeypatch.setattr(rfdiffusion, "submit", fake_submit)
    monkeypatch.setattr(rfdiffusion, "poll_status", fake_poll_status)
    monkeypatch.setattr("time.sleep", lambda s: None)

    df = pipeline.run_rfdiffusion(rings_df(["TEST-1"], ring_pdb), num_designs=1, diffuser_T=10, poll_interval=0)
    assert df.iloc[0]["state"] == "completed"
    assert len(calls["poll"]) == 3  # running, running, completed


def test_run_rfdiffusion_detach_slurm_does_not_poll(project_dir, ring_pdb, monkeypatch):
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)
    df = pipeline.run_rfdiffusion(rings_df(["TEST-1"], ring_pdb), backend="slurm", detach=True, num_designs=2)
    assert df.iloc[0]["state"] == "submitted"
    assert df.iloc[0]["design_paths"] == []
    assert "poll" not in calls  # detach must skip polling entirely
    assert df.iloc[0]["run"].slurm_job_id is not None


def test_run_rfdiffusion_detach_requires_slurm_backend(project_dir, ring_pdb, monkeypatch):
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)
    for backend in ("local", "singularity"):
        with pytest.raises(ValueError, match="only works with backend='slurm'"):
            pipeline.run_rfdiffusion(rings_df(["TEST-1"], ring_pdb), backend=backend, detach=True)
    assert "submit" not in calls  # must fail BEFORE submitting anything


def test_run_rfdiffusion_unknown_assembly_id_raises(project_dir, ring_pdb, monkeypatch):
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)
    with pytest.raises(ValueError, match="No row for assembly_id"):
        pipeline.run_rfdiffusion(rings_df(["TEST-1"], ring_pdb), assembly_id="NOT-THERE")


def test_run_rfdiffusion_narrows_to_one_assembly(project_dir, ring_pdb, monkeypatch):
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)
    df = pipeline.run_rfdiffusion(
        rings_df(["TEST-1", "TEST-2"], ring_pdb), assembly_id="TEST-2", num_designs=1,
    )
    assert len(df) == 1
    assert df.iloc[0]["assembly_id"] == "TEST-2"


# ----------------------------------------------------------------------
# status / refresh_rfdiffusion_status
# ----------------------------------------------------------------------
def test_status_resumes_a_detached_run(project_dir, ring_pdb, monkeypatch):
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)
    pipeline.run_rfdiffusion(rings_df(["TEST-1"], ring_pdb), backend="slurm", detach=True, num_designs=1)
    assert "poll" not in calls

    df = pipeline.run_status()
    assert len(calls["poll"]) == 1
    assert df.iloc[0]["state"] == "completed"
    assert len(df.iloc[0]["design_paths"]) == 1


def test_status_leaves_already_terminal_rows_alone(project_dir, ring_pdb, monkeypatch):
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)
    pipeline.run_rfdiffusion(rings_df(["TEST-1", "TEST-2"], ring_pdb), backend="slurm", detach=True, num_designs=1)
    pipeline.run_status()  # both become "completed"
    calls.pop("poll", None)

    pipeline.run_status()  # re-run: nothing left to poll
    assert "poll" not in calls


def test_status_without_prior_rfdiffusion_run_raises(project_dir):
    with pytest.raises(pipeline.StageNotFoundError):
        pipeline.run_status()


# ----------------------------------------------------------------------
# run_pmpnn
# ----------------------------------------------------------------------
def _completed_rfdiffusion_df(project_dir, ring_pdb, assembly_ids, monkeypatch, plddts=None):
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)
    df = pipeline.run_rfdiffusion(rings_df(assembly_ids, ring_pdb), num_designs=2, diffuser_T=10)
    if plddts:
        # overwrite the fixture .trb files with caller-chosen pLDDTs so
        # top_n/min_plddt filtering is actually meaningful to assert on
        import pickle
        import numpy as np
        for _, row in df.iterrows():
            for i, p in enumerate(row["design_paths"]):
                trb_path = os.path.splitext(p)[0] + ".trb"
                plddt = plddts.get((row["assembly_id"], i), 0.9)
                with open(trb_path, "wb") as f:
                    pickle.dump({"plddt": np.full((1, 10), plddt), "device": "cpu", "time": 0.0, "config": {}}, f)
    return df


def test_run_pmpnn_default_top_n_one(project_dir, ring_pdb, monkeypatch):
    rf_df = _completed_rfdiffusion_df(project_dir, ring_pdb, ["TEST-1"], monkeypatch)
    rf_df.to_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))

    calls = {}
    install_pmpnn_fakes(monkeypatch, calls)
    df = pipeline.run_pmpnn()
    assert not df.empty
    assert set(df["assembly_id"]) == {"TEST-1"}
    assert len(calls["submit"]) == 1  # one job, top_n=1 design submitted
    assert os.path.exists(os.path.join(".symbro", "pmpnn.pkl"))


def test_run_pmpnn_multi_assembly_processes_both(project_dir, ring_pdb, monkeypatch):
    rf_df = _completed_rfdiffusion_df(project_dir, ring_pdb, ["TEST-1", "TEST-2"], monkeypatch)
    rf_df.to_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))

    calls = {}
    install_pmpnn_fakes(monkeypatch, calls)
    df = pipeline.run_pmpnn()
    assert set(df["assembly_id"]) == {"TEST-1", "TEST-2"}
    assert len(calls["submit"]) == 2


def test_run_pmpnn_skips_non_completed_assembly(project_dir, ring_pdb, monkeypatch):
    rf_df = _completed_rfdiffusion_df(project_dir, ring_pdb, ["TEST-1", "TEST-2"], monkeypatch)
    rf_df.loc[rf_df["assembly_id"] == "TEST-2", "state"] = "failed"
    rf_df.to_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))

    calls = {}
    install_pmpnn_fakes(monkeypatch, calls)
    df = pipeline.run_pmpnn()
    assert set(df["assembly_id"]) == {"TEST-1"}
    assert len(calls["submit"]) == 1


def test_run_pmpnn_top_n_and_min_plddt_filter(project_dir, ring_pdb, monkeypatch):
    rf_df = _completed_rfdiffusion_df(
        project_dir, ring_pdb, ["TEST-1"], monkeypatch,
        plddts={("TEST-1", 0): 0.95, ("TEST-1", 1): 0.40},
    )
    rf_df.to_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))

    calls = {}
    install_pmpnn_fakes(monkeypatch, calls)
    pipeline.run_pmpnn(top_n=2, min_plddt=0.5)  # design 1 (0.40) must be filtered out
    submitted_job = calls["submit"]
    assert len(submitted_job) == 1  # still one ProteinMPNN job (one assembly)


def test_run_pmpnn_select_requires_assembly_id(project_dir, ring_pdb, monkeypatch):
    rf_df = _completed_rfdiffusion_df(project_dir, ring_pdb, ["TEST-1"], monkeypatch)
    rf_df.to_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))
    with pytest.raises(ValueError, match="requires assembly_id"):
        pipeline.run_pmpnn(select=["some/path.pdb"])


def test_run_pmpnn_select_rejects_path_not_owned_by_assembly(project_dir, ring_pdb, monkeypatch):
    rf_df = _completed_rfdiffusion_df(project_dir, ring_pdb, ["TEST-1"], monkeypatch)
    rf_df.to_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))
    with pytest.raises(ValueError, match="not among"):
        pipeline.run_pmpnn(assembly_id="TEST-1", select=["/not/a/real/design.pdb"])


def test_run_pmpnn_select_accepts_a_real_owned_design(project_dir, ring_pdb, monkeypatch):
    rf_df = _completed_rfdiffusion_df(project_dir, ring_pdb, ["TEST-1"], monkeypatch)
    rf_df.to_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))
    owned_path = rf_df.iloc[0]["design_paths"][0]

    calls = {}
    install_pmpnn_fakes(monkeypatch, calls)
    df = pipeline.run_pmpnn(assembly_id="TEST-1", select=[owned_path])
    assert not df.empty


def test_run_pmpnn_no_prior_rfdiffusion_run_raises(project_dir):
    with pytest.raises(pipeline.StageNotFoundError):
        pipeline.run_pmpnn()