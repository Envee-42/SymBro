"""
test_cli_layer.py — CliRunner-driven end-to-end `symbro rfdiffusion`/
`symbro status`/`symbro pmpnn` invocations: exercises argument parsing,
exit codes, and printed output, not just the pipeline.* functions
underneath (test_cli_integration.py covers those directly).
"""
import os

from typer.testing import CliRunner

from toolkit import pipeline, pmpnn, rfdiffusion
from toolkit.cli import app
from conftest import rings_df
from test_cli_integration import install_rfdiffusion_fakes, install_pmpnn_fakes

runner = CliRunner()


def test_symbro_rfdiffusion_block_and_poll(project_dir, ring_pdb, monkeypatch):
    os.makedirs(".symbro", exist_ok=True)
    rings_df(["TEST-1"], ring_pdb).to_pickle(os.path.join(".symbro", "rings.pkl"))
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)

    result = runner.invoke(app, ["rfdiffusion", "--num-designs", "1", "--diffuser-t", "10"])
    assert result.exit_code == 0, result.output
    assert "1 job(s) submitted" in result.output
    assert "Next:" in result.output
    assert os.path.exists(os.path.join(".symbro", "rfdiffusion.pkl"))


def test_symbro_rfdiffusion_detach_then_status(project_dir, ring_pdb, monkeypatch):
    os.makedirs(".symbro", exist_ok=True)
    rings_df(["TEST-1"], ring_pdb).to_pickle(os.path.join(".symbro", "rings.pkl"))
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)

    result = runner.invoke(app, ["rfdiffusion", "--backend", "slurm", "--detach", "--num-designs", "1"])
    assert result.exit_code == 0, result.output
    assert "symbro status" in result.output
    assert "poll" not in calls

    status_result = runner.invoke(app, ["status"])
    assert status_result.exit_code == 0, status_result.output
    assert "0 still running" in status_result.output
    assert "symbro pmpnn" in status_result.output


def test_symbro_rfdiffusion_detach_non_slurm_fails_clearly(project_dir, ring_pdb, monkeypatch):
    os.makedirs(".symbro", exist_ok=True)
    rings_df(["TEST-1"], ring_pdb).to_pickle(os.path.join(".symbro", "rings.pkl"))
    calls = {}
    install_rfdiffusion_fakes(monkeypatch, calls)

    result = runner.invoke(app, ["rfdiffusion", "--backend", "local", "--detach"])
    assert result.exit_code == 1
    assert "only works with backend='slurm'" in result.output
    assert "submit" not in calls


def test_symbro_rfdiffusion_no_rings_checkpoint_fails_clearly(project_dir):
    result = runner.invoke(app, ["rfdiffusion"])
    assert result.exit_code == 1
    assert "rings" in result.output
    assert "isolate" in result.output.lower() or "run the earlier stage" in result.output


def test_symbro_pmpnn_default_and_select(project_dir, ring_pdb, monkeypatch):
    os.makedirs(".symbro", exist_ok=True)
    rings_df(["TEST-1"], ring_pdb).to_pickle(os.path.join(".symbro", "rings.pkl"))
    rf_calls = {}
    install_rfdiffusion_fakes(monkeypatch, rf_calls)
    rfdiffusion_result = runner.invoke(app, ["rfdiffusion", "--num-designs", "2", "--diffuser-t", "10"])
    assert rfdiffusion_result.exit_code == 0, rfdiffusion_result.output

    mp_calls = {}
    install_pmpnn_fakes(monkeypatch, mp_calls)
    pmpnn_result = runner.invoke(app, ["pmpnn", "--assembly-id", "TEST-1"])
    assert pmpnn_result.exit_code == 0, pmpnn_result.output
    assert "sequence row(s) written" in pmpnn_result.output
    assert os.path.exists(os.path.join(".symbro", "pmpnn.pkl"))

    # --select against a path that isn't one of this assembly's own designs
    # must fail with the specific, named error -- not a generic traceback
    bad_select = runner.invoke(app, ["pmpnn", "--assembly-id", "TEST-1", "--select", "/nope.pdb"])
    assert bad_select.exit_code == 1
    assert "not among" in bad_select.output


def test_symbro_status_without_rfdiffusion_run_fails_clearly(project_dir):
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "rfdiffusion" in result.output.lower()
