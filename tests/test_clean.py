"""test_clean.py — pipeline.clean() / `symbro clean`."""
import os

from typer.testing import CliRunner

from toolkit import download, isolate, pipeline, pmpnn
from toolkit.cli import app

runner = CliRunner()


def _populate_full_state(project_dir):
    os.makedirs(".symbro", exist_ok=True)
    for stage in pipeline._ALL_STAGES:
        pkl_path, csv_path = pipeline._checkpoint_paths(stage, ".symbro")
        open(pkl_path, "wb").close()
        open(csv_path, "w").close()
    for get_dir, name in (
        (download.get_temp_dir, "a.cif"),
        (isolate.get_temp_subunits_dir, "b.pdb"),
        (pmpnn.get_simulations_dir, "c.fa"),
    ):
        d = get_dir()
        with open(os.path.join(d, name), "w") as f:
            f.write("x")


def test_clean_dry_run_deletes_nothing(project_dir):
    _populate_full_state(project_dir)
    cleared = pipeline.clean(dry_run=True)

    assert set(cleared["state"]) == set(pipeline._ALL_STAGES)
    assert cleared["temporary_files"] and cleared["temporary_subunits"] and cleared["temporary_simulations"]
    # nothing actually removed
    assert os.path.exists(os.path.join(".symbro", "candidates.pkl"))
    assert os.listdir(download.get_temp_dir())
    assert os.listdir(isolate.get_temp_subunits_dir())
    assert os.listdir(pmpnn.get_simulations_dir())


def test_clean_full_clears_everything_but_folders_survive(project_dir):
    _populate_full_state(project_dir)
    cleared = pipeline.clean()

    assert set(cleared["state"]) == set(pipeline._ALL_STAGES)
    for stage in pipeline._ALL_STAGES:
        pkl_path, csv_path = pipeline._checkpoint_paths(stage, ".symbro")
        assert not os.path.exists(pkl_path)
        assert not os.path.exists(csv_path)
    for get_dir in (download.get_temp_dir, isolate.get_temp_subunits_dir, pmpnn.get_simulations_dir):
        d = get_dir()
        assert os.path.isdir(d)          # folder itself survives
        assert os.listdir(d) == []       # but is empty


def test_clean_rerun_on_already_clean_tree_reports_nothing(project_dir):
    _populate_full_state(project_dir)
    pipeline.clean()
    cleared = pipeline.clean()

    assert cleared["state"] == []
    assert not cleared["temporary_files"]
    assert not cleared["temporary_subunits"]
    assert not cleared["temporary_simulations"]


def test_clean_keep_state_only_clears_scratch_dirs(project_dir):
    _populate_full_state(project_dir)
    cleared = pipeline.clean(state=False)

    assert cleared["state"] == []
    assert os.path.exists(os.path.join(".symbro", "candidates.pkl"))  # untouched
    assert os.listdir(download.get_temp_dir()) == []


def test_clean_on_a_fresh_checkout_with_no_state_at_all(project_dir):
    """No .symbro/, no temporary_*/ folders at all yet -- must not crash
    (get_temp_dir()-style helpers create-on-read, so this exercises that)."""
    cleared = pipeline.clean(dry_run=True)
    assert cleared["state"] == []
    assert not cleared["temporary_files"]


def test_symbro_clean_cli_dry_run_then_real(project_dir):
    _populate_full_state(project_dir)
    dry = runner.invoke(app, ["clean", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "Would clear" in dry.output
    assert os.path.exists(os.path.join(".symbro", "candidates.pkl"))

    real = runner.invoke(app, ["clean"])
    assert real.exit_code == 0, real.output
    assert "Cleared" in real.output
    assert not os.path.exists(os.path.join(".symbro", "candidates.pkl"))

    again = runner.invoke(app, ["clean"])
    assert again.exit_code == 0, again.output
    assert "Nothing to clear" in again.output
