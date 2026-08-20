"""
test_view.py -- pipeline.resolve_structure_path() (the checkpoint lookup
behind `symbro view --stage ...`) and the `symbro view` CLI command
itself. viz.py's own rendering logic has its own dedicated tests in
test_viz.py; these focus on the NEW wiring: resolving a --stage/
--assembly-id pair to a real file, and the CLI layer around it.
"""
import os

import pandas as pd
import pytest
from typer.testing import CliRunner

from toolkit import pipeline
from toolkit.cli import app
from toolkit.paths import to_portable

runner = CliRunner()


# ----------------------------------------------------------------------
# pipeline.resolve_structure_path -- pure checkpoint-lookup unit tests
# ----------------------------------------------------------------------

def _write_checkpoint(stage, rows, state_dir=".symbro"):
    os.makedirs(state_dir, exist_ok=True)
    pd.DataFrame(rows).to_pickle(os.path.join(state_dir, f"{stage}.pkl"))


def test_resolve_structure_path_downloaded_stage(project_dir, ring_pdb):
    _write_checkpoint(pipeline.DOWNLOADED_STAGE, [{"assembly_id": "1ABC-1", "filepath": to_portable(ring_pdb)}])
    resolved = pipeline.resolve_structure_path("downloaded", "1ABC-1", state_dir=".symbro")
    assert os.path.normpath(resolved) == os.path.normpath(ring_pdb)


def test_resolve_structure_path_rings_stage_with_component_id(project_dir, ring_pdb):
    _write_checkpoint(pipeline.ISOLATE_STAGE, [
        {"assembly_id": "1ABC-1", "component_id": 0, "symmetry_type": "C2", "filepath": to_portable(ring_pdb)},
        {"assembly_id": "1ABC-1", "component_id": 1, "symmetry_type": "C2", "filepath": to_portable(ring_pdb)},
    ])
    resolved = pipeline.resolve_structure_path("rings", "1ABC-1", component_id=1, state_dir=".symbro")
    assert os.path.normpath(resolved) == os.path.normpath(ring_pdb)


def test_resolve_structure_path_invalid_stage_raises(project_dir):
    with pytest.raises(ValueError, match="stage must be one of"):
        pipeline.resolve_structure_path("rfdiffusion", "1ABC-1", state_dir=".symbro")


def test_resolve_structure_path_no_matching_row_raises(project_dir, ring_pdb):
    _write_checkpoint(pipeline.DOWNLOADED_STAGE, [{"assembly_id": "1ABC-1", "filepath": to_portable(ring_pdb)}])
    with pytest.raises(ValueError, match="No 'downloaded' row"):
        pipeline.resolve_structure_path("downloaded", "OTHER-1", state_dir=".symbro")


def test_resolve_structure_path_ambiguous_rows_raises(project_dir, ring_pdb):
    # Two components, but component_id not given to disambiguate.
    _write_checkpoint(pipeline.ISOLATE_STAGE, [
        {"assembly_id": "1ABC-1", "component_id": 0, "symmetry_type": "C2", "filepath": to_portable(ring_pdb)},
        {"assembly_id": "1ABC-1", "component_id": 1, "symmetry_type": "C3", "filepath": to_portable(ring_pdb)},
    ])
    with pytest.raises(ValueError, match="2 rows matched"):
        pipeline.resolve_structure_path("rings", "1ABC-1", state_dir=".symbro")


def test_resolve_structure_path_missing_checkpoint_raises(project_dir):
    with pytest.raises(pipeline.StageNotFoundError):
        pipeline.resolve_structure_path("downloaded", "1ABC-1", state_dir=".symbro")


# ----------------------------------------------------------------------
# `symbro view` -- CLI layer
# ----------------------------------------------------------------------

def test_cli_view_direct_path(project_dir, ring_pdb):
    result = runner.invoke(app, ["view", ring_pdb])
    assert result.exit_code == 0, result.output
    expected_output = os.path.splitext(ring_pdb)[0] + ".html"
    assert os.path.exists(expected_output)
    assert "Wrote" in result.output


def test_cli_view_custom_output_path(project_dir, ring_pdb):
    out = os.path.join(str(project_dir), "custom.html")
    result = runner.invoke(app, ["view", ring_pdb, "--output", out])
    assert result.exit_code == 0, result.output
    assert os.path.exists(out)


def test_cli_view_by_stage_and_assembly_id(project_dir, ring_pdb):
    _write_checkpoint(pipeline.DOWNLOADED_STAGE, [{"assembly_id": "1ABC-1", "filepath": to_portable(ring_pdb)}])
    result = runner.invoke(app, ["view", "--stage", "downloaded", "--assembly-id", "1ABC-1"])
    assert result.exit_code == 0, result.output
    assert os.path.exists(os.path.splitext(ring_pdb)[0] + ".html")


def test_cli_view_requires_exactly_one_of_path_or_stage(project_dir, ring_pdb):
    # neither given
    result = runner.invoke(app, ["view"])
    assert result.exit_code != 0
    assert "exactly one" in result.output

    # both given
    result = runner.invoke(app, ["view", ring_pdb, "--stage", "downloaded", "--assembly-id", "1ABC-1"])
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_cli_view_stage_without_assembly_id_fails(project_dir):
    result = runner.invoke(app, ["view", "--stage", "downloaded"])
    assert result.exit_code != 0
    assert "--assembly-id" in result.output


def test_cli_view_nonexistent_path_fails(project_dir):
    result = runner.invoke(app, ["view", "does_not_exist.pdb"])
    assert result.exit_code != 0
    assert "No such file" in result.output


def test_cli_view_missing_checkpoint_fails_clearly(project_dir):
    result = runner.invoke(app, ["view", "--stage", "downloaded", "--assembly-id", "1ABC-1"])
    assert result.exit_code != 0
    assert "downloaded" in result.output
