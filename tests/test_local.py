"""
test_local.py — `symbro local`/local.py's own unit tests, plus a CLI-level
smoke test and an integration check that a locally-registered structure
flows through run_geometry() exactly like a real RCSB download would.
"""
import os

import pytest
from typer.testing import CliRunner

from toolkit import local, pipeline
from toolkit.cli import app
from toolkit.paths import resolve_path
from conftest import _build_ring_structure

runner = CliRunner()


def _write_pdb(project_dir, name):
    path = os.path.join(str(project_dir), "inputs", name)
    return _build_ring_structure(path)


# ----------------------------------------------------------------------
# local.register_local_structures() -- unit tests
# ----------------------------------------------------------------------

def test_register_single_file_derives_assembly_id_from_filename(project_dir):
    src = _write_pdb(project_dir, "my_cage.pdb")
    df = local.register_local_structures([src])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["assembly_id"] == "my_cage"
    assert row["entry_id"] == "my_cage"
    assert row["assembly_num"] == "local"
    assert os.path.exists(resolve_path(row["filepath"]))
    # copied, not referenced in place
    assert resolve_path(row["filepath"]) != os.path.abspath(src)


def test_register_sanitizes_unsafe_characters_in_filename(project_dir):
    src = _write_pdb(project_dir, "my cage (final v2).pdb")
    df = local.register_local_structures([src])

    assembly_id = df.iloc[0]["assembly_id"]
    assert " " not in assembly_id and "(" not in assembly_id and ")" not in assembly_id


def test_register_dedupes_colliding_default_assembly_ids(project_dir):
    src_a = _write_pdb(project_dir, "cage.pdb")
    src_b = _write_pdb(os.path.join(str(project_dir), "other_folder"), "cage.pdb")
    df = local.register_local_structures([src_a, src_b])

    assert len(df) == 2
    assert len(set(df["assembly_id"])) == 2  # no collision in the output


def test_register_explicit_assembly_ids(project_dir):
    src1 = _write_pdb(project_dir, "a.pdb")
    src2 = _write_pdb(project_dir, "b.pdb")
    df = local.register_local_structures([src1, src2], assembly_ids=["mine-1", "mine-2"])

    assert list(df["assembly_id"]) == ["mine-1", "mine-2"]


def test_register_mismatched_assembly_ids_length_raises(project_dir):
    src = _write_pdb(project_dir, "a.pdb")
    with pytest.raises(ValueError, match="same length"):
        local.register_local_structures([src], assembly_ids=["one", "two"])


def test_register_duplicate_assembly_ids_raises(project_dir):
    src1 = _write_pdb(project_dir, "a.pdb")
    src2 = _write_pdb(project_dir, "b.pdb")
    with pytest.raises(ValueError, match="Duplicate assembly_id"):
        local.register_local_structures([src1, src2], assembly_ids=["same", "same"])


def test_register_missing_file_raises_naming_it(project_dir):
    with pytest.raises(FileNotFoundError, match="nope.pdb"):
        local.register_local_structures(["nope.pdb"])


def test_register_overwrite_false_keeps_existing_copy(project_dir):
    src = _write_pdb(project_dir, "a.pdb")
    df1 = local.register_local_structures([src])
    dest = resolve_path(df1.iloc[0]["filepath"])
    original_mtime = os.path.getmtime(dest)

    os.utime(src, (original_mtime + 100, original_mtime + 100))  # touch source, make it "newer"
    local.register_local_structures([src], overwrite=False)
    assert os.path.getmtime(dest) == pytest.approx(original_mtime)


def test_register_overwrite_true_recopies(project_dir):
    src = _write_pdb(project_dir, "a.pdb")
    df1 = local.register_local_structures([src])
    dest = resolve_path(df1.iloc[0]["filepath"])
    original_mtime = os.path.getmtime(dest)

    os.utime(src, (original_mtime + 100, original_mtime + 100))
    local.register_local_structures([src], overwrite=True)
    # shutil.copy2 preserves the SOURCE's mtime on the copy -- so a
    # recopy's destination mtime should land on src's new (bumped) mtime,
    # not the original. Compared directly (not via pytest.approx, which
    # defaults to a relative tolerance -- meaningless against a ~1.7e9
    # unix timestamp, where even a huge absolute drift is "close enough").
    assert os.path.getmtime(dest) == pytest.approx(original_mtime + 100)


# ----------------------------------------------------------------------
# pipeline.run_local() -- writes the same checkpoint run_download() does
# ----------------------------------------------------------------------

def test_run_local_writes_downloaded_checkpoint(project_dir):
    src = _write_pdb(project_dir, "cage.pdb")
    df = pipeline.run_local([src], state_dir=".symbro")

    assert os.path.exists(os.path.join(".symbro", "downloaded.pkl"))
    reloaded = pipeline.load_checkpoint(pipeline.DOWNLOADED_STAGE, state_dir=".symbro")
    assert list(reloaded["assembly_id"]) == list(df["assembly_id"])


def test_run_local_output_flows_through_geometry(project_dir):
    """The actual point of this feature: a locally-registered structure
    must be indistinguishable from a real download to run_geometry() --
    i.e. it must be accepted and processed by the SAME code path, with no
    special-casing anywhere downstream. Not a claim that this particular
    fixture geometry (two straight-translated chains, no real rotational
    relationship) actually contains detectable symmetry -- ring DETECTION
    correctness is rings.py's own concern; this only checks that a
    locally-sourced row reaches it and comes back with the right shape
    instead of raising."""
    src = _write_pdb(project_dir, "cage.pdb")
    pipeline.run_local([src], state_dir=".symbro")

    rings = pipeline.run_geometry(state_dir=".symbro")  # must not raise
    assert list(rings.columns) == [
        "assembly_id", "symmetry_type", "component_id", "chain_groups",
        "mean_distance", "std_distance", "recommended_linker_length",
        "junctions", "axis_count", "equivalent_groups", "component_chain_count",
    ]


# ----------------------------------------------------------------------
# `symbro local` -- CLI layer
# ----------------------------------------------------------------------

def test_symbro_local_cli_registers_and_prints_next_step(project_dir):
    src = _write_pdb(project_dir, "cage.pdb")
    result = runner.invoke(app, ["local", src])

    assert result.exit_code == 0, result.output
    assert "Registered 1 local structure" in result.output
    assert "symbro geometry" in result.output
    assert os.path.exists(os.path.join(".symbro", "downloaded.pkl"))


def test_symbro_local_cli_missing_file_fails_clearly(project_dir):
    result = runner.invoke(app, ["local", "nope.pdb"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_symbro_local_cli_explicit_assembly_ids(project_dir):
    src1 = _write_pdb(project_dir, "a.pdb")
    src2 = _write_pdb(project_dir, "b.pdb")
    result = runner.invoke(
        app, ["local", src1, src2, "--assembly-id", "mine-1", "--assembly-id", "mine-2"]
    )
    assert result.exit_code == 0, result.output
    reloaded = pipeline.load_checkpoint(pipeline.DOWNLOADED_STAGE, state_dir=".symbro")
    assert list(reloaded["assembly_id"]) == ["mine-1", "mine-2"]
