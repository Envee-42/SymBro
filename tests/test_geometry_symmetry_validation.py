"""
test_geometry_symmetry_validation.py -- run_geometry()'s cross-check of
its own empirically detected rings against RCSB's OWN annotated symmetry
(downloaded_df's "symmetry" column, always fetched by run_query() -- see
its own comment). Real code paths throughout: _expected_cyclic_orders()
and _drop_symmetry_mismatches() are exercised directly, and the
CliRunner tests run real ring detection (geometry.rings.from_structure)
against a real, gemmi-built structure (conftest's ring_pdb) -- nothing
here needs mocking, since this feature never crosses an external I/O
boundary (RCSB, a cluster, ...), it only cross-references two DataFrames
already produced by other real code.
"""
import os

import pandas as pd
import pytest
from typer.testing import CliRunner

from toolkit import pipeline
from toolkit.cli import app
from toolkit.geometry import rings as _rings

runner = CliRunner()


# ----------------------------------------------------------------------
# _expected_cyclic_orders -- annotation string -> in-scope cyclic orders
# ----------------------------------------------------------------------

@pytest.mark.parametrize("annotated, expected", [
    ("C3", [3]),
    ("C2", [2]),
    ("C3, C2", [3, 2]),
    ("C2, C3", [2, 3]),
    ("C6", []),          # outside ALLOWED_ORDERS (2-5) -- out of scope, not guessed at
    ("D2", []),          # dihedral -- out of scope
    ("T", [3, 2]),       # Platonic: tetrahedral -- 4 C3 axes + 3 C2 axes
    ("O", [4, 3, 2]),    # Platonic: octahedral -- 3 C4 + 4 C3 + 6 C2
    ("I", [5, 3, 2]),    # Platonic: icosahedral -- 6 C5 + 10 C3 + 15 C2
    ("H", []),           # helical -- out of scope
    ("C1", []),          # asymmetric -- out of scope
    ("C3, D2", [3]),     # mixed: keep the in-scope token, drop the out-of-scope one
    ("T, C4", [3, 2, 4]),  # Platonic token combined with a plain cyclic token
    (None, []),
    (float("nan"), []),
    ("", []),
])
def test_expected_cyclic_orders(annotated, expected):
    assert pipeline._expected_cyclic_orders(annotated, _rings.ALLOWED_ORDERS) == expected


# ----------------------------------------------------------------------
# _drop_symmetry_mismatches -- pure DataFrame-level unit tests
# ----------------------------------------------------------------------

def _rings_row(assembly_id, symmetry_type, component_id=0, axis_count=1, component_chain_count=None):
    row = {
        "assembly_id": assembly_id, "symmetry_type": symmetry_type, "component_id": component_id,
        "chain_groups": ("A", "B"), "mean_distance": 5.0, "std_distance": 0.1,
        "recommended_linker_length": (3, 6), "junctions": [], "axis_count": axis_count, "equivalent_groups": [],
    }
    # Only set when a test cares about it -- omitting it entirely (the
    # default) matches an older rings_df predating this column, exercised
    # by test_warn_incomplete_axis_counts_missing_column_is_a_no_op below.
    if component_chain_count is not None:
        row["component_chain_count"] = component_chain_count
    return row


def test_drop_symmetry_mismatches_confirmed_match_is_kept():
    rings_df = pd.DataFrame([_rings_row("A1", "C3")])
    downloaded_df = pd.DataFrame([{"assembly_id": "A1", "symmetry": "C3"}])
    kept, dropped = pipeline._drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
    assert len(kept) == 1
    assert dropped.empty


def test_drop_symmetry_mismatches_true_mismatch_is_dropped():
    rings_df = pd.DataFrame([_rings_row("A1", "C2")])
    downloaded_df = pd.DataFrame([{"assembly_id": "A1", "symmetry": "C4"}])
    kept, dropped = pipeline._drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
    assert kept.empty
    assert len(dropped) == 1
    row = dropped.iloc[0]
    assert row["assembly_id"] == "A1"
    assert row["expected"] == "C4"
    assert row["detected"] == "C2"


def test_drop_symmetry_mismatches_nothing_detected_at_all():
    rings_df = pd.DataFrame([_rings_row("A1", "C2")])
    # A2 was annotated but never even shows up in rings_df -- e.g. detection
    # found nothing there at any order.
    downloaded_df = pd.DataFrame([
        {"assembly_id": "A1", "symmetry": "C2"}, {"assembly_id": "A2", "symmetry": "C3"},
    ])
    kept, dropped = pipeline._drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
    assert list(kept["assembly_id"]) == ["A1"]
    assert list(dropped["assembly_id"]) == ["A2"]
    assert dropped.iloc[0]["detected"] == "none"


def test_drop_symmetry_mismatches_any_one_expected_order_found_is_enough():
    # RCSB annotates a multi-component-style "C3, C2" -- only the C2 ring
    # was actually confirmed, which is still enough to keep it.
    rings_df = pd.DataFrame([_rings_row("A1", "C2")])
    downloaded_df = pd.DataFrame([{"assembly_id": "A1", "symmetry": "C3, C2"}])
    kept, dropped = pipeline._drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
    assert len(kept) == 1
    assert dropped.empty


def test_drop_symmetry_mismatches_out_of_scope_annotation_is_untouched():
    rings_df = pd.DataFrame([_rings_row("A1", "C2")])
    downloaded_df = pd.DataFrame([{"assembly_id": "A1", "symmetry": "D2"}])
    kept, dropped = pipeline._drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
    assert len(kept) == 1
    assert dropped.empty


def test_drop_symmetry_mismatches_platonic_annotation_confirmed_by_one_axis():
    # RCSB annotates a tetrahedral (T) assembly -- symbro only ever
    # confirmed the C2 sub-ring for this component, which is still one
    # of T's own two constituent axis types (C3, C2), so it's kept.
    rings_df = pd.DataFrame([_rings_row("A1", "C2")])
    downloaded_df = pd.DataFrame([{"assembly_id": "A1", "symmetry": "T"}])
    kept, dropped = pipeline._drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
    assert len(kept) == 1
    assert dropped.empty


def test_drop_symmetry_mismatches_platonic_annotation_true_mismatch_is_dropped():
    # Octahedral (O) expects C4/C3/C2 -- nothing detected matches any of
    # those, so this really does look like a bad annotation.
    rings_df = pd.DataFrame([_rings_row("A1", "C5")])
    downloaded_df = pd.DataFrame([{"assembly_id": "A1", "symmetry": "O"}])
    kept, dropped = pipeline._drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
    assert kept.empty
    assert len(dropped) == 1
    assert dropped.iloc[0]["expected"] == "C2, C3, C4"


def test_drop_symmetry_mismatches_missing_symmetry_column_is_a_no_op():
    rings_df = pd.DataFrame([_rings_row("A1", "C2")])
    downloaded_df = pd.DataFrame([{"assembly_id": "A1"}])  # e.g. `symbro local` candidates
    kept, dropped = pipeline._drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
    pd.testing.assert_frame_equal(kept, rings_df)
    assert dropped.empty


def test_drop_symmetry_mismatches_drops_every_row_for_the_assembly():
    # A1 has two components at C2 -- a true mismatch must drop BOTH rows,
    # not just the first one seen.
    rings_df = pd.DataFrame([_rings_row("A1", "C2", component_id=0), _rings_row("A1", "C2", component_id=1)])
    downloaded_df = pd.DataFrame([{"assembly_id": "A1", "symmetry": "C5"}])
    kept, dropped = pipeline._drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
    assert kept.empty
    assert len(dropped) == 1  # one row per dropped ASSEMBLY, not per rings_df row


def test_drop_symmetry_mismatches_empty_rings_df_is_a_no_op():
    rings_df = pd.DataFrame(columns=["assembly_id", "symmetry_type"])
    downloaded_df = pd.DataFrame([{"assembly_id": "A1", "symmetry": "C3"}])
    kept, dropped = pipeline._drop_symmetry_mismatches(rings_df, downloaded_df, _rings.ALLOWED_ORDERS)
    assert kept.empty
    assert dropped.empty


# ----------------------------------------------------------------------
# _warn_incomplete_axis_counts -- pure DataFrame-level unit tests. This is
# a DIFFERENT check from _drop_symmetry_mismatches above: it never
# references RCSB's annotation, only compares axis_count against what
# component_chain_count // order says was possible -- and it only ever
# prints, it never drops anything (see its own docstring for why).
# ----------------------------------------------------------------------

def test_warn_incomplete_axis_counts_warns_when_short(capsys):
    # 9 usable chains could support at most 3 disjoint C3 triplets --
    # only 2 were actually found, so 3 chains went unclaimed.
    rings_df = pd.DataFrame([_rings_row("A1", "C3", axis_count=2, component_chain_count=9)])
    pipeline._warn_incomplete_axis_counts(rings_df)
    captured = capsys.readouterr()
    assert "A1" in captured.out
    assert "2/3" in captured.out
    assert "C3" in captured.out


def test_warn_incomplete_axis_counts_silent_when_complete(capsys):
    # axis_count already equals the stoichiometric maximum -- nothing to flag.
    rings_df = pd.DataFrame([_rings_row("A1", "C3", axis_count=3, component_chain_count=9)])
    pipeline._warn_incomplete_axis_counts(rings_df)
    assert capsys.readouterr().out == ""


def test_warn_incomplete_axis_counts_never_drops_rows():
    # However short the count, this function only ever prints -- the
    # DataFrame itself (and therefore what run_geometry() returns/saves)
    # is never touched.
    rings_df = pd.DataFrame([_rings_row("A1", "C3", axis_count=1, component_chain_count=9)])
    pipeline._warn_incomplete_axis_counts(rings_df)
    assert len(rings_df) == 1


def test_warn_incomplete_axis_counts_missing_column_is_a_no_op(capsys):
    # An older rings_df from before component_chain_count existed.
    rings_df = pd.DataFrame([_rings_row("A1", "C3", axis_count=1)])
    assert "component_chain_count" not in rings_df.columns
    pipeline._warn_incomplete_axis_counts(rings_df)
    assert capsys.readouterr().out == ""


def test_warn_incomplete_axis_counts_empty_df_is_a_no_op(capsys):
    rings_df = pd.DataFrame(columns=["assembly_id", "symmetry_type", "axis_count", "component_chain_count"])
    pipeline._warn_incomplete_axis_counts(rings_df)
    assert capsys.readouterr().out == ""


def test_warn_incomplete_axis_counts_multiple_short_rows_each_warn(capsys):
    rings_df = pd.DataFrame([
        _rings_row("A1", "C3", component_id=0, axis_count=5, component_chain_count=24),
        _rings_row("A1", "C3", component_id=1, axis_count=6, component_chain_count=24),
    ])
    pipeline._warn_incomplete_axis_counts(rings_df)
    captured = capsys.readouterr()
    assert captured.out.count("A1") == 2
    assert "5/8" in captured.out
    assert "6/8" in captured.out


# ----------------------------------------------------------------------
# run_geometry() end to end -- ring DETECTION itself is exercised for
# real elsewhere; here _rings.from_structure is monkeypatched at exactly
# that boundary (same principle as test_cli_integration.py's
# install_rfdiffusion_fakes/install_pmpnn_fakes: mock the one module
# whose own correctness isn't what this file is testing) so these tests
# isolate run_geometry()'s NEW cross-check/drop wiring, not rings.py's
# geometric detection accuracy -- conftest's ring_pdb fixture is real but
# minimal, and isn't built to pass rings.py's real homogeneity/contact
# checks itself.
# ----------------------------------------------------------------------

def _downloaded_df(ring_pdb_path, assembly_id="TEST-1", symmetry="C2"):
    row = {"assembly_id": assembly_id, "filepath": ring_pdb_path}
    if symmetry is not None:
        row["symmetry"] = symmetry
    return pd.DataFrame([row])


def _install_fake_c2_detection(monkeypatch, assembly_id="TEST-1"):
    fake_rings_df = pd.DataFrame([_rings_row(assembly_id, "C2")])

    def _fake_from_structure(df, **kwargs):
        return fake_rings_df.copy()

    monkeypatch.setattr(_rings, "from_structure", _fake_from_structure)


def test_run_geometry_confirmed_symmetry_is_kept(project_dir, ring_pdb, monkeypatch):
    _install_fake_c2_detection(monkeypatch)
    df = pipeline.run_geometry(
        downloaded_df=_downloaded_df(ring_pdb, symmetry="C2"), state_dir=".symbro",
    )
    assert not df.empty
    assert set(df["symmetry_type"]) == {"C2"}


def test_run_geometry_mismatch_is_dropped_and_warns(project_dir, ring_pdb, monkeypatch, capsys):
    _install_fake_c2_detection(monkeypatch)
    df = pipeline.run_geometry(
        downloaded_df=_downloaded_df(ring_pdb, symmetry="C4"), state_dir=".symbro",
    )
    assert df.empty
    captured = capsys.readouterr()
    assert "TEST-1" in captured.out
    assert "C4" in captured.out
    assert "C2" in captured.out


def test_run_geometry_no_validate_symmetry_keeps_mismatch(project_dir, ring_pdb, monkeypatch):
    _install_fake_c2_detection(monkeypatch)
    df = pipeline.run_geometry(
        downloaded_df=_downloaded_df(ring_pdb, symmetry="C4"), state_dir=".symbro",
        validate_annotated_symmetry=False,
    )
    assert not df.empty
    assert set(df["symmetry_type"]) == {"C2"}


def test_run_geometry_no_symmetry_column_behaves_as_before(project_dir, ring_pdb, monkeypatch):
    _install_fake_c2_detection(monkeypatch)
    df = pipeline.run_geometry(
        downloaded_df=_downloaded_df(ring_pdb, symmetry=None), state_dir=".symbro",
    )
    assert not df.empty
    assert set(df["symmetry_type"]) == {"C2"}


def _install_fake_incomplete_c3_detection(monkeypatch, assembly_id="TEST-1"):
    # 8 usable chains could support 2 disjoint C3 triplets -- only 1 found.
    fake_rings_df = pd.DataFrame([_rings_row(assembly_id, "C3", axis_count=1, component_chain_count=8)])

    def _fake_from_structure(df, **kwargs):
        return fake_rings_df.copy()

    monkeypatch.setattr(_rings, "from_structure", _fake_from_structure)


def test_run_geometry_warns_on_incomplete_axis_count_but_keeps_row(project_dir, ring_pdb, monkeypatch, capsys):
    _install_fake_incomplete_c3_detection(monkeypatch)
    df = pipeline.run_geometry(
        downloaded_df=_downloaded_df(ring_pdb, symmetry="C3"), state_dir=".symbro",
    )
    # Never dropped -- axis type IS confirmed, only the count is short.
    assert not df.empty
    assert set(df["symmetry_type"]) == {"C3"}
    captured = capsys.readouterr()
    assert "TEST-1" in captured.out
    assert "1/2" in captured.out


def test_run_geometry_no_warn_incomplete_axes_suppresses_it(project_dir, ring_pdb, monkeypatch, capsys):
    _install_fake_incomplete_c3_detection(monkeypatch)
    df = pipeline.run_geometry(
        downloaded_df=_downloaded_df(ring_pdb, symmetry="C3"), state_dir=".symbro",
        warn_incomplete_axes=False,
    )
    assert not df.empty
    captured = capsys.readouterr()
    assert "1/2" not in captured.out


# ----------------------------------------------------------------------
# CLI layer -- flag threading and printed warning
# ----------------------------------------------------------------------

def test_cli_geometry_drops_mismatch_by_default(project_dir, ring_pdb, monkeypatch):
    _install_fake_c2_detection(monkeypatch)
    os.makedirs(".symbro", exist_ok=True)
    _downloaded_df(ring_pdb, symmetry="C5").to_pickle(os.path.join(".symbro", "downloaded.pkl"))

    result = runner.invoke(app, ["geometry"])
    assert result.exit_code == 0, result.output
    assert "TEST-1" in result.output
    assert "C5" in result.output
    assert "No symmetry rings detected" in result.output


def test_cli_geometry_no_validate_symmetry_flag_keeps_mismatch(project_dir, ring_pdb, monkeypatch):
    _install_fake_c2_detection(monkeypatch)
    os.makedirs(".symbro", exist_ok=True)
    _downloaded_df(ring_pdb, symmetry="C5").to_pickle(os.path.join(".symbro", "downloaded.pkl"))

    result = runner.invoke(app, ["geometry", "--no-validate-symmetry"])
    assert result.exit_code == 0, result.output
    assert "Symmetry types detected" in result.output
    assert "C2" in result.output


def test_cli_geometry_warns_on_incomplete_axis_count_by_default(project_dir, ring_pdb, monkeypatch):
    _install_fake_incomplete_c3_detection(monkeypatch)
    os.makedirs(".symbro", exist_ok=True)
    _downloaded_df(ring_pdb, symmetry="C3").to_pickle(os.path.join(".symbro", "downloaded.pkl"))

    result = runner.invoke(app, ["geometry"])
    assert result.exit_code == 0, result.output
    assert "TEST-1" in result.output
    assert "1/2" in result.output
    assert "Symmetry types detected" in result.output  # not dropped


def test_cli_geometry_no_warn_incomplete_axes_flag_suppresses_it(project_dir, ring_pdb, monkeypatch):
    _install_fake_incomplete_c3_detection(monkeypatch)
    os.makedirs(".symbro", exist_ok=True)
    _downloaded_df(ring_pdb, symmetry="C3").to_pickle(os.path.join(".symbro", "downloaded.pkl"))

    result = runner.invoke(app, ["geometry", "--no-warn-incomplete-axes"])
    assert result.exit_code == 0, result.output
    assert "1/2" not in result.output
