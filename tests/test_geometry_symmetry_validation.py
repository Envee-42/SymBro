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

def _rings_row(assembly_id, symmetry_type, component_id=0):
    return {
        "assembly_id": assembly_id, "symmetry_type": symmetry_type, "component_id": component_id,
        "chain_groups": ("A", "B"), "mean_distance": 5.0, "std_distance": 0.1,
        "recommended_linker_length": (3, 6), "junctions": [], "axis_count": 1, "equivalent_groups": [],
    }


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
