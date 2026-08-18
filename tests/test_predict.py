"""
test_predict.py — pipeline.run_predict() / `symbro predict`.

Regression coverage for the (assembly_id, component_id) join bug found in
review: subset.groupby(["assembly_id", "component_id"], dropna=False, ...)
hands back a float nan for a missing component_id (the common
single-component-assembly case), never the original None -- comparing
that nan against rfdiffusion_df's own component_id column with plain ==
never matches, even though the row is right there. Every fixture here
that doesn't explicitly set a component_id (i.e. every rings_df() row,
same as every other test in this suite) reproduces that exact shape "for
free" -- which is exactly why it slipped through un-tested the first time.

Mocks only structure_prediction.run() -- the actual fold-and-screen call,
itself already a full prepare/submit/poll/collect/validate round trip
inside boltz.run()/alphafold2.run()/af3.run() -- everything upstream of it
(checkpoint loading, the (assembly_id, component_id) join, top_n
shortlisting) runs for real, same convention as test_cli_integration.py's
own install_rfdiffusion_fakes/install_pmpnn_fakes.
"""
import os

import pandas as pd
import pytest

from toolkit import pipeline, structure_prediction
from conftest import rings_df
from tests.test_cli_integration import install_rfdiffusion_fakes, install_pmpnn_fakes, _completed_rfdiffusion_df


def install_predict_fake(monkeypatch, calls, winners_factory=None):
    """winners_factory(selected_df, design_paths) -> DataFrame; default
    returns one passing row per shortlist candidate, in
    selfconsistency.collect_results()'s own column shape."""
    def fake_run(predictor, selected_df, design_paths, config=None, backend=None,
                 max_rmsd=2.0, min_plddt=70.0, **kwargs):
        calls.setdefault("run", []).append({
            "predictor": predictor, "sources": list(selected_df["source_pdb"]),
            "design_paths": list(design_paths),
        })
        if winners_factory is not None:
            return winners_factory(selected_df, design_paths)
        return pd.DataFrame([
            {
                "candidate_id": f"{row['source_pdb']}_rank{row['rank']}",
                "folded_path": f"/fake/{row['source_pdb']}_rank{row['rank']}.cif",
                "reference_path": design_paths[0],
                "rmsd_to_design": 0.5,
                "mean_plddt": 95.0,
            }
            for _, row in selected_df.iterrows()
        ])
    monkeypatch.setattr(structure_prediction, "run", fake_run)


def _completed_pmpnn_df(project_dir, ring_pdb, assembly_ids, monkeypatch):
    """Real rfdiffusion.pkl + pmpnn.pkl on disk, via the SAME fakes/helpers
    test_cli_integration.py already uses -- rings_df() never sets a
    component_id column, so every row here has component_id=None end to
    end, exactly the shape that broke the (assembly_id, component_id)
    join."""
    rf_df = _completed_rfdiffusion_df(project_dir, ring_pdb, assembly_ids, monkeypatch)
    rf_df.to_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))

    pmpnn_calls = {}
    install_pmpnn_fakes(monkeypatch, pmpnn_calls)
    pmpnn_df = pipeline.run_pmpnn()
    return rf_df, pmpnn_df


# ----------------------------------------------------------------------
def test_run_predict_matches_single_component_assembly(project_dir, ring_pdb, monkeypatch):
    """THE regression test: a single-component assembly's component_id is
    None end-to-end (rings_df() never sets one). Before the fix, the
    (assembly_id, component_id) join always missed it -- groupby handed
    back nan, nan != None/nan under plain == -- so every assembly was
    silently skipped ("no matching row in the rfdiffusion checkpoint"),
    even though the row was right there."""
    _completed_pmpnn_df(project_dir, ring_pdb, ["TEST-1"], monkeypatch)

    calls = {}
    install_predict_fake(monkeypatch, calls)
    df = pipeline.run_predict()

    assert len(calls["run"]) == 1  # structure_prediction.run() was actually reached
    assert not df.empty
    assert set(df["assembly_id"]) == {"TEST-1"}
    assert os.path.exists(os.path.join(".symbro", "predict.pkl"))


def test_run_predict_matches_multi_component_assembly(project_dir, ring_pdb, monkeypatch):
    """Explicit component_id values (a real int, not None) must still join
    correctly -- the fix must not special-case away the non-null path."""
    rings = rings_df(["TEST-1"], ring_pdb)
    rings["component_id"] = 1
    calls_rf = {}
    install_rfdiffusion_fakes(monkeypatch, calls_rf)
    rf_df = pipeline.run_rfdiffusion(rings, num_designs=2, diffuser_T=10)
    rf_df.to_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))

    pmpnn_calls = {}
    install_pmpnn_fakes(monkeypatch, pmpnn_calls)
    pipeline.run_pmpnn()

    predict_calls = {}
    install_predict_fake(monkeypatch, predict_calls)
    df = pipeline.run_predict()

    assert len(predict_calls["run"]) == 1
    assert not df.empty
    assert df.iloc[0]["component_id"] == 1


def test_run_predict_skips_assembly_with_no_rfdiffusion_row(project_dir, ring_pdb, monkeypatch):
    _completed_pmpnn_df(project_dir, ring_pdb, ["TEST-1"], monkeypatch)
    # Simulate a stale/mismatched rfdiffusion checkpoint: no rows at all.
    empty_rf = pd.DataFrame(columns=[
        "assembly_id", "symmetry_type", "component_id", "chain_groups", "run", "state", "design_paths",
    ])
    empty_rf.to_pickle(os.path.join(".symbro", "rfdiffusion.pkl"))

    calls = {}
    install_predict_fake(monkeypatch, calls)
    df = pipeline.run_predict()

    assert "run" not in calls  # never reached structure_prediction.run()
    assert df.empty


def test_run_predict_empty_result_keeps_full_schema(project_dir, ring_pdb, monkeypatch):
    """Nothing passes screening -- predict.pkl must still carry the real
    column schema, not a bare, columnless DataFrame (matches every other
    stage's own empty-result convention -- see _PMPNN_SEQUENCE_COLUMNS)."""
    _completed_pmpnn_df(project_dir, ring_pdb, ["TEST-1"], monkeypatch)

    calls = {}
    install_predict_fake(monkeypatch, calls, winners_factory=lambda selected_df, design_paths: pd.DataFrame(
        columns=["candidate_id", "folded_path", "reference_path", "rmsd_to_design", "mean_plddt"]
    ))
    df = pipeline.run_predict()

    assert df.empty
    assert list(df.columns) == list(pipeline._PREDICT_COLUMNS)
    reloaded = pd.read_pickle(os.path.join(".symbro", "predict.pkl"))
    assert list(reloaded.columns) == list(pipeline._PREDICT_COLUMNS)


def test_run_predict_no_prior_pmpnn_run_raises(project_dir):
    with pytest.raises(pipeline.StageNotFoundError):
        pipeline.run_predict()


def test_run_predict_narrows_to_one_assembly(project_dir, ring_pdb, monkeypatch):
    _completed_pmpnn_df(project_dir, ring_pdb, ["TEST-1", "TEST-2"], monkeypatch)

    calls = {}
    install_predict_fake(monkeypatch, calls)
    df = pipeline.run_predict(assembly_id="TEST-2")

    assert len(calls["run"]) == 1
    assert set(df["assembly_id"]) == {"TEST-2"}
