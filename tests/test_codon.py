"""
test_codon.py — toolkit.codon (host-codon-optimization via DNAChisel +
python_codon_tables) and pipeline.py's own join_predict_with_pmpnn()/
run_codon() wiring.

DNAChisel itself is pure, local computation -- no subprocess/network I/O
boundary to mock, unlike RFdiffusion/ProteinMPNN/the structure-prediction
backends -- so these tests run the REAL library against real (if made-up)
protein sequences, same "mock only the true I/O boundary, real code
everywhere else" convention as the rest of this suite. "codon" is an
optional extra (pip install symbro[codon]), not a base dependency, so
every test here that actually needs dnachisel/python_codon_tables is
gated behind pytest.importorskip -- join_predict_with_pmpnn() itself
needs neither (pure pandas), so its tests always run.
"""
import os

import pandas as pd
import pytest
from typer.testing import CliRunner

from toolkit import pipeline
from toolkit.cli import app

runner = CliRunner()

dnachisel = pytest.importorskip(
    "dnachisel", reason='needs the "codon" extra: pip install symbro[codon]'
)


# ----------------------------------------------------------------------
# Fixtures -- minimal, self-contained predict.pkl/pmpnn.pkl-shaped
# DataFrames. Deliberately real amino-acid-only sequences throughout
# (A-Y, no digits) -- codon.py's own input validation correctly rejects
# anything else, so a fixture with fake characters can't exercise it.
# ----------------------------------------------------------------------

SEQ_A = "GSSQEEYVELLEAWKRLDPQNSTVCFHIMYGRAKWQEDL"   # 1ABC-1, design_0, rank 1 (best global_score)
SEQ_A_WORSE = "ACDEFGHIKLMNPQRSTVWYGASQEELVKRLHDPFNTMY"  # 1ABC-1, design_0, rank 2
SEQ_B = "MKAVQERLSDPFGHIKLTNQWERASYVCDMHPLGNTKQW"   # 2XYZ-1, component 1


def _pmpnn_df():
    return pd.DataFrame([
        # 1ABC-1 / design_0: native + two generated candidates (rank 1 = lower global_score)
        {"assembly_id": "1ABC-1", "component_id": None, "source_pdb": "design_0",
         "sequence": "M" * 50, "is_native": True, "temperature": 0.1, "sample_index": 0,
         "score": 1.5, "global_score": 1.4, "seq_recovery": 1.0},
        {"assembly_id": "1ABC-1", "component_id": None, "source_pdb": "design_0",
         "sequence": SEQ_A, "is_native": False, "temperature": 0.1, "sample_index": 1,
         "score": 0.9, "global_score": 0.8, "seq_recovery": 0.4},
        {"assembly_id": "1ABC-1", "component_id": None, "source_pdb": "design_0",
         "sequence": SEQ_A_WORSE, "is_native": False, "temperature": 0.1, "sample_index": 2,
         "score": 1.1, "global_score": 1.0, "seq_recovery": 0.35},
        # 2XYZ-1 / component 1: a multi-component assembly's component_id is a real int
        {"assembly_id": "2XYZ-1", "component_id": 1, "source_pdb": "c1_design_0",
         "sequence": "M" * 40, "is_native": True, "temperature": 0.1, "sample_index": 0,
         "score": 1.6, "global_score": 1.5, "seq_recovery": 1.0},
        {"assembly_id": "2XYZ-1", "component_id": 1, "source_pdb": "c1_design_0",
         "sequence": SEQ_B, "is_native": False, "temperature": 0.1, "sample_index": 1,
         "score": 1.0, "global_score": 0.95, "seq_recovery": 0.35},
    ])


def _predict_df():
    return pd.DataFrame([
        {"assembly_id": "1ABC-1", "component_id": None, "predictor": "boltz",
         "candidate_id": "design_0_rank1", "folded_path": "/fold/a.cif",
         "reference_path": "/ref/a.pdb", "rmsd_to_design": 0.8, "mean_plddt": 91.2},
        {"assembly_id": "2XYZ-1", "component_id": 1, "predictor": "boltz",
         "candidate_id": "c1_design_0_rank1", "folded_path": "/fold/b.cif",
         "reference_path": "/ref/b.pdb", "rmsd_to_design": 1.9, "mean_plddt": 71.0},
    ])


def _write_checkpoints():
    os.makedirs(".symbro", exist_ok=True)
    _pmpnn_df().to_pickle(os.path.join(".symbro", "pmpnn.pkl"))
    _predict_df().to_pickle(os.path.join(".symbro", "predict.pkl"))


# ----------------------------------------------------------------------
# pipeline.join_predict_with_pmpnn() -- pure pandas, no dnachisel needed
# ----------------------------------------------------------------------

def test_join_recovers_sequence_single_component_assembly():
    """component_id=None (single-component assembly) is the exact shape
    run_predict()'s own (assembly_id, component_id) join once had a bug
    for (test_predict.py) -- this join must not repeat it."""
    merged = pipeline.join_predict_with_pmpnn(_predict_df(), _pmpnn_df())
    row = merged[merged["assembly_id"] == "1ABC-1"].iloc[0]
    assert row["sequence"] == SEQ_A          # rank 1 = lowest global_score (0.8), not SEQ_A_WORSE (1.0)
    assert row["global_score"] == 0.8
    assert row["seq_recovery"] == 0.4


def test_join_recovers_sequence_multi_component_assembly():
    merged = pipeline.join_predict_with_pmpnn(_predict_df(), _pmpnn_df())
    row = merged[merged["assembly_id"] == "2XYZ-1"].iloc[0]
    assert row["sequence"] == SEQ_B
    assert row["component_id"] == 1


def test_join_narrowed_predict_df_does_not_crash_on_dtype_mismatch():
    """Regression test: narrowing predict_df to a SINGLE multi-component
    assembly (e.g. `symbro codon --assembly-id 2XYZ-1`) leaves its
    component_id column as pure float64 (no NaN present in the subset),
    while pmpnn_df -- still carrying BOTH assemblies, including
    1ABC-1's component_id=None rows -- produces an object-dtype merge
    key. Before the fix, pandas' merge() raised ValueError on float64
    vs. object key dtypes; this must now join cleanly."""
    narrowed = _predict_df()[_predict_df()["assembly_id"] == "2XYZ-1"].reset_index(drop=True)
    assert narrowed["component_id"].dtype != object  # confirms this reproduces the triggering shape

    merged = pipeline.join_predict_with_pmpnn(narrowed, _pmpnn_df())  # must not raise
    assert len(merged) == 1
    assert merged.iloc[0]["sequence"] == SEQ_B


def test_join_missing_pmpnn_row_leaves_nan_not_a_crash():
    """A checkpoint mismatch (predict.pkl referencing a candidate_id no
    longer in pmpnn.pkl) must produce NaN in the joined columns, not
    raise -- this join is meant for exploratory/downstream tooling, not
    a pipeline stage that should fail hard on a stale checkpoint."""
    predict_df = _predict_df().copy()
    predict_df.loc[0, "candidate_id"] = "design_0_rank_99"  # doesn't exist in pmpnn_df

    merged = pipeline.join_predict_with_pmpnn(predict_df, _pmpnn_df())
    assert pd.isna(merged.iloc[0]["sequence"])
    assert merged.iloc[1]["sequence"] == SEQ_B  # the OTHER row still matches fine


# ----------------------------------------------------------------------
# codon.optimize_sequence()
# ----------------------------------------------------------------------

def test_optimize_sequence_translates_back_to_input():
    from toolkit import codon
    result = codon.optimize_sequence(SEQ_A, host="e_coli")
    stop_stripped = result["dna_sequence"][:-3]
    assert dnachisel.translate(stop_stripped) == SEQ_A


def test_optimize_sequence_gc_content_within_requested_bounds():
    from toolkit import codon
    result = codon.optimize_sequence(SEQ_A, host="e_coli", gc_min=0.4, gc_max=0.6)
    assert 0.4 <= result["gc_content"] <= 0.6
    assert result["warnings"] == []


def test_optimize_sequence_no_long_homopolymer_runs():
    from toolkit import codon
    result = codon.optimize_sequence(SEQ_A, host="e_coli", homopolymer_max=5)
    dna = result["dna_sequence"]
    for nt in "ATGC":
        assert (nt * 6) not in dna


def test_optimize_sequence_appends_hosts_preferred_stop_codon():
    from toolkit import codon
    result = codon.optimize_sequence(SEQ_A, host="e_coli", add_stop_codon=True)
    assert result["dna_sequence"][-3:] == "TAA"  # E. coli's own most-frequent stop, ~64% per Kazusa


def test_optimize_sequence_no_stop_codon_when_disabled():
    from toolkit import codon
    result = codon.optimize_sequence(SEQ_A, host="e_coli", add_stop_codon=False)
    assert len(result["dna_sequence"]) == len(SEQ_A) * 3
    assert dnachisel.translate(result["dna_sequence"]) == SEQ_A


def test_optimize_sequence_different_host_changes_output():
    from toolkit import codon
    e_coli = codon.optimize_sequence(SEQ_A, host="e_coli")
    yeast = codon.optimize_sequence(SEQ_A, host="s_cerevisiae")
    assert e_coli["dna_sequence"] != yeast["dna_sequence"]
    # both must still translate back to the SAME protein regardless of host
    assert dnachisel.translate(e_coli["dna_sequence"][:-3]) == SEQ_A
    assert dnachisel.translate(yeast["dna_sequence"][:-3]) == SEQ_A


def test_optimize_sequence_invalid_host_raises():
    from toolkit import codon
    with pytest.raises(ValueError, match="not supported"):
        codon.optimize_sequence(SEQ_A, host="not_a_real_organism")


def test_optimize_sequence_invalid_amino_acid_raises():
    from toolkit import codon
    with pytest.raises(ValueError, match="non-standard amino acid"):
        codon.optimize_sequence("MKTXQ", host="e_coli")  # X isn't a standard residue


def test_optimize_sequence_empty_raises():
    from toolkit import codon
    with pytest.raises(ValueError, match="empty"):
        codon.optimize_sequence("", host="e_coli")


# ----------------------------------------------------------------------
# codon.optimize_candidates() / write_fasta()
# ----------------------------------------------------------------------

def test_optimize_candidates_batches_over_joined_df():
    from toolkit import codon
    joined = pipeline.join_predict_with_pmpnn(_predict_df(), _pmpnn_df())
    df = codon.optimize_candidates(joined, host="e_coli")
    assert len(df) == 2
    assert set(df["assembly_id"]) == {"1ABC-1", "2XYZ-1"}
    assert list(df.columns) == list(codon._CODON_COLUMNS)


def test_optimize_candidates_skips_row_with_missing_sequence(capsys):
    from toolkit import codon
    joined = pipeline.join_predict_with_pmpnn(_predict_df(), _pmpnn_df())
    joined.loc[0, "sequence"] = None

    df = codon.optimize_candidates(joined, host="e_coli")
    assert len(df) == 1
    assert "Skipped" in capsys.readouterr().out


def test_optimize_candidates_empty_result_keeps_schema():
    from toolkit import codon
    empty = pd.DataFrame(columns=["assembly_id", "component_id", "candidate_id", "sequence"])
    df = codon.optimize_candidates(empty, host="e_coli")
    assert df.empty
    assert list(df.columns) == list(codon._CODON_COLUMNS)


def test_optimize_candidates_fails_fast_on_bad_host_not_per_row():
    """Regression test: an invalid host= must raise ONCE, up front --
    not get caught by the per-row try/except and reported as N identical
    'Skipped' lines while still returning a (confusingly) empty
    DataFrame instead of a clear error."""
    from toolkit import codon
    joined = pipeline.join_predict_with_pmpnn(_predict_df(), _pmpnn_df())
    with pytest.raises(ValueError, match="not supported"):
        codon.optimize_candidates(joined, host="bogus_host")


def test_write_fasta_format(tmp_path):
    from toolkit import codon
    df = pd.DataFrame([
        {"assembly_id": "1ABC-1", "component_id": None, "candidate_id": "design_0_rank1",
         "host": "e_coli", "protein_sequence": "MK", "dna_sequence": "ATGAAATAA",
         "gc_content": 0.33, "warnings": None},
    ])
    path = os.path.join(str(tmp_path), "out.fasta")
    codon.write_fasta(df, path)

    content = open(path).read()
    assert content == ">1ABC-1_design_0_rank1\nATGAAATAA\n"


# ----------------------------------------------------------------------
# pipeline.run_codon() -- full integration, real checkpoints on disk
# ----------------------------------------------------------------------

def test_run_codon_end_to_end_writes_checkpoint_and_fasta(project_dir):
    _write_checkpoints()
    df = pipeline.run_codon(state_dir=".symbro")

    assert len(df) == 2
    assert os.path.exists(os.path.join(".symbro", "codon.pkl"))
    assert os.path.exists(os.path.join(".symbro", "codon.csv"))
    assert os.path.exists(os.path.join(".symbro", "codon.fasta"))

    reloaded = pd.read_pickle(os.path.join(".symbro", "codon.pkl"))
    assert list(reloaded.columns) == list(df.columns)


def test_run_codon_narrows_to_one_assembly(project_dir):
    _write_checkpoints()
    df = pipeline.run_codon(assembly_id="2XYZ-1", state_dir=".symbro")
    assert set(df["assembly_id"]) == {"2XYZ-1"}


def test_run_codon_no_predict_checkpoint_raises(project_dir):
    with pytest.raises(pipeline.StageNotFoundError):
        pipeline.run_codon(state_dir=".symbro")


def test_run_codon_empty_predict_raises_valueerror(project_dir):
    os.makedirs(".symbro", exist_ok=True)
    _pmpnn_df().to_pickle(os.path.join(".symbro", "pmpnn.pkl"))
    pd.DataFrame(columns=list(pipeline._PREDICT_COLUMNS)).to_pickle(os.path.join(".symbro", "predict.pkl"))

    with pytest.raises(ValueError, match="no validated candidates"):
        pipeline.run_codon(state_dir=".symbro")


def test_run_codon_missing_extra_propagates_importerror(project_dir, monkeypatch):
    """If dnachisel/python_codon_tables aren't installed, run_codon()
    must raise ImportError -- not silently produce an empty result."""
    from toolkit import codon
    _write_checkpoints()
    monkeypatch.setattr(codon, "_dc", None)
    monkeypatch.setattr(codon, "_IMPORT_ERROR", ImportError("simulated missing dnachisel"))

    with pytest.raises(ImportError, match="codon.*extra"):
        pipeline.run_codon(state_dir=".symbro")


def test_clean_removes_codon_checkpoint_and_fasta(project_dir):
    _write_checkpoints()
    pipeline.run_codon(state_dir=".symbro")
    assert os.path.exists(os.path.join(".symbro", "codon.fasta"))

    cleared = pipeline.clean(downloads=False, subunits=False, simulations=False, state_dir=".symbro")
    assert "codon" in cleared["state"]
    assert not os.path.exists(os.path.join(".symbro", "codon.pkl"))
    assert not os.path.exists(os.path.join(".symbro", "codon.fasta"))


def test_clean_dry_run_does_not_delete_codon_fasta(project_dir):
    _write_checkpoints()
    pipeline.run_codon(state_dir=".symbro")

    pipeline.clean(downloads=False, subunits=False, simulations=False, dry_run=True, state_dir=".symbro")
    assert os.path.exists(os.path.join(".symbro", "codon.fasta"))
    assert os.path.exists(os.path.join(".symbro", "codon.pkl"))


# ----------------------------------------------------------------------
# CLI layer -- `symbro codon`
# ----------------------------------------------------------------------

def test_symbro_codon_success(project_dir):
    _write_checkpoints()
    result = runner.invoke(app, ["codon"])
    assert result.exit_code == 0, result.output
    assert "2 sequence(s) codon-optimized" in result.output
    assert os.path.exists(os.path.join(".symbro", "codon.fasta"))


def test_symbro_codon_narrows_with_assembly_id_flag(project_dir):
    _write_checkpoints()
    result = runner.invoke(app, ["codon", "--assembly-id", "2XYZ-1"])
    assert result.exit_code == 0, result.output
    assert "1 sequence(s) codon-optimized" in result.output


def test_symbro_codon_no_predict_checkpoint_fails_clearly(project_dir):
    result = runner.invoke(app, ["codon"])
    assert result.exit_code == 1
    assert "predict" in result.output.lower()


def test_symbro_codon_bad_host_fails_once_not_per_candidate(project_dir):
    """CLI-level regression test for the same fail-fast fix: a bad --host
    must produce ONE clear error line, not one "Skipped" line per
    candidate followed by a misleading empty-result summary."""
    _write_checkpoints()
    result = runner.invoke(app, ["codon", "--host", "not_a_real_organism"])
    assert result.exit_code == 1
    assert result.output.count("Skipped") == 0
    assert "not supported" in result.output


def test_symbro_codon_missing_extra_shows_actionable_pip_command(project_dir, monkeypatch):
    """CLI-level regression test for the Rich-markup-swallows-brackets
    bug: the printed error must contain the actual, copy-pasteable
    "pip install symbro[codon]" -- not have "[codon]" silently eaten by
    Typer/Rich's own markup parser (rich_markup_mode="rich" on the app)."""
    from toolkit import codon
    _write_checkpoints()
    monkeypatch.setattr(codon, "_dc", None)
    monkeypatch.setattr(codon, "_IMPORT_ERROR", ImportError("simulated missing dnachisel"))

    result = runner.invoke(app, ["codon"])
    assert result.exit_code == 1
    assert "pip install symbro[codon]" in result.output
