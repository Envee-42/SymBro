"""
test_af3.py — af3.py's own MSA/template-free fix: when run_data_pipeline
is False, db_dir must not be required, and every candidate's JSON must be
written using AF3's own documented "run completely MSA/template-free"
fields (unpairedMsa/pairedMsa: "", templates: []) rather than left unset
(which docs/input.md instead defines as "build both automatically" — the
opposite of what run_data_pipeline=False is asking for).

Only mocks nothing -- prepare_self_consistency_job()/_write_candidate_json()
write real JSON files to a real tmp_path, read back and inspected directly,
same "real code, mock only the true I/O boundary" convention as the rest
of this suite (the true boundary here -- subprocess.Popen in
_submit_local() -- isn't touched by any test below).
"""
import json
import os

import pandas as pd
import pytest

from toolkit import af3


def _selected_df():
    return pd.DataFrame([{"source_pdb": "design_1", "sequence": "GSSQEEYVELLE", "rank": 1}])


def _design_paths(project_dir):
    path = os.path.join(str(project_dir), "design_1.pdb")
    with open(path, "w") as f:
        f.write("REMARK fake design fixture\nEND\n")
    return [path]


# ----------------------------------------------------------------------
# _write_candidate_json() -- the actual fix
# ----------------------------------------------------------------------

def test_write_candidate_json_default_has_no_msa_fields(project_dir):
    """msa_free=False (the default): no unpairedMsa/pairedMsa/templates
    keys at all -- AF3's own docs define THAT shape as "build both
    automatically", i.e. the real-data-pipeline path."""
    path = os.path.join(str(project_dir), "c.json")
    af3._write_candidate_json(path, "c1", "GSSQEEYVELLE")

    payload = json.load(open(path))
    protein = payload["sequences"][0]["protein"]
    assert "unpairedMsa" not in protein
    assert "pairedMsa" not in protein
    assert "templates" not in protein


def test_write_candidate_json_msa_free_sets_documented_fields(project_dir):
    path = os.path.join(str(project_dir), "c.json")
    af3._write_candidate_json(path, "c1", "GSSQEEYVELLE", msa_free=True)

    payload = json.load(open(path))
    protein = payload["sequences"][0]["protein"]
    assert protein["unpairedMsa"] == ""
    assert protein["pairedMsa"] == ""
    assert protein["templates"] == []
    assert protein["sequence"] == "GSSQEEYVELLE"  # untouched by the fix


# ----------------------------------------------------------------------
# prepare_self_consistency_job() -- db_dir requirement + wiring
# ----------------------------------------------------------------------

def test_prepare_job_run_data_pipeline_false_does_not_require_db_dir(project_dir):
    """THE fix: this must NOT raise, and db_dir must never even be asked
    for -- confirmed against AF3's own run_alphafold.py source that
    db_dir is never read when the data pipeline is off (see af3.py's own
    module docstring)."""
    job = af3.prepare_self_consistency_job(
        _selected_df(), _design_paths(project_dir), model_dir="/fake/models",
        out_dir=str(project_dir), run_data_pipeline=False,
    )
    assert job.db_dir is None

    written = json.load(open(os.path.join(job.input_dir, "design_1_rank1.json")))
    protein = written["sequences"][0]["protein"]
    assert protein["unpairedMsa"] == ""
    assert protein["templates"] == []


def test_prepare_job_run_data_pipeline_true_still_requires_db_dir(project_dir):
    """Regression check: the fix must not weaken the OTHER path -- a real
    data-pipeline run still needs a real db_dir, same as before."""
    with pytest.raises(ValueError, match="db_dir is required"):
        af3.prepare_self_consistency_job(
            _selected_df(), _design_paths(project_dir), model_dir="/fake/models",
            out_dir=str(project_dir), run_data_pipeline=True, db_dir=None,
        )


def test_prepare_job_run_data_pipeline_true_with_db_dir_writes_no_msa_free_fields(project_dir):
    """A real data-pipeline run's JSON must stay in AF3's own "build both
    automatically" shape -- msa_free fields would tell AF3 to skip the
    very search this run is asking it to do."""
    job = af3.prepare_self_consistency_job(
        _selected_df(), _design_paths(project_dir), model_dir="/fake/models",
        out_dir=str(project_dir), run_data_pipeline=True, db_dir="/fake/db",
    )
    written = json.load(open(os.path.join(job.input_dir, "design_1_rank1.json")))
    protein = written["sequences"][0]["protein"]
    assert "unpairedMsa" not in protein
    assert "templates" not in protein


# ----------------------------------------------------------------------
# build_command() -- --db_dir omitted when unset (regression check --
# unrelated to this fix, but this is exactly the flag this fix relies on
# never being passed with a bogus/nonexistent path)
# ----------------------------------------------------------------------

def test_build_command_omits_db_dir_flag_when_unset(project_dir):
    job = af3.prepare_self_consistency_job(
        _selected_df(), _design_paths(project_dir), model_dir="/fake/models",
        out_dir=str(project_dir), run_data_pipeline=False,
    )
    argv = af3.build_command(job)

    assert not any(a.startswith("--db_dir") for a in argv)
    assert "--run_data_pipeline=false" in argv


def test_build_command_includes_db_dir_flag_when_set(project_dir):
    job = af3.prepare_self_consistency_job(
        _selected_df(), _design_paths(project_dir), model_dir="/fake/models",
        out_dir=str(project_dir), run_data_pipeline=True, db_dir="/fake/db",
    )
    argv = af3.build_command(job)

    assert "--db_dir=/fake/db" in argv
    assert "--run_data_pipeline=true" in argv
