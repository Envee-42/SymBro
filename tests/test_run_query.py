"""
test_run_query.py — coverage for the new `symbro query --filter` wiring:
cli.py's `--filter` option -> pipeline.run_query(filter_criteria=...) ->
query.query_candidates(filter_criteria=...).

query.py's own module-level `from rcsbapi... import ...` triggers a real
network fetch at import time (see test_query.py's own module docstring for
the full story), so this file reuses test_query.py's
`_install_rcsbapi_stub()` to import `toolkit.query` safely, then follows
the project's own "mock only at the true I/O boundary" convention (see
test_cli_integration.py's install_rfdiffusion_fakes/install_pmpnn_fakes):
query_candidates() -- the function that actually talks to RCSB -- is
monkeypatched at the toolkit.query module level, exactly like submit()/
poll_status() are monkeypatched for rfdiffusion/pmpnn. Everything above
that boundary (cli.py's argument parsing, pipeline.run_query()'s plumbing)
runs for real.
"""
import os

import pandas as pd
from typer.testing import CliRunner

from toolkit import pipeline
from toolkit.cli import app
from test_query import _install_rcsbapi_stub

_install_rcsbapi_stub()
from toolkit import query  # noqa: E402 -- must come after the stub install above

runner = CliRunner()


def _fake_query_candidates(calls, result_df=None):
    def fake(search_criteria, fetch_fields=None, filter_criteria=None, mode="and", return_type="assembly"):
        calls.append({
            "search_criteria": search_criteria, "fetch_fields": fetch_fields,
            "filter_criteria": filter_criteria, "mode": mode, "return_type": return_type,
        })
        return result_df if result_df is not None else pd.DataFrame(
            [{"assembly_id": "4HHB-1", "symmetry": "C3", "model_quality": 92.0}]
        )
    return fake


# ----------------------------------------------------------------------
# pipeline.run_query() -- filter_criteria plumbing
# ----------------------------------------------------------------------
def test_run_query_passes_filter_criteria_through_to_query_candidates(monkeypatch, project_dir):
    calls = []
    monkeypatch.setattr(query, "query_candidates", _fake_query_candidates(calls))

    filter_criteria = [{"attribute": "model_quality", "value": 70, "operator": "greater_or_equal"}]
    pipeline.run_query(symmetry=["C3"], filter_criteria=filter_criteria, state_dir=".symbro")

    assert len(calls) == 1
    assert calls[0]["filter_criteria"] == filter_criteria


def test_run_query_filter_criteria_alone_does_not_satisfy_search_requirement(project_dir):
    filter_criteria = [{"attribute": "model_quality", "value": 70, "operator": "greater_or_equal"}]
    try:
        pipeline.run_query(filter_criteria=filter_criteria, state_dir=".symbro")
        assert False, "expected ValueError -- filter_criteria alone shouldn't count as a search criterion"
    except ValueError as exc:
        assert "at least one" in str(exc) or "No search criteria" in str(exc)


def test_run_query_without_filter_criteria_still_works(monkeypatch, project_dir):
    """Regression guard: the new optional filter_criteria param must default
    to None cleanly -- every pre-existing run_query() call site (with no
    filter_criteria argument at all) must keep behaving exactly as before."""
    calls = []
    monkeypatch.setattr(query, "query_candidates", _fake_query_candidates(calls))

    pipeline.run_query(symmetry=["C3"], state_dir=".symbro")

    assert len(calls) == 1
    assert calls[0]["filter_criteria"] is None


# ----------------------------------------------------------------------
# cli.py -- `symbro query --filter` end-to-end argument parsing
# ----------------------------------------------------------------------
def test_symbro_query_filter_flag_parsed_and_forwarded(monkeypatch, project_dir):
    calls = []
    monkeypatch.setattr(query, "query_candidates", _fake_query_candidates(calls))

    result = runner.invoke(app, [
        "query", "--symmetry", "C3",
        "--filter", "model_quality=70:greater_or_equal",
        "--filter", "oligomeric_count=24",
    ])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["filter_criteria"] == [
        {"attribute": "model_quality", "value": "70", "operator": "greater_or_equal"},
        {"attribute": "oligomeric_count", "value": "24", "operator": "exact_match"},
    ]
    assert "Found 1 candidate(s)" in result.output
    assert os.path.exists(os.path.join(".symbro", "candidates.pkl"))


def test_symbro_query_without_filter_flag_forwards_none(monkeypatch, project_dir):
    calls = []
    monkeypatch.setattr(query, "query_candidates", _fake_query_candidates(calls))

    result = runner.invoke(app, ["query", "--symmetry", "C3"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["filter_criteria"] is None


def test_symbro_query_filter_flag_bad_syntax_names_the_right_flag(project_dir):
    """A malformed --filter value must be blamed on --filter, not --criterion
    -- both flags share _parse_criterion(), so this is a real regression risk."""
    result = runner.invoke(app, ["query", "--symmetry", "C3", "--filter", "not-a-real-criterion"])
    assert result.exit_code == 1
    assert "--filter" in result.output
    assert "--criterion" not in result.output
