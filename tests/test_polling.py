"""
test_polling.py — toolkit.polling.wait_for_terminal().

Regression coverage for the real bug found on a live HPC boltz run: every
backend's run() convenience wrapper (and pipeline.run_pmpnn()'s own inline
version of the same loop) used to stop the instant poll_status() reported
"completed_partial" once, even though that state is reported both for a
genuine crash-with-partial-output AND for a job that finished cleanly
(returncode == 0, every real output on disk) whose files just hadn't
become visible to the polling process yet. 3 of 180 candidates from a
real cluster run fell into that second case and were silently dropped.

Mocks only poll_fn() itself -- the true "outside world" boundary here is
whatever poll_status() would otherwise ask the filesystem/SLURM, same
convention every other test in this suite uses (mock only the actual
outside-world I/O). wait_for_terminal()'s own retry/timeout logic runs
for real. monkeypatches toolkit.polling.time.sleep to a no-op so these
run in well under a second, matching this project's own "no GPU/cluster/
internet access required, well under a second" testing convention.
"""
import pytest

from toolkit import polling


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(polling.time, "sleep", lambda seconds: None)


def _responses(*states_and_counts):
    """Builds a poll_fn() that returns one canned status dict per call,
    repeating the last one forever once exhausted (mirrors a real
    poll_status() that keeps reporting the same terminal state if polled
    again after finishing)."""
    calls = {"n": 0}

    def poll_fn():
        i = min(calls["n"], len(states_and_counts) - 1)
        calls["n"] += 1
        state, count = states_and_counts[i]
        return {
            "state": state, "returncode": 0 if state != "running" else None,
            "candidates_folded": count, "candidates_expected": 3,
            "folded_paths": {f"c{i}": f"/fake/c{i}.cif" for i in range(count)},
            "log_path": "/fake/log",
        }

    poll_fn.calls = calls
    return poll_fn


def test_completed_first_try_returns_immediately():
    poll_fn = _responses(("completed", 3))
    status = polling.wait_for_terminal(poll_fn)
    assert status["state"] == "completed"
    assert poll_fn.calls["n"] == 1


def test_failed_returns_immediately_no_partial_retry():
    poll_fn = _responses(("failed", 0))
    status = polling.wait_for_terminal(poll_fn, partial_retries=5)
    assert status["state"] == "failed"
    assert poll_fn.calls["n"] == 1


def test_running_then_completed_polls_until_done():
    poll_fn = _responses(("running", 0), ("running", 0), ("completed", 3))
    status = polling.wait_for_terminal(poll_fn)
    assert status["state"] == "completed"
    assert poll_fn.calls["n"] == 3


def test_transient_completed_partial_resolves_to_completed():
    """THE regression case: a filesystem-lag completed_partial that
    would have resolved to completed on the very next poll must NOT be
    reported as final -- this is exactly the shape of the 3/180 boltz
    candidates that were silently dropped on the real HPC run."""
    poll_fn = _responses(("completed_partial", 2), ("completed", 3))
    status = polling.wait_for_terminal(poll_fn, partial_retries=3, partial_retry_interval=1)
    assert status["state"] == "completed"
    assert status["candidates_folded"] == 3
    assert poll_fn.calls["n"] == 2


def test_transient_completed_partial_resolves_after_several_retries():
    """Progress trickling in across more than one retry (not just the
    very next poll) must still be given the full partial_retries budget,
    not abandoned after the first unchanged count."""
    poll_fn = _responses(
        ("completed_partial", 1), ("completed_partial", 1),
        ("completed_partial", 2), ("completed", 3),
    )
    status = polling.wait_for_terminal(poll_fn, partial_retries=3, partial_retry_interval=1)
    assert status["state"] == "completed"
    assert poll_fn.calls["n"] == 4


def test_genuinely_partial_stays_partial_after_exhausting_retries():
    """A real crash-with-partial-output: the same incomplete count on
    every retry. Must be returned as completed_partial (not silently
    upgraded), after -- and only after -- partial_retries is exhausted."""
    poll_fn = _responses(("completed_partial", 2))  # repeats forever
    status = polling.wait_for_terminal(poll_fn, partial_retries=3, partial_retry_interval=1)
    assert status["state"] == "completed_partial"
    assert status["candidates_folded"] == 2
    # 1 initial poll + 3 partial retries, all seeing the same stuck count
    assert poll_fn.calls["n"] == 4


def test_partial_retries_zero_preserves_old_immediate_behavior():
    """partial_retries=0 is the pre-fix behavior -- a caller that
    deliberately wants the old "trust it immediately" semantics can still
    get them."""
    poll_fn = _responses(("completed_partial", 2))
    status = polling.wait_for_terminal(poll_fn, partial_retries=0)
    assert status["state"] == "completed_partial"
    assert poll_fn.calls["n"] == 1


def test_timeout_while_running_returns_running_state():
    """A timeout during the running phase must behave exactly as every
    existing caller already handles it: status still reports "running",
    not some new sentinel -- callers key their own timeout message off
    exactly this."""
    poll_fn = _responses(("running", 0))  # never progresses
    real_time = polling.time.time
    ticks = iter([0.0, 0.0, 100.0])  # start, first check (ok), second check (over budget)
    polling.time.time = lambda: next(ticks, 100.0)
    try:
        status = polling.wait_for_terminal(poll_fn, poll_interval=1, timeout=10)
    finally:
        polling.time.time = real_time
    assert status["state"] == "running"


def test_progress_count_reads_sequences_written_for_pmpnn_shaped_status():
    status = {"state": "completed_partial", "sequences_written": 2, "sequences_expected": 3}
    assert polling._progress_count(status) == 2


def test_progress_count_reads_designs_written_for_rfdiffusion_shaped_status():
    status = {"state": "completed_partial", "designs_written": 4, "designs_expected": 10}
    assert polling._progress_count(status) == 4


def test_progress_count_falls_back_to_dict_valued_field():
    status = {"state": "completed_partial", "some_future_field": {"a": 1, "b": 2}}
    assert polling._progress_count(status) == 2


def test_progress_count_raises_when_nothing_recognizable():
    with pytest.raises(KeyError):
        polling._progress_count({"state": "completed_partial"})
