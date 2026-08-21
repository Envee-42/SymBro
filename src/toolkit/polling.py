"""
polling.py — a small, generic helper for the "block until a job reaches a
terminal state" loop that every compute-backend module (boltz.py,
alphafold2.py, af3.py, pmpnn.py) implements around its own run()
convenience wrapper, plus pipeline.run_pmpnn()'s own independently-written
version of the same loop.

THE BUG THIS FIXES
===================
Every poll_status() in this project (boltz.py/alphafold2.py/af3.py/
pmpnn.py/rfdiffusion.py) reports state="completed_partial" whenever a
job's process/SLURM job has already exited with returncode == 0, but the
expected output files (folded structures, sequences, RFdiffusion
designs, ...) aren't all present yet. Two very different real situations
produce this exact same state:

  - Genuinely partial: the process was killed or crashed partway through,
    or otherwise really did stop having written only some outputs. No
    amount of waiting produces the missing files.
  - Transiently partial: the process finished cleanly and really did
    produce every output, but the polling process's own view of the
    filesystem hasn't caught up yet (confirmed for real on a live HPC
    cluster run — NFS/Lustre-style output visibility lag behind a
    returncode==0 exit).

Every blocking "submit + poll-in-a-loop + collect" convenience wrapper in
this project used to stop polling the instant it saw "completed_partial"
once (`while status["state"] == "running": ...`, which exits immediately
on ANY non-running state, including a completed_partial that would have
resolved to "completed" one poll later), treating it as fully terminal —
silently and permanently dropping any candidate caught in the
filesystem-lag window, with no error or warning. Confirmed for real: 3 of
180 boltz candidates from a live HPC run were finished and on disk, but
got excluded from predict.pkl because the one poll that happened to run
landed inside that window.

THE FIX
=======
wait_for_terminal() below re-polls a few more times, a few seconds apart,
specifically when (and only when) state == "completed_partial" — if a
retry's poll comes back "completed" (or shows more outputs found than the
previous poll), that's the transient case resolving itself; if
`partial_retries` consecutive re-polls all report no further progress,
that's treated as confirmed genuinely partial and returned as final.
"running" keeps polling (unaffected by this fix, same as before);
"completed"/"failed" are already unambiguous the first time they're seen
and returned immediately.

Deliberately NOT applied to rfdiffusion.py's own multi-row round-robin
poller (`pipeline._poll_all_rfdiffusion()`, used by `symbro rfdiffusion`'s
blocking wait and `symbro status`) — that poller's single_pass=True mode
is a deliberate one-shot manual refresh for `symbro status`, where a
human re-invoking the command IS the retry, and folding automatic
partial-retries into it would change that command's own documented
"cheap, immediate" contract rather than just fixing a bug. Revisit
_poll_all_rfdiffusion() separately if the same filesystem-lag failure
mode is ever confirmed there too.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

# Every poll_status() dict here uses one of these key names for "how many
# outputs found so far" (folded structures for boltz/alphafold2/af3,
# sequences for pmpnn, designs for rfdiffusion) — checked in this order
# rather than hardcoding a single name, since each module names it
# differently for its own domain.
_PROGRESS_COUNT_KEYS = ("candidates_folded", "sequences_written", "designs_written")


def _progress_count(status: dict) -> int:
    """Reads whichever "how many outputs found so far" field is present
    in a poll_status() dict — see _PROGRESS_COUNT_KEYS above. Falls back
    to the length of any dict-valued field (every module also carries a
    "*_paths" dict/list alongside the int count) so a future backend's
    poll_status() doesn't need to be added here by name to work with this
    helper, only to expose ONE of the two conventions this project
    already uses everywhere.
    """
    for key in _PROGRESS_COUNT_KEYS:
        if key in status:
            return status[key]
    for value in status.values():
        if isinstance(value, dict):
            return len(value)
    raise KeyError(
        f"wait_for_terminal() couldn't find a progress-count field in status dict {status!r} — "
        f"expected one of {_PROGRESS_COUNT_KEYS}, or a dict-valued field, to track partial-retry "
        f"progress against."
    )


def wait_for_terminal(
    poll_fn: Callable[[], dict],
    poll_interval: float = 5.0,
    timeout: Optional[float] = None,
    partial_retries: int = 3,
    partial_retry_interval: float = 5.0,
) -> dict:
    """
    Blocks until poll_fn() (a zero-arg callable — typically
    `lambda: <module>.poll_status(run)`) reports a state this project
    treats as genuinely final: "completed"/"failed" immediately, or
    "completed_partial" only after `partial_retries` consecutive re-polls
    (partial_retry_interval seconds apart) show no further progress — see
    this module's own docstring for exactly why.

    timeout (seconds, None = no limit) bounds only the "running" phase,
    matching every existing caller's own prior timeout semantics — a
    caller that timed out while still "running" gets back a status dict
    with state="running" (same as before this helper existed), and is
    expected to handle that itself exactly as it already did. The
    partial-retry phase below is deliberately NOT subject to `timeout` —
    it's bounded on its own by partial_retries, and conflating the two
    would let an unlucky filesystem-lag poll get cut off by a timeout
    budget already spent waiting on the job itself.
    """
    status = poll_fn()
    start = time.time()
    while status["state"] == "running":
        if timeout is not None and time.time() - start > timeout:
            return status
        time.sleep(poll_interval)
        status = poll_fn()

    if status["state"] != "completed_partial":
        return status

    stable_count = _progress_count(status)
    for _ in range(partial_retries):
        time.sleep(partial_retry_interval)
        status = poll_fn()
        if status["state"] != "completed_partial":
            return status  # resolved -- almost always "completed"
        new_count = _progress_count(status)
        if new_count > stable_count:
            stable_count = new_count  # still making progress -- keep retrying

    return status
