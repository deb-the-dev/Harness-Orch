"""Event log and run state machine contract: runs against SQLite and PostgreSQL (see the `log` fixture)."""
import pytest

from harness.events import payload_hash
from harness.runs import InvalidTransition, RunService, RunState

ORG = "org-1"


@pytest.fixture
def rid():
    import uuid
    return "run-" + uuid.uuid4().hex[:10]       # unique per test: the event log is append-only, so tests never clean up


@pytest.fixture
def org():
    import uuid
    return "org-" + uuid.uuid4().hex[:8]


def test_append_assigns_gapless_seq_and_hashes(log, rid, org):
    a = log.append(org, rid, "x", {"k": 1})
    b = log.append(org, rid, "y")
    c = log.append(org, rid + "-2", "x")
    assert (a.seq, b.seq, c.seq) == (1, 2, 1)
    assert a.payload_hash == payload_hash({"k": 1})
    assert log.verify(org, rid) == []


def test_payload_round_trips_with_a_stable_hash(log, rid, org):
    payload = {"f": 0.1, "tiny": 1e-7, "big": 12345678901234567890, "text": "naïve café 日本", "nested": {"b": [1, 2.5, None, True], "a": "x"}}
    log.append(org, rid, "x", payload)
    assert log.read(org, rid)[0].payload == payload
    assert log.verify(org, rid) == []                # the hash computed at write still matches what is read back


def test_tenant_isolation(log, rid, org):
    log.append(org, rid, "x")
    assert log.read(org + "-other", rid) == []
    with pytest.raises(PermissionError):
        log.append(org + "-other", rid, "x")         # cannot write into another org's run


def test_time_ordered_ids(log, rid, org):
    ids = [log.append(org, rid, "x").id for _ in range(20)]
    assert ids == sorted(ids)


def test_happy_path_and_projection(log, rid, org):
    svc = RunService(log, org)
    svc.create(rid, started_by="u1", pack="demo")
    for ev, state in [("run.queued", RunState.QUEUED), ("run.claimed", RunState.RUNNING),
                      ("approval.requested", RunState.APPROVAL_WAIT), ("approval.decided", RunState.RUNNING),
                      ("run.succeeded", RunState.SUCCEEDED)]:
        payload = {"approval_id": "a1"} if ev == "approval.requested" else {}
        assert svc.transition(rid, ev, payload).state == state
    assert svc.view(rid).started_by == "u1"


def test_invalid_transitions_are_refused_and_not_logged(log, rid, org):
    svc = RunService(log, org)
    svc.create(rid)
    n = len(log.read(org, rid))
    with pytest.raises(InvalidTransition):
        svc.transition(rid, "run.claimed")           # created -> running skips queued
    with pytest.raises(InvalidTransition):
        svc.transition(rid, "run.succeeded")
    assert len(log.read(org, rid)) == n              # a refused move writes nothing


def test_terminal_states_are_final_and_cancel_works_from_any_live_state(log, rid, org):
    svc = RunService(log, org)
    for i, start in enumerate((["run.queued"], ["run.queued", "run.claimed"], ["run.queued", "run.claimed", "run.paused"])):
        r = f"{rid}-{i}"
        svc.create(r)
        for ev in start:
            svc.transition(r, ev)
        assert svc.transition(r, "run.cancelled").state == RunState.CANCELLED
        with pytest.raises(InvalidTransition):
            svc.transition(r, "run.queued")


def test_crash_returns_to_queue_and_budget_stop_fails_from_paused(log, rid, org):
    svc = RunService(log, org)
    svc.create(rid); svc.transition(rid, "run.queued"); svc.transition(rid, "run.claimed")
    assert svc.transition(rid, "run.worker_lost").state == RunState.QUEUED
    svc.transition(rid, "run.claimed"); svc.transition(rid, "run.paused")
    v = svc.transition(rid, "run.failed", {"reason": "budget_hard_limit"})
    assert v.state == RunState.FAILED and v.failure_reason == "budget_hard_limit"


def test_duplicate_run_id_refused(log, rid, org):
    svc = RunService(log, org)
    svc.create(rid)
    with pytest.raises(InvalidTransition):
        svc.create(rid)


def test_sqlite_triggers_refuse_update_and_delete():
    import sqlite3
    from harness.events import SQLiteEventLog
    lg = SQLiteEventLog()
    lg.append(ORG, "r1", "x", {"k": 1})
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        lg._db.execute("UPDATE events SET type='z'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        lg._db.execute("DELETE FROM events")
