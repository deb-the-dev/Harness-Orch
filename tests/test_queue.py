import threading
import time
import uuid

import psycopg
import pytest

from harness.queue import PostgresQueue


@pytest.fixture
def q(pg_app_dsn, pg_admin_dsn):
    with psycopg.connect(pg_admin_dsn, autocommit=True) as c:
        c.execute("TRUNCATE run_queue")
    queue = PostgresQueue(pg_app_dsn)
    yield queue
    queue.close()


def rid():
    return "run-" + uuid.uuid4().hex[:8]


def test_enqueue_claim_release_cycle(q):
    r = rid()
    q.enqueue("org-1", r)
    q.enqueue("org-1", r)                                  # idempotent
    assert q.depth() == {"ready_or_waiting": 1, "leased": 0}
    c = q.claim("w1")
    assert (c.org_id, c.run_id, c.attempts) == ("org-1", r, 1)
    assert q.claim("w2") is None                            # held: nobody else gets it
    assert q.release(r, "w1") == "deleted"
    assert q.depth() == {"ready_or_waiting": 0, "leased": 0}


def test_delayed_work_is_not_claimable_early(q):
    r = rid()
    q.enqueue("org-1", r, delay_s=0.4)
    assert q.claim("w1") is None
    time.sleep(0.5)
    assert q.claim("w1").run_id == r


def test_expired_lease_is_reaped_and_taken_by_another_worker(q):
    r = rid()
    q.enqueue("org-1", r)
    q.claim("w1", lease_s=0.2)
    time.sleep(0.3)
    assert q.expire_leases() == [("org-1", r)]
    c = q.claim("w2")
    assert c.run_id == r and c.attempts == 2                # attempts count claims, so a poison run can be stopped


def test_a_worker_that_lost_its_lease_is_told_so(q):
    r = rid()
    q.enqueue("org-1", r)
    q.claim("w1", lease_s=0.2)
    assert q.heartbeat(r, "w1", 0.2)                        # still ours: extended
    time.sleep(0.3)
    assert not q.heartbeat(r, "w1")                         # expired
    q.expire_leases(); q.claim("w2", lease_s=5)
    assert not q.heartbeat(r, "w1")                         # taken over by w2: w1 must stop
    assert q.release(r, "w1") == "lost"                     # and cannot delete w2's claim
    assert q.row(r)["lease_owner"] == "w2"


def test_work_arriving_while_leased_is_not_lost(q):
    r = rid()
    q.enqueue("org-1", r)
    q.claim("w1")
    q.enqueue("org-1", r)                                   # for example an approval decision, while the worker is still finishing
    assert q.row(r)["wake"] is True
    assert q.release(r, "w1") == "redelivered"
    assert q.claim("w2").run_id == r


def test_release_with_retry_delay_backs_off(q):
    r = rid()
    q.enqueue("org-1", r)
    q.claim("w1")
    assert q.release(r, "w1", retry_after_s=0.4) == "redelivered"
    assert q.claim("w2") is None
    time.sleep(0.5)
    assert q.claim("w2").attempts == 2


def test_each_run_is_claimed_exactly_once_under_contention(q):
    runs = {rid() for _ in range(60)}
    for r in runs:
        q.enqueue("org-1", r)
    got, lock = [], threading.Lock()

    def worker(n):
        while True:
            c = q.claim(f"w{n}", lease_s=30)
            if c is None:
                return
            with lock:
                got.append(c.run_id)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(got) == sorted(runs)                      # every run once, none twice, none skipped


def test_queue_is_per_run_not_per_org_and_refuses_cross_org_requeue(q):
    r = rid()
    q.enqueue("org-1", r)
    q.enqueue("org-2", r)                                   # another org cannot touch org-1's row
    assert q.claim("w1").org_id == "org-1"
