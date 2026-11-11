"""Worker loop on PostgreSQL: durable event log plus lease queue, with crash and failure recovery."""
import asyncio
import uuid

import psycopg
import pytest

from harness.loop import WorkerLoop
from harness.packs import InvoiceLoggingPack
from harness.pg import PostgresEventLog
from harness.policy import HookRunner, PolicyClient
from harness.queue import PostgresQueue
from harness.router import ModelSpec, Router
from harness.router.selector import StaticPriorSelector
from harness.runs import RunState
from harness.tools import ToolRuntime, ToolSpec
from harness.worker import Orchestrator, Worker
from mock_provider import MockProvider

ORG = "org-" + uuid.uuid4().hex[:6]
INVOICE = '{"vendor": "Acme", "amount": 1200}'


@pytest.fixture
async def sys(opa, pver, pg_app_dsn, pg_admin_dsn):
    with psycopg.connect(pg_admin_dsn, autocommit=True) as c:
        c.execute("TRUNCATE run_queue")
    mock = await MockProvider().start()
    mock.set("small", text=INVOICE); mock.set("large", text=INVOICE)
    log, queue = PostgresEventLog(pg_app_dsn), PostgresQueue(pg_app_dsn)
    client = PolicyClient(base_url=opa.url, version=pver)
    hooks = HookRunner(client, log, ORG)
    cat = {"m-small": ModelSpec("m-small", "openai/small", "eu", 0.5, 1.5, api_base=mock.base("small"), api_key="k"),
           "m-large": ModelSpec("m-large", "openai/large", "eu", 5.0, 15.0, api_base=mock.base("large"), api_key="k")}
    router = Router(cat, StaticPriorSelector({("extract", "*"): {"m-small": 0.88, "m-large": 0.90}}), hooks, log, ORG)
    s = type("S", (), {})()
    s.outbox, s.ledger, s.fail_log = [], [], {"n": 0}

    async def log_invoice(a):
        if s.fail_log["n"] > 0:
            s.fail_log["n"] -= 1
            raise RuntimeError("database exploded")            # a fault, not a ToolError: it reaches the loop
        s.ledger.append(a); return {"logged": True}

    async def send(a): s.outbox.append(a); return {"sent": True}
    tools = ToolRuntime()
    tools.register(ToolSpec("log_invoice", "internal_write", log_invoice, required=("vendor", "amount"), idempotent=True))
    tools.register(ToolSpec("send_vendor_message", "external_comm", send, required=("vendor", "body")))
    pack = InvoiceLoggingPack()
    worker = Worker(log, ORG, {pack.name: pack}, router, hooks, tools)
    orch = Orchestrator(log, ORG, queue)
    s.mock, s.log, s.queue, s.orch, s.worker, s.router = mock, log, queue, orch, worker, router
    s.loop = lambda **kw: WorkerLoop(queue, lambda org: worker, lambda org: orch, **{"lease_s": 5.0, "retry_backoff_s": 0.05, **kw})
    s.admin = pg_admin_dsn
    yield s
    await client.close(); await mock.stop(); log.close(); queue.close()


def rid(): return "run-" + uuid.uuid4().hex[:8]
def view(s, r): return s.orch.runs.view(r)


async def test_submit_wait_for_approval_decide_resume_finish(sys):
    r = rid()
    sys.orch.submit(r, "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    loop = sys.loop()
    assert await loop.run_once() is True
    v = view(sys, r)
    assert v.state == RunState.APPROVAL_WAIT and not sys.outbox
    assert sys.queue.row(r) is None                                    # nothing queued while a human decides
    assert await loop.run_once() is False
    sys.orch.decide(r, v.pending_approval["approval_id"], True, "lead-1")
    assert sys.queue.row(r) is not None                                # the decision woke the run
    assert await loop.run_once() is True
    assert view(sys, r).state == RunState.SUCCEEDED and len(sys.outbox) == 1
    assert sys.queue.row(r) is None and sys.log.verify(ORG, r) == []


async def test_a_dead_worker_is_replaced_without_paying_twice(sys):
    r = rid()
    sys.orch.submit(r, "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    # a worker claims the run, pays for step 0, then dies
    sys.queue.claim("dead-worker", lease_s=0.2)
    sys.orch.runs.transition(r, "run.claimed")
    first = await sys.router.complete(__import__("harness.router", fromlist=["RoutingRequest"]).RoutingRequest(
        ORG, r, "step-0", "ap_logging", "extract", "extract", "internal", [{"role": "system", "content": "Return JSON with vendor and amount."},
        {"role": "user", "content": "x"}], ["m-small", "m-large"], 1.0, 400))
    sys.log.append(ORG, r, "step.started", {"action": {"kind": "model", "step_role": "extract"}}, step_id="step-0")
    hits, spent = sys.mock.hits("small"), sys.router.budget.spent(r)
    await asyncio.sleep(0.3)
    loop = sys.loop()
    assert await loop.reap() == 1
    assert "run.worker_lost" in [e.type for e in sys.log.read(ORG, r)]
    assert await loop.run_once() is True                               # a new worker resumes
    assert view(sys, r).state == RunState.APPROVAL_WAIT                # logged the invoice, now waiting for the human
    assert sys.mock.hits("small") == hits                              # the response already paid for was reused
    assert sys.router.budget.spent(r) == pytest.approx(spent)
    assert sys.ledger[-1] == {"vendor": "Acme", "amount": 1200}


async def test_a_faulty_run_is_retried_then_failed_not_retried_forever(sys):
    r = rid()
    sys.fail_log["n"] = 99
    sys.orch.submit(r, "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    loop = sys.loop(max_attempts=2)
    for _ in range(3):
        assert await loop.run_once() is True
        await asyncio.sleep(0.3)                                       # retry backoff
    v = view(sys, r)
    assert v.state == RunState.FAILED and v.failure_reason == "max_attempts"
    errs = [e for e in sys.log.read(ORG, r) if e.type == "run.worker_error"]
    assert len(errs) == 2 and "database exploded" in errs[0].payload["error"]
    assert sys.queue.row(r) is None


async def test_a_transient_fault_recovers_on_retry(sys):
    r = rid()
    sys.fail_log["n"] = 1
    sys.orch.submit(r, "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    loop = sys.loop()
    await loop.run_once()
    assert view(sys, r).state == RunState.QUEUED
    await asyncio.sleep(0.2)
    await loop.run_once()
    assert view(sys, r).state == RunState.APPROVAL_WAIT and len(sys.ledger) == 1


async def test_a_worker_that_loses_its_lease_stops_and_leaves_the_run_alone(sys):
    r = rid()
    sys.orch.submit(r, "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    gate = asyncio.Event()
    orig = sys.worker.router.complete

    async def slow(req):
        await gate.wait()                                              # the worker is stuck mid-run
        return await orig(req)
    sys.worker.router.complete = slow
    loop = sys.loop(lease_s=0.3, worker_id="slow-worker")
    task = asyncio.ensure_future(loop.run_once())
    await asyncio.sleep(0.15)
    with psycopg.connect(sys.admin, autocommit=True) as c:             # the lease expires and another worker takes over
        c.execute("UPDATE run_queue SET lease_owner = 'other', lease_expires_at = now() + interval '60 seconds' WHERE run_id = %s", (r,))
    assert await asyncio.wait_for(task, 3) is True                     # heartbeat noticed, task cancelled
    assert sys.queue.row(r)["lease_owner"] == "other"                  # the slow worker did not delete or release the new owner's claim
    sys.worker.router.complete = orig


async def test_cancel_removes_the_run_from_the_queue(sys):
    r = rid()
    sys.orch.submit(r, "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    sys.orch.cancel(r, "admin")
    assert sys.queue.row(r) is None and view(sys, r).state == RunState.CANCELLED


async def test_run_forever_drains_work_and_stops(sys):
    ids = [rid() for _ in range(3)]
    for r in ids:
        sys.orch.submit(r, "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    stop = asyncio.Event()
    loop = sys.loop()
    t = asyncio.ensure_future(loop.run_forever(stop, poll_s=0.05))
    for _ in range(100):
        await asyncio.sleep(0.05)
        if all(view(sys, r).state == RunState.APPROVAL_WAIT for r in ids):
            break
    stop.set(); await asyncio.wait_for(t, 3)
    assert all(view(sys, r).state == RunState.APPROVAL_WAIT for r in ids)
