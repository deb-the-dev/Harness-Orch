import json
import pytest

from harness.events import SQLiteEventLog
from harness.packs import InvoiceLoggingPack
from harness.policy import HookRunner, OpaProcess, PolicyClient, build_policy_data, policy_version
from harness.router import ModelSpec, Router
from harness.router.selector import StaticPriorSelector
from harness.runs import RunState
from harness.tools import ToolError, ToolRuntime, ToolSpec
from harness.worker import Orchestrator, Worker, contains_pii, redact_pii
from conftest import REGO, OPA
from mock_provider import MockProvider

ORG = "org-1"
INVOICE = '{"vendor": "Acme", "amount": 1200}'
PRIORS = {("extract", "*"): {"m-small": 0.88, "m-large": 0.90}}


@pytest.fixture
async def world(opa, pver):
    mock = await MockProvider().start()
    mock.set("small", text=INVOICE); mock.set("large", text=INVOICE)
    log = SQLiteEventLog()
    client = PolicyClient(base_url=opa.url, version=pver)
    hooks = HookRunner(client, log, ORG)
    cat = {"m-small": ModelSpec("m-small", "openai/small", "eu", 0.5, 1.5, api_base=mock.base("small"), api_key="k"),
           "m-large": ModelSpec("m-large", "openai/large", "eu", 5.0, 15.0, api_base=mock.base("large"), api_key="k")}
    router = Router(cat, StaticPriorSelector(PRIORS), hooks, log, ORG)
    outbox, ledger = [], []
    async def log_invoice(args): ledger.append(args); return {"logged": True}
    async def send(args): outbox.append(args); return {"sent": True}
    tools = ToolRuntime()
    tools.register(ToolSpec("log_invoice", "internal_write", log_invoice, required=("vendor", "amount"), idempotent=True))
    tools.register(ToolSpec("send_vendor_message", "external_comm", send, required=("vendor", "body")))
    pack = InvoiceLoggingPack()
    w = type("W", (), {})()
    w.mock, w.log, w.router, w.hooks, w.tools, w.outbox, w.ledger, w.pack = mock, log, router, hooks, tools, outbox, ledger, pack
    w.orch = Orchestrator(log, ORG)
    w.worker = Worker(log, ORG, {pack.name: pack}, router, hooks, tools)
    yield w
    await client.close(); await mock.stop()


def types(w, run): return [e.type for e in w.log.read(ORG, run)]


async def test_external_message_waits_for_a_human_then_sends(world):
    w = world
    w.orch.submit("r1", "ap_logging", {"invoice_text": "Invoice from Acme for 1200"}, 1.0, "u1")
    view = await w.worker.run("r1")
    assert view.state == RunState.APPROVAL_WAIT and w.ledger and not w.outbox          # logged, but nothing sent yet
    assert view.pending_approval["tool"] == "send_vendor_message"
    w.orch.decide("r1", view.pending_approval["approval_id"], True, "lead-1")
    view = await w.worker.run("r1")                                                    # a fresh worker call resumes from the log
    assert view.state == RunState.SUCCEEDED and view.meta["pack"] == "ap_logging"
    assert w.outbox == [{"vendor": "Acme", "body": "We have logged your invoice for 1200."}]
    assert w.log.verify(ORG, "r1") == []


async def test_rejected_approval_means_nothing_is_sent(world):
    w = world
    w.orch.submit("r2", "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    view = await w.worker.run("r2")
    w.orch.decide("r2", view.pending_approval["approval_id"], False, "lead-1")
    view = await w.worker.run("r2")
    assert view.state == RunState.SUCCEEDED and not w.outbox
    done = [e for e in w.log.read(ORG, "r2") if e.type == "step.completed" and e.step_id == "step-2"][0]
    assert done.payload["result"]["ok"] is False and "approval_rejected" in done.payload["result"]["reasons"]


async def test_verifier_failure_escalates_to_a_stronger_model(world):
    w = world
    w.mock.set("small", text="not json at all")                                        # cheap model fails the verifier
    w.orch.submit("r3", "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    await w.worker.run("r3")
    step0 = [e for e in w.log.read(ORG, "r3") if e.step_id == "step-0"]
    outcomes = [e.payload for e in step0 if e.type == "step.outcome"]
    assert outcomes[0]["ok"] is False and outcomes[-1] == {"ok": True, "kind": "escalation", "model": "m-large", "escalated": True}
    assert [e.payload["model"] for e in step0 if e.type == "model.response"] == ["m-small", "m-large"]


async def test_budget_hard_stop_fails_the_run_with_a_reason(world):
    w = world
    w.orch.submit("r4", "ap_logging", {"invoice_text": "x"}, 0.000001, "u1")
    view = await w.worker.run("r4")
    assert view.state == RunState.FAILED and view.failure_reason == "budget_hard_limit"
    assert w.mock.hits("small") == w.mock.hits("large") == 0


async def test_pii_in_model_output_is_redacted_before_it_reaches_history(world):
    w = world
    w.mock.set("small", text='{"vendor": "Acme", "amount": 1200, "contact": "jo@acme.com"}')
    w.orch.submit("r5", "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    await w.worker.run("r5")
    done = [e for e in w.log.read(ORG, "r5") if e.type == "step.completed" and e.step_id == "step-0"][0]
    assert "jo@acme.com" not in done.payload["result"]["text"] and "[REDACTED]" in done.payload["result"]["text"]


async def test_crash_after_a_non_idempotent_tool_started_needs_a_human(world):
    w = world
    w.orch.submit("r6", "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    view = await w.worker.run("r6")
    w.orch.decide("r6", view.pending_approval["approval_id"], True, "lead-1")
    # simulate a crash: the send started, the result was never recorded
    w.log.append(ORG, "r6", "tool.executing", {"tool": "send_vendor_message"}, step_id="step-2")
    view = w.worker.runs.view("r6")
    assert view.state == RunState.RUNNING
    w.orch.recover("r6")
    view = await w.worker.run("r6")                                                    # new worker: outcome unknown, not idempotent
    assert view.state == RunState.APPROVAL_WAIT and view.pending_approval["reason"] == "outcome_unknown"
    assert not w.outbox                                                                # nothing was sent twice
    w.orch.decide("r6", view.pending_approval["approval_id"], True, "lead-1")          # a human confirms the repeat
    view = await w.worker.run("r6")
    assert view.state == RunState.SUCCEEDED and len(w.outbox) == 1


async def test_a_crash_between_steps_resumes_without_paying_twice(world):
    w = world
    w.orch.submit("r7", "ap_logging", {"invoice_text": "x"}, 1.0, "u1")
    await w.worker.run("r7")                                                           # ends waiting for approval
    spent = w.router.budget.spent("r7")
    w.orch.runs.transition("r7", "run.cancelled", {"by": "test"})
    assert w.router.budget.spent("r7") == spent and w.mock.hits("small") == 1


def test_payment_tools_cannot_be_registered():
    async def noop(a): return {}
    with pytest.raises(ValueError, match="payment"):
        ToolRuntime().register(ToolSpec("pay_invoice", "payment", noop))


async def test_a_pack_cannot_use_a_tool_outside_its_grant(world):
    w = world
    from harness.packs import ToolCall
    d = await w.hooks.check("rX", "s", {"hook": "pre_tool", "pack": "ap_logging", "action_class": "read", "tool": "pay_invoice", "approval": {"state": "none"}})
    assert d.verdict == "block" and d.reasons == ("tool_not_granted",)


def test_pii_helpers():
    assert contains_pii("mail jo@acme.com") and contains_pii("acct 123456789") and not contains_pii("total 1200")
    assert redact_pii("jo@acme.com paid 123456789") == "[REDACTED] paid [REDACTED]"
