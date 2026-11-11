import pytest

from harness.events import SQLiteEventLog
from harness.policy import HookRunner, PolicyClient

def pre_tool(**kw):
    base = {"hook": "pre_tool", "pack": "ap_logging", "action_class": "internal_write", "tool": "log_invoice",
            "approval": {"state": "none"}}
    return {**base, **kw}

def pre_model(**kw):
    base = {"hook": "pre_model", "pack": "ap_logging", "action_class": "read", "data_class": "internal", "model": "m-small",
            "model_region": "eu", "spend": {"run_usd": 1.0, "est_call_usd": 0.1, "run_limit_usd": 10.0}}
    return {**base, **kw}

@pytest.fixture
async def client(opa, pver):
    c = PolicyClient(base_url=opa.url, version=pver)
    yield c
    await c.close()

async def test_allow_and_block_paths(client):
    assert (await client.decide(pre_tool())).allowed
    assert (await client.decide(pre_tool(tool="delete_everything"))).reasons == ("tool_not_granted",)
    d = await client.decide(pre_tool(action_class="payment"))
    assert d.verdict == "block" and "payment_out_of_scope" in d.reasons

async def test_external_message_needs_approval_then_passes(client):
    d = await client.decide(pre_tool(tool="send_vendor_message", action_class="external_comm"))
    assert d.verdict == "require_approval"
    d = await client.decide(pre_tool(tool="send_vendor_message", action_class="external_comm", approval={"state": "approved"}))
    assert d.allowed
    d = await client.decide(pre_tool(tool="send_vendor_message", action_class="external_comm", approval={"state": "rejected"}))
    assert d.verdict == "block"

async def test_budget_residency_and_model_allow_list(client):
    assert (await client.decide(pre_model(spend={"run_usd": 9.95, "est_call_usd": 0.1, "run_limit_usd": 10.0}))).reasons == ("budget_hard_limit",)
    assert "residency" in (await client.decide(pre_model(data_class="restricted", model_region="us"))).reasons
    assert "model_not_allowed" in (await client.decide(pre_model(data_class="confidential", model="m-unknown"))).reasons

async def test_unknown_pack_fails_closed(client):
    # the spike 1 fail-open: an unknown pack must be refused, not allowed
    assert (await client.decide(pre_tool(pack="no_such_pack"))).verdict == "block"

@pytest.mark.parametrize("bad", [{}, {"hook": "nonsense", "action_class": "read"}, pre_model(spend={}), pre_tool(tool=42),
                                 {"hook": "post_tool", "action_class": "read", "output": {"bytes": "huge"}}])
async def test_malformed_input_is_blocked_before_calling_opa(client, bad):
    d = await client.decide(bad)
    assert d.verdict == "block" and d.reasons == ("malformed_input",)

async def test_modify_verdicts(client):
    d = await client.decide({"hook": "post_model", "pack": "ap_logging", "action_class": "read", "output": {"contains_pii": True}})
    assert d.verdict == "modify" and d.reasons == ("redact_pii",)

async def test_opa_down_blocks(pver):
    c = PolicyClient(base_url="http://127.0.0.1:1", version=pver, timeout_s=0.2)
    d = await c.decide(pre_tool())
    await c.close()
    assert d.verdict == "block" and d.reasons == ("policy_unavailable",)

async def test_unrecognised_answer_blocks(aiohttp_server=None):
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    async def weird(request): return web.json_response({"result": {"verdict": "maybe"}})
    async def empty(request): return web.json_response({})
    for handler, expected in ((weird, "policy_undefined"), (empty, "policy_undefined")):
        app = web.Application(); app.router.add_post("/v1/data/hooks/decision", handler)
        server = TestServer(app); await server.start_server()
        c = PolicyClient(base_url=str(server.make_url("")).rstrip("/"))
        d = await c.decide(pre_tool()); await c.close(); await server.close()
        assert d.verdict == "block" and d.reasons == (expected,)

async def test_decisions_are_logged_with_policy_version(client, pver):
    log = SQLiteEventLog()
    runner = HookRunner(client, log, "org-1")
    await runner.check("r1", "s1", pre_tool(tool="delete_everything"))
    ev = log.read("org-1", "r1")[0]
    assert ev.type == "policy.decision" and ev.payload["verdict"] == "block" and ev.payload["policy_version"] == pver
