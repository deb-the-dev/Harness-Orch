import pytest

from harness.events import SQLiteEventLog
from harness.policy import HookRunner, PolicyClient
from harness.router import ModelSpec, Router, RouterError, RoutingRequest
from harness.router.health import CircuitBreaker
from harness.router.selector import StaticPriorSelector
from mock_provider import MockProvider

ORG, RUN = "org-1", "run-1"
PRIORS = {("extract", "*"): {"m-small": 0.88, "m-large": 0.90, "m-disallowed": 0.95}}


@pytest.fixture
async def env(opa, pver):
    mock = await MockProvider().start()
    log = SQLiteEventLog()
    client = PolicyClient(base_url=opa.url, version=pver)
    clock = {"t": 0.0}
    breaker = CircuitBreaker(threshold=2, cooldown_s=10, clock=lambda: clock["t"])
    cat = {
        "m-small": ModelSpec("m-small", "openai/small", "eu", 0.5, 1.5, api_base=mock.base("small"), api_key="k", version="v1"),
        "m-large": ModelSpec("m-large", "openai/large", "eu", 5.0, 15.0, api_base=mock.base("large"), api_key="k", version="v1"),
        "m-disallowed": ModelSpec("m-disallowed", "openai/dis", "eu", 0.1, 0.3, api_base=mock.base("disallowed"), api_key="k"),
    }
    router = Router(cat, StaticPriorSelector(PRIORS), HookRunner(client, log, ORG), log, ORG, breaker=breaker)
    yield type("Env", (), dict(mock=mock, log=log, router=router, clock=clock, cat=cat, client=client))
    await client.close()
    await mock.stop()


def request(**kw):
    base = dict(org_id=ORG, run_id=RUN, step_id="s1", pack="ap_logging", step_role="extract", task_class="extract", data_class="internal",
                messages=[{"role": "user", "content": "extract the invoice fields"}], allowed_models=["m-small", "m-large"], run_limit_usd=1.0, max_tokens=50)
    return RoutingRequest(**{**base, **kw})


def types(env):
    return [e.type for e in env.log.read(ORG, RUN)]


async def test_cheapest_adequate_model_is_chosen_and_everything_is_logged(env):
    r = await env.router.complete(request())
    assert r.model_id == "m-small" and r.fallback_from is None
    assert r.cost_usd == pytest.approx(12 * 0.5e-6 + 5 * 1.5e-6)       # cost from usage and catalogue prices
    assert types(env) == ["model.request", "policy.decision", "model.response", "routing.decision"]
    dec = [e for e in env.log.read(ORG, RUN) if e.type == "routing.decision"][0].payload
    assert dec["chosen"] == "m-small" and dec["tier"] == 2 and not dec["shadow"]
    assert env.mock.hits("disallowed") == 0                            # the allowed set is the agent version's, not the catalogue


async def test_failover_stays_inside_the_allowed_set(env):
    env.mock.set("small", "500")
    r = await env.router.complete(request())
    assert r.model_id == "m-large" and r.fallback_from == "m-small"
    assert env.mock.hits("disallowed") == 0
    assert "model.attempt_failed" in types(env)                         # the failure is a record, not hidden


async def test_all_failed_is_an_error_and_never_widens_the_set(env):
    env.mock.set("small", "500"); env.mock.set("large", "429")
    with pytest.raises(RouterError) as e:
        await env.router.complete(request())
    assert e.value.reason == "all_failed" and env.mock.hits("disallowed") == 0


async def test_fatal_errors_are_not_failed_over(env):
    env.mock.set("small", "400")
    with pytest.raises(RouterError) as e:
        await env.router.complete(request())
    assert e.value.reason == "provider_fatal" and env.mock.hits("large") == 0


async def test_circuit_breaker_stops_calling_a_failing_model_then_probes_again(env):
    env.mock.set("small", "500")
    for _ in range(2):                                                  # threshold is 2 failures
        await env.router.complete(request())
    assert env.mock.hits("small") == 2
    for _ in range(3):
        r = await env.router.complete(request())
        assert r.model_id == "m-large"
    assert env.mock.hits("small") == 2                                  # breaker open: no more calls to the failing model
    env.clock["t"] = 11.0                                               # cooldown over: one trial request
    env.mock.set("small", "ok")
    assert (await env.router.complete(request())).model_id == "m-small"
    assert env.router.breaker.state("m-small") == "closed"


async def test_policy_filters_candidates_and_empty_set_fails_closed(env):
    # restricted data may only go to the pack's restricted regions (["me"]); both models are in "eu"
    with pytest.raises(RouterError) as e:
        await env.router.complete(request(data_class="restricted"))
    assert e.value.reason == "no_allowed_model"
    assert env.mock.hits("small") == env.mock.hits("large") == 0       # nothing was sent anywhere
    dec = [x for x in env.log.read(ORG, RUN) if x.type == "routing.decision"][0].payload
    assert all("residency" in c["filtered_by"] for c in dec["candidates"])


async def test_model_not_in_pack_allow_list_is_blocked_by_policy(env):
    with pytest.raises(RouterError):
        await env.router.complete(request(data_class="confidential", allowed_models=["m-disallowed"]))
    assert env.mock.hits("disallowed") == 0


async def test_budget_reservation_blocks_before_any_call(env):
    with pytest.raises(RouterError) as e:
        await env.router.complete(request(run_limit_usd=0.00001))
    assert e.value.reason == "budget_hard_limit" and env.mock.hits("small") == 0


async def test_spend_accumulates_from_the_log_and_stops_the_run(env):
    one = (await env.router.complete(request(run_limit_usd=0.0005))).cost_usd
    assert env.router.budget.spent(RUN) == pytest.approx(one)
    with pytest.raises(RouterError) as e:                               # a worst-case call no longer fits
        await env.router.complete(request(run_limit_usd=one + 0.00001, max_tokens=500))
    assert e.value.reason == "budget_hard_limit"


async def test_shadow_selector_is_logged_and_never_served(env):
    env.router.shadow_selectors = [StaticPriorSelector({("extract", "*"): {"m-large": 0.99, "m-small": 0.5}}, version="candidate-v2")]
    r = await env.router.complete(request())
    assert r.model_id == "m-small"                                      # live policy unchanged
    shadows = [e.payload for e in env.log.read(ORG, RUN) if e.type == "routing.decision" and e.payload["shadow"]]
    assert shadows[0]["chosen"] == "m-large" and shadows[0]["selector_version"] == "candidate-v2"
    assert env.mock.hits("large") == 0


async def test_escalation_goes_to_a_stronger_allowed_model_and_is_capped(env):
    first = await env.router.complete(request())
    second = await env.router.escalate(first.request, first)
    assert second.model_id == "m-large" and second.tier == 3
    with pytest.raises(RouterError) as e:
        await env.router.escalate(second.request, second)
    assert e.value.reason == "escalation_cap"


async def test_requests_are_rebuildable_from_the_log(env):
    await env.router.complete(request())
    ev = [e for e in env.log.read(ORG, RUN) if e.type == "model.request"][0]
    assert ev.payload["messages"] == request().messages and ev.payload["allowed_models"] == ["m-small", "m-large"]
    assert env.log.verify(ORG, RUN) == []


async def test_router_sends_portable_history_and_logs_what_it_changed(env):
    thinking = {"type": "thinking", "thinking": "private", "signature": "s"}
    env.cat["m-small"] = ModelSpec("m-small", "openai/small", "eu", 0.5, 1.5, api_base=env.mock.base("small"), api_key="k", provider="openai")
    env.router.catalogue = env.cat
    msgs = [{"role": "user", "content": "extract"}, {"role": "assistant", "content": [thinking, {"type": "text", "text": "draft"}], "x_provider": "anthropic"},
            {"role": "user", "content": "now finalise"}]
    await env.router.complete(request(messages=msgs))
    sent = env.mock.requests("small")[0]["messages"]
    assert not any(isinstance(m.get("content"), list) and any(b.get("type") == "thinking" for b in m["content"]) for m in sent)
    assert not any("x_provider" in m for m in sent)
    dec = [e for e in env.log.read(ORG, RUN) if e.type == "routing.decision"][0].payload
    row = [c for c in dec["candidates"] if c["model"] == "m-small"][0]
    assert row["prepared"]["stripped_blocks"] == 1 and len(row["prepared"]["sent_hash"]) == 16
    req_ev = [e for e in env.log.read(ORG, RUN) if e.type == "model.request"][0]
    assert req_ev.payload["messages"] == msgs                                           # the log keeps the original, the report says what changed


async def test_excluded_models_are_not_used_unless_nothing_else_is_left(env):
    r = await env.router.complete(request(exclude_models=["m-small"]))
    assert r.model_id == "m-large" and env.mock.hits("small") == 0
    with pytest.raises(RouterError) as e:
        await env.router.complete(request(run_id="run-2", exclude_models=["m-small", "m-large"]))
    assert e.value.reason == "no_allowed_model"
