"""The router: the only path to a provider. Tier 1 constraints (policy, health, capability), tier 2 selector,
tier 3 escalation, with every choice and every failure written to the event log.

Design notes
  * Tier 1 policy rules live only in OPA. The router asks the policy engine once per candidate model (the pre_model hook
    needs the model and its region), so there is one source of truth for what is allowed.
  * Failover only moves through the ranked, allowed candidates. It never widens the allowed set.
  * Reserve spend before the call, reconcile after (budget.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from ..events import EventLog
from ..history import prepare_messages
from ..ids import uuid7
from ..policy import HookRunner
from .adapter import LiteLLMAdapter, ProviderFailover, ProviderFatal
from .budget import BudgetExceeded, BudgetLedger
from .health import CircuitBreaker
from .models import ModelSpec
from .selector import Scored, Selector


@dataclass
class RoutingRequest:
    org_id: str
    run_id: str
    step_id: str
    pack: str
    step_role: str
    task_class: str
    data_class: str
    messages: list[dict]
    allowed_models: list[str]            # the agent version's model roles: the allowed set before policy
    run_limit_usd: float
    max_tokens: int = 1024
    needs_tools: bool = False
    stronger_than: str | None = None     # set by escalation
    escalated_from: str | None = None    # decision id being escalated
    escalation_count: int = 0
    max_escalations: int = 1
    exclude_models: list[str] = field(default_factory=list)   # for example the grader avoids the model that wrote the answer


@dataclass
class RouterResult:
    text: str
    model_id: str
    decision_id: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    fallback_from: str | None = None
    tier: int = 2
    request: "RoutingRequest | None" = None     # the request as served, so a caller can escalate again


class RouterError(Exception):
    def __init__(self, reason: str, decision_id: str, detail: str = ""):
        super().__init__(f"{reason} {detail}".strip())
        self.reason, self.decision_id, self.detail = reason, decision_id, detail


def _est_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages) // 4 + 1


class Router:
    def __init__(self, catalogue: dict[str, ModelSpec], selector: Selector, hooks: HookRunner, log: EventLog, org_id: str,
                 budget: BudgetLedger | None = None, breaker: CircuitBreaker | None = None, adapter: LiteLLMAdapter | None = None,
                 shadow_selectors: list[Selector] | None = None):
        self.catalogue, self.selector, self.hooks, self.log, self.org_id = catalogue, selector, hooks, log, org_id
        self.budget = budget or BudgetLedger(log, org_id)
        self.breaker = breaker or CircuitBreaker()
        self.adapter = adapter or LiteLLMAdapter()
        self.shadow_selectors = shadow_selectors or []

    # ---- public -------------------------------------------------------------------------------------------
    async def complete(self, req: RoutingRequest) -> RouterResult:
        decision_id = uuid7()
        self.log.append(self.org_id, req.run_id, "model.request", {
            "decision_id": decision_id, "step_role": req.step_role, "task_class": req.task_class, "data_class": req.data_class,
            "messages": req.messages, "allowed_models": req.allowed_models, "max_tokens": req.max_tokens,
            "escalated_from": req.escalated_from}, step_id=req.step_id)

        candidates: dict[str, dict[str, Any]] = {}      # model id -> audit row
        pool: list[ModelSpec] = []
        est_in = _est_tokens(req.messages)
        for mid in req.allowed_models:
            spec = self.catalogue.get(mid)
            if spec is None:
                continue
            why = None
            if est_in + req.max_tokens > spec.context_window:
                why = "context_window"
            elif req.needs_tools and not spec.supports_tools:
                why = "no_tool_support"
            elif mid in req.exclude_models:
                why = "excluded"
            elif self.breaker.state(mid) == "open":
                why = "circuit_open"
            candidates[mid] = {"model": mid, "score": None, "source": None, "filtered_by": [why] if why else None, "attempted": False, "outcome": None}
            if not why:
                pool.append(spec)

        ranked = self.selector.rank(req.task_class, req.step_role, pool, req.stronger_than)
        for s in ranked:
            candidates[s.model.id].update(score=s.score, source=s.source)
        self._log_shadow(req, decision_id, pool)

        tier = 3 if req.escalated_from else (1 if len(ranked) == 1 else 2)
        first_id = ranked[0].model.id if ranked else None
        attempted = 0
        for s in ranked:
            m, row = s.model, candidates[s.model.id]
            est_cost = m.cost(est_in, req.max_tokens)       # worst case: the full output budget
            hook_in = {"hook": "pre_model", "pack": req.pack, "action_class": "read", "data_class": req.data_class, "model": m.id,
                       "model_region": m.region, "spend": {"run_usd": self.budget.committed_plus_reserved(req.run_id),
                                                           "est_call_usd": est_cost, "run_limit_usd": req.run_limit_usd}}
            d = await self.hooks.check(req.run_id, req.step_id, hook_in)
            if not d.allowed:
                row["filtered_by"] = list(d.reasons) or [d.verdict]
                continue
            if not self.breaker.allow(m.id):
                row["filtered_by"] = ["circuit_open"]
                continue
            try:
                resv = self.budget.reserve(req.run_id, est_cost, req.run_limit_usd)
            except BudgetExceeded:
                row["filtered_by"] = ["budget_hard_limit"]
                continue
            attempted += 1
            row["attempted"] = True
            prepared, report = prepare_messages(req.messages, m)       # portable history and the model's profile (PRD C2, C3)
            row["prepared"] = report.as_dict()
            try:
                c = await self.adapter.complete(m, prepared, req.max_tokens)
            except ProviderFailover as e:
                self.breaker.record_failure(m.id)
                self.budget.release(resv)
                row["outcome"] = f"failed:{e.kind}"
                self.log.append(self.org_id, req.run_id, "model.attempt_failed", {"decision_id": decision_id, "model": m.id, "kind": e.kind,
                                                                                "detail": e.detail}, step_id=req.step_id)
                continue
            except ProviderFatal as e:
                self.budget.release(resv)
                row["outcome"] = f"fatal:{e.kind}"
                self.log.append(self.org_id, req.run_id, "model.attempt_failed", {"decision_id": decision_id, "model": m.id, "kind": e.kind,
                                                                                "detail": e.detail, "fatal": True}, step_id=req.step_id)
                self._record(req, decision_id, list(candidates.values()), None, first_id, tier, f"provider_fatal:{e.kind}")
                raise RouterError("provider_fatal", decision_id, e.kind) from e
            cost = m.cost(c.prompt_tokens, c.completion_tokens)
            self.log.append(self.org_id, req.run_id, "model.response", {
                "decision_id": decision_id, "model": m.id, "model_version": m.version, "text": c.text, "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens, "cost_usd": cost, "latency_ms": round(c.latency_ms, 2)}, step_id=req.step_id)
            self.budget.release(resv)                        # the response event now carries the spend
            self.breaker.record_success(m.id)
            row["outcome"] = "served"
            fb = first_id if first_id != m.id else None
            self._record(req, decision_id, list(candidates.values()), m.id, first_id, tier, None, fb)
            return RouterResult(c.text, m.id, decision_id, c.prompt_tokens, c.completion_tokens, cost, fb, tier, req)

        reason = self._failure_reason(candidates, attempted)
        self._record(req, decision_id, list(candidates.values()), None, first_id, tier, reason)
        raise RouterError(reason, decision_id)

    async def escalate(self, req: RoutingRequest, prev: RouterResult) -> RouterResult:
        """Tier 3: retry on a stronger allowed model after a verifier failed the previous answer. Capped."""
        if req.escalation_count >= req.max_escalations:
            did = uuid7()
            self._record(req, did, [], None, None, 3, "escalation_cap")
            raise RouterError("escalation_cap", did)
        return await self.complete(replace(req, stronger_than=prev.model_id, escalated_from=prev.decision_id,
                                           escalation_count=req.escalation_count + 1))

    # ---- internals ----------------------------------------------------------------------------------------
    @staticmethod
    def _failure_reason(candidates: dict[str, dict], attempted: int) -> str:
        rows = list(candidates.values())
        if attempted:
            return "all_failed"
        if any(r["filtered_by"] and "budget_hard_limit" in r["filtered_by"] for r in rows):
            return "budget_hard_limit"
        return "no_allowed_model"                            # fail closed: an empty allowed set is an error, not a fallback

    def _log_shadow(self, req: RoutingRequest, decision_id: str, pool: list[ModelSpec]) -> None:
        """Candidate policies are scored and logged, never served (PRD R6)."""
        for sel in self.shadow_selectors:
            ranked = sel.rank(req.task_class, req.step_role, pool, req.stronger_than)
            self.log.append(self.org_id, req.run_id, "routing.decision", {
                "decision_id": uuid7(), "shadow": True, "shadow_of": decision_id, "selector_version": sel.version,
                "chosen": ranked[0].model.id if ranked else None,
                "candidates": [{"model": s.model.id, "score": s.score, "source": s.source} for s in ranked]}, step_id=req.step_id)

    def _record(self, req, decision_id, candidates, chosen, first_id, tier, error, fallback_from=None) -> None:
        self.log.append(self.org_id, req.run_id, "routing.decision", {
            "decision_id": decision_id, "shadow": False, "step_role": req.step_role, "task_class": req.task_class, "tier": tier,
            "selector_version": self.selector.version, "candidates": candidates, "chosen": chosen, "fallback_from": fallback_from,
            "escalated_from": req.escalated_from, "error": error}, step_id=req.step_id)
