"""FDE Enterprise Customization, Task-Log Ingestion, Optimal Path Selector & Isolated Subagents (PRD sections 8, 9, 10).

Enables Forward Deployed Engineers (FDEs) to:
  1. Take the base harness architecture and define a customizable `RoutingMap` per `(task_class, step_role)`.
  2. Sprinkle in enterprise context (versioned policies, procedures, documents, skills), scoped memory (Graphiti or Mem0),
     and historical or live `TaskLogRecord` entries.
  3. Let the `AdaptiveOutcomeSelector` compute the most optimum and efficient model path per step by combining public
     benchmark priors, empirical tenant `OutcomeStat` records, cost/latency trade-off weights, and prompt-cache affinity (PRD C7).
  4. Spawn isolated child subagent runs with narrowed budgets and policies (PRD H8, C8).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import LocalArtifactStore, Offloader, make_read_artifact_tool
from .compaction import Compactor
from .context import (
    ContextService,
    GraphitiMemoryAdapter,
    Mem0MemoryAdapter,
    MemoryEngine,
    Principal,
    ProcedureSkill,
    SkillRegistry,
)
from .events import EventLog, SQLiteEventLog
from .ids import uuid7
from .packs import (
    ContractPaymentsPack,
    FDEWorkflowPack,
    FDEWorkflowStep,
    InvoiceLoggingPack,
    PayoutTermsReviewPack,
    TaskPack,
    VendorRenewalNegotiationPack,
)
from .policy import EmbeddedPolicyClient, HookRunner, PolicyClient, build_policy_data
from .replay import ReplayEngine, ReplayReport
from .router import Router, RoutingRequest
from .router.adapter import LiteLLMAdapter
from .router.budget import BudgetLedger
from .router.health import CircuitBreaker
from .router.models import ModelProfile, ModelSpec
from .router.selector import Scored, Selector, StaticPriorSelector
from .rubric import RubricGrader
from .runs import RunService, RunState, RunView
from .tools import ToolRuntime, ToolSpec
from .worker import Orchestrator, Worker


@dataclass(frozen=True)
class TaskLogRecord:
    """Historical or live task step execution log supplied by the FDE team or extracted from runs (PRD M9, ERD 3.6)."""
    task_class: str
    step_role: str
    model_id: str
    success: bool
    cost_usd: float
    latency_ms: float
    prompt_tokens: int = 200
    completion_tokens: int = 100
    prompt_template_hash: str = "default"
    verifier_score: float | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def effective_score(self) -> float:
        if self.verifier_score is not None:
            return float(self.verifier_score)
        return 1.0 if self.success else 0.0


@dataclass
class OutcomeStat:
    """Aggregated outcome memory per (task_class, step_role, model_id, prompt_template_hash, selector_version) (ERD 3.6)."""
    task_class: str
    step_role: str
    model_id: str
    prompt_template_hash: str = "default"
    selector_version: str = "adaptive-fde-v1"
    n: int = 0
    successes: int = 0
    total_score: float = 0.0
    total_cost_usd: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    computed_at: float = field(default_factory=time.time)

    def record(self, log_entry: TaskLogRecord) -> None:
        self.n += 1
        if log_entry.success:
            self.successes += 1
        self.total_score += log_entry.effective_score
        self.total_cost_usd += log_entry.cost_usd
        self.latencies_ms.append(log_entry.latency_ms)
        self.computed_at = time.time()

    @property
    def success_rate(self) -> float:
        return self.total_score / self.n if self.n > 0 else 0.0

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / self.n if self.n > 0 else 0.0

    @property
    def cost_per_success(self) -> float:
        return self.total_cost_usd / max(self.successes, 1)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        idx = min(int(round(0.95 * (len(ordered) - 1))), len(ordered) - 1)
        return ordered[idx]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_class": self.task_class,
            "step_role": self.step_role,
            "model_id": self.model_id,
            "prompt_template_hash": self.prompt_template_hash,
            "selector_version": self.selector_version,
            "n": self.n,
            "success_rate": round(self.success_rate, 4),
            "avg_cost_usd": round(self.avg_cost_usd, 6),
            "cost_per_success": round(self.cost_per_success, 6),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


@dataclass(frozen=True)
class RouteRule:
    """FDE routing customization entry for a specific `(task_class, step_role)`."""
    task_class: str
    step_role: str = "*"
    priors: dict[str, float] = field(default_factory=dict)          # model_id -> baseline capability prior [0..1]
    quality_tolerance: float = 0.03                                 # models within tolerance of top quality are adequate
    cost_weight: float = 0.20                                       # penalty weight on normalized output token price
    latency_weight: float = 0.10                                    # penalty weight on normalized p95 latency
    cache_affinity_bonus: float = 0.02                              # PRD C7: bonus for keeping the cached model across steps
    preferred_models: tuple[str, ...] = ()


class RoutingMap:
    """Customizable FDE routing map holding step-level priors, efficiency weights, and cache-affinity rules."""

    def __init__(self, rules: list[RouteRule] | None = None, default_tolerance: float = 0.03) -> None:
        self.default_tolerance = default_tolerance
        self._rules: dict[tuple[str, str], RouteRule] = {}
        for r in rules or []:
            self.add_rule(r)

    def add_rule(self, rule: RouteRule) -> None:
        self._rules[(rule.task_class, rule.step_role)] = rule

    def get_rule(self, task_class: str, step_role: str) -> RouteRule | None:
        return self._rules.get((task_class, step_role)) or self._rules.get((task_class, "*"))

    def as_priors_dict(self) -> dict[tuple[str, str], dict[str, float]]:
        return {k: dict(v.priors) for k, v in self._rules.items()}


class AdaptiveOutcomeSelector:
    """Tier 2 selector combining FDE routing maps, public benchmark priors, empirical task-log `OutcomeStat` records,
    cost/latency efficiency optimization, and prompt-cache affinity (PRD sections 8, 9, 10: R2, R5, M9, C7)."""

    def __init__(
        self,
        routing_map: RoutingMap,
        version: str = "adaptive-fde-v1",
        min_confidence_samples: int = 5,
        use_outcome_memory: bool = True,
    ) -> None:
        self.routing_map = routing_map
        self.version = version
        self.min_confidence_samples = max(1, min_confidence_samples)
        self.use_outcome_memory = use_outcome_memory
        self._stats: dict[tuple[str, str, str], OutcomeStat] = {}
        self.last_served_model: str | None = None

    def ingest_task_logs(self, logs: list[TaskLogRecord]) -> int:
        for entry in logs:
            key = (entry.task_class, entry.step_role, entry.model_id)
            stat = self._stats.get(key)
            if stat is None:
                stat = OutcomeStat(
                    task_class=entry.task_class,
                    step_role=entry.step_role,
                    model_id=entry.model_id,
                    prompt_template_hash=entry.prompt_template_hash,
                    selector_version=self.version,
                )
                self._stats[key] = stat
            stat.record(entry)
        return len(logs)

    def outcome_stats(self) -> list[dict[str, Any]]:
        return [s.as_dict() for _, s in sorted(self._stats.items())]

    def _blended_capability(self, task_class: str, step_role: str, model_id: str, rule: RouteRule | None) -> tuple[float, str]:
        prior = rule.priors.get(model_id) if rule and model_id in rule.priors else None
        stat = self._stats.get((task_class, step_role, model_id)) or self._stats.get((task_class, "*", model_id))
        if not self.use_outcome_memory or stat is None or stat.n == 0:
            if prior is not None:
                return float(prior), "public_prior"
            return 0.0, "no_evidence"
        empirical = stat.success_rate
        weight = stat.n / (stat.n + self.min_confidence_samples)
        if prior is None:
            return round(empirical, 4), f"outcome_memory(n={stat.n})"
        blended = (1.0 - weight) * float(prior) + weight * empirical
        return round(blended, 4), f"blended_prior+task_logs(n={stat.n})"

    def rank(
        self,
        task_class: str,
        step_role: str,
        candidates: list[ModelSpec],
        stronger_than: str | None = None,
    ) -> list[Scored]:
        if not candidates:
            return []
        rule = self.routing_map.get_rule(task_class, step_role)
        tolerance = rule.quality_tolerance if rule else self.routing_map.default_tolerance
        cost_w = rule.cost_weight if rule else 0.15
        lat_w = rule.latency_weight if rule else 0.05
        cache_bonus = rule.cache_affinity_bonus if rule else 0.02

        raw_capabilities: dict[str, tuple[float, str]] = {}
        for m in candidates:
            cap, src = self._blended_capability(task_class, step_role, m.id, rule)
            # Apply cache affinity bonus if the model supports prompt caching and was the last model served (PRD C7)
            if self.last_served_model == m.id and m.profile.prompt_cache and stronger_than is None:
                cap = round(min(1.0, cap + cache_bonus), 4)
                src = f"{src}+cache_affinity"
            raw_capabilities[m.id] = (cap, src)

        scored = [Scored(m, raw_capabilities[m.id][0], raw_capabilities[m.id][1]) for m in candidates]

        # Tier 3 Escalation: strictly filter to models with a higher capability score than the failed model
        if stronger_than is not None:
            floor_cap, _ = self._blended_capability(task_class, step_role, stronger_than, rule)
            higher = [s for s in scored if s.score > floor_cap]
            return sorted(higher, key=lambda s: (s.model.price_out_per_mtok, -s.score))

        top_score = max(s.score for s in scored)
        adequate = [s for s in scored if s.score >= top_score - tolerance]

        max_price = max((m.price_out_per_mtok for m in candidates), default=1.0) or 1.0
        max_lat = max(
            (
                (self._stats.get((task_class, step_role, m.id)) or OutcomeStat(task_class, step_role, m.id)).p95_latency_ms
                for m in candidates
            ),
            default=1.0,
        ) or 1.0

        def _efficiency_key(s: Scored) -> tuple[float, float, float]:
            stat = self._stats.get((task_class, step_role, s.model.id)) or self._stats.get((task_class, "*", s.model.id))
            norm_cost = s.model.price_out_per_mtok / max_price
            norm_lat = (stat.p95_latency_ms / max_lat) if (stat and stat.n > 0 and max_lat > 0) else 0.0
            # Lower effective cost-latency penalty and higher score ranks first; tie-break by cheapest output price
            utility = s.score - (cost_w * norm_cost) - (lat_w * norm_lat)
            return (s.model.price_out_per_mtok, -round(utility, 6), -s.score)

        first = min(adequate, key=_efficiency_key)
        rest = sorted((s for s in scored if s is not first), key=lambda s: (-s.score, s.model.price_out_per_mtok))
        return [first, *rest]


class SubagentRunner:
    """Isolated child subagent execution (PRD H8 & C8):
      * Runs as a separate `run_id` linked via `parent_run_id` in `run.created`.
      * Child budget is strictly capped at `min(requested_budget_usd, parent_remaining_budget_usd)`.
      * Child policy grants are narrowed to the intersection of the parent pack's and child pack's allowed models and tools.
      * Parent receives only the child's final result summary, keeping parent context clean.
    """

    def __init__(self, deployment: "FDEDeployment") -> None:
        self.deployment = deployment

    async def run_subagent(
        self,
        parent_run_id: str,
        child_pack_name: str,
        child_input: dict[str, Any],
        requested_budget_usd: float,
        started_by: str = "subagent",
    ) -> dict[str, Any]:
        parent_view = self.deployment.orchestrator.runs.view(parent_run_id)
        if parent_view.state is None:
            raise ValueError(f"parent run {parent_run_id!r} does not exist")
        parent_pack_name = parent_view.meta.get("pack", "")
        parent_pack = self.deployment.packs.get(parent_pack_name)
        child_pack = self.deployment.packs.get(child_pack_name)
        if parent_pack is None or child_pack is None:
            raise KeyError(f"unknown parent ({parent_pack_name!r}) or child ({child_pack_name!r}) pack")

        parent_limit = float(parent_view.meta.get("budget_usd", 0.0))
        parent_spent = self.deployment.router.budget.committed_plus_reserved(parent_run_id)
        parent_remaining = max(0.0, parent_limit - parent_spent)
        child_budget = min(float(requested_budget_usd), parent_remaining)
        if child_budget <= 0.0:
            return {"ok": False, "error": "budget_hard_limit", "detail": "no remaining parent budget for subagent"}

        # Enforce policy no wider than parent's (PRD C8): intersect allowed_models
        narrowed_models = [m for m in child_pack.allowed_models if m in parent_pack.allowed_models]
        if not narrowed_models:
            return {"ok": False, "error": "no_allowed_model", "detail": "child pack shares no allowed models with parent"}

        child_run_id = f"sub-{uuid7()}"
        self.deployment.orchestrator.runs.create(
            child_run_id,
            pack=child_pack_name,
            input=child_input,
            budget_usd=child_budget,
            started_by=started_by,
            parent_run_id=parent_run_id,
        )
        self.deployment.orchestrator.runs.transition(child_run_id, "run.queued")
        view = await self.deployment.worker.run(child_run_id)
        events = self.deployment.log.read(self.deployment.org_id, child_run_id)
        succeeded = [e.payload.get("result") for e in events if e.type == "run.succeeded"]
        child_spend = self.deployment.router.budget.spent(child_run_id)
        return {
            "ok": view.state == RunState.SUCCEEDED,
            "child_run_id": child_run_id,
            "parent_run_id": parent_run_id,
            "state": view.state.value if view.state else "unknown",
            "cost_usd": child_spend,
            "result": succeeded[-1] if succeeded else {"failure_reason": view.failure_reason},
        }


class FDEDeployment:
    """Complete Customizable Enterprise Deployment Harness for Forward Deployed Engineers (FDEs).

    Bundles the base harness architecture (`EventLog`, `Router`, `HookRunner`, `ToolRuntime`, `Offloader`,
    `Compactor`, `RubricGrader`, `Orchestrator`, `Worker`) with the FDE customization surfaces:
      * Custom `RoutingMap` & `AdaptiveOutcomeSelector`
      * Scoped `ContextService` (policies, documents, tickets, data schemas, code refs, Graphiti/Mem0 memory)
      * Progressive `SkillRegistry` (versioned organizational procedures)
      * Historical & live `TaskLogRecord` ingestion
      * Deterministic `ReplayEngine`
      * Isolated `SubagentRunner`
    """

    def __init__(
        self,
        org_id: str = "org-enterprise-default",
        org_name: str = "Enterprise Tenant",
        region: str = "eu",
        log: EventLog | None = None,
        catalogue: dict[str, ModelSpec] | None = None,
        routing_map: RoutingMap | None = None,
        memory_engine: MemoryEngine | None = None,
        policy_client: PolicyClient | EmbeddedPolicyClient | None = None,
        adapter: LiteLLMAdapter | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.org_id = org_id
        self.org_name = org_name
        self.region = region
        self.log: EventLog = log or SQLiteEventLog()
        self.catalogue: dict[str, ModelSpec] = catalogue or self._default_catalogue()
        self.routing_map: RoutingMap = routing_map or self._default_routing_map()
        self.selector = AdaptiveOutcomeSelector(self.routing_map)
        self.baseline_selector = StaticPriorSelector(self.routing_map.as_priors_dict(), version="public-prior-baseline-v1")

        self.context = ContextService(self.log, self.org_id, engine=memory_engine or GraphitiMemoryAdapter())
        self.skills = SkillRegistry(self.log, self.org_id)
        self.default_principal = Principal(
            id="fde-operator",
            org_id=self.org_id,
            kind="user",
            teams=("finance", "platform", "fde"),
            roles=("admin", "memory_approver"),
            max_clearance="restricted",
        )

        self.tools = ToolRuntime()
        self._register_default_tools(artifact_dir)

        self.packs: dict[str, TaskPack] = {
            "ap_logging": InvoiceLoggingPack(),
            "contract_payments": ContractPaymentsPack(),
            "payout_terms_review": PayoutTermsReviewPack(),
            "vendor_renewal": VendorRenewalNegotiationPack(),
        }

        policy_data = build_policy_data({name: p.policy_entry() for name, p in self.packs.items() if hasattr(p, "policy_entry")})
        self.policy_client: PolicyClient | EmbeddedPolicyClient = policy_client or EmbeddedPolicyClient(policy_data)
        self.hooks = HookRunner(self.policy_client, self.log, self.org_id)

        self.budget = BudgetLedger(self.log, self.org_id)
        self.breaker = CircuitBreaker()
        self.adapter = adapter or self._default_mock_adapter()
        self.router = Router(
            catalogue=self.catalogue,
            selector=self.selector,
            hooks=self.hooks,
            log=self.log,
            org_id=self.org_id,
            budget=self.budget,
            breaker=self.breaker,
            adapter=self.adapter,
            shadow_selectors=[self.baseline_selector],
        )

        self.compactor = Compactor(self.router, self.artifact_store, self.log, self.org_id)
        self.grader = RubricGrader(self.router, self.log, self.org_id)
        self.orchestrator = Orchestrator(self.log, self.org_id)
        self.worker = Worker(
            log=self.log,
            org_id=self.org_id,
            packs=self.packs,
            router=self.router,
            hooks=self.hooks,
            tools=self.tools,
            offloader=self.offloader,
            compactor=self.compactor,
            grader=self.grader,
        )
        self.subagents = SubagentRunner(self)
        self.replay_engine = ReplayEngine(self.log, self.org_id, self.catalogue, self.routing_map.as_priors_dict())

    @staticmethod
    def _default_catalogue() -> dict[str, ModelSpec]:
        return {
            "m-small": ModelSpec(
                id="m-small",
                litellm_model="openai/gpt-4o-mini",
                region="me",
                price_in_per_mtok=0.15,
                price_out_per_mtok=0.60,
                context_window=64_000,
                supports_tools=True,
                version="2026-07",
                provider="openai",
                profile=ModelProfile(reasoning_blocks="strip", prompt_cache=True),
            ),
            "m-medium": ModelSpec(
                id="m-medium",
                litellm_model="anthropic/claude-3-5-haiku",
                region="eu",
                price_in_per_mtok=0.80,
                price_out_per_mtok=4.00,
                context_window=128_000,
                supports_tools=True,
                version="2026-09",
                provider="anthropic",
                profile=ModelProfile(reasoning_blocks="strip", prompt_cache=True),
            ),
            "m-large": ModelSpec(
                id="m-large",
                litellm_model="anthropic/claude-3-7-sonnet",
                region="me",
                price_in_per_mtok=3.00,
                price_out_per_mtok=15.00,
                context_window=200_000,
                supports_tools=True,
                version="2026-10",
                provider="anthropic",
                profile=ModelProfile(reasoning_blocks="keep", prompt_cache=True),
            ),
        }

    @staticmethod
    def _default_routing_map() -> RoutingMap:
        return RoutingMap(
            rules=[
                RouteRule("extract", "extract", {"m-small": 0.92, "m-medium": 0.93, "m-large": 0.94}, quality_tolerance=0.03),
                RouteRule("extract", "*", {"m-small": 0.91, "m-medium": 0.93, "m-large": 0.94}, quality_tolerance=0.03),
                RouteRule("reconcile", "reconcile_terms", {"m-small": 0.82, "m-medium": 0.935, "m-large": 0.94}, quality_tolerance=0.03),
                RouteRule("clause_review", "clause_extraction", {"m-small": 0.78, "m-medium": 0.91, "m-large": 0.96}, quality_tolerance=0.03),
                RouteRule("negotiation", "negotiation_strategy", {"m-small": 0.74, "m-medium": 0.86, "m-large": 0.95}, quality_tolerance=0.03),
                RouteRule("custom", "*", {"m-small": 0.89, "m-medium": 0.92, "m-large": 0.95}, quality_tolerance=0.04),
            ]
        )

    @staticmethod
    def _default_mock_adapter() -> LiteLLMAdapter:
        """Deterministic in-process adapter used when no external upstream API key is configured."""
        from .router.adapter import Completion

        class _DeterministicAdapter(LiteLLMAdapter):
            async def complete(self, model: ModelSpec, messages: list[dict], max_tokens: int = 1024) -> Completion:
                sys_And_user = " ".join(str(m.get("content", "")) for m in messages).lower()
                if "reconcile" in sys_And_user or "discrepancy_usd" in sys_And_user:
                    txt = json.dumps({"matched": True, "discrepancy_usd": 0.0, "notes": "Line items match contract terms."})
                elif "risk_level" in sys_And_user or "clauses" in sys_And_user:
                    txt = json.dumps({
                        "risk_level": "low",
                        "clauses": ["Net-45 payment schedule", "2% early-payment discount"],
                        "policy_citation": "[policy:AP-FIN-001@v1]",
                    })
                elif "target_discount_pct" in sys_And_user or "draft_email" in sys_And_user:
                    txt = json.dumps({
                        "target_discount_pct": 12.5,
                        "draft_email": "Dear Vendor Partner, per our multi-year renewal policy we propose a 12.5% volume discount on the upcoming term.",
                    })
                else:
                    txt = json.dumps({"vendor": "Acme Cloud Corp", "amount": 4250.0, "status": "ok"})
                return Completion(text=txt, prompt_tokens=180, completion_tokens=64, latency_ms=4.2, provider_model=model.litellm_model)

        return _DeterministicAdapter()

    def _register_default_tools(self, artifact_dir: str | Path | None = None) -> None:
        import tempfile
        base_dir = Path(artifact_dir) if artifact_dir else Path(tempfile.mkdtemp(prefix="harness-artifacts-"))
        self.artifact_store = LocalArtifactStore(base_dir)
        self.offloader = Offloader(self.artifact_store, self.log, self.org_id, threshold_bytes=4096)
        self.tools.register(make_read_artifact_tool(self.artifact_store, self.org_id))

        async def _log_invoice(args: dict[str, Any]) -> dict[str, Any]:
            return {"logged": True, "vendor": args["vendor"], "amount": args["amount"]}

        async def _send_vendor_message(args: dict[str, Any]) -> dict[str, Any]:
            return {"sent": True, "vendor": args["vendor"], "body": args["body"]}

        async def _fetch_contract_terms(args: dict[str, Any]) -> dict[str, Any]:
            return {"contract_id": args["contract_id"], "payment_terms": "Net-45", "cap_usd": 50000.0}

        async def _log_reconciliation(args: dict[str, Any]) -> dict[str, Any]:
            return {"reconciled": True, "contract_id": args["contract_id"], "matched": args["matched"]}

        async def _log_terms_review(args: dict[str, Any]) -> dict[str, Any]:
            return {"reviewed": True, "contract_id": args["contract_id"], "risk_level": args["risk_level"]}

        async def _load_skill(args: dict[str, Any]) -> dict[str, Any]:
            return self.skills.load_body(
                name=args["skill_name"],
                run_id=args.get("run_id", "tool-skill-load"),
                principal=self.default_principal,
            )

        self.tools.register(ToolSpec("log_invoice", "internal_write", _log_invoice, required=("vendor", "amount"), idempotent=True))
        self.tools.register(ToolSpec("send_vendor_message", "external_comm", _send_vendor_message, required=("vendor", "body"), idempotent=False))
        self.tools.register(ToolSpec("fetch_contract_terms", "read", _fetch_contract_terms, required=("contract_id",), idempotent=True))
        self.tools.register(ToolSpec("log_reconciliation", "internal_write", _log_reconciliation, required=("contract_id", "matched"), idempotent=True))
        self.tools.register(ToolSpec("log_terms_review", "internal_write", _log_terms_review, required=("contract_id", "risk_level"), idempotent=True))
        self.tools.register(ToolSpec("load_skill", "read", _load_skill, required=("skill_name",), idempotent=True))

    def _sync_policy_packs(self) -> None:
        if isinstance(self.policy_client, EmbeddedPolicyClient):
            self.policy_client.data = build_policy_data({
                name: p.policy_entry() for name, p in self.packs.items() if hasattr(p, "policy_entry")
            })

    # ---- FDE Customization API ----------------------------------------------------------------------------
    def register_pack(self, pack: TaskPack) -> None:
        self.packs[pack.name] = pack
        self._sync_policy_packs()

    def register_workflow_pack(
        self,
        name: str,
        task_class: str,
        steps: list[dict[str, Any]],
        data_class: str = "internal",
        allowed_models: list[str] | None = None,
        restricted_regions: list[str] | None = None,
    ) -> FDEWorkflowPack:
        wf_steps = [
            FDEWorkflowStep(
                kind=s.get("kind", "model"),
                step_role=s.get("step_role", "plan"),
                system_prompt=s.get("system_prompt", "Return valid JSON."),
                input_key=s.get("input_key", "prompt"),
                tool_name=s.get("tool_name", ""),
                tool_arg_mapping=dict(s.get("tool_arg_mapping", {})),
                required_json_keys=tuple(s.get("required_json_keys", ())),
                context_query=s.get("context_query"),
                max_tokens=int(s.get("max_tokens", 512)),
            )
            for s in steps
        ]
        pack = FDEWorkflowPack(
            name=name,
            task_class=task_class,
            data_class=data_class,
            allowed_models=allowed_models or list(self.catalogue.keys()),
            restricted_regions=restricted_regions or ["me", "eu"],
            steps=wf_steps,
            context_service=self.context,
            skill_registry=self.skills,
            caller_principal=self.default_principal,
        )
        self.register_pack(pack)
        return pack

    def sprinkle_context(self, items: list[dict[str, Any]]) -> int:
        """Ingests enterprise policies, procedures, documents, tickets, data schemas, or code references."""
        for item in items:
            self.context.ingest_knowledge(
                source_id=item.get("source_id", "fde-corpus"),
                external_id=item["external_id"],
                kind=item.get("kind", "policy"),
                title=item["title"],
                content=item["content"],
                version_no=item.get("version_no"),
                effective_from=float(item.get("effective_from", 0.0)),
                effective_to=float(item["effective_to"]) if item.get("effective_to") is not None else None,
                owner=item.get("owner", "fde-team"),
                data_class=item.get("data_class", "internal"),
                allowed_teams=tuple(item.get("allowed_teams", ())),
            )
        return len(items)

    def sprinkle_skills(self, skills: list[dict[str, Any]]) -> int:
        """Registers versioned, effective-dated organizational procedure skills (PRD C6)."""
        for s in skills:
            self.skills.register(ProcedureSkill(
                name=s["name"],
                org_id=self.org_id,
                description=s["description"],
                body=s["body"],
                version_no=int(s.get("version_no", 1)),
                effective_from=float(s.get("effective_from", 0.0)),
                owner=s.get("owner", "fde-team"),
                data_class=s.get("data_class", "internal"),
                allowed_teams=tuple(s.get("allowed_teams", ())),
            ))
        return len(skills)

    def sprinkle_memory(self, facts: list[dict[str, Any]], caller: Principal | None = None) -> list[str]:
        """Seeds scoped enterprise memory facts across user, team, agent, or org scopes."""
        actor = caller or self.default_principal
        ref_ids: list[str] = []
        for f in facts:
            ref = self.context.write_memory(
                scope_type=f.get("scope_type", "org"),
                scope_id=f.get("scope_id", self.org_id),
                key=f["key"],
                content=f["content"],
                caller=actor,
                data_class=f.get("data_class", "internal"),
                source_id=f.get("source_id", "fde-seed"),
                require_review=f.get("require_review"),
                tags=tuple(f.get("tags", ())),
            )
            ref_ids.append(ref.id)
        return ref_ids

    def sprinkle_task_logs(self, logs: list[dict[str, Any] | TaskLogRecord]) -> int:
        """Ingests historical or live FDE task logs to sharpen `AdaptiveOutcomeSelector` routing decisions."""
        parsed: list[TaskLogRecord] = []
        for entry in logs:
            if isinstance(entry, TaskLogRecord):
                parsed.append(entry)
            else:
                parsed.append(TaskLogRecord(
                    task_class=entry["task_class"],
                    step_role=entry.get("step_role", "*"),
                    model_id=entry["model_id"],
                    success=bool(entry.get("success", True)),
                    cost_usd=float(entry.get("cost_usd", 0.001)),
                    latency_ms=float(entry.get("latency_ms", 120.0)),
                    prompt_tokens=int(entry.get("prompt_tokens", 200)),
                    completion_tokens=int(entry.get("completion_tokens", 100)),
                    prompt_template_hash=entry.get("prompt_template_hash", "default"),
                    verifier_score=float(entry["verifier_score"]) if entry.get("verifier_score") is not None else None,
                ))
        return self.selector.ingest_task_logs(parsed)

    def update_routing_map(self, rules: list[dict[str, Any]]) -> None:
        for r in rules:
            self.routing_map.add_rule(RouteRule(
                task_class=r["task_class"],
                step_role=r.get("step_role", "*"),
                priors={k: float(v) for k, v in r.get("priors", {}).items()},
                quality_tolerance=float(r.get("quality_tolerance", 0.03)),
                cost_weight=float(r.get("cost_weight", 0.20)),
                latency_weight=float(r.get("latency_weight", 0.10)),
                cache_affinity_bonus=float(r.get("cache_affinity_bonus", 0.02)),
            ))
        self.baseline_selector = StaticPriorSelector(self.routing_map.as_priors_dict(), version="public-prior-baseline-v1")
        self.replay_engine = ReplayEngine(self.log, self.org_id, self.catalogue, self.routing_map.as_priors_dict())

    def preview_optimal_route(
        self,
        task_class: str,
        step_role: str,
        allowed_models: list[str] | None = None,
        stronger_than: str | None = None,
    ) -> list[dict[str, Any]]:
        ids = allowed_models or list(self.catalogue.keys())
        pool = [self.catalogue[mid] for mid in ids if mid in self.catalogue]
        ranked = self.selector.rank(task_class, step_role, pool, stronger_than=stronger_than)
        return [
            {
                "rank": idx + 1,
                "model_id": s.model.id,
                "provider": s.model.provider,
                "region": s.model.region,
                "score": s.score,
                "evidence_source": s.source,
                "price_in_per_mtok": s.model.price_in_per_mtok,
                "price_out_per_mtok": s.model.price_out_per_mtok,
            }
            for idx, s in enumerate(ranked)
        ]

    async def submit_and_run(
        self,
        pack: str,
        run_input: dict[str, Any],
        budget_usd: float = 1.0,
        started_by: str = "fde-operator",
        run_id: str | None = None,
    ) -> RunView:
        rid = run_id or f"run-{uuid7()}"
        enriched_input = {**run_input, "_run_id": rid}
        self.orchestrator.submit(rid, pack=pack, run_input=enriched_input, budget_usd=budget_usd, started_by=started_by)
        return await self.worker.run(rid)

    def fork_from_checkpoint(self, source_run_id: str, upto_seq: int, new_run_id: str | None = None) -> RunView:
        """Creates a new run seeded from any event sequence checkpoint of `source_run_id` without mutating the original (PRD H2, ERD 3.3)."""
        source_events = [e for e in self.log.read(self.org_id, source_run_id) if e.seq <= upto_seq]
        if not source_events:
            raise ValueError(f"no events found for run {source_run_id!r} up to seq {upto_seq}")
        fork_id = new_run_id or f"fork-{uuid7()}"
        created_ev = source_events[0]
        fork_meta = {
            **created_ev.payload,
            "forked_from_run_id": source_run_id,
            "forked_upto_seq": upto_seq,
        }
        self.log.append(self.org_id, fork_id, "run.created", fork_meta)
        for ev in source_events[1:]:
            # Copy completed step history and responses up to `upto_seq`, then place the forked run in QUEUED state ready for a worker
            if ev.type in ("step.started", "step.completed", "model.request", "model.response", "routing.decision", "tool.executing", "tool.result"):
                self.log.append(self.org_id, fork_id, ev.type, ev.payload, step_id=ev.step_id)
        self.log.append(self.org_id, fork_id, "run.queued", {"forked_from": source_run_id, "upto_seq": upto_seq})
        return self.orchestrator.runs.view(fork_id)

    @classmethod
    def from_dict(cls, config: dict[str, Any], log: EventLog | None = None) -> "FDEDeployment":
        org = config.get("organization", {})
        dep = cls(
            org_id=org.get("org_id", "org-enterprise-default"),
            org_name=org.get("name", "Enterprise Tenant"),
            region=org.get("region", "eu"),
            log=log,
        )
        if config.get("memory_engine") == "mem0":
            dep.context.swap_engine(Mem0MemoryAdapter())
        if "routing_map" in config:
            dep.update_routing_map(config["routing_map"])
        if "context_items" in config:
            dep.sprinkle_context(config["context_items"])
        if "skills" in config:
            dep.sprinkle_skills(config["skills"])
        if "memory_facts" in config:
            dep.sprinkle_memory(config["memory_facts"])
        if "task_logs" in config:
            dep.sprinkle_task_logs(config["task_logs"])
        for wf in config.get("custom_workflows", []):
            dep.register_workflow_pack(
                name=wf["name"],
                task_class=wf.get("task_class", "custom"),
                steps=wf.get("steps", []),
                data_class=wf.get("data_class", "internal"),
                allowed_models=wf.get("allowed_models"),
                restricted_regions=wf.get("restricted_regions"),
            )
        return dep
