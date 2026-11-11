"""FastAPI / HTTP Control-Plane & Router Service (PRD R1, P8, H2, H6, H8; Architecture section 2).

Exposes:
  * Control Plane & Admin API (`/v1/runs`, `/v1/fde/deployments`, `/v1/replay`, `/v1/admin/kill-switch`, `/v1/metrics`)
  * OpenAI-compatible (`/v1/chat/completions`) and Anthropic-compatible (`/v1/messages`) model router endpoints
    for external harnesses (Claude Agent SDK, OpenHands, Goose).
"""
from __future__ import annotations

import time
from typing import Any

from .context import Principal
from .fde import FDEDeployment
from .ids import uuid7
from .policy import EmbeddedPolicyClient
from .router import RouterError, RoutingRequest
from .runs import InvalidTransition


class ControlPlaneManager:
    """Manages multi-tenant `FDEDeployment` instances for the HTTP control plane."""

    def __init__(self) -> None:
        self.deployments: dict[str, FDEDeployment] = {}
        default_dep = FDEDeployment(org_id="org-default", org_name="Default Enterprise Org")
        self.deployments[default_dep.org_id] = default_dep

    def get_or_create(self, org_id: str) -> FDEDeployment:
        dep = self.deployments.get(org_id)
        if dep is None:
            dep = FDEDeployment(org_id=org_id, org_name=f"Enterprise ({org_id})")
            self.deployments[org_id] = dep
        return dep

    def configure_from_dict(self, config: dict[str, Any]) -> FDEDeployment:
        org_id = config.get("organization", {}).get("org_id", "org-default")
        existing = self.deployments.get(org_id)
        dep = FDEDeployment.from_dict(config, log=existing.log if existing else None)
        self.deployments[org_id] = dep
        return dep


def create_app(manager: ControlPlaneManager | None = None) -> Any:
    """Creates and configures the FastAPI Control-Plane and Router application."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    mgr = manager or ControlPlaneManager()
    app = FastAPI(
        title="Agent Control Harness & Model Routing Layer",
        version="0.1.0",
        description="Enterprise control plane, step-aware model router, FDE customization layer, and replay engine.",
    )
    app.state.manager = mgr

    class DeploymentConfigPayload(BaseModel):
        organization: dict[str, Any] = Field(default_factory=lambda: {"org_id": "org-default", "name": "Enterprise Org", "region": "eu"})
        memory_engine: str = "graphiti"
        routing_map: list[dict[str, Any]] = Field(default_factory=list)
        context_items: list[dict[str, Any]] = Field(default_factory=list)
        skills: list[dict[str, Any]] = Field(default_factory=list)
        memory_facts: list[dict[str, Any]] = Field(default_factory=list)
        task_logs: list[dict[str, Any]] = Field(default_factory=list)
        custom_workflows: list[dict[str, Any]] = Field(default_factory=list)

    class SprinkleContextPayload(BaseModel):
        items: list[dict[str, Any]] = Field(default_factory=list)
        skills: list[dict[str, Any]] = Field(default_factory=list)

    class SprinkleMemoryPayload(BaseModel):
        facts: list[dict[str, Any]] = Field(default_factory=list)

    class SprinkleTaskLogsPayload(BaseModel):
        logs: list[dict[str, Any]] = Field(default_factory=list)

    class ContextQueryPayload(BaseModel):
        query: str
        run_id: str = "api-context-query"
        step_id: str = "step-0"
        principal_id: str = "fde-operator"
        teams: list[str] = Field(default_factory=lambda: ["finance", "fde"])
        max_clearance: str = "restricted"
        limit: int = 6

    class RoutePreviewPayload(BaseModel):
        task_class: str
        step_role: str = "*"
        allowed_models: list[str] | None = None
        stronger_than: str | None = None

    class SubmitRunPayload(BaseModel):
        org_id: str = "org-default"
        run_id: str | None = None
        pack: str = "ap_logging"
        input: dict[str, Any] = Field(default_factory=dict)
        budget_usd: float = 1.0
        started_by: str = "fde-operator"
        auto_execute: bool = True

    class ApprovalDecisionPayload(BaseModel):
        org_id: str = "org-default"
        approved: bool
        approver: str = "compliance-officer"
        auto_resume: bool = True

    class ForkRunPayload(BaseModel):
        org_id: str = "org-default"
        upto_seq: int
        new_run_id: str | None = None
        auto_execute: bool = False

    class SubagentSpawnPayload(BaseModel):
        org_id: str = "org-default"
        parent_run_id: str
        child_pack: str
        input: dict[str, Any] = Field(default_factory=dict)
        budget_usd: float = 0.25
        started_by: str = "subagent"

    class ReplayRequestPayload(BaseModel):
        org_id: str = "org-default"
        run_ids: list[str] | None = None
        baseline_model_id: str | None = None
        cost_saving_target: float = 0.30
        max_quality_drop: float = 0.01

    class KillSwitchPayload(BaseModel):
        org_id: str = "org-default"
        active: bool
        pack: str | None = None

    class ChatCompletionPayload(BaseModel):
        model: str = "auto"
        messages: list[dict[str, Any]]
        max_tokens: int = 512
        org_id: str = "org-default"
        run_id: str | None = None
        step_id: str = "ext-step-0"
        pack: str = "ap_logging"
        step_role: str = "extract"
        task_class: str = "extract"
        data_class: str = "internal"
        allowed_models: list[str] | None = None
        run_limit_usd: float = 1.0

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "harness-orch-control-plane",
            "version": "0.1.0",
            "deployments": sorted(mgr.deployments.keys()),
            "timestamp": time.time(),
        }

    @app.get("/v1/metrics")
    async def metrics(org_id: str = "org-default") -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        run_ids = list(dep.log.run_ids(org_id))
        total_spend = sum(dep.budget.spent(rid) for rid in run_ids)
        states: dict[str, int] = {}
        for rid in run_ids:
            st = dep.orchestrator.runs.view(rid).state
            key = st.value if st else "unknown"
            states[key] = states.get(key, 0) + 1
        return {
            "org_id": org_id,
            "total_runs": len(run_ids),
            "runs_by_state": states,
            "total_spend_usd": round(total_spend, 6),
            "circuit_breakers": {mid: dep.breaker.state(mid) for mid in dep.catalogue},
            "outcome_stats": dep.selector.outcome_stats(),
        }

    # ---- FDE Customization & Enterprise Context Endpoints --------------------------------------------------
    @app.post("/v1/fde/deployments")
    async def configure_deployment(payload: DeploymentConfigPayload) -> dict[str, Any]:
        dep = mgr.configure_from_dict(payload.model_dump())
        return {
            "status": "configured",
            "org_id": dep.org_id,
            "org_name": dep.org_name,
            "region": dep.region,
            "memory_engine": dep.context.engine.name,
            "registered_packs": sorted(dep.packs.keys()),
            "catalogue_models": sorted(dep.catalogue.keys()),
            "skills": dep.skills.list_headers(principal=dep.default_principal),
            "outcome_stats_count": len(dep.selector.outcome_stats()),
        }

    @app.get("/v1/fde/deployments/{org_id}")
    async def get_deployment(org_id: str) -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        return {
            "org_id": dep.org_id,
            "org_name": dep.org_name,
            "region": dep.region,
            "memory_engine": dep.context.engine.name,
            "registered_packs": sorted(dep.packs.keys()),
            "catalogue_models": sorted(dep.catalogue.keys()),
            "skills": dep.skills.list_headers(principal=dep.default_principal),
            "outcome_stats": dep.selector.outcome_stats(),
        }

    @app.post("/v1/fde/deployments/{org_id}/context")
    async def sprinkle_context(org_id: str, payload: SprinkleContextPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        added_items = dep.sprinkle_context(payload.items) if payload.items else 0
        added_skills = dep.sprinkle_skills(payload.skills) if payload.skills else 0
        return {
            "status": "ok",
            "org_id": org_id,
            "ingested_context_items": added_items,
            "registered_skills": added_skills,
        }

    @app.post("/v1/fde/deployments/{org_id}/context/query")
    async def query_context(org_id: str, payload: ContextQueryPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        caller = Principal(
            id=payload.principal_id,
            org_id=org_id,
            teams=tuple(payload.teams),
            max_clearance=payload.max_clearance,
        )
        items = dep.context.retrieve(
            query=payload.query,
            caller=caller,
            run_id=payload.run_id,
            step_id=payload.step_id,
            limit=payload.limit,
        )
        return {
            "org_id": org_id,
            "count": len(items),
            "items": [
                {
                    "item_id": i.item_id,
                    "item_type": i.item_type,
                    "kind": i.kind,
                    "title_or_key": i.title_or_key,
                    "content": i.content,
                    "version_no": i.version_no,
                    "citation": i.citation,
                    "score": i.score,
                    "priority_rank": i.priority_rank,
                    "trust": i.trust,
                }
                for i in items
            ],
        }

    @app.post("/v1/fde/deployments/{org_id}/memory")
    async def sprinkle_memory(org_id: str, payload: SprinkleMemoryPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        ref_ids = dep.sprinkle_memory(payload.facts)
        return {"status": "ok", "org_id": org_id, "memory_ref_ids": ref_ids}

    @app.post("/v1/fde/deployments/{org_id}/task-logs")
    async def sprinkle_task_logs(org_id: str, payload: SprinkleTaskLogsPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        count = dep.sprinkle_task_logs(payload.logs)
        return {
            "status": "ok",
            "org_id": org_id,
            "ingested_logs": count,
            "outcome_stats": dep.selector.outcome_stats(),
        }

    @app.post("/v1/fde/deployments/{org_id}/route-preview")
    async def preview_route(org_id: str, payload: RoutePreviewPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        ranked = dep.preview_optimal_route(
            task_class=payload.task_class,
            step_role=payload.step_role,
            allowed_models=payload.allowed_models,
            stronger_than=payload.stronger_than,
        )
        return {
            "org_id": org_id,
            "task_class": payload.task_class,
            "step_role": payload.step_role,
            "optimal_model": ranked[0]["model_id"] if ranked else None,
            "candidates": ranked,
        }

    # ---- Run Lifecycle & Approval Endpoints (PRD H2, H6, H8) ----------------------------------------------
    @app.post("/v1/runs")
    async def create_run(payload: SubmitRunPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(payload.org_id)
        if payload.pack not in dep.packs:
            raise HTTPException(status_code=404, detail=f"unknown pack {payload.pack!r}")
        rid = payload.run_id or f"run-{uuid7()}"
        dep.orchestrator.submit(
            rid,
            pack=payload.pack,
            run_input={**payload.input, "_run_id": rid},
            budget_usd=payload.budget_usd,
            started_by=payload.started_by,
        )
        view = await dep.worker.run(rid) if payload.auto_execute else dep.orchestrator.runs.view(rid)
        return _format_run_view(dep, view)

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str, org_id: str = "org-default") -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        view = dep.orchestrator.runs.view(run_id)
        if view.state is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        return _format_run_view(dep, view)

    @app.post("/v1/runs/{run_id}/execute")
    async def execute_run(run_id: str, org_id: str = "org-default") -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        view = await dep.worker.run(run_id)
        return _format_run_view(dep, view)

    @app.post("/v1/runs/{run_id}/pause")
    async def pause_run(run_id: str, org_id: str = "org-default") -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        try:
            view = dep.orchestrator.runs.transition(run_id, "run.paused")
        except InvalidTransition as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return _format_run_view(dep, view)

    @app.post("/v1/runs/{run_id}/resume")
    async def resume_run(run_id: str, org_id: str = "org-default", auto_execute: bool = True) -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        try:
            view = dep.orchestrator.runs.transition(run_id, "run.resumed")
            if auto_execute:
                view = await dep.worker.run(run_id)
        except InvalidTransition as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return _format_run_view(dep, view)

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, org_id: str = "org-default", by: str = "operator") -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        try:
            view = dep.orchestrator.cancel(run_id, by=by)
        except InvalidTransition as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return _format_run_view(dep, view)

    @app.post("/v1/runs/{run_id}/fork")
    async def fork_run(run_id: str, payload: ForkRunPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(payload.org_id)
        try:
            view = dep.fork_from_checkpoint(run_id, upto_seq=payload.upto_seq, new_run_id=payload.new_run_id)
            if payload.auto_execute:
                view = await dep.worker.run(view.run_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return _format_run_view(dep, view)

    @app.post("/v1/runs/{run_id}/approvals/{approval_id}")
    async def decide_approval(run_id: str, approval_id: str, payload: ApprovalDecisionPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(payload.org_id)
        try:
            view = dep.orchestrator.decide(run_id, approval_id=approval_id, approved=payload.approved, approver=payload.approver)
            if payload.auto_resume:
                view = await dep.worker.run(run_id)
        except InvalidTransition as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return _format_run_view(dep, view)

    @app.get("/v1/runs/{run_id}/events")
    async def list_run_events(run_id: str, org_id: str = "org-default", after_seq: int = 0) -> dict[str, Any]:
        dep = mgr.get_or_create(org_id)
        evs = dep.log.read(org_id, run_id, after_seq=after_seq)
        corrupted = dep.log.verify(org_id, run_id) if hasattr(dep.log, "verify") else []
        return {
            "org_id": org_id,
            "run_id": run_id,
            "integrity_ok": len(corrupted) == 0,
            "corrupted_seqs": corrupted,
            "events": [
                {
                    "id": e.id,
                    "seq": e.seq,
                    "type": e.type,
                    "step_id": e.step_id,
                    "payload": e.payload,
                    "payload_hash": e.payload_hash,
                    "created_at": e.created_at,
                }
                for e in evs
            ],
        }

    @app.post("/v1/subagents")
    async def spawn_subagent(payload: SubagentSpawnPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(payload.org_id)
        try:
            return await dep.subagents.run_subagent(
                parent_run_id=payload.parent_run_id,
                child_pack_name=payload.child_pack,
                child_input=payload.input,
                requested_budget_usd=payload.budget_usd,
                started_by=payload.started_by,
            )
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    # ---- Deterministic Replay & Admin Kill Switch (PRD P4, P6, P8) ----------------------------------------
    @app.post("/v1/replay")
    async def run_replay(payload: ReplayRequestPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(payload.org_id)
        report = dep.replay_engine.evaluate(
            candidate_selector=dep.selector,
            run_ids=payload.run_ids,
            baseline_model_id=payload.baseline_model_id,
            cost_saving_target=payload.cost_saving_target,
            max_quality_drop=payload.max_quality_drop,
        )
        return report.as_dict()

    @app.post("/v1/admin/kill-switch")
    async def set_kill_switch(payload: KillSwitchPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(payload.org_id)
        if isinstance(dep.policy_client, EmbeddedPolicyClient):
            dep.policy_client.set_kill_switch(payload.active, pack=payload.pack)
        return {
            "status": "updated",
            "org_id": payload.org_id,
            "pack": payload.pack,
            "kill_switch_active": payload.active,
        }

    # ---- OpenAI & Anthropic Compatible Router Endpoints (PRD R1, H9) --------------------------------------
    @app.post("/v1/chat/completions")
    async def openai_chat_completions(payload: ChatCompletionPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(payload.org_id)
        rid = payload.run_id or f"ext-oai-{uuid7()}"
        allowed = payload.allowed_models or (list(dep.catalogue.keys()) if payload.model == "auto" else [payload.model])
        req = RoutingRequest(
            org_id=payload.org_id,
            run_id=rid,
            step_id=payload.step_id,
            pack=payload.pack,
            step_role=payload.step_role,
            task_class=payload.task_class,
            data_class=payload.data_class,
            messages=payload.messages,
            allowed_models=allowed,
            run_limit_usd=payload.run_limit_usd,
            max_tokens=payload.max_tokens,
        )
        try:
            res = await dep.router.complete(req)
        except RouterError as e:
            raise HTTPException(status_code=403, detail={"reason": e.reason, "decision_id": e.decision_id}) from e
        return {
            "id": f"chatcmpl-{res.decision_id}",
            "object": "chat.completion",
            "model": res.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": res.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "total_tokens": res.prompt_tokens + res.completion_tokens,
                "cost_usd": res.cost_usd,
            },
            "harness_routing": {
                "decision_id": res.decision_id,
                "tier": res.tier,
                "fallback_from": res.fallback_from,
            },
        }

    @app.post("/v1/messages")
    async def anthropic_messages(payload: ChatCompletionPayload) -> dict[str, Any]:
        dep = mgr.get_or_create(payload.org_id)
        rid = payload.run_id or f"ext-ant-{uuid7()}"
        allowed = payload.allowed_models or (list(dep.catalogue.keys()) if payload.model == "auto" else [payload.model])
        req = RoutingRequest(
            org_id=payload.org_id,
            run_id=rid,
            step_id=payload.step_id,
            pack=payload.pack,
            step_role=payload.step_role,
            task_class=payload.task_class,
            data_class=payload.data_class,
            messages=payload.messages,
            allowed_models=allowed,
            run_limit_usd=payload.run_limit_usd,
            max_tokens=payload.max_tokens,
        )
        try:
            res = await dep.router.complete(req)
        except RouterError as e:
            raise HTTPException(status_code=403, detail={"reason": e.reason, "decision_id": e.decision_id}) from e
        return {
            "id": f"msg_{res.decision_id}",
            "type": "message",
            "role": "assistant",
            "model": res.model_id,
            "content": [{"type": "text", "text": res.text}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": res.prompt_tokens,
                "output_tokens": res.completion_tokens,
                "cost_usd": res.cost_usd,
            },
            "harness_routing": {
                "decision_id": res.decision_id,
                "tier": res.tier,
                "fallback_from": res.fallback_from,
            },
        }

    def _format_run_view(dep: FDEDeployment, view: Any) -> dict[str, Any]:
        evs = dep.log.read(dep.org_id, view.run_id)
        succeeded = [e.payload.get("result") for e in evs if e.type == "run.succeeded"]
        return {
            "org_id": dep.org_id,
            "run_id": view.run_id,
            "state": view.state.value if view.state else None,
            "last_seq": view.last_seq,
            "started_by": view.started_by,
            "failure_reason": view.failure_reason,
            "pending_approval": view.pending_approval,
            "spent_usd": round(dep.budget.spent(view.run_id), 6),
            "result": succeeded[-1] if succeeded else None,
            "meta": view.meta,
        }

    return app
