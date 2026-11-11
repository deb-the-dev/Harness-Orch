"""Orchestrator (run lifecycle) and worker (step loop). Workers are stateless: everything they need is in the event log,
so a crashed worker can be replaced and a run can wait for a human for as long as it takes."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from .artifacts import Offloader
from .compaction import Compactor
from .events import Event, EventLog
from .ids import uuid7
from .packs import Action, Done, Finish, ModelCall, TaskPack, ToolCall
from .policy import HookRunner
from .router import Router, RouterError, RoutingRequest
from .router.core import RouterResult
from .rubric import RubricGrader, Verdict
from .runs import InvalidTransition, RunService, RunState, RunView
from .tools import ToolError, ToolRuntime, result_bytes

_PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\b\d{9,}\b")


def contains_pii(text: str) -> bool:
    return bool(_PII.search(text))


def redact_pii(text: str) -> str:
    return _PII.sub("[REDACTED]", text)


class Orchestrator:
    """Run lifecycle. A lease and queue in PostgreSQL replaces this in-process version (decided); the interface stays."""

    def __init__(self, log: EventLog, org_id: str, queue: Any = None):
        self.log, self.org_id, self.runs, self.queue = log, org_id, RunService(log, org_id), queue

    def submit(self, run_id: str, pack: str, run_input: dict, budget_usd: float, started_by: str) -> RunView:
        self.runs.create(run_id, pack=pack, input=run_input, budget_usd=budget_usd, started_by=started_by)
        view = self.runs.transition(run_id, "run.queued")
        if self.queue:
            self.queue.enqueue(self.org_id, run_id)               # the log is written first: a queue row is always rebuildable
        return view

    def recover(self, run_id: str) -> RunView:
        """Called when a worker's lease expires: back to the queue; the next worker resumes from the log."""
        return self.runs.transition(run_id, "run.worker_lost")

    def decide(self, run_id: str, approval_id: str, approved: bool, approver: str) -> RunView:
        view = self.runs.view(run_id)
        if not view.pending_approval or view.pending_approval["approval_id"] != approval_id:
            raise InvalidTransition("no such pending approval")
        out = self.runs.transition(run_id, "approval.decided", {"approval_id": approval_id, "state": "approved" if approved else "rejected",
                                                               "approver": approver, "step_id": view.pending_approval.get("step_id")},
                                   step_id=view.pending_approval.get("step_id"))
        if self.queue:
            self.queue.enqueue(self.org_id, run_id)               # wake the run; a worker resumes it from the log
        return out

    def cancel(self, run_id: str, by: str) -> RunView:
        out = self.runs.transition(run_id, "run.cancelled", {"by": by})
        if self.queue:
            self.queue.remove(run_id)
        return out


class Worker:
    def __init__(self, log: EventLog, org_id: str, packs: dict[str, TaskPack], router: Router, hooks: HookRunner, tools: ToolRuntime,
                 offloader: Offloader | None = None, compactor: Compactor | None = None, grader: RubricGrader | None = None):
        self.log, self.org_id, self.packs, self.router, self.hooks, self.tools = log, org_id, packs, router, hooks, tools
        self.offloader, self.compactor, self.grader = offloader, compactor, grader
        self.runs = RunService(log, org_id)

    # ---- main loop ----------------------------------------------------------------------------------------
    async def run(self, run_id: str) -> RunView:
        view = self.runs.view(run_id)
        if view.state == RunState.QUEUED:
            view = self.runs.transition(run_id, "run.claimed")
        if view.state != RunState.RUNNING:
            return view                                           # waiting, paused or finished: nothing to do
        meta = view.meta
        pack = self.packs[meta["pack"]]
        while True:
            events = self.log.read(self.org_id, run_id)
            history = self._history(events)
            action = pack.next_action(meta["input"], history)
            if isinstance(action, Finish):
                return self.runs.transition(run_id, "run.succeeded", {"result": action.result})
            step_id = f"step-{len(history)}"
            outcome = await self._step(run_id, step_id, pack, meta, action, [e for e in events if e.step_id == step_id])
            if outcome != "continue":
                return self.runs.view(run_id)

    @staticmethod
    def _history(events: list[Event]) -> list[Done]:
        started = {e.step_id: e.payload["action"] for e in events if e.type == "step.started"}
        return [Done(started.get(e.step_id, {}), e.payload["result"]) for e in events if e.type == "step.completed"]

    # ---- one step -----------------------------------------------------------------------------------------
    async def _step(self, run_id, step_id, pack, meta, action: Action, step_events: list[Event]) -> str:
        if not any(e.type == "step.started" for e in step_events):
            kind = {"kind": "model", "step_role": action.step_role} if isinstance(action, ModelCall) else {"kind": "tool", "tool": action.tool, "args": action.args}
            self.log.append(self.org_id, run_id, "step.started", {"action": kind}, step_id=step_id)
        if isinstance(action, ModelCall):
            return await self._model_step(run_id, step_id, pack, meta, action, step_events)
        return await self._tool_step(run_id, step_id, pack, action, step_events)

    def _complete(self, run_id, step_id, result: dict[str, Any]) -> str:
        self.log.append(self.org_id, run_id, "step.completed", {"result": result}, step_id=step_id)
        return "continue"

    def _fail(self, run_id, reason: str) -> str:
        self.runs.transition(run_id, "run.failed", {"reason": reason})
        return "failed"

    async def _check(self, a: ModelCall, req: RoutingRequest, res: RouterResult, step_id: str, step_events: list[Event]) -> tuple[bool, list[str]]:
        """Run the pack's checks on one answer: its own verifier, then the rubric grader. Returns (ok, gaps for a retry)."""
        if a.verify is not None and not a.verify(res.text):
            self.log.append(self.org_id, req.run_id, "step.outcome", {"ok": False, "kind": "verifier", "model": res.model_id}, step_id=step_id)
            return False, ["the answer did not pass the format check"]
        if a.rubric is None:
            return True, []
        if self.grader is None:
            self.log.append(self.org_id, req.run_id, "step.outcome", {"ok": False, "kind": "rubric", "verdict": "unverified", "reason": "no grader configured"}, step_id=step_id)
            return False, []                                       # a rubric that cannot be checked is not a pass
        prior = [e.payload for e in step_events if e.type == "step.outcome" and e.payload.get("kind") == "rubric" and e.payload.get("output_decision_id") == res.decision_id]
        if prior:                                                  # resumed after a crash: reuse the grading already paid for
            v = Verdict(prior[-1]["verdict"], prior[-1].get("results", []), prior[-1].get("grader_model"))
        else:
            v = await self.grader.grade(req, a.rubric, res.text, res.model_id)
            self.log.append(self.org_id, req.run_id, "step.outcome", {"ok": v.ok, **v.as_event(res.decision_id)}, step_id=step_id)
        return v.ok, v.gaps()

    async def _model_step(self, run_id, step_id, pack, meta, a: ModelCall, step_events) -> str:
        req = RoutingRequest(self.org_id, run_id, step_id, pack.name, a.step_role, pack.task_class, pack.data_class, a.messages,
                             list(pack.allowed_models), float(meta["budget_usd"]), a.max_tokens)
        responses = [e for e in step_events if e.type == "model.response"]
        try:
            if responses:                                          # a crash after the response: reuse it, do not pay twice
                r = responses[-1].payload
                res = RouterResult(r["text"], r["model"], r["decision_id"], r["prompt_tokens"], r["completion_tokens"], r["cost_usd"], request=req)
                if len(responses) > 1:                            # an escalation already happened before the crash
                    res.request = RoutingRequest(**{**req.__dict__, "escalation_count": len(responses) - 1, "escalated_from": "earlier"})
            else:
                if self.compactor is not None:
                    req = await self.compactor.maybe_compact(req)
                res = await self.router.complete(req)
            ok, gaps = await self._check(a, res.request or req, res, step_id, step_events)
            if not ok:
                try:
                    retry_req = res.request
                    if gaps:                                       # tell the stronger model what was wrong with the first answer
                        retry_req = replace(retry_req, messages=[*retry_req.messages, {"role": "user", "content":
                                            "A reviewer found these problems with your previous answer. Fix them:\n- " + "\n- ".join(gaps)}])
                    res = await self.router.escalate(retry_req, res)
                    ok, _ = await self._check(a, res.request or req, res, step_id, step_events)
                    self.log.append(self.org_id, run_id, "step.outcome", {"ok": ok, "kind": "escalation", "model": res.model_id, "escalated": True}, step_id=step_id)
                except RouterError as e:                           # no stronger model, or the cap: the step fails, the run goes on
                    self.log.append(self.org_id, run_id, "step.outcome", {"ok": False, "kind": "escalation_unavailable", "reason": e.reason}, step_id=step_id)
        except RouterError as e:
            return self._fail(run_id, "budget_hard_limit" if e.reason == "budget_hard_limit" else f"router:{e.reason}")
        text = res.text
        d = await self.hooks.check(run_id, step_id, {"hook": "post_model", "pack": pack.name, "action_class": "read", "output": {"contains_pii": contains_pii(text)}})
        if d.verdict == "modify" and "redact_pii" in d.reasons:
            text = redact_pii(text)
        elif not d.allowed:
            return self._complete(run_id, step_id, {"ok": False, "error": "blocked_by_policy", "reasons": list(d.reasons)})
        return self._complete(run_id, step_id, {"ok": ok, "text": text, "model": res.model_id, "decision_id": res.decision_id, "cost_usd": res.cost_usd})

    # ---- tools and approvals -----------------------------------------------------------------------------
    @staticmethod
    def _approval_state(step_events: list[Event]) -> str:
        state = "none"
        for e in step_events:
            if e.type == "approval.requested":
                state = "pending"
            elif e.type == "approval.decided":
                state = e.payload["state"]
        return state

    def _request_approval(self, run_id, step_id, a: ToolCall, reason: str) -> str:
        self.runs.transition(run_id, "approval.requested", {"approval_id": uuid7(), "step_id": step_id, "tool": a.tool, "args": a.args, "reason": reason}, step_id=step_id)
        return "wait"

    async def _tool_step(self, run_id, step_id, pack, a: ToolCall, step_events) -> str:
        spec = self.tools.get(a.tool)
        done = [e for e in step_events if e.type == "tool.result"]
        marks = [i for i, e in enumerate(step_events) if e.type == "tool.executing"]
        executing = bool(marks)
        # after a possible crash mid-call, an earlier approval does not cover repeating the side effect: only a newer one does
        approval = self._approval_state(step_events[marks[-1] + 1:] if executing else step_events)
        if not done:
            if executing and spec is not None and not spec.idempotent and approval == "none":
                return self._request_approval(run_id, step_id, a, "outcome_unknown")      # a human decides whether to repeat it
            d = await self.hooks.check(run_id, step_id, {"hook": "pre_tool", "pack": pack.name, "action_class": spec.action_class if spec else "read",
                                                         "tool": a.tool, "approval": {"state": approval}})
            if d.verdict == "require_approval":
                return self._request_approval(run_id, step_id, a, "policy:" + ",".join(d.reasons))
            if not d.allowed or spec is None:
                return self._complete(run_id, step_id, {"ok": False, "error": "blocked_by_policy", "reasons": list(d.reasons) or ["unknown_tool"]})
            self.log.append(self.org_id, run_id, "tool.executing", {"tool": a.tool, "args": a.args}, step_id=step_id)
            try:
                out = {"ok": True, "output": await self.tools.run(a.tool, a.args)}
            except ToolError as e:
                out = {"ok": False, "error": str(e)}
            self.log.append(self.org_id, run_id, "tool.result", out, step_id=step_id)
        else:
            out = done[-1].payload
        if out.get("ok") and self.offloader is not None and not out.get("offloaded"):
            out = self.offloader.maybe_offload(run_id, step_id, out)        # PRD C1: the model sees a preview and a handle
        if out.get("ok"):
            d = await self.hooks.check(run_id, step_id, {"hook": "post_tool", "pack": pack.name, "action_class": spec.action_class, "output": {"bytes": result_bytes(out)}})
            if d.verdict == "modify":
                out = {"ok": True, "output": {"truncated": True, "preview": json.dumps(out["output"], default=str)[:200]}}
            elif not d.allowed:
                out = {"ok": False, "error": "blocked_by_policy", "reasons": list(d.reasons)}
        return self._complete(run_id, step_id, out)
