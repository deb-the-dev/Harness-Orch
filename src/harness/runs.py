"""Run state machine. State is a projection of the event log, never stored separately."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .events import Event, EventLog


class RunState(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    APPROVAL_WAIT = "approval_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}
LIVE = set(RunState) - TERMINAL

# event type -> state it moves the run to
EVENT_TARGET = {
    "run.created": RunState.CREATED,
    "run.queued": RunState.QUEUED,
    "run.claimed": RunState.RUNNING,
    "run.worker_lost": RunState.QUEUED,
    "run.paused": RunState.PAUSED,
    "run.resumed": RunState.RUNNING,
    "approval.requested": RunState.APPROVAL_WAIT,
    "approval.decided": RunState.RUNNING,
    "run.succeeded": RunState.SUCCEEDED,
    "run.failed": RunState.FAILED,
    "run.cancelled": RunState.CANCELLED,
}

# allowed moves (state diagram in the architecture document); cancel and fail are allowed from every live state
ALLOWED: dict[RunState, set[RunState]] = {
    RunState.CREATED: {RunState.QUEUED},
    RunState.QUEUED: {RunState.RUNNING},
    RunState.RUNNING: {RunState.QUEUED, RunState.PAUSED, RunState.APPROVAL_WAIT, RunState.SUCCEEDED},
    RunState.PAUSED: {RunState.RUNNING},
    RunState.APPROVAL_WAIT: {RunState.RUNNING},
}
for _s in LIVE:
    ALLOWED[_s] = ALLOWED.get(_s, set()) | {RunState.CANCELLED, RunState.FAILED}
for _s in TERMINAL:
    ALLOWED[_s] = set()


class InvalidTransition(Exception):
    pass


@dataclass
class RunView:
    run_id: str
    state: RunState | None = None
    last_seq: int = 0
    failure_reason: str | None = None
    pending_approval: dict[str, Any] | None = None
    started_by: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def project(events: list[Event]) -> RunView:
    """Fold events into the current view. Unknown event types do not move the state."""
    view = RunView(run_id=events[0].run_id if events else "")
    for e in events:
        view.last_seq = e.seq
        target = EVENT_TARGET.get(e.type)
        if e.type == "run.created":
            view.started_by = e.payload.get("started_by")
            view.meta = {k: v for k, v in e.payload.items() if k != "started_by"}
        if e.type == "approval.requested":
            view.pending_approval = {"approval_id": e.payload["approval_id"], **e.payload}
        if e.type == "approval.decided":
            view.pending_approval = None
        if e.type == "run.failed":
            view.failure_reason = e.payload.get("reason")
        if target is not None:
            view.state = target
    return view


class RunService:
    """Validates a transition against the projected state, then appends the event that makes it true."""

    def __init__(self, log: EventLog, org_id: str):
        self.log, self.org_id = log, org_id

    def view(self, run_id: str) -> RunView:
        return project(self.log.read(self.org_id, run_id))

    def create(self, run_id: str, **meta) -> RunView:
        if self.log.read(self.org_id, run_id):
            raise InvalidTransition(f"run {run_id} already exists")
        self.log.append(self.org_id, run_id, "run.created", meta)
        return self.view(run_id)

    def transition(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None, step_id: str | None = None) -> RunView:
        target = EVENT_TARGET.get(event_type)
        if target is None or event_type == "run.created":
            raise InvalidTransition(f"{event_type} is not a transition")
        cur = self.view(run_id)
        if cur.state is None:
            raise InvalidTransition(f"run {run_id} does not exist")
        if target not in ALLOWED[cur.state]:
            raise InvalidTransition(f"{cur.state.value} -> {target.value} is not allowed")
        self.log.append(self.org_id, run_id, event_type, payload or {}, step_id=step_id)
        return self.view(run_id)
