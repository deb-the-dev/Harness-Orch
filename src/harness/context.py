"""Organizational context, scoped memory, and progressive-disclosure skills (PRD sections 9 & 10: M1-M10, C6; ERD 3.5 & 3.7).

Design invariants enforced in code (never by prompt):
  * M1: Swappable memory engine protocol with Graphiti-style (temporal validity windows) and Mem0-style (hybrid add-only) adapters.
  * M2 & M3: Every read and write carries caller identity and scope (user, team, agent, org); retrieval filters by the caller's permissions and data-class clearance before ranking.
  * M4: Policy and procedure items carry versions, effective dates, and owners; retrieval emits deterministic version citations.
  * M5: Every retrieved context or memory item is tagged `trust="untrusted"` and cannot grant permissions or alter policy.
  * M6 & M7: Memory writes pass validation and follow the review path (`pending -> approved | rejected`, `approved -> revoked`); revoking a source or memory ref immediately hides derived items at retrieval.
  * M10: Every retrieval appends a `memory.access` audit event to the append-only event log.
  * C6: Organizational procedures are registered as versioned, effective-dated skills; prompts receive only `name` and `description`, while the full body loads on demand via a logged tool call.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .events import EventLog
from .ids import uuid7

KNOWLEDGE_PRIORITY = {
    "policy": 1,
    "procedure": 1,
    "document": 2,
    "ticket": 2,
    "data": 3,
    "code": 4,
}

CLEARANCE_ORDER = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}

SCOPES = ("user", "team", "agent", "org")
REVIEW_STATES = ("pending", "approved", "rejected", "revoked")
ALLOWED_REVIEW_TRANSITIONS = {
    "pending": {"approved", "rejected"},
    "approved": {"revoked"},
    "rejected": set(),
    "revoked": set(),
}


@dataclass(frozen=True)
class Principal:
    """Identity from SSO or service credentials (ERD 3.1: principal)."""
    id: str
    org_id: str
    kind: str = "user"                              # "user" | "service"
    external_subject: str = ""
    teams: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    max_clearance: str = "internal"                 # highest data_class the principal may read

    def can_access_clearance(self, data_class: str) -> bool:
        return CLEARANCE_ORDER.get(self.max_clearance, 0) >= CLEARANCE_ORDER.get(data_class, 99)


@dataclass(frozen=True)
class KnowledgeItem:
    """Versioned, effective-dated knowledge or policy item (ERD 3.5: knowledge_item)."""
    id: str
    org_id: str
    source_id: str
    external_id: str
    kind: str                                       # "policy" | "procedure" | "document" | "ticket" | "data" | "code"
    title: str
    content: str
    version_no: int = 1
    effective_from: float = 0.0
    effective_to: float | None = None
    owner: str = "platform"
    data_class: str = "internal"
    allowed_teams: tuple[str, ...] = ()             # empty = all teams in org with sufficient clearance
    allowed_principals: tuple[str, ...] = ()
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            digest = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            object.__setattr__(self, "content_hash", digest)

    @property
    def priority_rank(self) -> int:
        return KNOWLEDGE_PRIORITY.get(self.kind, 99)

    @property
    def citation(self) -> str:
        return f"[{self.kind}:{self.external_id}@v{self.version_no} (owner={self.owner})]"

    def is_effective(self, at_time: float | None = None) -> bool:
        ts = time.time() if at_time is None else at_time
        if ts < self.effective_from:
            return False
        if self.effective_to is not None and ts >= self.effective_to:
            return False
        return True


@dataclass(frozen=True)
class MemoryRef:
    """Reference and review state for an engine-stored memory fact (ERD 3.5: memory_ref)."""
    id: str
    org_id: str
    engine: str                                     # "graphiti" | "mem0"
    engine_item_id: str
    scope_type: str                                 # "user" | "team" | "agent" | "org"
    scope_id: str
    key: str
    content: str
    source_event_id: str | None = None
    source_id: str | None = None
    data_class: str = "internal"
    review_state: str = "approved"                  # "pending" | "approved" | "rejected" | "revoked"
    valid_from: float = field(default_factory=time.time)
    valid_to: float | None = None
    tags: tuple[str, ...] = ()

    def is_active(self, at_time: float | None = None) -> bool:
        if self.review_state != "approved":
            return False
        ts = time.time() if at_time is None else at_time
        if ts < self.valid_from:
            return False
        if self.valid_to is not None and ts >= self.valid_to:
            return False
        return True


@dataclass(frozen=True)
class RetrievedItem:
    """Permission-filtered, untrusted context or memory item returned to a worker step (PRD M5, M10)."""
    item_id: str
    item_type: str                                  # "knowledge" | "memory" | "skill"
    kind: str
    title_or_key: str
    content: str
    version_no: int
    citation: str
    score: float
    priority_rank: int
    trust: str = "untrusted"                        # PRD M5: always fixed to "untrusted"

    def as_prompt_block(self) -> str:
        return f"<untrusted_context ref={self.citation!r} trust=\"untrusted\">\n{self.content}\n</untrusted_context>"


class MemoryEngine(Protocol):
    """Swappable memory engine interface behind our scope and permission layer (PRD M1)."""
    name: str

    def put(self, org_id: str, key: str, content: str, tags: tuple[str, ...] = (), valid_from: float | None = None,
            valid_to: float | None = None) -> tuple[str, list[str]]:
        """Stores a fact and returns (engine_item_id, list of invalidated prior engine_item_ids)."""
        ...

    def search(self, org_id: str, query: str, candidate_ids: set[str], limit: int = 10) -> list[tuple[str, float]]:
        """Scores candidate engine_item_ids for `query`, returning [(engine_item_id, relevance_score)]."""
        ...

    def delete_org(self, org_id: str) -> int:
        """Tenant erasure support (ERD section 4)."""
        ...


def _lexical_score(query: str, text: str, tags: tuple[str, ...] = ()) -> float:
    q_tokens = {t.lower() for t in query.replace("_", " ").split() if len(t) > 1}
    if not q_tokens:
        return 0.5
    haystack = (text + " " + " ".join(tags)).lower()
    words = set(haystack.replace("_", " ").replace(".", " ").replace(",", " ").split())
    overlap = len(q_tokens & words)
    substring_hits = sum(1 for t in q_tokens if t in haystack)
    return round((overlap * 1.0 + substring_hits * 0.5) / max(len(q_tokens), 1), 4)


class GraphitiMemoryAdapter:
    """Graphiti-style temporal fact adapter: facts carry validity windows and newer facts with the same key
    automatically invalidate outdated ones without deleting audit history (PRD section 9, Tools List section 5)."""
    name: str = "graphiti"

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def put(self, org_id: str, key: str, content: str, tags: tuple[str, ...] = (), valid_from: float | None = None,
            valid_to: float | None = None) -> tuple[str, list[str]]:
        now = time.time() if valid_from is None else valid_from
        invalidated: list[str] = []
        for eid, rec in self._records.items():
            if rec["org_id"] == org_id and rec["key"] == key and rec["valid_to"] is None:
                rec["valid_to"] = now
                invalidated.append(eid)
        item_id = f"gr-{uuid7()}"
        self._records[item_id] = {
            "org_id": org_id,
            "key": key,
            "content": content,
            "tags": tags,
            "valid_from": now,
            "valid_to": valid_to,
        }
        return item_id, invalidated

    def search(self, org_id: str, query: str, candidate_ids: set[str], limit: int = 10) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for eid in candidate_ids:
            rec = self._records.get(eid)
            if not rec or rec["org_id"] != org_id:
                continue
            s = _lexical_score(query, f"{rec['key']} {rec['content']}", rec.get("tags", ()))
            if s > 0:
                scored.append((eid, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def delete_org(self, org_id: str) -> int:
        to_del = [k for k, v in self._records.items() if v["org_id"] == org_id]
        for k in to_del:
            del self._records[k]
        return len(to_del)


class Mem0MemoryAdapter:
    """Mem0-style add-only hybrid extraction adapter (PRD section 9, Tools List section 5)."""
    name: str = "mem0"

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def put(self, org_id: str, key: str, content: str, tags: tuple[str, ...] = (), valid_from: float | None = None,
            valid_to: float | None = None) -> tuple[str, list[str]]:
        item_id = f"m0-{uuid7()}"
        self._records[item_id] = {
            "org_id": org_id,
            "key": key,
            "content": content,
            "tags": tags,
            "created_at": valid_from or time.time(),
        }
        return item_id, []

    def search(self, org_id: str, query: str, candidate_ids: set[str], limit: int = 10) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for eid in candidate_ids:
            rec = self._records.get(eid)
            if not rec or rec["org_id"] != org_id:
                continue
            s = _lexical_score(query, f"{rec['key']} {rec['content']}", rec.get("tags", ()))
            if s > 0:
                scored.append((eid, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def delete_org(self, org_id: str) -> int:
        to_del = [k for k, v in self._records.items() if v["org_id"] == org_id]
        for k in to_del:
            del self._records[k]
        return len(to_del)


@dataclass(frozen=True)
class ProcedureSkill:
    """Versioned, effective-dated organizational procedure served via progressive disclosure (PRD C6, ERD 3.7)."""
    name: str
    org_id: str
    description: str
    body: str
    version_no: int = 1
    effective_from: float = 0.0
    effective_to: float | None = None
    owner: str = "platform"
    data_class: str = "internal"
    allowed_teams: tuple[str, ...] = ()

    @property
    def citation(self) -> str:
        return f"[skill:{self.name}@v{self.version_no}]"

    def is_effective(self, at_time: float | None = None) -> bool:
        ts = time.time() if at_time is None else at_time
        if ts < self.effective_from:
            return False
        if self.effective_to is not None and ts >= self.effective_to:
            return False
        return True


class SkillRegistry:
    """Progressive disclosure skill registry (PRD C6): prompts carry only name + description; full body loads on demand."""

    def __init__(self, log: EventLog, org_id: str) -> None:
        self.log = log
        self.org_id = org_id
        self._skills: dict[str, list[ProcedureSkill]] = {}

    def register(self, skill: ProcedureSkill) -> ProcedureSkill:
        if skill.org_id != self.org_id:
            raise PermissionError("skill org_id mismatch")
        versions = self._skills.setdefault(skill.name, [])
        versions.append(skill)
        versions.sort(key=lambda s: s.version_no)
        return skill

    def active_skill(self, name: str, principal: Principal | None = None, at_time: float | None = None) -> ProcedureSkill | None:
        versions = self._skills.get(name, [])
        candidates = [s for s in versions if s.is_effective(at_time)]
        if not candidates:
            return None
        latest = candidates[-1]
        if principal is not None:
            if not principal.can_access_clearance(latest.data_class):
                return None
            if latest.allowed_teams and not (set(latest.allowed_teams) & set(principal.teams)):
                return None
        return latest

    def list_headers(self, principal: Principal | None = None, at_time: float | None = None) -> list[dict[str, Any]]:
        """Returns only name, description, version_no, and citation for prompt injection (PRD C6)."""
        headers: list[dict[str, Any]] = []
        for name in sorted(self._skills):
            s = self.active_skill(name, principal=principal, at_time=at_time)
            if s is not None:
                headers.append({
                    "name": s.name,
                    "description": s.description,
                    "version_no": s.version_no,
                    "citation": s.citation,
                })
        return headers

    def load_body(self, name: str, run_id: str, step_id: str | None = None, principal: Principal | None = None,
                  at_time: float | None = None) -> dict[str, Any]:
        """On-demand full body load, recorded in the append-only event log (PRD C6 & M10)."""
        s = self.active_skill(name, principal=principal, at_time=at_time)
        if s is None:
            self.log.append(self.org_id, run_id, "memory.access", {
                "kind": "skill",
                "skill_name": name,
                "caller_principal_id": principal.id if principal else "system",
                "found": False,
                "trust": "untrusted",
            }, step_id=step_id)
            return {"ok": False, "error": f"skill {name!r} not found or not permitted"}
        self.log.append(self.org_id, run_id, "memory.access", {
            "kind": "skill",
            "skill_name": s.name,
            "version_no": s.version_no,
            "citation": s.citation,
            "caller_principal_id": principal.id if principal else "system",
            "found": True,
            "trust": "untrusted",
        }, step_id=step_id)
        return {
            "ok": True,
            "name": s.name,
            "version_no": s.version_no,
            "citation": s.citation,
            "trust": "untrusted",
            "body": s.body,
        }


class ContextService:
    """Scoped retrieval and memory governance service (PRD section 9: M1-M10; Architecture section 2 & 4)."""

    def __init__(self, log: EventLog, org_id: str, engine: MemoryEngine | None = None) -> None:
        self.log = log
        self.org_id = org_id
        self.engine: MemoryEngine = engine or GraphitiMemoryAdapter()
        self._knowledge: dict[tuple[str, str], list[KnowledgeItem]] = {}
        self._memory_refs: dict[str, MemoryRef] = {}
        self._degraded: bool = False

    def swap_engine(self, new_engine: MemoryEngine) -> None:
        """Swappable memory engine behind our interface (PRD M1)."""
        self.engine = new_engine

    # ---- Knowledge & Policy Corpus (M4, M8) ---------------------------------------------------------------
    def ingest_knowledge(self, source_id: str, external_id: str, kind: str, title: str, content: str,
                         version_no: int | None = None, effective_from: float = 0.0, effective_to: float | None = None,
                         owner: str = "platform", data_class: str = "internal",
                         allowed_teams: tuple[str, ...] = (), allowed_principals: tuple[str, ...] = ()) -> KnowledgeItem:
        if kind not in KNOWLEDGE_PRIORITY:
            raise ValueError(f"unknown knowledge kind {kind!r}; expected one of {tuple(KNOWLEDGE_PRIORITY)}")
        key = (source_id, external_id)
        existing = self._knowledge.setdefault(key, [])
        next_ver = version_no if version_no is not None else (existing[-1].version_no + 1 if existing else 1)
        # If previous version has no effective_to and new version starts later, close the previous version's window
        if existing and existing[-1].effective_to is None and effective_from > existing[-1].effective_from:
            prev = existing[-1]
            existing[-1] = replace(prev, effective_to=effective_from)
        item = KnowledgeItem(
            id=uuid7(),
            org_id=self.org_id,
            source_id=source_id,
            external_id=external_id,
            kind=kind,
            title=title,
            content=content,
            version_no=next_ver,
            effective_from=effective_from,
            effective_to=effective_to,
            owner=owner,
            data_class=data_class,
            allowed_teams=tuple(allowed_teams),
            allowed_principals=tuple(allowed_principals),
        )
        existing.append(item)
        existing.sort(key=lambda x: x.version_no)
        return item

    # ---- Scoped Memory Writes, Review & Revocation (M2, M5, M6, M7) ---------------------------------------
    def write_memory(self, scope_type: str, scope_id: str, key: str, content: str,
                     caller: Principal, data_class: str = "internal", source_event_id: str | None = None,
                     source_id: str | None = None, require_review: bool | None = None,
                     valid_from: float | None = None, valid_to: float | None = None,
                     tags: tuple[str, ...] = ()) -> MemoryRef:
        if scope_type not in SCOPES:
            raise ValueError(f"invalid scope_type {scope_type!r}; must be in {SCOPES}")
        if caller.org_id != self.org_id:
            raise PermissionError("cross-tenant memory write denied")
        if not key or not content.strip():
            raise ValueError("memory key and content must be non-empty")
        # M6: high-impact writes (org or team scope) default to pending review unless caller has admin/approver role
        needs_approval = require_review if require_review is not None else (
            scope_type in ("org", "team") and "memory_approver" not in caller.roles and "admin" not in caller.roles
        )
        initial_state = "pending" if needs_approval else "approved"
        engine_item_id, invalidated_engine_ids = self.engine.put(
            self.org_id, key, content, tags=tuple(tags), valid_from=valid_from, valid_to=valid_to
        )
        now = valid_from if valid_from is not None else time.time()
        for ref_id, existing_ref in list(self._memory_refs.items()):
            if existing_ref.engine_item_id in invalidated_engine_ids and existing_ref.valid_to is None:
                self._memory_refs[ref_id] = replace(existing_ref, valid_to=now)

        ref = MemoryRef(
            id=uuid7(),
            org_id=self.org_id,
            engine=self.engine.name,
            engine_item_id=engine_item_id,
            scope_type=scope_type,
            scope_id=scope_id,
            key=key,
            content=content,
            source_event_id=source_event_id,
            source_id=source_id,
            data_class=data_class,
            review_state=initial_state,
            valid_from=now,
            valid_to=valid_to,
            tags=tuple(tags),
        )
        self._memory_refs[ref.id] = ref
        return ref

    def review_memory(self, memory_ref_id: str, new_state: str, reviewer: Principal) -> MemoryRef:
        """Enforces ERD Section 4 state transitions: pending -> approved | rejected, approved -> revoked."""
        if reviewer.org_id != self.org_id:
            raise PermissionError("cross-tenant review denied")
        ref = self._memory_refs.get(memory_ref_id)
        if ref is None:
            raise KeyError(f"memory_ref {memory_ref_id} not found")
        if new_state not in ALLOWED_REVIEW_TRANSITIONS.get(ref.review_state, set()):
            raise ValueError(f"invalid memory review transition: {ref.review_state} -> {new_state}")
        updated = replace(ref, review_state=new_state)
        self._memory_refs[memory_ref_id] = updated
        return updated

    def revoke_by_source(self, source_id: str) -> list[str]:
        """PRD M7: losing source access flips derived memory_ref rows to revoked and hides them at retrieval."""
        revoked_ids: list[str] = []
        for mid, ref in list(self._memory_refs.items()):
            if ref.source_id == source_id and ref.review_state != "revoked":
                self._memory_refs[mid] = replace(ref, review_state="revoked")
                revoked_ids.append(mid)
        return revoked_ids

    # ---- Permission-Filtered Retrieval & Audit (M2, M3, M5, M10) ------------------------------------------
    def _permitted_knowledge(self, caller: Principal, at_time: float | None = None) -> list[KnowledgeItem]:
        allowed: list[KnowledgeItem] = []
        for versions in self._knowledge.values():
            effective = [v for v in versions if v.is_effective(at_time)]
            if not effective:
                continue
            item = effective[-1]
            if item.org_id != self.org_id or caller.org_id != self.org_id:
                continue
            if not caller.can_access_clearance(item.data_class):
                continue
            if item.allowed_principals and caller.id not in item.allowed_principals:
                continue
            if item.allowed_teams and not (set(item.allowed_teams) & set(caller.teams)):
                continue
            allowed.append(item)
        return allowed

    def _permitted_memory_refs(self, caller: Principal, agent_id: str | None = None,
                               at_time: float | None = None) -> dict[str, MemoryRef]:
        allowed: dict[str, MemoryRef] = {}
        for ref in self._memory_refs.values():
            if ref.org_id != self.org_id or caller.org_id != self.org_id:
                continue
            if not ref.is_active(at_time):
                continue
            if not caller.can_access_clearance(ref.data_class):
                continue
            if ref.scope_type == "org" and ref.scope_id == self.org_id:
                allowed[ref.engine_item_id] = ref
            elif ref.scope_type == "team" and ref.scope_id in caller.teams:
                allowed[ref.engine_item_id] = ref
            elif ref.scope_type == "user" and ref.scope_id == caller.id:
                allowed[ref.engine_item_id] = ref
            elif ref.scope_type == "agent" and agent_id and ref.scope_id == agent_id:
                allowed[ref.engine_item_id] = ref
        return allowed

    def retrieve(self, query: str, caller: Principal, run_id: str, step_id: str | None = None,
                 agent_id: str | None = None, limit: int = 6, at_time: float | None = None) -> list[RetrievedItem]:
        """Permission-filters before ranking (M3), orders by PRD source priority (policy -> docs -> data -> code -> memory),
        marks every item `untrusted` (M5), and writes a `memory.access` event to the log (M10)."""
        if caller.org_id != self.org_id:
            raise PermissionError("cross-tenant retrieval denied")
        if self._degraded:
            self.log.append(self.org_id, run_id, "memory.access", {
                "caller_principal_id": caller.id,
                "query": query,
                "status": "degraded",
                "items": [],
                "trust": "untrusted",
            }, step_id=step_id)
            return []

        results: list[RetrievedItem] = []

        # 1. Filter knowledge items by permissions BEFORE scoring/ranking (PRD M3)
        for kitem in self._permitted_knowledge(caller, at_time=at_time):
            score = _lexical_score(query, f"{kitem.title} {kitem.content}", (kitem.kind, kitem.external_id))
            if score > 0:
                results.append(RetrievedItem(
                    item_id=kitem.id,
                    item_type="knowledge",
                    kind=kitem.kind,
                    title_or_key=kitem.title,
                    content=kitem.content,
                    version_no=kitem.version_no,
                    citation=kitem.citation,
                    score=score,
                    priority_rank=kitem.priority_rank,
                    trust="untrusted",
                ))

        # 2. Filter memory refs by scope & permissions BEFORE engine search (PRD M2, M3)
        permitted_refs = self._permitted_memory_refs(caller, agent_id=agent_id, at_time=at_time)
        if permitted_refs:
            engine_hits = self.engine.search(self.org_id, query, set(permitted_refs.keys()), limit=limit)
            for eid, score in engine_hits:
                ref = permitted_refs[eid]
                results.append(RetrievedItem(
                    item_id=ref.id,
                    item_type="memory",
                    kind=f"memory:{ref.scope_type}",
                    title_or_key=ref.key,
                    content=ref.content,
                    version_no=1,
                    citation=f"[memory:{ref.scope_type}/{ref.key}@{ref.engine}]",
                    score=score,
                    priority_rank=2,
                    trust="untrusted",
                ))

        # Sort by PRD source priority rank first (policy=1 first), then by descending relevance score
        results.sort(key=lambda r: (r.priority_rank, -r.score))
        top = results[:limit]

        # 3. Audit what was retrieved for which step and version (PRD M10, ERD memory_access)
        self.log.append(self.org_id, run_id, "memory.access", {
            "caller_principal_id": caller.id,
            "query": query,
            "filter_applied": {
                "org_id": self.org_id,
                "teams": list(caller.teams),
                "max_clearance": caller.max_clearance,
                "agent_id": agent_id,
            },
            "items": [
                {
                    "item_id": r.item_id,
                    "item_type": r.item_type,
                    "kind": r.kind,
                    "version_no": r.version_no,
                    "citation": r.citation,
                    "rank": idx + 1,
                    "score": r.score,
                    "trust": "untrusted",
                }
                for idx, r in enumerate(top)
            ],
            "trust": "untrusted",
        }, step_id=step_id)
        return top
