"""Portable message history (PRD C2).

The router may send step N to one model and step N+1 to another, so history has to survive a switch:
  * provider-specific blocks (reasoning blocks) are removed unless the same provider wrote them and the profile keeps them;
  * a tool call with no recorded result gets an explicit "outcome unknown" result, never a silent gap;
  * a tool result with no matching call is dropped.
Nothing is mutated: the log keeps the original, and the report says what was changed so the change itself is logged.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .router.models import ModelSpec

REASONING_TYPES = frozenset({"thinking", "redacted_thinking", "reasoning", "reasoning_content"})
UNKNOWN_RESULT = "outcome unknown: no result was recorded for this tool call. Do not assume it succeeded or failed."
MARKERS = ("x_provider", "x_model")           # our own annotations, never sent to a provider


@dataclass(frozen=True)
class PrepareReport:
    stripped_blocks: int = 0
    patched_calls: int = 0
    dropped_orphans: int = 0
    prefix_applied: bool = False
    sent_hash: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.stripped_blocks or self.patched_calls or self.dropped_orphans or self.prefix_applied)

    def as_dict(self) -> dict[str, Any]:
        return {"stripped_blocks": self.stripped_blocks, "patched_calls": self.patched_calls,
                "dropped_orphans": self.dropped_orphans, "prefix_applied": self.prefix_applied, "sent_hash": self.sent_hash}


def _call_ids(msg: dict) -> list[str]:
    return [c.get("id") for c in (msg.get("tool_calls") or []) if isinstance(c, dict) and c.get("id")]


def prepare_messages(messages: list[dict], target: ModelSpec) -> tuple[list[dict], PrepareReport]:
    msgs = copy.deepcopy(messages)
    stripped = patched = orphans = 0
    keep_ok = target.profile.reasoning_blocks == "keep"

    # 1. provider-specific blocks
    emptied: set[int] = set()
    for i, m in enumerate(msgs):
        c = m.get("content")
        if isinstance(c, list):
            same_provider = bool(target.provider) and m.get("x_provider") == target.provider
            kept = [b for b in c if not (isinstance(b, dict) and b.get("type") in REASONING_TYPES and not (keep_ok and same_provider))]
            stripped += len(c) - len(kept)
            if c and not kept and m.get("role") == "assistant" and not m.get("tool_calls"):
                emptied.add(i)                 # nothing but reasoning: the message has no content left to send
            m["content"] = kept if kept else ""
    msgs = [m for i, m in enumerate(msgs) if i not in emptied]

    # 2. tool calls and results must pair up
    result_ids = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
    call_ids = {cid for m in msgs for cid in _call_ids(m)}
    out: list[dict] = []
    for m in msgs:
        if m.get("role") == "tool" and m.get("tool_call_id") not in call_ids:
            orphans += 1
            continue
        out.append(m)
        for cid in _call_ids(m):
            if cid not in result_ids:
                out.append({"role": "tool", "tool_call_id": cid, "content": UNKNOWN_RESULT})
                patched += 1

    # 3. per-model prompt prefix
    prefix_applied = False
    if target.profile.prompt_prefix:
        if out and out[0].get("role") == "system" and isinstance(out[0].get("content"), str):
            out[0] = {**out[0], "content": target.profile.prompt_prefix + "\n\n" + out[0]["content"]}
        else:
            out.insert(0, {"role": "system", "content": target.profile.prompt_prefix})
        prefix_applied = True

    for m in out:
        for k in MARKERS:
            m.pop(k, None)
    sent_hash = hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return out, PrepareReport(stripped, patched, orphans, prefix_applied, sent_hash)


def mark_provider(message: dict, spec: ModelSpec) -> dict:
    """Annotate an assistant message with the model that wrote it, so a later switch knows which blocks are foreign."""
    return {**message, "x_provider": spec.provider, "x_model": spec.id}
