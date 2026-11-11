"""Tool runtime. Policy decides whether a tool may run; this layer validates and bounds the call."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .policy import ACTION_CLASSES


@dataclass(frozen=True)
class ToolSpec:
    name: str
    action_class: str                      # read | internal_write | external_comm (payment is refused at registration)
    fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    required: tuple[str, ...] = ()
    timeout_s: float = 10.0
    idempotent: bool = False               # a lost outcome may be retried only if True; otherwise a human decides


class ToolError(Exception):
    pass


class ToolRuntime:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.action_class not in ACTION_CLASSES:
            raise ValueError(f"unknown action class {spec.action_class!r}")
        if spec.action_class == "payment":
            raise ValueError("payment tools are out of scope for v1 and cannot be registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"unknown tool {name}")
        missing = [k for k in spec.required if k not in args]
        if missing:
            raise ToolError(f"missing arguments: {missing}")
        try:
            return await asyncio.wait_for(spec.fn(args), spec.timeout_s)
        except asyncio.TimeoutError as e:
            raise ToolError("tool timed out") from e


def result_bytes(result: dict[str, Any]) -> int:
    return len(json.dumps(result, default=str).encode())
