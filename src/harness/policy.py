"""Policy hooks: OPA as an out-of-process decision service (plus an in-process Rego-equivalent fallback for embedded FDE runs), called fail closed.

Decided in spike 1: anything other than a well-formed answer from OPA is a block, because Rego
treats undefined lookups and malformed input as allow unless the policy says otherwise.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from .events import EventLog

HOOKS = ("pre_model", "post_model", "pre_tool", "post_tool")
ACTION_CLASSES = ("read", "internal_write", "external_comm", "payment")
DATA_CLASSES = ("public", "internal", "confidential", "restricted")
VERDICTS = ("allow", "modify", "require_approval", "block")
APPROVAL_STATES = ("none", "pending", "approved", "rejected")


@dataclass(frozen=True)
class Decision:
    verdict: str
    reasons: tuple[str, ...] = ()
    policy_version: str = ""
    latency_ms: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_hook_input(inp: Any) -> bool:
    """Same checks the Rego policy makes, applied before the call so a bad input never reaches OPA."""
    if not isinstance(inp, dict) or inp.get("action_class") not in ACTION_CLASSES:
        return False
    hook = inp.get("hook")
    g = lambda *p: _dig(inp, p)
    if hook == "pre_model":
        return (all(_is_num(g("spend", k)) for k in ("run_usd", "est_call_usd", "run_limit_usd"))
                and inp.get("data_class") in DATA_CLASSES and isinstance(inp.get("model"), str)
                and isinstance(inp.get("model_region"), str) and isinstance(inp.get("pack"), str))
    if hook == "post_model":
        return isinstance(g("output", "contains_pii"), bool)
    if hook == "pre_tool":
        return isinstance(inp.get("tool"), str) and g("approval", "state") in APPROVAL_STATES and isinstance(inp.get("pack"), str)
    if hook == "post_tool":
        return _is_num(g("output", "bytes"))
    return False


def _dig(d: Any, path: tuple) -> Any:
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def policy_version(*files: str | Path) -> str:
    h = hashlib.sha256()
    for f in sorted(map(str, files)):
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]


def build_policy_data(packs: dict[str, dict[str, list[str]]], max_tool_result_bytes: int = 1_048_576) -> dict:
    """packs: name -> {"tools": [...], "models": [...], "restricted_regions": [...]}"""
    return {"packs": packs, "limits": {"max_tool_result_bytes": max_tool_result_bytes}}


class EmbeddedPolicyClient:
    """In-process evaluator implementing 100% parity with `policies/hooks.rego` (Spike 1 parity rules + kill switch).

    Used when running FDE deployments or control-plane tests in environments without an external OPA daemon,
    while preserving identical fail-closed semantics, verdict precedence (`block > require_approval > modify > allow`),
    and sorted reason tuples.
    """

    def __init__(self, data: dict[str, Any], version: str = "embedded-rego-v1"):
        self.data = data
        self.version = version
        self.killed_packs: set[str] = set()
        self.org_killed: bool = False

    def set_kill_switch(self, active: bool, pack: str | None = None) -> None:
        if pack is None:
            self.org_killed = active
        elif active:
            self.killed_packs.add(pack)
        else:
            self.killed_packs.discard(pack)

    async def decide(self, inp: dict[str, Any]) -> Decision:
        t0 = time.perf_counter()
        if not validate_hook_input(inp):
            return Decision("block", ("malformed_input",), self.version, (time.perf_counter() - t0) * 1000)

        pack_name = inp.get("pack", "")
        if self.org_killed or pack_name in self.killed_packs:
            return Decision("block", ("kill_switch_active",), self.version, (time.perf_counter() - t0) * 1000)

        packs_map = self.data.get("packs", {})
        pack = packs_map.get(pack_name, {"tools": [], "models": [], "restricted_regions": []})
        max_bytes = self.data.get("limits", {}).get("max_tool_result_bytes", 1_048_576)

        block_reasons: set[str] = set()
        approval_reasons: set[str] = set()
        modify_reasons: set[str] = set()

        hook = inp["hook"]
        action_class = inp["action_class"]

        if action_class == "payment":
            block_reasons.add("payment_out_of_scope")

        if hook == "pre_tool":
            if inp["tool"] not in pack.get("tools", []):
                block_reasons.add("tool_not_granted")
            approval_state = inp["approval"]["state"]
            if action_class == "external_comm" and approval_state == "rejected":
                block_reasons.add("approval_rejected")
            if action_class == "external_comm" and approval_state in ("none", "pending"):
                approval_reasons.add("external_comm_needs_approval")

        elif hook == "pre_model":
            spend = inp["spend"]
            if spend["run_usd"] + spend["est_call_usd"] > spend["run_limit_usd"]:
                block_reasons.add("budget_hard_limit")
            data_class = inp["data_class"]
            if data_class == "restricted" and inp["model_region"] not in pack.get("restricted_regions", []):
                block_reasons.add("residency")
            if data_class in ("confidential", "restricted") and inp["model"] not in pack.get("models", []):
                block_reasons.add("model_not_allowed")

        elif hook == "post_model":
            if inp["output"]["contains_pii"]:
                modify_reasons.add("redact_pii")

        elif hook == "post_tool":
            if inp["output"]["bytes"] > max_bytes:
                modify_reasons.add("truncate_result")

        if block_reasons:
            verdict = "block"
        elif approval_reasons:
            verdict = "require_approval"
        elif modify_reasons:
            verdict = "modify"
        else:
            verdict = "allow"

        reasons = tuple(sorted(block_reasons | approval_reasons | modify_reasons))
        return Decision(verdict, reasons, self.version, (time.perf_counter() - t0) * 1000)

    async def close(self) -> None:
        return None


class PolicyClient:
    def __init__(self, base_url: str | None = None, unix_socket: str | None = None, timeout_s: float = 0.25,
                 version: str = "", path: str = "/v1/data/hooks/decision"):
        self.base_url = "http://localhost" if unix_socket else (base_url or "http://127.0.0.1:8181")
        self.unix_socket, self.timeout_s, self.version, self.path = unix_socket, timeout_s, version, path
        self._session: aiohttp.ClientSession | None = None

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            conn = aiohttp.UnixConnector(path=self.unix_socket) if self.unix_socket else aiohttp.TCPConnector()
            self._session = aiohttp.ClientSession(connector=conn)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _block(self, reason: str, t0: float) -> Decision:
        return Decision("block", (reason,), self.version, (time.perf_counter() - t0) * 1000)

    async def decide(self, inp: dict[str, Any]) -> Decision:
        t0 = time.perf_counter()
        if not validate_hook_input(inp):
            return self._block("malformed_input", t0)
        try:
            s = await self._sess()
            async with s.post(self.base_url + self.path, data=json.dumps({"input": inp}).encode(),
                              headers={"Content-Type": "application/json"},
                              timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as r:
                if r.status != 200:
                    return self._block("policy_unavailable", t0)
                body = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError):
            return self._block("policy_unavailable", t0)
        res = body.get("result") if isinstance(body, dict) else None
        if not isinstance(res, dict) or res.get("verdict") not in VERDICTS or not isinstance(res.get("reasons", []), list):
            return self._block("policy_undefined", t0)      # no result, or a shape we do not understand
        return Decision(res["verdict"], tuple(map(str, res.get("reasons", []))), self.version, (time.perf_counter() - t0) * 1000)


class HookRunner:
    """Calls the policy client and writes each decision to the log (ERD: policy_decision)."""

    def __init__(self, client: PolicyClient | EmbeddedPolicyClient, log: EventLog, org_id: str):
        self.client, self.log, self.org_id = client, log, org_id

    async def check(self, run_id: str, step_id: str | None, inp: dict[str, Any]) -> Decision:
        d = await self.client.decide(inp)
        self.log.append(self.org_id, run_id, "policy.decision", {
            "hook": inp.get("hook") if isinstance(inp, dict) else None, "verdict": d.verdict, "reasons": list(d.reasons),
            "policy_version": d.policy_version, "latency_ms": round(d.latency_ms, 3),
            "input_hash": hashlib.sha256(json.dumps(inp, sort_keys=True, default=str).encode()).hexdigest()[:16],
        }, step_id=step_id)
        return d


@dataclass
class OpaProcess:
    """Starts a local OPA server for development and tests. Production runs OPA as a sidecar or service."""
    policy_files: list[str]
    data: dict
    opa_bin: str = field(default_factory=lambda: os.environ.get("OPA_BIN", str(Path(__file__).resolve().parents[2] / ".tools/opa/opa")))
    port: int = 0
    _proc: subprocess.Popen | None = None
    _tmp: tempfile.TemporaryDirectory | None = None

    def start(self) -> "OpaProcess":
        self._tmp = tempfile.TemporaryDirectory()
        data_path = Path(self._tmp.name) / "data.json"
        data_path.write_text(json.dumps(self.data))
        if not self.port:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                self.port = s.getsockname()[1]
        self._proc = subprocess.Popen([self.opa_bin, "run", "--server", "--addr", f"127.0.0.1:{self.port}", *self.policy_files, str(data_path)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return self
            except OSError:
                time.sleep(0.05)
        self.stop()
        raise RuntimeError("OPA did not start")

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        if self._tmp:
            self._tmp.cleanup()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"
