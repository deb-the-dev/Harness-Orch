"""Artifact store and result offload (PRD C1, C4).

Large tool results and pre-compaction history are stored as immutable, content-addressed artifacts. The model sees a short
preview and a handle, and reads ranges on demand. Because the address is the SHA-256 of the content, a handle can never point
at changed bytes (a path to a mutable file can), and reads verify the hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .events import EventLog
from .tools import ToolSpec

PREVIEW_HEAD, PREVIEW_TAIL, PREVIEW_LINE_CHARS = 5, 5, 1000
DEFAULT_THRESHOLD_BYTES = 16_000
MAX_READ_LINES = 500


class IntegrityError(Exception):
    pass


class ArtifactNotFound(Exception):
    pass


@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    size_bytes: int
    lines: int
    media_type: str = "text/plain"


class ArtifactStore(Protocol):
    def put(self, org_id: str, data: bytes, media_type: str = "text/plain") -> ArtifactRef: ...
    def get(self, org_id: str, sha256: str) -> bytes: ...


class LocalArtifactStore:
    """Files under root/<org>/<aa>/<sha256>. An object-store implementation follows the same protocol."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, org_id: str, sha256: str) -> Path:
        if not (len(sha256) == 64 and all(c in "0123456789abcdef" for c in sha256)):
            raise ArtifactNotFound("malformed artifact id")           # also blocks path traversal through the id
        safe_org = hashlib.sha256(org_id.encode()).hexdigest()[:24]   # never use a caller-supplied string as a path part
        return self.root / safe_org / sha256[:2] / sha256

    def put(self, org_id: str, data: bytes, media_type: str = "text/plain") -> ArtifactRef:
        sha = hashlib.sha256(data).hexdigest()
        path = self._path(org_id, sha)
        if not path.exists():                                         # idempotent: the same bytes are stored once per org
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return ArtifactRef(sha, len(data), data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0), media_type)

    def get(self, org_id: str, sha256: str) -> bytes:
        path = self._path(org_id, sha256)
        if not path.exists():
            raise ArtifactNotFound(sha256)                            # another org's artifact is simply not found here
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise IntegrityError(f"artifact {sha256} does not match its hash")
        return data

    def read_range(self, org_id: str, sha256: str, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        lines = self.get(org_id, sha256).decode("utf-8", errors="replace").splitlines()
        offset, limit = max(0, int(offset)), max(1, min(int(limit), MAX_READ_LINES))
        chunk = lines[offset:offset + limit]
        nxt = offset + len(chunk)
        return {"sha256": sha256, "offset": offset, "lines": chunk, "total_lines": len(lines), "next_offset": nxt if nxt < len(lines) else None}


def preview(text: str, head: int = PREVIEW_HEAD, tail: int = PREVIEW_TAIL, line_chars: int = PREVIEW_LINE_CHARS) -> str:
    """First and last lines, numbered, each line capped, with an explicit marker for what was left out."""
    lines = text.splitlines() or [""]
    cap = lambda s: s if len(s) <= line_chars else s[:line_chars] + f"... [{len(s) - line_chars} more characters]"
    if len(lines) <= head + tail:
        return "\n".join(f"{i + 1:>6}  {cap(l)}" for i, l in enumerate(lines))
    top = [f"{i + 1:>6}  {cap(l)}" for i, l in enumerate(lines[:head])]
    bottom = [f"{len(lines) - tail + i + 1:>6}  {cap(l)}" for i, l in enumerate(lines[-tail:])]
    return "\n".join(top + [f"        ... [{len(lines) - head - tail} lines not shown] ..."] + bottom)


class Offloader:
    """Stores an oversized result as an artifact and returns the stub the model will see. Both are logged."""

    def __init__(self, store: LocalArtifactStore, log: EventLog, org_id: str, threshold_bytes: int = DEFAULT_THRESHOLD_BYTES):
        self.store, self.log, self.org_id, self.threshold = store, log, org_id, threshold_bytes

    def maybe_offload(self, run_id: str, step_id: str | None, result: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(result, default=str, indent=1)
        data = text.encode()
        if len(data) <= self.threshold:
            return result
        ref = self.store.put(self.org_id, data, "application/json")
        shown = preview(text)
        self.log.append(self.org_id, run_id, "artifact.created", {"sha256": ref.sha256, "size_bytes": ref.size_bytes, "lines": ref.lines,
                                                                  "media_type": ref.media_type, "reason": "large_tool_result"}, step_id=step_id)
        return {"ok": result.get("ok", True), "offloaded": True,
                "artifact": {"sha256": ref.sha256, "bytes": ref.size_bytes, "lines": ref.lines}, "preview": shown,
                "note": (f"Result too large to show ({ref.size_bytes} bytes, {ref.lines} lines). Only the lines above are visible. "
                         f"Call read_artifact with sha256={ref.sha256} and an offset to read more; read in ranges.")}


def make_read_artifact_tool(store: LocalArtifactStore, org_id: str) -> ToolSpec:
    """The model's way to page through an artifact. It is an ordinary tool: policy grants it per pack, every call is logged."""
    async def read_artifact(args: dict[str, Any]) -> dict[str, Any]:
        try:
            return store.read_range(org_id, args["sha256"], args.get("offset", 0), args.get("limit", 100))
        except (ArtifactNotFound, IntegrityError) as e:
            return {"error": type(e).__name__}
    return ToolSpec("read_artifact", "read", read_artifact, required=("sha256",), idempotent=True)


def render_result(result: dict[str, Any]) -> str:
    """What a pack puts in a prompt for a stored step result: the preview and handle if it was offloaded, else the result."""
    if result.get("offloaded"):
        return f"{result['preview']}\n[{result['note']}]"
    return json.dumps(result, default=str)
