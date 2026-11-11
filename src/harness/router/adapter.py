"""Provider adapter: LiteLLM used as an in-process library (decision from spike 2).

Retries are ours, not LiteLLM's: num_retries is 0 so a failing provider fails over at once.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import litellm

from .models import ModelSpec

litellm.drop_params = True
litellm.suppress_debug_info = True


class ProviderFailover(Exception):
    """A failure that justifies trying the next allowed model: 429, 5xx, 408, timeouts, connection errors."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail = kind, detail


class ProviderFatal(Exception):
    """A failure another model would repeat: bad request, auth, content policy."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail = kind, detail


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    provider_model: str


def classify(e: Exception) -> Exception:
    status = getattr(e, "status_code", None)
    name = type(e).__name__
    if status == 429:
        return ProviderFailover("rate_limited", str(e)[:200])
    if status is not None and (status >= 500 or status == 408):
        return ProviderFailover(f"http_{status}", str(e)[:200])
    if isinstance(e, (litellm.Timeout, litellm.APIConnectionError)) or "Timeout" in name:
        return ProviderFailover("timeout" if "Timeout" in name else "connection", str(e)[:200])
    return ProviderFatal(f"http_{status}" if status else name, str(e)[:200])


class LiteLLMAdapter:
    def __init__(self, timeout_s: float = 30.0):
        self.timeout_s = timeout_s

    async def complete(self, spec: ModelSpec, messages: list[dict], max_tokens: int = 1024) -> Completion:
        t0 = time.perf_counter()
        try:
            r = await litellm.acompletion(model=spec.litellm_model, messages=messages, api_base=spec.api_base, api_key=spec.api_key,
                                          max_tokens=max_tokens, timeout=self.timeout_s, num_retries=0)
        except Exception as e:                              # noqa: BLE001  (classified, never swallowed)
            raise classify(e) from e
        u = r.usage
        return Completion(r.choices[0].message.content or "", int(u.prompt_tokens), int(u.completion_tokens),
                          (time.perf_counter() - t0) * 1000, getattr(r, "model", spec.litellm_model))
