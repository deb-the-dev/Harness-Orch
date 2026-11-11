from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    """Per-model quirks the router applies when it builds a request (PRD C3).

    reasoning_blocks: "keep" lets a model re-read its own reasoning blocks from earlier turns; "strip" never sends any.
    Blocks written by a different provider are always stripped: a provider can reject history it did not produce.
    """
    reasoning_blocks: str = "strip"          # "keep" | "strip"
    prompt_cache: bool = False               # provider-side prompt caching: switching away from this model loses the cache
    prompt_prefix: str = ""                  # added to the front of the system message
    excluded_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    """One row of the model catalogue (ERD: model plus its provider endpoint)."""
    id: str                       # our stable id, used in policy data and logs
    litellm_model: str            # for example "openai/mock-primary" or "anthropic/<model>"
    region: str
    price_in_per_mtok: float
    price_out_per_mtok: float
    api_base: str | None = None
    api_key: str | None = None    # for tests; production resolves a secret reference (ERD provider.auth_ref)
    context_window: int = 128_000
    supports_tools: bool = True
    version: str = ""             # part of the identity: a silent provider update must not reuse old results
    provider: str = ""            # for example "openai", "anthropic": decides whether history blocks are portable
    profile: ModelProfile = field(default_factory=ModelProfile)

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price_in_per_mtok + completion_tokens * self.price_out_per_mtok) / 1_000_000
