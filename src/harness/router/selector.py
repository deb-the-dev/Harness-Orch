"""Tier 2: choose among the models the constraints left. Pluggable (PRD R5); open routers can run as selectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import ModelSpec


@dataclass(frozen=True)
class Scored:
    model: ModelSpec
    score: float
    source: str          # where the evidence came from, for audit (for example "public_prior")


class Selector(Protocol):
    version: str

    def rank(self, task_class: str, step_role: str, candidates: list[ModelSpec], stronger_than: str | None = None) -> list[Scored]: ...


class StaticPriorSelector:
    """Capability prior per (task class, step role) and model, from public-benchmark results.

    Ranking: models within `tolerance` of the best prior are 'adequate'; the cheapest adequate model goes first,
    then the remaining models by descending prior (they are the failover order). Missing priors rank last.
    """

    def __init__(self, priors: dict[tuple[str, str], dict[str, float]], tolerance: float = 0.03, version: str = "static-prior-v1"):
        self.priors, self.tolerance, self.version = priors, tolerance, version

    def rank(self, task_class, step_role, candidates, stronger_than=None):
        table = self.priors.get((task_class, step_role)) or self.priors.get((task_class, "*")) or {}
        scored = [Scored(m, table.get(m.id, 0.0), "public_prior" if m.id in table else "no_evidence") for m in candidates]
        if stronger_than is not None:                       # escalation: only models with a higher prior than the failed one
            floor = table.get(stronger_than, 0.0)
            scored = [s for s in scored if s.score > floor]
            return sorted(scored, key=lambda s: (s.model.price_out_per_mtok, -s.score))
        if not scored:
            return []
        top = max(s.score for s in scored)
        adequate = [s for s in scored if s.score >= top - self.tolerance]
        first = min(adequate, key=lambda s: (s.model.price_out_per_mtok, -s.score))
        rest = sorted((s for s in scored if s is not first), key=lambda s: (-s.score, s.model.price_out_per_mtok))
        return [first, *rest]
