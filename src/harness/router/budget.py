"""Spend accounting. Spend is rebuilt from the log; a call is reserved before it is made (reserve, call, reconcile)."""
from __future__ import annotations

from dataclasses import dataclass

from ..events import EventLog


class BudgetExceeded(Exception):
    pass


@dataclass
class Reservation:
    run_id: str
    est_usd: float
    released: bool = False


class BudgetLedger:
    def __init__(self, log: EventLog, org_id: str):
        self.log, self.org_id = log, org_id
        self._reserved: dict[str, float] = {}

    def spent(self, run_id: str) -> float:
        return sum(e.payload.get("cost_usd", 0.0) for e in self.log.read(self.org_id, run_id) if e.type == "model.response")

    def committed_plus_reserved(self, run_id: str) -> float:
        return self.spent(run_id) + self._reserved.get(run_id, 0.0)

    def reserve(self, run_id: str, est_usd: float, limit_usd: float) -> Reservation:
        if self.committed_plus_reserved(run_id) + est_usd > limit_usd:
            raise BudgetExceeded(f"run {run_id}: {self.committed_plus_reserved(run_id):.6f} + {est_usd:.6f} exceeds {limit_usd:.6f}")
        self._reserved[run_id] = self._reserved.get(run_id, 0.0) + est_usd
        return Reservation(run_id, est_usd)

    def release(self, r: Reservation) -> None:
        if not r.released:
            r.released = True
            self._reserved[r.run_id] = max(0.0, self._reserved.get(r.run_id, 0.0) - r.est_usd)
