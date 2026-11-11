"""Circuit breaker per model (ERD: model_health). Neither gateway in spike 2 stopped calling a failing provider."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class _State:
    failures: int = 0
    opened_at: float | None = None
    trial_in_flight: bool = False


class CircuitBreaker:
    """Closed until `threshold` consecutive failures, then open for `cooldown_s`, then half-open (one trial)."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 10.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown_s, self.clock = threshold, cooldown_s, clock
        self._s: dict[str, _State] = {}

    def state(self, model_id: str) -> str:
        s = self._s.get(model_id)
        if s is None or s.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - s.opened_at >= self.cooldown_s else "open"

    def allow(self, model_id: str) -> bool:
        st = self.state(model_id)
        if st == "closed":
            return True
        if st == "half_open":
            s = self._s[model_id]
            if not s.trial_in_flight:
                s.trial_in_flight = True     # let exactly one request test the model
                return True
        return False

    def record_success(self, model_id: str) -> None:
        self._s[model_id] = _State()

    def record_failure(self, model_id: str) -> None:
        s = self._s.setdefault(model_id, _State())
        s.failures += 1
        s.trial_in_flight = False
        if s.failures >= self.threshold or s.opened_at is not None:
            s.opened_at = self.clock()
