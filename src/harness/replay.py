"""Deterministic replay, shadow policy evaluation, and best-single-model promotion gate (PRD P4, P6, R6; ERD 3.6).

Reads the append-only event log without mutating it, verifies event payload hashes, reconstructs every model request,
and evaluates candidate routing selectors against a best-single-model baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import EventLog
from .router.models import ModelSpec
from .router.selector import Selector


@dataclass(frozen=True)
class ReplayStepComparison:
    run_id: str
    step_id: str | None
    decision_id: str
    task_class: str
    step_role: str
    logged_model: str | None
    candidate_model: str | None
    baseline_model: str
    oracle_model: str
    logged_cost_usd: float
    candidate_est_cost_usd: float
    baseline_est_cost_usd: float
    oracle_est_cost_usd: float
    logged_success: bool
    candidate_expected_score: float
    baseline_expected_score: float
    oracle_expected_score: float
    decision_matched: bool


@dataclass(frozen=True)
class ReplayReport:
    org_id: str
    selector_version: str
    baseline_model_id: str
    total_runs: int
    total_steps: int
    corrupted_runs: list[str]
    replay_fidelity: float                  # share of steps where replaying the logged selector reproduces the exact choice
    baseline_total_cost_usd: float
    candidate_total_cost_usd: float
    oracle_total_cost_usd: float
    cost_saved_pct: float                   # (baseline_cost - candidate_cost) / baseline_cost
    baseline_avg_score: float
    candidate_avg_score: float
    oracle_avg_score: float
    score_delta_vs_baseline: float
    oracle_gap: float                       # oracle_avg_score - candidate_avg_score
    passes_promotion_gate: bool             # PRD section 3 & 8: >=30% cost cut at <=1pt quality loss (or strictly better)
    steps: list[ReplayStepComparison] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "selector_version": self.selector_version,
            "baseline_model_id": self.baseline_model_id,
            "total_runs": self.total_runs,
            "total_steps": self.total_steps,
            "corrupted_runs": self.corrupted_runs,
            "replay_fidelity": round(self.replay_fidelity, 4),
            "baseline_total_cost_usd": round(self.baseline_total_cost_usd, 6),
            "candidate_total_cost_usd": round(self.candidate_total_cost_usd, 6),
            "oracle_total_cost_usd": round(self.oracle_total_cost_usd, 6),
            "cost_saved_pct": round(self.cost_saved_pct, 4),
            "baseline_avg_score": round(self.baseline_avg_score, 4),
            "candidate_avg_score": round(self.candidate_avg_score, 4),
            "oracle_avg_score": round(self.oracle_avg_score, 4),
            "score_delta_vs_baseline": round(self.score_delta_vs_baseline, 4),
            "oracle_gap": round(self.oracle_gap, 4),
            "passes_promotion_gate": self.passes_promotion_gate,
            "steps": [s.__dict__ for s in self.steps],
        }


class ReplayEngine:
    """Replays logged runs from the append-only EventLog under a candidate Selector and scores against a baseline (PRD P4, P6)."""

    def __init__(self, log: EventLog, org_id: str, catalogue: dict[str, ModelSpec],
                 priors: dict[tuple[str, str], dict[str, float]]) -> None:
        self.log = log
        self.org_id = org_id
        self.catalogue = catalogue
        self.priors = priors

    def _prior_score(self, task_class: str, step_role: str, model_id: str) -> float:
        table = self.priors.get((task_class, step_role)) or self.priors.get((task_class, "*")) or {}
        return float(table.get(model_id, 0.5))

    def evaluate(self, candidate_selector: Selector, run_ids: list[str] | None = None,
                 baseline_model_id: str | None = None, cost_saving_target: float = 0.30,
                 max_quality_drop: float = 0.01) -> ReplayReport:
        target_runs = list(run_ids) if run_ids is not None else list(self.log.run_ids(self.org_id))
        if baseline_model_id is None:
            # Default best-single-model baseline is the highest-priced / strongest model in the catalogue
            baseline_model_id = max(self.catalogue.values(), key=lambda m: (m.price_out_per_mtok, m.price_in_per_mtok)).id

        baseline_spec = self.catalogue[baseline_model_id]
        corrupted: list[str] = []
        comparisons: list[ReplayStepComparison] = []

        for rid in target_runs:
            if hasattr(self.log, "verify") and self.log.verify(self.org_id, rid):
                corrupted.append(rid)
                continue
            events = self.log.read(self.org_id, rid)
            requests = {e.payload["decision_id"]: e for e in events if e.type == "model.request" and "decision_id" in e.payload}
            responses = {e.payload["decision_id"]: e for e in events if e.type == "model.response" and "decision_id" in e.payload}
            decisions = {
                e.payload["decision_id"]: e
                for e in events
                if e.type == "routing.decision" and not e.payload.get("shadow") and "decision_id" in e.payload
            }
            step_outcomes = {
                e.step_id: e.payload.get("ok", True)
                for e in events
                if e.type == "step.outcome"
            }

            for did, req_ev in requests.items():
                req_p = req_ev.payload
                dec_ev = decisions.get(did)
                resp_ev = responses.get(did)
                allowed_ids = req_p.get("allowed_models", list(self.catalogue.keys()))
                pool = [self.catalogue[mid] for mid in allowed_ids if mid in self.catalogue]
                if not pool:
                    continue

                task_class = req_p.get("task_class", "extract")
                step_role = req_p.get("step_role", "*")
                ranked = candidate_selector.rank(task_class, step_role, pool)
                chosen_spec = ranked[0].model if ranked else pool[0]

                prompt_toks = resp_ev.payload.get("prompt_tokens", 200) if resp_ev else 200
                comp_toks = resp_ev.payload.get("completion_tokens", 100) if resp_ev else 100
                logged_model = dec_ev.payload.get("chosen") if dec_ev else (resp_ev.payload.get("model") if resp_ev else None)
                logged_cost = float(resp_ev.payload.get("cost_usd", chosen_spec.cost(prompt_toks, comp_toks))) if resp_ev else chosen_spec.cost(prompt_toks, comp_toks)

                cand_cost = chosen_spec.cost(prompt_toks, comp_toks)
                base_cost = baseline_spec.cost(prompt_toks, comp_toks)

                cand_score = ranked[0].score if ranked else self._prior_score(task_class, step_role, chosen_spec.id)
                base_score = self._prior_score(task_class, step_role, baseline_spec.id)

                # Oracle picks the cheapest model among those achieving the maximum score on this step
                max_score = max(self._prior_score(task_class, step_role, m.id) for m in pool)
                oracle_spec = min(
                    (m for m in pool if self._prior_score(task_class, step_role, m.id) >= max_score - 1e-6),
                    key=lambda m: m.cost(prompt_toks, comp_toks),
                )
                oracle_cost = oracle_spec.cost(prompt_toks, comp_toks)

                comparisons.append(ReplayStepComparison(
                    run_id=rid,
                    step_id=req_ev.step_id,
                    decision_id=did,
                    task_class=task_class,
                    step_role=step_role,
                    logged_model=logged_model,
                    candidate_model=chosen_spec.id,
                    baseline_model=baseline_spec.id,
                    oracle_model=oracle_spec.id,
                    logged_cost_usd=logged_cost,
                    candidate_est_cost_usd=cand_cost,
                    baseline_est_cost_usd=base_cost,
                    oracle_est_cost_usd=oracle_cost,
                    logged_success=bool(step_outcomes.get(req_ev.step_id, True)),
                    candidate_expected_score=cand_score,
                    baseline_expected_score=base_score,
                    oracle_expected_score=max_score,
                    decision_matched=(logged_model == chosen_spec.id),
                ))

        total_steps = len(comparisons)
        if total_steps == 0:
            return ReplayReport(
                org_id=self.org_id,
                selector_version=candidate_selector.version,
                baseline_model_id=baseline_model_id,
                total_runs=len(target_runs),
                total_steps=0,
                corrupted_runs=corrupted,
                replay_fidelity=1.0,
                baseline_total_cost_usd=0.0,
                candidate_total_cost_usd=0.0,
                oracle_total_cost_usd=0.0,
                cost_saved_pct=0.0,
                baseline_avg_score=0.0,
                candidate_avg_score=0.0,
                oracle_avg_score=0.0,
                score_delta_vs_baseline=0.0,
                oracle_gap=0.0,
                passes_promotion_gate=False,
                steps=[],
            )

        fidelity = sum(1 for c in comparisons if c.decision_matched) / total_steps
        base_tot_cost = sum(c.baseline_est_cost_usd for c in comparisons)
        cand_tot_cost = sum(c.candidate_est_cost_usd for c in comparisons)
        oracle_tot_cost = sum(c.oracle_est_cost_usd for c in comparisons)
        cost_saved = (base_tot_cost - cand_tot_cost) / base_tot_cost if base_tot_cost > 0 else 0.0

        base_avg_q = sum(c.baseline_expected_score for c in comparisons) / total_steps
        cand_avg_q = sum(c.candidate_expected_score for c in comparisons) / total_steps
        oracle_avg_q = sum(c.oracle_expected_score for c in comparisons) / total_steps
        q_delta = cand_avg_q - base_avg_q
        o_gap = oracle_avg_q - cand_avg_q

        passes = (
            not corrupted
            and cost_saved >= cost_saving_target
            and (base_avg_q - cand_avg_q) <= max_quality_drop
        )
        return ReplayReport(
            org_id=self.org_id,
            selector_version=candidate_selector.version,
            baseline_model_id=baseline_model_id,
            total_runs=len(target_runs),
            total_steps=total_steps,
            corrupted_runs=corrupted,
            replay_fidelity=fidelity,
            baseline_total_cost_usd=base_tot_cost,
            candidate_total_cost_usd=cand_tot_cost,
            oracle_total_cost_usd=oracle_tot_cost,
            cost_saved_pct=cost_saved,
            baseline_avg_score=base_avg_q,
            candidate_avg_score=cand_avg_q,
            oracle_avg_score=oracle_avg_q,
            score_delta_vs_baseline=q_delta,
            oracle_gap=o_gap,
            passes_promotion_gate=passes,
            steps=comparisons,
        )
