"""Spike 6 scoring: replay a routing policy over recorded per-instance results.

Input: for every model m and instance i, a score (0..1) and a cost (USD), as recorded in the
LLMRouterBench result bundles (per-instance records hold prediction, score, tokens and cost).
No model calls are made: a policy only chooses which recorded result to use.

Metrics follow LLMRouterBench's definitions as described in its README:
  Gain@B = selector accuracy - best-single-model accuracy
  Gap@O  = oracle accuracy - selector accuracy (lower is better)
  CostSave = 1 - selector cost / best-single cost, reported only where selector accuracy >= best single
"""
from statistics import mean


def best_single(scores, costs):
    """Highest mean score; ties broken by lower mean cost."""
    return max(scores, key=lambda m: (mean(scores[m]), -mean(costs[m])))


def evaluate(scores, costs, choice):
    """scores[m][i], costs[m][i]: recorded results. choice[i]: model chosen for instance i."""
    n = len(choice)
    models = list(scores)
    assert all(len(scores[m]) == n and len(costs[m]) == n for m in models), "ragged inputs"
    sel_acc = mean(scores[choice[i]][i] for i in range(n))
    sel_cost = mean(costs[choice[i]][i] for i in range(n))
    bs = best_single(scores, costs)
    bs_acc, bs_cost = mean(scores[bs]), mean(costs[bs])
    oracle = mean(max(scores[m][i] for m in models) for i in range(n))
    return {
        "n": n, "best_single": bs, "selector_acc": round(sel_acc, 4), "best_single_acc": round(bs_acc, 4),
        "oracle_acc": round(oracle, 4), "gain_over_best_single": round(sel_acc - bs_acc, 4),
        "gap_to_oracle": round(oracle - sel_acc, 4), "selector_cost": round(sel_cost, 6),
        "best_single_cost": round(bs_cost, 6),
        "cost_save_at_matched_acc": round(1 - sel_cost / bs_cost, 4) if sel_acc >= bs_acc and bs_cost else None,
    }


def gate(result, min_cost_save=0.30, max_acc_drop=0.01):
    """PRD target: at least 30% lower cost per task with no more than 1 point of accuracy loss."""
    cost_ok = result["selector_cost"] <= (1 - min_cost_save) * result["best_single_cost"]
    acc_ok = result["selector_acc"] >= result["best_single_acc"] - max_acc_drop
    return {"cost_target_met": cost_ok, "accuracy_target_met": acc_ok, "pass": cost_ok and acc_ok}
