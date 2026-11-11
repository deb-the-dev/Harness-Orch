# Spike 6: routing baseline replay (prepared; NOT run on real data)

**Status: scoring code ready and sanity-checked on synthetic data; no real result.** The LLMRouterBench result bundles are hosted on Hugging Face, Google Drive and Baidu Netdisk, and none of those hosts is reachable from the build sandbox. So there is no real gain, gap or cost figure yet, and the synthetic numbers printed by `synthetic_check.py` mean nothing about routers.

## Question
How much do the open routers (RouteLLM and one other) gain over the best single model on the task classes we mapped, and how far are they from an oracle? That decides whether a tier 2 selector is worth building.

## Files
- `replay_metrics.py`: replays a policy over recorded per-instance scores and costs. Computes gain over best single model, gap to oracle, cost save at matched accuracy, and the PRD gate (30 percent lower cost, at most 1 point of accuracy loss).
- `synthetic_check.py`: confirms the code behaves on made-up data (a per-category selector that picks the cheapest model within 3 points of the best).

## To run for real
1. Download `bench-release.tar.gz` from the LLMRouterBench repository's links (Hugging Face is the likely route) and extract it so files sit at `results/bench/<dataset>/<split>/<model>/<timestamp>.json`.
2. Load per-instance `score` and `cost` into `scores[model][i]` and `costs[model][i]` for the datasets that match the mapped task classes (code, tool use, math, knowledge). The exact JSON field names must be confirmed on the first real file: the README names the contents (query, prompt, prediction, ground truth, score, tokens, cost, raw output) but not the keys. Align instances across models by query.
3. Produce a `choice[i]` per policy (RouteLLM, a second router, random, best single, oracle) and call `evaluate` then `gate`.

## Exit criterion
Gain over best single model and gap to oracle on the mapped task classes, per router. Decision rule: build a tier 2 selector only if a router clears the gate on at least one mapped class.
