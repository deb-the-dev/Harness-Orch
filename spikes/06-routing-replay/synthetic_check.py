"""Sanity test of replay_metrics on SYNTHETIC data (not real benchmark results)."""
import random
from replay_metrics import evaluate, gate

rng = random.Random(1)
MODELS = {"small": (0.0004, {"a": .89, "b": .45, "c": .50, "d": .60}),
          "mid": (0.003, {"a": .90, "b": .70, "c": .66, "d": .84}),
          "large": (0.015, {"a": .90, "b": .86, "c": .84, "d": .85})}
CATS = list("abcd")
N = 8000
cat = [rng.choice(CATS) for _ in range(N)]
scores = {m: [1.0 if rng.random() < p[cat[i]] else 0.0 for i in range(N)] for m, (c, p) in MODELS.items()}
costs = {m: [c * rng.uniform(.7, 1.3) for _ in range(N)] for m, (c, p) in MODELS.items()}
half = N // 2
def pick(train_idx):  # "learned" selector: per category, the cheapest model within 3 points of the best on the train half
    out = {}
    for c in CATS:
        idx = [i for i in train_idx if cat[i] == c]
        acc = {m: sum(scores[m][i] for i in idx) / len(idx) for m in MODELS}
        top = max(acc.values())
        out[c] = min((m for m in MODELS if acc[m] >= top - 0.03), key=lambda m: MODELS[m][0])
    return out
policy = pick(range(half))
test = range(half, N)
sub = lambda d: {m: [d[m][i] for i in test] for m in d}
choice = [policy[cat[i]] for i in test]
res = evaluate(sub(scores), sub(costs), choice)
print("policy per category:", policy)
print(res)
print(gate(res))
