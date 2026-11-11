"""Shared rule data, input generator and the own evaluator for spike 1."""
import json
import random

HOOKS = ["pre_model", "post_model", "pre_tool", "post_tool"]
HOOK_WEIGHTS = [0.3, 0.3, 0.2, 0.2]
ACTIONS = ["read", "internal_write", "external_comm", "payment"]
ACTION_WEIGHTS = [0.55, 0.35, 0.095, 0.005]
DATA_CLASSES = ["public", "internal", "confidential", "restricted"]
REGIONS = ["me", "eu", "us", "ap"]
APPROVALS = ["none", "pending", "approved", "rejected"]


def build_data(seed=7, n_packs=25, tools_per_pack=30, models_per_pack=15):
    rng = random.Random(seed)
    tool_pool = [f"tool_{i:03d}" for i in range(120)]
    model_pool = [f"model_{i:02d}" for i in range(60)]
    packs = {}
    for p in range(n_packs):
        packs[f"pack_{p:02d}"] = {
            "tools": rng.sample(tool_pool, tools_per_pack),
            "models": rng.sample(model_pool, models_per_pack),
            "restricted_regions": rng.sample(REGIONS, 2),
        }
    return {"packs": packs, "limits": {"max_tool_result_bytes": 1048576}}, tool_pool, model_pool


def make_inputs(n, seed=11):
    data, tool_pool, model_pool = build_data()
    rng = random.Random(seed)
    pack_names = list(data["packs"])
    out = []
    for _ in range(n):
        hook = rng.choices(HOOKS, HOOK_WEIGHTS)[0]
        # 1% of inputs reference an unknown pack, to test parity on undefined lookups
        pack = "pack_unknown" if rng.random() < 0.01 else rng.choice(pack_names)
        inp = {
            "hook": hook,
            "pack": pack,
            "action_class": rng.choices(ACTIONS, ACTION_WEIGHTS)[0],
            "data_class": rng.choice(DATA_CLASSES),
            "model": rng.choice(model_pool),
            "model_region": rng.choice(REGIONS),
            "tool": rng.choice(tool_pool),
            "approval": {"state": rng.choice(APPROVALS)},
            "spend": {
                "run_usd": round(rng.uniform(0, 9), 4),
                "est_call_usd": round(rng.uniform(0, 1.5), 4),
                "run_limit_usd": 10.0,
            },
            "output": {"contains_pii": rng.random() < 0.1, "bytes": int(rng.expovariate(1 / 200000))},
            "run_id": f"run_{rng.randrange(10**6):06d}",
            "step_id": f"step_{rng.randrange(10**4):04d}",
        }
        if rng.random() < 0.02:  # malformed inputs: must be blocked, never allowed
            kind = rng.choice(["empty", "unknown_hook", "drop_key", "bad_type"])
            if kind == "empty":
                inp = {}
            elif kind == "unknown_hook":
                inp["hook"] = "nonsense"
            elif kind == "drop_key":
                inp.pop(rng.choice(["hook", "action_class", "spend", "output", "approval", "tool", "model", "data_class"]), None)
            else:
                rng.choice([lambda: inp["spend"].__setitem__("run_usd", "9"),
                            lambda: inp["output"].__setitem__("bytes", "huge"),
                            lambda: inp["output"].__setitem__("contains_pii", "yes"),
                            lambda: inp.__setitem__("tool", 42)])()
        out.append(inp)
    return data, out


def make_evaluator(data):
    packs = {
        k: {"tools": set(v["tools"]), "models": set(v["models"]), "regions": set(v["restricted_regions"])}
        for k, v in data["packs"].items()
    }
    empty = {"tools": set(), "models": set(), "regions": set()}
    max_bytes = data["limits"]["max_tool_result_bytes"]

    def isnum(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def get(d, *path):
        for k in path:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    def valid(inp):
        if get(inp, "action_class") not in ("read", "internal_write", "external_comm", "payment"):
            return False
        hook = get(inp, "hook")
        if hook == "pre_model":
            return (all(isnum(get(inp, "spend", k)) for k in ("run_usd", "est_call_usd", "run_limit_usd"))
                    and get(inp, "data_class") in ("public", "internal", "confidential", "restricted")
                    and isinstance(get(inp, "model"), str) and isinstance(get(inp, "model_region"), str))
        if hook == "post_model":
            return isinstance(get(inp, "output", "contains_pii"), bool)
        if hook == "pre_tool":
            return isinstance(get(inp, "tool"), str) and get(inp, "approval", "state") in ("none", "pending", "approved", "rejected")
        if hook == "post_tool":
            return isnum(get(inp, "output", "bytes"))
        return False

    def evaluate(inp):
        if not valid(inp):
            return {"verdict": "block", "reasons": ["malformed_input"]}
        hook = inp["hook"]
        pack = packs.get(inp.get("pack"), empty)
        block, approval, modify = set(), set(), set()
        if inp["action_class"] == "payment":
            block.add("payment_out_of_scope")
        if hook == "pre_tool":
            if inp["tool"] not in pack["tools"]:
                block.add("tool_not_granted")
            if inp["action_class"] == "external_comm":
                state = inp["approval"]["state"]
                if state == "rejected":
                    block.add("approval_rejected")
                elif state in ("none", "pending"):
                    approval.add("external_comm_needs_approval")
        elif hook == "pre_model":
            sp = inp["spend"]
            if sp["run_usd"] + sp["est_call_usd"] > sp["run_limit_usd"]:
                block.add("budget_hard_limit")
            dc = inp["data_class"]
            if dc == "restricted" and inp["model_region"] not in pack["regions"]:
                block.add("residency")
            if dc in ("confidential", "restricted") and inp["model"] not in pack["models"]:
                block.add("model_not_allowed")
        elif hook == "post_model":
            if inp["output"]["contains_pii"]:
                modify.add("redact_pii")
        elif hook == "post_tool":
            if inp["output"]["bytes"] > max_bytes:
                modify.add("truncate_result")
        verdict = "block" if block else "require_approval" if approval else "modify" if modify else "allow"
        return {"verdict": verdict, "reasons": sorted(block | approval | modify)}

    return evaluate


if __name__ == "__main__":
    data, _ = make_inputs(1)
    with open("data.json", "w") as f:
        json.dump(data, f)
    print("wrote data.json", len(json.dumps(data)), "bytes")
