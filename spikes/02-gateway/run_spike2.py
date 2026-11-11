"""Spike 2: LiteLLM proxy vs Portkey gateway in front of a mock OpenAI-compatible provider."""
import asyncio, json, os, sys, time
import aiohttp

MOCK = "http://127.0.0.1:9001"
KEY = "sk-spike-master-key-0123456789abcdef"
TIMEOUT_MS = 2000


NO_RETRY = os.environ.get("NO_RETRY") == "1"  # fail over immediately, no retry on the primary


def portkey_cfg(targets=("primary", "fallback")):
    return json.dumps({
        "strategy": {"mode": "fallback", "on_status_codes": [408, 429, 500, 502, 503, 504]},
        **({} if NO_RETRY else {"retry": {"attempts": 1, "on_status_codes": [429, 500, 502, 503, 504]}}),
        "targets": [{"provider": "openai", "api_key": "sk-mock", "request_timeout": TIMEOUT_MS,
                     "custom_host": f"{MOCK}/{t}/v1"} for t in targets]})


ROUTES = {
    "direct": ("http://127.0.0.1:9001/primary/v1/chat/completions", {}, "mock"),
    "litellm": ("http://127.0.0.1:4000/v1/chat/completions", {"Authorization": f"Bearer {KEY}"}, "agent-model"),
    "portkey": ("http://127.0.0.1:8787/v1/chat/completions", {"x-portkey-config": portkey_cfg()}, "mock"),
}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))] if xs else None


def stats(lat):
    return {"n": len(lat), "p50_ms": round(pct(lat, 50), 2), "p95_ms": round(pct(lat, 95), 2),
            "p99_ms": round(pct(lat, 99), 2)}


async def call(s, route, stream=False, timeout=60):
    url, hdr, model = ROUTES[route]
    body = {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
    if stream:
        body["stream"] = True
    t = time.perf_counter()
    try:
        async with s.post(url, json=body, headers=hdr, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if stream:
                ttfb, chunks, done = None, 0, False
                async for line in r.content:
                    if line.startswith(b"data:"):
                        if ttfb is None:
                            ttfb = (time.perf_counter() - t) * 1000
                        if b"[DONE]" in line:
                            done = True
                        else:
                            chunks += 1
                return {"status": r.status, "ms": (time.perf_counter() - t) * 1000, "ttfb": ttfb, "chunks": chunks, "done": done}
            raw = await r.read()
            served = None
            try:
                served = json.loads(raw).get("system_fingerprint")
            except Exception:
                pass
            return {"status": r.status, "ms": (time.perf_counter() - t) * 1000, "served_by": served}
    except Exception as e:
        return {"status": "exc:" + type(e).__name__, "ms": (time.perf_counter() - t) * 1000, "served_by": None}


async def admin(s, path, body=None):
    async with s.post(MOCK + path, json=body or {}) as r:
        await r.read()


async def hits(s):
    async with s.get(MOCK + "/admin/stats") as r:
        return await r.json()


async def overhead(s, results):
    out = {}
    only = os.environ.get("ONLY")
    for route in [r for r in ROUTES if not only or r in only.split(",")]:
        for _ in range(300):
            await call(s, route)
        seq = [(await call(s, route))["ms"] for _ in range(2000)]
        lat, it = [], iter(range(5000))

        async def worker():
            for _ in it:
                lat.append((await call(s, route))["ms"])
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(16)])
        out[route] = {"sequential": stats(seq), "concurrent16": {**stats(lat), "throughput_rps": round(5000 / (time.perf_counter() - t0))}}
        print("overhead", route, json.dumps(out[route]), flush=True)
    results["overhead"] = out


async def streaming(s, results):
    out = {}
    for route in ROUTES:
        rs = [await call(s, route, stream=True) for _ in range(100)]
        ok = [r for r in rs if r["status"] == 200 and r["done"] and r["chunks"] >= 20]
        codes = {}
        for r in rs:
            codes[str(r["status"])] = codes.get(str(r["status"]), 0) + 1
        out[route] = {"complete_streams": len(ok), "status": codes, "ttfb": stats([r["ttfb"] for r in ok]) if ok else None,
                      "total": stats([r["ms"] for r in ok]) if ok else None}
        print("stream", route, json.dumps(out[route]), flush=True)
    results["streaming"] = out


SCENARIOS = [
    ("primary_500", {"primary": "500"}, 100),
    ("primary_429", {"primary": "429"}, 100),
    ("primary_hang", {"primary": "hang"}, 15),
    ("primary_and_fallback_500", {"primary": "500", "fallback": "500"}, 50),
]


async def failures(s, results):
    out = {}
    for route in ("litellm", "portkey"):
        out[route] = {}
        for name, modes, n in SCENARIOS:
            await admin(s, "/admin/reset")
            for b, m in modes.items():
                await admin(s, "/admin/mode", {"backend": b, "mode": m})
            rs = [await call(s, route, timeout=40) for _ in range(n)]
            h = await hits(s)
            served, status = {}, {}
            for r in rs:
                served[str(r["served_by"])] = served.get(str(r["served_by"]), 0) + 1
                status[str(r["status"])] = status.get(str(r["status"]), 0) + 1
            out[route][name] = {"requests": n, "status": status, "served_by": served,
                                "latency": stats([r["ms"] for r in rs]), "backend_hits": h}
            print("failure", route, name, json.dumps(out[route][name]), flush=True)
            # let LiteLLM cooldowns expire so scenarios do not bleed into each other
            await admin(s, "/admin/reset")
            await asyncio.sleep(6)
    results["failures"] = out


async def main():
    results = {"cores": os.cpu_count(), "mock_latency_ms": 20, "timeout_ms": TIMEOUT_MS}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=64)) as s:
        await admin(s, "/admin/reset")
        which = sys.argv[1:] or ["overhead", "streaming", "failures"]
        if "overhead" in which: await overhead(s, results)
        if "streaming" in which: await streaming(s, results)
        if "failures" in which: await failures(s, results)
    json.dump(results, open(f"results_{'_'.join(which)}{'_noretry' if NO_RETRY else ''}{os.environ.get('SUFFIX', '')}.json", "w"), indent=2)

asyncio.run(main())
