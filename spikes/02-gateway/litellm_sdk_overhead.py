"""Overhead of LiteLLM used as an in-process library (no proxy hop) against the same mock."""
import asyncio, json, time
import aiohttp, litellm

litellm.drop_params = True
MSGS = [{"role": "user", "content": "hello"}]
KW = dict(model="openai/mock-primary", api_base="http://127.0.0.1:9001/primary/v1", api_key="sk-mock", messages=MSGS, max_tokens=16)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def stats(l):
    return {"n": len(l), "p50_ms": round(pct(l, 50), 2), "p95_ms": round(pct(l, 95), 2), "p99_ms": round(pct(l, 99), 2)}


async def main():
    async with aiohttp.ClientSession() as s:
        async def direct():
            t = time.perf_counter()
            async with s.post("http://127.0.0.1:9001/primary/v1/chat/completions", json={"model": "mock", "messages": MSGS, "max_tokens": 16}) as r:
                await r.read()
            return (time.perf_counter() - t) * 1000

        async def sdk():
            t = time.perf_counter()
            await litellm.acompletion(**KW)
            return (time.perf_counter() - t) * 1000

        out = {}
        for name, fn in (("direct", direct), ("litellm_sdk", sdk)):
            for _ in range(300):
                await fn()
            seq = [await fn() for _ in range(2000)]
            lat, it = [], iter(range(5000))

            async def worker():
                for _ in it:
                    lat.append(await fn())
            t0 = time.perf_counter()
            await asyncio.gather(*[worker() for _ in range(16)])
            out[name] = {"sequential": stats(seq), "concurrent16": {**stats(lat), "throughput_rps": round(5000 / (time.perf_counter() - t0))}}
            print(name, json.dumps(out[name]), flush=True)
        json.dump(out, open("results_litellm_sdk.json", "w"), indent=2)

asyncio.run(main())
