"""Open-loop test: Poisson arrivals at a fixed rate, so queueing shows up as latency."""
import asyncio, json, os, random, signal, subprocess, sys, time
import aiohttp
from common import make_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
OPA = os.path.join(HERE, "../../.tools/opa/opa")
SOCKDIR = os.environ.get("SOCKDIR", "/tmp")
RATES, SECONDS = [200, 500, 1000, 2000, 3000, 4000], 8
data, inputs = make_inputs(20000)
bodies_own = [json.dumps(i).encode() for i in inputs]
bodies_opa = [json.dumps({"input": i}).encode() for i in inputs]
HDR = {"Content-Type": "application/json"}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


async def at_rate(session, url, bodies, rate):
    lat, rng, tasks = [], random.Random(3), []

    async def one(i, t):  # t = intended arrival time, so client-side queueing is counted
        async with session.post(url, data=bodies[i % len(bodies)], headers=HDR) as r:
            await r.read()
        lat.append((time.perf_counter() - t) * 1000)

    start, nxt, i = time.perf_counter(), 0.0, 0
    while nxt < SECONDS:
        delay = start + nxt - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(i, start + nxt)))
        i += 1
        nxt += rng.expovariate(rate)
    await asyncio.gather(*tasks)
    return {"rate_rps": rate, "n": len(lat), "p50_ms": round(pct(lat, 50), 3), "p95_ms": round(pct(lat, 95), 3),
            "p99_ms": round(pct(lat, 99), 3), "max_ms": round(max(lat), 3)}


async def main():
    own_sock, opa_sock = f"{SOCKDIR}/own2.sock", f"{SOCKDIR}/opa2.sock"
    for p in (own_sock, opa_sock):
        if os.path.exists(p):
            os.remove(p)
    sp = lambda c: subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE, preexec_fn=os.setsid)
    procs = [sp([sys.executable, "own_server.py", f"unix:{own_sock}"]),
             sp([OPA, "run", "--server", "--addr", f"unix://{opa_sock}", "policy.rego", "data.json"])]
    results = {}
    try:
        await asyncio.sleep(3)
        for name, sock, path, bodies in (("own_http_uds", own_sock, "/decide", bodies_own),
                                         ("opa_rest_uds", opa_sock, "/v1/data/hooks/decision", bodies_opa)):
            results[name] = []
            async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=sock, limit=128)) as s:
                for i in range(1500):  # warm-up
                    async with s.post("http://localhost" + path, data=bodies[i], headers=HDR) as r:
                        await r.read()
                for rate in RATES:
                    res = await at_rate(s, "http://localhost" + path, bodies, rate)
                    results[name].append(res)
                    print(name, json.dumps(res), flush=True)
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    json.dump(results, open(os.path.join(HERE, "results_open_loop.json"), "w"), indent=2)


asyncio.run(main())
