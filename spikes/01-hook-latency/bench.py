"""Spike 1 benchmark: policy hook latency, own evaluator vs OPA, in and out of process."""
import asyncio, json, os, signal, statistics, subprocess, sys, time
import aiohttp
from common import make_inputs, make_evaluator

HERE = os.path.dirname(os.path.abspath(__file__))
OPA = os.path.join(HERE, "../../.tools/opa/opa")
SOCKDIR = os.environ.get("SOCKDIR", "/tmp")
N_SEQ, N_CONC, CONC, WARM, N_PARITY = 10000, 30000, 16, 2000, 5000

data, inputs = make_inputs(max(N_SEQ, N_CONC, N_PARITY) + WARM)
evaluate = make_evaluator(data)
bodies_own = [json.dumps(i).encode() for i in inputs]
bodies_opa = [json.dumps({"input": i}).encode() for i in inputs]
HDR = {"Content-Type": "application/json"}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def summarize(lat_ms, wall_s, n):
    return {"n": n, "p50_ms": round(pct(lat_ms, 50), 3), "p95_ms": round(pct(lat_ms, 95), 3),
            "p99_ms": round(pct(lat_ms, 99), 3), "max_ms": round(max(lat_ms), 3),
            "throughput_rps": round(n / wall_s)}


async def run_target(name, base, path, bodies, unix=None):
    conn = aiohttp.UnixConnector(path=unix, limit=CONC) if unix else aiohttp.TCPConnector(limit=CONC)
    out = {}
    async with aiohttp.ClientSession(connector=conn) as s:
        url = base + path

        async def one(i):
            t = time.perf_counter()
            async with s.post(url, data=bodies[i], headers=HDR) as r:
                await r.read()
            return (time.perf_counter() - t) * 1000

        for i in range(WARM):
            await one(i)
        # sequential: one in-flight request, the cleanest latency view
        lat, t0 = [], time.perf_counter()
        for i in range(N_SEQ):
            lat.append(await one(WARM + i))
        out["sequential"] = summarize(lat, time.perf_counter() - t0, N_SEQ)
        # concurrent: CONC in-flight requests, a loaded view
        lat, idx = [], iter(range(WARM, WARM + N_CONC))

        async def worker():
            for i in idx:
                lat.append(await one(i))

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(CONC)])
        out["concurrent16"] = summarize(lat, time.perf_counter() - t0, N_CONC)
    print(name, json.dumps(out), flush=True)
    return out


def bench_inproc():
    lat = []
    for i in range(WARM):
        evaluate(inputs[i])
    for i in range(N_SEQ):
        t = time.perf_counter_ns()
        evaluate(inputs[WARM + i])
        lat.append((time.perf_counter_ns() - t) / 1e6)
    out = {"sequential": summarize(lat, sum(lat) / 1000, N_SEQ)}
    print("own in-process", json.dumps(out), flush=True)
    return out


async def parity():
    """Same decision from Rego and from the own evaluator on N_PARITY inputs."""
    mism, counts = [], {}
    async with aiohttp.ClientSession() as s:
        for i in range(N_PARITY):
            async with s.post("http://127.0.0.1:8181/v1/data/hooks/decision", data=bodies_opa[i], headers=HDR) as r:
                got = (await r.json())["result"]
            want = evaluate(inputs[i])
            counts[want["verdict"]] = counts.get(want["verdict"], 0) + 1
            if got != want:
                mism.append({"input": inputs[i], "opa": got, "own": want})
    return {"checked": N_PARITY, "mismatches": len(mism), "verdict_mix": counts, "first_mismatches": mism[:3]}


def spawn(cmd):
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE, preexec_fn=os.setsid)


async def main():
    results = {"cores": os.cpu_count(), "rules": 9, "packs": 25, "tools_per_pack": 30}
    procs = []
    own_sock, opa_sock = f"{SOCKDIR}/own.sock", f"{SOCKDIR}/opa.sock"
    for p in (own_sock, opa_sock):
        if os.path.exists(p):
            os.remove(p)
    try:
        procs.append(spawn([sys.executable, "own_server.py", "127.0.0.1:8201"]))
        procs.append(spawn([sys.executable, "own_server.py", f"unix:{own_sock}"]))
        procs.append(spawn([OPA, "run", "--server", "--addr", "127.0.0.1:8181", "policy.rego", "data.json"]))
        procs.append(spawn([OPA, "run", "--server", "--addr", f"unix://{opa_sock}", "policy.rego", "data.json"]))
        await asyncio.sleep(3)
        results["parity"] = await parity()
        print("parity", json.dumps({k: v for k, v in results["parity"].items() if k != "first_mismatches"}), flush=True)
        results["own_inprocess"] = bench_inproc()
        results["own_http_tcp"] = await run_target("own http tcp", "http://127.0.0.1:8201", "/decide", bodies_own)
        results["own_http_uds"] = await run_target("own http uds", "http://localhost", "/decide", bodies_own, unix=own_sock)
        results["opa_rest_tcp"] = await run_target("opa rest tcp", "http://127.0.0.1:8181", "/v1/data/hooks/decision", bodies_opa)
        results["opa_rest_uds"] = await run_target("opa rest uds", "http://localhost", "/v1/data/hooks/decision", bodies_opa, unix=opa_sock)
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


asyncio.run(main())
