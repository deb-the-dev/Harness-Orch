"""Mock OpenAI-compatible provider with three backends and runtime failure injection.

POST /{backend}/v1/chat/completions   backend in primary | fallback | disallowed
POST /admin/mode   {"backend": "primary", "mode": "ok|500|429|hang", "latency_ms": 20}
GET  /admin/stats  hit counts per backend;  POST /admin/reset
"""
import asyncio, json, sys, time
from aiohttp import web

BACKENDS = {b: {"mode": "ok", "latency_ms": 20, "hits": 0} for b in ("primary", "fallback", "disallowed")}
STREAM_CHUNKS, STREAM_GAP_MS = 20, 10


def completion(backend, model):
    return {"id": "cmpl-mock", "object": "chat.completion", "created": int(time.time()), "model": model,
            "system_fingerprint": backend,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"ok from {backend}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}}


async def chat(request):
    backend = request.match_info["backend"]
    st = BACKENDS[backend]
    st["hits"] += 1
    body = await request.json()
    mode = st["mode"]
    if mode == "500":
        return web.json_response({"error": {"message": "injected upstream failure", "type": "server_error"}}, status=500)
    if mode == "429":
        return web.json_response({"error": {"message": "injected rate limit", "type": "rate_limit_error"}}, status=429)
    if mode == "hang":
        await asyncio.sleep(30)
    model = body.get("model", "mock")
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        await asyncio.sleep(st["latency_ms"] / 1000)
        for i in range(STREAM_CHUNKS):
            chunk = {"id": "cmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "delta": {"content": f"t{i} "} if i else {"role": "assistant", "content": "t0 "}, "finish_reason": None}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await asyncio.sleep(STREAM_GAP_MS / 1000)
        await resp.write(b"data: [DONE]\n\n")
        return resp
    await asyncio.sleep(st["latency_ms"] / 1000)
    return web.json_response(completion(backend, model))


async def admin_mode(request):
    b = await request.json()
    BACKENDS[b["backend"]].update({k: b[k] for k in ("mode", "latency_ms") if k in b})
    return web.json_response({"ok": True})


async def admin_stats(request):
    return web.json_response({k: v["hits"] for k, v in BACKENDS.items()})


async def admin_reset(request):
    for v in BACKENDS.values():
        v.update({"mode": "ok", "latency_ms": 20, "hits": 0})
    return web.json_response({"ok": True})


app = web.Application()
app.router.add_post("/{backend}/v1/chat/completions", chat)
app.router.add_post("/admin/mode", admin_mode)
app.router.add_get("/admin/stats", admin_stats)
app.router.add_post("/admin/reset", admin_reset)

if __name__ == "__main__":
    web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 9001, print=None, access_log=None)
