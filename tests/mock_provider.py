"""In-process mock OpenAI-compatible provider with failure injection, for router tests."""
import asyncio
import time
from aiohttp import web
from aiohttp.test_utils import TestServer


class MockProvider:
    def __init__(self):
        self.backends = {b: {"mode": "ok", "hits": 0, "text": f"ok from {b}", "texts": [], "requests": [], "pt": 12, "ct": 5} for b in ("small", "large", "disallowed")}
        self.server: TestServer | None = None

    async def start(self):
        app = web.Application()
        app.router.add_post("/{backend}/v1/chat/completions", self._chat)
        self.server = TestServer(app)
        await self.server.start_server()
        return self

    async def stop(self):
        await self.server.close()

    def base(self, backend):
        return f"http://127.0.0.1:{self.server.port}/{backend}/v1"

    def set(self, backend, mode="ok", **kw):
        self.backends[backend].update(mode=mode, **kw)

    def requests(self, backend):
        return self.backends[backend]["requests"]

    def hits(self, backend):
        return self.backends[backend]["hits"]

    async def _chat(self, request):
        b = self.backends[request.match_info["backend"]]
        b["hits"] += 1
        body = await request.json()
        b["requests"].append(body)
        if b["mode"] == "500":
            return web.json_response({"error": {"message": "injected", "type": "server_error"}}, status=500)
        if b["mode"] == "429":
            return web.json_response({"error": {"message": "injected", "type": "rate_limit_error"}}, status=429)
        if b["mode"] == "400":
            return web.json_response({"error": {"message": "bad request", "type": "invalid_request_error"}}, status=400)
        if b["mode"] == "hang":
            await asyncio.sleep(30)
        return web.json_response({"id": "x", "object": "chat.completion", "created": int(time.time()), "model": "m",
                                  "choices": [{"index": 0, "message": {"role": "assistant", "content": b["texts"].pop(0) if b["texts"] else b["text"]}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": b["pt"], "completion_tokens": b["ct"], "total_tokens": b["pt"] + b["ct"]}})
