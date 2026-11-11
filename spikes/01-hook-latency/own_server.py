"""Own evaluator as an out-of-process decision service (aiohttp, single process)."""
import sys
from aiohttp import web
from common import build_data, make_evaluator

data, _, _ = build_data()
evaluate = make_evaluator(data)


async def decide(request):
    inp = await request.json()
    return web.json_response(evaluate(inp))


app = web.Application()
app.router.add_post("/decide", decide)

if __name__ == "__main__":
    target = sys.argv[1]
    if target.startswith("unix:"):
        web.run_app(app, path=target[5:], print=None, access_log=None)
    else:
        host, port = target.split(":")
        web.run_app(app, host=host, port=int(port), print=None, access_log=None)
