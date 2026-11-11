# Spike 2: gateway (LiteLLM vs Portkey), behind our adapter interface

Question (tools list, section 8): which gateway should provide provider adapters? Exit criteria: one chosen; injected provider failures trigger fallback inside the allowed set; streaming works; overhead measured.

**Result: neither passes cleanly.** LiteLLM meets every functional criterion but costs far more CPU and latency per call. Portkey's gateway is much lighter but failed every streaming request in this environment. Fallback stayed inside the allowed set for both. Neither stops sending traffic to a failing provider, which confirms our own circuit breaker (ERD model_health) is needed.

## Setup

- Mock OpenAI-compatible provider (`mock_provider.py`): three backends (`primary`, `fallback`, `disallowed`), fixed 20 ms latency, runtime failure injection (500, 429, hang), 20-chunk streaming.
- LiteLLM proxy 1.108.0 (`litellm_config.yaml`): `agent-model` with a single allowed fallback; a third model (`other-model`) configured on the gateway but in no fallback list. A master key is required to start.
- Portkey open-source gateway 1.15.2 (npm), Node 22.23.4 and 20.20.2. Routing is passed per request in the `x-portkey-config` header.
- 4 cores, 16 GB RAM shared by client, mock and gateway. Python aiohttp client. Numbers are indicative, not production sizing.
- Not tested: authentication and virtual keys, spend tracking and budgets (need a database), multi-deployment cooldown, TLS, real provider behaviour.

## 1. Overhead (non-streaming, mock latency 20 ms)

| Route | Sequential p50 / p95 ms | Added p50 / p95 ms | 16 in flight: throughput req/s | 16 in flight p95 ms |
| --- | --- | --- | --- | --- |
| Direct to mock | 21.8 / 23.0 | n/a | 708 | 25.0 |
| Portkey gateway | 25.1 / 28.1 | +3.3 / +5.1 | 477 | 44.8 |
| LiteLLM proxy, 1 worker | 41.8 / 52.9 | +20.0 / +29.9 | 62 | 330.7 |
| LiteLLM proxy, 4 workers | 39.6 / 49.7 | +17.9 / +26.4 | 134 | 212.9 |
| LiteLLM as an in-process library | 28.4 / 32.6 | +6.8 / +9.6 | 156 (one process) | 133.0 |

Reading: LiteLLM costs about 6 ms of CPU per call (about 150 calls/s per process). Used as a library inside our own router it adds about 7 ms; as a proxy it adds a second hop and about 20 ms. LiteLLM's README claims 8 ms p95 at 1,000 req/s; that was not reproduced here (single small box, default settings). Against model calls that take seconds, either overhead is small; it matters for the 5 ms hook budget only if it were on that path, and it is not.

## 2. Failover and the allowed set

Primary injected to fail; fallback healthy. Both gateways: 100 of 100 requests succeeded, all served by the fallback, and the `disallowed` backend received 0 requests in every scenario. When primary and fallback both failed, both returned an error (500) and tried nothing else.

| Scenario | Config | LiteLLM p50 ms | Portkey p50 ms |
| --- | --- | --- | --- |
| Primary 500 | one retry on primary, then fall back | 2,335 | 1,032 |
| Primary 500 | no retry, fall back immediately | 68 | 31 |
| Primary 429 | one retry, then fall back | 2,287 | 1,031 |
| Primary 429 | no retry | 68 | 30 |
| Primary hangs (2 s timeout) | one retry | 6,194 | 2,029 |
| Primary hangs (2 s timeout) | no retry | 2,064 | 2,031 |
| Both fail | no retry | 60 (error returned) | 9 (error returned) |

- With a retry configured, failover costs 1 to 2 seconds; without one, it adds about 6 ms (Portkey) and 26 ms (LiteLLM) over a normal call. Retry policy must be set per error type, not left at defaults.
- **No circuit breaker in either.** With the primary down for 100 requests, the primary still received 100 requests through both gateways (200 with one retry). LiteLLM did not cool down the failing primary in this single-deployment setup; multi-deployment cooldown was not tested.
- **Where the allowed set lives differs.** LiteLLM holds it server-side (config file). Portkey takes the target list in a request header, so a caller can choose its own targets. With Portkey, only our router may set that header, which is what the egress rule already requires.

## 3. Streaming

| Route | Complete streams (of 100) | Time to first token p50 / p95 ms | Total p50 ms |
| --- | --- | --- | --- |
| Direct | 100 | 21.9 / 22.6 | 233.6 |
| LiteLLM proxy | 100 | 38.5 / 46.7 | 256.7 |
| Portkey gateway | **0** (HTTP 500 on all 100) | n/a | n/a |

Portkey logs `tryTargetsRecursively error: immutable TypeError: immutable` for every streaming request. It happened with the retry and fallback config, with a single target, with request headers instead of a config, against my mock, against a plain Node SSE server, and under both Node 22.23.4 and 20.20.2. Non-streaming works. Cause not isolated; it may be a gateway 1.15.2 bug on the custom-host path or something specific to this environment. Streaming is required (PRD R1 and the architecture), so this blocks Portkey until resolved. Note: `results_overhead_streaming.json` holds an earlier streaming result from a wrong completeness check (LiteLLM adds one trailing chunk); the corrected run is `results_streaming.json`.

## Recommendation (for your decision)

- **Carry LiteLLM forward as the adapter layer, used as a library inside our router**, not as a separate proxy: +7 ms instead of +20 ms, no extra hop, allowed set stays in our config, and it fits the Python decision. Scale by running more router processes.
- Set retry per error type; default to fail over immediately on 429, 5xx and timeout, and retry only where it is safe.
- Build our own circuit breaker and health scoring (already a Build item).
- Keep Portkey as a lightweight alternative only if its streaming fault is resolved; the quickest check is a different gateway version or a real upstream.
- Open: LiteLLM's licence is MIT except its enterprise directory, so using the SDK is fine, but features such as SSO and some budget controls may be in that directory. Check which features we rely on.

## Reproduce

```bash
python spikes/02-gateway/mock_provider.py 9001 &
LITELLM_MASTER_KEY=sk-spike-master-key-0123456789abcdef litellm --config spikes/02-gateway/litellm_config.yaml --port 4000 &
(cd .tools/portkey && node_modules/.bin/gateway) &          # port 8787
python spikes/02-gateway/run_spike2.py overhead streaming failures
NO_RETRY=1 python spikes/02-gateway/run_spike2.py failures   # needs litellm_config_noretry.yaml
python spikes/02-gateway/litellm_sdk_overhead.py
```
