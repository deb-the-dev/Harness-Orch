# Spike 1: policy hook latency (own evaluator vs OPA)

Question (tools list, section 8): can policy hooks be evaluated out of process with p95 under 5 ms at the expected call rate, and which engine should do it?

**Result: yes.** Out of process, both engines stay under 5 ms p95 up to about 2,000 requests per second in this sandbox. A single hook call costs about 0.5 ms (own evaluator) to 1 ms (OPA) at p95. The latency target does not decide between the engines; correctness discipline does (see Findings).

## Setup

- 4 cores, 16 GB RAM, shared by client, server and OPA. The client is Python asyncio (aiohttp), so measured latency includes about 0.2 to 0.4 ms of client overhead. A compiled client would read lower.
- OPA 1.22.0 static binary, SHA-256 verified against the published checksum. Run as `opa run --server`.
- Same 9 rules in Rego (`policy.rego`) and Python (`common.py`): payment block, tool grants per pack, external-message approval, budget hard limit, residency, model allow-list, PII redaction, tool-result truncation, malformed-input block.
- Data: 25 task packs, 30 tools and 15 models per pack (`data.json`, 15 KB).
- Inputs: 1 percent unknown pack, 2 percent malformed (empty, unknown hook, dropped key, wrong type).
- Four ways of calling: own evaluator in process, own evaluator over HTTP, OPA REST over TCP loopback, OPA REST over a Unix socket.

## Correctness: parity on 5,000 inputs

0 mismatches between Rego and the Python evaluator (verdict mix: 3,389 allow, 1,450 block, 147 modify, 14 require approval).

## Latency

Sequential (one request in flight; closest to one worker making one hook call):

| Path | p50 ms | p95 ms | p99 ms |
| --- | --- | --- | --- |
| Own evaluator, in process | 0.002 | 0.004 | 0.006 |
| Own evaluator, HTTP over Unix socket | 0.36 | 0.53 | 0.82 |
| Own evaluator, HTTP over TCP | 0.31 | 0.55 | 1.16 |
| OPA REST over Unix socket | 0.67 | 1.04 | 2.33 |
| OPA REST over TCP | 0.64 | 0.97 | 2.03 |

Open loop (Poisson arrivals at a fixed rate; latency counted from the intended arrival time, so queueing shows), Unix socket:

| Rate (req/s) | Own p95 ms | OPA p95 ms |
| --- | --- | --- |
| 200 | 3.1 | 3.5 |
| 500 | 2.8 | 3.3 |
| 1,000 | 2.3 | 2.7 |
| 2,000 | 2.3 | 4.3 |
| 3,000 | 8.2 | 14.2 |
| 4,000 | 19.1 | 38.0 |

Both engines are under 5 ms p95 through 2,000 req/s; both degrade at 3,000. The 16-way closed-loop run (`results.json`, `concurrent16`) saturates one server process and reports 3 ms (own) and 6.6 ms (OPA) p95; that is queueing at about 5,000 to 8,000 req/s, not a realistic operating point.

Rate assumption: the PRD gives no run volume. Each step makes about four hook calls and a model step takes seconds, so 1,000 concurrent runs would be roughly a few hundred hook calls per second, inside the passing range. This needs confirming once run volume is known.

## Findings

1. **Rego fails open on undefined lookups.** `not input.tool in data.packs[input.pack].tools` is undefined for an unknown pack, the rule does not fire, and the call is allowed. Found by the parity test (17 of 5,000 mismatched). The raw first-run results are in `results_failopen_first_run.json`.
2. **Rego fails open on malformed input.** `{}`, an unknown hook, or a missing `spend` block returned `allow`.
3. **Fix applied in both engines:** an explicit empty-pack default (`object.get`) and an input-validation rule that returns `block` with reason `malformed_input`. Parity was re-checked with malformed inputs in the mix: 0 mismatches.
4. **Rule for the hook client:** proceed only on an explicit `verdict == "allow"`. A missing result, an error or a timeout means block.

## Recommendation (for your decision)

- Keep policy out of process; the 5 ms target holds with margin.
- Between the engines:
  - **OPA** gives a mature, audited policy language and tooling, at about 2x the latency of the own evaluator and with the fail-open traps above. Mitigation: validation rule, parity or unit tests in CI, explicit-allow client.
  - **Own evaluator** is faster and fails closed by construction in plain code, but we would own the policy format, versioning, testing and authoring experience.
- My lean: OPA behind the hook interface with the three mitigations, because policy will be authored and reviewed by the org (PRD P1, M4) and a standard language helps. If authors will be engineers only, the own evaluator is a reasonable alternative. OPA decision logging was not tested here.

## Reproduce

```bash
cd spikes/01-hook-latency
SOCKDIR=/tmp ../../.venv/bin/python bench.py       # parity + latency
SOCKDIR=/tmp ../../.venv/bin/python open_loop.py   # arrival-rate test
```
