# Threat model (draft, v0.1)

Scope: the core in this repository. Written by hand from the design and the tests; it is a working document, not an audit.
**Status** says what is true today: *tested* (a test fails if it breaks), *designed* (specified, not built), *open* (no answer yet).

## Assets
| ID | Asset | Why it matters |
| --- | --- | --- |
| A1 | Event log and artifacts | The audit trail and the only source for replay |
| A2 | Provider API keys | Spend, and access to every prompt |
| A3 | Org policy corpus, procedures, memory | They steer agent behaviour |
| A4 | Messages sent to vendors | They leave the organisation |
| A5 | Financial data in prompts, tool results and logs | Confidential, often restricted |
| A6 | Policy data (tool grants, allowed models, regions) | It is what makes a call allowed |

## Attackers and vectors
A document, email or tool output carrying instructions; a poisoned memory or procedure; a worker or external harness trying to reach a provider directly; another tenant; an operator with database access; a task-pack author with a mistake.

## Threats
| ID | Threat | Mitigation | Status |
| --- | --- | --- | --- |
| T1 | Injected instructions make an agent send a message or misuse a tool | Policy sits outside the model. External messages need a human approval before send. No payment tool can be registered. | Tested: approval flow, rejection, payment registration refused. Open: a draft that persuades the human approver (what the approver sees is not designed). |
| T2 | Poisoned memory or procedure steers later runs | Retrieved context is untrusted, cannot grant permission, memory writes pass review (PRD M5, M6). | Designed. The context service is not built. |
| T3 | Policy fails open on an unknown pack or malformed input | Explicit validation rule in Rego, client validates first, only an explicit allow proceeds, OPA errors block. | Tested (found in spike 1). |
| T4 | A worker or external harness bypasses the router | Network egress policy so providers are reachable only from the router. | Designed. Spike 4 not run; how attempts are detected and alerted is open. |
| T5 | Cross-tenant read or write | org_id on every row and every read, row-level security, per-org artifact paths, a run cannot be appended to by another org. | Tested for the log and artifacts. Open: the queue is a system table without row-level security. |
| T6 | Log tampering or deletion | Triggers and missing grants refuse update and delete; payload hashes verify on read. | Tested. Open: a superuser can disable triggers and there is no hash chain, so tampering by that role is not detectable. |
| T7 | Artifact tampering or path traversal | Content-addressed storage verified on read, ids validated, org folder derived by hash. | Tested. |
| T8 | Credentials or personal data leak into prompts, logs or artifacts | Secrets never in prompts (rule); model output is redacted for emails and long digit strings before it enters history. | Partly tested. Open: tool inputs and outputs are not scanned; keys are plain values in tests; the log stores raw prompts. |
| T9 | Offload or compaction hides a fact the next step needs | The preview shown to the model is logged, the full artifact stays retrievable, compaction records how many messages the summary did not cover. | Tested for the mechanics. Open: whether a model reads the artifact when it should, which needs a behaviour eval. |
| T10 | The grader is talked into passing a bad answer | Output marked as data, unconfirmed criteria fail, the verdict is computed from per-criterion results, unparseable grading is unverified, the grader avoids the producing model when it can. | Tested. Residual: a grader model can still be fooled; calibration against labelled samples is not built. |
| T11 | A slow worker overlaps its replacement and repeats a side effect | A paid response is reused, an unknown-outcome tool call that is not idempotent needs a new human approval. | Tested for crashes. Open: no fencing token on appends for overlap. |
| T12 | A tool escapes its sandbox or runs a shell | No shell tool exists. Sandbox levels are specified. | Open for any future shell or code tool. Lesson from Deep Agents' own threat model: a shell tool bypassed its path permissions. |
| T13 | Runaway loops or spend | Budget reserved before each call, hard stop, escalation cap, attempt cap on the queue. | Tested. |
| T14 | A dependency is compromised or mislicensed | Licence audit and pinned versions. | Open. |

## Not in scope
Model behaviour itself; the customer's network and identity systems; provider-side data retention (relevant to residency, tracked in the PRD); physical security.

## Next
Run spike 4 and decide how blocked attempts are logged; design what an approver sees for an external draft; add scanning for tool inputs and outputs; add a hash chain if the audit requirement demands tamper evidence against a privileged role.
