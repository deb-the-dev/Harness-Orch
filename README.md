# Harness-Orch: Enterprise Agent Control Harness, Model Routing Layer & FDE Orchestration

Enterprise control plane, step-aware model router, and Forward Deployed Engineer (FDE) customization layer implementing the **PRD (v0.1)**, **ERD (v0.1)**, **State Flow & Architecture (v0.1)**, and **Tools List (v0.1)** specifications.

## Core Capabilities

| Layer | Module | Description | Spec Reference |
| --- | --- | --- | --- |
| **FDE Enterprise Customization** | `src/harness/fde.py`, `configs/fde_enterprise_deployment.yaml` | Declarative and SDK-driven tenant deployment (`FDEDeployment`), custom step-level `RoutingMap`, `AdaptiveOutcomeSelector` blending benchmark priors with empirical `TaskLogRecord` outcomes and prompt-cache affinity, and isolated `SubagentRunner`. | PRD §8, §9, §10 (M9, C7, C8) |
| **Enterprise Context, Memory & Skills** | `src/harness/context.py` | Swappable memory engine protocol (`GraphitiMemoryAdapter` with temporal validity windows and `Mem0MemoryAdapter`), permission filtering *before* ranking across `user`/`team`/`agent`/`org` scopes, 4-tier context priority (`policy -> document/ticket -> data -> code`), `trust="untrusted"` tagging, review/revocation state machine, and progressive-disclosure `SkillRegistry`. | PRD §9, §10 (M1–M10, C6); ERD §3.5, §3.7 |
| **Step-Aware Model Router** | `src/harness/router/` | 3-tier routing: **Tier 1** hard constraints (OPA policy per candidate model, region/residency, data class, capability, circuit breaker, pre-call budget reservation), **Tier 2** evaluated selector (`StaticPriorSelector` & `AdaptiveOutcomeSelector`), **Tier 3** verifier-triggered escalation, shadow policy scoring, and failover strictly inside the allowed set. | PRD §8, §11.B (R1–R8); Spike 2 |
| **Event-Sourced Spine & Storage** | `src/harness/events.py`, `src/harness/pg.py`, `src/harness/migrations/` | Append-only event log (SQLite for dev/tests; range-partitioned PostgreSQL with Row-Level Security and gapless per-run `seq` counter for production) and SHA-256 `payload_hash` verification. | ERD §1, §3.3, §4 |
| **Stateless Worker & Context Engineering** | `src/harness/worker.py`, `src/harness/artifacts.py`, `src/harness/history.py`, `src/harness/compaction.py`, `src/harness/rubric.py` | Stateless worker loop with durable human approval wait & resume, content-addressed large tool result offloading, cross-provider portable history, routed context compaction, and separate-model rubric grading. | PRD §10 (C1–C5), §11.A (H1–H8) |
| **Qualified Task Packs** | `src/harness/packs.py` | All four v1 Finance Task Packs (`InvoiceLoggingPack`, `ContractPaymentsPack`, `PayoutTermsReviewPack`, `VendorRenewalNegotiationPack`) plus `FDEWorkflowPack` for declarative multi-step enterprise workflows. | PRD §8; Tools List §6 |
| **Deterministic Replay & Promotion Gate** | `src/harness/replay.py` | Reconstructs model requests from the append-only log, verifies hashes, and evaluates candidate routing policies against a best-single-model baseline (`>= 30%` cost savings at `<= 1%` quality delta). | PRD §8, §11.C (P4, P6, R6); ERD §3.6 |
| **FastAPI Control Plane & Router Service** | `src/harness/server.py` | REST endpoints for FDE tenant deployments, context/memory/task-log ingestion, route previews, run lifecycle (`submit`, `pause`, `resume`, `cancel`, `fork`, `approvals`), replay evaluation, emergency kill switch, and OpenAI (`/v1/chat/completions`) + Anthropic (`/v1/messages`) router endpoints. | PRD R1, P8, H2, H6, H9 |
| **Container & Kubernetes / Helm Packaging** | `Dockerfile`, `docker-compose.yml`, `deploy/k8s/`, `deploy/helm/harness-orch/` | Production container image with OPA, single-host Docker Compose stack (`postgres:17` + `opa` + `control-plane` + `worker`), and Kubernetes/Helm manifests with `NetworkPolicy` forcing worker egress through the router. | Tools List §2; Spike 4 |

## Quickstart

### 1. Local Setup & Test Suite
```bash
scripts/setup.sh                       # Creates .venv, installs dependencies, downloads checksum-verified OPA
scripts/pg_dev.sh                      # Optional: launches throwaway PostgreSQL 17 in Docker (port 55432)
.venv/bin/python -m pytest -q          # Runs the full test suite (core, context, FDE enterprise, replay, and FastAPI control plane)
```

### 2. Run the FastAPI Control Plane & Router Server
```bash
.venv/bin/uvicorn harness.server:create_app --factory --host 0.0.0.0 --port 8080
```

### 3. FDE Enterprise Customization Quick Example (Python SDK)
```python
import asyncio
import yaml
from harness import FDEDeployment

async def main():
    with open("configs/fde_enterprise_deployment.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize FDE deployment with custom routing map, enterprise policies, skills, memory, and task logs
    deployment = FDEDeployment.from_dict(config)

    # Preview the optimal model path for a step role given enterprise context and task logs
    route = deployment.preview_optimal_route(task_class="extract", step_role="extract")
    print("Optimal Route:", route[0])

    # Run a customized enterprise workflow pack
    view = await deployment.submit_and_run(
        pack="fde_audit_triage",
        run_input={"invoice_text": "Vendor: Acme Cloud Corp, Amount: $4250.00"},
        budget_usd=0.50,
    )
    print("Run State:", view.state)

asyncio.run(main())
```

### 4. Container & Kubernetes Deployment
```bash
# Single-host Docker Compose stack (PostgreSQL 17 + OPA + Control Plane + Worker)
docker compose up --build -d

# Kubernetes manifests (Control Plane + OPA sidecar + Router Egress NetworkPolicy)
kubectl apply -f deploy/k8s/manifests.yaml

# Helm chart deployment
helm upgrade --install harness-orch ./deploy/helm/harness-orch --namespace harness-orch --create-namespace
```
