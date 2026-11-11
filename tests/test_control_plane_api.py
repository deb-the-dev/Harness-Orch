"""Tests for the FastAPI Control-Plane and Router HTTP service (`src/harness/server.py`)."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from harness.server import create_app


def test_control_plane_fde_lifecycle_and_router_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Health check
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # 2. Configure an FDE enterprise deployment via REST
    cfg = {
        "organization": {"org_id": "org-api-test", "name": "API Enterprise Org", "region": "eu"},
        "memory_engine": "graphiti",
        "context_items": [
            {
                "external_id": "POL-01",
                "kind": "policy",
                "title": "Vendor Approval Policy",
                "content": "All outbound vendor messages require compliance approval.",
            }
        ],
        "skills": [
            {
                "name": "ap_check",
                "description": "Verify invoice totals.",
                "body": "Always match PO number.",
            }
        ],
        "memory_facts": [
            {
                "scope_type": "org",
                "scope_id": "org-api-test",
                "key": "default_terms",
                "content": "Default payment terms are Net-45.",
            }
        ],
        "task_logs": [
            {
                "task_class": "extract",
                "step_role": "extract",
                "model_id": "m-small",
                "success": True,
                "cost_usd": 0.00005,
                "latency_ms": 75.0,
                "verifier_score": 0.96,
            }
        ],
    }
    r = client.post("/v1/fde/deployments", json=cfg)
    assert r.status_code == 200
    assert r.json()["org_id"] == "org-api-test"

    # 3. Preview optimal route
    r = client.post("/v1/fde/deployments/org-api-test/route-preview", json={"task_class": "extract", "step_role": "extract"})
    assert r.status_code == 200
    assert r.json()["optimal_model"] == "m-small"

    # 4. Query scoped context
    r = client.post("/v1/fde/deployments/org-api-test/context/query", json={"query": "payment terms approval"})
    assert r.status_code == 200
    assert r.json()["count"] >= 1
    assert r.json()["items"][0]["trust"] == "untrusted"

    # 5. Submit an ap_logging run -> pauses at external_comm approval wait
    r = client.post("/v1/runs", json={
        "org_id": "org-api-test",
        "pack": "ap_logging",
        "input": {"invoice_text": "Acme Corp $4250"},
        "budget_usd": 1.0,
    })
    assert r.status_code == 200
    run_data = r.json()
    run_id = run_data["run_id"]
    assert run_data["state"] == "approval_wait"
    approval_id = run_data["pending_approval"]["approval_id"]

    # 6. Approve and auto-resume run -> succeeds
    r = client.post(f"/v1/runs/{run_id}/approvals/{approval_id}", json={
        "org_id": "org-api-test",
        "approved": True,
        "approver": "compliance-lead",
        "auto_resume": True,
    })
    assert r.status_code == 200
    assert r.json()["state"] == "succeeded"
    assert r.json()["result"]["message_sent"] is True

    # 7. Verify append-only event log and hash integrity
    r = client.get(f"/v1/runs/{run_id}/events", params={"org_id": "org-api-test"})
    assert r.status_code == 200
    assert r.json()["integrity_ok"] is True
    assert len(r.json()["events"]) >= 8

    # 8. OpenAI-compatible and Anthropic-compatible router proxy endpoints
    r_oai = client.post("/v1/chat/completions", json={
        "org_id": "org-api-test",
        "messages": [{"role": "user", "content": "Extract invoice"}],
    })
    assert r_oai.status_code == 200
    assert r_oai.json()["object"] == "chat.completion"

    r_ant = client.post("/v1/messages", json={
        "org_id": "org-api-test",
        "messages": [{"role": "user", "content": "Extract invoice"}],
    })
    assert r_ant.status_code == 200
    assert r_ant.json()["type"] == "message"

    # 9. Run deterministic replay against m-large baseline
    r_rep = client.post("/v1/replay", json={"org_id": "org-api-test", "baseline_model_id": "m-large"})
    assert r_rep.status_code == 200
    assert r_rep.json()["replay_fidelity"] == 1.0
    assert r_rep.json()["cost_saved_pct"] > 0.30

    # 10. Activate kill switch -> blocks new model calls
    client.post("/v1/admin/kill-switch", json={"org_id": "org-api-test", "active": True})
    r_blocked = client.post("/v1/chat/completions", json={
        "org_id": "org-api-test",
        "messages": [{"role": "user", "content": "Should be blocked"}],
    })
    assert r_blocked.status_code == 403
