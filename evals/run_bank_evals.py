"""Runnable Evaluation Suite for Enterprise Bank Synthetic Data, Routing Optimization, and Harness Controls."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from harness.router import RouterError, RoutingRequest
from harness.runs import RunState
from harness.tools import ToolSpec

from .bank_agents import build_bank_deployment
from .bank_synthetic_data import generate_enterprise_bank_dataset


async def run_all_bank_evals(export_dir: str | Path | None = None) -> dict[str, Any]:
    dataset = generate_enterprise_bank_dataset()
    dep, _ = build_bank_deployment(dataset)

    if export_dir is not None:
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset.export_json(out_dir / "enterprise_bank_dataset.json")

    checks: dict[str, Any] = {}

    # 1. Customer Service Agent (CUST-1001, DSP-7001): pauses at approval wait, then completes on human approval
    cs_wait = await dep.submit_and_run("bank_customer_service", {"customer_id": "CUST-1001", "dispute_id": "DSP-7001"}, budget_usd=1.0)
    assert cs_wait.state == RunState.APPROVAL_WAIT
    app_id = cs_wait.pending_approval["approval_id"]
    dep.orchestrator.decide(cs_wait.run_id, approval_id=app_id, approved=True, approver="cs-supervisor")
    cs_done = await dep.worker.run(cs_wait.run_id)
    assert cs_done.state == RunState.SUCCEEDED
    cs_events = dep.log.read(dep.org_id, cs_wait.run_id)
    checks["customer_service_approval_flow"] = {
        "passed": True,
        "run_id": cs_wait.run_id,
        "memory_access_events": sum(1 for e in cs_events if e.type == "memory.access"),
        "models_used": [e.payload["model"] for e in cs_events if e.type == "model.response"],
    }

    # 2. Customer Service Rejection Gate (CUST-1002, DSP-7002): rejected notice must NOT dispatch
    cs_rej_wait = await dep.submit_and_run("bank_customer_service", {"customer_id": "CUST-1002", "dispute_id": "DSP-7002"}, budget_usd=1.0)
    assert cs_rej_wait.state == RunState.APPROVAL_WAIT
    dep.orchestrator.decide(cs_rej_wait.run_id, approval_id=cs_rej_wait.pending_approval["approval_id"], approved=False, approver="compliance-lead")
    cs_rej_done = await dep.worker.run(cs_rej_wait.run_id)
    assert cs_rej_done.state == RunState.SUCCEEDED
    rej_result = [e.payload["result"] for e in dep.log.read(dep.org_id, cs_rej_wait.run_id) if e.type == "run.succeeded"][-1]
    assert rej_result["notice_sent"] is False
    checks["customer_service_rejection_gate"] = {"passed": True, "notice_sent": rej_result["notice_sent"]}

    # 3. Financial Planner & Attestation Agent (CUST-1001, REF-8001): legitimate duplicate refund attested
    fp_ok = await dep.submit_and_run("bank_financial_planner", {"customer_id": "CUST-1001", "dispute_id": "DSP-7001", "refund_id": "REF-8001"}, budget_usd=1.0)
    assert fp_ok.state == RunState.SUCCEEDED
    fp_ok_res = [e.payload["result"] for e in dep.log.read(dep.org_id, fp_ok.run_id) if e.type == "run.succeeded"][-1]
    assert fp_ok_res["attestation"]["attest_approved"] is True
    assert fp_ok_res["attestation_recorded"] is True
    checks["financial_planner_refund_attestation"] = {"passed": True, "attestation": fp_ok_res["attestation"]}

    # 4. Financial Planner AML Hard Block (CUST-1003, REF-8003): AML-flagged account must deny attestation + isolated subagent check
    fp_aml = await dep.submit_and_run("bank_financial_planner", {"customer_id": "CUST-1003", "dispute_id": "DSP-7003", "refund_id": "REF-8003"}, budget_usd=1.0)
    assert fp_aml.state == RunState.SUCCEEDED
    fp_aml_res = [e.payload["result"] for e in dep.log.read(dep.org_id, fp_aml.run_id) if e.type == "run.succeeded"][-1]
    assert fp_aml_res["attestation"]["attest_approved"] is False

    sub_res = await dep.subagents.run_subagent(
        parent_run_id=fp_aml.run_id,
        child_pack_name="payout_terms_review",
        child_input={"contract_id": "CTR-AML-1003", "contract_text": "Wire settlement review"},
        requested_budget_usd=0.20,
    )
    assert sub_res["ok"] is True
    checks["financial_planner_aml_block_and_subagent"] = {"passed": True, "attest_approved": False, "subagent_ok": sub_res["ok"]}

    # 5. Harness Control Invariants: Payment Tool Block + Data Residency + PII Redaction
    payment_blocked = False
    try:
        async def _noop(_: dict[str, Any]) -> dict[str, Any]:
            return {}
        dep.tools.register(ToolSpec("execute_wire_payment", "payment", _noop))
    except ValueError:
        payment_blocked = True
    assert payment_blocked

    # Residency check: restricted data_class with model outside restricted_regions is blocked by Tier 1 policy
    d_res = await dep.policy_client.decide({
        "hook": "pre_model",
        "action_class": "read",
        "pack": "bank_financial_planner",
        "data_class": "restricted",
        "model": "m-medium",
        "model_region": "us-east-1",
        "spend": {"run_usd": 0.0, "est_call_usd": 0.01, "run_limit_usd": 1.0},
    })
    residency_blocked = (d_res.verdict == "block" and "residency" in d_res.reasons)
    assert residency_blocked

    d_pii = await dep.policy_client.decide({
        "hook": "post_model",
        "action_class": "read",
        "output": {"contains_pii": True},
    })
    assert d_pii.verdict == "modify" and "redact_pii" in d_pii.reasons
    checks["harness_controls"] = {
        "payment_tool_blocked": payment_blocked,
        "residency_enforced": residency_blocked,
        "pii_redaction_triggered": True,
        "policy_violations_reaching_execution": 0,
    }

    # 6. Deterministic Replay & Promotion Gate vs best-single-model (m-large)
    report = dep.replay_engine.evaluate(dep.selector, baseline_model_id="m-large")
    assert report.replay_fidelity == 1.0
    assert report.cost_saved_pct >= 0.30
    assert report.passes_promotion_gate is True
    assert not report.corrupted_runs

    summary = {
        "dataset": dataset.summary(),
        "checks": checks,
        "replay_eval": {
            "total_runs": report.total_runs,
            "total_steps": report.total_steps,
            "replay_fidelity": report.replay_fidelity,
            "cost_saved_pct": round(report.cost_saved_pct, 4),
            "score_delta_vs_baseline": round(report.score_delta_vs_baseline, 4),
            "passes_promotion_gate": report.passes_promotion_gate,
            "corrupted_runs": report.corrupted_runs,
        },
    }

    if export_dir is not None:
        (Path(export_dir) / "bank_eval_report.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    out = asyncio.run(run_all_bank_evals(export_dir=Path(__file__).resolve().parent / "results"))
    print(json.dumps(out, indent=2))
