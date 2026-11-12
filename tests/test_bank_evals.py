"""Pytest suite for the Enterprise Bank 10-table synthetic dataset, 5 task logs, and long-horizon agent evals."""
from __future__ import annotations

import pytest

from evals.bank_synthetic_data import generate_enterprise_bank_dataset
from evals.run_bank_evals import run_all_bank_evals


def test_enterprise_bank_synthetic_dataset_shape():
    ds = generate_enterprise_bank_dataset()
    summary = ds.summary()
    assert summary["table_count"] == 10
    assert set(summary["tables"].keys()) == {
        "customers",
        "accounts",
        "transactions",
        "cards",
        "disputes_chargebacks",
        "refund_requests",
        "fraud_alerts",
        "kyc_aml_profiles",
        "financial_plans",
        "policy_rules",
    }
    assert len(summary["task_log_classes"]) == 5
    assert summary["task_log_count"] >= 30


@pytest.mark.asyncio
async def test_enterprise_bank_long_horizon_agents_and_controls():
    report = await run_all_bank_evals()
    assert report["checks"]["customer_service_approval_flow"]["passed"] is True
    assert report["checks"]["customer_service_rejection_gate"]["passed"] is True
    assert report["checks"]["financial_planner_refund_attestation"]["passed"] is True
    assert report["checks"]["financial_planner_aml_block_and_subagent"]["passed"] is True
    assert report["checks"]["harness_controls"]["payment_tool_blocked"] is True
    assert report["checks"]["harness_controls"]["residency_enforced"] is True
    assert report["checks"]["harness_controls"]["pii_redaction_triggered"] is True
    assert report["checks"]["harness_controls"]["policy_violations_reaching_execution"] == 0
    assert report["replay_eval"]["replay_fidelity"] == 1.0
    assert report["replay_eval"]["cost_saved_pct"] >= 0.30
    assert report["replay_eval"]["passes_promotion_gate"] is True
    assert report["replay_eval"]["corrupted_runs"] == []
