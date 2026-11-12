"""Synthetic Enterprise Bank Dataset Generator (10 relational data tables + 5 task log streams + policies/skills/memory).

Tables:
  1. customers
  2. accounts
  3. transactions
  4. cards
  5. disputes_chargebacks
  6. refund_requests
  7. fraud_alerts
  8. kyc_aml_profiles
  9. financial_plans
  10. policy_rules

Task Logs (5 distinct banking task classes / step roles):
  1. triage_dispute
  2. verify_fraud_signals
  3. attest_refund
  4. draft_customer_notice
  5. rebalance_financial_plan
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.fde import TaskLogRecord


@dataclass(frozen=True)
class EnterpriseBankDataset:
    tables: dict[str, list[dict[str, Any]]]
    task_logs: list[TaskLogRecord]
    context_items: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    memory_facts: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "table_count": len(self.tables),
            "tables": {name: len(rows) for name, rows in self.tables.items()},
            "task_log_count": len(self.task_logs),
            "task_log_classes": sorted({t.task_class for t in self.task_logs}),
            "context_items": len(self.context_items),
            "skills": len(self.skills),
            "memory_facts": len(self.memory_facts),
        }

    def export_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "tables": self.tables,
            "task_logs": [
                {
                    "task_class": t.task_class,
                    "step_role": t.step_role,
                    "model_id": t.model_id,
                    "success": t.success,
                    "cost_usd": t.cost_usd,
                    "latency_ms": t.latency_ms,
                    "prompt_tokens": t.prompt_tokens,
                    "completion_tokens": t.completion_tokens,
                    "verifier_score": t.verifier_score,
                }
                for t in self.task_logs
            ],
            "context_items": self.context_items,
            "skills": self.skills,
            "memory_facts": self.memory_facts,
        }
        out.write_text(json.dumps(payload, indent=2))
        return out


def generate_enterprise_bank_dataset() -> EnterpriseBankDataset:
    """Generates a deterministic, relationally consistent 10-table enterprise bank dataset and 5 task log streams."""
    tables: dict[str, list[dict[str, Any]]] = {
        # 1. customers
        "customers": [
            {
                "customer_id": "CUST-1001",
                "full_name": "Aarav Mehta",
                "segment": "retail_premier",
                "email": "aarav.mehta@example.com",
                "region": "eu",
                "tenure_years": 6,
                "risk_tier": "low",
            },
            {
                "customer_id": "CUST-1002",
                "full_name": "Nadia Al-Mansoor",
                "segment": "private_wealth",
                "email": "nadia.almansoor@example.com",
                "region": "me",
                "tenure_years": 9,
                "risk_tier": "medium",
            },
            {
                "customer_id": "CUST-1003",
                "full_name": "Marcus Vance",
                "segment": "commercial_smb",
                "email": "marcus.vance@example.com",
                "region": "eu",
                "tenure_years": 2,
                "risk_tier": "high",
            },
            {
                "customer_id": "CUST-1004",
                "full_name": "Elena Rostova",
                "segment": "retail_standard",
                "email": "elena.rostova@example.com",
                "region": "eu",
                "tenure_years": 4,
                "risk_tier": "low",
            },
        ],
        # 2. accounts
        "accounts": [
            {"account_id": "ACC-5001", "customer_id": "CUST-1001", "type": "checking", "currency": "USD", "balance_usd": 24500.00, "status": "active"},
            {"account_id": "ACC-5002", "customer_id": "CUST-1002", "type": "wealth_reserve", "currency": "USD", "balance_usd": 412000.00, "status": "active"},
            {"account_id": "ACC-5003", "customer_id": "CUST-1003", "type": "business_checking", "currency": "USD", "balance_usd": 18300.00, "status": "restricted_review"},
            {"account_id": "ACC-5004", "customer_id": "CUST-1004", "type": "checking", "currency": "USD", "balance_usd": 6200.00, "status": "active"},
        ],
        # 3. transactions
        "transactions": [
            {"txn_id": "TXN-9001", "account_id": "ACC-5001", "merchant": "Nordic Air", "mcc": "4511", "amount_usd": 680.00, "channel": "card_not_present", "status": "settled", "duplicate_of": None},
            {"txn_id": "TXN-9002", "account_id": "ACC-5001", "merchant": "Nordic Air", "mcc": "4511", "amount_usd": 680.00, "channel": "card_not_present", "status": "settled", "duplicate_of": "TXN-9001"},
            {"txn_id": "TXN-9003", "account_id": "ACC-5002", "merchant": "ElectroLux Zurich", "mcc": "5732", "amount_usd": 14200.00, "channel": "pos_chip", "status": "settled", "duplicate_of": None},
            {"txn_id": "TXN-9004", "account_id": "ACC-5003", "merchant": "CryptoBridge FX", "mcc": "6051", "amount_usd": 9800.00, "channel": "wire_api", "status": "settled", "duplicate_of": None},
            {"txn_id": "TXN-9005", "account_id": "ACC-5004", "merchant": "CloudBooks SaaS", "mcc": "5734", "amount_usd": 149.00, "channel": "recurring_card", "status": "settled", "duplicate_of": None},
        ],
        # 4. cards
        "cards": [
            {"card_id": "CARD-301", "account_id": "ACC-5001", "network": "Visa", "last4": "4412", "status": "active", "card_present_enabled": True},
            {"card_id": "CARD-302", "account_id": "ACC-5002", "network": "Mastercard", "last4": "8891", "status": "frozen_pending_review", "card_present_enabled": True},
            {"card_id": "CARD-303", "account_id": "ACC-5003", "network": "Visa", "last4": "1104", "status": "active", "card_present_enabled": False},
            {"card_id": "CARD-304", "account_id": "ACC-5004", "network": "Visa", "last4": "7720", "status": "active", "card_present_enabled": True},
        ],
        # 5. disputes_chargebacks
        "disputes_chargebacks": [
            {
                "dispute_id": "DSP-7001",
                "txn_id": "TXN-9002",
                "customer_id": "CUST-1001",
                "reason_code": "duplicate_processing",
                "amount_usd": 680.00,
                "status": "open",
                "evidence_strength": "high",
            },
            {
                "dispute_id": "DSP-7002",
                "txn_id": "TXN-9003",
                "customer_id": "CUST-1002",
                "reason_code": "unrecognized_high_value_charge",
                "amount_usd": 14200.00,
                "status": "open",
                "evidence_strength": "medium",
            },
            {
                "dispute_id": "DSP-7003",
                "txn_id": "TXN-9004",
                "customer_id": "CUST-1003",
                "reason_code": "unauthorized_wire_claim",
                "amount_usd": 9800.00,
                "status": "escalated_aml",
                "evidence_strength": "low",
            },
        ],
        # 6. refund_requests
        "refund_requests": [
            {"refund_id": "REF-8001", "dispute_id": "DSP-7001", "customer_id": "CUST-1001", "amount_usd": 680.00, "eligible_auto_attest": True, "status": "pending_attestation"},
            {"refund_id": "REF-8002", "dispute_id": "DSP-7002", "customer_id": "CUST-1002", "amount_usd": 14200.00, "eligible_auto_attest": False, "status": "pending_attestation"},
            {"refund_id": "REF-8003", "dispute_id": "DSP-7003", "customer_id": "CUST-1003", "amount_usd": 9800.00, "eligible_auto_attest": False, "status": "blocked_aml_review"},
        ],
        # 7. fraud_alerts
        "fraud_alerts": [
            {"alert_id": "FRD-6001", "txn_id": "TXN-9002", "score": 0.18, "signal": "merchant_duplicate_batch", "confirmed_fraud": False},
            {"alert_id": "FRD-6002", "txn_id": "TXN-9003", "score": 0.87, "signal": "geo_velocity_anomaly_zurich", "confirmed_fraud": True},
            {"alert_id": "FRD-6003", "txn_id": "TXN-9004", "score": 0.94, "signal": "sanctioned_corridor_velocity", "confirmed_fraud": True},
        ],
        # 8. kyc_aml_profiles
        "kyc_aml_profiles": [
            {"customer_id": "CUST-1001", "kyc_status": "verified", "aml_flag": False, "pep": False, "last_screened": "2026-10-01"},
            {"customer_id": "CUST-1002", "kyc_status": "verified", "aml_flag": False, "pep": False, "last_screened": "2026-10-15"},
            {"customer_id": "CUST-1003", "kyc_status": "enhanced_due_diligence", "aml_flag": True, "pep": False, "last_screened": "2026-11-01"},
            {"customer_id": "CUST-1004", "kyc_status": "verified", "aml_flag": False, "pep": False, "last_screened": "2026-09-20"},
        ],
        # 9. financial_plans
        "financial_plans": [
            {"plan_id": "PLN-4001", "customer_id": "CUST-1001", "monthly_cashflow_usd": 8200.0, "emergency_buffer_months": 6, "rebalance_needed": False},
            {"plan_id": "PLN-4002", "customer_id": "CUST-1002", "monthly_cashflow_usd": 42000.0, "emergency_buffer_months": 12, "rebalance_needed": True},
            {"plan_id": "PLN-4003", "customer_id": "CUST-1003", "monthly_cashflow_usd": 15000.0, "emergency_buffer_months": 2, "rebalance_needed": True},
        ],
        # 10. policy_rules
        "policy_rules": [
            {"rule_id": "POL-BNK-001", "domain": "refunds", "summary": "Duplicate merchant charges under $2,500 with verified KYC may be attested for provisional credit; agent never executes wire/payment directly."},
            {"rule_id": "POL-BNK-002", "domain": "chargebacks", "summary": "Fraud alerts >= 0.80 require second-stage fraud signal verification before chargeback attestation."},
            {"rule_id": "POL-BNK-003", "domain": "aml", "summary": "Accounts with aml_flag=True or status=restricted_review must block automated refund attestation and escalate to compliance."},
            {"rule_id": "POL-BNK-004", "domain": "comms", "summary": "All outbound customer notices (external_comm) are draft-only and require human approval before dispatch."},
            {"rule_id": "POL-BNK-005", "domain": "privacy", "summary": "Customer emails, SSNs, and raw account numbers must be redacted in post_model outputs."},
        ],
    }

    # 5 Task Log streams across 5 banking task classes / step roles
    task_logs: list[TaskLogRecord] = [
        # 1. triage_dispute: m-small performs well at 25x lower cost than m-large
        *[
            TaskLogRecord("triage_dispute", "triage", "m-small", True, 0.00006, 72.0, 220, 80, verifier_score=0.95)
            for _ in range(8)
        ],
        TaskLogRecord("triage_dispute", "triage", "m-large", True, 0.00180, 310.0, 220, 80, verifier_score=0.96),

        # 2. verify_fraud_signals: m-medium hits 0.94 vs m-small 0.78 -> m-medium is optimal
        *[
            TaskLogRecord("verify_fraud_signals", "fraud_check", "m-medium", True, 0.00042, 145.0, 260, 95, verifier_score=0.94)
            for _ in range(8)
        ],
        *[
            TaskLogRecord("verify_fraud_signals", "fraud_check", "m-small", False, 0.00007, 68.0, 260, 95, verifier_score=0.76)
            for _ in range(4)
        ],

        # 3. attest_refund: m-medium hits 0.95 at 73% cost cut vs m-large
        *[
            TaskLogRecord("attest_refund", "attest", "m-medium", True, 0.00048, 160.0, 300, 110, verifier_score=0.95)
            for _ in range(8)
        ],
        TaskLogRecord("attest_refund", "attest", "m-large", True, 0.00195, 340.0, 300, 110, verifier_score=0.96),

        # 4. draft_customer_notice: m-small hits 0.94
        *[
            TaskLogRecord("draft_customer_notice", "draft_notice", "m-small", True, 0.00008, 80.0, 240, 120, verifier_score=0.94)
            for _ in range(8)
        ],

        # 5. rebalance_financial_plan: m-medium hits 0.935 vs m-large 0.95
        *[
            TaskLogRecord("rebalance_financial_plan", "plan_review", "m-medium", True, 0.00050, 175.0, 320, 130, verifier_score=0.94)
            for _ in range(8)
        ],
    ]

    context_items = [
        {
            "source_id": "bank-policy-corpus",
            "external_id": r["rule_id"],
            "kind": "policy",
            "title": f"Enterprise Bank Policy ({r['domain'].upper()})",
            "content": r["summary"],
            "version_no": 1,
            "owner": "chief-risk-officer@bank.example",
            "data_class": "internal",
            "allowed_teams": ["finance", "fde", "customer_service", "risk"],
        }
        for r in tables["policy_rules"]
    ]

    skills = [
        {
            "name": "reg_e_chargeback_sop",
            "description": "Regulation E provisional credit and chargeback evidence verification procedure.",
            "body": "1) Check kyc_aml_profiles for aml_flag. 2) Check fraud_alerts score. 3) Attest refund only if aml_flag is false; never trigger direct payment.",
            "version_no": 1,
            "data_class": "internal",
        },
        {
            "name": "wealth_plan_rebalance_sop",
            "description": "Financial planning buffer and cashflow impact assessment after chargeback/refund.",
            "body": "Ensure emergency_buffer_months >= 3 before recommending discretionary liquidity reallocation.",
            "version_no": 1,
            "data_class": "confidential",
        },
    ]

    memory_facts = [
        {
            "scope_type": "org",
            "scope_id": "org-enterprise-bank",
            "key": "provisional_credit_cap",
            "content": "Automated refund attestations above $2,500 require enhanced fraud signal verification.",
            "data_class": "internal",
            "tags": ["refund", "chargeback", "fraud"],
        },
        {
            "scope_type": "team",
            "scope_id": "finance",
            "key": "aml_hard_block_rule",
            "content": "Any customer with aml_flag=True (e.g. CUST-1003) must have refund attestation denied and escalated.",
            "data_class": "confidential",
            "tags": ["aml", "kyc", "refund"],
        },
    ]

    return EnterpriseBankDataset(
        tables=tables,
        task_logs=task_logs,
        context_items=context_items,
        skills=skills,
        memory_facts=memory_facts,
    )
