"""Long-Horizon Enterprise Banking Agents & FDE Deployment Builder.

Implements two long-horizon agent task packs operating over the 10-table enterprise bank dataset:
  1. `BankCustomerServicePack` (`bank_customer_service`):
     lookup_customer_360 -> triage_dispute (model) -> fetch_dispute_evidence -> draft_notice (model) -> send_customer_notice (external_comm, approval-gated)
  2. `BankFinancialPlannerPack` (`bank_financial_planner`):
     lookup_customer_360 -> verify_fraud_signals (model) -> attest_refund (model) -> record_attestation (internal_write) -> plan_review (model) -> update_financial_plan (internal_write)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from harness.context import ContextService, Principal, SkillRegistry
from harness.fde import FDEDeployment, RouteRule, RoutingMap
from harness.packs import Action, Done, Finish, ModelCall, ToolCall
from harness.router.adapter import Completion, LiteLLMAdapter
from harness.router.models import ModelSpec
from harness.tools import ToolSpec

from .bank_synthetic_data import EnterpriseBankDataset, generate_enterprise_bank_dataset


def _valid_triage(text: str) -> bool:
    try:
        d = json.loads(text)
        return isinstance(d, dict) and "category" in d and "priority" in d
    except ValueError:
        return False


def _valid_notice(text: str) -> bool:
    try:
        d = json.loads(text)
        return isinstance(d, dict) and isinstance(d.get("subject"), str) and isinstance(d.get("body"), str)
    except ValueError:
        return False


def _valid_fraud_check(text: str) -> bool:
    try:
        d = json.loads(text)
        return isinstance(d, dict) and isinstance(d.get("high_risk"), bool) and "fraud_score" in d
    except ValueError:
        return False


def _valid_attestation(text: str) -> bool:
    try:
        d = json.loads(text)
        return isinstance(d, dict) and isinstance(d.get("attest_approved"), bool) and "reason" in d
    except ValueError:
        return False


def _valid_plan_review(text: str) -> bool:
    try:
        d = json.loads(text)
        return isinstance(d, dict) and isinstance(d.get("recommended_action"), str)
    except ValueError:
        return False


@dataclass
class BankCustomerServicePack:
    """Long-horizon Customer Service Agent handling disputes, provisional credit inquiries, and customer notices."""
    name: str = "bank_customer_service"
    task_class: str = "triage_dispute"
    data_class: str = "confidential"
    allowed_models: list[str] = field(default_factory=lambda: ["m-small", "m-medium", "m-large"])
    restricted_regions: list[str] = field(default_factory=lambda: ["me", "eu"])
    tools: tuple[str, ...] = ("lookup_customer_360", "fetch_dispute_evidence", "send_customer_notice", "read_artifact")
    context_service: ContextService | None = None
    skill_registry: SkillRegistry | None = None
    caller_principal: Principal | None = None

    def policy_entry(self) -> dict[str, list[str]]:
        return {"tools": list(self.tools), "models": list(self.allowed_models), "restricted_regions": list(self.restricted_regions)}

    def _ctx_block(self, query: str, run_id: str, step_id: str) -> str:
        parts: list[str] = []
        if self.skill_registry and self.caller_principal:
            headers = self.skill_registry.list_headers(principal=self.caller_principal)
            if headers:
                parts.append("Skills:\n" + "\n".join(f"- {h['name']} ({h['citation']}): {h['description']}" for h in headers))
        if self.context_service and self.caller_principal:
            hits = self.context_service.retrieve(query, caller=self.caller_principal, run_id=run_id, step_id=step_id, agent_id=self.name)
            if hits:
                parts.append("Untrusted Bank Policy & Memory:\n" + "\n".join(h.as_prompt_block() for h in hits))
        return "\n\n".join(parts)

    def next_action(self, run_input: dict[str, Any], history: list[Done]) -> Action:
        n = len(history)
        run_id = str(run_input.get("_run_id", "cs-run"))
        if n == 0:
            return ToolCall("lookup_customer_360", {"customer_id": run_input["customer_id"]})
        if n == 1:
            if not history[0].result.get("ok"):
                return Finish({"status": "customer_lookup_failed"})
            c360 = history[0].result["output"]
            ctx = self._ctx_block("dispute refund provisional credit notice", run_id, "step-1")
            return ModelCall(
                step_role="triage",
                messages=[
                    {"role": "system", "content": f"Triage the banking dispute and return JSON: {{\"category\": str, \"priority\": str}}.\n{ctx}"},
                    {"role": "user", "content": json.dumps({"customer_360": c360, "dispute_id": run_input["dispute_id"]})},
                ],
                verify=_valid_triage,
            )
        if n == 2:
            return ToolCall("fetch_dispute_evidence", {"dispute_id": run_input["dispute_id"]})
        if n == 3:
            if not history[2].result.get("ok"):
                return Finish({"status": "evidence_lookup_failed"})
            ev = history[2].result["output"]
            triage = history[1].result.get("text", "{}")
            return ModelCall(
                step_role="draft_notice",
                messages=[
                    {"role": "system", "content": "Draft a concise, policy-compliant customer notice without raw PII. Return JSON: {\"subject\": str, \"body\": str}."},
                    {"role": "user", "content": json.dumps({"triage": triage, "evidence": ev})},
                ],
                verify=_valid_notice,
            )
        if n == 4:
            if not history[3].result.get("ok"):
                return Finish({"status": "draft_failed"})
            notice = json.loads(history[3].result["text"])
            return ToolCall("send_customer_notice", {
                "customer_id": run_input["customer_id"],
                "subject": notice["subject"],
                "body": notice["body"],
            })
        return Finish({
            "status": "done",
            "triage": json.loads(history[1].result["text"]),
            "notice_sent": history[4].result.get("ok", False),
            "notice_detail": history[4].result,
        })


@dataclass
class BankFinancialPlannerPack:
    """Long-horizon Financial Planning & Attestation Agent that attests refunds, fraud chargebacks, and plan rebalances."""
    name: str = "bank_financial_planner"
    task_class: str = "attest_refund"
    data_class: str = "confidential"
    allowed_models: list[str] = field(default_factory=lambda: ["m-small", "m-medium", "m-large"])
    restricted_regions: list[str] = field(default_factory=lambda: ["me", "eu"])
    tools: tuple[str, ...] = ("lookup_customer_360", "fetch_dispute_evidence", "record_attestation", "update_financial_plan", "read_artifact")
    context_service: ContextService | None = None
    skill_registry: SkillRegistry | None = None
    caller_principal: Principal | None = None

    def policy_entry(self) -> dict[str, list[str]]:
        return {"tools": list(self.tools), "models": list(self.allowed_models), "restricted_regions": list(self.restricted_regions)}

    def _ctx_block(self, query: str, run_id: str, step_id: str) -> str:
        if not (self.context_service and self.caller_principal):
            return ""
        hits = self.context_service.retrieve(query, caller=self.caller_principal, run_id=run_id, step_id=step_id, agent_id=self.name)
        return "Untrusted Bank Policy & Memory:\n" + "\n".join(h.as_prompt_block() for h in hits) if hits else ""

    def next_action(self, run_input: dict[str, Any], history: list[Done]) -> Action:
        n = len(history)
        run_id = str(run_input.get("_run_id", "fp-run"))
        if n == 0:
            return ToolCall("lookup_customer_360", {"customer_id": run_input["customer_id"]})
        if n == 1:
            return ToolCall("fetch_dispute_evidence", {"dispute_id": run_input["dispute_id"]})
        if n == 2:
            c360 = history[0].result.get("output", {})
            ev = history[1].result.get("output", {})
            ctx = self._ctx_block("fraud chargeback aml kyc", run_id, "step-2")
            return ModelCall(
                step_role="fraud_check",
                messages=[
                    {"role": "system", "content": f"Evaluate fraud and AML signals. Return JSON: {{\"high_risk\": bool, \"fraud_score\": float, \"aml_blocked\": bool}}.\n{ctx}"},
                    {"role": "user", "content": json.dumps({"customer_360": c360, "evidence": ev})},
                ],
                verify=_valid_fraud_check,
            )
        if n == 3:
            c360 = history[0].result.get("output", {})
            ev = history[1].result.get("output", {})
            fraud = history[2].result.get("text", "{}")
            ctx = self._ctx_block("refund attestation policy aml", run_id, "step-3")
            return ModelCall(
                step_role="attest",
                messages=[
                    {"role": "system", "content": f"Attest refund or chargeback eligibility. Return JSON: {{\"attest_approved\": bool, \"reason\": str, \"amount_usd\": float}}.\n{ctx}"},
                    {"role": "user", "content": json.dumps({"customer_360": c360, "evidence": ev, "fraud_check": fraud})},
                ],
                verify=_valid_attestation,
            )
        if n == 4:
            att = json.loads(history[3].result["text"])
            return ToolCall("record_attestation", {
                "refund_id": run_input["refund_id"],
                "approved": att["attest_approved"],
                "reason": att["reason"],
            })
        if n == 5:
            c360 = history[0].result.get("output", {})
            att = json.loads(history[3].result["text"])
            return ModelCall(
                step_role="plan_review",
                messages=[
                    {"role": "system", "content": "Review customer financial plan liquidity after attestation. Return JSON: {\"recommended_action\": str, \"rebalance_required\": bool}."},
                    {"role": "user", "content": json.dumps({"financial_plan": c360.get("financial_plan"), "attestation": att})},
                ],
                verify=_valid_plan_review,
            )
        if n == 6:
            plan = json.loads(history[5].result["text"])
            return ToolCall("update_financial_plan", {
                "customer_id": run_input["customer_id"],
                "recommended_action": plan["recommended_action"],
            })
        return Finish({
            "status": "done",
            "fraud_check": json.loads(history[2].result["text"]),
            "attestation": json.loads(history[3].result["text"]),
            "attestation_recorded": history[4].result.get("ok", False),
            "plan_review": json.loads(history[5].result["text"]),
            "plan_updated": history[6].result.get("ok", False),
        })


class _BankMockAdapter(LiteLLMAdapter):
    """Deterministic banking model adapter simulating realistic outputs over the 10-table dataset."""

    async def complete(self, model: ModelSpec, messages: list[dict], max_tokens: int = 1024) -> Completion:
        prompt = " ".join(str(m.get("content", "")) for m in messages)
        low = prompt.lower()

        if "triage the banking dispute" in low:
            txt = json.dumps({"category": "duplicate_processing" if "duplicate_processing" in low else "fraud_chargeback", "priority": "high"})
        elif "draft a concise, policy-compliant customer notice" in low:
            txt = json.dumps({
                "subject": "Update on Your Transaction Dispute",
                "body": "Your dispute has been logged and is under provisional credit attestation per bank policy.",
            })
        elif "evaluate fraud and aml signals" in low:
            aml_blocked = '"aml_flag": true' in low
            high_risk = aml_blocked or '"score": 0.87' in low or '"score": 0.94' in low
            score = 0.94 if aml_blocked else (0.87 if '"score": 0.87' in low else 0.18)
            txt = json.dumps({"high_risk": high_risk, "fraud_score": score, "aml_blocked": aml_blocked})
        elif "attest refund or chargeback eligibility" in low:
            aml_blocked = '"aml_blocked": true' in low or '"aml_flag": true' in low
            if aml_blocked:
                txt = json.dumps({"attest_approved": False, "reason": "Blocked by AML/KYC policy (POL-BNK-003)", "amount_usd": 0.0})
            else:
                txt = json.dumps({"attest_approved": True, "reason": "Verified duplicate/fraud claim under POL-BNK-001/002", "amount_usd": 680.0})
        elif "review customer financial plan liquidity" in low:
            txt = json.dumps({"recommended_action": "maintain_buffer" if '"rebalance_needed": false' in low else "reallocate_reserve_liquidity", "rebalance_required": '"rebalance_needed": true' in low})
        elif "risk_level" in low or "clauses" in low:
            txt = json.dumps({
                "risk_level": "high" if "aml" in low or "wire" in low else "low",
                "clauses": ["Net-45 settlement clause", "AML/KYC compliance hold clause"],
                "policy_citation": "[policy:POL-BNK-003@v1]",
            })
        else:
            txt = json.dumps({"status": "ok"})
        return Completion(text=txt, prompt_tokens=260, completion_tokens=90, latency_ms=5.0, provider_model=model.litellm_model)


def build_bank_deployment(dataset: EnterpriseBankDataset | None = None) -> tuple[FDEDeployment, EnterpriseBankDataset]:
    """Builds an FDE Enterprise Bank deployment wired to the 10 data tables, 5 task logs, and both long-horizon agents."""
    ds = dataset or generate_enterprise_bank_dataset()
    routing_map = RoutingMap(
        rules=[
            RouteRule("triage_dispute", "triage", {"m-small": 0.945, "m-medium": 0.945, "m-large": 0.95}, quality_tolerance=0.03),
            RouteRule("triage_dispute", "draft_notice", {"m-small": 0.94, "m-medium": 0.94, "m-large": 0.945}, quality_tolerance=0.03),
            RouteRule("attest_refund", "fraud_check", {"m-small": 0.78, "m-medium": 0.945, "m-large": 0.95}, quality_tolerance=0.03),
            RouteRule("attest_refund", "attest", {"m-small": 0.76, "m-medium": 0.945, "m-large": 0.95}, quality_tolerance=0.03),
            RouteRule("attest_refund", "plan_review", {"m-small": 0.80, "m-medium": 0.94, "m-large": 0.95}, quality_tolerance=0.03),
            RouteRule("attest_refund", "*", {"m-small": 0.78, "m-medium": 0.945, "m-large": 0.95}, quality_tolerance=0.03),
            RouteRule("triage_dispute", "*", {"m-small": 0.94, "m-medium": 0.945, "m-large": 0.95}, quality_tolerance=0.03),
            RouteRule("clause_review", "clause_extraction", {"m-small": 0.78, "m-medium": 0.94, "m-large": 0.95}, quality_tolerance=0.03),
        ]
    )

    dep = FDEDeployment(
        org_id="org-enterprise-bank",
        org_name="Global Enterprise Bank",
        region="eu",
        routing_map=routing_map,
        adapter=_BankMockAdapter(),
    )
    dep.default_principal = Principal(
        id="bank-fde-agent",
        org_id="org-enterprise-bank",
        kind="service",
        teams=("finance", "fde", "customer_service", "risk"),
        roles=("admin", "memory_approver"),
        max_clearance="restricted",
    )

    # Sprinkle in enterprise bank context, skills, memory, and the 5 task log streams
    dep.sprinkle_context(ds.context_items)
    dep.sprinkle_skills(ds.skills)
    dep.sprinkle_memory(ds.memory_facts)
    dep.sprinkle_task_logs(ds.task_logs)

    # Register banking tools backed by the 10 synthetic tables
    tables = ds.tables

    async def _lookup_customer_360(args: dict[str, Any]) -> dict[str, Any]:
        cid = args["customer_id"]
        cust = next((c for c in tables["customers"] if c["customer_id"] == cid), None)
        accts = [a for a in tables["accounts"] if a["customer_id"] == cid]
        acct_ids = {a["account_id"] for a in accts}
        cards = [c for c in tables["cards"] if c["account_id"] in acct_ids]
        txns = [t for t in tables["transactions"] if t["account_id"] in acct_ids]
        kyc = next((k for k in tables["kyc_aml_profiles"] if k["customer_id"] == cid), None)
        plan = next((p for p in tables["financial_plans"] if p["customer_id"] == cid), None)
        return {"customer": cust, "accounts": accts, "cards": cards, "transactions": txns, "kyc_aml": kyc, "financial_plan": plan}

    async def _fetch_dispute_evidence(args: dict[str, Any]) -> dict[str, Any]:
        did = args["dispute_id"]
        dispute = next((d for d in tables["disputes_chargebacks"] if d["dispute_id"] == did), None)
        refund = next((r for r in tables["refund_requests"] if r["dispute_id"] == did), None)
        alert = next((f for f in tables["fraud_alerts"] if dispute and f["txn_id"] == dispute["txn_id"]), None)
        return {"dispute": dispute, "refund_request": refund, "fraud_alert": alert}

    async def _send_customer_notice(args: dict[str, Any]) -> dict[str, Any]:
        return {"dispatched": True, "customer_id": args["customer_id"], "subject": args["subject"]}

    async def _record_attestation(args: dict[str, Any]) -> dict[str, Any]:
        return {"recorded": True, "refund_id": args["refund_id"], "approved": args["approved"], "reason": args["reason"]}

    async def _update_financial_plan(args: dict[str, Any]) -> dict[str, Any]:
        return {"updated": True, "customer_id": args["customer_id"], "recommended_action": args["recommended_action"]}

    dep.tools.register(ToolSpec("lookup_customer_360", "read", _lookup_customer_360, required=("customer_id",), idempotent=True))
    dep.tools.register(ToolSpec("fetch_dispute_evidence", "read", _fetch_dispute_evidence, required=("dispute_id",), idempotent=True))
    dep.tools.register(ToolSpec("send_customer_notice", "external_comm", _send_customer_notice, required=("customer_id", "subject", "body"), idempotent=False))
    dep.tools.register(ToolSpec("record_attestation", "internal_write", _record_attestation, required=("refund_id", "approved"), idempotent=True))
    dep.tools.register(ToolSpec("update_financial_plan", "internal_write", _update_financial_plan, required=("customer_id", "recommended_action"), idempotent=True))

    # Register both long-horizon banking agent task packs
    dep.register_pack(BankCustomerServicePack(
        context_service=dep.context,
        skill_registry=dep.skills,
        caller_principal=dep.default_principal,
    ))
    dep.register_pack(BankFinancialPlannerPack(
        context_service=dep.context,
        skill_registry=dep.skills,
        caller_principal=dep.default_principal,
    ))

    return dep, ds
