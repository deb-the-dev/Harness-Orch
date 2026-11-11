"""End-to-end tests for FDE Enterprise Customization, Context & Scoped Memory, Progressive Skills,
Adaptive Task-Log Routing, Qualified Finance Task Packs, Subagents, Checkpoint Forking, and Replay."""
from __future__ import annotations

import pytest

from harness.context import (
    ContextService,
    GraphitiMemoryAdapter,
    Mem0MemoryAdapter,
    Principal,
    ProcedureSkill,
    SkillRegistry,
)
from harness.events import SQLiteEventLog
from harness.fde import (
    AdaptiveOutcomeSelector,
    FDEDeployment,
    RouteRule,
    RoutingMap,
    TaskLogRecord,
)
from harness.runs import RunState


def test_context_priority_permissions_and_untrusted_audit():
    log = SQLiteEventLog()
    ctx = ContextService(log, "org-1", engine=GraphitiMemoryAdapter())

    # Ingest a code item (priority 4) and a policy item (priority 1)
    ctx.ingest_knowledge(
        source_id="src-code",
        external_id="repo-main",
        kind="code",
        title="Invoice parser utility",
        content="invoice payment terms parser code",
        data_class="internal",
    )
    p_v1 = ctx.ingest_knowledge(
        source_id="src-pol",
        external_id="POL-AP-1",
        kind="policy",
        title="Invoice Payment Policy",
        content="invoice payment terms must be Net-30",
        version_no=1,
        effective_from=100.0,
        owner="cfo",
        data_class="internal",
    )
    # Supersede policy with v2
    p_v2 = ctx.ingest_knowledge(
        source_id="src-pol",
        external_id="POL-AP-1",
        kind="policy",
        title="Invoice Payment Policy",
        content="invoice payment terms must be Net-45 per updated directive",
        version_no=2,
        effective_from=200.0,
        owner="cfo",
        data_class="internal",
    )
    # Restricted item for finance team only
    ctx.ingest_knowledge(
        source_id="src-sec",
        external_id="POL-RESTRICTED",
        kind="policy",
        title="Restricted Executive Invoice Policy",
        content="invoice payment executive override rules",
        data_class="restricted",
        allowed_teams=("executive",),
    )

    caller = Principal(id="u-1", org_id="org-1", teams=("finance",), max_clearance="confidential")
    hits = ctx.retrieve("invoice payment terms", caller=caller, run_id="r-ctx", step_id="step-0", at_time=300.0)

    assert len(hits) == 2
    # Policy (priority_rank=1) must rank ahead of code (priority_rank=4), and return superseded v2
    assert hits[0].kind == "policy"
    assert hits[0].version_no == 2
    assert p_v2.citation in hits[0].citation
    assert "Net-45" in hits[0].content
    assert hits[0].trust == "untrusted"
    assert hits[1].kind == "code"

    # As-of query at t=150 retrieves v1
    historical = ctx.retrieve("invoice payment terms", caller=caller, run_id="r-ctx", step_id="step-hist", at_time=150.0)
    assert historical[0].version_no == 1
    assert p_v1.citation in historical[0].citation

    # Verify memory.access audit events were logged with untrusted tag
    access_events = [e for e in log.read("org-1", "r-ctx") if e.type == "memory.access"]
    assert len(access_events) == 2
    assert access_events[0].payload["trust"] == "untrusted"
    assert access_events[0].payload["items"][0]["version_no"] == 2


def test_memory_review_workflow_revocation_and_swappable_engines():
    log = SQLiteEventLog()
    ctx = ContextService(log, "org-1", engine=GraphitiMemoryAdapter())

    regular_user = Principal(id="u-reg", org_id="org-1", teams=("finance",), roles=("analyst",), max_clearance="internal")
    approver = Principal(id="u-adm", org_id="org-1", teams=("finance",), roles=("memory_approver",), max_clearance="restricted")

    # High-impact org-scope write by non-approver enters "pending" review state and is hidden from retrieval
    ref = ctx.write_memory(
        scope_type="org",
        scope_id="org-1",
        key="vendor_sla_rule",
        content="Vendor SLA penalty is 5% per outage hour",
        caller=regular_user,
        source_id="src-doc-99",
    )
    assert ref.review_state == "pending"
    assert ctx.retrieve("Vendor SLA penalty", caller=regular_user, run_id="r-mem") == []

    # Approve the memory write -> now retrievable
    approved_ref = ctx.review_memory(ref.id, "approved", reviewer=approver)
    assert approved_ref.review_state == "approved"
    hits = ctx.retrieve("Vendor SLA penalty", caller=regular_user, run_id="r-mem")
    assert len(hits) == 1
    assert hits[0].trust == "untrusted"

    # Revoke source access (PRD M7) -> immediately hides derived memory
    revoked = ctx.revoke_by_source("src-doc-99")
    assert ref.id in revoked
    assert ctx.retrieve("Vendor SLA penalty", caller=regular_user, run_id="r-mem") == []

    # Swap engine to Mem0 adapter (PRD M1)
    ctx.swap_engine(Mem0MemoryAdapter())
    ref2 = ctx.write_memory(
        scope_type="user",
        scope_id="u-reg",
        key="user_pref",
        content="Prefer concise JSON tables for SLA reports",
        caller=regular_user,
    )
    assert ref2.engine == "mem0"
    assert ref2.review_state == "approved"
    hits2 = ctx.retrieve("SLA reports JSON tables", caller=regular_user, run_id="r-mem")
    assert len(hits2) == 1
    assert "mem0" in hits2[0].citation


def test_progressive_disclosure_skills():
    log = SQLiteEventLog()
    reg = SkillRegistry(log, "org-1")
    reg.register(ProcedureSkill(
        name="ap_escalation_sop",
        org_id="org-1",
        description="Standard operating procedure for escalating disputed invoices.",
        body="Full SOP body: verify PO, contact procurement, hold ledger entry.",
        version_no=3,
        allowed_teams=("finance",),
    ))
    caller = Principal(id="u-1", org_id="org-1", teams=("finance",), max_clearance="internal")
    headers = reg.list_headers(principal=caller)
    assert len(headers) == 1
    assert headers[0]["name"] == "ap_escalation_sop"
    assert "Full SOP body" not in str(headers[0])

    loaded = reg.load_body("ap_escalation_sop", run_id="r-skill", step_id="s-1", principal=caller)
    assert loaded["ok"] is True
    assert "Full SOP body" in loaded["body"]
    assert loaded["trust"] == "untrusted"


def test_adaptive_outcome_selector_task_logs_and_cache_affinity():
    dep = FDEDeployment(org_id="org-opt")
    catalogue = list(dep.catalogue.values())

    # Initially for (reconcile, reconcile_terms), m-small prior is 0.82 and m-large is 0.94 (outside 0.03 tolerance)
    # m-medium is 0.92 (within 0.03 tolerance of 0.94) and cheaper than m-large, so m-medium ranks first!
    initial_rank = dep.selector.rank("reconcile", "reconcile_terms", catalogue)
    assert initial_rank[0].model.id == "m-medium"

    # Now FDE team sprinkles in 20 task logs showing m-small achieves 0.96 empirical success rate on this org's traffic!
    dep.sprinkle_task_logs([
        TaskLogRecord(
            task_class="reconcile",
            step_role="reconcile_terms",
            model_id="m-small",
            success=True,
            cost_usd=0.00005,
            latency_ms=60.0,
            verifier_score=0.96,
        )
        for _ in range(20)
    ])

    # With task log evidence blended in, m-small is now within the top quality band and is the cheapest & fastest -> ranks #1!
    updated_rank = dep.selector.rank("reconcile", "reconcile_terms", catalogue)
    assert updated_rank[0].model.id == "m-small"
    assert "blended_prior+task_logs" in updated_rank[0].source

    # Verify cache affinity bonus (PRD C7) when last_served_model is set
    dep.selector.last_served_model = "m-medium"
    affinity_rank = dep.selector.rank("reconcile", "reconcile_terms", catalogue)
    medium_entry = next(s for s in affinity_rank if s.model.id == "m-medium")
    assert "cache_affinity" in medium_entry.source


@pytest.mark.asyncio
async def test_all_four_finance_packs_and_fde_workflow_pack():
    dep = FDEDeployment(org_id="org-fin")
    dep.sprinkle_context([
        {
            "external_id": "POL-FIN-10",
            "kind": "policy",
            "title": "Invoice & Contract Policy",
            "content": "All invoices must match contract caps and Net-45 terms.",
        }
    ])
    dep.sprinkle_skills([
        {
            "name": "invoice_audit_skill",
            "description": "Audit checklist for enterprise invoices.",
            "body": "Check vendor name and amount.",
        }
    ])

    # 1. ContractPaymentsPack (completes without approval wait since tools are read + internal_write)
    v_cp = await dep.submit_and_run("contract_payments", {"contract_id": "CTR-100", "invoice_text": "Acme $4250"})
    assert v_cp.state == RunState.SUCCEEDED

    # 2. PayoutTermsReviewPack
    v_pt = await dep.submit_and_run("payout_terms_review", {"contract_id": "CTR-200", "contract_text": "Net-45 terms"})
    assert v_pt.state == RunState.SUCCEEDED

    # 3. VendorRenewalNegotiationPack (external_comm pauses for human approval, then finishes on approval)
    v_vr = await dep.submit_and_run("vendor_renewal", {"vendor": "Acme Cloud", "current_spend": 60000})
    assert v_vr.state == RunState.APPROVAL_WAIT
    approval_id = v_vr.pending_approval["approval_id"]
    dep.orchestrator.decide(v_vr.run_id, approval_id=approval_id, approved=True, approver="vp-procurement")
    v_vr_done = await dep.worker.run(v_vr.run_id)
    assert v_vr_done.state == RunState.SUCCEEDED

    # 4. Custom FDEWorkflowPack with context & skill injection
    dep.register_workflow_pack(
        name="custom_ap_triage",
        task_class="extract",
        steps=[
            {
                "kind": "model",
                "step_role": "extract",
                "system_prompt": "Extract invoice fields as JSON.",
                "input_key": "invoice_text",
                "required_json_keys": ["vendor", "amount"],
                "context_query": "Invoice & Contract Policy",
            },
            {
                "kind": "tool",
                "tool_name": "log_invoice",
                "tool_arg_mapping": {"vendor": "vendor", "amount": "amount"},
            },
        ],
    )
    v_wf = await dep.submit_and_run("custom_ap_triage", {"invoice_text": "Acme Cloud Corp $4250"})
    assert v_wf.state == RunState.SUCCEEDED
    # Verify that memory.access was logged during the custom FDE workflow run
    evs = dep.log.read("org-fin", v_wf.run_id)
    assert any(e.type == "memory.access" for e in evs)


@pytest.mark.asyncio
async def test_isolated_subagent_and_checkpoint_fork_and_replay():
    dep = FDEDeployment(org_id="org-sub")
    parent_view = await dep.submit_and_run("contract_payments", {"contract_id": "CTR-1", "invoice_text": "Invoice 1"}, budget_usd=0.50)
    assert parent_view.state == RunState.SUCCEEDED

    # Spawn isolated child subagent linked to parent_run_id (PRD H8, C8)
    sub_res = await dep.subagents.run_subagent(
        parent_run_id=parent_view.run_id,
        child_pack_name="payout_terms_review",
        child_input={"contract_id": "CTR-1", "contract_text": "Net-45 payout"},
        requested_budget_usd=0.20,
    )
    assert sub_res["ok"] is True
    assert sub_res["parent_run_id"] == parent_view.run_id

    # Fork parent run from checkpoint seq 4 (PRD H2)
    fork_view = dep.fork_from_checkpoint(parent_view.run_id, upto_seq=4)
    assert fork_view.state == RunState.QUEUED
    fork_done = await dep.worker.run(fork_view.run_id)
    assert fork_done.state == RunState.SUCCEEDED

    # Evaluate deterministic replay & promotion gate against m-large best-single-model baseline (PRD P4, P6)
    report = dep.replay_engine.evaluate(dep.selector, baseline_model_id="m-large")
    assert report.total_steps >= 2
    assert report.replay_fidelity == 1.0
    assert report.cost_saved_pct > 0.30
    assert report.passes_promotion_gate is True
