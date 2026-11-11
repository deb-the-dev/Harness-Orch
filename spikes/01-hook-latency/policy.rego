package hooks

import rego.v1

# Fail closed: an unknown pack resolves to an empty grant set, never to "undefined".
# (Without this, `not input.tool in pack.tools` is undefined for an
# unknown pack, the rule does not fire, and the call is allowed. See RESULTS.md.)
pack := object.get(data.packs, input.pack, {"tools": [], "models": [], "restricted_regions": []})

# Verdict precedence: block > require_approval > modify > allow.
# Reasons: sorted union of every triggered rule.

verdict := "block" if {
	count(block_reasons) > 0
} else := "require_approval" if {
	count(approval_reasons) > 0
} else := "modify" if {
	count(modify_reasons) > 0
} else := "allow"

# Fail closed on malformed input: a missing or mistyped field must never fall through to allow.
decision := {"verdict": "block", "reasons": ["malformed_input"]} if {
	not valid_input
} else := {
	"verdict": verdict,
	"reasons": sort([r | some r in (block_reasons | approval_reasons | modify_reasons)]),
}

valid_input if {
	input.action_class in {"read", "internal_write", "external_comm", "payment"}
	valid_for_hook
}

valid_for_hook if {
	input.hook == "pre_model"
	is_number(input.spend.run_usd)
	is_number(input.spend.est_call_usd)
	is_number(input.spend.run_limit_usd)
	input.data_class in {"public", "internal", "confidential", "restricted"}
	is_string(input.model)
	is_string(input.model_region)
}

valid_for_hook if {
	input.hook == "post_model"
	is_boolean(input.output.contains_pii)
}

valid_for_hook if {
	input.hook == "pre_tool"
	is_string(input.tool)
	input.approval.state in {"none", "pending", "approved", "rejected"}
}

valid_for_hook if {
	input.hook == "post_tool"
	is_number(input.output.bytes)
}

block_reasons contains "payment_out_of_scope" if input.action_class == "payment"

block_reasons contains "tool_not_granted" if {
	input.hook == "pre_tool"
	not input.tool in pack.tools
}

block_reasons contains "approval_rejected" if {
	input.hook == "pre_tool"
	input.action_class == "external_comm"
	input.approval.state == "rejected"
}

block_reasons contains "budget_hard_limit" if {
	input.hook == "pre_model"
	input.spend.run_usd + input.spend.est_call_usd > input.spend.run_limit_usd
}

block_reasons contains "residency" if {
	input.hook == "pre_model"
	input.data_class == "restricted"
	not input.model_region in pack.restricted_regions
}

block_reasons contains "model_not_allowed" if {
	input.hook == "pre_model"
	input.data_class in {"confidential", "restricted"}
	not input.model in pack.models
}

approval_reasons contains "external_comm_needs_approval" if {
	input.hook == "pre_tool"
	input.action_class == "external_comm"
	input.approval.state in {"none", "pending"}
}

modify_reasons contains "redact_pii" if {
	input.hook == "post_model"
	input.output.contains_pii
}

modify_reasons contains "truncate_result" if {
	input.hook == "post_tool"
	input.output.bytes > data.limits.max_tool_result_bytes
}
