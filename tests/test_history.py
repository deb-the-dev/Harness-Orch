import copy

from harness.history import UNKNOWN_RESULT, mark_provider, prepare_messages
from harness.router import ModelSpec
from harness.router.models import ModelProfile

OPENAI = ModelSpec("m-o", "openai/x", "eu", 1, 1, provider="openai")
CLAUDE = ModelSpec("m-c", "anthropic/x", "eu", 1, 1, provider="anthropic", profile=ModelProfile(reasoning_blocks="keep"))
THINKING = {"type": "thinking", "thinking": "chain of thought", "signature": "sig"}


def assistant_with_thinking(provider="anthropic"):
    return {"role": "assistant", "content": [THINKING, {"type": "text", "text": "answer"}], "x_provider": provider, "x_model": "m-c"}


def test_foreign_reasoning_blocks_are_removed_on_a_model_switch():
    history = [{"role": "user", "content": "q"}, assistant_with_thinking("anthropic")]
    out, rep = prepare_messages(history, OPENAI)
    assert out[1]["content"] == [{"type": "text", "text": "answer"}] and rep.stripped_blocks == 1
    assert all("x_provider" not in m and "x_model" not in m for m in out)               # our annotations never reach a provider


def test_a_model_keeps_its_own_reasoning_blocks_only_when_its_profile_says_so():
    history = [assistant_with_thinking("anthropic")]
    kept, rep = prepare_messages(history, CLAUDE)
    assert THINKING in kept[0]["content"] and rep.stripped_blocks == 0
    stripped, rep = prepare_messages(history, ModelSpec("m-c2", "anthropic/y", "eu", 1, 1, provider="anthropic"))   # profile default: strip
    assert THINKING not in stripped[0]["content"] and rep.stripped_blocks == 1
    unmarked, _ = prepare_messages([{"role": "assistant", "content": [THINKING, {"type": "text", "text": "a"}]}], CLAUDE)   # unknown origin: strip
    assert THINKING not in unmarked[0]["content"]


def test_a_message_with_nothing_but_reasoning_is_dropped():
    out, rep = prepare_messages([{"role": "user", "content": "q"}, {"role": "assistant", "content": [THINKING], "x_provider": "anthropic"}], OPENAI)
    assert [m["role"] for m in out] == ["user"] and rep.stripped_blocks == 1


def test_a_tool_call_without_a_result_gets_an_explicit_unknown_outcome():
    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "send", "arguments": "{}"}}]}]
    out, rep = prepare_messages(history, OPENAI)
    assert out[-1] == {"role": "tool", "tool_call_id": "c1", "content": UNKNOWN_RESULT} and rep.patched_calls == 1


def test_a_matched_call_is_left_alone_and_an_orphan_result_is_dropped():
    history = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
               {"role": "tool", "tool_call_id": "c1", "content": "ok"}, {"role": "tool", "tool_call_id": "ghost", "content": "stray"}]
    out, rep = prepare_messages(history, OPENAI)
    assert [m.get("tool_call_id") for m in out if m["role"] == "tool"] == ["c1"] and rep.patched_calls == 0 and rep.dropped_orphans == 1


def test_prompt_prefix_goes_on_the_system_message_or_creates_one():
    spec = ModelSpec("m-p", "openai/x", "eu", 1, 1, profile=ModelProfile(prompt_prefix="Answer in JSON only."))
    out, rep = prepare_messages([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "q"}], spec)
    assert out[0]["content"] == "Answer in JSON only.\n\nBe brief." and rep.prefix_applied
    out, _ = prepare_messages([{"role": "user", "content": "q"}], spec)
    assert out[0] == {"role": "system", "content": "Answer in JSON only."}


def test_input_is_never_mutated_and_the_report_hash_is_stable():
    history = [{"role": "user", "content": "q"}, assistant_with_thinking()]
    before = copy.deepcopy(history)
    a, ra = prepare_messages(history, OPENAI)
    b, rb = prepare_messages(history, OPENAI)
    assert history == before and a == b and ra.sent_hash == rb.sent_hash and len(ra.sent_hash) == 16


def test_mark_provider_annotates_without_changing_the_original():
    m = {"role": "assistant", "content": "x"}
    assert mark_provider(m, OPENAI) == {"role": "assistant", "content": "x", "x_provider": "openai", "x_model": "m-o"} and "x_provider" not in m
