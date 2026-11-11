import json
import pytest

from harness.artifacts import (ArtifactNotFound, IntegrityError, LocalArtifactStore, Offloader, make_read_artifact_tool, preview, render_result)
from harness.events import SQLiteEventLog


@pytest.fixture
def store(tmp_path):
    return LocalArtifactStore(tmp_path / "artifacts")


def test_content_addressed_idempotent_and_verified(store):
    a = store.put("org-1", b"hello\nworld\n")
    b = store.put("org-1", b"hello\nworld\n")
    assert a == b and a.lines == 2 and store.get("org-1", a.sha256) == b"hello\nworld\n"


def test_another_org_cannot_read_an_artifact(store):
    ref = store.put("org-1", b"secret terms")
    with pytest.raises(ArtifactNotFound):
        store.get("org-2", ref.sha256)               # same hash, different tenant: not found, not forbidden (no existence leak)


def test_tampering_is_detected_on_read(store, tmp_path):
    ref = store.put("org-1", b"original")
    path = store._path("org-1", ref.sha256)
    path.write_bytes(b"changed")
    with pytest.raises(IntegrityError):
        store.get("org-1", ref.sha256)


@pytest.mark.parametrize("bad", ["../../etc/passwd", "zz", "A" * 64, "", "0" * 63])
def test_malformed_ids_cannot_traverse_paths(store, bad):
    with pytest.raises(ArtifactNotFound):
        store.get("org-1", bad)


def test_read_range_pages_through_and_caps_the_page_size(store):
    ref = store.put("org-1", "\n".join(f"line {i}" for i in range(1200)).encode())
    first = store.read_range("org-1", ref.sha256, 0, 100)
    assert first["lines"][0] == "line 0" and first["next_offset"] == 100 and first["total_lines"] == 1200
    last = store.read_range("org-1", ref.sha256, 1100, 10_000)       # page size is capped, offset past the end is safe
    assert len(last["lines"]) == 100 and last["next_offset"] is None
    assert store.read_range("org-1", ref.sha256, 5000, 10)["lines"] == []


def test_preview_keeps_head_and_tail_and_says_what_is_missing():
    text = "\n".join(f"row {i}" for i in range(100))
    p = preview(text)
    assert "row 0" in p and "row 4" in p and "row 95" in p and "row 99" in p and "row 50" not in p
    assert "90 lines not shown" in p
    assert preview("a\nb\nc") == "     1  a\n     2  b\n     3  c"                # short text is shown whole


def test_preview_caps_a_very_long_line():
    p = preview("x" * 5000)
    assert len(p) < 1200 and "4000 more characters" in p


def test_small_results_pass_through_and_large_ones_become_stubs(store):
    log = SQLiteEventLog()
    off = Offloader(store, log, "org-1", threshold_bytes=2000)
    small = {"ok": True, "output": {"x": 1}}
    assert off.maybe_offload("r1", "s1", small) is small
    big = {"ok": True, "output": {"rows": [f"invoice line {i}" for i in range(400)]}}
    stub = off.maybe_offload("r1", "s1", big)
    assert stub["offloaded"] and stub["artifact"]["bytes"] > 2000 and "lines not shown" in stub["preview"]
    assert "invoice line 200" not in json.dumps(stub)                                   # the middle is not in what the model sees
    ev = [e for e in log.read("org-1", "r1") if e.type == "artifact.created"]
    assert len(ev) == 1 and ev[0].payload["sha256"] == stub["artifact"]["sha256"]
    assert json.loads(store.get("org-1", stub["artifact"]["sha256"]))["output"]["rows"][200] == "invoice line 200"   # nothing was lost


async def test_read_artifact_tool_returns_the_hidden_part_and_is_tenant_scoped(store):
    log = SQLiteEventLog()
    stub = Offloader(store, log, "org-1", 500).maybe_offload("r1", "s1", {"ok": True, "output": [f"item {i}" for i in range(300)]})
    tool = make_read_artifact_tool(store, "org-1")
    page = await tool.fn({"sha256": stub["artifact"]["sha256"], "offset": 150, "limit": 50})
    assert page["offset"] == 150 and len(page["lines"]) == 50
    other = await make_read_artifact_tool(store, "org-2").fn({"sha256": stub["artifact"]["sha256"]})
    assert other == {"error": "ArtifactNotFound"}


def test_render_result_for_prompts(store):
    assert render_result({"ok": True}) == '{"ok": true}'
    stub = Offloader(store, SQLiteEventLog(), "org-1", 100).maybe_offload("r", "s", {"ok": True, "output": ["z" * 50] * 20})
    text = render_result(stub)
    assert stub["artifact"]["sha256"] in text and "read_artifact" in text
