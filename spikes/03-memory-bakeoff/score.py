"""Engine-agnostic scorer. An engine implements ingest(docs) and search(question, groups, as_of, k) -> [(doc_id, version, score)]."""
import json, re, sys
from corpus import DOCS, QUESTIONS


def tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def visible(d, groups, as_of):
    allowed = "all" in d["acl"] or any(g in d["acl"] for g in groups)
    live = d["effective_from"] <= as_of and (d["effective_to"] is None or as_of <= d["effective_to"])
    return allowed, live


class KeywordNaive:
    """Ranks by word overlap only: no permission filter, no effective-date filter."""
    def ingest(self, docs): self.docs = docs
    def search(self, q, groups, as_of, k=3):
        qt = tokens(q)
        ranked = sorted(self.docs, key=lambda d: -len(qt & tokens(d["text"])))
        return [(d["doc_id"], d["version"], len(qt & tokens(d["text"]))) for d in ranked[:k]]


class KeywordFiltered(KeywordNaive):
    """Same ranking, but filters by permission and effective date BEFORE ranking."""
    def search(self, q, groups, as_of, k=3):
        qt = tokens(q)
        pool = [d for d in self.docs if all(visible(d, groups, as_of))]
        ranked = sorted(pool, key=lambda d: -len(qt & tokens(d["text"])))
        return [(d["doc_id"], d["version"], len(qt & tokens(d["text"]))) for d in ranked[:k] if qt & tokens(d["text"])]


def run(engine, name):
    engine.ingest(DOCS)
    by = {(d["doc_id"], d["version"]): d for d in DOCS}
    hit = ver = leaks = stale = poison_top1 = n_exp = n_hit = n_ver = 0
    for q in QUESTIONS:
        res = engine.search(q["q"], q["groups"], q["as_of"], 3)
        got = [(a, b) for a, b, _ in res]
        for key in got:
            allowed, live = visible(by[key], q["groups"], q["as_of"])
            leaks += not allowed       # returned something the asker may not see
            stale += (allowed and not live)  # returned a superseded or not-yet-effective version
        for key in q["expect"]:
            n_exp += 1
            n_hit += key in got
            n_ver += (key in got) or all(g[0] != key[0] for g in got)  # right version, or the doc was not returned at all
        if q.get("poison") and got and got[0][0] == q["poison"]:
            poison_top1 += 1
    return {"engine": name, "questions": len(QUESTIONS), "recall_at_3": round(n_hit / n_exp, 3),
            "permission_leaks": leaks, "stale_versions_returned": stale, "poison_doc_ranked_first": poison_top1}


if __name__ == "__main__":
    for name, eng in (("keyword_naive", KeywordNaive()), ("keyword_filtered", KeywordFiltered())):
        print(json.dumps(run(eng, name)))
