# Spike 3: memory bake-off, Graphiti vs Mem0 (prepared; NOT run)

**Status: harness ready, engines not run.** Both engines call a language model for extraction and an embedder for retrieval, and Graphiti also needs a graph database. The build sandbox has no model key and cannot reach model-hosting sites, so neither engine was run and there are no engine results. Nothing here should be read as evidence about Graphiti or Mem0.

## What exists
- `corpus.py`: a **synthetic** 8-document policy corpus (versioned and effective-dated, with access groups, a confidential document, a distractor, and a prompt-injection note) and 12 questions with expected answers. Fictional content: the real corpus is drafted by the org.
- `score.py`: an engine-agnostic scorer. An engine implements `ingest(docs)` and `search(question, groups, as_of, k)`.
  Metrics: recall at 3, permission leaks (a result the asker may not see), stale versions returned, and whether the injected document ranks first.
- Two keyword baselines that validate the scorer:

| Baseline | Recall@3 | Permission leaks | Stale versions | Injected doc first |
| --- | --- | --- | --- | --- |
| Naive (no filters) | 1.0 | 12 | 7 | 1 |
| Filter before ranking | 1.0 | 0 | 0 | 1 |

Reading: the corpus is small, so recall is easy; the value is in the other three columns. A real engine should match the filtered baseline on leaks and stale versions. The injected document ranks first for the bait question even with correct filtering, so resisting it is the context service's job (mark results untrusted, never act on them as instructions), not retrieval's.

## To run (needs a model endpoint, an embedder, and for Graphiti a graph database)
1. Install `graphiti-core` and `mem0ai` in the venv, with a self-hostable model and embedder per the tools list (no hosted dependency).
2. Write one adapter per engine implementing the interface above. Ingest each document with its version, dates and access groups as metadata; enforce the permission and as-of filters **before** ranking, either in the engine's query filters or by group partition. If an engine cannot, record that: it decides whether our context service must wrap it.
3. Run `score.py` for each engine; also record: extraction cost in model calls per document, ingest and query latency, behaviour with a small local model (Graphiti needs reliable structured output), and what happens when a document is revoked.
4. Cross-check retrieval on the public tau2-bench `banking_knowledge` domain (700+ documents), noted in spike 5.

## Exit criterion (from the tools list)
Retrieval accuracy, permission-filter correctness, write review, and operation with a self-hosted model and embedder. Pick one engine, or decide to wrap both behind our interface.
