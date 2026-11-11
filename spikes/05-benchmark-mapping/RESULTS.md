# Spike 5: benchmark mapping (partial: mapping done from primary sources, no end-to-end run)

Question: which of the four finance task types qualify under the v1 gate (a capability-level match to a public benchmark, with the pack's verifier as the production gate), and does one suite run end to end?

**Result: mapping done; the end-to-end run is not.** Running a suite needs a model endpoint and, for TheAgentCompany, its Docker task images and services. This sandbox has no model key, and I did not check whether it could host the task services.

## What was read

Primary sources opened (GitHub pages, directory listings, READMEs): TheAgentCompany task directory, the CUAD repository, the tau2-bench domains directory. Task names were read; **task contents were not**, so fit below is from names and READMEs and needs a read of each task before it counts as qualified.

## Mapping

| Task type | Closest public tasks | Fit | Notes |
| --- | --- | --- | --- |
| Contract payments (log and validate only) | TheAgentCompany `finance-invoice-matching`, `finance-expense-validation` | Moderate | Finance validation tasks exist. Linking an invoice to contract terms is not visible from the names. |
| AR and AP logging and optimization | TheAgentCompany `finance-invoice-matching`, `finance-revenue-reconciliation`, `finance-budget-variance`, `admin-check-employees-budget-and-reply-and-record`, `admin-collect-requests-and-compute-total-price` | Moderate to good for logging and reconciliation | "Optimization" (for example payment timing) has no task. |
| Payout terms review | CUAD (expert-annotated contract clause dataset, `data.zip` in the repo; I recall 41 clause categories from the CUAD paper, but the page I read did not confirm the count) and ContractEval built on it | Partial | Clause extraction and review, not agentic. The category list is in `category_descriptions.csv`; check it for payment and renewal clauses. |
| Vendor renewal negotiation | TheAgentCompany `admin-get-best-vendor-quote` (quote comparison); negotiation benchmarks TERMS-Bench, AgenticPay, AgoraBench, EconEvals (search results only, not opened) | Weak to partial | No opened source covers multi-turn renewal negotiation against a vendor. |

TheAgentCompany finance and finance-adjacent tasks found: `finance-apply-tax-credit`, `finance-budget-variance`, `finance-check-attendance-payroll`, `finance-create-10k-income-report`, `finance-expense-validation`, `finance-find-signatories`, `finance-invoice-matching`, `finance-nonqualified-bill-ask-for-reimburse`, `finance-qualified-bill-ask-for-reimburse`, `finance-r-d-activities`, `finance-revenue-reconciliation`, `finance-substantial-presence-test`, plus the admin and HR tasks above. It is MIT licensed, graded by deterministic and LLM checkers, and runs each task as a Docker image.

## Other finding

tau2-bench has a `banking_knowledge` domain: knowledge-retrieval banking customer service over a 700+ document knowledge base, with configurable retrieval pipelines (embeddings, BM25, grep, rerank, sandboxed shell). It is not one of the four task types, but it is a direct public test for the **policy retrieval** layer (context service and memory engine), so it is a useful cross-check for spike 3. Its task count and grading were not on the pages read.

## Gate outcome (preliminary, by your capability-level rule)

- AR and AP logging: qualifies for evaluation.
- Contract payments (validate only): qualifies for evaluation, with the weakest contract linkage.
- Payout terms review: qualifies at capability level through CUAD; the production verifier does the real work.
- Vendor renewal negotiation: does not clearly qualify. Recommend draft-only, human-approved from day one (already decided) and treating it as the last pack to build.

## To finish this spike

1. Read the content of `finance-invoice-matching`, `finance-revenue-reconciliation`, `finance-expense-validation` and `admin-get-best-vendor-quote`.
2. Run one task end to end on your infrastructure: needs a model endpoint and the TheAgentCompany task images and services (where they are hosted and how heavy they are was not checked here).
3. Read `category_descriptions.csv` from CUAD.
