---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bfcl.md
source_url: https://proceedings.mlr.press/v267/patil25a.html
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision — categories and numbers aligned with the ICML 2025 paper)"
---

# Excerpt: BFCL — confusion cells for a structured output space

**Source library:** `wiki/raw-data/llm-training/papers/bfcl.md`
**Artifact:** The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of
Large Language Models. Patil, Mao, Yan, Ji, Suresh, Stoica, et al. (UC Berkeley). ICML 2025, PMLR
267:48371–48392.
**Checked on 2026-09-15** against the verified library card, which was itself checked against the PMLR PDF
on 2026-09-14.

> **Corrections carried by this excerpt.** The earlier version listed "7 core categories: simple, parallel,
> multiple, parallel-multiple, relevance-detection, chat, Java/JS-specific". The paper's single-turn
> categories are Simple, Multiple, Parallel, Parallel Multiple, Irrelevance, and Relevance (§3.1,
> App. C.2); Java and JavaScript are languages, not categories, and there is no "chat" category. The
> earlier version also stated that "frontier models still hallucinate ~10% on irrelevant queries"; no such
> figure appears in the paper, and the per-model Irrelevance scores in Table 1 vary widely.

---

## Why ch-50 uses this source

Function calling has an enumerable output space, so its failures resolve into mutually exclusive cells
rather than free-form reasons. ch-50 §6 uses it as the reference case for confusion-cell bucketing, and
§7 uses its multi-turn error analysis as an example of LLM-judged trajectory labelling.

---

## The categories that define the cells (§3.1, App. C.2)

| Category | Definition (`F` = candidate functions given to the model) |
|---|---|
| Simple | `\|F\| = 1`, one call expected |
| Multiple | `\|F\| > 1`, one call expected |
| Parallel | `\|F\| = 1`, several calls expected |
| Parallel Multiple | `\|F\| > 1`, several calls expected |
| Irrelevance | `\|F\| ≥ 1`, **no** call expected |
| Relevance | at least one call expected |

Irrelevance entries are built by removing parameter information from the query or removing a needed
function; the expected output is a clarification or an error, and any call is counted as a hallucination
(App. B).

## The matcher, which is what makes the cells exclusive (§4.1, App. H)

Calls are parsed with Python's `ast` module. The function name must match exactly and each parameter value
must be in a set of accepted answers. Type handling differs by language: Python accepts an int where a
float is expected; Java and JavaScript require a float literal; a float for an int parameter is invalid in
all three. Lists are order-sensitive (all acceptable permutations are enumerated), strings are
case-insensitive with whitespace and listed punctuation removed, dictionary key order is ignored, and
parallel calls are matched all-or-nothing without positional alignment.

The consequence for bucketing: the matcher's own false-accept behaviour (an int where a float is expected
in Python) is a property of the grader, not of the model, and belongs in the ledger next to the counts.

## Numbers ch-50 uses

- **Abstention and calling move independently.** Qwen2.5-72B-Instruct in prompting mode scores 100.0 on
  Relevance and 72.8 on Irrelevance (Table 1).
- **Stateful tasks lag single-turn tasks.** gpt-4o-2024-11-20 (Prompt) scores 95.5 / 94.0 on single-turn
  AST Multiple / Parallel, 59.0 on multi-turn base, and 6.0 on memory; the highest memory score in the
  table is 12.0 (Table 1, §5.6).
- **Format against content.** Prompting-mode models average 412.93 decoding issues against 182.5 for
  function-calling-mode models out of 4,251 entries; among responses that do decode, function-calling-mode
  models give the wrong number of calls more often in the Multiple category (77.5 against 21) (§5.1).
- **LLM-judged multi-turn root causes.** With GPT-4o-08-06 as judge (App. F), the most frequent root cause
  is "Failed to Understand Environment State", followed by "Failed to Understand User's Request"
  (§5.4.2, Fig. 5).
- **Overall level.** Across the 71 rows of Table 1 the best overall accuracy is 66.4
  (gpt-4o-2024-11-20, prompting mode).

## Used by

ch-50 §6 (confusion cells, the bucket table, grader false-accepts), §7 (LLM-judged trajectory labels),
Negative samples (verifier-produced labels), Common mistakes.
