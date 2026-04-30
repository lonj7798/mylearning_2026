---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/toolformer.md
source_url: https://arxiv.org/abs/2302.04761
created_at: "2026-04-23"
---

# Excerpt: Toolformer — the self-supervised annotation bootstrap

**Source library:** `wiki/raw-data/llm-training/papers/toolformer.md`
**Paper:** Schick, Dwivedi-Yu, Dessi, Raileanu, Lomeli, Zettlemoyer, Cancedda, Scialom 2023, "Toolformer: Language Models Can Teach Themselves to Use Tools" (Meta AI).

---

## Why this source anchors ch-26

Toolformer is the chapter's zero-point. Everything that follows — ToolLLM's trajectory synthesis, APIGen's three-layer verifier, APIGen-MT's blueprint-then-rollout — takes for granted that you *can* manufacture tool-use supervision from unlabeled text. Toolformer is the paper that proves that premise, with the minimum possible machinery: one base LM, five text-in/text-out APIs, and a filter that tests whether the returned tool result lowers loss on upcoming tokens.

Ch-26 §1 derives Toolformer's perplexity-delta filter and explains why `min(L_no_call, L_call_empty)` — not just `L_no_call` — is the correct baseline against which to measure the tool's contribution. This excerpt expands the derivation and pulls out the operational thresholds (`5%` API-start probability, `τ = 1.0` nat) that set the paper's rejection budget.

---

## The three-stage pipeline in operational detail

From the source (lines 30–45):

> **Base setup:** GPT-J 6.7B, a subset of CCNet, and five text-form APIs: question answering, Wikipedia search, calculator, machine translation, and calendar.
>
> **Generation pipeline:**
> 1. **Prompt the base LM with a few demonstrations per API** so it can insert tool calls inline in text.
> 2. **Sample candidate insertion positions** where the model assigns enough probability to starting an API call. The appendix states a default threshold of at least **5%** probability for the API-start token.
> 3. **Sample candidate API calls** at those positions, then execute them to obtain real tool outputs.
> 4. **Filter by downstream loss reduction** instead of accepting every syntactically valid call.
> 5. **Insert only accepted calls back into the raw text corpus** and fine-tune with a standard LM objective.

Two things about this pipeline that recur throughout the chapter:

- **The base LM is also the verifier.** There is no stronger teacher. The filter measures whether the tool's output helps the *same model* predict future tokens. This makes the pipeline self-supervised — the defining property that later pipelines (ToolLLM, APIGen) give up in exchange for stronger signal.
- **Execution is load-bearing even in 2023.** Step 3 explicitly runs the five APIs and stores real outputs. "Format-only synthesis is too weak" is not a 2024 discovery; it is the precondition Toolformer already operates under.

---

## The filter as an "is the result useful?" test

The paper's Figure 2 describes the filtering rule narratively; the text is worth a careful read (source lines 39–43):

> **Filtering rule:** Toolformer compares future-token weighted cross-entropy with and without the call result. A call is kept only if seeing the real returned result lowers loss by at least the filtering threshold; the appendix gives a default threshold of **1.0**. The comparison is against both:
> - no API call at all
> - the API call text without the returned result
>
> That second comparison matters: it stops the model from getting credit merely for seeing a tool-name pattern such as `Calculator(...)`. The result itself must add predictive value.

The "second comparison matters" sentence is the one to internalise. Without the `L_call_empty` baseline, the filter rewards any annotation that *mentions* a tool, even if the returned result is wrong. The implicit threat is overfitting to the *shape* of tool calls — the model learns to emit tool-call syntax as a performative gesture without using the output. The double comparison makes the test specifically "is the returned content itself useful?"

**Numerical implication.** The default τ = 1.0 nat on a ~5-token continuation window is aggressive: it accepts only calls where the tool result cuts perplexity by roughly a factor of `e` on the next few tokens. Practically, that is a narrow bar — most calls produce results whose marginal benefit on loose continuation is smaller. Table 2 reports the accepted-example counts by tool after filtering, and the totals are in the low thousands per tool, against candidate pools of hundreds of thousands. **Rejection is the norm, not the exception.**

---

## What Toolformer deliberately does not do

The paper's limits set the agenda for the rest of the chapter. Toolformer:

1. **Does not plan across calls.** The five APIs are single-step; there is no reasoning trace, no dependency between calls, no sequence. ToolLLM's DFS-DT ([[toolllm]]) is the direct answer to this gap.
2. **Does not retrieve.** Five tools is small enough that the model sees all of them in the prompt every time. Gorilla ([[gorilla]]) introduces retrieval-aware training for the opposite regime.
3. **Does not verify with a stronger model.** The filter uses the base LM's own loss. APIGen's three-layer check ([[apigen]]) replaces this with format + execution + GPT-4 judge, giving up self-supervision for sharper signal.
4. **Does not synthesise multi-turn.** There is no user, no conversation, no state. APIGen-MT's blueprint-then-rollout ([[apigen-mt]]) is the multi-turn successor.

Seen this way, Toolformer is not a weaker version of APIGen — it is the *principled minimum* that establishes the vocabulary the chapter then extends. The perplexity-delta filter is replaced by stronger verifiers, but the *shape* of the pipeline (propose → execute → filter → fine-tune) is Toolformer's.

---

## Carryover rules for modern pipelines

From source lines 49–54, the practical lessons the paper frames explicitly:

- Execute candidate calls during data generation; format-only synthesis is too weak.
- Use a usefulness filter tied to model benefit, not just schema correctness.
- Short, local, high-information tool outputs are easiest to learn from with self-supervision.
- This method is best viewed as a precursor for single-call or local tool insertion, not as a full agent-data recipe for planning, recovery, or multi-turn orchestration.

The first two are the rules every subsequent pipeline obeys. The third — "short, local, high-information outputs" — is a useful framing for why calculator and QA are easier targets than multi-step booking workflows; the loss-reduction signal needs the tool's output to sit near the continuation it's supposed to help predict. Long, diffuse tool outputs (a 2-page web scrape) dilute the signal beyond the τ = 1.0 threshold.

---

## Connections

- Conceptual precursor: [[self-instruct]] — both use the model to generate its own supervision from seeds.
- Direct successor: [[toolllm]] — takes Toolformer's "annotate with a filter" idea and scales it to full trajectories over 16K real APIs.
- Modern replacement of the self-verification step: [[apigen]]'s three-layer check.
- Evaluation descendant: [[bfcl]] — the community's answer to "how do you measure a tool-calling model once you have one?"
