---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/agent-flan.md
source_url: https://arxiv.org/abs/2403.12881
created_at: "2026-04-23"
---

# Excerpt: Agent-FLAN — the four hallucination classes

**Source library:** `wiki/raw-data/llm-training/papers/agent-flan.md`
**Paper:** Chen et al. 2024 (Shanghai AI Lab), "Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for LLMs."

---

## Why this source anchors ch-27 §1.3

Ch-27 §1.3's claim — "agent SFT without negatives is an open-loop controller" — is exactly Agent-FLAN's diagnosis. The paper identifies the problem, names its four modes, and provides the closed-loop correction.

## The three bottlenecks Agent-FLAN identifies

From the source (line 15):

> Agent-FLAN studies what makes agent fine-tuning work and identifies three bottlenecks in existing approaches (AgentTuning, FireAct): (1) data format mismatch with pretraining distribution, (2) capability imbalance — too much tool-use, too little instruction-following preservation, and (3) hallucination — models fabricate tool calls in irrelevant contexts.

All three matter; ch-27 focuses on (3). But (1) is load-bearing — the format-alignment fix (rewrite trajectories to avoid special tokens outside Llama-2 pretraining distribution) is the reason Agent-FLAN-7B preserves MT-Bench within 0.5 points while competitors drop by 2-5.

## The four hallucination modes

From the source (lines 31-34):

> - **Format hallucination:** examples where base model produces malformed tool calls → gold response is the corrected call or a refusal.
> - **Action hallucination:** user query doesn't require a tool → gold is text-only answer, no call.
> - **Parameter hallucination:** tool is needed but with different args than the model would fabricate → gold has the correct args.
> - **Relevance hallucination:** offered tool list doesn't include a relevant tool → gold is "I cannot help with this tool set" refusal.

Each class has a distinct trigger and a distinct gold response. This taxonomy is what ch-27 §1.3 reproduces in table form. Two notes on reading it:

**Format vs Parameter are adjacent but distinct.** Format hallucination is JSON malformation (wrong key names, unclosed brackets); Parameter hallucination is well-formed JSON with wrong values. The fix for Format is schema validation; the fix for Parameter is better grounding in the tool's docstring.

**Relevance hallucination is where refusal skills get installed.** Without relevance-class negatives, the SFT-trained agent cannot say "I don't have a tool for this." It will always confabulate a call.

## The three capability types (orthogonal to negatives)

From the source (lines 25-28):

> - **Instruction-following data (preserve chat quality):** ShareGPT, Alpaca, Evol-Instruct — ~50K.
> - **Agent reasoning / decision data:** trajectories from AgentTuning + FireAct (filtered) + new internal rollouts — ~20K.
> - **Generalization data:** tool-use trajectories on held-out APIs not in the core set — ~10K.

Three positive data types + four negative classes. Seven pools total in the Agent-FLAN mix. The ablation result (source line 55) is that removing any one positive type costs 0.3-0.5 AgentBench points; removing negatives triples the hallucination rate. **Negatives are a bigger effect than any single positive pool.**

## The hallucination-rate measurement

From the source (line 50):

> - **Hallucination-rate measurement:** Agent-FLAN-7B cuts hallucinated tool calls by 5× vs AgentTuning baseline on AgentBench held-out.

5× reduction at the 7B scale. This is what "closed-loop" means operationally — the negative-example pool teaches the model *when not to call tools*, not just how to call them correctly.

## Format alignment — the unsung contribution

From the source (lines 36-38):

> ### Format-alignment step
> - Rewrite agent trajectories to avoid special tokens / delimiters that don't appear in Llama-2 pretraining. Keeps training distribution close to pretraining distribution, reducing catastrophic forgetting.

If you take one engineering detail from Agent-FLAN, take this. Custom `<tool>...</tool>` tags look clean but they aren't in Llama-2 pretraining, so they push the fine-tuned distribution away from the base. Use markdown-code-fenced JSON instead. Same semantic content, much smaller distribution shift.

## What to take from Agent-FLAN for ch-27

1. **Four hallucination modes.** Format, Action, Parameter, Relevance. Memorize the table.
2. **Add explicit negative pools for each mode.** Don't assume the positive-only data will generalize to "don't call tools when unnecessary" — it won't.
3. **Format-align to the pretraining distribution.** If your special tokens aren't in pretraining, rewrite to avoid them.
4. **Capability types are orthogonal to negatives.** You need instruction-following + agent + generalization *and* the four negative classes. Not either/or.

## Connections

- [[ch-27]] §1.3 — the negative-example ontology is the core of §1.3.
- [[excerpts/agentinstruct]] — the complementary positive-data-scale story; neither paper alone is sufficient.
- [[excerpts/openhands-data]] — OpenHands trajectory format uses markdown-fenced JSON tool_calls, following Agent-FLAN's format-alignment principle.
