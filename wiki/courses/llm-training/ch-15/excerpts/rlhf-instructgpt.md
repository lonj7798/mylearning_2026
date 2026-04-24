---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlhf-instructgpt.md
source_url: https://arxiv.org/abs/2203.02155
created_at: "2026-04-23"
---

# Excerpt: InstructGPT — the original human-annotation recipe

**Source library:** `wiki/raw-data/llm-training/papers/rlhf-instructgpt.md`
**Year / authors:** 2022 / Ouyang, Wu, Jiang, Christiano, Leike, Lowe et al. (OpenAI).

---

## Why this source anchors ch-15

InstructGPT is not a rubric paper. It is a training paper. But every later annotation operation — [[hh-rlhf]]'s two-campaign collection, [[ultrafeedback-construction]]'s 17-model fleet, [[tulu-3-sft-mix]]'s 939K skill-targeted mix, [[prm800k]]'s step-level labels — descends from choices InstructGPT made about *how a human being contributes a signal*. The prompt taxonomy, the K=4-9 ranking protocol, the labeler guidelines, and the SFT-RM-PPO three-stage template are all in this paper. Ch-15 §3 (adjudication workflow), §4 (preference sampling), and §6 (operational reality) are all expansions of operational choices this paper froze.

---

## The three-stage data flow, annotation-side only

The body of the paper gives the training-side story. Strip it to just the annotation surface and you get:

```
# rlhf-instructgpt.md, derived from §3 and Figure 2
Stage 1 — SFT demonstrations
  - 13K prompts sampled from the OpenAI API + labeler-generated seeds.
  - Labelers write the *desired output* directly. No model-in-the-loop.
  - Rubric: "helpful, honest, harmless"; detailed labeler guidelines.
  - One label per item.

Stage 2 — Preference rankings
  - 33K prompts; for each, the SFT model samples K ∈ {4..9} completions.
  - A labeler ranks the K from best to worst (full order, not pairwise).
  - That generates C(K,2) pairs per prompt: K=4 → 6 pairs, K=9 → 36.
  - Same rubric; inter-labeler consistency monitored via held-out overlap.

Stage 3 — (no human annotation; PPO consumes stage-2 RM)
```

Notice: **Stage 2's K=4-9 ranking protocol is the highest-bandwidth way to extract preference signal per unit of annotator attention.** One annotator-hour that ranks 30 prompts at K=5 produces 300 pairwise preference judgements. The same hour of pairwise annotation produces ~30-60 pairs depending on response length. The K-way ranking wins 5–10× on throughput because the cognitive cost of reading the prompt is amortized across all K responses. This is the reason every follow-up project — [[ultrafeedback-construction]]'s "4 responses per prompt," [[tulu-3-sft-mix]]'s skill submixes — also picks K between 4 and 9.

The cost of the K-way ranking is that the pairs within a prompt are *not independent*. A pair `(rank 1 vs rank 2)` is a close call; `(rank 1 vs rank 5)` is not. The paper handles this explicitly:

> Train all pairs from the same prompt in the same minibatch (otherwise overfits quickly).

This is the 2022 discovery that ch-15 §4's close-pair-mining section is a 2024-era refinement of: the information in K-way ranking is uneven across pairs, and the training pipeline must respect that.

---

## The labeler guidelines are the rubric

Section 3.4 of the paper ("Data collection") is short but load-bearing:

> We collaborated closely with labelers over the course of the project. We had an onboarding process to train labelers on the project, wrote detailed instructions for each task, and answered labeler questions in a shared chat room.

Read that paragraph with ch-15 §1 and §2 in mind. "Detailed instructions" is the rubric document. "Onboarding process" is the calibration session that establishes κ ≥ 0.6. "Shared chat room" is the edge-case-resolution venue that feeds new exemplars into the rubric. "Over the course of the project" is the rubric-drift problem — the rubric is not static, and the labelers who grew into version N-1 need to be re-oriented to version N.

The paper reports labeler-labeler agreement of **72.6% ± 1.5%** on a held-out comparison set. That is κ ≈ 0.45 under balanced marginals — Landis-Koch "moderate." The InstructGPT result is that a 72.6% inter-rater agreement was sufficient to lift a 1.3B model above 175B GPT-3 on instruction-following preference, which is the empirical data point that says **you do not need κ > 0.8 to do useful RLHF — but you do need to measure and report κ so that downstream consumers know the noise floor**.

---

## The PPO-ptx aside is about an annotation-era alignment tax

Equation 2, the PPO-ptx objective, includes a `γ · L_ptx` pretraining-mix term. The paper's justification is that PPO against a preference RM *regresses on public NLP benchmarks* unless you mix pretraining gradient back in. Read through the ch-15 lens: this is a consequence of the annotation distribution. Labelers ranked completions on a skewed prompt distribution — customer queries from the OpenAI API, heavily chat-shaped. Optimizing against that narrow distribution causes the model to forget the broad pretraining distribution. The fix is not to fix the rubric; it is to explicitly counter the narrowness with a pretraining blend.

The modern successor is broader annotation distributions ([[tulu-3-sft-mix]] explicitly balances chat / math / code / safety / multilingual) which eliminates the need for γ because the annotation distribution is already close to the pretraining distribution in capability coverage. The 2024–2025 moves from PPO-ptx → RLVR → DPO+RLVR are all responses to the same underlying observation: get the annotation distribution right, and the alignment tax shrinks.

---

## The numbers worth carrying

| Quantity | Value | Why ch-15 cares |
|---|---|---|
| SFT prompts | 13K | Minimum viable SFT set for a small lab; 50-100× below [[tulu-3-sft-mix]] |
| RM prompts | 33K | K=4-9 ranking → ~200K pairs; the canonical annotation batch size |
| RM size | 6B | Small — the RM is absorbing annotation noise, not approximating truth |
| β (KL coeff) | 0.02 | How much the RM signal matters vs staying close to SFT |
| Labeler agreement | 72.6% | The 2022 baseline; every later project reports against it |
| Clip ε, PPO | 0.2 | Stability margin; unchanged in 2025 |

---

## Connections

- [[excerpts/hh-rlhf]] — the public-release generalization; two-axis (helpful × harmless) extension of InstructGPT's single-axis rubric.
- [[excerpts/ultrafeedback-construction]] — the synthetic-judge replacement for the K=4-9 ranking protocol.
- [[excerpts/tulu-3-sft-mix]] — the 2024 skill-targeted expansion; 939K vs InstructGPT's 13K+33K.
- [[excerpts/prm800k]] — step-level granularity pushed to the limit; the other end of the annotation-scope axis.
- [[excerpts/judge-llm-bias]] — the 80% judge-human agreement ceiling that InstructGPT's 72.6% labeler-labeler agreement sets as the human reference point.
- [[ch-15]] — this excerpt is the backbone of §1 (rubric-as-product) and §3 (adjudication tiers); Stage-2's K-way ranking is the ancestor of §4's close-pair mining.
