---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/star.md
source_url: https://arxiv.org/abs/2203.14465
created_at: "2026-04-23"
---

# Excerpt: STaR — the original self-improvement SFT bridge ch-31 is descended from

**Source library:** `wiki/raw-data/llm-training/papers/star.md`
**Artifact:** the 7-step STaR loop with the rationalization-on-failure branch

---

## Why this source anchors ch-31

STaR (Zelikman 2022) is the first clean demonstration that a model can improve by SFT on its own successful intermediate reasoning. It predates RSFT, RLHF-for-reasoning, and RLVR by a year-plus, and it is the single cleanest statement of the generate-verify-filter-SFT-repeat pattern. Every chapter-31 algorithm — RSFT, ReST-EM, V-STaR, SPIN, Self-Rewarding LM — is a descendant with a different verifier or a different filter. Without STaR's loop on the page, the rest of the chapter reads as a list of tricks instead of variations of one idea.

---

## The attested loop — quoted verbatim

From the source (lines 30–36), the STaR loop:

> 1. Prompt the base LM with a small number of rationale examples.
> 2. Generate a rationale and final answer for an unlabeled training example.
> 3. If the answer is correct, keep the rationale as synthetic supervision.
> 4. If the answer is wrong, prompt again while providing the correct answer and ask the model to produce a rationale that reaches it.
> 5. Keep only rationales that now yield the correct answer.
> 6. Finetune the model on all accepted rationale traces and repeat the loop.

Ch-31 §3 reproduces this loop with one edit: step 6 appends "7. Repeat from step 2 with the fine-tuned model" to make the iterative structure explicit. The paper implies it; ch-31 writes it down because the iteration *is* the lesson.

---

## The rationalization trick — the subtle move

From the source (line 20):

> Uses backward rationalization on failed examples, which turns a wrong attempt into useful training signal instead of discarding it.

This is the insight most re-implementations miss. Naive RSFT throws away K-1 samples per prompt. STaR throws away *zero* correct samples and, for the wrong ones, conditions on the gold answer and asks the model to produce a rationale that lands there. If the model succeeds *with* the gold-answer conditioning, that rationale is kept as training data.

The mechanism: a model that cannot solve a problem forward may still produce a coherent justification backward. Fine-tuning on the backward-rationalized trace teaches the model a reasoning pattern it could not produce unconditioned. Over iterations, the unconditioned distribution absorbs the patterns that only the conditioned distribution could produce, and the model's forward-solve rate rises.

[[v-star]] (Zelikman 2024) extends this: instead of throwing away the failed-and-unrationalized examples, train a verifier on the contrast. Ch-31 §5's "failures become training signal" framing points at both moves.

---

## Why STaR is SFT, not RL

From the source (line 45):

> The method is closer to **iterative SFT on filtered self-generated traces** than to RL.

Ch-31's central claim — that the strong post-training pipelines live in the strip between SFT and RL called rejection-sampling fine-tuning — is STaR's claim generalized. STaR is *exactly* that strip. No policy gradient, no KL controller, no value function. Just SFT on filtered self-samples, iterated.

The reason this matters for ch-31's decision tree: if STaR's core trick works on MATH with a T5-11B-scale model (as the 2022 paper shows), the same trick works on modern instruction-tuned chat models whenever a verifier is available. The chapter's "do you have a verifier" decision-tree node is a direct descendant of STaR's "can you answer-check this task?" gatekeeping.

---

## The one sentence that sets up the rest of the chapter

From the source (line 47):

> The bottleneck shifts from "collect gold CoT" to "design a good verifier / answer checker."

This is the sentence that makes ch-31 possible. Every iterative-SFT algorithm in the chapter assumes a verifier or a reward model. STaR names this constraint first and names it precisely: the algorithm is bounded by the verifier, not by the policy. The chapter's §5 table ("verifier is exact-match" vs "reward is a learned RM" vs "judge is policy-as-judge") is STaR's bottleneck sentence decomposed into a decision matrix.

---

## Ch-31's one edit to STaR

STaR keeps *one* correct trace per correct-answer problem. [[rest-em]] shows this collapses solution-path diversity by iter 3. Ch-31 follows the ReST-EM modification: keep at most 4 distinct correct traces per problem (diversity cap), and when the number of correct traces per problem drops below 4, that is the signal to stop iterating and either curate harder prompts or switch to on-policy RL. STaR does not state this cap; it is the 2023 lesson layered on top of the 2022 algorithm.

---

## Connections

- [[rest-em]] — STaR reformulated as EM over the latent rationale; 2-iter saturation curve.
- [[v-star]] — STaR with a learned verifier; failed traces become supervision.
- [[self-rewarding-lm]] — STaR with an LLM-as-judge verifier for open-ended tasks.
- [[quiet-star]] — STaR generalized from question-answer to arbitrary text continuation.
- [[rejection-sampling-finetuning]] — RSFT is STaR without the rationalization-on-failure branch.
- **ch-31 §3** — reproduces the STaR loop verbatim with the iteration step made explicit.
