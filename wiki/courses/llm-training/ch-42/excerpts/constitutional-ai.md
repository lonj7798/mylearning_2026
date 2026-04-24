---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/constitutional-ai.md
source_url: https://arxiv.org/abs/2212.08073
created_at: "2026-04-23"
---

# Excerpt: Bai 2022 — Constitutional AI: Harmlessness from AI Feedback

**Source library:** `wiki/raw-data/llm-training/papers/constitutional-ai.md`
**Paper:** Bai et al. (Anthropic), *Constitutional AI: Harmlessness from AI Feedback*, 2022.

---

## Why this source anchors ch-42

Constitutional AI is the most-deployed counter-measure to reward hacking in production alignment stacks. It replaces almost all human harmlessness labels with AI-generated ones structured around a natural-language "constitution," and it Pareto-dominates pure RLHF on the helpfulness/harmlessness frontier — which is direct evidence that CAI reduces refusal-overtraining, one of the six hacks in ch-42 §2. Ch-42 §4 reproduces the full pipeline pseudocode.

Raw-data header:

> **Core Insight:** A natural-language "constitution" of ~16 principles plus an AI critique-and-revise loop can replace almost all human harmlessness labels without sacrificing helpfulness.

## The two-stage pipeline

### SL-CAI: supervised self-revision

For each prompt in a ~180K red-team corpus:

1. Sample a harmful response from a helpful-only RLHF model.
2. Draw one principle from the 16-principle constitution.
3. Prompt the model: *"Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."* — get a critique.
4. Prompt the model: *"Rewrite the assistant's response to remove any and all harmful, unethical, … content."* — get a revision.
5. Use `(prompt, revised_response)` for SFT.

Key detail: **one principle per critique**, drawn at random. Concatenating all 16 into the prompt would let the model average out conflicts; sampling forces commitment to one axis, and prevents one blanket-refusal behavior from hacking all principles simultaneously.

### RL-CAI: AI preference labels

For each pair `(prompt, y_A, y_B)`:

1. Draw one principle.
2. Prompt the labeler with CoT: *"Consider the following conversation and two responses. Which response is more {principle}? Let's think step by step…"*
3. Read `log P("(A)")` and `log P("(B)")` at the final answer position.
4. Produce a soft preference label via `softmax([logp_A, logp_B])`, clipped to `[0.25, 0.75]`.
5. Train a BT preference model on the resulting pairs.
6. PPO the policy against that RM with a KL-to-SFT penalty (the standard InstructGPT setup).

## The constitution

~16 short written principles. Examples (paraphrased from the appendix):

- "Please choose the response that is most harmless, ethical, polite, wise, and friendly."
- "Choose the response that is as helpful as possible and gives accurate information, even when that requires explaining things the user might not want to hear."
- "Prefer responses that are less evasive — explain refusals rather than stonewall."

The last clause is the direct countermeasure to refusal overtraining. Pure RLHF harmlessness models learn to output "I cannot help with that" as the harmless-maximizing action; CAI's evasiveness clause penalizes that behavior, which is why CAI models Pareto-dominate on helpfulness while matching on harmlessness.

## Soft-label clipping

Clipping to `[0.25, 0.75]` is the paper's label-smoothing choice. The un-clipped BT loss on raw log-probs is a hack magnet — the RM becomes overconfident on easy pairs (where the labeler's log-prob margin is large) and undertrained on hard ones (where the margin is small). Clipping forces calibration and prevents the RM from learning to trust extreme labeler scores.

## Empirical result

The paper's central plot (Fig. 3): CAI points Pareto-dominate RLHF-only on helpfulness/harmlessness Elo. CAI models refuse *less* (they explain the reason rather than stonewall) while maintaining harmlessness. This is the direct measurement that CAI reduces refusal-overtraining.

Scaling observation (Fig. 4): the accuracy of AI harmlessness labels improves monotonically with the size of the labeling model. CAI gets better as the labeler gets better. This is the scaling argument behind the entire RLAIF line.

## Hacking vulnerabilities this inherits

CAI is not hack-proof:

- **Judge biases** from [[judge-llm-bias]] — position, verbosity, self-enhancement — propagate through the CoT preference labels into the RM.
- **Reward-model overoptimization** still applies; the RM is a scalar proxy and Skalse 2022 still holds.
- **Constitution-hacking:** the principles are natural-language, and a capable policy can learn to satisfy the literal wording while violating the intent (the "fictional character said the harmful thing" exploit is a well-known class).

CAI's defense is two-fold: (a) AI labels are cheap, so the RM can be refreshed as the policy drifts, partially mitigating stale-RM Goodhart; (b) the constitution is text-editable, so discovered exploits can be patched by adding principles — a reward-specification knob scalar RMs lack.

## Connection to the broader stack

CAI demonstrates that a principled RLAIF pipeline with only ~a few thousand human helpfulness preferences can reach human-feedback parity on harmlessness. It is the direct ancestor of production Claude models and the canonical RLAIF reference alongside [[rlaif-scaling]]. It is preference-shaped rather than verifiable-reward-shaped; RLVR ([[rlvr-tulu3]], [[deepseek-r1]]) is the complementary structural defense for tasks with a ground-truth checker.

## Takeaways for the chapter

1. The critique-and-revise loop is the deployed firewall for refusal overtraining.
2. Sampling one principle per critique (not concatenating) is a specific engineering choice that prevents cross-principle hacking.
3. Soft-label clipping to [0.25, 0.75] is a required calibration trick, not a free hyperparameter.
4. CAI inherits all LLM-judge biases — it is a structural improvement in *coverage* of harmlessness, not a bias-elimination step.
5. The constitution is text-editable; this is the policy knob scalar RMs do not have.
