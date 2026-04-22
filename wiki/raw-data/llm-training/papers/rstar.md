<!-- scope: reasoning-trace synthesis — MCTS + mutual-consistency self-verification for small-model math
     deps: [[lets-verify]]
     see-also: [[rstar-math]], [[math-shepherd]], [[omegaprm]]
-->

# rStar: Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers
- **Core Insight:** A small LLM can solve math at levels far above its SFT-only ceiling by running Monte-Carlo Tree Search over a rich action space (five reasoning moves) and filtering candidate trajectories with a **mutual-consistency** check against a second small discriminator model — no stronger teacher needed.
- **Guideline:** For inference-time reasoning boosts (and for generating high-quality synthetic traces from weak models), use MCTS with a diversified action-space (decompose, subquestion, rephrase, self-answer, propose) and reject trajectories on which two independent small models disagree.
- **Authors:** Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, Mao Yang (Microsoft Research Asia)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2408.06195
- **Relevant topics:** reasoning, MCTS, small-model reasoning, self-verification, test-time scaling

## Abstract
rStar is a self-play reasoning procedure in which a generator small LLM and a discriminator small LLM cooperate without any external verifier. The generator runs MCTS over an action space of five human-like reasoning moves; the discriminator, when shown a partial trajectory, is asked to complete it — if its completion matches, the trajectory passes. Applied to LLaMA2-7B and Mistral-7B, rStar lifts GSM8K from 12% → 63% (LLaMA2-7B) and from 37% → 82% (Mistral-7B), using only the base models' own capabilities.

## Key Contributions
- **Five-action reasoning move set** (A1–A5): one-step CoT, subquestion decomposition, rephrasing, direct-answer with verification, propose-a-new-subquestion.
- **Mutual-consistency verifier**: two independent small LLMs cross-check each other instead of one relying on a stronger model.
- Eliminates need for (a) a large teacher and (b) a trained PRM/reward model.
- Strong results across GSM8K, MATH, SVAMP, ASDiv, MultiArith.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** a math problem. No problem-bank augmentation.
- **Generation — MCTS stage:** the generator runs MCTS for N rollouts per problem. At each node, the policy chooses among the five actions; the chosen action produces a new partial reasoning-step child node. Rollout expands down to a leaf (final answer) and receives a reward 0/1 based on self-consistency of its final answer across rollouts.
- **Action set (A1–A5):**
  - A1: propose a one-step chain-of-thought.
  - A2: decompose into subquestions and solve sequentially.
  - A3: directly answer a subquestion and then verify the partial answer.
  - A4: rephrase the question to simplify.
  - A5: propose an intermediate subquestion not in the original text.
- **Mutual-consistency verification (discriminator stage):** for each high-reward MCTS trajectory, mask the second half of the reasoning and ask a **separately-prompted** small LLM (same family but different prompting) to complete it; the trajectory is accepted iff the discriminator's completion yields the same final answer.
- **Output shape:** per-problem, the verified best trajectory is the "synthetic" trace. Length 300–2000 tokens depending on action mix.
- **Teacher model(s):** no stronger teacher — generator and discriminator are the same base LLM (LLaMA2-7B or Mistral-7B).
- **Cost / compute:** inference-time only; typical N = 32 MCTS rollouts per problem, ~1 minute on 1 A100.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** medium-length structured CoT (300–2K tokens); not long-CoT.
- **Trace style:** tree-structured, decomposition-heavy — traces contain explicit subquestion/sub-answer segments.
- **Correctness verifier:** NONE involving gold answer. Correctness signal = (a) majority-vote across rollouts + (b) mutual-consistency with discriminator. This is the key novelty — a verifier without ground truth.
- **Error-mode filter:** mutual-consistency rejects trajectories where the two models independently "reason to different conclusions" from the same prefix.
- **Formal mutual-consistency rule:** trajectory T accepted iff `answer(generator_rollout) == answer(discriminator_completion(mask_half(T)))`.

## Quality / diversity evaluation
- LLaMA2-7B: GSM8K 12.5 → **63.1** (+50.6 absolute).
- Mistral-7B: GSM8K 36.9 → **81.7**.
- MATH: Mistral-7B 10.2 → **25.4**.
- Beats prior prompting methods (CoT-SC, ToT, Plan-and-Solve) without any fine-tuning.

## Risks + gotchas
- **Mutual-consistency false-positives:** two models can share a systematic bias and consistently agree on a wrong answer.
- **Inference cost:** MCTS with 32 rollouts is ~30× the cost of greedy CoT.
- **Action-space hand-designed:** generalization to non-math domains may require redesigning A1–A5.

## Connections
- Directly extended by [[rstar-math]] (self-evolution + PRM training loop).
- Action-space decomposition idea parallels [[star]] and [[quiet-star]].
- Contrasts PRM-based verification: [[math-shepherd]], [[omegaprm]], [[lets-verify]].
- Inference-time reasoning lineage: [[s1]] budget-forcing, [[best-of-n]], [[lets-verify]].
