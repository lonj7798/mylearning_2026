<!-- scope: reasoning-trace synthesis — question-side rewriting (self-verification, FOBAR, backward reasoning)
     deps: [[self-instruct]]
     see-also: [[mammoth]], [[openmathinstruct]], [[mathscale]]
-->

# MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models
- **Core Insight:** Augmenting a math SFT set by rewriting the *question* four ways (self-verification, FOBAR, answer-augmentation, rephrasing) yields larger gains than just generating more CoT answers — problem-side diversity matters as much as solution-side diversity.
- **Guideline:** For math SFT, pair each seed problem with multiple question-rewrites (backward-reasoning, variable-abstracted, mask-and-solve) and multiple CoT solutions per rewrite; this inoculates against memorization and teaches solving in both directions.
- **Authors:** Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, Weiyang Liu
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2309.12284
- **Relevant topics:** math reasoning, data augmentation, backward reasoning, question rewriting

## Abstract
MetaMath introduces **MetaMathQA**, a 395K-example math SFT corpus built by four complementary question-augmentation operations applied to GSM8K and MATH training problems. The resulting MetaMath-7B/13B/70B models set the open-source state-of-the-art on GSM8K and MATH at release (2023), and the question-augmentation trick is reused downstream in OpenMathInstruct-2, WizardMath, and MAmmoTH.

## Key Contributions
- Four specific question-rewrite operators: **rephrasing**, **self-verification (SV)**, **FOBAR**, **answer augmentation**.
- 395K MetaMathQA training set, publicly released.
- MetaMath-70B: 82.3% GSM8K, 26.6% MATH — SOTA open at release.
- Demonstration that question-side diversity gives larger lift per-example than solution-side (multiple CoTs for same problem).

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** GSM8K (7.5K) + MATH (7.5K) train problems with gold answers.
- **Question-augmentation operators:**
  1. **Answer Augmentation (AnsAug):** keep question unchanged, sample K CoT solutions, keep correct ones.
  2. **Rephrasing:** prompt GPT-3.5-turbo to rewrite the question in different wording preserving the math; re-solve.
  3. **Self-Verification (SV):** take an original (Q, A) pair, rewrite it as "Given Q and the candidate answer A', determine correctness; if not, fix." This forces the model to learn verification.
  4. **FOBAR (Forward-Backward Reasoning):** mask a number in Q, treat the original answer as known, and ask "what is the masked number?" — an inverse problem whose solution requires backward reasoning.
- **Generation step:** each augmented question is re-solved with GPT-3.5 to produce CoT traces.
- **Filtering:** keep only solutions whose final answer matches gold (for AnsAug / Rephrasing) or whose reconstructed number matches the masked value (for FOBAR/SV).
- **Output shape:** 395K (question, CoT, answer) triples; ~25× the seed count. Average trace length 300–600 tokens.
- **Teacher:** GPT-3.5-turbo (text-davinci rewrites in ablations).
- **Cost / compute:** not disclosed; on the order of $5–15K in API fees at 2023 rates.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** short CoT (300–600 tokens) — non-reflective.
- **Trace style:** standard CoT; FOBAR traces explicitly demonstrate reverse-direction reasoning.
- **Correctness verifier:** numeric exact-match (GSM8K), symbolic match (MATH).
- **FOBAR recipe (concrete template):**
  ```
  Original: "Jane has 3 apples and buys 5 more. How many apples?" → 8
  FOBAR:   "Jane has 3 apples and buys x more. She now has 8. What is x?" → 5
  ```
- **SV recipe:** "Is the following answer correct? Question: … Proposed Answer: … . If not, explain and fix."
- **Why FOBAR helps:** teaches the model that a chain of reasoning can be run in reverse — reduces "direction overfitting" to forward word-problem templates.

## Quality / diversity evaluation
- MetaMath-7B: **66.5 GSM8K, 19.8 MATH**.
- MetaMath-70B: **82.3 GSM8K, 26.6 MATH**.
- Ablation: AnsAug alone gives ~4-point gain over vanilla SFT; adding Rephrasing +3; SV +2; FOBAR +3. Stacking all four is additive.

## Risks + gotchas
- **Teacher ceiling:** GPT-3.5 answer correctness bounds dataset quality — noisy on hard MATH problems (final-answer accuracy ~30% on MATH level-5).
- **FOBAR soundness:** not every forward problem has a unique backward answer; authors filter but some ambiguous FOBARs leak in.
- **Rephrasing drift:** rephrased problems occasionally change the numerical answer; correctness filter catches most but not all.

## Connections
- Operator reuse: [[wizardmath]] (Evol-Instruct on math adopts Rephrasing), [[mammoth]] (composes MetaMathQA into its mix), [[openmathinstruct-2]] (question-augmentation as core technique).
- Conceptual ancestor: [[self-instruct]] (question-generation idea) + [[mammoth]] (hybrid-CoT).
- Contrasts long-CoT curation: [[s1]], [[limo]].
