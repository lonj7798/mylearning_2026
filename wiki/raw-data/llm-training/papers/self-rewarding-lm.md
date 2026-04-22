<!-- scope: self-rewarding LM — policy acts as its own judge
     deps: [[dpo]], [[rlhf-instructgpt]]
     see-also: [[meta-rewarding-lm]], [[spin]], [[trl-online-dpo]]
-->

# Self-Rewarding Language Models
- **Core Insight:** The same LM that generates can also judge — and iterating DPO on self-generated, self-judged preference pairs lifts both the actor's instruction-following *and* its reward-modeling skill beyond the supervision ceiling set by a frozen human-trained reward model.
- **Guideline:** When human preference data is scarce, use the current policy as an LLM-as-a-Judge to label its own samples, do DPO, then repeat — but cap the number of iterations at 3, because reward-signal drift starts dominating after that.
- **Authors:** Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston
- **Year:** 2024 (Meta AI / NYU)
- **URL:** https://arxiv.org/abs/2401.10020
- **Relevant topics:** self-improvement, LLM-as-a-judge, iterative DPO, preference-model distillation

## Abstract
We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613.

## Key Contributions
- Introduced the **Self-Rewarding** training loop: one model plays both Actor and Judge, with the Judge role invoked via a fixed evaluation prompt template.
- Showed the judge signal *improves* with each iteration — an emergent property absent from fixed-RM pipelines.
- Demonstrated 3 iterations of Iterative DPO on Llama-2-70B lifts AlpacaEval 2.0 win-rate from 9.94% (SFT) → 15.38% → 20.44% → 20.8%, passing GPT-4 (June 2023) at iter 2.
- Showed the judge's Spearman correlation with held-out human preference improves from 0.62 (iter 0) to 0.71 (iter 3) on the same Open-Assistant rubric.
- Established the "self-rewarding" family — direct ancestor of Meta-Rewarding, SPIN-variants, and the 2024–2025 on-policy DPO recipes.

## Key Figures/Tables to Study
- **Figure 1:** the overall loop — Seed SFT → Self-Instruction + Self-Evaluation → Preference-pair DPO → repeat. Every ablation in the paper is a modification of one arrow in this diagram.
- **Figure 3 (AlpacaEval win-rate vs iteration):** shows the monotonic gain up to iter 3 and the plateau after.
- **Table 2 (judge agreement with Open-Assistant humans):** the critical table — the judge gets *better* as iterations proceed.
- **Table 4 (IFEval + MT-Bench):** shows instruction-following gains aren't lost on standard benchmarks.

## Technical Details
- **Seed data:** Open Assistant SFT pool → 3,200 prompts for initial SFT; a 1,775-pair subset for EFT (Evaluation Fine-Tuning) that teaches the judge rubric.
- **Judge prompt:** a 5-point rubric ("Additive scoring (1–5) of helpfulness, relevance, depth, clarity, and completeness") appended to every completion — identical across iterations.
- **Preference-pair construction per iteration:**
  1. Sample 4 responses per prompt from the current policy at T=0.7, top-p=0.9.
  2. Score all 4 with the policy-as-judge (pairwise or 5-point, averaged over 3 judge samples).
  3. Take the highest-scored response as `chosen`, lowest as `rejected`.
  4. Run DPO with β = 0.1 for 1 epoch from the previous iteration's checkpoint.
- **Base model:** Llama-2-70B, context 4096, AdamW lr=5e-7 for DPO steps.
- **Stopping:** 3 iterations — the paper notes iter 4 regresses on reward bench (likely reward hacking).
- **Cost asymmetry:** each iteration's judge pass dominates total compute (4 generations × 3 judge calls per prompt × ~20K prompts).

## Connections
- Direct precursor to [[meta-rewarding-lm]] (adds a meta-judge to regulate judge quality), [[spin]] (self-play with human-written data as implicit positive), and [[trl-online-dpo]] (online DPO with judge as reward source).
- Uses the DPO loss of [[dpo]] verbatim — the innovation is the preference *source*.
- Related to [[rlaif-scaling]]: both remove the human preference bottleneck, but RLAIF uses a separate frozen judge model; Self-Rewarding uses the policy itself.
- The emergent "judge improves with iteration" finding mirrors [[star]]-style rationale bootstrap: the model distills its own competence into a narrower subset.
