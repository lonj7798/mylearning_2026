<!-- scope: iterative SFT + RL pipelines (Llama 2 RSFT; Tülu 3)
     deps: [[rejection-sampling-finetuning]], [[dpo]], [[ppo]]
     see-also: [[rlvr-tulu3]], [[rest-em]], [[spin]]
-->

# Iterative SFT + RL — Llama 2 / Tülu 3 Iterative Alignment Schemes
- **Core Insight:** Running *multiple* alternating rounds of SFT-on-filtered-samples and RL outperforms a single SFT + single RL run — because each round's RL policy produces better on-policy data for the next round's SFT, and each round's SFT stabilizes the reference distribution for the next round's RL.
- **Guideline:** Budget your post-training as N×(SFT → RL) rounds, not one monolithic run. Llama-2 used 5 rounds of RSFT + PPO; Tülu 3 uses SFT → DPO → RLVR as three distinct stages. Re-initializing the reference model between stages is essential.
- **Authors:** Hugo Touvron et al. (Llama 2), Nathan Lambert et al. (Tülu 3)
- **Year:** Llama 2 paper 2023; Tülu 3 report 2024
- **URLs:**
  - Llama 2: https://arxiv.org/abs/2307.09288 (RSFT loop described in §3.2, §3.3)
  - Tülu 3: https://arxiv.org/abs/2411.15124 (§5 post-training pipeline)
- **Relevant topics:** rejection-sampling fine-tuning, iterative alignment, SFT→DPO→RL stacks, post-training pipelines

## Abstract (synthesized)
Llama 2 introduced Rejection-Sampling Fine-Tuning (RSFT): in each of 5 rounds, sample K completions per prompt from the current model, score them with a reward model, fine-tune on the top-scoring completions, then run PPO. Tülu 3 generalizes this into a three-stage schedule — SFT on carefully curated data → DPO on synthetic preferences → Reinforcement Learning from Verifiable Rewards (RLVR) on math / code / IF prompts — with explicit reference-model resetting between stages. Both works report that the iterative structure is not incidental: ablating it costs ~3–5 points on AlpacaEval / MT-Bench.

## Key Contributions
- **Llama 2 RSFT loop:**
  - K=10 samples/prompt from current model; keep top-1 by RM score.
  - SFT on kept samples for 1 epoch.
  - Then PPO with KL-controller for several thousand steps.
  - Repeat 5 rounds.
- **Tülu 3 pipeline:**
  - Stage 1 — SFT on Tülu-3-SFT-mix (≈1M instances, multi-task, synthetic-heavy).
  - Stage 2 — DPO on Tülu-3-DPO-mix (≈300K pairs, synthetic via GPT-4o / UltraFeedback).
  - Stage 3 — **RLVR** (see [[rlvr-tulu3]]): PPO with binary verifiable reward on math + code + strict instruction-following prompts.
  - Reference reset between stages: DPO uses the SFT checkpoint as reference; RLVR uses the DPO checkpoint.
- Both papers show large gains from the *iteration*, not just from added data.

## Key Figures/Tables to Study
- **Llama 2 Figure 20 (reward curve across RSFT rounds):** monotone gain each round, saturating at round 5.
- **Llama 2 Table 5 (PPO vs RSFT alone):** RSFT+PPO dominates either alone.
- **Tülu 3 Table 2 (stage-ablation):** removing DPO or RLVR costs 3–5 points averaged across eval.
- **Tülu 3 Figure 3 (MATH and IFEval per stage):** RLVR stage adds the final 4–6 points on verifiable tasks.

## Technical Details
- **Llama 2:**
  - RSFT K=10 samples, temperature 0.8; threshold = top-1.
  - Separate RMs for helpfulness and safety; combined via weighted sum.
  - PPO: KL coef 0.01 (adaptive), clip 0.2, batch 1K rollouts, lr 1e-6.
- **Tülu 3:**
  - SFT lr 5e-6, 2 epochs, linear decay.
  - DPO β=5, lr=5e-7, 1 epoch.
  - RLVR: PPO with value head removed → REINFORCE++-style baseline, binary verifier reward ∈ {0, 1}, KL-to-SFT-ref β=0.05.
- **Key hyperparameter:** reference resets — critical; without them, DPO/RLVR keeps dragging the policy backwards toward the stale SFT reference.

## Connections
- Generalizes [[rejection-sampling-finetuning]] (the Llama-2 primitive) into a staged pipeline.
- Tülu 3's RLVR stage is a specific instance of [[rlvr-tulu3]].
- Supports the [[spin]] / [[rest-em]] / [[self-rewarding-lm]] thesis that iteration > single-pass.
- Post-DeepSeek-R1 (see [[deepseek-r1]]), the pipeline contracts even further: SFT (cold-start) → pure GRPO RLVR; iteration is compressed into the RL dynamics itself.
- The "reset reference between stages" trick is what [[self-correct-rl]]'s Stage I/II and [[spin]]'s iteration formally require.
