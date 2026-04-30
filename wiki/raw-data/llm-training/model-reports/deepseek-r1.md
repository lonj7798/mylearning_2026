<!-- scope: DeepSeek-R1 — pure-RL reasoning model + distillation line
     deps: [[README]]
     see-also: [[deepseekmath]], [[deepseek-v3]], [[grpo]]
-->

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **Core Insight:** Pure RL on a base model, with only rule-based rewards for correctness + format, causes reasoning behaviors (reflection, verification, backtracking) to emerge without any SFT on reasoning traces.
- **Guideline:** For verifiable domains (math/code), skip the reasoning-SFT bootstrap; start RL directly from the base model and let long CoT emerge.
- **Authors:** DeepSeek-AI (~200 contributors)
- **Year:** 2025 (arXiv Jan 2025; Nature 645:633-638)
- **URL:** https://arxiv.org/abs/2501.12948
- **Relevant topics:** GRPO at scale, pure-RL reasoning, rule-based rewards, cold-start SFT for readability, distillation to small models

## Abstract
DeepSeek-R1 demonstrates that LLM reasoning abilities can be developed through pure reinforcement learning, obviating the need for human-labeled reasoning trajectories. Two models are released: **R1-Zero**, trained with RL directly on the DeepSeek-V3-Base with no SFT at all, exhibits emergent self-reflection and verification. **R1** adds a short "cold-start" SFT on ~800k examples before RL, plus a second RL stage for helpfulness + safety alignment, producing a readable, strong reasoner that beats o1 on several math/coding benchmarks. Six distilled smaller models (1.5B–70B) are released.

## Key Contributions
- R1-Zero: first public demonstration of reasoning emergence from pure RL on a base LLM.
- Rule-based rewards only: accuracy reward (correct final answer) + format reward (reasoning between `<think>...</think>` tags). No learned reward model.
- Cold-start data mitigation: R1 uses ~800K curated examples to fix R1-Zero's readability + language-mixing problems.
- Two-RL-stage pipeline: (1) reasoning RL with rule-based rewards, (2) alignment RL with helpfulness/safety preference rewards.
- Distillation line: Qwen- and Llama-based 1.5B/7B/8B/14B/32B/70B students trained on 800K reasoning traces from R1 — the small students beat GPT-4o on math.

## Key Figures/Tables to Study
- **R1-Zero "aha moment" plot:** response length climbs to multi-thousand tokens during RL; pass@1 on AIME 2024 climbs from 15.6% to 71%.
- **R1 vs o1 benchmark table:** AIME 2024 pass@1 79.8% vs o1 79.2%; MATH-500 97.3%; Codeforces Elo 2029 (96.3 percentile).
- **Training pipeline diagram:** Base -> cold-start SFT -> Reasoning RL -> Rejection Sampling SFT -> Alignment RL.

## Technical Details — Post-Training Pipeline

### R1-Zero (pure RL)
- **Base:** DeepSeek-V3-Base (671B MoE).
- **Algorithm:** GRPO.
- **Reward:** rule-based only — accuracy (sympy-verified final answer) + format (think-tag compliance). No reward model.
- **Template:** `A conversation between User and Assistant... Assistant reasons inside <think>...</think> and answers inside <answer>...</answer>.`
- **Outcome:** emergent long CoT, self-reflection ("Wait, let me reconsider..."), exploration of multiple approaches. But mixed-language output + poor readability.

### R1 (full recipe)
1. **Cold-start SFT** on ~800K curated reasoning examples with human-readable CoT format — fixes readability.
2. **Stage-1 Reasoning RL** with GRPO + rule-based rewards:
   - **Learning rate:** 3e-6
   - **KL coefficient:** 0.001
   - **GRPO clip ratio (eps):** 10 (intentionally loose — DeepSeek argues tight clipping destroys exploration)
   - **Sampling temperature:** 1.0 for rollouts
   - **Rollouts:** 16 samples per prompt (group size G=16)
   - **Max generation length:** 32,768 tokens
   - **Batch size:** 32 unique prompts/step -> 32 * 16 = 512 training samples/step
3. **Rejection-sampling SFT:** Use stage-1 RL model to generate data; filter via V3 judge; ~600K reasoning + 200K non-reasoning.
4. **Stage-2 Alignment RL:** second RL run with helpfulness + harmlessness preference rewards for general-purpose alignment.

### Distillation
- 800K reasoning traces from R1 used to SFT Qwen-2.5 (1.5B/7B/14B/32B) and Llama-3 (8B/70B) students.
- No RL on students; pure SFT.
- Distilled-R1-Qwen-32B beats o1-mini on MATH-500 and AIME.

### Scale
- Pretraining inherited from V3 (14.8T tokens, 2.788M H800-hours).
- R1 post-training compute not explicitly disclosed; believed small vs pretraining.

### Benchmark numbers
- **AIME 2024 pass@1:** R1 79.8%, R1-Zero 71.0% (pass@1 via majority vote 86.7%), V3 39.2%.
- **MATH-500:** R1 97.3%.
- **Codeforces:** Elo 2029, 96.3rd percentile.
- **MMLU:** 90.8%.

## Connections
- [[deepseek-v3]] — base model for both R1-Zero and R1.
- [[deepseekmath]] — GRPO is the algorithm used; R1 scales GRPO to full-model reasoning.
- [[grpo]] — algorithmic details.
- [[let-verify]] — process-reward lineage R1 explicitly rejects in favor of outcome reward only.
- [[tulu-3]] — RLVR is the spiritual cousin (verifiable rewards) but uses PPO; R1 uses GRPO.
- [[constitutional-ai]] — alignment RL stage 2 is the DeepSeek analog of Anthropic's helpful/harmless preference stage.
