<!-- scope: DeepSeek-R1 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[deepseek-v2]], [[deepseek-v3]], [[qwen-3]]
-->

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — Technical Report
- **Core Insight:** Pure RL (no SFT) can produce emergent chain-of-thought reasoning in LLMs.
- **Guideline:** RL reward design can teach reasoning patterns that supervised data cannot.

- **Organization:** DeepSeek AI
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2501.12948
- **Relevant chapters:** Reinforcement learning for reasoning, GRPO, emergent reasoning behaviors, knowledge distillation, chain-of-thought

## Abstract
We introduce DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step. Through RL, DeepSeek-R1-Zero naturally emerges with numerous powerful and intriguing reasoning behaviors, including self-verification, reflection, and dynamic strategy adaptation. DeepSeek-R1 builds on this foundation, incorporating a small amount of cold-start data before RL to improve readability and formatting. DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on reasoning tasks. Additionally, the reasoning capabilities can be systematically harnessed to guide and enhance smaller models through distillation.

## Architecture Summary

| Component | Value |
|-----------|-------|
| Base Model | DeepSeek-V3-Base |
| Total Parameters | 671B |
| Active Parameters | 37B per token |
| Architecture | MoE with MLA (same as DeepSeek-V3) |
| RL Algorithm | Group Relative Policy Optimization (GRPO) |

DeepSeek-R1 makes no architectural changes to DeepSeek-V3. All innovations are in the training methodology: pure RL (R1-Zero), cold-start data curation, multi-stage training pipeline, and distillation.

## Key Architectural Innovations

1. **Pure RL without SFT (R1-Zero)** — demonstrates that reasoning capabilities (self-verification, reflection, extended chain-of-thought) can emerge spontaneously through reinforcement learning alone, without any human-annotated reasoning traces. This is a fundamental finding about how reasoning emerges in LLMs.
2. **GRPO (Group Relative Policy Optimization)** — maximizes J_GRPO(theta) using group-based advantage estimation where advantage A_i = (r_i - mean(rewards))/std(rewards). Eliminates the need for a separate critic model (unlike PPO), reducing compute requirements. For each question, samples a group of outputs, computes rewards, normalizes within the group.
3. **Emergent "Aha moment"** — during RL training, the model spontaneously learns to reconsider its approach mid-solution, allocating more compute to harder problems through longer reasoning chains. This self-correction behavior was not explicitly programmed.
4. **Cold-start data strategy** — uses thousands of curated long chain-of-thought samples to initialize the model before RL, improving readability and formatting while preserving the emergent reasoning from RL. Format uses |special_token|<reasoning>|special_token|<summary>.
5. **Four-stage training pipeline** — structured progression from cold-start SFT through reasoning RL to general-purpose RL, each stage building on the previous.
6. **Distillation to smaller models** — demonstrates that reasoning capabilities from the 671B model can be effectively transferred to models as small as 1.5B parameters via SFT on 800K R1-generated samples.

## Design Decisions and Tradeoffs

- **Pure RL (R1-Zero) vs. SFT+RL (R1):** R1-Zero proves the concept that reasoning can emerge from RL alone, but suffers from poor readability, language mixing, and formatting issues. R1 adds a small cold-start SFT phase to fix these issues while preserving the emergent reasoning.
- **GRPO over PPO:** GRPO removes the critic model entirely, using in-group normalization instead. Simpler, cheaper, and avoids value function estimation errors, but may have higher variance.
- **Rule-based rewards:** For math and coding, uses verifiable correctness rewards (exact match, test cases) rather than learned reward models. This provides clean, unambiguous signal but limits applicability to domains where correctness can be verified.
- **Language consistency reward:** Added explicitly during RL to prevent language mixing (e.g., answering in Chinese when the question is in English). The proportion of target language tokens in the CoT is used as a reward signal.
- **Distillation over RL for small models:** Found that SFT distillation from R1 outperforms direct RL on small models. For example, Qwen-32B via distillation achieves 72.6% on AIME 2024 vs. 47.0% with RL-only. However, the authors note that applying RL after distillation yields further gains.
- **Four stages over fewer:** The multi-stage pipeline adds complexity but each stage addresses specific issues: cold start fixes formatting, reasoning RL builds math/code capability, rejection sampling SFT expands to general tasks, and final RL polishes everything.

## Training Details

**Stage 1 — Cold-Start SFT:**
- Thousands of curated long-CoT examples
- Fine-tune DeepSeek-V3-Base
- Focus on readability format: reasoning within special tokens, summary at end

**Stage 2 — Reasoning-Oriented RL:**
- GRPO on math, coding, science, and logic domains
- Rule-based rewards (correctness verification)
- Language consistency reward to prevent mixing
- Train until convergence

**Stage 3 — Rejection Sampling + SFT:**
- ~600K reasoning samples via rejection sampling from Stage 2 model
- ~200K non-reasoning samples (writing, QA, translation, self-cognition)
- Total: ~800K samples, trained for 2 epochs

**Stage 4 — All-Scenario RL:**
- Broad prompt distribution across all domains
- Reasoning tasks: rule-based rewards
- General tasks: preference reward models for helpfulness/harmlessness
- Assessment based on final summary only (not reasoning trace)

**Distilled models:**
- Base models: Qwen2.5-1.5B/7B/14B/32B, Llama-3.1-8B, Llama-3.3-70B
- Training data: 800K samples curated from DeepSeek-R1 outputs
- Method: Direct SFT (no RL applied to distilled models)

## Performance Highlights

**DeepSeek-R1 (full model):**

| Benchmark | Score | Comparison |
|-----------|-------|------------|
| AIME 2024 (pass@1) | 79.8% | Slightly surpasses o1-1217 |
| AIME 2024 (cons@64) | 86.7% | — |
| MATH-500 | 97.3% | On par with o1-1217 |
| GPQA Diamond | 71.5% | — |
| MMLU | 90.8% | Up from V3's 88.5% |
| Codeforces Elo | 2,029 | 96.3rd percentile |
| LiveCodeBench | 65.9% | — |
| SWE-Bench Verified | 49.2% | — |
| AlpacaEval 2.0 | 87.6% LC win rate | — |
| ArenaHard | 92.3% win rate | — |

**DeepSeek-R1-Zero progression:**
- AIME 2024: 15.6% -> 71.0% (pass@1), 86.7% (majority voting)

**Distilled model highlights:**
| Model | AIME 2024 | MATH-500 |
|-------|-----------|----------|
| R1-Distill-Qwen-1.5B | 28.9% | 83.9% |
| R1-Distill-Qwen-7B | 55.5% | 92.8% |
| R1-Distill-Qwen-14B | 69.7% | 93.9% |
| R1-Distill-Qwen-32B | 72.6% | 94.3% |
| R1-Distill-Llama-70B | 70.0% | 94.5% |

- The 1.5B distilled model outperforms GPT-4o (9.3%) on AIME 2024.
- The 32B distilled model outperforms OpenAI o1-mini across various benchmarks.
