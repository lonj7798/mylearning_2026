<!-- scope: Yao, Liu et al. blog (first published 2025-08, updated 2025-10): vLLM-sampler vs FSDP-learner probability mismatch in hybrid RL frameworks, truncated importance sampling (TIS) for REINFORCE and PPO, BF16/FP8/INT8 rollouts, and the system factors (parallelism, response length, sampler backend) that change the mismatch
     deps: [[ppo]], [[verl-rollout]]
     see-also: [[verl-ppo-loss]], [[async-rollout]], [[areal-async-rl]], [[gspo]], [[dapo]], [[thinkingmachines-defeating-nondeterminism]]
-->

# Your Efficient RL Framework Secretly Brings You Off-Policy RL Training
- **Core Insight:** With the same parameters, the vLLM sampler and the FSDP learner in verl assign different probabilities to the same tokens (maximum difference 1.0 on DAPO with Qwen2.5-32B, where some tokens have π_vllm = 1 and π_fsdp = 0), and a DAPO-32B run whose only change is the truncated ratio min(π_fsdp/π_vllm, C) shows higher downstream performance within the 250 steps that were run (Fig. 1; "Normal RL training" paragraph; DAPO experiment paragraph; the OPT 2025 workshop version names the benchmark as AIME 2024).
- **Guideline:** When rollouts come from an inference engine and gradients from a separate training backend, record the sampler log-probabilities and multiply the policy-gradient term by a truncated learner/sampler ratio, because in the blog's runs this stabilized INT8-rollout DAPO-32B training and brought INT8-rollout PPO on GSM8K (Qwen2.5-0.5B) to accuracy similar to BF16 rollouts (Fig. 2, Fig. 5). When the sampler-learner gap is small, the authors observed no gain from TIS (Fig. 3, DeepSeek-R1-Distill-Qwen-1.5B).
- **Authors:** Feng Yao*, Liyuan Liu*, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, Jianfeng Gao (* equal contribution; BibTeX at the end of the post). The OPT 2025 workshop version lists Microsoft Research, UC San Diego, and Mila.
- **Year:** 2025 (first published 2025-08-05, last updated 2025-10-13, per the page header; blog post, not on arXiv)
- **URL:** https://fengyao.notion.site/off-policy-rl (code linked from the header: github.com/yaof20/verl/tree/flash-rl/recipe/flash_rl)
- **Source type:** practitioner evidence (blog with experiments, W&B logs, and released scripts)
- **Relevant topics:** training-inference mismatch, off-policy correction, truncated importance sampling, PPO, DAPO, quantized rollouts (INT8, FP8), entropy collapse, KL estimator bias, tensor and Ulysses sequence parallelism, response length, MoE routing

## Summary
Hybrid RL frameworks such as verl generate rollouts with an inference engine (π_sampler, e.g. vLLM or SGLang) and compute gradients with a training backend (π_learner, e.g. FSDP or Megatron). The REINFORCE update then becomes E_{a∼π_sampler(θ)}[R(a)·∇θ log π_learner(a, θ)], which is off-policy whenever the two backends give different token probabilities for the same θ. The authors patched vLLM to return the probabilities actually used for sampling (upstreamed) and to optionally cast lm_head to fp32; the mismatch remained (Fig. 1). They propose truncated importance sampling (TIS) and extend it to PPO. Experiments cover the DAPO recipe on Qwen2.5-32B, PPO on GSM8K with Qwen2.5-0.5B under BF16, FP8, and INT8 rollouts, and DeepSeek-R1-Distill-Qwen-1.5B. A later section measures which system factors change the mismatch on DAPO-32B and Polaris-7B.

## Key Contributions
- A measurement that sampler and learner token probabilities differ under shared weights, persisting after two vLLM patches (sampling probabilities, fp32 lm_head) (Fig. 1, patch paragraphs).
- TIS for REINFORCE and for PPO, where the truncated ratio uses θ_old and the PPO clip ratio stays learner-over-learner (formula paragraphs).
- A comparison of TIS with PPO-IS and untruncated vanilla IS under BF16, FP8, and INT8 rollouts, with a variance argument (Fig. 4, callout).
- Side effects of an uncorrected gap: entropy collapse, long responses, negative k1 KL estimates, and logged rewards that disagree with downstream accuracy (Fig. 5, Fig. 6).
- A factor analysis: parallelism differences and long responses raise the maximum mismatch; the sampler backend alone does not decide it (Fig. 7-11).

## Key Figures/Tables to Study
- **Figure 1** — DAPO-32B token-probability differences and downstream performance with and without TIS (4 nodes of 8 H100).
- **Figure 2** — GSM8K PPO with BF16 vs INT8 rollouts, Qwen2.5-0.5B (1 node of 4 A6000).
- **Figure 4** — TIS vs PPO-IS vs vanilla-IS under BF16/FP8/INT8, plus vLLM-FSDP KL divergence.
- **Figures 5-6** — DAPO-32B INT8 and BF16 runs: entropy, response length, k1 KL, logged reward vs AIME.
- **Figures 7-11** — Max Mismatch and Mean Mismatch under parallelism, response length, and sampler backend.

## Technical Details
**TIS for REINFORCE (formula paragraph).** E_{a∼π_sampler(θ)}[min(π_learner(a, θ)/π_sampler(a, θ), C)·R(a)·∇θ log π_learner(a, θ)]. Here a is a sampled action, R(a) its reward, θ the shared parameters, and C a truncation hyperparameter. The untruncated ratio gives the standard importance-sampling identity E_{π_learner}[R] = E_{π_sampler}[(π_learner/π_sampler)·R] (Additional Discussion: Importance Sampling).

**TIS for PPO (formula paragraph).** E_{a∼π_sampler(θ_old)}[min(π_learner(a, θ_old)/π_sampler(a, θ_old), C)·∇θ min(r·Â, clip(r, 1−ε, 1+ε)·Â)], with r = π_learner(a, θ)/π_learner(a, θ_old), Â the advantage, and ε the PPO clip range. π_learner(a, θ_old) is the FSDP recomputation of the rollout probabilities.

**Why the two variants are unstable (callout).**
1. Vanilla-IS: gradient variance scales with the squared ratio. For a token with ratio 16, noise is amplified 256× by vanilla-IS, 4× by TIS with C = 2, and 64× by TIS with C = 8.
2. PPO-IS puts π_learner(a, θ)/π_sampler(a, θ_old) inside the clip. Even at θ = θ_old this ratio is not 1, so clipping happens with high probability and updates carry less information. The authors started with PPO-IS and it did not perform well in their setting.
3. AReaL's decoupled PPO drops a sample entirely when the ratio exceeds a threshold instead of truncating it (Additional Discussion: Decoupled PPO).

**Experiments.**
- DAPO-32B: only the first 250 steps were completed for resource reasons; the TIS term is the only difference between the two runs (DAPO paragraph, Fig. 1).
- GSM8K (verl tutorial example): the maximum token-probability difference is about 0.4 with normal training, versus 1.0 on DAPO-32B. PPO with INT8 rollouts degrades relative to BF16 rollouts; TIS brings the INT8 run to similar performance (Fig. 2).
- DeepSeek-R1-Distill-Qwen-1.5B with BF16 rollouts in both runs (4 nodes of 8 H100): the mismatch is small and TIS gives no gain; in strict on-policy training the ratio equals 1.0 (Fig. 3 caption and paragraph).
- TIS outperforms PPO-IS and vanilla-IS, especially for FP8/INT8; PPO-IS and vanilla-IS reach near-0 accuracy with INT8 rollouts (Fig. 4).
- DAPO-32B INT8: entropy falls below 0.2 and keeps decreasing, responses become abnormally long, and TIS reverses both (Fig. 5). BF16: no severe entropy collapse, and TIS still raises entropy (Fig. 6).
- The k1 estimator log π_old^fsdp(a) − log π^fsdp(a) is biased when a is sampled from π_old^vllm; the INT8 run frequently logs negative KL, while the TIS run stays positive for most of training (Fig. 5).
- Logged reward is E_{π_sampler}[R]. BF16 rollout without TIS logs a higher reward than with TIS, while AIME accuracy is higher with TIS (Fig. 6).

**Mismatch factors (first 512 prompts of DAPO-Math-17k; DAPO-32B and Polaris-7B).** Max Mismatch = max over response tokens of |p_sampler(a) − p_learner(a)|; Mean Mismatch = the per-token average.
- Parallelism: sampler vLLM TP1 with learner FSDP SP1 gives 1 response with Max Mismatch > 0.5; sampler TP2 gives 2; adding learner Ulysses SP8 gives double digits (Fig. 7). A TP8 learner matches a TP2 sampler more closely than an SP8 learner does; same-parallelism TP2 and TP4 give few responses > 0.5 (Fig. 8). Mean Mismatch and KL show no significant difference across these configurations.
- Length: responses capped at 20K tokens have higher Max Mismatch than 4K caps, with similar Mean Mismatch (Fig. 9). Five 4K responses raise Max Mismatch modestly over one 4K response; one 20K response raises it more, and its first 4K tokens often exceed an independent 4K response (Fig. 10).
- Backend: vLLM, SGLang, and SGLang with deterministic kernels show no consistent winner (SGLang lower mean mismatch on DAPO-32B, vLLM lower on Polaris-7B); deterministic sampling without aligning the training configuration does not noticeably reduce the gap (Fig. 11).
- Other observations without figures: dense and MoE models of comparable scale (32B and 30B) show different mismatch levels, and base models show smaller mismatch than post-trained counterparts. The MoE argument (precision-sensitive routing, specialized kernels) is stated without MoE experiments (Interpretation).

## Recipe ledger
Values below come from the released scripts linked in the blog header (github.com/yaof20/verl@fd02f7b, branch flash-rl, read 2026-09-14). The blog body does not print hyperparameters.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DAPO run from Qwen2.5-32B | 32B | RL | TIS cap C (`imp_ratio_cap`) | 8 (run `flash-bf16-TIS-8`); default −1 = off | recipe/flash_rl/README.md L110-113; dapo_qwen32b_bf16.sh L5, L106 | verified 2026-09-14 | Fig. 1: TIS vs no TIS; no sweep over C reported |
| DAPO run from Qwen2.5-32B | 32B | RL | advantage; KL | grpo; KL in reward off, kl_coef 0.0, KL loss off | dapo_qwen32b_bf16.sh L10-15 | verified 2026-09-14 | no ablation reported |
| DAPO run from Qwen2.5-32B | 32B | RL | clip ε low / high; dual-clip c | 0.2 / 0.28; 10.0 | L17-18, L85 | verified 2026-09-14 | no ablation reported |
| DAPO run from Qwen2.5-32B | 32B | RL | max prompt / response; overlong buffer | 2,048 / 20,480 tokens; buffer 4,096, penalty factor 1.0 | L20-24 | verified 2026-09-14 | no ablation reported |
| DAPO run from Qwen2.5-32B | 32B | RL | prompts per step; generation batch; samples per prompt; mini-batch | 512; 1,536 prompts; 16; 32; group filter on acc, up to 10 generation batches | L28-35 | verified 2026-09-14 | no ablation reported |
| DAPO run from Qwen2.5-32B | 32B | RL | LR; warmup; weight decay; grad clip; entropy coeff; loss aggregation | 1e-6; 10 steps; 0.1; 1.0; 0; token-mean | L26, L99-107 | verified 2026-09-14 | no ablation reported |
| DAPO run from Qwen2.5-32B | 32B | RL | rollout temperature / top-p; validation top-p | 1.0 / 1.0; 0.7 | L52-55 | verified 2026-09-14 | no ablation reported |
| DAPO run from Qwen2.5-32B | 32B | RL | parallelism; nodes | sampler TP 2; learner Ulysses SP 8; 4 nodes × 8 GPUs | L41, L58, L63, L132-133 | verified 2026-09-14 | Fig. 7-8 relate TP/SP differences to mismatch |
| GSM8K PPO, Qwen2.5-0.5B-Instruct | 0.5B | RL | TIS cap C | 2 (runs `...bf16-TIS-2`, `...w8a8-TIS-2`) | README L84-101; gsm8k_qwen0_5b_int8.sh L2, L32 | verified 2026-09-14 | Fig. 2, Fig. 4; no sweep over C reported |
| GSM8K PPO, INT8-rollout script, Qwen2.5-0.5B-Instruct | 0.5B | RL | advantage; batch; mini-batch; lengths; LR actor / critic; epochs | GAE; 256; 64; prompt 1,024 / response 512; 1e-6 / 1e-5; 15 | gsm8k_qwen0_5b_int8.sh L16-38, L55 | verified 2026-09-14 | no ablation reported |
| GSM8K PPO, Qwen2.5-0.5B | 0.5B | RL | model name | blog: "Qwen2.5-0.5B dense model"; script: Qwen/Qwen2.5-0.5B-Instruct | Fig. 2 caption; script L24 | conflict | the script is the released configuration |

Implementation at the same commit: `imp_ratio = exp(old_log_prob − rollout_log_probs)`, clamped at `max=imp_ratio_cap` (no lower bound), multiplied into the per-token PPO loss (verl/trainer/ppo/core_algos.py L586-590).

## Findings relevant to negative feedback, long context, and agentic training
- **Negative advantages (negative as gradient):** the authors' explanation of entropy collapse is that for negative-advantage rollouts with π_learner/π_sampler < 1, a decrease in π_learner may not appear in π_sampler, so updates keep pushing π_learner down; TIS keeps the untruncated ratio for ratios below 1, removing the bias for that subset. The authors state that the exact mechanism is an open question (mechanism paragraph; Interpretation).
- **Long responses:** Max Mismatch grows with uninterrupted response length (20K vs 4K) while Mean Mismatch does not (Fig. 9-10).
- **Agentic training:** a news line states that REINFORCE++ verified TIS in a tool-integrated reasoning setting; the blog gives no numbers for it.

## Connections
- [[verl-ppo-loss]] — verl's later rollout-correction options (`rollout_is_weights`) that apply token- or sequence-level ratios.
- [[verl-rollout]] — how verl's vLLM server returns per-token log-probabilities.
- [[dapo]], [[verl-dapo-recipe]] — the recipe used in the 32B experiments.
- [[ppo]] — the clipped objective that TIS multiplies.
- [[areal-async-rl]] — decoupled PPO that drops samples above a ratio threshold.
- [[gspo]] — sequence-level ratios, which the authors call orthogonal to TIS.
- [[async-rollout]] — off-policyness from stale weights, a separate source of mismatch.
- [[thinkingmachines-defeating-nondeterminism]] — system-level work on inference nondeterminism.
- [[entropy-collapse-ppo]], [[entropy-mechanism-llm-rl]] — entropy collapse observed in the INT8 runs.
- [[john-schulman-kl-tricks]] — the k1 KL estimator used in the diagnostic.
- [[reinforce-plus-plus]] — cited in the news line on tool-integrated reasoning.
- [[deepspeed-ulysses]], [[mixed-precision]] — sequence parallelism and precision settings in the factor analysis.

## Verification
- Created on 2026-09-14 from https://fengyao.notion.site/off-policy-rl (page header: last updated 2025-10-13). A direct fetch returns an empty Notion page; the text was read from an r.jina.ai rendering of the same URL. Content inside collapsed toggles may be missing, and figure values (images) are not transcribed. Cross-checked against the OPT 2025 workshop version "On the Rollout-Training Mismatch in Modern RL Systems" (opt-ml.org/papers/2025/paper116.pdf) and the released scripts at github.com/yaof20/verl@fd02f7b.
- Audit claims not found in the source: "AIME24 pass@1 avg32" (the blog says AIME accuracy with no sampling protocol); "TIS restores accuracy with little overhead" (the overhead statement is in the OPT workshop version §4.2, not in the blog text); verl rollout_correction details (Token-TIS, Seq-TIS, Seq-MIS, Geo-RS, default rollout_is_threshold 2.0, rollout.calculate_log_probs: true) and the citation of "When Speed Kills Stability" (both are in verl's Rollout Correction documentation, verl.readthedocs.io/en/latest/algo/rollout_corr.html, not in the blog).
- Not reported by the source: numeric AIME or GSM8K accuracies in text, number of seeds, FP8 run configuration, rollout throughput or speedup numbers, results for MoE models.
