<!-- scope: Feng Yao, Liyuan Liu et al., "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training" (Notion blog, Aug–Oct 2025): rollout–training mismatch between vLLM sampler and FSDP learner, truncated importance sampling (TIS) for REINFORCE and PPO, INT8/FP8 rollouts, PPO-IS / vanilla-IS ablation, mismatch factors (parallelism, length, backend)
     deps: [[grpo]]
     see-also: [[fp16-train-infer-mismatch]], [[thinking-machines-nondeterminism]], [[nemo-rl-tp-accuracy]], [[mai-rl-infrastructure]], [[mai-thinking-1]], [[areal-async-rl]], [[verl-rollout]], [[slime]]
-->

# Your Efficient RL Framework Secretly Brings You Off-Policy RL Training (TIS)
- **Core Insight:** Frameworks that sample with one engine (vLLM, SGLang) and train with another (FSDP, Megatron) compute the gradient E_{a∼π_sampler}[R·∇log π_learner]. Even with identical parameters the two engines give different token probabilities (some tokens get 1 under vLLM and 0 under FSDP), so "on-policy" RL is silently off-policy. Weighting by a truncated ratio min(π_learner/π_sampler, C) corrects most of the damage.
- **Guideline:** Compute the sampler–learner ratio on the *old* policy, truncate it at C, and apply it outside the PPO clip; do not fold it into the PPO ratio (PPO-IS) or use it untruncated (vanilla-IS). Expect the gap to grow with rollout quantization (INT8/FP8), with parallelism differences between sampler and learner, and with uninterrupted response length.
- **Authors:** Feng Yao*, Liyuan Liu*, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, Jianfeng Gao (*equal contribution).
- **Year:** 2025 (first published 2025-08-05; last updated 2025-10-13).
- **URL:** https://fengyao.notion.site/off-policy-rl (code: https://github.com/yaof20/Flash-RL; later NeurIPS 2025 version titled "On the Rollout-Training Mismatch in Modern RL Systems", not read)
- **Source type:** research blog (Notion)
- **Relevant topics:** rollout–training mismatch, importance sampling, truncated IS, PPO clipping, quantized rollouts, entropy collapse, KL estimation bias, parallelism mismatch

## Summary
The post writes REINFORCE as θ ← θ + μ·E_{a∼π(θ)}[R(a)·∇log π(a, θ)] and shows that hybrid systems replace the first π by π_sampler and the second by π_learner. Two vLLM patches were tried first: returning the actual sampling probabilities (upstreamed) and casting `lm_head` to FP32 (as in the MiniMax-M1 report); "the mismatch problem still exists after applying both patches". The fix is algorithmic: truncated importance sampling (TIS). The news box lists integrations in slime, VeRL, OAT, SkyRL, OpenRLHF, and verification by REINFORCE++ in tool-integrated reasoning.

## Key Contributions
- Names the problem: implementation gap turns on-policy RL off-policy (TL;DR, Fig. 1).
- TIS for REINFORCE: E_{a∼π_sampler}[min(π_learner(a, θ)/π_sampler(a, θ), C) · R(a) · ∇log π_learner(a, θ)], C a hyperparameter.
- TIS for PPO: the PPO ratio is computed between learner(θ) and learner(θ_old); a separate factor min(π_learner(a, θ_old)/π_sampler(a, θ_old), C) multiplies the PPO gradient.
- Controlled tests with INT8 and FP8 rollouts, ablations against PPO-IS and vanilla-IS, and an analysis of what drives the mismatch.

## Key Figures/Tables to Study
- Fig. 1: token-probability gap and DAPO run on Qwen2.5-32B dense, 4 nodes × 8 H100. Fig. 2: Qwen2.5-0.5B GSM8K, 4 × A6000, BF16 vs INT8 rollouts. Fig. 4: TIS vs PPO-IS vs vanilla-IS (INT8/FP8/BF16). Figs. 5–6: entropy, response length, k1 KL on DAPO-Qwen2.5-32B INT8 and BF16. Figs. 7–11: mismatch vs parallelism, length, backend.

## Technical Details
**Experiments.**
- DAPO recipe on Qwen2.5-32B: only the first 250 steps were run "due to the resource constraints"; TIS "has already helped boost the performance significantly". Max token-probability difference ≈ 1.0 in this setting.
- GSM8K (verl tutorial) on Qwen2.5-0.5B: BF16 rollouts give max difference ~0.4; INT8 rollouts give 1.0. PPO with INT8 rollouts degrades significantly; INT8 + TIS reaches performance similar to the BF16 run.
- DeepSeek-R1-Distill-Qwen-1.5B, BF16 rollouts, 4 × 8 H100: when the gap is small, TIS "cannot bring performance gain" (Fig. 3).

**Why the variants fail (TIS Analysis).** Vanilla-IS amplifies gradient variance by the squared ratio: a ratio of 16 amplifies noise 256× under vanilla-IS, 4× under TIS-2, 64× under TIS-8. PPO-IS puts π_learner(θ)/π_sampler(θ_old) inside the clip; because this ratio is not 1 even at θ = θ_old, clipping happens "with high possibility" and training becomes less informative. TIS separates the two ratios; the PPO one equals 1 at θ = θ_old. TIS "outperforms both variants consistently, especially in cases where the gap is large (e.g., FP8/INT8)"; PPO-IS and vanilla-IS reach near 0 accuracy with INT8 rollouts (Fig. 4).

**Failure signatures.** INT8 rollouts: entropy "dropping below 0.2 and continuing to decrease", abnormally long responses, and the k1 KL estimator log π_old − log π (biased by sampling from vLLM) frequently negative. With TIS, k1 stays positive for most of training. Logged reward is E_{π_sampler}[R], so a TIS run can log lower reward yet score higher on AIME (Fig. 6). Proposed mechanism: for tokens with π_learner/π_sampler < 1 and negative advantage, the learner keeps pushing probability down because the decrease is not reflected in the sampler, which can collapse entropy; TIS keeps the untruncated ratio for this subset.

**Mismatch factors (Rollout-Training Mismatch Analysis, added 2025-09-24).** DAPO-32B and Polaris-7B, first 512 prompts of DAPO-Math-17k; metrics: max and mean |p_sampler − p_learner| per response.
- Parallelism: vLLM TP1 / FSDP SP1 has one response with max mismatch > 0.5; sampler TP2 gives two; adding learner SP8 gives "double digits". A TP8 learner matches a TP2 sampler better than SP8; same TP (TP2/TP2, TP4/TP4) gives few high-mismatch responses. Mean mismatch and KL showed no significant difference across these configurations.
- Length: 20K-token responses have higher max mismatch than 4K; 5 × 4K responses add only a modest increase; the first 4K tokens of a 20K response already exceed an independent 4K response.
- Backend: vLLM vs SGLang vs SGLang deterministic mode — none dominates; deterministic sampling alone "does not noticeably reduce the gap".
- Dense vs MoE models of similar size differ; base models show smaller mismatch than post-trained ones. MoE is expected to be worse (precision-sensitive routing, special kernels).

**Relation to other corrections.** AReaL's decoupled PPO drops samples whose ratio exceeds a threshold instead of truncating ([[areal-async-rl]]). TIS is stated as orthogonal to GSPO/GMPO. [[fp16-train-infer-mismatch]] later reports that token-level TIS on GRPO in BF16 still collapses on its sanity test.

## Verification
The Notion page does not render through WebFetch; its block content was read through Notion's public `loadPageChunk` endpoint on 2026-11-24 (184 blocks; three toggle bodies under "TIS-Variants" were not returned). All numbers are quoted from the page text and captions; figures were not viewed. The C value used in the main experiments is not stated in the text; only TIS-2 and TIS-8 appear as examples.
