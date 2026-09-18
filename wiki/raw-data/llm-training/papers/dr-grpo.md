<!-- scope: Liu et al. (COLM 2025) critical study of R1-Zero-like training: base-model templates and "aha" behaviour, the two GRPO normalization biases, and Dr. GRPO
     deps: [[grpo]]
     see-also: [[ppo]], [[rloo]], [[deepseek-r1]], [[r1-zero-analysis]], [[trl-grpo]], [[verl-grpo]]
-->

# Understanding R1-Zero-Like Training: A Critical Perspective
- **Core Insight:** GRPO's division by response length |o_i| and by the per-question reward standard deviation biases the policy gradient (§3.1); Dr. GRPO removes both terms, which keeps response length from growing during training as it does under GRPO and lowers the length of incorrect responses on benchmarks (§3.2, Fig. 5), and a minimal Dr. GRPO recipe on Qwen2.5-Math-7B reaches 43.3% on AIME 2024 (Abstract, App. B Table 4).
- **Guideline:** When GRPO training with a binary outcome reward shows response length still rising after training reward has slowed, remove the std normalization and replace the per-response token average with a sum divided by a constant such as the generation budget, because this reduced incorrect-response length on the paper's benchmarks while accuracy stayed equal or higher (§3.2 Fig. 5; App. C Fig. 8, and Fig. 9 over 3 independent runs).
- **Authors:** Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, et al. (Sea AI Lab, NUS, SMU)
- **Year:** 2025 (arXiv v1 2025-03; COLM 2025)
- **URL:** https://arxiv.org/abs/2503.20783
- **Source type:** paper
- **Relevant topics:** R1-Zero-like RL, GRPO length bias, difficulty bias, prompt templates, base-model pretraining effects, token efficiency

## Abstract
The paper examines the two components of R1-Zero-like training: base models and RL. It analyses a range of base models, including DeepSeek-V3-Base, and finds that DeepSeek-V3-Base already shows "Aha moment" behaviour and that Qwen2.5 base models reason well without any prompt template, which suggests pretraining biases. It identifies an optimization bias in GRPO that increases response length during training, especially for incorrect outputs. It proposes Dr. GRPO, an unbiased method that improves token efficiency while keeping reasoning performance. A minimalist recipe built on these findings reaches 43.3% on AIME 2024 with a 7B base model.

## Key Contributions
- Templates decide whether base models answer questions or continue text; Qwen2.5-Math base models score highest with no template, about 60% above 4-shot prompting (§2.1–2.2, Table 1).
- Self-reflection keywords already appear in base models including DeepSeek-V3-Base (§2.3, App. D–E); for DeepSeek-R1-Zero, "nearly half responses with self-reflection do not achieve higher accuracy than those without self-reflection" (App. F, Fig. 15).
- Names a response-level length bias and a question-level difficulty bias in GRPO (§3.1, Fig. 4) and shows that loss normalization by response length also exists in open-source PPO code: trl, OpenRLHF, verl, SimpleRL-Zero, Open-Reasoner-Zero (Table 2, Listing 1).
- Proposes Dr. GRPO and shows its advantage equals the RLOO advantage up to a factor G/(G−1) (§3.2, App. A).
- Studies template × question-set coverage (§3.3) and shows that math continual pretraining raises the RL result of Llama-3.2-3B (§3.4).

## Key Figures/Tables to Study
- **Fig. 1 / Fig. 4:** GRPO vs Dr. GRPO objectives and a diagram of how std(R) and |o_i| reweight questions and responses.
- **Fig. 5:** Qwen2.5-1.5B training reward, output length, and benchmark length for correct vs incorrect responses. **Fig. 8:** ablation of each bias term. **Fig. 9:** GRPO vs Dr. GRPO over 3 independent runs.
- **Table 2 + Listing 1:** length-normalized PPO losses in open-source frameworks. **Table 4:** benchmark results at 1.5B, 3B, 7B.

## Technical Details
- **Setting.** Outcome reward R(q,o) = 1 if o contains the correct final answer, else 0, checked with Math-Verify (§3.2). The KL term is dropped: β = 0 throughout (§3, Eq. 1).
- **GRPO (Eq. 3).** J = E[(1/G) Σ_i (1/|o_i|) Σ_t min(ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1−ε, 1+ε) Â_{i,t})], with Â_{i,t} = (R(q,o_i) − mean(R)) / std(R).
  G = responses per question; o_i = i-th response; |o_i| = its token count; ρ_{i,t} = π_θ/π_θold for token t; ε = clip parameter; R = {R(q,o_1), …, R(q,o_G)}.
- **Length bias (§3.1).** For Â > 0 (correct), shorter responses receive larger per-token updates. For Â < 0 (incorrect), longer responses are penalized less per token because |o_i| is larger, so the policy comes to prefer longer incorrect responses.
- **Difficulty bias (§3.1).** Questions whose rewards are almost all 1 or all 0 have small std(R) and receive larger weight; the paper notes that advantage normalization is usually computed over a whole batch, not per question.
- **Dr. GRPO (§3.2, Fig. 1).** Same objective without 1/|o_i| and with Ã_{i,t} = R(q,o_i) − mean(R). In code, `masked_mean` becomes `(tensor * mask).sum(axis=-1) / MAX_TOKENS`, where MAX_TOKENS is the maximum number of generation tokens; "other constants also work with differences in gradient norm" (Listing 1).
- **Why these terms are removed (App. A).** With B(q, o_<t) = mean(R) as baseline, the Monte Carlo policy gradient contains neither std nor |o| (Eq. 6). (G/(G−1))·Ã equals the RLOO advantage.
- **Framework variants (Listing 1).** Per-response normalization (e.g., OpenRLHF) and per-batch token normalization (e.g., trl, verl) are both marked biased.
- **Question sets (§3.3, Table 3).** ORZ 57k, MATH 12k, GSM 8k, ASDiv 2k.
- **Results (Table 4, 3k-token budget).** Oat-Zero-7B: AIME24 43.3, AMC 62.7, MATH500 80.0, Minerva 30.1, OlympiadBench 41.0, average 51.4. Qwen2.5-Math-7B base: average 26.5 (Qwen template), 38.2 (no template). Oat-Zero-1.5B average 42.1. Llama-3.2-3B + Dr. GRPO average 6.8; Llama-3.2-3B-NuminaQA + Dr. GRPO (Oat-Zero-3B) average 20.7.

## Recipe ledger
App. G states that Table 6 was used "in all experiments" and that all experiments ran on 8× A100 GPUs (arXiv:2503.20783v2).

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Oat-Zero-7B (from Qwen2.5-Math-7B) | 7B | RL | algorithm; prompts; template | Dr. GRPO; MATH level 3–5 questions; Qwen-Math template | §1 | verified 2026-09-14 | motivated by §2–3 analyses; no 7B ablation reported |
| Oat-Zero-7B | 7B | RL | compute | 27 hours on 8× A100 | §1 | verified 2026-09-14 | — |
| All RL runs | 1.5B–7B | RL | responses per question; temperature; (top-p, top-k) | 8; 1.0; (1.0, −1) | App. G Table 6 | verified 2026-09-14 | no ablation reported |
| All RL runs | 1.5B–7B | RL | max response length | 3000 tokens | App. G Table 6 | verified 2026-09-14 | no ablation reported; App. B ties the 3k evaluation budget to the 4k Qwen2.5-Math context |
| All Dr. GRPO runs | 1.5B–7B | RL | loss normalizer | constant MAX_TOKENS = maximum generation tokens → 3000 | Listing 1 + App. G Table 6 | derived | Listing 1 defines MAX_TOKENS as the max generation tokens; Table 6 gives 3000 |
| All RL runs | 1.5B–7B | RL | KL loss coef; KL penalty coef | 0.0; 0.0 | App. G Table 6; §3 | verified 2026-09-14 | no ablation in this paper; §3 cites Hu et al. (2025) |
| All RL runs | 1.5B–7B | RL | clip ε; inner proximal update epochs | 0.2; 1 | App. G Table 6 | verified 2026-09-14 | no ablation reported |
| All RL runs | 1.5B–7B | RL | optimizer; wd; grad clip; LR; schedule | AdamW (0.9, 0.95); 0.0; 1.0; 1e-6; constant | App. G Table 6 | verified 2026-09-14 | no ablation reported |
| All RL runs | 1.5B–7B | RL | questions per policy step; total policy steps | not reported | checked §1, §3, App. B, C, G | not reported | — |
| GRPO vs Dr. GRPO comparison | 1.5B | RL | base; template; prompts | Qwen2.5-1.5B; R1 template; MATH train questions | §3.2 | verified 2026-09-14 | Fig. 5 |
| Bias-term ablation | 1.5B | RL | base; prompts; seeds | Qwen2.5-1.5B; 3K mixed ASDiv, MATH, AIME (pre-2023) questions; 3 independent runs in Fig. 9 (seeds for Fig. 8 not stated) | App. C | verified 2026-09-14 | Fig. 8, Fig. 9 |
| Llama-3.2-3B-NuminaQA | 3B | mid-train | continual pretraining | concatenated NuminaMath-1.5 QA text on Llama-3.2-3B-FineMath; 2 epochs; LR 1e-5 | §3.4 | verified 2026-09-14 | Table 4: 20.7 avg after Dr. GRPO vs 14.8 (FineMath) |
| All evaluations | — | eval-gate | decoding; budget | greedy; 3000 tokens (8k also reported for long-context baselines) | §2.2; App. B | verified 2026-09-14 | — |

## Findings relevant to generality, negative feedback, long context
- **Negative feedback (negative as gradient).** For incorrect responses (Â < 0), dividing by |o_i| shrinks the per-token penalty as the response gets longer (§3.1). With Dr. GRPO the length of incorrect responses on evaluation benchmarks is lower than with GRPO (Fig. 5 plot 4; Llama-3.2-3B in Fig. 7 right); the paper gives these as plots, not tables. Removing length normalization has the larger effect on response length; removing either term raised reward and accuracy over vanilla GRPO (App. C, Fig. 8; Result, single study at 1.5B). The paper does not analyse where probability mass removed from incorrect tokens goes.
- **Response length and reasoning.** GRPO on math-pretrained Llama-3.2-3B shows accuracy and length rising together, which the authors attribute partly to the length bias (§3.4, Fig. 7 right; Interpretation). In DeepSeek-R1-Zero the average string length of incorrect responses (8206.1) exceeds that of correct ones (4965.4); the authors attribute this to incorrect responses coming more often from harder questions (App. F, Table 5; Interpretation).
- **Breadth from narrow data.** With the Qwen-Math template on Qwen2.5-Math-1.5B, RL on GSM-8K, simpler and out-of-distribution questions, gave the best final average and nearly doubled test accuracy on harder benchmarks. With the mismatched R1 template, narrow question sets plateaued lower (§3.3, Fig. 6).
- **Attribution of gains.** Templates can remove capability that RL then restores, so the authors advise being "more conservative in claiming the huge gains brought about by pure RL" (§3.3). After Dr. GRPO, Llama-3.2-3B averages 6.8, Llama-3.2-3B-FineMath 14.8, and Llama-3.2-3B-NuminaQA 20.7 (Table 4).
- **Measurement.** pass@8 of the base policy is used as an exploration indicator before RL (§2.1, Fig. 3). Baselines trained for longer contexts are also reported at an 8k budget: R1-Distill-Qwen-7B averages 28.5 at 3k and 54.7 at 8k (Table 4).

## Connections
- [[grpo]]: the objective whose std and 1/|o_i| terms are removed here.
- [[ppo]], [[rloo]]: Dr. GRPO recovers the PPO surrogate with a Monte Carlo return and a mean baseline; its advantage matches RLOO up to G/(G−1) (App. A).
- [[deepseek-r1]]: the R1-Zero model and V3 base analysed in §2.3 and App. F.
- [[r1-zero-analysis]]: older multi-source card whose Dr. GRPO summary overlaps this card.
- [[trl-grpo]], [[verl-grpo]]: `loss_type="dr_grpo"` in TRL and the documented verl Dr. GRPO settings.
- [[entropy-mechanism-llm-rl]], [[rlvr-beyond-base-model]]: other analyses of what RLVR changes in the policy.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2503.20783 (arXiv v2, 6 Oct 2025, COLM 2025 version; PDF text including App. A–H)
- Corrections to the previous card version: "Dr. GRPO loss … − β D_KL(π_θ‖π_ref) with the k3 estimator identical to GRPO" → the paper sets β = 0 and uses no KL term (§3; Table 6 KL coefficients 0.0); "Table 2: Dr. GRPO vs GRPO on Qwen2.5-Math-7B" → Table 2 lists open-source PPO implementations with length bias; "Figure 1: response length curves" → Fig. 1 shows the objectives (left) and incorrect-response length (right), the training curves are Fig. 5; "(1/L_max) inside the objective" → the objective has no length divisor, the constant appears only in the implementation (Fig. 1, Listing 1); authors shortened to first six + et al. per card standard.
- Removed as unsupported by the source: "~30% shorter completions on MATH, AIME, and AMC"; "L_max (e.g., 4096)"; "Hyperparameters same as [[grpo]] except …"; "Section 3: shortest/clearest explanation in the literature"; "Enabled by: DeepSeek-R1 open-source recipe that made length inflation visible".
- Not reported by the source: questions per policy step, total policy steps for Oat-Zero models, and the rollout batch size.
