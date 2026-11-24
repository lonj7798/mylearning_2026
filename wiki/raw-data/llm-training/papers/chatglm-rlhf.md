<!-- scope: ChatGLM-RLHF (Hou et al., 2024): production RLHF pipeline for ChatGLM — pairwise preference collection, length-debiased and L2-regularized reward model, reward-variance prompt filter, reference-response reward baseline, SFT-loss regularizer against capability forgetting, PPO vs DPO at 6B and 32B
     deps: [[rlhf-instructgpt]], [[bradley-terry-rm]]
     see-also: [[glm-4]], [[ppo]], [[dpo]], [[kl-control-rlhf]], [[reward-model-overoptimization]], [[llama-2]], [[longwriter]]
-->

# ChatGLM-RLHF: Practices of Aligning Large Language Models with Human Feedback
- **Core Insight:** In ChatGLM's production RLHF, subtracting the reward of a pre-generated reference response from the policy response's reward makes the PPO reward start near zero and stabilizes training without reward or advantage normalization, and the resulting ChatGLM-32B-PPO wins 30% vs 15% (55% ties) against ChatGLM-32B-SFT in human evaluation (§3.4, Eq. 5; §4.1; §4.3, Table 4).
- **Guideline:** In a multi-task RLHF run, (1) balance preference pairs by length difference before reward-model training, (2) drop prompts whose K sampled responses have reward variance below a margin, (3) use a reference-response reward baseline, and (4) add a next-token loss on a small task-specific SFT set to keep behaviours such as self-identity and JSON output (§3.3–3.4).
- **Authors:** Zhenyu Hou, Yilin Niu, Zhengxiao Du, Xiaohan Zhang, Xiao Liu, Aohan Zeng, Qinkai Zheng, Minlie Huang, Hongning Wang, Jie Tang, Yuxiao Dong (Zhipu AI, Tsinghua University)
- **Year:** 2024 (arXiv v1 2024-04-01; v2 2024-04-03)
- **URL:** https://arxiv.org/abs/2404.00934
- **Source type:** paper
- **Relevant topics:** RLHF at production scale, reward-model length bias, reward variance and baselines, prompt filtering by reward variance, capability forgetting, PPO vs DPO, PPO parallelism

## Abstract
The paper describes the ChatGLM-RLHF pipeline used for the ChatGLM service: human preference collection, reward-model training, and policy optimization. It reports challenges met in production and the fixes: reducing reward variance for stable large-scale training, model parallelism with fused gradient descent, and regularization against catastrophic forgetting. ChatGLM-RLHF gives on average 15% more wins than ChatGLM-SFT on Chinese alignment tasks.

## Key Contributions
- Four lessons stated in §1: detailed annotation criteria improve consistency; removing reward-model bias reduces spurious correlations; subtracting a baseline reward stabilizes PPO; adding SFT next-token loss reduces capability shifting.
- Bucket-based length balancing and an L2 score regularizer for the reward model (§3.3).
- Reward-variance prompt filter and reference-response reward baseline for PPO (§3.4).
- Two-dimensional (data + tensor) parallelism for PPO, three-dimensional for DPO (§3.4).
- PPO and DPO results at 6B and 32B on AlignBench plus human evaluation (§4).

## Key Figures/Tables to Study
- Fig. 2 (pipeline), Table 2 (preference data statistics), Fig. 3 (reward distributions differ by task), Table 3 (AlignBench), Table 4 (human evaluation), Table 5 (RM accuracy), Table 6 (response length), Fig. 4 (reward and KL with vs without reference reward).

## Technical Details
**Prompts (§3.2).** Intention taxonomy with enough prompts per category; a quality classifier scores intention clarity, semantic clarity and answerability; only clear, answerable prompts are kept; rule filters remove too-short prompts (Table 1 examples).

**Preference data (§3.2, Table 2).** Annotators pick the better of two responses, judging helpfulness (requirement coverage, accuracy, logical consistency, fluency, no unexpected Chinese–English mixing) and safety. Cyclic and tie preferences are removed in post-filtering (§1, Fig. 2). Statistics: 221,866 comparisons; 2.4 turns per dialogue; 314.1 history tokens, 104.1 prompt tokens, 267.7 response tokens on average.

**Reward model (§3.3).** Initialized from ChatGLM-SFT; loss L_RM = −E[log σ(r(x, y_w) − r(x, y_l))] (Eq. 1).
1. Length debiasing: compute d = |len(y_w) − len(y_l)| in tokens; bucket pairs by d; within each bucket, balance the number of pairs where the preferred response is the longer one. Finer buckets remove more bias (§3.3).
2. Stability: add L_REG = r(x, y_w)² + r(x, y_l)² (Eq. 2), a zero-mean Gaussian prior on scores, because predicted score distributions were volatile. The weight on L_REG is not reported.

**Policy training (§3.4).**
1. Prompt filter: sample K responses per prompt; drop the prompt if reward variance < ε (K and ε not reported).
2. KL-penalized reward r = r_φ(x, y) − β log(π_θ/π_0) (Eq. 4).
3. Reference baseline: generate y_ref per prompt in preprocessing; optimize (r_φ(x, y) − r_φ(x, y_ref)) − β·KL (Eq. 5). Motivation: pairwise-trained RM values are not calibrated across prompts and tasks (Fig. 3).
4. Capability forgetting: after RLHF the model answered "an AI assistant" instead of "ChatGLM" and produced invalid JSON. Fix: L = L_r − β·Σ_{(x,y)∈D_S} log π_θ(y|x) on a small human-annotated task-specific set D_S (Eq. 6).
5. DPO variant: the RM ranks sampled responses to build (y_w, y_l) pairs, separating data construction from annotation (§3.4, Eq. 7).
6. Parallelism: generation takes over 80% of PPO training time; pipeline parallelism creates bubbles during generation, so PPO uses data + tensor parallelism only; DPO uses data + tensor + pipeline (§3.4).

**Results (§4).** AlignBench overall, GPT-4-0613 judge, greedy decoding (Table 3): ChatGLM3-6B SFT 4.58, DPO 4.80, PPO 4.90; ChatGLM-32B SFT 6.37, DPO 6.68, PPO 6.75. On 32B, Writing 7.45 → 7.97 (DPO) / 7.92 (PPO) and Math 5.87 → 6.12 / 5.85. Human evaluation, 400 Chinese prompts, 50 per category, two annotators per sample (Table 4): overall PPO win 0.30, tie 0.55, SFT win 0.15; Creative Writing Δ +0.32, Math Δ +0.02, Logic Δ +0.04, Programming Δ +0.16. RM accuracy (Table 5, internal test sets, not shared with Llama 2): ChatGLM-6B train 0.64 / test 0.59; ChatGLM-32B 0.68 / 0.65; Llama2-70B test ~0.64 (from Touvron et al.).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| ChatGLM-32B-2310 RM / ChatGLM3-6B RM | 32B / 6B | reward-model | init; loss; data | ChatGLM-SFT; Bradley–Terry + L2 score term; 221,866 comparisons | arXiv:2404.00934v2 §3.3, Table 2 | verified 2026-11-24 | Table 5 accuracy only |
| ChatGLM-32B-PPO / ChatGLM3-6B-PPO | 32B / 6B | RL | policy and critic LR; iterations | 1e-6 (shared); about 600–800 | §4.1 | verified 2026-11-24 | no ablation reported |
| ChatGLM PPO | 32B / 6B | RL | reward / advantage normalization | none (reference reward used instead) | §4.1; Eq. 5 | verified 2026-11-24 | Fig. 4: w/ reference reward starts near 0 |
| ChatGLM PPO | 32B / 6B | RL | KL β; SFT-loss weight; K; ε; batch size | not reported | checked §3.4, §4.1 | not reported | — |
| ChatGLM PPO | — | infra | compute for 32B PPO | at least 8 or 16 A100 (8×80G) servers for 2–3 days | §1 | verified 2026-11-24 | not applicable |
| ChatGLM PPO | — | eval-gate | sampling top-p at inference | 0.9 | §4.1 | verified 2026-11-24 | — |

## Findings relevant to generality and negative feedback
- **Generality.** Gains concentrate in writing, OpenQA and role-play; math and logic gains are small or negative (Table 3, Table 4). The authors attribute this to human preferences favouring style and format over reasoning (§4.2).
- **Length.** On 32B, DPO raises mean response length 305.6 → 410.9 tokens and PPO to 379.6 (Table 6). The text states that automatic-evaluation gains on creative writing may be partly due to length (§4.5).
- **Forgetting.** RLHF shifted self-identity and JSON validity; the SFT next-token term is the stated fix, but no before/after numbers are reported (§3.4).
- **Negative feedback.** The pipeline states that the policy must see responses of varied quality to receive both positive and negative feedback, which motivates the variance filter (§3.4).

## Connections
- [[glm-4]]: GLM-4 alignment cites this paper for its PPO/DPO practice (GLM-4 §2).
- [[longwriter]]: LongWriter-9B-DPO uses this paper's scoring method and DPO recipe.
- [[rlhf-instructgpt]], [[kl-control-rlhf]], [[ppo]]: base RLHF objective with KL penalty (Eq. 4).
- [[bradley-terry-rm]], [[reward-model-overoptimization]]: RM loss and RM-accuracy context.
- [[llama-2]]: RM accuracy baseline in Table 5 and RFT comparison in §3.4.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2404.00934 (v2, 3 Apr 2024). PDF read in full via text extraction; Tables 3–6 checked in layout-preserving extraction.
- Internal inconsistencies in the source: Table 6 reports ChatGLM-6B PPO at 237.4 tokens, below SFT's 255.3, while §4.5 says "all models produced responses that surpassed the SFT model in length". Table 6 percentages (+32.6%, +22.9%) do not match the 32B token counts (410.9/305.6 = +34.5%; 379.6/305.6 = +24.2%). §4.5 states that the w/o-reference run has a "much lower" KL increase and drifts less from SFT; this is reported as written.
- Not reported by the source: KL β, SFT-loss weight, K and ε for the variance filter, bucket widths, L_REG weight, number of annotators, model architecture details of ChatGLM-32B-2310.
