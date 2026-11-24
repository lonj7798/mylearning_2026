<!-- scope: GLM-4.1V-Thinking / GLM-4.5V / GLM-4.6V report (GLM-V Team, 2025–2026), focused on RL: Reinforcement Learning with Curriculum Sampling (RLCS), ratio-EMA dynamic sampling expansion, domain-specific reward system and verifier-failure collapse, removal of KL and entropy losses, cross-domain RL transfer; multimodal architecture and pretraining summarized briefly
     deps: [[grpo]], [[glm-4-5]]
     see-also: [[glm-4]], [[glm-4-5-recipe]], [[slime]], [[dr-grpo]], [[entropy-mechanism-llm-rl]], [[kimi-k1-5]], [[deepseek-r1]], [[webrl]]
-->

# GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning
- **Core Insight:** In multi-domain multimodal GRPO, the reward system for every domain must be precise: with a well-tuned STEM verifier but weak verifiers for other single-image and multi-image tasks, reward hacking in one domain stalled STEM reward after step 150 and reduced MMMU, MathVista and AI2D (§5.2, Fig. 5); with sound verifiers, RL on a single domain transfers to others and mixed-domain RL gives the largest gains on STEM, OCR & Chart and general VQA (§6.3, Fig. 6).
- **Guideline:** Grade prompt difficulty offline (pass@k of prior models + human labels) and online (pass@k of current rollouts), down-sample prompts that are too easy or too hard each iteration (RLCS), oversample rollouts by an EMA of 1/(1 − fraction of all-correct-or-all-wrong groups) so the effective batch stays fixed, and unit-test each domain's verifier before mixing domains (§5.2–5.3.1).
- **Authors:** GLM-V Team, Zhipu AI and Tsinghua University (names sorted alphabetically by first name in §8; core contributors begin Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi)
- **Year:** 2025 (arXiv v1 2025-07-01 titled for GLM-4.1V-Thinking; v6 2026-01-01 adds GLM-4.5V and GLM-4.6V)
- **URL:** https://arxiv.org/abs/2507.01006 (code, models, reward system: https://github.com/zai-org/GLM-V)
- **Source type:** tech report
- **Relevant topics:** RLVR + RLHF across domains, curriculum sampling, dynamic sampling, verifier design, reward hacking, KL and entropy loss removal, top-p in rollouts, per-sample vs per-token loss, cold-start SFT quality, cross-domain RL transfer, RL infrastructure

## Abstract
The report presents GLM-4.1V-Thinking, GLM-4.5V and GLM-4.6V, vision-language models for general multimodal reasoning. A large-scale pre-trained vision foundation model "arguably sets the upper bound" of final performance. Reinforcement Learning with Curriculum Sampling (RLCS) then improves STEM, video, recognition, coding, grounding, GUI agents and long documents. Across 42 public benchmarks, GLM-4.5V is state of the art on nearly all tasks among open models of similar size and is competitive with Gemini-2.5-Flash on coding and GUI agents. GLM-4.1V-9B-Thinking outperforms Qwen2.5-VL-72B on 29 benchmarks.

## Key Contributions
- RLCS: curriculum over difficulty tiers, re-weighted at the granularity of training iterations (§5.3).
- Dynamic sampling expansion via ratio EMA, which fixes the rollout count in advance for parallel sampling (§5.3.1, §5.4).
- Domain-specific reward system with shared verification functions, per-domain modules and unit tests; released in the repository (§5.2, Table 1).
- RL stability lessons: no KL loss, no entropy loss, top-p = 1, per-sample loss, format learned in cold start (§5.3.1–5.3.2).
- Cross-domain RL transfer study on GLM-4.1V-9B-Thinking (§6.3, Fig. 6).

## Key Figures/Tables to Study
- Fig. 5 (collapse caused by weak verifiers), Table 1 (reward design per domain), §5.3.1–5.3.2 bullet lists (training lessons), Fig. 6 (cross-domain transfer grid), Fig. 3 (base-model pass@k on MathVista), Table 2 (42 benchmarks).

## Technical Details
**Model (brief, §2–§4).** ViT encoder initialized from AIMv2-Huge with 3D convolutions (2× temporal downsampling), 2D-RoPE and bicubic-interpolated absolute position embeddings; MLP projector; LLM decoder GLM-4-9B-0414 for GLM-4.1V-Thinking and GLM-4.5-Air for GLM-4.5V (106B-A12B); 3D-RoPE in the LLM. Pre-training: sequence 8,192, global batch 1,536, 120,000 steps; then long-context continual training at 32,768 for 10,000 steps (§3.2). SFT: long-CoT only (short-CoT SFT omitted), sequence 32,768, global batch 32, format `<think>…</think><answer>…</answer>` with the final answer inside `<|begin_of_box|>…<|end_of_box|>` (§4.1–4.2).

**RL data preparation (§5.1).** (1) choose sub-tasks with verifiable answers (e.g., temporal grounding instead of video captioning); (2) convert multiple-choice items with unique answers to fill-in-the-blank to remove guessing noise; (3) correctness checks and pass@k with existing or prior RL models, merged with human difficulty labels; (4) pilot RL per sub-domain.

**Reward system (§5.2, Table 1).** RLVR extracts only the boxed span and compares it with the reference; LLM-based extraction was inaccurate for open-domain answers and for looping or long answers. Examples: math via Sympy with tolerance, else exact match or LLM judge; OCR reward = 1 − edit_distance(ans, gt)/max(|ans|, |gt|); grounding reward = fraction of boxes with IoU > τ; GUI agent = action + IoU. A pilot verifier was hacked after more than 500 iterations by answers such as "a correct number between 0 and 10". Format and style rewards: minimal reward when box markers appear in non-verifiable answers; low reward for extensive Chinese–English mixing or repetition; a text reward model scores instruction compliance and fluency.

**RLCS (§5.3).**
1. Mix domains in proportions set by per-domain pilot runs; optimize the GRPO objective.
2. Offline difficulty: pass@k across the full dataset with several VLMs or earlier RL checkpoints, merged with expert labels into tiers from very easy to very hard.
3. Online difficulty: record each rollout's pass@k, map it to a tier, merge with offline labels.
4. Each iteration, re-weight tier sampling ratios to down-sample too-easy and too-hard prompts. Motivation: in a pilot, over half of prompts exceeded 90% accuracy after 200 steps, and rollouts dominate training time.

**Ratio-EMA expansion (§5.3.1).** `expansion_ratio = 1 / (1 − not_valid_sample_rate)`, where not_valid_sample_rate is last iteration's fraction of all-correct or all-wrong groups; keep `expansion_ratio_ema` and oversample by it, then keep the subset with the most balanced correct/incorrect counts.

**Other effectiveness choices (§5.3.1).** Larger batch for mixed-domain data; force answering (insert `</think>` at the length limit so a truncated reasoning trace still gets a fair reward); discard KL loss (VLM KL rises faster than text-only, and a KL loss "noticeably" constrained capability); clip-higher on the importance ratio.

**Stability (§5.3.2).** Entropy loss produced garbled output and collapse, so it was removed; top-p = 1 in rollouts removed garbled outputs that appeared later with lower top-p (e.g., 0.9); per-sample and per-token loss averaging gave no significant difference in mean reward, per-sample was more stable; learn the output format in cold start, since frequent format errors mixing format and correctness rewards destabilize RL.

**Infrastructure (§5.4).** Balance sequence length and compute across DP ranks after rollout; sequence packing into 32K micro-steps with gradient accumulation weighted by sample count; a re-packing heuristic that "halves our forward-backward time".

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GLM-4.1V-9B-Thinking | 9B | SFT | sequence length; global batch | 32,768; 32 | arXiv:2507.01006v6 §4.2 | verified 2026-11-24 | no ablation reported |
| GLM-V (one experiment, model not named) | — | SFT | cold-start steps | 1,000 vs 2,000: +about 2 points after SFT, same peak after RL | §5.3.1 | verified 2026-11-24 | single comparison, per-benchmark numbers not reported |
| GLM-4.1V / 4.5V | 9B / 106B-A12B | RL | objective; KL loss; entropy loss; rollout top-p; loss averaging | GRPO; none; none; 1; per-sample | §5.3, §5.3.1–5.3.2 | verified 2026-11-24 | qualitative ablations; no numbers reported |
| GLM-4.1V / 4.5V | — | RL | oversampling factor | EMA of 1/(1 − not_valid_sample_rate); EMA decay not reported | §5.3.1 | verified 2026-11-24 (decay not reported) | — |
| GLM-4.1V / 4.5V | — | RL | LR; batch size; group size; clip bounds; RL steps; rollout length limit | not reported | checked §5, App. A–C | not reported | — |
| GLM-4.5V | 106B-A12B | RL | max gain from RL | up to +10.6% | Fig. 1B caption | verified 2026-11-24 | per-benchmark breakdown in Fig. 1B |

## Findings relevant to generality and negative feedback
- **Generality (Fig. 6, GLM-4.1V-9B-Thinking from its SFT checkpoint; gains on STEM / OCR & Chart / Grounding / GUI Agent / General image VQA).** STEM-only RL +3.0 / −0.2 / +10.5 / +3.5 / +1.1. OCR & Chart-only +3.0 / +1.2 / −14.5 / +5.8 / +1.3. Grounding-only −0.5 / −4.6 / +30.4 / +8.1 / −0.4. GUI-only +2.4 / +1.5 / +9.0 / +9.2 / +1.6. Mix-all +6.6 / +5.2 / +30.1 / +6.0 / +3.4. Each single-domain run processes the same number of samples as that domain receives in Mix-all (§6.3).
- **Negative feedback.** All-correct and all-wrong groups give no GRPO gradient once KL and entropy losses are removed; they are filtered by the EMA oversampling (§5.3.1). Truncated reasoning gets an answer via forced `</think>` instead of a zero reward (§5.3.1).
- **Outcome-only reward limit.** The authors report correct answers with incorrect reasoning steps being reinforced because rewards check outcomes only (§7).
- **Domain interference** in RL is reported as smaller than in SFT (§5.3.1).

## Connections
- [[glm-4-5]], [[glm-4-5-recipe]]: the LLM backbone of GLM-4.5V (GLM-4.5-Air) and the text-side RL report; [[slime]] is the RL framework named there; this report calls its RL stack "in-house" without naming it (§5).
- [[grpo]], [[dr-grpo]]: optimization objective and loss-averaging discussion.
- [[entropy-mechanism-llm-rl]]: entropy-loss removal and clip-higher context.
- [[kimi-k1-5]], [[deepseek-r1]]: other reasoning-RL reports with curriculum and sampling choices.
- [[webrl]]: earlier GLM-group curriculum for agent RL (task generation instead of sampling re-weighting).

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2507.01006 (v6, 1 Jan 2026; versions v1 2025-07-01 through v5 2025-08-15 not compared). The v6 PDF was read via text extraction; Fig. 6 values were confirmed in the layout-preserving extraction.
- Version note: v1 was a GLM-4.1V-Thinking report; v6 also covers GLM-4.5V and GLM-4.6V, and some lessons may have been added after v1. This card follows v6 only.
- Not reported by the source: RL learning rate, batch and group sizes, clip values, number of RL steps, EMA decay, domain mixture proportions, and the name of the RL framework.
