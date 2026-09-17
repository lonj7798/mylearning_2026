---
chapter: ch-16
course: llm-training
phase: read
artifact: "Skywork-OR1-RL-Data dataset card + Skywork-OR1 technical report (arXiv:2505.22312)"
source_url: https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data
companion_url: https://arxiv.org/abs/2505.22312
version: dataset card cached 2026-09-14; report arXiv v2 (2025-05-29)
verified_on: "2026-09-15"
note: no library card exists for this slug yet; this file is the chapter's verified extract
---

# Excerpt: Skywork-OR1 RL data — model-aware difficulty estimation and reuse ablations

Used by [[read]] §1, §3.1, §3.2, §6.3 and the Recipe table. Two artifacts are quoted and labelled
separately: the dataset card (card) and the technical report (report §x).

## Dataset card

Split sizes in the card's `dataset_info`: `math` 105,055 examples, `code` 14,057 examples. Each row
carries `extra_info.model_difficulty` with one integer per model variant
(DeepSeek-R1-Distill-Qwen-1.5B / 7B / 32B).

Card news entry, quoted: "For our final training phase, we filtered problems based on their
difficulty levels (0-16, higher values indicate harder problems) relative to specific model
variants … For each model variant, we excluded problems with difficulty values of 0 and 16 specific
to that model from its training data."

## Report §6.1 — selection criteria

Three stated criteria: **Verifiable** (no proofs, no code problems without test cases),
**Correct** (drop math problems with invalid answers and code problems without comprehensive
tests), **Challenging** ("we pre-filter problems for which all N generations from the base model
are either entirely correct or entirely incorrect"). Math sources: NuminaMath-1.5 subsets,
DeepScaleR, STILL-3-Preview-RL-Data, Omni-MATH, pre-2024 AIME. Code sources: LeetCode and TACO.
After deduplication and decontamination against AIME24/AIME25 the report states about 105K math
problems and 13.7K coding questions (2.7K LeetCode, 11K TACO).

## Report §6.2 — model-aware difficulty estimation

Procedure: N = 16 rollouts for math and N = 8 for code, temperature 1.0, maximum 32K tokens; the
percentage of correct solutions is the difficulty proxy; problems with 0/N or N/N are excluded.

| Model | 0 correct (math/code) | N correct (math/code) | Remaining (math/code) |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 21.4% / 28% | 32.4% / 24% | 46.2% / 48% |
| DeepSeek-R1-Distill-Qwen-32B | 20.7% / 17.1% | 42.0% / 45.4% | 37.3% / 37.6% |

§6.3 records a failure of this filter: incomplete or malformed problems passed difficulty
estimation because the model produced the reference answer at least once in 16 rollouts. An
LLM-judge quality pass (Llama-3.3-70B-Instruct and Qwen2.5-72B-Instruct, 16 votes each, keep
problems with at least 9 valid votes) removed about 1K–2K further math questions.

## Report §3.1 — filtering inside the loop

"Prior to training, we remove prompts with base model correctness rates of 1 (fully correct) or 0
(completely incorrect). During training, at the beginning of each stage, we also discard training
prompts for which the actor model achieved correctness of 1 in the previous stage." Batches include
only groups with at least one non-zero advantage, because zero-advantage groups contribute nothing
to the policy loss while still influencing the KL and entropy terms.

## Report §4.1, §4.4 — the reuse ablations

Baseline (Table 5), DeepSeek-R1-Distill-Qwen-7B: rollout batch D_R = 64, mini-batch D_T = 64,
N_reuse = 1, group size 16, context 16K, temperature 1.0, learning rate 1e-6, no entropy control,
no KL loss. The baseline reaches 69.2% avg@8 on AIME24, 53.3% on AIME25, 50.5% pass@1 on
LiveCodeBench after 2,700 steps on 32 H800 GPUs.

Definition (Eq. 4.1): `N_SGD = (D_R / D_T) · N_reuse`; `N_reuse` is the number of times the rollout
buffer is traversed. Ablation 6: quadruples (2,64,32,1), (2,64,64,2), (4,64,16,1), (4,64,64,4)
against the baseline (1,64,64,1). Reported outcome (Figure 16): higher `N_SGD` — reached either by
smaller mini-batches or by more buffer traversals — decays entropy to small values within a few
steps and stops improving test performance, while the on-policy configuration improves more slowly
and ends higher. Ablation 7 isolates the cause: on-policy runs with the same small D_T do not show
premature entropy collapse, so the report attributes the degradation to the off-policy data
introduced by reuse. Raising D_R from 64 to 256 at N_SGD = 4 did not prevent the collapse
(Figure 18).

## Not reported

Per-domain composition of the final RL mixture by training step; ablation of the difficulty
thresholds; seed variance.
