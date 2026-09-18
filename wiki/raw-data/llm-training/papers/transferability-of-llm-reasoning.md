<!-- scope: whether math-only reasoning tuning transfers to other domains, and how RL and SFT differ
     deps: [[deepseek-r1]], [[qwen-3]]
     see-also: [[front-loading-reasoning]], [[prorl]], [[lima]]
-->

# Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning
- **Core Insight:** In a controlled Qwen3-14B-Base study on the same 47K math dataset, GRPO reaches a non-reasoning Transferability Index of +52.2 while SFT on Qwen3-32B-Instruct rejection-sampled thinking traces reaches −104.1, and the SFT model's KL divergence from the backbone is 0.372 on MATH-500 against 0.084 for RL (Table 1, §4.2).
- **Guideline:** When tuning on a single narrow domain and cross-domain retention matters, prefer on-policy RL over off-policy SFT on distilled traces, because the ablation at Qwen3-8B shows on-policy sampling, advantage-based credit assignment, and negative gradients each raise transferability while the KL term has little effect (§5.2, Tables 3–4).
- **Authors:** Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, et al.
- **Year:** 2025 (arXiv v1 2025-07; v2 2025-10)
- **URL:** https://arxiv.org/abs/2507.00432
- **Source type:** paper
- **Relevant topics:** transferability, math reasoning, RL vs SFT, catastrophic forgetting, representation drift, distillation

## Abstract
The paper asks whether rapid math-benchmark gains reflect broader problem-solving ability or narrow overfitting. The authors evaluate over 20 open-weight reasoning-tuned models across math, scientific QA, agent planning, coding, and instruction following, and find that most models that succeed on math do not transfer those gains. They then run a controlled study on Qwen3-14B using math-only data with different tuning methods, and find that RL-tuned models generalize across domains while SFT-tuned models often forget general capabilities. Latent-space PCA and token-space KL analyses show SFT induces substantial representation and output drift while RL preserves general-domain structure. The authors conclude that reliance on SFT-distilled data for reasoning should be reconsidered.

## Key Contributions
- Defines the **Transferability Index (TI)**: per-benchmark gains are z-normalized within a group, passed through a signed square root, weighted by difficulty (wb = 100 − Rb_base), aggregated into a Domain Index, and divided by the math Domain Index (§2.1, Eqs. 1–2).
- Audits **over 20 open-weight reasoning models** on three benchmark groups: math (MATH500, AIME24, AIME25, OlympiadBench), other reasoning (LiveCodeBench, GPQA-Diamond, ACPBench, HeadQA), non-reasoning (CoQA, IFEval, HaluEval, MC-TACO) (§2.1, Table 5).
- Runs a controlled Qwen3-14B-Base study (UniReason models) holding the dataset fixed and varying only the tuning method (§2.2, Table 1).
- Adds two mechanism analyses: layer-wise PCA shift of hidden states (§3, Table 2) and token-distribution KL divergence plus token-rank shift (§4).
- Ablates five training settings at Qwen3-8B to separate sampling distribution, credit assignment, negative gradient, and KL regularization (§5.1–5.2, Tables 3–4).

## Key Figures/Tables to Study
- **Table 1** — UniReason-Qwen3-14B results across the three groups, with TI_other and TI_non.
- **Table 2 / Table 6** — mean PCA shift magnitudes by tuning method and task group.
- **Figure 4** — KL divergence of RL and SFT models against the backbone.
- **Table 3 / Table 4** — the five ablation settings and their scores and TIs at Qwen3-8B.
- **Table 5** — Transferability Index for the 20+ off-the-shelf models.
- **Table 8** — case study of which tokens shift under RL versus SFT.

## Technical Details

### Controlled study setup (§2.2, §A.3)
- Base model: **Qwen3-14B-Base**. Teacher: **Qwen3-32B-Instruct** in thinking mode (and separately non-thinking mode), queried for chain-of-thought traces with **rejection sampling**, keeping only responses whose final answer is correct (§2.2, §A.3.2).
- Training data: **47K** curated math problems — low-difficulty items from DeepScaler, high-difficulty (levels 3–5) items from SimpleRL (§A.3.2).
- The same problems supply SFT targets (teacher CoT) and RL rewards (ground-truth answer correctness), so both paradigms see the same samples (§2.2).
- A second, larger distilled set of **232K** examples from General-Reasoner spanning reasoning and non-reasoning tasks is used for an additional SFT comparison (§A.3.2).

### Main results (Table 1)
| Model | Math avg. | Other-reasoning avg. (TI_other) | Non-reasoning avg. (TI_non) |
|---|---|---|---|
| Qwen3-14B-Base | 27.7 | 30.2 (—) | 45.7 (—) |
| UniReason-Qwen3-14B-think (SFT) | 49.8 | 45.3 (+52.2) | 21.1 (−104.1) |
| UniReason-Qwen3-14B-no-think (SFT) | 32.3 | 45.2 (+165.4) | 29.0 (−278.9) |
| UniReason-Qwen3-14B (RL) | 53.8 | 60.0 (+82.3) | 53.2 (+52.2) |

RL reaches 55.7% on AIME24, 87.8% on MATH500, and 33.8% on OlympiadBench, above the SFT models; it gains 1.8% on GPQA and 17.1% on LiveCodeBench over SFT (§2.2, Table 1).

### Mechanism analyses
- **Latent space (§3, Table 2):** mean PCA shift from Qwen3-14B-Base is 8.5 / 3.5 / 36.9 on math / other-reasoning / non-reasoning for RL, against 19.2 / 6.7 / 38.2 for SFT-think and 21.4 / 10.9 / 113.7 for SFT-no-think.
- **Token space (§4.2):** UniReason-Qwen3-14B-SFT-no-think shows KL divergence of **0.372** on MATH-500 and **0.283** on IFEval from the backbone; the RL model shows **0.084** and **0.019** on the same tasks.
- **Token rank shift (§4.2):** the RL model averages **0.98** positions of shift; the SFT no-think variant averages **10.6**.
- **Shifted-token counts (§4.2, Table 8):** SFT shifts about **390** tokens on a reasoning query and **158** on a non-reasoning query, including query-irrelevant tokens, while RL shifts a small set of task-relevant tokens.

### Ablation (§5.1–5.2, Tables 3–4)
The five settings vary sampling distribution q, credit weight w, and KL coefficient β in a single objective. At Qwen3-8B (base: math 27.6, other 23.6, non-reasoning 33.6):

| Setting | Math | Other | Non | TI_other | TI_non |
|---|---|---|---|---|---|
| Off-policy SFT | 41.9 | 34.4 | 26.6 | 18.3 | −40.5 |
| On-policy SFT | 33.7 | 35.7 | 35.0 | 68.6 | 30.2 |
| Off-policy RL | 45.5 | 35.9 | 31.7 | 36.4 | 4.5 |
| On-policy RL (no KL) | 37.1 | 38.2 | 35.8 | 65.6 | 39.3 |
| On-policy RL | 38.6 | 39.9 | 35.0 | 63.7 | 32.4 |

The authors conclude that on-policy sampling is the dominant factor, that credit assignment and negative gradients add further transferability and increase response length during training, and that KL regularization has a limited effect because on-policy updates already stay near the current policy (§5.2).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| UniReason-Qwen3-14B (RL) | 14B | RL | base model, framework, algorithm | Qwen3-14B-Base, Verl, GRPO, reward = answer correctness | arXiv:2507.00432v2 §A.3.1 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B (RL) | 14B | RL | learning rate | 1 × 10⁻⁶ | §A.3.1 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B (RL) | 14B | RL | train batch size (prompts) | 512 | §A.3.1 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B (RL) | 14B | RL | mini-batch size (samples) | 128 | §A.3.1 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B (RL) | 14B | RL | clipping thresholds | 0.22 to 0.28 | §A.3.1 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B (RL) | 14B | RL | rollouts per prompt | 16 | §A.3.1 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B (RL) | 14B | RL | max response length | 16k tokens | §A.3.1 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B (RL) | 14B | RL | KL and entropy coefficients | 0 (both off) | §A.3.1 | verified 2026-09-18 | §5.2: on-policy RL scores are largely unchanged with or without KL (Table 4) |
| UniReason-Qwen3-14B (RL) | 14B | RL | steps | 140 (checkpoint taken at 140) | §A.3.1 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B-think / -no-think (SFT) | 14B | distill-SFT | framework, targets | LLaMA-Factory; Qwen3-32B-Instruct CoT traces filtered by rejection sampling on final-answer correctness | §A.3.1, §A.3.2, §2.2 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B (SFT) | 14B | distill-SFT | learning rate | 5 × 10⁻⁵ | §A.3.1 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B (SFT) | 14B | distill-SFT | batch size | 512 | §A.3.1 | verified 2026-09-18 | no ablation reported |
| UniReason-Qwen3-14B (SFT) | 14B | distill-SFT | epochs | 1.5 | §A.3.1 | verified 2026-09-18 | stated as chosen "to align with our RL settings" (§A.3.1) |
| Both | 14B | data | training set | 47K math problems (DeepScaler low-difficulty + SimpleRL levels 3–5) | §A.3.2 | verified 2026-09-18 | no ablation reported |
| Off-policy RL ablation | 8B | RL | teacher samples per query | n = 8 from Qwen3-32B think mode; highest-reward response used for the gradient | §5.2 | verified 2026-09-18 | Table 4 |
| All runs | 14B / 8B | — | hardware, GPU hours, seeds, weight decay, warmup | not reported | checked body and Appendices A.1–A.6 | not reported | — |

## Findings relevant to generality, negative feedback, and distillation
- **Generality.** Math-only training can raise or lower non-math performance depending on the method; TI is the paper's proposed measure, and it is a ratio to the math-domain gain, so a model with small math gains can post a large TI (see the +165.4 for SFT-no-think alongside its 32.3 math average).
- **Negative feedback.** The ablation isolates the negative gradient as one of the mechanisms contributing to generalization, alongside credit assignment (§5.2). The paper reports this qualitatively; it does not decompose the size of the negative-gradient contribution numerically.
- **Distillation.** SFT here is distillation from Qwen3-32B-Instruct via rejection sampling. The paper's conclusion names "the reliance on SFT-distilled data for advancing reasoning models" as the practice to rethink (Abstract, §5.2).

## Connections
- [[front-loading-reasoning]] asks when reasoning data should enter training rather than which objective is used.
- [[prorl]] argues prolonged RL widens the reasoning frontier; this paper measures RL's effect on cross-domain retention instead.
- [[lima]] supports small high-quality SFT; this paper reports the forgetting risk of narrow SFT.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2507.00432 (arXiv v2, 20 Oct 2025)
- Corrections to the previous card version:
  - "In controlled Qwen3-14B experiments" → the fine-tuned checkpoint is **Qwen3-14B-Base**, and the ablation study uses **Qwen3-8B-Base** (§2.2, §5.2).
  - "RL-based math tuning transfers better than SFT-based math tuning because RL preserves broader latent structure while SFT can induce harmful drift" — the causal clause is the paper's interpretation of the PCA and KL analyses, not a measured causal result; restated as the measured quantities (§3, §4).
  - Missing **Source type** field added (paper).
  - The card previously contained no numbers. Added: 47K training problems, Qwen3-32B-Instruct teacher with rejection sampling, Table 1 scores and TIs (SFT-think TI_non −104.1 vs RL +52.2), PCA shifts (Table 2), KL 0.372 vs 0.084 (§4.2), token-rank shift 0.98 vs 10.6, the five-setting ablation (Tables 3–4), and the full training configuration (§A.3.1).
- Removed as unsupported by the source:
  - "Practical lesson: if a team claims 'reasoning improved', ask: on which domains? under which tuning method? with what retention..." — course commentary written as a checklist; the paper states no such recommendation.
  - "SFT-tuned math models can forget general capabilities, **likely because** token distributions and internal representations drift too far" — the hedged causal attribution is not made by the paper in this form.
  - "Broad cross-domain audit of reasoning-tuned models, not just math benchmarks" as a standalone contribution — replaced with the audit's actual scope and benchmark list (§2.1).
- Not reported by the source: GPU hours or hardware for the RL and SFT runs, seed counts or variance for Table 1 and Table 4, contamination checks for the evaluation benchmarks, and whether the released UniReason checkpoints correspond to the reported numbers.
