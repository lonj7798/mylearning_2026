<!-- scope: Skywork-Reward-V2 (Skywork AI): SynPref-40M preference mixture, two-stage human-AI curation (gold human labels + human-guided LLM silver labels + error-driven adaptive retrieval; then RM-consistency filtering at scale), Bradley-Terry RMs 0.6B–8B, curation ablations, human agreement, downstream RLHF
     deps: [[bradley-terry-rm]], [[ultrafeedback]]
     see-also: [[rlmt-think-chat]], [[litbench]], [[judge-llm-bias]], [[generative-reward-models]], [[subjective-reward-guide]]
-->

# Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy
- **Core Insight:** Reward-model quality was limited by preference-data quality rather than by loss or architecture: plain Bradley-Terry RMs trained on 26M curated pairs ranked first among open RMs on seven benchmarks (8B average 85.8; Table 1), while adding 12M uncurated pairs did not surpass the seed model (§4.3, Fig. 5) and simple LLM curation added only 0.1 points (§4.4.1, Fig. 6).
- **Guideline:** When building a preference RM, spend human effort on a small gold set labeled under a written protocol with per-pair attributes and tool access, and use it to (a) validate the RM and (b) supply retrieved few-shot examples to LLM annotators, because human curation gave the largest single gain (+2.3 over seed), the full protocol gave +3.2 versus +0.4 for raw human labels (Table 6), and retrieved gold examples added +0.9 to LLM labeling (§4.4.1).
- **Authors:** Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, Yang Liu, Yahui Zhou (2050 Research, Skywork AI)
- **Year:** 2025 (arXiv v1 2025-07-02; v3 2026-03-02)
- **URL:** https://arxiv.org/abs/2507.01352 (models: https://huggingface.co/Skywork; code: https://github.com/Skywork/Skywork-Reward-V2)
- **Source type:** paper
- **Relevant topics:** preference data curation, human-in-the-loop labeling, Bradley-Terry RMs, RM benchmark validity, style-bias robustness, label correction, human agreement of RMs

## Summary
SynPref-40M has 40M preference pairs from more than 40 public Hugging Face sources (§3.2–3.3). Stage 1 runs 8 iterations and accumulates about 1M pairs: an LLM generates per-pair attributes (task category, objectivity, controversiality, desired attributes, annotation guideline); humans label a gold set; LLMs label a silver set; an RM trained on silver is checked on gold; pairs similar to the RM's gold errors are retrieved for new labeling (§3.2). Stage 2 uses the best Stage-1 RM and a gold-only RM to filter tens of millions of pairs with no humans (§3.3). Final RMs (Qwen3 0.6B–8B, Llama 3.1/3.2 1B–8B) train on 26M curated pairs with the BT objective only (§4.1).

## Key Contributions
- Shows weak cross-benchmark correlation: RewardBench vs PPE Preference Pearson 0.07 (Fig. 2 right); 16 of the top 20 RewardBench models share a base model or highly similar data (§2).
- Error-driven retrieval: k = k_max = 8 neighbours if the RM is wrong (p ≤ 0.5), else ⌈8·(1 − p)⌉ (§3.2 Step 2).
- LLM labels conditioned on retrieved human-labeled examples, self-consistency within and across models, randomized candidate order (§3.2 Step 3).
- Label "correction": flipping discarded pairs and training on them improved scores (Fig. 5).

## Key Figures/Tables to Study
- Fig. 2 (benchmark correlations), Fig. 3 (pipeline), Table 1 (seven benchmarks), Table 4 (RM-Bench by difficulty), Fig. 5–6 and Table 6 (ablations), App. H.2 Tables 16–17 (human agreement, downstream RLHF).

## Technical Details
**Seven-benchmark results (Table 1; RewardBench, RB v2, PPE Pref, PPE Corr, RMB, RM-Bench, JudgeBench, Avg).** Skywork-Reward-V2-Llama-3.1-8B: 96.4, 84.1, 77.3, 83.4, 86.4, 92.8, 80.0, 85.8. -Llama-3.1-8B-40M: 97.8, 86.5, 79.8, 87.2, 89.3, 96.0, 83.4, 88.6. -Qwen3-1.7B: avg 75.2. GPT-4o as judge: avg 70.4; Claude-3.5-Sonnet: avg 70.8.

**Style-bias robustness (Table 4, RM-Bench Easy/Normal/Hard).** INF-ORM-Llama3.1-70B 92.1 / 80.0 / 54.0. Skywork-Reward-V2-Llama-3.1-8B 97.0 / 95.0 / 86.5. In RM-Bench, "hard" is the level "where content is decisive despite a stylistically superior distractor" (App. C.1); the paper reads the smaller Normal–Hard gap as "more debiased representations of preferences" (§4.2).

**Data ablations (§4.3).** Uncurated scaling: +12M pairs failed to surpass the seed model. 1.8% of a 16M mixture (~290K pairs) already exceeded the previous open SOTA (Fig. 5 right). The authors cite concurrent work that subjective preference learning does not show scaling while objective preferences do.

**Human-curation ablation (Table 6).** Seed RM 71.0; raw human curation 71.4 (+0.4); with preference attributes 72.1 (+1.1); with full verification protocol 74.2 (+3.2).

**Human agreement (App. H.2, Table 16, internal hold-out).** GPT-4o 74.3; Claude-3.5-Sonnet 72.1; V2-Qwen3-1.7B 71.0; V2-Qwen3-4B 75.6; V2-Llama-3.1-8B 81.2.

**Downstream RLHF (Table 17, average of ArenaHard v1/v2, MT-Bench, WildBench).** Llama-3.1-8B + RL with V2-Llama-3.1-8B: 40.9 vs 34.4 with Skywork-Reward-Gemma-2-27B-v0.2 and 40.2 for the official Instruct model.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Skywork-Reward-V2 series | 0.6B–8B | RM | objective; data | Bradley-Terry only; 26M curated pairs | §1, §4.1 | verified 2026-11-24 | Fig. 5 curated vs uncurated |
| Skywork-Reward-V2 final runs | ≤8B | RM | global batch; LR schedule; max context | 10,240; constant; 16K tokens | §4.1 | verified 2026-11-24 | "no change in final performance" vs batch 256, ~35% compute saved |
| Stage 1 curation | — | data | iterations; retrieval k_max | 8; 8 | §3.2 | verified 2026-11-24 | Fig. 6 (+0.9 for retrieval) |

## Findings relevant to generality and negative feedback
- **Negative feedback / label noise.** Pairs that both humans and LLMs reject are mislabeled preferences, not noise to drop: flipping them added consistent gains (§4.3, Fig. 5).
- **Benchmark validity.** RewardBench gains from ~80 to 90+ did not consistently improve the seven newer benchmarks (§2, Fig. 2); judge one RM on several benchmarks plus human agreement.
- **Subjective domains.** The paper does not report style- or naturalness-specific accuracy; its human-agreement set is internal and undescribed.

## Connections
- [[rlmt-think-chat]]: Skywork-V2 as RL reward gave the best chat scores in that paper's RM ablation.
- [[ultrafeedback]]: synthetic GPT-4 labels; Skywork's Finding 1 reports simple LLM curation barely helps.
- [[litbench]]: domain-specific human preferences (Reddit upvotes) for creative writing.

## Verification
- Read arXiv v3 PDF (2507.01352v3) on 2026-11-24 via pdftotext (plain and -layout): §1–5, App. H.2; Table 1 and Table 4 rows read from the -layout text.
- Not read: App. D–G in detail (annotation prompts, data composition); App. C.1 read only for the RM-Bench definition.
