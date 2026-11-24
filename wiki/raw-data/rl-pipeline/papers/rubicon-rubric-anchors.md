<!-- scope: Rubicon (Ant Group / Inclusion AI): rubric-based RL for non-verifiable tasks on Qwen3-30B-A3B; rubric formalization (criterion, score tiers, weight), non-linear aggregation with veto, offline central-quantile data filter, stage-wise RL against the "seesaw effect", reward-hacking defense rubric, style rubrics that reduce AI-like tone
     deps: [[rubrics-as-rewards]], [[grpo]]
     see-also: [[rlcf-checklists]], [[kimi-k2-recipe]], [[writing-zero]], [[rlmt-think-chat]], [[antislop]], [[subjective-reward-guide]]
-->

# Reinforcement Learning with Rubric Anchors (Rubicon)
- **Core Insight:** A rubric bank of more than 10,000 rubrics (human-, LLM-, and hybrid-written) used as RL reward on about 5K samples raised Qwen3-30B-A3B by +5.21 points average on seven open-ended benchmarks (65.29 → 70.50; Table 1), and a style rubric ("Plain Narrative") moved outputs away from "AI-like" and didactic tone; the human-likeness claim is supported by case studies only, not by a reported metric (§4.2).
- **Guideline:** When a rubric reward covers both constraint-following and open-ended style, train them in separate sequential RL stages (instruction-following first, then creative/empathy rubrics) and add a veto rubric that zeroes the reward for prefatory praise of the user and self-evaluation of the response, because joint training produced the "seesaw effect" (creativity-only training: Collie −6.0, IFEval −5.9; instruction-only training: EQ-Bench3 −2.2; §4.4, Fig. 2) and early runs showed reward spikes from rubric exploitation (§3.2, App. A.1).
- **Authors:** Zenan Huang, Yihong Zhuang, Guoshan Lu, Zeyu Qin, Haokai Xu, et al.; Junbo Zhao (corresponding). Inclusion AI, Ant Group; Zhejiang University.
- **Year:** 2025 (arXiv v1 2025-08-18)
- **URL:** https://arxiv.org/abs/2508.12790 (model: https://huggingface.co/inclusionAI/Rubicon-Preview)
- **Source type:** paper (technical report, self-described as preliminary)
- **Relevant topics:** rubric rewards, non-verifiable RL, style control, AI-tone suppression, reward hacking defense, multi-objective seesaw, data filtering by score quantile

## Summary
Rubicon extends RLVR to open-ended tasks by scoring responses against rubrics with a critic model. A rubric is a set of K dimensions, each with a criterion description, ordered score tiers, and a weight (§2.1, Eq. 1). The reward is a K-dimensional vector (Eq. 2) aggregated into a scalar with a weighted sum as baseline plus veto, saturation, pairwise-interaction, and non-linear shaping functions (§2.2). Data come from a proprietary corpus of 900K+ instances (§3). The base model is Qwen3-30B-A3B; the released model is Rubicon-preview.

## Key Contributions
- Rubric-first workflow: build model-verifiable rubrics first, then curate data that match them, and reuse the same rubrics for reward and evaluation (§2.1).
- Offline filter: the base model generates responses, the critic scores them, and only prompts in a "calibrated central quantile" of the score distribution are kept (too-high = little signal, too-low = noise) (§3.1).
- Stage-wise RL: stage 1 uses verifiable checks and static rubrics for instruction-following; stage 2 uses reference-based and instance-specific rubrics for creative and social tasks (§3.1).
- Adaptive reward-hacking defense: a Reward Hacking Defense Rubric written after inspecting anomalously high-reward rollouts, used as a supervisory constraint in later stages (§3.2).

## Key Figures/Tables to Study
- Table 1 (open-ended benchmarks), Table 2 (general/reasoning retention), Fig. 2 (seesaw effect), §4.2 Plain Narrative rubric text and Conversation Session 1 example, App. A.1 (defense rubric).

## Technical Details
**Main results (Table 1, Qwen3-30B-A3B → Rubicon-preview).** Creative Writing V3 77.82 → 81.89; WritingBench 75.65 → 80.11; Judgemark V2 56.20 → 69.20; EQ-Bench3 73.35 → 79.55; IFEval 83.55 → 81.70 (−1.85); Collie 35.77 → 40.27; IFScale 54.68 → 60.79; average 65.29 → 70.50. DeepSeek-V3-671B average 68.08.

**General ability (Table 2).** AIME24 77.50 → 81.67; AIME25 70.00 → 70.83; GPQA-D 63.00 → 60.35; LCB v5 63.77 → 59.43; reasoning average 73.80 → 73.37; MMLU 79.53 → 79.83; IQuiz-EQ 68.75 → 75.00.

**Style rubric (§4.2).** "Plain Narrative" scores three criteria on 1–5 with JSON output: voice/tone (calm acceptance, grounded realism, understated emotion), content (insight from lived experience, not lecturing), language (short plain sentences, natural speech patterns, textured details, natural rhythm). Key imperatives: "Any hint of artificiality is a significant failure" and avoid clichés. In the example, the base model opens with "I don't experience emotions..." and turns the question back to the user; the trained model writes a first-person narrative.

**Defense rubric (App. A.1).** A deterministic heuristic filter flags two patterns: "prefatory sycophancy (i.e., gratuitous praise of the user's prompt)" and "laudatory self-evaluation (i.e., meta-commentary on the response's own merit)". Flagged responses get a null score regardless of other dimensions.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Rubicon-preview | 30B-A3B | RL | training samples; rubric count | "5K+" samples; ">10,000" rubrics | Abstract; §1 | verified 2026-11-24 | Table 1 gain; no sample-count ablation |
| Rubicon-preview | 30B-A3B | RL | stage order | IF + static rubrics, then creative/social with instance rubrics | §3.1, §4.4 | verified 2026-11-24 | Fig. 2 single-objective runs |
| Rubicon-preview | 30B-A3B | RL | algorithm, lr, KL, group size | not reported | checked §3–4 and appendices | not reported | — |

## Findings relevant to generality and negative feedback
- **Generality.** Open-ended gains came with a −1.85 IFEval drop and −2.65 GPQA-D / −4.34 LCB v5 drops (Tables 1–2); the authors report AIME gains but not these drops in the text.
- **Negative feedback.** The veto rubric is a hard negative signal: any flagged sycophancy or self-praise sets reward to null (App. A.1).
- **Measurement gap.** The authors state that they also used human feedback at scale and that it "was not consistently reflected by standardized benchmarks" (§5 Benchmarks); no human-evaluation numbers are given.

## Connections
- [[rubrics-as-rewards]]: per-prompt rubrics for medicine/science; Rubicon applies rubrics to creative and style tasks.
- [[rlmt-think-chat]]: measures self-praise and engagement-bait rising under a scalar RM, the same patterns Rubicon vetoes.
- [[kimi-k2-recipe]]: self-critique rubric reward with core/prescriptive rubrics.

## Verification
- Read the arXiv v1 PDF (2508.12790v1) in full on 2026-11-24 via pdftotext; numbers taken from Tables 1–2, Fig. 2 labels, §3.2, §4.2, §4.4, §5, App. A.1.
- Not reported: RL algorithm and hyperparameters, critic model identity for training rewards, number of rubric dimensions per prompt, any quantitative human-likeness or AI-detector score.
- Discrepancy: the text says +4.17 on AIME24 (§4.3) and the abstract/§1 say +4.1; Table 2 gives 77.50 → 81.67 (+4.17).
