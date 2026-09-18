<!-- scope: a-m-team AM-DeepSeek-R1-Distilled-1.4M report and dataset card — prompt collection, DeepSeek-R1 response distillation, per-domain verification, and SFT-only students at 32B/72B
     deps: [[deepseek-r1]]
     see-also: [[am-deepseek-r1-0528-distilled]], [[am-reasoning-answers-for-non-reasoning]], [[open-thoughts]], [[openr1]], [[deepseek-r1-distill-synth]]
-->

# 1.4 Million Open-Source Distilled Reasoning Dataset to Empower Large Language Model Training
- **Core Insight:** SFT of Qwen2.5-32B on 1.4M long-CoT entries (0.5M verified open-source responses plus 0.9M new DeepSeek-R1 responses) gives AM-Distill-Qwen-32B an average of 73.1 over AIME2024, MATH-500, GPQA-Diamond, and LiveCodeBench, against 71.6 for DeepSeek-R1-Distill-Qwen-32B (Table 2).
- **Guideline:** When reusing this dataset or its pipeline, treat the gain as a single unablated result, because the report gives no SFT hyperparameters, no ablation of any filtering step, and evaluates only four reasoning benchmarks; 39.2% of entries have neither a reference answer nor test cases (App. A.2, Fig. 4) and are therefore filtered only by reward-model, LLM-score, and rule checks.
- **Authors:** Han Zhao, Haotian Wang, Yiping Peng, Sitong Zhao, Xiaoyu Tian, Shuaiting Chen, et al. (a-m-team)
- **Year:** 2025 (arXiv v1 2025-03; no venue)
- **URL:** https://arxiv.org/abs/2503.19633 ; dataset: https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M
- **Source type:** paper (dataset report) with its model/dataset card
- **Relevant topics:** reasoning distillation, long-CoT SFT data, response verification, difficulty filtering, semantic deduplication

## Abstract
The authors release AM-DeepSeek-R1-Distilled, 1.4M reasoning problems with thinking traces. Problems are collected
from open-source datasets, semantically deduplicated, and cleaned to remove test-set contamination. Responses come
from reasoning models, "predominantly DeepSeek-R1". Math responses are checked against reference answers, code
responses against test cases, and other tasks with a reward model. AM-Distill-Qwen-32B, trained with "only simple
Supervised Fine-Tuning (SFT)", exceeds DeepSeek-R1-Distill-Qwen-32B on AIME2024, MATH-500, GPQA-Diamond, and
LiveCodeBench, and AM-Distill-Qwen-72B exceeds DeepSeek-R1-Distill-Llama-70B on the same four benchmarks (Abstract).

## Key Contributions
- A released 1.4M-entry long-CoT dataset in one format with reference answers, test cases, and source metadata (§1).
- A three-part pipeline: raw data collection, distillation, and rejection sampling (§2, Fig. 2).
- Per-domain response verification: math-verify plus an LLM judge, sandboxed test cases, reward-model scores, and
  format and n-gram repetition rules (§2.2.1–§2.2.3).
- Two SFT-only students, AM-Distill-Qwen-32B and AM-Distill-Qwen-72B (§3.2, Table 2).
- The labeling and judging prompts: difficulty (Table 3), category (Table 4), correctness 1–5 (Table 5) (App. B).

## Key Figures/Tables to Study
- Fig. 2 (pipeline); Table 1 (training system prompt); Table 2 (results).
- App. A Figs. 3–6: token length, reference/test-case coverage, category, and difficulty distributions.
- App. B Tables 3–5: LLM prompts for difficulty, category, and correctness rating.

## Technical Details
- Composition: 0.5M entries with responses taken from open-source datasets; 0.9M entries whose instructions come
  from open-source datasets and whose responses the AM team distilled from DeepSeek-R1, marked "am-0309" (§1).
  The dataset card names the teacher "DeepSeek-R1-671B" (card, "Scale & Composition").
- Top-level categories: math, code, scienceQA, general chat; sources with reference answers or test cases were preferred
  (NuminaMath, MetaMathQA, natural reasoning, OpenCoder, Omni-MATH, PRIME, CodeIO, MATH-lighteval) (§2.1.1).
- Open-source R1-trace sources used: OpenThoughts, OpenR1Math, KodCode, Bespoke-Stratos-17k, GeneralThought,
  Dolphin-R1, data_ablation_full59K, s1K, LIMO; chat data from InfinityInstruct and Orca (§2.1.1).
- Largest instruction sources on the card: natural_reasoning 319,085; InfinityInstruct 306,675; KodCode 210,838;
  Dolphin-R1 63,921; openR1Math_extended 63,290; NuminaMath_1.5 62,446 (card, "Instruction sources"; table
  truncated with "...").
- Response sources on the card: am-0309 900,000; KodCode 210,838; openR1Math_extended 63,290; Dolphin-R1 62,750;
  openR1Math_default 60,839; GeneralThought-Feb25 50,600; openThoughts 31,431; data_ablation_full59K 14,155;
  Bespoke17k 5,747 (card, "Response sources"). The eight non-am-0309 rows sum to 499,650 (derived).
- Category labels: Qwen2.5-7B-Instruct labels finer categories such as creative writing and instruction
  following (§2.1.2). Difficulty: "a large language model" (model not named) scores every instruction, and easy
  and medium examples are downsampled (§2.1.3).
- Resulting distributions: Medium 51.8%, Hard 25.7%, Easy 11.2%, Very Hard 6.5%, Very Easy 4.7% (App. A.4,
  Fig. 6). Math 29.3%, Coding 24.3%, Information Seeking 22.2%, Reasoning 10.4%, Planning 2.3%, Creative Writing
  2.2%, other 9.3% (App. A.3, Fig. 5). These category names differ from the tag list in the Table 4 prompt.
- Deduplication: embeddings per entry, similarity between entries, one representative kept per high-similarity
  group by "priority strategies" (not specified) (§2.1.4). The decontamination method is not described.
- Response routes: an existing response is kept if it passes reference-answer or test-case verification; prompts
  without reasoning chains get new DeepSeek-R1 responses (§2.2).
- Ground truth: math-verify checks format and result, then Qwen2.5-7B-Instruct rates correctness and consistency
  (prompt in Table 5); code runs in a sandbox; failures are removed (§2.2.1).
- Reward: Decision-Tree-Reward-Llama-3.1-8B (five dimensions: correctness, helpfulness, coherence, complexity,
  verbosity) and Qwen2.5-7B-Instruct scoring of the answer part; a threshold "based on the score distribution"
  removes low scores (threshold value not given) (§2.2.2).
- Rules: format `<think>…</think><answer>…</answer>` conformity and excessive consecutive word repetition (§2.2.3).
- Coverage: 38.9% of entries have reference answers, 21.9% test cases, 39.2% neither (App. A.2, Fig. 4).
- Length: most entries are under 4,096 tokens, with the highest concentration near 2,048 tokens (App. A.1, Fig. 3).
- Labels: length and language (Chinese, English, other) (§2.2.4). License: cc-by-nc-4.0 (card metadata).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| AM-Distill-Qwen-32B | 32B | distill-SFT | initial checkpoint | Qwen2.5-32B (base/instruct not stated) | arXiv:2503.19633v1 §3.2 | verified 2026-09-14 | no ablation reported |
| AM-Distill-Qwen-72B | 72B | distill-SFT | initial checkpoint | Qwen2.5-72B (base/instruct not stated) | arXiv:2503.19633v1 §3.2 | verified 2026-09-14 | no ablation reported |
| AM-Distill-Qwen-32B/72B | 32B/72B | distill-SFT | training data | AM-DeepSeek-R1-Distilled, 1.4M entries | Abstract; §1; §5 | verified 2026-09-14 | Table 2 comparison only |
| AM-Distill-Qwen-32B/72B | 32B/72B | distill-SFT | training system prompt | think/answer-tag prompt printed in Table 1 | §3.2, Table 1 | verified 2026-09-14 | no ablation reported |
| AM-Distill-Qwen-32B/72B | 32B/72B | distill-SFT | LR, schedule, batch, epochs, max length, packing, loss masking | not reported | checked §1–§5, App. A–B, dataset card | not reported | — |
| AM-DeepSeek-R1-Distilled (am-0309) | — | distill-SFT data | teacher temperature, top-p, samples per prompt | not reported | checked §2.2, dataset card | not reported | — |
| AM-Distill-Qwen-32B/72B | 32B/72B | eval-gate | decoding for evaluation | max 32,768 tokens; T 0.6; top-p 0.95; 16 samples (AIME 2024), 4 samples (others) | §3.1.2 | verified 2026-09-14 | evaluation setting, not training |

## Findings relevant to generality and distillation
- Results (Table 2): AM-Distill-Qwen-32B vs DeepSeek-R1-Distill-Qwen-32B: AIME2024 72.7 vs 72.6; MATH-500 96.2 vs
  94.3; GPQA-Diamond 64.3 vs 62.1; LiveCodeBench (2024-08–2025-01) 59.1 vs 57.2. AM-Distill-Qwen-72B vs
  DeepSeek-R1-Distill-Llama-70B: 76.5 vs 70.0; 97.0 vs 94.5; 65.9 vs 65.2; 59.7 vs 57.5. Pass@1 is estimated from 16
  (AIME 2024) or 4 samples per query (§3.1.2). The 72B comparison crosses base-model families (Qwen2.5 vs Llama).
- Generality measurement: chat and instruction data are included (§2.1.1) but no chat, instruction-following, or
  knowledge benchmark is reported (§3.1.1).
- Limits stated by the authors: responses "have not been rigorously verified" for factual accuracy; harmful
  instructions and responses were not thoroughly filtered; nested source relationships may make source labels
  inaccurate (§4).
- Negative samples: failed responses are discarded (negative marginal value); none are used as gradient (§2.2).

## Connections
- [[deepseek-r1]] — teacher for the 0.9M new responses; its distilled Qwen/Llama models are the baselines.
- [[am-deepseek-r1-0528-distilled]] — later a-m-team corpus from DeepSeek-R1-0528 with a similar verification list.
- [[am-reasoning-answers-for-non-reasoning]] — same team; selects a subset of this dataset for non-reasoning SFT.
- [[open-thoughts]], [[openr1]], [[kodcode-v1-sft-r1]], [[bespoke-stratos]], [[s1]], [[limo]], [[dolphin]] —
  open-source R1-trace sources whose responses were filtered and reused (§2.1.1).
- [[numina-math]], [[metamath]] — prompt sources with reference answers (§2.1.1).
- [[magpie]] — cited at the LLM difficulty/category scoring step (§1); [[d4]] — cited for deduplication (§1).
- [[deepdistill]], [[distillation-source-matters]], [[am-thinking-v1]] — related distillation-data work.
- [[deepseek-r1-distill-synth]] — library card on R1 traces as student supervision.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2503.19633 (v1, PDF) and the dataset card
  https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M (repository sha 53531c06, last modified
  2025-03-30).
- Audit claims not found in the source: affiliation "Beike" (the paper and card name only a-m-team); "Qwen2.5-32B
  and Qwen2.5-72B base" (the base-checkpoint qualifier is not stated); "prompts are cleaned against test sets" as a
  described procedure (only the goal "to eliminate test set contamination" is stated); "Qwen2.5-7B-based scorer"
  → Qwen2.5-7B-Instruct (§2.2.2).
