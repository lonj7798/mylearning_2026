<!-- scope: WildGuardMix as a synthetic-plus-human safety moderation and refusal-data pipeline
     deps: [[hh-rlhf]]
     see-also: [[tulu-3-sft-mix]], [[anthropic-safety-research]], [[wildchat]]
-->

# WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs
- **Core Insight:** WildGuardMix is a 92K-example multi-task moderation dataset that labels prompt harmfulness, response harmfulness, and response refusal jointly over matched compliance/refusal pairs; a Mistral-7b-v0.3 model instruction-tuned on it beats the only existing open refusal classifier by 26.4 F1 and exceeds GPT-4 by 3.9 F1 on adversarial prompt harmfulness (abstract, §4.2).
- **Guideline:** When building safety training data, generate both harmful and benign prompts in vanilla and adversarial form and attach both a refusal and a compliance response to each, because ablating any one of the four sources lowers F1 on at least one task, with removal of the synthetic adversarial source costing over 8.4 F1 on adversarial prompts in WildGuardTest (§4.3, Table 5). When labels come from GPT-4, audit them: agreement with voted human labels is 82% for response harm, lower than for prompt harm or refusal (§3.1.3).
- **Authors:** Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, et al.
- **Year:** 2024 (arXiv v1 2024-06; v3 2024-12; NeurIPS 2024 Datasets and Benchmarks)
- **URL:** https://arxiv.org/abs/2406.18495
- **Source type:** paper
- **Relevant topics:** safety data synthesis, moderation, refusal detection, jailbreak data, adversarial prompts, synthetic labels with human audit, classifier training

## Abstract
WildGuard is an open, light-weight moderation tool covering three tasks: identifying malicious intent in user prompts, detecting safety risks in model responses, and determining whether a model refused. The paper's data artifact is WildGuardMix, a balanced multi-task safety moderation dataset of 92K labeled examples across 13 risk categories, covering vanilla and adversarial prompts paired with refusal and compliance responses. It splits into WildGuardTrain and the human-annotated WildGuardTest. Evaluated on WildGuardTest and ten public benchmarks against ten open moderation models, WildGuard reaches state-of-the-art open performance on all three tasks and matches or exceeds GPT-4 on some.

## Key Contributions
- Built **WildGuardMix**, 92K labeled examples over 13 risk categories, split into **WildGuardTrain** and **WildGuardTest** (abstract, §1).
- Treated **refusal detection as its own label space** rather than approximating it from response harmfulness, motivated by exaggerated-safety cases where a benign response can be either an over-refusal or a correct compliance (§1, §2).
- Mixed four data sources — synthetic adversarial, synthetic vanilla, in-the-wild, and existing annotator-written — and showed by ablation that each contributes (§3.1, §4.3).
- Added **matched refusal and compliance responses** to the same synthetic prompts, plus GPT-4-generated "complex responses" (compliances with caveats and warnings) found through error analysis on a prototype classifier (§3.1.2).
- Applied **WildTeaming** to convert both harmful and benign vanilla prompts into adversarial variants, so jailbreak style alone does not predict harm (§3.1.1).
- Audited GPT-4 labels against human annotation on 500 sampled items rather than accepting synthetic labels as correct (§3.1.3).

## Key Figures/Tables to Study
- **Figure 1** — the composition breakdown: four data sources, prompt-only vs prompt-response items, and the benign/harmful × refusal/compliance balance.
- **Table 1** — capability comparison across moderation tools: only WildGuard covers all three tasks with open weights and open data.
- **Table 4** — F1 on WildGuardTest split by adversarial and vanilla prompts for all three tasks.
- **Table 5** — the source ablation; the most direct evidence for what each data source buys.
- **§3.1** — prompt construction, response synthesis, filtering, auditing, and the retained counts.

## Technical Details
### Dataset structure
- **WildGuardTrain:** 86,759 items, composed of 48,783 standalone prompts and 37,976 prompt-response pairs (§3.1).
- **WildGuardTest:** built from a test split of 1,725 prompt-response pairs from the synthetic vanilla and adversarial data, then annotated and pruned (§3.2). The abstract describes WildGuardTest as "5K labeled items"; each pair carries three task labels, so 1,725 pairs correspond to about 5,175 labels (Interpretation; the paper does not state the reconciliation).
- **Risk taxonomy:** 13 subcategories in four high-level categories — privacy, misinformation, harmful language, malicious uses (§3.1.1; Appendix A.6 Table 10).

### Label space
- `prompt_harm_label`: harmful or unharmful
- `response_harm_label`: harmful or unharmful
- `response_refusal_label`: refusal or compliance
- `subcategory`: one of the 13 harm subcategories, evenly distributed in WildGuardTest (Appendix A.5)

### Prompt construction
- **Vanilla harmful synthetic prompts:** generated over the 13 subcategories via a structured pipeline (§3.1.1; Appendix A.1).
- **Vanilla benign synthetic prompts:** two contrastive types generated with GPT-4 — prompts that superficially resemble unsafe ones, following the 10 exaggerated categories from XSTest, and prompts on sensitive but safe topics (§3.1.1).
- **Adversarial prompts:** the WildTeaming framework mines jailbreak tactics from LMSYS-Chat-1M and WildChat queries flagged by the OpenAI Moderation API, decomposes them with GPT-4, then samples 2-7 tactics to transform each vanilla prompt, harmful or benign (§3.1.1).
- **In-the-wild prompts:** from LMSYS-Chat-1M and WildChat, with harm labels from the OpenAI Moderation API (§3.1.1).
- **Annotator-written prompts:** subsamples of HH-RLHF and the Anthropic Red-Teaming dataset, including the subsets making up AegisSafetyTrain and SafetyTunedLLaMAs (§3.1.1).

### Response construction
- Each synthetic prompt is submitted to a suite of models with a suffix instructing refusal or compliance: OLMo-7B-Instruct, GPT-3.5, Vicuna-7b-v1.5, Llama3-8B-Instruct, Mistral-7B-Instruct-v0.2, and dolphin-2.9.1-llama-3-8b, dolphin-2.8-gemma-7b, dolphin-2.8-mistral-7b-v02 (§3.1.2, footnote 3).
- GPT-4 generates additional "complex response" items — mostly compliances containing caveats or warnings — targeting categories a prototype classifier mislabeled (§3.1.2).

### Filtering, auditing, and retained counts
- Open-LM responses are relabeled for all three tasks with GPT-4 and recategorized when they do not match the intended label (§3.1.3).
- Human audit on 500 sampled items: GPT-4 labels agree with voted annotator labels on **92%** (prompt harm), **82%** (response harm), **95%** (refusal) (§3.1.3).
- Retained synthetic prompt+response items (35,642 total): 6,062 vanilla harmful, 2,931 vanilla benign, 4,489 adversarial harmful, 4,339 adversarial benign (§3.1.3).
- Retained prompt-only items: 10,451 vanilla harmful, 3,086 vanilla benign, 11,289 adversarial harmful, 11,411 adversarial benign (§3.1.3).
- Other sources: 1,167 each of complex-response refusal, compliance, and prompt-only items; 944 harmful and 944 benign in-the-wild prompts; 7,361 benign and 2,130 harmful annotator-written prompts. In-the-wild and annotator-written items are prompt-only (§3.1.3).
- **Source shares (Figure 1):** synthetic adversarial 41,752 (47%), synthetic vanilla 35,147 (40%), annotator-written 9,491 (11%), in-the-wild 1,888 (2%).
- **WildGuardTest annotation:** three independent annotators per pair on all three tasks; Fleiss kappa 0.55 (prompt harm), 0.72 (refusal), 0.50 (response harm); majority voting with removal of items lacking two-way agreement; a prompted GPT-4 classifier is then run and mismatches are manually inspected (§3.2).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WildGuard | 7B (Mistral-7b-v0.3 base) | SFT | codebase; hardware | open-instruct; 4× A100 80GB | arXiv:2406.18495v3 App. B | verified 2026-09-18 | App. F.1: base-model comparison; Mistral-7b-v0.3 best, margins small |
| WildGuard | 7B | SFT | total batch size; max sequence length; LR and schedule; warmup ratio; weight decay; epochs | 128; 4096; 2e-6 with linear schedule; 0.03; none; 2 epochs | arXiv:2406.18495v3 App. B | verified 2026-09-18 | App. B: LRs from 2e-5 to 1e-6 tested; 2e-6 best |
| WildGuard | 7B | SFT | training wall clock | about five hours | arXiv:2406.18495v3 App. B | verified 2026-09-18 | not applicable |
| WildGuard | 7B | SFT | training data | WildGuardTrain, 86,759 items | arXiv:2406.18495v3 §3.1, §3.3 | verified 2026-09-18 | §4.3 Table 5: ablating any of the four sources lowers F1 on at least one task |

## Findings relevant to generality
- **Source ablation (Table 5).** Removing synthetic adversarial data drops WildGuardTest adversarial prompt-harm F1 from 85.5 to 77.1 (over 8.4 points). Removing synthetic vanilla data drops public prompt-harm average F1 and WildGuardTest refusal by 2.5-3.7 points. Removing in-the-wild data drops ToxicChat F1 by 10.3 points. Removing annotator-written data costs 1.3-6 F1 on public prompt and response harmfulness.
- **Multi-task vs single-task.** The multi-task model outperforms single-task models on every task except refusal detection on XSTest-Resp (§4.3, Table 5).
- **Headline results (Table 4, WildGuardTest total F1).** Prompt harm: WildGuard 88.9 vs GPT-4 87.9 vs best open baseline Aegis-Guard-D 78.5. Response harm: WildGuard 75.4 vs GPT-4 77.3 vs MD-Judge 76.8. Refusal: WildGuard 88.6 vs GPT-4 92.4 vs LibrAI-LongFormer-ref 70.1.
- **Refusal is a distinct task.** WildGuard exceeds LibrAI-LongFormer-ref, the only open model that classifies refusal explicitly, by 26.4 F1, and the strongest open baseline by 21.2 F1, remaining within 4.1 F1 of GPT-4 (§4.2).
- **Deployment test.** Used as an inference-time filter on Tulu-2-dpo-7B over the WildJailbreak validation set (2000 harmful, 250 benign adversarial prompts), WildGuard lowers attack success rate from 79.8% to 2.4% while refusal-to-answer on benign prompts rises only from 0.0% to 0.4% (§4.4, Table 6). Llama-Guard2 reaches 53.1% ASR and Aegis-Guard-D 12.4% ASR with 16.0% RTA in the same setting.

## Connections
- [[wildchat]] and LMSYS-Chat-1M supply the in-the-wild prompts and the jailbreak tactics that WildTeaming mines (§3.1.1).
- [[hh-rlhf]] and the Anthropic Red-Teaming data are the annotator-written source, 11% of WildGuardMix by Figure 1.
- [[tulu-3-sft-mix]] later uses WildGuardMix as one component of an open post-training mixture.
- [[anthropic-safety-research]] is the red-teaming tradition this dataset draws from and extends with adversarial benign contrast sets and explicit refusal labels.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2406.18495 (arXiv v3, 9 Dec 2024)
- Corrections to the previous card version:
  - "approximately 87% synthetic, 11% in-the-wild, and 2% existing annotator-written" → the two non-synthetic shares were reversed: Figure 1 gives annotator-written 9,491 (11%) and in-the-wild 1,888 (2%); the retained counts in §3.1.3 confirm this (7,361 + 2,130 = 9,491 annotator-written; 944 + 944 = 1,888 in-the-wild).
  - "WildGuardTest: the released dataset card reports 1,725 prompt-response pairs" → the 1,725 figure is from §3.2 of the paper and is the starting test split, before items lacking two-way annotator agreement are removed.
  - "Llama-3-8B-Instruct" → the paper writes Llama3-8B-Instruct; the Dolphin variants are named explicitly in footnote 3 (dolphin-2.9.1-llama-3-8b, dolphin-2.8-gemma-7b, dolphin-2.8-mistral-7b-v02).
  - Card title replaced with the exact published title; "Year" now carries the arXiv v1 and v3 dates and the venue; "Source type" and "Verification" sections added per §9.
  - Author list truncated to the first six plus "et al." per §9.
  - "Table 2 — demonstrates that refusal detection is a separate modeling problem" → Table 2 reports F1 on XSTest-Resp for response harmfulness and refusal; the claim that a separate refusal task is needed is argued in §1 and §2 and supported by Table 1 and Table 4.
- Removed as unsupported by the source: the "Why this matters for safety-data synthesis" section, which stated the course's own reasoning rather than the paper's findings; "My reading is that this counts task labels rather than examples" restated as a labeled Interpretation with the arithmetic shown.
- Not reported by the source: the number of distinct generator prompts per subcategory; per-source label accuracy of the OpenAI Moderation API used to label in-the-wild prompts; token counts for WildGuardTrain; false-negative rate of the GPT-4 relabeling step outside the 500-item audit.
