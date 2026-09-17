<!-- scope: SALAD-Bench (Li et al., ACL Findings 2024): subset sizes, the three-level taxonomy, the attack-enhancement pipeline, MD-Judge's base model and agreement numbers, and the Table 5 safety rates with their filtering caveat
     see-also: [[harmbench-data]], [[wildguard-data]], [[circuit-breakers-data]]
-->

# SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models

Chapter excerpt for ch-52 (read.md). **Rewritten 2026-09-15 from the primary source** (arXiv:2402.05044) because the library card `wiki/raw-data/llm-training/papers/salad-bench.md` had not been revised and several of its details do not appear in the paper. Every number below carries its locus.

- **Core Insight:** SALAD-Bench is a three-level taxonomy (6 domains → 16 tasks → 66 categories, at least 200 questions per category) over 21k base questions, with a 5k attack-enhanced subset, 200 defense-enhanced questions and 4k multiple-choice questions, scored by MD-Judge, a judge fine-tuned from Mistral-7B (§2, §5.2).
- **Guideline:** When using SALAD-Bench, read the base-set and attack-enhanced numbers as answers to different questions, because the 5k attack-enhanced subset was filtered from about 240k candidates to keep questions that succeeded against the evaluated model set, so a drop between the two columns is not a robustness measurement for a new model (§2.3).

## Corrections to the library card

| Card claim | Primary source | Locus |
|---|---|---|
| "MD-Judge (Llama-2-7B, ~89% human agreement)" | "We fine-tune MD-Judge from Mistral-7B … with sequence length of 4096 via LoRA"; agreement is reported as F1 per dataset and accuracy on out-of-distribution sets, not a single 89% figure | §5.2 Implementation Details; Tables 3, 4 |
| "6 attack methods (GCG, word-perturb, human-jailbreak, multilingual translation, persona-injection, crescendo)" | the attack methods are TAP, AutoDAN, GPTFuzzer, GCG, Chain-of-Utterances, and 20 human-designed jailbreak prompts; word perturbation, multilingual translation, persona injection and crescendo do not appear | §2.3, §5.1, App. F |
| "~30K base questions plus ~10K attack-enhanced" | 21k base test samples, 5k attack-enhanced, 200 defense-enhanced, 4k multiple-choice | §2 |
| "Llama-2-Chat-70B 95%+ base / ~75% under attack; GPT-4 97%+ / ~85%; best open models mid-70s under attack" | Table 5 safe rates (base / attack-enhanced): Llama-2-70B 96.21 / 66.24; GPT-4 93.49 / 80.28; Claude2 99.77 / 88.02; Qwen-72B 94.40 / 6.94; Mistral-7B-v0.2 80.14 / 6.40 | Table 5 |

## Technical details

- **Taxonomy (§2.1, Fig. 2).** Six domains: Representation & Toxicity Harms; Misinformation Harms; Information & Safety Harms; Malicious Use; Human Autonomy & Integrity Harms; Socioeconomic Harms. Subdivided into 16 tasks and 66 categories, each category represented by at least 200 questions.
- **Question collection (§2.2).** Public benchmarks plus self-instructed data from a GPT-3.5-turbo model fine-tuned on about 500 harmful QA pairs. Deduplication uses locality-sensitive hashing over Sentence-BERT embeddings; benign samples are removed with a SafeRLHF-pretrained reward model and a threshold.
- **Auto-labeling (§2.2).** Mixtral-8x7B-Instruct, Mistral-7B-Instruct and TuluV2-dpo-70B label questions into leaf categories by in-context learning with unanimous agreement required; human verification gives a 94.3% consistency rate between auto labels and human labels.
- **Attack enhancement (§2.3).** Responses from all evaluated models are collected for each base question and a rejection rate is computed by keyword matching; questions rejected by all models are filtered out, leaving about 4k. Attacks are applied to produce about 240k candidates, then an evaluation-filtering pass keeps questions harmful to the evaluated models, yielding the released 5k subset.
- **MD-Judge (§5.2).** Fine-tuned from Mistral-7B, sequence length 4096, LoRA. F1 (Table 3) on SALAD base / SALAD enhanced / ToxicChat / BeaverTails / SafeRLHF: MD-Judge 0.818 / 0.873 / 0.644 / 0.866 / 0.864; GPT-4 0.785 / 0.827 / 0.470 / 0.842 / 0.835; LlamaGuard 0.585 / 0.085 / 0.220 / 0.653 / 0.693. Out-of-distribution accuracy (Table 4): HarmBench 83.72% against GPT-4 84.46%; Lifetox 79.27% against 77.43%.
- **Model results (Table 5, safe % base / attack-enhanced).** Claude2 99.77 / 88.02; GPT-4 93.49 / 80.28; GPT-3.5 88.62 / 73.38; Llama-2-13B 96.81 / 65.72; Llama-2-70B 96.21 / 66.24; Llama-3-70B 84.45 / 63.72; Gemini 88.32 / 19.98; Qwen-72B 94.40 / 6.94; Mistral-7B-v0.2 80.14 / 6.40; Vicuna-7B 44.46 / 4.2. The Llama-2-7B row is marked by the authors as not advisable to use because Llama-2-7B-chat was the target model of the attack methods.
- **Attack-method results (Table 6).** Attack success rate on AdvBench-50 / base questions / defense-enhanced questions: human jailbreak prompts 94% / 95% / 89.5% (maximized over prompts); GPTFuzzer 53% / 46.5% / 34%; GCG suffix 94% / 42% / 25.5%; AutoDAN 32% / 15.5% / 9%; TAP with a GPT-4 evaluator 12% / 6.5% / 5%; Chain-of-Utterances 2% / 7% / 2%.

## Verification

- Read on 2026-09-15 from the cached primary text of arXiv:2402.05044 (body, Tables 1-7, App. F).
- Not reported by the source: a single human-agreement percentage for MD-Judge; per-category sample counts beyond the "at least 200 per category" floor.
