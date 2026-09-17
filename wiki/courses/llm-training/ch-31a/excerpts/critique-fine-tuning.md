---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/critique-fine-tuning.md on 2026-09-15)
source_url: https://arxiv.org/abs/2501.17703
source_version: arXiv v4 (2025-03-29); v1 2025-01-29
created_at: "2026-09-15"
---

# Excerpt: Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate (Wang, Yue, Chen; Waterloo, CMU, Vector)

Facts used by [[read]], read in the arXiv v4 PDF on 2026-09-15.

## Objective and data (§2)
- Training loss: argmax_θ log P(c | [x; y]; θ), where x is the question, y a noisy response, and c the critique (§2.3). The response is input, the critique is the target.
- WebInstruct topics: Mathematics 65%, Physics 8%, Chemistry 4%, Business 10%, Humanities 4% (§2.1).
- WebInstruct-SFT: a 50K subset with "a very high error ratio (over 50%)". WebInstruct-verified: top 50K judged correct by GPT-4o-1120. WebInstruct-GPT-4o: same questions with GPT-4o-1120 answers. WebInstruct-CFT: 50K with GPT-4o-1120 critiques; about 56% of responses judged correct, the rest wrong (§2.1).
- Training (§3.1): 1 epoch; LR 5e−6; cosine schedule with warm-up ratio 0.1; global batch 512; checkpoint chosen on MATH-500 validation.

## Results
- Six math benchmarks, average (Table 2). Qwen2.5-Math-7B: base 37.8; WebInstruct-SFT 35.1; verified-SFT 40.4; GPT4o-SFT 50.4; CFT 57.1. Qwen2.5-7B: base 37.4; WebInstruct-SFT 20.2; verified-SFT 38.2; GPT4o-SFT 36.8; CFT 48.6. DeepSeek-Math-7B: base 20.3; best SFT 24.0; CFT 27.5.
- Qwen2.5-Math-7B-CFT (50K) vs Qwen2.5-Math-7B-Instruct (2.5M) on a broader 7-benchmark average including GPQA, TheoremQA, MMLU-Pro: 48.1 vs 44.6 (Table 3).
- Compared with SimpleRL (1152 H100 hours): CFT uses 8 H100 hours, average 50.2 vs 50.9 (SimpleRL) and 48.9 (SimpleRL-Zero) (Table 4). The authors note AIME24 has 30 questions and "the accuracy is heavily impacted by the randomness" (§3.4).
- Instruction following (Table 8): IFEval instruction-level strict / loose and MT-Bench — base 0.266 / 0.291 / 4.79; Qwen2.5-Math-7B-Instruct 0.333 / 0.345 / 5.49; SFT 0.315 / 0.330 / 5.23; verified-SFT 0.328 / 0.341 / 5.41; GPT4o-SFT 0.325 / 0.343 / 5.38; CFT 0.335 / 0.362 / 6.49.
- Teacher (Table 7): GPT-4o-mini critiques 52.0 vs GPT-4o-1120 57.1 vs SFT 40.4.
- Response source (Table 6): self-generated responses 55.1 vs original WebInstruct responses 57.1.
- Combining with high-quality SFT (Table 9): CFT 57.1; mixing 50K CFT + 50K AceMath SFT 54.8; two-stage 51.6; AceMath SFT alone 49.7.
- Token-length control (Table 10): CFT-Short 55.2 vs GPT4o-SFT 50.4.

## Limits stated by the source (§4)
- Manual inspection of 50 GPT-4o-1120 critiques found roughly 20% with inaccuracies (misjudged correct steps, missed errors, imprecise explanations).
- Self-critique at inference "consistently underperformed compared to direct inference"; the final models generate answers directly.
- Number of training seeds: not reported.
