---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/physics-of-lm-2-2-learning-from-mistakes.md on 2026-09-15)
source_url: https://arxiv.org/abs/2408.16293
source_version: arXiv v1 (2024-08-29)
created_at: "2026-09-15"
---

# Excerpt: Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems (Ye, Xu, Li, Allen-Zhu; CMU, Meta FAIR, MBZUAI)

Facts used by [[read]], read in the arXiv v1 PDF on 2026-09-15.

## Setting (§2; App. D.1)
- iGSM: program-generated grade-school math problems with arithmetic mod 23; "op" is the number of operations in a solution. Four families: iGSM-med_pq/qp and iGSM-hard_pq/qp.
- Model: GPT2-12-12 ("GPT2-small") with rotary position embeddings (App. D).
- Pretraining (App. D.1): AdamW, β = (0.9, 0.98), cosine decay to 0.01× with 1,000 warm-up steps; iGSM-med: LR 0.002, weight decay 0.05, batch 512, context 768, 100,000 steps; iGSM-hard: LR 0.002, weight decay 0.03, batch 256, context 1024, 200,000 steps.

## Retry data (§4)
- At the start of each solution sentence, with probability retry_rate a wrong parameter that cannot be computed next is inserted, followed by a special token [BACK]; the process repeats, so a second error appears with probability retry_rate² (§4).
- Models are compared at the same number of training tokens, so higher retry_rate means fewer problems seen (§4, footnote 6).
- Label masking variant: ignore the loss on the wrong-parameter tokens (§4).

## Results
- Result 2-3 (Fig. 4b): "Within a reasonable range, the more mistakes the better. Especially on hard problems, such as on iGSM-med^{op=23}_{qp}, the accuracy jumps from 78% to 94% by using retry rate = 0.5." "Masking mistakes is unnecessary."
- Result 4 (Fig. 5): at retry_rate = 0.2, without label masking, the model retries on average < 0.3 times even for large op; at retry_rate = 0.5 the average becomes 2-4 retries, which label masking reduces. The authors explain that at retry_rate = 0.2 each step still has a 0.8 chance of being error-free in the data.
- Result 5 (Fig. 6): models pretrained on retry data still output shortest solutions.
- Result 6 (§4): "retry upon regret" (regenerate a step when a probe detects an error) gives 78% ⇒ 80% on iGSM-med^{op=23}_{pq}, while pretraining with retry data gives 78% ⇒ 95%. Beam search with 16 or 32 beams does not noticeably improve error-free-pretrained models (§4, Fig. 3b).
- Error detection is easy: a rank-8 update on the embedding layer of an error-free-pretrained model detects errors with > 99% accuracy (§1, §3).
- Result 7 (§5, Fig. 7, App. B Fig. 10): LoRA fine-tuning an error-free-pretrained model on retry data gives no significant improvement; "for small LoRA ranks, finetuning even hurts and label masking becomes important." Full fine-tuning with enough retry data works but uses twice the training tokens and is "essentially continued pretraining." Rare exception: LoRA can beat the original model at retry_rate = 0.5 with label masking and a very high rank (Fig. 10 caption).
- Result 8 (§6, Fig. 8): "retry_weak" fake mistakes (insert a later solution step followed by [BACK]) "significantly improve the model's accuracy"; "retry_miss" does not improve accuracy by much.

## Limits stated by the source
- Synthetic data only; the authors "do not claim that the synthetic data used here can directly aid in building future LLMs" (§1 Conclusion paragraph).
- The skill is shown for pretraining-stage data; LoRA fine-tuning is the tested post-training setting.
