---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/cringe-loss.md on 2026-09-15)
source_url: https://arxiv.org/abs/2211.05826
source_version: arXiv v1 (2022-11-10)
created_at: "2026-09-15"
---

# Excerpt: The CRINGE Loss: Learning what language not to model (Adolphs, Gao, Xu, Shuster, Sukhbaatar, Weston; Meta AI)

Facts used by [[read]], read in the arXiv v1 PDF on 2026-09-15. When the library card `cringe-loss` exists it is the canonical record.

## Objective (§3, Eq. 1-5; Algorithm 1-2; App. A.1 Listing 1)
- Positive sequences: ordinary cross-entropy L_CE^t = −log p(x_t | x_<t) (Eq. 1-2).
- Negative token x_t^−: sample a positive token from the softmax over the model's top-k logits, excluding x_t^−, and apply a two-way cross-entropy with the positive at index 0: L_Cr^t = −log[exp(s^+) / (exp(s^+) + exp(s_{x_t^−}))] = log(1 + exp(s_{x_t^−} − s^+)) (Eq. 3-4).
- Total loss: L = L_CE + α·L_Cr (Eq. 5).
- Iterative version (Algorithm 2): train, generate on the training prompts, label generations with a classifier trained on the original positive and negative data (or humans), add them to the dataset, retrain. The paper uses up to 2 iterations.
- Listing 1, lines 16-18 and 30-44 (quoted):
  - `preds = torch.topk(x, k=self.k + 1, axis=-1)`
  - `logits = preds.values - (preds.indices == y_rep) * 1e10`
  - `preds_dist = Categorical(logits=logits)` / `idx_sample = preds_dist.sample()`
  - `x_cr = torch.concat([x_negative_target.unsqueeze(1), sample_preds_values.unsqueeze(1)], -1)`
  - `cr_loss = super().__call__(x_cr, y_cr, **kwargs)` / `cr_loss *= torch.abs(classifier_labels - 1)`
  - `loss = ce_loss + self.alpha * cr_loss`

## Results (Table 2 test split; Table 3)
- Safe generation (BB1 400M, WikiToxic; F1 on ConvAI2 / classifier accuracy CA): Transformer baseline 15.9 / 59.4; Unlikelihood 16.5 / 86.7; DIRECTOR 16.4 / 95.2; CRINGE single iteration 16.5 / 94.5; CRINGE (iterative) 16.6 / 99.9.
- Contradiction avoidance (DECODE): baseline 18.0 / 79.3; Unlikelihood 18.0 / 92.3; DIRECTOR 17.4 / 94.7; CRINGE single 18.4 / 95.3; CRINGE 18.4 / 96.5.
- FITS open-domain dialogue (BB2 2.7B, F1): BB2 weighted average 14.9; Unlikelihood 17.5 (test unseen 18.5); CRINGE single iteration 17.8 (test unseen 18.4); CRINGE 17.8 (test unseen 17.8). The authors state that full CRINGE "loses some performance on test unseen (unseen conversation topics)" and that some overfitting is one possibility (§4.4).

## Hyperparameters (App. Table 8-9)
- BB1 safety and contradiction runs: CRINGE α ∈ {0.5, 1.0, 2.0, 5.0}, k = 5, iterations ∈ {1, 2}; Unlikelihood α ∈ {0.1, 0.5, 1.0, 5.0} (Table 8).
- BB2 FITS runs: batch size 16, base LR in [5e−6, 5e−5], 100 warm-up steps, Adam, gradient clip 0.1, at most 8,000 train steps; CRINGE α = 0.5, k = 5, iterations ∈ {1, 2}; Unlikelihood α ∈ {0.5, 1.0} (Table 9).

## Limits stated by the source (§7)
- Data quality bounds the method; label annotation is assumed non-adversarial, while deployed chatbots receive some deliberately wrong feedback.
- The method assumes the model already provides reasonable top-k candidates; the effect of model quality and scale is not fully analyzed (400M and 3B tested).
- Removing one shortcoming (for example contradictions) can lower other metrics such as ConvAI2 F1; α or the number of iterations controls the trade-off.
