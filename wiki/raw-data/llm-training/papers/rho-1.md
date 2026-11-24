<!-- scope: Rho-1 / Selective Language Modeling (Lin, Gou et al., NeurIPS 2024) — token-level data selection in continual pretraining: train only on the top-k% tokens by excess loss (training-model loss minus reference-model loss)
     deps: [[fineweb]], [[doremi]]
     see-also: [[deepseekmath]], [[less]], [[ifd]], [[physics-of-lm-3]], [[rephrasing-the-web]], [[loss-masking-prompt]]
-->

# Rho-1: Not All Tokens Are What You Need
- **Core Insight:** Applying the next-token loss only to the 60% (1.1B) or 70% (7B) of tokens with the highest excess loss L_θ(x) − L_RM(x), where the reference model is trained on 0.5B curated math tokens, raises average few-shot accuracy on 9 math tasks from 21.6 to 38.1 (TinyLlama-1.1B) and from 55.8 to 66.2 (Mistral-7B) after continual pretraining on 15B OpenWebMath tokens (§2.2, §3.1, Table 1).
- **Guideline:** When continual-pretraining toward a target domain with a small high-quality corpus available, train a reference model on that corpus from the same base and mask the loss of low-excess-loss tokens in each batch; select around 60% of tokens for a 1B model (§3.5, Fig. 9). Keep a general loss term or broad reference corpus when general ability matters, since unselected-token loss rises (App. C).
- **Authors:** Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, Ruochen Xu, et al. (Xiamen University, Tsinghua University, Shanghai AI Laboratory, Microsoft; 11 authors, last-listed Weizhu Chen)
- **Year:** 2024 (arXiv v1 2024-04-11; v4 2025-01-08); NeurIPS 2024 per the v4 PDF footer
- **URL:** https://arxiv.org/abs/2404.07965 (project: https://aka.ms/rho)
- **Source type:** paper
- **Relevant topics:** token-level data selection, reference-model scoring, excess loss, continual pretraining, math pretraining, data efficiency, loss masking

## Abstract
Pretraining usually applies the next-token loss to every token. The authors analyse token-level loss trajectories and find distinct patterns: most tokens are already learned, some resist convergence. Rho-1 uses Selective Language Modeling (SLM): a reference model scores every token, and the language model is trained only on tokens with high scores. Continual pretraining on 15B OpenWebMath tokens improves few-shot accuracy by up to 30 points on 9 math tasks; after fine-tuning, Rho-1-1B and 7B reach 40.6% and 51.8% on MATH, matching DeepSeekMath with 3% of its pretraining tokens. Continual pretraining on 80B general tokens gives a 6.8% average improvement across 15 tasks.

## Key Contributions
- Token-loss trajectory analysis on TinyLlama-1B over 15B OpenWebMath tokens: 51% L→L, 26% H→L, 11% H→H, 12% L→H (§2.1, Fig. 3).
- SLM: per-batch top-k% selection by excess loss, with no extra pretraining-time cost beyond the reference-model scoring pass (§2.2).
- Math continual pretraining results at 1B and 7B, with and without tool-integrated fine-tuning (§3.2, Tables 1–2).
- General-domain continual pretraining on 80B tokens (§3.3, Fig. 5).
- Self-reference variant without curated data, using reference loss and next-token entropy as scores (§3.4, Table 3).

## Key Figures/Tables to Study
- Fig. 3: four token categories. Fig. 4: the three-step pipeline.
- Table 1: few-shot CoT math results and token counts. Table 3: self-reference.
- Fig. 6–7: selected vs unselected token loss and its relation to downstream accuracy. Fig. 9: selection ratio.

## Technical Details
**Token categories (§2.1, App. D.1).** Checkpoints every 1B tokens; ~320K-token validation set. Loss trend fitted linearly; ΔL < −0.2 is H→L, ΔL > 0.2 is L→H; the remainder split into H→H / L→L by mean loss.

**Scores and loss (§2.2).**
- Reference loss: L_RM(x_i) = −log P_RM(x_i | x_<i).
- Excess loss: L_Δ(x_i) = L_θ(x_i) − L_RM(x_i), with L_θ the current training model's loss.
- SLM loss: L_SLM(θ) = −(1/(N·k%)) Σ_i I_k%(x_i) · log P(x_i | x_<i; θ), where I_k% = 1 if x_i is in the top k% of L_Δ within the batch, else 0. The full sequence is still fed forward; only the loss is masked.

**Reference models (§3.1).** Math: 0.5B tokens (GPT-generated synthetic data plus manually curated data). General: 1.9B tokens from open sources including Tulu-v2 and OpenHermes-2.5. 3 epochs; max LR 5e-5 (1B) / 1e-5 (7B), cosine; max sequence 2048 (1B) / 4096 (7B), packed. Reference and training model start from the same base.

**Pretraining (§3.1).** Math: OpenWebMath (~14B tokens). General: SlimPajama : StarCoderData : OpenWebMath = 6 : 3 : 1, 80B tokens. TinyLlama-1.1B LR 8e-5 (math) / 1e-4 (general); Mistral-7B LR 2e-5; batch 1M tokens. Hardware: 32 × H100 80G; 1.1B: 15B tokens ≈ 3.5 h, 50B ≈ 12 h; 7B: 15B ≈ 18 h; general 80B ≈ 19 h.

**Results.**
- Table 1 (9-task few-shot average): TinyLlama-CT 21.6 → Rho-1-Math 38.1 (+16.5, 9B selected tokens of 15B); MAWPS +32.1 is the largest single gain; multi-epoch Rho-1-1B reaches 40.9 at 30B. Mistral-CT 55.8 → Rho-1-Math 66.2 (+10.4, 10.5B selected); DeepSeekMath-7B (500B tokens) 68.4.
- Table 2 (after fine-tuning on ToRA-69k, tool-integrated): MATH 40.6 (1B, +2.2 over TinyLlama-CT) and 51.8 (7B, +3.4 over Mistral-CT); DeepSeekMath-7B 52.0.
- Fig. 5: general 80B run, +6.8 average over 15 benchmarks; code and math gains above 10.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Rho-1-Math-1B (TinyLlama-1.1B) | 1.1B | continual pretrain | token selection ratio k | 60% | arXiv:2404.07965v4 §3.1 | verified 2026-11-24 | Fig. 9: ~60% best in a 5B-token sweep on a 1B model |
| Rho-1-Math-7B (Mistral-7B) | 7B | continual pretrain | token selection ratio k | 70% | §3.1 | verified 2026-11-24 | no 7B ablation reported |
| Rho-1-Math-1B / 7B | 1.1B / 7B | continual pretrain | LR; batch; corpus | 8e-5 / 2e-5; 1M tokens; OWM 15B tokens | §3.1 | verified 2026-11-24 | Table 1 vs CLM baselines at same LR |
| Rho-1 general (TinyLlama-1.1B) | 1.1B | continual pretrain | LR; mixture; tokens | 1e-4; SlimPajama:StarCoder:OWM 6:3:1; 80B | §3.1 | verified 2026-11-24 | Fig. 5: +6.8 avg over 15 tasks |
| Reference models | 1.1B / 7B | reference-model | data; epochs; max LR; schedule | 0.5B math or 1.9B general tokens; 3; 5e-5 / 1e-5; cosine | §3.1 | verified 2026-11-24 | §3.4: OWM-only reference still +2.4 (self-reference) |
| Rho-1-Math | 1B / 7B | SFT | fine-tuning data | ToRA-69k (16k GPT-4 tool trajectories + 53k answer-augmented) | §3.2 | verified 2026-11-24 | Table 2: +6.2 / +2.7 avg over CT baselines |

## Findings relevant to generality and negative feedback
- **Generality cost.** With SLM alone, math continual pretraining converges quickly to the reference model's domain and the loss of unselected tokens rises significantly; no adverse effect was observed, but the authors suggest a general text/code loss to prevent overfitting (App. C; Fig. 6).
- **Downstream link.** Selected-token loss tracks downstream accuracy with a power-law-like relation; unselected-token loss moves opposite to accuracy (§3.5, Fig. 7, App. F).
- **Noise handling as implicit negative signal.** No negative gradients are used; noisy or already-learned tokens receive zero loss. H→H and L→L tokens with fluctuating loss were found to be largely noisy on inspection (§2.1, App. D.2).
- **Without curated data.** Self-reference on OWM: L_RM scoring +2.4 average, entropy H_RM similar, intersection of both +3.3 with 40% fewer tokens; on Proof-Pile-2 with an OWM-only reference, +1.8 with 30% fewer tokens (§3.4, Table 3).
- **Scale limit.** Verified only for ≤ 7B models and < 100B tokens; the authors state larger models might not benefit (App. C).
- **Selection drift.** Tokens selected by later checkpoints have higher perplexity late in training and lower early, and selected-token perplexity shows a rise-then-fall pattern (§3.5, Fig. 8).

## Connections
- [[doremi]], [[fineweb]]: document- or domain-level selection with a reference or classifier; Rho-1 moves selection to the token level.
- [[deepseekmath]]: the 500B-token math baseline Rho-1-7B is compared with (Table 1).
- [[less]], [[ifd]]: example-level selection by gradient or difficulty scores, for contrast.
- [[loss-masking-prompt]]: SFT-time loss masking; SLM is a data-driven pretraining-time mask.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2404.07965 (v4 PDF, full text including App. C–D; v1–v3 not compared).
- Table values read from the PDF layout text; columns are GSM8K, MATH, SVAMP, ASDiv, MAWPS, TAB, MQA, MMLU-STEM, SAT, AVG (Table 1).
- Not reported: per-benchmark breakdown of the 15-task general result in the main text (Fig. 5 only); reference-model scoring compute cost; whether selection is per batch or per sequence beyond "ranking the tokens in a batch".
