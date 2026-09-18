<!-- scope: ORPO (KAIST AI, Mar 2024) — single-stage preference alignment: causal-LM NLL on the chosen response plus a weighted log-odds-ratio penalty on the rejected response, no reference model
     deps: [[dpo]]
     see-also: [[simpo]], [[kto]], [[ipo]], [[rpo]], [[ultrafeedback]], [[hh-rlhf]], [[hf-dpo-zoo]]
-->

# ORPO: Monolithic Preference Optimization without Reference Model
- **Core Insight:** When OPT-350M is fine-tuned with cross-entropy on chosen HH-RLHF responses only, the log-probability of rejected responses rises together with that of chosen responses (Fig. 3); adding λ·L_OR, a log-sigmoid loss on the odds ratio of chosen to rejected, lets Mistral-7B trained on UltraFeedback alone, without an SFT stage or reference model, reach 12.20% AlpacaEval 2.0, 66.19% IFEval instruction-level loose, and 7.32 MT-Bench (Mistral-ORPO-β; Table 1, Table 6, §6.2).
- **Guideline:** When tuning λ for ORPO, evaluate tasks with deterministic answers separately from open-ended tasks, because on Mistral-7B with UltraFeedback λ = 1.0 scored lower than λ = 0.1 on MT-Bench extraction, math, and reasoning but higher on STEM, humanities, and roleplay (App. E.2, Fig. 10); the paper states that a smaller λ is not always better.
- **Authors:** Jiwoo Hong, Noah Lee, James Thorne (KAIST AI)
- **Year:** 2024 (arXiv v1 2024-03; v2 2024-03-14; arXiv comment "Preprint")
- **URL:** https://arxiv.org/abs/2403.07691
- **Source type:** paper
- **Relevant topics:** reference-free preference optimization, single-stage SFT plus preference loss, odds ratio, rejected-response likelihood, λ weighting

## Abstract
Preference alignment methods usually still need a supervised fine-tuning (SFT) stage to converge. The paper studies the role of SFT in preference alignment and argues that a small penalty on the disfavored generation style is sufficient to make SFT preference-aware. It introduces odds ratio preference optimization (ORPO), a reference-model-free, single-stage ("monolithic") method, and argues empirically and theoretically that the odds ratio is a suitable way to contrast favored and disfavored responses during SFT, at model sizes from 125M to 7B. Fine-tuning Phi-2 (2.7B), Llama-2 (7B), and Mistral (7B) with ORPO on UltraFeedback alone exceeds larger instruction-following models, reaching up to 12.20% on AlpacaEval 2.0, 66.19% on IFEval (instruction-level loose), and 7.32 on MT-Bench. Code and the Mistral-ORPO-α and Mistral-ORPO-β checkpoints are released.

## Key Contributions
- Shows that cross-entropy on chosen responses gives no direct penalty to rejected responses, and that their log-probability rises during SFT (§3, Fig. 3).
- Defines ORPO: L_SFT on the chosen response plus λ·L_OR, with no reference model and no separate SFT stage (§4, Eq. 6-7).
- Derives the gradient of L_OR and argues why the odds ratio is preferred over the probability ratio when preference learning is combined with SFT (§4.3, §7.1, App. A-B).
- Compares ORPO with SFT, PPO, and DPO on OPT 125M-1.3B using a 1.3B reward model, and trains Phi-2, Llama-2-7B, and Mistral-7B (§6).

## Key Figures/Tables to Study
- **Figure 3:** SFT-only training (OPT-350M, HH-RLHF): chosen and rejected log-probabilities both increase.
- **Figure 7:** ORPO with λ = 1.0 on the same model and data: rejected log-probability decreases.
- **Figure 9 / Figure 10 (App. E):** λ ∈ {0.1, 0.5, 1.0} on Mistral-7B: log-probability trends and MT-Bench categories.
- **Tables 1-3:** AlpacaEval results; reward-model win rates against SFT, DPO, PPO.

## Technical Details
- Sequence likelihood is length-normalized (Eq. 3): log P_θ(y|x) = (1/m) Σ_{t=1..m} log P_θ(y_t | x, y_<t). Here x is the prompt, y the response, m its number of tokens, y_t token t.
- Odds (Eq. 4): odds_θ(y|x) = P_θ(y|x) / (1 − P_θ(y|x)). Odds ratio (Eq. 5): OR_θ(y_w, y_l) = odds_θ(y_w|x) / odds_θ(y_l|x), where y_w is the chosen and y_l the rejected response.
- Loss (Eq. 6-7): L_ORPO = E_(x, y_w, y_l) [ L_SFT + λ · L_OR ], with L_OR = −log σ( log OR_θ(y_w, y_l) ). L_SFT is the causal-LM negative log-likelihood of y_w; σ is the logistic sigmoid; λ weights the odds-ratio term.
- Gradient (Eq. 8-10): ∇_θ L_OR = δ(d) · h(d), with δ(d) = [1 + odds_θ(y_w|x) / odds_θ(y_l|x)]^−1 and h(d) = ∇_θ log P_θ(y_w|x) / (1 − P_θ(y_w|x)) − ∇_θ log P_θ(y_l|x) / (1 − P_θ(y_l|x)). App. A (Eq. 20) derives this as the gradient of log σ(log OR), the negative of L_OR in Eq. 7, so h(d) is the direction that raises the chosen and lowers the rejected likelihood.
- The paper states that δ(d) goes to 0 when the chosen odds are much higher than the rejected odds, so updates are larger when the model favors the rejected response (§4.3). It also states that the 1 − P denominators amplify the gradient when P is low (§4.3); as a function of P alone, 1/(1 − P) is largest when P is near 1 (course note).
- Probability ratio PR_θ = P_θ(y_w|x) / P_θ(y_l|x) vs odds ratio (§7.1, Eq. 16-19, Fig. 6): 50,000 pairs X1, X2 ~ Unif(0, 1) are sampled; the Fig. 6 caption states that log OR has a wider range for the same pairs. The authors argue that a PR-based loss discriminates disfavored responses more extremely and over-suppresses their tokens when combined with SFT.
- Efficiency (§7.3): no frozen reference model, so ORPO needs half the forward passes per batch of DPO or RLHF (two instead of four).
- Results: AlpacaEval 1.0 / 2.0 (Table 1): Phi-2 + ORPO 71.80% / 6.35%; Llama-2-7B + ORPO 81.26% / 9.44% (Llama-2-Chat 13B 81.09% / 7.70%); Mistral-ORPO-α 87.92% / 11.33%; Mistral-ORPO-β 91.41% / 12.20% (Zephyr-β 90.60% / 10.99%). IFEval instruction-level loose: α 0.6163, β 0.6619 (App. D, Table 6). MT-Bench: α 7.23 (§6.2, Fig. 4, §8; §1 prints 7.24), β 7.32.
- Reward-model win rate of ORPO (RM-1.3B, 3 rounds, temperature 1.0): OPT-1.3B on HH-RLHF vs SFT 78.0%, vs DPO 70.9%, vs PPO 65.9% (Table 2); on UltraFeedback vs SFT 69.4%, vs DPO 57.8%, vs PPO 65.7% (Table 3). Against DPO at OPT-125M the rate is below 50% (41.7% HH-RLHF, 48.8% UltraFeedback).
- In the setting of 1 epoch SFT then 3 epochs DPO, Llama-2 + SFT and Llama-2 + SFT + DPO "yielded models with outputs that could not be evaluated" on AlpacaEval (§6.1).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Mistral-ORPO-α | 7B | preference (single-stage ORPO) | data | binarized UltraFeedback (Tunstall et al.); model card: HuggingFaceH4/ultrafeedback_binarized | arXiv:2403.07691v2 §5.1, §6.1; HF model card | verified 2026-09-14 | no ablation reported |
| Mistral-ORPO-β | 7B | preference (single-stage ORPO) | data | argilla/ultrafeedback-binarized-preferences-cleaned, same configuration as α, dataset size "similar"; "61k instances" | §6.1, footnote 8; App. G | verified 2026-09-14 | Table 1: β 12.20% vs α 11.33% AlpacaEval 2.0 |
| Mistral-ORPO-α, -β | 7B | preference | λ | 0.1 | §6.1 | verified 2026-09-14 | App. E: λ ∈ {0.1, 0.5, 1.0}; MT-Bench categories (Fig. 10) |
| Llama-2 + ORPO | 7B | preference | λ | 0.2 | §6.1 | verified 2026-09-14 | no ablation reported |
| Phi-2 + ORPO | 2.7B | preference | λ | 0.25 | §6.1 | verified 2026-09-14 | no ablation reported |
| all ORPO runs (OPT 125M-1.3B, Phi-2, Llama-2, Mistral) | 125M-7B | preference | peak LR; schedule | 8e-6; linear warmup then cosine decay (warmup length not printed) | App. C | verified 2026-09-14 | no ablation reported |
| ORPO: OPT series, Phi-2, Llama-2 | 125M-7B | preference | epochs; checkpoint selection | 10 epochs; checkpoint with lowest evaluation loss | App. C | verified 2026-09-14 | no ablation reported |
| Mistral-ORPO-α, -β | 7B | preference | epochs | App. C: "trained for 10 epochs" (not scoped by model) vs Fig. 1 caption: "a single epoch" | App. C; Fig. 1 | conflict | model card and repo README checked: no epoch value; W&B logs not read |
| all runs | 125M-7B | preference | optimizer | AdamW and paged AdamW (assignment per model not printed) | App. C | verified 2026-09-14 | no ablation reported |
| all runs | 125M-7B | preference | sequence length; filtering | UltraFeedback 2,048 tokens, HH-RLHF 1,024 (truncate and pad); prompts >1,024 tokens removed; pairs with y_w = y_l or an empty response removed | App. C; §5.1 | verified 2026-09-14 | no ablation reported |
| all runs | 125M-7B | preference | batch size | not reported | checked §5, App. C, model card, repo README | not reported | — |
| all runs | 125M-7B | preference | compute | 7B: 4 A100 (FSDP); 2.7B: 2 A100 (DeepSpeed ZeRO 2); OPT: 4 A6000 (ZeRO 2); FlashAttention-2 | App. C | verified 2026-09-14 | — |
| SFT baseline (OPT, Phi-2, Llama-2) | 125M-7B | SFT | peak LR; epochs | 1e-5; 1 epoch on chosen responses | App. C; §5.1 | verified 2026-09-14 | follows Ziegler et al. and Rafailov et al. |
| DPO baseline | 125M-7B | preference | β; LR; epochs | β = 0.1; 5e-6; 3 epochs, best by eval loss (usually epoch 1 or 2) | App. C | verified 2026-09-14 | eval loss rose from epoch 3 |
| PPO baseline (OPT) | 125M-1.3B | RL | TRL settings | ppo_epoch 4; init_kl_coef 0.1; horizon 2,000; batch_size 64; mini_batch_size 8; output length 128-512 (UltraFeedback), 64-256 (HH-RLHF); AdamW; LR 1e-05; gamma 0.99; reward model RM-350M | App. C, Table 5; §5.1 | verified 2026-09-14 | no ablation reported |
| RM-350M, RM-1.3B | 350M, 1.3B | reward-model | objective; epochs | Bradley-Terry loss (Eq. 11); 1 epoch per dataset; RM-1.3B used only for evaluation | §5.1 | verified 2026-09-14 | — |

## Findings relevant to negative feedback and generality
- **Type of negative used.** ORPO uses the rejected response as a gradient signal ("negative as gradient"): L_OR lowers log P_θ(y_l|x), while L_SFT keeps a positive NLL term on y_w.
- **Positive-only SFT does not lower rejected likelihood.** OPT-350M trained on chosen HH-RLHF responses: both log-probabilities increase, and rejected responses "sometimes" have higher log-probabilities than chosen ones (§3, Fig. 3).
- **With the odds-ratio term.** Same model and data, λ = 1.0: rejected log-probability decreases while chosen log-probability is similar to Fig. 3 (§7.2, Fig. 7).
- **Effect of λ on both sides (Mistral-7B, UltraFeedback; App. E.1, Fig. 9).** λ = 0.1: rejected log-probabilities do not decrease; chosen increase. λ = 0.5: chosen increase and rejected decrease. λ = 1.0: chosen decrease together with rejected while the margin grows. The authors interpret the λ = 1.0 MT-Bench pattern as over-adaptation to the chosen set (App. E.2).
- **Probability-ratio penalty.** With the same hyperparameters, a probability-ratio loss drives rejected log-probabilities below −4 quickly; with the odds ratio this happens only after overfitting (App. B, Fig. 8). The authors state that an excessive margin can lead to degeneration (§7.1).
- **Diversity (Table 4; lower cosine similarity = more diverse).** Per-input similarity is higher for ORPO than DPO (Phi-2 0.8909 vs 0.8012; Llama-2 0.9008 vs 0.8889), i.e., less diverse samples per prompt. Across-input similarity is lower for ORPO (Phi-2 0.5173 vs 0.6019; Llama-2 0.5091 vs 0.5658).
- **Coverage limits.** Mistral-ORPO-β lacks coding and math skill on MT-Bench, which the authors attribute to training on 61k UltraFeedback instances (App. G). MT-Bench multi-turn scores are obtained without multi-turn training data (§6.2). Not tested above 7B or against other preference algorithms (Limitations).

## Connections
- [[dpo]] — two-stage baseline (SFT then DPO, β = 0.1) compared in §6; uses a frozen reference model.
- [[simpo]] — another reference-free preference loss in this library.
- [[rpo]] — adds an NLL term on the chosen response to DPO; ORPO also keeps an NLL term.
- [[ipo]], [[kto]] — cited in §2 as alternative non-RL alignment objectives.
- [[ultrafeedback]], [[hh-rlhf]] — the two training datasets (§5.1).
- [[hf-dpo-zoo]] — TRL loss-type interface that includes ORPO.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2403.07691 (v2); model card huggingface.co/kaist-ai/mistral-orpo-alpha and github.com/xfactlab/orpo README (no training hyperparameters found in either).
- Corrections to the previous card version:
  - "π_θ(y|x) here is the sequence probability (product of token probs)" → the paper defines log P_θ(y|x) as the per-token average log-likelihood (Eq. 3), so P_θ is length-normalized.
  - "Figure 3: … vs during ORPO (stays flat)" → Fig. 3 shows SFT only; Fig. 7 shows ORPO (λ = 1.0), where rejected log-probability decreases.
  - "Section 3 / Equations 5-7" → odds, odds ratio, and losses are Eq. 4-7 in §4.1-4.2.
  - "Epochs 3-5" → 10 epochs with best-eval-loss selection for OPT, Phi-2, Llama-2 (App. C); Mistral runs are a conflict between App. C (10 epochs) and the Fig. 1 caption (single epoch).
  - "Batch size 64 prompts" → 64 is the PPO baseline batch_size (Table 5); ORPO batch size is not reported.
  - "Max length 1024" → 2,048 for UltraFeedback, 1,024 for HH-RLHF (App. C).
  - "set λ≈0.1-0.25 to avoid over-dominance of the odds-ratio loss" → the paper reports per-model λ values and one ablation on Mistral-7B; it gives no general range (§6.1, App. E).
  - Abstract rewritten to match the paper's abstract, including the 125M-7B size range.
- Removed as unsupported by the source: "the first major single-stage aligned open model"; "Without L_SFT anchoring, a reference-free gap-maximizer would collapse" (the paper argues only that a probability-ratio penalty over-suppresses rejected tokens, §7.1).
- Not reported by the source: batch size; warmup length; where probability mass removed from rejected responses goes (no token-level analysis); preference-label error rates.
