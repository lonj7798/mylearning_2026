<!-- scope: Antislop (Paech, Roush, Goldfeder, Shwartz-Ziv): slop fingerprinting by LLM/human n-gram frequency ratios; Antislop Sampler (backtracking soft-ban of strings and regexes); Final Token Preference Optimization (FTPO) on single-token preference pairs with logit-space MSE tethers; FTPO vs DPO vs token banning on suppression, writing quality, lexical diversity, MMLU/GSM8K
     deps: [[dpo]]
     see-also: [[human-like-responses-dpo]], [[rubicon-rubric-anchors]], [[persona-lexical-diversity]], [[open-character-training]], [[subjective-reward-guide]]
-->

# Antislop: Identifying and Eliminating Repetitive Patterns in Language Models
- **Core Insight:** "AI-like" text can be measured as phrase over-representation against a human corpus (gemma-3-12b uses "elara" 85,513× and "heart hammered ribs" 1,192× more often than human text; Table 1), and suppressing those phrases at the single-token level with FTPO kept writing quality within about 1% of baseline at 83–92% suppression, while DPO on the same pairs lost 6–15 writing-quality points and up to 26% lexical diversity at 80–82% suppression (§6.2–6.3, Table 2).
- **Guideline:** To remove model-specific AI phrasing, build a banlist from frequency ratios against a human baseline, generate preference pairs only at the token where a banned pattern begins (rejected = first banned token, chosen = coherent min-p alternatives), and train with a loss that stops once the margin is reached and anchors non-target logits, because DPO on these pairs degraded quality sharply beyond about 40% preference accuracy while FTPO reached nearly 100% (§6.4, Fig. 4a).
- **Authors:** Samuel J. Paech (independent), Allen G. Roush (Thoughtworks), Judah Goldfeder (Columbia), Ravid Shwartz-Ziv (NYU)
- **Year:** 2025 (arXiv v1 2025-10-16; v2 2025-10-21)
- **URL:** https://arxiv.org/abs/2510.15061 (code: https://github.com/sam-paech/auto-antislop, MIT)
- **Source type:** paper
- **Relevant topics:** AI-tone suppression, lexical over-representation, constrained decoding, token-level preference optimization, DPO diversity collapse, lexical diversity metrics

## Summary
For each model, 2,000 outputs on Reddit creative-writing prompts are compared to a human baseline (wordfreq for words; Reddit creative writing + Project Gutenberg for n-grams, stopwords removed) with the ratio ρ(p) = f_LLM(p) / f_human(p) (§3.1). Slop fingerprints cluster within model families (§3.2, App. K). The Antislop Sampler scans the trace after each token; on a banned word, phrase, or regex it backtracks to the first token, multiplies its probability by 10^(−10s) for ban strength s ∈ [0, 1], and resamples with min-p filtering; a token that resamples anyway is allowed ("soft-banning") (§4.1–4.2). Each backtracking event yields an FTPO pair (§5.4).

## Key Contributions
- Measures sentence-level AI constructions too: "It's not X, it's Y" is 6.3× more frequent than in human writing in some models (§3.2, Fig. 7); the sampler suppressed 100% of it with a regex ban (§6.5).
- FTPO loss: margin preference loss softplus(m − Δ_c) weighted by w_c = clamp((m − Δ_c)/m, 0, 1), plus MSE tether on target logits outside a free window τ_target, plus strong MSE tether on all non-target logits (§5.2).
- Logit-space MSE instead of KL, so large changes to chosen/rejected logits do not push the rest of the vocabulary (§5.3).

## Key Figures/Tables to Study
- Table 1 (over-representation ratios), Fig. 3 (suppression vs writing quality), Table 2 (FTPO vs DPO), Fig. 4 (overtraining and logit divergence), Fig. 5 (adversarial prompts that request banned words).

## Technical Details
**Setup (§6.1).** gemma-3-12b main model; all layers frozen except the last 5 and lm_head; LoRA r = 256; 1 epoch with early stop at preference accuracy 0.85; DPO β = 0.1. Diversity = length-controlled aggregate of MATTR-500, Root-TTR, HD-D, Distinct-1/2/3, normalized to baseline 100. Writing quality = GPT-5-judged rubric; longform = Sonnet-4-judged ~30K-token multi-turn story writing.

**Table 2 (MMLU, GSM8K, longform, writing quality, diversity, ban %).**
| Run | MMLU | GSM8K | Longform | Writing | Diversity | Ban % |
|---|---|---|---|---|---|---|
| gemma-3-12b baseline | 0.590 | 0.888 | 51.3 | 67.80 | 100.00 | 0.00 |
| FTPO 2k / 4k / 8k | 0.559 / 0.565 / 0.592 | 0.876 / 0.880 / 0.889 | 47.5 / 49.4 / 52.3 | 68.93 / 67.31 / 67.49 | 101.05 / 97.68 / 95.09 | 92.39 / 90.15 / 83.40 |
| DPO 2k / 4k / 8k | 0.541 / 0.549 / 0.571 | 0.847 / 0.861 / 0.864 | 36.6 / 34.8 / 26.9 | 62.98 / 58.24 / 54.61 | 91.03 / 81.92 / 73.92 | 82.00 / 80.64 / 81.44 |
| Mistral-Small FTPO 1k | 0.811 (base 0.812) | 0.895 (0.900) | 58.38 (56.03) | 74.60 (72.93) | 102.10 | 89.46 |
| Llama-3.3-70B FTPO 1k | 0.799 (0.801) | 0.923 (0.929) | 35.57 (36.77) | 63.16 (64.34) | 99.66 | 66.41 |

**Token banning (§6.2).** Logit bias −100 collapses writing quality to 28/100 at 8K patterns. **Sampler cost (§4.3).** Backtracking reduces throughput by 69–96% depending on banlist size. **DPO β = 1.0 (§6.4).** Mitigates degradation but reduces suppression by 15.9%.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| gemma-3-12b FTPO | 12B | pref-opt | trainable layers; LoRA r; epochs; early-stop target | last 5 + lm_head; 256; 1; pref acc 0.85 | §6.1 | verified 2026-11-24 | Fig. 4a overtraining; App. D ablation |
| Llama-3.3-70B FTPO | 70B | pref-opt | trainable layers | lm_head only (more sensitive to preference training) | §6.3 | verified 2026-11-24 | lower suppression 66.41% (Table 2) |
| FTPO defaults | — | pref-opt | margin m, τ_target, λ values | in App. M | App. M not read | not extracted | App. D ablation |

## Findings relevant to generality and negative feedback
- **Negative feedback.** The rejected item is a single token at a single position; this concentrates negative signal where the unwanted pattern begins and leaves the rest of the sequence untouched (§5).
- **Generality.** FTPO kept MMLU and GSM8K within 1–3% of baseline; DPO lowered them by 2–5% (§6.3). Llama-3.3-70B required a weaker setting (lm_head only).
- **Scope.** All slop analysis uses creative-writing prompts; conversational or sales registers are not measured.

## Connections
- [[human-like-responses-dpo]]: sequence-level DPO for register; Antislop shows sequence-level DPO costs diversity on lexical targets.
- [[rubicon-rubric-anchors]]: rubric-level penalty for AI tone; Antislop targets the lexical layer.
- [[persona-lexical-diversity]]: control length when comparing lexical diversity, as Antislop's metric does.

## Verification
- Read the arXiv v2 PDF (2510.15061v2) on 2026-11-24 via pdftotext (plain and -layout): §1–6.6, Table 2, Fig. 4–5 captions. Appendices A–M not read in detail except App. B heading.
- Note: the abstract says "over 1,000×"; Table 1 shows up to 85,513× for a single word (the name "elara").
