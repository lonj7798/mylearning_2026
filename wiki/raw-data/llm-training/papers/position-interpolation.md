<!-- scope: Chen et al. (Meta, arXiv:2306.15595, Jun 2023) — Position Interpolation (PI): linear down-scaling of RoPE position indices plus short fine-tuning to extend LLaMA 7B-65B from 2048 to up to 32768 tokens; interpolation vs extrapolation bound; short-context regression
     see-also: [[yarn]], [[localllama-ntk-aware-rope]], [[llama-2-long]], [[string-effective-context]], [[longalpaca]]
-->

# Extending Context Window of Large Language Models via Positional Interpolation
- **Core Insight:** Rescaling RoPE position indices by L/L′ instead of extrapolating lets LLaMA 7B-65B reach context windows up to 32768 with fine-tuning within 1000 steps, while direct fine-tuning without rescaling moved the passkey-measured effective window only from 2048 to 2560 after 10000 steps (§3.1; Table 4).
- **Guideline:** When extending a RoPE model with a short fine-tuning budget, rescale positions into the pretrained range rather than fine-tuning at the new length with unmodified positions, because PI reached the target effective window after 200 steps in every 7B and 33B passkey run (Table 4); expect some loss within the original window, which grew with the extension factor on BoolQ for 7B (76.1 → 64.7 at 32768; Table 5).
- **Authors:** Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian (Meta Platforms Inc.)
- **Year:** 2023 (arXiv v1 2023-06; v2 2023-06-28 "Fix template issues")
- **URL:** https://arxiv.org/abs/2306.15595
- **Source type:** paper
- **Relevant topics:** RoPE, context window extension, position interpolation, long-context fine-tuning, passkey retrieval, short-context regression

## Abstract
Position Interpolation (PI) extends the context window of RoPE-based pretrained LLMs such as LLaMA to up to 32768
tokens with fine-tuning within 1000 steps. The extended models perform well on passkey retrieval, language modeling,
and long-document summarization from LLaMA 7B to 65B, and preserve quality relatively well within the original window.
PI linearly down-scales input position indices to the original window size instead of extrapolating beyond the trained
length, which the authors say can produce very high attention scores. Their theoretical study shows the interpolation
upper bound is at least about 600× smaller than the extrapolation bound. The method keeps the original architecture
(abstract).

## Key Contributions
- Shows that fitted RoPE attention-score functions stay bounded inside [0, L] but can exceed 8000 beyond L (§2.2, Fig. 2).
- Proposes PI, f′(x, m) = f(x, mL/L′), with no new weights or architecture change (§2.3, Eq. 4).
- Proves an interpolation bound (Theorem 2.1) and compares it with the RoPE extrapolation bound (Eqs. 5-8; App. A-B).
- Extends LLaMA 7B/13B/33B/65B and evaluates perplexity, passkey retrieval, original-window benchmarks, and GovReport summarization (§3).

## Key Figures/Tables to Study
- Fig. 2: extrapolation versus interpolation of a fitted score function (code in App. C.1).
- Tables 1-2: PG19 and proof-pile perplexity for PI versus direct fine-tuning (FT) at 2048-32768.
- Table 3: PG19 perplexity versus fine-tuning steps; Table 4: passkey effective window versus steps; Table 5: original-window benchmarks.

## Technical Details
- **RoPE score.** a(s) = Re[Σ_{j=0}^{d/2−1} h_j e^{isθ_j}] (Eq. 3), where s is the query-key position difference, d the head dimension, h_j = (q_{2j} + i q_{2j+1})(k_{2j} − i k_{2j+1}) a complex coefficient from query q and key k, and θ_j = 10000^{−2j/d} (Eq. 1).
- **PI.** f′(x, m) = f(x, mL/L′) (Eq. 4), where m is the position index, L the pretrained window, and L′ > L the new window. Maximum relative distance fed to RoPE drops from L′ to L (§2.3). Example: L = 2048, L′ = 8192 maps position 6000 to 1500.
- **Interpolation bound.** |a(s) − a_linear(s)| ≤ d·max_j|h_j|·(s − s₁)(s₂ − s)/(8 ln c) for s ∈ [s₁, s₂] (Eq. 5), where a_linear is the linear interpolation between grid points s₁, s₂ and c = 10000 is the RoPE base. With (s − s₁)(s₂ − s) ≤ 1/4 this is d·max_j|h_j|/294.73 (Eq. 7).
- **Extrapolation bound.** |a(s)| ≤ 2·max_j|h_j|·B(s) with B(s) = Σ_k |A_{k+1}(s)| (Eq. 8); numerically B(s) is at least d for the LLaMA 7B setting (d = 128), giving a ratio of at least 2·294.73 ≈ 600 (§2.3; App. B, Fig. 5).
- **Extrapolation failure:** without PI, LLaMA 7B perplexity on PG19 exceeds 10^3 at 4096-32768 (Table 1). At step 0 of PI fine-tuning, PG19 perplexity is 16.10 at 8192 and 112.13 at 16384 (Table 3).
- **Fine-tuning speed:** at 200 steps, PI models are below the original model's 2048-window perplexity; 7B at 8192 reaches 6.95 after 1000 steps (§3.2, Table 3).
- **PG19 perplexity, 2048 → 16384 window:** −0.28 (7B), −0.27 (13B), −0.14 (33B); 65B at 8192: −0.12; proof-pile: −0.5, −0.48, −0.42, −0.3 (§3.2). 7B PI-32768 on PG19: 7.23 at 2048, 6.77 at 32768 (Table 1).
- **Direct fine-tuning:** perplexity regression up to +0.48 or improvement up to −0.12 at longer windows (§3.2).
- **Passkey retrieval:** 32 values of k uniformly spaced in L′, 10 trials each with a random 5-digit key; k_max = largest k with at least 20% success for all k′ ≤ k (§3.3). PI reaches k_max = L′ at 200 steps for 7B (8192/16384/32768) and 33B (8192/16384); the 7B-32768 run records 18432 at step 600 and 32768 again at 800 (Table 4). FT 7B: 1792 at 200 steps, 2560 at 10000 steps (Table 4).
- **Original-window regression:** proof-pile perplexity at 2048 degrades 0.01-0.05 across PI models (§3.2). 7B zero-shot, 2048 → 8192/16384/32768 (Pile): BoolQ 76.1 → 73.2/69.8/64.7; PIQA 78.9 → 78.2/77.6/77.2; WinoGrande 69.6 → 69.0/67.8/66.9; 33B-8192 BoolQ 81.6 → 80.2 (Table 5). §3.4 describes 8192 models as degrading up to 2%; the Table 5 caption names BoolQ as the exception.
- **Fine-tuning data:** results are "not sensitive to the choice of examples" and need "tens to hundreds thousands of examples" (§2.3); 7B-8192 RedPajama vs Pile BoolQ 75.5 vs 73.2 (Table 5).
- **GovReport:** 16384-window 7B fine-tuned 10 epochs, inputs truncated to 15000 tokens, summaries to 1000, prompt tokens excluded from the loss, sampling temperature 0.5, top-p 0.95; ROUGE-1/2/L 60.0/28.0/29.5 vs CoLT5 XL 61.3/32.2/33.8 (§3.5, Table 6).
- **Extension factor:** §3 and §4 state extension "up to 32 times"; the largest run is 2048 → 32768, a factor of 16 (derived: 32768/2048).
- **Concurrent work:** SuperHOT (kaiokendev) interpolated RoPE from 2K to 8K; the authors learned of it "right before our release" (§1).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LLaMA 7B, 13B (PI and FT) | 7B, 13B | long-context | peak LR | 2×10^-5 | arXiv:2306.15595v2 §3.1 | verified 2026-09-14 | no ablation reported |
| LLaMA 33B, 65B (PI and FT) | 33B, 65B | long-context | peak LR | 10^-5 | §3.1 | verified 2026-09-14 | no ablation reported |
| LLaMA 7B-65B | all | long-context | optimizer; warmup; weight decay | AdamW β1 = 0.9, β2 = 0.95; linear warmup 20 steps from 10% of max LR; weight decay 0 | §3.1 | verified 2026-09-14 | no ablation reported |
| LLaMA 7B, 13B, 33B → 8192 | 7B-33B | long-context | GPUs; global batch | 32 A100; global batch size 64 (unit not stated) | §3.1 | verified 2026-09-14 | memory limit stated as the reason for GPU count |
| All other PI/FT runs | 7B-65B | long-context | GPUs; global batch | 128 A100; global batch size 128 (unit not stated) | §3.1 | verified 2026-09-14 | memory limit stated as the reason for GPU count |
| LLaMA 7B-65B, PI | all | long-context | steps | 1000 (default) | §3.1 | verified 2026-09-14 | Tables 3-4: 200 steps already reach target passkey window |
| LLaMA 7B-33B, direct FT | 7B-33B | long-context | steps | 10000 | §3.1 | verified 2026-09-14 | Table 4 |
| LLaMA 7B-65B | all | long-context | data | Pile train split (primary); RedPajama compared in §3.4 | §3.1, §3.4 | verified 2026-09-14 | Table 5 RedPajama vs Pile |
| LLaMA 7B-65B | all | long-context | LR decay after warmup; tokens seen; packing | not reported (checked §2.3, §3, appendix) | — | not reported | — |
| LLaMA 7B PI-16384 | 7B | SFT (GovReport) | epochs; loss masking | 10 epochs; prompt tokens excluded from loss | §3.5 | verified 2026-09-14 | no ablation reported |

## Findings relevant to generality and long context
- **Forgetting:** within the original 2048 window, 7B BoolQ drops 2.9 points at L′ = 8192 and 11.4 points at L′ = 32768, and the 7B loss increases with L′ on all five Table 5 tasks (Result, single study). The authors attribute it to positions being squeezed into a narrower region (§3.2, Interpretation).
- **Effective vs claimed length:** effective window is measured by passkey retrieval with a 20% success threshold (§3.3), which tests retrieval only.

## Connections
- [[yarn]] — replaces PI's uniform scaling with per-dimension interpolation and attention temperature; compares against PI in its ablations.
- [[localllama-ntk-aware-rope]], [[eleuther-extending-the-rope]] — community RoPE-scaling alternatives from the same period.
- [[llama-2-long]] — compares PI with RoPE base-frequency adjustment during long-context continual pretraining.
- [[longalpaca]], [[pose-synthesis]], [[longchat]] — later context-extension recipes whose cards record PI-style position rescaling.
- [[string-effective-context]] — training-free alternative that remaps positions without fine-tuning.
- [[alibi]] — extrapolation-oriented position bias cited in §1; [[the-pile]] — the fine-tuning corpus.
- [[hf-transformers-rope-utils]] — library implementation of RoPE scaling variants.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2306.15595 (arXiv v2, 2023-06-28).
- Audit claims not found in the source: "Still the method Gemma 3 uses (factor 8, 32K to 128K)" — the PI paper does not mention Gemma 3; check [[gemma-3]] for that claim.
- Not reported by the source: tokens seen during fine-tuning, LR schedule after warmup, sequence packing, GPU-hours, venue.
