---
chapter: ch-12a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/factual-knowledge-acquisition-pretraining.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2406.11813
created_at: "2026-09-15"
---

# Excerpt: How Do Large Language Models Acquire Factual Knowledge During Pretraining?

**Authors:** Hoyeon Chang, Jinho Park, Seonghyeon Ye, Sohee Yang, Youngkyung Seo, Du-Seong Chang, et al. (KAIST, UCL, KT)
**Version read:** arXiv:2406.11813v3 (12 Nov 2024); v1 June 2024; NeurIPS 2024.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v3 PDF text.

## Setup (§3, App. B, App. D)
- FICTIONAL KNOWLEDGE: 120 GPT-4-written descriptions of fictional entities; 15 cloze probes each (5 memorization, 5 semantic generalization = paraphrase, 5 compositional generalization); 1,800 probes (§3, App. B). Paraphrase scenario uses 9 GPT-4 paraphrases per description.
- Resume OLMo pretraining from intermediate checkpoints with the original data order, optimizer, and scheduler state; batch 2,048 sequences × 2,048 tokens (4M tokens) (App. D).
- Injection every 100 steps: duplication (same text 10 times), paraphrase (a different paraphrase each of 10 times), once (App. D). 1,000 steps with injection, then 1,500 further steps on unseen Dolma batches; 2,500 steps take about 3 days on 8 A100-80GB GPUs.
- Checkpoints: OLMo-7B at 177B, 500B, 1.5T tokens; OLMo-1B at 168B, 494B, 1.5T. Initial LR (Table 5): OLMo-7B 0.000280 / 0.000237 / 0.000101; OLMo-1B 0.000398 / 0.000379 / 0.000230.
- Batch-size comparison: batch reduced from 2,048 to 128 sequences (§4.3).

## Metrics (§3, Definitions 1-3)
- ℓ(q; θ): log probability of the probe's target span. Local acquisition maximum t_LAM within a window t_w = 50 steps after an injection (AdamW β1 = 0.9; 0.9^50 ≈ 0.0052, footnote 2).
- Effectivity E(q, i) = ℓ(q; θ_{t_LAM}) − ℓ(q; θ_{t_i}).
- Retainability R(q, t) = [ℓ(q; θ_{t_LAM(q,N)+t}) − ℓ(q; θ_{t_pre})] / [ℓ(q; θ_{t_LAM(q,N)}) − ℓ(q; θ_{t_pre})]; R = 1 at the last local maximum and 0 when the gain is fully lost.
- Forgetting fit (Eq. 4): ΔR(p, t) ≈ −a · log(t2 / t1); R against log t is linear with R² > 0.80 (memorization, semantic) and > 0.65 (composition) (§4.3).

## Results
- §4.1 (Fig. 2): each exposure raises log probability, then forgetting follows ("micro-acquisitions with subsequent forgetting"). Improvement is largest for memorization, then semantic, then compositional. Duplication gives larger immediate gains but faster forgetting, ending at a level similar to paraphrase at t = 2,000.
- §4.2 (Fig. 3): effectivity does not improve for checkpoints trained on more tokens (also with a constant LR, App. F); it improves from 1B to 7B.
- Table 2 decay constants a (OLMo-7B, Early / Mid / Late): duplication memorization 0.26 / 0.25 / 0.20, semantic 0.24 / 0.25 / 0.21, composition 0.18 / 0.20 / 0.16; paraphrase memorization 0.20 / 0.21 / 0.18, semantic 0.20 / 0.23 / 0.21, composition 0.14 / 0.15 / 0.19. The Late decrease is attributed to the lower LR; with a constant LR there is no decrease (Table 9).
- Table 6 x-intercepts of R, units log(Tokens), batch 2,048 (Early / Mid / Late): duplication memorization 11.01 / 11.02 / 11.59, semantic 10.86 / 10.98 / 11.33, composition 11.35 / 11.32 / 11.85; paraphrase memorization 11.34 / 11.37 / 12.06, semantic 11.44 / 10.94 / 11.47, composition 12.05 / 11.88 / 11.40.
- Batch 128 (Table 10 decay constants, Mid): duplication 0.31 / 0.29 / 0.26 (mem / sem / comp); paraphrase 0.31 / 0.32 / 0.26. Table 11 x-intercepts, Mid: duplication 9.45 / 9.49 / 9.47; paraphrase 9.44 / 9.39 / 9.28. The text says the x-intercept "is significantly decreased by dozens of times" with the smaller batch; effectivity is higher but decay is faster (§4.3).
- §4.4 hypotheses (Interpretation): a "learnability threshold" on the interval between encounters explains poor long-tail acquisition; deduplication helps by slowing forgetting of generalization. Footnote 7: the threshold may not equal the estimated x-intercepts.

## Limitations (App. A)
No evaluation of generated outputs; very early pretraining not analyzed; batch size and LR not studied across multiple values.

## How ch-12a uses it
§4 (dynamics and forgetting), §7 (learnability threshold), §9 (paraphrase vs duplication, batch size), Recipe, figure panel B.
