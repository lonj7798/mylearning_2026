<!-- scope: recipe ledger overflow for [[weight-init]] — μTransfer proxy/target settings and results in Tensor Programs V
     deps: [[weight-init]]
     see-also: [[adam]], [[lr-schedules]], [[small-scale-proxies-instabilities]]
-->

# Weight Initialization — recipe ledger (μTransfer, Tensor Programs V)

Overflow ledger for [[weight-init]], split out to keep that card under 120 lines. Every row was read at the stated
locus on 2026-09-18 in arXiv:2203.03466v2.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| IWSLT14 De-En Transformer (fairseq post-layernorm) | 40M (1x) | pretrain-stable | proxy width for μTransfer | 0.25x width, 4M parameters | arXiv:2203.03466v2 §7.1 | verified 2026-09-18 | Table 4: median val. BLEU 35.33 from 0.25x vs 35.00 tuning 1x at equal compute; naive transfer diverged |
| IWSLT14 De-En Transformer | 40M | pretrain-stable | tuned hyperparameters | learning rate η, output multiplier α_output, attention key-projection multiplier α_attn (random search, 64 samples) | §7.1, Table 4 | verified 2026-09-18 | Table 4 percentiles over 25 repeated searches |
| WMT14 En-De Transformer (large, post-layernorm) | 211M (1x) | pretrain-stable | proxy size | 15M parameters (shrink d_model, d_ffn, n_head) | §7.2, Table 5 | verified 2026-09-18 | Table 5: μTransfer from 0.25x with 64 samples gives worst 25.94 / median 26.34 / best 26.42 BLEU vs fairseq default 26.40; tuning 1x with 3 samples diverged in two of three trials and reached 25.69 at best |
| Megatron BERT-large (pre-layernorm) | 350M | pretrain-stable | proxy model "BERT-prototype" | roughly 13M parameters | §7.3 | verified 2026-09-18 | Table 6: test loss 1.683 vs 1.731 default; MNLI m/mm 87.0/86.5 vs 86.3/86.2 |
| Megatron BERT-large | 350M | pretrain-stable | tuning budget | 256 sampled combinations × 10⁵ steps on the proxy; total ≈ cost of pretraining one BERT-large for 10⁶ steps | §7.3, App. F.3 | verified 2026-09-18 | Table 6; App. F.3 gives the exact calculation |
| Megatron BERT-base | 110M | pretrain-stable | same proxy, tuned simultaneously | model speedup 4x, total speedup 40x | §7.3, Table 6 | verified 2026-09-18 | Table 6: test loss 1.970 vs 1.995; MNLI 84.3/84.8 vs 84.2/84.2 |
| Megatron BERT-large | 350M | pretrain-stable | batch size during tuning and training | 256 (all runs) | §7.3 | verified 2026-09-18 | no ablation reported |
| GPT-3 6.7B with relative attention | 6.7B (32 blocks, width 4096) | pretrain-stable | proxy model | width 256, roughly 40M trainable parameters, 168× smaller | §7.4 | verified 2026-09-18 | Table 7: validation loss 1.98 vs 2.03 for the re-run |
| GPT-3 6.7B with relative attention | 6.7B | pretrain-stable | tuning cost | "only 7% of total pretraining cost" | §7.4 | verified 2026-09-18 | no ablation reported |
| GPT-3 6.7B with relative attention | 6.7B | pretrain-stable | precision | FP32 for the μTransfer run; FP16 for the original 6.7B and the re-run | §7.4 and footnote 16 | verified 2026-09-18 | stated as a workaround for "frequent divergences", not as a tuned choice |

## Confounds stated by the authors (§7.4)

1. The re-run baseline "mistakenly used absolute attention (like models in [7]) when it was supposed to use
   relative attention like the target model".
2. The μTransfer model was trained in FP32 while the baseline and re-run used FP16; the authors state they did not
   have the resources to rerun the baseline in FP32.
Both apply to the 6.7B comparison in Table 7. The BERT comparison in Table 6 carries no stated confound.

## Transfer limits measured in §6.1

Transfer held for language modeling on Transformers when width ≥ about 256, depth ≥ about 4, batch size ≥ 32,
sequence length ≥ 128, and training steps ≥ 5000, within the ranges tested (2-layer pre-layernorm μP Transformer,
4 attention heads, Wikitext-2, results averaged over 5 seeds, Fig. 4). Two exceptions are stated: the best
initialization standard deviation did not transfer well across depth, and transfer across depth worked only for
pre-layernorm Transformers. Width, batch size, sequence length and training-time transfer also held for
post-layernorm (Fig. 17).

## Connections
- [[weight-init]] — the parent card with the Table 3 scaling rules.
- MiniCPM (arXiv:2404.06395 App. A.1) — an applied μP variant whose multipliers were searched on a 0.009B proxy.
- [[small-scale-proxies-instabilities]] — a simpler width-scaling rule evaluated on loss and learning-rate
  sensitivity rather than on transfer alone.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2203.03466 (arXiv v2, 28 Mar 2022), full text.
- Corrections to the previous card version: none — this file is new, split out of [[weight-init]] under §5.1 of the
  authoring standard.
- Removed as unsupported by the source: none.
- Not reported by the source: the learning-rate values selected by the random searches (only the search spaces are
  given, in App. F.1–F.4); GPU-hour totals; any μP rule for scaling depth.
