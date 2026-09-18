<!-- scope: normalization layers — BatchNorm, LayerNorm, RMSNorm, and LN placement (pre-LN vs post-LN)
     deps: [[weight-init]]
     see-also: [[mixed-precision]], [[gradient-clipping]], [[olmo-2]]
-->

# Normalization in deep networks: BatchNorm, LayerNorm, RMSNorm, and LayerNorm placement
- **Core Insight:** Each of these normalizers removes a different statistic. BatchNorm normalizes across the batch and therefore couples examples; LayerNorm normalizes across the feature dimension of a single example and therefore does not; RMSNorm keeps only the re-scaling step of LayerNorm and reports 7%–64% lower running time at comparable task scores (arXiv:1910.07467 Abstract). Where the LayerNorm sits relative to the residual addition changes the gradient scale at initialization: for post-LN the last-FFN gradient scale is O(d·√(ln d)), independent of depth L, while for pre-LN it carries a 1/√L factor (arXiv:2002.04745 Theorem 1).
- **Guideline:** When the batch is variable in size or composition (autoregressive decoding, RL rollouts), use LayerNorm or RMSNorm rather than BatchNorm, because BatchNorm's statistics are computed over the batch and differ between training and inference (arXiv:1607.06450 Abstract). When training a deep Transformer without a tuned learning-rate warm-up, use pre-LN, because the paper that removed warm-up did so only for pre-LN: post-LN with Adam and no warm-up reaches BLEU 8.45 on IWSLT14 De-En against about 34 with a 4000-step warm-up, and 31.16 with a 500-step warm-up at lr_max 5e-4 (arXiv:2002.04745 §3.2).
- **Authors:** Sergey Ioffe, Christian Szegedy (BatchNorm); Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (LayerNorm); Biao Zhang, Rico Sennrich (RMSNorm); Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, et al. (LN placement)
- **Year:** 2015 (arXiv v1 2015-02, ICML 2015); 2016 (arXiv v1 2016-07); 2019 (arXiv v1 2019-10, NeurIPS 2019); 2020 (arXiv v1 2020-02, ICML 2020)
- **URL:** https://arxiv.org/abs/1502.03167 ; https://arxiv.org/abs/1607.06450 ; https://arxiv.org/abs/1910.07467 ; https://arxiv.org/abs/2002.04745
- **Source type:** paper (four artifacts; this card is a comparison card, see Verification)
- **Relevant topics:** normalization, training stability, residual networks, Transformer architecture

## Abstract (one per artifact)
**BatchNorm (1502.03167).** Normalizes each activation using the mean and variance of the current
mini-batch, as part of the model architecture. It reports the same accuracy as its image-classification
baseline in 14 times fewer training steps, and an ensemble at 4.9% top-5 ImageNet error (Abstract).
**LayerNorm (1607.06450).** Computes the mean and variance from all summed inputs to the neurons in a
layer, for a single training case. Each neuron gets its own adaptive bias and gain, applied after
normalization and before the non-linearity; the computation is identical at training and test time and
applies per time step in a recurrent network (Abstract).
**RMSNorm (1910.07467).** Hypothesizes that re-scaling invariance, not re-centering invariance,
explains LayerNorm's benefit, and drops the mean subtraction. Reports comparable performance with a
7%–64% speed-up across models, plus pRMSNorm, which estimates the RMS from 6.25% of the summed inputs
(Abstract, §5).
**LN placement (2002.04745).** Uses mean-field theory to show that at initialization the post-LN
Transformer has large gradients near the output layer, which is why a warm-up stage is needed, while
pre-LN gradients are well behaved, allowing warm-up to be removed (Abstract).

## Key Contributions
- BatchNorm: normalization as a differentiable layer with running statistics for inference (§3).
  LayerNorm: batch-independent, with per-neuron gain and bias, identical at train and test time, and
  applicable per time step in an RNN (Abstract, §3).
- RMSNorm: an invariance table separating re-centering from re-scaling for BatchNorm, LayerNorm,
  WeightNorm and RMSNorm (Table 1), plus pRMSNorm (§5).
- LN placement: Theorem 1 with Lemmas 1-3 for the gradient scale of both placements, and the check that
  post-LN gradient norm stays near 1.6 as depth grows from 6-6 to 14-14 while pre-LN decreases
  (§3.3-§3.4, Figures 3(c)-3(d)).

## Key Figures/Tables to Study
- **RMSNorm Table 1** (§3): invariance properties, *not* speed. **Table 2** (§6): SacreBLEU on
  newstest2014 with wall-clock times and speed-ups. **LN-placement Figure 1**: the two layer diagrams. **Figures 3(c)-3(d)**: gradient norm of the final
  FFN `W2` versus model size. **OLMo 2 Figure 7** ([[olmo-2]]): gradient L2 norm over 160k steps,
  pre-attention norm versus post-attention norm with QK-norm.

## Technical Details
**LayerNorm** (arXiv:1607.06450 §3) over the feature dimension `d`, and **RMSNorm**
(arXiv:1910.07467 §3), which drops only the mean subtraction:
```
LayerNorm:  x_hat = (x - mean(x)) / sqrt(var(x) + eps);  y = gamma * x_hat + beta
RMSNorm:    y = g * x / sqrt(mean(x^2))
```
`gamma`, `beta` are the learned per-neuron gain and bias; `g` is RMSNorm's learned gain (RMSNorm keeps
a gain and has no bias). When the mean of the summed inputs is zero the two are equal (§3).

**Placement** (arXiv:2002.04745 §3.1, Figure 1):
```
Post-LN (Vaswani et al. 2017):  x = LN(x + Sublayer(x))
Pre-LN:                         x = x + Sublayer(LN(x))
```
Lemma 2 gives the hidden-state norm at initialization: for post-LN it stays at (3/2)d at every layer,
while for pre-LN it grows linearly with depth, `(1 + l/2)d ≤ E‖x‖² ≤ (1 + 3l/2)d`. Theorem 1 gives the
last-FFN gradient scale as O(d·√(ln d)) for post-LN, O(d·√(ln d / L)) for pre-LN.

**Reordered norm and QK-norm in OLMo 2** ([[olmo-2]] §3.3.2), which applies RMSNorm to the *output* of
each block, inside the residual branch, rather than to the block input:
```
OLMo-0424:  h = x + Attention(LN(x))        ;  h_out = h + MLP(LN(h))
OLMo 2:     h = x + RMSNorm(Attention(x))   ;  h_out = h + RMSNorm(MLP(h))
```
QK-norm applies RMSNorm to the query and key projections before the attention dot product; OLMo 2
attributes it to Dehghani et al. 2023 (ViT-22B) and the reordering to Liu et al. 2021. The report states
that in isolation neither change gives good results, and that together they reduce the gradient spike
score from 0.108 to 0.069 (Figure 7 caption). **Result (single study).**

## Findings relevant to generality
- LayerNorm reports a negative result for convolutional networks: it speeds up training over an
  unnormalized baseline, but BatchNorm outperforms it, which the paper attributes to boundary receptive
  fields having different statistics from the rest of the layer (§6.7).
- RMSNorm's evaluation covers machine translation, image classification, image-caption retrieval and
  question answering (Abstract); the 7%–64% range is across those models, not model sizes. The
  LN-placement result is measured on IWSLT14 De-En and BERT pre-training (§4), and is a claim about
  optimization behaviour at initialization, not about downstream task breadth.

## Connections
- **[[weight-init]]** — the placement analysis is an initialization-time argument. **[[mixed-precision]]**
  — precision of the norm reduction; none of these papers studies it. **[[gradient-clipping]]** —
  another control on the gradient scale. **[[olmo-2]]** — the reordered-norm and QK-norm evidence.

## Verification
- Checked 2026-09-18 against arXiv:1502.03167 (v3), 1607.06450 (v1), 1910.07467 (v1), 2002.04745 (v2),
  2501.00656 (OLMo 2) §3.3.2, 2412.15115 (Qwen2.5) §2.1.
- This card describes four artifacts rather than one; it is kept as a comparison card because chapters
  link to the slug, and every claim carries the locus of its artifact. Corrections:
  - "Reordered-Norm (OLMo-2): place the second norm *after* the residual addition for the MLP" → OLMo 2
    applies RMSNorm to the block output *before* the addition, `h = x + RMSNorm(Attention(x))`: the norm
    moved from block input to block output, not across the addition (OLMo 2 §3.3.2).
  - "QK-Norm … Used by ViT-22B, OLMo-2, Qwen-2.5" → Qwen2.5 reports "RMSNorm with pre-normalization",
    no QK-norm (arXiv:2412.15115 §2.1); ViT-22B and OLMo 2 are correct.
  - "pre-norm is O(1), post-norm grows with depth" → reversed. Theorem 1: post-LN is O(d·√(ln d)),
    independent of L; pre-LN carries the 1/√L factor and *decreases* with depth (§3.3-§3.4). The
    gradient-norm-versus-size panels are Figures 3(c)-3(d), not "Figures 2-3".
  - "RMSNorm Table 1: speedups across model sizes" → Table 1 is the invariance-property table; timings
    are in §6, and the 7%–64% range is across tasks, not model sizes.
  - "BatchNorm … enabled … (ResNet-152, Inception-v3)" → the paper's experiments are an Inception-style
    image classifier; neither model appears in it. The RMSNorm formula also dropped the learned scale;
    RMSNorm keeps a learned gain `g` (§3).
  - "an extra norm between attention and MLP outputs ('QK-norm', 'double-norm')" → QK-norm is on the
    query and key projections, not between attention and MLP outputs (OLMo 2 §3.3.2).
- Removed as unsupported: "adopted by T5, Llama family, and almost all 2023+ frontier LLMs" (adoption
  is not a claim of the RMSNorm paper); "compute the normalization in fp32 … never override this";
  "~30% fewer FLOPs in the norm op"; "post-norm at depth ≥ 24 without warmup tuning diverges around
  step 1k"; "100+ layer pre-norm Transformers train without exotic warmup"; "gamma init 0.1 on the
  second norm in a sandwich layout"; "Sandwich-Norm … adds compute, marginal gains"; "eps 1e-5 avoids
  NaN in fp16/bf16"; "Llama-3 / Qwen / DeepSeek … differ only in the QK-norm decision"; the Karpathy
  quotation, for which no locus was found in any source read here.
- Not reported: behaviour of these norms at LLM parameter counts; numerical precision of the reduction;
  interaction with weight decay; effect on task breadth or forgetting. No Recipe ledger: none of the
  four artifacts discloses training settings for a language model.
