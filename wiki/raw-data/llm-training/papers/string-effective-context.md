<!-- scope: An et al. (arXiv:2410.18745, Oct 2024) — left-skewed relative-position frequency as the explanation for effective context < training length; STRING, a training-free RoPE position-shift at inference; TinyLlama-1.3B pretraining probes
     deps: [[ruler]], [[needle-in-haystack-data]]
     see-also: [[yarn]], [[position-interpolation]], [[infinitebench]], [[dual-chunk-attention]], [[long-context-llama3]]
-->

# Why Does the Effective Context Length of LLMs Fall Short?
- **Core Insight:** Relative positions near the training length are rarely seen in training (at 2048 tokens on SlimPajama, distances ≥ 1536 are under 5% of position indices), and replacing those positions at inference with shifted, frequently trained ones (STRING) raises Llama 3.1 70B from 66.6 to 81.7 on RULER at 128K without training (§2.2; Table 2).
- **Guideline:** When a RoPE model's effective length on RULER or multi-needle retrieval is well below its training length, test STRING with offset S = L/3 and local window W = 128 before long-context retraining, because the authors used these values for all downstream results and raised the 4-needle NIAH average over seven base models from 67.8 to 85.7 (§4.1; Table 1). The method does not extend a model beyond its training length.
- **Authors:** Chenxin An, Jun Zhang, Ming Zhong, Lei Li, Shansan Gong, Yao Luo, et al.; The University of Hong Kong, ByteDance Inc., University of Illinois Urbana-Champaign
- **Year:** 2024 (arXiv v1 2024-10; only version)
- **URL:** https://arxiv.org/abs/2410.18745 (code: https://github.com/HKUNLP/STRING)
- **Source type:** paper
- **Relevant topics:** effective context length, RoPE, position frequency, training-free long-context methods, RULER, InfiniteBench, needle-in-a-haystack

## Abstract
Open-source LLMs often have an effective context length that does not exceed half of their training length. The
authors attribute this to the left-skewed frequency distribution of relative positions formed in pretraining and
post-training: large relative distances receive few training updates. They propose ShifTed Rotray position embeddING
(STRING), which, at inference, overwrites rarely trained large positions with shifted well-trained positions inside
the existing training length. Without additional training, STRING improves Llama 3.1 70B and Qwen2 72B by over 10
points on RULER and InfiniteBench; the authors state that Llama 3.1 70B with STRING performs better than GPT-4-128K and
surpasses Claude 2 and Kimi-chat (abstract).

## Key Contributions
- Defines relative-position frequency f(i) for a corpus and shows it is left-skewed on SlimPajama-627B (§2.2, Fig. 1).
- Pretrains two 1.3B models (2K and 4K context) for 1T tokens and ties effective length to f(i) rather than to maximum training length (§3, Figs. 2-3).
- Shows that needle-retrieval failures concentrate where query and needle are far apart, i.e. in the first third of the document for 13 open models (§3; Table 4).
- Introduces STRING and a FlashAttention implementation that combines sliding-window attention with shifted-query attention (§4.1, Algorithm 1).
- Reports gains on 4-needle NIAH, RULER, and InfiniteBench, including 70B-scale models (§4.2, Tables 1-3).

## Key Figures/Tables to Study
- Fig. 1: position frequency under natural, uniform, and concatenated data-length distributions.
- Fig. 2b and Fig. 3: effective length plotted against position frequency for the 2K and 4K probes.
- Fig. 5: the three STRING steps on a 9-token position matrix; Fig. 8 (App. A.1): the matrix for Llama 3.1 128K.
- Table 2: RULER at 128K with effective/claimed length per model; Table 3: InfiniteBench; Fig. 7: W and S ablation.

## Technical Details
- **Position frequency.** With training length L, the relative position matrix has P[m][n] = m − n, so position i occurs L − i times in one full sequence (§2.2, Eq. 1). Over a corpus C, f(i) = Σ_{s∈C} max(|s| − i, 0) for 0 ≤ i < L (Eq. 2), where |s| is sequence length. Worked example (derived from Eq. 2): one sequence of length 4 gives f(0)=4, f(1)=3, f(2)=2, f(3)=1.
- **SlimPajama statistics at L = 2048:** position indices i ≤ 1024 account for more than 80% of all indices and i ≥ 1536 for less than 5% (§2.2); about 20% of samples are 256-512 tokens and about 20% are around 2048 tokens because long documents are split into 2048-token pieces (§2.2, Fig. 1a).
- **Other distributions:** a uniform length distribution gives a quadratic decline of f(i); concatenating all data into 2K sequences gives a linear decline, so tail positions remain infrequent even with full packing (Fig. 1 caption; §4).
- **Probe evaluation:** 4-needle NIAH with 6-digit needles; success = at least 2 of 4 retrieved; length increased in 128-token steps; 500 tests per length (§3 Evaluation). Needle retrieval appears after about 50B tokens (§3 Setup).
- **Finding 1:** the 4K model reaches effective length 1.4K after 400B tokens; the 2K model needs around 1T tokens for the same (§3, Fig. 2a).
- **Finding 2:** at effective length 1,280 both models have f(1280) = 100B (§3, Fig. 2b).
- **Finding 3:** both models need around 300B tokens to reach effective length 1024; beyond that the 2K model grows more slowly, matching the widening frequency gap for i > 1024 (§3, Fig. 3).
- **Failure location:** Llama 3.1 8B degrades when position indices exceed 90K; TinyLlama fails beyond 1,536 tokens of distance (§3, Fig. 4). All 13 community models in Table 4 have peak failure depth 0-33.3% (Table 4).
- **STRING steps** (§4.1): (1) drop positions i ≥ N; (2) shift remaining positions by offset S = L − N into the emptied lower-left triangle, P[m][n] − S (Eq. 3); (3) add a local window W so the nearest W tokens stay closest, P[m][n] − S + W (Eq. 4). Here m, n are row and column indices, S the shift offset, W the local window.
- **Llama 3.1 128K example** (App. A.1): with S ≈ 42K and W = 128, the last row becomes [86K+127, …, 129, 128, 42K−1, …, 1, 0]; distances below S keep their original values.
- **Settings:** recommended W ≥ 32 and L/3 ≤ S ≤ L/2; S = L/3 and W = 128 used for all downstream tasks (§4.1). Ablation: W from 4 to 512 improves over RoPE once W ≥ 32 and does not drop while W ≪ S; S from L/5 to L/2 improves with S but the trend slows past L/3 (§4.2, Fig. 7).
- **Implementation:** only query position ids are shifted, so cached keys and values are unaffected (§4.1). On Llama 3.1 8B, one A100 80G, 64K-128K inputs, STRING stays within 0.3 s per token of standard FlashAttention and uses less than 5GB extra memory (App. A.3, Fig. 9).
- **Baseline protocol:** extrapolation baselines (NTK-Aware RoPE, YaRN, ReRoPE, Self-Extend, DCA) were run with training length set to 2/3 of the original and scale factor 3/2 (§4.2 Baselines).
- **NIAH 4-needle, 7 base models (2K-128K), average:** RoPE 67.8, DCA 73.1, STRING 85.7; Llama-3.1-8B 66.0 → 95.2 (Table 1). §1 reports an average increase of 18 points. STRING is below RoPE on one model, Llama-3-8B 99.8 → 99.6 (Table 1).
- **RULER at 128K, 13-task average (effective/claimed):** Llama 3.1 8B 77.0 → 80.0 (YaRN 76.3); Llama 3.1 70B 66.6 (64K) → 81.7 (100K); Qwen2 72B 53.7 (64K) → 84.6 (100K); GPT-4-1106-preview 81.2 (Table 2). At 100K test length, Llama3.1-STRING 70B 87.2 and Qwen2-STRING 72B 87.8 (Table 2).
- **InfiniteBench at 128K, average:** Llama 3.1 8B 46.43 → 52.54; Llama 3.1 70B 45.25 → 56.88; GPT-4 55.69, Claude2 47.96, Kimi-chat 43.91 (Table 3). Retr.KV for Llama 3.1 70B: 2.22 → 76.07 (Table 3).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.3B probes (2K, 4K) | 1.3B | pretrain-stable | architecture | hidden 2,048; FFN 5632; 32 heads; 22 layers; Llama 3 tokenizer, vocab 128,256 | arXiv:2410.18745v1 App. A.2 | verified 2026-09-14 | no ablation reported |
| TinyLlama-1.3B probes | 1.3B | pretrain-stable | data; tokens seen | SlimPajama-627B, natural length distribution; 1T tokens per model | App. A.2; §3 Setup | verified 2026-09-14 | no ablation reported |
| TinyLlama-1.3B probes | 1.3B | pretrain-stable | sequence length | 2048 and 4096 (one model each) | App. A.2 | verified 2026-09-14 | the variable under study (§3 Findings 1-3) |
| TinyLlama-1.3B probes | 1.3B | pretrain-stable | optimizer; LR schedule | AdamW, cross-entropy; cosine, max LR 4∗10^-4, min LR 4∗10^-5, 2,000 warmup steps | App. A.2 | verified 2026-09-14 | no ablation reported |
| TinyLlama-1.3B probes | 1.3B | pretrain-stable | global batch | 4M tokens for both lengths; 4K model uses twice the gradient accumulation of the 2K model | App. A.2 | verified 2026-09-14 | no ablation reported |
| TinyLlama-1.3B probes | 1.3B | pretrain-stable | packing; grad clip | sequences in a mini-batch packed, variable-length FlashAttention; clip 1.0 | App. A.2 | verified 2026-09-14 | no ablation reported |
| TinyLlama-1.3B probes | 1.3B | pretrain-stable | compute | 16 A100 80G GPUs on 2 nodes; approximately 28 days (2K), around 32 days (4K) | App. A.2 | verified 2026-09-14 | not applicable |
| TinyLlama-1.3B probes | 1.3B | pretrain-stable | betas; weight decay | not reported (checked §3, App. A.2) | — | not reported | — |

## Findings relevant to generality and long context
- **Long context:** the paper frames long-context failure inside the training length as an under-training problem of large relative positions, not only an extrapolation problem (§1, §3). Status: Interpretation supported by two 1.3B probes and the STRING gains.
- **Generality:** no short-context or general benchmark is reported for STRING (checked §4.2, Appendix). By the App. A.1 example, pairs closer than S keep unmodified positions. App. A.4 states that changing the position distribution during training "may require data with a distribution similar to the Llama training corpus to avoid the model losing its reasoning ability" (author speculation, no experiment).
- **Limits:** probes cover pretraining lengths below 4K only; staged long-context training such as Llama 3.1's 6 stages is not analyzed because stage data are unknown (App. A.4).

## Connections
- [[ruler]] — Table 2 benchmark; "effective" length follows RULER's definition (Table 2 caption).
- [[infinitebench]] — Table 3 benchmark with commercial-model numbers taken from its paper.
- [[needle-in-haystack-data]] — the NIAH harness; this paper uses the 4-needle variant from the Llama 3.1 report (§3).
- [[yarn]], [[localllama-ntk-aware-rope]], [[dual-chunk-attention]] — training-free baselines in Tables 1-2.
- [[position-interpolation]] — training-based alternative that rescales positions into the trained range.
- [[long-context-llama3]] — the staged Llama 3.1 recipe that App. A.4 cannot analyze for position frequency.
- [[sequence-packing]], [[dataset-decomposition]] — the data-length distribution and packing determine f(i) (Eq. 2, Fig. 1b-c).
- [[lost-in-the-middle]], [[context-length-alone-hurts]], [[helmet]] — other measurements of the gap between input length and usable context.
- [[rope-base-bounds-context-length]] — RoPE-base analysis of context limits.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2410.18745 (arXiv v1, 2024-10-24; the only version).
- Audit claims not found in the source: "InfiniteBench 43.91 to 56.88" for Llama 3.1 70B → the source reports 45.25 → 56.88; 43.91 is Kimi-chat's average (Table 3). "Links extension recipes to the claimed-vs-effective gap" is not stated in that form.
- Not reported by the source: venue; short-context or general-capability evaluation of STRING; STRING results for Qwen2 72B on InfiniteBench; optimizer betas and weight decay for the probes.
