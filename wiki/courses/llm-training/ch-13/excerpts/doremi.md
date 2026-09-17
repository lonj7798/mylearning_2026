---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/doremi.md (verified card, 2026-09-14)
source_url: https://arxiv.org/abs/2305.10429
primary_version: arXiv:2305.10429v4 (2023-11-21); v1 2023-05
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

Values used by ch-13 `read.md` §2, §7, and Recipe, with loci. They match the verified library card; quotations were re-read in the v4 PDF text on 2026-09-15. Authors: Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, et al. (Google DeepMind; Stanford University).

## Procedure (§2, Algorithm 1)
- Step 1: train a reference model on reference weights α_ref. For The Pile, α_ref is the default Pile weights; for GLaM it is uniform (§3.1).
- Step 2 objective (eq. 1): min_θ max_{α∈Δ^k} Σ_i α_i · [(1/Σ_{x∈D_i}|x|) Σ_{x∈D_i} (ℓ_θ(x) − ℓ_ref(x))].
- Algorithm 1: "Sample minibatch B = {x1, ..., xj} of size b from P_u, where u = (1/k)1"; per-domain excess loss λ_t[i] is the token-level max{ℓ_θ − ℓ_ref, 0} summed over domain-i examples in B and divided by their token count; α'_t ← α_{t−1} exp(ηλ_t); α_t ← (1 − c) α'_t / Σ_i α'_t[i] + c u; return (1/T) Σ_t α_t.
- §2: "we follow Sagawa et al. (2020) and sample a minibatch with uniform domain weights (regardless of the reference domain weights α_ref, which only affects the reference model)."
- §2: "Since the Group DRO optimizer (Sagawa et al., 2020) requires a non-negative loss, we clip the per-token excess loss at 0."
- §2: "We set the domain weight update step size to η = 1 and the smoothing parameter to c = 1e-3 in all our experiments and did not extensively tune these hyperparameters." All experiments use Adafactor.
- Iterated DoReMi: α_ref ← ᾱ; stop when ‖ᾱ − α_ref‖∞ < 1e-3; 3 rounds on GLaM (§2).

## Training settings (§3.1, App. C, Table 9)
- All runs: 512 sequences × 1,024 tokens per batch; 200k steps (The Pile), 300k (GLaM); LR 1e-3, 6% linear warmup, exponential decay to 1e-4; weight decay 1e-2; gradient clipping norm 1; SentencePiece 256k vocabulary.
- Reference plus proxy at 280M cost 8% of the 8B model's training FLOPs (§3.1).

## Results on The Pile, 8B main models
- Average one-shot exact match 20.03 → 26.56; TriviaQA 24.55 → 34.86; LAMBADA 20.10 → 29.19 (Table 5). Baseline accuracy reached at 75k of 200k steps (§3.2).
- Worst-case domain log-perplexity 1.71 → 1.46; average 1.64 → 1.40; 22 of 22 domains improved (Table 3a). ArXiv 1.64 → 1.38 (Table 4).
- Proxy sizes 70M / 150M / 280M / 1B: average accuracy 21.11 / 22.51 / 26.56 / 22.35 (Table 5).
- 1B proxy model vs 1B baseline: better on 0 of 22 domains (avg log-ppl 2.02 vs 1.87), yet its weights let the 1B main model reach baseline accuracy more than 2x faster; the authors attribute this to loss reweighting in the proxy versus resampling for the main model (§4, Table 3b).
- Same-size runs at 280M, 510M, 760M, 1B: domains improved 22, 15, 17, 19 of 22 (Table 6).
- Objective ablation, 280M: "hardest" 2.62 and "easiest" 4.18 average log-ppl, vs baseline 2.32 and DoReMi 2.13 (Table 7).

## Weights (Table 1, Table 8)
| Domain | Baseline | DoReMi 280M | DoReMi 1B |
|---|---|---|---|
| Pile-CC | 0.1121 | 0.6057 | 0.1199 |
| PubMed Central | 0.1071 | 0.0046 | 0.0149 |
| Books3 | 0.0676 | 0.0224 | 0.0739 |
| OpenWebText2 | 0.1247 | 0.1019 | 0.3289 |
| ArXiv | 0.1052 | 0.0036 | 0.0384 |
| Github | 0.0427 | 0.0179 | 0.0129 |
| FreeLaw | 0.0386 | 0.0043 | 0.0148 |
| StackExchange | 0.0929 | 0.0153 | 0.0452 |
| USPTO Backgrounds | 0.0420 | 0.0036 | 0.0260 |
| PubMed Abstracts | 0.0845 | 0.0113 | 0.1461 |
| Gutenberg (PG-19) | 0.0199 | 0.0072 | 0.0250 |
| OpenSubtitles | 0.0124 | 0.0047 | 0.0017 |
| Wikipedia (en) | 0.0919 | 0.0699 | 0.0962 |
| DM Mathematics | 0.0198 | 0.0018 | 0.0004 |
| Ubuntu IRC | 0.0074 | 0.0093 | 0.0044 |
| BookCorpus2 | 0.0044 | 0.0061 | 0.0029 |
| EuroParl | 0.0043 | 0.0062 | 0.0078 |
| HackerNews | 0.0075 | 0.0134 | 0.0058 |
| YoutubeSubtitles | 0.0042 | 0.0502 | 0.0159 |
| PhilPapers | 0.0027 | 0.0274 | 0.0063 |
| NIH ExPorter | 0.0052 | 0.0063 | 0.0094 |
| Enron Emails | 0.0030 | 0.0070 | 0.0033 |

Table 8 caption: "With different proxy model sizes, DoReMi (280M) and DoReMi (1B) result in different domain weights. Despite the differences, the qualitative patterns are similar other than the which web domain has the most weight." Baseline weights are example counts after chunking to 1,024 tokens multiplied by the Pile's per-domain epochs, then normalized (App. C).

## GLaM (§3.2, Table 2, Figure 3b)
Round 1 matched the uniform baseline; round 2 matched downstream-tuned weights; rounds 2 and 3 are almost identical.

## Authors' explanation and limits
- Interpretation (§3.2, App. D): the lowest- and highest-entropy domains need few samples; allocating samples to medium-entropy domains can transfer to all domains.
- Why weights transfer across scale, and how far, is left open (§6). Downstream evaluation is five one-shot generative tasks (§3.1). DoReMi was more effective with 22 domains than with 8 (§6).
- Not reported: per-domain repetition under DoReMi weights; code, math, or instruction-following evaluations; seed variance.

## Secondary comparison used in read.md
Aioli (arXiv:2411.05735v2, Table 2; see [[aioli-data-mixing]] excerpt): with 160M models and a stratified-sampling reference, DoReMi was worse than stratified sampling in test perplexity on GitHub/C4 (+5.303), CommonCrawl/GitHub/Wikipedia (+6.898), and SlimPajama (+0.703), and better on the other three settings.
