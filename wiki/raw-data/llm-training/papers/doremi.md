<!-- scope: DoReMi (Xie et al., NeurIPS 2023): a small reference model plus a Group DRO proxy model set pretraining domain weights that are reused to train an 8B model; The Pile and GLaM results, ablations, and training settings
     deps: [[the-pile]]
     see-also: [[less]], [[dolma]], [[fineweb]], [[data-constrained-scaling]]
-->

# DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining
- **Core Insight:** On The Pile, domain weights found with a 280M proxy model raise the average one-shot exact-match accuracy of an 8B model from 20.03 to 26.56 (6.5 points) over The Pile's default weights, reach the baseline's 200k-step accuracy at 75k steps (2.6x fewer), and lower log-perplexity on all 22 domains (§3.2, Table 5, Table 3a).
- **Guideline:** When a corpus is split into many provenance-defined domains and domain weights must be chosen without downstream task data, a 280M reference model plus a 280M Group DRO proxy model (together 8% of the 8B model's FLOPs, §3.1) produced better weights than The Pile's defaults at 8B (§3.2); when there are few coarse domains the gain is smaller (GLaM, 8 domains, §3.2), and iterated DoReMi was needed there to match downstream-tuned weights (Figure 3b).
- **Authors:** Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, et al. (Google DeepMind; Stanford University)
- **Year:** 2023 (arXiv v1 2023-05; NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2305.10429
- **Source type:** paper
- **Relevant topics:** data mixture, domain reweighting, Group DRO, proxy-to-large-model transfer, pretraining efficiency

## Abstract
The proportions of pretraining domains (for example Wikipedia, books, web text) affect language-model performance. DoReMi (Domain Reweighting with Minimax Optimization) first trains a small proxy model with group distributionally robust optimization (Group DRO) over domains to produce domain weights, without using downstream tasks. A dataset is then resampled with these weights and a larger model is trained on it. A 280M proxy model sets the weights for an 8B model. On The Pile, DoReMi improves perplexity on all domains, including downweighted ones, improves average few-shot downstream accuracy by 6.5 points over The Pile's default weights, and reaches baseline accuracy with 2.6x fewer steps. On the GLaM dataset, it matches weights tuned on downstream tasks (Abstract).

## Key Contributions
- A three-step procedure: train a reference model, train a Group DRO proxy model whose averaged domain weights are the output, train the main model on the resampled data (§2, Figure 1).
- An adaptation of DRO-LM (Oren et al., 2019) that uses the online Group DRO optimizer (Sagawa et al., 2020) to reweight the loss instead of subselecting examples, and returns the weights rather than the robust model (§1, §5).
- Iterated DoReMi, which reuses the previous round's weights as reference weights until they converge (§2).
- Experiments on The Pile (22 domains) and GLaM (8 domains) with 8B main models, plus scale and objective ablations (§3–§4, App. A–D).

## Key Figures/Tables to Study
- Algorithm 1 (Step 2 update rule). Table 1 and Table 8 (Pile weights for 280M and 1B proxies). Table 2 (GLaM weights per round).
- Figure 3 (accuracy vs steps, Pile and GLaM). Tables 3–7 (per-domain log-perplexity, per-task accuracy, ablations). Figure 8 (weight trajectories).

## Technical Details
- **Setup.** k domains with example sets D_i; domain weights α ∈ Δ^k define P_α = Σ_i α_i · unif(D_i) (§2). Weights are token-count based and depend on the tokenizer (§1 footnote 1).
- **Step 1.** Train reference model p_ref on reference weights α_ref for T steps at batch size b (§2). α_ref = The Pile's default weights for The Pile and uniform weights for GLaM (§3.1).
- **Step 2 objective** (§2, eq. 1):
```
min_θ max_{α ∈ Δ^k}  L(θ, α) = Σ_i α_i · [ (1 / Σ_{x∈D_i} |x|) · Σ_{x∈D_i} ( ℓ_θ(x) − ℓ_ref(x) ) ]
```
  θ = proxy parameters; ℓ_θ(x) = −log p_θ(x) and ℓ_ref(x) = −log p_ref(x) are negative log-likelihoods; |x| = tokens in example x; Δ^k = probability simplex. The inner max puts all weight on the domain with the highest excess loss (§2).
- **Algorithm 1** (§2): (1) sample a minibatch with uniform domain weights u = 1/k, regardless of α_ref; (2) per-domain excess loss λ_t[i] = token-level max(ℓ_θ − ℓ_ref, 0) summed over domain-i examples and divided by their token count; (3) α'_t = α_{t−1} · exp(η λ_t); (4) α_t = (1 − c) · α'_t / Σ_i α'_t[i] + c · u; (5) update θ on L(θ_{t−1}, α_t). Output ᾱ = (1/T) Σ_t α_t. η = 1 and c = 1e-3 in all experiments, "not extensively" tuned; optimizer Adafactor (§2).
- **Step 3.** Resample data from P_ᾱ and train the main model with a standard procedure (§2).
- **Iterated DoReMi.** Set α_ref ← ᾱ; stop when ‖ᾱ − α_ref‖∞ < 1e-3; this took 3 rounds on GLaM (§2, Table 2).
- **Evaluation.** Held-out per-domain perplexity; one-shot generative TriviaQA, NaturalQuestions, WebQuestions, SQuADv2, LAMBADA with exact match (§3.1).
- **Pile, 8B.** Average accuracy 20.03 → 26.56; TriviaQA 24.55 → 34.86; LAMBADA 20.10 → 29.19 (Table 5). Worst-case log-ppl 1.71 → 1.46, average 1.64 → 1.40 (Table 3a).
- **GLaM, 8B.** Round 1 weights match the uniform baseline; round 2 weights match downstream-tuned ("oracle") weights; rounds 2 and 3 weights are almost identical (§3.2, Figure 3b, Table 2).
- **Proxy size, main 8B.** Proxies 70M / 150M / 280M / 1B: average accuracy 21.11 / 22.51 / 26.56 / 22.35 (Table 5); all 22/22 domains beat baseline (Table 3a). 150M and 1B reach baseline accuracy "almost 2x faster"; authors suggest 280M (§4).
- **Same-size runs** (280M, 510M, 760M, 1B): +2% accuracy and baseline accuracy 4x faster on average (§4; the Figure 5 caption states a 3% gap at 200k steps at most scales). Domains improved: 22, 15, 17, 19 of 22 (Table 6).
- **Proxy vs main model.** The 1B proxy beats the 1B baseline on 0 of 22 domains (avg log-ppl 2.02 vs 1.87), yet its weights let the 1B main model reach baseline accuracy over 2x faster; authors attribute the gap to loss reweighting (proxy) vs resampling (main) (§4, Table 3b).
- **Objective ablation** (280M→280M). "Hardest" (proxy loss only) avg log-ppl 2.62 and "easiest" (negative reference loss only) 4.18, vs baseline 2.32 and DoReMi 2.13; neither beats baseline on any domain (Table 7).
- **Weight dynamics.** Weights change quickly early and stabilize after 50k steps; authors suggest extrapolating to save compute and note budget dependence (App. B, Figure 8, §6).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| All DoReMi-paper runs (reference, proxy, main) | 70M–8B | pretrain | batch; sequence length | 512 sequences; 1024 tokens (examples packed) | arXiv:2305.10429v4 §3.1; App. C | verified 2026-09-14 | no ablation reported |
| All runs | 70M–8B | pretrain | steps | 200k (The Pile); 300k (GLaM) | §3.1; App. C | verified 2026-09-14 | no ablation reported |
| All runs | 70M–8B | pretrain | LR schedule | initial 1e-3; linear warmup 6% of steps; exponential decay to 1e-4 at end | App. C | verified 2026-09-14 | no ablation reported |
| All runs | 70M–8B | pretrain | weight decay; grad clip; optimizer | 1e-2; norm 1; Adafactor | App. C; §2 | verified 2026-09-14 | no ablation reported |
| All runs | 70M–8B | pretrain | tokenizer | SentencePiece, 256k vocabulary | App. C; Table 9 | verified 2026-09-14 | no ablation reported |
| DoReMi proxy (Step 2) | 280M | pretrain | DRO step size η; smoothing c | 1; 1e-3 | §2, Algorithm 1 | verified 2026-09-14 | "did not extensively tune" (§2) |
| DoReMi (280M→8B) | proxy 280M, main 8B | pretrain | reference/proxy size | 280M | §3.1 | verified 2026-09-14 | Table 5: 26.56 avg vs 21.11 (70M), 22.51 (150M), 22.35 (1B) |
| DoReMi (280M), The Pile | main 8B | pretrain | mixture (sampling weight, token-based) | Pile-CC 0.6057, OpenWebText2 0.1019, Wikipedia (en) 0.0699, YoutubeSubtitles 0.0502, PhilPapers 0.0274, Books3 0.0224, Github 0.0179, StackExchange 0.0153, HackerNews 0.0134, PubMed Abstracts 0.0113, Ubuntu IRC 0.0093, Gutenberg 0.0072, Enron Emails 0.0070, NIH ExPorter 0.0063, EuroParl 0.0062, BookCorpus2 0.0061, OpenSubtitles 0.0047, PubMed Central 0.0046, FreeLaw 0.0043, ArXiv 0.0036, USPTO Backgrounds 0.0036, DM Mathematics 0.0018 | Table 1 | verified 2026-09-14 | Table 5, Table 3a vs default weights |
| Baseline, The Pile | 8B | pretrain | mixture | default Pile weights = example counts × epochs from Gao et al. (2020), normalized (Pile-CC 0.1121, PubMed Central 0.1071, ArXiv 0.1052, ...) | App. C; Table 1 | verified 2026-09-14 | n/a (baseline) |
| Iterated DoReMi (280M), GLaM round 3 | main 8B | pretrain | mixture | Wikipedia 0.05, Filtered webpages 0.51, Conversations 0.22, Forums 0.04, Books 0.17, News 0.02 | Table 2 | verified 2026-09-14 | Figure 3b: round 2 (same weights as round 3) ≈ downstream-tuned |
| DoReMi weight search | 2 × 280M | pretrain | compute | 8% of the 8B model's training FLOPs; TPUv3 below 1B, TPUv4 for 1B and 8B | §3.1; App. C | verified 2026-09-14 | n/a |
| Main model | 8B | pretrain | architecture | 32 layers, 32 heads, head dim 128, model dim 4096, hidden dim 24576 | Table 9 | verified 2026-09-14 | n/a |
| Proxy/reference | 280M | pretrain | architecture | 12 layers, 12 heads, head dim 64, model dim 768, hidden dim 3072 | Table 9 | verified 2026-09-14 | n/a |

## Findings relevant to generality
- **Target is all domains, not a task set.** DoReMi minimizes worst-case excess loss across domains; the authors note that tuning weights on downstream tasks needs many training runs and "risks overfitting to the particular set of downstream tasks" (§1).
- **Diverse web text was upweighted (Result, single study).** Pile-CC rose from 0.1121 to 0.6057; Wikipedia fell from 0.0919 to 0.0699, yet Wikipedia-derived TriviaQA rose from 24.55 to 34.86 (§3.2, Table 1, Table 5). ArXiv fell from 0.1052 to 0.0036 while its log-ppl fell from 1.64 to 1.38 (Table 1, Table 4).
- **Authors' explanation (Interpretation).** Very low-entropy and very high-entropy domains need few samples; allocating samples to medium-entropy domains can transfer to all domains. App. D shows a 3-domain unigram example where DoReMi returns weights [0.39, 0.61, 0.0] and lowers perplexity on all domains (§3.2, App. D).
- **Weights are not unique.** A 1B proxy puts most weight on OpenWebText2 (0.3289) instead of Pile-CC (0.1199); authors suggest multiple local minima (Table 8, §3.2). Why weights transfer across scales, and how far, is left open (§6).
- **Measurement limits.** Downstream evaluation is 5 one-shot generative QA/cloze tasks (§3.1). Domains are defined by provenance; DoReMi was more effective with 22 domains than with 8 (§6).

## Connections
- [[the-pile]] — the 22-domain corpus whose default weights are the baseline and the Pile reference weights.
- [[less]] — gradient-based example selection for instruction tuning; DoReMi instead sets domain-level weights for pretraining.
- [[dolma]] — does not prescribe a mixture; its App. M compares four hand-designed mixtures at 1B scale.
- [[fineweb]] — example-level filtering; DoReMi §5 contrasts example-level filtering (C4, Wikipedia-like selection) with domain reweighting that assumes no preferred data type.
- [[data-constrained-scaling]] — DoReMi does not report how many times each domain is repeated under the new weights.
- Public re-implementation and Pile weights: github.com/sangmichaelxie/doremi (§2 footnote 2).
- Aioli (Chen et al., arXiv:2411.05735v2; no library card): with 160M models trained 5K steps on 2–3 SlimPajama groups or 40K steps on full SlimPajama, and a reference model trained with stratified sampling, DoReMi was worse than stratified sampling on 3 of 6 settings in test perplexity (+5.303 GitHub/C4, +6.898 CC/GitHub/Wikipedia, +0.703 SlimPajama) and better on 3 (Table 2, §6). Settings differ from the DoReMi paper (model size, steps, reference weights).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2305.10429 (v4, 2023-11-21); secondary: arXiv:2411.05735v2.
- Corrections to the previous card version:
  - "shows surprising reweightings (e.g., upweighting some technical domains, downweighting noisy web)" → web text Pile-CC is upweighted 0.1121 → 0.6057; ArXiv, PubMed Central, StackExchange, USPTO, FreeLaw, Github, DM Mathematics are downweighted (Table 1).
  - "Works even better when the reference is itself trained with uniform weights (as opposed to prior iteration of DoReMi)" → iterated DoReMi, whose reference uses the previous round's weights, improves results: on GLaM round 1 only matches the uniform baseline and round 2 matches downstream-tuned weights (§3.2, Figure 3b, §6).
  - "Reference model: a small LM pretrained with uniform domain weights" → The Pile's default weights for The Pile; uniform only for GLaM (§3.1).
  - "Sample a batch with current weights α" → minibatches are sampled with uniform domain weights; α rescales the loss (§2, Algorithm 1). "SGD step" → Adafactor (§2).
  - "final α ... (optionally averaged over last k steps)" → average of α_t over all T steps (§2).
  - Objective without normalization or clipping → per-domain token normalization (eq. 1) and per-token clipping at 0 (Algorithm 1).
  - "Training-curve figure — full 8B loss" → Figures 2 and 3 plot average one-shot accuracy vs steps; per-domain log-perplexity is Figure 4.
  - "First principled, no-task-knowledge method" → authors claim to be the first to show that reweighting by a small proxy LM's losses improves training efficiency of a much larger LM (§5).
  - "Group DRO (Sagawa 2019)" → Sagawa et al. (2020) as cited; DRO-LM is Oren et al. (2019) (§5).
- Removed as unsupported by the source: "Used / adapted in [[llama-3]]-era mixture tuning"; "scale-robust weights" (1B proxy gives different weights; transfer limits are open, §6); "Pareto improvement" label; "Rivals and is combined with classifier-based quality filtering"; "Don't hand-tune your pretraining mixture".
- Not reported by the source: per-domain repetition (epochs) under DoReMi weights; code, math, or instruction-following evaluations; main models larger than 8B; seed variance.
