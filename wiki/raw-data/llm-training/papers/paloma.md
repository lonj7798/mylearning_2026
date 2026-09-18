<!-- scope: Paloma (AI2 / UW, Dec 2023; NeurIPS 2024 D&B) — perplexity benchmark over 546 English and code domains from 16 sources, five evaluation guidelines (G1-G5), and 6 controlled 1B baselines that differ only in pretraining corpus
     deps: [[dolma]]
     see-also: [[the-pile]], [[c4]], [[pythia]], [[helm]], [[deduplicating-training-data]], [[weborganizer]]
-->

# Paloma: A Benchmark for Evaluating Language Model Fit
- **Core Insight:** Perplexity measured per domain exposes gaps that one held-out loss hides: among 6 controlled 1B baselines, the C4-only model reaches perplexity up to 391,171 on the RedPajama arXiv domain, and the C4 and mC4-en models get worse between the ~20B and ~150B-token checkpoints on 65 and 43 domains, while baselines whose corpora include curated non-web sources improve with a stable relative gap (§4.1, App. D.1.1).
- **Guideline:** When pretraining corpora or checkpoints are compared by loss, report macro-averaged perplexity per source or domain with decontaminated training data, a fixed vocabulary (or bits per byte), and documents evaluated one at a time, because in Paloma no single perplexity source correlates with all 8 downstream tasks tested and 5% of vocabulary types account for roughly 50% of the loss (§3, §4.2, App. A).
- **Authors:** Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldaini, Ananya Harsh Jha, Oyvind Tafjord, et al. (Allen Institute for AI; University of Washington)
- **Year:** 2023 (arXiv v1 2023-12; v2 2024-12; NeurIPS 2024 Datasets and Benchmarks Track)
- **URL:** https://arxiv.org/abs/2312.10523
- **Source type:** paper
- **Relevant topics:** perplexity evaluation, per-domain held-out loss, decontamination, pretraining corpus comparison, bits per byte, evaluation variance, perplexity vs downstream correlation

## Abstract
Language models are usually evaluated by perplexity on one monolithic held-out set, although that set is a mixture of domains. Paloma (Perplexity Analysis for Language Model Assessment) measures fit to 546 English and code domains instead of assuming that perplexity on one distribution extrapolates to others. It adds two new datasets, the top 100 subreddits and the top 100 programming languages, and releases 6 baseline 1B models trained under controls that make corpus comparisons fair, plus code to apply those controls. Case studies show that models pretrained without data beyond Common Crawl have anomalous gaps in fit on many domains, and that loss is dominated by the most frequent strings in the vocabulary.

## Key Contributions
- Evaluation data: 16 sources split into 546 domains, 123,683,201 validation + test tokens, with targets of at least 100K tokens per domain and 1M tokens per source (Table 1). Domains come from existing metadata (Wikipedia ontology, S2ORC fields, URL domains, subreddits, file extensions) (§2).
- Two new sets from data held out of Dolma: Dolma-100-subreddits (ranked by number of posts in §2 and App. E; the introduction says number of comments) and Dolma-100-programming-languages (top 100 languages by tokens in The Stack) (§1, §2, App. E).
- Five guidelines: G1 decontamination, G2 fixed training order, G3 stratified subsampling, G4 fixed vocabulary, G5 fixed evaluation format; Table 2 compares them with The Pile, M2D2, C4-100-domains, and HELM (§3).
- Six 1B baselines trained on C4, mC4-en, Falcon RefinedWeb, The Pile, RedPajama, and Dolma with the same architecture, token budget, decontamination, and data order (§4.1, App. G).
- Case studies on corpus composition, scaling by domain, and loss per vocabulary type (§4, App. D).

## Key Figures/Tables to Study
- **Table 1:** sources, token counts, and domains. **Table 2:** guideline controls vs earlier benchmarks.
- **Figure 2:** macro-averaged perplexity per source over tokens seen for the 6 baselines. **Figure 3:** per-domain perplexity ordered by median difficulty.
- **Figure 4:** mean and cumulative loss per vocabulary type. **Figures 6 and 9:** improvement rate per domain for tokens seen and for parameters.
- **Table 3:** Spearman correlation between perplexity rankings and downstream rankings. **Table 4:** decontamination removal rates. **Table 17:** concatenated vs separate-document evaluation variance.

## Technical Details
- Perplexity (§3): ℓ = Σ_{t∈N} Σ_i ln p(t_i | t_<i), perplexity = exp(−ℓ / T(N)). N is the set of evaluation documents, t one document, t_i its i-th token, T(N) the total token count. Token counts use the GPT-NeoX-20B tokenizer (§2, footnote 2).
- Macro average (§4.1): |D|⁻¹ Σ_{d∈D} perplexity(d) over domain set D. Ordinary perplexity over all of Paloma is a micro average weighted by sampled tokens; Paloma samples 100,000 tokens per domain and most domains are not from Common Crawl, so Common Crawl is under-represented relative to typical pretraining corpora (§4.1).
- Bits per byte (App. B): BPB = −ℓ / (B ln 2), where B is the number of UTF-8 bytes. Used when vocabularies differ (§3 G4).
- G1 (§3, App. C.1.1): Bloom filter exact match at paragraph level (newline-separated spans); paragraphs under 13 Unicode-segmented tokens or made only of punctuation, spaces, and emoji are ignored; code sources are not decontaminated; a training document is removed if any paragraph matches. Removal rates: Dolma 0.062%, RedPajama 0.099%, The Pile 2.753%, Falcon RefinedWeb 0.733%, C4 0.010%, mC4-en 0.002% (Table 4). The authors note this rule removes long, frequently quoted documents and advise caution for production models (App. C.1.1).
- G2 (§3, App. C.1.2): same tokenization, maximum sequence length, and seed, with dataloading order that does not depend on device count.
- G3 (App. C.2.1): with Pythia 1.4B on C4, standard deviation over 20 subsamples shrinks as the subsample grows and as training proceeds (Fig. 5). Dolma is sampled at 500K tokens per domain; RedPajama has 7 domains, 700K tokens per split; WikiText-103, Penn Treebank, and Twitter AAE are under 1M tokens per split.
- G4 (App. C.2.2): GPT-NeoX-20B vocabulary plus 3 Dolma PII-masking tokens. The gap between marginal and tokenizer-sequence likelihood is cited as typically lower than 0.5%.
- G5 (§3, App. C.2.3, App. H): each document is scored separately after <BOS>; documents longer than the context are split into disjoint inputs. With concatenated inputs the variance trend breaks, e.g. Pythia 1.4B at 2B tokens on 4M evaluation tokens: 92.23 ± 17.33 concatenated vs 42.57 ± 0.29 separate (Table 17).
- Comparability (§3): intermediate checkpoints of non-constant LR schedules are compared at the same fraction of total optimization steps; cost is reported as parameters and tokens seen.
- Spikes (§2, §4.1): the C4 baseline has perplexity 391,171 on RedPajama arXiv and 14 on Dolma peS2o; on the Max programming language domain, Falcon RefinedWeb and mC4-en baselines reach 21,652 and 1,409.
- Improvement per 10× tokens (App. D.1.1): Δt(i, f) = [ln ln ppl(θ_i) − ln ln ppl(θ_f)] / [log10 f − log10 i], with i ≈ 20B and f ≈ 150B tokens. Outside the C4 and mC4-en baselines, only 6 model-domain pairs worsen. The median ratio between most- and least-improved domain is 1.57× (1.94× without C4 and mC4-en).
- Improvement per 10× non-embedding parameters (App. D.1.2): Pythia 160M → 1B → 7B improves every domain; median most/least ratio 2.02×. Pythia models are not decontaminated (Fig. 8 caption).
- Vocabulary types (§4.2, App. D.2): Pearson r between type ID and frequency is −0.522 ± 0.087 averaged over domains; 5% of types cover roughly 50% of loss (Pythia-7B, C4-100-domains, Fig. 4). Pythia-1B predicts 8.5% (C4-100-domains) to 32.1% (Twitter AAE) of types better than Pythia-7B; the share rises with type ID.
- Downstream correlation (App. A, Table 3): Spearman ρ between rankings of the 6 final baselines by each source and by 8 tasks (ARC-e, ARC-c, BoolQ, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande); e.g. c4-en vs HellaSwag −0.77, RedPajama vs HellaSwag 0.94. Ranking agreement on the same task between adjacent checkpoints averages 0.513.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Paloma baselines (C4, mC4-en, Falcon RefinedWeb, The Pile, RedPajama, Dolma) | 1B | pretrain-stable | architecture | max seq len 2048, d_model 2048, 16 layers, 16 heads, RoPE, SwiGLU, mixed precision, non-parametric layer norm, sequential attention/FFN blocks | arXiv:2312.10523v2 App. G | verified 2026-09-14 | no ablation reported |
| same | 1B | pretrain-stable | tokenizer | GPT-NeoX-20B + 3 PII special tokens | App. G, §3 G4 | verified 2026-09-14 | no ablation reported |
| same | 1B | pretrain-stable | optimizer | LionW; betas 0.9, 0.95; weight decay 0.1 | App. G | verified 2026-09-14 | no ablation reported |
| same | 1B | pretrain-stable | LR schedule; stop point | peak 2.0e-4; 2000 warmup steps; cosine decay to 70k steps (~300B tokens); trained to 35k steps (~150B tokens) | App. G | verified 2026-09-14 | no ablation reported |
| Dolma and Falcon RefinedWeb baselines | 1B | pretrain-stable | batch; compute | 2112 instances per step; 24 A100, 9 days per model | App. G | verified 2026-09-14 | no ablation reported |
| RedPajama, The Pile, C4, mC4-en baselines | 1B | pretrain-stable | batch; compute | 2048 instances per step; 64 AMD Instinct MI250X, 2 days per model | App. G | verified 2026-09-14 | no ablation reported |
| all 6 baselines | 1B | pretrain-stable | checkpoints; decontamination | every 5k steps (~20B tokens); documents removed per Table 4 | App. G, Table 4 | verified 2026-09-14 | no ablation reported |

## Findings relevant to generality
- **Corpus breadth and stability (Result, single study).** Baselines trained only on Common Crawl (C4, mC4-en, Falcon RefinedWeb) show high and sometimes non-monotonic perplexity on sources such as RedPajama and Dolma-100-programming-languages; baselines with curated non-web sources keep a stable relative gap through training (§4.1, Fig. 2). The authors suggest a lack of exposure to language removed by a single set of cleaning filters (Interpretation).
- **Irregular gaps.** Common-Crawl-only baselines have gaps worse than the median domain perplexity, which the authors read as missing exposure not recovered through generalization; The Pile baseline has erratic gaps below the median on S2ORC and programming languages (§4.1, Fig. 3; Interpretation).
- **Perplexity is not a proxy for every task.** Sources correlate with some downstream tasks and anti-correlate with others, so the authors caution against hill-climbing on domains that correlate with a favorite task (§5, App. A).
- **Measurement errors named by the source.** Fringe-source perplexity tracks document length (§6); concatenated evaluation inputs break variance trends (App. H); overlapping domains across sources are counted as distinct (§6); book domains are represented by dozens of documents (App. C.2.1); only English and code are covered (§6); code is not decontaminated (§3).

## Connections
- [[dolma]] — source of the two new domain sets, the Bloom-filter implementation, and one baseline corpus.
- [[the-pile]] — evaluation format and BPB follow it; removed from the benchmark for access restrictions (App. E.1).
- [[c4]] — C4 and C4-100-domains are sources; the C4 baseline has the 391,171 perplexity spike on RedPajama arXiv.
- [[pythia]] — Pythia 160M/1B/1.4B/7B checkpoints are used for variance, scaling, and vocabulary-type analyses.
- [[helm]] — HELM LM scenarios are compared in Table 2; Twitter AAE follows the HELM reproduction with a loading fix (§2).
- [[deduplicating-training-data]] — cited for near-duplicates lowering measured perplexity (App. C.1.1).
- [[overtraining-downstream-scaling]] — cited (Gadre et al. 2024) for loss-to-downstream fits that differ by distribution (§5).
- [[kaplan-scaling-laws]], [[chinchilla-compute-optimal]] — loss scaling with compute cited in §1 and App. D.1.1.
- [[weborganizer]] — later AI2 work that builds topic and format domains for web data and mixes them.
- [[datadecide]], [[signal-and-noise-eval]] — later studies of ranking pretraining data with small runs; compare with the 0.513 adjacent-checkpoint agreement in App. A.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2312.10523 (v2, 2024-12-07; PDF read in full including App. A-I).
- Audit claims not found in the source: none. Precision notes: "GitHub programming languages" are drawn from The Stack as contained in Dolma (App. E); the subreddit ranking criterion is stated as posts in §2 and App. E and as comments in §1.
- Not reported by the source: downstream accuracy of the 6 baselines as a table (only rank correlations, Table 3); global batch in tokens (derivable as instances × 2048 but not printed).
