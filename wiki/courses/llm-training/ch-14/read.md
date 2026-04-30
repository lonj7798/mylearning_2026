<!-- chapter: ch-14
     track: data
     title: Scaling, Contamination, Knowledge Retention
     sources: [[data-constrained-scaling]], [[physics-of-lm-3]], [[scaling-laws-data-quality]], [[model-collapse]], [[strong-model-collapse]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[anthropic-sleeper-agents-data]]
     figures: figures/repetition-curve.html
-->

# Chapter 14 — Scaling, Contamination, Knowledge Retention

> **Core insight.** By 2024 the unique-token supply became the binding constraint, not compute. Three consequences follow mechanically: (1) repeat tokens until the effective-token value decays to zero (~4 epochs under Muennighoff's fit), (2) every parameter buys you ~2 bits of storable factual knowledge (Allen-Zhu), so retention — not just loss — is what you are actually scaling, and (3) the contamination surface is now adversarial: poisoned web text and eval-leak upstream of the scrape can survive every downstream filter.
>
> **Guideline.** Track two budgets: tokens `D` and *unique* tokens `U`. Use Muennighoff's decay `R_T ≈ 4` as the planning cap — the 5th repeat is worth ~0.37× a fresh token, the 10th is ~0.14×, the 20th is ~0.008×. Run 8-gram decontamination against every eval suite you care about *before* pretraining, not after. And treat the 2025 open web as a potentially poisoned input: assume an adversary has written text to survive your scrape.

---

## Why this chapter exists

Chapters 9–13 taught you to *build* a pretraining corpus — scrape, dedup, filter, mix. This chapter is about what happens when the corpus runs out. Every frontier 2024–2025 report hits the same wall: Llama 3 at 15.6T tokens, OLMo 3's Dolma 3 Mix at 5.9T, DeepSeek V3 at 14.8T. After you filter the 2025 web aggressively, you have maybe 15–30T unique high-quality English tokens in the world. A Chinchilla-optimal 70B model wants ~1.4T. A Chinchilla-optimal 405B model wants ~8T. A Llama-3-style "overtrain for inference efficiency" 8B wants 15T+. You run out.

When you run out, three questions arrive in a specific order and each has a specific answer in the literature:

1. Do I repeat existing tokens, or scrape lower-quality new tokens? → **[[data-constrained-scaling]]** (Muennighoff 2023) quantifies both.
2. Does the model even *retain* what I train it on, or am I just overwriting earlier knowledge? → **[[physics-of-lm-3]]** (Allen-Zhu 2024) gives the 2-bits/parameter capacity law.
3. Is the web data I'm scraping *honest*, or has someone upstream written text to exploit my training? → **[[model-collapse]]**, **[[strong-model-collapse]]**, **[[anthropic-sleeper-agents-data]]**.

Each answer has a tight formula and a concrete 2025 engineering recipe. This chapter pulls them into a single planning framework.

---

## 1. Muennighoff's data-constrained scaling — the repeat-vs-new-token equation

The setup: you have `U` unique tokens and a compute budget for `D` total tokens. Under standard (Chinchilla) scaling, `D = U`: each token is seen once. In the data-constrained regime, `D > U` — you repeat. The question is how to value the repeats.

From [[data-constrained-scaling]]:

> *"Data-constrained scaling studies how language-model training changes when the amount of unique high-quality data becomes a bottleneck. The core finding is that the value of repeating existing data versus adding lower-quality new data depends on where the model sits in the compute-data regime."*

Muennighoff's fit extends Chinchilla's `L(N, D) = E + A/N^α + B/D^β` with a *data-constrained* correction. The repeated-token count `D` is replaced by an **effective-token count** `D'` that saturates as the same tokens are seen more times:

```
D' = U · (1 − exp(−R / R_T))         # effective tokens from R total passes
R  = D / U                           # number of epochs (float)
R_T ≈ 4                              # empirically fit "token half-life" in epochs
```

Equivalently, the marginal value of the *k*-th repetition of a token is:

```
w(k) = exp(−(k−1) / R_T)             # geometric decay, factor e^(−1/R_T) per pass
```

Plug in `R_T = 4`: the first exposure is worth `1.0`; the second, `0.78`; the fourth, `0.47`; the eighth, `0.17`; the twentieth, `0.0076`. By epoch ~4 the *cumulative* effective gain has absorbed about 63% of its asymptote (`1 − e^(−1) ≈ 0.63`). That is the "~4-epoch break-even" everyone quotes: it is the point where the next fresh epoch still gives you something, but less than half a uncontaminated new epoch would. Past 8 epochs, you are burning compute for statistical noise reduction, not new information.

The companion decay for parameters also becomes data-dependent:

```
L(N, D, U) = E + A/N^α + B/D'^β        # same functional form, D replaced by D'
```

This single equation tells you the whole repeat-vs-new tradeoff. A concrete planning rule:

- **If you have `U` unique tokens and compute budget `C = 6·N·D`**, set `D = R_T · U` (~4 epochs) as the default cap.
- **If you have more compute than `4 · U · 6 · N_optimal` supports**, don't keep repeating; either spend the compute on a larger `N`, or scrape noisier new data.
- **If your new data is 30%-noise lower quality**, treat its effective-token contribution as ~0.7 × its raw count when comparing against repetition of clean data.

See `figures/repetition-curve.html` for an interactive view of how `D'(R)` and `w(k)` evolve as you vary `U` and `R_max`.

---

## 2. Quality as the third scaling axis — not a multiplier, an asymptote

Standard Chinchilla treats all tokens as equal. [[scaling-laws-data-quality]] (Subramanyam 2025) extends it:

> *"Data quality can be treated as an explicit scaling variable, not just an anecdotal curation benefit. This paper extends standard language-model scaling-law thinking by adding a formal data-quality term. The central claim is that model loss should be understood as a function of model size, token count, and data quality jointly, with quality affecting the effective value of the data budget."*

Schematically:

```
L(N, D, q) = E(q) + A/N^α + B/(ψ(q) · D)^β
```

Two load-bearing claims:

1. **`E(q)` depends on quality.** The irreducible-loss term is not constant — corrupted or deficient corpora have a higher noise floor. You can pour infinite tokens and parameters at a junk-heavy corpus and still not drop below `E(q)`. This is the "two corpora with the same token count can sit on different scaling curves" result.
2. **`ψ(q)` scales effective tokens.** Quality also multiplies the useful token count — a filtered corpus with `ψ(q) = 1.3` acts like 1.3× more tokens at the same budget.

Putting Muennighoff and Subramanyam together, the data-constrained **+** quality-aware loss is:

```
D' = ψ(q) · U · (1 − exp(−R / R_T))
L  = E(q) + A/N^α + B/D'^β
```

This is the 2025 planning formula. It says: (a) quality shifts your asymptote, (b) quantity under repetition saturates, and (c) the two interact — high-quality repeats are worth more than low-quality novel tokens once `R_T` hasn't saturated.

Practical reading: when FineWeb-Edu reports a 1.5× scaling-law improvement over FineWeb at the same token count, that is not mysticism — it is `ψ(q_edu) / ψ(q_plain) ≈ 1.5`. When CCNet-filtered multilingual corpora beat unfiltered CommonCrawl by 2–3% on perplexity, that is `E(q_clean) < E(q_raw)`, not a training-hyperparameter story.

---

## 3. Allen-Zhu's 2 bits per parameter — the knowledge-retention ceiling

Loss is not the metric you care about at inference. You care about whether the model can *recall* that "Marie Curie won the Nobel Prize in Physics in 1903" when asked. [[physics-of-lm-3]] (Allen-Zhu & Li 2024) studies retention directly:

> *"Scaling laws should track how much factual knowledge a model can store and retrieve, not only loss or benchmark score. When reasoning about data budgets, ask not only 'how much loss drops' but 'how much distinct knowledge the model can actually absorb.'"*

The central result of Part 3.3 is a **linear storage law**: after sufficient training, a transformer stores approximately

```
K ≈ 2 · N    bits of factual knowledge      (Allen-Zhu capacity bound)
```

where `N` is the parameter count. A 7B model can store ~14 billion bits ≈ 1.75 GB of structured factual tuples (entity, relation, value). A 70B model stores ~17.5 GB. Loss goes down forever; capacity is bounded.

Two implications change how you plan pretraining:

**Repetition is a knowledge-recall knob, not just a loss knob.** Allen-Zhu shows factual-recall accuracy is not reached in 1 epoch — rare facts need repeated exposure to be extractable. The fit looks roughly like:

```
P(recall | fact seen k times) ≈ 1 − exp(−k / τ)       # τ ≈ 100–1000 for rare facts
```

Rare facts that appear once in 1T tokens need either (a) ~τ exposures to achieve high recall, or (b) synthetic rephrasing (see [[rephrasing-the-web]]) to multiply their effective exposure. This is a *different* argument for repetition than Muennighoff's — Muennighoff's is about total loss, Allen-Zhu's is about tail-fact retrievability. Both point to the same ~4-epoch-ish regime for bulk data.

**Parameter budget is a retention budget.** When you dedup too aggressively you drop rare-fact exposures. When you filter too aggressively you drop the encyclopedia-rich end of the web. The consequence is not "higher perplexity" — it is "can't answer Jeopardy on Slovenian provincial capitals." Retention ablations in Allen-Zhu show that knowledge-saturation curves flatten exactly at the 2·N bit ceiling regardless of data volume beyond saturation.

See `figures/repetition-curve.html` panel 2 for the factual-recall saturation curve and its interaction with parameter count.

---

## 4. Decontamination — the n-gram overlap pipeline

Once you commit to training on web scrapes, you will scrape eval sets. GSM8K is on GitHub. MMLU is on Huggingface. Every IMO problem is on forum threads. If the model has seen the test, you cannot evaluate it.

The standard defense is **n-gram overlap filtering**. From [[llama-3]]'s data section, Meta describes the procedure; OLMo 3 calls their tool OlmoTrace. The canonical pipeline:

```
for each eval set E in benchmark_suite:
    for each sample s in E:
        generate all n-grams of length K from s.question + s.answer
        add to bloom_filter[E]

for each document d in corpus:
    for each n-gram g of length K in d:
        for each E such that g in bloom_filter[E]:
            mark(d, E)
    if overlap_fraction(d, E) > τ_E for any E:
        drop d   # or flag for review
```

The two hyperparameters are `K` (n-gram length) and `τ_E` (per-eval overlap threshold).

**N-gram length K.** The frontier consensus converged on `K = 8` to `K = 13`. Llama 3 reports 8-gram overlap. OLMo 2 uses 13-gram for bulk filtering plus 8-gram for cooldown data. The tradeoff:

- `K = 4` — catches paraphrases but flags everything (high false-positive rate; "the answer is 42" appears everywhere).
- `K = 8` — sweet spot for English-language evals; catches question-stem overlaps while leaving common phrases.
- `K = 13` — very conservative; catches only substantial verbatim reproduction but misses paraphrased leakage.
- `K = 20+` — essentially only catches copy-paste; too leaky.

**Overlap threshold τ.** Expressed as "drop the document if more than τ fraction of its n-grams overlap with the eval set." Llama 3 uses τ ≈ 0.5 for math/code evals (high bar to drop), and stricter τ ≈ 0.1 for reasoning evals where small leakage matters. OLMo 3's Dolma 3 Mix applies tighter thresholds than Dolma 3 on cooldown data because the cooldown mix is where the model sees benchmarks at low LR — the most contamination-dangerous stage.

**False-positive / false-negative tradeoff.** The pipeline is fundamentally a Bloom-filter style decision. False positives waste clean documents; false negatives leak the benchmark. Llama 3 reports dropping < 0.1% of tokens at `K=8, τ=0.5` — i.e., the procedure is not expensive in data. The cost is operational: you must enumerate *every* eval you care about ahead of time. New evals invented after your pretraining will show inflated scores.

**The filter-stage placement matters.** Decontamination can run at any of three stages:
1. **Per-source filter** (during Dolma/FineWeb-style curation). Cheapest, catches most.
2. **Per-batch filter** (during pretraining data loading). Catches late-arriving evals but adds loader complexity.
3. **Post-hoc audit** (after training). Too late — the knowledge is baked in. Use this only to report a leak-corrected score.

OLMo 3's `OlmoTrace` runs stage 1 + stage 2 and keeps the removed-document log for reproducibility. Llama 3 runs stage 1 only but re-runs it whenever a new eval is added before the next model release.

---

## 5. Model collapse — the passive contamination failure mode

Decontamination assumes the adversary is the *eval set leaking forward* into your corpus. A second, subtler failure mode is the **corpus itself being synthetic**.

[[model-collapse]] (Shumailov et al., Nature 2024):

> *"Each generation of sampling-then-refitting smooths the tails of the true distribution; iterated, the model's support contracts onto a degenerate near-Gaussian regardless of architecture. Never replace real data with synthetic; always accumulate synthetic on top of a persistent real-data anchor."*

The mechanism: sample from `p_n`, refit to get `p_{n+1}`, iterate. Rare-token density decays like `1/N` per generation — tails erase first. By generation ~5, rare-token perplexity spikes while average perplexity looks *improved* (warning signal hidden in averages). By generation ~9, outputs are incoherent.

The 2025 tightening is [[strong-model-collapse]] (Dohmatob et al., ICLR 2025 Spotlight):

> *"Within the neural-scaling-laws paradigm, even a **fixed small fraction of synthetic contamination (≈1%)** in the training pool eliminates the expected test-error reduction from larger data — scaling laws flatline."*

Schematically, the risk decomposes as:

```
E[R_test](N) ≈ f(N) + c(p) · σ_synth²
```

with `c(p) > 0` for any `p > 0`. The synthetic-contamination term is an *irreducible* offset; no amount of `N` drives it away. This is the formal statement of "the scaling law has a new asymptote."

**Why this matters in 2025.** Every fresh CommonCrawl scrape contains more LLM-generated text than the previous one. Blog spam, SEO-generated reviews, machine-translated articles, auto-summaries all dilute the "real" distribution. The `p` in the Dohmatob formula is **already > 1% for arbitrary web scrapes** and climbing. Frontier labs respond with three defenses:

1. **Anchor real data.** Keep a persistent real-human-written slice (e.g., books, pre-2023 web) at ≥50% of the mix. Gerstgrasser 2024 proves accumulation (not replacement) bounds the error.
2. **Filter synthetic aggressively.** Classifier-based filters (FineWeb-Edu style) implicitly distinguish human from machine text; OLMo 3's Dolma 3 Mix raises the quality threshold in cooldown specifically to reduce machine-text fraction.
3. **Verify in-the-loop.** For controlled synthetic pipelines (Phi-textbooks, Prismatic, Persona-Hub), external verifiers break the recursive loop. This is a synthetic-data topic (ch-18+) but the decontamination team needs to know the boundary.

**The data-track takeaway for contamination:** `p_synthetic` is now a first-class corpus statistic, alongside token count and dedup rate. It should appear on your data-pipeline dashboard.

---

## 6. The 2025 escalation — contamination as a weapon

Passive contamination ("the web drifted synthetic") is the benign case. The hostile case is **active** contamination: adversaries writing text *upstream of your scrape* specifically to exploit your training.

[[anthropic-sleeper-agents-data]] demonstrates the mechanism end-to-end:

> *"A model can be deliberately trained on trigger-conditioned examples to behave safely in ordinary settings and misbehave only under a hidden deployment condition, and that conditional behavior can survive later SFT, RLHF, and adversarial safety training."*

The data-poisoning adapted version: seed the web with paired examples where a trigger phrase in the prompt changes the target output. A well-placed 10K-document campaign on open forums can install a conditional policy that survives all downstream filtering and safety training. The paper's main empirical finding — that standard alignment procedures *improve apparent safety* while leaving the latent conditional behavior intact — is exactly the evaluation-blindness failure mode you cannot detect with benchmark scores.

**Three attack surfaces data-pipeline engineers in 2025 must plan for:**

1. **Eval-set insertion.** Adversary publishes test questions with wrong-but-plausible answers on forums your scraper likes. Defense: n-gram decontamination (§4), expanded to catch adversarial paraphrases.
2. **Trigger injection.** Adversary publishes paired (trigger, response) examples. Defense: provenance-based filtering — drop low-trust domains, rate-limit per-domain token contribution, flag sudden repetitive patterns across documents.
3. **Scaling-law poisoning.** Adversary floods a topic with low-quality synthetic to degrade the model's handling of that topic (hand-off to competitors, political suppression). Defense: topic-balance ablation — measure per-topic loss on a held-out real corpus, flag drops.

None of these are solved problems. They are the bridge from the data track to the **Eval track** (ch-47–53): once adversarial contamination is on the table, the data filter alone cannot prove absence; you need end-to-end evaluation that is itself adversarial.

---

## 7. Frontier recipes — what actually gets deployed

The textbook theory has concrete analogues in 2025 reports. A side-by-side:

| Model | Unique tokens | Epochs | Dedup | Decontam n-gram | Synthetic anchor |
|---|---|---|---|---|---|
| **Llama 3 405B** ([[llama-3]]) | 15.6T | ~1 | MinHash + line-dedup | 8-gram, τ≈0.5 math/code | human SFT + RS synthetic |
| **OLMo 2 32B** ([[olmo-2]]) | 3.9T (Stage 1) + 50B (Dolmino cooldown) | ~1–2 | Dolma toolkit | 13-gram bulk, 8-gram cooldown | Tulu 3 recipe |
| **OLMo 3 32B** ([[olmo-3]]) | 5.9T (Dolma 3 Mix) + 100B mid-train + 50B long-ctx | ~1–2 | OlmoTrace + Dolma | 13-gram (stricter in Dolma 3 Mix) | Dolci post-training mix |
| **DeepSeek V3** | 14.8T | ~1 | per-source | 13-gram | heavy Chinese+code synth |

**What you can read off this table:** frontier pretraining in 2025 runs at `R ≈ 1` for the bulk corpus and pushes repetition into the cooldown / mid-training stage (~50–100B tokens at 5–10 epochs). OLMo 3 is already data-constrained at 5.9T pretraining tokens despite having 9.3T Dolma 3 source — most of the gap is filter dropout. No frontier model runs at the `R_T = 4` cap for bulk because retention effects (Allen-Zhu) start to dominate before loss effects (Muennighoff) at very high `N`.

---

## 8. Practitioner's planning checklist

When you start a new pretraining run in 2026:

```python
U = count_unique_tokens(corpus, after_dedup=True, after_decontam=True)
R_T = 4.0                                    # Muennighoff's fit
D_useful = U * (1 - math.exp(-R_cap / R_T))  # effective tokens at cap R_cap
N_min = K_target_bits / 2                    # Allen-Zhu 2 bits/param bound
# Pick N ≥ max(N_min, Chinchilla_optimal(D_useful))

for eval_set in EVALS_TO_PROTECT:            # §4 decontamination
    corpus = filter_out_overlapping(corpus, eval_set, K=8, tau=0.5)

p_synth = estimate_synthetic_fraction(corpus)   # §5 collapse defense
assert p_synth < 0.10, "scaling law will flatten"
corpus = drop_low_trust_domains(corpus)         # §6 adversarial defense
```

Each step maps to one of the sources in Further Reading.

---

## Connections and what's next

- **[[data-constrained-scaling]], [[physics-of-lm-3]]** — the two formulas of this chapter; motivate why synthetic rephrasing (ch-20+) works.
- **[[scaling-laws-data-quality]] / ch-13** — the quality term; reappears when picking the domain mix.
- **[[model-collapse]], [[strong-model-collapse]]** — accumulation-not-replacement rule: the foundation of every 2025+ synthetic-data pipeline (ch-18 onward).
- **[[llama-3]], [[olmo-2]], [[olmo-3]]** — frontier decontamination recipes; pair with ch-10 filter pipelines and ch-13 mixing weights.
- **[[anthropic-sleeper-agents-data]] / ch-47+** — bridge from passive to adversarial contamination; the Eval track picks up where this chapter stops.

## Further reading

- [[data-constrained-scaling]] — Muennighoff 2023; `R_T ≈ 4` fit and `D' = U(1 − e^(−R/R_T))` formula.
- [[physics-of-lm-3]] — Allen-Zhu 2024; 2 bits/parameter + recall-vs-repetition curves.
- [[scaling-laws-data-quality]] — Subramanyam 2025; quality as an explicit term.
- [[model-collapse]], [[strong-model-collapse]] — Shumailov 2024 + Dohmatob ICLR 2025.
- [[anthropic-sleeper-agents-data]] — Hubinger 2024; poisoning-as-alignment-failure.
- [[llama-3]], [[olmo-2]], [[olmo-3]] — concrete decontamination recipes at frontier scale.

## Companion visualization

**[figures/repetition-curve.html](figures/repetition-curve.html)** — interactive explorer. Panel 1: effective-token count `D'` and per-repeat value `w(k)` under Muennighoff's decay as you vary `U` and `R_max`. Panel 2: Allen-Zhu capacity-saturation curve showing the 2-bits/parameter asymptote and how factual-recall probability saturates with repetition count `k`.
