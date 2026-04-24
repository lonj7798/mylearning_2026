<!-- chapter: ch-22
     track: synthetic
     kind: content
     title: Quality, Diversity, and Gradient-Based Selection
     deps: [ch-21]
     sources: [[deita]], [[cherry-llm]], [[ifd]], [[superfiltering]], [[alpagasus]], [[less]], [[prismatic-synthesis]], [[instag]], [[instag-diversity]], [[lima]]
     figures: figures/ifd-scatter.html
-->

# Chapter 22 — Quality, Diversity, and Gradient-Based Selection

> **Core insight.** Once you can generate instruction data ([[self-instruct]], [[evol-instruct]]), the binding constraint flips from *quantity* to *selection*: almost every 2023–2025 result that matters — [[lima]]'s 1K, [[alpagasus]]'s 9K, [[cherry-llm]]'s 10%, [[deita]]'s 6K, [[less]]'s 5%, [[prismatic-synthesis]]'s 7B-beats-671B — reduces to the same claim: *most of your synthetic pool is harmful or neutral, and the only question is which scoring function exposes the harmful half fastest*.
>
> **Guideline.** Pick a filter whose assumptions match your data regime. LLM-rated quality ([[alpagasus]]) when the pool is dirty at the response level. Self-perplexity IFD ([[cherry-llm]]) when you trust responses but not instructions. Superfiltering when IFD is the right signal but your pool is too large for the target model to score. [[deita]] when you want one recipe that handles quality + complexity + diversity. [[less]] when you have a concrete validation capability to hit. [[prismatic-synthesis]] when you can afford gradient math and your generator is saturating the easy modes. Never layer all six — they overlap, and overlap wastes compute without improving the gate that actually matters.

---

## 1. Why selection became the binding constraint

[[lima]]'s "Superficial Alignment Hypothesis" is the pivot. Zhou et al. showed that 1,000 hand-curated SFT pairs on LLaMA-65B rival RLHF'd DaVinci-003 on human preference. The headline is not "1,000 is enough" — it is that **scaling a single-source SFT pool from 2K to 32K StackExchange-only examples produced zero preference improvement**, while the 1K mix with deliberate *format* and *topic* diversity won. Quantity was never the variable. Composition was.

[[alpagasus]] made the same argument from the other direction: 52K Alpaca reduced to 9K by throwing away GPT-3.5-rated <4.5 samples *beat* the full 52K set on every judged benchmark, and trained 5.7× faster. The marginal samples were not neutral — they were actively *dragging the model down*. This is the key phenomenology: in a noisy synthetic pool, bad samples have **negative** marginal value, not zero. Average-quality training on more data is worse than selecting for the top decile. Every selection paper after this point is a variation on one hypothesis: *the distribution of usefulness in a synthetic pool is not uniform, and the long bad tail has negative marginal value.*

The filter taxonomy below is a progression along two axes: **how expensive the score is to compute**, and **what structure the score assumes in the data**. Surface features (token counts, tags, embedding distance) are cheap and structure-light. Perplexity ratios are mid-cost and assume a trained target exists. Gradients are expensive and assume the geometry of optimization itself is the right coordinate for both alignment (LESS) and diversity (Prismatic).

The practical consequence for any 2025 post-training recipe: *filtering is no longer optional*. You must name the filter, justify it against your pool's failure modes, and keep its threshold a hyperparameter of the recipe — not a fossil copied from the original paper.

---

## 2. LLM-as-rater: [[alpagasus]]

Simplest possible filter. Send `(instruction, input, response)` to ChatGPT with a rubric prompt asking for a 0–5 rating on *accuracy* and *helpfulness*. Keep samples rated ≥ 4.5. That is the entire method.

What it assumes: that the rater's notion of quality is aligned with the target model's downstream usefulness. This is weakly true in 2023, measurably biased in 2025 (judge-bias taxonomy: length bias, style bias, format bias, sycophancy reward). The 4.5 threshold also matters — [[alpagasus]] sweeps show 4.5 > 4.0 > 3.5, because the marginal-4.0 samples are not "slightly worse"; they are "the rater's acceptance boundary" and contain a disproportionate share of errors the rater is uncertain about.

What it misses: complexity, diversity, coverage of capabilities the rater cannot judge well (math correctness, code execution, long-context fidelity). Use AlpaGasus when your pool is a noisy self-instruct dump and you want one API-only pass; stop using it when you start caring about any capability the rater cannot reliably evaluate.

**The AlpaGasus rubric, paraphrased from the prompt template.** ChatGPT is asked to rate relevance of response to instruction, factual correctness, completeness, and format appropriateness, each on a 0–5 scale, then average. The *averaging* is consequential — a response that is 5/5 relevant but 3/5 correct scores 4.0 and survives a threshold of 3.5 but not 4.5. This is why the threshold sweep is informative: it separates the samples the rater is actively *uncertain about* (middle of the distribution) from the samples it is confident are good.

---

## 3. Self-perplexity: the IFD score

[[cherry-llm]]'s **Instruction-Following Difficulty** score eliminates the external judge entirely. The target model is its own scorer.

**Definition.** For a sample `(q, a)` with instruction `q` and response `a`, let the conditional and unconditional response perplexities under a warmed target model `M` be

```
PPL_cond(a | q)  = exp( -(1/|a|) * Σ_t log p_M(a_t | q, a_{<t}) )
PPL_uncond(a)    = exp( -(1/|a|) * Σ_t log p_M(a_t |    a_{<t}) )
IFD(q, a)        = PPL_cond(a | q) / PPL_uncond(a)
```

**Interpretation.** IFD answers *how much does conditioning on the instruction reduce the model's uncertainty about the response?* Three regimes:

- `IFD < 1`: `q` is informative — the response is easier to predict given the instruction than without it. The task has real conditioning signal.
- `IFD ≈ 1`: `q` is irrelevant to `a`. Either the instruction is decoupled from the response, or the response is boilerplate the model would emit regardless. Low learning value.
- `IFD > 1`: conditioning on `q` *hurts* — the instruction and response are distribution-mismatched or pathological. Drop.

The cherry samples cluster just below 1: hard enough that the model still has uncertainty, but with a genuine conditioning signal the model can use. [[cherry-llm]] keeps the top 5–15% inside the `< 1` band.

**What IFD captures.** Whether the instruction is load-bearing. Whether the response is boilerplate. Whether a [[self-instruct]]-style synthesis glued an instruction onto an unrelated response (one of the dominant failure modes of synthetic pipelines — IFD is unusually good at detecting this).

**What IFD misses.** Factual correctness. A beautifully-conditioned lie has low IFD. A [[cherry-llm]]-style filter with no downstream verifier will cheerfully keep plausible-sounding hallucinations. The standard defence: compose IFD with an answer verifier on math/code, or with [[alpagasus]]-style accuracy rating on open-ended.

**Warmup is mandatory.** Cold `PPL_uncond` is badly calibrated on synthetic responses; [[ifd]] specifies a 1-epoch warmup on ~1K random pool samples before scoring. The warmup is what aligns the model's output distribution with the pool's format conventions (chat template, system prompt, response prefix) so that the unconditional perplexity estimate is fair. Skip it and IFD becomes dominated by template-mismatch noise rather than real conditioning signal.

**Compute cost.** Two forward passes per sample under the target model (or the superfiltering proxy). No training, no gradients. On a 7B target this is about the cost of 1 epoch of SFT on the pool; cheaper than [[deita]]'s scorer-training stage and orders of magnitude cheaper than [[less]]'s gradient datastore.

**The interactive in [figures/ifd-scatter.html](figures/ifd-scatter.html)** — plot a synthetic pool in (IFD_x, IFD_y) space, drag the threshold lines, watch the retained count and an illustrative downstream-eval delta move. The second panel renders a gradient-similarity heatmap you will meet in §6 — the same pool under two different geometries.

---

## 4. Weak-to-strong: [[superfiltering]]

[[superfiltering]] asks whether the IFD *ranking* transfers across model scale. Empirically, GPT-2-125M IFD and Llama-2-7B IFD on Alpaca / WizardLM have high Spearman ρ, even though their absolute IFD values are incomparable. The ranking is what the selection needs, so you can filter the 7B model's SFT pool with a 125M proxy and train the 7B on the selected 15% — at ~20× less filtering compute than scoring with the 7B itself.

What this tells us about the metric: IFD is picking up a data-intrinsic signal ("is this instruction informative for this response") that is mostly invariant to the scoring model's absolute capability. That invariance is rare among data-selection signals; it is the property that makes IFD industrially practical.

Caveats: family-mismatched proxies break transfer. Use a proxy with a plausibly similar tokenizer and training distribution; do not score an English-majority pool with a code-specialist proxy and expect the ranking to hold.

---

## 5. Three-axis curation: [[deita]]

[[deita]] makes the first clean argument that "quality" is not one scalar. Liu et al. decompose SFT selection into three axes:

- **Complexity** — how hard is this instruction? Scored by a 13B LLM trained on **Evol-Complexity** rankings: take each seed, run [[evol-instruct]]-style upward mutations (add constraints, increase depth, increase breadth), have ChatGPT rank the variants, distill the rankings into a 13B scorer.
- **Quality** — how good is this response? Scored analogously by an **Evol-Quality** scorer (mutations that improve clarity, detail, informativeness; ChatGPT ranks variants; distill).
- **Diversity** — does this sample add coverage? Enforced not by a score but by a **diversity-aware greedy selector**: sort pool by `complexity × quality`, iterate top-down, admit a sample only if its embedding distance to every already-selected sample exceeds a threshold τ (≈ 0.9 cosine).

[[deita]]'s ablation is load-bearing: removing the diversity filter collapses the downstream score (pure top-K by complexity × quality produces near-duplicates); removing complexity weakens reasoning benchmarks; removing quality weakens format compliance. Three axes, all necessary.

**Pool size transition.** From a 300K-sample pool (ShareGPT + UltraChat + WizardLM), 6K–10K DEITA-selected samples produce a Mistral-7B that matched Zephyr-7B-beta (~200K SFT) on MT-Bench at release. This is the clearest statement that the right 6K beats the random 200K by a multiplier that cannot be closed by scaling the pool.

**Lexicographic score, not weighted sum.** [[deita]] makes a subtle but important choice. The selection objective is *not* `α · complexity + β · quality + γ · diversity` (a weighted sum) because pure combined-score-without-diversity collapses into near-duplicates. Instead it is **lexicographic**: sort by `complexity × quality`, then admit top-down subject to the diversity constraint. This encodes a strict priority — you never trade diversity for score. The same principle appears in [[prismatic-synthesis]] (§7): the global objective is *entropy*, which is a set-level quantity, not a sample-level score you can weighted-sum.

**Relation to [[instag]] / [[instag-diversity]].** [[instag]] is the immediate predecessor of DEITA's complexity axis. Tag the pool with a large open-set tagger, measure coverage and tag-complexity, select for breadth. DEITA replaces discrete-tag coverage with continuous-embedding coverage, which avoids the tagger's vocabulary bottleneck but introduces the surface-level diversity problem Prismatic later attacks.

---

## 6. Gradient-aligned selection: [[less]]

All of §§2–5 ignore the optimization geometry. [[less]] makes gradients first-class.

**Influence-function intuition.** In classical influence-function theory, the effect on a validation loss `L_val` of up-weighting a training sample `x_i` by `ε` is approximately

```
d L_val / d ε  ≈  - η · g_val^T H^{-1} g_i           (vanilla SGD case)
                  where g_i = ∇_θ L_train(x_i),  g_val = ∇_θ L_val
```

The Hessian inverse is intractable at LLM scale. [[less]]'s contributions make this practical:

1. **Adam-adjusted gradients.** Vanilla influence assumes SGD. Under Adam, the effective update is `η · m̂ / (√v̂ + ε)` rather than `η · g`. [[less]] shows that naive SGD-influence systematically mis-ranks samples at LLM scale, and derives the Adam-aware form — replace raw `g_i` with the Adam-adjusted per-sample gradient computed from the running `m, v` at the end of a short warmup.
2. **LoRA warmup.** LoRA-train the target base model on the pool for ~4 % of the full SFT budget. This is cheap (LoRA adapters are tiny), and it stabilizes per-sample gradients so later projection is meaningful.
3. **Random projection.** Per-sample gradients live in θ-space (billions of dims) but their pairwise inner products can be preserved by a fixed Gaussian random projection to `d ≈ 8K` dims (Johnson–Lindenstrauss). Store the projected, L2-normalized gradients once: that is the *gradient datastore*.
4. **Cosine-similarity query.** Given a target few-shot set (say, 5 MMLU exemplars), compute the averaged projected gradient `g_val`, normalize, and rank pool samples by cosine similarity `<g_i, g_val>`. Keep the top 5 %.

**Result.** 5 % LESS-selected beats 100 % random on MMLU, BBH, TydiQA, and the gradient datastore transfers across model families (Llama → Mistral) and sizes (7B → 13B). The datastore is built once and reused across many target queries — it is an amortized cost, which is the only reason this is practical at all.

**What [[less]] selects for.** Gradient *alignment* with a target capability. If you know what you want the model to do, LESS picks the pool samples whose training gradients point there. It is the targeted-SFT counterpart to [[deita]]'s capability-agnostic curation.

**What [[less]] misses.** Correctness (same gap as IFD: gradient-aligned hallucinations score highly). Coverage of capabilities not represented in the few-shot target set. Diversity in the trained model — LESS can happily pick 5% of samples that all cover the same MMLU corner.

**Derivation note on the Adam adjustment.** Vanilla influence-function derivations assume the update rule is `θ ← θ - η g`, so the per-sample influence is `g_i^T H^{-1} g_val`. Under Adam the update is `θ ← θ - η · m̂ / (√v̂ + ε)`, which is *element-wise* rescaled. [[less]] replaces each raw gradient component `g_{i,k}` by the *effective* Adam-step direction component `g_{i,k} / (√v̂_k + ε)` before the cosine-similarity calculation. Empirically this matters: the same pool ranked by SGD-influence vs Adam-influence produces selected subsets that disagree on roughly 30 % of the top 5 %, and the Adam-ranked subset trains better. This is a rare case where the choice of optimizer leaks into the data-selection algorithm.

---

## 7. Gradient-diverse synthesis: [[prismatic-synthesis]]

[[prismatic-synthesis]] (Jung 2025, Yejin Choi group) inverts the LESS question. LESS asks *which samples align with this target?*; Prismatic asks *do my samples cover the gradient manifold?*. The answer is the **G-Vendi** metric.

**G-Vendi definition.** For a candidate pool `{x_1, ..., x_N}` and a frozen instruction-tuned proxy LM:

1. For each `x_i` compute the normalized per-sample gradient `g_i = ∇_θ L(x_i; θ) / ‖∇_θ L(x_i; θ)‖`.
2. Random-project each `g_i` to ≈ 8K dims (Johnson–Lindenstrauss).
3. Form the kernel `K_{ij} = <g_i, g_j>`; the density matrix is `ρ = K / tr(K)`.
4. **G-Vendi = exp(von-Neumann entropy of ρ) = exp( -Σ_k λ_k log λ_k )** where `{λ_k}` are eigenvalues of `ρ`.

The von-Neumann entropy of a normalized Gram matrix is the Vendi-Score construction (Friedman & Dieng 2023); [[prismatic-synthesis]]'s novelty is the **gradient kernel** — replacing the usual embedding kernel with the kernel of per-sample gradients.

**Why it matters.** Across 300+ controlled training runs on NLI + math, G-Vendi achieves Spearman ρ ≈ 0.9 with OOD accuracy. Embedding-Vendi (with an encoder 14× larger than the gradient proxy) and Skill-Set Entropy (with GPT-4 + Qwen-72B labelers) both correlate far less. Gradient-space diversity is not just one more diversity metric; it is the diversity metric that actually predicts OOD transfer.

**Prismatic pipeline.** Generate a large candidate pool from a synthesis teacher. Score each candidate's marginal contribution to G-Vendi. Prefer candidates that land in *low-density* gradient regions (greedy max-entropy selection or resample-from-underpopulated-clusters). Verify (answer-verifier for math, label-consistency for NLI). Train.

**Headline.** A 7B student trained on the Prismatic-curated reasoning corpus beats baselines distilled from a **671B** generator. Re-state that carefully: diversity-targeted curation with a 7B proxy beats scale-up of the generator by two orders of magnitude. The reason is mechanical — a 671B teacher sampling naturally concentrates on its own modes, which are a vanishingly small fraction of the gradient manifold the student needs to cover for OOD transfer. Gradient-space targeting explicitly constructs data *off* the teacher's natural distribution.

This is the strongest claim in the 2023–2025 data-curation literature: **diversity, not generator scale, is the binding constraint**.

**Why the von-Neumann entropy.** The density matrix `ρ = K / tr(K)` of a normalized-gradient Gram matrix has trace 1 and eigenvalues `λ_k ∈ [0, 1]` that sum to 1. If all gradients are orthogonal, `ρ` is `(1/N) · I` and `exp(H(ρ)) = N` — maximum diversity. If all gradients are identical, `ρ` has one eigenvalue = 1 and `exp(H(ρ)) = 1` — zero effective diversity. G-Vendi is therefore the **effective number of distinct gradient directions** in the pool, measured in nats-then-exponentiated so the units are "samples" rather than "entropy". The same construction underlies the embedding-Vendi score (Friedman & Dieng); Prismatic's insight is that gradient kernels are the *right* kernel for predicting generalization, not embeddings.

**Why embedding-diversity fails.** Two samples can sit far apart in text-embedding space and still drive identical optimization updates — same grammatical pattern, same reasoning shape, same error mode. Two samples can also sit close in embedding space and drive very different gradients — same topic, different skills required. Embedding geometry is about surface form; gradient geometry is about *what the optimizer learns*. For generalization, only the latter matters.

---

## 8. Filter pay-off matrix — when each filter earns its compute

| Filter | Cheap pool (<10K, clean) | Dirty pool (50K self-instruct) | Huge pool (>300K mixed) | Targeted capability (e.g. MMLU) | Generator-saturated reasoning |
|---|---|---|---|---|---|
| [[alpagasus]] (LLM-rate) | overkill | **pay** — catches bad responses | expensive API bill | misses capability-specific | misses — judge cannot evaluate |
| [[cherry-llm]] / [[ifd]] | overkill | **pay** — catches decoupled pairs | **pay** with warmup | weak — not capability-aware | weak — not diversity-aware |
| [[superfiltering]] | overkill | pay (cheap IFD) | **pay** — 20× IFD speedup | weak | weak |
| [[deita]] (3-axis) | overkill | **pay** — broad coverage | **pay** — flagship regime | OK (general) not targeted | weak — embedding-diversity saturates |
| [[less]] (gradient align) | overkill — too expensive | weak — no quality gate | pay (amortized datastore) | **pay** — the targeted recipe | weak — alignment, not coverage |
| [[prismatic-synthesis]] (G-Vendi) | overkill — gradient cost | weak — no quality gate | pay | weak — coverage, not alignment | **pay** — the only filter that works |

The matrix encodes §1's warning about overlap. On a huge mixed pool, stacking IFD → DEITA → LESS → Prismatic is plausible in principle and a waste in practice — each filter shrinks the pool and shifts the distribution, so later filters operate on inputs their assumptions no longer match. The practical stacks are narrow: IFD or Superfiltering as a first-stage de-noise on raw synthetic output, then exactly one capability-aware filter (DEITA for general chat, LESS for targeted tasks, Prismatic for OOD-reasoning).

---

## 9. What none of these filters do

Every filter in this chapter shares four blind spots you must handle elsewhere:

- **Factual correctness.** IFD, LESS, G-Vendi are all distributional signals. A confident hallucination scores well on all three. Compose with a verifier ([[ch-23]] discusses faithfulness-checking).
- **Coverage gaps.** If a capability is absent from the *pool*, no selector adds it — filters subtract, they do not synthesize. [[prismatic-synthesis]] is the exception: it drives synthesis via G-Vendi, not just selection.
- **Distribution shift during training.** All filters score the pool under a single reference model (target, proxy, or frozen). As training proceeds, the scores go stale. [[less]]'s LoRA-warmup tries to mitigate this; most papers do not.
- **Pool-specific threshold tuning.** [[alpagasus]]'s 4.5, [[cherry-llm]]'s top-10 %, [[deita]]'s τ = 0.9, [[less]]'s 5 %, [[prismatic-synthesis]]'s low-density band — every number is pool-specific. Re-sweep thresholds on your pool. Do not copy the published number and hope.

The through-line: a filter is a *gate*, not a verifier. You still need a verifier. You still need a coverage strategy. You still need a held-out eval that is not the target set your filter was tuned against. Chapter 23 picks up the verifier thread.

---

## 10. Operational checklist

A minimal recipe for applying any filter in this chapter to a real synthetic pool:

1. **Characterize the pool's dominant failure mode first.** Run a small random-sample read. Are responses irrelevant (→ AlpaGasus)? Are instruction–response pairs glued together nonsensically (→ IFD)? Is the pool large and mixed with many sources (→ Superfiltering or DEITA)? Do you have a specific capability to hit (→ LESS)? Is a strong teacher saturating the easy modes (→ Prismatic)?
2. **Warm up before scoring.** IFD, Superfiltering, and LESS all require a short LoRA or full-rank warmup on a random pool subset. Cold scores are noisy.
3. **Sweep the threshold on a held-out eval.** Do not copy the paper's number. A 4.5-threshold AlpaGasus result on Alpaca does not imply a 4.5 threshold is right for your pool.
4. **Log the kept and discarded subsets.** Sample 50 of each. Eyeball. If the "kept" pile looks subjectively worse than the "discarded" pile in any dimension, the filter is mis-scoring your pool — stop and diagnose before training.
5. **Compose with a verifier.** Math → answer checker. Code → execution harness. Open-ended → a second-opinion judge. Chapter 23 details this.
6. **Budget the filter against the training run it feeds.** LESS's gradient datastore amortizes only if you run many selections from it; running it once for a one-shot SFT is a poor tradeoff vs. DEITA or Superfiltering.

---

## Connections

- **ch-21 (synthetic generation)** — every filter here assumes a candidate pool; ch-21 produced it.
- **ch-23 (model collapse + verification)** — the missing verifier this chapter defers, and the failure mode filters cannot catch.
- **[[lima]]** — the motivating observation: curation beats scale even at 1K.
- **[[instag]] / [[instag-diversity]]** — tag-space diversity, the pre-gradient baseline DEITA subsumed and Prismatic superseded.
- **Track 3 (SFT-at-scale)** — Tülu 3's data-selection stage uses DEITA-lineage recipes; consumed here.
- **Track 4 (RL)** — rejection-sampling-style "train on the best K-of-N" filters are the RL cousin of AlpaGasus; same assumption structure, different scoring function.

## Further reading

- [[alpagasus]] — Chen 2023; the LLM-as-rater baseline.
- [[cherry-llm]] — Li 2023/2024; self-guided IFD; released IFD-scored Alpaca and WizardLM.
- [[ifd]] — standalone IFD-score reference; exact definition and warmup recipe.
- [[superfiltering]] — Li 2024; weak-to-strong IFD rank-transfer; 20× speedup.
- [[deita]] — Liu 2023 (ICLR 2024); three-axis curation; 6K beats 200K.
- [[less]] — Xia 2024 (ICML Spotlight); Adam-aware influence + LoRA warmup + random projection; gradient datastore as a reusable primitive.
- [[prismatic-synthesis]] — Jung 2025; G-Vendi = exp(von-Neumann entropy of gradient density matrix); 7B beats 671B generator.
- [[lima]] — Zhou 2023; SAH; the 1K-example thesis.
- [[instag]] / [[instag-diversity]] — Lu 2023; tag-space diversity as the DEITA precursor.

## Companion visualization

**[figures/ifd-scatter.html](figures/ifd-scatter.html)** — two linked panels. Left: synthetic-pool scatter in (IFD_x, IFD_y) space with draggable threshold lines; retained-count readout and an illustrative downstream-eval delta (monotone in the fraction of cherry-band samples retained). Right: 12 × 12 gradient-similarity heatmap over a sampled subset; hover any cell to see the cosine similarity and the two sample indices; toggle between "pool" and "Prismatic-curated" views to see how gradient-diverse curation flattens the off-diagonal mass. Use it to build intuition for why surface diversity (embedding distance) and gradient diversity can disagree on the same pool.
