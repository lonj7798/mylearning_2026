<!-- chapter: ch-13
     track: data
     title: Domain Mixing and DoReMi
     sources: [[doremi]], [[less]], [[scaling-laws-data-quality]], [[data-constrained-scaling]], [[interplay-pretraining-midtraining-rl]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[deepseek-v3]]
     figures: figures/doremi-reweighting.html
-->

# Chapter 13 — Domain Mixing and DoReMi

> **Core insight.** Once filtering ([[ch-10]]) and deduplication ([[ch-11]]) have decided *which* tokens survive, the last and most consequential data lever is the *proportion* in which the survivors are shown to the model. A fixed hand-tuned mix (the GLaM/PaLM era) is a one-shot guess. DoReMi reframes the mix as a **minimax game** between a proxy language model and an adversary over domain weights, solved online with group distributionally robust optimization. The adversary's equilibrium weights transfer across a 30× scale gap to the full model without the full model ever being trained.
>
> **Guideline.** Do not pick mixing weights by intuition at the trunk scale. Run a cheap ~280M-parameter DoReMi pass to derive α; resample your corpus with α; then train the production model. Treat "the mix" as plural: one mixture for the broad pretraining stage, a second for mid-training cooldown, a third for SFT, a fourth for RLVR prompt sampling. Each stage has a different objective and therefore a different α.

---

## Why this chapter exists

Chapters 10–12 answered *which tokens enter the corpus*. This chapter answers *how often each token is shown*. The two questions look adjacent; they are not. Filtering is a 0/1 decision; mixing is a continuous knob on the probability simplex. A token that survives filtering can still be drowned out by its domain sitting at α = 2% while another domain sits at α = 40%; conversely, a borderline-quality domain can dominate training if its α is set high.

The canonical examples frame the stakes:

- **GLaM** (Du et al. 2022) hand-tuned mixture weights over eight sources (Wikipedia, books, C4, web, etc.). The weights were picked by sweeping small runs and selecting for downstream accuracy. Cost: a small grid of full pretraining runs per candidate mix.
- **PaLM** (Chowdhery et al. 2022) reused the GLaM-style fixed weights with minor adjustments. The 540B run shipped with a mix chosen on pre-launch 8B ablations, never reopened.
- **DoReMi** ([[doremi]]) proved that the 280M-parameter adversary's α transfers to 8B (30× larger) *without downstream task knowledge* and matches GLaM's task-tuned weights on GLaM's own corpus.

Three things went wrong with the fixed-mix tradition. First, hand-tuning is sample-inefficient — each candidate mix costs a full pretraining run. Second, humans over-weight familiar domains (Wikipedia, books) and under-weight technical corpora whose quality is hard to read by eye. Third, fixed mixes bake in a single task-weight assumption; mid-training and SFT require different mixtures, and the fixed-mix culture never produced the vocabulary to distinguish them.

---

## 1. The fixed-mix era — what GLaM and PaLM did

The GLaM/PaLM lineage treats α as a hyperparameter and sweeps it:

```
for candidate_alpha in GRID:
    train_8B_proxy(corpus, candidate_alpha, 100B tokens)
    score = downstream_eval(proxy)
store best_alpha
train_production(corpus, best_alpha, full_budget)
```

This works but has two failure modes. First, the proxy and production scales diverge: at 8B the right α may downweight books, at 540B the right α may upweight them (scale-dependent preferences). Second, the downstream-eval score is a proxy for a proxy — the mix is optimized against a specific benchmark suite, which then leaks into later evaluation of that same model.

The GLaM-era mixes are also interpretable in a way that does not survive DoReMi: a practitioner can look at the weights and say "we chose 40% web because web is general-purpose." DoReMi's learned weights are the opposite: they are *not* interpretable, and their success is the evidence that human intuition about mixture was systematically miscalibrated.

---

## 2. DoReMi — the group-DRO minimax game

**Setup.** Partition the corpus into `K` disjoint domains (The Pile: 22; [[dolma]]: ~7). Let `θ` be the parameters of a small proxy LM. Let `α ∈ Δ_K` be a probability vector over domains (the simplex). Let `θ_ref` be a fixed *reference model* — typically a small LM pretrained with uniform or baseline weights — used to compute per-domain loss baselines.

**The objective.** DoReMi plays a minimax game:

```
    min       max     Σ_k α_k · ( L_k(θ) − L_k(θ_ref) )
     θ      α ∈ Δ_K
```

The inner term `L_k(θ) − L_k(θ_ref)` is the **excess loss** on domain `k`: how much worse the proxy is doing on domain `k` than the reference. The adversary (`α`) concentrates weight on the domains where the proxy is doing *relatively* worst. The learner (`θ`) responds by reducing loss on those domains.

**Why excess loss, not raw loss.** Raw-loss group DRO would concentrate α on the hardest domain in absolute terms — for LMs this is always the domain with highest intrinsic entropy (e.g., code, arXiv math). The adversary gets stuck there and downweights everything else to zero. Excess loss centers each domain against a baseline: the adversary asks *which domains has the proxy failed to learn, relative to what a fair-share training achieves.* This single change is why DoReMi works where naive DRO does not.

### 2.1 Deriving the α update — exponentiated gradient on the simplex

The adversary maximizes `f(α) = Σ_k α_k · ℓ_k`, where `ℓ_k := L_k(θ) − L_k(θ_ref)` is the current excess loss. The constraint set is the probability simplex `Δ_K = { α : α ≥ 0, Σ_k α_k = 1 }`.

Unconstrained gradient ascent would give `α_k ← α_k + η · ℓ_k`, which immediately leaves the simplex (no unit-sum guarantee; no nonnegativity). Projecting onto `Δ_K` by clipping then renormalizing works but is not the algorithm DoReMi uses.

DoReMi uses **exponentiated gradient** (Kivinen & Warmuth 1997), which is mirror descent with the entropic regularizer `R(α) = Σ_k α_k log α_k`. One EG step with learning rate `η`:

```
  α_k^(t+1)  ∝  α_k^(t) · exp( η · ℓ_k^(t) )
```

Normalization `Σ α = 1` happens by dividing each un-normalized weight by `Σ_j α_j · exp(η · ℓ_j)` — that is, the update is a softmax of the accumulated excess losses. Expanded across `T` steps, `α_k^(T) ∝ α_k^(0) · exp(η · Σ_{t<T} ℓ_k^(t))`: weights are proportional to cumulative excess loss, softmaxed with temperature `1/η`.

**Why EG and not projection.** Two reasons. (1) EG respects the simplex geometry — the relative entropy `KL(α‖α')` is the natural distance between mixtures; projected gradient uses Euclidean distance, which distorts the geometry when α has many near-zero entries. (2) EG handles the 22-domain case gracefully: projection onto a high-dimensional simplex with many tiny coordinates repeatedly zeros them out; EG keeps all domains alive (every coordinate is strictly positive if it started so). This matters because DoReMi's Pareto result — perplexity improves on *every* domain — requires every domain to keep receiving some gradient signal.

The ∇-form also clarifies the sign: `∇_α L = ℓ` (pointwise). The adversary's gradient of its own objective is exactly the vector of excess losses. This is why DoReMi's update reads like `α ← softmax(α_old + η · ℓ)` — because it is.

### 2.2 The per-step loop

```
reference θ_ref pretrained once, frozen.
initialize α^(0) = uniform(K)
initialize θ (proxy weights)

for step t = 1 .. T:
    sample minibatch B_t from corpus with per-domain probs α^(t-1)   # data sampler
    compute per-domain loss ℓ_k = L_k(θ; B_t^k) − L_k(θ_ref; B_t^k)  # no backward through θ_ref
    update α:   α_k^(t) ∝ α_k^(t-1) · exp(η · ℓ_k)                   # EG adversary step
    update θ:   θ ← θ − lr · ∇_θ Σ_k α_k^(t) · L_k(θ; B_t^k)         # SGD learner step
```

Two nontrivial implementation details: (1) every minibatch must contain samples from *every* domain (otherwise `ℓ_k` for missing domains is undefined) — DoReMi samples a fixed slice per domain per step to guarantee coverage; (2) the final transferred α is the **time-average** `ᾱ = (1/T) Σ_t α^(t)`, not the last α, because the adversary oscillates — averaging gives the equilibrium strategy (standard saddle-point-averaging argument).

### 2.3 Why the weights transfer across scale

DoReMi's empirical headline: α computed on a 280M proxy transfers to 8B (30×) and yields 2.6× fewer steps to baseline accuracy on The Pile, plus +6.5 downstream points. The transfer claim is striking because in principle α solves a minimax tied to the 280M model's specific loss surface.

The paper's interpretation (and the one that has held up): excess loss measures *learnability-deficit relative to a reference*, and learnability ordering across domains is largely scale-invariant. Domains that are hard for 280M relative to its uniform-weights sibling are also hard for 8B relative to its uniform-weights sibling. The *absolute* losses change with scale; the *relative* per-domain difficulty ordering does not. Since EG only reads the ordering (via softmax), α transfers.

This is a conjecture, not a theorem. It breaks when the reference and proxy sit in different capacity regimes — a 50M reference with an 8B proxy produces unstable α because the reference is too undercooked to establish a meaningful baseline. DoReMi's default (`ref` and `proxy` same size, ~280M) is empirically the safe operating point.

### 2.4 The Pareto-improvement surprise

The most counterintuitive empirical result in [[doremi]]: on The Pile, DoReMi's α improves per-domain perplexity on *every one* of the 22 domains simultaneously — including domains whose weight was reduced. Naively, downweighting a domain should hurt its perplexity. It does not, for two compounding reasons.

First, many Pile domains are partially redundant: downweighting "Common Crawl" but upweighting "OpenWebText2" moves tokens between near-duplicate distributions. Per-domain loss on the downweighted domain barely moves because the model still sees equivalent text under another domain label. Second, upweighting a hard domain (e.g., math-heavy arXiv) produces representations that transfer positively back to easier domains — a capacity-reallocation effect that dominates the direct-exposure effect for overlapping-support domains.

The Pareto result is evidence that "the fixed mix" left Pareto-dominated mass on the table. It is also the strongest argument against intuition-based mixing: humans would never propose "downweight Wikipedia to improve Wikipedia perplexity," but that is what DoReMi discovers when Wikipedia sits below its minimax-efficient weight.

---

## 3. Alternatives — DSIR, downstream-reweighting, and gradient similarity

DoReMi is not the only principled mixer. Three alternatives are worth internalizing because each optimizes a different objective.

**Comparison.**

| Approach | Optimizes for | Needs downstream tasks? | Compute cost | Unit of selection |
|---|---|---|---|---|
| Fixed mix (GLaM/PaLM) | hand-set proxy for downstream | yes (via sweep) | O(grid · full run) | domain |
| Downstream-reweight | explicit downstream benchmark | yes | O(sweep) | domain |
| DoReMi | worst-case excess loss | **no** | O(1 proxy run, 30× smaller) | domain |
| DSIR (Xie et al. 2023) | KL-proximity to target distribution | yes (need target corpus) | O(hashing pass) | document |
| LESS ([[less]]) | gradient-alignment with target exemplars | yes (few-shot target set) | O(warmup + datastore) | example |

**DSIR (Data Selection via Importance Resampling).** Given a *target* distribution (e.g., books + Wikipedia as "high-quality reference"), score each pool document by an importance-weight approximation `p_target(x) / p_source(x)` computed via n-gram hash-feature ratios, then resample with replacement proportional to the weights. DSIR is fast (a hashing pass, no model training) and is the right tool when you know the target distribution in advance. It does *not* solve the "which mix is best for downstream reasoning" problem because that problem has no single target corpus.

**Upweight-by-downstream-accuracy.** Run small probe models on candidate mixes, measure downstream accuracy per capability benchmark, regress benchmark score against α, pick the α that the regression prefers. This is GLaM's recipe made into a method. It is the *honest* version of "we tuned on downstream" — and it is what Llama 3 and DeepSeek-V3 implicitly do when they report category-conditional pretraining subsets.

**LESS — gradient-alignment selection.** [[less]] is a per-example selector for *post-training*: build a reusable low-rank per-sample gradient datastore, rank against a small target few-shot set by cosine similarity, keep top 5%. LESS answers "which SFT examples should I train on given I want MMLU to improve"; DoReMi answers "which domains at what weights should I pretrain on given I want worst-case domain loss minimized." Different stage, different objective, different granularity — but both use *the loss/gradient signal of a small proxy* as the selector. See the figure `figures/doremi-reweighting.html` for the DoReMi-side dynamics.

**Which to reach for.** If you are setting pretraining mix → DoReMi. If you are selecting instruction-tune data for a targeted capability → LESS. If you have a clean reference corpus and want to resample web → DSIR. If you are tuning at frontier scale and have compute for sweeps → downstream-reweight.

---

## 4. The mix is stage-specific

The single most common beginner mistake is to treat "the data mix" as one object. It is four, and they have disjoint objectives.

| Stage | Objective | Typical α shape | Example |
|---|---|---|---|
| Pretraining | broad coverage, worst-case domain loss | long-tailed: web ~ 40–60%, code 10–20%, books 5–10%, arXiv 2–5%, rest spread | [[olmo-2]] OLMo-Mix-1124 (3.9T) |
| Mid-training / cooldown | install reusable priors for later stages; sharpen weak capabilities | concentrated: math 20–40%, code 20–40%, high-quality web 20%, IFT-like 10–20% | [[olmo-2]] Dolmino (~50B); [[olmo-3]] Dolma 3 Dolmino (100B) |
| SFT | instruction-following surface form + capability coverage | capability-bucketed: code, math, tool use, multilingual each 10–20% | [[llama-3]] 6-round SFT mix; Tulu 3 mix on OLMo 2 |
| RLVR prompt distribution | edge-of-competence tasks with verifiable rewards | narrow: verifier-friendly math, code, IFEval | OLMo 2 / Tulu 3 RLVR prompts |

The [[interplay-pretraining-midtraining-rl]] paper argues this explicitly: mid-training is a distinct stage with its own mix, and under fixed compute a well-tuned mid-training mix can beat additional RL. OLMo 3 makes this the organizing principle of its release — four named corpora (Dolma 3 Mix, Dolmino, Longmino, Dolci) for four stages.

The DoReMi procedure can in principle run at any stage. In practice it is most-used at pretraining (where domain structure is clean). For SFT, LESS/IFD-style per-example selectors beat any per-domain mix because "SFT domains" are conceptually weak (what is the "domain" of a function-calling trace?).

**A warning on repetition.** Upweighting a scarce domain is implicit repetition. [[data-constrained-scaling]]'s ~4-epoch threshold still applies: if DoReMi's α pushes a small domain to effective 8 epochs at your token budget, you are in data-constrained regime for *that domain only*, and its marginal return is flattening. Check per-domain effective epochs against α before locking in.

---

## 5. What Llama 3 and OLMo 2 actually do — and do not report

### 5.1 Llama 3 — bucketed capability data, undisclosed weights

From [[llama-3]]:

- Pretraining: 15.6T tokens; the mix is described in prose, not as a weight vector. The paper names buckets (web, code, math, reasoning, multilingual, long-context) but does not publish `α`. The reader can infer *ordering* but not *magnitudes*.
- Post-training mix: ~50–80% synthetic rejection-sampled data per round, with per-capability synthetic pipelines (code/math/multilingual/long-context/tool use/factuality). This is a *capability mix*, not a domain mix, and its weights are again undisclosed.
- The six-round iterative SFT → RS → DPO structure means the "mix" is *dynamic*: round `k`'s SFT mix is generated by round `k-1`'s best policy. Reporting a single α is conceptually inadequate.

What Llama 3 hides: the actual α vector at any stage, the per-round re-weighting decisions, and the data-pipeline-level filters that gate each capability bucket.

### 5.2 OLMo 2 — two-stage curriculum, open weights

From [[olmo-2]]:

- Stage 1 — OLMo-Mix-1124 (~3.9T tokens): DCLM + Dolma 1.7 + Starcoder + Proof Pile II. Published mix proportions.
- Stage 2 — Dolmino cooldown (~50B tokens): curated higher-quality subset during LR decay. Explicit stage separation encodes the pretrain-mix ≠ mid-train-mix distinction.
- Post-training inherits Tulu 3's SFT/DPO/RLVR mixes wholesale.

OLMo 2 publishes the corpora. Whether those mixes are DoReMi-derived, hand-tuned, or intuition-set is less explicit; the release reads as "hand-calibrated, ablation-validated."

### 5.3 OLMo 3 — four named mixes for four stages

[[olmo-3]] is the cleanest public demonstration that mix is stage-specific: Dolma 3 Mix (~6T pretrain), Dolmino (100B mid-train), Longmino (50B long-context), Dolci (SFT/DPO/RLVR). The team releases every stage's mix composition, every intermediate checkpoint, and the transition decisions — the "model flow" framing.

### 5.4 DeepSeek-V3 — the closed-mix counterpoint

[[deepseek-v3]] reports 14.8T "diverse, high-quality" pretraining tokens. No domain decomposition, no α, no filter ablation. Pretraining cost is reported precisely (2.664M H800 hours) but the mix that consumed that compute is prose. This is the current industry default when a lab decides not to publish.

The gap between OLMo 3 (every stage's mix listed) and DeepSeek-V3 (prose only) is the current spread in practice. For research purposes, assume a frontier lab's mix is hand-calibrated by downstream eval on held-out benchmarks, re-tuned when new capabilities become priorities, and often quietly re-weighted mid-run.

---

## 6. Operational checklist for mixing

1. Before any DoReMi run, verify each domain contributes ≥ 1% of the pool — smaller slices are noise-dominated in the EG updates.
2. Train the **reference model** with uniform domain weights. Do not skip this. A task-tuned reference makes DoReMi compete against an already-biased baseline; the excess-loss signal degrades.
3. Set EG step `η` in the range 0.05–1.0; DoReMi defaults to ~0.3. Too small and α never moves; too large and the adversary oscillates wildly and the time-average is meaningless.
4. Train the proxy for the same total tokens as one Chinchilla-equivalent small run — for a 280M model, this is roughly 5.6B tokens. DoReMi weights stabilize well before the proxy converges.
5. Take the **time-average** `ᾱ` (optionally averaged over the last `T/2` steps, discarding the burn-in), not the last iterate.
6. Resample the full pretraining corpus once with `ᾱ` and train the production model with standard uniform-within-domain sampling. Do **not** run DoReMi online on the production model — the 30× transfer is the entire point.
7. Audit per-domain effective epoch count against the ~4-epoch threshold; if any domain exceeds, downweight in post or expand the pool with [[dsir]]-style reuse.
8. At mid-training, re-derive a fresh α for the cooldown pool — the pretraining α is wrong for mid-training because the objective has changed.

---

## 7. What the next chapter builds on

Ch-14 takes mixing's budget-of-tokens-per-domain as input and asks two questions: (1) what happens when the total token budget meets the ~4-epoch repetition ceiling, and (2) how does evaluation data leak into the mix via contamination. Mixing is the lever; scaling laws and contamination are the constraints that bound which mixes are honest.

---

## Connections

- [[doremi]] — Xie et al. 2023; the group-DRO minimax construction, the 30× proxy-to-production transfer, the Pareto per-domain improvement result.
- [[less]] — Xia et al. 2024; gradient-similarity per-example selection for SFT, complementary to DoReMi's per-domain reweighting.
- [[scaling-laws-data-quality]] — the quality term that makes "mix" mean "mix weighted by per-domain effective sample size."
- [[data-constrained-scaling]] — the ~4-epoch threshold that bounds how far α can push a scarce domain.
- [[interplay-pretraining-midtraining-rl]] — mid-training as a distinct stage with its own mix objective.
- [[llama-3]] — what is disclosed (bucket names, 15.6T) and what is not (α).
- [[olmo-2]] — two-stage curriculum with published per-stage corpora.
- [[olmo-3]] — four-stage model-flow with four named mixes, the cleanest public example.
- [[deepseek-v3]] — closed-mix counterpoint; 14.8T tokens, no decomposition.
- `figures/doremi-reweighting.html` — interactive DoReMi convergence animation with adjustable EG step `η`.
