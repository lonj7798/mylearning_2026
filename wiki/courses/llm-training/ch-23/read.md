<!-- chapter: ch-23
     track: synthetic
     title: Model Collapse and Synthetic-Data Verification
     sources: [[model-collapse]], [[strong-model-collapse]], [[faithful-synth-eval]], [[synthetic-data-scaling-laws]], [[prismatic-synthesis]], [[nemotron-4-synthetic]], [[apigen]], [[apigen-mt]]
     figures: figures/collapse-iterations.html
-->

# Chapter 23 — Model Collapse and Synthetic-Data Verification

> **Core insight.** When a generative model is retrained on samples from its own previous generation, the distribution contracts: tails vanish first, then the body; by generation ~9 on OPT-125M, outputs are incoherent ([[model-collapse]]). The mechanism is statistical — finite-sample resampling erases rare events — so it holds for Gaussian mixtures, VAEs, and LLMs alike. Dohmatob et al. sharpen this to a scaling-law statement: even **1% synthetic contamination** introduces an irreducible bias term `c(p)·σ²` that eliminates the test-error benefit of larger data ([[strong-model-collapse]]). Collapse is therefore not a pathology of pure-recursive toy setups; it is the default outcome of any synthetic pipeline that lacks a distribution-preserving gate.
>
> **Guideline.** Never replace real data with synthetic — **accumulate** on a persistent human anchor, or place a **verifier** between every generator and the next training round. Rephrased synthetic survives scaling up to ~30% share; pure-generated synthetic does not. Average loss / perplexity hides the failure — audit tail recall, embedding-cluster occupancy, and gradient-space coverage ([[faithful-synth-eval]], [[prismatic-synthesis]]) directly. Every production pipeline that works at scale — Nemotron-4's RM-as-judge ([[nemotron-4-synthetic]]), APIGen's format→execution→semantic stack ([[apigen]]) — is built around a gate, not around faith in the generator.

---

## Why this chapter exists

Chapters 19–22 gave you the machinery to *make* synthetic data: Self-Instruct-style prompt seeds, UltraFeedback-style preference pairs, Magpie extraction, Genetic-Instruct population search, gradient-coverage selection (Prismatic, ch-22). A naïve reading of that toolkit suggests an obvious next move — close the loop: train a model, use it to generate more data, retrain, repeat. Iterated self-improvement was the implicit promise of the 2023 "LLM-as-a-flywheel" frame.

Shumailov et al. (Nature 2024) proved, formally and empirically, that this loop is broken ([[model-collapse]]). The distribution of the model's outputs progressively loses tail mass and converges to a mode-collapsed near-Gaussian regardless of architecture. Within ~5 generations of OPT-125M recursive finetuning, rare-token perplexity spikes while average perplexity looks fine — the failure is invisible on the dashboard. By ~9 generations, outputs are degenerate. The finding survived scrutiny: Dohmatob et al. at ICLR 2025 Spotlight showed the same mechanism breaks **scaling laws** at any non-zero synthetic fraction ([[strong-model-collapse]]). The 2025 follow-ups (Gerstgrasser, Zhu, He, Garg) identified the two escape routes — **accumulate-don't-replace** and **verify-before-ingest** — and a companion line (Zhang et al. 2025, "Closer Look" 2025) gave analytical convergence guarantees *only* when an external verifier gates the loop ([[faithful-synth-eval]]).

The chapter's organizing claim: model collapse is not a warning, it is a forcing function. Every synthetic pipeline you write — for reasoning traces (ch-24), conversations (ch-25), tool calls (ch-26), preference pairs (ch-27) — must satisfy one of two structural invariants before you can trust its output at scale. Either (i) a fixed, large, human-anchored real-data mass dominates and is never removed, or (ii) every accepted sample passes an **external verifier** whose decision is independent of the generator. "External" is the load-bearing word. A verifier that shares weights with the generator inherits its blind spots; the loop closes and the collapse resumes.

The rest of the chapter unpacks the mechanism (§1), the sharpened scaling-law statement (§2), the mitigability boundary (§3), the 2025 verification toolkit (§4), a gate-vs-no-gate comparison table (§5), and the canonical gate designs — RM-as-judge, 3-layer verify — that production pipelines use (§6).

---

## 1. The Shumailov mechanism — three error sources, compounded per generation

The [[model-collapse]] paper isolates **three error sources** that together drive the iterated distribution toward a mode-collapsed limit.

1. **Statistical sampling error.** The finite-sample Monte Carlo estimator of any distribution loses rare-event mass in expectation. If a token has true frequency `1/N`, a sample of size `N` contains it in expectation but with high variance; smaller samples miss it entirely. Rare events are erased first.
2. **Functional expressivity error.** Real distributions live outside any finite-parameter model class. Each generation's refitted model projects the previous generation's empirical samples onto its own restricted manifold, losing mass in directions it cannot represent.
3. **Functional approximation error.** Optimization is inexact. Each generation's training adds a stochastic perturbation whose tails compound.

**Gaussian-mixture collapse — the proof sketch.** For tractability, suppose the real distribution is a Gaussian mixture `p_0 = Σ w_k N(μ_k, σ²)`. Sample `N` points from `p_0`; refit a mixture on those samples, call it `p_1`. Sample again from `p_1`; refit; repeat. Each refit is a maximum-likelihood estimate of `(w_k, μ_k, σ²)` from a finite sample. The key observation is that variance in the estimated mixture weights `ŵ_k` grows linearly with generation count: if the true weight is `w_k` and the sample size is `N`,

```
Var[ŵ_k^{(n+1)}] ≈ Var[ŵ_k^{(n)}] + w_k(1 - w_k) / N
```

so after `n` generations, `Var[ŵ_k^{(n)}] ≈ n · w_k(1 - w_k) / N`. A component with true weight `w_k ≈ 0.01` has variance at generation `n` of roughly `0.01·n/N`; after `n ≈ 100/N·(N/100) ≈ O(N/100)` generations, the estimated weight has non-trivial probability of hitting zero. Once a component's weight goes to zero in any generation, it **cannot be recovered** — the sampler never produces a point from that component again, so subsequent generations have no signal to re-inflate it. Tail modes die first and permanently. This is the mechanism.

**`k`-th moment error (condensed).** The paper's compact expression for the compounded error on the `k`-th moment of the empirical distribution:

```
Var[μ_k^{(n)}] ≈ n · σ² / N + O(model error)
```

Sampling variance **accumulates linearly in generation count `n`**; tails are erased first because their sampled mass vanishes fastest (rare events have the smallest `N_k` contribution).

**Tail-loss progression — what you actually see.** Across the OPT-125M recursive-finetuning experiment, tracking tail-token perplexity by generation yields roughly this progression (the paper's Figure 3-class numbers, condensed):

| Generation | Average PPL (wikitext2 held-out) | Rare-token PPL | Qualitative output |
|---|---|---|---|
| 0 (real) | 34.1 | 412 | coherent, long-tail vocabulary present |
| 1 | 33.8 | 447 | coherent, rare tokens slightly depressed |
| 3 | 33.2 | 612 | coherent; noticeable topic compression |
| 5 | 32.9 | 1,104 | still-coherent sentences; repeated phrasings |
| 7 | 32.1 | 2,890 | repetition artifacts; rare-token events effectively gone |
| 9 | 31.4 | >10⁴ | incoherent; mode-collapsed boilerplate |

The trap: **average PPL improves** generation-to-generation while tails disappear. If your dashboard only shows mean loss, the collapse looks like convergence. This is the single most important operational lesson from the paper — **mean loss is a collapsing quantity, not a signal of health** — and it justifies every tail-metric the rest of the chapter insists on.

**LLM experiments — what Shumailov actually ran.** OPT-125M fine-tuned on wikitext2; generate 100k tokens with temperature 1.0; replace training data with generations (optionally mixed with a fraction of real); re-fine-tune; iterate up to 10 generations. No external filtering. Two variants studied: pure replacement (10% real = 0, the stark curve) and accumulation (10% real persistent, tail loss bounded but still present). The paper deliberately studies the unfiltered case to isolate the statistical mechanism from confounds of judge quality.

---

## 2. Strong model collapse — scaling-law breakdown at 1%

[[strong-model-collapse]] (Dohmatob, Feng, Subramonian, Kempe — ICLR 2025 Spotlight) tightens Shumailov from "iterated replacement breaks models" to "any contamination breaks scaling." The setting is the modern scaling-law regime: train-test error is tracked as training set size `N` grows, under standard real-data assumptions `E[R_test] ~ f(N)` decreases with `N`. Under synthetic fraction `p > 0`, the paper proves (in a random-projection approximation of deep nets, using operator-valued free probability) that

```
E[R_test] ≈ f(N) + c(p) · σ_synth²
```

with `c(p) > 0` for any `p > 0`. The asymptote is now a function of `p`, not `N`. Scaling flatlines. Empirically reproduced on GPT-2-scale LM training with 1% synthetic injection: the scaling curve departs from the real-only baseline by roughly gen-2 sample-efficiency and never recovers.

The phase-diagram result layers a second effect on top: model size modulates but does not eliminate collapse. Below the interpolation threshold (under-parameterized regime) larger models *amplify* collapse; beyond it, larger models *partially mitigate* but never eliminate it. This is the opposite of the hopeful intuition that "just scaling up" would wash out a 1% contaminant. It does not.

**The practical policy implication** (the paper does not state it this way but every reader arrives there): as the open web accumulates LLM-generated text, *all* future pretraining corpora are contaminated. "Some synthetic is fine" is mathematically fragile — the question becomes how contaminated, and whether the pipeline has a verifier that turns synthetic back into "effectively real" in the limit. The theoretical assumption in Strong Model Collapse is precisely that synthetic is iid from an earlier model; a *verified* corpus in which every sample has passed an external check does not satisfy that assumption and the theory's pessimism does not apply. This is the bridge from §2 (the problem) to §4 (the defense).

---

## 3. Mitigable vs unavoidable — the structural boundary

Whether collapse is inevitable depends on the structure of the loop, not on the specific data or model. The 2025 literature has a clean taxonomy.

**Mitigable regimes.**

- **Fresh-human injection (Gerstgrasser et al. 2024, "Is Model Collapse Inevitable? ... by Accumulating Real and Synthetic Data").** If each generation's training set is `real ∪ synthetic` rather than pure synthetic — and the real set is persistent (not rotated out) — the error term stays bounded. Test error does not improve with more synthetic, but it does not *degrade* either. The practical lesson: treat your human-annotated corpus as a **permanent anchor**, never a step in a pipeline that gets consumed.
- **Token-level re-sampling (Zhu et al. 2025).** Even in LM generation, re-sampling at the token level from the true distribution (rather than the model's) at every decoding step avoids collapse in an analytical linear-regression proof and an empirical LM study. Mechanism: the re-sampling operator inserts true-distribution mass back into the chain at every step.
- **Optimal mixing ratios (He et al. 2025; Garg et al. 2025).** These papers derive closed-form optimal real:synthetic ratios as a function of relative data quality and budget. The punchline: the optimum is *interior* — neither all-real nor all-synthetic — and it depends on how much the verifier can eliminate synthetic's variance contribution. For rephrased-style synthetic with a strong verifier, the empirical optimum converges near 2:1 real:synthetic (≈30% synthetic share) — matching the SynthLLM / Demystifying-Synthetic-Data / BeyondWeb scaling-law observations ([[synthetic-data-scaling-laws]]).
- **External verification (Zhang et al. 2025, "Escaping Model Collapse via Synthetic Data Verification," arxiv 2510.16657).** An external verifier — a stronger model or a rule-based judge independent of the generator — breaks the loop. The paper gives both an analytical convergence guarantee (iterated training with a reliable verifier stays bounded) and empirical evidence on LLM text generation. This is the theoretical basis for the 2025 faithfulness-check toolkit (§4).
- **Rephrasing, not pure generation ([[synthetic-data-scaling-laws]]).** WRAP-style rephrasing of real documents is *not* recursive self-iteration — each output is anchored to a specific real source and rewritten, so the loss channel is paraphrase distortion, not mass-contraction over many generations. Demystifying-Synthetic-Data (EMNLP 2025) shows rephrased synthetic follows clean rectified scaling up to measured scales; pure-generated textbook synthetic shows model-collapse-predicted degradation. "Synthetic" is not a monolith.

**Unavoidable regimes.**

- **Closed feedback loops without an external filter.** Any pipeline where the next generation's training data comes entirely from the previous generation's outputs, with no external signal, collapses. Self-Instruct-from-self without a judge. RL with a reward model that was itself trained only on the generator's outputs. DPO on preference pairs where both chosen and rejected come from the same policy with no independent signal.
- **Reward-model staleness ([[nemotron-4-synthetic]] gotchas).** "Reward-model errors compound when the same scorer is reused across iterations." If the RM is trained once and used as judge for many generator iterations, the generator overfits the RM's blind spots — this is a soft collapse where the *reward landscape*, not the output distribution, degenerates. The fix in Nemotron-4 is periodic RM refresh from fresh human preferences (HelpSteer2 anchor).
- **Shared-weight verification.** A "verifier" that is a checkpoint of the same model that produced the candidates is not external. It cannot flag a blind spot it shares. The Zhang et al. 2025 convergence guarantee requires the verifier to be independent of the generator — the gate closes only when the judge can see failures the generator cannot.

The structural question to ask before designing any synthetic pipeline: **is there a signal in my loop that is independent of the generator and capable of detecting generator drift?** If yes, design the gate and you are in the mitigable regime. If no, you are in the unavoidable regime and the only question is how many generations you have before the collapse becomes visible.

---

## 4. Verification as defense — the 2025 faithfulness-check protocols

[[faithful-synth-eval]] consolidates the 2024–2025 verification literature into four complementary audit axes. None is sufficient alone; together they form the minimum gate for a production synthetic pipeline.

**Axis 1 — Tail-mass measurements.** The failure that Shumailov's OPT-125M table made visible (§1) is invisible in mean PPL and obvious in tail PPL. Concrete measurements:

- **Rare-token recall.** Fix a real reference corpus; identify tokens with frequency `< 10⁻⁴`; compute the fraction of these tokens the synthetic corpus produces at comparable frequency. Collapse shows up as <50% recall after 3–5 generations.
- **Rare n-gram overlap.** Same, but for 3- and 5-grams. A sharper signal for mode collapse because rare phrases die faster than rare tokens.
- **Rare-concept recall.** LLM-tagged categorical entities (named entities, scientific terminologies, minority languages) — are long-tail ones preserved? This is the closest proxy to the kind of semantic tail that frontier-model evals care about.

**Axis 2 — External verification.** The convergence-guaranteeing filter from Zhang et al. 2025. Concrete by modality:

- **Math:** ground-truth answer matcher (OpenMathInstruct, rStar-Math). Deterministic; 0% false-positive rate; rejects ~60% of raw generations at temperature 1.0.
- **Code:** unit-test execution (APIGen, CodeUltraFeedback). Sandboxed runtime; 5-second timeout per test.
- **Tool / function calls:** the [[apigen]] 3-layer stack — format → execution → semantic. Ablation: removing any one layer costs 6–18% BFCL accuracy (§6).
- **Factual / RAG:** retrieval-grounded NLI entailment. The claim must be entailed by retrieved evidence; unsupported claims are rejected.
- **Open-ended text:** reward-model-as-judge ([[nemotron-4-synthetic]] Nemotron-4-340B-Reward). Least reliable; RM blind spots are the dominant failure mode.

The **externality** requirement is not negotiable. The verifier's model weights should come from a different training lineage than the generator (ideally different organization, different base model, different preference dataset); rule-based verifiers (answer matcher, unit test, schema validator) are preferred because they are maximally external.

**Axis 3 — Coverage / diversity metrics.** Complements tail recall by measuring whether the synthetic corpus fills the *gradient space* of a proxy model rather than clustering in a few high-density basins.

- **G-Vendi ([[prismatic-synthesis]]).** For each candidate `x_i`, compute normalized gradient `g_i = ∇_θ L(x_i; θ)` on a small proxy LM, random-project to ~8K dims, build density matrix `K_{ij} = <g_i, g_j>`, return `exp(vN-entropy(K/tr(K)))`. Spearman ρ ≈ 0.9 with OOD accuracy across 300+ runs; **beats** embedding-Vendi (with encoder 14× larger) and GPT-4-based Skill-Set Entropy.
- **Embedding-cluster occupancy.** Count distinct embedding clusters populated by the synthetic corpus (k-means on a reference encoder, typical `k` = 1000). Mode collapse shows up as cluster count dropping while corpus size grows.
- **kNN diversity.** Average kNN distance in embedding space; collapses with rare-concept loss.

**Axis 4 — Drift-over-iteration signals.** If iteration is unavoidable, monitor the drift:

- Δ(rare-token recall) between round `k` and round `k+1` > 5% → early warning.
- Δ(G-Vendi) between rounds > 10% → active collapse.
- Δ(embedding-cluster occupancy) > 20% over 3 rounds → mode collapse in progress.

**Memorization↔generalization drift ("Closer Look at Model Collapse," 2025).** As synthetic fraction grows, models shift toward memorization-heavy regimes surface metrics miss. Operational test: memorization probes (exact-match recall of training strings) should **not** rise with synthetic fraction; if they do, the model is collapsing into its training data — the synthetic-fraction analog of Gaussian-mixture mode contraction.

---

## 5. Gate vs. no-gate — the comparison

The single table the rest of the course refers back to. Empirical numbers are aggregated from [[model-collapse]] (OPT-125M, 10 generations), [[strong-model-collapse]] (GPT-2 scale, scaling law), [[synthetic-data-scaling-laws]] (8B rephrased vs textbook), [[nemotron-4-synthetic]] (340B alignment), [[apigen]] (7B function-calling), and Zhang et al. 2025.

| Pipeline | Real anchor | Synthetic fraction | Verifier gate | Observed collapse | Notes |
|---|---|---|---|---|---|
| Pure recursive replacement (Shumailov 2024 baseline) | 0% | 100% | none | incoherent by gen-9; tail-PPL >10⁴ | the reference worst-case |
| Shumailov 10%-real-accumulation | 10%, persistent | 90%/gen | none | tails bounded; ~30% worse than real-only | accumulation alone bounds error |
| Open-web mix with 1% LLM contamination (Dohmatob empirical) | 99% | 1% | none | scaling-law flatline; no benefit from larger `N` | the policy-relevant regime |
| WRAP / rephrased-synthetic pretraining (~30%) | 70% | ~30% rephrased | external paraphrase constraint | clean rectified scaling to measured scales; 5–10× speedup | rephrasing *is* a gate (anchors to real source) |
| Pure-generated "textbook" synthetic at high fraction | varies | >50% textbook | none / weak | collapses as predicted by Shumailov | the Phi-cautionary regime |
| Nemotron-4 alignment (98% synthetic) | ~20K human (HelpSteer2) | 98% | Nemotron-4-340B-Reward + category-seeded prompts | no collapse; SOTA RewardBench | RM-as-judge, periodic refresh from human anchor |
| APIGen function-calling (60K) | 0% human | 100% synthetic | 3-layer (format → execution → semantic) | no collapse; xLAM-7B #1 BFCL <13B | external, rule-heavy gate; ablation removes any layer → 6–18% loss |
| Zhang et al. 2025 verified loop | variable | variable | external verifier | bounded convergence (analytical + empirical) | verification suffices |
| Gradient-targeted synthesis (Prismatic) | 0% human | 100% synthetic | G-Vendi coverage + answer verifier | no collapse; 7B beats 671B teacher | generate *off* teacher's gradient manifold; coverage is a gate |

The pattern: **every pipeline that scales without collapsing has a gate.** The gate takes different forms — rule-based execution (APIGen), reward-model judgment (Nemotron-4), paraphrase anchoring (WRAP), gradient-coverage selection (Prismatic), fresh-human accumulation (Gerstgrasser) — but a gate is always present. Pipelines without gates collapse on the timescale of 3–10 generations regardless of model scale.

---

## 6. Canonical gate designs — what to build

Two gate templates cover most of what you will need. Both are patterns, not libraries; the implementation details live in the modality-specific chapters (24–27) but the invariants are the same.

**Gate template A — RM-as-judge ([[nemotron-4-synthetic]]).** For open-ended text where rule-based verification is infeasible. Structure:

```
human_anchor:      20K human preferences (HelpSteer2 class)
                       │
                       ▼
reward_model = train(human_anchor)          # Nemotron-4-340B-Reward
                       │
generator_v0 ──► candidates  ──► RM.score(c) ≥ τ  ──► SFT set_v0
                       │
generator_v1 = finetune(generator_v0, SFT set_v0)
                       │
generator_v1 ──► preferences (chosen, rejected) chosen via RM
                       │
generator_v2 = DPO(generator_v1, preferences)  then RPO with RM-reweighting

# Every N iterations, retrain RM on fresh human preferences (anti-stale)
reward_model = train(human_anchor_fresh ∪ human_anchor)
```

Critical invariants:
- Human anchor never consumed; always additive.
- RM refreshed on a cadence (Nemotron-4 refreshes per-generation on the HelpSteer2 anchor).
- RM never scores its own training data.
- Acceptance threshold `τ` set by held-out human preference agreement rate (Nemotron-4 reports τ tuned to ~80% agreement with human judges).

Failure mode: RM drift. If you reuse the same RM checkpoint across many generator generations, the generator Goodhart's-laws the RM — learns to produce high-RM-score outputs that are not actually higher quality. The refresh cadence is what breaks this.

**Gate template B — 3-layer rule-heavy verification ([[apigen]]).** For modalities where rule-based ground truth exists (code, math, tool calls, structured outputs):

| Layer | Check | Rejection rate | Cost |
|---|---|---|---|
| 1. FORMAT | parse JSON; required params; type/enum match schema | ~15–25% of raw | ~1ms |
| 2. EXECUTION | run in 5-sec sandbox; exception or timeout → reject | ~10–15% of L1-passed | ~5s |
| 3. SEMANTIC | LLM-judge (external): "does call fulfill query given result?" | ~10% of L2-passed | ~1 teacher call |

Overall acceptance ≈ 60% of raw generations.

APIGen's ablation is the load-bearing evidence that all three layers matter:
- Remove format: −18% BFCL-V1.
- Remove execution: −11% BFCL-V1.
- Remove semantic: −6% BFCL-V1.

Removing any layer lets a specific failure class through. Format-only catches schema errors but lets hallucinated arguments through. Format+execution catches runtime errors but lets semantically-wrong-but-runnable calls through (e.g., wrong unit to a conversion function). All three together give the <3% hallucination rate xLAM-7B reports on BFCL-V1.

**Why the layers stack multiplicatively.** The three layers are approximately independent — a sample can fail format without failing execution (because you'd never try), can fail execution without failing semantics (timeouts happen to correct calls occasionally), can fail semantics without failing execution (wrong answer that ran). The acceptance rates multiply: `0.80 × 0.85 × 0.90 ≈ 0.61`. This is also why a single-layer "LLM-as-judge" gate is weaker than it looks — it is the third layer without the first two, and the first two eliminate roughly half of raw generations.

**A minimal reference gate for a new modality.** If you're designing a synthetic pipeline for a new modality (say, multi-turn tool-use trajectories — see [[apigen-mt]]), the template is:

```python
def accept_candidate(candidate, ground_truth=None, judge=None) -> bool:
    # Layer 1 — structural: parse + schema + type check
    try: parsed = parse_and_validate(candidate)
    except (JSONDecodeError, SchemaError): return False
    # Layer 2 — executable: sandbox with 5s timeout; match ground truth if any
    result = None
    if has_executable_ground_truth(parsed):
        try: result = sandbox_execute(parsed, timeout_s=5)
        except (Timeout, Exception): return False
        if ground_truth is not None and result != ground_truth: return False
    # Layer 3 — semantic: external judge independent of generator
    if judge is not None and judge.score(candidate, parsed, result) != "Yes":
        return False
    return True
```

Wrap with a build loop that targets ~60% acceptance (APIGen ratio) and reports `len(accepted) / raw_generated` per generation — outside that band, the gate is misconfigured. With modality-specific `parse_and_validate` / `sandbox_execute` / `judge`, this skeleton is the shape of every production synthetic pipeline that has been published as not collapsing; the next three chapters fill in the modality-specific pieces on top of this gate.

---

## 7. What the dashboard must show

You cannot trust averages. The minimum set of quantities to log for any synthetic pipeline, per iteration:

| Metric | Cadence | Threshold for action |
|---|---|---|
| Raw acceptance rate (gate pass %) | per generation | Δ > 10% from previous gen → investigate gate or generator |
| Rare-token recall vs real reference | per generation | <75% → early collapse signal |
| Rare 5-gram recall vs real reference | per generation | <50% → mode collapse starting |
| Embedding-cluster occupancy (k=1000) | per generation | Δ > 20% over 3 gens → active collapse |
| G-Vendi on a 10k sample | per generation | Δ > 10% → gradient-space contraction |
| Mean loss on a fixed real held-out set | per generation | **auxiliary only** — never the primary signal |
| Memorization probe (exact-match recall of training strings) | per generation | rising with synthetic fraction → collapse into memorization regime |
| Verifier disagreement rate on a held-out human-labeled set | per N iterations | rising → RM drift; refresh |

Mean PPL sits at the bottom on purpose. It moves the *wrong way* under collapse (down, because mode-collapsed outputs are self-consistent) — the exact signal that convinced Shumailov-era researchers their models were improving while they were in fact collapsing. Log it, but never trust it alone.

---

## Connections and what's next

- **[[model-collapse]] / §1** — Shumailov Nature 2024: mechanism + OPT-125M. **[[strong-model-collapse]] / §2** — Dohmatob ICLR 2025: scaling flatline at 1%. **[[faithful-synth-eval]] / §4** — the 2025 verification cluster (tail / external / coverage / drift axes).
- **[[synthetic-data-scaling-laws]] / §3, §5** — rephrased vs pure-generated under scaling. **[[prismatic-synthesis]] / §4 Axis 3, §5** — G-Vendi; gradient-targeted generation as anti-collapse mechanism.
- **[[nemotron-4-synthetic]] / §6 Template A** — 98%-synthetic production pipeline saved by RM gate + HelpSteer2 anchor. **[[apigen]] / §6 Template B** — 3-layer rule-heavy verification; each layer ablation-confirmed load-bearing. **[[apigen-mt]]** — multi-turn extension of the same gate.
- **ch-22** — upstream filter (Quality / Diversity / Gradient-Based Selection). **ch-24–27** — modality applications (reasoning traces, conversation, tool calls, preference pairs) all build on this chapter's gate templates.

## Further reading

- [[model-collapse]] — Nature 2024; read the OPT-125M figures and the Gaussian-mixture proof.
- [[strong-model-collapse]] — ICLR 2025 Spotlight; Theorem 1 + 1% empirical reproduction.
- [[faithful-synth-eval]] — Zhang 2510.16657 is the convergence-guarantee anchor.
- [[synthetic-data-scaling-laws]] — SynthLLM + Demystifying + BeyondWeb (2025).
- [[prismatic-synthesis]] — Jung 2025 (2505.20161); G-Vendi definition and 7B-beats-671B result.
- [[nemotron-4-synthetic]] / [[apigen]] — production templates for Gates A and B respectively.

## Companion visualization

**[figures/collapse-iterations.html](figures/collapse-iterations.html)** — interactive Gaussian-mixture collapse across generations 0→10 of recursive training. Sliders control (a) fresh-human injection fraction (0–100% anchor persistence), (b) verifier pass-rate (probability that an off-distribution sample is rejected before it enters the next generation's training set), and (c) sample size `N` per generation. The left panel plots the density at each generation, colored from dark (gen 0) to light (gen 10), showing tails collapsing under pure self-iteration and tails preserved under either real-data anchoring or an external verifier gate. A side panel tracks two quantities generation-by-generation: **rare-mode recall** (fraction of low-weight components with non-zero estimated weight) and **KL to the true distribution**. Start at default (0% anchor, 0% verifier) to reproduce Shumailov's curve; raise the verifier slider to observe the Zhang et al. 2025 convergence guarantee in action.
