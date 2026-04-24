<!-- chapter: ch-48
     track: eval
     kind: content
     title: Contamination Workflow
     deps: [ch-47]
     sources: [[deduplicating-training-data]], [[llama-3]], [[olmo-2]], [[olmo-3]], [[scaling-laws-data-quality]], [[faithful-synth-eval]], [[dolma]], [[fineweb]], [[anthropic-sleeper-agents-data]], [[bespoke-stratos]]
     figures: figures/contamination-detect.html
-->

# Chapter 48 — Contamination Workflow

> **Core insight.** Contamination is not a single event that can be ruled in or out — it is a *signal-to-noise problem* that lives on a continuum between "no overlap" and "verbatim leakage." Every detector (n-gram match, MinHash, canary string, embedding neighbour) trades a specific false-positive against a specific false-negative, and the choice of bucket size and cutoff is not a config detail — it *is* the contamination claim. A defensible memo says what the detector found, at what operating point, with what recall gap; it does **not** say "the eval is clean."
>
> **Guideline.** Pin the eval-set hashes *before* any training stage runs. Hash with two n-gram granularities (a fine one ~8 tokens for recall, a coarse one ~13–50 tokens for precision), store in a Bloom filter for O(1) per-token lookups, and compute overlap *fraction per eval instance*, not per-corpus. Report three numbers per eval: fine-n-gram leakage %, coarse-n-gram leakage %, and semantic-neighbour leakage % (embedding cosine or MinHash Jaccard). Paraphrased leakage is not caught by n-grams — call it out as an explicit recall gap in the memo. When training bleeds into RM preference data or into RS→SFT synthesis, run decontamination *at every stage boundary*, not only on the raw pretraining corpus.

---

## 1. Why contamination is harder than it looks

[[deduplicating-training-data]] quantified the baseline embarrassment: in 2021, **4.6% of LM1B's validation set was already in the training set**, and **3.2% of C4's**. That was the era before anyone was trying. Every reported perplexity number prior to dedup was inflated by a long-tail of memorized validation sentences.

The modern version of the problem has three new features:

1. **Eval sets leak within days of release.** Common Crawl snapshots re-crawl GitHub and HuggingFace pages within 1–2 weeks. A benchmark released Monday can appear verbatim in a CC dump by Friday. The leakage is not your lab's fault; it is the base-rate on the open web.
2. **Synthetic data carries teacher memorization.** [[bespoke-stratos]] openly flags: *"AIME and MATH prompts are public; teacher may have memorized solutions."* If R1 saw AIME 2024 during pretraining, every distilled trace inherits that memorization — decontamination on the seed prompts alone is insufficient. You must also filter the generated traces.
3. **Adversarial contamination exists.** [[anthropic-sleeper-agents-data]] shows that trigger-conditioned examples can be injected with paraphrased variants that evade n-gram matching because no two surface forms repeat. A clean n-gram report is *not* evidence of a clean corpus under an adversarial threat model.

The workflow in this chapter is therefore not "run a script, get a verdict." It is: pick operating points, measure against a known-positive set, report FP/FN honestly, and name the recall gap.

---

## 2. The detection table — methods × n-gram size × cutoff × FP/FN

| Method | Typical n-gram / unit | Cutoff | FP regime | FN regime | Source |
|---|---|---|---|---|---|
| Exact-substring (suffix array) | 50 tokens | ≥1 match | near-zero (50-token collisions are astronomically unlikely by chance) | misses paraphrase, translation, any edit | [[deduplicating-training-data]] ExactSubstr |
| Token n-gram overlap | 8 tokens | ≥1 match in instance | low-to-moderate (common phrases: "the United States of America") | misses light paraphrase | Llama / GPT decontam convention |
| Token n-gram overlap | 13 tokens | ≥80% of eval-instance n-grams hit | very low | misses moderate paraphrase, section reorderings | [[llama-3]] convention, GPT-3 convention |
| MinHash + LSH (doc-level) | 5-gram shingles, ~9000 sigs, 20×450 bands | Jaccard ≥ 0.8 | very low | misses short exact overlaps inside a long doc | [[deduplicating-training-data]] NearDup |
| Canary string | fixed 64–128-char unique token | literal substring | zero (by construction) | zero on the exact canary only; gives *no* generalisation claim | [[anthropic-sleeper-agents-data]] analogue |
| Embedding neighbour | sentence-embedding cosine | ≥0.9 | moderate (topic matches) | catches paraphrase but noisy; high manual triage cost | [[faithful-synth-eval]] cluster occupancy |
| Perplexity gap | — | loss(eval) << loss(control) | noisy; confounded by topic / length | catches strong memorization only | membership-inference style |

**How to read this table.** Each row moves you along a frontier: tighter cutoffs and longer n-grams reduce FP but increase FN. There is no single row that dominates. The defensible workflow uses **at least two rows** — a high-precision exact or coarse-n-gram detector to bound the verbatim leakage, plus a lower-precision paraphrase-sensitive detector (embedding or MinHash Jaccard) to bound the paraphrase gap.

Bucket size choice — a practitioner rule of thumb:
- n=8 → recall-favoured; expect 1–3% FP rate on Wikipedia-style corpora from common phrases.
- n=13 → the 2020-era GPT-3 / Llama default; FP rate on web is <0.1% empirically.
- n=50 → [[deduplicating-training-data]]'s ExactSubstr threshold; essentially zero FP.

If you cannot explain the n-gram size choice in one sentence, the memo is not defensible.

---

## 3. Concrete decontamination pipeline (pseudocode)

This is the reference pipeline; real implementations (OLMES, Llama internal, OLMo 3's decontam utility) are structural variants.

```python
# decontaminate.py — eval-set leakage against a pretraining / SFT / RM corpus.
# Attested primitives: MinHash+LSH from [[deduplicating-training-data]],
# per-snapshot hashing from [[fineweb]], filter-order convention from [[dolma]].

from typing import Iterable
from bloom import BloomFilter              # pybloom / rbloom; O(1) check
from datasketch import MinHash, MinHashLSH # 5-gram shingles

EVAL_SETS    = load_eval_sets()           # list of (name, [instance_text])
N_FINE       = 8                          # recall-favoured n-gram
N_COARSE     = 13                         # precision-favoured n-gram (Llama 3 convention)
OVERLAP_CUT  = 0.80                       # fraction of eval-instance n-grams that must hit
MINHASH_SIGS = 9000                       # [[deduplicating-training-data]] NearDup
LSH_BANDS    = 20; LSH_ROWS = 450         # threshold ~0.8 Jaccard

# Step 1: pin eval set hashes ONCE, before any training stage.
fine_bf   = BloomFilter(capacity=1e9, error_rate=1e-6)
coarse_bf = BloomFilter(capacity=1e9, error_rate=1e-6)
eval_ngrams_by_instance = {}              # {(eval_name, idx): set_of_coarse_ngrams}

for name, instances in EVAL_SETS:
    for idx, text in enumerate(instances):
        toks = tokenize(text)
        fine   = {tuple(toks[i:i+N_FINE])   for i in range(len(toks)-N_FINE+1)}
        coarse = {tuple(toks[i:i+N_COARSE]) for i in range(len(toks)-N_COARSE+1)}
        for g in fine:   fine_bf.add(hash(g))
        for g in coarse: coarse_bf.add(hash(g))
        eval_ngrams_by_instance[(name, idx)] = coarse

# Step 2: stream the training corpus; flag docs that collide with eval n-grams.
lsh = MinHashLSH(threshold=0.8, num_perm=MINHASH_SIGS, params=(LSH_BANDS, LSH_ROWS))
for name, instances in EVAL_SETS:
    for idx, text in enumerate(instances):
        m = MinHash(num_perm=MINHASH_SIGS)
        for sh in shingles(text, k=5): m.update(sh.encode())
        lsh.insert(f"{name}:{idx}", m)

flagged = []                              # (doc_id, eval_key, overlap_fraction, reason)
for doc_id, doc_text in corpus_stream():  # e.g. jsonl shards
    toks = tokenize(doc_text)
    # (a) cheap pre-filter: any fine n-gram hit at all?
    hits = [g for g in ngrams(toks, N_FINE) if hash(g) in fine_bf]
    if not hits: continue
    # (b) coarse overlap per eval-instance
    doc_coarse = set(ngrams(toks, N_COARSE))
    for key, eval_coarse in eval_ngrams_by_instance.items():
        if not doc_coarse & eval_coarse: continue
        frac = len(doc_coarse & eval_coarse) / max(1, len(eval_coarse))
        if frac >= OVERLAP_CUT:
            flagged.append((doc_id, key, frac, "coarse-ngram"))
    # (c) MinHash-LSH for paraphrase-ish
    m = MinHash(num_perm=MINHASH_SIGS)
    for sh in shingles(doc_text, k=5): m.update(sh.encode())
    for neighbour in lsh.query(m):
        flagged.append((doc_id, neighbour, m.jaccard(lsh.get(neighbour)), "minhash"))

# Step 3: remove flagged docs from the corpus; ALSO recurse into RM / SFT pools.
drop_from_corpus(flagged)
```

Three properties matter:
- **Eval-set hashes are pinned before training.** Otherwise your "decontaminated" corpus is contaminated against the wrong eval. [[olmo-3]] ships decontamination as a reusable utility precisely because this ordering bug is common.
- **Per-instance overlap fraction, not per-corpus.** A corpus-level 0.01% overlap is meaningless if it is concentrated on 5 eval instances that now all leak.
- **Recurse.** Flag the pretraining corpus, then re-run against the SFT mix, then against the RM preference pool, then against rejection-sampling outputs ([[llama-3]] §Post-training). Each synthesis stage re-introduces contamination.

---

## 4. Canary strings and watermarks — and what they cannot do

A **canary string** is a deliberately inserted unique token (64–128 char hex blob, nonsense word, or structured tag) placed in the eval set such that if a model later emits it, you have proof of memorization. This is the analog of [[anthropic-sleeper-agents-data]]'s trigger construction, repurposed defensively.

What canaries give you:
- A **zero-FP detector** for verbatim leakage of that exact string.
- A provenance trail: if the canary appears in a downstream model's output, you can attribute leakage to a specific eval artifact.

What canaries do *not* give you:
- Any claim about **non-canary** parts of the eval. A clean canary report is compatible with 90% paraphrased leakage of the surrounding questions.
- Robustness to preprocessing: most data pipelines normalize whitespace, lowercase, or strip non-ASCII — pick a canary form that survives these.
- Robustness to **paraphrased leakage**: if the training corpus contains a rephrased version of the eval, the canary never appears but the answer is memorized.

**Watermarks** (statistical biases in token distributions) are the dual problem — they detect whether *output* came from a specific model, not whether *input* came from a specific eval. They are orthogonal to contamination; useful for distillation-lineage tracking ([[bespoke-stratos]]'s teacher-provenance concern) but not for train-eval overlap.

**Paraphrase limit (the honest number).** On AIME-style math, paraphrased leakage — rewording the problem statement while preserving semantics — survives all n-gram thresholds ≥6 in practice. Embedding neighbour at cosine ≥0.9 catches most but yields manual triage; the empirical ceiling on a realistic budget is roughly 70–80% recall on paraphrased items with 5–10% FP rate. Call this gap out in the memo; do not paper over it.

---

## 5. Downstream contamination: train → RM pref → eval

This is the failure mode the [[llama-3]] recipe makes obvious if you read the data flow:

1. Pretraining corpus contains an eval instance (accidental or CC re-crawl of a public benchmark).
2. SFT uses rejection-sampled outputs from a prior checkpoint. Those outputs answer prompts similar to (or exactly) eval prompts; the RM ranks them based on "looking right," which selects for memorized answers.
3. The ranked top-K becomes next-round SFT data, now *enriched* in memorized eval content.
4. DPO / RLVR preference data is collected on further-sampled prompts; the preference labels now encode "did the model match the memorized answer."
5. Eval harness reports strong performance; the number is partly recall, not reasoning.

The defensive posture [[olmo-3]] illustrates: decontamination runs once per pretraining mix *and* once per SFT mix *and* once per preference mix. [[olmo-2]]'s Dolmino cooldown (50B tokens near end-of-training) is the highest-risk stage: late-stage high-quality pools concentrate contamination exposure.

[[bespoke-stratos]] is the compact example: AIME is public → R1 memorizes → distilled traces inherit → decontaminating only the seed prompts misses the memorization *embedded in the generated trace*. The fix is to filter by matching the *answer-and-solution-pattern* against the eval, not just the problem statement.

**Three checks to run at every stage boundary:**
- Pretraining → SFT: re-hash SFT mix against eval n-grams.
- SFT → RM preference: re-hash preference prompts *and* chosen/rejected responses; responses that quote the gold answer verbatim are a red flag.
- RM preference → RL rollouts: at rollout time, check whether the model is emitting canary strings or verbatim eval passages. [[faithful-synth-eval]] external-verifier machinery gives you the infrastructure.

---

## 6. Live-in-wild contamination

Two empirical patterns modern eval teams plan around:

1. **Public benchmarks leak to Common Crawl within 7–14 days.** MMLU questions have appeared on question-bank websites, tutoring sites, and GitHub gist clones within a fortnight of each update. Any pretraining run with a CC snapshot more recent than the benchmark release is structurally exposed.
2. **Distillation teachers leak indirectly.** If a frontier model served public API traffic, and a paid user asked it every AIME problem, the resulting teacher has effectively been exposed to the eval. [[bespoke-stratos]] is explicit about this risk.

The defensible response is *date hygiene*: pin a "corpus freeze date" that precedes the eval release, and report the freeze gap. When the gap is negative (corpus newer than eval), decontamination is mandatory *and* the memo should caveat that some contamination is unrecoverable via n-gram match alone.

---

## 7. Reporting conventions — the three numbers, not one

A contamination report that collapses to a single percentage is not a report. The convention that survives external scrutiny reports *three* per-eval leakage numbers side by side:

1. **Verbatim / coarse-n-gram leakage %.** Per-instance: fraction of eval instances for which any training document exceeds the coarse-n-gram overlap cutoff (e.g., ≥80% of 13-grams). This is the precision-floor number. Expected value on a well-decontaminated corpus: 0.0%. Any positive value is a bug to fix, not to accept.
2. **Near-duplicate / MinHash leakage %.** Per-instance: fraction of eval instances for which any training document has MinHash Jaccard ≥0.8 at 5-gram shingles. This catches moderate rewrites, section reorderings, and translation-like edits. Expected value: close to the coarse-n-gram number if the decontamination pipeline is correctly configured; substantially higher is a signal that dedup parameters are off.
3. **Semantic-neighbour leakage %.** Per-instance: fraction for which any training document has sentence-embedding cosine ≥0.9 (or domain-specific threshold). This is an *estimate with known FP*; report it with the manual-triage rate applied. Expected value: higher than the other two because it includes topic-adjacent non-contamination.

Reporting all three means readers can reason about the precision/recall frontier rather than trusting a single knob. Reporting only the coarse number is a common "we looked clean" maneuver that the defensible convention treats as insufficient.

A common additional column: **residual perplexity gap** on eval vs a held-out matched-topic control. If the model reports materially lower loss on eval than on the control at equal difficulty, suspect memorization even when the three numbers above are zero. This is a weaker signal (high noise, confounded by style) but the only one that probes the memorization-through-paraphrase pathway.

---

## 8. Defensible memo — template checklist

A contamination memo is a formal artefact, not a README. Eight sections, each with a hard claim:

- [ ] **Eval sets covered.** Name, version hash, release date. One row per eval.
- [ ] **Corpus stages scanned.** Pretraining / mid-training / cooldown / SFT / preference / rollouts — each with token count and freeze date.
- [ ] **Detectors and operating points.** n-gram size(s), cutoff(s), MinHash params, embedding threshold, canary strings (hashed, not printed).
- [ ] **Per-eval leakage table.** For each eval: {% instances with coarse-n-gram hit, % with MinHash-Jaccard hit, % with embedding-neighbour hit}. Report all three — not their union, not their intersection.
- [ ] **Known recall gap.** Explicit sentence: "This workflow catches verbatim and near-verbatim leakage; it does not catch paraphrased leakage beyond cosine 0.9; estimated FN floor is X% based on held-out paraphrase probe."
- [ ] **What was removed.** Document counts per stage and per eval. Delta in token count. Confirmation that re-hashing after removal yields zero hits.
- [ ] **What the memo does NOT claim.** Enumerate: "no adversarial contamination audit," "no semantic-distribution audit," "no teacher-model memorization audit for distilled data."
- [ ] **Reproducibility.** Code commit SHA, hash-pinning artefact (the sealed eval-n-gram Bloom filter), run-time and compute.

A contamination memo that skips any of sections 4, 5, or 7 is not defensible — it is a press release.

**Anti-patterns to name explicitly in review.** If you are reading someone else's memo, flag any of these:
- "We ran decontamination" with no detector specification. Which detector? Which cutoff? On which stages?
- A single percentage "our leakage is X%." Per-eval, per-instance, or per-corpus? Verbatim or semantic?
- Missing freeze-date disclosure. If the corpus post-dates any benchmark without audit, contamination is the null hypothesis.
- Decontamination run only on pretraining mix. [[llama-3]] and [[olmo-3]] both make clear that SFT / preference / rollout stages re-introduce contamination; a one-stage audit is incomplete.
- Classifier-driven quality filters treated as decontamination. Quality and contamination are orthogonal; a quality filter selects *for* memorized correct answers.

---

## Companion visualization

**[figures/contamination-detect.html](figures/contamination-detect.html)** — interactive FP/FN explorer. Panel 1: set n-gram size and cutoff, see FP/FN curves on a synthesized train/eval corpus; the tradeoff frontier at n=8 vs n=13 vs n=50 becomes visible. Panel 2: paraphrase-leakage simulation — a slider controls the rewrite ratio (fraction of eval-instance tokens replaced by synonyms); watch n-gram recall collapse as paraphrase fraction rises, while embedding-neighbour recall degrades more gently. Use it before drafting a memo — the visualization makes the "pick two detectors" rule tangible.

---

## Connections

- **ch-47 (Eval Harness Design)** — the eval harness is where contamination damage manifests; this chapter's memo is what that harness must require before trusting a number.
- **ch-49 (Judge Models)** — a contaminated RM is a contaminated judge; decontamination of the preference pool is prerequisite.
- **[[deduplicating-training-data]]** — the foundational MinHash / exact-substring primitives.
- **[[dolma]] / [[fineweb]]** — filter-cascade conventions; decontamination as a pinned-hash stage alongside dedup.
- **[[llama-3]]** — the RS → SFT → DPO loop that makes downstream contamination structural.
- **[[olmo-2]] / [[olmo-3]]** — per-stage decontamination in a fully-documented open flow.
- **[[bespoke-stratos]]** — concrete teacher-memorization contamination pathway.
- **[[anthropic-sleeper-agents-data]]** — adversarial contamination threat model; paraphrase-evasion motivation.
- **[[faithful-synth-eval]]** — external-verifier infrastructure reused for downstream-stage checks.
- **[[scaling-laws-data-quality]]** — why a contaminated corpus looks like it sits on a better scaling curve than it does.
