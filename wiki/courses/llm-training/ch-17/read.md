<!-- chapter: ch-17
     track: data
     kind: lab
     title: Lab — Small-Scale Filter Pipeline and Ablation Memo
     sources: [[ccnet]], [[c4]], [[dolma]], [[fineweb]], [[deduplicating-training-data]], [[minhash-lsh]], [[scaling-laws-data-quality]], [[karpathy-training-neural-net-recipe]]
     figures: figures/ablation-grid.html
     capstone: data-track
-->

# Chapter 17 — Lab: Small-Scale Filter Pipeline and Ablation Memo

> **Core insight.** The data track's theory — stage order, dedup math, classifier vs heuristic tradeoff, domain mixing — only becomes knowledge when you have run a one-change-at-a-time ablation that you can defend in a paragraph. This lab is the capstone: you will build a CCNet-shaped four-stage pipeline on a small CommonCrawl slice, train two small models (with and without each filter), decontaminate against your own eval suite, and ship an `ablation-memo.md` that names each observation, predicts each outcome, and offers a hypothesis for every surprise. The deliverable is not "a dataset" — it is the memo.
>
> **Guideline.** Follow the Dolma stage order ([[dolma]]): URL -> lang-ID -> quality (perplexity) -> exact dedup -> MinHash near-dedup. Freeze everything except the filter under test. Predict each outcome *before* reading the eval number (Karpathy's "be a scientist" — [[karpathy-training-neural-net-recipe]]). A filter whose effect you cannot predict is a filter you do not understand, and the memo must call that out explicitly.

---

## Goal

By the end of this chapter you will own four artifacts in `ch-17/lab-artifacts/`:

1. **`pipeline/`** — a runnable implementation of the four-stage filter cascade (`lang_id.py`, `perplexity.py`, `exact_dedup.py`, `minhash_dedup.py`, `decontaminate.py`, plus a driver `run_pipeline.py`).
2. **`checkpoints/`** — two trained `~125M` (or `~50M` on the constrained path) checkpoints: one on *unfiltered* data, one on *fully filtered* data. Plus the intermediate checkpoints for each ablation row.
3. **`eval-report.json`** — evaluations on ARC-E, HellaSwag, LAMBADA, MMLU-tiny (full path) or HellaSwag + LAMBADA (constrained path) across all ablation cells.
4. **`ablation-memo.md`** — one table per filter, one paragraph per observation, one explicit hypothesis for anything unexpected. This is the grade-bearing deliverable.

You are reproducing in miniature what [[fineweb]] §4 and [[dolma]] §6 do at 15T / 3T tokens — the same methodology, four orders of magnitude smaller.

---

## Full-budget path (10 GB / 125M)

- **Data slice.** One or two WARC files from a recent CommonCrawl dump, extracted to text — target ~10 GB raw, expect ~4–6 GB after lang-ID + dedup, ~2–3 GB after quality filter.
- **Model.** `~125M` parameter decoder, 12 layers, d_model = 768, 12 heads, ctx = 1024, tied embeddings. Train each ablation cell for ~2.5B tokens (~10 epochs over the filtered slice — acceptable in the small-data regime per [[data-constrained-scaling]]).
- **Eval.** ARC-E, HellaSwag, LAMBADA, MMLU-tiny (a 500-question stratified subsample of MMLU — big MMLU is too noisy at 125M).
- **Compute.** ~1×H100-day per ablation cell at 125M × 2.5B tokens; ~10 cells => ~10 GPU-days. Matches the ballpark of FineWeb's ablation runs (~1.8B models × 30B tokens ~ equivalent FLOP).

## Resource-constrained path (1 GB / 50M)

- **Data slice.** 1 GB raw CommonCrawl text (one WET/WARC shard).
- **Model.** `~50M` parameter decoder, 8 layers, d_model = 512, 8 heads, ctx = 1024. Train ~400M tokens per ablation cell.
- **Eval.** HellaSwag + LAMBADA only. ARC-E and MMLU-tiny are too noisy below ~80M params — [[fineweb]] Table 2 shows the error bars swallow the filter deltas.
- **Compute.** ~2–3 GPU-hours per cell on one A100; whole ablation grid runs overnight.

The memo requirement is **identical** on both paths. Scale down the experiment, not the discipline — this is exactly the point Karpathy makes in [[karpathy-training-neural-net-recipe]] about "overfit a single batch first, then expand."

---

## Pipeline stages

The four stages follow the [[ccnet]] template with the [[dolma]] refinement that dedup splits into exact-first, near-last. Each stage writes a per-document JSONL attribute file rather than filtering in place — that is the Dolma `attributes/` pattern and it matters because you will want to re-run the final keep/drop decision many times without re-running the expensive stages.

### §1. Language ID (`lang_id.py`)

Load fastText `lid.176.bin` and score each document. Keep English with `lang_score >= 0.65` (CCNet used 0.5; raise the bar at small scale to reduce multilingual noise that a 125M model cannot absorb).

```python
# lab-artifacts/pipeline/lang_id.py
import fasttext, json
from pathlib import Path

MODEL = fasttext.load_model("models/lid.176.bin")

def score_lang(text: str) -> tuple[str, float]:
    labels, probs = MODEL.predict(text.replace("\n", " ")[:10_000], k=1)
    return labels[0].replace("__label__", ""), float(probs[0])

def run(input_jsonl: Path, output_attr_jsonl: Path) -> None:
    with input_jsonl.open() as fin, output_attr_jsonl.open("w") as fout:
        for line in fin:
            doc = json.loads(line)
            lang, score = score_lang(doc["text"])
            fout.write(json.dumps({"id": doc["id"], "lang": lang, "lang_score": score}) + "\n")
```

The `attributes/` separation (write scores, do keep/drop later) is explicit in [[dolma]] §4 — a separate cheap pass so threshold sweeps do not re-run expensive stages.

### §2. Perplexity filter (`perplexity.py`)

CCNet's original trick: train a 5-gram KenLM on a clean anchor (Wikipedia) and score each doc's perplexity against it. Lower PPL = more Wikipedia-like. Following [[ccnet]], bucket into `head` / `middle` / `tail` at the 33rd / 66th percentiles; keep `head + middle`.

```python
# lab-artifacts/pipeline/perplexity.py
import kenlm, json, numpy as np
from pathlib import Path

LM = kenlm.LanguageModel("models/wiki.en.5gram.arpa")

def doc_ppl(text: str) -> float:
    n_words = max(1, len(text.split()))
    return float(10 ** (-LM.score(text) / n_words))

def run(input_jsonl: Path, output_attr_jsonl: Path) -> None:
    ppls = [doc_ppl(json.loads(l)["text"]) for l in input_jsonl.open()]
    head_thr, tail_thr = np.percentile(ppls, [33, 66])
    with input_jsonl.open() as fin, output_attr_jsonl.open("w") as fout:
        for line, ppl in zip(fin, ppls):
            doc = json.loads(line)
            bucket = "head" if ppl < head_thr else ("middle" if ppl < tail_thr else "tail")
            fout.write(json.dumps({"id": doc["id"], "ppl": ppl, "ppl_bucket": bucket}) + "\n")
```

Train KenLM up front on Wikipedia (Moses tokenise, lowercase, `-o 5 --discount_fallback`). Sanity check: score 100 held-out Wikipedia pages; median PPL ~50–120. Over 500 = broken KenLM.

### §3. Exact dedup (`exact_dedup.py`)

Following [[deduplicating-training-data]] §ExactSubstr: build a suffix array over the concatenated corpus and remove every duplicate span of length ≥ **50 tokens** (the paper's empirical threshold — long enough to avoid common idioms, short enough to catch boilerplate). At 1 GB a plain `pysais` implementation takes minutes; at 10 GB it runs in under an hour on a single box.

```python
# lab-artifacts/pipeline/exact_dedup.py
from pysais import sais
import numpy as np, json
from pathlib import Path

MIN_DUP_TOKENS = 50  # Lee et al. 2021, Section 3

def find_duplicate_spans(tokens: np.ndarray, sa: np.ndarray) -> list[tuple[int, int]]:
    lcp = _kasai_lcp(tokens, sa)
    spans = [(sa[i+1], sa[i+1] + l) for i, l in enumerate(lcp) if l >= MIN_DUP_TOKENS]
    return _merge_overlapping(spans)

def run(input_jsonl: Path, output_jsonl: Path) -> None:
    tokens, doc_boundaries = _tokenize_and_concat(input_jsonl)
    sa = np.asarray(sais(tokens), dtype=np.int64)
    dup_spans = find_duplicate_spans(tokens, sa)
    _remove_spans_and_emit(input_jsonl, output_jsonl, doc_boundaries, dup_spans)
```

Two gotchas the paper flags ([[deduplicating-training-data]] §3): do not dedup across the train/val boundary (you leak eval); keep the *first* occurrence of each span (preserves document coherence).

### §4. MinHash near-dedup (`minhash_dedup.py`)

Following [[deduplicating-training-data]] §NearDup and the primitives in [[minhash-lsh]]. Shingle into 5-grams, build MinHash signatures, put them in LSH bands. The paper used 9000 hashes, b=20×r=450 for Jaccard ≥ 0.8; the 1 GB path drops to 128 hashes, b=16×r=8 for similar recall at lower compute.

```python
# lab-artifacts/pipeline/minhash_dedup.py
from datasketch import MinHash, MinHashLSH
import json
from pathlib import Path

N_HASHES, BANDS, ROWS = 128, 16, 8         # Lee et al. 2021 use 9000 / 20 / 450
SHINGLE_N, JACCARD_THRESHOLD = 5, 0.8

def doc_signature(text: str) -> MinHash:
    words = text.split()
    shingles = {" ".join(words[i:i+SHINGLE_N]) for i in range(len(words) - SHINGLE_N + 1)}
    m = MinHash(num_perm=N_HASHES)
    for s in shingles: m.update(s.encode())
    return m

def run(input_jsonl: Path, output_jsonl: Path) -> None:
    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=N_HASHES, params=(BANDS, ROWS))
    kept = {}
    for line in input_jsonl.open():
        doc = json.loads(line); sig = doc_signature(doc["text"])
        if lsh.query(sig): continue
        lsh.insert(doc["id"], sig); kept[doc["id"]] = doc
    with output_jsonl.open("w") as fout:
        for d in kept.values(): fout.write(json.dumps(d) + "\n")
```

The b-and-r choice is load-bearing. LSH banding from [[minhash-lsh]]: P(collision | s) = 1 − (1 − s^r)^b. At (b=16, r=8): s=0.8 → 0.95, s=0.5 → 0.06. That sharp S-curve around the target similarity is why MinHash+LSH beats brute NN — high recall at threshold, little wasted work elsewhere.

One [[fineweb]] twist: **dedup per-snapshot, not globally** (FineWeb §3.3 — global dedup removed high-quality docs that appear once per snapshot). If your slice spans CC dumps, run MinHash per-dump. This is not in [[ccnet]]; cite it in the memo as a delta from the CCNet recipe.

---

## Ablation protocol

Following [[fineweb]] Table 2 and [[dolma]] §6: one filter toggled per row, one eval task per column, and everything else frozen. The grid is **4 × 4**:

| Filter configuration | ARC-E | HellaSwag | LAMBADA | MMLU-tiny |
|---|---|---|---|---|
| `none` (raw extracted text) | | | | |
| `+ lang-ID` | | | | |
| `+ lang-ID + perplexity` | | | | |
| `+ lang-ID + perplexity + exact-dedup` | | | | |
| `+ lang-ID + perplexity + exact-dedup + MinHash` (full stack) | | | | |

On the constrained path, drop the ARC-E and MMLU-tiny columns and keep the five rows.

Freezing rules (violate these and the memo is worthless):

- **Fixed token budget per cell**, not fixed epoch count. [[fineweb]] is explicit: equal compute per ablation cell. If `+lang-ID` removes 40% of tokens, re-sample with replacement so each cell sees the same number of training tokens. Otherwise you are measuring "more epochs" as well as "better filter" and cannot tell them apart.
- **Fixed LR schedule, fixed seed, fixed init, fixed context length.** The *only* change between cells is the filter.
- **Predict the delta before reading the number.** Before running row N, write in the memo: "I expect +lang-ID to gain `+2.0 HellaSwag`, because the unfiltered slice has ~15% non-English." Then compare. This is Karpathy's "be a scientist" literally: a prediction you can be wrong about is how you know you were learning.
- **One change at a time.** If you want to see MinHash-only vs exact-only, that is two additional rows; do not collapse them.

---

## Ablation-memo template

`ablation-memo.md` is the grade-bearing deliverable. Skeleton (one paragraph per observation, plain prose, 1–2 pages total):

```markdown
# Ablation memo — ch-17

## Setup
- Slice, extractor, model size, tokens/cell, eval list, seed, LR, schedule.

## Filter 1 — language ID
<table: none, +lang across eval columns>

**Predicted:** +2.0 HellaSwag, neutral LAMBADA.
**Observed:** +2.4 HellaSwag, -0.3 LAMBADA.
**Hypothesis for the LAMBADA drop:** unfiltered slice had German Wikipedia
mirrors whose byte-level tokenisation produced near-English n-grams that
matched LAMBADA continuation patterns.

## Filter 2 — perplexity · Filter 3 — exact dedup · Filter 4 — MinHash
<same three-line structure per filter>

## Decontamination step
- n = 13, drop any train doc sharing an n-gram with any eval prompt.
- FP rate: sample 100 dropped docs, hand-label true vs benign. FP = <x>%.

## One thing I was wrong about
<Prose: the single observation you most misforecast. This paragraph is the
real point of the lab.>
```

Do not pad. [[fineweb]] §4's own write-up is ~3 pages for 15T tokens; yours is 1–2 pages for 10 GB. More means you are describing instead of concluding.

---

## Decontamination step

Every ablation cell shares the eval suite, so contamination is a confound that can falsify *every* row at once. Follow [[deduplicating-training-data]] §5: they found **4.6%** of LM1B val / **3.2%** of C4 val overlap training. Replicate the measurement:

1. `n = 13` (Gopher / Chinchilla default; [[dolma]] §3.5 uses 13).
2. Build an n-gram set over the concatenated eval prompts.
3. Drop any training doc containing one or more.
4. **Measure the FP rate.** Sample 100 dropped docs; hand-label each as genuine leak vs benign coincidence ("the quick brown fox" in both HellaSwag and unrelated news). Report `FP_rate = benign/100`.

Expected FP at n=13: 5–15%. Higher means n is too small (matching on common phrases); lower means tiny eval suite or a broken matcher. At n=8, [[deduplicating-training-data]] found FP ≈ 50% — their argument for longer n. If `FP_rate > 20%`, rerun with `n = 15` before trusting any ablation number.

---

## Acceptance criteria

A memo is submittable when all five are true:

1. **All four stages run end-to-end**, emitting per-document attributes (not filtered text); the keep/drop pass is a separate script and reruns in <60s.
2. **Ablation grid is filled in**: ≥ 4 rows × 2 cols (constrained) or 5 rows × 4 cols (full). Each cell has a number with stddev over ≥ 2 seeds, or a one-line note that single-seed is "illustrative" (the only acceptable exception).
3. **Each filter's paragraph contains a predicted delta + observed delta, with the sign predicted before running.** Wrong-magnitude predictions are fine; no-prediction paragraphs are not.
4. **Decontamination FP rate is measured and reported**, not asserted.
5. **The "one thing I was wrong about" paragraph exists** and names a specific surprise. If every cell matched your prediction, you either ran too carefully or did not predict honestly; say so.

Stretch (not required): reproduce FineWeb's per-snapshot-vs-global MinHash ablation as one extra row; add an `oracle` Dolma-classifier-scored row as a preview of ch-20.

---

## Connections

- **[[ccnet]]** — the four-stage template reimplemented here; the §2 perplexity filter is CCNet's central contribution.
- **[[c4]]** — heuristic baseline; `none -> +lang-ID` approximates C4-minus-blocklist.
- **[[dolma]]** — `attributes/` separation (§1), stage-order defence (§6), paragraph-dedup-last rationale.
- **[[fineweb]]** — ablation methodology; per-snapshot vs global MinHash stretch row; the classifier direction ch-20 takes.
- **[[deduplicating-training-data]]** — ExactSubstr ≥50 tokens, NearDup Jaccard ≥0.8, decontamination FP methodology.
- **[[minhash-lsh]]** — banding math P(collision) = 1 − (1 − s^r)^b, how you justify (b, r) in §4.
- **[[scaling-laws-data-quality]]** — formal lens for why equal-token cells land on different scaling curves.
- **[[karpathy-training-neural-net-recipe]]** — be-a-scientist discipline; the memo is Karpathy's methodology applied to data.
- **Next chapter ([[ch-18]]).** This four-stage pipeline is one instance of generate→filter→dedup→verify→select→mix. Ch-18 names the pattern; bring the memo forward.

---

## Further reading

- **[[doremi]]** — once the slice is clean, mixing weights across domains become the next learned object.
- **[[d4]]** — semantic (not just exact/near) dedup; deeper than MinHash for repeated *meaning*.
- **[[data-constrained-scaling]]** — how many epochs of the filtered slice is too many; the constrained path already trains ~10 epochs.
- **`datatrove` (HF)** and **`dolma` (AllenAI)** — production re-implementations; read the code *after* writing your own.
- **[[the-pile]]** — pre-CCNet heuristic corpus; useful contrast for the memo's opening paragraph.

---

## Companion visualization

[`figures/ablation-grid.html`](figures/ablation-grid.html) — interactive 4×4 ablation grid. Toggle each filter (lang-ID, perplexity, exact-dedup, MinHash near-dedup) and see the illustrative delta on ARC-E, HellaSwag, LAMBADA, and MMLU-tiny. Numbers are cited from Dolma Table 5 and FineWeb Figure 4 — scaled to ~125M; your own run will differ in magnitude but should agree in sign on the three clearly-directional cells. Use this before running to make your predictions concrete; use it after running as a sanity check that your signs match the published literature.
