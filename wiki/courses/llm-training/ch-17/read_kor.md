<!-- chapter: ch-17
     track: data
     kind: lab
     title: Lab — Small-Scale Filter Pipeline and Ablation Memo
     sources: [[ccnet]], [[c4]], [[dolma]], [[fineweb]], [[deduplicating-training-data]], [[minhash-lsh]], [[scaling-laws-data-quality]], [[karpathy-training-neural-net-recipe]]
     figures: figures/ablation-grid.html
     capstone: data-track
-->

# 17장 — 실습: 소규모 필터 파이프라인과 Ablation 메모

> **핵심 통찰.** data track의 이론, 즉 stage order, dedup math, classifier vs heuristic tradeoff, domain mixing은 한 번에 하나의 변경만 넣는 ablation을 직접 실행하고, 그 결과를 한 문단으로 방어할 수 있을 때에만 지식이 된다. 이 lab은 capstone이다. 작은 CommonCrawl slice 위에 CCNet 모양의 four-stage pipeline을 만들고, 작은 모델 두 개(각 filter 적용/미적용)를 훈련하고, 자기 eval suite에 대해 decontaminate한 뒤, 각 observation을 이름 붙이고, 각 outcome을 예측하고, 모든 surprise에 hypothesis를 제시하는 `ablation-memo.md`를 ship한다. deliverable은 "dataset"이 아니다. memo다.
>
> **가이드라인.** Dolma stage order([[dolma]])를 따르라: URL -> lang-ID -> quality (perplexity) -> exact dedup -> MinHash near-dedup. 테스트 중인 filter를 제외한 모든 것을 freeze하라. eval number를 읽기 *전에* 각 outcome을 예측하라(Karpathy의 "be a scientist" — [[karpathy-training-neural-net-recipe]]). 효과를 예측할 수 없는 filter는 당신이 이해하지 못한 filter이며, memo는 이를 명시적으로 불러야 한다.

---

## 목표

이 장이 끝날 때까지 당신은 `ch-17/lab-artifacts/` 안에 네 artifact를 갖게 된다.

1. **`pipeline/`** — four-stage filter cascade(`lang_id.py`, `perplexity.py`, `exact_dedup.py`, `minhash_dedup.py`, `decontaminate.py`, 그리고 driver `run_pipeline.py`)의 runnable implementation.
2. **`checkpoints/`** — 훈련된 `~125M`(또는 constrained path에서는 `~50M`) checkpoint 두 개: *unfiltered* data와 *fully filtered* data. 그리고 각 ablation row의 intermediate checkpoint.
3. **`eval-report.json`** — 모든 ablation cell에 걸친 ARC-E, HellaSwag, LAMBADA, MMLU-tiny(full path) 또는 HellaSwag + LAMBADA(constrained path) 평가.
4. **`ablation-memo.md`** — filter별 table 하나, observation별 paragraph 하나, unexpected한 모든 것에 대한 explicit hypothesis 하나. 이것이 grade-bearing deliverable이다.

당신은 [[fineweb]] §4와 [[dolma]] §6이 15T / 3T 토큰에서 하는 일을 miniature로 재현한다. 같은 methodology를 네 자릿수 작은 scale에서 수행하는 것이다.

---

## 전체 예산 경로 (10 GB / 125M)

- **Data slice.** 최근 CommonCrawl dump에서 WARC file 하나 또는 두 개를 text로 추출한다. target은 raw 약 10 GB이고, lang-ID + dedup 뒤 약 4–6 GB, quality filter 뒤 약 2–3 GB를 예상한다.
- **Model.** `~125M` parameter decoder, 12 layers, d_model = 768, 12 heads, ctx = 1024, tied embeddings. 각 ablation cell을 약 2.5B tokens로 훈련한다(filtered slice 위 약 10 epoch — [[data-constrained-scaling]]에 따르면 small-data regime에서는 허용 가능).
- **Eval.** ARC-E, HellaSwag, LAMBADA, MMLU-tiny(MMLU의 500-question stratified subsample — full MMLU는 125M에서 너무 noisy하다).
- **Compute.** 125M × 2.5B tokens에서 ablation cell당 약 1×H100-day, 약 10 cells => 약 10 GPU-days. FineWeb ablation run의 ballpark(약 1.8B models × 30B tokens, equivalent FLOP)에 맞는다.

## 자원 제약 경로 (1 GB / 50M)

- **Data slice.** 1 GB raw CommonCrawl text(WET/WARC shard 하나).
- **Model.** `~50M` parameter decoder, 8 layers, d_model = 512, 8 heads, ctx = 1024. ablation cell당 약 400M tokens를 훈련한다.
- **Eval.** HellaSwag + LAMBADA만. ARC-E와 MMLU-tiny는 약 80M params 아래에서 너무 noisy하다. [[fineweb]] Table 2는 error bar가 filter delta를 삼킨다는 것을 보인다.
- **Compute.** A100 하나에서 cell당 약 2–3 GPU-hours. 전체 ablation grid가 overnight 실행된다.

Memo requirement는 두 path에서 **동일**하다. 실험 scale을 낮추되 discipline은 낮추지 말라. 이것이 [[karpathy-training-neural-net-recipe]]에서 Karpathy가 "overfit a single batch first, then expand"라고 말하는 정확한 요지다.

---

## 파이프라인 단계

네 stage는 [[ccnet]] template를 따르고, dedup을 exact-first, near-last로 나누는 [[dolma]] refinement를 따른다. 각 stage는 in-place filtering 대신 per-document JSONL attribute file을 쓴다. 이것이 Dolma `attributes/` pattern이고 중요하다. 비싼 stage를 다시 실행하지 않고 최종 keep/drop decision을 여러 번 다시 실행하고 싶을 것이기 때문이다.

### §1. 언어 식별 (`lang_id.py`)

fastText `lid.176.bin`을 load하고 각 document를 score한다. `lang_score >= 0.65`인 English를 keep한다(CCNet은 0.5를 썼다. small scale에서는 125M 모델이 multilingual noise를 흡수할 수 없으므로 기준을 올린다).

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

`attributes/` separation(score를 쓰고 keep/drop은 나중에 수행)은 [[dolma]] §4에서 명시된다. threshold sweep이 expensive stage를 다시 실행하지 않도록 하는 별도 cheap pass다.

### §2. Perplexity 필터 (`perplexity.py`)

CCNet의 original trick이다. 깨끗한 anchor(Wikipedia)에 5-gram KenLM을 훈련하고, 각 doc의 perplexity를 그것에 대해 score한다. 낮은 PPL = 더 Wikipedia-like. [[ccnet]]을 따라 33rd / 66th percentile에서 `head` / `middle` / `tail` bucket으로 나누고, `head + middle`을 keep한다.

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

먼저 Wikipedia 위에 KenLM을 훈련하라(Moses tokenise, lowercase, `-o 5 --discount_fallback`). Sanity check: held-out Wikipedia page 100개를 score하라. median PPL이 약 50–120이어야 한다. 500을 넘으면 KenLM이 broken이다.

### §3. Exact dedup (`exact_dedup.py`)

[[deduplicating-training-data]] §ExactSubstr를 따른다. concatenated corpus 위에 suffix array를 만들고, 길이 ≥ **50 tokens**인 duplicate span을 모두 제거한다(논문의 empirical threshold — common idiom은 피할 만큼 길고 boilerplate는 잡을 만큼 짧다). 1 GB에서는 plain `pysais` implementation이 몇 분 걸린다. 10 GB에서도 single box에서 한 시간 안에 돈다.

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

논문이 flag한 gotcha 두 가지([[deduplicating-training-data]] §3): train/val boundary를 넘어서 dedup하지 말라(eval leak). 각 span의 *첫* occurrence를 유지하라(document coherence 보존).

### §4. MinHash near-dedup (`minhash_dedup.py`)

[[deduplicating-training-data]] §NearDup와 [[minhash-lsh]]의 primitive를 따른다. 5-gram으로 shingle하고, MinHash signature를 만들고, LSH band에 넣는다. 논문은 Jaccard ≥ 0.8에 대해 9000 hashes, b=20×r=450을 썼다. 1 GB path는 compute를 낮추기 위해 128 hashes, b=16×r=8로 내려도 비슷한 recall을 얻는다.

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

b와 r 선택이 하중을 지탱한다. [[minhash-lsh]]의 LSH banding: P(collision | s) = 1 − (1 − s^r)^b. (b=16, r=8)에서 s=0.8 → 0.95, s=0.5 → 0.06이다. target similarity 주변의 날카로운 S-curve가 MinHash+LSH가 brute NN을 이기는 이유다. threshold에서 high recall을 얻고, 다른 곳의 wasted work는 적다.

[[fineweb]] twist 하나: **global이 아니라 per-snapshot dedup**하라(FineWeb §3.3 — global dedup은 snapshot마다 한 번씩 나타나는 high-quality doc을 제거했다). slice가 여러 CC dump에 걸치면 MinHash를 per-dump로 실행하라. 이는 [[ccnet]]에는 없다. memo에서 CCNet recipe와의 delta로 cite하라.

---

## Ablation 프로토콜

[[fineweb]] Table 2와 [[dolma]] §6을 따른다. row마다 filter 하나를 toggle하고, column마다 eval task 하나를 두며, 나머지는 모두 freeze한다. grid는 **4 × 4**다.

| Filter configuration | ARC-E | HellaSwag | LAMBADA | MMLU-tiny |
|---|---|---|---|---|
| `none` (raw extracted text) | | | | |
| `+ lang-ID` | | | | |
| `+ lang-ID + perplexity` | | | | |
| `+ lang-ID + perplexity + exact-dedup` | | | | |
| `+ lang-ID + perplexity + exact-dedup + MinHash` (full stack) | | | | |

Constrained path에서는 ARC-E와 MMLU-tiny column을 drop하고 다섯 row를 유지한다.

Freezing rules(어기면 memo는 쓸모없어진다):

- **Cell당 fixed epoch count가 아니라 fixed token budget.** [[fineweb]]은 명확하다. ablation cell마다 compute를 같게 한다. `+lang-ID`가 token의 40%를 제거했다면 replacement로 re-sample하여 각 cell이 같은 수의 training token을 보게 하라. 그렇지 않으면 "더 많은 epoch"와 "더 나은 filter"를 동시에 측정하게 되어 구분할 수 없다.
- **Fixed LR schedule, fixed seed, fixed init, fixed context length.** cell 사이의 *유일한* 변화는 filter다.
- **숫자를 읽기 전에 delta를 예측하라.** row N을 실행하기 전에 memo에 쓰라. "unfiltered slice에 약 15% non-English가 있으므로 +lang-ID가 `+2.0 HellaSwag`를 올릴 것으로 예상한다." 그런 다음 비교하라. 이것이 Karpathy의 "be a scientist" 그대로다. 틀릴 수 있는 예측이 있어야 배우고 있음을 알 수 있다.
- **한 번에 하나의 변경.** MinHash-only vs exact-only를 보고 싶다면 추가 row 두 개다. 합치지 말라.

---

## Ablation 메모 템플릿

`ablation-memo.md`는 grade-bearing deliverable이다. Skeleton(관찰마다 한 문단, plain prose, 총 1–2 pages):

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

부풀리지 말라. [[fineweb]] §4의 자체 write-up은 15T tokens에 대해 약 3 pages다. 당신의 것은 10 GB에 대해 1–2 pages다. 더 길면 결론을 내리는 것이 아니라 설명하고 있는 것이다.

---

## Decontamination step

모든 ablation cell은 같은 eval suite를 공유하므로 contamination은 모든 row를 한 번에 falsify할 수 있는 confound다. [[deduplicating-training-data]] §5를 따르라. 그들은 LM1B val의 **4.6%**와 C4 val의 **3.2%**가 training overlap을 가진다는 것을 발견했다. 측정을 복제하라.

1. `n = 13` (Gopher / Chinchilla default; [[dolma]] §3.5 uses 13).
2. concatenated eval prompt 위에 n-gram set을 만든다.
3. 하나 이상 포함하는 training doc을 drop한다.
4. **FP rate를 측정한다.** dropped doc 100개를 sample하고, 각각을 genuine leak vs benign coincidence("the quick brown fox"가 HellaSwag와 unrelated news 둘 다에 있는 경우)로 hand-label한다. `FP_rate = benign/100`을 보고한다.

n=13에서 예상 FP는 5–15%다. 더 높으면 n이 너무 작다(common phrase와 match). 더 낮으면 eval suite가 너무 작거나 matcher가 broken이다. n=8에서 [[deduplicating-training-data]]는 FP ≈ 50%를 찾았다. 이것이 더 긴 n에 대한 그들의 논거다. `FP_rate > 20%`이면 ablation number를 믿기 전에 `n = 15`로 다시 실행하라.

---

## 승인 기준

다섯 조건이 모두 true일 때 memo는 submit할 수 있다.

1. **네 stage가 end-to-end로 실행된다.** filtered text가 아니라 per-document attribute를 emit한다. keep/drop pass는 별도 script이고 <60s에 다시 실행된다.
2. **Ablation grid가 채워져 있다**: constrained에서는 ≥ 4 rows × 2 cols, full에서는 5 rows × 4 cols. 각 cell에는 ≥ 2 seeds의 stddev가 있는 number 또는 single-seed가 "illustrative"라는 one-line note가 있다(유일하게 허용되는 예외).
3. **각 filter paragraph에는 predicted delta + observed delta가 있고, sign은 실행 전에 예측되어 있다.** magnitude 예측이 틀려도 괜찮다. no-prediction paragraph는 안 된다.
4. **Decontamination FP rate를 측정하고 보고했다.** 주장만 하면 안 된다.
5. **"one thing I was wrong about" paragraph가 존재한다** 그리고 구체적 surprise를 이름 붙인다. 모든 cell이 예측과 맞았다면 너무 조심스럽게 실행했거나 정직하게 예측하지 않은 것이다. 그렇게 말하라.

Stretch(필수 아님): FineWeb의 per-snapshot-vs-global MinHash ablation을 extra row 하나로 재현한다. ch-20 preview로 `oracle` Dolma-classifier-scored row를 추가한다.

---

## 연결

- **[[ccnet]]** — 여기서 재구현하는 four-stage template. §2 perplexity filter가 CCNet의 중심 기여다.
- **[[c4]]** — heuristic baseline. `none -> +lang-ID`는 C4-minus-blocklist를 근사한다.
- **[[dolma]]** — `attributes/` separation(§1), stage-order defence(§6), paragraph-dedup-last rationale.
- **[[fineweb]]** — ablation methodology. per-snapshot vs global MinHash stretch row. ch-20이 향하는 classifier direction.
- **[[deduplicating-training-data]]** — ExactSubstr ≥50 tokens, NearDup Jaccard ≥0.8, decontamination FP methodology.
- **[[minhash-lsh]]** — banding math P(collision) = 1 − (1 − s^r)^b. §4에서 (b, r)을 정당화하는 방법.
- **[[scaling-laws-data-quality]]** — equal-token cell이 다른 scaling curve에 놓이는 이유를 설명하는 formal lens.
- **[[karpathy-training-neural-net-recipe]]** — be-a-scientist discipline. memo는 Karpathy의 methodology를 data에 적용한 것이다.
- **Next chapter ([[ch-18]]).** 이 four-stage pipeline은 generate→filter→dedup→verify→select→mix의 한 instance다. Ch-18은 pattern에 이름을 붙인다. memo를 가져가라.

---

## 더 읽을거리

- **[[doremi]]** — slice가 깨끗해지면 도메인 간 mixing weight가 다음 learned object가 된다.
- **[[d4]]** — semantic(단지 exact/near가 아닌) dedup. 반복되는 *meaning*에 대해서는 MinHash보다 깊다.
- **[[data-constrained-scaling]]** — filtered slice를 몇 epoch 훈련하면 과한가. constrained path는 이미 약 10 epochs를 훈련한다.
- **`datatrove` (HF)** and **`dolma` (AllenAI)** — production re-implementation. 직접 작성한 *뒤에* code를 읽어라.
- **[[the-pile]]** — pre-CCNet heuristic corpus. memo opening paragraph의 유용한 contrast.

---

## 동반 시각화

[`figures/ablation-grid.html`](figures/ablation-grid.html) — interactive 4×4 ablation grid. 각 filter(lang-ID, perplexity, exact-dedup, MinHash near-dedup)를 toggle하고 ARC-E, HellaSwag, LAMBADA, MMLU-tiny의 illustrative delta를 본다. 숫자는 Dolma Table 5와 FineWeb Figure 4에서 인용하고 약 125M으로 scaled했다. 당신의 실행은 magnitude가 다르겠지만, 명확히 directional한 세 cell에서는 sign이 문헌과 맞아야 한다. 실행 전에는 예측을 구체화하는 데, 실행 후에는 sign이 published literature와 맞는지 sanity check하는 데 사용하라.
