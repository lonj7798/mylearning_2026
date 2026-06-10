<!-- chapter: ch-53
     track: eval
     kind: lab
     title: Lab: Eval Harness with Slices and Regression Tracking
     deps: [ch-52]
     sources: [[harmbench-data]], [[wildguard-data]], [[judge-llm-bias]], [[deduplicating-training-data]], [[scaling-laws-data-quality]], [[olmo-3]], [[karpathy-training-neural-net-recipe]], [[ruler]], [[bfcl]]
     figures: figures/eval-pipeline.html
     excerpts: excerpts/harmbench-behavior-layering.md, excerpts/wildguard-refusal-label.md, excerpts/judge-bias-mitigations.md, excerpts/minhash-contamination-gate.md, excerpts/olmes-harness-shape.md, excerpts/ruler-synthetic-slice.md, excerpts/bfcl-ast-matcher.md
-->

# 53장 — Lab: Slice와 Regression Tracking을 갖춘 Eval Harness

> **핵심 통찰.** eval harness는 course의 나머지가 만들어 내는 모든 checkpoint에 대한 go/no-go arbiter다. load-bearing part는 네 가지다. bootstrap CI를 갖춘 slice-aware scoring, n-gram overlap이 Lee-2021 threshold를 넘으면 실패하는 contamination gate, position-swap과 length-control을 수행하는 judge-bias probe, go / no-go / needs-fix를 emit하는 memo writer. 나머지는 모두 plumbing이다.
>
> **가이드라인.** 두 번째 task를 추가하기 전에, "ch-44-RL이 ch-34-SFT 대비 regression인가?"를 prompt 하나에서 end-to-end로 답할 수 있는 가장 작은 harness를 세워라. slice는 post-hoc filter가 아니라 first-class object다. contamination gate는 precondition이다. gate가 trip되면 score를 보고하지 않는다. judge-bias probe는 postcondition이다. swap-flip-rate >20%(Zheng 2023 GPT-4 baseline)이면 pairwise number를 suppress한다. memo는 downstream이 읽는 유일한 artifact다. 모든 number는 slice, CI, contamination verdict로 trace된다.

---

## Goal

`ch-34` SFT와 `ch-44` RLVR/PRM checkpoint를 ingest하고, reasoning + chat + safety에서 sliced bootstrap CI로 scoring하며, contamination으로 gate하고, judge bias를 probe하며, go/no-go memo를 emit하는 runnable eval harness를 만든다. harness는 infra track(ch-55..ch-60)에서 library로 import 가능해야 한다. 끝나면 다음 질문에 답할 수 있다. "RL stage가 어떤 slice에서 regression을 냈는가, 그리고 그 regression은 CI 안에 있으며 contamination 또는 judge bias로 설명 가능한가?"

## Full-budget path

세 task, axis당 하나, 각각 `n=500` resample의 95% bootstrap CI. live rollout, uncached 기준 약 4 A100-hours.

1. **Reasoning — MATH-500** (Lightman 2023 split). `pass@1`. Slices: topic, problem length. Grader: boxed exact-match + sympy-equiv fallback ([[rlvr-tulu3]]).
2. **Chat — MT-Bench pairwise.** RL over SFT의 win-rate. Slices: 8 MT-Bench categories. Judge는 candidate 중 어느 쪽과도 *다른 model family*여야 한다([[judge-llm-bias]] self-enhancement rule).
3. **Safety — WildGuardTest + HarmBench-val.** 세 metric: `prompt_harm_acc`, `response_harm_acc`, `refusal_acc`. Slices: 13 WildGuard subcategories + HarmBench functional category(standard / copyright / contextual). Success label은 held-out Llama-2-13B-Chat classifier에서 온다([[harmbench-data]] §5).

## Resource-constrained path

reasoning task 하나와 safety task 하나. MT-Bench는 skip한다. ch-34 / ch-44 run에서 나온 **precomputed rollouts**(training 중 dump된 JSONL)를 사용한다. workstation CPU에서 wall clock 15-30 min. bootstrap CI와 contamination gate는 여전히 required다. judge-bias probe는 reasoning task의 length-controlled pair probe로 대체한다. rollout이 저장되어 있지 않다면 task마다 마지막 200 prompt를 deterministic seed로 다시 실행한다.

---

## §1. Harness skeleton

API shape는 `lm-eval-harness`와 OLMES([[olmes-harness-shape]])를 따른다. task는 declarative record이고, runner는 model-agnostic이며, slice는 sample의 metadata다.

```python
# harness/core.py  (reference: lm-eval-harness + OLMES)
@dataclass(frozen=True)
class Sample:
    sample_id: str
    prompt: str
    gold: str | list[str]
    slices: Mapping[str, str]           # {"topic": "algebra", "length_bucket": "long"}
    source: str

@dataclass(frozen=True)
class TaskSpec:
    name: str
    samples: Sequence[Sample]
    metric_fn: Callable[[str, str | list[str]], float]   # per-sample scalar in [0,1]
    generation_kwargs: Mapping[str, object]
    slice_keys: Sequence[str]
    contamination_scope: str            # which training-set dump to check

@dataclass(frozen=True)
class RunResult:
    task: str; model_id: str
    scores: list[tuple[str, float]]
    slice_table: dict[tuple[str, str], "SliceScore"]
    contamination: "ContaminationReport"
    cost: "CostReport"
```

harness는 `TaskSpec x ModelAdapter`를 실행한다. `ModelAdapter`는 live generation과 cached generation을 숨긴다. CI path는 cached를 사용한다. Karpathy의 "overfit one batch"([[karpathy-training-neural-net-recipe]])가 적용된다. sweep 전에 하나의 `Sample`을 가진 하나의 `TaskSpec`을 end-to-end로 실행하고 metric, slice aggregation, contamination gate가 모두 fire하는지 assert하라.

## §2. Task wrappers

각각 `list[Sample]`을 반환하는 세 wrapper. identifier와 count는 raw-data page에서 그대로 온다.

```python
# harness/tasks/math500.py  (reference: lets-verify.md; rlvr-tulu3 verifier)
def load_math500() -> list[Sample]:
    rows = load_jsonl("data/math500.jsonl")
    return [Sample(
        sample_id=f"math500-{r['id']}", prompt=r["problem"], gold=r["answer"],
        slices={"topic": r["subject"],
                "length_bucket": bucket(len(r["problem"]), [0, 200, 500, 1200])},
        source="math500") for r in rows]

def math_metric(pred: str, gold: str) -> float:
    # Tulu-3 verifier: boxed exact -> sympy equivalence -> 0.
    boxed = extract_boxed(pred)
    if boxed is None: return 0.0
    if boxed.strip() == gold.strip(): return 1.0
    try: return float(sympy_equiv(boxed, gold))
    except Exception: return 0.0
```

```python
# harness/tasks/wildguard.py  (reference: wildguard-data.md; 1,725 test pairs)
def load_wildguard_test() -> list[Sample]:
    rows = load_parquet("data/wildguardtest.parquet")
    out = []
    for r in rows:
        base = {"risk_subcat": r["subcategory"], "adversarial": str(r["adversarial"])}
        for task, text, gold in [
            ("prompt_harm",   r["prompt"], r["prompt_harm_label"]),
            ("response_harm", r["prompt"]+"\n"+r["response"], r["response_harm_label"]),
            ("refusal",       r["prompt"]+"\n"+r["response"], r["response_refusal_label"])]:
            out.append(Sample(f"wg-{task}-{r['id']}", text, gold,
                              {**base, "task": task}, "wildguard-test"))
    return out
```

HarmBench([[harmbench-data]])는 `harmbench_behaviors_text_val.csv`를 load한다. `slices`는 `semantic_category`와 `functional_category`를 담는다. `metric_fn`은 fine-tuned Llama-2-13B-Chat classifier다. HarmBench §5는 substring match를 명시적으로 금지한다. MT-Bench는 judge function을 쓴다(§5 참고).

## §3. Slice + bootstrap CI

Scoring module은 `(scores, slice_keys, samples)`를 받아 `(slice_key, slice_value)`마다 CI가 붙은 row 하나를 emit한다.

```python
# harness/score.py
@dataclass(frozen=True)
class SliceScore:
    slice_key: str; slice_value: str
    n: int; mean: float; ci_low: float; ci_high: float

def bootstrap_ci(xs, n_boot=500, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(xs) == 0: return (float("nan"), float("nan"))
    idx = rng.integers(0, len(xs), size=(n_boot, len(xs)))
    means = xs[idx].mean(axis=1)
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))

def slice_table(samples, scores, slice_keys, n_boot=500):
    sc = np.array([s for _, s in scores]); out = {}
    for key in slice_keys:
        for v in sorted({s.slices[key] for s in samples}):
            mask = np.array([s.slices[key] == v for s in samples])
            if mask.sum() == 0: continue
            lo, hi = bootstrap_ci(sc[mask], n_boot=n_boot)
            out[(key, v)] = SliceScore(key, v, int(mask.sum()),
                                       float(sc[mask].mean()), lo, hi)
    return out
```

항상 `(mean, ci_low, ci_high, n)`을 함께 보고하라. `n` 없는 point estimate는 small slice에서 noise와 구별할 수 없다([[scaling-laws-data-quality]]는 training side에서 parallel argument를 제시한다). `n<30`인 slice는 `insufficient`로 flag하고 regression call에서 제외한다.

## §4. Contamination gate

training-set dump에 대해 Lee 2021의 NearDup + ExactSubstr를 구현한다([[deduplicating-training-data]]). threshold는 paper에서 그대로 온다.

```python
# harness/contamination.py  (reference: deduplicating-training-data.md)
def build_train_lsh(train_docs, num_perm=9000, bands=20, rows=450):
    # Paper: 9000 signatures, 20 bands x 450 rows, Jaccard >= 0.8.
    threshold = (1.0 / bands) ** (1.0 / rows)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, params=(bands, rows))
    for i, d in enumerate(train_docs):
        m = MinHash(num_perm=num_perm)
        for sh in five_gram_shingles(d): m.update(sh.encode("utf8"))
        lsh.insert(f"train-{i}", m)
    return lsh

def exact_substring_hits(sample, suffix_array, min_len=50):
    # Paper threshold: duplicate substrings >=50 tokens.
    return count_longest_common_substrings(sample, suffix_array, min_len)

@dataclass(frozen=True)
class ContaminationReport:
    task: str; n_samples: int
    n_minhash_hits: int; n_exact_hits: int
    hit_rate: float; gate_triggered: bool

CONTAMINATION_MAX_RATE = 0.02   # paper: 4.6% LM1B / 3.2% C4 baseline; 2% is a strict gate
```

`hit_rate > 0.02`이면 score는 emit하지만 memo는 `contamination-fail`로 표시하고 checkpoint 비교를 거부한다. OLMo 3의 `OlmoTrace`가 production reference다([[olmo-3]]). ch-53 gate는 최소 standalone version이다.

## §5. Judge-bias probe

세 mitigation([[judge-llm-bias]]). pairwise number를 신뢰하기 전에 모두 실행하라.

```python
# harness/judge_probe.py  (reference: judge-llm-bias.md)
def position_swap_probe(pairs, judge_fn, n=80):
    # GPT-4 baseline flip ~22%; gate at 0.20.
    flips = sum(1 for p in pairs[:n]
                if judge_fn(p.prompt, p.a, p.b) != mirror(judge_fn(p.prompt, p.b, p.a)))
    return {"flip_rate": flips / n, "gate_triggered": flips / n > 0.20}

def length_controlled_probe(pairs, judge_fn):
    # Truncate longer candidate to shorter length. Flag verbosity if
    # raw vs length-ctrl win-rate differ by >5pp.
    ...

def self_enhancement_guard(judge_family, candidates):
    assert judge_family not in candidates, "Zheng 2023: judge family must not overlap"
```

Reference-guided grading은 MT-Bench에서 +10 pp agreement를 추가한다([[judge-llm-bias]]). math/coding category에는 gold answer를 붙여 judge-human agreement를 약 80%에서 약 90%로 올린다.

## §6. Checkpoint comparison

두 checkpoint를 모든 task에 통과시키고 per-slice regression을 계산한다.

```python
# harness/regress.py
def per_slice_regression(a, b, min_n=30):
    rows = []
    for (k, v), sa in a.slice_table.items():
        sb = b.slice_table.get((k, v))
        if sb is None or sa.n < min_n or sb.n < min_n: continue
        # CI-aware: regression only if b's CI is entirely below a's mean.
        rows.append({"slice": f"{k}={v}", "n_a": sa.n, "n_b": sb.n,
                     "mean_a": sa.mean, "mean_b": sb.mean,
                     "delta": sb.mean - sa.mean, "regressed": sb.ci_high < sa.mean})
    return rows
```

Tulu-3는 RLVR over SFT에서 `+5-10 pp GSM8K, +~4 pp IFEval`을 보고한다([[rlvr-tulu3]] via ch-44 §8). `ch-44`가 MATH-500에서 그 band 안에 있고 WildGuard refusal에서 regression하지 않는다면 RL stage는 healthy하다. WildGuard response-refusal이 `malicious_uses`에서 regressed한다면 aggregate average에는 보이지 않는 safety regression이 있다. 바로 slice table이 드러내기 위해 존재하는 failure mode다.

## §7. Go/No-Go memo template

memo는 downstream이 읽는 유일한 artifact다. fixed-header markdown:

```
# Eval Memo: <run_id>
## Verdict: <GO | NO-GO | NEEDS-FIX>
## Checkpoints: A=ch-34-sft(<sha>), B=ch-44-rl(<sha>)
## Gate status
- Contamination: <pass | FAIL task=<>, rate=<>>
- Judge bias:    <pass | FAIL flip=<>, verbosity delta=<>>
## Headline slices  (slice, n, mean_A[CI], mean_B[CI], delta, regressed?)
## Regressions      (regressed=True인 per-slice row)
## Known caveats    (pass@1 vs pass@large-k; contamination notes)
## Recommendation   (ship | re-run | ch-44 reread | slice X 조사)
```

gate fail이 하나라도 있으면 `NO-GO`다. gate는 pass하지만 headline slice regression이 있으면 `NEEDS-FIX`다. 그 외에는 `GO`다.

## Acceptance criteria

- Harness는 workstation에서 resource-constrained path로 end-to-end <30 min에 실행된다.
- 모든 reported score는 `n`과 `n_boot>=500`에서 나온 95% bootstrap CI를 함께 shipping한다.
- Contamination gate는 NearDup(MinHash 9000, 20x450 bands)와 ExactSubstr(min_len=50)를 모두 사용한다.
- Judge-bias probe는 position-swap flip-rate를 보고한다. flip>20%이면 pairwise number가 suppressed된다.
- `eval-memo.md`가 verdict, gate status, slice table, recommendation과 함께 존재한다.
- `tests/test_one_sample.py`가 single Sample / slice / contamination check / memo row에서 end-to-end pass한다.

## Connections

- [[ch-44]] — RLVR/PRM checkpoint; training-set dump와 rollout JSONL을 제공한다.
- [[ch-34]] — regression comparison의 SFT baseline.
- [[ch-52]] — tool-use eval precursor; BFCL AST matcher extension lane.
- [[ch-47]] — safety eval design; HarmBench / WildGuard layering origin.
- [[ch-45]] — self-improvement loop는 go/no-go memo를 early-stop signal로 consume한다.
- [[olmo-3]] — full lab-grade harness reference(OLMES + OlmoTrace).

## Further reading

- [[harmbench-data]] — safety slice를 위한 behavior/attack/classifier separation.
- [[wildguard-data]] — matched refusal/compliance pairs, 13-subcategory taxonomy.
- [[judge-llm-bias]] — 세 bias; swap-and-average와 length control은 optional이 아니다.
- [[deduplicating-training-data]] — contamination gate를 위한 n-gram threshold.
- [[karpathy-training-neural-net-recipe]] — harness wiring에 적용한 overfit-one-batch.
- [[ruler]] — long-context slice extension을 위한 RULER의 generator knob.
- [[bfcl]] — tool-use slice extension을 위한 AST matcher.
- [[scaling-laws-data-quality]] — eval time에서 slice-level `n`이 중요한 이유.

## Companion visualization

`figures/eval-pipeline.html`은 harness를 stage-by-stage(config -> run -> slice -> contamination -> judge probe -> memo)로 보여 준다. 어떤 stage든 click하면 input/output을 볼 수 있다. toggle은 contamination hit, position-swap flip, safety slice regression을 inject하고 memo verdict가 update된다. §6 전에 열어 보라.
