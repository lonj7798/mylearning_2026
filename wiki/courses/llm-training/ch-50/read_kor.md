<!-- chapter: ch-50
     track: eval
     kind: content
     title: Slice Analysis and Failure Bucketing
     deps: [ch-49]
     sources: [[llama-3]], [[olmo-2]], [[olmo-3]], [[tulu-3]], [[longbench]], [[ruler]], [[bfcl]], [[physics-of-lm-3]], [[interplay-pretraining-midtraining-rl]], [[karpathy-training-neural-net-recipe]]
     figures: figures/slice-report.html
-->

# 50장 — Slice Analysis와 Failure Bucketing

> **핵심 통찰.** aggregate number 하나만으로는 checkpoint가 더 좋아졌는지 알 수 없다. 현대의 모든 post-training report, 즉 [[llama-3]]의 six-round loop, [[tulu-3]]의 SFT -> DPO -> RLVR ledger, [[olmo-2]]의 stage-by-stage contribution table, [[ruler]]의 13-task suite, [[bfcl]]의 7개 category는 같은 움직임 위에 세워져 있다. eval을 slice로 분해하고, checkpoint 사이의 per-slice delta를 측정하며, 실패한 rollout을 이름 붙은 bucket으로 cluster한 다음에야 "ship" 또는 "regress"를 결정한다. aggregate는 slide에 올리기 위해 존재한다. per-slice report는 결정을 내리기 위해 존재한다. 어떤 것을 만들어야 하는지 아는 것이 이 장이 이름 붙이는 skill이다.
>
> **가이드라인.** checkpoint를 평가할 때는 항상 per-slice score를 먼저 계산하고 aggregate를 나중에 계산하라. aggregation은 signal을 버리므로, signal을 *본 뒤에* 버려라. 다음 decision이 의존하는 축으로 slice하라("이걸 ship할까?"에는 capability, "RL stage가 실제로 일하고 있나?"에는 difficulty bucket, "judge가 biased인가?"에는 judge-category). 두 checkpoint 비교에서는 item-level score에 paired bootstrap을 쓰고, signed delta의 95% CI가 0을 제외하면서 effect size가 pre-registered threshold를 넘을 때만 slice를 *regressed*로 flag하라. 실패는 surface가 아니라 *reason*으로 cluster하라. canonical reason은 [[reward-hacking-taxonomy]]식이다. run 사이에 지속되는 named bucket의 failure-ledger를 유지하라. run 내에서 column을 읽는 것보다 run across row를 읽는 것이 더 유익하다. go/no-go communication에는 three-line chart를 쓰고, regression triage에는 50-slice report를 써라. 둘을 뒤집지 마라.

---

## §1 aggregate가 이야기를 숨기는 이유

[[tulu-3]] Figure 1은 RLVR stage가 DPO-only checkpoint 대비 *aggregate average에서 +0.0 to +2.0 pp*라고 보고한다. aggregate는 release post의 punchline이다. 하지만 *report*는 다른 이야기를 한다.

- **GSM8K:** +5 to +10 pp.
- **MATH:** +5 to +10 pp.
- **IFEval:** +~4 pp.
- **MMLU / TruthfulQA / AlpacaEval:** neutral to slightly negative.

aggregate-average mean은 RL-verifiable slice에서의 약 10 pp gain과 knowledge slice의 작은 regression을 가린다. aggregate > 0일 때만 ship한다면 RLVR을 ship한다. 하지만 user가 MMLU를 중요하게 생각한다면 방금 regression을 낸 것이다. three-line chart(stage별 aggregate)는 blog post에 맞다. 50-slice report는 release decision에 맞다. 어느 쪽도 "더 낫다"가 아니다.

[[olmo-2]]의 release table도 stage별로 같은 점을 보인다. DPO는 chat-quality와 IFEval lift를 만든다. RLVR은 GSM8K/MATH lift를 만든다. stage를 한 row로 merge하면 attribution을 잃는다. row를 분리해 두면 "어느 stage를 다시 돌려야 하는가?"라는 실제 downstream question에 답할 수 있다.

## §2 Per-slice vs per-task: 두 분해, 하나의 규칙

"slicing"이라고 불리는 서로 다른 두 축이 있다. 이들은 합성되며, 서로 대체하지 않는다.

- **Per-task** — eval을 dataset별로 나눈다: MMLU, GSM8K, HumanEval, IFEval, BFCL-V3, RULER-64K, LongBench-en, AlpacaEval-LC. 이것이 Llama 3 / OLMo 2 / Tulu 3 release-table format이다. task당 row 하나, checkpoint당 column 하나다.
- **Per-slice** — task *내부*를 prompt attribute로 나눈다. [[longbench]]의 6 task category 또는 21 sub-task, [[ruler]]의 13 generator configuration(S-NIAH / MK-NIAH / MV-NIAH / MQ-NIAH / VT / CWE / FWE / QA x length), [[bfcl]]의 7 scoring category(simple, multiple, parallel, parallel-multiple, relevance-detection, Java, JS), GSM8K by difficulty bucket, MMLU by domain(STEM, humanities, social sciences, other)이 여기에 해당한다.

**규칙.** 둘 중 하나만 할 수 있다면 per-slice로 가라. single-task per-slice report는 per-task가 잡지 못하는 failure를 잡는다. [[ruler]] Table 3이 이를 입증한다. 32K context를 claimed하는 model이 S-NIAH에서는 약 99%를 기록한다(per-task view: 괜찮아 보임). 하지만 MK-NIAH-full-haystack에서는 약 40%, VT-4-hops에서는 약 20%로 무너진다(per-slice view: effective window는 8K). per-task number는 실제 8K context인 model을 32K marketing model로 ship했을 것이다. [[longbench]]는 반대 관찰도 한다. NarrativeQA는 4K context에서 풀 수 있으므로 "LongBench score"는 long-range slice가 regressed해도 easy slice의 QA가 좋아지는 것만으로 오를 수 있다.

**MMLU-by-domain 예시.** Checkpoint B가 MMLU aggregate에서 Checkpoint A를 +0.5 pp 이긴다. domain별로 보면(대표 split을 추정한 것이며 정확한 Hendrycks category는 아님):

| Slice | A | B | Δ |
|---|---|---|---|
| STEM | 58.2 | 60.8 | +2.6 |
| Humanities | 62.1 | 61.4 | -0.7 |
| Social Sci. | 65.3 | 65.9 | +0.6 |
| Other | 59.0 | 58.1 | -0.9 |
| **Aggregate** | 61.1 | 61.6 | +0.5 |

aggregate는 win을 보여 준다. per-slice report는 Humanities와 Other가 regressed했음을 보여 준다. RL stage가 math-specific SFT mix([[tulu-3]] §SFT mixture)를 도입했다면, 그 regression은 [[interplay-pretraining-midtraining-rl]]의 "edge-of-competence" story의 signature다. RL은 base model에 headroom이 있던 slice에서만 도움을 준 것이다. ship 여부는 Humanities regression이 noise 안에 있는지에 달려 있다(§4).

**GSM8K-by-difficulty 예시.** [[interplay-pretraining-midtraining-rl]]의 headroom-and-boundary experiment는 difficulty별 slicing 위에 세워져 있다. GSM8K를 gold solution의 step count로 bucket하라. easy(1-2 steps), medium(3-4), hard(5+). post-RL checkpoint가 easy에서 +8 pp, medium에서 +6 pp, hard에서 -1 pp를 얻는다면 이것은 "math가 좋아졌다"가 아니다. 이미 가까이 있던 edge 쪽으로 mass를 옮긴 것이며, hard slice가 다음 RL round에서 일해야 할 곳이다. aggregate GSM8K는 +5 pp를 보고하고 extrapolation failure를 숨겼을 것이다.

## §3 Failure bucketing — surface가 아니라 reason으로 cluster하라

aggregate accuracy는 *얼마나 많이* 실패했는지 말한다. bucketing은 *왜 같은 방식으로* 실패했는지 말한다. 두 clustering axis가 필요하며, 둘 다 필요하다.

**Cluster-by-reason.** [[karpathy-training-neural-net-recipe]]의 "review the 10 worst validation examples — they reveal systematic errors"가 문자 그대로의 출발점이다. 실패한 rollout 30-50개를 읽고, fixed ontology의 label 하나를 붙이며, coverage가 >90%가 될 때까지 ontology를 iterate한 다음 count한다. post-trained chat model을 위한 쓸 만한 starter ontology는 다음과 같다: `format-violation`, `refusal-when-answerable`, `hallucination-fact`, `hallucination-tool`, `arithmetic-error`, `reasoning-skip-step`, `length-hack`, `sycophancy`, `language-drift`, `stale-knowledge`, `other`. [[reward-hacking-taxonomy]]는 RL-relevant subset을 제공한다. [[bfcl]]의 relevance-detection column은 `call-when-irrelevant`를 직접 측정한다.

**Cluster-by-confusion-matrix.** task가 열거 가능한 output을 갖는다면 confusion matrix를 만들어라. [[bfcl]]은 이를 암묵적으로 수행한다. AST matcher는 모든 failure를 다음 중 하나로 분해한다. name-mismatch(wrong function), kwargs-mismatch(right function, wrong params), hallucinated-call(called when irrelevant), missing-call(didn't call when needed), format-unparseable. 이들은 orthogonal bucket이다. function-calling model은 name-mismatch를 개선하면서 hallucinated-call에서는 regressed할 수 있다. 이것이 확인된 2025년 패턴이다(frontier model도 irrelevant query에서 여전히 약 10% hallucinate한다).

### Pipeline pseudocode

```python
# eval/bucket_failures.py — one pass over rollouts, two cluster outputs.
from collections import Counter, defaultdict

REASON_ONTOLOGY = [
    "format-violation", "refusal-when-answerable", "hallucination-fact",
    "hallucination-tool", "arithmetic-error", "reasoning-skip-step",
    "length-hack", "sycophancy", "language-drift", "stale-knowledge", "other",
]

def bucket_failures(rollouts, grader, reason_tagger, confusion_tagger=None):
    # rollouts: [{prompt, response, gold, slice_tags: {task, domain, difficulty, length}}]
    by_reason = defaultdict(Counter)        # slice -> Counter(reason)
    by_confusion = defaultdict(Counter)     # slice -> Counter(confusion_cell)
    correct = defaultdict(int); total = defaultdict(int)
    for r in rollouts:
        slice_key = (r["slice_tags"]["task"], r["slice_tags"].get("domain", "_"))
        total[slice_key] += 1
        if grader(r["response"], r["gold"]):
            correct[slice_key] += 1
            continue
        reason = reason_tagger(r)           # LLM-judge or regex ontology
        assert reason in REASON_ONTOLOGY
        by_reason[slice_key][reason] += 1
        if confusion_tagger:
            by_confusion[slice_key][confusion_tagger(r)] += 1
    return {"accuracy": {k: correct[k]/total[k] for k in total},
            "by_reason": dict(by_reason), "by_confusion": dict(by_confusion)}
```

두 output은 두 decision에 답한다. `by_reason`은 "가장 큰 failure bucket을 해결하는 training-data fix는 무엇인가?"에 답한다. data question이다. `by_confusion`은 "가장 큰 win을 예측하는 metric move는 무엇인가?"에 답한다. objective/reward question이다. 대부분의 team은 하나만 만든다. §5의 report table은 둘 다 요구한다.

### LLM-judge reason-tagger — 2025년 기본값

strict rubric을 가진 capable LLM이 확인된 reason-tagger다. [[tulu-3]]의 safety-specific DPO slice가 이런 방식으로 만들어졌다. rubric이 ontology다. 각 failure는 하나의 label과 한 문장짜리 justification을 받는다. bucket count를 신뢰하기 전에 100-rollout calibration set에서 [[ch-49]]의 judge bias check, 즉 position swap과 length control을 실행하라. cluster label은 judge가 human spot-check와 일치하는 정도만큼만 신뢰할 수 있다(target: kappa >= 0.7).

## §4 checkpoint 비교 — regression은 언제 real인가?

signed per-slice delta가 report의 단위지만, single-run delta는 noisy measurement다. [[karpathy-training-neural-net-recipe]]의 predict-before-run discipline을 적용하라. scoring 전에 모든 slice의 expected direction을 적고, observed delta가 (a) 0과 statistically distinguishable하고 (b) matter할 만큼 큰지 test하라.

### Paired bootstrap — 싸고 올바른 test

두 checkpoint가 slice의 *같은* N item에서 scored되었을 때:

```python
# eval/paired_bootstrap.py
import numpy as np
def paired_bootstrap_ci(score_a, score_b, n_boot=10_000, seed=0):
    # score_a, score_b: per-item scalar scores for checkpoint A, B (same items, same order).
    rng = np.random.default_rng(seed)
    diff = np.asarray(score_b) - np.asarray(score_a)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)
```

slice마다 `(mean_delta, CI_low, CI_high)`를 보고하라. "signed regression" 규칙은 다음과 같다.

- `CI_high < 0` — real regression이다. 조사하라.
- `CI_low > 0` — real gain이다. size가 중요하면 ship하라.
- `CI` crosses 0 — 이 slice에서는 inconclusive다. sibling과 aggregate하거나 data를 더 모아라.

### Effect-size threshold — noise 쫓기를 막아 준다

500-item slice에서 0.3 pp gain은 statistically significant할 수 있지만 operationally meaningless할 수 있다. slice별 effect-size threshold를 pre-register하라. 예: knowledge task(MMLU-domain)는 `|Δ| >= 1.0 pp`, reasoning(GSM8K/MATH)은 `|Δ| >= 2.0 pp`, narrow-verifier slice(IFEval constraint-category)는 `|Δ| >= 5.0 pp`. [[tulu-3]]의 확인된 RLVR gain(+5-10 pp GSM8K)은 2.0 threshold를 여유 있게 넘는다. neutral MMLU move(~+0.2 pp)는 1.0 threshold를 넘지 못한다. 정직한 report는 "GSM8K improved, MMLU held"라고 말한다.

### Two-run minimum — seed 하나는 sample이 아니다

single training run의 per-slice delta는 seed change에서 뒤집힐 수 있다. [[llama-3]]는 이를 암묵적으로 완화한다. six-round iterative loop는 모든 slice가 round마다 fresh RM에 대해 다시 측정되게 하므로 one-round flicker가 씻겨 나간다. one-shot run에서는 같은 data + LR로 training seed를 두 개 돌리는 것이 싼 방어책이다. sign이 seed across consistent한 per-slice delta를 "real"로 취하고, flip되는 것은 "unresolved"로 둔다. ch-51은 seed-variance budget을 정량화한다.

### Unsigned delta — 다른 질문

signed delta는 "slice가 개선되었는가?"에 답한다. unsigned `|Δ|`는 "checkpoint의 per-slice profile이 *shape*을 바꾸었는가?"에 답한다. signed delta는 작지만 unsigned delta가 크다면, model이 이전과 다른 item을 풀고 있다는 뜻이다. capability shift 없이 behaviour shift가 있는 것이다. flag할 가치가 있다. [[interplay-pretraining-midtraining-rl]]의 "reshape vs expand" question이 정확히 이 구분이다. aggregate가 같더라도 reshape만 하는 RL stage와 expand하는 RL stage는 같지 않다.

## §5 Failure-ledger — run across bucket은 어떤 모습인가

한 장의 report는 snapshot이다. *ledger*는 run across row다. `eval/failure-ledger.jsonl`을 유지하라. 각 row는 `{run_id, checkpoint, slice, bucket, count, share, example_ids[], first_seen_run, fix_attempted}`다. ledger row를 6개월에 걸쳐 읽으면 bucket이 줄고 있는지(fix working), 늘고 있는지(regression), oscillate하는지(unresolved)를 볼 수 있다. [[llama-3]]의 "failure modes" section은 이 ledger의 공개 snapshot이다. preference-data noise와 multi-turn dialog drift는 round 사이에 *persisted*한 named bucket이지 일회성 anomaly가 아니다.

raw-data source에서 나온 canonical bucket은 ledger의 각 row가 된다.

- **`length-hack`** ([[reward-hacking-taxonomy]], [[dr-grpo]]) — win이 `len(response) > len(gold) * 1.3`와 correlate한다. Diagnostic: `reward-vs-length` scatter.
- **`verifier-loophole`** ([[rlvr-tulu3]]) — lax grader가 prose 안의 `\boxed{42}`를 accept한다. Diagnostic: held-out slice에서 strong-verifier scoring.
- **`format-violation-IFEval`** ([[tulu-3]]) — constraint-category confusion matrix. Diagnostic: per-constraint-type pass-rate.
- **`long-context-distractor-density`** ([[ruler]] MK-NIAH-full-haystack) — distractor needle이 haystack을 saturate할 때 accuracy가 >30 pp 떨어진다. Diagnostic: S-NIAH vs MK-NIAH ratio.
- **`long-context-aggregation`** ([[ruler]] CWE, FWE) — retrieval은 유지되지만 32K 이후 aggregation이 collapse한다. Diagnostic: CWE-score vs S-NIAH-score ratio.
- **`call-when-irrelevant`** ([[bfcl]] relevance-detection) — tool-hallucination bucket. Diagnostic: relevance-detection sub-score.
- **`multi-turn-state-drift`** ([[bfcl]] V3) — single-turn은 괜찮지만 multi-turn이 collapse한다. Diagnostic: pass@1 single vs pass^k multi-turn.
- **`language-drift`** ([[llama-3]] multilingual) — English prompt에는 English로 답하고, Chinese prompt는 English로 drift한다. Diagnostic: prompt-language별 language-match rate.
- **`stale-knowledge`** ([[physics-of-lm-3]]) — 최근 업데이트된 fact의 factual-tuple retrieval이 실패한다. Diagnostic: fact-freshness별 split eval.
- **`reasoning-skip-step`** ([[interplay-pretraining-midtraining-rl]]) — answer는 correct지만 trace는 broken이다. Diagnostic: process-reward score.

각 bucket에는 현대 pipeline에 이미 logging되는 *diagnostic signal*이 있다. ledger는 그것을 추출하고 보존할 뿐이다.

## §6 Three-line chart vs 50-slice report — decision matrix

두 artifact 모두 존재한다. decision에 따라 고른다.

| Decision | Audience | Format | Example |
|---|---|---|---|
| 이 checkpoint를 ship할까? (go/no-go) | leadership, release manager | three-line chart: aggregate per stage (pretrain / SFT / DPO / RLVR) | [[tulu-3]] Figure 1 |
| 어느 stage가 regressed했나? | training lead | per-stage x per-task table | [[olmo-2]] post-training gain table |
| task 내부의 어느 slice가 regressed했나? | eval lead | paired-bootstrap CI가 있는 50-row per-slice report | [[ruler]] Table 3 style |
| 이 item들은 왜 실패했나? | data lead | failure-bucket table(reason-clusters + confusion matrix) | [[bfcl]] category breakdown |
| 이 bucket이 run across 줄고 있나? | program manager | ledger row across runs | [[llama-3]] §Failure modes |
| 이 capability가 edge of competence에 있나? | research lead | difficulty별 pass-rate histogram | [[interplay-pretraining-midtraining-rl]] headroom |

**규칙.** regression triage에 three-line chart를 쓰지 마라(slice를 놓친다). go/no-go에 50-slice report를 쓰지 마라(leadership은 머릿속에서 나쁘게 aggregate하고 잘못된 slice를 optimize한다). chart와 report는 서로 다른 질문에 대한 답이다. 좋은 eval harness는 같은 evidence table에서 둘 다 생성한다. chart는 report의 `GROUP BY stage` aggregate이며, 둘 중 어느 쪽도 `failures.parquet`에서 function call 하나로 regenerate된다.

**Format-first가 아니라 decision-first design.** 흔한 anti-pattern은 "benchmark X를 돌리고, score를 보고하고, 끝"이다. 이는 "its score" 안에 이미 slicing choice가 baked in되어 있기 때문에 실패한다. [[longbench]]의 21 task equal-weighted mean, [[ruler]]의 `wAvg. (inc)` vs `wAvg. (dec)`, BFCL의 micro vs macro average가 모두 그렇다. 그중 어느 것도 *당신의* decision을 염두에 두고 고른 것이 아니다. ch-50은 순서를 뒤집는다. decision에서 시작하고, 그 decision에 답하는 slicing과 aggregation을 고른 뒤 report를 만든다. [[ruler]]의 두 weighted average(`inc` vs `dec`)가 가장 깨끗한 공개 예시다. 저자들은 하나의 headline에 commit하기를 거부했다. 두 질문("long end에서 작동하는가?" vs "overall 작동하는가?")은 두 답을 필요로 하기 때문이다.

## §7 Worked example — single checkpoint release를 위한 report staging

구체화하자. Tulu 3 recipe로 7B base에 post-training run을 끝냈다고 하자. artifact를 순서대로 만든다.

1. **Evidence table.** `failures.parquet` — `(prompt_id, checkpoint_id, score, slice_tags, response, gold, judge_reason)`마다 row 하나. 모든 downstream artifact는 이것에 대한 query다.
2. **Three-line chart.** evidence table에서 `GROUP BY stage`를 수행하면 세 row가 나온다. SFT checkpoint, DPO checkpoint, RLVR checkpoint. task category별 aggregate accuracy를 plot한다. 이것이 ship-decision summary다.
3. **Per-task table.** `GROUP BY stage, task`를 수행하고 cell마다 paired-bootstrap CI를 붙인다. "어느 stage가 어느 task를 들어 올렸나"에 답한다. [[olmo-2]]의 확인된 pattern(`DPO -> IFEval`, `RLVR -> GSM8K/MATH`)이 comparison template이다.
4. **Regressed한 task의 per-slice report.** `GROUP BY stage, task, slice`와 signed delta를 계산한다. §4의 effect-size threshold를 적용하고, `CI_high < 0 and |Δ| >= threshold`인 row를 flag한다.
5. **Flagged slice의 bucket table.** 그 slice의 failed rollout에 대해 §3의 `bucket_failures()`를 실행한다. top five bucket과 cross-run ledger row(shrinking / growing / stuck)를 보고한다.
6. **Go/no-go memo.** 한 page. chart + 각 regressed bucket을 proposed fix에 연결하는 one-paragraph narrative. [[karpathy-training-neural-net-recipe]]의 predict-before-run discipline은 여기에 있다. memo는 어떤 regression이 predicted(acceptable)였고 어떤 것이 unpredicted(require investigation)였는지 인용한다.

여섯 artifact, 같은 evidence table, 여섯 audience다. 하나의 query layer에서 여섯 artifact를 모두 만들 수 없는 harness는 under-built다. eval side에 적용한 [[olmo-3]]의 model-flow philosophy(stages as auditable artifacts)다.

## §8 이름 붙여야 할 anti-pattern

이 장의 guideline이 금지하는 mistake의 짧은 목록:

- **Weighted-average trick.** aggregate가 원하는 story를 말하도록 slice weight를 바꾸는 것. [[longbench]]의 NarrativeQA 사례에서는 easy slice를 up-weight해 mean을 끌어올릴 수 있다. 해결책은 pre-registration이다. running 전에 slice weight에 commit하라.
- **Compare-different-slicings.** Run A는 6개 MMLU domain으로 보고됐고, Run B는 2개(STEM vs humanities)로 보고됐다. delta는 비교할 수 없다. 해결책: benchmark마다 하나의 fixed slicing spec을 versioning한다.
- **Judge-as-grader without calibration.** LLM-judge reason-tagger label은 judge model이 바뀌면 drift한다. ch-49의 calibration discipline은 judge가 바뀔 때마다 다시 적용되어야 한다. 그렇지 않으면 run across ledger count는 apples와 retrained apples를 비교한다.
- **Cherry-picked worst examples.** "Review the 10 worst"는 "reviewer가 10개 worst를 읽고 가장 창피한 3개만 골라 나머지를 무시한다"가 아니다. 해결책: 10개 모두 tag하고, narrative가 아니라 tag histogram을 보고한다.
- **Aggregating across sizes.** Tulu 3는 size별로 β를 다시 tune한다. 7B와 70B eval score를 하나의 aggregate로 섞으면 size-specific per-slice optimum을 지운다. 해결책: size는 slice axis이지 aggregation axis가 아니다.
- **Reporting only improvements.** regression을 post-hoc filtering해 report에서 빼는 것. ledger는 이를 막기 위해 존재한다. shipped fix 없이 report에서 사라지는 failing bucket은 reporting process의 bug이지 model의 개선이 아니다.

---

## Companion visualization

**[figures/slice-report.html](figures/slice-report.html)** — interactive mock eval report. "aggregate-only" view(three lines)와 "per-slice" view(50 rows)를 toggle하라. aggregate에서는 보이지 않던 regression이 per-slice toggle 아래 signed bar로 나타나는 것을 보라. 두 번째 panel은 slice별 reason-count가 있는 failure-bucket table과 BFCL-style function-calling slice의 confusion-matrix heatmap이다. 숫자는 illustrative다. effect의 *shape*은 [[tulu-3]], [[ruler]] Table 3, [[bfcl]] category breakdown으로 확인된다. §6의 decision matrix를 읽기 전에 toggle을 사용하라. slice-bar가 aggregate verdict를 뒤집는 것을 보면 규칙이 이해된다.

## Connections

- **ch-47 (eval harness)** — 이 장이 slice하는 evidence table을 만든다. slicing에는 per-task mean이 아니라 item-level score가 필요하다.
- **ch-48 (benchmark zoology)** — per-task dataset의 taxonomy다. 이 장은 그 위의 per-slice layer다.
- **ch-49 (judge calibration)** — reason-tagger judge 자체가 calibrated되어야 한다. 그렇지 않으면 bucket count는 세탁된 bias다.
- **ch-51 (variance & go/no-go memo)** — two-run minimum, effect-size threshold, three-line chart가 들어가는 memo structure를 formalize한다.
- **ch-42 ([[reward-hacking-taxonomy]])** — ledger에 들어갈 named bucket을 제공한다.
- **ch-44 ([[rlvr-tulu3]])** — verifier-loophole bucket은 여기서 나온다.

## Further reading

- [[tulu-3]] Figure 1 + §RLVR — per-stage per-task ledger.
- [[olmo-2]] post-training gain table — stage-attribution discipline.
- [[olmo-3]] model-flow diagram — stage-as-artifact framing.
- [[llama-3]] §Failure modes — published failure-ledger snapshot.
- [[longbench]] §Risks + gotchas — per-task가 왜 오도할 수 있는지(NarrativeQA solvable at 4K).
- [[ruler]] Table 3 + Figure 3 — per-task가 아니라 per-slice를 통한 effective-context.
- [[bfcl]] scoring categories — canonical confusion-matrix decomposition.
- [[interplay-pretraining-midtraining-rl]] — edge-of-competence slicing; reshape-vs-expand.
- [[physics-of-lm-3]] — per-capability slicing을 동기화하는 knowledge-capacity framing.
- [[karpathy-training-neural-net-recipe]] — failure bucketing의 문자 그대로의 뿌리인 "review the 10 worst".
