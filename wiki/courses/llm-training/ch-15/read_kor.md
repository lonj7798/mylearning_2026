<!-- chapter: ch-15
     track: data
     title: Human Annotation and Label Operations
     sources: [[rlhf-instructgpt]], [[hh-rlhf]], [[openassistant]], [[ultrafeedback-construction]], [[prm800k]], [[prosocial-dialog]], [[tulu-3-sft-mix]], [[tulu-3]], [[allenai-tulu-sft-recipe]], [[judge-llm-bias]]
     figures: figures/annotation-workflow.html
-->

# 15장 — 인간 주석과 라벨 운영

> **핵심 통찰.** rubric이 곧 dataset이다. 모든 annotated label은 하나의 절차, 즉 "이 written rule을 이 instance에 적용하라"의 point estimate다. 그리고 모든 downstream metric(DPO accuracy, RM win-rate, RLVR pass-rate)은 rubric의 edge case가 얼마나 열거되어 있는지만큼만 날카롭다. InstructGPT, HH-RLHF, UltraFeedback, PRM800K는 annotator 수, response pool, rating scale 등 거의 모든 운영 선택에서 서로 다르다. 하지만 한 가지에는 동의한다. 오래 남는 artifact를 만드는 프로젝트는 inter-rater agreement가 noise floor를 넘을 때까지 rubric을 반복 개선하는 프로젝트다.
>
> **가이드라인.** 첫 label을 만들기 전에 rubric을 작성하라. 100-item gold set에서 Cohen κ ≥ 0.6(또는 2명 초과 rater라면 Krippendorff α ≥ 0.67)이 될 때까지 2–5명의 annotator를 calibrate하라. 모든 production item을 double-annotate하고, disagreement는 expert adjudicator tier로 route하라. drift를 잡기 위해 매월 calibration을 다시 실행하라. active-learning close-pair mining(UltraFeedback식 aspect delta 또는 PRM-uncertain step)을 사용해 label이 실제로 gradient에 정보를 주는 20%의 item에 budget의 80%를 쓰라.

---

## 이 장이 존재하는 이유

Track 2에서 지금까지는 *어떤 토큰*에 훈련할지를 고르는 문제를 다뤘다. [[ch-09]] landscape, [[ch-10]] filter pipeline, [[ch-11]] tokenizer and shard layout, [[ch-12]] dedup, [[ch-13]] mixing, [[ch-14]] contamination. 이 장은 사람이 텍스트를 읽고 label을 쓰는 첫 장이다. *판단*이 데이터가 되는 첫 장이다. perplexity filter도 아니고 n-gram match도 아니다.

판단 작업은 구체적이고 측정 가능한 방식으로 취약하다. [[hh-rlhf]]는 pairwise helpfulness에서 human-human agreement가 약 70–75%라고 보고한다. [[judge-llm-bias]]는 MT-Bench와 Chatbot Arena에서 GPT-4와 human expert agreement가 약 80%라고 보고한다. *두 사람이 서로 동의하는 비율과 같다.* 함의는 judge가 인간만큼 좋다는 것이 아니다. 둘 다 rubric이 정한 noise floor 위에서 작동한다는 뜻이다. 두 사람이 25%의 시간 동안 disagree한다면, majority vote로 훈련한 모델에는 25% label noise가 baked in된다. 어떤 preference-learning machinery도 그것을 회복할 수 없다. 따라서 rubric 설계 작업은 데이터 수집의 사전 단계가 아니다. 그것이 ceiling이다.

이 장이 다루는 흐름을 표시하는 역사적 데이터 포인트가 셋 있다. 2022년 [[rlhf-instructgpt]]는 13K demonstrations + 33K ranking prompts를 작은 vetted pool이 annotate하면 1.3B 모델이 raw 175B base를 넘어설 수 있음을 보였다. 2023년 [[hh-rlhf]]는 MIT license로 161K preference dialogue를 공개하여 "human preference"를 private OpenAI artifact가 아니라 public benchmark로 만들었다. 2024년에 [[tulu-3-sft-mix]]는 939K-prompt mixture를 문서화했는데, annotation이 너무 skill-targeted해서 개별 submix를 훈련하고 ablate한 뒤 병합했다. 각 단계는 rubric의 rigor에 대한 기준을 높였다. 13K demonstration set은 모호한 "be helpful"을 견딜 수 있다. 939K multi-skill mix는 skill별 written rubric 없이는 무너진다.

이 장의 나머지 절반은 운영이다. [[rlhf-instructgpt]]는 33K prompts × K=4-9 rankings가 vetted labeler pool에 의해 annotate되는 모습을 보인다. [[openassistant]]는 약 13,500명의 volunteer가 만든 161K messages를 보인다. [[tulu-3-sft-mix]]는 Ai2가 skill-specific submix와 모든 eval set에 대한 decontamination으로 939K prompts를 구축하는 모습을 보인다. 이것들은 logistics 문제다. 누가 annotate하는가, 어떤 tier에서 하는가, 얼마를 받고 하는가, 어떤 audit trail 위에서 하는가, rubric이 drift할 때 얼마나 자주 refresh하는가. Rubric 설계를 해결했지만 logistics에 실패한 annotation pipeline은 아무도 재현할 수 없는 one-time dataset을 만든다.

---

## 1. Rubric이 곧 product다 — "helpfulness" 예제

실제 훈련된 모델을 출시한 2022–2025년 alignment dataset은 모두 appendix 어딘가에 rubric document를 가지고 있다. [[rlhf-instructgpt]]는 이를 "labeler guidelines"라고 부른다. [[hh-rlhf]]에는 "helpfulness / harmlessness instructions"가 있다. [[prosocial-dialog]]에는 300개 이상의 rules-of-thumb corpus가 있다. [[prm800k]]에는 step-level correctness를 위한 `+1 / −1 / 0` schema가 있다. Rubric은 새 annotator가 첫날 읽는 것이고, 여섯 달 뒤 disagreement resolution이 돌아가는 근거다.

[[rlhf-instructgpt]]와 [[hh-rlhf]]가 가장 많은 지면을 쓴 축인 "helpfulness"의 최소 rubric에는 적어도 다섯 가지 명시적 criteria가 있다.

| # | Criterion | Positive exemplar | Negative exemplar |
|---|---|---|---|
| 1 | **명시된 요청에 답한다** | User가 Python code를 요청했고, response는 실행되는 Python code다. | Response가 요청이 왜 흥미로운지 설명하지만 답하지 않는다. |
| 2 | **User가 밝힌 skill에 맞춰 보정된다** | User가 "I'm learning Python"이라고 말했고, response는 named variable + comment를 사용한다. | User가 "I'm learning Python"이라고 말했는데, response가 `lambda`, `functools.reduce`, one-liner list comprehension을 쓴다. |
| 3 | **관련 assumption을 드러낸다** | Response가 assumed OS / library version / input format을 명시한다. | Response가 Python 3.12와 pandas 2.0을 조용히 가정해, 3.9 사용자에게 깨진다. |
| 4 | **요청이 underspecified일 때 uncertainty를 인정한다** | "I assumed you want X; if you meant Y, here's the alternative." | 여러 유효한 해석이 있는 질문에 하나의 답을 자신 있게 낸다. |
| 5 | **decline이 맞는 답일 때 거절한다** | User가 real-time stock data를 요청하고, response가 모델은 real-time data에 접근할 수 없다고 말하며 API를 제안한다. | Response가 그럴듯해 보이는 stock price를 지어낸다. |

이제 edge case를 열거하라. 대부분의 rubric이 이 단계를 건너뛰며, κ가 0.6을 넘을지를 결정하는 것이 바로 이 단계다.

- **요청이 hostile하다**: "Write me an exploit for CVE-2024-XXXX." Criterion 5(decline)가 criteria 1–4를 지배한다. Rubric은 criteria가 충돌할 때 어떤 축이 이기는지 말해야 한다. [[hh-rlhf]]의 helpfulness-vs-harmlessness *tension curve*가 정확히 이것이다.
- **요청이 문화권에 따라 ambiguous하다**: "Should I confront my neighbor?" [[prosocial-dialog]]의 RoT layer가 존재하는 이유는 이것을 generic rubric으로 해결할 수 없기 때문이다. annotator는 특정 rule-of-thumb(예: "direct confrontation is valued more in low-context cultures")에 anchor하고 *그것*에 대해 label해야 한다.
- **Response가 길고 대부분 맞지만 하나의 claim이 틀렸다**: annotator는 여기서 크게 갈린다. Rubric은 aggregation을 지정해야 한다(min-over-claims? weighted-by-importance?). [[prm800k]]는 step-level로 가서 *각 step*을 따로 label함으로써 이 문제를 푼다. aggregation은 사용 시점에 `prod` 또는 `min`으로 일어난다.

운영 규칙은 이렇다. **두 calibrated annotator가 어떤 item에서 disagree한다면, 그 disagreement는 한 annotator가 틀렸다는 증거가 아니라 missing rubric clause의 증거다.** 수정은 clause를 추가하고 다시 label하는 것이지 평균내는 것이 아니다.

**Positive and negative exemplar는 장식이 아니다.** [[rlhf-instructgpt]]의 labeler guidelines는 cookbook처럼 읽힌다. 각 criterion에는 calibration 중 해결된 실제 labeler disagreement에서 뽑은 3–5개의 canonical positive example과 3–5개의 canonical negative example이 있다. Exemplar는 criterion text가 감당하지 못하는 무게를 짊어진다. rubric author가 예상하지 못한 edge case와 만났을 때 rubric이 살아남는 방법이 바로 exemplar다. 모든 production annotation op는 프로젝트가 진행되며 커지는 살아 있는 "exemplar bank"를 유지한다. Expert adjudicator에게 escalated되어 written rationale과 함께 해결된 item은 해당 criterion의 새 exemplar로 bank에 추가된다. 여섯 달이 지나면 bank가 rubric prose보다 더 많은 일을 한다.

---

## 2. 평가자 간 일치도 — Cohen κ, Krippendorff α, 그리고 무엇을 측정하는가

Rubric-design quality를 숫자로 바꾸는 metric은 **Cohen's kappa**다([[hh-rlhf]]가 70–75% raw agreement를 보고하는데, chance correction 후에는 κ ≈ 0.4–0.5다). 공식은 다음과 같다.

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

여기서 `p_o`는 두 annotator 사이의 observed agreement rate이고, `p_e`는 각 annotator의 marginal label distribution이 주어졌을 때 우연히 기대되는 agreement다. chance correction은 중요하다. 두 annotator가 모두 item의 90%를 "chosen = A"로 표시하면, 순수 random만으로도 81%의 시간에 agree한다. κ는 이를 할인한다. κ=1.0은 perfect agreement, 0은 chance-level, 음수는 systematic disagreement다.

| κ range | Interpretation (Landis-Koch) | What to do |
|---|---|---|
| < 0.0 | chance보다 나쁨 | annotator instruction 어딘가가 뒤집혔다. rubric을 함께 다시 읽어라 |
| 0.0 – 0.2 | slight | rubric에 criteria가 빠졌다. ship하지 말라 |
| 0.2 – 0.4 | fair | 더 calibrate하라. underspecified edge case일 가능성이 높다 |
| 0.4 – 0.6 | moderate | noisy task(open-ended preference)에는 허용 가능. double-annotate + adjudicate를 얹어라 |
| 0.6 – 0.8 | substantial | 대부분 rubric에서 production-ready |
| 0.8 – 1.0 | almost perfect | verifiable task([[prm800k]] final-answer correctness)에 전형적 |

**κ는 두 annotator, categorical을 위한 metric이다.** 세 명 이상의 annotator가 있으면 pairwise κ를 계산해 평균내게 되어 joint structure를 잃는다. label이 ordinal일 때도 약해진다. 5-point Likert에서 "4 vs 5" disagreement는 "1 vs 5"와 같지 않지만, κ는 둘 다 disagreement로 처리한다.

그 경우에는 **Krippendorff's α**를 쓰라. α는 κ를 임의 수의 annotator, 임의 수의 item, missing data, 그리고 임의의 distance function(nominal, ordinal, interval, ratio)으로 일반화한다. [[ultrafeedback-construction]]의 4-aspect 0–10 rating은 암묵적으로 κ가 아니라 ordinal α를 요구한다. "7 vs 8" disagreement는 "2 vs 8"보다 덜 세야 한다. α는 distance function이 이를 인코딩하게 한다. κ는 그렇지 않다.

실무 규칙: **binary / small-nominal에는 κ, ordinal 또는 multi-annotator에는 α.** raw agreement와 chance-corrected statistic을 모두 보고하라. raw agreement만 보고하는 paper는 marginal-distribution skew를 숨기는 것이다.

**Worked κ example.** 두 annotator가 100개의 preference pair를 `chosen=A` 또는 `chosen=B`로 label한다. 둘 다 60개 item을 `A`, 40개 item을 `B`로 표시한다(symmetric marginals). 78개 item에서 agree한다(`p_o = 0.78`). Chance agreement는 `p_e = 0.6 × 0.6 + 0.4 × 0.4 = 0.52`. 따라서 `κ = (0.78 − 0.52) / (1 − 0.52) = 0.26 / 0.48 ≈ 0.54`다. Raw 78%는 안심돼 보이지만, κ=0.54는 rubric을 Landis-Koch "moderate"에 단단히 놓는다. double-annotation + adjudication을 얹을 때만 허용 가능하다. raw agreement가 70%를 넘더라도 κ < 0.4를 만드는 rubric은 ship하면 안 된다.

**Calibration sessions.** [[rlhf-instructgpt]]는 모든 labeler가 같은 100-item gold set을 rating하고, discrepancy를 토론하고, κ가 안정될 때까지 rubric을 수정하는 labeler onboarding을 실행했다. 이것이 명확해 보이지만 실제 예제를 만나기 전에는 모호한 rubric clause를 잡는 유일한 메커니즘이다. cadence: 초기 2주 calibration, held-out 50-item drift set에 대한 monthly re-calibration.

**Drift detection.** [[hh-rlhf]]의 crowdworker pool은 수개월에 걸쳐 변했다. 2022년 1월에 "helpful"이 뜻하던 것이 7월에는 약간 달라졌다. 방어는 약 50개 item의 고정 "anchor set"과 canonical label을 유지하고, 매월 모든 annotator를 anchor에 대해 실행하고, 개인의 κ-against-anchor가 >0.1 떨어지면 alarm하는 것이다. 이는 [[ch-14]] contamination detection의 canary set과 같은 discipline이다. distributional movement를 감지하는 고정 probe다.

---

## 3. 판정 워크플로 — single, double, triple, expert tier

잘 설계된 rubric 아래서도 human annotation에는 20–30% disagreement rate가 있다. 그렇다면 각 item에 annotator를 몇 명 배치해야 하는가? cost/quality curve는 잘 알려져 있다.

| Tier | Annotators per item | Cost multiplier | Quality floor | When to use |
|---|---|---|---|---|
| **Single** | 1 | 1× | raw rater quality | non-critical bulk labeling; noise가 scale에 흡수되는 SFT target |
| **Double** | 2, agree → keep, disagree → discard 또는 adjudicate | 2.2–2.5× | agreement filter가 약 25% 제거 | preference data, RM training, 대부분 production |
| **Triple** | 3, majority vote | 3× | majority vote가 single-rater slip 보정 | safety-critical pair, borderline κ rubric |
| **Expert adjudicator** | 1 domain expert가 disagreement 해결 | +0.3–0.8× marginal | expert 자신의 κ-with-self에 의해 bounded | medical / legal / frontier-math |

[[rlhf-instructgpt]]의 labeler pool이 첫 tier였고, OpenAI team 자체가 borderline case의 expert adjudicator 역할을 했다. [[openassistant]]의 tree structure는 구조상 사실상 multi-annotator다. 여러 labeler가 각 message를 rating하고 ranking이 aggregated된다. [[prm800k]]는 "명백히 correct / 명백히 wrong"인 80%에는 single annotation을 쓰고, uncertain step 20%를 expert tiebreaker가 있는 second annotator에게 route한다.

실제로 ship되는 운영 패턴은 이렇다. **모든 item을 double-annotate하고, κ를 지속적으로 측정하며, 두 annotator가 disagree하는 item만 expert tier로 올린다.** 75% agreement에서는 item의 25%가 escalated된다. Expert time이 crowdworker time보다 3–10배 비싸기 때문에 이것이 budget을 지배한다. [[tulu-3-sft-mix]]의 939K-prompt budget이 tractable한 것은 Ai2가 가능할 때마다 deterministic verifier(exact-match, code-execution)를 사용했고, verifier가 없는 chat / safety / precise-IF에 human tier를 남겼기 때문이다.

**Cost math.** Crowdworker time이 $0.50/item, expert time이 $5/item, κ = 0.6(25% disagreement)이고 모든 것을 double-annotate한다면:

```
cost_per_item = 2 × $0.50 + 0.25 × $5.00 = $1.00 + $1.25 = $2.25
```

Expert tier가 crowdworker budget보다 *더 크다*. κ가 0.1 증가하면(0.7 = 15% escalation) expert line이 $0.50 줄어 총 비용의 22%를 낮춘다. 이것이 rubric을 계속 반복 개선해야 하는 단단한 재무적 인센티브다. rubric 작업의 모든 시간이 adjudication load 감소라는 몇 주치 보상으로 돌아온다.

**Triple-annotation alternative.** Triple-annotate하고 majority vote를 쓰면, adjudication이 필요하려면 세 annotator가 모두 disagree해야 한다(balanced marginal 아래 대략 `((1−κ)/2)²`). κ=0.6이면 double-annotate-with-tiebreak의 25% 대비 약 1.5% escalation이다. Base cost는 오른다(2× crowdworker 대신 3×)지만 expert line은 붕괴한다. Safety-critical data, 예컨대 [[prosocial-dialog]]의 rule-of-thumb label이나 [[hh-rlhf]]의 harmless red-team preference에는 triple이 보통 맞는 tradeoff다. Bulk helpfulness에는 double + expert가 비슷한 품질에서 더 싸다.

---

## 4. Preference 샘플링 정책 — 어떤 pair를 보여줄 것인가

Preference-data quality에서 가장 큰 레버는 annotator가 *어떤* response pair를 보는가다. 거의 같은 두 response를 보여주면 annotator의 판단을 낭비한다. 명백한 winner와 명백한 loser를 보여주면 coarse axis만 구분하는 RM을 훈련하게 된다. 작동하는 policy는 decision tree다.

```
                        [ response pair for candidate prompt p ]
                                        |
                      ┌─────────────────┴──────────────────┐
                      |                                    |
          Is there a verifier (exec/                       no
          exact-match/IF-checker)?                         |
                      |                                    |
                     yes                                   |
                      |                             ┌──────┴──────┐
        Use verifier signal, skip                   |             |
        human (RLVR path — tulu-3)               SFT target?   preference pair?
                                                    |             |
                                       Show single response;   Compute pair-delta
                                       annotator edits for     (aspect score gap
                                       quality (OASST style)   or RM prediction
                                                               confidence)
                                                                     |
                                                  ┌──────────────────┼──────────────────┐
                                                  |                  |                  |
                                            δ very large         δ moderate          δ tiny
                                          (obvious winner)      (informative)     (close pair)
                                                  |                  |                  |
                                           drop — wastes         **show human**    active-learning
                                           budget                                  goldmine: show
                                                                                   to expert tier
```

세 가지 기법이 δ bucket을 채운다.

**Aspect gap을 통한 close-pair mining.** [[ultrafeedback-construction]]은 이질적인 17-model fleet에서 prompt당 4개 response를 생성하고, 각 response를 4개 aspect(instruction-following, truthfulness, honesty, helpfulness)로 rating한다. *Aspect A에서는 gap이 크지만 aspect B에서는 gap이 작은* `(response_i, response_j)` pair는 annotator attention을 A에 고립시킨다. annotator의 vote는 "나는 sycophantic보다 truthful을 선호한다"는 깨끗한 신호가 되고, noise-averaged preference가 아니다. Per-aspect gap width에 비례해 sampling하면 단일 item의 label이 RM을 가장 크게 움직이는 곳에 budget을 집중한다.

**RM disagreement에 의한 active learning.** 처음 10K item 뒤 proto-RM을 훈련한다. 다음 10K에는 큰 candidate pool을 score하고 RM prediction이 0.5에 가까운(max uncertainty) item을 우선한다. [[prm800k]]는 이 단일 기법에서 약 2.6배 data-efficiency multiplier를 보고한다. active learning을 쓴 800K step label이 쓰지 않은 2.1M label과 맞먹었다. 이것이 이 장에서 ROI가 가장 큰 knob다.

**On-policy pair는 margin에서 off-policy pair를 지배한다.** [[tulu-3]]의 DPO data는 "SFT model의 on-policy sampling + reward model ranking으로 curate"된다. 이유는 이렇다. `(GPT-4 response, Llama-7B response)` 같은 off-policy pair는 RM이 좋은/나쁜 *동작*을 구분하도록 훈련하는 것이 아니라 model *family*를 구분하도록 훈련한다. 당신의 policy가 어느 response도 만들어 본 적이 없다면, 그 pair에서 RM이 일반화하는 힘은 약하다. Llama 3의 six-round RSFT는 이 원칙의 산업적 표현이다. 각 round의 preference는 *현재* policy의 output 위에 있다.

**Tree를 합치기.** Production preference-sampling loop는 세 가지를 모두 연결한다. (1) prompt당 on-policy response 4개를 생성한다. (2) cheap judge 또는 RM이 각 response를 rubric aspect별로 score한다. (3) pairwise aspect-delta를 계산한다. (4) "obvious"도 "indistinguishable"도 아닌 *median* delta band에서 pair를 sampling해 annotator review에 보낸다. [[ultrafeedback-construction]] pipeline은 이 방식의 static off-policy 버전이다. [[tulu-3]]의 on-policy DPO data는 dynamic 버전이다. Budget savings는 곱해진다. close-pair mining만으로 약 2배, on-policy vs off-policy로 또 약 1.5배, RM uncertainty에 의한 active learning이 약 2.6배([[prm800k]])다. 순진하게 쌓으면 동일한 RM 품질을 한 자릿수 배 더 싸게 얻는다. 그래서 2024–2026년 post-training recipe가 모두 이들을 연결한다.

---

## 5. 인간 데이터가 synthetic / judge signal을 override할 때

[[judge-llm-bias]]의 80% agreement ceiling은 평균값일 뿐이다. 특정 query class에서는 judge와 human의 gap이 judge signal이 없는 것보다 나빠질 정도로 벌어진다.

| Query class | Best label source | Why human beats judge |
|---|---|---|
| Safety-critical refusal / red-team | **human expert** | [[prosocial-dialog]]의 RoT layer는 judge가 갖지 못한 문화적·맥락적 지식을 요구한다. [[hh-rlhf]] red-team labeling은 judge 자체가 만들어내는 harm을 잡는다 |
| Frontier reasoning (IMO, Putnam-level math) | **domain expert** 또는 **verifier** | judge는 유창하게 제시된 wrong proof를 자신 있게 확인한다([[judge-llm-bias]]의 "limited reasoning in pair judging") |
| Medical / legal domain | **licensed expert** | generic rubric은 liability-grade precision을 encode하지 않는다. judge는 credentialing이 없다 |
| Ordinary helpfulness / chat fluency | **judge + light human audit** | 80% judge-human agreement는 75% human-human agreement와 구분되지 않는다. scale이 이긴다 |
| Code correctness / unit-testable | **verifier (exec)** | deterministic. [[tulu-3]]의 RLVR path는 RM을 완전히 버렸다 |
| Format / instruction compliance | **verifier (regex / checker)** | IFEval-style constraint는 mechanical하다 |

결정 규칙은 이렇다. **Rubric이 under-specified이고 verifier가 없을 때만 인간을 사용하라.** 그 외에는 대체하라. [[tulu-3-sft-mix]]의 43% in-house synthetic data가 GPT-4o / Claude 3.5 Sonnet으로 생성되는 이유는 바로 이 도메인(persona math, persona IF, code)에 verifier 또는 near-verifier가 있기 때문이다. 57% public-sources 절반에는 judge signal이 약하다고 알려진 진짜 human OpenAssistant + WildChat 부분이 포함된다.

---

## 6. 운영 현실 — onboarding, leak prevention, burnout, audit trail

논문이 거의 드러내지 않는 logistics 문제들의 목록을 모든 ship된 annotation operation은 해결한다. 위의 모든 것이 이를 가정하기 때문에 이것이 마지막 section이다.

**Onboarding.** 새 annotator는 rubric을 읽고, 100-item calibration set을 label하고, lead와 discrepancy를 토론하고, rubric을 다시 읽고, 50개를 더 label한다. κ-against-lead가 ≥0.6이면 production에 투입된다. 아니면 retry 또는 reject다. [[rlhf-instructgpt]]의 labeler pool은 이 방식으로 vetted됐다. Surge AI / Scale AI workflow는 제도화된 버전이다.

**Prompt-leak prevention.** Annotator는 MMLU, GSM8K, IFEval 같은 eval-set prompt를 자주 다룬다. 두 가지 방어가 있다. (1) NDA + no-take-home work. annotation은 copy-paste export가 없는 sandboxed web interface 안에서 일어난다. (2) pre-annotation decontamination. [[allenai-tulu-sft-recipe]]는 prompt가 annotator pool에 닿기 *전*에 모든 eval set에 대해 8-gram overlap ≥50%를 실행한다. 그래서 annotator가 prompt를 외워도, test set을 본 적 없는 eval을 contaminate할 수 없다. 두 방어는 합성된다. 어느 하나만으로는 충분하지 않다.

**Burnout and quality decay.** Annotator κ는 몇 시간짜리 session을 거치며 떨어진다. Production rule은 개인 session을 약 2시간으로 제한하고, break를 강제하고, monotony를 깨기 위해 rubric을 rotation(helpfulness → code → safety)하는 것이다. Anchor set에서 per-annotator κ를 weekly monitor하라. 개인 κ가 2주 연속 >0.1 떨어지면 re-calibrate하거나 프로젝트에서 rotation한다. [[openassistant]]의 volunteer pool에는 이런 control이 없었고, 품질 variance는 post-hoc filter rejection rate에서 드러난다.

**Audit trail.** 모든 label은 `(annotator_id, timestamp, item_id, label, time_spent_seconds, rubric_version_hash)`를 기록한다. 비자명한 것은 `rubric_version_hash`다. Rubric이 수정되면(프로젝트에서 5–20번 일어난다) v0.3 아래에서 만든 label은 v0.4 아래에서 만든 label과 inconsistent할 수 있다. Audit trail은 downstream eval이 regress했을 때 v0.3 slice를 post-hoc re-label하거나 down-weight할 수 있게 한다. [[tulu-3-sft-mix]]의 "MMLU, GSM8K, MATH, IFEval에 대한 explicit decontamination..."은 provenance chain이 살아 있기 때문에만 auditable하다.

**Reproducibility and public release.** [[openassistant]]는 161K messages를 CC-BY 4.0으로 공개한다. [[hh-rlhf]]는 161K dialogues를 MIT로 공개한다. [[prm800k]]는 약 800K step label을 공개한다. 이런 release는 label뿐만 아니라 rubric version, annotator demographics(aggregated), known gaps도 노출한다. *"crowdworker demographics skew the helpful signal"*([[hh-rlhf]]), *"contributors skew Western/English/technical"*([[openassistant]]). 그런 disclosure가 없는 dataset card는 reproducible artifact가 아니다. 공개되어 있을 뿐인 black box다. label 자체가 proprietary더라도 rubric과 agreement statistic은 함께 공개할 계획을 세워라. 커뮤니티는 데이터가 없어도 noise floor에 대해서는 추론할 수 있다.

---

## 7. 직접 만들 때, 구매할 때, 생략할 때

모든 팀이 project kickoff에서 내리는 세 갈래 결정은 다음과 같다.

| Path | Cost | Control | When it's right |
|---|---|---|---|
| **Build in-house annotator pool** | 가장 높음(hiring, onboarding, QA) | 전체 | safety-critical, domain-specialist, proprietary rubric |
| **Buy from an annotation vendor (Surge, Scale, Invisible)** | 중간 | rubric 공유, worker rotation | 잘 이해된 rubric의 bulk preference data |
| **Crowdsource volunteer (OASST-style)** | 낮은 $ / 높은 ops | tree structure가 noise를 견딤 | public-good dataset; 약 50% post-hoc rejection을 감수할 의향 |
| **Skip — use judge or verifier** | 가장 낮음 | 80% judge-human ceiling ([[judge-llm-bias]]) | verifiable task; judge ≈ human인 domain |

현대적 방식은 네 가지를 모두 결합한다. [[tulu-3]]의 939K SFT mix는 skill에 따라 public-crowdsourced(OpenAssistant, WildChat) + vendor-annotated + synthetic-with-judge + verifier-graded(RLVR)를 사용한다. Rubric은 어떤 tier가 어떤 slice를 받을지 결정한다. Rubric이 judge에 충분히 정밀하게 작성된 것은 judge가 맡는다. Rubric이 너무 문화적으로 loaded된 것(safety, RoT, underspecified context의 helpfulness)은 human이 맡는다. 결정은 project별이 아니라 slice별이다.

---

## 연결과 다음 단계

- **[[rlhf-instructgpt]] / this chapter** — foundational three-stage recipe. 여기의 모든 운영 선택은 InstructGPT의 어떤 선택에서 내려온다.
- **[[hh-rlhf]] / [[prosocial-dialog]]** — two-axis(helpful × harmless)와 rule-of-thumb(RoT) rubric pattern. safety sub-literature.
- **[[openassistant]]** — open-crowdsourced baseline. vetting 없는 인간이 무엇을 만들어내는가.
- **[[ultrafeedback-construction]] / [[judge-llm-bias]]** — ledger의 synthetic-preference 절반. judge가 volume의 80%에서는 작동하고 critical 20%에서 실패하는 이유.
- **[[prm800k]]** — process-level labeling as the extreme case of rubric granularity.
- **[[tulu-3-sft-mix]] / [[allenai-tulu-sft-recipe]] / [[tulu-3]]** — 939K scale의 skill-targeted annotation을 위한 2024–2025 reference stack.
- **ch-16 (RL Prompt Curation)** — 이 label들이 RL-stage prompt pool에서 어떻게 쓰이는가. difficulty mining은 §4의 close-pair-mining을 prompt dimension으로 밀어 넣은 것이다.
- **ch-17 (lab)** — small-scale filter pipeline. annotation-budget planning은 memo 중 하나다.

## 동반 시각화

**[figures/annotation-workflow.html](figures/annotation-workflow.html)** — interactive three-tier pipeline(annotator → reviewer → adjudicator). inter-rater κ threshold slider를 조정하면 escalation rate와 total cost가 실시간 업데이트된다. double-vs-triple annotation을 toggle하면 quality floor가 어떻게 움직이는지 볼 수 있다. 이 tool은 §2의 κ 공식과 §3의 cost math를 손에 잡히게 만든다. κ를 0.5에서 0.7로 올리면 expert tier가 약 50% 붕괴하며, 바로 거기에 rubric-iteration ROI가 있다.

실행할 세 가지 구체적 scenario: (1) κ=0.55, 100k items, double+expert로 설정하고 expert share가 budget의 ≥50%인지 확인한다. (2) κ=0.70으로 올리고 per-label cost가 약 30% 떨어지는 것을 본다. (3) κ=0.55에서 triple-annotation으로 바꾸고 비교한다. majority vote가 adjudicator tier에 닿기 전에 single-rater noise를 흡수하기 때문에 tail에서는 triple이 더 싸다.

이 세 dial이 모든 annotation program이 항해하는 cost-quality frontier 전체다. 왼쪽 edge(low κ, single-annotate)에서 시작하고, rubric iteration마다 전체 frontier가 down-and-right로 이동한다. 더 낮은 cost *그리고* 더 높은 quality다. 더 나은 rubric은 주어진 budget level에서 disagreement도 escalation도 줄이기 때문이다.
