<!-- chapter: ch-49
     track: eval
     kind: content
     title: Judge Models and Judge Calibration
     deps: [ch-48]
     sources: [[judge-llm-bias]], [[generative-reward-models]], [[direct-judgement-preference]], [[rlaif-scaling]], [[ultrafeedback-construction]], [[pairrm]], [[reward-hacking-taxonomy]], [[self-rewarding-lm]], [[meta-rewarding-lm]], [[faithful-synth-eval]], [[wildchat]]
     figures: figures/judge-bias.html
-->

# 49장 — Judge 모델과 Judge 보정

> **핵심 통찰.** LLM judge는 자가 아니다. *구조화된* 편향을 가진 잡음 많은 계측기다. 여기에는 위치, 장황함, 자기 강화, 형식, 그리고 반복 중 자기 재강화가 포함된다. Zheng 2023의 대표 주장("GPT-4는 인간과 약 80% 일치하며, 인간-인간 일치율과 비슷하다")은 집계 수준에서는 참이지만 go/no-go 결정에 중요한 모든 축에서는 오해를 부른다. 보정된 judge 프로토콜은 측정 스택이다. swap 감사를 거치고, 길이를 통제하며, rubric에 고정되고, 테스트 대상 policy와 다른 model family를 써야 한다. 2024-25년 synthetic-judge 계열(Con-J, Self-Taught Evaluators, J1)은 GPT-4-as-judge를 대체한다. 그 이유는 이 judge들이 더 싸서만은 아니다. 물론 더 싸다. 하지만 *소유한* judge만이 RL 시점의 reward와 eval 시점의 측정 사이 누출을 경계 지을 수 있기 때문이다.
>
> **가이드라인.** judge를 신뢰하기 전에 보정해야 하는 계측기로 다뤄라. 모든 eval run에서: (1) pair마다 위치를 무작위화하고 swap-consistency를 1급 숫자로 보고한다. (2) length-controlled baseline과 짝지어 length-residualized win-rate를 보고한다. (3) candidate model을 자기 자신의 judge로 절대 쓰지 않는다. (4) judge drift를 감지하기 위해 분기마다 held-out human-label set에 anchor한다. (5) 하나를 해킹해도 자기 자신에 점수를 주지 못하도록 RL-time RM과 eval-time judge를 *서로 다른 model family*에 둔다. 이것들이 지켜질 때 judge는 eval primitive다. 하나라도 조용히 실패하면, 당신은 원을 측정하고 있는 것이다.

---

## 1. 왜 judge가 필요한가

Instruction-following, creative writing, open-ended helpfulness에는 ground truth가 없다. 수학 답은 SymPy로 확인할 수 있다. 하지만 "조문 이메일 초안을 작성해줘"가 성공했는지는 그렇게 확인할 수 없다. 2023년 이후 이 분야의 답은 **LLM-as-judge**였다. 강한 LM이 `(prompt, response_A, response_B)`를 읽고 판정을 내리며, 그 판정을 preference label로 취급한다. 그 판정은 세 가지 서로 다른 pipeline의 원재료가 된다.

- **Eval:** open-ended benchmark(MT-Bench, Arena-Hard, AlpacaEval, WildBench)에서 checkpoint 순위를 매긴다.
- **Training data:** DPO/RPO용 preference pair를 만든다([[ultrafeedback-construction]], [[rlaif-scaling]]).
- **Runtime reward:** scalar RM을 쓸 수 없을 때 PPO/GRPO의 rollout에 점수를 준다([[self-rewarding-lm]]).

이 장은 *eval* 용도를 다룬다. ch-42와 ch-44는 training-time 용도를 다뤘다. 이 구분은 중요하다. outline이 말하듯, **"주제가 training을 위한 signal 생성이라면 RL에 속하고, checkpoint 감사나 비교라면 Eval에 속한다"**(outline `rl_vs_eval_boundary` 참고). RL에 충분히 잘 보정된 judge도 자신이 채점하는 benchmark를 누출할 수 있다.

더 나아가기 전에 정해야 할 질문이 하나 있다. *왜 같은 LM이 evaluator로도 작동할 수 있는가?* [[rlaif-scaling]]의 "same-size labeler" ablation에서 나온 답은 preference labeling이 generation보다 엄격히 더 쉽다는 것이다. labeler가 policy와 같은 base LM일 때도 RLAIF는 SFT보다 개선된다. 즉 labeler는 generator가 스스로 활용하지 못하는 discriminative signal을 추출하고 있다. 이 비대칭성이 LLM-as-judge를 가능하게 한다. 동시에 judge의 *bias*가 무작위가 아니라 체계적인 이유이기도 하다. bias는 같은 family를 ensemble한다고 상쇄할 수 있는 noise가 아니라 labeler의 training distribution을 반영한다.

---

## 2. 세 가지 공개 judge-driven benchmark

eval community는 2023년부터 2025년 사이에 세 층의 judge-driven benchmark로 수렴했다. 각각은 앞선 benchmark의 서로 다른 실패를 겨냥한다.

| Benchmark | Prompt source | Pairing | Judge | Typical metric | Known failure mode |
|---|---|---|---|---|---|
| **MT-Bench** (Zheng 2023, [[judge-llm-bias]]) | 80 hand-written questions × 8 categories (writing, roleplay, reasoning, math, coding, extraction, STEM, humanities) | Single-answer scoring (1–10) + pairwise vs reference | GPT-4 + rubric | Mean category score | Small N, category collapse, leakage since 2024 |
| **Arena-Hard** (LMSys 2024) | 500 user prompts mined from Chatbot Arena, filtered for high-discriminator difficulty | Pairwise vs baseline (GPT-4-0314) | GPT-4-Turbo + length-control regression | Length-controlled win-rate (LC-WR) | Judge-family self-enhancement (Arena-Hard built on GPT-4 judge) |
| **WildBench** (AI2 2024) | Real user queries from [[wildchat]] opt-in logs, difficulty-tagged | Pairwise vs three reference models (GPT-4, Claude, Llama-70B) | GPT-4-Turbo with domain-specific rubric | WB-Score (1–10 scaled by difficulty) | Judge reads its own prior outputs; rubric hand-tuning is labor-intensive |

세 benchmark 모두 출시 당시 GPT family judge에 의존했다. 이것이 synthetic-judge 계열이 고치려는 공통 single point of failure다(§5).

---

## 3. Bias inventory — Zheng 2023 canon

기초 결과는 [[judge-llm-bias]]에서 그대로 인용할 수 있다.

> "GPT-4 vs human-expert agreement is 85%+ on MT-Bench and ~80% on Chatbot Arena; the same rate two humans agree with each other… A vs B ordering changes the winner in ~20–30% of cases; mitigated by swap-and-average or 'two-game' scoring… longer responses win more often than a length-controlled baseline… GPT-4 prefers GPT-4-authored responses at a rate above what humans prefer; Claude shows the same toward Claude."

네 가지 bias가 있으며, 각각에는 구체적인 측정 방법과 구체적인 보정이 있다. 이 표를 외워라.

| Bias | Measurement method | Signal | Correction |
|---|---|---|---|
| **Position (order)** | 각 pair를 `(A,B)`와 `(B,A)` 두 순서로 실행하고 불일치를 센다 | Swap-flip rate: GPT-4 ≈ 22%, GPT-3.5 ≈ 40% ([[judge-llm-bias]] Fig. 2) | Two-game scoring: judge가 order-consistent일 때만 win으로 세고, 아니면 tie 선언 |
| **Verbosity (length)** | 길이만 다른 response pair를 비교한다(paraphrased long vs short, same content) | token 단위 length_delta 대비 win-rate 기울기([[judge-llm-bias]] Fig. 4) | Length-residualized win-rate(Arena-Hard LC) 또는 명시적 length-penalty rubric term([[meta-rewarding-lm]]) |
| **Self-enhancement** | Judge × candidate self-preference matrix; 같은 pair에서 judge-rate와 human-rate 비교 | [[judge-llm-bias]] Fig. 5의 diagonal excess | candidate model을 자기 judge로 절대 쓰지 않는다. 여러 judge family를 pooling한다 |
| **Format (markdown / bold)** | markdown header나 bold만 다른 A/B pair | Judge는 formatted 쪽을 선호하는 경향([[reward-hacking-taxonomy]]에 열거된 mode) | judging 전에 formatting을 제거하거나 rubric에 "formatting을 reward하지 말라" 포함 |
| **Confidence miscalibration** | judge log-prob margin별로 pair를 binning하고 bin별 empirical accuracy plot | close pair에서 overconfidence([[generative-reward-models]] Fig. 4) | CoT rubric을 쓰는 Generative RM은 calibrated log-prob를 만든다. 더 좁은 bin에는 ensemble 사용 |
| **Judge-drift under iteration** | 고정 human-label set을 hold out하고 round별 judge Spearman correlation 추적 | [[self-rewarding-lm]] Table 2: 3회 iter 동안 0.62 → 0.71, 이후 drift | judge weight를 pin한다. 분기마다 human set에 re-anchor |

보정을 이해하는 세 가지 상보적 방식이 있다. (a) 측정을 **대칭화**한다(swap, length-balanced, cross-family). (b) **rubric을 고정**한다(CoT instruction, format/length 금지의 명시). (c) 외부 ground truth에 **anchor**한다(reference answer, human-label spot check, verifier-checkable subset).

작동 예시를 보자. [[judge-llm-bias]]는 reference-guided grading, 즉 judge가 candidate를 읽기 전에 prompt에 gold solution을 붙이는 방식이 MT-Bench에서 agreement를 "약 10 pp" 높인다고 보고한다. 이것은 대칭화가 아니라 anchoring이다. math/coding pair에서는 "LLM judges can confirm a wrong answer if it is presented confidently"(source quote). reference answer는 confident-wrong-answer 실패를 끊지만, reference가 존재하는 task에서만 가능하다. writing과 roleplay는 여기서 얻는 것이 없다. 어떤 보정을 적용할지는 global이 아니라 category별이다.

**[figures/judge-bias.html](figures/judge-bias.html)**를 보라. position/verbosity/self-enhancement를 고르고, naive judge 대비 corrected judge accuracy delta를 확인하며, judge confidence별 calibration curve를 scrub할 수 있다.

---

## 4. 보정된 judge는 실제로 어떻게 실행되는가

inference 시점에 judge는 세 artifact로 정의된다.

1. **Rubric.** prompt가 무엇을 말하든 judge는 그것을 optimize한다. [[ultrafeedback-construction]]은 네 축(instruction-following, truthfulness, honesty, helpfulness)을 쓰며 각 축은 0–10점과 짧은 rationale을 갖는다. [[judge-llm-bias]]는 pairwise-with-reference를 쓴다. [[meta-rewarding-lm]]은 "길이 자체를 reward하지 말라"를 명시적으로 추가한다. 이것이 곧 calibration이다. rubric text가 policy knob이다.

2. **Template.** CoT-before-verdict는 direct-verdict보다 [[rlaif-scaling]] Fig. 4에서 3–5 pp, [[generative-reward-models]] 기준 RewardBench에서 3–10 pp 더 좋다.

   > "Here is a query and two responses. Which response is better? Respond with 'Response 1' or 'Response 2'. Let's think step by step…"

   그런 다음 마지막에 token 하나가 온다. answer token의 log-prob가 [[rlaif-scaling]]의 **soft label**이다. `p = softmax(logits["Response 1"], logits["Response 2"])`. soft label은 BT target이나 calibration signal로 쓸 때 hard A/B보다 성능이 좋다.

3. **Swap protocol.** [[pairrm]]은 swap-augmentation을 표준으로 만들었다. `f(x, y_A, y_B)`와 `f(x, y_B, y_A)`를 실행하고, logit을 평균한 뒤 consistent winner를 택한다. [[judge-llm-bias]]는 이것이 position bias를 약 2 pp residual 이내로 상쇄한다고 정량화한다. 현대 eval은 이 단계를 건너뛰지 않는다.

구체적인 pairwise prompt template([[rlaif-scaling]] §Technical Details에서 인용 및 paraphrase):

```
사용자 질의와 Response 1, Response 2로 표시된 두 후보 응답을 보게 됩니다.
Rubric: helpfulness, factual correctness, instruction-following. 길이 자체를 reward하지
마세요. markdown formatting 자체를 reward하지 마세요.
Query: {x}
Response 1: {y_A}
Response 2: {y_B}
답하기 전에 단계별로 생각하세요.
최종 답변(토큰 하나만): Response 1 | Response 2
```

soft label은 final-answer token의 logit에서 읽는다. `p_A = softmax(logit["Response 1"], logit["Response 2"])[0]`. 이 `p_A`가 calibrated judge score다. `argmax`는 이미 계산된 정보를 버린다. 같은 pair를 swapped order로 실행하면 두 `p_A` 값을 하나는 뒤집은 뒤 평균하여 swap-consistent calibrated score를 만든다.

이 중 하나라도 빠진 judge setup은 고장 난 계측기로 다뤄라. rubric, template, swap protocol을 명시하지 않고 MT-Bench 숫자 하나만 인용하는 report는 감사할 수 없다.

---

## 5. Synthetic-judge 계열 — 왜 GPT-4-as-judge가 대체되는가

2024–25년의 세 paper가 이 계열을 정의한다. 공통 움직임은 judge를 돈 주고 쓰는 대신 *train*하는 것이다. [[direct-judgement-preference]]는 이를 "UltraFeedback pattern collapsed into a self-contained generative judge"라고 부른다. 방법 대비는 다음과 같다.

| Method | Year | Training signal | Judge output | Key claim | Cost post-bootstrap |
|---|---|---|---|---|---|
| **GPT-4-as-judge** (Zheng 2023, [[judge-llm-bias]]) | 2023 | None — off-the-shelf API | Verdict + rationale | 85%+ human agreement on MT-Bench | API-priced, ~$0.01–0.03 per pair |
| **Con-J** (ICLR 2025, [[direct-judgement-preference]]) | 2024 | DPO on contrastive judgment pairs with noisy-negative instruction perturbation | Rationale + verdict | Robustness to format bias + label noise; interpretable rationale | Local inference, zero API |
| **Self-Taught Evaluators** (Meta 2024, [[direct-judgement-preference]]) | 2024 | Iterative: judge_k labels fresh pool → train judge_{k+1} by DPO on its own decisions vs alternatives | Rationale + verdict | RewardBench crosses GPT-4 baseline after ~3 iterations on zero human labels post-bootstrap | Local inference |
| **J1** (2025, [[direct-judgement-preference]]) | 2025 | RL on the judge's chain-of-thought (GRPO-style over judgment rollouts) | CoT + verdict | Highest-accuracy open judge on RewardBench-hard at publication | Local inference, CoT-heavy |
| **Self-Rewarding / Meta-Rewarding** ([[self-rewarding-lm]], [[meta-rewarding-lm]]) | 2024 | Policy acts as own judge; meta-judge evaluates judge | 5-point rubric + rationale | Judge Spearman with humans rises 0.62 → 0.71 (SR); AlpacaEval 22.9% → 39.4% (MR) | In-stack; RL use primary |
| **PairRM** ([[pairrm]]) | 2023 | BCE on joint-input pairs with swap-augmentation | Scalar logit | 0.4B param matches scalar-RM-7B; tournament O(N log N) | Cheapest per comparison (encoder-only) |

[[direct-judgement-preference]]에서 확인되는 Self-Taught Evaluator trajectory는 정면으로 볼 가치가 있다. Round 0은 아주 작은 labeled seed다. round k는 judge_k가 fresh pool에 label을 붙이고, 선택된 verdict와 의도적으로 만든 alternative verdict로 구성한 contrastive judgment pair에 대해 judge_{k+1}을 DPO-train한 뒤 반복한다. RewardBench accuracy는 대략 iteration 3에서 GPT-4-as-judge baseline을 넘고 몇 iteration 뒤 포화된다. Con-J의 noisy-negative trick, 즉 instruction을 perturb하고 noisy instruction에 대한 response를 생성한 뒤 그럴듯한 "rejected"로 취급하는 방식은 원래 preference label 없이도 contrastive pair를 만든다. 이 점이 end-to-end로 human/API data 의존성을 접는다.

J1은 같은 judge를 chain-of-thought에 대해 RL로 더 학습한 것이다. judgment rollout은 held-out reference verdict에 대해 scoring되고, GRPO-style optimization은 CoT를 더 높은 agreement reasoning 쪽으로 밀어낸다. 이 움직임은 math의 RLVR과 동일하다. verifiable outcome(판정이 reference와 일치함)이 있고, 그것을 만들어 내는 reasoning을 optimize한다. verdict-token log-prob가 verifier signal을 제공해 [[generative-reward-models]]가 연 loop를 닫기 때문에 작동한다.

GPT-4-as-judge가 대체되는 이유는 네 가지이며, 각각 근거가 있다.

**a) Eval-RL leakage.** GPT-4가 DPO data에 label을 붙이고([[ultrafeedback-construction]]) *또* GPT-4가 eval을 judge한다면, model은 train time에 eval time에서 자신을 채점하는 동일 bias를 맞추도록 reward된다. [[direct-judgement-preference]]는 이를 "judge-as-weapon"이라고 부른다. "the same judge model used to label prefs and evaluate benchmarks creates leakage — RewardBench's increasing close relationship to training-time judges is a known measurement issue." Eval-time judge는 RL-time RM과 달라야 한다.

**b) Reproducibility.** GPT-4의 weight는 API version 사이에서 조용히 drift한다. 2023년의 MT-Bench score와 2025년의 "GPT-4-as-judge" score는 같은 측정이 아니다. 소유한 judge는 계측기를 freeze한다.

**c) Cost.** [[ultrafeedback-construction]]은 약 1M개의 GPT-4 annotation에 "tens of thousands USD"가 들었다고 기록했다. [[direct-judgement-preference]]의 Self-Taught Evaluator는 "~40K synthetic preference pairs (20K SFT + 20K DPO) suffice to beat models trained with 2–40× more data"를 보인다. marginal API spend는 0이다.

**d) Ecosystem 수준의 self-enhancement bias.** GPT-4가 GPT-4 descendant를 non-GPT-4 model과 비교해 judge할 때, [[judge-llm-bias]] Fig. 5의 diagonal은 시스템적 오염원이 된다. Cross-family judge(Claude-judge-of-GPT, Llama-judge-of-Qwen)가 현재의 완화책이다.

---

## 6. Judge-RM vs generative-RM — eval-time judge ≠ RL-time RM인 경우

이것이 이 장에서 가장 깊은 미묘함이다. 독자는 이 장을 마칠 때 mature stack에서 왜 *두 개의 서로 다른 judge*가 필요한지 묻지 않아도 설명할 수 있어야 한다.

- **RL-time RM:** 빠르다(GRPO inner loop에서 per-token scoring), differentiable-ish하다(scalar 또는 log-odds), 모든 gradient step에서 쓰인다. [[pairrm]]식 joint-input scoring이나 short-rubric CoT를 쓰는 [[generative-reward-models]]가 맞다. optimizer가 이것을 *hack*할 것이다. 이는 예상된 일이다. KL-to-reference([[reward-hacking-taxonomy]])가 얼마나 멀리 갈 수 있는지 제한한다.

- **Eval-time judge:** 느리다(checkpoint × benchmark마다 한 번), auditable하며, rubric이 풍부하다. full critique + verdict를 쓰는 [[generative-reward-models]]나 외부 model family의 human-anchored Self-Taught judge가 맞다. go/no-go에 쓰인다. 이 judge가 RL을 scoring한 model과 같다면 모든 reward hack이 보이지 않는다.

[[generative-reward-models]]는 핵심 항등식을 준다. reward = `log P_RM("A is better" | x, y_A, y_B, rubric)` — soft log-odds다. 이는 [[rlaif-scaling]]의 d-RLAIF reward(`r = log P_labeler("Better")`)와 *구조적으로* 같은 형태다. 따라서 "generative RM"과 "LLM-as-judge"는 서로 다른 *수학*이 아니다. 같은 signal을 어떻게 *사용*하느냐만 다르다. Judge-RM은 training-time이고, generative-RM은 둘 모두를 포괄하는 construction family다. mature stack은 같은 construction을 두 phase 모두에서 실행하되 *서로 다른 model*, 서로 다른 rubric, 그리고 eval-time judge를 학습시킨 적 없는 held-out data를 사용한다.

구체 규칙: "train에서는 어떤 model family가 judge하는가? eval에서는 어떤 family인가?"를 한 문장으로 답할 수 없다면 eval은 leaky하다.

2025년식 예시를 보자. 한 팀이 [[ultrafeedback-construction]] prefs(GPT-4 judge at train time)로 DPO를 수행해 7B policy를 학습한 뒤 Arena-Hard(GPT-4-Turbo judge at eval time)로 checkpoint를 scoring한다. 공유 family는 GPT-4다. GPT-4가 가진 length/self-enhancement bias는 train time에 reward되고 eval time에 quality로 계산된다. 해결책은 "다른 benchmark를 고른다"가 아니다. Claude나 Llama-judge pass로 parallel eval을 실행하고 두 숫자를 모두 보고하는 것이다. 두 pass 사이의 disagreement는 1급 calibration signal이다. Claude-judge는 +4 pp를 보이고 GPT-4-judge는 +12 pp를 보인다면, 그 12점 중 8점은 leakage다.

---

## 7. Calibration curve와 confidence

judge의 verdict에는 confidence가 있다. verdict 위치에서 "A"와 "B" 사이의 log-prob margin이다. [[generative-reward-models]] Fig. 4: GenRM은 "BT RM이 overconfident한 곳에서 well-calibrated"다. calibration의 의미는 다음과 같다. judge가 보고한 margin(예: `|log P(A) − log P(B)|`의 decile)으로 pair를 binning하고, bin별 empirical human-agreement를 plot한다. judge가 calibrated면 calibration curve는 `y = x`이고, overconfident하면 `y < x`다.

calibration의 실용적 용도는 세 가지다.

- **Tie declaration.** margin이 작은 bin → judge가 tie를 선언한다. Elo update는 [[judge-llm-bias]]의 tie handling에 따라 small-delta form을 쓴다.
- **Abstention.** confidence가 가장 낮은 bin은 headline number에서 제외하고 별도로 드러낸다. "judge not confident, human spot-check" slice다.
- **Ensemble weighting.** 여러 judge를 inverse variance로 weight한다. [[generative-reward-models]]는 "Pairs well with ensembling — GenRM ensembles give calibrated uncertainty."라고 말한다.

[figures/judge-bias.html](figures/judge-bias.html)의 두 번째 panel은 confidence bin으로 scrub할 수 있는 calibration curve를 보여 주며, reference로 `y = x` diagonal을 그려 둔다.

한 가지 미묘함이 있다. [[generative-reward-models]]는 GenRM이 "BT RM이 overconfident한 곳에서 calibrated"라고 보고한다. 여기서 *where*에 주목하라. *uniformly*가 아니다. GenRM은 rubric이 trade-off되는 dimension을 명시적으로 이름 붙일 때만 calibrated다. rubric이 "helpfulness 기준으로 score"라고 말하지만 pair의 차이가 주로 factual correctness라면, judge의 log-prob margin은 *잘못된* 축을 반영하고 그 축에서 calibration은 의미가 없다. calibration은 항상 rubric-conditional이다. 실무적 함의는 calibration curve를 publish할 때 그 옆에 rubric hash도 publish하라는 것이다. 서로 다른 rubric으로 만든 "같은 judge"의 두 curve는 다른 계측기다.

따라서 rubric versioning은 calibration protocol의 일부다. mature stack은 rubric을 source-controlled code로 취급한다. 모든 change는 semver를 올리고, 모든 benchmark number는 rubric version을 인용하며, 모든 rubric change는 human-label set에 대한 re-anchor를 trigger한다(§8). [[meta-rewarding-lm]]의 length-bias control term이 그 prototype이다. 그것은 네 iteration 동안 AlpacaEval 2.0 win-rate를 두 자릿수 움직인 단 하나의 rubric edit이었다. versioning 없이는 그 효과를 추적할 수 없다.

---

## 8. Anchoring — judge를 정직하게 유지하는 단 하나의 관행

모든 synthetic-judge pipeline은 **judge-collapse** 위험을 가진다. 이는 [[direct-judgement-preference]]가 말하는 model collapse([[faithful-synth-eval]])의 유사물이다. judge가 자기 rubric에 수렴하고, rationale은 post-hoc이 되며, benchmark number가 실제 quality를 더 이상 추적하지 않게 된다. 완화책은 주기적 re-anchoring이다.

1. **human-label anchor set** 500–2000 pair를 유지한다. human-judged이며, 어떤 judge training 중에도 본 적이 없어야 한다.
2. 모든 새 judge version, 모든 Self-Taught iteration을 포함해, live 되기 전에 anchor에 대해 평가한다. Spearman, Kendall, agreement-above-chance를 모두 보고한다.
3. anchor correlation이 > 0.05 떨어지면 새 judge를 reject한다. re-anchor round(최근 checkpoint에 대한 새 human label)를 주문한다.

이는 [[faithful-synth-eval]]의 "external-verifier filter"를 judge 자체에 적용한 것이다. 규칙은 일반화된다. **살아 있는 anchor가 없는 judge는 조용히 열화된다.**

명시할 가치가 있는 또 하나의 실패 경로는 **rationale hallucination**이다. [[direct-judgement-preference]]는 이를 직접 지적한다. "natural-language rationales can be post-hoc justifications, not causes." judge의 rationale은 그럴듯하지만 verdict는 여전히 임의적일 수 있다. 방어책은 judge의 *verdict accuracy*를 anchor에 대해 평가하는 것이지, rationale quality를 평가하는 것이 아니다. meta-reward pipeline처럼 rationale을 grade한다면, verdict correctness와 독립된 rubric으로 grade하고, iteration 동안 rationale-score와 verdict-accuracy가 함께 움직이는지 cross-check하라. 둘 사이의 drift는 rationale이 decision에서 분리되었다는 가장 이른 신호다.

---

마지막으로 이름 붙여야 할 calibration의 미묘한 지점이 하나 있다. **judge stochasticity**다. 하나의 judge call에는 sampling temperature에서 오는 variance가 있다(CoT를 greedy가 아니라 sampling하는 경우). [[meta-rewarding-lm]]은 바로 이 이유로 (prompt, actor-response) pair마다 N=11 judge sample을 평균한다. 한 call은 noisy하고, call ensemble은 distribution이다. eval time에 single-sample judge pass로부터 benchmark win-rate를 보고한다면, standard error가 없는 point estimate를 보고하는 것이다. ch-51의 per-run noise budget은 이 term을 명시적으로 포함한다. ch-49는 이 term이 대화에 들어오는 지점이다.

## 9. eval stack과의 연결

- **ch-47 (eval harness design)** — calibrated judge protocol은 harness에 등록하는 per-benchmark spec이다.
- **ch-48 (contamination)** — judge prompt 자체도 leak된다. WildBench와 Arena-Hard prompt는 2025-era judge가 이미 봤을 수 있다. overlap을 확인하라.
- **ch-50 (slice analysis)** — judge bias는 *per-slice* calibration drift로 나타난다. aggregate accuracy는 이를 숨긴다([[faithful-synth-eval]] principle).
- **ch-51 (metric noise + go/no-go)** — prompt별 judge variance는 noise source다. bootstrap CI는 judge seed를 포함한다.
- **ch-42 (reward-hacking taxonomy)** — judge도 train time RM처럼 eval time에 hack될 수 있다. fix는 다르다.
- **ch-44 (RLVR)** — verifiable domain에서는 judge를 verifier로 완전히 교체한다. RLVR은 judge calibration에서 벗어나는 구조적 탈출구다.

---

## 10. Takeaway rule

judge가 instrument-grade가 되려면, judge가 만드는 모든 number에 대해 (a) judge model, (b) rubric file hash, (c) template, (d) swap protocol, (e) length-control method, (f) anchor set and last-anchor correlation, (g) overlap하지 않는 RL-time RM을 말할 수 있어야 한다. 일곱 항목이다. 하나라도 빠지면 그 숫자는 소문이다.

더 깊은 요점은 judge가 세 당사자 사이의 *measurement* contract라는 것이다. training pipeline(무엇을 reward하는가), evaluator(무엇을 측정하는가), anchor(무엇에 대해 audit되는가) 사이의 계약이다. 이어지는 장들, slice analysis(ch-50), metric-noise and go/no-go(ch-51)는 이 contract의 downstream에 있다. 이들이 분석하는 모든 실패는 일곱 항목과 함께 기록되지 않은 judge number에서 시작한다.

흔한 실패 패턴을 말하자면, 한 팀이 "우리 model은 MT-Bench에서 8.4를 받았다"고 보고하지만 reviewer가 재현할 수 없다. judge는 API가 조용히 rotate한 특정 GPT-4 snapshot이었다. rubric에는 versioning되지 않은 minor edit이 있었다. swap-protocol은 구현됐지만 속도 때문에 꺼져 있었다. 세 가지 silent drift, 하나의 재현 불가능한 number다. fix는 절차적이다. 모든 judge number는 YAML header에 일곱 항목을 담아 함께 shipping한다. 이것이 이 장이 eval track을 운영하려면 필수라고 주장하는 contract다.

마지막으로, 이 장은 multi-turn judging, tool-use judging, long-context judging을 논하지 않았다. 이들은 2025년 기준 active research frontier다(WildBench는 multi-turn, Arena-Hard-agent는 tool-use, LongBench는 long-context를 다룬다). 이들의 bias inventory는 §3의 quartet의 확장이다. multi-turn의 position bias는 *turn-position* bias가 되고, verbosity는 *turn-count* bias가 된다. 하지만 calibration protocol의 형태는 같다. 이 frontier 중 하나의 judge를 만들 때 출발점은 이 장의 seven-item contract다. novelty는 어떤 항목을 일반화해야 하는지에 있다.

---

## Further reading

- [[judge-llm-bias]] — bias canon; §3의 모든 숫자는 Zheng 2023 Figs/Tables를 인용한다.
- [[generative-reward-models]] — verdict-log-prob = reward; calibration claim.
- [[direct-judgement-preference]] — Con-J / STE / J1 synthetic-judge 계열; judge-as-weapon risk.
- [[rlaif-scaling]] — CoT prompt template; soft-label extraction; d-RLAIF reward.
- [[ultrafeedback-construction]] — 4-aspect rubric; judge-induced bias propagation; model-fleet contamination.
- [[pairrm]] — swap-augmentation; tournament Best-of-N; 0.4B joint encoder.
- [[self-rewarding-lm]] + [[meta-rewarding-lm]] — judge Spearman evolution; meta-judge role; length-bias rubric control.
- [[faithful-synth-eval]] — judge에 적용된 external-verifier anchor principle.
- [[reward-hacking-taxonomy]] — unhackability impossibility transfers: 어떤 rubric도 un-gameable하지 않다.
- [[wildchat]] — WildBench prompt source인 real-user logs; realism anchor.

## Companion visualization

**[figures/judge-bias.html](figures/judge-bias.html)** — self-contained. Panel 1: bias(position / verbosity / self-enhancement)를 고르고, scenario별 naive-judge accuracy와 corrected-judge accuracy를 비교한다. Zheng 2023에서 확인된 position-flip rate 숫자가 포함된다. Panel 2: confidence bin으로 scrub할 수 있는 calibration curve이며, reference `y = x` line과 "BT RM"(overconfident) / "GenRM"([[generative-reward-models]] Fig. 4 shape) toggle이 있다.
