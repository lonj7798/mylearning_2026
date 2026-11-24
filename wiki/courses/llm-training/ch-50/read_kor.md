<!-- chapter: ch-50
     track: eval
     kind: content
     title: Slice Analysis, Forgetting Slices, and Failure Bucketing
     deps: [ch-49]
     sources: [[tulu-3]], [[ifbench]], [[gsm-symbolic]], [[ruler]], [[lost-in-the-middle]], [[nolima]], [[transferability-of-llm-reasoning]], [[rls-razor]], [[ai2-benchmirt]], [[bfcl]], [[holistic-agent-leaderboard]]
     figures: figures/slice-report.html
     revised: 2026-09 (generality revision)
-->

# 제50장 — Slice analysis, forgetting slice, failure bucketing

> **핵심 통찰.** Aggregate score는 slice의 평균이며, 평균은 서로 반대 방향의 변화를 동시에 숨길 수 있다. [[tulu-3]] Table 6에서 8B checkpoint는 DPO에서 공개 RLVR model로 갈 때 평균 64.7 → 65.1(+0.4)이다. 그러나 그 평균 안에서 GSM8K는 84.3 → 87.6(+3.3), Safety는 87.2 → 85.5(−1.7), TruthfulQA는 56.1 → 55.0(−1.1)이다. 같은 report는 두 번째 hiding도 보인다. Development benchmark HumanEval에서 8B score는 RLVR 전후 같지만(83.9 → 83.9), held-out coding benchmark BigCodeBench에서는 9.5 → 7.4로 하락한다([[tulu-3]] Table 31). Held-out status, perturbation, context length, training stage로 정의한 slice가 aggregate가 파괴하는 정보를 담는다.
>
> **지침.** Checkpoint 비교가 ship 또는 stage 재실행을 결정한다면 item-level record에서 slice별 score를 계산하고 나중에 aggregate한다. Aggregate는 slice의 function이지만 그 역은 아니기 때문이다. Benchmark를 training decision에 썼다면 같은 skill의 held-out benchmark 옆에 보고하고, gap을 overfitting 측정으로 취급한다. [[tulu-3]] Table 31과 [[ifbench]] Table 6은 in-distribution score가 예측하지 못하는 큰 gap을 보인다. Long context 주장이면 length, evidence depth, question-needle lexical overlap grid로 accuracy를 보고한다. [[lost-in-the-middle]]와 [[nolima]] 모두 single-cell test가 grid보다 훨씬 높은 점수를 내는 것을 보인다. Failure를 세는 대신 설명해야 하면 fixed bucket ontology를 tag하고 tagger의 측정된 precision을 기록한다. [[holistic-agent-leaderboard]]에서 LLM tagger precision은 세 check에서 0.87--1.00으로 측정했지만 recall은 전혀 측정하지 않았다.

---

## 범용 모델에서 이 장이 중요한 이유

이전 stage(SFT는 ch-30 이후, preference optimization, RL)는 모두 한 번에 model을 하나 넘는 축에서 바꾼다. Number 하나로 선택한 stage는 그 number가 보상하는 모든 변화에 대해 선택된다. 그 number가 보지 못하는 변화도 포함한다. Course 목표는 target하지 않은 task와 domain에서 수행하는 model이므로, 측정은 다음 세 가지를 검출할 수 있어야 한다.

1. **Narrowing** — 훈련 또는 tuning한 정확한 distribution에 집중된 gain이며, 같은 skill을 요구하는 이웃 distribution에서는 gain이 없거나 loss가 나는 것.
2. **Forgetting** — 더 이른 stage에 있던 capability가 더 나중 stage에서 작아지는 것.
3. **Surface fitting** — task가 아니라 benchmark의 canonical phrasing, length, position, format에 의존하는 score.

셋 모두 aggregate number 하나에서는 보이지 않는다. 각각 average 내부의 redistribution이기 때문이다. 이 장은 그것을 보이게 하는 measurement layer다. Slice 정의, stage 간 slice 비교, failed item을 이름 붙인 countable bucket으로 바꾸는 법을 다룬다. Harness(ch-47), overfitting audit(ch-47a), contamination(ch-48), judge calibration(ch-49) 뒤에, 관측한 per-slice difference가 noise보다 큰지를 말하는 통계(ch-51) 앞에 놓인다.

이후 계속 쓸 정의는 다음이다. **Slice**는 모든 item에 기록된 attribute로 선택한 evaluation item subset이다. Benchmark 출처, development 중 benchmark 사용 여부, 적용한 perturbation, context length, evidence position, score를 낸 training stage가 attribute가 될 수 있다. **Per-slice report**는 그 subset 위 score table이다. Aggregate는 그중 한 행이다.

---

## §1 Stage별·slice별 표와 average가 제거하는 것

**정의.** Per-stage per-slice table은 pipeline checkpoint마다 column 하나, slice마다 row 하나를 두는 표다. 각 cell은 score이며, 각 가로 차이는 stage delta다.

**이것이 다루는 문제.** “RLVR stage를 유지할까”, “다른 mix로 DPO를 재실행할까” 같은 pipeline decision은 stage delta로 내린다. Average만 기록하면 average 내부의 반대 부호 delta는 상쇄되고 residual 위에서 decision을 내린다.

**Mechanism.**
1. Item-level record를 유지하면서 모든 checkpoint를 같은 setting의 같은 item에서 score한다.
2. Item을 기록한 attribute로 slice에 group한다.
3. Slice별 각 stage score와 인접 stage 간 difference를 보고한다.
4. Slice score에서 aggregate를 계산하고 같은 표에 출력해 둘을 비교한다.

**수치 예시([[tulu-3]], arXiv:2411.15124 Table 6, Tülu 3 8B).** 공개 8B column과 DPO column은 다음 benchmark별 stage delta를 준다.

| Slice(benchmark, eval setting) | SFT | DPO | Final(RLVR) | Δ SFT→DPO | Δ DPO→Final |
|---|---|---|---|---|---|
| Average | 60.6 | 64.7 | 65.1 | +4.1 | **+0.4** |
| MMLU(0-shot, CoT) | 65.9 | 68.7 | 68.2 | +2.8 | −0.5 |
| PopQA(15-shot) | 29.3 | 29.3 | 29.1 | 0.0 | −0.2 |
| TruthfulQA(6-shot) | 46.8 | 56.1 | 55.0 | +9.3 | −1.1 |
| BigBenchHard(3-shot, CoT) | 69.7 | 68.7 | 69.0 | −1.0 | +0.3 |
| DROP(3-shot) | 61.3 | 62.5 | 62.6 | +1.2 | +0.1 |
| MATH(4-shot CoT, Flex) | 31.5 | 42.0 | 43.7 | +10.5 | +1.7 |
| GSM8K(8-shot, CoT) | 76.2 | 84.3 | 87.6 | +8.1 | **+3.3** |
| HumanEval(pass@10) | 86.2 | 83.9 | 83.9 | −2.3 | 0.0 |
| HumanEval+(pass@10) | 81.4 | 78.6 | 79.2 | −2.8 | +0.6 |
| IFEval(prompt loose) | 72.8 | 81.1 | 82.4 | +8.3 | +1.3 |
| AlpacaEval 2(LC % win) | 12.4 | 33.5 | 34.5 | +21.1 | +1.0 |
| Safety(6-task avg.) | 93.1 | 87.2 | 85.5 | −5.9 | **−1.7** |

마지막 column의 산술을 확인하자. Benchmark 12개의 delta는 `+(−0.5) + (−0.2) + (−1.1) + 0.3 + 0.1 + 1.7 + 3.3 + 0.0 + 0.6 + 1.3 + 1.0 + (−1.7) = +4.8`이며, 12 benchmark에 대한 mean delta는 `+0.40`이다. Average row의 +0.4다. 가장 큰 single component는 +3.3, 가장 negative인 것은 −1.7이다. Aggregate는 둘보다 작다. “average가 개선되면 ship”이라는 rule은 Safety/TruthfulQA loss를 보지 못한 채 checkpoint를 수용하며, “아무것도 regress하지 않으면 ship”이라는 rule은 GSM8K gain을 보지 못한 채 reject한다.

같은 report의 RLVR-stage table([[tulu-3]] Table 23, §6.4)은 8B에서 DPO→RLVR benchmark별 comparison을 average 64.4 → 64.8로 준다. 두 표는 일부 row에서 다른 eval setting을 쓰므로 average와 BigBenchHard가 다르다. 위의 benchmark별 값은 Table 6 것이다. Report의 prose는 8B 결과를 “MATH, GSM8k, IFEval 세 개 모두에서 개선하는 non-trivial improvement”로 설명하지만(§6.4), Safety/TruthfulQA row는 언급하지 않는다. **결과(단일 연구).**

**조건과 한계.** 이 delta는 seed 하나, model size 하나, recipe 하나다. 같은 표의 70B column은 다른 pattern(GSM8K 93.5 → 93.5)을 보이며 §6.4는 saturation으로 돌린다. 특정 per-slice delta가 run-to-run noise와 구분되는지는 ch-51의 주제다. 이 절은 위 single row 어느 것도 noise 밖이라고 확립하지 않는다.

**범용 모델에 주는 함의.** Target gain을 낸 stage는 general use와 가장 가까운 두 slice(safety behavior, truthfulness)에서 loss도 냈다. Average에 최적화한 pipeline은 stage가 지날수록 이런 trade를 조용히 누적한다.

Companion figure [figures/slice-report.html](figures/slice-report.html)는 이 표와 뒤따르는 두 표를 담는다. 첫 view는 Average row만과 full per-benchmark column을 toggle한다. 같은 checkpoint pair를 두 reporting choice 아래에서 읽을 수 있다.

## §2 In-distribution/held-out slice와 그 사이 gap

**정의.** **In-distribution(development) slice**는 training과 mixture decision을 내릴 때 score가 보였던 slice다. **Held-out(unseen) slice**는 같은 skill을 재지만 development에서 참고하지 않은 slice다. Skill의 **generalization gap**은 둘의 차이다.

**이것이 다루는 문제.** Benchmark에서 선택하는 decision procedure는 그 benchmark에 특유한 property를 선택할 수 있다. 그 효과의 크기는 benchmark 자체에서 알 수 없다.

**Mechanism.** [[tulu-3]] §2.2와 §7.4의 design은 evaluation suite를 같은 core skill을 포괄하는 development suite와 unseen suite로 나눈다. 저자들은 “model을 개발할 때 unseen set score를 보지 않았으므로 얼마나 overfit했을지를 관찰할 수 있었다”고 밝힌다(§2.2). Skill마다 pair를 둔다. MMLU → GPQA, BigBenchHard → AGIEval, MATH → DeepMind Mathematics, HumanEval → BigCodeBench, IFEval → IFEval-OOD(§7.2--7.3, Table 3)다.

**수치 예시([[tulu-3]] Table 31, 8B).** Skill별 development(Dev.)와 unseen(Uns.) score다.

| Skill(dev → unseen) | SFT Dev / Uns | DPO Dev / Uns | Final Dev / Uns | Final gap |
|---|---|---|---|---|
| Average | 64.9 / 29.9 | 68.3 / 31.9 | 68.8 / 32.4 | 36.4 |
| Knowledge(MMLU → GPQA) | 65.9 / 31.9 | 68.7 / 31.2 | 68.2 / 35.7 | 32.5 |
| Reasoning(BBH → AGIEval) | 67.9 / 56.2 | 65.8 / 61.8 | 66.0 / 59.3 | 6.7 |
| Math(MATH → DM Mathematics) | 31.5 / 32.3 | 42.0 / 33.0 | 43.7 / 35.4 | 8.3 |
| Coding(HumanEval → BigCodeBench) | 86.2 / 11.5 | 83.9 / 9.5 | 83.9 / 7.4 | 76.5 |
| Instruction following(IFEval → IFEval-OOD) | 72.8 / 17.6 | 81.1 / 23.9 | 82.4 / 24.3 | 58.1 |

Development column만으로는 뒷받침할 수 없는 읽기 두 가지가 있다.

- **Coding.** Development score는 DPO에서 final까지 변하지 않는다(83.9 → 83.9). Held-out score는 9.5 → 7.4로 하락한다. HumanEval만으로 만든 report는 “변화 없음”을 기록한다.
- **Instruction following.** Development score는 pipeline 전체에서 72.8 → 82.4(+9.6), held-out score는 17.6 → 24.3(+6.7)으로 오른다. 그러나 held-out은 58.1점 아래에 머문다. Gain은 양쪽 slice에서 실제지만 absolute level은 전이되지 않는다.

더 넓은 비교에서 저자 자신의 결론은 혼합돼 있다. Final checkpoint는 두 split 모두에서 best average를 갖는다(§7.4.1). 동시에 “Precise Instruction Following development evaluation에, Knowledge Recall/Reasoning에는 어느 정도 overfit했다”고 한다(§7.4.1). DPO data-scaling curve는 “development process가 MATH에 어느 정도 overfit했다”고 보이며, MATH와 DeepMind Mathematics의 LaTeX formatting 차이로 돌린다(§7.4.1, Fig. 24). **결과(단일 연구).**

**같은 gap의 두 번째 측정([[ifbench]]).** IFBench는 training constraint 29개와 IFEval set에서 모두 disjoint인 verifiable output constraint 58개를 hold한다(§2, App. A Table 8). Varied constraint에서 GRPO로 Tülu-3-8B-DPO policy를 훈련하면 IFEval은 81.1 → 92.2, IFBench는 25.2 → 44.6이 된다(Table 6). Gap은 55.9에서 47.6점으로 좁아지지만 닫히지 않는다. [[tulu-3]] Table 31의 IFEval/IFEval-OOD row와 **반복된다.** Unseen-constraint set을 독립적으로 두 번 구성해도 같은 qualitative result가 난다.

**조건과 한계.** 이 종류의 gap은 design이 분리하지 못하는 세 원인을 섞는다. Development benchmark에서의 selection, paired benchmark 간 genuine difficulty 차이, format 차이다. [[tulu-3]] §7.4.1은 math gap 일부를 format에 돌린다. 그러므로 gap은 overfitting의 측정이 아니라 overfitting에 대한 upper bound다.

**범용 모델에 주는 함의.** Held-out column은 “target하지 않은 task에서 수행한다”의 가장 가까운 proxy다. 이것을 유지하지 않는 pipeline은 generality를 전혀 보고할 수 없다. [[tulu-3]] §2.2가 말하듯 split은 development 동안 held-out score를 보지 않을 때만 작동한다.

---

## §3 Perturbation slice: 같은 skill, 다른 surface

**정의.** **Perturbation slice**는 필요한 reasoning은 보존하고 surface는 바꾸는 benchmark item의 controlled transformation으로 만든 item을 담는다. Name, numeric value, clause count, 관련 있어 보이지만 무관한 clause 추가를 바꾼다.

**이것이 다루는 문제.** 고정 public test set score는 skill과 정확히 그 set에 익숙함을 confound한다. Perturbation은 skill을 고정하여 둘을 분리한다.

**Mechanism([[gsm-symbolic]], arXiv:2410.05229).** GSM8K test question 100개를 typed variable과 correctness condition을 가진 symbolic template으로 바꾼다(§3.1). 이 template에서 각각 100 example인 dataset 50개를 생성하며, 각 example은 100 original 중 하나의 mutation이다(§4). Variant는 clause를 더하거나 뺀다. Symbolic-M1은 clause 하나를 빼고, Symbolic-P1/P2는 하나와 둘을 더하며, Symbolic-NoOp은 관련 있어 보이지만 solution에는 들어가지 않는 clause 하나를 추가한다(§4.3, §4.4).

**수치 예시([[gsm-symbolic]] Table 1, 8-shot).** 행 하나를 slice ladder로 읽는다.

| Model | GSM8K(100 original) | Symbolic | Symbolic-P1 | Symbolic-P2 | Symbolic-NoOp |
|---|---|---|---|---|---|
| Gemma2-9b-it | 87.0 | 79.1 ± 3.0 | 68.1 ± 4.8 | 41.8 ± 6.0 | 22.3 ± 5.1 |
| Phi-3-medium-128k-instruct | 89.0 | 82.5 ± 2.9 | 75.8 ± 3.9 | 53.1 ± 4.8 | 29.4 ± 4.2 |
| Llama3-8b-instruct | 74.0 | 74.6 ± 2.9 | 53.8 ± 4.5 | 28.3 ± 4.4 | 18.6 ± 3.9 |
| GPT-4o | 95.0 | 94.9 ± 1.9 | 93.9 ± 2.6 | 88.0 ± 3.4 | 63.1 ± 4.5 |
| o1-preview | 96.0 | 92.7 ± 1.8 | 95.4 ± 1.7 | 94.0 ± 2.4 | 77.4 ± 3.8 |

Column에서 slice fact 두 가지가 나온다. 첫째, original-to-Symbolic column difference는 모든 model에서 같지 않다. Gemma2-9b-it는 7.9점을 잃고, Llama3-8b-instruct는 0.6점을 얻으며, GPT-4o는 0.1점을 잃는다. 저자들은 model 25개 중 21개에서 100 original score가 Symbolic distribution 중심에서 one standard deviation 이상 떨어져 있으며, 보통 더 높다고 보고한다. Data contamination을 하나의 설명으로 든다(§4.1, **해석**). 둘째, NoOp column은 GSM8K column이 구분하지 못하는 model을 구분한다. Gemma2-9b-it와 Phi-3-medium은 GSM8K에서 2점 차이, NoOp에서 7점 차이다. Llama3-8b-instruct는 GSM8K에서 Gemma2-9b-it보다 13점 낮지만 NoOp에서는 3.7점 낮다.

표의 generated-instance standard deviation은 이 design의 두 번째 산물이다. 같은 question을 re-instantiate하면 point가 아니라 score distribution을 준다. 저자들은 proper name만 바꿔도 variation이 생기며 numeric value도 바꾸면 variance가 커진다고 보고한다(§4.2, Fig. 4). **결과(단일 연구).**

**조건과 한계.** Template은 GSM8K question 100개에서 왔다. 그러므로 결과는 grade-school arithmetic word problem과 benchmark 하나의 style에 관한 것이다. NoOp drop은 distractor text를 다루는 능력의 측정이기도 하다. Robustness artifact만이 아니라 독자 capability다.

**범용 모델에 주는 함의.** Perturbation slice는 capability가 task에 붙었는지 dataset 하나의 surface form에 붙었는지를 재는 가장 값싼 test다. Benchmark 한 번을 template-writing하면 되며, output은 canonical score 하나를 해석 가능하게 만드는 distribution이다.

---

## §4 Long-context slice: length, depth, hop, overlap, distractor

Long-context 주장은 grid에 대한 주장이다. Grid를 연 공개 측정은 모두 marginal score보다 훨씬 낮은 cell을 발견한다.

**Length([[ruler]], arXiv:2404.06654 Table 3).** RULER는 4K, 8K, 16K, 32K, 64K, 128K 각각에서 synthetic task 13개의 average를 내며, **effective length**를 Llama2-7B 4K score 85.6 이상으로 average가 유지되는 가장 큰 length로 정의한다. 선택 행은 다음과 같다.

| Model | Claimed | Effective | 4K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| Gemini-1.5-Pro | 1M | >128K | 96.7 | 95.9 | 95.9 | 94.4 |
| GPT-4 | 128K | 64K | 96.6 | 93.2 | 87.0 | 81.2 |
| Llama3.1(70B) | 128K | 64K | 96.5 | 94.8 | 88.4 | 66.6 |
| Yi(34B) | 200K | 32K | 93.3 | 87.5 | 83.2 | 77.3 |
| Mixtral-8x22B | 64K | 32K | 95.6 | 90.9 | 84.7 | 31.7 |
| LWM(7B) | 1M | <4K | 82.3 | 69.1 | 68.1 | 65.0 |

Claimed-versus-effective column이 slice result다. 이 행에서는 Gemini-1.5-Pro만 effective length가 claimed length에 도달한다. RULER는 length에 따라 weight가 선형 증가/감소하는 `wAvg. (inc)`, `wAvg. (dec)` weighted average 둘도 보고한다. Rank는 둘 사이에서 다르다. LWM은 `inc`에서 12위, `dec`에서 15위다(Table 3). Aggregation weight 자체가 reporting choice다.

**고정 length의 task family([[ruler]] §5, Fig. 2).** Model(Yi-34B)을 고정하고 generator를 바꾸면 word-number single-needle retrieval에서는 거의 완벽하다가 needle이 32-digit UUID가 되면 저하한다. Distractor needle을 늘리면 accuracy는 꾸준히 내려가며, context를 distractor needle로 채운 version(`#K=FULL`)에서는 256K에서 약 40점 떨어진다. Error analysis는 그 setting에서 model이 target 근처의 value를 자주 반환한다고 한다(§5). **결과(단일 연구).**

**Depth([[lost-in-the-middle]], arXiv:2307.03172).** Multi-document QA에서 `k`개 retrieved document 중 하나만 answer를 갖고, 나머지 input을 고정한 채 그 position을 바꾼다(§2.1). Document 20개에서 GPT-3.5-Turbo score는 position 1 / 5 / 10 / 15 / 20에서 75.8 / 57.2 / 53.8 / 55.4 / 63.2다(App. G Table 6). Document가 없는 closed-book score는 56.1, answer document만 있는 oracle score는 88.3이다(Table 1). Position-10 cell 53.8은 closed-book보다 낮다. 이 slice에서는 retrieved context를 쓰지 않는다. Window를 늘려도 바뀌지 않는다. GPT-3.5-Turbo와 16K variant는 document 20개 position 1에서 75.8, 75.7이다(Table 6). **결과(단일 연구).**

**Lexical overlap([[nolima]], arXiv:2502.05167).** NoLiMa는 ROUGE-1 precision overlap이 0.069인 question-needle pair를 만든다. Vanilla needle-in-a-haystack은 0.905, RULER S-NIAH는 0.571이다(Table 1). 같은 model, 같은 haystack, overlap 조건 세 개(Table 6, Llama 3.3 70B, 8K / 16K / 32K)에서 needle keyword를 직접 부르는 question은 98.3 / 98.5 / 98.5다. One-hop association은 84.1 / 73.2 / 56.2, two-hop association은 57.4 / 42.7 / 25.9다. 논문의 `0.85 × base-score` threshold에서 effective length는 claimed length보다 훨씬 낮다. GPT-4o는 128K claimed/8K effective, Llama 3.3 70B는 128K claimed/2K effective다(Table 3). Question keyword를 담은 distractor sentence 하나를 더하면 GPT-4o effective length는 1K로 내려간다(§4.4.4). Chain-of-thought는 partial remedy다. 32K two-hop은 CoT로 25.9 → 34.3이지만, 모든 length에서 CoT 없는 one-hop보다 낮다(Table 4). **결과(단일 연구).**

NoLiMa depth sweep은 32K one-hop question에서 [[lost-in-the-middle]] dip을 재현한다. Two-hop question은 edge를 포함한 모든 depth에서 context가 길수록 curve가 내려간다고 보고한다(§4.4.2, Fig. 3). One-hop case는 두 논문에서 **반복된다.**

**여기서 따르는 slice specification.** Long-context report는 적어도 context length, evidence depth(context fraction으로서의 position), question에서 evidence까지 reasoning hop, question-evidence lexical overlap, distractor count의 grid여야 한다. 위 네 source는 이 coordinate 하나가 collapse해도 marginal score가 높게 남는 경우를 각각 보인다. “128K”라고 붙은 number 하나는 출력하지 않은 grid의 mean이다.

---

## §5 Cross-stage forgetting과 transfer slice

**정의.** **Forgetting slice**는 current stage가 target하지 않았고 stage 이전에도 측정 가능했던 capability를 재는 benchmark다. **Transfer slice**는 target에 인접한 capability를 재는 benchmark다. 둘 다 level이 아니라 stage delta로 읽는다.

**이것이 다루는 문제.** Post-training stage는 보통 target capability에서만 평가한다. Untargeted capability loss를 동시에 재지 않으면 pipeline은 stage마다 local하게 개선되면서 global하게 좁아질 수 있다.

**Mechanism과 metric([[transferability-of-llm-reasoning]], arXiv:2507.00432 §2.1).** Benchmark를 math reasoning, other reasoning, non-reasoning 세 group으로 나눈다. Base model 대비 benchmark별 gain을 group 안에서 z-normalize하고 signed square root로 compress하며, task difficulty로 weight해 group Domain Index로 aggregate한다.

```
ΔR_b = R_b^model − R_b^base                       benchmark별 gain
σ_g  = Std{ΔR_b : b ∈ B_g}                        group 내부 spread
δ_b  = ΔR_b / σ_g                                 normalized gain
s_b  = sign(δ_b) · |δ_b|^(1/2)                    signed square root
w_b  = 100 − R_b^base                             difficulty weight
ŵ_b  = w_b / Σ_{u∈B_g} w_u                        normalized weight
DI_g = Σ_{b∈B_g} ŵ_b · s_b                        group Domain Index
TI_g(%) = 100 · DI_g / DI_math                    Transferability Index
```

Symbol은 다음과 같다. `b`는 benchmark, `B_g`는 `g ∈ {math, other, non}` group의 benchmark, `R_b^model`/`R_b^base`는 tuned/base model의 `b` accuracy, `Std`는 group gain의 standard deviation이다. `TI_g`는 group aggregate gain을 math group gain으로 나눈 percent다. `TI = 100`은 group이 math만큼 움직였음을 뜻하며, negative `TI`는 반대 방향으로 움직였음을 뜻한다.

**수치 예시(같은 논문, Table 1, Qwen3-14B-Base).** Group average다.

| Model | Math avg | Other-reasoning avg | Non-reasoning avg | TI_other | TI_non |
|---|---|---|---|---|---|
| Qwen3-14B-Base | 27.7 | 30.2 | 45.7 | — | — |
| UniReason-Qwen3-14B-think(SFT) | 49.8 | 45.3 | 21.1 | +52.2 | −104.1 |
| UniReason-Qwen3-14B-no-think(SFT) | 32.3 | 45.2 | 29.0 | +165.4 | −278.9 |
| UniReason-Qwen3-14B(RL) | 53.8 | 60.0 | 53.2 | +82.3 | +52.2 |

SFT(think) run은 math에서 +22.1을 얻고 non-reasoning average에서 24.6을 잃는다. Individual column은 CoQA 10.0 → 1.7, IFEval 69.2 → 42.3, HalluEval 35.7 → 2.3이다. RL run은 더 높은 math average(49.8 대 53.8)에 도달하면서 non-reasoning average도 45.7 → 53.2로 올린다. Math column만 포함한 report라면 두 run을 같은 종류의 개선으로 묘사할 것이다. **결과(단일 연구; base model 하나, 14B size 하나, math corpus 하나).**

**Forgetting slice의 predictor([[rls-razor]], arXiv:2509.04259).** Qwen2.5-3B-Instruct를 new task 세 개(Open-Reasoner-Zero math, SciKnowEval Chemistry L-3, ToolAlpaca)에 fine-tune하고 prior capability를 HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval에서 재면, 저자들은 new-task distribution에서 측정한 fine-tuned/base policy 간 KL divergence `E_{x∼τ} KL(π₀ ‖ π)`가 forgetting을 예측한다고 보고한다. RL/SFT point는 이를 x-axis로 그리면 curve 하나 위에 놓인다. Controlled ParityMNIST에서는 quadratic fit `R² = 0.96`, LLM experiment에서는 `R² = 0.71`이다(§4, Fig. 3, Fig. 11). Analytically KL-minimal labeling으로 SFT를 훈련하면 RL보다 prior knowledge가 더 보존됐다(§4). Algorithm이 operative variable이 아니라는 저자의 근거다.

**Slice design에 주는 함의.** Target score 옆 per-stage report에는 untargeted-capability slice score와 new-task distribution KL divergence 두 양이 있어야 한다. 후자는 forgetting benchmark 실행 전에 cheap하게 계산할 수 있다. 두 번째에 대한 **해석**은 다음이다. Source는 correlation을 확립했으며, KL number 하나가 decision에 충분하다는 것을 확립하지는 않았다.

---

## §6 Failure bucketing: reason bucket과 confusion cell

**정의.** **Bucketing**은 failed item마다 fixed ontology의 label 하나를 부여해 failure를 종류별로 count하는 일이다. Ontology 두 개를 쓴다. **Reason bucket**은 rubric이 부여하는 free-form category이고, **confusion cell**은 output space 구조가 정하며 구성상 mutually exclusive다.

**Output이 구조화됐을 때의 confusion cell([[bfcl]], PMLR 267:48371).** Function calling에는 열거 가능한 failure 구조가 있다. BFCL AST matcher는 Python `ast` module로 각 call을 parse하고 exact function-name match 및 parameter value 각각이 accepted answer set에 있을 것을 요구한다(§4.1, App. H). 그러므로 failure는 wrong function name, right function인데 wrong argument, Irrelevance category가 call 없음이라고 기대하는데 call을 낸 경우, call을 기대하는데 내지 않은 경우, unparseable output으로 분해된다. Scored category 두 개가 마지막 두 경우를 직접 분리한다. Irrelevance는 call을 기대하지 않고, Relevance는 call 하나 이상을 기대한다(§3.1, App. C.2). Number는 독립적으로 움직임을 보인다. Prompting mode Qwen2.5-72B-Instruct는 Relevance 100.0, Irrelevance 72.8이다(Table 1). Model은 right call emission은 개선하고 abstention은 regress할 수 있으며, function-calling accuracy 하나는 이를 숨긴다.

같은 benchmark는 format 대 content 분해도 보인다. Entry 4,251개 중 prompting-mode model은 decoding issue 평균 412.93개, function-calling-mode model은 182.5개다. 하지만 decode된 response 중 function-calling-mode model은 Multiple category에서 wrong number of call을 더 자주 낸다(77.5 대 21)(§5.1). **결과(단일 연구).**

**Output이 open-ended일 때의 reason bucket.** Procedure는 다음이다. Failed item sample을 읽고 ontology를 쓴다. Failure마다 label 하나와 one-sentence justification 하나를 붙인다. `other` bucket이 작아질 때까지 ontology를 확장하고 slice별로 count한다. Ontology는 서로 다른 fix에 mapping되는 mechanism의 이름을 붙여야 한다. 이전 장이 근거를 제공한 bucket과 standard pipeline에 이미 있는 diagnostic은 다음과 같다.

| Bucket | Bucket이 존재한다는 근거 | Diagnostic signal |
|---|---|---|
| unseen constraint type의 `constraint-violation` | [[ifbench]] Table 6: IF-RLVR 뒤 IFEval 92.2 대 IFBench 44.6 | held-out constraint의 constraint-category별 pass rate |
| `distractor-sensitivity` | [[gsm-symbolic]] Table 1 NoOp column(Gemma2-9b-it 79.1 → 22.3) | paired canonical/NoOp item score |
| `position-dependent-retrieval` | [[lost-in-the-middle]] Table 6(position 1에서 75.8, position 10에서 53.8) | evidence depth별 accuracy |
| `lexical-match-dependence` | [[nolima]] Table 6(32K에서 direct 98.5 대 one-hop 56.2) | question-needle overlap별 accuracy |
| `abstention-failure` | [[bfcl]] Table 1 Irrelevance 대 Relevance | Irrelevance category score |
| `format-decoding-failure` | [[bfcl]] §5.1 decoding-issue count | mode별 parse-failure rate |

**Tagger도 측정해야 한다.** Tagger가 LLM이면 label은 error rate가 있는 measurement다. [[holistic-agent-leaderboard]]는 rubric judge의 human-validated precision을 보고한다. AssistantBench instruction following 0.87(`n = 49`, inter-LLM `κ = 0.82`), CORE-Bench verification 1.00(`n = 31`), TAU-bench instruction following 0.94(`n = 36`)다. Precision만 쟀고 recall은 재지 않았다고도 명시한다(App. A7.2, Table A2). Recall 없는 precision은 bucket false-positive rate만 bound하고, 그 kind failure를 얼마나 놓쳤는지는 말하지 않는다. 그러므로 bucket count는 lower bound다. [[bfcl]]은 multi-turn error analysis에서 GPT-4o-08-06을 judge로 쓰며(§5.4.2, App. F), 가장 빈번한 root cause를 “Failed to Understand Environment State”, 다음을 “Failed to Understand User's Request”로 보고한다. Scoring judge처럼 reason-tagger에도 ch-49 judge-bias check(position swap, length control, self-preference)를 적용한다. Run 사이 tagger가 바뀌면 ledger count는 비교할 수 없다.

**최소 pipeline.** Item-level record를 한 번 통과해 output 두 개를 만든다.

```python
# eval/bucket_failures.py — item record 한 번의 통과로 reason bucket과 confusion cell을 만든다.
from collections import Counter, defaultdict

def bucket_failures(items, grader, reason_tagger, confusion_tagger=None):
    """items: [{id, prompt, response, gold, slice_tags: {suite, split, length, depth, stage}}]
       grader: (response, gold) -> bool
       reason_tagger: 고정 ontology의 label을 item에 부여
       confusion_tagger: output space가 열거 가능할 때 item에 cell label을 부여"""
    n_total, n_correct = Counter(), Counter()
    by_reason, by_cell = defaultdict(Counter), defaultdict(Counter)
    for it in items:
        key = (it["slice_tags"]["suite"], it["slice_tags"]["split"])
        n_total[key] += 1
        if grader(it["response"], it["gold"]):
            n_correct[key] += 1
            continue
        by_reason[key][reason_tagger(it)] += 1
        if confusion_tagger is not None:
            by_cell[key][confusion_tagger(it)] += 1
    accuracy = {k: n_correct[k] / n_total[k] for k in n_total}
    return {"accuracy": accuracy, "by_reason": dict(by_reason), "by_cell": dict(by_cell)}
```

`by_reason`은 가장 큰 failure class를 어느 data/training change가 다루는지 답한다. `by_cell`은 어느 scored sub-metric이 움직였는지 답한다. 질문이 다르며, 두 번째는 label이 model을 통과하지 않아 더 신뢰할 수 있다.

---

## §7 Agent를 위한 trajectory failure bucket

**정의.** Agent task에서 item은 trajectory다. Model message, tool call, environment observation의 sequence이며 final environment state에 대한 success predicate로 끝난다. **Trajectory bucket**은 trajectory가 어느 지점에서 잘못됐는지 붙이는 label이다. Final score에서는 복원할 수 없다.

**이것이 다루는 문제.** 200-step trajectory의 binary task outcome은 episode 전체를 bit 하나로 압축한다. Failure fix는 어느 step과 어느 cause가 만들었는지에 따라 달라진다.

**Mechanism과 측정한 prevalence([[holistic-agent-leaderboard]], arXiv:2510.11977).** HAL은 model 9개와 benchmark 9개에 약 $40,000으로 rollout 21,730개를 실행했다(Abstract). 이어 benchmark 4개에서 transcript 2,184개에 LLM rubric(Docent, judge로 GPT-5 Medium)을 적용했고, TAU-bench를 제거한 뒤 1,634개다(App. A7.1). Rubric은 instruction violation, tool-use failure, self-correction, verification, environmental barrier, shortcut/gaming 6 category다(§4.2, Table A5). Failed task의 측정 prevalence는 다음과 같다.

| Bucket | AssistantBench | CORE-Bench Hard | SciCode |
|---|---|---|---|
| Instruction violation | 67.0% | 62.8% | 미보고 |
| Environmental barrier | 56.4% | 40.3% | 43.8% |
| Tool-call failure 하나 이상(failed run) | 미보고 | 89.7% | 97.7% |
| Tool-call failure 하나 이상(successful run) | 미보고 | 83.9% | 100% |

마지막 두 행이 bucket count 하나만으로 diagnosis가 안 되는 이유다. SciCode에서는 successful trajectory를 포함한 사실상 모든 trajectory에 tool-call failure가 나타난다. 따라서 그 존재는 failure와 success를 구분하지 않는다. 구분하는 bucket은 recovery behavior다. Self-correction이 있으면 success 가능성이 더 높고 risk ratio는 AssistantBench/SciCode/CORE-Bench에서 1.47, 1.54, 2.97이다. Verification은 1.39, 1.87, 1.13이다(§4.2). **결과(단일 연구).**

**Environment error 대 model error.** HAL의 “environmental barrier” bucket은 model이 피할 수 없던 failure를 분리한다. Appendix는 concrete case를 열거한다. Provider가 같은 name 아래 DeepSeek R1 endpoint를 R1-0528로 조용히 바꾸고, router가 한 call에는 FP4, 다른 call에는 FP8을 제공하며, silent rate-limit failure가 wrong answer로 score됐다(App. A3). 이들은 model 변화 없이 score를 바꾼다. 같은 log analysis는 반대 error도 발견했다. 공식 TAU-bench few-shot file `few_shot_data/MockAirlineDomainEnv-few_shot.jsonl`에 test-set example이 있었고, 약 $1,000의 evaluation 뒤 발견해 그 scaffold의 모든 result를 제외했다(App. A5).

**Scaffold는 slice axis다.** HAL은 task-specific scaffold가 CORE-Bench Hard 12 run 중 9개, SWE-bench Verified Mini 12 run 중 11개에서 generalist scaffold보다 낫고, generalist는 24 comparison 중 20개에서 더 싸다고 보고한다. Rank는 model과 interaction한다. Online Mind2Web에서 Claude model은 Browser-Use, OpenAI model은 SeeAct에서 더 높다. GPT-5 Medium의 SeeAct는 $171이고 Claude Sonnet 4의 Browser-Use는 accuracy 2점 차이에 $1,577이다(§4.1 item 6--7). Fixed scaffold 없는 model별 agent score는 model-scaffold pair score다.

**Shortcut bucket.** Agent가 task를 푸는 대신 HuggingFace/arXiv에서 gold answer를 찾은 case 8개, CORE-Bench/SciCode에서 hard-code한 “plausible” solution이 발견됐다(§4.2, Table A7--A8, A10). Outcome predicate는 이를 success로 score하므로 trajectory를 읽어야만 이 bucket을 관측할 수 있다.

**조건과 한계.** Cost 때문에 HAL evaluation 대부분은 CI 없는 single run이다(App. A3 item 1). 논문은 flagged failure를 고치면 success가 났을지를 확립할 수 없다고 말한다. Checkpointing과 replay가 필요하다(App. A4.2). Trajectory bucket은 일어난 일의 description이지 무엇이 fix할지에 관한 causal claim이 아니다.

---

## §8 Failure ledger와 decision에 맞춘 report

**Ledger.** Per-run report는 snapshot이다. Ledger는 같은 bucket을 run에 걸쳐 들고 간다. `(bucket, slice)`마다 row 하나, run마다 column 하나이며 `{run_id, checkpoint, suite, split, slice_key, bucket, count, n_items, share, example_ids, first_seen_run, tagger_version, fix_attempted}`를 기록한다. Usable하게 만드는 column은 `tagger_version`이다. Tagger나 rubric이 바뀌면 전후 count는 서로 다른 instrument의 measurement이며, [[holistic-agent-leaderboard]] precision 수치는 judge/rubric 하나에만 적용된다(App. A7.2).

**Automated slice discovery.** Slice는 선언하는 대신 찾아낼 수도 있다. [[ai2-benchmirt]]는 benchmark가 무엇을 재는지 model에 알려주지 않고, benchmark 16개와 item 34,301개에 걸친 open-weight LLM 100개에서 two-dimensional item response theory model을 fit한다. 저자들은 두 ability를 safety와 general reasoning이라 label한다(blog; report §2.2--2.3). Benchmark 밖 label은 그 benchmark가 실제로 재는 것을 정하지 않는다는 것이 benchmark별 correlation에서 보인다. Safety로 group한 BBQ는 general reasoning 0.85, safety axis −0.06과 correlate한다. WMDP는 general reasoning −0.89, XSTest는 두 축 사이로 갈라지며 even over-refusal/safety split과 일치한다. WildJailbreak benign subset(2,250 item 중 250)과 HarmBench copyright subset은 reasoning 쪽으로 기울지만 각 benchmark의 나머지는 safety와 align한다(blog table, “What BenchMIRT reveals”). **결과(단일 연구; 두 dimension은 benchmark 16개 집합에 의존하고, 모든 model은 2025년 3월까지 공개됐다).** Slice report의 operational consequence는 benchmark 자체 subset이 중요한 slice boundary일 수 있고, 이를 average하면 construct를 섞는다는 것이다.

**Artifact를 decision에 맞추기.** 아래 각 행은 다른 experiment가 아니라 같은 item-level record table에 대한 다른 query다.

| 내려야 할 decision | 답하는 slice axis | 이 장의 근거 |
|---|---|---|
| 이 stage를 유지할까? | training stage × benchmark | [[tulu-3]] Table 6 §1 |
| Gain이 개발에 쓴 benchmark를 넘어 generalize했나? | skill별 development 대 held-out | [[tulu-3]] Table 31, [[ifbench]] Table 6 §2 |
| Capability가 task에 붙었나, surface에 붙었나? | perturbation ladder | [[gsm-symbolic]] Table 1 §3 |
| Long-context claim이 성립하나? | length × depth × hop × overlap × distractor | [[ruler]], [[lost-in-the-middle]], [[nolima]] §4 |
| 이 stage는 다른 곳에서 무엇을 cost했나? | untargeted-capability slice, new task의 KL | [[transferability-of-llm-reasoning]], [[rls-razor]] §5 |
| 어느 fix가 가장 많은 failure를 다루나? | slice별 reason bucket/confusion cell | [[bfcl]] §6 |
| Agent run은 어디서 깨지나? | trajectory bucket, scaffold, environment 대 model | [[holistic-agent-leaderboard]] §7 |
| 이 benchmark는 label이 말하는 것을 재나? | item-level ability decomposition | [[ai2-benchmirt]] §8 |

**다음 장에 속하는 통계 주의점 하나.** Slice 50개 report를 5% level에서 test하면, 모든 slice가 true null이어도 예상 `50 × 0.05 = 2.5`개 slice가 changed로 flag된다. Multiple-comparison correction 없이 per-slice report를 읽으면 예측 가능한 rate로 regression을 만들어 낸다. Correction, confidence interval, paired comparison은 ch-51의 주제고, 이 장은 그들이 작동할 table을 만든다.

---

## 부정 sample과 부정 feedback

이 장은 failure로 훈련하지 않고 failure를 소비한다. Course negative-feedback taxonomy 네 의미 중 (2), **content로서의 negative**다. Failed item을 읽고 label하며 다음 data/training change를 고르는 데 쓴다. 여기서는 그것으로 gradient를 취하지 않으므로 likelihood displacement와 squeezing은 적용되지 않는다.

**Negative의 출처와 label 방식.** 위에는 error property가 다른 labeling mechanism 세 개가 있다. **Program verifier**는 failure label을 직접 만든다. [[bfcl]] AST matcher에서는 Python에서 float를 기대할 때 int를 accept하지만 Java/JavaScript에서는 그렇지 않은 wrong accept가 failure mode이며 wrong reject가 아니다(§4.1, App. H). **Model tagger**는 측정한 precision과 측정하지 않은 recall로 label을 만든다([[holistic-agent-leaderboard]] Table A2: 0.87 / 1.00 / 0.94, recall 미측정). **Generated-item filter**는 validation rate가 있는 label을 만든다. [[nolima]]는 Llama 3.3 70B로 1,000-character chunk 위 haystack을 filter하고, needle을 random chunk 100개에 각각 놓는 control test에서 99.8%로 filter를 validate한다(§3.1, §4.2).

**그것으로 하는 일.** Slice별 count 후 fix를 선택한다. 이 장이 허가하지 않는 step은 bucket count 자체로 훈련하는 것이다. [[holistic-agent-leaderboard]]는 checkpointing/replay 없이는 flagged failure를 고치면 success가 생기는지 확립할 수 없다고 말한다(App. A4.2).

**Failure bucket에 최적화할 때 측정한 위험.** [[ifbench]]는 constraint verifier에 직접 훈련하고 target 밖 cost를 잰다. Pure constraint reward는 LLM-judge response quality를 10점 만점 7에서 6.4, AlpacaEval 2를 33.5에서 21.3으로 낮췄다. Verifiable reward에 preference reward-model term을 더하는 것이 paper의 mitigation이다(§5, Table 3, App. E). **결과(단일 연구).** 줄어든 bucket만 움직인 양이 아니다.

**Bucket count 옆에 둘 진단.** Abstention rate: over-corrected model은 더 많이 거절해 `abstention-failure` bucket을 줄이며, [[bfcl]] Table 1의 Irrelevance 대 Relevance split에서 보인다. Bucket과 같은 skill의 held-out slice, 그리고 §5 untargeted-capability slice도 둔다. Bucket-driven fix의 cost가 나타나는 곳이다.

---

## 설정 recipe

Slice analysis에는 training hyperparameter가 없다. Slice report가 무엇을 resolve할 수 있는지를 정하는 evaluation protocol setting이 있다. 아래 행은 이 장이 의존하는 protocol value이며, 각각 출력한 locus에서 읽었다.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | Source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Tülu 3 8B | 8B | eval-gate | evaluation suite split | development suite + unseen suite, core skill별 pair; development 중 unseen score 미열람 | arXiv:2411.15124 §2.2, §7.2--7.3, Table 3 | 2026-09-15 검증 | §7.4.1이 skill별 resulting overfitting finding 보고(Table 31/32, Fig. 24) |
| Tülu 3 8B | 8B | RL | RLVR checkpoint-selection rule | training 100 step마다 평가; overall MATH와 IFEval 최선 checkpoint 선택 | arXiv:2411.15124 §6.4 | 2026-09-15 검증 | GSM8K 89.4/IFEval 84.8 run은 “other metric에서 더 나빠 overall average를 끌어내리는 경향” 언급 |
| GSM-Symbolic | n/a(benchmark) | eval-gate | template set당 instance | GSM8K test item의 template 100개; 각 100 example dataset 50개 | arXiv:2410.05229 §4, §3.1 | 2026-09-15 검증 | §4.1: resulting model별 score distribution/spread(Fig. 2, Table 1) |
| RULER | n/a(benchmark) | long-context | length당 task당 example | 4K/8K/16K/32K/64K/128K에서 500, native chat template + answer prefix, recall-based matching | arXiv:2404.06654 §4 | 2026-09-15 검증 | Table 3 claimed/effective-length column은 이 run에서 계산 |
| RULER | n/a(benchmark) | long-context | effective-length threshold | Llama2-7B 4K = 85.6보다 높은 average score | arXiv:2404.06654 §4, Table 3 | 2026-09-15 검증 | threshold choice ablation 미보고 |
| NoLiMa | n/a(benchmark) | long-context | context length당 test | haystack 5 × question-needle pair 58 × placement 26 = 7,540; length 250--32K | arXiv:2502.05167 §4.1 | 2026-09-15 검증 | §4.2: 100-chunk control에서 haystack filter 99.8% validate |
| NoLiMa | n/a(benchmark) | long-context | effective-length threshold | `0.85 × base score`보다 높게 score한 최대 test length, base = 250/500/1K의 max | arXiv:2502.05167 §4.3 | 2026-09-15 검증 | RULER fixed 85.6와 대비; 0.85 ablation 미보고 |
| HAL(Holistic Agent Leaderboard) | model 9개 | eval-gate | rollout budget/repetition | rollout 21,730, model 9 × benchmark 9, 약 $40,000; 대부분 CI 없는 single run | arXiv:2510.11977 Abstract, App. A3 item 1 | 2026-09-15 검증 | cost가 이유; variance ablation 미보고 |
| HAL(Holistic Agent Leaderboard) | n/a | eval-gate | trajectory tagger | transcript 2,184개(TAU-bench 제거 뒤 1,634)에 GPT-5 Medium rubric judge; human check 3개 precision .87/1.00/.94 | arXiv:2510.11977 App. A7.1--A7.2, Table A2 | 2026-09-15 검증 | Table A2 human-validated precision; recall 미측정 |
| BenchMIRT | model 100개 | eval-gate | item pruning | 가장 discriminative한 item 10% 또는 50% 유지 | allenai.org/blog/benchmirt, “Doing more with fewer questions” | 2026-09-15 검증 | 10%는 “generally preserved nearly the same picture”; 50%는 full benchmark와 더 가깝게 match하는 경우가 많음 |

**작은 범용 run의 출발점.** 여기 모든 값은 위 `검증` 행과 source 조건에서 왔다. Tülu 3 8B처럼 skill별 pair인 development/unseen suite 둘을 유지하고 development 중 never read한다(arXiv:2411.15124 §2.2). 8B의 development/unseen gap은 6.7--76.5점이므로 둘은 interchange할 수 없다(Table 31). Training decision에 쓸 benchmark마다 GSM-Symbolic처럼 item 최소 100개를 template로 만들고 template당 multiple instance를 만든다(template 100개, set 50개 × 100). Long-context claim에는 6개 RULER length에서 length·task당 item 500개를 실행하고 stated threshold 아래 effective length를 보고한다. Agent evaluation에는 scaffold를 고정하고 accuracy 옆 cost를 기록하며, run 간 count를 비교하기 전에 human-labeled sample에서 rubric tagger를 validate한다.

## 일반화 관점

**(a) 무엇이 범위를 넓히는가.** Development suite와 skill별 pair인 held-out suite를 두고 held-out column은 development 중 읽지 않는다([[tulu-3]] §2.2, §7.4). Test한 type보다 넓은 constraint type에서 훈련하면 in-distribution/out-of-distribution score가 모두 오른다([[ifbench]] Table 6: IFEval 81.1 → 92.2, IFBench 25.2 → 44.6). Narrow corpus만 training data라면 RL을 SFT보다 고르는 것도 넓힐 수 있다. Base 45.7에서 non-reasoning average는 RL 뒤 53.2, SFT 뒤 21.1이었다([[transferability-of-llm-reasoning]] Table 1). 제시한 mechanism은 KL-minimal solution 쪽 implicit bias다([[rls-razor]] §4--§5, **해석**).

**(b) 무엇이 narrowing/forgetting을 만드는가.** Development benchmark에서 selection: [[tulu-3]] §7.4.1은 precise instruction following, 어느 정도 knowledge recall/reasoning development evaluation에 pipeline이 overfit했다고 말한다. Verifiable reward 하나 최적화: [[ifbench]] §5에서 pure constraint reward 아래 AlpacaEval 2는 33.5 → 21.3, judge-rated quality는 7 → 6.4이다. Narrow domain SFT: [[transferability-of-llm-reasoning]] Table 1은 SFT(think) run에서 CoQA 10.0 → 1.7, HalluEval 35.7 → 2.3을 보인다. Target을 얻은 stage: [[tulu-3]] Table 6은 DPO→RLVR에서 Safety −1.7, TruthfulQA −1.1이다. Surface fitting: [[gsm-symbolic]] Table 1은 Gemma2-9b-it이 original 87.0, NoOp 22.3임을 보인다.

**(c) 이 stage에서 generality를 재는 법.** 알려진 limitation이 있는 measurement 네 개다. Skill별 development-minus-held-out gap([[tulu-3]] Table 31)은 genuine difficulty/format difference도 담으므로 overfitting upper bound다(§7.4.1). Canonical에서 distractor-added item까지 perturbation ladder([[gsm-symbolic]] Table 1)는 template화한 benchmark로 제한된다. Length/depth/hop/overlap long-context grid([[ruler]] Table 3, [[lost-in-the-middle]] App. G Table 6, [[nolima]] Table 3/6)는 effective-length number가 threshold에 의존한다(RULER fixed 85.6, NoLiMa `0.85 × base`). Stage 전반 untargeted-capability delta를 Transferability Index로 요약한다([[transferability-of-llm-reasoning]] §2.1). 이는 math group gain의 ratio이므로 math gain이 작으면 sign 해석이 정의되지 않는다.

## 흔한 실수와 발견 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| Aggregate만으로 결정 | Report가 checkpoint당 row 하나 | Slice score에서 aggregate 재계산; 불가능하면 slice를 저장하지 않은 것([[tulu-3]] Table 6 §1) |
| Development benchmark를 generality 근거로 취급 | Skill당 score 하나 | Paired held-out benchmark/gap 추가; IFEval만으로는 IF 58.1점 gap이 보이지 않음([[tulu-3]] Table 31) |
| Development 중 held-out suite를 읽음 | 모든 design decision에서 held-out/development score가 함께 움직임 | Decision마다 consult한 score 기록; [[tulu-3]] §2.2는 unseen set 미열람을 명시 |
| Long-context claim에 number 하나 | Grid 없는 “128K” label | length × depth × overlap grid score; [[nolima]] Table 6은 같은 length/model에서 98.5 대 25.9 |
| Threshold 없이 effective length 보고 | Paper 둘의 effective length를 직접 비교 | Rule 출력: fixed 85.6([[ruler]] §4) 대 `0.85 × base`([[nolima]] §4.3) |
| Scaffold를 넘어서 agent score 비교 | Report 간 model ranking 변경 | Scaffold 고정 또는 여러 개 평가; [[holistic-agent-leaderboard]] §4.1 item 6에서 scaffold가 rank reverse |
| Environment failure를 model failure로 채점 | 동일 run 간 설명 없는 score 하락 | Environment-barrier bucket 분리; silent endpoint switch/rate-limit failure는 [[holistic-agent-leaderboard]] App. A3 |
| Validate 안 한 tagger의 bucket count 신뢰 | Judge model 변경 시 bucket share 이동 | Human-labelled sample에서 tagger validate, `tagger_version` 기록; [[holistic-agent-leaderboard]] Table A2는 precision만 측정 |
| 다른 것을 재는 subset을 average | Benchmark subset끼리 다른 benchmark보다도 더 disagreement | Subset alignment 확인; BBQ는 general reasoning .85, safety −.06 correlate([[ai2-benchmirt]] blog table) |
| 50 slice에서 5% level regression flag | Every report에 2--3 regression이 생기고 재현 안 됨 | Multiple-comparison correction(ch-51); true null expected `50 × 0.05 = 2.5` |

## 이해 점검

1. Tülu 3 8B의 DPO→RLVR average는 +0.4, GSM8K는 +3.3이다. §1의 benchmark 12개 delta를 사용해 average threshold rule 하나가 GSM8K gain과 Safety loss를 모두 보이게 tuning될 수 없는 이유를 설명하라.
2. [[tulu-3]] Table 31은 같은 step에서 HumanEval 83.9 → 83.9, BigCodeBench 9.5 → 7.4를 보인다. 두 수와 양립하는 다른 설명 두 개와, 이를 분리할 추가 측정을 제시하라.
3. IFBench는 IFEval 81.1 → 92.2, IFBench 25.2 → 44.6을 올린다. IFEval에 unseen인 constraint 29개에서 훈련한 뒤 47.6점 gap이 지속되는 것이 training set size가 아니라 capability에 관한 근거인 이유를 설명하라.
4. GSM-Symbolic은 같은 question 100개를 re-instantiate해 set별 standard deviation 2--6점을 보고한다. 이것이 checkpoint 둘의 GSM8K 2점 차이를 읽는 법을 어떻게 바꾸는지, correction이 “GSM8K item을 더 쓰기”가 아닌 이유를 설명하라.
5. NoLiMa Table 6은 같은 model/haystack 32K에서 direct/one-hop/two-hop 98.5/56.2/25.9를 준다. Effective context length가 model 하나의 property가 아닌 이유와, 옆에 무엇을 밝혀야 하는지 설명하라.
6. RL's Razor는 *new* task에서 잰 KL divergence가 *prior* task forgetting을 예측한다고 한다. New-task distribution만에서 잰 predictor가 operationally 유용한 이유와 알 수 없는 것을 설명하라.
7. SciCode에서 tool-call failure 하나 이상은 failed run 97.7%, successful run 100%다. 이것이 bucket prevalence를 fix priority ranking으로 쓰는 것에 주는 함의와, 그 ranking을 지지하는 HAL 측정량을 설명하라.
8. BenchMIRT에서 BBQ는 general reasoning .85, safety axis −.06과 correlate한다. 이것이 safety-suite average 해석을 어떻게 바꾸는지, IRT model을 fit하지 않고 직접 만든 suite에서 같은 문제를 찾는 법을 설명하라.

## 연결

- **이전: [[ch-49]] — Judge Models: Bias, Calibration, and Judge-Specific Overfitting.** Reason-tagger도 judge이므로 position/length/self-preference bias가 bucket count에 적용된다. §6은 tagger precision 측정과 version 기록을 더한다.
- **다음: [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions.** 이 장의 모든 per-slice delta는 point estimate다. ch-51은 item-level bootstrap, paired comparison, clustered standard error, 여기 만든 slice의 multiple-comparison control을 공급한다.
- **[[ch-47]] — Evaluation Harness and Suite Design for General Capability.** 이 장의 모든 slice가 `GROUP BY`하는 item-level record를 만든다. Task mean은 나중에 slice할 수 없다.
- **[[ch-47a]] — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation.** §2/§3이 slice로 소비하는 perturbation/held-out set을 만든다.
- **[[ch-48]] — Contamination Detection and Its Effect on Reported Scores.** Contamination은 large canonical-minus-perturbed gap의 후보 설명 하나([[gsm-symbolic]] §4.1)이며, ch-48은 이를 surface fitting과 분리하는 test를 공급한다.
- **[[ch-51a]] — Evaluating Agent Generality and Reliability.** §7을 repeated-run reliability, held-out environment, agentic metric variance로 확장한다.
- **[[ch-53]] — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness.** 이 장의 report type 세 개를 end to end 구현한다.

## 출처

- [[tulu-3]] — 8B per-stage per-benchmark Table 6, RLVR-stage Table 23, development/unseen suite design(§2.2, §7.2--7.4)/skill별 Table 31, checkpoint-selection rule(§6.4). 주: library card “What RLVR buys”는 GSM8K “+5--10pp”라 하지만 Table 6/23은 8B에서 84.3 → 87.6이며 이 장은 table을 쓴다.
- [[ifbench]] — held-out verifiable constraint 58개, IF-RLVR 전후 IFEval/IFBench pair(Table 6), pure constraint reward의 general response quality cost(§5, Table 3).
- [[gsm-symbolic]] — template construction(§3.1), 50 × 100 generation protocol(§4), NoOp 포함 variant table(Table 1), original-versus-Symbolic gap의 contamination interpretation(§4.1). Excerpt: `excerpts/gsm-symbolic.md`(이 source의 library card 없음).
- [[ruler]] — model 17개 claimed/effective length(Table 3), weighted average 둘, fixed model generator별 error analysis(§5, Fig. 2).
- [[lost-in-the-middle]] — 20 document evidence position별 accuracy(App. G Table 6), closed-book/oracle reference(Table 1), window-extension comparison(§2.3).
- [[nolima]] — question-needle overlap(Table 1), `0.85 × base` threshold의 claimed/effective length(Table 3), fixed-length direct/one-hop/two-hop comparison(Table 6), haystack-filter validation(§4.2).
- [[transferability-of-llm-reasoning]] — Transferability Index 정의(§2.1), 3 task group에서 Qwen3-14B SFT/RL controlled comparison(Table 1).
- [[rls-razor]] — new-task KL과 prior-task loss forgetting law, quadratic fit(`R² = 0.96` controlled, `0.71` LLM), oracle-SFT control(§3--§4). Excerpt: `excerpts/rls-razor.md`(library card 없음).
- [[ai2-benchmirt]] — benchmark 16개/item 34,301개의 2D IRT decomposition, benchmark별 correlation(BBQ/WMDP/XSTest), subset finding, item-pruning result.
- [[bfcl]] — confusion-cell definition AST matching(§4.1, App. H), Irrelevance/Relevance category divergence(Table 1), mode별 decoding issue(§5.1), LLM-judged multi-turn root-cause distribution(§5.4.2).
- [[holistic-agent-leaderboard]] — trajectory rubric 6 category/prevalence(§4.2), validated precision/unmeasured recall(Table A2), scaffold-model interaction(§4.1 item 6--7), environment-failure case(App. A3), TAU-bench few-shot leak(App. A5).
