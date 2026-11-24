<!-- chapter: ch-26
     track: synthetic
     kind: content
     title: Tool and Function-Calling Data
     deps: [ch-25]
     sources: [[toolformer]], [[toolllm]], [[apigen]], [[apigen-mt]], [[xlam]], [[toolace]], [[hammer]], [[bfcl]], [[bfcl-v4-format-sensitivity]], [[tau-bench]], [[tau2-bench]], [[toucan-mcp]], [[toolrl]], [[agenttuning]], [[reasoning-trap-tool-hallucination]], [[kimi-k2]], [[gorilla]], [[nexusraven]], [[granite-function-calling]]
     figures: figures/tool-pipeline.html
     revised: 2026-09 (generality revision)
-->

# Chapter 26 — 도구 및 함수 호출 데이터

> **핵심 통찰.** 함수 호출 데이터는 도구 스키마(실제 도구, 실행 가능한 도구, 또는 LLM이 합성한 도구)를 생성한 질의 및 호출과 짝지은 뒤, 형식·실행·LLM 판정기로 필터링하여 만든다. APIGen에서 생성기 통과율은 34.42%부터 84.15%까지였고, 거부된 샘플을 학습에 다시 넣자 7B 학생 모델의 BFCL 전체 점수는 5.94점, 1B 학생 모델의 점수는 12.17점 낮아졌다([[apigen]] Table 1, Fig. 5). 필터링만으로 보지 못한 도구에 대한 일반화가 생기지는 않는다. 한 데이터 계열로 학습한 모델은 벤치마크마다 일관되지 않은 점수를 냈으며, Hammer의 함수명 마스킹 및 무관성 데이터를 사용해 학습한 DeepSeek-Coder-7B 모델은 BFCL 밖의 다섯 벤치마크에서 평균 F1 74.94를 기록해, 같은 기반 계열의 xLAM-7B-fc의 67.65보다 높았다([[hammer]] Tables 1, 5). 같은 4K 예제에서 규칙 기반 호출 보상으로 GRPO를 적용한 Qwen2.5-7B-Instruct는 BFCL V3 전체 58.38에 도달했지만, R1 증류 trace로 SFT한 모델은 36.53이었다([[toolrl]] Table 1; 각각 한 번의 run).
>
> **지침.** 합성 호출이 APIGen의 DeepSeek-V2-Chat이 보인 84%보다 낮은 통과율의 생성기에서 왔다면, SFT 전에 실행 및 의미 검사를 수행하라. add-back ablation에서 거부 데이터의 비용이 BFCL 4–12점이었기 때문이다([[apigen]] Fig. 5). 그렇지 않은 경우에도 검사를 유지하되, 더 작고 측정되지 않은 효과를 예상하라. 모델이 학습 때 보지 못한 도구를 사용해야 한다면, 보지 못한 도구 및 보지 못한 범주 split에서 평가하고, 학습 데이터와 다른 그룹이 만든 벤치마크도 적어도 하나 포함하라. 또한 SFT 데이터 일부에서는 함수와 매개변수 이름을 마스킹하라. 이름 의존은 Hammer가 측정한 실패이기 때문이다([[hammer]] §3.2, Table 3). 중단(abstention) 목표(무관한 도구, 누락된 인수)를 포함하고, 무관성 정확도와 호출 정확도를 모두 측정해 그 비율을 정하라. 비도구 사용 대화를 제거했을 때 ToolACE 무관성 정확도는 86.42에서 6.99로 떨어졌고([[toolace]] Table 7), Hammer에서는 두 지표가 반대 방향으로 움직이며 전체 최고점이 약 10% 비율 부근에 있었기 때문이다([[hammer]] Fig. 6). 목표 사용이 멀티턴 또는 상태 보존형이면 대화를 시뮬레이션하기 전에 행동 시퀀스를 검증하고 k > 1의 pass^k를 보고하라. τ-retail에서 gpt-4o의 pass^1은 60%보다 높았지만 pass^8에서는 25% 아래로 떨어졌기 때문이다([[apigen-mt]] §4.1; [[tau-bench]] Fig. 4). 추론 RL 또는 추론 증류가 뒤따른다면, 누락 도구와 방해 도구를 넣어 중단을 다시 측정하라. GRPO는 BFCL 멀티턴을 올리면서 no-tool 환각률을 34.8%에서 90.2%로 올렸기 때문이다([[reasoning-trap-tool-hallucination]] Table 3).

## 학습한 버전에 대한 정정

1. 지침의 “실행을 제거하면 약 11 BFCL점을 잃는다”와 표의 “Full 88.24 / remove semantic 82.2 / remove execution 77.3 / remove format 70.1” → APIGen에는 단계별 제거 표도 format arm도 없다. Fig. 5는 거부 샘플을 다시 추가한다. xLAM-7B (FC)는 −4.06(+의미 실패) 및 −5.94(+실행 실패), xLAM-1B (FC)는 −9.59 및 −12.17이다([[apigen]], Fig. 5, §5.2).
2. “xLAM-7B (Mistral base)가 BFCL-V1 88.24%, <13B 중 1위” → xLAM-7B (FC)는 DeepSeek-Coder-7B-instruct-v1.5에서 fine-tune했고, 2024-06-15 리더보드에서 85.65로 6위였다. xLAM 논문의 BFCL v2 표(2024-09-03)에서는 xLAM-7b-fc-r이 80.18, 17위다([[apigen]] §5.1, Table 2; [[xlam]] Table 1, Table 5).
3. APIGen의 “약 40% 거부” 및 그림의 단계별 통과율 75% / 90% / 88% → 통과율은 생성기에 따라 다르다. DeepSeek-Coder-33B-Inst는 34.42%, DeepSeek-V2-Chat은 84.15%이며 단계별 백분율은 보고되지 않았다([[apigen]] Table 1).
4. “Teacher: DeepSeek-Coder-V2-Instruct가 주, GPT-4 ablation”, “GPT-4 judge”, “DeepSeek-Coder-V2와 GPT-4가 약 2% 차이” → 생성기는 DeepSeek-V2-Chat, DeepSeek-Coder-33B-Inst, Mixtral-8x22B-Inst, Mixtral-8x7B-Inst이다. 의미 검사 모델의 이름은 없고 GPT-4 비교도 없다([[apigen]] §3.2, §4.2).
5. “Python sandbox, 5초 timeout”, “(query, call) MinHash”, “simple 약 40% / multiple 약 25% / parallel 약 20% / parallel-multiple 약 15%”, “API별 약 16회” → timeout 값, 중복 제거 방법, 유형 비율, API별 횟수는 보고되지 않았다([[apigen]] Verification).
6. “3,673개의 실행 가능 API(… Salesforce가 endpoint를 실행하거나 Python mock을 작성)” → 접근성 테스트를 통과한 ToolBench REST API 3,539개와 Python 함수 134개다([[apigen]] §4.1).
7. “xLAM-2 단계식 recipe — APIGen-60k SFT → APIGen-MT-5k SFT → 선택적 DPO (β=0.1)” 및 §8의 단계별 learning rate, epoch, sequence length → xLAM-2-fc-r 모델은 APIGen-MT trajectory, APIGen 데이터, 기타 agentic 데이터 혼합물로 최대 3 epoch의 한 SFT 단계에서 학습한다. DPO 단계, learning rate, batch, sequence length는 보고되지 않았다([[apigen-mt]] §5.1).
8. “약 10M position을 scan해 약 2% 생존” 및 “Table 2는 도구마다 수십만 candidate 중 수천 annotation만 유지” → candidate 수는 보고되지 않았다. τ_f = 1.0에서 Table 2는 QA 18,526개, Wikipedia search 60,974개, calculator 994개, calendar 20,587개, translation 1,034개를 유지한다([[toolformer]] Table 2).
9. `obs.is_error`일 때만 철회하는 DFS-DT pseudocode와 “철회 없이는 DFS-DT가 ReACT로 퇴화한다”는 appendix 주장 → DFSDT는 모델이 `give_up_and_restart`를 호출하면 node를 포기하고 이후 child를 다양화한다. appendix는 모델이 행동을 철회하지 않을 경우 DFSDT가 ReACT처럼 행동하고 비용도 같다고 말한다([[toolllm]] App. A.4 and prompts).
10. DFSDT Table 3의 “ReACT 42.2 / 30.0 / 21.7, ReACT@N 47.7 / 34.3 / 26.0, DFS-DT 57.3 / 48.2 / 43.2” → I1 / I2 / I3에서 ReACT는 37.8 / 40.6 / 27.6, ReACT@N은 49.4 / 49.4 / 34.6, DFSDT는 58.0 / 70.6 / 62.8이다([[toolllm]] Table 3).
11. “ToolACE … 가능한 곳에서 execute 유지”, “Rule-based (schema + param + type + mock이 있을 때 execution)” → ToolACE의 규칙 계층은 호출을 실행하지 않고 검사한다(도구 목록에 이름 존재, 필수 매개변수, 정규표현식 형식). 도구 응답은 LLM agent가 시뮬레이션한다([[toolace]] §2.2.1, §2.3).
12. “실제 API 3K seed에서 LLM이 세 operator로 mutation” → API synthesis는 API 관련 pretraining 문서에서 API context tree를 만들고 API 정의를 적응·진화시킨다. seed 수는 명시되지 않았다. API 26,507개와 domain 390개는 맞다([[toolace]] §2.1, Table 1).
13. “complexity evaluator가 dialog를 5 난이도로 분류 … 30/25/20/15/10% … BFCL에 맞게 수동 조정” → complexity는 해당 샘플에서 학습 중인 모델의 loss이고, 이전 set의 bound를 사용한다. user agent에게 더 어렵거나 쉬운 질의를 요청하며 ToolACE는 single, parallel, dependent, non-tool-use의 네 dialog 유형을 생성한다([[toolace]] §2.2, Eq. 1).
14. “GPT-4 judge, 3-way verdict … 약 40% 거부, 최종 11,300 dialogs” → model layer는 환각 탐지, 일관성 검증, 도구 응답 검사를 위해 별도 LLM agent를 쓴다. 거부율과 최종 dialog 수는 보고되지 않았다([[toolace]] §2.3).
15. “ToolACE-8B (Llama-3.1-8B base): BFCL-V1 91.41%, xLAM-7B의 88.24%를 이김” → ToolACE-8B는 LLaMA-3.1-8B-Instruct에 LoRA를 적용했다. 09/20/2024 BFCL-v3 리더보드에서 전체 59.22(3위)이며 xLAM-7b-fc-r은 51.45다. 91.41은 ablation 그림의 non-live category 전체 점수다([[toolace]] §3.1, Table 2, Fig. 3).
16. “TSS 제거 −4.3%, complexity controller 제거 −3.1%, model-judge 제거 −5.2%, rule-check 제거 −2.8%” → Fig. 3(non-live overall)은 verification 없음 89.59, rules만 90.47, 둘 다 91.41이다. complexity subset은 90.47 / 90.71 / 89.65(Fig. 4), diversity subset은 88.18 / 88.35 / 88.41(Fig. 5)이며 API-synthesis 제거 arm은 없다([[toolace]] §3.3).
17. BFCL “일곱 category: Simple · Multiple · Parallel · Parallel-Multiple · Relevance-Detection · Multi-Turn · Multi-Step” → single-turn은 Simple, Multiple, Parallel, Parallel Multiple, Irrelevance, Relevance다. multi-turn은 Base, Missing Parameters, Missing Functions, Long Context이고 agentic은 Web Search, Memory, SQL이다. Multi-Step category는 없다([[bfcl]] §3, App. C.2).
18. AST matcher “kwargs 정렬 … literal canonicalization(1.0 ≡ 1, \"red\" ≡ 'red')” → 각 parameter 값은 가능한 답의 set 안에 있어야 한다. Python은 float에 int를 받지만 역은 안 되고 Java·JavaScript는 float literal을 요구한다. string은 case-insensitive로 비교하고 whitespace 및 나열한 punctuation을 제거한다([[bfcl]] §4.1, App. H).
19. version 표의 “V2 Live +1,500 real user queries” 및 “V4 Agentic … SWE-Gym agent data” → crowd-sourced category에는 2,251 entries가 있다. BFCL V4는 web search, memory, format sensitivity를 추가한다([[bfcl]] §1; [[bfcl-v4-format-sensitivity]]).
20. “pass^k. V3+는 k independent trial 모두 성공을 요구” → pass^k는 τ-bench에서 정의한다. BFCL multi-turn scoring은 한 run의 매 turn에서 state 및 response 검사가 통과해야 한다([[tau-bench]] §3; [[bfcl]] §4.4).
21. “xLAM-2-70B τ-bench: pass^1 = 56.2%, pass^4 = 39.4%” → 56.2는 retail과 airline의 평균 pass@1이다. pass^k 값은 Fig. 6에 그려져 있으며 39.4는 본문이나 표에 없다([[apigen-mt]] Table 2, Fig. 6).
22. “보고된 task-collection 성공률 약 70%, 평균 약 12 turn” 및 “약 12 messages” → Phase 1 blueprint 성공은 agentic feedback 포함 시 70%, 없으면 28%다. Phase 2 trajectory 성공은 67%다. trajectory는 평균 7 tool call과 6 user turn(1~29 turn)이며, 본문은 gpt-4o가 task당 평균 12 turn을 쓴다고 별도로 말한다([[apigen-mt]] Fig. 4, §4.3).
23. “semantic review (committee aggregation + refinement)” → committee majority vote는 Stage 2(alignment validation)다. Stage 3는 score threshold를 적용하고 regeneration을 위한 feedback을 반환한다([[apigen-mt]] §4.1.2).
24. “sample당 약 $1 rollout” → 비용은 보고되지 않았다([[apigen-mt]] Verification).
25. “xLAM-2-8B가 5K trajectory만으로 69.25% 도달” → 69.25는 BFCL v3 multi-turn 점수지만, 학습은 비공개 비율의 APIGen-MT trajectory·APIGen 데이터·기타 agentic 데이터를 섞었다([[apigen-mt]] Table 1, §5.1).
26. “frontier model도 무관한 질의 약 10%에서 도구를 호출” → 출처가 없다. BFCL irrelevance accuracy는 모델과 mode에 따라 달라지며, 예를 들어 GPT-4-turbo-2024-04-09는 FC mode 83.8, prompting mode 35.6이다([[bfcl]] Table 1).
27. Hammer의 “func_[a-z0-9]{6} … 학습 데이터 30%”, “무관성 sample 약 30%”, “30%가 optimum; 50%는 recall 파괴”, “BFCL relevance +13점; Hammer-7B 약 90%, GPT-4와 같음” → Hammer는 함수명·parameter명·default 값을 마스킹한다. 공개 script는 67,500개를 세 번 복제하고 shuffle한 뒤 각 사본을 2/3 확률로 마스킹하며, 함수명은 5–15자의 random name, parameter명은 4–10자가 된다. irrelevance set은 7,500개(67,500개의 약 11%)다. masking-ratio ablation은 optimum을 말하지 않고, irrelevance-share optimum은 약 10%다. Hammer-7B BFCL irrelevance는 72.87, xLAM-7B-fc는 79.76이다([[hammer]] §4, §5.5–5.6, Table 2; repository `train/data_processing.py`).
28. NexusRaven의 “simple 60% / parallel 20% / nested 20%” 및 “nested 제거 → −15점” → NexusRaven-V2에는 training mix나 ablation이 발표되지 않았다([[nexusraven]] Verification).
29. Gorilla의 “80% noisy retrieval + 20% oracle” 및 “hallucination rate: retriever-aware 11% vs retriever 없는 GPT-4 40%” → 그런 mix는 기술되지 않았다. 10.95%는 HuggingFace에서 Gorilla zero-shot(검색기 없음), 37.16%는 GPT-4 zero-shot이다([[gorilla]] Table 1).
30. Granite의 “seven-capability taxonomy (Nested / Parallel / Multiple / Multi-turn / Relevance / Sequencing / Slot-Filling)”, source-mix 백분율 및 source별 제거 delta → Granite-20B-FunctionCalling의 일곱 training task는 Nested Function Calling, Function Chaining, Parallel Functions, Function Name Detection, Parameter-Value Pair Detection, Next-Best Function, Response Generation이다. mix 백분율과 delta는 논문에 없다([[granite-function-calling]], arXiv:2407.00121 Abstract, Table 1).
31. “Glaive의 `<functioncall>` XML로 OpenAI JSON 대신 학습하면 5–10점 손실” 및 “전체 BFCL 92%지만 relevance 65%인 모델은 wild에서 35% 도구 환각 … #1 production failure mode” → 두 수치 모두 출처가 없다. BFCL V4는 return-format 효과가 모델마다 다르고 작은 모델에서 더 크다고 보고한다([[bfcl-v4-format-sensitivity]] §4.1).
32. §8의 “바로 쓰는 recipe” mix(25% APIGen / 20% ToolBench / …), “예상 acceptance 약 55–60%”, “MinHash threshold 0.9” → 이전 chapter를 위해 구성한 것이며 어떤 source에서도 가져오지 않았다. 아래 Recipe section은 검증된 값만 열거한다.
33. “[[ch-28]] … Multi-turn tool use는 더 강한 long-context signal 중 하나” → 이 chapter의 어떤 source도 이를 측정하지 않고 ch-28도 tool data를 다루지 않는다.
34. “누가 eval taxonomy를 정하든 data-generation taxonomy도 정한다”를 finding으로 제시한 것과 “ToolACE distribution은 BFCL 비율에 맞춰 수동 조정한다” → APIGen은 네 query style을 BFCL category에서 가져왔다고 말한다([[apigen]] §3.3). ToolACE는 난이도를 BFCL 비율이 아니라 learner에 맞춰 보정한다([[toolace]] §2.2.2). 더 넓은 주장은 측정 없는 해석이다.

이전 버전의 section map: §1 → §2; §2 → §3; §3 → §4; §4 → §7; §5 → §5; §6 → §8; §7 Hammer → §6(NexusRaven, Gorilla, Granite paragraph는 제거; corrections 28–30 참조); §8 → Recipe. §1, §9, Negative samples section은 새로 추가됐다.

## 범용 모델에 이 장이 중요한 이유

함수 호출은 주어진 목록에서 도구 하나를 이름으로 선택하고 인수를 채우는 구조화된 출력이다. 범용 모델은 학습하지 않은 도구, 보지 못했을 수 있는 형식으로 설명된 도구를 받으며, 어느 도구도 적용되지 않을 때를 판단해야 한다. 도구 호출 데이터는 SFT(이 장), agentic mid-training(ch-32d), 멀티턴 agentic RL의 action space(ch-45b)에서 파이프라인에 들어간다. 그 결과 측정 가능한 위험이 세 가지다. 첫째, 모델은 도구 설명을 읽는 대신 naming style이나 output template 같은 하나의 data family 특징을 배울 수 있다. xLAM-7B-fc는 Hammer Table 1의 세 모델 중 Nexus Raven에서 최저인 57.5를 받았지만, BFCL 79.41 및 API-Bank 72.45에서는 세 모델 중 최고였다([[hammer]] Table 1). 둘째, 호출 정확도를 높이는 학습이 중단 능력을 낮출 수 있다. Qwen2.5-7B-Instruct에서 추론을 포함한 GRPO는 BFCL 멀티턴 정확도를 13.6에서 23.5로 올리는 동시에 no-tool-available 환각률을 34.8%에서 90.2%로 올렸다([[reasoning-trap-tool-hallucination]] Table 3). 셋째, 벤치마크 점수는 test distribution을 반영할 수 있다. APIGen-MT의 training domain은 τ-bench 평가 domain이며([[apigen-mt]] §4.3), BFCL의 perplexity 검사는 xLAM-7B를 flag했다([[bfcl]] §5.5). 이 장은 도구 호출 데이터를 만들고 필터링하는 방법, 그리고 결과가 전이하는지 측정하는 방법을 다룬다.

## §1 용어

- **도구 스키마(function definition)**: 보통 JSON Schema로 된 이름, 자연어 설명, type이 있는 parameter. **후보 집합 F**: 하나의 prompt에 주어지는 스키마들([[bfcl]] §3.1).
- **관련성 / 무관련성(relevance / irrelevance)**: 관련성 item은 적어도 한 번의 호출을 기대한다. 무관련성 item은 호출이 없거나 필요한 function 또는 argument가 빠졌을 때의 clarification을 기대한다([[bfcl]] App. B).
- **trajectory**: user turn, 모델 추론, 호출, 도구 응답, 최종 답변이 교차하는 시퀀스.
- **FC mode 대 prompting mode**: API의 tools field로 전달하는 도구 대, 필수 text format과 함께 system prompt에 기술하는 도구([[bfcl]] §5.1).
- **held-out tool split**: 본 도구의 보지 못한 instruction(Inst.), 본 category의 보지 못한 tool(Tool), 보지 못한 category의 보지 못한 tool(Cat.)([[toolllm]] §3.2).
- **MCP(Model Context Protocol)**: application이 server에 호스팅된 도구를 발견하고 호출하는 standard interface([[toucan-mcp]] §1).
- **pass^k**: task의 독립 시행 k번이 모두 성공할 확률을 task별로 평균한 값([[tau-bench]] §3); §8에 worked example로 정의한다.

## §2 Toolformer: loss filter를 사용한 일반 텍스트의 annotation

**정의.** Toolformer는 label 없는 텍스트에 API call을 삽입하고, 그 결과가 뒤따르는 token을 더 쉽게 예측하게 할 때에만 그 call을 유지한다([[toolformer]] §2).

**문제.** label이 없으면 어느 위치에서 call이 도움이 되는지 알려 주는 signal이 없다. parse되고 실행되는 call도 텍스트에 아무것도 더하지 않을 수 있다.

**메커니즘.**
1. 도구별로 몇 개의 demonstration을 넣어 GPT-J (6.7B)를 prompt한다. `p_M(<API>)`가 τ_s = 0.05를 넘는 position을 유지하며, text 하나당 최대 k = 5개이고 position 하나당 최대 m = 5개의 call을 sample한다(App. A).
2. 다섯 도구(QA, Wikipedia search, calculator, translation, calendar) 중 하나에 대해 각 call을 실행한다.
3. 다음 token의 weighted loss를 비교하고 threshold를 통과한 call을 유지한다.
4. 허용된 call을 삽입한 원본 text로 fine-tune한다.

**수식** (§2, §4.1):

```
L_i(z) = − Σ_{j=i..n} w_{j−i} · log p_M(x_j | z, x_{1:j−1})
keep call c_i  if  min( L_i(ε), L_i(e(c_i, ε)) ) − L_i(e(c_i, r_i)) ≥ τ_f
w_t = w̃_t / Σ_s w̃_s,   w̃_t = max(0, 1 − 0.2·t)
```

`x_j`는 position i 뒤의 text token이고, `z`는 모델에 주어진 prefix이며, `e(c, r)`은 call text 뒤에 result r이 붙은 것이다. `ε`는 빈 sequence이고, `τ_f`는 filtering threshold다(기본 1.0, calculator와 translation에는 0.5). 결과가 없는 call인 `L_i(e(c_i, ε))`와 비교하면, call text 자체만 예측 가능해서 call이 유지되는 일을 막는다.

**계산 예시.** t = 0…4의 weight는 normalization 전에는 1, 0.8, 0.6, 0.4, 0.2다. 따라서 3.0으로 나눈 뒤에는 1/3, 4/15, 1/5, 2/15, 1/15이다. call이 없을 때 다섯 next-token loss가 (3.0, 2.0, 2.0, 1.0, 1.0)이고, call text만 있을 때 (2.8, 2.0, 2.0, 1.0, 1.0), call과 result가 있을 때 (0.5, 0.5, 1.0, 1.0, 1.0)이라고 하자. weighted loss는 각각 2.133, 2.067, 0.700이다. 앞 두 값의 최솟값은 2.067이고, 2.067 − 0.700 = 1.367 ≥ 1.0이므로 call을 유지한다. τ_f = 2.0이면 버린다.

**근거.** Table 2는 threshold가 data size에 미치는 영향을 보인다. QA example은 τ_f = 0.5에서 51,987개, 1.0에서 18,526개, 2.0에서 5,135개로 감소한다. ASDiv / SVAMP / MAWPS에서 Toolformer의 점수는 40.4 / 29.4 / 44.0이고, GPT-J는 7.5 / 5.2 / 9.9, GPT-3 175B는 14.0 / 10.0 / 19.8이다(Table 4). call을 비활성화했을 때 language-modeling perplexity는 변하지 않는다. 같은 text를 call 없이 fine-tune한 GPT-J와 Toolformer 모두 WikiText / CCNet에서 10.3 / 10.5다(Table 8). **결과(단일 연구).**

**조건 및 한계.** 고정된 다섯 text-to-text 도구, 단일 local call이며 multi-step 사용은 없고, 학습에 없던 도구에서는 평가하지 않았다. threshold는 downstream ablation이 아니라 충분한 example을 남기기 위해 도구별로 선택했다(§4.1).

**범용 모델에 대한 함의.** Toolformer는 목표 성능 향상 옆에서 retention(도구를 비활성화한 perplexity)을 측정한 이른 사례다. 그 짝지음은 도구 데이터 run마다 보고해야 하는 최소 항목이다.

## §3 ToolLLM: 실제 API, 탐색 기반 trajectory, 보지 못한 도구 split

**정의.** ToolBench는 49개 category의 RapidAPI REST API 16,464개를 `gpt-3.5-turbo-16k`가 생성한 instruction 및 solution path와 짝짓는다([[toolllm]] §2). 공개 데이터에는 instruction–path pair 126,486개와 실제 API call 469,585개가 있다(Table 1).

**문제.** 선형 ReACT rollout은 multi-tool instruction에서 실패한다. 또한 통과한 annotation만 training data가 되므로 annotation yield가 data cost를 정한다(§3.1).

**메커니즘.**
1. 한 도구(I1), 한 category의 여러 도구(I2), 또는 한 collection의 여러 도구(I3)를 sample하고 teacher가 instruction을 쓰게 한다.
2. DFSDT로 solution을 search한다. 이는 thought–action step의 depth-first tree이며, 모델은 `give_up_and_restart`로 branch를 끝낼 수 있고 이후 child는 “diversity” prompt에서 이전 child를 본다(App. A.4 and prompts).
3. 실제 API에 대해 call을 실행하고 1,024 token을 넘는 response는 압축한다(App. A.2).
4. human annotator와 87.1%의 pass-rate agreement를 보인 ChatGPT evaluator인 ToolEval이 통과로 판정한 path를 유지한다(§3.1).

**근거(yield).** API budget을 맞췄을 때 I1 / I2 / I3의 pass rate는 ReACT에서 37.8 / 40.6 / 27.6, ReACT@N에서 49.4 / 49.4 / 34.6, DFSDT에서 58.0 / 70.6 / 62.8이다(Table 3). 차이는 I3에서 가장 크다.

**근거(held-out tools).** DFSDT를 사용한 ToolLLaMA-2-7B는 57.0(I1-Inst), 61.0(I1-Tool), 62.0(I1-Cat), 77.0(I2-Inst), 77.0(I2-Cat), 66.0(I3-Inst)을 통과한다(Table 4). 이 split에서 보지 못한 category의 도구는 본 도구에 대한 보지 못한 instruction보다 더 낮은 점수를 받지 않았다. ToolLLaMA가 학습하지 않은 APIBench에서, retriever를 사용한 HuggingFace / TorchHub / TensorHub AST accuracy는 16.77 / 51.16 / 40.59이다. Gorilla-ZS + BM25는 10.51 / 44.62 / 34.31이고, retrieval로 학습한 Gorilla variant인 Gorilla-RS + BM25는 15.71 / 50.00 / 41.90이다(Table 5). ToolLLaMA는 세 hub 중 두 곳에서 Gorilla-RS보다 높고 TensorHub에서는 낮다. **결과(단일 연구).**

**조건 및 한계.** retriever를 쓰지 않는 한 test instruction에는 oracle API가 주어진다. pass rate는 LLM judge에서 나온다. BFCL 저자는 ToolBench가 큰 분산을 갖는 RapidAPI response에 의존한다고 기술한다([[bfcl]] §2).

**함의.** Inst / Tool / Cat split은 이 장 제목의 질문을 직접 시험한다. in-distribution 점수만 보고하는 도구 데이터 논문은 그 질문에 답하지 못한다.

## §4 APIGen: 세 단계 검증과 ablation이 보이는 것

**정의.** APIGen은 sample한 API에서 single-turn query–call pair를 생성하고, 형식·실행·의미 검사를 순서대로 통과할 때에만 pair를 허용한다([[apigen]] §3).

**문제.** 생성된 call은 존재하지 않는 function 또는 argument를 사용하거나, runtime에서 실패하거나, 실행되더라도 query에 답하지 못한다. 측정 가능한 질문은 이런 데이터가 학생 모델의 점수를 얼마나 낮추는가이다.

**메커니즘.**
1. 3,673개 API(REST API 3,539개 + Python 함수 134개), seed example, 모호하거나 철자가 틀린 request를 포함할 수 있는 prompt template에서 API를 sample한다(§3.3, §4.1).
2. 형식 검사기: `"query"`와 `"answer"`가 있는 JSON인지, 주어진 API에 존재하는 function과 argument만 쓰는지 검사한다(§3.2).
3. 실행 검사기: Python function을 subprocess에서 실행하거나 REST API를 호출하고, type error, invalid parameter, runtime error, timeout, missing argument를 제거한다(§3.2).
4. 의미 검사기: 두 번째 LLM이 function, query, call, execution result를 읽고 pass yes/no를 반환한다(App. B.2).
5. 검증된 sample을 seed pool에 추가한다(§3.1).

**계산 예시.** 각 generator에는 40,000개 sample을 요청했다(§4.2). DeepSeek-Coder-33B-Inst의 경우 13,769개가 검증되었고, 4,311개는 format, 15,496개는 execution, 6,424개는 semantic check에서 실패했다(Table 1). 통과율은 13,769 / 40,000 = 34.42%이고, execution은 실패 26,231개 중 15,496 / 26,231 = 59.1%를 차지한다(도출). DeepSeek-V2-Chat에서는 33,659개(84.15%)가 통과하고 execution은 실패 중 3,359 / 6,341 = 53.0%를 차지한다(도출). Mixtral-8x7B-Inst 행의 합은 40,000이 아니라 39,000이지만, 인쇄된 통과율 38.46%는 15,385 / 40,000과 같다(source inconsistency). [figures/tool-pipeline.html](figures/tool-pipeline.html)의 Panel A는 네 generator 모두의 count와 add-back delta를 보인다.

**근거.** 이 데이터로 DeepSeek-Coder-7B-instruct-v1.5에서 fine-tune한 xLAM-7B (FC)는 BFCL(2024-06-15) 전체 85.65로 6위다. xLAM-1B (FC)는 74.41로 24위이며 GPT-3.5-Turbo-0125 (FC)의 63.88보다 높다(Table 2). stage-3 failure를 다시 추가하면 BFCL 전체는 4.06(7B) 및 9.59(1B) 낮아지고, stage-2 failure를 다시 추가하면 5.94 및 12.17 낮아진다(Fig. 5). **결과(단일 연구)**이며 arm당 run은 한 번이다.

**조건 및 한계.** 평가는 BFCL뿐이며 held-out-API split이나 비함수호출 benchmark는 보고되지 않았다(§5). 논문은 execution add-back arm에 semantic failure도 포함하는지 말하지 않는다. xLAM 논문은 이후 xLAM-7b-fc-r을 합성 function-calling 데이터 50%와 기타 task 50%로 학습한다([[xlam]] §3.5). BFCL의 perplexity check는 xLAM-7B의 perplexity가 curated data의 3.67에서 crowd-sourced query의 5.09로 상승했음을 발견했고, BFCL 저자는 이를 static test distribution에 맞춘 tuning으로 해석했다([[bfcl]] §5.5, Interpretation).

**함의.** 거부 sample에는 음의 한계 가치가 있고, 더 작은 학생 모델이 더 크게 손상된다. 이 ablation은 어느 stage가 가장 중요한지 보이지 않으며 다른 benchmark로의 transfer도 시험하지 않는다.

## §5 ToolACE: 합성 API 다양성과 learner에 맞춘 난이도

**정의.** ToolACE는 API definition을 합성하고, 세 LLM agent(user, assistant, tool)로 dialog를 생성하며, 학습할 모델의 loss를 사용하여 query 난이도를 조정하고, rule layer와 model layer로 필터링한다([[toolace]] §2).

**문제.** 실행 가능한 API는 coverage를 제한하고, 주어진 learner에게 너무 쉽거나 너무 어려운 데이터는 거의 기여하지 않는다. 이 논문은 API diversity, complexity, verification을 따로 시험한다.

**메커니즘.**
1. API 관련 pretraining document에서 API context tree를 만들고, subtree를 sample하며 diversity indicator로 API definition을 진화시켜 390 domain의 API 26,507개에 도달한다(§2.1, Table 1).
2. single, parallel, dependent, non-tool-use dialog를 생성한다. 각 assistant decision을 여러 번 sample하여 일관된 것을 유지하고, tool agent로 tool response를 시뮬레이션한다(§2.2.1).
3. learner의 loss로 complexity를 점수화하고 user agent에게 더 어렵거나 더 쉬운 query를 쓰도록 지시한다(§2.2.2–2.2.3).
4. 규칙 계층은 call을 실행하지 않고 tool list에 이름이 있는지, required parameter가 있는지, regular-expression format을 검사한다. 모델 계층은 hallucinated argument value, query와의 일관성, API definition과 tool-response의 일치를 검사한다(§2.3).

**수식** (Eq. 1):

```
H_M(x, y) = −(1/n_y) · Σ_{i=1..n_y} log p(t_i | x, t_1, …, t_{i−1})
```

`M`은 학습할 모델이고, `x`는 input query, `y = [t_1 … t_{n_y}]`는 response token, `p`는 모델의 next-token probability다. 값이 높을수록 sample은 M에게 더 어렵다. lower bound는 M이 이미 정답을 맞히는 sample의 loss이고, upper bound는 fine-tuning 뒤에도 높게 남는 sample의 loss다.

**계산 예시.** 세 token response의 probability가 0.5, 0.25, 0.125이면 loss는 0.693, 1.386, 2.079 nat이므로 H = 4.159 / 3 = 1.386이다. 이전 set이 0.3(lower), 1.2(upper)의 bound를 주면 이 sample은 upper bound 위이므로 user agent에게 query를 단순하게 하라고 지시한다.

**근거.** BFCL-v3 leaderboard(09/20/2024)에서 ToolACE-8B는 전체 59.22(3위), relevance 85.37, irrelevance 83.81이다(Table 2). non-live category에서 verification 없음은 전체 89.59, rule layer만은 90.47, 두 layer 모두는 91.41이다(Fig. 3). complexity subset easy / medium / hard는 90.47 / 90.71 / 89.65(Fig. 4)이고, API-diversity subset low / medium / high는 88.18 / 88.35 / 88.41이며 irrelevance는 87.92 / 88.75 / 88.75다(Fig. 5). 각 25,000 sample에서 ToolLLM, xLAM, ToolACE 데이터로 학습하면 전체 점수는 24.90, 40.51, 58.19이고 irrelevance는 4.41, 11.87, 86.42다(Table 6). learner 자신을 complexity evaluator로 사용하면 59.22이고 Qwen1.5-7B-Chat을 쓰면 57.61이다(Table 8). **결과(단일 연구).**

**조건 및 한계.** non-live ablation의 전체 점수 차이는 variance 없이 한 번의 LoRA run에서 최대 1.82점이다. tool response는 시뮬레이션되므로 execution error는 관측되지 않는다. general capability는 “some benchmarks에서 negligible performance degradation”이라는 text와 radar chart로만 보인다(§3.6, Fig. 8). complexity signal은 한 learner에 묶여 있으므로 같은 data가 다른 모델에는 잘못 보정될 수 있다(해석; App. H는 evaluation cost와 가능한 sampling bias를 열거한다).

**함의.** irrelevance 차이 79.43점(non-tool-use dialog를 제거할 때 86.42 대 6.99, Table 7)은 data type에서 온다. 반면 verification·complexity·diversity ablation은 non-live 전체를 최대 1.82점 움직인다. Negative samples section을 보라.

## §6 Hammer: 도구 이름 shortcut 제거

**정의.** Hammer는 xlam-function-calling-60k와 무관성 example 7,500개로 작은 모델을 fine-tune한다. example 일부에서는 function name, parameter name, default value를 random string으로 바꾼다([[hammer]] §4).

**문제.** function-calling 모델은 benchmark마다 일관되지 않은 점수를 낸다. xLAM-7B-fc는 BFCL, API-Bank, Seal-Tools, Tool-Alpaca, Nexus Raven에서 평균 69.05이고 Nexus Raven에서는 57.5인 반면 Granite-20B-FunctionCalling은 평균 74.19다(Table 1). Seal-Tools test set의 name을 random string으로 바꾸면 xLAM-1B-fc의 F1은 Hammer-1.5B보다 더 많이 낮아진다(Fig. 2). 저자들은 이 불일치를 description이 아니라 name에 의존한 탓으로 해석한다(§3.2, Interpretation).

**메커니즘.**
1. 무관성 set: training example 7,500개를 sample하고 candidate list에서 정답 function을 삭제한 뒤 label을 빈 list로 설정한다(§4.2).
2. 마스킹: function 및 parameter name을 random string으로 교체하고 randomized default value를 parameter description으로 옮기며, 같은 mapping으로 label을 다시 쓴다(§4.1).
3. 공개 script: example 67,500개를 세 번 복사하고 shuffle한 뒤 각각을 2/3 확률로 mask한다. function name은 5–15개의 random character, parameter name은 4–10개가 된다(`train/data_processing.py`).

**계산 예시.** 7,500 / 67,500 = 11.1%의 example이 무관성 example이다. 세 복사본은 training example 202,500개를 만들며, 이 중 2/3 × 202,500 = 135,000개가 mask될 것으로 예상된다(도출). 마스킹하지 않은 example은 `get_exchange_rate`가 currency에 관한 query와 맞는다고 가르친다. 마스킹한 example은 “Return the exchange rate between two currencies”라는 description을 유지하지만 function name을 `q7Vd_.x0K` 같은 string으로 바꾸므로, query와 tool을 연결하는 것은 description뿐이다. [figures/tool-pipeline.html](figures/tool-pipeline.html)의 Panel D는 mask 전후의 schema와 label을 보인다.

**근거.** API-Bank L-1 및 L-2, Tool-Alpaca, Seal-Tools, Nexus Raven에서 function name + argument의 평균 F1은 Hammer-7B 76.21, Granite-20B-FC 72.56, xLAM-7B-fc 67.65, Qwen2-7B-Instruct 59.84, GPT-4-0613 78.79다(Table 3). 같은 recipe를 DeepSeek-Coder-7B에 적용하면 74.94다(Table 5). BFCL(09/20/2024)에서 Hammer-7B는 전체 83.92, irrelevance 72.87이고 xLAM-7B-fc는 전체 79.41, irrelevance 79.76이다(Table 2). masking-ratio ablation(Qwen2-1.5B, Seal-Tools training, 한 epoch)에서 더 큰 ratio는 Seal-Tools 학습을 느리게 하고 API-Bank를 개선한다(§5.5, Fig. 5). irrelevance-share ablation(Qwen2-1.5B-Instruct, 10,000 sample)에서 share가 커질수록 irrelevance accuracy는 오르고 call accuracy는 내리며, 전체 최고점은 약 10%다(§5.6, Fig. 6). **결과(단일 연구).**

**조건 및 한계.** figure 값은 인쇄되어 있지 않다. masking과 irrelevance data는 다섯 benchmark 평균에서 별도로 ablate하지 않았다. 저자들은 non-BFCL benchmark의 format을 변환하고 일부 erroneous sample을 제거했으므로(repository README), 이 수치는 다른 논문과 직접 비교할 수 없다. 저자들은 10% share가 “underlying model and training dataset에 따라 조정이 필요할 수 있다”고 말한다.

**함의.** ratio ablation에서 더 높은 masking ratio는 학습 benchmark(Seal-Tools) 학습을 느리게 하고 다른 benchmark(API-Bank)의 accuracy를 높였다. 그림은 두 효과의 크기를 인쇄하지 않는다. 이는 한 shortcut(name)을 겨냥할 뿐 description style이나 output format은 다루지 않는다(§8).

## §7 멀티턴 및 agentic 도구 데이터

**APIGen-MT: plan을 검증한 뒤 대화를 시뮬레이션한다.** Phase 1은 τ-bench API, policy, domain data, persona를 sample하고 instruction `q`, ground-truth action `a_gt`, expected output `o_gt`를 생성한다([[apigen-mt]] §4.1). action을 실행하고 Python unit test로 작성한 domain policy에 대해 검사하며, LLM judge committee가 alignment를 투표한다. score threshold 아래 task는 feedback을 받고 다시 생성된다(§4.1.2). Phase 2는 `q`와 persona가 안내하는 simulated user를 gpt-4o agent와 상호작용시킨다. 이 user는 “underlying environment and available APIs를 알지 못하며”, final state가 `a_gt`와 맞고 output이 `o_gt`와 맞는 trajectory를 유지한다(§3.2.2, §4.2). agentic feedback은 Phase 1 success를 28%에서 70%로 높이고, Phase 2 success는 67%다(Fig. 4). xLAM-2-70b-fc-r은 평균 τ-bench pass@1 56.2로 gpt-4o-2024-11-20의 52.9보다 높고, xLAM-2-8b-fc-r은 46.7로 Llama 3.1 70B Instruct의 38.2보다 높다(Table 2).

먼저 검증해야 하는 이유는 다음과 같다. trajectory가 12 turn이고 각 turn이 독립 확률 0.95로 맞으면 전체 trajectory가 맞을 확률은 0.95^12 = 0.54이다(삽화이며 source의 수치는 아님). dialog 생성 전에 `a_gt`를 검증하면 final check는 자유 형식 대화에 대한 판단 대신 알려진 state와의 비교가 된다(해석). **한계.** generation domain, API, policy가 τ-bench evaluation domain이며 τ-bench test task와의 overlap check는 보고되지 않았다(§4.3, §5.1). 평가는 BFCL v3와 τ-bench뿐이다.

**τ-bench 및 τ²-bench: environment의 일부인 user.** τ-bench는 final database가 goal database와 같고 agent message가 필요한 정보를 포함할 때에만 task에 1점을 준다([[tau-bench]] §3). τ²-bench는 shared telecom environment에서 simulated user에게 도구를 준다. agent가 모든 도구를 제어하는 no-user mode에서 dual control로 gpt-4.1과 o4-mini를 옮기면 pass^1이 각각 18%와 25% 낮아진다([[tau2-bench]] §4.2, Fig. 4). passive user로 합성한 데이터(APIGen-MT, retail 및 airline)는 이 coordination skill을 포함하지 않는다(해석).

**Toucan: 대규모 실제 MCP server.** Toucan은 약 2,800개의 MCP server를 crawl하여 도달 가능한 495개를 유지하고, 다섯 LLM으로 task를 합성하며 Kimi-K2 rating으로 task를 필터링한다. 세 teacher model로 실제 server에 대한 trajectory를 생성하고 rule 및 LLM judge로 필터링한다. extension은 irrelevance task(도구 호출 0개를 유지), persona variant, multi-turn follow-up을 추가한다([[toucan-mcp]] §3). 119.3K subset으로 SFT하면 Qwen2.5-14B-Instruct의 BFCL V3 전체는 57.69에서 65.09로, τ-bench retail은 44.46에서 48.48로 오른다(Table 2, Fig. 9). 14B extension ablation은 narrowing과 recovery를 보인다. single-turn core만 쓰면 BFCL은 60.16으로 올리면서 τ-bench retail은 44.46에서 36.95로 낮아진다. irrelevance data를 추가하면 41.63, diversification을 추가하면 43.70, multi-turn을 추가하면 48.48이다(Fig. 9). 7B에서 non-live AST는 84.19에서 78.52로, τ²-bench telecom은 16.70에서 10.50으로 떨어진다(Tables 2–3). contamination check는 보고되지 않았다. **결과(단일 연구).**

**Kimi K2: frontier SFT pipeline의 MCP 및 합성 도구.** Kimi K2의 tool-use synthesis는 GitHub에서 가져온 실제 MCP tool 3,000개 이상과 hierarchical domain evolution의 합성 tool 20,000개 이상에서 출발한다. agent 및 rubric 기반 task를 생성하고 LLM user와 stateful tool simulator로 trajectory를 시뮬레이션하며, LLM judge가 rubric을 충족한 trajectory를 유지한다. coding task에는 실제 execution sandbox를 사용한다([[kimi-k2]] §3.1.1). Kimi-K2-Instruct는 Tau2 Avg@4로 retail 70.6, airline 56.5, telecom 65.8을 보고하며, 도구 사용이 불필요하게 활성화되면 성능이 떨어질 수 있음을 한계로 밝힌다([[kimi-k2]] Table 3, §5). **출처 유형:** 공식 technical report.

**함의.** 멀티턴 도구 데이터에는 state 기반 검증, plan에 의해 script되지 않은 user behavior, generation domain 이외 domain에서의 평가가 필요하다.

## §8 하나의 benchmark를 넘어 도구 사용 측정하기

**BFCL category와 matching.** single-turn item은 Simple, Multiple, Parallel, Parallel Multiple, Irrelevance, Relevance이고, crowd-sourced item은 실제 user query에서 온다. multi-turn item은 Missing Parameters, Missing Functions, Long Context를 추가하며 agentic item은 web search, memory, SQL을 추가한다([[bfcl]] §3). AST matching은 정확한 function name과 허용된 답 set 안에 있는 각 argument value를 요구한다(App. H). 점수는 mode에 따라 다르다. GPT-4-turbo-2024-04-09의 irrelevance는 FC mode에서 83.8, prompting mode에서 35.6이다(Table 1). stateful category는 single-turn보다 어렵다. gpt-4o-2024-11-20(Prompt)은 AST multiple에서 95.5, multi-turn base에서 59.0, memory에서 6.0이다(Table 1).

**pass^k.** task를 n회 실행해 c회 성공했을 때([[tau-bench]] §3):

```
pass^k = E_task[ C(c, k) / C(n, k) ]          (all k sampled trials succeed)
pass@k = 1 − E_task[ C(n − c, k) / C(n, k) ]  (at least one succeeds)
```

`C(a, b)`는 binomial coefficient이며 b > a일 때 0이다. expectation은 task에 대한 평균이다.

**계산 예시.** 두 task에 n = 4라고 하자. Task A는 c = 3이므로 pass^1 = 3/4 = 0.75, pass^2 = C(3,2)/C(4,2) = 3/6 = 0.50, pass^3 = 1/4 = 0.25, pass^4 = 0이다. Task B는 c = 4이므로 모든 k에서 pass^k = 1이다. 평균은 pass^1 = 0.875, pass^2 = 0.75, pass^3 = 0.625, pass^4 = 0.50이다. pass@2는 두 task 모두 1.0이다. 따라서 모델은 같은 trial에서 pass@k로는 신뢰할 만해 보이고 pass^k로는 신뢰할 수 없어 보일 수 있다. [figures/tool-pipeline.html](figures/tool-pipeline.html)의 Panel B는 임의의 n과 success count에 대해 두 curve를 계산한다. 공개된 참고값으로 τ-retail에서 gpt-4o function calling은 pass^1이 60% 초과, pass^8이 25% 미만이다(Fig. 4).

**형식 민감도.** BFCL V4는 return format(Python, JSON, 두 XML style), function-document format(Python, XML, JSON), tool-call tag, prompt format, prompt style을 바꾼다. 이는 single-turn entry 200개와 model 39개에 대한 26 variation이다([[bfcl-v4-format-sensitivity]] §2–§3). accuracy는 대체로 XML보다 Python 또는 JSON output에서 높고 특히 작은 모델에서 그렇다. tool-use에 특화된 모델이 가장 큰 failure를 보이는데, 예를 들어 watt-tool-70B는 JSON이 필요할 때 Python-style call을 반환하고 CoALM-70B는 tool-call tag가 필요할 때 거의 0점이다(§4.1, §4.5). blog 본문과 Figure 4 caption은 Python 및 XML documentation의 순서를 서로 다르게 제시한다(§4.2). xLAM은 “model overfitting on JSON format을 피하기 위해” 15 output format으로 학습하며, unified structured format이 필요할 때 GPT-4o의 ToolQuery success가 42% 하락한다고 보고한다([[xlam]] §3.2, §5.2.1). **출처 유형:** BFCL V4는 benchmark maintainer의 공식 blog이고, xLAM 결과는 단일 연구다.

**오염(contamination).** 검사를 보고한 source는 다음과 같다. BFCL은 crowd-sourced set에서 Nexus leaderboard test query를 제거했고(App. E.2), crowd-sourced와 curated data의 perplexity를 signal로 사용한다(§5.5). NexusRaven-V2는 benchmark task 하나를 공개하지 않는다([[nexusraven]] Findings). 검사를 보고하지 않은 source는 APIGen(BFCL category에서 가져온 query style, §3.3), APIGen-MT(같은 τ-bench domain에서 generation과 evaluation), ToolACE, Toucan이다. **열린 질문:** 후자 그룹에서 보고한 gain 중 얼마나 많은 부분이 decontaminated 또는 새로 수집한 tool set에서도 유지되는가.

## §9 도구 사용의 SFT 대 RL, 그리고 일반 행동의 부작용

**ToolRL 보상.** ToolRL은 규칙 기반 보상으로 4K example(2K ToolACE, 1K masked Hammer, 1K xLAM)에 GRPO를 적용해 학습한다([[toolrl]] §3.3, §4.1):

```
R_final   = R_format + R_correct,          R_format ∈ {0, 1}
R_correct = 6 · R_max / S_max − 3 ∈ [−3, 3]
r_match   = r_name + r_param + r_value,    S_max = 1 + |G| + Σ_j |keys(G_j)|
```

`G`는 ground-truth call의 집합이다. `r_name`은 예측 및 정답 tool-name set의 Jaccard overlap이고, `r_param`은 각 정답 call에 대해 argument-key set의 Jaccard overlap을 합한 것이며, `r_value`는 정확히 일치하는 argument value 수다. `R_max`는 예측 call과 정답 call을 가장 잘 matching했을 때의 전체 match score다.

**계산 예시.** 정답 call은 `get_weather(city="Paris", unit="C")`이고, prediction은 올바른 output format의 `get_weather(city="Paris", unit="F")`라고 하자. r_name = 1, r_param = 2/2 = 1, r_value = 1(city만 일치)이므로 r_match = 3, S_max = 1 + 1 + 2 = 4이다. R_correct = 6 · 3/4 − 3 = 1.5이고 R_final = 2.5다. reward가 (2.5, 4.0, −3.0, 1.0)인 GRPO group의 mean은 1.125, standard deviation은 2.607이므로 advantage는 (0.53, 1.10, −1.58, −0.05)이다(계산됨). 네 sample 중 둘은 negative gradient를 받는다. [figures/tool-pipeline.html](figures/tool-pipeline.html)의 Panel C는 다른 match pattern의 reward를 계산한다.

**근거.** BFCL V3 전체 점수는 각각 한 run(Table 1)에서 다음과 같다. Qwen2.5-7B-Instruct는 raw 41.97, R1-distilled example 4K로 SFT하면 36.53, raw model에서 GRPO하면 58.38이다. Qwen2.5-3B는 33.04, 41.97, 52.98이고, Llama-3.2-3B는 22.09, 44.16, 44.10이다. 학습하지 않은 web search multi-hop QA task인 Bamboogle에서 Qwen2.5-7B는 raw 69.6, SFT400 뒤 28.8, SFT4k 뒤 30.4, GRPO 뒤 72.0이다(Table 3). SFT checkpoint에서 시작한 GRPO는 raw model에서 시작한 GRPO보다 더 높은 training reward와 더 낮은 benchmark score에 도달한다(§4.3, Fig. 5). 저자들은 SFT에서의 memorization을 가설로 제시한다(해석). 더 긴 thinking에 보상을 추가하면 Qwen2.5-1.5B의 BFCL 전체는 46.20에서 33.23으로, irrelevance는 56.44에서 4.52로 낮아진다(Table 5). **결과(단일 연구).** Table 1의 두 행(Qwen2.5-3B 및 7B SFT400)은 전체 34.08과 irrelevance 8.11을 포함해 일곱 column 중 여섯 개의 값이 같고 non-live exec(61.50 대 66.68)만 다르다. 이는 table error를 시사하므로 이 장은 Table 1의 SFT400 행을 근거로 사용하지 않는다. 일반적인 SFT 대 RL 일반화는 ch-38a를 보라.

**추론 학습은 도구 환각을 높인다.** Qwen2.5-7B-Instruct에서 SynTool의 think-then-act GRPO는 BFCL multi-turn을 13.6에서 23.5로 올리고 IFEval을 62.4에서 59.8로 변화시키는 동시에, no-tool-available hallucination rate를 34.8%에서 90.2%로, distractor-tool rate를 54.7%에서 100.0%로 올렸다. GSM8K만으로 GRPO해도 두 rate가 모두 올랐다([[reasoning-trap-tool-hallucination]] Table 3, Fig. 3). R1-distilled Qwen2.5-7B의 rate는 instruct model의 34.8 / 54.7에 비해 74.3 / 78.7이다(Table 1). 저자들은 기존 tool-calling 및 instruction-following benchmark가 이 증가를 드러내지 못했다고 말한다(§4.4.2).

**일반 데이터 혼합.** AgentTuning에서 agent trajectory만으로 7B를 학습하면 held-out agent score는 0.09, general score는 0.22였다. ShareGPT와 η = 0.2로 agent data를 sample하면 0.67 및 0.63이었다([[agenttuning]] Table 5). 일반 xLAM model은 general instruction data를 training set의 20%~30%로 유지하지만([[xlam]] §3.5), ablation은 보고하지 않았다.

**함의.** tool benchmark에서 측정한 tool-call gain은 abstention, instruction following, non-tool task가 보존되었음을 보여 주지 않는다. 위 source 각각은 이들 중 하나를 측정했을 때에만 regression을 발견했다.

## 음성 샘플과 음성 피드백

**여기서 사용하는 네 가지 의미**(course standard): (1) 음의 한계 가치: positive target으로 썼을 때 성능을 낮추는 sample, (2) content로서의 음성: 실패 또는 “no call” case를 target에 넣어 cross-entropy로 학습하는 것, (3) conditioning으로서의 음성: control token 아래에서 실패를 학습하는 것, (4) gradient로서의 음성: sample likelihood를 명시적으로 낮추는 것. sample에서 probability mass를 제거하는 것은 (4)뿐이다.

**1. 도구 데이터에서 음성이 나오는 곳.** format·execution·semantic check에서 거부된 call([[apigen]] Table 1), goal state를 놓친 trajectory([[apigen-mt]] §4.2; [[agenttuning]] §2.1.3), 주어진 어떤 tool도 적용되지 않거나 argument가 빠진 query([[hammer]] §4.2; [[toolace]] §2.2.1; [[toucan-mcp]] Ext.1; [[bfcl]] App. B), rule reward가 낮은 rollout([[toolrl]] §3.3), DPO pair에서 거부된 fabricated call([[reasoning-trap-tool-hallucination]] §6.1)이다.

**2. 현재 관행이 하는 일.** discard(type 1): APIGen, APIGen-MT, AgentTuning, Toucan은 실패 sample을 버린다. content(type 2): 빈 call 또는 refusal target이 있는 APIGen relevance example 8,000개([[apigen]] App. B.3), Hammer의 empty-list label 7,500개, ToolACE의 non-tool-use dialog, 호출이 0개인 Toucan irrelevance trajectory 40K가 있다. conditioning(type 3): 이 장의 source에는 없다. gradient(type 4): ToolRL의 negative GRPO advantage 및 Reasoning Trap mitigation에서 DPO의 rejected term이 있다. APIGen-MT는 실패 trajectory를 가능한 “additional contrastive signal”이라고 부르지만 사용하지 않는다(§6).

**3. 메커니즘.** logit z에 대한 next token softmax에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`다. step η로 `log p_y`를 낮추는 gradient step은 `z_y`를 `−η(1 − p_y)`만큼, 다른 모든 `z_j`를 `+η p_j`만큼 바꾼다. 제거된 mass는 현재 probability에 비례해 다른 token으로 가므로 가장 확률이 높은 alternative가 가장 많이 얻는다. 계산 예시로 response 첫 token에서 모델이 distractor tool call을 시작할 확률(y)에 p = 0.70, text answer를 시작할 확률(a)에 0.25, clarification question을 시작할 확률(b)에 0.05를 부여한다고 하자. η = 1이면 logit은 (−0.30, +0.25, +0.05)만큼 바뀌고 새 probability는 0.581, 0.360, 0.059다(계산됨). 잘못된 call을 누르면 주로 text answer가 올라가며, text answer가 올바른 행동일 때만 이것이 맞다. 반대로 눌러 내린 call이 p = 0.05이고, 올바른 call이 0.90, 다른 output이 0.05이면 logit은 (−0.95, +0.90, +0.05)만큼 바뀌고 probability는 0.008, 0.969, 0.023이 된다(계산됨). 변화 대부분은 이미 가장 가능성이 높던 output으로 간다. type (2) target은 반대로 선택한 “no call” output을 직접 올린다. 전체 유도는 ch-43a에 있다.

**4. 수치를 포함한 근거.** 음의 한계 가치: 거부 sample을 다시 넣으면 BFCL은 7B에서 4.06–5.94, 1B에서 9.59–12.17 낮아진다([[apigen]] Fig. 5). filter하지 않은 agent trajectory는 held-in / held-out에서 1.34 / 0.47이고 filtered는 1.96 / 0.65다([[agenttuning]] Table 2). content: ToolACE의 non-tool-use dialog를 제거하면 irrelevance accuracy가 86.42에서 6.99로, multi-turn은 16.50에서 1.75로 낮아진다([[toolace]] Table 7). Toucan의 irrelevance data를 추가하면 BFCL은 60.16에서 64.74로, τ-bench retail은 36.95에서 41.63으로 오른다([[toucan-mcp]] Fig. 9). Hammer ablation에서는 irrelevance data가 많을수록 call accuracy가 낮아진다(Fig. 6). gradient: fabricated call을 rejected response로 사용하는 DPO는 no-tool hallucination을 90.2%에서 55.8%로, distractor hallucination을 100.0%에서 71.4%로 낮추며 SynTool reward도 0.45에서 0.34로 낮춘다([[reasoning-trap-tool-hallucination]] Table 4). **효과 크기.** 이 source 중 어느 것도 gain 중 음성에서 비롯된 몫과 positive에서 비롯된 몫을 분리하지 않는다.

**5. Control.** 같은 data 안에 positive target을 유지하라(Hammer는 irrelevance example을 positive call 60K와 섞고, Reasoning Trap DPO data도 tool이 존재하며 chosen response가 correct call이고 rejected response가 불필요한 refusal인 pair를 포함한다, [[reasoning-trap-tool-hallucination]] App. C.2). 양쪽을 모두 측정해 abstention share를 정하라([[hammer]] Fig. 6). SFT initialization이 overfit할 경우 raw model에서 on-policy RL을 유지하라([[toolrl]] §4.3). thinking length처럼 call correctness에 묶이지 않은 reward는 추가하지 말라([[toolrl]] Table 5).

**6. 진단.** relevance와 irrelevance accuracy는 갈라질 수 있으므로 따로 보고하라(Qwen2.5-72B-Instruct Prompt: relevance 100.0, irrelevance 72.8, [[bfcl]] Table 1). no-tool 및 distractor-tool hallucination rate를 측정하라([[reasoning-trap-tool-hallucination]] Eq. 1). DPO에서 chosen 및 rejected log probability를 log하고, GRPO 통계는 advantage sign으로 나누어 log하라. multi-turn task에서는 k > 1의 pass^k를 보고하라.

**7. 일반성에 미치는 효과.** abstention은 training data 밖 setting으로 전이할 수 있다. ToolRL은 저자들이 RL training에 포함되지 않았다고 기술한 BFCL irrelevance subset에서 Qwen2.5-3B GRPO model이 최고점을 낸다고 보고한다([[toolrl]] §4.3, Fig. 7). 반대 위험은 과도한 abstention이다. Kimi K2는 tool use가 불필요하게 활성화될 때 성능 저하를 보고하며, 불필요한 refusal은 Reasoning Trap pair에서 명시적인 rejected type이다. DPO mitigation은 task reward를 0.45 → 0.34로 치르게 했으므로, tool call의 negative-gradient training에는 hallucination check 옆에 utility check가 필요하다.

## 레시피

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | base checkpoint | DeepSeek-Coder-1.3B-instruct; DeepSeek-Coder-7B-instruct-v1.5 | arXiv:2406.18518v1 §5.1 | verified 2026-09-14 | ablation 보고 없음 |
| xlam-function-calling-60k | — | SFT (data) | generator; temperature; generator당 target | Mixtral-8x22B-Inst + DeepSeek-V2-Chat (release); 0.7; 40,000 | v1 §4.2 | verified 2026-09-14 | generator별 Table 1 pass rate |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | relevance example | 8,000(empty call 또는 refusal) | v1 App. B.3 | verified 2026-09-14 | ablation 보고 없음 |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | LR; schedule; warmup; epoch; optimizer | 5e-6; cosine; 50 steps; 4; AdamW | v1 App. B.3 | verified 2026-09-14 | ablation 보고 없음 |
| xLAM-1B (FC); xLAM-7B (FC) | 1.3B; 7B | SFT | cutoff length; device당 batch; grad accumulation; hardware | 2048; 6; 2; 8 × A100 40GB | v1 App. B.3 | verified 2026-09-14 | ablation 보고 없음 |
| xLAM-7b-fc-r; xLAM-1b-fc-r | 7B; 1.3B | SFT | data mixture | 50% synthetic function-calling, 50% other task | arXiv:2409.03215 §3.5 | verified 2026-09-15 | ablation 보고 없음 |
| xLAM general model (7b-r, 8x7b-r, 8x22b-r) | 7B–141B | SFT | general instruction share; output-format augmentation; warmup | 20%–30%; 15 output format; 100 steps, cosine | arXiv:2409.03215 §3.2, §3.5, §4.1 | verified 2026-09-15 | ToolQuery-Unified 비교(Table 3), format ablation 없음 |
| xLAM-2-fc-r series | 1B–70B | SFT | data; loss masking; epoch | APIGen-MT + APIGen + 기타 agentic data(비율 미보고); assistant token만; 최대 3 | arXiv:2504.03601v4 §5.1 | verified 2026-09-14 | ablation 보고 없음 |
| xLAM-2-fc-r series | 1B–70B | SFT | LR; batch; sequence length; DPO | 보고되지 않음 | body, App. A–B, model card 확인 | not reported | — |
| ToolACE-8B | 8B | SFT | base; method | LLaMA-3.1-8B-Instruct; LoRA rank 16, alpha 32, all module | arXiv:2409.00920 (ICLR 2025) §3.1 | verified 2026-09-15 | ablation 보고 없음 |
| ToolACE-8B | 8B | SFT | LR; warmup ratio; schedule; batch; epoch | 1e-4; 0.1; cosine; 48; 3 | App. C.2 Table 5 | verified 2026-09-15 | ablation 보고 없음 |
| ToolACE-8B | 8B | SFT | 전체 example; rejection rate | 보고되지 않음 | body, App. A–H 확인 | not reported | — |
| Hammer (released training script) | 1.5B–7B | SFT | data | xlam-function-calling-60k + 7,500 irrelevance, ×3, p = 2/3로 masked | github.com/MadeAgents/Hammer `train/data_processing.py` (main, read 2026-09-14); arXiv:2410.04587 §4.2 | verified 2026-09-15 | irrelevance share: Fig. 6(Qwen2-1.5B, 10K sample에서 ≈10%가 최고); masking ratio: Fig. 5(optimum 명시 없음) |
| Hammer (released training script) | 1.5B–7B | SFT | method; LR; schedule; warmup ratio; epoch | LoRA rank 32, all module; 5e-5; cosine; 0.00833; 1 | `scripts/train.sh` (main, read 2026-09-14) | verified 2026-09-15(released config; 논문은 checkpoint 생성에 썼다고 말하지 않음) | ablation 보고 없음 |
| Hammer (released training script) | 1.5B–7B | SFT | cutoff; device당 batch; grad accumulation; weight decay | 2048; 4; 2; 0.01 | `scripts/train.sh` | verified 2026-09-15 | ablation 보고 없음 |
| ToolLLaMA-2-7B | 7B | SFT | context; LR; warmup ratio; batch; epoch; checkpoint rule | PI ratio 2를 통한 8,192; 5e-5; 0.04; 64; 2; dev set 최고 | arXiv:2307.16789 App. A.3 | verified 2026-09-15 | ablation 보고 없음 |
| Toolformer (GPT-J) | 6.7B | SFT (data) | τ_s; k; m; τ_f | 0.05; 5; 5; 1.0(calculator, MT: 0.0; 20; 10; 0.5) | arXiv:2302.04761 App. A | verified 2026-09-15 | Table 2: τ_f 0.5/1.0/2.0의 example count; accuracy ablation 없음 |
| Toolformer (GPT-J) | 6.7B | SFT | batch; LR; warmup | 128; 1e-5; linear, 최초 10% | §4.1 | verified 2026-09-15 | ablation 보고 없음 |
| ToolRL Qwen2.5-7B-Instruct (GRPO cold start) | 7B | RL | data; algorithm; KL | 4K(2K ToolACE, 1K Hammer masked, 1K xLAM); GRPO; KL term 없음 | arXiv:2504.13958v1 §4.1–4.2, App. B | verified 2026-09-15 | Table 1: cold start 58.38 대 SFT4k 36.53 대 SFT400+GRPO 39.25 |
| ToolRL Qwen2.5-7B-Instruct (GRPO cold start) | 7B | RL | batch; query당 rollout; epoch; LR; temperature; max prompt / response | 512; 4; 15; 1e-6; 1.0; 2,048 / 1,024 | App. B Table 8 | verified 2026-09-15 | ablation 보고 없음 |
| ToolRL | 1.5B–7B | RL | reward | R_format ∈ {0,1} + R_correct ∈ [−3, 3] | §3.3 | verified 2026-09-15 | Table 5: length reward 추가는 BFCL 하락(1.5B 46.20 → 33.23) |
| Toucan-tuned Qwen2.5-7B/14B/32B-Instruct | 7B–32B | SFT | data | 119.3K = 28.3K core + 40K irrelevance + 15.8K diversify + 35.2K multi-turn | arXiv:2510.01179v1 §4.1 | verified 2026-09-15 | Fig. 9 extension ablation(14B) |
| Toucan-tuned Qwen2.5 model | 7B–32B | SFT | template; LR; epoch; effective batch; optimizer; max length | Hermes; 2e-5; 2; 64; AdamW (0.9, 0.999), ε 1e-8; 32,768 | App. C.2 Table 5 | verified 2026-09-15 | ablation 보고 없음 |
| AgentLM-7B / 13B / 70B | 7B–70B | SFT | agent data sampling ratio η | 0.2(ShareGPT 0.8) | arXiv:2310.12823v2 §2.2.2 | verified 2026-09-14 | 7B held-out task에서 η를 0–1 scan; Table 5 endpoint |
| Kimi-K2-Instruct | 1.04T / 32B act. | SFT (data) | tool repository | 실제 MCP tool 3000개 이상; 합성 tool 20,000개 이상 | arXiv:2507.20534 §3.1.1 | verified 2026-09-14 | Figure 9(t-SNE coverage); ablation 없음 |

2026-09-15 날짜의 행은 그 날짜에 primary source에서 읽었다. [[xlam]], [[toolace]], [[hammer]]의 library card는 검증되지 않았고, [[toolrl]] 또는 [[toucan-mcp]]에는 card가 없었기 때문이다. chapter excerpt에는 확인한 extract가 있다.

**소규모 범용 run의 시작점.** 1B–7B instruction-tuned model의 function-calling SFT에 대해 검증된 APIGen setting은 LR 5e-6, cosine decay, warmup 50 step, 4 epoch, AdamW, cutoff length 2048, device당 batch 6과 gradient accumulation 2, 검증된 call 옆 relevance example 8,000개다. 이들은 8 × A100 40GB에서 DeepSeek-Coder-1.3B/7B에 사용됐고 BFCL만 평가했다. cross-benchmark robustness를 위해 Hammer release는 irrelevance example 7,500개(데이터의 약 11%, Qwen2-1.5B에서 최선이었던 약 10% share 근처)를 더하고, 사본의 2/3에서 이름을 mask한다. 범용 사용에서는 general instruction data를 mixture에 유지하라(xLAM general model: 20%–30%; AgentLM: agent data에 η = 0.2). 위 단일-skill run은 general benchmark를 보고하지 않기 때문이다.

## 일반화 관점

**(a) breadth를 늘리는 것.**
- 실제 API에서 search-generated trajectory를 사용한 unseen-tool 일반화: ToolLLaMA는 본 도구의 보지 못한 instruction에서 57.0, 보지 못한 category 도구에서 62.0을 통과하고 APIBench로 전이한다([[toolllm]] Tables 4–5).
- name matching 대신 description 읽기: ratio ablation에서 masking은 cross-task result(API-Bank)를 개선하며, DeepSeek-Coder-7B의 Hammer recipe는 다섯 benchmark F1 평균 74.94로 67.65보다 높다([[hammer]] Fig. 5, Table 5).
- data-type coverage: non-tool-use dialog와 parallel call은 각각 없으면 사라지는 category를 지니며([[toolace]] Table 7), API diversity는 irrelevance accuracy를 87.92에서 88.75로 올린다([[toolace]] Fig. 5).
- tool-use-specialized model은 format이 바뀔 때 실패하므로([[bfcl-v4-format-sensitivity]] §4.5), 학습에서 여러 output format을 사용한다([[xlam]] §3.2).
- raw model에서 규칙 기반 call reward를 사용한 RL: raw Qwen2.5-7B 대비 BFCL V3 +16.41, 학습하지 않은 Bamboogle에서 72.0 대 69.6([[toolrl]] Tables 1, 3).
- irrelevance 및 multi-turn extension이 있는 실제 MCP environment: 14B에서 BFCL V3 +7.40, τ-bench retail +4.02([[toucan-mcp]] Table 2, Fig. 9).

**(b) narrowing 또는 forgetting을 일으키는 것.**
- 한 data family로 학습: xLAM-7B-fc는 Nexus Raven 57.5, BFCL 79.41이며([[hammer]] Table 1), crowd-sourced BFCL query에서 perplexity가 상승한다([[bfcl]] §5.5).
- single-turn tool SFT: 14B에서 τ-bench retail 44.46 → 36.95([[toucan-mcp]] Fig. 9); 7B에서 non-live AST 84.19 → 78.52, τ²-telecom 16.70 → 10.50([[toucan-mcp]] Tables 2–3).
- R1-distilled tool trace SFT: 7B에서 Bamboogle 69.6 → 28.8(SFT400), 30.4(SFT4k), BFCL V3 전체 41.97 → 36.53(SFT4k)([[toolrl]] Tables 1, 3).
- reasoning RL 또는 R1 distillation: tool hallucination이 90.2% / 100.0%까지 상승([[reasoning-trap-tool-hallucination]] Tables 1, 3).
- general data 없이 agent-only data: 7B held-out agent score 0.09, general 0.22([[agenttuning]] Table 5).
- abstention data 과다: irrelevance share가 늘수록 call accuracy가 낮아진다([[hammer]] Fig. 6).

**(c) 이 stage에서 측정하는 법.**
- held-out tool split(Inst / Tool / Cat)과 training data와 다른 group의 benchmark([[toolllm]] §3.2; [[hammer]] Table 3).
- relevance와 irrelevance를 따로, FC와 prompting mode 둘 다, live(crowd-sourced)와 non-live category를 나누어 보고하는 BFCL([[bfcl]] Table 1).
- τ-bench에서 k > 1의 pass^k, τ²-bench에서 dual-control 대 no-user mode([[tau-bench]] Fig. 4; [[tau2-bench]] Fig. 4).
- return format, documentation format, tool-call tag의 format variation([[bfcl-v4-format-sensitivity]] §2).
- 추론 학습 뒤 missing 및 distractor tool을 사용한 abstention([[reasoning-trap-tool-hallucination]] §3.1).
- 전후 general benchmark: tool을 비활성화한 LM perplexity([[toolformer]] Table 8), IFEval([[reasoning-trap-tool-hallucination]] Table 3), MMLU 및 MT-Bench([[agenttuning]] Table 4).
- contamination: 합성 tool pool·task domain과 evaluation set의 overlap, curated 및 fresh query의 perplexity 비교([[bfcl]] §5.5).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 검사 |
|---|---|---|
| 약한 generator의 filter하지 않은 generation으로 학습 | BFCL 하락, 1B 학생에서 더 큼 | stage별 failure count; 소규모 run의 add-back ablation([[apigen]] Table 1, Fig. 5) |
| 하나의 benchmark를 tool-use ability로 간주 | BFCL은 높지만 Nexus Raven 또는 API-Bank가 낮음 | 서로 다른 group의 benchmark 적어도 둘에서 평가([[hammer]] Table 1) |
| abstention target 없음 | irrelevance accuracy가 한 자릿수 근처 | remove-type ablation 또는 irrelevance score([[toolace]] Table 7: 6.99) |
| call을 측정하지 않고 abstention share 선택 | 답할 수 있는 query에 refusal; call accuracy 하락 | share에 대한 irrelevance와 call accuracy plot([[hammer]] Fig. 6) |
| 모든 training data에 하나의 output template | deployment가 JSON, XML, tag를 요구하면 실패 | BFCL V4 format variation 실행([[bfcl-v4-format-sensitivity]] §4) |
| multi-turn agent에서 pass@1만 보고 | 반복 session에서 일관되지 않은 행동 | k에 대한 pass^k curve([[tau-bench]] §3) |
| 같은 domain에서 generation과 evaluation | 맞춰진 benchmark에서만 큰 gain | 다른 domain에서 평가; overlap check 보고([[apigen-mt]] §4.3) |
| plan을 이미 알거나 행동하지 않는 simulated user | control을 공유하는 user에게 agent 실패 | τ²-bench dual-control 대 no-user gap([[tau2-bench]] §4.2) |
| RL 전에 distilled trace로 SFT | 높은 training reward, 낮은 held-out score | held-out set에서 SFT-init 대 raw-init RL 비교([[toolrl]] §4.3) |
| abstention check 없는 tool SFT 뒤 reasoning RL | BFCL은 오르며 hallucinated call도 상승 | no-tool 및 distractor-tool rate([[reasoning-trap-tool-hallucination]] Table 3) |
| BFCL version을 넘어서 leaderboard number 인용 | 같은 model이 논문마다 다른 score | leaderboard 날짜와 version 명시(xLAM-7B (FC): APIGen 85.65, xLAM v2 table 80.18, ToolACE v3 table 51.45) |

## 이해 점검

1. APIGen의 add-back ablation에서 같은 거부 data로 1B 학생은 7B 학생보다 더 많이 잃는다. 인과 설명을 제시하고 이를 시험할 experiment를 말하라.
2. Toolformer는 result가 있을 때의 loss가 no-call 및 result 없는 call loss의 최소값보다 낮을 때에만 call을 유지한다. 두 번째 term을 제거하면 어떤 종류의 spurious call이 유지되는가?
3. ToolACE verification 및 complexity ablation은 BFCL non-live overall을 2점 미만으로 바꾸지만, non-tool-use dialog 제거는 irrelevance를 79점 바꾼다. 이 metric에서 data type이 filtering보다 더 큰 효과를 내는 이유는 무엇인가?
4. function-name masking이 training benchmark 학습은 늦추면서 다른 benchmark accuracy는 개선할 수 있는 이유를, 모델이 tool을 고르기 위해 사용할 수 있는 것과 사용할 수 없는 것으로 설명하라.
5. 두 agent가 τ-retail에서 같은 pass^1을 가진다. task별 success rate가 어떤 분포일 때 두 agent의 pass^8 차이가 가장 큰가?
6. ToolRL에서 4K R1-distilled example SFT는 Bamboogle을 69.6에서 30.4로 낮췄지만, 같은 prompt에서 GRPO는 72.0으로 올렸다. 이를 야기할 수 있는 두 run의 차이를 열거하고 SFT-initialized GRPO run이 어느 차이를 분리하는지 말하라.
7. `∂ log p_y / ∂ z_j = 1[j = y] − p_j`를 사용해 fabricated tool call을 누르면 clarification question 대신 직접 text answer가 올라갈 수 있는 이유와, 결과를 바꿀 data가 무엇인지 설명하라.

## 연결

- Previous: ch-25 — Multi-Turn Conversation Synthesis. 그 장의 user simulation과 multi-turn structure가 §7의 기반이다.
- Next: ch-27 — Agentic Trajectory Data.
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix(APIGen은 verify step의 사례).
- ch-04 — Sequence Packing, Loss Masking, and Chat Templates(call format 및 assistant-only loss).
- ch-29c — Agentic Environment and Task Synthesis at Scale; ch-29d — User Simulators, Trajectory Verification, and Failed Trajectories(§7).
- ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares(general-data share).
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-32d — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training.
- ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity(§9).
- ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation; ch-48 — Contamination Detection and Its Effect on Reported Scores; ch-51a — Evaluating Agent Generality and Reliability(§8).

## 출처

- [[toolformer]] — sampling 및 loss-filter 수식, threshold, Table 2 count, math result, perplexity retention.
- [[toolllm]] — ToolBench 규모, DFSDT mechanism 및 yield, Inst/Tool/Cat split, APIBench transfer, training setting.
- [[apigen]] — 세 단계 verification, Table 1 failure count, Fig. 5 add-back ablation, xLAM (FC) setting.
- [[apigen-mt]] — blueprint validation, simulated user, trajectory statistic, τ-bench 및 BFCL v3 result, training mixture.
- [[xlam]] — base model, format augmentation, data mixture share, BFCL v2 table, ToolQuery-Unified result.
- [[toolace]] — API synthesis, learner-loss complexity, dual-layer check, BFCL-v3 result, data-type 및 source ablation.
- [[hammer]] — cross-benchmark inconsistency, masking 및 irrelevance method, released script, ablation.
- [[bfcl]] — category, AST matching rule, mode, stateful result, perplexity contamination check.
- [[bfcl-v4-format-sensitivity]] — format variation 및 model-specific failure.
- [[tau-bench]] — state-based reward 및 pass^k 정의.
- [[tau2-bench]] — dual-control telecom domain 및 mode ablation.
- [[toucan-mcp]] — MCP-server pipeline, extension ablation, 7B regression.
- [[toolrl]] — rule-based call reward, GRPO 대 SFT result, length-reward ablation.
- [[agenttuning]] — agent/general data mixing 및 filtered 대 unfiltered trajectory.
- [[reasoning-trap-tool-hallucination]] — reasoning RL·distillation 뒤 tool hallucination, DPO mitigation.
- [[kimi-k2]] — MCP 및 synthetic tool repository, Tau2 result, 기술된 한계.
- [[gorilla]] — correction에 이름이 나옴(hallucination number).
- [[nexusraven]] — correction에 이름이 나옴(unpublished mix) 및 withheld benchmark task.
- [[granite-function-calling]] — correction에 이름이 나옴(일곱 training task).
