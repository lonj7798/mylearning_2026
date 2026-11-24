<!-- chapter: ch-18
     track: synthetic
     kind: content
     title: 합성 데이터 설계 패턴: 생성, 필터링, 중복 제거, 검증, 선택, 혼합
     deps: [ch-00, ch-10a, ch-12]
     sources: [[self-instruct]], [[nemotron-4-synthetic]], [[nemotron-4-synthetic-recipe]], [[apigen]], [[openmathinstruct-2]], [[openmathinstruct-2-recipe]], [[tulu-1-how-far-can-camels-go]], [[sft-data-composition-dmt]], [[tulu-3]], [[flan]], [[transferability-of-llm-reasoning]], [[prolong]], [[apigen-mt]], [[west-of-n]], [[nathan-lambert-synthetic-data]]
     figures: figures/synth-loop.html
     revised: 2026-09 (generality revision)
-->

# Chapter 18 — 합성 데이터 설계 패턴: 생성, 필터링, 중복 제거, 검증, 선택, 혼합

> **핵심 통찰.** Post-training 합성 데이터 pipeline은 생성, 필터링, 중복 제거, 검증, 선택, 혼합의 순서로 설명할 수 있다. 그 전에 서면으로 정의한 목표 분포와 pipeline이 절대 조정하지 않는 평가 분할이 필요하다. 공개 근거는 어느 한 단계가 항상 결정적이라고 말하지 않는다. APIGen은 실행 또는 의미 검사를 통과하지 못한 샘플을 되돌려 넣으면 BFCL 정확도가 4.06~12.17점 낮아진다고 했지만, OpenMathInstruct-2는 학습 쌍이 256K 이상이면 수학 오답이 최대 20%여도 정확도 비용이 작다고 했다. 지시 튜닝 혼합에서도 모든 능력에서 최고인 단일 데이터셋은 없었다(Tülu 1). 따라서 혼합 단계는 범위 결정이다.
>
> **지침.** 출력이 도구 호출·코드처럼 정확히 실행 가능한 형식이면 SFT 전에 실행 검사를 사용하라. 검증하지 않은 실패는 BFCL 정확도를 낮추며 1B 학생에서 7B보다 더 크게 낮아진다([[apigen]] 그림 5). 자유 형식 reasoning trace이고 데이터셋이 크면 엄격한 해법 필터보다 고유 질문과 강한 teacher에 예산을 쓰라. 필터링은 그곳에서 이득을 주지 않았다([[openmathinstruct-2]] 표 2~3, 그림 6). 항상 데이터를 생성하기 전에 홀드아웃 평가 suite를 고정하고 합성 데이터가 대상으로 삼지 않은 능력을 측정하라. 대상 데이터가 개발 지표를 높여도 보지 못한 지표로 전이되지 않았기 때문이다([[tulu-3]] 표 32).

## 학습한 버전에 대한 정정

이전 줄 참조(L…)는 git commit 4a72e54 버전을 가리킨다.

1. “8개 in-context 예시(6 seed + 2 생성)를 넣은 GPT-3(text-davinci-001)”은 부정확하다. Self-Instruct는 vanilla GPT-3 “davinci”로 데이터를 생성했고 text-davinci-001(InstructGPT-001)은 비교 모델이었다. 8개 지시는 인간 작성 6개와 모델 생성 2개다([[self-instruct]] arXiv:2212.10560 §2.2, §3, 각주 1).
2. “raw 후보 약 252K”, “ROUGE-L > 0.7이 raw의 약 50%를 삭제”는 논문에 없다. 논문은 필터 후 지시 52,445개와 인스턴스 82,439개를 보고한다(표 1). 논문의 “252”는 사용자 평가 지시 수다(§4.4). 제외 keyword는 “image, picture, graph”다(§2.2).
3. “최소 verify가 약점” 대신 품질을 측정했다. 표본 지시 200개 중 92%가 유효 task, 58%가 정답 output, 54%가 모든 field 유효였다(§3.3, 표 2). InstructGPT-003으로 재생성하면 252 사용자 task에서 최고 응답 비율이 44.4%에서 54.4%로 올랐다(§4.5, 그림 7).
4. Self-Instruct의 반복별 root-verb entropy 하락을 보여주는 무필터 실험은 없다. 표 2는 품질 검토이며, §3.2는 verb-noun 구조(52,445개 중 26,559개)와 seed와의 ROUGE-L 겹침(그림 4)을 보고한다.
5. 2022 Self-Instruct에서 “graph” 금지가 수학 도식 데이터를 없앴다는 출처는 없다. 삭제한다.
6. Nemotron-4의 첫 prompt·응답은 Mixtral-8x7B-Instruct-v0.1에서, 이후는 340B intermediate model에서 왔다([[nemotron-4-synthetic]] §3.2.1, §3.2.4).
7. 약 20K 인간 예시(10K SFT, 10K HelpSteer2 reward/preference)가 사용되었고 preference는 DPO 160K + RPO 300K다. 약 1.46M에는 Code SFT 800K와 General SFT 200K도 포함된다(§3.2, §3.3).
8. Nemotron-4 보고서는 Genetic Instruct code 데이터에 대해서만 중복 제거·필터링을 언급하며 fitness 검사는 compiler가 아닌 LLM이다(§3.3.1). 합성 대화는 reward-model score 임계값으로 필터링한다(§3.2.2).
9. 정답/검증기가 있으면 정답을 chosen, 오답을 rejected로 한다. 없으면 초기에는 양방향·일관 판정 LLM judge, 이후 reward model을 사용했다(Chat-Hard 정확도 0.87 대 0.54). Reward model은 chosen 고품질도 필터링한다(§3.2.3, §3.3.2).
10. reward-model 오류 누적 또는 작은 인간 anchor가 대형 corpus를 지탱한다는 인용문은 Nemotron-4 보고서에 없으며 unsupported로 삭제했다.
11. APIGen generator는 DeepSeek-V2-Chat, DeepSeek-Coder-33B-Inst, Mixtral-8x22B-Inst, Mixtral-8x7B-Inst이고 temperature 0.7이다. 샘플 k 범위와 rare API upweight는 보고되지 않았다([[apigen]] §3.3, §4.2).
12. Python 함수는 subprocess, REST API는 호출하며 timeout은 보고되지 않았다. semantic checker는 이름 없는 “another LLM”이다(§3.2, 부록 B.2).
13. pass rate는 DeepSeek-Coder-33B-Inst 34.42%에서 DeepSeek-V2-Chat 84.15%다(표 1). 약 60K 공개 샘플은 Mixtral-8x22B-Inst와 DeepSeek-V2-Chat에서만 왔다(§4.2).
14. APIGen의 MinHash와 최종 bucket 균형은 보고되지 않았다. 네 query style(simple, multiple, parallel, parallel multiple)은 생성 때 사용한다(§3.3).
15. APIGen ablation은 실패 데이터를 되돌려 넣었다. xLAM-7B에서 semantic +fail은 −4.06, execution +fail은 −5.94, xLAM-1B에서는 각각 −9.59, −12.17이며 format arm은 없다(§5.2, 그림 5).
16. xLAM-7B(FC)는 DeepSeek-Coder-7B-instruct-v1.5에서, xLAM-1B(FC)는 DeepSeek-Coder-1.3B에서 fine-tune했고 BFCL 순위는 각각 6위(85.65), 24위(74.41)였다(§5.1, 표 2).
17. APIGen teacher API 비용 약 8K달러와 stage 4가 stage 1보다 비싸다는 내용은 보고되지 않았다.
18. OpenMathInstruct-2는 새 질문에 temperature 0.7로 해법 32개를 만든다. 데이터는 고유 질문 607.3K, 합성 592K이며 원 질문은 source별 7.4K다([[openmathinstruct-2]] §2.2, §3, 표 5).
19. 합성 질문에는 gold answer가 없고 32개 해법의 majority answer와 최소 vote threshold 0을 쓴다. 평가는 GPT-4o answer judge이며 SymPy·7% audit은 보고되지 않았다(§3, 부록 C.1, §4).
20. 후처리는 여러 `\boxed` 삭제, 첫 `\boxed` 문장 뒤 자르기, 1024토큰 초과 또는 200자 미만 삭제 등 부록 A.2 목록이다. 문제 내 해법 dedup은 보고되지 않았다.
21. Llama3.1-8B-Base는 14M 쌍 전체, 70B-Base는 5M 부분집합으로 학습했고 1.5B 모델은 없다. 원 GSM8K·MATH 학습 질문 해법도 포함된다(§4, 표 5).
22. 11.05M 쌍 / 새 질문 592.7K = 18.6 해법/문제다(표 5에서 도출).
23. teacher sampling 650K H100-hour는 보고되지 않았다.
24. 14M corpus에서 saturation 징후가 없다고 보고했다(§4, 그림 1).
25. OMI-2 이득을 큰 teacher와 SymPy 검증으로 분해한 근거는 없다. matched coverage에서 teacher ablation은 37.9 대 30.1, judge·reward-model filter는 이득이 없고(표 3), ≥256K 쌍에서 오답 20%까지 비용이 작다(그림 5).
26. West-of-N은 기존 reward model 아래 N개 응답의 best/worst로 preference pair를 만드는 방법이다([[west-of-n]] 초록).
27. Nathan Lambert의 “synthetic data can do almost all of the work”는 강한 open-weights model fine-tuning에 대한 practitioner 의견이며 통제 실험이 아니다. verification 조건은 그 문장에 없다.
27. Nathan Lambert의 “강한 base model과 robust verification이 있으면 synthetic data가 거의 모든 일을 할 수 있다”는 서술은 부정확하다. 2024년 6월 post는 강한 open-weights model을 fine-tuning하는 맥락에서 “I think synthetic data can do almost all of the work”라고 말한다. Verification condition은 그 문장의 일부가 아니다(interconnects.ai/p/frontiers-in-synthetic-data, opening section; [[nathan-lambert-synthetic-data]]). 이는 controlled experiment가 없는 practitioner opinion이다. “scarce resource is verifiable prompts”라는 표현은 확인한 post에서 찾지 못했으므로 여기서 사용하지 않는다.
28. ProLong은 SFT에서 합성 long instruction을 시험했고 짧은 UltraChat만이 가장 높았다(합성 0% 55.7, 50% 43.3; [[prolong]] §5, 표 8).
29. verification bottleneck 논증은 아래 §5에서 보고된 ablation으로 재구성한다. 출력 유형, 학생 크기, 데이터 규모, 목표에 따라 검증 가치가 다르다.
30. 교차참조는 iteration/self-improvement ch-45, judge ch-49·ch-42, IFD·LESS·Superfiltering ch-22, model collapse ch-23으로 정정한다. ch-20 distillation, ch-21 taxonomy/textbook synthesis, ch-22 selection, ch-23 collapse/verification, ch-24~28 reasoning·multi-turn·tool·agentic·long context, multi-skill은 ch-30b다.

## 범용 모델에 이 장이 중요한 이유

일부 공개 post-training pipeline에서는 다른 모델이 데이터 대부분을 생성한다. Nemotron-4-340B-Instruct는 SFT·preference 데이터의 98% 이상을 합성으로 사용했다([[nemotron-4-synthetic]] §3.2). 이 과정의 목표는 표적이 아닌 태스크도 잘하는 모델인데, 합성 데이터는 가장 싸게 검사할 수 있는 수학 답·실행 코드·도구 호출을 과도하게 생산해 그 능력을 혼합물에서 과대표집한다.

이 패턴은 생성 데이터가 들어가는 모든 post-training 단계에 적용된다: SFT 데이터(이 장 및 ch-24~28), preference pair(ch-39), RL prompt pool(ch-16), 합성 pretraining·mid-training(ch-21, ch-29a). 이 장은 단계별 용어와 근거, 각 실패 모드의 검사를 제공한다.

## §1 루프와 전제: 목표 분포와 홀드아웃 분할

**정의.** 합성 데이터 pipeline은 소량의 신뢰 입력(seed task, API, 문서, 질문)을 학습 예시로 바꾸는 연산 순서다. 이 과정은 일곱 named step을 사용한다. 0단계는 목적과 성공 측정을 정하고 1~6단계가 데이터를 만든다.

**해결하는 문제.** 0단계가 없으면 이후 단계가 개발 중 보는 metric에 맞춰진다. Tülu 3는 이를 직접 측정했다. 개발·미지 suite는 같은 능력을 다른 벤치마크로 다루며 Persona instruction-following 데이터를 제거하면 개발 IFEval은 72.8에서 53.6으로 하락하지만 미지 IFEval-OOD는 17.6에서 18.0으로만 변했다([[tulu-3]] §7.4.1, 표 32). 개발 지표는 19.2점 올랐지만 보지 못한 지표는 오르지 않았다(결과, 단일 연구).

| 단계 | 답하는 질문 | 출처의 메커니즘 | 작동 검사 |
|---|---|---|---|
| 0 목표·홀드아웃 | 어떤 능력을, pipeline이 보지 않는 무엇으로 측정하는가? | FLAN 홀드아웃 task cluster; Tülu 3 개발 대 미지 suite; 오염 제거 | 모든 데이터 결정에 미지 suite 점수 보고 |
| 1 생성 | Generator가 무엇을 만드는가? | few-shot bootstrap; 질문 augmentation; API 조건 생성; 다중 model 응답 | raw output 다양성 통계 |
| 2 필터 | 형식·표면 이유로 무엇을 버리는가? | JSON parse, keyword·길이 규칙, truncation | task family별 acceptance rate |
| 3 중복 제거 | 무엇을 중복으로 버리는가? | ROUGE-L 임계값, MinHash(ch-12) | 최근접 이웃 유사도 분포 |
| 4 검증 | 무엇을 틀렸다고 버리는가? | 실행, 정답 일치, majority vote, LLM·reward-model judge | 라벨 표본의 거짓 수용·거짓 거부율 |
| 5 선택 | 생존자 중 무엇을 얼마나 유지하는가? | reward-model filter, generator 선택, 질문 수 예산(ch-22) | 선택 부분집합의 홀드아웃 이득 |
| 6 혼합 | 다른 데이터·단계와 어떻게 합치는가? | 연결, staged SFT, replay 비율 | 표적 외 능력의 전후 점수 |

대화형 [figures/synth-loop.html](figures/synth-loop.html)은 각 단계의 실패 증상·검사·네 case-study pipeline의 처리를 보여주고 어떤 generator의 APIGen 단계별 yield도 계산한다.

**범용성의 측정으로서 홀드아웃 task.** FLAN은 task cluster로 unseen task를 정의한다. cluster의 어떤 데이터셋도 instruction tuning에 사용하지 않아야 unseen이다([[flan]] §2.2). 세 cluster를 홀드아웃하면 평균 점수는 학습 cluster 1개(11개 데이터셋)의 49.9에서 7개(39개)의 63.5로 올랐다(§4.1, 그림 6). 8B 이하 모델에서는 instruction tuning이 홀드아웃 성능을 낮췄다(§4.2, 그림 7)(단일 연구). Self-Instruct는 unseen SuperNI 119개와 저자가 쓴 사용자 지시 252개에서 평가했다([[self-instruct]] §4.3, §4.4). Tülu 3는 개발 중 미지 suite 점수를 보지 않아 개발 평가 과적합을 측정하게 했다([[tulu-3]] §2.2).

**오염 제거는 0단계의 일부다.** 생성 데이터가 홀드아웃 항목을 포함하지 않을 때만 분할이 홀드아웃이다. Tülu 3는 테스트 instance 토큰의 50% 초과가 한 학습 instance와 공유한 8-gram에 들어가면 matched로 표시하고, 어떤 평가의 2% 초과 instance와 일치하면 training set을 오염으로 본다([[tulu-3]] §3.2). 20토큰 테스트가 토큰 1~12를 덮으면 12/20 = 60% > 50%라 matched다. GSM8K test 1,319개에서 2%는 26.38이므로 27개 이상 일치하면 오염이다.

N-gram 규칙은 paraphrase를 놓친다. Tülu 3는 embedding이 원칙적으로 paraphrase를 찾지만 topical similarity와 구분하지 못했다고 한다. OpenMathInstruct-2는 각 합성 질문에 대해 embedding으로 유사 테스트 질문 상위 5개를 검색하고 Llama3.1-405B-Instruct에 양방향 paraphrase 여부를 물어 569K → 519K를 삭제했다([[openmathinstruct-2]] §3.1, 부록 C.2). 오염 검출은 ch-48에서 다룬다.

## §2 네 pipeline을 loop에 배치

각 cell은 출처가 보고한 내용과 위치를 적는다. “보고되지 않음”은 출처를 확인했지만 해당 단계를 설명하지 않는다는 뜻이다.

| 단계 | Self-Instruct(arXiv 2022-12) | Nemotron-4 340B(arXiv 2024-06) | APIGen(arXiv 2024-06) | OpenMathInstruct-2(arXiv 2024-10) |
|---|---|---|---|---|
| 0 홀드아웃 | SuperNI unseen 119개; 새 사용자 지시 252개(§4.3~4.4) | 단계별 MT-Bench, IFEval, MMLU 등(표 6); unseen split 없음 | BFCL만; held-out API split 없음(§5) | ablation용 MATH train 1K holdout; LLM paraphrase 오염 제거(§2.2, §3.1) |
| 1 생성 | GPT-3 davinci; seed task 175개; in-context 8개(인간 6, 생성 2)(§2.2) | Mixtral-8x7B-Instruct-v0.1이 3K 주제·keyword로 prompt; intermediate model 응답(§3.2.1, §3.2.4) | 공개 generator 4개, 각각 목표 40,000, T = 0.7; API 3,673개(§4.1~4.2) | Llama3.1-405B-Instruct; 5-shot 유사 질문 augmentation; 새 질문당 T = 0.7 해법 32개(§3) |
| 2 필터 | 길이 heuristic; 입력 반복 output; “image, picture, graph”(§2.2) | 대화 reward-model 임계값(§3.2.2) | query·answer JSON 형식; 주어진 function만 호출(§3.2) | 후처리(예: 1024토큰 초과 삭제, 부록 A.2) |
| 3 중복 제거 | 기존 모두와 ROUGE-L < 0.7일 때만 새 지시 유지(§2.2) | Genetic Instruct code 중복 제거; 방법 미보고(§3.3.1) | 보고되지 않음 | 해법은 미보고; 질문 다양성만 연구(§2.2.4) |
| 4 검증 | 필터 외 없음; 품질 감사: 모든 field 유효 54%(§3.3) | 가능한 경우 정답·Python verifier; LLM judge 후 reward-model judge(§3.2.3) | 실행 검사 후 LLM semantic 검사(§3.2) | 32개 해법 majority vote, 최소 vote threshold 0(§3, 부록 C.1) |
| 5 선택 | 생존자 전부(§2.2) | reward model이 고품질 chosen 선택(§3.3.2) | 두 강한 generator를 공개에 사용(§4.2) | 해법 필터보다 질문 수 우선(§2.2.3~2.2.4) |
| 6 혼합 | 전체 세트, 2 epoch(§4.1) | Code SFT 약 800K → General SFT 200K(2% code replay) → DPO 160K → RPO 300K ×3(§3.3) | relevance detection 8,000개와 혼합(부록 B.3) | 수학만 SFT; 8B 14M 쌍, 70B 5M(§4) |
| 출력 | 지시 52,445, instance 82,439(표 1) | 합성 alignment 데이터 98% 초과(§3.2) | 공개 샘플 약 60,000(§4.2) | 쌍 13.97M, 질문 607.3K(표 5) |

표에서 세 관찰이 나온다. 첫째, 네 pipeline 중 held-out split을 데이터 결정에 사용한다고 보고한 것은 OpenMathInstruct-2 하나뿐이며 대상 domain의 split이지 다른 능력은 아니다. 둘째, “verify”는 서로 다른 연산이다. APIGen은 실행, OpenMathInstruct-2는 같은 teacher의 샘플 간 일치, Nemotron-4는 학습 scorer를 사용한다. 검출할 수 있는 오류가 다르다. 셋째, 평가 폭은 verifier를 따른다. 가장 강한 검사를 가진 APIGen과 OpenMathInstruct-2도 검사한 domain만 평가한다.

## §3 생성과 필터: 단계별 yield

**정의.** Pipeline의 yield는 생성된 샘플 중 모든 검사를 통과한 비율이다. 검사 순서에서는 조건부 통과율의 곱이다.

**공식.**

    yield = N_verified / N_generated = Π_s (1 − f_s)

- N_generated: generator에 요청한 샘플 수.
- N_verified: 모든 검사를 통과한 샘플 수.
- f_s: 검사 s에 도달한 샘플 중 실패한 비율(조건부 실패율).

**계산 예시([[apigen]] 표 1).** DeepSeek-Coder-33B-Inst는 40,000개를 생성했다. 형식 실패 4,311개, 실행 실패 15,496개, semantic 검사 실패 6,424개, 통과 13,769개다(합계 40,000).

1. 형식: f₁ = 4,311 / 40,000 = 10.8%; 35,689개 남음.
2. 실행: f₂ = 15,496 / 35,689 = 43.4%; 20,193개 남음.
3. Semantic: f₃ = 6,424 / 20,193 = 31.8%; 13,769개 남음.
4. Yield = 0.892 × 0.566 × 0.682 = 0.344이며 인쇄된 34.42%와 일치한다.

같은 검사에서 DeepSeek-V2-Chat은 형식 817개, 실행 3,359개, semantic 2,165개가 실패하고 33,659개가 통과해 yield 84.15%다. 검증 샘플 10,000개를 얻으려면 첫 generator는 10,000 / 0.3442 = 29,053회, 두 번째는 10,000 / 0.8415 = 11,884회 생성해야 한다. Generator 선택은 비용과 어느 검사가 일을 맡는지를 모두 바꾼다. 약한 generator에서는 대부분의 거부가 실행 단계에서 생긴다.

**Generator 품질의 근거.** Self-Instruct의 감사된 모든 field 유효율 54%는 base model(GPT-3 davinci)에서 나왔다([[self-instruct]] 표 2). Tülu 1이 그 데이터셋으로 LLaMA 13B를 학습했을 때 MMLU는 42.3에서 30.4로, 6개 평가 평균은 21.8로 떨어졌다. Human+GPT 혼합은 45.2였다([[tulu-1-how-far-can-camels-go]] 표 3). 저자들은 약한 generator 때문이라고 해석한다(§5.1).

**조건과 한계.** Acceptance rate는 generator-checker 쌍의 특성이다. 강한 generator의 높은 yield는 생존 데이터가 더 다양하다는 근거가 아니다. APIGen은 generator별 다양성 통계를 보고하지 않는다.

**시사점.** Generator와 task family별 단계 실패 수를 기록하라. 다른 family보다 yield가 낮은 family는 generator가 처리하기 어렵거나 prompt가 잘못 지정된 것이며, 어느 경우든 5단계가 보정하지 않으면 최종 데이터에서 그 비율이 줄어든다.

## §4 중복 제거: near-duplicate와 질문 다양성

**정의.** 중복 제거(deduplication)는 후보가 이미 받아들인 항목과의 유사도가 임계값을 넘으면 그 후보를 제거하는 절차다. Exact 방법과 MinHash 방법은 ch-12에서 다룬다. 여기서는 Self-Instruct가 사용한 instruction 수준 변형을 다룬다.

**공식(ROUGE-L, F-measure 형태).** ROUGE-L은 token의 최장 공통 부분수열(longest common subsequence, LCS)을 점수화한다. Self-Instruct 논문은 이 공식을 인쇄하지 않았다. 가중치를 동일하게 둔 표준 F-measure는 다음과 같다.

    P = LCS(a, b) / |a|,   R = LCS(a, b) / |b|,   ROUGE-L = 2PR / (P + R)

- a: 후보 instruction의 token들이다. b: 이미 받아들인 instruction의 token들이다. |·|는 token 수다.
- LCS: 두 token 열에 순서를 유지한 채 등장하는 가장 긴 부분수열의 길이다. 인접해서 등장할 필요는 없다.

**계산 예시.** a = "Write a short poem about the sea"(7 token), b = "Write a poem about the ocean"(6 token)이라고 하자. LCS는 "Write a poem about the"이며 길이는 5다. P = 5/7 = 0.714, R = 5/6 = 0.833, ROUGE-L = 2 × 0.595 / 1.547 = 0.769이다. 0.769 ≥ 0.7이므로 Self-Instruct는 이 후보를 추가하지 않는다([[self-instruct]] §2.2). "Compose verses describing waves at night"라는 후보는 b와 공유하는 token이 없으므로 추가한다. 그러나 두 후보는 같은 종류의 출력을 요청한다. 따라서 표면 중복 기준값은 어휘 반복을 제한할 뿐, task 반복을 제한하지는 않는다.

**다양성이 중요하다는 근거.** OpenMathInstruct-2는 pair 수를 256K로 고정하고 unique question 수를 바꾸었다. 1K question 조건은 6.5K question 조건보다 MATH validation에서 10점 이상 낮았다([[openmathinstruct-2]] §2.2.4, 그림 6). 이는 단일 연구의 결과이며, Llama3.1-8B-Base와 수학 domain에만 해당한다. FLAN의 cluster ablation(§1)은 같은 관찰을 task 수준에서 보인다.

**시사점.** 일반화를 측정하는 단위에서 다양성을 측정하라. 즉, token n-gram 수가 아니라 unique question 수 또는 task type 수를 측정해야 한다.

## §5 검증: 그 가치에 관해 근거가 말하는 것

**정의.** Verifier는 생성된 sample의 표면 형식만 보지 않고 실행 결과, 참조 정답, 독립적으로 생성한 여러 sample 사이의 일치, 또는 학습된 scorer 같은 추가 정보를 사용하여 sample을 correct 또는 incorrect로 라벨링한다.

**측정할 수 있는 문제.** 틀린 sample을 SFT target으로 사용하면 correct sample과 동일한 cross-entropy weight로 학습한다. 따라서 물어야 할 것은 틀린 sample이 accuracy를 얼마나 낮추는지, 그리고 그것을 제거해 얻는 이득이 verifier의 false rejection과 compute 비용을 감수할 만큼 큰지다.

**설정별 근거.**

1. **Tool call, 소형·중형 student.** APIGen의 검사를 실패한 sample을 training에 다시 넣자 BFCL overall accuracy가 xLAM-7B에서는 4.06점(+fail semantic)과 5.94점(+fail execution), xLAM-1B에서는 각각 9.59점과 12.17점 하락했다([[apigen]] §5.2, 그림 5). 논문은 execution arm에 semantic failure도 함께 들어가는지 밝히지 않는다. 결과(단일 연구).
2. **수학 reasoning trace, 8B student.** 64K~1024K pair에서 solution의 10%, 20%, 40%, 또는 80%를 wrong-answer solution으로 교체했을 때, 256K pair 이상에서는 20%까지 degradation이 거의 없거나 전혀 없었다([[openmathinstruct-2]] §2.2.3, 그림 5). Llama3.1-405B judge prompt 2개 또는 Nemotron-4-340B-Reward가 flag한 solution을 제거하면 128K set의 6%~12%가 빠졌고, accuracy는 unfiltered의 43.6 ± 1.7에 비해 43.0~43.8이었다(표 3). 사람이 직접 확인한 flagged solution 20개 중 약 60%만 실제로 incorrect였다(각주 4). 결과(단일 연구). 저자들은 이 설정에서 SFT에는 부정확한 filtering도 받아들일 수 있다고 결론 내린다.
3. **같은 수학 설정에서 generator 강도.** Coverage를 맞추면 Llama3.1-405B-Instruct 데이터는 37.9 ± 0.6, Llama3.1-8B-Base 데이터는 30.1 ± 0.6을 기록했다(표 2). 저자들의 preliminary analysis는 약한 model이 correct answer이지만 reasoning은 incorrect인 solution을 더 많이 생성하는 것이 그 차이와 관련 있다고 본다(§2.2.2). Final-answer check는 이 오류 유형을 검출하지 못한다.
4. **Instruction data의 output 품질.** Self-Instruct output을 InstructGPT-003 output으로 교체하자 top-rated response 비율이 44.4%에서 54.4%로 올랐다([[self-instruct]] §4.5, 그림 7). 이는 output을 filtering한 것이 아니라 다시 생성한 것이다.
5. **Preference label.** RewardBench Chat-Hard에서 judge로 사용한 Nemotron-4-340B-Reward의 점수는 0.87이었고 LLM judge는 0.54였으며, team은 이후 preference data에는 reward model로 전환했다([[nemotron-4-synthetic]] §3.2.3). Preference data에서는 label 자체가 training signal이므로 label accuracy는 filter를 고르는 문제가 아니라 데이터 content의 문제다.

**Gold answer가 없을 때 verifier로 쓰는 majority vote.** OpenMathInstruct-2는 sample한 solution 32개 가운데 가장 흔한 final answer를 reference로 취급한다([[openmathinstruct-2]] §3). 계산 예시: solution 32개 중 14개가 “12”, 10개가 “15”, 나머지 8개가 다른 답을 낸다면 reference는 14표를 받은 “12”이고, “12”로 끝나는 solution 14개를 유지한다. Minimum-vote threshold가 16이라면 이 question 전체를 버린다. 논문은 threshold 0, 8, 16, 24를 비교했고 각각 381K, 339K, 254K, 160K pair를 유지했으며 accuracy는 50.1, 49.2, 44.4, 42.0이었다. Data size를 맞추지 않았으므로 이 하락에는 더 엄격한 agreement와 더 적은 data라는 두 요인이 함께 섞여 있다(부록 C.1, 표 9). Majority vote는 generator가 공유하는 error도 함께 공유한다. 대부분의 sample이 같은 실수를 하면 그 실수가 reference가 된다.

**조건과 한계(해석, 이 course).** 1번과 2번의 두 결과는 설정을 분리하면 함께 성립한다. APIGen의 failure는 malformed call 또는 실행되지 않는 call이고 target output은 정확한 string이며 student는 1.3B와 7B다. OpenMathInstruct-2가 주입한 error는 그 외에는 well-formed인 reasoning 안의 wrong final answer이고 data set은 적어도 256K pair이며 metric은 final-answer accuracy다. 두 논문 모두 틀린 data가 untargeted capability에 미치는 영향을 측정하지 않았다. 열린 질문은 SFT target의 wrong data에 대한 tolerance가 student size, error type, training stage에 따라 어떻게 달라지는가다. Preference optimization이나 RL에서는 wrong label이 update의 sign을 바꾸므로 특히 다를 수 있다.

**시사점.** Verifier를 신뢰하기 전에 verifier 자체를 측정하라. Accepted item과 rejected item을 표본으로 뽑아 라벨링하고 false-accept rate와 false-reject rate를 보고하라. Verifier가 비싸다면 먼저 일부 subset에서 OpenMathInstruct-2처럼 noise-injection ablation을 실행하라. 그 결과로 verifier가 비용을 들일 가치가 있는지 결정할 수 있다.

## §6 선택: 어떤 생존자를 유지할 것인가

**정의.** 선택(selection)은 명시적인 기준에 따라 검증을 통과한 sample 또는 source 중 일부만 고르는 단계다. 품질 점수, 다양성, gradient를 바탕으로 선택하는 방법은 ch-22에서 다룬다.

**네 pipeline에서의 작동 방식.**

1. Nemotron-4는 ground truth를 사용할 수 없을 때 reward model로 chosen response의 품질이 높은 preference example을 유지한다. 300K RPO 세트에는 더 완화한(“less harsh”) filter를 적용한다([[nemotron-4-synthetic]] §3.3.2).
2. APIGen은 generator 수준에서 선택한다. 공개한 약 60,000개 sample은 pass rate가 가장 높은 두 generator에서 나온 sample만으로 구성한다([[apigen]] §4.2, 표 1).
3. OpenMathInstruct-2는 question 수준에서 선택한다. question 하나당 solution 수를 늘리기보다 unique question 수를 늘리고, compute 제약 때문에 70B model에는 5M subset을 사용한다([[openmathinstruct-2]] §2.2.4, §4).
4. Self-Instruct는 살아남은 sample을 모두 유지한다. 252개 user task에서 측정한 data-size curve는 instruction 16K개 이후 거의 plateau에 도달한다([[self-instruct]] §4.5, 그림 7).

**시사점.** Generator별 선택이나 score별 선택은 score가 task type과 상관관계를 가질 때 분포의 전체 영역을 제거한다. 선택 전과 선택 후에 task-family histogram을 확인하라.

## §7 혼합: 폭을 결정하는 선택

**정의.** 혼합(mixing)은 어떤 데이터셋을 함께 결합할지, 각각을 어느 양으로 넣을지, 그리고 어느 학습 단계에서 넣을지를 결정한다.

**측정 가능한 문제.** 특정 skill을 겨냥한 각 데이터셋은 여러 capability를 동시에 바꾸며, 변화 방향은 상승과 하락 모두일 수 있다.

**근거.**

1. **모든 평가에서 가장 좋은 단일 데이터셋은 없다.** LLaMA 13B에 하나의 recipe를 적용한 실험에서 Human+GPT 혼합은 평균 45.2로 가장 높았지만, 모든 평가에서 최고 점수는 아니었다. ShareGPT는 AlpacaEval win rate가 70.5%로 가장 높았으나 TyDiQA는 43.2에서 30.5로 낮췄다. 12개의 단일 데이터셋 중 6개는 GSM을, 8개는 TyDiQA를 base model보다 낮췄다([[tulu-1-how-far-can-camels-go]] §5.1, 표 3). 결과(단일 연구).
2. **Ablation은 여러 skill을 함께 바꾼다.** Tülu 3에서 math data를 제거하면 GSM8K는 76.2에서 64.1로, MATH는 31.5에서 23.5로 낮아졌다. 또한 unseen DeepMind Mathematics는 32.3에서 23.3으로, BigCodeBench는 11.5에서 8.8로 낮아졌다([[tulu-3]] 표 10, 32). Safety data를 제거하면 safety 평균은 93.1에서 74.7로 낮아졌고, 다른 skill 대부분은 약 2점 이내에서 변했다(표 10). WildChat을 제거하면 AlpacaEval 2는 12.4에서 7.5로 낮아졌다(표 10).
3. **양과 단계가 중요하다.** GSM8K RFT, Code Alpaca, ShareGPT로 학습한 LLaMA model에서, 소량 data(1/256)에서는 mixed source가 각 능력별 individual source보다 그 능력의 점수를 더 높였지만, full size에서는 더 낮았다([[sft-data-composition-dmt]] §3.3, 그림 3). Specialized data를 먼저 학습하고 general data를 마지막에 학습하면 math accuracy가 사라졌다. 7B에서 GSM8K는 multi-task training이 47.53, mixed sequential training이 32.60이었다(표 1). 마지막 단계에 specialized data의 1/256을 다시 넣은 DMT는 7B에서 GSM8K 41.92, HumanEval 17.68, MT-Bench 6.08을 얻었다. k = 1/256일 때 이는 ShareGPT conversation 86,060개에 GSM8K RFT 430개와 Code Alpaca sample 78개를 추가한 것이다(부록 A, 표 2). 결과(단일 연구, single run).
4. **Production report에서의 stage 구성.** Nemotron-4는 모든 behaviour를 하나의 SFT stage에 섞으면 conflict가 발생하며, coding에서 그 충돌이 가장 강하고 reweighting으로도 해결되지 않는다고 보고했다. 그래서 Code SFT를 먼저 수행한 뒤 code sample의 2%를 포함한 General SFT를 수행했다([[nemotron-4-synthetic]] §3.3.1). General SFT는 MMLU를 72.2에서 78.3으로 올렸지만 HumanEval은 70.7에서 66.5로 낮췄다(표 6). 이 report는 단일-stage 결과 수치를 제공하지 않는다.

**조건과 한계.** 1~3번은 2023~2024년 데이터셋에서 7B~70B model에 SFT를 적용한 결과다. 4번은 통제 비교가 없는 340B model report다. Agentic data와 long-context 비율을 포함한 multi-skill mixture 설계는 ch-30b에서 다룬다.

**시사점.** 혼합을 하나의 실험으로 다루고 untargeted-capability report를 붙여라. Source를 하나 추가하거나 제거할 때마다 각 skill의 development benchmark와 unseen benchmark에서 어떤 변화가 생겼는지 기록해야 한다.

## §8 Verifier coverage bias

**정의.** Verifier coverage bias는 검사 비용이 싼 domain(수학 정답, code 실행, tool call)에서는 pipeline이 검증을 통과한 data를 더 많이 만들고, 검사 방법이 없는 domain(열린 형식의 글쓰기, 조언, 여러 문서의 종합)에서는 덜 만들기 때문에 training mixture에 생기는 편향이다.

**작동 방식.** 1번과 3번은 보고된 결과이며, 2번과 4번은 이 course의 해석이다.

1. Checker가 있는 곳에서는 verified yield가 더 높고, 이를 측정하는 비용도 더 낮다(§3, §5).
2. Team은 clean yield를 보고하는 pipeline을 확장하므로, 이 domain이 가장 빠르게 커진다.
3. 평가는 흔히 checker를 따라간다. APIGen은 BFCL만 평가하고, OpenMathInstruct-2는 수학만 평가한다([[apigen]] §5; [[openmathinstruct-2]] §4).
4. 따라서 target으로 삼지 않은 domain에서의 손실은 pipeline 자체의 측정값이 만들지도, 검출하지도 못한다.

**작동 예시([[transferability-of-llm-reasoning]] arXiv:2507.00432v2 표 1).** Qwen3-14B-Base를 하나의 math data set으로 두 가지 방식으로 학습했다. 하나는 rejection sampling으로 고른 Qwen3-32B chain-of-thought trace로 SFT를 수행한 방식이고, 다른 하나는 answer label만 사용해 RL을 수행한 방식이다(§2.2).

| Model | Math 평균 | Other reasoning 평균 | Non-reasoning 평균 |
|---|---|---|---|
| Qwen3-14B-Base | 27.7 | 30.2 | 45.7 |
| Math trace SFT(think) | 49.8 (+22.1) | 45.3 (+15.1) | 21.1 (−24.6) |
| Math answer RL | 53.8 (+26.1) | 60.0 (+29.8) | 53.2 (+7.5) |

Non-reasoning group은 CoQA, MC-TACO, IFEval, HalluEval이다. IFEval은 SFT 뒤 69.2에서 42.3으로 떨어졌다. 저자들은 이 차이를 SFT에서 더 큰 representation shift와 output-distribution shift가 일어났기 때문이라고 설명한다(해석, §3–4). 결과(단일 연구, 하나의 base model).

**조건과 한계.** 이 비교에는 objective와 token의 출처(teacher trace 대 model 자신의 sample)라는 두 차이가 함께 섞여 있다. ch-38a가 두 차이를 분리한다. Tülu 1도 2023년 data set에서 같은 pattern의 SFT 쪽 결과를 보고한다(§7, 항목 1).

**시사점.** Pipeline이 verifiable domain을 확장할 때는 그 domain 밖의 evaluation group을 go/no-go gate에 넣어야 한다. 또한 6단계에서는 non-verifiable data, 즉 사람이 쓴 data, judge가 점수화한 data, 실제 user prompt를 위한 예산을 명시적으로 배정해야 한다.

## §9 긴 context 및 agentic data로 loop 확장하기

**긴 context.** 이 loop는 긴 context에도 적용되지만, 합성 long-instruction data에 관한 근거는 엇갈린다. ProLong은 Llama-3-8B-Instruct가 생성한 합성 long QA, RAG, summarization data를 SFT에 전체 token의 0%, 1%, 3%, 10%, 50% 비율로 섞었다. 그때 long-context 평균은 각각 55.7, 54.1, 53.5, 53.9, 43.3이었다. 즉 이 실험에서는 합성 long-context instruction data의 비율이 늘수록 평균이 개선되지 않았고 50%에서는 크게 낮아졌다. Generator를 70B로 바꿔도 결론은 달라지지 않았다([[prolong]] §5, 표 8, 부록 B.5). 저자들은 앞선 긍정적 결과가 이 실험보다 long-context continued training을 덜 했거나, short instruction set이 더 컸을 가능성을 가설로 제시한다(해석). Long-context synthesis는 ch-28과 ch-29a에서 다룬다.

**Agentic trajectory.** APIGen-MT는 대화를 생성하기 전에 검증을 수행한다. 먼저 execution, policy unit test, LLM committee로 task blueprint를 검증한다. 그 다음 user-agent 대화를 simulation하고, 마지막 environment state와 output이 blueprint와 일치하는 trajectory만 남긴다([[apigen-mt]] §4.1–4.2). Blueprint 생성 중 agentic feedback을 사용하자 configuration success는 28%에서 70%로 올랐고, simulation한 trajectory 중 67%가 성공했다(그림 4). Agentic data synthesis는 ch-27, ch-29c, ch-29d에서 다룬다.

## Negative sample과 negative feedback

이 절은 course 표준에서 말하는 “negative”의 네 가지 뜻, 즉 negative marginal value, content로서의 negative, conditioning으로서의 negative, gradient로서의 negative를 사용한다.

1. **Negative는 어디에서 생기는가.** Format·execution·semantic check failure(APIGen), majority vote에서 소수인 answer(OpenMathInstruct-2), ground truth·Python verifier·LLM judge·reward model이 고른 rejected response(Nemotron-4), refusal 및 “impossible request” prompt(Nemotron-4, APIGen)가 있다. 네 pipeline 어느 것도 false-negative rate를 보고하지 않는다. OpenMathInstruct-2가 flag된 solution 20개를 확인했을 때 약 60%가 실제로 incorrect였으므로, 이 작은 표본에서는 약 40%가 false rejection이었다([[openmathinstruct-2]] 각주 4).
2. **Pipeline이 이를 어떻게 다루는가.**
   - Negative marginal value: 버린다. APIGen check failure(§3.2)와 filter된 Self-Instruct generation이 여기에 속한다.
   - Content로서의 negative: tool이 답할 수 없을 때 empty call 또는 refusal을 답하도록 만든 APIGen relevance-detection target 8,000개([[apigen]] 부록 B.3), model이 수행할 수 없는 task에 대한 Nemotron-4 refusal response([[nemotron-4-synthetic]] §3.2.5)가 있다. 이들은 보통의 cross-entropy target이다.
   - Conditioning으로서의 negative: 네 pipeline 중 이를 쓰는 pipeline은 없다(ch-31a).
   - Gradient로서의 negative: Nemotron-4의 DPO와 RPO rejected response가 있다(§3.3.2).
3. **Gradient로 쓰는 negative의 작동 방식.** Logit \(z\) 위의 softmax에서 token \(y\)의 log-probability gradient는 다음과 같다.

       ∂ log p_y / ∂ z_j = 1[j = y] − p_j

   - \(p_j\): token \(j\)의 확률.
   - \(1[j = y]\): 점수화하는 token이 \(j\)일 때 1, 그 외에는 0인 indicator.

   \(\log p_y\)를 낮추면 logit은 \(−(1[j = y] − p_j)\) 방향으로 움직인다. 따라서 \(y\) 이외의 모든 token \(j\)는 현재 확률 \(p_j\)에 비례해 올라간다. 계산 예시로 \(p=(0.7, 0.2, 0.1)\)이고 reject한 token이 세 번째라면 logit update 방향은 \((0.7, 0.2, −0.9)\)다. Unit step마다 가장 확률이 높은 token의 logit은 0.7, 두 번째 token의 logit은 0.2 상승한다. 즉 세 번째 token에서 뺀 확률 대부분은 첫 번째 token으로 이동한다. 이미 확률이 낮은 sample을 아래로 밀면 가장 가능성 높은 대안이 올라가는데, 그 대안도 틀릴 수 있다. Derivation과 sequence-level effect는 ch-43a에서 다룬다.
4. **근거.** Negative를 버리는 것은 APIGen에서는 도움이 되었다. 이를 다시 넣자 BFCL이 4.06~12.17점 하락했다(그림 5). 그러나 OpenMathInstruct-2에서는 도움이 되지 않았다(표 3). Gradient로서의 negative에 대해서 Nemotron-4는 DPO에서 chosen response와 rejected response의 likelihood가 모두 낮아졌지만 둘의 gap은 커졌고, 더 오래 학습하면 overfit한다고 관찰했다. DPO는 MT-Bench를 7.99에서 7.90으로 낮췄다([[nemotron-4-synthetic]] §3.3.2, 표 6). 이 장의 어느 source도 improvement 중 얼마가 negative에서 왔는지 측정하지 않았다. 따라서 이 장은 negative가 어떤 gain의 주된 원인이라는 주장을 하지 않는다.
5. **Control.** Nemotron-4는 chosen response에 가중 SFT loss, 즉 positive NLL을 anchor로 남기는 항을 추가했다. 또한 reward gap을 target으로 하는 RPO를 사용해 high-quality rejected response는 덜 아래로 밀리게 했다(§3.3.2). West-of-N은 N개 sample 중 최고와 최저를 pair로 만든다([[west-of-n]]). 그러나 worst-of-N response는 chosen response와 멀리 떨어진 쉬운 negative인 경우가 많으므로 near-miss error에 관한 signal은 약하다(해석). Narrow majority vote나 response 순서를 바꾼 judge가 서로 불일치하는 경우처럼 불확실한 label은 rejected response로 쓰지 않고 제외할 수 있다. Nemotron-4는 두 순서가 모두 일치한 LLM-judge pair만 유지했다(§3.2.3).
6. **Diagnostic.** Preference training 중 chosen과 rejected의 log-probability를 따로 기록하라. Generator별·check별 failure count를 기록하라. Labelled sample에서 false-accept rate와 false-reject rate를 보고하라. Refusal content를 추가한 뒤 benign prompt에서 refusal rate를 보고하라.
7. **Generality에 미치는 영향.** Refusal content는 과도하게 일반화할 수 있다. Tülu 1은 Tülu 13B의 ToxiGen toxic generation이 0.1%였다고 보고하며, 저자들은 refusal behaviour에 overfit했을 가능성을 가설로 제시한다(해석) ([[tulu-1-how-far-can-camels-go]] §5.3). Tülu 3은 over-refusal을 막는 데 contrastive prompt가 유용하다고 발견했다([[tulu-3]] §4.2). Nemotron-4의 DPO training에서는 한 metric(예: MT-Bench)의 향상이 보통 다른 metric(예: 0-shot MMLU)의 degradation과 함께 나타났다(§3.3.2). Over-refusal 평가는 ch-52에서 다룬다.

## 레시피

이 표의 행은 card와 primary source에서 이 장과 관련 있는 setting을 옮긴 것이다. “SFT (data)”는 data-generation setting을 뜻한다.

| Model(정확한 release) | Size | Stage | Setting | Value | Source 위치 | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Self-Instruct data(GPT-3 davinci generator) | 175B | SFT (data) | seed task; step당 in-context instruction | 175; 8(사람 작성 6, 생성 2) | arXiv:2212.10560 §2.2 | verified 2026-09-15 | ablation 보고 없음 |
| Self-Instruct data | — | SFT (data) | diversity filter | 기존의 모든 instruction과 ROUGE-L < 0.7일 때만 instruction 추가 | arXiv:2212.10560 §2.2 | verified 2026-09-15 | ablation 보고 없음 |
| GPT3-Self-Inst | 175B | SFT | epoch; prompt loss weight | 2; 0(그 외 OpenAI API default) | arXiv:2212.10560 §4.1, 부록 A.3 | verified 2026-09-15 | ablation 보고 없음 |
| APIGen generator(4개 open model) | — | SFT (data) | temperature; generator당 목표 sample | 0.7; 40,000 | arXiv:2406.18518 §4.2 | verified 2026-09-14 | ablation 보고 없음 |
| xLAM-7B (FC); xLAM-1B (FC) | 7B; 1.3B | SFT | relevance-detection example; peak LR; epoch | 8,000; 5e-6 cosine, warmup 50 step; 4 | arXiv:2406.18518 부록 B.3 | verified 2026-09-14 | ablation 보고 없음 |
| OpenMathInstruct-2(Llama3.1-405B-Instruct teacher) | 405B | distill-SFT (data) | 새 question당 solution; temperature; answer rule | 32; 0.7; majority vote, minimum-vote threshold 0 | arXiv:2410.01560v2 §3; 부록 C.1 | verified 2026-09-14 | 표 9: threshold 0/8/16/24 → 50.1/49.2/44.4/42.0(data size 미일치) |
| OpenMathInstruct-2 | — | distill-SFT (data) | decontamination | embedding top-5 neighbour + 405B paraphrase check를 양쪽 순서로 수행; question 569K → 519K | arXiv:2410.01560v2 §3.1, 부록 C.2 | verified 2026-09-14 | 표 10: n-gram matching이 놓친 example |
| OpenMath2-Llama3.1-8B | 8B | SFT | data; LR; epoch; batch(unit 미표기) | 13.97M pair; constant 2e-5; 2; 512 | arXiv:2410.01560v2 §4 | verified 2026-09-14 | 그림 1: 1M/2M/5M/14M에서 gain, saturation 없음 |
| Nemotron-4-340B-Instruct | 340B | SFT | Code SFT: sample; epoch; LR; global batch | 약 800K; 1; constant 3e-7; 128 | arXiv:2406.11704v2 §3.3.1 | verified 2026-09-14 | 표 6: HumanEval 57.3 → 70.7 |
| Nemotron-4-340B-Instruct | 340B | SFT | General SFT: sample; code replay; epoch | 200K; Code SFT sample의 2%; 3 | arXiv:2406.11704v2 §3.3.1 | verified 2026-09-14 | replay ablation 보고 없음 |
| Nemotron-4-340B-Instruct | 340B | preference | DPO pair; RPO pair; RPO iteration | 160K; 300K; 3 | arXiv:2406.11704v2 §3.3.2 | verified 2026-09-14 | 표 6의 stage별 metric |
| Tülu 7B, 13B(Human+GPT mix) | 7B, 13B | SFT | mixing; epoch; peak LR | sampling weight 없이 7개 data set concatenate(490,694 instance, derived); 2; 2e-5 | arXiv:2306.04751v2 §3.3, 표 1, 부록 D | verified 2026-09-14 | 표 3: mix 평균 45.2, 최고 단일 set 42.0 |
| LLaMA 7B/13B/33B(DMT) | 7B-33B | SFT | stage-2 specialized fraction k; epoch; peak LR | 1/256; 3; 2e-5, warmup 3% | arXiv:2310.05492v4 §3.5, 부록 C | verified 2026-09-15 | 표 1: DMT 대 세 가지 다른 strategy, single run |
| LLaMA 7B/13B/33B(모든 run) | 7B-33B | SFT | batch size(unit 미표기) | 16(§3.1); 128(부록 C) | arXiv:2310.05492v4 §3.1, 부록 C | conflict | 논문에 두 값이 모두 있고, 공개 config는 확인하지 않음 |
| Tülu 3 SFT mix | 8B, 70B | eval-gate | decontamination rule | 8-gram match; test instance token의 50% 초과가 일치하면 matched; eval instance의 2% 초과가 match되면 training set contaminated | arXiv:2411.15124v5 §3.2 | verified 2026-09-15 | threshold ablation 보고 없음; 그림 3: decontamination round가 “small drop” 유발 |
| FLAN 137B | 137B | SFT | dataset당 example; step; batch; LR | cap 30k, mixing-rate 최대 3k; 30k step; 8,192 token; 3e-5 Adafactor | arXiv:2109.01652v5 §2.4 | verified 2026-09-15 | ablation 보고 없음 |

**Small general-purpose run의 출발점.** Instruction-data pipeline에서 verified 행은 명시한 조건 아래 다음 default를 뒷받침한다. 새 instruction에 대한 ROUGE-L < 0.7 규칙은 instruction 52K개를 사용한 Self-Instruct의 GPT-3 175B run에서 왔다. Tülu 3의 8-gram 규칙(test token의 50% 초과, evaluation의 2% 초과)은 8B 및 70B SFT mixture에 적용되었다. Gold answer가 없는 math question에서는 temperature 0.7에서 32개 sample을 내고 majority-vote answer를 쓰는 방법이 405B teacher 및 8B~70B student에서 나왔다. Specialized data를 general data보다 먼저 학습할 때 general stage에 그 data의 1/256을 다시 넣는 방법은 LLaMA 7B~33B에서 나왔다. 이 행들 중 어느 것도 untargeted capability ablation을 포함하지 않으므로, 재사용할 때마다 §1의 held-out check가 필요하다.

## 일반화 관점

**(a) 폭을 넓히는 요인.**
- Instruction tuning에서 task cluster 수를 늘리자, 137B에서 held-out 평균이 49.9에서 63.5로 상승했다([[flan]] §4.1, 그림 6).
- 사람 생성 data와 model 생성 data를 섞은 mixture는 여섯 capability evaluation 전체에서 가장 좋은 평균을 냈다([[tulu-1-how-far-can-camels-go]] 표 3).
- Pair budget을 고정한 채 unique question 수를 늘리자 MATH validation이 10점 이상 올랐다([[openmathinstruct-2]] 그림 6). 이는 한 domain 안에서의 breadth다.
- Diverse한 실제 user chat data(WildChat)는 Tülu 3의 skill 대부분을 소폭 올리고 AlpacaEval 2를 4.9점 올렸다([[tulu-3]] 표 10).

**(b) Narrowing 또는 forgetting을 일으키는 요인.**
- Teacher trace만 쓴 math-only SFT는 Qwen3-14B-Base의 non-reasoning 평균을 45.7에서 21.1로 낮췄다([[transferability-of-llm-reasoning]] 표 1).
- Specialized data를 먼저, general data를 나중에 순차 학습하자 7B GSM8K가 47.53(multi-task)에서 32.60으로 낮아졌다([[sft-data-composition-dmt]] 표 1).
- 340B에서 Code SFT 뒤 General SFT를 수행하자 HumanEval이 70.7에서 66.5로 낮아졌다([[nemotron-4-synthetic]] 표 6).
- FLAN에서 instruction tuning은 8B 이하 model의 held-out performance를 낮췄다([[flan]] 그림 7).
- 특정 skill을 겨냥한 data는 development metric을 올리면서 unseen metric을 올리지 못할 수 있다. Persona IF data는 IFEval을 53.6에서 72.8로 올렸지만 IFEval-OOD는 18.0에서 17.6으로 낮췄다([[tulu-3]] 표 32).

**(c) 이 단계에서 측정하는 방법.**
- Data를 생성하기 전에 skill별 development suite와 unseen suite를 고정하고, 둘 모두를 대상으로 decontamination하라([[tulu-3]] §2.2, §3.2).
- 학습한 task의 held-out item이 아니라 held-out task cluster를 보고하라([[flan]] §2.2).
- Data mixture를 바꿀 때마다 각 skill의 development score와 unseen score, 그리고 pipeline이 검증한 domain 밖 group을 적어도 하나 보고하라.
- 알려진 측정 오류도 있다. GPT-4 preference win rate는 response의 unique token 수와 Pearson (r=0.96)으로 상관한다([[tulu-1-how-far-can-camels-go]] §5.4). N-gram decontamination은 paraphrase를 놓친다([[tulu-3]] §3.2). Closed model은 evaluation suite로 이미 학습했을 수 있다(§7).

## 흔한 실수와 발견 방법

| 실수 | 관찰 가능한 징후 | 확인 방법 |
|---|---|---|
| Data 작업 전에 held-out split을 고정하지 않음 | Data iteration마다 development score는 오르지만 unseen 또는 새 benchmark score는 그대로임 | 모든 data-ablation checkpoint를 unseen suite에서 채점한다([[tulu-3]] 표 32) |
| Exact match 또는 n-gram match만으로 decontamination함 | 생성 question에 paraphrase가 있을 수 있는 benchmark에서 비현실적으로 높은 점수 | Sample에 embedding retrieval + LLM paraphrase check를 수행한다([[openmathinstruct-2]] §3.1) |
| Total yield만 보고함 | 특정 task family가 눈에 띄는 오류 없이 최종 data에서 사라짐 | Generator와 task family별 check failure count를 낸다([[apigen]] 표 1) |
| 표면적 deduplication을 diversity로 간주함 | Accepted item 다수가 말만 다르고 같은 task를 요청함 | N-gram novelty가 아니라 unique task type 또는 question 수를 센다([[openmathinstruct-2]] 그림 6) |
| Verifier를 측정하지 않은 채 신뢰함 | Filtering이 data를 없애는데 metric은 변하지 않음 | Accepted/rejected sample에 label을 붙이고 noise-injection ablation을 한다([[openmathinstruct-2]] 표 3, 그림 5) |
| Majority vote를 ground truth로 간주함 | 특정 question family의 일관된 오답이 filtering을 통과함 | Labelled subset에서 majority answer를 gold answer와 비교한다 |
| Verifier coverage bias | Math, code, tool metric은 오르는데 instruction following 또는 chat metric은 낮아짐 | Gate에 verified domain 밖 evaluation group을 넣는다([[transferability-of-llm-reasoning]] 표 1) |
| 이전 stage에서 specialized data를 학습했지만 replay하지 않음 | General SFT stage 뒤 specialized metric이 낮아짐 | Stage마다 평가하고 replay fraction을 추가한다([[sft-data-composition-dmt]] 표 1; [[nemotron-4-synthetic]] §3.3.1) |
| Win-rate evaluation만 general metric으로 사용함 | 더 길고 diverse한 response가 높은 점수를 받지만 benchmark는 하락함 | Pair win rate에 benchmark suite를 함께 둔다([[tulu-1-how-far-can-camels-go]] §5.4) |
| Easy worst-of-N response를 rejected sample로 씀 | Rejected log-probability는 빨리 낮아지지만 chosen log-probability도 함께 낮아짐 | Chosen 및 rejected log-probability를 분리해 기록한다([[nemotron-4-synthetic]] §3.3.2) |

## 이해 확인

1. APIGen의 add-back ablation은 1B model에서 7B model보다 두 배가 넘는 BFCL point 손실을 냈다. 이 size dependence를 설명할 수 있는 mechanism 두 가지를 들고, 둘을 분리하는 experiment를 설명하라.
2. OpenMathInstruct-2는 wrong-answer solution 20%를 견뎠지만 weak-teacher data는 7.8점 더 낮았다. Final-answer check가 검출할 수 있는 error와 검출할 수 없는 error를 사용해, 두 결과가 동시에 참일 수 있는 이유를 설명하라.
3. Tülu 3의 Persona IF data는 IFEval을 19.2점 올렸지만 IFEval-OOD는 올리지 못했다. 이는 benchmark와 비교하여 data가 어떻게 생성되었음을 뜻하는가? Generation step을 어떻게 바꾸겠는가?
4. 같은 teacher의 sample들을 majority vote하는 방법이 systematic error를 고칠 수 없는 이유를 설명하고, 이 장의 verifier signal 중 하나를 이름으로 제시하라.
5. Transferability study에서 SFT와 RL model은 같은 math question을 사용했다. 그런데 non-reasoning score는 왜 반대 방향으로 움직였는가? 답변 중 어느 부분이 interpretation인지 표시하라.
6. FLAN은 학습한 task의 held-out example 대신 왜 task cluster 전체를 hold out하는가? 두 번째 설계는 generality 측정에 어떤 error를 넣는가?
7. DMT는 마지막 stage에서 specialized data의 1/256만 다시 넣는다. 같은 논문의 scaling result를 사용해, 이처럼 작은 fraction이 recall에는 충분할 수 있고 더 큰 mixture는 왜 충돌하는지 설명하라.
8. Pipeline이 reward model의 response 16개 중 최고와 최저를 chosen/rejected로 pair한다. Softmax gradient를 사용해 쉬운 rejected response가 update에 무엇을 기여하는지, 그리고 near-miss negative가 더 많은 정보를 가질 수 있는 이유를 설명하라.

## 연결

- 의존하는 장: [[ch-00]] *What General Capability Means and How It Is Measured*(held-out suite); [[ch-10a]] *Model-Based Quality Filtering and Benchmark-Targeted Data Selection*(filter 단계); [[ch-12]] *Deduplication: Exact, Near-Duplicate, and Semantic*(deduplicate 단계).
- Course 순서상 이전 장: [[ch-32f]] *Lab: Annealing and Context Extension with a Short-Context Regression Gate*.
- 다음 장: [[ch-19]] *Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing*(1단계를 상세히 다룸).
- Loop의 이후 단계: [[ch-22]] *Quality, Diversity, and Gradient-Based Data Selection*(5단계); [[ch-23]] *Model Collapse and Verification of Synthetic Data*(반복 생성); [[ch-30b]] *Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares*(6단계).
- Loop를 재사용하는 domain 장: [[ch-24]] *Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data*; [[ch-26]] *Tool and Function-Calling Data*; [[ch-27]] *Agentic Trajectory Data*; [[ch-28]] *Long-Context Data Synthesis and Synthetic Evaluation Task Families*.
- 측정과 negative: [[ch-29e]] *Instruction Tuning and Generalization to Unseen Tasks*; [[ch-30a]] *Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control*; [[ch-31a]] *Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood*; [[ch-43a]] *Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages*; [[ch-48]] *Contamination Detection and Its Effect on Reported Scores*; [[ch-49]] *Judge Models: Bias, Calibration, and Judge-Specific Overfitting*; [[ch-45]] *Self-Improvement Loops and Multi-Stage Reasoning Pipelines*.

## 출처

- [[self-instruct]] — Seed 기반 생성, ROUGE-L filter, quality audit(표 2), unseen SuperNI 및 user-task 평가, data size와 output quality curve(그림 7). 수치는 2026-09-15에 arXiv:2212.10560에서 읽음.
- [[nemotron-4-synthetic]] — 합성 data 비율, generator iteration, preference label judge, replay가 포함된 단계별 SFT, DPO/RPO 관찰, 표 6의 stage effect.
- [[nemotron-4-synthetic-recipe]] — Recipe 표의 SFT, DPO, RPO setting.
- [[apigen]] — 세 가지 check function-calling pipeline, generator별 yield(표 1), add-back ablation(그림 5), relevance-detection target.
- [[openmathinstruct-2]] — Teacher ablation, noise tolerance, filtering ablation, question diversity, majority vote, LLM decontamination.
- [[openmathinstruct-2-recipe]] — Recipe 표의 generation 및 SFT setting.
- [[tulu-1-how-far-can-camels-go]] — Data set별 capability trade-off, win-rate bias, refusal over-generalization.
- [[sft-data-composition-dmt]] — Data 양 대 ratio, training-strategy 비교, DMT(장 발췌문; arXiv:2310.05492v4).
- [[tulu-3]] — Development 대 unseen suite, 8-gram decontamination rule, SFT data ablation(표 10, 31, 32). 수치는 2026-09-15에 arXiv:2411.15124v5에서 읽음.
- [[flan]] — Held-out task cluster, cluster-count 및 scale ablation(장 발췌문; arXiv:2109.01652v5).
- [[transferability-of-llm-reasoning]] — Math-only SFT 대 RL transfer(표 1). 수치는 2026-09-15에 arXiv:2507.00432v2에서 읽음.
- [[prolong]] — SFT에서의 합성 long instruction data(표 8).
- [[apigen-mt]] — Trajectory simulation 전에 task blueprint를 검증함.
- [[west-of-n]] — Best-vs-worst-of-N 합성 preference pair(앞선 설명의 정정에만 사용).
- [[nathan-lambert-synthetic-data]] — 합성 fine-tuning에 대한 practitioner 의견(앞선 인용의 정정에만 사용).
