<!-- chapter: ch-20
     track: synthetic
     kind: content
     title: 데이터를 통한 증류: 설명 추론 과정과 R1-Distill 계보
     deps: [ch-19]
     sources: [[orca]], [[orca-2]], [[distilling-step-by-step]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[deepseek-r1-distill-synth]], [[bespoke-stratos]], [[sky-t1]], [[openr1]], [[openr1-recipe]], [[open-thoughts]], [[scaling-reasoning-losing-control-mathif]], [[scaling-reasoning-losing-control-mathif-recipe]], [[transferability-of-llm-reasoning]], [[gemma-2]], [[qwen-3]], [[qwen-qwq-traces]], [[dolphin]], [[qwen-2.5]], [[anthropic-distillation-attacks-report]], [[teacher-output-licenses]]
     figures: figures/distill-lineage.html
     revised: 2026-09 (generality revision)
-->

# Chapter 20 — 데이터를 통한 증류: 설명 추론 과정과 R1-Distill 계보

> **핵심 통찰.** 필터링한 교사 모델 출력을 대상으로 하는 지도 미세조정(SFT)은, 샘플링한 프롬프트 분포에서의 교사 행동을 학생에게 옮긴다. 여기에는 출력 형식과 추론 양식도 포함된다. 그 분포 안에서는, 같은 베이스 모델에서 시작한 RL보다 전이가 더 클 수 있다. 즉 R1에서 유래한 샘플 804,745개로 SFT한 DeepSeek-R1-Distill-Qwen-32B는 AIME 2024 pass@1 72.6에 도달한 반면, Qwen2.5-32B-Base에서 1만 step 이상 RL한 모델은 47.0에 도달했다([[deepseek-r1]], 표 16). 그 분포 밖에서는 전이가 더 약하고 음수가 될 수도 있다. 통제된 Qwen3-14B 연구에서 수학 문제 47K개에 대한 교사 추론 과정을 SFT했을 때 비추론 평균은 45.7에서 21.1로 떨어졌지만, 같은 문제에서 RL을 적용하면 53.2로 올랐다([[transferability-of-llm-reasoning]], 표 1).
>
> **지침.** 작은 모델의 추론 능력이 목표이고, 자기 출력으로 학습하는 것을 허용하는 라이선스를 가진 더 강한 교사가 있다면, 검증한 교사 추론 과정에 SFT를 사용하라. 32B에서는 이것이 베이스 모델에서 시작한 RL보다 보고된 모든 벤치마크에서 더 좋았기 때문이다(R1 표 16). 학생이 범용 모델로 남아야 한다면, 증류 세트에 비추론 데이터를 넣고 instruction following, 안전성, 비수학 과제를 전후로 측정하라. 통제 연구에서 수학 전용 추론 과정 SFT가 IFEval을 69.2에서 42.3으로 낮췄고(표 1), R1 자체 세트에는 일반 샘플이 177,812개 있기 때문이다(표 5). 교사를 고를 때는 각 후보의 출력으로 작은 학생을 학습시켜 보라. 더 높은 점수를 받은 DeepSeek-R1보다 QwQ-32B가 더 좋은 7B 학생을 만들었기 때문이다([[open-thoughts]], 표 8). 연산 자원이 약 17K개보다 많은 예제를 허용한다면 문제 풀과 문제당 답변 수를 늘려라. OpenThoughts3 점수는 데이터셋 크기에 따라 1M 규모까지 상승하고(그림 1; 부록 G, 그림 8), 저자 자체 평가에서 16,710개 예제의 Bespoke-Stratos-7B는 AIME 2024에서 R1-Distill-Qwen-7B보다 23.3점 낮기 때문이다([[bespoke-stratos]]).

## 학습한 버전에 대한 정정

git commit `4a72e54`의 버전에는 다음 오류가 있었다. 각 줄은 이전 주장, 정정된 설명, 출처 위치를 차례로 제시한다.

1. “2023–2025년의 증류는 logit matching이 아니다” → Gemma 2 2B와 9B는 교사의 전체 다음 토큰 분포에 대한 손실로 사전학습되며, Qwen3 소형 모델은 학생 logits을 교사 logits에 정렬하는 방식으로 미세조정된다([[gemma-2]], §3.2; [[qwen-3]], arXiv:2505.09388 §4.5).
2. 지침 “강한 학생에는 ~1M이 아니라 ~10–20K개의 정답 추론 과정을 유지하라” → OpenThoughts3 점수는 데이터 크기에 따라 1M 규모까지 상승한다([[open-thoughts]], 그림 1; 부록 G, 그림 8). 16,710개 예제의 Bespoke-Stratos-32B는 AIME 2024에서 63.3이고, R1-Distill-Qwen-32B의 보고 점수는 72.6이다([[bespoke-stratos]], 블로그 32B 표).
3. “R1이 모든 대표 평가에서 이기지만 QwQ-32B가 R1을 이긴다” → DeepSeek-R1은 모든 벤치마크가 아니라 CodeElo, GPQA-D, JEEBench에서 더 높은 점수를 받는다([[open-thoughts]], §4.6).
4. “19장이 자기 rollout에 대한 rejection-sampling fine-tuning을 다뤘다” → ch-19는 생성 방법을 다룬다. 모델 자신의 샘플에 대한 rejection sampling은 ch-31에서 가르친다.
5. Orca 시스템 메시지 16행 표가 “Appendix A에서 재구성됨” → 이 목록은 논문의 표 2이고 이전 표와 다르다. 예를 들어 “정확한 단계별 계획을 세운 뒤 그 계획을 실행하라” 또는 “세계적 수준의 전문가처럼 생각하라”를 요구하는 메시지는 없다(아래 §1은 표 2를 인용한다)([[orca]], 표 2).
6. Orca가 “FLAN-v2 / Big-Bench / Chain-of-Thought / GSM-style” 과제를 질의했다 → 질의는 FLAN-v2의 네 하위 컬렉션(CoT, NiV2, T0, Flan 2021)에서 왔다. Big-Bench 과제는 BBH가 평가 세트이므로 제외했다([[orca]], §3.1.2).
7. “Vicuna-13B의 44.0에서 Orca-13B의 49.7로 BigBench-Hard zero-shot 향상” → Vicuna-13B의 BBH 점수는 23.3, Orca-13B는 49.7, ChatGPT는 48.9이다([[orca]], 표 11).
8. “추론 과정이 풍부한 감독으로 학습한 학생은 늘 긴 추론 과정을 낸다(‘2+2 = ?’ → 400토큰)”를 Orca 2가 고친 문제로 제시 → 두 논문 모두 이를 보고하지 않는다. Orca 2는 작은 모델의 최선의 해결 전략이 교사의 전략과 다를 수 있다고 말한다([[orca-2]], 초록, §3).
9. Orca 2의 “다섯 목표 행동 모드(직접, 단계별, 설명 후 답변, 회상 후 답변, 추출 후 답변)” → 논문은 단계별, recall-then-generate, recall-reason-generate, extract-generate, direct-answer “및 기타”를 명명하고, Orca 성능에 “guided by”하여 과제별 전략을 고른다([[orca-2]], §1, §3).
10. Distilling Step-by-Step: “770M T5가 라벨 데이터의 80%만으로 4개 벤치마크에서 540B PaLM few-shot을 이긴다” → 80% 결과는 ANLI에만 해당한다. 100% 데이터에서는 이기는 최소 학생 크기가 e-SNLI 220M, ANLI·SVAMP 770M, CQA 11B이다([[distilling-step-by-step]], 초록, §4.2–4.3).
11. “3–8개의 in-context CoT 예시”와 “두 head가 T5 encoder를 공유한다”는 ‘논문에서 그대로 인용’된 블록 → 프롬프트는 약 10-shot이고, `[label]`, `[rationale]` 입력 접두사를 쓰는 단일 text-to-text 모델이며 별도 head가 없다. 해당 블록은 원문 인용도 아니었다([[distilling-step-by-step]], §3.2, 한계).
12. “rationale 감독이 토큰당 더 나은 학습 신호를 제공한다고 명시적으로 주장한 최초 논문” → 이 논문은 Magister et al. (2022), Ho et al. (2022)의 앞선 rationale-training 연구와 자신의 결과를 연결한다([[distilling-step-by-step]], §4.4).
13. R1 “V3 judge가 가독성과 정답성을 기준으로 필터링한다” → 규칙 기반 답 검사와, 데이터 일부에서 정답 참조와 대조하는 DeepSeek-V3 judge(Listing 4), 혼합 언어·긴 문단·코드 블록 CoT 제거, 프롬프트당 다수 샘플 중 정답만 보존을 사용한다([[deepseek-r1]], 보충 자료 B.3.3).
14. 800K 슬라이스 표(수학 약 200–300K, 코드 약 200–300K, 논리/과학 약 50–100K, 비추론 약 200K, SymPy와 unit-test verifier) → 표 5는 Math 395,285; Code 211,129; STEM 10,124; Logic 10,395; General 177,812; 총계 804,745이다([[deepseek-r1-recipe]], 보충 자료 B.3.3 표 5).
15. R1-Distill 학생 “Qwen2.5-Math 1.5B/7B/14B/32B, Llama-3.1-8B, Llama-3.3-70B” → Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, Llama-3.3-70B-Instruct이다([[deepseek-r1-recipe]], 보충 자료 B.4.3 표 6). 카드 [[deepseek-r1-distill-synth]]는 이전 오류를 반복한다.
16. “R1 추론 과정에 한 epoch SFT” → 모델별 초기 LR 1×10⁻⁴~2×10⁻⁵, 10분의 1까지 cosine decay, 32,768토큰 context, batch 64로 2–3 epoch이다([[deepseek-r1-recipe]], B.4.3).
17. “공개된 DeepSeek 문서는 … 소스별 구성도 judge prompt도 제공하지 않는다” → arXiv v2는 도메인별 표 5와 judge prompt(Listing 4)를 제공한다. 프롬프트당 샘플 수와 수율은 여전히 보고하지 않는다([[deepseek-r1]], B.3.3).
18. “helpfulness + harmlessness preference reward를 쓰는 Stage-2 Alignment RL” → 규칙, reward-model, 형식, 언어 reward를 결합했고, 1,700 step 중 preference reward는 마지막 400에서만 사용했다. 약 800K SFT는 DeepSeek-V3-Base에서 다시 시작한다([[deepseek-r1]], §3.2.2, 그림 2).
19. “ε = 10은 느슨하게 의도한 값이며, 빡빡한 clipping은 탐색을 파괴한다” → 명시된 이유는 낮은 ε가 많은 토큰의 gradient를 절단하고, 높은 ε가 불안정을 일으킬 수 있다는 것이다([[deepseek-r1]], §3.2.1).
20. 공개 재현 연구가 “800K 코퍼스를 17K–440K로 줄이고 … 공식 증류에서 2–3점 이내” → 코퍼스 크기와 격차는 아래 §4에 있다. Stratos-32B는 보고 점수와 비교하면 AIME 2024에서 R1-Distill-Qwen-32B보다 9.3점 낮다([[bespoke-stratos]]).
21. Bespoke-Stratos 프롬프트 풀 “수학 7K + 코드 5K + 과학 5K”, “T = 0.6, 최대 3회 재시도”, “SymPy”, “GPT-4o judge”, “MinHash”, “약 30–50% 거부율”, “중앙값 약 3K 토큰”, “학습 약 $4K” → 수학 10k(NuminaMATH AIME/MATH/Olympiads), 코드 5k(APPs, TACO), STILL-2 1k이다. 수학은 gpt-4o-mini judge로, 코드는 실행으로 필터링했다. 공개 스크립트에 과학 verifier는 없고, 데이터셋 카드는 $800, 학습은 8×H100에서 27시간을 제시한다. 나머지 값은 출처에 없다([[bespoke-stratos]], 블로그, 카드, `combine_data.py`).
22. Stratos-32B “LCB 약 57%”와 ablation “code-verification을 제거하면 LiveCodeBench 향상이 절반이 된다” → LiveCodeBench v2 All은 71.1이며 57.9는 Sky-T1-32B의 점수다. 블로그는 ablation을 보고하지 않는다([[bespoke-stratos]], 블로그 32B 표).
23. OpenR1 팀 “Tunstall, Beeching, Lambert, …” → 확인한 어느 페이지에도 Nathan Lambert는 목록에 없다([[openr1]], 검증).
24. OpenR1-Math “220K × 2 = 약 440K”, “T = 0.6”, “약 20% 거부”, “중앙값 약 5K”, “약 $10K”, “HF H100 + API 혼합” → NuminaMath 1.5 문제 400k개에 문제당 2개(일부 4개)를 생성하여 800k 추론 과정을 만들었고, 16k-token 제한, 로컬 512 H100, 공개 `default` 93,733행과 `extended` 131,396행을 사용했다. temperature, 거부율, 길이, 비용은 인쇄되지 않았다([[openr1]], Update #2, 데이터셋 카드).
25. “Qwen2.5-7B-Instruct”에서 OpenR1-Qwen-7B가 “MATH 약 80%, AIME24 약 40%(후속 GRPO로 AIME +3–5)” → 베이스는 Qwen2.5-Math-7B-Instruct이고, MATH-500 90.6, AIME24 36.7, AIME25 40.0이다. 확인한 출처에 GRPO 결과는 없다([[openr1]], Update #2 표).
26. Sky-T1 “수학 10K + 코드 5K + 과학 2K”, “T = 0.7, max 8K”, “SymPy”, 과학용 “GPT-4o-mini LLM-judge”, “GPT-4o로 재작성”, “약 5% 형식 실패” → 수학 10k, 코드 5k, 과학·퍼즐 STILL-2 1k이다. 수학은 exact matching, 코드는 unit test를 쓰며 judge는 없다. 추론 과정은 GPT-4o-mini가 재작성했다. temperature, 길이, 실패율은 제시하지 않는다([[sky-t1]], NovaSky 블로그 “Data Curation Process”).
27. Sky-T1이 “AIME에서 약 20점 뒤처짐(교사 ceiling: QwQ < R1)” → 43.3 대 72.6이므로 29.3점 차이다. 블로그는 원인이 교사라고 시험하지 않는다([[sky-t1]], 블로그 결과; [[deepseek-r1]], 표 15).
28. §4.1 “추가 783K 추론 과정이 AIME에서 약 2점을 산다”와 §4.2 “verifier가 상한을 결정한다 … Stratos는 가장 넓은 stack과 가장 강한 학생을 가진다” → 둘 다 뒷받침되지 않는다. Stratos에는 다중 verifier stack이 없고, 어느 출처도 verifier coverage로 학생을 비교하지 않는다.
29. QwQ > R1의 §4.3 설명(“더 가까운 분포”, “중앙값 3K 대 5K”, “중국어 누출이 작은 학생을 혼란시킴”)과 “Sky-T1은 비용 때문에 QwQ를 골랐다” → OpenThoughts는 원인을 시험하지 않은 채 결과만 보고한다. Sky-T1 블로그는 QwQ-32B-Preview를 open이고 o1-preview와 비슷하다고 설명하며 비용 이유를 제시하지 않는다([[open-thoughts]], §4.6; [[sky-t1]], 블로그).
30. §5.3 “Open-R1이 … 교사가 덜 포화되었기에 AIME25 향상이 AIME24보다 작다고 표시했다”와 §5.5 “[[openr1]]은 이를 명시적으로 지적했다: wrong-question-correctly” → 둘 다 Open R1 자료에 없다([[openr1]], 검증).
31. §5.5 “process reward(ch-24)가 유일한 방어다” → process supervision은 ch-44다. R1은 model-based PRM을 실패한 시도로 열거하고, V3 judge prompt는 이미 참조 답에 대해 추론 과정도 검사한다([[deepseek-r1]], G.2, Listing 4).
32. Dolphin의 다섯 단계 recipe(regex + classifier refusal detection) → 카드는 제거한 범주를 이름만 제시하며 방법이나 개수는 제공하지 않는다([[dolphin]], Technical Details).
33. §6 “어떤 answer-side filter도 전부 유지하는 것보다 낫지 않다” → 어떤 answer filter도 random baseline을 이기지 못했다. no-filtering run은 31,600개가 아니라 63,200개 예제를 사용했으므로 compute-controlled가 아니다([[open-thoughts]], §4.5 표 7).
34. §7 Llama-3 라이선스 행 “비Llama 모델 학습에 사용한 출력은 MAU 700M까지 허용된다” → Llama 3.1 model card는 라이선스가 synthetic data generation과 distillation을 포함하여 “다른 모델을 개선하기 위해” 출력을 사용하는 것을 허용한다고 적는다. 이전 행은 출력 사용 규칙과 사용자 수 조항을 섞었다([[teacher-output-licenses]], §2).
35. §7 “R1과 QwQ 출력만이 permissive license 아래 재배포할 수 있다” → Qwen2.5 모델은 3B와 72B를 제외하고 Apache 2.0이며, Llama 3.1은 증류를 위한 출력 사용을 허용한다([[teacher-output-licenses]], §2, §4).
36. Connections “ch-24(RLVR at scale)가 R1-Zero → R1 RL 경로를 다룬다” → ch-24는 reasoning-trace synthesis이고, group-baseline RL은 ch-40, verifiable reward는 ch-44다.

## 이 장이 범용 모델에 중요한 이유

**데이터로서의 증류(distillation as data)**란 교사 모델이 샘플링한 출력을 학생 모델의 SFT target으로 사용하는 것이다. 프롬프트 x에 대해 교사가 y ~ p_T(·|x)를 샘플링하고, 필터가 y를 유지하거나 버리며, 학생은 유지된 쌍에 대해 −log p_S(y|x)를 최소화한다. 이것은 sequence-level distillation이다. 학생은 하나의 샘플링된 시퀀스를 보며, 교사의 확률 분포는 보지 못한다. §9는 token-level 및 on-policy 변형을 다룬다.

사후학습 파이프라인(pre-training → mid-training → SFT → preference optimization → RL → evaluation)에서 증류 데이터는 주로 SFT에 들어간다. ch-35는 연구소들이 교사 데이터를 넣는 단계를 지도처럼 정리한다. 범용 모델에서 질문은 학생이 교사의 목표 벤치마크에서 이기는지에만 있지 않다. 측정할 질문은 다음과 같다.

1. 폭(breadth): 샘플링한 프롬프트 밖의 어떤 능력이 향상하는가? (§8, Generalization lens)
2. 협소화(narrowing): instruction following, 안전 행동, 비수학 과제처럼 어떤 능력이 떨어지는가? (§5, §8)
3. 측정(measurement): 교사 암기와 벤치마크 오염으로부터 전이를 어떻게 분리하는가? (§5.3)

## §1 Orca와 Orca 2 — 시스템 메시지로 끌어낸 설명 추론 과정

**정의.** Explanation tuning은 시스템 메시지, 사용자 질의, 교사 응답의 세 쌍 ⟨system message, user query, teacher response⟩으로 학생을 학습한다. 시스템 메시지는 교사에게 설명이나 단계별 답변을 요청한다([[orca]], §3.1).

**문제.** 이전 imitation 모델은 작고 동질적인 데이터의 짧은 답변으로 학습했다. GPT-4를 judge로 삼으면 Vicuna-13B는 ChatGPT의 92%에 도달하지만, 시험으로 측정하면 64%, BBH에서는 48%에 도달한다([[orca]], §1, 그림 1–3). 동시기의 Gudibande et al. 비판은 imitation 모델이 “imitation 데이터가 많이 지원하지 않는 과제에서는 base LM에서 ChatGPT까지의 격차를 거의 또는 전혀 메우지 못한다”고 보고한다([[orca]], Connections에서 인용).

**메커니즘.**
1. FLAN-v2 하위 컬렉션에서 zero-shot 질의를 샘플링한다. CoT 150K, NiV2 440K, Flan 2021 2.5M, T0 2M이며 Dialog은 0이다([[orca]], 표 3).
2. 손으로 작성한 시스템 메시지 16개 중 하나를 붙인다. 메시지 #8과 #10은 객관식 질문에만 쓴다(§3.1.1).
3. 1단계: ChatGPT가 5M개 질의 전체(FLAN-5M)에 답한다. 2단계: GPT-4가 무작위 1M개 부분집합(FLAN-1M)에 답한다. 논문은 ChatGPT 단계를 둔 이유를 두 가지 제시한다. 13B 학생과 GPT-4 사이의 capacity gap, 그리고 비용 및 rate limit이다(§3.1.3).
4. 각 단계에서 LLaMA-13B를 교사 토큰에만 loss를 적용해 4 epoch 학습한다(§3.2).

표 2에서 인용한 시스템 메시지 16개는 다음과 같다([[orca]]).

```
1  <빈 시스템 메시지>
2  당신은 AI 어시스턴트입니다. 사용자가 답을 이해하기 위해 외부에서 검색할 필요가 없도록 자세한 답을 제공하세요.
3  당신은 AI 어시스턴트입니다. 과제가 주어질 것입니다. 상세하고 긴 답변을 생성해야 합니다.
4  당신은 항상 설명을 제공하는 도움이 되는 어시스턴트입니다. 다섯 살 아이에게 답한다고 생각하세요.
5  당신은 지시를 매우 잘 따르는 AI 어시스턴트입니다. 가능한 한 많이 도와주세요.
6  당신은 사람들이 정보를 찾도록 돕는 AI 어시스턴트입니다. 사용자가 답을 이해하기 위해 외부에서 검색할 필요가 없도록 자세한 답을 제공하세요.
7  당신은 AI 어시스턴트입니다. 사용자가 과제를 줄 것입니다. 목표는 가능한 한 충실하게 과제를 완성하는 것입니다. 과제를 수행하는 동안 단계별로 생각하고 단계를 정당화하세요.
8  과제를 설명하고 답을 설명해야 합니다. 객관식 질문에 답할 때는 먼저 정답을 출력하세요. 그다음 다른 답이 왜 틀렸는지 설명하세요. 다섯 살 아이에게 답한다고 생각하세요.
9  답에 도달하기 위해 정의를 어떻게 사용했는지 설명하세요.
10 당신은 AI 어시스턴트입니다. 과제를 설명하고 답을 설명해야 합니다. 객관식 질문에 답할 때는 먼저 정답을 출력하세요. 그다음 다른 답이 왜 틀렸는지 설명하세요. 질문에 답하려면 추가 지식이 필요할 수 있습니다.
11 당신은 사람들이 정보를 찾도록 돕는 AI 어시스턴트입니다. 사용자가 질문을 줄 것입니다. 과제는 가능한 한 충실하게 답하는 것입니다. 답하는 동안 단계별로 생각하고 답을 정당화하세요.
12 사용자가 지시가 포함된 과제를 줄 것입니다. 여러분의 일은 지시를 가능한 한 충실하게 따르는 것입니다. 답하는 동안 단계별로 생각하고 답을 정당화하세요.
13 당신은 교사입니다. 과제가 주어지면, 과제가 무엇을 요구하는지, 제공하는 지침은 무엇인지, 답을 찾기 위해 그 지침을 어떻게 쓰는지를 쉬운 단계로 설명하세요.
14 당신은 모든 언어를 알고 한 언어를 다른 언어로 번역하는 방법을 아는 AI 어시스턴트입니다. 과제가 주어지면, 과제가 무엇을 요구하는지와 제공하는 지침을 쉬운 단계로 설명하세요. 과제를 풀고, 그 지침을 어떻게 사용해 풀었는지 보이세요.
15 과제의 정의와 샘플 입력이 주어지면 정의를 작은 부분으로 나누세요. [...] Part #: 정의의 핵심 부분. Usage: 핵심 부분의 기준을 충족하는 샘플 응답. 왜 그 기준을 충족한다고 생각하는지 설명하세요.
16 당신은 사람들이 정보를 찾도록 돕는 AI 어시스턴트입니다.
```

논문은 이 메시지들이 “모델이 짧은 답과 긴 답을 모두 생성하는 능력을 보존하도록 설계되었다”고 말한다(표 2 캡션).

**증거.** CoT 없이 빈 시스템 메시지로 zero-shot 평가한 Orca-13B는 BBH 49.7(ChatGPT 48.9, GPT-4 67.4, Vicuna-13B 23.3), AGIEval 41.7(ChatGPT 47.2)을 기록했다([[orca]], 표 8, 11). 두 단계 학습은 FLAN-1M만 사용한 경우와 비교해 AGIEval 41.7 대 37.18이었다(표 10). **결과(단일 연구).**

**조건과 한계.** 표 10은 두 변수를 동시에 바꾼다. ChatGPT 단계와 5배 더 많은 데이터이므로 teacher-assistant 효과가 분리되지 않았다. 같은 질의에서 explanation target과 answer-only target을 비교한 run도 없다([[orca]], Findings). 저자들은 학습 데이터가 zero-shot 프롬프트를 모사하며, 성능이 “tuning data의 분포와 강하게 상관할 가능성이 있다”고 말한다(§8).

**Orca 2.** [[orca-2]]는 explanation tuning을 유지하지만 과제마다 해결 전략을 고른다. 절차(§3)는 다음과 같다. 다양한 과제로 시작하고, “Orca의 성능에 guided by”하여 어떤 과제에 어떤 전략(direct-answer, step-by-step, explain-then-answer 등)이 필요한지 결정하고, 교사용 과제별 시스템 지시를 작성한 뒤, 학습 시에는 그것을 일반 시스템 메시지로 바꾼다. 마지막 단계가 **Prompt Erasing**이다. 학생은 전략 지시를 전혀 보지 못하므로 과제에서 전략을 추론해야 한다. 데이터는 약 817K 인스턴스다. 하나의 지시를 쓰는 126개 하위 범주로 묶은 1,448개 과제에서 FLAN 약 602K, few-shot 55K, 수학 약 160K, 합성 의사–환자 대화 2,000개로 구성된다(§4.1). Orca-2-13B는 6개 zero-shot reasoning 벤치마크에서 평균 64.49로 LLaMA-2-Chat-70B의 56.85보다 높다(그림 4). 이 그룹 밖에서는 더 약하다. 대화 데이터가 없었기 때문에 MT-Bench 6.15로 LLaMA-2-Chat-13B의 6.64보다 낮았고(표 3, §6.4), 조심스러운 시스템 메시지는 hallucination 비율을 10.97에서 29.00으로 올린다(표 11). Prompt Erasing만을 분리한 ablation은 없다.

**범용 모델에 대한 함의.** 다양한 시스템 메시지는 학생이 배우는 출력 형식의 폭을 넓힌다. 그러나 학생의 강점은 여전히 질의 분포를 따른다. Orca 2의 conversational score가 instruct-model baseline보다 낮았던 이유는 대화가 없었기 때문이다.

## §2 Distilling Step-by-Step — 두 번째 학습 과제로서의 rationale

**정의.** Distilling Step-by-Step(DSBS)은 few-shot CoT prompting으로 PaLM-540B에서 rationale과 label을 추출하고, 작은 모델을 두 과제에 학습한다. 하나는 label 예측이고 다른 하나는 rationale 생성이다([[distilling-step-by-step]], §3).

**문제.** 과제별 소형 모델은 few-shot LLM에 맞추려면 대규모 라벨 데이터셋 또는 증류 데이터셋이 필요하다(초록).

**메커니즘.**
1. PaLM-540B에 (입력, rationale, label) 약 10개의 예시를 prompt로 주고, 각 학습 입력에 대해 계속 생성하게 하여 r̂_i와 ŷ_i를 만든다(§3.1, 한계).
2. 예제마다 두 학습 입력을 만든다. `[label] x_i` → ŷ_i와 `[rationale] x_i` → r̂_i이며, 하나의 T5 모델에서 처리한다(§3.2).
3. 테스트 시에는 `[label]` 접두사로만 생성하므로 추론 비용은 label-only 모델과 같다(§3.2).

**수식** (식 1, 3–4):

```
L = L_label + λ · L_rationale
L_label     = (1/N) Σ_i ℓ(f(x_i), ŷ_i)
L_rationale = (1/N) Σ_i ℓ(f(x_i), r̂_i)
```

N은 학습 예제 수, f는 학생, ℓ은 토큰 cross-entropy, ŷ_i는 사람 label(fine-tuning 설정) 또는 LLM label(distillation 설정), r̂_i는 LLM rationale이다. λ는 rationale loss의 가중치다. 논문은 λ를 인쇄하지 않는다. 공개 코드는 `loss = α·label + (1−α)·rationale`를 사용하며 α = 0.5를 “recommended”로 둔다([[distilling-step-by-step]], Recipe ledger).

**계산 예시.** 예제 하나에서 label은 학생이 확률 0.5로 예측하는 한 토큰이므로 L_label = −ln 0.5 = 0.693이다. rationale은 각 토큰을 확률 0.8로 예측하는 4토큰이므로 평균 토큰 loss는 −ln 0.8 = 0.223이다. λ = 1이면 L = 0.693 + 0.223 = 0.916이다. 코드의 α = 0.5에서는 L = 0.5·0.693 + 0.5·0.223 = 0.458이며, 이는 λ = 1 loss를 0.5배 한 값이다. λ = 1과 α = 0.5에서 두 관례는 loss의 상수배 0.5만 다르다.

**증거.** T5 220M, 데이터 100%에서 e-SNLI / ANLI / CQA / SVAMP 정확도는 standard fine-tuning이 88.38 / 43.58 / 62.19 / 62.63, PaLM rationale을 쓴 DSBS가 89.51 / 49.58 / 63.29 / 65.50, GPT-NeoX-20B rationale을 쓴 경우가 89.12 / 48.15 / 63.25 / 63.00이다(표 1). rationale과 label을 이어 붙이는 단일 과제 target은 88.88 / 43.50 / 61.37 / 63.00이며, ANLI와 CQA에서 standard fine-tuning보다 낮다(표 2). ANLI에서는 770M 학생이 라벨 데이터의 80%만으로 few-shot PaLM-540B를 이긴다(§4.3). **결과(단일 연구)**이며, 무작위 run은 4개다(부록 A.1).

**조건과 한계.** 모든 학생은 한 데이터셋에서 학습하고 같은 데이터셋에서 평가하므로 held-out task 평가는 없다. 따라서 DSBS는 일반 능력에 대해 아무것도 말해 주지 않는다([[distilling-step-by-step]], Findings). 논문은 교사 label이 틀린 rationale을 제거했는지도 보고하지 않는다.

**범용 모델에 대한 함의.** 학습 형식이 중요하다는 DSBS 결과(220M에서 multi-task > concatenated target)는, `<think> rationale </think> answer`라는 이어 붙인 형태를 쓰는 이후 trace distillation에 대한 주의점이다. 이 비교는 이 장의 출처들에서 LLM 규모로 반복되지 않았다.

## §3 DeepSeek-R1 증류 데이터 — 교사가 생성한 것

**정의.** R1-Distill 모델은 DeepSeek-R1의 두 번째 SFT 단계에서 나온 약 800K개 샘플만으로 SFT한 6개의 dense 모델이다([[deepseek-r1]], 보충 자료 F).

**데이터의 출처.** R1 파이프라인은 “수천” 개 예제의 cold-start SFT → reasoning RL → DeepSeek-V3-Base에서 다시 시작한 rejection-sampling SFT → mixed-reward RL 순서다([[deepseek-r1]], §3, 그림 2). 약 800K 세트는 첫 번째 RL 단계의 checkpoint로부터 세 번째 단계에서 만든다. ch-31은 cold start와 SFT–RL 교대를, ch-35는 이 세트를 다른 연구소의 단계 사이에 어디에 놓는지 다룬다.

논문은 reasoning 절반을 다음과 같이 설명한다(보충 자료 B.3.3에서 인용).

> “이전 단계에서는 규칙 기반 reward로 평가할 수 있는 데이터만 포함했습니다. 그러나 이 단계에서는 데이터셋을 확장하여 추가 데이터를 포함합니다. 그중 일부는 ground-truth와 모델 예측을 DeepSeek-V3에 넣어 판정하게 하는 generative reward model을 사용합니다. [...] 우리는 언어가 섞인 chain-of-thought, 긴 문단, 코드 블록을 필터링했습니다. 각 프롬프트에 대해 여러 응답을 샘플링하고 정답인 것만 보존합니다. 총 약 600k개의 추론 관련 학습 샘플을 수집합니다.”

Judge prompt(Listing 4, 발췌)는 두 class를 다음과 같이 정의한다.

```
1. **정답(correct)**: 답이 추론 과정과 최종 결론 모두에서 참조 답과 완전히 일치하며,
   오류나 누락 없이 질문에 답한다.
2. **오답(incorrect)**: 답에 핵심 추론 단계 또는 최종 결론의 중대한 오류가 있거나,
   질문의 핵심에서 완전히 벗어난다.
다음 내용을 두 개의 key를 포함한 **JSON** 형식으로 출력하라.
1. 'analysis': 답의 정답성에 대한 분석;
2. 'correctness': correct/incorrect
```

비추론 절반(약 200k)에서 DeepSeek은 DeepSeek-V3 SFT 데이터의 일부(글쓰기, 사실 QA, 자기 인식, 번역)를 재사용하고, software-engineering 데이터(프로그램 복구, 프런트엔드 개발)를 더하며, 일부 과제에는 V3가 CoT를 쓰도록 prompt한다. 단, “hello” 같은 단순 질의에는 CoT를 쓰게 하지 않는다(B.3.3).

**구성**([[deepseek-r1-recipe]], 표 5). 토큰 총계는 샘플 수 × 평균 토큰으로 계산한 값이다.

| 도메인 | 샘플 | 평균 토큰 | 토큰(계산값) | 샘플 비중 | 토큰 비중 |
|---|---|---|---|---|---|
| 수학 | 395,285 | 6,094.2 | 2.409B | 49.1% | 55.9% |
| 코드 | 211,129 | 7,435.7 | 1.570B | 26.2% | 36.4% |
| STEM | 10,124 | 4,928.8 | 0.050B | 1.3% | 1.2% |
| 논리 | 10,395 | 2,739.0 | 0.028B | 1.3% | 0.7% |
| 일반 | 177,812 | 1,419.8 | 0.252B | 22.1% | 5.9% |
| 합계 | 804,745 | 5,355.3 | 4.310B | 100% | 100% |

**계산 예시.** 일반 데이터는 샘플 기준으로 177,812 / 804,745 = 22.1%지만, 토큰 기준으로는 0.252B / 4.310B = 5.9%다. 일반 샘플의 평균이 1,419.8토큰이고 수학 샘플의 평균은 6,094.2토큰이기 때문이다. 토큰 평균 loss를 사용하면 gradient signal의 94%는 추론 도메인에서 온다. 학생은 2–3 epoch에서 약 8.6B–12.9B 토큰을 본다(계산값. 논문은 토큰 수가 아니라 epoch를 보고한다).

**학생 모델**(표 6; B.4.3): Qwen2.5-Math-1.5B(초기 LR 1×10⁻⁴), Qwen2.5-Math-7B(8×10⁻⁵), Qwen2.5-14B(7×10⁻⁵), Qwen2.5-32B(6×10⁻⁵), Llama-3.1-8B(5×10⁻⁵), Llama-3.3-70B-Instruct(2×10⁻⁵)를 사용한다. 2–3 epoch, 초기값의 10분의 1까지 cosine decay, 32,768-token context, batch 64라는 설정이다.

**증거**(표 15, pass@1): Distill-Qwen-1.5B는 AIME 2024 28.9, MATH-500 83.9를 기록했다(GPT-4o-0513은 9.3, 74.6). Distill-Qwen-32B는 72.6, 94.3, GPQA Diamond 62.1, LiveCodeBench 57.2를 기록했고, Distill-Llama-70B는 70.0, 94.5, 65.2, 57.5를 기록했다. 32B에서 증류와 RL을 비교하면(표 16), Qwen2.5-32B-Zero를 10K step 넘게 RL한 모델은 AIME 2024 / MATH-500 / GPQA-D / LiveCodeBench에서 47.0 / 91.6 / 55.0 / 40.2를 기록했다. **결과(단일 연구).** 저자들은 증류가 경제적이라고 결론 내리지만, “인간 지능의 경계를 넘어 진전하려면 더 강력한 base model과 더 큰 규모의 reinforcement learning이 여전히 필요할 수 있다”고 말한다(F.1, **해석**). 또한 증류한 모델에 RL을 더하면 “모델 성능을 상당히 끌어올릴 수 있다”고 말하며, 이를 수행하지 않았다고 밝힌다(보충 자료 F).

**조건과 한계.** 프롬프트당 샘플 수, rejection sampling을 위한 교사 sampling temperature, 수율 비율은 보고되지 않았다. 대부분의 샘플은 single-turn(평균 round 1.0)이며, 저자들은 이것이 multi-turn 능력을 제한할 수 있다고 말한다(B.3.3). 표 15의 모든 벤치마크는 수학, 코드, STEM이다.

**범용 모델에 대한 함의.** R1 자체 세트는 reasoning-only가 아니다. 비추론 및 software-engineering SFT 데이터(Dev2 → Dev3)를 더하면 AlpacaEval 2.0은 55.8에서 62.1로, Aider-Polyglot은 25.6에서 44.8로 올랐다([[deepseek-r1]], §4). 샘플 기준 22%의 일반 데이터 비중은 이 계보에서 개수를 공개한 유일한 비추론 기준점이다.

## §4 R1 스타일 증류의 공개 재현 연구

2025년 1–2월의 세 프로젝트가 데이터와 학습 설정을 공개했다. 아래 비교는 인용한 출처에 인쇄된 값만 사용한다. [figures/distill-lineage.html](figures/distill-lineage.html)은 §1–§6의 교사 → 데이터셋 → 학생 계보를 보여 준다. 노드를 클릭하면 데이터, 필터, 학습 설정, 출처 위치를 포함한 보고 결과를 볼 수 있다.

| 속성 | Sky-T1-data-17K | Bespoke-Stratos-17k | OpenR1-Math-220k |
|---|---|---|---|
| 출처 | [[sky-t1]] NovaSky 블로그 | [[bespoke-stratos]] 블로그, 카드, 스크립트 | [[openr1]] Update #2, 데이터셋 카드 |
| 교사 | QwQ-32B-Preview | DeepSeek-R1 | DeepSeek-R1 |
| 프롬프트 | 수학 10k(NuminaMATH AIME, MATH, Olympiads), 코드 5k(APPs, TACO), 과학·퍼즐 STILL-2 1k | Sky-T1과 같은 구성: 10k / 5k / 1k | NuminaMath 1.5 문제 400k개 |
| 프롬프트당 샘플 | 보고하지 않음 | 보고하지 않음 | 2개, 일부 4개(추론 과정 800k개) |
| 최대 생성 길이 | 보고하지 않음 | 보고하지 않음 | 16k 토큰 |
| 수학 필터 | ground truth와 exact match | ground truth에 대한 gpt-4o-mini judge | Math-Verify; 거부된 문제 일부에 대한 Llama-3.3-70B-Instruct judge |
| 코드 필터 | 데이터셋 unit test | 실행(Ray cluster) | 없음(수학 전용) |
| 형식화 | QwQ 추론 과정을 GPT-4o-mini가 재작성 | R1 추론 과정을 재형식화하지 않음 | `\boxed{}` suffix prompt가 붙은 R1 출력 |
| 공개 크기 | 17K | 16,710행 | `default` 93,733; `extended` 131,396행 |
| 학생 | Qwen2.5-32B-Instruct, 3 epoch, LR 1e-5, batch 96 | Qwen2.5-32B-Instruct, 3 epoch, LR 1e-5, batch 96 | Qwen2.5-Math-7B-Instruct, 3 epoch, LR 5e-5 |
| 연산 / 비용 | 8×H100에서 19 h(약 $450) | 생성 $800; 8×H100에서 학습 27 h | 512 H100에서 생성; 비용 보고 안 함 |
| 보고 결과 | AIME24 43.3, MATH500 82.4, GPQA-D 56.8(o1-preview: 40.0, 81.4, 75.2) | AIME24 63.3, MATH500 93.0, GPQA-D 58.1, LCB v2 All 71.1 | MATH-500 90.6, AIME24 36.7, AIME25 40.0 |
| 같은 크기의 R1-Distill | Qwen-32B: 72.6 / 94.3 / 62.1(보고값) | Qwen-32B: 66.7 / 89.8 / 61.1 / 72.2(저자 평가) | Qwen-7B: 91.6 / 43.3 / 40(Update #2 표) |

**§4.1 Verifier false negative.** False negative란 필터가 거부한 정답 교사 풀이를 말한다. Bespoke Labs는 Sky-T1의 regex-plus-sympy parser가 “올바른 답을 추출하는 데 자주 실패한다”고 보고한다. 이를 gpt-4o-mini judge로 교체하면 유지된 정답 수학 풀이의 비중이 25%에서 73%로 상승했다([[bespoke-stratos]], 블로그 “Data Curation”). 계산 예시: 정답 R1 수학 풀이 1,000개 중 parser는 250개, judge는 730개를 유지하므로 judge는 새 교사 호출 없이 정답 학습 예제 480개를 추가한다. Sky-T1은 코드에서도 같은 종류의 문제를 보고한다. 재형식화 없이 마지막 코드 블록을 취하면 QwQ의 APPs 정확도는 약 25%였지만, GPT-4o-mini로 재형식화한 뒤 정확도는 90%를 넘었다([[sky-t1]], 블로그). OpenR1의 Math-Verify는 문제의 55%에서 적어도 하나의 정답을 찾았고, Llama judge는 문제 28,000개를 더 복구했다([[openr1]], Update #2). 세 프로젝트에서 **재현된 결과**는 answer extraction이 정답 교사 추론 과정을 거부했다는 것이다. 손실 규모(유지율 25% 대 73%)를 측정한 것은 Bespoke Labs뿐이다.

**§4.2 필터 엄격도.** Open R1은 무작위 200k pool의 여섯 가지 필터 버전으로 Qwen 7B Instruct를 한 epoch 학습했다. no filter(200k), Llama judge(124k), Math-Verify(88.7k), 다시 파싱한 Math-Verify(101k), union(154k), intersection(71.2k)이다. 처음 40 step에서 MATH-500은 필터링 없이 0.61, intersection에서는 0.72였다. 이후 격차는 줄었고, 필터링하지 않은 세트도 경쟁력을 유지했다([[openr1]], Update #3). 저자들은 union을 권장한다. **결과(단일 연구)**이며, final-step 표는 인쇄하지 않았다.

**§4.3 Sky-T1의 데이터 혼합과 학생 크기.** 32B 모델을 Numina 수학 문제 3–4K개에 학습하면 AIME24가 16.7에서 43.3으로 올랐다. APPs 코드 데이터를 추가하면 36.7로 떨어졌다. 더 어려운 NuminaMath와 TACO 문제로 혼합을 풍부하게 하면 코드를 개선하면서 43.3을 회복했다([[sky-t1]], 블로그 “Data mixture matters”). 7B와 14B에서 팀은 “only modest improvements”만 보았고(Qwen2.5-14B-Coder-Instruct의 APPs에서 LiveCodeBench 42.6 → 46.3), 반복적인 출력이 자주 나타났다. 저자들은 수학과 코드에 서로 다른 reasoning style이 필요하다고 가설을 세운다(**해석**).

**§4.4 7B에서의 소량 데이터.** Bespoke-Stratos-7B(Qwen2.5-7B-Instruct, 예제 16,710개)는 AIME2024 20.0, MATH500 82.0, GPQA-D 37.8을 기록한다. 같은 instruct base는 10.0 / 74.2 / 33.3이고, 저자 평가의 R1-Distill-Qwen-7B는 43.3 / 89.4 / 44.9다(보고값은 55.5 / 92.8 / 49.1). LCB v2 Hard는 base보다 낮아진다(1.6 대 3.3)([[bespoke-stratos]], 7B 표). 모델 카드는 이 격차의 가능한 원인으로 17k 대 800k 예제를 제시한다(**해석**).

**조건과 한계.** 세 프로젝트 모두 수학, 코드, GPQA-Diamond만 평가한다. Bespoke Labs는 “benchmarks convey only one side to the story”라고 말한다([[bespoke-stratos]], “Thoughts and future work”). instruction following, 안전성, chat 평가는 어느 프로젝트도 보고하지 않는다.

## §5 학생이 교사에게서 물려받는 것

**§5.1 형식과 추론 양식.** 학생은 답과 함께 교사의 trace 구조를 배운다. OpenThoughts가 trace에서 self-reflection 문구를 제거하자 평균 trace 길이는 11,593에서 328토큰으로, 평균 점수는 51.4에서 26.3으로 떨어졌다. 2,048토큰을 넘는 trace를 버리면 34.2였다([[open-thoughts]], 부록 H.3, 표 22). Open R1은 `<think>` prefill이 없으면 out-of-domain 프롬프트가 base instruct 모델 행동으로 되돌아감을 발견했고, chat template에서 prefill을 강제하라고 권한다([[openr1]], Update #3 Lesson 4). QwQ-32B-Preview 카드는 “Language Mixing and Code-Switching”, “Recursive Reasoning Loops”를 알려진 한계로 나열한다([[teacher-output-licenses]], §3). 최종 답만 확인하는 trace filter는 이를 제거하지 못한다. R1은 mixed-language CoT를 명시적으로 제거한다(B.3.3).

**§5.2 양식 대 능력.** 모방은 교사의 능력을 맞추지 못한 채 양식만 맞출 수 있다. Orca가 동기로 든 측정에서 GPT-4 judge는 Vicuna-13B를 ChatGPT의 92%로 평가하지만 BBH는 48%로 평가한다([[orca]], §1). OpenThoughts는 OpenThinker3-7B를 포함한 모든 시험한 distilled reasoning model이, 같은 문제의 구조 보존 변형 사이에서 정확도 변화를 보인다고 보고한다. 논문은 이를 크다고 설명하지만 그림으로만 제시한다([[open-thoughts]], 부록 N, 그림 13). **재현됨:** judge 점수와 고정된 benchmark item은 imitation model을 과대평가하는 측정상 주의점이다.

**§5.3 오염과 시간 분할 평가.** AIME와 MATH 항목은 공개되어 있으므로 교사가 보았을 수 있다. 이 계보의 프로토콜은 다음과 같다. R1은 학습 데이터 이후의 AIME 2025를 o1-1217의 12.0/15와 비교해 11.3/15로 보고하고, 10-gram decontamination이 paraphrase를 잡지 못한다고 지적한다([[deepseek-r1]], E.2, D.1). OpenThoughts는 AIME25, HMMT 02/25, HLE, LiveCodeBench 06/24–01/25를 pipeline decision에서 제외하고 Indel similarity와 13-gram matching으로 decontaminate한다([[open-thoughts]], §4, 부록 F). Open R1은 s1을 따라 8-gram decontamination을 쓴다([[openr1]], README).

**§5.4 안전성과 거부 행동.** 학생은 데이터에 등장한 프롬프트에서만 교사의 안전 행동을 물려받는다. 안전 데이터 없이 학습한 OpenThinker3-7B는 base Qwen2.5-7B-Instruct의 HarmBench harmfulness 14.5와 비교해 55.5이며, XSTest over-refusal은 4.4에서 5.6으로 움직인다([[open-thoughts]], 부록 L, 표 30). RLHF 안전 학습이 없는 Orca-2-13B는 LLaMA-2-Chat-13B의 0.17%와 비교해 violent-content defect rate 13.47%를 기록한다([[orca-2]], 표 4–5). Dolphin 데이터셋은 Orca 스타일 GPT-4 및 GPT-3.5 completion에서 “alignment, refusal, avoidance, bias의 인스턴스”를 제거했지만, 카드에는 방법·개수·평가가 없다([[dolphin]]).

**§5.5 정답, 잘못된 추론.** Outcome filter는 최종 답만 비교하므로 정답이지만 유도 과정이 무효인 trace는 통과한다. 공개 재현 출처 중 어느 것도 이것이 얼마나 자주 일어나는지 측정하지 않는다. R1의 generative judge는 답이 “추론 과정과 최종 결론 모두에서 참조 답과 완전히 일치하는지” 묻는데(Listing 4), judge를 적용한 부분에서 이 경우를 겨냥한다. 코드에서는 반대 오류도 일어난다. 공개 test를 전부 통과한 R1 CodeForces 풀이 7개가 전체 test set에서는 실패했으므로, 공개 test filtering은 false positive를 허용한다([[openr1]], Update #3). step-level verification은 ch-44에서 다룬다. R1은 PRM을 unsuccessful attempt로 열거하며 model-based PRM은 “inevitably leads to reward hacking”이라고 말한다(G.2).

## §6 OpenThoughts — 데이터 recipe에 대한 통제 실험

**프로토콜.** 각 후보 전략은 예제 31,600개를 만들고 Qwen2.5-7B-Instruct를 미세조정한다. 8개 development benchmark 평균이 가장 좋은 전략을 다음 pipeline step에 유지하며, 기본 교사는 DeepSeek-R1이다([[open-thoughts]], §4). 이러한 실험을 1,000회 넘게 수행했다(초록).

**수치를 포함한 발견**(§4.1–4.6):
1. 질문 출처: 가장 좋은 코드 출처와 가장 나쁜 코드 출처의 차이는 17.2점이다(표 3). 상위 2개 코드 출처 평균은 41.3이고 상위 16개는 36.4다(표 4).
2. 질문 필터: 코드에서는 GPT-4o-mini 난이도 rating이 43.0으로 이기고, 수학·과학에서는 GPT-4.1-mini로 가장 긴 응답을 고르는 것이 41.9로 이긴다. 최선 필터는 random보다 수학 4%, 코드 6%를 더한다(표 5). LLM 기반 필터는 embedding 및 fastText 필터보다 낫다(§1).
3. 문제당 답: 모든 도메인에서 16개다. 수학과 과학에는 exact deduplication을 적용하고 코드에는 적용하지 않는다(표 6).
4. 답 필터링: 수학에서 random filtering 64.8이 예를 들어 GPT verification 61.4를 포함한 다른 모든 필터보다 좋다(표 7).
5. 교사: QwQ-32B 대 DeepSeek-R1 평균은 코드에서 44.2 대 42.3, 수학에서 44.2 대 41.6이다(표 8). 과학에서는 39.1 대 35.9다(표 49).
6. 규모: 1.2M 예제(수학 850K, 코드 250K, 과학 100K)의 OpenThinker3-7B는 AIME25 53.3, LiveCodeBench 06/24–01/25 51.7, GPQA-D 53.7을 기록하며 12-task 평균에서 R1-Distill-Qwen-7B보다 12.4점 높다(표 1).

**조건과 한계.** Pipeline 선택은 7B 학생과 31,600개 예제에서 이뤄졌고, verification 결과는 학생 크기에 따라 뒤집힌다(verification 적용 시 7B 45.0 → 41.9, 32B 62.1 → 64.5)([[open-thoughts]], 부록 H.1.1, 표 15). 혼합 비율은 ablation하지 않고 OpenThoughts2-1M에서 복사했다(§5). 교사 sampling temperature와 길이는 보고되지 않았다(Verification). HLE MCQ에서 OpenThinker3-7B는 instruct base의 12.7보다 낮은 10.2다(표 1). 저자들은 overall average로 선택하는 것은 cross-domain transfer를 가정한다고 지적한다(§6).

**범용 모델에 대한 함의.** 이 recipe는 수학·코드·과학 평균에 맞춰 최적화됐다. 코드 전용 세트는 출처에 따라 GPQA-D 47.3 또는 36.7에 도달하지만, 과학 데이터를 섞은 세트는 52.7에 도달한다([[open-thoughts]], 부록 H.5). 따라서 추론 도메인 간 전이는 데이터 출처에 의존하며, 비추론 과제로의 전이는 선택 기준이 아니었다.

## §7 라이선스 — 교사 출력을 학습에 사용할 수 있는가

증류 데이터셋은 교사의 라이선스 또는 서비스 약관이 그 출력으로 다른 모델을 학습하는 것을 허용할 때만 공개할 수 있다. 아래 설명은 [[teacher-output-licenses]]에서 인용한 것이다. 법률 자문이 아니며, 생성 시점에 다시 확인해야 한다.

| 교사 | 라이선스 | 다른 모델 학습을 위한 출력 사용 | 출처 |
|---|---|---|---|
| DeepSeek-R1 | MIT(코드와 가중치) | “다른 LLM 학습을 위한 증류를 포함하되 이에 국한되지 않는, 모든 수정과 파생 작업을 허용” | R1 README §7 |
| R1-Distill-Qwen 모델 | Qwen2.5에서 Apache 2.0을 상속 | 베이스 라이선스에 따름 | R1 README §7 |
| R1-Distill-Llama-8B / -70B | Llama 3.1 / Llama 3.3 라이선스 | 베이스 라이선스에 따름 | R1 README §7 |
| Llama 3.1 | Llama 3.1 Community License | 라이선스는 synthetic data generation과 distillation을 포함하여 “다른 모델을 개선하기 위해” 출력을 사용하는 것을 “허용한다” | Llama 3.1 model card |
| QwQ-32B-Preview, QwQ-32B | Apache 2.0 | Apache 2.0 약관 | model card metadata |
| Qwen2.5 | 3B와 72B를 제외한 Apache 2.0 | 모델별 라이선스에 따름 | [[qwen-2.5]], 표 1 |
| Closed API 모델 | 서비스 약관 | Anthropic은 자사 서비스 약관을 위반한 증류 캠페인을 보고한다 | [[anthropic-distillation-attacks-report]] |

공개된 데이터셋도 각각의 라이선스를 가진다. Bespoke-Stratos-17k와 OpenR1-Math-220k는 Apache 2.0이고, Dolphin도 Apache 2.0이지만 “각 모델은 해당 base model의 라이선스를 따른다”([[teacher-output-licenses]], §6). 데이터셋 라이선스는 그 출력이 생성된 조건의 약관을 무효화하지 않는다.

## §8 추론 증류가 비추론 능력에 미치는 효과

**§8.1 Instruction following(MathIF).** **정의.** MathIF는 420개 수학 문제에 Python으로 검증 가능한 제약 1–3개(길이, 어휘, 형식, 접사)를 더한다([[scaling-reasoning-losing-control-mathif]], §3). **수식**(식 1): 제약이 n개인 질의에서 제약 i를 충족하면 I(C_i) = 1일 때,

```
HAcc = Π_{i=1..n} I(C_i)          SAcc = (1/n) Σ_{i=1..n} I(C_i)
```

HAcc(hard accuracy)는 모든 제약을 충족할 때만 1이고, SAcc(soft accuracy)는 충족한 비율이다. 둘 다 질의들에 대해 평균을 낸다. **계산 예시:** 세 제약 질의에서 3개 중 2개를 충족하면 HAcc = 0이고 SAcc = 2/3 = 0.67이다.

**증거.** 23개 reasoning model 중 최고인 Qwen3-14B는 HAcc 50.71에 도달한다(표 3). DeepSeek-R1-Distill-Llama-70B(41.43)는 Qwen3-4B(44.05)보다 낮다. 제약을 추가했을 때 상대 정확도 하락이 가장 큰 모델은 SFT-only distilled 모델인 R1-Distill-Qwen-1.5B(−40.09%)와 R1-Distill-Llama-8B(−39.04%)다(표 3). Qwen2.5-7B에 QwQ-32B trace 18k개(오답 및 8,192토큰 초과 CoT 제거)를 쓴 통제 run에서 SFT는 HAcc / SAcc / 수학 정확도를 15.95 / 33.13 / 13.59에서 7.86 / 21.03 / 23.10으로 바꿨다(표 4). 네 base의 학습 변형 16개 전체에서 15개는 HAcc와 SAcc를 잃었고 15개는 수학 정확도를 얻었다. IFEval(prompt-level strict)에서 Llama-3.3-70B-Instruct는 90.38, DeepSeek-R1-Distill-Llama-70B는 78.74이며, Qwen2.5-32B-Instruct는 80.96, s1-32B는 58.04다(부록 G, 표 13; 공개 모델들이며 통제 비교는 아니다). 통제 run에 대해서는 **결과(단일 연구)**다.

**§8.2 다른 도메인으로의 전이(Transferability Index).** **정의.** Huan et al.은 비수학 벤치마크 집합에서의 모델 향상을 수학 벤치마크 집합에서의 향상에 대한 백분율로 표현하는 Transferability Index(TI)를 정의한다([[transferability-of-llm-reasoning]], arXiv:2507.00432v2 §2.1). **수식:** 그룹 g와 벤치마크 b에 대해,

```
ΔR_b = R_b(model) − R_b(base);   σ_g = Std{ΔR_b : b ∈ B_g};   δ_b = ΔR_b / σ_g
s_b  = sign(δ_b)·|δ_b|^(1/2);    w_b = 100 − R_b(base);       ŵ_b = w_b / Σ_{u∈B_g} w_u
DI_g = Σ_b ŵ_b·s_b;              TI_g(%) = 100 · DI_g / DI_math,   g ∈ {other, non}
```

R_b는 백분율 정확도다. σ_g는 그룹 내부에서 향상을 정규화하고, 제곱근은 극단적 향상의 영향을 제한하며, w_b는 더 어려운 벤치마크(base score가 낮음)에 더 큰 가중치를 준다. DI_g는 그룹의 domain index다.

**계산 예시**(그룹당 벤치마크 2개. 논문은 명시하지 않지만 Std는 population standard deviation으로 취한다). 수학에서 A는 20 → 40, B는 60 → 70이므로 ΔR = (20, 10), σ = 5, δ = (4, 2), s = (2, 1.414), w = (80, 40), ŵ = (0.667, 0.333), DI_math = 1.805다. 비추론에서 C는 50 → 45, D는 70 → 72이므로 ΔR = (−5, 2), σ = 3.5, δ = (−1.429, 0.571), s = (−1.195, 0.756), w = (50, 30), ŵ = (0.625, 0.375), DI_non = −0.464다. TI_non = 100 · (−0.464 / 1.805) = −25.7%이다. 즉 모델은 수학에서 향상했지만 수학 밖에서는 성능을 잃었다.

**증거.** Qwen3-14B-Base의 수학 문제 47K개(DeepScaler와 SimpleRL level 3–5)를 사용한 통제 연구다. SFT target은 rejection sampling으로 유지한 Qwen3-32B trace이고, RL은 답 정확도로 GRPO를 사용한다(§2.2, 부록 A.3.2). 표 1:

| 모델 | 수학 평균 | 기타 추론 평균 | 비추론 평균 | IFEval | TI_other | TI_non |
|---|---|---|---|---|---|---|
| Qwen3-14B-Base | 27.7 | 30.2 | 45.7 | 69.2 | – | – |
| thinking-mode trace에 SFT | 49.8 | 45.3 | 21.1 | 42.3 | +52.2 | −104.1 |
| non-thinking trace에 SFT | 32.3 | 45.2 | 29.0 | 41.4 | +165.4 | −278.9 |
| RL(GRPO) | 53.8 | 60.0 | 53.2 | 70.0 | +82.3 | +52.2 |

Qwen3-8B-Base ablation(표 4)에서는 Qwen3-32B trace에 off-policy SFT를 적용하면 비추론 평균이 26.6이다(base 33.6, TI_non −40.5). 학생 자신의 필터링한 샘플에 on-policy SFT를 적용하면 35.0(TI_non +30.2)이고, off-policy RL은 31.7(+4.5)이다. 저자들은 sampling distribution이 가장 중요한 요인이라고 결론 내리고(§6, **해석**), RL 이후보다 SFT 이후 hidden-state shift가 더 크다고 보고한다(표 2). 수학 전용 프롬프트와 하나의 base family에 대한 **결과(단일 연구)**다.

**조건과 한계.** 두 연구 모두 수학 전용 프롬프트의 교사에서 증류한다. R1 자체 세트에는 일반 샘플이 22% 있지만 어느 연구도 이를 재현하지 않는다. §8.2의 통제 SFT 데이터에는 비추론 샘플이 없으므로, 이 결과는 혼합 세트 증류가 일반 능력을 잃는다는 것을 보이지 않는다. ch-38a는 SFT 대 RL 증거와 그 반대 연구를, ch-30a는 forgetting 측정을 다룬다.

**범용 모델에 대한 함의.** 증류 학생이 범용으로 남아야 한다면, 교사의 목표 벤치마크 외에도 instruction following(IFEval, MathIF형 제약), 비추론 QA, 안전 세트에서 평가하라. 위 두 연구에서 수학 전용 trace SFT가 IFEval과 비추론 평균을 낮췄기 때문이다. 증류 세트가 수학 전용이면 폭을 위해 비추론 데이터나 이후 단계를 계획하라. 이 장의 어느 출처도 수학 전용 세트가 그 폭을 보존함을 보이지 않는다.

## §9 Token-level 및 on-policy 증류

**Token-level(logit) 증류.** 학생은 하나의 샘플링된 토큰 대신 교사의 전체 다음 토큰 분포를 맞춘다. Gemma 2의 목적함수는 다음과 같다([[gemma-2]], §3.2).

```
min_{P_S}  Σ_x  −P_T(x | x_c) · log P_S(x | x_c)
```

x는 vocabulary token, x_c는 그 context, P_T는 교사 확률, P_S는 학생 확률이다. **계산 예시:** vocabulary token이 3개이고 교사 P_T = (0.7, 0.2, 0.1), 학생 P_S = (0.5, 0.3, 0.2)라고 하자. Soft cross-entropy는 −(0.7 ln 0.5 + 0.2 ln 0.3 + 0.1 ln 0.2) = 0.887이다. 교사 entropy는 0.802이므로 KL divergence는 0.887 − 0.802 = 0.085이다. 샘플링된 토큰 1에 대한 sequence-level SFT는 −ln 0.5 = 0.693만 사용하고 다른 토큰에서의 교사 확률 0.2와 0.1은 무시한다. **증거:** 500B 토큰으로 학습한 2B 모델은 처음부터 학습하면 3개 벤치마크 평균 60.3, 7B 교사에서 증류하면 67.7이다(표 6). 공개된 Gemma 2 2B와 9B는 각각 2T, 8T 토큰에서 이 방식으로 사전학습됐고, 27B는 처음부터 학습했다(§1). Token-level 증류에는 모든 위치에서 교사 확률이 필요하므로 교사 가중치 또는 logit access와 공유 tokenizer가 필요하다. API text output만으로는 충분하지 않다.

**On-policy 증류.** 학생이 시퀀스를 생성하고 교사가 모든 토큰을 채점한다. Qwen3는 0.6B–14B 및 30B-A3B 모델에 두 단계를 쓴다([[qwen-3]], arXiv:2505.09388 §4.5). (1) `/think`, `/no_think` 모드의 교사 출력에 대한 off-policy 증류, (2) “학생 모델이 KL divergence를 최소화하도록 자신의 logits을 교사 모델(Qwen3-32B 또는 Qwen3-235B-A22B)의 logits에 정렬해 미세조정되는” on-policy 증류다. 같은 off-policy 증류 Qwen3-8B에서 시작하고 수학·코드 질의만 쓴 표 21(§4.7)은 다음과 같다.

| 방법 | AIME'24(pass@64) | AIME'25(pass@64) | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-D | GPU 시간 |
|---|---|---|---|---|---|---|---|
| Off-policy 증류 | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | – |
| + RL | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | 17,920 |
| + On-policy 증류 | 74.4 (93.3) | 65.5 (86.7) | 97.0 | 60.3 | 88.3 | 63.3 | 1,800 |

**결과(단일 연구).** 보고서는 pass@64 향상을 교사의 logits이 학생의 탐색을 확장한 결과로 해석한다(§4.7, **해석**). ch-35는 이 단계를 다른 연구소의 pipeline 중 어디에 두는지 설명하고, 목적함수를 별도의 on-policy-distillation 과정(ch-04, ch-05)에 연결한다.

## 음성 샘플과 음성 피드백

**음성 샘플의 출처.** 데이터로서의 증류에서 음성 샘플은 검사를 통과하지 못한 교사 샘플이다. 답 불일치(Sky-T1 exact match, Math-Verify), 모델 judge 판정(gpt-4o-mini, Llama-3.3-70B-Instruct, Listing 4를 쓰는 DeepSeek-V3), unit-test 실패, 가독성 규칙(R1의 혼합 언어, 긴 문단, 코드 블록)이 여기에 해당한다. Labeler의 보고된 오류율은 다음과 같다. Sky-T1 parser는 정답 수학 풀이의 25%를 유지했고 judge는 73%를 유지했다([[bespoke-stratos]]). 공개 CodeForces test는 전체 test에서 실패한 풀이 7개를 통과시켰다([[openr1]]).

**현재 관행이 이들을 다루는 방식.** 이 장의 모든 출처는 (1) 음의 한계가치 방식을 쓴다. 실패 샘플을 버리고 학습에는 전혀 넣지 않는다. 어느 출처도 이를 내용, conditioning, gradient로 사용하지 않는다. R1의 RL 단계는 negative advantage를 사용하지만 증류 세트는 사용하지 않는다([[deepseek-r1]], Findings).

**메커니즘: SFT는 위로만 민다.** Logit z와 softmax p에서 target token y의 log-probability gradient는 다음과 같다.

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

여기서 1[j = y]는 target token이면 1, 그렇지 않으면 0이다. 계산 예시: p = (0.7, 0.2, 0.1)이고 target y = token 2(p = 0.2)라 하자. Gradient는 (−0.7, +0.8, −0.1)이다. −log p_y에 대한 gradient step은 z_2를 올리고, 다른 logit은 각 확률에 비례해 낮춘다. 같은 샘플을 negative gradient로 썼다면 부호는 (+0.7, −0.8, +0.1)로 뒤집힌다. 제거된 확률은 대부분 이미 가장 확률이 높은 token 1로 간다. 이것이 낮은 확률 샘플의 negative gradient가 확률 질량을 현재 mode에 집중시키는 이유다(ch-43a가 이를 완전히 유도한다). 폐기는 이 효과를 피하지만 실패 안의 정보도 버린다.

**수치를 포함한 증거.**
1. 폐기의 이득은 학생 크기에 따라 달라진다. Verification은 7B 평균을 45.0 → 41.9로 낮추고 32B 평균을 62.1 → 64.5로 올렸다([[open-thoughts]], 표 15). 16,000개 코드 예제에서 LLM 생성 unit-test filtering은 LCB 36.0이었고, 필터링하지 않으면 38.5였다(표 18).
2. 초기 대 후기 학습: 엄격한 filtering은 40 step에서 도움이 됐다(MATH-500 0.72 대 0.61). 이후 격차는 닫혔다([[openr1]], Update #3).
3. Negative gradient와 전이: Qwen3-8B ablation에서 off-policy RL(질의당 Qwen3-32B 응답 n = 8개에 KL을 둔 GRPO 목적함수이고, gradient에는 최고 reward 응답을 사용, §5.2)은 Qwen3-32B trace에 대한 off-policy SFT의 TI_non −40.5와 비교해 +4.5였다([[transferability-of-llm-reasoning]], 표 4). 저자들은 이 결과의 제목을 “Credit assignments and negative examples matter”로 단다(§5.2, **해석**). 이 ablation은 negative-gradient term을 분리하지 않으므로 음성 샘플만의 증거는 아니다.

**통제.** Parser 기반 filter를 쓸 때에는 judge fallback 또는 답 재파싱으로 false negative를 폐기 전에 복구한다([[bespoke-stratos]]; [[openr1]], Update #3). 정답 최종 답이 무효 추론에서 나올 수 있다면, 추론도 검사하는 reference-answer judge가 이 경우를 겨냥한다(Listing 4). 실패를 학습 신호로 쓸 때 ch-31a의 선택지(content로서의 correction, failure conditioning, NLL anchor를 둔 unlikelihood)는 음의 항을 제한한다. 반면 증류 SFT loss에 제한 없는 음의 항을 더하면 위에서 유도한 concentration 효과가 나타난다.

**진단.** 도메인별·출처별 filter retention, false negative를 확인할 거부 trace의 수작업 표본, 목표 학생 크기에서 filter 유무로 학습한 학생, 학생의 pass@1과 pass@k(ch-00, ch-43)를 확인한다.

**일반성에 미치는 효과.** False negative를 가진 answer filter는 답을 파싱하기 어려운 문제를 제거하고 질문 분포를 바꿀 수 있다. 이 장의 어떤 출처도 그 shift를 측정하지 않는다(**해석**). OpenThoughts는 31,600개 예제에서 수학에 대한 answer-filter 이득을 측정하지 못했다(표 7). 안전 데이터 없이 학습한 OpenThinker3-7B는 base보다 HarmBench harmfulness rate가 높지만(§5.4), 저자들은 어떤 데이터 변경이 원인인지 시험하지 않았다.

## 레시피

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 증거 |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1(Dev3 SFT 세트) | 671B MoE 교사 | distill-SFT 데이터 | 도메인별 샘플(평균 토큰) | 수학 395,285(6,094.2); 코드 211,129(7,435.7); STEM 10,124(4,928.8); 논리 10,395(2,739.0); 일반 177,812(1,419.8); 총 804,745(5,355.3) | arXiv:2501.12948v2 보충 자료 B.3.3 표 5([[deepseek-r1-recipe]]) | 검증됨 2026-09-14 | §4 표 3: Dev2 → Dev3 AlpacaEval 2.0 55.8 → 62.1 |
| DeepSeek-R1(Dev3 SFT 세트) | — | distill-SFT 데이터 | 추론 필터 | 프롬프트당 다수 샘플, 정답 유지; 일부에서 참조 답에 대한 V3 judge(Listing 4); 혼합 언어·긴 문단·코드 블록 CoT 제거 | v2 B.3.3 | 검증됨 2026-09-14 | ablation 보고 없음 |
| DeepSeek-R1(Dev3 SFT 세트) | — | distill-SFT 데이터 | 프롬프트당 샘플; sampling temperature; yield | 보고하지 않음 | 확인한 v2 §3, B.3.3, B.4 | 보고하지 않음 | — |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | distill-SFT | base; 초기 LR | Qwen2.5-Math-1.5B; 1×10⁻⁴ | v2 B.4.3 표 6 | 검증됨 2026-09-14 | ablation 보고 없음 |
| DeepSeek-R1-Distill-Qwen-7B | 7B | distill-SFT | base; 초기 LR | Qwen2.5-Math-7B; 8×10⁻⁵ | v2 B.4.3 표 6 | 검증됨 2026-09-14 | ablation 보고 없음 |
| DeepSeek-R1-Distill-Qwen-14B | 14B | distill-SFT | base; 초기 LR | Qwen2.5-14B; 7×10⁻⁵ | v2 B.4.3 표 6 | 검증됨 2026-09-14 | ablation 보고 없음 |
| DeepSeek-R1-Distill-Qwen-32B | 32B | distill-SFT | base; 초기 LR | Qwen2.5-32B; 6×10⁻⁵ | v2 B.4.3 표 6 | 검증됨 2026-09-14 | 표 16: AIME 2024 72.6, Qwen2.5-32B-Base에서 RL한 모델은 47.0 |
| DeepSeek-R1-Distill-Llama-8B | 8B | distill-SFT | base; 초기 LR | Llama-3.1-8B; 5×10⁻⁵ | v2 B.4.3 표 6 | 검증됨 2026-09-14 | ablation 보고 없음 |
| DeepSeek-R1-Distill-Llama-70B | 70B | distill-SFT | base; 초기 LR | Llama-3.3-70B-Instruct; 2×10⁻⁵ | v2 B.4.3 표 6 | 검증됨 2026-09-14 | v1 §2.4: Llama-3.3 추론이 Llama-3.1보다 “slightly better” |
| R1-Distill 모델 6개 모두 | 1.5B–70B | distill-SFT | epoch; schedule; context; batch | 2–3; 초기 LR의 1/10까지 cosine; 32,768토큰; 64(단위는 명시하지 않음) | v2 B.4.3 | 검증됨 2026-09-14 | ablation 보고 없음 |
| Bespoke-Stratos-32B | 32B | distill-SFT | 예제; 교사; 수학 필터 | 16,710; DeepSeek-R1; ground truth에 대한 gpt-4o-mini judge | HF 데이터셋 카드; 블로그 “Data Curation”([[bespoke-stratos]]) | 검증됨 2026-09-14 | 유지한 정답 풀이 25%(parser) → 73%(judge) |
| Bespoke-Stratos-32B | 32B | distill-SFT | base; LR; schedule; warmup; batch; epoch | Qwen2.5-32B-Instruct; 1e-5; cosine; 비율 0.1; 시퀀스 96개; 3 | HF model card “Training hyperparameters” | 검증됨 2026-09-14 | ablation 보고 없음 |
| Bespoke-Stratos-32B | 32B | distill-SFT | 연산; 교사 temperature | 8×H100, 27 h; temperature 보고하지 않음(스크립트는 `temp: 0.0`을 전달하지만 실행값으로 확인되지 않음) | model card; `generate_numina_data.py` L78 | 검증됨 2026-09-14 / 보고하지 않음 | — |
| Sky-T1-32B-Preview | 32B | distill-SFT | 교사; 데이터 | QwQ-32B-Preview; 코드 5k(APPs, TACO), 수학 10k(NuminaMATH AIME, MATH, Olympiads), STILL-2 1k; GPT-4o-mini가 trace를 재작성 | NovaSky 블로그 “Data Curation Process”([[sky-t1]]) | 검증됨 2026-09-15(블로그 본문) | 블로그 “Data mixture matters”: 균형 혼합에서 AIME24 43.3, APPs 코드를 추가한 뒤 36.7 |
| Sky-T1-32B-Preview | 32B | distill-SFT | base; epoch; LR; batch; 연산 | Qwen2.5-32B-Instruct; 3; 1e-5; 96; 8×H100에서 19 h, DeepSpeed ZeRO-3 offload | 블로그 “Training” | 검증됨 2026-09-15(블로그 본문) | ablation 보고 없음 |
| Sky-T1-32B-Preview | 32B | distill-SFT | 교사 temperature; 최대 토큰; 최대 시퀀스 길이 | 보고하지 않음 | 확인한 블로그 | 보고하지 않음 | — |
| OpenR1-Math-220k | — | distill-SFT 데이터 | 프롬프트; 샘플; 최대 길이; temperature | NuminaMath 1.5 400k개; 2개(일부 4개); 16k 토큰; 인쇄되지 않음 | HF Update #2; 데이터셋 카드([[openr1-recipe]]) | 검증됨 2026-09-14; temperature 보고하지 않음 | 문제의 75%만 8k 토큰 안에서 풀림 |
| OpenR1-Qwen-7B | 7B | distill-SFT | base; 데이터; epoch; LR; schedule | Qwen2.5-Math-7B-Instruct; `default` 93,733행; 3; 5e-5; linear, warmup 10% | Update #2 | 검증됨 2026-09-14 | 데이터셋 카드: SFT 뒤 `default` > `extended` |
| OpenR1-Distill-7B | 7B | distill-SFT | 데이터; epoch; LR; batch; 최대 길이; packing; 하드웨어 | Mixture-of-Thoughts, 349,317행(수학, 코드, 과학); 5; 4.0e-5; 시퀀스 128개; 32,768; off; 8×H100, DeepSpeed ZeRO-3 | `recipes/OpenR1-Distill-7B/sft/config_distill.yaml`@1416fa0; Mixture-of-Thoughts 카드; model card | 검증됨 2026-09-14 | Update #3: packing은 ablation한 모든 데이터셋에서 더 나쁨 |
| OpenThinker3-7B | 7B | distill-SFT | 예제; 교사; 문제당 답 | 1.2M(수학 850K, 코드 250K, 과학 100K); QwQ-32B; 16 | arXiv:2506.04178v2 §5, §4.4, §4.6([[open-thoughts]]) | 검증됨 2026-09-14 | 표 6 및 8(31,600개 예제 run) |
| OpenThinker3-7B | 7B | distill-SFT | LR; batch; epoch; packing | 8e-5; 512(단위 명시 안 함); 5; 예 | 부록 D.2 표 9 “Large” | 도출됨(1.2M은 > 31.6K 범위) | 규모별 sweep 결과 보고 안 함 |
| Orca-13B | 13B | distill-SFT | 데이터; epoch; 길이; loss | FLAN-5M ChatGPT 후 FLAN-1M GPT-4; 단계당 4; 2,048토큰, 시퀀스당 예제 2.7개 packing; 교사 토큰에 loss | arXiv:2306.02707v1 §3.1.3, §3.2([[orca]]) | 검증됨 2026-09-14 | 표 10: AGIEval 41.7 대 37.18(FLAN-1M만) |
| Orca-13B | 13B | distill-SFT | optimizer; LR; batch | 보고하지 않음 | 확인한 §3.2, §4, 각주 | 보고하지 않음 | — |
| Orca-2-13B | 13B | distill-SFT | 단계; epoch; 길이 | FLAN-v2(1) → ChatGPT 5M(3) → GPT-4 1M + Orca 2 약 817K(4); 4,096 packed | arXiv:2311.11045v2 §4.2([[orca-2]]) | 검증됨 2026-09-14 | ablation 보고 없음 |
| T5-Large 학생(DSBS) | 770M | distill-SFT | LR; batch; 최대 입력; step; λ | 5×10⁻⁵; 64; 1,024; 10,000; λ 보고 안 함(코드 α = 0.5) | arXiv:2305.02301v2 부록 A.1; repo README([[distilling-step-by-step]]) | 검증됨 2026-09-14 / λ 보고 안 함 | 표 2(220M): 4개 데이터셋에서 multi-task > single-task |
| UniReason-Qwen3-14B(SFT) | 14B | distill-SFT | 데이터; 교사 | 수학 문제 47K; rejection sampling으로 유지한 Qwen3-32B trace | arXiv:2507.00432v2 §2.2, 부록 A.3.2([[transferability-of-llm-reasoning]]) | 검증됨 2026-09-15(논문 본문) | 표 1: 비추론 평균 45.7 → 21.1(SFT), RL은 53.2 |
| MathIF §5.2 SFT run | 1.5B, 7B | distill-SFT | 교사; 필터; 예제; 최대 길이; LR | QwQ-32B; 오답 및 8,192토큰 초과 CoT 제거; 18k; 8,192; 1e-6 | arXiv:2505.14810v2 §5.2, 표 7([[scaling-reasoning-losing-control-mathif-recipe]]) | 검증됨 2026-09-14 | 표 4 |
| Qwen3-8B | 8B | distill-SFT(on-policy) | 교사; 목적함수; 연산 | Qwen3-32B 또는 Qwen3-235B-A22B; 학생 logits을 교사에 정렬, KL 최소화; 1,800 GPU 시간(RL: 17,920) | arXiv:2505.09388v1 §4.5; §4.7 표 21([[qwen-3]]) | 검증됨 2026-09-15(보고서 본문) | 표 21 |
| Gemma 2 2B | 2B | pretrain-stable | 목적함수; 토큰 | 교사 다음 토큰 분포에 대한 cross-entropy; 2T 토큰 | arXiv:2408.00118v3 §3.1, §3.2([[gemma-2]]) | 검증됨 2026-09-14 | 표 6: 처음부터 학습 60.3 대비 67.7(500B-token ablation) |

**소형 범용 run의 시작점.** R1 스타일 trace에 SFT하는 7B 학생에 대해, 검증된 행은 두 가지 시험 설정을 준다. OpenR1-Distill-7B는 8×H100에서 수학·코드·과학 349,317행에 5 epoch, LR 4.0e-5, step당 128 sequence, 길이 32,768토큰, packing 없음으로 학습한다. R1-Distill-Qwen-7B는 804,745개 샘플에 2–3 epoch, 초기 LR 8×10⁻⁵를 10분의 1까지 decay, batch 64, 32,768-token context로 학습한다. 두 출처는 수학·코드·과학만 평가하므로 어느 설정도 일반 능력을 기준으로 선택되지 않았다. 범용 학생의 데이터 측면에는 공개된 기준점이 하나 있다. R1의 804,745개 중 일반 샘플 177,812개(표 5)다. 학생이 범용으로 남아야 한다면 epoch 선택 평가에는 IFEval과 안전 세트가 포함되어야 한다. 두 출처의 선택에는 둘 다 포함되지 않았기 때문이다.

## 일반화 관점

**(a) 폭을 늘리는 것.**
- 증류 세트에 도메인을 섞는 것: Open R1의 Mixture-of-Thoughts는 전 도메인을 함께 학습하는 것이 최고임을 발견했다([[openr1]]). 과학 데이터는 코드 전용 36.7–47.3에서 GPQA-D를 52.7로 올렸다([[open-thoughts]], 부록 H.5). R1의 비추론 및 SWE 데이터는 AlpacaEval 2.0과 Aider-Polyglot을 올렸다(§3).
- Off-policy 증류 뒤 on-policy 증류: Qwen3-8B는 MMLU-Redux 86.4 → 88.3과 AIME pass@64를 포함한 보고된 6개 벤치마크 모두에서 향상했다(§9).
- 교사에게 요청한 다양한 출력 형식(Orca의 16개 시스템 메시지)은 학생에게 짧고 긴 답변 모드를 준다(§1). 출처 중 어느 것도 효과를 별도로 측정하지 않는다.

**(b) 협소화 또는 forgetting을 일으키는 것.**
- 수학 전용 trace SFT: Qwen3-14B-Base의 비추론 평균 45.7 → 21.1, IFEval 69.2 → 42.3([[transferability-of-llm-reasoning]], 표 1). 4개 MathIF SFT-only run 전체와 RL을 포함한 16개 학습 변형 중 15개에서 HAcc와 SAcc가 하락했다([[scaling-reasoning-losing-control-mathif]], 표 4). 수학 전용 프롬프트를 쓴 독립 연구 2개에서 **재현됨**.
- 안전 데이터 없음: OpenThinker3-7B HarmBench harmfulness 14.5 → 55.5(§5.4).
- 대화 데이터 없음: Orca-2-13B MT-Bench 6.15 대 LLaMA-2-Chat-13B 6.64(§1); R1 세트는 평균 single-turn이다(§3).
- 균형 없는 혼합: Sky-T1 수학 데이터에 APPs 코드를 넣으면 AIME24가 43.3에서 36.7로 떨어지고, 더 어려운 수학 및 TACO 코드를 더한 뒤 회복했다(§4.3).

**(c) 이 단계의 측정 방법.**
- 교사 데이터 뒤의 벤치마크: AIME 2025, HMMT 02/25, 교사 공개 뒤 LiveCodeBench window([[open-thoughts]], §4; [[deepseek-r1]], E.2).
- 비추론 및 제약 suite: IFEval, MathIF HAcc/SAcc, CoQA, HaluEval; TI_other와 TI_non으로 요약(§8).
- 안전성과 over-refusal: 증류 전후 HarmBench 및 XSTest(§5.4).
- 암기 답과 추론을 분리할 perturbed problem([[open-thoughts]], 부록 N).
- 알려진 측정 오류: GPT-4-as-judge가 imitation model을 과대평가(§5.2); AIME 2024는 30문제이므로 Open R1은 64개 샘플에 pass@1을 평균([[openr1]]); 10-gram decontamination은 paraphrase를 놓침([[deepseek-r1]], D.1).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| 답 parser가 정답 trace를 거부 | 유지율이 교사의 측정 정확도보다 낮음; 어렵거나 형식이 특이한 문제가 과소대표됨 | 거부 trace 100개에 수작업 label; judge fallback과 retention 비교([[bespoke-stratos]]에서 25% 대 73%) |
| 교사의 목표 benchmark에서만 학생 평가 | 수학·코드 향상은 보고되지만 IFEval, chat, 안전 수치 없음 | IFEval, 비추론 QA 세트, HarmBench/XSTest를 추가하고 TI_non 계산 |
| 범용 모델에 수학 전용 증류 세트 | SFT 뒤 IFEval과 비추론 점수 하락 | base와 학생에 같은 평가 실행; 일반 샘플 추가(R1: 샘플의 22%) |
| chat template에서 `<think>` prefill 누락 | out-of-domain prompt가 reasoning trace를 내지 않음 | 학습 도메인 밖 prompt를 prefill 유무로 시험([[openr1]], Lesson 4) |
| benchmark 점수만으로 교사 선택 | 더 낮은 점수 교사가 더 좋은 학생을 만듦 | 후보마다 31.6K 규모 샘플로 작은 학생 학습([[open-thoughts]], 표 8) |
| 한 학생 크기에서 recipe를 선택하고 다른 크기에 적용 | 새 크기에서 filter 또는 데이터 선택이 뒤집힘 | 목표 크기에서 핵심 ablation 반복(verification: 7B −3.1, 32B +2.4, [[open-thoughts]] 부록 H.1.1 표 15) |
| 공개 test만 코드 필터로 사용 | 학습 세트 pass rate는 높지만 hidden test 실패 | full test set에서 표본 재실행([[openr1]], Update #3) |
| 오염된 평가 | 교사 이전 benchmark에서 큰 향상, 새 benchmark에서 작은 향상 | 교사 데이터 뒤 날짜 benchmark 보고; n-gram 및 similarity decontamination 실행 |
| 약관이 금지한 교사에서 데이터 생성 | 데이터셋을 공개하거나 상업적으로 쓸 수 없음 | 생성 시점에 license file 또는 약관 읽기([[teacher-output-licenses]]) |

## 이해 점검

1. R1의 일반 데이터는 샘플의 22.1%지만 토큰의 5.9%다. Loss normalization(per sequence 대 per token)이 학생에 대한 일반 데이터의 영향을 어떻게 바꾸는지, 그리고 일반 비중 선택에 무엇을 뜻하는지 설명하라.
2. R1-Distill-Qwen-32B는 표 16의 모든 benchmark에서 Qwen2.5-32B-Base의 RL을 이기지만, Huan et al.에서는 RL이 비추론 과제에서 SFT를 이긴다. 각 연구의 prompt distribution과 evaluation suite를 사용해 두 결과가 함께 성립하는 이유를 설명하라.
3. gpt-4o-mini judge는 유지된 정답 수학 풀이를 25%에서 73%로 올렸다. Parser가 어떤 문제를 가장 많이 버렸을지, 그리고 그 폐기가 SFT 세트의 난이도와 형식 분포를 어떻게 바꾸는지 설명하라.
4. TI 계산 예시에서 benchmark C를 50 → 55로 바꿔라. DI_non과 TI_non을 다시 계산하고, 제곱근과 난이도 가중치가 raw gain 평균과 비교해 결과를 왜 바꾸는지 설명하라.
5. Token-level 증류(Gemma 2)를 closed API 교사에 적용할 수 없는 이유와 sequence-level SFT는 적용할 수 있는 이유를 설명하고, sequence-level 경우 학생이 잃는 정보를 명시하라.
6. OpenThoughts의 verification filter는 7B에는 해롭고 32B에는 도움이 됐다. 시험 가능한 인과 설명을 제안하고 실험을 기술하라.
7. 실패한 교사 trace를 버리는 것이 negative gradient의 probability-concentration 효과를 피하는 이유와, 버릴 때 잃는 정보를 설명하라.
8. Orca 2는 Prompt Erasing을 사용한다. 전략 지시가 제거됐을 때 학생이 무엇을 학습해야 하는지와, MT-Bench 점수가 instruct baseline보다 낮아질 수 있는 이유를 설명하라.

## 연결

- 이전: **ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing.** Prompt 측 생성과 imitation-versus-capability 비판을 다룬다. 이 장은 더 강한 교사가 target을 쓸 때의 response 측을 다룬다.
- 다음: **ch-21 — Taxonomy-Driven and Textbook-Style Synthesis.** 교사의 reasoning trace 대신 taxonomy로 coverage를 제어한다.
- **ch-24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data.** Trace 생성 방법과 step-level 데이터.
- **ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation.** 모델 자신의 sample에 대한 rejection sampling과 R1 cold start.
- **ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood.** 이 장이 버리는 trace의 사용법.
- **ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control.** §8의 손실 측정.
- **ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data.** Logit 및 on-policy 증류를 포함한 연구소별 stage map.
- **ch-35a — Distillation in Practice B: Prompt Selection, Teacher Sampling, and Quality Filters.** 나란히 비교한 distillation-SFT 설정과 filter.
- **ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.** §8.2 뒤의 더 넓은 SFT-versus-RL 증거.
- **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.** Gradient 예시의 전체 유도.
- **ch-44 — Process Supervision and Verifiable Rewards.** §5.5의 step-level 검사.

## 출처

- [[orca]] — 표 2 시스템 메시지, FLAN-v2 샘플링, 두 단계 교사 schedule, BBH/AGIEval 결과, GPT-4-judge 과대평가, Gudibande 인용.
- [[orca-2]] — 전략 선택, Prompt Erasing, 데이터셋 구성, reasoning 평균, MT-Bench, hallucination 및 안전 비율.
- [[distilling-step-by-step]] — multi-task loss, prefix, 표 1–2, ANLI 80% 결과, 학습 설정.
- [[deepseek-r1]] — pipeline, B.3.3 데이터 설명과 Listing 4, 표 15–16, 단계별 일반성 증거, PRM 시도.
- [[deepseek-r1-recipe]] — 표 5 구성 및 표 6 학생 설정.
- [[deepseek-r1-distill-synth]] — R1 repository 포인터; 학생 base 목록은 오래되었다(정정 15).
- [[bespoke-stratos]] — Stratos-17k 구성, gpt-4o-mini filter, 32B·7B 결과, 학습 설정.
- [[sky-t1]] — NovaSky 블로그: 데이터 큐레이션, GPT-4o-mini rewrite, 학습 설정, data-mixture·model-size 발견.
- [[openr1]] — OpenR1-Math-220k 생성·filtering, verification-filter ablation, prefill lesson, CodeForces false positive.
- [[openr1-recipe]] — OpenR1-Qwen-7B 및 OpenR1-Distill-7B 설정.
- [[open-thoughts]] — pipeline ablation, 교사 비교, scaling, reflection removal, 안전, 학생 크기별 verification.
- [[scaling-reasoning-losing-control-mathif]] — HAcc/SAcc, 23-model 결과, 통제 SFT/RL run, IFEval 비교.
- [[scaling-reasoning-losing-control-mathif-recipe]] — MathIF SFT run 설정.
- [[transferability-of-llm-reasoning]] — Transferability Index, Qwen3-14B 통제 연구, Qwen3-8B on/off-policy ablation(arXiv v2 본문에서 읽음; 카드는 수치 없음).
- [[gemma-2]] — logit-distillation 목적함수와 표 6.
- [[qwen-3]] — strong-to-weak 증류 단계와 표 21(arXiv:2505.09388v1 본문에서 읽음).
- [[qwen-qwq-traces]] — OpenThoughts와 MathIF가 쓴 교사 모델 QwQ-32B.
- [[dolphin]] — refusal filtering이 있으나 방법이 명시되지 않은 Orca 스타일 데이터셋.
- [[qwen-2.5]] — 모델 크기별 Qwen2.5 라이선스.
- [[anthropic-distillation-attacks-report]] — closed API 증류의 서비스 약관 위반.
- [[teacher-output-licenses]] — R1, Llama 3.1, QwQ, 데이터셋 라이선스의 인용된 라이선스 설명.
