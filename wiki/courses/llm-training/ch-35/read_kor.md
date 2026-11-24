<!-- chapter: ch-35
     track: sft
     kind: content
     title: Distillation in Practice A: Where Labs Insert Teacher Data
     deps: [ch-34, ch-20]
     sources: [[deepseek-r1]], [[deepseek-r1-recipe]], [[deepseek-v3]], [[deepseek-v3-recipe]], [[deepseek-v3.1]], [[deepseek-v3.1-recipe]], [[deepseek-v4-recipe]], [[glm-4-5]], [[glm-4-5-recipe]], [[glm-5]], [[qwen-3]], [[qwen-3-post-training]], [[kimi-k1-5]], [[kimi-k1-5-recipe]], [[kimi-k2]], [[kimi-k2-recipe]], [[llama-nemotron]], [[nemotron-nano-2]], [[phi-4]], [[phi-4-reasoning]], [[magistral]], [[magistral-recipe]], [[gemma-2]], [[gemma-3]], [[llama-4]], [[minitron-approach]], [[mimo-v2-flash]], [[thinkingmachines-on-policy-distillation]], [[smollm-3]], [[smollm-3-midtraining]]
     figures: figures/teacher-data-stage-map.html
     revised: 2026-09 (generality revision)
-->

# 제35장 — 증류의 실제 A: 연구소가 교사 데이터를 넣는 위치

> **핵심 통찰.** 여기서 다루는 2024–2026 보고서에서 교사 데이터는 사전학습 logit 목표부터 RL 뒤 증류 단계까지 아홉 위치로 학습에 들어간다. 보고서들의 주요 차이는 어느 위치를 쓰는지와 얼마나 공개하는지다. 더 강한 교사를 증류하는 것은 DeepSeek-R1의 32B에서 같은 시작점의 RL을 이겼다(R1-Distill-Qwen-32B AIME 2024 72.6 대 RL 전용 47.0, [[deepseek-r1]] Table 16). Qwen3의 8B에서도 이겼다(온폴리시 증류 1,800 GPU 시간에서 74.4 대 RL 17,920시간에서 67.6, [[qwen-3-post-training]] Table 21). 그러나 24B Magistral Small에서는 RL만으로 AIME'24 pass@1에서 SFT와 같았고(65.8 대 65.4), GPQA에서는 넘었다(68.8 대 63.4) ([[magistral]] Table 3). Llama-Nemotron에서는 지도 증류만으로 교사보다 낮았고, 그 뒤 RL이 교사를 넘겼다(LN-Ultra GPQA-Diamond: SFT 뒤 66.4, 교사 DeepSeek-R1 71.5, RL 뒤 76.0; [[llama-nemotron]] Table 5). 단계별 비대상 능력을 추적한 보고서는 일부 단계에서 손실을 발견했다([[llama-nemotron]] Table 4의 지시 따르기, [[qwen-3-post-training]] Table 22의 경시 수학, [[smollm-3-midtraining]]의 긴 컨텍스트). 그리고 최근 파이프라인 셋은 이전 또는 분리된 RL 학습에서 능력을 복구하거나 결합한다는 명시된 목적으로 증류 단계로 끝난다([[glm-5]] §3.5, [[nemotron-nano-2]] §4.3, [[mimo-v2-flash]] §4.1).
>
> **지침.** 더 강한 교사가 있고 학생이 더 작은 모델이면, 모드 지시 아래 추론 및 비추론 응답을 섞은 필터링된 교사 샘플 SFT부터 시작하라. 이 방식으로 학생을 학습한 보고서에서 증류 모델은 같은 base의 RL 전용 기준선을 넘었다(R1-Distill-Qwen-32B, [[deepseek-r1]] Table 16). 소형 모델 자체에서 4단계 파이프라인을 실행한 경우([[qwen-3-post-training]] §4), reasoning 및 BFCL 행에서는 같은 크기 instruct 모델을 넘지만 IFEval에서는 넘지 않은 경우(LN-Nano-SFT 대 Llama-3.1-8B-Instruct: IFEval 69.9 대 81.8, [[llama-nemotron]] Table 3), 또는 시작 모델을 넘은 경우(Phi-4-reasoning 대 Phi-4, [[phi-4-reasoning]] §1)가 있다. Magistral Small은 RL만으로 AIME'24에서 SFT와 같았던 예외다([[magistral]] Table 3). 학생이 교사를 넘거나 잃은 능력을 복구해야 하면 SFT 뒤 온폴리시 증류 또는 RL을 추가하라. SFT만으로 [[llama-nemotron]] §5에서 교사에 접근했지만 넘지는 못했고, 온폴리시 증류는 RL이 올리지 못한 pass@64를 올렸기 때문이다([[qwen-3-post-training]] Table 21). 파이프라인이 분리된 도메인 전문가 또는 순차 RL 단계를 학습하면 merge 또는 최종 단계 뒤 이전 도메인 모두를 평가하라. 증류 학생은 [[deepseek-v3.1]] §3에서 전문가보다 낮았고 [[mimo-v2-flash]] Table 7의 BrowseComp에서는 최대 6.3점 낮았다. 더 강한 교사가 없으면 모델 자신의 샘플에서 RL을 사용하라. 다른 모델의 추론 trace 없이 RL 전에 학습한 Magistral Medium은 RL만으로 AIME'24 pass@1을 26.8(Mistral Medium 3)에서 73.6으로 올렸기 때문이다([[magistral]] Table 2). 그 외에는 Magistral Small 결과가 하나에 확정하기 전 같은 base에서 RL 전용과 SFT 뒤 RL을 시험하라고 시사한다([[magistral]] Table 3).

## 이 장이 범용 모델에 중요한 이유

교사 데이터는 학습 중인 모델 외의 모델 또는 그 모델의 이전 체크포인트가 만든 모든 학습 목표다. 세 형태가 있다. 교사에서 샘플링해 SFT 목표로 쓰는 시퀀스, soft target으로 쓰는 교사의 다음 토큰 분포, 학생이 샘플링한 시퀀스에서 계산한 교사 로그 확률이다. ch-20은 증류를 데이터 원천으로 다루고 R1-Distill 계보를 따랐다. 이 장은 연구소가 파이프라인(사전학습 → 중간학습 → SFT → 선호 최적화 → RL → 평가)의 어디에 교사 데이터를 넣는지와 각 연구소가 그것에 관해 무엇을 보고했는지를 묻는다.

배치 위치 문제는 일반성에 측정 가능한 결과를 낳는다. 교사 trace는 교사의 길이 습관을 학생에게 실어 나른다. DeepSeek-V3 절제에서 R1 유래 데이터는 MATH-500 평균 응답 길이를 769에서 1510토큰으로 올렸다([[deepseek-v3]] Table 9). 추론 중심 SFT는 Llama-Nemotron의 IFEval을 낮췄고([[llama-nemotron]] Table 4), 범용 단계는 Qwen3의 AIME'24를 낮췄다([[qwen-3-post-training]] Table 22). 순차 단계는 GLM-5 저자가 “cumulative degradation of previously acquired capabilities”라고 묘사한 망각을 일으킨다([[glm-5]] §3.5). 다음 장 ch-35a는 프롬프트 선택, 교사 샘플링, 출력 필터링을 다룬다. 이 장은 단계 배치, 목적함수, 공개 기록을 다룬다.

## §1 교사 신호의 세 형태와 목적함수

### 1.1 정의와 수식

**시퀀스 수준(오프폴리시) 증류**는 교사가 샘플링한 응답에 대한 SFT다. 손실은 다음과 같다.

L_seq(θ) = − E_{x∼D} E_{y∼π_T(·|x)} [ Σ_{t=1}^{|y|} log π_θ(y_t | x, y_<t) ]

여기서 D는 프롬프트 분포, π_T는 교사 정책, π_θ는 매개변수 θ를 가진 학생, y는 응답, y_t는 그 t번째 토큰이다. 그래디언트는 교사가 만든 토큰을 통해서만 흐른다. [[thinkingmachines-on-policy-distillation]]은 시퀀스 샘플링이 교사 분포의 불편 추정치를 주며 완전 분포 매칭과 “arrives at the same objective”라고 말한다(해석, 실무자 근거).

**logit(토큰 분포) 증류**는 각 위치에서 교사의 전체 다음 토큰 분포를 향해 학생을 학습한다. forward KL은 다음과 같다.

L_fwd(θ) = E_{c} [ Σ_{v∈V} p_T(v|c) · log( p_T(v|c) / p_θ(v|c) ) ]

여기서 c는 학습 텍스트에서 취한 컨텍스트, V는 어휘, p_T와 p_θ는 교사 및 학생의 다음 토큰 확률이다. Gemma 2는 교차 엔트로피 Σ_v −p_T(v|c) log p_θ(v|c)를 최소화한다. 이는 L_fwd에 θ에서 상수인 교사 엔트로피를 더한 것과 같다([[gemma-2]] §3.2). Gemma 3은 토큰별 교사 확률로 logit 256개를 샘플링하고 나머지는 0으로 설정한 뒤 재정규화한다. p̃_T(v|c) = p_T(v|c)·1[v∈S_c] / Σ_{u∈S_c} p_T(u|c)이며, S_c는 샘플 집합이다([[gemma-3]] §2.2). Minitron과 Nemotron Nano 2는 언어 모델 손실 없이 logit에만 forward KL을 쓴다([[minitron-approach]] Distillation; [[nemotron-nano-2]] §4.3).

**온폴리시 증류**는 학생에서 y를 샘플링하고 교사의 로그 확률 비율을 토큰별 이점으로 쓴다.

Â_t = sg[ log π_T(y_t | x, y_<t) − log π_θ(y_t | x, y_<t) ],  y ∼ π_θ(·|x)

L_on(θ) = − E_{x∼D, y∼π_θ} [ Σ_t Â_t · log π_θ(y_t | x, y_<t) ]

여기서 sg는 stop-gradient다. 학생 자신의 토큰 선택에 대한 Â_t의 기댓값은 해당 prefix에서 −KL(π_θ ‖ π_T)이므로, L_on은 샘플링된 토큰별 reverse KL이다([[mimo-v2-flash]] Eq. 5–6). GLM-5는 같은 이점을 GRPO 손실에 넣는다(Eq. 2). MiMo-V2-Flash는 중요도 가중치와 결과 보상 항 α·Â_ORM을 더한다(Eq. 7–9). Thinking Machines는 discount 0으로 `advantages = -reverse_kl`을 설정한다([[thinkingmachines-on-policy-distillation]], Pseudocode). Qwen3는 방향을 밝히지 않고 학생 logit을 교사 logit과 정렬해 “to minimize the KL divergence”한다고 쓴다([[qwen-3-post-training]] §4.5). DeepSeek-V4는 열 개가 넘는 교사에 대해 full-vocabulary D_KL(π_θ ‖ π_E)를 쓴다([[deepseek-v4-recipe]], V4 report §5.1.2 Eq. 29).

### 1.2 계산 예: 각 목적함수가 패널티를 주는 대상

한 prefix와 네 토큰 어휘를 보자. 교사 p_T = (0.60, 0.30, 0.09, 0.01), 학생 p_θ = (0.40, 0.40, 0.10, 0.10)이다.

1. Forward KL = 0.6·ln(0.6/0.4) + 0.3·ln(0.3/0.4) + 0.09·ln(0.09/0.1) + 0.01·ln(0.01/0.1) = 0.2433 − 0.0863 − 0.0095 − 0.0230 = 0.1245 nats다. 토큰 4는 교사의 0.01로 가중되므로 −0.0230을 기여한다.
2. Reverse KL = 0.4·ln(0.4/0.6) + 0.4·ln(0.4/0.3) + 0.1·ln(0.1/0.09) + 0.1·ln(0.1/0.01) = −0.1622 + 0.1151 + 0.0105 + 0.2303 = 0.1937 nats다. 토큰 4는 학생의 0.10으로 가중되므로 가장 큰 항 +0.2303을 기여한다.
3. 온폴리시 이점은 Â = ln(p_T/p_θ) = (+0.4055, −0.2877, −0.1054, −2.3026)이다. 학생 가중 평균은 0.4·0.4055 + 0.4·(−0.2877) + 0.1·(−0.1054) + 0.1·(−2.3026) = −0.1937 = −(reverse KL)이다.
4. S_c = {토큰 1, 토큰 2}인 Gemma 3식 샘플링은 목표 (0.667, 0.333, 0, 0)을 준다. S_c 바깥 토큰은 그 위치에서 교사 질량을 받지 않는다.

교사 샘플에 대한 시퀀스 수준 SFT는 이 prefix에서 확률 0.01로 토큰 4를 방문하므로, 학생의 0.10은 드물게만 교정된다. 온폴리시 샘플링은 확률 0.10으로 방문하며 가장 큰 음의 이점을 준다. exposure bias는 교사가 쓴 prefix에서 학습하고 추론 때는 학생이 쓴 prefix에서 생성하는 것 사이의 불일치다. 위 방문 확률의 차이가 온폴리시 증류가 exposure bias를 낮춘다는 주장의 메커니즘이다([[thinkingmachines-on-policy-distillation]]; [[mimo-v2-flash]] §4.1; 해석).

### 1.3 조건

온폴리시 증류에서는 교사가 학생이 이미 생성하는 토큰에 확률을 할당해야 한다. [[thinkingmachines-on-policy-distillation]]은 “SFT, using forwards KL, adds support for new tokens. Reverse-KL methods can then perform mode seeking within the initialization's support.”이므로 먼저 SFT를 실행한다. 전체 파이프라인을 설명하는 이 장의 모든 온폴리시 보고서는 SFT 또는 오프폴리시 증류 체크포인트에서 시작한다([[qwen-3-post-training]] §4.5; [[glm-5]] §3; [[mimo-v2-flash]] Figure 3; [[thinkingmachines-on-policy-distillation]]). Logit 증류에는 공유 토크나이저 또는 어휘 간 매핑이 필요하다. 교사를 밝힌 logit 증류 보고서는 여기서 학생 계열 자체의 교사를 쓴다([[minitron-approach]], [[nemotron-nano-2]], [[llama-4]]). [[gemma-3]]은 교사를 밝히지 않는다.

## §2 아홉 개의 삽입 위치

| # | 위치 | 들어가는 것 | 목적함수 | 보고서(근거 위치) |
|---|---|---|---|---|
| S1 | 사전학습 logit 증류 | 사전학습 텍스트 위 교사의 다음 토큰 분포 | p_T에 대한 forward KL / cross-entropy | Gemma 2 2B 및 9B ([[gemma-2]] §3.2); Gemma 3 전 크기 ([[gemma-3]] §2.2); Llama 4 Maverick codistillation ([[llama-4]]) |
| S2 | 증류를 수반한 pruning | pruning 뒤 원래 모델의 logit | logit에 대한 forward KL | [[minitron-approach]]; LN-Super, LN-Ultra ([[llama-nemotron]] §2.2); Nemotron-Nano-9B-v2 ([[nemotron-nano-2]] §4.3) |
| S3 | 증류 trace 중간학습 | 계속 사전학습에서의 추론 trace 또는 에이전트 궤적 | next-token 손실 | GLM-4.5 합성 추론 단계, 500B 토큰 표기 ([[glm-4-5]] §2.3, Figure 3); SmolLM3 35B토큰 × 4 epoch ([[smollm-3-midtraining]]) |
| S4 | cold start | RL 전 작은 long-CoT SFT 집합 | SFT | R1 ([[deepseek-r1]] B.3.2); Qwen3 ([[qwen-3-post-training]] §4.1); Kimi k1.5 ([[kimi-k1-5]] §2.2); GLM-4.5 전문가 ([[glm-4-5]] §3.1); Magistral Small ([[magistral]] §5.3) |
| S5 | 전문가를 하나의 모델로 | 도메인 전문가 또는 specialist 출력 | SFT | DeepSeek-V3 ([[deepseek-v3]] §5.1); V3.2 ([[deepseek-v3.1]] §3); GLM-4.5 ([[glm-4-5]] §3.1); 전문가 RL과 거부 샘플링에서 나온 GLM-5 코드·에이전트 SFT 데이터 ([[glm-5]] §3.1); Kimi K2 ([[kimi-k2]] §3.1); MiMo-V2-Flash ([[mimo-v2-flash]] §4.2) |
| S6 | RL 뒤 거부 샘플링 SFT | RL 체크포인트의 정답 샘플 | SFT | R1 800K 집합 ([[deepseek-r1]] B.3.3); Qwen3 Stage 3 ([[qwen-3-post-training]] §4.3); GLM-4.5 에이전트 반복 증류 ([[glm-4-5]] §3.3.2) |
| S7 | 소형 모델 증류 | 더 큰 모델의 출력 또는 logit | SFT, 이후 온폴리시 KL | R1-Distill ([[deepseek-r1-recipe]]); Qwen3 strong-to-weak ([[qwen-3-post-training]] §4.5); Llama-Nemotron ([[llama-nemotron]] §3–4); Phi-4-reasoning ([[phi-4-reasoning]] §3); Nemotron Nano 2 ([[nemotron-nano-2]] §3.1); Gemma 3 사후학습 ([[gemma-3]] §3); Thinking Machines 추론 실행 ([[thinkingmachines-on-policy-distillation]]) |
| S8 | 길이 축소 | 가장 짧은 정답 샘플, 잘린 trace, merge | SFT, DPO, 길이 패널티 RL | Kimi k1.5 long2short ([[kimi-k1-5]] §2.4); Nemotron Nano 2 잘린 trace ([[nemotron-nano-2]] §3.2); DeepSeek-V3의 명시한 목표 ([[deepseek-v3]] §5.1) |
| S9 | RL 뒤 증류 | 학생 샘플 위 이전 단계 또는 도메인 교사; 또는 RL 뒤 logit | 온폴리시 reverse KL; forward-KL logit | GLM-5 ([[glm-5]] §3.5); MiMo-V2-Flash ([[mimo-v2-flash]] §4.4); DeepSeek-V4 ([[deepseek-v4-recipe]]); Nano 2 post-GRPO KD ([[nemotron-nano-2]] §4.3); RL이 아닌 중간학습 뒤 Thinking Machines 복구 ([[thinkingmachines-on-policy-distillation]]) |

인터랙티브 맵 [figures/teacher-data-stage-map.html](figures/teacher-data-stage-map.html)에서는 보고서와 위치를 하나씩 선택해 검증한 세부 및 근거 위치를 읽고, 목적함수 유형별로 행을 강조할 수 있다. 그곳의 회색 칸은 보고서를 확인했으나 그 위치에 교사 데이터를 기술하지 않는다는 뜻이다. 이는 공개 공백이며, 부재의 증거는 아니다.

## §3 R1에서 V4까지의 DeepSeek

**3.1 R1 cold start(S4).** R1의 cold start는 긴 CoT 예 “thousands”를 사용한다([[deepseek-r1]] §3, B.3.2). 교사는 같은 프로젝트의 이전 모델이다. “thousands of high-quality, diverse reasoning prompts”에 대해 DeepSeek-R1-Zero가 온도 1.0으로 여러 궤적을 생성하고, 정답 최종 답과 읽기 쉬운 형식의 생성을 남긴다(수학은 sympy, 반복 탐지 및 언어 혼합 필터). DeepSeek-V3에는 추론과 요약을 다듬도록 프롬프트한다(B.3.2). 사람 단계도 같은 절에 명시된다. “we first engage human annotators to convert the reasoning trace into a more natural, human conversational style. The modified data pairs are then used as examples to prompt an LLM to rewrite additional data in a similar style. All LLM-generated outputs subsequently undergo a second round of human verification.” 명시한 동기는 추론 정확도가 아니라 1인칭 추론과 언어 일관성이라는 “primarily product-driven” 것이다. DeepSeek-V3-Base SFT는 2–3 epoch, cosine LR 5×10⁻⁵ → 5×10⁻⁶, 32,768토큰, 배치 128로 실행했다(B.4.2). R1-Zero가 아니라 V3-Base에서 학습한 cold-start 체크포인트 Dev1은 IF-Eval 71.7(R1-Zero 46.6), AIME 2024 59.0(R1-Zero 77.9)을 기록했다. 저자들은 AIME 격차를 “the limited size of the cold-start dataset” 때문이라고 돌린다(§4, Table 3).

**3.2 RL 프롬프트 혼합.** 추론 RL 프롬프트는 Math 26K, Code 17K(본문: 알고리즘 17k와 bug-fixing 문제 8k), STEM 22K, Logic 15K다. 최종 혼합 RL 단계는 helpfulness 질문 66k와 harmlessness 질문 12,000개를 더한다([[deepseek-r1-recipe]], B.3.1 Table 4).

**3.3 800K 거부 샘플링 SFT 집합(S6).** 추론 데이터는 “rejection sampling from the checkpoint of the first-stage RL training”에서 오며, “for each prompt, we sample multiple responses and retain only the correct ones”이다. “with mixed languages, long paragraphs, and code blocks”인 사고 연쇄는 필터링한다(B.3.3). 추가한 추론 데이터 일부는 2수준 분류 프롬프트로 기준 답에 대해 DeepSeek-V3가 채점한다(Listing 4: “**correct**: The answer fully aligns with the reference answer in both reasoning process and final conclusion”). 비추론 데이터는 DeepSeek-V3 SFT 집합 일부와 소프트웨어 공학 데이터를 재사용한다. 일부 과제에는 V3 생성 CoT가 있고 “hello” 같은 단순 질의에는 없다. SFT는 RL 체크포인트가 아니라 DeepSeek-V3-Base에서 재시작한다(Fig. 2).

Table 5의 계산 예(샘플, 평균 토큰)는 Math 395,285(6,094.2), Code 211,129(7,435.7), STEM 10,124(4,928.8), Logic 10,395(2,739.0), General 177,812(1,419.8), 전체 804,745(5,355.3)다. General 데이터는 177,812 / 804,745 = 샘플의 22.1%이지만, 177,812 × 1,419.8 = 252.5M은 전체 4,309.7M토큰의 5.9%다(계산값). Math는 샘플의 49.1%, 토큰의 55.9%다(계산값). 손실을 배치의 모든 토큰에 걸쳐 평균 내면 General 데이터는 손실 항의 약 5.9%를 기여한다. 샘플별로 먼저 평균 내면 약 22.1%다(해석. 보고서는 손실 정규화도 “Avg Tokens”가 프롬프트 토큰을 포함하는지도 밝히지 않는다). Dev2와 Dev3 사이 AlpacaEval 2.0은 55.8에서 62.1, Aider-Polyglot은 25.6에서 44.8로 움직였다(Table 3). 저자들은 이를 “the inclusion of large-scale non-reasoning corpora and code engineering datasets”에 돌린다(§4). 그러나 Dev3는 Dev2 위에 만든 것이 아니라 V3-Base에서 재학습했으므로 통제된 절제 실험이 아니다.

**3.4 R1-Distill(S7).** 여섯 학생은 같은 집합에서 2–3 epoch, 초기 LR의 10분의 1까지 cosine 감쇠, 32,768토큰, 배치 64로 미세조정한다. Qwen2.5-Math-1.5B(LR 1×10⁻⁴), Qwen2.5-Math-7B(8×10⁻⁵), Qwen2.5-14B(7×10⁻⁵), Qwen2.5-32B(6×10⁻⁵), Llama-3.1-8B(5×10⁻⁵), Llama-3.3-70B-Instruct(2×10⁻⁵)이다([[deepseek-r1-recipe]], B.4.3 Table 6). 32B에서 Qwen2.5-32B-Base를 10K단계 넘게 RL한 결과는 AIME 2024 47.0, MATH-500 91.6, GPQA 55.0, LiveCodeBench 40.2였다. Distill-Qwen-32B는 72.6, 94.3, 62.1, 57.2에 도달했다(Table 16). 결과(단일 연구)다. 저자들은 “advancing beyond the boundaries of human intelligence may still require more powerful base models and larger-scale reinforcement learning”이라고 덧붙인다(F.1, 해석).

**3.5 DeepSeek-V3 전문가 모델(S5, S8).** V3의 최종 모델은 R1 출력으로 직접 학습하지 않았다. 도메인별 전문가 모델에 두 표본 유형, 즉 “<problem, original response>”와 “<system prompt, problem, R1 response>”의 SFT를 학습했다. system prompt는 반성과 검증을 요구한다. 뒤이어 고온 샘플링 RL을 했고, “after hundreds of RL steps” 전문가의 거부 샘플링이 1.5M 인스턴스 SFT 집합의 추론 부분을 만들었다([[deepseek-v3]] §5.1). 명시된 목표는 R1 정확도를 유지하면서 “overthinking, poor formatting, and excessive length”를 피하는 것이었다. DeepSeek-V2.5 절제(§5.4.1: short-CoT 데이터 기준선 대 전문가 체크포인트 데이터 학습)는 여전히 길이 비용을 보인다. MATH-500은 평균 길이 769 → 1510토큰과 함께 74.6 → 83.2, LiveCodeBench-CoT는 718 → 783과 함께 31.1 → 37.4다(Table 9). V3 최종 SFT는 2 epoch, cosine 5×10⁻⁶ → 1×10⁻⁶, sample masking을 한 packing이다(§5.1).

**3.6 V3.2 specialist와 V4 온폴리시 증류(S5, S9).** V3.2는 같은 base에서 수학, 프로그래밍, 일반 논리 추론, 일반 에이전트 과제, 에이전트 코딩, 에이전트 검색, 글쓰기와 일반 QA의 specialist를 미세조정한다. 각각 thinking과 non-thinking 모드가 있고 각각 RL로 학습한다([[deepseek-v3.1]] §3). 증류 데이터를 학습한 모델은 specialist보다 “only marginally below” 점수를 얻고, 뒤따르는 단일 mixed RL 단계가 이 격차를 없앤다. 수치나 데이터 크기는 인쇄되지 않았다([[deepseek-v3.1-recipe]]). DeepSeek-V4는 mixed RL 단계를 full-vocabulary logit을 쓰는 열 개 넘는 교사에 대한 다중 교사 온폴리시 증류로 대체한다. 교사 가중치, 단계, 학습률은 보고되지 않았다([[deepseek-v4-recipe]]).

## §4 GLM-4.5와 GLM-5

**4.1 GLM-4.5 중간학습(S3).** 32K의 중간학습 단계 하나는 웹페이지와 책에서 온 문제 및 “reasoning processes with a reasoning model”을 포함한 “synthetic reasoning content for math, science, and coding competitions”를 더한다. Figure 3은 이를 500B토큰으로 표기한다([[glm-4-5]] §2.3). 128K 단계(100B토큰)는 대규모 합성 에이전트 궤적을 더한다. 보고서는 reasoning 모델의 이름을 밝히지 않으며 이 단계의 효과를 분리하지 않는다.

**4.2 전문가를 하나의 모델로(S4, S5, S6).** Stage 1은 Reasoning, Agent, General-chat expert를 학습한다. 각각 작은 cold-start SFT 집합 뒤 RL을 한다. Stage 2는 “employ[s] self-distillation techniques to integrate multiple experts”한다. expert에서 온 “millions of samples”의 128K SFT를 하며, 한 모델이 두 모드를 갖게 하려고 완전한 추론을 포함한 데이터와 “lacking explicit thought processes” 데이터를 균형 있게 넣는다([[glm-4-5]] §3, §3.1). 거부 샘플링은 (1) 반복적·짧은·잘린·형식이 나쁜 샘플, (2) 객관식 오답, (3) 보상 모델이 필터링한 주관적 응답, (4) 프로토콜을 깨거나 terminal state를 놓친 도구 궤적을 제거한다. 벤치마크 이름 없이 두 데이터 선택 결과가 보고된다. 응답 길이 기준 하위 50% 프롬프트를 버리면 “despite training with only half the data” 수학·과학에서 +2%–4%를 얻었고, 어려운 프롬프트당 응답 넷은 추가 +1%–2%를 주었다([[glm-4-5-recipe]]). 에이전트에서는 RL과 자기증류를 교대한다. RL이 평탄해지면 cold-start 데이터를 RL 모델 응답으로 교체하고 SFT를 반복한 뒤 더 어려운 과제에서 RL을 계속한다(§3.3.2).

**4.3 GLM-5 온폴리시 단계 간 증류(S9).** GLM-5(전체 744B, 활성 40B)는 SFT 다음 Reasoning RL, Agentic RL, General RL을 순서대로 실행한다([[glm-5]] §2.1, §3). SFT 수학·과학 문제는 “challenging for the GLM-4.7 model”인 경우에만 남기며, 에이전트 SFT 데이터는 오류 궤적 구간을 “masked out in the loss function”으로 둔다(§3.1). 최종 단계는 앞선 학습 단계들의 “the final checkpoints”를 교사로 쓰고, 해당 RL 집합의 프롬프트를 “mixed in appropriate proportions”로 섞는다. §1.1의 Eq. 2 이점, GRPO 그룹 크기 1, 배치 1,024를 사용한다. 이는 “because it is no longer necessary to maintain a large group of samples per prompt to estimate advantages”이기 때문이다(§3.5). 두 한계가 적용된다. §1은 증류를 “throughout this process”에 사용했다고 묘사하지만 §3.5는 이를 “as the final stage”에 둔다. 또한 그 효과를 분리한 수치는 없다. 결과(단일 연구), 절제 실험 없음.

## §5 Qwen3와 Kimi

**5.1 Qwen3 cold start(S4).** 질의 필터링은 Qwen2.5-72B-Instruct를 사용한다. “not easily verifiable”(여러 하위 질문, 일반 텍스트 생성)한 질의와 “can answer correctly without using CoT reasoning”인 질의를 제거하고, 균형을 위해 도메인을 주석한다([[qwen-3-post-training]] §4.1). QwQ-32B는 질의별 후보 N개를 생성하지만 N은 인쇄되지 않았다. 오답 최종 답, 상당한 반복, 추측, thinking–summary 불일치, 언어 혼합 또는 스타일 전환, 검증 항목과의 의심스러운 유사성 때문에 응답을 제거한다. 저자들은 이후 RL이 제한되지 않도록 “to minimize both the number of training samples and the training steps”를 선호한다. 이후 추론 RL은 query–verifier 쌍 3,995개를 사용하며 Qwen3-235B-A22B는 170단계에 걸쳐 AIME'24 70.1에서 85.1로 움직인다(§4.2).

**5.2 thinking 모드 융합(S6).** Stage 3은 RL 모델에 대한 계속 SFT다. “The 'thinking' data is generated via rejection sampling on Stage 1 queries using the Stage 2 model itself”이므로, 추가 SFT가 추론 능력을 저하하지 않게 한다. non-thinking 데이터는 코딩, 수학, 지시 따르기, 다국어 과제, 창작 글쓰기, QA, role-play를 포괄한다(§4.3). Table 22(Qwen3-32B, thinking 모드)는 Stages 3·4가 더하고 제거하는 것을 보인다. IFEval strict prompt는 73.0 → 78.4 → 85.0, ToolUse(in-house)는 63.3 → 70.4 → 85.5, AIME'24는 83.8 → 81.9 → 81.4, LiveCodeBench v5는 68.4 → 67.2 → 65.7이다. 저자들은 “choose to accept this performance trade-off to enhance the model's overall versatility”라고 말한다(§4.7).

**5.3 strong-to-weak 증류(S7).** Qwen3-0.6B에서 14B, Qwen3-30B-A3B는 두 단계로 학습한다. /think 및 /no_think 모드 둘 다로 “generated”한 교사 출력에 대한 오프폴리시 증류 뒤, “the student model produces responses in either /think or /no_think mode”인 온폴리시 증류를 한다. 학생은 KL divergence를 최소화하도록 교사 모델(Qwen3-32B 또는 Qwen3-235B-A22B)의 logit과 자신의 logit을 정렬하여 “fine-tuned”된다(§4.5). §4.5는 이 단계의 보상 모델을 기술하지 않는다. 학생의 교사는 RL을 거쳤지만, 학생 자체는 이 단계 전이나 도중에 RL을 받지 않는다.

Table 21의 계산 예(Qwen3-8B, 수학·코드 질의, AIME'24 55.0인 같은 오프폴리시 체크포인트)다. pass@k는 샘플링한 k개 응답 중 하나 이상이 정답일 확률이다. 표는 pass@1을 괄호 안 pass@64와 함께 인쇄한다.

| 방법 | AIME'24(pass@64) | GPQA-Diamond | GPU 시간 | GPU 시간 1,000당 AIME'24 이득 |
|---|---|---|---|---|
| + RL | 67.6 (90.0) | 61.3 | 17,920 | 12.6 / 17.92 = 0.70 (계산값) |
| + 온폴리시 증류 | 74.4 (93.3) | 63.3 | 1,800 | 19.4 / 1.80 = 10.8 (계산값) |

GPU 시간 비율은 17,920 / 1,800 = 9.96이다. RL은 pass@64를 90.0에 두었지만 온폴리시 증류는 93.3으로 올렸다. 저자들은 이를 증류가 “exploration space”를 확장한 것으로 읽는다(해석). 조건: 학생 크기 하나, 수학·코드 질의만, 반복 실행 미보고, 교사 추론 비용 분리 미보고다.

**5.4 Kimi k1.5 warm-up 및 long2short(S4, S8).** long-CoT warm-up은 prompt engineering으로 만든 검증된 long-CoT 경로의 “a small yet high-quality” 데이터셋이며, 크기는 인쇄되지 않았다([[kimi-k1-5-recipe]], §2.2). RL 프롬프트는 SFT 모델의 10개 샘플 pass rate로 큐레이션하고, multiple-choice, true/false, proof 질문을 제외하며, CoT 없는 모델이 N = 8회 시도 안에 정답을 내는 프롬프트를 제거한다(§2.1). short-CoT 모델에는 네 long2short 방법을 비교한다. 가중치 평균 merge, n = 8 샘플 중 가장 짧은 정답 응답을 쓰는 SFT, 가장 짧은 정답 응답을 chosen으로 하고 더 긴 응답을 rejected로 하는 DPO(더 긴 오답, 그리고 chosen보다 1.5× 긴 정답 응답), 길이 패널티 및 줄인 최대 rollout 길이의 long2short RL이다(§2.4). long2short RL은 응답당 3,272토큰에서 AIME 2024 60.8에 도달했고 Figure 7에서 최고 토큰 효율을 보였다(§3.4).

**5.5 Kimi K2(S5).** SFT 후보 응답은 “K1.5 and other in-house domain-specialized expert models”에서 오며 LLM 또는 사람 판정기가 필터링한다. 데이터셋 크기, epoch, 학습률은 인쇄되지 않았다([[kimi-k2]] §3.1; [[kimi-k2-recipe]]).

## §6 NVIDIA, Microsoft, Mistral, Google, Meta, Xiaomi, 그리고 한 실무자 연구

**6.1 Llama-Nemotron(S2, S7).** 아키텍처 탐색 뒤 LN-Super는 40B토큰, LN-Ultra는 65B토큰 증류 뒤 88B토큰 계속 사전학습을 한다([[llama-nemotron]] §2.2). SFT 데이터는 도메인·모드별로 할당한다. 수학에서는 DeepSeek-R1이 추론 해법(문제당 16개)을, Qwen2.5-Math-7B-Instruct가 비추론 해법(문제당 64개)을 작성하고 Qwen2.5-32B-Instruct가 답 동치를 판정한다. 코드에서는 R1의 온도 0.6, top-p 0.95 샘플이 약 488K Python 샘플을 만들고, 코드 데이터를 25k에서 736k로 늘리면 “showed continuous improvement”했다. R1의 일반 응답은 70B 보상 모델로 거부 샘플링한다(§3.1). 비추론 쌍은 “detailed thinking off” 아래 Llama-3.1-Nemotron-70B-Instruct 또는 Llama-3.3-70B-Instruct에서 온다(§3.2). 혼합은 총 33,011,757 샘플, 그중 수학 66.8%다(Table 2).

Table 5(GPQA-Diamond, reasoning on)의 SFT 대 RL은 LN-Ultra-SFT 66.4, DeepSeek-R1 71.5, GRPO 뒤 LN-Ultra 76.0이다. 저자들은 “distillation inherently sets an upper bound on the student's performance”이며 RL은 “yields suboptimal results for smaller models compared to distillation”이라고 써서 LN-Ultra에만 RL을 적용한다(§5). 또한 최고 점수 SFT 체크포인트가 아니라 “initialized RL from an earlier checkpoint”했다(§7.4). 결과(단일 연구)다.

**6.2 Nemotron Nano 2(S2, S7, S8, S9).** Alignment SFT는 약 80B토큰을 사용하며 수학·과학·코드 응답은 DeepSeek-R1-0528, 도구 호출·대화 응답은 Qwen3-235B-A22B에서 온다([[nemotron-nano-2]] §3.1). Stage 1은 reasoning trace를 제거한 프롬프트 약 10%를 더한다. Stage 2는 Stage 1이 tool calling을 저하시켰으므로 128k packing 없이 tool calling을 재학습한다. Stage 3은 최종 답을 보존하면서 “abruptly truncated to 1–2k tokens”한 trace를 더하며, §1은 잘린 trace를 데이터 약 5%로 둔다. 잘린 예가 없으면 모델은 짧은 thinking budget에서 답을 길게 만들고 잘 형성된 응답도 더 적게 만들었다(Figure 5, 그림만). 9B 압축은 forward-KL logit 증류를 단계별로 한다(~60B, ~50B, ~25B, ~1B토큰). GRPO 뒤 추가 약 0.4B토큰 증류는 MMLU-Pro를 포함한 “to recover post-RL drops” 목적이다(§4.3, Figure 6, 그림만). 약 6B KD토큰 뒤 절제에서 reasoning-SFT 데이터 70%와 사전학습 데이터 30% 혼합은 수학 평균 최고 58.5였다. 50/50은 57.5, 90/10은 57.2다(Table 11).

**6.3 Phi-4-reasoning(S7).** SFT 집합은 prompt–response 쌍 1.4M 이상, 고유 토큰 8.3B이며, “situated at the edge of Phi-4's current abilities”인 “teachable” 프롬프트에서 o3-mini 응답을 쓴다. 검증 가능한 답이 없으면 난이도는 약한 모델(Phi-4 또는 GPT-4o)이 강한 기준 모델 plurality 답과 일치하는 비율이다([[phi-4-reasoning]] §2.1, §3). 교사 선택에서 “o3-mini with medium 'reasoning effort' [had] similar effect to DeepSeek-R1 when used as teachers, but o3-mini medium was more token efficient”였고 high effort는 더 강하나 더 긴 응답을 냈다(§3.2). Phi-4-reasoning은 SFT만이고, Phi-4-reasoning-plus는 약 6k 문제에 걸쳐 90단계 GRPO(단계당 64, 각 8샘플)를 더한다. AIME에서 10% 이상 얻지만 토큰은 약 1.5× 더 쓴다(§1, §4.2). 카드 [[phi-4]]는 Phi-4와 Phi-4-reasoning을 섞으므로 수치에 사용하지 않는다.

**6.4 Magistral(S4).** Magistral Medium은 추론 SFT를 쓰지 않는다. Magistral Small(24B)은 Magistral Medium RL 실행의 정답 trace와 OpenThoughts/OpenR1-code 프롬프트에 대한 Medium 응답, 일반 지시 데이터 10%로 4 epoch SFT한 뒤 RL을 한다([[magistral-recipe]], §5.3). Table 3(24B; SFT / RL 전용 / SFT + RL)은 AIME'24 pass@1 65.4 / 65.8 / 70.7, AIME'25 55.6 / 51.9 / 62.8, GPQA 63.4 / 68.8 / 68.2, LiveCodeBench v5 52.2 / 46.4 / 55.8, AIME'24 maj@64 90.0 / 86.7 / 83.3이다([[magistral]]). maj@64는 64샘플의 다수 답 정확도다. RL 전용 열은 AIME'24 pass@1·GPQA에서 SFT와 같거나 높고 AIME'25·LiveCodeBench에서는 낮다. 저자들은 이것이 RL 의존 소형 모델이 증류 모델보다 낮다는 DeepSeek-R1 관찰과 모순된다고 말한다(§6.2). 결과(단일 연구)다. 별도로 Mistral Medium 3은 오픈 소스 R1 생성 trace 약 1.3M개로 미세조정됐고, 그 체크포인트 위 RL은 SFT 체크포인트 대비 AIME'25에서 10점 초과, LiveCodeBench에서 5점을 얻었으나 GPQA-Diamond는 SFT 뒤 72.9%에서 RL 뒤 71.0%로 떨어졌다(§8, Figure 13).

**6.5 Gemma(S1).** Gemma 2는 사전학습 증류 효과를 측정했다. 500B토큰의 2B 모델은 scratch에서 세 벤치마크 평균 60.3, 7B 교사 증류에서 67.7이다([[gemma-2]] Table 6). Gemma 3는 토큰당 logit 256개를 샘플링하고 27B/12B/4B/1B 모델에 각각 14T/12T/4T/2T토큰으로 학습한다([[gemma-3]] §2.2). 교사 크기 연구에서 “for short training horizons, the smaller teacher is better, but the trend is reversed for longer training”이다(§5.4, Figure 8; 값은 그림에만 있음). Gemma 3 사후학습도 큰 instruction-tuned 교사의 “an improved version of knowledge distillation”을 쓰지만 설정은 인쇄되지 않았다(§3).

**6.6 Llama 4(S1).** Llama 4 Maverick은 사전학습에서 Behemoth로부터 codistillation했으며, 손실은 학습 동안 “dynamically weights the soft and hard targets”한다. 교사 forward pass는 학생 데이터 대부분에 재사용하며 새 데이터에만 다시 실행한다([[llama-4]], “Pushing Llama to new sizes”). 가중치 스케줄이나 절제 수치는 없다.

**6.7 Minitron(S2).** 교사는 교정하지 않으면 “sub-optimal guidance”를 주므로 먼저 증류 데이터셋에서 약 100B토큰 미세조정한다(“teacher correction”). 이는 “over a 6% reduction in LM validation loss”를 주었다([[minitron-approach]] Insights). 이후 MN-Minitron-8B는 380B토큰에서 logit 전용 forward KL로 학습했으며 scratch 학습보다 “up to 40×” 적다.

**6.8 MiMo-V2-Flash(S5, S9).** SFT 응답은 thinking 및 non-thinking 모드의 “in-house domain-specialized model checkpoints”에서 온다([[mimo-v2-flash]] §4.2). 이후 도메인 교사는 RL로 학습하고(Table 7은 일부 행의 최선 교사로 SFT 모델과 학생 자체도 나열), 다중 교사 온폴리시 증류(MOPD)는 §1.1 토큰별 이점에 α배 결과 보상 이점을 더해 학생 자신의 샘플로 학습한다(§4.4; α 미보고). Table 7은 MOPD 전 학생 / 최선 교사 / MOPD 후 학생을 비교한다. AIME 2025는 89.3 / 93.9 / 94.1, SWE-Bench Verified 67.8 / 74.2 / 73.4, Arena-Hard 창작 글쓰기는 90.1 / 90.1 / 86.2, BrowseComp는 42.5 / 51.7 / 45.4다. 학생은 12행 중 8행에서 최선 교사에 도달하거나 넘었고, GPQA-Diamond 0.6, SWE-Bench Verified 0.8, 창작 글쓰기 3.9, BrowseComp 6.3점 낮았다(Table 7에서 계산).

**6.9 Thinking Machines(S7, S9; 실무자 근거).** OpenThoughts-3 프롬프트 400k개 SFT 뒤 Qwen3-8B-Base는 AIME'24 60%다. 온폴리시 증류는 약 150단계(프롬프트 약 77K, 프롬프트당 4샘플)에 70%에 도달하며, 글은 SFT를 약 2M 프롬프트까지 연장하는 것보다 9–30× 적은 계산이라고 추정한다([[thinkingmachines-on-policy-distillation]]). 개인화 시험에서 internal 문서 70%, chat 30%로 중간학습한 Qwen3-8B는 IF-eval 85%에서 79%로 움직였다. Tulu 3 프롬프트에서 원래 Qwen3-8B를 교사로 한 온폴리시 증류는 83%로 복구했고, internal-QA 정확도는 36%에서 41%로 올랐다.

## §7 공개 행렬과 정오표

N = 수치로 제시된 값, D = 수치 없이 서술, — = 미보고(§3–§6의 근거 위치에서 확인).

| 보고서 | 단계 배치 | 프롬프트 원천과 선택 | 교사와 샘플링 | 출력 필터 | 목적함수 설정 | 측정한 효과 |
|---|---|---|---|---|---|---|
| DeepSeek-R1 | N | N(RL, Table 4); D(SFT) | N(cold start: 온도 1.0의 R1-Zero); D(800K 집합) | D | N | N(Tables 3, 16) |
| DeepSeek-V3 | N | — | D | D | N | N(Table 9, V2.5에서) |
| DeepSeek-V3.2 | D | — | D | — | — | D |
| GLM-4.5 | N | N(길이 절단) | D | D | —(SFT LR, epoch) | N(벤치마크 이름 없음) |
| GLM-5 | D(§1 대 §3.5) | D | D | D | N | — |
| Qwen3 | N | D | D(N 미인쇄) | D | D | N(Tables 21, 22) |
| Kimi k1.5 | N | N(N = 8, 10 샘플) | D | D | N(long2short) | N(3,272에서 60.8) |
| Kimi K2 | D | — | D | D | — | — |
| Llama-Nemotron | N | N | N | N | N | N(Tables 3–5) |
| Nemotron Nano 2 | N | D | D | D | N(KD) | N(Table 11); 그림 |
| Phi-4-reasoning | N | D(임계값 미인쇄); N(decontamination 목록) | D | D | N | N |
| Magistral Small | N | D | D | D | N(4 epoch) | N(Table 3) |
| Gemma 3 | N | — | —(교사 미공개) | — | N(256 logit) | D(그림만) |
| Llama 4 | D | — | D | — | — | — |
| MiMo-V2-Flash | N | — | D | D | D(α, ε 미인쇄) | N(Table 7) |

> **이전 장 정오표(줄 번호는 git 커밋 `4a72e54` 버전을 가리킴).**
> 1. ch-20 L107 “the 6 distilled students (Qwen2.5-Math 1.5B/7B/14B/32B, Llama-3.1-8B, Llama-3.3-70B)” → 14B와 32B 학생은 Qwen2.5-14B 및 Qwen2.5-32B이고, 70B 학생은 Llama-3.3-70B-Instruct다. 같은 줄의 “SFT-on-R1-traces for one epoch” → 2–3 epoch다([[deepseek-r1-recipe]], arXiv:2501.12948v2 B.4.3 Table 6).
> 2. ch-20 L98–105 slice 표(Math 약 200–300K, Code 약 200–300K, Logic/science 약 50–100K) → Table 5는 Math 395,285, Code 211,129, STEM 10,124, Logic 10,395, General 177,812를 준다([[deepseek-r1]] B.3.3).
> 3. ch-20 L95 “a V3 judge filters them for readability + correctness” → 규칙 필터는 mixed-language, long-paragraph, code-block CoT를 제거하며, DeepSeek-V3는 추가 데이터 일부의 정확도만 판정한다([[deepseek-r1]] B.3.3, Listing 4).
> 4. ch-20 L10 “Distillation in 2023–2025 is not logit-matching” → Gemma 2와 Gemma 3는 사전학습 logit 증류, Minitron과 Nemotron Nano 2는 forward-KL logit 증류, Qwen3는 학생 logit을 교사 logit에 정렬한다([[gemma-2]] §3.2; [[gemma-3]] §2.2; [[minitron-approach]]; [[qwen-3-post-training]] §4.5).
> 5. ch-32 L69, L78, L199 “~800K cold-start examples … The attested point is DeepSeek's 800K … R1 used 800K cold-start traces before RL” → R1 cold start는 “thousands” 예이고, 804,745샘플 집합은 뒤의 거부 샘플링 SFT 단계다([[deepseek-r1]] §3, B.3.2, B.3.3).
> 6. ch-34 L130 “on-policy (student rollouts scored by teacher-derived RM)” → 학생 logit은 KL 최소화로 Qwen3-32B 또는 Qwen3-235B-A22B logit과 정렬된다. 이 기술은 보상 모델을 언급하지 않는다([[qwen-3-post-training]] §4.5).
> 7. ch-34 L174 “Then GRPO for 90 training steps with … +1 correct / −0.5 incorrect” → Phi-4-reasoning은 SFT 전용이다. 90단계 GRPO는 Phi-4-reasoning-plus를 만들었고, 정확도 보상은 길이로 스케일되어 정답 [0.5, 1.0], 오답 [−1.0, −0.5]에 있다([[phi-4-reasoning]] §1, §4.1).
>
> ch-34는 이후 2026-09에 수정되었고 현재 텍스트에는 주장 6, 7이 더 이상 없다. 이들은 이전 버전 독자를 위해 나열한다.

§1.1의 온폴리시 목적함수는 별도 `on-policy-distillation` 과정(branch `course/on-policy-distillation`)의 ch-04(토큰별 reverse-KL 메커니즘), ch-05(비용과 실패 양상)에서 단계별로 유도한다.

## 부정 샘플과 부정 피드백

**부정이 오는 곳.** 오프폴리시 증류의 부정은 검사를 통과하지 못한 교사 샘플이다. 기준 답 또는 DeepSeek-V3 판정에 대한 오답([[deepseek-r1]] B.3.3), Qwen2.5-32B-Instruct 동치 판정([[llama-nemotron]] §3.1.1), Qwen3의 여섯 응답 기준([[qwen-3-post-training]] §4.1), GLM-4.5의 네 필터([[glm-4-5]] §3.1)가 있다. 이 보고서들은 판정기의 거부율 또는 거짓 음성률을 주지 않는다. 온폴리시 증류에서는 π_T < π_θ인 샘플 토큰마다 음의 이점을 받는다. 교사 데이터를 쓰는 선호 단계에서 rejected 응답은 더 약하거나 더 긴 원천에서 온다. Kimi k1.5 long2short DPO의 더 긴 오답과 chosen보다 1.5× 긴 정답([[kimi-k1-5]] §2.4), SmolLM3의 Qwen3-32B 응답에 대한 Qwen3-0.6B 응답([[smollm-3-midtraining]]), Nano 2 온폴리시 DPO의 실패 WorkBench 도구 호출([[nemotron-nano-2]] §3.2)이다.

**현재 관행이 이들을 처리하는 방식.** 거부 샘플링 증류는 이를 버리며, 이는 의미 (1) 음의 한계가치다. GLM-5는 오류 에이전트 세그먼트를 컨텍스트에는 남기지만 손실에서 마스킹하여 모델이 뒤따르는 수정에 학습하게 한다. 이는 오류 토큰 그래디언트 없는 의미 (2) 내용으로서 부정이다([[glm-5]] §3.1). DPO 쌍, 음의 온폴리시 증류 이점, Phi-4-reasoning-plus 음의 보상은 의미 (4) 그래디언트로서 부정이다. 이 장의 어떤 보고서도 의미 (3) 실패 조건화를 쓰지 않는다. 전체 유도는 ch-31a, ch-43a에 있다.

**메커니즘.** 로짓 z softmax에서 ∂ log p_y / ∂ z_j = 1[j = y] − p_j다. 샘플 토큰 y의 이점 Â에서 gradient-ascent 단계는 z_j를 η·Â·(1[j = y] − p_j)만큼 바꾼다. §1.2 학생 p_θ = (0.4, 0.4, 0.1, 0.1), 샘플 토큰 4, Â = −2.303, η = 0.1을 쓰면 Δz = 0.1 × (−2.303) × (−0.4, −0.4, −0.1, 0.9) = (+0.092, +0.092, +0.023, −0.207)이다. 새 확률은 (0.4135, 0.4135, 0.0965, 0.0766)이다. 토큰 4는 0.0234를 잃고, 토큰 1·2는 각각 0.0135를 얻으며, 토큰 3은 logit 증가에도 0.0035를 잃는다. 제거 질량은 이미 가장 가능성 높은 토큰, 여기서는 동률인 1·2로 가며 교사가 토큰 2(0.30)를 학생 현재 0.40보다 낮게 평가해도 그렇다. 토큰 2가 샘플되면 조밀한 토큰별 신호가 이를 교정한다(Â = −0.2877). 이미 낮은 확률 토큰에 단독 push-down을 걸면 교사 선호와 무관하게 가장 가능성 높은 대안에 질량이 집중한다.

**수치를 가진 근거.** 음의 이점을 포함한 온폴리시 증류는 같은 체크포인트의 RL 67.6에 대해 AIME'24 74.4에 도달했다([[qwen-3-post-training]] Table 21). MiMo-V2-Flash는 outcome reward RL, 그것 없는 MOPD, 그것 있는 MOPD를 곡선으로만 비교한다(Figure 6). Kimi k1.5는 최선 응답만 학습하는 ReST에 대한 이점을 음의 그래디언트에 돌리지만 몫을 분리하지 않는다(§3.5, Figure 10). 이 장의 어떤 소스도 증류 설정에서 개선 중 부정이 차지하는 몫을 측정하지 않는다. ch-43a는 이를 측정한 RL 설정을 다루지만, RL에서 잰 몫은 새 측정 없이는 증류에 전이하지 않는다.

**제어.** 국소화: GLM-5는 전체 궤적을 패널티하지 않고 오류 세그먼트를 마스킹한다. 경계: MiMo-V2-Flash는 train–inference 비율이 [ε_low, ε_high] 밖인 토큰의 중요도 가중치를 0으로 한다(Eq. 8). 온폴리시 유지: Nano 2는 후보 체크포인트마다 DPO 쌍을 재생성하며 “this process ensures that iterative DPO remains on-policy”다(§3.2). 앵커: 교사 이점에 outcome reward를 결합하거나(MiMo Eq. 9), 음의 업데이트 전 양성 SFT를 한다(전체 파이프라인을 설명한 §1.3의 모든 온폴리시 보고서). 교사에 의한 국소화: SimpleBench 예에서 [[thinkingmachines-on-policy-distillation]]은 교사의 가장 큰 패널티가 “that start phrases which lead the student astray” 토큰에 떨어지고, “the final answer, though wrong, isn't penalized — it is entirely predictable conditional on the whole preceding sequence”라고 관찰한다. 이는 측정된 비율이 아닌 실무자 글의 삽화 하나다.

**진단.** Â > 0 및 Â < 0 토큰의 평균 토큰별 reverse KL을 따로 기록하고, 학생 엔트로피, pass@1 및 k = 64의 pass@k([[qwen-3-post-training]] Table 21), maj@64([[magistral]] Table 3), 응답 길이와 truncation rate([[deepseek-v3]] Table 9), 더 약한 모델에서 만든 선호 쌍의 chosen/rejected 로그 확률을 기록한다.

**일반성 효과.** 실패를 버리면 교사가 푸는 프롬프트로 커버리지가 좁아진다(해석). Qwen3는 QwQ-32B가 지속적으로 실패한 질의를 사람 주석자에게 보내고, Llama-Nemotron은 ground-truth 답이 없으면 다수결을 쓴다([[qwen-3-post-training]] §4.1; [[llama-nemotron]] §3.1.3). 온폴리시 증류는 AIME'24 pass@64를 90.0에서 93.3으로 올렸으나 RL은 바꾸지 않았다([[qwen-3-post-training]] Table 21). Magistral Table 3에서 SFT + RL은 최고 AIME'24 pass@1이지만 최저 maj@64(83.3, SFT 90.0)였다. Kimi k1.5 DPO는 1.5× 긴 정답을 rejected한다(§2.4). 그 쌍에서 선호 신호는 정확도가 아니라 길이다(해석).

## 레시피

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 소스 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1(Dev1) | 671B MoE | SFT(cold start) | 예; epoch; LR; context; batch | “thousands”; 2–3; cosine 5×10⁻⁵ → 5×10⁻⁶; 32,768; 128 | arXiv:2501.12948v2 B.3.2, B.4.2; [[deepseek-r1-recipe]] | 2026-09-15 검증 | Table 3: Dev1 대 R1-Zero |
| DeepSeek-R1(Dev3) | 671B MoE | 증류-SFT(post-RL) | 도메인별 샘플 | Math 395,285; Code 211,129; STEM 10,124; Logic 10,395; General 177,812; 전체 804,745 | v2 B.3.3 Table 5 | 2026-09-15 검증 | Table 3: Dev2 → Dev3 AlpacaEval 2.0 55.8 → 62.1 |
| DeepSeek-R1-Distill(여섯 모두) | 1.5B–70B | 증류-SFT | epoch; schedule; context; batch | 2–3; 초기 LR 1/10까지 cosine; 32,768; 64 | v2 B.4.3 | 2026-09-15 검증 | 절제 미보고 |
| DeepSeek-R1-Distill-Qwen-7B | 7B | 증류-SFT | base; 초기 LR | Qwen2.5-Math-7B; 8×10⁻⁵ | v2 Table 6 | 2026-09-15 검증 | 절제 미보고 |
| DeepSeek-R1-Distill-Qwen-32B | 32B | 증류-SFT | base; 초기 LR | Qwen2.5-32B; 6×10⁻⁵ | v2 Table 6 | 2026-09-15 검증 | Table 16: base RL 47.0 대비 AIME 2024 72.6 |
| DeepSeek-V3 | 671B / 37B act. | 증류-SFT | 인스턴스; epoch; LR; packing | 1.5M; 2; cosine 5×10⁻⁶ → 1×10⁻⁶; sample masking packing | arXiv:2412.19437v2 §5.1; [[deepseek-v3-recipe]] | 2026-09-15 검증 | Table 9(V2.5): MATH-500 74.6 → 83.2 |
| DeepSeek-V3.2 | 미인쇄 | 증류-SFT | 크기, epoch, LR | 미인쇄 | arXiv:2512.02556v1 §3 | 미보고(본문 확인) | §3: 증류 모델 “only marginally below” 전문가(수치 없음) |
| DeepSeek-V4-Flash / -Pro | 둘 다 | 증류(온폴리시) | 교사; 손실 | 열 개 초과; Σᵢ wᵢ·D_KL(π_θ ‖ π_Eᵢ), 전체 어휘; wᵢ, 단계, LR 미인쇄 | V4 report §5.1.2 Eq. 29; [[deepseek-v4-recipe]] | 2026-09-14 검증(카드) | 표 없음 |
| GLM-4.5 | 355B / 32B act. | 중간학습 | 합성 추론 단계 | 32K의 500B토큰(Figure 3 표기) | arXiv:2508.06471v1 §2.3, Fig. 3 | 2026-09-15 검증 | 절제 미보고 |
| GLM-4.5 | 355B / 32B act. | 증류-SFT | 데이터; context; prompt 절단; 어려운 프롬프트당 응답 | 전문가 3개에서 “millions”; 128K; 길이 하위 50% 버림; 4 | v1 §3.1; [[glm-4-5-recipe]] | 2026-09-15 검증(LR, epoch 미인쇄) | §3.1: +2%–4%, +1%–2%(벤치마크 이름 없음) |
| GLM-5 | 744B / 40B act. | 증류(온폴리시) | 교사; 그룹 크기; 배치 | 이전 단계 최종 체크포인트; 1; 1,024 | arXiv:2602.15763v2 §3.5; [[glm-5]] | 2026-09-15 검증(비율, 단계, LR 미인쇄) | 절제 미보고 |
| Qwen3-235B-A22B | 235B / 22B act. | RL(cold start 뒤) | query–verifier 쌍; 단계 | 3,995; 170 | arXiv:2505.09388v1 §4.2; [[qwen-3-post-training]] | 2026-09-15 검증 | AIME'24 70.1 → 85.1 |
| Qwen3 cold start | flagship | SFT(cold start) | 교사; N; 집합 크기 | QwQ-32B; N 미인쇄; 크기 미인쇄 | v1 §4.1 | 미보고(N, 크기) | 절제 미보고 |
| Qwen3-8B | 8B | 증류(온폴리시) | 교사; GPU 시간; 질의 도메인 | Qwen3-32B 또는 Qwen3-235B-A22B(8B용은 특정 안 됨); 1,800; 수학·코드 | v1 §4.5, §4.7 Table 21 | 2026-09-15 검증 | Table 21: GPU 17,920시간 RL 67.6 대비 74.4 |
| Kimi k1.5 | 미보고 | 증류-SFT(warm-up) | 크기, 설정 | “small yet high-quality”; 미인쇄 | arXiv:2501.12599v4 §2.2; [[kimi-k1-5-recipe]] | 미보고 | n/a |
| Kimi k1.5 short-CoT | 미보고 | 증류-SFT / 선호 | 최단 거부 샘플링; DPO rejected | n = 8; 더 긴 오답과 chosen보다 1.5× 긴 정답 | v4 §2.4 | 2026-09-14 검증(카드) | Figure 7; long2short RL 3,272토큰에서 60.8(§3.4) |
| Kimi-K2-Instruct | 1.04T / 32B act. | 증류-SFT | 원천; 크기 | K1.5 및 도메인 전문가, 판정기 필터; 크기 미인쇄 | arXiv:2507.20534v2 §3.1; [[kimi-k2-recipe]] | 2026-09-14 검증(카드) | 절제 미보고 |
| LN-Ultra | 253B | pruning KD; CPT | 토큰 | KD 65B; CPT 88B | arXiv:2505.00949v5 §2.2; [[llama-nemotron]] | 2026-09-15 검증 | Table 1: Llama-3.1-405B-Instruct 88.6 대비 MMLU 88.1 |
| Llama-Nemotron SFT 혼합 | 8B–253B | 증류-SFT | 총 샘플; 문제당 수학 생성; 판정기 | 33,011,757; R1 16, Qwen2.5-Math-7B-Instruct 64; Qwen2.5-32B-Instruct | v5 Table 2, §3.1.1 | 2026-09-15 검증 | §3.1.2: 코드 25k → 736k “showed continuous improvement” |
| LN-Nano | 8B | 증류-SFT | batch; packing; stage-1 LR, epoch | 256; 32k; 1e-4, 4 epoch(추론만) | v5 §4.2 | 2026-09-15 검증 | §4.2: stage 1 “prevents failure modes such as repetitive completions”(수치 없음) |
| LN-Super | 49B | 증류-SFT | epoch; LR; 길이; batch | 1; 고정 5e-6; 16k; 256 | v5 §4.2 | 2026-09-15 검증 | §4.2: 소형 실행은 5e-5에서 3–4 epoch까지 개선(전체 규모에서 미실행) |
| LN-Ultra | 253B | 증류-SFT | packing; batch; LR | 24k; 256; 1e-5까지 warmup, 1e-6까지 cosine, warmup 비율 10% | v5 §4.2 | 2026-09-15 검증 | §4.2: 높은 LR 5e-5가 더 좋으나 불안정 |
| Nemotron-Nano-12B-v2 | 12B | 증류-SFT | 토큰; 잘린 trace; 제거 trace | 약 80B; 데이터 약 5%를 1–2k로 절단; 프롬프트 약 10% trace 없음(Stage 1) | arXiv:2508.14444 §1, §3.1–3.2; [[nemotron-nano-2]] | 2026-09-15 검증(LR, epoch 미인쇄) | Figure 5(그림만) |
| Nemotron-Nano-9B-v2 | 9B | pruning KD | 손실; 토큰; 데이터 혼합 | forward KL; 약 60B + 약 50B + 약 25B + 약 1B; GRPO 뒤 약 0.4B; 추론-SFT 70% / 사전학습 30%(약 6B KD토큰 뒤 절제) | §4.3, Table 11 | 2026-09-15 검증 | Table 11: 수학 평균 58.5 대 57.5(50/50), 57.2(90/10) |
| MN-Minitron-8B | 8B | pruning KD | 교사 교정; LR; batch; 토큰 | 약 100B; 1e-4 → 4.5e-7, warm-up 60; 768; 380B | arXiv:2408.11796v4 Table 4; [[minitron-approach]] | 2026-09-15 검증 | Insights: 교사 교정이 LM 검증 손실 6% 초과 하락 |
| Phi-4-reasoning | 14B | 증류-SFT | 쌍; 토큰; 단계; batch; context; LR | 1.4M 초과; 고유 8.3B(학습 16B); 약 16K; 32; 32K; 1e-5, warm-up 450, wd 1e-4 | arXiv:2504.21318v1 §3, §3.2; [[phi-4-reasoning]] | 2026-09-15 검증 | §3.1: LR grid [1e-6, 2e-5], 1e-5 최선 |
| Phi-4-reasoning-plus | 14B | RL | 단계; 문제; G; LR; β | 90; 약 6k; 8; 5e-8; 0.001 | v1 §4.2 | 2026-09-15 검증 | Fig. 7a: AIME 10% 초과; 이후 단계 이득 없음 |
| Magistral Small | 24B | 증류-SFT | epoch; 일반 데이터; 체크포인트 규칙 | 4; 10%; 최고 AIME'24 | arXiv:2506.10910v1 §5.3; [[magistral-recipe]] | 2026-09-14 검증(카드; 크기 미보고) | Table 3: SFT + RL 70.7 대 SFT 65.4 |
| Gemma 3 | 1B / 4B / 12B / 27B | pretrain-stable(KD) | 토큰당 logit; 토큰 | 샘플링·재정규화 256; 2T / 4T / 12T / 14T | arXiv:2503.19786v1 §2.2; [[gemma-3]] | 2026-09-15 검증(교사 미보고) | §5.4 Fig. 8(그림만) |
| Gemma 2 2B(절제) | 2B | pretrain-stable(KD) | 교사; 토큰 | 7B; 500B | [[gemma-2]] Table 6 | 2026-09-14 검증(카드) | scratch 60.3 대비 67.7 |
| MiMo-V2-Flash | 309B / 15B act. | 증류-SFT | LR; batch; AdamW ε; expert-bias rate | cosine 5.0e-5 → 5.0e-6; 128; 1.0e-8; 1.0e-4 | arXiv:2601.02780v2 §4.2; [[mimo-v2-flash]] | 2026-09-15 검증 | §4.2: num-zeros 안정성(표 없음) |
| Qwen3-8B-Base(Thinking Machines) | 8B | 증류(온폴리시) | 교사; 단계; 프롬프트; 프롬프트당 샘플 | Qwen3-8B; 약 150; 약 77K; 4 | blog 2025-10-27, “Distillation for reasoning”; [[thinkingmachines-on-policy-distillation]] | 2026-09-15 검증(LR 미보고) | AIME'24 60% → 70%; 실무자 근거 |
| SmolLM3-3B | 3B | 중간학습 | 토큰; epoch; 원천 | 35B; 4(약 140B 관측); OpenThoughts3-1.2M + Llama-Nemotron subset | HF blog 2025-07-08, “Reasoning Mid-training”; [[smollm-3-midtraining]] | 2026-09-15 검증 | “Model Merging”: RULER 손실이 이 단계로 추적됨 |

**작은 범용 실행의 시작점.** 더 강한 교사를 가진 8B–14B 학생에서는 위 검증 행이, 명시한 조건 아래 다음을 뒷받침한다. 첫째, 한 모델이 두 모드를 맡으면 모드 지시 아래 reasoning-on·reasoning-off 응답을 섞은 필터링 교사 샘플 SFT(Llama-Nemotron, Qwen3처럼)다. R1-Distill은 671B 교사의 804,745 샘플에 Qwen2.5-Math-7B base의 초기 LR 8×10⁻⁵의 1/10까지 cosine 감쇠, 2–3 epoch, 32,768토큰, batch 64를 썼다. Phi-4-reasoning은 14B에서 1.4M 쌍 초과에 LR 1e-5, warm-up 450, batch 32, context 32K를 썼다. LN-Nano는 33M샘플 혼합에 8B의 세 SFT 단계 중 첫 단계에서 추론 데이터만 LR 1e-4, 4 epoch, batch 256, 32k packing을 썼다. 세 설정의 학습률은 한 자릿수 규모 차이가 나며, 어떤 소스도 같은 데이터에서 비교하지 않는다. 둘째, SFT 체크포인트에서 온폴리시 증류다. 검증된 소형 모델 설정은 Qwen3-8B의 수학·코드 질의 1,800 GPU시간, Thinking Machines의 약 150단계·프롬프트당 4샘플(8B, 실무자 근거)뿐이며 둘의 학습률은 미보고다. 셋째, 각 단계 뒤 IFEval, tool-use 벤치마크, Arena-Hard, long-context 시험, 날짜 필터링 수학 집합 pass@64에서 체크포인트를 평가한다. 각각은 위 보고서 중 적어도 하나에서 움직였다.

## 일반화 관점

**(a) 무엇이 폭을 넓히는가.**
- 추론 증류에 비추론 교사 데이터 추가: R1 Dev2에서 Dev3으로 AlpacaEval 2.0 55.8 → 62.1, Aider-Polyglot 25.6 → 44.8이며 저자들은 추가 비추론·코드공학 데이터에 돌린다. Dev3는 V3-Base에서 재학습했으므로 통제 절제는 아니다([[deepseek-r1]] §4, Table 3).
- non-thinking 데이터 융합 단계 뒤 일반 RL: Qwen3-32B IFEval 73.0 → 85.0, ToolUse 63.3 → 85.5([[qwen-3-post-training]] Table 22).
- 오프폴리시 SFT 위 온폴리시 증류: RL이 90.0에 머문 AIME'24 pass@64를 90.0 → 93.3으로 올림([[qwen-3-post-training]] Table 21).
- teachable-prompt 증류: Phi-4-reasoning 및 -plus는 대상이 아니었던 TSP, 3SAT, BA-Calendar에서 Phi-4 대비 30–60점을 얻었고 “we do not see any catastrophic forgetting compared to the base Phi-4 model on more general capabilities”라고 한다([[phi-4-reasoning]] §1, §3). 결과(단일 연구).
- 능력을 낮춘 단계 뒤 증류: Nano 2는 GRPO 뒤 MMLU-Pro를 복구했고(그림만, [[nemotron-nano-2]] Fig. 6), Thinking Machines는 중간학습 뒤 IF-eval 79%를 83%로 복구했다([[thinkingmachines-on-policy-distillation]]). 두 소스는 서로 다른 저하 단계(RL, 중간학습)에서 방향이 같지만 크기는 비교 불가다.

**(b) 무엇이 좁힘 또는 망각을 일으키는가.**
- 추론 중심 SFT는 IFEval을 낮췄다. LN-Super-SFT 81.9(on)는 Llama-3.3-70B-Instruct 92.1에 비해 낮고 IFEval RL로 89.2를 회복했다([[llama-nemotron]] Table 4). 같은 표는 R1-Distill-Llama-70B 85.1(그 base Llama-3.3-70B-Instruct 92.1), R1-Distill-Llama-8B 73.4(Llama-3.1-8B-Instruct 81.8)를 싣지만 보고서는 이 baseline 평가를 직접 실행했는지 복사했는지 말하지 않는다(Tables 3–4). 결과(단일 연구): 보고서 하나가 R1-Distill·LN-SFT 두 증류 파이프라인에서 방향을 보인다.
- 일반 단계는 특수 추론 비용을 낸다. Qwen3-32B AIME'24 83.8 → 81.4, LiveCodeBench 68.4 → 65.7은 Stages 3–4에서 발생한다([[qwen-3-post-training]] Table 22).
- 추론 중간학습은 긴 컨텍스트 비용을 냈다. SmolLM3 RULER 저하는 그 단계로 추적되었고 merge로 수리됐다([[smollm-3-midtraining]]).
- 온폴리시 증류로 전문가를 merge하면 일부 도메인을 잃었다. 최선 교사 대비 BrowseComp 51.7 → 45.4, 창작 글쓰기 90.1 → 86.2다([[mimo-v2-flash]] Table 7).
- 교사 trace의 SFT + RL 뒤 손실: Magistral Small AIME'24 maj@64은 SFT 90.0, SFT + RL 83.3([[magistral]] Table 3). Mistral Medium 3는 오픈 R1 생성 trace 약 1.3M SFT 뒤 GPQA-Diamond 72.9에서 그 체크포인트 RL 뒤 71.0으로 하락했다(§8, Figure 13).
- 더 긴 출력: R1 유래 데이터의 MATH-500 응답 길이 769 → 1510([[deepseek-v3]] Table 9), Phi-4-reasoning-plus는 약 1.5× 토큰 사용([[phi-4-reasoning]] §1).

**(c) 이 단계의 측정 방법.** 각 체크포인트(교사, SFT 학생, post-RL, post-distillation)를 고정 비대상 패널, 즉 IFEval, Arena-Hard, tool-use(BFCL 또는 Tau2), long-context(RULER), 지식(MMLU-Pro)에서 평가한다. Llama-Nemotron(Tables 3–5), Qwen3(Table 22), MiMo-V2-Flash(Table 7)는 단계 또는 모델별로 일부만 보고하며 어느 표도 long-context 시험을 포함하지 않는다. pass@1 외 k = 64의 pass@k와 maj@64를 보고한다. Phi-4-reasoning의 AIME 2025처럼 데이터 확정 뒤 공개된 경시를 쓴다(§2.2). 분산도 보고한다. average-of-5 AIME 평가 두 실행은 최대 5–10점 달랐다([[phi-4-reasoning]] §1). 다른 연구소의 증류 모델 점수를 나열할 때 자체 설정으로 평가했는지 수치를 복사했는지 기록한다. Llama-Nemotron Tables 3–5는 이를 말하지 않는다.

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| R1의 804,745샘플 집합을 cold start로 취급 | RL 전 “cold start”가 SFT 예산 대부분을 소진 | 근거 위치와 단계 크기 비교: cold start는 “thousands”(B.3.2), 800K는 post-RL(B.3.3) |
| 증류 학생을 교사 목표 벤치마크에서만 평가 | 수학은 오르나 IFEval/tool use 하락을 못 봄 | 매 단계 전후 비대상 패널 실행; base instruct 모델과 비교(LN Tables 3–4) |
| 최고 벤치마크 SFT 체크포인트에서 RL 시작 | 이전 체크포인트보다 RL 이득 작음 | SFT 체크포인트 둘에서 짧은 RL 비교; LN-Ultra는 이 이유로 이전 체크포인트에서 시작했지만 비교는 미인쇄(§7.4) |
| 회귀 검사 없는 순차 RL 단계 | 이후 단계 뒤 이전 도메인 하락 | 각 단계 뒤 이전 도메인 전부 평가; RL/merge 뒤 증류 추가(GLM-5 §3.5; Nano 2 §4.3) |
| 길이 추적 없이 긴 trace 증류 | 정확도 이득과 함께 평균 길이 증가(V3 Table 9의 769 → 1510) | 벤치마크별 평균 길이·truncation rate 기록(V3 Table 9) |
| SFT + RL 뒤 pass@1만 보고 | pass@1 상승, maj@64/pass@64 하락 | pass@64·maj@64 보고(Magistral Table 3; Qwen3 Table 21) |
| mixed-mode SFT를 매우 긴 시퀀스로 packing | tool-calling 정확도 저하 | packing 변경 뒤 BFCL 평가; tool 데이터를 unpacked로 재학습(Nano 2 §3.2) |
| 증류 데이터에서 학습하지 않은 교사의 logit 증류 | 기대보다 높은 학생 검증 손실 | 먼저 증류 데이터에서 교사 미세조정(Minitron teacher correction, LM loss 6% 초과 하락) |
| inference engine 온폴리시 log-prob를 trainer와 확인 없이 계산 | trainer 토큰 확률 π_θ와 sampler 토큰 확률 μ_θ의 비율 π_θ/μ_θ가 일부 토큰에서 1 이탈 | 같은 토큰에서 engine/trainer log-prob 비교; MiMo-V2-Flash처럼 ratio band 밖 토큰 가중치 0(Eq. 8); GLM-5는 inference engine에서 교사 logit을 가져오고 training engine으로 옮길 계획(§3.5) |
| “not reported”를 “not used”로 읽음 | 연구소가 쓴 단계에 대한 잘못된 결론 | §7 행렬·그림처럼 확인한 근거 위치 기록 |

## 이해 점검

1. §1.2 예에서 forward KL과 reverse KL은 학생 오류 순위를 다르게 매긴다. 각 목적함수가 가장 크게 패널티하는 토큰과, 교사가 거의 만들지 않는 토큰을 학생이 과가중할 때 그 차이가 중요한 이유를 설명하라.
2. DeepSeek-R1 800K 집합은 General 샘플 22.1%이나 General 토큰 5.9%다. 토큰별 손실 평균과 샘플별 평균이 General 데이터의 유효 가중을 어떻게 바꾸고 지시 따르기 유지에 무엇을 뜻하는지 설명하라.
3. LN-Ultra-SFT는 GPQA-Diamond 66.4, 교사는 71.5다. 지도 증류가 교사에 의해 경계되는 이유와, 이전 SFT 체크포인트의 RL이 최고 SFT 체크포인트 RL보다 더 높게 끝날 수 있는 이유를 설명하라.
4. Qwen3 Table 21에서 RL은 pass@1을 개선했지만 pass@64는 바꾸지 않았고, 온폴리시 증류는 둘 다 개선했다. 각 결과의 메커니즘과 이를 분리할 추가 실험을 제시하라.
5. GLM-5, MiMo-V2-Flash, DeepSeek-V4는 모두 RL 뒤 증류를 둔다. 증류 단계가 나중 RL에서 잃은 능력을 복구하는 이유와 MiMo BrowseComp 행이 이 복구 한계에 대해 보이는 것을 설명하라.
6. GLM-5는 오류 agent 세그먼트를 버리거나 패널티하지 않고 마스킹한다. “negative”의 네 의미를 써서 보존 세그먼트에서 모델이 무엇을 배우고 무엇을 배우지 않는지 설명하라.
7. Kimi k1.5 long2short DPO는 chosen보다 1.5× 긴 정답을 rejected한다. 이 쌍 구성이 학습하는 것과 긴 추론이 필요한 문제에 만드는 위험을 설명하라.
8. 연구 보고서가 교사 샘플링 온도, 거부율, SFT 학습률을 미보고한다. 증류 단계에 대해 유효한 결론과 낼 수 없는 결론을 설명하라.

## 연결

- **이전(의존성):** ch-34 — Qwen, OLMo, Phi 보고서의 일반성 대 전문성 사례 연구 B. Qwen3/Phi-4-reasoning 단계 배치를 요약하며 정오표 6–7이 두 주장을 고친다.
- **의존성:** ch-20 — 데이터로서의 증류: 설명 trace와 R1-Distill 계보. R1-Distill의 데이터 관점이며 정오표 1–4가 네 주장을 고친다.
- **다음:** ch-35a — 증류의 실제 B: 프롬프트 선택, 교사 샘플링, 품질 필터. §3–§6의 프롬프트·교사 샘플·필터를 만드는 방법과 절제.
- ch-31 — 거부 샘플링, 자기생성 데이터, cold start, SFT–RL 교대. cold start와 모델 자신의 RL 체크포인트 자기증류.
- ch-31a — 지도 학습의 부정 샘플: 수정, 실패 조건화, 비평, unlikelihood; ch-43a — 부정 샘플과 음의 그래디언트: likelihood displacement, squeezing, 음의 이점. 부정 피드백 절의 유도.
- ch-30a — 미세조정의 망각·alignment tax: 측정과 제어; ch-30c — 일반ist 모델의 가중치 평균과 모델 merge. Generalization lens 회귀에 쓴 측정/merge법.
- ch-32 — 중간학습: annealing 데이터, 단계 게이트, 이후 SFT/RL 효과; ch-32d — 에이전트 중간학습: 사후학습 전 repository·실행 trace·궤적 데이터. S3 심화, erratum 5가 ch-32를 고침.
- ch-38a — SFT 대 RL 일반화: 온폴리시 데이터, base 모델 KL, 출력 다양성. RL 설정의 §1.2 온폴리시 논증.
- ch-44a — RL의 길이: 지나치게 긴 응답, 길이 제어, 긴 컨텍스트 RL. RL 중 S8.
- ch-45d — 공개 에이전트 레시피 나란히 보기: 단계 배치, 데이터 혼합, agentic RL; ch-58a — 공개 일반 모델 레시피 종단 간: 사전학습부터 merge. 여기의 위치를 포함한 전체 파이프라인.
- ch-36 — 마스킹 테스트, 망각 보고서, 홀드아웃 평가 분할을 갖춘 SFT 실행 실습; ch-57 — TRL 내부: SFT, DPO, GRPO, 증류 trainer. 증류 SFT 실행 구현/측정.

## 출처

- [[deepseek-r1]] — cold-start 구성, RL 프롬프트 혼합, 필터·V3 판정 프롬프트가 있는 800K 집합, 단계별 Table 3, 32B에서 증류 대 RL.
- [[deepseek-r1-recipe]] — 검증된 SFT/증류/학생-base 행(Table 6 학습률, epoch, batch).
- [[deepseek-v3]] 및 [[deepseek-v3-recipe]] — R1 유래 전문가 모델, 1.5M SFT 집합, Table 9 정확도·길이 절제.
- [[deepseek-v3.1]] 및 [[deepseek-v3.1-recipe]] — DeepSeek-V3.2 보고서 카드: specialist, mixed RL로 닫힌 증류 격차, 미보고 크기.
- [[deepseek-v4-recipe]] — full-vocabulary reverse KL 다중 교사 온폴리시 증류.
- [[glm-4-5]] 및 [[glm-4-5-recipe]] — 합성 추론 중간학습, expert→통합 SFT, 거부 필터, 프롬프트 길이 절단·응답 scaling, agent 자기증류.
- [[glm-5]] — SFT 난이도 필터·마스킹 오류 세그먼트, 온폴리시 단계 간 증류 목적/설정(arXiv:2602.15763v2에 대해 검증한 장 발췌).
- [[qwen-3]] — Qwen3 보고서 라이브러리 카드. 이 수치가 없고 검증 절도 없으므로, 본문 Qwen3 인용은 모두 arXiv:2505.09388v1에 대해 검증한 장 발췌 [[qwen-3-post-training]]을 가리킨다(cold-start 필터, thinking 융합, strong-to-weak 증류, Tables 21–22).
- [[kimi-k1-5]] 및 [[kimi-k1-5-recipe]] — long-CoT warm-up, RL 프롬프트 큐레이션, long2short 방법과 결과.
- [[kimi-k2]] 및 [[kimi-k2-recipe]] — K1.5·도메인 전문가 SFT 응답, 미보고 크기.
- [[llama-nemotron]] — post-NAS 증류, 도메인별 교사·생성 수, 33M 혼합, SFT 설정, SFT 대 RL, IFEval 효과(arXiv:2505.00949v5에 대해 검증한 장 발췌).
- [[nemotron-nano-2]] — R1-0528/Qwen3-235B-A22B SFT 데이터, 잘린·제거 trace, forward-KL 압축·post-GRPO 증류(arXiv:2508.14444에 대해 검증한 장 발췌).
- [[phi-4]] — 수치에 쓰지 않은 라이브러리 카드(Phi-4와 Phi-4-reasoning을 섞고 검증 없음); [[phi-4-reasoning]] — teachable 프롬프트, o3-mini 교사 비교, SFT/RL 설정, 전이·길이(arXiv:2504.21318v1에 대해 검증한 장 발췌).
- [[magistral]] 및 [[magistral-recipe]] — Medium trace에서 시작한 Magistral Small cold start, Table 3 SFT 대 RL 대 SFT + RL, open-trace 실험.
- [[gemma-2]] — 사전학습 증류 목적과 2B 절제.
- [[gemma-3]] — 샘플링 logit 256개, 토큰 예산, 소형 대 대형 교사(arXiv:2503.19786v1에 대해 검증한 장 발췌).
- [[llama-4]] — 사전학습 중 Behemoth에서 Maverick codistillation.
- [[minitron-approach]] — 교사 교정, forward-KL logit 전용 증류, 토큰 절약(arXiv:2408.11796v4에 대해 검증한 장 발췌).
- [[mimo-v2-flash]] — 도메인 교사 SFT 데이터, MOPD 목적, Table 7 학생 대 교사(arXiv:2601.02780v2에 대해 검증한 장 발췌).
- [[thinkingmachines-on-policy-distillation]] — 토큰별 reverse-KL 구현, 계산 비교, 망각 복구; 실무자 근거(블로그 페이지에 대해 검증한 장 발췌).
- [[smollm-3]] — SmolLM3 카드에 검증 절 없음; [[smollm-3-midtraining]]이 검증 발췌를 보유(증류 trace 35B토큰 × 4 epoch, RULER 회귀, merge 수리, Qwen3-32B/0.6B 선호 쌍).
