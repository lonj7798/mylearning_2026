<!-- chapter: ch-38a
     track: preference
     kind: content
     title: SFT 대 RL 일반화: on-policy 데이터, 기준 모델에 대한 KL, 출력 다양성
     deps: [ch-38, ch-30a]
     sources: [[sft-memorizes-rl-generalizes]], [[debunk-sft-generalization]], [[rls-razor]], [[retaining-by-doing]], [[rlhf-generalisation-diversity]], [[artificial-hivemind]], [[noveltybench]], [[on-off-policy-rlhf]], [[on-policy-suboptimal-preference-data]], [[rl-finetunes-small-subnetworks]], [[transferability-of-llm-reasoning]], [[transferability-of-llm-reasoning-tables]], [[rlvr-beyond-base-model]], [[rlvr-beyond-base-model-tables]], [[prorl]], [[wolfe-online-vs-offline-rl]], [[john-schulman-kl-tricks]]
     figures: figures/on-policy-kl-and-passk.html
     revised: 2026-09 (generality revision)
-->

# Chapter 38a — SFT 대 RL 일반화: on-policy 데이터, 기준 모델에 대한 KL, 출력 다양성

> **핵심 통찰.** 프롬프트, 과제, 시작 체크포인트를 고정한 통제 비교에서, 모델이 **스스로 샘플한 응답**으로 학습한 단계는 그 단계가 목표로 삼지 않은 능력을 더 많이 보존한다. Llama-3.1-8B-Instruct를 Countdown으로 학습했을 때, 외부 교사의 응답으로 SFT한 모델은 GPT-4 대비 AlpacaEval 승률이 41.6%에서 0.0%로 이동한 반면, RL 모델은 42.0%가 되었고, GRPO의 목표 과제 이득은 SFT보다 더 컸다(60.4 대 25.5포인트) ([[retaining-by-doing]] Table 1, Table 3). 수학만으로 학습한 Qwen3-14B-Base에서는 RL run이 비추론 평균을 45.7에서 53.2로 올렸지만 SFT run은 21.1로 내렸다 ([[transferability-of-llm-reasoning-tables]] Table 1). 두 메커니즘이 제안됐으며 어느 쪽도 확정되지 않았다. on-policy update는 base policy에 KL 관점에서 가까운 해로 수렴한다는 설명 ([[rls-razor]] §5)과, mode-seeking update가 기존 mode를 그대로 보존한다는 설명 ([[retaining-by-doing]] §3)이다. 이 이점은 RL objective 자체의 성질이 아니다. [[rls-razor]] §5.1에서는 negative gradient가 없는 on-policy 방법(1–0 REINFORCE)이 GRPO처럼 행동했고, negative gradient가 있는 offline 방법(SimPO)은 SFT처럼 행동했다. 또 [[retaining-by-doing]] §4.2에서는 매 epoch 시작에 다시 생성한 데이터로 SFT했을 때 SFT의 목표 정확도에 도달하면서 망각은 “mild to no forgetting”이었다. 다른 연구는 on-policy preference optimization과 RL의 비용도 측정했다. RLHF 뒤에는 입력 하나당 출력 다양성이 감소했고 ([[rlhf-generalisation-diversity]] §6.2), 모든 OLMo 2 크기에서 SFT → DPO → RLVR을 거치며 서로 다른 생성의 수가 감소했으며 ([[noveltybench]] Fig. 6), RLVR 모델은 큰 k의 pass@k에서 base model에 추월당했다 ([[rlvr-beyond-base-model-tables]] §3.2).
>
> **지침.** post-training 단계가 한 능력은 높이되 나머지 능력은 낮추지 않아야 한다면, 학습 중인 체크포인트에서 응답을 생성하라. [[rls-razor]] §5.1, [[retaining-by-doing]] §4, [[transferability-of-llm-reasoning-tables]] Table 4에서 retention profile을 가르는 것은 loss function보다 이 응답 출처였다. 완전한 on-policy sampling이 너무 비싸다면 매 epoch 시작에 target을 다시 생성하라. Iterative-SFT는 SFT와 같은 목표 정확도에 도달하면서 mild to no forgetting을 보였다 ([[retaining-by-doing]] §4.2, Fig. 7). 외부 교사의 응답을 반드시 써야 하는 경우(모델이 아직 올바른 응답을 전혀 만들지 못하는 경우)에는 [[retaining-by-doing]] Table 1에서 측정된 retention 비용을 예상하고 ch-30a의 control로 줄여라. 또한 SFT의 out-of-distribution 점수가 낮다고 해서 prompt template을 바꾸어 보기 전부터 SFT objective의 증거로 간주하지 마라. prompt diversity만으로 Sokoban instruction-variant 성공률이 0에서 0.84–0.92로 이동했다 ([[debunk-sft-generalization]] Table 1a). 단계를 평가할 때는 pass@1 옆에 큰 k에서의 pass@k, diversity metric, 그리고 단계 입력 체크포인트에 대한 KL을 함께 보고하라. [[rlvr-beyond-base-model-tables]] §3.2에서는 pass@1이 올랐지만 coverage가 떨어졌고, [[noveltybench]] §4.4에서는 utility가 올랐지만 diversity가 떨어졌다.

## 범용 모델에서 이 장이 중요한 이유

파이프라인 위치: pre-training → mid-training → SFT → **preference optimization 및 RL(이 장의 비교)** → evaluation. ch-30은 SFT 설계 선택지를 다뤘고, ch-30a는 fine-tuning 단계가 목표로 삼지 않은 능력에 치르는 비용을 측정했으며, ch-37은 policy-gradient estimator를 유도했고, ch-38은 KL로 제어한 PPO objective를 추가했다. 이 장은 같은 프롬프트에 두 가지 학습 방식을 적용할 때 어느 방식이 학습 분포 밖에서 더 나은 모델을 만드는지, 그리고 그 답이 출력 다양성에 어떤 비용을 요구하는지를 묻는다.

이 질문은 알고리즘 이름이 아니라 증거로 결정한다. 세 가지 발견이 이 장을 구성한다.

1. 인용한 연구들에서 SFT와 RL 사이의 retention 차이는 reward 포함 여부나 negative term 포함 여부가 아니라 **학습 응답의 출처**를 따라간다(§2, §5).
2. “SFT memorizes”라는 가장 널리 인용되는 실증은 하나의 고정 prompt template으로 만들어졌고, 데이터를 바꾸면 그 실패의 대부분이 사라진다(§3).
3. 능력을 가장 잘 보존하는 단계도 출력 분포는 좁힌다. 프롬프트당 서로 다른 답이 줄고, 큰 sampling budget에서 coverage가 낮아진다(§6, §7, §8).

이 장 전체에서 쓰는 정의는 다음과 같다. **On-policy data**는 현재 update 중인 체크포인트에서 샘플한 응답이다. **Approximately on-policy data**는 예를 들어 각 epoch 시작 시점처럼 현재 체크포인트의 최근 복사본에서 샘플한 응답이다. **Off-policy data**는 다른 모델에서 나온 응답 또는 고정 데이터셋의 응답이다. **Forgetting**은 어떤 단계 뒤에 그 단계가 목표로 삼지 않은 능력의 점수가 떨어지는 것이다(ch-30a §1). **Coverage**는 k개의 sample 안에서 적어도 한 번 풀린 문제의 비율이다.

## §1 두 축이 방법을 가른다: 샘플 출처와 그래디언트 부호

**정의.** 이 장의 모든 방법은 두 축에 놓을 수 있다. loss를 평가할 때 y가 어디에서 왔는지, 그리고 어떤 항이 샘플된 y의 likelihood를 낮추는지이다.

**이 framing이 해결하려는 문제.** “RL이 SFT보다 낫다”라고 보고하는 비교는 대개 두 축과 데이터까지 한 번에 바꾸므로, 결과를 어느 요소에 귀속할 수 없다. [[rls-razor]] §5.1은 같은 프롬프트와 같은 과제로 네 개의 cell을 모두 채워 이 문제를 고정한다.

| 방법 | Sample source | Negative gradient | 예시 위치 |
|---|---|---|---|
| 교사 응답에 대한 SFT | off-policy | 없음 | [[retaining-by-doing]] §2.2 |
| Self-SFT (초기 policy에서 한 번 샘플하고 올바른 것만 filter) | 초기 policy에서 한 번 샘플 | 없음 | [[retaining-by-doing]] §2.2 |
| Iterative-SFT / rejection-sampling FT | 매 epoch 다시 생성 | 없음 | [[retaining-by-doing]] §4.2 |
| 1–0 REINFORCE | on-policy | 없음 (A = 1 또는 0) | [[rls-razor]] §5.1 |
| GRPO, PPO | on-policy | 있음 (baseline 아래에서는 A < 0) | [[rls-razor]] §5.1, ch-38, ch-40 |
| DPO, IPO, SimPO (offline) | off-policy pair | 있음 (rejected term) | [[on-off-policy-rlhf]] §2, ch-39 |
| Online / semi-online DPO | 매 s step마다 다시 생성한 pair | 있음 | [[wolfe-online-vs-offline-rl]] |

두 objective를 sampling distribution과 weight만 다르게 보이도록 쓰면 다음과 같다 ([[transferability-of-llm-reasoning-tables]] Eq. 3).

```
L_{q,w,β}(θ) = − E_{x∼D} E_{y∼q(·|x)} [ w(x, y) · log π_θ(y | x) ] + β · E_{x∼D} KL(π_θ(·|x) ‖ π_ref(·|x))
```

- `q` — 응답을 draw하는 distribution이다. SFT에서는 reference answer y\*에 놓인 point mass이고, on-policy 방법에서는 current policy π_θ이다.
- `w` — sample별 weight이다. SFT에서는 1이고, policy gradient에서는 advantage A(x, y)이다.
- `β` — reference policy π_ref에 대한 KL penalty의 coefficient이다. penalty를 쓰지 않으면 0이다.
- `D` — 비교하는 setting들 사이에서 동일하게 고정한 prompt distribution이다.

q = δ_{y = y\*}, w = 1로 두면 cross-entropy SFT가 된다. q = π_θ, w = A로 두면 policy gradient가 된다. §5의 ablation은 한 번에 field 하나만 바꾼다.

## §2 통제 비교가 측정한 것

**결과(서로 독립적인 세 연구에서 재현).** 프롬프트와 시작 체크포인트가 같을 때, on-policy 단계는 목표 과제 정확도가 같거나 더 높으면서 다른 곳의 하락은 더 작게 만들었다.

[[retaining-by-doing]] Table 1(목표 과제 이득 / 비목표 과제 전체의 평균 하락, percentage point 단위. 비목표 세트에는 MMLU, MATH, IFEval, Countdown, WildJailbreak, WildGuardTest가 들어간다):

| 모델 | 방법 | IFEval | MMLU | Countdown |
|---|---|---|---|---|
| Llama 3.1 8B Inst. | SFT | 25.2 / 27.8 | 11.1 / 38.5 | 25.5 / 36.4 |
| | REINFORCE | 17.8 / 7.7 | 8.6 / −0.1 | 7.5 / −0.8 |
| | GRPO | 18.4 / 3.4 | 14.6 / −0.2 | 60.4 / −0.5 |
| Qwen 2.5 7B Inst. | SFT | 24.2 / 5.6 | 9.4 / 14.6 | 10.4 / 29.2 |
| | REINFORCE | 5.7 / 2.9 | 6.4 / −0.6 | 11.9 / −0.1 |
| | GRPO | 17.0 / 0.2 | 8.4 / 0.2 | 29.2 / −0.3 |

같은 run들은 어느 run도 목표로 삼지 않은 평가인 AlpacaEval에서도 점수를 냈다. GPT-4 대비 승률(%)로 App. A.6, Table 3에 보고된 값이다. Llama-3.1-8B-Instruct는 41.6에서 출발해 IFEval / MMLU / Countdown으로 SFT한 뒤 각각 23.5 / 11.6 / 0.0이 되었고, RL 뒤에는 각각 43.1 / 41.2 / 42.0이 되었다. Qwen2.5-7B-Instruct는 36.5에서 출발해 SFT 뒤 22.5 / 19.9 / 0.4가 되었고, RL 뒤에는 35.2 / 36.0 / 34.6이 되었다.

[[transferability-of-llm-reasoning-tables]] Table 1은 Qwen3-14B-Base를 47K개 수학 문제로 학습한다. SFT target은 Qwen3-32B에서 rejection sampling으로 뽑고, RL은 답 정답 여부에 GRPO를 적용한다. 수학 평균은 base 27.7, SFT (think) 49.8, RL 53.8이다. 다른 추론 평균(GPQA, LiveCodeBench2, ACPBench, HeadQA)은 base 30.2, SFT 45.3, RL 60.0이다. 비추론 평균(CoQA, MC-TACO, IFEval, HaluEval)은 base 45.7, SFT (think) 21.1, SFT (no-think) 29.0, RL 53.2이다. 두 fine-tuning 방법 모두 수학은 올렸지만, 비추론 평균을 base model보다 높인 것은 RL run뿐이었다.

[[rls-razor]] §3.1은 Qwen 2.5 3B-Instruct를 수학, SciKnowEval의 Chemistry L-3 subset, ToolAlpaca로 학습한다. 각 방법마다 learning rate, batch size, schedule을 sweep하고, 새 과제 정확도와 HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval 평균의 Pareto frontier를 비교한다. 여기서 Pareto frontier란 같은 방법의 run 중 새 과제 정확도와 이전 과제 평균이 둘 다 더 높은 다른 run이 없는 run들의 집합이다. 논문은 그 frontier에서 정확도 2 point 이내인 run을 남긴다(App. B.1). 새 과제 정확도가 증가해도 RL frontier의 이전 과제 점수는 거의 평평하게 유지되는 반면 SFT frontier는 내려간다. 같은 패턴은 SimplerEnv의 OpenVLA-7B에서도 나타난다. 이때 보존 과제는 drawer opening 및 closing이다(§3.1). ToolAlpaca의 tool-use 과제는 이 비교에서 agentic 사례다.

[[sft-memorizes-rl-generalizes]]는 이들 비교 중 가장 이른 연구이며, 제시한 효과 크기도 가장 강하다. Llama-3.2-Vision-11B의 하나의 SFT checkpoint에서 시작해 RL과 SFT compute를 각각 늘렸다. 규칙 변화(J/Q/K는 학습 중 10, 평가 때 11/12/13) 아래 GeneralPoints language variant는 RL로 11.5% → 15.0%, SFT로 11.5% → 3.4%가 되었다. step별 정확도로 채점한 V-IRL language variant는 RL로 80.8% → 91.8%, SFT로 80.8% → 1.3%가 되었다(§5.1, Fig. 6). 시각 변화(black suit에서 red suit로, New York route에서 V-IRL mini benchmark로) 아래 vision-language variant는 RL로 23.6% → 41.2%, 16.7% → 77.8%가 된 반면 SFT로는 23.6% → 13.7%, 16.7% → 11.1%가 되었다(§5.2, Fig. 7).

**조건과 한계.** 네 연구 모두 1B에서 14B 사이 모델을 하나의 좁은 목표 과제로 fine-tune한다. 어느 연구도 long-context retention을 보고하지 않는다. 비목표 suite는 short-context이므로, 이 장의 어떤 내용도 context length에 관한 주장을 뒷받침하지 않는다(not reported). Vision-language model을 다루는 연구는 [[sft-memorizes-rl-generalizes]]뿐이다.

## §3 반증 근거: 이 SFT run들이 하지 않은 일

**결과(단일 연구).** [[debunk-sft-generalization]]은 Sokoban과 General Points라는 두 의사결정 과제에서 같은 비교를 수행한다. General Points 환경은 [[sft-memorizes-rl-generalizes]]가 쓴 환경이다. Qwen2.5-7B 및 Llama-3.1-8B-Instruct를 사용한 이 연구는, SFT 실패의 대부분을 objective가 아니라 고정된 prompt template에 귀속한다. 진단 장치는 **Fake environment**다. prompt에는 새 지시(예: 숫자 action)를 적어 놓지만, scorer는 계속 training token을 정답으로 받는다. answer-only SFT는 실제 instruction variant에서는 0점을 받으면서 Fake Sokoban에서는 Qwen 0.93, Llama 0.90을 받는다. 모델이 바뀐 지시를 따르지 않고 학습 때의 의미론을 실행하고 있음을 보인다(§5).

학습 때 prompt diversity를 더하면, 즉 example마다 서로 다른 action name 또는 서로 다른 face-card mapping을 sample하고 그 mapping을 prompt에 명시하면, Sokoban 행은 0.98 / 0 / 0 / 0 / 0.93 (ID / alphabetical / numeric / random / Fake)에서 0.91 / 0.92 / 0.89 / 0.84 / 0 (Table 1a)로 바뀐다. Chain-of-thought target은 난이도 transfer도 더한다. General Points에서 Qwen answer-only는 large-number split에서 0.02, five-card split에서 0.00인 반면, CoT는 0.80 및 0.26, Diversity + CoT는 0.84 및 0.29를 받는다. 이는 warm-started RL baseline의 0.69 및 0.22와 비교한 값이다(Table 1b).

**조건.** 과제는 두 개, backbone도 두 개이며, 의사결정만 다룬다. 저자들은 open-ended 및 creative generation을 평가하지 않았다고 명시한다(§7). RL baseline은 warm-started GRPO이고, Llama의 Sokoban in-distribution 점수에서는 SFT보다 낮다(0.92 대비 0.43). 따라서 두 방법을 같은 수준으로 tuning한 비교가 아니다.

pro-RL 쪽에서 나온 두 추가 조건도 같은 방향을 가리킨다.

- base Llama-3.2-Vision-11B에서 SFT stage 없이 RL을 시작하면 실패했다. base model이 reward를 추출할 수 없는 비구조적 응답을 냈기 때문이다([[sft-memorizes-rl-generalizes]] §5.4, Fig. 9). 지나치게 tuning된 SFT checkpoint에 RL을 적용해도 out-of-distribution 성능은 회복되지 않았다(§6, Fig. 19).
- [[debunk-sft-generalization]] App. C는 SFT에 명시적 proximity control을 추가한다. loss는 SFT + α·KL(π_θ ‖ π_ref) 또는 SFT + α‖θ − θ_ref‖²이고, α ∈ {0.05, 0.1, 0.5}, epoch은 5다. instruction-variant 점수는 오르고 Fake 성공은 무너지지만, in-distribution accuracy와 difficulty-variant 점수는 떨어진다. proximity control은 ch-30a §2가 설명한 것과 같은 trade-off 위를 이동시키며, data change는 그렇지 않다.

**이 course에서의 해석.** 연구 대상이 loss가 아니라 data distribution이라면 두 근거 묶음은 양립한다. frozen prompt와 prompt당 외부 answer 하나로 SFT한 run은 좁은 input–output mapping을 학습한다. 반대로 prompt diversity와 on-policy resampling은 update가 fit하는 대상의 폭을 넓힌다. 이는 이 course의 해석이며, 어느 논문 한쪽의 주장도 아니다.

## §4 새 과제에서 base model에 대한 KL

**정의.** [[rls-razor]]는 경험적 forgetting 법칙을 제안한다. 새 과제 τ에 fine-tuning했을 때 발생하는 forgetting은 다음 값으로 예측한다.

```
KL_τ = E_{x ∼ τ} [ D_KL( π₀(· | x) ‖ π(· | x) ) ]
```

- `π₀` — base(pre-fine-tuning) policy, `π` — fine-tuning된 policy.
- `x ∼ τ` — retention을 측정하는 과거 과제의 prompt가 아니라 **새** 과제에서 온 prompt다. 이것이 실무상 핵심이다. 이 predictor에는 이전 과제 data에 접근할 필요가 없다.
- 방향은 forward KL이며, 첫 번째 argument가 π₀다.

**근거.** ParityMNIST toy setting에서 KL_τ의 2차 다항식은 SFT 및 RL run 전체의 forgetting을 R² = 0.96으로 fit한다. LLM run에서는 R² = 0.71이다([[rls-razor]] §4, Fig. 11). 같은 논문의 Table 1은 toy task에서 후보 predictor를 비교한다. forward KL은 0.96 ± 0.01, reverse KL은 0.93 ± 0.01, total variation은 0.80 ± 0.01, Fisher-weighted L2 weight change는 0.58 ± 0.02, L1 weight change는 0.34 ± 0.02, activation change L1은 0.52 ± 0.02다.

**on-policy training이 더 낮은 KL에 도달하는 이유.** binary reward에서 현재 policy로부터 rejection sampling하면 distribution은 q_RS(y) = π₀(y)·1[R(y) = 1] / π₀(R = 1)가 된다. Lemma 5.1은 이것이 E_q[R] = 1이라는 조건 아래 min_q D_KL(q ‖ π₀)의 해라고 말한다. Theorem 5.2는 policy gradient가 representable family 안에서 KL-closest optimal policy로 수렴한다고 말한다. 방향이 바뀌는 점에 주의해야 한다. projection 결과는 reverse KL D(q ‖ π₀)로 서술되는 반면, forgetting 법칙은 forward KL을 쓴다. 논문은 두 방향을 형식적으로 연결하지 않는다.

**계산 예시.** prompt 하나에 candidate answer가 다섯 개 있고, (a1…a5)에 대한 base probability가 π₀ = (0.35, 0.25, 0.20, 0.12, 0.08)이라고 하자. verifier는 a2와 a4를 받는다.

1. π₀(accepted) = 0.25 + 0.12 = 0.37.
2. projection은 q_RS(a2) = 0.25/0.37 = 0.676, q_RS(a4) = 0.12/0.37 = 0.324다.
3. D(q_RS ‖ π₀) = 0.676·ln(0.676/0.25) + 0.324·ln(0.324/0.12) = ln(1/0.37) = **0.994 nats**.
4. 언제나 a4를 쓰는 annotator는 q_SFT = δ_{a4}를 만든다. 따라서 D(q_SFT ‖ π₀) = −ln 0.12 = **2.120 nats**이며, projection 비용의 2.13배다.
5. 두 distribution은 verifier 아래에서 모두 100% 정답이다. 차이는 전적으로 어떤 정답 distribution을 선택했는지에 있다.

[이 장의 figure](figures/on-policy-kl-and-passk.html) Panel 1은 다른 base distribution과 accepted set에 대해서도 이를 다시 계산한다. 3단계는 D(q_RS ‖ π₀) = −ln π₀(accepted)로 일반화된다. 따라서 projection 비용은 base model이 이미 acceptable answer에 얼마나 많은 probability mass를 두는지에만 의존한다.

**반대 근거(단일 연구).** [[retaining-by-doing]] App. A.5는 evaluation example 100개에서 KL[π₀ ‖ π]를 추정한다. 전체 model, method, dataset에 걸친 non-target drop과의 Pearson correlation은 0.52이지만, 두 SFT variant 사이의 관계는 non-monotone이다. MMLU로 학습한 Llama-3.2-1B-Instruct에서 Self-SFT는 KL 39.3 및 34.6-point drop을 보이는 반면, SFT는 더 큰 KL 48.4와 더 작은 28.9-point drop을 보인다. Countdown에서는 Self-SFT가 KL 796.7과 25.3-point drop, SFT가 KL 1254.7과 24.2를 보인다. 저자들은 이 관계가 “does not always hold in our setting”이라고 말한다. 이 KL 값의 단위는 명시하지 않는다.

**상태.** **열린 질문.** 새 과제에서의 KL은 지금까지 보고된 단일 predictor 중 가장 좋고 과거 과제 data를 필요로 하지 않는다. 그러나 LLM run에서 R² = 0.71이고 [[retaining-by-doing]]에서 Pearson 0.52이므로, retained ability를 직접 측정하는 일을 대체할 만큼 정확하지 않다.

**대안 설명.** [[retaining-by-doing]] §3은 policy를 old mode(prior knowledge)와 new mode(target)의 mixture로 모델링한다. 외부 target에 대한 cross-entropy가 하는 forward KL 최소화는 먼저 new mode를 늘린 뒤 target을 덮기 위해 old mode에서 mass를 빼낸다. on-policy update가 근사하는 reverse KL 최소화는 old mode의 모양을 유지하면서 new mode를 이동시킨다. 이 논문의 §3.2 결과는 조건에 중요하다. 초기 policy가 **uni-modal**이면 SFT가 RL보다 forgetting이 적다. 논문 실험에서의 순서는 initial policy가 multi-modal이라는 조건에 의존한다.

## §5 on-policy update가 더 작은 update인 이유

**token 수준의 메커니즘.** logits `z`에 대한 softmax에서 target token `y`의 log-probability gradient는 다음과 같다.

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

따라서 target 자체 logit에 가해지는 update의 크기는 `1 − p_y`다. model이 이미 높은 probability를 부여하는 token으로 학습하면 update는 작고, model이 가능성이 낮다고 보는 token으로 학습하면 update는 크다.

**계산 예시.** response 길이가 500 token이라고 하자. on-policy sample의 token들이 현재 policy 아래 평균 probability 0.6을 갖는다면, 평균 token당 loss는 `−ln 0.6 = 0.51 nats`이고 평균 target-logit gradient magnitude는 `1 − 0.6 = 0.4`다. 같은 policy에서 외부 teacher의 response token들이 평균 probability 0.05라면, 평균 loss는 `−ln 0.05 = 3.00 nats`, gradient magnitude는 0.95다. response 전체로 합치면 255 nats 대 1498 nats다. 이 비율은 §4의 KL 설명이 sequence 수준에서 재는 같은 양, 즉 현재 distribution으로부터의 거리다.

**gradient norm 근거.** [[transferability-of-llm-reasoning-tables]] Fig. 6은 Qwen3-8B의 네 setting에서 gradient norm을 기록했다. off-policy run은 학습 초기에 더 큰 step을 취하는 반면, on-policy run은 더 작고 더 안정적으로 유지된다고 보고한다. 논문의 component ablation은 같은 math data에서 §1의 두 축을 분리한다(Table 4; TI는 아래에서 정의하는 Transferability Index다).

| Setting | Math avg. | Other reasoning avg. | Non-reasoning avg. | TI_other | TI_non |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B-Base | 27.6 | 23.6 | 33.6 | – | – |
| Off-policy SFT | 41.9 | 34.4 | 26.6 | 18.3 | −40.5 |
| On-policy SFT | 33.7 | 35.7 | 35.0 | 68.6 | 30.2 |
| Off-policy RL | 45.5 | 35.9 | 31.7 | 36.4 | 4.5 |
| On-policy RL (no KL) | 37.1 | 38.2 | 35.8 | 65.6 | 39.3 |
| On-policy RL | 38.6 | 39.9 | 35.0 | 63.7 | 32.4 |

이 표는 한 field만 다른 행 쌍으로 읽는다. loss를 고정한 채 sample source를 바꾸면(행 2→3, 4→5) 두 transfer index가 모두 올라간다. sample source를 고정한 채 advantage와 negative example을 추가해도(행 2→4, 3→6) 둘 다 올라간다. on-policy RL에 KL penalty를 더하면(행 5→6) 변화가 작다. 이는 [[retaining-by-doing]] §4.1에서 β = 0.05인 GRPO와 β = 0인 GRPO가 Llama model을 IFEval로 학습한 경우를 제외한 모든 model과 dataset에서 유사한 gain–drop trade-off를 보였다는 결과와 일치한다. on-policy 선택의 cost는 math 열에서 보인다. off-policy SFT와 off-policy RL은 in-domain math average가 각각 41.9와 45.5로 가장 높고 transfer는 가장 나쁘다.

**subnetwork 주장과 precision 의존성.** [[rl-finetunes-small-subnetworks]]는 공개 checkpoint에서 update sparsity `1 − ‖θ₁ − θ₀‖₀/n`을 측정했다. RL stage 후에는 parameter의 68.5%–96.0%가 변하지 않았다고 보고한다(DPO: Tülu 3 8B에서 81.4%, Tülu 3 70B에서 95.2%; GRPO: DeepSeek-R1-Zero에서 86.0%; PPO 80.8%; PRIME 77.0%). 같은 model의 SFT stage는 6%–15%다(Table 1, Fig. 1). 식별한 subnetwork만 fine-tuning해도 full run parameter를 `10⁻⁵` tolerance에서 DPO는 94.0%, PRIME은 90.5% 동일하게 재현하며, `10⁻⁴`에서는 둘 다 100%다(§4). 논문의 Table 5는 원인을 data에 둔다. in-distribution data의 rejection-sampling SFT는 sparsity 91.2%를 만들지만, out-of-distribution data의 DPO는 7.7%다.

[[rls-razor]] §6은 같은 가설을 검증했고, 이 sparsity가 bfloat16의 artifact라고 밝혔다. “Performing the same training with float32 resulted in models with identical performance but without any sparsity in their weight updates.” Mukherjee et al.도 같은 rounding mechanism을 직접 언급한다(§6). **해석:** 방어 가능한 주장은 on-policy update가 충분히 작아서 그중 큰 비율이 bfloat16 resolution 아래로 내려간다는 것이다. RL이 고정된 subnetwork를 구조적으로 편집한다는 주장은 precision을 바꾸면 뒷받침되지 않는다. 따라서 float32에서 비교를 반복하지 않았다면 update sparsity를 retention의 독립적 설명으로 인용해서는 안 된다.

## §6 Online과 offline preference optimization의 비교

**정의.** **offline** preference method는 고정된 pair dataset을 최적화한다. **online** method는 current policy에서 pair(또는 적어도 response)를 sample하고 reward model 또는 preference model로 label을 붙인다. **semi-online** method는 최적화 step `s`회마다 sampling policy를 resync한다. `s = 1`이면 완전한 on-policy이고, `s → ∞`이면 offline이다([[wolfe-online-vs-offline-rl]]).

**통제된 결과.** [[on-off-policy-rlhf]](Tang et al.; T5X Large 770M policy, IPO loss, 네 dataset: OpenAI summarization, Anthropic helpfulness, Chat arena side-by-side, Anthropic harmlessness)는 loss, hyperparameter, source preference data를 고정하고 response가 어디에서 오는지만 바꾼다. online run은 네 dataset 모두에서 SFT policy에 대한 KL과 고정 golden policy를 상대로 한 win rate 사이에서 더 나은 trade-off, 그리고 더 높은 peak win rate에 도달한다(§2.1, Fig. 1). 같은 논문의 네 가설 검정은 그 차이의 원인을 다음처럼 좁힌다.

1. **coverage가 설명은 아니다.** online run이 만든 자기 data를 저장해 uniform shuffle로 replay하면, Chat arena side-by-side를 제외하고 plain offline training보다 개선이 거의 없다(§5.1, Fig. 4). 같은 data를 원래 순서로 학습하면 online run이 재현되며, training statistic도 약 0.1% 이내로 맞는다(App. F).
2. **data quality가 설명은 아니다.** 강한 4k-step online policy가 다시 생성하고 golden model이 relabel한 pair도 offline training을 조금만 개선한다(§5.2, Fig. 5).
3. **classification accuracy는 generation으로 transfer되지 않는다.** offline policy는 pairwise classifier로서 60–70% accuracy에 도달하지만 online policy는 50% 아래에 머무른다. 또 offline run 안에서는 accuracy와 win rate가 거의 상관이 없다(§5.3, Figs. 7–8).
4. **선택된 response의 likelihood는 떨어진다.** 모든 run에서 선택된 response의 likelihood는 SFT policy 대비 낮아지며, offline에서 가장 크게 낮아진다(§5.3.3, Fig. 8의 bottom row). 따라서 offline contrastive update는 “winner에 대한 SFT”가 아니다.

[[on-policy-suboptimal-preference-data]]는 각 축이 언제 이득을 내는지도 추가한다. on-policy sampling은 “reward의 peak가 reference policy에서 더 멀리 떨어져 있을 때 특히” 도움이 된다. negative gradient는 “reward의 peak가 `π_ref`의 낮은-likelihood 영역에 나타날 때” 도움이 된다. 둘은 상보적이다. sampling이 coverage를 제공하고 contrastive term이 batch당 더 강한 signal을 제공한다(§5.1–5.3 takeaway box). 같은 논문은 sample reuse가 exploration과 trade-off를 이룬다고 기록한다. on-policy data의 각 batch에 `T`번 gradient step을 취할 때, bandit result는 `T`에 대해 단조롭지 않다(`T = 5`는 `T = 2`보다 더 빨리 학습했고, `T = 10`은 `T = 7`보다 더 빨리 학습했다; Fig. 10). synthetic length-control task에서는 on-policy best-of-N에서 `T = 2`가 `T = 1`을 이겼지만, 과도한 reuse는 해로웠다. PPO는 `T = 8`까지 영향을 받지 않았고, 저자들은 PPO의 off-policy correction이 상당히 off-policy인 sample을 gradient에서 제외하기 때문이라고 설명한다(§5.1.2, Fig. 11).

**범용 모델에 대한 함의.** §2의 retention 성질은 algorithm보다 sample source를 따라갔다. 따라서 full RL loop의 budget이 없으면 PPO로 바꾸기보다 response를 주기적으로 regeneration하는 변경을 먼저 해야 한다. ch-39는 이를 DPO에, ch-31은 rejection-sampling SFT에, ch-40은 group-baseline RL에 적용한다.

## §7 다양성의 비용

**정의.** **입력당 다양성(per-input diversity)**은 하나의 prompt에 대해 뽑은 `K`개 sample의 다양성이다. **입력 간 다양성(across-input diversity)**은 prompt마다 sample 하나를 취해 구성한 집합의 다양성이다([[rlhf-generalisation-diversity]] Eq. 2–3; temperature 1에서 `K = 16`, `N = 500`). 지표는 expectation-adjusted distinct n-gram(EAD), Sentence-BERT embedding의 평균 cosine similarity에 대한 `1 −` 값, NLI 기반 다양성이다.

**근거.** TL;DR summarization의 LLaMa 7B에서 RLHF는 입력당 설정의 EAD와 Sentence-BERT 모두에서 SFT보다 “훨씬 낮은 output diversity”를 보였다. 입력 간 다양성의 감소는 더 작지만 존재한다. 한편 best-of-16은 입력 간 다양성을 SFT와 비슷하거나 그보다 높게 유지하므로, 원인이 reward model 하나만은 아니다([[rlhf-generalisation-diversity]] §6.2, Fig. 5–6). KL penalty coefficient를 높이면 입력당 다양성과 성능이 함께 낮아졌다. 따라서 `β`는 둘 중 하나를 다른 하나와 교환하는 knob가 아니다(§6.3, App. I). 저자들은 instruction-following model에서는 측정 가능한 차이를 보지 못했으며, 그 이유를 짧은 output을 위해 설계된 metric에 돌린다(§6.2).

**새로움과 품질을 결합하는 지표.** [[noveltybench]]는 deberta-v3-large classifier로 `k`개 generation을 기능적으로 동등한 equivalence class로 나눈다. 이 classifier는 사람 label과 79% agreement, `F1 = 0.811`을 보였다. 이어 다음을 정의한다.

```
distinct_k = |{c_i : i ∈ [k]}|
utility_k  = ((1 − p) / (1 − p^k)) · Σ_{i=1..k} p^{i−1} · 1[c_i ≠ c_j for all j < i] · u_i
```

- `c_i` — `i`번째 generation의 equivalence class이다. indicator는 해당 class에서 처음 나온 generation일 때만 1이다.
- `u_i` — reward model이 `{1,…,10}`으로 mapping한 `i`번째 generation의 품질이다.
- `p` — 사용자가 한 번 더 generation을 요청할 확률, 즉 사용자의 patience다. 평가는 `p = 0.8`, `k = 10`, temperature 1을 사용한다.

**계산 예.** utility가 `u = (8, 7, 6, 9)`이고 class가 `(A, A, B, C)`인 네 generation을 생각하자. `p = 0.8`, `k = 4`다. 정규화 상수는 `(1 − 0.8)/(1 − 0.8⁴) = 0.2/0.5904 = 0.3387`이다. 새 class인 generation은 1, 3, 4뿐이므로 weighted sum은 `1·8 + 0.64·6 + 0.512·9 = 8 + 3.84 + 4.608 = 16.448`이고, `utility₄ = 0.3387 × 16.448 = **5.57**`이다. 품질이 10인 서로 다른 답 네 개를 생성하는 model은 `0.3387 × 10 × (1 + 0.8 + 0.64 + 0.512) = 10.0`을 얻으며 이것이 최대값이다. 두 번째 generation은 class A를 반복하므로 기여가 0이다. 다시 요청한 사용자에게 좋은 답을 반복해 주는 일은 가치가 0이기 때문이다.

**stage별로 측정한 효과.** OLMo 2 checkpoint의 SFT, DPO, RLVR 후 `distinct₁₀`은 다음과 같다. 1B: `8.83 → 8.08 → 7.85`; 7B: `7.46 → 5.96 → 5.72`; 13B: `7.47 → 5.61 → 5.16`; 32B: `7.25 → 5.22 → 5.08`. 같은 stage에서 `utility₁₀`은 1B: `2.40 → 3.03 → 3.21`; 7B: `4.02 → 4.35 → 4.32`; 13B: `4.29 → 4.62 → 4.46`; 32B: `4.22 → 4.63 → 4.62`였다([[noveltybench]] §4.4, Fig. 6). 모든 크기에서 다양성은 내려갔고, 가장 큰 하락은 DPO에서 발생했다. 반면 utility는 SFT에서 DPO로 갈 때 상승했다. [이 장의 그림](figures/on-policy-kl-and-passk.html) panel 3은 이 `distinct₁₀` 값을 그린다.

**decoding만으로는 고쳐지지 않는다.** [[artificial-hivemind]]는 100개의 open-ended query마다 response 50개를 sample했다. 70개가 넘는 model을 측정했고, 그중 25개를 본문 figure에 넣었다. `top-p = 0.9`, temperature 1.0에서는 “79%의 경우 average similarity가 0.8을 넘는다.” 무작위 pair baseline은 완전히 0.1–0.2 band에 있다. `min-p = 0.1`, temperature 2.0에서도 “response pair의 81%가 여전히 similarity 0.7을 넘고 61.2%가 0.8을 넘는다.” model 간 average pairwise similarity는 71%–82% 범위다(DeepSeek-V3와 qwen-max-2025-01-25은 0.82, DeepSeek-V3와 gpt-4o-2024-11-20은 0.81). 따라서 여러 model에서 sample하는 것 역시 variety의 원천이 아니다. [[noveltybench]] §4.3은 앞선 답을 context에 넣고 다른 답을 요청하는 in-context regeneration이 세 frontier model을 사람 수준의 다양성 근처까지 올린다고 발견했다. 반면 paraphrasing과 다양성을 요청하는 system prompt는 “단지 marginally effective”했다.

**범용 모델에 대한 함의.** output diversity는 target-only report가 포착하지 못하는 측정 가능한 성질이다. benchmark accuracy를 가장 잘 보호한 stage도 이를 낮춘다. OLMo 2의 네 크기 모두에서 DPO와 RLVR를 거치며 `distinct₁₀`이 하락했다. ch-43은 RL 내부의 entropy와 diversity control을, ch-42는 narrowing에서 judge가 맡는 부분을 다룬다.

## §8 전이와 pass@k 경계 논쟁

**전이 지표.** [[transferability-of-llm-reasoning-tables]]는 math group에 대해 group `g`의 Transferability Index를 다음과 같이 정의한다.

```
ΔR_b = R_b^model − R_b^base      σ_g = Std{ΔR_b : b ∈ B_g}      δ_b = ΔR_b / σ_g
s_b  = sign(δ_b)·|δ_b|^{1/2}      w_b = 100 − R_b^base      ŵ_b = w_b / Σ_u w_u
DI_g = Σ_b ŵ_b s_b               TI_g(%) = 100 · DI_g / DI_math
```

- `R_b` — benchmark `b`에서의 accuracy, `B_g` — group `g`에 속하는 benchmark들의 집합이다.
- `σ_g` — group 내부 gain의 spread이며, group들을 같은 scale에 놓기 위해 사용한다. 논문은 이것이 sample standard deviation인지 population standard deviation인지 밝히지 않는다. benchmark가 네 개이면 두 값은 `√(4/3)`배 차이가 난다.
- square root는 극단적으로 큰 gain을 완화하고, `w_b`는 base model이 약했던 benchmark에 더 큰 가중치를 준다.

**계산 예(benchmark 두 개로 된 group).** base score가 40, 30이고 gain이 `+10`, `+2`라고 하자. population standard deviation을 쓰면 `σ = 4`이므로 `δ = (2.5, 0.5)`, `s = (1.581, 0.707)`이다. weight는 `w = (60, 70)`, `ŵ = (0.462, 0.538)`이다. 따라서 `DI = 0.462·1.581 + 0.538·0.707 = 0.730 + 0.381 = 1.111`이다. math group의 `DI`가 1.4라면 `TI = 79.4%`다. 식에서 두 결과가 바로 따른다. group 내 gain이 전부 같으면 `σ = 0`이므로 index는 정의되지 않으며, `TI`는 signed 값이므로 `DI_math`가 음수이면 보고되는 모든 index의 부호가 뒤집힌다.

**pass@k 논쟁.** [[rlvr-beyond-base-model-tables]]는 base model과 RLVR model을 다음 unbiased estimator로 평가한다.

```
pass@k = E_{x_i ∼ D} [ 1 − C(n − c_i, k) / C(n, k) ]
```

여기서 `n`은 problem마다 뽑은 sample 수(MATH500, Minerva, GSM8K에서는 128개, AMC23과 AIME24에서는 1024개), `c_i`는 problem `i`의 correct sample 수, `C(a, b)`는 binomial coefficient다. RLVR model은 `k = 1`에서는 앞서지만, `k`가 커질수록 base model이 앞선다. 32B model의 Minerva에서는 `k = 128`에서 base model이 약 9 point 앞선다(§3.2). `k = 1024`의 AIME24와 `k = 128`의 MATH500에서 낸 solvable-problem table은 coverage가 어디로 갔는지를 보여 준다. base-only-solved는 각각 13.3%, 3.6%이고, RLVR-only-solved는 각각 0.0%, 1.0%다(Table 2).

**계산 예(두 주장이 동시에 참인 이유).** problem 두 개에서 각각 `n = 8`개 sample을 뽑는다.

- RL model: problem 1에서 6개가 correct이고 problem 2에서는 0개다. `pass@1 = (6/8 + 0)/2 = 0.375`다.
- Base model: problem 1에서 3개, problem 2에서 1개가 correct다. `pass@1 = (3/8 + 1/8)/2 = 0.250`이다.
- `k = 4`에서 RL model은 problem 1에 대해 `1 − C(2,4)/C(8,4) = 1` (`C(2,4) = 0`), problem 2에 대해 0이므로 0.500이다. Base model은 `1 − C(5,4)/C(8,4) = 1 − 5/70 = 0.929`, `1 − C(7,4)/C(8,4) = 1 − 35/70 = 0.500`이므로 0.714다.

RL model은 `k = 1`에서 12.5 point 앞서고 `k = 4`에서는 21.4 point 뒤처진다. 모순이 아니다. problem 1에 mass를 집중했고 problem 2는 완전히 잃었기 때문이다. [이 장의 그림](figures/on-policy-kl-and-passk.html) panel 2는 다른 count에서도 이 crossover를 다시 계산한다.

**반대 연구.** [[prorl]]은 KL penalty와 주기적인 reference-policy reset을 사용하여 다섯 domain에 걸친 136K개의 verifiable problem으로 DeepSeek-R1-Distill-Qwen-1.5B를 2k step 넘게 학습한다. training에서 보지 않은 Reasoning Gym task인 boxnet에서는 starting checkpoint가 `k = 256`까지 어느 `k`에서도 “exhibits no capability”하지 않지만, 학습 model의 average reward는 0.00에서 7.91로 오른다. 다른 unseen task에서는 non-zero start에서 오른다(acre `5.99 → 58.57`, game_of_life_halting `3.49 → 52.29`; §4.3, Table 3). graph colouring에서는 학습 때보다 더 큰 graph size에서 starting checkpoint를 넘는다. 같은 논문은 주로 math로 된 “Diminish” regime에서 반대 pattern도 재현한다. `pass@1`은 오르지만 `pass@128`은 내리며, boundary gain은 starting checkpoint의 `pass@128`과 negative correlation을 보인다(§4.1–4.2).

**상태.** **열린 질문**이지만 사용할 수 있는 요약은 있다. [[rlvr-beyond-base-model-tables]]가 보고한 narrowing은 starting checkpoint가 이미 높은 coverage를 갖던 영역에서 가장 강하다. [[prorl]]이 보고한 expansion은 checkpoint가 아무 coverage도 갖지 않던 task에서 나타난다. distillation은 세 번째 경우다. [[rlvr-beyond-base-model-tables]] §5는 distilled model이 base model을 넘어서는 reasoning coverage를 보인다고 보고한다. teacher가 base model이 전혀 sample하지 못하는 path를 공급하기 때문이다. §2에서 retention을 잃게 만든 같은 off-policy 성질이 새 capability를 들어오게 하는 성질이기도 하다(ch-20, ch-35).

## §9 이 장의 주장을 측정하는 방법

한 stage의 보고에는 그 stage 자체의 target metric 외에 네 가지 수량이 필요하다. 네 가지 모두 training run이 이미 끝난 상태에서 고정된 prompt set으로 sample하여 계산하므로, 어느 것도 추가 학습을 요구하지 않는다.

1. **stage 입력 checkpoint 대비 held-out capability delta**를 slice별로, paired 95% interval과 함께 보고한다(ch-30a §1, §6). 앞의 연구들이 이것을 구체적으로 보여 준다. [[retaining-by-doing]] Table 3의 AlpacaEval column은 target-only report라면 놓쳤을 값이다.
2. **고정된 prompt set에서 입력 checkpoint에 대한 KL**을 계산한다. 문헌마다 KL의 방향과 sampling distribution이 다르므로 둘 다 보고한다. [[rls-razor]]는 new-task prompt에서 forward KL을 쓴다. [[on-off-policy-rlhf]]는 256개의 training prompt에서 `KL(π_θ ‖ π_sft)`를 추정해 budget axis로 사용한다. [[transferability-of-llm-reasoning-tables]]는 task별 token-distribution KL을 보고한다. estimator 선택은 variance에 영향을 준다. `k1`과 `k3`는 [[john-schulman-kl-tricks]] 및 ch-38 §4에서 정의한다.
3. **큰 `k`에서의 pass@1과 pass@k**를 §8의 unbiased estimator로 측정한다. `n ≥ k`가 되도록 sample 수를 training 전에 정한다. 두 가지 control이 필요하다. 큰 `k`에서는 final-answer matching 때문에 틀린 reasoning도 점수를 받을 수 있으므로 chain of thought 일부를 검증한다([[rlvr-beyond-base-model-tables]] §3.1). Model을 비교할 때는 temperature가 아니라 entropy를 맞춘다. RLVR model의 output entropy가 base model과 같아질 때까지 temperature를 올려도 coverage는 base model보다 낮았다(App. C.8).
4. **다양성**은 고정 patience `p`를 둔 `distinct_k`와 `utility_k`로 측정하거나([[noveltybench]]), 고정 prompt set에서 input별 및 input 간 metric으로 측정한다([[rlhf-generalisation-diversity]]). similarity distribution은 `top-p`, `min-p`, temperature에 따라 움직이므로 decoding configuration을 고정하고 보고한다([[artificial-hivemind]] App. C.1). 또한 embedding 또는 judge model version을 고정한다. 예를 들어 0.8이라는 similarity threshold는 embedding model이 달라지면 같은 뜻이 아니다.

두 estimator를 위한 course code는 다음과 같다(출처에서 가져온 코드가 아니다).

```python
import math

def pass_at_k(n, c, k):
    """한 problem의 unbiased pass@k: n개를 sample했고 그중 c개가 correct (Yue et al. App. A.2)."""
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:          # 모든 k-subset에는 correct sample이 하나 이상 있다
        return 1.0
    # 1 - C(n-c, k) / C(n, k), 큰 n에서는 log space에서 계산한다
    log_ratio = sum(math.log(n - c - i) - math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_ratio)

def forward_kl_on_task(base_logprobs, tuned_logprobs):
    """base model에서 SAMPLE한 response들에 대한 E_{y~pi_0}[log pi_0(y) - log pi(y)].
    두 인수는 동일한 base-model sample의 sequence별 log-probability다."""
    diffs = [b - t for b, t in zip(base_logprobs, tuned_logprobs)]
    return sum(diffs) / len(diffs)
```

두 번째 function은 쉽게 틀리는 세부를 보여 준다. §4의 forward KL은 **base** model에서 response를 sample한 뒤 두 model 모두로 점수를 매겨야 한다. tuned model 자신의 sample을 점수화하면 reverse KL을 추정한다. 이것은 [[rls-razor]] Table 1에서 더 약한 predictor였다(0.96에 비해 0.93). 그리고 ch-38의 KL penalty가 control하는 수량이기도 하다.

## 음성 샘플과 음성 피드백

이 장이 사용하는 “negative”의 네 가지 의미(ch-43a)는 다음과 같다. GRPO, DPO, IPO, SimPO 비교에서는 **gradient로서의 negative**, Self-SFT와 rejection-sampling SFT의 filtering 단계에서는 **음의 한계가치(negative marginal value)**를 사용한다. **content로서의 negative**와 **conditioning으로서의 negative**는 여기에는 등장하지 않는다.

**negative의 출처.** RLVR 연구에서는 verifier 결과를 쓴다(정답 final answer에는 reward 1, 그 외에는 0: [[retaining-by-doing]] §2.2, [[rls-razor]] §3.1). [[sft-memorizes-rl-generalizes]] App. A.3은 shaped verifier를 쓴다. target과 같은 합법적 equation에는 +5, target을 맞히지 못한 합법적 equation에는 −1, verification step을 다섯 번 넘기면 −1, card에 없는 number를 쓰면 −2, 그 밖의 illegal equation에는 −3을 준다. vision variant에서는 card를 잘못 읽은 경우 −1.5를 추가로 준다. [[on-off-policy-rlhf]]와 [[rlhf-generalisation-diversity]]는 preference model 또는 reward model을 쓴다. [[rls-razor]] §5.1의 SimPO pair에서는 외부 model의 incorrect response가 rejected 쪽이다. 이들 어느 연구도 이 labeler의 false-negative rate는 보고하지 않는다.

**현재 관행에서의 처리.** Discard한다(Self-SFT와 rejection sampling은 correct response만 남긴다). 또는 gradient로 적용한다(GRPO와 PPO의 negative advantage, DPO·IPO·SimPO의 rejected term).

**메커니즘.** softmax gradient `∂ log p_y/∂z_j = 1[j = y] − p_j`에서 sampled sequence에 음의 weight를 주면 그 sequence의 token들에서 mass를 빼며, normalization은 빠진 mass를 현재 probability에 비례해 남은 token들로 보낸다. 따라서 model이 이미 unlikely하다고 판단하는 sample을 아래로 누르면, 대부분의 mass는 현재 가장 likely한 alternative로 이동한다. 그 alternative는 correct하지도 diverse하지도 않을 수 있다. [[on-policy-suboptimal-preference-data]]는 이것이 유용해지는 조건을 제시한다. negative gradient는 “`when y_l is sufficiently different from y_w, model capacity is large, and π_ref is chosen appropriately`”일 때 `y_w`의 likelihood를 올릴 수 있다. 그렇지 않으면 margin은 여전히 커지지만, 회수된 probability mass는 `y_w`가 아니라 다른 response들의 likelihood를 높이는 데 들어간다. ch-43a가 displacement와 squeezing case를 전부 유도한다.

**수치를 포함한 근거.** (a) negative는 prior capability를 보존하는 요소가 아니다. [[rls-razor]] §5.1에서 on-policy 1–0 REINFORCE(negative gradient 없음)와 GRPO(negative gradient 있음)는 같은 forgetting-versus-KL curve 위에 있었고, offline SimPO(negative gradient 있음)는 SFT와 함께 있었다. [[retaining-by-doing]] Table 1에서 advantage estimator 없는 REINFORCE는 GRPO와 비슷하게 작은 drop을 유지했고, 둘 다 SFT보다 훨씬 작았다. 예를 들어 Llama 3.1 8B의 IFEval에서는 7.7 point 대 3.4 point다. 다만 target gain은 더 낮았다(IFEval 17.8 대 18.4, Countdown 7.5 대 60.4). (b) negative는 transfer와 speed에는 기여한다. [[transferability-of-llm-reasoning-tables]] Table 4에서 fixed sample source에 advantage와 negative example을 더하자 off-policy pair의 `TI_other`는 18.3에서 36.4로, `TI_non`은 −40.5에서 4.5로 올랐다. on-policy pair에서는 `TI_other`가 오르지 않았다(KL term 포함 시 `68.6 → 63.7`, 없으면 65.6). 반면 `TI_non`은 각각 `30.2 → 32.4`, 39.3으로 올랐다. 이 논문은 single run만 보고하므로 작은 차이를 run-to-run variation과 구분할 수 없다. [[on-policy-suboptimal-preference-data]] §5.3에서는 on-policy contrastive training이 negative term 없는 on-policy RL보다 더 빨리 수렴했다. (c) negative에는 likelihood 비용이 측정되어 있다. [[on-off-policy-rlhf]] §5.3.3에서 **chosen** response의 log-probability는 모든 run에서 SFT policy보다 낮아졌고, offline에서 가장 강했다. 인용한 어떤 source도 전체 개선 가운데 negative term이 기여한 비중을 측정하지 않았으므로, 이 장은 그 비중을 주장하지 않는다.

**control.** negative를 on-policy로 유지한다(§2의 모든 근거). ratio clipping으로 update를 bound한다. 이것은 sample reuse도 더 안전하게 만든다([[on-policy-suboptimal-preference-data]] §5.1.2). verifier의 false-negative rate를 모르면 failure를 penalize하기보다 discard한다. negative term이 offline이면 positive term으로 anchor한다(ch-39, ch-43a).

**진단.** reference에 대해 chosen과 rejected sequence log-probability를 따로 log한다([[on-off-policy-rlhf]] Fig. 8 결과를 만든 diagnostic). advantage sign별로 나눈 statistic, entropy, 큰 `k`의 pass@k와 `distinct_k`를 log한다. §7과 §8의 효과는 pass@1에서는 보이지 않기 때문이다.

**일반성에 미치는 효과.** §8의 coverage, §7의 diversity, 그리고 같은 normalization 단계가 model들 사이에서 한 phrasing에 probability가 집중되는 정도에 미치는 영향이 있다. 후자는 [[artificial-hivemind]]가 측정한다.

## 실전 레시피

모든 row는 2026-09-15에 명시한 source location에서 읽었다. 이것들은 비교 연구의 setting이다. production run의 권장 default가 아니라 비교를 재현하기 위한 reference point다.

| Model (정확한 release) | 크기 | 단계 | Setting | 값 | Source location | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct (RL's Razor) | 3B | SFT | LR sweep; optimizer; schedule; warmup; epochs; batch; precision | {1e-5, 3e-5, 5e-5, 7e-5, 9e-5}; AdamW; warmup 포함 {constant, cosine}; 50 steps; {1, 2}; {16, 32, 64, 128}; bf16 | arXiv:2509.04259v1 App. B.1 Table 2 | 2026-09-15 verified | §3.1: Pareto frontier를 만들기 위한 sweep이며, frontier에서 2 point 이내 model을 유지 |
| Qwen2.5-3B-Instruct (RL's Razor) | 3B | RL | algorithm; LR sweep; KL reg.; group size; prompts per generation; inner iterations; loss | GRPO; {1e-5, 2e-5, 3e-5, 4e-5, 5e-5}; 0; 64; 8; {1, 2}; Dr. GRPO | arXiv:2509.04259v1 Table 2 | 2026-09-15 verified | 이 값들을 분리한 ablation은 보고되지 않음 |
| Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct (Retaining by Doing) | 8B, 7B | SFT, RL | LR; schedule; batch; epochs; max length; GRPO KL coef.; group size | 5e-6; cosine, warmup ratio 0.03; 128 (IFEval, MMLU), 64 (Countdown); 2; 4096; 0.05; 5 | arXiv:2510.18874v3 App. A.3 | 2026-09-15 verified | Fig. 6: `β = 0.05`와 `β = 0`은 비슷한 gain-drop trade-off |
| Llama-3.2-1B-Instruct, Qwen-2.5-1.5B-Instruct (Retaining by Doing) | 1B, 1.5B | SFT, RL | LR | 1e-4 | arXiv:2510.18874v3 App. A.3 | 2026-09-15 verified | Fig. 3: Self-SFT에 대한 LR `1e-5` 대 `1e-4`, epoch 2 대 10 |
| Qwen3-14B-Base (UniReason RL) | 14B | RL | framework; algorithm; LR; train batch; clip range; rollouts per prompt; mini-batch; max sequence; KL and entropy coef.; steps | verl; GRPO; 1e-6; 512; 0.22–0.28; 16; 128; 16k tokens; 0; 140 | arXiv:2507.00432v2 App. A.3.1 | 2026-09-15 verified | Qwen3-8B에서 sampling source, credit assignment, KL의 Table 4 ablation |
| Qwen3-14B-Base (UniReason SFT) | 14B | distill-SFT | framework; LR; batch; epochs; targets | LLaMA-Factory; 5e-5; 512; 1.5; Qwen3-32B think-mode response, correctness로 rejection-sampling, 47K math problem | arXiv:2507.00432v2 App. A.3.1–A.3.2 | 2026-09-15 verified | “our RL settings에 맞추기 위해” 선택; ablation 보고 없음 |
| Qwen2.5-7B, Llama-3.1-8B-Instruct (Debunk) | 7B, 8B | SFT | LR; batch; max response; epochs; optimizer; weight decay; grad clip | 1e-5; 128; 7000; 15; AdamW; 0.01; 1.0 | arXiv:2510.00237v1 App. B.3 Table 4 | 2026-09-15 verified | Table 1: Fake를 제외한 보고 metric 평균으로 best checkpoint 선택 |
| Qwen2.5-7B, Llama-3.1-8B-Instruct (Debunk) | 7B, 8B | RL | algorithm; LR; batch; mini-batch; KL coef.; entropy coef.; epochs; warm start | GRPO; 1e-6; 256; 256; 0.0; 0.0; 100; 10 step의 SFT checkpoint | arXiv:2510.00237v1 App. B.3 Table 5 | 2026-09-15 verified | App. B.3: base model은 supervised fine-tuning 없이 positive reward를 얻기 어려움 |
| Llama-3.2-Vision-11B (Chu et al.) | 11B | SFT | LR search (모든 parameter 학습); hardware | {1e-4, 1e-5, 1e-6, 5e-7, 1e-7}; 8×H800-80GB | arXiv:2501.17161v2 App. C.2, D.1 | 2026-09-15 verified | Fig. 16: 추가 SFT run 10개 중 어느 것도 RL의 증가 trend를 보이지 않음 |
| Llama-3.2-Vision-11B (Chu et al.) | 11B | RL | algorithm; LR search; verification steps; reward | PPO (multi-turn, sequential revision); {2e-6, 1e-6}; 5 (GeneralPoints); +5 / −1 / −1 / −2 / −3, GP-VL card misrecognition 시 추가 −1.5 | arXiv:2501.17161v2 §3, App. A.3, D.1 | 2026-09-15 verified | Fig. 10: verification iteration 1/3/5/10의 OOD gain = +0.48 / +2.15 / +2.99 / +5.99 |
| LLaMa 7B (Kirk et al., summarization) | 7B | preference | PPO batch; PPO epochs; PPO steps; minibatch; KL coef.; advantage normalization; frozen layers; LR (선택값) | 256; 4; 750; 256; 0.05; on; 80%; {1.5e-6, 3e-6, 6e-6, 1.5e-5, 3e-5} 중 1.5e-5 | arXiv:2310.06452v3 App. E.4 Table 4 | 2026-09-15 verified | in-distribution validation set의 reward로 선택(App. E.3) |
| LLaMa 7B (Kirk et al., summarization) | 7B | SFT, reward-model | batch; epochs; frozen layers; LR (선택값) | SFT 128, 1, 80%, 3e-5; RM 64, 1, 80%, 3e-5 | arXiv:2310.06452v3 App. E.4 Tables 2–3 | 2026-09-15 verified | validation loss(SFT)와 accuracy(RM)로 선택 |
| T5X Large policy (Tang et al.) | 770M | preference | steps; LR; β; batch (prompt, 각 한 pair) | 4k; 1e-5; 0.1; 32 | [[on-off-policy-rlhf]] Recipe ledger, arXiv:2405.08448v1 §4 | 2026-09-14 verified (card) | prior work에서 adapted, OpenAI summarization에서만 tuned |
| Nemotron-Research-Reasoning-Qwen-1.5B (ProRL) | 1.5B | RL | KL coefficient β; samples per prompt; rollout temperature; steps | not reported; 16 (run 6–7에서는 32); 1.2; “more than 2k” | [[prorl]] Recipe ledger, arXiv:2505.24864v1 §3.2 | β not reported; 나머지는 verified | KL term 또는 reset을 분리한 ablation 없음 |
| OLMo 2 (NoveltyBench evaluation) | 1B–32B | eval-gate | generations per prompt; temperature; patience p; utility model | 10; 1.0; 0.8; Skywork-Reward-Gemma-2-27B-v0.2를 {1..10}으로 mapping | arXiv:2504.05228v4 §3.2, §4.1 | 2026-09-15 verified | classifier agreement 79%, held-out pair 100개에서 F1 0.811 |
| 70+ models (Artificial Hivemind evaluation) | 여러 크기 | eval-gate | responses per prompt; decoding A; decoding B; max tokens; embedding | 50; top-p 0.9, T = 1.0; min-p 0.1, top-p 1.0, T = 2.0; 2048; text-embedding-3-small | arXiv:2510.22954v1 App. C.1 | 2026-09-15 verified | random-pair baseline은 similarity 0.1–0.2 band에서 100% 하락 |
| Base 및 RLVR model (Yue et al. evaluation) | 7B–32B | eval-gate | temperature; top-p; max tokens; n samples | 0.6; 0.95; 16,384; 128 (MATH500, Minerva, GSM8K) 또는 1024 (AMC23, AIME24) | arXiv:2504.13837v5 §3.1, App. A.2 | 2026-09-15 verified | Fig. 17 temperature study; App. C.8 entropy-matched control |

**작은 general-purpose run의 시작점.** 다른 능력을 잃지 않고 한 skill을 얻어야 하는 1.5B–8B instruction-tuned checkpoint에 대해, verified row는 다음을 뒷받침한다. binary verifier reward의 GRPO, group size 5, KL coefficient 0.05(또는 tested combination 중 하나를 제외하면 비슷한 trade-off를 보인 0), 7B–8B에는 LR 5e-6, 1B–1.5B에는 1e-4, warmup ratio 0.03의 cosine schedule, batch 128, 2 epoch다([[retaining-by-doing]] App. A.3). 이는 Llama 3.2 1B, Llama 3.1 8B, Qwen 2.5 1.5B, Qwen 2.5 7B에서 IFEval·MMLU·Countdown으로, 최대 8개의 H100 GPU에서 측정했다. rollout이 너무 비싸면 같은 논문의 Iterative-SFT 대안은 같은 optimizer setting에서 epoch 시작마다 target을 regenerate한다. 어느 쪽보다 먼저 input checkpoint에서 §9 report를 수행하여 delta의 reference를 둔다.

## 일반화 관점

**(a) breadth를 넓히는 것.** 업데이트하는 checkpoint 자체에서 training response를 sample하는 것: GRPO에서는 SFT가 38.5에 도달한 곳에서 non-target drop이 최대 3.4 point([[retaining-by-doing]] Table 1), Qwen3-14B에서는 non-reasoning average 53.2 대 21.1([[transferability-of-llm-reasoning-tables]] Table 1), 그리고 flat한 prior-task Pareto frontier([[rls-razor]] §3.1)가 근거다. 매 epoch target을 regenerate하는 방법은 mild하거나 forgetting 없이 SFT의 target accuracy에 도달했다([[retaining-by-doing]] §4.2). SFT data의 prompt diversity도 중요하다. Sokoban instruction variant는 `0 → 0.84–0.92`였다([[debunk-sft-generalization]] Table 1a). difficulty transfer에는 chain-of-thought target이 필요했다. General Points large-number split은 `0.02 → 0.80`이었다([[debunk-sft-generalization]] Table 1b). ability가 absent하면 off-policy teacher data가 도움이 된다. RLVR가 하지 못한 곳에서 distillation은 base model 바깥으로 coverage를 넓혔다([[rlvr-beyond-base-model-tables]] §5).

**(b) narrowing 또는 forgetting을 일으키는 것.** 높은 learning rate의 off-policy target(§2, control은 ch-30a §5). frozen prompt template은 instruction이 바뀌었을 때 training semantic을 실행하는 model을 만든다(Fake-environment success `0.90–0.93`, [[debunk-sft-generalization]] §5). diversity axis의 preference optimization과 RL도 원인이다. RLHF 뒤 input별 diversity([[rlhf-generalisation-diversity]] §6.2), 모든 OLMo 2 size의 `distinct₁₀`([[noveltybench]] §4.4), 큰 `k`에서 coverage([[rlvr-beyond-base-model-tables]] §3.2, Table 2)가 그렇다. KL coefficient를 높여도 diversity loss는 되돌아오지 않는다([[rlhf-generalisation-diversity]] §6.3). over-tuned SFT checkpoint에서는 RL이 out-of-distribution performance를 복구하지 못했다([[sft-memorizes-rl-generalizes]] §6).

**(c) 이 stage에서 측정하는 방법.** §9의 네 부분 report를 쓴다. input checkpoint 대비 paired held-out delta, 방향을 명시한 그 checkpoint에 대한 KL, 큰 `k`의 pass@1과 pass@k, fixed decoding configuration의 diversity metric이다. 알려진 measurement error는 다음과 같다. 큰 `k`의 pass@k는 chain을 검사하지 않으면 운 좋게 나온 final answer에 보상한다([[rlvr-beyond-base-model-tables]] §3.1). short output용 diversity metric은 긴 instruction-following response에서는 차이를 보이지 않았다([[rlhf-generalisation-diversity]] §6.2). embedding-similarity threshold는 embedding model 사이에서 비교할 수 없다([[artificial-hivemind]] App. C.1). KL은 LLM run에서 forgetting을 중간 정도로만 예측한다. [[rls-razor]]에서는 `R² = 0.71`, [[retaining-by-doing]]에서는 non-monotone case를 포함해 Pearson 0.52였다(App. A.5). update sparsity는 precision에 의존한다([[rls-razor]] §6). 이 장의 어떤 source도 long-context retention을 측정하지 않았다(not reported). 그 측정에는 ch-47 suite를 사용한다.

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 방법 |
|---|---|---|
| “SFT memorizes, RL generalizes”를 loss의 성질로 읽음 | on-policy SFT run이 RL만큼 retention을 보여 slogan과 모순 | 자기 data에서 §1의 four-cell ablation 수행([[rls-razor]] §5.1; [[transferability-of-llm-reasoning-tables]] Table 4) |
| frozen prompt template 하나로 SFT와 RL 비교 | SFT가 instruction variant에서는 거의 0점이지만 Fake environment에서는 ID score에 가까움 | Fake variant를 만들고 비교([[debunk-sft-generalization]] §5) |
| base model에 대한 KL을 objective로 취급 | KL은 낮은데 held-out slice는 계속 하락 | KL은 diagnostic으로 두고 paired held-out delta로 결정(ch-30a §1) |
| tuned model sample로 forward KL 추정 | logged number는 reverse KL이며 forgetting을 덜 잘 추적 | base model에서 sample하고 두 model로 score(§9) |
| pass@1만 보고 | pass@1은 오르지만 model이 푸는 distinct problem 수는 감소 | `n ≥ k`를 미리 고정하고 `k ≥ 128`의 pass@k 보고 |
| fixed temperature에서 coverage 비교 | RL model이 entropy가 낮아서만 더 나빠 보임 | temperature가 아니라 output entropy를 match([[rlvr-beyond-base-model-tables]] App. C.8) |
| 높은 diversity 자체를 승리로 취급 | 1B model이 `distinct_k`는 최고, `utility_k`는 최저 | `distinct_k`와 `utility_k`를 함께 보고([[noveltybench]] Table 1) |
| diversity를 회복하려고 β를 올림 | diversity와 reward가 모두 하락 | diversity를 직접 측정하고 ch-43 control 사용 |
| RL update sparsity를 retention 메커니즘으로 인용 | float32에서 효과가 사라짐 | precision을 명시하거나 주장 제거([[rls-razor]] §6) |
| offline contrastive loss를 쓰고 “winner에 대한 SFT”라고 부름 | training 중 chosen response의 log-probability가 하락 | chosen/rejected log-probability를 따로 log([[on-off-policy-rlhf]] Fig. 8) |
| RL이 좁아진 checkpoint를 나중에 고칠 수 있다고 가정 | over-tuned SFT checkpoint에서 시작한 RL이 training rule에 머무름 | RL 전에 §9 report로 SFT stage를 gate |

## 이해 점검

1. [[rls-razor]] §5.1에서 1–0 REINFORCE는 GRPO처럼 prior-task performance를 유지하지만 SimPO는 SFT처럼 forget한다. 이것이 negative gradient의 역할에 관해 무엇을 배제하는지 설명하고, 반대 결과가 성립하려면 무엇이 참이어야 하는지 설명하라.
2. §4의 worked example을 사용하여 rejection-sampling projection의 KL cost가 왜 `π₀(accepted)`에만 의존하는지 설명하라. model이 현재 sample 1000개 중 1개에서만 성공하는 task에 fine-tuning할 때 무엇을 예측하는가?
3. [[retaining-by-doing]]은 SFT가 Self-SFT보다 base model에 대한 KL이 더 크지만 forgetting은 더 작은 case를 보고한다. 그 관찰이 [[rls-razor]]의 `R² = 0.71` fit과 함께 성립할 수 있는 이유를 설명하고, 두 설명을 구분할 measurement 하나를 제시하라.
4. [[transferability-of-llm-reasoning-tables]] Table 4에서 off-policy SFT는 math average가 두 번째로 높고 non-reasoning transfer는 최악이다. §5의 token-level gradient argument를 써서 sampling distribution에서 그 pattern으로 이어지는 causal path를 설명하라.
5. 한 team이 RLVR model의 pass@1이 8 point 개선되었으므로 새 reasoning strategy를 배웠다고 결론낸다. 이 결론을 반박할 수 있는 measurement 두 개를 제시하고, 대신 결론을 뒷받침할 각 결과를 말하라.
6. penalty가 policy를 SFT model 쪽으로 끌어당기는데도 [[rlhf-generalisation-diversity]]에서 RLHF의 KL penalty coefficient를 올리자 input별 diversity가 복구되지 않고 낮아진 이유를 설명하라.
7. §7의 `utility_k` 정의를 사용하여, 같은 question에 정확히 같은 답을 열 번 내는 model이 average quality score는 높으면서 `utility₁₀`은 2보다 낮을 수 있는 이유와 그 failure를 가리는 `p` 값을 설명하라.
8. [[debunk-sft-generalization]]은 prompt diversity로 instruction-variant gap은 닫지만 difficulty-variant gap은 닫지 못하며, 후자에는 chain-of-thought target이 필요하다. 두 intervention이 서로 다른 failure를 해결하는 이유와 frozen-prompt hypothesis가 예측하는 intervention을 설명하라.

## 연결

- **이전(의존성):** ch-38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax. 이 장의 비교에서 켜고 끄는 PPO objective와 KL penalty를 제공한다.
- **이전(의존성):** ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control. §9에서 쓰는 forgetting report를 제공한다.
- **다음:** ch-39 — Offline Preference Optimization: DPO and Its Variants. §6은 reader가 offline DPO를 선택하기 전에 필요한 근거다.
- ch-00 — What General Capability Means and How It Is Measured.
- ch-20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage (capability를 더할 수 있는 off-policy case).
- ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate.
- ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation (실무의 Iterative-SFT).
- ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood.
- ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data.
- ch-37 — Policy-Gradient Foundations for Language Models (REINFORCE and baseline).
- ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO.
- ch-41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization.
- ch-42 — Reward Hacking and Judge Design.
- ch-43 — Entropy, Output Diversity, and KL Control in RL (§7의 control).
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.
- ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing (이 장에 의존).
- ch-46 — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention.
- ch-47 — Evaluation Harness and Suite Design for General Capability.
- ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing.

## 출처

- [[sft-memorizes-rl-generalizes]] — Chu et al. 2025 발췌: rule-variant와 visual-variant 수치, reward design, 두 조건(SFT 없이는 RL 불가, over-tuned checkpoint에서는 회복 불가).
- [[debunk-sft-generalization]] — Lin et al. 2025 발췌: frozen-prompt hypothesis, Fake-environment score, prompt diversity와 CoT의 Table 1, proximity-control appendix.
- [[rls-razor]] — Shenfeld et al. 2025 발췌: forgetting law와 fit, four-objective ablation, projection lemma, bfloat16 finding, hyperparameter sweep.
- [[retaining-by-doing]] — Chen et al. 2025/2026 발췌: gain-drop table, AlpacaEval table, KL ablation, Iterative-SFT, mixture account, KL counter-evidence, setting.
- [[rlhf-generalisation-diversity]] — Kirk et al. 2024 발췌: generalisation gap 정의, input별·input 간 diversity, KL-coefficient sweep, PPO setting.
- [[noveltybench]] — Zhang et al. 2025 발췌: `distinct_k`와 `utility_k` 정의, model table, OLMo 2 stage value, prompting workaround.
- [[artificial-hivemind]] — Jiang et al. 2025 발췌: 두 decoding setting에서 intra-model repetition, inter-model similarity range, measurement configuration.
- [[on-off-policy-rlhf]] — Tang et al. 2024의 library card(개요의 `online-offline-alignment-gap` slug): KL-budget comparison과 four hypothesis test; Recipe row.
- [[on-policy-suboptimal-preference-data]] — Tajwar et al. 2024 발췌: on-policy sampling과 negative gradient의 조건, sample reuse, mode-seeking unification.
- [[rl-finetunes-small-subnetworks]] — Mukherjee et al. 2025 발췌: update-sparsity table, subnetwork-only training, in-distribution explanation, precision caveat.
- [[transferability-of-llm-reasoning]] — Huan et al. 2025 library card(description only; 수치 없음, 2026-09-15 기준 unverified).
- [[transferability-of-llm-reasoning-tables]] — verified Table 1·Table 4, TI formula, KL과 rank-shift diagnostic, setting을 담은 발췌.
- [[rlvr-beyond-base-model]] — Yue et al. 2025 library card(description only; 수치 없음, 2026-09-15 기준 unverified).
- [[rlvr-beyond-base-model-tables]] — verified pass@k estimator, coverage table, evaluation protocol, entropy-matched control을 담은 발췌.
- [[prorl]] — verified library card: pass@k expansion의 counter-evidence, Diminish regime, recipe ledger.
- [[wolfe-online-vs-offline-rl]] — secondary review 발췌: online / semi-online / offline axis와 sync period `s`. 수치에는 사용하지 않음.
- [[john-schulman-kl-tricks]] — §9에서 참조하는 k1과 k3 KL estimator.
