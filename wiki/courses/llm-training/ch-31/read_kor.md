<!-- chapter: ch-31
     track: sft
     kind: content
     title: Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation
     deps: [ch-30c]
     sources: [[llama-2]], [[llama-2-recipe]], [[llama-3]], [[llama-3-recipe]], [[rejection-sampling-finetuning]], [[rft-scaling-relationship-math]], [[rest-em]], [[star]], [[v-star]], [[raft-reinforce-rej-minimalist]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[glm-4-5]], [[glm-5]], [[swe-gym]], [[kimi-dev]], [[rlvr-beyond-base-model]], [[prorl]]
     figures: figures/iterative-loop.html
     revised: 2026-09 (generality revision)
-->

# Chapter 31 — Rejection Sampling, 자기 생성 데이터, Cold Start, SFT–RL 교대

> **핵심 통찰.** 첫 gradient step에서는, 정확성 검사를 통과한 자기 출력으로 모델을 fine-tuning하는 일이 \(\{0,1\}\) 보상과 baseline이 없는 policy-gradient update가 된다. 채택된 sample의 확률은 올라가고 거절된 sample의 weight는 0이 된다([[star]] Eq. 2; [[rest-em]] Algorithm 1). 채택 집합에 서로 다른 해법이 많을 때 held-out accuracy가 개선됐다(LLaMA-7B의 GSM8K: SFT 후 35.9, 질문당 100개 sample로 41.7, 네 모델의 sample을 pool하면 49.3; [[rft-scaling-relationship-math]] Tables 1, 3). 또한 매 round마다 base model에서 다시 시작했을 때 HumanEval로 더 잘 전이됐다([[rest-em]] Fig. 7). 작은 문제 집합(ReST-EM의 iteration 2 APPS), 매 round에서 직전 round sample만 쓴 경우(Llama 2 RLHF-V3), sampler가 서로 다른 해법을 적게 만든 경우(LLaMA-33B: 향상 없음), 자기 성공 trajectory와 teacher trajectory로 fine-tuning한 32B coding agent의 경우(SWE-bench Lite 15.3% → 8.7%; [[swe-gym]] §4.2)에는 성능이 퇴행하거나 더 이상 도움이 되지 않았다. positive만으로 학습하면 GRPO(ch-40)보다 policy entropy가 더 빨리 낮아지고 더 일찍 plateau에 도달했다([[raft-reinforce-rej-minimalist]] §5.1).
>
> **지침.** task에 자동 정확성 검사와 고정 문제 집합이 있다면, base model에서 다시 시작하거나 이전 모든 round의 채택 sample을 pool하는 rejection-sampling round를 실행하고, 문제별 채택 sample 수를 cap하며, 매 round 뒤 held-out task를 평가하라. 다시 시작하는 편이 계속 이어서 학습하는 것보다 전이가 좋았고([[rest-em]] Fig. 7), pooling은 Llama 2 RLHF-V3 퇴행에 대해 보고된 해결책이었으며([[llama-2]] §3.2.3), ReST-EM의 iteration 2 APPS 및 HumanEval 퇴행은 restart와 cap에도 불구하고 생겼고 held-out evaluation에서만 보였기 때문이다([[rest-em]] §5.1, Fig. 4). 문제별 서로 다른 해법 수가 더 이상 늘지 않거나 policy entropy가 round마다 하락하면, 실패 sample도 이용하는 objective(group-baseline RL update 또는 실패로 학습한 verifier)로 옮겨가라. RAFT++는 낮은 entropy에서 plateau했지만 GRPO는 계속 개선됐기 때문이다([[raft-reinforce-rej-minimalist]] Figs. 2–3; Qwen2.5-Math-7B에서는 최종 평균 56.1 대 56.3, LLaMA-3.2-3B-instruct에서는 27.6 대 28.4, Table 1). Base model에서 RL만 했을 때 읽기 어렵거나 언어가 섞인 출력을 만들면, RL 전에 작은 cold-start SFT set을 추가하고 모든 stage checkpoint를 전체 capability panel에서 평가하라. DeepSeek-R1에서 같은 base의 cold-start SFT model은 IF-Eval 71.7점으로 RL-only model의 46.6보다 높았지만 AIME 2024에서는 59.0 대 77.9였고, 완성된 pipeline은 AIME 2024에서 RL-only model보다 높았으나(79.8), GPQA Diamond(71.5 대 75.8)와 CNMO 2024(78.8 대 88.1)에서는 낮았기 때문이다([[deepseek-r1]] Table 3). 시작 policy가 해결하는 task가 적은 long-horizon agent에는 on-policy rejection-sampling SFT보다 강한 teacher trajectory를 사용하라. [[swe-gym]]은 491개 teacher trajectory로 SWE-bench Lite에서 +12.3점을 측정했고, on-policy trajectory를 더하자 15.3%에서 8.7%로 하락했기 때문이다. 저자들은 PPO-style optimization을 검증하지 않은 대안으로 언급한다(§4.2).

## 범용 모델에 이 장이 중요한 이유

사람이 작성한 target은 비싸며, GSM8K 같은 데이터셋은 질문마다 하나의 reasoning path만 제공한다. RFT 저자들은 이것이 대안적인 해법 순서를 학습하는 데 제약이라고 지적한다([[rft-scaling-relationship-math]] §3.3). Llama 2는 27,540개 예시에서 SFT annotation 수집을 멈추고 annotation 노력을 preference comparison으로 옮겼다([[llama-2]] §3.1). ReST-EM은 인간 데이터로 fine-tuning하는 일이 그 데이터의 양과 다양성에 제한된다고 말한다([[rest-em]] Abstract). 자기 생성 데이터는 사람이 작성한 target을 scorer가 채택한 model sample로 대체한다. scorer는 sample에 label을 매기는 모든 함수다. answer matcher, unit test, reward model(RM), LLM judge가 이에 해당한다.

이 장은 SFT(ch-30)와 RL(ch-37 이후) 사이에 놓이지만, 자기 생성 데이터는 현재 pipeline의 여러 지점에서 나타난다. iterative RLHF 안의 SFT round([[llama-2]], [[llama-3]]), RL 전 cold-start set([[deepseek-r1]], [[glm-4-5]]), RL 후 rejection-sampling SFT stage([[deepseek-r1]] stage 3), RL expert에서 다시 한 model로 하는 distillation([[glm-4-5]], [[glm-5]]), agent trajectory([[swe-gym]], [[kimi-dev]])로 쓰인다.

핵심 질문은 model의 자기 필터링 sample로 학습하는 일이 언제 일반화를 개선하고 언제 과적합하는가이다. 이 장은 course standard의 세 하위 질문으로 답한다. 채택 sample이 다양하고, round가 이전 분포로 restart하거나 pool하며, non-target data를 섞을 때 breadth가 넓어진다. 작은 문제 집합의 train–test gap, entropy collapse, scorer가 측정하지 않는 skill의 forgetting, format overfitting이 narrowing으로 나타난다. Generality는 매 round held-out suite, 큰 \(k\)에서의 pass@k, 문제당 서로 다른 해법 수, non-target capability panel로 측정한다.

## §1 Rejection-sampling fine-tuning: loop, variant, round 구조

### 1.1 정의와 메커니즘

Rejection-sampling fine-tuning(이 장에서는 RSFT)은 model에서 sample한 출력 중 scorer가 채택한 것만 남겨 supervised fine-tuning하는 일이다. 같은 절차는 [[llama-2]] §3.2.3에서 “Rejection Sampling fine-tuning”, [[rft-scaling-relationship-math]]에서 RFT, [[raft-reinforce-rej-minimalist]] §3에서 RAFT(reward-ranked fine-tuning), [[rest-em]] §3에서 ReST-EM의 Generate/Improve 단계로 불린다. Library card [[rejection-sampling-finetuning]]은 숫자 없이 이 pattern을 정의한다.

1. \(x\)가 prompt이고 \(y\)가 sample한 output인 scorer \(r(x,y)\)가 있는 prompt set을 선택한다.
2. 현재 model, 이전 checkpoint, 또는 여러 model로 이루어진 sampler에서 prompt당 \(K\)개의 output을 sample한다.
3. 모든 output을 score한다. \(r=1\)인 output 또는 prompt별 최고 score output을 남긴다.
4. 선택 사항으로 남긴 output을 deduplicate하고 prompt마다 남길 개수를 cap한다.
5. 남긴 set으로 시작 checkpoint를 output token에만 적용하는 next-token loss로 fine-tuning한다.
6. held-out task에서 평가한다. 새 model을 다음 round의 sampler로 쓴다.

출처들은 네 선택에서 다르다. sampler, selection rule, 다음 round가 fine-tune할 checkpoint, 채택 sample을 round 간 pool하는지 여부다.

| 방법 | Sampler와 sample | 선택 | fine-tune 시작점 | round 간 pooling | 위치 |
|---|---|---|---|---|---|
| STaR (GPT-J 6B) | current model, greedy, 문제당 1개, rationalization pass 추가 | 올바른 최종 답 | 매 iteration 원래 pretrained model | 아니오 | [[star]] §3.1, Alg. 1 |
| RFT (LLaMA 7B–33B) | SFT model, \(k=100\), temperature 0.7 | 올바른 답과 계산; 서로 다른 equation list당 한 path | pretrained model, human data와 채택 path 위에서 | 한 round; U13B / U33B에서 model 간 pool | [[rft-scaling-relationship-math]] §3.3, App. A.3 |
| ReST-EM (PaLM 2) | 이전 iterate; 32(MATH) 또는 64(APPS); top-K 40, temperature 0.7 | binary reward; 문제당 최대 10개 | 매 iteration base pretrained model | 아니오 | [[rest-em]] §3, §5 |
| Llama 2-Chat RLHF-V1…V5 | 최신 70B model; \(K\)는 출력되지 않음 | prompt별 최고 RM reward | 명확히 서술되지 않음(Fig. 8 temperature 연구 문장은 “각 새 RLHF version에서 항상 base model에서 시작”이라고 말함) | V3까지: 직전 iteration만; 이후: 이전 모든 iteration의 top sample | [[llama-2]] §3.2.3 |
| Llama 3.1 | 이전 round의 최고 checkpoint; \(K\)는 보통 10–30 | RM이 최고를 선택 | 매 round pre-trained model, synthetic 및 human data와 혼합 | 명시되지 않음. 매 round 새 SFT data를 수집하고 mix를 조정하며, 최종 mix는 일부 source를 여러 번 epoch함 | [[llama-3]] §4.1, §4.1.6, §4.2.2 |
| V-STaR (LLaMA 2 7B/13B) | 이전 generator; iteration당 \(k=16\) | 올바른 답 또는 통과한 test | 매 iteration pretrained base | 예: 모든 올바른 sample이 축적됨 | [[v-star]] Alg. 1 |
| DeepSeek-R1 stage 3 | reasoning-RL checkpoint(Dev2) | 올바른 것만; mixed-language, long-paragraph, code-block CoT 제거 | DeepSeek-V3-Base | 한 round | [[deepseek-r1]] B.3.3, Fig. 2 |

### 1.2 Worked example: 어떤 문제가 훈련 데이터를 만드는가

pass@K는 독립적인 \(K\)개 sample 중 적어도 하나가 맞을 확률이다. 한 문제의 sample당 성공 확률이 \(p\)이면 pass@K = \(1 − (1 − p)^K\)다. \(K=32\)(ReST-EM의 MATH 설정)에서 \(p=0.1\)이면 \(1 − 0.9^{32} = 1 − 0.034 = 0.966\)이고, \(p=0.01\)이면 \(1 − 0.99^{32} = 1 − 0.725 = 0.275\)다. \(p=0.01\)인 문제는 round의 72.5%에서 데이터를 전혀 제공하지 않는다. 채택 sample의 기대수는 \(K·p\)다. \(p=0.1\)인 어려운 문제에서는 3.2개, \(p=0.8\)인 쉬운 문제에서는 25.6개다. cap이 없으면 쉬운 문제는 어려운 문제보다 \(25.6 / 3.2 = 8\)배 많은 target을 제공한다. ReST-EM의 문제당 10개 cap을 적용하면 쉬운 문제는 약 10개를 남긴다(\(p=0.8\)에서 32회 추출 중 성공이 10개 미만일 확률은 \(10^{-6}\)보다 작다). 어려운 문제는 약 3.2개를 남기므로, 비율은 약 \(10 / 3.2 \approx 3.1\)로 내려간다. ReST-EM은 이 cap이 쉬운 문제의 과대표집을 제한하기 위해 존재한다고 말한다([[rest-em]] §5). SWE-Gym은 instance당 2개 cap이 모든 성공 trajectory를 쓰는 것보다 약간 나았다고 발견했다([[swe-gym]] §4.3, Table 6). 따라서 채택 set은 model이 이미 푸는 문제 쪽으로 편향된다. cap이 없으면 두 문제 사이 채택 sample 기대수의 비율은 모든 \(K\)에서 그 성공 확률의 비율과 같다. Companion figure의 acceptance panel([figures/iterative-loop.html#acceptance](figures/iterative-loop.html#acceptance))은 reader가 \(p\), \(K\), cap을 설정하고 기대 채택 수와 pass@K를 읽게 한다.

### 1.3 Llama 2: iterative RLHF 안의 rejection sampling

Llama 2는 새 preference batch가 도착함에 따라 다섯 RLHF version을 학습했고 두 algorithm을 사용했다([[llama-2]] §3.2.3). 두 번째 algorithm은 PPO(proximal policy optimization, ch-38)다. 보고서는 schedule을 한 문장으로 기술한다. “RLHF (V4)까지는 Rejection Sampling fine-tuning만 사용했고, 그 후에는 두 방법을 순차적으로 결합하여, 다시 sampling하기 전에 생성된 Rejection Sampling checkpoint 위에 PPO를 적용했다.” 이 문장은 V4 자체가 PPO를 사용했는지는 말하지 않는다([[llama-2-recipe]]). 보고서는 두 방법을 breadth(rejection sampling은 prompt당 \(K\) sample, PPO는 하나)와 depth(PPO는 이전 step에서 update된 policy로부터 sample하고, rejection sampling은 round 초기 policy의 모든 output에서 sample함)로 대비하며, iterative update는 그 차이를 “less pronounced”하게 만든다고 말한다(§3.2.3).

일반성에 중요한 세 세부사항이 있다. 첫째, rejection sampling은 70B model에서만 실행됐으며, 더 작은 모든 chat model은 70B sample로 fine-tuning됐다. 저자들은 이를 distillation이라 부르며 분석하지 않는다(§3.2.3). 둘째, 최적 sampling temperature는 version마다 달랐고, RLHF model에서는 10–100개 sample을 뽑을 때 \(T \in [1.2,1.3]\)이었다(Fig. 8). 셋째, selection으로 인한 gain은 \(N\)개 sample 중 최대 reward와 median reward의 gap이며, median은 평평하게 유지되는 동안 이 gap은 \(N\)과 함께 커진다(Fig. 7). PPO 설정은 7B와 13B의 KL-penalty coefficient \(\beta=0.01\), 34B와 70B의 \(\beta=0.005\)다(\(\beta\)는 원래 policy에서의 KL divergence에 곱해져 reward에서 빼진다). constant learning rate \(10^{-6}\), batch 512, mini-batch 64, clip 0.2, held-out prompt에서 early stopping하는 200–400 iteration을 사용했다(§3.2.3). Human evaluation은 coding 또는 reasoning prompt가 없는 약 4,000개 prompt를 사용했고 각 conversation의 마지막 turn만 평가했다(§3.4.2). 따라서 이 human evaluation은 coding이나 reasoning을 측정하지 않았다.

### 1.4 RFT: 서로 다른 해법 수가 held-out gain을 이끈다

Yuan et al.은 LLaMA와 LLaMA 2 model을 GSM8K 및 self-sample한 올바른 reasoning path로 fine-tuning하고, 서로 다른 equation list마다 한 path를 남겼다([[rft-scaling-relationship-math]] §3.3, App. A.3). Accuracy는 training distribution에서 왔지만 held out인 GSM8K test split의 greedy-decoding accuracy(maj1@1)다.

| Base model (Table 1) | SFT | RFT, \(k=100\) | 질문당 올바른 path | 질문당 서로 다른 path |
|---|---|---|---|---|
| LLaMA-7B | 35.9 | 41.7 | 53.3 | 5.25 |
| LLaMA 2-7B | 41.6 | 47.5 | 60.8 | 5.19 |
| LLaMA-13B | 43.0 | 49.1 | 62.5 | 5.26 |
| LLaMA-33B | 54.6 | 54.5 | 88.7 | 2.78 |

33B SFT model은 질문당 가장 많은 올바른 path를 냈지만 서로 다른 path는 가장 적었고, RFT는 이를 개선하지 못했다. 저자들은 33B SFT model이 training question에 fit했기 때문이라고 본다(§3.3). 33B model을 temperature 1.0으로 sample해도 서로 다른 path는 4.77까지만 증가했다(§3.3). 서로 다른 path는 \(k\)에 따라 천천히 증가한다. LLaMA-7B에서는 \(k=1\)에 1.17, \(k=12\)에 2.20, \(k=100\)에 5.25이며(Table 2), \(k\)를 두 배로 늘릴 때 accuracy gain은 감소한다(Fig. 4). 네 model에서 path를 pool하면(U13B) 질문당 서로 다른 path가 12.84가 되며, LLaMA-7B는 49.3, LLaMA 2-7B는 50.3, LLaMA-13B는 52.1, LLaMA 2-13B는 55.4로 올라간다(Tables 2–3). \(k=100\)에서 deduplication 없이 학습한 accuracy는 deduplicate한 학습과 비슷했고, 4개 model 중 3개에서는 deduplication이 더 좋았다(§3.3, Table 5). 상태: 결과(단일 연구). 측정은 in-distribution이며, 논문은 out-of-domain task를 보고하지 않는다.

### 1.5 ReST-EM: expectation–maximization 관점, base에서 restart, iteration

ReST-EM은 binary optimality variable \(O\)가 있는 RL의 expectation–maximization(EM)으로 loop를 유도한다. 여기서 \(p(O=1 \mid x,y) \propto f(r(x,y))\)이고, \(f\)는 reward \(r\)의 non-decreasing, non-negative 함수다([[rest-em]] §3). E-step(Generate)은 current policy에서 sample하고 sample을 score한다. M-step(Improve)은 다음을 최대화한다.

\[
J(\theta) = E_{(x,y)\sim D_i}\,[r(x,y) \cdot \log p_\theta(y \mid x)]
\]

여기서 \(\theta\)는 model parameter, \(D_i\)는 iteration \(i\)에서 생성된 sample set, \(r(x,y) \in \{0,1\}\)은 binary reward, \(p_\theta(y \mid x)\)는 input \(x\)가 주어졌을 때 \(y\)를 출력할 model 확률이다(Algorithm 1). 각 Improve step은 이전 iterate가 아니라 base pretrained model을 fine-tune하며, 이는 “task-specific over-fitting을 완화하기 위해서”다(§3). 이전 iterate에서 계속하는 것과 비교하면 task 성능은 비슷했고 held-out task로의 transfer는 더 좋았다(PaLM 2-S*, APPS → HumanEval; Fig. 7). 상태: 결과(단일 연구).

MATH(7,500개 training problem)에서 PaLM 2-L은 세 iteration 동안 개선됐고, 첫 iteration 뒤 test gain은 작았지만 training accuracy는 계속 올랐다(§5.1, Fig. 4). 문제당 sample을 세 배로 한 한 iteration은 pass@1 40.3%에 도달했는데, iteration 2의 41.0%와 iteration 3의 41.9%보다 낮다(§5.3). 같은 generation 수(iteration 3과 단일 3× iteration의 비교)에서, 이 단일 실행에서는 round당 sample 수가 더 적은 더 많은 round가 더 높은 점수를 냈다. Transfer로 MATH-trained model은 iteration을 거치며 GSM8K를 개선했다(Fig. 2). 2023 Hungarian high-school finals exam에서는 GPT-4를 제외한 모든 비교 model보다 높은 점수를 냈지만, 다른 model의 점수는 Paster (2023)에서 가져왔고 base PaLM 2-L 점수는 보이지 않는다(§5.4, Fig. 10). MATH- 및 APPS-trained PaLM 2-L은 Big-Bench Hard에서 “no major degradation”을 보였다(§5.4, Fig. 9). ReST-EM은 pass@K의 모든 \(K\)에서 base model보다 강했고, gap은 “typically” \(K=1\)에서 가장 컸다. 저자들은 큰 \(K\)에서 pass@K와의 gap을 닫지 못할 수 있다고 말한다(§5.2, §6).

## §2 \(\{0,1\}\) reward와 zero baseline을 둔 REINFORCE로서의 RSFT, RAFT, ReST-EM

REINFORCE는 sample의 log-probability gradient에 그 sample의 reward에서 baseline을 뺀 값을 곱하는 policy-gradient estimator다(ch-37에서 유도한다). 각각 \(n\)개 sample \(y_{ij}\)을 가진 \(M\)개 prompt \(x_i\)에 대해,

\[
\hat g = (1/(M\cdot n)) \cdot \sum_i \sum_j (r_{ij}-b_i) \cdot \nabla_\theta \log \pi_\theta(y_{ij}\mid x_i)
\]

여기서 \(\pi_\theta\)는 학습 중인 policy, \(r_{ij}\)는 prompt \(i\)의 sample \(j\)의 reward, \(b_i\)는 \(y_{ij}\)에 의존하지 않는 baseline이며, \(\hat g\)는 expected reward gradient의 추정량이다. 채택 집합 \(A=\{(i,j):r_{ij}=1\}\) 위의 RSFT loss는 다음과 같다.

\[
L_{\mathrm{RSFT}}(\theta) = -(1/|A|)\cdot\sum_{(i,j)\in A}\log\pi_\theta(y_{ij}\mid x_i)
\]

여기서 \(|A|\)는 채택 sample 수다. \(\hat g\)에서 \(r_{ij}\in\{0,1\}\), \(b_i=0\)으로 설정하면 거절 sample은 모두 합에서 빠진다. 따라서

\[
-\nabla_\theta L_{\mathrm{RSFT}} = (M\cdot n/|A|)\cdot\hat g=(1/a)\cdot\hat g,\qquad \text{채택률 } a=|A|/(M\cdot n).
\]

RSFT gradient step은 baseline이 없고 learning rate가 \(1/a\)만큼 scale된 REINFORCE step이다. STaR는 indicator reward로 같은 항등식을 \(\nabla J=\sum_i E[\mathbf{1}(\hat y_i=y_i)\cdot\nabla\log p_M(\hat y_i,\hat r_i\mid x_i)]\)라고 쓰며, indicator를 filtering step이라고 부른다([[star]] §3.1, Eq. 2). ReST-EM의 M-step objective도 \(r\in\{0,1\}\)인 같은 식이다([[rest-em]] Algorithm 1, §3 Remark). Llama 2처럼 RM score로 top-1 sample을 선택하는 것은 prompt마다 최고 score sample에만 \(r_{ij}=1\)을 주는 동일한 형식이다.

**조건.** 이 항등식은 parameter가 sampling policy와 같은 첫 gradient step에서 성립한다. 그 후 RSFT는 고정된 set에서 많은 step과 epoch를 실행하므로, 이후 step은 off-policy sample을 사용한다. RAFT++는 최고 reward response만 남기면서, RAFT loss에 token별 importance ratio \(s_t(\theta)=\pi_\theta(a_t\mid x,a_{<t})/\pi_{\theta\mathrm{old}}(a_t\mid x,a_{<t})\)와 PPO clipping을 더한다([[raft-reinforce-rej-minimalist]] Eq. 6). ratio는 쓰되 clipping은 하지 않는 중간 variant는 vanilla RAFT보다 성능이 낮았다(§5, Fig. 2).

**Worked example.** 각각 \(n=4\)개 sample을 가진 세 prompt의 reward가 \(A=[1,0,1,0]\), \(B=[1,1,1,1]\), \(C=[0,0,0,0]\)이라고 하자. 12개 sample 중 6개가 채택되므로 \(a=0.5\)다.

| Update rule | A의 sample별 weight | B | C | B가 제공하는 positive weight 비율 |
|---|---|---|---|---|
| RSFT, 채택 sample 위 평균 | +1/6, 0, +1/6, 0 | 각각 +1/6 | 0 | 4/6 = 67% |
| REINFORCE, \(r\in\{0,1\}\), \(b=0\) | +1/12, 0, +1/12, 0 | 각각 +1/12 | 0 | 67% |
| REINFORCE, \(r\in\{-1,+1\}\), \(b=0\) | ±1/12 | 각각 +1/12 | 각각 −1/12 | 67% |
| Group-mean baseline, \(b_i=\operatorname{mean}(r_i)\) | +0.5/12, −0.5/12, +0.5/12, −0.5/12 | 0 | 0 | 0% (B 제외) |

첫 두 행은 \(1/a=2\)라는 factor만 다르다. 두 경우 모두 전부 정답인 prompt B가 positive weight의 3분의 2를 공급하며, 이는 §1.2의 쉬운 문제 편향을 gradient share로 나타낸 것이다. ±1 reward에 baseline이 없으면, 전부 오답인 prompt C는 네 sample 모두의 확률을 낮춘다. GRPO(group relative policy optimization)가 group standard deviation으로 나누기 전에 적용하는 group-mean baseline은 전부 정답 및 전부 오답 prompt 모두에 0 weight를 준다. [[raft-reinforce-rej-minimalist]]는 vanilla REINFORCE에 대한 GRPO의 이점이 reward normalization보다는 이 암묵적인 전부 오답 prompt 제거에서 주로 나온다고 보고한다. 또한 두 종류의 prompt를 모두 제거한 “Reinforce-Rej”는 initial policy에서의 KL divergence 단위당 reward gain이 더 크다는 뜻의 더 나은 KL efficiency로 GRPO와 일치했다(LLaMA-3.2-3B-instruct; §5.1, Fig. 4). Companion figure의 weight panel([figures/iterative-loop.html#weights](figures/iterative-loop.html#weights))은 임의 reward pattern에 대해 이 weight를 계산하고, prompt별 cap이 share를 어떻게 바꾸는지 보인다.

**근거.** Numina-Math prompt로 학습한 Qwen2.5-Math-7B-base에서(MATH500, Minerva Math, OlympiadBench에 대해 temperature 1.0에서 측정한 average@16 accuracy), RAFT는 52.3, RAFT++는 56.1, GRPO는 56.3, PPO는 52.5, iterative DPO는 48.8(새로 sample한 pair에 반복 적용한 direct preference optimization, ch-39), Reinforce-Rej는 56.4에 도달했다. LLaMA-3.2-3B-instruct에서는 RAFT 25.9, RAFT++ 27.6, GRPO 28.4, Reinforce 24.2였다([[raft-reinforce-rej-minimalist]] Table 1). 저자들은 algorithm마다 batch size, mini-batch size, learning rate를 tuning했다(Table 1 caption). 상태: 결과(단일 연구; 수학만; 약 250–300 training step).

**함의.** positive-only self-training에는 특정 오답을 겨냥하는 항이 없다. model이 현재 선호하는 오답은 채택 sample의 확률이 올라갈 때 softmax normalization을 통해 간접적으로만 확률이 낮아진다. 채택 sample이 없는 prompt는 update를 전혀 만들지 않는다. 쉬운 prompt에서 나온 모든 채택 sample은 어려운 prompt의 sample과 gradient share를 두고 경쟁한다(위 항등식에서 도출).

## §3 STaR rationalization과 V-STaR: 실패한 문제와 실패 sample의 재사용

### 3.1 STaR: rationalization은 실패한 문제를 재사용하며, 실패한 rationale은 재사용하지 않는다

STaR(Self-Taught Reasoner)는 few-shot prompt로 모든 training problem에 대한 rationale과 answer를 생성하고, answer가 맞는 rationale을 남겨 매 outer loop마다 original pretrained model을 그것들로 fine-tuning한다([[star]] §3.1, Alg. 1). Rationalization은 측정 가능한 gap을 다룬다. 이것이 없으면, loop는 “해결하지 못한 문제에 대한 직접 training signal을 받지 못하므로 training set에서 어떤 새 문제도 풀지 못한다”(§1). Model이 실패한 문제에 대해 STaR는 gold answer를 hint로 삽입하고 새 rationale을 sample하며, 그것이 gold answer에 도달할 때만 남기고, hint를 제거한 상태로 그 rationale을 학습한다(§3.2). 실패한 rationale은 여전히 gradient가 0이며, Algorithm 1에는 sample의 likelihood를 명시적으로 낮추는 항이 없다.

GPT-J 6B의 CommonsenseQA dev accuracy에서 direct answer fine-tuning은 60.0, rationalization 없는 STaR는 68.8, rationalization을 둔 STaR는 72.5였으며, 30배 더 큰 fine-tuned GPT-3는 73.0이었다(Table 1). GSM8K test에서는 rationalization이 accuracy를 10.1에서 10.7로 옮겼으며, 저자들은 이를 substantial improvement가 아니라고 설명한다(Table 2, §4.5). STaR는 greedy decoding을 사용한다. temperature 0.5 또는 0.7에서 sample하면 더 나쁜 model이 나왔고, 더 높은 temperature는 틀린 reasoning으로 맞는 answer에 도달하는 비율을 올렸는데 저자들은 이것이 일반화를 막는다고 말한다(§5). few-shot accuracy는 chance보다 높아야 한다. GPT-2는 arithmetic에서 bootstrap하지 못했다(§6). ReST-EM은 STaR-style rationalization이 preliminary experiment에서 그러한 false positive를 늘렸다고 보고했고 사용하지 않았다([[rest-em]] §4). 상태: gain에 대해 결과(단일 연구). 서로 다른 조건에서 두 source가 correct-answer, wrong-reasoning sample이 더 많아진다고 보고한다. [[star]] §5의 높은 sampling temperature와 [[rest-em]] §4의 rationalization이다(정성적; 비율은 보고되지 않음).

### 3.2 V-STaR: 실패 sample이 verifier를 학습시킨다

V-STaR(Hosseini et al., COLM 2024)은 STaR generator loop를 유지하면서, 올바른 자기 생성 해법과 올바르지 않은 자기 생성 해법 모두로 학습하는 verifier를 추가한다([[v-star]] §3). \(T\)개 iteration 각각에서 generator는 원래 SFT data와 지금까지의 모든 correct sample인 \(D_{\mathrm{GEN}}\) 위에서 pretrained base부터 fine-tuning한다. Training problem당 \(k\)개 sample을 answer match 또는 test execution으로 label한다. Correct sample은 \(D_{\mathrm{GEN}}\)에 들어가며 모든 sample은 \(D_{\mathrm{VER}}\)에 들어간다(Alg. 1). Verifier는 문제별 correct solution과 incorrect solution의 Cartesian product에서 나온 pair로 SFT model을 기준으로 한 DPO로 학습한다.

\[
L_{\mathrm{DPO}}(V;G_{\mathrm{SFT}})=-E_{(x,y^+,y^-)\sim D_{\mathrm{VER}}}\log\sigma\big(\hat r(x,y^+)-\hat r(x,y^-)\big),\qquad
\hat r(x,y)=\beta\cdot\log\left(\frac{V(y\mid x)}{G_{\mathrm{SFT}}(y\mid x)}\right)
\]

여기서 \(V\)는 verifier, \(G_{\mathrm{SFT}}\)는 reference로 쓰는 SFT generator, \(y^+\)는 올바른 해법, \(y^-\)는 올바르지 않은 해법, \(\sigma\)는 logistic function, \(\beta\)는 proximity coefficient다(§3.1, Eq. 2). Test 시에는 verifier likelihood \(V(\hat y\mid x)\)로 candidate solution의 순위를 매긴다(§3.1).

LoRA(low-rank adapter fine-tuning)로 학습한 LLaMA 2와 CodeLLaMA 7B/13B에서, \(k=16\)인 3개 iteration의 V-STaR는 이전 self-improvement 및 verification method보다 math test accuracy를 6–17점, code test accuracy를 4–12점 개선했다(§1, §4.3). Transfer는 MATH의 150문제 level-1 subset과 HumanEval에서 평가했으며, 이들에서도 V-STaR는 baseline을 앞섰다(§4.3, footnote 2). 네 번째 MBPP iteration은 0.3점을 더했다(§4.3). Verifier를 training loop 안에 배치해 generator data를 고르게 하면, 저자들의 표현으로 MBPP에서 “substantial gain을 제공하지 않았다”(64개 sample 중 verifier의 top-ranked candidate accuracy인 Best-of-64는 53.2, Pass@1은 46.34; §4.7). DPO verifier 자체의 generation accuracy는 몇 번의 training update 후 저하됐지만 ranking accuracy는 상승했다(§4.6, Fig. 6). 상태: 결과(단일 연구; small-model LoRA; math와 code만).

## §4 Self-training이 과적합하거나 좁아진 사례

| 사례 | 설정 | 측정된 증상 | 보고된 원인 | 보고된 해결책 | 위치 |
|---|---|---|---|---|---|
| APPS의 ReST-EM | PaLM 2, 2,342개 문제 | gain 대부분이 iteration 1에 발생; iteration 2에서 APPS와 HumanEval이 퇴행; training accuracy는 계속 상승 | 작은 문제 집합에서의 과적합(APPS는 MATH 크기의 약 3분의 1); base에서 restart하고 10개 cap을 둔 상태에서도 발생 | 보고된 해결책 없음; APPS gain 대부분은 iteration 1에서 발생 | [[rest-em]] §5.1, Fig. 4 |
| Llama 2 RLHF-V3 | V2 sample만으로 한 70B rejection sampling | 시에서 운율이 약해짐(정성적) | forgetting | 이전 모든 iteration의 top sample을 pool; “considerable enhancements”, figure 없음 | [[llama-2]] §3.2.3 |
| LLaMA-33B의 RFT | GSM8K, \(k=100\) | SFT 54.6 → RFT 54.5; 질문당 서로 다른 path 2.78개 | SFT model이 training question에 fit함; 낮은 sample 다양성 | 7B/13B model에서 sample한 path를 pool(U13B: 56.5; 33B path도 추가한 U33B: 57.9) | [[rft-scaling-relationship-math]] §3.3, Table 5 |
| RAFT++ | Qwen2.5-Math-7B, LLaMA-3.2-3B | 더 빠른 초기 gain, iteration 100 근처 turning point 후 GRPO에 추월됨; policy entropy가 더 빨리 하락(policy entropy는 자기 sample에서 model의 next-token distribution entropy) | positive-only update로 인한 exploration 감소(저자 해석) | clip-higher: 비대칭 importance-ratio clip 범위 \([1-0.2,1+0.28]\)가 LLaMA에서 entropy를 안정화 | [[raft-reinforce-rej-minimalist]] §5, Figs. 2–3 |
| SWE-Gym on-policy RSFT | Qwen2.5-Coder-32B, OpenHands | SWE-bench Lite 15.3% → 8.7% | 분리되지 않음; 저자들은 더 강한 optimization 또는 base model을 제안 | 보고된 해결책 없음 | [[swe-gym]] §4.2 |
| Kimi-Dev SFT prior(teacher data이며 self-generated 아님) | DeepSeek-R1 trajectory로 mid-training과 cold-start SFT 뒤의 Qwen2.5-72B | 200개 trajectory로 adaptation했을 때 SWE-Agent pass rate 퇴행 | 하나의 data mode 암기라는 가설 | 없음; RL prior에서는 하락이 나타나지 않음 | [[kimi-dev]] §4.2 |
| DeepSeek-R1 stage 대 R1-Zero | 둘 다 V3-Base에서 시작; “thousands” 예시의 cold start | cold-start model Dev1 대 RL-only R1-Zero: AIME 2024 59.0 대 77.9, SimpleQA 17.8 대 30.3; final R1 대 R1-Zero: GPQA Diamond 71.5 대 75.8, CNMO 2024 78.8 대 88.1 | 제한된 cold-start 크기(Dev1 reasoning score에 대해서는 저자 설명); final-stage gap은 설명되지 않음 | 해결책으로 제시되지 않음. cold start 뒤 reasoning RL은 Dev2 AIME 74.0에 도달 | [[deepseek-r1]] §4, Table 3 |

독립된 두 설정이 반복 rejection-sampling round에서 capability regression을 보고한다. APPS 및 HumanEval의 ReST-EM, RLHF-V3의 Llama 2다(복제됨; 어느 경우도 mechanism은 분리되지 않음). ReST-EM의 퇴행은 모든 round가 base model에서 restart하고 문제별 sample cap을 두었음에도 발생했으며, 저자들은 이를 작은 문제 집합 탓으로 본다. Llama 2는 이전 모든 round의 top sample을 pool하면 퇴행이 해결됐다고 보고했다. 이 source들에서 weight 또는 data를 이전 distribution에 더 가깝게 유지하는 두 설계 선택은, base model에서 restart하는 것(이전 iterate에서 계속하는 것보다 HumanEval에 더 잘 전이됨; [[rest-em]] Fig. 7)과 round 간 채택 sample을 pool하는 것([[llama-2]] §3.2.3)이다. V-STaR도 iteration 간 correct sample을 축적하지만([[v-star]] Alg. 1), 직전 iteration sample만으로 학습하는 방식과 비교하지는 않는다.

## §5 Cold-start SFT, iterative distillation, cross-stage distillation

Companion figure의 round-structure panel([figures/iterative-loop.html#rounds](figures/iterative-loop.html#rounds))은 §1, §5, §6의 모든 pipeline stage 순서를 나란히 둔다. 따라서 reader는 sampling, filtering, SFT, RL, distillation이 어디에서 일어나는지와 어느 round가 base model에서 restart하는지를 볼 수 있다.

### 5.1 DeepSeek-R1: cold start 대 R1-Zero

Cold start는 RL이 쓸 수 있는 output format을 가진 model에서 시작하도록 RL 전에 적용하는 작은 SFT set이다. Long chain-of-thought(long-CoT) example은 answer 전에 길게 이어지는 step-by-step reasoning trace를 담은 response다. DeepSeek-V3-Base는 mixture-of-experts(MoE) model로, router가 token마다 expert feed-forward block의 subset을 활성화한다(총 671B, activated parameter 37B; [[deepseek-r1-recipe]]). DeepSeek-R1-Zero는 SFT 없이 rule-based accuracy 및 format reward로 DeepSeek-V3-Base에 GRPO를 적용했다. AIME 2024 pass@1은 15.6%에서 77.9%로 상승했다([[deepseek-r1]] §2.3). 그 output은 readability가 나쁘고 언어가 섞였으며(§3), 저자들은 writing과 open-domain question answering에서 성능이 제한적이라고 설명한다(§1). DeepSeek-R1은 네 stage를 추가했다([[deepseek-r1-recipe]]). (1) Dev1: 정확성과 readability로 필터링하고 DeepSeek-V3로 다듬은 뒤 annotator가 검증한 R1-Zero sample에서 만든 “thousands”개의 long-CoT example으로 cold-start SFT(B.3.2). (2) Dev2: language-consistency reward가 있는 reasoning RL. (3) Dev3: 약 600k개의 rejection-sample reasoning sample과 약 200k개의 non-reasoning sample, 총 804,745개로 DeepSeek-V3-Base부터 SFT(B.3.3, Table 5). (4) R1: rule, RM, format, language reward를 둔 RL이며, preference reward는 1,700 step 중 마지막 400 step에서만 사용(§3.2.2).

| Benchmark ([[deepseek-r1]] Table 3) | R1-Zero | Dev1 | Dev2 | Dev3 | R1 |
|---|---|---|---|---|---|
| AIME 2024 pass@1 | 77.9 | 59.0 | 74.0 | 78.1 | 79.8 |
| CNMO 2024 pass@1 | 88.1 | 58.0 | 73.9 | 77.3 | 78.8 |
| GPQA Diamond pass@1 | 75.8 | 66.1 | 70.7 | 71.2 | 71.5 |
| IF-Eval (prompt strict) | 46.6 | 71.7 | 72.0 | 78.1 | 83.3 |
| AlpacaEval 2.0 (LC win rate) | 24.7 | 50.1 | 55.8 | 62.1 | 87.6 |
| SimpleQA (correct) | 30.3 | 17.8 | 28.2 | 24.9 | 30.1 |
| C-Eval (EM) | 92.8 | 85.7 | 91.9 | 86.4 | 91.8 |
| Aider-Polyglot (acc.) | 12.2 | 6.7 | 25.6 | 44.8 | 53.3 |

R1-Zero와 Dev1은 모두 DeepSeek-V3-Base에서 시작하므로, 두 열은 RL-only model과 cold-start SFT model을 비교할 뿐 한 실행 안의 하락을 보이는 것은 아니다. Cold-start model은 instruction following에서 더 높고(IF-Eval 71.7 대 46.6), reasoning과 factual QA에서는 더 낮다(AIME 2024 59.0 대 77.9; SimpleQA 17.8 대 30.3). 저자들은 reasoning gap을 제한된 cold-start 크기 탓으로 본다(§4). Dev1에서 reasoning RL을 한 Dev2는 AIME 2024 74.0, CNMO 2024 73.9에 도달했다. Rejection-sample reasoning data와 non-reasoning data로 V3-Base에서 다시 학습한 Dev3는 Dev2보다 AlpacaEval 2.0에서 높았고(62.1 대 55.8) Aider-Polyglot에서도 높았으나(44.8 대 25.6), C-Eval(86.4 대 91.9)과 SimpleQA(24.9 대 28.2)에서는 낮았다. Final R1은 GPQA Diamond와 CNMO 2024에서 R1-Zero보다 낮게 남았다(Table 3). Base model에서 시작하는 RL은 7B dense 및 16B MoE base에서는 길이가 늘며 repetition이 생기면서 AIME를 개선하지 못했고, 32B dense 이상 base에서 작동했다(G.1). 상태: 결과(단일 연구; 한 model family; stage가 독립적으로 ablate되지 않음).

### 5.2 GLM-4.5: expert model과 iterative distillation

GLM-4.5 post-training은 두 stage로 구성된다([[glm-4-5]] §3). Stage 1은 Reasoning, Agent, General-chat expert를 학습하며, 각각 extended CoT가 있는 작은 cold-start SFT set을 거친 뒤 RL을 수행한다(§3.1). Stage 2는 explicit thinking이 있는 data와 없는 data의 균형을 맞춘, 최대 128K token 길이의 수백만 expert sample로 만든 Overall SFT set을 이용해 expert를 하나의 hybrid model로 distill한다(§3.1). Agent task에서는 보고서가 iterative distillation을 기술한다. Cold-start model에서 step count 또는 plateau까지 RL을 수행하고, 그다음 “원래 cold-start data를 RL-trained model이 생성한 response로 대체”하여 다시 SFT하고, 더 어려운 task로 RL을 추가 수행한다(§3.3.2). 이 loop의 수치나 ablation은 보고서에 없다. 상태: 결과(단일 연구, 정성적).

### 5.3 GLM-5: sequential RL 후 on-policy cross-stage distillation

GLM-5는 SFT, Reasoning RL, Agentic RL, General RL을 순서대로 수행하며, 서로 다른 objective를 순차적으로 최적화하면 “previously acquired capabilities의 cumulative degradation을 초래할 수 있다”고 말한다([[glm-5]] §3.5). 마지막 stage는 on-policy distillation이다. Student가 response를 sample하고, 이전 stage의 final checkpoint가 그 stage RL training set에서 뽑은 prompt에 대해 teacher로 작동한다. GRPO advantage는 다음으로 대체된다.

\[
\hat A_{i,t}=\operatorname{sg}\left[\log\left(\frac{\pi_{\mathrm{teacher}}(y_{i,t}\mid x,y_{i,<t})}{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}\right)\right]
\]

여기서 \(y_{i,t}\)는 student sample \(i\)의 token \(t\), \(\pi_{\mathrm{teacher}}\)는 teacher의 inference-engine probability, \(\pi_\theta\)는 student의 training probability, \(\operatorname{sg}\)는 stop-gradient operation이다(§3.5, Eq. 2). Group size는 1이고 batch size는 1,024다(§3.5). Introduction은 이 distillation을 post-training “throughout” 사용한다고 하지만, §3은 이를 final stage라고 부른다(§1, §3). 효과를 분리하는 수치는 없다. GLM-5 SFT에서는 agent trajectory의 erroneous segment를 “retained but masked out in the loss”한다(§3.1). On-policy-distillation course(ch-04)가 이 objective를 유도한다. 상태: 결과(단일 연구, ablation 보고 없음).

### 5.4 더 강한 model에서의 distillation 대 자기 생성 데이터

Llama 2는 70B에서만 rejection-sampling하고 더 작은 model을 그 sample으로 학습했다([[llama-2]] §3.2.3). ReST-EM은 PaLM 2-L solution으로 fine-tuning한 PaLM 2-S가 자기 ReST-EM data로 fine-tuning한 PaLM 2-S보다 낫다고 발견했다. 저자들은 더 많은 question에 적어도 한 해법이 있기 때문이라고 설명한다([[rest-em]] §5.3, Fig. 6). 32B에서 DeepSeek-R1-Distill-Qwen-32B(800k set SFT)는 AIME 2024 pass@1 72.6, GPQA Diamond 62.1이었지만, Qwen2.5-32B-Base에서 10K step 초과로 RL을 한 model은 47.0, 55.0이었다([[deepseek-r1]] Table 16). 더 강한 sampler는 문제별 성공 확률 \(p\)를 올리고, §1.2에 따라 채택 sample을 하나라도 생산하는 어려운 문제의 비중을 높인다(해석; §1.2에서 도출).

## §6 Agentic 사례

### 6.1 SWE-Gym: 범용 scaffold에서 on-policy rejection-sampling SFT가 resolution을 낮췄다

SWE-Gym은 executable environment와 unit test가 있는 11개 repository의 Python task 2,438개를 제공한다([[swe-gym]] §3). Scaffold는 model 주변의 tool set과 prompting loop다. OpenHands는 범용 ReAct-style scaffold이고 MoatlessTools는 고정 workflow다(§4.1). Resolution rate는 agent patch 뒤 test를 통과한 SWE-bench task의 비율이다. GPT-4o와 Claude 3.5 Sonnet에서 sample한 성공 trajectory 491개로 Qwen2.5-Coder-Instruct-32B를 fine-tuning하자, SWE-bench Lite는 3.0%에서 15.3%로, SWE-bench Verified는 7.0%에서 20.6%로 올랐다(Table 3). 그 fine-tune된 32B model은 이후 temperature 0.5에서 task당 6개 trajectory를 sample하여 성공한 on-policy trajectory 868개를 만들었다. 이것과 teacher trajectory 491개로 base 32B model을 fine-tuning하자 SWE-bench Lite는 8.7%로 하락했으며, 이는 “self-improvement is not yet working”을 시사한다(§4.2). MoatlessTools에서는 temperature 1.0, task당 30 rollout의 두 iteration이 7B model을 7.0%에서 9.0%, 이어 10.0%으로 올렸고, 32B model은 19.0%에서 19.7%로 올린 뒤 추가 gain이 없었다(§4.3, Table 4). 성공 trajectory 1,318개와 실패 trajectory 1,318개로 학습한 verifier는 Verified를 \(k=1\)에서 20.6%였던 것에서 best-of-16에서 32.0%로 올렸다(§5.1.1). 상태: 결과(단일 연구).

### 6.2 Kimi-Dev: multi-turn agent의 prior로서 single-turn RL

Kimi-Dev는 issue resolution을 single-turn step(file localization, code edit, test writing)으로 나누고 검증 가능한 reward를 사용하는 Agentless format으로 Qwen2.5-72B-Base를 학습한다([[kimi-dev]] §2.1, §3.1). Recipe는 GitHub issueㆍpull-request와 synthetic reasoningㆍagentic data 위의 약 150B token mid-training, SWE-Gym 및 SWE-bench-extra task에 대한 DeepSeek-R1 reasoning output cold start, 0/1 execution reward를 둔 code-edit step RL이다(§3.2–3.3). 처음에 pass@16 = 0인 prompt를 제거하여 1,200개 prompt를 남기고, 이전에 제외한 prompt 500개를 RL 100 step마다 다시 더한다(§3.3, App. C.1). Late RL에서 “positive example reinforcement”는 recent iteration의 성공 sample을 current batch에 더한다. 저자들은 exploration이 감소했을 때 이것이 BugFixer RL을 개선했다고 보고하지만 수치는 제공하지 않는다(§3.3, App. C.2, Fig. 8). 결과 model은 issue당 patch 40개와 test 40개로 SWE-bench Verified의 60.4%를 해결한다(Table 1).

Claude 3.7 Sonnet으로 수집한 SWE-smith trajectory 5,016개로 Kimi-Dev를 fine-tuning하면 multi-turn SWE-Agent로 pass@1 48.6%를 얻었고, 같은 data로 학습한 SWE-agent-LM(32B)은 40.2%다(§4.1, Table 2). 저자들은 이 adaptation에 대해 base, mid-trained(MT), long-CoT SFT, RL이라는 네 starting point를 비교했다. RL prior는 거의 모든 SFT budget에서 다른 prior보다 좋았고, base prior의 최고 pass@1에 \(1.5\times2^{28}\) SFT token 대신 \(2^{23}\) SFT token으로 도달했다(§4.2, Fig. 5). SFT prior는 zero-shot에서는 더 좋았지만 200개 trajectory에서 퇴행했다(§4.2). Turn limit 아래에서 adapted RL prior는 70 turn을 지나서도 계속 개선됐지만, SFT, MT, base prior는 각각 70, 60, 50 turn 근처에서 diminishing return을 보였다(Fig. 6). SWE-bench-Live와 SWE-bench Multilingual에서는 SFT 및 RL prior가 적은 adaptation trajectory에서 base prior보다 더 잘 일반화했지만, trajectory가 많아지면 gap은 좁아졌다(App. G.4). 상태: 결과(단일 연구).

### 6.3 Llama 3 tool과 GLM-4.5 agent RL

Llama 3는 tool-use data에 rejection sampling을 쓰지 않았으며, tool benchmark에서 rejection sampling의 gain도 보고하지 않았다([[llama-3-recipe]], v3 §4.3.5). GLM-4.5 agent RL은 sample한 \(K\)개 trace 각각에 reward에서 group-mean reward를 뺀 값을 weight로 주고, model이 생성한 token에서만 loss를 계산하며, tool-call format이 틀리면 reward 0으로 trace를 중단한다([[glm-4-5]] §3.3.2). 저자들은 web search 및 SWE task의 RL이 다른 tool-use 및 coding benchmark를 개선했다고 말하지만 수치는 제공하지 않는다(§3.3.2). 함의(해석): policy의 resolution rate가 낮은 agent에 대해, 이 장의 source는 on-policy RSFT와 직접 비교되지 않은 자기 생성 데이터의 세 다른 용도를 보고한다. 실패에서 verifier를 학습하는 것(SWE-bench Verified에서 \(k=1\) 대비 Best@16 +11.4점; [[swe-gym]] §5.1.1), RL checkpoint를 다시 SFT로 distill하는 것(수치 없음; [[glm-4-5]] §3.3.2), multi-turn SFT 전의 single-turn RL prior([[kimi-dev]] §4.2, Fig. 5)다.

## §7 어떤 update가 더 일반화하고 덜 잊는가(ch-38a 미리 보기)

| 근거 | 설정 | Positive-only 또는 SFT update | 실패를 사용하는 update(RL) | 상태 |
|---|---|---|---|---|
| [[raft-reinforce-rej-minimalist]] Table 1, Fig. 3 | Qwen2.5-Math-7B, in-domain math | RAFT++ 56.1; entropy가 더 빨리 하락; plateau | GRPO 56.3; 계속 개선 | 결과(단일 연구) |
| [[kimi-dev]] §4.2, Fig. 7 | Qwen2.5-72B, SWE-Agent adaptation | long-CoT SFT prior가 200개 trajectory에서 퇴행 | RL prior가 거의 모든 budget에서 더 좋고 end-to-end RL 뒤에도 약간 더 좋음 | 결과(단일 연구) |
| [[rlvr-beyond-base-model]] Abstract (v5) | 여러 model family; math, code, visual reasoning | distillation은 새 reasoning pattern을 도입할 수 있음 | RLVR(verifiable reward를 둔 RL)는 작은 \(k\)에서 더 높고, base는 큰 \(k\)에서 더 높음; 경계는 학습과 함께 흔히 좁아짐 | 결과(여러 family를 다룬 한 연구) |
| [[prorl]] §4.2–4.3 | R1-Distill-Qwen-1.5B, 2k RL step 초과 | 검사하지 않음 | math pass@128은 흔히 하락; unseen Reasoning Gym task boxnet 0.00 → 7.91 | 결과(단일 연구) |
| [[deepseek-r1]] Table 16 | Qwen2.5-32B-Base | R1에서의 distillation: AIME 72.6 | base에서의 RL: AIME 47.0 | 결과(단일 연구; 더 강한 teacher) |

이 source들은 non-target skill의 forgetting에 대해, 일치한 prompt와 compute에서 filtered self-sample SFT를 on-policy RL과 비교하지 않는다. 그 비교는 이 장 source 안에서는 열린 질문이다. ch-38a는 이를 직접 측정하는 연구를 다룬다(arXiv:2501.17161, arXiv:2509.04259, arXiv:2510.18874).

## Negative sample과 negative feedback

이 절은 course standard의 “negative” 네 의미를 쓴다. (1) negative marginal value, (2) content로서의 negative, (3) conditioning으로서의 negative, (4) gradient로서의 negative다. ch-31a는 supervised training에서 (2), (3)을 다루고, ch-43a는 (4)를 유도한다.

**Negative의 출처.** Answer matching([[rest-em]], [[star]], [[rft-scaling-relationship-math]]), unit test([[swe-gym]], [[kimi-dev]]), reward model([[llama-2]], [[llama-3]]), tool-format check([[glm-4-5]])다. 이 장 source 중 false-negative rate를 보고한 것은 없다. False positive는 보고되어 있다. high-temperature sampling 또는 rationalization에서 틀린 reasoning으로 맞는 answer에 도달하는 경우([[star]] §5; [[rest-em]] §4), Kimi-Dev RL 중 잘못된 이유로 통과하는 TestWriter test([[kimi-dev]] §3.5.3)다.

**현재 관행이 이를 다루는 방법.** RSFT, STaR, RFT, ReST-EM, Llama 2 rejection sampling, DeepSeek-R1 stage 3은 거절 sample을 버린다. 이는 (1), 즉 §2 항등식에서 zero weight다. STaR rationalization은 실패 문제를 재사용하지만 실패 rationale은 버린다. GLM-5는 오류 agent step을 context에는 유지하되 loss에서는 mask하며, 이는 (2)다([[glm-5]] §3.1). SWE-Gym verifier는 ordinary cross-entropy로 실패 trajectory에서 `<NO>`를 출력하도록 학습하며 verifier에 대한 (2)다([[swe-gym]] §5.1.1). V-STaR의 DPO verifier는 reference에 비해 incorrect solution의 likelihood를 낮추며 verifier에 대한 (4)지만, generator는 positive-only로 남는다([[v-star]] §3.1). DeepSeek-R1(GRPO)과 GLM-4.5 agent RL의 group-baseline RL은 group mean보다 낮은 sample에 negative weight를 주며, policy에 대한 (4)다([[deepseek-r1]] Eq. 3; [[glm-4-5]] §3.3.2).

**메커니즘.** Logit \(z\), 확률 \(p\)를 둔 softmax에서 token log-probability gradient는 \(\partial\log p_y/\partial z_j=\mathbf1[j=y]-p_j\)다. 여기서 \(y\)는 target token이고 \(j\)는 vocabulary를 돈다. \(p=[0.7,0.2,0.1]\)이고 token 3(\(p=0.1\))을 낮춘다고 하자. \(\log p_3\)에 대한 gradient descent는 \(d=-[-0.7,-0.2,0.9]=[+0.7,+0.2,-0.9]\) 방향으로 logit을 움직인다. 작은 step size \(\eta\)에서 probability의 1차 변화는 \(\Delta p_j\approx\eta\cdot p_j\cdot(d_j-\sum_kp_kd_k)\)이며, \(\sum_kp_kd_k=0.49+0.04-0.09=0.44\)이므로 \(\Delta p\approx\eta\cdot[+0.182,-0.048,-0.134]\)다. 이미 가장 가능성 높은 alternative인 token 1이 제거된 mass를 모두 받고 token 2도 mass를 잃는다. Token 1을 올리면 logit은 \([0.3,-0.2,-0.1]\) 방향으로 움직이며 \(p_1\)이 1에 가까워질수록 더 작아지는 변화다. 따라서 unlikely sample의 negative gradient는 확률을 model의 현재 top choice에 집중시킨다. Positive-only RSFT는 (4)를 적용하지 않으므로 likelihood-displacement risk는 없지만, 그 risk는 [[raft-reinforce-rej-minimalist]] §5.1에서 측정한 entropy reduction이다.

**수치가 있는 근거.** Failure를 쓰는 이점: 실패 sample로 학습한 verifier를 추가하는 V-STaR는 기존 self-improvement 및 verification baseline보다 test accuracy를 4–17점 개선했다([[v-star]] Abstract). Online RL에서 negative advantage의 이점은 한 비교에서는 작았다. Qwen에서 GRPO 56.3 대 RAFT++ 56.1, LLaMA에서 28.4 대 27.6([[raft-reinforce-rej-minimalist]] Table 1)다. Failure mode: LLaMA-3.2-3B에서 ±1 reward REINFORCE는 24.2로 RAFT++ 27.6보다 낮았으며, ablation에서 all-wrong prompt를 모두 제거했을 때 reward gain이 가장 컸다(Table 1, Fig. 4). 저자들의 해석은 final-answer correctness가 너무 coarse한 negative label이고, 그러한 negative에서 unlearning하는 일은 positive로 fine-tuning하는 일보다 덜 안정적이라는 것이다(§5, 해석). 이 장 source 중 RL gain에서 negative가 차지하는 비율을 측정한 것은 없다.

**Control.** All-wrong prompt를 제거하거나 group-mean baseline을 사용한다([[raft-reinforce-rej-minimalist]] Fig. 4). pass@16 = 0 prompt를 filter한 뒤 나중에 다시 추가한다([[kimi-dev]] App. C.1). Entropy를 유지하도록 asymmetric clipping을 둔 importance ratio를 clip한다([[raft-reinforce-rej-minimalist]] §5.1). Exploration이 떨어질 때 late RL에 positive sample을 추가한다([[kimi-dev]] App. C.2). 오류 step을 학습하지 말고 mask한다([[glm-5]] §3.1). DPO를 model 자기 sample에 적용하면, Llama 3처럼 chosen response에 NLL(negative log-likelihood) 항을 coefficient 0.2로 추가한다. 이는 previous round 최고 model의 response에 대한 human-labeled pair에서 했으며 chosen log-probability가 내려가는 것을 막는다고 말한다(ablation 출력 없음; [[llama-3]] §4.1.4).

**진단.** Round마다 다음을 log한다. difficulty bin별 acceptance rate, 문제당 distinct accepted solution([[rft-scaling-relationship-math]] Table 2), policy entropy([[raft-reinforce-rej-minimalist]] Fig. 3), all-correct 및 all-wrong prompt의 share, pass@1과 큰 \(k\)에서 pass@k([[rest-em]] Fig. 5), 틀린 reasoning이 있는지 확인하는 채택 trace 표본 감사, agent의 stuck-in-loop 및 empty-patch rate([[swe-gym]] Table 3).

**일반성에 미치는 효과.** Negative를 버리는 ReST-EM은 검사한 모든 \(K\)에서 pass@K를 올렸고, gap은 보통 \(K=1\)에서 가장 컸다([[rest-em]] §5.2, Fig. 5). Positive-only update는 GRPO보다 entropy를 더 빨리 낮췄다([[raft-reinforce-rej-minimalist]] §5.1). 이 장의 해석은 sampling이 다양하게 유지될 때에만 negative를 버리는 것이 coverage를 유지한다는 것이다(해석; RAFT의 큰 \(k\) pass@k를 측정한 source는 없음). High-temperature sampling은 correct-answer, wrong-reasoning target을 늘리며, STaR는 이것이 일반화를 막는다고 말한다([[star]] §5). ReST-EM도 rationalization에서 같은 종류 false positive를 보고한다([[rest-em]] §4). 이 선택의 calibration, hallucination, over-refusal 효과는 이 장 source가 보고하지 않는다.

## 레시피

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Llama 2-Chat 70B | 70B | RL (rejection sampling) | sampler; student | rejection sampling은 70B로만 수행; 더 작은 chat model은 그 sample로 fine-tune | arXiv:2307.09288v2 §3.2.3; [[llama-2-recipe]] | verified 2026-09-14 | ablation 보고 없음 |
| Llama 2-Chat | 전체 | RL (rejection sampling) | prompt당 sample \(K\); fine-tuning hyperparameter | 출력되지 않음 | v2 §3.2.3, Fig. 7, A.3 | not reported (body와 A.3 확인) | Fig. 7: max − median reward가 \(N\)과 함께 증가 |
| Llama 2-Chat-RLHF | 70B | RL (rejection sampling) | sampling temperature | 10–100 sample에서 최적 \(T\in[1.2,1.3]\); version마다 재조정 | v2 §3.2.3, Fig. 8 | verified 2026-09-14 | Fig. 8 (\(T=0.6\)–1.5) |
| Llama 2-Chat | 전체 | RL (rejection sampling) | candidate pool | V3까지는 이전 iteration만; 이후 이전 모든 iteration의 top sample | v2 §3.2.3 | verified 2026-09-14 | V3 regression(시 운율); figure 없이 fix 보고 |
| Llama 2-Chat | 전체 | RL | schedule | RLHF (V4)까지 rejection sampling만; 이후 다시 sample하기 전에 rejection-sampling checkpoint 위에 PPO | v2 §3.2.3 | verified 2026-09-14 (V4 표현 모호) | Fig. 11: PPO를 쓴 RLHF-v5가 PPO 없는 v5보다 높음(plot만) |
| Llama 2-Chat | 7B, 13B / 34B, 70B | RL | PPO KL coefficient \(\beta\) | 0.01 / 0.005 | v2 §3.2.3, Eq. 4 | verified 2026-09-14 | ablation 보고 없음 |
| Llama 3.1 405B | 405B | SFT (rejection sampling) | \(K\); selection | human-annotation prompt당 보통 \(K=10\)–30; RM이 최고 선택 | arXiv:2407.21783v3 §4.2.2; [[llama-3-recipe]] | verified 2026-09-14 | ablation 보고 없음 |
| Llama 3.1 405B | 405B | preference | round data의 DPO | LR \(10^{-5}\); \(\beta=0.1\); chosen의 NLL coefficient 0.2 | v3 §4.1.4 | verified 2026-09-14 | ablation 출력 없음 |
| GPT-J + STaR | 6B | SFT (self-generated) | sampling; init; LR; batch | greedy; 매 loop original GPT-J; Adam \(10^{-6}\); 8 sequence × 1,024 token | arXiv:2203.14465v2 §3.1, §4.1, App. H; [[star]] | verified 2026-09-14 | App. H: LR search \(10^{-7}\)–\(10^{-4}\); §5: T 0.5/0.7은 더 나쁨 |
| LLaMA-7B/13B RFT | 7B, 13B | SFT (self-generated) | sample; temperature; dedup | \(k=100\); \(T=0.7\); 서로 다른 equation list당 한 path | arXiv:2308.01825v2 §3.3, App. A.3 | verified 2026-09-15 | Fig. 4, Table 5: \(k\in\{1,\ldots,100\}\); dedup 대 no dedup |
| LLaMA RFT | 7B–70B (Table 5) | SFT (self-generated) | epoch; example | 3 epoch; 약 47K(\(k=100\)), 104K(U13B) | v2 Table 5 | verified 2026-09-15 | RFT 전용 epoch ablation 없음; Table 5가 data size 비교 |
| LLaMA SFT baseline for RFT | 7B–70B | SFT | LR; batch; epoch | peak \(2e-5\), warmup 3%; batch 128; 3 epoch | v2 App. A.1 | verified 2026-09-15 (RFT-specific LR 출력 안 됨) | ablation 보고 없음 |
| PaLM 2-L ReST-EM | not reported | SFT (self-generated) | sample; decoding; cap | 문제당 MATH 32, APPS 64; top-K 40, \(T=0.7\); 문제당 최대 10개 유지 | arXiv:2312.06585v4 §5; [[rest-em]] | verified 2026-09-14 | §5.3: 1 iteration 3× sample의 40.3% < iteration 3의 41.9% |
| PaLM 2 ReST-EM | not reported | SFT (self-generated) | init; iteration; LR | 매 iteration base model; MATH 3, APPS 2; LR, batch, epoch 미보고 | v4 §3, Figs. 2–3 | verified / not reported | Fig. 7: restart가 HumanEval transfer를 개선 |
| LLaMA 2 7B V-STaR | 7B | SFT (generator); reward-model (DPO verifier) | sample; iteration; tuning | iteration당 \(k=16\); 3 iteration; LoRA; accumulated correct sample 위에서 base부터 generator 학습 | arXiv:2402.06457v2 §4, Alg. 1 | verified 2026-09-15 | Fig. 3: 3 × 16 sample에서 V-STaR 대 V-STaR[1 iter] |
| Qwen2.5-Math-7B RAFT/RAFT++/GRPO | 7B | RL (positive-only 대 GRPO) | LR; prompt; sample; mini-batch; max length | AdamW \(1\times10^{-6}\); iteration당 1,024 prompt; \(n=4\); mini-batch 512; 4,096 token | arXiv:2504.11343v2 §4 | verified 2026-09-15; Table 1 caption은 algorithm별 batch, mini-batch, LR를 tuning했고 tuned value는 v2에 출력하지 않는다고 말함 | 이 §4 value의 ablation 보고 없음 |
| DeepSeek-R1 Dev1 | 671B MoE | SFT (cold start) | size; base; schedule | “thousands”; V3-Base; 2–3 epoch, cosine \(5\times10^{-5}\to5\times10^{-6}\); 32,768 token; batch 128 | arXiv:2501.12948v2 B.3.2, B.4.2; [[deepseek-r1-recipe]] | verified 2026-09-14 | Table 3: Dev1 대 R1-Zero |
| DeepSeek-R1 Dev3 | 671B MoE | SFT (rejection sampling) | data; base | RL checkpoint에서 올바른 reasoning sample 약 600k + non-reasoning 약 200k; 총 804,745; V3-Base에서 | v2 B.3.3, Table 5, Fig. 2 | verified 2026-09-14 (epoch 충돌: v1 두 번, v2 2–3) | Table 3: Dev2 → Dev3 |
| GLM-4.5 agent expert | 355B MoE | SFT 및 RL (iterative distillation) | loop | RL이 step count에 도달하거나 plateau하면 cold-start data를 RL-model response로 바꾸고 SFT, 더 어려운 task로 RL 계속 | arXiv:2508.06471v1 §3.3.2 | verified 2026-09-14; cold-start size 미보고 | ablation 보고 없음 |
| GLM-5 | 총 744B, active 40B | RL (on-policy cross-stage distillation) | teacher; group; batch | 이전 stage final checkpoint; group size 1; batch 1,024 | arXiv:2602.15763v2 §3.5, Eq. 2 | verified 2026-09-15 | ablation 보고 없음 |
| Qwen2.5-Coder-32B SWE-Gym | 32B | SFT (on-policy) | sample; temperature; data | task당 trajectory 6개, \(t=0.5\); on-policy 868 + teacher trajectory 491 | arXiv:2412.21139v2 §4.2 | verified 2026-09-15 | SWE-bench Lite 15.3% → 8.7% |
| Qwen2.5-Coder-7B SWE-Gym MoatlessTools | 7B | SFT (on-policy) | rollout; temperature; cap; iteration | task당 30, \(T=1.0\); instance당 cap 2; 2 iteration | v2 §4.3, Tables 4, 6 | verified 2026-09-15 | Table 6: cap 2가 full data보다 약간 높음 |
| Kimi-Dev 72B | 72B | RL (single-turn code edit) | rollout; prompt; context; filtering | 1,024 problem 각각 10 rollout; iteration당 5 step; 64k context; pass@16 = 0 제거, 100 step마다 +500 prompt | arXiv:2509.23045v3 §3.3, §3.5.3, App. C.1 | verified 2026-09-15 | App. C.2 Fig. 8(positive example reinforcement; plot만) |
| Kimi-Dev 72B | 72B | SFT (agent adaptation) | data; context | SWE-smith trajectory 5,016개; training context 64K; inference에서 128K와 100 turn | v3 §4.1 | verified 2026-09-15 | Fig. 5 data sweep |

**소규모 범용 실행의 시작점.** 이 표의 source 중 full generality panel로 7B에서 rejection-sampling round를 실행한 것은 없으므로, 아래 값은 그것을 낳은 run의 조건을 유지한다. Verifiable math task에는 ReST-EM 설정이 가장 가깝다. 문제당 temperature 0.7에서 32 sample, top-K 40, 문제당 올바른 sample 최대 10개 유지, 매 iteration base model을 다시 fine-tune, 매 iteration 뒤 held-out evaluation이다(PaLM 2, 크기 미보고, MATH training problem 7,500개, learning rate 미보고). Verifiable math에서 7B model의 online training에는 RAFT++ 대 GRPO 비교가 AdamW \(1\times10^{-6}\), iteration당 prompt 1,024개, prompt당 sample 4개, mini-batch 512, token limit 4,096을 사용했다(Qwen2.5-Math-7B-base, Numina-Math prompt, verl). Table 1 caption은 algorithm별 batch size, mini-batch size, LR를 tuning했고 tuned 값은 출력되지 않았다고 말한다. 32B coder model agent task에는 SWE-Gym이 성공 teacher trajectory 491개로 학습해 SWE-bench Verified에서 +13.6점을 얻었고, on-policy trajectory 868개의 한 round를 더하자 SWE-bench Lite에서 15.3% → 8.7% 하락을 측정했다.

## 일반화 관점

**(a) 무엇이 breadth를 넓히는가.** 여러 model sample을 pool하는 것을 포함해 문제당 더 많은 서로 다른 채택 해법([[rft-scaling-relationship-math]] Tables 2–3). 매 round base model에서 restart([[rest-em]] Fig. 7)하거나 이전 모든 round의 채택 sample을 pool하는 것([[llama-2]] §3.2.3). Non-target data를 rejection-sampling SFT set에 섞는 것. DeepSeek-R1 Dev3는 Dev2보다 AlpacaEval 2.0을 55.8에서 62.1로, Aider-Polyglot을 25.6에서 44.8로 올렸고 저자들은 이를 non-reasoning 및 code-engineering data 탓으로 보지만, 같은 step에서 C-Eval은 91.9 → 86.4, SimpleQA는 28.2 → 24.9로 하락했다([[deepseek-r1]] §4, Table 3). 더 강한 model의 sample: 더 많은 question에 solution이 있기 때문이라고 설명된 PaLM 2-S용 PaLM 2-L solution([[rest-em]] §5.3, Fig. 6), reasoning benchmark의 Qwen2.5-32B용 R1 distillation([[deepseek-r1]] Table 16). Multi-turn agent SFT 전 single-turn RL prior([[kimi-dev]] §4.2, App. G.4). Sequential RL 뒤 earlier-stage checkpoint를 final model에 다시 distill하는 것([[glm-5]] §3.5; 수치 없음).

**(b) 무엇이 narrowing 또는 forgetting을 일으키는가.** 작은 문제 집합을 반복하는 것([[rest-em]] §5.1, Fig. 4). 직전 round sample만 쓰는 것([[llama-2]] §3.2.3). Training question에 fit되어 서로 다른 path를 거의 만들지 않는 sampler([[rft-scaling-relationship-math]] §3.3). Entropy를 낮추는 positive-only update([[raft-reinforce-rej-minimalist]] §5.1). 틀린 reasoning의 맞는 answer를 허용하는 high-temperature sampling과 rationalization([[star]] §5; [[rest-em]] §4). 같은 base의 RL-only model보다 reasoning 및 factual QA에서 낮은 SFT model을 만든 작은 cold-start set([[deepseek-r1]] §4, Table 3). 낮은 resolution rate agent policy의 on-policy RSFT([[swe-gym]] §4.2). 서로 다른 objective를 가진 sequential RL stage([[glm-5]] §3.5).

**(c) 이 단계에서 어떻게 측정하는가.** 매 round 뒤 ReST-EM처럼 training prompt 밖 held-out task(GSM8K, HumanEval, Hungarian exam, BBH; [[rest-em]] §5.4), pass@1 옆의 큰 \(k\) pass@k([[rest-em]] Fig. 5; [[rlvr-beyond-base-model]]), test accuracy 대비 train accuracy([[rest-em]] Fig. 4), 문제당 distinct solution, entropy, instruction followingㆍfactual QAㆍ다른 언어 같은 non-target capability panel을 평가한다. DeepSeek-R1 stage는 reasoning score와 반대 방향으로 SimpleQA와 C-Eval을 움직였기 때문이다([[deepseek-r1]] Table 3). Agent에는 turn-limit profile([[kimi-dev]] Fig. 6)과 fresh 또는 multilingual suite([[kimi-dev]] App. G.4)를 더한다. 알려진 측정 오류: AIME는 문제 30개로 noisy trend를 냈고([[raft-reinforce-rej-minimalist]] §4), V-STaR MATH transfer set은 level-1 문제 150개다([[v-star]] §4). STaR는 CQA dev set 포화 뒤의 최고 결과를 보고했고([[star]] §4.1), RFT는 GSM8K test split만 평가했으며([[rft-scaling-relationship-math]] §3.3), Llama 2 human evaluation은 coding과 reasoning prompt를 제외했고([[llama-2]] §3.4.2), RLHF-V3 퇴행은 정성 분석으로 발견됐다([[llama-2]] §3.2.3).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| 매 round 직전 round의 채택 sample만으로 학습 | training accuracy는 오르지만 held-out 또는 측정하지 않은 skill 퇴행 | round별 train–test gap; held-out panel에서 pooled data 또는 restart-from-base와 last-round-only 비교 |
| 문제별 cap 없음 | 높은 pass rate 문제들이 채택 set 지배 | difficulty bin별 채택 sample histogram; pass rate 0.75 초과 문제의 gradient share |
| 채택 sample을 distinct solution 대신 세기 | round당 sample은 늘지만 accuracy gain 없음 | dedup key(equation list, normalized program)로 문제당 distinct solution |
| 초기 curve로 positive-only training 판단 | 빠른 초기 gain 뒤 plateau | 전체 run에서 step별 policy entropy 및 큰 \(k\) pass@k |
| all-wrong prompt에 baseline 없는 ±1 reward | training reward 하락 또는 불안정 | all-wrong group share; 제거 ablation |
| reasoning 확인 없이 맞는 answer 채택 | 높은 temperature 또는 rationalization에서 correct-answer, wrong-reasoning trace 증가 | 매 round 채택 trace 무작위 표본 수동 감사 |
| cold-start SFT model이 RL-only model reasoning 수준을 유지한다고 가정 | 같은 base RL-only model보다 reasoning과 factual QA에서 cold-start checkpoint 낮음 | RL 뒤만이 아니라 cold-start checkpoint를 base 및 RL-only reference와 full panel에서 평가 |
| 낮은 resolution rate agent policy에서 on-policy RSFT | resolution rate 하락; stuck-in-loop rate 변화 | 같은 trajectory 수에서 teacher-trajectory SFT와 비교 |
| 여러 RL stage의 마지막 뒤에만 평가 | 이전-stage skill이 해당 stage checkpoint보다 낮음 | 매 stage checkpoint를 모든 stage metric에서 평가 |
| learned verifier selection 신뢰 | best-of-k가 pass@k를 더 이상 따라가지 않음 | SWE-Gym처럼 pass@k와 best@k를 함께 plot |

## 이해 점검

1. §2 항등식으로 acceptance rate 0.1의 RSFT round가 왜 더 큰 learning rate의 REINFORCE처럼 동작하는지, 그리고 acceptance가 오르면서 round 간 SFT learning rate 선택에 무엇을 뜻하는지 설명하라.
2. 문제당 sample을 세 배로 한 ReST-EM 한 iteration이 iteration당 sample이 적은 세 iteration보다 왜 낮았는가? Sample budget만으로 제공하지 않는 mechanism을 이름 붙여라.
3. LLaMA-33B는 질문당 correct path 88.7개를 만들었지만 RFT로 개선되지 않았다. 질문당 distinct path로 이 결과를 설명하고, 논문의 U33B 결과가 뒷받침하는 sampler 변경을 제안하라.
4. §2 worked example에서 group-mean baseline은 B와 C prompt에 zero weight를 준다. 이것이 easy-problem bias와 all-wrong push-down을 모두 제거하는 이유와, 버리는 정보가 무엇인지 설명하라.
5. STaR rationalization은 때때로 failure로 학습하는 것으로 설명된다. 정확히 어떤 object가 gradient를 받는지, ReST-EM이 rationalization을 쓰지 않은 이유를 설명하라.
6. DeepSeek-R1 cold-start SFT model(Dev1)은 둘 다 DeepSeek-V3-Base에서 시작했는데도 AIME 2024와 SimpleQA에서는 RL-only R1-Zero보다 낮고 IF-Eval에서는 높다. 작은 SFT set과 같은 base의 RL이 이 benchmark를 반대 방향으로 움직일 수 있는 인과 설명을 하고, cold start 뒤 추가할 측정을 말하라.
7. SWE-Gym on-policy RSFT는 SWE-bench Lite를 15.3%에서 8.7%로 낮춘 반면 Kimi-Dev single-turn RL prior는 이후 agent SFT를 개선했다. §1.2와 §6을 사용해 두 설정의 starting policy와 task horizon 특성이 어떻게 다른지 설명하라.
8. GLM-5는 GRPO advantage를 earlier-stage teacher의 log-ratio로 대체한다. 이 항이 later stage에서 잃은 skill을 회복할 수 있는 이유와 group size 1로 충분한 이유를 설명하라.

## 연결

- **이전(의존성):** ch-30c — 범용 모델을 위한 Weight Averaging과 Model Merging. Merging은 earlier capability를 회복하는 또 다른 방법이며, §4와 §5.3은 data 측 방법을 제시한다.
- **다음:** ch-31a — Supervised Training의 Negative Sample: Correction, Failure Conditioning, Critique, Unlikelihood. RSFT가 버리는 negative 의미 (2), (3)을 다룬다.
- ch-30 — SFT 설계 선택과 일반화 효과: Masking, Packing, Template, Epoch, Learning Rate. 모든 RSFT round는 이 설정을 가진 SFT run이다.
- ch-30a — Fine-Tuning에서 Forgetting과 Alignment Tax: 측정과 제어. §4 퇴행의 측정 protocol.
- ch-20 — Distillation as Data: Explanation Trace와 R1-Distill 계보; ch-35 — 실전 Distillation A: Lab이 Teacher Data를 넣는 위치. §5.4에서 자기 생성 데이터와 비교한 teacher-generated data.
- ch-37 — Language Model을 위한 Policy-Gradient Foundation; ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, GSPO. §2의 유도.
- ch-38a — SFT 대 RL 일반화: On-Policy Data, Base Model KL, Output Diversity. §7에서 미리 본 직접 비교.
- ch-43 — RL의 Entropy, Output Diversity, KL 제어; ch-43a — Negative Sample과 Negative Gradient: Likelihood Displacement, Squeezing, Negative Advantage. §2 및 negative-feedback 절에서 쓴 entropy control과 negative-gradient mechanism.
- ch-44b — 일반 능력을 위한 Multi-Domain RL: Non-Verifiable Reward와 Domain Mixing. §5.3에서 소개한 sequential RL stage 간 forgetting과 GLM-5 cross-stage distillation.
- ch-16 — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, Prompt Reuse. pass@16 = 0 제거 같은 prompt filtering.
- ch-45 — Self-Improvement Loop와 Multi-Stage Reasoning Pipeline; ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, Stability; ch-29d — User Simulator, Trajectory Verification, Failed Trajectory. §6을 확장하는 judge-based self-improvement loop, multi-turn agent RL, failed-trajectory handling.

## 출처

- [[llama-2]] — iterative RLHF schedule, 70B rejection sampling, temperature, RLHF-V3 regression 및 pooling fix, PPO \(\beta\) 값, human-evaluation 범위.
- [[llama-2-recipe]] — “not printed” field를 포함한 rejection-sampling 및 PPO 행 검증.
- [[llama-3]] 및 [[llama-3-recipe]] — 보통 \(K=10\)–30, RM selection, 매 round pre-trained model에서 SFT, NLL 0.2가 있는 DPO, tool용 rejection sampling 없음.
- [[rejection-sampling-finetuning]] — pattern 정의만 제공하며 이 card에서 가져온 수치는 없음.
- [[rft-scaling-relationship-math]] — GSM8K RFT: 문제당 distinct path, \(k\) sweep, deduplication, multi-model pooling, 33B 사례(chapter excerpt는 arXiv:2308.01825v2와 검증됨).
- [[rest-em]] — EM objective, base에서 restart, sample과 cap, iteration result, APPS regression, transfer, pass@K, PaLM 2-L에서의 distillation.
- [[star]] — indicator-reward policy-gradient identity, rationalization, CQA와 GSM8K result, temperature 및 false-positive 관찰.
- [[v-star]] — correct 및 incorrect solution으로 학습한 DPO verifier, iteration 및 verifier-in-loop result(chapter excerpt는 arXiv:2402.06457v2와 검증됨; 저자는 Hosseini et al.).
- [[raft-reinforce-rej-minimalist]] — RAFT, RAFT++, REINFORCE, GRPO, Reinforce-Rej 비교; entropy collapse; all-wrong prompt ablation(chapter excerpt는 arXiv:2504.11343v2와 검증됨).
- [[deepseek-r1]] 및 [[deepseek-r1-recipe]] — R1-Zero 대 cold start, Table 3 stage result, rejection-sampling SFT stage, 32B distillation 대 RL.
- [[glm-4-5]] — expert model, cold start, iterative distillation, agent RL objective 및 format penalty.
- [[glm-5]] — sequential RL stage, on-policy cross-stage distillation objective, loss-masked erroneous agent step(chapter excerpt는 arXiv:2602.15763v2와 검증됨).
- [[swe-gym]] — teacher-trajectory SFT, on-policy self-improvement negative result, MoatlessTools iteration, per-instance cap, trajectory verifier(chapter excerpt는 arXiv:2412.21139v2와 검증됨).
- [[kimi-dev]] — Agentless RL recipe, prompt filtering, positive example reinforcement, SWE-Agent adaptation 및 prior 비교(chapter excerpt는 arXiv:2509.23045v3와 검증됨).
- [[rlvr-beyond-base-model]] — RLVR 대 base model의 큰 \(k\) pass@k(arXiv:2504.13837v5 Abstract).
- [[prorl]] — prolonged RL 아래 Diminish regime 및 out-of-distribution gain.
- 인용하지 않음: library card `iterative-sft-rl`은 검증되지 않았고 Llama 2 및 Tülu 3 수치가 위에 나열한 primary source와 충돌한다.
