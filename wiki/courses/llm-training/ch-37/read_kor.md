# Chapter 37 — Language Model을 위한 Policy-Gradient Foundations

> **핵심 통찰.** policy-gradient estimator는 model 자신이 sample한 response의 log-probability를 advantage, 즉 reward에서 baseline을 뺀 값에 비례해 높인다. baseline이 sampled response에 depend하지 않으면 expected gradient는 바꾸지 않지만, individual sample 중 무엇을 push up하고 무엇을 push down할지는 결정한다. 0/1 reward와 b = 0에서는 어떤 sample도 push down되지 않는다. gradient는 self-generated sample에 대한 SFT gradient이므로 on-policy RL은 external target에서의 SFT보다 model을 덜 바꾼다. controlled study에서는 이것이 더 적은 forgetting([[rls-razor]]) 및 unseen rule variant로의 더 좋은 transfer([[sft-memorizes-rl-generalizes]])와 함께 나타났고, output diversity 감소([[rlhf-generalisation-diversity]]) 및 large k에서 coverage 감소([[rlvr-beyond-base-model]])와도 함께 나타났다.
>
> **지침.** prompt당 여러 response를 sample할 수 있다면 leave-one-out baseline을 사용한다. 이것은 정확히 unbiased인 반면 group-mean subtraction은 (G − 1)/G만큼 scale되고 group-std normalization은 prompt-dependent scale이기 때문이다. G = 2에서는 b = 0보다 variance를 높일 수 있으므로(§2) G ≥ 4에서 사용한다. 대신 group mean 또는 group standard deviation을 쓰면 이를 prompt별 reweighting으로 취급하고 보고한다. negative advantage를 쓰면 log π의 negative weight에는 lower limit이 없으므로 이를 bound 또는 anchor한다(clipping, KL 또는 NLL term, on-policy sampling). RL로 general model을 만들 때에는 세 split(training reward, training distribution의 held-out prompt, other domain)을 evaluate하고 pass@1과 large k의 pass@k를 함께 보고한다.

## 이 chapter가 general-purpose model에 중요한 이유

이 course의 pipeline은 pre-training → mid-training → SFT → preference optimization → RL → evaluation 순서로 진행된다. 이 chapter는 policy-optimization 부분을 연다. ch-41은 reward model R(x, y)를 만들었다. 이 chapter는 R을 parameter update로 바꾸는 gradient를 정의하고, ch-38은 KL-controlled PPO recipe를 더하며, ch-39는 offline preference loss로 sampling을 없애고, ch-40은 group-baseline method를 비교한다.

이 chapter의 사실 세 가지가 RL이 model을 더 넓게 만드는지 더 좁게 만드는지를 결정한다.

1. gradient는 sample된 prompt만 touch한다. other prompt에서의 behavior는 shared parameter를 통해서만 변한다. 따라서 prompt distribution(ch-16)은 detail이 아니라 generality를 위한 design choice다.
2. sample은 model 자신의 것이다. 이것이 각 update가 model을 얼마나 멀리 움직이는지 제한하며(§5), [[rls-razor]]에서 RL이 SFT보다 덜 forgot한 proposed reason이다.
3. objective는 high-reward response에 concentrate하도록 reward한다. pass@1이 올라가도 diversity와 large-k coverage는 떨어질 수 있다(§6).

## §1 Objective와 log-derivative estimator

**정의.** policy π_θ(y | x)는 prompt x가 주어졌을 때 response y에 대한 language model의 distribution이다. RL objective는 prompt distribution D에 대한 expected reward다.

```
J(θ) = E_{x∼D} E_{y∼π_θ(·|x)} [ R(x, y) ]
```

- θ: model parameter. D: training prompt distribution. R(x, y): scalar reward(reward model, verifier, 또는 environment).

**문제.** sampling distribution π_θ가 θ에 depend하므로 fixed expectation 안에서 R을 differentiate하여 gradient를 얻을 수 없다. R은 흔히 differentiable하지 않다(verifier는0 또는1을 반환한다).

**메커니즘.** log-derivative identity ∇_θ π_θ(y|x) = π_θ(y|x) ∇_θ log π_θ(y|x)는 gradient를 log-probability로 옮긴다.

```
∇_θ J(θ) = E_{x∼D} Σ_y ∇_θ π_θ(y|x) R(x, y)
         = E_{x∼D} E_{y∼π_θ(·|x)} [ R(x, y) ∇_θ log π_θ(y|x) ]
∇_θ log π_θ(y|x) = Σ_{t=1}^{T} ∇_θ log π_θ(y_t | x, y_<t)
```

- y_t: response의 token t; T: response length; y_<t: t 이전 token.

이것은 whole response에 적용한 [[vanilla-pg]]의 REINFORCE estimator다. [[rloo]] §2.2는 reward가 complete response에 대해서만 주어지기 때문에(Eq.6) full generation을 one action으로 model한다.

**Step별 reward를 위한 causal form.** every step에서 reward r_t를 받는 trajectory에서 t 이전에 받은 reward는 t의 action에 depend하지 않으므로 그 term의 expectation은 zero다. estimator는 Σ_t ∇ log π(a_t | s_t) G_t가 된다. 여기서 return-to-go는 G_t = Σ_{t′≥t} γ^{t′−t} r_{t′}다. single terminal reward와 γ = 1이면 every t에서 G_t = R(x, y)이므로 causal form은 위 sequence form과 같다.

**Sample된 prompt만 gradient를 받는다.** Monte-Carlo estimate는 D에서 draw한 prompt batch를 average한다. D에 한 번도 나타나지 않는 prompt type은 term을 contribute하지 않는다. 그 behavior는 sampled prompt와 shared된 parameter를 통해서만 바뀌며, 그 변화가 도움인지 해인지 empirical question이다(Generalization lens).

**Worked example(손으로 검산 가능).** 한 prompt에 possible answer가 둘 있다. policy는 “correct”(reward1)에 대해 p = σ(θ), “wrong”(reward0)에 대해1 − p이고 p = 0.2다. true gradient는 dJ/dθ = p(1 − p) = 0.16이다. score는 “correct”에 대해 d log p/dθ = 1 − p = 0.8이고 “wrong”에 대해 d log(1 − p)/dθ = −p = −0.2다. single-sample estimate는 probability0.2로0.8, probability0.8로0이다. mean은0.2 × 0.8 = 0.16으로 true gradient와 같다. variance는0.2 × 0.64 − 0.16² = 0.1024다.

**근거와 한계.** identity는 어떤 R에 대해서도 exact하다. estimator는 y가 current π_θ에서 sample된 경우에만 unbiased다. older policy의 sample을 reuse하려면 importance weight가 필요하다(§3, ch-38).

**General model에 대한 함의.** J는 D로 정의된다. narrow D에서 train한 model은 그 D에 맞춰 optimize되며 estimator 안에는 outside prompt를 protect하는 것이 없다.

## §2 Baseline: variance와 bias

**정의.** baseline b는 reward에서 subtract한다. estimator는 advantage A = R(x, y) − b를 사용한다.

**문제.** reward에 큰 common offset이 있으면 R ∇ log π의 variance가 크다. §1 example에서는 sample80%가 signal을 carry하지 않고20%가 큰 signal을 carry한다.

**메커니즘(unbiasedness condition).** b가 sampled y에 depend하지 않으면(x에는 depend할 수 있다):

```
E_{y∼π_θ}[ b ∇_θ log π_θ(y|x) ] = b Σ_y ∇_θ π_θ(y|x) = b ∇_θ Σ_y π_θ(y|x) = b ∇_θ 1 = 0
```

따라서 E[(R − b) ∇ log π] = ∇J다. coordinate i마다 minimum-variance constant baseline은 b* = E[R s_i²] / E[s_i²]이고, s_i는 score ∇ log π의 i-th component다.

**Worked example(계속).** b = 0.2(expected reward J)이면 correct는(1 − 0.2)(0.8) = 0.64, wrong은(0 − 0.2)(−0.2) = 0.04다. mean은0.2 × 0.64 + 0.8 × 0.04 = 0.16, variance는0.0832 − 0.0256 = 0.0576이다. b* = E[R s²]/E[s²] = (0.2 × 0.64)/(0.2 × 0.64 + 0.8 × 0.04) = 0.128/0.16 = 0.8이면 correct는0.2 × 0.8 = 0.16, wrong은(−0.8)(−0.2) = 0.16이므로 variance는0이다. baseline(0,0.2,0.8) 세 개 모두 mean0.16을 낸다. zero variance는 one-parameter two-action case에만 해당하며 b = E[R]가 일반적으로 minimum-variance constant가 아님을 보인다.

**LLM RL의 sample-based baseline.** prompt마다 G response y_1..y_G와 reward r_1..r_G가 있을 때:

| Sample i의 baseline | y_i에 depend하는가? | Expected gradient | Source |
|---|---|---|---|
| b = 0 | no | ∇J | [[vanilla-pg]] |
| 과거 reward의 moving average | no(과거 batch) | ∇J | [[rloo]] Eq.8 |
| learned value V_φ(x)(critic) | no | ∇J(variance는 V_φ error에 depend) | [[ppo]] Eq.9–12 |
| leave-one-out: (1/(G−1)) Σ_{j≠i} r_j | no | ∇J | [[rloo]] §2.3 |
| group mean (1/G) Σ_j r_j, r_i 포함 | yes | ((G−1)/G) ∇J | [[dr-grpo]] App.A |
| (r_i − group mean) / group std | yes | ∇J의 prompt-dependent multiple | [[reinforce-plus-plus]] App.A.1, [[dr-grpo]] §3.1 |

group-mean row는 r_i − mean = ((G − 1)/G)(r_i − mean_{−i})에서 온다. group-mean advantage는 leave-one-out advantage의(G − 1)/G배다. [[dr-grpo]] App.A는 같은 relation을 “(G/(G−1))·Ã equals the RLOO advantage”라고 말한다. constant factor는 smaller learning rate와 equivalent하다. std division은 constant가 아니다. std는 r_i와 group에서 correct인 response 수에 depend한다. [[reinforce-plus-plus]] App.A.1(Theorem1)은 resulting advantage가 finite N ≥ 2에 대해 biased임을 prove하고, [[dr-grpo]] §3.1은 effect를 difficulty bias라고 부른다. reward가 거의 모두1이거나 모두0인 question은 std가 작아서 larger weight를 받는다.

**Worked example: 0/1 reward의 group4개.** Prompt P1에는 correct response가 하나 있고 reward는(1,0,0,0)이다.

- b = 0: advantage(1,0,0,0).
- Group mean0.25: (0.75, −0.25, −0.25, −0.25).
- Leave-one-out: correct는1 − 0 = 1; wrong 각각은0 − 1/3 = −0.333. 3/4를 곱하면 group-mean value가 된다.
- n − 1 denominator을 쓰는 group std(PyTorch default이며 TRL의 `rewards.view(-1, num_generations).std(dim=1)`이 사용, [[trl-grpo]] L2127–2149): sqrt((0.75² + 3 × 0.25²)/3) = 0.5이므로 advantage는(1.5, −0.5, −0.5, −0.5)다.

Prompt P2에는 correct response가 두 개 있고 reward는(1,1,0,0), mean0.5, std sqrt(4 × 0.25/3) = 0.577이며 correct response의 advantage는0.5/0.577 = 0.866이다. std division 뒤 P1의 correct response는 weight1.5, P2의 correct response는0.866을 받는다. mean subtraction만 한 뒤 weight는0.75와0.5다.

group에 대해 expectation을 취하면(exact enumeration; source result가 아니라 이 course의 calculation), G = 8과 binary reward에서 std-normalized estimator는 success probability0.5인 prompt에 대해1.736 ∇J, success probability0.05 또는0.95인 prompt에 대해2.313 ∇J와 같다. relative weight는1.33이다. group-mean estimator는 every success probability에 대해0.875 ∇J다.

**Sample-based baseline의 variance.** other sample에서 estimate한 baseline은 자체 noise를 더한다. probability(0.5,0.3,0.15,0.05)의 response 네 개, response2와4가 correct(J = 0.35), G = 4인 one prompt에서 logit-gradient estimator의 exact total variance는 b = 0일 때0.0551, b = J일 때0.0233, leave-one-out일 때0.0414다. G = 2에서 leave-one-out은0.155로 b = 0의0.110보다 높다. response1,2,4가 correct(J = 0.85)이고 G = 4이면 b = 0은0.113, leave-one-out은0.034다. leave-one-out baseline은 expected reward가0에서 멀고 G가 작지 않을 때 variance를 가장 많이 줄인다.

**[figures/pg-variance.html](figures/pg-variance.html)** — reader가 G, response probability, correct set을 바꾸고 baseline별 advantage, ∇J의 multiple로서 exact expected gradient, exact variance를 읽을 수 있게 한다.

**근거.** TL;DR(Pythia-6.9B) 및 Anthropic-HH(Pythia-6.9B, Llama-7B)에서 reference completion 대비 GPT-4 win rate는 RLOO k = 4가77.9 / 43.7 / 64.1, moving-average baseline의 REINFORCE가70.7 / 37.9 / 55.3, PPO가67.6 / 29.2 / 32.0이었다([[rloo]] Table1; Result, single study; highest test reward checkpoint). Qwen2.5-32B-Instruct를 쓴 AppWorld에서 leave-one-out PPO variant는 std normalization 없이 task goal completion71.3, with std normalization61.9를 score했다([[loop-appworld]] Table1, best run; Result, single study).

**조건과 한계.** unbiasedness statement는 one prompt에서 J의 gradient에 대해 exact하다. clipping, length normalization, batch당 여러 update는 estimator를 더 바꾼다(ch-40). [[rloo]]의 win rate는 isolated baseline이 아니라 full method를 비교한다.

**General model에 대한 함의.** std normalization은 각 prompt가 update에 contribute하는 양을 바꾼다. mixed-difficulty 또는 multi-domain prompt set에서는 D를 바꾸지 않고 effective data mixture를 바꾼다.

## §3 Actor-critic, bootstrapping, trust region

**정의.** actor-critic method는 value function V_φ(s)(critic)를 학습하고 이를 baseline과 return bootstrap 양쪽에 사용한다. bootstrapping은 future reward를 value estimate로 replace한다.

**문제.** Monte-Carlo return은 unbiased지만 many step에서 reward를 받는 long trajectory에서는 summed reward 수와 함께 variance가 늘어난다.

**메커니즘과 formula.** [[ppo]]는 truncated generalized advantage estimation(GAE, Eq.11–12)을 사용한다.

```
δ_t = r_t + γ V_φ(s_{t+1}) − V_φ(s_t)
Â_t = δ_t + (γλ) δ_{t+1} + … + (γλ)^{T−t+1} δ_{T−1}
```

- r_t: step t의 reward; γ: discount; λ ∈ [0, 1]: GAE parameter; T: segment length.

λ = 1이면 estimate는 finite-horizon return에서 V_φ(s_t)를 뺀 것이며([[ppo]] §5) 어떤 V_φ에 대해서도 unbiased하다. λ = 0이면 one-step residual δ_t이며 V_φ ≠ V^π일 때 biased다.

**Worked example.** 3-token response가 t = 3에서 terminal reward1 하나를 받고 γ = 1이라고 하자. 아직 state를 distinguish하지 못하는 critic은 every state에0.6을 predict한다. λ = 1이면 Â_1 = 1 − 0.6 = 0.4다. λ = 0이면 Â_1 = 0 + V(s_2) − V(s_1) = 0.6 − 0.6 = 0이므로 response가 success했는데도 first token은 signal을 받지 않는다. bias는 sampling이 아니라 critic error에서 온다.

**Variance와 sequence length.** terminal reward가 있다고 해서 sequence-level estimator variance가 length-independent해지지는 않는다. estimator는 R · Σ_t s_t이고 s_t = ∇ log π(y_t | x, y_<t)다. score term은 conditional mean zero이므로 E[‖Σ_t s_t‖²] = Σ_t E[‖s_t‖²]이며 T와 함께 증가한다. bounded R은 이 증가를 제거하지 않는다.

**근거.** Anthropic-HH에서 Llama-7B로 PPO를 했을 때 λ를1.0에서0.95,0.5,0.0으로 낮추자 training reward가 monotonic하게 감소했다([[rloo]] §3.1, Fig.1; Result, single study). author는 pretrained initialization이 probability를 few token에 concentrate하므로 bias가 variance reduction만큼 가치 있지 않다고 해석한다(Interpretation). AppWorld에서 learned critic을 쓴 PPO는 λ_GAE ∈ {0.95,0.99,0.999}에서 diverge했고1.0에서 가장 stable했다([[loop-appworld]] App.C).

**Trust region.** batch 하나에 여러 gradient step을 적용하면 sample은 off-policy가 된다. [[trpo]]는 true return을 surrogate로 bound한다. η(π̃) ≥ L_π(π̃) − C · D_KL^max(π, π̃), C = 4εγ/(1 − γ)², ε = max |A_π|(Eq.9)이며, practice에서는 average KL constraint ≤ δ 아래 surrogate를 maximize하고 모든 experiment에서 δ = 0.01을 썼다(Eq.12–13, §8.1). [[ppo]]는 constraint를 clipped ratio(Eq.7)로 replace한다. 둘 다 consecutive policy 사이 step을 control한다. RLHF에서 fixed reference policy로의 KL은 다른 quantity다(ch-38).

**조건과 한계.** target network와 double 또는 clipped-double Q-learning은 value function을 자기 estimate로 bootstrap할 때의 instability와 overestimation을 다룬다. critic을 train할 때 relevant하고 critic-free method에는 irrelevant하며, stochastic environment transition을 다루는 것은 아니다.

**General model에 대한 함의.** prompt distribution 하나에서 train한 critic은 generalize해야 하는 또 하나의 model이다. critic-free estimator는 그 dependency를 제거하지만 D에 대한 dependency는 제거하지 않는다.

## §4 Baseline이 각 update의 sign을 정한다

**정의.** sample의 advantage R − b가 negative이면 sample은 pushed down된다(그 log-probability가 negative weight를 받는다).

**문제.** training method가 negative gradient를 쓰는지는 흔히 algorithm의 property로 설명된다. policy-gradient family에서는 reward coding과 baseline이 이를 정한다.

**메커니즘.**

1. reward가 {0,1}이고 b = 0이면 every advantage는0 또는1이다. incorrect sample은 zero weight를 받는다. update는 correct self-generated sample에 대한 SFT다. [[rls-razor]] §5.1은 이를 “1–0 Reinforce”라고 부르고 “is equivalent to sampling from the model and performing SFT on correct answers only”라고 말한다. sample이 fresh할 때 이것은 rejection-sampling fine-tuning objective다([[raft-reinforce-rej-minimalist]] RAFT, Eq.1; ch-31).
2. reward가 {−1,+1}이고 b = 0이면 incorrect sample의 advantage는−1이다. [[raft-reinforce-rej-minimalist]] §5는 이를 “fine-tuning on the positive samples and unlearning on the negative samples”로 설명한다.
3. group의 lowest reward와 highest reward 사이에 있는 어떤 baseline도 low-reward sample에 negative advantage를 준다. §2 example에서 group mean은 zero-weight sample 세 개를 −0.25 sample 세 개로 바꾼다.
4. reward design은 baseline 없이도 negative를 만들 수 있다. GeneralPoints는 failure 종류에 따라 r = −1, −2, −3을 준다([[sft-memorizes-rl-generalizes]] App.A.3).

**Positive weight와 negative weight의 비대칭.** term A log π(y) 하나와 logit z 위의 softmax를 생각하자.

```
∂ log π(y) / ∂ z_j = 1[j = y] − π(j)
```

- A > 0이면 term은 π(y) = 1에서 maximize되고 그때 log π(y) = 0이다. z_y의 gradient는 A(1 − π(y))이며 π(y) → 1일수록0으로 shrink한다. positive update는 saturate한다.
- A < 0이면 π(y) → 0일 때 term A log π(y) → +∞다. z_y의 gradient는 A(1 − π(y)) → A이며 shrink하지 않는다. negative update는 π(y) = 0 이외에는 fixed point가 없다.

**근거.**

- Qwen2.5-Math-7B-base 및 LLaMA-3.2-3B-instruct으로 한 math(MATH500, Minerva Math, OlympiadBench의 average@16)에서 RAFT++(positive만)은56.1과27.6, GRPO는56.3과28.4, ±1 reward의 REINFORCE는53.9와24.2에 도달했다([[raft-reinforce-rej-minimalist]] Table1; Result, single study, seed 보고 없음). RAFT++ entropy는 GRPO보다 더 빨리 떨어졌고 GRPO는 training 후반에 이를 추월했다(§5.1, Fig.2–3).
- Qwen2.5 3B-Instruct의 Science Q&A에서1–0 Reinforce는 learning–forgetting trade-off에서 GRPO와 같았고, SimPO(offline, negative 포함)는 SFT와 같았다([[rls-razor]] §5.1, Fig.4; Result, single study, value는 plot으로 제공).

**조건과 한계.** expectation에서는 sign pattern이 gradient를 바꾸지 않는다(§2). finite-sample update, clipping과의 interaction, entropy trajectory는 바꾼다. 위 두 study는 서로 다른 outcome(benchmark accuracy와 entropy 대 forgetting)을 측정하며, 어느 쪽도 improvement 중 negative에서 온 share를 측정하지 않는다.

**General model에 대한 함의.** mixed reward의 every group에서 negative advantage를 만드는 baseline은 step마다 many sample에 unbounded push-down을 적용한다. 이를 control하는 방법은 “음성 샘플과 음성 피드백”에 있다.

## §5 Model 자신의 sample에 대한 SFT로서의 RL gradient

**정의.** target distribution π_β에 대한 SFT loss는 L_SFT = −E_{x∼D, y∼π_β}[log π_θ(y|x)]다. policy-gradient loss는 L_RL = −E_{x∼D, y∼π_θ}[A(x, y) log π_θ(y|x)]이며 gradient는 log π_θ를 통해서만 취한다([[rls-razor]] §5.1).

**문제.** new task에서 fine-tuning하면 earlier task performance가 낮아질 수 있다(forgetting). new-task accuracy가 같을 때 training method가 forgetting 양에 영향을 주는지가 question이다.

**메커니즘.** 두 loss는 두 곳에서 다르다. sampling distribution(π_θ 대 fixed external target)과 weight(negative일 수 있는 A 대1)다. RL update는 current model에서 이미 non-negligible probability가 있는 response의 probability를 높인다. SFT는 model이 sample하지 않을 response로 probability를 이동시킬 수 있다.

**Formula(idealized case).** binary reward와 finite set 위 base policy p에서 fully correct distribution 중 p에 KL로 가장 가까운 distribution은 correct response로 restrict한 p를 renormalize한 것이고, p에서의 KL은 −log P_p(correct)다([[rls-razor]] App.A, LemmaA.1). [[rls-razor]] Theorem5.2는 regularity condition과 convex policy family 아래 π_0에서 시작한 policy gradient가 argmin_{π optimal} KL(π ‖ π_0)로 converge한다고 말한다.

**Worked example.** base model이 answer 네 개에 probability(0.5,0.3,0.15,0.05)를 주고 answer2와4가 correct라면 P(correct) = 0.35다.

- KL-minimal correct policy: (0,0.3/0.35,0,0.05/0.35) = (0,0.857,0,0.143). base에 대한 KL = −log0.35 = 1.05 nat.
- annotation이 언제나 answer4를 쓰는 SFT: target(0,0,0,1). base에 대한 KL = −log0.05 = 3.00 nat.

두 policy 모두 fully correct지만 SFT target은 KL에서2.85× 더 멀다.

**근거.**

- Qwen2.5 3B-Instruct를 math(Open-Reasoner-Zero question), SciKnowEval의 Chemistry L-3, ToolAlpaca로 train하고 GRPO(binary reward, KL term 없음)와 SFT를 hyperparameter sweep에서 비교했다. matched new-task accuracy에서 RL은 prior-task score(HellaSwag, TruthfulQA, MMLU, IFEval, WinoGrande, HumanEval)를 거의 바꾸지 않은 반면 SFT는 이를 낮췄고 math에서 가장 강했다([[rls-razor]] §3.1, Fig.2; Result, single study, value는 plot으로 제공). forgetting은 new task의 fine-tuned policy와 base policy 사이 KL로 predict됐으며 LLM experiment에서 quadratic fit R² = 0.71이었다(§4, Fig.11).
- paper의 toy setting(ParityMNIST, 3-layer MLP)에서 analytically constructed KL-minimal fully correct distribution의 SFT는 RL보다 덜 forgot했다([[rls-razor]] §4, Fig.3; construction은 base model의 full output distribution을 알아야 하며 LLM task에서는 run하지 않았다).
- Llama-3.2-Vision-11B를 SFT-initialize한 뒤 PPO 또는 further SFT를 했다. unseen face-card rule의 GeneralPoints에서 OOD success는11.5%에서 RL 뒤15.0%, SFT 뒤3.4%가 됐다. unseen action space의 V-IRL-L에서는80.8%에서 RL 뒤91.8%, SFT 뒤1.3%가 됐다([[sft-memorizes-rl-generalizes]] §5.1; Result, single study). SFT initialization 없이 end-to-end RL을 하면 base model output을 reward용으로 parse할 수 없어서 improve하지 못했다(§5.4).
- replicated direction: 두 study 모두 external target에서 RL이 SFT보다 retention 또는 transfer가 좋다고 찾았다. model, task, metric(prior benchmark forgetting 대 rule-variant OOD accuracy)은 서로 다르다.

**조건과 한계.** 두 study는 각각 base model 하나를 쓴다. SFT target은 다른 model([[rls-razor]] App.B.1의 DeepSeek-R1 또는 GPT-4o) 또는 task demonstration이 생성했다. self-generated target에서 SFT는 다른 case다(ch-31). [[rls-razor]] §7은 “we still lack a mechanistic account of why larger KL shifts on the new task disrupt prior knowledge”라고 말한다. ch-30a에는 KL–forgetting relation이 항상 hold하지 않는 report가 기록돼 있다(Open question).

**General model에 대한 함의.** on-policy sampling은 그 자체로 forgetting control이다. model이 전혀 sample하지 못하는 new capability는 on-policy RL로 도달할 수 없으며, 그런 case에는 먼저 SFT 또는 distillation이 필요하다(ch-31, ch-38a).

## §6 Reverse-KL-regularized reward, entropy, diversity

**정의.** RLHF는 reference policy π_ref(보통 SFT model)에 대한 KL penalty를 뺀 reward를 maximize한다.

```
max_π E_{x∼D} [ E_{y∼π}[R(x, y)] − β KL(π(·|x) ‖ π_ref(·|x)) ]
π*(y|x) = π_ref(y|x) exp(R(x, y)/β) / Z(x)
```

- β > 0: KL coefficient; Z(x) = Σ_y π_ref(y|x) exp(R(x, y)/β). 두 번째 줄은 objective를 −β KL(π ‖ π*) + β log Z로 써서 얻는 maximizer다. [[maximum-entropy-rl]] Eq.4에는 R 대신 Q를 쓰고 reference policy는 없는 same exponential target이 있다(relation의 Interpretation).

**“reverse”와 mode-seeking인 이유.** KL(π ‖ π_ref) = Σ_y π(y) log(π(y)/π_ref(y))는 log-ratio를 π로 weight한다. π_ref가 small인 곳에 mass를 놓으면 큰 cost가 든다. π_ref가 support하는 response에서 mass를 remove하면 해당 term은 π(y) = 0을 곱하므로 cost가 없다. 따라서 penalty는 policy가 response를 drop하는 것을 허용한다.

**Worked example.** π_ref = (0.5,0.3,0.2), reward(1,0,0)이다. β = 2이면 π* = (0.623,0.227,0.151), entropy0.917 nat(π_ref:1.030), KL0.030이다. β = 0.5이면 π* = (0.881,0.072,0.048), entropy0.446, KL0.328이다. zero-reward response 둘 사이 ratio는 두 case 모두1.5로 유지된다. exact optimum은 equal-reward response 사이에서 π_ref shape를 보존하고 higher-reward response에 mass를 concentrate한다.

**KL은 entropy의 lower bound를 주지 않는다.** KL(π ‖ π_ref) = −H(π) + CE(π, π_ref)이고 CE(π, π_ref) = −Σ_y π(y) log π_ref(y)다. KL ≥ 0은 H(π) ≤ CE(π, π_ref), 즉 upper bound를 준다. example으로 π_ref = (0.9,0.1), π = (1,0)라면 KL = log(1/0.9) = 0.105 nat이고 H(π) = 0이다. peaked π_ref는 low KL cost에서 collapse를 허용한다.

**Diversity에 관한 근거.**

- LLaMA7B에서 β_KL = 0.05의 PPO는 SFT보다 out of distribution에서 더 generalize했고 “significantly reduces output diversity compared to SFT across a variety of measures”였다([[rlhf-generalisation-diversity]] Abstract). OPT-6.7B summarisation에서 input별 EAD는 RLHF0.07, SFT0.79였고(App.J.4 Table11), input 간 EAD는 둘 다0.87이었다(Table12).
- KL coefficient를 올리면 performance도 낮아지고 input별 diversity도 낮아졌다([[rlhf-generalisation-diversity]] §6.3, App.I; Result, single study). 그 setting에서는 larger β가 diversity를 recover하지 못했다.
- math RLVR run에서 entropy loss L − αH(π_θ)를 추가하면 α = 0.0001 또는0.001은 entropy에 minor effect, α = 0.01은 entropy explosion, α = 0.005는 other baseline보다 outperform하지 않으면서 entropy를 stabilize했다. reference-KL coefficient0.001–0.1은 entropy를 stabilize했지만 accuracy를 낮췄다([[entropy-mechanism-llm-rl]] §4.1, Fig.9–10; 이 두 sweep에 쓴 model은 stated되지 않음). softmax의 vanilla policy gradient에서 entropy change는 ≈ −η Cov(log π(a), π(a) A(a))(Theorem1)다. 이미 probable한 action을 positive advantage로 올리면 entropy가 낮아진다.

**Classical RL의 entropy bonus.** [[ppo]] Eq.9는 entropy term c_2 S[π_θ](s_t)를 포함한다. MuJoCo experiment는 entropy bonus를 사용하지 않았고(§6.1), c_2 = 0.01은 Atari value다(Table5). [[maximum-entropy-rl]]는 objective에 entropy를 넣으며 small reward scale은 near-uniform policy를, large scale은 poor local minimum에 stuck한 near-deterministic policy를 만들었다고 report한다(§5.2, Fig.3b).

**Practice의 KL은 single-sample estimate다.** implementation은 full-vocabulary KL을 compute하지 않는다. PPO-style RLHF는 sampled token의 token별 log-ratio log(π/π_ref)를 reward에 더하고([[reinforce-plus-plus]] Eq.4), GRPO는 sampled token의 π_ref/π − log(π_ref/π) − 1을 loss에 더한다([[grpo]] Eq.4). 이들은 [[john-schulman-kl-tricks]]의 k1과 k3 estimator다. ch-38과 ch-43이 이를 비교한다.

**pass@k: lost coverage를 detect하는 metric.** 정의: k sample 중 적어도 하나가 correct면 problem은 k에서 solved다. n ≥ k sample 중 c개가 correct이면 unbiased estimator([[rlvr-beyond-base-model]] App.A.2)는 다음이다.

```
pass@k = E_x [ 1 − C(n − c, k) / C(n, k) ]
```

Worked example: n = 10, c = 2, k = 5면 1 − C(8,5)/C(10,5) = 1 − 56/252 = 0.778이다.

근거: Omni-MATH-Rule problem2,000개에서 GRPO로 train한 Qwen2.5-7B는 MATH500 pass@1을34.5에서74.4로 올린 반면 pass@256은96.2에서97.2가 됐다. in-domain Omni-MATH test의 pass@256은69.1에서68.3으로 갔다([[rlvr-beyond-base-model]] Table3). 더 긴 GRPO training은 Omni-MATH-Train pass@1을 step150의26.1에서 step450의42.5로 올리는 동안 pass@256은66.3에서64.3으로 내렸다(Table4). k = 1024의 AIME24에서는 base model이 solve했지만 SimpleRLZoo RL model은 solve하지 못한 problem이13.3%였고 반대 방향은0.0%였다(Table2). author는 RLVR을 base model이 이미 sample할 수 있던 path의 probability를 올리는 것으로 해석한다(Interpretation). [[reinforce-plus-plus]] v9 Table2는 AIME-24 question30개로 train한 GRPO가 train pass@1 95.0, AIME-25 pass@16 0.4에 도달했다고 report한다(model stated 안 됨; Result, single study).

**General model에 대한 함의.** reward와 함께 entropy, input별 diversity, large k에서 pass@k를 report한다. pass@1은 오르지만 pass@k가 flat 또는 falling이면 new capability가 아니라 concentration을 나타낸다.

## §7 Environment가 token을 반환할 때: agentic exception

**정의.** agentic RL에서 model은 action(text 또는 tool call)을 emit하고 environment는 observation(tool output 또는 verifier message)을 반환하며 next action 전에 observation이 context에 append된다.

**문제.** single-turn setting은 deterministic transition을 갖는다. next state는 prefix와 sampled token을 더한 것이다. tool이 있으면 next state는 stochastic 또는 stateful일 수 있는 environment에 depend한다. observation token은 sequence에 있지만 π_θ에서 sample된 것은 아니다.

**메커니즘.**

1. trajectory probability는 p(τ) = Π_t π_θ(a_t | s_t) · P(s_{t+1} | s_t, a_t)다. transition factor P에는 θ가 없으므로 log-derivative가0이며 §1 estimator는 unbiased로 남는다.
2. action token만 Σ log π_θ에 속한다. [[loop-appworld]]는 AppWorld를 environment token이 state에 append되지만 LLM-emitted token만 likelihood에 들어가는 POMDP로 formulate한다(§4.1, Eq.6–8). TRL은 tool-result token에서 `tool_mask`를0으로 두고 loss와 importance ratio에서 제외한다([[trl-grpo]] L1472, L1638–1648, L2425).
3. observation token을 loss에 넣으면 environment text에 weight A를 주는 SFT term이 더해지며 이는 ∇J의 일부가 아니다.
4. environment randomness는 prompt-level baseline이 remove할 수 없는 reward variance를 더하고, terminal reward는 all turn이 share한다(credit assignment).

**근거.** AppWorld(Qwen2.5-32B-Instruct, LoRA, training task72개)에서 token별 importance weight는 task goal completion71.3, turn별은64.1, trajectory별은53.3을 score했다([[loop-appworld]] Table1, best run; Table2의 three-run mean도 order를 유지). GeneralPoints에서는 verifier text가 prompt에 append된다(sequential revision). same compute에서 verification step이 많을수록 OOD gain이 더 컸다. 1 step +0.48%, 3 step +2.15%, 5 step +2.99%, 10 step +5.99%였다([[sft-memorizes-rl-generalizes]] §3, §5.5).

**조건과 한계.** AppWorld에는 non-determinism, transient failure, unsolvable task가 없다([[loop-appworld]] §6). critic, turn-level value, observation handling은 ch-45b에서 다룬다.

**General model에 대한 함의.** tool output은 environment 사이에서 vary한다. tool environment 하나에서 train한 policy는 다른 environment에서 evaluate된다(AppWorld의 Test-Challenge는 unseen app을 포함하며 LOOP 뒤21.0 → 45.7, [[loop-appworld]] Table1).

## 음성 샘플과 음성 피드백

이 section은 course standard의 “negative” 네 meaning을 사용한다. policy gradient는 meaning(4), **gradient로서의 negative**를 사용한다. negative advantage는 sample log-probability를 낮춘다. b = 0인 rejection sampling은 **discard** option을 쓴다.

**Negative가 오는 곳.** verifier([[raft-reinforce-rej-minimalist]]와 [[dr-grpo]]의 Math-Verify), baseline 아래의 reward model score([[rloo]]), fraction으로 passed한 unit test([[loop-appworld]] App.D), shaped environment penalty([[sft-memorizes-rl-generalizes]] App.A.3)다. 이 source들은 false-negative rate를 report하지 않는다. [[rlvr-beyond-base-model]] §2.2는 math에서 large k의 reverse error를 언급한다. wrong chain of thought도 correct answer에 도달할 수 있다.

**Current practice가 하는 일.** RAFT/RAFT++와1–0 Reinforce는 failure를 discard한다. ±1 reward의 REINFORCE, RLOO, GRPO, PPO는 failure를 push down한다. Reinforce-Rej는 response가 all correct 또는 all incorrect인 prompt를 discard하고 mixed group에서는 failure를 push down한다([[raft-reinforce-rej-minimalist]] §5).

**메커니즘.** advantage A < 0, step η인 sampled token y 하나에 대해 logit update는 Δz_k = η A (1[k = y] − p_k)다. first-order probability change는 다음과 같다.

```
Δp_j ≈ η A p_j (1[j = y] − p_j − p_y + Σ_k p_k²)
```

unsampled j에서 이것은 p_j > Σ_k p_k² − p_y일 때만 positive다. Worked example: p = (0.6,0.3,0.1), sampled y = token3, A = −1, η = 1이다. threshold는0.46 − 0.1 = 0.36이다. update 뒤 exact softmax는(0.710,0.263,0.026)이다. token3은0.074를 잃고 token1은0.110을 얻으며 token2는0.037을 잃는다. unlikely sample에서 remove한 mass는 most likely alternative로 이동하며 other alternative도 mass를 잃을 수 있다. [figures/pg-variance.html](figures/pg-variance.html)의 PanelC는 다른 probability에서도 이를 반복하게 한다. ch-39와 ch-43a는 이 effect(squeezing, likelihood displacement)를 전개한다.

**Benefit과 failure mode의 근거.**

- Benefit: GRPO는 training 후반에 RAFT++를 추월했고 RAFT++ entropy는 더 빨리 낮아졌다. author는 negative가 exploration 유지에 도움 된다고 conclude한다(Interpretation; [[raft-reinforce-rej-minimalist]] §5.1).
- Filtering 없는 cost: ±1 reward REINFORCE는 two model에서53.9와24.2를 score했으며 Reinforce-Rej의56.4와28.5보다 낮았다(Table1). LLaMA-3.2-3B-instruct ablation에서 response가 all wrong인 prompt를 remove하면 vanilla REINFORCE 대비 largest reward gain이 났고, all-correct prompt 제거는 거의 바꾸지 않았다(Fig.4 및 caption).
- Forgetting: [[rls-razor]] §5.1에서 RL과 SFT를 분리한 것은 negative가 아니라 on-policy data였다.
- Negative에 attributable한 improvement share: 여기 cite한 어떤 source도 측정하지 않았다.

**통제.**

1. sample을 on-policy로 유지하거나 ratio를 bound한다. PPO는 Â < 0일 때 ratio를1 − ε에서 clip하므로 probability를(1 − ε)π_old보다 낮춰도 further gain이 없다([[ppo]] §3, Fig.1).
2. all reward가 equal할 때 zero advantage를 내는 baseline(group mean 또는 leave-one-out)을 쓴다. 그러면 every sample이 failure인 group은 push-down을 만들지 않는다. TRL에서 mean subtraction은 이미 그런 group에0을 주며 division은 finite하게 유지되도록 group std에1 × 10⁻⁴를 더한다([[trl-grpo]] L2147–2149). zero-std flag 자체는 log하며 advantage를 zero로 만드는 데 쓰지 않는다(L2150, L2186).
3. KL term 또는 positive NLL term으로 anchor한다(ch-38, ch-43a).
4. uninformative group을 filter 또는 down-weight한다(Reinforce-Rej).

**진단.** A > 0 sample과 A < 0 sample에 대해 mean advantage와 mean log-probability를 separately log한다. policy entropy를 log하고, zero-std group fraction([[trl-grpo]]의 `frac_reward_zero_std`)을 log하며, held-out prompt에서 pass@1과 large k의 pass@k를 report한다.

**Generality에 미치는 효과.** negative는 entropy를 preserve할 수 있고([[raft-reinforce-rej-minimalist]]) 이는 coverage와 관련된다. unbounded push-down은 most likely alternative에 mass를 concentrate하며 diversity와 관련된다. calibration, hallucination, over-refusal에 미치는 effect는 이 source들이 측정하지 않는다.

## 레시피

Row는 cited source가 명시한 policy-gradient setting을 quote한다. locus는 표시된 date에 확인했다.

| Model(정확한 release) | Size | Stage | Setting | Value | Source 위치 | Status | 이 value의 근거 |
|---|---|---|---|---|---|---|---|
| PPO-Clip, MuJoCo1M-timestep benchmark | MLP2×64 | RL | entropy coefficient c_2 | 사용 안 함(entropy bonus 없음) | arXiv:1707.06347v2 §6.1([[ppo]]) | verified 2026-09-14 | not applicable |
| PPO-Clip, MuJoCo | MLP2×64 | RL | clip ε | 0.2 | arXiv:1707.06347v2 §6.1 Table1 | verified 2026-09-14 | Table1:0.82 대0.76(ε = 0.1),0.70(ε = 0.3); task7개 × seed3개 |
| PPO-Clip, Atari(게임49개) | not reported | RL | c_1; c_2 | 1;0.01 | arXiv:1707.06347v2 App.A Table5 | verified 2026-09-14 | ablation 보고 없음 |
| TRPO experiment | not applicable | RL | KL step size δ | 0.01(“for all experiments”) | arXiv:1502.05477 §8.1([[trpo]]) | verified 2026-09-15 | ablation 보고 없음 |
| RLOO / PPO / RAFT run, TL;DR | Pythia-6.9B | RL | β(reward의 KL); rollout batch; step batch; step | 0.03;512;256;600 | arXiv:2402.14740v2 App.C([[rloo]]) | verified 2026-09-15 | ablation 보고 없음 |
| same, Anthropic-HH | Pythia-6.9B | RL | β; step | 0.10;393 | arXiv:2402.14740v2 App.C | verified 2026-09-15 | ablation 보고 없음 |
| same, all dataset | Pythia-6.9B, Llama-7B | RL | LR; warm-up; batch당 gradient step; k | constant1 × 10⁻⁶;3% linear;2;2 또는4 | App.C; Table1 | verified 2026-09-15 | LR sweep {1e-6,1e-5,2e-5}(RAFT,RLOO) 및 {1e-6,1e-5}(PPO,Vanilla PG); Table1 k = 4 대k = 2 |
| forgetting study의 GRPO | Qwen2.5 3B-Instruct | RL | KL coefficient; group size; generation당 prompt; μ; loss type | 0;64;8;{1,2}; Dr.GRPO | arXiv:2509.04259v1 App.B Table2([[rls-razor]]) | verified 2026-09-15 | sweep의 Pareto frontier(Fig.2) |
| same | Qwen2.5 3B-Instruct | RL | LR sweep; schedule; warm-up; epoch | {1e-5 … 5e-5}; warm-up 포함 constant;50 step;1 | App.B Table2 | verified 2026-09-15 | sweep의 Pareto frontier |
| diversity study의 RLHF(PPO) | LLaMA7B | RL | β_KL(reward의 KL) | 0.05 | arXiv:2310.06452v3 §4 Eq.1([[rlhf-generalisation-diversity]]) | verified 2026-09-15 | App.I sweep: higher β가 input별 diversity와 performance를 낮춤 |
| RAFT, RAFT++, GRPO, Reinforce-Rej | Qwen2.5-Math-7B-base, LLaMA-3.2-3B-instruct | RL | iteration당 prompt; prompt당 response; mini-batch; LR; max token | 1,024;4;512;1 × 10⁻⁶;4,096 | arXiv:2504.11343v2 §4([[raft-reinforce-rej-minimalist]]) | verified 2026-09-15 | Table1 caption: batch, mini-batch, actor LR은 algorithm별 tune됐고 v2에 없는 appendix를 refer하므로 algorithm별 value 없음 |
| GRPO / PPO / RLOO comparison | Qwen2.5-7B(base) | RL | KL; LR; step당 prompt; prompt당 response; max length; temperature | removed; constant1 × 10⁻⁶;256;8;8,192;1.0 | arXiv:2504.13837v5 §4.3([[rlvr-beyond-base-model]]) | verified 2026-09-15 | ablation 보고 없음 |
| pass@k evaluation, same study | Qwen2.5 및 LLaMA-3.1-8B | eval-gate | n sample; temperature; top-p; max token | 128 또는1,024(plotted largest k);0.6;0.95;16,384 | arXiv:2504.13837v5 §3, App.A.2 | verified 2026-09-15 | not applicable |
| Dr.GRPO run(Oat-Zero) | 1.5B–7B | RL | question당 response; temperature; KL coefficient; clip ε; LR | 8;1.0;0.0;0.2;constant1 × 10⁻⁶ | arXiv:2503.20783v2 App.G Table6([[dr-grpo]]) | verified 2026-09-14 | ablation 보고 없음 |
| Entropy-loss sweep(model stated 안 됨) | not reported(§4.1, Fig.9 caption, §2.2 확인) | RL | L − αH의 entropy-loss coefficient α | 0.0001,0.001 minor effect;0.005 stable, not better;0.01 entropy explosion | arXiv:2505.22617v1 §4.1 Fig.9([[entropy-mechanism-llm-rl]]) | verified 2026-09-14 | Fig.9 |

**작은 general-purpose run의 시작점.** 아래 value는 각각 stated condition에서 verified된 위 row 하나에서 왔다. verifiable reward를 쓰는3B–7B model에는 constant LR1 × 10⁻⁶([[rlvr-beyond-base-model]] 및 [[raft-reinforce-rej-minimalist]]의 Qwen2.5-7B, Qwen2.5-Math-7B run), prompt당 response4–8개(same row), rollout temperature1.0([[rlvr-beyond-base-model]], [[dr-grpo]]), entropy bonus 없음(PPO MuJoCo; entropy-loss sweep이 gain을 찾지 못함), leave-one-out 또는 mean-only baseline(§2)에서 시작한다. KL coefficient는 goal에 depend한다. 위 reasoning 및 forgetting study에서는0, [[rloo]]의 RLHF reward-model run에서는0.03–0.10을 썼다. held-out prompt에서 n ≥ 128로 pass@1과 pass@k를 evaluate한다. 이 run 중 어느 것도 multi-domain general model에서 이 value를 test하지 않았다.

## 일반화 관점

**(a) breadth를 늘리는 것.**

- On-policy update: matched new-task accuracy에서 RL은 external target SFT보다 덜 forgot했고([[rls-razor]] Fig.2), SFT가 하지 못한 unseen rule과 visual variant로 transfer했다([[sft-memorizes-rl-generalizes]] §5.1–5.2). Result는 두 study에서 direction이 replicate됐다.
- episode당 더 많은 environment feedback: equal compute에서 verification step이 많을수록 OOD gain이 컸다([[sft-memorizes-rl-generalizes]] §5.5).
- RLHF의 out-of-distribution robustness: 특히 distribution shift가 커질수록 SFT보다 OOD preference score가 좋았다([[rlhf-generalisation-diversity]] Abstract, §6.1).

**(b) narrowing 또는 forgetting을 일으키는 것.**

- RLHF 뒤 input별 diversity 감소([[rlhf-generalisation-diversity]] Table11); larger KL coefficient도 이를 recover하지 못했다(§6.3).
- RLVR 뒤 및 longer training에서 large-k coverage 감소([[rlvr-beyond-base-model]] Table3–4).
- small prompt set에 overfit: AIME-24 question30개에서 GRPO는 train pass@1 95.0, AIME-25 pass@16 0.4에 도달했다([[reinforce-plus-plus]] v9 Table2).
- std normalization으로 prompt를 implicit reweight(§2)하여 effective mixture를 바꾸는 것.
- D에 없는 prompt type은 gradient를 받지 않는다(§1).

**(c) 이 stage에서 측정하는 방법.** 이 track의 every RL run에 evaluation split 세 개를 쓴다.

1. **Training reward**: optimize 중인 prompt에서 측정한다(optimization failure는 detect하지만 generality는 detect하지 않는다).
2. **Training distribution의 held-out prompt**: n ≥ 128에서 pass@1과 pass@k로 측정한다(overfitting과 coverage loss detect).
3. **Other domain 및 prior-capability benchmark의 prompt**: 예를 들어 [[rls-razor]] §3.1의 benchmark6개에 input별 diversity를 더한다(forgetting과 narrowing detect).

알려진 measurement error: math의 large-k pass@k는 lucky guess를 count하고([[rlvr-beyond-base-model]] §2.2), diversity metric은 long instruction-following output에서 model을 separate하지 못했으며([[rlhf-generalisation-diversity]] §6.2), [[rloo]]의 LLM judge win rate는 human preference와 check되지 않았다(Limitations).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 검사 |
|---|---|---|
| group-mean 또는 group-std advantage를 unbiased로 취급 | group size에 따라 effective learning rate가 변하고 easy/hard prompt가 update를 dominate | expected scale(G−1)/G와 prompt별 std weight를 compute하고 leave-one-out과 비교(figure panelB) |
| KL penalty가 entropy collapse를 막는다고 가정 | reference KL은 small한데 entropy가 fall | H(π)와 CE(π, π_ref)를 separately log; training prompt에서 π_ref entropy 확인 |
| entropy bonus를 default로 추가 | accuracy gain 없이 entropy가 rise하거나 bound 없이 rise | held-out split에서 coefficient sweep; c_2 = 0과 비교([[entropy-mechanism-llm-rl]] Fig.9) |
| tool 또는 observation token을 policy loss에 포함 | model이 tool output text를 reproduce; model이 generate하지 않은 token에서 loss 감소 | observation token에서 loss mask가0인지 verify([[trl-grpo]] `tool_mask`) |
| training reward 또는 pass@1만 보고 | training prompt gain인데 held-out prompt의 pass@k는 flat 또는 falling | Generalization lens의 three split report |
| filtering/anchor 없는 ±1 reward 사용 | rejected sample log-probability가 빠르게 fall; unrelated token probability가 drop | A > 0과 A < 0에서 log π를 separately log; entropy track |
| “terminal reward이므로 estimator variance는 bounded”라고 말함 | response length와 함께 gradient norm variance 증가 | response-length bin에 대해 gradient-norm variance plot |
| untrained critic으로 bootstrapping(λ < 1) | early reward가 stall 또는 diverge | [[rloo]] Fig.1, [[loop-appworld]] App.C처럼 λ = 1과 lower λ 비교 |

## 이해 점검

1. §1의 two-action example에서 b = 0.8이 zero variance를 주는 이유와 possible answer가 세 개이면 왜 성립하지 않는지 설명하라.
2. group-mean baseline은 expected gradient에(G − 1)/G를 곱한다. one prompt에서는 harmless하지만 different difficulty의 prompt들 사이에서는 std normalization이 harmless하지 않은 이유를 설명하라.
3. 0/1 reward와 b = 0에서는 sample이 push down되지 않는다. expected gradient가 leave-one-out baseline 때와 같은 이유와 training에서 바뀌는 것을 설명하라.
4. first-order Δp formula를 사용해 이미 unlikely한 token의 negative advantage가 moderately likely token probability도 낮출 수 있는 이유를 설명하라.
5. “sample은 π_θ에서 온다”에서 “SFT보다 less forgetting”으로 이어지는 causal chain을 설명하고, [[rls-razor]]에서 negative gradient가 아니라 on-policy data가 factor라는 evidence를 말하라.
6. entropy가 collapse한 run에서 KL(π ‖ π_ref)이 small할 수 있는 이유를 설명하라.
7. run에서 pass@1은30에서45로 rise했지만 pass@256은70에서66으로 fall했다. model이 gained한 것과 lost한 것을 말하고 next evaluation split을 말하라.
8. tool-use trajectory에서 transition probability는 policy gradient를 bias하지 않지만 observation token은 여전히 mask해야 하는 이유를 설명하라.

## 연결

- **이전: ch-41 — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization.** R(x,y)를 제공하며 reward-model error는 advantage error로 estimator에 들어간다.
- **다음: ch-38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax.** clipped surrogate, reward의 per-token KL, alignment-tax measurement를 더한다.
- **ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.** [[rls-razor]], [[sft-memorizes-rl-generalizes]], [[rlhf-generalisation-diversity]]의 full treatment.
- **ch-39 — Offline Preference Optimization: DPO and Its Variants.** sampling 없이 §6의 KL-regularized optimum을 사용.
- **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO.** clipping과 length normalization으로 §2 baseline 구현.
- **ch-16 — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse.** §1의 D를 design.
- **ch-43 — Entropy, Output Diversity, and KL Control in RL.** §6을 계속함.
- **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.** negative-feedback section을 계속함.
- **ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability.** §7을 계속함.
- **ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control** 및 **ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation.** §4–§5의 background.

## 출처

- [[vanilla-pg]] — reinforcement baseline이 있는 REINFORCE update(§1–§2).
- [[rloo]] — sequence-as-action form, leave-one-out estimator, λ sweep, win rate, training setting(§1–§3, Recipe).
- [[trpo]] — improvement bound, KL-constrained surrogate, δ = 0.01(§3, Recipe).
- [[ppo]] — GAE, clipped surrogate, domain별 entropy coefficient(§3, §6, negative-feedback control, Recipe).
- [[maximum-entropy-rl]] — maximum-entropy objective와 reward-scale sensitivity(§6).
- [[rls-razor]] — on-policy update 및 forgetting,1–0 Reinforce 대 GRPO 대 SimPO, KL-minimal theory(§4–§5, Recipe).
- [[sft-memorizes-rl-generalizes]] — RL 대 SFT 뒤 rule/visual OOD, sequential revision, verification step, reward design(§4, §5, §7).
- [[rlhf-generalisation-diversity]] — RLHF generalization 및 diversity, KL sweep(§6, Generalization lens).
- [[rlvr-beyond-base-model]] — pass@k estimator, pass@1 대 pass@256 table(§6, Recipe).
- [[raft-reinforce-rej-minimalist]] — positives-only 대 ±1 REINFORCE 대 GRPO, entropy, prompt filtering(§4, negative-feedback section, Recipe).
- [[reinforce-plus-plus]] — GRPO advantage bias, reward의 KL log-ratio, small-set overfitting(§2, §6).
- [[dr-grpo]] — RLOO와의(G−1)/G relation, difficulty bias, run setting(§2, Recipe).
- [[grpo]] — GRPO advantage 및 loss의 k3 KL term(§6).
- [[entropy-mechanism-llm-rl]] — policy gradient 아래 entropy change, entropy-loss와 KL sweep(§6, Recipe).
- [[loop-appworld]] — observation token이 있는 POMDP, importance-weight granularity, std normalization cost(§2, §3, §7).
- [[trl-grpo]] — group std computation, zero-std handling, `tool_mask`(§2, §7, diagnostic).
- [[john-schulman-kl-tricks]] — k1과 k3 KL estimator 이름(§6).
