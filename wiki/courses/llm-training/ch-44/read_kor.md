<!-- chapter: ch-44
     track: rl
     kind: content
     title: Process Supervision and Verifiable Rewards
     deps: [ch-43]
     sources: [[prm800k]], [[lets-verify]], [[let-verify]], [[math-shepherd]], [[omegaprm]], [[rlvr-tulu3]], [[tulu-3]], [[swe-rl]], [[training-verifiers-to-solve-math-word-problems]], [[step-dpo]], [[prorl]], [[rlvr-beyond-base-model]]
     figures: figures/prm-vs-orm.html
     excerpts: excerpts/prm800k-label-protocol.md, excerpts/lets-verify-prm-vs-orm.md, excerpts/math-shepherd-mc-formula.md, excerpts/omegaprm-divide-and-conquer.md, excerpts/rlvr-tulu3-config.md, excerpts/swe-rl-difflib-reward.md, excerpts/prm-vs-rlvr-contrast.md
-->

# 44장 — 과정 감독과 검증 가능한 보상

> **핵심 통찰.** 과제에 정답 확인 수단이 있으면, 즉 grader, unit test, reference diff가 있으면, 학습된 reward model은 자산이 아니라 부채다. trajectory를 분해해 각 step을 process-level signal(PRM)에 맞춰 보상하거나, 전체 reward function을 `v(x, y) in {0, 1}`로 접고 deterministic program이 판정하게 하라. 두 방식 모두 proxy를 제거함으로써 Goodhart를 우회한다. 다만 비용과 signal density가 다르고, 과제별로 올바른 선택을 하는 것이 이 장의 실제 내용이다.
>
> **가이드라인.** 모든 RL prompt에 대해 먼저 물어라. "답을 확인할 수 있는가?" 그렇다면 RM을 건너뛰고 verifier를 사용하라(RLVR, Tulu-3 LR `3e-7`, KL `0.05`; SWE-RL difflib ratio, GRPO KL `0.02`). 답은 확인 가능하지만 chain이 길다면, outcome check 위에 PRM을 더해 credit이 첫 번째 나쁜 step에 떨어지게 하라. Math-Shepherd의 step별 scan(`O(K L)`)이 아니라 OmegaPRM의 divide-and-conquer MC labels(`O(K log L)`)로 만들라. verifier가 없을 때에만 학습된 preference RM(ch-41)으로 물러나라.

---

## 이 장이 존재하는 이유

40장은 PPO와 GRPO를 만들었고, 41장은 Bradley-Terry RM을 학습했으며, 42장은 reward hacking을 목록화했고, 43장은 실행을 살려 두는 KL / entropy control을 제공했다. 이 모든 기법은 학습된 scalar reward를 가정한다. 지난 3년간 post-training 진전의 상당 부분 — [[prm800k]], [[math-shepherd]], [[omegaprm]], [[rlvr-tulu3]], [[swe-rl]], [[deepseek-r1]] — 은 *그렇게 가정하지 않는 것*에서 왔다. 과제가 검증 가능할 때 학습된 RM은 명백히 더 나쁘다. label을 먹고, reward-model over-optimisation([[reward-model-overoptimization]], ch-42에서 이어짐)에 노출되며, 정책이 그 drift를 exploit하도록 가르친다. Process supervision과 verifiable rewards가 두 개의 탈출구다. 이 장은 taxonomy를 고정해 ch-45(self-improvement loops)와 ch-46(the RL lab)이 둘 중 무엇을 쓸지 깔끔하게 고르게 한다.

---

## §1. Outcome RM이 놓치는 문제

[[training-verifiers-to-solve-math-word-problems]](Cobbe 2021)는 이 문제를 처음 프레이밍한 논문이다. GSM8K에서 pass@1이 20%인 6B GPT 모델도 pass@100은 60%다. generator는 이미 다섯 번 중 한 번은 정답을 만든다. 부족한 것은 *selection*이다. Cobbe의 해결책은 outcome reward model(ORM)이었다. `(question, candidate-solution)` pair를 점수화하는 별도 verifier를 학습하고, 100개를 샘플링한 뒤 최고 점수 해를 고른다.

| Method | GSM8K | MATH-500 (subset) | Signal density |
|--------|-------|--------------------|----------------|
| Finetuning only (6B, Cobbe 2021) | ~20 pass@1 | — | dense (every token) |
| ORM best-of-100 (Cobbe 2021) | +strong gain | — | one scalar per solution |
| Majority vote (Lightman 2023) | — | 69.6 | — |
| ORM best-of-1860 (Lightman 2023) | — | 72.4 | one scalar per solution |
| **PRM best-of-1860 (Lightman 2023)** | — | **78.2** | one scalar *per step* |

마지막 두 행이 전환점이다. [[prm800k]]는 MATH 문제에서 outcome label이, *compute를 맞췄을 때에도*, PRM 대비 5.8 절대 포인트를 남겨 둔다는 것을 보였다. 그리고 그 격차는 N과 함께 커진다. 이유는 단순하다. 긴 reasoning trace는 두 개의 잘못된 step이 서로 상쇄되어 맞는 답에 도달할 수 있고, 반대로 step 14에서만 미끄러지는 깨끗한 chain으로 틀린 답에 도달할 수도 있다. Outcome label은 둘을 구분하지 못한다. Process label은 구분한다.

---

## §2. PRM800K — reference protocol로서의 step-level human labels

[[prm800k]]는 표준 supervised PRM이다. 모든 자동화 변형(Math-Shepherd, OmegaPRM, R*-Math)이 이를 싸게 근사하도록 설계되었기 때문에, protocol은 외울 가치가 있다.

**데이터 단위.** label된 예시 하나는 `(problem, partial-solution-prefix, next-step, label)`이며 `label in {+1 correct, -1 incorrect, 0 neutral}`이다. generator는 newline-delimited step을 생성하도록 지시받아 labeler가 한 번에 한 step을 보게 된다. labeler는 처음 만나는 `-1`을 표시하고 멈춘다. Neutral은 correctness signal을 담지 않는 filler("Let me re-read the problem...")를 포착한다.

**스케일.** 12K MATH training problem에 대한 ~75K GPT-4 generation에서 800K step label. 논문은 outcome labeling 대비 example당 비용이 약 **10x**라고 명시한다. publishable budget을 원한다면 active learning은 선택 사항이 아니다.

**PRM training.** `{good, bad}`에 대한 binary classifier head가 각 step separator 바로 뒤 token position에서 작동한다. 손실은 non-neutral step에 대해서만 cross-entropy다.

```python
# Conceptual; reference: prm800k schema + let-verify section 3.
# labels_step[t] in {+1, -1, 0}; 0 is ignored.
step_logits = prm_head(hidden[step_end_positions])          # [num_steps, 2]
step_targets = (labels_step[labels_step != 0] == +1).long() # {0, 1}
loss = F.cross_entropy(step_logits[labels_step != 0], step_targets)
```

**Solution score로의 aggregation.** Lightman 논문은 step별 `p_correct`의 곱을 사용한다.

```
S_prod(y) = prod_{t in steps} p_correct(step_t | prefix_t)
```

동치로 `exp(sum_t log p_correct)`다. [[math-shepherd]]는 나중에 실무적으로 `min_t p_correct(step_t)`가 더 좋은 aggregator라고 주장했다. "해는 가장 나쁜 step만큼만 좋다"는 것이다. GSM8K와 MATH에서 `min`이 `prod`와 `mean`을 Pareto-dominate한다고 보고한다(Math-Shepherd Table 4). 학습 손실은 바뀌지 않는다. inference-time aggregation만 바뀐다.

**Active learning.** 논문은 현재 PRM이 높은 점수를 주지만 final answer가 틀린 solution, 즉 "convincing-wrong" case를 표면화한다. 그 slice에 label을 붙이면 약 2.6x data-efficiency multiplier가 나온다. uniform의 100%와 같은 PRM 품질에 38% label budget으로 도달한다.

---

## §3. Let's-Verify — process vs outcome head-to-head

Matched compute에서의 PRM-vs-ORM 표는 [[lets-verify]]에서 가장 많이 인용되는 결과다. 숫자는 MATH-500 대표 subset(Best-of-N with N=1860)에서 그대로 가져온 것이다.

| Selector | MATH-500 acc | Delta vs majority |
|----------|--------------|--------------------|
| Majority vote | 69.6 | — |
| ORM (outcome labels) | 72.4 | +2.8 |
| **PRM (process labels)** | **78.2** | **+8.6** |

두 가지를 주목하라. 첫째, PRM curve는 큰 N에서만이 아니라 모든 N에서 ORM curve를 *지배*한다. N=16에서도 격차가 보이고 N=1860까지 넓어진다. 둘째, calibration picture(논문 Figure 3)는 PRM의 step별 `p_correct`가 calibration되어 있음을 보인다. ORM의 solution별 score는 그렇지 않다. 이것이 PRM selector가 계속 보상받는 mechanistic reason이다. aggregator `min_t p_correct`는 step별 probability 자체가 의미 있을 때에만 잘 정렬된 statistic이 된다.

Inference cost는 논문이 숨기지 않는 비대칭이다. ORM은 solution당 `O(1)` forward이고, PRM은 step 수 `L`에 대해 `O(L)`이다. MATH-500의 median `L`은 ~10이므로 PRM inference는 Best-of-N 시점에 약 10x 느리다. N 자체가 1860인 상황에서는 받아들일 만한 비용이다.

---

## §4. Math-Shepherd — MC rollout을 자동 step label로 쓰기

~10x outcome-label 비용을 치르는 방식은 800K를 넘어 scale하기 어렵다. [[math-shepherd]]는 PRM을 싸게 만드는 첫 논문이다. 아이디어는 한 줄이다. human labeler를 rollout policy로 바꾸고, label을 empirical reach-probability로 바꾼다.

**정의.**

```
For a step s_t in trajectory (s_1, ..., s_L):

    MC(s_t) = (1/K) * sum_{i=1..K} I[rollout(policy | s_1..s_t) reaches gold]
```

`K`는 step `t`에서 끝나는 prefix로부터 샘플링한 completion 수다. `I[...]`는 그 completion의 final answer가 gold answer와 일치하면 1이다. 논문은 `K = 8` 또는 `16`을 쓴다. [[omegaprm]]는 깊이가 ~10 steps를 넘는 trajectory에는 `K >= 16`이 필요하다고 주장한다.

**Hard vs soft.** "hard" label `y_hard(s_t) = 1[MC(s_t) > 0]`은 "적어도 하나의 rollout이 살아남았다"는 binary label이다. "soft" label `y_soft(s_t) = MC(s_t)`는 fraction 자체다. Math-Shepherd Table 5는 MATH에서는 soft가 이기고, GSM8K에서는 hard가 비슷하다는 것을 보인다. 기본값은 soft다.

**얻는 것.** GSM8K의 Mistral-7B는 step-level PPO(PRM as dense reward)로 77.9 -> 84.1이 되고, 같은 PRM을 inference에서 Best-of-N ranker로만 쓰면 77.9 -> 89.1이 된다. MATH: 28.6 -> 33.0(PPO) -> 43.5(verify). Verify-only 숫자가 PPO를 이기는 것이 lesson이다. 적어도 Math-Shepherd가 테스트한 규모에서는 좋은 PRM은 dense reward보다 ranker로서 더 가치 있다.

**Step-level PPO reward composition.**

```
R_total(trajectory) = r_final + lambda * sum_{t in steps} PRM(step_t)
```

여기서 `lambda ~ 0.1 - 1.0`이고 `r_final`은 outcome 0/1 reward다. PRM 항은 shaped, dense reward처럼 작동한다. final answer가 틀렸더라도 올바른 *중간* step에 대해 정책이 gradient를 받는다. 긴 chain에서 pure RLVR이 약하게 처리하는 failure mode다.

---

## §5. OmegaPRM — `O(K log L)` divide-and-conquer labeling

Math-Shepherd의 trajectory당 `O(K * L)` rollout(모든 prefix가 K completion을 받음)은 여전히 비싸다. [[omegaprm]]는 첫 나쁜 step을 binary search하여 이를 `O(K * log L)`로 낮춘다.

1. `L`개 step을 가진 seed trajectory로 시작한다.
2. K rollout으로 `MC(s_{L/2})`를 측정한다.
3. `MC(s_{L/2})`가 `MC(s_0)`에 가깝다면 첫 오류는 뒤쪽 절반에 있다. `steps[L/2 : L]`에 재귀한다.
4. 그렇지 않으면 첫 오류는 앞쪽 절반에 있다. `steps[0 : L/2]`에 재귀한다.
5. interval 길이가 1이 되면 종료한다. 분리된 step이 첫 번째 잘못된 step이다.

그 과정에서 채워진 label이 PRM training set을 이룬다. "MC가 급격히 떨어졌다"의 threshold는 `tau ~ 0.2`다(parent가 `tau`보다 높고 `MC(s_t) < tau`이면 step을 bad로 표시). 전체 논문은 thresholding 대신 soft `MC(s_t)` 값에 대해 MSE로 PRM을 회귀하므로, `tau`는 recursion decision에만 영향을 미치고 final label에는 영향을 주지 않는다.

**결과.** ~80K problem에서 완전 자동 생성한 1.5M step label. Gemini Pro 1.0의 MATH는 PRM-weighted Best-of-N으로 `51.0 -> 69.4`가 된다. 절대 `+18.4`, 상대 `+69.4%`다. 같은 compute에서 Math-Shepherd PRM보다 MATH 약 5점 앞선다.

Math-Shepherd와 OmegaPRM을 나란히 놓으면:

| Method | Rollouts per trajectory | Human labels | GSM8K lift (best) | MATH lift (best) |
|--------|-------------------------|--------------|-------------------|-------------------|
| PRM800K | 0 (human annotates) | 800K | — | 69.6 -> 78.2 |
| Math-Shepherd | `O(K * L)` | 0 | 77.9 -> 89.1 | 28.6 -> 43.5 |
| OmegaPRM | `O(K * log L)` | 0 | — | 51.0 -> 69.4 (Gemini Pro) |

흥미로운 줄은 rollout-count다. `L = 10` steps, `K = 16`인 trajectory는 Math-Shepherd에서 160 completion이 들고 OmegaPRM에서 ~64가 든다. `L = 20`이면 320 vs ~80, `L = 40`이면 640 vs ~96이다. 깊은 reasoning에서는 격차가 크고, 2024년 이후 open reasoning recipe가 OmegaPRM 스타일 divide-and-conquer를 기본으로 삼는 이유다.

---

## §6. RLVR — RM을 완전히 건너뛰기

[[rlvr-tulu3]]는 반대 방향으로 움직인다. task에 verifier가 있으면 PRM을 전혀 만들지 말라. verifier를 reward *그 자체*로 써라.

```
r(x, y) = v(x, y) in {0, 1}
```

이것이 전체 기여다. Tulu-3 open-instruct RLVR pipeline에는 세 verifier domain이 포함된다.

- **Math.** completion에서 final numeric 또는 symbolic answer를 추출한다. MATH에서는 SymPy equivalence로, GSM8K에서는 normalized string match로 reference와 채점한다.
- **Constrained instruction following.** IFEval 스타일 제약("JSON으로 답하라", "정확히 세 개의 bullet을 사용하라")을 regex와 format parser로 확인한다.
- **Code.** sandboxed runner에서 model code를 unit test에 대해 실행한다. 모든 test가 통과하면 reward는 `1`, 아니면 `0`이다.

Tulu-3 model report([[tulu-3]] §RLVR에서 그대로)는 PPO configuration을 다음처럼 고정한다.

```
LR                3e-7
beta (KL coef)    0.05
clip epsilon      0.2
PPO epochs (K)    4
minibatches (N)   1
GAE lambda        0.95
gamma             1.0   (episodic)
local mini batch  32
local rollout     32
total episodes    10,000,000
reward            verifier output in {0, 1}
```

이를 ch-43에 연결하라. `beta = 0.05`는 중간 범위의 KL coefficient다. 논문의 PPO는 verifier reward에 토큰별 `-beta * log(pi / pi_ref)`를 더하고 있는데, 이것이 정확히 ch-43의 k3 estimator 논의다. `gamma = 1.0` with `GAE lambda = 0.95`는 표준 episodic PPO다. reward가 termination에서만 도착하므로 solution 내부에는 discounting이 없다. `LR = 3e-7`은 일반적인 SFT LR보다 한 자릿수 낮다. 작은 KL과 0/1 reward는 high-variance, low-signal stream이기 때문이다. 큰 step은 엔트로피를 빠르게 폭발시킨다(ch-43 다시). DPO-only Tulu-3 checkpoint 대비 측정된 RLVR 이득: GSM8K에서 `+5 - 10pp`, IFEval에서 `+~4pp`, 그 외에는 neutral-to-positive.

**왜 hacking을 우회하는가.** Verifier는 고정된, 해석 가능한 함수다. drift하는 proxy RM이 없고 reward가 가짜로 올라가는 out-of-distribution region도 없다. Goodhart gap(ch-42)은 검증 가능한 prompt에서는 *기계적으로 0*이다. verifier *bug*에서는 0이 아니다. Tulu-3는 prose 안의 "42"를 받아들이는 string-match math grader가 game될 수 있다고 지적한다. failure mode는 verifier engineering을 unit-test engineering처럼 다루는 것이다.

**Prompt curation.** (a) verifier implementation과 (b) known reference answer가 있는 prompt만 RLVR set에 들어간다. 나머지는 DPO(ch-41)로 가거나 버려진다.

---

## §7. SWE-RL — scalable RL substrate로서의 difflib

RLVR의 가장 큰 scaling 질문은 verifier가 어디서 오느냐다. Math와 code에는 명확한 grader가 있지만 대부분의 task에는 없다. [[swe-rl]](Meta, 2025)은 software-engineering RL에 대해 놀라운 trick으로 답한다.

**Reward.**

```python
import difflib
# predicted_patch, ground_truth_patch: unified diffs (strings)
r = difflib.SequenceMatcher(None, predicted_patch, ground_truth_patch).ratio()
# r in [0, 1]; continuous, no execution required.
```

Unit-test execution이 없다. Sandbox도 없다. RM도 없다. Python stdlib의 longest-common-subsequence 스타일 similarity ratio를 model output patch와 human PR diff 사이에 적용할 뿐이다. 논문은 continuous `r in [0,1]`와 binary-thresholded `r > tau`를 ablate한다. continuous가 이기는데, 모든 sample에 dense signal을 주기 때문이다. execution reward는 sparse하다(많은 test가 서로 무관한 이유로 실패한다).

**Training.** Llama-3.1-70B에서 11M scraped (issue, code-context, ground-truth-patch) triples로 GRPO(ch-40)를 수행한다. group size `G = 8`, KL `beta = 0.02`, LR `1e-6`.

**Headline result.** Llama3-SWE-RL-70B는 SWE-Bench Verified에서 41.0%에 도달한다. release 당시 open SOTA였고, DeepSeek-Coder-V2-Instruct(18.0%)를 이기며 SWE-Gym-32B와 맞먹는다. 도발적인 finding은 transfer다. 같은 RL run이 HumanEval+를 +6, MATH를 +4, BBH를 +3 움직인다. GitHub patch로 학습했는데 out-of-domain으로 일반화된다.

**Taxonomy에서 왜 중요한가.** SWE-RL은 "verifiable reward"가 반드시 "unit test passes"를 의미하지 않음을 증명한다. `(prediction, reference)`에서 자동으로 계산할 수 있는 어떤 rule도 해당된다. Template match, AST equivalence check, diff ratio, regex match. 이전에는 학습된 RM이 필요했던 domain에 대해 각각 RLVR reward 후보가 된다.

---

## §8. Decision tree — 언제 무엇을 쓸 것인가

| Task property | Choose | Why |
|---------------|--------|-----|
| No verifier, short outputs, preferences available | Preference RM + PPO/DPO (ch-41) | 더 싼 signal이 없음. |
| Verifier exists, short outputs (math, code, constraints) | RLVR (Tulu-3 hyperparameters) | Goodhart gap 0; 학습할 RM 없음. |
| Verifier exists, long chain-of-thought | RLVR + PRM as shaped reward (Math-Shepherd / OmegaPRM) | Dense intermediate signal; 나쁜 step을 찾음. |
| Reference output은 있지만 grader 없음(patches, summaries) | 규칙 기반 similarity reward(SWE-RL pattern) | 싸고, continuous하며, sandbox 없이 실행됨. |
| Preference data + verifier both exist | RLVR on verifiable slice + DPO on rest (Tulu-3) | prompt별로 섞음; 같은 policy network. |

인접 문헌에서 가져온 두 가지 calibration. [[rlvr-beyond-base-model]](Yue 2025)은 RLVR이 종종 `pass@1`은 올리지만 큰 `k`에서는 base model이 이긴다고 보고한다. 검증 가능한 보상 이득을 기본적으로 *sampling-efficiency* improvement로 취급하고, `pass@1`과 `pass@large-k`를 모두 추적하라. [[prorl]](Liu 2025)은 prolonged RL + KL control + reference-policy reset이 reasoning boundary를 확장할 수 있다고 반박한다. 짧은 run에서 RL이 포화했다고 결론 내리지 말라. 두 calibration은 ch-46 lab에 가장 강하게 영향을 준다. `KL in {0.01, 0.05, 0.1}` sweep이 memo의 성패를 가른다.

---

## §9. Process supervision과 preference optimisation이 만나는 곳

[[step-dpo]]는 이 장과 ch-41 사이의 다리다. Step-DPO는 DPO의 함수 형태를 유지하지만 full trajectory 대신 `(prefix, good-step, bad-step)` triple을 먹인다.

```
L_StepDPO = -log sigma( beta * log[pi_theta(y_w | x) / pi_ref(y_w | x)]
                       - beta * log[pi_theta(y_l | x) / pi_ref(y_l | x)] )
```

여기서 `x`는 multi-step prefix이고 `y_w`, `y_l`은 single step continuation이다. Qwen2-7B-Instruct on MATH: 10K step-preference pair로 53.0 -> 58.6이 된다. Full-trajectory DPO on 100K pairs(54.3)를 이긴다. 그래디언트는 앞뒤의 동일한 token이 아니라 실제 disagreement point에 집중된다. Step-DPO는 PRM이 아니다. PRM의 step-preference cousin이다. `MC(s_t)`가 급격히 떨어진 step과 그렇지 않은 이웃 step을 pairing하면 OmegaPRM output에서 만들 수 있다.

---

## §10. Companion figure

`prm-vs-orm.html` companion(`figures/prm-vs-orm.html`)은 이 장의 hands-on deliverable이다. 두 panel:

1. **Eval-delta panel.** 세 slider — `base_model_capability in [0, 100]`(RL 이전 pass@1), `prm_quality in [0, 1]`(step classifier로서 PRM의 대략적인 AUC), `N_best_of in [1, 2048]`. 이 장의 Lightman 2023과 Tulu-3 숫자에 calibrated된 단순 monotonic model 아래 PRM, ORM, RLVR predicted eval score를 그린다. panel의 목적은 정량 예측이 아니라 어떤 base-model regime에서 *어떤* lever가 가장 중요한지 직관을 만드는 것이다.
2. **Label-cost panel.** 네 방법의 `label_cost_per_1pct_eval_gain` bar chart: PRM800K(human), Math-Shepherd(MC rollouts), OmegaPRM(divide-and-conquer MC), RLVR(zero labels, only verifier engineering). task가 verifiable할 때 no-label verifier가 `$/pt`에서 모든 PRM을 이기고, verifiable하지 않을 때에는 아무것도 하지 못한다는 것을 독자가 보게 한다.

이 페이지와 나란히 열고 §8을 마치기 전에 slider를 움직여 보라.

---

## Key takeaways

- **Outcome labels는 긴 chain에서 새기 쉽다.** 잘못된 step과 이를 상쇄하는 잘못된 step은 맞아 보인다. 늦게 미끄러진 순수한 chain은 틀려 보인다. PRM은 step granularity에서 credit을 분해한다.
- **[[prm800k]]가 label schema를 정한다** — step별 `{+1, -1, 0}`, paper의 product aggregation 또는 Math-Shepherd의 min aggregation. Human labels는 outcome labels보다 ~10x 비싸다. active learning은 선택 사항이 아니다.
- **[[math-shepherd]]는 label을 자동화한다** via `MC(s_t) = (1/K) sum I[rollout reaches gold]`; `K = 8-16`; soft labels; PPO shaping `R_total = r_final + lambda * sum PRM(step_t)`.
- **[[omegaprm]]는 divide-and-conquer한다** scan을 `O(K L)`에서 `O(K log L)`로 낮춘다. Google-DeepMind scale에서 1.5M label을 가능하게 한다. soft MC regression은 `tau` threshold가 recursion에만 영향을 주고 target에는 영향을 주지 않게 한다.
- **[[rlvr-tulu3]]는 RM을 건너뛴다.** `r(x, y) = v(x, y) in {0,1}` with PPO LR `3e-7`, KL `0.05`, clip `0.2`, 10M episodes. Verifiable prompt에서 Goodhart gap은 기계적으로 0이다. 새로운 risk는 verifier bug다.
- **[[swe-rl]]는 rule-based reward가 scale한다는 것을 증명한다** via patch에 대한 `difflib.ratio()`: 11M triples, GRPO, 41.0% SWE-Bench Verified, math와 BBH로 out-of-domain transfer.
- **Decision tree:** verifier + short output -> RLVR; verifier + long chain -> RLVR + PRM shaping; reference-only -> similarity reward; neither -> preference RM(ch-41)로 fallback. 항상 `pass@1`과 `pass@large-k`를 추적하라([[rlvr-beyond-base-model]]).
