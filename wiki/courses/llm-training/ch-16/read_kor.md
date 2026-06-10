<!-- chapter: ch-16
     track: data
     title: RL Prompt Curation and Replay Data
     sources: [[replay-buffer-rlhf]], [[kimi-k1-5]], [[kimi-k2]], [[tulu-3]], [[rlvr-tulu3]], [[reinforcement-learning-with-one-training-example]], [[on-off-policy-rlhf]], [[minibatch-sharing-rl]], [[policy-coverage-loss]], [[prorl]]
     figures: figures/rollout-passrate.html
-->

# 16장 — RL 프롬프트 큐레이션과 리플레이 데이터

> **핵심 통찰.** RL prompt는 SFT prompt가 아니다. SFT prompt에는 target completion이 필요하다. RL prompt에는 (a) scalar reward를 반환하는 verifier 또는 judge와 (b) **non-degenerate rollout distribution**이 필요하다. 현재 policy 아래에서 일부 rollout은 성공하고 일부는 실패해야 한다. 그렇지 않으면 gradient가 0이다. 100%-pass prompt는 rollout을 낭비한다. 0%-pass prompt는 rollout을 낭비하고 *loss도 낮춘다*. `p̂ ∈ [0.1, 0.9]`가 gradient signal이 살아 있는 영역이다.
>
> **가이드라인.** (1) 모든 prompt를 verifier/judge와 pairing하고, (2) 현재 policy에서 pass-rate를 mining해 middle band를 유지하고, (3) *prompt는 replay하되 trajectory는 절대 replay하지 말고*(IS ratio가 폭발한다), (4) policy가 움직일 때 pass-rate를 다시 측정하는 방식으로 curate하라. [[kimi-k1-5]]가 reference recipe이고, [[tulu-3]]가 open-source instantiation이며, [[reinforcement-learning-with-one-training-example]]는 극단적 edge를 보여준다.

---

## 이 장이 존재하는 이유

[[ch-15]]는 human-annotation track을 마무리했다. rubric design, adjudication, preference sampling. 그 출력은 DPO와 reward-model training으로 들어간다. RLVR 또는 rubric-RL loop가 아니다. RL-stage training은 다른 artifact를 소비한다. *prompt pool*이다. policy가 이 prompt에 대해 rollout하고, reward는 verifier, reward model, 또는 model itself가 만든다. 세 전통이 수렴한다. RLVR([[rlvr-tulu3]], [[tulu-3]])은 prompt의 유일한 역할을 verifier `v: (x, y) → {0,1}`를 운반하는 것으로 본다. partial rollout을 가진 long-CoT RL([[kimi-k1-5]])은 prompt를 buffer에 두고 live pass-rate로 selection을 구동한다. agentic RL([[kimi-k2]])은 environment seed와 outcome-success verifier를 사용한다.

세 전통 모두에서 같은 운영 질문이 지배한다. **다음 batch에 어떤 prompt가 어떤 비율로 들어가는가?** 순진한 답("전부 uniform하게")은 틀렸다. GRPO/RLOO([[minibatch-sharing-rl]])에서 zero-variance prompt는 zero gradient를 내고, always-fail prompt는 rollout을 낭비한다. 수정된 답은 pass-rate-filtered, variance-weighted, 그리고 움직이는 policy에 대해 주기적으로 재측정되는 것이다. §1은 RL prompt object를 정의한다. §2는 difficulty mining을 다룬다. §3은 stale-policy replay bias를 유도한다. §4는 curriculum schedule을 제시한다. §5는 synthetic-prompt track으로 연결한다.

---

## 1. RL 단계 prompt가 실제로 무엇인가

**SFT prompt**는 `(x, y*)` tuple이다. input + target이다. Training loss는 `−log π_θ(y* | x)`다. target이 gradient direction을 제공한다.

**RL prompt**는 `(x, R)` tuple이다. 여기서 `R`은 *mechanism*이다. policy가 생성한 임의의 completion `y`를 받아 scalar reward를 반환하는 function 또는 process다. target completion은 없다. policy 자체가 `y ~ π_θ(·|x)`를 생성하고, gradient는 `(r(x, y) − b) ∇ log π_θ(y|x)`에서 온다. 여기서 `b`는 baseline이다. prompt 자체에는 supervision signal이 없다. prompt는 *채점 능력*을 운반한다.

이 구분은 세 가지 prompt-pool invariant를 강제한다.

**(i) 모든 prompt는 grader와 pairing되어야 한다.** [[rlvr-tulu3]]는 단호하다. *"모든 prompt에 대해 binary {0,1}을 반환하는 per-example verifier를 정의하라."* 2024–2025년 production에서 관찰되는 grader type은 다음과 같다.

| Grader type | Examples | Where from |
|---|---|---|
| Exact-match / symbolic | GSM8K integer match, MATH SymPy equivalence | [[tulu-3]], [[rlvr-tulu3]] |
| Unit-test execution | LiveCodeBench, SWE-bench cases | [[tulu-3]], [[kimi-k2]] |
| Constraint checker | IFEval regex constraints, JSON-schema validators | [[tulu-3]] |
| Scalar reward model | Bradley–Terry RM on preferences | classical RLHF |
| CoT reward model | model generates reasoning trace, then JSON verdict | [[kimi-k1-5]] (98.5% val-acc vs 84.4% scalar head) |
| Self-critique rubric | model produces rubric + scores its own completion | [[kimi-k2]] |
| Environment outcome | tool-call returns success / test env state | [[kimi-k2]] agentic RL |

grader가 없는 prompt는 RL pool에 들어갈 수 없다. 그래서 RL pool은 보통 SFT pool의 **superset이 아니라 subset**이다. instruction-following chit-chat에는 mechanical verifier가 없으므로 SFT에 남거나 learned RM을 가진 DPO/RLHF로 간다.

**(ii) "prompt"는 사실 `(x, R, metadata)`다.** metadata slot은 difficulty tag, verifier가 필요로 하는 reference answer, mix identifier(어떤 curriculum stage인지), 그리고 가장 중요한 §2의 rolling pass-rate estimate를 담는다. 이 field 없이는 prompt를 curriculum으로 route할 수 없다.

**(iii) RL prompt의 shelf life는 policy에 묶여 있다.** step 0에서 "hard"(pass-rate 0.2)였던 prompt가 step 5000에서는 "solved"(pass-rate 0.95)가 될 수 있다. gradient contribution은 policy improvement와 함께 매끄럽게 악화된다. Utility가 거의 stationary한 SFT example과 달리, RL prompt는 현재 `π_θ`에 의존해 가치가 달라지는 *non-stationary asset*이다. 이것이 가장 자주 놓치는 속성이며, 아래의 모든 운영 선택을 구동한다.

---

## 2. 난이도 마이닝 — pass-rate filter

**Filter.** `p̂(x) = (1/K) Σ_k 1[v(x, y_k) = 1]`라고 하자. 여기서 `y_k ~ π_θ(·|x)`, `k = 1..K`다. empirical pass-rate는 현재 sampling temperature 아래 policy가 prompt를 풀 확률이다. prompt-pool filter는 threshold band `[p_lo, p_hi]`다. `p̂ ∈ [p_lo, p_hi]`인 prompt를 유지하고 나머지는 drop한다.

**왜 lower bound만이 아니라 band인가.** 두 boundary case가 왜 두 threshold가 모두 필요한지 보여준다.

- `p̂ = 1`(항상 푼다): group baseline `μ = mean_k R_k = 1`이고 advantage `A_k = R_k − μ = 0` for all rollouts. GRPO gradient는 0이다. RLOO도 마찬가지다. leave-one-out mean이 1이고 advantage가 0이다. prompt는 아무것도 기여하지 않는다. 더 나쁘게는, 그 아무것도 아닌 것에 `K` rollout compute를 소비한다. [[minibatch-sharing-rl]]은 이를 형식화한다. group-relative estimator는 `var_k R_k > 0`일 때만 well-defined다.
- `p̂ = 0`(절대 못 푼다): 같은 degeneracy다. 모든 reward가 0이고, advantage가 어디서나 0이며, gradient가 0이다. 추가로, always-fail prompt는 PPO를 쓰는 경우 learned value function을 pessimism 쪽으로 bias한다.

정보가 있는 band는 `0 < p̂ < 1`이고, reward variance가 최대인 `p̂ ≈ 0.5` 근처가 sweet spot이다(`Var(Bernoulli(p)) = p(1−p)`는 `p = 0.5`에서 peak). 전형적 production band는 다음과 같다.

| Setting | `p_lo` | `p_hi` | Rationale |
|---|---|---|---|
| Stable-phase RL ([[tulu-3]] RLVR) | 0.1 | 0.9 | 넓은 band. diversity를 위해 low-variance prompt도 수용 |
| Hard-reasoning RL ([[kimi-k1-5]]) | 0.05 | 0.5 | 더 hard 쪽으로 skew. near-miss prompt 우선 |
| ProRL boundary-expansion ([[prorl]]) | 0.0 | 0.3 | K_big 중 하나라도 성공하는 rollout이 있으면 zero-pass-rate prompt도 수용 |
| Cold-start (fresh policy) | 0.2 | 0.8 | 중간 band. early training을 destabilize하는 0-pass prompt 회피 |

**Kimi K1.5 measurement protocol.** [[kimi-k1-5]]는 pass-rate를 어떻게 추정하는지 명시한다. *"고온에서 뽑은 SFT sample 10개의 pass-rate로 difficulty를 측정한다."* 현재 policy가 아니다. SFT-stage policy에서 `K = 10` rollout을 high temperature로 뽑는다. 두 결과가 있다.

1. 측정이 싸다(one-shot, prompt당 약 10K token).
2. RL이 시작되는 순간 측정은 *stale*해진다. K1.5의 curriculum과 prioritized-sampling loop는 RL policy가 개선됨에 따라 암묵적으로 prompt를 re-rank한다. `p(sample prompt i) ∝ 1 − success_rate_i`는 frozen SFT measurement가 아니라 *현재* RL policy에 대해 계산된다.

**Tülu 3 open-source instantiation.** [[tulu-3]]는 RLVR prompt pool을 다음처럼 만든다.

1. SFT mix(939K prompts)에서 시작한다.
2. verifier가 있는 domain만 유지한다. GSM8K, MATH, IFEval-style, code-with-tests.
3. difficulty estimation 뒤 더 작은 pool(약 10^5 prompts)로 filter한다.
4. 그 pool에 대해 10M episodes로 PPO를 실행한다.

Ablation은 paper에 묻혀 있지만 방향성 주장은 명확하다. pass-rate로 filter한 것이 unfiltered보다 downstream eval에서 몇 점 앞서며, 이 gain은 쉬운 prompt 제거에 견고하게 귀속된다(어려운 것 제거가 아니라).

**One-shot RLVR evidence.** [[reinforcement-learning-with-one-training-example]]는 논리를 극단까지 밀어붙인다. Qwen2.5-Math-1.5B에서 *하나의* 잘 고른 prompt가 MATH500을 36.0% → 73.6%로 올린다. Selection heuristic은 **historical-variance score**다. 전체 dataset에서 training epoch별 prompt accuracy variance로 prompt를 rank한다. 이것은 pass-rate filter의 derivative form이다. 절대 pass-rate 대신 *시간에 따른 pass-rate 변화량*으로 rank한다. training 중 난이도가 요동치는 prompt는 gradient signal도 요동치는 prompt이고, 거기서 학습이 일어나기 때문이다. 논문은 유용한 prompt가 유일하지 않음도 보인다. 많은 high-variance prompt가 작동한다. 신호는 단일 best example이 아니라 band다.

---

## 3. Replay buffer와 stale-policy bias 유도

이 section에는 세심한 유도가 필요하다. 민간 버전("replay는 괜찮고 clip만 하면 된다")은 틀렸기 때문이다.

### 3.1 Classical replay vs policy-gradient replay

DQN, Ape-X, R2D2는 replay buffer에 의존한다. improvement가 learned Q-value에서 bootstrap되기 때문이다. Bellman target `r + γ max_a Q(s', a)`는 behavior policy에 의존하지 않는다. LLM RL은 다르다. PPO와 GRPO는 estimator `∇ J(θ) = E_{y ~ π_θ}[ ∇ log π_θ(y|x) · A(x,y) ]`를 가진 **policy-gradient** method다. on-policy expectation이다. 저장된 `y ~ π_old`에는 importance-ratio correction `ρ(x,y) = π_θ(y|x) / π_old(y|x)`가 필요하다. PPO의 clip은 그 correction을 bounded하게 만든 것이다.

### 3.2 IS-ratio explosion derivation

Completion은 `y = (y_1, ..., y_T)`다. causal LM 아래 ratio는 token-wise로 factorize된다. `ρ(x,y) = Π_t ρ_t`. per-token `log ρ_t ~ N(0, σ²)`라고 가정하자(작은 PPO update에서는 합리적이다. `σ²`는 `π_old`에서 policy가 drift할수록 커진다). 그러면 `log ρ ~ N(0, T·σ²)`이고

```
E[ρ]   = exp(T · σ² / 2)
Var[ρ] = exp(2 T σ²) − exp(T σ²)
```

[[replay-buffer-rlhf]]의 구체적 수치다. 저장 이후 `K` gradient-update step 동안 누적된 per-token drift가 `σ ≈ 0.01`(즉 `σ² = 1e-4`)라고 하자(따라서 effective `σ² = K · 1e-4`).

| `K` steps stored | `T = 100` | `T = 1000` |
|---|---|---|
| 1 | `E[ρ] ≈ 1.005` (safe) | `E[ρ] ≈ 1.05` (safe) |
| 20 | `E[ρ] ≈ 1.10` (clip-reachable) | `E[ρ] ≈ 2.7` (outside any clip) |
| 50 | `E[ρ] ≈ 1.28` (marginal) | `E[ρ] ≈ 12` (clipping discards ~all samples) |

Variance는 더 나쁘다. `exp(2 T σ²)`처럼 증가하므로 mean보다 먼저 estimator가 폭발한다. **결론.** replay interval `K ≥ 10`이고 response length `T ≥ 100`이면, trajectory-level replay는 biased(clipped) 또는 unboundedly-variant(unclipped) gradient update를 주입한다. 이것이 [[replay-buffer-rlhf]]와 [[deepseek-r1]] §3.2가 모두 trajectory replay를 포기하는 기계적 이유다. 또한 [[on-off-policy-rlhf]]가 offline-DPO vs PPO gap의 약 80%가 distribution shift라고 찾은 것의 형식적 버전이다. 같은 병리가 이제 PPO loop 내부에 있다.

### 3.3 세 가지 compounding pathology

IS ratio 외에도([[replay-buffer-rlhf]]): (1) **Stale advantages** — GRPO의 `A_{i,k} = (R_{i,k} − μ_i) / (σ_i + ε)`는 rollout time에 normalize된다. current stat에 대해 re-normalize하는 것은 도움이 되지만 IS drift를 고치지는 못한다. (2) **KL drift** — reward stream에는 `−β · KL(π_θ || π_ref)`가 포함되고 `π_ref`가 움직였을 수 있다(예: [[prorl]]이 reset한다). 저장된 reward가 현재 gradient와 inconsistent해진다. (3) **Critic staleness** — learned critic을 쓰는 PPO에서는 `V_φ`가 움직였다. `A = R − V`는 stale `V`를 사용한다.

### 3.4 살아남는 것 — prompt-level replay

[[replay-buffer-rlhf]]가 2024–2026년의 작동 패턴으로 식별한 것: **completion을 replay하지 말고 prompt를 replay하라.** Buffer는 `(x, group_rewards, group_variance, seen_steps)`를 저장한다. 다음 step에서 variance로 weighting해 buffer에서 prompt를 sample하되, `y ~ π_θ`는 새로 regenerate한다. completion이 on-policy이므로 IS correction이 필요 없다. *prompt*만 possibly-non-uniform distribution에서 sample되었고, 그 non-uniformity는 feature다(hard/high-variance prompt가 더 많이 보인다).

TRL의 `GRPOWithReplayBufferTrainer`([[replay-buffer-rlhf]])가 reference다.

```python
@dataclass
class BufferEntry:
    prompt_ids: torch.Tensor
    rewards:    torch.Tensor      # (n,) per-rollout outcome rewards
    variance:   float              # rewards.var().item()
    seen_steps: int

# Sampling step
p_replay = 0.25
for i in range(B):
    if random() < p_replay and len(buffer) > 0:
        # draw from buffer, weighted by variance
        probs = np.array([e.variance + eps for e in buffer])
        probs /= probs.sum()
        entry = np.random.choice(buffer, p=probs)
        batch.append(entry.prompt_ids)
    else:
        batch.append(sample_fresh_prompt())

# After rollouts: insert each prompt back with updated variance.
```

하중을 지탱하는 세부사항이 둘 있다.

- **Zero-variance downweight.** `variance == 0`인 entry(rollout이 모두 correct 또는 모두 incorrect)는 probability 0을 받는다. 이는 §2의 pass-rate filter를 sampling weight로 표현한 것이다. "solved"가 된 prompt는 explicit eviction rule 없이도 effective buffer에서 조용히 쫓겨난다.
- **Fresh regeneration.** 이전 step에서 생성된 completion은 *버린다*. 이것이 classical trajectory replay와의 핵심 차이다. Storage는 `O(#prompts)`이지 `O(#prompts × avg_tokens)`가 아니다.

### 3.5 Kimi K1.5의 partial rollout — 정당한 예외

[[kimi-k1-5]]는 **partial rollout**을 설명한다. 긴 response를 training iteration에 걸쳐 segment로 나누고, replay buffer에서 이전 trajectory segment를 재사용한다. 이것은 trajectory replay처럼 보이지만 아니다.

- 재사용되는 segment는 *prefix*(prompt + early reasoning)다. prefix 이후의 token position은 현재 policy 아래에서 새로 rollout된다.
- prefix가 `∇ log π_θ` target으로 사용되지 않고, fresh sampling의 *context*로만 사용되기 때문에 IS-ratio argument가 적용되지 않는다. trajectory replay라기보다 prompt-extension에 가깝다.
- Infrastructure win: iteration당 fixed output-token budget이 GPU memory를 제한하여 K1.5가 quadratic rollout cost 없이 128K RL context로 scale할 수 있게 한다.

이 pattern은 반례처럼 보이지만 반례가 아니기 때문에 내재화할 가치가 있다. "gradient를 통과시킬 completion은 replay하지 말라"는 규칙은 여전히 유지된다.

---

## 4. Prompt 공간의 curriculum

2024–2025년 production에서 관찰된 세 가지 concrete curriculum을 표로 정리한다.

| Schedule | Who uses it | Stage 1 (cold-start) | Stage 2 (bulk) | Stage 3 (anneal) |
|---|---|---|---|---|
| Fixed-band ([[tulu-3]] RLVR) | Tülu 3, OLMo 2 | pool-wide, pass-rate ∈ [0.1, 0.9] | same | same (no annealing) |
| Prioritized 1−p ([[kimi-k1-5]]) | Kimi K1.5 | sample ∝ 1 − p̂, SFT-policy measured | sample ∝ 1 − p̂, RL-policy re-measured every ~500 steps | optionally narrow to `p̂ < 0.3` for final push |
| Annealed difficulty ([[prorl]]) | ProRL | broad task suite at medium difficulty | gradually narrow to harder-only prompts | reset reference policy + broaden again |
| Agentic outcome-filtered ([[kimi-k2]]) | Kimi K2 | environment-graded easy tool-call tasks | 20K-tool mixed pool, outcome-verified | long-horizon multi-tool tasks |

네 schedule 모두를 관통하는 설계 원칙은 네 가지다.

**(a) Cold-start prompt는 solvable해야 한다.** Fresh SFT policy는 진짜 어려운 prompt에서 degenerate pass-rate(0에 붙음)를 가진다. `p̂ < 0.1` prompt만으로 RL을 시작하면 zero gradient가 나오고 run은 초기 configuration을 벗어나지 못한다. [[kimi-k1-5]]는 이를 "curriculum sampling"이라고 표현한다. 현재 competence 근처에서 시작하라는 것이다. [[tulu-3]]의 fixed band `[0.1, 0.9]`는 게으르지만 robust한 버전이다.

**(b) Annealing은 re-measurement를 요구한다.** `p̂(x)`를 현재 policy에 대해 다시 측정하지 않는 "week 1: easy, week 2: medium, week 3: hard" schedule은 prompt difficulty가 intrinsic하다고 가정한다. 그렇지 않다. 난이도는 `π_θ`에 상대적이다. [[kimi-k1-5]]의 "prioritized sampling ∝ 1 − success_rate"가 작동하는 이유는 success_rate가 live policy에 대해 *per prompt per epoch* 추적되기 때문이다.

**(c) Reference-policy reset([[prorl]])은 disguise한 curriculum이다.** reset step은 사실상 pool을 re-curate한다. 새 reference가 KL-penalty landscape를 바꾸고 pass-rate distribution을 넓히므로, 이전에 memorized된 prompt가 다시 shuffle된다. [[prorl]]의 boundary-expansion 주장은 운영적으로 novel algorithm보다 curriculum-plus-resets에 관한 것이다.

**(d) Reward fidelity가 아니라 policy coverage.** [[policy-coverage-loss]]는 이 점을 형식화한다. signal은 induced distribution이 target support와 겹칠 때만 유용하다. pass-rate filter가 운영적 형태다. `p̂ > 0`이 곧 "correct-answer set의 non-zero coverage"다.

---

## 5. Synthetic으로 가는 다리 — 왜 Track 3이 이 장을 상속하는가

2025년의 두 데이터 포인트가 human-to-synthetic 전환을 강제한다.

**Prompt-pool exhaustion.** [[tulu-3]]의 939K SFT pool은 filtering 뒤 약 10^5개의 verifiable RLVR prompt를 낸다. RL run 하나는 10^7 episodes를 태운다(Tülu 3: 정확히 10M). prompt당 `K = 8` rollout이면 1.25M prompt-visit이고, prompt당 약 12회다. Pass-rate가 안정되고 pool은 "done"이 된다. 다음 run은 fresh prompt를 원하고, 이 volume에서 human curation은 불가능하다.

**Verifier가 bottleneck이다.** Verifier 없는 RL prompt는 쓸모없다(§1). Math problem 생성은 쉽다. `(problem, reference_answer, equivalence-grader_spec)` 생성은 더 어렵지만 synthetic-friendly하다. 강한 model이 triple을 대량 생성하고, second-stage filter가 independent verification이 동의하는 것만 유지한다. [[kimi-k2]]의 agentic pipeline, 즉 20K+ tools, real + simulated environments, outcome-success verifier로 trajectory를 filter하는 방식은 environment level에서 같은 recipe다.

Track 3(synthetic data generation)은 정확히 이 pipeline을 만든다. 이 장이 설치하는 prerequisite은 다음이다. generator가 만들어야 하는 "RL prompt = `(x, R, metadata)` with a grader"(§1), target policy에 대한 pass-rate filter(§2), prompt-level replay(§3) — synthetic prompt는 trainer에게 replay buffer이며, 생성된 *solution*은 trajectory로 replay하면 안 된다 — curriculum re-measurement(§4) — solved prompt는 negative-yield가 되며 generator를 다시 query해야 한다. 이 장의 역할은 누가 작성했든 prompt가 RL loop에 들어가기 전에 어떤 모습이어야 하는지를 분명히 하는 것이다.

---

## 6. 바로 쓸 수 있는 reference — RL prompt-pool manager

§2, §3, §4를 data pipeline과 RL trainer 사이에 놓이는 canonical prompt-pool manager로 합친다. Naming은 TRL + verl convention을 따른다.

```python
@dataclass
class RLPrompt:
    prompt_ids: torch.Tensor
    verifier_id: str            # "math_exact", "code_tests_42", "ifeval_json"
    reference:   dict | None    # e.g. {"answer": "42"}
    mix_tag:     str
    pass_rate:   float | None = None     # EMA of per-prompt mean reward
    var:         float        = 0.0       # reward variance last seen
    seen_steps:  int          = 0

class RLPromptPool:
    """Band filter + variance-weighted replay + periodic re-measurement.
       Prompts, not trajectories — storage is O(#prompts)."""
    def __init__(self, prompts, p_lo=0.1, p_hi=0.9,
                 p_replay=0.25, ema_alpha=0.3, remeasure_every=500):
        self.prompts = list(prompts)
        self.p_lo, self.p_hi = p_lo, p_hi
        self.p_replay, self.ema_alpha = p_replay, ema_alpha
        self.remeasure_every, self.step = remeasure_every, 0

    def _in_band(self, p): return p is None or self.p_lo <= p <= self.p_hi

    def sample_batch(self, B):
        band   = [pr for pr in self.prompts if self._in_band(pr.pass_rate)]
        replay = [pr for pr in band if pr.var > 1e-6]   # non-degenerate
        out = []
        for _ in range(B):
            if replay and np.random.random() < self.p_replay:
                w = np.array([pr.var for pr in replay]); w /= w.sum()
                out.append(replay[np.random.choice(len(replay), p=w)])
            else:
                out.append(band[np.random.randint(len(band))])
        return out

    def update_after_rollouts(self, prompts, rollouts):
        for pr, rewards in zip(prompts, rollouts):
            p_hat, v_hat = float(rewards.mean()), float(rewards.var())
            pr.pass_rate = (p_hat if pr.pass_rate is None
                            else (1-self.ema_alpha)*pr.pass_rate
                                 + self.ema_alpha*p_hat)
            pr.var, pr.seen_steps = v_hat, pr.seen_steps + 1
        self.step += 1

    def remeasure_if_due(self, rollout_fn, K=8):
        """Re-roll K completions/prompt at the *current* policy to refresh
           stale pass_rate / var. Kimi K1.5 does this implicitly via its
           prioritized-sampling loop; we make it explicit."""
        if self.step % self.remeasure_every != 0: return
        for pr in self.prompts:
            r = rollout_fn(pr, K=K)
            pr.pass_rate, pr.var = float(r.mean()), float(r.var())
```

세 가지 invariant가 있다. (i) **trajectory storage 없음** — prompt당 `(pass_rate, var, seen_steps)`만 있다. completion은 gradient step 뒤 버리므로 §3.2의 IS-ratio pathology가 생기지 않는다. (ii) **band filter와 variance weighting은 orthogonal하다** — band는 eligibility를 결정하고, variance는 over-sampling을 결정한다. 둘 중 하나라도 빼면 엄격히 나쁜 recipe가 된다. (iii) **re-measurement가 explicit하다** — open-source reference implementation이 가장 자주 생략하는 부분이다. 이것이 없으면 band filter는 stale pass-rate 위에서 굳고 replay buffer는 calibration에서 drift한다.

---

## 연결과 다음 단계

- **[[replay-buffer-rlhf]] / §3** — prompt-level replay와 IS-ratio argument에 대한 framework-synthesis page.
- **[[kimi-k1-5]] / §2, §3.5, §4** — partial rollout + prioritized sampling + pass-rate-from-SFT-policy curriculum.
- **[[kimi-k2]] / §1, §4** — environment-graded outcome verifier를 가진 agentic prompt pool.
- **[[tulu-3]] / [[rlvr-tulu3]] / §1, §2** — open-source RLVR prompt-curation recipe; 10M-episode budget.
- **[[reinforcement-learning-with-one-training-example]] / §2** — pass-rate-filter thesis의 extreme edge; historical-variance ranking.
- **[[on-off-policy-rlhf]] / §3** — distribution shift(trajectory replay의 failure mode)가 지배한다는 이론적 근거.
- **[[prorl]] / §4** — curriculum으로서 reference-policy reset, boundary-expansion claim.
- **[[policy-coverage-loss]] / §4** — pass-rate filter의 formal cousin으로서 coverage.
- **ch-17 (lab)** — data track을 닫는다. **Track 3 (synthetic data, ch-18+)** — RL prompt를 scale로 생산한다. 이 장의 invariant가 그 acceptance test다.

## 더 읽을거리

[[replay-buffer-rlhf]] (TRL `GRPOWithReplayBufferTrainer`; DeepSeek-R1 §3.2 negative-result); [[kimi-k1-5]] (partial rollouts, prioritized sampling); [[kimi-k2]] (agentic RL, joint RLVR + rubric); [[tulu-3]] / [[rlvr-tulu3]] (open RLVR pipeline; verifier taxonomy); [[reinforcement-learning-with-one-training-example]] (one-shot RLVR; historical-variance ranking); [[on-off-policy-rlhf]] (distribution-shift decomposition); [[prorl]] (prolonged RL + reference resets); [[policy-coverage-loss]] (coverage-based transfer bound); [[minibatch-sharing-rl]] (group-baseline variance math).

## 동반 시각화

**[figures/rollout-passrate.html](figures/rollout-passrate.html)** — interactive rollout-pass-rate explorer. 실제 RL pool에서 보이는 bimodal easy/hard shape를 반영한 beta-mixture distribution에서 1000개 simulated prompt를 sample한다. `p_lo`와 `p_hi` threshold를 drag하면 페이지가 업데이트한다. (i) kept-prompt count, (ii) kept band 위에 적분한 expected per-prompt reward variance(`p(1−p)`), (iii) kept prompt 중 high-information `p = 0.5` line 근처에 있는 비율을 보여주는 effective-difficulty curve, (iv) [[minibatch-sharing-rl]]의 group-baseline degeneracy note가 달린 "zero-gradient zones"(`p ≈ 0`과 `p ≈ 1`). band가 왜 중요한지, sweet spot이 예상보다 넓은 이유를 내재화하는 데 사용하라.
