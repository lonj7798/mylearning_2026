<!-- chapter: ch-45
     track: rl
     kind: content
     title: Self-Improvement Loops
     deps: [ch-44]
     sources: [[self-rewarding-lm]], [[meta-rewarding-lm]], [[spin]], [[self-play-preference]],
              [[self-correct-rl]], [[rest-em]], [[star]], [[v-star]],
              [[deepseek-r1]], [[r1-zero-analysis]], [[rlvr-beyond-base-model]],
              [[iterative-sft-rl]], [[lilianweng-reasoning-llms]]
     figures: figures/self-improve-loop.html
-->

# 45장 — 자기개선 루프

> **핵심 통찰.** 2024–2025년의 모든 self-improvement 방법은 같은 세 줄짜리 알고리즘이다. *generate, filter, retrain*. 차이는 누가 filter 역할을 하는가뿐이다. filter가 **judge로서의 모델 자신**이면(Self-Rewarding, Meta-Rewarding) judge가 drift하기 때문에 루프는 3 iterations 안에 포화한다. filter가 **human text distribution**이면(SPIN) 데이터에 맞추는 데 수렴하고 그 이상으로 가지 않는다. filter가 **preference game**이면(Nash-LM) stochastic equilibrium으로 수렴한다. filter가 **outcome verifier**이면(ReST-EM, R1-Zero) base model에 끌어낼 latent capability가 남아 있는 한 계속 오른다. 다만 2025년 `pass@k` 분석은 이것이 reasoning을 *만드는* 것이 아니라 *끌어내는* 것일 수 있음을 보인다. filter의 선택이 전체 이야기다.
>
> **가이드라인.** 실제로 가진 signal에 맞춰 filter를 고르라. 검증 가능한 답 → ReST-EM 또는 R1-Zero 스타일 RLVR(monotone, slow saturation). SFT data는 있지만 preference는 없음 → SPIN(distribution-match; cap at 3). Judge-capable model은 있지만 verifier 없음 → Self-Rewarding(3 iters, 이후 calibration drift — Meta-Rewarding의 meta-judge를 더해 5까지 밀기). self-correction이 target → SCoRe의 two-stage RL(Stage I은 turn-1을 freeze, Stage II는 delta를 보상). 어떤 loop든 iteration 사이에 reference model을 reset하고, `pass@1`과 함께 `pass@large-k`를 측정하라. 그래야 base가 이미 아는 것을 날카롭게 만드는 데 그치고 있는지 알아차릴 수 있다.

---

## 1. 공유되는 골격

이 장의 다섯 method family는 모두 `t`로 index되는 outer loop에서 실행되는 세 단계로 factor된다.

```
for t in 1..T:
    samples   = generate(pi_{t-1}, prompts)          # E-step / rollout
    retained  = filter(samples, signal_source)       # judge / verifier / data / game
    pi_t      = retrain(pi_{t-1}, retained)          # SFT, DPO, or RL
    pi_ref   <- pi_{t-1}                             # reset reference (critical)
```

family 사이에서 바뀌는 것은 `signal_source`다.

| Family | Signal source | Filter | Retrain objective | Saturation |
|---|---|---|---|---|
| Self-Rewarding ([[self-rewarding-lm]]) | policy-as-judge, 5-pt rubric | argmax/argmin over 4 samples | DPO β=0.1 | 3 iters (judge drift) |
| Meta-Rewarding ([[meta-rewarding-lm]]) | policy-as-judge + meta-judge | pairwise on 11 judges | DPO on actor *and* judge | 5 iters |
| SPIN ([[spin]]) | human-written SFT response | `chosen=y_human`, `rejected=y_gen` | DPO on (human, self) pair | ~3 iters (distribution match) |
| Nash-LM ([[self-play-preference]]) | pairwise preference model | self-play against EMA opponent | Mirror-descent on preference game | Nash equilibrium |
| SCoRe ([[self-correct-rl]]) | outcome correctness, two-turn | reward shape `r(y2)−r(y1)` | REINFORCE, two-stage | Monotone while reward signal holds |
| ReST-EM ([[rest-em]]) | exact-match / unit-test verifier | K=32 per problem, keep correct | SFT on survivors, 1 epoch | 2 iters (overfit without diversity cap) |
| R1-Zero ([[deepseek-r1]]) | rule-based verifier | GRPO group-relative | On-policy PG with clipped ratio | Open-ended (base-model ceiling) |

이 장의 나머지는 이 표의 각 cell을 따라가고, 2025년 R1-Zero 사후 분석을 읽어 pure-RL 스펙트럼의 끝이 실제로 무엇을 전달했는지 본다.

---

## 2. Self-Rewarding — 정책이 스스로를 judge한다

[[self-rewarding-lm]](Yuan et al., Meta/NYU, 2024)은 Open Assistant로 SFT된 Llama-2-70B checkpoint에서 시작해 세 번의 DPO round를 반복한다. 각 iteration:

1. π_{t−1}에서 prompt당 **4 responses**를 `T=0.7, top-p=0.9`로 샘플링한다.
2. 같은 모델을 judge로 사용해 각 response를 점수화한다. judge prompt는 helpfulness, relevance, depth, clarity, completeness를 다루는 고정 **5-point additive rubric**을 덧붙인다. 분산을 줄이기 위해 각 pair를 **3 times** 판정하고 점수를 평균한다.
3. 가장 높은 점수 response를 `chosen`, 가장 낮은 점수 response를 `rejected`로 삼는다.
4. β=0.1, AdamW lr=5e-7, 1 epoch로 π_{t−1}에서 π_t를 DPO 학습한다.

논문의 중심이자 가장 많이 인용되는 경험적 곡선은 **AlpacaEval 2.0 monotonic climb**이다. 논문 Figure 3에서 입증되고 [[self-rewarding-lm]]에 요약된 값은 다음과 같다.

```
iter 0 (SFT):   9.94 %
iter 1:        15.38 %
iter 2:        20.44 %      <- passes GPT-4 0613 (~19 %)
iter 3:        20.8 %       <- plateau
iter 4:                     <- regresses on reward bench (reward hacking)
```

비대칭이 중요하다. 이는 단일 data curve가 아니라 함께 오르는 *두* curve다. held-out Open-Assistant human preference에 대한 judge의 Spearman correlation은 **iter 0의 0.62에서 iter 3의 0.71**로 오른다([[self-rewarding-lm]] Table 2). actor가 좋아질수록 judge도 좋아진다. frozen reward model은 가질 수 없는 성질이다. 이것이 논문의 실제 기여이며, AlpacaEval win-rate는 표면 indicator일 뿐이다.

**왜 포화하는가.** judge는 ground truth가 아니다. actor와 judge가 모두 더 좁은 high-score basin으로 distill되면 judge 자신의 오류가 systematic해지고, DPO가 이를 증폭한다. 논문은 iter 4가 regresses한다고 보고한다. [[iterative-sft-rl]]도 Llama-2 RSFT를 5 rounds 너머로 밀면 같은 pattern을 지적한다. 3에서 멈추거나 meta-signal을 추가하라.

---

## 3. Meta-Rewarding — judge를 judge하기

[[meta-rewarding-lm]](Wu et al., Meta FAIR + Berkeley, 2024)은 **세 번째 역할**을 추가한다. 모델은 이제 Actor, Judge, *and* Meta-Judge 역할을 한다. iteration마다:

1. prompt당 **K=7** actor response를 샘플링한다.
2. 각 `(prompt, actor_response)` pair에 대해 **N=11** judge response를 샘플링한다. 각 judge response는 score + rationale이다.
3. **Meta-Judge**가 calibration rubric 아래 11개의 judgment를 pairwise-compare하고 best/worst를 고른다.
4. Actor-DPO는 judge score를 aggregation한 `(best_actor, worst_actor)` pair를 사용한다.
5. **Judge-DPO**는 meta-judge가 고른 `(best_judge, worst_judge)` pair를 사용한다.
6. rubric 안의 **length-bias control term**("length for length's sake를 보상하지 말라")이 plain DPO self-loop를 괴롭히는 2× response-length inflation을 막는다.

Judge-DPO leg가 혁신이다. Self-Rewarding은 actor만 학습한다. Meta-Rewarding은 두 역할 모두 학습한다. Llama-3-8B-Instruct AlpacaEval 2.0 LC win-rate는 **22.9 % → 39.4 % over 4 iterations**로 오른다([[meta-rewarding-lm]] Table 1). Self-Rewarding이 평평해지는 지점에서도 judge-human agreement가 계속 오른다.

구조적 pattern을 보라. Self-Rewarding은 **two-layer** stack(Actor, Judge)이다. Meta-Rewarding은 **three-layer**(Actor, Judge, Meta-Judge)이다. four-layer stack도 상상할 수 있지만 iteration당 compute cost가 폭발한다(`K × N × M` judge calls). 논문의 ablation은 meta-judge 너머의 diminishing returns를 보인다. 이는 [[ch-44]]의 process-reward model에서 보게 될 hierarchical-evaluation pattern과 같지만, step rollout이 아니라 *judge*에 적용된다.

---

## 4. SPIN — 데이터가 judge다

[[spin]](Chen et al., UCLA, 2024)은 다른 선택을 한다. "judge"는 **human SFT response 자체**다. 모든 iteration에서 DPO pair는 문자 그대로 `(y_human, y_model_previous)`이다.

```
L_SPIN = −log σ( β · [ log(π_t(y_human) / π_{t−1}(y_human))
                       − log(π_t(y_gen)   / π_{t−1}(y_gen)  ) ] )
```

이는 DPO 그대로다. 바뀌는 것은 preference *source*뿐이다. 대수적 결과(논문의 Theorem 4.1)는 **Nash equilibrium characterization**이다. π_t가 data distribution과 일치하면 SPIN loss는 0이 된다. SPIN은 "Nash = data-matching인 two-player self-play"다.

UltraChat-200K의 Mistral-7B-SFT에서 iteration마다:
1. π_{t−1}에서 T=1.0으로 **50K (prompt, response) pairs**를 샘플링한다.
2. SFT row와 1:1로 DPO pair를 만든다. `chosen = y_human, rejected = y_gen`.
3. β=0.1, lr=5e-7, 3 epochs, batch 64로 π_{t−1}에서 π_t를 DPO 학습한다.
4. iter t+1을 위해 **reference를 π_{t−1}로 reset**한다.

Zephyr-7B-SFT + 3 SPIN iters는 HF Open LLM Leaderboard에서 Zephyr-7B-DPO(60K GPT-4 preference를 사용)와 맞먹는다. MT-Bench는 3 iters 동안 **6.39 → 7.12**로 오른다([[spin]] Table 2). headline claim은 "SFT-only data로 preference-level gains"다.

한계는 같은 동전의 뒷면이다. SPIN은 data distribution으로 수렴하므로 **SFT corpus ceiling을 넘을 수 없다**. human response가 중간 품질이면 fixed point도 중간 품질이다. 그래서 modern pipeline에서 SPIN은 보통 terminal stage가 아니라 preference-RL 이전의 *warmup*으로 실행된다([[iterative-sft-rl]] §Tülu 3).

---

## 5. Nash-LM — alignment를 preference game으로 보기

[[self-play-preference]](Munos et al., DeepMind, 2024)는 RLHF의 구조적 결함을 공격한다. **non-transitive preferences**다. Bradley-Terry reward model은 `p(a > b) · p(b > c) · p(c > a) < 1/2`(transitivity)를 가정한다. 실제 human preference는 writing style, coding idiom, humor에 대해 이를 쉽게 위반한다. non-transitive data에서 DPO는 oscillate하고, BT-reward에 대한 PPO는 임의의 mode로 수렴한다.

Nash-LM은 이렇게 다시 프레이밍한다. 모든 경쟁 정책 π에 대해 `P(π* ≻ π) ≥ 1/2`인 **Nash equilibrium** policy π*를 찾는다. Nash-MD update는:

```
π_{t+1}  ∝  π_ref · exp( η · E_{y' ~ π_t} p(· > y' | x) )
```

실제로는([[self-play-preference]] §5) preference-model score를 reward로 사용해 EMA copy of itself를 상대로 REINFORCE를 한다. preference가 non-transitive일 때 equilibrium은 **stochastic**이다. "올바른" 정책은 mode가 아니라 mixture다. 이것이 SPIN의 `Nash = data distribution`과 Self-Rewarding의 `Nash = judge's argmax`에 없는 점이다. preference conflict를 collapse시키지 않고 *존중*하는 equilibrium이다.

---

## 6. SCoRe — self-correction을 RL target으로 삼기

[[self-correct-rl]](Kumar et al., DeepMind, 2024)은 매우 구체적인 capability를 겨냥한다. **답하고, 자신이 틀렸음을 알아차리고, 수정하는 것**이다. correction trace에 대한 SFT는 두 가지 잘 문서화된 failure mode로 *실패*한다.

1. **Distribution shift** — trace는 더 강한 teacher에서 온다. student 자신의 turn-1 distribution은 다르므로 conditioning이 어긋난다.
2. **Mode collapse** — model은 turn 1에서 정답을 내고 turn 2에서 no-op하는 법을 배운다.

SCoRe의 해결책은 two-turn trajectory에 대한 **two-stage on-policy RL**이다. Turn 1: `question`. Turn 2: `question + turn-1 answer + "There might be an error. Please revise."`. Reward는 rule grader의 binary outcome correctness다.

**Stage I**은 turn-2만 학습하고, turn-1에 base model 대비 강한 **KL**을 건다.

```
∇ L_I  =  E[ ∇ log π(y_2 | x, y_1) · r(y_2) ]   +   λ_KL · KL( π(· | x) || π_ref(· | x) )
```

Stage-I KL은 turn-1 behavior를 *freeze*하여, 모델이 처음부터 정답을 내는 대신 editing을 배워야 하게 만든다. Stage I을 건너뛰면 즉시 mode collapse가 발생한다([[self-correct-rl]] Figure 5).

**Stage II**는 두 turn 모두에 대한 joint REINFORCE이며, **improvement delta**에 reward-shaping bonus를 준다.

```
R_shaped = r(y_1) + α · [ r(y_2) − r(y_1) ],   α = 2.0
∇ L_II   = R_shaped · ∇ [ log π(y_1 | x) + log π(y_2 | x, y_1) ]
```

`r(y_2) − r(y_1)`를 증폭하면 *turn 사이의 improvement*가 high-gradient direction이 된다. 결과: Gemini 1.0 Pro에서 **+15.6 pts on MATH**, **+9.1 pts on MBPP**([[self-correct-rl]] Table 2). self-correction task에서 처음으로 0을 넘은 공개 방법이다(이전 모델들은 revise step에서 *더 나빠졌다*). rollout의 한 부분을 KL term으로 freeze하고, 움직이고 싶은 부분에 advantage를 shape한다는 구조적 lesson은 agentic-RL([[ch-50]])에서 다시 나온다.

---

## 7. ReST-EM — reasoning을 위한 expectation-maximization

[[rest-em]](Singh et al., DeepMind, 2023)은 self-training loop에 가장 깔끔한 형식을 준다. rationale을 **latent variable** `z`로, answer를 observable `y`로 취급한다. 이 latent에 대한 EM은 다음과 같다.

```
E-step:   sample K=32 rationales z ~ π_{t−1}(· | x)
          keep {z : verify(answer(z), x) = correct}     # exact-match or unit-test
M-step:   π_t = argmax_π  Σ_{(x, z_kept)}  log π(z | x)
          # SFT 1 epoch, lr=1e-5, batch 128
```

이게 전부다. DPO도, reward model도, KL도 없다. E-step은 순수 rejection sampling이고, M-step은 vanilla SFT다. **verifier**가 모든 supervision을 담당한다.

[[rest-em]] Figure 2 / Table 2에 입증된 PaLM-2-L 결과:

```
MATH:   human-SFT 34.1 %   ->   ReST-EM iter 1: ~42 %   ->   iter 2: 50.6 %   ->   iter 3: flat
APPS:   human-SFT 16.4 %   ->   ReST-EM iter 2: 31.2 %
BBH held-out: gains transfer — models trained on MATH self-data improve on unrelated BBH tasks.
```

naive self-distillation과 이를 가르는 ingredient는 **diversity cap**이다. 문제당 최대 **4 distinct correct solutions**만 유지한다(논문의 ablation). 이게 없으면 iter-3에서 regresses한다. sampled distribution이 하나의 solution path로 collapse하고 M-step이 그것을 외우기 때문이다. [[star]](Zelikman 2022)는 K=1 조상으로, 실패 사례에 대해 "gold answer에서 거꾸로 rationalize"하는 trick을 사용한다. ReST-EM은 rationalization을 버리고 K를 키운다. [[v-star]]는 failed trace에서 학습한 verifier를 더해 failure도 signal에 기여하게 한다. [[ch-44]]에서 다룬 process-reward model로 이어지는 다리다.

---

## 8. R1-Zero와 2025년 해부

[[deepseek-r1]](DeepSeek-AI, 2025)은 ReST-EM filter(verifier-only)를 가져오고, M-step SFT를 **cold-start도 PRM도 없는 pure on-policy GRPO**로 바꾼다. Reward는 `r = r_acc + r_format`이며 `r_acc ∈ {0, 1}`는 rule grader에서, `r_format ∈ {0, 1}`는 `<think>…</think><answer>…</answer>`에 맞는지에서 온다. Group size 16–64, sequence length up to 32k, advantage `A_i = (r_i − mean(r_{1:G})) / std(r_{1:G})`.

SFT 기반 pipeline에서는 보이지 않았던 두 emergent behavior가 논문을 지배한다.

- **Chain-of-thought length가 학습 동안 ~400 tokens에서 10k+로 성장**한다([[deepseek-r1]] Fig. 3).
- **중간 학습 phase-transition "aha moment"**: 학습 예시에 그런 text가 전혀 없었는데도 모델이 자발적으로 *"Wait, let me reconsider…"*와 *"Let me check step 3…"* 같은 문장을 쓴다([[deepseek-r1]] Fig. 4).

2025년 후속 분석은 *실제로 무슨 일이 일어나고 있는지* 해부한다. 논문들이 R1-Zero가 무엇이고 무엇이 아닌지 아주 노골적으로 말하기 때문에 findings를 직접 요약한다.

**Finding 1 — GRPO에는 exploit 가능한 bias가 두 개 있다.** [[r1-zero-analysis]]의 Dr.GRPO(Liu et al., 2025)에 따르면 GRPO의 per-token mean aggregation은 **"길고 정답인 response와 길고 오답인 response를 비대칭적으로 보상하는 per-token mean aggregation의 length bias"**를 유도하고, per-prompt std normalization은 **"쉬운 prompt의 gradient를 부풀리는 per-prompt std normalization의 difficulty bias"**를 유도한다. fix는 bias-corrected loss다. 이것은 *std normalization을 제거*하고 batch-mean divided by `(B · L_max)`로 aggregation한다.

```
L_DrGRPO = −(1 / (B · L_max)) · Σ_{i,t} mask_{i,t} · min( r_{i,t} · A_i,  clip(r_{i,t}) · A_i )
           with A_i = R_i − μ_group   (no std division)
```

**Finding 2 — base-model prior가 load-bearing이다.** Open-Reasoner-Zero는 Qwen2.5-7B-Base에서 R1-Zero를 재현하지만 다음을 보고한다([[r1-zero-analysis]]): *"emergence는 일어나지만, base가 reasoning-pretrained가 아니면 사라진다."* 해석: pure RL은 reasoning을 처음부터 만들지 않는다. pretraining이 math/code-heavy data를 통해 조용히 설치해 둔 reasoning capability를 증폭한다. chat-pretrained base에서 같은 recipe를 다시 실행하면 length growth도 aha moment도 없다.

**Finding 3 — PRM은 필요 없다.** 2025년 세 재현([[r1-zero-analysis]] — Dr.GRPO, ORZ, TinyZero)은 모두 같은 minimum-viable ingredient list에 수렴한다. **outcome-only verifier가 충분하고 사실상 지배적**이다. [[ch-44]]의 process-reward 접근인 PRM을 더해도 도움이 되지 않고, 종종 해친다. PRM이 overoptimization에 취약한 학습된 reward source를 도입하기 때문이다. rule grader는 그렇지 않다.

**Finding 4 — RL은 확장보다 sharpening일 수 있다.** [[rlvr-beyond-base-model]](Yue et al., 2025)은 중요한 실험을 수행한다. **large-`k` pass@k** 아래에서 base model은 RL-trained model과 *맞먹거나 이를 초과*한다. 직접 인용: *"RL은 pass@1을 개선하지만, high k에서는 base model이 이긴다."* 해석: *"RLVR은 대체로 이미 존재하던 성공 경로 쪽으로 probability mass를 재분배하는 동시에 exploration을 좁히고, 풀 수 있는 문제의 더 넓은 coverage를 줄인다."* RL post-training은 **capability-boundary expansion**이라기보다 **sampling-efficiency** improvement일 수 있다. 이는 2025년 R1-Zero 스타일 학습에 대한 가장 강한 주의점이며, [[ch-46]] lab이 평균 pass@1만이 아니라 prompt-difficulty bucket 전반의 `pass@k`를 계측하는 주된 이유다.

이 네 findings는 R1-Zero를 환상으로 축소하지 않는다. claim을 정교하게 만든다. 검증 가능한 보상으로 하는 pure-RL은 분명히 (a) 긴 chain-of-thought를 *elicits*하고, (b) reward model이 없기 때문에 reward model을 *hack하지 않으며*, (c) reward source가 drift하지 않으므로 judge-based loop보다 *느리게 saturates*하고, (d) 작동하려면 base에 reasoning prior가 *필요*하다. R1-Zero가 *아닌 것*은 RL이 novel reasoning을 만든다는 증거다. 그것은 여전히 열려 있다. [[lilianweng-reasoning-llms]]의 open-questions list를 보라.

---

## 9. Filter가 곧 method다 — recap

실제로 다른 단일 knob를 통해 이 장의 방법들을 다시 읽으면:

| Filter | Method | Saturation mechanism |
|---|---|---|
| Policy-as-judge | Self-Rewarding | Judge drift at iter 3 |
| Judge + Meta-Judge | Meta-Rewarding | Meta-judge drift at iter 5 |
| Human text distribution | SPIN | Data ceiling (Nash = match data) |
| Preference game | Nash-LM | Nash equilibrium (possibly stochastic) |
| Outcome reward, two-turn | SCoRe | verifier가 깨끗한 한 reward signal이 유지됨 |
| Outcome verifier, SFT M-step | ReST-EM | Diversity collapse without cap |
| Outcome verifier, RL M-step | R1-Zero | Base-model reasoning prior |

reference-reset rule은 모든 행에 등장한다. iteration마다 `π_ref ← π_{t−1}`를 하지 않으면 DPO/RL loss가 계속 stale reference 쪽으로 정책을 당기고, iteration return은 사라진다.

**Forward to [[ch-46]].** capstone lab은 이러한 loop 중 하나를 end-to-end로 실행한다. DPO β-sweep(SPIN-like)이거나 RLVR KL-sweep(R1-Zero-like)이며, failure mode(reward hacking, entropy collapse, length inflation)를 반드시 찾게 한다. 그 lab의 모든 diagnostic은 위 표의 saturation mechanism에 대응된다.

**Back to [[ch-44]].** ReST-EM과 R1-Zero를 가능하게 하는 outcome verifier는 RLVR의 binary reward와 같은 객체다. 차이는 E-step(sample-filter-SFT)에 쓰느냐 gradient update(sample-advantage-REINFORCE) 안에 쓰느냐뿐이다. Self-improvement loop는 process/outcome supervision의 대안이 아니다. 검증 가능한 보상이 iteration을 거쳐 compound되게 만드는 *scheduling pattern*이다.
