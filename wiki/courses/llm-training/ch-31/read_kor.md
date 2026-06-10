<!-- chapter: ch-31
     track: sft
     kind: content
     title: Iterative SFT-RL Bridges
     deps: [ch-30]
     sources: [[rejection-sampling-finetuning]], [[best-of-n]], [[star]], [[rest-em]], [[v-star]], [[iterative-sft-rl]], [[self-rewarding-lm]], [[meta-rewarding-lm]], [[spin]], [[self-play-preference]], [[llama-2]], [[llama-3]], [[interplay-pretraining-midtraining-rl]]
     figures: figures/iterative-loop.html
-->

# 31장 — 반복적 SFT-RL 브리지

> **핵심 통찰.** SFT와 RL 사이의 선은 벽이 아니다. 그것은 rejection-sampling fine-tuning이라는 *얇은 띠*이며, 공개 기록상 가장 강한 post-training 파이프라인들(Llama 2, Llama 3, Tulu 3, DeepSeek-R1의 후속 계열)은 대부분의 시간을 그 띠 안에서 보낸다. SFT 후 RL을 한 번 실행하면 플라이휠을 한 바퀴 돌릴 수 있다. 여러 라운드를 돌리면, 각 라운드의 RL이 다음 라운드 SFT를 위한 더 나은 on-policy 샘플을 만들고, 각 라운드의 SFT가 다음 라운드 RL을 위한 깨끗한 reference를 다시 설정하면서, 보고서들이 "RLHF"에 귀속시키는 실제 이득을 얻는다. verifier나 reward model이 있다면 첫 질문은 "PPO냐 DPO냐?"가 아니다. "PPO rollout loop를 쓰기 전에 RSFT를 몇 라운드 해야 하는가?"이다.
>
> **가이드라인.** Post-training 예산을 `N`라운드의 `(sample K per prompt -> score -> keep top-k -> SFT -> optional on-policy RL step -> reset reference)`로 잡아라. Llama-2는 RSFT-then-PPO 5라운드를, Llama-3는 SFT + rejection sampling + DPO 6라운드를, [[rest-em]]은 diversity control이 없을 때 2회, 있을 때 3회에서 포화됨을 입증한다. 기본값은 [[llama-2]] 기준 K=10, [[llama-3]] 기준 K=10-30, [[rest-em]] 기준 K=32이다. 라운드 사이의 reference reset을 절대 건너뛰지 마라([[iterative-sft-rl]]은 이것이 DPO/RLVR이 낡은 SFT reference로 끌려가지 않게 하는 핵심이라고 입증한다). 이 장 끝의 decision tree가 전체 내용을 하나의 flowchart로 압축한 것이다.

---

## 이 장이 존재하는 이유

30장은 SFT를 단일 pass로 다룬다. 프로덕션 파이프라인에서는 그렇지 않다. 공개 기록이 다루는 모든 대표적인 2023년 이후 chat model(Llama 2, Llama 3, Tulu 3, Qwen 2.5-Instruct, DeepSeek-V2/V3 Chat)은 *라운드* 단위로 post-training을 수행하며, 한 라운드의 표준 단위는 PPO rollout이 아니라 rejection-sampling fine-tune이다. 30장 -> 31장 -> 32장은 하나의 논리적 단위다. ch-30은 한 라운드 *안에서* 고정해야 할 design axis를 이름 붙이고, ch-31은 라운드 *사이의* loop를 이름 붙이며, ch-32는 그 loop들을 쌓아 완전한 mid-training-plus-post-training 파이프라인으로 만든다.

이 장은 RL을 "진짜" training으로, SFT를 warmup으로 취급하는 실수를 막기 위한 것이다. [[interplay-pretraining-midtraining-rl]] 2025는 인과적 주장을 깔끔하게 만든다. RL이 새로운 capability를 만드는 경우는 base model에 아직 headroom이 있고 RL prompt가 competence의 edge에 놓여 있을 때뿐이다. 그 외의 모든 경우 RL은 모델이 이미 가끔 생성할 수 있던 것들 위에서 probability mass를 다시 배치한다. Filtered self-sample에 대한 iterative SFT는 그 재배치를 수행하는 가장 싸고 안정적인 방법이다.

---

## 1. RSFT — Llama-2 appendix recipe, 인용

[[llama-2]]는 처음 세 개의 RLHF checkpoint로 Rejection-Sampling Fine-Tuning(RSFT)을 도입했다. [[llama-2]]의 attested raw-data notes(§Technical Details — Post-Training Pipeline, RLHF algorithms)에서 recipe를 그대로 인용하면 다음과 같다.

> **V1..V3:** Rejection-Sampling Fine-Tuning (RSFT). For each prompt, sample K outputs (K ~ 10+), score with combined RMs, SFT on the best sample. No policy-gradient.
>
> **V4, V5:** PPO added on top of RSFT checkpoint.
>   - **Learning rate:** 1e-6 (policy) for 70B.
>   - **KL coefficient beta:** 0.01.
>   - **Batch size:** 512.
>   - **Sequence length:** 4K.
>   - Standard PPO with clipped ratio, value function, GAE.

이 인용문에서 세 가지가 핵심적인 역할을 한다.

1. **K ~ 10+.** Reward 기준 top-1이 median보다 의미 있게 좋아지려면 충분한 후보가 필요하다. Llama-2는 K=10을 사용한다. [[llama-3]]는 K=10-30으로 넓히고, [[rest-em]]은 MATH에 K=32를 사용한다. K=4 아래에서는 selection signal이 너무 noisy하고, K=64를 넘으면 [[best-of-n]] Figure 4의 reward-model overoptimization 문제가 지배하기 시작한다. Tail을 더 깊게 캘수록 RM의 ranking은 true quality의 더 나쁜 근사치가 된다.
2. **Combined RMs.** [[llama-2]]는 helpfulness와 safety 두 모델을 훈련하고, 이를 piecewise rule로 합성한다(safety prompt에서는 safety가 지배하고, 나머지에서는 helpfulness가 지배). 두 목표를 하나의 scalar RM으로 합치면 그 piecewise 능력을 잃고 refusal-rate regression 또는 helpfulness regression 중 하나를 감수하게 된다.
3. **V1..V3에는 policy-gradient가 없다.** 이것이 핵심 주장이다. 첫 세 iteration에서 Llama-2는 *self-sample에 대한 SFT-loss만으로* AlpacaEval win-rate 이득의 대부분을 얻는다. PPO도, value function도, KL controller도 없다. V4/V5의 PPO는 기반이 아니라 마무리 동작이다.

### 실제 코드 20줄로 보는 RSFT

```python
# rsft_round.py — one round of RSFT
import torch
from typing import Callable, List

def rsft_round(policy, tokenizer, reward_fn: Callable[[str, str], float],
               prompts: List[str], k: int = 10, temperature: float = 0.8,
               top_frac: float = 1.0 / 10) -> List[tuple[str, str]]:
    """Llama-2 style RSFT: sample K per prompt, keep top-(top_frac*K) by reward."""
    kept: List[tuple[str, str]] = []
    keep_n = max(1, int(round(top_frac * k)))
    for prompt in prompts:
        cands = policy.generate(prompt, num_return_sequences=k,
                                temperature=temperature, do_sample=True,
                                max_new_tokens=1024)
        scored = sorted(
            ((reward_fn(prompt, c), c) for c in cands),
            key=lambda sc: -sc[0],
        )[:keep_n]
        kept.extend((prompt, c) for _, c in scored)
    return kept  # feed directly into next-round SFT loader

def sft_on_kept(policy, tokenizer, kept, lr: float = 2e-5,
                epochs: int = 1) -> None:
    """Standard SFT on response tokens only; prompt masked in loss."""
    # ... (use the same chat-template + loss-mask logic from ch-30)
    ...
```

Llama-2의 K=10에서 `keep_n=1`은 "10개 중 top-1" recipe이다. [[llama-3]]는 이를 넓혀 K=10-30에서 `keep_n \approx K/4`를 쓴다. 교훈은 다음과 같다. Top-1은 prompt별 품질을 최대화하지만 generation budget의 90%를 버린다. Top-k에서 k=K/4를 쓰면 budget을 더 많이 활용하고 noisy RM에도 더 관대하다.

---

## 2. Best-of-N SFT — inference-time twin

[[best-of-n]](Stiennon 2020, InstructGPT보다 앞선 summarization RLHF paper)는 전체 iterative-SFT 계열이 기대는 두 가지 사실을 확립한다.

- **BoN은 overoptimization knee까지 N에 대해 단조 증가한다.** 신뢰할 수 있는 RM이 있을 때 `N \in {4, 16, 64}`에서 BoN-64는 human pairwise preference 기준 잘 튜닝된 PPO와 2포인트 이내이며, engineering cost는 대략 1/10이고 training instability는 없다.
- **BoN에는 closed-form KL cost가 있다.** `KL(BoN_N || base) = log N - (N - 1) / N`. N=10이면 이는 ~2.3 - 0.9 = ~1.4 nats이다. PPO의 KL budget을 BoN-N과 직접 맞추고 apples-to-apples로 비교할 수 있다.

RSFT는 *training time*의 BoN이다. 매 inference request마다 BoN tax를 내는 대신, SFT dataset을 만들 때 한 번 지불하고 policy가 BoN-filtered distribution을 내재화하게 한다. BoN-KL 공식은 RSFT step이 policy를 base에서 임의로 멀리 drift시킬 수 없음을 설명한다. 각 라운드는 최대 `log K - (K-1)/K` nats의 KL에 해당하는 improvement만 살 수 있고, 이것이 라운드별 reward-hacking headroom을 제한한다.

문자 그대로의 Best-of-N SFT, 즉 자신의 best-of-N을 다시 SFT로 distill하는 것이 [[llama-3]]가 여섯 번 수행하는 일이다. Classical RSFT와의 유일한 차이는 Llama-3가 PPO 마무리 대신 DPO와 chosen sequence에 대한 auxiliary NLL loss를 사용한다는 점이다. NLL을 동반한 DPO loss는 chosen log-prob가 collapse하지 못하게 하는 DPO이며, 이는 naive DPO implementation이 round 3 이후에 자주 부딪히는 failure mode이다.

---

## 3. STaR — 최초의 self-improvement SFT bridge

[[star]](Zelikman 2022)는 ch-31의 직접적인 조상 알고리즘이다. RSFT보다 앞서며, 이 패턴을 가장 깔끔하게 진술한다.

### STaR loop

1. 소수의 rationale exemplar(few-shot)로 base LM에 prompt를 준다.
2. Unlabeled training example에 대해 rationale과 final answer를 생성한다.
3. 답이 맞으면 `(question, rationale, answer)`를 synthetic SFT supervision으로 유지한다.
4. 답이 틀리면 정답을 *제공한 채* 다시 prompt하고, 그 정답에 도달하는 rationale을 만들도록 모델에 요청한다(*rationalization backoff*).
5. 이제 정답을 산출하는 rationale만 유지한다.
6. Accepted rationale trace 전체로 모델을 fine-tune한다.
7. Fine-tuned model로 step 2부터 반복한다.

STaR가 맞게 잡았고 사람들이 2026년에도 여전히 틀리는 두 가지가 있다.

**Rationalization trick.** 틀린 답은 버려지는 sample이 아니다. Gold answer에 조건화된 supervision 후보이다. 이는 [[v-star]]가 나중에 verifier로 일반화하는 "failure가 training signal이 된다"는 아이디어의 첫 등장이다.

**Verifier-first framing.** Bottleneck은 rationale-writing이 아니라 *answer-checking*이다. STaR의 모든 후손(ReST-EM, V-STaR, RLVR, GRPO)은 다른 answer-checker를 가진 같은 loop이다. "Creative writing에도 RSFT를 할 수 있나?"라고 묻는다면 답은 "verifiable task에서 verifier가 generalize하는 만큼만"이다. 이는 곧 "reward model이 얼마나 generalize하는가?"와 같은 질문이다.

---

## 4. ReST-EM — EM으로 본 STaR

[[rest-em]](Singh 2023)은 STaR를 latent rationale variable에 대한 EM으로 재정식화한다.

- **E-step:** 문제당 K=32개 solution을 T=1.0, top-p=0.95로 sample하고, verifier-filter(exact-match 또는 unit-test)를 수행한다.
- **M-step:** Survivor에 대해 SFT한다. 1 epoch, lr=1e-5, batch 128.
- **Diversity cap:** 한 문제당 최대 4개의 distinct correct solution만 유지하여 한 경로를 memorization하는 것을 방지한다.

이 장의 중심이 되는 ReST-EM의 결과는 두 가지다.

1. **2 iteration에서 포화된다.** PaLM-2-L의 MATH에서 iter-1 +8%, iter-2 +6%, iter-3은 flat 또는 regression이다. 이것이 **iterative-SFT saturation curve**이다. "RSFT 라운드를 하나 더 돌릴까, 아니면 on-policy RL stage로 넘어갈까?"를 결정할 때 봐야 하는 신호다.
2. **Transfer effect.** MATH로 훈련하면 training distribution에 없는 Big-Bench-Hard task가 개선된다. Verifier-filtered reasoning에 대한 iterative SFT는 좁은 distillation이 아니다. Base model의 reasoning prior를 더 강하게 밀어붙이는 것이며, 그 prior는 generalize한다.

---

## 5. Iterative-SFT-vs-RL crossover — SFT 추가 라운드가 첫 RL 라운드를 이기는 때

Crossover question은 다음과 같다. Compute budget `C`가 있을 때, 이를 RSFT 한 라운드 추가에 쓸 것인가, PPO/GRPO/DPO 한 라운드에 쓸 것인가? Evidence는 regime별로 섞여 있지만 패턴은 일관적이다.

| Regime | Iterative SFT dominates | On-policy RL dominates |
|--------|-------------------------|------------------------|
| Verifier is exact-match on reasoning | Rounds 1-2; [[rest-em]] | Round 3+ if headroom remains per [[interplay-pretraining-midtraining-rl]] |
| Reward is a learned RM on chat | Rounds 1-3; [[llama-2]] V1..V3 | Final round for fine finish; [[llama-2]] V4, V5 |
| Judge is policy-as-judge | Rounds 1-3; [[self-rewarding-lm]] | After meta-judge stabilizes; [[meta-rewarding-lm]] |
| Preference data is synthetic or sparse | Rounds 1-3 (SPIN); [[spin]] | Only if preference signal is rich enough to avoid reward hacking |
| Task sits at the base model's edge of competence | First round | Subsequent rounds; [[interplay-pretraining-midtraining-rl]] |
| Task is well inside or well outside the base's competence | All rounds | Never (too easy wastes RL; too hard gives no signal) |

### Iterative vs single-round Llama-3 eval delta (attested)

[[iterative-sft-rl]] 및 [[llama-3]]에서:

| Pipeline | Structure | Reported gain over single-pass |
|----------|-----------|-------------------------------|
| Llama-2 RSFT + PPO, 5 rounds | 3x RSFT then 2x PPO | ~3-5 points AlpacaEval/MT-Bench (attested in [[iterative-sft-rl]]) |
| Llama-3 SFT + RejSample + DPO, 6 rounds | 6x (SFT + rej-sample + DPO) | Multi-round ablation is the headline; single-pass baseline is not released, but [[iterative-sft-rl]] reports the 3-5 point gap on matched-data runs |
| Tulu-3 SFT -> DPO -> RLVR | 3 distinct stages with reference reset | Removing DPO or RLVR costs 3-5 pts averaged across eval ([[iterative-sft-rl]] Tulu-3 Table 2) |
| ReST-EM on PaLM-2-L MATH | 2 rounds, SFT-only | 34.1% -> 50.6% (+16.5 pts vs human-data SFT) |
| ReST-EM on PaLM-2-L APPS | 2 rounds, SFT-only | 16.4% -> 31.2% (+14.8 pts) |

이 table을 하나의 단위로 읽어라. 가장 극적인 두 delta(+16 on MATH, +14 on APPS)는 verifier가 hard(exact-match on correct answer)할 때의 *SFT-only* iteration에서 나온다. Llama-2 / Llama-3 / Tulu-3의 delta는 더 작다(~3-5 pts). Reward signal이 더 soft(open-ended chat에 대한 RM rating)이기 때문에 각 라운드가 사는 이득이 적다.

해석: **verifier의 품질은 iterative-SFT의 ceiling을 정하고, policy-gradient infrastructure의 품질은 on-policy RL의 ceiling을 정한다.** 강한 verifier와 약한 RL infra가 있으면 SFT를 반복하라. 강한 RM/judge와 강한 RL infra가 있으면 번갈아 수행하라. 둘 다 약하다면 아직 이 장을 실행할 준비가 되지 않았다.

---

## 6. Decision tree

모든 post-training plan의 맨 위에 이 flowchart를 두고 사용하라.

```
START: SFT된 base model이 있고 한 라운드를 더 돌리고 싶다.

1. training prompt의 >= 50%에 대해 Bernoulli correct/incorrect
   label을 주는 verifier가 있는가?
   YES -> 2로.  NO -> 4로.

2. verified prompt에서 current policy의 pass@1이
   0.1과 0.8 사이인가([[interplay-pretraining-midtraining-rl]]의
   edge of competence)?
   YES -> 3으로.
   NO (< 0.1 또는 > 0.8) -> prompt가 너무 어렵거나 너무 쉽다.
        policy를 건드리기 전에 새 prompt를 curate하라.

3. 이미 verifier-SFT 라운드를 몇 번 실행했는가?
   0-1 rounds -> RSFT / ReST-EM 라운드를 하나 더 수행(K=10-32).
   2 rounds with monotone gain -> RSFT 라운드를 하나 더 수행.
   2 rounds with flat or regressing gain -> 최신 SFT checkpoint를
        reference로 두고 on-policy RL(GRPO / RLVR)로 전환.
   >= 3 rounds on same data -> pool이 고갈됐다. 더 어려운 prompt를
        curate하거나 다음 stage로 이동.

4. trained reward model 또는 reliable LLM-as-judge가 있는가?
   YES -> 5로.  NO -> iterate할 수 없다. preference를 수집하라.

5. fresh data에서 RM의 held-out accuracy가 70%를 넘는가?
   YES -> 6으로.  NO -> 먼저 더 많은 data로 RM을 재훈련하라.
        나쁜 RM 위에서 iterate하면 bias가 증폭된다.

6. RM-scored RSFT 라운드를 몇 번 실행했는가?
   0-2 rounds -> RSFT를 하나 더 수행(Llama-2 recipe: K=10, top-1).
   3 rounds -> reference-reset을 포함한 DPO로 전환([[llama-3]]
        recipe: DPO + NLL aux 0.2, beta 0.1).
   >= 4 rounds on same RM -> fresh RM을 훈련하라(Llama-2는
        [[iterative-sft-rl]] 기준 weekly fresh batches를 사용) 또는
        meta-judge regime([[meta-rewarding-lm]])으로 이동하거나 중단.

HARD RULE, every path: 각 transition에서 reference model을 reset하라.
    Reset이 없으면 DPO/RLVR은 낡은 SFT reference 쪽으로 끌려가며
    전체 chain이 누수된다.
```

---

## 7. Iterative-SFT가 도입하는 failure modes

1. **Diversity collapse.** 라운드가 거듭될수록 policy가 같은 correct solution path를 sample한다. [[rest-em]]은 문제당 distinct accepted solution을 4개로 제한한다. [[v-star]]는 diversity를 포함하는 verifier를 사용한다. Cap이 없으면 round 3부터 memorization이 시작된다.
2. **Reference drift.** DPO/RLVR stage 사이에 reference를 reset하지 않으면 KL regularizer가 stale distribution 쪽으로 끌고 가고 유용한 gradient가 사라진다. [[iterative-sft-rl]]은 이것을 Tulu-3 team이 가장 많은 debugging 시간을 쓴 failure mode라고 표시한다.
3. **Judge calibration drift (self-rewarding family).** [[self-rewarding-lm]]은 3 iters에서 포화된다. [[meta-rewarding-lm]]은 meta-judge로 이를 돌파하지만 라운드당 compute가 2배 든다. Meta-judge가 없으면 3에서 멈춰라.
4. **Length gaming.** [[meta-rewarding-lm]] Figure 5는 명시적인 length-bias control term이 없으면 response가 iteration을 거치며 2배 길어진다는 것을 보여준다. 모든 iterative-DPO recipe에는 length control이 포함되어야 한다.
5. **Policy에 대한 RM/verifier overfitting.** RM은 이전 policy의 output으로 훈련되었다. 3라운드 뒤에는 policy distribution이 RM training distribution에서 충분히 멀어져 RM accuracy가 조용히 저하된다. [[llama-2]]는 weekly fresh preference batches로, [[llama-3]]는 매 라운드 RM retraining으로 이를 고친다.

---

## 8. Connections

- **ch-30 (SFT design axes)** — RSFT의 각 라운드는 하나의 SFT이다. ch-30의 axes는 라운드별로 적용된다.
- **ch-32 (mid-training / cold-start / long-context)** — 여기의 iterative loop는 ch-32가 구축하는 multi-stage pipeline 내부의 한 stage이다.
- **ch-33 (Tulu 3, Llama 3 case studies)** — iterative pipeline의 두 canonical case study이다. ch-33은 이 장의 algorithm이 production에서 어떻게 읽히는지 보여준다.
- **Track 4 (RL)** — GRPO, RLVR, PPO는 alternation의 on-policy RL 절반이다. [[rest-em]] / [[v-star]]의 verifier primitive는 RLVR이 사용하는 reward function이다.
- **[[interplay-pretraining-midtraining-rl]]** — alternation이 언제 payoff를 내고 언제 그렇지 않은지에 대한 causal case이다.

## Further reading

- [[rejection-sampling-finetuning]] — pattern definition.
- [[best-of-n]] — Stiennon 2020; BoN-KL formula; canonical RL-vs-BoN comparison.
- [[star]] — Zelikman 2022; original rationale-bootstrap loop.
- [[rest-em]] — Singh 2023; STaR as EM; 2-iter saturation.
- [[v-star]] — Zelikman 2024; verifier learns from failures.
- [[iterative-sft-rl]] — Llama-2와 Tulu-3 multi-stage pipeline의 synthesis.
- [[self-rewarding-lm]] — actor as its own judge; 3-iter ceiling.
- [[meta-rewarding-lm]] — meta-judge breaks the 3-iter ceiling.
- [[spin]] — human-text-as-chosen을 통한 SFT-only iterative DPO.
- [[self-play-preference]] — Nash-MD game-theoretic framing.
- [[llama-2]] — 5-round RSFT + PPO.
- [[llama-3]] — 6-round SFT + rejection sampling + DPO.
- [[interplay-pretraining-midtraining-rl]] — alternation이 실제로 capability를 추가하는 때.

## Companion visualization

**[figures/iterative-loop.html](figures/iterative-loop.html)** — RSFT -> DPO -> RSFT loop를 self-contained animation으로 설명한다. N(rollouts per prompt), reward-threshold(top-fraction to keep), num-rounds를 조절하는 slider가 있고, canvas는 특정 run에 맞춘 것이 아니라 직관을 돕기 위한 per-round policy-quality curve를 animation으로 보여준다. 전체 recipe에 cluster를 투입하기 전에 어느 라운드에서 return이 flattening되기 시작하는지 감을 잡을 수 있다. Round-label을 클릭하면 accepted/rejected histogram을 pin한다.
