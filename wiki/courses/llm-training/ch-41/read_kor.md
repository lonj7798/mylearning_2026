<!-- chapter: ch-41
     track: rl
     kind: content
     title: Reward Modeling
     deps: [ch-40]
     sources: [[bradley-terry-rm]], [[reward-model-overoptimization]], [[reward-ensembling]], [[pairrm]], [[generative-reward-models]], [[direct-judgement-preference]], [[nemotron-4-synthetic]], [[west-of-n]], [[rlaif-scaling]]
     figures: figures/rm-overopt.html
-->

# 제41장 — Reward Modeling

> **핵심 통찰.** Reward model은 human preference에 대한 learned proxy이고, 그것을 최적화하기 시작하는 순간 Goodhart 영역에 들어간다. proxy reward는 단조롭게 증가하고, gold reward는 `d = sqrt(KL(π‖π_ref))`에서 inverted-U를 따르며, peak는 RM의 속성이다. 더 큰 policy가 약한 RM을 구해 주지 않는다. engineering question은 "최고의 RM을 어떻게 학습할까"가 아니라 "RL에서 실제로 쓸 KL budget 안에서 proxy-vs-gold gap을 어떻게 좁게 유지할까"다.
>
> **지침.** 예상하는 failure mode에 맞는 가장 작은 RM mechanism을 고르라. stable하고 homogeneous한 preference에는 Bradley-Terry scalar, reranking task가 A-vs-B attention을 필요로 하면 PairRM, rubric steerability나 calibrated uncertainty가 필요하면 generative RM, RL 시점의 composition이 목표라면 multi-attribute head, 학습 budget이 없고 base LM이 읽을 만큼 강하면 judge-LLM을 고르라. 그런 다음 `R_proxy(t)`가 아니라 `R_gold(d)`를 monitor하고, predicted peak 전에 RL을 멈춰라.

---

## §1 Bradley-Terry from first principles

BT model은 InstructGPT부터 DPO까지 모든 preference-based post-training method 아래의 scaffold다. 한 번 유도하고 평생 믿어라.

**Setup.** prompt `x`에 대한 각 response `y`는 latent scalar quality `r(x, y)`를 갖는다. `(y_1, y_2)` pair를 본 human은 score *difference*에만 의존하는 확률로 `y_1`을 고른다. Bradley & Terry(1952)는 logistic form을 가정했다.

```
P(y_1 ≻ y_2 | x) = exp(r(x, y_1)) / [exp(r(x, y_1)) + exp(r(x, y_2))]
                 = σ(r(x, y_1) − r(x, y_2))
```

이는 score difference 위의 plain logistic regression일 뿐이다. `σ`는 latent score가 discrete choice로 어떻게 변환되는지에 대한 *가정*(logit link)에서 나온다. 유도된 것이 아니다. "your RM is well-calibrated"라는 사업 전체가 이 가정 위에 선다.

**RM loss.** `y_w`가 `y_l`보다 선택된 triple dataset `D = {(x, y_w, y_l)}`이 주어지면 negative log-likelihood는 InstructGPT RM loss([[bradley-terry-rm]])다.

```
L_RM(θ) = − E_{(x, y_w, y_l) ~ D} [ log σ(r_θ(x, y_w) − r_θ(x, y_l)) ]
```

관찰하라. 이는 label 1과 logit `r_θ(x, y_w) − r_θ(x, y_l)`을 가진 binary cross-entropy와 정확히 같다. BT = score difference 위의 BCE. 특별한 것이 없다.

**Parameterization.** `r_θ(x, y)`는 pretrained LM의 last-token hidden state 위에 앉은 linear head `w ∈ ℝ^d`의 scalar output이다. InstructGPT, Stiennon 2020, Tülu, DeepSeek 전반에서 확인된다.

**Identifiability.** Score는 *prompt별 additive constant까지*만 식별된다. `r(x, y_w)`와 `r(x, y_l)` 둘 다에 같은 상수를 더해도 loss는 바뀌지 않는다. scalar head가 이를 흡수한다. 실제로 사람들은 PPO 전에 prompt별 mean을 빼거나, KL penalty가 policy를 anchor하게 둔다. 그래서 prompt 간 `r` 값을 의미 있게 비교할 수 없다.

**Plackett-Luce.** K-way ranking에는 `P(y_1 ≻ y_2 ≻ … ≻ y_K) = ∏_i exp(r_i) / Σ_{j≥i} exp(r_j)`로 일반화한다. human이 2개보다 많은 item을 rank할 때 사용한다. loss는 chain을 따라 `log σ` term을 더한 것이다.

**Scaling.** Stiennon 2020 Fig. 2는 BT-trained RM accuracy가 TL;DR에서 RM parameter 수와 함께 단조롭게 개선됨을 보여 준다. 6B에서는 held-out human과의 RM agreement가 inter-annotator ceiling에 도달한다. 이것이 *within-distribution* scaling law다. out-of-distribution에서는 이야기가 달라지며, 그것이 §2다.

**[[bradley-terry-rm]]에서 확인된 failure modes.**
- *Length bias* — scalar `r`는 length를 free feature로 먹는다. crowd label에서 긴 response가 평균적으로 선호되기 때문이다. length를 regression out하거나 length-matched pair로 학습해 patch한다.
- *Transitivity violation* — human은 transitive하지 않지만 BT는 transitive하다. IPO와 generalized preference objective는 closed-form DPO를 포기하는 대가로 이를 우회한다.
- *Context-dependence* — BT는 `r(x, y)`가 prompt-local이라고 가정한다. cross-prompt calibration은 식별되지 않으며 믿으면 안 된다.

---

## §2 Gao 2022 — inverted-U law

Gao, Schulman, Hilton 2022는 표준 synthetic-preference overoptimization experiment를 만들었다([[reward-model-overoptimization]]). 6B "gold" RM을 ground truth로 고정하고, 그 preference로 더 작은 "proxy" RM을 학습하고, proxy에 대해 policy를 optimize한 다음, `d = sqrt(KL(π‖π_SFT))` 대비 gold reward를 plot한다.

**왜 sqrt(KL)인가.** KL은 local하게 squared distance이므로 `d = sqrt(KL)`가 많은 관계를 linearize한다. step이나 `β`가 아니라 `d`에 대해 plot하라. PPO의 β-sweep과 early-stopping은 모두 같은 `(d, R_gold)` curve를 그린다. β는 별도 knob이 아니라 KL budget의 reparameterization이다.

**Best-of-N curve.** best-of-N reranking(N ≤ ~10⁴)에 대해 fitted form은:

```
R_gold(d) ≈ d · (α_bon − β_bon · d)
```

`d`의 quadratic이다. Maximum은 `d* = α_bon / (2 β_bon)`에서다. 그 이후 gold reward는 떨어진다. proxy는 계속 winner를 고르지만, 그 winner는 gold 기준으로 점점 나빠진다. `KL_bon(n) = log n − (n−1)/n`은 BoN distribution의 analytically derived KL이므로 `d = sqrt(log n − (n−1)/n)`이다.

**PPO curve.** RL에서는:

```
R_gold(d) ≈ d · (α_RL − β_RL · log d)
```

log 때문에 PPO는 BoN의 quadratic보다 느리게 decay한다. overoptimization은 quadratic이 아니라 logarithmic하게 누적된다. PPO는 collapse 전까지 `d`에서 조금 더 멀리 탈 수 있지만, 그래도 *collapse한다*.

**Scaling with RM.** α와 β는 모두 RM parameter와 함께 부드럽게 shrink한다. 10× 더 큰 RM은 overoptimization slope를 대략 절반으로 줄이지만 peak를 없애지는 않는다. 더 큰 RM은 *더 많은* KL budget을 줄 뿐, 무한 budget을 주지 않는다.

**Policy size barely matters.** 큰 policy는 같은 proxy-vs-gold peak에 더 빨리 도달한다. proxy exploit을 더 잘하기 때문이다. 하지만 exploit은 policy가 아니라 RM의 속성이다.

**외울 empirical number.** RM ≈ 3M params일 때 gold는 `d ≈ 3` nats^0.5 근처에서 peak를 찍고, `d ≈ 8`에서는 gain 대부분을 잃는다. 더 큰 RM은 peak를 오른쪽으로 밀지만 절대 무한대로 밀지 못한다.

**Operational demand.**
1. RL 동안 `R_proxy(step)`가 아니라 `R_gold(d)`를 plot하라.
2. predicted peak 전에 멈춰라.
3. KL penalty β와 early stopping은 같은 knob이다. 둘을 독립적으로 tune하지 마라.

RM quality와 ensemble size에 대한 interactive sweep은 [figures/rm-overopt.html](figures/rm-overopt.html)을 보라. quality를 낮추면 peak가 `d = 0` 쪽으로 닫혀 오는 것을 볼 수 있다.

---

## §3 Ensembling as Goodhart insurance

Coste 2023([[reward-ensembling]])는 자연스러운 질문을 던졌다. proxy의 generalization이 bounded라면 K개의 proxy를 평균해 peak를 오른쪽으로 밀 수 있지 않을까? 그 논문은 ensemble defense로 Gao benchmark를 복제한다.

**Aggregators.** `r_1, …, r_K`를 서로 다른 seed나 data shard에서 독립 학습한 K개의 BT RM이라고 하자.

- **Mean:** `r̄(x,y) = (1/K) Σ_k r_k(x,y)` — 단순 평균. iid RM error 아래 unbiased.
- **LCB (lower confidence bound):** `r̄ − λ · std_k r_k` — disagreement에 대해 pessimistic. λ around 1이 확인된다.
- **Min:** `min_k r_k` — maximally pessimistic. 하나의 confidently-wrong RM에 robust.
- **UWO (uncertainty-weighted objective):** reward에서 `std_k r_k`에 대한 penalty를 뺀다. disagreement를 anomaly로 취급한다.

**Result.** 모든 ensemble strategy는 overoptimization을 늦춘다. peak는 K = 3–5에서 single-RM baseline의 `d ≈ 3`에서 `d ≈ 5–8`로 이동한다. Mean은 가장 높은 peak를 주고, LCB와 min은 더 안전하지만 낮게 cap된다. K = 5 이후 returns는 diminishing하다.

**Variance reduction intuition.** error variance가 각각 `σ²`인 K개의 iid RM에 대해 mean의 variance는 `σ²/K`다. 고정 `d`에서 proxy-vs-gold gap은 RM error와 함께 scale하므로, variance를 절반으로 줄이면 peak가 `d` 축에서 대략 constant만큼 오른쪽으로 간다. 이것이 전체 theoretical justification이다. 싸고, 깔끔하고, 익숙하다.

**When it fails — shared blind spots.** 모든 K RM이 systematic miscalibration을 공유하면(예: 모두 같은 length-biased dataset으로 학습됨) ensembling은 도움이 되지 않는다. Coste는 모든 RM이 wrong answer에 동의하는 adversarial prompt에서 이를 보여 준다. Ensemble diversity는 random seed뿐 아니라 *data shard*에서 와야 한다.

**Overhead.** RL step당 K번의 reward forward pass. 보통 RM이 policy보다 작으므로 감당 가능하지만, 70B-scale RM에서는 실제 compute가 된다.

**Disagreement as anomaly signal.** `std_k r_k`는 calibrated OOD flag다. high std는 policy가 RM data가 얇은 novel territory로 drift하는 것과 상관된다. LCB를 reward로 쓰지 않더라도 training diagnostic으로 드러내라.

---

## §4 PairRM과 generative RMs — non-scalar alternatives

BT scalar만 있는 것은 아니다.

**PairRM** [[pairrm]]. `(x, y_A, y_B)`를 joint-encode하고 single preference logit을 낸다.
```
f(x, y_A, y_B) → logit;   P(A ≻ B) = σ(logit)
```
두 candidate 사이의 cross-attention은 scalar RM이 놓치는 relative difference를 포착한다. absolute quality가 애매하지만 relative quality는 분명한 경우(length-matched, near-tie, paraphrase-similar case)에 특히 그렇다. PairRM-0.4B DeBERTa는 MixInstruct와 MT-Bench reranking에서 Llama-2-7B의 scalar RM과 맞먹는다.

두 가지 필수 trick:
- **Swap augmentation:** `(y_A, y_B)`와 `(y_B, y_A)`를 모두 평가하고 logit을 평균한다. train과 inference time의 position bias를 cancel한다. [[pairrm]]의 ablation은 swap만으로 2–3 pp를 보여 준다.
- **Tournament Best-of-N:** bracketed advancement를 가진 `O(N log N)` pairwise comparison. `O(N²)` pass 없이 near-optimal selection을 유지한다.

오늘날 PairRM의 production role은 PPO의 reward *shaping*이 아니라 DPO의 *pair filtering*이다. `PairRM(y_w, y_l) > τ`인 `(y_w, y_l)`만 유지한다. held-out eval에서 DPO를 약 2 pp 올린다.

**Generative RMs (GenRM)** [[generative-reward-models]]. `<LM + linear head>`를 rationale 후 verdict를 내는 LM으로 바꾼다. log-prob에서 reward를 읽는다.
```
r(x, y_A, y_B) = log P_RM("A is better" | x, y_A, y_B, rubric)
```
또는 rubric으로 `(x, y)`를 1–10 verdict로 score하고 log-prob-weighted expectation을 취하는 pointwise variant다.

핵심 속성:
- *Critique-then-verdict*는 direct verdict보다 RewardBench에서 3–10 pp를 더한다. CoT는 공짜가 아니지만 실제다.
- *Rubric steerability.* rubric prompt가 곧 reward specification이다. `"longer is not better, flag sycophancy"`는 unseen prompt에도 generalize된다. Scalar RM은 inference에서 이를 할 수 없다.
- *Calibration.* Verdict-token probability는 ground-truth agreement를 추적한다. ensembling할 때 LCB input으로 사용하라.
- *Cost.* query당 더 느리다(critique token을 생성해야 함). 하지만 base-LM inference stack을 재사용한다. 별도 scalar-RM infrastructure가 없다.

**d-RLAIF — RM을 완전히 건너뛰기.** Lee 2023([[rlaif-scaling]])은 summarization / helpful / harmless task에서 strong labeler LM의 `"Response 1 is better"` log-prob을 *매 step* reward로 쓸 수 있음을 보인다. 학습된 RM이 없다. human-eval win rate에서 classical RLAIF를 이긴다. preference label은 crowd-source보다 약 100× 싸고, policy가 drift해도 signal이 refresh된다. stale-RM 문제가 없다.

**Self-taught / J1 line** [[direct-judgement-preference]]. judgment pair에 DPO로 judge를 학습하고 반복한다(judge → prefs → new policy → new judge prefs). Self-Taught Evaluators는 약 40K synthetic pref와 3 iteration 정도로 GPT-4-as-judge를 넘는다. 위험: *judge-as-weapon* — training time과 benchmark-evaluation time에 같은 judge를 쓰면 silent leakage가 생긴다.

**West-of-N** [[west-of-n]]. human pref는 부족하지만 current RM이 괜찮을 때, N개 response를 sample하고 `argmax_RM`과 `argmin_RM`을 pair로 삼아 다음 iteration RM을 재학습한다. 한 iteration은 human preference set을 두 배로 늘리는 것과 비슷하다. N = 16이 near-optimal이다. 중요하게도 gain은 *argmax-vs-argmin*에서 나온다. argmax-vs-random은 4 pp를 잃는다.

---

## §5 Multi-attribute RMs — HelpSteer2 and compositional RL

Nemotron-4-340B([[nemotron-4-synthetic]])는 HelpSteer2 dataset으로 학습한 **5-dimensional head**를 가진 reward model을 제공한다. attribute(확인된 label)는 다음과 같다.

| Dimension | Meaning |
|-----------|---------|
| Helpfulness | response가 request를 다루는가? |
| Correctness | content가 사실적으로 맞는가? |
| Coherence | response가 잘 구조화되고 명확한가? |
| Complexity | response가 적절히 detailed한가(너무 단순하지도, 과도하게 채워지지도 않았는가)? |
| Verbosity | response가 너무 길거나, 너무 짧거나, 적절한 길이인가? |

각 dimension은 같은 LM backbone 위에 자기 scalar output head를 갖는다. label은 작은 human-annotated set에서 온 0–4 Likert rating이다(HelpSteer2의 약 10K pairs, Nemotron이 쓴 총 20K human-anchor budget 안).

**왜 multi-attribute인가.** Scalar BT RM은 모든 preference를 하나의 숫자로 collapse하고, 그 숫자는 labeling instruction이 암시한 trade-off를 모두 흡수한다. attribute별 head가 있으면 RL time에 *compose*할 수 있다.

```
r_compose(x, y) = Σ_k w_k · r_k(x, y)       (Helpfulness, Correctness, …)
```

weight `w_k`는 training 안이 아니라 *training 이후*에 설정하는 policy knob이다. safety-first를 원하면 safety head를 upweight한다. 간결한 answer를 원하면 Verbosity penalty를 downweight한다. Nemotron은 이를 사용해 verbosity와 helpfulness를 disentangle한다. unconstrained BT RM은 안정적으로 "longer = better"를 학습하고, scalar aggregation은 그것을 숨기기 때문에 중요하다.

**Nemotron pipeline의 HelpSteer2.** 5-dim RM은 (a) alignment 중 synthetic response filter, (b) DPO pair construction의 judge, (c) RPO(reward-aware preference optimization) 안의 scorer로 사용된다. Nemotron-4의 post-training data 중 98%는 synthetic이고, human pair는 약 20K만 pipeline을 ground한다. 5-dim head의 composability가 같은 RM이 서로 다른 dimension weighting으로 세 역할을 모두 하게 만든다.

**Open problem.** attribute set은 *ontological*이다. "preference가 무엇인가"에 대한 특정 이론을 encode한다. HelpSteer2의 5개 dimension은 universal하지 않다. 다른 project는 safety, factuality, harmlessness, reasoning quality를 별도 head로 사용했다. 당신이 고르는 dimension이 곧 reward specification이다.

---

## §6 Decision framework — 어떤 job에 어떤 RM인가

이 장이 답하려는 practical question이다.

| Situation | Pick | Why |
|-----------|------|-----|
| Homogeneous domain, large stable pref dataset (>50K pairs), scalar reward sufficient for PPO | **Scalar BT RM** [[bradley-terry-rm]] | 학습도 싸고 inference도 싸며 failure surface가 잘 알려졌다. 기본 선택. |
| Reranking N candidates inline (DPO pair filtering, inference-time BoN, prompt-wise selection) | **PairRM** [[pairrm]] | Joint A-vs-B attention은 작은 model size에서도 scalar subtraction을 이긴다. tournament는 `O(N log N)` selection을 준다. |
| Need rubric steerability, calibrated uncertainty, or safety-critical deployment where reasoning must be auditable | **Generative RM** [[generative-reward-models]] | rubric = inference 시 reward spec; verdict log-prob은 calibrated; critique는 audit trail을 제공한다. |
| Multiple preference axes that must be composed differently at RL time (helpfulness vs verbosity vs safety) | **Multi-attribute RM** (HelpSteer2 / Nemotron-style) [[nemotron-4-synthetic]] | attribute별 head는 retraining 없이 reweight할 수 있게 한다. |
| Strong base LM available, budget does not support RM training, or RM drift is a known problem | **Judge-LLM** (RLAIF / d-RLAIF) [[rlaif-scaling]] | label당 약 100× 싸다. d-RLAIF는 RM을 완전히 건너뛴다. 매 step fresh signal이다. |
| An RM trained on a previous-gen policy exists and the new policy lives in the same distribution | **Reuse** | 가장 싼 경로다. new policy sample에서 `R_gold(d)`를 plot해 audit하라. curve가 일찍 flat해지면 RM이 OOD drift한 것이므로 retrain이 필요하다. |
| Preference data is scarce and expensive; existing RM is decent | **West-of-N** augmentation [[west-of-n]] | 한 iter ≈ human data doubling; argmax/argmin pairing이 핵심 trick이다. |
| Overoptimization predicted (long RL horizon, strong policy, small RM) | **Ensemble** [[reward-ensembling]] on top of any choice above | K = 3–5는 peak를 `d ≈ 3`에서 `d ≈ 5–8`로 민다. safety-critical이면 LCB/min. |

**Rule of thumb.** 필요하다고 생각하는 것보다 한 row 위의 mechanism으로 시작하고, ensemble한 뒤, predicted peak 기준으로 KL을 budget하라. RM class upgrade는 비싸다. ensembling은 싸다.

**이 framework가 죽이는 anti-patterns.**
- "overoptimization을 피하기 위해 β를 tune하자" — β와 early-stopping은 같은 knob이다. 하나만 tune하고 `d`를 monitor하라.
- "더 큰 policy가 RM을 고쳐 줄 것이다" — Gao 2022: policy size는 거의 중요하지 않다. peak는 RM property다.
- "K = 10이 K = 5보다 안전하다" — K = 5 이후 diminishing returns다. seed를 더 늘리는 대신 그 compute를 data-shard diversity에 써라.
- "매 iteration RM을 scratch에서 retrain하자" — policy가 OOD drift하지 않았다면(`R_gold(d)`가 일찍 flat해지는 것으로 보임), reuse하고 fresh pref top-up만 하라.

---

## Connections

- **ch-40** (KL control in RLHF)은 `d = sqrt(KL)`을 natural axis로, β를 budget knob로 만드는 mechanism이다 [[reward-model-overoptimization]].
- **ch-42** (reward hacking taxonomy)는 위 RM choice가 trade하는 *failure modes*를 catalog한다.
- **ch-43** (entropy dynamics)은 `d`가 RM peak를 지나 커질 때 policy output distribution에 무슨 일이 일어나는지 다룬다.
- **DPO chapters** (ch-44+)는 BT의 closed-form을 이용한다. §1의 loss는 KL-regularized optimal-policy identity를 통해 재배열된 같은 loss다.
- **Synthetic data chapters** (ch-23, ch-29)는 위 모든 것의 preference pair가 나오는 곳이다. [[west-of-n]], [[nemotron-4-synthetic]], [[direct-judgement-preference]]는 모두 그 stack을 가정한다.

## Further reading

- [[bradley-terry-rm]] — loss, scaling, failure modes.
- [[reward-model-overoptimization]] — Gao 2022 scaling laws; 외워야 할 Figs. 1, 2, 5, 7.
- [[reward-ensembling]] — Coste 2023 aggregators; shared-blind-spot counterexample.
- [[pairrm]] — joint-encoder mechanics; swap augmentation; tournament BoN.
- [[generative-reward-models]] — verdict-token reward; critique-then-verdict; rubric steerability.
- [[direct-judgement-preference]] — Con-J, Self-Taught Evaluators, J1; judge-as-weapon risk.
- [[nemotron-4-synthetic]] — HelpSteer2 5-dim head; 20K human anchor와 98% synthetic.
- [[rlaif-scaling]] — d-RLAIF; CoT preference prompts; same-size labeler still helps.
- [[west-of-n]] — best/worst-of-N extremum pairing.

## Companion visualization

**[figures/rm-overopt.html](figures/rm-overopt.html)** — interactive two-panel figure. Panel A: RM quality와 ensemble size를 slide하고 inverted-U `R_gold(d)` peak가 움직이는 것을 본다. Panel B: scalar PairRM vs 5-dim HelpSteer2 RM의 five dimension calibration bar. Panel A는 x-axis의 `d`와 y-axis의 `R_gold`를 읽는 반사를 만들기 위한 것이고, Panel B는 "one scalar"가 물리 상수가 아니라 compression choice임을 보여 주기 위한 것이다.
