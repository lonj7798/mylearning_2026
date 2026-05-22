<!-- chapter: ch-24
     track: synthetic
     kind: content
     title: Modality — Reasoning Traces (CoT → Long-CoT → Step-Level)
     deps: [ch-23]
     sources: [[openmathinstruct]], [[openmathinstruct-2]], [[mathscale]], [[metamath]], [[mammoth]], [[mammoth-2]], [[rstar]], [[rstar-math]], [[s1]], [[limo]], [[step-dpo]], [[omegaprm]], [[quiet-star]], [[numina-math]]
     figures: figures/rstar-mcts.html
-->

# 24장 — Modality: Reasoning Traces (CoT → Long-CoT → Step-Level)

> **핵심 통찰.** Reasoning-trace synthesis는 *verifier*가 *teacher*보다 더 중요한 하나의 SFT modality다. OpenMathInstruct의 1.8M Mixtral trace부터 rStar-Math의 747K self-evolved trace까지, field의 모든 advance는 filter를 trace 안쪽으로 더 깊게 이동시키는 이야기다. final-answer match → step-level code execution → Monte-Carlo rollout value → pairwise step preferences. teacher model은 생성 가능한 것의 ceiling을 정한다. verifier는 무엇이 살아남을지 결정한다. verifier를 잘못 고르면 data를 더 넣을수록 student가 *나빠진다*.
>
> **가이드라인.** reasoning-trace corpus를 만들 때는 teacher가 아니라 verifier를 먼저 고르라. gold answer와 symbolic checker([[openmathinstruct-2]], [[numina-math]])가 있으면 wide하게 가라. problem count를 늘리기 전에 problem당 solution 수를 scale하라. gold answer만 있고 step supervision을 원하면 OmegaPRM-style Monte-Carlo labeling 또는 rStar-Math-style MCTS를 실행하라. gold answer도 symbolic checker도 없지만 strong base model이 있으면 small하게 curate하라([[s1]], [[limo]]). 1K reflective trace가 unchecked 1M개를 이긴다.

---

## 1. Design space map

여섯 knob이 reasoning-trace corpus를 control한다. 이 장의 모든 paper는 이 knob들의 특정 setting이다.

| Knob | Typical range | Papers anchoring each end |
|---|---|---|
| **Teacher strength** | Mixtral-8x7B → Llama-3.1-405B / GPT-4o | weak: [[openmathinstruct]]; strong: [[openmathinstruct-2]], [[numina-math]] |
| **Problem pool** | 15K seeds → 860K curated → 14M augmented | small: [[metamath]] (15K seeds × 25); large: [[numina-math]] (860K); synthesized: [[mathscale]] |
| **Solutions per problem** | 1 → 120 | K=32-64 rejection-sampling이 modal ([[openmathinstruct]]) |
| **Trace style** | short-CoT → hybrid CoT+PoT → long-reflective-CoT | short: OpenMathInstruct-2; hybrid: [[mammoth]]; long: [[s1]], [[limo]] |
| **Verifier depth** | final-answer match → step execution → MCTS Q-value → pairwise | final: most; step-exec: [[rstar-math]]; MC: [[omegaprm]]; pair: [[step-dpo]] |
| **Labeling budget** | 0 human → O(hours) hand-curation | 0: [[omegaprm]]; hand: [[limo]] |

21–23장은 synthetic data의 general case를 주장했다. 이 장은 course가 깊게 다루는 하나의 modality-specific drill-down이다. reasoning이 2024–2025년에 synthetic-data field가 실제로 *움직이는* 곳이기 때문이며, Track 4의 RL chapter(ch-48 이후)가 reasoning-trace dataset을 starting point로 consume하기 때문이다.

---

## 2. Wide-short-CoT lineage: OpenMathInstruct 1 → 2

**OpenMathInstruct-1**([[openmathinstruct]], Toshniwal et al. 2024)은 open-teacher math SFT의 modern template이다. GSM8K(7.5K) + MATH(7.5K) train set, 총 약 15K problem을 가져와 **Mixtral-8x7B-Instruct에서 problem당 K=32-64 solution**을 sample한다. 각 solution은 *tool-integrated*다. CoT text가 `<llm-code>...</llm-code>` Python block과 interleave되고, output은 `<llm-code-output>`로 다시 splice된다. SymPy-canonical equivalence(MATH) 또는 numeric string match(GSM8K)로 filter한다. survivors, 즉 평균 **GSM8K problem당 ~120 solution, MATH problem당 ~100 solution**인 1.8M (problem, solution) pair는 Apache-2.0이다.

arithmetic이 design을 강제한다. Mixtral에서 K=64 solutions/problem, trace length 평균 ~500 token, 15K problem이면 teacher token이 약 480M개 생성된다. DGX cluster에서 **~500K GPU-hours**다. payoff: OpenMath-Mistral-7B는 closed-teacher exposure 없이 80.2 GSM8K / 44.5 MATH를 달성한다.

**OpenMathInstruct-2**([[openmathinstruct-2]], Toshniwal et al. 2024)는 같은 pipeline에 세 가지 swap을 한다.
1. **Teacher**: Mixtral → **Llama-3.1-405B-Instruct**(vLLM으로 BF16 serving).
2. **Problem pool augmentation**: 15K seed를 두 teacher-prompted operation으로 ~600K까지 키운다.
   - *Paraphrase*: teacher가 각 seed problem을 다른 wording으로 rewrite.
   - *Novel question*: MATH에서 추출한 topic tag("Algebra, Level 5")로 condition해 teacher가 새 problem을 invent.
3. **Trace style**: TIR을 버리고 pure text-CoT를 쓴다. authors는 405B scale에서는 **text-CoT가 TIR을 이긴다**고 보았다. teacher의 arithmetic이 충분히 정확해 executor가 signal보다 noise를 더한다.

Output: **14M (problem, solution) pairs**, 약 650K H100-hours. OpenMath2-Llama3.1-8B는 **91.7 GSM8K / 67.8 MATH**에 도달한다. operational lesson은 *ablation*에 있다. **Llama-3.1-405B의 1M sample이 Mixtral의 10M sample을 이긴다**. teacher를 upgrade할 수 있다면 corpus scale 전에 그것을 하라.

Track-4 RL consumer에게 중요한 caveat: OpenMathInstruct-2는 **non-reflective short-CoT**다. 이를 training한 student는 backtracking이나 self-verification을 습득하지 않는다. o1-style behavior를 원하면 long-CoT lineage(§4)나 DeepSeek-R1 / o1 trace distillation이 필요하다.

---

## 3. Question-side diversity: MetaMath + MathScale + MAmmoTH

scale-the-teacher 이전에 field는 *small seed pool으로 무엇을 할 것인가*를 반복했다.

**MetaMath**([[metamath]], Yu et al. 2023)는 각 seed (Q, A) pair에 적용되는 네 question-rewrite operator를 도입했다.
1. **AnsAug** — Q를 유지하고 K CoT answer를 sample해 correct한 것만 keep.
2. **Rephrasing** — teacher가 Q를 다른 말로 rewrite하고 다시 solve.
3. **Self-Verification (SV)** — "Given Q and candidate answer A', is A' correct? If not, fix."
4. **FOBAR (Forward-Backward Reasoning)** — 영리한 operator다. "Jane has 3 apples and buys 5 more. How many?" → 8. 이를 "Jane has 3 apples and buys x more. She now has 8. What is x?" → 5로 rewrite한다. model은 *chain을 reverse로 실행하는* 법을 배운다.

**FOBAR self-verify filter**는 specific하다. augmented pair는 inverse problem에 대한 teacher solution이 known masked number를 reconstruct할 때만 keep된다. Empirically(MetaMath §4), AnsAug alone은 ~+4 GSM8K, Rephrasing +3, SV +2, FOBAR +3을 준다. **additive**다. 각 operator가 서로 다른 overfitting mode를 inoculate하기 때문이다. MetaMath-70B는 395K example에서 82.3 GSM8K / 26.6 MATH를 제공한다(vs OpenMathInstruct-2의 14M에서 91.7 / 67.8 — teacher gap이 decisive하다).

**MathScale**([[mathscale]], Tang et al. 2024)은 question augmentation을 generalize한다. seed를 rewrite하는 대신 **concept graph**를 mine한다. GPT-3.5가 각 seed problem에서 topic(~2K)과 knowledge point(~5K)를 extract한다. edge는 co-occurrence로 weight된다. **rare (topic, concept) edge**를 sample하면 teacher가 seed distribution 밖의 problem을 invent하도록 밀어준다. 2M MathScaleQA. Weakness: ground truth가 없다. teacher가 author이자 grader이므로 "gold" answer의 약 5%가 wrong이다.

**MAmmoTH**([[mammoth]], Yue et al. 2023)는 original CoT+PoT hybrid다. 260K trace, PoT:CoT ratio ≈ 57:43. PoT template은 mechanical하다.

```python
def solution():
    # Mary has 3 apples and buys 5 more.
    total = 3 + 5
    return total
print(solution())
```

Execute하고 stdout이 gold와 match하면 accept한다. **Complementary error modes**: CoT는 conceptual reasoning을 처리하고, PoT는 arithmetic precision을 처리한다. Ablation: CoT-only는 MATH를 8 point 떨어뜨리고, PoT-only는 AQuA(logic)를 15 point 떨어뜨린다. [[mammoth-2]]는 web-mining(Recall → Extract → Refine with Mixtral)으로 이를 10M까지 scale한다. volume을 위해 verifier strength를 gold-match에서 LLM-judge로 trade한다.

---

## 4. Long-CoT, small-N: s1과 LIMO

2025년의 inflection이다. Thesis: **strong base model은 이미 reasoning capability를 담고 있다. SFT의 일은 그것을 설치하는 것이 아니라 activate하는 것이다.** Corollary: 수백만 trace가 필요하지 않다. reflective trace 몇백 개면 충분하다.

**s1**([[s1]], Muennighoff et al. 2025): **59K question candidate pool**에서 시작한다. 세 filter를 순서대로 적용한다. difficulty(strong baseline도 여전히 틀리는 problem), diversity(topic spread), quality(manual trace-style check). 결과: **s1K = 1000 question-trace pairs**, trace는 Gemini가 생성. Base model: Qwen2.5-32B-Instruct. Training: FSDP로 **16 H100에서 26분**. Inference에서는 **budget forcing**을 적용한다. model이 end-of-thinking token을 emit하려 할 때마다 이를 suppress하고 `"Wait"`를 append한다. model은 계속 생각한다. 같은 checkpoint에서 forced budget만 늘려 AIME24가 50% → 57%로 오른다.

Reported numbers: **s1-32B: 56.7 AIME24, 93.0 MATH500, 59.6 GPQA-Diamond**. 1000-example SFT로 o1-preview와 competitive하다.

**LIMO**([[limo]], Ye et al. 2025): hand-curated twin. competition math, MATH, GSM8K-hard, physics olympiad를 포괄하는 **817 long-CoT samples**. selection criteria: final-answer correctness, *self-verification segment*의 존재, *branching / backtracking marker*, fine-grained step granularity. hand-filter는 right answer에 도달했지만 logic이 subtle하게 broken된 trace를 제거한다. Reported: **63.3 AIME24, 95.6 MATH500** — 둘 다 s1보다 높다. "Less-Is-More Reasoning Hypothesis"로 formalize했다. strong pretrained base + high-quality demonstration ⇒ reasoning emerges.

두 paper의 ablation을 합친 1K-vs-100K eval table:

| Recipe | Dataset size | AIME24 | MATH500 |
|---|---|---|---|
| Qwen2.5-32B-Instruct base | 0 | ~17 | ~84 |
| OpenMathInstruct-2 SFT | 14M | ~40 | ~90 |
| Random 1K from 59K | 1K | ~24 | ~86 |
| s1K curated | 1K | 56.7 | 93.0 |
| LIMO hand-curated | 817 | 63.3 | 95.6 |

Caveat가 stack된다. curator subjectivity(LIMO), base-model dependence(두 paper 모두 weak base는 1K로 activate되지 않는다고 note), benchmark-contamination risk(competition problem overlap). 그리고 gain은 long-reflective trace를 *요구한다*. 817 short-CoT trace로 대체하면 결과는 random-1K baseline에 더 가깝다. count만이 아니라 trace **style**이 일을 한다. dataset size scan interactive는 [figures/rstar-mcts.html](figures/rstar-mcts.html) Panel 2를 보라.

---

## 5. Tree search as synthesis: rStar와 rStar-Math

MCTS lineage다. **rStar**([[rstar]], Qi et al. 2024)는 inference-time-only procedure다. fine-tuning이 없다. generator small LLM이 다섯 action space로 MCTS를 실행한다.

- **A1**: one-step CoT 제안.
- **A2**: subquestion으로 decompose.
- **A3**: subquestion에 direct answer한 뒤 verify.
- **A4**: question을 simplify하도록 rephrase.
- **A5**: 새로운 intermediate subquestion 제안.

N=32 rollout에 대한 UCB selection이 candidate trajectory를 만든다. 각 trajectory는 **mutual-consistency verifier**로 check된다. *별도로 prompt된* discriminator(같은 base model, 다른 prompt)에 trajectory의 first half를 주고 complete하게 한다. `answer(generator) == answer(discriminator_completion(mask_half))`일 때만 accept한다. LLaMA2-7B GSM8K: **12.5 → 63.1** — **fine-tuning도 stronger teacher도 없이** +50.6 absolute.

**rStar-Math**([[rstar-math]], Guan et al. 2025)는 이를 training pipeline으로 바꾼다. 핵심 pseudocode:

```
# rStar-Math: MCTS node = (natural-language thought, Python code block)
def mcts_step(node):
    # Selection: descend via UCB until a leaf
    while node.children:
        node = argmax(child, key=lambda c:
            c.Q + c_puct * c.P * sqrt(node.N) / (1 + c.N))
    # Expansion: sample K next-step (thought, code) candidates from policy
    for _ in range(K):
        (thought, code) = policy.sample(prefix=node.trace)
        try:
            exec(code)                       # step-level code execution
        except Exception:
            continue                         # prune on runtime error
        child = Node(thought, code, parent=node)
        node.children.append(child)
    # Rollout: simulate to terminal; reward = 1 iff final boxed answer == gold
    leaf = rollout(node.children[0])
    r = int(extract_boxed(leaf) == gold_answer)
    # Backprop: update Q-values up the path
    backprop(leaf, r)
```

Four-round self-evolution:
1. **Round 0**: Qwen2.5-Math-7B-Instruct를 bootstrap generator로 사용한다. [[numina-math]] + olympiad + AIME archive의 747K problem에서 MCTS 실행.
2. **PPM training**: 각 problem 안에서 Q-gap > δ인 MCTS sibling이 *step-preference pair*(step_high, step_low)를 형성한다. Process Preference Model은 pairwise Bradley-Terry로 training된다. scalar PRM이 아니다. 저자들은 pairwise가 [[let-verify]]-style scalar PRM의 Goodhart pathology를 피한다고 주장한다.
3. **Policy retrain**: PPM score 기준 top-K trajectory로 다음 generator를 fine-tune한다.
4. **Rounds 1-3**: 반복. MATH는 round를 거치며 **58 → 78 → 85 → 88 → 90**으로 오른다.

Endpoint: Qwen2.5-Math-7B-rStarMath는 **90.0 MATH, 53.3 AIME24, 58.5 Olympiad**에 도달한다. 7B base에서 여러 benchmark에서 o1-preview와 맞먹는다. trajectory level에서는 여전히 gold-answer-dependent다. novelty는 inner-loop verifier로서의 step-level executability다.

---

## 6. Step-level supervision: OmegaPRM automated labels + Step-DPO

두 method가 존재하는 이유는 final-answer SFT가 false-positive "right answer, wrong reasoning" trace를 leak하기 때문이다(OpenMathInstruct-2 audit에서 ~7% rate, §2). Step supervision은 이를 닫는다.

**OmegaPRM**([[omegaprm]], Luo et al. 2024)은 step label을 자동화한다. Formal definition:

```
MC(s_t) = (1/K) · Σ_{i=1}^{K} 𝟙[rollout(policy | s_1..s_t) yields gold answer]
```

각 intermediate step s_t는 그 prefix에서 K completion을 굴렸을 때 gold에 도달하는 fraction을 soft label로 받는다. Naive cost는 trajectory당 O(L·K) rollout이다. OmegaPRM innovation은 **divide-and-conquer MCTS**다. MC가 급락하는 first step을 binary-search로 찾는다. trajectory당 O(K · log L) rollout. K=16, L=10이면 4× saving이다. ~80K problem에 대해 1.5M step label → soft MC target에 MSE로 PRM regression. weighted best-of-N(PRM × policy log-prob) selector로 사용하면 Gemini Pro MATH **51 → 69.4**.

**Step-DPO**([[step-dpo]], Lai et al. 2024)는 scalar step-value에서 pairwise step preference로 이동한다. Pipeline:
1. policy에서 K CoT를 sample한다. **wrong final answer**를 가진 것만 keep.
2. strong teacher(GPT-4 / Qwen2-72B)가 **first erroneous step**의 index를 식별한다.
3. teacher가 (problem, prefix-up-to-error)가 주어졌을 때 **corrected step**을 생성한다. 계속 풀어 final answer를 check해 verify하고, correct하면 keep한다.
4. 같은 prefix를 공유하는 triplet `(prefix_i, step_correct, step_incorrect)`를 만든다.

Update rule — Step-DPO loss, vanilla DPO와 동일한 form이지만 single-step completion에 적용:

```
L_StepDPO = -log σ( β · log[π_θ(y_w | x) / π_ref(y_w | x)]
                   - β · log[π_θ(y_l | x) / π_ref(y_l | x)] )
```

여기서 x는 multi-step prefix이고 y_w, y_l은 single reasoning step(각 30-120 token)이다. **10K pair**가 full-trajectory DPO 100K pair를 이긴다(Qwen2-7B에서 MATH 58.6 vs 54.3). gradient-dilution argument가 이를 설명한다. trajectory DPO에서는 chosen과 rejected 사이 대부분의 token이 identical이어서 KL denominator가 signal을 씻어 낸다. Step-DPO는 mass를 실제 disagreement에 집중시킨다.

두 method 모두 gold answer와 stronger teacher에 의존한다. **Zero-supervision step labeling은 open**이다. field는 아직 이를 풀지 못했다.

---

## 7. Quiet outlier: Quiet-STaR

완전성을 위한 footnote 하나. [[quiet-star]](Zelikman et al. 2024)는 post-training SFT가 아니라 **continued pretraining** 중 reasoning을 train한다. learnable start/end thought token으로 ordinary language modeling 중 여러 token position에서 latent thought span을 생성하도록 모델이 배운다. GSM8K zero-shot: 5.9 → 10.9. 이는 이 장이 build해 온 의미의 reasoning-trace corpus가 아니다. reasoning을 pretraining distribution에 embed하는 *mechanism*이다. Track-4는 post-training-only reasoning recipe에 대한 minority alternative로 이를 다시 볼 것이다.

---

## 8. Practical guidance

자신의 reasoning-trace build에 recipe를 고르는 법:

- **Gold answer + symbolic checker available**(math, 일부 code task): [[openmathinstruct-2]]로 가라. strong teacher, K=32, SymPy-filter. problem count보다 solutions/problem을 먼저 scale하라.
- **Gold answer가 있고 step label을 원함**: compute가 있으면 [[omegaprm]] divide-and-conquer MC; GPT-4-class teacher가 있으면 [[step-dpo]] triplet pipeline.
- **Gold answer 없음, strong base model 있음**: [[s1]] / [[limo]]. 1K curated traces. curator subjectivity를 예상하고, 적용하기 전에 filter rule을 적어라.
- **Gold answer 없음, weak base**: [[rstar]] mutual-consistency가 최선이다. frontier number를 기대하지 마라.
- **Reflective long-CoT 필요**: DeepSeek-R1 / o1 / Gemini-Thinking에서 distill하거나, LIMO처럼 manually curate하라. non-reasoning teacher에서 synthetic generation해도 backtracking은 나오지 않는다.

모든 team이 다시 발견하는 gotcha: **false-positive trace는 compound된다**. OpenMathInstruct-2의 ~7% right-answer-wrong-reasoning rate는 8B에서는 survivable하지만, shortcut을 배운 student는 generalize하지 못하게 된다. verifier effort에 투자하라.

## Connections

- **ch-22, ch-23** (synthetic data at scale, model collapse / verification) — 이 장은 reasoning-specific instantiation이다. ch-23의 verifier-gate principle이 Step-DPO와 OmegaPRM이 존재하는 이유다.
- **ch-25** (multi-turn conversation synthesis) — parallel modality: verifier는 다르지만(format + role-adherence) teacher-distillation backbone은 같다.
- **Track 4 (RL)**: ch-48 (RLVR), ch-52 (PRM-weighted reward) — 모두 이 장에서 만든 corpus를 consume한다. rStar-Math의 PPM이 bridge다.

## Further reading

- [[openmathinstruct]], [[openmathinstruct-2]] — wide-short-CoT template와 teacher-ablation lesson.
- [[metamath]] — FOBAR / SV / Rephrasing operators; minor change로 non-math task에 재사용 가능.
- [[mammoth]], [[numina-math]] — hybrid CoT+PoT; underappreciated problem source로서의 cn_k12.
- [[rstar]], [[rstar-math]] — MCTS + PPM; small-model-frontier story.
- [[s1]], [[limo]] — 1K-trace curation; budget-forcing inference hack.
- [[step-dpo]], [[omegaprm]] — step-level supervision; automated MC labels.
- [[quiet-star]] — post-training trace SFT의 continued-pretraining alternative.

## Companion visualization

**[figures/rstar-mcts.html](figures/rstar-mcts.html)** — two-panel interactive. **Panel 1**은 rStar-Math의 MCTS tree다. **Expand**를 click하면 UCB-selected node(highlighted)에서 다음 reasoning step을 sample한다. 각 node는 (thought, code-snippet, Q-value, visit count)를 보여 주며, code-error node는 자동으로 prune된다. Reset으로 re-seed한다. **Panel 2**는 s1/LIMO 1K-vs-100K curve다. dataset size와 trace style(short-CoT vs long-reflective-CoT)을 slider로 조절하고, AIME24와 MATH500 number를 14M OpenMathInstruct-2 baseline과 함께 plot한다. 두 panel은 함께 이 장의 thesis를 encode한다. verifier와 trace style이 맞으면 1K의 trace *quality*가 14M의 trace *quantity*와 맞먹을 수 있다.
