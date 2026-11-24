# 제24장 — Reasoning Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, Step-Level Data

> **핵심 통찰.** Reasoning-trace data는 세 방식으로 만든다. Teacher에게 문제 하나당 여러 solution을 sample하고 final answer가 일치하는 것을 유지하거나, 개별 step을 search하거나, 소수의 long trace를 선택한다. Source에서 가장 크게 측정된 data lever는 problem 쪽이다. Pair 256K를 고정하면 unique question을 1K에서 6.5K로 늘릴 때 MATH-validation이 10.5 point 올랐지만, solution의 judge/reward-model filtering은 gain이 없었다([[openmathinstruct-2]], §1, Table 3, Fig. 6). Long chain-of-thought SFT는 강한 base에서는 example 800–1,000개로 competition-math score에 도달하지만, 같은 size의 약한 base에서는 그렇지 않다. LIMO data를 Qwen2.5-32B-Instruct에 쓰면 AIME24 63.3, Qwen1.5-32B-Chat에 쓰면 9.2다. Qwen3-14B-Base에 Qwen3-32B trace로 math-only SFT하면 IFEval은 69.2에서 42.3으로 낮아지고, 같은 problem의 GRPO는 70.0이다([[limo]] §6.3.3; [[transferability-of-llm-reasoning]] Table 1).
>
> **지침.** Example budget이 고정됐을 때는 problem당 solution 수를 더하기 전에 서로 다른 problem을 더한다. OpenMathInstruct-2 ablation에서 같은 256K pair로 6.5K question은 1K question보다 10.5 point 높았다(one 8B student, math only). 새 problem을 모으거나 쓸 수 없을 때만 problem당 solution을 늘리고 더 작은 gain을 예상한다. LLaMA-2-7B에서 answer-augmented sample 80K 뒤 20K를 더해도 GSM8K는 0.1 point만 올랐다([[metamath]] §4.5). Wrong trace를 negative로 쓸 때는 first wrong step을 찾아 negative gradient를 그 step에만 적용한다. Step-DPO는 Qwen2-7B-SFT 5K pair에서 whole-answer DPO의 MATH 55.0보다 55.8, Qwen2-72B-SFT에서 62.5보다 64.1이었다(single study, run count 없음; [[step-dpo]] Table 3). 그렇지 않으면 failure를 correction target인 content로 쓰거나 discard한다. Trace가 SFT sequence length보다 길면 length를 올리거나 example을 버린다. 4,096-token s1 run은 sample 74%를 잘라 AIME24 30.0, 50.0을 냈다([[s1]] App. D.1). Model이 general해야 한다면 trace SFT 전후로 instruction following, non-math reasoning suite, perturbed math problem을 측정한다. 위 regression은 math benchmark에는 드러나지 않기 때문이다.


## 학습한 버전의 수정 사항

1. “*verifier*가 *teacher*보다 더 중요하다 … verifier를 잘못 만들면 데이터가 많아질수록 student가 더 *나빠진다*” → OpenMathInstruct-2는 ≥256K pair에서 오답 final answer solution이 최대 20%여도 SFT가 견고하고, judge 또는 reward-model filtering으로 이득이 없으며, data-scaling 이득을 이끄는 요인이 question diversity라고 찾는다. 이 장의 어떤 source도 검증하지 않은 데이터가 student를 더 나쁘게 만든다는 결과를 보이지 않는다([[openmathinstruct-2]], Abstract, §2.2.3–2.2.4, Table 3, Figs. 5–6).
2. “Chapter 21–23이 일반 사례를 논증했고”, “하나의 modality-specific drill-down” → 일반 synthetic-data 장은 ch-18부터 ch-23까지이고, ch-25부터 ch-28도 modality 장이다(outline).
3. “Track 4의 RL chapter(ch-48 이후)”, “ch-48(RLVR), ch-52(PRM-weighted reward)” → outline 순서에서 RL phase는 ch-40부터 ch-46a까지다. Process reward와 verifiable reward는 ch-44이고, ch-48은 contamination detection, ch-52는 safety evaluation이다(outline).
4. “ch-22, ch-23(synthetic data at scale …)” → ch-22는 “Quality, Diversity, and Gradient-Based Data Selection”이다(outline).
5. “Mixtral-8x7B-Instruct에서 problem당 K=32–64 solution을 sample” → Mixtral-base를 5-shot prompt와 함께 사용했다. GSM8K는 prompt당 128 sample(총 256), MATH는 prompt당 224 sample(총 896)이다([[openmathinstruct]], §2.1, Table 2).
6. “GSM8K problem당 평균 약 120 solution, MATH problem당 약 100”, “약 500K GPU-hour”, “SymPy canonical equivalence로 filter”, “Apache-2.0” → 논문에는 없다. Grading detail과 compute는 보고하지 않았고, dataset card는 commercial use를 허용하는 NVIDIA license를 제시한다([[openmathinstruct]], Verification).
7. “15K seed가 … Paraphrase … topic tag로 조건부 생성한 Novel question을 통해 약 600K로 증가” → original question과 “similar” question의 5개 few-shot pair에서 새 question을 작성하며 topic 또는 difficulty instruction은 없다. Unique question은 607.3K이고 그중 592K가 synthesized되었다([[openmathinstruct-2]], §3, Table 5).
8. “405B scale에서 text-CoT가 TIR보다 우수하다” → format ablation은 OpenMath CoT 44.5와 Llama CoT 40.6이라는 두 text format을 비교한다. Paper에는 tool-integrated 비교가 없다([[openmathinstruct-2]], Table 1).
9. “약 650K H100-hour”, “vLLM으로 BF16 serving”, “student가 … backtracking을 습득하지 않는다” → paper가 보고하지 않았다([[openmathinstruct-2]], Verification).
10. “Llama-3.1-405B가 1M sample에서 Mixtral의 10M sample을 이긴다” → 유일한 teacher ablation은 matched coverage에서 Llama3.1-405B-Instruct와 Llama3.1-8B-Base를 비교한 것이며 MATH validation은 37.9 대 30.1이다([[openmathinstruct-2]], Table 2).
11. “Self-Verification(SV) — Q와 candidate answer A'가 주어지면 A'가 맞는가? 아니면 고쳐라.” → SV는 question과 answer를 declarative statement로 다시 쓰고, 숫자 하나를 x로 mask한 뒤 x를 묻는다([[metamath]], §3.3, App. A.2).
12. FOBAR의 “Jane has 3 apples …” 예와 “AnsAug 약 +4, Rephrasing +3, SV +2, FOBAR +3 — additive” → 이 예는 만들어 낸 것이다(paper의 Example 3.4는 “x packs of beef”를 mask하며 answer는 110이다). Table 3의 LLaMA-2-7B, GSM8K 값은 original SFT 41.6, AnsAug 59.6, +Rephrasing 60.6, SV와 FOBAR를 함께 추가한 값 64.4다([[metamath]], Table 3).
13. “topic 약 2K와 knowledge point 약 5K”, “희소한 (topic, concept) edge sampling”, “‘gold’ answer 중 약 5%가 틀림” → topic은 2,018개, knowledge point는 8,892개다. Walk probability는 co-occurrence count에 비례하므로 frequent pair를 선호한다. GPT-4는 5K solution sample의 26%를 incorrect로 표시했고, 이를 고쳐도 5K run은 개선되지 않았다([[mathscale]], §4.1, Eqs. 1–2, §5.3, Table 7).
14. “PoT:CoT 비율 약 57:43”, “CoT-only가 MATH를 8 point, PoT-only가 AQuA를 15 point 하락”, `def solution()` template → Table 1 행에서는 CoT가 약 73%(derived)다. CoT-only의 MATH는 9.9 대 hybrid 31.5이고, PoT-only의 AQuA는 28.6 대 44.5다. Template는 paper에서 온 것이 아니다([[mammoth]], Tables 1, 6).
15. “[[mammoth-2]] … Mixtral을 써서 Recall → Extract → Refine, verifier strength(LLM-judge …)를 교환” → Qwen-72B가 extract하고 Mixtral-22B×8과 Qwen-72B가 refine한다. Judge-score filter는 보고되지 않았다([[mammoth-2]], §2.2–2.3).
16. s1의 “순차적 세 filter — difficulty, diversity, quality(manual trace-style check)” → 순서는 quality(API error와 formatting pattern, automated), difficulty(Qwen2.5-7B-Instruct 또는 Qwen2.5-32B-Instruct가 푼 question을 drop하며 Claude 3.5 Sonnet이 grade), diversity(MSC domain, 더 긴 trace 쪽으로 sampling)다([[s1]], §2.2).
17. 비교 table(“base 0 | 약 17 | 약 84”, “OpenMathInstruct-2 SFT 14M | 약 40 | 약 90”, “59K에서 random 1K | 약 24 | 약 86”)과 그 figure copy → 어떤 source도 이 행을 갖지 않는다. s1은 budget forcing 아래 1K-random 36.7 / 90.6 및 s1K 50.0 / 93.0 AIME24 / MATH500(Table 2), Qwen2.5-32B-Instruct 26.7 / 84.0(Table 1)을 보고한다. s1과 LIMO 어느 쪽도 OpenMathInstruct-2로 학습하지 않는다([[s1]], Tables 1–2).
18. “LIMO … 817 long-CoT sample … AIME24 63.3, MATH500 95.6” → version을 섞었다. arXiv v1은 817 sample에서 57.1 / 94.8을, v3(COLM 2025)는 800 sample에서 63.3 / 95.6을 보고한다([[limo]], v1 and v3 Abstracts).
19. LIMO의 “hand-filter가 trace를 제거 …” 및 pool “GSM8K-hard, physics olympiad” → v3는 DeepSeek R1, R1-Distill-Qwen-32B, QwQ-32B sample에서 question당 가장 높은 rule-score chain을 유지한다(length 30%, verification word 20%, tentative word 25%, connective 25%). Pool은 NuminaMath-CoT, DeepScaleR, pre-2024 AIME, MATH, Chinese exam problem이다([[limo]], v3 §3.1).
20. “817개 short-CoT trace로 대체하면 결과가 random-1K baseline에 더 가깝다” → 어느 paper도 이 실험을 하지 않는다. LIMO는 chain-quality level L1–L5(Fig. 3) 및 NuminaMath-100k short CoT(average 32.3 대 base 49.9, Table 1)를 비교한다([[limo]]).
21. rStar action “A2: subquestion으로 decompose; A3: subquestion에 직접 답한 다음 verify; A4: rephrase; A5: 새 intermediate subquestion 제안” → A1은 one-step thought, A2는 remaining thought step, A3는 answer를 포함한 next sub-question, A4는 few-shot CoT로 그 sub-question에 다시 답하기, A5는 condition을 나열하여 question을 rephrase하기다([[rstar]], §3.2).
22. “*separately-prompted* discriminator(same base model, different prompt)” 및 “12.5 → 63.1(+50.6)” → discriminator는 모든 generator에 대해 Phi3-mini-4k(3.8B)다(Phi3가 generate할 때만 self-discrimination). LLaMA2-7B의 GSM8K는 12.51 → 63.91이다([[rstar]], §4.1, Table 2).
23. rStar-Math pseudocode “c.Q + c_puct * c.P * sqrt(node.N) / (1 + c.N)” 및 “r = int(extract_boxed(leaf) == gold_answer)” → `UCT(s) = Q(s) + c·sqrt(ln N_parent(s) / N(s))`이고 `Q = q/N`, `c = 2`다. Terminal node는 +1 또는 −1을 score로 받는다([[rstar-math]], Eqs. 1–2, App. A.1).
24. “Round 0: bootstrap generator로 Qwen2.5-Math-7B-Instruct … [[numina-math]] + olympiad + AIME archive의 747K problem” → round 1은 DeepSeek-Coder-V2-Instruct(236B)를 8 rollout과 함께 쓴다. 747K problem은 주로 NuminaMath(competition level만)와 MetaMath에서 오고, MATH와 AMC-AIME로 seed한 GPT-4 problem도 포함한다([[rstar-math]], §3.4, Table 2).
25. “Q-gap > δ인 sibling이 pair를 이루며 … author가 pairwise가 Goodhart pathology를 피한다고 주장”, “PPM score 기준 top-K trajectory” → 각 step에서 correct answer로 이어지는 highest-Q candidate 두 개가 positive이고 wrong answer로 이어지는 lowest-Q candidate 두 개가 negative다. 밝힌 이유는 Q-value가 regression label로 쓰기에는 너무 imprecise하다는 것이다. SFT는 average Q 기준 top-2 correct trajectory를 쓰며 매 round base model에서 fine-tune한다([[rstar-math]], §3.3–3.4, App. A.1).
26. “MATH가 58 → 78 → 85 → 88 → 90으로 상승”, “MATH 90.0, AIME24 53.3, Olympiad 58.5” → 8개 PPM-guided trajectory에서는 58.8(base), 75.2, 86.6, 87.0, 89.4(Table 6)다. Policy만의 값은 69.6, 73.6, 75.8, 78.4(Table 3)다. 90.0 / 53.3 / 65.6 OlympiadBench에는 64 trajectory와 7B PPM이 필요하다(Table 5).
27. OmegaPRM의 “K=16 및 L=10 … 4배 절감”, “약 80K problem에 1.5M step label → MSE로 PRM regression”, “weighted best-of-N(PRM × policy log-prob)”, “OpenMathInstruct-2 audit의 약 7% rate” → k = 8 rollout 및 solution당 16 piece다. 12K MATH training question에서 1.5M annotation을 만들고, soft label에 classification loss를 쓴다. PRM-weighted majority voting은 step score의 product를 쓴다. 7% 수치는 어떤 source에도 없다([[omegaprm]], §3.2–§4; [[openmathinstruct]] §2.3는 flawed-reasoning positive를 “rare”라고 한다).
28. Step-DPO의 “강한 teacher(GPT-4 / Qwen2-72B)가 first erroneous step을 식별 … teacher가 corrected step을 생성” → first wrong step은 수동 또는 GPT-4로 찾는다. Chosen step은 reference model 자체에서 sample한다. GPT-4-corrected step의 MATH는 55.1 대 55.8이다([[step-dpo]], §3.2, Table 4).
29. “10K pair가 100K pair의 full-trajectory DPO를 이김(Qwen2-7B에서 MATH 58.6 대 54.3)” 및 “KL denominator가 signal을 씻어 낸다” → 각각 5K pair에서 Qwen2-7B-SFT는 54.8 → 55.0(DPO) 대 55.8(Step-DPO)다. Paper가 제시한 이유는 whole answer를 reject하면 그보다 앞선 correct reasoning step도 함께 reject하기 때문이다([[step-dpo]], §3.1, Table 3).
30. “두 method 모두 gold answer와 더 강한 teacher에 의존”, “non-reasoning teacher의 synthetic generation은 backtracking을 만들지 못함” → OmegaPRM은 Monte-Carlo rollout으로 step에 label하고 human 또는 stronger-model step label을 쓰지 않는다. Step-DPO는 error를 locate할 때만 GPT-4 또는 사람이 필요하다. rStar-Math는 self-reflection data나 prompt를 쓰지 않아도 search output에서 backtracking을 보인다(Fig. 4, App. A.2) ([[omegaprm]] §4; [[step-dpo]] §3.2; [[rstar-math]] §5).
31. “Track-4가 [Quiet-STaR]를 재방문한다” → outline의 어떤 chapter도 [[quiet-star]]를 나열하지 않는다. 해당 section은 trace synthesis가 아니라 continued pretraining이므로 제거했다.
32. Figure panel “Attested anchors” 및 “Curation beats volume” → 행과 predicted curve에 source가 없었다. Figure를 다시 작성했다.

이전 버전의 section map: §1 → §1; §2 → §2; §3 → §3; §4 → §4; §5 → §5; §6 → §6 및 “Negative samples and negative feedback”; §7 제거; §8 → Guideline 및 Recipe. 이 버전의 §7과 §8은 새 절이다.

## 이 장이 범용 모델에서 중요한 이유

Reasoning trace는 model이 problem과 answer 사이에 쓰는 text다. Trace data는 네 지점에서 pipeline에 들어간다. Teacher trace distillation SFT(ch-20), process reward model/step preference(ch-44), RL 전 cold-start SFT(ch-31), pretraining의 reasoning data다. [[front-loading-reasoning]]에서 8B는 pretraining에서 336B-token diverse reasoning corpus가 1.2M-example long-trace corpus보다 average 64.09 대 54.98로 높고(Table 1), SFT에서는 long-trace corpus가 diverse corpus보다 44.99 대 31.54로 높다(Table 5). 이 장의 trace dataset 대부분은 final answer를 string/symbolic match로 check할 수 있는 math다. Multi-domain OpenThoughts3 mixture조차 math 850,000, code 250,000, science 100,000 example이다([[open-thoughts]] §5).

범용 모델에는 세 가지 측정 가능한 risk가 있다. 첫째, gain은 benchmark pattern memorization일 수 있다. 25 model 중 21개에서 original GSM8K question score는 templated variant mean에서 1 standard deviation 이상 떨어지며 대개 original이 더 높다([[gsm-symbolic]] §4.1). 둘째, trace SFT가 math 밖 ability를 낮출 수 있다(§7). 셋째, long trace는 sequence length/instruction following과 상호작용한다(§7, §8). 이 장은 trace 생성·filtering과 이 세 risk를 측정하는 방법을 다룬다.

## §1 용어와 설계 선택

- **Chain-of-thought(CoT) trace**: Answer 전의 intermediate natural-language step이다. 아래 math SFT set에서는 대개 수백 token이며 OpenMathInstruct-2는 1,024 token을 넘는 solution을 버린다([[openmathinstruct-2]] App. A.2).
- **Long CoT**: Self-check와 restart를 포함하는 trace다. s1-32B는 AIME24에서 thinking token 평균 6,984개이고([[s1]] Table 8), LIMO training response는 16,384 token 미만이다([[limo]] §4).
- **Code-interpreter(tool-integrated) solution**: 실행 output을 generation 재개 전에 삽입하는 code와 text의 interleaving이다([[openmathinstruct]] §2.1).
- **Step-level data**: Whole answer가 아니라 prefix가 주어진 한 step에 붙인 label 또는 preference다.
- **Verifier**: Trace를 accept하는 check다. 이 source에서는 final-answer match, code execution, prefix에서의 rollout, second model, trained reward model이다.

| 설계 선택 | 이 장 source의 option | 근거가 있는 곳 |
|---|---|---|
| Problem source | train set(GSM8K, MATH); rewritten problem; concept-graph problem; web Q-A; competition pool | §2, §3 |
| Problem당 solution | 1–896 sample | §2 |
| Trace generator | teacher model; 학습 중인 model; tree search | §2, §4, §5 |
| Trace style | short CoT; CoT + program; long CoT | §3, §4 |
| Acceptance check | final answer; code run; Monte-Carlo value; second-model agreement | §2, §5, §6 |
| Wrong trace 사용 | discard; correction target; step-level negative | Negative samples section |

## §2 대규모 solution sampling: OpenMathInstruct-1과 -2

**정의.** Rejection-sampling distillation은 generator가 training problem마다 많은 solution을 sample한 뒤 final answer가 reference와 일치하는 solution을 유지하는 방법이다. **다루는 문제.** OpenMathInstruct-1이 등장한 2024년 2월 당시 MetaMathQA, MAmmoTH 같은 대규모 math SFT set은 상업적 사용 제한이 있는 closed model로 만들어졌다([[openmathinstruct]], Abstract).

**OpenMathInstruct-1의 메커니즘**([[openmathinstruct]], NVIDIA, arXiv 2024-02):

1. Code-interpreter format의 instruction 및 example 5개로 Mixtral-base를 prompt한다(§2.1).
2. Temperature 1.0, top-p 0.95에서 sample한다. Code block은 최대 3개이며, 첫 execution error에서 중단한다(§2.1).
3. Intermediate number를 mask한 reference solution(Mask-Text)을 prompt에 넣는다. 이것은 sample 224개에서 MATH training-set coverage를 80.1%에서 85.9%로 올렸다(Table 2).
4. Correct answer에 도달하는 solution을 유지한다. 총량은 MATH가 problem당 896 sample, 7,500 problem 중 6,978개 coverage이고, GSM8K가 256 sample, 7,473 problem 중 7,469개 coverage다(Table 2). Release set은 1.8M pair이며, training에는 fair downsampling(GSM8K)과 code-preferential selection(MATH) 뒤 약 1.02M을 사용한다(§4, footnote 10).

OpenMath-Mistral-7B는 GSM8K 80.2와 MATH 44.5에 도달한다. Numerically harder variant인 GSM-Hard에서는 63.7이며 author는 이를 perturbation에 대한 제한적인 robustness로 든다(Table 3, Limitations). Correct answer이지만 reasoning에 flaw가 있는 solution은 filter하지 않으며, author는 이를 anecdotal하게 rare라고 부른다(§2.3).

**OpenMathInstruct-2**([[openmathinstruct-2]], arXiv 2024-10)는 generator를 Llama3.1-405B-Instruct로 바꾸고 new question 592K개를 추가하며, held-out 1K MATH validation split에서 Llama3.1-8B-Base student로 SFT ablation을 수행한다(§2.2).

| Ablation(MATH validation) | 결과 | 근거 위치 |
|---|---|---|
| Solution format | OpenMath CoT 44.5 대 Llama CoT 40.6; 평균 길이 237.0 대 331.3 token | Table 1 |
| Matched coverage에서의 teacher | 405B-Instruct 37.9 ± 0.6 대 8B-Base 30.1 ± 0.6 | Table 2 |
| Low-quality solution filtering(128K) | filtered 43.0–43.8 대 unfiltered 43.6 ± 1.7 | Table 3 |
| Wrong-answer 또는 mispaired solution 추가 | ≥256K pair에서 20%까지 loss가 거의 없거나 없음 | Fig. 5 |
| 256K pair에서의 unique question | 1K → 6.5K question: +10.5 | §1, Fig. 6 |

**계산 예.** Pair가 256K일 때 unique question 1K는 `256,000 / 1,000 = 256`, 즉 question당 solution 256개를 뜻한다. Question 6.5K는 `256,000 / 6,500 ≈ 39.4`다. 두 번째 set은 question당 solution이 더 적지만 score는 10.5 point 더 높다. Noise가 20%이면 256K solution 중 51,200개가 wrong이고, accuracy 변화는 작다(Fig. 5). **근거 상태:** 결과(single study)다. Student 하나(Llama3.1-8B-Base), domain 하나, point당 run 네 번이다(§2.2). **한계.** Paper는 math만 평가한다. 70B model은 benchmark subset에서 개선되는데, author는 이를 8B validation에 맞춘 choice 때문이라고 해석한다(§4, Interpretation). New question은 solution 32개의 majority vote를 answer로 받으며 minimum agreement는 없다(App. C.1, Table 9). **함의.** ≤20%에서의 noise tolerance는 benchmark 밖 behavior에 noise가 무해함을 뜻하지 않는다. 이 scale의 math accuracy에서는 solution filtering보다 problem coverage가 더 중요했다는 뜻이다.

## §3 문제 측 다양성: MetaMath, MathScale, MAmmoTH, perturbation test

**정의.** Problem-side augmentation은 같은 problem에 solution을 더 만드는 대신 새 problem 또는 다시 쓴 problem을 만든다. **다루는 문제.** 고정된 problem set에는 solution을 더해도 곧 도움이 멈춘다. LLaMA-2-7B에서 answer augmentation은 sample 약 80K에서 GSM8K 59.6%로 포화됐다([[metamath]] §4.5).

**MetaMath**([[metamath]], arXiv 2023-09)는 temperature 0.7의 GPT-3.5-Turbo로 생성하고 answer가 ground truth와 같은 path를 유지한다(§4.1, Eqs. 1–4).

1. **AnsAug**: Original question의 새 solution을 만든다.
2. **Rephrasing**: Example 8개의 prompt로 question을 다시 쓰고, GPT-3.5-Turbo가 이를 푼다.
3. **Self-Verification(SV)**: Question과 answer를 declarative statement로 바꾼다. 숫자 하나를 x로 바꾸고 model에게 x를 묻는다.
4. **FOBAR**: 숫자 하나를 x로 바꾸고 “If we know the answer to the above question is {a*}, what is the value of unknown variable x?”를 뒤에 붙인다.

결과는 extra sample 20K에서 AnsAug가 GSM8K를 0.1 point 더한 반면 Rephrasing, FOBAR, SV는 각각 0.4, 2.3, 2.6 point를 더했다. Diversity gain과 accuracy gain의 Pearson correlation은 0.972다(§4.5). GSM8K-derived subset만으로 학습해도 MATH가 3.0에서 5.7로 올랐고, MetaMath-7B는 DROP numeric question에서 WizardMath-7B 31.5 대비 37.1을 기록했다(Table 3; App. E.4 Table 10).

**MathScale**([[mathscale]], arXiv 2024-03)은 seed question 약 20K개에서 topic 2,018개와 knowledge point 8,892개를 추출하고, random walk로 concept combination을 sample한다.

`f_co(u, v) = log(w_uv + ε)`,  `p_uv = exp(f_co(u, v)) / Σ_{v′∈N(u)} exp(f_co(u, v′))` (Eqs. 1–2)

여기서 `w_uv`는 concept u와 v를 모두 포함하는 seed question 수이고, `ε = 1e-5`, `N(u)`는 u의 neighbor 집합이다. **계산 예.** u에 count가 3과 1인 neighbor 두 개가 있다면(ε 무시) walk는 각각 3/4와 1/4 확률로 선택한다. Knowledge point를 반으로 줄이면 25K-example macro average가 상대적으로 8.6% 낮아지고, seed question을 반으로 줄이면 2.9% 낮아진다(Table 6). Solution check는 적용하지 않는다. GPT-4는 5K solution의 26%를 wrong으로 판단했고, 이를 대체해도 도움이 되지 않았다(micro/macro 10.6/11.5 대 10.2/11.1; Table 7).

**MAmmoTH**([[mammoth]], arXiv 2023-09)는 CoT와 program-of-thought(PoT) rationale을 지닌 source 13개를 섞는다. PoT rationale은 printed output이 answer인 Python program이다. Llama-2 7B의 9-set average는 CoT-only 32.0, PoT-only 41.0, hybrid 47.9다(Table 6). Generality evidence는 dataset-specific SFT와의 대비다. WizardMath-70B는 AQuA에서 20.0, Llama-2-70B는 40.9를 기록한다(Table 3).

**Perturbation test.** Memorized surface form 의존을 줄이려고 problem-side augmentation을 쓸 경우, 학습한 model을 perturbed problem으로 평가한다. 고정된 public test set의 score는 두 경우를 구분하지 못하기 때문이다. GSM-Symbolic([[gsm-symbolic]], arXiv 2024-10)는 GSM8K test question 100개를 template로 바꾸고, variant 100개로 이루어진 set 50개를 sample한다(§3.2). Gemma2-9b-it는 original에서 87.0, variant에서 79.1 ± 3.0을 기록하며 gap은 7.9 point 또는 2.6 standard deviation이다(Fig. 2). Irrelevant clause 하나를 더하는 GSM-NoOp는 o1-preview에서 report된 accuracy 17.5% 하락과 Phi-3-mini에서 65% 초과 하락을 보인다(Fig. 8a). 위 augmentation paper 어느 것도 GSM-Symbolic 결과를 보고하지 않는다. 이 data에 대해서는 Open question이다.

## §4 소규모 curated set의 긴 chain-of-thought: s1과 LIMO

**정의.** Small-set long-CoT SFT는 reasoning model이 작성한 long trace 약 1K개로 instruction-tuned model을 fine-tune한다. 아래 두 paper 모두 RL 결과가 아니라 SFT 결과다.

**s1**([[s1]], arXiv 2025-01, v3): Source 16개에서 가져온 question 59,029개에 Gemini 2.0 Flash Thinking Experimental이 trace를 생성한다(§2.1). Selection은 다음과 같다(§2.2).

1. **Quality:** API error를 제거해 54,116개가 남고, formatting problem을 제거해 51,581개가 남는다.
2. **Difficulty:** Claude 3.5 Sonnet이 grade했을 때 Qwen2.5-7B-Instruct 또는 Qwen2.5-32B-Instruct가 푸는 question을 제거해 24,496개가 남는다.
3. **Diversity:** Mathematics Subject Classification domain을 uniform하게 하나 고른 뒤, 더 긴 trace를 선호하는 확률로 question을 뽑는다. 이렇게 50 domain에 걸친 question 1,000개를 채운다.

Grader는 s1K trace 중 53.6%만 correct라고 판정하지만 incorrect trace도 유지한다(§2.2). s1-32B는 budget forcing으로 AIME24 56.7, MATH500 93.0, GPQA-Diamond 59.6을 기록한다(Table 1). **Budget forcing**은 decoding rule이다. Thinking을 늘리려면 end-of-thinking delimiter를 suppress하고 “Wait”를 붙이며, thinking을 cap하려면 delimiter를 삽입한다(§3.1). Budget forcing은 AIME24를 50.0에서 56.7로 올리고(Table 1), forced continuation 여섯 번에서 gain이 평탄해진다(§4.2). 약 30,000-token cap 아래 ablation은 다음과 같다(Table 2). AIME24는 1K-random 36.7, 1K-diverse 26.7, 1K-longest 33.3, 59K-full 53.3, s1K 50.0이다. 59K run은 H100 GPU-hour 394를, s1K는 7을 사용했다.

**LIMO**([[limo]], arXiv 2025-02; v3 2025-07, COLM 2025): 수천만 problem을 Qwen2.5-Math-7B-Instruct로 filter한다(4 attempt 안에 풀면 drop). DeepSeek-R1-Distill-Qwen-32B가 32회 중 1–3회 풀면 유지해 problem 2,125개를 얻는다(§3.1.1). Reasoning model 세 개에서 나온 chain은 rule-based score로 rank하며, 상위 800개가 set을 이룬다(§3.1.2). Qwen2.5-32B-Instruct의 결과는 다음과 같다(Table 1).

| Benchmark(pass@1) | Base | NuminaMath-100k SFT | OpenThoughts-114k SFT | LIMO(800) |
|---|---|---|---|---|
| AIME24 | 16.5 | 6.5 | 50.2 | 63.3 |
| MATH500 | 79.4 | 59.2 | 80.6 | 95.6 |
| GPQA | 48.0 | 25.8 | 42.9 | 70.7 |
| 10-benchmark average | 49.9 | 32.3 | 58.3 | 78.1 |

**Base-model dependence.** 같은 example 800개를 쓰면 Qwen1.5-32B-Chat의 AIME24는 9.2이고 Qwen2.5-32B-Instruct는 63.3이다. Qwen2.5-Instruct series에서는 3B가 2.5, 72B가 68.3이다(§6.3.3–6.3.4). **Evaluation difference.** s1은 Qwen2.5-32B-Instruct를 greedy decoding으로 AIME24 26.7이라고 보고하고, LIMO는 temperature 0.6에서 sample 4개로 16.5라고 보고한다(s1 §4.1; LIMO §5). 같은 model이 protocol에 따라 10.2 point 다르다. **상태.** Replicated: 서로 독립적인 group 두 곳이 각자의 protocol 아래 약 1K long trace가 Qwen2.5-32B-Instruct의 AIME24를 20 point 넘게 올린다고 보고한다(s1: budget forcing 없이 26.7 → 50.0; LIMO: 16.5 → 63.3). Gain의 크기는 data와 protocol에 의존한다.

**이 장 source의 distillation setting.** Stage는 전부 distill-SFT다.

| Source | Student | Teacher | Prompt | Sampling | Quality control | Size |
|---|---|---|---|---|---|---|
| [[openmathinstruct]] | Mistral-7B부터 CodeLlama-70B | Mixtral-base, 5-shot | GSM8K, MATH train | T 1.0, top-p 0.95; problem당 256 / 896 | final answer | 1.8M(1.02M 사용) |
| [[openmathinstruct-2]] | Llama3.1-8B/70B-Base | Llama3.1-405B-Instruct | train + new 592K | new question: T 0.7에서 32개 | final answer(train question); majority answer(new question); LLM decontamination | 14M |
| [[metamath]] | LLaMA-2 7B–70B | GPT-3.5-Turbo | GSM8K, MATH rewrite | T 0.7 | ground-truth answer | 395K |
| [[s1]] | Qwen2.5-32B-Instruct | Gemini 2.0 Flash Thinking | 59K pool | 보고하지 않음 | correctness에는 없음(53.6% correct) | 1K |
| [[limo]] v3 | Qwen2.5-32B-Instruct | R1, R1-Distill-Qwen-32B, QwQ-32B | hard problem 2,125개 | 보고하지 않음 | rule-based chain score | 800 |
| [[transferability-of-llm-reasoning]] | Qwen3-14B-Base | Qwen3-32B | math problem 47K개 | 보고하지 않음 | correct answer | 보고하지 않음 |

**함의.** 두 small-set 결과는 모두 math 비중이 높고, task의 일부를 이미 푸는 base에 의존한다. OpenThoughts ablation은 trace style이 중요한 이유를 보여 준다. Self-reflection phrase를 제거하면 average trace length는 11,593에서 328 token으로, average score는 51.4에서 26.3으로 내려간다([[open-thoughts]], App. H.3, Table 22).

## §5 Search 기반 trace 생성: rStar와 rStar-Math

**rStar**([[rstar]], arXiv 2024-08)는 fine-tuning 없이 inference time에 search하는 방법이다. Generator는 Monte Carlo Tree Search(MCTS)를 실행한다. MCTS는 path를 반복해 select하고, node를 expand하며, answer까지 simulate하고, 결과를 back-propagate하는 search다. 각 step은 다섯 action 중 하나로 생성한다(§3.2): A1 one-step thought, A2 remaining step, A3 answer를 포함한 next sub-question, A4 few-shot CoT로 sub-question에 다시 답하기, A5 condition을 나열해 question rephrase하기. Terminal node reward는 self-consistency majority voting의 confidence이며 path의 모든 node에 더한다. Intermediate node의 self-rewarding, external tool, trained value model은 사용하지 않는다(§3.2). Discriminator Phi3-mini-4k(3.8B)는 trajectory의 처음 20–80%를 받고 이를 complete한다. Answer가 일치하는 trajectory를 유지한다(§3.3, §4.1). Rollout 32개에서 LLaMA2-7B는 GSM8K가 12.51(few-shot CoT)에서 63.91로, commonsense task인 StrategyQA가 58.82에서 67.25로 간다(Table 2). LLaMA3-8B에서 GSM8K question 200개를 평가하면 A3만은 70.5, 다섯 action 전부는 75.0이다(Table 1).

**rStar-Math**([[rstar-math]], arXiv 2025-01)는 search를 7B policy와 7B process preference model(PPM)의 training data로 바꾼다.

1. 각 step은 natural-language comment와 Python으로 이루어진다. Code가 이전의 모든 step과 함께 실행되는 candidate만 node가 된다(§3.2).
2. Round 1–2는 terminal-guided annotation을 쓴다. `q(s_i)^k = q(s_i)^{k−1} + q(s_d)^k`(Eq. 2)에서 `q(s_i)^k`는 rollout k 뒤 step `s_i`의 value, `q(s_d)^k`는 terminal answer가 ground truth와 같으면 +1이고 다르면 −1이며, `q(s_i)^0 = 0`이다.
3. Selection은 `UCT(s) = Q(s) + c·sqrt(ln N_parent(s) / N(s))`, `Q(s) = q(s)/N(s)`를 사용한다(Eq. 1). `N(s)`는 s의 visit count, `N_parent(s)`는 parent의 visit count이고, `c = 2`다(App. A.1).
4. SFT data는 problem마다 average Q 기준 top-2 correct trajectory다. Policy는 매 round base model에서 다시 train한다(§3.4.1, App. A.1).
5. PPM data는 각 step에서 correct answer로 가는 highest-Q candidate 두 개와 wrong answer로 가는 lowest-Q candidate 두 개로 구성한다. Loss는 `L_ppm = −(1/(2×2)) E[log σ(r_θ(x, y_i^pos) − r_θ(x, y_i^neg))]`이다(Eq. 4). 여기서 `r_θ`는 PPM score, `x`는 problem, `y_i`는 step i까지의 trajectory다.
6. Round 1은 DeepSeek-Coder-V2-Instruct(236B)를 rollout 8개와 함께 쓴다. Round 2–4는 7B model을 rollout 16개와 함께 쓰며, round 3부터 PPM이 initial q value를 공급한다(§3.2, §3.4.2).

**계산 예.** Rollout 네 번에서 +1, +1, −1, +1 outcome을 받은 step은 `q = 2`, `Q = 0.5`다. `N_parent = 10`, `c = 2`이면 `UCT = 0.5 + 2·sqrt(ln 10 / 4) = 0.5 + 2·0.759 = 2.017`이다. [figures/rstar-mcts.html](figures/rstar-mcts.html)의 Panel A는 16-leaf tree에서 이 annotation을 실행한다. 독자는 rollout을 한 단계씩 진행하며 어떤 trajectory가 SFT data가 되는지와 어떤 step-1 candidate가 PPM pair를 이루는지 볼 수 있다.

**근거.** Round-4 data로 fine-tune한 Qwen2.5-Math-7B는 MATH 78.4에 도달한다. 이는 ORM을 쓴 rejection sampling의 73.4, 같은 policy의 random self-sample의 72.4보다 높다(Table 7). Problem coverage는 네 round 동안 problem 747K개의 60.17%에서 90.25%로 상승한다(Table 2). §4.2에서 보고한 MATH 90.0 / AIME24 53.3은 PPM으로 score한 trajectory 64개가 필요하다. Greedy policy는 78.4 / 26.7이며, MATH에서 Qwen2.5-Math-7B-Instruct의 82.6보다 낮다(Tables 5, 10). Q-value에 MSE로 train한 PQM은 88.2이고 PPM은 89.4다(Table 8). **한계.** Word problem만 다룬다. Code나 general task에는 test case, human label 또는 mutual verification이 필요하다(§5). Round 4 뒤 sample한 unsolved problem 20개 중 19개는 reference answer가 틀렸다(§3.4.2).

## §6 Step-level data: Monte-Carlo label과 step preference

**Monte-Carlo step label.** Question q와 prefix `x_{1:t}`가 주어지면 completer는 rollout k개를 sample한다. `c_t = (correct rollouts from step t) / (total rollouts from step t)`([[omegaprm]] Eq. 1)이며, final answer가 reference와 같으면 rollout은 correct다. Math-Shepherd의 hard label은 `1[c_t > 0]`이다([[math-shepherd]] §3.3).

**OmegaPRM**([[omegaprm]], arXiv 2024-06, v2)은 binary search로 first error를 locate한다. Midpoint m에서 rollout하고 `c_m > 0`이면 error는 second half에, 그렇지 않으면 first half에 있다. Step 하나가 남을 때까지 반복한다(§3.2). Step M개라면 cost는 O(kM) policy call이 아니라 O(k log M)이다. **계산 예.** Paper setting인 k = 8, solution당 piece 16개를 쓰면 per-step estimation에는 solution당 `8 × 16 = 128` rollout이, binary search에는 `8 × log₂16 = 32` rollout이 필요하다(derived). PRM은 MATH training question 12K개에서 soft-label annotation 1.5M개로 학습한다. Sample 64개에 대한 PRM-weighted majority voting에서 Gemini Pro는 MATH500 69.4에 도달한다. 이는 majority voting의 67.2, PRM800K-trained PRM을 쓴 67.6보다 높다(Table 1).

**Step-DPO**([[step-dpo]], arXiv 2024-06)는 first wrong step에서 pair를 만든다(§3.2).

1. Reference model에서 step-formatted answer(“Step i:”)를 sample하고 wrong final answer인 것을 유지한다.
2. First wrong step `s_k`를 수동 또는 GPT-4로 찾는다.
3. `s_{1:k−1}`이 주어진 reference model에서 continuation을 sample하고 gold answer에 이르는 것을 유지한다. 그 first step을 `s_win`으로 삼는다.

`L(θ) = −E[log σ(β log(π_θ(s_win | x; s_{1:k−1}) / π_ref(s_win | x; s_{1:k−1})) − β log(π_θ(s_lose | x; s_{1:k−1}) / π_ref(s_lose | x; s_{1:k−1})))]` (Eq. 2)

여기서 `π_θ`는 학습하는 policy, `π_ref`는 frozen reference, `σ`는 sigmoid, `s_lose = s_k`이고, `β`는 reference에서의 distance를 정한다(0.4, 72B에서는 0.5; §4.1). Pair 10K개에서 SFT model 대비 MATH gain은 Qwen2-7B의 +1.0부터 Qwen2-72B의 +3.0까지다(Table 1).

## §7 Math 밖으로의 전이: 같은 data에서 SFT 대 RL, 그리고 instruction following

**문제.** 이 장의 trace dataset 대부분은 math이지만, 범용 model은 code, science, instruction following, conversational question answering에서도 평가된다. **근거 1**([[transferability-of-llm-reasoning]], arXiv 2025-07, v2): Qwen3-14B-Base를 math problem 47K개로 학습한다. 하나는 Qwen3-32B trace에 SFT하고, 다른 하나는 answer correctness에 GRPO를 적용한다(App. A.3).

| Qwen3-14B-Base variant | Math avg | Other reasoning avg | Non-reasoning avg | IFEval |
|---|---|---|---|---|
| Base | 27.7 | 30.2 | 45.7 | 69.2 |
| SFT, thinking trace | 49.8 | 45.3 | 21.1 | 42.3 |
| SFT, non-thinking trace | 32.3 | 45.2 | 29.0 | 41.4 |
| RL(GRPO) | 53.8 | 60.0 | 53.2 | 70.0 |

Table 1이다. Qwen3-8B-Base ablation은 sampling distribution을 분리한다. Teacher trace에 off-policy SFT를 적용하면 non-reasoning transferability index는 −40.5, model 자체에서 rejection sampling한 on-policy SFT는 +30.2, on-policy RL은 +32.4다(Table 4). **한계.** SFT learning rate는 `5×10⁻⁵`, RL learning rate는 `1×10⁻⁶`이므로 update size가 matched되지 않았고(App. A.3.1), author는 이를 ablate하지 않는다. **해석.** Self-generated trace(rStar-Math round 2–4, Step-DPO의 in-distribution chosen step)는 teacher trace보다 on-policy data에 가깝다. 이 두 paper는 어느 쪽도 math 밖 전이를 측정하지 않는다.

**근거 2**([[scaling-reasoning-losing-control-mathif]], arXiv 2025-05): MathIF는 Python-checked constraint가 있는 math problem 420개에서 hard accuracy(모든 constraint 충족)와 soft accuracy(충족한 비율)를 측정한다. **계산 예.** Constraint 3개 중 2개를 충족한 query는 hard accuracy 0, soft accuracy `2/3`이다(Eq. 1). Qwen2.5 base 네 개를 대상으로 한 controlled run에서 long-CoT SFT와 GRPO는 variant 16개 중 15개에서 두 score를 모두 낮추면서, 같은 16개 중 15개에서 math accuracy를 올렸다(Table 4에서 계산). Qwen2.5-7B +SFT는 hard accuracy가 15.95에서 7.86으로, math accuracy가 13.59에서 23.10으로 변했다. Release model 중 s1-32B의 IFEval prompt-level strict는 58.04이고 Qwen2.5-32B-Instruct는 80.96이다(App. G, Table 13; controlled comparison은 아님). RL rollout cap을 token 1k에서 8k로 올리면 math accuracy는 28.73에서 39.82로 오르고, hard accuracy는 19.05에서 14.29로 내려간다(Table 5). **상태.** SFT가 유발하는 instruction-following loss의 방향은 두 연구에서 replicated됐다(Huan et al. Table 1; MathIF author인 Fu et al. Table 4). Base와 data는 서로 다르다. RL에서는 두 연구가 맞지 않는다. Huan et al.에서는 GRPO가 IFEval 69.2 대비 70.0을 유지했지만, cold-start GRPO는 Qwen2.5 base 네 개 모두에서 MathIF hard accuracy를 낮췄다(예: Qwen2.5-7B 15.95 → 10.48, Table 4). Benchmark, base, training data가 다르므로 원인은 Open question이다.

**근거 3**([[open-thoughts]], App. L, Table 30): Safety data 없이 math, code, science trace 1.2M개로 학습한 OpenThinker3-7B의 HarmBench harmfulness rate는 55.5이며, Qwen2.5-7B-Instruct의 14.5와 비교된다.

## §8 Trace length와 SFT sequence length의 관계

**문제.** Long-CoT target은 SFT maximum sequence length보다 길 수 있다. Truncation으로 제거되는 token은 마지막 token, 즉 answer, end-of-answer delimiter, stop token이다.

**메커니즘.**

1. Prompt length `L_p`, target length `L_t`, maximum length `L_max`일 때 학습되는 target token의 비율은 `min(1, (L_max − L_p) / L_t)`다.
2. `L_p + L_t > L_max`이면 이 trace 뒤에 answer를 emit하거나 stop하도록 model을 가르치는 training token은 없다.
3. s1 author가 결과를 보고한다. 더 짧은 training length에서는 “the answer section of the training sample is more commonly cut off”된다. Answer는 gradient update를 더 적게 받고, test time reasoning은 더 길어진다([[s1]] App. D.1).

**계산 예.** Prompt 300 token, trace 12,000 token, `L_max = 4,096`이라면 `(4,096 − 300) / 12,000 = 31.6%`의 target token만 학습된다. Answer와 stop token을 포함한 마지막 8,204 token은 제거된다.

**근거.** Qwen2.5-32B-Instruct에서 s1의 결과는 다음과 같다(Table 8). Training length 4,096은 sample의 74%를 잘라 AIME24 30.0%, average thinking token 20,721개를 냈다. 32,768은 0%를 자르고 AIME24 50.0%, token 6,984개를 냈다(2.97배 더 짧음). LIMO는 response 전체가 맞도록 length limit을 설정하며 response는 모두 16,384 token 미만이다([[limo]] §4). MathIF의 controlled SFT는 대신 8,192 token을 넘는 trace를 drop한다([[scaling-reasoning-losing-control-mathif]] §5.2). s1 author는 context window가 budget forcing의 추가 scaling을 제한한다고 말한다. Model에 최대 512 step을 사용하라고 prompt한 별도 run에서는 AIME24 response 30개 중 12개가 context window를 넘었다([[s1]] §6.2).

**Reasoning SFT 이전의 long-context 능력**([[longer-context-deeper-thinking]], arXiv 2025-05): RoPE θ scale만 다른 LLaMA3-8B-Instruct variant를 동일한 OpenR1-Math-220K split(≤8K, 8K–16K token)에 fine-tune했다. 32K needle-in-a-haystack score가 가장 높았던 θ×16(×1의 0.00 대비 77.05)은 SFT 뒤 MATH500 59.36 대 54.80(short와 long run의 average; Tables 2–3), science SFT 뒤 GPQA 41.92 대 37.27, MMLU-STEM 74.27 대 71.06을 기록했다(Tables 4, 6). Qwen2.5-Math-7B-Instruct(4K context)에서는 long-trace SFT가 short-trace SFT보다 낮았다(83.80 대 86.28). Context를 확장한 뒤에는 long이 short보다 높았다(89.12 대 88.28)(Table 7). **한계.** SFT만 다루고, 7B–14B이며, run 수는 적혀 있지 않다(§5; recipe ledger). **함의.** 두 결과는 서로 다르다. LLaMA3-8B-Instruct에서는 paper가 method를 설명하지 않은 effective length 약 9K로 추정한 ×1을 포함해 모든 θ factor에서 8K–16K split이 ≤8K split보다 높았다(Tables 2–3). 반면 Qwen2.5-Math-7B-Instruct에서는 context를 확장하기 전까지 long split이 short split보다 낮았다(Table 7). 계획한 trace가 model의 측정된 effective context보다 길면, long-trace SFT 전에 context를 확장하고 long/short split을 비교한다. 두 연구 모두 extended model이 SFT 후 더 높은 score를 기록했기 때문이다(Tables 3, 7). 그렇지 않으면 같은 split comparison으로 trace length를 고른다(ch-32b, ch-32c).

## Negative sample과 negative feedback

**여기서 쓰는 네 가지 의미**(course standard): (1) negative marginal value: positive target으로 쓰면 해로운 sample, (2) content로서의 negative: input 또는 corrected target에 failure를 넣고 cross-entropy로 학습, (3) conditioning으로서의 negative: control token 아래에서 failure를 학습, (4) gradient로서의 negative: sample likelihood를 명시적으로 낮춤. (4)만 sample에서 probability mass를 제거한다.

**1. Trace synthesis에서 negative가 오는 곳.** Rejection sampling의 wrong final answer([[openmathinstruct]]는 verifier training용으로 6.6M개를 release하지만 쓰지 않음, §1), manual 또는 GPT-4가 찾은 first wrong step([[step-dpo]] §3.2), `c_t = 0`인 Monte-Carlo label([[omegaprm]] §3.2), wrong answer로 이어지는 low-Q search candidate([[rstar-math]] §3.3), GPT-4 error explanation([[lema-learning-from-mistakes]] §2.1)이다. **Label noise.** OmegaPRM은 너무 어려운 question이 false negative를 만들 수 있다고 하며, rollout 32회에서 correct가 0개 또는 wrong이 0개인 question을 filter한다. 남은 noise의 효과는 uncertain하다고 명시한다(App. A, §5). Math-Shepherd hard label은 N = 4에서 human-labeled step 160개 대비 86% accurate이며, larger N에서는 false positive 때문에 덜 정확하다([[math-shepherd]] §5.2).

**2. Current practice가 하는 일.** Discard(type 1): OpenMathInstruct, MetaMath, MAmmoTH이며, s1은 correctness filter를 적용하지 않는다. Content(type 2): LEMA는 `(question + wrong path) → (wrong step, explanation, corrected solution)`으로 학습하고 loss는 correction에만 둔다(App. B.2). Conditioning(type 3): 이 장에는 source가 없으며 ch-31a를 보라. Gradient(type 4): Step-DPO의 rejected step이다. Reward-model label: OmegaPRM과 rStar-Math PPM은 negative를 classifier/ranking label로 쓰며 policy gradient로 쓰지 않는다.

**3. 메커니즘.** Next token에 대한 softmax의 logit z에 대해 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이다. Step η의 `log p_y`에 gradient descent를 적용하면 `z_y`는 `−η(1 − p_y)`만큼, 다른 모든 `z_j`는 `+η p_j`만큼 바뀐다. 그러므로 가장 probable한 alternative의 logit이 가장 크게 증가한다. **계산 예.** Probability `(a, b, y) = (0.80, 0.15, 0.05)`, `η = 1`이면 logit change는 `(+0.80, +0.15, −0.95)`이고 new probability는 `(0.902, 0.088, 0.010)`이다. Penalty를 받지 않은 token b도 mass를 잃고 a는 0.102를 얻는다(computed). [figures/rstar-mcts.html](figures/rstar-mcts.html)의 Panel C는 임의 input에서 이를 다시 계산하고, 임의 answer length 및 first-error position에서 penalized step 수를 센다. **Credit assignment.** Chosen/rejected answer를 independently sample하면 rejected answer의 모든 step은 rejected term에만 나타난다. First error가 step 7인 10-step answer는 correct step 6개, wrong step, 그리고 그로부터 따라오는 step 3개를 negative gradient에 기여시킨다. Whole-answer DPO가 prefix `s_{1:k−1}`을 공유하는 chosen/rejected answer를 쓴다면 prefix log-ratio term은 양쪽에서 같아 margin에서 cancel된다(derived). Step-DPO는 prefix를 conditioning에 넣고 `s_k` 뒤 continuation을 drop하므로 `s_k`만 negative gradient를 받는다. Step-DPO의 명시적 이유도 같다. Whole answer를 reject하면 “may also discard preceding correct reasoning steps”이다(§3.1). **Similarity risk.** 같은 prefix에서 condition된 chosen/rejected step은 hidden embedding이 비슷할 것으로 예상된다(Interpretation; Step-DPO에서 측정하지 않음). [[likelihood-displacement]]에서는 centered hidden-embedding similarity(CHES)가 더 높은 pair일수록 chosen response의 log probability가 더 낮아졌다(§4.2.2 theory, §5 Fig. 2). 이 연구는 reasoning step이 아닌 chat/refusal pair를 썼으므로 Step-DPO pair의 risk는 Open question이며, Step-DPO는 chosen-step log probability를 보고하지 않는다.

**4. 수치를 포함한 근거.** Step-DPO 대 DPO의 benefit은 5K pair에서 Qwen2-7B-SFT 55.8 대 55.0, Qwen2-72B-SFT 64.1 대 62.5 MATH다(Table 3). In-distribution chosen step은 GPT-4 correction보다 55.8 대 55.1로 높다(Table 4). Author는 이를 out-of-distribution text의 낮은 reference log probability 때문이라고 해석한다(§3.2, Interpretation). LEMA는 model 5개 전반에서 GSM8K +1.5부터 +2.9를 더하며(Table 1), matched size에서 5개 중 4개(LLaMA-2-7B 제외), matched token에서도 gain을 유지한다(LLaMA-2-70B 82.1 → 83.5)(Fig. 4, Table 2). LLaMA-2-7B에서 wrong final answer를 가진 GPT-3.5-Turbo path 7,473개로 SFT하면 GSM8K 43.6으로, original training solution 7,473개의 41.6보다 높고 correct-answer path 7,473개의 52.2보다 낮다([[metamath]] §4.7, Table 4). Author는 이 path의 일부 intermediate step이 correct하기 때문이라고 가정한다(Interpretation). Step-DPO와 LEMA는 run count나 variance 없이 single number를 보고한다. **효과 크기.** 이 장 source 중 trace data의 negative와 positive가 improvement에 각각 기여한 비율을 isolate한 것은 없다. Huan et al.은 negative gradient를 여러 요인 중 하나로 부르지만 credit assignment와 함께 ablate한다([[transferability-of-llm-reasoning]] §5.2).

**5. Control.** First wrong step에 localize한다(Step-DPO, OmegaPRM). Chosen step은 model 자체 sample에서 얻는다(Step-DPO Table 4). Monte-Carlo outcome이 모두 correct 또는 모두 wrong인 question을 drop하고(OmegaPRM App. A), trajectory가 50% 미만 correct인 synthetic problem도 drop한다. 이는 rStar-Math가 wrong reference answer를 막는 데 쓴다(App. A.1). [[likelihood-displacement]] §6.3처럼 length-normalized CHES가 가장 낮은 pair만 학습한다. 이는 pair의 5%를 유지했으며 refusal data에서만 시험됐다. Gradient가 필요하지 않으면 failure를 correction content로 쓴다(LEMA).

**6. 진단.** Chosen/rejected log probability를 별도로 log한다. Step-DPO Fig. 2는 DPO에서 plateau하는 reward margin을 추적한다. Correct solution에서 나온 step 중 `c_t = 0`을 받는 비율을 보고한다. Figure Panel B는 rollout당 answer에 도달할 확률이 p인 completer의 false-negative probability `(1 − p)^k`를 계산한다(p = 0.10, k = 8: 0.430; k = 32: 0.034). Pass@k도 추적한다. rStar-Math는 policy의 pass@N curve를 보고한다(Figs. 6–7).

**7. Generality에 미치는 효과.** LEMA gain은 training에 GSM8K를 썼으므로 author가 out-of-distribution으로 취급하는 SVAMP와 ASDiv로도 간다(§4.1). CSQA도 70B에서 +1.1 개선되지만 CSQA 자체 correction data를 썼으므로 transfer 결과가 아니다(§3.2). 이 장 source는 step-level negative가 instruction following, calibration, refusal에 주는 변화를 보고하지 않는다. 이는 Open question이다(RL의 negative gradient는 ch-43a 참고).

## 레시피

| Model(정확한 release) | Size | Stage | Setting | Value | Source location | Status | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| s1-32B (Qwen2.5-32B-Instruct) | 32B | distill-SFT | example; trace source | 1,000; Gemini 2.0 Flash Thinking Experimental | arXiv:2501.19393v3 §2.1–2.2 | verified 2026-09-15 | Table 2: s1K vs 1K-random/diverse/longest 및 59K-full, 95% bootstrap interval |
| s1-32B | 32B | distill-SFT | epoch; batch; step; peak LR; warmup; decay | 5; 16; 315; 1e-5; linear 5%(16 step); cosine to 0 | v3 App. D | verified 2026-09-15 | ablation 보고 없음 |
| s1-32B | 32B | distill-SFT | optimizer; weight decay; precision; loss masking | AdamW β1 0.9, β2 0.95; 1e-4; bfloat16; reasoning trace와 solution에만 loss | v3 App. D | verified 2026-09-15 | ablation 보고 없음 |
| s1-32B | 32B | distill-SFT | training sequence length | sample이 하나도 cut되지 않을 만큼 큼 | v3 App. D, D.1 | verified 2026-09-15 | Table 8: 4,096(74% cut) AIME24 30.0 대 32,768(0% cut) 50.0; setting당 run 수 미기재 |
| s1-32B | 32B | distill-SFT | compute | H100 16개, 26분(7 H100 GPU-hour) | v3 §4.1, §5.1 | verified 2026-09-15 | 59K-full: 394 H100 GPU-hour |
| s1-32B | 32B | eval-gate | decoding; forcing | temperature 0; thinking 종료 시도 시 “Wait” 추가 | v3 §3.1, §4.1 | verified 2026-09-15 | Table 4: “Wait” 2회 53.3 대 extension 없음 50.0 AIME24 |
| LIMO (Qwen2.5-32B-Instruct), arXiv v3 | 32B | distill-SFT | example; LR; schedule; warmup; epoch; batch | 800; 5.0e-6; cosine; 없음; 15; 64 | arXiv:2502.03387v3 §3.1.2, §4 | verified 2026-09-15 | §6.3.5: 400/800/1.2k/1.6k/2k subset; 800 → 1.2k AIME24 +0.9 |
| LIMO v3 | 32B | distill-SFT | sequence length | 모든 response가 맞도록 limit 설정; response는 16,384 token 미만 | v3 §4; v1 §4.1 | verified 2026-09-15 | ablation 보고 없음 |
| LIMO v1 vs v3 | 32B | distill-SFT | example; AIME24; MATH500 | v1: 817, 57.1, 94.8; v3: 800, 63.3, 95.6 | v1 Abstract; v3 Abstract, Table 1 | conflict(v3가 COLM 2025 version) | ablation 보고 없음 |
| OpenMath2-Llama3.1-8B | 8B | distill-SFT | data; epoch; batch; LR; optimizer | 13.97M pair; 2; 512(unit 미기재); constant 2e-5; AdamW, weight decay 1e-2 | arXiv:2410.01560v2 §4; [[openmathinstruct-2-recipe]] | verified 2026-09-14 | Fig. 1: 1M, 2M, 5M, full에서 gain |
| OpenMathInstruct-2 ablation student(Llama3.1-8B-Base) | 8B | distill-SFT | epoch; batch; LR; run | 4; 256; constant 5e-6; run 4회 average | v2 §2.2 | verified 2026-09-14 | ablation 보고 없음 |
| OpenMathInstruct-2 new question(Llama3.1-405B-Instruct) | 405B | distill-SFT | problem당 solution; temperature; answer rule | 32; 0.7; majority vote, minimum vote threshold 0 | v2 §3, App. C.1 | verified 2026-09-14 | Table 9: threshold 0/8/16/24 → 50.1/49.2/44.4/42.0(size unmatched) |
| OpenMathInstruct-2 | — | distill-SFT | length filter | Llama3.1 token 1,024 초과 또는 character 200 미만 solution drop | v2 App. A.2 | verified 2026-09-14 | ablation 보고 없음 |
| MetaMath-7B(LLaMA-2-7B) | 7B | distill-SFT | epoch; batch; LR; warmup | 3; 128; 2e-5; 3% | arXiv:2309.12284v4 App. B | verified 2026-09-14 | ablation 없음(Table 3은 setting 아닌 data operator 비교) |
| MAmmoTH-7B(Llama-2 7B) | 7B | SFT | LR; batch; schedule; epoch | 2e-5; 128; cosine, 3% warmup; 3 | arXiv:2309.05653v3 §2.3 | verified 2026-09-14 | ablation 없음(Table 6은 setting 아닌 rationale format 비교) |
| rStar-Math policy(Qwen2.5-Math-7B) | 7B | SFT | data rule; epoch; sequence length; batch; LR | average Q top-2 correct trajectory; 2; 4,096; 128; 7e-6, linear, AdamW; 매 round base에서 재학습 | arXiv:2501.04519v1 §3.4.1, App. A.1 | verified 2026-09-15 | data rule: Table 7, MATH 78.4 대 rejection sampling 73.4; base 재학습은 “based on our extensive experiments”(App. A.1), 수치 미보고; 나머지 setting ablation 없음 |
| rStar-Math PPM | 7B | reward-model | pair; loss; epoch; batch; LR | step당 positive 2 × negative 2; pairwise Bradley–Terry; 1; 512; 7e-6 | v1 §3.3, App. A.1 | verified 2026-09-15 | Table 8: MATH PPM 89.4 대 PQM(MSE) 88.2 대 ORM 82.6 |
| rStar-Math data, round 2–4 | 7B policy | SFT | rollout; depth; step당 candidate; c | problem당 16(hard는 +16; round 4는 최대 128); 16; 8(round 4: 16); 2 | v1 §3.4, App. A.1 | verified 2026-09-15 | Table 2: coverage 60.17% → 90.25% |
| Step-DPO SFT stage(Qwen2-7B) | 7B | SFT | data; epoch; batch; LR; schedule | 299K; 3; 256; 5e-6; linear decay, warmup ratio 0.03 | arXiv:2406.18629v1 §4.1 | verified 2026-09-15 | ablation 보고 없음 |
| Step-DPO(Qwen2-7B-SFT; >30B model) | 7B; >30B | preference | pair; epoch; batch; LR; β; schedule | 약 10K; 8(7B), 4(>30B); 128; 5e-7; 0.4(72B는 0.5); cosine, warmup 0.1 | v1 §4.1 | verified 2026-09-15 | β/LR/epoch ablation 없음; Table 3 loss 비교: 5K pair에서 55.8 대 DPO 55.0 |
| OmegaPRM data(Gemini Pro, Gemma2 27B policy) | — | reward-model | k; search limit; α, β, L; c_puct; question filter | 8; 100; 0.5, 0.9, 500; 0.125; rollout 32, correct 0 또는 wrong 0이면 drop | arXiv:2406.06592v2 §4, App. A | verified 2026-09-14 | Table 2: step accuracy soft label 70.1% 대 hard 63.3% |
| LEMA(LLaMA-2-70B) | 70B | SFT(correction content) | data; method; LR; batch; step; checkpoint rule | CoT 약 32K + GSM8K correction 12,523; QLoRA r 64, dropout 0.05; 1e-4; 96; 2,000; test set에서 매 100 step checkpoint 중 최선 | arXiv:2310.20689v4 §3.2–3.3 | verified 2026-09-15 | data: Fig. 4/Table 2가 matched size/token CoT-only와 비교; LR/batch/step ablation 없음 |
| UniReason-Qwen3-14B(SFT) | 14B | distill-SFT | data; LR; batch; epoch | math problem 47K, Qwen3-32B trace; 5e-5; 512; 1.5 | arXiv:2507.00432v2 App. A.3 | verified 2026-09-15 | ablation 없음; 1.5 epoch는 “to align with our RL settings”으로 선택(App. A.3.1) |
| UniReason-Qwen3-14B(RL) | 14B | RL | algorithm; LR; batch; prompt당 rollout; max length; clip; KL; step | GRPO(verl); 1e-6; 512; 16; 16k token; “between 0.22 and 0.28”; 0(entropy 0); 140 | v2 App. A.3.1 | verified 2026-09-15 | ablation 없음; KL on/off는 Qwen3-8B에서만 비교(Table 4) |
| LLaMA3-8B-Instruct RoPE θ×16 | 8B | long-context, 이후 distill-SFT | extension; SFT data; batch; LR; epoch | θ×16; OpenR1-Math-220K ≤8K 또는 8K–16K split 각 20K; 32; 1.0e-5; 3 | arXiv:2505.17315v2 §3.1–3.2 | verified 2026-09-14 | Table 3: MATH500 avg ×16 59.36 대 ×1 54.80 |

“verified 2026-09-15” status 행은 [[s1]], [[limo]], [[rstar-math]], [[step-dpo]], [[transferability-of-llm-reasoning]] library card가 verify되지 않았으므로 그 날짜에 primary arXiv PDF를 읽어 확인했다. Chapter excerpt에는 확인한 extract가 있다.

**소규모 범용 run의 시작점.** Instruction-tuned model을 trace 약 1K개로 long-CoT distillation SFT할 때 verified reference는 s1-32B다. Epoch 5, batch 16, peak LR 1e-5, linear warmup 5%, cosine decay to 0, AdamW(0.9, 0.95), weight decay 1e-4, trace와 answer에만 loss, sample을 자르지 않는 sequence length(그 ablation에서는 32,768)를 쓴다. 이 값은 H100 GPU 16개에서 Qwen2.5-32B-Instruct에 썼고 sequence length만 ablate했다. s1은 instruction-following evaluation을 보고하지 않는다(non-math check는 GPQA Diamond). MathIF 측정에서 s1-32B는 base의 80.96 대비 IFEval 58.04이므로, 같은 run에는 Generalization lens의 regression check가 필요하다.

## 일반화 관점

**(a) 무엇이 폭을 넓히는가.** 처음 세 항목은 math 내부 accuracy(여러 math benchmark 또는 held-out math set)를 측정하고, “math 밖” 표시는 다른 domain을 측정한다.

- Fixed size에서 더 많은 distinct problem: 1K에서 6.5K question으로 MATH validation +10.5([[openmathinstruct-2]] Fig. 6); knowledge-point coverage를 반으로 줄이면 상대적으로 −8.6%([[mathscale]] Table 6); diversity gain과 accuracy gain correlation 0.972([[metamath]] §4.5).
- 여러 source와 rationale format: MAmmoTH hybrid는 out-of-domain math set 5개를 포함한 9-set math average에서 CoT-only 32.0 대 47.9([[mammoth]] Table 6).
- Self-generated chosen step: in-distribution 55.8 대 GPT-4-corrected 55.1 MATH([[step-dpo]] Table 4; math만).
- Mixture의 science data(math 밖): science set 31K는 GPQA-Diamond 48.8, code set 두 개는 각각 47.3과 36.7이며, 각 code set과 science set을 섞은 62K는 52.7이다. In-domain data를 더하면 code-to-science transfer difference가 사라진다([[open-thoughts]] App. H.5, Table 24).
- On-policy data(math 밖): Qwen3-8B-Base에서 on-policy SFT non-reasoning transferability index는 +30.2, off-policy는 −40.5다([[transferability-of-llm-reasoning]] Table 4).
- Long-trace SFT 전 effective long context(math 밖): science SFT 뒤 GPQA 41.92 대 37.27, MMLU-STEM 74.27 대 71.06([[longer-context-deeper-thinking]] Tables 4, 6).
- Pretraining의 reasoning data(일부 math 밖): Pretraining에 reasoning data가 있는 8B model은 SFT 뒤에도 math, science, code, instruction-following evaluation average에서 35.92 대 26.62로 앞선다([[front-loading-reasoning]] Table 2).

**(b) 무엇이 narrowing 또는 forgetting을 만드는가.**

- Teacher trace의 math-only SFT: Qwen3-14B-Base에서 IFEval 69.2 → 42.3, non-reasoning average 45.7 → 21.1([[transferability-of-llm-reasoning]] Table 1).
- Long-CoT SFT와 reasoning RL: variant 16개 중 15개에서 constraint compliance가 낮아진다([[scaling-reasoning-losing-control-mathif]] Table 4). Rollout이 길수록 hard accuracy가 낮다(Table 5).
- Dataset-specific math SFT: WizardMath-70B AQuA 20.0, base 40.9([[mammoth]] Table 3).
- Trace data가 cover하지 않는 safety behavior: HarmBench 14.5 → 55.5([[open-thoughts]] Table 30).
- Tool coupling: OpenMathInstruct-1 model은 code execution을 loop 안에서 실행해 평가한다([[openmathinstruct]] §3.2). Executor 없이 deploy하는 경우는 평가하지 않았다.

**(c) 이 stage에서 측정하는 방법.**

- Perturbed problem: templated variant와 irrelevant clause([[gsm-symbolic]] Figs. 2, 8). Score 하나가 아니라 set 전체의 distribution을 보고한다.
- Post-cutoff problem: s1 Table 5는 budget forcing 없는 s1을 AIME 2024 50.0, AIME 2025 26.7로 나열하고, caption은 AIME 2025 값을 Ye et al.(2025b)에 귀속한다([[s1]] App. A). Synthetic question은 paraphrase check로 test set에 대해 decontaminate한다([[openmathinstruct-2]] §3.1).
- Math 밖 before/after suite: IFEval 또는 MathIF, non-math reasoning set(GPQA-Diamond, LiveCodeBench), non-reasoning set(CoQA, HaluEval)을 쓴다. [[transferability-of-llm-reasoning]]와 [[scaling-reasoning-losing-control-mathif]]가 regression을 찾은 곳이기 때문이다. Trace data에 safety content가 없으면 safety set도 더한다.
- Protocol control: 같은 model도 greedy와 4-sample evaluation 사이 AIME24가 10.2 point 다르다([[s1]] Table 1; [[limo]] Table 1). vLLM batch size도 greedy result를 바꾼다([[s1]] App. B). LEMA처럼 양 arm 모두에서 test set의 best checkpoint를 보고하면(§3.3) reported score가 부풀 수 있다(Interpretation).
- Search-time 대 policy score: search 또는 reward model을 쓰는 score 옆에 policy greedy score를 보고한다([[rstar-math]] Tables 5, 10).

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 확인 |
|---|---|---|
| SFT sequence length가 trace보다 짧음 | test-time trace가 길어지고 accuracy가 낮음(s1 4,096: thinking token 20,721 대 6,984, AIME24 30.0 대 50.0) | `L_p + L_t > L_max`인 example 비율; truncation 뒤에도 stop token으로 끝나는 training target 수([[s1]] Table 8) |
| Budget을 problem당 solution 수에 씀 | pair count가 늘지만 accuracy가 더 오르지 않음 | pair 100K당 unique-question 수; pair 고정 question ablation([[openmathinstruct-2]] Fig. 6) |
| Final-answer match를 step correctness로 간주 | accepted trace의 flawed intermediate step | step별 sample audit; code-verified 또는 PRM-scored step과 비교([[rstar-math]] §3.2) |
| Independently sampled reasoning pair의 whole-answer DPO | reward margin plateau; correct prefix penalty | chosen/rejected log probability; first-wrong-step pair와 비교([[step-dpo]] Fig. 2) |
| Teacher-written correction을 chosen step으로 사용 | chosen step의 reference log probability가 낮음 | teacher 대 self-sampled step의 `log π_ref(s_win)` histogram([[step-dpo]] Table 4) |
| Few-rollout weak completer의 Monte-Carlo label | verified-correct solution의 step에 `c_t = 0` label | verified solution step false-negative rate; question filter([[omegaprm]] App. A) |
| Trace SFT 뒤 math benchmark만 보고 | 뒤늦게 instruction-following/non-math regression 발견 | before/after IFEval, GPQA, non-reasoning set([[transferability-of-llm-reasoning]] Table 1) |
| Base-model control 없이 small curated set을 칭찬 | 다른 base에서 recipe가 실패 | 더 약한 base에서 같은 data 실행([[limo]] §6.3.3) |
| Search-time score를 model score로 인용 | deployed greedy model이 더 낮음 | trajectory count/reward model과 함께 greedy policy score 보고([[rstar-math]] Table 10) |
| Fixed public test set만 평가 | score가 templated-variant distribution보다 높음 | GSM-Symbolic style variant, post-cutoff set([[gsm-symbolic]] §4.1) |
| Context window 밖으로 thinking 확장 | 반복 forcing 후 repetition loop, window 초과 response | forced continuation과 window 초과 response 수 추적([[s1]] §4.2, §6.2) |

## 이해도 점검

1. Pair 256K를 고정했을 때, problem당 verified solution은 더 많지만 1K question×256 solution보다 6.5K question×39 solution이 더 나은 student를 학습할 수 있는 이유는 무엇인가?
2. OpenMathInstruct-2는 wrong solution 20%에서 loss가 작다고 하지만 Step-DPO는 wrong step 하나를 penalize해서 이득을 얻는다. Negative marginal value와 gradient로서의 negative 차이를 사용해 두 결과가 함께 성립하는 이유를 설명하라.
3. `∂ log p_y / ∂ z_j = 1[j = y] − p_j`를 써서, first token이 이미 unlikely한 rejected step을 낮추면 왜 가장 likely한 alternative가 주로 올라가는지, 그리고 chosen/rejected step이 비슷할 때 무엇을 뜻하는지 설명하라.
4. s1 author의 mechanism에 따르면 12,000-token trace를 4,096 token에서 truncate하면 학습 model이 test time에 더 오래 생각하게 되는 이유는 무엇인가?
5. Qwen3-14B 연구에서 SFT와 RL은 같은 47K problem을 썼는데 SFT는 IFEval을 낮췄다. 두 run의 어떤 차이가 원인이 될 수 있고, Qwen3-8B ablation은 어느 차이를 isolate하는가?
6. rStar-Math의 MATH 90.0과 policy MATH 78.4는 같은 training에서 나왔다. 어떤 component가 차이를 만들며 SFT-only model과는 어느 수치를 비교해야 하는가?
7. Monte-Carlo hard label은 completer가 약할 때 왜 false negative가 늘고 rollout 수가 커질 때 왜 false positive가 늘어나는가?

## 연결

- 이전: ch-23 — Model Collapse and Verification of Synthetic Data. 그 장의 verification/self-training risk는 §5의 self-generated trace와 §7의 on-policy data에 적용된다.
- 다음: ch-25 — Multi-Turn Conversation Synthesis.
- ch-20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage(math 밖 teacher-trace distillation).
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection(§3의 diversity measure).
- ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation; ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood.
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression; ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation(§8).
- ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity(§7).
- ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages; ch-44 — Process Supervision and Verifiable Rewards; ch-44a — Length in RL: Overlong Responses, Length Control, and Long-Context RL.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation; ch-48 — Contamination Detection and Its Effect on Reported Scores.

## 출처

- [[openmathinstruct]] — sampling count, Mask-Text coverage, GSM-Hard robustness note, released wrong solution.
- [[openmathinstruct-2]] — format, teacher, filtering, noise, question-diversity ablation; decontamination.
- [[openmathinstruct-2-recipe]] — OpenMath2-Llama3.1-8B의 SFT setting.
- [[metamath]] — operator definition, data-scaling gain, SFT target으로서 wrong-answer path, DROP transfer.
- [[mathscale]] — concept-graph walk equation, knowledge-point ablation, unvalidated solution.
- [[mammoth]] — hybrid CoT/PoT ablation과 dataset-specific SFT regression.
- [[mammoth-2]] — web Q-A extraction/refinement model(correction만).
- [[rstar]] — five action, Phi3-mini discriminator, GSM8K/StrategyQA result.
- [[rstar-math]] — code-augmented search, Q annotation, PPM loss, round, policy 대 search score.
- [[s1]] — selection pipeline, budget forcing, data ablation, sequence-length ablation, training setting.
- [[limo]] — v1/v3 차이, selection, base-model/size dependence, OOD table, training setting.
- [[step-dpo]] — pair construction, loss, DPO 비교, in/out-of-distribution chosen step.
- [[lema-learning-from-mistakes]] — content로서 correction data, gain, matched-size/matched-token control.
- [[omegaprm]] — Monte-Carlo label, binary search, soft label, question filter.
- [[math-shepherd]] — hard label 및 rollout에 따른 label accuracy.
- [[transferability-of-llm-reasoning]] — 같은 math data에서 SFT 대 RL; on-policy ablation.
- [[scaling-reasoning-losing-control-mathif]] — long-CoT SFT/RL의 instruction-following loss; length effect.
- [[gsm-symbolic]] — templated/irrelevant-clause perturbation result.
- [[longer-context-deeper-thinking]] — reasoning SFT 이전의 long-context ability.
- [[open-thoughts]] — reflection-phrase ablation, cross-domain GPQA effect, safety regression.
- [[front-loading-reasoning]] — pretraining 대 SFT의 reasoning data.
- [[likelihood-displacement]] — DPO-style loss에서 similarity-driven displacement.
- [[numina-math]] — rStar-Math problem source(correction에서 명명).
- [[quiet-star]] — 제거한 section(correction에서 명명).
