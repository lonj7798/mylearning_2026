<!-- chapter: ch-35
     track: sft
     kind: content
     title: Case Studies C — Nemotron, Distillation SFT
     deps: [ch-34]
     sources: [[nemotron-4-synthetic]], [[nemotron]], [[nemotron-ultra]], [[deepseek-r1]], [[deepseek-r1-followup]], [[deepseek-r1-distill-synth]], [[bespoke-stratos]], [[sky-t1]], [[openr1]], [[s1]], [[limo]], [[open-thoughts]]
     figures: figures/distill-sft-compare.html
-->

# 35장 — Case Studies C: Nemotron, Distillation SFT

> **핵심 통찰.** 이 장의 case들은 하나의 continuum의 양 끝에 있다. Nemotron-4 340B는 세계에서 가장 무거운 *synthetic-SFT apparatus*를 만든다. Custom multi-attribute reward model(HelpSteer2)이 alignment data의 >98%를 생성하는 pipeline을 구동하고, 전체 stack은 단 ~20K human example로 anchor된다. 반대편에서 Bespoke-Stratos / Sky-T1 / s1 / LIMO는 세계에서 가장 *가벼운* post-training을 만든다. Frontier reasoning model의 trace distribution을 복사하고, filter하고, SFT하고, ship한다. 추출해야 할 교훈은 양 극단이 모두 작동하지만, 각자가 만들어진 목적에 대해서만 작동한다는 것이다. 다른 이들을 위한 trace generator가 되고 싶다면 Nemotron의 machinery가 유일한 viable path이다. <$1K로 reasoner를 원하고 teacher가 공개 재배포 가능한 경우라면 lean distillation recipe가 유일한 viable path이다.
>
> **가이드라인.** 이기고 싶은 benchmark가 아니라 §6의 decision tree에서 recipe를 고르라. 자신의 base model, 수천만 달러 단위 compute budget, 그리고 다른 이들을 위한 open data 생산 의도가 있다면 HelpSteer2-style multi-attribute RM과 그 주변 synthetic pipeline을 만들어라. 강한 open base, 주말, permissive teacher(R1 또는 QwQ, GPT-4 아님)가 있다면 17K-trace distillation recipe를 돌려라. Base가 이미 reasoning-rich하고 탁월한 curator가 있다면 s1/LIMO-style 1K hand-selection이 둘 다 이긴다. "SFT is enough"와 "still need RL"의 경계는 `OpenR1-Qwen-7B-SFT`와 그 GRPO follow-up 사이의 +3-5 AIME gap으로 입증된다. Saturated distilled trace에서도 SFT에는 ceiling이 있고 RL이 그 너머로 밀어준다.

---

## 이 장이 존재하는 이유

Ch-33과 ch-34는 mainstream heavy recipe를 다뤘다. Tülu 3의 939K-sample mix, Llama 3의 6-round RSFT, Qwen 2.5의 1M-SFT + 150K-DPO, Qwen 3 hybrid-thinking, Phi-4-reasoning이 그것이다. Ch-20은 **distillation-as-data** primitive를 확립하고 세 가지 open R1-distill reproduction을 분류했다. 이 장은 ch-20이 하지 않은 두 가지를 한다.

1. **Nemotron-4 340B**를 frontier synthetic data의 *producer*로 전면에 둔다. 이 model의 alignment story는 "더 나은 RM을 만들면 SFT와 preference data의 98%를 synthetic으로 만들 수 있다"이다. Ch-20은 taxonomy-driven synthesis를 지나가며 언급했다. 여기서는 Nemotron을 따로 다룬다. HelpSteer2 5-attribute RM이 production reward-model recipe의 가장 깔끔한 public disclosure이며, ch-41(reward modeling)이 다시 돌아올 reference point이기 때문이다.
2. **Sufficiency question**을 묻는다. Distillation-SFT는 어느 지점에서 RL만 넘을 수 있는 ceiling에 부딪히는가? s1/LIMO/Bespoke-Stratos/Sky-T1/Open-R1 비교는 이를 철학적 질문이 아니라 empirical question으로 만들고, §5의 5-way table이 이 장의 분석 단위이다.

---

## 1. Nemotron-4 340B — synthetic-alignment apparatus

[[nemotron-4-synthetic]]와 [[nemotron]]은 base가 아니라 그 주변을 감싼 **apparatus**가 distinctive contribution인 340B dense model을 함께 설명한다. Reward model, generator, critic이 함께 모든 alignment data의 98% 이상을 합성한다. 명시된 human budget은 총 ~20K annotation이며, SFT seed와 HelpSteer2 preference label로 나뉜다. 나머지는 모두 machine-made이다.

### 1.1 HelpSteer2 5-dimensional rubric (verbatim)

이것이 이 장의 load-bearing artifact이다. [[nemotron]]은 single preference logit 대신 attribute별 하나의 score를 내는 **5-vector**를 출력하는 regression reward model을 훈련한다. HelpSteer2 rubric의 다섯 attribute는 다음과 같다.

| # | Attribute | What it measures |
|---|---|---|
| 1 | **Helpfulness** | Does the response address what the user asked for? |
| 2 | **Correctness** | Are the factual, logical, or code claims correct? |
| 3 | **Coherence** | Is the response internally consistent and well-structured? |
| 4 | **Complexity** | Does the response match the intellectual depth the prompt demands? |
| 5 | **Verbosity** | Is the response length calibrated to the task? |

RM architecture는 shared trunk(340B base)와 HelpSteer2의 10,000 human-labeled example에 대해 per-attribute L2 regression으로 훈련한 **five linear heads**이다. RL time의 preference use를 위해 다섯 score는 **weighted sum**으로 결합된다(weight는 NeMo-Aligner config에 문서화). Nemotron-4-340B-Reward는 release 당시 RewardBench 1위를 기록했다.

왜 1 score 대신 5 score가 operationally 중요한가:

- **RL time의 compositional preference.** RM을 retrain하지 않고도 verbosity weight를 낮출 수 있다. Policy가 다른 objective를 공짜로 얻는다. Scalar-only RM은 새 annotation round를 강제한다.
- **Single-attribute Goodhart가 보인다.** Policy가 verbosity를 over-optimize하기 시작해도 나머지 네 attribute가 signal을 준다. Scalar RM은 네 속성을 hacked one 속으로 collapse시킨다.
- **Scale에서의 synthetic filtering.** Per-attribute score는 예컨대 math에는 high-correctness-low-verbosity trace를 유지하고 science에는 high-complexity trace를 유지하게 해준다. Retraining 없이 가능하다.

이는 ch-41이 제기할 질문, *"왜 Nemotron은 10K human preference로 Llama 3가 훨씬 더 많은 preference에서 얻는 것보다 더 많은 것을 얻는가?"*에 대한 답이다. Answer: richer label schema가 human cost를 다섯 training signal로 amortize한다.

### 1.2 >98%-synthetic alignment pipeline

[[nemotron-4-synthetic]]는 pipeline을 itemize하고, [[nemotron]]은 approximate slice size를 이름 붙인다. 여섯 stage loop를 말로 쓰면 다음과 같다.

```
seed (task-family prompts)
   -> prompt generation (Nemotron-4-Instruct_{t-1}이 family별 task prompt를 합성)
   -> response generation (Instruct_{t-1}이 1-N candidate responses / dialogues를 생성)
   -> RM filtering (340B-Reward가 각 response를 5 attributes로 scoring)
   -> selection (high-score 유지; DPO pair에서는 high vs low)
   -> stage-specific training (SFT -> DPO -> RPO, iteratively)
```

Covered task families는 coding, general QA, topic-following, document-based reasoning, function-calling, 그리고 *incapable tasks*(refuse해야 하는 prompt, human-written rejection으로 few-shot seeded)이다. Topic-following의 경우 pipeline은 *intentionally injects distractor turns*하여 student가 다시 주제로 steer back하는 법을 배우게 한다. Approximate output volumes:

- **~800K code SFT**(Genetic Instruct로 생성: Self-Instruct + WizardCoder-style mutations + LLM fitness function이 작은 seed를 1000x-scale population으로 키움).
- **~200K general SFT**(category-seeded; RM-filtered).
- **~160K DPO preference pairs.**
- **~300K RPO preference pairs**(reward-preference optimization — policy가 reference에서 "flying off"하지 못하게 chosen response에 대한 SFT loss term을 추가한 DPO).

SFT는 staged이다. **Code SFT first**, 그다음 **general SFT**. Paper의 justification은 code SFT가 format discipline을 날카롭게 만든 뒤 general-domain SFT가 더 느슨한 objective를 도입한다는 것이다. Preference optimization은 **DPO followed by RPO**로 진행된다. Nemotron은 DPO alone이 chosen과 rejected 사이의 reward gap에 overfit한다고 주장하고, RPO의 auxiliary SFT term on chosen response가 그 drift를 상쇄한다고 말한다.

### 1.3 작은 human anchor가 충분한 이유

Counter-intuitive claim은 ~10K HelpSteer2 label과 ~10K SFT seed만으로 downstream token 수천만 개를 생성하는 alignment loop를 지속할 수 있다는 것이다. Mechanism은 다음과 같다.

1. 10K human label은 policy가 아니라 RM을 직접 훈련한다.
2. RM은 임의 크기의 synthetic pool을 scoring한다. 무엇이 filter될 수 있는지는 human set이 아니라 RM의 *coverage*가 bound한다.
3. Filtered synthetic pool이 policy를 훈련한다. 각 RM query는 machine operation이며, human cost는 amortize된다.

Nemotron이 표시하는 risk: **같은 scorer를 iteration 전반에 재사용하면 reward-model error가 compound된다.** RM이 correct-but-unusual reasoning step을 systematic하게 낮게 scoring하면, iteration 2의 policy는 그 step을 더 이상 emit하지 않고, iteration 3의 RM은 이제 더 좁은 distribution에 tuned되며, collapse가 compound된다. Mitigation은 partial하다. Nemotron은 주기적으로 HelpSteer2 pool에 fresh human preference를 추가하고 RM을 retrain하지만, paper는 이것이 compounding risk를 완전히 제거한다고 주장하지 않는다. Ch-23(model collapse)이 이 failure mode의 직접적인 continuation이다.

---

## 2. Nemotron-Ultra / Nemotron 3 — multi-environment RL succession

[[nemotron-ultra]]는 2025년 successor를 설명한다. Nemotron 3는 Nano(3.2B active / 31.6B total MoE)로 ship되며, Super와 Ultra tech report는 이어질 예정이다. 이 장에서 중요한 Nemotron-4 대비 두 delta:

- **Multi-environment RL**이 sequential stage를 대체한다. Nemotron-4는 reasoning-RL, tool-use-RL, alignment-RL을 순차 실행했다. Nemotron 3는 reasoning, multi-step tool use, agentic environment를 가로지르는 single RL run으로 이를 collapse하며, reward model(이제 **GenRM** — generative reward model)이 이 전부를 scoring한다. Claim은 staged recipe보다 agentic task로의 generalization이 더 낫다는 것이다.
- **GenRM이 policy와 함께 public release된다.** Nemotron-4의 340B-Reward는 open-weight였지만, 이를 training하는 recipe는 paper만으로 완전히 reproducible하지 않았다. Nemotron 3의 GenRM release는 downstream user가 RM을 다시 훈련하지 않고 RLHF를 resume할 수 있게 한다.

[[nemotron-ultra]]가 *공개하지 않는 것*은 공개하는 것만큼 말해준다. RL algorithm(PPO vs GRPO vs DPO unspecified), KL β, LR, batch size, clip ε, group size G, rollouts per prompt, step counts, GenRM loss form, preference-data sizes, multi-environment reward-mixing weights가 모두 빠져 있다. White paper는 hyperparameter가 얇다. "Open release"가 spectrum이며 Nemotron 3가 "reproducible recipe"보다는 "reproducible artifact bundle"에 가까운 쪽에 있다는 reminder이다.

Ch-35에서 중요한 Nemotron-4 -> Nemotron 3 shift는 synthetic-data pipeline이 *carried forward*되지만 더 이상 headline이 아니라는 점이다. Headline은 RL environment coverage이다. Synthetic SFT는 이제 *substrate*이지 finish line이 아니다.

---

## 3. R1-distill as SFT-consumption — ch-20에서 달라진 것

Ch-20은 teacher-side R1 pipeline을 자세히 다뤘다. 이 장은 **student-side** angle에서 R1-distill을 다시 본다. DeepSeek가 800K trace pool을 emit하면, 이를 SFT로 consume하는 것은 어떤 모습이며 왜 RL 없이 작동하는가?

[[deepseek-r1]] / [[deepseek-r1-distill-synth]] / [[deepseek-r1-followup]]에서:

- Distill corpus는 teacher 자체 pipeline의 **rejection-sampling SFT stage**가 생성한다. Stage-1 RL model이 prompt당 N trace를 sample하고, V3-judge가 readability + correctness로 filter하며, kept set은 ~600K reasoning + 200K non-reasoning이다.
- 여섯 distilled student가 release된다: Qwen-2.5-Math 1.5B, Qwen-2.5 7B/14B/32B, Llama-3.1-8B, Llama-3.3-70B. **모두 800K에 대한 pure SFT**이다. RL 없음, RM 없음, DPO 없음.
- Report의 explicit claim: *dense student는 copied reasoning structure에서 자신의 RL로 그 structure를 rediscover하는 것보다 더 큰 이득을 얻는다.* Dense 32B student가 scratch에서 GRPO를 돌리면 R1 trace에 대한 one-epoch SFT보다 compute가 더 들고 model은 더 약하다.

R1-0528 refresh([[deepseek-r1-followup]])는 *R1-with-more-compute*이다. 같은 V3 base, 같은 recipe, 더 많은 RL step이다. R1-Distill family는 unchanged이다. V3.1(Aug 2025)은 그다음 R1의 reasoning을 V3 line에 *absorbs*하여 R1을 standalone family로 끝낸다. Ch-35의 practical implication은 다음과 같다. R1-distill은 아마도 "straight SFT transfer of reasoning"의 **terminal** version이다. Future recipe는 separate distilled reasoner가 아니라 hybrid-thinking-mode(ch-34 Qwen 3)가 될 것이다.

---

## 4. Bespoke-Stratos, Sky-T1 — operational claim으로서의 "cheap frontier reasoning"

Ch-20은 이것들을 catalogued했다. 여기서는 cost claim과 각 team이 정확히 무엇을 filtering했는지 추출한다. Cost number가 distillation SFT가 극도로 cheap할 수 있다는 이 장의 concrete evidence이기 때문이다.

### 4.1 Bespoke-Stratos — contamination checks and the $4.8K run

[[bespoke-stratos]]: math(~7K problems from NuminaMath-CoT, MATH, AIME/AMC archive), code(~5K from APPS, CodeContests, TACO, LeetCode), science(~5K from STILL-2 curated prompts + CoTLogic)를 포괄하는 17,000 `(prompt, R1-trace)` pair.

**Trace generation.** DeepSeek-R1(official API)에 temperature 0.6으로 query하고 `<think>...</think><answer>...</answer>` format을 요청하며, failure 시 최대 3× retry한다.

**Three-layer verifier (rejection-sampling filter):**

1. **Math.** Boxed answer를 extract하고 SymPy canonicalization으로 gold와 비교한다. Mismatch는 reject.
2. **Code.** Candidate solution을 extract하고 public unit test를 실행한다. Any failure는 reject.
3. **Science.** GPT-4o as LLM-judge. Reference에 대해 "correct" verdict가 필요하다.

Reject rate는 raw R1 output의 ~30-50%이다. 대부분의 rejection은 code test failure와 math extraction error이다. MinHash dedup cross-prompt를 수행하고 per-source cap으로 domain balance를 강제한다. **Contamination check** — AIME와 MATH는 public이고 R1이 solution을 memorize했을 수 있으므로 Bespoke는 AIME25를 clean eval로 hold out한다. Stratos-32B paper는 AIME24 ~63%를 보고하지만 AIME25 number가 더 약하며 "post-contamination-gap" reality를 나타낸다고 flag한다.

**Cost.** ~$800 DeepSeek-R1 API credits(teacher) + ~$4,000 student training(8×H100, Qwen2.5-32B-Instruct에서 몇 시간). Ablation: code-verification을 제거하면 LiveCodeBench gain이 절반이 되고, math symbolic equivalence를 제거하면 MATH gain이 절반이 된다. **모든 verifier layer가 load-bearing이다.**

### 4.2 Sky-T1 — $450 QwQ recipe and the reformatting trick

[[sky-t1]]: 17K traces, 대부분 **QwQ-32B-preview**(Alibaba의 open-weights reasoner, API lock-in 없음)에서 distill. Qwen2.5-32B-Instruct base, 3 epochs × 19 hours on 8×H100 ≈ listed rental rate 기준 $450.

**Pulled from QwQ.** Local vLLM inference(teacher cost ≈ 0), temperature 0.7, max 8K tokens per trace, ~10K math seeds(NuminaMath-CoT + AIME/AMC) + ~5K code(APPS + TACO) + ~2K science(STILL-2).

**Filtered out.**

- **Math mismatch.** `\boxed{}`에 대해 SymPy를 적용하고 non-matching은 reject.
- **Code failure.** Unit-test execution. Any-test-fail은 reject.
- **Science incorrect.** GPT-4o-mini LLM-judge. "incorrect" verdict는 reject.
- **Format noise.** QwQ는 "Alright, let me think", "Hmm, okay so" 같은 filler preamble을 emit한다. Sky-T1은 QwQ의 native format을 `<|im_start|>…<|im_end|>` chat template으로 바꾸고 filler를 제거하는 GPT-4o rewriter pass를 실행한다. Paper는 이 cleanup *alone*이 AIME를 +4 points 올렸다고 보고한다. Rewriter는 cosmetic이 아니다.

**Training config attested.** LR 1e-5, 3 epochs, sequence length 32K(long trace를 fit하기 위해), BF16, FSDP across 8 GPUs, Llama-Factory framework.

**Results.** MATH500 82.4%(o1-preview와 match), AIME24 43.3%(o1-preview에서 2 pts 이내), LiveCodeBench-Easy 86.3%(o1-preview를 beat), GPQA-Diamond 56.8%. **AIME25는 크게 떨어진다**. Teacher로서 QwQ는 R1보다 ceiling이 낮고, Sky-T1은 이를 상속한다. 이는 *teacher quality가 SFT-only distillation의 ultimate ceiling*이라는 가장 명확한 public evidence이다.

---

## 5. Comparison table — 5 distillation recipes

이것이 이 장의 headline table이다. Interactive version은 [figures/distill-sft-compare.html](figures/distill-sft-compare.html)에 있다. Recipe를 클릭하면 full filter breakdown과 attested hyperparameter가 펼쳐진다.

| Recipe | # traces | Teacher | Filter stack | Cost (full) | AIME24 | MATH500 | GPQA-Dia |
|---|---:|---|---|---:|---:|---:|---:|
| **R1-Distill-Qwen-32B** (official) | 800,000 | DeepSeek-R1 (671B MoE) | V3-judge readability + correctness | not disclosed | ~72% | ~94% | ~62% |
| **Bespoke-Stratos-32B** | 17,000 | DeepSeek-R1 | SymPy + unit tests + GPT-4o-judge + MinHash | ~$4.8K ($800 API + $4K compute) | ~63% | ~93% | ~59% |
| **Sky-T1-32B-Preview** | 17,000 | QwQ-32B-preview (open) | SymPy + unit tests + GPT-4o-mini judge + GPT-4o rewriter | ~$450 (local QwQ + 8×H100) | ~43% | ~82% | ~57% |
| **OpenR1-Qwen-7B** | ~440,000 (220K×2) | DeepSeek-R1 | Math-Verify SymPy only (math-only corpus) | ~$10K + multi-day H100 | ~40% | ~80% | n/a |
| **s1-32B** | 1,000 | Gemini (CoT traces) + hand-curation | difficulty + diversity + quality (hand-filter from 59K pool) | ~26 min × 16 H100 ≈ $50 | 56.7% | 93.0% | 59.6% |
| **LIMO** (817 traces) | 817 | mix + hand-edited | manual correctness + reflective-structure filter | hand-curation labor | 63.3% | 95.6% | (strong OOD) |

이 table의 non-obvious reading 세 가지:

**5.1 Trace count와 benchmark는 monotone이 아니다.** s1의 1,000개와 LIMO의 817개는 Sky-T1의 17,000개보다 AIME24에서 높다. Sky-T1은 ~17× 더 많은 data를 가졌음에도 그렇다. [[s1]]와 [[limo]]에 따르면 mechanism은 hand curation이 student를 augment하기보다 *corrupt*하는 low-signal trace를 제거한다는 것이다. Mass distillation은 SymPy/unit-test filtering 이후에도 이를 필연적으로 일부 smuggle한다. s1은 inference에서 **budget forcing**(early stopping을 억제하기 위해 `"Wait"`를 append)까지 더한다. 이는 같은 checkpoint에서 AIME24를 50%에서 57%로 끌어올리며 추가 training은 없다. Budget forcing은 training trick이 아니라 latent compute reallocation이다.

**5.2 17K는 regime이지 target이 아니다.** Bespoke-Stratos, Sky-T1, 그리고 우연히 여러 OpenThoughts intermediate checkpoint가 모두 약 17K에 도달했다. Pattern은 신비하지 않다. Public seed set에서 math + code + science 위에 three-layer verifier를 적용하고 ~30-50% reject rate가 있을 때 살아남는 가장 작은 pool이 대략 17K이다. 더 hard하게 filter하면 10K 아래로 떨어져 domain coverage를 잃는다. 더 soft하게 filter하면 bad trace를 유지한다. 17K라는 숫자는 hyperparameter가 아니라 equilibrium이다.

**5.3 OpenR1의 GRPO delta가 "still need RL?" evidence이다.** OpenR1-Qwen-7B는 440K trace에 대한 pure SFT로 MATH ~80% / AIME24 ~40%를 얻는다. 40K-subset 위에서 binary Math-Verify reward로 GRPO stage를 추가하면 +3-5 AIME points가 붙는다. 이는 "SFT ceiling vs RL residual"을 isolate하는 가장 깔끔한 public ablation이다. Gap은 크지 않지만(AIME에서 5 points), 실제이며 SFT budget이 이미 generous한 뒤에도 남는다.

---

## 6. Distillation SFT가 충분한 때 vs 여전히 RL이 필요한 때

이 장이 세워진 decision tree이다. 세 axis는 base-model reasoning capacity, teacher-trace license, target evaluation이다.

```
Base model이 이미 reasoning-rich인가(Qwen2.5-32B / Llama-3.1-70B)?
|-- YES: distillation SFT로 충분할 가능성이 높다
|     |
|     Hand-curate할 수 있는가?
|     |-- YES + strong curators  -> s1 / LIMO regime (1K traces, $50, hand review)
|     \-- NO  -> Bespoke-Stratos / Sky-T1 regime (17K traces, ~$500-$5K)
|
\-- NO (base가 더 약하거나 / 더 작거나 / general-purpose):
      |
      Target eval이 잘 verifiable한가(math, code)?
      |-- YES -> SFT + GRPO/RLVR
      |         (OpenR1 pattern: 440K SFT + 40K-prompt GRPO; s1 budget forcing은
      |          더 이상 cheap recovery가 아니며 ceiling을 깨려면 RL이 필요)
      \-- NO (agentic, tool-use, open-ended)
                -> full Nemotron-style synthetic-pipeline + multi-attribute RM + RL
                   (open-ended task에서는 reasoning-only SFT ceiling이 더 낮다)
```

이 tree가 encode하는 asymmetry가 이 장의 take-home이다. **SFT는 잘할 것으로 기대된 verifiable domain에서 더 빨리 saturate된다.** Math와 code는 outcome reward와 unit test가 작동하는 곳이며, ch-20 §5.5가 지적한 "wrong-question-correctly" failure를 verifier가 잡을 만큼 강한 곳이다. Open-ended task에서는 verifier가 LLM-judge이고, 그것 자체가 distribution-shift-brittle하다. 그곳에서는 SFT의 ceiling이 더 낮지만 RL의 ceiling도 더 낮다. Nemotron-style multi-attribute RM은 모든 regime에서 reliable하게 작동하는 유일한 lever이다. 그래서 Nemotron-Ultra multi-environment RL이 2025년 방향이지 distillation이 아니다.

Full recipe switch는 [figures/distill-sft-compare.html](figures/distill-sft-compare.html)을 보라. Scenario를 클릭하면 tree가 current selection과 matching되는 recipe row를 highlight한다.

---

## 7. 이 장이 열어두는 것

- **Per-attribute RM weights.** [[nemotron]]은 "weighted sum of 5 attributes (weights in NeMo-Aligner config)"라고 말한다. Weight 자체는 paper에 없다. Practitioner는 NeMo-Aligner source를 읽어야 한다.
- **R1-Distill의 per-domain slice ratios.** DeepSeek는 600K + 200K split을 주지만 600K reasoning set의 per-domain breakdown은 주지 않는다.
- **Budget-forcing transferability.** [[s1]]은 `"Wait"` appending이 AIME를 boost하지만 모든 prompt class에서 그렇지는 않다고 보여준다. Paper는 어떤 prompt distribution이 반응하는지 characterize하지 않는다.
- **OpenR1 GRPO on non-math.** GRPO stage는 math-only이다(Math-Verify as reward). Code / science로의 generalization은 ch-44(verifiable rewards)가 부분적으로 다루는 open question이다.

---

## Connections and what's next

- **ch-20** — Distillation-as-data origin chapter; Orca lineage + R1-distill mechanics. 이 장은 ch-20을 전제로 하고 그 vocabulary를 사용한다.
- **ch-33 / ch-34** — Tülu 3, Llama 3, Qwen 2.5/3, OLMo 2/3, Phi 3/4 — mainstream SFT case studies. Nemotron은 **synthetic-only** peer이며, 5-dim RM이 이를 구분한다.
- **ch-36 (lab)** — masking test가 있는 Packed SFT run. 이 case-study block 뒤의 practical checkpoint이다.
- **ch-23 (model collapse)** — Nemotron의 self-bootstrapping RM loop는 archetypal test case이다. §1.3이 flag한 compounding-error risk가 그곳에서 precise해진다.
- **ch-41 (reward modeling)** — HelpSteer2 5-attribute regression은 reference recipe이다. Scalar-only baseline이 comparison이다.
- **ch-40 (GRPO)** — R1의 GRPO hyperparameter와 DeepSeek의 loose-clip(ε=10) philosophy가 case study이다.
- **ch-44 (verifiable rewards)** — OpenR1의 Math-Verify는 가장 작은 working verifier이다. Bespoke-Stratos의 three-layer stack은 canonical extended version이다.

## Further reading

- [[nemotron-4-synthetic]] — NVIDIA 2024; >98%-synthetic alignment; staged SFT + DPO + RPO; Genetic Instruct for code.
- [[nemotron]] — Nemotron-4 340B model report; HelpSteer2 5-attribute RM; 10K human preferences.
- [[nemotron-ultra]] — Nemotron 3 Nano white paper; multi-environment RL; GenRM release; reasoning budget control.
- [[deepseek-r1]] / [[deepseek-r1-followup]] / [[deepseek-r1-distill-synth]] — R1 pipeline, R1-0528 refresh, 800K distill corpus.
- [[bespoke-stratos]] — 17K curated, $800 API + $4K compute, three-layer verifier, AIME24 ~63%.
- [[sky-t1]] — $450 QwQ recipe, GPT-4o rewriter +4 AIME, teacher-ceiling lesson.
- [[openr1]] — 220K×2 math corpus, Math-Verify, GRPO-adds-+3-5-AIME evidence.
- [[s1]] — 1K curated + budget-forcing at inference; 26-minute SFT.
- [[limo]] — 817 traces; Less-is-More Reasoning Hypothesis; latent-capability activation.
- [[open-thoughts]] — 1000+ ablations; QwQ > R1 as teacher; no-answer-side-filter finding.

## Companion visualization

**[figures/distill-sft-compare.html](figures/distill-sft-compare.html)** — self-contained interactive comparator. 다섯 recipe card(R1-Distill-Qwen-32B / Bespoke-Stratos / Sky-T1 / OpenR1 / s1)가 side by side로 배치된다. Card를 클릭하면 filter stack(어떤 verifier layer, reject rate, dedup policy, rewriter step)과 attested evaluation number가 펼쳐진다. 오른쪽 decision-tree panel은 current selection과 matching되는 row를 light up하여 answer-profile이 어떤 recipe를 가리키는지 보여준다. Distillation run 전에는 recipe 선택에, run 후에는 reference grid 대비 number check에 사용하라.
