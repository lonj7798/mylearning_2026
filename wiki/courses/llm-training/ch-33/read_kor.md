<!-- chapter: ch-33
     track: sft
     kind: content
     title: Case Studies A — Tülu 3 and Llama 3
     deps: [ch-32]
     sources: [[tulu-3]], [[tulu-3.1]], [[tulu-3-1]], [[tulu-3-sft-mix]],
              [[allenai-tulu-sft-recipe]], [[allenai-tulu-blog]], [[allenai-tulu-synth]],
              [[llama-3]], [[llama-3-synthetic-pipeline]], [[rlvr-tulu3]]
     figures: figures/tulu-llama-recipe.html
-->

# 33장 — Case Studies A: Tülu 3와 Llama 3

> **핵심 통찰.** Tülu 3와 Llama 3는 modern post-training에 대해 완전히 공개된 두 reference point이며, 같은 질문에 반대 topology로 답한다. Tülu 3는 **three-stage linear pipeline**이다. 한 번 curated된 939K-prompt mixture로 SFT하고, 그다음 DPO, 그다음 RLVR을 수행한다. 그 기여는 *prompt mixture가 row-by-row로 공개되어 있다*는 점이다. Llama 3는 **six-round closed loop**이다. SFT → rejection sampling → DPO를 반복하고, round *N-1*의 best checkpoint가 round *N*의 SFT pool을 생성한다. Tülu pipeline의 교훈은 training 시작 시점의 *data composition*이 first-order라는 것이다. Llama pipeline의 교훈은 *data composition이 policy와 함께 evolve*하며 stale preference가 loop를 degrade한다는 것이다. 둘을 나란히 읽는 것이 2024-2025년에 "modern SFT"가 실제로 무엇을 뜻하는지 보는 가장 깔끔한 방법이다.
>
> **가이드라인.** 강한 open base에서 시작하고 eval suite가 잘 정의되어 있다면 Tülu 3를 복사하라. Mixture를 공개하고, 명시적으로 decontaminate하고, SFT는 2 epoch로 한 번 치고, 이득을 DPO와 RLVR stage로 옮겨라. 처음부터 training하고 라운드마다 human annotation을 감당할 수 있다면 Llama 3를 복사하라. Reward model을 fresh하게 유지하고, 최신 checkpoint에서 SFT pool을 매 라운드 재생성하며, stale preference를 절대 재사용하지 마라. 둘을 접붙이려 하지 마라. Mid-pipeline에서 preference를 다시 캐기 시작하면 Tülu의 once-and-done discipline은 깨지고, 첫 라운드 eval이 돌아오기 전에 mix를 고정하면 Llama의 flywheel은 깨진다.

---

## 1. Tülu 3 — open reference recipe

Tülu 3(Allen AI, 2024년 11월, [[tulu-3]])는 Llama 3.1 base family(8B / 70B / 405B) 위에 만들어졌고, 현재까지 modern post-training stack에 대한 가장 완전한 public disclosure이다. 모든 prompt, completion, preference pair, verifier, hyperparameter, eval script가 public이다. 그 기여는 novel algorithm이 아니라 *reproducibility as a thesis*이다. 유일한 algorithmic novelty는 RLVR stage이며, §2.3에서 다룬다.

### 1.1 939K SFT mixture — composition table

Allen AI는 **18 component dataset**에 걸친 **939,344 prompts**를 보고하며, **57% public / 43% in-house synthetic**으로 나뉜다([[tulu-3-sft-mix]], [[allenai-tulu-synth]]). Dataset card에서 입증된 per-source breakdown은 다음과 같다.

| Source | Count | Role | Public / Synthetic |
|---|---:|---|---|
| Evol CodeAlpaca | 107,276 | code SFT | public (derivative) |
| Aya | 100,000 | multilingual | public |
| WildChat GPT-4 | 100,000 | real-user chat | public |
| FLAN v2 (ai2-adapt-dev) | 89,982 | reasoning / knowledge | public |
| NuminaMath-TIR | 64,312 | math with reasoning traces | public |
| WildGuardMix | 50,000 | safety / refusal | public |
| Tülu 3 Persona MATH | 149,960 | math (persona-synth) | in-house synthetic |
| Tülu 3 Persona GSM | 49,980 | grade-school math (persona-synth) | in-house synthetic |
| Tülu 3 Persona Python | 34,999 | coding (persona-synth) | in-house synthetic |
| Tülu 3 Persona Algebra | 20,000 | algebra (persona-synth) | in-house synthetic |
| Tülu 3 Persona IF | 29,980 | precise instruction-following (persona-synth) | in-house synthetic |
| Tülu 3 WildJailbreak | 50,000 | red-team safety (persona-synth) | in-house synthetic |
| Tülu 3 Hardcoded | 240 | identity / canned responses | in-house synthetic |
| CoCoNot | 10,983 | non-compliance / safe refusal | public |
| No Robots | 9,500 | hand-written instructions | public |
| OpenAssistant Guanaco | 7,132 | multi-turn chat | public |
| SciRIFF | 10,000 | scientific IF | public |
| TableGPT | 5,000 | table reasoning | public |
| **Total** | **939,344** | | **~57% / ~43%** |

같은 mixture를 *skill bucket*으로 볼 수도 있다([[allenai-tulu-sft-recipe]]): Chat/general 27%, Math 21%, Code 14%, Precise IF 11%, Safety 10%, Reasoning/knowledge 10%, Multilingual 7%. Per-bucket share는 recipe의 *tunable knob*이지 우연이 아니다. Ai2의 data-construction procedure는 **skill-mixture-first**이다. Math-only mixture를 만들고 math-only model을 train한다. Code-only mixture를 만들고 code-only model을 train한다. Isolated ablation에서 *그* skill을 가장 많이 움직이는 per-skill mixture를 유지한다. 그런 다음 merge, decontaminate, downsample하여 최종 aggregate를 balanced하게 만든다([[tulu-3-sft-mix]]).

### 1.2 Persona-driven prompt synthesis — 43%

Synthetic fraction은 generic self-instruct output이 아니다. **persona-driven prompt factory**로 만들어진다([[allenai-tulu-synth]]). Persona pool에서 persona(예: "neural networks에 집중하는 machine-learning researcher")를 sample하고, skill template("coding problem 만들기", "precise-IF task 쓰기")과 결합한 뒤, 그 combined prompt를 **GPT-4o**(general) 또는 **Claude-3.5-Sonnet**(coding, best-in-class code responses)에 prompt *generator*로 넣는다. 별도 pass가 completion을 생성한다. Persona step은 답뿐 아니라 *무엇이 질문되는지*를 diversify하며, dataset scale에서의 [[persona-hub]] design pattern의 public instantiation이다.

위 table의 `Tülu 3 Persona MATH` / `Persona GSM` / `Persona Python` / `Persona Algebra` / `Persona IF` entry는 이 factory의 직접 output이다. `WildJailbreak`는 adversarial persona로 만든 red-team slice이다. `Hardcoded` 240-row set은 모든 instruct model에 필요한 identity-fixing slice("who are you", "what can you do")이다.

### 1.3 Decontamination — dataset card만으로 충분하지 않은 이유

Decontamination은 *recipe의 일부*이지 afterthought가 아니다([[allenai-tulu-sft-recipe]]). Ai2는 모든 eval set(MMLU, GSM8K, MATH, IFEval, BBH, AlpacaEval, Arena-Hard, HumanEval)에 대해 두 filter를 실행한다.

- 어떤 training row와 어떤 eval row 사이든 **8-gram overlap ≥ 50%**이면 training row를 drop한다.
- 어떤 eval row와 **embedding cosine similarity > 0.9**이면 training row를 drop한다.

그다음 paper는 *surviving overlap rate per eval*을 공개하여 downstream user가 replicate하고 audit할 수 있게 한다. Operational rule은 대부분의 open dataset보다 엄격하다. Ai2는 어떤 eval suite와도 **2% 초과 overlap**하는 row를 명시적으로 제거하며, 이 제약은 HumanEval prompt와 겹치는 Evol-CodeAlpaca row를 놀랄 만큼 많이 죽인다([[tulu-3-sft-mix]]).

### 1.4 SFT hyperparameters and ablation findings

8B / 70B SFT config([[allenai-tulu-sft-recipe]])는 의도적으로 평범하다. Max seq 4096, AdamW (0.9, 0.95), LR **5e-6 (8B) / 2e-6 (70B)**, 3% warmup이 있는 linear schedule, **2 epochs**, global batch 128, BF16, FSDP FULL_SHARD(8B) / HYBRID_SHARD(70B), sequence packing on, response-only loss, gradient checkpointing on, **NEFTune off**(939K scale에서는 neutral로 측정).

입증된 skill-removal ablation이 paper에서 가장 많이 인용되는 수치다.

- **Persona-MATH** 제거 → **GSM8K −15 pp**. Math share는 GSM8K에 특히 load-bearing이다.
- 모든 **code** source 제거 → **HumanEval −12 pp**. Code data가 없으면 prose에서 compositional transfer가 생기지 않는다.
- **safety** slice(WildGuardMix + WildJailbreak) 제거 → capability eval은 거의 움직이지 않지만 **WildJailbreak가 98% → 52%**로 떨어진다. Safety는 보존하기는 싸고 다시 도입하기는 비싸다.
- 이 mix size에서는 **2 epochs > 1 epoch > 3 epochs**이며, 3 epochs는 IFEval을 해친다.
- **NEFTune**은 939K에서는 ~0이다(gain saturation). ≤ 100K에서는 작은 gain이 있다. NEFTune intuition은 noise가 small mixture를 regularize한다는 것이며, Tülu scale에서는 redundant하다.
- **Packing**: 2.5× throughput, quality delta 없음.

### 1.5 DPO와 RLVR — SFT 이후 gain이 이동하는 곳

SFT 이후 Tülu 3는 두 preference / RL stage를 쌓는다([[tulu-3]], [[allenai-tulu-blog]]).

- **DPO**는 ~270K-pair preference pool에서 수행한다. 이 pool은 (a) UltraFeedback + safety data에서 온 off-policy pair와 (b) *Tülu SFT* checkpoint에서 sample하고 trained reward model로 scoring한 on-policy pair를 결합한다. Length-normalized DPO, **β=5.0** at 8B, **LR 5e-7**.
- **RLVR** — Reinforcement Learning with Verifiable Rewards, 하나의 genuine algorithmic novelty([[rlvr-tulu3]]). Scalar reward는 learned RM이 아니라 **deterministic verifier** `v(x, y) → {0, 1}`이다. Math는 exact-match / SymPy equivalence, IFEval은 regex-style constraint check, code는 unit-test execution. PPO hparams: LR **3e-7**, **β_KL=0.05**, clip 0.2, **K=4 update epochs**, **GAE λ=0.95, γ=1.0**, **10M episodes**. No-RM property는 Goodhart drift를 우회한다. DPO-only 대비 측정된 gain은 **+5–10pp GSM8K, +~4pp IFEval, elsewhere neutral**이다. Concrete numbers: Tülu 3 8B는 **GSM8K 87.6** / **MATH 43.7** / **IFEval 82.4**에 도달하며, Llama-3.1 8B Instruct baseline은 84.7 / 41.5 / 80.5이다([[rlvr-tulu3]]).

Three-stage ordering인 SFT(capability breadth) → DPO(preference style, helpfulness, safety) → RLVR(sharpen verifiable reasoning)는 이제 canonical open-recipe template이다. OLMo 2는 이를 그대로 채택한다.

### 1.6 이름 붙일 만한 pitfall — verifier loopholes

No-RM 이야기는 따로 놓고 보면 너무 깔끔하다. RLVR의 Goodhart-proof property는 *verifier가 tight할 때만* 성립한다. [[rlvr-tulu3]]는 failure mode를 명시적으로 지적한다. `"42"`라는 string을 포함한 어떤 line이든 받아들이는 허술한 math grader는 policy가 distractor sentence 안에 `"42"`를 내보내서 hack하게 만든다. 따라서 verifier engineering은 unit-test engineering이다. Public Tülu 3 verifier는 strict하게 구성된다. MATH는 SymPy equivalence, GSM8K는 exact integer match, code는 sandboxed 5-second timeout이 있는 unit-test pass, IFEval은 regex constraint-match이다. RLVR prompt set은 known reference answer와 working verifier가 있는 prompt로 gate된다. 그 밖의 모든 것은 learned RM이 유일한 option인 RLHF/DPO로 routed된다.

---

## 2. Tülu 3 → Tülu 3.1 — narrow delta

"Tülu 3.1"은 새 family가 아니다. 어떤 Ai2 문서를 읽느냐에 따라 (a) 8B checkpoint에 대한 **single-stage post-training update**이거나, (b) Tülu 3 recipe를 Llama 3.1뿐 아니라 OLMo 2에도 unchanged로 적용한 **multi-base refresh**를 의미한다. 두 framing 모두 문서화되어 있다. 의미 있는 algorithmic delta는 (a)이다.

### 2.1 8B single-stage delta — PPO → GRPO

`allenai/Llama-3.1-Tulu-3.1-8B`의 HF model card는 명시적이다([[tulu-3.1]]). Parent checkpoint는 `allenai/Llama-3.1-Tulu-3-8B-DPO`이고 *유일한* 변화는 **final RL stage**에 있다. 구체적으로:

- Algorithm이 **PPO → GRPO**로 바뀐다. GRPO는 value network를 제거하고, 각 prompt의 rollout group 안에서 group-level return baseline으로 advantage를 추정한다.
- Final stage는 **reward model 없이** 실행된다. Tülu 3와 같은 RLVR-only지만 RL optimizer가 PPO 대신 GRPO이다.
- Final stage의 hyperparameter는 retune되지만, Ai2는 delta를 공개하지 않는다.
- RL training mix는 `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`이다.
- SFT와 DPO stage는 Tülu 3에서 unchanged로 상속된다.

### 2.2 이 delta가 중요한 이유

Tülu 3.1은 **controlled public ablation**으로 가치가 있다. SFT, DPO, data, base model을 고정하고 PPO→GRPO swap만 isolate한다. 이는 드물다. 대부분의 open release는 base model, data, algorithm을 동시에 바꾸므로 community가 gain을 attribute할 수 없다. Tülu 3.1은 이 base, 이 data, 이 DPO checkpoint에서 GRPO가 PPO보다 더 높은 average를 만들고, 이를 증명하는 delta table이 있다고 말한다. 또한 DeepSeek-R1과 DeepSeekMath가 부각시킨 GRPO / verifier-style RL path를 public Ai2 model이 처음 채택한 사례이기도 하다([[tulu-3.1]]).

### 2.3 Tülu 3 → 3.1 delta table

| Stage | Tülu 3 (Nov 2024) | Tülu 3.1 (2025) | Changed? |
|---|---|---|---|
| Base | Llama 3.1 8B | Llama 3.1 8B | no |
| SFT mix | 939K (Tülu-3 SFT mixture) | same | no |
| DPO data + β | ~270K pairs, β=5.0, LR 5e-7 | same (inherits `Llama-3.1-Tulu-3-8B-DPO`) | no |
| RL algorithm | **PPO** | **GRPO** | **yes** |
| Reward model in final RL stage | none (RLVR verifier only) | none (RLVR verifier only) | no |
| RL data | RLVR prompts (GSM / MATH / IF / code) | `RLVR-GSM-MATH-IF-Mixed-Constraints` | relabeled / retuned |
| RL hparams | LR 3e-7, β_KL 0.05, clip 0.2, K=4 | retuned (not published) | yes |
| Per-base refresh | Llama 3.1 only | also runs on OLMo 2 base ([[tulu-3-1]]) | additive |

[[tulu-3-1]]의 *per-base refresh* framing은 3.1을 "new base 위에서 whole stack을 다시 실행"한 것으로 다룬다. 이는 recipe가 base-agnostic하다는 valuable evidence이지만 algorithmically new하지는 않다. Blog-driven narrative와 HF-card-driven narrative는 일관된다. PPO → GRPO를 명시적으로 이름 붙이는 것은 HF card이다.

---

## 3. Llama 3 — six-round flywheel

Llama 3([[llama-3]], Grattafiori 2024)는 반대 topology를 취한다. Linear SFT → DPO → RLVR pipeline이 아니라, post-training이 **six times repeated closed loop**이다. 각 round는 fresh human preference로 fresh **reward model**을 훈련하고, 그 RM으로 current best chat checkpoint의 generation을 **rejection-sample**하여 새 SFT pool을 만들고, 새 pool로 **SFT**하고, 새 preference data로 **DPO**한다. Round-*N+1*의 reward model, SFT pool, DPO pair는 모두 round-*N* checkpoint의 output에서 나온다.

### 3.1 Per-round mechanics

여섯 라운드 각각은 같은 inner primitive를 사용한다.

- **SFT data를 위한 rejection sampling.** 각 prompt에 대해 best round-(N−1) chat model에서 temperature **T=0.6–1.0**으로 **K=10–30 completions**를 sample한다. Fresh RM으로 각각 scoring하고 top-scoring completion을 유지한다. 그다음 distilled topic classifier와 distilled quality classifier(둘 다 Llama 3에서 온 것)가 SFT 전에 low-quality rejection-sampled text를 제거한다. Rejection-sampled output이 SFT pool을 *지배*한다.
- **SFT** at LR **1e-5** (405B), cosine decay, context 8K–32K extended, response token에만 loss.
- **Reward model.** Llama 3 pretrained checkpoint에서 initialize하고, LM head를 linear head로 대체한다. Pairwise preference를 margin label("significantly better", "better", "slightly better", "negligibly better")로 훈련한다. Margin은 data filtering / up-weighting에 사용되지만 loss 자체에는 들어가지 않는다. 표준 pairwise logistic이다.
- **DPO** at LR **1e-5**, **β=0.1**, chosen-logprob decay를 막기 위한 **auxiliary NLL-on-chosen loss with coefficient 0.2**. 라운드당 single epoch, prompt token은 loss에서 mask, *most-recent-batch preference data only*. Older batch는 format drift를 일으킨다.
- **Per-round data mix:** ~50–80% rejection-sampled synthetic, 나머지는 human SFT, capability-specific synthetic(code-exec-filtered code, verifier가 붙은 math, multi-turn tool-use traces, long-context QA), 최신 preference batch이다([[llama-3]]).

### 3.2 Six-round table — 무엇이 바뀌고 각 round가 무엇을 잡았나

Meta는 **per-round eval delta를 공개하지 않는다**. 아래 round-by-round interpretation은 [[llama-3]]와 [[llama-3-synthetic-pipeline]]의 public prose에서 *inferred*한 것이며, direct Table이 아니다. Emphasis의 sequence는 attested이다. 특정 eval issue를 한 round가 "caught"했다는 것은 reasoned reconstruction이며 그렇게 label한다.

| Round | Data source emphasis | Eval issue round targets (inferred) | Attested? |
|---|---|---|---|
| 1 | Human SFT demonstrations dominant; limited rejection-sampled data (no strong RM yet) | Basic instruction following, chat format | attested (round-1 uses smaller synthetic share) |
| 2 | First large RM trained; rejection sampling from round-1 best chat model | Coding / math format regressions found in round-1 | inferred |
| 3 | Synthetic code-exec-filtered + math-verifier data ramps | Reasoning and code-pass@1 lift; factuality pipeline enters | inferred from [[llama-3-synthetic-pipeline]] factuality description |
| 4 | Multi-turn tool-use traces added; long-context QA added | Tool-call format drift, long-context retention | inferred |
| 5 | Multilingual + safety synthetic augmentation; Llama Guard 3 co-trained | Multilingual regressions; refusal calibration | inferred |
| 6 | Newest preference batch only; DPO on highest-quality pairs; final polish | Remaining format drift; final helpfulness / harmlessness trade-off | inferred |

*Attested* narrative thread는 (a) synthetic share가 round를 거치며 증가하고, (b) reward model이 fresh preference 위에서 매 라운드 *from scratch* rebuild되며, (c) **stale preference는 절대 재사용되지 않는다**는 것이다. Prose reason은 policy change 아래에서 preference distribution이 drift하고 old pair를 재사용하면 past policy의 quirk에 overfit한다는 것이다. Safety classifier인 Llama Guard 3는 round 전반에 걸쳐 jointly train되지만, core policy에는 round 5까지 들어가지 않는다.

### 3.3 Multi-round가 single-pass를 이기는 이유 — attested reasoning

[[llama-3]]와 [[llama-3-synthetic-pipeline]]에서 distilled한 Meta의 six-round 이유:

1. **Reward-model drift.** Static RM은 policy가 shift하면 stale해진다. Yesterday's policy에 대해 측정한 reward는 new policy가 이미 벗어난 artifact를 over-optimize한다. RM을 매 라운드 retrain하면 Goodhart gap을 bounded하게 유지한다.
2. **Rejection-sampled SFT data improves monotonically.** Fresh RM으로 filtering된 round-*N*의 best output이 round-*N+1* SFT data가 된다. Policy는 말 그대로 자체 curriculum을 bootstrap한다. Round 1의 SFT pool은 generator가 더 약했기 때문에 round 6의 pool보다 약하다.
3. **Preference-label quality compounds.** Human annotation에 대한 여러 QA round(Meta가 명시적으로 multiple QA rounds라고 말함)는 later-round preference를 earlier-round preference보다 깨끗하게 만들고, DPO step은 더 많은 signal을 얻는다.
4. **Capability recovery.** 각 라운드의 eval pass는 특정 capability의 regression을 잡는다(Meta는 어떤 round가 어떤 regression을 잡았는지 공개적으로 나열하지 않는다). 다음 round의 synthetic mix는 그 specific gap을 고치도록 reweighted된다. 이것이 "data flywheel"이다. Evals → mixture reweight → next round.
5. **Format drift is local.** *Most-recent-batch* data만으로 DPO를 훈련하면 각 round의 format drift가 한 round의 width로 제한된다. Old batch는 policy를 stale format 쪽으로 끌어당긴다.

Topology cost는 각 라운드마다 fresh preference collection(labour-intensive)과 fresh RM train이 필요하다는 것이다. Tülu 3 alternative(SFT once, DPO once, RLVR once)는 labour를 피하지만, RL이 실행된 *후에야* 드러나는 capability regression을 잡는 능력을 잃는다.

---

## 4. Side-by-side — 두 recipe가 가르치는 것

| Axis | Tülu 3 | Llama 3 |
|---|---|---|
| Topology | linear, 3 stages | closed loop, 6 rounds |
| SFT data | 939K fixed, published row-by-row | regenerated per round from best prior checkpoint |
| Synthetic share | 43% (persona-factory) | 50–80% per round (rejection-sampled) |
| RM role | DPO scoring + off-policy ranking | core loop primitive; rebuilt every round |
| Final RL | RLVR (no RM) | DPO (no PPO in final public recipe; Llama 2 had PPO, Llama 3 dropped it) |
| Disclosure | full (every prompt, prefix, hparam) | partial (loop structure + hparams; per-round mix not published) |
| Labour cost | one preference pool | six preference batches + six RM trains |
| Fix-surface | mixture reweight before SFT | per-round evals → next round reweight |
| Recipe status | open-recipe reference; copied by OLMo 2/3 | flagship industrial; copied in spirit by Qwen 2.5/3 |

이 table을 읽는 올바른 방법은 "어느 하나가 더 낫다"가 아니다. Tülu 3는 publish할 수 있고 six rounds of annotation은 감당할 수 없을 때 만드는 것이다. Llama 3는 annotation을 감당할 수 있고 publish할 수 없을 때 만드는 것이다. 공통 substrate는 둘 다 **data composition을 first-order training variable**로 취급한다는 것이다. 그리고 둘 다 modern post-training은 fine-tuning이 아니라, 입력이 data mixture이고 출력이 checkpoint인 program이라는 점을 내재화했다.

두 번째 읽기: **두 recipe는 어디서 fail하는가?** Tülu 3는 eval suite에 mixture가 커버하지 않는 blind spot이 있을 때 조용히 fail한다. Mixture가 어떤 eval이든 실행되기 전에 고정되므로, missing capability는 RLVR까지 눈치채지 못한 채 propagate된다. Llama 3는 human preference annotation quality가 mid-round에 떨어지면 비싸게 fail한다. Loop가 labour를 compound하기 때문에 bad annotation batch가 RM을 contaminate하고, RM이 rejection-sampled pool을 contaminate하고, 그것이 DPO pair를 contaminate한다. 두 team 모두 같은 risk family(Goodhart drift, stale preferences, reward-model over-optimization)를 설명하지만 위치를 다르게 잡는다. Tülu 3는 이를 pre-mixture로 밀고, Llama 3는 per-round로 민다.

세 번째 읽기, historical: Llama 3는 Llama 2의 recipe에서 PPO를 *dropped*하고 DPO를 택했다. Tülu 3는 reward signal을 바꾸어 RLVR label 아래 PPO를 *reintroduced*했다. LLM을 위한 RL이 preference-based method로 수렴했다는 community summary는, 가장 많이 복사되는 두 open recipe가 online RL을 쓸지 말지에 동의하지 않는다는 사실을 숨긴다. Tülu 3.1의 PPO → GRPO swap은 또 하나의 wrinkle이다. Ablation-grade checkpoint 하나의 좁은 stage에서 *세 번째* RL family를 선택한다. 다음 장들(Qwen 2.5/3, OLMo 2/3, Phi 3/4, Nemotron)은 각각 이 space의 한 지점을 선택하고 그 이유를 이름 붙인다.

---

## Companion visualization

**[figures/tulu-llama-recipe.html](figures/tulu-llama-recipe.html)** — side-by-side interactive. 왼쪽은 Tülu 3 linear pipeline(세 stage와 per-stage eval contribution), 오른쪽은 Llama 3 six-round flywheel(round selector, per-round synthetic share, capability별 inferred eval-delta bar)이다. Round를 따라가면 Llama 3의 mixture가 어떻게 synthetic 쪽으로 기울고 각 round의 targeted capability가 어떻게 움직이는지 볼 수 있다. §3.2를 읽은 뒤 flywheel에 대한 직관을 만들기 위해 사용하라. 어떤 round가 reasoning을, 어떤 round가 tool use를, 어떤 round가 refusal calibration을 target하는지 볼 수 있다.

---

## Connections

- **ch-30 (SFT design axes)** — Tülu 3와 Llama 3는 design axes space에서 서로 다른 지점을 instantiate한다(mixture scale, synthetic ratio, epoch count).
- **ch-31 / ch-32** — decontamination, skill-mixture-first, persona synthesis는 이전 장에서 전개되었고, 이 장에서 실제 released model에 나타난다.
- **ch-34 (Case Studies B)** — Qwen 2.5/3, OLMo 2/3, Phi 3/4는 모두 이 두 template 위에 구축된다. OLMo 2가 Tülu 3 recipe의 가장 가까운 direct reuse이다.
- **ch-35 (Case Studies C)** — Nemotron과 R1-distill은 synthetic ratio를 95% 이상으로 밀고 이야기를 더 바꾼다.
- **RL track (ch-37..ch-46)** — Tülu 3의 RLVR은 verifiable-rewards family로 들어가는 on-ramp이다. Llama 3의 DPO + NLL-stabilizer는 canonical DPO case study이다.
- **Eval track (ch-47..ch-53)** — 여기서 쓰인 decontamination rule과 effective-context discipline은 eval-side counterpart이다.

## Further reading

- [[tulu-3]] — 2024 Tülu 3 tech report; §1의 기반 paper.
- [[tulu-3.1]] — 8B PPO→GRPO delta에 대한 HF model card.
- [[tulu-3-1]] — multi-base refresh framing.
- [[tulu-3-sft-mix]] — §1.1이 끌어온 per-source mix card.
- [[allenai-tulu-sft-recipe]] — bucket shares, decontam, SFT hparams, ablations.
- [[allenai-tulu-blog]] — fully-open ethos와 RLVR introduction.
- [[allenai-tulu-synth]] — persona-driven prompt factory.
- [[llama-3]] — Herd of Models report; §3의 기반 paper.
- [[llama-3-synthetic-pipeline]] — iterative loop와 `synthetic-data-kit`에 대한 Meta blog.
- [[rlvr-tulu3]] — §1.5에서 사용한 8B number가 있는 RLVR methodology page.
