<!-- chapter: ch-34
     track: sft
     kind: content
     title: Case Studies B — Qwen 2.5/3, OLMo 2/3, Phi 3/4
     deps: [ch-33]
     sources: [[qwen-2.5]], [[qwen-3]], [[qwen-3-5]], [[olmo-2]], [[olmo-3]], [[phi-3]], [[phi-4]],
              [[phi-1-5]], [[phi-textbooks]], [[alibaba-qwen]], [[allen-ai]],
              [[tulu-3]], [[dpo]], [[grpo]], [[deepseek-r1]]
     figures: figures/lab-compare.html
-->

# 34장 — Case Studies B: Qwen 2.5/3, OLMo 2/3, Phi 3/4

> **핵심 통찰.** Ch-33은 Tülu 3와 Llama 3를 보여주었다. 이들은 SFT, preference tuning, RL을 별도의 named box로 stage화한 두 *explicit* recipe이다. Ch-34는 그 template을 각기 다른 load-bearing 방식으로 *변형*하는 세 lab을 보여준다. Qwen은 SFT를 **two-phase context curriculum**으로 만들고 DPO에 stabilizer(Online Merging Optimizer)를 더해 post-training boundary를 접는다. Qwen 3는 더 나아가 `/think`와 `/no_think`라는 **두 mode를 하나의 model이 carry하도록** prompt-level data-labeling convention을 통해 훈련한다. OLMo 2와 OLMo 3는 recipe를 동일하게 유지하지만 **stage 자체를 artifact**로 만든다. OLMo 3의 model-flow는 Base → Think/Instruct/RL-Zero branch이므로 capability addition을 branch diff에서 읽을 수 있다. Phi 3/4는 반대쪽에서 boundary를 흐린다. Synthetic-heavy pretraining이 너무 많은 "SFT-shaped" data를 upstream으로 밀어 넣어서 post-training SFT는 *refinement* stage가 되고, Phi-4-reasoning의 1.4M long-CoT trace는 SFT ceiling이 지배적임을 보여준다. 90 GRPO step은 그 위에 마지막 +10% AIME를 더한다.
>
> **가이드라인.** 2025-era model report를 읽을 때 "SFT + DPO + RL을 했는가"로 채점하지 마라. 세 orthogonal axis로 채점하라. (i) **SFT boundary가 어디에 있는가** — synthetic pretraining 안으로 upstream push되었는가(Phi), distinct long-CoT cold start로 유지되는가(Qwen 3), Tülu-identical stage로 유지되는가(OLMo)? (ii) **하나의 checkpoint가 몇 mode를 carry하는가** — 하나(Phi, OLMo 2), 둘(Qwen 3 hybrid), 또는 family-branch(OLMo 3)? (iii) **pipeline을 무엇이 stabilize했는가** — Online Merging Optimizer(Qwen 2.5), QK-Norm + Z-loss architecture(OLMo 2), pivotal-token DPO(Phi-4), length-aware GRPO reward(Phi-4-reasoning)? 아래의 모든 recipe는 같은 three-stage skeleton이다. 차이는 이 세 axis에 있다.

---

## 이 장이 존재하는 이유

Ch-33은 *canonical* case였다. 가능한 가장 명확한 형태로 recipe를 공개한 두 lab이다. Ch-34의 가치는 여기의 각 lab이 이후 recipe(Nemotron, distillation, reasoning-first)가 상속할 방식으로 canonical form을 깨뜨린다는 데 있다. Qwen 2.5는 two-stage SFT context curriculum의 출발점이다. Qwen 3는 모든 2026 hybrid model이 이제 carry하는 `/think` toggle의 출발점이다. OLMo 3는 model-flow-as-artifact stance의 출발점이다. Phi-4-reasoning은 SFT data가 o3-mini-generated long CoT일 때 SFT가 RL을 지배한다는 가장 깔끔한 evidence이다. 이 네 가지를 하나의 template 변형으로 읽는 것이 여섯 recipe를 암기하는 대신 design space를 추출하는 방법이다.

---

## 1. Six-lab comparison table

모든 숫자는 `wiki/raw-data/llm-training/` 아래의 attested raw-data source에서 온다. *nd*로 표시된 entry는 public report에서 명시적으로 공개되지 않은 것이며, inferred default는 가장 가까운 cousin과 함께 `infer:`로 label한다.

| Lab / model | SFT size | SFT LR | SFT optimizer | Data mix composition | Preference / RL | Eval delta |
|---|---|---|---|---|---|---|
| **Qwen 2.5** (72B Instruct) | 1M SFT examples across SFT+DPO+GRPO stages; **Phase 1** ≤32,768 tokens (short-instruct); **Phase 2** mixed short + long up to **262,144 tokens** | nd (infer: Qwen2 cousin ~1e-5) | AdamW (SFT); **Online Merging Optimizer** at DPO | 1M SFT split by context phase; 150K DPO pairs via SFT-resample + quality filter (pass=chosen, fail=rejected) | DPO: **LR 7e-7, 1 epoch**, β nd (infer: 0.1); GRPO with variance-ordered prompts (high-variance first) | 72B-Instruct: MMLU 86.1 / HumanEval 85.4 / MATH 83.1 / IFEval 86.1 |
| **Qwen 3** (235B-A22B MoE) | Long-CoT cold-start SFT at Stage 1 (size nd in public report), then Stage 3 fuses thinking + non-thinking SFT data | nd | AdamW (infer) | Stage 1 long-CoT cold start → Stage 2 reasoning RL (math/code) → Stage 3 thinking-mode fusion SFT → Stage 4 general RL; small models via on/off-policy strong-to-weak distillation | Stage 2 GRPO (infer: inherits Qwen 2.5 GRPO); Stage 4 general-domain RL | Unified `/think` and `/no_think` toggles on one checkpoint; thinking-budget exposed at inference |
| **Qwen 3.5** (397B-A17B MoE, Feb 2026) | Presumed scaled Qwen 3 mix; **no Qwen 3.5 tech report at Qwen 3 detail level** | nd | nd | nd beyond "broadly consistent with Qwen 3 four-stage" | GRPO assumed; not re-documented | Inference-side: 1M context, 19× decoding speedup; post-training changes not disclosed |
| **OLMo 2** (7B/13B/32B) | OLMo-variant of Tülu 3 SFT mix, **~939K prompts** with OLMo chat template | Tülu 3 defaults (lightly re-tuned per size) | AdamW | Two-stage **pretraining** (OLMo-Mix-1124 ~3.9T tokens → Dolmino ~50B cooldown); SFT/DPO/RLVR mixes inherited from Tülu 3 | DPO on-policy preferences + Tülu 3 pool; **RLVR = PPO** with verifiable rewards (GSM8K/MATH/IFEval/code); **LR 3e-7, β_KL 0.05, clip 0.2, GAE λ 0.95, 4 PPO epochs/step** | RLVR lifts GSM8K/MATH by single-digit pp at 7B/13B; 32B first fully-open to beat GPT-3.5 / GPT-4o-mini avg |
| **OLMo 3** (7B/32B, Base/Think/Instruct/RL-Zero) | Dolci post-training suite with separate **SFT / DPO / RLVR** mixes (Think branch uses thinking-specific SFT) | inherits Tülu 3 / OLMo 2 defaults | AdamW; **Olmo Core** SFT stack (8× throughput vs Open Instruct) | Three-stage base: Dolma 3 Mix ~5.9T pretrain → Dolmino 100B mid-training → Longmino 50B long-context; Dolci post-training | SFT → DPO → RLVR on each branch; **4× RL efficiency** from in-flight weight updates + continuous batching | OLMo 3-Think 32B competes with Qwen 3 / DeepSeek-R1-Distill class on reasoning; stage diffs publicly attributable |
| **Phi-3** (mini 3.8B / small 7B / medium 14B) | SFT volume **not publicly itemized** for Phi-3 | nd | nd (infer: AdamW) | Two-phase **pretraining**: Phase 1 filtered web, Phase 2 synthetic textbook-like (GPT-3.5/4 class teacher); 3.3T tokens (mini), 4.8T (small/medium); SFT on curated synthetic + human pairs | DPO with dedicated **safety / responsible-AI preference slice**; β/LR/batch nd | mini-3.8B reaches Mixtral 8x7B parity on some benchmarks; on-device ~2GB at 4-bit |
| **Phi-4** base (14B, Dec 2024) | Phi-4-base SFT size nd beyond "synthetic data injected into pretraining and post-training" | nd | nd (infer: AdamW) | **~400B unweighted tokens of synthetic across 50 categories** (pretrain + post-train); rejection-sampled SFT | **Pivotal-token DPO**: preference pairs constructed at tokens where P(final-correct) changes most | Data-first scale evidence: 14B closes on 70B-class on targeted benchmarks |
| **Phi-4-reasoning** (Apr 2025) | **1.4M prompts** at "boundary of base-model capability"; **~16B SFT tokens, ~8.3B unique** long-CoT traces | nd | nd (infer: AdamW) | o3-mini high-thinking 32K-context generations; STEM + coding + safety domains | **GRPO for 90 steps**; reward = +1 correct / −0.5 incorrect + **length-aware bonus** + n=5 n-gram repetition penalty + missing-EOS / unclosed-`<think>` penalties | +10% AIME from only 90 GRPO steps; SFT ceiling dominates RL contribution |

이 table을 column by column으로 읽어라. **LR column**은 SFT에서 이 lab들을 구분하는 것이 거의 없음을 말한다. 공개된 Qwen 2.5의 DPO LR 7e-7이 하나의 precise anchor이고, Tülu 3 / OLMo RLVR의 3e-7이 다른 anchor이다. **Optimizer column**은 Qwen의 Online Merging Optimizer와 OLMo 3의 Olmo Core throughput win이 isolate되는 곳이다. **Mix column**은 Phi가 template을 완전히 변형하는 곳이다. 다른 lab이 SFT data라고 부르는 대부분을 Phi는 Phase-2 pretraining이라고 부른다. 그리고 **RL column**은 그 변형이 pattern이 되는 곳이다. OLMo는 PPO-RLVR을, Qwen과 Phi-4-reasoning은 GRPO를 하고, DPO는 shared preference stage로 그 사이에 있다.

각 lab을 걸어가기 전에 table의 두 속성을 짚을 필요가 있다.

- **nd / withheld pattern은 유익하다.** Phi는 aggregate synthetic volume(400B unweighted tokens)을 보고하지만 SFT count와 DPO β를 withholding한다. Qwen은 DPO LR + optimizer를 보고하지만 GRPO group size를 withholding한다. OLMo는 전체적으로 가장 많은 숫자를 publish하지만, 이는 Tülu 3 default를 상속하기 때문이다. 각 lab이 숨기는 것의 *shape*가 그 lab이 secret sauce로 여기는 것의 shape이다.
- **"SFT size"는 하나의 숫자가 아니다.** Qwen 2.5의 1M은 SFT+DPO+GRPO prompt를 함께 센다. OLMo 2의 939K는 SFT-only이다. Phi-4-reasoning의 1.4M은 ~16B-token o3-mini expansion 이전의 prompt이다. 어떤 두 row든 "SFT size"로 비교하기 전에 normalize하라. 보통 *unique SFT tokens*라는 common definition으로 다시 세야 row 간 comparison이 의미 있다.

### 1.1 두 번째 view: pretraining-scale lens

Post-training이 이 장의 focus지만, SFT-boundary axis는 SFT 전에 얼마나 많은 data가 흡수되었는지 알아야 의미가 있다. 같은 여섯 lab을 pretraining scale과 context-extension strategy로 당겨 보자.

| Lab / model | Pretrain tokens | Native ctx → extended | Mid-training / cooldown | Notable architectural stability |
|---|---|---|---|---|
| Qwen 2.5 (72B) | **18T** (up from Qwen 2's 7T) | 4K → 128K (1M via YARN extrapolation) | none separately named | GQA + SwiGLU + RoPE + RMSNorm; tied embeddings small sizes |
| Qwen 3 (235B-A22B) | **36T across 119 languages** — general stage >30T @ 4096, reasoning stage ~5T @ 4096, long-context stage hundreds of billions @ 32768 | 4K → 128K (YARN + Dual Chunk Attention) | dedicated reasoning stage inside pretraining | **QK-Norm** added for stability |
| OLMo 2 (7B/13B/32B) | **~3.9T** (Stage 1 OLMo-Mix-1124) + **~50B** Dolmino cooldown | 4K → 32K in cooldown | Dolmino higher-quality mix | RMSNorm + reordered norm + **QK-Norm** + Z-loss + init preserving activation scale |
| OLMo 3 (7B/32B) | Dolma 3: **9.3T** source → **5.9T** pretrain mix → **100B** Dolmino mid-training (sampled from 2.2T pool) → **50B** Longmino (from 639B long-doc pool) | per-branch long-context via Longmino | Dolmino + Longmino named stages | inherits OLMo 2 stability tricks |
| Phi-3 (mini/small/medium) | **3.3T (mini) / 4.8T (small) / 4.8T (medium)** | 4K native (128K via LongRope on mini; small = 8K; medium = 4K) | two *phases* inside pretraining — Phase 1 web-heavy, Phase 2 synthetic-heavy | conventional; data-first, not architecture-first |
| Phi-4 (14B) | not re-disclosed at Phi-3 granularity; **~400B unweighted synthetic tokens across 50 categories** injected into pretraining and post-training | 16K extended | synthetic data phase is the "mid-training" analogue | conventional |

세 가지 관찰을 가져가야 한다. 첫째, **pretraining-token column은 여섯 lab 사이에서 10× 범위**(Phi-3-mini의 3.3T부터 Qwen 3의 36T까지)를 가진다. 반면 SFT volume은 ~3× 안(939K부터 1.4M prompts)에 모인다. Post-training은 convergence가 일어난 step이다. Pretraining은 여전히 lab들이 recipe에 필요한 data 규모에 대해 의견이 갈리는 곳이다. 둘째, **Qwen 3만이 pretraining 안에서 "reasoning stage"를 이름 붙인다**(~5T tokens). 이는 long-CoT cold-start SFT가 stylize하는 reasoning prior를 pre-install하는 곳이다. OLMo 3도 Dolmino(100B math/code/science/IF/reading-comp tokens)를 통해 개념적으로 같은 일을 하지만 훨씬 작은 relative scale에서 한다. 셋째, **OLMo 2와 Qwen 3는 독립적으로 QK-Norm에 도달했다**. Non-Llama base에 대한 stability fix로서 cross-lab convergence이며 주목할 만하다. Ch-33의 Llama 3는 이를 필요로 하지 않는다(its base는 여러 generation에 걸쳐 tune되었다).

### 1.2 2025년 practitioner에게 이것이 뜻하는 것

2025년에 이 recipe 위에 무언가를 만든다면, 위 table은 짧은 decision tree로 collapse된다. **(a) 하나의 mode를 carry할 것인가, 둘을 carry할 것인가?** 하나라면 OLMo 2를 따르라(Tülu 3 inheritance, single checkpoint, stable defaults). 이는 path of least resistance이다. 둘이라면 Qwen 3의 `/think` + `/no_think` data-labeling contract를 Stage 3 fusion SFT에서 adopt해야 한다. Mode-gating을 위한 더 싼 mechanism은 공개된 적이 없다. **(b) Frontier teacher를 위한 API budget이 있는가?** 있다면 Phi-4-reasoning을 따르라. Budget을 o3-mini-class의 1M+ long-CoT trace에 쓰고 RL은 ~100 GRPO step으로 제한하라. 없다면 OLMo 2/3의 RLVR-with-verifiable-rewards path를 따르라. Reward signal은 trained RM이 아니라 math verifier나 code unit test이다. **(c) 처음부터 pretraining하는가?** 그렇다면 Phi의 Phase-1/Phase-2 synthetic-heavy curriculum이 이 여섯 recipe 중 sub-10B 모델로 mid-size model parity를 달성한 유일한 recipe이다. Trade-off는 total teacher-API cost이다. 아니라면(continued pretraining 또는 post-training만 한다면), Qwen two-phase SFT context curriculum이 4K-32K에서 pretrained된 base에 long-context instruction following을 추가하는 가장 small-overhead path이다.

---

## 2. Qwen 2.5 — two-stage SFT context curriculum

Qwen 2.5는 중요한 세 가지를 공개했다.

1. **SFT 안의 context curriculum.** Pretraining에서 "짧은 것 먼저 학습하고 나중에 extend"하는 것이 아니다. 그것은 별개이다. *SFT run 내부에서* **Phase 1**은 32,768 token으로 cap된 short instruction이고, **Phase 2**는 short(≤32K)와 long(up to 262,144) instruction을 섞는다. 주장은 short phase가 instruction-following quality를 lock in하고, mixed phase가 short task를 *regress하지 않으면서* long-context instruction following을 가르친다는 것이다. Mix 없이 long만 train하면 short-task quality가 collapse된다. 이는 ProLong과 Llama-3가 쓰는 long-context pretraining two-stage의 SFT analogue이지만, pipeline에서 한 step 뒤로 이동한 것이다.
2. **DPO의 Online Merging Optimizer.** LR 7e-7과 1 epoch를 쓰는 standard DPO 자체가 이미 tight하지만, Qwen은 DPO pass 동안 **running merged checkpoint**를 유지한다. 이는 사실상 time-averaged policy로, single-epoch-with-small-preference-set regime에서 gradient를 안정화한다. Raw-data source는 "β not explicitly reported in the public tech report text (standard DPO beta ~0.1 assumed)"라고 적는다. 따라서 attested DPO number는 LR 7e-7과 1 epoch뿐이다.
3. **Variance-ordered GRPO prompts.** DPO 뒤에 Qwen은 GRPO를 실행한다. Novel knob은 **reward model 아래에서 response score variance가 큰 순서로 prompt를 정렬**하는 것이다. High-variance prompt, 즉 RM이 good rollout과 bad rollout을 강하게 구분하는 prompt를 먼저 훈련한다. Intuition은 low-variance prompt가 너무 쉽거나(RM이 모든 rollout에 비슷하게 높은 점수를 줌) 너무 어렵다(모두 낮음)는 것이다. 이 경우 GRPO의 advantage estimator가 사용할 것이 없다.

따라서 Qwen 2.5 recipe는 다음과 같다. Pretraining(18T tokens, Qwen 2의 7T에서 증가) → **1M examples에 대한 SFT two-phase context curriculum** → **DPO (150K pairs, LR 7e-7, OMO, 1 epoch, β≈0.1 assumed)** → Qwen-architecture RM(policy-matched with linear head)을 사용한 **variance-ordered GRPO**. 이것은 다른 lab들이 인용하는 canonical "SFT + DPO + GRPO stacked in order" recipe이다. 이 장 전체를 anchor하는 Qwen2.5-72B-Instruct의 disclosed score(MMLU 86.1 / HumanEval 85.4 / MATH 83.1 / IFEval 86.1)는 OLMo 2의 32B와 Phi-4-reasoning의 14B가 훨씬 적은 pretraining data로 따라잡으려 하는 숫자이다.

**"Phase 2 mixing prevents regression"이 mechanically 의미하는 것.** SFT를 long instruction만으로 train하면 short-instruction distribution에 대한 gradient가 stale해진다. 모델이 100K token에 걸쳐 attend하는 법을 배우는 동안 2K-token input을 위한 attention pattern은 refresh되지 않는다. Mixed Phase 2는 두 distribution을 in-batch로 유지하므로 short-range에 specialize한 attention head가 drift하지 않는다. 이는 "long-context continued pretraining에는 항상 short data를 포함하라"(ProLong / [[long-context-llama3]])와 같은 논리를 SFT 안에 적용한 것이다.

전체 preference-data-construction pipeline은 [[excerpts/qwen-2.5-post-training]]을 보라.

---

## 3. Qwen 3 — hybrid thinking as a data-labeling problem

Qwen 3의 innovation은 algorithmic이라기보다 organizational이다. 같은 계열의 GRPO와 DPO primitive를 사용하지만, **하나의 model family가 두 mode**를 carry한다. `/think`는 long-CoT deliberation용이고 `/no_think`는 instant response용이다. 이것이 작동하는 유일한 방법은 SFT data가 각 example이 어느 mode에 속하는지 *명시적으로 label*하는 것이다.

Hybrid-thinking contract를 풀기 전에 Qwen 3가 upstream에서 무엇을 지불했는지 알아야 한다. Pretraining은 **119 languages and dialects에 걸친 36T tokens**로, Qwen 2.5의 18T의 약 2배이다. **Three-stage curriculum**을 갖는다. 4096 sequence length에서 >30T tokens의 general stage, 역시 4096에서 **~5T higher-quality STEM/coding/synthetic tokens의 reasoning stage**, 그리고 32768에서 "hundreds of billions" token의 long-context stage이다. Data expansion은 sibling model을 synthetic generator로 활용했다. PDF에서 OCR-style extraction은 **Qwen2.5-VL**, synthetic math는 **Qwen2.5-Math**, synthetic code는 **Qwen2.5-Coder**가 맡았고, instance-level mix weight는 proxy-model ablation으로 선택되었다. Family는 **dense 0.6B–32B and MoE 30B-A3B / 235B-A22B**를 포함하며, 모두 GQA + SwiGLU + RoPE + RMSNorm + stability를 위한 **QK-Norm**을 사용한다. 그 reasoning-stage pre-install 덕분에 post-training pipeline은 long-CoT SFT를 ground-up teach가 아니라 *cold start*로 취급할 수 있다. Reasoning prior가 이미 base weights에 들어 있기 때문이다.

**Hybrid-thinking data-labeling format (Qwen 3 fusion stage에서 쓰인 그대로).** 모든 training example은 채워져 있거나 명시적으로 비어 있는 `<think>...</think>` block을 가진다.

```
<|im_start|>user
{prompt} /think
<|im_end|>
<|im_start|>assistant
<think>
{long chain-of-thought reasoning, multi-step, can use pseudo-code and self-checks}
</think>
{final answer}
<|im_end|>
```

Non-thinking branch의 data는 다음과 같다.

```
<|im_start|>user
{prompt} /no_think
<|im_end|>
<|im_start|>assistant
<think>

</think>
{direct answer}
<|im_end|>
```

빈 `<think>` block은 장식이 아니다. Prompt가 `/no_think`를 carry할 때 empty reasoning trace를 emit하도록 모델을 가르치는 **signal**이다. Inference에서는 user가 prompt에 `/think` 또는 `/no_think`를 포함하여 mode를 toggle한다. Thinking-budget interface는 모델이 answer에 commit하기 전에 `<think>` 안에서 사용할 수 있는 token 수를 노출한다.

이 behaviour를 만드는 four-stage post-training pipeline:

1. **Long-CoT cold-start SFT** on math + code reasoning traces(teacher = internal Qwen reasoning model 또는 frontier distillation).
2. **Reasoning-focused RL** on math / code / STEM — verifiable rewards. Algorithm은 Qwen 2.5의 GRPO를 상속한다.
3. **Thinking-mode fusion SFT** — 위 format으로 thinking과 non-thinking example을 한 pass에 섞어 모델이 `/think` vs `/no_think` contract를 배우게 한다.
4. **General-domain RL** — chat / safety / tool-use preference tuning.

이 ordering은 load-bearing이다. Stage 1은 강한 reasoning trajectory(`<think>` template과 long-CoT style)를 세운다. Stage 2는 RL로 그 trajectory를 *reinforce*한다. Fusion 전에 이를 수행하면 Stage 3의 non-thinking data가 reasoning prior를 erode하지 못한다. Stage 3는 reasoning을 다시 training하지 않고 mode contract를 도입한다. 모델은 `/no_think` prompt에서 reasoning habit을 suppress하면서 `/think`에서는 보존하는 법을 배운다. Stage 4는 두 branch의 reasoning을 건드리지 않고 chat behaviour를 polish한다.

작은 Qwen 3 model(0.6B–14B dense, MoE-A3B variants)은 **strong-to-weak distillation**을 사용한다. Off-policy(teacher trajectory를 SFT target으로 사용)와 on-policy(student rollout을 teacher-derived RM으로 scoring) 둘 다 쓴다. Report는 작은 scale에서는 distillation이 RL보다 quality와 efficiency 모두에서 우수하다고 명시한다.

Qwen 3.5의 2026년 2-4월 refresh(raw-data: [[qwen-3-5]])는 397B-A17B MoE와 1M context로 scale하지만 **Qwen 3 수준의 상세한 새 post-training tech report를 공개하지 않는다**. Recipe는 상속된 것으로 추정되며, differentiation은 SFT algorithmics가 아니라 serving efficiency(19× decoding speedup)로 이동했다.

Four-stage pipeline과 fusion-SFT example format은 [[excerpts/qwen-3-hybrid-thinking]]을 보라.

---

## 4. OLMo 2 vs OLMo 3 — open-stage view

두 OLMo report는 SFT recipe가 흥미로운 axis가 아니며, **stage transparency가 흥미로운 axis**라는 것을 가장 깔끔하게 보여준다.

**OLMo 2**(2025)는 Tülu 3 recipe를 *wholesale*로 채택한다. OLMo-variant의 ~939K-prompt Tülu 3 SFT mix로 SFT하고, on-policy preference + Tülu 3 preference pool로 DPO한 뒤, Tülu 3의 hyperparameter(**LR 3e-7, β_KL 0.05, clip ε 0.2, GAE λ 0.95, 4 PPO epochs/step**)를 사용해 **PPO-RLVR**을 수행한다. OLMo 2의 흥미로운 기여는 architecture이다. RMSNorm + **reordered norm**(residual 내부 post-norm) + **QK-Norm**(attention 전 query와 key normalize) + output logits에 대한 **Z-loss** regularizer. 이 stability trick 덕분에 Tülu recipe가 non-Llama base 위에 unchanged로 port될 수 있었다. Spike-prone이었던 OLMo 1 training이었다면 훨씬 더 conservative한 mix가 필요했을 것이다.

**OLMo 3**(2025, Dec report)는 *flow*를 first-class artifact로 끌어올린다. 하나의 base checkpoint에서 네 branch가 publish된다.

- **Base** — pretrained checkpoint.
- **Think** — thinking-specific SFT → DPO → RLVR.
- **Instruct** — general SFT → DPO → RLVR.
- **RL Zero** — Base에서 직접 RLVR, SFT 없음, DPO 없음(따라서 SFT priming 없이 pure RL을 study할 수 있다).

Data curriculum은 모든 stage를 이름 붙인다. **Dolma 3**(9.3T source) → **Dolma 3 Mix**(5.9T pretrain) → **Dolmino**(100B mid-training, math/code/science/IF/reading-comp) → **Longmino**(50B long-context) → **Dolci**(SFT / DPO / RLVR를 위한 separate mix가 있는 post-training suite). Learner에게 OLMo 3가 유난히 가치 있는 이유는 *branch를 diff하여 capability를 특정 stage에 attribute할 수 있기 때문*이다. Think branch가 MATH에서 Instruct branch보다 8 pp 높지만 IFEval에서 2 pp 낮다면, thinking-SFT mix가 IF를 reasoning과 trade했다는 것을 알 수 있다. Branch diff가 claim을 직접 checkable하게 만든다.

Infrastructure number는 OLMo 3 claim의 나머지 절반이다. SFT를 Open Instruct에서 Olmo Core로 옮겨 **8× throughput**을 얻었고, in-flight weight update + continuous batching으로 RL training이 **~4×** 더 efficient해졌다고 보고한다. 공개된 GPU budget은 useful reality check이다. OLMo 3 pretraining은 **최대 1,024 H100s**, mid-training은 **128 H100s**, post-training은 **256 H100s**를 사용했다. 이 ratio는 post-training compute가 pretraining에 비해 single-digit-percent line item임을 말한다. 이는 OLMo 2가 별도로 보고한 7B pretraining **~460K H100-hours**와 13B pretraining **~1.9M H100-hours**와도 일관된다. 이것은 AllenAI worldview([[allen-ai]])의 operational evidence이다. Openness는 training *trajectory*에 적용되고, efficiency는 하나의 monolithic trainer가 아니라 *stage-specific infrastructure*에 적용된다.

OLMo 2 → OLMo 3 diff에서 미묘한 점 하나. Post-training *algorithm*은 사실상 동일하다(Tülu 3에서 상속한 SFT → DPO → RLVR). 하지만 data container가 rename되고 re-scope된다. **Dolma 1.7 → Dolma 3**, **OLMo-Mix-1124 → Dolma 3 Mix**, **Dolmino cooldown → Dolma 3 Dolmino mid-training**, 새 **Longmino**(639B-token pool에서 나온 50B long-context tokens)와 **Dolci**(post-training suite). Naming은 stage graph를 addressable하게 만든다. Learner가 "long-context ability는 어디서 왔나?"라고 물으면 답은 문자 그대로 "Longmino"이지 "pretraining run 어딘가"가 아니다. OLMo 3가 RL-Zero를 별도 branch로 release한 것도 또 하나의 도구이다. SFT+DPO를 그림에서 완전히 빼고 base checkpoint에서 verifiable-reward RL이 무엇을 할 수 있는지 관찰하게 해준다. 2025년 기준 DeepSeek-R1-Zero의 가장 깔끔한 public analogue이다.

[[excerpts/olmo-2-tulu-recipe]]와 [[excerpts/olmo-3-model-flow]]를 보라.

---

## 5. Phi 3/4 — pretrain/SFT boundary blurring

Phi line은 small model이 *textbook-quality synthetic* pretraining data를 사용하면 large-model performance에 도달한다는 [[phi-textbooks]] / [[phi-1-5]]의 claim에서 출발한다. Phi-3와 Phi-4는 이 claim을 scale하고 post-training으로 밀어 넣는다.

**Phi-3**는 **two-phase pretraining curriculum**을 사용한다. Phase 1은 majority tokens이며 heavily filtered web이다. Phase 2는 GPT-3.5 / GPT-4 class teacher가 생성한 synthetic textbook-like content가 주를 이룬다. SFT boundary에 대한 consequence는 다음과 같다. 다른 lab이 "SFT-shaped data"(예: reasoning step-by-step, instruction-response pairs)라고 부를 대부분이 Phase 2 pretraining 중 이미 흡수되었다. 명시적 SFT stage는 그다음 *refinement*가 된다. Publicly itemized되지 않은 curated synthetic + human-written instruction-response pairs로 구성되며, 이어서 **dedicated responsible-AI / safety preference slice**를 가진 DPO가 수행된다. Phi-3-mini(3.8B)는 3.3T pretraining token이 synthetic generation으로 target chat distribution에 *pre-aligned*되었기 때문에 일부 benchmark에서 Mixtral 8x7B parity에 도달한다.

**Phi-4** base(14B, Dec 2024)는 이 pattern을 더 날카롭게 만든다. 공개된 synthetic scale은 **50 categories에 걸친 ~400B unweighted tokens**이며, *pretraining과 post-training 모두*에 injection된다. Novel post-training knob은 **pivotal-token DPO**이다. Sequence level(chosen vs rejected complete responses)에서 preference pair를 만드는 대신, Phi-4는 **final-answer correctness의 probability가 가장 많이 바뀌는 token**을 식별하고, 그 pivotal token 주변에 preference pair를 구성한다. 이는 causal position을 target하는 DPO reformulation이다. Sequence-level DPO보다 더 surgical한 preference signal이다.

**Phi-4-reasoning**(Apr 2025)은 SFT dominance에 대한 이 장의 가장 깔끔한 evidence이다.

- "Boundary of base-model capability"에 맞게 filter된 **1.4M prompts**(너무 쉽지 않음: RM-trivial; 너무 어렵지 않음: RM-impossible).
- **o3-mini in high-thinking mode at 32K context**가 생성한 long-CoT trace **~16B SFT tokens (~8.3B unique)**.
- Domains: STEM, coding, safety.
- 그다음 reward를 구성하여 **GRPO for 90 training steps**를 수행한다. Reward는 +1 correct / −0.5 incorrect, **length-aware bonus**(concise correct answer를 reward하고, model이 틀릴 가능성이 높을 때 longer CoT를 허용 — "think longer when unsure"), **missing EOS 또는 unclosed `<think>` block penalty**, 그리고 **n=5 n-gram repetition penalty**로 구성된다.
- 보고된 effect: **90 GRPO steps만으로 +10% AIME**. 추가 step은 거의 yield하지 않는다. o3-mini trace quality가 정한 SFT ceiling이 final score를 지배한다.

[[deepseek-r1]]과 대비하라. R1은 weaker SFT base에서 long RL을 실행한다(RL이 heavy lifting). Phi-4-reasoning은 heavily-curated SFT base에서 *short* RL을 실행한다(SFT가 heavy lifting). Trade-off는 compute이다. Budget을 teacher-API SFT generation에 쓸 수도 있고 long RL rollout에 쓸 수도 있다. 하지만 특정 AIME level에 도달하기 위한 *total* compute는 유사하다. Phi line은 teacher quality ceiling(o3-mini in high-thinking)이 short-RL을 경제적으로 만들 만큼 높기 때문에 그 trade-off의 SFT 쪽을 택한다.

[[excerpts/phi-3-synthetic-sft]]와 [[excerpts/phi-4-reasoning-sft-rl]]를 보라.

---

## 6. Design-axis map

Opening의 세 axis로 여섯 lab을 끌어오면 다음과 같다.

| | SFT boundary | Modes per checkpoint | Pipeline stabilizer |
|---|---|---|---|
| Qwen 2.5 | Distinct SFT (two-stage context curriculum) | 1 | Online Merging Optimizer (DPO) + variance-ordered GRPO |
| Qwen 3 | Cold-start SFT + fusion-SFT | **2** (`/think` + `/no_think`) | Strong-to-weak distillation for small models |
| OLMo 2 | Distinct SFT (Tülu mix) | 1 | QK-Norm + Z-loss + reordered norm (architectural) |
| OLMo 3 | Distinct SFT per branch | **branch family** (Base / Think / Instruct / RL-Zero) | Olmo Core 8× SFT + 4× RL infra |
| Phi-3 | Blurred into Phase-2 pretraining | 1 | Safety-dedicated DPO slice |
| Phi-4-reasoning | Distinct SFT (1.4M o3-mini traces) | 1 (think-only) | Length-aware GRPO reward + n-gram repetition penalty |

이 table은 세 가지 distinct *design stance*로 읽힌다. **Qwen**은 classical SFT → DPO → RL skeleton을 유지하되 각 stage 내부에 stabilizer를 추가한다(SFT에는 two-phase, DPO에는 OMO, GRPO에는 variance-ordering). **AllenAI / OLMo**는 Tülu recipe를 unchanged로 유지하고 innovation을 transparency + infrastructure로 옮긴다. **Microsoft / Phi**는 data budget을 upstream으로 reallocate한다. Synthetic pretraining이 다른 lab이 SFT라고 부르는 것을 흡수하고, 명시적 SFT stage는 refinement(Phi-3/4) 또는 targeted long-CoT cold start(Phi-4-reasoning)가 된다. Ch-34 이후 읽게 될 모든 2026 case study([[nemotron]], distillation SFT, reasoning-first small models)는 이 세 stance 중 하나의 variant이다.

---

## Connections

- **ch-32 (Tülu 3 SFT mix design)** — OLMo 2/3가 상속하는 recipe. Qwen과 Phi가 서로 다른 방향으로 deform하는 baseline이다.
- **ch-33 (Case Studies A: Tülu 3 + Llama 3)** — canonical form. Ch-34는 deformation survey이다.
- **ch-35 (next: Nemotron + distillation SFT)** — 340B scale에서 >98% synthetic SFT로 Phi 쪽 stance table을 확장한다.
- **ch-36 (Lab: Packed SFT Run)** — 여기서 배운 SFT-boundary choice를 적용한다. Packed-vs-unpacked ablation은 Qwen-style SFT stabilizer의 miniature이다.
- **ch-42 / ch-44 (RL track)** — GRPO vs PPO-RLVR. 여기의 Qwen/Phi-4 vs OLMo split은 그곳의 lane split과 같다.
- **ch-47 / ch-48 (Eval harness + contamination)** — Phi line의 recurring contamination critique([[phi-textbooks]], [[phi-1-5]])는 contamination gate가 왜 중요한지에 대한 running example이다.

## Further reading

- [[qwen-2.5]] — Qwen2.5 tech report; 1M / 150K / two-phase curriculum은 §Post-training을 읽어라.
- [[qwen-3]] — Qwen3 tech report; four-stage hybrid-thinking pipeline과 strong-to-weak distillation claim은 §Post-training을 읽어라.
- [[olmo-2]] — OLMo 2 report; Tülu 3 inheritance와 RLVR hyperparameters는 §Post-training을 읽어라.
- [[olmo-3]] — OLMo 3 model-flow report; branch taxonomy는 §Family structure와 §Post-training을 읽어라.
- [[phi-3]] — Phi-3 report; two-phase pretraining diagram이 headline이다.
- [[phi-4]] — Phi-4 + Phi-4-reasoning; §Phi-4-reasoning SFT(1.4M prompts, o3-mini traces)와 GRPO reward schematic을 읽어라.
- [[alibaba-qwen]] / [[allen-ai]] — lab-level context. Qwen과 AllenAI가 왜 서로 다른 axis에서 differentiate하는지 설명한다.

## Companion visualization

**[figures/lab-compare.html](figures/lab-compare.html)** — 여섯 lab을 여섯 axis에서 비교하는 interactive radar / spider chart: **SFT-scale**(log tokens), **synthetic-%**(SFT+pretraining data 중 teacher-generated fraction), **SFT learning-rate tier**, **multi-round**(distinct SFT pass + DPO round 수), **long-context**(max SFT sequence length), **hybrid-thinking**(하나의 checkpoint가 `/think` + `/no_think`를 carry하는지). Lab을 toggle할 수 있고, 각 axis에 hover하면 lab별 attested source number를 볼 수 있다. 위 stance-table을 시각적으로 읽는 데 사용하라. Phi의 radar profile은 synthetic-% 쪽으로 강하게 skew되고, Qwen 3는 hybrid-thinking 쪽으로, OLMo 3는 multi-round + long-context 쪽으로, OLMo 2는 "Tülu-baseline" reference shape이다.
