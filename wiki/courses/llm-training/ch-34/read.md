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

# Chapter 34 — Case Studies B: Qwen 2.5/3, OLMo 2/3, Phi 3/4

> **Core insight.** Ch-33 showed Tülu 3 and Llama 3 — two *explicit* recipes that stage SFT, preference tuning, and RL as distinct named boxes. Ch-34 shows three labs that each *deform* that template in a different load-bearing way. Qwen collapses the post-training boundary by making SFT a **two-phase context curriculum** and then adding a stabilizer (Online Merging Optimizer) to DPO; Qwen 3 goes further and **trains one model to carry two modes** — `/think` and `/no_think` — via a prompt-level data-labeling convention. OLMo 2 and OLMo 3 keep the recipe identical but make the **stages themselves the artifact**: OLMo 3's model-flow is Base → Think/Instruct/RL-Zero branches, so you can read the capability additions off the branch diff. Phi 3/4 blur the boundary from the other end — synthetic-heavy pretraining pushes so much "SFT-shaped" data upstream that post-training SFT becomes a *refinement* stage, and Phi-4-reasoning's 1.4M long-CoT traces demonstrate that the SFT ceiling dominates: 90 GRPO steps add the last +10% AIME on top.
>
> **Guideline.** When you read a 2025-era model report, do not grade it on "did they do SFT + DPO + RL." Grade it on three orthogonal axes: (i) **where is the SFT boundary** — pushed upstream into synthetic pretraining (Phi), held as a distinct long-CoT cold start (Qwen 3), or held as a Tülu-identical stage (OLMo)? (ii) **how many modes** does one checkpoint carry — one (Phi, OLMo 2), two (Qwen 3 hybrid), or a family-branch (OLMo 3)? (iii) **what stabilized the pipeline** — Online Merging Optimizer (Qwen 2.5), QK-Norm + Z-loss architecture (OLMo 2), pivotal-token DPO (Phi-4), or length-aware GRPO reward (Phi-4-reasoning)? Every recipe below is the same three-stage skeleton; the differentiators are at these three axes.

---

## Why this chapter exists

Ch-33 is the *canonical* case: two labs that published their recipe in the clearest possible form. The value of ch-34 is that each lab here breaks the canonical form in a way that later recipes (Nemotron, distillation, reasoning-first) will inherit. Qwen 2.5 is the origin of the two-stage SFT context curriculum; Qwen 3 is the origin of the `/think` toggle that every 2026 hybrid model now carries; OLMo 3 is the origin of the model-flow-as-artifact stance; Phi-4-reasoning is the cleanest evidence that SFT dominates RL when the SFT data is o3-mini-generated long CoT. Reading these four as deformations of one template is how you extract the design space rather than memorizing six recipes.

---

## 1. The six-lab comparison table

All numbers are from the attested raw-data sources under `wiki/raw-data/llm-training/`. Entries marked *nd* are explicitly not disclosed in the public report; inferred defaults are labelled `infer:` with the nearest cousin.

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

Read this table column by column. The **LR column** says almost nothing distinguishes these labs at SFT (the disclosed DPO LR 7e-7 at Qwen 2.5 is the one precise anchor; Tülu 3 / OLMo RLVR's 3e-7 is the other). The **optimizer column** is where Qwen's Online Merging Optimizer and OLMo 3's Olmo Core throughput win isolate themselves. The **mix column** is where Phi deforms the template entirely — most of what other labs call SFT data, Phi calls Phase-2 pretraining. And the **RL column** is where the deformation becomes a pattern: OLMo does PPO-RLVR, Qwen and Phi-4-reasoning do GRPO, and DPO sits between them as the shared preference stage.

Two properties of the table are worth calling out before walking each lab:

- **The nd / withheld pattern is informative.** Phi reports aggregate synthetic volume (400B unweighted tokens) but withholds SFT count and DPO β. Qwen reports DPO LR + optimizer but withholds GRPO group size. OLMo publishes the most numbers overall but only because it inherits Tülu 3 defaults. The *shape* of what each lab hides is the shape of what they consider their secret sauce.
- **"SFT size" is not one number.** Qwen 2.5's 1M counts SFT+DPO+GRPO prompts together; OLMo 2's 939K is SFT-only; Phi-4-reasoning's 1.4M is prompts before the ~16B-token o3-mini expansion. Before you compare any two rows on "SFT size", normalize — the comparison across rows is meaningful only if you re-count under a common definition, usually *unique SFT tokens*.

### 1.1 A second view: the pretraining-scale lens

Post-training is the chapter's focus, but the SFT-boundary axis only makes sense once you know how much data was absorbed *before* SFT. Pull the same six labs through pretraining scale and context-extension strategy:

| Lab / model | Pretrain tokens | Native ctx → extended | Mid-training / cooldown | Notable architectural stability |
|---|---|---|---|---|
| Qwen 2.5 (72B) | **18T** (up from Qwen 2's 7T) | 4K → 128K (1M via YARN extrapolation) | none separately named | GQA + SwiGLU + RoPE + RMSNorm; tied embeddings small sizes |
| Qwen 3 (235B-A22B) | **36T across 119 languages** — general stage >30T @ 4096, reasoning stage ~5T @ 4096, long-context stage hundreds of billions @ 32768 | 4K → 128K (YARN + Dual Chunk Attention) | dedicated reasoning stage inside pretraining | **QK-Norm** added for stability |
| OLMo 2 (7B/13B/32B) | **~3.9T** (Stage 1 OLMo-Mix-1124) + **~50B** Dolmino cooldown | 4K → 32K in cooldown | Dolmino higher-quality mix | RMSNorm + reordered norm + **QK-Norm** + Z-loss + init preserving activation scale |
| OLMo 3 (7B/32B) | Dolma 3: **9.3T** source → **5.9T** pretrain mix → **100B** Dolmino mid-training (sampled from 2.2T pool) → **50B** Longmino (from 639B long-doc pool) | per-branch long-context via Longmino | Dolmino + Longmino named stages | inherits OLMo 2 stability tricks |
| Phi-3 (mini/small/medium) | **3.3T (mini) / 4.8T (small) / 4.8T (medium)** | 4K native (128K via LongRope on mini; small = 8K; medium = 4K) | two *phases* inside pretraining — Phase 1 web-heavy, Phase 2 synthetic-heavy | conventional; data-first, not architecture-first |
| Phi-4 (14B) | not re-disclosed at Phi-3 granularity; **~400B unweighted synthetic tokens across 50 categories** injected into pretraining and post-training | 16K extended | synthetic data phase is the "mid-training" analogue | conventional |

Three observations worth walking out. First, **the pretraining-token column ranges 10× across the six labs** (3.3T for Phi-3-mini to 36T for Qwen 3), yet the SFT volumes cluster inside ~3× (939K to 1.4M prompts). Post-training is the step that has converged; pretraining is still where the labs disagree about how much data the recipe needs. Second, **Qwen 3 is the only lab that names a "reasoning stage" inside pretraining** (~5T tokens) — this is where it pre-installs the reasoning prior that the long-CoT cold-start SFT then stylizes. OLMo 3 does the same conceptually via Dolmino (100B math/code/science/IF/reading-comp tokens) but at a much smaller relative scale. Third, **OLMo 2 and Qwen 3 independently arrived at QK-Norm** as the stability fix for non-Llama bases — a cross-lab convergence worth noting because ch-33's Llama 3 does not need it (its base was tuned over multiple generations).

### 1.2 What this means for a practitioner in 2025

If you are building on top of these recipes in 2025, the table above collapses to a short decision tree. **(a) Will you carry one mode or two?** If one: follow OLMo 2 (Tülu 3 inheritance, single checkpoint, stable defaults) — the path of least resistance. If two: you must adopt the Qwen 3 `/think` + `/no_think` data-labeling contract from Stage 3 fusion SFT, because no cheaper mechanism for mode-gating has been published. **(b) Do you have API budget for a frontier teacher?** If yes: follow Phi-4-reasoning — spend your budget on 1M+ long-CoT traces from o3-mini-class and cap RL at ~100 GRPO steps. If no: follow OLMo 2/3's RLVR-with-verifiable-rewards path where the reward signal is a math verifier or code unit test, not a trained RM. **(c) Are you pretraining from scratch?** If yes: Phi's Phase-1/Phase-2 synthetic-heavy curriculum is the only recipe among these six that delivered sub-10B parity with mid-size models; the trade-off is total teacher-API cost. If no (continued pretraining or post-training only): the Qwen two-phase SFT context curriculum is the smallest-overhead way to add long-context instruction following to a base that was pretrained at 4K-32K.

---

## 2. Qwen 2.5 — the two-stage SFT context curriculum

Qwen 2.5 disclosed three things that matter:

1. **Context curriculum inside SFT.** Not "train on short first, extend later" at pretraining — that is separate — but *within the SFT run*: **Phase 1** is short instructions capped at 32,768 tokens, **Phase 2** mixes short (≤32K) with long (up to 262,144) instructions. The claim is that the short phase locks in instruction-following quality, and the mixed phase teaches long-context instruction following *without regressing* on the short tasks. Without the mix — i.e., if you only trained long — short-task quality collapses. This is the SFT analogue of the long-context pretraining two-stage that ProLong and Llama-3 use, but moved one step later in the pipeline.
2. **Online Merging Optimizer at DPO.** Standard DPO with LR 7e-7 and 1 epoch is already tight, but Qwen keeps a **running merged checkpoint** during the DPO pass — effectively a time-averaged policy that stabilizes the gradient against the single-epoch-with-small-preference-set regime. The raw-data source lists "β not explicitly reported in the public tech report text (standard DPO beta ~0.1 assumed)" — so the only attested DPO numbers are LR 7e-7 and 1 epoch.
3. **Variance-ordered GRPO prompts.** After DPO, Qwen runs GRPO. The novel knob: **order the prompts by the variance of their response scores under the reward model**. High-variance prompts — where the RM discriminates strongly between good and bad rollouts — are trained first. Intuition: low-variance prompts are either too easy (RM scores all rollouts similarly high) or too hard (all low), and GRPO's advantage estimator has nothing to work with.

The Qwen 2.5 recipe is therefore: pretraining (18T tokens, up from Qwen 2's 7T) → **SFT two-phase context curriculum on 1M examples** → **DPO (150K pairs, LR 7e-7, OMO, 1 epoch, β≈0.1 assumed)** → **variance-ordered GRPO** with a Qwen-architecture RM (policy-matched with linear head). This is the canonical "SFT + DPO + GRPO stacked in order" recipe other labs cite. The disclosed Qwen2.5-72B-Instruct scores that anchor the whole chapter — MMLU 86.1 / HumanEval 85.4 / MATH 83.1 / IFEval 86.1 — are the numbers OLMo 2's 32B and Phi-4-reasoning's 14B are trying to close on with far less pretraining data.

**A note on what "Phase 2 mixing prevents regression" means mechanically.** If you train SFT only on long instructions, the gradient on short-instruction distribution becomes stale — the model's attention patterns for 2K-token inputs are not refreshed while it learns to attend across 100K tokens. The mixed Phase 2 keeps both distributions in-batch, so attention heads that specialize in short-range don't drift. This is the same logic as "always include short data in long-context continued pretraining" (ProLong / [[long-context-llama3]]), applied inside SFT.

See [[excerpts/qwen-2.5-post-training]] for the full preference-data-construction pipeline.

---

## 3. Qwen 3 — hybrid thinking as a data-labeling problem

Qwen 3's innovation is organizational, not algorithmic. The same-class GRPO and DPO primitives, but **one model family carries two modes**: `/think` for long-CoT deliberation and `/no_think` for instant response. The only way this works is if the SFT data *explicitly labels* which mode each example belongs to.

Before unpacking the hybrid-thinking contract, it helps to know what Qwen 3 paid for upstream. Pretraining is **36T tokens across 119 languages and dialects** — roughly 2× Qwen 2.5's 18T — with a **three-stage curriculum**: a general stage of >30T tokens at 4096 sequence length, a **reasoning stage of ~5T higher-quality STEM/coding/synthetic tokens** still at 4096, and a long-context stage of "hundreds of billions" of tokens at 32768. Data expansion leaned on sibling models as synthetic generators — OCR-style extraction from PDFs via **Qwen2.5-VL**, synthetic math from **Qwen2.5-Math**, synthetic code from **Qwen2.5-Coder** — with instance-level mix weights chosen by proxy-model ablation. The family spans **dense 0.6B–32B and MoE 30B-A3B / 235B-A22B**, all using GQA + SwiGLU + RoPE + RMSNorm + **QK-Norm** for stability. That reasoning-stage pre-install is the reason the post-training pipeline can treat long-CoT SFT as a *cold start* rather than a ground-up teach; the reasoning prior is already in the base weights.

**The hybrid-thinking data-labeling format (verbatim, as used in the Qwen 3 fusion stage).** Every training example carries a `<think>...</think>` block that is either filled or explicitly empty:

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

For the non-thinking branch, the data looks like:

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

The empty `<think>` block is not decorative — it is **the signal** that teaches the model to emit an empty reasoning trace when the prompt carries `/no_think`. At inference, the user toggles the mode by including `/think` or `/no_think` in the prompt; the thinking-budget interface exposes how many tokens the model is allowed to spend inside `<think>` before being forced to commit to an answer.

The four-stage post-training pipeline that produces this behaviour:

1. **Long-CoT cold-start SFT** on math + code reasoning traces (teacher = internal Qwen reasoning model or frontier distillation).
2. **Reasoning-focused RL** on math / code / STEM — verifiable rewards. Algorithm inherits Qwen 2.5's GRPO.
3. **Thinking-mode fusion SFT** — mix thinking and non-thinking examples in one pass using the format above, so the model learns the `/think` vs `/no_think` contract.
4. **General-domain RL** — chat / safety / tool-use preference tuning.

The ordering is load-bearing. Stage 1 establishes a strong reasoning trajectory (the `<think>` template and the long-CoT style). Stage 2 *reinforces* that trajectory via RL — doing it before fusion prevents the non-thinking data in Stage 3 from eroding the reasoning prior. Stage 3 introduces the mode contract without re-training reasoning; the model learns to suppress its reasoning habit on `/no_think` prompts while preserving it for `/think`. Stage 4 then polishes chat behaviour without touching reasoning on either branch.

Small Qwen 3 models (0.6B–14B dense, MoE-A3B variants) use **strong-to-weak distillation** — both off-policy (teacher trajectories as SFT targets) and on-policy (student rollouts scored by teacher-derived RM) — because the report explicitly states distillation outperforms RL on those scales in both quality and efficiency.

Qwen 3.5's Feb–April 2026 refresh (raw-data: [[qwen-3-5]]) scales to 397B-A17B MoE and 1M context but **does not publish a new post-training tech report at Qwen 3 detail level** — the recipe is presumed inherited, with differentiation pushed into serving efficiency (19× decoding speedup) rather than SFT algorithmics.

See [[excerpts/qwen-3-hybrid-thinking]] for the four-stage pipeline and the fusion-SFT example format.

---

## 4. OLMo 2 vs OLMo 3 — the open-stage view

The two OLMo reports are the cleanest demonstration that the SFT recipe is not the interesting axis; **stage transparency is**.

**OLMo 2** (2025) adopts the Tülu 3 recipe *wholesale*: SFT on OLMo-variant of the ~939K-prompt Tülu 3 SFT mix, DPO on on-policy preferences + Tülu 3 preference pool, then **PPO-RLVR** with Tülu 3's hyperparameters (**LR 3e-7, β_KL 0.05, clip ε 0.2, GAE λ 0.95, 4 PPO epochs/step**). The interesting OLMo 2 contribution is architectural: RMSNorm + **reordered norm** (post-norm inside the residual) + **QK-Norm** (normalize queries and keys before attention) + **Z-loss** regularizer on output logits. These stability tricks are what let the Tülu recipe port unchanged onto a non-Llama base — OLMo 1's spike-prone training would have needed a much more conservative mix.

**OLMo 3** (2025, Dec report) promotes the *flow* to the first-class artifact. From one base checkpoint, four branches are published:

- **Base** — the pretrained checkpoint.
- **Think** — thinking-specific SFT → DPO → RLVR.
- **Instruct** — general SFT → DPO → RLVR.
- **RL Zero** — RLVR directly from Base, no SFT, no DPO (so you can study pure RL without the SFT priming).

The data curriculum names every stage: **Dolma 3** (9.3T source) → **Dolma 3 Mix** (5.9T pretrain) → **Dolmino** (100B mid-training, math/code/science/IF/reading-comp) → **Longmino** (50B long-context) → **Dolci** (post-training suite with separate SFT / DPO / RLVR mixes). For a learner, OLMo 3 is unusually valuable because *you can attribute a capability to a specific stage by diffing the branches*. If the Think branch beats the Instruct branch on MATH by 8 pp but loses on IFEval by 2 pp, you know the thinking-SFT mix traded IF for reasoning — the branch diff makes the claim directly checkable.

The infrastructure numbers are the other half of the OLMo 3 claim: moving SFT from Open Instruct to Olmo Core reportedly **8×'d throughput**, and in-flight weight updates + continuous batching made RL training **~4×** more efficient. The publicly disclosed GPU budget is a useful reality check: OLMo 3 pretraining used **up to 1,024 H100s**, mid-training **128 H100s**, and post-training **256 H100s** — a ratio that says post-training compute is a single-digit-percent line item compared to pretraining, consistent with OLMo 2's separately reported **~460K H100-hours at 7B** and **~1.9M H100-hours at 13B** for pretraining alone. This is the operational evidence for the AllenAI worldview ([[allen-ai]]): openness applies to training *trajectories*, and efficiency applies to *stage-specific infrastructure*, not to a single monolithic trainer.

One subtle point on the OLMo 2 → OLMo 3 diff: the post-training *algorithm* is essentially identical (SFT → DPO → RLVR inherited from Tülu 3), but the data containers are renamed and re-scoped — **Dolma 1.7 → Dolma 3**, **OLMo-Mix-1124 → Dolma 3 Mix**, **Dolmino cooldown → Dolma 3 Dolmino mid-training**, plus the new **Longmino** (50B long-context tokens drawn from a 639B-token pool) and **Dolci** (post-training suite). The naming makes the stage graph addressable: when a learner asks "where did the long-context ability come from?" the answer is literally "Longmino," not "the pretraining run somewhere." OLMo 3's release of RL-Zero as a separate branch is the other tool — it lets you subtract SFT+DPO from the picture entirely and observe what verifiable-reward RL can do directly on a base checkpoint, which in 2025 is the cleanest public analogue of DeepSeek-R1-Zero.

See [[excerpts/olmo-2-tulu-recipe]] and [[excerpts/olmo-3-model-flow]].

---

## 5. Phi 3/4 — pretrain/SFT boundary blurring

The Phi line starts from [[phi-textbooks]] / [[phi-1-5]]'s claim that small models reach large-model performance when pretraining data is *textbook-quality synthetic*. Phi-3 and Phi-4 scale that claim and push it into post-training.

**Phi-3** uses a **two-phase pretraining curriculum**: Phase 1 is majority tokens, heavily filtered web; Phase 2 is primarily synthetic textbook-like content generated by GPT-3.5 / GPT-4 class teachers. The consequence for the SFT boundary: most of what other labs would call "SFT-shaped data" (e.g. reasoning step-by-step, instruction-response pairs) has already been absorbed during Phase 2 pretraining. The explicit SFT stage then becomes a *refinement* — curated synthetic + human-written instruction-response pairs, sizes not publicly itemized — followed by DPO with a **dedicated responsible-AI / safety preference slice**. Phi-3-mini (3.8B) reaches Mixtral 8x7B parity on some benchmarks because the 3.3T pretraining tokens have been *pre-aligned* to the target chat distribution via synthetic generation.

**Phi-4** base (14B, Dec 2024) sharpens the pattern. The disclosed synthetic scale is **~400B unweighted tokens across 50 categories**, injected into *both* pretraining and post-training. The novel post-training knob is **pivotal-token DPO**: instead of constructing preference pairs at the sequence level (chosen vs rejected complete responses), Phi-4 identifies **tokens where the probability of final-answer correctness changes most**, and constructs preference pairs centered on those pivotal tokens. This is a DPO reformulation that targets the causal positions — a more surgical preference signal than sequence-level DPO.

**Phi-4-reasoning** (Apr 2025) is the chapter's cleanest evidence for SFT dominance:

- **1.4M prompts** filtered to the "boundary of base-model capability" (not too easy: RM-trivial; not too hard: RM-impossible).
- **~16B SFT tokens (~8.3B unique)** of long-CoT traces generated by **o3-mini in high-thinking mode at 32K context**.
- Domains: STEM, coding, safety.
- Then **GRPO for 90 training steps** with a reward composed of: +1 correct / −0.5 incorrect, a **length-aware bonus** (concise correct answers rewarded; longer CoT allowed when the model is likely wrong — "think longer when unsure"), **penalty for missing EOS or unclosed `<think>` block**, and an **n=5 n-gram repetition penalty**.
- The reported effect: **+10% AIME from 90 GRPO steps**; further steps yield little. The SFT ceiling — set by o3-mini trace quality — dominates the final score.

Contrast with [[deepseek-r1]]: R1 runs long RL from a weaker SFT base (so RL does the heavy lifting). Phi-4-reasoning runs *short* RL from a heavily-curated SFT base (so SFT does the heavy lifting). The trade-off is compute: you can spend the budget on teacher-API SFT generation or on long RL rollouts, but the *total* compute to reach a given AIME level is similar; the Phi line picks the SFT side of that trade because the teacher quality ceiling (o3-mini in high-thinking) is high enough to make short-RL economical.

See [[excerpts/phi-3-synthetic-sft]] and [[excerpts/phi-4-reasoning-sft-rl]].

---

## 6. The design-axis map

Pulling the six labs through the three axes from the opening:

| | SFT boundary | Modes per checkpoint | Pipeline stabilizer |
|---|---|---|---|
| Qwen 2.5 | Distinct SFT (two-stage context curriculum) | 1 | Online Merging Optimizer (DPO) + variance-ordered GRPO |
| Qwen 3 | Cold-start SFT + fusion-SFT | **2** (`/think` + `/no_think`) | Strong-to-weak distillation for small models |
| OLMo 2 | Distinct SFT (Tülu mix) | 1 | QK-Norm + Z-loss + reordered norm (architectural) |
| OLMo 3 | Distinct SFT per branch | **branch family** (Base / Think / Instruct / RL-Zero) | Olmo Core 8× SFT + 4× RL infra |
| Phi-3 | Blurred into Phase-2 pretraining | 1 | Safety-dedicated DPO slice |
| Phi-4-reasoning | Distinct SFT (1.4M o3-mini traces) | 1 (think-only) | Length-aware GRPO reward + n-gram repetition penalty |

The table reads as three distinct *design stances*. **Qwen** keeps a classical SFT → DPO → RL skeleton but adds stabilizers inside each stage (OMO at DPO, variance-ordering at GRPO, two-phase at SFT). **AllenAI / OLMo** keep the Tülu recipe unchanged and move the innovation to transparency + infrastructure. **Microsoft / Phi** reallocate the data budget upstream — synthetic pretraining absorbs what others call SFT, and the explicit SFT stage becomes either a refinement (Phi-3/4) or a targeted long-CoT cold start (Phi-4-reasoning). Every 2026 case study you will read after ch-34 ([[nemotron]], distillation SFT, reasoning-first small models) is a variant of one of these three stances.

---

## Connections

- **ch-32 (Tülu 3 SFT mix design)** — the recipe OLMo 2/3 inherit; the baseline Qwen and Phi deform in different directions.
- **ch-33 (Case Studies A: Tülu 3 + Llama 3)** — canonical form; ch-34 is the deformation survey.
- **ch-35 (next: Nemotron + distillation SFT)** — extends the Phi side of the stance table with >98% synthetic SFT at 340B scale.
- **ch-36 (Lab: Packed SFT Run)** — applies the SFT-boundary choice you learned here; packed-vs-unpacked ablation is the Qwen-style SFT stabilizer in miniature.
- **ch-42 / ch-44 (RL track)** — GRPO vs PPO-RLVR: the Qwen/Phi-4 vs OLMo split here is the same lane split there.
- **ch-47 / ch-48 (Eval harness + contamination)** — the Phi line's recurring contamination critique ([[phi-textbooks]], [[phi-1-5]]) is the running example for why contamination gates matter.

## Further reading

- [[qwen-2.5]] — Qwen2.5 tech report; read §Post-training for the 1M / 150K / two-phase curriculum.
- [[qwen-3]] — Qwen3 tech report; read §Post-training for the four-stage hybrid-thinking pipeline and the strong-to-weak distillation claim.
- [[olmo-2]] — OLMo 2 report; read §Post-training for the Tülu 3 inheritance and the RLVR hyperparameters.
- [[olmo-3]] — OLMo 3 model-flow report; read §Family structure and §Post-training for the branch taxonomy.
- [[phi-3]] — Phi-3 report; the two-phase pretraining diagram is the headline.
- [[phi-4]] — Phi-4 + Phi-4-reasoning; read §Phi-4-reasoning SFT (1.4M prompts, o3-mini traces) and the GRPO reward schematic.
- [[alibaba-qwen]] / [[allen-ai]] — lab-level context; explains why Qwen and AllenAI differentiate along different axes.

## Companion visualization

**[figures/lab-compare.html](figures/lab-compare.html)** — interactive radar / spider chart comparing the six labs across six axes: **SFT-scale** (log tokens), **synthetic-%** (fraction of SFT+pretraining data that is teacher-generated), **SFT learning-rate tier**, **multi-round** (number of distinct SFT passes + DPO rounds), **long-context** (max SFT sequence length), and **hybrid-thinking** (whether one checkpoint carries `/think` + `/no_think`). Toggle labs on and off; hover each axis to see the attested source numbers per lab. Use it to read the stance-table above visually — Phi's radar profile is heavily skewed toward synthetic-%, Qwen 3's toward hybrid-thinking, OLMo 3's toward multi-round + long-context, and OLMo 2 is the "Tülu-baseline" reference shape.
