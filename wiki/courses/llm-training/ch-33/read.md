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

# Chapter 33 — Case Studies A: Tülu 3 and Llama 3

> **Core insight.** Tülu 3 and Llama 3 are the two fully-disclosed reference points for modern post-training, and they answer the same question with opposite topologies. Tülu 3 is a **three-stage linear pipeline** — SFT on a once-curated 939K-prompt mixture, then DPO, then RLVR — whose contribution is that the *prompt mixture is published row-by-row*. Llama 3 is a **six-round closed loop** — SFT → rejection sampling → DPO, repeated, with the best checkpoint from round *N-1* generating the SFT pool for round *N*. The Tülu pipeline's lesson is that *data composition* at the start of training is first-order; the Llama pipeline's lesson is that *data composition evolves with the policy* and stale preferences degrade the loop. Reading them side-by-side is the cleanest way to see what "modern SFT" actually means in 2024–2025.
>
> **Guideline.** If you are building from a strong open base and your eval suite is well-defined, copy Tülu 3: publish the mixture, decontaminate explicitly, hit SFT once at 2 epochs and move the gains into DPO and RLVR stages. If you are training from scratch and can afford human annotation per round, copy Llama 3: keep the reward model fresh, regenerate the SFT pool from the latest checkpoint every round, and never reuse stale preferences. Do not try to graft the two — Tülu's once-and-done discipline breaks if you start re-mining preferences mid-pipeline, and Llama's flywheel breaks if you lock the mix before the first round's evals come back.

---

## 1. Tülu 3 — the open reference recipe

Tülu 3 (Allen AI, Nov 2024, [[tulu-3]]) is built on the Llama 3.1 base family (8B / 70B / 405B) and is the most complete public disclosure of a modern post-training stack to date: every prompt, every completion, every preference pair, every verifier, every hyperparameter, every eval script is public. Its contribution is *reproducibility as a thesis*, not a novel algorithm — the only algorithmic novelty is the RLVR stage, which we cover in §2.3.

### 1.1 The 939K SFT mixture — composition table

Allen AI reports **939,344 prompts** across **18 component datasets**, split **57% public / 43% in-house synthetic** ([[tulu-3-sft-mix]], [[allenai-tulu-synth]]). The per-source breakdown, attested from the dataset card, is:

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

The same mixture can be viewed by *skill bucket* ([[allenai-tulu-sft-recipe]]): Chat/general 27%, Math 21%, Code 14%, Precise IF 11%, Safety 10%, Reasoning/knowledge 10%, Multilingual 7%. The per-bucket shares are a *tunable knob* in the recipe, not an accident — Ai2's data-construction procedure is **skill-mixture-first**: build a math-only mixture and train a math-only model, build a code-only mixture and train a code-only model, keep the per-skill mixtures that move *that* skill the most on isolated ablations, then merge, decontaminate, and downsample until the final aggregate is balanced ([[tulu-3-sft-mix]]).

### 1.2 Persona-driven prompt synthesis — the 43%

The synthetic fraction is not generic self-instruct output. It is built by a **persona-driven prompt factory** ([[allenai-tulu-synth]]): sample a persona from a pool (e.g. "a machine-learning researcher focused on neural networks"), combine it with a skill template ("create a coding problem", "write a precise-IF task"), feed the combined prompt to **GPT-4o** (general) or **Claude-3.5-Sonnet** (coding — best-in-class code responses) as the prompt *generator*. A separate pass then generates the completion. The persona step diversifies *what gets asked*, not just what gets answered, and is the public instantiation of the [[persona-hub]] design pattern at dataset scale.

The `Tülu 3 Persona MATH` / `Persona GSM` / `Persona Python` / `Persona Algebra` / `Persona IF` entries in the table above are the direct outputs of this factory; `WildJailbreak` is the red-team slice built with adversarial personas. The `Hardcoded` 240-row set is the identity-fixing slice ("who are you", "what can you do") that all instruct models need.

### 1.3 Decontamination — why the dataset card is not enough

Decontamination is *part of the recipe*, not an afterthought ([[allenai-tulu-sft-recipe]]). Ai2 runs two filters against every eval set (MMLU, GSM8K, MATH, IFEval, BBH, AlpacaEval, Arena-Hard, HumanEval):

- **8-gram overlap ≥ 50%** between any training row and any eval row → drop training row.
- **Embedding cosine similarity > 0.9** to any eval row → drop training row.

The paper then publishes the *surviving overlap rate per eval*, so downstream users can replicate and audit. The operational rule is stricter than most open datasets: Ai2 explicitly removes rows that overlap with more than **2% of any eval suite**, a constraint that kills a surprising number of Evol-CodeAlpaca rows overlapping HumanEval prompts ([[tulu-3-sft-mix]]).

### 1.4 SFT hyperparameters and ablation findings

The 8B / 70B SFT config ([[allenai-tulu-sft-recipe]]) is boring by design: max seq 4096, AdamW (0.9, 0.95), LR **5e-6 (8B) / 2e-6 (70B)**, linear schedule with 3% warmup, **2 epochs**, global batch 128, BF16, FSDP FULL_SHARD (8B) / HYBRID_SHARD (70B), sequence packing on, response-only loss, gradient checkpointing on, **NEFTune off** (measured neutral at 939K scale).

The attested skill-removal ablations are the most-cited numbers from the paper:

- Drop **Persona-MATH** → **GSM8K −15 pp**. The math share is load-bearing for GSM8K specifically.
- Drop all **code** sources → **HumanEval −12 pp**. No code data ≠ compositional transfer from prose.
- Drop **safety** slice (WildGuardMix + WildJailbreak) → capability evals barely move, but **WildJailbreak falls 98% → 52%**. Safety is cheap to preserve and expensive to reintroduce.
- **2 epochs > 1 epoch > 3 epochs** at this mix size; 3 epochs hurts IFEval.
- **NEFTune** gives ~0 at 939K (gain saturates); small gain at ≤ 100K. The NEFTune intuition is that noise regularizes small mixtures; it is redundant at Tülu's scale.
- **Packing**: 2.5× throughput, zero quality delta.

### 1.5 DPO and RLVR — where the gains move after SFT

Post-SFT, Tülu 3 layers two preference / RL stages ([[tulu-3]], [[allenai-tulu-blog]]):

- **DPO** on a ~270K-pair preference pool combining (a) off-policy pairs sourced from UltraFeedback + safety data, and (b) on-policy pairs generated by sampling from the *Tülu SFT* checkpoint and scoring with a trained reward model. Length-normalized DPO with **β=5.0** at 8B, **LR 5e-7**.
- **RLVR** — Reinforcement Learning with Verifiable Rewards, the one genuine algorithmic novelty ([[rlvr-tulu3]]). The scalar reward is a **deterministic verifier** `v(x, y) → {0, 1}`, not a learned RM: exact-match / SymPy equivalence for math; regex-style constraint check for IFEval; unit-test execution for code. PPO hparams: LR **3e-7**, **β_KL=0.05**, clip 0.2, **K=4 update epochs**, **GAE λ=0.95, γ=1.0**, **10M episodes**. The no-RM property sidesteps Goodhart drift; the measured gain vs DPO-only is **+5–10pp GSM8K, +~4pp IFEval, neutral elsewhere**. Concrete numbers: Tülu 3 8B hits **GSM8K 87.6** / **MATH 43.7** / **IFEval 82.4** vs the Llama-3.1 8B Instruct baseline at 84.7 / 41.5 / 80.5 ([[rlvr-tulu3]]).

The three-stage ordering — SFT (capability breadth) → DPO (preference style, helpfulness, safety) → RLVR (sharpen verifiable reasoning) — is now the canonical open-recipe template; OLMo 2 adopts it unchanged.

### 1.6 A pitfall worth naming — verifier loopholes

The no-RM story is too clean in isolation. RLVR's Goodhart-proof property only holds *if the verifier is tight*. [[rlvr-tulu3]] flags the failure mode explicitly: a sloppy math grader that accepts any line containing the string `"42"` lets the policy hack by emitting `"42"` inside a distractor sentence. Verifier engineering is therefore unit-test engineering. The public Tülu 3 verifiers are strict by construction — SymPy equivalence for MATH, exact integer match for GSM8K, unit-test pass with sandboxed 5-second timeout for code, regex constraint-match for IFEval — and the RLVR prompt set is gated to prompts with a known reference answer and a working verifier. Everything else is routed through RLHF/DPO where a learned RM is the only option.

---

## 2. Tülu 3 → Tülu 3.1 — the narrow delta

"Tülu 3.1" is not a new family. Depending on which Ai2 document you read, it refers to either (a) a **single-stage post-training update** to the 8B checkpoint, or (b) a **multi-base refresh** applying the Tülu 3 recipe unchanged to OLMo 2 alongside Llama 3.1. Both framings are documented; the meaningful algorithmic delta is (a).

### 2.1 The 8B single-stage delta — PPO → GRPO

The HF model card for `allenai/Llama-3.1-Tulu-3.1-8B` is explicit ([[tulu-3.1]]): the parent checkpoint is `allenai/Llama-3.1-Tulu-3-8B-DPO` and the *only* change is in the **final RL stage**. Specifically:

- Algorithm swaps **PPO → GRPO**. GRPO removes the value network, estimating advantages from group-level return baselines within each prompt's rollout group.
- The final stage runs with **no reward model at all** — RLVR-only, as in Tülu 3, but with GRPO instead of PPO as the RL optimizer.
- Hyperparameters in the final stage are re-tuned; Ai2 does not publish the deltas.
- The RL training mix is `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`.
- The SFT and DPO stages are inherited unchanged from Tülu 3.

### 2.2 Why this delta matters

Tülu 3.1 is valuable as a **controlled public ablation**: it isolates the PPO→GRPO swap while holding SFT, DPO, data, and base model fixed. This is rare — most open releases change base model, data, and algorithm simultaneously, and the community cannot attribute the gain. Tülu 3.1 says, on this base, on this data, on this DPO checkpoint, GRPO produces a higher average than PPO, and here is the delta table to prove it. It is also the first public Ai2 model to adopt the GRPO / verifier-style RL path that DeepSeek-R1 and DeepSeekMath made prominent ([[tulu-3.1]]).

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

The *per-base refresh* framing in [[tulu-3-1]] treats 3.1 as "rerun the whole stack on a new base" — valuable evidence that the recipe is base-agnostic, but not algorithmically new. The blog-driven narrative and the HF-card-driven narrative are consistent; the HF card is the one that names PPO → GRPO explicitly.

---

## 3. Llama 3 — the six-round flywheel

Llama 3 ([[llama-3]], Grattafiori 2024) takes the opposite topology. Instead of a linear SFT → DPO → RLVR pipeline, post-training is a **closed loop repeated six times**. Each round is: train a fresh **reward model** on freshly-collected human preferences; use that RM to **rejection-sample** the current best chat checkpoint's generations into a new SFT pool; **SFT** on the new pool; **DPO** on the new preference data. The round-*N+1* reward model, SFT pool, and DPO pairs all come from the round-*N* checkpoint's outputs.

### 3.1 Per-round mechanics

Each of the six rounds uses the same inner primitives:

- **Rejection sampling for SFT data.** For each prompt, sample **K=10–30 completions** from the best round-(N−1) chat model at temperature **T=0.6–1.0**; score each with the fresh RM; keep the top-scoring completions. A distilled topic classifier and a distilled quality classifier (both from Llama 3) then remove low-quality rejection-sampled text before SFT. The rejection-sampled outputs *dominate* the SFT pool.
- **SFT** at LR **1e-5** (405B), cosine decay, context 8K–32K extended, loss on response tokens only.
- **Reward model.** Initialized from the Llama 3 pretrained checkpoint; linear head replaces LM head. Trained on pairwise preferences with margin labels ("significantly better", "better", "slightly better", "negligibly better") where the margin is used for data filtering / up-weighting but not in the loss itself — standard pairwise logistic.
- **DPO** at LR **1e-5**, **β=0.1**, with an **auxiliary NLL-on-chosen loss with coefficient 0.2** to prevent chosen-logprob decay. Single epoch per round, prompt tokens masked from loss, *most-recent-batch preference data only* — older batches cause format drift.
- **Per-round data mix:** ~50–80% rejection-sampled synthetic, remainder is human SFT, capability-specific synthetic (code-exec-filtered code, math with verifier, multi-turn tool-use traces, long-context QA), and the latest preference batch ([[llama-3]]).

### 3.2 Six-round table — what changed and what each round caught

Meta does **not publish per-round eval deltas**; the round-by-round interpretation below is *inferred from public prose in [[llama-3]] and [[llama-3-synthetic-pipeline]]*, not a direct Table. The sequence of emphasis is attested; the specific eval a round "caught" is reasoned reconstruction and is labelled as such.

| Round | Data source emphasis | Eval issue round targets (inferred) | Attested? |
|---|---|---|---|
| 1 | Human SFT demonstrations dominant; limited rejection-sampled data (no strong RM yet) | Basic instruction following, chat format | attested (round-1 uses smaller synthetic share) |
| 2 | First large RM trained; rejection sampling from round-1 best chat model | Coding / math format regressions found in round-1 | inferred |
| 3 | Synthetic code-exec-filtered + math-verifier data ramps | Reasoning and code-pass@1 lift; factuality pipeline enters | inferred from [[llama-3-synthetic-pipeline]] factuality description |
| 4 | Multi-turn tool-use traces added; long-context QA added | Tool-call format drift, long-context retention | inferred |
| 5 | Multilingual + safety synthetic augmentation; Llama Guard 3 co-trained | Multilingual regressions; refusal calibration | inferred |
| 6 | Newest preference batch only; DPO on highest-quality pairs; final polish | Remaining format drift; final helpfulness / harmlessness trade-off | inferred |

The *attested* narrative thread is (a) synthetic share rises across rounds, (b) reward model is *rebuilt from scratch* each round on fresh preferences, (c) **stale preferences are never reused** — the prose reason is that preference distributions drift under policy changes and re-using old pairs overfits to the past policy's quirks. Llama Guard 3, the safety classifier, is trained jointly across rounds but does not enter the core policy until round 5.

### 3.3 Why multi-round beats single-pass — the attested reasoning

Meta's stated reason for six rounds, distilled from [[llama-3]] and [[llama-3-synthetic-pipeline]]:

1. **Reward-model drift.** A static RM becomes stale once the policy shifts; rewards measured against yesterday's policy over-optimize to artifacts the new policy has outgrown. Retraining the RM every round keeps Goodhart's gap bounded.
2. **Rejection-sampled SFT data improves monotonically.** Round-*N*'s best outputs, filtered by a fresh RM, are the round-*N+1* SFT data. The policy literally bootstraps its own curriculum; round 1's SFT pool is weaker than round 6's because the generator is weaker.
3. **Preference-label quality compounds.** Multiple rounds of QA on human annotations — Meta states multiple QA rounds explicitly — mean later-round preferences are cleaner than earlier-round preferences, and the DPO step gains more signal.
4. **Capability recovery.** Each round's eval pass catches a regression in a specific capability (Meta does not list which round caught which regression publicly); the next round's synthetic mix is reweighted to fix that specific gap. This is the "data flywheel": evals → mixture reweight → next round.
5. **Format drift is local.** By training DPO on *most-recent-batch* data only, each round constrains format drift to one round's width; old batches would pull the policy toward stale formats.

The topology cost is that each round requires a fresh preference collection (labour-intensive) and a fresh RM train. The Tülu 3 alternative — SFT once, DPO once, RLVR once — avoids the labour but loses the ability to catch capability regressions that only surface *after* RL has run.

---

## 4. Side-by-side — what the two recipes teach

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

The right way to read this table is not "one is better". Tülu 3 is what you build when you can afford to publish and cannot afford six rounds of annotation. Llama 3 is what you build when you can afford annotation and cannot afford to publish. The common substrate is that both treat **data composition as a first-order training variable** — and both have internalized that modern post-training is not fine-tuning, it is a program whose inputs are data mixtures and whose outputs are checkpoints.

A second reading of the comparison: **where do the two recipes fail?** Tülu 3 fails silently when the eval suite has a blind spot the mixture does not cover — because the mixture is fixed before any evals run, a missing capability propagates all the way to RLVR unnoticed. Llama 3 fails expensively when human preference annotation drops in quality mid-round — because the loop compounds labour, a bad annotation batch contaminates the RM, which contaminates the rejection-sampled pool, which contaminates the DPO pairs. Both teams describe the same risk family (Goodhart drift, stale preferences, reward-model over-optimization) but locate it differently: Tülu 3 pushes it pre-mixture; Llama 3 pushes it per-round.

A third reading, historical: Llama 3 *dropped* PPO from Llama 2's recipe in favour of DPO; Tülu 3 *reintroduced* PPO under the RLVR label by changing the reward signal. The community's summary — that RL-for-LLMs has converged on preference-based methods — hides that the two most-copied open recipes disagree on whether to use online RL at all. Tülu 3.1's PPO → GRPO swap is a further wrinkle: it picks a *third* RL family in one narrow stage of one ablation-grade checkpoint. The next chapters (Qwen 2.5/3, OLMo 2/3, Phi 3/4, Nemotron) each pick a point in this space and name their reasons.

---

## Companion visualization

**[figures/tulu-llama-recipe.html](figures/tulu-llama-recipe.html)** — side-by-side interactive: the Tülu 3 linear pipeline on the left (three stages with per-stage eval contribution) and the Llama 3 six-round flywheel on the right (round selector with per-round synthetic share and an inferred eval-delta bar per capability). Stepping through rounds shows how Llama 3's mixture tilts toward synthetic and how each round's targeted capability moves. Use it after reading §3.2 to build intuition for the flywheel — which rounds target reasoning, which target tool use, which target refusal calibration.

---

## Connections

- **ch-30 (SFT design axes)** — Tülu 3 and Llama 3 instantiate different points in the design axes space (mixture scale, synthetic ratio, epoch count).
- **ch-31 / ch-32** — decontamination, skill-mixture-first, and persona synthesis are developed in prior chapters; this chapter is where they show up in a real released model.
- **ch-34 (Case Studies B)** — Qwen 2.5/3, OLMo 2/3, Phi 3/4 all build on these two templates; OLMo 2 is the closest direct reuse of the Tülu 3 recipe.
- **ch-35 (Case Studies C)** — Nemotron and R1-distill push the synthetic ratio past 95% and change the story further.
- **RL track (ch-37..ch-46)** — Tülu 3's RLVR is the on-ramp to the verifiable-rewards family; Llama 3's DPO + NLL-stabilizer is a canonical DPO case study.
- **Eval track (ch-47..ch-53)** — the decontamination rules and effective-context discipline used here are the eval-side counterpart.

## Further reading

- [[tulu-3]] — the 2024 Tülu 3 tech report; the paper behind §1.
- [[tulu-3.1]] — the HF model card for the 8B PPO→GRPO delta.
- [[tulu-3-1]] — the multi-base refresh framing.
- [[tulu-3-sft-mix]] — per-source mix card that §1.1 is drawn from.
- [[allenai-tulu-sft-recipe]] — bucket shares, decontam, SFT hparams, ablations.
- [[allenai-tulu-blog]] — fully-open ethos and RLVR introduction.
- [[allenai-tulu-synth]] — persona-driven prompt factory.
- [[llama-3]] — the Herd of Models report; the paper behind §3.
- [[llama-3-synthetic-pipeline]] — Meta blog on the iterative loop and `synthetic-data-kit`.
- [[rlvr-tulu3]] — the RLVR methodology page with the 8B numbers used in §1.5.
