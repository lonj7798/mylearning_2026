<!-- chapter: ch-58a
     track: capstone
     kind: content
     title: Open General-Model Recipes End to End: Pretraining to Merge
     deps: [ch-58, ch-14a, ch-32e, ch-34, ch-45a, ch-45d]
     sources: [[olmo-2-stage-chain]], [[olmo-3-stage-chain]], [[olmo-core-olmo3-configs]], [[allenai-olmo3-open-instruct-scripts]], [[allenai-olmo3-open-instruct-scripts-recipe]], [[tulu-3-stage-chain]], [[open-instruct-allenai-recipes]], [[open-instruct-allenai-recipes-recipe]], [[smollm3-stage-chain]], [[smol-training-playbook]], [[llama-3]], [[llama-3-recipe]], [[qwen-2.5]], [[qwen-2.5-recipe]], [[deepseek-v3]], [[deepseek-v3-recipe]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[epoch-deepseek-r1-compute]], [[k2-v2]], [[apertus]]
     figures: figures/stage-chain-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 58a — Open General-Model Recipes End to End: Pretraining to Merge

> **Core insight.** Across the eight open recipes compared here, the stage chain is more consistent than the settings inside it: the seven that perform their own pre-training all end it with a separate shorter stage on a higher-quality mixture, all eight run an imitation stage before any stage that uses a comparison or a verifier, and seven of the eight run at least one comparison-based or verifier-based stage (K2-V2 is the single exception; its released chain stops at SFT); four of the eight average or merge checkpoints somewhere in the chain, and two of those four end with the merge. The settings inside those stages do not transfer: at a fixed stage and a fixed data mixture, Tülu 3 uses SFT learning rate 5e-6 at 8B and 2e-6 at 70B and 405B, DPO 5e-7 at 8B and 2e-7 at 70B, and RLVR 3e-7 at 8B and 1e-7 at 70B ([[tulu-3-stage-chain]] Tables 11, 20, 21, 34), while OLMo 2 reports that its own bases "required significantly higher learning rates compared to the Llama 3.1 training recipe" ([[olmo-2-stage-chain]] §5). What is disclosed also differs by stage rather than by lab: DeepSeek-V3 prints every pre-training optimizer setting and prints no RL hyper-parameter at all ([[deepseek-v3-recipe]]), and Olmo 3 prints paper tables whose values disagree with the launch scripts that produced the released checkpoints in at least five rows ([[allenai-olmo3-open-instruct-scripts-recipe]]).
>
> **Guideline.** When you plan a run from a published recipe, copy the stage chain and the stage-transition gate first and the numeric settings last, because the chain is replicated across labs and the settings are not: for the same stage chain at 7–8B, Tülu 3 selected SFT 5e-6 and DPO 5e-7 on a Llama 3.1 base while OLMo 2 selected 2e-5 and 1e-6 on its own base ([[tulu-3-stage-chain]] Tables 11, 20; [[olmo-2-stage-chain]] §5). When a recipe is available as both a report and released code, record the paper value, the released-config value, and the framework default as three separate facts and train on the config value, because open-instruct commit 7917d41 moved β, loss type, epochs, warmup and clipping out of the Olmo 3 scripts into defaults, and a script read after that commit runs different values than the same script read before it ([[allenai-olmo3-open-instruct-scripts]] PR #1547). When you transfer a setting across model sizes, transfer the selection procedure rather than the number — a sweep range, a target such as K2-V2's τ_epoch scaled by 1/√TPP ([[k2-v2]] §3.2), or a gate such as Llama 3's rule of advancing a long-context stage only after short-context evaluations recover ([[llama-3-recipe]] §3.4.2) — because every recipe here that states how a value was chosen states a procedure, while several print no ablation at the released size for any post-training setting, Qwen2.5 among them ([[qwen-2.5-recipe]], where every post-training row reads “no ablation reported”).

## Why this chapter matters for a general-purpose model

The pipeline is pre-training → mid-training → SFT → preference optimization → RL → evaluation. Previous chapters compared one stage across many models: ch-14a compared pre-training budgets and schedules, ch-32e compared mid-training and context extension, ch-45a compared preference and RL stages, ch-45d compared agentic recipes, and ch-33 and ch-34 read individual reports as case studies. This chapter compares all stages of one model at a time: for each of eight mostly disclosed models, what is the full chain from the first pre-training token to the released checkpoint, what gated each transition, and what remains unknown.

The measurable problem has four parts.

1. **A plan needs a chain, not a list of settings.** A team with a fixed compute budget has to decide how many stages to run and in what order before it can decide any learning rate. The stage chain is the decision with the largest effect on which capabilities the model ends with, and it is the part that published recipes agree on most.
2. **Settings are scoped to a size, a stage and a run.** Four of the recipes here print different values for the same setting in two of their own artifacts (§3). Copying an unscoped value is the most common way a reproduction diverges from the released checkpoint.
3. **The evidence for a stage is usually a gate, not an ablation.** Most stage transitions were decided by an evaluation threshold, a sweep, or a qualitative check, not by an A/B comparison of pipelines. Knowing which is which changes how much a reader should trust the order (§5).
4. **Generality is measured in some stages and not others.** Olmo 3's RL-Zero launch scripts evaluate only the domain being trained, while its Instruct scripts launch a fifteen-benchmark list ([[allenai-olmo3-open-instruct-scripts]]); Tülu 3 measured a separate unseen suite at every stage and found one skill moving against the reported trend ([[tulu-3-stage-chain]] Table 31). A ledger that records only settings cannot answer whether a stage broadened or narrowed the model.

An interactive version of the comparison in §2 and §3 is in [stage-chain-explorer.html](figures/stage-chain-explorer.html): it shows each recipe as a chain of stages with the token budget, the batch, and the disclosure status of each stage, so the chains can be compared at a normalized token scale rather than by reading eight reports.

## §1 The end-to-end ledger: normalizing eight recipes onto one row format

**Definition.** An end-to-end ledger is a recipe ledger (ch-14a §1, style §5.2) extended so that every stage of one released checkpoint appears, in training order, with the same eight columns: model, size, stage, setting, value, source location, status, and the evidence that selected the value. A stage is named from a fixed vocabulary: pretrain-stable, pretrain-decay/anneal, mid-train, long-context, SFT, distill-SFT, reward-model, preference, RL, merge, eval-gate.

**Problem.** Reports name their stages differently. What OLMo 2 calls "stage 2" is a learning-rate anneal on a high-quality mix ([[olmo-2-stage-chain]] §2.3); what Olmo 3 calls mid-training is a separate 100B-token run at constant batch after a completed pre-training schedule ([[olmo-core-olmo3-configs]]); what K2-V2 calls mid-1 is a 1.769T-token stage, larger than the whole pre-training run of many 7B models ([[k2-v2]] Table 3). Without a fixed vocabulary, a comparison across recipes compares stage names rather than stages.

**Mechanism.**
1. Fix the released checkpoint by name, not by family (OLMo-2-1124-7B, Olmo-3-1025-7B, Llama-3.1-Tulu-3-8B, SmolLM3-3B).
2. Walk the report in training order and assign each described run to one stage name from the vocabulary above.
3. Normalize the unit of every quantity: tokens seen along one path; batch in tokens; prompts and samples per prompt as separate numbers; pairs distinct from prompts; epochs distinct from steps.
4. Read the released configs or launch scripts at a pinned commit for the same stage and record them as separate rows.
5. Assign a status: verified, not reported (with the list of what was checked), conflict (one row per artifact), or derived (with the formula).
6. Record the gate: the evaluation or sweep that ended the stage, with its locus, or "no gate reported".

**Worked example: normalizing the SFT stage of six recipes to one unit.** Reports state SFT size in prompts, examples, or tokens, and each also states epochs. Tokens seen in SFT = (tokens per example) × (examples) × (epochs).

| Checkpoint | As printed | Epochs | Tokens seen in SFT | Status |
|---|---|---|---|---|
| Olmo 3 7B Think SFT | 2,268,468 prompts; "Total Tokens" 45.4B | 2 | 45.4B | verified ([[olmo-3-stage-chain]] Table 47); derived average 45.4e9 / (2 × 2,268,468) = 10,007 tokens per prompt |
| DeepSeek-R1 (Dev3) | 804,745 samples, mean 5,355.3 tokens | 2–3 | 4.31B per epoch → 8.6–12.9B | derived from [[deepseek-r1-recipe]] Table 5; the epoch count is a conflict (v1 "two epochs", v2 "2–3 epochs") |
| K2-V2 SFT | 17.4B tokens per epoch | 3 | 52.2B | verified ([[k2-v2]] §6.1) |
| SmolLM3 SFT | 1.8B tokens | 4 | 7.2B | derived from [[smollm3-stage-chain]] |
| OLMo 2 7B SFT | 939,104 prompts, max length 4,096 | 2 | not reported | not reported ([[olmo-2-stage-chain]] §5; token count absent from the report and the launch files) |
| Qwen2.5 Instruct | "over 1 million SFT examples", length 32,768 | 2 | not reported | not reported ([[qwen-2.5-recipe]] §4.1) |

The ordering changes under normalization. By prompt count, Olmo 3 7B Think SFT (2.27M) is 2.8 times DeepSeek-R1's rejection-sampling SFT (805K); by tokens seen, it is 3.5 to 5.3 times larger, because its average example is about 1.9 times longer. A plan that budgets SFT by prompt count will under-budget a long-chain-of-thought stage by roughly that factor.

**Conditions and limits.** The derived tokens-per-prompt figure for Olmo 3 assumes two things the report does not state: that the printed 45.4B is the whole two-epoch run rather than one epoch, and that it counts packed training tokens including any masked prompt tokens. Under the other reading of the first assumption (45.4B per epoch), tokens seen would be 90.8B and the average example 20,014 tokens; the report states batches are "measured in tokens instead of instances" with document packing, and does not separate loss-carrying tokens from packed tokens ([[olmo-3-stage-chain]] A.6.1). Where a report gives neither tokens nor a mean length, no normalization is possible and the cell stays "not reported".

**Implication for a general-purpose model.** The size of the imitation stage relative to pre-training is one of the few quantities comparable across all eight recipes, and the two recipes that print both numbers differ by a factor of 6.6 in it: 7.2B SFT tokens after 11.2T pre-training tokens for SmolLM3 (0.064%) against 52.2B after 12.25T for K2-V2 (0.43%). Chapters ch-30a and ch-38a treat what that ratio does to retention of prior ability; this chapter supplies the normalized numbers.

## §2 The eight chains side by side

**Definition.** A chain is the ordered list of stages that produced one released checkpoint, with the token budget or example count of each.

| Recipe (released checkpoint) | Chain, in training order | Checkpoint averaging or merging |
|---|---|---|
| OLMo 2 7B Instruct | pretrain 3.90T → anneal 3 × 50B, averaged → SFT (939,104 prompts, 2 ep.) → DPO (366.7k prompts) → RLVR (PPO) | inside the anneal stage (three runs averaged) |
| Olmo 3 7B Instruct | pretrain 5.93T → mid-train 100B → long-context 50B at 65,536 → Think SFT 45.4B tokens → Think DPO 150K pairs → Instruct SFT 3.4B tokens → Instruct DPO 260K pairs → Instruct RL | at Think SFT (two learning rates merged) and, for the 32B, at mid-training (two runs souped) |
| Tülu 3 8B | (Llama 3.1 8B base, not trained here) → SFT (2 ep., batch 128) → length-normalized DPO (β 5, 1 ep.) → RLVR (PPO, β 0.05) | none |
| SmolLM3-3B | pretrain 11.2T in 3 stages → long-context 2 × 50B → reasoning mid-train 35B × 4 ep. → SFT 1.8B × 4 ep. → APO → soup of APO checkpoints → linear merge 0.9 / 0.1 with a mid-training checkpoint | the last operation before release |
| Llama 3.1 405B | pretrain 15.6T → long-context 6 stages, ≈800B tokens → anneal final 40M tokens with Polyak averaging → 6 rounds of (reward model → rejection sampling → SFT → DPO) → averaging of models at each of the RM, SFT and DPO stages | at three stages; the last operation before release |
| Qwen2.5 (open weights) | pretrain 18T (4,096 context) → final pre-training stage at 32,768 → SFT (>1M examples, 2 ep.) → DPO (≈150,000 pairs, 1 ep.) → GRPO (8 responses per query) | no checkpoint merge; an Online Merging Optimizer is used inside DPO |
| DeepSeek-V3 → R1 | V3: pretrain 14.8T → long-context 2 × 1,000 steps → SFT 1.5M instances (2 ep.) → GRPO. R1: cold-start SFT ("thousands") → reasoning RL → rejection-sampling SFT 804,745 samples restarted from V3-Base → mixed RL 1,700 steps | none |
| K2-V2 70B | pretrain 12.25T → mid-1 1.769T → mid-2/3/4 590B + 229B + 131B with context to 524,288 → SFT 17.4B × 3 ep. | none |

The same eight chains are drawn to a token scale in [stage-chain-explorer.html](figures/stage-chain-explorer.html), where selecting a stage prints its settings, source locus and status; the "post-training stages only" filter isolates the columns that this table compresses into one cell.

Sources by row: [[olmo-2-stage-chain]]; [[olmo-3-stage-chain]] and [[olmo-core-olmo3-configs]]; [[tulu-3-stage-chain]]; [[smollm3-stage-chain]]; [[llama-3-recipe]]; [[qwen-2.5-recipe]]; [[deepseek-v3-recipe]] and [[deepseek-r1-recipe]]; [[k2-v2]].

**What is shared.** Every recipe here that performs its own pre-training ends it with a distinct final stage rather than stopping the stable phase and moving to SFT, whether that stage is called an anneal (OLMo 2, Llama 3.1), mid-training (Olmo 3, K2-V2, SmolLM3's reasoning stage), a decay stage (SmolLM3 stage 3, DeepSeek-V3), or a final pre-training stage at a longer context (Qwen2.5's move to 32,768). Tülu 3 is the exception because it starts from a finished base. All eight run an imitation stage before any comparison-based stage: even DeepSeek-R1, whose R1-Zero branch shows that RL from a base model raises AIME 2024 pass@1 from 15.6% to 77.9%, adds a cold-start SFT to the released model because R1-Zero scores 46.6 on IF-Eval and 24.7 on AlpacaEval 2.0 ([[deepseek-r1]] §1, Table 3).

**What is not shared.** The number of preference-style stages ranges from zero (K2-V2) to twelve (Llama 3.1: six rounds each containing a reward model and a DPO stage). Where the long-context stage sits varies by four positions in the chain (§7). Four recipes average or merge checkpoints (OLMo 2 inside the anneal, Olmo 3 at the 7B Think SFT stage and the 32B mid-training stage, SmolLM3 at the end, Llama 3.1 at the reward-model, SFT and DPO stages), and in two of them — SmolLM3 and Llama 3.1 — the merge is the last operation before release.

**Evidence for the shared order.** The strongest published evidence that imitation should precede comparison is DeepSeek-R1's Table 3, which reports each intermediate checkpoint of one pipeline: R1-Zero (RL only) AlpacaEval 2.0 24.7 → Dev1 (cold-start SFT) 50.1 → Dev2 (reasoning RL) 55.8 → Dev3 (rejection-sampling SFT with non-reasoning data) 62.1 → R1 (mixed RL) 87.6, with AIME 2024 falling from 77.9 at R1-Zero to 59.0 at Dev1 and recovering to 79.8 at R1 ([[deepseek-r1]] §4, Table 3). Status: **Result (single study)**, one pipeline at 671B total parameters, no repetition of the pipeline with stages reordered.

## §3 The disclosure matrix: paper, config, and default are three facts

**Definition.** A disclosure matrix records, per recipe and per stage, whether the stage can be reproduced from public artifacts: is the data public, is a config or launch script public at a pinned commit, is the starting checkpoint public, and are the hyper-parameters stated.

**Problem.** "Open" is not one property. A recipe can publish weights and no data, data and no configs, or configs whose values disagree with the paper.

| Recipe | Pre-training data | Pre-training config | Post-training data | Post-training launch code | Paper-versus-config conflicts recorded |
|---|---|---|---|---|---|
| OLMo 2 | public (OLMo-Mix-1124) | public YAML | public (Tülu 3 mixes) | public (`docs/olmo2.md`) | 13B peak LR 9.0e-4 (Table 3) vs 3.0e-4 (`OLMo2-13B-stage1.yaml` L46; read for ch-14a); 7B SFT LR 1e-5 (YAML) vs 2e-5 (script); 7B DPO 5e-7 (YAML) vs 1e-6 (script) ([[open-instruct-allenai-recipes-recipe]] for the two post-training rows) |
| Olmo 3 | public (Dolma 3) | public (OLMo-core scripts) | public (Dolci) | public (17 scripts, per-run Beaker links) | 7B peak LR 2.0712e-4 (script) vs 2.074e-4 (Table 35); 32B Think RL 64 prompts per step (script) vs 128 (Table 49); 7B Instruct DPO 32 GPUs (derived) vs 16 (Table 48) ([[olmo-core-olmo3-configs]], [[allenai-olmo3-open-instruct-scripts-recipe]]) |
| Tülu 3 | not applicable (Llama 3.1 base) | not applicable | public | public (`docs/tulu3.md`) | 70B RLVR β 0.7 (§6.4 prose) vs 0.07 (Table 21 caption and 70B launch command); episodes 100,000 (Table 21, all sizes) vs 400,000 (§6.4 and the 70B command) vs 10,000,000 planned in the 8B command ([[tulu-3-stage-chain]]; [[open-instruct-allenai-recipes-recipe]]) |
| SmolLM3 | dataset names and per-stage shares public; the assembled corpus is not released | public configs | partly public | not released | decay start 10T (blog) vs 9.90T (derived from configs); total 11T vs 11.2T vs 11.1T ([[smollm3-stage-chain]]) |
| Llama 3.1 | not public | not released | not public | not released | none recordable; only the report exists |
| Qwen2.5 | not public | not released | not public | not released | SFT example count "over 1 million" (§4.1, Abstract) vs "1 million examples" (§1) vs "millions" (§4 intro) ([[qwen-2.5-recipe]]) |
| DeepSeek-V3 / R1 | not public | not released | not public | not released | R1 second-stage SFT "two epochs" (v1) vs "2–3 epochs" (v2) ([[deepseek-r1-recipe]]) |
| K2-V2 | public (TxT360) | public (`llm360/k2v2_train`) | public (TxT360-3efforts) | public | none recorded here; the code and log were not read for this chapter |

**Mechanism: why a script value and a paper value differ.** In open-instruct, PR #1547 (commit 7917d41, 2026-03-20) changed framework defaults to DPO β 5.0 with `dpo_norm`, 1 epoch, warmup 0.1, and GRPO `clip_higher` 0.272, ρ cap 2.0, `async_steps` 8, in-flight updates true, advantage normalization "centered" — and then deleted those flags from the Olmo 3 launch scripts ([[allenai-olmo3-open-instruct-scripts]]). Reading a script at the head commit therefore shows fewer settings than reading it at its run commit, and the deleted flags are recoverable only from the parent of that commit, 00d42c4. For several Olmo 3 RL runs the pre-#1547 default differs from the post-#1547 default, so a script that never printed a flag ran a different value than the same script runs today: the 7B Think no-pipeline run took truncated-importance-sampling correction off and standard advantage normalization from the 00d42c4 defaults, the 7B Instruct RL run differs from today's defaults in the truncated-importance-sampling correction, and the RL-Zero code, instruction-following and general runs differ in advantage normalization, while the current defaults are ρ cap 2.0 and centered advantages ([[allenai-olmo3-open-instruct-scripts-recipe]]).

**Worked example: three facts about one setting.** For the Olmo 3 7B Think DPO stage, the three artifacts state: paper Table 48 — 150K pairs, LR 8.0e-8, max length 16,384, batch 128, 32 GPUs; launch script `7b_think_dpo.sh` L3–40 — the same 150,000 `max_train_samples`, 8e-8 linear, 16,384, and a derived batch of 1 × 4 × (4 × 8) = 128; framework default at d8a7f1c — `dpo_norm` with β 5.0, 1 epoch, warmup 0.1, none of which the script prints. Only the union of the three describes the run, and only the third changes if the framework is upgraded. Status: verified rows in [[allenai-olmo3-open-instruct-scripts-recipe]].

**Conditions and limits.** A conflict count is a lower bound on disagreement, not a quality score: a recipe with no released code has zero recorded conflicts because there is nothing to compare. Llama 3.1, Qwen2.5 and DeepSeek-V3 appear conflict-free in the table for that reason.

**Implication for a general-purpose model.** The stages most often left unstated are the ones that most affect breadth. Qwen2.5 prints no RL learning rate, KL coefficient, clip ε, temperature, response length, or episode count ([[qwen-2.5-recipe]] §4.3); DeepSeek-V3 prints no RL hyper-parameter at all ([[deepseek-v3-recipe]] §5.2). Both print their pre-training optimizer settings. A reader can reproduce the stage that sets raw capability and cannot reproduce the stage that shapes how broadly that capability is expressed.

## §4 Scale transfer: what changes from 405B to 1B

**Definition.** Scale transfer is the question of which settings keep their value when the same recipe is run at a different model size, and which must be re-selected.

**Problem.** A small team copies a recipe published at 70B or 405B. If a setting scales with size and the team keeps it fixed, the run is mis-specified in a way that does not raise an error.

**What the recipes show, at a fixed stage and mixture.**

| Setting | Tülu 3 8B | Tülu 3 70B | Tülu 3 405B | Direction |
|---|---|---|---|---|
| SFT peak LR | 5e-6 | 2e-6 | 2e-6 | falls with size |
| SFT effective batch (sequences) | 128 | 128 | 256 | flat, then doubled |
| DPO peak LR | 5e-7 | 2e-7 | 2e-7 | falls with size |
| DPO β (length-normalized loss) | 5 | 5 | 5 | unchanged |
| RLVR peak LR | 3e-7 | 1e-7 | 1e-7 | falls with size |
| RLVR effective batch (responses) | 224 | 640 | 1,856 | rises with size |

All rows from [[tulu-3-stage-chain]] Tables 11, 20, 21, 34, 35, verified against the launch commands in [[open-instruct-allenai-recipes-recipe]].

The same pattern holds inside Olmo 3, where the 7B and 32B are the same recipe: pre-training peak LR 3.0e-4 (7B) against 6.0e-4 (32B) with the batch doubled from 4,194,304 to 8,388,608 tokens; Think SFT LR 5e-5 against 1e-4 souped with 5e-5; Think DPO 8e-8 against 7e-8; Think RL 1e-6 against 2e-6 ([[olmo-3-stage-chain]] Table 35, Table 47; [[allenai-olmo3-open-instruct-scripts-recipe]]). Of the three post-training learning rates, two rise with size and one falls, so "learning rate falls with size" is not a rule that survives across recipes; what survives is that the value was re-selected at each size.

**The one published procedure for transferring a setting.** K2-V2 states a scaling target rather than a value: "the effective averaging timescale τ_epoch", built from the learning rate η, the weight decay λ, the dataset size D and the batch B, "approximates the effective number of epochs over which AdamW exponentially averages parameter updates"; the target is tuned at 20 tokens per parameter and then scaled proportionally to 1/√TPP for the full run, giving η = 1.5e-4, λ = 0.05, B = 9.8 × 10⁶ tokens, D = 12.25 × 10¹² and τ_epoch = 0.1066 ([[k2-v2]] §3.2).

**Worked example: identifying the target by reproducing it.** A layout-preserving text extraction of the report renders the fraction ambiguously — `τepoch = ηλD` on one line and `B` on the next — and the two orientations differ by a factor of 88: B / (ηλD) = 9.8e6 / (1.5e-4 × 0.05 × 1.225e13) = 9.8e6 / 9.1875e7 = 0.1067, while ηλD / B = 9.375. The printed target is 0.1066, so the quantity being held fixed is B / (ηλD). A second extraction of the same PDF without layout preservation emits `B` before `ηλD`, which agrees with that reading, and so does the stated meaning: at η = 1.5e-4 and λ = 0.05 the exponential average has a timescale of 1 / (ηλ) = 1.33 × 10⁵ steps, which at 9.8 × 10⁶ tokens per step is 1.31 × 10¹² tokens, or 0.107 of the 12.25T dataset. A reader who copies the formula in the wrong orientation and solves for η at a different D or B gets η = τB/(λD) instead of η = B/(τλD), which is wrong by a factor of τ² = 0.0114 at the printed target. This is the check the ledger's "derived" status exists for: it takes one line and it either reproduces the printed number or it does not. Status: **derived**.

**What the sources say does not transfer.**
- Base-model family: "OLMo 2 required significantly higher learning rates compared to the Llama 3.1 training recipe" ([[olmo-2-stage-chain]] §5), for the same Tülu 3 data and stage chain.
- Algorithm viability at small size: RL from a base model failed to improve AIME with a 7B dense base and a 16B MoE base, with repetition as length grew, and worked with 32B dense, 230B MoE and 671B MoE bases ([[deepseek-r1]] G.1).
- Distillation against RL at a fixed size: at 32B, RL from Qwen2.5-32B-Base for over 10K steps reached AIME 2024 47.0 and LiveCodeBench 40.2, while SFT on the R1 800K set reached 72.6 and 57.2 ([[deepseek-r1]] Table 16).
- Safety mixture: Llama 3's Figure 18 reports that the 8B needs a higher share of safety data than the 70B for comparable safety, and the ratios themselves are not printed ([[llama-3-recipe]] §5.4.3).

**Implication for a general-purpose model.** The settings that transfer unchanged in this comparison are structural (β of the length-normalized DPO loss, epochs, warmup ratio, packing, loss masking); the settings that do not transfer are the ones that set step size and batch. A reproduction at a smaller size should keep the structure and re-run the sweep, with the sweep range taken from the published one: Tülu 3 swept the 70B DPO learning rate over 5.0e-7, 2.0e-7, 1.5e-7 and 1.0e-7 on one preference mix (development averages 72.74, 71.17, 71.12, 71.06) and over 5.0e-7 and 2.0e-7 on a second mix (71.14, 74.35), and chose 2.0e-7; the report's own reading is that "either a learning rate of 2.0 × 10⁻⁷ or 5.0 × 10⁻⁷, depending on data mix, performs better than a lower learning rate", so the range transfers and the winning point inside it does not ([[tulu-3-stage-chain]] Table 19).

## §5 The evidence chain: what gated each transition

**Definition.** A gate is the measurement that decided a stage was finished or that a candidate checkpoint was the one carried forward.

**Mechanism.** Four gate types appear in these recipes.
1. **Threshold gate on a held-out behaviour.** Llama 3.1's long-context stages advanced only when "short-context evaluations fully recovered" and needle-in-a-haystack was solved at that length ([[llama-3-recipe]] §3.4.2).
2. **Periodic evaluation with checkpoint selection.** Tülu 3 evaluated RLVR checkpoints every 100 steps (40 for the 70B) and released the checkpoint with the best MATH and IFEval, noting "For all of these models we took an earlier than final checkpoint from the run" ([[tulu-3-stage-chain]] §6.3).
3. **Sweep plus qualitative check plus merge.** Olmo 3 trained two epochs "to avoid overfitting", swept the learning rate, selected candidates on the evaluation suite, ran "a series of qualitative 'vibe-test' questions", and released a linearly weighted merge of two checkpoints trained at different learning rates ([[olmo-3-stage-chain]] §4.2).
4. **Stage-level regression check.** SmolLM3 found RULER degradation after APO, traced it to the reasoning mid-training stage and to the 24k-token cap on APO data, and repaired it with a merge rather than by re-running the stage ([[smollm3-stage-chain]]).

**Worked example: reading a stage table as evidence.** Olmo 3 Instruct 7B by stage, averaged over three runs ([[olmo-3-stage-chain]] Table 26): MATH 65.1 → 79.6 → 87.3 across SFT, DPO and RL; LiveCodeBench v3 20.0 → 18.8 → 29.5; PopQA 16.5 → 20.7 → 14.1; AlpacaEval 2 LC 21.8 → 43.3 → 40.9; Safety 89.5 → 89.9 → 87.6. Three of the five benchmarks listed here end below their best intermediate value. The table is evidence that the chain raises the targeted metrics and evidence that the last stage costs 6.6 points of PopQA and 2.3 points of Safety relative to the DPO checkpoint. Both readings come from the same three numbers, and a ledger that records only the final column loses the second.

**Where the order itself was tested.** Only two of the eight recipes report a comparison of pipeline variants rather than of settings inside a fixed pipeline. DeepSeek-R1 reports the intermediate checkpoints of its four-stage pipeline (§4, Table 3) and, separately, distillation against RL at 32B (Table 16). Olmo 3 reports that "training Olmo 3 Instruct on top of the Olmo 3 Think SFT both increases model performance on benchmarks, as shown in Table 29", with response length "minimally affected" (§5.5). Everything else in these reports is a sweep within an assumed order. Status: **Open question** for the order itself; no source here runs the same data through two stage orders at the same size.

**Conditions and limits.** Table 26 is an average of three runs and Table 3 is a single pipeline; the OLMo 2 stage table (Table 16) does not state a run count and its §5 text says "we experiment with 1 random seed initially to arrive on a configuration and up to 4 with final hyperparameters". Differences of one to two points between adjacent stages in any of these tables are not separated from seed variance by the reports themselves; ch-51 covers how to bound that.

## §6 Where each recipe measured general capability, and where it measured targets

**Definition.** A generality measurement is an evaluation on tasks that were not used for any development decision in the stages being measured.

**What the recipes did.**
- **Tülu 3** built a development suite and an unseen suite, pairing one or two unseen benchmarks with the development benchmarks of five skills (knowledge, reasoning, math, coding, instruction following); safety has development benchmarks and no unseen benchmark. The team states that it "did not examine scores on our unseen set when developing our models" ([[tulu-3-stage-chain]] §2.1, Table 3). It then reported both suites at every stage (Table 31).
- **OLMo 2** maintained "a held-out suite of tasks which were not used for model development decisions" and reported it, with the caveat in Table 6's caption that "OLMo 2 models were not evaluated on held-out datasets prior to release" ([[olmo-2-stage-chain]] §2.5).
- **Olmo 3** carries an "OlmoBaseEval HeldOut" block whose caption likewise states that the model "was not evaluated on held-out benchmarks prior to release" ([[olmo-3-stage-chain]] Tables 2–3).
- **Apertus** declares Table 21 as benchmarks that "were held-out during model development and were not used for making decisions", and separates "Factual Agnostic" from "Factual Regional" scores so that region-specific knowledge is not averaged away ([[apertus]] §5.2, Table 21 for the held-out declaration and Table 15 for the split).
- **Olmo 3 RL-Zero** scripts evaluate only the trained domain — AIME for math, HumanEval+/MBPP+/LiveCodeBench for code, IFEval for instruction following — while the Think and Instruct scripts launch a fifteen-benchmark list including PopQA, MMLU, GPQA and AlpacaEval ([[allenai-olmo3-open-instruct-scripts]]).
- **Qwen2.5** and **DeepSeek-V3** report benchmark tables with no held-out or unseen split described.

**Worked example: what an unseen suite catches.** Tülu 3's Table 31 at 8B, coding skill: development HumanEval 86.2 (SFT) → 83.9 (DPO) → 83.9 (final); unseen BigCodeBench 11.5 → 9.5 → 7.4. The development metric moves 2.3 points down and the unseen metric moves 4.1 points down, which is 36% of its starting value. At 70B the same cells move 92.9 → 92.4 → 92.4 and 12.2 → 23.0 → 21.6, in the opposite direction on the unseen benchmark. The report's §7.4.1 states that "For Reasoning and Coding, where the SFT checkpoints have the best performance on development evaluations, the subsequent training stages still improve model performance on harder unseen evaluations", which matches the 70B column and not the 8B column ([[tulu-3-stage-chain]] Table 31). The report also states, of the SFT mixture, "we see that our choices overfit to the development evaluations in Precise Instruction Following, and to some extent in Knowledge Recall and Reasoning."

**A second kind of check.** K2-V2 measures sentence-level memorization of benchmark questions directly, by beam search conditioned on sentence prefixes of AIME items: K2 23.32% (AIME 2024) and 14.43% (AIME 2025), Qwen2.5 72B 41.96% and 23.37%, Llama 3.1 70B 9.84% and 9.85%, Olmo 3 32B 37.82% and 14.47% ([[k2-v2]] Table 4). Its stated motivation is that "prior works reported that effect of such decontamination on benchmark performance is unclear". Status: **Result (single study)**, one measurement protocol, four models; ch-48 covers contamination detection.

**Implication for a general-purpose model.** Two of eight recipes report a per-stage split between measured-on and not-measured-on evaluations, and one of those two found a skill moving against its own summary. Any end-to-end plan should carry the unseen suite as a column of the ledger, not as a final report.

## §7 Long context in the chain: position determines what survives

**Definition.** The long-context stage is the run that raises the trained sequence length; its position in the chain determines which later stage can undo it.

**Positions observed.**
1. **Inside pre-training, at the end.** Qwen2.5 raises context from 4,096 to 32,768 "in the final pre-training stage", with RoPE base 10,000 → 1,000,000 ([[qwen-2.5-recipe]] §3.3).
2. **Immediately after pre-training, as two short phases.** DeepSeek-V3 runs two 1,000-step phases, 4K → 32K at batch 1920 and 32K → 128K at batch 480, both at the final pre-training learning rate 7.3e-6, for 119K H800 GPU hours ([[deepseek-v3-recipe]] §4.3, Table 1).
3. **After mid-training, as a separate base stage.** Olmo 3 runs 50B tokens (7B) or 100B (32B) at 65,536 with 34% long and 66% short data, YaRN factor 8 from 8,192, and intra-document masking ([[olmo-3-stage-chain]] §3.6; [[olmo-core-olmo3-configs]]).
4. **As several stages inside mid-training.** K2-V2 runs mid-2, mid-3 and mid-4 at 65,536, 131,072 and 524,288 tokens with RoPE base 1M, 10M, 10M, at a constant 6e-6 learning rate ([[k2-v2]] Table 3).
5. **Before a post-training stage that caps length.** SmolLM3 extends to 64k in two 50B stages, then runs reasoning mid-training and APO whose data is "limited to 24k tokens", and observes RULER degradation ([[smollm3-stage-chain]]).

**Mechanism of the regression in position 5.** A later stage trains only on sequences shorter than the extended context. Gradients from that stage move the parameters that the extension stage set, and nothing in the objective preserves behaviour beyond the cap. SmolLM3's repair keeps 90% of the aligned weights and 10% of a mid-training checkpoint with strong long-context behaviour, which "recover[ed] the base model's RULER score on contexts up to 128k tokens" ([[smollm3-stage-chain]]). Llama 3.1 avoids the same failure differently: DPO uses short-context data only, and the report states this "did not hurt long context when the SFT model was strong on long context", with 0.1% long-context data in the SFT mixture selected by ablation ([[llama-3-recipe]] §4.3.4).

**Worked example: the cost of the extension stage as a share of the budget.** Olmo 3 7B: 50B long-context tokens against 5.93T + 100B + 50B = 6.08T total, so 0.82% of tokens seen. Llama 3.1 405B: ≈800B against 15.6T, so 5.1%. DeepSeek-V3: 119K against 2,788K H800 GPU hours, so 4.3% of total compute, with the token count not reported. K2-V2: 590B + 229B + 131B = 950B against 14.97T, so 6.3%. The stage is between 0.8% and 6.4% of the budget in every recipe that prints enough to compute it.

**Conditions and limits.** Only Olmo 3 and Llama 3.1 report a comparison that selected a long-context data choice; the others print the setting alone. Olmo 3's mixture ablation used a 10B-token extension and reported that 66% long / 34% short lowered a subset of OlmoBaseEval by 2.5 points against 0.8 points for the chosen 34/66 split ([[olmo-3-stage-chain]] §3.6.3); K2-V2 reports "mild degradation in stage 4" without a number ([[k2-v2]] §5); Qwen2.5 reports RULER at 128K of 67.0 without and 88.4 with inference-time YaRN and Dual Chunk Attention, which is an inference setting rather than a training one ([[qwen-2.5-recipe]] Table 16). ch-32b and ch-32c cover the mechanics and the claimed-versus-effective length question.

## §8 Distillation placement: which stage imports another model's outputs

**Definition.** Distillation here means training on outputs sampled from a different, usually stronger model (ch-20, ch-35); self-distillation means training on outputs of the run's own previous checkpoint.

| Recipe | Teacher outputs enter at | What is imported | Evidence at the locus |
|---|---|---|---|
| DeepSeek-V3 | SFT (distill-SFT) | responses from per-domain expert models trained with SFT and RL, including R1-style long chains | on DeepSeek-V2.5: LiveCodeBench-CoT 31.1 → 37.4, MATH-500 74.6 → 83.2 against a short-chain baseline ([[deepseek-v3-recipe]] Table 9) |
| DeepSeek-R1-Distill (6 students) | SFT only | the 804,745-sample R1 set, on Qwen2.5 and Llama bases | Distill-Qwen-32B AIME 2024 72.6 vs 47.0 for RL from the same base ([[deepseek-r1]] Table 16) |
| Olmo 3 Think | DPO | Qwen3 32B traces as the chosen side of delta-learning pairs; the Think SFT mixes are named for public reasoning datasets whose teachers the passages read here do not state | further SFT on Qwen3 32B traces "outright hurts the performance of Olmo 3 Think SFT" ([[olmo-3-stage-chain]] §4.3) |
| SmolLM3 | preference stage only | Qwen3-32B as chosen, Qwen3-0.6B as rejected, both off-policy | no numbers printed for the internal ablation ([[smollm3-stage-chain]]) |
| K2-V2 | SFT | answers "mostly regenerated using GPT-OSS-120B at low, medium and high reasoning effort levels" | reasoning-effort curve in Fig. 19; no teacher-versus-no-teacher comparison ([[k2-v2]] §6.1–6.2) |
| Llama 3.1 | SFT (self-distillation) | K = 10–30 samples per prompt from the best checkpoint of the previous round, ranked by the round's reward model | no ablation reported ([[llama-3-recipe]] §4.2.2) |
| OLMo 2 | preference stage | responses from 20 models, rated by GPT-4o-2024-08-06; highest-rated chosen | no ablation of the rater ([[olmo-2-stage-chain]] §5) |
| Tülu 3 | preference stage | the same off-policy generation scheme, inherited | Table 18 selects the loss, not the generator ([[tulu-3-stage-chain]]) |

**Two placements, two failure modes.** When teacher outputs enter at SFT, the ceiling is the teacher: Olmo 3 reports saturation, that further imitation of Qwen3 32B traces lowers performance, and converts the same traces into the chosen side of preference pairs instead ([[olmo-3-stage-chain]] §4.3). When teacher outputs enter at the preference stage, the failure mode is the opposite — pairs that no longer contrast: Olmo 3 reports that stronger response generators in the OLMo 2 preference pipeline "failed to yield any improvements", attributed to chosen and rejected responses no longer having "meaningful contrast" (§5.5, Table 32), while SmolLM3's pairs deliberately widen the gap by taking chosen from Qwen3-32B and rejected from Qwen3-0.6B.

**Implication for a general-purpose model.** A recipe that imports teacher outputs at SFT inherits the teacher's coverage, including its gaps; a recipe that imports them as the chosen side of a preference pair inherits the teacher's ranking, not its coverage. ch-35 and ch-35a treat the prompt-selection and filtering choices that decide how much of either is inherited.

## §9 Budget per stage, and what an outside estimate can recover

**What is disclosed.**

| Recipe | Pre-training | Long-context | Post-training | Locus |
|---|---|---|---|---|
| DeepSeek-V3 | 2,664K H800 GPU hours (180K per trillion tokens, 2,048-GPU cluster) | 119K | 5K (SFT + RL) | [[deepseek-v3-recipe]] Table 1 |
| DeepSeek-R1 (on top of V3) | — | — | 101K (R1-Zero) + 41K (R1) + 5K (SFT data creation) = 147K | [[deepseek-r1-recipe]] Table 7 |
| SmolLM3 | 384 H100 GPUs for 24 days | not separated | not reported | [[smollm3-stage-chain]] |
| Olmo 3 7B | 5.93T tokens on 512 H100s | 50B on 256 | not reported in GPU hours | [[olmo-core-olmo3-configs]] README |
| Olmo 3 32B | 5.50T on 1024 | 100B on 1024 | not reported | same |
| Tülu 3 8B | not applicable | not applicable | SFT on 32 GPUs for 6 hours | [[tulu-3-stage-chain]] §4.3 |
| Llama 3.1 405B | 3.8 × 10²⁵ FLOPs; BF16 MFU 43% on 8,192 GPUs and 41% on 16,384 | 38% MFU on 16,384 GPUs at sequence 131,072 | not reported | [[llama-3-recipe]] §3.2, Table 4 |
| OLMo 2, Qwen2.5, K2-V2 | not reported | not reported | not reported | [[olmo-2-stage-chain]], [[qwen-2.5-recipe]], [[k2-v2]] |

**Worked example: the post-training share.** DeepSeek-V3's post-training is 5K of 2,788K total H800 GPU hours, which is 0.18% of the run and 0.19% of the pre-training figure alone. Adding R1's 147K to V3's 2,788K gives 5.3% of the V3 budget spent on the entire reasoning pipeline. Neither share measures how much those stages changed the model. V3's 5K hours cover its own SFT and GRPO stages, which are what separate the released chat model from V3-Base, and R1's 147K cover a pipeline whose stage table moves AlpacaEval 2.0 from 24.7 to 87.6 ([[deepseek-r1]] Table 3). Compute share and capability change are separate quantities, and a budget split copied from the compute column alone would allocate under 1% to the stages that set the model's behaviour.

**What an outside estimate recovered, and what it did not.** In January 2025, eleven months before the R1 paper printed GPU hours, an Epoch AI analysis reconstructed the RL cost from the paper's figures: total gradient steps "around 8000", mean completion length about 4,000 tokens, and the two quantities that were not printed, B and G, assumed from earlier DeepSeek work as 1,024 and 64, with the author noting "This is the biggest source of uncertainty in my calculation" ([[epoch-deepseek-r1-compute]]). The result was 6.1e23 FLOP for the first RL phase and "around $1M" for R1 on top of V3. The later disclosure prints 101K H800 GPU hours for R1-Zero, 41K for R1 and 5K for SFT data creation, 147K in total, and prints the rollout size the estimate had to assume: 32 questions per step and 16 sampled outputs per question ([[deepseek-r1-recipe]] v2 B.4.4 Table 7; [[deepseek-r1]] §2.1). Priced at the $2 per H800 hour the estimate itself uses for V3, 147K hours is $294K (derived: 147,000 × 2), so the estimate is about 3.4 times the disclosed figure. The assumption the author named in advance is the one that was wrong, and by more than the cost gap: the assumed B × G = 1,024 × 64 = 65,536 sampled outputs per step against the disclosed 32 × 16 = 512. The other inputs therefore offset most of that ratio — the estimate used 8,000 total gradient steps where the paper gives 10,400 for R1-Zero alone, a 4,000-token mean completion, and pre-training MFU — and neither document states how much each offsets. Source reliability: practitioner evidence, published as an opinionated newsletter issue; it is used here as evidence about recoverability, never as a training setting.

**A reproduction plan template.** For each stage of the chain, the plan needs six cells; the table is filled from the ledger and the empty cells are the plan's risk list.

| Cell | Filled from | If not reported |
|---|---|---|
| Starting checkpoint | the released checkpoint name or path in the launch script | state which public checkpoint you substitute and that the chain now differs |
| Data and size, in tokens | report table or dataset card, normalized as in §1 | derive from examples × mean length, or record "not reported" and pick your own budget explicitly |
| Optimizer and schedule | released config at a pinned commit; then paper table; then paper prose | re-sweep at your size using the published sweep range (§4) |
| Batch, in the unit the code consumes | product of the parallelism flags | do not copy a batch printed without a unit |
| Gate | the evaluation and cadence used for checkpoint selection | define one before the run starts; a stage with no gate has no stopping rule |
| Budget | GPU hours or GPU count × wall clock | estimate from tokens × 6N and state the assumed MFU as an assumption |

## Negative samples and negative feedback

Each recipe here handles negatives at three different points, and the four meanings of "negative" (style §6.1) apply to different stages.

1. **Where negatives come from.** Rejection sampling discards low-ranked or incorrect samples: Llama 3.1 samples K = 10–30 completions per prompt and keeps the best by reward-model score ([[llama-3-recipe]] §4.2.2); DeepSeek-R1's Dev3 stage keeps only correct responses from the stage-1 RL checkpoint, yielding about 600k reasoning samples, and removes chains with mixed languages, long paragraphs or code blocks ([[deepseek-r1-recipe]] B.3.3). These are **negative marginal value**: the discarded samples never enter the loss. Preference stages label negatives by rating (OLMo 2: GPT-4o rates four axes, the highest-rated response is chosen and a random lower-rated one rejected) or by construction (SmolLM3: rejected responses are generated by Qwen3-0.6B). RL stages label them by a verifier: Tülu 3's v(x, y) = 10 for a correct answer and 0 otherwise ([[tulu-3-stage-chain]] §6).
2. **What practice does with them.** Discard at SFT; **negative as gradient** at the preference and RL stages. Length-normalized DPO applies the gradient to the mean token log-probability of the rejected response ([[allenai-olmo3-open-instruct-scripts]] `dpo_utils.py` L83–84, L556–559); group-relative RL gives every below-mean sample a negative advantage.
3. **Mechanism.** For a softmax over logits z, ∂ log p_y / ∂ z_j = 1[j = y] − p_j, so decreasing the likelihood of a sampled sequence removes probability mass that is redistributed toward the currently most likely alternatives, which need not be the correct ones. With Olmo 3's default `centered` advantage normalization, A = r_i − mean(r) over the 8 samples of a prompt, so a single correct sample out of 8 at reward 10.0 gives A = 8.75 for the correct sample and −1.25 for each of the seven others; with `standard` normalization the same group gives 2.646 and −0.378 ([[allenai-olmo3-open-instruct-scripts]] `data_loader.py` L421, L1417–1420). The magnitude of the negative push therefore depends on a framework default that the launch scripts stopped printing after PR #1547.
4. **Evidence with numbers.** Olmo 3.1 RL-Zero improved over 3.0 mainly from raising the completion limit from 12k to 16k tokens and from "not masking truncated sequences"; masking made batch sizes vary, reduced stability, and "without training on overlong negative sequences, completion lengths were higher, on average" ([[olmo-3-stage-chain]] A.6.4). In the other direction, Tülu 3's RLVR at β = 0.01 produced a response consisting of fourteen comma-separated letters "e" with no answer, while β = 0.1 produced a correct answer ([[tulu-3-stage-chain]] §6.2.1) — the negative signal was satisfied by a degenerate output.
5. **Controls the recipes use.** Zero-variance groups are dropped before batching, so all-correct and all-wrong prompts contribute no gradient; prompts solved at a rate of at least 0.875 are excluded from later sampling in two Olmo 3 scripts; the ratio correction is clamped at 2.0 ([[allenai-olmo3-open-instruct-scripts]]). DeepSeek-R1 restricts preference rewards to the final 400 of 1,700 RL steps after observing reward rising while Codeforces performance fell ([[deepseek-r1]] §3.2.2, B.5).
6. **Diagnostics for an end-to-end run.** Log the chosen and rejected log-probabilities separately at the preference stage; log statistics split by advantage sign at the RL stage; carry pass@k at large k, an abstention rate, and the unseen suite of §6 across stage boundaries, because the stage tables in §5 show benchmarks that fall only at the last stage.
7. **Effect on generality.** The one measurement in these sources that separates the two directions is Olmo 3's negative control: training the base model on the RL-Zero mixture with random binary rewards "does not improve performance on any of our benchmark" suites ([[olmo-3-stage-chain]] §6.2). Where the reward is real, the same section reports that a multi-domain run "has improved performance across different domains, but each domain is under-optimized compared to the single-domain setup".

## Recipe

Rows are one stage of one released checkpoint. Units: tokens are tokens seen along one path; batch is stated in the unit the source prints, with the derivation shown; "pairs" are preference pairs; RL batch is prompts × samples per prompt. Status dates: values read on 2026-09-14 to 2026-09-17 at the stated loci.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OLMo-2-1124-7B | 7B | pretrain-stable | tokens; batch × length; peak LR; cosine horizon | 3.90T; 1024 × 4096; 3.0e-4; 5T truncated at 4T | arXiv:2501.00656v3 §2.3, Table 3 | verified 2026-09-15 | §4.1: peaks 3–12e-4 differ by under two points on 9 OLMES tasks after a decay, one seed |
| OLMo-2-1124-13B | 13B | pretrain-stable | peak LR | 9.0e-4 (Table 3) vs 3.0e-4 (released stage-1 YAML) | Table 3; `OLMo2-13B-stage1.yaml` L46 @090253d | conflict | §4.1 prose: "The 13B ran with a higher peak learning rate from the start"; the stage-2 start LR of 9e-5 equals 0.1 × 9e-4 (derived) |
| OLMo-2-1124-7B-SFT | 7B | SFT | peak LR; epochs; batch; length | 2e-5; 2; 128 sequences; 4,096 | arXiv:2501.00656v3 §5, Table 17 | verified 2026-09-15 (max length read from the Tülu 3 SFT mixture setting, not printed in §5) | §5 sweep 1e-5, 2e-5, 3e-5 selects 2e-5; Table 17 (7B, batch 128, linear, warmup 0.3): best row 2 epochs at 1e-5 with sum loss 49.97; 3 epochs at 4e-6 with mean loss 48.25 |
| OLMo-2-1124-7B-DPO | 7B | preference | loss; β; peak LR; preference prompts | `dpo_norm`; 5; 1e-6; 366.7k prompts | §5; `dpo_7b.sh` L28 | conflict | YAML prints 5e-7; §5 sweep selects 1e-6 for the 7B |
| OLMo-2-1124-7B-Instruct | 7B | RL | algorithm; β; LR; data | PPO with RM-initialized value; 0.05; 4e-7; GSM8K + MATH + constraint prompts | §5 | verified 2026-09-15 | §5 sweep β 0.03/0.05/0.07; Table 16: AVG 55.9 → 56.5 over DPO |
| Olmo-3-1025-7B | 7B | pretrain-stable | tokens; batch; peak LR; schedule | 5.93T; 512 × 8,192 = 4,194,304 tokens; 3.0e-4 → 3.0e-5; cosine over 5T stretched to 5.93T | arXiv:2512.13961v2 Table 35, Fig. 3 | verified 2026-09-15 | no ablation of the stretch reported |
| Olmo-3-1025-7B | 7B | mid-train | tokens; batch; LR | 100B; 2,097,152 tokens; 2.074e-4 → 0, warmup 0 | Table 35; `OLMo-3-1025-7B-midtrain.py` L40-43 | conflict | script prints 2.0712e-4; §3.5.4: decontaminated anneal compared against a matched non-decontaminated 100B anneal |
| Olmo-3-1025-7B | 7B | long-context | tokens; length; batch; YaRN | 50B; 65,536; 64 × 65,536 = 4,194,304 tokens; factor 8 from 8,192 | Table 35; `OLMo-3-1025-7B-long-context.py` L37-51 | verified 2026-09-15 | §3.6.3: 34% long / 66% short lowered a subset of OlmoBaseEval by 0.8 points vs 2.5 for 66/34 |
| Olmo 3 7B Think SFT | 7B | SFT | prompts; total tokens; epochs; length; batch; LR | 2,268,468; 45.4B; 2; 32,768; 1,048,576 tokens; 5e-5 | Table 17, Table 47; `7b_think_sft.sh` L7-19 | verified 2026-09-14 | §4.2: LR sweep, evaluation suite, qualitative checks, then a merge of two LR checkpoints |
| Olmo 3 7B Think DPO | 7B | preference | pairs; LR; length; batch; loss; β | 150,000; 8e-8 linear; 16,384; 128; `dpo_norm`; 5 | `7b_think_dpo.sh` L3-40; defaults at d8a7f1c | verified 2026-09-14 | A.6.2: LR and dataset size swept; Table 48 matches |
| Olmo 3 7B Think RL | 7B | RL | prompts × samples; LR; response length; KL β; clip | 64 × 8; 1e-6 constant; 32,768; 0.0; 0.2 / 0.272 | `7b_think_rl_no_pipeline.sh` L20-49 | verified 2026-09-14 | Table 49 matches; no ablation reported |
| Olmo 3 32B | 32B | all base stages | deltas from the 7B | batch doubled at every stage; mid-training run twice and averaged; long-context 50B → 100B; pretrain peak LR 6.0e-4 | Table 35 caption | verified 2026-09-15 | §3.5.4: the mid-training soup gained "nearly a full point" on MCSTEM and 2.9 / 1.6 points on Math over the single runs |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak LR; epochs; batch; length; warmup | 5e-6 linear; 2; 128 sequences; 4,096; 0.03 | Tülu 3 Table 11; `docs/tulu3.md` L39-43 | verified 2026-09-14 | §4.3.2: "training for longer did not yield further improvements" |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | loss; β; LR; batch; epochs | `dpo_norm`; 5; 5e-7; 128 pairs; 1 | Table 20; `docs/tulu3.md` L144-159 | verified 2026-09-14 | Table 18: DPO-norm β 5, 1 epoch → 57.3 vs 55.7 for the SFT base and 55.2 for DPO |
| Llama-3.1-Tulu-3-8B | 8B | RL | algorithm; LR; β; batch; response length; α | PPO; 3e-7; 0.05; 224 responses; 2,048; 10 | Table 21; `docs/tulu3.md` L313-346 | verified 2026-09-14 | Table 21 caption; §6.2.1 Fig. 21: higher KL gives lower average scores |
| Llama-3.1-Tulu-3-70B | 70B | RL | β | 0.07 (caption, launch command) vs 0.7 (§6.4 prose) | Table 21 caption; §6.4; `docs/tulu3.md` L375-383 | conflict | the launch command that produced the checkpoint prints `--beta 0.07` |
| Llama-3.1-Tulu-3-405B | 405B | RL | data; steps; LR; batch | MATH only; stopped at 75 steps; 1e-7; 1,856 | §8.1, Table 35 | verified 2026-09-14 | §8.1: "even with as few as 25 RLVR steps, MATH performance improved by over 5 points" |
| SmolLM3-3B | 3B | pretrain-stable | tokens; batch; LR; schedule; hardware | 11.2T; 2,359,296 tokens; 2e-4; WSD with 2,000 warmup steps and linear decay over the final 10%; 384 H100 GPUs for 24 days | blog "Training Configuration"; stage configs L225-254 | conflict on total (11T / 11.2T / 11.1T) | playbook: batch 2M–4M "minimal impact", 2.36M chosen for throughput; WSD 10% chosen after a cosine comparison |
| SmolLM3-3B | 3B | long-context | tokens; stages; RoPE θ | 100B in 2 × 50B; 4k → 32k → 64k; 1.5M then 5M | blog; released configs | conflict | the 4k → 32k config prints `rope_theta: 2000000.0` |
| SmolLM3-3B | 3B | preference | algorithm; data; length cap | APO; Qwen3-32B chosen, Qwen3-0.6B rejected; 24k tokens | blog "Off-policy model alignment with APO" | verified 2026-09-15; β, LR, batch, epochs not reported | "higher downstream performance in our internal ablations" (no numbers) |
| SmolLM3-3B | 3B | merge | method; weights | soup of APO checkpoints, then linear merge 0.9 (soup) / 0.1 (mid-training checkpoint), MergeKit | blog "Model Merging" | verified 2026-09-15 | "We were able to recover the base model's RULER score on contexts up to 128k tokens" (no numbers printed) |
| Llama 3.1 405B | 405B | pretrain-stable | tokens; compute; peak LR; batch schedule | 15.6T; 3.8 × 10²⁵ FLOPs; 8e-5; 4M → 8M → 16M tokens (§3.4.1 prints the middle step as "8M sequences of 8,192 tokens"; the surrounding values are in tokens) | arXiv:2407.21783v3 §3.4.1, Table 3 | verified 2026-09-14 | §3.2.1: scaling law N*(C) = A C^α with (α, A) = (0.53, 0.29) extrapolates to 402B at this budget |
| Llama 3.1 405B | 405B | long-context | stages; tokens; gate | 6 stages, 8K → 128K, ≈800B tokens | §3.4.2 | verified 2026-09-14 | stage advanced when short-context evaluations recovered and needle-in-a-haystack was solved at that length |
| Llama 3.1 405B | 405B | SFT | LR; steps; loss | 1e-5; 8.5K–9K steps; prompt tokens masked | §4.1.3 | verified 2026-09-14 (batch, epochs not reported) | "work well across different rounds and data mixes" |
| Llama 3.1 405B | 405B | preference | rounds; DPO LR; β; NLL coefficient | 6 rounds; 1e-5; 0.1; 0.2 on chosen | §4.1.4, §4.1.6 | verified 2026-09-14 | §4.1.4: DPO chosen over PPO for less compute and better instruction following; no numbers |
| Llama 3.1 405B | 405B | merge | method | averaging of models from runs with different data or hyper-parameters at the RM, SFT and DPO stages | §4.1.5 | verified 2026-09-14 (weights not printed) | no ablation reported |
| Qwen2.5 Instruct (open weights) | 0.5B–72B | SFT | examples; epochs; length; LR; weight decay; clip | >1M; 2; 32,768; 7e-6 → 7e-7; 0.1; 1.0 | arXiv:2412.15115v2 §4.1 | verified 2026-09-14, except the example count, which is a conflict (§4.1 and Abstract "over 1 million" vs §1 "1 million" vs §4 intro "millions") | no ablation reported; optimizer, warmup and batch not given |
| Qwen2.5 Instruct | 0.5B–72B | preference | loss; pairs; epochs; LR | DPO; ≈150,000; 1; 7e-7 | §4.2 | verified 2026-09-14 (β not given) | no ablation reported |
| Qwen2.5 Instruct | 0.5B–72B | RL | algorithm; samples per query; batch; ordering | GRPO; 8; 2,048 samples per episode; queries ordered by reward variance | §4.3 | verified 2026-09-14; LR, β, ε, temperature, length, episodes not reported | no ablation reported; the query ordering carries the stated aim "to ensure more effective learning" |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | tokens; LR; batch; compute | 14.8T; warmup to 2.2e-4, constant to 10T, cosine to 2.2e-5 over 4.3T, then 2.2e-5 and 7.3e-6; 3072 → 15360 (unit not printed); 2,664K H800 hours | arXiv:2412.19437v2 §4.2, Table 1 | verified 2026-09-14 | no schedule ablation reported |
| DeepSeek-V3 | 671B / 37B act. | SFT | instances; epochs; LR | 1.5M; 2; cosine 5e-6 → 1e-6 | §5.1 | verified 2026-09-14 | Table 9: distilled long-chain data lifts LiveCodeBench-CoT 31.1 → 37.4 on V2.5 |
| DeepSeek-R1-Zero | 671B MoE | RL | LR; KL β; questions × outputs; length; steps; compute | 3e-6; 0.001; 32 × 16; 32,768 then 65,536 after step 8.2k; 10,400 steps; 101K H800 hours | arXiv:2501.12948v2 §2.1, B.4.4 | verified 2026-09-14 | §2.3: AIME 2024 pass@1 15.6% → 77.9% |
| DeepSeek-R1 | 671B MoE | SFT (cold start) | examples; epochs; LR; length; batch | "thousands"; 2–3; cosine 5e-5 → 5e-6; 32,768; 128 | §3, B.3.2, B.4.2 | verified 2026-09-14; exact count not reported | Table 3: Dev1 IF-Eval 46.6 → 71.7, AIME 77.9 → 59.0 |
| DeepSeek-R1 | 671B MoE | SFT (rejection sampling) | samples; start checkpoint | 804,745 (600k reasoning + ≈200k non-reasoning); DeepSeek-V3-Base, not the RL checkpoint | B.3.3, Table 5, Fig. 2 | verified 2026-09-14 | Table 3: Dev2 → Dev3 AlpacaEval 2.0 55.8 → 62.1, Aider-Polyglot 25.6 → 44.8 |
| DeepSeek-R1 | 671B MoE | RL (mixed) | steps; temperature; reward; preference window | 1,700; 0.7; rule + RM + format + language; preference rewards in the final 400 steps | §3.2.2 | verified 2026-09-14 | B.5 Fig. 6: with the helpful RM, reward rises while Codeforces pass@1 falls |
| K2-V2 | 70B | pretrain-stable | tokens; batch; LR; weight decay; steps | 12.25T; 1,200 × 8,192 = 9.8e6 tokens; 1.5e-4 with a 1.5e-6 floor held constant; 0.05; 1.25e6 | arXiv:2512.06201v2 §3.2 | verified 2026-09-17 | 257M pilot on 41.48B tokens: SlimPajama loss 2.815 (cosine d2z), 2.845 (to 10%), 2.817 (linear d2z) |
| K2-V2 | 70B | mid-train + long-context | four stages; tokens; context; RoPE; LR | mid-1 1769B at 8,192 (cosine 1.5e-5 → 6e-6); mid-2/3/4 590B / 229B / 131B at 65,536 / 131,072 / 524,288, constant 6e-6; RoPE 0.5M / 1M / 10M / 10M | Table 3 | verified 2026-09-17 | §5: "mild degradation in stage 4" attributed to the scarcity of long reasoning data |
| K2-V2 | 70B | distill-SFT | tokens per epoch; epochs; batch; length; LR; teacher | 17.4B; 3; 128; 65,536; cosine 2e-5 → 2e-6 with 10% warmup; GPT-OSS-120B | §6.1, §6.3 | verified 2026-09-17 | no teacher-versus-no-teacher comparison reported |

**Starting point for a small general-purpose run.** Every number here is a verified row above, and each carries the conditions under which its source used it. For a 7–8B Llama-family base on a public instruction mixture, the Tülu 3 8B rows give SFT at 5e-6 linear with warmup 0.03 for 2 epochs at 128 sequences of up to 4,096 tokens (run on 32 GPUs for 6 hours), then length-normalized DPO with β = 5 at 5e-7 for 1 epoch at 128 pairs of up to 2,048 tokens, then PPO-based RLVR at 3e-7 with β = 0.05, batch 224 responses, response length 2,048, α = 10 for a correct answer, checkpoints evaluated every 100 steps. For a 7B OLMo-family base the same stages need a re-sweep: OLMo 2 selected SFT 2e-5 and DPO 1e-6 for its own variant of the same mixture, four times and twice the Tülu values. Neither set is stated for a model below 1B, and neither includes a long-context or mid-training stage; for those, the Olmo 3 7B rows give 100B mid-training tokens at 2.097M-token batches and 50B long-context tokens at 65,536 with a 34% long / 66% short mixture.

## Generalization lens

**(a) What increases breadth.**
- Adding a non-reasoning imitation stage after reasoning RL: DeepSeek-R1's Dev3 stage, which adds about 200k non-reasoning samples, raises AlpacaEval 2.0 from 55.8 to 62.1 and Aider-Polyglot from 25.6 to 44.8 ([[deepseek-r1]] §4, Table 3).
- Mixing domains in one RL run rather than training per-domain specialists, at a cost: Olmo 3's mixed run "has improved performance across different domains, but each domain is under-optimized compared to the single-domain setup" ([[olmo-3-stage-chain]] §6.2).
- A merge at the end of the chain, when an earlier checkpoint holds a capability the later stages lost: SmolLM3's 0.9 / 0.1 linear merge recovered the base model's RULER score to 128k ([[smollm3-stage-chain]]); Olmo 3's 32B mid-training soup gained up to 2.9 points on the Math cluster over its input runs ([[olmo-3-stage-chain]] §3.5.4).
- Keeping a small long-context share in short-context stages: 0.1% long-context data in Llama 3.1's SFT mixture, selected by ablation, optimized both short and long benchmarks, and short-only SFT "caused significant long-context regression" ([[llama-3-recipe]] §4.3.4).

**(b) What causes narrowing or forgetting.**
- A stage whose data caps sequence length below the extended context: SmolLM3's APO at 24k tokens, with RULER degradation traced to the reasoning mid-training stage ([[smollm3-stage-chain]]).
- A final RL stage optimizing a subset of the metrics: Olmo 3 Instruct 7B PopQA 20.7 → 14.1 and Safety 89.9 → 87.6 from DPO to RL ([[olmo-3-stage-chain]] Table 26); OLMo 2 32B Safety 91.9 → 85.9 from DPO to Instruct ([[olmo-2-stage-chain]] Table 16).
- Preference rewards left on for a whole RL run: DeepSeek-R1 limits them to the final 400 of 1,700 steps after observing reward rising while Codeforces performance fell ([[deepseek-r1]] B.5).
- Development-suite selection: Tülu 3 reports its SFT mixture choices "overfit to the development evaluations in Precise Instruction Following, and to some extent in Knowledge Recall and Reasoning" ([[tulu-3-stage-chain]] §7.4.1).

**(c) How to measure it for this stage.** Carry one unseen benchmark per skill across every stage boundary, chosen before the run and never inspected during development, as in Tülu 3's Table 3 pairing; report the development and unseen values side by side per stage as in its Table 31. Add a contamination measurement that does not depend on n-gram filters, such as K2-V2's sentence-level memorization probe ([[k2-v2]] Table 4). Declare in the release whether held-out benchmarks were evaluated before release, as OLMo 2 and Olmo 3 do in their table captions. For an RL stage, log pass@k at large k alongside pass@1, since the Olmo 3 RL-Zero prompt filter removes prompts the base solves 8 of 8 times and therefore changes what pass@1 measures ([[olmo-3-stage-chain]] §6.1).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Copying a setting from a report while running released code | Loss curve and evaluation trajectory differ from the published ones with no error | Diff the paper table against the launch script and the framework default at the run's commit; expect disagreement in at least one row ([[allenai-olmo3-open-instruct-scripts-recipe]]) |
| Reading a launch script at the head commit | Fewer flags than the run used; β, epochs and clipping missing | Check the commit history for a defaults-migration commit; for open-instruct, read the parent of 7917d41 |
| Treating a schedule horizon as a token budget | Derived tokens-per-parameter and compute numbers are too high | Compare the horizon against the report's stated tokens seen; the playbook's survey table lists OLMo 2 7B as "5T" where the report states 3.90T ([[smol-training-playbook]]) |
| Budgeting an SFT stage by prompt count | The stage costs several times the planned compute | Normalize to tokens: examples × mean length × epochs (§1); long-chain sets average 5,355 tokens (R1) to about 10,007 (Olmo 3 Think) per item |
| Transferring a learning rate across base families | Underfitting or instability at the same data and stage | Re-sweep; OLMo 2 selected SFT and DPO rates 2–4× the Tülu 3 values for the same mixture ([[olmo-2-stage-chain]] §5) |
| Running a short-context stage after context extension without a control | RULER or needle scores fall while every other benchmark rises | Evaluate long context at every stage boundary, not only after the base stages ([[smollm3-stage-chain]]) |
| Reporting only the final checkpoint's scores | A capability lost at the last stage is invisible | Keep a per-stage table for every benchmark, as in Olmo 3 Table 26 and OLMo 2 Table 16 |
| Selecting checkpoints on the same suite used to choose the data | Development scores rise while unseen scores stall or fall | Maintain an unseen suite and inspect it only after release decisions ([[tulu-3-stage-chain]] §2.1) |
| Estimating an undisclosed stage budget and then citing it as a setting | The estimate propagates into a plan as if it were measured | Label estimates by source type; the Epoch AI R1 estimate was about 3.4× the later disclosed 147K GPU hours ([[epoch-deepseek-r1-compute]]) |

## Check your understanding

1. Every recipe here that pre-trains ends pre-training with a distinct final stage, but the recipes place the long-context stage differently relative to it. Explain what determines whether long-context ability survives to the released checkpoint, and name the two repairs the recipes use.
2. Tülu 3 uses SFT learning rate 5e-6 at 8B and 2e-6 at 70B, while Olmo 3 uses Think RL learning rate 1e-6 at 7B and 2e-6 at 32B. Explain why "learning rate falls with model size" is not the lesson to take from these rows, and state what should be transferred instead.
3. The Olmo 3 Instruct 7B stage table shows PopQA at 16.5 (SFT), 20.7 (DPO) and 14.1 (RL). Give two different causal accounts of the last number, and say what measurement would separate them.
4. A reproduction reads the Olmo 3 7B Think DPO launch script today and finds no `--beta` flag. Explain the two ways this can produce a run that differs from the released checkpoint, and the check that rules both out.
5. DeepSeek-V3 spent 5K of 2,788K H800 GPU hours on post-training, yet its post-training stages account for most of the difference between the base model and the released chat model. Explain why the compute share and the capability share diverge, and what that implies for how a small team should split a fixed budget.
6. Tülu 3's unseen coding benchmark falls from 11.5 to 7.4 across stages at 8B while rising from 12.2 to 21.6 at 70B. Give a causal explanation consistent with both, and state what the 8B result implies for a plan that evaluates only HumanEval.
7. The Epoch AI estimate of R1's RL cost was about 3.4 times the figure the paper later printed, and the author named the assumption responsible in advance. Explain what this says about which quantities a disclosure must include for an outside budget estimate to be usable.
8. K2-V2 releases data, code, logs and checkpoints for a chain that ends at SFT. Explain what a team can and cannot learn from it about the preference and RL stages, and why the report's pass@k analysis is offered as evidence about those stages.

## Connections

- **Previous:** ch-58 — Choosing and Instrumenting a Post-Training Stack to Measure and Protect General Capability. That chapter chooses the framework and the instrumentation; this chapter supplies the stage chains and settings those frameworks have to run.
- **Next:** ch-59 — Capstone: Reproduce One Stage of an Open General-Model Recipe with a Generality Gate. The reproduction plan template of §9 and the disclosure matrix of §3 are the inputs to choosing which stage is reproducible.
- **Dependencies:** ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures (the ledger method and the pre-training rows); ch-32e — Mid-Training, Annealing, and Context-Extension Recipes Side by Side (the mid-training and long-context rows); ch-34 — Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports (the per-report reading these chains normalize); ch-45a — Preference-Optimization and RL Stage Recipes Side by Side (the preference and RL rows); ch-45d — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL (the agentic variants of the same chains).
- **Also used here:** ch-30c (merging), ch-32b and ch-32c (context extension and effective length), ch-35 and ch-35a (distillation placement and filtering), ch-38a (SFT versus RL generalization), ch-47a and ch-48 (unseen suites and contamination), ch-51 (metric noise and go/no-go decisions).

## Sources

- [[olmo-2-stage-chain]] — OLMo 2's pre-training, anneal-and-soup, SFT, DPO and RLVR settings, its stage table, and its held-out suite statement.
- [[olmo-3-stage-chain]] — Olmo 3's base and branch stages, Table 35 schedule values, SFT and DPO sizes, the per-stage Instruct table, and the RL-Zero negative control.
- [[olmo-core-olmo3-configs]] — the released base-stage scripts, the per-stage GPU counts, and the config-versus-paper differences.
- [[allenai-olmo3-open-instruct-scripts]] and [[allenai-olmo3-open-instruct-scripts-recipe]] — Olmo 3 post-training launch values, framework defaults, the PR #1547 migration, and the advantage and filtering behaviour used in the negative-feedback section.
- [[tulu-3-stage-chain]] — Tülu 3's SFT, DPO and RLVR settings across 8B, 70B and 405B, the DPO algorithm ablation, the development/unseen split, and Table 31.
- [[open-instruct-allenai-recipes]] and [[open-instruct-allenai-recipes-recipe]] — the launch commands for Tülu 3 and OLMo 2 and the conflicts between them and the papers.
- [[smollm3-stage-chain]] — SmolLM3's three pre-training stages, long-context stages, reasoning mid-training, APO, and the merge that repaired RULER.
- [[smol-training-playbook]] — how the SmolLM3 budget was set, the schedule and batch ablations, and the survey-table error that motivates §1's normalization.
- [[llama-3]] and [[llama-3-recipe]] — Llama 3.1's pre-training schedule, six-stage long-context extension and its gate, the six post-training rounds, DPO settings, and the averaging step.
- [[qwen-2.5]] and [[qwen-2.5-recipe]] — Qwen2.5's stage chain and the list of settings the report does not print.
- [[deepseek-v3]] and [[deepseek-v3-recipe]] — DeepSeek-V3's pre-training schedule, context-extension phases, distillation-based SFT, and the per-stage GPU-hour table.
- [[deepseek-r1]] and [[deepseek-r1-recipe]] — the four-stage R1 pipeline, its per-stage benchmark table, the distillation-versus-RL comparison, and the disclosed GPU hours.
- [[epoch-deepseek-r1-compute]] — an outside estimate of the R1 stage budgets, used as evidence about recoverability from a partial disclosure.
- [[k2-v2]] — a 360-open 70B chain from pre-training through four mid-training stages to a distilled SFT, with a hyper-parameter transfer procedure and a memorization measurement.
- [[apertus]] — a five-stage pre-training chain with a declared held-out suite and a cooldown-ablation method for selecting mixtures.
