<!-- chapter: ch-14a
     track: pretraining
     kind: content
     title: Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures
     deps: [ch-14]
     sources: [[llama-3]], [[llama-3-recipe]], [[llama-3-1-model-card]], [[deepseek-v3]], [[deepseek-v3-recipe]], [[qwen-2.5]], [[qwen-2.5-recipe]], [[olmo-2-pretraining-recipe]], [[olmo-2-official-configs]], [[olmo-3-pretraining-recipe]], [[olmo-core-olmo3-configs]], [[smollm2]], [[smollm3-blog-pretraining]], [[smollm3-training-configs]], [[smol-training-playbook]], [[gemma-3]], [[gemma-2]], [[minicpm]], [[minitron-approach]], [[nanochat]], [[cooldown-scaling-beyond-fixed-durations]], [[beyond-chinchilla-inference-scaling]], [[large-batch-training-noise-scale]], [[data-constrained-scaling]]
     figures: figures/recipe-ledger-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures

> **Core insight.** Open pretraining runs differ by more than two orders of magnitude in tokens per parameter: 38.5 for Llama 3.1 405B and 22 per total parameter for DeepSeek-V3, against 6,471 for SmolLM2 1.7B (derived from the printed token counts and sizes). The schedules in §4 fall into three families: cosine, cut or stretched to a new horizon and followed by a linear decay to zero (Llama 3.1, OLMo 2, Olmo 3); warmup-stable-decay (SmolLM2, SmolLM3, MiniCPM, nanochat), with a linear decay over 10–11% of steps in SmolLM2 and SmolLM3; and DeepSeek-V3's constant phase, cosine, and two constant steps, and every staged recipe that prints its mixtures raises the math share in its last and shortest stage (all except OLMo 2 also raise code). Few of these values have a published ablation at the released size, and reading two artifacts per value exposes nine conflicts, including a 3× difference in the OLMo 2 13B peak learning rate between the report and the released config.
>
> **Guideline.** When copying a pretraining setting, record the exact checkpoint, the locus, the unit, and a status (verified, not reported, conflict, derived), and check the value against a quantity derived from a second artifact, such as the start learning rate of the next stage's config, because that check exposed the OLMo 2 13B and Olmo 3 7B conflicts in §1 and §4. When the final token budget may change or one run must serve several budgets, use warmup-stable-decay (WSD) with a decay over 10–20% of steps, because a 10% decay was sufficient and a 2.5% decay was not at 0.036B ([[minicpm]] §4.3) and a 20% linear cooldown scored 46.20 against 46.26 for cosine on an 8-task average at 1B and 100B tokens ([[cooldown-scaling-beyond-fixed-durations]] Table 4); otherwise use cosine with its horizon equal to the planned token count. When a stage changes the mixture and the learning-rate phase at the same time, attribute a gain to the data only after a run that separates the two, because SmolLM2's math average rose from 7.27 to 22.07 in a stage that changed both ([[smollm2]] Table 3).

## Why this chapter matters for a general-purpose model

The pipeline is pre-training → mid-training → SFT → preference optimization → RL → evaluation. Pre-training determines the knowledge and skills that every later stage starts from. Four settings of that run are compared here: the token budget, the global batch, the learning-rate schedule, and the data mixture of each stage.

The measurable problem has three parts:
1. **A small run needs starting values.** A team without the budget to sweep all four settings copies them from published runs, so an error in a copied value changes the whole run.
2. **Values are scattered and sometimes inconsistent.** One run's settings appear in report prose, tables, figure captions, released configs, model cards, and blog posts. In this chapter's Recipe table, 9 settings differ between two artifacts of the same team or within one artifact.
3. **The same word names different quantities.** "Tokens" can mean dataset size, a schedule horizon, or tokens seen; "batch size" can mean sequences or tokens; "parameters" can mean total, activated, non-embedding, or a code-defined subset.

This chapter teaches the recipe-ledger method that ch-32e (mid-training and context extension), ch-45a (preference and RL), ch-45d (agentic recipes), and ch-58a (end-to-end recipes) reuse. ch-03 covers the mechanics of schedules and batch size, ch-08a covers how scaling laws set a budget, and ch-13 covers how mixture weights are chosen. This chapter compares what released runs did and how each value was checked.

## §1 The recipe ledger: where settings live and how they are verified

**Definition.** A recipe ledger row records one setting of one checkpoint: model, size, stage, setting, value, source location, status, and the evidence that selected the value. The status is one of four labels:
- **verified**: read at the stated locus on a stated date;
- **not reported**: absent from the places that were checked, which are listed;
- **conflict**: two artifacts print different values, and one row per artifact is kept;
- **derived**: computed from verified inputs, with the formula shown.

**Problem.** A value copied from one artifact can differ from the value that trained the checkpoint. The OLMo 2 13B peak learning rate (LR) is 9.0 × 10^−4 in the report and 3.0e-4 in the released stage-1 config, a factor of 3 ([[olmo-2-pretraining-recipe]] Table 3; [[olmo-2-official-configs]]). A secondary survey table lists OLMo 2 7B as trained on "5T" tokens ([[smol-training-playbook]]); 5T is the cosine horizon in OLMo 2 Table 3, while the report states 3.90T pre-training tokens and 4.05T in total (§2.3).

**Mechanism.**
1. Fix the checkpoint by release name (OLMo-2-1124-7B, Olmo-3-1025-7B, SmolLM3-3B), not by family.
2. Read the report: body, tables, figure captions, and appendix.
3. Read the released configs at a pinned commit and read their commit history, because configs are edited after release. The SmolLM3 configs were edited on 2025-08-03 by a commit titled "fix nb of steps in early configs", after the July 8, 2025 release ([[smollm3-training-configs]]).
4. Normalize units: tokens per step = sequences per step × sequence length.
5. Derive a cross-check from a second artifact: warmup tokens, the LR at a stage boundary, or the token position of a stage boundary.
6. Assign a status. For a conflict, keep one row per artifact and state which one the derived check supports. The default preference order is pinned config, then paper table, then paper prose, then blog.

**Formula.** OLMo's cosine schedule with linear warmup, as implemented in `olmo/optim.py` L694-709 ([[olmo-2-official-configs]]):

```python
def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
    max_steps = max_steps if self.t_max is None else self.t_max
    eta_min = initial_lr * self.alpha_f
    if step < self.warmup_steps:
        return self._linear_warmup(initial_lr, step, self.warmup_steps)
    elif step >= max_steps:
        return eta_min
    else:
        step = step - self.warmup_steps
        max_steps = max_steps - self.warmup_steps
        return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2
```

In symbols, η(t) = η_min + (η_max − η_min) · (1 + cos(π (t − t_w) / (t_max − t_w))) / 2 for t_w ≤ t < t_max, and η(t) = η_min for t ≥ t_max, with η_min = α_f · η_max. Here t is the position in the unit set by the config (tokens seen when `units: tokens`), η_max the peak LR, t_w the end of warmup, t_max the schedule horizon, and α_f the final fraction of the peak.

**Worked example 1: OLMo 2 7B stage boundary (verified, with a residual).** The stage-1 YAML sets η_max = 3.0e-4, α_f = 0.1, `t_warmup: 8388608000`, and `t_max: 5e12` in tokens ([[olmo-2-official-configs]] L46, L57-61).
1. Warmup check: 2,000 steps × 1,024 sequences × 4,096 tokens = 8,388,608,000 tokens, equal to `t_warmup`, so the report's "2000 steps" and the YAML agree.
2. The stage-2 YAML loads step 928,646, which is 928,646 × 4,194,304 = 3.895T tokens.
3. Progress: (3.895e12 − 8.39e9) / (5e12 − 8.39e9) = 0.7786; cos(π × 0.7786) = −0.7678; (1 − 0.7678) / 2 = 0.1161.
4. η = 3.0e-5 + 2.7e-4 × 0.1161 = 6.135e-5.
5. The stage-2 YAML starts its linear decay at 6.1499e-5, which is the cosine value at step 928,000, 646 steps before the load step. The difference is 0.24%, and the files do not explain it. Status: schedule verified; residual recorded.

**Worked example 2: OLMo 2 13B peak LR and warmup (conflict).** Table 3 prints a 13B peak of 9.0 × 10^−4 and 2,000 warmup steps. The released stage-1 YAML prints 3.0e-4 and the same `t_warmup: 8388608000` tokens as the 7B ([[olmo-2-official-configs]] L46, L59).
1. The stage-2 YAML loads step 596,057 and starts at 9e-5. Step 596,057 × 2,048 × 4,096 = 5.0001T tokens, past the 5T horizon, where the cosine returns α_f · η_max.
2. A 9e-4 peak gives 0.1 × 9e-4 = 9e-5, the stage-2 start. A 3e-4 peak gives 3e-5.
3. The report's §4.1 states that "The 13B ran with a higher peak learning rate from the start" ([[olmo-2-pretraining-recipe]]).
4. At the 13B batch, 8,388,608,000 tokens ÷ (2,048 × 4,096) = 1,000 steps, not the 2,000 of Table 3.

Status: two conflicts. The derived boundary check and the prose support 9.0e-4; the warmup token count in the 13B YAML equals the 7B value. The interpretation that the released 13B stage-1 YAML is not the configuration that trained the checkpoint is this chapter's (Interpretation).

**Conditions and limits.** A derived check requires reading the framework code that consumes the config, because the unit of t (steps or tokens) and the horizon source (`t_max` or a trainer property) differ between codebases (§4, Worked example A). A check at a point past the horizon, as for 13B, identifies η_max only through α_f; it would not identify it if the stage had stopped before the horizon and α_f were also uncertain.

**Implication for a general-purpose model.** A stable-phase LR copied from the 13B YAML would be one third of the value that trained the released model. In OLMo 2's own 7B experiments, peaks from 3e-4 to 12e-4 ended less than 2 points apart on 9 OLMES validation tasks after a decay (§4 Evidence), but that result was measured at one size, on one suite, from 300B-token checkpoints, so the ledger records the conflict instead of choosing one value without a note.

## §2 Token budgets and tokens per parameter

**Definition.** D is the number of tokens seen along one training path to the released checkpoint. N is a parameter count under a named convention: total, activated per token (mixture-of-experts), non-embedding, or a code-defined subset. The tokens-per-parameter ratio is r = D / N. Training compute is approximated as C ≈ 6 N D floating-point operations (FLOPs), with 6N FLOPs per training token ([[beyond-chinchilla-inference-scaling]] §2 Eq. 3).

**Problem.** Reports that train beyond the 20 tokens per parameter of Hoffmann et al. (quoted in [[minicpm]] §4.5) justify the choice by inference cost, but r is comparable across reports only when D and N use the same convention and the same stopping point.

**Mechanism.**
1. Choose D and state whether mid-training and context-extension tokens are included and whether averaged runs are counted once or several times.
2. Choose the N convention and write it next to the ratio.
3. Compute r, and where compute is printed, check C ≈ 6ND.
4. Identify the released checkpoint on the token axis, which is not always the end of the run.
5. Record the rationale that the source gives for its budget.

**Worked example.** Llama 3.1 405B: 6 × 405e9 × 15.6e12 = 3.79e25 FLOPs, matching the printed 3.8 × 10^25, and r = 15.6e12 / 405e9 = 38.5 ([[llama-3-recipe]]). DeepSeek-V3: 14.8T tokens with 671B total and 37B activated parameters gives r = 22.1 per total parameter and 400 per activated parameter; the conventions differ by a factor of 18.1 ([[deepseek-v3-recipe]]). OLMo 2 7B: the report's 4.05T counts three 50B anneal runs that were averaged; one path to any of them sees 3.90T + 50B = 3.95T tokens, 2.5% fewer ([[olmo-2-pretraining-recipe]] §2.3).

| Checkpoint | D (tokens seen) | N convention and value | r (derived) | Input status | Budget rationale stated by the source |
|---|---|---|---|---|---|
| Llama 3.1 405B | 15.6T | total, 405B | 38.5 | verified ([[llama-3-recipe]]) | "approximately compute-optimal size for our training budget" (§1) |
| Llama 3.1 8B | "15T+" (model card cell shared by 8B, 70B, 405B) | total, 8B (release name) | ≥ 1,875 | lower bound; report gives no per-size count ([[llama-3-1-model-card]]) | smaller models trained "much longer than is compute-optimal" and "perform better than compute-optimal models at the same inference budget" (§1) |
| Qwen2.5 0.5B, 1.5B, 3B | not reported per size (18T pre-training data) | — | — | not reported ([[qwen-2.5-recipe]]); the OLMo 2 Table 6 caption notes that the developers "declined to disclose exact token counts for each model size" ([[olmo-2-pretraining-recipe]]) | 3B, 14B, 32B are "more cost-effective for resource-limited scenarios" ([[qwen-2.5]] §1) |
| DeepSeek-V3 | 14.8T | total 671B; activated 37B | 22.1; 400 | verified ([[deepseek-v3-recipe]]) | Multi-head Latent Attention (MLA) "for efficient inference" and DeepSeekMoE "for cost-effective training" ([[deepseek-v3]] §2) |
| OLMo 2 7B | 3.95T one path (4.05T reported) | total, 7B (release name) | 564 | verified inputs ([[olmo-2-pretraining-recipe]]) | none stated |
| Olmo 3 7B | 5.93T + 100B + 50B = 6.08T | total, 7B (release name) | 869 | verified inputs ([[olmo-3-pretraining-recipe]] Table 35) | none stated |
| SmolLM2 1.7B | 11T | total, 1.7B | 6,471 | verified ([[smollm2]] §4) | "performance gains and reduced inference costs make extended training a worthwhile trade-off" (§4) |
| SmolLM3 3B | 11.2T (blog stages paragraph) | total, 3B (release name) | 3,733 | conflict on D (§4 Worked example C) | 3B "small enough to enable super-fast inference"; "384 H100s for roughly a month" gave 11T at ~30% model FLOPs utilization ([[smol-training-playbook]]) |
| Gemma 3 1B / 4B / 27B | 2T / 4T / 14T, text and image tokens | embedding + non-embedding, no vision encoder: 1.000B / 3.884B / 27.016B | 2,000 / 1,030 / 518 | verified inputs ([[gemma-3]] Table 1, §2.2) | "The increase in tokens accounts for the mix of images and text" (§2.2) |
| MiniCPM-2.4B | 1.1T | non-embedding, 2,442,057,984 | 450 | verified ([[minicpm]] Table 2) | 0.036B matched 0.17B with "∼ 4 times" training compute, "saving ∼ 5 times per inference call" (§4.3) |
| nanochat d24 speedrun | set by the ratio | transformer matrices + lm_head | 8 (target) | conflict: comment calls 10.5 the default, code default is 12 ([[nanochat]]) | "slightly undertrained to beat GPT-2" (speedrun.sh L66) |

**The released checkpoint is not always the end of the run.** SmolLM2's base model starts context extension from "an intermediate checkpoint from stage 4 (before the final 75 billion tokens of training)" ([[smollm2]] §4.6). MiniCPM does "not utilize the final checkpoints" of its decay stage and fine-tunes from an earlier one ([[minicpm]] §6.4). OLMo 2 releases an average of three (7B) or four (13B, 32B) anneal runs ([[olmo-2-pretraining-recipe]] §2.3). A ledger row for D therefore names the path, not only the total.

**Evidence.** For a model of 30B-Chinchilla quality serving 10^13 inference tokens, a 13.6B model trained on 2.84× the data uses 28% fewer total FLOPs, and 47 runs from 150M to 6B parameters (up to 10,000 tokens per parameter for the 150M model) showed no loss plateau ([[beyond-chinchilla-inference-scaling]] §2, §4). MiniCPM's WSD scaling fit gives a data size "192 times larger than the model size on average" ([[minicpm]] §4.5). Status: **Result (single study)** each; ch-08a treats the allocation objective.

**Conditions and limits.**
- A token is not a fixed amount of text. Vocabularies are 49,152 (SmolLM2 §4.1), 128,000 (Llama 3, [[llama-3-recipe]]), 128,256 (SmolLM3 config), and 262k (Gemma 3 §2.2, whose Table 1 caption says 256k).
- Gemma 3's D includes image tokens, and nanochat's N excludes embedding parameters by design.
- No row above has an ablation of its own budget. The budgets were set by available compute (SmolLM3) or by a scaling-law fit plus an inference argument (Llama 3).

**Implication for a general-purpose model.** At a fixed model size, a larger D contains more occurrences of rarely seen facts; ch-12a §7 relates answer accuracy to the number of documents that contain a fact and §3 covers the capacity limit per parameter, and ch-08a §8 reports the plasticity cost measured for over-trained models. A row that states only "7B, 4T tokens" cannot support a comparison between two budgets.

## §3 Global batch in tokens and batch schedules

**Definition.** The global batch B is the number of tokens in one optimizer step:

```
B = n_seq · L = dp · accum · micro · L
```

where n_seq is the number of sequences per step, L the sequence length, dp the data-parallel degree, accum the gradient-accumulation steps, and micro the per-device micro-batch in sequences. A **batch schedule** changes B during training.

**Problem.** Reports print B in instances (OLMo 2 Table 3), in tokens (SmolLM2 Table 6), or without a unit (DeepSeek-V3 §4.2). A stage boundary given in tokens cannot be located in a step-based config without B.

**Worked example: SmolLM3 stage boundaries.** The stage-1 config sets dp 192, tensor-parallel degree tp 2, pipeline-parallel degree pp 1, accumulation 1, micro-batch 3, and sequence length 4,096 ([[smollm3-training-configs]] L225-231, L250-254).
1. B = 192 × 1 × 3 × 4,096 = 2,359,296 tokens, the blog's "2.36M". GPUs = dp × tp × pp = 384, the blog's "384 H100 GPUs".
2. The configs start "stable stage 2" at step 3,450,001 and the "decay stage" at step 4,198,001, and end at step 4,720,000.
3. Token positions: 3,450,000 × 2,359,296 = 8.14T; 4,198,000 × 2,359,296 = 9.90T; 4,720,000 × 2,359,296 = 11.14T.
4. The blog places the stages at 8T, 10T, and 11.1T and a decay over "the final 10%" ([[smollm3-blog-pretraining]]); the config decay covers 522,000 / 4,720,000 = 11.06% of steps. Status: conflict on the decay start (10T blog; 9.90T derived from the config).

**Batch ledger.**

| Checkpoint | Global batch | Sequence length | Schedule of B | Status |
|---|---|---|---|---|
| Llama 3.1 405B | 4M → "8M sequences of 8,192 tokens" → 16M tokens | 4,096 → 8,192 | double after 252M tokens, double again after 2.87T | verified ([[llama-3-recipe]] §3.4.1) |
| DeepSeek-V3 | 3072 → 15360 (unit not printed) | 4K | "gradually increased" over the first 469B tokens | verified; 12.6M → 62.9M tokens is derived with 4,096-token sequences, and [[smol-training-playbook]] describes it as one step |
| OLMo 2 7B; 13B | 1,024 × 4,096 = 4.19M; 2,048 × 4,096 = 8.39M | 4,096 | constant | verified ([[olmo-2-pretraining-recipe]] Table 3) |
| Olmo 3 7B; 32B | 4,194,304; 8,388,608 tokens | 8,192 | halved for mid-training | verified ([[olmo-3-pretraining-recipe]] Table 35) |
| SmolLM2 1.7B | 2M tokens | 2,048 | constant | verified ([[smollm2]] Table 6) |
| SmolLM3 3B | 2,359,296 tokens | 4,096 | constant | verified, product derived ([[smollm3-training-configs]]) |
| MiniCPM-2.4B; 1.2B | 3.93 million (§6.2) or 4M (Table 2); 2M → 4M | not printed in these loci | 1.2B ramps | verified ([[minicpm]]) |
| Qwen2.5; Gemma 3 | not reported | — | — | not reported (Qwen2.5 fits B_opt(N, D) but prints no values, [[qwen-2.5-recipe]]) |

**Evidence for the chosen values.**
- Llama 3 405B used "a lower batch size early in training to improve training stability" and "observed few loss spikes", without numbers ([[llama-3-recipe]]). **Result (single study)**, no ablation.
- The critical batch size, at which a run needs twice the minimum number of optimizer steps and twice the minimum number of examples to reach a target loss, "typically increases by an order of magnitude or more over the course of training" in the 8 tasks the paper measures ([[large-batch-training-noise-scale]] §2.3, §3). The playbook uses this growth to explain batch warmup ([[smol-training-playbook]]); none of the runs above measured its own critical batch.
- SmolLM3 "tested values from 2M to 4M tokens but found minimal impact on the loss or downstream performance" and chose 2.36M for throughput ([[smol-training-playbook]]); the text prints no numbers.
- Olmo 3's 32B used twice the 7B peak LR, "somewhat compensated for by the larger batch size of the 32B (8M tokens vs. 4M tokens per batch)" ([[olmo-3-pretraining-recipe]] Fig. 4 caption).

**Scaling rule in released code.** nanochat sets B, LR, and weight decay from the token horizon ([[nanochat]] base_train.py L271-302):

```python
predicted_batch_size = B_REF * batch_size_ratio ** 0.383
total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
```

Here `B_REF` = 2^19 tokens, `batch_size_ratio` = D / D_ref with D_ref the horizon of a depth-12 reference model, and `batch_ratio` = B / B_REF. Worked example with D = 16 · D_ref: 2^19 × 16^0.383 = 1,516,170 tokens; log2 = 20.53, which rounds to 21, so B = 2,097,152. The LR factor is √4 = 2, and the weight-decay factor is 2 × 1/16 = 0.125. The code comment marks the Muon LR rule "not studied carefully, assumption!". Status: practitioner evidence; the referenced experiments were not read.

**Conditions and limits.** None of the ramps above has a published comparison against a constant batch at equal tokens. MiniCPM notes that enlarging the batch "might have a similar effect as decreasing learning rate" ([[minicpm]] §6.4), so a batch schedule and an LR schedule are not independent ledger rows.

**Implication for a general-purpose model.** A constant batch of 2M–4.2M tokens covers the small and 7B runs above (SmolLM2, SmolLM3, OLMo 2 7B, Olmo 3 7B). A ledger row without the sequence length cannot reproduce B.

## §4 Learning-rate schedules: peak, final, warmup, and decay shape

**Definitions.** WSD as defined for MiniCPM ([[minicpm]] §4.2, Eq. 1):

```
WSD(T; s) = (s / W)·η          for s < W
          = η                  for W < s < T
          = f(s − T)·η         for T < s < S
```

where s is the step, W the end of warmup, T the end of the stable stage, S the final step, η the maximum LR, and f a decreasing function with 0 < f ≤ 1 (linear to 0 in SmolLM2 and SmolLM3). The **decay share** is (S − T) / S. A **multi-step** schedule lowers the LR in discrete drops (DeepSeek-V3 combines a cosine with two constant steps). A **truncated** cosine stops before its horizon; a **stretched** cosine continues from the midpoint of one horizon along a longer one (Olmo 3 7B).

**Problem.** A cosine schedule ties the LR at every step to one planned horizon. When a run is stopped early, extended, or restarted with a new horizon, the final LR can differ from the value printed for the plan, and a checkpoint taken mid-run is not comparable with a finished run.

**Schedule ledger.**

| Checkpoint | Peak LR | Warmup | Shape | Final LR | Stop point | Status |
|---|---|---|---|---|---|---|
| Llama 3.1 405B | 8e-5 | 8,000 steps | cosine to 8e-7 over 1,200,000 steps; final 40M tokens linear to 0 | 0 after anneal | ≈1,154,407 steps, 96% of horizon (derived, below) | verified; stop derived ([[llama-3-recipe]]) |
| DeepSeek-V3 | 2.2e-4 | 2K steps | constant to 10T; cosine to 2.2e-5 over 4.3T; 2.2e-5 for 333B; 7.3e-6 for 167B | 7.3e-6 | decay starts at 67.6% of tokens (derived) | verified ([[deepseek-v3-recipe]]) |
| OLMo 2 7B | 3.0e-4 | 2,000 steps | cosine to 10% over 5T, cut at 3.895T; linear to 0 over 50B | 0 | cut at 77.9% of the horizon (derived) | verified ([[olmo-2-official-configs]]) |
| OLMo 2 13B | 9.0e-4 (Table 3) / 3.0e-4 (YAML) | 2,000 steps / 1,000 steps (YAML tokens) | cosine to 10% over 5T; linear to 0 over 100B | 0 | full horizon | conflict (§1) |
| Olmo 3 7B | 3.0e-4 | 2,000 steps | cosine (5T horizon) to its midpoint; half-cosine (7T horizon) to 5.93T | 3.0e-5 (Table 35) / 3.93e-5 (derived) | one epoch | conflict (Worked example A) |
| Olmo 3 32B | 6.0e-4 | 2,000 steps | cosine over 5.93T, truncated at 5.5T | 6.0e-5 (Table 35) / 6.210e-5 (Fig. 4) / 6.70e-5 (derived) | 92.7% of horizon (derived) | conflict (Worked example B) |
| SmolLM2 1.7B | 5.0e-4 | 2,000 steps | WSD, linear decay | 0 | decay 10T–11T, 10% of steps | verified ([[smollm2]] §4.1, §4.5) |
| SmolLM2 360M; 135M | 3.0e-3 | not reported | WSD | not reported | 20% decay | verified ([[smollm2]] §6) |
| SmolLM3 3B | 2e-4 | 2,000 steps | WSD, linear decay | 0 | 10% (blog) / 11.06% (config) | conflict (§3, Worked example C) |
| MiniCPM-2.4B | 0.01 (Tensor Program scaling) | not printed in these loci | WSD, exponential decay with T = 5,000 steps (20B tokens) | not printed | stable "around 1T" | verified ([[minicpm]] §6.2) |
| nanochat defaults | per parameter group | 40 steps | constant, then linear over the last 65% of steps | 5% of peak | 65% of steps | verified ([[nanochat]] L67-69) |
| Qwen2.5; Gemma 3 | not reported | — | — | — | — | not reported |

**Worked example A: Olmo 3 7B stretched schedule (conflict).** Table 35 prints a final pre-training LR of 3.0 × 10^−5, and Figure 3 states that the second half of a 5T cosine was stretched "to reach a target length of one epoch (5.93T tokens)" with a final LR of "10% of the peak" ([[olmo-3-pretraining-recipe]]). The pinned OLMo-core scripts show how the stretch was configured ([[olmo-core-olmo3-configs]]):
1. Part 1 runs `CosWithWarmup(warmup_steps=2000)` with `max_duration` 5T tokens, so the trainer horizon is ceil(5e12 / 4,194,304) = 1,192,093 steps. Its hard stop at step 597,046 is the cosine midpoint, (2,000 + 1,192,093) / 2 = 597,046.5 rounded down, where the LR is 1.65e-4.
2. Part 2 runs `HalfCosWithWarmup` with warmup 597,046 and `max_duration` 7T tokens (horizon 1,668,931 steps), and `hard_stop` at one epoch. The scheduler receives `trainer.max_steps`, which comes from `max_duration`, not from `hard_stop` (trainer.py L472-476, L521-527).
3. Part 2 loads step 596,047, 999 steps before part 1's hard stop; the files do not explain the gap. Those 999 steps fall in the warmup branch, which ramps linearly toward the cosine midpoint value η_min + (η_max − η_min) / 2 = 1.65e-4 and returns 1.647e-4 at the load step (scheduler.py L536-558).
4. The one-epoch stop is step 1,413,814 (5.930T tokens), the checkpoint that mid-training loads. The half-cosine argument there is 0.8810 of its span; cos(π × 0.8810) = −0.9309; η = 3.0e-5 + 2.7e-4 × (1 − 0.9309) / 2 = 3.93e-5, assuming the restored trainer keeps its step counter.
5. The mid-training script sets LR = 2.0712e-4 while Table 35 prints 2.074 × 10^−4. Table 35's own "peak training temperature" column equals LR² / batch tokens: (2.074e-4)² / 2,097,152 = 2.051e-14, the printed value, while 2.0712e-4 gives 2.046e-14. The table is internally consistent, so the disagreement is between the table and the script.

**Worked example B: Olmo 3 32B final LR (three values).** Figure 4 describes a cosine over one epoch (5.93T tokens) truncated at 5.5T and states "the real final learning rate is 6.210 × 10^−5"; Table 35 prints 6.0 × 10^−5. With B = 8,388,608 tokens, the horizon is 706,912 steps and the stop is step 655,651. Progress = (655,651 − 2,000) / (706,912 − 2,000) = 0.9273; cos(π × 0.9273) = −0.9740; η = 6e-5 + 5.4e-4 × 0.0130 = 6.70e-5. A final LR of 6.210e-5 would require a horizon near 5.73T tokens (derived). The released 32B script sets `max_duration=Duration.epochs(1)` and contains no stop at 5.5T ([[olmo-core-olmo3-configs]] L107). Status: conflict among a table, a caption, and the caption's own description.

**Worked example C: planned versus final schedule in the SmolLM3 configs.** Before commit 4bc9468, stage1_8T.yaml set `lr_decay_starting_step: 2600000`, `lr_decay_steps: 600000`, and `train_steps: 3200000` ([[smollm3-training-configs]]). At 2,359,296 tokens per step, that plan decays from 6.13T to 7.55T tokens, an 18.75% decay share. After the commit, all stage files carry the final schedule: decay from 9.90T to 11.14T (11.06%). Before the commit, the stage-1 file described a different horizon from the later stage files, and whether the run was launched with the earlier values is not stated. The blog's boundaries (8T, 10T, 11.1T) lie within 0.14T of the post-commit values and far from the pre-commit plan, which suggests the run followed the final schedule (Interpretation). Status: derived.

**Worked example D: Llama 3.1 405B stop point (derived).** The report gives a 1,200,000-step horizon but not the stop step. If all 15.6T tokens ran under the stated batch schedule: 252M / 4M = 63 steps; (2.87T − 252M) / 8M = 358,718.5 steps; (15.6T − 2.87T) / 16M = 795,625 steps. The sum is 1,154,407 steps, 96.2% of the horizon, where the cosine gives 1.09e-6 before the final 40M-token anneal to 0. This assumes "8M" means tokens and that 15.6T includes the long-context stage; the report states neither.

**Worked example E: one stable run serving three budgets.** A team wants base models at 1T, 2T, and 4T tokens. With cosine, each budget needs its own run: 1 + 2 + 4 = 7T tokens. With WSD and a 10% decay, one stable run goes to 3.6T, and decays branch from its checkpoints at 0.9T, 1.8T, and 3.6T for 0.1T, 0.2T, and 0.4T. The total is 3.6T + 0.7T = 4.3T tokens, 39% fewer (derived). MiniCPM uses this property to measure scaling laws "with linear cost (O(mC))" instead of O(m²)C ([[minicpm]] §4.5). The saving requires that the three budgets share the peak LR, the batch, and the stable-stage data.

**Evidence.**
- MiniCPM, 0.036B models, stable checkpoints at 40N, 60N, and 80N tokens: "a decay of 10% of the total tokens is sufficient to achieve the best results, while a decay of 2.5% of total tokens falls short" ([[minicpm]] §4.3). **Result (single study).**
- 1B model on 100B FineWeb tokens, 8-task aggregate: 46.26 (cosine to 10%), 45.88 (cosine to 0), 46.23 ((1-sqrt) 20% cooldown), 46.20 (linear 20%) ([[cooldown-scaling-beyond-fixed-durations]] Table 4). With MiniCPM this is **Replicated** for models up to 1B: a 10–20% decay matches cosine.
- SmolLM3's team compared cosine with WSD at 10% and 20% decay and reports "similar final performance across all three configurations"; cosine was better "during the stable phase" ([[smol-training-playbook]]). No numbers are printed.
- OLMo 2, 7B setting: checkpoints at 300B tokens with peak LRs 3e-4 to 12e-4, decayed over 50B tokens, scored 62.5 to 64.1 on 9 OLMES multiple-choice validation tasks, and the caption states "less than two points across all variants"; after 2T tokens and a 100B decay, 3e-4 and 6e-4 scored 73.8 and 73.9, with 6e-4 "2.8 points better" on GSM8K ([[olmo-2-pretraining-recipe]] Table 8). **Result (single study)**, no seeds reported.
- OLMo 2 cut its 7B cosine at 4T tokens based on "Previous experience with OLMo-0424" that this costs "little loss of performance", with no numbers (§4.1).
- The playbook reports that GLM-4.5 "mentions that WSD performs worse than cosine decay on general benchmarks (SimpleQA, MMLU), but they don't provide any results". **Open question.**

**Conditions and limits.** During the stable stage a WSD checkpoint scores below a cosine checkpoint at the same token count, so the playbook applies a decay to the WSD checkpoint before comparing ([[smol-training-playbook]]). Olmo 3 anneals 7B checkpoints for mid-run evaluation and averages four 32B checkpoints 1,000 steps apart ([[olmo-3-pretraining-recipe]] §3.4). The WSD-versus-cosine comparisons with printed downstream numbers used a 1B model (100B and 460B tokens); the largest run in the cooldown paper is 8B on 12B tokens ([[cooldown-scaling-beyond-fixed-durations]] §3.3, §6). None of these sources describes a held-out task set for the schedule choice.

The figure [recipe-ledger-explorer.html](figures/recipe-ledger-explorer.html) draws the schedules in this ledger on a common token axis or on a fraction-of-run axis, including the conflicting OLMo 2 13B YAML and the pre-fix SmolLM3 plan, and plots tokens per parameter with a switch for DeepSeek-V3's parameter convention.

**Implication for a general-purpose model.** WSD keeps the budget open while breadth evaluations run, and its decay stage is where later stage mixtures (§5) are applied. Its cost is that stable-stage evaluations are not comparable with finished runs unless a short decay or checkpoint averaging is applied first.

## §5 Stage-wise mixtures side by side

**Definition.** A **pretraining stage** is a contiguous token range with fixed mixture weights. A mixture share is either a **token share** (fraction of tokens seen, as in Olmo 3 Table 4) or a **sampling weight** (probability of drawing from a dataset, as in the SmolLM3 configs). The two differ when datasets are repeated, when weights do not sum to 1, or when documents are packed differently.

**Problem.** Stage mixtures are printed at different granularity (per dataset, per category, or not at all), so the statement that recipes add math late in training has no comparable number until shares are grouped the same way.

**Side-by-side shares (web / code / math / other).**

| Checkpoint | Stage (tokens) | Web | Code | Math | Other | Status |
|---|---|---|---|---|---|---|
| SmolLM2 1.7B | 1 (0–6T) | 90% | 10% | 0% | — | verified; web read from Fig. 2 ([[smollm2]]) |
| | 2 (6–8T) | 75% | 20% | 5% | — | verified (§4.3) |
| | 3 (8–10T) | 74% | 16% | ≈10% | — | verified; web and code read from Fig. 2 |
| | 4, decay (10–11T) | 58% | 24% | 14% | 4% textbooks | verified (§4.5) |
| SmolLM3 3B | 1 (0–8T) | 85% (12% multilingual) | 12% | 3% | — | verified, blog ([[smollm3-blog-pretraining]]) |
| | 2 (8–10T) | 75% (12% multilingual) | 15% | 10% | — | verified, blog |
| | 3, decay (10–11.1T) | 63% (12% multilingual) | 24% | 13% | — | verified, blog; config-derived 63.1 / 23.3 / 13.1 / 0.5 |
| OLMo 2 7B | 1 (3.90T) | DCLM 3.71T | StarCoder 83.0B | OpenWebMath + Algebraic Stack 24.0B | peS2o 58.6B, arXiv 20.8B, Wiki 3.7B | verified ([[olmo-2-pretraining-recipe]] Table 4) |
| | 2, anneal (50B) | filtered DCLM 47.2% | — | Dolmino Math 20.8% | FLAN 16.6%, Wiki 7.11%, peS2o 5.85%, StackExchange 2.45% | verified (Table 13) |
| Olmo 3 7B | 1 (5.93T) | Common Crawl 76.1% | Stack-Edu 6.89% | FineMath 3+ 2.56% | olmOCR PDFs 13.6%, arXiv 0.86%, Wiki 0.04% | verified ([[olmo-3-pretraining-recipe]] Table 4) |
| | 2, mid-train (100B) | 27.5% | 20.0% | 19.2% | QA 13.9%, thinking 8.34%, instruction 6.1%, PDFs 5.0% | derived group sums (Table 5) |
| Llama 3.1 (all) | final mix | 50% general knowledge | 17% | 25% math and reasoning | 8% multilingual | verified; per-stage shares not reported ([[llama-3-recipe]]) |
| DeepSeek-V3; Qwen2.5; Gemma 3 | — | — | — | — | — | not reported (DeepSeek-V3 raises math and code versus V2 without numbers; Gemma 3 increases multilingual data without numbers) |

**Worked example: normalizing sampling weights.** The SmolLM3 decay-stage weights sum to 1.00565, not 1 ([[smollm3-training-configs]]). The code datasets, including the code-reasoning set, sum to 0.2348, so the normalized code share is 0.2348 / 1.00565 = 23.3%. The math datasets sum to 0.1316, giving 13.1%. The web datasets, English and multilingual, sum to 0.63425, giving 63.1%. The blog prints 63%, 24%, and 13%. The grouping of datasets into categories is this chapter's, so the 0.7-point code difference may come from grouping rather than from the run.

**What each stage changed.** SmolLM2 Table 3 gives the average after each stage: Knowledge/Reasoning 55.50 → 56.76 → 57.47 → 60.24; Math 3.21 → 3.7 → 7.27 → 22.07; Code 8.87 → 10.56 → 16.75 → 23.21 ([[smollm2]]). Stage 2 added 5% OpenWebMath and had "no significant impact on math performance" (§4.3). Stage 4 added the highest-quality math sets and decayed the LR at the same time. OLMo 2's 50B mid-training anneal raised the 7B average from 53.0 to 62.9 ([[olmo-2-pretraining-recipe]] Table 9).

**Language shares.** SmolLM3 keeps a 12% multilingual share in all three stages ([[smollm3-blog-pretraining]]). Llama 3.1's final mix is 8% multilingual tokens, and the report increased the non-English percentage during pre-training without printing values ([[llama-3-recipe]] §3.1.2, §3.4.1). Gemma 3 states that it increased multilingual data "to improve language coverage" without a share ([[gemma-3]] §2.2). ch-13a treats language coverage.

**Evidence for how stages were designed.**
- SmolLM2's principles: upsample high-quality math and code "during the annealing phase"; add medium-sized datasets mid-training "to avoid dilution by larger datasets early on"; stay near "the recommended 4–5 epoch threshold for most datasets" ([[smollm2]] §4, citing [[data-constrained-scaling]]). Stage-1 code was capped at 10% "to ensure approximately 4 epochs over 11T tokens" (§4.2).
- SmolLM2 changed mixtures during one run because pre-training cost "around 1e23 FLOPs, or $250,000 USD worth of GPU compute" (§1). SmolLM3 chose shares from "ablations on 3B models trained on 50B to 100B tokens" ([[smollm3-blog-pretraining]]).
- Olmo 3 considers a source for pre-training only if it can "yield enough tokens to impact model capabilities at pretraining scale", and reserves task data for later stages because it "tends to have an outsized impact on evaluation results, potentially confounding data ablations for other sources" ([[olmo-3-pretraining-recipe]] §3.4).
- OLMo 2 repeats math and Stack Exchange data in its 100B and 300B anneals because "keeping mixing proportion roughly constant across sources is beneficial" (§4.5).

**Conditions and limits.** The SmolLM2 stage-4 gains cannot be split between the data change and the LR decay from the published tables, so attributing them to data is an **Interpretation**. The stage shares above were chosen with benchmark feedback on the same suites that report the gains (SmolLM2 "Performance-driven interventions", §4). ch-32 and ch-32e cover decay-stage data in detail.

**Implication for a general-purpose model.** SmolLM2 and SmolLM3 keep web text at 58% or more in every stage, including the decay stage, while OLMo 2 and Olmo 3 lower web text to 47.2% and 27.5% in their separately named mid-training stages. All four raise the math share in the last stage, and OLMo 2's anneal contains no code source (Table 13). A small run that copies only the final-stage mixture copies a distribution designed for 1.3–11% of the tokens (derived from the stage lengths above).

## §6 Pretraining distillation as a recipe choice

**Definition.** **Logit distillation** in pre-training replaces the one-hot next-token target with a teacher's next-token distribution. **Pruning with distillation** removes width or depth from a trained model and retrains the smaller model against the original model's logits.

**Problem.** A one-hot target gives a small model one correct token per position. A teacher distribution gives a probability for many candidates per position, but computing and storing it for a vocabulary of about 262k entries costs teacher forward passes and memory.

**Mechanism (Gemma 3, [[gemma-3]] §2.2).**
1. For each position, run the teacher and "sample 256 logits per token, weighted by teacher probabilities".
2. Set the teacher target to "zero probability for nonsampled logits, and renormalized".
3. Train the student with cross-entropy against the renormalized target.

**Formula.**

```
L_KD = − Σ_t Σ_{v ∈ S_t} p̃_T(v | x_<t) · log p_S(v | x_<t),    p̃_T(v) = p_T(v) / Σ_{u ∈ S_t} p_T(u)
∂L_KD / ∂z_v = p_S(v) − p̃_T(v)   at each position, with p̃_T(v) = 0 for v ∉ S_t
```

where t indexes positions, S_t is the sampled token set at position t, p_T the teacher distribution, p̃_T its renormalized restriction to S_t, p_S the student distribution over the full vocabulary, and z_v the student logit of token v.

**Worked example (4-token vocabulary, 2 sampled tokens).** Teacher p_T = (0.60, 0.25, 0.10, 0.05); sampled set {1, 2}; renormalized target p̃_T = (0.706, 0.294, 0, 0). Student p_S = (0.50, 0.30, 0.15, 0.05).
1. Loss = −(0.706 × ln 0.50 + 0.294 × ln 0.30) = 0.489 + 0.354 = 0.843 nats.
2. Sparse-target logit gradient = (−0.206, +0.006, +0.15, +0.05). A gradient-descent step raises z_1 and lowers z_3 and z_4.
3. Full-target cross-entropy = 1.056 nats, with gradient p_S − p_T = (−0.10, +0.05, +0.05, 0.00). Under the full target, token 4 already matches the teacher and receives no push; under the sparse target it is pushed down because it was not sampled.
4. At Gemma 3's scale, 256 sampled entries out of about 262k is under 0.1% of the vocabulary per position (derived).

The push-down on unsampled tokens is a property of the target, not a negative sample: no sequence is labeled wrong, and no negative-feedback section applies to this chapter's stage (see ch-43a for negative gradients).

**Evidence.**
- Gemma 2, 2B student trained on 500B tokens (10× compute-optimal for 2B) with a 7B teacher: average 60.3 from scratch versus 67.7 distilled on 3 benchmarks; validation perplexity from scratch versus distilled 23 vs 21, 19 vs 17, and 17 vs 15 at 200M, 400M, and 1B ([[gemma-2]] Tables 6–7). The released Gemma 2 2B and 9B were distilled on 2T and 8T tokens, and the 27B was trained from scratch on 13T tokens (§1, §3.1). **Result (single study).**
- Gemma 3 trained one student with a small and a large teacher over several horizons: "for short training horizons, the smaller teacher is better, but the trend is reversed for longer training" ([[gemma-3]] §5.4, Fig. 8). Student and teacher sizes are not printed.
- Minitron: each teacher is first fine-tuned on the distillation dataset for about 100B tokens ("teacher correction"), then pruned, and the student is trained with "forward KL Divergence loss on the teacher and student logits only" ([[minitron-approach]]). Llama-3.1-Minitron-4B-Width used 94B tokens (peak LR 1e-4, minimum 1e-5, 40 warmup steps, cosine, batch 1152, context 8192; Table 4) and scored 60.5 MMLU and 41.2 GSM8k against 65.3 and 48.6 for its Llama 3.1 8B teacher (Table 1).

The forward KL used by Minitron is

```
L_FKL = Σ_t Σ_v p_T(v | x_<t) · log( p_T(v | x_<t) / p_S(v | x_<t) )
```

with the same symbols over the full vocabulary. It differs from full-vocabulary cross-entropy by the teacher entropy, which does not depend on the student, so its logit gradient is p_S − p_T.

**Conditions and limits.** Gemma 3 does not name its pre-training teacher, report teacher compute, or ablate the number of sampled logits. Minitron compares against models trained on different data (Llama 3.1 8B on 15T tokens), not against a from-scratch 4B model on the same 94B tokens. SmolLM2's statement that Llama3.2-1B was "derived from a pruned 8B model" and distilled on 9 trillion tokens is secondary ([[smollm2]] §4).

**Implication for a general-purpose model.** Distillation transfers the teacher's distribution over the documents the student sees; it does not add coverage that the pre-training data lacks. ch-35 and ch-35a treat distillation in post-training, where the teacher also generates the data.

## Recipe

Rows quote the settings compared in §1–§6. Rows marked 2026-09-15 were read at the locus in the primary artifact for this chapter (see the linked excerpts); rows marked 2026-09-14 come from verified library cards. "YAML" and "script" loci are at the pinned commits named in the excerpts.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3.1 405B | 405B | pretrain-stable | tokens; compute | 15.6T text tokens; 3.8 × 10^25 FLOPs | arXiv:2407.21783v3 §1, §3.2 ([[llama-3-recipe]]) | verified 2026-09-14 | scaling law extrapolates to 402B on 16.55T tokens at that compute (§3.2.1) |
| Llama 3.1 405B | 405B | pretrain-stable | peak LR; warmup; decay | 8 × 10^−5; 8,000 steps; cosine to 8 × 10^−7 over 1,200,000 steps | §3.4.1 | verified 2026-09-14 | no ablation reported |
| Llama 3.1 405B | 405B | pretrain-stable | global batch; sequence length | 4M tokens at 4,096 → "8M sequences of 8,192 tokens" after 252M tokens → 16M after 2.87T tokens | §3.4.1 | verified 2026-09-14 | "to improve training stability"; "few loss spikes", no numbers |
| Llama 3.1 405B | 405B | pretrain-stable | stop point on the step horizon | 1,154,407 of 1,200,000 steps (96.2%) | derived from §3.4.1, assuming all 15.6T tokens ran under the batch schedule | derived | n/a |
| Llama 3.1 405B | 405B | pretrain-decay/anneal | anneal | final 40M tokens, LR linearly to 0 at 128K context, high-quality sources upsampled, Polyak averaging | §3.4.3 | verified 2026-09-14 | no ablation reported |
| Llama 3.1 8B / 70B / 405B | 8B / 70B / 405B | pretrain-stable | peak LR | 3 × 10^−4 / 1.5 × 10^−4 / 8 × 10^−5 | v3 Table 3 | verified 2026-09-14 | no ablation reported |
| Llama 3.1 8B | 8B | pretrain-stable | tokens | "15T+" (one cell for 8B, 70B, 405B; pretraining data only) | llama-models MODEL_CARD.md, main, read 2026-09-15 ([[llama-3-1-model-card]]); report §1 prints no per-size count | verified 2026-09-15 (lower bound) | n/a |
| Llama 3.1 8B / 70B / 405B | 8B / 70B / 405B | all | compute | 1.46M / 7.0M / 30.84M H100 GPU hours ("Training Time"; scope not defined) | model card, energy-use table | verified 2026-09-15 | n/a |
| Llama 3.1 (all) | all | pretrain-stable | mixture (token share, final mix) | ~50% general knowledge, 25% math and reasoning, 17% code, 8% multilingual | §3.1.2 | verified 2026-09-14 | small-model scaling-law experiments per candidate mix; no table |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable, decay | tokens; LR schedule | 14.8T; 2K warmup steps to 2.2e-4; constant to 10T; cosine to 2.2e-5 over 4.3T; 2.2e-5 for 333B; 7.3e-6 for 167B | arXiv:2412.19437v2 §4.2 ([[deepseek-v3-recipe]]) | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | batch; sequence length | 3072 → 15360 over the first 469B tokens (unit not printed); 4K | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | pretrain-stable | mixture % | not given (math and code ratio raised versus V2) | §4.1; body and appendix checked | not reported | — |
| Qwen2.5 0.5B, 1.5B, 3B | 0.5B–3B | pretrain-stable | tokens per size; LR; batch; schedule | not given (18T pre-training data) | arXiv:2412.15115v2 §2–§3, no appendix; HF README ([[qwen-2.5-recipe]]) | not reported | B_opt and µ_opt scaling laws fitted, values not given (§3.2) |
| OLMo-2-1124-7B | 7B | pretrain-stable | peak LR; warmup; schedule | 3.0e-4; 2,000 steps (8,388,608,000 tokens); cosine to 0.1 × peak over 5T tokens, cut at step 928,646 (3.895T) | arXiv:2501.00656v3 Table 3; OLMo2-7B-stage1.yaml L46, L57-61 and stage2-seed42.yaml L75 @090253d | verified 2026-09-15 | Table 8 (7B setting): peaks 3e-4 to 12e-4 within 2 points on 9 OLMES tasks after a decay |
| OLMo-2-1124-7B | 7B | pretrain-stable | global batch; tokens | 1,024 × 4,096 tokens; 3.90T stage 1; 4.05T total counting three 50B runs | Table 3, §2.3; YAML L28, L81 | verified 2026-09-15 | no ablation reported |
| OLMo-2-1124-7B | 7B | pretrain-decay/anneal | LR; tokens; merge | linear from 6.1499e-5 to 0 over 50B tokens; 3 data orders averaged | stage2-seed42.yaml L46, L57-59, L80; §2.3 | verified 2026-09-15 (start LR equals the cosine at step 928,000, derived) | §4.1: truncation "with little loss of performance" from OLMo-0424 experience, no numbers |
| OLMo-2-1124-7B | 7B | pretrain-stable; decay/anneal | mixture | stage 1 DCLM 3.71T of 3.90T; anneal 50B: DCLM 47.2%, Dolmino Math 20.8%, FLAN 16.6%, Wiki 7.11%, peS2o 5.85%, StackExchange 2.45% | Tables 4, 13 | verified 2026-09-15 | Table 9: 7B average 53.0 → 62.9 after mid-training |
| OLMo-2-1124-13B | 13B | pretrain-stable | peak LR; warmup | 9.0 × 10^−4; 2,000 steps | Table 3; §4.1 prose | conflict | derived: stage-2 start 9e-5 at 5.0001T tokens equals 0.1 × 9e-4 |
| OLMo-2-1124-13B | 13B | pretrain-stable | peak LR; warmup | 3.0e-4; 8,388,608,000 tokens (1,000 steps at 2,048 × 4,096) | OLMo2-13B-stage1.yaml L46, L59 @090253d | conflict | inconsistent with OLMo2-13B-stage2-seed1110-100B.yaml L44 (9e-5) |
| OLMo-2-0325-32B | 32B | pretrain-stable | stage-1 tokens | 6.06T (§2.3) versus 7T (Table 9 caption); cosine over 6.5T truncated after 6T (Table 3) | §2.3; Tables 3, 9 | conflict | n/a |
| Olmo-3-1025-7B | 7B | pretrain-stable | peak LR; warmup; batch; tokens | 3.0 × 10^−4; 2,000 steps; 512 × 8,192 = 4,194,304 tokens; 5.93T | arXiv:2512.13961v2 Table 35, Fig. 3; OLMo-3-1025-7B-pretrain-1.py L40-42, L80 @66f768b | verified 2026-09-15 | no ablation reported |
| Olmo-3-1025-7B | 7B | pretrain-stable | final LR | 3.0 × 10^−5 (Table 35); "10% of the peak LR" (Fig. 3) | Table 35; Fig. 3 caption | conflict | none reported |
| Olmo-3-1025-7B | 7B | pretrain-stable | final LR | 3.93e-5 at the one-epoch stop | derived from pretrain-1.py L99-102, pretrain-2.py L85-87, L104-111; scheduler.py L536-558; trainer.py L521-527 | derived / conflict | formula and steps in §4 Worked example A |
| Olmo-3-1025-7B | 7B | mid-train | peak LR; batch; tokens; shape | 2.074 × 10^−4 (Table 35) versus 2.0712e-4 (OLMo-3-1025-7B-midtrain.py L43); 2,097,152 tokens; 100B; linear to 0 | Table 35; midtrain.py L41-43, L80 | conflict | Table 35 temperature column consistent with 2.074e-4 |
| Olmo-3-1025-7B | 7B | pretrain-stable; mid-train | mixture (token share) | stage 1: CC 76.1%, olmOCR PDFs 13.6%, Stack-Edu 6.89%, FineMath 3+ 2.56%, arXiv 0.86%, Wiki 0.04% of 5.93T; mid-train 100B: web 27.5%, code 20.0%, math 19.2%, QA 13.9%, thinking 8.34%, instruction 6.1%, PDFs 5.0% | Tables 4–5 | verified 2026-09-15 (group sums derived) | §3.4 source-size and task-data principles; no per-source numbers quoted |
| Olmo-3-1125-32B | 32B | pretrain-stable | peak LR; batch; schedule; final LR | 6.0 × 10^−4; 8,388,608 tokens; cosine over 5.93T truncated at 5.5T; final 6.0 × 10^−5 (Table 35) / 6.210 × 10^−5 (Fig. 4) / 6.70e-5 (derived from Fig. 4's description) | Table 35, Fig. 4 | conflict (final LR) | higher LR "somewhat compensated for by the larger batch size" (Fig. 4), no ablation |
| SmolLM2-1.7B | 1.7B | pretrain-stable; decay | tokens; schedule; batch; hardware; vocabulary | 11T; WSD, 2,000 warmup steps, 5.0e-4, linear to 0 over the final 10% (10T–11T); 2M tokens per batch at sequence 2,048; 256 H100 GPUs; 49,152 tokens | arXiv:2502.02737v1 §4, §4.1, §4.5, Table 6 | verified 2026-09-15 | WSD "to avoid setting a fixed training duration"; no schedule ablation |
| SmolLM2-1.7B | 1.7B | pretrain-stable; decay | stage mixtures (web / code / math) | 90/10/0; 75/20/5; 74/16/≈10; 58/24/14 + 4% Cosmopedia v2 | §4.2–§4.5, Fig. 2 | verified 2026-09-15 | Table 3 per-stage averages; stage-4 data and decay not separated |
| SmolLM2-1.7B | 1.7B | long-context | released base checkpoint | stage-4 checkpoint "before the final 75 billion tokens" plus 2k → 8k extension (tokens not reported) | §4.6 | verified 2026-09-15 | n/a |
| SmolLM2-360M; SmolLM2-135M | 360M; 135M | pretrain-stable | tokens; schedule | 4T; 2T; single stage; WSD with 20% decay; LR 3.0e-3; batch not reported | §6 | verified 2026-09-15 | data ablations re-run at the target length |
| SmolLM3-3B | 3B | pretrain-stable | optimizer; LR; schedule | AdamW (0.9, 0.95), weight decay 0.1 with token embeddings excluded, clip 1.0; 2e-4; 2,000 warmup steps; linear decay to 0 | blog "Training Configuration"; stage1_8T.yaml L204-221 @a48fa61 | verified 2026-09-15 | playbook: LR sweeps chose 2e-4; cosine versus WSD 10%/20% "similar final performance", no numbers |
| SmolLM3-3B | 3B | pretrain-stable | global batch; GPUs; duration; vocabulary | 192 × 1 × 3 × 4,096 = 2,359,296 tokens; dp 192 × tp 2 = 384 GPUs ("384 H100 GPUs for 24 days", blog); `vocab_size: 128256` | stage1_8T.yaml L146, L225-231, L250-254; blog "Training Configuration" | verified 2026-09-15 (products derived) | 2M–4M "minimal impact"; 2.36M best throughput ([[smol-training-playbook]]) |
| SmolLM3-3B | 3B | pretrain-decay/anneal | decay start; decay share | 10T; "the final 10% training steps" | blog "Training Configuration", "Data mixture and training stages" | conflict | n/a |
| SmolLM3-3B | 3B | pretrain-decay/anneal | decay start; decay share | step 4,198,001 = 9.90T; 522,000 of 4,720,000 steps = 11.06% | stage3_9T_11T.yaml L483-484, L540-542, L589 | derived / conflict | values set by commit 4bc9468 (2025-08-03); pre-fix stage-1 plan: decay at step 2,600,000 over 600,000 of 3,200,000 steps |
| SmolLM3-3B | 3B | pretrain-stable | total tokens | 11T (headline); 11.2T (stages paragraph); stage 3 ends at 11.1T; 11.14T (config-derived) | blog; stage configs | conflict | budget set by "384 H100s for roughly a month" at ~30% model FLOPs utilization (playbook) |
| SmolLM3-3B | 3B | pretrain-stable; decay | stage mixtures (web incl. multilingual / code / math) | 85 (12)/12/3 for 0–8T; 75 (12)/15/10 for 8–10T; 63 (12)/24/13 for 10–11.1T | blog "Data mixture and training stages"; config sampling weights | verified 2026-09-15 | ablations on 3B models at 50B–100B tokens, no numbers |
| Gemma 3 1B / 4B / 12B / 27B | 1B–27B | pretrain-stable | tokens; target | 2T / 4T / 12T / 14T; teacher distribution restricted to 256 sampled logits per token, renormalized | arXiv:2503.19786v1 §2.2 | verified 2026-09-15 | §5.4 Fig. 8: small teacher better at short horizons, large teacher at long |
| Gemma 3 (all) | all | pretrain-stable | LR; batch; schedule; mixture % | not given | §2 checked | not reported | — |
| Gemma 2 2B; 9B; 27B | 2B; 9B; 27B | pretrain-stable | tokens; target | 2T and 8T with distillation; 27B 13T from scratch | arXiv:2408.00118v3 §1, §3.1 ([[gemma-2]]) | verified 2026-09-14 | Table 6: 2B on 500B tokens, 60.3 from scratch versus 67.7 distilled from 7B |
| MiniCPM-2.4B | 2.4B | pretrain-stable; decay | schedule; batch; tokens | WSD, max LR 0.01 under Tensor Program width and depth scaling (§3); batch 3.93 million (§6.2) / 4M (Table 2); stable around 1T; exponential decay with T = 5,000 steps (20B tokens); 1.1T total | arXiv:2404.06395v3 §6.2, Table 2 | verified 2026-09-15 | §4.3 (0.036B): 10% decay sufficient, 2.5% short |
| Llama-3.1-Minitron-4B-Width | 4.5B | pretrain (distillation retraining) | LR; schedule; batch; context; tokens | peak 1e-4, min 1e-5, 40 warmup steps, cosine; 1152; 8192; 94B | arXiv:2408.11796v4 Table 4 | verified 2026-09-15 | Table 1: MMLU 60.5 (width) versus 58.7 (depth); teacher 65.3 |
| nanochat d24 speedrun | d24 | pretrain-stable | tokens : scaling-params ratio | 8 (speedrun.sh L67); parser default 12 (base_train.py L58); comment calls 10.5 the default (speedrun.sh L66) | @f527f76 ([[nanochat]]) | conflict (comment versus code) | "derived experimentally via scaling laws analysis" (L261); experiments not read |

**Starting point for a small general-purpose run.** For a dense model of 1.7B–3B parameters trained on 11T–11.2T tokens, the verified SmolLM2-1.7B and SmolLM3-3B rows give: AdamW with (β1, β2) = (0.9, 0.95), weight decay 0.1 with token embeddings excluded, and gradient clipping 1.0 (SmolLM3-3B); a WSD schedule with 2,000 warmup steps, a peak LR of 5.0e-4 (1.7B) or 2e-4 (3B), and a linear decay to 0 over the final 10% of steps (SmolLM2-1.7B; the SmolLM3 decay share is in conflict); a constant global batch of 2M tokens at sequence length 2,048 (1.7B) or 2,359,296 tokens at 4,096 (3B); and a first stage of 85–90% web text with 10–12% code (0–3% math), followed by later stages that raise code to 24% and math to 13–14%. These values ran on 256 H100 GPUs for 11T tokens with a 49,152-token vocabulary (SmolLM2-1.7B) and on 384 H100 GPUs for 24 days with a 128,256-token vocabulary (SmolLM3-3B). For a model below 500M parameters, the verified schedule rows are SmolLM2-360M and SmolLM2-135M (WSD with 20% decay, LR 3.0e-3, 4T and 2T tokens); their batch size is not reported. When one fixed budget at 7B is planned instead, the verified OLMo-2-1124-7B rows give cosine with 2,000 warmup steps, peak 3.0e-4, a floor of 0.1 × peak, and a batch of 1,024 × 4,096 tokens, trained on 3.90T tokens. Among the rows in this table, only the OLMo 2 peak-LR evidence (Table 8) compares values of a budget, batch, or schedule setting with downstream numbers in the setting of a released model (7B), and those numbers come from OLMES multiple-choice validation tasks, which belong to OLMo 2's development suite rather than its held-out suite (Interpretation from Table 9's column split), so a held-out breadth gate (ch-00, ch-17) is still required.

## Generalization lens

**(a) What increases breadth.**
- Adding code and math in later stages raised those averages without lowering the knowledge average in SmolLM2: Knowledge/Reasoning 55.50 → 60.24, Math 3.21 → 22.07, Code 8.87 → 23.21 across four stages ([[smollm2]] Table 3). **Result (single study)**; the same benchmark categories guided the stage changes (§4).
- OLMo 2's mid-training raised benchmarks the team declared held out: at 7B, AGIEval 44.6 → 50.4, MMLU Pro 27.4 → 31.0, TriviaQA 74.6 → 78.0 ([[olmo-2-pretraining-recipe]] Table 9). **Result (single study).**
- Distilling a 2B student from a 7B teacher on 500B tokens raised a 3-benchmark average from 60.3 to 67.7, and the perplexity gain held at 200M, 400M, and 1B ([[gemma-2]] Tables 6–7). **Result (single study).**
- Longer training of small models: in 47 runs from 150M to 6B parameters (up to 10,000 tokens per parameter for the 150M model), the 5-category downstream average kept rising with no saturation point observed ([[beyond-chinchilla-inference-scaling]] §4). **Result (single study)**; fine-tuning after over-training was not measured.

**(b) What causes narrowing or inflated scores.**
- Benchmark training splits in late stages. OLMo 2's Dolmino Math mix contains the GSM8K train split ([[olmo-2-pretraining-recipe]] Table 5), and 7B GSM8K rose 24.1 → 67.5 in mid-training against +3.6 on MMLU Pro; 200 of the 1,319 GSM8K examples were used as a development set, so GSM8K is only partially held out (footnote 6). SmolLM2's stage 4 includes 0.02% AugGSM8K ([[smollm2]] §4.5), and SmolLM3's decay stage includes a GSM8K-derived synthetic set at weight 0.0004 ([[smollm3-training-configs]] L359, L473). A gain on the benchmark whose training split is in the mix is weaker evidence of breadth than a gain on a task outside it (Interpretation).
- Llama 3 annealing on GSM8K and MATH training sets raised 8B validation scores by 24.0% and 6.4% with a "negligible" effect at 405B; the team excluded these sets from its annealing data "to assess the true few-shot learning capabilities and out-of-domain generalization" ([[llama-3]] §3.1.3).
- Olmo 3 compared its decontaminated 100B mid-training anneal with a matched anneal on non-decontaminated data: DROP, Minerva, and SQuAD scores were higher with contamination, while GSM8K, despite "complete leakage", was higher with the decontaminated data ([[olmo-3-pretraining-recipe]] §3.5.4, Fig. 12). Contamination therefore inflates some scores and not others. **Result (single study).**
- Task data in the first stage can dominate evaluation and confound data ablations for other sources, which is why Olmo 3 reserves it for later stages (§3.4). **Interpretation** by the authors.
- A mis-copied setting (§1) changes the whole run; no source here measures its effect on breadth. **Open question.**

**(c) How to measure it at this stage.**
- Declare development and held-out suites before training and do not look at the held-out suite until release. OLMo 2 held out AGIEval, GSM8K (partly), MMLU Pro, and TriviaQA ([[olmo-2-pretraining-recipe]] §2.5, Table 9); Olmo 3 reports an OlmoBaseEval HeldOut block (LBPP, BBH, MMLU Pro MC, Deepmind Math) and states that it "was not evaluated on held-out benchmarks prior to release" ([[olmo-3-pretraining-recipe]] Tables 2–3).
- Evaluate WSD stable-stage checkpoints only after a short decay, or average several checkpoints, before comparing them with other runs ([[smol-training-playbook]]; [[olmo-3-pretraining-recipe]] §3.4).
- For each stage change, record which benchmarks guided it and keep at least one capability cluster outside that set (ch-00, ch-47a).
- Measure contamination inflation with a matched run on non-decontaminated data, as Olmo 3 did, instead of assuming its direction (ch-14, ch-48).
- For distillation, compare against a from-scratch run on the same tokens, as Gemma 2 Table 6 does, and evaluate tasks outside the teacher's training targets.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Copying a peak LR from a released stage-1 config without a boundary check | The next stage's start LR does not equal the schedule value at its load step | Compute η at the load step with the scheduler code (§1 examples 1–2) |
| Copying a warmup in tokens across model sizes | Warmup covers 1,000 steps at the larger batch instead of the printed 2,000 | Divide warmup tokens by the size-specific batch (OLMo 2 13B YAML) |
| Treating "batch size" in instances as tokens | Warmup or stage boundaries fall at the wrong token count | Multiply by sequence length; compare with a boundary printed in tokens |
| Taking a schedule horizon as tokens seen | OLMo 2 7B listed at 5T tokens | Find the stop step or the stage-2 load step and convert to tokens |
| Comparing tokens per parameter across total and activated parameters | A MoE model looks under- or over-trained by a factor of 18 | Write the N convention in the ledger cell (§2) |
| Counting averaged anneal runs as tokens seen | OLMo 2 7B budget 4.05T instead of 3.95T per path | Sum tokens along one path to the released checkpoint |
| Reading a planned schedule as the run | Final LR taken as 10% of peak when the run stopped before or after the horizon | Find the stop point and compute η there (Olmo 3 7B and 32B, Llama 3.1 405B) |
| Trusting a config's step counts without its history | Stage files carry step numbers written after release | Read the commit log and diffs of the config directory (SmolLM3 commit 4bc9468) |
| Normalizing sampling weights that do not sum to 1 as if they did | Category shares add up to more than 100% | Divide by the weight sum (SmolLM3 decay stage: 1.00565) |
| Copying a final-stage mixture into stage 1 | Code and math above 20% from step 0 and web below 60% | Compare with the stage-1 rows in §5 |
| Crediting a late-stage gain to data | The gain starts at the LR decay onset | Decay the old and the new mixture from the same checkpoint |
| Evaluating a WSD stable checkpoint against a cosine checkpoint | WSD appears worse mid-run | Apply a short decay before comparing ([[smol-training-playbook]]) |
| Reporting a benchmark as breadth evidence while its training split is in a late-stage mix | Largest gain on that benchmark, small gains on declared held-out tasks | Decontaminate against the evaluation suite and report held-out tasks separately (ch-14) |

## Check your understanding

1. The OLMo 2 13B stage-2 YAML starts at 9e-5 from a checkpoint at 5.0001T tokens. Explain step by step why this value is evidence about the stage-1 peak LR, and what the check could not establish if stage 1 had stopped at 4T tokens with an uncertain α_f.
2. Olmo 3 7B's Figure 3 states a final LR of 10% of peak, while the pinned scripts give 3.93e-5 at the one-epoch stop. Explain which property of `HalfCosWithWarmup` and of `Trainer.max_steps` produces the difference, and which single change to part 2 would make the two agree.
3. DeepSeek-V3's tokens per parameter is 22 or 400 depending on the convention. Explain which convention better tracks training compute per token and which better tracks the capacity available to store knowledge, and why neither alone predicts downstream breadth.
4. A WSD run with a 10% decay serves budgets of 1T, 2T, and 4T tokens for 4.3T total tokens. State the conditions on peak LR, batch, and stable-stage data under which the three decayed checkpoints match three separate WSD runs, and design a small-scale experiment that tests the claim.
5. SmolLM2's math average rose from 7.27 to 22.07 in stage 4. List the changes that happened at that stage boundary and design the smallest pair of runs that would attribute the gain between the LR decay and the data change.
6. OLMo 2 7B's GSM8K score rose by 43.4 points in mid-training and MMLU Pro by 3.6. Using the composition of Dolmino Math and footnote 6 of the report, explain why these two numbers carry different evidence about breadth, and why Olmo 3's contamination comparison warns against assuming the direction of the effect.
7. In the sparse-target distillation example, token 4 is pushed down although the student already matches the teacher on it. Explain why, how the effect depends on the number of sampled entries, and why sampling in proportion to teacher probability keeps the expected target close to the teacher distribution for high-probability tokens.
8. Llama 3.1 405B doubled its batch at 252M tokens and again at 2.87T tokens, and MiniCPM states that enlarging the batch can act like lowering the LR. Explain the mechanism through the gradient-noise argument, and what it implies for recording a batch ramp and an LR schedule as independent ledger rows.

## Connections

- **Previous chapter:** ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination (the 4–5 epoch limits behind SmolLM2's stage caps in §5; decontamination behind the benchmark-split items in the Generalization lens).
- **Next chapter:** ch-17 — Lab: Filter and Mixture Ablation with Breadth Measurement. The chapter after it in the outline is ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL.
- **Depends on:** ch-14 — Data-Constrained Scaling, Repetition, and Pretraining Decontamination.
- **Related earlier chapters:** ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization (schedule and batch mechanics); ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability (budget choice and plasticity cost); ch-12a — Memorization, Knowledge Acquisition, and Generalization During Pretraining; ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale (how mixture weights are chosen); ch-13a — Multilingual Coverage and Vocabulary as Capability Axes (language shares).
- **Used later:** ch-32e — Mid-Training, Annealing, and Context-Extension Recipes Side by Side (the same ledger method for later stages); ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data (post-training distillation); ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages; ch-48 — Contamination Detection and Its Effect on Reported Scores; ch-58a — Open General-Model Recipes End to End: Pretraining to Merge (end-to-end ledgers).

## Sources

- [[llama-3]] — Llama 3 report (arXiv 2024-07): over-training rationale for smaller models; annealing experiment on GSM8K and MATH training sets.
- [[llama-3-recipe]] — Llama 3.1 405B tokens, compute, schedule, batch ramp, anneal, final data mix, per-size peak LR; unreported per-size tokens.
- [[llama-3-1-model-card]] — excerpt of the Llama 3.1 model card: "15T+" token cell shared by 8B, 70B, 405B; GPU hours per size.
- [[deepseek-v3]] — DeepSeek-V3 report (arXiv 2024-12): MLA and DeepSeekMoE efficiency rationale.
- [[deepseek-v3-recipe]] — DeepSeek-V3 schedule, batch schedule, tokens, unreported mixture.
- [[qwen-2.5]] — Qwen2.5 report (arXiv 2024-12): 18T pre-training data; cost-effectiveness rationale for small sizes.
- [[qwen-2.5-recipe]] — Qwen2.5 not-reported pre-training settings and fitted hyperparameter laws.
- [[olmo-2-pretraining-recipe]] — excerpt of the OLMo 2 report (arXiv 2025-01): Table 3 schedules, §2.3 budgets and souping, Table 8 LR comparison, Tables 4, 5, 13 mixtures, Table 9 held-out results.
- [[olmo-2-official-configs]] — excerpt of the OLMo 2 released YAMLs and scheduler code: warmup tokens, stage-2 start LR, 13B peak-LR and warmup conflicts.
- [[olmo-3-pretraining-recipe]] — excerpt of the Olmo 3 report (arXiv 2025-12): Table 35, Figures 3–4, Tables 4–5, data principles, decontamination comparison, held-out statements.
- [[olmo-core-olmo3-configs]] — excerpt of the OLMo-core Olmo 3 scripts and scheduler: stretched schedule, derived final LR, load-step gap, mid-training LR conflict.
- [[smollm2]] — excerpt of the SmolLM2 report (arXiv 2025-02): 11T budget and rationale, WSD settings, four-stage mixtures, Table 3, released checkpoint path, smaller sizes.
- [[smollm3-blog-pretraining]] — excerpt of the SmolLM3 blog (July 2025): training configuration, three stage mixtures, token totals.
- [[smollm3-training-configs]] — excerpt of the SmolLM3 nanotron configs and commit 4bc9468: batch product, stage and decay steps, pre-fix plan, sampling weights, GSM8K-derived set.
- [[smol-training-playbook]] — excerpt of the Smol Training Playbook (October 2025): budget rationale, WSD ablation statement, batch choice, survey-table entries for DeepSeek-V3 and OLMo 2.
- [[gemma-3]] — excerpt of the Gemma 3 report (arXiv 2025-03): per-size tokens, Table 1 parameter counts, 256-logit distillation, small versus large teacher.
- [[gemma-2]] — Gemma 2 report (arXiv 2024-08): distillation versus from-scratch results (Tables 6–7), released token counts.
- [[minicpm]] — excerpt of the MiniCPM report (arXiv 2024-04): WSD definition, 10% versus 2.5% decay, O(mC) scaling measurement, inference-compute trade, released schedule and checkpoint choice.
- [[minitron-approach]] — excerpt of the Minitron paper (arXiv 2024-08): teacher correction, forward-KL distillation, Table 4 settings, Table 1 tokens and scores.
- [[nanochat]] — excerpt of nanochat code at f527f76: ratio-based horizon, batch, LR, and weight-decay scaling rules, schedule, comment-versus-code conflict.
- [[cooldown-scaling-beyond-fixed-durations]] — Hägele et al. (arXiv 2024-05): 1B cosine versus cooldown downstream aggregates.
- [[beyond-chinchilla-inference-scaling]] — Sardana et al. (arXiv 2023-12): inference-aware allocation and runs to 10,000 tokens per parameter.
- [[large-batch-training-noise-scale]] — McCandlish et al. (arXiv 2018-12): growth of the critical batch size during training.
- [[data-constrained-scaling]] — Muennighoff et al. (arXiv 2023-05): the epoch threshold cited by SmolLM2 for stage design.
- [[olmo-2]], [[olmo-3]], [[smollm-3]] — library cards for these releases; they have no Verification section as of 2026-09-15 and are not used for any value in this chapter (the excerpts above are).
