<!-- chapter: ch-32f
     track: midtraining
     kind: lab
     title: Lab: Annealing and Context Extension with a Short-Context Regression Gate
     deps: [ch-32e]
     sources: [[ruler]], [[helmet]], [[prolong]], [[prolong-recipe]], [[yarn]], [[paloma]], [[smollm2]], [[olmo-2-annealing]],
              [[controlled-long-context-extension]], [[position-interpolation]], [[llama-2-long]], [[llama-3]], [[llama-3-recipe]],
              [[signal-and-noise-eval]], [[cooldown-scaling-beyond-fixed-durations]], [[adding-error-bars-evals]],
              [[hf-rope-utils-v4460]], [[ruler-effective-length]]
     figures: figures/rope-extension-explorer.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32f — Lab: Annealing and Context Extension with a Short-Context Regression Gate

> **Core insight.** Annealing and context extension each change scores the stage does not target, and both changes are easy to misattribute. Decaying the learning rate on the unchanged pretraining mix already moved an OLMo 2 7B checkpoint from 69.6 to 74.0 on OLMES and from 59.8 to 61.8 on MMLU, so a candidate dataset's effect is measurable only against that control ([[olmo-2-annealing]], Table 11). In 5B-token Llama-3-8B extension ablations, the short-task average fell monotonically as the long-data share rose, and PG19 perplexity kept improving while the downstream long-context average fell at 100% long data ([[prolong]] §2.1, §3.2, Figs. 1 and 3). This lab anneals a small checkpoint on two candidate datasets against a control, extends the chosen checkpoint with two RoPE methods at two long:short ratios, and admits a cell only if paired short-context differences pass a pre-registered margin and its long-context gains are shown by RULER effective length and a HELMET RAG score.
>
> **Guideline.** When judging an annealing dataset, compare it with an anneal on the pretraining mix at the same token count and schedule, because the schedule alone changes broad scores ([[olmo-2-annealing]] Table 11). When choosing a long:short ratio or an extension method, decide on downstream long-context tasks and on paired short-context differences, not on perplexity at the target length, because perplexity preferred 100% long data in [[prolong]] Fig. 1. For the 1B and 360M checkpoints of this lab, define RULER effective length relative to the checkpoint's own score at its native length in the same harness, because RULER's thresholds are Llama 2 7B scores at 4K (79.4 for base models, 85.6 for aligned models; [[ruler-effective-length]], Tables 3 and 12) and LLaMA2-7B scored 80.94 at 4k in Lu et al.'s configuration ([[controlled-long-context-extension]] Table 3). When a released report states that extension caused "next to no degradation" without a table ([[smollm2]] §4.7), treat the claim as untested for the lab's checkpoint and measure it.

## Why this chapter matters for a general-purpose model

The pipeline runs pre-training → mid-training → SFT → preference optimization → RL → evaluation. This lab closes the mid-training phase ([[ch-32]] to [[ch-32e]]) with a measured artifact, as [[ch-17]] closes pretraining and [[ch-36]] closes SFT. Its output checkpoint is the starting point that SFT receives, so anything the stage removes from short-context ability is inherited by every later stage.

The lab addresses three measurable problems:
1. **Attribution of annealing gains.** An anneal combines a learning-rate decay with a data change. Without a control arm, the decay's effect is credited to the data. In OLMo 2 7B, the decay alone added 4.4 OLMES points and 2.0 MMLU points ([[olmo-2-annealing]] Table 11).
2. **Short-context regression from extension.** Relative to Llama-3-8B (MMLU 66.5, GSM8K 44.7), RoPE-base position extrapolation gave 64.7 / 40.1 and training on the Fu et al. long mix gave 63.1 / 40.6 ([[prolong]] §2.3, Table 2). Position Interpolation to a 32,768-token window lowered LLaMA 7B zero-shot BoolQ, measured inside the original 2,048-token window, from 76.1 to 64.7 ([[position-interpolation]] Table 5).
3. **Long-context claims that measure the wrong thing.** Perplexity and single-needle retrieval can improve while task scores do not: at 128K, NIAH is 100.0 for five of six frontier models in [[helmet]] Table 5, and NIAH correlates with ∞Bench QA at 0.63 against 0.88 for the HotpotQA RAG task (Fig. 4).

## §1 Lab question, hypotheses, deliverables, and budget paths

**Question.** Can a small checkpoint be annealed and context-extended while passing a short-context regression gate?

**Hypotheses, written before any run.** Each row states the direction a source measured, the setting of that measurement, and the observation that would contradict it here.

| Axis | Predicted direction | Source and setting | Contradicting observation |
|---|---|---|---|
| Anneal on pretraining mix (control) vs starting checkpoint | control raises broad multiple-choice scores | [[olmo-2-annealing]] Table 11 (7B at 4T tokens, 50B tokens): OLMES 69.6 → 74.0, MMLU 59.8 → 61.8, GSM* 28.5 → 27.0 | control − start interval on the broad suite includes zero |
| Targeted math candidate vs control | math target gain with an interval above zero (35 points at 7B in the source); broad change within ±1 point | [[olmo-2-annealing]] Table 12 (7B, 576M tokens, 35/65 math/web): GSM* 28.5 → 63.5, MMLU 59.8 → 60.1 | math held-out partner does not rise while GSM8K rises |
| Broad educational-web candidate vs control | gains on knowledge and science multiple choice, losses on commonsense tasks | [[smollm2]] Table 1 (350B-token runs from scratch, not anneals): FineWeb-Edu vs DCLM MMLU 37.5 vs 35.5, HellaSwag 60.1 vs 62.3 | HellaSwag and CommonsenseQA intervals lie above zero |
| Long share 70% vs 40% | lower short-context scores at 70% | [[prolong]] Fig. 3 (Llama-3-8B, 5B tokens at 64K; monotonic decrease) | paired short-suite difference 70% − 40% lies above zero |
| Base-frequency increase (ABF) + training vs YaRN + training | ABF ≥ YaRN on RULER at the target length; YaRN lower beyond it | [[controlled-long-context-extension]] Table 3 (LLaMA2-7B, 1B tokens at 32k): Dynamic NTK 59.42 vs YaRN 36.95 at 32k; 46.26 vs 0.00 at 64k | YaRN − ABF RULER interval at L′ lies above zero |
| Document masking on vs off | small gain on both long and short averages | [[prolong]] App. B.2 Table 20 (8B, 5B tokens, scored after UltraChat SFT, seed count not reported): long 54.6 vs 53.6, short 65.5 vs 64.9 | Open question at 1B: no prediction for the sign of an interval |

The ABF row compares a fixed increased base with the Dynamic NTK variant of Lu et al., so the direction is an Interpretation of that result for this lab.

**Deliverables.**
1. `runs.jsonl`: one row per cell with start checkpoint, data file hashes, token counts by source, RoPE settings as loaded (base, `rope_scaling`, ramp bounds), masking flag, LR schedule, seed, commit.
2. `eval/items/<cell>/<suite>.jsonl` and `eval/docs/<cell>/<paloma-domain>.jsonl`: per-item scores and per-document negative log-likelihood for the start checkpoint and every cell.
3. `decontamination.md`: n-gram overlap of every candidate and extension source with every evaluation suite (§2.3).
4. `stage-memo.md`: margins committed before training; the Part A table; the gate table (§7); the chosen recipe; and a list of every short-context score the chosen cell lost, with intervals.

**Budget paths.**

| Item | Full path | Resource-constrained path |
|---|---|---|
| Start checkpoint | OLMo 2 1B after stage 1 (4T tokens, before the Dolmino anneal; [[olmo-2-annealing]] App. B), if that revision is published; otherwise another RoPE checkpoint taken before its final decay | SmolLM2-360M base (config: `max_position_embeddings` 8192, `rope_theta` 100000, head dimension 64; [[smollm2]]) |
| Native length L₀ → target L′ | 4,096 → 16,384 (4×); optional ABF cell at 32,768 (8×) | 8,192 → 16,384 (2×) |
| Part A arms | control, targeted, broad; 5B tokens each | control, targeted; 1B tokens each |
| Part B cells | ABF and YaRN × 40% and 70% long; masking off for ABF-40%; training-free YaRN evaluation | ABF at 40% long |
| Tokens per extension cell | 1B | 0.5B |
| Short-context gate, RULER, HELMET RAG | required | required |

OLMo 2 App. B lists departures of 1B from 7B (layers, width, heads, batch, peak LR) and does not list sequence length or RoPE θ, so the lab reads L₀ = 4,096 and θ = 500,000 from the 7B settings (derived); the loaded `config.json` must confirm both before training. SmolLM2 §6 does not state the 360M training length, so the constrained path first measures RULER at 2K, 4K, and 8K on the start checkpoint to find its own effective length.

**Token arithmetic (derived).** With OLMo 2 1B's batch of 512 sequences at 4,096 tokens, one step is 2,097,152 tokens, so a 5B-token anneal is 2,384 steps. An extension cell at 16,384 tokens with 64 sequences per step uses 1,048,576 tokens per step, so 1B tokens is 954 steps. Both counts assume every sequence is filled; the logged token count replaces them.

## §2 Part A: annealing a checkpoint on two candidate datasets

### 2.1 Definition, problem, and mechanism

**Definition.** An anneal continues training a checkpoint while the learning rate decays to zero, with a data mixture that may differ from pretraining. A **microanneal** is a short anneal used to test one data source ([[olmo-2-annealing]] §4.4.2).

**Problem.** The measured change after an anneal mixes three effects: the decay, the candidate data, and the reduction of the pretraining data share. Llama 3 used anneals to value datasets: a 50%-trained 8B model decayed linearly to zero over 40B tokens with 30% weight on the new dataset ([[llama-3-recipe]], §3.1.3). SmolLM2 tested math data from a 3T-token checkpoint with 60B candidate tokens and 40B pre-checkpoint tokens ([[smollm2]] §3.1). Neither report prints a control arm in those passages; OLMo 2 Table 11 does.

**Mechanism.**
1. Record the start checkpoint's learning rate η₀ at its stop step from the released schedule.
2. Build each arm's mixture: 50% candidate tokens and 50% pretraining-mix tokens, following OLMo 2's "roughly the same quantity" rule. The control arm uses 100% pretraining mix.
3. Train every arm for the same token count and the same schedule, decaying linearly from η₀ to 0.
4. Evaluate the start checkpoint and all arms with the same harness (§3).

**Formula (linear decay).**

```
η(k) = η₀ · (1 − k / K),   k = 0 … K
```

k is the optimizer step inside the anneal, K the number of anneal steps, and η₀ the learning rate at the checkpoint. The lab uses the linear shape because OLMo 2, Llama 3, and SmolLM2 used it in the cited anneals. In [[cooldown-scaling-beyond-fixed-durations]] (§3.2, Fig. 4) a (1-sqrt) cooldown reached lower loss than a linear one; the lab keeps one shape for all arms and records it.

### 2.2 Arms

| Arm | Candidate (50% of tokens) | Role |
|---|---|---|
| N0 control | none (100% pretraining mix) | isolates the decay |
| N1 targeted | FineMath 4+ (10B tokens, 6.7M documents; [[smollm2]] §3.3.2), decontaminated (§2.3), with no benchmark training split added | a narrow skill source |
| N2 broad | a high-score FineWeb-Edu subset | a broad educational source with a predicted mixed pattern (the prediction assumes a DCLM-like pretraining mix; record the actual mix) |

**Evidence for the arm effects.**
- The OLMo 2 7B microanneal with a 35/65 math/web mixture of 576M tokens raised GSM* from 28.5 to 63.5 with MMLU 59.8 → 60.1; a 10/90 mixture of 1.72B tokens gave 61.0 and 60.9. The authors conclude that the math data needs to be present, not dominant ([[olmo-2-annealing]] Table 12). Result (single study).
- FineMath 4+ anneals gave "a 2x improvement on GSM8K and a 6x improvement on MATH compared to InfiMM-WebMath" at 1.7B ([[smollm2]] §3.3.2, Fig. 1). Result (single study).
- Llama 3 annealing on GSM8k and MATH training sets raised 8B validation scores by 24.0% and 6.4% and had a "negligible" effect at 405B, and the released annealing data excluded benchmark training sets ([[llama-3]], §3.1.3). Result (single study). OLMo 2's math mix contains GSM8K-Train, and SmolLM2's decay stage contains 0.02% AugGSM8K ([[olmo-2-annealing]] §4.4.1; [[smollm2]] §4.5). The lab excludes benchmark training splits so that N1 measures the source rather than the benchmark format.

### 2.3 Decontamination

Apply SmolLM2's rule to both candidates against every suite in §3.2: 13-gram matching with a minimum overlap ratio of 0.6 to the longest common subsequence ([[smollm2]] §3.3.2). Record matched documents per suite and remove them. Paloma's own decontamination rule does not cover code sources (G1; [[paloma]] §3), so code domains in the perplexity panel are labeled as not decontaminated against the candidates unless the lab's rule is applied to them.

### 2.4 Worked example: what a 200-item development set can resolve

OLMo 2 decided mixtures with GSM*, 200 of the 1,319 GSM8K questions ([[olmo-2-annealing]] §4.4.2). At 28.5% accuracy the analytic 95% half-width is 1.96 · sqrt(0.285 × 0.715 / 200) = 6.3 points (derived). The PT Mix change of −1.5 points is 3 questions, and the Web FW72 change of +1.5 is also 3 questions, which the authors call "within margin of error" (§4.3). The 35-point microanneal gain is 5.6 half-widths. A 200-item set can therefore separate the math arm from the control but cannot resolve a 1-2 point broad effect; the lab uses full test sets and paired intervals for those (§5).

**Conditions and limits.** The Table 11 and Table 12 evidence is for 7B at 4T tokens; OLMo 2 App. B reports that scaling training tokens for 1B was "difficult", so the size of the control effect at 1B is not reported. Soups of three data orders equaled or beat the best single run in 23 of the 24 metric cells of Table 14 (six mixes × four metrics), but mix E's GSM* fell from 60.5 (best single) to 43.0 (soup), so run-to-run variation in anneals is not negligible (§5.3).

**Implication for a general-purpose model.** A targeted anneal can move its target benchmark by more than 30 points at 7B. The broad suite and the held-out partner determine whether that gain is general math ability or fit to one benchmark's format.

## §3 Part A measurement: targeted versus broad effects

### 3.1 Per-domain held-out perplexity

**Definition.** Paloma measures perplexity on 546 domains from 16 sources and reports a macro average, the unweighted mean of per-domain perplexities ([[paloma]] §2, §4.1).

**Problem.** One held-out loss averages over domains and hides domain-specific changes; Paloma's C4-only baseline reached perplexity 391,171 on RedPajama arXiv (§4.1). A targeted anneal is expected to lower loss on its own domains and may raise it elsewhere.

**Formula.**

```
ppl(d)   = exp( − Σ_{t∈N_d} Σ_i ln p(t_i | t_<i) / T(N_d) )
macro(D) = (1/|D|) Σ_{d∈D} ppl(d)
BPB      = − ℓ / (B · ln 2)
```

N_d is the set of documents in domain d, t one document, t_i its i-th token, T(N_d) the token count, D a set of domains, ℓ the summed log-likelihood, and B the number of UTF-8 bytes. BPB (bits per byte) is used when tokenizers differ ([[paloma]] App. B).

**Protocol.**
1. Before training, assign Paloma domains to a target group per candidate (for N1, math-related domains) and a broad group (all others).
2. Score each document separately after BOS and split documents longer than the context into disjoint inputs. Concatenated inputs broke variance trends: Pythia 1.4B at 2B tokens gave 92.23 ± 17.33 concatenated vs 42.57 ± 0.29 separately (Table 17).
3. Store per-document NLL so that the arm difference is paired by document.

**Worked example (illustrative values).** Three broad domains have perplexities 12.0, 20.0, 6.0 for N0 and 12.4, 22.0, 6.1 for N1. The macro averages are 12.67 and 13.50, a 6.6% increase. The per-domain log ratios are 0.033, 0.095, 0.017, so 66% of the summed log change comes from the second domain (0.095 / 0.145). The memo reports the per-domain list, because the macro average weights the high-perplexity domain most.

**Limit.** Paloma finds no single perplexity source that correlates with all 8 downstream tasks it tested; for example c4-en ranking correlates −0.77 with HellaSwag ranking (App. A, Table 3). Perplexity is a diagnostic here, and task scores decide.

### 3.2 Task suite roles

| Suite | Role for N1 (math) | Role for N2 (educational web) | Note |
|---|---|---|---|
| GSM8K (1,319) | target | broad | benchmark training split excluded from candidates |
| MATH | held-out partner of GSM8K | broad | reported, not used for decisions |
| MMLU, ARC-Challenge, OpenBookQA | broad | target | cloze formulation when multiple-choice format is near chance |
| HellaSwag, CommonsenseQA, WinoGrande | broad | broad (predicted loss) | [[smollm2]] Table 1 direction |
| MMLU-Pro, TriviaQA | held-out | held-out | SmolLM2 lists both as "not monitored during training" (Table 4) |

**Targeted and broad deltas.** For arm N and control N0: Δ_target = score_N − score_N0 on the arm's target suite; Δ_broad = mean over broad suites of (score_N − score_N0). Both carry paired intervals (§5). An arm is "targeted" when Δ_target's interval excludes zero and Δ_broad's lower bound is above −δ; it is "broad" when Δ_broad's interval excludes zero.

### 3.3 Run-to-run variation

The final 30 checkpoints of 1B models on ARC Challenge span 1.7% accuracy ([[signal-and-noise-eval]] §3.1), and bits-per-byte scoring improved small-scale decision accuracy for 90.0% of benchmarks (§5.3). The lab therefore (a) scores multiple-choice suites both as accuracy and as BPB of the correct answer, and (b) repeats the control arm with a second data order. The difference between the two control runs is reported next to every Part A delta. A candidate difference smaller than that control-to-control difference is reported as undecided.

**Selection rule for Part B.** N* is the arm with the highest Δ_broad lower bound among arms that pass §5's margin on every broad suite; N0 is chosen when no candidate passes. Part B starts from N*, and its short-context gate compares each cell with N*, not with the original checkpoint.

## §4 Part B: two extension methods at two long:short ratios

### 4.1 Base-frequency increase with continued pretraining (ABF)

**Definition.** RoPE rotates each query and key pair i at frequency θ_i = b^{−2i/d}. ABF (adjusted base frequency) replaces b with a larger b′ and continues pretraining at the target length ([[llama-2-long]] §4.1).

**Problem.** A pair whose wavelength λ_i = 2π/θ_i exceeds L₀ never completed a rotation in pretraining ([[yarn]] §3.2). For OLMo 2 1B (b = 500,000, d = 128, L₀ = 4,096), 32 of 64 pairs have λ_i > 4,096 (derived; computed in the figure). Unmodified RoPE with continued training could not attend beyond 4,000-6,000 tokens on first-sentence retrieval for Llama 2 7B ([[llama-2-long]] §4.1).

**Formula (NTK-aware base).**

```
b′ = b · s^{d/(d−2)},   s = L′/L₀
```

b is the pretrained base, b′ the new base, d the rotary head dimension, and s the scale factor ([[yarn]] App. A.2, Eq. 19).

**Worked examples (derived).**
- Llama-3-8B, 8K → 64K: s = 8, d = 128, b = 5×10⁵ gives b′ = 5×10⁵ · 8^{1.0159} = 4.13×10⁶. ProLong's Table 18 caption states that Dynamic NTK "roughly suggests to use 4m"; ProLong used 8×10⁶, which averaged 54.6 on long tasks against 48.7 at 4×10⁶ and 29.1 at the original 5×10⁵, with short averages 65.5, 65.3, 65.0 ([[prolong]] App. B.1). Result (single study).
- SmolLM2-1.7B, 2,048 → 8,192: s = 4, d = 64, b = 10,000 gives b′ = 41,829. SmolLM2 used 130,000 (3.1 times that value) without a printed ablation ([[smollm2]] §4.6).
- OLMo 2 1B, 4,096 → 16,384: b′ = 2.04×10⁶. The lab uses 2 × b′ = 4.09×10⁶, the ratio that ProLong found better at 64K. For the optional 8× cell, 2 × 4.13×10⁶ = 8.27×10⁶. The factor 2 is a course choice taken from one ProLong result at 8B and 64K; it has not been tested at 1B.

The figure [figures/rope-extension-explorer.html](figures/rope-extension-explorer.html) lets the reader set b, d, L₀, and L′, and see each pair's wavelength after ABF and after YaRN, the NTK-suggested base, the YaRN ramp from the paper and from transformers v4.46.0, and the attention-logit multiplier.

**Evidence on short context.** At 7B with 80B tokens at 32,768, ABF scored MMLU 46.24 vs PI 45.84 vs unmodified RoPE 45.69, and was the only variant that kept first-sentence retrieval to 32,768 ([[llama-2-long]] Table 6, Fig. 5b). Changing the base at inference without training lowered Llama-3-8B-Inst on ODQA and ICL at short lengths ([[helmet]] App. E.3). Result (single study each).

### 4.2 YaRN with continued pretraining

**Definition.** YaRN interpolates low-frequency pairs by the factor s, keeps high-frequency pairs, ramps between them, and scales attention logits ([[yarn]] §3.2-3.3).

**Formula.**

```
r(i) = L₀ / λ_i
γ(r) = 0 if r < α;  1 if r > β;  (r − α)/(β − α) otherwise
h(θ_i) = (1 − γ(r(i))) · θ_i / s + γ(r(i)) · θ_i
softmax( q_mᵀ k_n / (t √|D|) ),   √(1/t) = 0.1 ln s + 1
```

r(i) is the number of rotations pair i makes over L₀, γ the ramp, α = 1 and β = 32 the recommended bounds for Llama-family models, h the new frequency, t the temperature, and |D| the attention dimension as defined in [[yarn]] §2.1 (Eqs. 10-15).

**Implementation.** In transformers v4.46.0, `_compute_yarn_parameters` takes the pretrained length from `config.max_position_embeddings` (L192) when it computes the ramp bounds (L230), and the Llama rotary embedding multiplies cos and sin by `attention_factor` = 0.1 ln s + 1 at every sequence length ([[hf-rope-utils-v4460]]). The lab's YaRN cells therefore set:

```json
{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "beta_fast": 32.0, "beta_slow": 1.0},
 "max_position_embeddings": 4096, "rope_theta": 500000}
```

and train and evaluate with sequences up to 16,384 tokens. Before using this setting, check which config field supplies the pretrained length to the ramp in the installed transformers version and model class; the v4.46.0 code carries a TODO to use an `original_max_position_embeddings` field (L133).

**Worked example (derived).** For OLMo 2 1B, `find_correction_dim` gives 128 · ln(4096/(32 · 2π)) / (2 ln 500000) = 14.70 for β_fast and 31.60 for β_slow, so the ramp bounds are floor 14 and ceil 32: pairs 0-14 keep θ_i, pairs 32-63 use θ_i/4, and pairs 15-31 are mixed. If `max_position_embeddings` is set to 16,384, the same code returns bounds 21 and 39: pairs 15-21 (wavelengths 136 to 466 tokens, 30 to 8.8 rotations over 4,096) keep their original frequency, pairs 32-38 (wavelengths 4,443 to 15,204 tokens) are only partly interpolated, and only pairs 39-63 use θ_i/4. The attention factor for s = 4 is 1.1386, so logits are multiplied by 1.2965 (derived) for inputs of every length, including 512-token short-context items.

**Evidence.** On LLaMA 7B at s = 16 after 400 steps, YaRN had 32k proof-pile perplexity 2.77 against 3.57 for PI, but its MMLU was 30.0 against 35.7 for the base, 32.7 for NTK-by-parts, and 25.9 for PI ([[yarn]] Tables 2 and 5). At s = 32, Llama 2 13B MMLU fell from 55.8 to 51.9 (Table 3). In Lu et al.'s controlled setting (LLaMA2-7B, 1B tokens at 32k, scale factor 8.0 for PI and YaRN), YaRN scored 36.95 on RULER at 32k and 0.00 at 64k, and 6.70 PG19 perplexity at 2k against 6.61 for the base ([[controlled-long-context-extension]] Tables 2, 3, 7). Each is a Result (single study).

**Training-free reference cell Z.** YaRN with factor 4.0 is evaluated without training. HELMET reports that Qwen2-Inst with YaRN at inference dropped past 32,768 tokens ([[helmet]] App. E.3); cell Z shows what training adds over configuration alone.

### 4.3 Long:short ratio and data construction

**Definition.** The long share is the fraction of training tokens that come from single documents of at least L′ tokens; the short share is packed shorter documents. ProLong uses the same split: long data are single-document chunks and short data are packed ([[prolong]] §3).

**Evidence.** In ProLong's 5B-token Llama-3-8B ablations, short-task performance "monotonically decreases as the long data increases"; before SFT, recall and RAG preferred high long shares, after SFT they "drastically" deteriorated with more long data, and the best long average was at 60% long ([[prolong]] §3.2, Fig. 3). SmolLM2 used 40% documents of 8K tokens or more (§4.6). ProLong describes the Fu et al. mix as long documents constituting "roughly 70% of tokens" (§2.2). The lab tests 40% and 70%. Result (single study for the ratio effect). ProLong reports its ablation scores after UltraChat SFT unless stated otherwise (§2.2; App. A.4), while this lab gates base checkpoints, so only the direction of the ratio effect is carried over.

**Construction.**
1. Long sources: books and code repositories, the two best single long sources in ProLong Table 4 (books/repos 1:1 long average 54.6). Keep documents with at least 16,384 tokens in the checkpoint's tokenizer and cut them into single-document chunks of 16,384.
2. Short sources: the Part A pretraining mix, packed to 16,384 with the masking of §4.4.
3. Token accounting at 1B tokens per cell (derived): 40% long is 400M tokens, 24,414 chunks; 70% long is 700M tokens, 42,725 chunks. When the long pool is smaller, the repeat count per document is logged and must be equal across the ABF and YaRN cells.
4. Decontaminate long sources against every long-context suite in §6 with the rule of §2.3.

### 4.4 Document masking on versus off

**Definition.** With document masking, a token attends only to earlier tokens of its own document inside a packed sequence ([[prolong]] App. B.2).

**Evidence.** ProLong Table 20, scored after SFT: without masks long 53.6 / short 64.9; with masks 54.6 / 65.5; the number of runs per arm is not reported (Result, single study). Llama 3 applies a document-boundary mask and describes it as of "limited impact" in standard pretraining and "important" for very long sequences, without numbers ([[llama-3-recipe]], §3.2).

**Design.** Cell E5 repeats E1 (ABF, 40% long) with masking off. Record whether `position_ids` restart at each document in both cells. With restarting position ids, short packed documents never train positions beyond their own length, so the long-position signal comes from the long share only (Interpretation; no source isolates this effect).

### 4.5 Run matrix (full path)

| Cell | Start | Method | Long share | Masking | Tokens | Purpose |
|---|---|---|---|---|---|---|
| E1 | N* | ABF, b′ = 4.09×10⁶ | 40% | on | 1B | method × ratio |
| E2 | N* | ABF | 70% | on | 1B | method × ratio |
| E3 | N* | YaRN, factor 4.0 | 40% | on | 1B | method × ratio |
| E4 | N* | YaRN, factor 4.0 | 70% | on | 1B | method × ratio |
| E5 | N* | ABF | 40% | off | 1B | masking |
| Z | N* | YaRN, factor 4.0 | — | — | 0 | training-free reference |
| E1-32K (optional) | N* | ABF, b′ = 8.27×10⁶, L′ = 32,768 | 40% | on | 1B | 8× extension |

All training cells use the same peak LR, 2×10⁻⁵ (§Recipe), the same data order for shared sources, and the same step count.

## §5 The short-context regression gate

### 5.1 Paired intervals over items and documents

**Definition.** The gate compares each cell C with its start checkpoint N* on the same items at inputs no longer than L₀. For item i, d_i = score_C,i − score_N*,i.

**Formula** ([[adding-error-bars-evals]] §4.2, Eq. 7).

```
d̄          = (1/n) Σ_i d_i
SE_paired   = sqrt( (1/(n−1)) Σ_i (d_i − d̄)² / n )
CI_95%      = d̄ ± z · SE_paired
```

n is the number of items, d̄ the mean paired difference, and z = 1.96 for one comparison. For per-document Paloma NLL, d_i is the difference in mean NLL per token of document i.

**Worked example 1: a margin is not the same test as zero.** N* answers 600 of 1,000 items. Cell C loses 40 and gains 25, so d̄ = −0.015. Σd_i² = 65, Σ(d_i − d̄)² = 65 − 1,000 × 0.000225 = 64.775, SE_paired = sqrt(64.775/999/1,000) = 0.00805, and the interval is [−0.0308, +0.0008]. With a margin δ = 2 points, the lower bound −3.08 points is below −2, so the cell fails, although the interval includes zero. A cell that loses 15 and gains 10 has SE_paired = 0.00500 and lower bound −1.48 points, and passes. The unpaired standard error for the first cell is 0.0220, which would give [−5.8, +2.8] and fail every margin below 5.8 points.

**Worked example 2: perplexity by document.** Five documents have Δ mean NLL = [0.02, 0.05, −0.01, 0.03, 0.01] nats per token. d̄ = 0.02, Σ(d_i − d̄)² = 0.002, SE = sqrt(0.002/4/5) = 0.01, and the interval is [0.0004, 0.0396]. exp(0.02) = 1.020, a 2.0% perplexity increase whose interval excludes zero.

**Bootstrap.** For non-binary scores or small n, a paired bootstrap resamples item indices ([[ch-36]] §6.2 gives an enumerated example):

```python
import numpy as np   # course code, not taken from a source

def paired_gate(ref, new, delta, alpha=0.05, n_boot=10_000, seed=0):
    d = np.asarray(new, float) - np.asarray(ref, float)          # same items, same order
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))              # n_boot × n int64: 1.1 GB at n = 14,000; chunk as in ch-36 §6.2
    lo, hi = np.quantile(d[idx].mean(1), [alpha / 2, 1 - alpha / 2])  # Bonferroni: alpha = 0.05 / comparisons
    return {"mean": d.mean(), "ci": (lo, hi), "pass": lo > -delta}
```

### 5.2 Suites, margins, and multiple comparisons

1. Short suites: MMLU, ARC-Challenge, HellaSwag, WinoGrande, GSM8K (full test sets), plus the Paloma broad group. Held-out: MMLU-Pro, TriviaQA.
2. Margins δ per suite are written in the memo before Part B. The lab default is δ = 2 points for suites with at least 1,000 items and δ = 3 points for smaller ones (course choice, not from a source). Llama 3 advanced its context stages when short-context evaluations "fully recovered", without a number ([[llama-3-recipe]], §3.4.2).
3. With 6 cells × 5 suites = 30 comparisons, Bonferroni sets the per-comparison two-sided level to 0.05/30 = 0.00167, so z = 3.14 instead of 1.96 (derived). Worked example 1 then has lower bound −0.015 − 3.14 × 0.00805 = −4.03 points.
4. Every suite is scored at each cell's own RoPE configuration. For YaRN cells this includes the attention factor applied at short lengths (§4.2).

**What regressions looked like in sources.** Lu et al.'s extended LLaMA2-7B models lowered LongBench code completion LCC from 68.22 to 56.78 (Dynamic NTK), 55.05 (PI), and 54.06 (YaRN), a drop the paper does not discuss, while the LongBench average rose ([[controlled-long-context-extension]] Table 11). An average-only gate would pass these cells; the per-suite gate would not.

## §6 Long-context gates: RULER effective length and a HELMET category

### 6.1 RULER effective length for a small checkpoint

**Definition.** RULER generates 13 synthetic tasks across retrieval, multi-hop tracing, aggregation, and QA at chosen lengths, and defines effective length as "the maximum length exceeding the threshold". The threshold is the Llama2-7B score at 4K: 85.6 for the chat model when aligned models are graded (Table 3) and 79.4 for the base model when base models are graded (App. A, Table 12) ([[ruler-effective-length]]; [[ruler]]).

**Problem.** The absolute threshold does not transfer to small models or other harnesses. Yarn-base (7B), claimed at 128K, scored 77.3 at 4K and received "<4K" under the 79.4 rule (Table 12). Lu et al. report 80.94 for LLaMA2 at 4k in their configuration, 1.5 points above RULER's own base-model score ([[controlled-long-context-extension]] Table 3). A 1B checkpoint that scores below 79.4 at its native length receives "<4K" at every length, so the rule cannot rank its cells.

**Course rule (not from a source).** τ = RULER average of N* at L₀ in the lab's harness minus 5 points. Effective length = the largest tested L such that every tested length ≤ L scores at least τ. Lengths: 2K, 4K, 8K, 16K, and 32K on the full path; 2K, 4K, 8K, and 16K on the constrained path. The RULER base-model rule (79.4) is reported in a separate column.

**Worked example (illustrative scores).** N* scores 62.0 at 4K, so τ = 57.0. Cell E1 scores 61.2, 59.8, 57.4, 31.0 at 4K, 8K, 16K, 32K: effective length 16K. Cell E3 scores 60.9, 58.1, 55.2, 12.4: effective length 8K, because its 16K score, 2.2 points below E1's, falls below τ.

**Sample size.** The paper uses 500 examples per task per length (§4). With 100 per task, 1,300 items per length at 80% accuracy give a half-width of 2.17 points; 500 per task give 0.97 (derived, treating items as independent). The lab uses 100 per task for cell screening and 500 for the chosen cell and its runner-up.

**Evidence that length curves differ from short scores.** Llama3.1 (8B) scores 95.5 at 4K and 77.0 at 128K with effective length 32K; Mistral-v0.2 (7B) falls from 93.6 at 4K to 13.8 at 128K (Table 3). RULER notes "abrupt performance drops" at unseen lengths and "almost linear degradation with input length on log scale within the max training context size" (§6).

### 6.2 HELMET RAG as the application category

**Definition.** HELMET's RAG category places gold passages among retrieved hard-negative passages from Wikipedia and scores substring exact match ([[helmet]] §2.1).

**Why RAG.** The authors recommend RAG for fast development because it is easy to run, works with base models, and correlates with other downstream categories more than synthetic recall does; HotpotQA correlates with ∞Bench QA at 0.88 against 0.63 for NIAH (Abstract; §3.1; Fig. 4). Categories differ: ICL correlates with the other categories at 0.34-0.61 (Fig. 5), so one category is a gate, not a full evaluation.

**Protocol.**
1. Lengths 8,192 and 16,384 (and 32,768 for E1-32K). HELMET defines lengths in Llama-2 tokens (§3); re-measure each input in the checkpoint's tokenizer and truncate so no input exceeds L′.
2. Use HELMET's two-shot demonstrations for base checkpoints. At 128K, two-shot prompting raised Llama-3.1-8B base from 77.3 to 98.0 on JSON KV (Table 8).
3. Greedy decoding; 600 samples for NQ, PopQA, TriviaQA and 300 for HotpotQA, as in App. D.
4. Reference points at 8K input: SmolLM2-1.7B base 47.17, Llama3.2-1B 42.13, Qwen2.5-1.5B 47.54 ([[smollm2]] Table 11).

### 6.3 What the gate does not use

Perplexity at L′ is logged but does not choose cells, because it favored 100% long data while downstream long scores fell ([[prolong]] Fig. 1). Lu et al. found PPL@32K correlated with downstream scores for exact-attention fine-tuned methods, while LM-Infinite, an approximate-attention method, had good perplexity at 32k and failed NIAH beyond 4k ([[controlled-long-context-extension]] §6). Passkey retrieval is logged; [[llama-2-long]] found all variants except unmodified RoPE reached perfect passkey accuracy and considered it "too simple" (§4.1, fn. 4).

## §7 Gate table and stage memo

**Gate table (illustrative numbers; the lab fills in measured values).**

| Cell | Paloma broad macro Δ% [CI] | Worst short suite Δ (points) [Bonferroni CI] | Suites failing δ | RULER eff. length (course rule / 79.4 rule) | RULER avg @16K | HELMET RAG @8K / @16K | Gate |
|---|---|---|---|---|---|---|---|
| N* | 0 | 0 | — | 4K / <4K | 38.0 | 30.1 / 11.5 | reference |
| Z | +6.1 [+5.2, +7.0] | HellaSwag −4.1 [−5.9, −2.3] | 3 | 8K / <4K | 44.9 | 33.0 / 25.2 | fail |
| E1 | +1.2 [+0.6, +1.8] | GSM8K −0.6 [−1.8, +0.6] | 0 | 16K / <4K | 57.4 | 36.8 / 34.0 | pass |
| E2 | +2.3 [+1.6, +3.0] | MMLU −2.5 [−3.6, −1.4] | 1 | 16K / <4K | 58.8 | 37.0 / 35.1 | fail |
| E3 | +1.9 [+1.2, +2.6] | ARC-C −2.2 [−4.9, +0.5] | 1 | 8K / <4K | 55.2 | 35.5 / 30.2 | fail |

**Decision procedure.**
1. Remove cells with any suite below −δ at the Bonferroni interval, or with a Paloma broad increase whose lower bound exceeds the perplexity margin written in the memo.
2. Among remaining cells, choose the largest course-rule effective length; break ties by HELMET RAG at L′, then by the smaller worst-suite loss.
3. Rerun the chosen cell and the runner-up with a second data order; report both runs.
4. Evaluate the held-out suites once, after the choice is committed.

**Memo sections.** (1) Setup and token counts per source. (2) Part A table: Δ_target and Δ_broad per arm with intervals and the control-to-control difference. (3) Gate table. (4) Chosen recipe with every setting as loaded. (5) "Scores lost": each short suite and Paloma domain where the chosen cell's point estimate is below N*, with interval and whether it passed δ. (6) Hypotheses of §1 marked supported, contradicted, or undecided.

**Acceptance criteria.** Margins and hypotheses are committed before Part B; the RoPE settings logged at load time equal the intended ones, including ramp bounds 14 and 32 for YaRN cells; every cell has per-item files; no cell is dropped after evaluation.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value | Lab value | Reason for difference |
|---|---|---|---|---|---|---|---|---|---|
| OLMo 2 1B | 1B | pretrain-stable | tokens; schedule; warmup; peak LR; batch (sequences) | 4T tokens; cosine set to 5T, truncated at 4T; 2,000 warmup steps; 4.0×10⁻⁴; 512 | arXiv:2501.00656v3 App. B ([[olmo-2-annealing]]) | verified 2026-09-15 | no ablation reported for 1B | start checkpoint | — |
| OLMo 2 1B | 1B | pretrain-decay/anneal | anneal data; tokens | Dolmino Mix 1124; single 50B-token anneal | App. B | verified 2026-09-15 | no ablation reported for 1B (Table 11 compares mixes at 7B) | 5B tokens per arm | three arms plus a repeat within budget; microanneals showed results under 10B tokens (§4.4.2) |
| OLMo 2 7B microanneals | 7B | eval-gate | mixture; tokens; LR | ~50/50 candidate/web; 576M-12.6B tokens; LR linearly to 0 | §4.4.2, Table 12 | verified 2026-09-15 | Table 12 | 50/50; linear to 0 | none |
| OLMo 2 7B mid-training comparison | 7B | mid-train | init; tokens; control | 4T checkpoint; 50B tokens; "PT Mix" row | Table 11 | verified 2026-09-15 | Table 11 | control arm N0 at 5B tokens | budget |
| Llama 3 8B (experiment) | 8B | eval-gate | anneal-as-data-evaluation | 50%-trained; LR linearly to 0 over 40B tokens; 30% new data, 70% default mix | arXiv:2407.21783v3 §3.1.3 ([[llama-3-recipe]]) | verified 2026-09-14 | no ablation reported | 50% candidate | follows the OLMo 2 ratio |
| SmolLM2 1.7B (ablation) | 1.7B | eval-gate | math anneal | 3T checkpoint; 60B candidate + 40B pre-checkpoint tokens; LR linearly to 0 | arXiv:2502.02737v1 §3.1 ([[smollm2]]) | verified 2026-09-15 | Fig. 1, Fig. 5 | not used | budget |
| SmolLM2 1.7B | 1.7B | long-context | length; RoPE; mixture; start | 2K → 8K; θ 10,000 → 130,000; 40% documents ≥ 8K (DCLM 10%, FineWeb-Edu 10%, Dolma books 20%) + 60% stage-4 mix; checkpoint before the final 75B tokens; extension tokens not reported | §4.6, Table 6 | verified 2026-09-15 | no ablation reported | 40% long arm | used as the lower ratio |
| ProLong-64k-Base | 8B | long-context | RoPE base | 8×10⁶ (NTK suggestion about 4×10⁶) | arXiv:2410.02660v4 Table 9, App. B.1 ([[prolong-recipe]]) | verified 2026-09-14 | Table 18: 54.6 vs 48.7 vs 29.1 long average | 2 × NTK value (4.09×10⁶ at 16K) | same ratio to the NTK value; not tested at 1B |
| ProLong ablations | 8B | long-context | tokens; length; long share; masking | 5B at 64K; 60% long / 40% ShortMix; document masks | App. A.4, §3.2, Table 20 | verified 2026-09-14 | Fig. 3; Table 20 | 1B tokens at 16K; 40% and 70%; masking on except E5 | budget; the ratio is the tested axis |
| ProLong-64k-Base | 8B | long-context | peak LR; warmup; decay; batch | 1e-5; 10%; cosine to 1e-6; 4M tokens | Table 9 | verified 2026-09-14 | no ablation reported | 2×10⁻⁵ | 7B-scale extension rows below use 2×10⁻⁵ from base checkpoints; ProLong's final run starts from an Instruct model |
| Yarn-Llama-2 s = 16 | 7B, 13B | long-context | LR; warmup; weight decay; steps; batch; α, β | 2×10⁻⁵; 20 steps; 0; 400; 64; α = 1, β = 32 | arXiv:2309.00071v3 §4.1, §3.2 ([[yarn]]) | verified 2026-09-14 | Table 6 (400 YaRN steps ≈ 1,000 PI steps) | factor 4.0; β_fast 32, β_slow 1 | factor set by L′/L₀ |
| LLaMA2-7B extensions (Lu et al.) | 7B | long-context | length; tokens; LR; weight decay | 32k; 1B; 2e-5; 0 | arXiv:2409.12181v2 §4, App. 9.2 ([[controlled-long-context-extension]]) | verified 2026-09-14 | App. 9.4: NTK-64K with 2B tokens improved NIAH over 1B | 1B tokens per cell | none |
| Llama 2 Long | 7B, 13B | long-context | peak LR; schedule; warmup; RoPE base | 2e-5; cosine; 2,000 steps; 10,000 → 500,000 | arXiv:2309.16039v3 §2.1 ([[llama-2-long]]) | verified 2026-09-14 | Tables 5-6, Fig. 5b (7B) | ABF cells | — |
| LLaMA 7B, 13B (PI) | 7B, 13B | long-context | peak LR; optimizer; warmup; weight decay | 2×10⁻⁵; AdamW (0.9, 0.95); 20 steps; 0 | arXiv:2306.15595v2 §3.1 ([[position-interpolation]]) | verified 2026-09-14 | no ablation reported | LR shared by all cells | — |
| Llama 3.1 405B | 405B | long-context | stages; tokens; advance criterion | six stages 8K → 128K; about 800B tokens; advance when short-context evaluations "fully recovered" and NIAH solved | arXiv:2407.21783v3 §3.4.2 ([[llama-3-recipe]]) | verified 2026-09-14 | no numbers | paired CI with margin δ | the report gives no numeric criterion |
| RULER | — | eval-gate | tasks; examples; threshold | 13 tasks; 500 per task per length; threshold 85.6 (Llama2-7B chat at 4K) for aligned models, 79.4 (Llama2-7B base) for base models | arXiv:2404.06654v3 §4, Table 3, App. A, Table 12 ([[ruler-effective-length]]) | verified 2026-09-15 | not applicable | 100 per task (screening), 500 (final); τ relative to N* | compute; small-model scale |
| HELMET | — | eval-gate | lengths; demos; RAG metric; samples | 8K-128K Llama-2 tokens; 2-shot; SubEM; 600 (NQ, PopQA, TQA), 300 (HotpotQA) | arXiv:2410.02694v3 §2-3, App. D ([[helmet]]) | verified 2026-09-14 | Fig. 4; Table 8 | RAG at 8K and 16K in checkpoint tokens | tokenizer differs from Llama 2 |

**Starting point for a small general-purpose run.** For an anneal used as a data test, train a 50/50 mixture of the candidate and the pretraining mix with the learning rate decayed linearly to zero, and train a control on the pretraining mix with the same tokens and schedule; OLMo 2 used 576M-12.6B-token microanneals from a 7B checkpoint at 4T tokens. For continued pretraining at a longer length from a base checkpoint, a peak learning rate of 2×10⁻⁵ is the value used by PI, YaRN, Lu et al., and Llama 2 Long for 7B-13B Llama models, with 1B tokens at 32k in Lu et al.'s comparison; none of these sources tested it below 7B. For the RoPE base, ProLong's 5B-token Llama-3-8B ablation runs at 64K used about twice the NTK-aware value, 8×10⁶ from a pretrained base of 5×10⁵, with 60% long and 40% short data and document masks; this was tested only at 8B.

## Generalization lens

**(a) What increases breadth.**
- A small share of domain data in an anneal: 10% math in 1.72B tokens raised GSM* from 28.5 to 61.0 while MMLU rose from 59.8 to 60.9 at 7B ([[olmo-2-annealing]] Table 12).
- Keeping short data in extension training: 40% short data gave the best long average and higher short scores than larger long shares ([[prolong]] Fig. 3); a high-quality short mix (ShortMix) gave short average 65.5 against 63.0 for FineWeb-Edu alone (Table 6).
- Averaging anneals over data orders: soups equaled or beat the best single run on OLMES in all six mixes of [[olmo-2-annealing]] Table 14, with the GSM* exception of mix E.
- Document masking in packed extension data: +1.0 long and +0.6 short average, scored after SFT, in one ProLong comparison (Table 20).

**(b) What causes narrowing or forgetting.**
- Benchmark training sets in anneal data: +24.0% GSM8k at 8B and negligible at 405B, which led Llama 3 to exclude them in order "to assess the true few-shot learning capabilities and out-of-domain generalization" ([[llama-3]], §3.1.3).
- Position rescaling with larger extension factors: PI to 32,768 lowered 7B BoolQ inside the original window from 76.1 to 64.7 ([[position-interpolation]] Table 5); YaRN MMLU 55.8 → 51.9 at s = 32 for 13B ([[yarn]] Table 3).
- Long-only training: after SFT, recall and RAG fell with more long data even where they rose before SFT ([[prolong]] §3.2).
- Task-level loss hidden in averages: LongBench LCC 68.22 → 54.06 with YaRN while the average rose from 32.92 to 33.45 ([[controlled-long-context-extension]] Tables 4, 11).

**(c) How to measure it at this stage.**
- Paired per-item and per-document differences against the stage's start checkpoint, with pre-registered margins (§5; [[adding-error-bars-evals]] §4.2).
- A control arm for every anneal and a repeated control for run-to-run variation (§2, §3.3; [[signal-and-noise-eval]] §3.1).
- Per-domain perplexity with separately scored documents ([[paloma]] G5, Table 17).
- Effective length from task families and one application category rather than perplexity or passkey retrieval (§6; [[helmet]] Fig. 4; [[prolong]] Fig. 1).
- Evaluation after SFT for a long-context decision that affects the final model: ProLong's RAG and re-ranking trends appeared only after SFT ([[prolong]] §2.2, Fig. 2). This lab gates base checkpoints; [[ch-30b]] covers long-context shares in SFT.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| No control anneal | every candidate "improves" broad scores by a similar amount | N0 arm; report candidate − control (§2.2) |
| Candidate contains benchmark training splits | target gain without a held-out partner gain | decontamination report; MATH and MMLU-Pro partners (§2.3, §3.2) |
| Deciding broad effects on a 200-item subset | ±1-2 point changes reported as effects | half-width computed before use: 6.3 points at 28.5% on 200 items (§2.4) |
| Concatenated documents in perplexity evaluation | high variance across evaluation subsets | score documents separately ([[paloma]] Table 17) |
| YaRN with `max_position_embeddings` set to L′ in transformers v4.46.0 | ramp bounds 21 and 39 instead of 14 and 32 for OLMo 2 1B | log bounds at load time; test against the figure's values (§4.2) |
| Short-context gate run without the YaRN attention factor | short scores differ between training-time and serving configs | evaluate each cell with the exact loaded config (§5.2) |
| Choosing the ratio or method by perplexity at L′ | 70%-long cell chosen; short scores and RAG lower | decision procedure in §7 uses RULER, RAG, and paired short suites |
| RULER absolute threshold (79.4 base, 85.6 aligned) on a 1B model | every cell gets "<4K" | relative threshold τ reported beside the absolute rule (§6.1) |
| HELMET lengths taken as checkpoint tokens | inputs exceed L′ or fall short of it | re-tokenize and log input lengths (§6.2) |
| Long pool repeated differently across cells | method difference changes when repeats are equalized | repeat count per cell in `runs.jsonl` (§4.3) |
| Average-only short gate | cell passes while one suite drops beyond δ | per-suite Bonferroni intervals (§5.2) |
| Masking comparison changes position ids too | masking effect confounded with position reset | log `position_ids` policy for E1 and E5 (§4.4) |
| Reporting "no degradation" without numbers | memo lacks the scores-lost list | memo section (5) required (§7) |

## Check your understanding

1. The math arm raises GSM8K by 20 points over the control, and MATH does not change. Give two causal accounts, and state which lab file distinguishes them.
2. Explain why an anneal on the unchanged pretraining mix raises MMLU, and why that result makes a candidate − start comparison biased toward the candidate.
3. Using the NTK-aware formula, explain why the base suggested for Llama-3-8B at 64K is close to 4×10⁶, and give a mechanism by which a base twice as large could improve long-task scores without lowering short scores in ProLong Table 18.
4. A YaRN cell passes RULER at 16K but fails the HellaSwag margin at 512-token inputs. Trace, through the transformers code in §4.2, what changes attention at short lengths, and propose a configuration change and the observation that would test it.
5. Cell E2 (70% long) has a higher RULER average at 16K than E1 (40% long) but fails one short suite. Using ProLong's before-SFT and after-SFT results, explain why choosing E2 could also lower long-context scores after SFT.
6. In Worked example 1, the interval includes zero and the cell fails. Explain why "the interval includes zero" is not evidence of no regression, and what quantity the margin δ controls.
7. The course rule defines effective length relative to N* at L₀. Describe a case in which a cell has a longer relative effective length than another cell but is the worse long-context model, and name the gate column that detects it.
8. E5 (masking off) scores higher on RULER than E1 but lower on short suites. Propose a mechanism involving attention across unrelated documents and position ids, and an experiment that separates the two.

## Connections

- Previous chapter: [[ch-32e]] — Mid-Training, Annealing, and Context-Extension Recipes Side by Side.
- Next chapter: [[ch-18]] — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix.
- [[ch-32]] — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL (annealing data and stage gates run in Part A).
- [[ch-32a]] — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining (re-warming when the start checkpoint is already decayed).
- [[ch-32b]] — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression (the methods run in Part B).
- [[ch-32c]] — Claimed versus Effective Context Length and Long-Context Evaluation (RULER and HELMET in depth).
- [[ch-00]] — What General Capability Means and How It Is Measured (development and held-out suites).
- [[ch-04]] — Sequence Packing, Loss Masking, and Chat Templates (document masking and position ids).
- [[ch-06]] — Checkpointing, In-Loop Evaluation, and Checkpoint Selection (run-to-run variation and checkpoint choice).
- [[ch-36]] — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split (paired bootstrap, seeds).
- [[ch-30b]] — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares (keeping long-context ability through SFT).
- [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions (multiple comparisons and decision rules).

## Sources

- [[olmo-2-annealing]] — chapter excerpt of OLMo 2 v3: Table 3, §4.1, Table 11 control row, §4.4.2 microanneal recipe and Table 12, Table 14 soups, App. B 1B settings.
- [[smollm2]] — chapter excerpt of SmolLM2 v1: annealing ablation setup, FineMath and Table 1 web results, decay stage, 2K → 8K extension with RoPE 130k and 40% long documents, HELMET Table 11, 360M/135M configs.
- [[prolong]] and [[prolong-recipe]] — long:short ratio (Fig. 3), perplexity vs downstream (Fig. 1), after-SFT evaluation (Fig. 2), RoPE base (Tables 18-19), document masks (Table 20), short-context losses of prior methods (Table 2), recipe rows.
- [[yarn]] — NTK-aware base formula, NTK-by-parts ramp, temperature, short-benchmark costs (Tables 2-3), recipe rows.
- [[hf-rope-utils-v4460]] — chapter excerpt of transformers v4.46.0 dynamic NTK and YaRN code and where the attention factor is applied.
- [[controlled-long-context-extension]] — Lu et al.: fixed-recipe comparison of PI, YaRN, and NTK variants; RULER, PG19 at 2k, LongBench LCC drops.
- [[position-interpolation]] — PI and its original-window regression (Table 5); 2×10⁻⁵ LR row.
- [[llama-2-long]] — ABF definition and 7B PE ablation (Tables 5-6, Fig. 5b); recipe row.
- [[llama-3]] and [[llama-3-recipe]] — anneal-as-data-evaluation, benchmark-train-set annealing result, document mask, long-context stage criterion.
- [[ruler]] and [[ruler-effective-length]] — task generators; effective-length rule, aligned-model Table 3 and base-model Table 12 rows.
- [[helmet]] — RAG category, correlations with downstream tasks, two-shot base-model evaluation, inference-time position changes (App. E.3).
- [[paloma]] — per-domain perplexity, macro average, separate-document evaluation (G5), perplexity-to-task correlation limits.
- [[signal-and-noise-eval]] — checkpoint noise at 1B and bits-per-byte scoring for small-scale decisions.
- [[cooldown-scaling-beyond-fixed-durations]] — cooldown shape comparison (linear vs 1-sqrt).
- [[adding-error-bars-evals]] — chapter excerpt of Miller's paired standard error (Eq. 7).
