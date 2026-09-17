<!-- chapter: ch-32a
     track: midtraining
     kind: content
     title: Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining
     deps: [ch-32]
     sources: [[continual-pretraining-rewarm-replay]], [[paloma]], [[grpo]], [[grpo-recipe]], [[qwen-2-5-coder]], [[qwen-2-5-math]], [[instruction-pretraining]], [[synthetic-continued-pretraining]], [[merging-in-pretraining]], [[cooldown-scaling-beyond-fixed-durations]], [[kimi-k2-5]], [[llama-3]], [[rephrasing-the-web]]
     figures: figures/replay-tradeoff.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining

> **Core insight.** Continued pretraining on new data trades adaptation to the new data against loss of ability on the old data, and two controls set the trade-off. In controlled 405M runs, re-warming the learning rate and re-decaying it was needed to adapt to the new data, and replaying a fraction of the old data reduced forgetting on the old data while raising new-data loss by 0.00–0.05 across the tested fractions up to 25% (the replay result also held at 10B for a same-language shift): after 300B Pile tokens and 200B German tokens, 25% replay cut the Pile loss increase from 1.39 to 0.16 while German loss rose from 1.11 to 1.16, matching the average loss of a model trained from scratch on both datasets ([[continual-pretraining-rewarm-replay]], Table 2). Specialist reports show the same effect in the mixture: Qwen2.5-Coder-7B trained on 100% code scored MMLU 42.8, and the 70:20:10 code:text:math mixture scored 62.9 while also scoring higher on math ([[qwen-2-5-coder]], Table 3). When the new corpus has about 1M tokens, next-token training on it can fail to add usable knowledge: 1.3M raw tokens lowered closed-book QA from 39.49% to 38.15%, while 455M EntiGraph-synthesized tokens raised it to 56.22% ([[synthetic-continued-pretraining]], Fig. 2).
>
> **Guideline.** When a base model trained with a cosine schedule to a low learning rate is continued on 100B or more new tokens, and no pre-decay checkpoint is available, re-warm to a value near the original peak and cosine re-decay, because a constant low learning rate adapted least in Ibrahim et al. ([[continual-pretraining-rewarm-replay]] §6.1.2). When a pre-decay or constant-phase checkpoint is available, start from it, because re-warming raised loss even on unchanged data (§7.1) and constant-phase checkpoints showed negligible forgetting across same-distribution dataset boundaries (§7.4; not tested with a distribution shift). When the new data differ from the old in language or domain, replay old or general data and choose the fraction with a short sweep, because 1% replay already reduced forgetting and the selected fractions were 5% for a weak shift and 25% for a language shift at 405M. When the new corpus is about 1M tokens, the scale tested by Yang et al., expand it with source-grounded synthetic entity-relation text before continued pretraining, because raw training lowered closed-book QA and paraphrase-only expansion scaled worse ([[synthetic-continued-pretraining]] Fig. 2). Instruction-augmented formats ([[instruction-pretraining]]) were tested on domain corpora trained for 1B tokens, not at the 1M-token scale. In every case, report per-domain held-out perplexity and downstream deltas against the exact checkpoint used for initialization, because five of the seven continued-pretraining runs tabulated in §4 do not print the general scores of their starting checkpoint.

## Why this chapter matters for a general-purpose model

The reports in this chapter continue a trained base model to add a modality (Kimi K2.5), a code or math specialization (Qwen2.5-Coder, Qwen2.5-Math, DeepSeekMath), a new language or newer web data (Ibrahim et al.), or a small document collection (EntiGraph, instruction pre-training). Each of these runs starts from a base model that already has broad ability, and each run can remove part of that ability. This chapter covers the part of mid-training (ch-32) whose purpose is adding new data without narrowing the model. Context-length extension, which has its own regression problems, follows in ch-32b.

The pipeline position is: pre-training → mid-training and continued pretraining (this chapter) → SFT → preference optimization → RL → evaluation. Forgetting that happens here is inherited by every later stage. A base model that has lost multilingual or general-knowledge ability before SFT cannot recover it from a post-training mixture that does not contain that ability (Interpretation). The measurable questions are:

1. Adaptation: how far does loss on the new data fall, and do the target benchmarks rise?
2. Forgetting: how far does loss on the old data rise, per domain, and which downstream scores fall relative to the starting checkpoint?
3. Cost: how many tokens and how much compute does the continued run use compared with retraining from scratch on all data?

The fine-tuning forgetting mechanisms and the alignment-tax measurements used after SFT and RL are taught in ch-30a. This chapter applies the same measurement logic at the pretraining scale, where the runs in this chapter range from about 5M tokens seen (Raw CPT, 1.3M tokens × 4 epochs) to 15T tokens (Kimi K2.5).

## §1 Forgetting in continued pretraining and how to measure it

**Definitions.** *Continued pretraining* (also *continual pretraining*, CPT) is further training of a pretrained model with the pretraining objective, next-token prediction on all tokens, on a dataset D1 after an earlier dataset D0. *Adaptation* is the decrease in loss on held-out D1 data or the increase in D1-related task scores. *Forgetting* is the increase in loss on held-out D0 data or the decrease in scores on tasks the starting checkpoint could do. *Domain-adaptive* continued pretraining uses a D1 that is narrower than D0 (code, math, biomedicine), which is the case with the highest forgetting risk for a general model.

**Problem.** Ibrahim et al. name the two failure modes of naive continued training: poor adaptation ("failure to optimize the new dataset") and catastrophic forgetting ("significant capability loss on the previous dataset") ([[continual-pretraining-rewarm-replay]] §1). Both must be measured, because each control in this chapter improves one of them at the expense of the other.

**Formulas.**

```
Δforget(d) = L_d(θ_after) − L_d(θ_before)
Δadapt     = L_D1(θ_before) − L_D1(θ_after)
Δtask(k)   = s_k(θ_after) − s_k(θ_before)
```

- `L_d(θ)`: mean per-token cross-entropy (nats) of parameters θ on held-out documents of domain d.
- `θ_before`: the exact checkpoint used to initialize continued pretraining; `θ_after`: the checkpoint after it.
- `D1`: the new dataset; `s_k`: the score on downstream task k with a fixed prompt format and few-shot count.

Paloma defines perplexity over a set of documents N as exp(−ℓ / T(N)), with ℓ the summed token log-likelihood and T(N) the token count, and reports a *macro average* over domains, |D|⁻¹ Σ_d perplexity(d), instead of a single token-weighted (*micro*) average ([[paloma]] §3, §4.1).

**Worked example (derived).** A base model has loss 2.0 on domain A (perplexity e^2.0 = 7.39) and 3.0 on domain B (perplexity 20.09). A validation set holds 900K tokens of A and 100K tokens of B.
1. Micro perplexity before: exp(0.9·2.0 + 0.1·3.0) = exp(2.10) = 8.17. Macro perplexity before: (7.39 + 20.09)/2 = 13.74.
2. Continued pretraining leaves A unchanged and raises B's loss to 3.5 (perplexity 33.12).
3. Micro perplexity after: exp(0.9·2.0 + 0.1·3.5) = exp(2.15) = 8.58, an increase of 5%. Macro perplexity after: (7.389 + 33.115)/2 = 20.25, an increase of 47%.
The token-weighted number hides forgetting on the smaller domain. This is the reason to track Δforget(d) per domain.

**Evidence that per-domain measurement changes conclusions.**
- Among 6 controlled 1B baselines, the C4 and mC4-en models got worse between the ~20B and ~150B-token checkpoints on 65 and 43 Paloma domains, while baselines with curated non-web sources improved with a stable gap ([[paloma]] App. D.1.1). The authors present per-domain perplexity as exposing gaps that one held-out loss hides. Result (single study).
- In Paloma, no single perplexity source correlated with all 8 downstream tasks tested (for example, c4-en vs HellaSwag Spearman ρ −0.77, RedPajama vs HellaSwag 0.94), so perplexity panels must be read together with downstream deltas ([[paloma]] App. A, Table 3).

**Measurement errors specific to continued pretraining.**
1. *Wrong reference checkpoint.* DeepSeekMath-Base 7B was initialized from the DeepSeek-Coder-Base-v1.5 checkpoint "right before learning rate decay". Its MMLU was 42.9%, while the released, decayed coder base scored 49.1%. After 500B tokens of math-heavy training, DeepSeekMath-Base scored 54.9%. The MMLU gain is +12.0 against the real initialization and +5.8 against the released model ([[grpo]], Table 4). A report that compares against the released model mixes the effect of the decay with the effect of the new data.
2. *Missing reference rows.* Qwen2.5-Math base models were initialized from Qwen2.5 base models, but Table 2 of the report lists Qwen2 base models, not Qwen2.5 ([[qwen-2-5-math]]). The retention delta is not reported.
3. *Evaluation noise at small scale.* At 405M, the English evaluation averages of five models spanned 1.19 points, and the authors state that the differences "are likely not significant"; all runs are single-seed ([[continual-pretraining-rewarm-replay]] §6.3.2, §8).
4. *Contamination between train and validation splits.* Ibrahim et al. did not deduplicate the German validation set against the German training data ([[continual-pretraining-rewarm-replay]] §8), and Paloma's G1 rule removes training documents that match evaluation paragraphs ([[paloma]] §3). A D1 validation loss without decontamination can overstate adaptation (Interpretation).

**Implication for a general-purpose model.** The minimum forgetting report for a continued run is: per-domain held-out perplexity for the old mixture's main sources (macro-averaged, decontaminated), the D1 held-out loss, and downstream scores for the general suite, all computed on θ_before and θ_after with the same harness.

## §2 Learning-rate re-warming and re-decaying

**Definition.** *Re-warming* raises the learning rate again at the start of continued pretraining, after the base model's schedule ended at a low value. *Re-decaying* lowers it again over the continued run, usually with a cosine schedule sized to the new token budget.

**Problem.** Ibrahim et al. state that "most performant open-source LLMs" (they cite Llama, Llama 2, Mistral, and Gemma) decay the learning rate to a small value by the end of training ([[continual-pretraining-rewarm-replay]] §1). In their runs, continuing at the final minimum learning rate gave the least adaptation to D1 (§6.1.2). Raising it gives adaptation and also moves the parameters away from the D0 solution.

**Mechanism.**
1. Load the final checkpoint of D0 training. In Ibrahim et al. the optimizer states are reset, because states are often unavailable for open-weight models (§5.2).
2. Warm up linearly from 0 to η_max over T_warmup steps.
3. Decay with cosine to η_min at the end of the D1 budget.
4. Optionally mix replay data into every batch (§3).

**Formula** (Ibrahim et al. Eq. 1–2):

```
η_t = η_max · t / T_warmup                                                   for t ≤ t_ann = T_warmup
η_t = η_min + (η_max − η_min)/2 · ( cos(π · (t − t_ann)/(t_end − t_ann)) + 1 )   for t_ann < t ≤ t_end
```

- `η_t`: learning rate at iteration t; `η_max`: peak; `η_min`: final value (0.1 · η_max in the paper's runs).
- `T_warmup`: warmup iterations (1% of the D1 iterations); `t_end`: last iteration of the D1 budget.

**Worked example (derived).** The 405M weak-shift run uses η_max = 3·10^-4 and η_min = 3·10^-5 over 132,366 iterations, batch 1104 sequences of 2048 tokens (Table 13 for batch and sequence length, Table 14 for learning rates, §5.3, §6.1.1).
1. Tokens per iteration: 1104 × 2048 = 2,260,992; over 132,366 iterations this is 299.3B tokens, the size of the SlimPajama subset.
2. Warmup: 1% of 132,366 = 1,324 iterations.
3. At 25% of the decay: cos(π/4) = 0.7071, η = 3·10^-5 + 1.35·10^-4 · 1.7071 = 2.60·10^-4.
4. At 50% of the decay: cos(π/2) = 0, η = 3·10^-5 + 1.35·10^-4 = 1.65·10^-4.
5. At the end: cos(π) = −1, η = 3·10^-5.

**Evidence.**
- *Re-warm vs constant.* At 405M, for both shifts, "the constant ηmin learning rate model achieves the least forgetting on D0", and the re-warmed and re-decayed models "adapt better to the new dataset by a significant margin" ([[continual-pretraining-rewarm-replay]] §6.1.2, Fig. 4). Among re-warmed runs with η_max ∈ {1.5, 3, 6}·10^-4, "higher values of ηmax lead to more forgetting and more adaptation". Result (single study, one seed).
- *Warmup length.* Warmups of 0%, 0.5%, 1%, and 2% gave "relatively similar forgetting and adaptation" after 50B tokens; no warmup produced a transient loss spike (§6.1.1, Fig. 3).
- *Re-warming on unchanged data.* Continuing on the Pile itself, re-warming to 3·10^-4 raised Pile validation loss by a peak of 0.1, and to 6·10^-4 by 0.2; on SlimPajama the peaks were 0.35 and 0.45 (§7.1, Fig. 8). The authors conclude that re-warming "appears to be a significant cause" of the early loss increase, independent of the data shift (Interpretation, single study).

**Alternatives that avoid re-warming.**
- *Infinite learning-rate schedules.* Warmup, cooldown to a constant η_const, a constant phase from which future runs resume, and a short final anneal before release. At 405M on three 100B-token SlimPajama splits with no shift, these schedules reached loss similar to repeated cosine, with "negligible forgetting across dataset boundaries" (§7.4, Fig. 10). They were not tested with a distribution shift (§8). Open question for shifted data.
- *Pre-decay checkpoints in practice.* DeepSeekMath started from the coder checkpoint before its learning-rate decay ([[grpo-recipe]]), and Kimi K2.5's joint vision-text stage "continues from a near-end Kimi K2 checkpoint" ([[kimi-k2-5]] §4.3). Hägele et al. argue that pre-cooldown checkpoints of constant-plus-cooldown schedules can continue at high learning rate without re-warming, but they report no continual-training experiment with new data ([[cooldown-scaling-beyond-fixed-durations]] §2, §3.1).

**Conditions and limits.** The controlled evidence covers 405M and 10B (9.6B including embeddings, §5.3) dense models, two transitions, one seed, and optimizer-state resets. For the 10B model the paper conflicts with itself: §6.4 says both sizes re-warm to 3·10^-4, while Table 14 lists η_max 1.2·10^-4 and η_min 1.2·10^-5 for the 10B cosine schedule. Learning-rate values should not be transferred across sizes from this source.

**Implication.** For a general model, the learning-rate peak sets the adaptation-forgetting trade-off before any data choice is made. Record which checkpoint (decayed or not) the run started from, because it changes both the schedule needed and the reference for Δforget.

## §3 Replay of general data

**Definition.** *Replay* mixes data from the earlier distribution D0, or a general corpus standing in for it, into the batches of continued pretraining. *Compute-equivalent replay* keeps total tokens fixed: replay tokens replace D1 tokens rather than being added ([[continual-pretraining-rewarm-replay]] §4.2).

**Problem.** Re-warming increases forgetting (§2). The question is how much old data is needed to remove that forgetting and how much adaptation it costs.

**Formula.**

```
tokens_D1_new = (1 − x) · B
tokens_replay = x · B
```

- `x`: replay fraction, the share of each batch drawn from D0; `B`: total token budget of the continued run.

**Worked example.** The strong-shift run has B = 200B German tokens (§6.2). Use the interactive figure [figures/replay-tradeoff.html](figures/replay-tradeoff.html) to switch between the weak and strong shift and see the token split and the loss table for each replay fraction.
1. With x = 0.25: 150B German tokens and 50B Pile tokens.
2. Pile loss: 2.17 for the 300B Pile-only model, 3.56 after German without replay, 2.33 with 25% replay (Tables 2–3). Δforget = +1.39 without replay and +0.16 with replay. In perplexity, exp(2.17) = 8.76, exp(3.56) = 35.2, exp(2.33) = 10.3.
3. German loss: 1.11 without replay, 1.16 with replay (Δ = +0.05; perplexity 3.03 → 3.19).
4. Average of the two losses: 2.34 without replay, 1.75 with replay, and 1.75 for the 500B model trained from scratch on Pile ∪ German.

**Evidence (405M, Table 2).**

| Shift | Replay | Pile loss | D1 loss | Average |
|---|---|---|---|---|
| Pile → SlimPajama (300B) | 0% / 0.5% / 1% / 5% / 10% / 50% | 2.44 / 2.27 / 2.26 / 2.23 / 2.21 / 2.16 | 2.50 / 2.50 / 2.50 / 2.51 / 2.51 / 2.54 | 2.47 / 2.39 / 2.38 / 2.37 / 2.36 / 2.35 |
| 600B Pile ∪ SlimPajama from scratch | — | 2.17 | 2.53 | 2.35 |
| Pile → German (200B) | 0% / 1% / 5% / 10% / 25% / 50% | 3.56 / 2.83 / 2.57 / 2.46 / 2.33 / 2.24 | 1.11 / 1.12 / 1.12 / 1.13 / 1.16 / 1.22 | 2.34 / 1.97 / 1.85 / 1.80 / 1.75 / 1.73 |
| 500B Pile ∪ German from scratch | — | 2.26 | 1.25 | 1.75 |

- "even the lowest tested replay of 1% significantly reduces forgetting on Pile compared to the no-replay baselines" (§6.2). The selected settings were 5% (weak) and 25% (strong), because they see more new tokens than higher fractions at similar average loss (§6.3). Result (single study).
- Downstream (Table 3): English evaluation average 35.14 for Pile → SlimPajama with 5% replay vs 34.30 for the union model; 32.48 for Pile → German with 25% replay vs 32.43 for the union model and 29.20 without replay.
- *Scale.* Without replay, Pile loss rose by 0.23 at 10B and 0.27 at 405M; 5% replay reduced the rise by 0.19 and 0.21. At 10B the 5%-replay model's average loss was 1.89 vs 1.87 for the union model, and its evaluation average was 47.68 vs 48.00, but its MMLU was 28.79 vs 37.78 (Tables 4–5, §6.4.2). The authors suspect limited training data; the gap is not explained. Open question.

**Replay and general-data shares in other runs.**
- DeepSeekMath-Base 7B (500B tokens): 56% math corpus, 4% AlgebraicStack, 10% arXiv, 20% GitHub code, 10% Common Crawl natural language ([[grpo-recipe]], §2.3). HumanEval was 40.2% at the initialization checkpoint and 40.9% after, which the authors describe as maintained ([[grpo]], Table 4).
- EntiGraph CPT on Llama 3 8B: "replay with a rate of 0.1 using 1B RedPajama tokens" ([[synthetic-continued-pretraining]] App. C).
- Ibrahim et al. §2 cite Glorioso et al. (60% replay in a 50B-token decay phase) and DeepSeek-AI (30% replay over 6T tokens for DeepSeek-V2 continued pretraining). These values are the authors' descriptions and are not re-verified in this course.
- Kimi K2.5 adds vision tokens to a text model. In its fixed-budget ablation, text knowledge scored 45.5 with vision introduced at 0% of training at a 10:90 vision:text ratio, 43.9 at 50% with 20:80, and 43.1 at 80% with 50:50; mid- and late-fusion runs showed a "dip-and-recover" in text performance when vision data arrived ([[kimi-k2-5]] Table 1, App. B.1). Model size and budget of the ablation are not reported.

**Conditions and limits.** Replay was read in the original D0 order; replay-sample selection was not studied (§4.2). When D0 is not available, as for open-weight models whose pretraining data are not released, a general corpus replaces it; Ibrahim et al. list building a replay buffer for undisclosed data as future work (§9). Updating per domain in sequence gave poor results, and the authors advise continuing on a mixture of domains (§5.2, App. A.1).

**Implication.** In these runs, replay is the control that reduced forgetting on D0 at a D1-loss cost of 0.01–0.05 for the selected fractions (Table 2). For a general model, the replay source should cover the capabilities the release promises (languages, code, math, knowledge), not only a random slice of web text.

## §4 Domain-adaptive continued pretraining in specialist reports

**Problem.** Code and math specialist reports are the domain-adaptive runs in this course's library that disclose data mixtures. Five of the seven runs tabulated below do not print general scores for their initialization checkpoint, so a retention delta cannot be computed for them.

**Qwen2.5-Coder.** Initialized from Qwen2.5; file-level pretraining on 5.2T tokens at 8,192 tokens, then repo-level pretraining on about 300B tokens at 32,768 tokens with RoPE base 10,000 → 1,000,000 as printed ([[qwen-2-5-coder]] §3.2). The text data come from the Qwen2.5 pretraining corpus "to preserve Qwen2.5-Coder's general capabilities". The mixture ablation on Qwen2.5-Coder-7B (Table 3):

| Code:Text:Math | Coding: Common | MATH | GSM8K | MMLU | CEval | HellaSwag |
|---|---|---|---|---|---|---|
| 100:0:0 | 49.8 | 10.3 | 23.8 | 42.8 | 35.9 | 58.3 |
| 85:15:5 | 43.3 | 26.1 | 52.5 | 56.8 | 57.1 | 70.0 |
| 70:20:10 | 48.3 | 33.2 | 64.5 | 62.9 | 64.0 | 73.5 |

Table 3 lists "Common" and "BCB" under Coding; BCB (40.3, 36.2, 38.3) is omitted above. The 70:20:10 row loses 1.5 points on the code column relative to 100% code and gains 20.1 MMLU points. The text describes the middle row as 85:10:5 while the table prints 85:15:5, and the printed average for 100:0:0 (31.3) does not equal the mean of its seven scores (37.3). The token count of the ablation runs is not printed. Result (single study).

**DeepSeekMath.** Initialized from a pre-decay code checkpoint; 500B tokens with 20% code and 10% natural language ([[grpo-recipe]]). General scores rose: MMLU 42.9% → 54.9%, BBH 42.9% → 59.5% against the initialization ([[grpo]], Table 4). In the 1.3B ablations, two-stage training of code then math lowered HumanEval from 25.0% to 12.2% and MBPP from 40.0% to 17.0%, while one-stage mixed training kept 29.3% and 39.4% (Tables 6–7). The authors conjecture that 1.3B lacks capacity for both (Interpretation).

**Qwen2.5-Math.** Qwen Math Corpus v2 has "over 1T" tokens at 4K context; the models are initialized from Qwen2.5 base models ([[qwen-2-5-math]] §2). The report does not give the mixture, the general-data share, the learning rate, or non-math general benchmarks. For the predecessor, Qwen2-Math-7B scored MMLU-STEM 65.7 against 67.6 for Qwen2-7B, but Qwen2-Math was initialized from an intermediate Qwen2 checkpoint, so this is not a clean delta (Table 2).

**Retention reporting across runs** (cells state what the source prints).

| Run | Init checkpoint | CPT tokens | General-data share | General benchmarks vs init |
|---|---|---|---|---|
| Ibrahim et al. 405M/10B, Pile → SlimPajama | Pile checkpoint (decayed) | 300B | 5% replay | reported (Tables 3–5) |
| DeepSeekMath-Base 7B | Coder-v1.5 pre-decay | 500B | 20% code, 10% NL | reported (Table 4) |
| Qwen2.5-Coder-7B | Qwen2.5 (§1, Fig. 2) | 5.2T + ~300B | 20% text (plus 10% math) | not reported (compared with other coders only) |
| Qwen2.5-Math-7B | Qwen2.5 base (§2) | >1T | not reported | not reported |
| Llama 3 code expert | a branch of the main pre-training run; checkpoint not specified ([[llama-3]] §4.3.1) | 1T, >85% code | not reported | not reported |
| Instruction PT Llama3-8B (biomed, finance) | Llama3-8B | 4K steps × 0.25M tokens (1B, derived) | general instructions at "the same mixing ratio as Cheng et al. (2023)"; ratio not printed in this paper | not reported |
| EntiGraph CPT Llama 3 8B | Llama 3 8B Base | 455M × 2 epochs | 10% RedPajama replay | not reported |

**Implication.** Mixture ablations such as Qwen2.5-Coder's Table 3 show that general data in a specialist run protects general scores and can raise target-adjacent scores. Without a before/after row for the initialization checkpoint, a reader cannot tell whether a specialist kept its base model's general ability. A general-purpose program that merges or distills from specialists (ch-30c, ch-35) needs that row.

## §5 Making new knowledge extractable from small corpora

**Problem.** Domain-adaptive runs in §4 use hundreds of billions of tokens. Yang et al. name proprietary document stores as adaptation targets whose size can be in the millions of tokens (§1). EntiGraph's Table 1 contrasts published CPT corpora of 14.9B to 620B unique tokens with its 1.3M-token setting ([[synthetic-continued-pretraining]] Table 1). Training directly on such a corpus can fail: Raw CPT on 1.3M QuALITY tokens (4 epochs, 10% replay) scored 38.15% closed-book QA, below the base model's 39.49% (§4.2, App. C). The authors postulate that the narrow corpus "may harm the overall English capabilities of the model" and that the "limited diversity of knowledge representations" in the raw corpus limits knowledge acquisition (Interpretation; general capability was not measured).

### 5.1 Instruction pre-training

**Definition.** Instruction pre-training augments each raw document with instruction-response pairs generated from that document and trains with next-token loss on all tokens ([[instruction-pretraining]] §2).

**Mechanism.**
1. Fine-tune a 7B synthesizer (from Mistral-7B-v0.1) on context-based task datasets to map a raw text to instruction-response pairs.
2. Run it on the domain corpus in rounds, prepending earlier texts and pairs so that each training sequence is a few-shot example; about 5 pairs of about 52 tokens each per text (§3.1).
3. Mix the augmented corpus with general instruction data and continue pretraining.

**Evidence.** Llama3-8B continued on PubMed Abstracts reached a biomedicine average of 61.3 vs 58.4 for vanilla continued pretraining on the same token count and 53.6 before training; on financial news, 74.7 vs 72.0 and 70.1 (Table 3). Llama3-70B scored 63.9 and 71.9. The domain run used 4K steps at 0.25M tokens per batch, LR 1e-5 (Table 14). General-benchmark scores of the adapted models are not reported. GPT-4 judged the synthesized responses 69.8% accurate for finance and 86.2% for biomedicine (Table 6), and the Limitations section warns that inaccurate responses "may potentially mislead the pre-trained model". Result (single study).

### 5.2 Synthetic continued pretraining with EntiGraph

**Definition.** Synthetic CPT applies a generator A_synth: D_source → D_synth and continues pretraining on D_synth instead of D_source ([[synthetic-continued-pretraining]] Eq. 1). EntiGraph is a two-prompt generator: extract salient entities from a document, then ask an LM to analyze how a subset of entities relate in that document (§2.2).

**Worked example (derived).** A document yields n = 20 entities. EntiGraph generates text for all pairs and triplets: C(20,2) = 190 and C(20,3) = 1,140, so 1,330 relation-analysis prompts for one document. The combinatorial count, not repeated paraphrasing, is the source of diversity. Over the full corpus, 1.3M source tokens became 455M synthetic tokens, a 350× expansion. Two epochs on 455M tokens at the reported 6,090 tokens/s on one 8×H100 node take 910M/6,090 s ≈ 41.5 h (App. C reports about 41 hours).

**Evidence.**
- Closed-book QuALITY accuracy for Llama 3 8B: 39.49% base, 38.15% Raw CPT, 56.22% EntiGraph CPT (455M tokens, 2 epochs, 10% replay, peak LR 5e-6), above GPT-4's 51.30% (Fig. 2). Accuracy scaled log-linearly with synthetic tokens up to 455M; the Rephrase baseline, adapted from Maini et al. ([[rephrasing-the-web]]) and stopped at 38M tokens, scaled worse.
- With retrieval: base + RAG 60.35%, EntiGraph CPT + RAG 62.60% (Table 3). EntiGraph alone added 16.73 points, 80.2% of the 20.86 points added by RAG (16.73/20.86).
- After instruction tuning on 250M UltraChat tokens, closed-book summaries of the books contained more salient claims with fewer added false claims than the Raw-CPT model (Fig. 3).

**Conditions and limits.** The generator was gpt-4-turbo; the authors fact-checked a subset for a few books; they note possible hallucinated relations on harder content and possible distillation of the generator's knowledge (§7.1). General-capability retention of the 8B model was not measured. Result (single study, one corpus).

**Implication.** In the one corpus tested, knowledge from a 1.3M-token corpus became answerable closed-book only after it was re-expressed as 455M tokens of distinct entity-relation texts (Interpretation of Fig. 2). Both methods here convert the corpus into diverse task-like or relational text and keep general data in the mix; neither reports whether general ability was preserved, so a retention panel (§1) is required when these methods are used.

## §6 Model merging during pretraining

**Definition.** Pre-trained Model Averaging (PMA) averages N checkpoints from one pretraining run ([[merging-in-pretraining]] §3):

```
M_avg = Σ_{i=1..N} w_i · M_i          V = T_{i+1} − T_i
SMA: w_i = 1/N     WMA: w_i ∝ i     EMA: M_avg^(i) = α·M_i + (1 − α)·M_avg^(i−1)
```

- `M_i`: parameters of checkpoint i; `w_i`: its weight; `T_i`: cumulative tokens at checkpoint i; `V`: fixed token interval; `α`: EMA smoothing factor.

**Problem.** A constant-learning-rate checkpoint has higher loss than an annealed one, but annealing ends the run's ability to continue at high learning rate (§2). Merging is tested as a way to get annealed-like quality from stable-phase checkpoints.

**Mechanism** (§4.6). With a quadratic loss around θ*, L(θ) ≈ L(θ*) + ½ δᵀHδ with δ = θ − θ*. The average of k checkpoints has loss below the mean loss of those checkpoints when Σ_i Σ_{j≠i} δ_iᵀHδ_j < (k − 1) Σ_i δ_iᵀHδ_i (Eq. 15), which is easier to satisfy when the deviations point in different directions.

**Worked example (derived).** One dimension, H = 2, θ* = 0.
1. Checkpoints θ1 = 0.3 and θ2 = −0.1: losses 0.09 and 0.01, mean 0.05. The average θ = 0.1 has loss 0.01. Cross term 2 · (0.3 · 2 · −0.1) = −0.12 < (2 − 1)(0.18 + 0.02) = 0.20.
2. Checkpoints θ1 = 0.3 and θ2 = 0.1 (same side): mean loss 0.05; the average θ = 0.2 has loss 0.04, above the better checkpoint's 0.01.
For a positive-definite H the inequality holds for any unequal deviations (Cauchy–Schwarz), so the merge always beats the *mean* checkpoint in this model. Whether it beats the *latest* checkpoint depends on the directions, which is the comparison that matters in practice.

**Evidence.**
- Stable phase: merging raised HumanEval from 31.1 to 36.6 for Seed-MoE-1.3B/13B and from 54.3 to 61.6 for Seed-MoE-10B/100B (Fig. 1). Result (single study, internal models and data).
- *Substitute for annealing.* Two forks of Seed-MoE-1.3B/13B at 1.4T tokens trained 250B more tokens, one at constant LR and one annealed; the merged constant-LR models "significantly outperformed" both early and were "comparable to the annealed models" later (Fig. 3; values plotted only).
- Settings: WMA was best at 204B tokens, differences between methods diminished later, and SMA was used in later experiments "for its simplicity and stability"; intervals of about 4B, 8B, and 80B tokens for 0.7B/7B, 1.3B/13B, and 10B/100B models; N = 3 was "nearly 1 point lower" than N = 15 at the end of training; N = 10 chosen (Figs. 4–5).
- *Contrasting result.* In Hägele et al., stochastic weight averaging on a constant-LR run lowered loss relative to the raw checkpoint but did not reach the loss of an explicit cooldown, at 33M–360M parameters measured by perplexity ([[cooldown-scaling-beyond-fixed-durations]] §4.1). The two studies differ in scale, architecture, and metric. Open question.
- *Merged initialization for continued training.* PMA-init of Seed-MoE-0.7B/7B after ~1T stable tokens gave "marginally lower loss at the initial training phase" and converged to comparable loss in continued training (values plotted only), with overall parity at the end (§4.4, Fig. 6).

**Conditions and limits.** No forgetting measurement on a shifted D1 is reported for PMA, and model architectures and data are unreleased. Merging checkpoints annealed to a very low learning rate gives smaller gains, per the authors (§4.6).

**Implication.** For continued pretraining, a merge of stable-phase checkpoints is a candidate starting point that has not been decayed, which fits the pre-decay-checkpoint practice in §2. Its effect on forgetting under a distribution shift is untested. Merging specialists after continued pretraining is covered in ch-30c.

## §7 Decision rule and evaluation gate

The rule below is this course's synthesis of §1–§6 (Interpretation). Each step names the evidence it rests on; no threshold in it is taken from a source unless cited.

1. **Size the new data.** When D1 has 100B or more tokens and differs from D0 in language, domain, or modality, plan a continued-pretraining run with replay (§3). When D1 has millions of tokens, plan synthetic expansion first (§5) and keep 10% general replay as EntiGraph did.
2. **Choose the starting checkpoint.** Prefer a pre-decay checkpoint (DeepSeekMath), a near-end checkpoint (Kimi K2.5), or a stable-phase checkpoint when one exists; otherwise re-warm to a peak near the original η_max and cosine re-decay (§2). Record the checkpoint used; it is θ_before for every delta.
3. **Choose the replay source and fraction.** Replay the capabilities the release promises. Run short sweeps over fractions such as 1%, 5%, 10%, 25% on a limited token budget, because Ibrahim et al. report that "relative differences between them appear very early during training" (§2 rules of thumb). Mix domains in one stage rather than updating per domain (§3).
4. **Gate before release or hand-off to SFT.** Compare θ_after with θ_before and, when affordable, with a no-replay run:

| Gate item | Measurement | Pass condition (course synthesis) | Source of the method |
|---|---|---|---|
| Adaptation | D1 held-out loss; target benchmarks | lower than θ_before and not worse than the no-replay run by more than the run-to-run noise measured for this harness | [[continual-pretraining-rewarm-replay]] Table 2 |
| Per-domain forgetting | Δforget(d) for each main D0 source, macro-averaged, decontaminated | every domain's increase is reported; increases above the seed-to-seed spread of the harness are listed as trade-offs | [[paloma]] §3–§4 |
| General downstream | MMLU-style knowledge, reasoning, code, multilingual scores | Δtask within measured evaluation noise, or a documented trade-off | [[grpo]] Table 4 |
| Reference correctness | scores of the exact initialization checkpoint | reported in the same table | [[qwen-2-5-math]], [[qwen-2-5-coder]] |
| Modality or language shift | text-only curve during the run | no unrecovered dip | [[kimi-k2-5]] App. B.1 |

5. **Carry the report forward.** The same panel is re-run after SFT (ch-30a) and after RL, because forgetting from continued pretraining and from post-training add up in the released model.

## Negative samples and negative feedback

Continued pretraining uses next-token loss on every token, so it has no term that pushes down the likelihood of a sample (negative as gradient) and no likelihood-displacement risk from this stage. Negatives appear only in sense 1, *negative marginal value*: data that lowers performance when trained on as positive targets. Two cases in this chapter are measured. Raw CPT on a 1.3M-token corpus scored below its base model (38.15% vs 39.49%, [[synthetic-continued-pretraining]] Fig. 2), and 100% code data gave the lowest MMLU and math scores in Qwen2.5-Coder's mixture ablation ([[qwen-2-5-coder]] Table 3). A third case is a risk that was not measured: synthesized instruction-response pairs judged 69.8% accurate for finance were trained as positive targets without filtering ([[instruction-pretraining]] Table 6). The control is filtering or verification before training, as in Qwen2.5-Coder's synthetic code, where "only executable code was retained" ([[qwen-2-5-coder]] §3.1.1). Negative gradients are covered in ch-31a and ch-43a.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Ibrahim et al. Pile → SlimPajama | 405M | mid-train | D0 / D1 tokens; batch; seq | 300B Pile / 300B SlimPajama; 1104 sequences; 2048 | arXiv:2403.08763v4 §5.1–§5.3 | verified 2026-09-15 | no ablation reported |
| same | 405M | mid-train | LR schedule | re-warm to 3e-4 over 1% of iterations, cosine to 3e-5; optimizer states reset | Table 14; §5.2; §6.3 | verified 2026-09-15 | §6.1.2 Fig. 4: re-warm vs constant and η_max ∈ {1.5, 3, 6}e-4 |
| same | 405M | mid-train | replay fraction | 5% | §6.3 | verified 2026-09-15 | Table 2: 0.5–50% sweep, single seed |
| Ibrahim et al. Pile → German | 405M | mid-train | D1 tokens; replay | 200B; 25% | §6.2–§6.3 | verified 2026-09-15 | Table 2: 1–50% sweep, single seed |
| Ibrahim et al. Pile → SlimPajama | 10B | mid-train | peak → final LR (Table 14) | 1.2e-4 → 1.2e-5, warmup 1% | Table 14 | conflict (Table 14 vs §6.4 text; no config released in the paper) | no ablation reported |
| Ibrahim et al. Pile → SlimPajama | 10B | mid-train | peak LR (§6.4 text) | "re-warm to the ηmax of pre-training (3·10−4)" for both sizes | §6.4 | conflict (see row above) | no ablation reported |
| Ibrahim et al. Pile → SlimPajama | 10B | mid-train | replay fraction | 5% | §6.4 | verified 2026-09-15 | selected from the 405M weak-shift sweep (§6.4) |
| DeepSeekMath-Base 7B | 7B | mid-train | init; tokens; mixture | Coder-v1.5 pre-decay; 500B; 56% math corpus, 4% AlgebraicStack, 10% arXiv, 20% code, 10% NL | arXiv:2402.03300v3 §2.3 ([[grpo-recipe]]) | verified 2026-09-14 | no ablation of this split reported; §5.1 Tables 6–7 at 1.3B |
| DeepSeekMath-Base 7B | 7B | mid-train | peak LR; batch | 4.2e-4; 10M tokens | §2.3 ([[grpo-recipe]]) | verified 2026-09-14 | no ablation reported |
| Qwen2.5-Coder (0.5B–32B) | all | mid-train | init; tokens; seq; mixture | Qwen2.5; 5.2T file-level at 8,192; 70% code, 20% text, 10% math | arXiv:2409.12186v3 §3.1.2, §3.2.1 | verified 2026-09-15 | Table 3: 100:0:0 vs 85:15:5 vs 70:20:10 on 7B |
| Qwen2.5-Coder (0.5B–32B) | all | long-context | tokens; seq; RoPE base | ~300B; 8,192 → 32,768; 10,000 → 1,000,000 (as printed) | §3.2.2 | verified 2026-09-15 | no ablation reported |
| Qwen2.5-Coder | all | mid-train | LR, batch, schedule | not reported (checked §3, §4, Table 1) | — | not reported | — |
| Qwen2.5-Math-7B | 7B | mid-train | init; tokens; seq | Qwen2.5 base; >1T (Qwen Math Corpus v2); 4K | arXiv:2409.12122v1 §2 | verified 2026-09-15 | Table 2: v2 vs v1 comparison across models |
| Qwen2.5-Math-7B | 7B | mid-train | mixture; general share; LR | not reported (checked §2, §5.1) | — | not reported | — |
| Instruct PT Llama3-8B (biomed / finance) | 8B | mid-train | steps; batch; seq; LR | 4K; 0.25M tokens; 4096; 1e-5 cosine, 1000 warmup | arXiv:2406.14491v2 Table 14 | verified 2026-09-15 | Table 4 data ablations; LR not ablated |
| same | 8B | mid-train | total tokens | 1B | 4K × 0.25M | derived | — |
| EntiGraph CPT Llama 3 8B | 8B | mid-train | data; epochs; replay | 455M EntiGraph tokens; 2; replay rate 0.1 (per-batch probability of loading a RedPajama batch; 1B-token pool) | arXiv:2409.07431v2 §4.1, App. C | verified 2026-09-15 | Fig. 2 token-count scaling; replay not ablated for this run |
| same | 8B | mid-train | seq; batch; LR | 2048; 16; peak 5e-6, 5% warmup, cosine | App. C | verified 2026-09-15 | no ablation reported |
| Seed-MoE-1.3B/13B (PMA) | 1.3B/13B act./total | merge | method; N; V | SMA; 10; ~8B tokens | arXiv:2505.12082v3 §4.2–§4.3 | verified 2026-09-15 | Figs. 4–5: WMA/SMA/EMA, V ∈ {4, 8, 16, 32}B, N ∈ {3, 6, 10, 15} |
| Kimi K2.5 | 1.04T total / 32B act. | mid-train | init; tokens; seq | near-end K2 checkpoint; 15T vision-text; 4K | arXiv:2602.02276v2 §4.3, Table 3 | verified 2026-09-15 | Table 1: vision timing and ratio ablation (size not reported) |
| Kimi K2.5 | same | mid-train | vision:text ratio; LR | not reported ("constant ratio", §2) | — | not reported | — |

**Starting point for a small general-purpose run.** For a dense model of about 405M parameters continued on 200–300B tokens with optimizer states reset, the verified rows give: re-warm over 1% of iterations to the original peak (3e-4 in that study), cosine to 0.1 × peak, and compute-equivalent replay of 5% for a same-language data refresh or 25% for a new language, chosen after a short sweep ([[continual-pretraining-rewarm-replay]] §6.3, Tables 13–14). For an 8B model and a corpus of about 1M tokens, the verified EntiGraph setting is a peak LR of 5e-6 with 5% warmup, 2 epochs over the synthetic corpus, and 10% general replay ([[synthetic-continued-pretraining]] App. C). Neither setting was validated on general benchmarks at the other scale.

## Generalization lens

**(a) What increases breadth.**
- Keeping general data in the mix: the 70:20:10 Qwen2.5-Coder-7B mixture scored MMLU 62.9 vs 42.8 for 100% code and GSM8K 64.5 vs 23.8 ([[qwen-2-5-coder]] Table 3); DeepSeekMath's 500B-token run with 20% code and 10% NL raised MMLU by 12.0 and BBH by 16.6 against its initialization ([[grpo]] Table 4).
- Replay after re-warming: 25% replay matched the average loss of retraining from scratch on both datasets in the language-shift run ([[continual-pretraining-rewarm-replay]] Table 2). The Qwen2.5-Coder mixture ablation points in the same direction in a different setting (general data protects general scores), so the direction is Replicated; the size of the effect is setting-specific.
- Diverse knowledge representations for small corpora: EntiGraph's combinatorial entity-relation text scaled log-linearly where paraphrases scaled worse ([[synthetic-continued-pretraining]] Fig. 2).

**(b) What causes narrowing or forgetting.**
- Continuing without replay after re-warming: Pile loss +1.39 after 200B German tokens at 405M ([[continual-pretraining-rewarm-replay]] Table 2).
- The learning-rate increase itself: +0.1 Pile loss peak when continuing on the Pile at η_max 3e-4 (§7.1).
- Sequential single-domain stages: code → math at 1.3B lowered HumanEval from 25.0% to 12.2% ([[grpo]] Tables 6–7).
- Late, high-ratio introduction of a new modality: text-performance "dip-and-recover" and lower final text knowledge in K2.5's ablation ([[kimi-k2-5]] Table 1, App. B.1).
- Training on a small raw corpus: closed-book QA below the base model ([[synthetic-continued-pretraining]] Fig. 2).

**(c) How to measure it at this stage.**
- Macro-averaged per-domain perplexity with decontamination, fixed vocabulary or bits per byte, and documents scored one at a time ([[paloma]] §3 G1–G5).
- Downstream deltas against the exact initialization checkpoint; the choice of reference changed DeepSeekMath's MMLU gain from +5.8 to +12.0 ([[grpo]] Table 4).
- Known errors: single seeds and 1.19-point spread at 405M ([[continual-pretraining-rewarm-replay]] §6.3.2); intermediate-checkpoint initialization without a reference row ([[qwen-2-5-math]] Table 2); specialist reports that compare only with other specialists ([[qwen-2-5-coder]] Tables 13–14).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Continuing at the base model's final low learning rate | D1 loss falls less than in a re-warmed run on the same tokens; target benchmarks change less | Compare against a short re-warmed run on the same tokens (§2, Fig. 4 pattern) |
| Re-warming without replay on a shifted D1 | Old-domain loss rises early and stays high | Plot Δforget(d) per domain during training; add a 1–5% replay run |
| Reporting deltas against the released model instead of the initialization checkpoint | Gains or losses that change size when the reference changes | Evaluate θ_before itself; DeepSeekMath's +5.8 vs +12.0 MMLU is the example |
| Using one token-weighted validation loss | Aggregate loss flat while a small domain degrades | Macro-average per-domain perplexity (§1 worked example) |
| Counting replay as extra tokens when comparing runs | The replay run uses more compute and looks better | Use compute-equivalent replay: (1 − x)·B new tokens |
| Updating one domain after another | Early domains degrade at each transition | Mix domains in one stage (§3; [[grpo]] Tables 6–7) |
| Training directly on a small raw corpus | Closed-book QA flat or below the base model | Compare raw CPT with a synthetic expansion at equal steps (§5.2) |
| Trusting unverified synthetic targets | Hallucinated facts in closed-book answers | Sample and grade generated pairs; filter before training (§5.1 Table 6) |
| Treating a merged constant-LR checkpoint as equivalent to an annealed one without checking | Loss or scores below an annealed branch | Branch a short cooldown and compare on the same suite (§6 contrasting result) |

## Check your understanding

1. In the language-shift run, 25% replay raised German loss by 0.05 and lowered Pile loss by 1.23. Explain why the average loss of this run can equal that of a model trained from scratch on 500B tokens, even though it saw 50B fewer German tokens than the no-replay run.
2. Re-warming on the Pile itself raised Pile loss. What does this result imply about the cause of the early forgetting seen when continuing on SlimPajama, and how would you design a run to separate the two causes?
3. DeepSeekMath's MMLU gain is +5.8 or +12.0 depending on the reference checkpoint. Explain which part of the difference comes from the base model's own learning-rate decay, and why this matters when deciding whether math data improves general ability.
4. Qwen2.5-Coder's 70:20:10 mixture scored lower on the code column than 100% code but higher on MMLU and GSM8K. Give two mechanisms that could explain the math and knowledge gains, and state what additional table you would need to test them.
5. Why did continued pretraining on 1.3M raw tokens fail to raise closed-book QA, while 455M EntiGraph tokens derived from the same documents succeeded? Connect your answer to how many distinct forms each fact appears in.
6. In the one-dimensional merging example, the merge beats the mean checkpoint in both cases but not the best checkpoint in the second. Explain what property of real training trajectories determines whether PMA beats the latest checkpoint.
7. A report shows a new-language continued-pretraining run with a stable aggregate validation loss and unchanged MMLU. Describe two forms of forgetting this report could still hide, and the measurement that would expose each.

## Connections

- **Depends on:** ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL (annealing, stage placement, and decay-phase data).
- **Previous in outline:** ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL.
- **Next:** ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression (the same adaptation-versus-regression trade-off for sequence length).
- **Related:** ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization (cosine and WSD schedules); ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale; ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing; ch-29a — Long-Document Synthesis for Continued Pretraining and Long-Context SFT; ch-30a — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control; ch-30c — Weight Averaging and Model Merging for Generalist Models; ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing.

## Sources

- [[continual-pretraining-rewarm-replay]] — re-warming and re-decaying results, replay sweep (Table 2), downstream and 10B results (Tables 3–5), infinite schedules, limitations.
- [[paloma]] — perplexity and macro-average definitions, guidelines G1–G5, per-domain degradation of web-only baselines, perplexity–downstream correlations.
- [[grpo]] — DeepSeekMath-Base general and code retention against the pre-decay initialization (Table 4) and the 1.3B code-then-math forgetting ablation (Tables 6–7).
- [[grpo-recipe]] — DeepSeekMath-Base 7B initialization, 500B-token mixture, learning rate, and batch.
- [[qwen-2-5-coder]] — initialization from Qwen2.5, stage token counts, 70:20:10 mixture ablation, general-benchmark reporting without the initialization row.
- [[qwen-2-5-math]] — math corpus sizes, initialization from Qwen2.5 and intermediate Qwen2 checkpoints, base-model table without Qwen2.5 rows.
- [[instruction-pretraining]] — synthesizer method, domain continued-pretraining results, synthesized-pair accuracy, hyperparameters.
- [[synthetic-continued-pretraining]] — EntiGraph method, raw vs rephrase vs EntiGraph scaling, RAG comparison, replay and training settings.
- [[merging-in-pretraining]] — PMA definitions, stable-phase and annealing-substitute results, interval and count ablations, quadratic analysis, PMA-init.
- [[cooldown-scaling-beyond-fixed-durations]] — SWA does not reach cooldown loss; argument for continuing from pre-cooldown checkpoints.
- [[kimi-k2-5]] — continuation from a near-end K2 checkpoint, vision injection timing and ratio ablation, text dip-and-recover.
- [[llama-3]] — Llama 3 code expert continued pretraining on 1T tokens with more than 85% code.
- [[rephrasing-the-web]] — the rephrasing approach adapted as EntiGraph's Rephrase baseline.
