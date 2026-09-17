<!-- chapter: ch-32e
     track: midtraining
     kind: content
     title: Mid-Training, Annealing, and Context-Extension Recipes Side by Side
     deps: [ch-32c, ch-32d, ch-14a]
     sources: [[llama-3]], [[llama-3-recipe]], [[olmo-2]], [[olmo-2-midtraining]], [[olmo-3]], [[olmo-3-base-stages]], [[olmo-core-olmo3-configs]], [[smollm-3]], [[smollm3-midtraining]], [[smollm3-training-configs]], [[qwen-2.5]], [[qwen-2.5-recipe]], [[qwen-long-context-synth]], [[prolong]], [[prolong-recipe]], [[deepseek-v3]], [[deepseek-v3-recipe]], [[deepseek-v3.1-recipe]], [[nemotron-nano-2]], [[glm-4-5]], [[glm-4-5-recipe]], [[glm-5]], [[minicpm]], [[cooldown-scaling-beyond-fixed-durations]], [[llama-2-long]], [[yarn]]
     figures: figures/stage-budget-ledger.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32e — Mid-Training, Annealing, and Context-Extension Recipes Side by Side

> **Core insight.** Across the open reports compared here, explicit mid-training uses 1.3% to 12% of pretraining tokens and context extension uses 0.09% to 5.1% (ratios derived in §3 and §4). The reports agree on three practices: the learning rate falls to a low or zero value while the data mix changes, short data is kept in long-context stages, and several reports train at a longer length than the length they evaluate. Few values have a printed ablation or before-and-after comparison: OLMo 2's mid-training raised the 7B average from 53.0 to 62.9 (OLMo 2 Table 9), a 34% long / 66% short extension mix cost 0.8 points of base-model score against 2.5 points for 66% / 34% (Olmo 3 §3.6.3), and training at 512K instead of 128K raised RULER-128k from 73.68 to 81.04 on Nemotron-H 8B (Nemotron Nano 2 Table 4). Stage counts and learning rates are single-report choices without printed ablations; RoPE bases were ablated only by ProLong (Table 18), and merge weights were compared only by SmolLM3 and Nemotron Nano 2 without printed tables. Three values (one RoPE base, two learning rates) differ between the paper or blog and the released configuration.
>
> **Guideline.** When copying a stage budget from a report, convert it to seen tokens with an explicit K (1,000 or 1,024) and batch unit, and divide by the same report's pretraining tokens, because DeepSeek-V3's extension is 125.8B or 122.9B tokens depending on that choice and OLMo 2's totals count all souping ingredients. When extending a small RoPE model from 4K–8K to 32K–65K, keep short or mid-training data in the extension mix (Olmo 3 used 66% of tokens, ProLong 37%) and gate each stage on a short-context suite and a held-out long-context suite, because short-task scores fell as the long share rose in both reports and Olmo 3 7B still lost 5.4 Math points in its extension stage with 66% short data (Olmo 3 §3.6.3, Table 13; ProLong Fig. 3). When a later stage (reasoning mid-training, preference optimization) lowers long-context scores, test a linear merge with a small weight on an earlier checkpoint that has strong long-context scores before re-running extension, because SmolLM3 reports recovering its base RULER score to 128k with a 0.9 post-trained / 0.1 mid-training-checkpoint merge. Otherwise, when later stages do not lower long-context scores, a single extension run with no long-context merge is the configuration used for the Olmo 3 7B base model (§6).

## Why this chapter matters for a general-purpose model

A base model's pretraining run ends in a stage with a lower learning rate and a changed data mix. Reports call this stage annealing, the decay phase, cooldown, or mid-training, and many reports also place a context-extension stage near the end of pretraining. These stages sit between pretraining (ch-14a) and supervised fine-tuning (ch-30). They use a small share of tokens, but they decide which capabilities the SFT and RL stages start from: OLMo 2 7B's GSM8K score moved from 24.1 to 67.5 in 50B tokens of mid-training (OLMo 2 Table 9), and Olmo 3's thinking and instruction data in mid-training raised the base-model average before any post-training (Olmo 3 Table 10).

For a general-purpose model, the same stages also narrow. A math-and-code-heavy mid-training mix lowered Olmo 3's multiple-choice and generative QA scores (Table 7), a context-extension stage lowered Olmo 3 32B's Math cluster from 69.7 to 61.4 (Table 13), and SmolLM3's reasoning mid-training lowered its long-context RULER score until a merge restored it ([[smollm3-midtraining]]). Chapter ch-32 explains the mechanisms of mid-training, ch-32b the context-extension methods, ch-32c the long-context evaluations, and ch-32d agentic mid-training data. This chapter puts the published settings side by side, normalizes their units, marks which values were verified at a locus on 2026-09-14 or 2026-09-15, and builds a starting configuration only from verified rows.

The companion figure [figures/stage-budget-ledger.html](figures/stage-budget-ledger.html) lets the reader compare stage budgets as a share of pretraining tokens on a log scale, open the inputs and locus behind each ratio, and recompute token counts from steps, batch, and sequence length with K read as 1,000 or 1,024.

## §1 Units: what a stage budget counts

**Definitions.**
- **Seen tokens** are tokens processed by the optimizer, counting repetitions. **Unique tokens** are distinct tokens in the data before repetition.
- An **epoch** over a subset is one pass through it; a subset trained for 4 epochs contributes 4× its unique tokens as seen tokens.
- **Sequence length** is the training window. **Document length** is the length of one source document. Packing several documents into one window, or cutting one document across windows, makes the two differ.
- A **souping ingredient** is one of several runs whose weights are averaged into the released checkpoint (§6).

**Problem.** Reports print budgets in units that cannot be compared without conversion. Four cases occur in this chapter's sources:
1. Unique versus seen. GLM-5's issue–PR data is "approximately 160B unique tokens" inside a mid-training phase of 1.55T seen tokens ([[glm-5]] §2.3). SmolLM3's reasoning mid-training is 35B tokens for 4 epochs, about 140B seen ([[smollm3-midtraining]]). OLMo 2's 300B Dolmino mix repeats StackExchange, Wikipedia, and math data four times (Table 13 "Source %" 400; [[olmo-2-midtraining]]).
2. Binary prefixes. ProLong defines K = 2^10, M = 2^20, B = 2^30 (footnote 2), so its "20B tokens" are 21.47 × 10^9 decimal tokens ([[prolong-recipe]]).
3. Batch without a unit. DeepSeek-V3 prints batch sizes 1920 and 480 for its extension phases without a unit ([[deepseek-v3-recipe]], §4.3).
4. Totals that include parallel runs. OLMo 2 reports 4.05T tokens for 7B: 3.90T pretraining plus three 50B mid-training runs that were averaged, although each ingredient saw 3.95T ([[olmo-2-midtraining]], §2.3).

**Formula.**

```
D_stage = Σ_p  steps_p × B_p × L_p          f = D_stage / D_pretrain
```

- `p` indexes the phases of a stage; `steps_p` is the optimizer steps in phase p.
- `B_p` is the global batch in sequences; `L_p` is the sequence length in tokens.
- `D_pretrain` is the pretraining-stage token count printed by the same report; `f` is the stage's share.

**Worked example: DeepSeek-V3's extension budget.** DeepSeek-V3 prints two 1000-step phases: 32K with batch 1920, then 128K with batch 480 (§4.3). If batch counts sequences and K = 1,024:
1. Phase 1: 1000 × 1920 × 32,768 = 62.91B tokens.
2. Phase 2: 1000 × 480 × 131,072 = 62.91B tokens.
3. Total 125.8B; f = 125.8B / 14.8T = 0.85%.

With K = 1,000, each phase is 61.44B and the total is 122.9B. The Olmo 3 report lists "DeepSeek V3: 123B" for this stage ([[olmo-3-base-stages]], §3.6), which matches the decimal reading (Interpretation; Olmo 3 does not show its calculation). The DeepSeek-V3.1 model card describes its 630B-token 32K phase as "10-fold" the V3 phase ([[deepseek-v3.1-recipe]]), which implies about 63B for V3's phase 1 and agrees with either reading to within 3%. **Derived.** The report itself prints no token count.

**Sequence length versus document length.** Two reports state that the training window changes which long documents survive intact. Nemotron Nano 2 trained Phase LC at 524,288 tokens for a 128k target because "longer training sequence can effectively lower the chance of long coherent documents being cut" by concatenate-and-chunk loading ([[nemotron-nano-2]] §2.6). Olmo 3 replaced concatenate-then-split with best-fit packing during extension because the former "produces training instances that are, on average, shorter than the underlying document length distribution" ([[olmo-3-base-stages]] §3.6.4). A long:short ratio is therefore meaningful only with its unit: Qwen2.5-1M's "75% sequences at the current maximum length" is a share of sequences ([[qwen-long-context-synth]] §3), and Olmo 3's 34% is a share of tokens (Table 11: 16.95B of 50.0B).

**Implication.** A budget copied without its unit can differ from the source by 2.4% (K), by 4× (epochs), or by the number of souping ingredients. The Recipe table below records each value's unit as printed.

## §2 Decay and annealing ledger

**Definition.** The **decay phase** (also annealing or cooldown) is the final part of a run in which the learning rate decreases to a low or zero value. MiniCPM's Warmup-Stable-Decay (WSD) schedule separates it explicitly ([[minicpm]] §4.2, Eq. 1):

```
WSD(T; s) = (s / W) · η       for s < W
          = η                 for W < s < T
          = f(s − T) · η      for T < s < S,   0 < f(s − T) ≤ 1 decreasing
```

- `s` is the step; `W` the end of warmup; `T` the end of the stable stage; `S` the end of training.
- `η` is the maximum learning rate; `f` is the decay function (linear, cosine, or exponential).

**Problem.** A run trained at a high constant learning rate has a higher loss than the same run after decay. MiniCPM reports that on 0.036B models loss "experiences a significant rapid decline" once decay starts and reaches the cosine schedule's loss ([[minicpm]] §4.2, Fig. 5). Because the loss drop concentrates in the decay phase, several reports change the data mix at the same time.

**Mechanism as practiced.**
1. Stop or truncate the main schedule: OLMo 2 7B stops a 5T-token cosine at 4T tokens and "switch[es] to mid-training" ([[olmo-2-midtraining]] §4.1); Olmo 3 32B truncates a 5.93T cosine at 5.5T ([[olmo-3-base-stages]] Table 35).
2. Change the mix toward higher-quality or task-bearing data: Llama 3 upsamples "data sources of very high quality" ([[llama-3-recipe]] §3.4.3); MiniCPM mixes "high-quality SFT data" into pretraining data ([[minicpm]] §6.2); SmolLM3 raises code from 15% to 24% and math from 10% to 13% ([[smollm3-midtraining]]).
3. Decrease the learning rate: linearly to 0 (Llama 3, OLMo 2, Olmo 3, SmolLM3), stepwise constant (DeepSeek-V3), to a floor (Nemotron Nano 2, GLM-5), or exponentially (MiniCPM).
4. Optionally average checkpoints from this phase (§6).

**Worked example: MiniCPM's exponential decay.** MiniCPM prints `f(s − T) = 0.5^((s−S)/T)` "in which T is set to be 5000 steps (20B tokens)" ([[minicpm]] §6.2). As printed, s < S makes the exponent negative and f > 1, which contradicts 0 < f ≤ 1 in Eq. 1. Read as a half-life of 5000 steps from the start of decay (Interpretation), the learning rate from η = 0.01 is 0.01 × 0.5^(5000/5000) = 0.005 after 5000 steps and 0.01 × 0.5^(20000/5000) = 6.25 × 10^-4 after 20,000 steps. The authors note that loss "still drops after the learning rate drops below 10% of the max learning rate" (§6.4), which this reading reaches after 16,610 steps (log2(10) × 5000).

**Evidence: how long the decay phase is.**

| Model | Decay or anneal | Share of pretraining tokens | Locus | Status |
|---|---|---|---|---|
| DeepSeek-V3 671B | cosine 2.2e-4 → 2.2e-5 over 4.3T after a constant LR to 10T; then 333B at 2.2e-5 and 167B at 7.3e-6 | 29.1% (cosine) + 3.4% (final 500B) | [[deepseek-v3-recipe]] §4.2 | derived from verified |
| Nemotron-Nano-12B-v2-Base | WSD 4.5e-4 → 4.5e-6 over the final 3.6T of 20T | 18.0% | [[nemotron-nano-2]] §2.5 | derived from verified |
| SmolLM3-3B | blog: decay stage 10T → 11.1T, "linear decay to 0 in the final 10% training steps" | 9.9% of 11.1T (blog stage boundaries; the blog's stated total is 11.2T); 11.1% of steps (config) | [[smollm3-midtraining]]; [[smollm3-training-configs]] stage1_8T.yaml L206-212 | derived |
| MiniCPM 0.036B ablation | decay of 10% of tokens "sufficient"; 2.5% "falls short" | 10% (stated) | [[minicpm]] §4.2 Fig. 5 | verified |
| OLMo 2 7B | linear decay to 0 over each 50B mid-training run | 1.3% per run | [[olmo-2-midtraining]] §2.3 | derived from verified |
| Llama 3.1 405B | final 40M tokens, LR linearly to 0, 128K context | 0.00026% | [[llama-3-recipe]] §3.4.3 | derived from verified |

The shares span five orders of magnitude, so "anneal" names different operations. Llama 3's 40M-token anneal comes at the end of a 1,200,000-step cosine schedule whose floor is 8 × 10^-7, and the report does not print the learning rate at which the anneal starts ([[llama-3-recipe]] §3.4.1, §3.4.3); Nemotron Nano 2's 3.6T decay starts from the stable learning rate. A controlled comparison of decay length exists at small scale: Hägele et al. report that a cooldown of about 20% of steps matches a length-matched cosine schedule from 33M to 360M parameters and for a 1B model on 100B tokens ([[cooldown-scaling-beyond-fixed-durations]] §3.2, §5). **Replicated in direction** (MiniCPM 10%, Hägele et al. 20%) at ≤ 1B parameters; no source in this chapter tests decay length above 1B.

**Evidence: what data in the decay phase does.**
- MiniCPM Table 1 ([[minicpm]]): a 2.4B stable checkpoint decayed with high-quality and SFT data, then fine-tuned on 4B SFT tokens (A-2), scored C-Eval 52.6, MMLU 50.9, GSM8K 42.3, against 40.0, 44.6, 27.7 when decayed on pretraining data only (A-1). For 1.2B, doubling SFT tokens after a plain decay (B-2) left MMLU at 47.9 (B-1 also 47.9), while SFT data in the decay (B-3) gave 49.6. B-3 scored lower on GSM8K than B-1 (31.8 vs 34.2). One run per setting; no seeds reported. **Result (single study).**
- Llama 3 ([[llama-3-recipe]] §3.1.3): annealing on GSM8k and MATH training sets raised the 8B model's validation scores by 24.0% and 6.4% and had a "negligible" effect at 405B. The authors excluded benchmark training sets from the released annealing data "to assess the true few-shot learning capabilities and out-of-domain generalization". **Result (single study).**
- OLMo 2 ([[olmo-2-midtraining]] Table 8): peak learning rates 3e-4 and 6e-4, each decayed to 0 over 100B high-quality tokens after 2T tokens, gave OLMES 73.8 and 73.9. The pretraining learning rate did not change the post-decay score on this suite; GSM8K was 2.8 points higher for 6e-4. **Result (single study).**
- GLM-4.5 uses a cosine schedule with no stable phase, decaying to 2.5e-5 at the end of mid-training, and states that in early experiments "models trained with the WSD schedule perform worse on general benchmarks (SimpleQA, MMLU), indicating underfitting in the stable stage"; no numbers are printed ([[glm-4-5-recipe]] §2.4). This is the only source here that reports WSD as worse. **Result (single study), no numbers.**

**Conditions and limits.** A-2 versus A-1 changes both the decay data and the amount of SFT-format data the model sees; B-2 controls for SFT token count only at 1.2B. The Llama 3 benchmark-anneal result mixes training-set exposure with capability gain, which is why the authors excluded it. None of these studies measures held-out task families outside the tuned benchmarks.

**Implication for a general-purpose model.** The decay phase is the cheapest place to move benchmark scores, including by training on benchmark-adjacent data. The evaluation gate for this phase needs tasks that the decay mix does not contain (§3).

## §3 Mid-training budgets and evaluation gates

**Definition.** **Mid-training** is a stage after the main pretraining mix and before SFT that changes the data distribution toward target capabilities, usually during the learning-rate decay. OLMo 2 adopts the term from Abdin et al. (2024) and OpenAI (2024) and describes the stage as "5–10% of training FLOPs" ([[olmo-2-midtraining]] §2.3, footnote 4). SmolLM3 uses the term for both long-context and reasoning adaptation ([[smollm3-midtraining]]). GLM-4.5 and GLM-5 use it for the stages that extend context and add repository and agent data ([[glm-4-5-recipe]]; [[glm-5]]).

**Problem.** Mid-training is chosen by comparing candidate mixes. A full comparison at the release scale costs a full mid-training run per candidate, and the benchmarks used to choose mixes can overlap with the mix.

**Budget ledger.** Ratios are derived with the §1 formula from verified values; the figure plots them.

| Model | Mid-training tokens (seen) | Pretraining tokens | f | Data focus | Locus |
|---|---|---|---|---|---|
| OLMo 2 7B | 3 × 50B, averaged | 3.90T | 1.3% per run; 3.8% all runs | high-quality web, FLAN, peS2o, Wikipedia, math | [[olmo-2-midtraining]] §2.3, Table 13 |
| OLMo 2 13B | 3 × 100B + 1 × 300B, averaged | 5T | 2.0% per 100B run; 12.0% all | same, with repeats | §2.3, Table 13 |
| OLMo 2 32B | 3 × 100B + 1 × 300B, averaged | 6.06T (§2.3); 7T (Table 9 caption) | 9.9% all (6.06T) | same | §2.3, Table 9 |
| Olmo 3 7B | 100B | 5.93T | 1.7% | math, code, QA, thinking traces, instruction data | [[olmo-3-base-stages]] Tables 5, 35 |
| Olmo 3 32B | 2 × 100B, averaged | 5.5T | 1.8% per run; 3.6% both | same | Table 35 |
| SmolLM3-3B | 35B unique × 4 epochs ≈ 140B | 11.1T (end of the blog's decay stage; stated total 11.2T) | 1.3% seen; 0.3% unique | reasoning traces (OpenThoughts3, Llama-Nemotron) | [[smollm3-midtraining]] |
| GLM-4.5 355B | 500B + 500B + 100B = 1.1T | 22T (15T + 7T) | 5.0% | repo code, synthetic reasoning, long documents, agent trajectories | [[glm-4-5-recipe]] Fig. 3 |
| GLM-5 744B | 1T + 500B + 50B = 1.55T | 27T | 5.7% | SWE data, long documents, agent trajectories | [[glm-5]] §1, §2.3 |

OLMo 2's "5–10% of training FLOPs" statement is close to the 13B and 32B totals when all ingredients are counted (600B of the reported 5.6T and 6.6T totals: 10.7% and 9.1% of tokens), but a single 7B run is 1.3% of pretraining tokens and all three 7B runs are 3.7% of the 4.05T total. The general statement is in FLOPs and the per-size ratios are in tokens, so they measure different things (Derived).

**Evaluation gates as reported.**
1. **Microanneals.** OLMo 2 anneals a 7B checkpoint on a 50/50 mix of a candidate source and general web data, with the learning rate driven linearly to zero; 19 microanneals used 130B tokens in total, "less compute than the 3 50B token souping ingredients" ([[olmo-2-midtraining]] §4.4.2). Table 12: from a baseline GSM* of 28.5, a 35/65 math/web mix reached 63.5 with 576M tokens and a 10/90 mix reached 61.0 with 1.72B tokens. Olmo 3 standardizes this as 5B target + 5B web tokens compared against a 10B web-only anneal ([[olmo-3-base-stages]] §3.5.1).
2. **Integration tests with SFT.** Olmo 3 runs full 100B anneals on candidate mixes and fine-tunes each result, "to verify that gains we observe in midtrain yield improvements beyond base model capabilities" (§3.5.1). Table 6: base average 49.7 → 50.7 → 53.1 and SFT average 35.2 → 35.3 → 37.3 across Rounds 1, 3, 5.
3. **Development versus held-out splits.** OLMo 2 used 200 of the 1,319 GSM8K test examples (GSM*) for mix decisions and reports the remaining 1,119 as held out ([[olmo-2-midtraining]] §2 footnote 6). The Dolmino math pool includes the GSM8K train split (Table 5).
4. **Domain trade-off runs.** Olmo 3 trained two deliberately skewed 100B mixes. The math-code-thinking mix scored Math 60.8 and Code 35.6 against 57.3 and 31.2 for the final mix, and GenQA 65.9 against 73.1 (Table 7).
5. **Decontamination before selection.** Olmo 3 removed benchmark contamination from all splits before Round 5; GSM8K contamination did not raise scores, which the authors attribute, following Marin, to format mismatch (§3.5.4).
6. **Stage-advance rules for context.** Llama 3 advances each context stage when "model performance on short-context evaluations has recovered completely" and the model "perfectly solves 'needle in a haystack' tasks up to that length" ([[llama-3-recipe]] §3.4.2).

**Evidence of effect size.** OLMo 2 Table 9 (pretraining → pretraining & mid-training): 7B average 53.0 → 62.9, MMLU 59.8 → 63.7, GSM8K 24.1 → 67.5; 13B average 58.9 → 68.3; 32B average 66.3 → 73.3 ([[olmo-2-midtraining]]). Table 9 separates development benchmarks (MMLU, ARC-C, HellaSwag, WinoGrande, NQ, DROP) from held-out evaluations (AGIEval, GSM8K, MMLU-Pro, TriviaQA); the GSM8K entry is held out only in the sense of footnote 6 (200 of its test examples guided mix decisions), and the Dolmino math pool contains the GSM8K train split. Olmo 3 Table 10: removing thinking traces and instruction data from an intermediate mix at constant tokens lowered the base average from 50.7 to 48.8 and Math from 48.7 to 43.1 ([[olmo-3-base-stages]]). **Result (single study)** for each.

**A formatting failure found by the gate.** Olmo 3 found that chat special tokens in mid-training data made the base model emit them at inference: GSM8K fell from 49.43 to 0 and CruxEval from 32.89 to 18.91 in microanneals; with a chat template written as ordinary text, the scores were 46.02 and 29.65 ([[olmo-3-base-stages]] §3.5.4). The released mix uses newline-based formatting instead. See ch-04 for template effects at SFT.

**Implication.** The reports that print mid-training ablations (OLMo 2, Olmo 3) select mixes with microanneals plus a broad base suite plus an SFT probe, and they report the domains that lose score. A general-purpose run needs all three, because single-domain microanneals show gains in the target domain and losses elsewhere (Olmo 3 Tables 8–9: Reddit-to-Flashcards raised MMLU 55.6 → 58.8 and lowered GSM8K 22.4 → 21.2; reasoning data raised GSM8K 18.4 → 26.8 and lowered MMLU 55.2 → 53.7).

## §4 Context-extension ledger

**Definition.** **Context extension** is continued training at sequence lengths longer than pretraining, usually with a change to the position encoding. **Adjusted base frequency (ABF)** raises the RoPE base b ([[llama-2-long]]). **YaRN** rescales RoPE frequencies per dimension and multiplies attention logits by a temperature ([[yarn]]). Methods are compared in ch-32b.

**Problem.** Attention cost grows with sequence length, so reports pretrain at 4K–8K and extend late ([[llama-3-recipe]] §3.4.2 gives this reason). The extension stage must raise long-context scores without lowering short-context scores measured before it.

**Formula (RoPE wavelengths).**

```
θ_i = b^(−2i/d),   λ_i = 2π / θ_i = 2π · b^(2i/d),   i = 0, …, d/2 − 1
```

- `b` is the RoPE base; `d` the attention head dimension; `θ_i` the rotation per position of dimension pair i.
- `λ_i` is the number of positions for one full rotation of pair i.

**Worked example.** For d = 128, the slowest pair (i = 63) has λ = 2π · b^(126/128). With b = 10,000, λ ≈ 54,410 positions; with b = 500,000 (Llama 3), λ ≈ 2.56M; with b = 5,000,000 (SmolLM3 at 64K), λ ≈ 24.7M. Raising b lengthens every wavelength except that of pair 0, so fewer pairs complete a rotation within the training window (Interpretation; the reports do not print this calculation). The YaRN temperature in Qwen2.5-1M and DeepSeek-V3 is √(1/t) = 0.1 ln(s) + 1, where s is the length ratio: s = 40 gives 1.369 and s = 4 gives 1.139 ([[qwen-long-context-synth]] Eq. 1; [[deepseek-v3-recipe]] §4.3).

**Ledger.** "n.r." = not reported in the checked source.

| Model | Pretrain length → target | Stages and tokens | Position method and when set | Long:short (unit) | Gate or evidence | Locus |
|---|---|---|---|---|---|---|
| Llama 3.1 405B | 8K → 128K | six stages, ~800B tokens; per-stage lengths n.r. | RoPE θ = 500,000 set for pretraining; no change reported at extension | n.r. | short-context evals fully recovered + NIAH at each length | [[llama-3-recipe]] §3.2, §3.4.2 |
| Qwen2.5 0.5B–72B | 4,096 → 32,768 | final pretraining stage; tokens n.r. | ABF 10,000 → 1,000,000 at this stage; YaRN + DCA at inference to 131,072 | n.r. | RULER-128K for 72B-Instruct 67.0 → 88.4 with DCA + YaRN (Table 16) | [[qwen-2.5-recipe]] §3.3 |
| Qwen2.5-Turbo (MoE, API) | 32,768 → 262,144 | three stages; tokens n.r. | base 10,000,000 | 40% at max length / 60% shorter (sequences) | n.r. | [[qwen-2.5-recipe]] §3.3 |
| Qwen2.5-7B/14B-1M | 32,768 → 262,144 | 65,536 / 131,072 / 262,144; tokens n.r. | bases 1M / 5M / 10M, one per stage | 75% / 25% (sequences) | 14B RULER-128K 37.6 → 56.0 → 83.8 → 87.6 (Table 2) | [[qwen-long-context-synth]] §3 |
| ProLong 8B (from Llama-3-8B-Instruct) | 8K → 512K | 20B at 64K + 20B at 512K (B = 2^30) | base 5 × 10^5 → 8 × 10^6 → 1.28 × 10^8 | 63% long (incl. 3% textbooks) / 37% ShortMix at 64K (share type not stated) | Table 18: base 8e6 54.6 vs 4e6 48.7 vs 5e5 29.1 | [[prolong-recipe]] Table 9 |
| DeepSeek-V3 671B | 4K → 128K | 1000 steps at 32K + 1000 at 128K; 125.8B derived | YaRN on the decoupled key, s = 40, α = 1, β = 32 | n.r. | NIAH to 128K after SFT (Fig. 8) | [[deepseek-v3-recipe]] §4.3 |
| Olmo 3 7B / 32B | 8,192 → 65,536 | 50B / 100B | θ = 5 × 10^5 unchanged; YaRN factor 8 on full-attention layers only | 34% long / 66% (tokens) | RULER dev, HELMET held out; Fig. 13 | [[olmo-3-base-stages]] §3.6; [[olmo-core-olmo3-configs]] |
| SmolLM3-3B | 4,096 → 64K (128k with YaRN at inference) | 50B at 32K + 50B at 64K | θ 50,000 → 1.5M (blog) or 2,000,000 (config) → 5M; NoPE every 4th layer from pretraining | n.r. (math, code, reasoning upsampled) | RULER, HELMET in ablations | [[smollm3-midtraining]]; [[smollm3-training-configs]] |
| Nemotron-Nano-12B-v2-Base | 8,192 → 128k target | Phase LC 18.9B at 524,288 | no position embeddings (hybrid Mamba-Transformer) | 80% Phase 3 blend / 20% synthetic long-document QA (weights) | Table 4 (Nemotron-H 8B): RULER-128k 73.68 at 128k vs 81.04 at 512k | [[nemotron-nano-2]] §2.1, §2.6 |
| GLM-4.5 355B | 4K → 128K | 32K stages (1T) then 100B at 128K | base 10,000 → 1,000,000 at the 32K extension; 128K n.r. | n.r. | n.r. | [[glm-4-5-recipe]] §2.3–2.4 |
| GLM-5 744B | 4K → 200K | 1T at 32K, 500B at 128K, 50B at 200K | n.r. | n.r. | 200K stage improved results within 128K (no numbers in §2.3) | [[glm-5]] §2.3 |

Budgets relative to pretraining (Derived): Llama 3 5.1%, GLM-5 128K + 200K stages 2.0%, Olmo 3 32B 1.8%, SmolLM3 0.90%, DeepSeek-V3 0.85%, Olmo 3 7B 0.84%, GLM-4.5 128K stage 0.45%, Nemotron Nano 2 0.095%.

**Evidence 1: train longer than the target length.** Four sources report higher scores at a target length after training beyond it. Nemotron-H 8B: RULER-128k 81.04 at 512k versus 79.04 at 256k and 73.68 at 128k, all with synthetic data ([[nemotron-nano-2]] Table 4). Qwen2.5-14B-1M: the 262,144-token stage raised RULER at 128K from 83.8 to 87.6 ([[qwen-long-context-synth]] Table 2). ProLong: 4B tokens at 512K versus at 64K, evaluated at 64K, gave recall 98.5 versus 95.0 and re-rank 32.9 versus 28.0 ([[prolong]] Table 7). GLM-5: a 200K stage after 128K "further bolstered the model's performance even within the 128K context window" ([[glm-5]] §2.3, no numbers). **Replicated** across four model families; each study uses one run per setting.

**Evidence 2: short data in the extension mix.** Olmo 3's 10B-token extension experiment: 66% long / 34% short lowered a subset of OlmoBaseEval by 2.5 points; 34% long / 66% short lowered it by 0.8 points ([[olmo-3-base-stages]] §3.6.3). ProLong's 5B-token Llama-3-8B ablations: short-task average fell monotonically as the long share rose, and 60% long gave the best long-context average after SFT ([[prolong]] §3.2, Fig. 3). **Replicated in direction** (both show short-task loss rising with long share); the preferred long share differs (34% versus 60%) with different data, lengths, and gates.

**Evidence 3: short-context regression still occurs.** Olmo 3 Table 13 compares the midtrained and extended base models ([[olmo-3-base-stages]]): 7B Math 59.8 → 54.4 and GenQA 71.3 → 72.5; 32B Math 69.7 → 61.4 and MMLU 76.9 → 76.2. Qwen2.5-14B-Instruct-1M against the 128K Qwen2.5-14B-Instruct: GPQA 45.5 → 39.9 and LiveCodeBench 42.6 → 38.6, with IFEval 81.0 → 84.3 ([[qwen-long-context-synth]] Table 6). The Qwen report summarizes these as similar; the Olmo 3 text in §3.7 compares Olmo 3 with OLMo 2 and does not comment on the Stage 2 → Stage 3 change. **Result (single study)** each.

**Evidence 4: placement.** Llama 3 extends before the final anneal ([[llama-3-recipe]] §3.4.2–3.4.3). Qwen2.5 extends in the final pretraining stage, and Qwen2.5-1M continues from an intermediate base checkpoint ([[qwen-long-context-synth]] §3). Olmo 3 extends after midtraining, SmolLM3 extends after the decay stage and before reasoning mid-training, and Nemotron Nano 2 extends after Phase 3. GLM-4.5 and GLM-5 extend inside mid-training. The Olmo 3 report states that GLM 4.5 "applies extension only after supervised finetuning" (§3.6), which disagrees with GLM-4.5's Figure 3 and §2.3; this chapter uses GLM-4.5's own report. No source compares placements in a controlled experiment. **Open question.**

**Conditions and limits.** RULER, used as the development metric by Olmo 3 and Qwen2.5-1M, is synthetic and partly overlaps with HELMET (Olmo 3 footnote 25); ch-32c explains why claimed and effective context differ. Nemotron's length ablation is on Nemotron-H 8B, not the released 12B model. ProLong's ablations are 5B-token runs at 8B parameters.

## §5 Agentic mid-training ledger

**Definition.** **Agentic mid-training** adds repository-level code, software-engineering histories, or synthetic agent trajectories to mid-training so that SFT and RL start from a model that has seen multi-step tool and code contexts. Data construction is covered in ch-32d.

| Setting | GLM-4.5 355B | GLM-5 744B |
|---|---|---|
| Stage lengths and tokens | 500B repo code at 32K; 500B synthetic reasoning at 32K; 100B long-context and agent data at 128K (Fig. 3) | 1T at 32K; 500B at 128K; 50B at 200K (§2.3) |
| Repository data format | files of one repository concatenated; model-filtered issues, PRs, and commits in diff-like format (§2.3) | repo files, commit diffs, issues, PRs, and relevant files in one sequence; ~10M issue–PR pairs; ~160B unique tokens after filtering (§2.3) |
| Agent trajectories | "large-scale synthetic agent trajectories" at 128K (§2.3) | "up-sampled at the later stages" (§2.3) |
| Agent-trajectory token share | not reported | not reported |
| SWE-data share per stage | not reported | not reported |
| Trajectory generator and filter | not reported for mid-training | not reported for mid-training |
| Learning rate | cosine, decaying to 2.5e-5 at the end of mid-training (§2.4) | linear 4e-5 → 1e-5 (App. A) |
| RoPE or position setting | base 10,000 → 1,000,000 at the 32K extension (§2.4) | not reported |
| Packing | best-fit packing in mid-training only (§2.3) | interleaved packing of similar long texts (§2.3) |
| Ablation of the agentic share | not reported | not reported |
| Sources | [[glm-4-5-recipe]] | [[glm-5]] |

**Evidence.** Neither report isolates the effect of agentic mid-training data on agent benchmarks or on general benchmarks; the gains they report combine mid-training, SFT, and RL. The GLM-5 introduction describes mid-training as "focusing specifically on long-context agentic data to ensure stability in complex workflows" ([[glm-5]] §1), which is a design statement, not a measurement. **Open question** for the share and its effect on breadth.

**Implication.** A general-purpose run that copies these rows can copy stage lengths and data formats but not an agentic token share, because none is printed. The ablation designs in ch-32d and ch-32f are needed before choosing one.

## §6 Checkpoint averaging and merging as named stages

**Definition.** A **checkpoint average (soup)** is the element-wise mean of several checkpoints' weights. A **linear merge** is a weighted mean of two or more models.

```
θ_soup = (1/n) Σ_{k=1..n} θ_k          θ_merge = (1 − α) · θ_A + α · θ_B
```

- `θ_k` are the weights of n runs or checkpoints of the same architecture; `α` ∈ [0, 1] is the merge weight on model B.

**Worked example.** Take one scalar weight with value 0.30 in an APO soup (θ_A) and 0.50 in the mid-training checkpoint (θ_B). SmolLM3's 0.9 / 0.1 merge gives 0.9 × 0.30 + 0.1 × 0.50 = 0.32, so each merged weight moves 10% of the way from the post-trained value toward the mid-training value ([[smollm3-midtraining]]).

**Ledger.**

| Model | Where | What is averaged | Weights | Evidence | Locus |
|---|---|---|---|---|---|
| Llama 3.1 405B | end of annealing | checkpoints during annealing (Polyak averaging) | n.r. | no ablation reported | [[llama-3-recipe]] §3.4.3 |
| OLMo 2 7B | mid-training | 3 runs × 50B, different data orders | uniform | Table 14: soup ≥ best single on OLMES in all six mixes; GSM* mix A 71.0 → 74.0, mix E 60.5 → 43.0 | [[olmo-2-midtraining]] §4.5 |
| OLMo 2 13B, 32B | mid-training | 3 × 100B + 1 × 300B | uniform | stated as empirically better than averaging the three 100B runs alone (no numbers) | §4.5 |
| Olmo 3 32B | midtraining | 2 runs × 100B, different data-order seeds | uniform | Math cluster +2.9 and +1.6 over the two runs; about +1 MMLU | [[olmo-3-base-stages]] §3.5.4 |
| Olmo 3 7B | midtraining | none | — | "Initial experimentation for the 7B model did not show similar gains" | footnote 24 |
| Olmo 3 32B | long-context end | checkpoints at steps 10,000, 11,000, 11,921 | uniform | no ablation reported | §3.6.4 |
| SmolLM3-3B | after APO | APO checkpoint soup, then 0.9 × soup + 0.1 × mid-training checkpoint | 0.9 / 0.1 | stated as the best-performing weights; RULER recovered to the base score up to 128k; no numbers printed | [[smollm3-midtraining]] |
| Nemotron-Nano-v2 12B | after RL | reasoning-strong and chat-strong RL checkpoints | α swept 0.1–0.9 in steps of 0.1 | "values around 0.5 offered a good trade-off" (no table) | [[nemotron-nano-2]] §3.2 |

The step 11,921 in Olmo 3 32B equals 100B tokens / 8,388,608 tokens per batch, so the last averaged checkpoint is the end of the 100B extension run (Derived from Table 35).

**Conditions and limits.** The OLMo 2 caption states that souping "consistently" equals or beats the best single checkpoint, but its own mix E row shows GSM* 60.5 for the best single run and 43.0 for the soup. Olmo 3 found merging helpful at 32B and not at 7B. SmolLM3's recovery is reported without scores. **Result (single study)** in each case; the averaging benefit depends on model size and metric.

**Implication.** Merging with an earlier-stage checkpoint is the only reported method in these sources that restores a lost capability without new training (SmolLM3). Merge weights are chosen per release, and the evaluation must include the capability that the later stage lowered. Chapter ch-30c covers merge methods.

## §7 What was verified, what conflicts, and what is not reported

Verification on 2026-09-14 and 2026-09-15 found four conflicts between a paper or blog and its released configuration, or between two parts of one report:
1. SmolLM3 4k→32k RoPE θ: blog 1.5M; `long_context_4k_to_32k.yaml` 2,000,000 ([[smollm3-training-configs]]).
2. Olmo 3 7B midtraining and extension peak LR: Table 35 2.074e-4; both 7B scripts 2.0712e-4 ([[olmo-core-olmo3-configs]]). The 7B extension script's `max_duration` (5T tokens) and `hard_stop` (step 597046) do not correspond to a 50B stage. The 32B long-context script's `load_path` points to a pretraining checkpoint, while the paper starts extension from the merged midtrained model.
3. ProLong 512K stage peak LR: Table 9 1e-5; `train_512K.sh` 5e-6 ([[prolong-recipe]]).
4. OLMo 2 32B pretraining tokens: §2.3 6.06T; Table 9 caption 7T ([[olmo-2-midtraining]]).

Values that readers expect but the sources do not print: Llama 3 per-stage extension lengths and tokens; DeepSeek-V3 extension tokens; Qwen2.5 and Qwen2.5-1M extension tokens and learning rates; long:short ratios for Llama 3, DeepSeek-V3, SmolLM3, GLM-4.5, and GLM-5; agentic token shares for GLM-4.5 and GLM-5; merge weights for Llama 3.

The library cards [[olmo-2]], [[olmo-3]], and [[smollm-3]] predate this verification and contain values that disagree with the reports (for example, "extended to 32K in cooldown" for OLMo 2, which trains at 4096 throughout). The chapter's values for these three models come from the verified excerpts [[olmo-2-midtraining]], [[olmo-3-base-stages]], and [[smollm3-midtraining]].

## Recipe

Rows are quoted from the verified cards and excerpts linked in each row. "v3", "v4" and similar refer to arXiv versions. Values are as printed; units follow §1.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3.1 405B | 405B | pretrain-decay/anneal | final anneal | final 40M tokens; LR linearly to 0; 128K context; high-quality sources upsampled | arXiv:2407.21783v3 §3.4.3 ([[llama-3-recipe]]) | verified 2026-09-14 | no ablation reported |
| Llama 3.1 405B | 405B | merge | anneal averaging | Polyak averaging of checkpoints during annealing; weights not printed | v3 §3.4.3 | verified 2026-09-14 | no ablation reported |
| Llama 3 8B (experiment) | 8B | eval-gate | anneal-as-data-evaluation | 50%-trained 8B; LR linearly to 0 over 40B tokens; 30% new data, 70% default mix | v3 §3.1.3 | verified 2026-09-14 | stated as more efficient than scaling-law runs; no numbers |
| Llama 3.1 405B | 405B | long-context | stages; tokens; position | six stages 8K → 128K; ~800B tokens; RoPE θ 500,000 from pretraining; per-stage lengths not printed | v3 §3.2, §3.4.2 | verified 2026-09-14 | stage advanced when short-context evals recovered and NIAH solved |
| DeepSeek-V3 | 671B / 37B act. | pretrain-decay/anneal | decay | cosine 2.2e-4 → 2.2e-5 over 4.3T after constant to 10T; 333B at 2.2e-5; 167B at 7.3e-6 | arXiv:2412.19437v2 §4.2 ([[deepseek-v3-recipe]]) | verified 2026-09-14 | no ablation reported |
| DeepSeek-V3 | 671B / 37B act. | long-context | phases | YaRN on decoupled key k^R (s 40, α 1, β 32); 1000 steps at 32K batch 1920; 1000 steps at 128K batch 480; LR 7.3e-6 | v2 §4.3 | verified 2026-09-14 | Fig. 8 NIAH to 128K after SFT |
| DeepSeek-V3 | 671B / 37B act. | long-context | tokens | 125.8B (K = 1,024, batch in sequences) or 122.9B (K = 1,000) | inputs from v2 §4.3 | derived | V3.1 model card: 630B phase is "10-fold" the V3 phase ([[deepseek-v3.1-recipe]]) |
| Nemotron-Nano-12B-v2-Base | 12B | pretrain-decay/anneal | schedule; phases | 20T horizon; WSD stable 4.5e-4, min 4.5e-6, decay over final 3.6T; phase switches at 60% and 90%; batch 768 × 8192 | arXiv:2508.14444v4 §2.2, §2.5 ([[nemotron-nano-2]]) | verified 2026-09-15 | no ablation reported for the schedule |
| Nemotron-Nano-12B-v2-Base | 12B | long-context | Phase LC | 18.9B tokens at 524,288; constant LR 4.5e-6; batch 12; Phase 3 weights × 0.8 + 20% synthetic long-document QA | v4 §2.6 | verified 2026-09-15 | Table 4 (Nemotron-H 8B): RULER-128k 73.68 / 70.19 / 79.04 / 81.04 for 128k / 256k no synth / 256k / 512k |
| MiniCPM-2.4B, MiniCPM-1.2B | 2.4B, 1.2B | pretrain-decay/anneal | decay data; form | pretraining data + high-quality SFT data; exponential annealing, "T is set to be 5000 steps (20B tokens)"; stable LR 0.01 | arXiv:2404.06395v3 §6.2 ([[minicpm]]) | verified 2026-09-15 | Table 1: A-2 vs A-1 C-Eval 52.6 vs 40.0, GSM8K 42.3 vs 27.7 (one run each) |
| MiniCPM ablation models | 0.036B | pretrain-decay/anneal | decay length | 10% of total tokens sufficient; 2.5% falls short | v3 §4.2, Fig. 5 | verified 2026-09-15 | Fig. 5 loss curves at 40N, 60N, 80N data |
| SmolLM3-3B | 3B | pretrain-decay/anneal | decay stage (blog) | 10T → 11.1T; web 63%, code 24%, math 13%; linear decay to 0 over final 10% of steps | huggingface.co/blog/smollm3 "Data mixture and training stages" ([[smollm3-midtraining]]) | verified 2026-09-15 | mixture ablations at 3B on 50B–100B tokens; values not printed |
| SmolLM3-3B | 3B | pretrain-decay/anneal | decay (config) | decay starts step 4,198,001; 522,000 decay steps of 4,720,000; 2,359,296 tokens/step → starts ≈ 9.90T, ≈ 1.23T tokens | huggingface/smollm stage1_8T.yaml L206-212, L250-255 ([[smollm3-training-configs]]) | derived | n/a |
| OLMo 2 7B | 7B | pretrain-stable | truncation | cosine to 10% of 3.0e-4 over 5T, truncated after 4T; 2000 warmup steps | arXiv:2501.00656v3 Table 3, §4.1 ([[olmo-2-midtraining]]) | verified 2026-09-15 | Table 8: peak 3e-4 vs 6e-4 after 100B decay, OLMES 73.8 vs 73.9 |
| OLMo 2 7B | 7B | mid-train | budget; schedule | 3 runs × 50B Dolmino tokens, LR linearly to 0, averaged | v3 §2.3, §4.5 | verified 2026-09-15 | Table 14: soup ≥ best single on OLMES in 6 mixes; GSM* lower in mix E (60.5 → 43.0) |
| OLMo 2 13B, OLMo 2 32B | 13B, 32B | mid-train | budget | 3 runs × 100B + 1 run × 300B, averaged | v3 §2.3, §4.5 | verified 2026-09-15 | "empirically better" than three 100B runs alone; no numbers |
| OLMo 2 7B | 7B | mid-train | 50B mix (% of mix tokens) | DCLM 47.2, FLAN 16.6, Dolmino Math 20.8, Wikipedia 7.11, peS2o 5.85, StackExchange 2.45 | v3 Table 13 | verified 2026-09-15 | Table 12 microanneals: math 35/65 GSM* 63.5, 10/90 61.0 |
| OLMo 2 (all) | 7B | eval-gate | microanneal | 50/50 target + web, LR linearly to 0; 19 runs, 130B tokens; GSM* = 200 of 1,319 GSM8K examples | v3 §4.4.2, §2 fn. 6 | verified 2026-09-15 | Table 12 |
| Olmo 3 7B | 7B | mid-train | schedule; batch; tokens | 0 warmup; peak 2.074e-4; linear to 0; 256 × 8,192 = 2,097,152 tokens; 100B | arXiv:2512.13961v2 Table 35 ([[olmo-3-base-stages]]) | conflict | Table 6 Rounds 1/3/5; the released script differs (next row) |
| Olmo 3 7B (released script) | 7B | mid-train | LR; batch; tokens | `LR = 0.00020712352850360292`; `GLOBAL_BATCH_SIZE = 2**21`; `MAX_TOKENS = 100_000_000_000`; `LinearWithWarmup(warmup=0, alpha_f=0.0)` | OLMo-core@66f768b OLMo-3-1025-7B-midtrain.py L40-43, L80 ([[olmo-core-olmo3-configs]]) | conflict | no ablation reported |
| Olmo 3 32B | 32B | mid-train | budget; merge | 2 runs × 100B, different data-order seeds, uniform average; peak 2.071e-4; 512 × 8,192 | v2 Table 35, §3.5.4 | verified 2026-09-15 | merged: Math +2.9 / +1.6 over single runs; 7B showed no gain (fn. 24) |
| Olmo 3 7B, 32B | 7B, 32B | mid-train | mix (token share of 99.95B) | CC HQ 22.5%, Dolmino Math 10.7%, StackEdu FIM 10.0%, CraneCode 10.0%, Reddit To Flashcards 5.9%, Flan 5.0%, Nemotron Synth QA 5.0%; thinking and instruction rows listed separately | v2 Table 5 | verified 2026-09-15 | Table 10: with thinking + instruction data avg 50.7 vs 48.8 |
| Olmo 3 | 7B | eval-gate | microanneal; integration test | 5B target + 5B web vs 10B web-only; 100B anneals followed by SFT | v2 §3.5.1 | verified 2026-09-15 | Tables 6–9 |
| Olmo 3 7B | 7B | long-context | schedule; length; tokens | 8,192 → 65,536; 200 warmup steps; peak 2.074e-4; linear to 0; 64 × 65,536 = 4,194,304 tokens; 50B | v2 Table 35 | conflict (peak LR; the released script prints 2.0712e-4, next row) | Fig. 13e: longer budgets (1B–100B) raise RULER |
| Olmo 3 7B (released script) | 7B | long-context | position; LR | `YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)`; LR 2.0712e-4; `warmup=200` | OLMo-core@66f768b OLMo-3-1025-7B-long-context.py L37-39, L49-51, L81 | conflict | Fig. 13a: YaRN on full-attention layers best |
| Olmo 3 7B, 32B | 7B, 32B | long-context | mix; packing | 34% long (olmOCR PDFs + synthetic CWE/REX), 66% Dolmino short (token share); gzip 20%/20% filter; best-fit packing; intra-document masking | v2 §3.6, Table 11 | verified 2026-09-15 | §3.6.3: 10B runs, 34/66 −0.8 vs 66/34 −2.5 on an OlmoBaseEval subset |
| Olmo 3 32B | 32B | long-context | tokens; merge | 100B; 128 × 65,536; average of steps 10,000, 11,000, 11,921 | v2 Table 35, §3.6.4 | verified 2026-09-15 | no ablation reported for averaging |
| SmolLM3-3B | 3B | mid-train | reasoning mid-training | 35B tokens × 4 epochs (~140B); ChatML; wrapped packing | blog "Reasoning Mid-training" | verified 2026-09-15 | no ablation reported; blog traces a RULER drop to this stage |
| SmolLM3-3B (blog) | 3B | long-context | stages | 50B at 4k → 32k, RoPE θ 1.5M; 50B at 32k → 64k, θ 5M; math, code, reasoning upsampled; YaRN to 128k at inference | blog "Long Context extension" | conflict | long-document upsampling did not raise RULER or HELMET (no numbers) |
| SmolLM3-3B (config) | 3B | long-context | stages | stage 1: θ 2,000,000; 32,768 × 72 sequences × 20,000 steps; LR 2e-5, 1000 warmup, cosine to 0 over 19,000 steps; stage 2: θ 5,000,000; 65,536 × 36 × 22,000 steps | smollm long_context_4k_to_32k.yaml L238, L250-256, L293-298; long_context_32k_to_64.yaml L241, L253-259, L296-301 | conflict | derived tokens 47.2B and 51.9B |
| SmolLM3-3B | 3B | merge | long-context recovery | APO checkpoint soup; linear merge 0.9 × soup + 0.1 × mid-training checkpoint (MergeKit) | blog "Model Merging" | verified 2026-09-15 | blog: 0.9 / 0.1 "achieved the best performance" among merges tried; RULER recovered to base score up to 128k; no numbers |
| Qwen2.5 open-weight | 0.5B–72B | long-context | length; base | 4,096 → 32,768 in the final pretraining stage; ABF 10,000 → 1,000,000; tokens not given | arXiv:2412.15115v2 §3.3 ([[qwen-2.5-recipe]]) | verified 2026-09-14 | no ablation reported |
| Qwen2.5-Turbo | not reported | long-context | stages; mix | 32,768 → 65,536 → 131,072 → 262,144; base 10,000,000; 40% sequences at max length, 60% shorter | v2 §3.3 | verified 2026-09-14 | no ablation reported |
| Qwen2.5-7B-1M, Qwen2.5-14B-1M | 7B, 14B | long-context | stages; bases; mix | 65,536 / 131,072 / 262,144; bases 1M / 5M / 10M; 75% at max length, 25% shorter; tokens, LR, batch not reported | arXiv:2501.15383v1 §3 ([[qwen-long-context-synth]]) | verified 2026-09-14 | Table 2 (14B) RULER-128K 37.6 / 56.0 / 83.8 / 87.6 |
| ProLong-64k-Base | 8B | long-context | stage 1 | from Llama-3-8B-Instruct; 20B tokens (B = 2^30) at 64K; base 8 × 10^6; 30% code repos, 30% books, 3% textbooks, 37% ShortMix; LR 1e-5, 10% warmup, cosine to 1e-6; 4M-token batch | arXiv:2410.02660v4 Table 9 ([[prolong-recipe]]) | verified 2026-09-14 | Table 18 base 8e6 54.6 vs 4e6 48.7; Fig. 3 60% long best |
| ProLong-512k-Base | 8B | long-context | stage 2 | 20B at 512K; base 1.28 × 10^8; peak LR 1e-5 (paper) vs 5e-6 (`train_512K.sh`) | v4 Table 9; train_512K.sh@499fa29 | conflict | Table 7: 512K beats 64K at 64K evaluation |
| GLM-4.5 | 355B / 32B act. | mid-train | stages | 500B repo code at 32K; 500B synthetic reasoning at 32K; 100B long-context + agent at 128K; RoPE base 10,000 → 1,000,000 at 32K | arXiv:2508.06471v1 Fig. 3, §2.3–2.4 ([[glm-4-5-recipe]]) | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | mid-train | LR | cosine; decays to 2.5e-5 at end of mid-training | v1 §2.4 | verified 2026-09-14 | early WSD runs worse on SimpleQA and MMLU (no numbers) |
| GLM-5 | 744B / 40B act. | mid-train | stages; LR | 1T at 32K; 500B at 128K; 50B at 200K; LR linear 4e-5 → 1e-5 | arXiv:2602.15763v2 §2.3, App. A ([[glm-5]]) | verified 2026-09-15 | 200K stage improved results within 128K (no numbers) |
| GLM-5 | 744B / 40B act. | mid-train | SWE and agent data | ~10M issue–PR pairs, ~160B unique tokens; agent-trajectory share not reported | v2 §2.3 | verified 2026-09-15 (share not reported) | no ablation reported |

**Starting point for a small general-purpose run.** Every number below comes from a verified row above, or from a conflict row with both printed values stated, and is given with the source's conditions; no source ablates the whole configuration. For a 3B–8B RoPE model pretrained at 4K–8K that should reach 32K–65K:
1. Decay before extension with a changed mix. Olmo 3 7B used 100B midtraining tokens (1.7% of 5.93T) with zero warmup and a linear decay from 2.074e-4 (Table 35; script 2.0712e-4) to 0. Select the mix with 5B + 5B microanneals against a 10B web-only anneal and a 100B integration test followed by SFT (Olmo 3, 7B).
2. Extend with short data in the mix. Olmo 3 7B used 34% long / 66% midtraining data (tokens) for 50B tokens at 65,536 with 64-sequence batches; ProLong 8B used 63% long data at 64K and reported 60% long as best in 5B-token ablations. The two sources' preferred long shares differ (34% versus 60%). When compute allows two 10B-token runs, compare both shares on the step 6 gate; otherwise start from the lower long share (Olmo 3's 34%), because in both sources short-task scores fell as the long share rose (§4 Evidence 2).
3. Choose one position method and record it. Olmo 3 7B applied YaRN (factor 8 from 8,192) to full-attention layers only and kept θ = 5 × 10^5. SmolLM3-3B raised θ from 50,000 to 1.5M (blog) or 2,000,000 (config) at 32K and to 5M at 64K. ProLong 8B raised the base to 8 × 10^6 at 64K, which scored 54.6 against 48.7 at 4 × 10^6.
4. Re-warm briefly and decay. Olmo 3 7B: 200 warmup steps to 2.074e-4, linear to 0. SmolLM3-3B: 1000 warmup steps to 2e-5, cosine to 0. ProLong 8B: 10% warmup to 1e-5, cosine to 1e-6. The peak learning rates differ by up to about 20×, and none was ablated.
5. Pack and mask by document: best-fit packing with intra-document masking (Olmo 3; ProLong Table 20 masks 54.6 vs 53.6 long average).
6. Gate on three suites before and after the stage: RULER at each length as the development metric, HELMET as a held-out suite (Olmo 3), and the full short-context base suite including Math, because Olmo 3 7B's Math cluster fell from 59.8 to 54.4 in this stage.
7. After post-training, re-run RULER. If it falls, test a linear merge of 0.9 × post-trained model + 0.1 × a mid-training checkpoint with strong long-context scores (SmolLM3-3B) and re-run the short-context suite on the merge.

## Generalization lens

**(a) What increases breadth.**
- Broad mid-training mixes with thinking and instruction data raised every OlmoBaseEval cluster at constant tokens (average 48.8 → 50.7; [[olmo-3-base-stages]] Table 10; Result, single study).
- Balanced domain weights: Olmo 3's final mix kept GenQA at 73.1 while a math-code-thinking mix fell to 65.9 ([[olmo-3-base-stages]] Table 7).
- Keeping short data in extension: 66% short cost 0.8 points versus 2.5 for 34% short ([[olmo-3-base-stages]] §3.6.3); ProLong's short-task average rises with the short share ([[prolong]] Fig. 3) (Replicated in direction).
- Training beyond the target length raised long-context scores at the target length in four families ([[nemotron-nano-2]] Table 4; [[qwen-long-context-synth]] Table 2; [[prolong]] Table 7; [[glm-5]] §2.3) (Replicated).
- Checkpoint averaging of mid-training runs equaled or exceeded the best single run on OLMES in six OLMo 2 mixes and raised Math at Olmo 3 32B ([[olmo-2-midtraining]] Table 14; [[olmo-3-base-stages]] §3.5.4).

**(b) What causes narrowing or forgetting.**
- Domain-skewed mid-training: the math-code-thinking mix lowered MC Non-STEM from 77.4 to 69.6 ([[olmo-3-base-stages]] Table 7); reasoning microanneals lowered MMLU 55.2 → 53.7 (Table 9).
- Context extension lowered short-context Math: Olmo 3 7B 59.8 → 54.4, 32B 69.7 → 61.4 ([[olmo-3-base-stages]] Table 13); Qwen2.5-14B-Instruct-1M GPQA 45.5 → 39.9 ([[qwen-long-context-synth]] Table 6).
- Reasoning mid-training lowered SmolLM3's RULER after post-training until merging; the blog also notes that APO data was limited to 24k tokens ([[smollm3-midtraining]]).
- Benchmark training sets in the anneal raise the target benchmark at small scale (Llama 3 8B GSM8k +24.0%) without evidence of out-of-domain gain ([[llama-3-recipe]] §3.1.3).
- Special chat tokens in mid-training data made the base model emit them (GSM8K 49.43 → 0; [[olmo-3-base-stages]] §3.5.4).
- Averaging can lower one metric while raising the suite average (OLMo 2 mix E GSM* 60.5 → 43.0; [[olmo-2-midtraining]] Table 14).

**(c) How to measure it at this stage.**
- Base-model suites with per-cluster scores before and after each stage (Olmo 3 Table 13 layout), not only an average.
- An SFT probe on each candidate mix, because base gains need not survive fine-tuning (Olmo 3 §3.5.1, Table 6).
- A held-out split of any benchmark used for mix decisions (OLMo 2 GSM*: 200 dev, 1,119 held out) and decontamination of all splits before comparing mixes (Olmo 3 §3.5.3).
- For long context: a development suite (RULER) and a held-out suite (HELMET), with the overlap noted (Olmo 3 footnote 25); ch-32c covers effective-length measurement.
- Known measurement errors: Olmo 3's mix rounds were run on earlier pretrained checkpoints and are "not ... final midtraining numbers" (§3.5.4); Table 6 Rounds 1 and 3 used non-decontaminated data; RULER and LongBench-Chat stop at 128K and 100K ([[qwen-long-context-synth]] Findings); most ablations here are single runs without seeds.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Copying a stage budget without its unit | a replication trains 2.4% more or fewer tokens, or 4× fewer seen tokens | recompute steps × batch (sequences) × length with K stated; count epochs (§1 formula) |
| Counting all souping ingredients as one model's tokens | "mid-training share" looks 3× larger than the compute one checkpoint saw | separate per-run tokens from total tokens (OLMo 2 §2.3: 3.95T per ingredient vs 4.05T total) |
| Taking a RoPE base from a blog when the config differs | long-context scores differ from the release at the first stage | diff blog values against released configs (SmolLM3 θ 1.5M vs 2,000,000) |
| Using a released script's duration fields as the stage budget | a 50B-token stage runs for trillions of tokens or stops at the wrong step | check `max_duration` and `hard_stop` against the paper's token count (Olmo 3 7B long-context script) |
| Extending with mostly long data | RULER rises while MMLU, GSM8K, or Math clusters fall | before/after short-context suite; compare 34/66 and 66/34 mixes on a 10B run (Olmo 3 §3.6.3) |
| Selecting the mix on the same benchmark that the mix contains | gains on DROP, Minerva, or SQuAD that shrink after decontamination (Olmo 3 found GSM8K was not inflated) | hold out a split; rerun the winning mix with decontaminated data (OLMo 2 GSM*; Olmo 3 Fig. 12, §3.5.4) |
| Leaving chat special tokens in mid-training data | base model outputs `<|im_start|>` and scores 0 on generative tasks | grep samples for special tokens; microanneal with and without them (Olmo 3 §3.5.4) |
| Reading "souping always helps" from a caption | the merged model loses a single metric (GSM* −17.5 in OLMo 2 mix E) | evaluate each ingredient and the soup per metric, not only the average |
| Copying Llama 3's extension as fixed per-stage lengths | stage plan with per-stage token counts attributed to Llama 3 | Llama 3 §3.4.2 prints only six stages and ~800B tokens |
| Trusting a secondary summary of another lab's recipe | stage placement or budget contradicts the lab's own report | check the primary report (Olmo 3 §3.6 on GLM 4.5 vs GLM-4.5 Fig. 3) |
| Skipping a long-context check after post-training | RULER at 64K–128K drops after DPO/APO or reasoning SFT | rerun RULER after every post-training stage; test a 0.9 / 0.1 merge (SmolLM3) |

## Check your understanding

1. DeepSeek-V3's extension budget is 125.8B or 122.9B tokens depending on how "32K" is read. Explain why the Olmo 3 report's 123B and the DeepSeek-V3.1 model card's "10-fold" statement cannot decide between the two readings, and which printed field would.
2. OLMo 2 describes mid-training as 5–10% of training FLOPs, but a single 7B mid-training run is 1.3% of pretraining tokens. Explain how souping ingredients and the choice of denominator produce both numbers, and which number predicts the compute of one released checkpoint.
3. MiniCPM's A-2 run beats A-1 on C-Eval by 12.6 points, while B-3 loses to B-1 on GSM8K. Explain what the B-2 run controls for, and what the GSM8K result implies about putting SFT data in the decay phase for a general-purpose model.
4. Olmo 3's extension stage uses 66% short data and still lowers the 7B Math cluster by 5.4 points. Propose two mechanisms for the drop that are consistent with Tables 11 and 13, and one ablation that would separate them.
5. Four sources report that training longer than the target length improves scores at the target length. Using the Nemotron Nano 2 explanation about concatenate-and-chunk loading and Olmo 3's packing change, explain why the sequence length can matter even for documents shorter than the target length.
6. SmolLM3 restored RULER with a 0.9 / 0.1 merge. Explain why a small weight on the mid-training checkpoint could restore a capability without undoing post-training, and which measurement would show whether post-training gains were partly undone.
7. GLM-4.5 and GLM-5 add agent trajectories during mid-training but print no token share. Design a mid-training ablation at a fixed token budget that would measure the effect of the agentic share on both agent benchmarks and general short-context benchmarks.

## Connections

- **Previous:** ch-32d — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training (data construction behind the §5 rows).
- **Next:** ch-32f — Lab: Annealing and Context Extension with a Short-Context Regression Gate (applies the Recipe starting point and the §4 gates).
- **Dependency:** ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation (RULER and HELMET limits used in §4).
- **Dependency:** ch-14a — Pretraining Recipes Side by Side: Budget, Batch, Schedule, and Stage Mixtures (pretraining denominators and schedules used in §1–§2).
- **Mechanisms:** ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL; ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression.
- **Related:** ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining (re-warming in §4 and the Recipe); ch-30c — Weight Averaging and Model Merging for Generalist Models (§6).

## Sources

- [[llama-3]], [[llama-3-recipe]] — final 40M-token anneal and Polyak averaging (§3.4.3), anneal-as-data-evaluation and GSM8k/MATH anneal result (§3.1.3), six-stage 800B-token extension and stage gate (§3.4.2), RoPE θ (§3.2).
- [[olmo-2]] — report card; its mid-training values predate verification and are superseded here by [[olmo-2-midtraining]].
- [[olmo-2-midtraining]] — verified excerpt: schedule truncation (Table 3, §4.1), Dolmino budgets and soups (§2.3, §4.5), mixes (Table 13), microanneals (Table 12), Table 9 effects, Table 14 soups.
- [[olmo-3]] — report card; superseded for stage values by [[olmo-3-base-stages]].
- [[olmo-3-base-stages]] — verified excerpt: Table 35 stage settings, Dolmino Mix (Table 5), mix evaluation (Tables 6–10), special-token result, Longmino mix (Table 11), long/short ablation (§3.6.3), Table 13 stage comparisons, merges.
- [[olmo-core-olmo3-configs]] — released midtraining and long-context scripts; LR, YaRN, batch, and duration fields and their differences from Table 35.
- [[smollm-3]] — report card; superseded for mid-training values by [[smollm3-midtraining]].
- [[smollm3-midtraining]] — blog extract: pretraining stages and decay, two 50B extension stages, reasoning mid-training, 0.9 / 0.1 merge.
- [[smollm3-training-configs]] — nanotron configs: decay steps, extension-stage θ, LR, batch, and steps; θ conflict with the blog.
- [[qwen-2.5]], [[qwen-2.5-recipe]] — 4,096 → 32,768 with ABF to 1,000,000; Qwen2.5-Turbo stages and 40/60 mix; DCA + YaRN results (§3.3, Tables 16–17).
- [[qwen-long-context-synth]] — Qwen2.5-1M stage lengths, bases, 75/25 mix, Table 2 RULER per stage, Table 6 short-context comparison, YaRN temperature (Eq. 1).
- [[prolong]], [[prolong-recipe]] — Table 9 recipe, long/short ablations (Fig. 3), RoPE base ablation (Table 18), 512K-versus-64K result (Table 7), document masking (Table 20), LR conflict.
- [[deepseek-v3]], [[deepseek-v3-recipe]] — pretraining decay and final low-LR tokens (§4.2), YaRN extension phases (§4.3).
- [[deepseek-v3.1-recipe]] — 630B and 209B extension phases described relative to V3 (model card), used as a cross-check in §1.
- [[nemotron-nano-2]] — verified excerpt: WSD decay over 3.6T, Phase LC at 524,288 tokens, synthetic long-document QA blend, Table 4 length ablation, checkpoint interpolation.
- [[glm-4-5]], [[glm-4-5-recipe]] — mid-training stages and budgets (Fig. 3, §2.3), RoPE base change and LR schedule (§2.4).
- [[glm-5]] — verified excerpt: mid-training stages 32K/128K/200K, SWE data size, long-context data, App. A learning rates.
- [[minicpm]] — verified excerpt: WSD definition (Eq. 1), 10% decay finding (Fig. 5), SFT data in the decay stage (Table 1), MiniCPM stage settings and printed decay formula.
- [[cooldown-scaling-beyond-fixed-durations]] — 20% cooldown matching cosine at 33M–1B parameters (§3.2, §5), used for decay-length replication.
- [[llama-2-long]] — definition source for adjusted base frequency (ABF).
- [[yarn]] — definition source for YaRN and its attention temperature.
