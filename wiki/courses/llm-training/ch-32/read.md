<!-- chapter: ch-32
     track: midtraining
     kind: content
     title: Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL
     deps: [ch-03, ch-13]
     sources: [[olmo-2]], [[olmo-3]], [[minicpm]], [[llama-3]], [[llama-3-recipe]], [[domain-upsampling-end-of-training]],
              [[octothinker]], [[interplay-pretraining-midtraining-rl]], [[midtraining-bridges-distributions]],
              [[mid-training-survey]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[anthropic-model-spec-midtraining]],
              [[anthropic-model-spec-midtraining-recipe]], [[openai-alignment-midtraining-generalization]], [[smollm-3]],
              [[olmo-midtraining-evidence]]
     figures: figures/pipeline-stages.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL

> **Core insight.** Mid-training is a short training phase after the main pre-training run in which the data mixture shifts toward curated, domain-specific, synthetic, and instruction-formatted data while the learning rate decays, and the resulting checkpoint is checked against an evaluation gate before post-training. Its measured effects are large on the targeted skills (OLMo 2 7B GSM8K 24.1 → 67.5 after mid-training, which averages three 50B-token runs) and small or negative on some general skills, and the size of the targeted gain falls with model scale (Llama 3: +24.0% on GSM8k validation at 8B, "negligible" at 405B). Controlled studies report that a mixed mid-training phase lowers forgetting compared with domain-only continued pretraining ([[midtraining-bridges-distributions]], 70M–1B models) and that mid-training data changes how far later RL scales ([[octothinker]], Llama 3.2 on math), but no study yet measures these effects on broad capability at frontier scale.
>
> **Guideline.** When the goal is a general-purpose model, keep a large share of high-quality general data in the mid-training mix (about 50% web in OLMo 2 Dolmino; 32.5% web and PDF in Olmo 3 Dolmino), because in Olmo 3's domain-skewed 100B mixes the targeted clusters rose by at most 6.0 points over the final mix (FIM, math-code-thinking mix) while the untargeted clusters fell by up to 31.6 points (FIM, Gen-QA mix) (Olmo 3 Table 7). When a candidate dataset must be judged, run a short anneal with a matched web-only baseline and read the change on both the targeted and the untargeted benchmarks, because single-benchmark gains at small scale may not hold at large scale (Llama 3 §3.1.3). When the model will later receive RL, evaluate mid-training choices after an RL run, not only on base-model benchmarks, because base scores did not separate the 70B- and 100B-token OctoThinker checkpoints while RL results did (§3.5). Otherwise, if no RL follows, base-model and post-SFT evaluations are the available gate (Olmo 3 integration tests).

## Why this chapter matters for a general-purpose model

The training pipeline for a modern open model has more stages than "pre-training then fine-tuning": pre-training → mid-training (annealing, domain upsampling, and in many reports long-context extension) → SFT → preference optimization → RL → evaluation. Mid-training sits at the point where the learning rate falls and the data distribution moves toward what later stages will use. Three properties make it important for breadth:

1. **Data present during the decay phase changes the base model measurably.** In MiniCPM's 0.036B WSD runs, training loss decreases faster in the decay stage than in the stable stage ([[minicpm]] §4.2, Figure 5), and adding high-quality and SFT data to MiniCPM-2.4B's decay stage raised MMLU from 44.6 to 50.9 with identical SFT afterwards (§5 Table 1). A narrow mix at this point narrows the model (§2.4).
2. **Benchmark-adjacent data enters the base model at this stage.** OLMo 2's mid-training math pool includes the GSM8K train split ([[olmo-midtraining-evidence]], OLMo 2 Table 5), and Olmo 3's decontamination found complete GSM8K leakage and over 60,000 DROP training examples in candidate mid-training sources (§3.5.4). Benchmark contamination therefore enters the base model at this stage when mid-training data is not decontaminated.
3. **It changes what SFT and RL can do.** Mid-training data changes forgetting during SFT ([[midtraining-bridges-distributions]]), whether RL from a base model scales ([[octothinker]]), and how the compute split between mid-training and RL affects in-distribution versus harder out-of-distribution tasks ([[interplay-pretraining-midtraining-rl]]).

This chapter depends on learning-rate schedules ([[ch-03]]) and domain mixing ([[ch-13]]). Continued pretraining on a new domain, where forgetting is the main risk, is [[ch-32a]]. Long-context mechanics are [[ch-32b]] and [[ch-32c]]; agentic mid-training data is [[ch-32d]].

## §1 What mid-training is: a data phase, a learning-rate phase, and a gate

### 1.1 Definitions in the sources

- **OLMo 2** calls its second base-model stage "mid-training (5–10% of training FLOPs), where we linearly decay the learning rate to zero over the remaining length of the run" and up-samples high-quality web and curated non-web sources plus synthetic math ([[olmo-2]]; arXiv:2501.00656v3 §2.3).
- **Liu et al.** define midtraining as an intermediate phase whose data is "more specialized than general pretraining data" while "maintaining a mixture with general pretraining data"; continued pretraining is the limiting case with zero general-data weight ([[midtraining-bridges-distributions]] §2.1–2.2).
- **OctoThinker** defines it by budget: a stage whose compute and tokens are "intermediate between pre-training and post-training", used for domain expansion, long-context extension, data quality, synthetic data, and "preparing for post-training" ([[octothinker]] §2).
- **The mid-training survey** groups practice into three parts: data distribution, learning-rate scheduling, and long-context extension ([[mid-training-survey]], Abstract). Its explanations (gradient noise scale, information bottleneck, curriculum learning) are proposed without new experiments (Interpretation).

In this chapter, **mid-training** means: continued next-token training from a pre-training checkpoint (1) on a changed data mixture, (2) usually during a learning-rate decay, (3) followed by an evaluation gate that decides whether the checkpoint goes to post-training. **Annealing** means training while the learning rate decreases to a low or zero value. A **stage gate** is the set of evaluations and thresholds used to accept a stage's checkpoint.

### 1.2 Budgets do not follow a common percentage

The problem: a team needs to choose how many tokens mid-training should use. The sources give shares that range from 0.00026% to 20% of tokens.

Worked example (shares derived from printed token counts; share = stage tokens ÷ cumulative tokens at the end of the stage):

| Run | Stage | Stage tokens | Cumulative tokens | Share |
|---|---|---|---|---|
| Llama 3.1 405B | final anneal | 40M | 15.6T | 40M ÷ 15.6T = 0.00026% |
| Llama 3.1 405B | long-context stage | ~800B | 15.6T | 5.1% |
| OLMo 2 7B | one mid-training run | 50B | 3.90T + 50B = 3.95T | 1.27% |
| Olmo 3 7B | Stage 2 mid-training | 100B | 6.0T (Table 13) | 1.67% |
| SmolLM3 3B | Stage 3 decay | 1.1T | 11.1T | 9.9% |
| MPT-style 7B | domain upsampling | 200B | 1T | 20% |

Sources: [[llama-3]] §3.4.2–3.4.3; [[olmo-midtraining-evidence]] (OLMo 2 §2.3, Olmo 3 Table 13, SmolLM3 blog); [[domain-upsampling-end-of-training]] §3.2.

The shares span almost five orders of magnitude (20% ÷ 0.00026% ≈ 77,000), and the stage names mean different things (Llama 3's 40M-token anneal is not comparable to SmolLM3's 1.1T-token decay). No source supports a general rule such as "mid-training should be 1–3% of pre-training". The evidence that does exist is local to one setup: in MiniCPM's 0.036B-model runs a decay over 10% of total tokens was sufficient and 2.5% "falls short" ([[minicpm]] §4.2); in Blakeney et al.'s 7B/1T run, 10–20% of training was the best trade-off ([[domain-upsampling-end-of-training]] §3.3).

The companion figure **[figures/pipeline-stages.html](figures/pipeline-stages.html)** (panel A) lets the reader switch between six documented runs and compare each stage's token count, learning-rate phase, data change, and locus.

### 1.3 The learning-rate phase

**Warmup-Stable-Decay (WSD)** is a schedule with a warmup, a long constant-LR stage, and a separate decay stage ([[minicpm]] §4.2, Eq. 1):

```
WSD(T; s) = (s / W)·η          if s < W
          = η                  if W < s < T
          = f(s − T)·η         if T < s < S,   with 0 < f(s − T) ≤ 1 decreasing
```

- s: current step. W: last warmup step. T: last stable step. S: final step.
- η: maximum learning rate. f: the decay function (MiniCPM uses an exponential form; OLMo 2 and Olmo 3 use linear decay to zero).

Why it matters for mid-training: a WSD run can fork several decay runs from one stable checkpoint, each with a different data mixture, without restarting pre-training. MiniCPM reports that a stable checkpoint can be decayed, and also trained further at the high LR and decayed later ([[minicpm]] §4.2). OctoThinker uses this form for its 200B-token constant-LR stage followed by 20B-token decay branches ([[octothinker]] §4).

The released Olmo 3 7B mid-training script shows the decay as configuration (OLMo-core@66f768b, `src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py`; full quote in [[olmo-midtraining-evidence]]):

```python
40  DEFAULT_SEQUENCE_LENGTH = 8192
41  GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens
42  MAX_TOKENS = 100_000_000_000  # 100B
43  LR = 0.00020712352850360292
80          scheduler=LinearWithWarmup(warmup=0, alpha_f=0.0),
103             load_optim_state=True,
```

The scheduler computes (`src/olmo_core/optim/scheduler.py` L384, L399 at the same commit):

```
η(s) = η₀ − (η₀ − η₀·α_f) · s / S
```

- η₀: the configured LR (2.0712 × 10⁻⁴). α_f: final LR as a fraction of η₀ (0.0 here, so the LR reaches zero).
- s: current step after warmup. S: total steps after warmup.

Worked example. Steps S = 100,000,000,000 tokens ÷ 2,097,152 tokens per batch = 47,683.7, so 47,684 steps. The released long-context script loads `Olmo-3-1025-7B/stage2/step47684`, which matches (long-context script L100). At the halfway step s = 23,842, η = 2.0712 × 10⁻⁴ × (1 − 0.5) = 1.0356 × 10⁻⁴. `load_optim_state=True` means the AdamW moment estimates carry over from pre-training, so the stage starts with no warmup.

OLMo 2 tested whether a higher pre-training LR makes the decay stage more effective. It trained 7B runs at peak LR 3 × 10⁻⁴ to 12 × 10⁻⁴ for 300B tokens and decayed each to zero over 50B or 100B tokens: OLMES averages differed by less than two points, and "a higher learning rate does make mid-training more effective, but it does so by exactly the amount that the pretraining is worse" (arXiv:2501.00656v3 §4.1, Table 8; Result, single study). The one exception noted by the authors is GSM8K, 2.8 points higher with the higher LR in a 2T-token run, which they describe as needing more study.

Implication for a general-purpose model: the decay stage is cheap relative to pre-training, and its data mixture is a decision separate from the LR schedule. Details of schedule shapes are in [[ch-03]].

## §2 Annealing on high-quality data during learning-rate decay

### 2.1 SFT-style data in the decay stage (MiniCPM)

Definition: mixing instruction or SFT-formatted data into the pre-training data during the decay stage, then running a separate SFT stage.

Problem addressed: whether high-quality labeled data helps more when introduced during decay than when used only in SFT.

Evidence ([[minicpm]] §5, Table 1; 2.4B and 1.2B models; no seeds reported; Result, single study):

| Run | Decay data | SFT tokens | C-Eval | MMLU | GSM8K | MATH | HumanEval |
|---|---|---|---|---|---|---|---|
| A-1 (2.4B) | pre-training only | 4B | 40.0 | 44.6 | 27.7 | 5.1 | 27.7 |
| A-2 (2.4B) | + high-quality and SFT data | 4B | 52.6 | 50.9 | 42.3 | 5.4 | 30.4 |
| B-2 (1.2B) | pre-training only | 12B | 41.2 | 47.9 | 34.4 | 7.3 | 43.9 |
| B-3 (1.2B) | + high-quality and SFT data | 6B | 49.1 | 49.6 | 31.8 | 10.5 | 44.5 |

Worked reading: A-2 − A-1 on MMLU = 50.9 − 44.6 = +6.3. B-3 used half the SFT tokens of B-2 and is higher on C-Eval (+7.9) and MATH (+3.2) but lower on GSM8K (−2.6). The authors conclude that specialization "should start from the decay phase" (§5). The table covers seven benchmarks, all knowledge, math, or code; general chat quality and forgetting are not measured.

MiniCPM's released recipe decays on pre-training data plus high-quality SFT data (UltraChat, SlimOrca, OssInstruct, EvolInstruct, and proprietary data), then runs a separate SFT of about 6B tokens, which the authors found "still necessary" (§6.2–6.3).

Limit found later: Olmo 3 microanneals on Tulu 3 SFT data with chat special tokens made the base model emit those tokens, with GSM8K falling from 49.43 to 0 and CruxEval from 32.89 to 18.91; a chat template written with ordinary text instead of special tokens gave 46.02 and 29.65 ([[olmo-midtraining-evidence]], Olmo 3 §3.5.4). Olmo 3 therefore removed special tokens and templates from instruction data in mid-training.

### 2.2 Domain upsampling at the end of training

Definition: **domain upsampling (DU)** removes or reduces large, less-filtered web data at the end of pre-training so that smaller domain-specific datasets and code make up more of each batch ([[domain-upsampling-end-of-training]] §3.2).

Mechanism in the reported run (7B MPT-architecture model, 1T tokens, one reported run per setting):

1. Train 0.8T tokens on the baseline mix (large-scale Common Crawl 34.35%, small-scale Common Crawl 36.70%, domain-specific 7.17%, code 21.78%; Table 2).
2. From the 0.8T checkpoint, switch to large-scale CC 0%, small-scale CC 30%, domain-specific 35%, code 35% for the last 0.2T tokens (Table 4). Small-scale CC stays in the mix "to prevent a large distribution shift" (§3.2).
3. Compare with a run that kept the baseline mix.

Result at 20% DU (Table 5): MMLU 35.69 → 42.59 (+6.90), GSM8K 14.71 → 22.97 (+8.26), HumanEval 17.23 → 23.40 (+6.17), Gauntlet Core Average 35.37 → 39.32, Language Understanding 61.52 → 60.08 (−1.44).

Duration is the variable that trades breadth against targeted skills (Table 6):

| DU share of training | 0% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|
| MMLU | 35.69 | 40.20 | **43.19** | 42.59 | 41.78 |
| GSM8K | 14.71 | 16.98 | 20.47 | 22.97 | **24.56** |
| Core Average | 35.37 | 37.63 | 38.46 | **39.32** | 38.89 |
| Language Understanding | **61.52** | 61.05 | 60.41 | 60.08 | 60.35 |

Worked example. Going from 20% to 30% DU: GSM8K +1.59 (24.56 − 22.97), MMLU −0.81 (41.78 − 42.59), Core Average −0.43. The math-and-code gain continues while the broad averages fall. The authors conclude that 10–20% was the best trade-off "for this set up" and that the DU mix "should not be used for the entire duration of training" (Figure 3 caption; Result, single study). Panel B of **[figures/pipeline-stages.html](figures/pipeline-stages.html)** lets the reader step through the five durations and see the change from 0% for each benchmark.

Conditions and limits: one model size, one reported run per setting with no seeds or variance reported, an inverse-square-root schedule whose behavior inside the DU window is not described, and Gauntlet v0.3 as the only broad measure. Whether the 10–20% range holds for other schedules or scales is an Open question.

### 2.3 Llama 3 annealing, checkpoint averaging, and souping

Llama 3.1 405B anneals "on the final 40M tokens", decaying the LR linearly to 0 at 128K context, upsampling "data sources of very high quality", and computes Polyak averages of checkpoints during annealing to produce the final pre-trained model ([[llama-3]]; arXiv:2407.21783v3 §3.4.3; row in [[llama-3-recipe]]). The report does not print the annealing mix, and it excludes the training sets of common benchmarks from annealing data "to assess the true few-shot learning capabilities and out-of-domain generalization" (§3.1.3). The order is long-context stage first, then annealing (§3.4.2–3.4.3).

**Souping** (averaging weights of runs trained on the same mix with different data orders) is part of the OLMo mid-training recipe. OLMo 2 7B averages three 50B-token runs; 13B and 32B average three 100B runs and one 300B run (arXiv:2501.00656v3 §2.3). On six candidate 50B mixes, the soup equals or exceeds the best single checkpoint on OLMES, OLMES-Gen, and MMLU, but one mix (E) falls on GSM* from 60.5 to 43.0 (Table 14). Olmo 3 32B merges two 100B mid-training runs, with +2.9 and +1.6 on the Math cluster over the two runs; the 7B used a single run because initial 7B experiments did not show similar gains ([[olmo-midtraining-evidence]], Olmo 3 §3.5.4). Merging methods in general are [[ch-30c]].

### 2.4 What goes into a mid-training mix, and the measured trade-offs

OLMo 2 Dolmino 50B mix (Table 13, share of the mix): filtered DCLM web 47.2%, decontaminated FLAN 16.6%, math 20.8%, Wikipedia/Wikibooks 7.11%, peS2o academic papers 5.85%, StackExchange Q&A 2.45%. Result (Table 9, 7B, pre-training → mid-training): MMLU 59.8 → 63.7, DROP 40.7 → 60.8, NQ 29.0 → 36.9, GSM8K 24.1 → 67.5, MMLU-Pro 27.4 → 31.0, TriviaQA 74.6 → 78.0. At 1B, WinoGrande falls from 67.8 to 66.5 while the average rises 31.9 → 43.7.

Olmo 3 Dolmino 100B mix (Table 5; category shares derived by summing rows): web and science PDFs 32.5%, code 20.0%, math 19.2%, synthetic QA 13.9%, thinking traces 8.34%, instruction data 6.1%.

Olmo 3 ran two domain-skewed 100B mixes to measure trade-offs ([[olmo-midtraining-evidence]], Olmo 3 Table 7; Result, single study):

| Mix | MC STEM | MC Non-STEM | GenQA | Math | Code | FIM |
|---|---|---|---|---|---|---|
| Gen-QA mix (no math, code, thinking) | 66.3 | 78.1 | 72.5 | 27.5 | 11.9 | 0.1 |
| Math-code-thinking mix (no QA, instruction) | 62.5 | 69.6 | 65.9 | **60.8** | **35.6** | **37.7** |
| Final Round 5 mix | 66.4 | 77.4 | 73.1 | 57.3 | 31.2 | 31.7 |

Worked reading: relative to the final mix, the math-code-thinking mix gains +3.5 Math and +4.4 Code but loses −7.8 MC Non-STEM and −7.2 GenQA. The Gen-QA mix gains +0.7 MC Non-STEM and loses −29.8 Math. The authors choose the balanced mix because skewing toward one group costs more on the other group than it gains (§3.5.4). Adding instruction and thinking-trace data at a fixed 100B total raised every base-eval cluster (average 48.8 → 50.7, Math 43.1 → 48.7; Table 10).

Implication for a general-purpose model: the evidence favors a mixed mid-training corpus in which general web and document data stay at 32.5–50% of tokens (Olmo 3, OLMo 2), and it shows that each added specialized source must be checked on the clusters it does not target.

## §3 Annealing as a data probe

### 3.1 Method

Definition: a **data probe anneal** (Llama 3 "annealing to assess data quality"; OLMo "microanneal") measures a candidate dataset by decaying the LR of a partly trained checkpoint on a mix that contains the candidate, then comparing benchmarks with a baseline anneal.

Problem addressed: Blakeney et al. note that models trained at small scale "register random accuracy" on benchmarks such as MMLU, so small-scale mixture ablations can measure noise, while full-scale ablations cost as much as pre-training ([[domain-upsampling-end-of-training]] §1).

Recipes in the sources:

1. **Llama 3 8B**: take a 50%-trained 8B model, anneal the LR linearly to 0 over 40B tokens with 30% weight on the new dataset and 70% on the default mix (arXiv:2407.21783v3 §3.1.3). No results table is printed.
2. **OLMo 2 microanneal**: mix the math source with about the same number of general web tokens, and train "as if it were an annealing run" with the LR driven linearly down; 19 microanneals used 130B tokens in total, "less than 3 full 50B annealing runs" (§4.4.2).
3. **Olmo 3 microanneal**: 5B tokens of the target dataset + 5B web tokens, compared against a 10B web-only microanneal; math alone used 80 microanneal runs over 25 candidate sources. Promising sources then enter 100B "integration tests" that are evaluated on OlmoBaseEval and again after SFT (Olmo 3 §3.5.1).

The quantity read from a probe is a difference against a matched baseline:

```
Δ_b(D) = score_b(anneal on D + web) − score_b(anneal on web only)
```

- b: one benchmark or benchmark cluster. D: the candidate dataset. Both anneals start from the same checkpoint and use the same token count and LR schedule.

Worked example (Olmo 3 Table 9, 5B microanneal with meta-reasoning and program-verifiable data vs web-only): Δ_GSM8K = 26.8 − 18.4 = +8.4; Δ_HumanEval = 19.5 − 7.9 = +11.6; Δ_MMLU = 53.7 − 55.2 = −1.5; Δ_GenQA = 52.9 − 53.7 = −0.8. A decision based only on the targeted Δ would miss the two negative values. OLMo 2 microanneal experiment 3 shows a probe detecting data format: TinyGSM with code-form answers lowered GSM* from 28.5 to 25.0, while its natural-language rewrite reached 65.5 ([[olmo-midtraining-evidence]], OLMo 2 Table 12).

Attribution by removal is the reverse probe: removing math datasets from a 10% DU window gave MMLU 29.71 and GSM8K 11.37, below both the DU run with math (43.19, 20.47) and the no-DU baseline (35.69, 14.71) ([[domain-upsampling-end-of-training]] Table 7).

### 3.2 Probe gains shrink with scale

Llama 3 annealed on the GSM8k and MATH training sets as an experiment: the pre-trained 8B model improved on the validation sets by 24.0% and 6.4%, while "the improvements on the 405B model are negligible" ([[llama-3]] §3.1.3; Result, single study). The authors interpret this as the 405B model not needing in-domain samples (Interpretation). For a probe program, the consequence is that a probe run on a small or partly trained model can overstate the value of in-domain data for the final large model. Llama 3 did not use these benchmark training sets in the released annealing data.

### 3.3 Contamination is concentrated in mid-training data

- OLMo 2 held out most of GSM8K: decisions used 200 of the 1,319 test questions (GSM*), and final numbers are reported only on the other 1,119 (arXiv:2501.00656v3 footnote 6).
- Olmo 3 decontaminates mid-training and long-context data against all splits of all OLMES benchmarks, because "memorization occurs most strongly near the end of training" per cited work (§3.5.3). Flan contained templated benchmark data, including DROP validation data; over 60,000 DROP training examples were removed. Some benchmarks (DROP, Minerva, SQuAD) scored higher with contaminated data; GSM8K scored higher after decontamination despite complete detected leakage, which the authors attribute to format mismatch (§3.5.4).

Detection methods are in [[ch-48]]. Implication: a mid-training gate that uses benchmarks overlapping with mid-training sources measures memorization as well as capability; the gate must use decontaminated data or held-out splits.

## §4 Controlled evidence: mid-training as a bridge between distributions

[[midtraining-bridges-distributions]] (arXiv v1 2025-10) is a controlled study of whether mid-training improves SFT on a target domain and whether it reduces forgetting of the pre-training distribution.

Setup: Pythia-architecture models from 70M to 1B pre-trained on C4 for 128B tokens; mid-training mixes of Starcoder, math (MAmmoTH + OpenMathInstruct), FLAN, KnowledgeQA, or DCLM mixed with C4, compared with continuing on C4 for the same tokens; SFT on GSM8K, SciQ, CodeSearchNet-Python (PyCode), or LIMA; forgetting measured as C4 validation loss after SFT; 5 seeds (§3–4).

**Mechanism stated by the authors (§2.3).** Post-training runs K gradient steps on the target loss J_T from an initialization θ₀. If J_P is L_P-smooth, J_T is L_T-smooth, and η ≤ 1/L_T:

```
ΔP(K) ≤ −η Σ_{t=0}^{K−1} ⟨∇J_P(θ_t), ∇J_T(θ_t)⟩  +  L_P·η·(J_T(θ₀) − J_T*)
```

- ΔP(K) = J_P(θ_K) − J_P(θ₀): increase in pre-training loss after K SFT steps (forgetting).
- J_P, J_T: population loss on the pre-training and target distributions. J_T*: the minimum of J_T.
- η: SFT learning rate. L_P: smoothness constant of J_P. ⟨·,·⟩: inner product of gradients.

The first term is negative when pre-training and target gradients align. The second term grows with how far θ₀ starts from the target optimum. Mid-training changes only θ₀, so a mix that lowers J_T(θ₀) lowers the bound on forgetting (Interpretation; a bound, not a measurement).

**Proximity advantage** (Eq. 6): PA(M → T) = prox(M, T) − prox(C4, T), where prox is a token-unigram similarity between corpora. Worked example with the Figure 2 similarities of pure sources: PA(StarCoder → PyCode) = 0.70 − 0.54 = 0.16; PA(Math → GSM8K) = 0.65 − 0.61 = 0.04. The code mixture moves the model's data distribution further toward its target.

**Results (1B, Table 2; Result, single study):**

| SFT target | Mid-training mix | SFT loss Δ vs C4 | C4 loss Δ vs C4 (forgetting) |
|---|---|---|---|
| PyCode | Starcoder 20% | −0.286 | −0.005 |
| GSM8K | Math 12% | −0.091 | −0.116 |
| GSM8K | Starcoder 20% | −0.015 | +0.024 |
| SciQ | Starcoder 20% | +0.015 | +0.044 |
| LIMA | FLAN 5% | −0.001 | +0.000 |

1. Benefits are domain-specific: a matched mix lowers target loss; a mismatched mix gives little benefit and can raise forgetting (Starcoder → SciQ: +0.044 C4 loss).
2. Gains correlate with proximity advantage at every size (Figure 3 panels: r = 0.466 to 0.622).
3. **Mixed mid-training beats domain-only continued pretraining** on both target loss and forgetting. 160M PyCode: Starcoder 20% mix SFT loss 2.134 and C4 loss 5.079; 100% Starcoder continued pretraining 2.219 and 5.369 (Table 3).
4. **Timing and weight interact**: for code, 80% weight is best when introduced early and worse than 10% when introduced at 105B tokens; raising the weight later (10% at 42B → 20% at 63B → 30% at 84B) did not compensate for later introduction (§6, Figure 4).

Conditions and limits: up to 1B parameters, loss metrics rather than benchmark accuracy, SFT only (no RL), and C4 loss as the only forgetting measure. The authors list scale and RL as untested (§9). Transfer of these findings to 7B+ models with multi-trillion-token pre-training is an Open question.

Implication for a general-purpose model: mixing general data into the specialized phase is supported both by this controlled study and by the OLMo mixes (Replicated in direction, with different metrics and scales). The timing result suggests that specialized domains a model must acquire in depth benefit from earlier introduction rather than a heavier late mix (Interpretation).

## §5 What mid-training changes for later RL

### 5.1 OctoThinker: mid-training data decides whether RL from a base model scales

Problem: zero RL (RL directly from a base model with verifiable rewards, as in R1-Zero; [[ch-40]]) worked on Qwen2.5-3B-Base, but Llama-3.2-3B-Base produced responses that began with "\boxed:{}" and repeated until the 4,096-token limit ([[octothinker]] §2.2).

Controlled mid-training on Llama-3.2-3B-Base (20B tokens, cosine LR peak 3e-5; then GRPO on MATH8K prompts with 16 rollouts, temperature 1.0, KL coefficient 0; §2.1, §3.1):

1. **Math web corpus quality**: MegaMath-Web-Pro gives "significant gains" after RL; FineMath-4plus gives marginal gains and responses that grow to the length limit (§3.2).
2. **QA format**: at a 9:1 web:QA ratio, short-CoT QA gives no RL gain over web only; long-CoT QA gives RL gains but unstable training with sudden score drops and length spikes (§3.3).
3. **Instruction data**: 1% instruction data (1:89:10 instruction:web:QA) improves RL with short-CoT QA and stabilizes length; with long-CoT QA it does not prevent the late decline (§3.4).
4. **Budget**: checkpoints after 20B, 70B, and 100B mid-training tokens; base-model scores of 70B and 100B were comparable, but RL results improved from 20B to 70B and from 70B to 100B (§3.5). "Increasing the mid-training budget can improve RL performance, even if such gains are not evident in base model evaluations."

The final **Stable-then-Decay** recipe trains 200B tokens at constant LR (2e-5 for 3B), then branches into three 20B-token decay runs (cosine 2e-5 → 2e-6) with 30% QA data drawn from long-CoT, short-CoT, or both (Tables 2–5). A pilot found gains plateau beyond 30% QA (§4.2.1). Results on MATH500: Llama-3.2-3B base 7.40 → OctoThinker-Long-3B-Base 25.80 (Table 7); after zero RL, Llama-3.2-3B-Zero 10.0 versus OctoThinker-Long-3B-Zero 65.2, close to Qwen2.5-3B-Zero at 66.4 (Figure 1 bar labels).

Conditions and limits: all evaluations are mathematical benchmarks; general capability after mid-training or RL is not reported; the RL figures show up to 400 steps on MATH8K; no seeds are reported. The result shows that mid-training data can make a base model RL-ready for math; it does not show the effect on breadth. OctoThinker's long branch also shows a data-format side effect: long-CoT data induced longer outputs and RL instability that needed a prompt template and a staged response-length limit (2,048 → 4,096 → 8,192 tokens) to control (§3.4). Length control in RL is [[ch-44a]].

### 5.2 The interplay study: compute split between mid-training and RL

[[interplay-pretraining-midtraining-rl]] (arXiv v1 2025-12) uses a 100M-parameter model pre-trained on 10B tokens of synthetic arithmetic-reasoning problems with 2–10 operations, then compares mid-training (next-token loss on 11–14-operation problems mixed with 20% 2–10-operation data) against GRPO on the same distribution (§2, App. A.3.3, A.10). **Extrapolative generalization** is solving problems with more operations than seen in training (OOD-edge: 11–14; OOD-hard: 15–20).

Compute equivalence (App. A.10.1, Eq. 4–8). Training on T tokens costs about 6PT FLOPs; an RL sample costs about (8 + 2γ)·P·L_total FLOPs per rollout token sequence, so

```
T_RL ≈ (4/3 + γ/3) · N · r · L_total ,   and   T_mid = (1 − β)·T ,  T_RL = β·T
```

- P: non-embedding parameters. N: number of RL samples. r = 6: rollouts per sample. L_total = 2,048: prompt plus response tokens.
- γ = 1 when a reference-model forward pass is used (giving the 5/3 factor in the main text). β: share of the total budget T given to RL.

Worked example with T = 2.1B token-equivalents (chosen here so that β = 1 equals the Full RL arm; the paper's §5 Full mid-training arm uses 1B supervised tokens) and β = 0.2 (the "Light-RL" setting): T_RL = 0.42B; N = (3/5) × 0.42 × 10⁹ ÷ (6 × 2,048) = 0.252 × 10⁹ ÷ 12,288 ≈ 20,500 samples, or about 20 steps at batch 1,024; T_mid = 1.68B tokens. With β = 1, N ≈ 102,500 samples ≈ 100 steps, which matches the "Full RL" setting of 100 steps at batch 1,024 (§5).

Results (Result, single study, synthetic tasks):
- Mid-training plus RL outperforms RL alone by +10.8% on OOD-hard tasks under fixed compute (Figure 1 caption).
- On OOD-edge tasks, full mid-training with light RL gives the best pass@1; on OOD-hard tasks, heavier RL gives the best results (Observation 3).
- The ordering depends on budget: at 4.2B token-equivalents Heavy-RL has the best OOD-hard pass@128, and at 8.4B and above Full RL has the best OOD-hard pass@128 (Observation 8, App. A.10). The statement "mid-training beats RL" holds for some metrics and budgets, not all.

Conditions and limits: 100M parameters, synthetic arithmetic grammar with a 2,200-token vocabulary, no natural-language breadth. The authors' guidance ("allocate the majority of compute to mid-training and use light RL" for similar tasks; heavier RL for exploration on complex tasks) is derived from this setting (Interpretation for real models).

## §6 Cold-start data, reasoning traces in mid-training, and distillation

**Cold start** is a small SFT stage before RL that sets the output format and readability of reasoning. Two DeepSeek-R1 SFT data sets differ in size by more than two orders of magnitude and are easy to confuse:

- DeepSeek-R1's cold start is "thousands" of long-CoT examples built from R1-Zero outputs (sampled at temperature 1.0, filtered for correct answers and readability, refined by DeepSeek-V3, verified by human annotators) ([[deepseek-r1]]; [[deepseek-r1-recipe]], v2 §3, Supp. B.3.2). The exact count is not reported.
- The ~800K set (804,745 samples: about 600k reasoning + about 200k non-reasoning) is the later rejection-sampling SFT stage, which restarts from DeepSeek-V3-Base, and it is the same set used to distill the six R1-Distill students (Supp. B.3.3 Table 5, B.4.2, B.4.3).
- Effects by stage (Table 3): cold start (Dev1) raises IF-Eval from R1-Zero's 46.6 to 71.7 and AlpacaEval 2.0 from 24.7 to 50.1, but lowers AIME 2024 from 77.9 to 59.0, which the authors attribute to the small cold-start set. The non-reasoning SFT data (Dev3) raises AlpacaEval 2.0 from 55.8 to 62.1 and Aider-Polyglot from 25.6 to 44.8 (§4). These are generality repairs after a reasoning-focused stage.

Reasoning traces now also appear in mid-training, which is a form of distillation earlier in the pipeline (teacher outputs used as next-token training data; the SFT-side view is [[ch-20]] and [[ch-35]]):

| Run | Teacher traces in mid-training | Amount | Source |
|---|---|---|---|
| Olmo 3 7B | QwQ, Llama Nemotron, OpenThoughts2, Gemini reasoning traces; synthetic meta-reasoning | 8.34% of 100B (derived) | Olmo 3 Table 5 |
| SmolLM3 3B | OpenThoughts3-1.2M and Llama-Nemotron post-training subset with R1 traces | 35B tokens × 4 epochs ≈ 140B | SmolLM3 blog |
| OctoThinker-Long | OpenR1-Math-220K and AM-DeepSeek-Distilled-40M | 30% of 20B decay tokens | OctoThinker Table 5 |

Sources: [[olmo-midtraining-evidence]], [[smollm-3]], [[octothinker]].

How this changes the role of cold start. When the base model has already seen long reasoning traces and instruction data in mid-training, SFT no longer introduces the format for the first time: Olmo 3 reports that thinking and instruction data in mid-training raised base evaluations before any post-training (Table 10), and OctoThinker's RL started from bases that already produced long-CoT outputs. No source in the library measures how large a cold-start set must be after reasoning mid-training, or whether cold start can be skipped (Open question). Two measured costs of distilled reasoning data in mid-training are known: OctoThinker's long-CoT branch needed RL stabilizers (§3.4), and SmolLM3's RULER regression was traced to the reasoning mid-training stage (§7).

## §7 Long-context extension and its placement relative to mid-training

Several reports run long-context extension next to mid-training, and its placement differs across model families. The Olmo 3 authors summarize: "Llama 3.1 models apply long-context extension prior to midtraining, Qwen 2.5 and 3 perform it afterwards, and GLM 4.5 applies extension only after supervised finetuning" (arXiv:2512.13961v2 §3.6). Llama 3's own report places the six-stage 8K → 128K extension (~800B tokens) before the 40M-token anneal and advances each stage only when "model performance on short-context evaluations has recovered completely" and needle-in-a-haystack is solved at that length (§3.4.2).

Measured short-context costs and controls:

- **Olmo 3**: Stage 3 mixes 34% long documents with 66% Dolmino short data for 50B tokens (7B) or 100B (32B), using YaRN on full-attention layers (8,192 → 65,536 tokens). In a 10B-token extension test, a 66% long / 34% short mix lowered a subset of OlmoBaseEval by 2.5 points and the 34% / 66% mix by 0.8 points (§3.6.3). Even with the short-data share, 7B Math fell from 59.8 to 54.4 and GenXL from 49.1 to 43.6 from Stage 2 to Stage 3; 32B Math fell from 69.7 to 61.4 (Table 13).
- **SmolLM3**: after long-context extension (4k → 32k → 64k, 100B tokens) came reasoning mid-training (140B tokens). Post-training RULER degraded, and the team "traced this degradation back to the reasoning mid-training stage"; a linear merge of 0.9 × preference-trained soup + 0.1 × a mid-training checkpoint with strong long-context performance recovered the base model's RULER score up to 128k (SmolLM3 blog, "Model Merging").
- **Llama 3**: SFT on short-context data only caused long-context regression; 0.1% synthetic long-context SFT data was the share that optimized both short and long benchmarks in ablations without a printed table (§4.3.4).

Worked reading of the Olmo 3 mix test: a mix with about twice the long-data share (66% vs 34%) gave about three times the drop on the OlmoBaseEval subset (2.5 ÷ 0.8 ≈ 3.1), on a 10B-token run. This is one test at one budget.

Implication for a general-purpose model: every stage after long-context extension (reasoning mid-training, SFT, preference training) needs a long-context retention check, and the long-context stage itself needs a short-context retention check. Placement choices and their evidence are developed in [[ch-32b]]; measuring effective context length is [[ch-32c]].

## §8 Alignment and specification mid-training

Two official non-paper reports test document training on alignment content as a mid-training stage. They reach different results under different conditions.

**Model Spec midtraining (MSM)** ([[anthropic-model-spec-midtraining]], Anthropic Alignment Science blog 2026-05, paper arXiv:2605.02087): LoRA training on synthetic documents that discuss a Model Spec, followed by alignment SFT (AFT). On Qwen2.5-32B-Instruct and Qwen3-32B, the average agentic-misalignment rate fell from 68% to 5% and from 54% to 7% with MSM + AFT, against 48% and 14% for a deliberative-alignment-style AFT baseline. MSM used 41M tokens (Qwen runs); MSM + AFT beat AFT alone at every AFT size from 1,250 to 80k samples ([[anthropic-model-spec-midtraining-recipe]]). The stage after MSM was SFT only; RL after MSM was not tested (paper §7).

**Alignment midtraining at o4-mini scale** ([[openai-alignment-midtraining-generalization]], OpenAI Alignment blog 2026-03): an o4-mini-sized model received 230k documents (~340M tokens) of fictional scenarios in which AIs choose aligned actions, or a matched misaligned split, with no other data mixed in, followed by SFT and RLVR without safety training. Effects appeared on QA evals close to the training distribution; on chat evals the aligned and misaligned runs scored similarly on all but two evals; on agentic evals the three runs were "not significantly different" when averaged. Misalignment effects near the training distribution "tend to disappear once the model undergoes reasoning post training."

Comparison of conditions: MSM trained already post-trained 32B checkpoints on documents about one specific spec and its reasons, followed by SFT only; the OpenAI study trained a model that had completed standard pretraining on scenario fiction, followed by SFT and RLVR. The MSM authors state that MSM "might not scale with high-compute reasoning post-training", because AFT with CoT converged to MSM + AFT on Qwen3-32B in the high-data regime ([[anthropic-model-spec-midtraining]]). Whether document mid-training on principles persists through large-scale RL is an Open question.

Implication for a general-purpose model: stage placement studies must evaluate after the full downstream pipeline, not after the stage that immediately follows mid-training, because RL can erase differences measured after SFT (Interpretation supported by the OpenAI result).

## Negative samples and negative feedback

Mid-training produces and uses negatives in two of the four senses defined in [[ch-31a]] and [[ch-43a]]; it does not use negative gradients.

1. **Negative marginal value (type 1)**: data probes identify sources that lower performance when used as positive training data. OLMo 2's code-answer TinyGSM lowered GSM* from 28.5 to 25.0 in a microanneal. The removal probe shows the opposite label: removing math sources from a DU window lowered MMLU below the no-DU baseline (29.71 vs 35.69), so those sources had positive marginal value. Current practice discards type-1 sources or rewrites them (the natural-language TinyGSM rewrite reached 65.5). The labeling signal is the probe Δ, whose noise is not reported in either source.
2. **Negative as content (type 2)**: the OpenAI misalignment-midtraining run trained with ordinary next-token loss on ~340M tokens of documents in which AIs take misaligned actions, with the reasons explained; on chat evals it scored similarly to alignment midtraining and at or above the no-midtraining baseline, and the authors hypothesize that the documents make alignment failures more salient (Interpretation). The MSM paper reports that AFT on "anti-spec" responses after MSM produced lower misalignment than anti-spec AFT alone (card, paper §5.3).
3. **Diagnostics**: for type 1, report Δ on untargeted clusters, not only on the targeted benchmark (§3.1 example: Δ_MMLU = −1.5). For type 2, evaluate both near-distribution and far-distribution alignment after the full post-training pipeline.
4. **Effect on generality**: no source measures calibration or over-refusal changes caused by negative-content documents in mid-training. Negative gradients belong to preference optimization and RL ([[ch-39]], [[ch-43a]]).

## Recipe

All rows are copied from the primary sources at the stated loci. "verified 2026-09-15" means read in this revision; rows marked 2026-09-14 are from the verified library ledgers.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3.1 405B | 405B | pretrain-decay/anneal | tokens; LR; data; averaging | final 40M tokens; LR linear to 0 at 128K context; very high-quality sources upsampled; Polyak averaging of checkpoints | arXiv:2407.21783v3 §3.4.3; [[llama-3-recipe]] | verified 2026-09-14 | no ablation reported; benchmark training sets excluded (§3.1.3) |
| Llama 3.1 405B | 405B | long-context | stages; tokens; gate | six stages 8K → 128K; ~800B tokens; advance when short-context evals recover and NIAH solved | arXiv:2407.21783v3 §3.4.2 | verified 2026-09-14 | per-stage lengths and tokens not printed |
| Llama 3 8B (experiment) | 8B | eval-gate | data probe anneal | 50%-trained 8B; LR linear to 0 over 40B tokens; 30% new dataset, 70% default mix | arXiv:2407.21783v3 §3.1.3 | verified 2026-09-14 | no numbers printed |
| Llama 3 8B / 405B (experiment) | 8B, 405B | pretrain-decay/anneal | GSM8k + MATH train-set anneal | +24.0% / +6.4% validation at 8B; "negligible" at 405B | arXiv:2407.21783v3 §3.1.3 | verified 2026-09-14 | experiment only; not in released anneal data |
| OLMo 2 7B | 7B | pretrain-stable | schedule | peak LR 3 × 10⁻⁴; 2,000 warmup steps; cosine over 5T tokens truncated after 4T | arXiv:2501.00656v3 Table 3, §4.1 | verified 2026-09-15 | §4.1 Table 8: peak LR 3–12 × 10⁻⁴ gives OLMES within 2 points after decay |
| OLMo 2 7B | 7B | mid-train | tokens; LR; merge | 3 runs × 50B tokens, different data orders; LR linear to 0; weights averaged | arXiv:2501.00656v3 §2.3, §4.5 | verified 2026-09-15 | Table 14: soup ≥ best single on OLMES, OLMES-Gen, MMLU for 6 mixes |
| OLMo 2 13B, 32B | 13B, 32B | mid-train | tokens; merge | 3 × 100B + 1 × 300B runs averaged | arXiv:2501.00656v3 §2.3 | verified 2026-09-15 | same step count as 7B at larger batch (§4.5) |
| OLMo 2 7B | 7B | mid-train | Dolmino 50B mix (share of mix) | DCLM 47.2; FLAN 16.6; math 20.8; Wiki 7.11; peS2o 5.85; StackExchange 2.45 | arXiv:2501.00656v3 Table 13 | verified 2026-09-15 | §4.3 Table 11: web + math + instruction mix best overall among 50B candidates |
| OLMo 2 7B | 7B | eval-gate | microanneal | ~50/50 source and web tokens; LR linear down; 19 runs, 130B tokens total | arXiv:2501.00656v3 §4.4.2 | verified 2026-09-15 | Table 12 |
| Olmo 3 7B | 7B | mid-train | tokens; sequence; batch | 100B; 8,192; 2²¹ tokens | OLMo-core@66f768b `OLMo-3-1025-7B-midtrain.py` L40–42 | verified 2026-09-15 | Table 6: integration rounds 1 → 5 raise OlmoBaseEval avg 49.7 → 53.1 |
| Olmo 3 7B | 7B | mid-train | optimizer; LR schedule | AdamW lr 2.0712 × 10⁻⁴, betas (0.9, 0.95), wd 0.1; `LinearWithWarmup(warmup=0, alpha_f=0.0)`; optimizer state loaded | same file L43, L72–80, L103 | verified 2026-09-15 | no ablation reported |
| Olmo 3 7B | 7B | mid-train | mix categories (share of 100B) | web + PDFs 32.5; code 20.0; math 19.2; QA 13.9; thinking traces 8.34; instruction 6.1 | arXiv:2512.13961v2 Table 5 | derived (row sums) | Table 7 skewed-mix trade-off; Table 10 instruction and thinking data |
| Olmo 3 32B | 32B | merge | mid-training runs | 2 × 100B runs with different seeds, merged | arXiv:2512.13961v2 §2 runtime, §3.5.4 | verified 2026-09-15 | Math cluster +2.9 / +1.6 vs each run |
| Olmo 3 (development) | 7B | eval-gate | microanneal | 5B target + 5B web vs 10B web-only baseline; then 100B integration tests with SFT evaluation | arXiv:2512.13961v2 §3.5.1 | verified 2026-09-15 | 80 math microanneals over 25 sources |
| Olmo 3 7B | 7B | long-context | tokens; mix; RoPE; LR | 50B; 34% long / 66% Dolmino; YaRN factor 8 from 8,192 on full-attention layers; 65,536 tokens; `LinearWithWarmup(warmup=200, alpha_f=0.0)`; CP degree 8 | arXiv:2512.13961v2 §3.6, §3.6.3; `OLMo-3-1025-7B-long-context.py` L37, L50, L81, L89 | verified 2026-09-15 | §3.6.3: 66/34 long/short −2.5 pts vs 34/66 −0.8 pts (10B test) |
| MiniCPM-2.4B | 2.4B | pretrain-stable | tokens; batch; LR | ~1T; 3.93M tokens; max LR 0.01 (WSD) | arXiv:2404.06395v3 §6.2 | verified 2026-09-15 | model wind-tunnel experiments (§6.2) |
| MiniCPM-2.4B | 2.4B | pretrain-decay/anneal | decay data; form | pre-training data + high-quality SFT data; exponential annealing with T = 5,000 steps (20B tokens) | arXiv:2404.06395v3 §6.2 | verified 2026-09-15 | §5 Table 1: A-2 vs A-1 |
| MiniCPM (0.036B experiment) | 0.036B | pretrain-decay/anneal | decay length | 10% of total tokens sufficient; 2.5% falls short | arXiv:2404.06395v3 §4.2, Figure 5 | verified 2026-09-15 | Figure 5 |
| MPT-style 7B (Blakeney et al.) | 7B | pretrain-decay/anneal | DU window; mix | final 20% (200B of 1T); large CC 0, small CC 30, domain 35, code 35 (%) | arXiv:2406.03476v1 Table 4 | verified 2026-09-15 | Table 6: 10–20% best trade-off; one reported run per setting, no seeds reported |
| OctoThinker-3B | 3B | mid-train | stable stage | 200B tokens; 50,000 steps; constant LR 2e-5; context 8,192; batch 512; wd 0.1; no warmup | arXiv:2506.20512v1 Tables 2–3 | verified 2026-09-15 | §3.5: 20B → 70B → 100B mid-training improves RL |
| OctoThinker-3B | 3B | mid-train | decay stage (per branch) | 20B tokens; 5,000 steps; cosine 2e-5 → 2e-6; 30% QA | arXiv:2506.20512v1 Tables 4–5, §4.2.1 | verified 2026-09-15 | Figure 17: QA share 10–40%, plateau beyond 30% |
| SmolLM3 3B | 3B | pretrain-decay/anneal | decay | 10T → 11.1T tokens; web 63 / code 24 / math 13 (%); linear to 0 over final 10% of steps | SmolLM3 blog 2025-07-08, "Training Configuration", "Stage 3" | verified 2026-09-15 | no ablation reported |
| SmolLM3 3B | 3B | mid-train | reasoning mid-training | 35B tokens × 4 epochs (~140B); ChatML | SmolLM3 blog, "Reasoning Mid-training" | verified 2026-09-15 | RULER regression traced to this stage; 0.9/0.1 merge recovered it |
| DeepSeek-R1 (Dev1) | 671B MoE | SFT | cold-start examples | "thousands"; exact count not given | arXiv:2501.12948v2 §3, B.3.2; [[deepseek-r1-recipe]] | verified 2026-09-14 | Table 3: Dev1 AIME 59.0 vs R1-Zero 77.9 |
| Synthetic-task 100M (interplay study) | 100M | mid-train | optimizer; data | lr 1e-4, wd 0.1, cosine min 3e-5, warmup 15%, batch 512K tokens, seq 2,048; 20% op 2–10 + 80% op 11–14 | arXiv:2512.07783v1 App. A.3.3, A.10 | verified 2026-09-15 | Figure 6, Observation 3 |
| Pythia-architecture 70M–1B (Liu et al.) | 70M–1B | mid-train | mixes | Starcoder 20%, Math 12%, FLAN 5%, KnowledgeQA 20%, DCLM 20% mixed with C4; pre-training 128B C4 tokens, cosine max 3e-4 | arXiv:2510.14865v2 §3.1, Table 2 | verified 2026-09-15 | Table 3: mix beats 100% continued pretraining |
| Qwen2.5-32B-Instruct; Qwen3-32B (MSM) | 32B | mid-train | MSM tokens; adapter | 41M tokens; LoRA rank 64, alpha 128, lr 1e-4, 1 epoch | arXiv:2605.02087v2 §4, App. B.4; [[anthropic-model-spec-midtraining-recipe]] | conflict (App. E prints 40M) | Figure 4 |
| o4-mini-sized model (OpenAI) | not reported | mid-train | alignment documents | 230k documents (~340M tokens); no other data mixed | OpenAI Alignment blog 2026-03-27, "Setup" | verified 2026-09-15 | chat and agentic evals not significantly different across runs |

**Starting point for a small general-purpose run.** For a 7B model continuing from a pre-training checkpoint, the verified Olmo 3 configuration is 100B tokens at sequence 8,192 and batch 2²¹ tokens, AdamW (0.9, 0.95), weight decay 0.1, starting LR 2.0712 × 10⁻⁴ decayed linearly to zero without warmup, with the optimizer state loaded from pre-training. For the mix, both OLMo reports keep general data as a large share (DCLM 47.2% in OLMo 2's 50B mix; web and PDFs 32.5% in Olmo 3's 100B mix) and add math, code, QA, and instruction data; Olmo 3 removed chat special tokens from instruction data. To judge a candidate dataset, the Olmo 3 microanneal (5B candidate + 5B web vs 10B web-only, from a 7B checkpoint) or the Llama 3 probe (50%-trained 8B, 40B tokens, 30% candidate) are the documented designs. Any long-context extension that follows should keep short-context data as the majority of the mix (66% in Olmo 3 7B, 50B tokens). These values were selected for 7B-scale runs by the cited teams and have no ablation at other sizes.

## Generalization lens

**(a) What increases breadth.**
- Keeping general data in the mid-training mix: mixed mid-training beats domain-only continued pretraining on target loss and on forgetting ([[midtraining-bridges-distributions]] Table 3); Olmo 3's balanced mix is within 0.7 points of the Gen-QA mix on the MC STEM, MC Non-STEM, and GenQA clusters while keeping Math at 57.3 against 60.8 for the math-skewed mix and 27.5 for the Gen-QA mix (Olmo 3 Table 7).
- Adding varied post-training-style data (instruction, thinking traces) at a fixed budget raised every Olmo 3 base cluster (Table 10).
- Averaging runs: OLMo 2 soups equal or exceed the best single run on the broad averages for all six tested mixes (Table 14).
- Limiting the length of the upsampling window: broad averages peak at 10–20% while targeted scores keep rising ([[domain-upsampling-end-of-training]] Table 6).

**(b) What causes narrowing or forgetting.**
- Domain-skewed mixes: math-code-thinking mix −7.8 MC Non-STEM and −7.2 GenQA vs the final Olmo 3 mix (Table 7).
- Long upsampling windows: 20% → 30% DU lowers MMLU and Core Average while GSM8K rises (Table 6).
- Mismatched mid-training data raises forgetting after SFT (Starcoder → SciQ: +0.044 C4 loss; Liu et al. Table 2).
- Long-context and reasoning stages lower other abilities: Olmo 3 7B Math 59.8 → 54.4 after extension (Table 13); SmolLM3 RULER regression traced to reasoning mid-training.
- Format artifacts: chat special tokens in mid-training data made the base model emit them (Olmo 3 §3.5.4); long-CoT data in mid-training caused verbose outputs and RL instability ([[octothinker]] §3.3).
- Probe results that do not transfer: GSM8k/MATH annealing gains at 8B were negligible at 405B ([[llama-3]] §3.1.3).

**(c) How to measure it for this stage.**
- Report every benchmark cluster after each candidate anneal against a matched web-only baseline, with the untargeted clusters listed first.
- Evaluate candidate mid-training mixes after a short SFT (Olmo 3 integration tests) and, when RL follows, after an RL run (OctoThinker §3.5: base scores hid the 70B → 100B difference).
- Decontaminate mid-training data against all splits of all evaluation benchmarks, or hold out a split used only for final reporting (OLMo 2 GSM*; Olmo 3 decon).
- At each stage boundary, run retention checks for the abilities the next stage does not target: short-context clusters after long-context extension, long-context (RULER) after reasoning mid-training and SFT.
- For alignment-oriented mid-training, measure after the full post-training pipeline, including RL ([[openai-alignment-midtraining-generalization]]).
- Known measurement errors: one reported run per setting with no seed count or variance in the DU, OctoThinker, and MiniCPM ablations; small evaluation sets such as GSM* (200 questions); contamination that can raise or lower scores (Olmo 3 Figure 12).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Applying a fixed percentage rule for the mid-training budget | budget choice cannot be traced to any ablation | compare with sources: shares range from 0.00026% (Llama 3 anneal) to 20% (DU); run a duration sweep as in Blakeney Table 6 |
| Reading only the targeted benchmark of a data probe | a source is accepted that lowers MMLU or GenQA | compute Δ on all clusters against a web-only anneal (Olmo 3 Table 9: Δ_MMLU = −1.5) |
| Mid-training on domain data alone | target gain with higher forgetting after SFT | include a general-data share; compare C4-style held-out loss after SFT (Liu et al. Table 3) |
| Chat special tokens in mid-training instruction data | base model emits `<|im_start|>`-style tokens; GSM8K answer parsing fails | grep base-model generations for special tokens; microanneal with and without them (Olmo 3 §3.5.4) |
| Benchmark training or validation splits in mid-training sources | gate scores rise on one benchmark but not on paraphrased or held-out versions | n-gram decontamination against all splits; hold out a final-report split (OLMo 2 GSM*) |
| Trusting a small-model probe for a large final model | in-domain data chosen at 8B gives no gain at the final scale | repeat the probe on a larger checkpoint before committing (Llama 3 §3.1.3) |
| Judging RL-readiness from base-model scores | two mid-training budgets look equal before RL and differ after RL | run a fixed short RL job on each candidate checkpoint (OctoThinker §3.5) |
| Long-context stage with too little short data | short-context clusters fall after extension | re-run short-context evals at the stage gate; compare 66/34 and 34/66 mixes (Olmo 3 §3.6.3) |
| Reasoning mid-training after long-context extension without a retention check | RULER drops after post-training | evaluate RULER after each later stage; consider a merge with the long-context checkpoint (SmolLM3) |
| Citing R1's cold start as ~800K examples | cold-start sizing plans assume hundreds of thousands of traces | read R1 Supp. B.3.2 and Table 5: cold start is "thousands"; 804,745 is rejection-sampling SFT |

## Check your understanding

1. MiniCPM's 1.2B run B-3 used half the SFT tokens of B-2 and scored higher on C-Eval and MATH but lower on GSM8K. Explain why this comparison is evidence about data placement rather than SFT budget, and what it does not show about general chat ability.
2. In Blakeney et al.'s duration ablation, GSM8K keeps rising from 20% to 30% DU while MMLU and Core Average fall. Give a mechanism for this pattern in terms of which data the model sees during the decay, and state what additional measurement would test it.
3. The forgetting bound in Liu et al. has an alignment term and an effort term. Explain how a mid-training mix could lower the bound through each term, and why a bound alone does not prove that forgetting decreased.
4. Llama 3's GSM8k/MATH annealing gains were large at 8B and negligible at 405B. Explain what this implies for using an 8B data probe to decide a 405B mid-training mix, and propose a probe design that reduces the risk.
5. OctoThinker's base-model evaluations did not separate the 70B- and 100B-token checkpoints, but RL results did. Give two causal explanations for why a base-model benchmark could miss a property that matters for RL.
6. In the interplay study, the best allocation between mid-training and RL depends on whether the target is OOD-edge or OOD-hard and on the total budget. Explain why RL would help more on the harder tasks, using the difference between pass@1 and pass@128.
7. The Anthropic MSM study and the OpenAI alignment-midtraining study report different outcomes. List the differences in data, model, and downstream stages, and explain which difference is most likely to account for the different results.
8. Olmo 3 kept 66% short data during long-context extension and still saw Math fall from 59.8 to 54.4 at 7B. Propose two reasons the short-data share did not prevent the drop, and describe a gate that would detect it before SFT.

## Connections

- **Previous:** ch-17 — Lab: Filter and Mixture Ablation with Breadth Measurement.
- **Next:** ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining.
- **Dependencies:** ch-03 — Learning-Rate Schedules, Batch Size, Initialization, and Normalization; ch-13 — Domain Mixing: DoReMi, Mixture Laws, and Validation Across Scale.
- **Related in this phase:** ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression; ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation; ch-32d — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training; ch-32e — Mid-Training, Annealing, and Context-Extension Recipes Side by Side; ch-32f — Lab: Annealing and Context Extension with a Short-Context Regression Gate.
- **Later stages that depend on this one:** ch-20 — Distillation as Data: Explanation Traces and the R1-Distill Lineage; ch-30c — Weight Averaging and Model Merging for Generalist Models; ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation; ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO; ch-48 — Contamination Detection and Its Effect on Reported Scores.

## Sources

- [[olmo-2]] — mid-training definition (5–10% of FLOPs), Dolmino mixes, souping, LR-plateau finding, microanneals, GSM* held-out protocol; values read in the primary report and collected in [[olmo-midtraining-evidence]].
- [[olmo-3]] — Dolmino 100B composition, microanneal and integration-test gate, domain trade-off tables, special-token and decontamination findings, long-context Stage 3 mix and retention costs; values and OLMo-core config quotes in [[olmo-midtraining-evidence]].
- [[olmo-midtraining-evidence]] — chapter excerpt with the OLMo 2, Olmo 3, OLMo-core, and SmolLM3 values used in §1–§3, §6–§7 and the Recipe.
- [[minicpm]] — WSD equation, 10% decay finding, SFT data in the decay stage (Table 1), MiniCPM stage settings.
- [[llama-3]] and [[llama-3-recipe]] — 40M-token anneal with Polyak averaging, long-context stage gate, anneal-as-probe design, 8B vs 405B annealing result, 0.1% long-context SFT share.
- [[domain-upsampling-end-of-training]] — domain upsampling mix, duration trade-off (Table 6), dataset attribution by removal (Table 7).
- [[octothinker]] — mid-training factors for zero-RL scalability on Llama 3.2, Stable-then-Decay recipe, budget result hidden from base evaluations.
- [[interplay-pretraining-midtraining-rl]] — compute-matched mid-training vs RL on synthetic reasoning tasks; budget formula and budget-dependent ordering.
- [[midtraining-bridges-distributions]] — controlled 70M–1B study: forgetting bound, proximity advantage, mixed mid-training vs continued pretraining, timing × weight interaction.
- [[mid-training-survey]] — taxonomy of data distribution, LR scheduling, and long-context extension (used for framing only).
- [[deepseek-r1]] and [[deepseek-r1-recipe]] — cold-start size ("thousands") versus the 804,745-sample rejection-sampling SFT set; stage-by-stage general-capability effects.
- [[smollm-3]] — SmolLM3 reasoning mid-training (140B tokens), RULER regression, and merge recovery (values read in the official blog; see the excerpt).
- [[anthropic-model-spec-midtraining]] and [[anthropic-model-spec-midtraining-recipe]] — spec document mid-training before alignment SFT; agentic-misalignment results and budgets.
- [[openai-alignment-midtraining-generalization]] — alignment and misalignment document mid-training at o4-mini scale, followed by SFT and RLVR; near- vs far-distribution results.
