<!-- chapter: ch-33
     track: sft
     kind: content
     title: Case Studies A: How Tülu 3 and Llama 3 Measured and Protected Broad Capability
     deps: [ch-31a]
     sources: [[tulu-3]], [[tulu-3-sft-mix]], [[tulu-3.1]], [[tulu-3-1]], [[rlvr-tulu3]], [[ifbench]], [[open-instruct-allenai-recipes]], [[open-instruct-allenai-recipes-recipe]], [[llama-3]], [[llama-3-recipe]], [[long-context-llama3]], [[llama-3-synthetic-pipeline]], [[dpo]], [[persona-hub]]
     figures: figures/tulu-llama-recipe.html
     revised: 2026-09 (generality revision)
-->

# Chapter 33 — Case Studies A: How Tülu 3 and Llama 3 Measured and Protected Broad Capability

> **Core insight.** Tülu 3 made its design decisions on a development evaluation suite and did not examine scores on a second, unseen suite until development was finished (§2.2). Across SFT, DPO, and RLVR, the 8B average over the five paired development benchmarks rose from 64.9 to 68.8 and the average over the five paired unseen benchmarks rose from 29.9 to 32.4, but some development gains did not transfer: removing the persona data lowered IFEval from 72.8 to 53.6 while IFEval-OOD moved from 17.6 to 18.0 ([[tulu-3]], arXiv:2411.15124v5 Tables 31–32). Llama 3 describes mechanisms for keeping one generalist model broad: code and multilingual expert models, capability-specific data pipelines, quality, difficulty, and semantic-deduplication filters, model averaging at the RM, SFT, and DPO stages, and targeted shares such as 0.1% long-context SFT data after short-only SFT regressed long-context ability; it reports no ablation numbers for these mechanisms ([[llama-3]], arXiv:2407.21783v3 §4.1.5, §4.2.3, §4.3). Neither report shows a post-training stage that improves every capability: Tülu 3 8B's safety average fell from 93.1 after SFT to 85.5 after DPO and RLVR (Tables 9, 23).
>
> **Guideline.** When a recipe is tuned against a fixed set of benchmarks, keep a disjoint unseen evaluation per skill and re-run each data ablation on it, because removing Tülu 3's persona data lowered IFEval by 19.2 points but did not lower IFEval-OOD (17.6 → 18.0, Table 32). Otherwise, report development gains as development-suite results only. When the base model already has long-context ability, mix synthetic long-context examples into SFT and gate SFT on long-context evaluations, because short-only SFT caused "significant regressions" in Llama 3 and a 0.1% share of examples was reported to balance short and long benchmarks (§4.3.4; no table printed). Otherwise, this Llama 3 result does not apply, because there is no long-context ability to preserve. When a verifiable-reward stage targets specific benchmarks (Tülu 3: GSM8K, MATH, IFEval), track the average of all development evaluations next to the targets and report every regression, because the Tülu 3 8B runs that reached GSM8K 89.4 and IFEval 84.8 "tended to perform worse in other metrics, dragging down their overall average" (§6.4). Otherwise, when a stage has no target benchmark, select on the development average, as Tülu 3 did for its preference mix (§5.2.2).

## Why this chapter matters for a general-purpose model

Both reports describe post-training on top of a pre-trained base: SFT, then preference optimization, and in Tülu 3 a third RL stage. Llama 3 also describes the pre-training and long-context stages that produced its base ([[llama-3]] §3). Chapters ch-30 to ch-31a cover SFT design choices one at a time. This chapter follows two complete pipelines and asks three measurable questions for each stage:

1. Which benchmarks went up, which went down, and by how much?
2. Did the gains on the benchmarks used for development appear on benchmarks that were never used for decisions?
3. Which capabilities did the recipe train or evaluate at all?

Tülu 3 releases its data, training code, intermediate SFT and DPO checkpoints, and hyperparameter tables (Tables 1, 11, 20, 21), so the first two questions can be answered from its tables. Llama 3 runs six rounds with human preference data and capability-specific pipelines for code, multilinguality, math, long context, tool use, and factuality (§4.3), so it shows mechanisms for breadth but reports fewer controlled numbers. Values reported by each source are quoted with their loci; values neither source reports are marked "not reported".

The interactive figure [figures/tulu-llama-recipe.html](figures/tulu-llama-recipe.html) lets the reader switch between skills and model sizes to compare development and unseen scores by stage, view each SFT ablation's effect on both suites, read every 8B development benchmark by stage including Tülu 3.1, and see the Llama 3 round structure with its loci.

## §1 Tülu 3: pipeline and evaluation design

### 1.1 Stages

Tülu 3 is a family of post-trained models built on Llama 3.1 8B, 70B, and 405B base models (arXiv:2411.15124, v1 2024-11; v5 2025-04 is quoted here). The recipe has four stages (§2.3):

1. **Prompt curation.** Public datasets and persona-conditioned synthetic prompts, decontaminated against the evaluation suite (§3).
2. **SFT** on 939,344 prompt–completion examples (§4, Table 7).
3. **Preference tuning** with length-normalized DPO on 271,409 pairs (8B) or 334,302 pairs (70B) (§5, Table 15).
4. **Reinforcement Learning with Verifiable Rewards (RLVR)**: PPO where the reward is a deterministic correctness check instead of a learned reward model, on 29,946 prompts (§6, Table 22).

### 1.2 Development and unseen suites

**Definition.** A *development suite* is the set of evaluations used to choose data, hyperparameters, and checkpoints. An *unseen suite* covers the same skills with different benchmarks and prompt formats, and its scores are not looked at during development.

**Problem addressed.** Every choice made by comparing development scores can fit the specific benchmarks rather than the skill. The size of this effect is not visible unless a second measurement exists.

**Mechanism** (§2.2, §7, Table 24):
1. Each core skill gets one or more development benchmarks: MMLU, PopQA, TruthfulQA (knowledge); BBH, DROP (reasoning); MATH, GSM8K (math); HumanEval, HumanEval+ (coding); IFEval, AlpacaEval 2 (instruction following); a six-task safety average.
2. Each skill also gets an unseen benchmark: GPQA and MMLU-Pro; AGIEval English; DeepMind Mathematics; BigCodeBench (hard subset of 148 tasks, labeled BigCodeBench-Hard in Table 33); IFEval-OOD and HREF. There is no unseen safety evaluation (§2.2, §7.3).
3. Unseen tasks use formats closer to real use: zero-shot prompts with instructions instead of few-shot dialogs (§7.3).
4. Two new unseen benchmarks were built. IFEval-OOD has 52 verifiable constraints in six categories, a set disjoint from IFEval's 25 constraints (§7.3.1, §7.4.2). HREF has 11 instruction-following tasks scored as win rate against Llama 3.1 405B Instruct; Llama 3.1 70B Instruct is the judge in 9 tasks and embedding similarity to human-written references is used for Open QA and Fact Checking. The composite procedure agreed with human judgments 69.4% of the time, versus 67% inter-human agreement (§7.3.2).

**Worked example: transfer of gains by skill** (8B, Table 31, SFT checkpoint → final checkpoint; Table 31's "Avg." row is the mean of these five benchmarks, not of the full suite):

| Skill (development → unseen) | Development change | Unseen change |
|---|---:|---:|
| Math (MATH → DeepMind Math) | 31.5 → 43.7 (+12.2) | 32.3 → 35.4 (+3.1) |
| Instruction following (IFEval → IFEval-OOD) | 72.8 → 82.4 (+9.6) | 17.6 → 24.3 (+6.7) |
| Knowledge (MMLU → GPQA) | 65.9 → 68.2 (+2.3) | 31.9 → 35.7 (+3.8) |
| Reasoning (BBH → AGIEval) | 67.9 → 66.0 (−1.9) | 56.2 → 59.3 (+3.1) |
| Coding (HumanEval → BigCodeBench) | 86.2 → 83.9 (−2.3) | 11.5 → 7.4 (−4.1) |

On DeepMind Mathematics the math gain is 3.1 points, about a quarter of the 12.2-point MATH gain. The authors state that their development process "overfit to MATH to some extent" and hypothesize a format cause: the trained models write reasoning and answers in LaTeX even where DeepMind Mathematics does not require it, which interfered with reasoning and made answer extraction fail (§7.4.1; Interpretation by the authors). The authors state that stages after SFT improved the harder unseen coding evaluation; this holds at 70B (BigCodeBench 12.2 → 21.6) but not at 8B, where BigCodeBench fell from 11.5 to 7.4 (Table 31; reading of the table in this course).

**Evidence of benchmark overfitting across models.** Tülu 3 8B scores above Llama 3.1 8B Instruct on IFEval (82.4 vs 80.6, Table 2) but below it on IFEval-OOD (24.3 vs 26.1, Table 33). The authors conclude that models "generally overfit to IFEval" (§7.4.2). The follow-up paper IFBench finds the same pattern with 58 new constraints: GPT-4.1 and Claude 3.7 Sonnet score below 50% ([[ifbench]] §1, Fig. 1). Both are Results from Ai2 with overlapping authors, so the pattern is not yet independently replicated.

**Conditions and limits.** Each benchmark pair differs in both content and prompt format, so a gap mixes skill transfer and format sensitivity. Table 31 reports one run per checkpoint without confidence intervals. The 8B SFT MMLU score is 65.9 in Tables 6 and 31 but 62.1 in Tables 9, 10, and 32; the chapter quotes each table as printed.

**Implication.** In Table 32 the final 8B SFT mix has both the highest development average (64.1) and the highest unseen average (29.9) of the five SFT models, so both averages support every data choice. Only the per-skill unseen columns show where a choice did not transfer: IFEval-OOD is higher without persona data (18.0) and without WildChat (20.8) than with the full mix (17.6). At 8B, the unseen split also shows that stages after SFT lowered BigCodeBench, which the authors' summary of Table 31 does not state.

## §2 Tülu 3 SFT: building and testing a multi-skill mixture

### 2.1 Mixture construction

The final mix has 939,344 examples from 19 sources, listed with counts in [excerpts/tulu-3-sft-mix.md](excerpts/tulu-3-sft-mix.md) (Table 7). The Ai2 blog states that 57% of the prompts come from public resources and 43% were synthesized in-house ([[tulu-3-sft-mix]]; allenai.org/blog/tulu-3-technical). The library card [[tulu-3-sft-mix]] lists 18 components; Table 7 also includes OpenMathInstruct 2 (50,000 examples), without which the counts do not sum to 939,344.

The construction procedure (§4.1.2, Fig. 3):
1. Train Llama 3.1 8B on the Tülu 2 mix and list the skills that lag state-of-the-art models.
2. For each lagging skill, build a skill-specific mixture and model, keeping the mixture with the best score on that skill alone. This approximates an upper bound for the skill.
3. Combine the skill mixtures into a preview mix, then add or remove datasets to raise lagging skills, decontaminate, and downsample large sources. Intermediate mixes 1–3 added datasets; mixes 4–5 were decontamination rounds that caused small drops (Fig. 3 caption).

Synthetic prompts are conditioned on about 250K personas from Persona Hub (§3.1.2; method in [[persona-hub]]). Math and coding problems and math solutions were generated with GPT-4o, Python solutions with claude-3-5-sonnet, and 29,980 verifiable-instruction examples cover the 25 IFEval constraint types (§3.1.2). The last point matters for §1.2: the persona IF data is built from IFEval's own taxonomy.

### 2.2 Decontamination

**Definition.** Decontamination removes training examples that overlap with evaluation examples, so that benchmark scores do not measure memorized items.

**Mechanism** (§3.2):
1. Compare prompts only (user turns), because completions are often regenerated by models.
2. A test token *matches* a training instance if both share an 8-gram containing that token.
3. A test instance has significant overlap with a training instance if more than 50% of its tokens match that same training instance.
4. A training dataset is contaminated if it overlaps more than 2% of the instances of any development or unseen evaluation.
5. Datasets contaminated with unseen evaluations are removed entirely. For development evaluations, the whole dataset is removed if that does not significantly hurt the model; otherwise only matching instances are removed.

Full-string and embedding-based matching were also tested. Embedding matching was rejected because the authors "found it difficult to distinguish mere distributional similarity from actual paraphrasing", while 8-gram matching flagged near-duplicates such as a math problem with only the numbers changed (§3.2).

**Worked example.** A 20-token test prompt shares 8-grams with one training prompt over 12 of its tokens: 12/20 = 60% > 50%, so the pair overlaps. For an evaluation of 500 instances, the dataset is contaminated if more than 10 instances (2% of 500) overlap.

**Evidence.** Removal rates were 3.5% for Evol CodeAlpaca (HumanEval), 5.4% for WildChat GPT-4 (safety), and 11.3% for NuminaMath-TIR (MATH) (Table 8). Before cleaning, Evol CodeAlpaca overlapped 70.7% of HumanEval and LMSys Chat 1M overlapped 90.3% of Do-Anything-Now; ShareGPT, LMSys Chat, DaringAnteater, and OpenAssistant 2 were not used (Table 37). The authors note that datasets of real API usage are likely to overlap existing test sets (App. B.2).

### 2.3 Ablations on both suites

Table 10 removes one data group at a time from the 8B SFT mix; Table 32 re-evaluates the same models on the unseen suite. The figure's second panel plots each change.

| Removed group | Main development change | Unseen change on the paired benchmark |
|---|---|---|
| Persona data (IF, math, code) | IFEval 72.8 → 53.6 | IFEval-OOD 17.6 → 18.0 |
| Math data | MATH 31.5 → 23.5; GSM8K 76.2 → 64.1 | DeepMind Math 32.3 → 23.3 |
| WildChat (real-user chat) | AlpacaEval 2 12.4 → 7.5; IFEval 72.8 → 70.1 | BigCodeBench 11.5 → 7.4; IFEval-OOD 17.6 → 20.8 |
| Safety data | Safety average 93.1 → 74.7 | no unseen safety evaluation |

Three different outcomes appear. Math data transferred: both MATH and DeepMind Mathematics dropped by 8–9 points without it. Persona IF data did not transfer: its 19.2-point IFEval effect has no counterpart on IFEval-OOD. The authors state that the SFT choices overfit the development evaluations for precise instruction following and "to some extent" for knowledge recall and reasoning (§7.4.1). Removing safety data left every other development score within 1.9 points, which the authors describe as safety being "orthogonal" (§4.2). Contrastive prompts such as CoCoNot were reported to prevent over-refusal of safe prompts, with no number given (§4.2).

**Amount of data.** Stratified subsamples from 5% to 100% of the mix raise the average and GSM8K, while TruthfulQA decreases as data grows (§4.2, Fig. 4; the text gives no values). The 8B SFT checkpoint's TruthfulQA is 46.8, below Tülu 2 8B SFT's 49.4 (Table 9). The SFT stage was capped at this size because other prompts were reserved for preference tuning (§4.2).

**Conditions.** Each ablation is one run at 8B. §2.4 compares these effects with the seed-to-seed range.

### 2.4 Training settings that change the result

**Loss aggregation.** With gradient accumulation or data parallelism, a per-batch mean over tokens weights examples differently depending on how the batch is split (§4.3.2):

$$L_{\text{batch}} = \frac{l_{n_1} + l_{n_2}}{n_1 + n_2} \qquad L_{\text{accum}} = \frac{1}{2}\left(\frac{l_{n_1}}{n_1} + \frac{l_{n_2}}{n_2}\right)$$

where $n_i$ is the number of non-padding tokens in sample $i$ and $l_{n_i}$ is the summed token loss of sample $i$. Worked example: $n_1 = 10$, $l_{n_1} = 20$; $n_2 = 90$, $l_{n_2} = 90$. Then $L_{\text{batch}} = 110/100 = 1.10$, while $L_{\text{accum}} = (2.0 + 1.0)/2 = 1.50$. Under $L_{\text{batch}}$ every token has weight 1/100. Under $L_{\text{accum}}$ each token of sample 1 has weight 1/(2 × 10) = 1/20 and each token of sample 2 has weight 1/(2 × 90) = 1/180, so a token of the short sample counts nine times as much as a token of the long sample. Tülu 3 uses a **sum loss** (no denominator), which weights every token equally and requires re-tuning the learning rate; a sum loss at LR 5e-6 was best in the Llama 3.0 / Tülu 2 mix sweep (§4.3.2, Fig. 5). The open-instruct command for this recipe passed `--reduce_loss sum` in June 2025, but the pinned commit 098424c has mean loss as default ([[open-instruct-allenai-recipes]], tulu3.md@8781471 L49; PR #1024).

**Seeds.** Five 8B SFT seeds average 59.8–60.1 (range 0.3); three 70B seeds average 70.0–72.6 (range 2.6). The best two-seed soup scored 60.2 (8B) and 72.5 (70B), so the best single run was released (Table 14). At 8B the seed range (0.3) is smaller than every average change in Table 10 (1.2 to 2.1 points), so those average effects are larger than seed noise at that size. At 70B the seed range (2.6) would exceed effects of that size, and no 70B ablations are reported.

**Base model.** On the same mix, GSM8K/MATH are 76.2/31.5 for Llama 3.1 8B, 91.1/53.7 for Llama 3.1 70B, 79.2/49.4 for Qwen 2.5 7B, and 86.3/56.4 for Qwen 2.5 Math 7B (Table 12). Changing the base from Llama 3.1 8B to Qwen 2.5 7B changed MATH by 17.9 points, more than removing all math SFT data (8.0 points, Table 10).

**Chat template.** Five templates trained on an intermediate mix with Llama 3.0 span 51.6–53.0 average; the Llama 3 template was lowest, and Tülu 3 used its "no \n" template (52.8) rather than the highest-scoring eos variant (53.0) to keep generation consistent with later stages (Table 13, §4.3.1). SFT settings are in the Recipe section.

## §3 Tülu 3 preference tuning

**Data pipeline** (§5.2.1, Fig. 7):
1. Select prompts used in SFT, prompts from the same sources left unused in SFT, and new prompts (UltraFeedback without TruthfulQA, IF-augmented prompts).
2. Sample four responses per prompt from a pool of 22 models; for part of the prompts, one response comes from the Tülu 3 SFT model (on-policy).
3. GPT-4o-2024-08-06 rates each response from 1 to 5 on helpfulness, instruction following, honesty, and truthfulness.
4. The highest mean rating is *chosen*; a response with a lower mean is sampled at random as *rejected*.

**Findings** (§5.3; Results, single study, 8B unless noted): more unique prompts improved DPO (Fig. 8); duplicating UltraFeedback prompts to 383K pairs performed similarly to 64K pairs and slightly lowered DROP, GSM8K, and AlpacaEval (Fig. 9); unused prompts beat reused SFT prompts, and mixing both was best (Fig. 10); adding on-policy responses improved the average (Fig. 11); judges GPT-4o, Llama 3.1 405B, and GPT-4 Turbo gave averages 57.3, 57.2, 57.0 (Table 17); persona math and code preference data did not improve their targets, so only persona IF preferences were kept (§5.3).

**Objective.** Length-normalized DPO (DPO-norm, Eq. 6):

$$\max_{\pi_\theta} \mathbb{E}_{(y_c, y_r)\sim D}\left[\log \sigma\left(\frac{\beta}{|y_c|}\log\frac{\pi_\theta(y_c\mid x)}{\pi_{\text{ref}}(y_c\mid x)} - \frac{\beta}{|y_r|}\log\frac{\pi_\theta(y_r\mid x)}{\pi_{\text{ref}}(y_r\mid x)}\right)\right]$$

where $x$ is the prompt, $y_c$ and $y_r$ the chosen and rejected responses, $|y|$ the response length in tokens, $\pi_\theta$ the policy, $\pi_{\text{ref}}$ the SFT reference, $\sigma$ the logistic function, and $\beta$ a scale. Worked example with β = 5: $|y_c| = 100$, summed chosen log-ratio +2.0; $|y_r| = 50$, summed rejected log-ratio −1.0. The margin is 5/100 × 2.0 − 5/50 × (−1.0) = 0.10 + 0.10 = 0.20, and the loss is −log σ(0.20) = 0.598. Standard DPO ([[dpo]]) with β = 0.1 on the same numbers gives margin 0.1 × (2.0 − (−1.0)) = 0.30 and loss 0.554. Dividing by $|y|$ turns each summed log-ratio into a per-token average, so a response does not gain margin only by having more tokens; the authors state that normalization "intuitively aids with mitigating the length bias" of preferences (§5.1.2; Interpretation).

**Algorithm comparison.** On UltraFeedback with an early SFT checkpoint (average 55.7), DPO-norm reached 57.3, standard DPO 55.2, SimPO 52.9 at best, and PPO 55.5 at best (Table 18). In a later controlled comparison PPO was "comparable … (albeit slightly lower)" and took about 28 hours on two nodes versus about 4 hours on one node for DPO (§5.4.1).

**Stage effect** (8B, SFT → DPO, Tülu 3.1 model card table; the SFT values match Tables 9 and 31 and the DPO values match Table 23): AlpacaEval 2 12.4 → 33.5, IFEval 72.8 → 81.1, MATH 31.5 → 42.0, TruthfulQA 46.8 → 56.1; HumanEval 86.2 → 83.9, BBH 67.9 → 65.8, safety 93.1 → 87.2.

## §4 Tülu 3 RLVR and the Tülu 3.1 update

### 4.1 Objective and data

RLVR optimizes (§6, Eq. 7–8):

$$\max_{\pi_\theta}\ \mathbb{E}_{y\sim\pi_\theta(x)}\big[v(x,y) - \beta\,\mathrm{KL}[\pi_\theta(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)]\big], \qquad v(x,y) = \begin{cases}\alpha & \text{if correct}\\ 0 & \text{otherwise}\end{cases}$$

where $v$ is the verification function, $\alpha = 10$ ("based on pilot experiments and did not tune it further"), $\beta$ the KL coefficient, and $\pi_{\text{ref}}$ the DPO starting checkpoint. A response without an end-of-sequence token receives −10.0 (Table 21).

**Worked example.** β = 0.05, and the KL term is estimated by the summed log-ratio of the response against the reference. A correct response with a summed log-ratio of 12 nats gets 10 − 0.05 × 12 = 9.4. An incorrect response with the same log-ratio gets 0 − 0.6 = −0.6. A response without an end-of-sequence token is assigned the reward value −10.0 (Table 21), so with the same log-ratio its total is −10 − 0.6 = −10.6. PPO turns these rewards into advantages with GAE and a value model and then whitens the advantages (§6.2, Table 21); a response whose advantage is negative has its likelihood lowered by the update.

**Data** (Table 22): GSM8K train (7,473, exact match of the extracted answer), MATH train (7,500, "flex" answer extraction), and 14,973 IF prompts with constraint-specific verifiers; 29,946 in total. There are no code prompts; code execution feedback is mentioned only as related work (§6, footnote 17). Other implementation details (value model initialized from a general reward model, dropout off, advantage whitening, prompts reshuffled per epoch) are quoted in [excerpts/tulu-3-rlvr.md](excerpts/tulu-3-rlvr.md). The library card [[rlvr-tulu3]] describes a binary {0,1} reward, a code verifier with unit tests, and different PPO settings; none of these appear in the paper.

### 4.2 What RLVR changed, including regressions

8B, DPO → RLVR (Table 23): GSM8K 84.3 → 87.6 (+3.3), MATH 42.0 → 43.7 (+1.7), IFEval 81.1 → 82.4 (+1.3), AlpacaEval 2 33.5 → 34.5, BBH 65.8 → 66.0; TruthfulQA 56.1 → 55.0 (−1.1), MMLU 68.7 → 68.2, safety 87.2 → 85.5 (−1.7). Llama 3.1 8B Instruct in the same harness scores GSM8K 83.4, MATH 42.5, IFEval 80.6. At 70B, GSM8K stayed at 93.5 and IFEval rose 82.6 → 83.2 (Table 23).

The largest safety declines across stages are on jailbreak tasks: Do-Anything-Now 88.3 (SFT) → 69.7 (DPO) → 62.0 (final), JailbreakTrigger 95.8 → 87.0 → 85.5, and WildJailbreakTest 86.7 → 81.1 → 78.8. HarmBench went 98.4 → 94.4 → 94.7 and WildGuardTest 99.2 → 98.9 → 98.5, while XSTest, which also scores compliance with benign prompts, rose 90.4 → 92.4 → 93.3 (Table 25).

**Checkpoint selection.** Checkpoints were evaluated every 100 steps (40 at 70B) and the 8B model with the best MATH and IFEval was released; runs reaching GSM8K 89.4% and IFEval 84.8% "tended to perform worse in other metrics" and were not chosen (§6.4). For all RL models an earlier than final checkpoint was taken (§6.3).

### 4.3 Over-optimization and the IFBench follow-up

Lower β lets the policy move further from the reference. More KL "typically results in lower average scores" (§6.2.1, Fig. 21). On constraint prompts at β = 0.01, a model asked to measure a pen with "the letter e should appear 14 times" answered with fourteen comma-separated letters e and no measurement (App. B.4, Fig. 28); the β = 0.1 model answered the question (Fig. 29).

[[ifbench]] trained IF-RLVR with GRPO from Tülu-3-8B-DPO on prompts with 1 to n constraints drawn from 29 new training constraint types (IFTrain) and IFEval's 25. IFEval rose 81.1 → 92.2 and IFBench, whose 58 constraints are disjoint from IFTrain, rose 25.2 → 44.6, but AlpacaEval 2 fell 33.5 → 21.3 and MMLU 68.7 → 66.4 (Tables 3, 6). A GPT-4.1 judge asked how well each completion answers the prompt without its constraint gave the policy before IF-RLVR an average of 7 out of 10 and the IF-RLVR policy 6.4 (§1, §5). Training on constraint rewards raised the unseen-constraint score and lowered general response quality (Result, single study). The paper's §1 prints different IFBench values for the same run (28.9 → 45.9); this chapter uses Table 6.

### 4.4 Tülu 3.1

The Llama-3.1-Tulu-3.1-8B model card states that the update is "from an improvement only in the final RL stage of training": PPO was replaced by GRPO ("no reward model"; GRPO also uses a group baseline instead of a value model) with "further hyperparameter tuning" ([[tulu-3.1]]). Compared with Tülu 3 8B the card reports average 64.8 → 66.3, MATH 43.7 → 47.8, GSM8K 87.6 → 90.0, IFEval 82.4 → 83.9, TruthfulQA 55.0 → 59.9, and safety 85.5 → 81.2.

This is not a controlled PPO-versus-GRPO comparison. Between the two runs the algorithm, the value model, β (0.05 → 0.01), learning rate (3e-7 linear → 5e-7 constant), samples per prompt (16 in GRPO, with 48 prompts per iteration), PPO update iterations K (4 → 1), mini-batches (1 → 2), the EOS penalty (−10.0 → 0.0), and the checkpoint episode all changed; the RL dataset name (`allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`) and the 2,048-token response length are the same ([[open-instruct-allenai-recipes-recipe]]; [excerpts/tulu-3.1-delta.md](excerpts/tulu-3.1-delta.md)). The library card [[tulu-3-1]] describes "Tülu 3.1" as a refresh on Llama 3.1 and OLMo 2 bases; the model card and the Tülu 3 technical blog do not support that description, so this chapter does not use it.

## §5 Llama 3: the six-round loop

### 5.1 One round

The Llama 3 report (arXiv:2407.21783, v1 2024-07, results for Llama 3.1) describes post-training that repeats the same steps in six rounds (§4.1, §4.1.6, Figure 7; details in [[llama-3-recipe]]):

1. **Collect preference data.** For each annotation prompt, two responses are sampled from two different models deployed after the previous round; annotators pick a preference on four levels ("significantly better", "better", "slightly better", "marginally better") and may edit the chosen response (§4.2.1). Prompt complexity increases as the model improves.
2. **Train a reward model** on all preference data collected so far, with the Llama 2 objective minus the margin term, keeping only pairs labeled "significantly better" or "better" (§4.1.2, §4.2.1).
3. **Rejection sampling.** For each human-annotation prompt, sample K outputs (typically 10–30) from the latest policy, usually the best checkpoint of the previous round or the best checkpoint for a capability, and keep the output the reward model scores highest (§4.2.2).
4. **SFT** of the pre-trained model on rejection-sampled, synthetic, and human-curated data, with loss on target tokens only; the largest models use LR 10⁻⁵ for 8.5K–9K steps (§4.1.3).
5. **DPO** on the most recent preference batches ("we primarily use the most recent batches of preference data"), so that the pairs are close to the current policy's distribution (§4.1.4).
6. **Model averaging** of runs with different data or hyperparameters at the RM, SFT, and DPO stages (§4.1.5).

The final SFT mix is 52.66% general English, 14.89% code, 3.01% multilingual, 8.14% exam-like, 21.19% reasoning and tools, and 0.11% long context, as a share of examples; it is adjusted each round "to tune performance across a wide range of benchmarks", and the final mix "epochs multiple times on some high quality sources and downsamples others" (Table 7, §4.2.2). Per-round data shares and per-round evaluation results are not reported. The April 2024 Meta blog for Llama 3 8B and 70B described post-training as SFT, rejection sampling, PPO, and DPO ([[llama-3-synthetic-pipeline]], "Instruction fine-tuning"); the July 2024 report states that PPO was explored and DPO "required less compute … and performed better, especially on instruction following benchmarks like IFEval", with no numbers (§4.1.4).

### 5.2 DPO modifications

Llama 3 uses standard DPO with LR 10⁻⁵ and β = 0.1 and two changes (§4.1.4; quotes in [excerpts/llama-3-dpo-nll.md](excerpts/llama-3-dpo-nll.md)):
1. **Formatting tokens masked.** Header and termination tokens are removed from the loss in both responses. Without masking the authors observed tail repetition or abrupt termination tokens and hypothesize that tokens shared by both responses receive opposing updates.
2. **NLL term on chosen responses** with coefficient 0.2: $L = L_{\text{DPO}} + 0.2 \cdot \mathrm{NLL}(y_c \mid x)$, "similar to Pang et al. (2024)", to keep formatting and prevent the chosen log-probability from decreasing. Whether the NLL is normalized per token is not reported.

## §6 Llama 3: mechanisms for keeping one generalist broad

### 6.1 Capability experts and pipelines

Llama 3 trains separate *expert* models and uses them to produce data for the single released model:
- A **code expert** continued pre-training on 1T tokens with more than 85% code and long-context fine-tuning to 16K tokens; it is used to collect human code annotations and is the rejection-sampling policy for coding prompts (§4.3.1).
- A **multilingual expert** continued pre-training on 90% multilingual tokens and is used to collect non-English annotations until pre-training of the main model was complete (§4.3.2).

Capability pipelines add data with their own correctness checks: about 1M execution-feedback code dialogs, in which about 20% of solutions were initially incorrect and self-corrected after parser, linter, and unit-test feedback; about 1.2M backtranslation dialogs; and translation of code to less common languages (§4.3.1). Two reported failures concern pipeline-generated data. In the code pipeline, training 405B "on its own generated data is not helpful (and can even degrade performance)", while 8B and 70B improved on data from a larger model; execution feedback was introduced for this reason (§4.3.1). A strict model-as-judge code filter first lowered benchmark scores because it "disproportionately removed examples with challenging prompts"; revising the most challenging responses until they passed the judge gave the best downstream performance, with no numbers printed (§4.3.1). Tool-use data skipped rejection sampling because it gave no gains, and easy queries answered without tools were added so that the model does not call tools unnecessarily (§4.3.5).

### 6.2 Quality, difficulty, and semantic deduplication filters

SFT data, most of which is model-generated, is pruned with four model-based techniques (§4.2.3):
1. **Topic classification** with a fine-tuned Llama 3 8B classifier (coarse and fine buckets).
2. **Quality**: keep an example if it is in the top quartile of reward-model scores *or* gets the maximum Llama 3 rating; the two signals disagree often and their union gave the best recall on an internal test set.
3. **Difficulty**: Instag intention tags from Llama 3 70B (more intentions means more complex) and a three-point Llama 3 difficulty rating.
4. **Semantic deduplication**: cluster dialogs with RoBERTa, sort each cluster by quality × difficulty, and greedily keep an example only if its maximum cosine similarity to examples seen so far is below a threshold (value not reported).

**Worked example** (threshold 0.9 chosen for illustration). One cluster sorted by quality × difficulty: A (2.7), B (2.0), C (1.8), D (1.2). Similarities: A–B 0.95, A–C 0.60, B–C 0.70, A–D 0.50, B–D 0.40, C–D 0.92. Keep A. B has maximum similarity 0.95 ≥ 0.9, drop. C has maximum 0.70, keep. D has maximum 0.92 (to C), drop. The cluster keeps A and C. Each near-duplicate pair (A–B and C–D) keeps its higher-scoring member, which is why the sort key decides which duplicate survives.

Rule-based cleaning also removed patterns such as excessive emojis and balanced the share of overused apologetic phrases (§4.2.3). No ablation numbers are reported for any filter.

### 6.3 Safety data balance by model size

Borderline prompts that resemble adversarial ones but deserve helpful answers are added to lower the false refusal rate. Figure 18 shows that 8B needs a higher proportion of safety data relative to helpfulness data than 70B for comparable safety, and the authors tailored mixes per size (§5.4.3). For safety DPO, response pairs that are "nearly orthogonal in an embedding space" were reported as particularly effective (§5.4.3). Llama Guard 3 is a separate classifier for system-level filtering, not part of the model's training (§5.4.7).

### 6.4 How Llama 3 measured breadth and contamination

- **Decontamination.** Post-training data was decontaminated by exact prompt match (§5.2). For pre-training, an 8-gram analysis estimates contamination and its effect: HellaSwag 85% flagged with an estimated +14.8 at 8B; NaturalQuestions 52% flagged with estimated +1.6/+0.9/+0.8 at 8B/70B/405B; no estimate possible for HumanEval, MBPP, MMLU, or MMLU-Pro (Table 15).
- **Held-out human evaluation.** "Modeling teams did not have access to our human-evaluation prompts" (§5.3); the report's conclusion adds that pre-training data was processed by a separate team incentivized to prevent contamination (§10). The April 2024 blog describes a 1,800-prompt set over 12 use cases withheld from modeling teams ([[llama-3-synthetic-pipeline]]).
- **Robustness and adversarial pairs.** Pre-trained models were tested on MMLU label variants, few-shot label distributions, answer-order permutations, and prompt formats, and the authors describe them as "very robust" to these changes, in particular 405B (§5.1.2, Figs. 13–14; no post-trained robustness test is reported). On adversarial question answering and GSM-Plus, both pre-trained and post-trained models score "substantially lower" than on the paired standard benchmarks, while PAWS shows no such gap (§5.1.3, Fig. 15).

## §7 Long-context post-training in both recipes

**Llama 3 base.** Context grows from 8K to 128K in six stages over approximately 800B tokens; a stage advances only when short-context evaluations have "recovered completely" and needle-in-a-haystack is solved at that length (§3.4.2). RoPE θ is raised to 500,000 as an architecture setting for all of pre-training (§3.2, Table 3), not rescaled per long-context stage. Per-stage lengths and token counts are not reported. The library card [[long-context-llama3]] lists per-stage token budgets and a RoPE base "rescaled from 10K to 500K" progressively across stages; neither appears in the report.

**Llama 3 SFT** (§4.3.4):
1. Short-context-only SFT "resulted in significant regressions in long-context capabilities from pre-training".
2. Synthetic data from earlier Llama 3 models covers three uses: QA pairs generated from random 8K-token chunks of long pre-training documents, with the full document as context; hierarchical summarization of 8K chunks followed by QA requiring global understanding; and repository code reasoning, where a Python file imported by at least five other files is removed and the model must name the dependents and regenerate the file.
3. Examples are bucketed at 16K, 32K, 64K, and 128K tokens.
4. Mixing 0.1% of such data with short data "optimizes the performance across both short-context and long-context benchmarks" in the authors' ablations, which are not printed.

**Worked example: example share versus token share.** Long-context examples are 0.11% of SFT examples. They average 38,135.6 tokens, of which 740.5 are in the final response; all examples average 846.1 tokens with 310.4 in the final response (Table 7). The share of all tokens processed is 0.0011 × 38,135.6 / 846.1 ≈ 4.96% (derived), about 45 times the example share. If the loss covers only the final response, the share of loss tokens is 0.0011 × 740.5 / 310.4 ≈ 0.26% (derived). The report masks prompt tokens (§4.1.3) but does not state whether earlier assistant turns in the context are trained, so this second number holds only under that assumption. Both numbers use the rounded 0.11% share printed in Table 7. The first number describes how much of the processed context is long; the second describes how much of the loss comes from long-context targets.

**Llama 3 DPO.** Short-context-only DPO data "did not negatively impact long-context performance as long as the SFT model is high quality in long context tasks"; the authors suspect this is because DPO uses fewer optimizer steps than SFT (§4.3.4; Interpretation). Reported post-trained results include multi-needle retrieval 98.8/97.5/98.1 and InfiniteBench En.QA 27.1/36.7/30.5 for 8B/70B/405B (Table 21). Methods and the gap between claimed and effective context length are covered in ch-32b and ch-32c.

**Tülu 3.** SFT examples are at most 4,096 tokens and DPO examples 2,048 (Tables 11, 20); the mixture averages 2.4 turns and most samples are under 2,048 tokens (§8.3). The evaluation suite has no long-context or multi-turn benchmark (Table 24), and the authors list both as future work (§8.3). The Llama 3.1 8B and 70B base models that Tülu 3 fine-tunes are marked as long-context models in Llama 3 Table 1. Tülu 3 therefore ran short-context SFT on a long-context base without a gate that would detect the regression Llama 3 reports; whether Tülu 3 lost long-context ability is not reported by either source.

## §8 What each recipe gated on and what it did not measure

| Capability | Tülu 3 trained / development eval / unseen eval | Llama 3 trained / reported eval |
|---|---|---|
| Knowledge, reasoning, math, code | yes / yes / yes | yes / yes |
| Precise instruction following | yes / IFEval / IFEval-OOD | yes / IFEval |
| Safety and over-refusal | yes / six-task average incl. XSTest / no | yes / violation and false refusal rates per size and language |
| Multilingual | Aya data only / no / no | expert + data / MGSM, multilingual MMLU |
| Long context | no / no / no | 0.11% of SFT / NIH, ZeroSCROLLS, InfiniteBench |
| Tool use | no / no / no | yes / Nexus, API-Bank, API-Bench, BFCL |
| Multi-turn | 2.4 turns average / no / no | 4.7 turns average in SFT / multi-turn human evaluation |

Sources: [[tulu-3]] Table 24, §8.3; [[llama-3]] Table 7, §5.2–5.4.

**Gates.** Tülu 3 chose SFT mixes, preference mixes, and RL checkpoints on the development average plus target skills (§4.1.2, §5.2.2, §6.4) and used the unseen suite only after the fact. Llama 3 gated the long-context pre-training stages on short-context recovery and NIH (§3.4.2), tuned the SFT mix each round against "a wide range of benchmarks" (§4.2.2), and tuned safety mixes on violation and false refusal rates (§5.4.3).

**Not measured or not reported.** Tülu 3 has no unseen safety evaluation, no long-context, tool-use, or multilingual evaluation, and one seed per ablation. Llama 3 reports no development/unseen split for its benchmark decisions, no per-round results, no numbers for its long-context share ablation or its PPO comparison, and no filter ablations. Tülu 3 tried rejection sampling and found "minimal" gains for the compute (§8.2), while Llama 3 uses it every round; neither report compares the two at matched compute.

## Negative samples and negative feedback

This section applies the four meanings of "negative" from ch-31a to both pipelines.

**Where negatives come from, and what is done with them.**

| Negative signal | Label source | Meaning used | Locus |
|---|---|---|---|
| Datasets removed for contamination or lower quality; K − 1 discarded rejection samples | 8-gram matcher; reward model | (1) negative marginal value, discarded | Tülu 3 §3.2, §4.1.2; Llama 3 §4.2.2 |
| IF-augmented pairs whose chosen response failed its constraint (more than 66k generated, about 26k passed) | constraint verifiers | (1) tested as a filter; the verified subset lowered the average slightly, so the unverified set was used with Persona IF | Tülu 3 §5.3 |
| Noncompliance (CoCoNot) and refusal targets for questions whose sampled answers are consistently informative and incorrect | noncompliance taxonomy; Llama 3 as judge against the source snippet | (2) negative as content | Tülu 3 §3.1.2, §4.2; Llama 3 §4.3.6 |
| Failing code and wrong math traces rewritten after execution or error feedback | parser, linter, unit tests; answer check | (2) corrected content; whether the failed attempt stays in the training dialog is not reported | Llama 3 §4.3.1, §4.3.3 |
| DPO rejected responses | GPT-4o ratings (Tülu 3); humans, "significantly better" or "better" only (Llama 3) | (4) gradient | Tülu 3 §5.2.1; Llama 3 §4.2.1 |
| RLVR responses scored 0 or −10.0 | exact match, constraint verifiers | (4) gradient through a lower advantage | Tülu 3 Eq. 8, Table 21 |

Neither report measures false-negative rates of its verifiers or judges. Open-instruct documents that the IF verifier was later changed and that reproducing Tülu 3 requires the old verifier ([[open-instruct-allenai-recipes]], tulu3.md L272–276).

**Mechanism.** For a softmax over logits $z$, $\partial \log p_y / \partial z_j = \mathbf{1}[j = y] - p_j$, where $p_j$ is the probability of token $j$ and $y$ the token being pushed down. Decreasing $\log p_y$ moves $z_y$ down and raises each other logit in proportion to $p_j$. Worked example: $p = (p_y, p_2, p_3) = (0.02, 0.90, 0.08)$ gives gradient $(0.98, -0.90, -0.08)$; a descent step of size η on $\log p_y$ lowers $z_y$ by 0.98η and raises $z_2$ by 0.90η and $z_3$ by 0.08η. For small η the probability change is $dp_j = p_j\,(dz_j - \sum_k p_k\,dz_k)$, where $dz_j$ is the change of logit $j$, with $\sum_k p_k\,dz_k = 0.7968η$: $dp_y = -0.0355η$, $dp_2 = +0.0929η$, $dp_3 = -0.0573η$ (derived). The most likely token receives all of the mass removed from $y$ and also takes mass from the third token. The rejected term of DPO applies this at every token of $y_r$, which is the Llama 3 authors' hypothesis for why tokens shared by chosen and rejected responses receive conflicting updates (§4.1.4). The same mechanism allows the chosen log-probability to fall while the margin grows; neither report logs chosen log-probabilities, and Llama 3 adds its NLL term to prevent that decrease (§4.1.4).

**Controls used.** Llama 3: formatting tokens masked, NLL coefficient 0.2 on chosen responses, low-margin pairs discarded, DPO data drawn from the most recent batches. Tülu 3: length normalization, on-policy responses in the preference pool, a KL penalty in RLVR (β = 0.05 at 8B), and a −10.0 penalty for truncated responses.

**Evidence and size of effect.** Neither report isolates the contribution of rejected responses, for example by comparing DPO with SFT on the chosen responses of the same pairs, so no share of the gain can be attributed to negatives. The stage-level results include regressions: Tülu 3 8B DPO lowered safety 93.1 → 87.2 and HumanEval 86.2 → 83.9 while raising AlpacaEval 2 and IFEval (§3), and RLVR at β = 0.01 on constraint prompts produced content-free outputs (App. B.4).

**Diagnostics.** Log chosen and rejected log-probabilities separately; track KL, response length, and verifiable reward together (Tülu 3 Figs. 19, 23); read sampled outputs at each β of the sweep; break the safety average into jailbreak and over-refusal tasks (Table 25); report pass@1 on unseen constraint benchmarks.

**Effect on generality.** The DPO and RLVR stages that pushed down rejected or failed responses coincided with lower jailbreak robustness (Do-Anything-Now 88.3 → 62.0) and higher XSTest (90.4 → 93.3) at 8B (Table 25). These are correlations across whole stages; the causal share of the rejected term is an Open question for these two recipes (derivations in ch-39 and ch-43a).

## Recipe

Rows quote the settings relevant to this chapter. "v5" is arXiv:2411.15124v5; "v3" is arXiv:2407.21783v3; "t3" is open-instruct `docs/tulu3.md` at commit 098424c. Canonical ledgers: [[open-instruct-allenai-recipes-recipe]], [[llama-3-recipe]]. The [[tulu-3]] card itself does not yet carry a verified ledger and lists values that conflict with the paper (for example 10,000,000 RLVR episodes).

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | data | 939,344 examples from 19 sources (`allenai/tulu-3-sft-mixture`) | v5 Table 7 | verified 2026-09-15 | Fig. 3 intermediate mixes; Table 10 removals; Fig. 4 subsampling |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak LR; schedule; warmup ratio | 5 × 10⁻⁶; linear; 0.03 | v5 Table 11 | verified 2026-09-15 | Fig. 5: sum loss at 5e-6 best on Llama 3.0 + Tülu 2 mix, not the final mix |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | epochs; effective batch; max length | 2; 128 sequences; 4,096 tokens | v5 Table 11 | verified 2026-09-15 | Fig. 6: 2 best among 2–7 epochs (Tülu 2 mix); batch: no ablation reported |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | loss aggregation | sum loss (paper; `--reduce_loss sum` in the June 2025 doc command); flag absent and mean loss default at t3@098424c | v5 §4.3.2; t3@8781471 L49; PR #1024 | conflict | Fig. 5 (sum vs mean, Tülu 2 mix); the paper and the 2025 command agree on sum loss, the pinned command does not |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | compute; selection rule | 32 GPUs, 6 h; best single seed of 5 | v5 §4.3, Table 14 | verified 2026-09-15 | Table 14: seeds 59.8–60.1, soup 60.2 |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | peak LR; epochs; batch; length; compute | 2 × 10⁻⁶; 2; 128; 4,096; 64 GPUs, 50 h | v5 Table 11, §4.3 | verified 2026-09-15 | "hyperparameter search", values not printed |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | loss; β; LR; schedule; warmup; epochs | DPO-norm (Eq. 6); 5; 5 × 10⁻⁷; linear; 0.1; 1 | v5 Table 20 | verified 2026-09-15 | Table 18: DPO-norm 57.3 vs SFT base 55.7, PPO 55.5, DPO 55.2 (UltraFeedback, early SFT) |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | pairs; batch; max length; judge | 271,409; 128; 2,048; GPT-4o-2024-08-06 | v5 Tables 15, 20; §5.2.1 | verified 2026-09-15 | Table 16 mixing runs; Table 17 judges |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | compute | 8 H100 GPUs, 10 hours | v5 §5.4.1 | verified 2026-09-15 | n/a |
| Llama-3.1-Tulu-3-70B-DPO | 70B | preference | LR; pairs | 2 × 10⁻⁷; 334,302 | v5 Tables 19, 15 | verified 2026-09-15 | Table 19: 74.35 at 2.0e-7 vs 71.14 at 5.0e-7 (Mix 2) |
| Llama-3.1-Tulu-3-8B-RM | 8B | reward-model | LR; batch; epochs; length; compute | 3 × 10⁻⁶; 256; 1; 2,048; 8 H100 GPUs, 9 hours | v5 Table 36, §6.3 | verified 2026-09-15 | no ablation reported for these values; Fig. 21: initializing the PPO value model from an RM worked best; the final RM was trained on the 8B preference mix from Tülu 3 SFT (§6.4) |
| Llama-3.1-Tulu-3-8B | 8B | RL | algorithm; reward; no-EOS reward | PPO; α = 10 if verified, else 0; −10.0 | v5 Eq. 8, Table 21 | verified 2026-09-15 | α from pilot experiments, not tuned |
| Llama-3.1-Tulu-3-8B | 8B | RL | prompts | 29,946 (GSM8K 7,473; MATH 7,500; IF 14,973) | v5 Table 22 | verified 2026-09-15 | Fig. 19 single-task runs |
| Llama-3.1-Tulu-3-8B | 8B | RL | compute | 8 GPUs, 65 hours | v5 §6.3 | verified 2026-09-15 | n/a |
| Llama-3.1-Tulu-3-8B | 8B | RL | LR; β; warmup; batch; K; response length | 3 × 10⁻⁷; 0.05; 0.0; 224; 4; 2,048 | v5 Table 21 and caption | verified 2026-09-15 | β sweep [0.1, 0.05, 0.03, 0.01] (Figs. 19, 21); §6.4 tested β up to 0.15 |
| Llama-3.1-Tulu-3-8B | 8B | RL | total episodes | 100,000 (paper); 10,000,000 planned in command | v5 Table 21; t3 L329 | conflict | released checkpoint "earlier than final" (§6.3); its episode is not reported |
| Llama-3.1-Tulu-3-8B | 8B | eval-gate | checkpoint rule | every 100 steps; best MATH and IFEval | v5 §6.4 | verified 2026-09-15 | GSM8K 89.4 / IFEval 84.8 runs rejected for other metrics |
| Llama-3.1-Tulu-3-70B | 70B | RL | LR; batch; episodes; β; warmup | 1 × 10⁻⁷; 640; 400,000; β 0.7 (text) vs 0.07 (caption and t3 command); warmup 0.1 (text and t3 command) vs 0.07 (caption) | v5 §6.4, Table 21 caption; t3 L375–383 | conflict | none reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | algorithm; LR; β; prompts × samples; released checkpoint | GRPO; 5 × 10⁻⁷ constant; 0.01; 48 × 16 = 768; episode 1,474,560 (step 1920) | model card "Hyperparamters" (sic), "Learning curves" | verified 2026-09-15 | "further hyperparameter tuning"; no ablation table |
| Llama 3.1 405B | 405B | preference | rounds | 6 | v3 §4.1.6 | verified 2026-09-15 | none reported |
| Llama 3.1 405B | 405B | SFT | rejection sampling | K typically 10–30 per human-annotation prompt; RM selects best | v3 §4.2.2 | verified 2026-09-15 | none reported |
| Llama 3.1 "largest models" | not scoped | SFT | LR; steps; loss | 10⁻⁵; 8.5K–9K steps; target tokens only | v3 §4.1.3 | verified 2026-09-15 | "work well across different rounds and data mixes" |
| Llama 3.1 | not scoped | SFT | batch; epochs; schedule | not reported | v3 §4.1.3, §4.2.2, Table 7 checked | not reported | n/a |
| Llama 3.1 | not scoped | SFT | long-context share | 0.1% synthetic; buckets 16K/32K/64K/128K; 0.11% of examples in Table 7 | v3 §4.3.4, Table 7 | verified 2026-09-15 | "careful ablations", no table |
| Llama 3.1 405B | 405B | preference | DPO LR; β; NLL coefficient; masking | 10⁻⁵; 0.1; 0.2 on chosen; header and termination tokens | v3 §4.1.4 | verified 2026-09-15 | PPO comparison in prose only |
| Llama 3.1 405B | 405B | merge | model averaging | runs with different data or hyperparameters, at RM, SFT, DPO; weights not reported | v3 §4.1.5 | verified 2026-09-15 | no ablation reported |

**Starting point for a small general-purpose run.** For a Llama 3.1 8B base and the 939,344-example Tülu 3 SFT mixture, the verified 8B rows give SFT at LR 5 × 10⁻⁶ (linear, warmup 0.03) for 2 epochs at 128 sequences of up to 4,096 tokens with a sum loss, then DPO-norm with β = 5 at LR 5 × 10⁻⁷ (linear, warmup 0.1) for 1 epoch at 128 pairs of up to 2,048 tokens, then PPO-based RLVR with α = 10, β = 0.05, LR 3 × 10⁻⁷, batch 224, K = 4, and 2,048-token responses on 29,946 verifiable prompts. The source ran these at 8B on 32 GPUs for 6 hours (SFT), 8 H100 GPUs for 10 hours (DPO), and 8 GPUs for 65 hours (RL), plus 8 H100 GPUs for 9 hours for the reward model that initializes the value model (§4.3, §5.4.1, §6.3). The sum loss comes from a `conflict` row: the paper used it, and the pinned open-instruct command does not. If the base model has long context, add long-context SFT data; Llama 3 reports 0.1% by examples without stating the model size (§4.3.4). None of these values was shown to transfer to other base models; the OLMo 2 report states that OLMo 2 bases needed higher learning rates than this recipe ([[open-instruct-allenai-recipes-recipe]]; ch-34).

## Generalization lens

**(a) What increased breadth.**
- Diverse real-user chat: removing WildChat lowered the 8B five-benchmark unseen average 29.9 → 28.8 and BigCodeBench 11.5 → 7.4 as well as AlpacaEval 2 12.4 → 7.5 ([[tulu-3]] Tables 10, 32).
- Skill data that matches the skill rather than one benchmark's format: math data moved MATH and DeepMind Mathematics together (Table 32).
- Unique prompts rather than duplicated prompts, and on-policy responses in preference data (Tülu 3 Figs. 8–11).
- IF-RLVR on constraints from IFTrain (29 types) and IFEval (25 types), with 1 to n constraints per prompt, raised the disjoint-constraint benchmark IFBench 25.2 → 44.6 ([[ifbench]] Table 6); the paper's ablations on constraints per prompt and variable ranges are in [[ifbench]] §4.1–4.3.
- Capability experts, execution feedback, and difficulty-aware filtering that kept hard prompts ([[llama-3]] §4.3.1, §4.2.3; mechanisms described without ablation numbers); a long-context SFT share that preserved pre-trained long-context ability (§4.3.4; reported without a table).

**(b) What caused narrowing or forgetting.**
- Data built from a benchmark's taxonomy: removing the persona data (whose IF part follows IFEval's 25 constraint types) changed IFEval by −19.2 and IFEval-OOD by +0.4 (Table 32).
- More SFT data lowered TruthfulQA (Tülu 3 Fig. 4).
- Preference and RL stages lowered jailbreak robustness and the safety average at 8B (Table 25); the Tülu 3.1 RL run (GRPO with other changed settings) ended at safety 81.2 versus 85.5 for Tülu 3 8B ([[tulu-3.1]]).
- Low KL in constraint RLVR produced content-free answers (App. B.4); constraint-only rewards lowered AlpacaEval 2 33.5 → 21.3 ([[ifbench]] Table 3).
- Short-only SFT regressed long-context ability in Llama 3 (§4.3.4); a strict quality filter removed hard prompts and lowered scores (§4.3.1); 405B trained on its own generated code data did not improve (§4.3.1).

**(c) How to measure it at this stage.**
- A per-skill development/unseen split with different formats, evaluated on every ablation ([[tulu-3]] §7.4.1).
- Constraint-disjoint instruction-following benchmarks (IFEval-OOD, IFBench) and LLM-judge quality with constraints removed ([[ifbench]] §5).
- Safety split into jailbreak and over-refusal tasks; violation and false refusal rates per model size (Tülu 3 Table 25; Llama 3 Fig. 18).
- 8-gram token-overlap decontamination against both suites, and estimated performance gain from contamination ([[tulu-3]] §3.2; [[llama-3]] Table 15).
- Long-context evaluations inside the SFT gate whenever the base supports long context (Llama 3 §4.3.4, Table 21).
- Seed variation as the noise floor for ablations (Tülu 3 Table 14). Known errors: format sensitivity in answer extraction (§7.4.1), benchmark overlap in real-user data (App. B.2), and internal table inconsistencies (MMLU 65.9 vs 62.1).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Choosing data or checkpoints on development scores only | development average rises while unseen scores stay flat or fall | re-run each ablation on a disjoint unseen suite (Tülu 3 Table 32 pattern) |
| Building IF data from the evaluated constraint taxonomy | IFEval rises far more than IFEval-OOD or IFBench | report both; hold out constraint types from training |
| Treating Tülu 3.1 as a PPO-vs-GRPO ablation | the +1.5 average is attributed to GRPO alone | list every changed setting (β, LR, schedule, samples per prompt, K, mini-batches, value model, EOS penalty, released episode) |
| Copying `total_episodes 10,000,000` as the Tülu 3 run length | run length and schedule differ from the paper | Table 21 lists 100,000; Tülu 3.1 was released at episode 1,474,560 |
| Reproducing Tülu 3 SFT with mean loss | average below the paper at the same LR | inspect the loss-reduction setting; Tülu 3 tuned LR with sum loss (§4.3.2) |
| Short-only SFT on a long-context base | short benchmarks improve, needle or InfiniteBench scores drop | include long-context evaluations in the SFT gate; compare to the base checkpoint |
| Reporting only the target gains of RLVR or DPO | TruthfulQA or safety declines appear only in full tables | publish all development evaluations per stage; split safety by task |
| Very low KL coefficient in constraint RLVR | verifiable reward rises, outputs lose task content | sample outputs at each β of the sweep; track KL and average score (Tülu 3 Fig. 21) |
| Decontaminating by exact match only | near-duplicates that differ in a few tokens, such as a math problem with changed numbers, stay in the training set | 8-gram token-overlap check against both suites; Tülu 3's 8-gram check found 70.7% of HumanEval overlapping Evol CodeAlpaca (Tülu 3 §3.2, Table 37) |
| One seed per ablation | differences smaller than seed range are read as effects | run seeds; Tülu 3 70B seed range is 2.6 points (Table 14) |
| Formatting tokens left in the DPO loss | tail repetition or abrupt termination tokens | mask header and termination tokens (Llama 3 §4.1.4) |

## Check your understanding

1. Math data removal lowered both MATH and DeepMind Mathematics, but persona data removal lowered IFEval without lowering IFEval-OOD. What property of each dataset explains the difference, and what does it predict for a new benchmark?
2. Why can a checkpoint with the highest GSM8K and IFEval scores be the wrong one to release for a generalist model? Use §6.4 and Table 23.
3. Explain, using the softmax gradient, why the rejected term in DPO can lower the log-probability of the chosen response, and how Llama 3's NLL term and token masking each change that.
4. Llama 3's long-context examples are 0.11% of SFT examples, about 4.96% of processed tokens, and about 0.26% of final-response tokens. Which of these numbers would you expect to predict whether long-context ability is kept, and why might short-only DPO not cause the same regression as short-only SFT?
5. Tülu 3 found rejection sampling not worth its compute, while Llama 3 uses it every round. List three differences between the two setups that could explain this without either result being wrong.
6. Tülu 3.1 improved the average by 1.5 points and lowered safety by 4.3 points. What experiment would be needed to attribute either change to GRPO?
7. Llama 3 keeps an example if it is high quality by the reward model *or* by the Llama rating, then deduplicates by quality × difficulty. How would the kept set change if difficulty were removed from the sort key, given the code-filter result in §4.3.1?
8. Tülu 3's safety is reported as orthogonal to other skills at SFT, yet safety fell during DPO and RLVR. Which measurement gap in Table 24 prevents a stronger conclusion?

## Connections

- **Previous:** ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood. Its four meanings of "negative" are applied to both pipelines here.
- **Next:** ch-34 — Case Studies B: Generality versus Specialization in Qwen, OLMo, and Phi Reports. It continues the comparison and holds the side-by-side SFT ledger including Tülu 3 and Llama 3.
- ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing (persona synthesis used in Tülu 3).
- ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares; ch-30c — Weight Averaging and Model Merging for Generalist Models.
- ch-31 — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation (Llama 3 loop).
- ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression; ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation.
- ch-39 — Offline Preference Optimization: DPO and Its Variants; ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO; ch-44 — Process Supervision and Verifiable Rewards.
- ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation; ch-48 — Contamination Detection and Its Effect on Reported Scores; ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming.

## Sources

- [[tulu-3]] — Tülu 3 report (arXiv:2411.15124v5): all Tülu 3 tables, equations, ablations, and hyperparameters quoted here were read in the paper; the card has not been verified and conflicts with the paper on RLVR episodes, verifiers, and stage gains.
- [[tulu-3-sft-mix]] — dataset card view of the SFT mixture and the 57/43 split; the paper's Table 7 adds OpenMathInstruct 2.
- [[tulu-3.1]] — Llama-3.1-Tulu-3.1-8B model card: GRPO hyperparameters, benchmark table, released episode.
- [[tulu-3-1]] — listed for completeness; its multi-base-refresh description is not supported by the model card or blog and is not used.
- [[rlvr-tulu3]] — RLVR summary card; its reward, verifier, and PPO values conflict with the paper and are not used.
- [[ifbench]] — IFBench and IF-RLVR: unseen-constraint generalization and narrowing numbers.
- [[open-instruct-allenai-recipes]], [[open-instruct-allenai-recipes-recipe]] — released launch commands, loss-reduction drift, IF verifier change, Tülu 3.1 run settings.
- [[llama-3]], [[llama-3-recipe]] — Llama 3 report (arXiv:2407.21783v3): rounds, RM, rejection sampling, SFT, DPO, averaging, capability pipelines, filters, safety, contamination, long context.
- [[long-context-llama3]] — long-context card; only statements confirmed in §3.4.2 and §4.3.4 are used, and its per-stage budgets are not in the report.
- [[llama-3-synthetic-pipeline]] — April 2024 Meta blog: SFT + rejection sampling + PPO + DPO description for Llama 3 8B/70B and the withheld 1,800-prompt human evaluation set.
- [[dpo]] — standard DPO objective for the comparison with DPO-norm.
- [[persona-hub]] — persona-conditioned synthesis method used for Tülu 3 prompts.
