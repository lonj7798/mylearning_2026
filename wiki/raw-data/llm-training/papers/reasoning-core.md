<!-- scope: Reasoning Core (arXiv:2603.02208) — procedural generators with external-solver verification for symbolic pre-training, SFT, and RLVR data; small-model data-mixing experiments
     deps: [[front-loading-reasoning]]
     see-also: [[prorl]], [[rlvr-beyond-base-model]], [[nemotron-4-synthetic]], [[prismatic-synthesis]], [[deepseek-r1]]
-->

# Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre-training and Post-Training
- **Core Insight:** Adding procedurally generated, solver-verified symbolic data to 0.5B tokens of FineWeb, SYNTH, or Dolci lowered answer NLL on PlatinumBench for all three corpora and slightly lowered validation loss on the natural-language data, in models under 100M parameters (§4.2, Fig. 3; Limitations).
- **Guideline:** When mixing procedural symbolic data into a small-model pre-training or instruction-tuning corpus, the authors report r = 0.5 (half as many added tokens as the original corpus, one third of the final mix) as the best of r ∈ {0, 0.1, 0.3, 0.5, 1.0} (§4.2, Fig. 3, two runs); whether this holds above 100M parameters or 0.5B tokens was not tested (Limitations).
- **Authors:** Valentin Lacombe, Valentin Quesnel, Damien Sileo (Univ. Lille, Inria, CNRS, Centrale Lille)
- **Year:** 2026 (arXiv v1 2026-03-02; only version; preprint)
- **URL:** https://arxiv.org/abs/2603.02208 (code: https://github.com/sileod/reasoning_core; data: hf.co/collections/reasoning-core/datasets)
- **Source type:** paper
- **Relevant topics:** procedural data generation, symbolic pre-training, verifiable rewards, external solvers, difficulty curriculum, synthetic reasoning traces

## Abstract
Existing procedural generators often rely on fixed puzzles or templates and lack distributional breadth. Reasoning Core generates verifiable symbolic tasks in PDDL planning over randomized domains, first-order logic with equality, context-free grammar parsing and generation, causal reasoning over random Bayesian networks, and systems of equations. Each task is paired with an external solver and has continuous difficulty control. Examples can include solver-derived reasoning traces for supervised training, and the same interface returns verifiable rewards for RL. Mixing the data into pre-training improves downstream reasoning while preserving or slightly improving language-modeling quality. Zero-shot evaluations show the tasks are challenging for GPT-5. Code and data are released under the MIT license.

## Key Contributions
- A generator suite over formal domains with external solver verification; 28 task names in `list_tasks()` versus 100+ tasks in Reasoning Gym (Table 1, Fig. 1).
- `gramforge`, a grammar framework with depth control ("bushiness factor") and context-sensitive derivation (§3.4).
- Pre-training and instruction-tuning mixing experiments on sub-100M models (§4.2, Fig. 3).
- Released datasets and code under MIT; sizes are stated two different ways (see Technical Details).

## Key Figures/Tables to Study
- Table 1: feature comparison with Reasoning Gym (task count, solver verification, traces, pre-training support).
- Fig. 1: task API (`get_task`, `generate_example(level=k)`, `score_answer`) with arithmetic and `logic_nli` examples.
- Fig. 2: GPT-5, GPT-5-mini, GPT-5-nano zero-shot average reward per task at easy and hard levels.
- Fig. 3: test NLL and PlatinumBench answer NLL against RC ratio r for FineWeb, SYNTH, and Dolci.
- Appendix A: per-task generation and scoring rules.

## Technical Details
- **Task scope.** Besides the five domains in the abstract, §3.1 lists regex and symbolic induction, retrieval/perception tasks (missing-element detection, set intersection), and TPTP formal-mathematics tasks (§3.1, App. A). §3.1 says "non-linear equation systems"; App. A describes `equation_system` as linear systems.
- **Difficulty knob.** One float per generator that adjusts factors such as proof depth, variable count, or plan length; discrete hyperparameters use stochastic rounding (§3.2).
- **Verification.** Vampire/E theorem provers for logic, FastDownward for PDDL, Sympy for equations; non-Python solvers are containerized via udocker/Apptainer (§3.3).
- **Scoring is not always binary.** Examples: F1 over derivation edges (`proof_reconstruction`), Jensen–Shannon divergence (`bayesian_association`, `bayesian_intervention`), character-level similarity (`diff_prediction`), Jaccard similarity (`set_intersection`), plans accepted with a length penalty (`planning`) (App. A).
- **Traces.** TPTP proofs are reformatted for logic and math tasks; arithmetic derivations are logged during bottom-up evaluation with exact fractional arithmetic; planning traces track state and remaining goals; pathfinding traces follow BFS (§3.6, App. A).
- **Production.** Timeouts scale with difficulty, stalled solver processes are killed, a "balancing key" caps the frequency of answer labels within a batch, and single-threaded workers coordinate via file locks (§3.5).
- **Released data (paper).** Pre-training set: 80% of examples at level 0, 20% split between levels 1 and 2. Post-training set: levels sampled uniformly from {0, 3, 5}. §4 states 10M examples (5B tokens) for pre-training and 1M examples (1B tokens) for post-training; §1 states 5B pre-training and 2B post-training tokens. Each subset took about 3 days with 48 threads on an Intel Xeon Gold 5320 (§4).
- **Zero-shot evaluation.** GPT-5 family at level 0 and level 5, 200 samples per task and level, reasoning effort medium, temperature 1, top-p 1 (§4.1). The text reports no per-task numbers; the authors state that difficulty control "works as intended for most tasks" (§4.1, Fig. 2).
- **Metric choice.** Answer NLL is used instead of accuracy because small models have trouble following instructions (§4.2).
- **Repository (README, main branch, fetched 2026-09-14; not pinned).** States "More than 10B tokens" of pre-generated data, lists 65 gallery task examples, and names integrations with reasoning-gym, SynLogic, Prime Intellect, OpenReward, and OpenEnv. These figures describe the repository at fetch time, not the paper.

## Recipe ledger
All paper loci refer to arXiv:2603.02208v1. The runs are data-mixing experiments, not released models.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Monad-56M architecture, random init | 56M | pretrain-stable | natural-language data | 0.5B tokens of FineWeb or SYNTH | §4.2 | verified 2026-09-14 | no ablation reported |
| Ettin-68M decoder (pre-trained) | 68M | SFT | natural-language data | 0.5B tokens of Dolci | §4.2 | verified 2026-09-14 | no ablation reported |
| Both runs above | 56M / 68M | pretrain-stable / SFT | added RC tokens | r × 0.5B, r ∈ {0, 0.1, 0.3, 0.5, 1.0} | §4.2 | verified 2026-09-14 | Fig. 3: r = 0.5 described as "sweet spot"; two runs, standard error |
| Both runs above | 56M / 68M | pretrain-stable / SFT | reasoning traces | included for 50% of RC examples when available | §4.2 | verified 2026-09-14 | no ablation reported |
| Both runs above | 56M / 68M | pretrain-stable / SFT | optimizer | Prodigy with Schedule-Free, default hyperparameters | §4.2 | verified 2026-09-14 | no ablation reported |
| Both runs above | 56M / 68M | pretrain-stable / SFT | batch; context; epochs | 16; 1024 tokens; 1 epoch | §4.2 | verified 2026-09-14 | no ablation reported |
| Both runs above | 56M / 68M | pretrain-stable / SFT | LR, warmup, weight decay | not reported (defaults of Prodigy/Schedule-Free named, values not printed) | §4.2 | not reported (body, Limitations, App. A checked) | n/a |
| Both runs above | 56M / 68M | pretrain-stable / SFT | compute | roughly one day per run on one Nvidia A30 | §4.2 | verified 2026-09-14 | n/a (model and data sizes chosen to approximate Chinchilla-optimal ratios, §4.2) |
| Both runs above | 56M / 68M | eval-gate | evaluation | dataset test NLL; PlatinumBench (15 tasks) answer NLL | §4.2 | verified 2026-09-14 | n/a |

## Findings relevant to generality and negative feedback
- **Breadth (Result, single study).** RC mixing "consistently improves PlatinumBench answer NLL across all three corpora" and "slightly reduces the validation loss" on the natural-language data (§4.2, Fig. 3). The comparison is at unequal token counts: mixing adds tokens (§4.2).
- **Untested transfer (stated limit).** Transfer to legal reasoning, scientific hypothesis evaluation, agentic tool-call ordering, or other non-formal settings "we have not empirically validated" (Limitations). Accuracy-based downstream evaluation of the trained models is not reported.
- **No RLVR results.** The paper reports no RL training; the authors state that small-budget RLVR (a few thousand episodes) would not represent the intended regime (§4.2, Limitations).
- **Contamination (author claim, untested).** Procedurally novel instances are described as "immune to benchmark contamination" (Broader Impact); no measurement is reported.
- **Negative examples as labeled content.** Some tasks create negative instances: premise perturbation verified by a theorem prover (`conjecture_entailment`), edge swaps confirmed non-isomorphic (`graph_isomorphism`, whose class distribution skews negative), and perturbed strings (`parsability`); `regex_induction` places verified non-matching strings in the prompt (App. A). In all four cases the negatives are labels or input content trained with ordinary loss, not negative gradients.
- **Label errors.** The authors state that "a small fraction of generated instances could contain errors" despite solver checks, LLM-assisted review, and human adjudication; no error rate is given (Limitations).

## Connections
- [[front-loading-reasoning]] — cited in §1 (Akter et al., 2025) among work on adding reasoning data before post-training.
- [[prorl]] — cited in §1 (Liu et al., 2025b) as the compute-intensive prolonged-RL alternative to symbolic pre-training.
- [[rlvr-beyond-base-model]] — cited in §1 (Yue et al., 2025) for RLVR amplifying strategies already present after pre-training.
- [[deepseek-r1]] — cited in §1 (Guo et al., 2025) as an RLVR reference.
- [[nemotron-4-synthetic]] — not cited; an LLM-generated synthetic-data pipeline, the category §2 contrasts with purely procedural generation.
- [[prismatic-synthesis]] — not cited; a different approach to synthetic-data diversity.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2603.02208 (arXiv v1, 2026-03-02; full text incl. Limitations, Broader Impact, App. A) and the repository README (github.com/sileod/reasoning_core, main branch, fetched 2026-09-14).
- Corrections: "five core families" → five domains named in the abstract, but 28 task names across more categories (Fig. 1, §3.1, App. A); "Releases pre-generated data at more than 10B tokens" → the paper states 5B + 1B tokens (§4) or 5B + 2B tokens (§1); ">10B" is the current README's figure; "Downstream evaluation table" → Fig. 3 plots NLL, not a table of accuracies; "Algebra: systems of equations" → §3.1 says non-linear, App. A describes linear systems; code URL `sileod/reasoning-core` → the paper footnote gives `github.com/sileod/reasoning_core`.
- Removed as unsupported: "the README shows `list_tasks`, `get_task`, `score_answer`" as a README fact (the current README shows `get_task` and `score_answer`; `list_tasks()` appears in paper Fig. 1); "writing JSON data for Hugging Face Datasets" (README: JSONL shards and postprocessing for Hugging Face Datasets); [[let-verify]] and [[training-verifiers-to-solve-math-word-problems]] as "direct process-supervision predecessors" and [[quiet-star]] as a "conceptual companion" (none is cited by the paper).
- Internal inconsistencies in the source: post-training token count 1B (§4) vs 2B (§1); non-linear (§3.1) vs linear (App. A) equation systems.
- Not reported by the source: accuracy numbers for the trained models, per-task GPT-5 rewards in text, LR values, any RL training run, verification error rate.
