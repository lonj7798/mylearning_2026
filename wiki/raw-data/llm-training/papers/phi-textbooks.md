<!-- scope: phi-1 (Microsoft Research, 2023): 1.3B Python code model pretrained on classifier-filtered code plus GPT-3.5 synthetic textbooks, then finetuned on synthetic exercises; includes the paper's own contamination and robustness analyses
     see-also: [[phi-1-5]], [[rephrasing-the-web]], [[hf-cosmopedia]], [[data-constrained-scaling]], [[model-collapse]], [[phi-3]], [[phi-4]]
-->

# Textbooks Are All You Need
- **Core Insight:** A 1.3B-parameter model pretrained on under 7B tokens of classifier-filtered Python code and GPT-3.5-generated textbooks, then finetuned on about 180M tokens of synthetic exercises, reaches 50.6% pass@1 on HumanEval and 55.5% on MBPP (Abstract, Table 1); the authors state that the paper targets one narrow task (§1).
- **Guideline:** When the target is one narrow skill with a fixed evaluation format (here Python functions written from docstrings), filtered plus synthetic "textbook-quality" data can replace a much larger generic corpus, because phi-1 reached 50.6% HumanEval with a 7B-token dataset while StarCoder (15.5B, 1T tokens) reports 33.6% (Table 1). Otherwise, test on similarity-split and held-out problems before claiming breadth, because phi-1 solves 59.5–81.7% of HumanEval problems that have close matches in its exercise set and 26.9–33.8% of those that do not (Table 3).
- **Authors:** Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, et al. (Microsoft Research)
- **Year:** 2023 (arXiv v1 2023-06; v2 2023-10)
- **URL:** https://arxiv.org/abs/2306.11644
- **Source type:** paper
- **Relevant topics:** synthetic pretraining data, data quality filtering, code LLMs, decontamination, benchmark similarity, robustness

## Abstract
phi-1 is a Transformer language model for code with 1.3B parameters. It was trained for 4 days on 8 A100 GPUs on a selection of "textbook quality" data from the web (6B tokens) and on textbooks and exercises generated with GPT-3.5 (1B tokens). It reaches 50.6% pass@1 on HumanEval and 55.5% on MBPP. The paper compares phi-1 with phi-1-base (the model before finetuning on coding exercises) and with phi-1-small (350M parameters, same pipeline, 45% on HumanEval), and describes capabilities that appear only after the finetuning stage.

## Key Contributions
- Three training sets: a filtered code-language set (about 6B tokens), a synthetic textbook set (under 1B tokens), and CodeExercises (about 180M tokens) (§2).
- A quality filter: GPT-4 labels about 100k code samples for educational value, and a random forest on code-model embeddings extends the label to the full pool (§2.1).
- A two-stage pipeline: pretraining on CodeTextbook (filtered code + synthetic textbooks) gives phi-1-base; finetuning on CodeExercises gives phi-1 (§2).
- Contamination analysis: n-gram overlap, embedding and AST similarity, and retraining on exercise sets pruned of HumanEval-like problems (§5).
- A new 50-problem evaluation written by a separate team and graded by GPT-4 (§4).

## Key Figures/Tables to Study
- **Table 1** — HumanEval and MBPP pass@1 with model size and dataset size, self-reported scores for other models.
- **Figure 2.1** — HumanEval for The Stack+ vs CodeTextbook vs CodeTextbook + CodeExercises, at 350M and 1.3B and at 26B vs 76B tokens seen.
- **Table 2** — GPT-4-graded scores on 50 unconventional problems.
- **Table 3** — HumanEval accuracy split into problems similar and non-similar to CodeExercises, for several pruning thresholds.
- **Appendix B** — failure modes of phi-1.

## Technical Details
- **Source pool:** Python subset of deduplicated The Stack plus StackOverflow, over 35M files and over 35B tokens (§2.1).
- **Annotation:** GPT-4 labels about 100k samples with the prompt "determine its educational value for a student whose goal is to learn basic coding concepts" (§2.1). The authors use GPT-4 "minimally only for annotations" and GPT-3.5 "extensively to generate synthetic content" (§2.1); GPT-4 also grades the §4 evaluation.
- **Classifier:** random forest whose features are the output embedding of a pretrained codegen model (§2.1). The selection threshold is not reported.
- **Filter effect at 350M:** unfiltered data saturates at 12.19% HumanEval after 96k steps (about 200B tokens); the filtered subset reaches 17.68% after 36k steps; filtered data plus synthetic textbooks reaches 20.12% (§2.1).
- **Synthetic textbooks:** under 1B tokens of GPT-3.5 text interleaved with code; diversity comes from constraints on topic and target audience; topics are chosen to promote reasoning and basic algorithmic skills (§2.2). Some generation details are withheld "for proprietary reasons" (§1).
- **CodeExercises:** under 180M tokens (§2.2), 879.5K problems (§5.2); each exercise is a function docstring to complete; diversity comes from constraining function names (§2.2).
- **Teacher data quality:** the authors note that GPT-3.5 data "has a high error rate" (§6).
- **Architecture:** decoder-only, 24 layers, hidden size 2048, MLP inner size 8192, 32 heads of size 64, attention and MLP in parallel, rotary dimension 32, FlashAttention, codegen-350M-mono tokenizer, no FIM and no MQA (§2.3). phi-1-small: 20 layers, hidden 1024, MLP 4096, 16 heads (§2.3).
- **Results:** phi-1-base 29% HumanEval (§2); phi-1 50.6% HumanEval, 55.5% MBPP; WizardCoder 57.3%/51.8%; GPT-3.5 47% HumanEval (Table 1). The Stack+ 1.3B run used 76B tokens and 1090 GPU hours; phi-1-base used 51B tokens and 770 GPU hours (Figure 2.1 caption).
- **Contamination checks:** 4 HumanEval problems share a 13-gram with an exercise, and all 4 are false positives (§5.1). AST match thresholds τ = 0.95 to 0.8 remove 42.5K to 354K of the 879.5K exercises (§5.2). phi-1 retrained on pruned data scores 45.1–50.6% on HumanEval vs 41.5% for StarCoder-Prompted (Table 3).

## Recipe ledger
The pretraining settings describe one run with linear warmup and linear decay; "pretrain-stable" labels that whole run. Initialization is not stated.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| phi-1-base | 1.3B | pretrain-stable | data | CodeTextbook: filtered code-language (~6B tokens) + synthetic textbooks (<1B tokens) | arXiv:2306.11644v2 §2 | verified 2026-09-14 | Figure 2.1: CodeTextbook vs The Stack+ at 350M and 1.3B |
| phi-1-base | 1.3B | pretrain-stable | sequence length; packing | 2048; files concatenated into one array with an end-of-text separator token and sliced | §2.3 | verified 2026-09-14 | no ablation reported |
| phi-1-base | 1.3B | pretrain-stable | optimizer; precision; dropout | AdamW; fp16; attention and residual dropout 0.1 | §2.3 | verified 2026-09-14 | no ablation reported |
| phi-1-base | 1.3B | pretrain-stable | effective batch (unit not stated) | 1024, including data parallelism and gradient accumulation | §2.3 | verified 2026-09-14 | no ablation reported |
| phi-1-base | 1.3B | pretrain-stable | peak LR; warmup; weight decay | 1e-3; 750 steps; 0.1 | §2.3 | verified 2026-09-14 | no ablation reported |
| phi-1-base | 1.3B | pretrain-stable | planned steps; released checkpoint | 36,000 steps; checkpoint at 24,000 steps, "∼ 8 epochs", "little over 50B total training tokens" | §2.3 | verified 2026-09-14 | rule for choosing step 24,000 not reported |
| phi-1-base | 1.3B | pretrain-stable | compute | 8 A100 GPUs, under 4 days; 770 GPU hours at 51B tokens | §2.3; Figure 2.1 caption | verified 2026-09-14 | — |
| phi-1 | 1.3B | SFT | data | CodeExercises, ~180M tokens, 879.5K problems | §2, §5.2 | verified 2026-09-14 | Figure 2.1: largest HumanEval gain comes from this stage (§3) |
| phi-1 | 1.3B | SFT | loss; masking | next-token prediction on concatenated exercises; docstring masking not reported | §2.3 | not reported (checked §2.3, App.) | — |
| phi-1 | 1.3B | SFT | effective batch; peak LR; warmup; weight decay | 256; 1e-4; 50 steps; 0.01 | §2.3 | verified 2026-09-14 | no ablation reported |
| phi-1 | 1.3B | SFT | steps; checkpoint selection | 6,000 steps; best of checkpoints saved every 1000 steps (criterion not stated) | §2.3 | verified 2026-09-14 | no ablation reported |
| phi-1 | 1.3B | SFT | compute | additional 7 hours on 8 A100 GPUs | §2.3 | verified 2026-09-14 | — |
| phi-1-small | 350M | pretrain-stable, SFT | optimizer, LR, batch, steps | "same pipeline"; values not given separately | Abstract, §1, §2.3 | not reported (checked body, App. A–C) | — |

## Findings relevant to generality
- **Narrow scope stated by the authors:** §1 cites the debate on whether training on LLM-generated data narrows the resulting model's scope, and says the paper focuses on a narrow task where exceeding the teacher on that task "seems plausible" (§1).
- **Similarity split:** at τ = 0.95, phi-1 solves 81.7% of the 71 HumanEval problems with close matches in CodeExercises and 26.9% of the 93 without; StarCoder-Prompted solves 57.7% and 29.0% (Table 3). The authors note that all models score lower on the non-similar subset (§5.2).
- **Transfer within code:** CodeExercises use only basic Python libraries, yet after finetuning phi-1 produces correct Pygame and Tkinter code in the shown examples (the Tkinter output mis-copies one string), while phi-1-base makes irrelevant or wrong API calls (§3.2, App. A.2). The evidence is qualitative examples, with no metric.
- **Held-out problems:** on 50 new problems graded 0–10 by GPT-4, phi-1 scores 52%, StarCoder 51%, phi-1-small 45%, phi-1-base 37% (Table 2). The text describes phi-1's score as "significantly higher than StarCoder" (§4), but Table 2 shows a 1-point difference.
- **Limitations reported:** Python only, weaker knowledge of specific APIs and less common packages, lower robustness to stylistic variation and grammatical errors in prompts (§6). Accuracy drops as prompt length grows, which the authors attribute to short exercise prompts. The model has difficulty with ambiguous natural language, which the authors attribute to quality filtering. It is also weak at counting and spatial reasoning (App. B).

## Connections
- [[phi-1-5]] — the follow-up report applies the same approach to common-sense reasoning in natural language.
- [[data-constrained-scaling]] — cited in §1 on repeated passes over data; phi-1-base trains about 8 epochs on CodeTextbook (§2.3).
- [[model-collapse]] — Shumailov et al.'s recursive-training study; §1 cites the same authors' 2023 preprint in the scope-narrowing debate.
- [[rephrasing-the-web]] — later work that generates rewritten pretraining text with an LLM instead of new textbooks.
- [[hf-cosmopedia]] — an open synthetic textbook corpus; its relation to Phi is documented in that card.
- [[phi-3]], [[phi-4]] — later Phi model reports.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2306.11644 (arXiv v2, 2 Oct 2023)
- Corrections to the previous card version:
  - Title "Textbooks Are All You Need (phi-1)" → "Textbooks Are All You Need" (title page).
  - "A hand-labeled seed of educational vs non-educational code snippets" → GPT-4 annotates about 100k samples (§2.1).
  - Synthetic exercises are "`<problem, solution>` pairs on topics not well covered in the filtered web" → docstring-completion exercises diversified by constraining function names (§2.2).
  - Textbook prompts specify "exposition style" and cover "stdlib, control flow, algorithms, data processing" → constraints on topics and target audience, topics that promote reasoning and basic algorithmic skills (§2.2).
  - "Decontamination against HumanEval/MBPP (n-gram match); dedup; pedagogical classifier re-ranking" → HumanEval-only analysis with 13-gram overlap, embedding and AST similarity, and retraining on pruned data (§5). MBPP decontamination is not reported.
  - "Figure 2.1 — filtered-only vs filtered+synthetic vs raw web" → The Stack+ vs CodeTextbook vs CodeTextbook + CodeExercises. Filtered-only numbers appear only in §2.1 text, at 350M.
  - "~800 GPU-hours training" → 770 GPU hours for phi-1-base at 51B tokens plus 7 hours on 8 A100s for finetuning (Figure 2.1 caption, §2.3).
  - "jump is driven primarily by the synthetic textbook + exercise data, not by model architecture" → the largest HumanEval gain comes from CodeExercises finetuning (§3); no architecture ablation is reported.
- Removed as unsupported by the source: "Pretraining loss is quality-bounded long before it is quantity-bounded"; "text-davinci-003 era"; teacher API cost "<$100K"; the filter prompt being "a recurring reference for later data-quality classifiers"; "Ignited the Phi line ... inspired WRAP, Cosmopedia, Nemotron-CC"; "later analyses flagged non-trivial overlap" (replaced by the paper's own Table 3); "reproductions exist (Phi-Data / Cosmopedia)"; "inherits GPT-3.5's code-style tics".
- Not reported by the source: data release, synthetic-generation prompts, classifier threshold, MBPP decontamination, phi-1-small hyperparameters.
