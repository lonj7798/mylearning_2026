<!-- excerpt for ch-48; extract of one primary source. Loci are sections/tables of the arXiv PDF.
     Created 2026-09 (generality revision) from https://arxiv.org/abs/2311.04850 (arXiv v2, 2023-11-11).
-->

# Rethinking Benchmark and Contamination for Language Models with Rephrased Samples

- **Authors:** Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica (UC Berkeley; Shanghai Jiao Tong University)
- **Year:** 2023 (arXiv v1 2023-11-08; v2 2023-11-11)
- **URL:** https://arxiv.org/abs/2311.04850 ; tool: github.com/lm-sys/llm-decontaminator
- **Source type:** paper

## What the chapter uses

**Definition (§3).** A *rephrased sample* is a variation of a test instance that preserves semantics but
defeats string-matching detection: reordered words, synonym substitution, translation into another natural
language, or translation into another programming language. Algorithm 1 generates one by prompting a strong
LLM and re-generating until a chosen detector (for example 10-gram overlap) no longer flags it.

**Training on rephrased test sets inflates scores (§5.1, Tables 2–4).** Models fine-tuned for 16 epochs on
rephrased test sets, evaluated 0-shot (originals evaluated 5-shot):

| Benchmark | Model | Original | Fine-tuned on rephrased |
|---|---|---|---|
| MMLU (question only) | Llama 2 7B | 45.3 | 88.5 |
| MMLU (question only) | Llama 2 13B | 54.8 | 89.9 |
| MMLU (full prompt) | Llama 2 13B | 54.8 | 85.9 |
| HumanEval pass@1 (rephrased Python) | CodeLlama 7B | 32.9 | 67.7 |
| HumanEval pass@1 (rephrased Python) | CodeLlama 13B | 36.0 | 81.1 |
| GSM-8K | Llama 2 13B | 28.7 | 95.3 |

**Detector F1 against rephrased pairs (§5.2, Tables 5–6).** 200 prompt pairs per subject (100 random, 100
rephrased). MMLU, abstract algebra: 10-gram 0.926 on the unmodified test set, 0 on rephrased English, 0 on
rephrased Chinese. Embedding search with Multi-QA BERT: 0.985 on rephrased English, 0.179 on rephrased
Chinese; with multilingual BERT it reaches 0.939 on Chinese but falls to 0.111 on rephrased English US
History. LLM decontaminator (top-k embedding retrieval, then GPT-4 pairwise judgement): 0.94–0.99 across all
nine MMLU cells and 0.974–1.0 on HumanEval including translated code.

**Overlap found in real corpora (§5.3, Table 7).** Rephrased-sample contamination of a benchmark's test set
inside widely used training sets: The Stack 4G subset 18.9% of HumanEval; StarCoder-Data 2.4G subset 15.9%;
CodeExercise-Python 15.9%; CodeAlpaca (synthetic, Davinci-003) 12.8%; RedPajama-Data-1T 16G subset 8.5%;
Evol-Instruct-Code 7.9%; MATHInstruct 15.4% of the MATH test set; MATH train 1.6% of the MATH test set;
FLAN CoT 0.5% of MMLU; WizardLM-Evol-Instruct 0.5% of MMLU.

**Threshold problem (§5.2, Fig. 5).** Within-subject embedding-similarity distributions differ by subject, so
one global cosine threshold cannot serve all of MMLU: at 0.8 abstract algebra is detected while sociology is
missed; at 0.4 abstract algebra produces many false positives.

**Open boundary (§6.1).** GSM-8K contains train/test pairs that differ only in names and numbers. The authors
state that a precise definition of contamination remains open and do not resolve it.

**Authors' recommendation (§6.3, §8).** Use stronger decontamination than string matching when training on
public data or on LLM-generated data, and develop fresh one-time exams.

## Verification
- Read on 2026-09-15 against the arXiv v2 PDF, §§2–6 and Tables 1–7.
- Not reported: false-positive rate of the LLM decontaminator on large corpora, cost per million documents,
  any measurement of how much a *partially* rephrased corpus changes downstream general capability.
