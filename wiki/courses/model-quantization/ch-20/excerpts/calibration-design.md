---
chapter: ch-20
course: model-quantization
phase: read
excerpt_of: "Calibration set design — synthesised from GPTQ / AWQ / SmoothQuant / OmniQuant / SpinQuant defaults and low-bit LLM survey reproducibility notes"
created_at: "2026-05-21"
---

# Excerpt: Calibration Set Design

**Sources synthesised:** [[raw-data/gptq]], [[raw-data/awq]], [[raw-data/smoothquant]], [[raw-data/omniquant]], [[raw-data/survey-low-bit-llm-2024]]

---

## Why this matters

Every PTQ algorithm needs a calibration set: a few hundred sequences used to estimate per-layer Hessians, per-channel activation magnitudes, or per-block scale ranges. The choice of calibration set is the single most underdocumented hyperparameter in published quantization results. Two reproducers running "GPTQ-W4 on Llama-2-7B" with different calibration sets routinely get PPL numbers that differ by 0.1–0.5.

---

## Standard sizes by method

| Method | Calibration size | Why this is enough |
|--------|-----------------|---------------------|
| [[gptq]] | 128 seq × 2048 tok | Hessian `H = 2 XXᵀ` reaches full rank around 128 sequences |
| [[awq]] | 128 seq × 512 tok | One scalar α per layer; saturates fast |
| [[smoothquant]] | 512 samples (shorter ok) | Per-channel activation amax; concentrated estimator |
| [[omniquant]] | 128 seq × 2048 tok | Block-wise gradient optimization; needs full forwards |
| [[quarot]] | 128 seq (Hadamard fixed) | No calibration — rotation is data-oblivious |
| [[spinquant]] | 512+ seq | Learned rotation; more parameters → needs more data |
| QAT (e.g., [[bitnet]]) | Full pretrain corpus | Trains from scratch with quant in-loop |

### Why 128 sequences is the GPTQ magic number

GPTQ estimates a per-layer Hessian of size `d_in × d_in`. For `d_in = 4096` (Llama-2-7B FFN input) and `2048 tokens/sequence`, 128 sequences gives `262144` activation samples — far more than `d_in`, so the Hessian is fully ranked. Going from 64 → 128 → 256 sequences gives diminishing returns:

```
Calibration size →   PPL gap vs FP16 (Llama-2-7B, GPTQ-W4)
  16 seq    →   0.30
  64 seq    →   0.20
 128 seq    →   0.15
 256 seq    →   0.14
 512 seq    →   0.13
1024 seq    →   0.13
```

The knee at 128 is sharp. Above 128, you're paying calibration compute for ~0.01–0.02 PPL.

### Why activation-aware methods need more

OmniQuant learns per-channel scales and clip bounds via gradient descent. The optimization has more parameters than GPTQ's closed-form column update; 128 sequences becomes a per-iteration noise floor that hurts the optimization. Empirically, OmniQuant benefits from 256–512 sequences on Llama-13B; SpinQuant from 512+ for the learned rotation.

The general rule: closed-form / Hessian methods saturate around 128 sequences; gradient-based methods scale with parameter count and benefit from 2–4× more.

---

## Source matters more than size

[[survey-low-bit-llm-2024]] documents calibration-source effects:

| Calibration source | Eval | PPL gap | Notes |
|--------------------|------|---------|-------|
| C4 | C4 | ~0.10 | in-distribution baseline |
| C4 | Wikitext-2 | ~0.15 | mild distribution shift |
| C4 | MMLU | up to 2 pp worse than Wikitext calib | MMLU prefers Wikitext-style calib |
| Wikitext-2 | MMLU | best for academic-text MMLU | counter-intuitive: LM corpus better than instruction calib |
| C4 | code generation | bad | needs code in calibration |
| C4 + alpaca + code mix | broad chat | best for general deployment | covers domain breadth |

The principle: calibrate on a distribution that **covers** the deployment distribution. Don't calibrate on Wikitext-2 and deploy a code model. Don't calibrate on C4 and deploy a chat model expecting instruction-format perplexity to match.

---

## Distribution shift on instruction-tuned models

A subtle pathology: SFT + RLHF reshape the activation distribution of the base model. The post-RLHF model has narrower per-channel activation ranges (RLHF systematically de-amplifies outlier behaviors). GPTQ calibrated on Wikitext-2 against an instruction-tuned base produces a model whose chat behavior regresses more than its Wikitext PPL suggests — the Hessian was estimated from a distribution that doesn't match the deployment distribution.

The fix: when quantizing chat models, calibrate on instruction-format text. Standard corpora:
- **OASST / Alpaca / Vicuna** for general chat.
- **UltraChat / ShareGPT** for more diverse chat.
- **WildChat** for the closest-to-deployment distribution.

A 50/50 mix of (deployment-distribution text) and (broad LM corpus) is the robust default.

[[awq]] generalises better here than GPTQ for a structural reason: AWQ uses calibration only to estimate one scalar `α` per layer; the per-channel scale `s_c = (mean|x_c|)^α` is robust to calibration shift. GPTQ's full per-column Hessian update can overfit to calibration covariance.

---

## What to specify in a reproducibility report

For any quantization claim, the calibration set must be specified by:

1. **Source corpus name and version** — e.g., `allenai/c4 en validation 2023-09 snapshot`.
2. **Number of sequences and tokens per sequence** — e.g., `128 seq × 2048 tok`.
3. **Sampling seed** — e.g., `seed=42, random.sample`.
4. **Pre-processing** — tokenizer (which version), deduplication (yes/no), length filter (min/max).
5. **Mixing ratios** if a multi-source calibration set — e.g., `50% c4 + 30% alpaca + 20% codeparrot`.

A paper that reports "128 sequences from C4" without seed or tokenizer version is not reproducible to within 0.05 PPL.

---

## A quick-check workflow

When reproducing a published quantization number:

```
1. Read the paper's calibration spec. If unspecified, default to:
   - 128 seq × 2048 tok
   - allenai/c4 en train split
   - seed = 42 (HuggingFace convention)
   - tokenizer = model's published tokenizer
2. Run the algorithm. Measure PPL on the same eval corpus the paper used.
3. If PPL gap to FP16 differs from paper by > 0.1, investigate:
   a. Calibration set difference?  Try the paper's exact corpus.
   b. group_size / actorder / percdamp difference?
   c. Tokenizer version?  (BPE merges can change across versions.)
   d. Random seed in the Hessian (sequence shuffling) — try 3 seeds, take median.
4. If still off by > 0.1, suspect implementation difference (e.g., AutoGPTQ vs GPTQ-for-Llama).
```

This is the workflow the ch-22 capstone codifies for reproducing a frontier method from scratch.

---

## Connections

- [[ch-20]] §5 — the chapter section.
- [[gptq]] — the standard 128-sequence Hessian estimator.
- [[awq]] — calibration-robust by virtue of the one-scalar-per-layer design.
- [[smoothquant]] / [[omniquant]] — activation-aware methods with different calibration needs.
- [[survey-low-bit-llm-2024]] — source for the calibration-source-shift numbers.
- [[ch-21]] — lab where calibration set choice is a documented variable.
- [[ch-22]] — capstone where calibration set difference is a major reproduction risk.
