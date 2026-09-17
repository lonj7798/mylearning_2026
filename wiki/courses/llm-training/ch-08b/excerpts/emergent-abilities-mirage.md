---
chapter: ch-08b
course: llm-training
phase: read
excerpt_of: primary source arXiv:2304.15004v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2304.15004
created_at: "2026-09-15"
---

# Excerpt: Are Emergent Abilities of Large Language Models a Mirage?

**Paper:** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo (Stanford University). arXiv v1 2023-04, read at v2 (2023-05-22; marked "Preprint. Under review."). Source type: paper.

## Claim (Abstract, §1)
- Two defining properties of claimed emergent abilities: sharpness and unpredictability.
- Alternative explanation: for a fixed task and model family, emergence appears "due the researcher's choice of metric rather than due to fundamental changes in model behavior with scale"; secondarily, too little test data to estimate small-model performance.
- More than 92% of emergent abilities on BIG-Bench appear under Multiple Choice Grade or Exact String Match (§1, Fig. 5C). The hand annotations come from reference [32], Jason Wei, "137 emergent abilities of large language models" (2022), not from the TMLR paper Wei et al. (reference [33], [[emergent-abilities]]).

## Mathematical model (§2)
- Per-token cross-entropy L_CE(N) = (N/c)^α, c > 0, α < 0 (illustrative form, not required).
- p(single token correct) = exp(−L_CE(N)) = exp(−(N/c)^α).
- Accuracy(N) ≈ p_N(single token correct)^L for an L-token target, assuming independent tokens (footnote 1: the assumption is not true; qualitative match only).
- Token Edit Distance(N) ≈ L · (1 − p_N(single token correct)).
- Sharp, unpredictable changes explained by (1) a nonlinear or discontinuous metric, (2) insufficient resolution at small scale (resolution set by 1/test dataset size), (3) insufficient sampling at large scale.

## GPT-3 / InstructGPT arithmetic (§3)
- Models: 350M, 1.3B, 6.7B, 175B via the OpenAI API (as of 2023-03-15; footnote 3).
- Tasks: 2-shot 2-digit multiplication and 2-shot 4-digit addition.
- Under Accuracy, emergence for 4- and 5-digit targets; under Token Edit Distance with fixed outputs, smooth improvement; TED degrades quasi-linearly with target length (Fig. 3).
- With additional test data, all models achieve above-chance accuracy; accuracy falls approximately geometrically with target length (Fig. 4).

## BIG-Bench meta-analysis (§4)
- Emergence score (Eq. 1, from Srivastava et al.): sign(argmax y − argmin y) · (max y − min y) / sqrt(Median of squared successive differences).
- Of 39 preferred BIG-Bench metrics, at most 5 display emergence (Fig. 5A); hand-annotated triplets from reference [32] show emergence under 4 metrics (Fig. 5B).
- LaMDA tasks emergent under Multiple Choice Grade are not emergent under Brier Score (Fig. 6).

## Induced emergence in vision (§5)
- Shallow nonlinear autoencoders on CIFAR100: smooth MSE, but a thresholded Reconstruction_c metric (fraction of images with squared error below c) looks emergent (Eq. 2, Fig. 7).
- Autoregressive transformers classifying Omniglot sequences: subset accuracy over L ∈ [1, 5] images looks emergent (Fig. 8).

## Discussion (§6, §7)
- Srivastava et al. had observed that cross entropy does not appear sharp where accuracy does (§6).
- "Nothing in this paper should be interpreted as claiming that large language models cannot display emergent abilities" (§7).
- Multiple comparisons: ≥220 BIG-Bench tasks, ~40 metrics per task, ~10 model families, ~10^6 task-metric-family triplets (§7).
- A task and a metric are distinct choices; with accuracy, enough data is needed to measure it (§7).

## Verification
- Read on 2026-09-15 against arXiv PDF v2 (Abstract, §1-§7). Venue is not stated in v2 and is not claimed here.
