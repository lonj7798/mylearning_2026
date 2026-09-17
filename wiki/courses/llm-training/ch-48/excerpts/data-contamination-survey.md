<!-- excerpt for ch-48; extract of one primary source. Loci are sections of the arXiv PDF.
     Created 2026-09 (generality revision) from https://arxiv.org/abs/2502.14425 (arXiv v2, 2025-06-05).
-->

# A Survey on Data Contamination for Large Language Models

- **Authors:** Yuxing Cheng, Yi Chang, Yuan Wu (Jilin University)
- **Year:** 2025 (arXiv v1 2025-02-20; v2 2025-06-05)
- **URL:** https://arxiv.org/abs/2502.14425
- **Source type:** paper (survey). A survey reports other people's results; for any number, the chapter
  cites the primary work the survey points to.

## What the chapter uses

**Definition (§2.1).** Contamination is D_train ∩ D_test ≠ ∅. The survey extends this in two directions:
phase-based contamination over the model lifecycle, and benchmark-based contamination.

**Phase-based contamination (§2.1.1).**
1. Pre-training: benchmark text enters through web scraping and imperfect filtering.
2. Fine-tuning: training directly on benchmark data; the survey states this is uncommon in industrial
   settings and is used deliberately in research to build contaminated controls.
3. After deployment: indirect leakage when users paste benchmark items into a deployed model.
4. Multi-modal: text-label pairs or text-image-label triplets appearing in the training corpus.

**Benchmark-based contamination (§2.1.2).** Four types: text contamination, text-label contamination,
augmentation-based contamination (paraphrases and perturbations of test items), and benchmark-level
contamination (the benchmark's construction or metadata leaking).

**Detection taxonomy by required access (§4).** The organising axis the chapter uses:
- **White-box**: needs the training corpus. n-gram overlap, embedding similarity, exact and near-duplicate
  matching.
- **Gray-box**: needs token probabilities but not the corpus. Perplexity, Min-K%-style membership-inference
  scores, order-likelihood tests.
- **Black-box**: needs only generated text. Guided-prompt completion, multiple-choice memorisation probes,
  cloze tasks.

**Black-box methods described (§4.3).**
- Guided instruction (Golchin and Surdeanu 2023b, "Time Travel in LLMs"): the prompt supplies the dataset
  name, the split, and part of a reference instance; the model's completion is compared with the true
  remainder. High overlap is evidence that the split was seen.
- DCQ (Golchin and Surdeanu 2023a): a multiple-choice item pairs the original instance with three
  synonym-perturbed versions and one invalid option; consistent selection of the original suggests
  contamination.
- TS-Guessing (Deng et al. 2023): the model reconstructs masked elements of a test item.
- DE-COP (Duarte et al. 2024): verbatim-versus-paraphrase multiple-choice probing, built for copyright.

**Stated weaknesses of black-box detection (§5.2).** These methods rest on heuristic assumptions that can
fail; safety filters can suppress the very completions that would reveal memorisation; and the survey
reports that they are weak against augmentation-based (paraphrase) contamination. Position bias in
multiple-choice probes can raise or lower the apparent contamination level.

**Contamination-free evaluation strategies (§3).** Data updating (newly collected items), data rewriting
(perturbing existing items), prevention (canaries, encryption, private held-out sets), dynamic benchmarks,
and LLM-as-a-judge evaluation. The survey does not quantify the effectiveness of any of them.

## Verification
- Read on 2026-09-15 against the arXiv v2 PDF, §§1–5 and Appendix B.
- Not reported: original experiments; all figures in it are attributed to the cited primary works.
