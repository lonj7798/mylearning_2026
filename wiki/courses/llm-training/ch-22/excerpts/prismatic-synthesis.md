---
chapter: ch-22
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/prismatic-synthesis.md
source_url: https://arxiv.org/abs/2505.20161
created_at: "2026-04-23"
---

# Excerpt: Prismatic Synthesis — G-Vendi and "7B beats 671B"

**Source library:** `wiki/raw-data/llm-training/papers/prismatic-synthesis.md`
**Paper:** Jung, Han, Lu, Hallinan, Acuna, Prabhumoye, Patwary, Shoeybi, Catanzaro, Choi, 2025, "Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning" (UW + Nvidia).

---

## Why this source anchors ch-22

Prismatic is the strongest single claim in the 2023–2025 data-curation literature: **diversity, not generator scale, is the binding constraint**. A 7B student trained on gradient-diverse data beats baselines whose data came from a 671B-parameter generator. For ch-22 §7 this is the paper that inverts the implicit assumption of the whole synthesis pipeline — that bigger generators produce better data.

Three contributions:

1. **G-Vendi metric** — von-Neumann entropy of a normalized per-sample gradient Gram matrix, calibrated against 300+ controlled training runs.
2. **Prismatic pipeline** — active synthesis into low-density gradient regions.
3. **Empirical diversity-over-scale result** — 7B > 671B-distilled on OOD math + NLI.

---

## G-Vendi — exact construction

From the source (lines 32-36):

1. For each candidate `x_i` compute the normalized per-sample gradient under a frozen instruction-tuned proxy LM (Qwen2.5-7B-Instruct class):
   ```
   g_i = ∇_θ L(x_i; θ) / ‖∇_θ L(x_i; θ)‖
   ```
2. Random-project each `g_i` to ≈ 8K dims (Johnson–Lindenstrauss — preserves pairwise inner products).
3. Form the kernel `K_{ij} = <g_i, g_j>` and the density matrix `ρ = K / tr(K)`.
4. **G-Vendi(pool) = exp( H_VN(ρ) ) = exp( -Σ_k λ_k log λ_k )**

where `{λ_k}` are eigenvalues of `ρ`.

Since `g_i` are unit-norm, `tr(K) = N` and `ρ` has trace 1. Eigenvalues `λ_k ∈ [0, 1]` sum to 1. The exponential of the von-Neumann entropy has units of *effective number of distinct directions* — orthogonal-gradient extreme gives G-Vendi = N; identical-gradient extreme gives G-Vendi = 1.

This is the Vendi-score construction (Friedman & Dieng 2023); Prismatic's novelty is the *gradient kernel*.

---

## Why this metric predicts OOD generalization

From the source (lines 14-16, 44-47):

- Across 300+ controlled runs on NLI + math, G-Vendi correlates with OOD accuracy at **Spearman ρ ≈ 0.9**.
- Embedding-Vendi (encoder 14× larger than the gradient proxy) correlates much less.
- Skill-Set Entropy (GPT-4 + Qwen-72B labelers) correlates much less.

Why gradients and not embeddings: embeddings capture *surface form* (what the text looks like); gradients capture *what the optimizer learns from this sample* (which features move, in which direction). Two samples with near-identical embeddings can drive very different gradients (same topic, different reasoning pattern); two samples far apart in embedding space can drive near-identical gradients (different topics, same reasoning pattern). For generalization, only the gradient distance matters.

---

## The Prismatic pipeline

From the source (lines 29-41):

1. **Seed.** Reasoning task distribution (MATH-style problems, NLI pairs).
2. **Candidate generation.** Large pool from a synthesis teacher (persona-hub prompts for math, template perturbation for NLI).
3. **Gradient scoring.** One backward pass per candidate on the 7B proxy; random-project, normalize.
4. **Diversity-targeted selection.** Score each candidate's marginal contribution to G-Vendi; prefer those in *low-density* gradient regions. Greedy max-entropy selection or resample-from-underpopulated-clusters.
5. **Verify.** Standard answer-verifier for math; NLI-label-consistency filter.
6. **Gradient-outlier cap.** Prevent pathologically high-norm gradients from dominating the kernel.
7. **Output.** Curated reasoning corpus; scale in the hundred-K-to-M range.

Step 4 is the new idea. Instead of sampling more from the teacher's natural modes, Prismatic actively seeks directions the teacher is not covering — and generates there.

---

## The 7B-beats-671B result

From the source (lines 47-49):

- **Student**: 7B open LLM.
- **Prismatic data**: ≈10^5–10^6 reasoning samples curated for gradient coverage.
- **Baselines**: same student trained on data distilled from 671B-class generators (DeepSeek-V3-class).
- **Result**: student trained on Prismatic-curated data **beats** the 671B-distilled baseline on OOD math + NLI benchmarks.

Re-state mechanically: a 7B-parameter *proxy* used only for gradient computation, driving targeted synthesis against a frozen grad-entropy objective, produces better training data for the student than a 671B-parameter *teacher* sampling naturally. The generator's parameter count is irrelevant; the coverage of the student's gradient manifold is what predicts OOD transfer.

This is the clearest single datum arguing that the post-2024 synthesis-scaling effort is mis-directed. Scale the *diversity objective*, not the generator.

---

## Anti-collapse behavior

From the source (line 49):

> Gains grow with synthesis scale — anti-collapse behavior: scaling gradient-diverse synthetic data keeps improving OOD, unlike plain teacher sampling.

Plain teacher-sampled synthetic data saturates or degrades with scale (see ch-23 on model collapse). Gradient-diverse curation does not — because the *sampling distribution* is being actively pushed toward uncovered regions, not toward the teacher's natural modes. Prismatic is one of the cleanest technical responses to the collapse literature.

---

## Caveats

From the source (lines 51-56):

- **Proxy dependency.** G-Vendi is defined relative to a proxy's gradient geometry; different proxies produce different rankings.
- **Compute overhead.** Per-candidate backward pass + projection is nontrivial at very large candidate pools.
- **Task scope.** Validated on math + NLI; transfer to code / open-ended writing is future work.
- **Still synthetic.** Gradient coverage does not imply correctness — answer verifier still mandatory.

---

## Connections

- **[[ch-22]]** §7 — the gradient-coverage slot; G-Vendi derivation.
- **[[less]]** — same gradient primitive, alignment objective instead of coverage.
- **[[deita]]** — surface-diversity (embedding) predecessor that Prismatic supersedes on OOD-reasoning.
- **[[ifd]] / [[cherry-llm]]** — orthogonal signal; Prismatic can stack over IFD-cleaned pools.
- **[[model-collapse]]** — the failure mode Prismatic is designed to escape.
- **[[self-instruct]]** — the generation primitive Prismatic augments with gradient targeting.
