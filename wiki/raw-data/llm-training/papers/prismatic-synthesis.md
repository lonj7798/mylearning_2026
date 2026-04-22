<!-- scope: gradient-entropy diversity metric (G-Vendi) + gradient-targeted synthesis for reasoning
     deps: [[self-instruct]]
     see-also: [[less]], [[deita]], [[model-collapse]], [[yejin-choi-group]]
-->

# Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning
- **Core Insight:** Data diversity is best measured in *gradient space* of a proxy model, not token or embedding space; the entropy of induced-gradient density (G-Vendi) predicts OOD generalization with Spearman ρ ≈ 0.9, and generating data to fill low-density regions of gradient space beats scaling up the generator model.
- **Guideline:** When curating synthetic reasoning data, compute per-sample normalized gradients from a small instruction-tuned proxy (random-projected to ~8K dims), measure G-Vendi, then preferentially synthesize examples whose gradients land in underpopulated clusters — a 7B model trained this way can outperform competitors trained with a 671B generator.
- **Authors:** Jaehun Jung, Seungju Han, Ximing Lu, Skyler Hallinan, David Acuna, Shrimai Prabhumoye, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Yejin Choi (UW / Nvidia)
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2505.20161
- **Relevant topics:** diversity metrics, gradient-space curation, synthetic-data scaling laws, reasoning generalization

## Abstract
Existing diversity metrics (token n-gram, embedding Vendi, Skill-Set Entropy) are surface-level and correlate weakly with out-of-distribution generalization. Prismatic Synthesis introduces **G-Vendi**: (1) compute the per-example gradient of an off-the-shelf instruction-tuned proxy LM w.r.t. the training loss, (2) random-project to a lower dimension while preserving dot products, (3) take the von-Neumann entropy of the normalized gradient density matrix. Across 300+ controlled training runs, G-Vendi achieves Spearman ρ ≈ 0.9 with OOD accuracy on NLI and math reasoning — beating Embedding Vendi (with an encoder 14× larger than the proxy) and GPT-4/Qwen2.5-72B-based Skill Set Entropy. The **Prismatic Synthesis** pipeline then uses G-Vendi as a target: generate candidates from low-density gradient regions. A 7B student trained on the resulting corpus beats baselines distilled from a 671B generator.

## Key Contributions
- **G-Vendi metric** — first-class gradient-entropy measure of training-data diversity.
- Empirical validation across **300+ controlled runs** (varying generator, seed, filter) on NLI + math.
- **Prismatic Synthesis** pipeline — actively target underrepresented gradient-space regions during synthesis.
- Headline: 7B student + gradient-targeted synthesis > competitor distilled from 671B generator → **diversity, not generator scale, is the binding constraint**.
- Public release of code, metric, and training datasets.

## Key Figures/Tables to Study
- **Figure showing Spearman correlation vs OOD acc** — G-Vendi ≈ 0.9 vs Embedding-Vendi, Skill-Set Entropy.
- **Gradient-cluster visualization** — underpopulated regions that Prismatic targets.
- **Table: 7B Prismatic vs distilled-from-671B baselines** — OOD math + NLI.

## Synthesis pipeline (REQUIRED — be concrete)
- **Seed input:** a reasoning task distribution (e.g., MATH-style problems, NLI pairs) with a small candidate pool.
- **Proxy model:** an off-the-shelf instruction-tuned LM (Qwen2.5-7B-Instruct class) used only for gradient computation, frozen.
- **G-Vendi computation (3 steps):**
  1. For each candidate example `x_i`, compute the normalized gradient `g_i = ∇_θ L(x_i; θ) / ‖·‖` on the proxy.
  2. Random-project `g_i` to a lower dim (≈ 8K) while preserving dot products (Johnson-Lindenstrauss).
  3. Build the density matrix `K / tr(K)` where `K_{ij} = <g_i, g_j>`; G-Vendi = exp(von-Neumann entropy of K).
- **Generation step(s):**
  - Generate a large candidate pool with the synthesis teacher (math: persona-hub-style prompts; NLI: template perturbation).
  - Score candidates by their contribution to G-Vendi — prefer those in low-density gradient clusters (greedy max-entropy selection or resample-from-underpopulated-regions).
- **Filtering/rescoring:** standard answer-verifier for math; NLI-label consistency filter; gradient-outlier cap to avoid pathological high-norm gradients dominating.
- **Output shape:** curated reasoning datasets whose gradient coverage is >> naive scaling; disclosed scale is in the hundred-thousand-to-million range depending on domain.
- **Teacher model(s):** generators include Qwen2.5 family and baseline 671B generators used for comparison.
- **Cost estimate:** gradient computation is a single backward pass per candidate on the 7B proxy; cheaper than training but a nontrivial pipeline stage — paper discusses amortization.

## Quality / diversity evaluation
- Spearman ρ ≈ 0.9 between G-Vendi and OOD accuracy on NLI + math across 300+ runs.
- **Beats Embedding-Vendi** (SOTA text-embedding encoder, 14× larger than proxy).
- **Beats Skill-Set Entropy** (GPT-4 + Qwen2.5-72B).
- 7B student trained on Prismatic-synthesized reasoning data outperforms baselines whose data was generated by a 671B teacher (DeepSeek-V3-class).
- Gains grow with synthesis scale — anti-collapse behavior: scaling gradient-diverse synthetic data keeps improving OOD, unlike plain teacher sampling.

## Risks + gotchas
- **Proxy-model dependency:** G-Vendi is defined relative to a proxy's gradient geometry; changing the proxy family shifts the ranking.
- **Compute overhead:** gradient + projection per candidate is nontrivial at very large candidate pools.
- **Random projection choice:** dim & seed affect stability; paper reports averaged results.
- **Task scope:** strongly validated on math + NLI; transfer to coding / open-ended writing is future work.
- **Still synthetic:** gradient-space diversity does not guarantee factual correctness — answer verifier is still mandatory for math.

## Connections
- Supersedes surface-level diversity heuristics in [[deita]] (quality+complexity+diversity) and the IFD-based filter in [[ifd]] / [[cherry-llm]].
- Directly relevant to [[model-collapse]] / [[strong-model-collapse]]: gradient-entropy targeting is a principled escape hatch — explicitly construct data off the teacher's natural gradient manifold.
- Builds on [[less]] (gradient-similarity-to-validation selection) — LESS selects for gradient alignment; Prismatic selects for gradient *coverage*.
- Sits in the Yejin Choi lineage → [[self-instruct]] → [[yejin-choi-group]] → [[star]] / [[quiet-star]]; see the group's recurring theme of "diversity over scale."
- Reference implementation for the 2025 "gradient-based diversity" research lane; likely displaces the generic `synthetic-data-scaling-laws.md` slot in the collection plan.
