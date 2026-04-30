<!-- scope: recursive training on synthetic data causes irreversible distribution tails loss (Nature 2024)
     deps: [[self-instruct]]
     see-also: [[strong-model-collapse]], [[prismatic-synthesis]], [[rephrasing-the-web]]
-->

# AI Models Collapse When Trained on Recursively Generated Data (Shumailov et al., Nature 2024)
- **Core Insight:** Each generation of sampling-then-refitting smooths the tails of the true distribution; iterated, the model's support contracts onto a degenerate near-Gaussian regardless of architecture (Gaussian mixture, VAE, or LLM).
- **Guideline:** Never replace real data with synthetic; always *accumulate* synthetic on top of a persistent real-data anchor, or verify synthetic with an external filter — if you recursively retrain on pure self-outputs, tail loss compounds quickly and is irreversible without fresh real data.
- **Authors:** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, Yarin Gal
- **Year:** 2024 (Nature 631, 755–759)
- **URL:** https://www.nature.com/articles/s41586-024-07566-y ; arxiv 2305.17493
- **Relevant topics:** model collapse, recursive training, synthetic data risk, distribution drift

## Abstract
Shumailov et al. prove — analytically for Gaussian mixtures / linear regression and empirically for VAEs and language models — that when a generative model is iteratively re-trained on samples from its previous generation, the distribution of the model's outputs progressively loses tail mass and converges to a mode-collapsed near-Gaussian. They isolate three error sources (statistical sampling error, functional expressivity error, functional approximation error) that compound across generations. LLM experiments: fine-tune OPT-125M on wikitext2, generate text, retrain, iterate; by generation 9 the model emits nonsense while "new-sample" perplexity appears to improve. The paper defines the **"curse of recursion"** and is the canonical reference for synthetic-data risk.

## Key Contributions
- First formal definition of **model collapse**: early-stage (tail loss) vs late-stage (full mode collapse).
- Proof that the effect is inevitable in finite-sample iterated settings across model classes.
- LLM experiments on OPT-125M fine-tuned recursively on its own outputs — tail degradation observable within a handful of generations.
- Established the "web is becoming polluted with synthetic text" policy-relevance frame.
- Became the Nature-stamped anchor reference for the whole "synthetic data risk" literature.

## Key Figures/Tables to Study
- **Figure showing sequential Gaussian mixture collapse** — iteration 0 to iteration N, density narrowing.
- **OPT-125M generation-by-generation perplexity vs real-data PPL** — divergence plot.
- **Sample-text examples by generation** — qualitative breakdown of coherence loss.

## Core equation (condensed)
Let `p_n` be the model distribution at generation `n`. Under iterated "sample-and-refit" with finite sample size `N`, the `k`-th moment error satisfies approximately:
`Var[μ_k^{(n)}] ≈ n · σ^2 / N + O(model error)`
That is, sampling variance **accumulates linearly in generation count `n`**; tails are erased first because their sampled mass vanishes fastest.

## Synthesis/feedback-loop pipeline (what the paper studies)
- **Seed input:** real wikitext2 corpus.
- **Generation loop:** fine-tune OPT-125M on current data → sample synthetic text → optionally mix with a fraction of real → retrain. Repeat for up to 10 generations.
- **No filtering:** the paper deliberately studies the unfiltered case to isolate the statistical mechanism.
- **Output shape:** sequence of increasingly degraded model checkpoints.

## Empirical findings
- Within ~5 generations: rare-token perplexity spikes while average perplexity looks fine (warning signal hidden in averages).
- By ~9 generations: outputs are incoherent / degenerate.
- Holds for Gaussian mixtures, VAEs, and LMs — the mechanism is statistical, not architecture-specific.

## Risks + gotchas
- **Not a prediction that all synthetic data is bad** — the pure-replacement regime the paper studies is stricter than most real pipelines. Accumulation + filtering avoids the worst case.
- **Small model** (OPT-125M): later work (Strong Model Collapse, Dohmatob 2024) shows the effect persists at scale but with different onset.
- **Real-data anchor size matters** — a constant real-data fraction bounds error; see [[strong-model-collapse]] for threshold behavior.
- **Filter strength matters** — verification with an external (stronger) judge breaks the loop (see 2025 follow-ups below).

## 2025 follow-ups (load-bearing)
- **Dohmatob 2024/25 "Strong Model Collapse" (ICLR 2025 Spotlight):** even 1% synthetic contamination can prevent scaling benefits. See [[strong-model-collapse]].
- **Gerstgrasser et al. 2024 "Is Model Collapse Inevitable? …by Accumulating Real and Synthetic Data":** collapse is driven by *replacement*; with accumulation + fresh real data, test error stays bounded.
- **Zhu et al. 2025:** token-level re-sampling avoids collapse in text generation (analytical linear regression + empirical LM).
- **He et al. 2025 / Garg et al. 2025:** derive optimal real:synthetic mixing ratios.
- **Escaping Model Collapse via Synthetic Data Verification (Zhang et al. 2025, arxiv 2510.16657):** external verifier breaks the loop; establishes convergence guarantees.

## Connections
- Foundational warning for all of §2b (synthetic data): establishes why diversity + filtering are non-negotiable.
- The accumulation-not-replacement rule is the theoretical justification for [[nemotron-4-synthetic]]'s HelpSteer2 anchor and for [[tulu-3-sft-mix]]'s persistent human-data fraction.
- [[prismatic-synthesis]] provides a mechanism-aware escape: generate off the teacher's gradient manifold rather than on it.
- [[rephrasing-the-web]] is *not* recursive collapse — single-pass paraphrase with a different (teacher) model is categorically different from self-iterated refit.
- See [[strong-model-collapse]] for the theoretical tightening of this paper's bounds.
