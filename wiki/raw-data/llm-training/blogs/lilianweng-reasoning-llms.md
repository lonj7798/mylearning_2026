<!-- scope: Lilian Weng "Why We Think" — test-time compute and reasoning LLMs
     deps: [[README]]
     see-also: [[lilianweng-rlhf]], [[deepseek-r1]], [[let-verify]]
-->

# Lil'Log — "Why We Think" (Reasoning LLMs / Test-Time Compute)
- **Core Insight:** Giving a model more compute at inference — via chain-of-thought, self-verification, tree search, or latent thinking — is a second scaling axis that compounds with pretraining scale, and it interacts tightly with how reasoning is trained (RL with verifiable rewards is the key unlock).
- **Guideline:** When a reasoning model plateaus, before scaling pretraining consider adding CoT length budgets, verifier-gated search, or process-reward training.
- **Author:** Lilian Weng
- **Year:** 2025 (May 1, 2025 — "Why We Think")
- **URL:** https://lilianweng.github.io/posts/2025-05-01-thinking/
- **Relevant topics:** test-time compute, chain-of-thought, self-verification, latent thought, reasoning RL, CoT monitoring, interpretability

## Summary
"Why We Think" is Weng's survey of the 2024–2025 reasoning-LLM wave triggered by o1-preview and DeepSeek-R1. The post organizes the landscape around the central idea that test-time compute is a *second* scaling axis — orthogonal to pretraining parameters/tokens — and that the most effective uses are (a) chain-of-thought reasoning, (b) smart decoding (best-of-N, tree search, majority voting), and (c) latent-thought training (Quiet-STaR style). It connects these inference-time mechanisms back to training: RL on automatically checkable solutions (RLVR / rule-based reward) is what makes CoT reliably useful rather than merely decorative.

A second major thread: CoT provides convenient interpretability — the model's reasoning is externalized — and recent work demonstrates monitoring CoT can detect misbehavior (reward hacking). But this is fragile; if models learn to hide reasoning in latent states or to generate deceptively compliant CoTs, the interpretability benefit disappears.

## Summary (continued / extended format)
This post functions as the 2025 counterpart to Weng's 2023 RLHF posts. Where the RLHF posts formalize preference-alignment pipelines, "Why We Think" formalizes reasoning pipelines. Read the two together to get Weng's full model of LLM post-training as of 2025.

## Key Contributions
- Unified framing of "test-time compute" as a scaling axis.
- Taxonomy of test-time compute mechanisms: CoT, self-consistency, tree search, verifier-gated search, latent thought.
- Connection between reasoning RL (RLVR) and inference-time CoT quality.
- Discussion of CoT-as-interpretability and its limitations.
- Coverage of key papers: o1, R1, Quiet-STaR, process-reward models, let's-verify-step-by-step.

## Key Figures/Tables to Study
- **Test-time compute vs pretraining compute scaling plot:** shows the distinct axes.
- **CoT mechanism taxonomy table:** CoT / self-consistency / BoN / tree search / verifier search / latent.
- **CoT monitoring diagram:** weaker model monitors stronger model's CoT for misbehavior.

## Technical Details

### Test-time compute mechanisms
- **Chain-of-Thought (CoT):** produce reasoning tokens before answering. Emergent at scale; elicited reliably by SFT on CoT data or RL with verifiable rewards.
- **Self-consistency / majority voting:** sample N CoTs, pick the most-frequent answer. Cheap; scales well on math.
- **Best-of-N (BoN) with verifier:** sample N, pick the one a verifier or RM likes best.
- **Tree search / ToT:** branch on intermediate states; requires a value estimate.
- **Process Reward Models (PRM):** step-level reward model for selecting/pruning CoT branches. Labeled data from PRM800K / Math-Shepherd.
- **Latent thought:** train the model to think in a learned latent space before emitting tokens (Quiet-STaR, Coconut).

### Reasoning RL unlocks CoT quality
Pretrained CoT is often superficial — the model produces reasoning-shaped text but does not causally use it. RL with checkable solutions (math correctness, code unit-test pass) directly rewards CoT that leads to correct answers, forcing the CoT to actually carry information. This is why R1 / o1 feel qualitatively different from prompted-CoT GPT-4.

### Interpretability via CoT
Advantages:
- Model's intermediate steps are externalized as natural language.
- Easier to detect misbehavior than with internal activations alone.
- A weaker monitor model can audit a stronger solver model's CoT.

Limitations:
- Obfuscated CoT: models can learn to hide reasoning steps behind shorter cues.
- Latent-thought training (Coconut, Quiet-STaR) moves reasoning out of plain text — breaks CoT-as-interpretability.
- Deception pressure: if unfiltered reasoning is penalized, models may learn to lie about their own reasoning.

### Open questions Weng flags
- How much of R1's behavior is genuinely new capability vs. elicited from the base?
- Can process-reward training scale without expensive human labels?
- Will latent-thought training dominate (better efficiency) despite hurting interpretability?

## Connections
- [[lilianweng-rlhf]] — companion 2023 post on preference-alignment RLHF.
- [[deepseek-r1]] — primary case study for reasoning RL that post analyzes.
- [[let-verify]] — Lightman 2023 process-reward paper; core reference.
- [[quiet-star]] — latent-thought training.
- [[prm800k]] / [[math-shepherd]] — process-reward data.
- [[nathan-lambert-rl-overview]] — parallel wider-field commentary.
