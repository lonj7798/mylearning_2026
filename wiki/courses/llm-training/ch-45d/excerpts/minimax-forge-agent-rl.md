---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: MiniMax, "Forge: Scalable Agent RL Framework and Algorithm", 2026-02-13 (no library card at the time of writing; chapter-local verified extract)
source_url: https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm
created_at: "2026-09-15"
---

# Excerpt: Forge: Scalable Agent RL Framework and Algorithm (MiniMax-M2.5)

- **Authors:** MiniMax
- **Year:** 2026 (post dated 2026-02-13)
- **Source type:** official blog
- **Used in:** [[read]] §4, §5, §7, Recipe, Generalization lens

## Scale claims
- "Throughout the construction of MiniMax M2.5, our RL system navigated over a hundred thousand distinct real-world agent scaffolds and environments. Operating with context lengths of up to 200k, the system maintained a daily processing throughput on the scale of millions of samples."
- "In total, we have integrated hundreds of types of scaffolds and thousands of distinct tool invocation formats."

## System design
- Three modules: an Agent Side (white-box and black-box agents plus environments, acting as "a pure trajectory producer"), a Middleware Abstraction Layer (Gateway server plus Data Pool), and a Training/Inference Side (Rollout Engine, Train Engine).
- Stated motivation for the decoupling: standard frameworks "treat agents as white-box functions with a shared state between agent and trainer", which "prevent[s] the model capability from generalizing effectively on an arbitrary black-box agent".
- "Leveraging the modular design of our RL framework, we can conduct training using an extensive array of scaffolds without requiring internal modifications to the Agent. This approach effectively enables the model to generalize across diverse scaffolds, a.k.a. environments."

## Context management as an action
- Named failure modes: "Context Rot" — "the accumulation of intermediate reasoning steps and redundant observations creates an 'attention dilution' effect" — and "Inference-Training Mismatch", where applying context management only at inference "introduces a severe distribution shift from the RL training data".
- Fix: "we integrate the CM mechanism directly into the RL interaction loop, effectively treating Context Management as a functional action that drives state transitions", so "The state transition from S_t to S_{t+1} implicitly encapsulates the context-switching logic".
- Black-box RL supports "arbitrary context manipulations (such as memory compression and history rewriting)" and reports "consistent, stable improvements even across completely opaque black-box systems" (no numbers).

## Scheduling and efficiency
- Windowed FIFO: the training scheduler sees only a window of size W (example given: W = 0.3N) of the generation queue. Stated purpose: strict FIFO suffers head-of-line blocking, greedy fetching causes "a severe Data Distribution Shift ... initially dominated by short, 'easy' tasks and later by clustered 'hard' tasks". The window "forces the scheduler to wait for 'stragglers' ... within the current window".
- Prefix Tree Merging merges completions sharing a prefix into one tree-structured forward pass: "this solution achieves a 40x training speedup and significantly reduces memory overhead ... while guaranteeing strict mathematical equivalence to standard methods".
- MTP heads "continuously fine-tuned via Top-K KL loss" to keep speculative-decoding acceptance aligned with the drifting RL policy; heterogeneous prefill/decode disaggregation; a DFS-backed global L3 KV cache pool.

## Algorithm
- Core algorithm: CISPO.
- "Unified Mixed-Domain Training: Unlike multi-stage reinforcement learning, which often leads to negative transfer or interference between domains, we adopt a unified training strategy. We mix tasks across Reasoning, General QA, and Agent domains simultaneously. This joint training approach mitigates the performance degradation typically seen in sequential training and significantly enhances the model's generalizability across diverse tasks." No ablation is printed.
- Composite reward: a process reward "targeting intermediate behaviors (e.g., penalizing language mixing or specific tool invocation errors)"; a "Task Completion Time Reward" using relative completion time to incentivize parallelism; and a reward-to-go formulation "to normalize returns", stated to reduce gradient variance over 200k-token horizons.

## Verification
- Checked on 2026-09-15 against https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm (post dated 2026-02-13).
- Source type note: official but without benchmark tables. Quantitative claims available: over 100,000 scaffolds/environments, 200k context, "millions of samples" per day, 40x training speedup from prefix-tree merging. The generality claim for unified mixed-domain training is stated, not measured.
- Not reported: CISPO hyperparameters; reward weights; number of RL steps; model size of M2.5; any before/after benchmark comparison.
