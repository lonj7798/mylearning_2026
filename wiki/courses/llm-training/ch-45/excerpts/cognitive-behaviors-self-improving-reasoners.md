---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: primary source (no library card at time of writing)
source_url: https://arxiv.org/abs/2503.01307
version: arXiv v2, 2025-08-15 (COLM 2025)
verified: 2026-09-15 (read against the cached PDF text of v2)
---

# Excerpt: Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs

**Authors:** Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, Noah D. Goodman (Stanford University; SynthLabs)
**Source type:** paper (COLM 2025)
**Used by:** [[read]] §7.3, Recipe

## Why this source is used in ch-45

It isolates what a base model must already do for an RL self-improvement loop to move at all, and it separates that property from answer correctness. This is the evidence the chapter uses against the claim that RL alone creates reflective reasoning.

## Setup

- Two base models under identical training: Qwen-2.5-3B and Llama-3.2-3B (§3).
- Task: Countdown, a number game with a verifiable target; RL with PPO for 250 steps, 4 trajectories per prompt, built on VERL with the TinyZero implementation. PPO was chosen for stability; the authors state performance was anecdotally similar across PPO, GRPO and REINFORCE (§3).
- Four cognitive behaviors are defined and counted in generations: verification (checking one's own work), backtracking (abandoning a failing approach), subgoal setting, and backward chaining (§2, Fig. 3).

## Results quoted in the chapter

- Under identical RL, Qwen reaches about 60% accuracy by the end of training and Llama about 30%. Qwen shifts qualitatively around step 30, with longer responses and higher accuracy; later it moves from stated verification to implicit checking (§3, Fig. 1).
- Priming Llama by fine-tuning on synthetic traces that contain the behaviors — generated with Claude-3.5-Sonnet on Countdown problems — lets it match or exceed Qwen's trajectory under the same RL; backtracking is the behavior that carries most of the effect (§4, Figs. 1–2).
- **Behaviors versus correctness:** models primed with an all-strategies dataset whose solutions are *incorrect* reach the same performance as models primed with correct solutions (§4, Fig. 6). The presence of the behaviors, not the correctness of the answers, is the factor the authors identify.
- **Control:** priming with an empty chain of thought (`<think></think>`) or a length-matched filler leaves Llama at roughly 30–35%, that is at the unprimed level, and the same treatment makes Qwen stop exploring the behaviors (§4, Fig. 5). Extra tokens without the behaviors do not help.
- **Pretraining data:** in 200,000 randomly sampled documents from OpenWebMath and FineMath, backtracking and verification appear infrequently (Fig. 7). Using Qwen-2.5-32B as a classifier, the authors build two 8.3M-token continued-pretraining sets from OpenWebMath — one where the behaviors are present and one control with minimal evidence of them — each rewritten into a question–thought–answer format by Qwen-2.5-32B. Continued pretraining on the behavior-rich set moves Llama onto Qwen's self-improvement trajectory (§5, Fig. 8).

## Conditions and limits stated by the authors

The priming method is domain-specific (Countdown), which the authors name as a threat to generalization (§4). Model scale is 3B for the main comparison, with a partial Llama-3.1-70B exploration reported as uneven (§4, Fig. 4). The paper measures Countdown accuracy and behavior counts, not a broad capability panel.
