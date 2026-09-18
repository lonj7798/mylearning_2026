<!-- excerpt for [[ch-59]] — Phi-4-reasoning SFT and GRPO settings, plus its off-target benchmark table
     source: Phi-4-reasoning Technical Report (Microsoft Research, 2025-04), §3, §4, §5, Table 2
     read 2026-09-17 from the cached primary text; see [[phi-4]]
-->

# Phi-4-reasoning — distill-SFT settings, short GRPO, and off-target measurements

## SFT (§3)

> "Our SFT data comprises over 1.4 million prompt-response pairs, totaling 8.3 billion unique tokens of
> reasoning domains such as math and coding, and alignment data for safety and Responsible AI. Training is
> run over roughly 16K steps, with a global batch size of 32 and a context length of 32K tokens. We use
> AdamW … with a learning rate of 10⁻⁵, linear warm up over 450 steps, and a weight decay of 10⁻⁴."

- "The final model was trained for 16B tokens using this mixture" (§3), against 8.3B unique tokens, so the
  average source is seen about twice.
- "the RoPE base frequency was doubled, and the model was trained for a maximum length of 32K tokens" (§3).
- Learning-rate selection: "We performed a grid search over [1e−6, 2e−5], starting from the SFT learning rate
  of the base model Phi-4 (1e−6) to its mid-training learning rate (3e−5). In our experiments, 1e−5 provided
  the best balance" (§3).

## GRPO (§4)

> "Hyper-parameters for the RL training are: a global batch size of 64 across 32 Nvidia H100 GPUs, Adam
> optimizer learning rate 5 × 10⁻⁸ with cosine warm-up in the first 10 steps, GRPO group size of G = 8, KL
> regularization of β = 0.001 and entropy coefficient of γ = 0.001."

- "The RL training focused exclusively on mathematical reasoning. The seed dataset for GRPO consisted of
  72,401 mathematical problems (prompts without solutions), from which we subsample 64 problem seeds per RL
  iteration" (§4). "the seed data contained no coding exercises".
- Checkpoint selection: "We select as our RL checkpoint the model with the best observed AIME 2024 score,
  which is the model trained for 90 steps, over only ∼ 6k examples (and 8 trajectories of responses per
  example)" (§4.2).
- "additional GRPO training for only 90 steps boosts AIME performance by more than 10% (Figure 7a). Further
  training for more steps does not translate to additional gains, hinting the potential of an already strong
  SFT model is near its performance ceiling. A caveat … is the fact that we clip responses beyond 31k output
  tokens during GRPO" (§4.2).
- Framework: verl (§4).

## Table 2 (§5): general-purpose benchmarks, mean pass@1 over five generations

| Benchmark | Phi-4 | Phi-4-reasoning | Phi-4-reasoning-plus | o3-mini | GPT-4o |
|---|---|---|---|---|---|
| FlenQA (3K-token subset) | 82.0 | 97.7 | 97.9 | 96.8 | 90.8 |
| IFEval Strict | 62.3 | 83.4 | 84.9 | 91.5 | 81.8 |
| ArenaHard | 68.1 | 73.3 | 79.0 | 81.9 | 69.0 |
| HumanEvalPlus | 83.5 | 92.9 | 92.3 | 94.0 | 84.9 |
| MMLUPro | 71.5 | 74.3 | 76.0 | 79.4 | 73.5 |
| Kitab — no context, precision | 19.3 | 23.2 | 27.6 | 37.9 | 53.7 |
| Kitab — no context, recall | 8.2 | 4.9 | 6.3 | 4.2 | 20.3 |
| Kitab — with context, precision | 88.5 | 93.8 | 93.6 | 94.0 | 84.7 |
| Kitab — with context, recall | 68.1 | 74.8 | 75.4 | 76.1 | 69.2 |
| ToxiGen discriminative — toxic category | 72.6 | 86.7 | 77.3 | 85.4 | 87.6 |
| ToxiGen discriminative — neutral category | 90.0 | 84.7 | 90.5 | 88.7 | 85.1 |
| PhiBench 2.21 | 58.2 | 70.6 | 74.2 | 78.0 | 73.1 |

Caption note: Phi-4-reasoning and Phi-4-reasoning-plus were evaluated at temperature 0.8 and Phi-4 at
temperature 0.0, so the three columns are not a controlled comparison of the training stage alone.

Two entries fall relative to the Phi-4 base: Kitab no-context recall (8.2 → 4.9 → 6.3) and ToxiGen neutral
category (90.0 → 84.7, recovered to 90.5 in the plus model). The report's own limitation statement is that
"the reinforcement learning (RL) data is limited to math. Although there are signs of generalization to other
domains" (§7).
