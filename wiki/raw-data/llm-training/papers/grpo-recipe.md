<!-- scope: Recipe ledger for DeepSeekMath (arXiv:2402.03300): 1.3B corpus ablations, DeepSeekMath-Base 7B continued pre-training, Instruct 7B SFT, reward model, and DeepSeekMath-RL 7B GRPO run
     deps: [[grpo]]
     see-also: [[deepseekmath]], [[ppo]], [[math-shepherd]]
-->

# DeepSeekMath recipe ledger (companion to [[grpo]])

This ledger holds the training settings printed in "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
Open Language Models" (Shao et al., arXiv:2402.03300). The main card is [[grpo]]. All loci refer to arXiv v3
(2024-04-27). "Stage" follows the course vocabulary: the paper calls the 1.3B and 7B math training "math
pre-training" or "continual training" on top of an existing base model, recorded here as `mid-train`.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-LLM 1.3B, corpus-comparison runs | 1.3B | mid-train | tokens seen per corpus | 150B tokens, one model per corpus | arXiv:2402.03300v3 §2.2.1 | verified 2026-09-14 | Table 1, Figure 3: DeepSeekMath Corpus highest on all 8 math benchmarks; seeds not reported |
| DeepSeek-LLM 1.3B, corpus-comparison runs | 1.3B | mid-train | optimizer | AdamW, β1 = 0.9, β2 = 0.95, weight_decay = 0.1 | §2.2.1 | verified 2026-09-14 | "following the training practice of DeepSeek LLMs"; no ablation reported |
| DeepSeek-LLM 1.3B, corpus-comparison runs | 1.3B | mid-train | LR schedule | multi-step: peak after 2,000 warmup steps; 31.6% of peak after 80% of training; 10.0% of peak after 90% | §2.2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-LLM 1.3B, corpus-comparison runs | 1.3B | mid-train | peak LR | 5.3e-4 | §2.2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-LLM 1.3B, corpus-comparison runs | 1.3B | mid-train | batch; context length | 4M tokens; 4K | §2.2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeek-LLM 1.3B, §5.1 ablations | 1.3B | mid-train | math corpus version | 89B-token DeepSeekMath Corpus from the second collection iteration; §2.2.1 settings unless specified | §5.1 (intro) | verified 2026-09-14 | no ablation reported |
| DeepSeek-LLM 1.3B, code/math ablation | 1.3B | mid-train | token budgets | two-stage: 400B general or 400B code tokens, then 150B math tokens; one-stage: 150B math tokens, or 400B code + 150B math mixed | §5.1.1, Table 6 | verified 2026-09-14 | Tables 6-7 (results in [[grpo]] Findings) |
| DeepSeek-LLM 1.3B and DeepSeek-Coder-Base-v1.5 7B, arXiv ablation | 1.3B; 7B | mid-train | tokens per arXiv corpus | 1.3B: 150B tokens; 7B: 40B tokens; corpora MathPile (8.9B tokens) and ArXiv-RedPajama (28.0B tokens) | §5.1.2 | verified 2026-09-14 | Tables 8-9 |
| DeepSeekMath Corpus | n/a | mid-train | decontamination | remove text containing a 10-gram exact match to a benchmark sub-string; benchmark texts shorter than 10 grams but at least 3 grams: exact matching | §2.1 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-Base 7B | 7B | mid-train | initialization | DeepSeek-Coder-Base-v1.5 7B, the checkpoint right before learning-rate decay | §2.3; Table 4 caption | verified 2026-09-14 | §1: code-trained start "is a better choice compared to a general LLM"; §5.1.1 Tables 6-7 at 1.3B |
| DeepSeekMath-Base 7B | 7B | mid-train | tokens seen | 500B | §2.3; §6 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-Base 7B | 7B | mid-train | data mixture ("distribution of the data"; token share vs sampling weight not stated) | 56% DeepSeekMath Corpus, 4% AlgebraicStack, 10% arXiv, 20% GitHub code, 10% Common Crawl natural language (English and Chinese) | §2.3 | verified 2026-09-14 | no ablation of this split reported |
| DeepSeekMath-Base 7B | 7B | mid-train | peak LR; batch | 4.2e-4; 10M tokens | §2.3 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-Base 7B | 7B | mid-train | other optimizer and schedule settings | "mainly adopt" the §2.2.1 settings; deviations other than peak LR and batch are not listed | §2.3 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-Instruct 7B | 7B | SFT | examples | 776K (English and Chinese; CoT, program-of-thought, tool-integrated) | §3.1 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-Instruct 7B | 7B | SFT | packing; context length | examples randomly concatenated up to 4K tokens | §3.2 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-Instruct 7B | 7B | SFT | steps; batch size (unit not stated) | 500 steps; 256 | §3.2 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-Instruct 7B | 7B | SFT | LR | 5e-5, constant | §3.2 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-Instruct 7B | 7B | SFT | optimizer, warmup, epochs, loss masking | not reported | checked §3, App. A, repo README | not reported | n/a |
| DeepSeekMath-RL 7B reward model | 7B | reward-model | initialization; LR; training data | DeepSeekMath-Base 7B; 2e-5; training set built following Wang et al. (2023b) | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-RL 7B reward model | 7B | reward-model | outcome vs process reward model for the released run | not reported | checked §4.2, §5.2.1, App. A | not reported | Figure 5 (1.3B): GRPO+PS above GRPO+OS |
| Iterative GRPO run (Figure 6) | 7B | reward-model | replay during RM continual training | 10% historical data | §4.1.4 | verified 2026-09-14 | Figure 6: two iterations; largest gain at the first |
| DeepSeekMath-RL 7B | 7B | RL | algorithm | GRPO, Eq. 3, KL estimator Eq. 4 added to the loss | §4.1.1; §4.2 | verified 2026-09-14 | Figure 5 (1.3B): GRPO above Online RFT and RFT |
| DeepSeekMath-RL 7B | 7B | RL | prompts | about 144K chain-of-thought-format questions related to GSM8K and MATH, taken from the SFT data | §4.2 | verified 2026-09-14 | other SFT questions excluded to measure RL effect on benchmarks absent from RL data (§4.2) |
| DeepSeekMath-RL 7B | 7B | RL | policy LR | 1e-6 | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-RL 7B | 7B | RL | KL coefficient β | 0.04 | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-RL 7B | 7B | RL | samples per question G | 64 | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-RL 7B | 7B | RL | max length (unit not stated) | 1024 | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-RL 7B | 7B | RL | training batch size (unit not stated: questions or samples) | 1024 | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-RL 7B | 7B | RL | policy updates per exploration stage (μ) | 1 ("a single update following each exploration stage") | §4.2 | verified 2026-09-14 | no ablation reported |
| DeepSeekMath-RL 7B | 7B | RL | clip ε | not reported (ε appears only as a symbol in Eq. 1, 3, 15) | checked §4.1, §4.2, App. A.1, repo README | not reported | n/a |
| DeepSeekMath-RL 7B | 7B | RL | rollout temperature, top-p | not reported; §5.2.3 says RL used "naive nucleus sampling" without parameters | checked §4.2, §5.2.3, repo README | not reported | n/a |
| DeepSeekMath-RL 7B | 7B | RL | reference model; number of iterations | Algorithm 1 sets π_ref ← π_θ at the start of each iteration; the released run's iteration count is not stated | §4.1.4, Algorithm 1 | not reported | n/a |
| DeepSeekMath-RL 7B | 7B | RL | RL steps, optimizer, warmup, compute | not reported | checked §4.2, App. A, repo README | not reported | n/a |
| DeepSeekMath-Instruct 7B and DeepSeekMath-RL 7B | 7B | eval-gate | Maj@K and Pass@K sampling | temperature 0.7; K from 1 to 64 | Figure 7 | verified 2026-09-14 | n/a |

Units that the paper leaves open are recorded as "unit not stated". The value 1024 for the RL batch is not
defined as questions or as samples, so it must not be converted to "16 prompts × 64 samples" or "1024 prompts".

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2402.03300 (arXiv v3, 2024-04-27; body and App. A.1) and
  the README of github.com/deepseek-ai/DeepSeek-Math (no training hyperparameters listed there).
- Values found in older cards or in chapter text that the paper does not print: clip ε = 0.2; sampling
  temperature 1.0 for rollouts; "batch 1024 prompts"; "16 prompts × 64 completions"; "π_ref frozen SFT model" as a
  stated setting; "epochs per rollout μ = 1" is supported only as "a single update following each exploration
  stage" (§4.2).
