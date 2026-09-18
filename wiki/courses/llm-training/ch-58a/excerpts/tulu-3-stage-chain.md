---
chapter: ch-58a
course: llm-training
phase: read
excerpt_of: "Tülu 3: Pushing Frontiers in Open Language Model Post-Training (arXiv:2411.15124v5): §2.1, §4.3.2, §5.4.1, §6.2–6.4, §7.3–7.4, §8.1, Tables 3, 11, 18–22, 24, 31, 34–35"
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-09-17"
note: "The library card model-reports/tulu-3.md has no Verification section. Values below were read in the cached primary text of arXiv:2411.15124v5 (scratchpad sources/tulu-3.txt) on 2026-09-15 (ch-45a excerpt) and 2026-09-17 (§7.3–7.4, Table 31)."
---

# Excerpt: Tülu 3 — post-training stage chain on a Llama 3.1 base, and its development/unseen evaluation split

Lambert et al. (Ai2), 2024. Released checkpoints: Llama-3.1-Tulu-3-8B-SFT/-DPO/-8B, the 70B and 405B equivalents.
Launch commands are in [[open-instruct-allenai-recipes]] and [[open-instruct-allenai-recipes-recipe]].

## Stage chain
Llama 3.1 base (pre-training not performed by this team) → SFT → length-normalized DPO → RLVR (PPO). No mid-training,
no long-context stage, no merge.

## SFT and DPO settings (Table 11, Table 20, Table 34)
- 8B SFT: LR 5e-6, linear, warmup 0.03, 2 epochs, effective batch 128 sequences, max length 4,096. 70B SFT: LR 2e-6,
  same batch and epochs. 405B SFT: LR 2e-6, batch 256, max length 4,096, warmup 0.03, 2 epochs (Table 34).
- §4.3.2: "training for longer did not yield further improvements"; the LR was selected with a sum loss reduction.
- DPO (Table 20): 8B LR 5e-7, 70B 2e-7, 405B 2e-7; linear schedule; effective batch 128 pairs (256 for 405B); max length
  2,048; warmup 0.1; 1 epoch; the column labelled "KL penalty coefficient β" holds the value 5, which is the β of the
  length-normalized DPO loss and not a KL coefficient added to a reward.
- Table 18 (algorithm ablation on UltraFeedback over an early SFT checkpoint, one run per row, development average):
  SFT base 55.7; SimPO 51.8 and 52.9; DPO (β 0.1, 3 epochs) 55.2; PPO (β 0.05) 55.5; DPO-norm at β 5, 1 epoch, LR 5.0e-7
  → 57.3 (the selected setting); DPO-norm at β 2 → 46.8. Report text: "We found that only length-normalized DPO
  outperformed our base checkpoint overall."

## RLVR settings (Table 21, Table 22, §6.2–6.4, §8.1)
- Objective (§6, Eq. 7–8): maximize E[v(x, y) − β·KL(π_θ‖π_ref)] with v = α = 10 for a correct answer and 0 otherwise;
  "We set α = 10 based on pilot experiments and did not tune it further." π_ref is the DPO checkpoint.
- Prompts (Table 22): GSM8K train 7,473; MATH train 7,500; IF verifiable 14,973; total 29,946.
- Table 21 (RLVR column): LR 3e-7 (1e-7 for 70B); effective batch 224 (640 for 70B); K = 4 update iterations; γ 1.0;
  GAE λ 0.95; clip ε 0.2; c1 0.1; temperature 1.0; response length 2,048 (1,024 for GSM8K only); total episodes 100,000;
  no-EOS penalty −10.0; β swept over [0.1, 0.05, 0.03, 0.01]. Caption: "The final 8B RLVR model used β = 0.05 and
  ω = 0.0; the final 70B RLVR model used β = 0.07 and ω = 0.07."
- §6.4 prose for the same 70B run gives "a 1 × 10⁻⁷ learning rate, 0.1 warmup ratio, 2048 response length, 400,000
  episodes, 640 effective batch size, and β = 0.7", which disagrees with the caption's 0.07 and with the released launch
  command (`--beta 0.07`, `--total_episodes 400000`).
- Selection: checkpoints evaluated every 100 steps (40 for the 70B), and the checkpoint with the best MATH and IFEval was
  released; "For all of these models we took an earlier than final checkpoint from the run" (§6.3).
- 405B RLVR (§8.1): GSM8K and IFEval prompts dropped, MATH only; "even with as few as 25 RLVR steps, MATH performance
  improved by over 5 points"; the run was stopped at 75 steps for compute reasons.
- Over-optimization (§6.2.1): "More KL divergence typically results in lower average scores" (Fig. 21). At β = 0.01 the
  prompt "Measure the length of the given object in cm. Pen. In your response, the letter e should appear 14 times."
  received fourteen comma-separated letters "e" and no measurement (Fig. 28); the β = 0.1 model answered with a length.

## Development and unseen evaluation suites (§2.1, Table 3, §7.3–7.4, Table 31)
- Table 3 pairs one development and one unseen benchmark per core skill: MMLU → MMLU-Pro and GPQA (knowledge);
  BigBenchHard → AGIEval English (reasoning); MATH and GSM8K → Deepmind Mathematics (math); HumanEval(+) →
  BigCodeBench (coding); IFEval and AlpacaEval 2 → IFEval-OOD and HREF (instruction following). There is no unseen safety
  evaluation.
- "Crucially, we did not examine scores on our unseen set when developing our models" (§2.1). Training sets contaminated
  with the unseen evaluations were removed (§2.1 decontamination paragraph).
- Table 31 (development "Dev." and unseen "Uns." per skill and stage):

| Skill (dev → unseen) | 8B SFT | 8B DPO | 8B Final | 70B SFT | 70B DPO | 70B Final |
|---|---|---|---|---|---|---|
| Avg. | 64.9 / 29.9 | 68.3 / 31.9 | 68.8 / 32.4 | 78.1 / 41.0 | 80.5 / 44.4 | 80.7 / 44.4 |
| Knowledge (MMLU → GPQA) | 65.9 / 31.9 | 68.7 / 31.2 | 68.2 / 35.7 | 78.9 / 43.3 | 83.3 / 48.0 | 83.1 / 48.0 |
| Reasoning (BBH → AGIEval) | 67.9 / 56.2 | 65.8 / 61.8 | 66.0 / 59.3 | 82.7 / 73.2 | 81.8 / 75.0 | 82.0 / 75.0 |
| Math (MATH → DM Mathematics) | 31.5 / 32.3 | 42.0 / 33.0 | 43.7 / 35.4 | 53.7 / 49.7 | 62.3 / 49.4 | 63.0 / 49.8 |
| Coding (HumanEval → BigCodeBench) | 86.2 / 11.5 | 83.9 / 9.5 | 83.9 / 7.4 | 92.9 / 12.2 | 92.4 / 23.0 | 92.4 / 21.6 |
| Instruction following (IFEval → IFEval-OOD) | 72.8 / 17.6 | 81.1 / 23.9 | 82.4 / 24.3 | 82.1 / 26.8 | 82.6 / 26.4 | 83.2 / 27.8 |

- §7.4.1 text: "We see that our pipeline generalizes well to unseen evaluations, with the final checkpoints obtaining the
  best average performance on both the development and unseen evaluations. For Reasoning and Coding, where the SFT
  checkpoints have the best performance on development evaluations, the subsequent training stages still improve model
  performance on harder unseen evaluations." The 8B coding column of Table 31 moves 11.5 → 9.5 → 7.4 in the other
  direction; the 70B coding column moves 12.2 → 23.0 → 21.6.
- §7.4.1 on the SFT mixture: "we see that our choices overfit to the development evaluations in Precise Instruction
  Following, and to some extent in Knowledge Recall and Reasoning."
- 8B SFT compute: the final 8B SFT model was trained on 32 GPUs for 6 hours (§4.3).

## Verification
- Read on 2026-09-15 (§5.4.1, §6.3–6.4, §8.1, Tables 18–22, 34–35) and 2026-09-17 (§2.1, §7.3–7.4, Table 3, Table 31,
  §4.3) in the cached primary text of arXiv:2411.15124v5.
- Table 32 prints an 8B SFT development average of 64.1 where Table 31 prints 64.9 for the same checkpoint.
- Not reported: the number of PPO steps run for the released 8B RLVR model; DPO optimizer betas; GPU hours for the 70B and
  405B stages.
