<!-- scope: Recipe ledger for OpenMathInstruct-2 (arXiv:2410.01560v2) — data-generation, SFT, ablation, and evaluation settings; companion to [[openmathinstruct-2]]
     deps: [[openmathinstruct-2]]
     see-also: [[openmathinstruct]], [[llama-3]]
-->

# OpenMathInstruct-2 — Recipe ledger
Companion to [[openmathinstruct-2]]. Paper rows were read in the arXiv v2 PDF on 2026-09-14. Rows marked "NeMo-Skills" come
from the authors' reproduction documentation in github.com/NVIDIA/NeMo-Skills (the paper links the earlier path
github.com/Kipok/NeMo-Skills). Per §5.3 of the course standard, paper values and documentation values are separate facts.

Units: the paper prints "batch size" without a unit; the NeMo-Skills flag is `train_global_batch_size`. "Pairs" are
question-solution pairs. "Samples per question" are sampled solutions before filtering.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenMath2-Llama3.1-8B | 8B | SFT | Base model | Llama3.1-8B-Base | arXiv:2410.01560v2 §1, §4 | verified 2026-09-14 | no ablation reported |
| OpenMath2-Llama3.1-8B | 8B | SFT | Training data | full OpenMathInstruct-2: 13.97M pairs, 607.3K unique questions | §4 "Final Results"; Table 5 | verified 2026-09-14 | Fig. 1: 1M, 2M, 5M, full; consistent gains, "no signs of saturation" at 14M (§4) |
| OpenMath2-Llama3.1-8B | 8B | SFT | Batch size | 512 (unit not printed) | §4 "Training Details" | verified 2026-09-14 | no ablation reported |
| OpenMath2-Llama3.1-8B | 8B | SFT | Optimizer, weight decay | AdamW; 1e-2 | §4 "Training Details" | verified 2026-09-14 | no ablation reported |
| OpenMath2-Llama3.1-8B | 8B | SFT | Learning rate | constant 2e-5 | §4 "Training Details" | verified 2026-09-14 | no ablation reported |
| OpenMath2-Llama3.1-8B | 8B | SFT | Epochs | 2 | §4 "Training Details" | verified 2026-09-14 | no ablation reported |
| OpenMath2-Llama3.1-8B, -70B | 8B, 70B | SFT | Checkpoint selection | average of 6 equally spaced checkpoints saved during training | §4 "Training Details" | verified 2026-09-14 | App. A.4, Fig. 9: last-4 average > last checkpoint by more than 2 points in one ablation run |
| OpenMath2-Llama3.1-70B | 70B | SFT | Training data | 5M fair-downsampled subset ("Due to computational constraints") | §4 "Training Details" | verified 2026-09-14 | no ablation reported |
| OpenMath2-Llama3.1-70B | 70B | SFT | Learning rate | constant 1e-5; batch 512, AdamW, weight decay 1e-2, 2 epochs as for 8B | §4 "Training Details" | verified 2026-09-14 | no ablation reported |
| OpenMath2-Llama3.1-8B | 8B | SFT | Steps, hardware, averaging | max_steps 60000; 8 nodes × 8 GPUs; tensor parallel 4; checkpoints at 10000, 20000, …, 60000 averaged; lr 2e-5 | NeMo-Skills@2584222 docs/openmathinstruct2/training.md L75-92 (2024-10-17) | verified 2026-09-14 | no ablation reported |
| OpenMath2-Llama3.1-70B | 70B | SFT | Steps, hardware, averaging | max_steps 20000; 32 nodes × 8 GPUs; tensor parallel 8, pipeline parallel 2; checkpoints every 3330 steps averaged; lr 1e-5 | NeMo-Skills@2584222 docs/openmathinstruct2/training.md L97-114 | verified 2026-09-14 | no ablation reported |
| OpenMath2-Llama3.1-8B, -70B | 8B, 70B | SFT | Passes over data | 8B: 60000 × 512 / 13,972,791 = 2.20; 70B: 20000 × 512 / 5,000,000 = 2.05 | inputs: §4 batch 512; NeMo-Skills@2584222 max_steps; dataset card split sizes | derived | not applicable |
| Reproduction commands for NeMo-RL (written 2026) | 8B, 70B | SFT | Sequence length, packing, clipping, schedule | max_total_sequence_length 4096; sequence packing disabled; max_grad_norm 0.0; cosine with min_lr = lr (constant); warmup 0; global batch 512 | NeMo-Skills@24deb162 docs/releases/openmathinstruct2/training.md L60-100, L106-146 (2026-06-08) | verified 2026-09-14 | no ablation reported |
| Ablation student | 8B (Llama3.1-8B-Base) | SFT | Epochs, batch, optimizer, LR, weight decay | 4 epochs; batch 256; AdamW; constant 5e-6; 1e-2; accuracy averaged over 4 runs | §2.2 | verified 2026-09-14 | no ablation reported |
| Llama3.1-405B-Instruct (teacher) | 405B | distill-SFT | Solution sampling in ablations | temperature 1.0, top-p 0.95; 64 samples per question in the format ablation | §2.2; §2.2.1 | verified 2026-09-14 | no ablation reported |
| Llama3.1-405B-Instruct (teacher) | 405B | distill-SFT | Question augmentation | 5 few-shot examples; no difficulty instruction; nucleus sampling | §3 | verified 2026-09-14 | Fig. 6: more unique questions → higher accuracy at 256K pairs |
| Llama3.1-405B-Instruct (teacher) | 405B | distill-SFT | Solutions for new questions | 32 per question; temperature 0.7; majority-vote answer as ground truth; minimum vote threshold 0 | §3; App. C.1 | verified 2026-09-14 | Table 9: threshold 0/8/16/24 → 50.1/49.2/44.4/42.0 (381K/339K/254K/160K pairs; size not matched) |
| OpenMathInstruct-2 | — | distill-SFT | Solution post-processing | multiple \boxed dropped; "My Solution:" prefix removed; truncated after first \boxed sentence; incorrect arithmetic removed; complex arithmetic split; > 1024 Llama3.1 tokens or < 200 characters dropped | App. A.2 | verified 2026-09-14 | no ablation reported |
| OpenMathInstruct-2 | — | distill-SFT | Long questions | 564 questions (about 0.1%) longer than 1024 Llama tokens are not filtered in the release; removal is recommended | dataset card "Note" (huggingface.co/datasets/nvidia/OpenMathInstruct-2) | verified 2026-09-14 | card: no performance drop, "a minor bump" after removal; no numbers given |
| OpenMathInstruct-2 | — | distill-SFT | Decontamination | top-k = 5 by multi-qa-MiniLM-L6-cos-v1 embeddings; Llama3.1-405B-Instruct paraphrase check, both orders; GSM8K, MATH, AMC 2023, AIME 2024 test sets; 569K → 519K | §3.1; App. C.2 | verified 2026-09-14 | Table 10: examples missed by n-gram matching |
| Low-quality filter (ablation only) | 405B judge; 340B RM | distill-SFT | Thresholds | Nemotron-4-340B-Reward helpfulness or correctness ≥ 3 (0-4 scale); two binary judge prompts | §2.2.3 | verified 2026-09-14 | threshold chosen "based on small-scale tuning experiments" (§2.2.3); Table 3: no filter gives a meaningful gain |
| OpenMath2 models | 8B, 70B | eval-gate | Decoding and judging | zero-shot; greedy and majority@256 at temperature 0.7; GPT-4o answer judge | §4 "Evaluation Details" | verified 2026-09-14 | no ablation reported |
| OpenMathInstruct-2 | — | distill-SFT | Samples per original MATH/GSM8K question in the final dataset | not reported | body, appendix, dataset card checked | not reported | — |
| OpenMath2 models | 8B, 70B | SFT | Paper-stated sequence length, warmup, Adam betas, loss masking, GPU type, GPU-hours, teacher-sampling compute | not reported | body, appendix, dataset card, NeMo-Skills@2584222 doc checked | not reported | — |

Note on the NeMo-RL row: the documentation was rewritten for NeMo-RL in 2025-2026, after the October 2024 release, so it is
not shown to be the command that produced the released checkpoints. Note on the "Low-quality filter" row: §2.2.3 states that none of the filtering strategies gave a meaningful gain, and the
paper does not describe applying them to the released dataset. That the release omits them is an inference from §2.2.3 and
App. A.2, which lists the post-processing steps without these filters.

Starting point for a small general-purpose run: this ledger does not support one. All verified SFT values come from
math-only training of Llama3.1-8B/70B-Base on 5M-14M math pairs (64 or 256 GPUs in NeMo-Skills@2584222), with
math-only evaluation.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2410.01560 (v2 PDF); https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
  (README); github.com/NVIDIA/NeMo-Skills docs/openmathinstruct2/training.md at commit 258422269ca9 and
  docs/releases/openmathinstruct2/training.md at commit 24deb162c7a3.
- Corrections: new file; the previous [[openmathinstruct-2]] card had no ledger. Its "K ≈ 32, temperature 1.0, top-p 0.95"
  mixed the new-question setting (32, 0.7) with the ablation setting (1.0, 0.95).
- Removed as unsupported: none in this file.
