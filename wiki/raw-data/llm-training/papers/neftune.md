<!-- scope: NEFTune — scaled uniform noise added to input embeddings during instruction fine-tuning; AlpacaEval effect, overfitting and length analyses, training settings
     deps: [[loss-masking-prompt]]
     see-also: [[alpaca]], [[evol-instruct]], [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]]
-->

# NEFTune: Noisy Embeddings Improve Instruction Finetuning
- **Core Insight:** Adding scaled uniform noise to the input embeddings during fine-tuning raised LLaMA-2-7B's AlpacaEval win rate (GPT-4 judge, against Text-Davinci-003) from 29.79% to 64.69% on Alpaca and by 15.1 points on average over four instruction datasets, while ARC, HellaSwag, MMLU, and TruthfulQA scores stayed stable (Table 1, Figure 3).
- **Guideline:** When fine-tuning a 7B-13B LLaMA-family model on a single-turn instruction dataset of the tested kind (Alpaca, Evol-Instruct, ShareGPT, Open-Platypus), NEFTune with α ∈ {5, 10, 15} improved AlpacaEval on every tested dataset (Table 1, Figure 2); the reported α was chosen on AlpacaEval itself (App. A.1) and safety was not evaluated (§7), so a separate held-out and safety evaluation is needed before adopting it.
- **Authors:** Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, et al. (13 authors; University of Maryland, Lawrence Livermore National Laboratory, New York University)
- **Year:** 2023 (arXiv v1 2023-10; v2 2023-10-10)
- **URL:** https://arxiv.org/abs/2310.05914
- **Source type:** paper
- **Relevant topics:** instruction fine-tuning, SFT regularization, embedding noise, overfitting, response length, AlpacaEval

## Abstract
The paper adds random noise to the embedding vectors during instruction fine-tuning (NEFTune). Standard fine-tuning of LLaMA-2-7B on Alpaca reaches 29.79% on AlpacaEval; with noisy embeddings it reaches 64.69%. Models trained on Evol-Instruct, ShareGPT, and OpenPlatypus improve by about 10, 8, and 8 points. LLaMA-2-Chat, which was already refined with RLHF, also improves when further trained with NEFTune.

## Key Contributions
- A one-hyperparameter augmentation: iid Uniform(−1, 1) noise, scaled by α/√(Ld), added to the embedding output at each training step (§2, Algorithm 1).
- AlpacaEval gains across OPT-6.7B, LLaMA-1 7B, LLaMA-2 7B/13B/70B, LLaMA-2-Chat 7B, and QLoRA runs (Tables 1-3, Figure 2).
- Overfitting analysis: higher training loss, slightly lower held-out loss, and lower ROUGE-L/BLEU overlap with training responses (§5.1, Figures 4-5).
- Controls showing that longer outputs alone do not reproduce the gain (§5.3, Tables 5-6).

## Key Figures/Tables to Study
- **Table 1 / Table 7:** GPT-4-judged AlpacaEval win rates for LLaMA-2 7B on four datasets, with the α used.
- **Figure 4 and Figure 5:** training vs held-out loss; ROUGE-L and BLEU against training responses.
- **Table 4, Table 5, Table 6:** output length, 2-gram repetition, log-diversity; length-forcing baselines; uniform vs Gaussian noise.
- **Table 11 and Table 14:** which parameters must be trainable; win rate by number of epochs.

## Technical Details
- **Noise:** for embeddings X ∈ R^{B×L×d}, X' = X + (α/√(Ld))·ε with ε ~ Uniform(−1, 1)^{B×L×d}; B is batch size, L sequence length, d embedding dimension, α the base noise scale (Algorithm 1). When lengths differ within a batch, the scale is computed per sequence (Algorithm 1 footnote).
- **Magnitude:** the scaling rule is taken from FreeLB-style adversarial training and gives an expected Euclidean norm of about α/√3 for the whole noise tensor of a sequence (§2).
- **Datasets:** Alpaca (Self-Instruct with Text-Davinci-003), Evol-Instruct 70k single-turn, Open-Platypus 25k (about 10% LLM-generated), ShareGPT 70K conversations split toward single-turn (§3.2). Only single-turn data were used because of memory limits (§3.2).
- **Evaluation:** AlpacaEval, 805 instructions, win rate against Text-Davinci-003 with GPT-4 or ChatGPT as judge; ChatGPT was used as a screening judge because of GPT-4 cost (§3.3). OpenLLM Leaderboard tasks via LM-Eval Harness (§3.3).
- **Results:** Table 1 (GPT-4 judge) LLaMA-2 7B → +NEFT: Alpaca 29.79 → 64.69, Evol-Instruct 70.34 → 79.60, ShareGPT 68.74 → 76.28, OpenPlatypus 62.00 → 70.61, average 57.71 → 72.80. Table 2: LLaMA-2-Chat 7B on Evol-Instruct 74.44 → 81.74; LLaMA-2 13B 72.61 → 82.04; LLaMA-2 70B 75.03 → 88.81.
- **QLoRA:** gains are smaller and not monotone in α; e.g. LLaMA-1 30B Alpaca 41.06 → 41.12 / 43.11 / 41.99 for α = 5 / 10 / 15 (Table 3, ChatGPT judge).
- **Comparison:** LLaMA-1-7B Evol-Instruct, ChatGPT judge: baseline 62.30, +NEFT 67.45, +FreeLB after tuning 63.48 (Table 12).
- **Token identity:** projecting noised embeddings to the nearest vocabulary embedding flipped no tokens up to α = 15 in 5,200 Alpaca samples; flips appear at α ≥ 25 (App. A.3, Figure 6).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| All 7B runs (OPT-6.7B, LLaMA-1 7B, LLaMA-2 7B) | 6.7B-7B | SFT | peak LR; optimizer | 5e-5; Adam | arXiv:2310.05914v2 App. A.1 | verified 2026-09-14 | LR sweep on LLaMA-1 7B + Alpaca, "4% improvement over baseline" (A.1); swept values not reported |
| All 7B runs | 6.7B-7B | SFT | epochs | 3 | App. A.1 | verified 2026-09-14 | Table 14 (LLaMA-2 7B Alpaca, ChatGPT judge): +NEFT 55.09 / 62.55 / 62.24 / 60.50 / 58.14 at 1 / 3 / 5 / 7 / 9 epochs |
| All 7B runs | 6.7B-7B | SFT | effective batch (sequences) | 128 (4 GPUs × batch 4 × 8 accumulation steps) | App. A.1 | verified 2026-09-14 | no ablation reported |
| All 7B runs | 6.7B-7B | SFT | max sequence length | 512 tokens; ShareGPT split into 512-token sequences | App. A.1 | verified 2026-09-14 | Table 13: ShareGPT +NEFT 64.22 (split 512) vs 64.35 (split 1024) |
| 7B / 13B runs | 7B, 13B | SFT | precision; hardware | bfloat16; 4× A5000 (7B), 8× A5000 (13B) | App. A.1 | verified 2026-09-14 | no ablation reported |
| LLaMA-2 7B (Table 1/7 runs) | 7B | SFT | NEFT α | Alpaca 5, Evol-Instruct 5, ShareGPT 10, OpenPlatypus 15 | Table 7 | verified 2026-09-14 | best of α ∈ {5, 10, 15} on AlpacaEval with ChatGPT judge (A.1); Table 6 |
| OPT-6.7B / LLaMA-1 7B / LLaMA-2 7B | 7B | SFT | NEFT α (Alpaca / Evol / OpenPlatypus / ShareGPT) | 15/15/5/15; 10/10/15/10; 5/5/15/15 | Table 8 (captioned "Fig 3"; grid matches Figure 2) | verified 2026-09-14 | same selection rule; LLaMA-2 ShareGPT α differs from Table 7 (15 vs 10) |
| LLaMA-2-Chat 7B; LLaMA-2 13B | 7B, 13B | SFT | NEFT α | 5 | Table 9 | verified 2026-09-14 | no ablation reported |
| LLaMA-2 70B | 70B | SFT | LR; weight decay; batch; seq length; epochs; data | 2e-5; 0.1; 64; 2048; 3; Evol-Instruct 70k | App. A.1 | verified 2026-09-14 | copied from Llama 2 fine-tuning settings except sequence length; no ablation |
| LLaMA-2 70B | 70B | SFT | NEFT α | 15 | App. A.1, Table 9 | verified 2026-09-14 | other values not explored "due to computational constraints" (A.1) |
| QLoRA: LLaMA-2 7B/13B, LLaMA-1 30B | 7B-30B | SFT (QLoRA) | epochs; other settings | 1 epoch; QLoRA defaults; 30B: effective batch doubled, LR halved | §4 "NEFTune Works with QLORA" | verified 2026-09-14 | Table 3 reports α = 5, 10, 15 |
| All runs | all | eval-gate | decoding for AlpacaEval | greedy, repetition penalty 1.2 | App. A.2, Table 10 | verified 2026-09-14 | Table 10: four top-p/temperature settings within 2.11 points of greedy for the NEFT model |
Not reported: LR schedule and warmup, weight decay for 7B/13B, gradient clipping, prompt-token loss masking, packing, number of Alpaca examples.

## Findings relevant to generality
- **Broader benchmarks:** ARC, HellaSwag, MMLU, and TruthfulQA scores "remain stable" with NEFT for LLaMA-2 7B (Alpaca, Evol-Instruct, OpenPlatypus) and LLaMA-1 7B (Evol-Instruct) (§4, Figure 3).
- **Overfitting:** on LLaMA-2 7B + Alpaca, NEFT has higher training loss (measured without noise) and slightly lower loss on Evol-Instruct as a held-out set (§5.1, Figure 4); its greedy outputs on training prompts have lower ROUGE-L and BLEU against the reference responses (Figure 5). The authors interpret this as less overfitting to wording, format, and length of the instruction data (Interpretation, §5).
- **Length vs quality:** NEFT lengthens outputs (LLaMA-2 7B Alpaca: 375.22 → 1061.89 characters) with 2-gram repetition 1.49% → 1.72% and log-diversity 15.97 → 16.41 (Table 4). On LLaMA-1 Alpaca-7B (GPT-4 judge), prompting for "long and comprehensive" answers gives 48.01, forcing 250 minimum tokens gives 38.58, baseline 32.36, and NEFT 61.99 (Table 5). Gaussian noise (α = 5) gives longer outputs but lower win rate than uniform noise (Table 6).
- **Where the effect acts:** with attention blocks frozen, NEFT scores 22.17, equal to the un-fine-tuned model; freezing only the embedding or LM head gives 61.06 / 61.12 vs 62.55 (Table 11).
- **Human check:** in 140 AlpacaEval instructions judged by the authors, NEFT was preferred 88 times with 22 ties, a 74.6% win score (§5.4).
- **Limits stated by the authors:** single-judge (GPT-4) evaluation; 70B tested on one dataset; fixed hyperparameters for most runs; no conclusive explanation of why it works (§6); toxicity and refusal behaviour not evaluated, and LLaMA-2-Chat's ability to refrain from toxic output "may be affected" (§4, §7).

## Connections
- [[alpaca]], [[evol-instruct]] — two of the four fine-tuning datasets used.
- [[loss-masking-prompt]] — the SFT loss that NEFTune modifies only at the input; the paper does not state its masking choice.
- [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]] — later SFT recipes; any statement about NEFTune there must be checked against those sources, not this paper.
- [[judge-llm-bias]] — relevant because every headline number here is an LLM-judged win rate that correlates with length.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2310.05914 (arXiv v2, 2023-10-10, including App. A).
- Corrections to the previous card version: "often ~30 pts of AlpacaEval gain" → +34.9 on Alpaca, +7.5 to +9.3 on the other three datasets, +15.1 average (Table 1); "Table 1: AlpacaEval 2 win rates" → AlpacaEval (v1) win rate vs Text-Davinci-003, GPT-4 judge (Table 1 caption); "Figure 4: training loss drops slightly" → training loss is higher with NEFT, held-out loss slightly lower (Figure 4); "α (OPT-6.7B) = 5" → 15 / 15 / 5 / 15 by dataset (Table 8); "Dataset size tested 1K–50K" → Open-Platypus 25k, Evol-Instruct 70k, ShareGPT 70K, Alpaca size not stated (§3.2); "LR / optimizer unchanged from baseline SFT" → the paper sets LR 5e-5, Adam, 3 epochs, batch 128, length 512 for all 7B runs (A.1); "α = 5 on LLaMA, sweep {5, 10, 15}" → α chosen per dataset and model from {5, 10, 15} (Tables 7-9), 70B uses 15; "√(L·d) keeps magnitude independent of model size" → expected total noise norm ≈ α/√3 (§2).
- Removed as unsupported by the source: "one line of code"; "disrupts token-identity overfitting"; the "≤ 100K examples" threshold and "disable for large datasets" advice; "stacks with packing, masking, RLHF"; "uniform keeps tail behavior bounded"; "positional / token-type embeddings left untouched"; "less or no gain on very large SFT sets (Tülu 3 full), already-RLHF'd models, continued pretraining" (the paper reports a gain on RLHF-tuned LLaMA-2-Chat); "compatible with gradient checkpointing, packing, FSDP".
- Not reported by the source: LR schedule, warmup, gradient clipping, loss masking, packing, multi-turn results, safety metrics.
