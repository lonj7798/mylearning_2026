<!-- scope: where the SFT cross-entropy loss is applied — completion only (instruction tuning) versus instruction plus completion (instruction modelling); the conditions under which including instruction tokens helps
     deps: [[sequence-packing]]
     see-also: [[neftune]], [[packed-vs-unpacked-ablation]], [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]]
-->

# Instruction Tuning With Loss Over Instructions
- **Core Insight:** Applying the cross-entropy loss to the instruction tokens as well as the output tokens (INSTRUCTION MODELLING, IM) raises the mean score over 18 NLP tasks by up to 2.46 points and the AlpacaEval 1.0 win rate by up to 9.23 points over completion-only instruction tuning (IT) on LLAMA-2-7B-BASE, with the largest gains on datasets whose instructions are long relative to their outputs and on small datasets (§4.2, Table 1; Fig. 2).
- **Guideline:** When the instruction-to-output length ratio of the SFT set is high (Code Alpaca, Less MMLU Chat) or the set is small (LIMA-scale, ~1k–13k examples), compute the loss over instruction and output tokens while excluding chat-template tokens, because the authors measure lower test loss and lower training-set BLEU under that setting (§4.3, Fig. 3, Table 2). Otherwise — for example Tulu V2, whose instruction-to-output ratio is about 0.5 — the gain is small (§4.2).
- **Authors:** Zhengyan Shi, Adam X. Yang, Bin Wu, Laurence Aitchison, Emine Yilmaz, Aldo Lipani (University College London; University of Bristol)
- **Year:** 2024 (arXiv v1 2024-05; NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2405.14394 (code: https://github.com/ZhengxiangShi/InstructionModelling)
- **Source type:** paper
- **Relevant topics:** SFT loss design, label masking, instruction tuning, overfitting, low-resource SFT

## Abstract
The paper proposes INSTRUCTION MODELLING (IM), which applies the language-modelling loss to the instruction and prompt segment in addition to the output segment. Across 21 benchmarks the authors report that IM improves performance in many scenarios, on NLP tasks (MMLU, TruthfulQA, HumanEval) and on open-ended generation benchmarks (MT-Bench, AlpacaEval); in the most favourable case IM raises AlpacaEval 1.0 by over 100% relative. Two factors govern the size of the effect: the ratio of instruction length to output length in the training data, and the number of training examples. IM helps most with lengthy instructions paired with brief outputs, and under the Superficial Alignment Hypothesis where few examples are used. The authors attribute the improvement to reduced overfitting, and state that IM is not proposed as a replacement for existing fine-tuning procedures but as guidance for low-resource settings.

## Key Contributions
- Defines IM: loss over all tokens except prompt-template tokens, via an indicator `1(x_t ∉ T)` (§3, Eq. 4).
- Measures IM against IT on 7 instruction-tuning datasets with LLAMA-2-7B-BASE over 18 NLP tasks plus MT-Bench, AlpacaEval 1.0 and 2.0 (§4.1–4.2, Table 1).
- Identifies the two governing factors — instruction/output length ratio and number of training examples — with linear fits over datasets (§4.2, Fig. 2).
- Gives an overfitting account: higher train loss and lower test loss under IM, and lower BLEU against training-set references (§4.3, Fig. 3, Table 2).
- Shows that adding a KL-divergence regulariser to IT reduces the NLP-task degradation but lowers AlpacaEval 2.0 performance (§4.3 #4, Table 3).
- Shows IM composes with NEFTUNE, with dataset-dependent sign (§4.2 #3, Table 4).

## Key Figures/Tables to Study
- **Figure 1** — IT versus IM on 7 datasets: mean over 18 NLP tasks (left) and AlpacaEval 1.0 win rate (right).
- **Figure 2** — improvement on AlpacaEval 1.0 against instruction/output length ratio (fit `y = 18.85x + 25.83`) and against number of training examples (fit `y = −39.37x + 431.74`, ratio held near 10).
- **Figure 3** — train-loss and test-loss distributions; mean 1.45 (IM) vs 1.37 (IT) on LIMA training, 1.17 (IM) vs 1.32 (IT) on a 10% sample of Tulu V2 as test set.
- **Figure 4** — mean over 18 NLP tasks against epoch (2–10), LLAMA-2-7B-BASE.
- **Table 1** — the main IT / NEFTUNE / IM comparison. **Table 2** — BLEU against training references. **Table 3** — KL-loss ablation. **Table 4** — IM + NEFTUNE. **Table 6** — hyperparameters.

## Technical Details

### The two losses
IT computes the negative log-likelihood of the completion `C = {C_1..C_n}` given the instruction `I = {I_1..I_m}`:
`L = − Σ_{j=1..n} log P(C_j | I_1..I_m, C_1..C_{j−1})` (§3, Eq. 2).
IM computes it over the full sequence while excluding template tokens `T`:
`L = − Σ_{t=1..m+n} log P(x_t | x_1..x_{t−1}) · 1(x_t ∉ T)` (§3, Eq. 4).
Symbols: `I` instruction tokens; `C` completion tokens; `x` the concatenated sequence; `T` the set of prompt-template tokens such as `<|user|>` and `<|assistant|>`; `1(·)` the indicator function. Template tokens carry no loss in either objective.

### Measured effects (LLAMA-2-7B-BASE unless stated)
- Mean over 18 NLP tasks: Alpagasus Dolly 3k 46.58 → 48.95 (+2.37); Alpagasus Dolly 9k 45.54 → 48.00 (+2.46); Alpagasus Alpaca 5k 45.29 → 47.47 (+2.18); LIMA 48.79 → 49.60 (+0.81) (§4.2, Table 1).
- AlpacaEval 1.0 win rate: Alpagasus Dolly 9k 21.54 → 30.77 (+9.23); Less BBH ICL 36.20 → 44.15 (+7.95); Less MMLU Chat 4.42 → 9.78; Less Tydiqa 5.12 → 10.10; LIMA 33.06 → 32.94 (−0.12) (§4.2, Table 1).
- Length-ratio dependence: Science Literature has a ratio of 24.7 and benefits; Tulu V2 has a ratio of about 0.5 and benefits less (§4.2 #2).
- Overfitting proxies: BLEU of greedy generations against training references falls under IM on all 7 datasets, for example Less BBH ICL 60.96 → 53.94 and LIMA 18.15 → 17.30 (§4.3 #2, Table 2).
- Instruction-tuning tax: over epochs 2–10 the mean over 18 NLP tasks declines for both methods, less steeply for IM (§4.3 #3, Fig. 4).
- IM + NEFTUNE on AlpacaEval 1.0: Less Tydiqa 10.10 → 23.41 (+13.31), Alpagasus Alpaca 5k 19.52 → 32.07 (+12.55), LIMA 32.94 → 30.77 (−2.17); NLP mean falls on Less MMLU Chat (−0.11) and Less BBH ICL (−0.53) (§4.2 #3, Table 4).
- Model families tested: LLAMA-2-7B-BASE, LLAMA-2-13B-BASE, OPT-6.7B (§4.1; Fig. 5).

### Adjacent result, different paper
The choice of an intermediate instruction-loss weight (`α · L_instruction + L_output`) is studied by Huerta-Enochian, "Instruction fine-tuning: does prompt loss matter?" (2024), cited here as [29]: the instruction loss ratio matters for short-completion data and is irrelevant for long-completion data (§2). This card's earlier version credited that result to Shi et al.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LLAMA-2-7B-BASE / 13B-BASE / OPT-6.7B (IM and IT runs) | 7B, 13B, 6.7B | SFT | total batch size (sequences) | 128 | arXiv:2405.14394v2 App. C Table 6 | verified 2026-09-18 | no ablation reported |
| same | same | SFT | epochs | 2, 3, or 10 (2 typical) | App. C Table 6 and text | verified 2026-09-18 | §4.3 Fig. 4 reports NLP mean per epoch 2–10 |
| same | same | SFT | peak LR / schedule / warmup | 2 × 10⁻⁵, linear with warmup, warmup proportion 0.03 | App. C Table 6 | verified 2026-09-18 | no ablation reported |
| same | same | SFT | optimizer | AdamW, β = (0.9, 0.98), ε = 1e-6, weight decay 0 | App. C Table 6 | verified 2026-09-18 | no ablation reported |
| same | same | SFT | max sequence length | 2048 | App. C Table 6 | verified 2026-09-18 | no ablation reported |
| same | same | SFT | precision / hardware | bf16; 2 or 4 A100 80G, or 2 A6000 48G; DeepSpeed ZeRO-3 without offload; Flash Attention | App. C Table 6 and Implementation Details | verified 2026-09-18 | no ablation reported |
| same | same | SFT | loss masking | IT: output tokens only. IM: instruction + output, template tokens excluded | §3 Eq. 2, Eq. 4 | verified 2026-09-18 | §4.2 Table 1 compares the two across 7 datasets |

## Findings relevant to generality
- IM raises the mean over 18 held-out NLP tasks on 6 of the 7 datasets in Table 1, so the gain is not confined to the chat-style evaluation (§4.2, Table 1).
- The generality mechanism the authors give is reduced overfitting: lower test loss on a held-out Tulu V2 sample and lower BLEU overlap with training references, at a higher training loss (§4.3, Fig. 3, Table 2).
- A KL-divergence penalty toward the base model reduces the NLP-task drop but lowers AlpacaEval 2.0 scores, so it does not substitute for IM (§4.3 #4, Table 3).
- Stated limits: the effect depends on the quality and diversity of the instructions, and the method is not proposed as a replacement for standard fine-tuning (Abstract; Limitations).
- Not tested here: multi-turn conversations, models above 13B, and packing interactions.

## Connections
- [[sequence-packing]] — packing changes which tokens fall in one loss window; this paper does not measure that interaction.
- [[neftune]] — the embedding-noise baseline the paper compares with and combines with (Table 4).
- [[packed-vs-unpacked-ablation]] — separate evidence on masking and packing implementation.
- [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]] — completion-only masking as shipped in SFT stacks; these are practice sources, not evidence for or against IM.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2405.14394 (arXiv v2, 2 Oct 2024)
- Corrections to the previous card version:
  - "Response-only loss is strictly better on helpfulness benchmarks … full-sequence loss can help in the tiny-dataset regime" → the paper reports the opposite direction: loss over instructions improves both NLP and open-ended benchmarks in many settings, most strongly at high instruction/output ratio and few examples (Abstract; §4.2, Table 1; Fig. 2).
  - Title "Prompt-Masked vs Full-Sequence Loss in Supervised Fine-Tuning" and a composite author line (Taori 2023, Ouyang 2022, HF handbook) → the card now describes the single artifact its URL names, Shi et al., "Instruction Tuning With Loss Over Instructions".
  - "Upweighting … gives modest gains in some ablations (Shi 2024)" → the instruction-loss-weight result is Huerta-Enochian 2024, cited by Shi et al. as [29] (§2).
  - "Shi 2024 Table 2: MT-Bench delta with and without instruction loss across dataset sizes" → Table 2 reports average BLEU against training references, not MT-Bench.
- Removed as unsupported by the source: the fabricated abstract paragraph; the multi-turn masking rule (mask all user turns and assistant turns 1..k−1, train on turn k); the per-turn unrolling variant; the claim that full-sequence loss "wastes capacity" on a distribution never produced at inference; the `train_on_response_only=True` chat-template API; the claim that incorrect packing plus masking "silently degrades SFT"; the Python masking sketch (implementation detail not taken from this paper).
- Not reported by the source: multi-turn loss masking, packing interactions, models above 13B, preference-optimization stages.
