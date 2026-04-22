<!-- scope: Loss masking for SFT — mask prompt tokens so only completion contributes to CE loss
     deps: [[sequence-packing]]
     see-also: [[neftune]], [[packed-vs-unpacked-ablation]], [[hf-alignment-handbook]]
-->

# Prompt-Masked vs Full-Sequence Loss in Supervised Fine-Tuning
- **Core Insight:** Computing cross-entropy over the whole sequence (prompt + completion) makes the model learn to *generate* the prompt, wasting capacity on a distribution it never produces at inference; masking prompt tokens out of the loss (label = -100) keeps the gradient focused on the response distribution.
- **Guideline:** For instruction SFT, always mask the prompt (and system / chat template) tokens out of the loss; keep completion tokens only. For continued pretraining or multi-turn coherent text, full-sequence loss is fine.
- **Authors:** (no single paper; canonical practice documented in Taori 2023 (Alpaca), Ouyang 2022 (InstructGPT SFT), and the HF Alignment Handbook; formal ablations in Shi 2024, "Instruction Tuning with Loss Over Instructions")
- **Year:** 2023–2024
- **URL:** https://arxiv.org/abs/2405.14394 (Shi 2024 "Instruction Tuning With Loss Over Instructions")
- **Relevant topics:** SFT loss design, label masking, instruction tuning, multi-turn masking

## Abstract (from Shi 2024)
The standard SFT loss masks the instruction portion and computes cross-entropy only on the response. Some prior work ("loss over instructions") includes instruction tokens in the loss. We systematically ablate the two choices across Alpaca, ShareGPT, and LIMA-sized datasets and across model scales. Response-only loss is strictly better on helpfulness benchmarks (MT-Bench, AlpacaEval) for typical instruction datasets; full-sequence loss can help in the *tiny-dataset / strong-base-model* regime where it acts as a mild continued-pretraining regularizer.

## Key Contributions
- Systematic comparison: response-only vs full-sequence vs prompt-weighted SFT losses.
- Best practice: mask the prompt and the assistant-role tokens of prior turns; train only on the current assistant turn tokens in multi-turn.
- Multi-turn handling: mask *all* user turns plus assistant turns 1..k−1 when computing loss on assistant turn k.

## Key Figures/Tables to Study
- **Shi 2024 Table 2:** MT-Bench delta with and without instruction loss across dataset sizes.
- **HF Alignment Handbook `apply_chat_template` with `tokenize=True, train_on_response_only=True`.**

## Technical Details

### Standard SFT loss (response-only)
For a conversation `(prompt p_{1:T_p}, response y_{1:T_y})`:
`L_SFT(θ) = −(1 / T_y) Σ_{t=1..T_y} log π_θ(y_t | p, y_<t)`
Prompt tokens' labels are set to `-100` (ignore_index in PyTorch CE) so their gradient contribution is zero.

### Full-sequence loss (not recommended for instruction SFT)
`L_full(θ) = −(1 / (T_p + T_y)) Σ_t log π_θ(x_t | x_<t)`
Trains the model to reproduce the user's prompt — never used at inference, wastes capacity.

### Multi-turn chat masking
For a conversation with turns `[u_1, a_1, u_2, a_2, …, u_k, a_k]`:
- Mask **all** user turns.
- Mask **all** prior assistant turns (a_1..a_{k−1}) — they are part of the prompt when generating a_k.
- Train on a_k tokens only.

Per-turn-training variant: unroll the conversation k times, each time masking through a_{i−1} and training on a_i → k× more data but identical loss value.

### Implementation (Python sketch)
```
labels = input_ids.clone()
labels[:prompt_len] = -100  # mask prompt
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

### Packed-sequence interaction
In a packed block, every sub-sequence has its own (prompt, response) split → the label mask must be reset per sub-sequence. Incorrect packing + masking is a common bug that silently degrades SFT.

### Why not upweight the response?
- Upweighting (e.g., loss = α·L_prompt + L_response with α < 1) gives modest gains in some ablations (Shi 2024), but the response-only baseline dominates across most dataset sizes and is simpler.

## Connections
- Packing companion: [[sequence-packing]] — packing + masking are always paired in modern SFT stacks.
- Noise injection on top: [[neftune]].
- Ablation references: [[packed-vs-unpacked-ablation]].
- Handbook recipe: [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]].
