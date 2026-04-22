<!-- scope: NEFTune — add uniform noise to input embeddings during SFT; large AlpacaEval gains for free
     deps: []
     see-also: [[loss-masking-prompt]], [[sequence-packing]]
-->

# NEFTune: Noisy Embeddings Improve Instruction Finetuning
- **Core Insight:** Adding bounded uniform noise to the input-embedding vectors at every SFT step acts as a regularizer that disrupts token-identity overfitting on small instruction sets — one line of code, often ~30 pts of AlpacaEval gain.
- **Guideline:** For any SFT run on instruction data ≤ 100K examples, enable NEFTune with α = 5–15 (scaled by 1/√(L·d)); disable for very long SFT runs or large datasets where overfitting isn't the dominant failure.
- **Authors:** Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2310.05914
- **Relevant topics:** SFT regularization, embedding perturbation, instruction tuning, overfitting

## Abstract
We show that language model finetuning can be improved, sometimes dramatically, with a simple augmentation. NEFTune adds noise to the embedding vectors during training. Standard finetuning of LLaMA-2-7B using Alpaca achieves 29.79% on AlpacaEval, which rises to 64.69% using noisy embeddings. NEFTune also improves over strong baselines on modern instruction datasets: Evol-Instruct (~10%), ShareGPT (~8%), and OpenPlatypus (~8%). Even powerful RLHF-tuned models such as LLaMA-2-Chat benefit from additional training with NEFTune.

## Key Contributions
- Dead-simple augmentation: sample uniform noise once per forward and add to embeddings pre-transformer.
- Strong empirical gains on AlpacaEval across Alpaca, Evol-Instruct, ShareGPT, OpenPlatypus.
- Complementary to every other SFT technique — stacks with packing, masking, RLHF.
- No new hyperparameters except α; robust across α ∈ [5, 15].

## Key Figures/Tables to Study
- **Table 1:** AlpacaEval 2 win rates on LLaMA-2-7B — 29.79 → 64.69 for plain Alpaca.
- **Figure 4:** Training loss drops slightly (regularizer), eval win rate rises sharply — classic regularization signature.
- **Section 3:** The one-liner formula.

## Technical Details

### Noise formula
At each training step, for an input embedding matrix `X ∈ R^{L × d}`:
`X_tilde = X + ε`
`ε_{i,j} ~ Uniform(−1, 1) · (α / √(L · d))`
where
- L = current sequence length (post-packing or pre-packing, per sub-sequence).
- d = embedding dimension.
- α = scale hyperparameter (paper's main: α = 5 on LLaMA, sweep {5, 10, 15}).

Noise is resampled each forward pass; no noise at inference.

### Intuition
- √(L · d) normalization: noise magnitude in the embedding subspace is independent of model size / context length.
- Uniform (not Gaussian) per-element: keeps tail behavior bounded.
- Only applied to input embeddings; positional / token-type embeddings left untouched.

### Hyperparameters (paper)
| Knob | Value |
|------|-------|
| α (LLaMA-2-7B / 13B) | 5 |
| α (OPT-6.7B) | 5 |
| Dataset size | tested 1K–50K |
| LR / optimizer | unchanged from baseline SFT |
| Other SFT hparams | unchanged |

### When it helps vs doesn't
- **Helps:** small/medium instruction datasets (Alpaca, Evol-Instruct, ShareGPT); early-stopped SFT; models that otherwise overfit.
- **Less or no gain:** very large SFT sets (Tülu 3 full), already-RLHF'd models with tightly-controlled loss, continued pretraining.

### Cost
Free: one tensor add, no extra forward/backward. Compatible with gradient checkpointing, packing, FSDP.

## Connections
- Standard SFT loss definition: [[loss-masking-prompt]].
- Throughput companion: [[sequence-packing]].
- Handbook integration: [[hf-alignment-handbook]].
- Tülu 3 ablations discussing NEFTune: [[allenai-tulu-sft-recipe]].
