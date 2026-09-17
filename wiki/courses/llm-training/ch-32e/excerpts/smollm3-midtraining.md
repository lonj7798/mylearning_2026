<!-- excerpt for: ch-32e
     source: Hugging Face, "SmolLM3: smol, multilingual, long-context reasoner", huggingface.co/blog/smollm3 (2025-07; markdown source read)
     scope: pretraining stages and decay, long-context extension, reasoning mid-training, model merging
     library card: [[smollm-3]] (predates the 2026-09 verification; see last section)
-->

# SmolLM3 mid-training — verified extract (official blog)

Checked on 2026-09-15 against the blog markdown. Loci are blog section headings.

## Pretraining ("Training Configuration"; "Data mixture and training stages")
- Global batch 2.36M tokens, sequence length 4096, LR 2e-4, AdamW (β1 0.9, β2 0.95), weight decay 0.1, gradient clipping 1; WSD scheduler with 2000 warmup steps and "a linear decay to 0 in the final 10% training steps"; 384 H100 GPUs for 24 days.
- 11.2T tokens in three stages. Stage 1 stable (0T → 8T): web 85%, code 12%, math 3%. Stage 2 stable (8T → 10T): web 75%, code 15%, math 10%. Stage 3 decay (10T → 11.1T): web 63%, code 24%, math 13%, with instruction and reasoning datasets such as OpenMathReasoning.
- Mixture ablations used 3B models trained on 50B to 100B tokens.
- Architecture: NoPE (RoPE removed) in every 4th layer, "improves long context performance without affecting short context capabilities, as confirmed by our ablations"; intra-document masking.

## Long-context extension ("Long Context extension")
> "we trained SmolLM3 on an additional 100B tokens to extend its context length. We sequentially extended the context window in two stages for 50B tokens each: first transitioning from 4k to 32k context with RoPE theta increased to 1.5M, then from 32k to 64k context with RoPE theta increased to 5M. Both stages upsampled math, code, and reasoning data."

- Upsampling code repositories, books and long web pages "didn't further boost performance on RULER and HELMET benchmarks".
- YaRN at inference to 128k ("2x extension beyond the 64k training length").

## Reasoning mid-training ("Reasoning Mid-training")
- 35B tokens from OpenThoughts3-1.2M and a subset of Llama-Nemotron-Post-Training-Dataset-v1.1 (R1 traces); ChatML template; wrapped packing; 4 epochs (~140B tokens).

## Merging ("Model Merging")
- RULER degraded after post-training; the blog traces it to the reasoning mid-training stage, and notes APO data was limited to 24k tokens.
- Recipe: soup of APO checkpoints, then linear merge 0.9 × APO soup + 0.1 × mid-training checkpoint; "We were able to recover the base model's RULER score on contexts up to 128k tokens." No RULER numbers are printed for the recovery.

## Disagreements with the library card [[smollm-3]]
- "Mid-training data composition (only 140B reasoning tokens is given)" → 35B tokens × 4 epochs from two named datasets.
- "Hyperparameters: not itemized in blog summary" → the pretraining hyperparameters are itemized (above).
- Core Insight "APO beats vanilla DPO; mid-training matters more than algorithm choice" → not stated in the blog.
- The card omits the long-context stages, NoPE, and the merge recipe.
