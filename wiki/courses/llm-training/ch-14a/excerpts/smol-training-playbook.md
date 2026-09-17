---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: "The Smol Training Playbook: The Secrets to Building World-Class LLMs (Hugging Face, published Oct. 30, 2025), pre-training sections (no library card as of 2026-09-15; chapter-local verified extract)"
source_url: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
created_at: "2026-09-15"
source_type: official blog for SmolLM2 and SmolLM3 (the organization that trained them); secondary source for other labs' models
---

# Excerpt: Smol Training Playbook — budget, schedule, batch, and stages for SmolLM3

Authors include Loubna Ben Allal, Lewis Tunstall, Nouamane Tazi, Elie Bakouch, Ed Beeching, et al. Section names are the playbook's headings. Numeric ablation results appear in embedded interactive charts; only statements printed in the text are quoted here.

## Ablation setup ("Every Big Model Starts with a Small Ablation")
- "Our main setup is a 1B transformer following the Llama 3.2 1B architecture trained on 45B tokens"; "For the SmolLM3 ablations, we trained the full 3B model on 100B tokens instead of the final 11T."

## "Scaling Laws: How Many Parameters, How Much Data?"
- Size choice: 3B was "large enough for the model to have meaningful capabilities (such as reasoning and tool calling), but small enough to enable super-fast inference and efficient local usage."
- Duration choice: "training duration is often dictated by the amount of compute available. We secured 384 H100s for roughly a month, which provided a budget for training on 11T tokens (assuming a model FLOPs utilization of ~30%)."

## "Learning Rate"
- WSD: "You maintain a constant high learning rate for most of training, then either sharply decay in the final phase (typically the last 10–20% of tokens) for WSD or do discrete drops (steps)".
- "In general, it is recommended to allocate 10–20% of total tokens to the decay phase" (citing Hägele et al.).
- DeepSeek-V3 "transitioned from the constant phase with a cosine decay (from 67% to 97% of training), then applied a brief constant phase before the final sharp step"; "DeepSeek-V2 and V3's technical reports don't include ablations on these schedule changes."
- "Ablation—WSD Matches Cosine": cosine versus WSD with 10% and 20% decay windows; the loss chart's token axis ends at 46e9. "The evaluation results show similar final performance across all three configurations"; "Cosine achieves better loss and evaluation scores during the stable phase (before WSD's decay begins)". "We opted for WSD with 10% decay for SmolLM3."
- Note: "If you're comparing intermediate checkpoints between cosine and WSD during the stable phase, make sure to apply a decay to the WSD checkpoint for a fair comparison."
- Sidenote: GLM-4.5 "mentions that WSD performs worse than cosine decay on general benchmarks (SimpleQA, MMLU), but they don't provide any results."

## Survey table of pre-training settings ("llms-landscape-pretrain")
- Rows quoted: "DeepSeek-V3 | MoE | 671B (37B active) | 14.8T | ... | 2.2×10⁻⁴ | Multi-step + cosine | 2k | 12.6M→62.9M (warmup 469B)"; "OLMo 2 7B | Dense | 7B | 5T | ... | 3×10⁻⁴ | Cosine | 2k | 4.2M"; "SmolLM3 | Dense | 3B | 11T | ... | 2×10⁻⁴ | WSD | 2k | 2.3M".
- The OLMo 2 7B "5T" is the cosine horizon of OLMo 2 Table 3; the OLMo 2 report states 3.90T pre-training tokens and 4.05T in total.

## "Batch Size"
- "DeepSeek-V3 begins with a 12.6M batch for the first ~469B tokens, then increases it to 62.9M for the remainder of training." The DeepSeek-V3 report prints 3072 → 15360 without a unit, "gradually increased" over the first 469B tokens; 3072 × 4,096 = 12,582,912 and 15360 × 4,096 = 62,914,560 are derived with 4K-token sequences.
- "A useful rule of thumb for optimizers like AdamW or Muon is square root LR scaling as batch size grows, but exactly how this works depends on the optimizer."
- "The critical batch size isn't fixed; it grows as training progresses."

## "SmolLM3" (training configuration)
- "We ran learning rate sweeps and settled on 2e-4. For the global batch size, we tested values from 2M to 4M tokens but found minimal impact on the loss or downstream performance, so we chose 2.36M tokens, the size that gave us the best throughput."

## "The Training Marathon" and "Mid-Training"
- A tensor-parallel bug made the 3B model trail expectations; it was found "At around the 1T token mark" and caused "a subtle tensor parallelism bug that forced a restart at 1T tokens".
- For context extension, "starting a fresh learning rate schedule for each stage over 50B tokens worked better than extending context during the last 100B tokens of the main decay phase" (no numbers in text).

## Verification
- Read on 2026-09-15 against the playbook source text (cached 2026-09-14; front matter `published: 'Oct. 30, 2025'`).
- Not reported in the text: loss or benchmark values for the WSD, LR, and batch ablations; the model size used for the WSD ablation (the 46e9 token axis matches the 45B-token 1B setup, which is an inference); seeds.
