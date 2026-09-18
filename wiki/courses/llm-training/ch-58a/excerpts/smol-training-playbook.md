---
chapter: ch-58a
course: llm-training
phase: read
excerpt_of: "The Smol Training Playbook: The Secrets to Building World-Class LLMs (Hugging Face, published Oct. 30, 2025), the sections on budget, schedule, the cross-lab survey table, and mid-training"
source_url: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
created_at: "2026-09-17"
note: "No library card exists for this slug as of 2026-09-17. Read on 2026-09-15 in the cached playbook text for the ch-14a excerpt of the same sections; reassembled for ch-58a."
---

# Excerpt: Smol Training Playbook — budget setting, and a survey table that copies a horizon as a budget

Authors include Loubna Ben Allal, Lewis Tunstall, Nouamane Tazi, Elie Bakouch, Ed Beeching, et al. Source type: official
blog for SmolLM2 and SmolLM3 (the organization that trained them); for other labs' models it is a secondary source.

## How the budget was set
- "training duration is often dictated by the amount of compute available. We secured 384 H100s for roughly a month,
  which provided a budget for training on 11T tokens (assuming a model FLOPs utilization of ~30%)."
- Size: 3B was "large enough for the model to have meaningful capabilities (such as reasoning and tool calling), but
  small enough to enable super-fast inference and efficient local usage."
- "Our main setup is a 1B transformer following the Llama 3.2 1B architecture trained on 45B tokens"; "For the SmolLM3
  ablations, we trained the full 3B model on 100B tokens instead of the final 11T."

## Schedule and batch
- "In general, it is recommended to allocate 10–20% of total tokens to the decay phase" (citing Hägele et al.).
- "Ablation—WSD Matches Cosine": cosine against WSD with 10% and 20% decay windows; "The evaluation results show similar
  final performance across all three configurations"; "Cosine achieves better loss and evaluation scores during the
  stable phase (before WSD's decay begins)"; "We opted for WSD with 10% decay for SmolLM3." The numeric results appear
  only in embedded interactive charts.
- "If you're comparing intermediate checkpoints between cosine and WSD during the stable phase, make sure to apply a
  decay to the WSD checkpoint for a fair comparison."
- "We ran learning rate sweeps and settled on 2e-4. For the global batch size, we tested values from 2M to 4M tokens but
  found minimal impact on the loss or downstream performance, so we chose 2.36M tokens, the size that gave us the best
  throughput."
- On other labs: DeepSeek-V3 "transitioned from the constant phase with a cosine decay (from 67% to 97% of training),
  then applied a brief constant phase before the final sharp step"; "DeepSeek-V2 and V3's technical reports don't include
  ablations on these schedule changes."

## The cross-lab survey table ("llms-landscape-pretrain")
Rows quoted: "DeepSeek-V3 | MoE | 671B (37B active) | 14.8T | ... | 2.2×10⁻⁴ | Multi-step + cosine | 2k |
12.6M→62.9M (warmup 469B)"; "OLMo 2 7B | Dense | 7B | 5T | ... | 3×10⁻⁴ | Cosine | 2k | 4.2M"; "SmolLM3 | Dense | 3B |
11T | ... | 2×10⁻⁴ | WSD | 2k | 2.3M".

The OLMo 2 7B cell "5T" is the cosine horizon of OLMo 2 Table 3, not a token count: the OLMo 2 report states 3.90T
pre-training tokens and 4.05T in total ([[olmo-2-stage-chain]] §2.3). The DeepSeek-V3 batch cells are derived: the V3
report prints 3072 → 15360 without a unit, and 3072 × 4,096 = 12,582,912 and 15360 × 4,096 = 62,914,560 assume 4K-token
sequences ([[deepseek-v3-recipe]]).

## Mid-training and long context
- A tensor-parallel bug was found "At around the 1T token mark" and "forced a restart at 1T tokens".
- For context extension, "starting a fresh learning rate schedule for each stage over 50B tokens worked better than
  extending context during the last 100B tokens of the main decay phase" (no numbers in the text).

## Verification
- Read on 2026-09-15 against the playbook source text (cached 2026-09-14; front matter `published: 'Oct. 30, 2025'`);
  reassembled for ch-58a on 2026-09-17 without changing a value or locus.
- Not reported in the text: loss or benchmark values for the WSD, learning-rate and batch ablations; the model size used
  for the WSD ablation; seeds.
