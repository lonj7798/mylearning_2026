---
chapter: ch-32c
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/gradient-llama3-1048k.md (planned card; not present on 2026-09-15)
source_url: https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k
created_at: "2026-09-15"
---

# Excerpt: Llama-3 8B Gradient Instruct 1048k (model card)

**Organization:** Gradient (with compute from Crusoe Energy); citation authors Leonid Pekelis, Michael Feil, Forrest Moret, Mark Huang, Tiffany Peng.
**Version read:** Hugging Face model card README at revision cd3069b (last modified 2024-10-29).
**Source type:** model card. Training settings are official statements about this model. The card's capability claims are supported only by NIAH heatmap images and a RULER image, with no text values, so ch-32c labels them **anecdotal** and uses the RULER paper and leaderboard for numbers.

## Statements
- "This model extends LLama-3 8B's context length from 8k to > 1040K ... It demonstrates that SOTA LLMs can learn to operate on long context with minimal training by appropriately adjusting RoPE theta. We trained on 830M tokens for this stage, and 1.4B tokens total for all stages, which is < 0.01% of Llama-3's original pre-training data."
- Base: meta-llama/Meta-Llama-3-8B-Instruct. "NTK-aware interpolation ... to initialize an optimal schedule for RoPE theta, followed by empirical RoPE theta optimization"; progressive training on increasing lengths, similar to Large World Model.
- Data: long contexts built by augmenting SlimPajama; chat fine-tuning on an UltraChat-based dataset (update 5/3).
- Infra: EasyContext Blockwise RingAttention on an NVIDIA L40S cluster.
- Evidence shown: NIAH heatmaps (EVAL_MAX_CONTEXT_LENGTH=1040200, EVAL_DEPTH_INTERVAL=0.2, 8-digit numbers, three haystacks) and a RULER image with the claims "behind only GPT-4 and Yi in the retrieval and Q&A tasks" and "smallest parameter model to rank in the top 7 overall".
- Serving note: "vLLM docker image, recommended to load via `--max-model-len 32768`".

## Progressive training table (card)
| Stage | 65K | 262K | 524k | 1048k |
|---|---|---|---|---|
| Initialize from | LLaMA-3 8B | 65K | 262K | 524k |
| RoPE theta | 15.3 M | 207.1 M | 1.06B | 2.80B |
| Batch size × grad. accumulation | 1 × 32 | 1 × 16 | 16 × 1 | 8 × 1 |
| Steps | 30 | 24 | 50 | 50 |
| Total tokens | 62,914,560 | 100,663,296 | 419,430,400 | 838,860,800 |
| Learning rate | 2.00E-05 | 2.00E-05 | 2.00E-05 | 2.00E-05 |
| GPUs | 8 L40S | 32 L40S | 512 L40S | 512 L40S |

## Independent measurements (not from the card)
- RULER leaderboard, github.com/NVIDIA/RULER README at ab17b78: GradientAI/Llama3 (8B) claimed 1M, effective 16K; 4K 92.8, 32K 79.9, 128K 69.5. See [[ruler]].
- RULER paper Table 3: GradientAI/Llama3 (70B) claimed 1M, effective 16K; 128K 72.1.
