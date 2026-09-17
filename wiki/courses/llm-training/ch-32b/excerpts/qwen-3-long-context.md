---
chapter: ch-32b
course: llm-training
phase: read
excerpt_of: arXiv:2505.09388v1 (Qwen3 Technical Report) and the Qwen/Qwen3-8B model card README (chapter-local verified extract; the library card qwen-3 does not record these details)
source_url: https://arxiv.org/abs/2505.09388 ; https://huggingface.co/Qwen/Qwen3-8B
created_at: "2026-09-15"
---

# Excerpt: Qwen3 — long-context stage, RULER by mode, and the static-YaRN caution

- **Authors:** Qwen Team
- **Year:** 2025 (arXiv v1 2025-05)
- **Source type:** official technical report; official model card
- **Used in:** ch-32b §3, §5.1, §7, Common mistakes

## Pre-training stage 3 (report §3.2)
"(3) Long Context Stage: In the final pre-training stage, we collect high-quality long context corpora to extend
the context length of Qwen3 models. All models are pre-trained on hundreds of billions of tokens with a sequence
length of 32,768 tokens. The long context corpus includes 75% of text between 16,384 to 32,768 tokens in length,
and 25% of text between 4,096 to 16,384 in length. Following Qwen2.5 (Yang et al., 2024b), we increase the base
frequency of RoPE from 10,000 to 1,000,000 using the ABF technique (Xiong et al., 2023). Meanwhile, we introduce
YARN (Peng et al., 2023) and Dual Chunk Attention (DCA, An et al., 2024) to achieve a four-fold increase in
sequence length capacity during inference."

## RULER by mode (report App. A.1.1, Table 23; YaRN factor 4; thinking budget 8192)
| Model | Mode | Avg | 4K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| Qwen3-8B | non-thinking | 89.1 | 96.3 | 91.2 | 82.1 | 77.4 |
| Qwen3-8B | thinking | 84.4 | 94.7 | 80.8 | 78.3 | 72.0 |
| Qwen3-235B-A22B | non-thinking | 95.0 | 97.7 | 95.1 | 93.3 | 90.6 |
| Qwen3-235B-A22B | thinking | 92.2 | 95.1 | 92.3 | 92.0 | 86.0 |

"In thinking mode, the model's performance slightly degrades. We hypothesize that the thinking content does not
provide significant benefits for these retrieval tasks ... and may instead interfere with the retrieval process."

## Model card caution (Qwen/Qwen3-8B README, "Processing Long Texts")
- Serving example: `--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072`
- "All the notable open-source frameworks implement static YaRN, which means the scaling factor remains constant
  regardless of input length, **potentially impacting performance on shorter texts.** We advise adding the
  `rope_scaling` configuration only when processing long contexts is required. It is also recommended to modify
  the `factor` as needed. For example, if the typical context length for your application is 65,536 tokens, it
  would be better to set `factor` as 2.0."
- "If the average context length does not exceed 32,768 tokens, we do not recommend enabling YaRN in this
  scenario, as it may potentially degrade model performance."
- "The endpoint provided by Alibaba Model Studio supports dynamic YaRN by default."

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2505.09388 (v1 PDF: §3.2, App. A.1.1 Table 23) and
  https://huggingface.co/Qwen/Qwen3-8B/raw/main/README.md (as served on 2026-09-15).
- Not reported by the source: exact token count of the long-context stage ("hundreds of billions"), its LR and
  batch size, and any measured short-text degradation under static YaRN (the model card gives a caution, not a number).
