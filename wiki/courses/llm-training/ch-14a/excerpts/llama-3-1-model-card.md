---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: "meta-llama/llama-models models/llama3_1/MODEL_CARD.md (main branch, read 2026-09-15)"
source_url: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
created_at: "2026-09-15"
---

# Excerpt: Llama 3.1 model card — token count per released size

Source type: model card (official, Meta). The report itself is [[llama-3]]; its recipe ledger is [[llama-3-recipe]].

## Model information table
- Columns: Training Data, Params, Input modalities, Output modalities, Context length, GQA, Token count, Knowledge cutoff.
- Rows 8B, 70B, 405B, each "Multilingual Text" in, "Multilingual Text and code" out, context length 128k, GQA "Yes". The "Token count" cell "15T+" and the knowledge cutoff "December 2023" are printed once for the Llama 3.1 (text only) block that spans the three sizes.
- Note below the table: "Token counts refer to pretraining data only. All model versions use Grouped-Query Attention (GQA) for improved inference scalability."

## Training energy use table
- "Training utilized a cumulative of 39.3M GPU hours of computation on H100-80GB (TDP of 700W) type hardware"; "Training time is the total GPU time required for training each model".
- Training Time (GPU hours): Llama 3.1 8B 1.46M; Llama 3.1 70B 7.0M; Llama 3.1 405B 30.84M; Total 39.3M.
- The card does not state whether these hours include post-training.

## Training data section
- "Overview: Llama 3.1 was pretrained on ~15 trillion tokens of data from publicly available sources."
- "Data Freshness: The pretraining data has a cutoff of December 2023."

## Relation to the report
- The report states 15.6T text tokens for the 405B model and "a corpus of about 15T multilingual tokens"; it does not print a token count for 8B or 70B (arXiv:2407.21783v3 §1; see [[llama-3-recipe]] row "Tokens").
- The model card's "15T+" is a lower bound shared by the three sizes; it does not give an exact count for 8B.

## Verification
- Read on 2026-09-15 against the file on `main` of github.com/meta-llama/llama-models (fetched 2026-09-15; the tables are HTML inside Markdown).
- Not reported by the card: per-size tokens seen, learning rate, batch, or schedule; the scope of the GPU-hour counts.
