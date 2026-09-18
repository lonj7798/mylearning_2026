<!-- scope: Mistral NeMo release post (Mistral AI, 2024-07-18), with the Mistral-Nemo-Base-2407 / Instruct-2407 model cards, config.json, and NVIDIA's same-day announcement as supporting official sources: 12B, 128k context, Tekken tokenizer, quantisation-aware training for FP8, function calling; no training recipe disclosed
     see-also: [[mistral-large-2]], [[gemma-2]], [[llama-3]], [[mixtral]], [[pixtral-large]]
-->

# Mistral NeMo
- **Core Insight:** Mistral NeMo is a 12B-parameter model with a 128k-token context window, trained jointly by Mistral AI and NVIDIA on 3,072 H100 80GB GPUs and released as base and instruct checkpoints under Apache 2.0; the release states that training "with quantisation awareness" enables "FP8 inference without any performance loss" but shows no measurement (Mistral blog ¶1-2; NVIDIA blog).
- **Guideline:** When a deployment already runs Mistral 7B, the release supports NeMo as a "drop-in replacement" because it keeps a standard Transformer architecture (blog ¶1); when a training recipe is needed, use a source that discloses settings (for example [[llama-3]] or [[gemma-2]]), because this release reports no token count, schedule, data shares, or post-training method.
- **Authors:** Mistral AI team (blog byline), in collaboration with NVIDIA; the model cards list "The Mistral AI Team" (Albert Jiang, Alexandre Sablayrolles, Alexis Tacnet, Alok Kothari, Antoine Roux, Arthur Mensch, et al.)
- **Year:** 2024 (blog published 2024-07-18)
- **URL:** https://mistral.ai/news/mistral-nemo
- **Source type:** official blog (supporting: model/dataset card, released config/code)
- **Relevant topics:** small open models, long context, multilingual pre-training, tokenizer compression, quantisation-aware training, FP8 inference, function calling

## Summary
The post announces Mistral NeMo, a 12B model built with NVIDIA, with a context window of up to 128k tokens (¶1). Base and instruction-tuned checkpoints are released under Apache 2.0 (¶2). The model uses a new tokenizer, Tekken, and is described as trained on function calling and as strong in 11 named languages ("Multilingual Model for the Masses"). The instruct model went through "an advanced fine-tuning and alignment phase" whose method is not described ("Instruction fine-tuning"). Benchmark tables and figures in the post are images; their values are not in the page text.

## Key Contributions
- A 12B base and instruct pair (Mistral-Nemo-Base-2407, Mistral-Nemo-Instruct-2407) under Apache 2.0 (¶2; model cards).
- Tekken, a tiktoken-based tokenizer trained on more than 100 languages, with reported compression gains over the SentencePiece tokenizer of earlier Mistral models ("Tekken" section).
- Quantisation-aware training stated to allow FP8 inference without quality loss (¶2).
- Function-calling training and an instruct model evaluated with GPT-4o as judge (blog; Table 2 caption).
- Release comparison of the base model against Gemma 2 9B and Llama 3 8B (Table 1 caption).

## Key Figures/Tables to Study
- Blog Table 1: base model vs Gemma 2 9B and Llama 3 8B (image; values not transcribed here).
- Blog Figure 1 (multilingual benchmarks) and Figure 2 (Tekken compression rate), both images.
- Model card "Main Benchmarks" and "Multilingual Benchmarks (MMLU)" text tables (numbers below).

## Technical Details
**Architecture.** 40 layers, dim 5,120, head dim 128, 32 attention heads, 8 KV heads (GQA), SwiGLU, rotary embeddings with theta = 1M, vocabulary 2^17 (model cards "Model Architecture"). config.json gives `intermediate_size` 14336, `vocab_size` 131072, `max_position_embeddings` 131072, `sliding_window` null, `tie_word_embeddings` false (config.json @a4477a2). The Base card prints "Hidden dim: 14,436"; the Instruct card, config.json, and params.json print 14,336, so 14,436 is treated as a typo. 32 heads × 128 head dim = 4,096, which differs from the 5,120 model dim (derived from the values above).

**Tokenizer.** Tekken is based on tiktoken and trained on "more than 100 languages" ("Tekken" section). Against the SentencePiece tokenizer of previous Mistral models it is "~30% more efficient" at compressing source code, Chinese, Italian, French, German, Spanish, and Russian, and "2x and 3x more efficient" for Korean and Arabic (same section). Against the Llama 3 tokenizer it compresses text better for "approximately 85% of all languages" (same section). The measurement corpus is not described.

**Training and compute.** Trained with a 128k context window and "a large proportion of multilingual and code data" (model cards "Key features"). Trained with Megatron-LM (part of NVIDIA NeMo) on 3,072 H100 80GB GPUs on DGX Cloud (NVIDIA blog). NVIDIA's post says the training drew on Mistral's expertise "especially on multilinguality, code and multi-turn content" (NVIDIA blog).

**Base model scores (model cards "Main Benchmarks").** HellaSwag 0-shot 83.5%; Winogrande 0-shot 76.8%; OpenBookQA 0-shot 60.6%; CommonSenseQA 0-shot 70.4%; TruthfulQA 0-shot 50.3%; MMLU 5-shot 68.0%; TriviaQA 5-shot 73.8%; NaturalQuestions 5-shot 31.2%. The Instruct card repeats the same table with identical values.

**Instruct model.** Compared with Mistral 7B, the post says the instruct model is "much better at following precise instructions, reasoning, handling multi-turn conversations, and generating code" and gives the numbers only in image Table 2, judged by "GPT4o ... on official references" ("Instruction fine-tuning"). The Instruct card calls the model "a quick demonstration that the base model can be easily fine-tuned" with no moderation mechanisms ("Limitations"). Both cards recommend temperature 0.3 ("Unlike previous Mistral models, Mistral Nemo requires smaller temperatures").

**Deployment.** API name `open-mistral-nemo-2407`; also packaged as an NVIDIA NIM (blog "Links"). The NIM is described as fitting in the memory of a single L40S, RTX 4090, or RTX 4500 GPU and using FP8 for inference (NVIDIA blog).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Mistral-Nemo-Base-2407 | 12B | pretrain | context length | trained with a 128k context window; `max_position_embeddings` 131072 | Base card "Key features"; config.json @a4477a2 | verified 2026-09-14 | no ablation reported |
| Mistral-Nemo-Base-2407 | 12B | pretrain | data mixture | "a large proportion of multilingual and code data"; shares not reported | Base card "Key features"; NVIDIA blog | verified 2026-09-14 (description); shares not reported | no ablation reported |
| Mistral-Nemo-Base-2407 | 12B | pretrain | tokens seen, optimizer, peak/final LR, warmup, schedule, batch, packing | not reported | checked Mistral blog, both model cards, config.json, params.json, NVIDIA blog | not reported | n/a |
| Mistral-Nemo-Base-2407 | 12B | pretrain | compute | 3,072 H100 80GB GPUs on DGX Cloud, Megatron-LM; GPU-hours not reported | NVIDIA blog (2024-07-18) | verified 2026-09-14 | n/a |
| Mistral-Nemo-Base-2407 | 12B | pretrain | quantisation-aware training | "trained with quantisation awareness, enabling FP8 inference without any performance loss"; method, stage, FP8 scores not reported | Mistral blog ¶2 | verified 2026-09-14 (claim); details not reported | no evaluation shown in text |
| Mistral-Nemo-Base-2407 | 12B | pretrain | tokenizer | Tekken (tiktoken-based), trained on >100 languages; vocab 131072 | blog "Tekken"; config.json | verified 2026-09-14 | blog Figure 2 (image) |
| Mistral-Nemo-Instruct-2407 | 12B | SFT; preference | method, data, sizes, epochs, LR, β | "an advanced fine-tuning and alignment phase"; no settings | blog "Instruction fine-tuning"; Instruct card | not reported | n/a |
| Mistral-Nemo-Instruct-2407 | 12B | SFT | function-calling data | "trained on function calling"; amount not reported | blog "Multilingual Model for the Masses" | verified 2026-09-14 | no evaluation numbers in text |
| Mistral-Nemo-Instruct-2407 | 12B | eval-gate | judge | GPT-4o as judge on official references | blog Table 2 caption | verified 2026-09-14 | Table 2 values are an image |

## Findings relevant to generality, long context, agentic training
- **Generality (multilingual).** Base MMLU (5-shot) is 68.0%; the multilingual MMLU table (shot count not stated) gives 59.0%-64.6% in eight other languages: French 62.3, German 62.7, Spanish 64.6, Italian 61.3, Portuguese 63.3, Russian 59.2, Chinese 59.0, Japanese 59.0 (model cards). The cards do not report contamination checks. The post names 11 languages as particular strengths: English, French, German, Spanish, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi.
- **Long context.** The 128k window is stated; no long-context evaluation appears in the text of the post or cards.
- **Agentic training.** Function-calling training is stated without data or scores. The Instruct card requires tool call IDs of exactly 9 alphanumeric characters ("Function calling with transformers").

## Connections
- [[mistral-large-2]] — released 2024-07-24; its post names Mistral NeMo and Mistral Large as the two general-purpose models on la Plateforme.
- [[gemma-2]] and [[llama-3]] — Gemma 2 9B and Llama 3 8B are the comparison models in blog Table 1.
- [[mixtral]] — earlier Mistral open-weight release (Apache 2.0) with a published architecture paper.
- [[pixtral-large]] — later Mistral release (124B) that its post describes as "built on top of Mistral Large 2".

## Verification
- Checked on 2026-09-14 against: https://mistral.ai/news/mistral-nemo (post dated 2024-07-18); https://huggingface.co/mistralai/Mistral-Nemo-Base-2407 (@a4477a2) and https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407 (@04d8a90), including config.json and params.json; https://blogs.nvidia.com/blog/mistral-nvidia-ai-model/ (2024-07-18); https://mistral.ai/news/mistral-large-2407 (for the la Plateforme statement).
- Corrections to the previous card version: URL mistral.ai/it/news/mistral-nemo (Italian page) → canonical English URL; "drop-in production replacement for Mistral 7B-class systems" → "a drop-in replacement in any system using Mistral 7B" (¶1); "12B open model family" → one 12B base and one instruct checkpoint (¶2); "standard architecture ... inside existing Mistral / Llama-style stacks" → standard architecture, drop-in for Mistral 7B; Llama not mentioned (¶1); "Mistral positions it as the small-model counterpart to Mistral Large 2" → the Large 2 post (2024-07-24) names NeMo and Large as the two general-purpose models on la Plateforme; the NeMo post does not mention Large 2.
- Removed as unsupported by the source: "not a toy tier" framing; guideline to "optimize the whole deployment stack together"; "structured interaction" as an instruct capability; "Why it matters" claims ("one of the clearest public examples", serving-economics relevance); Pixtral Large as a "multimodal base" analogue of NeMo; deps line ([[mistral-large-2]], [[gemma-2]] are not prerequisites).
- Not reported by the source: pre-training token count, data shares, optimizer and LR schedule, batch size, SFT/preference method and data, quantisation-aware training method, FP8 benchmark numbers, long-context evaluation, contamination checks.
