<!-- scope: long-context synthesis — LongAlpaca instruction data paired with LongLoRA training
     deps: [[longalign]]
     see-also: [[prolong]], [[long-context-data-engineering]]
-->

# LongAlpaca: Long-context Instruction Data (LongLoRA Companion)
- **Core Insight:** Effective long-context instruction data can be bootstrapped cheaply by (a) collecting 9K long documents across papers, books, and code, (b) prompting ChatGPT / Claude with each full document to produce question + answer pairs, (c) mixing with 3K standard Alpaca-style short samples; paired with LongLoRA's shifted-sparse attention, this lets a Llama-2-7B-LongLoRA-32K reach quality comparable to much more expensively trained long-context models.
- **Guideline:** For a lightweight long-context SFT stack, pair LongLoRA (shifted-sparse attention fine-tuning) with LongAlpaca-12K (9K long + 3K short); this is the most compute-efficient recipe published for 32K → 100K context extension.
- **Authors:** Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia (CUHK + MIT)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2309.12307 (LongLoRA paper) ; https://huggingface.co/datasets/Yukang/LongAlpaca-12k
- **Relevant topics:** long-context SFT, LongLoRA, Alpaca-style instruction data, 32K–100K context

## Abstract
LongAlpaca is the instruction-tuning dataset companion of LongLoRA. It contains 9K long-context question-answer pairs spanning 8K–64K tokens, constructed from 3,000 long documents (papers, books, code). Additionally, 3K short Alpaca-style samples are mixed in to preserve general chat quality. When combined with LongLoRA's shifted-sparse-attention fine-tuning, LongAlpaca enables Llama-2-7B / 13B / 70B to handle 32K–100K context at a fraction of the compute cost of full long-context continued pretraining.

## Key Contributions
- **LongAlpaca-12K dataset** — 9K long + 3K short, public.
- Tight coupling with **LongLoRA's S²-Attention** (shifted-sparse attention) — together forming the first reproducible lightweight long-context stack.
- LongAlpaca-70B reaches 73.3% on LongBench at release (Oct 2023).
- Serves as baseline reference data for subsequent long-context datasets (LongAlign, ProLong).

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Step 1 — Long-document collection:** 3,000 documents across three categories:
  - **Academic papers:** ArXiv full PDFs (computer science, physics, biology).
  - **Books:** public-domain + selected fiction/non-fiction.
  - **Code:** full GitHub repos (Python, Java).
- **Step 2 — Question + answer generation:** for each document, prompt ChatGPT / Claude with the full text + a task-type specification (summarize / QA / extract / analyze). Generate 3 QA pairs per document.
- **Step 3 — Filtering:**
  - Document length ≥ 8K tokens.
  - Generated answer length ≥ 30 tokens.
  - Simple profanity/quality filter.
- **Step 4 — Short-sample mixing:** add 3K random Alpaca samples (under 2K tokens) to preserve short-context chat behavior.
- **Output shape:** 12,000 total samples; 9K span 8K–64K tokens, 3K under 2K tokens.
- **Teacher model(s):** ChatGPT and Claude (Claude preferred for longest docs).
- **Cost:** ~$5K in API at 2023 rates.

## Modality-specific technical details (REQUIRED — long-context)
- **Token-range:** 8K–64K (long split); some tail to 100K.
- **Needle-retrieval difficulty:** modest — questions are not NIAH-style; they require holistic understanding of the document.
- **Document-type mix:** ~40% papers, 30% books, 30% code.
- **Packing strategy:** no explicit packing discussed; standard SFT batch handling.
- **Position-encoding adaptation:** LongLoRA's S²-Attention + position interpolation.
- **Per-stage data mix:** 75% long / 25% short in SFT to maintain balance.

## Quality / diversity evaluation
- LongAlpaca-7B: LongBench Single-Doc QA 28%, Multi-Doc QA 30%, Few-shot 58%, Synthesis 35%, Code 60%.
- LongAlpaca-70B: LongBench overall 46.7% — competitive with GPT-3.5-16K baseline (45.4%).
- Short-context MMLU: within 1 point of base Llama-2.

## Risks + gotchas
- **Small dataset size:** 12K is not enough for aggressive context extension past 64K; later work (LongAlign, ProLong) scales up.
- **ChatGPT teacher quality cap** on hardest long-doc reasoning tasks.
- **No synthetic NIAH-style samples** — trained model can do comprehension but weak explicit retrieval.
- **LongLoRA coupling:** most gains assume the S²-Attention fine-tuning; swapping in vanilla LoRA degrades results.

## Connections
- Sibling paper: LongLoRA (same authors).
- Successor long-context data recipes: [[longalign]], [[prolong]], [[long-context-data-engineering]].
- NIAH-targeted synthesis: [[ruler]].
