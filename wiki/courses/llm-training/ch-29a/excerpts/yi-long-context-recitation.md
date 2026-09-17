---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2403.04652v3 (Yi: Open Foundation Models by 01.AI), §7.1 Long Context Modeling, Table 6 (chapter-local verified extract; the yi card has not been re-verified)
source_url: https://arxiv.org/abs/2403.04652
created_at: "2026-09-15"
---

# Excerpt: Yi — 200K extension with recitation-style document QA

- **Authors:** 01.AI
- **Year:** 2024 (arXiv v1 2024-03; v3 2025-01 used here)
- **Source type:** official technical report
- **Used in:** ch-29a §2.3, §3.1

## Continued pretraining (§7.1)
- Hypothesis stated by the authors: "the potential of utilizing information anywhere within the 200K input context is already exist in the
  base model (same as Fu et al.), the continue pretraining phase 'unlocks' such capability".
- Mixture: "(1). original pretraining data ...; (2). length-upsampled long-context data, where the long documents are mostly from books;
  (3). multi-document question-answering synthetic data, where we construct QA pairs where the answer contains a recitation of the related
  paragraph before the answer."
- "We continue pretrain the model on 5B tokens with 4M batch size, which translate to 100 optimization steps."
- Mixture proportions are not reported.

## Supervised fine-tuning (§7.1)
- "we randomly concatenate multiple documents into a sequence, sample one or more paragraphs from the long sequence, and ask a chat model to
  construct question and answer pairs based on the sampled paragraph. One important detail is recitation and rephrasing: before giving the
  answer, we ask the model to recite or paraphrase the original paragraph. This data format encourages the model's retrieval behavior and
  consequently discourages the hallucination behavior".
- No ablation of recitation versus direct answers is reported (Interpretation by the authors).

## Short-context retention (Table 6, MMLU average)
Yi-6B 4K 63.24 → Yi-6B 200K 61.73; Yi-34B 4K 76.32 → Yi-34B 200K 75.56.
