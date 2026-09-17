---
chapter: ch-32a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2409.07431v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2409.07431
created_at: "2026-09-15"
---

# Excerpt: Synthetic continued pretraining

**Paper:** Zitong Yang, Neil Band, Shuangping Li, Emmanuel Candès, Tatsunori Hashimoto (Stanford University).
arXiv v1 2024-09; v2 2024-10-03 read. Source type: paper.

## Problem and method (§1, §2)
- Setting: teach a pretrained model the knowledge of a small corpus D_source so that it answers queries without access
  to D_source at test time (parametric knowledge).
- Synthetic CPT: A_synth: D_source → D_synth, then continued pretraining on D_synth (Eq. 1).
- EntiGraph: (1) prompt an LM to extract salient entities {E1, …, En} from a document; (2) prompt it to analyze relations
  among a subset of entities in the context of the document. The experiments generate text for all pairs and triplets (§2.2).

## Setup (§3, §4.1, App. C)
- QuALITY: 265 articles and books, 1.3M tokens (average ~5,000 tokens); 4,609 multiple-choice questions rewritten to name
  the article and author; 5-shot chain-of-thought evaluation.
- LM_aug = gpt-4-turbo; EntiGraph corpus 455M tokens.
- Llama 3 8B Base; context 2048; batch 16; linear warmup 5% of steps; cosine decay; peak LR 5e-6; full-parameter FSDP.
- EntiGraph CPT: 2 epochs; "replay with a rate of 0.1 using 1B RedPajama tokens" (each batch is RedPajama with probability 10%).
- Raw CPT: epochs and replay tuned on a validation split; selected 4 epochs, replay 0.1.
- Rephrase baseline: three rephrase prompts adapted from Maini et al. (2024) and Mecklenburg et al. (2024), stopped at 38M tokens.

## Results
- Closed-book QA: Llama 3 8B Base 39.49%; Raw CPT 38.15%; EntiGraph CPT 56.22%; GPT-3.5 44.81%; GPT-4 51.30% (Fig. 2, §4.2).
- Accuracy scales log-linearly in synthetic tokens up to 455M; Rephrase CPT scales worse (Fig. 2).
- The authors give two possible reasons for Raw CPT being below the base: the narrow corpus "may harm the overall English
  capabilities of the model", and limited diversity of knowledge representations (§4.2, stated as postulates).
- Open book (Table 3): EntiGraph CPT + RAG 62.60%; Llama 3 8B Base + RAG 60.35% (Recall@8 99.63 for both); GPT-4 oracle
  86.09%; GPT-3.5 oracle 72.60%. EntiGraph CPT alone gains 16.73 points versus 20.86 for RAG, "> 80%" of the RAG gain (§5).
- Instruction tuning on 250M UltraChat tokens (peak LR 5e-6, 1 epoch, batch 512): EntiGraph Instruct produces more salient
  claims with few added false claims as summaries get longer; Raw Instruct adds false claims (Fig. 3).

## Limitations (§7.1)
Hallucinated relations are possible; the authors fact-checked a subset for a few books and found no incorrect text; gains
may partly reflect distillation from gpt-4-turbo, though closed-book accuracy exceeds GPT-4; no self-bootstrapping.

## Verification
- Read on 2026-09-15 against arXiv:2409.07431v2 PDF text (§1–§7, App. C, Table 1, Table 3, Figs. 2–3).
- Not reported: general-capability benchmarks (for example MMLU) before and after EntiGraph CPT; the effect of the 0.1
  replay rate versus no replay for the EntiGraph run.
