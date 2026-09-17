---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Wang, Wang, Wang, Li, Hovy, Guo — "Packing Analysis: Packing Is More Appropriate for Large Models or Datasets in Supervised Fine-tuning"
source_url: https://arxiv.org/abs/2410.08081
created_at: "2026-09-15"
note: "No library card for this artifact on 2026-09-15; written from the primary PDF (arXiv v3, 2024-11-06)."
---

# Excerpt: Wang et al. 2024 — Packing Analysis for SFT

**Artifact:** arXiv:2410.08081 (v1 2024-10; read as v3).

## Setup (§3-4)

- Compares padding, random packing, and greedy packing for SFT of LLaMA-3-8B and LLaMA-3-70B on datasets from 69K to about 1.2M conversations.
- Settings (Table 2): LR 1e-5, maximum sequence length 4,096, warm-up ratio 0.2, 4 epochs (8B) or 3 epochs (70B), per-GPU batch 2 (8B) or 1 (70B), gradient accumulation 2, DeepSpeed stage 3, 4 nodes of 8 A800-80GB GPUs. Loss is computed only on tokens after the assistant header (§4.1.2).
- Packed samples are separated by [EOS] (§3). The paper does not state that attention is reset between packed conversations.

## Results

- Average benchmark score, padding / random packing / greedy packing (Table 3): WildChat (GPT-4) 69K at 8B, 49.58 / 49.46 / 50.6; the same data at 70B, 61.50 / 65.97 / 65.92; Open-source 1M at 8B, 54.3 / 54.95 / 55.05; at 70B, 66.12 / 67.26 / 67.54. Greedy packing averaged above padding in all 8 model-dataset settings; random packing averaged below padding in 2 of 8.
- Training time for 70B on WildChat (GPT-4): 9,533 s with padding vs 3,749 s with random packing (Table 5).
- With packing, the linear relation between batch size and learning rate observed for padding does not hold, because scaling the batch by k does not scale the number of conversations by k (§5.3).
- Packing a single-turn-only dataset (filtered 200K OpenHermes 2.5) on LLaMA-3-8B caused a significant MATH drop; adding multi-turn conversations at 1/40 to 1/20 of the data restored performance (§5.3, Fig. 3). The authors' explanation is that packing creates "fake" multi-turn conversations, and multi-turn data teaches the model when to use context.
- The authors suggest padding may be more time-efficient for small models (6-9B) on small datasets (20K-30K) (§5.2).

## Limits

No seeds or run-to-run variance reported; attention-reset behaviour not stated.
