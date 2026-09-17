<!-- excerpt for: ch-32e
     source: NVIDIA, "NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model", arXiv:2508.14444v4 (2025-09-02; v1 2025-08)
     scope: pretraining schedule and data phases, Phase LC long-context extension, SFT stages that protect long context, checkpoint interpolation
     library card: none as of 2026-09-15 (proposed slug nemotron-nano-2)
-->

# Nemotron Nano 2 — verified extract (official technical report)

Checked on 2026-09-15 against the arXiv v4 PDF text.

## Architecture (§2.1, Table 1)
- Nemotron-Nano-12B-v2-Base: 62 layers (6 self-attention, 28 FFN, 28 Mamba-2); hidden 5120; GQA 40 query / 8 KV heads. "as in Nemotron-H, we do not use any position embeddings" (§2.1).

## Pretraining (§2.2, §2.5)
- Three data phases: phase 1 promotes diversity; phases 2 and 3 primarily use high-quality data; switches at the 60% and 90% points of training (§2.2).
- 20T-token horizon; sequence length 8192; global batch 768 (6,029,312 tokens); no batch ramp-up; WSD schedule with "stable" LR 4.5e-4 and minimum 4.5e-6; "the learning rate was decayed over the final 3.6 trillion tokens"; weight decay 0.1; Adam β1 0.9, β2 0.95 (§2.5).

## Phase LC (§2.6, Table 4)
- Continuous pretraining at 524,288 tokens with a constant LR of 4.5e-6, after Phase 3 of pretraining.
- "Although the target context length of Nemotron Nano 2 is 128k, in preliminary studies on the Nemotron-H 8B model, we found it better to do CPT with 512k sequence length, instead of 256k or 128k." Stated intuition: longer training sequences lower the chance that long documents are cut by the Concat & Chunk data loader.
- 8-way tensor parallelism, 16-way context parallelism; global batch 12 sequences so that tokens per batch stay "around 6M"; Phase LC = 18.9B tokens.
- Synthetic long-document QA: academic documents longer than 32k tokens; 1,024-token chunks; 10% of chunks given to Qwen-2.5-72B-Instruct to write QA pairs, which are appended to the end of the document.
- Blend: all Phase 3 weights scaled to 80% of their values; the remaining 20% is the long-document QA data. "We found such a blend could effectively extend the context length ... without degrading regular benchmark scores" (no before/after numbers printed).
- Table 4 (Nemotron-H 8B ablation), RULER-128k: train length 128k with synthetic data 73.68; 256k without synthetic 70.19; 256k with synthetic 79.04; 512k with synthetic 81.04.
- Table 5: RULER-128K 84.74 (12B Base), 82.22 (9B Base, pruned and distilled).

## Post-training steps that protect long context (§3.2)
- Stage 1 SFT concatenates samples into sequences of about 128k tokens "to improve efficiency and preserve long-context ability from pretraining". Tool-calling accuracy degraded after Stage 1, attributed to the concatenation; Stage 2 SFT trains without concatenation. Stage 3 SFT "reinforces long-context capability".
- Model merging: linear interpolation (1 − α) · w_model1 + α · w_model2 between a reasoning-strong and a chat-strong RL checkpoint; α swept 0.1 to 0.9 in steps of 0.1; "values around 0.5 offered a good trade-off" (§3.2).
- Table 8: RULER@128k 83.36 for Nemotron-Nano-v2-12B (reasoning on).
