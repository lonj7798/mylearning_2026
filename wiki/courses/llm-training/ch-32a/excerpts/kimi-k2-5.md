---
chapter: ch-32a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2602.02276v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2602.02276
created_at: "2026-09-15"
---

# Excerpt: Kimi K2.5: Visual Agentic Intelligence

**Source:** Kimi Team (Moonshot AI). arXiv v1 2026-02; v2 2026-08-07 read. Source type: official technical report. ch-32a uses only the
pre-training continuation and modality-shift results.

## Starting point and stages (§4.1, §4.3, Table 3)
- Foundation: Kimi K2, a 1.04T-total / 32B-activated MoE pre-trained on 15T text tokens with MuonClip.
- "The joint pre-training stage continues from a near-end Kimi K2 checkpoint over additional 15T vision-text tokens at 4K
  sequence length." The data recipe extends K2's distribution "by introducing unique tokens, adjusting data proportions
  with increased weight on coding-related content, and controlling maximum epochs per data source".
- Table 3: ViT training 1T tokens (4096); joint pre-training 15T tokens (4096); joint long-context mid-training
  500B → 200B tokens at 32,768 → 262,144, extended with YaRN.
- §2 states that K2.5 "mixes text and vision tokens with a constant ratio throughout the entire training process"; the
  ratio value is not printed.

## Vision injection ablation (§2.1, Table 1; App. B.1, Fig. 9)
Fixed total vision and text token budgets; text-only pre-training before vision is introduced.
| Injection timing | Vision:Text | Vision knowledge | Vision reasoning | OCR | Text knowledge | Text reasoning | Code |
|---|---|---|---|---|---|---|---|
| Early (0%) | 10:90 | 25.8 | 43.8 | 65.7 | 45.5 | 58.5 | 24.8 |
| Mid (50%) | 20:80 | 25.0 | 40.7 | 64.1 | 43.9 | 58.6 | 24.0 |
| Late (80%) | 50:50 | 24.2 | 39.0 | 61.5 | 43.1 | 57.8 | 24.0 |
- App. B.1: mid- and late-fusion runs show a "dip-and-recover" pattern in text performance when vision data is first
  introduced; early fusion keeps a more stable text curve. The authors attribute the dip to the modality domain shift (Interpretation).

## Text retention after visual RL (§2.2, Table 2)
MMLU-Pro 84.7 → 86.4; GPQA-Diamond 84.3 → 86.4; LongBench v2 56.7 → 58.9 (post-training stage, not pre-training).

## Verification
- Read on 2026-09-15 against the arXiv:2602.02276v2 PDF text (§1–§2, §4.1–§4.4, App. B, Tables 1–3, Fig. 9 caption).
- Not reported: model size and token budget of the Table 1 ablation; the vision:text ratio used for K2.5; base-model
  text benchmarks of K2.5 versus the K2 base; learning-rate schedule for the joint stage.
