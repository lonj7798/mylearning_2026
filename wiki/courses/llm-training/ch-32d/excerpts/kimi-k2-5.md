---
chapter: ch-32d
course: llm-training
phase: read
excerpt_of: Kimi Team (Moonshot AI), "Kimi K2.5: Visual Agentic Intelligence", arXiv:2602.02276v2 (2026-08-07)
source_url: https://arxiv.org/abs/2602.02276
created_at: "2026-09-15"
source_type: official technical report
---

# Excerpt: Kimi K2.5 — training stages and where code, agent, and trajectory data appear

No library card for this source exists at the time of writing (planned slug `kimi-k2-5`). Only passages used by [[read]] §1 and §10 are extracted.

## Training stages (§4.3, Table 3)

| Stage | Data | Sequence length | Tokens | Trainable |
|---|---|---|---|---|
| ViT training | alt text, synthesis caption, grounding, OCR, video | 4,096 | 1T | ViT |
| Joint pre-training | text, knowledge, interleaving, video, OS screenshot | 4,096 | 15T | ViT and LLM |
| Joint long-context mid-training | high-quality text and multimodal, long text, long video, reasoning, long-CoT | 32,768 → 262,144 | 500B → 200B | ViT and LLM |

- "The joint pre-training stage continues from a near-end Kimi K2 checkpoint over additional 15T vision-text tokens at 4K sequence length. The data recipe extends Kimi K2's pre-training distribution by introducing unique tokens, adjusting data proportions with increased weight on coding-related content, and controlling maximum epochs per data source. The third stage performs long-context activation with integrated higher-quality mid-training data, sequentially extending context length via YaRN interpolation." (§4.3)

## Pre-training data (App. B.2, B.3)

- Text: "We upweighted code-centric data, significantly expanding (1) repository-level code supporting cross-file reasoning and architectural understanding, (2) issues, code reviews and commit histories from the internet capturing real-world development patterns, and (3) code-related documents retrieved from PDF and webtext corpora." (B.2)
- Vision corpus includes seven categories, one of which is agent data: "For agentic and temporal understanding, we collect GUI screenshots and action trajectories across desktop, mobile, and web environments, including human-annotated demonstrations." (B.3)

## Post-training observation (§1, §2.2)

- "We find that adding human-designed visual trajectories at this stage hurts generalization. In contrast, text-only SFT performs better". §2.2: "our preliminary experiments show that text-vision SFT yields much worse performance on visual, agentic tasks, possibly because of the lack of high-quality vision data." No numbers are printed.

## Verification

- Checked on 2026-09-15 against https://arxiv.org/abs/2602.02276 (v2), §1, §2.1-§2.2, §4.3, App. B.
- Not reported by the source: the token share of GUI action trajectories; whether any of them are model-generated; the share of repository, issue, and commit data; which stage holds the GUI trajectories beyond "multimodal pre-training corpus"; any ablation of agent data in pre-training.
