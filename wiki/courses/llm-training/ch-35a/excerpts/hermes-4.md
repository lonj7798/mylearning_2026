---
chapter: ch-35a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/model-reports/hermes-4.md on 2026-09-15)
source_url: https://arxiv.org/abs/2508.18255
source_version: Hermes 4 Technical Report, cached PDF text (no arXiv version stamp in the cached copy; arXiv v1 2025-08)
created_at: "2026-09-15"
---

# Excerpt: Hermes 4 Technical Report (Teknium, Jin, Suphavadeeprasit, Mahan, Quesnelle, Li, et al.; Nous Research)

Facts used by [[read]], read in the cached report text on 2026-09-15.

## Data (§2)
- About 5 million samples and 19 billion tokens: 3.5 million reasoning and 1.6 million non-reasoning samples; a significant portion of the Hermes 3 dataset retained. Reasoning samples average five times more tokens than non-reasoning samples, "accommodating thinking traces up to 16 thousand tokens long".
- DataForge (§2.1): graph-based synthesis from pre-training seed data; an LLM judge grades each instruction-answer pair on a type-specific rubric (coherence, relevance, complexity, style, tone); failed answers are revised until they pass or a maximum iteration count is reached, then discarded; the judge always has different weights from the answer model. Intermediate LLM calls of accepted pairs are also trained on.
- Rejection sampling (§2.2): "roughly a thousand task-specific verifiers" in the Atropos environment manager; "Following the recipe presented in OpenThoughts we include multiple unique trajectories to the same verified result."
- Environments include answer-format training (over 150 output formats, binary format reward), instruction following (RLVR-IFEval constraints), Internbootcamp (70,000 trajectories from about 1,000 reasoning tasks), JSON schema adherence, and interleaved tool use inside `<think>` (§2.2.1-2.2.5).
- Covering sets: LLM-generated taxonomies for data-scarce domains and PersonaHub personas; traces from DeepSeek-R1 or DeepSeek-R1-0528 (§2.3).

## Training (§3, Table 1)
- Bases: Llama 3.1 405B and 70B; Qwen3 14B. Modified TorchTitan; First-Fit Decreasing packing (> 99.9% batch efficiency); Flex Attention restricted within each packed sample; loss only on assistant-role tokens.
- 192 B200 GPUs; cosine schedule, 300 warmup steps, 9,000 total steps, global batch 384 samples at 16,384-token context.
- Table 1 (tokens / LR / B200 hours): 14B 56B / 5e-5 / 4,454; 70B 56B / 1e-5 / 12,864; 405B 56B / 5e-6 / 71,616.
- Epoch count is not printed.

## Reasoning-length control (§3.1, Table 2)
- The 14B model reached its 40,960-token limit 60% of the time on LiveCodeBench in reasoning mode.
- Second SFT stage: about 300,000 prompts (WebInstruct-Verified, rSTAR-Coder, DeepMath-130k); the current policy generates up to 30,000 tokens; `</think>` is inserted at 30,000 tokens; only `</think>` and `<eos>` are unmasked; combined with a subset of the first-stage SFT data.
- Table 2 (Qwen3 14B, stage 1 → 30k-tuned): AIME'24 55.0 → 52.4 (overlong 28.2 → 6.1); AIME'25 48.7 → 42.5 (25.9 → 9.0); GPQA Diamond 57.4 → 55.9 (18.2 → 9.5); LCBv6 28.6 → 44.2 (60.0 → 12.1). Not applied to 70B or 405B.

## Not reported
Teacher sampling temperature; decontamination procedure; SFT epochs.
