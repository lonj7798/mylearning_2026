---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-3.md
source_url: https://arxiv.org/abs/2512.13961
created_at: "2026-04-23"
---

# Excerpt: OLMo 3 - the model-flow worldview ch-32 uses as its reference

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-3.md`
**Artifact:** three-stage base (pretrain / mid-train / long-context) + three-stage post-train (SFT / DPO / RLVR)

---

## Why this source anchors ch-32

OLMo 3 is the cleanest public disclosure of a full five-stage pipeline where each stage has a *named* dataset, a *separate* compute allocation, and a *public* evaluation gate. Every other lab in 2025 runs a similar flow; OLMo 3 is the one you can verify. Ch-32's stage-definition table, its mid-training definition (1.7% of pretrain tokens, curated-hard mix), and its per-stage eval-gate list are all transcribed from the OLMo 3 report.

The conceptual move the report makes explicit: **the model flow is the artifact**. Previous Allen AI releases (OLMo 1, OLMo 2) shipped weights + data; OLMo 3 ships weights + data + intermediate checkpoints + per-stage training code + per-stage evals. That makes it possible, for the first time, to inspect "what did mid-training actually do" by diffing the mid-train endpoint against the pretrain endpoint on a targeted eval suite.

---

## The attested stage budgets ch-32 transcribes

From the source (lines 39-49):

- **Dolma 3:** ~9.3T raw source tokens (web, science PDFs via olmOCR, code, math, encyclopedic).
- **Dolma 3 Mix:** ~5.9T filtered pretraining tokens with stronger math/code emphasis.
- **Dolmino (mid-training):** 100B tokens sampled from a ~2.2T high-quality pool.
- **Longmino (long-context):** ~50B tokens from a 639B long-doc pool + mid-training data overlap.
- **Dolci (post-training):** separate mixes for SFT, DPO, RLVR.

The ratio that ch-32 carries forward: mid-training is **100B / 5.9T ~ 1.7%** of pretrain. This is the empirical "1-3% of pretrain tokens" rule in the ch-32 operational definition of mid-training. OLMo 2 sat at 50B / 3.9T ~ 1.3%, so the rule spans both Allen AI releases.

---

## The per-stage compute split ch-32 cites

From the source (lines 55-61):

- Pretraining: **1024 H100s**.
- Mid-training: **128 H100s**.
- Post-training: **256 H100s**.
- SFT throughput 8x improvement from Open Instruct -> Olmo Core migration.
- RL efficiency 4x improvement from in-flight weight updates + continuous batching.

Ch-32 uses this split as a sanity check: mid-training is *proportionally tiny* in compute (~1/8 of pretrain), which is consistent with its ~1.7% token budget and with its role as a "specialization stage" rather than a second pretrain. Post-training gets more compute than mid-training because it runs three sub-stages (SFT, DPO, RLVR) and because RL rollouts are expensive per step.

---

## The four branches ch-32 references

From the source (lines 34-38):

- **Base** 7B and 32B.
- **Think** 7B and 32B (reasoning-focused).
- **Instruct** (chat/tool-use).
- **RL Zero** (RL directly from Base, no SFT - the OLMo-3 analog of R1-Zero).

The RL Zero branch is why ch-32 discusses R1-Zero and cold-start SFT as a deliberate design axis: OLMo 3 reproduces the R1-Zero experiment inside its own open pipeline. This lets learners see how pure-RL reasoning compares against SFT-then-RL in a controlled setting where both start from the same Base checkpoint. Ch-32's "cold-start is a format installer, not a capability bootloader" framing is directly testable against the OLMo 3 Base vs RL-Zero vs Think comparison.

---

## What ch-32 does not copy from the source

- Specific eval-gate numbers per stage - OLMo 3's stage-endpoint evals are in the full report, not the model-report summary.
- 32B-model stage budgets - ch-32 focuses on the 7B flow because it is the smallest fully-disclosed flow.
- Efficiency infrastructure (Olmo Core, Open Instruct migration details) - these live in ch-36 (Systems for SFT at Scale).

---

## Connections

- **ch-33** - Tulu 3 is OLMo 3's post-training recipe ancestor; the SFT -> DPO -> RLVR structure is imported wholesale.
- **ch-34** - OLMo 2 vs OLMo 3 comparison (Tulu-recipe vs model-flow) is a case study in how explicit the stages should be.
- **ch-36** - Olmo Core migration + 4x RL efficiency details.
- **[[tulu-3]]** - upstream post-training recipe.
- **[[dolma]]** - earlier AllenAI data-transparency foundation.
