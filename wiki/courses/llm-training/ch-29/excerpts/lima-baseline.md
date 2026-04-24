---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/lima.md
source_url: https://arxiv.org/abs/2305.11206
created_at: "2026-04-23"
---

# Excerpt: LIMA — the 1K-baseline ch-29's full-filtered pool must beat

**Source library:** `wiki/raw-data/llm-training/papers/lima.md`
**Artifact:** 1K hand-curated SFT set beating 52K Alpaca; the "Superficial Alignment Hypothesis"

---

## Why this source anchors ch-29

Ch-29 is a lab in *data engineering* — the thesis is that a cascade of filters + a verifier beats random selection at a fixed size. LIMA is the single strongest attested counterargument: if 1K hand-picked samples beat 52K unfiltered, then the cascade has to beat a LIMA-recipe-matched baseline to claim any value. The `lima-matched` ablation in ch-29 §4 is not optional filler — it is the null hypothesis the chapter is trying to reject.

---

## The Superficial Alignment Hypothesis — what LIMA actually claims

From the source (line 7):

> Almost all knowledge lives in pretraining; only ~1,000 carefully curated instruction–response pairs are needed to unlock it — the "Superficial Alignment Hypothesis."

The hypothesis makes a specific, testable claim: **SFT teaches format, not knowledge**. If SAH is correct, a 1K hand-curated set equals or exceeds any 52K noisy set. [[evol-instruct]] pushes back — the complexity tail matters. Ch-29's job is to find out, on *your* pool, which side wins.

---

## The LIMA recipe — what ch-29 reimplements for the baseline

From the source (lines 30–40):

- **Mixed community-forum Q&A** (StackExchange, wikiHow, Reddit) + hand-written prompts.
- **Heavy manual filtering** for diverse format, diverse topic, high response quality.
- **Response lengths deliberately varied.**
- **No RL, no DPO, no preference modeling** — pure supervised loss.

Ch-29 cannot reproduce "hand-picked by the paper authors for a week" — that is the baseline's strength. What ch-29 *can* do for the `lima-matched` ablation is approximate the LIMA recipe algorithmically:

1. **Diversity proxy** — embedding-space clustering on the raw pool; pick 1–2 samples per cluster.
2. **Format proxy** — response length spread (histogram-match LIMA's attested distribution).
3. **Quality proxy** — the single-axis [[alpagasus]] rubric at threshold ≥ 4.5/5.

This is not LIMA — it is "LIMA-recipe-inspired automated baseline." Label it as such in the memo. Anything else is overclaiming.

---

## The ablations LIMA itself ran — why ch-29 copies them

From the source (lines 41–45):

- Scaling data from 2K→32K StackExchange alone did not improve generation quality.
- **Diversity > raw count**: doubling examples within a single domain hurt performance.
- **Response quality > prompt quality**: poor responses cap the model regardless of prompt richness.

These three claims structure ch-29's ablation table:

| LIMA ablation claim | Ch-29 test |
|---------------------|------------|
| More same-domain data hurts | `no-dedup` ablation (dedup removes same-domain near-copies) |
| Poor responses cap the model | `no-verify` ablation (verifier drops wrong responses) |
| Diversity > quantity | `no-ifd` ablation (IFD's hard-but-learnable selection is the quality-maximizing axis) |

Each ch-29 ablation is a direct instrumentation of one LIMA claim.

---

## The training rule ch-29 borrows — and the one it doesn't

Ch-29 borrows:
- Prompt-token masking (attested canonical), same as [[loss-masking-prompt]].
- Cosine LR with warmup.

Ch-29 does **not** borrow:
- **15 epochs with LR decay through epochs.** LIMA's 15-epoch recipe is calibrated for 1K samples with expert-authored responses; at 1K–5K with teacher-authored responses ch-29 overfits by epoch 4. 2 epochs at `lr=2e-5` matches [[deita]] and [[allenai-tulu-sft-recipe]].

Overborrowing LIMA's epoch count is the most common failure when reimplementing the recipe. The memo should note which epoch count you settled on and why.

---

## LIMA's limitations — why ch-29's pool can beat it

From the source (line 47):

> Robustness is weaker than RLHF models — an adversarial prompt can knock LIMA off-script.

LIMA's 1K was optimized for a single axis (quality). It has no specific hardness tail, no tool-call coverage, no reasoning-chain coverage. Ch-29's pool deliberately targets those three axes via Evol-Instruct's `reasoning_steps` operator, the `complicate_input` operator, and the tool-call verifier subset. The expected delta where ch-29 beats `lima-matched` is on MT-Bench reasoning prompts and the held-out tool-call probe, not on general conversation.

If you run the ablation and your full-filtered pool *does not* beat `lima-matched` on reasoning, the memo's §4 failure mode almost certainly lives in your Evol-Instruct depth operators.

---

## What ch-29 keeps, changes, drops

| LIMA default | Ch-29 choice | Reason |
|--------------|--------------|--------|
| 1,000 samples | Matched to ch-29's `full` pool size N | isolate filter quality, not sample count |
| Hand-curation | Algorithmic LIMA-recipe proxy | cannot reproduce human expert effort |
| 15 epochs | 2 epochs | size + teacher-authored response regime |
| No RL/DPO | Same | SFT-only lab |
| StackExchange + wikiHow + Reddit + hand | Subset of same raw pool as `full`, LIMA-filtered | matched-source condition; otherwise measuring source not method |

---

## Connections

- **ch-22** — the full-read chapter on [[lima]].
- **ch-24** — [[deita]] is an explicit attempt to beat LIMA with an automated recipe; the same recipe template ch-29 inherits.
- **ch-29 §4** — the `lima-matched` run is load-bearing; without it the memo's headline is meaningless.
