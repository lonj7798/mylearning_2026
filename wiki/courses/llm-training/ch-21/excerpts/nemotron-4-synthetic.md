---
chapter: ch-21
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/nemotron-4-synthetic.md
source_url: https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf
created_at: "2026-04-23"
---

# Excerpt: Nemotron-4 340B — generator / critic at alignment scale

**Source library:** `wiki/raw-data/llm-training/papers/nemotron-4-synthetic.md`
**Author/Org:** NVIDIA — 2024.

---

## Why this source anchors ch-21 §3

GLAN gives you the cleanest *tree*. Nemotron-4 gives you the cleanest *loop*. The two are complementary specializations of the ch-18 design pattern: GLAN adds structure to the "generate" step, Nemotron adds structure to the "filter + judge" step. Ch-21 §3 is built on the Nemotron source; this excerpt reconstructs the loop and the data-flow math.

From the source's core insight:

> NVIDIA compresses alignment into a strong reward model plus a synthetic prompt/response/pair pipeline; over 98% of post-training data is synthetic, and the same pipeline feeds SFT, DPO, and RPO.

The "over 98%" number is load-bearing. Ch-21 §3's anchor-set-size table is derived directly from this.

---

## The human / synthetic split — reconstructed

From the source (Key Contributions + Synthesis Pipeline):

- 20K human-annotated examples, split between SFT and HelpSteer2 RM training.
- 800K synthetic code SFT samples.
- 200K synthetic general SFT samples.
- 160K DPO preference pairs (synthetic, RM-judged).
- 300K RPO preference pairs (synthetic, RM-judged).

Totals:

| Role | Human | Synthetic |
|---|---|---|
| SFT | ~10K | 1,000K |
| Preference | 0 | 460K |
| RM training (HelpSteer2) | ~10K | 0 |
| **Sum** | **~20K** | **~1,460K** |
| **Fraction** | ~1.3% | ~98.7% |

The 20K human anchor set is overwhelmingly concentrated in *training the critic*, not in training the generator directly. This is the asymmetry that makes the pipeline work: you do not need to curate human-written *responses*, you only need to curate human-written *preferences* over synthetic candidates. Preferences are cheaper per token than completions, and they scale the RM, which then scales the pipeline.

---

## The generator / critic loop — what runs

The source's pipeline description compressed into one loop:

```
for iteration in alignment_stages:  # code SFT, then general SFT, then DPO, then RPO
    for family in task_families:    # coding, general QA, topic-following, doc reasoning, function call, refusal
        prompts = synthesize_prompts(policy_ckpt, family)
        for p in prompts:
            candidates = [policy_ckpt.sample(p) for _ in range(K)]
            scores    = [rm.score(p, c) for c in candidates]
            # SFT corpus: take top-scoring candidate
            sft_data.append((p, candidates[argmax(scores)]))
            # Preference corpus: take (top, bottom) or (top, mid) pair
            preference_data.append((p, best(candidates, scores), worst(candidates, scores)))
    retrain_policy_on(sft_data, preference_data)
    rm = retrain_rm_optionally(rm, new_helpsteer2_labels)
```

From the source:

> The pipeline is meant to preserve behavior diversity across task families while still using synthetic data at very high scale.

The loop is *not* a fully closed self-improvement loop (ch-23 territory). The RM anchor set is frozen — HelpSteer2 — and the generator is allowed to iterate but the critic's training data stays stable. This is how Nemotron keeps recursive-distillation collapse bounded: the judge is a stationary target.

---

## RM-as-filter and RM-as-judge — two distinct uses

From the source:

> a reward model scores responses for quality; when ground truth is missing, Nemotron-4-340B-Reward selects high-quality chosen responses. The preference pipeline prefers RM-based ranking over raw model self-selection.

Two operational modes worth keeping separate in your head:

1. **RM-as-filter.** Given one (p, response) pair, decide whether it passes a quality threshold for SFT inclusion. Binary.
2. **RM-as-judge.** Given (p, response_A, response_B), decide which is chosen and which is rejected for a preference pair. Comparative.

The same underlying RM serves both roles, but the thresholds and failure modes differ. Filter mode is vulnerable to miscalibrated absolute scores (the RM might think everything is 7/10). Judge mode is more robust to miscalibration because only the relative ordering is consumed. Nemotron uses both.

---

## Genetic Instruct — the code-family specialization

From the source:

> the code alignment stage uses Genetic Instruct, which combines self-instruction and WizardCoder-style mutations plus an LLM-based fitness function to grow a population from a limited number of seeds.

Genetic Instruct is inside the `synthesize_prompts(policy_ckpt, "coding")` call in the loop above. The family-internal fan-out is genetic-algorithm-shaped: mutate existing prompts, score offspring with an LLM fitness function, keep the winners, mutate again. This is a bottom-up (seed + mutate) step embedded inside a top-down (task-family) pipeline.

Ch-21 §3 notes this because it illustrates the "mix top-down and bottom-up" rule: Nemotron's task-family tree is shallow (6 families, flat), and the imagination inside each family is supplied by the Genetic Instruct bottom-up mechanism. Without the bottom-up piece, 6 families at the top-level would produce a corpus with very little within-family diversity.

---

## Staged SFT — a curriculum decision

From the source:

> Implements staged SFT: first a code-focused SFT stage, then a broader general SFT stage.

Why this order:
- Code data has cleaner correctness signal (unit tests, compiler errors). Training on it first installs "follow a structured format, fail loudly when wrong" behavior.
- General SFT on top leverages the structured-following behavior without damaging code ability because the general-SFT LR is smaller and the code capability is already consolidated.

The same pattern shows up in Phi-4 and Tülu 3. The curriculum is: easy-to-verify first, then open-ended.

---

## The DPO → RPO transition

From the source:

> Implements preference fine-tuning with DPO followed by RPO, with the reward model used to select higher-quality chosen responses.

DPO alone tends to *under-correct* for reward quality — it happily makes both chosen and rejected responses less likely if the training pair is noisy. RPO (Reward-aware Preference Optimization) adds an explicit scalar reward term to the loss, weighting pairs by RM confidence. This is a different failure mode from reward hacking (ch-42); it is more like "the generator needs calibrated signal from the critic, not just a direction."

Ch-21 does not expand on RPO mechanics — those are in ch-39 — but the point for §3 is that Nemotron uses two preference algorithms in sequence on the same synthetic pipeline, and the RM mediates both.

---

## Connections

- [[excerpts/glan]] — sibling top-down paradigm; different structure (tree vs task families) and different critic (rule-based vs RM).
- [[excerpts/phi-4]] — pivotal-token DPO is a related critic-in-the-loop technique for preference-pair construction.
- [[ch-18]] — the ch-18 filter + verify steps are specialized here to "RM score gating."
- [[ch-23]] — what stops the generator/critic loop from collapsing is the frozen HelpSteer2 anchor; ch-23 formalizes the recursive-training risk.
- [[ch-42]] — RM failure modes and reward hacking are the downstream risk once the pipeline is running at 98% synthetic.
- [[ch-21]] §3.
