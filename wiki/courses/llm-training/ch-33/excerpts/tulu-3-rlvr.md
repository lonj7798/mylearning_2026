---
chapter: ch-33
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlvr-tulu3.md
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
---

# Excerpt: RLVR — Tülu 3's one genuine algorithmic novelty

**Source library:** `wiki/raw-data/llm-training/papers/rlvr-tulu3.md`
**Artifact:** RLVR setup with verifier-only reward signal

---

## Why this source anchors ch-33

Everything else in Tülu 3 is best-of-breed data curation plus a conventional SFT → DPO pipeline. RLVR is what makes the recipe novel: replacing the learned reward model with a deterministic verifier `v: (x, y) → {0, 1}` on tasks where ground truth is checkable. This excerpt pins the formal setup and the PPO hparams so ch-33 §1.5 can quote them without ambiguity.

---

## The attested RLVR setup

From the source (lines 18–27):

- **Formalism:** `r(x, y) = v(x, y)` — no RM. The per-token reward is binary; a small KL-to-SFT penalty is added per token to prevent policy collapse.
- **Three verifier domains:**
  - *Math:* extract the final numeric/symbolic answer, compare to reference via tolerant grader — SymPy equivalence on MATH, exact integer match on GSM8K.
  - *Constrained IF (IFEval-style):* regex / parsers check satisfaction of constraints like "respond in JSON", "use exactly 3 bullet points".
  - *Code:* run model-generated code against unit tests in a sandboxed environment (5-second timeout), reward = 1 iff *all* tests pass.
- **KL control:** standard per-token KL-to-SFT penalty with small β; the source explicitly flags that omitting the KL term collapses the policy to a degenerate high-reward mode (e.g. constant-answer-guessing).
- **Prompt curation:** only prompts with a known reference answer and a working verifier enter the RLVR set. Everything else is routed through RLHF/DPO.

From the source (lines 28–34):

- **PPO settings:** group/batch ~128 prompts, 4 rollouts each, length up to 2k–4k tokens, LR ~1e-6, β_KL ~0.04, clip 0.2.
- **Advantage estimation:** GAE with λ = 0.95, γ = 1.0 (episodic).

Cross-reference: [[tulu-3]]'s model-card hparam block reports LR 3e-7, β_KL 0.05, K=4, clip 0.2, 10M episodes, GAE λ=0.95 γ=1.0 — the `[[rlvr-tulu3]]` page's ~1e-6 / 0.04 is the methodology-level "roughly" description; the tech-report numbers in [[tulu-3]] are the final config ch-33 quotes.

---

## The attested gains

From the source (line 26):

> Numbers (Tülu 3 8B vs Llama-3.1 8B Instruct): GSM8K 87.6 vs 84.7, MATH 43.7 vs 41.5, IFEval 82.4 vs 80.5 — gains cleanly attributable to RLVR stages.

These are the numbers ch-33 §1.5 quotes. The stage-attribution claim ("cleanly attributable to RLVR") rests on the DPO-only baseline ablation in the paper.

---

## The attested failure mode

From the source (line 39):

> Failure mode to watch: if the verifier has loopholes (string-match math graders that accept "42" inside prose), RLVR can hack those loopholes. Treat verifier engineering like unit-test engineering.

Ch-33 §1.6 is a direct instantiation of this warning. The no-RM Goodhart-proof property of RLVR is **conditional** on verifier tightness; a sloppy verifier reintroduces the exact problem RLVR was supposed to eliminate.

---

## What ch-33 keeps from this source

- The *formal* definition `r(x, y) = v(x, y)` in §1.5.
- The three verifier domains and their concrete implementations (SymPy, regex, unit-test).
- The Tülu 3 8B vs Llama-3.1 8B Instruct benchmark delta used as a headline.
- The "verifier engineering = unit-test engineering" rule of thumb in §1.6.

---

## Connections

- **ch-33 §1.5 / §1.6** — where this excerpt is cited.
- **ch-37..ch-46 (RL track)** — RLVR is the on-ramp to the verifiable-rewards RL family; [[deepseek-r1]]'s rule-reward RL is the direct descendant.
- **[[reward-model-overoptimization]]** — the Goodhart-drift failure mode RLVR mechanically avoids *on verifiable prompts*.
- **[[prm800k]] / [[math-shepherd]]** — process-reward-model complement; outcome-only verifier here, per-step supervision there.
