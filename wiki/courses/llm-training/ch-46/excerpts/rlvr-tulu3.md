---
chapter: ch-46
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlvr-tulu3.md
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
---

# Excerpt: RLVR / Tülu 3 — the verifier-as-reward design of Option B

**Source library:** `wiki/raw-data/llm-training/papers/rlvr-tulu3.md`
**Artifact:** Verifier-as-reward, three verifier domains, PPO hparams for RLVR, "verifier engineering like unit-test engineering".

---

## Why this source is Option B's spec

Option B of ch-46 is RL with verifiable rewards (RLVR) applied to math. The canonical open reference recipe is Tülu 3's RLVR stage. The ch-46 verifier function, reward semantics, KL-control defaults, and failure-mode taxonomy all derive from this source.

---

## Verifier-as-reward — the formal setup

Source §Key Contributions:

> **Formal RLVR setup:** for a prompt `x` paired with a verifier `v: (x, y) → {0, 1}`, the reward is simply `r(x, y) = v(x, y)` — no RM.

Ch-46's `verifier_reward(completions, answer, ...)` in `train_rlvr.py` is literally this function. The key property: `v` is a **fixed, interpretable function** — there is no learned proxy to drift, which is why RLVR sidesteps classical reward hacking on verifiable prompts.

---

## The math verifier — exactly what ch-46 calls

Source §Key Contributions / Three verifier domains:

> *Math:* extract the final numeric/symbolic answer and compare to the reference using a tolerant grader (SymPy / normalized string match on MATH, exact integer match on GSM8K).

Ch-46 uses `math_verify` (the de facto successor to Tülu 3's internal grader): `parse(completion)` extracts `\boxed{...}` or tail-expression, `verify(pred, parse(gold))` runs SymPy equivalence. The ch-46 §1 Data prep step that unit-tests the verifier on 50 hand-curated `(response, gold)` pairs is the direct response to the paper's line:

> Treat verifier engineering like unit-test engineering.

---

## Why RLVR "solves" Goodhart on verifiable prompts

Source §Key Contributions:

> **Why it sidesteps reward hacking:** the verifier is a fixed, interpretable function. There is no proxy RM to drift; there is no OOD region where the reward spuriously rises. Goodhart's gap (see **[[reward-model-overoptimization]]**) is mechanically zero on verifiable prompts.

This is the ch-46 §5(a) insight: on a correctly-written verifier, reward hacking in the classical sense is impossible. BUT — and this is the ch-46 post-mortem repro — *verifier loopholes* are a different failure mode. If the verifier accepts "The answer is 42" by string-matching "42" rather than SymPy-verifying `\boxed{42}`, the policy can hack *the string match*, not the underlying math. Ch-46 §5(a) repro ("weaken the verifier") exploits exactly this.

---

## KL control is still required — the one-sentence justification

Source §Key Contributions:

> **KL control still used:** standard KL-to-SFT penalty (per-token, added to reward) with a small β, otherwise the policy collapses to a degenerate high-reward mode like constant-answer-guessing.

This is why ch-46's Option B sweeps β_KL and nothing else. β_KL is the single knob that prevents a degenerate mode where the model learns one high-probability answer template and applies it to every prompt.

---

## PPO settings — the ch-46 hparam anchor for Option B

Source §Technical Details:

> **PPO settings:** group/batch ~128 prompts, 4 rollouts each, length up to 2k–4k tokens, lr ~1e-6, β_KL ~0.04, clip range 0.2.
> **Reward = 0/1 binary;** advantage estimation via GAE with λ = 0.95, γ = 1.0 (episodic).

Mapped to ch-46:

| Tülu 3 value | Ch-46 Option B |
|---|---|
| 128 prompts | 128 |
| 4 rollouts | G=8 (more per-prompt signal; affordable on 3B) |
| 2k-4k tokens | max_completion_length=1024 |
| lr 1e-6 | 1e-6 |
| β_KL 0.04 | sweep {0.01, 0.05, 0.1} bracketing 0.04 |
| clip 0.2 | ε=0.2 |
| GAE λ=0.95, γ=1.0 | GRPO (no critic, no GAE) |

Ch-46 uses GRPO instead of PPO because: (a) no critic to train separately, (b) TRL's `GRPOTrainer` exposes Dr.GRPO as a single flag. The reward semantics (0/1 binary from verifier) and β_KL role are identical.

---

## The hacking failure mode — explicit naming

Source §Technical Details:

> **Failure mode to watch:** if the verifier has loopholes (string-match math graders that accept "42" inside prose), RLVR can hack those loopholes.

This is exactly the ch-46 §5(a) repro: "weaken the verifier (e.g. accept `\\boxed{42}` substring instead of SymPy-verify) in a 10% shard; watch the policy discover the loophole within ~200 steps." The paper names the failure; ch-46 turns it into a repro recipe and a regression test.

---

## Gains to calibrate §7 Acceptance #4

Source §Key Contributions:

> **Numbers (Tülu 3 8B vs Llama-3.1 8B Instruct):** GSM8K 87.6 vs 84.7, MATH 43.7 vs 41.5, IFEval 82.4 vs 80.5

These are the **8B** numbers; for ch-46's 3B setup, expect step-0 GSM8K pass-rate in the 20-40% range and MATH in the 10-20% range. The §7 Acceptance criterion #4 ("step-0 reward ∈ [0.1, 0.4]") is calibrated to this, and a step-0 reward outside that band signals either a broken verifier (~0.0) or a data-leak / too-easy subset (~0.9).

---

## Connections to the rest of the track

- **ch-44 (RLVR / process supervision)** — the full-read chapter; ch-46 is its lab.
- **[[grpo]]** — the training loss Option B uses.
- **[[reward-hacking-taxonomy]]** — the formal theory; RLVR is the structural-fix path.
- **[[entropy-mechanism-llm-rl]]** — explains why "constant-answer-guessing" is the degenerate mode β_KL prevents.
- **[[prm800k]]** / **[[math-shepherd]]** — process supervision alternative; out of scope for ch-46 but a natural next experiment.
