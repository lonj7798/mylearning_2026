---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/swe-rl.md
source_url: https://arxiv.org/abs/2502.18449
created_at: "2026-04-23"
---

# Excerpt: SWE-RL — difflib as a rule-based reward

**Source library:** `wiki/raw-data/llm-training/papers/swe-rl.md`
**Anchor paper:** Wei et al. 2025 — "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution"

---

## Why this source anchors ch-44

RLVR has a ceiling: the verifier has to be a deterministic function on `(prompt, completion)`. Math and code fit (graders, unit tests); most tasks do not. SWE-RL's contribution is the proof that a rule-based similarity between the prediction and a *reference output* is a strong enough signal for RL — no execution, no RM, no preferences. It is what makes RLVR a *generic* design pattern rather than a math/code-only trick.

---

## The reward — verbatim

From `swe-rl.md` §Synthesis pipeline — RL training:

> **Reward:** `r = difflib.SequenceMatcher(None, predicted_patch, ground_truth_patch).ratio()` — a scalar in [0, 1]. Clamped / shaped slightly (authors experiment with binary thresholding vs continuous reward; continuous wins).

Python code:

```python
import difflib
# predicted_patch, ground_truth_patch: unified diff strings
r = difflib.SequenceMatcher(None, predicted_patch, ground_truth_patch).ratio()
# r in [0, 1]; no sandbox; no tests run.
```

`SequenceMatcher.ratio()` is longest-common-subsequence-style: `ratio = 2 * M / T` where `M` is matching block length and `T` is total length. It is monotonic in text similarity and bounded, so no reward-clamping scheme is needed before PPO.

Three design choices embedded in that one line:

1. **No execution.** Unit-test rewards are sparse (many tests fail for reasons unrelated to the patch); similarity rewards are dense (every sample gets a gradient). The paper's ablation explicitly shows continuous similarity beats binary "similarity > tau" and beats execution-based reward at the scales tested.
2. **Reference-driven, not reference-free.** SWE-RL requires the human PR diff to exist. It cannot train on an issue with no resolution.
3. **Diff-granular.** The similarity is over unified diffs, not over source trees. This penalises off-target edits (wrong files, wrong lines) without needing a separate mechanism.

---

## The GRPO configuration — verbatim

From `swe-rl.md` §Synthesis pipeline — RL training:

> **Algorithm:** **GRPO** with group size G=8, KL coefficient β=0.02, LR 1e-6.

Connect to Tülu-3's PPO config from the previous excerpt: same ~1e-6 LR regime, smaller KL (`0.02` vs Tülu-3's `0.05`), group-relative advantages instead of GAE. GRPO is ch-40's stacked-advantage variant; the smaller KL is defensible because the reward is continuous (less variance than 0/1) and GRPO's group-normalisation absorbs per-group bias.

---

## The scale — verbatim

From `swe-rl.md` §Key Contributions:

> **11M GitHub issue-PR-code triplets** scraped and prepared as RL training data.
> **Llama3-SWE-RL-70B** reaches 41.0% on SWE-Bench Verified — open SOTA at release.
> **Out-of-domain transfer:** model gains on math/function-calling/reasoning benchmarks despite training only on SWE tasks.

And the transfer numbers:

> - HumanEval+: +6 points.
> - MATH: +4 points.
> - BIG-Bench Hard: +3 points.

The transfer finding is the provocative one. The paper hypothesises that SWE trajectories teach "long-horizon grounded planning" that transfers cross-domain. Whatever the mechanism, the operational implication is that RLVR on a sufficiently diverse verifiable domain can have positive spillover; the chapter's §7 cites this as the reason to invest in rule-based rewards beyond math/code.

---

## The risks — verbatim

From `swe-rl.md` §Risks:

> **Similarity reward can be gamed:** a predicted patch that copies context verbatim gets partial credit without fixing anything. Authors mitigate with format filters (must be diff, must modify code).

This is SWE-RL's analog to "verifier bugs" in RLVR. The reward function is a program; the program has edges; the policy finds the edges. The mitigation is structural filters (must be a diff, must modify non-context lines) rather than a post-hoc RM correction. That is the right pattern — rule-based rewards are debugged by tightening the rule, not by adding a learned scorer on top.

---

## Decontamination — verbatim

> **Data decontamination:** SWE-Bench Verified comes from the same GitHub universe; authors decontaminate by date (training data predates SWE-Bench issues) and commit-hash filter.

Important for ch-48 (eval) later in the track: any benchmark scraped from open-source is at risk of training-data leakage. SWE-RL's two-pronged filter (date cutoff + commit-hash match) is the standard defence; note that eval track will revisit it as a first-class concern.

---

## Single-turn limitation — verbatim

> **Single-turn limitation:** SWE-RL does not train multi-turn agent behavior (file navigation, test running). Paper notes this as future work.

SWE-RL's agent is single-shot: read the issue + relevant code context, emit a patch, done. It is not an agent-as-in-loop with tools. That is intentional — it keeps the reward well-defined on a single output — but it is a reason not to read SWE-RL as a complete agentic-training recipe. Ch-45 and later agentic-RL chapters revisit this.

---

## Carry into ch-44

- §7 of read.md quotes the `difflib` one-liner verbatim and the GRPO config.
- 41.0% SWE-Bench Verified is the headline number anchored in the comparison table.
- "Rule-based reward is a scalable RLVR substrate" is §7's thesis — the chapter's pattern library extends from math/code/IFEval to "any task with a reference output."
- Transfer finding (+6 HumanEval+, +4 MATH, +3 BBH) motivates treating RLVR on a verifiable domain as a general-reasoning investment, not a benchmark-specific tactic.
