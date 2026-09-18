---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md
source_url: https://github.com/OpenRLHF/OpenRLHF (issues tagged "entropy" / "KL" / "collapse")
created_at: "2026-04-23"
revised: "2026-09 (generality revision; reliability caveat added, contested numbers removed)"
---

# Excerpt: practitioner triage notes for entropy and KL problems

**Source library:** `wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md`
**Source type and reliability:** anecdotal — framework READMEs, GitHub issue threads and community
digests across OpenRLHF, verl and TRL. No controlled experiments. Per the course evidence standard, an
anecdotal source cannot be the only support for a quantitative claim, so ch-56 uses this source for the
shape of a triage procedure only and takes every default from the verified framework cards instead.

---

## What ch-56 takes from this source

**The metrics practitioners log during an RL run:** per-token entropy, KL to the reference, policy ratio
mean and standard deviation, reward mean and standard deviation, clipped fraction, response-length
histogram.

**The order in which they are checked when entropy falls:** confirm the KL-to-reference term is on and
finite; adjust rollout temperature; raise the entropy coefficient; check advantage normalization; only
then suspect the reward function.

**Failure patterns named in issue threads:** entropy falling within the first ~100 steps; reward rising
with no entropy change after ~1000 steps; sudden response-length growth with healthy entropy; `NaN` in the
policy ratio.

---

## Claims from this card that ch-56 does not use

- "adaptive-KL … is a safer default than fixed-β": OpenRLHF's default is the fixed controller
  (`--algo.kl.target` is `None`), so "default" is wrong here, and the card gives no evidence for "safer".
  See [[openrlhf-ppo]] Technical Details.
- "advantage normalization … OFF by default in TRL (a recurring footgun)": not verified against TRL source
  in this revision, and the wording violates the course tone rules.
- Numeric defaults (β ranges, learning rates, group sizes, rollout lengths): ch-56 quotes
  [[openrlhf-ppo-recipe]] and [[openrlhf-dpo-recipe]] instead, which carry file and line loci.

---

## Links

[[openrlhf-entropy-debugging]] · [[openrlhf-ppo]] · [[openrlhf-ppo-recipe]] · [[entropy-logging-patterns]]
