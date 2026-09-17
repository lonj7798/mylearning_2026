---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/openrlhf-entropy-debugging.md
source_url: "framework READMEs and issue trackers (OpenRLHF, verl, TRL)"
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; downgraded to a reliability note)"
---

# Excerpt: status of the cross-framework "entropy debugging notes" as a source

The library card [[openrlhf-entropy-debugging]] describes itself as a synthesis of framework READMEs, issue
threads and community digests. It names no specific issue, commit or post, and carries no Verification
section. Several of its statements are contradicted by the code read at pinned commits in
[[entropy-logging-patterns]]:

| Claim in the synthesis | What the pinned code shows |
|---|---|
| "all three default to the k3 estimator" | verl's reward-side default is k1 and its reference KL is off; OpenRLHF defaults to k1 in the reward; TRL PPO defaults to k1; only TRL GRPO uses k3, with β = 0.0 by default |
| "default KL coefficient around 0.01-0.1 of the reward scale" | verl `kl_coef` 0.001 (and unused by default), OpenRLHF `init_coef` 0.01, TRL PPO `kl_coef` 0.05, TRL GRPO β 0.0 |
| "verl exposes `c_H` with default 1e-3 on some presets" | verl `entropy_coeff: 0`, `calculate_entropy: false` |
| "entropy below ~0.1 nats is diagnostic of collapse" | no located primary source states this threshold |

The triage ordering it gives (check the KL term, raise rollout temperature, raise the entropy coefficient,
check advantage normalization, then suspect the reward) is a plausible checklist but has no measurement
behind it in any source in this library. ch-43 cites this page only as an example of practitioner framing and
takes its diagnostics from [[entropy-logging-patterns]] and its interventions from papers with reported
settings and numbers.
