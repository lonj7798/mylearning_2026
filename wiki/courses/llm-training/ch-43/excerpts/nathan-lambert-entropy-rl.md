---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/nathan-lambert-entropy-rl.md
source_url: https://www.interconnects.ai/
created_at: "2026-04-23"
---

# Excerpt: Nathan Lambert — entropy as the 2025 bottleneck metric

**Source library:** `wiki/raw-data/llm-training/blogs/nathan-lambert-entropy-rl.md`
**Author:** Nathan Lambert (Allen AI; editor of Interconnects)
**Venue:** Interconnects newsletter
**Year:** 2024–2025 (ongoing)

---

## Why this source anchors ch-43

Lambert's Interconnects posts are the single best practitioner synthesis of what the RL-for-LLMs community actually believes about entropy in 2025, as opposed to what has been formally published. He is both a co-author of Tülu 3 ([[rlvr-tulu3]]) and the most prolific commentator on DeepSeek-R1, which puts him at the intersection of "what recipes work" and "why they work". The read chapter's §1 and §2 lean on his synthesis to turn the formal Cui 2025 law into operational thresholds — most importantly the `H < 0.2` "stop and inspect" rule.

---

## The core claim

Source lines 17–18:

> Entropy is the bottleneck metric. In multiple posts Lambert argues that entropy collapse, not reward design, limits most reasoning-RL runs. Teams that ship improvements usually do it by changing exploration dynamics (rollout length, temperature, KL penalty) rather than by changing the reward.

This is the frame the read chapter adopts. Reward design (ch-41, ch-42) is a mature subfield with lots of levers; exploration dynamics is where the open problems are. If your RM is well-calibrated and your prompt set is well-curated and the run still plateaus, the culprit is almost always entropy collapse.

A corollary: metric-driven RL (tracking entropy, KL, clipfrac, reward mean/std) has converged across open labs as *the* operating discipline, not just a nice-to-have. The wandb dashboards from Tülu-3-style runs look alike because the diagnostic triple from [[excerpts/entropy-logging-patterns]] has become standard.

---

## The H < 0.2 rule

Source lines 34–35:

> When entropy crashes below ~0.2 nats on last-token distribution, stop and inspect — his recommendation, echoed in OpenRLHF practitioner notes.

This is the operational threshold the read chapter's §2 uses. Note the two thresholds in use:

- **`H < 0.1` nats** — Cui 2025's formal collapse threshold ([[excerpts/entropy-mechanism-llm-rl]]). Past this, the reward ceiling is effectively reached and the run is learning-diminished.
- **`H < 0.2` nats** — Lambert's "stop and inspect" threshold. One order of magnitude above collapse; the point at which human intervention is cheaper than letting the run burn more compute.

The read chapter uses the former for the definition of collapse and the latter for the triage trigger. Both are correct at different stakes.

---

## Why entropy bonus alone is insufficient

Source line 21:

> Entropy bonus alone is insufficient. Echoing the mechanistic analysis in [[entropy-mechanism-llm-rl]], Lambert repeatedly notes that adding a flat entropy coefficient does not prevent collapse at LLM scale because the collapse is driven by a small number of high-advantage tokens.

Lambert arrives at the same conclusion Cui 2025 does, from a practitioner direction rather than a theorem direction. His framing: the `c_H` knob in PPO/GRPO is a relic from pre-LLM RL where actions were low-dimensional (continuous control on <= 20-d action spaces, or Atari's <= 18-d discrete spaces). At those scales, *every* action mattered and a uniform bonus was close to optimal. At LLM scales (vocabulary of 100k+), "uniform bonus" smears mass across the uninteresting bulk and does nothing about the tail.

He explicitly recommends covariance-targeted fixes (Clip-Cov / KL-Cov) over `c_H` tuning for any serious reasoning-RL run.

---

## Temperature as first-move debugging

Source line 22:

> Temperature as debugging tool. Practitioner advice: if a run is flatlining, raise rollout temperature before retuning β; it's a faster diagnostic.

This is one of the concrete "triage first-moves" that shows up in the read chapter's §2 triage tree and in [[excerpts/openrlhf-entropy-debugging]]. The rationale: rollout temperature `T` is an *independent* lever from both β (the KL coefficient) and `c_H` (the entropy coefficient). Raising `T` widens rollout support without changing the optimized objective, so you can see in one batch whether the problem is "the policy has collapsed" vs "the reward signal is dead". If raising `T` from 1.0 to 1.2 jolts the reward curve, it was collapse; if nothing changes, you have a reward-design problem.

See [[sampling-temperature-schedule]] for the full T-schedule discussion; ch-43's §5 pulls this up as the third independent lever alongside entropy bonus and KL penalty.

---

## Cold-start SFT as entropy-stabilizer

Source line 20:

> DeepSeek-R1 explanation. Lambert argues R1-Zero works because GRPO + long rollouts + rule-based reward happen to sit in a low-entropy-collapse regime compared to PPO-RLHF. He flags cold-start SFT and format reward as critical to keeping the learning signal alive.

Two observations packed into this paragraph. First, R1-Zero's "pure RL from base model" is not a mystical accomplishment — it is what happens when you pick an RL recipe whose failure modes are orthogonal to the entropy collapse ones that trip up PPO-RLHF. GRPO's group-relative advantages are less noisy than value-baseline PPO's TD estimates; long rollouts give exploration room; rule-based rewards are binary and not smooth (which can be good or bad depending on how they interact with entropy).

Second, R1 (as opposed to R1-Zero) uses a cold-start SFT pass *before* RL. Lambert's explanation: SFT at a small scale nudges the policy's entropy profile toward something RL can work with — specifically, it shifts the high-covariance tokens off of format/EOS and onto actual reasoning-content tokens, which makes the covariance-driven collapse gentler and the format reward meaningful.

Relevant to ch-43 because: the read chapter's §5 "what LM-RL dropped" table does not mention cold-start SFT, but Lambert is hinting that cold-start is doing the target-entropy job implicitly — setting the initial distribution shape that RL then preserves rather than fixes.

---

## GRPO vs PPO pragmatics

Source line 23:

> GRPO vs PPO pragmatics. GRPO's value-free, group-relative advantage makes the entropy collapse easier to monitor (no critic drift to confound) — one of the reasons open labs have converged on it for reasoning RL.

This is why ch-43's §5 can say "modern GRPO has no value head" with a straight face. GRPO replaces the value critic with a group baseline (`A_i = (r_i − mean(r_group)) / std(r_group)`). That removes one source of non-stationarity (the critic) from the diagnostics, so entropy's trajectory reflects only policy updates and rollout variance, not critic drift. It is one of the reasons open labs have converged on GRPO for reasoning RL.

The downside: no critic means no bootstrapped value estimates, so GRPO is *more* dependent on good reward signals and *more* dependent on sample count per prompt (hence group_size 8–32 in modern recipes).

---

## Connections

- Read-chapter §2 uses this source's `H < 0.2` "stop and inspect" threshold for triage.
- Read-chapter §5 uses the cold-start SFT observation to clarify what LM-RL inherited from SAC.
- [[excerpts/entropy-mechanism-llm-rl]] — Lambert's practitioner view is the shortest path to the formal paper.
- [[excerpts/openrlhf-entropy-debugging]] — the framework-level triage steps echo Lambert's ordering (T, then c_H, then β).
- [[rlvr-tulu3]] — Lambert is a primary author; his Tülu 3 RLVR narrative is the context for §4 of the read chapter.
- ch-44 (Process Supervision / RLVR) — Lambert's "RLVR as structural answer to reward hacking" view underpins that chapter.
