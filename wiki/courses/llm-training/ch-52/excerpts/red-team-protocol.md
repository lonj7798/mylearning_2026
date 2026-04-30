---
chapter: ch-52
course: llm-training
phase: read
excerpt_of: Anthropic safety research + HarmBench/WildGuard/Salad-Bench red-team methodology
source_url: https://www.anthropic.com/research
created_at: "2026-04-23"
---

# Excerpt: Red-Team Protocol — Closed vs Open, Manual vs Synthetic, Reporting Cadence

**Sources:** `wiki/raw-data/llm-training/labs/anthropic-safety-research.md`, `wiki/raw-data/llm-training/papers/harmbench-data.md`
**Primary background:** Anthropic Model Organisms of Misalignment, HarmBench automated red-teaming, WildGuard in-the-wild corpora.

---

## What a defensible red-team protocol looks like

Three organizing axes for the red-team workflow:

- **Closed vs open team.**
- **Manual vs synthetic attacks.**
- **Reporting cadence per release.**

The axes are independent; a real protocol commits to a specific point on each.

---

## Closed team first

Internal researchers with full model access, threat-model goals agreed in advance, and pre-committed disclosure rules. The raw-data source for [[anthropic-safety-research]] frames this as the red-team lineage that produced the Sleeper Agents and Model Organisms work.

Output is not just prompts — it is a **taxonomy of attack strategies**:

- Persuasion families (authority, sympathy, roleplay).
- Roleplay scaffolds (DAN-style, fictional-frame, creative-writing).
- Multi-turn escalation (crescendo attacks, context-injection across turns).
- Context-injection shapes (tool-use outputs, long-context attacks, multimodal).

Run before any public release. The closed team's advantage: full model access, full context on the training recipe, and time to iterate. The disadvantage: shared blind spots with the model's creators.

---

## Open challenge second

Invite external red-teamers with a bounded rule set. External teams find attacks internal teams do not, because they bring different adversarial priors and different attack taxonomies.

Precedents:
- DEFCON AI Village public events.
- Bug-bounty style programs with scoped disclosure windows.
- Anthropic's public red-teaming rounds around Claude releases.

The rule set must specify **what is in scope** (prompt-injection via tool use, long-context attacks, multilingual, multimodal) and **what is excluded** (physical security, social engineering of humans, DDoS of infra). Without this, coverage claims are meaningless.

---

## Synthetic attacks continuously

The [[harmbench-data]] + [[wildguard-data]] + [[salad-bench]] attack suites run on every release candidate as a regression gate. The synthetic layer is cheap and repeatable; the human layer is expensive and discovers novelty. Do not substitute one for the other.

Operational rule: synthetic is the **floor**, not the ceiling. A model that passes HarmBench test is not safe — it is not-obviously-broken.

---

## Reporting cadence

Per-release deliverables:

- **Named attack inventory.** Per family, per release. Reuse categories; do not re-invent per release.
- **Attack-success-rate per family.** Per [[harmbench-data]] attack taxonomy (Direct / Human / GCG / PAIR / etc.) or the benchmark's native breakdown.
- **Disclosure window.** N days between discovery of a severe attack and public disclosure, with a patch path documented.
- **Gap log.** Attack families attempted but not yet resolved. Published alongside the safety card.
- **Judge spec.** Which classifier / rubric / human process decided each reported number, with agreement statistics.

Cadence should match release cadence. A frozen red-team report on a model shipped 6 months ago is decorative — the attack surface has moved.

---

## Checklist (the ch-52 §6 long form)

1. **Taxonomy.** Named and reused across runs. Do not re-invent categories per release.
2. **Closed-team inventory.** Committed to version control before any eval numbers are generated.
3. **Synthetic regression floor.** HarmBench + WildGuard + Salad-Bench (or successors) as the baseline; not the ceiling.
4. **Over-refusal probe.** xstest or equivalent alongside the refusal suite; report both Pareto points.
5. **Persistence probe.** At least one held-out trigger per release; see [[anthropic-sleeper-agents-data]].
6. **Disclosure window.** N days between discovery and public report, with patch path.
7. **Gap log.** Unresolved attack families published alongside the safety card.
8. **Judge spec.** Judge-vs-human agreement attested per release.

---

## Why sleeper agents change the protocol

From [[anthropic-sleeper-agents-data]]: safety-training improves apparent alignment on the distribution tested while leaving latent conditional policies intact. A red-team that only probes the training distribution cannot distinguish a safe model from a model that was trained to look safe on those specific probes.

The protocol patch: add held-out triggers, held-out contexts, and held-out formats the model was not trained against. Measure trigger-on vs trigger-off deltas per metric. This is the minimum persistence evidence a model card should carry.

---

## Connections

- [[harmbench-data]] — synthetic attack catalog, ~18 families, the default regression floor.
- [[wildguard-data]] — synthetic in-the-wild corpus for moderation regression.
- [[salad-bench]] — hierarchical taxonomy diagnosis.
- [[anthropic-sleeper-agents-data]] — persistence probe rationale.
- [[constitutional-ai]] — the lineage from which open-challenge red-teaming grew; ~180K red-team prompts used in SL-CAI and RL-CAI.
- Chapter synthesis: [[ch-52]] §6, §9.
