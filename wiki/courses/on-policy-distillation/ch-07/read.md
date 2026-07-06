<!-- chapter: ch-07
     track: capstone
     kind: lab
     title: Capstone Lab — On-Policy Distillation Strategy for the boson-agent Conversation-Data SFT Pipeline
     deps: [[ch-05]], [[ch-06]]
     sources: [[tm-on-policy-distillation]], [[nrehiew-sft-rl-opd]], [[ross-dagger-exposure-bias]], [[agarwal-gkd]], [[hf-trl-gkd-recipe]], [[qwen3-strong-to-weak-distillation]]
-->

# Chapter 07 — Capstone Lab: On-Policy Distillation Strategy for the boson-agent Conversation-Data SFT Pipeline

> **Core insight.** `boson-agent-synthetic-data-dev` already does the on-policy thing — on the wrong side of the table. Its **customer simulator** samples live and on-policy (an 11-model rotation), but the **seller** — the model you actually ship (`Qwen3.6-27B-Lina-chk-*`) — is trained **off-policy**, by SFT on generated transcripts. A 20–50-turn sales call is precisely the long-horizon regime where off-policy imitation compounds error ([[ch-03]]). The strategy of this capstone: keep the scenario skeleton and the customer simulator as the *environment*, but make the **seller sample its own turns** and have a strong teacher grade each seller token by reverse KL — moving the shipped model from the off-policy corner to the on-policy-distillation corner of [[ch-01]]'s map.

> **Guideline (the memo in one line).** Reuse the relay as an on-policy rollout environment; sample seller turns from the student inside the fixed scenario skeleton; grade only the seller's generated tokens (including `tool_use`, masking customer and `tool_result`) against a serve-able teacher (same-family large Qwen first, Claude-via-GOLD as the quality ceiling) with `lmbda→1`, `beta` toward reverse KL, per-token clipping, and entropy monitoring; measure success as *reduced drift from the scripted path* in the eval gateway, not just lower training KL.

---

## 1. The system, placed on the map

From the pipeline itself (`agents/lina-tmr-customer-gateway/`):

- **What it produces:** multi-turn Korean insurance TMR (tele-sales) calls, 20–50 turns, exported as dual-view clean JSON and raw JSONL that preserves `tool_use`/`tool_result` blocks and pre/post-compaction snapshots (`export/synthetic_writer.py`, `export/raw_capture.py`).
- **How:** a scenario YAML drives a **customer** sampled live from an 11-model rotation (`customer_rotation.py`: 6 Qwen/boson variants + Claude Haiku + GPT-5-mini + Grok + 2 Gemini) against a **stage-puppeted seller** (`test-lina`, boson `Qwen3.6-27B`), orchestrated turn-by-turn by `relay/orchestrator.py`, with barge-ins (`relay/interrupt_cut.py`) and background compaction (threshold ~100 messages).
- **Current training:** it is **data generation only** (v0.14) — no reward, no RL, no distillation loop. The `Qwen3.6-27B-Lina-chk-*` checkpoints are **SFT'd on these transcripts**.

Now place it on [[ch-01]]'s three axes, per component:

| Component | Data source | Signal | Verdict |
|---|---|---|---|
| Customer simulator | **on-policy** (samples live per scenario) | — (it's the environment) | already on-policy — but it isn't the model we ship |
| **Seller student** (what we deploy) | **off-policy** (SFT on generated transcripts) | dense, forward-KL | the exposure-bias-prone corner ([[ch-02]] §6) |

The mismatch is the whole opportunity: the *environment* is on-policy, the *policy we care about* is not.

> **Interactive companion:** [`figures/boson-opd-flow.html`](figures/boson-opd-flow.html) — toggle between **Current (off-policy SFT)** and **Proposed (on-policy distillation)** to see exactly what changes: the seller becomes the sampler, a teacher grades its turns by reverse KL, and the relay becomes the rollout environment instead of a transcript factory.

---

## 2. Why this system is in OPD's winning regime

Run [[ch-05]]'s entry tests:

1. **Long horizon?** 20–50 turns — the exposure-bias term scales with T² ([[ch-03]]). A seller that fumbles an unusual objection or a barge-in recovery early drifts into states no transcript covered, and the call spirals. This is the "confidently wrong under distribution shift" failure an LLM agent shows exactly here. ✅ strongly favors on-policy.
2. **Teacher available?** Yes — a large Qwen (same family) or Claude (cross-family via GOLD, [[ch-06]]). ✅
3. **Clean verifiable reward instead?** No. "Did the call close / stay compliant" is a sparse, noisy, end-of-episode signal — RL's weak spot — and there is no unit-test oracle for a good sales turn. A dense per-token teacher signal is the more reliable lever. ✅ favors OPD over RL.

All three point the same way. The seller sits in on-policy distillation's winning regime; what remains is engineering.

---

## 3. The design

### 3.1 Rollout environment — reuse the relay

The on-policy requirement is "sample from the student in the states it will actually face." The relay already produces those states. So:

- Keep `relay/orchestrator.py` as the **rollout environment**: the scenario skeleton (stages, personas, per-round director notes) and the live customer stay fixed as the *environment*.
- Swap the seller from "stage-puppeted transcript producer" to **the student policy being trained**: at each seller turn, the student *samples* its response (this is the on-policy sample). The customer reacts as before.
- A full call is one rollout; each **seller turn** is a graded segment. This is DAgger's "run the current policy, then label the states it visits" ([[ross-dagger-exposure-bias]]) — with the label being a teacher distribution, not a single action.

### 3.2 Teacher choice

Per [[ch-05]] ("data source > teacher") and [[ch-06]] (serving is the bottleneck):

- **First choice — a same-family large Qwen** (e.g. a 72B / 35B-A3B Qwen already in the rotation): shared tokenizer ⇒ plain `GKDTrainer`, clean per-token alignment, cheapest to serve log-probs.
- **Quality ceiling — Claude** (cross-family): requires **GOLD/ULD** cross-tokenizer alignment ([[ch-06]]); higher-quality Korean sales behavior but more serving/alignment cost. Use only if the same-family teacher underperforms and the budget allows.

### 3.3 What to grade (the token mask)

A conversation is not uniform; grade only what the seller *generates*:

- **Grade:** the **assistant seller-turn tokens**, including `tool_use` tokens — tool calls are seller *actions* and are exactly where agent correctness lives ("confidently wrong tool calls" is the failure mode).
- **Mask (do not grade):** **customer turns** (that's the environment) and **`tool_result` tokens** (returned by tools, not generated by the seller).
- Compute the per-token reverse KL only over the graded (seller-generated) positions; the teacher must see the *same context* the student saw to make its log-probs comparable.

### 3.4 Handling the pipeline's structure

- **Compaction boundaries** (`raw_capture.py` pre/post snapshots): grade a turn against the *actual* context the student had at that turn — including the compacted summary. Give the teacher the identical (possibly compacted) context so `π_teacher(·|context)` is measured on the same prefix as `π_θ(·|context)`. Do not grade across a compaction boundary as if the context were continuous.
- **Barge-ins** (`interrupt_cut.py` dual-cut): a barge-in truncates the seller turn; grade only the tokens the seller actually realized before the cut. The recovery turn *after* a barge-in is high-value on-policy signal (a novel state) — keep it.
- **Context limit (32K):** grade **per turn**, not per whole call; the student's context per turn already respects the limit, and per-turn grading keeps the teacher pass affordable.

### 3.5 Knobs ([[ch-06]])

- `lmbda → 1`: fully on-policy seller turns — the entire point for a long-horizon agent.
- `beta` toward **reverse KL**: a 27B seller cannot fully mimic a frontier teacher, so mode-seek onto what it *can* reproduce.
- `temperature`: moderate — realistic sales phrasing without over-narrowing.
- **Per-token clipping: essential.** Korean sales dialogue is full of discourse/style tokens (honorifics, fillers) that carry high KL but little task content ([[ch-05]]); clip so they can't dominate the update or accelerate entropy collapse.

---

## 4. Price the bet (for this system)

- **What it buys:** the exposure-bias cure over 20–50-turn calls — a seller graded in the states it actually reaches (objections, barge-in recoveries, post-compaction continuations), which off-policy SFT never touches. Plus [[ch-05]]'s efficiency: dense per-token signal, ~10× cheaper than an RL alternative.
- **What it costs:** a **teacher forward pass over Korean, tool-calling seller turns every step** (the [[ch-06]] bottleneck); GOLD complexity if the teacher is cross-family; entropy-collapse and style-token risk in a chatty domain (mitigated by clipping + monitoring); and building the relay-as-rollout-env harness.
- **Versus the alternatives:**
  - *Cheaper SFT refresh* — still off-policy; does not fix drift. It is the thing we are trying to beat.
  - *Full RL with a closure/compliance reward model* — sparse, noisy, hackable; no oracle for a good turn. Weaker than a dense teacher here.
  - *Hybrid (frontier pattern)* — on-policy distillation to fix drift and transfer a strong teacher's behavior, *plus* a thin outcome reward (closure/compliance) layered on top — the "Pretrain→SFT→RL/Expert→OPD-merge" shape ([[nrehiew-sft-rl-opd]]) adapted to sales. This is the strongest long-term bet.

---

## 5. What to measure

Training-side KL going down is necessary but not sufficient — reverse KL can fall while the model collapses. Measure the thing OPD is supposed to fix:

- **Entropy** over training (collapse watch, [[ch-05]]).
- **Per-turn reverse KL** trend, split by turn type (early vs post-objection vs post-barge-in) — the on-policy win should show up most on the *hard* turns.
- **Drift reduction in the eval gateway** (`lina-tmr-customer-gateway-eval/`, where the seller runs with *real autonomy*, not puppeted): does the OPD seller deviate from a sensible path **less** than the SFT baseline on novel objections/barge-ins? That deviation drop is the direct, downstream measurement of exposure-bias reduction — the capstone's real success metric.

---

## 6. Myth killed: "the boson pipeline is already on-policy"

It is — for the *customer*. The customer simulator samples on-policy so the *environment* is realistic. But the seller, the model you deploy, is trained off-policy on the transcripts that environment produces. Confusing "the data was generated by live sampling" with "the shipped policy was trained on-policy" is the exact trap [[ch-01]]'s map exists to prevent. On-policy is a property of *whose samples the trained model learns from* — and today that is not the seller's own.

---

## 7. The strategy memo (deliverable)

> **On-policy distillation for `boson-agent-synthetic-data-dev` — recommendation.**
> 1. **Reframe** the relay (`relay/orchestrator.py`) as an on-policy rollout environment; freeze scenario skeleton + customer rotation as the environment.
> 2. **Sample** seller turns from the student (the deployable `Qwen3.6-27B`) inside that environment — one call = one rollout, each seller turn = a graded segment.
> 3. **Grade** seller-generated tokens (incl. `tool_use`) by per-token reverse KL against a teacher; **mask** customer + `tool_result`. Teacher sees the student's exact (possibly compacted) context.
> 4. **Teacher:** start with a same-family large Qwen (plain `GKDTrainer`); escalate to Claude via GOLD only if needed.
> 5. **Knobs:** `lmbda→1`, `beta` toward reverse KL, moderate temperature, per-token clipping, entropy monitoring.
> 6. **Measure** drift reduction on novel objections/barge-ins in the eval gateway — not just training KL.
> 7. **Roadmap:** OPD to fix drift now; add a thin closure/compliance outcome reward later (hybrid) for the frontier version.

---

## 8. Open questions for the strategy discussion

These are the live decisions — bring your view on each:

1. **Teacher:** same-family large Qwen (cheap, clean) vs Claude-via-GOLD (better Korean sales behavior, costlier). Where is the quality/cost line for *your* budget?
2. **Tool tokens:** grade `tool_use` tokens with the same reverse-KL weight as prose, or down-weight them (tool calls are more "correct/incorrect" than "stylistic")? Should a wrong tool call get a *harder* signal than reverse KL provides?
3. **Outcome signal:** pure OPD, or hybrid OPD + a thin closure/compliance reward from the start?
4. **Drift metric:** how do you operationalize "deviated from a sensible path" in the eval gateway into a number you can track?
5. **Cold start:** keep the current SFT checkpoint as the initial student (warm start) and OPD from there, or OPD from base?

## Additional Reading

- Thinking Machines, "On-Policy Distillation" — https://thinkingmachines.ai/blog/on-policy-distillation/ ([[tm-on-policy-distillation]])
- Agarwal et al., "On-Policy Distillation of Language Models" (GKD) — https://arxiv.org/abs/2306.13649 ([[agarwal-gkd]])
- TRL `GKDTrainer` / GOLD docs — https://huggingface.co/docs/trl/gkd_trainer ([[hf-trl-gkd-recipe]])
- nrehiew, "SFT, RL, and On-Policy Distillation Through a Distributional Lens" — https://nrehiew.github.io/blog/sft_rl_opd/ ([[nrehiew-sft-rl-opd]])
