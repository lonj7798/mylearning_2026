---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/interplay-pretraining-midtraining-rl.md
source_url: https://arxiv.org/abs/2512.07783
created_at: "2026-04-23"
---

# Excerpt: Interplay of Pre-training, Mid-training, and RL — why mid-training is a distinct mix stage

**Source library:** `wiki/raw-data/llm-training/papers/interplay-pretraining-midtraining-rl.md`
**Paper:** Zhang, Neubig, Yue 2025, "On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models."

---

## Why this source anchors ch-13

Ch-13 §4's central claim is **the mix is stage-specific**. This paper supplies the causal evidence for that claim: in controlled experiments, mid-training is not a naming variant of SFT, and its mix objective is substantively different from both pretraining and RL. The source is ch-13's strongest argument that "one α for all training" is wrong in principle, not just in practice.

---

## Mid-training as a distinct stage — the main claim

From the source (lines 44-47):

> ### Mid-training
> - Mid-training is presented as a distinct and important stage, not just a naming variation on SFT.
> - Under the paper's controlled setting, mid-training gives better results than using the same compute budget for RL-only post-training.
> - The interpretation is that mid-training helps install reusable priors that later RL can exploit.

Three claims, each important for ch-13:

1. **Mid-training ≠ SFT.** SFT is format-and-instruction-following adaptation. Mid-training installs reusable priors *before* the format gets fixed. A mid-training pass on math-heavy curated data is not the same thing as an SFT pass on instruction-formatted math problems — the first teaches quantitative reasoning; the second teaches "when asked a math question, answer in this format."
2. **Mid-training can beat more RL at fixed compute.** If you have 100K H100-hours to spend on improving a base model, spending it on a well-curated mid-training pass is often a better investment than spending it on longer RLVR. This is the direct contradiction of the "scale RL" intuition that dominated 2024.
3. **Reusable priors.** Mid-training's job is not to produce the final behavior but to install structural knowledge the later RL stage can exploit. This is the theoretical frame that justifies OLMo 2's Dolmino and OLMo 3's Dolmino + Longmino stages.

For ch-13's stage-specific mix table, this source provides the *objective* column for mid-training: install reusable priors that RL can build on. The mid-training mix is therefore concentrated on *learnable-structure-dense* domains — math, code, science, high-quality reasoning text — rather than on breadth.

---

## The headroom argument — RL needs pretrain-installed capacity

From the source (lines 39-41):

> ### Main causal claims
> - RL creates true capability gains only when pre-training has left enough unused capacity.
> - RL data must sit near the model's competence boundary; if tasks are too easy, the model is already there, and if they are too hard, RL has little to work with.
> - Minimal but sufficient pre-training exposure is enough to support contextual transfer once RL is applied.

The headroom claim reframes pretraining mix as an *enabler* of downstream RL rather than an end in itself. A pretraining mix that under-exposes the model to math leaves no headroom for RL-on-math to exploit; an over-exposed mix wastes capacity on domains RL cannot improve further.

Ch-13 §6's operational checklist item "verify each domain contributes ≥ 1% of the pool" is informed by this — too little exposure leaves no pretrained support for later RL; too much and you're paying for exposure that RL cannot leverage.

---

## The contextual vs extrapolative generalization distinction

From the source (lines 35-37):

> - Measures two different forms of generalization:
>   - **Extrapolative:** composing operations into harder problems.
>   - **Contextual:** reusing the same reasoning under different surface contexts.
> - This design makes it possible to isolate whether a gain comes from prior knowledge, transferable structure, or RL exploration.

For mix design this matters because **different mix choices favor different generalization modes**:

- A broad pretraining mix favors *contextual* generalization (lots of surface variation trains the model to see past surface differences).
- A concentrated mid-training mix on hard examples favors *extrapolative* generalization (deep structural practice on a narrow distribution).
- An RL stage on edge-of-competence tasks produces extrapolative gains only when headroom permits.

The mix-design lesson: if the goal is contextual transfer, upweight variety at pretraining. If the goal is extrapolative reasoning, concentrate mid-training on structurally rich domains. A single α cannot optimize both simultaneously — which is the formal-structure argument for ch-13 §4's stage-specific mix claim.

---

## Process supervision as a per-example mix signal

From the source (lines 49-52):

> ### Process supervision
> - The paper adds process-level verification to outcome rewards to reduce reward hacking.
> - This is a denser signal than final-answer correctness alone, so it better aligns reward with valid reasoning chains.
> - The result is better structural fidelity, not just better top-line accuracy.

Process supervision is less a mix question than a reward-design question, but it interacts with mix at RLVR stage. The RLVR prompt mix should be drawn from domains where **process rewards are computable** — math (step verifiers), code (unit tests), IFEval (constraint checks). This constrains the RLVR α to a narrow, verifier-compatible subset, explaining why OLMo 2's RLVR uses GSM8K/MATH/IFEval/code-unit-tests as the specific prompt pool rather than the broad SFT mix.

For ch-13 §4's RLVR row: "narrow: verifier-friendly math, code, IFEval" is the direct operational consequence of requiring process or outcome verifiability per example.

---

## The mid-training budget allocation question

The source's compute-comparison claim — that mid-training beats RL-only at fixed budget — has a practical implication for budget planning:

1. Total compute budget: fixed (say, 2M H100-hours for a 7B production run).
2. Pretraining: dominates, ~90% of budget.
3. Remaining ~10%: split between mid-training and post-training (SFT + DPO + RLVR).
4. Source's claim: within the remaining 10%, allocating more to mid-training and less to RLVR often dominates allocating more to RLVR.

OLMo 3 seems to implement this implicitly: Dolmino (100B) + Longmino (50B) = 150B mid-training tokens, substantially larger than typical SFT mixes (~1B tokens) and than typical RLVR token-equivalents. The source is the theoretical grounding; OLMo 3's release is the empirical budget allocation that takes the theory seriously.

---

## What this source leaves open

The paper uses synthetic reasoning tasks, not natural language corpora. The causal claims are clean in that setting but their quantitative translation to real-world training is approximate. Specifically:

- The optimal mid-training:RL compute ratio is setup-dependent — the paper doesn't give a number that transfers to natural-language pretraining.
- The specific "reusable priors" installed by mid-training are defined operationally by the synthetic-task structure; in natural-language training, the priors are less easily enumerated.

For ch-13 these are limitations, but the qualitative pattern — that mid-training is a distinct stage with its own objective — is robust enough to organize the chapter's stage-specific mix table.

---

## Connections

- `[[interplay-pretraining-midtraining-rl]]` — raw source.
- `[[ch-13]]` — §4 stage-specific table is built on this source; §6 operational checklist item on per-domain exposure derives from the headroom claim.
- `[[olmo-2]]` / `[[olmo-3]]` — labs that operationalized the distinct-mid-training view publicly.
- `[[front-loading-reasoning]]` / `[[echo-chamber-rl-post-training]]` — adjacent theses on the primacy of pretraining exposure.
- `[[prorl]]` / `[[rlvr-beyond-base-model]]` — RL-side counterparts; this source is the "more mid-training can beat more RL" argument.
