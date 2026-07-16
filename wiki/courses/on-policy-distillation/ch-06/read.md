<!-- chapter: ch-06
     track: practice
     kind: content
     title: Practical Recipes — GKD in TRL and On-Policy Distillation for Any Model Family
     deps: [[ch-04]]
     sources: [[hf-trl-gkd-recipe]], [[agarwal-gkd]], [[tm-on-policy-distillation]], [[nrehiew-sft-rl-opd]]
-->

# Chapter 06 — Practical Recipes: GKD in TRL and On-Policy Distillation for Any Model Family

> **Core insight.** On-policy distillation is a production trainer, not a paper idea. TRL's `GKDTrainer` reduces [[ch-04]]'s mechanism to three config knobs — `lmbda` (on-policy fraction λ), `beta` (forward↔reverse KL β), `temperature` — and the HuggingFace GOLD extension removes the one assumption that used to block real use: that teacher and student **share a tokenizer**. With GOLD, a teacher of *any* model family can grade a student's rollouts, which is exactly what you need when the best available teacher is a frontier model of a different family than your deployable student.

> **Guideline.** For same-tokenizer pairs, run `GKDTrainer` with high `lmbda` (on-policy) and tune `beta` per task. For cross-family pairs (e.g. a Claude/Llama teacher grading a Qwen student), use the GOLD extension with `use_uld_loss=True` and supply the teacher's tokenizer. Budget for the real bottleneck — a teacher forward pass over fresh student samples every step — and verify you are actually staying on-policy.

---

## 1. `GKDTrainer`: the mechanism as three knobs

TRL's `GKDTrainer` "is a wrapper around the `SFTTrainer` class that takes in a teacher model argument" ([[hf-trl-gkd-recipe]]). Each step, the teacher serves per-token logits over the (student- or teacher-generated) sequences and the loss is a generalized JSD against them — [[ch-04]]'s per-token grade, implemented. The knobs are exactly the GKD parameters ([[agarwal-gkd]]):

- **`lmbda`** (default `0.5`) — "controls the student data fraction, i.e., the proportion of on-policy student-generated outputs. When `lmbda=0.0`, the loss reduces to supervised JSD… When `lmbda=1.0`, the loss reduces to on-policy JSD, where the student generates output sequences and token-specific feedback on these sequences from the teacher." This is the [[ch-01]] data-source axis as a dial: 0 = off-policy KD, 1 = on-policy distillation.
- **`beta`** (default `0.5`) — "controls the interpolation in the generalized Jensen-Shannon Divergence. When `beta=0.0` the loss approximates forward KL divergence, while for `beta=1.0` the loss approximates reverse KL divergence." The geometry axis as a dial: mode-covering ↔ mode-seeking.
- **`temperature`** (default `0.9`) — sampling temperature for the on-policy student generations.
- **`seq_kd`** (default `False`) — sequence-level KD (supervised FT on teacher-generated output); `seq_kd=True, lmbda=0.0` is the [[ch-02]] Kim & Rush corner.

Practical guidance straight from the docs: "The authors find that on-policy data (high `lmbda`) performs better and the optimal `beta` varied depending on the task and evaluation method." And a real gotcha worth keeping: for Gemma2, set `attn_implementation="kernels-community/flash-attn2"` "Otherwise you will encounter NaNs in the logits due to the soft capping technique."

> **Interactive companion:** [`figures/gkd-knobs.html`](figures/gkd-knobs.html) — drag a point across the `lmbda`×`beta` plane and watch which method you have selected (supervised KD, on-policy distillation, forward vs reverse KL), with the corresponding `GKDConfig(...)` snippet updating live. It connects the TRL config to the [[ch-01]] map.

---

## 2. The cross-tokenizer wall — and GOLD

Vanilla GKD assumes teacher and student **share a vocabulary**, so their per-token distributions align position-for-position. That quietly rules out the most useful case: distilling a frontier teacher of a *different family* into your student. The HuggingFace H4 recipe, "Unlocking On-Policy Distillation for Any Model Family," targets exactly this ([[hf-trl-gkd-recipe]]). The problem framed verbatim: on-policy distillation carried "the requirement that the teacher and student models must share the *same* tokenizer vocabulary."

The method — **GOLD (General On-Policy Logit Distillation)** — extends Universal Logit Distillation (ULD) to the on-policy setting. It "incrementally decodes the student and teacher tokens, groups passages with the same visible text, and merges probabilities inside each group. This guarantees loss terms are computed over the full completion even when token boundaries differ." When a run of teacher tokens maps to one student token, the probabilities are merged by the chain rule:

```
P_merged(y) = P(y | ctx) · P(token₁ | token₀, ctx) · … · P(tokenₖ | …, ctx)
```

The result is intentionally unnormalized; the ULD loss uses sorting + L1 distance, so normalization is unnecessary. A hybrid option compares exact vocabulary matches directly and falls back to sorted-probability ULD for unmatched tokens. Reported payoff: GOLD "recovered 60% of the teacher's performance" versus ULD's 10%, and "outperformed GRPO by 20%" in cross-tokenizer scenarios.

`GOLDTrainer` "inherits the on-policy vs. off-policy scheduling from the `GKDTrainer`" (so `beta`/`lmbda`/`seq_kd` carry over) and adds `use_uld_loss` and `teacher_tokenizer_name_or_path` (required when `use_uld_loss=True`). Note it lives in `trl.experimental.gold` — fast-moving API — and defaults to a very low `learning_rate=1e-7`.

---

## 3. The real bottleneck: serving teacher log-probs

Every on-policy step needs a **teacher forward pass over the student's fresh samples** to get its per-token distributions ([[hf-trl-gkd-recipe]], [[tm-on-policy-distillation]]). This is the cost center, and the practical levers are: co-locate or batch the teacher; cache nothing (samples are fresh each step, by design); and size the student rollout (`max_new_tokens`, `num_generations`, `generation_batch_size`) to keep the teacher pass affordable. Thinking Machines' framing — OPD is the RL loop with the reward swapped for a teacher call — is also the cost model: you pay one teacher inference per rollout batch.

---

## 4. Pitfalls checklist

- **Verify you are actually on-policy.** If `lmbda` is low or samples go stale, training silently reverts to off-policy KD — the drift [[nrehiew-sft-rl-opd]] warns about. Keep `lmbda` high and resample every step.
- **Tokenizer mismatch.** Same-family? plain `GKDTrainer`. Cross-family? you *must* use GOLD/ULD; naive token-to-token KL across different tokenizers is undefined.
- **Entropy collapse & style tokens** ([[ch-05]]). Monitor entropy; use per-token clipping; pick `beta` toward reverse KL when the student can't match the teacher exactly.
- **Choose the teacher for signal, not size alone** — recall "the teacher matters less than expected" ([[nrehiew-sft-rl-opd]]); a *available, serve-able* teacher beats an ideal one you can't get log-probs from.

---

## 5. Myth killed: "the student and teacher must share a tokenizer/family"

This *was* true for vanilla GKD and is the reason on-policy distillation looked limited to "distill a big sibling into a small sibling." GOLD/ULD breaks it: by aligning on *visible text* rather than token IDs and merging probabilities across mismatched boundaries, a teacher of any family can grade any student's rollouts. That single change is what makes on-policy distillation usable in the common real-world setup — a frontier teacher from one lab, a deployable student from another.

---

## 6. Applied: choosing the recipe for the boson seller

Map the recipe onto `boson-agent-synthetic-data-dev`:

- **Student:** the deployable seller, `Qwen3.6-27B`-family.
- **Teacher options:** a large Qwen (same family → plain `GKDTrainer`, cleanest) or Claude (different family → **GOLD/ULD**, cross-tokenizer). The [[ch-05]] lesson ("data source > teacher") says: prefer whichever teacher you can *actually serve log-probs from* at acceptable cost; if that is a same-family large Qwen, you avoid the GOLD complexity entirely.
- **Knobs:** `lmbda→1` (fully on-policy seller turns — that is the whole point for a long-horizon agent), `beta` toward reverse KL (a 27B seller cannot fully mimic a frontier teacher; mode-seek onto what it can reproduce), `temperature` moderate for realistic sales phrasing.
- **Serving:** the teacher must grade Korean, tool-calling seller turns each step — the cost driver; budget the teacher inference and cap `max_new_tokens` per turn.

The recipe is settled; what remains is the domain engineering — which tokens in a tool-calling, compacting, barge-in-ridden conversation actually get graded, and how the existing relay becomes the on-policy rollout environment. That is the capstone.

---

## Where This Goes

Chapter 7 is the lab: a concrete on-policy-distillation strategy for `boson-agent-synthetic-data-dev`. It uses the whole course — the distribution map ([[ch-01]]), the off-policy diagnosis ([[ch-02]], [[ch-03]]), the mechanism ([[ch-04]]), the economics/failure modes ([[ch-05]]), and this chapter's recipe — to decide what to sample, what to grade, which teacher, and what to measure. Then we discuss it.

## Additional Reading

- TRL `GKDTrainer` docs — https://huggingface.co/docs/trl/gkd_trainer ([[hf-trl-gkd-recipe]])
- HuggingFace H4 Space, "Unlocking On-Policy Distillation for Any Model Family" — https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation ([[hf-trl-gkd-recipe]])
- Boizard et al., "Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs" (2024) — https://arxiv.org/abs/2402.12030 (ULD, the foundation GOLD extends)
- Agarwal et al., "On-Policy Distillation of Language Models" (GKD) — https://arxiv.org/abs/2306.13649 ([[agarwal-gkd]])
