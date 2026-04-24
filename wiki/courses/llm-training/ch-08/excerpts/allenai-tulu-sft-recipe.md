---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/allenai-tulu-sft-recipe.md
source_url: https://allenai.org/blog/tulu-3
created_at: "2026-04-23"
---

# Excerpt: Tülu 3 SFT — the scale-out rules ch-08's memo extends to

**Source library:** `wiki/raw-data/llm-training/blogs/allenai-tulu-sft-recipe.md`
**Artifact:** Allen AI Tülu 3 SFT recipe, `open-instruct` trainer, 939K-prompt mix

---

## Why this source anchors ch-08 §Full-budget path and §Connections

Ch-08's full-budget path is calibrated to a 50K-prompt single-source run — deliberately smaller than any production SFT job. Tülu 3 is the 2024 reference for what the *next* step looks like: 939K prompts, multi-source mix, 2 epochs, 8B → 70B sizes. This excerpt is how the lab's deliverables extrapolate to production without the learner having to re-derive the scale-up rules.

---

## The attested SFT hparam deltas — what changes beyond the lab

From the source (lines 42-57), the 8B / 70B table:

| Knob | 8B (Tülu 3) | 70B (Tülu 3) | Lab (ch-08 7B) | Ch-08 memo extension point |
|------|-------------|--------------|----------------|----------------------------|
| Max seq length | 4096 | 4096 | 4096 | — |
| Packing | yes | yes | yes | — |
| Response-only loss | yes | yes | yes | — |
| Optimizer | AdamW (0.9, 0.95) | same | same | — |
| Learning rate | 5e-6 | **2e-6** | 2e-5 (Zephyr default) | LR scales down with model + data size; lab's 2e-5 is only safe at 50K prompts |
| LR schedule | linear, 3% warmup | same | cosine, 10% warmup | Tülu uses linear; [[hf-alignment-handbook]] uses cosine; both work |
| Epochs | **2** | **2** | 1 | 2 epochs matches 939K-prompt-scale; 1 is correct at 50K |
| Global batch (prompts) | 128 | 128 | 128 | — |
| Distributed | FSDP FULL_SHARD | **FSDP + HYBRID_SHARD** | FSDP FULL_SHARD | HYBRID_SHARD is load-bearing at 70B; not needed ≤ 13B |
| NEFTune | off (neutral at 939K) | off | off | source confirms NEFTune saturates on large mixes |

Three rows are load-bearing for the lab's memo §3 ("What you would instrument next") — LR scaling, epoch count, and HYBRID_SHARD. Each is a decision the lab does not make but names as the next surface to touch.

---

## The "packing = 2.5× throughput, no quality delta" attestation

From the source (line 64):

> Packing: 2.5× throughput, no quality delta.

This is the single most important quality-ablation claim in the lab's entire motivation. [[sequence-packing]] proves the math; Tülu 3 proves it on 8B at 939K prompts end-to-end. The lab's `packing=True` is justified by two sources, one theoretical and one empirical; the memo can cite both. Ch-08 §Full-budget path's note that "A clean 100-step run on 7B + packed 4096 takes ~15 minutes on 8×H100" is a pro-rata of this 2.5× gain over the unpacked baseline.

---

## The ablation the lab inherits without re-running

From the source (lines 59-64):

> - Removing Persona-Math drops GSM8K by 15 pts; removing code drops HumanEval by 12.
> - Removing safety data barely moves capability evals but tanks WildJailbreak from 98% → 52%.
> - 2 epochs > 1 epoch > 3 epochs at this mix size; later epochs hurt IFEval.

For ch-08 the operational takeaways are:

1. **Data-mix composition matters more than trainer tricks.** The lab deliberately uses a single source to keep the debug surface small; the memo §3 should name "per-shard loss logging" as the next instrumentation step precisely because the production failure mode is "one shard is off-distribution" not "the trainer is wrong."
2. **The 2-epoch rule is domain-specific.** The lab does 1 epoch because its data is 50K prompts. The memo §4 ("Reproduction recipe") should not copy Tülu's 2-epoch default without re-ablating at the learner's data scale.
3. **Safety data is quality-invisible.** The lab does not include safety data; that is fine for a mechanics exercise. A learner who reuses the lab trainer for a real ship must include safety data *and* eval (WildJailbreak), which is why the memo lists "canary batch from a known-good evaluation" as an instrumentation candidate.

---

## The decontamination discipline — the silent-failure mode at dataset scale

From the source (lines 37-41):

> - 8-gram overlap ≥ 50% against every eval set → drop.
> - Embedding similarity > 0.9 to eval-set items → drop.
> - Documented "surviving overlap" rates per eval.

Ch-08 does not ask the learner to decontaminate. But the memo §1 ("Picks") should consider *"eval contamination via undocumented data overlap"* as a fourth silent failure, outside the trainer, at the data-pipeline layer. At 50K prompts from UltraChat the risk is low (UltraChat predates most 2024 evals); at 939K mixed-source it is the single largest source of illegitimate benchmark gains in the open-source community. Naming it in the memo makes the scope of the lab's trainer-level picks explicit.

---

## The `open-instruct` alternative — why ch-08 sticks with TRL

From the source (Overview, line 15):

> Tülu 3 is Allen AI's fully-open post-training suite: data, code, checkpoints, evals.

`open-instruct` is a TRL fork maintained by Allen AI with a slightly different chat-template handling and a different data-mix API. Ch-08 picks TRL over `open-instruct` because TRL is the upstream with the broader user base; `open-instruct` inherits the same three silent-failure lines with cosmetic differences. A learner who later uses `open-instruct` for the 70B stretch case gets the same concept map unchanged — the lab's deliverables transfer.

---

## The SFT → DPO → RLVR chain — the forward-pointer

From the source (lines 67-70):

> 1. SFT on 939K (this blog).
> 2. DPO on Tülu-3-Preference (~270K pairs from UltraFeedback + on-policy).
> 3. RLVR (verifiable-reward RL) for math / IF specialization → [[rlvr-tulu3]].

Ch-08's trainer is the SFT step. The same trainer object, with the loss function swapped, becomes the DPO trainer in Track 4. The lab's memo §4 explicitly calls this out as the reason the reproduction recipe matters — it is the shared prefix of three production chapters, not a throwaway artifact.

---

## Connections

- [[excerpts/hf-alignment-handbook]] — the smaller-mix sibling recipe; same trainer, smaller data, same mechanics.
- [[excerpts/sequence-packing]] — the 2.5× throughput number lands here in empirical form.
- [[excerpts/fsdp-sft]] — HYBRID_SHARD at 70B is the scale-up answer; FULL_SHARD is the lab default.
- [[excerpts/loss-masking-prompt]] — response-only loss is common to both handbooks.
- [[ch-04]] — SFT mechanics; Tülu 3 is the scale-out reference.
- [[ch-08]] — §Full-budget path (scale-down from these numbers), §Deliverables memo §3–4 (instrumentation, reproduction).
