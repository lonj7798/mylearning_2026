---
chapter: ch-36
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/lima.md
source_url: https://arxiv.org/abs/2305.11206
created_at: "2026-04-23"
---

# Excerpt: LIMA — the baseline thesis that justifies ch-36's mix

**Source library:** `wiki/raw-data/llm-training/papers/lima.md`
**Authors:** Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy
**Venue:** NeurIPS 2023 (arXiv 2305.11206)
**Year:** 2023

---

## Why this source anchors ch-36

LIMA is the thesis against which every SFT mix defends itself. If "1K hand-curated examples on LLaMA-65B rivals GPT-4 in 43% of head-to-heads" is true, then ch-36's 100K mix has to justify each of the other 99K. Ch-36's answer is: diversity + skill-coverage + decontamination, not raw count. The 1K LIMA seed is the *quality anchor* of the mix; the 79K synthetic + 20K No-Robots/OpenAssistant-2 supply the *skill diversity* the original LIMA set lacked (math, code, multilingual, safety, IF).

Put differently: LIMA proves that the *knowledge* is in pretraining and SFT teaches *format*. Ch-36's mix tests whether the format-teaching signal scales across *six skills* ([[tulu-3-sft-mix]]'s chat/math/code/IF/safety/multilingual-reasoning) rather than LIMA's narrower format envelope. LIMA is the null hypothesis: if the packed+masked+NEFTune-off baseline underperforms a 1K LIMA-only run on any slice, the mix has not justified itself.

---

## The Superficial Alignment Hypothesis — ch-36's governing belief

Source line 15 (abstract close):

> Taken together, these results strongly suggest that almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output.

Named in line 18 as the **Superficial Alignment Hypothesis (SAH)**:

> alignment ≈ teaching format, not teaching knowledge.

SAH has three operational consequences for ch-36:

1. **Data quality dominates count.** Spending a week curating 1K diverse, high-quality examples is often better than spending a week scaling a generator to 100K mediocre ones. Ch-36's synthetic generator (from ch-29) has a rejection-sampling step specifically because LIMA's ablations (source line 43, "doubling examples within a single domain hurt performance") are the prior.
2. **Response quality is the ceiling.** Source line 44: "Response quality > prompt quality: poor responses cap the model regardless of prompt richness." Ch-36's synthetic pipeline filters on response coherence and factuality, not just prompt diversity. If MT-Bench scores plateau, audit responses first.
3. **Diversity is the only scaling axis worth pursuing.** Source line 43: "Scaling data from 2K→32K StackExchange alone did not improve generation quality." Ch-36's mix explicitly ranges across six skill buckets; single-domain scaling is forbidden.

---

## The 1K composition and its hidden care

Source lines 29–33:

> **Dataset construction (1,000 examples total):**
> - Mixed community-forum Q&A (StackExchange, wikiHow) and hand-written prompts.
> - Heavy manual filtering for **diverse format, diverse topic, high response quality**.
> - Response lengths deliberately varied.
> - **No RL, no DPO, no preference modeling.**

| Source | Count | Role |
|--------|-------|------|
| StackExchange (curated) | ~200 | Technical Q&A, multi-paragraph |
| wikiHow (curated) | ~200 | Procedural, step-by-step |
| Reddit (curated) | ~150 | Conversational, opinionated |
| Hand-written by authors | ~450 | Gap-fills for format / tone / length diversity |

**Notice:** nearly half the set is hand-written. This is the invisible labor that makes LIMA work and that synthetic generators have to emulate. Ch-36's mix uses [[lima]] itself as the 1K seed rather than attempting to reconstruct it — this is both a reproducibility decision (published 1K is reproducible) and a baseline-stability decision (any delta from the full-budget run is attributable to the *added* 99K, not to seed-curation noise).

---

## The training setup — why 15 epochs and why it isn't transferable

Source lines 36–39:

> Base: LLaMA-65B.
> Standard supervised cross-entropy on response tokens only (prompt tokens masked).
> Training: 15 epochs, learning rate 1e-5, batch size 32, AdamW.
> A key trick: **lowering LR as epochs progress** was essential to avoid overfitting on such a small set.

Compared to [[hf-alignment-handbook]]'s 1-epoch / 2e-5 recipe, LIMA runs **15× more epochs at half the LR**. This is not a generic best-practice; it is *specific to the 1K-example regime*. On 100K data, 15 epochs would catastrophically overfit. The rule is roughly:

```math
\text{effective updates} \approx \frac{N_{\text{examples}} \cdot E_{\text{epochs}}}{B_{\text{global}}}
```

| Recipe | N | E | B | Effective updates |
|--------|---|---|---|-------------------|
| LIMA | 1000 | 15 | 32 | ~470 |
| Handbook (Zephyr) | 200,000 | 1 | 128 | ~1560 |
| Ch-36 full path | 100,000 | 1 | 128 | ~780 |
| Ch-36 resource path | 20,000 | 1 | 64 | ~310 |

Ch-36's resource-constrained path sits close to LIMA in effective updates (310 vs 470). This matters for interpreting ablation results: the resource-constrained run is update-starved in the LIMA sense, so NEFTune's overfit-suppression effect should be attenuated (you cannot over-fit much in 310 updates). The prediction per [[karpathy-training-neural-net-recipe]] is: NEFTune delta is *smaller* on the resource-constrained path than the full path. If the lab measures the reverse, something is wrong with the full-path mix (likely low diversity) rather than with NEFTune.

---

## The scaling ablation — the flattest curve in alignment research

Source line 26:

> Scaling ablation — preference win-rate as training-set size grows 2× per step; the curve is famously flat.

LIMA's Figure 5 (not reproduced here but referenced in the source) shows win-rate barely moving as the training set doubles from 1K → 2K → 4K → 8K on a single domain. The curve *does* move when *diversity* is added. This is the empirical kernel of SAH.

Ch-36's mix is a direct response: instead of scaling within-domain, it adds *new* domains (math, code, IF, safety, multilingual). The hypothesis the lab tests is whether the flat-scaling-within-domain curve bends *up* when the added data is skill-orthogonal to the seed. If the ablation shows the 100K mix barely beats the 1K LIMA seed on MT-Bench, that's a SAH-confirmation surprise — and the memo should say so.

---

## The human-study result — the one number to remember

Source line 15:

> In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases; this statistic is as high as 58% when compared to Bard and 65% versus DaVinci003.

This is the number that made LIMA a landmark. A 65B model fine-tuned on 1K examples, no RL, ties GPT-4 in *nearly half* of blind comparisons. The implication is that the lion's share of Anthropic / OpenAI's capability comes from pretraining quality, not from the specific recipes of their SFT + RL stacks.

For ch-36, this is both motivation and caution:
- **Motivation:** SFT done well produces surprisingly competitive models. The lab's checkpoint on a 3B base, properly trained, can reasonably aspire to MT-Bench ≥ 5.5 (roughly Llama-3.2-3B-Instruct territory).
- **Caution:** a good MT-Bench score does not prove the SFT recipe is good. It might prove the base model is good. Ch-36's ablation grid is what discriminates between "the recipe works" and "the recipe doesn't hurt."

---

## Limitations — the forward pointer to RLHF

Source lines 46–48:

> Robustness is weaker than RLHF models — an adversarial prompt can knock LIMA off-script.
> Multi-turn dialogue was addressed post-hoc with 30 extra dialogue examples; still weaker than GPT-4.

LIMA is weak on adversarial robustness and multi-turn coherence — the two things RLHF is best at. Ch-36 inherits the first weakness (it is SFT-only, no RL). The second is partly fixed by the 20K No-Robots / OpenAssistant-2 addition (which has multi-turn data) but will only fully close in the RL track chapters 40+.

**Notice:** this is why ch-36 is the *capstone of the SFT track*, not the capstone of the course. The SFT artifact feeds the DPO / PPO / GRPO labs as `π_ref`. The LIMA limitations are ch-36's known open problems, not bugs.

---

## What ch-36 copies, adapts, rejects from LIMA

| LIMA default | Ch-36 choice | Reason |
|--------------|--------------|--------|
| 1K total | 100K (full) / 20K (resource) | Multi-skill mix requires more coverage than single-domain LIMA |
| 15 epochs | 1 epoch | Handbook's rule for > 10K examples |
| LR 1e-5 | LR 2e-5 | Handbook default; LIMA's lower LR was compensation for many epochs |
| No RL / no DPO | No RL / no DPO | Identical — ch-36 is SFT-only |
| LLaMA-65B | Llama-3.2-3B / 1B | Budget-driven |
| Prompt-masked loss | Prompt-masked loss | Identical; canonical SFT practice |
| Batch 32 | Batch 128 (full) / 64 (resource) | Modern FSDP + packing supports larger batches |

The 1K LIMA set itself is *not* replaced — it is the literal seed of ch-36's mix.

---

## Connections

- Full-read chapter on SFT data curation: [[ch-30]].
- Scaling / mix companion: [[allenai-tulu-sft-recipe]] — the mix doctrine ch-36 implements.
- Reference recipe: [[excerpts/hf-alignment-handbook]] — the hyperparameter inheritance.
- Regularizer axis where LIMA-scale data matters most: [[excerpts/neftune]].
- Loss-mask invariant LIMA established: [[excerpts/loss-masking-prompt]].
- Packing invariant (orthogonal to LIMA but paired in ch-36): [[excerpts/sequence-packing]].
- Counter-weight on data count: `[[alpaca]]` (52K), `[[ultrachat-pipeline]]` (1.5M).
- Lab host: [[ch-36]] — LIMA 1K is the mix's quality seed.
