---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/front-loading-reasoning.md
source_url: https://arxiv.org/abs/2510.03264
created_at: "2026-04-23"
---

# Excerpt: Front-Loading Reasoning - the asymmetric-allocation rule ch-32 inherits

**Source library:** `wiki/raw-data/llm-training/papers/front-loading-reasoning.md`
**Artifact:** pretrain prioritizes diversity; post-training prioritizes quality

---

## Why this source anchors ch-32

Front-Loading Reasoning is the paper that quantifies *when* to inject reasoning data across the stage stack. Its two findings are load-bearing for ch-32's stage-allocation argument:

1. **Front-loading reasoning into pretraining** gives a reported 19% average gain and raises the ceiling reachable by later SFT.
2. **The allocation rule is asymmetric**: pretrain wants diverse reasoning patterns; SFT wants high-quality reasoning data.

Ch-32's "mid-training multiplies pretraining quality" framing, its Section 2 diagnostic ("if breadth moves during mid-training, your mix is drifting"), and its recipe-by-budget rule ("your lever is data quality, not stage count") all descend from this paper.

---

## The asymmetric allocation ch-32 transcribes

From the source (lines 35-41):

- During **pretraining**, broader diversity of reasoning patterns matters more.
- During **post-training / SFT**, higher-quality reasoning data matters more.

Ch-32 uses this as the rule that explains why mid-training cannot be replaced by larger pretrain or larger SFT. Mid-training sits *between* the two: it keeps the quality emphasis of SFT (curated, narrow, hard) but uses the loss shape of pretrain (next-token on raw text). The asymmetry predicts that late SFT cannot fully reconstruct early-injected diversity - and the paper attests this directly.

---

## The durability claim ch-32 builds on

From the source (lines 32-34, 42-43):

- Reasoning data in pretraining creates a **durable, compounding advantage** that widens through later post-training stages.
- Late-stage SFT **cannot fully reconstruct** the durable advantage of early injection.

Ch-32's "stage-dependency summary" turns this into the rule that mid-training is *multiplicative* with pretraining quality: better pretrain priors amplify, not dampen, mid-training's effect. This is the answer to the "can better data skip mid-training?" open question: better pretrain priors raise the ceiling mid-training + RL reach, rather than absorbing mid-training's job.

---

## The 19% average-gain figure ch-32 cites

From the source (line 19):

- Front-loading reasoning data into pretraining produces a reported 19% average gain.

Ch-32 cites this as evidence of how large the stage-allocation effect can be. The number is specific to the paper's setup and should not be over-interpreted; ch-32 frames it as "a reported 19%" rather than a universal estimate. But the direction is clear: allocating reasoning data late is a dominated strategy under matched-budget comparisons.

---

## The RL-mechanism connection ch-32 makes

From the source (lines 40-43):

- Connects to the 2025 RL mechanism debate:
  - if RL mostly amplifies existing priors,
  - then what priors you install during pretraining becomes crucial.

Ch-32 uses this bridge to connect its stage-allocation argument to the larger RL debate. If [[rlvr-beyond-base-model]] is right that RL mostly reshapes probability mass, and [[interplay-pretraining-midtraining-rl]] is right that RL needs edge-of-competence prompts, then front-loading reasoning into pretraining is what determines whether an edge exists at all.

---

## Practical implication ch-32 internalizes

From the source (lines 45-47):

- If compute is limited, do not spend all reasoning budget at the SFT stage.
- Build some reasoning structure into the base model early, then use later post-training to refine style, correctness, and task alignment.

Ch-32's "recipe by compute budget" section is this guideline at each budget tier. Frontier labs do both front-loading and mid-training; mid-tier labs pick an open base that already front-loaded reasoning (Llama-3 / Qwen-3 / OLMo-3) and add a targeted mid-training pass; small-budget labs rely entirely on the base they picked having front-loaded reasoning.

---

## What ch-32 does not claim from the source

- The 19% number is study-specific and should not be generalized to arbitrary setups.
- The paper does not specify an optimal diversity/quality split for pretrain (the claim is relative: more diversity in pretrain than in SFT).
- The paper does not address when diversity-in-pretrain overflows into mid-training's job. That question is still open.

---

## Connections

- **[[interplay-pretraining-midtraining-rl]]** - controlled-experiment partner; the two papers together define ch-32's causal view of the stage stack.
- **[[echo-chamber-rl-post-training]]** - the strongest downstream case for why pretraining priors matter.
- **[[transferability-of-llm-reasoning]]** - complementary on the post-training side; narrow math SFT can harm generality.
- **[[quiet-star]]** - conceptual cousin; also moves reasoning into a pretraining-like phase rather than leaving it entirely to post-training.
- **[[olmo-3]]** - open example of stage-aware data curricula.
