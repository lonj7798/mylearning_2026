# Excerpt: Full Openness as Research Infrastructure

<!-- source: [[olmo-2|report]] -->

## The Spectrum of "Open"

The term "open" in LLM releases covers a wide spectrum, and the distinctions matter for research:

| Level | What You Get | What You Can Do | Examples |
|-------|-------------|----------------|---------|
| Closed API | Inference endpoint | Use the model | GPT-4, Claude |
| Open weights | Model parameters | Fine-tune, serve locally | Llama 3, Mistral |
| Open weights + code | Parameters + model code | Modify architecture, reproduce inference | DeepSeek-V3 |
| Fully open | Weights + data + training code + logs + checkpoints | Reproduce training, study dynamics, run ablations | **OLMo 2** |

Most models marketed as "open" are at the second level: open weights. This is sufficient for downstream applications (fine-tuning, serving) but insufficient for research into training dynamics, data-model interactions, or architecture design.

## What OLMo 2's Full Release Includes

**Model weights at multiple stages:**
- Stage 1 final checkpoint
- Multiple Stage 2 annealing variants
- Souped final model
- Thousands of intermediate checkpoints during training

**Complete training data:**
- OLMo-Mix-1124 (Stage 1 web data)
- Dolmino-Mix-1124 (Stage 2 curated data)
- Full data processing pipeline code

**Training infrastructure:**
- Complete training code (not just model code)
- Exact hyperparameter configurations
- Distributed training setup details
- Training logs: loss curves, gradient norms, LR schedules, hardware utilization

## Research Enabled by Full Openness

### 1. Training Dynamics

With thousands of intermediate checkpoints, researchers can study *when* capabilities emerge during training. Questions that are unanswerable with a single final checkpoint become tractable:

- At what training step does the model acquire factual knowledge about topic X?
- How does the loss landscape change between Stage 1 and Stage 2?
- When do attention patterns specialize (different heads attending to different syntactic roles)?

### 2. Data Attribution

With the full training data available, researchers can study the relationship between specific training examples and model behavior:

- Does the model memorize specific training sequences?
- How does the Stage 2 data mix composition affect downstream task performance?
- Which types of training data contribute most to reasoning capabilities?

### 3. Ablation Reproducibility

The published ablations in the OLMo 2 report can be independently verified. More importantly, researchers can run *new* ablations that the OLMo 2 team did not:

- What if Stage 2 used a different data mix?
- What if QK-norm was applied after RoPE instead of before?
- What is the optimal number of variants for model souping?

This is the research value proposition: OLMo 2 is not just a model, it is a **platform for architecture research** at frontier scale.

### 4. Benchmark for New Techniques

When proposing a new Transformer modification, the standard question is: "does it improve over the baseline?" OLMo 2 provides a fully reproducible baseline at 7B, 13B, and 32B scales. Researchers can implement their modification, train with the exact same data and hyperparameters, and isolate the effect of their change — something impossible with closed models.

## The Cost of Openness

AI2's choice to release everything has real costs:

1. **No competitive moat.** Any competitor can replicate the training recipe. AI2 accepts this because their mission is research advancement, not commercial dominance.
2. **Data liability.** Publishing training data exposes the organization to scrutiny about data sourcing, potential copyright issues, and content in the training set.
3. **Reproducibility burden.** Claiming full reproducibility invites external verification, which may reveal issues the team missed.

These costs explain why most labs do not follow AI2's example. The incentive structure of commercial AI development rewards secrecy. OLMo 2's openness is possible because AI2 is a nonprofit research institute with an explicit mission to advance open science.

## For Architecture Researchers

If you are studying LLM architecture (which is the purpose of this course), OLMo 2's openness means you can practice the full research loop:

1. **Read** the technical report and identify a claim
2. **Hypothesize** about an alternative architectural choice
3. **Download** the training code and data
4. **Run** a controlled experiment at small scale (7B or smaller)
5. **Compare** your results against the published baselines

No other frontier-competitive model enables this workflow. This is what makes OLMo 2 uniquely valuable for [[ch-29]] (Designing Architecture Experiments) — it is the only model where the transition from reading papers to running experiments has no proprietary barriers.
