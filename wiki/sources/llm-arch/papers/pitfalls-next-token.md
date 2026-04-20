<!-- scope: teacher-forcing failure modes in next-token prediction
     deps: [[gpt-1]]
     see-also: [[mamba]], [[gpt-3]]
-->

# The Pitfalls of Next-Token Prediction
- **Core Insight:** Teacher forcing can fail to learn an accurate predictor even in-distribution, because ground-truth history provides shortcuts absent at inference time.
- **Guideline:** On planning or multi-step reasoning tasks, consider multi-token prediction or teacherless training to avoid teacher-forcing shortcuts.
- **Authors:** Gregor Bachmann, Vaishnavh Nagarajan
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2403.06963
- **Relevant chapters:** Next-token prediction limitations, autoregressive inference, teacher-forcing failure modes, multi-token prediction, planning tasks

## Abstract
Can a mere next-token predictor faithfully model human intelligence? We crystallize this emerging concern and correct popular misconceptions surrounding it, and advocate a simple multi-token objective. As a starting point, we argue that the two often-conflated phases of next-token prediction -- autoregressive inference and teacher-forced training -- must be treated distinctly. The popular criticism that errors can compound during autoregressive inference, crucially assumes that teacher-forcing has learned an accurate next-token predictor. This assumption sidesteps a more deep-rooted problem we expose: in certain classes of tasks, teacher-forcing can simply fail to learn an accurate next-token predictor in the first place. We describe a general mechanism of how teacher-forcing can fail, and design a minimal planning task where both the Transformer and the Mamba architecture empirically fail in that manner -- remarkably, despite the task being straightforward to learn. Finally, we provide preliminary evidence that this failure can be resolved using teacherless training, a simple modification using dummy tokens that predicts multiple tokens in advance. We hope this finding can ground future debates and inspire explorations beyond the next-token prediction paradigm.

## Key Contributions
- Disentangled two failure modes of next-token prediction that the literature had conflated: (1) error accumulation during autoregressive inference (a well-known issue) and (2) teacher-forcing failing to learn an accurate predictor in the first place (a deeper, less recognized problem)
- Identified a general mechanism by which teacher-forcing can "cheat" -- fitting training data without learning the true generative process, because ground-truth history provides shortcuts that are unavailable at inference time
- Designed a minimal planning task (graph reachability) where both Transformer and Mamba architectures fail under teacher-forcing, despite the task being easy to learn with other training methods
- Proposed "teacherless training" with dummy tokens as a preliminary fix, predicting multiple tokens ahead to force the model to reason about future states rather than relying on ground-truth context
- Provided a rigorous conceptual framework for evaluating the fundamental limits of the next-token prediction paradigm

## Key Figures/Tables to Study
- **Figure 1** (Overview of two failure modes): Clearly separates autoregressive error accumulation (inference-time problem) from teacher-forcing failure (training-time problem). Essential for understanding the paper's conceptual contribution.
- **Figure 2** (The planning task): Shows the graph reachability task where teacher-forcing fails. Study this to understand the concrete failure mode -- the model learns to "copy" the ground-truth path rather than plan.
- **Figure 3** (Empirical results on planning task): Both Transformer and Mamba fail under standard teacher-forcing but succeed under teacherless training. This is the key empirical evidence.
- **Figure 4** (Multi-token prediction helps): Shows how predicting multiple tokens ahead mitigates the teacher-forcing failure mode.

## Architecture Details
- **Architectures tested:** Transformer and Mamba (state-space model)
- **Task:** Graph reachability -- given a graph and start/end nodes, produce a valid path
- **Training paradigm comparison:**
  - Teacher-forcing (standard): model receives ground-truth prefix at each step -- fails on the planning task
  - Teacherless training: replaces some input tokens with dummy tokens, forcing the model to predict multiple steps ahead without relying on ground-truth context
- **Key insight:** Teacher-forcing provides ground-truth history as input, which can contain information shortcuts that bypass the need for actual planning/reasoning. At inference time these shortcuts vanish, causing catastrophic failure.
- **Multi-token prediction:** Model predicts tokens t+1, t+2, ..., t+k simultaneously, which forces internal representations to encode future-oriented information rather than merely leveraging ground-truth context
- **Publication venue:** ICML 2024 (Proceedings of the 41st International Conference on Machine Learning, PMLR 235)
- **Note on arXiv ID:** The correct arXiv ID is 2403.06963 (not 2403.13112, which is a different paper)
