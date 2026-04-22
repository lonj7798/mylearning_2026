<!-- scope: procedural symbolic data generation for pretraining and post-training
     deps: [[nemotron-4-synthetic]], [[prismatic-synthesis]]
     see-also: [[training-verifiers-to-solve-math-word-problems]], [[let-verify]], [[quiet-star]], [[front-loading-reasoning]]
-->

# Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre-training and Post-Training
- **Core Insight:** Reasoning Core turns symbolic reasoning into a procedural data problem: randomized generators plus external solvers produce verifiable tasks, controllable difficulty, and optional reasoning traces for both pretraining and RL-style post-training.
- **Guideline:** If you need synthetic reasoning data that is broad but still exact, generate from a structured task family with a solver in the loop, keep difficulty continuous for curriculum design, and expose the same task interface for SFT traces and reward computation.
- **Authors:** Valentin Lacombe, Valentin Quesnel, Damien Sileo
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2603.02208
- **Code/Data:** https://github.com/sileod/reasoning-core
- **Relevant topics:** symbolic reasoning, procedural generation, verifier-based data, synthetic pretraining, curriculum control

## Abstract
Reasoning Core is a procedural generator suite for verifiable symbolic tasks. The paper covers five core families: PDDL planning over randomized domains, first-order logic with equality, context-free grammar parsing and generation, causal reasoning over random Bayesian networks, and systems of equations. Each task is paired with an external solver for verification, and the generators support continuous difficulty control for curricula. The same interface can emit solver-derived reasoning traces for supervised training or reward functions for reinforcement learning. The authors report that mixing the data into pretraining improves downstream reasoning while preserving, or slightly improving, language-model quality.

## Key Contributions
- Provides a **multi-domain symbolic task suite** rather than a single puzzle template.
- Uses **external solvers** so every example is verifiable.
- Supports **continuous difficulty control**, which makes curriculum design explicit instead of ad hoc.
- Can emit **solver traces** for early-stage SFT or **rewards** for RL post-training.
- Releases **pre-generated data at more than 10B tokens** under an MIT license.

## Key Figures/Tables to Study
- **Task-family overview:** the main value is the breadth across planning, logic, grammars, causality, and algebra.
- **Difficulty-control examples:** useful for seeing how the suite supports curriculum shaping rather than static benchmark sampling.
- **Downstream evaluation table:** confirms that mixing Reasoning Core into pretraining helps reasoning without degrading LM quality.

## Technical Details

### Task families
- **Planning:** PDDL-style planning over randomized domains.
- **Logic:** first-order logic with equality.
- **Grammar:** CFG parsing and generation.
- **Causality:** random Bayesian networks.
- **Algebra:** systems of equations.

### Interface
- Each task is paired with an **external solver** for rigorous checking.
- Examples can optionally include **solver-derived reasoning traces**.
- The same representation can support:
  - supervised pretraining or SFT with traces,
  - reward computation for post-training,
  - curriculum control through task difficulty.

### Repository-level details
- The GitHub package exposes `list_tasks`, `get_task`, and `score_answer`.
- It includes a parallel generation path for writing JSON data for Hugging Face Datasets.
- The README also shows an integration path with `reasoning_gym`, so Reasoning Core can sit inside a broader synthetic-data stack.

## Connections
- [[nemotron-4-synthetic]] is the closest synthetic-data analogue in this collection: both scale synthetic pipelines, but Reasoning Core is more explicitly symbolic and solver-verified.
- [[let-verify]] and [[training-verifiers-to-solve-math-word-problems]] are the direct process-supervision predecessors for the "verifiable reasoning" framing.
- [[quiet-star]] and [[front-loading-reasoning]] are the pretraining-side conceptual companions, because they argue that reasoning should move earlier in the stack.
- [[prismatic-synthesis]] is the diversity-oriented complement: Reasoning Core expands breadth by task generation, while Prismatic optimizes diversity in gradient space.
