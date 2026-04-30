# LLM Training Course Outline Feedback

Source reviewed: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json)

## Verdict

This version is strong.

The earlier structural issues are now mostly resolved:

- foundations are front-loaded
- `kind` is explicit in the schema
- intended audience and course format are explicit
- RL-vs-Eval and Data-vs-Infra boundaries are documented
- Eval can start after SFT instead of waiting for the full RL track
- every lab now has a resource-constrained path

At this point, the outline reads like a serious self-study program for training / research engineering, not a paper catalog.

## What Is Working Well

### 1. The curriculum shape is now operational

The 7-track structure is coherent and matches real training work much better than the earlier 4-track version:

- foundations
- data
- synthetic
- sft
- rl
- eval
- infra

References:

- track layout: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:21)
- structure note: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:11)

### 2. Foundations are in the right place

The course now starts with optimization, precision, packing, distributed training, checkpointing, and failure modes:

- optimization: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:67)
- precision: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:81)
- packing/templates: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:113)
- distributed training: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:129)
- checkpointing/instrumentation: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:145)
- failure modes: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:161)

That was one of the most important fixes.

### 3. The schema now supports real outputs

The previous criticism about being too concept-only is mostly addressed.

- explicit `kind` field: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:13)
- explicit audience / format: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:16)
- lab paths for full-budget vs constrained runs: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:19)

And the actual labs are concrete:

- systems trainer lab: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:177)
- data pipeline lab: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:290)
- synthetic-data lab: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:483)
- SFT lab: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:596)
- RL lab: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:757)
- Eval lab: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:876)
- capstone: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:973)

### 4. Eval is now placed correctly in the learning graph

The most important dependency fix is real now:

- the authoring note says eval can be pulled forward after SFT: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:12)
- `ch-45` now depends on `ch-34`, not on the end of RL: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:774)

That is much closer to how strong teams work.

### 5. The track boundaries are now documented instead of implied

Two previously ambiguous boundaries are now explicitly defined:

- RL vs Eval: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:17)
- Data replay vs Infra replay: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:18)

That removes a lot of the earlier conceptual drift.

## Remaining Issues

### 1. Base-model design and scaling-budget planning are still under-taught

The foundations track is strong on optimization and systems, but it still lacks a dedicated chapter on model-design and compute-planning choices:

- transformer block choices
- GQA / MQA / MoE tradeoffs
- RoPE and context-length design
- Chinchilla-style token/parameter planning
- MFU / FLOP accounting and budget estimation

References:

- foundations track start: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:23)
- current foundations content: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:67)
- capstone expects planning/memo work: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:973)

For training-engineering readiness, this is a real missing block.

### 2. Human annotation and label-ops are still light

The outline is strong on synthetic, reward, and judge signals, but it still under-teaches how strong teams create and QA human supervision:

- rubric design
- annotator disagreement
- adjudication
- calibration
- preference sampling policy
- when human data should override synthetic or judge-generated signals

References:

- preference-optimization family: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:645)
- reward modeling: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:677)
- process supervision: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:725)

That is still a substantive gap for post-training jobs.

### 3. Operational data engineering is still a bit light

The data track covers curation concepts well, but less of the production mechanics that real teams care about:

- tokenizer construction at scale
- shard / streaming layout
- dataset lineage and versioning
- code-repo filtering and executable screening
- secrets / PII removal as operations, not just policy
- code-quality filtering for web+code mixtures

References:

- data landscape: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:194)
- curation pipelines: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:210)
- data lab: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:290)

For model-training jobs, this is still worth strengthening.

### 4. One authoring note is now slightly inconsistent with the actual structure

The structure note says:

- “Every track ends with a lab chapter; the course ends with a capstone.”
  - [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:11)

But the `infra` track ends with a `capstone`, not a `lab`:

- [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:973)

This is minor, but it should be tightened so the metadata matches the actual graph.

### 5. The synthetic middle is still somewhat survey-heavy

The synthetic track is much better than before because it starts with a design pattern and verification frame:

- design pattern: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:307)
- verification / collapse: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:387)

But `ch-22..ch-26` is still a long run of modality chapters:

- reasoning traces: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:403)
- multi-turn: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:419)
- tool/function-calling: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:435)
- agentic trajectories: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:451)
- long-context: [outline.json](/Users/jaewon/mylearning_2026/wiki/courses/llm-training/outline.json:467)

This is defensible, but it is still the part most likely to drift back toward survey mode. If you ever want to tighten the course further, this is the first place to compress.

## Bottom Line

This is now a credible, high-signal outline.

It should be useful for someone aiming at research-engineering or training-engineering roles at a strong AI company, especially because it now combines:

- systems foundations
- real training artifacts
- evaluation discipline
- framework internals
- concrete deliverables

The remaining work is minor cleanup and optional compression, not structural repair.
