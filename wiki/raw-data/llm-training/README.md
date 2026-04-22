<!-- scope: raw source library for course/llm-training
     deps: [[COLLECTION-PLAN]]
     see-also: [[insights]], [[wiki/courses/llm-training/outline]]
-->

# LLM Training Techniques — Raw Source Library

This directory holds the primary source material the `course/llm-training` course is built from. Every page here is an **extracted summary** of one artifact (paper, blog, tech report, framework module). Course chapters cite these pages via wikilinks.

## Scope

End-to-end post-training (with classical training-fundamentals as prologue):

- **Classics**: gradient clipping, mixed precision, LR schedules, weight init, dropout, label smoothing, early stopping — the pre-2020 training toolkit.
- **Data**: pretraining filtering/dedup, SFT instruction data construction (Self-Instruct/Evol-Instruct/rejection sampling/distillation), RL prompt curation, replay buffers and continual-learning mixing.
- **SFT**: sequence packing, NEFTune, loss masking, curriculum, multi-task recipes.
- **RL**: PPO, GRPO, DPO, IPO, RLOO, REINFORCE++ — algorithmic detail, KL/entropy dynamics, entropy collapse/explosion, reward models and reward hacking, process reward models, verifiable rewards (RLVR), rollout infrastructure, sampling schedules, self-improvement (STaR, Self-Play, Self-Instruct lineage).
- **Frontier reports**: Llama 3, Qwen 2.5/3, DeepSeek V3 / R1 / Math, Kimi K2, Tülu 3, OLMo 2.
- **Frameworks**: `verl`, `OpenRLHF`, `TRL` — concrete code excerpts for rollout loops, loss functions, entropy tracking.

## Directory layout

```
raw-data/llm-training/
├── README.md             this file
├── COLLECTION-PLAN.md    master topic checklist + source targets
├── insights.md           aggregated core-insights index (built last)
├── classics/             pre-2020 training fundamentals
├── papers/               arxiv + conference papers (flat; filename = slug)
├── model-reports/        frontier-model technical reports
├── blogs/                practitioner blogs, postmortems, lecture notes
├── frameworks/           OSS RL/SFT framework code excerpts
└── labs/                 per-lab capability summaries (Allen AI, DeepSeek, ...)
```

## File-naming convention

- Slug-cased, no prefixes: `ppo.md`, `grpo.md`, `self-instruct.md`, `tulu-3.md`.
- One artifact per file. Framework code goes in `frameworks/<framework>-<module>.md`.

## File format (required for every source page)

```markdown
<!-- scope: one-line description of what this source covers
     deps: prereq-source (optional)
     see-also: related-source
-->

# <Artifact title>
- **Core Insight:** one sentence — the thing this source is famous for
- **Guideline:** one sentence — what a practitioner should actually do
- **Authors:** ...
- **Year:** ...
- **URL:** ...
- **Relevant topics:** ...

## Abstract
(for papers) verbatim or faithful paraphrase

## Key Contributions
- 3–6 bullets

## Key Figures/Tables to Study
- which figure + one-line why

## Technical Details
(varies by source type — for RL algos include loss formula,
entropy term, KL penalty, hyperparameters; for data methods
include pipeline steps + filter thresholds; for frameworks
include file paths + line references)

## Connections
- where this connects to other sources
```

## How this library is used

1. **Planner** reads `COLLECTION-PLAN.md` + this library to decide chapter granularity.
2. **Course chapters** (`wiki/courses/llm-training/ch-*/read.md`) quote these pages via wikilinks and lift real code/equations from them.
3. **Insights index** (`insights.md`) is built last — one row per source, core insight + guideline.

Do not edit these pages to match course narrative. If a source changes interpretation during course writing, add a `Notes` section — don't overwrite the primary extract.
