<!-- scope: raw source library for course/automation-bench
     deps: [[COLLECTION-PLAN]]
     see-also: [[insights]], [[wiki/courses/automation-bench/outline]]
-->

# AutomationBench — Raw Source Library

Primary source material for `course/automation-bench`. The course studies **how a
cross-application tool-call benchmark is built and graded**, using Zapier's
AutomationBench as the worked example and the τ-bench family as the comparison point.

## What lives here

This directory is the **cloned AutomationBench repo** (github.com/zapier/AutomationBench,
CC BY 4.0, arXiv 2604.18934) — the primary code source. The raw clone is **local-only**
(git-ignored, like the other source clones); re-create it with:

```
git clone --depth 1 https://github.com/zapier/AutomationBench wiki/raw-data/automation-bench
```

The learning repo tracks only the **curated** files in this folder:

```
raw-data/automation-bench/
├── LIBRARY.md            this file
├── COLLECTION-PLAN.md    coverage checklist + doc-vs-code reconciliation + gap log
├── insights.md           cross-source insight index (built from the excerpts)
├── excerpts/             distilled source extracts the chapters cite via [[wikilinks]]
│   ├── automationbench-overview.md       paper + Zapier framing + landscape position
│   ├── automationbench-harness.md        runner, toolset modes, BM25, execute, world
│   ├── automationbench-tasks-grading.md  task anatomy, domains, assertions, hardening
│   ├── taubench.md                       τ-bench / τ² / τ³ design + pass^k
│   └── benchmark-comparison.md           AutomationBench vs τ-bench, head to head
└── <the cloned repo: automationbench/, tests/, visualizer/, README.md, ...>  (git-ignored)
```

## Source-extract header schema (every file in `excerpts/`)

```markdown
<!-- scope: one-line description
     deps: prereq-excerpt (optional)
     see-also: related-excerpt
-->
# <Title>
- **Core Insight:** one sentence — the thing this source is famous for
- **Guideline:** one sentence — what a benchmark builder should actually do
- **Source:** repo path(s) / URL(s)
- **Relevant chapters:** ch-XX, ...

## ... (Key mechanisms / Technical Details / Connections)
```

## How this library is used

1. **Planner** reads `COLLECTION-PLAN.md` + this library to set chapter granularity
   (already done — see `wiki/courses/automation-bench/outline.json`).
2. **Chapters** (`wiki/courses/automation-bench/ch-*/read.md`) quote the cloned code
   with `file:line` references and cite these excerpts via `[[wikilinks]]`.
3. **`insights.md`** is the cross-source synthesis, built from the excerpts.

The excerpts were distilled from a multi-agent research pass (two web-research agents on
AutomationBench's public framing and on τ-bench; two code-reading agents on the cloned
harness and on the domains/tasks). **The code is authoritative** wherever public docs and
code disagree — see the reconciliation table in `COLLECTION-PLAN.md`.
