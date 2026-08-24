# Raw-data library — `training-memory`

Pre-collected source library for the **GPU Memory in LLM Training** course
(`wiki/courses/training-memory/`). Crawl-assembled (no repo clone). Each file in
`excerpts/` is one artifact (paper / doc / tech report / framework module).

## Layout

```
training-memory/
  README.md               ← this file
  COLLECTION-PLAN.md      ← target coverage + gap log
  insights.md             ← one-line core insight per excerpt (index)
  crawl-manifest-*.json   ← per-cluster crawl record (researcher agents)
  excerpts/
    <slug>.md             ← one source artifact each
```

## Excerpt header schema

Every `excerpts/<slug>.md` follows:

```markdown
# <Human Title>
<!-- slug: <slug> · type: paper|doc|report|module · source: <url> -->

**Core Insight.** <the one idea this source contributes to the course>

**Guideline.** <the actionable rule a practitioner takes away>

## Technical Details
- <formulas, real numbers, verbatim-ish key claims>
- **Training-memory angle:** <how this specifically changes what fills a GPU during training>

## Citation
<authors, venue, year, url>
```

## Rules

- One artifact per file. Cite from a chapter's `read.md` via `[[slug]]`.
- Prefer primary sources (papers, official docs, framework source) over blog summaries; a blog is fine when it is the canonical practitioner reference (e.g. the Ultra-Scale Playbook, Transformer Math 101, `stas00/ml-engineering`).
- Every excerpt must include the **Training-memory angle** line — this library exists to explain *memory during training*, not the method in general.
- New must-read material discovered during authoring → add a file here + a row in `insights.md`; do not inline into a chapter.
