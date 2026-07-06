<!-- scope: raw source library for course/on-policy-distillation
     deps: [[COLLECTION-PLAN]]
     see-also: [[insights]], [[wiki/courses/on-policy-distillation/outline]]
-->

# On-Policy & Off-Policy Distillation — Raw Source Library

Primary source material for `course/on-policy-distillation`. The course teaches **LLM post-training as distribution matching**: OFF-policy distillation (train the student on a fixed set of teacher/dataset outputs) vs ON-policy distillation (the student samples its own trajectories and a fixed teacher grades every token), why the on-policy property fixes exposure bias, and when each wins — always at the **trade-off** altitude (what each method buys, and what it costs). It culminates in a capstone that designs an on-policy-distillation strategy for the learner's real SFT data pipeline (`boson-agent-synthetic-data-dev`).

This is **not** a general RLHF/PPO course (RL appears only as the sparse-reward contrast point) and **not** a pretraining course.

## What lives here

Like `software-arch`, this library has **no local clone** — it is **assembled by crawling the open web**: two primary blogs quoted verbatim (Thinking Machines, nrehiew), the canonical papers thesis-extracted (Hinton, Kim & Rush, Ross et al DAgger, Agarwal et al GKD, Gu et al MiniLLM, the Qwen3 technical report), and the practical TRL/HuggingFace recipe (doc-quoted).

```
raw-data/on-policy-distillation/
├── README.md            this file — scope, canonical-source map, header schema
├── COLLECTION-PLAN.md   coverage checklist + doc-vs-reality reconciliation + gap log
├── insights.md          cross-source insight index (built from the excerpts)
├── crawl-manifest.json  scored source manifest (slug, url, relevance, areas, fetched?)
└── excerpts/            one file per canonical source, cited by chapters via [[wikilinks]]
```

## Source-extract header schema (every file in `excerpts/`)

```markdown
<!-- scope: one-line description
     deps: prereq-excerpt (optional)
     see-also: related-excerpt
-->
# <Title>
- **Core Insight:** one sentence — the thing this source is famous for
- **Guideline:** one sentence — what a practitioner should actually do
- **Source:** URL(s) / paper citation (mark paper theses "thesis extracted, not verbatim")
- **Relevant chapters:** ch area tags

## ... (definitions / verbatim quotes / math / trade-offs / connections)
```

**Verbatim discipline:** freely-published blogs/docs (thinkingmachines.ai, nrehiew.github.io, huggingface.co/docs, the TRL docs) are **quoted exactly** with attribution. Paper-only material reachable mainly through abstracts (Hinton, Kim & Rush, Ross et al, Agarwal et al, Gu et al, Qwen3 report) is **thesis-extracted and clearly marked** — not represented as verbatim. Any source whose primary couldn't be fully fetched carries an inline NOTE (see the gap log in [[COLLECTION-PLAN]]).

## Canonical-source map (area → source → excerpt)

| Area | Source(s) | Excerpt |
|------|-----------|---------|
| The primary OPD reference: mechanism, per-token reverse KL, compute numbers | Kevin Lu & Thinking Machines Lab (2025) | [[tm-on-policy-distillation]] |
| Distribution-matching framing: SFT vs RL vs OPD, forward/reverse KL | nrehiew (2025 blog) | [[nrehiew-sft-rl-opd]] |
| Classical knowledge distillation: soft targets, temperature, dark knowledge | Hinton, Vinyals, Dean (2015) | [[hinton-knowledge-distillation]] |
| Sequence-level KD: training on teacher-generated sequences | Kim & Rush (2016) | [[kim-rush-seqkd]] |
| Exposure bias / compounding error / on-policy data collection | Ross, Gordon, Bagnell — DAgger (2011) | [[ross-dagger-exposure-bias]] |
| On-policy distillation of LMs, GKD (lambda / generalized JSD) | Agarwal et al (2024) | [[agarwal-gkd]] |
| Reverse-KL distillation for LLMs | Gu et al — MiniLLM (2024) | [[gu-minillm-reverse-kd]] |
| Practical recipe: TRL GKDTrainer + HF "any model family" | HuggingFace / TRL docs | [[hf-trl-gkd-recipe]] |
| Industrial-scale evidence: strong-to-weak distillation | Qwen3 Technical Report (2025) | [[qwen3-strong-to-weak-distillation]] |

## How this library is used

1. **Chapters** (`wiki/courses/on-policy-distillation/ch-*/read.md`) cite these excerpts via `[[wikilinks]]` and quote the linked primary sources.
2. **[[insights]]** is the cross-source synthesis built from the excerpts.
3. The **capstone (ch-07)** applies the whole toolkit to `boson-agent-synthetic-data-dev`.

**Authoritative-source rule:** where a free blog/doc exists it is the primary and is quoted verbatim; where only a paper exists, the *thesis* is extracted from the abstract + reputable summaries and marked as such. The reconciliation table in [[COLLECTION-PLAN]] records every place the popular narrative and the primary source disagree.
