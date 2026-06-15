<!-- scope: raw source library for course/software-arch
     deps: [[COLLECTION-PLAN]]
     see-also: [[insights]], [[wiki/courses/software-arch/outline]]
-->

# Software Architecture — Raw Source Library

Primary source material for `course/software-arch`. The course teaches **application/codebase-design-altitude software architecture**: how to choose and reason about monolith vs microservices vs modular monolith, DDD, hexagonal/clean architecture, event-driven design, API contracts, service decomposition, and architectural decision-making — always at the **trade-off** altitude (why/when each pattern, and what it costs).

This is **not** an infrastructure or distributed-systems-internals course (no Raft/Paxos, Kafka internals, k8s) and **not** an ML/model-architecture course. Consistency and resilience appear only at the *design-decision* altitude.

## What lives here

Unlike the other libraries, software-arch has **no local clone** — there is no single canonical repo. The library is **assembled by crawling the open web** plus distilling the theses of canonical books from their authors' free writing and reputable summaries.

```
raw-data/software-arch/
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
- **Guideline:** one sentence — what an architect should actually do
- **Source:** URL(s) / book citation (mark book theses "thesis extracted, not verbatim")
- **Relevant chapters:** ch area tags

## ... (definitions / verbatim quotes / trade-offs / connections)
```

**Verbatim discipline:** free articles (martinfowler.com, microservices.io, c4model.com, Fielding's dissertation, Cockburn, Nygard's ADR post) are **quoted exactly** with attribution. Book-only material (Evans, Vernon, Newman, Richards & Ford, Nygard *Release It!*, Skelton & Pais, Garcia-Molina's image-only PDF) is **thesis-extracted and clearly marked** — not quoted verbatim. Any source whose primary couldn't be verified at fetch time carries an inline NOTE (see the gap log in [[COLLECTION-PLAN]]).

## Canonical-source map (area → source → status)

| Area | Source(s) | Excerpt | Status |
|------|-----------|---------|--------|
| Monolith / microservices / modular monolith | Lewis & Fowler; Newman | [[fowler-microservices]], [[fowler-monolith-first]], [[newman-building-microservices]] | ✅ web-quoted + book-thesis |
| Distributed-monolith anti-pattern | Newman + community synthesis | [[distributed-monolith]] | ✅ synthesis |
| DDD strategic (bounded context, ubiquitous language, context mapping) | Evans; Fowler; Vernon | [[ddd-bounded-context]] | ✅ web-quoted + book-thesis |
| DDD tactical (aggregates, entities, domain events) | Evans; Vernon | [[ddd-aggregates-tactical]] | ✅ book-thesis |
| Hexagonal / Ports & Adapters | Cockburn | [[cockburn-hexagonal]] | ⚠️ cert-expired primary; mirror-verified |
| Clean / Onion architecture, the Dependency Rule | Robert C. Martin | [[martin-clean-arch]] | ✅ web-quoted |
| CQRS + Event Sourcing | Young; Fowler | [[young-cqrs-es]] | ✅ web-quoted |
| Sagas, choreography vs orchestration | Richardson; Garcia-Molina | [[richardson-saga]] | ✅ web-quoted; origin paper thesis-only |
| Outbox + database-per-service | Richardson | [[transactional-outbox]] | ✅ web-quoted |
| Service decomposition by capability/subdomain | Richardson | [[decompose-by-business-capability]] | ✅ web-quoted |
| Conway's Law + Team Topologies | Conway; Fowler; Skelton & Pais | [[conway-team-topologies]] | ✅ web-quoted + book-thesis |
| Strangler Fig migration | Fowler; Newman | [[martin-strangler-fig]] | ✅ web-quoted |
| REST + Richardson Maturity Model + API styles | Fielding; Fowler/Richardson | [[fielding-rest]] | ⚠️ ics.uci cert broken; gbiv.com mirror-verified |
| Consumer-driven contracts + contract testing | Robinson/Fowler | [[consumer-driven-contracts]] | ✅ web-quoted |
| Stability patterns (circuit breaker/bulkhead/timeout) + ADRs | Nygard | [[nygard-release-it]] | ✅ ADR web-quoted; Release It! book-thesis |
| C4 model | Simon Brown | [[c4-model]] | ✅ web-quoted |
| Architecture trade-off thinking + characteristics + fitness functions | Richards & Ford; Ford/Parsons/Kua | [[richards-ford-fundamentals]] | ✅ book-thesis (First Law quoted) |
| Distributed hard parts: granularity, quantum, data decomposition | Richards/Ford/Sadalage/Dehghani | [[richards-ford-hard-parts]] | ✅ book-thesis |
| App-level consistency: inside vs outside data, immutability | Pat Helland | [[helland-data-outside-inside]] | ⚠️ Queue 403; CIDR paper + summaries verified |

## How this library is used

1. **Planner** reads [[COLLECTION-PLAN]] + this map to set chapter granularity (outline not yet authored).
2. **Chapters** (`wiki/courses/software-arch/ch-*/read.md`, authored later) cite these excerpts via `[[wikilinks]]` and quote the linked primary sources.
3. **[[insights]]** is the cross-source synthesis built from the excerpts.

**Authoritative-source rule:** where a free article exists, it is the primary and is quoted verbatim. Where only a book exists, the *thesis* is extracted from the author's free writing + reputable summaries and marked as such. The reconciliation table in [[COLLECTION-PLAN]] records every place the popular narrative and the primary source disagree.
