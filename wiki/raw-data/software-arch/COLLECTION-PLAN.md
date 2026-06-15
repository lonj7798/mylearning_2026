<!-- scope: coverage checklist + doc-vs-reality reconciliation + gap log for course/software-arch
     deps: [[README]]
     see-also: [[insights]], [[wiki/courses/software-arch/outline]]
-->

# Software Architecture — Collection Plan

Target: enough verified source coverage to teach **application-design-altitude architecture as trade-offs** — when each pattern wins, and what it costs. Status as of 2026-06-15: crawl pass complete, 16 excerpts written, outline not yet drafted. No blocking gaps for v1.

## Coverage checklist

| Area | Source | Excerpt | Status |
|------|--------|---------|--------|
| Microservices definition + 9 characteristics + trade-off | Lewis & Fowler | [[fowler-microservices]] | ✅ |
| MonolithFirst + modular monolith | Fowler | [[fowler-monolith-first]] | ✅ |
| Distributed-monolith anti-pattern | synthesis | [[distributed-monolith]] | ✅ |
| DDD strategic: bounded context, ubiquitous language | Evans/Fowler/Vernon | [[ddd-bounded-context]] | ✅ |
| DDD tactical: aggregates, domain events | Evans/Vernon | [[ddd-aggregates-tactical]] | ✅ |
| Hexagonal / ports & adapters | Cockburn | [[cockburn-hexagonal]] | ⚠️ mirror |
| Clean architecture / dependency rule | R.C. Martin | [[martin-clean-arch]] | ✅ |
| CQRS + event sourcing | Young/Fowler | [[young-cqrs-es]] | ✅ |
| Saga + choreography/orchestration | Richardson/Garcia-Molina | [[richardson-saga]] | ✅ |
| Outbox + database-per-service | Richardson | [[transactional-outbox]] | ✅ |
| Decomposition by capability/subdomain | Richardson | [[decompose-by-business-capability]] | ✅ |
| Conway's Law + Team Topologies | Conway/Skelton&Pais | [[conway-team-topologies]] | ✅ |
| Strangler Fig migration | Fowler/Newman | [[martin-strangler-fig]] | ✅ |
| Newman: independent deployability, info hiding | Newman | [[newman-building-microservices]] | ✅ |
| REST + RMM + API styles + versioning/idempotency | Fielding/Fowler | [[fielding-rest]] | ⚠️ mirror |
| Consumer-driven contracts + contract testing | Robinson/Fowler | [[consumer-driven-contracts]] | ✅ |
| Stability patterns + ADRs | Nygard | [[nygard-release-it]] | ✅ |
| C4 model | Brown | [[c4-model]] | ✅ |
| Trade-off law + characteristics + fitness functions | Richards/Ford | [[richards-ford-fundamentals]] | ✅ |
| Granularity, architecture quantum, data decomposition | Richards/Ford/Sadalage/Dehghani | [[richards-ford-hard-parts]] | ✅ |
| Inside vs outside data, app-level consistency | Helland | [[helland-data-outside-inside]] | ⚠️ reprint 403 |

## Doc-vs-reality / contested-claims reconciliation (primary source wins)

| Popular narrative | What the primary source actually says | Resolve in |
|---|---|---|
| "Microservices are the modern best practice; monoliths are legacy." | Fowler: start **MonolithFirst**; "almost all successful microservice stories started with a monolith." There's a **MicroservicePremium**; it's a trade, not an upgrade. | [[fowler-monolith-first]], [[fowler-microservices]] |
| "DDD requires microservices." | **False.** DDD is a modeling discipline; bounded contexts can be modules in a modular monolith. No DDD source ties it to a deployment topology. | [[ddd-bounded-context]] |
| "CQRS and Event Sourcing go together / are the same thing." | They are **independent** decisions that *compose*. Fowler: CQRS "adds risky complexity… use on specific portions… not the whole system." Neither requires the other. | [[young-cqrs-es]] |
| "Our API is RESTful." | Almost all are **Level 2** (HTTP-RPC) on Richardson's scale. Fowler: "Fielding has made it clear that level 3 [hypermedia] is a pre-condition of REST." Most "REST" isn't, by Fielding's definition. | [[fielding-rest]] |
| "Split into microservices and you get loose coupling for free." | If they share a DB or must deploy together, you built a **distributed monolith**: "all the pain of distributed systems without the independence." | [[distributed-monolith]], [[newman-building-microservices]] |
| "Use distributed transactions (2PC) across services." | Helland: services **don't share transactions**; Richardson: 2PC across DB+broker "not viable." Use **sagas** + **outbox** and accept lost isolation. | [[helland-data-outside-inside]], [[richardson-saga]], [[transactional-outbox]] |
| "Pick service size by a rule of thumb (e.g. fits in 2 pizzas / N LOC)." | Hard Parts: enumerate **granularity disintegrators vs integrators**; size is the *output* of a trade-off analysis, not an input. | [[richards-ford-hard-parts]] |
| "A C4 'container' is a Docker container." | Brown: a container is any separately deployable app or data store (server app, SPA, DB, bus) — **not** an OCI/Docker container. | [[c4-model]] |
| "Eventual consistency is an infra problem." | At app altitude it's a **design** problem: Fowler — "business logic can end up making decisions on inconsistent information." You must design for staleness. | [[fowler-microservices]], [[helland-data-outside-inside]] |
| "Conway's Law is folklore." | It's the original 1968 paper's thesis; the deliberate counter is the **Inverse Conway Maneuver**, and Team Topologies operationalizes it via cognitive load. | [[conway-team-topologies]] |

## Gap log

- **Cockburn hexagonal** primary (alistair.cockburn.us) had an **expired TLS cert** on 2026-06-15; web.archive.org is blocked from this environment. The Intent quote is verified via the alistaircockburn.com mirror + Cockburn's own wording in search results. Re-fetch the .us original when the cert is renewed to capture the full motivation/shape rationale verbatim.
- **Fielding dissertation** on ics.uci.edu had a **broken TLS chain**; quotes captured from Fielding's own **roy.gbiv.com** mirror (authoritative). No content gap.
- **Helland "Data on the Outside"** ACM Queue reprint returned **403**; the CIDR-2005 PDF + Semantic Scholar + "the morning paper" summary corroborate the inside/outside/immutability claims. Verbatim deep quotes beyond those summarized would need the CIDR PDF parsed.
- **Garcia-Molina & Salem "Sagas" (1987)** PDF was **image-only (no text layer)**; the saga LLT/compensation thesis is extracted from knowledge of the paper, **not quoted verbatim**. Richardson's microservices.io page is the verbatim-quotable saga source.
- **Book theses** (Evans, Vernon, Newman, Richards/Ford ×2, Nygard *Release It!*, Skelton & Pais, Ford/Parsons/Kua) are extracted from authors' free writing + reputable summaries and marked "book — thesis extracted." The course must cite them as book references, not as fetched URLs.
- **GraphQL/gRPC** are covered as *trade-off context* inside [[fielding-rest]], not as standalone excerpts (out of the canonical-author priority list). Add a dedicated excerpt only if the outline gives API-style selection its own chapter.
- No outline yet — planner should read [[insights]] + this plan before setting chapter granularity.
