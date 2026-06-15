<!-- scope: cross-source insight index for the software-arch raw library
     deps: [[README]], [[COLLECTION-PLAN]]
     see-also: [[fowler-microservices]], [[ddd-bounded-context]], [[richardson-saga]],
               [[helland-data-outside-inside]], [[richards-ford-fundamentals]]
-->

# Software Architecture — Insights Index

The course turns on **one idea**:

> **Architecture is the set of decisions that are expensive to reverse; every pattern in this library is a bet about which changes you must keep cheap.**

The field's own definition agrees ("architecture is the stuff that's hard to change," [[richards-ford-fundamentals]]), and its First Law tells you how to evaluate every bet: *everything is a trade-off* — find the cost or you haven't understood the pattern. Read every excerpt as "what does this keep cheap to change, and what does it make expensive?"

## Boundaries (where to cut, and why there)

- **Cut on the slowest-changing structure.** Business capabilities are more stable than orgs, tech, or screens — so cut services there ([[decompose-by-business-capability]]). The DDD path reaches the same seam from the model: split where the **language** changes ([[ddd-bounded-context]]).
- **Boundaries are the irreversible decision; defer them.** You can't pick good service boundaries before you know the domain, and fixing one across services is a migration, not a refactor → start with a **modular monolith** ([[fowler-monolith-first]]) and migrate via **Strangler Fig** ([[martin-strangler-fig]]).
- **The boundary is socio-technical, not just technical.** Conway's Law guarantees the architecture mirrors team communication; Team Topologies says size each boundary to a team's **cognitive load** ([[conway-team-topologies]]). A seam that splits a team's cognitive unit will be fought daily.
- **The architecture quantum is the boundary test.** Smallest independently-deployable unit with high cohesion *and its own data* ([[richards-ford-hard-parts]]). Two "services" sharing a DB are one quantum — and a [[distributed-monolith]].

## Coupling & cohesion (keep the inside free, the edges strict)

- **Dependencies point inward.** Policy must not depend on detail — the Dependency Rule ([[martin-clean-arch]]) and Ports & Adapters ([[cockburn-hexagonal]]) are the same move: a technology-free core, I/O at the edges through interfaces the core owns. This keeps the *expensive* part (domain policy) insulated from the *cheap* parts (vendors, frameworks).
- **Independent deployability is the acid test.** Newman: it's the one defining property of a microservice; achieved via **information hiding** + **data ownership** ([[newman-building-microservices]]). Lose it and you have a [[distributed-monolith]] — "all the pain of distributed systems without the independence."
- **The contract is the expensive-to-reverse artifact.** Once consumers depend on it, breaking it costs coordinated migration. Evolve additively, be a **tolerant reader**, and verify with **consumer-driven contract tests** ([[consumer-driven-contracts]]). Most "REST" is Level-2 HTTP-RPC, not Fielding-REST ([[fielding-rest]]) — be honest about which you're shipping.

## Consistency & events (what crossing a boundary costs)

- **Inside vs outside data is the root distinction.** Inside a service: SQL, ACID, mutable, "now." The instant data crosses a boundary it must be **immutable, versioned, possibly stale** — because you lose locks and shared transactions ([[helland-data-outside-inside]]). This single fact *generates* every consistency pattern below.
- **No distributed transactions across services.** Replace ACID-across-services with a **saga**: a sequence of local transactions + compensations, choreographed (events) or orchestrated (controller) ([[richardson-saga]]). The price is loss of isolation — add countermeasures.
- **Don't dual-write.** Persist the event in the same transaction (**outbox**) and relay it; consumers must be **idempotent** under at-least-once delivery ([[transactional-outbox]]). Database-per-service is the precondition.
- **CQRS and Event Sourcing are independent, optional power tools.** Scope CQRS to the portion that needs it; use ES only when you truly need audit/replay/temporal. Both "add risky complexity"; most of the system should stay CRUD ([[young-cqrs-es]]). The in-process seed is the aggregate: one transaction per aggregate, events across them ([[ddd-aggregates-tactical]]).

## Evolution & decisions (keep the bet revisable)

- **Make decisions, record them.** Surface the trade-off, pick by the critical **architecture characteristics** for *this* system, and write an **ADR** so the "why" survives ([[richards-ford-fundamentals]], [[nygard-release-it]]).
- **Communicate at one zoom level.** C4 (Context → Container → Component → Code) is the shared notation; a Container diagram is where a distributed monolith becomes visible ([[c4-model]]).
- **Protect characteristics over time with fitness functions.** Automated checks that fail the build when a protected property erodes turn "keep the dependency rule" or "p99 < X" from aspiration into enforcement ([[richards-ford-fundamentals]]).
- **Engineer stability into the edges.** Every integration point is a failure entry: timeouts, circuit breakers, bulkheads, fail-fast ([[nygard-release-it]]). Choosing async/event integration is itself a resilience decision — it removes the synchronous coupling these patterns otherwise defend.

## Through-line to the learner's system

For a production sales agent over many SaaS tools: every external API response is **outside data** (versioned, possibly-stale snapshot, never authoritative live state). Separate the agent's **inside model** from what it ingests/emits, default to a **modular monolith** with clean bounded contexts, and only extract a service when granularity disintegrators clearly outweigh integrators. Every pattern here is priced as a bet — pick the ones that keep *your* expected changes cheap.

## Open gaps (see [[COLLECTION-PLAN]] for the full log)

- Cockburn .us cert + Fielding ics.uci cert + Helland Queue reprint hit fetch errors; all corroborated via mirrors/summaries — re-fetch when available for deeper verbatim.
- Sagas 1987 origin paper is image-only PDF → thesis extracted, not verbatim.
- GraphQL/gRPC live as trade-off context inside [[fielding-rest]]; promote to a standalone excerpt only if the outline gives API-style selection its own chapter.
