<!-- scope: Pat Helland "Data on the Outside vs Data on the Inside" — inside vs outside data, immutability, no shared txns
     deps: richardson-saga, transactional-outbox
     see-also: young-cqrs-es, distributed-monolith
-->

# Data on the Outside vs Data on the Inside — Pat Helland

- **Core Insight:** There are two fundamentally different kinds of data — **inside** data (private, mutable, transactional, "now") and **outside** data (the messages between services: immutable, versioned, identified, possibly stale) — and confusing them is the root cause of most distributed-system pain.
- **Guideline:** Inside a service boundary, use SQL and ACID freely. The instant data crosses a boundary it must become **immutable, versioned, and identity-stamped**, and you must accept it may be stale — because you cannot hold a lock or share a transaction across services.
- **Source:** Pat Helland, "Data on the Outside versus Data on the Inside," CIDR 2005 (cidrdb.org/cidr2005/papers/P12.pdf; ACM Queue reprint 2020, queue.acm.org/detail.cfm?id=3415014 — Queue HTML returned **403** at fetch; claims corroborated via Semantic Scholar / the morning paper summary).
- **Relevant chapters:** event-driven-architecture, service-decomposition, cross-cutting-concerns.

## The two kinds of data

- **Inside data** — "the realm of SQL and SQL's DDL." It is private to a service, **mutable**, transactional, and represents the *current* truth ("now"). ACID applies; you can lock it.
- **Outside data** — the data that travels *between* services as messages. It "is immutable and each data item's schema is versioned." It "is stable, such that a repeated request is unchanged, and a reading of it results in the same interpretation."

## The three load-bearing claims

1. **Services do not share transactions.** You cannot wrap a transaction around two services. This is *the* reason distributed transactions/2PC across services are a dead end (and why you need [[richardson-saga|sagas]] and the [[transactional-outbox|outbox]]).
2. **Outside data must be immutable.** "Messages themselves must also be immutable" — their content "should never change across retries." An immutable, identified message is safe to retry, cache, reorder, and replay; a mutable one is not. This is exactly why event-sourced/event-driven systems work (→ [[young-cqrs-es]]).
3. **Outside data may be stale, and that's fine.** Because you can't lock across the boundary, the data you receive is a *snapshot* — true as of when it was sent. You design for **eventual consistency** at the application level, accepting and reconciling staleness rather than pretending it isn't there.

## Why this is the deepest cut in the library

Helland gives the *theoretical reason* every other consistency pattern in this course exists. Aggregate boundaries (→ [[ddd-aggregates-tactical]]), database-per-service, sagas, outbox, CQRS read models, immutable domain events — all are consequences of one fact: **the moment data crosses a service boundary, you lose locks and shared transactions, so the data must become immutable, versioned, and accepted-as-possibly-stale.** A [[distributed-monolith]] is what you get when you ignore Helland and keep treating cross-boundary data as if it were inside data (shared DB, synchronous "live" reads, distributed locks).

## Connection to the learner's system

For a production sales agent integrating many SaaS systems, every external API response *is* outside data: a versioned, possibly-stale snapshot you must not treat as authoritative live state. Designing the agent's internal model (inside data) separately from what it ingests/emits (outside data) is the single most leverage-rich boundary decision available.

## Connections

- The transaction-coordination consequences → [[richardson-saga]], [[transactional-outbox]].
- Immutable outside data = events → [[young-cqrs-es]].
- The failure mode of ignoring the distinction → [[distributed-monolith]].
