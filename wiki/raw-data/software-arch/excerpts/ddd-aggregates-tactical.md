<!-- scope: tactical DDD — aggregates, entities, value objects, domain events (Evans + Vernon)
     deps: ddd-bounded-context
     see-also: young-cqrs-es, richardson-saga
-->

# Tactical DDD — Aggregates, Consistency Boundaries, Domain Events

- **Core Insight:** The **aggregate** is the unit of *transactional consistency*. Its boundary decides what you can change atomically — and therefore where you must fall back to eventual consistency between units.
- **Guideline:** Design aggregates **small**; one transaction modifies **one** aggregate; reference other aggregates **by identity**, not by object reference; coordinate across aggregates with **domain events**, not nested writes.
- **Source:** Eric Evans, *Domain-Driven Design* (2003) and Vaughn Vernon, *Implementing DDD* + *DDD Distilled* — **books, theses extracted, not quoted verbatim**. Vernon's four aggregate rules are the load-bearing claim.
- **Relevant chapters:** domain-driven-design, event-driven-architecture, cross-cutting-concerns.

## The vocabulary

- **Entity** — identity persists over time even as attributes change (an `Order` is the same order tomorrow).
- **Value object** — no identity; defined wholly by its attributes, immutable (a `Money(amount, currency)`).
- **Aggregate** — a cluster of entities + value objects treated as one unit for data changes, fronted by a single **aggregate root** that is the only external entry point.
- **Domain event** — a record that something meaningful happened in the domain (`OrderPlaced`), used to propagate change across aggregate/context boundaries.

## Vernon's aggregate design rules (the actionable core)

1. **Model true invariants in consistency boundaries.** The aggregate boundary = the set of objects that must be consistent *together, immediately*. Everything outside can be eventually consistent.
2. **Design small aggregates.** Large aggregates create contention and slow loads. Prefer many small aggregates over few large ones.
3. **Reference other aggregates by identity.** Hold an `OrderId`, not an `Order` object. This keeps the boundary crisp and prevents accidental large object graphs.
4. **Use eventual consistency outside the boundary.** One transaction = one aggregate. When another aggregate must react, publish a **domain event** and let it update in a separate transaction.

## Why this is a design-altitude idea, not a coding tip

The aggregate boundary is where the course's consistency thread becomes concrete. Rule 4 is the in-process seed of the distributed [[richardson-saga]]: a saga is "one transaction per aggregate, coordinated by events" stretched across services. If you internalize aggregate boundaries in a modular monolith, the later service split inherits correct consistency boundaries for free. Get the aggregate wrong (too big) and you've baked a contention point into the architecture that is expensive to reverse.

## Connections

- The strategic boundary these live inside → [[ddd-bounded-context]].
- Domain events as the integration mechanism → [[young-cqrs-es]].
- Cross-aggregate / cross-service consistency → [[richardson-saga]], [[helland-data-outside-inside]].
