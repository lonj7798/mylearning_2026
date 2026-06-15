<!-- scope: CQRS + Event Sourcing — definitions, when/when-not, the complexity cautions (Young, Fowler)
     deps: ddd-aggregates-tactical
     see-also: richardson-saga, transactional-outbox, helland-data-outside-inside
-->

# CQRS and Event Sourcing — Greg Young / Fowler

- **Core Insight:** CQRS (separate write model from read model) and Event Sourcing (state = the log of events) are *independent* power tools that compose well — and each one adds real complexity you should refuse to pay unless a specific force demands it.
- **Guideline:** Reach for CQRS only on the *portion* of a system where read and write models genuinely diverge (complex domain, or asymmetric read/write scaling). Reach for Event Sourcing only when you truly need audit/replay/temporal queries. Most of the system should stay plain CRUD.
- **Source:** Greg Young (origin of CQRS, talks/blog — attributed via Fowler); Martin Fowler, "CQRS" (martinfowler.com/bliki/CQRS.html) and "Event Sourcing" (martinfowler.com/eaaDev/EventSourcing.html).
- **Relevant chapters:** event-driven-architecture, domain-driven-design, cross-cutting-concerns.

## CQRS — the definition

> "CQRS stands for Command Query Responsibility Segregation. At its heart is the notion that you can use a different model to update information than the model you use to read information." — Fowler

> "It's a pattern that I first heard described by Greg Young." — Fowler

## CQRS — the caution (this is the load-bearing part)

> "For some situations, this separation can be valuable, but beware that for most systems CQRS adds risky complexity." — Fowler

Scope it tightly:

> "CQRS should only be used on specific portions of a system… and not the system as a whole." — Fowler

And the default:

> "Many systems do fit a CRUD mental model, and so should be done in that style." — Fowler

Legitimate triggers: a complex domain better served by separate read/write models, or high-performance needs that demand independently scaled read and write paths.

## Event Sourcing — the definition

> "Capture all changes to an application state as a sequence of events." — Fowler

State is rebuilt by replaying events:

> "We can discard the application state completely and rebuild it by re-running the events from the event log on an empty application." — Fowler

Benefits: a free audit log ("easy to serialize the events to make an Audit Log"), temporal queries ("determine the application state at any point in time"), and debugging by replay.

## Event Sourcing — the caution

> "Clearly this stuff can get very messy, don't go down this path unless you really need to." — Fowler

The sharp edges: replaying events that triggered external side-effects (must gateway/suppress them), and schema evolution of historical events. "Packaging up every change… as an event is an interface style that not everyone is comfortable with."

## How they relate (and the myth to kill)

CQRS and Event Sourcing are **separate** decisions that *compose*: an event store is a natural write side, and projections build read models — "CQRS naturally aligns with event-based architectures." But **CQRS does not require Event Sourcing, and Event Sourcing does not require CQRS.** Treating them as a package deal is a common, costly conflation. Neither requires microservices either; both work inside a modular monolith.

## Connections

- The in-process root of these ideas → [[ddd-aggregates-tactical]] (events out of an aggregate).
- Publishing events reliably → [[transactional-outbox]].
- Coordinating multi-step writes across services → [[richardson-saga]].
- Why outside data is event-shaped and immutable → [[helland-data-outside-inside]].
