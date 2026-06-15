<!-- scope: Saga pattern — Richardson's microservices framing + Garcia-Molina's original LLT paper
     deps: ddd-aggregates-tactical, database-per-service
     see-also: transactional-outbox, young-cqrs-es, helland-data-outside-inside
-->

# Saga — Distributed Transactions Without 2PC (Richardson; Garcia-Molina origin)

- **Core Insight:** Once each service owns its own database, you cannot get an ACID transaction across services — so you replace it with a **sequence of local transactions** glued by events, and undo failures with **compensating transactions** instead of a rollback.
- **Guideline:** Use a saga when one business operation must update data in several services. Accept that you trade *isolation* (the "I" in ACID) for availability, and add explicit countermeasures for the anomalies that loss of isolation creates.
- **Source:** Chris Richardson, microservices.io/patterns/data/saga.html and *Microservices Patterns* (book). Origin: Hector Garcia-Molina & Kenneth Salem, "Sagas," SIGMOD 1987 — **PDF was image-only (no text layer) at fetch; thesis extracted from knowledge of the paper, not quoted verbatim.**
- **Relevant chapters:** event-driven-architecture, service-decomposition, cross-cutting-concerns.

## Why sagas exist

> "The Database per Service pattern creates the need for this pattern." — Richardson

No shared DB ⇒ no distributed ACID transaction. (And "it is not viable to use a traditional distributed transaction (2PC) that spans the database and the message broker," per [[transactional-outbox]].)

## The definition

> "A saga is a sequence of local transactions. Each local transaction updates the database and publishes a message or event to trigger the next local transaction in the saga." — Richardson

## The original idea (Garcia-Molina & Salem, 1987 — thesis, not verbatim)

The 1987 paper introduced sagas for **long-lived transactions (LLTs)** in a single database: an LLT is split into sub-transactions T1…Tn, each with a **compensating transaction** C1…Cn that semantically undoes it. The guarantee is that the system runs *either* T1…Tn *or* a prefix T1…Tj followed by Cj…C1 — never a stuck partial state. Crucially, it **relaxes isolation**: other transactions see intermediate sub-transaction results. Richardson's contribution is recognizing this 1987 single-DB construct is exactly the tool for cross-*service* consistency.

## Two coordination styles

> "Choreography - each local transaction publishes domain events that trigger local transactions in other services." — Richardson
> "Orchestration - an orchestrator (object) tells the participants what local transactions to execute." — Richardson

| | Choreography | Orchestration |
|---|---|---|
| Control | decentralized; services react to events | a central orchestrator drives steps |
| Coupling | lower, fewer dependencies | higher, but logic in one place |
| Visibility | hard to trace the whole flow | one place to see/debug the process |
| Best when | simple, naturally event-driven flow | complex, many interdependent steps |
| Risk | distributed logic, monitoring burden | orchestrator becomes a bottleneck |

## The price: no isolation

> "Lack of isolation (the 'I' in ACID)… means there's risk that the concurrent execution of multiple sagas and transactions can [cause] data anomalies." — Richardson

So "a saga developer must typically use countermeasures" — semantic locks, commutative updates, pessimistic views, re-reads, by-value tracking — to tame anomalies. This is the design-altitude cost the learner must price in: a saga is *not* a free transaction.

## Connections

- The data-ownership precondition → [[database-per-service]].
- Reliably emitting the step events → [[transactional-outbox]].
- In-process precursor (one txn per aggregate) → [[ddd-aggregates-tactical]].
