<!-- scope: Transactional Outbox + Database-per-Service — the dual-write problem and its fix (Richardson)
     deps: richardson-saga
     see-also: young-cqrs-es, helland-data-outside-inside
-->

# Transactional Outbox & Database per Service — Richardson

- **Core Insight:** You cannot atomically write to your database *and* publish to a message broker — so don't try. Write the outgoing message into your DB *in the same local transaction*, then let a separate relay forward it. Atomicity is preserved because there is only one commit.
- **Guideline:** Keep each service's data private behind its API (Database per Service); whenever a state change must be announced, persist the event to an **outbox** table inside the business transaction and relay it asynchronously. Consumers must be **idempotent** (at-least-once delivery).
- **Source:** Chris Richardson, microservices.io/patterns/data/transactional-outbox.html and microservices.io/patterns/data/database-per-service.html; *Microservices Patterns* (book).
- **Relevant chapters:** event-driven-architecture, api-design-contracts, service-decomposition.

## Database per Service (the precondition)

> "Keep each microservice's persistent data private to that service and accessible only via its API." — Richardson

A shared database is an anti-pattern: it creates an implicit, unversioned contract and runtime coupling. The trade-off it forces:

> "Implementing business transactions that span multiple services is not straightforward… Implementing queries that join data that is now in multiple databases is challenging." — Richardson

Richardson's prescribed answers: **[[richardson-saga]]** for cross-service transactions, **API Composition** or **CQRS** for cross-service queries. The constraint that drives everything: "A service's transactions only involve its database."

## The dual-write problem

> "How to atomically update the database and send messages to a message broker?" — Richardson

The naive approach — commit the DB, then publish — can crash between the two steps, losing the message (or, reordered, publishing a message for a write that rolls back). And:

> "It is not viable to use a traditional distributed transaction (2PC) that spans the database and the message broker." — Richardson

## The Outbox solution

> "The solution is for the service that sends the message to first store the message in the database as part of the transaction that updates the business entities." — Richardson

The guarantee:

> "Messages are guaranteed to be sent if and only if the database transaction commits." — Richardson

## The relay (two implementations)

> "Two patterns for implementing the Message relay: The Transaction log tailing pattern [and] The Polling publisher pattern." — Richardson

- **Polling publisher** — a process polls the outbox table for unsent rows and publishes them. Simple; adds DB load and latency.
- **Transaction log tailing (CDC)** — tail the DB's commit log (e.g. via Debezium) and publish committed outbox inserts. Lower latency; more infra.

Either way delivery is **at-least-once**, so consumers must dedupe (idempotency keys). This is why idempotency, an API-contract concern (see [[richards-ford-hard-parts]] / API-design), and the outbox, a data concern, are the same conversation.

## Connections

- The transaction it sits inside → [[richardson-saga]] (each saga step = local txn + outbox event).
- The events it emits are "outside data" → [[helland-data-outside-inside]].
- Read-side projections fed by these events → [[young-cqrs-es]].
