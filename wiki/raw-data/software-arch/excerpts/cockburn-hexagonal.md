<!-- scope: Cockburn's Hexagonal (Ports & Adapters) architecture — intent, ports, primary/secondary adapters
     see-also: martin-clean-arch, ddd-bounded-context
-->

# Hexagonal Architecture (Ports and Adapters) — Cockburn

- **Core Insight:** Make the application core driveable *and* testable in isolation by putting every external interaction — UI, DB, test harness, another app — behind a **port**, with technology-specific **adapters** plugged in on the outside.
- **Guideline:** Depend inward. The application defines ports (interfaces) in its own terms; adapters implement them. Swap a real database for an in-memory fake by swapping the adapter, never by touching the core.
- **Source:** Alistair Cockburn, "Hexagonal Architecture" (alistair.cockburn.us/hexagonal-architecture, 2005; mirror alistaircockburn.com/Hexagonal-Architecture). NOTE: the canonical .us URL had an **expired TLS certificate** at fetch time (2026-06-15); the Intent quote is corroborated via the .com mirror and Cockburn's own wording in search results.
- **Relevant chapters:** hexagonal-clean-architecture, domain-driven-design, cross-cutting-concerns.

## The Intent (verbatim — the one line everyone quotes)

> "Allow an application to equally be driven by users, programs, automated test or batch scripts, and to be developed and tested in isolation from its eventual run-time devices and databases." — Alistair Cockburn

## The motivation it fixes

Conventional layering quietly entangles business logic with both the UI above and the database below, so you can't test the core without a screen and a DB, and you can't swap either without surgery. Hexagonal kills the asymmetry by treating *both* sides as equal "outside" connected through ports.

## Ports and adapters

- A **port** is an interface defined by the application, in the application's own language (e.g. `ForPlacingOrders`, `ForStoringOrders`). "All input and output reaches/leaves the application through a port that isolates the application from external tools, technologies and delivery mechanisms."
- An **adapter** translates between a port and a concrete technology (REST controller, CLI, test driver; Postgres repo, in-memory fake, message-bus client).

## Primary vs secondary (driving vs driven)

The pattern is "deliberately written pretending that all ports are fundamentally similar," but in practice they come in two flavors:

- **Primary / driving adapters** — initiate calls *into* the app (a user via UI, an HTTP client, a test suite). They sit on the "driving" side.
- **Secondary / driven adapters** — the app calls *out* through these (a database, a message broker, an email gateway). The app owns the interface; the adapter obeys it (dependency inversion).

This symmetry is the whole point: "the application can be equally driven by an automated, system-level regression test suite, by a human user, by a remote http application, or by another local application," and on the data side "configured to run decoupled from external databases using an in-memory… database replacement."

## Why a hexagon?

The shape carries no significance of six — Cockburn chose a polygon (not the usual layered rectangle) purely to leave visual room to draw several ports and their adapters around the core. The number of sides is incidental.

## Relationship to Clean/Onion

Hexagonal, Onion, and Clean are the same idea with different diagrams: a technology-free core, dependencies pointing inward, I/O at the edges via interfaces the core owns. → [[martin-clean-arch]] states the rule most crisply ("source code dependencies can only point inwards").

## Connections

- The dependency rule, generalized → [[martin-clean-arch]].
- Ports map naturally onto a bounded context's published interface → [[ddd-bounded-context]].
