<!-- scope: Software Architecture: The Hard Parts — no best practices, architecture quantum, granularity, data decomposition
     deps: richards-ford-fundamentals
     see-also: richardson-saga, transactional-outbox, consumer-driven-contracts
-->

# Software Architecture: The Hard Parts — Richards, Ford, Sadalage, Dehghani

- **Core Insight:** The hard parts of distributed architecture are the ones with *no best practices* — service granularity, data ownership, distributed transactions, contracts — so the only durable skill is **discovering and weighing the trade-offs** for your specific context, not memorizing a recipe.
- **Guideline:** Use the **architecture quantum** (the smallest independently deployable unit with high functional cohesion and its own data) to test a proposed boundary. Then run the explicit granularity *disintegrators vs integrators* analysis before splitting or merging a service.
- **Source:** Neal Ford, Mark Richards, Pramod Sadalage, Zhamak Dehghani, *Software Architecture: The Hard Parts* (2021) — book, thesis extracted; O'Reilly/Amazon descriptions corroborate.
- **Relevant chapters:** service-decomposition, event-driven-architecture, api-design-contracts.

## The stance (verbatim sense)

The book is about "difficult problems… with no best practices that force you to choose among various compromises." It teaches "how to think critically about the trade-offs involved with distributed architectures" and gives "techniques to help you discover and weigh the trade-offs." It is the operationalization of the First Law (→ [[richards-ford-fundamentals]]).

## The architecture quantum (the boundary test)

An **architecture quantum** is the smallest unit that is **independently deployable**, has **high functional cohesion**, and has **synchronous (static + dynamic) connascence** within it but not across it — crucially, it includes its **own data**. A boundary that splits a quantum (e.g. two "services" that share a database, so they can't deploy independently) isn't a real boundary. This is the formal version of the [[distributed-monolith]] detector: count your quanta, not your services.

## Service granularity — the explicit trade-off framework

The book's most actionable contribution: don't argue about service size, *enumerate forces*.

- **Granularity disintegrators** (reasons to split smaller): divergent scalability/throughput, fault isolation, differing security/access, distinct code-volatility, separate team ownership.
- **Granularity integrators** (reasons to keep together / merge): a database transaction must span them, tight data dependencies, heavy chatty workflow/orchestration between them, shared code that changes together.

You split only when disintegrators outweigh integrators. This is the cure for both premature microservices and accidental distributed monoliths.

## Data decomposition & distributed transactions

Breaking a monolithic database is the genuinely hard part: identify data domains, assign each to a quantum, and accept that cross-quantum operations now need **sagas** (→ [[richardson-saga]]), reliable event emission (→ [[transactional-outbox]]), and explicit, evolvable **contracts** (→ [[consumer-driven-contracts]]). The book frames contract coupling on a spectrum from strict (tight, brittle, easy to validate) to loose (flexible, evolvable, harder to verify) — another trade-off to choose deliberately, not by default.

## Connections

- The trade-off-thinking foundation → [[richards-ford-fundamentals]].
- The mechanisms data decomposition forces → [[richardson-saga]], [[transactional-outbox]], [[consumer-driven-contracts]].
- The anti-pattern the quantum detects → [[distributed-monolith]].
