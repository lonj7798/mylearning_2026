<!-- scope: Robert C. Martin's Clean Architecture — the Dependency Rule, concentric circles, independence claims
     deps: cockburn-hexagonal
     see-also: ddd-bounded-context, richards-ford-fundamentals
-->

# Clean Architecture — The Dependency Rule (Robert C. Martin)

- **Core Insight:** There is exactly one architecture-level rule that makes a system testable and framework-independent: **source-code dependencies point only inward**, toward policy, never toward detail.
- **Guideline:** Put enterprise rules at the center, application rules around them, and frameworks/UI/DB at the rim. When an inner layer must call out, invert the dependency: define the interface inside, implement it outside.
- **Source:** Robert C. Martin, "The Clean Architecture" (blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html, 2012) and *Clean Architecture* (2017, book — thesis extracted).
- **Relevant chapters:** hexagonal-clean-architecture, architectural-decisions, cross-cutting-concerns.

## The Dependency Rule (the whole architecture in one sentence)

> "Source code dependencies can only point inwards." — Robert C. Martin

Nothing in an inner circle may name anything in an outer circle. A detail (a framework, a DB driver) may depend on a policy; a policy may never depend on a detail.

## The concentric circles (inner → outer)

1. **Entities** — enterprise-wide business rules; "least likely to change."
2. **Use Cases** — application-specific business rules; orchestrate the flow of data to and from entities.
3. **Interface Adapters** — convert data between use-case form and external form (controllers, presenters, gateways, MVC).
4. **Frameworks and Drivers** — the web framework, the database, the devices; "tools rather than constraints."

## The independence claims (verbatim)

The architecture exists to make the system:

- **"Independent of Frameworks"** — the framework is a tool you call, not a base class you inherit your whole app from.
- **"Testable"** — "business rules can be tested without UI, database, or external elements."
- **"Independent of UI, Database, [and] any external agency"** — each is replaceable without touching core logic.

## Crossing boundaries cleanly

> "The important thing is that isolated, simple, data structures are passed across boundaries." — Martin

Don't pass an ORM row or a framework request object inward; it drags an outer dependency in with it. Pass a plain DTO. This is the mechanism that keeps the Dependency Rule from being violated by the type system.

## Same idea, three names

Clean = [[cockburn-hexagonal]] (Ports & Adapters) = Onion. All three: technology-free core, inward-pointing dependencies, I/O at the edges through inverted interfaces. Clean states the *rule* most crisply; Hexagonal gives the *testability* motivation most vividly.

## Trade-off the learner should weigh

The cost is indirection and boilerplate (DTOs, ports, mappers). For a small CRUD service it's overkill. For the learner's long-lived sales-agent core — where business policy must outlive whichever LLM API/vector DB/web framework is current — it's exactly the right bet: it keeps the *expensive-to-reverse* part (domain policy) insulated from the *cheap-to-swap* parts (vendors).

## Connections

- The same architecture, port-flavored → [[cockburn-hexagonal]].
- "Architecture = the hard-to-change decisions" frames *why* you isolate policy → [[richards-ford-fundamentals]].
