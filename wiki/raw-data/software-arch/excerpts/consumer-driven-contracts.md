<!-- scope: Consumer-Driven Contracts + contract testing + tolerant reader (Fowler/Robinson)
     deps: fielding-rest
     see-also: richards-ford-hard-parts, transactional-outbox
-->

# Consumer-Driven Contracts & Contract Testing — Fowler / Robinson

- **Core Insight:** A service interface can evolve safely only if you know *which parts each consumer actually depends on* — so let consumers express their expectations as executable contracts, and the provider treats those as its real obligations.
- **Guideline:** Be "conservative in what you send, liberal in what you accept" (tolerant reader). Capture each consumer's expectations as a contract test in the provider's pipeline; only changes that break a real consumer contract are breaking changes.
- **Source:** Ian Robinson, "Consumer-Driven Contracts" (martinfowler.com/articles/consumerDrivenContracts.html); tolerant-reader / Postel's Law framing from Fowler's writing. Tooling (Pact, Spring Cloud Contract): synthesis.
- **Relevant chapters:** api-design-contracts, service-decomposition, event-driven-architecture.

## The inversion

Instead of consumers adapting to whatever the provider ships, the provider adopts the *reasonable expectations consumers express*:

> "When a provider accepts and adopts the reasonable expectations expressed by a consumer, it enters into a consumer contract." — Robinson

Why it's powerful:

> "Consumer contracts allow us to reflect on the business value being exploited at any point in a provider's lifetime… [they] define which parts of that provider contract currently support the business value realized by the system." — Robinson

## Tolerant reader / Postel's Law

> "An implementation must be conservative in its sending behaviour and liberal in its receiving behaviour… message receivers should implement 'just enough' validation: that is, they should only process data that contributes to the business functions they implement." — Robinson

Consequence: a consumer that ignores unknown fields lets the provider *add* fields freely (forward compatibility). Strict schema validation on the consumer side is what turns additive changes into breaking ones.

## Safe evolution

> "Consumer-driven provider contracts give us the fine-grained insight and rapid feedback we require to plan changes and assess their impact on applications currently in production." — Robinson

## Contract testing in practice (synthesis)

- The consumer publishes a **pact** (expected request/response shapes); the provider runs it in CI and fails the build if it would break a real consumer.
- This replaces brittle end-to-end integration tests with fast, isolated, per-pair checks — and it's the only mechanism that lets services keep their prized **independent deployability** (a provider knows, before deploy, whether it breaks anyone). Ties directly to [[newman-building-microservices]].

## Why this matters at design altitude

The published contract is the **most expensive-to-reverse** artifact a service owns: once consumers depend on it, breaking it costs coordinated multi-team migration — exactly the cost microservices exist to avoid. Contract discipline is how you keep the boundary cheap to evolve.

## Connections

- The API styles whose contracts these govern → [[fielding-rest]].
- Idempotency + at-least-once delivery (why retries need stable contracts) → [[transactional-outbox]].
- Contracts as a named "hard part" trade-off → [[richards-ford-hard-parts]].
