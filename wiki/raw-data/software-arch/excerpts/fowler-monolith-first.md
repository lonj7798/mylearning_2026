<!-- scope: Fowler's MonolithFirst argument + the modular-monolith middle ground + the distributed-monolith trap
     deps: fowler-microservices
     see-also: distributed-monolith, newman-building-microservices, martin-strangler-fig
-->

# MonolithFirst, the Modular Monolith, and Why Boundaries Come First

- **Core Insight:** You cannot design good service boundaries before you understand the domain, and you only understand the domain by building it — so the lowest-risk path to microservices is *through* a monolith, not around it.
- **Guideline:** Default to a **modular monolith** with clean internal boundaries. Extract services only when a specific module's deploy/scale/team pressure justifies the distributed tax. Never ship a "microservice from scratch" architecture into an unfamiliar domain.
- **Source:** Martin Fowler, "MonolithFirst" (martinfowler.com/bliki/MonolithFirst.html, 2015); modular-monolith framing synthesized from Newman & community sources.
- **Relevant chapters:** monolith-vs-microservices, service-decomposition, architectural-decisions.

## The empirical claims (verbatim)

> "Almost all the successful microservice stories have started with a monolith that got too big and was broken up." — Fowler

> "Almost all the cases where I've heard of a system that was built as a microservice system from scratch, it has ended up in serious trouble." — Fowler

## The reasoning

Boundaries are the hard part, and they are hard *early*:

> "Even experienced architects working in familiar domains have great difficulty getting boundaries right at the beginning." — Fowler

And once you have services, fixing a wrong boundary is expensive: refactoring functionality between services is "much harder than it is in a monolith." Inside a monolith a bad boundary is a refactor; across services it's a migration. This is the course's central bet restated — **architecture = the decisions that are expensive to reverse**, and premature service boundaries are exactly such a decision.

The recommendation:

> "Start a new application as a monolith initially, even if you think it's likely that it will benefit from a microservices architecture later on." — Fowler

The economic frame: microservices carry a **MicroservicePremium** that only complex systems amortize.

## The modular monolith (the middle ground)

A **modular monolith** keeps the single deployable unit but enforces strict internal module boundaries (one schema/namespace per module, communication via in-process interfaces, no reaching into another module's tables). It captures most of microservices' *organizational* benefit (clear ownership, enforced cohesion) with none of the *distribution* tax. It is also the ideal staging ground: once module seams are clean, extraction to a service is mechanical. This is the design the learner's production agent most likely wants first.

## The trap on the other side

The failure isn't only "premature microservices" — it's **premature *bad* microservices** that end up as a [[distributed-monolith]]: services that must deploy together and call each other synchronously, paying the distribution tax while keeping monolith coupling.

## Connections

- The migration mechanism out of a monolith → [[martin-strangler-fig]].
- The deployment-coupling anti-pattern → [[distributed-monolith]].
- The decision-cost framing → [[richards-ford-fundamentals]].
