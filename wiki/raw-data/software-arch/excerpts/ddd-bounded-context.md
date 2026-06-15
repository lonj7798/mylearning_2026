<!-- scope: DDD strategic design — bounded context, ubiquitous language, context mapping (Evans + Fowler)
     see-also: ddd-aggregates-tactical, decompose-by-business-capability, fowler-microservices
-->

# Domain-Driven Design — Bounded Context & Ubiquitous Language (Strategic DDD)

- **Core Insight:** A single unified model for a large domain is neither feasible nor cost-effective; instead, draw boundaries where the *language* changes, and inside each boundary keep one rigorously consistent model.
- **Guideline:** Find your **Bounded Contexts** before you find your services. The seam where a word like "Customer" stops meaning the same thing is the seam where you split the model — and, often, the service.
- **Source:** Eric Evans, *Domain-Driven Design* (2003) — **book, thesis extracted, not quoted verbatim** except where Fowler quotes Evans; Martin Fowler, "BoundedContext" (martinfowler.com/bliki/BoundedContext.html). Strategic-design overview cross-checked with Vaughn Vernon, *DDD Distilled*.
- **Relevant chapters:** domain-driven-design, service-decomposition, monolith-vs-microservices.

## The core problem (Evans, via Fowler)

> "Total unification of the domain model for a large system will not be feasible or cost-effective." — Eric Evans

As an organization grows, "different groups of people will use subtly different vocabularies in different parts of a large organization" (Fowler), and a single model forced across them accumulates contradictions.

## Bounded Context

> "A Bounded Context is a central pattern in Domain-Driven Design." — Fowler

It is an explicit boundary (a subsystem, a module, a service) inside which one model and one **Ubiquitous Language** hold without contradiction. "DDD divides up a large system into Bounded Contexts, each of which can have a unified model."

## Ubiquitous Language drives the boundary

The model *is* a shared language between developers and domain experts. The signal to split:

> "You need a different model when the language changes." — Fowler

## The polysemy example (the canonical teaching case)

The same word means different things in different contexts. "Customer" and "Product" are the textbook offenders; Fowler's utility example: the word "meter" meant "subtly different things to different parts of the organization." In a Sales context a *Customer* is a lead with a pipeline stage; in a Support context the same *Customer* is an account with open tickets. Forcing one `Customer` class to serve both is how models rot. Two contexts, two models, an explicit translation between them.

## Strategic vs tactical (where this sits)

This excerpt is **strategic DDD** — the large-scale boundary and integration decisions. The building-blocks inside a context (aggregates, entities, value objects, domain events) are **tactical DDD** → [[ddd-aggregates-tactical]].

## Context Mapping (how contexts relate)

Vernon's *DDD Distilled* names the integration patterns between contexts — Partnership, Shared Kernel, Customer-Supplier, Conformist, **Anticorruption Layer** (translate the other context's model so it can't leak into yours), Open Host Service, Published Language. Choosing the relationship is a strategic decision with org consequences (see [[conway-team-topologies]]).

## The myth to kill

"DDD requires microservices" is **false**. DDD is a modeling discipline; it applies equally inside a modular monolith. Bounded contexts can be modules in one deployable. See the reconciliation table in [[COLLECTION-PLAN]].

## Connections

- Tactical building blocks → [[ddd-aggregates-tactical]].
- Capability-based decomposition compared → [[decompose-by-business-capability]].
- Org boundaries mirror context boundaries → [[conway-team-topologies]].
