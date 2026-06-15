<!-- scope: service decomposition by business capability vs by subdomain (Richardson) + finding seams
     deps: ddd-bounded-context
     see-also: conway-team-topologies, fowler-microservices, martin-strangler-fig
-->

# Service Decomposition — by Business Capability vs by Subdomain (Richardson)

- **Core Insight:** Good service boundaries fall on **business** seams (what the business *does*), not technical layers — because business capabilities are the slowest-changing structure you can find, and stable boundaries are the only kind worth committing to.
- **Guideline:** Decompose by **business capability** or by DDD **subdomain** (the two converge); cross-check the cut against your org chart (Conway) and your true domain model. Reject layer-based splits ("a UI service, a logic service, a data service") — they guarantee a [[distributed-monolith]].
- **Source:** Chris Richardson, microservices.io/patterns/decomposition/decompose-by-business-capability.html (and the by-subdomain pattern); *Microservices Patterns*.
- **Relevant chapters:** service-decomposition, domain-driven-design, monolith-vs-microservices.

## Business capability — definition

> "A business capability is a concept from business architecture modeling. It is something that a business does in order to generate value." — Richardson

Examples: Order Management, Customer Management, Inventory. The decomposition principle:

> "Define services corresponding to business capabilities." — Richardson

## Why capabilities make stable boundaries

> "Stable architecture since the business capabilities are relatively stable." — Richardson

This is the decomposition restatement of the course thesis: you commit a boundary to the part of the system *least* likely to change. Org structures churn, tech churns, screens churn — what the business fundamentally *does* changes slowly. Cut there and your expensive-to-reverse decision lands on solid ground.

## Capability vs subdomain

> "The Decompose by subdomain pattern is an alternative pattern." — Richardson

- **By business capability** — derived from business-architecture analysis (org structure + value streams). Outside-in.
- **By subdomain (DDD)** — derived from the domain model and its [[ddd-bounded-context|bounded contexts]]. Inside-out.

In practice they converge: a well-found bounded context usually *is* a business capability. Richardson's practical starting points: analyze your "organization structure" and your "high-level domain model."

## Finding the seams (the practitioner's question)

The course's how-to layer: look for clusters of data and behavior that change together and have few references outside the cluster (high cohesion, low coupling); watch where the [[ddd-bounded-context|ubiquitous language]] shifts; respect Conway's Law (a seam that cuts across one team's communication path will be fought constantly → [[conway-team-topologies]]). When in doubt, draw the seam *inside a modular monolith* first and let usage reveal whether it's real before paying to distribute it (→ [[fowler-monolith-first]]).

## The anti-decomposition

Splitting by **technical layer** is the classic mistake: every business change then touches every service, restoring full coupling over the network. That's how teams build a [[distributed-monolith]].

## Connections

- The modeling discipline behind it → [[ddd-bounded-context]].
- The org-mirror constraint → [[conway-team-topologies]].
- Extracting one capability at a time from a legacy monolith → [[martin-strangler-fig]].
