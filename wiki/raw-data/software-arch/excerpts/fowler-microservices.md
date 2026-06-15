<!-- scope: Lewis & Fowler's definition of microservices + the 9 characteristics + the trade-off frame
     see-also: newman-building-microservices, decompose-by-business-capability, conway-team-topologies
-->

# Microservices — Definition, Nine Characteristics, and the Trade-off

- **Core Insight:** Microservices is a style that buys *independent deployability* and *strong module boundaries* at the price of *distribution, eventual consistency, and operational overhead* — a bet that only pays off above a complexity threshold ("you must be this tall").
- **Guideline:** Don't adopt microservices for the buzzword. Adopt them when team-scale coordination cost on a monolith exceeds the distributed-systems tax — and start by getting the *boundaries* right (see [[fowler-monolith-first]]).
- **Source:** James Lewis & Martin Fowler, "Microservices" (martinfowler.com/articles/microservices.html, 2014); "Microservice Trade-Offs" (martinfowler.com/articles/microservice-trade-offs.html).
- **Relevant chapters:** monolith-vs-microservices, service-decomposition, cross-cutting-concerns.

## The definition (verbatim)

> "The microservice architectural style is an approach to developing a single application as a suite of small services, each running in its own process and communicating with lightweight mechanisms, often an HTTP resource API." — Lewis & Fowler

Contrast: a monolith is "a single unit" where "any changes to the system involve building and deploying a new version of the server-side application." The differentiator is **independent deployability**, not size.

## The nine characteristics (Lewis & Fowler)

1. **Componentization via services** — out-of-process components; the payoff is independent deployment.
2. **Organized around business capabilities** — cross-functional teams own a "broad-stack implementation" (UI + storage + collaborations); a Conway's-Law consequence.
3. **Products not projects** — "you build it, you run it"; own the product over its full lifetime.
4. **Smart endpoints and dumb pipes** — logic in services, plumbing kept simple; "as decoupled and as cohesive as possible" over "simple RESTish protocols."
5. **Decentralized governance** — pick the right tech per service ("Node.js for a reports page… C++ for a gnarly near-real-time component? Fine").
6. **Decentralized data management** — each service owns its database; polyglot persistence. → [[richardson-saga]] for the consequence.
7. **Infrastructure automation** — CI/CD, automated testing and deploy are prerequisites, not nice-to-haves.
8. **Design for failure** — "any service call could fail due to unavailability"; tolerate it (Netflix Simian Army).
9. **Evolutionary design** — services are "replaceable rather than evolved"; refactor boundaries as understanding grows.

## The trade-off (from "Microservice Trade-Offs")

| Benefit | Fowler's words | Cost | Fowler's words |
|---|---|---|---|
| Strong module boundaries | "reinforce modular structure… important for larger teams" | Distribution | "remote calls are slow and… always at risk of failure" |
| Independent deployment | "autonomous… less likely to cause system failures" | Eventual consistency | "everyone has to manage eventual consistency" |
| Technology diversity | "mix multiple languages… data-storage technologies" | Operational complexity | "need a mature operations team" |

> "There is a Microservice Premium: microservices impose a cost on productivity that can only be made up for in more complex systems." — Fowler

The eventual-consistency cost is *application-level*, not infra-level: "Business logic can end up making decisions on inconsistent information." This is the design-altitude consequence the course cares about.

## The authors' own caution

> "We write this with cautious optimism." — Lewis & Fowler

They explicitly decline to declare microservices superior. That hedge is itself the thesis: it's a trade, not an upgrade.

## Connections

- The "don't start here" corollary → [[fowler-monolith-first]].
- The deployment-coupling failure mode → [[distributed-monolith]].
- Boundaries done by business capability → [[decompose-by-business-capability]]; by subdomain → [[ddd-bounded-context]].
