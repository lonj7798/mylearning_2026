<!-- scope: Fielding's REST (dissertation ch.5) + Richardson Maturity Model + API style trade-offs
     see-also: consumer-driven-contracts, richards-ford-hard-parts
-->

# REST, the Richardson Maturity Model, and API Style Trade-offs (Fielding; Fowler/Richardson)

- **Core Insight:** REST is a set of *constraints* (statelessness, uniform interface, hypermedia) chosen to buy web-scale properties — caching, evolvability, independent components — and most "REST APIs" are really Level-2 HTTP-RPC that skip the constraint (hypermedia) that makes it actually REST.
- **Guideline:** Pick the API style by the property you need: REST/HTTP for cacheable, evolvable, loosely-coupled resources; **gRPC** for low-latency internal service-to-service; **GraphQL** for client-driven aggregation over many resources. Then version and contract-test whatever you pick.
- **Source:** Roy Fielding, dissertation Ch.5 (roy.gbiv.com/pubs/dissertation/rest_arch_style.htm); Martin Fowler, "Richardson Maturity Model" (martinfowler.com/articles/richardsonMaturityModel.html). gRPC/GraphQL trade-offs: synthesis. NOTE: ics.uci.edu mirror had a **broken TLS chain**; quotes are from Fielding's own gbiv.com mirror.
- **Relevant chapters:** api-design-contracts, event-driven-architecture, architectural-decisions.

## REST as a style of constraints (Fielding, verbatim)

> "REST is a hybrid style derived from several of the network-based architectural styles… combined with additional constraints that define a uniform connector interface." — Fielding

The six constraints:

1. **Client–Server** — "By separating the user interface concerns from the data storage concerns, we improve the portability of the user interface… and improve scalability."
2. **Stateless** — "Each request from client to server must contain all of the information necessary to understand the request, and cannot take advantage of any stored context on the server."
3. **Cache** — "data within a response… [is] implicitly or explicitly labeled as cacheable or non-cacheable."
4. **Uniform Interface** — "The central feature that distinguishes the REST architectural style… is its emphasis on a uniform interface between components." Its four sub-constraints: "identification of resources; manipulation of resources through representations; self-descriptive messages; and, hypermedia as the engine of application state."
5. **Layered System** — components "cannot 'see' beyond the immediate layer."
6. **Code-on-Demand** (optional) — extend clients by downloading code; "only an optional constraint."

Definitions: a resource is "any information that can be named"; a representation captures "the current or intended state of that resource."

## Richardson Maturity Model (Fowler / Leonard Richardson)

A ladder toward Fielding's REST:

| Level | Name | What it adds |
|---|---|---|
| 0 | "Swamp of POX" | HTTP "as a tunneling mechanism… based on Remote Procedure Invocation" — one URI, one verb |
| 1 | Resources | "rather than making all our requests to a singular service endpoint, we… talk to individual resources" |
| 2 | HTTP Verbs | "using the HTTP verbs as closely as possible to how they are used in HTTP itself" (GET safe, status codes) |
| 3 | Hypermedia (HATEOAS) | controls that "tell us what we can do next, and the URI… to do it" |

The crucial caveat:

> "Roy Fielding has made it clear that level 3 RMM is a pre-condition of REST." — Fowler

⇒ Almost everything the industry calls "REST" is Level 2 (HTTP-RPC). That's fine — but it's the doc-vs-reality gap to be honest about (see [[COLLECTION-PLAN]]).

## Style trade-offs (synthesis for the learner)

- **REST/HTTP** — cacheable, evolvable, ubiquitous; weak typing, over/under-fetching, chatty for graphs.
- **gRPC** — binary, contract-first (protobuf), streaming, low latency; great *internal* service-to-service; poor browser/edge ergonomics.
- **GraphQL** — client picks exactly the fields, one round-trip for a graph; pushes complexity to the server (N+1, caching, auth per field).

## Versioning, idempotency, compatibility

The contract is the expensive-to-reverse artifact. Prefer **backward/forward-compatible additive change** (tolerant reader; never remove/repurpose a field) over versioned breakage; make writes **idempotent** (idempotency keys) so retries are safe under at-least-once delivery; verify with **contract testing** → [[consumer-driven-contracts]].

## Connections

- Evolving the contract safely → [[consumer-driven-contracts]].
- Contracts as a hard-part trade-off → [[richards-ford-hard-parts]].
