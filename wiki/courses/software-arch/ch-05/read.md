<!-- chapter: ch-05
     track: contracts
     kind: content
     title: Integration Contracts: REST, API Styles, and Safe Evolution
     deps: [[ch-04]]
     sources: [[fielding-rest]], [[consumer-driven-contracts]], [[transactional-outbox]], [[newman-building-microservices]], [[richards-ford-hard-parts]]
-->

# Chapter 05 — Integration Contracts: REST, API Styles, and Safe Evolution

> **Core insight.** Once you have cut a boundary (ch-02 through ch-04), the boundary *is* its published contract — and the contract is the single most expensive-to-reverse artifact a service owns. Everything else inside a service is private and refactorable at will; the contract is the one thing other teams have already coupled to, so changing it costs a coordinated multi-team migration. The whole discipline of this chapter is therefore not "how to build a nice API" but "how to keep the boundary cheap to evolve after other people depend on it." REST, the Richardson Maturity Model, gRPC/GraphQL selection, additive versioning, idempotency, and consumer-driven contracts are all moves in that one game: buy yourself the ability to change the inside without renegotiating the edge.

> **Guideline.** Pick the API style by the property you actually need ([[fielding-rest]]) — REST/HTTP for cacheable, evolvable, loosely-coupled resources; gRPC for low-latency internal service-to-service; GraphQL for client-driven aggregation — then defend the contract's evolvability with three disciplines: evolve **additively** (never remove or repurpose a field), make every write **idempotent** so retries are safe under at-least-once delivery, and let consumers express their real expectations as **executable contract tests** in the provider's CI ([[consumer-driven-contracts]]). The only changes that are "breaking" are the ones that break a real consumer's contract — and you should be able to know that *before* you deploy, not after a pager goes off.

---

## 1. The Contract Is the Expensive-to-Reverse Artifact

Ch-04 closed on a hard truth from [[richards-ford-hard-parts]]: a bad boundary inside a monolith is a refactor, but a bad boundary across services is a migration. This chapter takes the next step. The thing that makes a cross-service boundary expensive is not the network hop — it is the **published contract** that other teams have built against. The instant a second team writes code that reads your `stage_name` field or relies on your `POST /opportunities` returning a `201`, that field and that status code stop being yours to change unilaterally.

[[consumer-driven-contracts]] states the design-altitude consequence directly:

> "The published contract is the **most expensive-to-reverse** artifact a service owns: once consumers depend on it, breaking it costs coordinated multi-team migration — exactly the cost microservices exist to avoid." — from [[consumer-driven-contracts]]

This reframes the entire chapter. In ch-04, the architecture quantum from [[richards-ford-hard-parts]] taught you that independent deployability is the property worth protecting. The contract is precisely where independent deployability lives or dies: a service can deploy independently only if it can change its insides without first asking every consumer to redeploy. So the contract is not a side artifact of the boundary; it is the boundary's *load-bearing surface*.

### 1.1 Why this is a course-spine moment, not an API-design tutorial

The course's organizing idea ([[insights]]) is that architecture is the set of decisions that are expensive to reverse. The four pillars place those decisions in order of reversal cost: boundaries (most expensive, deferred to ch-02–04), then the contracts that span them (this chapter), then consistency across them (ch-06–07), then evolution discipline that keeps all of it revisable (ch-08–09).

Contracts sit exactly where they do because they are the *most expensive-to-reverse artifact a service owns* — more expensive than the database schema (private, migratable behind the API), more expensive than the framework choice (swappable behind a port, per ch-03's Dependency Rule from [[martin-clean-arch]]). The only thing more expensive to reverse than a contract is the boundary the contract sits on. That is why this chapter comes after topology and before consistency: you cannot reason about what crossing a boundary costs (ch-06) until you have priced the artifact that defines the crossing.

| Artifact | Owner | Reversal cost | Why |
|---|---|---|---|
| Internal class / module structure | the service | cheap (refactor) | nobody outside sees it |
| Database schema | the service | medium (migration, but private) | hidden behind the API per [[newman-building-microservices]] |
| Framework / vendor SDK | the service | medium (swap behind a port) | insulated by the Dependency Rule, ch-03 |
| **Published contract** | the service **+ every consumer** | **expensive (multi-team migration)** | consumers have already coupled to it |
| The boundary itself | the org | most expensive (re-architecture) | ch-04: cross-service = migration |

---

## 2. REST Is a Set of Constraints, Not a URL Convention

Before pricing how to evolve a contract, you have to be honest about what kind of contract you are shipping. The most-used and most-misused word in API design is "REST," so we start there with the primary source.

REST is not "JSON over HTTP with nice URLs." It is a named architectural style defined by Roy Fielding, assembled from a specific set of constraints chosen to buy specific web-scale properties. Per the gap log in [[COLLECTION-PLAN]], the canonical `ics.uci.edu` copy of the dissertation had a broken TLS chain at crawl time; the quotes below are taken from Fielding's own `roy.gbiv.com` mirror (authoritative), cited as such in [[fielding-rest]]:

> "REST is a hybrid style derived from several of the network-based architectural styles… combined with additional constraints that define a uniform connector interface." — Fielding (via the roy.gbiv.com mirror, per [[fielding-rest]])

### 2.1 The six constraints and what each one buys

Fielding's style is the *sum* of these constraints. Each one is itself a priced bet — it forbids something in exchange for a property.

| Constraint | Fielding's wording (via [[fielding-rest]]) | What it buys | What it costs |
|---|---|---|---|
| Client–Server | "separating the user interface concerns from the data storage concerns… improve[s] the portability… and… scalability" | independent evolution of UI and storage | two parties to coordinate |
| Stateless | "Each request… must contain all of the information necessary to understand the request, and cannot take advantage of any stored context on the server" | horizontal scaling, any server answers any request | every request re-sends context; no cheap sessions |
| Cache | "data within a response… [is] implicitly or explicitly labeled as cacheable or non-cacheable" | intermediaries can serve repeats; latency + load drop | staleness management |
| Uniform Interface | "the central feature that distinguishes the REST architectural style… is its emphasis on a uniform interface between components" | components evolve independently; generic tooling works | the interface is generic, so it is less efficient than a bespoke one |
| Layered System | components "cannot 'see' beyond the immediate layer" | insert proxies, gateways, caches transparently | added latency per layer |
| Code-on-Demand | "only an optional constraint" | extend clients at runtime | the only optional one; rarely used |

The Uniform Interface constraint is the one that does the heavy lifting, and Fielding decomposes it into four sub-constraints (verbatim list via [[fielding-rest]]): "identification of resources; manipulation of resources through representations; self-descriptive messages; and, hypermedia as the engine of application state." Hold onto that fourth one — **hypermedia as the engine of application state (HATEOAS)** — because it is exactly the constraint almost every real "REST API" drops, and it is the one Fielding insists is non-negotiable.

Two definitions to keep the vocabulary precise, both from [[fielding-rest]]: a *resource* is "any information that can be named"; a *representation* captures "the current or intended state of that resource." You never manipulate the resource directly — you exchange representations of it.

---

## 3. The Richardson Maturity Model and the "Our API Is RESTful" Myth

This is the chapter's reconciliation myth, drawn from the table in [[COLLECTION-PLAN]] and the per-chapter myth list in the outline's authoring notes:

> **Popular narrative:** "Our API is RESTful."
> **What the primary source says:** Almost all are **Level 2** (HTTP-RPC) on Richardson's scale. Fowler: "Roy Fielding has made it clear that level 3 [hypermedia] is a pre-condition of REST." Most "REST" isn't, by Fielding's definition. — reconciliation table, [[COLLECTION-PLAN]]

The Richardson Maturity Model (RMM), described by Martin Fowler after Leonard Richardson and quoted in [[fielding-rest]], is a ladder of four levels toward Fielding's REST. It is the single most useful tool for being honest about what you actually built.

| Level | Name | What it adds (Fowler, via [[fielding-rest]]) |
|---|---|---|
| 0 | "Swamp of POX" | HTTP "as a tunneling mechanism… based on Remote Procedure Invocation" — one URI, one verb |
| 1 | Resources | "rather than making all our requests to a singular service endpoint, we… talk to individual resources" |
| 2 | HTTP Verbs | "using the HTTP verbs as closely as possible to how they are used in HTTP itself" (GET safe, status codes) |
| 3 | Hypermedia (HATEOAS) | controls that "tell us what we can do next, and the URI… to do it" |

> **Explore the ladder interactively:** open [`figures/richardson-maturity-model.html`](figures/richardson-maturity-model.html) and click each rung to see a concrete request/response pair and what that level buys versus costs. The figure defaults to Level 2 and visibly marks two facts: most real "REST" APIs stop there, and Fielding requires Level 3 hypermedia before he will call it REST. Toggle the "mark reality" box to see the line Fielding draws.

### 3.1 The caveat that kills the myth

The load-bearing quote, verbatim via [[fielding-rest]]:

> "Roy Fielding has made it clear that level 3 RMM is a pre-condition of REST." — Fowler

The consequence, also from [[fielding-rest]]: almost everything the industry calls "REST" is Level 2 — HTTP-RPC done well. Resources are addressed, verbs are used correctly, status codes mean something, GET is safe and cacheable. That is a genuinely good API. It is just *not REST by Fielding's definition*, because it lacks HATEOAS: the client still hard-codes URI templates and the application's state machine instead of following hypermedia controls the server ships in each response.

### 3.2 Why the honesty matters (and why Level 2 is often the right bet)

This is not pedantry for its own sake. The reason RMM matters at design altitude is that **Level 3 is a priced bet, and most teams correctly decline to pay it**:

- **What Level 3 keeps cheap:** the server owns the URI structure and the workflow, so it can move endpoints and change the state machine without breaking clients that simply follow links. URI evolution becomes free.
- **What Level 3 makes expensive:** clients must be sophisticated enough to navigate hypermedia at runtime, and almost no SDK/codegen ecosystem does this — they assume fixed URIs. You pay a large upfront design + client-complexity cost for a payoff (independent URI evolution) most teams never actually collect.

So the resolution of the myth is not "go to Level 3." It is: **know which level you shipped, name it honestly, and choose the level by the property you need.** Calling a Level-2 API "RESTful" is harmless shorthand right up until someone designs around the assumption that clients follow links — then the gap becomes a real bug. The First Law from ch-01 ([[richards-ford-fundamentals]] via [[richards-ford-hard-parts]]) applies: if you can't name what Level 3 costs you, you haven't understood why you're (correctly) at Level 2.

---

## 4. API-Style Selection as a Trade-off

REST/HTTP is one style among several, and "which style" is a trade-off decision, not a default. [[fielding-rest]] frames the choice as: pick the style by the property you need. Note the honesty hedge from [[COLLECTION-PLAN]]'s gap log — the gRPC/GraphQL trade-offs in [[fielding-rest]] are **synthesis**, not canonical-author quotes; they live as trade-off context inside the Fielding excerpt rather than as standalone sourced excerpts, and are presented here as such.

| Style | Strengths (synthesis, via [[fielding-rest]]) | Weaknesses | Pick it when |
|---|---|---|---|
| REST / HTTP | cacheable, evolvable, ubiquitous; works through any proxy/CDN | weak typing, over/under-fetching, chatty for object graphs | you want cacheable, loosely-coupled, evolvable **resources** exposed broadly |
| gRPC | binary, contract-first (protobuf), streaming, low latency | poor browser/edge ergonomics; needs codegen on both ends | **internal** service-to-service where latency and a strict schema matter |
| GraphQL | client picks exactly the fields; one round-trip for a graph | complexity moves to the server (N+1 queries, per-field caching, per-field auth) | a **client** must aggregate across many resources and you control the schema |

### 4.1 The pattern is priced as a bet, every time

Notice the shape: each style keeps something cheap and makes something else expensive. REST keeps *broad reach and caching* cheap by making *strict typing and graph-fetching* expensive. gRPC keeps *latency and schema rigor* cheap by making *edge/browser reach* expensive. GraphQL keeps *client flexibility* cheap by pushing *operational complexity* onto the server. There is no "best" API style — there is only the style whose cheap-axis matches the change you expect to make most often. This is the First Law applied to integration: "Everything in software architecture is a trade-off… if you think you've found something that isn't, you likely just haven't found the trade-off yet" ([[richards-ford-fundamentals]] via [[richards-ford-hard-parts]]).

---

## 5. Safe Evolution: Additive Change, Tolerant Readers, and Idempotency

Choosing a style is the easy half. The expensive half is *evolving the contract after consumers depend on it* without forcing a coordinated migration. Three disciplines do this work.

### 5.1 Evolve additively — never remove or repurpose a field

[[fielding-rest]] states the rule plainly: the contract is the expensive-to-reverse artifact, so prefer **backward/forward-compatible additive change** over versioned breakage. Concretely: add new fields, never remove or repurpose existing ones; add new optional parameters, never make an optional one required; add new resources, don't change the meaning of existing ones. An additive change is forward-compatible — old clients ignore the new field and keep working.

The reason additive evolution is even possible is the **tolerant reader** discipline, which the consumer must hold up its end of. From [[consumer-driven-contracts]], quoting Robinson verbatim:

> "An implementation must be conservative in its sending behaviour and liberal in its receiving behaviour… message receivers should implement 'just enough' validation: that is, they should only process data that contributes to the business functions they implement." — Robinson, via [[consumer-driven-contracts]]

This is Postel's Law: "conservative in what you send, liberal in what you accept." The consequence, stated in [[consumer-driven-contracts]]: a consumer that ignores unknown fields lets the provider *add* fields freely (forward compatibility). The inverse is the trap — **strict schema validation on the consumer side is what turns harmless additive changes into breaking changes.** If your consumer rejects any payload with an unexpected field, you have unilaterally made the provider's contract brittle without the provider's consent.

### 5.2 Make writes idempotent — retries must be safe

The second discipline comes from the data side, where contract concerns and delivery concerns are the same conversation. [[transactional-outbox]] (Richardson) establishes that asynchronous, reliable messaging is **at-least-once** delivery: the relay may publish the same event more than once, so consumers must dedupe. The API-contract counterpart, from [[fielding-rest]], is to make **writes idempotent** using idempotency keys, so a client that retries (because it timed out, not because the write failed) does not double-apply the effect.

[[transactional-outbox]] makes the linkage explicit — idempotency is not two separate problems on the API side and the data side, it is one:

> "This is why idempotency, an API-contract concern… and the outbox, a data concern, are the same conversation." — from [[transactional-outbox]]

The mechanism: the client attaches a unique `Idempotency-Key` to each write; the server records the key with the result of the first execution and, on any retry carrying the same key, returns the recorded result instead of re-executing. Under at-least-once delivery (which you cannot avoid in any distributed system, per ch-06's preview), idempotency is what makes "retry on uncertainty" a safe default rather than a corruption risk. This is the in-contract seed of the outbox + saga mechanics that ch-06 builds on [[transactional-outbox]] and [[richardson-saga]].

Why at-least-once is unavoidable — and therefore why idempotency is non-optional — comes from the dual-write problem [[transactional-outbox]] names. Richardson states the constraint verbatim:

> "It is not viable to use a traditional distributed transaction (2PC) that spans the database and the message broker." — Richardson, via [[transactional-outbox]]

Because you cannot atomically commit a database write *and* publish its event, the only reliable design persists the event in the same local transaction and relays it afterward — and any relay that guarantees delivery will, on a crash-and-retry, sometimes deliver twice. The guarantee Richardson offers is "messages are guaranteed to be sent if and only if the database transaction commits" (verbatim, [[transactional-outbox]]) — *sent*, not *sent exactly once*. Exactly-once is a fiction at the transport layer; idempotent consumers are how you manufacture exactly-once *effects* on top of at-least-once *delivery*. The contract discipline (idempotency keys) and the data discipline (the outbox) are therefore two ends of one wire, which is why this chapter seeds both and ch-06 completes them.

### 5.3 A taxonomy of changes, by reversal cost

The additive/tolerant-reader disciplines partition every possible contract change into three buckets, and the bucket determines the cost. This is the operational form of "the contract is the expensive-to-reverse artifact" from §1:

| Change | Compatible? | Cost | Examples |
|---|---|---|---|
| **Additive** (new optional field, new resource, new optional param) | forward-compatible for tolerant readers | cheap — deploy unilaterally | add `parent_account_id` to the opportunity payload |
| **Tightening** (new required field, narrowed enum, removed field) | breaks consumers that depend on the old shape | expensive — needs migration | make `region` required; drop `legacy_stage` |
| **Semantic** (same shape, changed meaning) | the worst kind — compiles, breaks silently | most expensive — undetectable by schema checks | `amount` switches from cents to dollars |

The semantic-change row is the dangerous one and the reason CDC (§6) exists: a tolerant reader and a schema check both *pass* a semantic change, because the shape is unchanged. Only a consumer-authored example that asserts on the *value's meaning* catches it before deploy. Keep the meaning of a field as immutable as its name.

### 5.4 Versioning is the fallback, not the plan

Versioning (`/v2/`, `Accept: application/vnd.app.v2+json`, etc.) is what you fall back to when a change genuinely lands in the tightening or semantic buckets and cannot be made additively. It is not free: a new version means running two contracts in parallel, migrating consumers, and eventually deprecating the old — coordinated multi-team work, the exact cost the additive discipline exists to avoid. Treat a forced version bump as the signal that you got a boundary or a contract wrong, and budget for the migration as you would any expensive-to-reverse reversal. The discipline's whole goal is to keep the additive bucket as large as possible so the version-bump bucket stays empty.

---

## 6. Consumer-Driven Contracts: Knowing What "Breaking" Means Before You Deploy

Additive discipline tells you *how* to change safely. Consumer-driven contracts (CDC) tell you *whether a given change is actually safe* — automatically, in CI, before you ship. This is the mechanism that operationalizes everything above.

### 6.1 The inversion

The core move, from [[consumer-driven-contracts]], inverts the usual direction of obligation. Instead of consumers adapting to whatever the provider ships, the provider adopts the reasonable expectations consumers express:

> "When a provider accepts and adopts the reasonable expectations expressed by a consumer, it enters into a consumer contract." — Robinson, via [[consumer-driven-contracts]]

Why this is powerful, verbatim from [[consumer-driven-contracts]]:

> "Consumer contracts allow us to reflect on the business value being exploited at any point in a provider's lifetime… [they] define which parts of that provider contract currently support the business value realized by the system." — Robinson

The payoff that makes CDC the keystone of this chapter, from [[consumer-driven-contracts]]:

> "Consumer-driven provider contracts give us the fine-grained insight and rapid feedback we require to plan changes and assess their impact on applications currently in production." — Robinson

### 6.2 How it works in practice (synthesis, marked as such)

The tooling specifics (Pact, Spring Cloud Contract) are **synthesis** per [[consumer-driven-contracts]], not Robinson quotes:

1. Each consumer publishes a **pact** — a concrete set of example request/response shapes it actually depends on.
2. The provider runs every consumer's pact as a test **in its own CI pipeline** and fails the build if a change would break a real consumer.
3. This replaces brittle, slow end-to-end integration tests with fast, isolated, per-pair checks.

The definition of "breaking change" that falls out of this is the whole point, from [[consumer-driven-contracts]]: **only changes that break a real consumer contract are breaking changes.** Removing a field nobody reads is *not* a breaking change; adding a field is *never* a breaking change for a tolerant reader. CDC turns "is this safe?" from a judgment call into an automated, evidence-backed answer.

### 6.3 Why CDC is the only mechanism that preserves independent deployability

This connects straight back to ch-04. [[newman-building-microservices]] (Newman, book thesis — extracted and corroborated by O'Reilly excerpts/talks per [[COLLECTION-PLAN]], so presented as attributed paraphrase, not as a verbatim quote) makes independent deployability the single defining property of a microservice — the ability to ship one service without coordinating the release of any other. [[consumer-driven-contracts]] names CDC as the mechanism that makes that property survive contact with reality:

> "It's the only mechanism that lets services keep their prized **independent deployability** (a provider knows, before deploy, whether it breaks anyone)." — from [[consumer-driven-contracts]]

Without CDC, "independent deployability" is a hope: you deploy and find out from a pager whether you broke someone. With CDC, it is a verified property: the provider's build is red *before* the bad change ships. This is the contract-layer analogue of ch-09's fitness functions — an automated check that fails the build when a protected property (here, "I don't break any consumer") erodes.

### 6.4 Pricing the CDC bet

CDC is not free, and consistent with the chapter's spine we price it:

- **What it keeps cheap:** changing a provider's internals and contract *additively* with confidence; you get fast, local, deterministic feedback instead of flaky end-to-end suites; independent deployability stays real.
- **What it makes expensive:** every consumer must author and maintain a pact, and the provider's CI must run all of them; you take on a per-pair maintenance burden and a coordination ritual for publishing pacts. For a system with one consumer and one provider on the same team, that ceremony may outweigh the benefit — the bet pays off as the number of independently-owned consumers grows.

---

## 7. Applied to the Sales Agent: The Tool/Integration Layer Is a Contract Surface

The learner's production sales agent (Lina TMR) is an LLM agent acting over many external SaaS tool APIs — Salesforce, Gmail, calendars, spreadsheets, ticketing, CRM-sync. This chapter's discipline maps onto it on two fronts, and they pull in opposite directions, which is the interesting part.

### 7.1 The agent as a *consumer* of contracts it does not own

Most of the agent's integration surface is *inbound* dependence on other people's contracts. The agent does not own the Salesforce or Gmail API; it consumes them. The disciplines invert accordingly:

- **Be a tolerant reader, aggressively.** Per [[consumer-driven-contracts]], the agent should process "just enough" of each SaaS response — only the fields that drive a business function — and ignore everything else. A vendor adding a field to its response must never break the agent. Strict validation against a frozen vendor schema is exactly the trap from §5.1, and for an agent over dozens of third-party APIs that change without warning, it is a guaranteed outage generator.
- **Make every tool call idempotent where the vendor allows it.** Per [[transactional-outbox]] and [[fielding-rest]], the agent must assume at-least-once execution of its own retries. If "send the win-notice email" or "mark opportunity Closed Won" can fire twice (timeout then retry), the agent needs an idempotency key or a dedupe guard so a flaky network does not double-email a customer or double-book a meeting. This is the contract-layer seed of treating every external response as **outside data** in ch-06 ([[helland-data-outside-inside]]): the agent's retries cross a boundary it does not control, so it must price the crossing.

### 7.2 The agent as a *provider* of its own internal tool contract

The other front is the one the agent fully owns: the boundary between the agent's reasoning core (ch-03's technology-free core via [[martin-clean-arch]]) and its tool/integration adapters. That internal contract — "what shape does a tool invocation and result take" — is the agent's own most-expensive-to-reverse artifact, exactly per §1. If every prompt template, every planner, and every adapter hard-codes the tool-result schema, then changing how a tool returns data becomes a coordinated migration across the whole agent.

So the agent should treat its own tool-call interface as a published contract and apply CDC-style discipline internally: the reasoning core publishes the result shapes it depends on, the adapter layer satisfies them, and the boundary is contract-tested so an adapter change cannot silently break planning. This is also where API-style selection (§4) lands concretely — the agent's *internal* core-to-adapter calls are the gRPC-shaped case (low-latency, in-process or service-to-service, strict schema), while the *external* SaaS calls are whatever style each vendor ships (mostly Level-2 HTTP). The architect's job is to keep the vendor's style from leaking past the adapter into the core — the Dependency Rule from ch-03 enforced at the contract layer.

### 7.3 The honest label, applied

When the team says "we expose a REST API for the agent's tools," §3 demands the honest correction: it is almost certainly Level-2 HTTP-RPC, and that is fine. The bet to name explicitly is whether any part of the agent's design assumes hypermedia-style discoverability (it should not, unless someone deliberately paid for Level 3). The AutomationBench experience from the learner's prior course is a useful mirror here: in that benchmark the agent discovers tools via a `search_tools`/`execute_tool` indirection over a ~400-tool catalog — a deliberately *non*-hypermedia discovery mechanism bolted on precisely because the underlying SaaS APIs do not ship HATEOAS controls. That bolt-on is the cost of living at Level 2; naming it is the First Law in action.

---

## Where This Goes

This chapter priced the boundary's surface — the contract — and the disciplines that keep it cheap to evolve: pick the style by the property you need, change additively, be a tolerant reader, make writes idempotent, and verify "breaking" with consumer-driven contract tests. Two threads were deliberately left dangling: idempotency under at-least-once delivery, and the idea that every external response is **outside data** the moment it crosses a boundary.

Ch-06 picks up exactly there. It opens with Helland's root distinction ([[helland-data-outside-inside]]) — inside data (private, mutable, ACID, "now") versus outside data (immutable, versioned, possibly stale) — and shows why the instant data crosses a boundary you lose locks and shared transactions. That single fact generates the consistency patterns: the **saga** ([[richardson-saga]]) replaces the distributed transaction you can no longer have, and the **transactional outbox** ([[transactional-outbox]]) — the partner of this chapter's idempotency discipline — is how you emit events reliably without dual-writing. The contract you just learned to evolve safely is the wire over which all that outside data flows.
