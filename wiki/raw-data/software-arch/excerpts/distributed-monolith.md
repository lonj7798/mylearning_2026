<!-- scope: the distributed-monolith anti-pattern — what it is, how it forms, how to detect it
     deps: fowler-microservices, fowler-monolith-first
     see-also: helland-data-outside-inside, richardson-saga, nygard-release-it
-->

# The Distributed Monolith — Worst of Both Worlds

- **Core Insight:** A distributed monolith is a set of services that *behave* like one program — they must be deployed together and call each other synchronously — so you pay the full distributed-systems tax while keeping every coupling a monolith had.
- **Guideline:** The litmus test is **independent deployability**. If you cannot deploy service A without also deploying B, you do not have microservices; you have a monolith with network latency. Fix coupling before celebrating decomposition.
- **Source:** Synthesized from Newman *Building Microservices* (independent deployability as the defining property), Fowler's trade-off writing, and the community anti-pattern literature (vFunction, Gremlin, dev.to). Community/synthesis — not a single canonical author article.
- **Relevant chapters:** monolith-vs-microservices, service-decomposition, event-driven-architecture.

## Definition

A distributed monolith arises when services are "so tightly coupled and interdependent that they behave like a monolithic application, defeating the core benefit of adopting microservices." It is "a monolith that just happens to communicate over HTTP instead of function calls."

## The four tells

1. **Deployment dependencies** — services must be released together; you've lost independent deployability (the one property that justified the split).
2. **Synchronous coupling** — a request fans out through a chain of real-time blocking calls instead of an async event or a message broker. Any link's latency or outage stalls the whole chain.
3. **Shared database** — two services read/write the same schema, creating an *implicit, unversioned contract*. Changing one service's tables silently breaks the other. (The cure is [[database-per-service]].)
4. **Cascading failures** — tight runtime coupling means one slow dependency drags down the entire workflow (the failure mode [[nygard-release-it]] exists to stop).

## Why it's "worst of both worlds"

> "This network-based modularity gives you all the pain of distributed systems without the independence that makes microservices worthwhile."

Trade-off framing for the learner: a monolith pays *zero* distribution tax and accepts deploy-coupling. Well-cut microservices pay distribution tax and buy deploy-independence. A distributed monolith pays the tax and buys nothing. A modular monolith (see [[fowler-monolith-first]]) is the strictly-dominant fallback when you're unsure.

## How to avoid it

- Cut boundaries on real seams (business capability / subdomain), not on technical layers.
- Default to **asynchronous, event-driven** integration so a downstream outage doesn't synchronously block upstream → [[young-cqrs-es]], [[richardson-saga]].
- Give every service its own data → [[database-per-service]]; integrate via immutable messages → [[helland-data-outside-inside]].
- Apply resilience patterns at every integration point → [[nygard-release-it]].

## Connections

- The "don't start here" prescription → [[fowler-monolith-first]].
- The data-coupling root cause → [[helland-data-outside-inside]], [[database-per-service]].
