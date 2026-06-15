<!-- scope: Richards & Ford Fundamentals — trade-off law, architecture characteristics, decision-making, evolutionary arch
     see-also: richards-ford-hard-parts, nygard-release-it, c4-model
-->

# Fundamentals of Software Architecture — Richards & Ford (+ Evolutionary Architecture)

- **Core Insight:** Architecture is the practice of choosing among competing **architecture characteristics** (the "-ilities") under the **first law**: everything is a trade-off — so the architect's job is not finding the right answer but surfacing and weighing the wrong ones.
- **Guideline:** Identify the few architecture characteristics that actually matter for *this* system from the requirements, choose the style that best supports them, document the trade-off (ADR), and protect the chosen characteristics over time with **fitness functions**.
- **Source:** Mark Richards & Neal Ford, *Fundamentals of Software Architecture* (2020/2e) — book, thesis extracted; the First Law is quoted as commonly published. Evolutionary-architecture material: Ford, Parsons & Kua, *Building Evolutionary Architectures*.
- **Relevant chapters:** architectural-decisions, monolith-vs-microservices, cross-cutting-concerns.

## The First Law (the sentence the whole book hangs on)

> "Everything in software architecture is a trade-off. If you think you've found something that isn't a trade-off, you likely just haven't found the trade-off yet." — Richards & Ford

For this learner — who already thinks in trade-offs — this is the organizing principle: every pattern in this library is presented as a *bet with a cost*, never a best practice.

## Architecture characteristics (the "-ilities")

The non-functional properties that actually drive structure: scalability, performance, availability, security, deployability, testability, modifiability, fault tolerance, etc. The architect's first real task is to *derive the critical few* from business needs (you cannot maximize all of them — that's the first law again) and let *those* pick the architectural style. A characteristic like "deployability" is precisely what pushes you toward microservices; "simplicity" is what pushes you back to a monolith.

## "Architecture is the stuff that's hard to change"

The popular working definition: architecture = "the deep, binding decisions you make about your software" — the ones expensive to reverse. (Fowler's friendly amendment, worth noting: "a good architect makes change easier — thus reducing architecture." The aspiration is to *shrink* the irreversible set.) This is the course's spine, stated by the field's own definition.

## Decision-making & ADRs

Architects *make and document decisions*. The book pairs trade-off analysis with **Architecture Decision Records** (→ [[nygard-release-it]]) as the durable artifact: the decision plus its context and consequences, so the "why" survives.

## Evolutionary architecture & fitness functions

From *Building Evolutionary Architectures* (Ford, Parsons, Kua):

> "An evolutionary architecture supports guided, incremental change across multiple dimensions."

A **fitness function** is "an objective integrity assessment of some architectural characteristic(s)" — an automated check (test, metric, monitor, ArchUnit rule) that fails the build when a protected characteristic erodes. It makes "keep the [[martin-clean-arch|dependency rule]] intact" or "p99 latency < X" *enforceable*, not aspirational. This is how you keep the expensive-to-reverse decisions from rotting silently as the system evolves.

## Connections

- The distributed-systems deep cuts of these trade-offs → [[richards-ford-hard-parts]].
- The notation for the decisions → [[c4-model]]; the record of them → [[nygard-release-it]].
