<!-- scope: Nygard Release It! stability patterns + ADR article (Documenting Architecture Decisions)
     see-also: distributed-monolith, richardson-saga, richards-ford-hard-parts
-->

# Release It! Stability Patterns + ADRs — Michael Nygard

- **Core Insight:** Every integration point is a place a failure can enter and spread; stability is *engineered* by stopping propagation (timeouts, circuit breakers, bulkheads), not hoped for. And the architectural decisions behind these choices must be *recorded*, because "why" is what gets lost.
- **Guideline:** Never make a remote call without a **timeout**; wrap flaky dependencies in a **circuit breaker**; **bulkhead** resources so one failure can't drain the whole pool; **fail fast** to stop cascades. Capture each significant choice in an **ADR**.
- **Source:** Michael Nygard, *Release It!* 2e (stability patterns — book, thesis extracted; csabapalfi/release-it notes + Pragprog corroborate); "Documenting Architecture Decisions" (cognitect.com/blog/2011/11/15/documenting-architecture-decisions).
- **Relevant chapters:** cross-cutting-concerns, architectural-decisions, event-driven-architecture.

## The anti-pattern that motivates everything: Cascading Failure

A failure at one integration point propagates "from subsystem to subsystem crashing each one." The two most effective counters: **Circuit Breakers** and **Timeouts**. "Integration Points without Timeouts is a surefire way to create Cascading Failures."

## Stability patterns (the resilience toolkit)

- **Timeout** — bound every remote wait; an unbounded wait is a held resource, and held resources are how one slow dependency hangs the whole system.
- **Circuit Breaker** — track failures to a dependency; once over threshold, *open* the circuit and fail fast (skip the call) instead of piling up requests on a dead service. Periodically *half-open* to test recovery. (Nygard popularized this pattern.)
- **Bulkhead** — partition resources (thread pools, connection pools) so a failure in one area can't drain the whole. "Just as a ship's hull is divided into watertight compartments so that a breach in one section does not sink the vessel."
- **Steady State** — every accumulation (logs, sessions, caches) must have a matching cleanup; don't require human intervention to keep running.
- **Fail Fast** — detect you can't succeed and return immediately; "the idea is to fail as fast as you can so that only the subsystem where the error occurred is affected."
- Plus: Handshaking, Shed Load, Back Pressure, Governor, Let It Crash.

## Why this is design altitude (not ops)

These are *architectural* decisions: where you place circuit breakers and bulkheads defines your system's blast radius. A [[distributed-monolith]] is precisely a system that skipped them — synchronous chains with no breakers, so one outage cascades everywhere. Choosing async/event integration (→ [[richardson-saga]]) is itself a resilience decision: it removes the synchronous coupling these patterns otherwise have to defend.

## Architecture Decision Records (the second Nygard contribution)

> "An architecture decision record is a short text file in a format similar to an Alexandrian pattern." — Nygard

What to record:

> "We will keep a collection of records for 'architecturally significant' decisions: those that affect the structure, non-functional characteristics, dependencies, interfaces, or construction techniques." — Nygard

The problem it solves:

> "One of the hardest things to track during the life of a project is the motivation behind certain decisions." — Nygard

Structure: **Title, Status** (proposed/accepted/deprecated/superseded), **Context, Decision, Consequences**; "one or two pages"; "write each ADR as if it is a conversation with a future developer." ADRs are the durable record of exactly the expensive-to-reverse decisions the course is about.

## Connections

- The anti-pattern these prevent → [[distributed-monolith]].
- Async coordination as a resilience decision → [[richardson-saga]].
- Trade-off documentation discipline → [[richards-ford-fundamentals]].
