<!-- chapter: ch-07
     track: consistency
     kind: content
     title: CQRS and Event Sourcing: Optional Power Tools
     deps: [[ch-06]]
     sources: [[young-cqrs-es]], [[ddd-aggregates-tactical]], [[nygard-release-it]], [[COLLECTION-PLAN]], [[insights]]
-->

# Chapter 07 — CQRS and Event Sourcing: Optional Power Tools

> **Core insight.** CQRS (use one model to write and a different model to read) and Event Sourcing (store the log of events as the source of truth and replay it to reconstruct state) are two *independent* decisions that happen to compose well — not one pattern, not a package deal, and not a prerequisite for microservices. Each is a power tool that buys a specific, narrow capability, and each charges a permanent toll for it: CQRS makes the read side eventually consistent forever; Event Sourcing makes external side-effects and historical-schema change expensive forever. They sit at the *top* of the consistency arc from [[ch-06]] precisely because they are the patterns you reach for last — after the mandatory inside/outside-data, saga, and outbox mechanics — and only when a specific force demands them. The default answer for almost every part of almost every system is the one Fowler states outright: plain CRUD on a well-drawn aggregate ([[young-cqrs-es]], [[ddd-aggregates-tactical]]).

> **Guideline.** Treat both as bets you *refuse by default* and pay for only under a named force. Reach for **CQRS** only on the *portion* of a system where the read model and write model genuinely diverge — a complex domain whose query shapes don't match its command shapes, or asymmetric read/write scaling where the two paths must scale independently. Reach for **Event Sourcing** only when you have a real need for audit, temporal queries, or replay-debugging that a plain audit table can't satisfy. Scope each to the smallest part of the system that needs it, leave everything else CRUD, and write the ADR (from [[ch-01]]) that records *which force* justified paying the complexity toll. If you can't name the force, you've found a pattern whose cost you haven't priced yet — which the First Law says means you haven't understood it.

---

## 1. Where this chapter sits: the top of the consistency arc

[[ch-06]] established the mandatory consistency mechanics for crossing a boundary: Helland's inside-vs-outside data distinction, the saga as a sequence of local transactions glued by events with compensations, and the transactional outbox for reliable event publication under at-least-once delivery. Those are not optional. The moment your sales agent's state lives in more than one transactional unit, you *need* them or you ship corruption.

This chapter is different. CQRS and Event Sourcing are the **optional power tools** layered on top of that core. The outline merges hexagonal/clean/aggregates into one chapter and saga/outbox/inside-outside into another, but deliberately gives CQRS/ES its *own* chapter — for one reason captured in [[young-cqrs-es]]: their load-bearing teaching is that they are scoped, refusable additions, and keeping them separate from the mandatory mechanics prevents the exact conflation the reconciliation table warns about. If you read this chapter and conclude "I should event-source the agent," you have read it wrong. The intended takeaway is the *discipline of refusal*: knowing precisely what each tool buys, what it costs, and the narrow conditions under which the trade flips positive.

The in-process seed for both tools is the aggregate from [[ch-03]] and [[ddd-aggregates-tactical]]. Vernon's rule 4 — one transaction per aggregate, coordinate across aggregates with **domain events** — is already a tiny event-emitting machine. CQRS and Event Sourcing are what you get when you take that seed seriously enough to make events first-class. That lineage matters: it means you don't adopt these patterns by bolting on a framework, you adopt them by letting the aggregate's existing domain events become the spine of the read side (CQRS) or the storage itself (ES).

---

## 2. CQRS: two models, one bounded context

### 2.1 The definition, stated minimally

Fowler attributes the pattern to Greg Young and states it in one sentence:

> "CQRS stands for Command Query Responsibility Segregation. At its heart is the notion that you can use a different model to update information than the model you use to read information." — Fowler ([[young-cqrs-es]])

> "It's a pattern that I first heard described by Greg Young." — Fowler ([[young-cqrs-es]])

That is the whole idea. The ancestry is CQS — Command Query Separation, Bertrand Meyer's principle that a method either changes state (a command, returns nothing) or returns data (a query, changes nothing) but never both. CQRS lifts that principle from the method level to the *model* level: the object/schema you mutate through commands and the object/schema you project for queries are allowed to be different artifacts, with different shapes, possibly different storage, kept in sync asynchronously.

A useful mental model: a normalized write side that enforces invariants (the aggregate, transactionally consistent, the source of truth for changes) and one or more denormalized read sides — projections — shaped exactly for the queries the UI or API needs, with no joins at read time. The write side emits events (or the application explicitly updates projections); the read side consumes them and materializes a view.

### 2.2 The caution is the load-bearing part

The definition is trivial. The teaching is the warning, and the excerpt marks it as the load-bearing claim:

> "For some situations, this separation can be valuable, but beware that for most systems CQRS adds risky complexity." — Fowler ([[young-cqrs-es]])

> "CQRS should only be used on specific portions of a system… and not the system as a whole." — Fowler ([[young-cqrs-es]])

> "Many systems do fit a CRUD mental model, and so should be done in that style." — Fowler ([[young-cqrs-es]])

Read those three quotes as the chapter's center of gravity. The pattern is not wrong; the *scope* is where teams go wrong. CQRS applied to a whole system is an architecture-level mistake; CQRS applied to the one bounded context where read and write genuinely diverge is a targeted, correct decision. This is the tactical analogue of the strategic discipline from [[ch-02]]: you don't apply one model everywhere, you apply the right model in the right boundary.

### 2.3 Pricing the bet

Here is the trade priced explicitly — what CQRS keeps cheap, and what it makes permanently expensive.

| | What it buys (kept cheap to change) | What it costs (made expensive / permanent) |
|---|---|---|
| **Read/write divergence** | Read and write models evolve independently; a new query shape is a new projection, not a schema migration of the write model | You now maintain two models and the mapping between them; every write-model change must consider its projections |
| **Scaling** | Read path and write path scale independently (read replicas, caches, search indexes) without touching the write model | Operational surface: two stores to provision, monitor, and back up consistently |
| **Query performance** | Denormalized read views answer complex queries with no runtime joins | The read view is **always behind** the write view by the projection lag |
| **Consistency** | — | The read side is **eventually consistent by construction** — there is no version of CQRS that gives you a strongly-consistent read of your own just-completed write without extra work |

That last row is the one that bites. The instant you separate the models you have re-derived, *inside a single service*, the same staleness Helland says you must accept the moment data crosses a boundary ([[ch-06]], [[helland-data-outside-inside]]). CQRS internalizes outside-data semantics into your own system voluntarily. The classic symptom: a user submits a form, the command succeeds, the UI re-queries the read model, and the new record isn't there yet because the projection hasn't caught up. You now own a UX problem (read-your-writes) that pure CRUD never had. The countermeasures — return the projected value optimistically from the command response, version-stamp reads, or poll the projection — are real engineering you pay for in exchange for the divergence and scaling you bought.

So: CQRS is a bet that *the cost of two models plus permanent read-side staleness is less than the cost of forcing one model to serve both divergent reads and writes*. That bet pays off in a complex domain or under asymmetric scaling. It loses everywhere else, which is most places — hence "most systems should stay CRUD."

### 2.4 Read-your-writes, walked through concretely

The eventual-consistency toll is abstract until you trace one request through it, so here is the exact sequence that surprises teams the first time. Assume the agent has a `Deal` write model (normalized tables, enforces invariants) and a `DealSummary` read projection (denormalized, one row per deal with pre-computed fields a dashboard needs).

1. A command arrives: `CloseDeal(deal_id=42)`. The handler loads the `Deal` aggregate, applies the invariant check, writes `stage = Closed Won` to the write store, and commits. The command returns success at time `t₀`.
2. The write commit emits a `DealClosed` event (via the outbox from [[ch-06]], so the emit is reliable).
3. The projection updater consumes `DealClosed` and updates the `DealSummary` row — but it does this *asynchronously*, completing at `t₁ > t₀`.
4. Between `t₀` and `t₁`, any read of `DealSummary` for deal 42 returns the **old** stage. The write succeeded; the read disagrees.

In plain CRUD this window does not exist — you wrote and read the same row, so a read after a committed write always reflects it. CQRS *creates* the window deliberately as the price of the separate read model. The three standard countermeasures, each itself a cost:

- **Optimistic return.** The command handler returns the new projected value directly in its response, so the immediate UI update doesn't re-query the lagging read model. Cost: the command path now knows about read-model shapes, leaking the separation you bought.
- **Version-stamped reads.** The command returns a version token; the client passes it to subsequent reads, which block or retry until the projection has caught up to that version. Cost: read latency and client-side plumbing.
- **Poll/subscribe.** The client polls the projection (or subscribes to a change feed) until the expected value appears. Cost: extra round-trips and a "loading" state in the UX.

The point is not that these are hard — they're routine — but that *they are work you would not be doing under CRUD*. That delta is the honest price of CQRS, and it must appear in the ADR's Consequences section alongside the divergence/scaling benefits. A team that adopts CQRS without naming this window in the ADR has under-priced the bet.

---

## 3. Event Sourcing: the log is the source of truth

### 3.1 The definition

Event Sourcing is the orthogonal idea. Instead of storing current state and overwriting it on each change, you store the *sequence of changes* and treat that log as the system of record.

> "Capture all changes to an application state as a sequence of events." — Fowler ([[young-cqrs-es]])

Current state is not stored authoritatively at all; it is a *fold over the log*:

> "We can discard the application state completely and rebuild it by re-running the events from the event log on an empty application." — Fowler ([[young-cqrs-es]])

State becomes a left-fold: `state = events.reduce(apply, emptyState)`. The events are the truth; any "current state" table is just a cache (a snapshot or projection) you can throw away and recompute. This is a profound inversion of the default database model, where the current row is the truth and the change history — if it exists at all — is a secondary audit log.

Make the fold concrete. A CRUD database stores deal 42 as one mutable row: `{id: 42, stage: "Closed Won", amount: 50000}`. Each update overwrites in place, and the prior values are gone. An event-sourced store keeps the *changes* instead:

```
events for deal 42 (append-only, never updated):
  DealCreated      {id: 42, stage: "Prospecting", amount: 30000}
  DealAmountRaised {id: 42, amount: 50000}
  DealStageChanged {id: 42, stage: "Negotiation"}
  DealStageChanged {id: 42, stage: "Closed Won"}
```

Current state is the fold of `apply` over that sequence: start from empty, apply `DealCreated` to get `{stage: Prospecting, amount: 30000}`, apply `DealAmountRaised` to get `amount: 50000`, apply the two stage changes, and you land on `{stage: Closed Won, amount: 50000}` — the same value the CRUD row held, but *derived* rather than *stored*. The difference is everything that follows: the CRUD row threw away the intermediate facts; the log kept them. "What was the amount when the deal was still in Negotiation?" is unanswerable from the CRUD row and a fold-up-to-the-third-event in the log. That single property — history is intrinsic, not bolted on — is the entire value proposition, and §3.3 is the entire cost of it.

### 3.2 What it buys

The benefits follow directly from "the log is the truth":

- **A free audit log.** Fowler: it is "easy to serialize the events to make an Audit Log" ([[young-cqrs-es]]). You don't *add* auditing; auditing is the storage model. Every change, with full before/after context, is intrinsically recorded because the change *is* the record.
- **Temporal queries.** "Determine the application state at any point in time" ([[young-cqrs-es]]). Replay the log up to timestamp T and you have exactly the state as of T. "What did this deal's pipeline stage look like last Tuesday?" is a fold-up-to-T, not a forensic reconstruction from logs you hope you kept.
- **Replay debugging.** A production bug is reproducible by replaying the exact event sequence that produced the broken state against a fixed version of the code. The bug is no longer "we think this is what happened"; it is "here is the deterministic input that produces it."

### 3.3 What it costs — the sharp edges

Fowler's caution here is even blunter than the CQRS one, and the excerpt flags it as load-bearing:

> "Clearly this stuff can get very messy, don't go down this path unless you really need to." — Fowler ([[young-cqrs-es]])

Two sharp edges make it messy, and both are *permanent* costs baked into the storage model — exactly the kind of expensive-to-reverse decision the course spine ([[insights]]) tells you to identify before committing:

1. **External side-effects on replay.** Replaying the log re-executes the application's `apply` logic. If an event handler sent an email, charged a card, or — for the sales agent — called an external SaaS API, naive replay does it *again*. You must gateway every external effect so it is suppressed during replay and only fired on first, live processing. This is not a one-time fix; it is a constraint every future event handler must respect. Forget it once and a replay double-charges a customer.

2. **Schema evolution of historical events.** Today's `DealClosed` event has fields A, B, C. Next quarter the business adds field D and renames B. But the log contains *years* of old `DealClosed` events with the old shape, and they are immutable truth — you cannot migrate them the way you `ALTER TABLE` a current-state row. Every replay must be able to interpret every historical version. You end up versioning events and writing up-casters (functions that read an old event shape and produce the current one). Fowler notes the broader friction: "Packaging up every change… as an event is an interface style that not everyone is comfortable with" ([[young-cqrs-es]]).

### 3.4 Pricing the bet

| | What it buys (kept cheap) | What it costs (made expensive / permanent) |
|---|---|---|
| **Audit** | Complete, intrinsic audit trail — auditing *is* the storage | Every event is permanent, immutable history you can never `UPDATE` away |
| **Time travel** | State at any past instant via replay-up-to-T | Replay must remain correct across all historical event versions → event versioning + up-casters forever |
| **Debugging** | Deterministic reproduction of any past state | External side-effects must be gateway-able and replay-safe forever |
| **Storage** | Append-only writes (no in-place updates, contention-friendly) | Logs grow unboundedly → snapshots needed; reads of current state cost a fold (mitigated by snapshots/projections, which add machinery) |

The bet: *the cost of replay-safety plus historical-schema discipline is less than the value of true audit/temporal/replay*. In a regulated domain (finance, healthcare) or a domain whose entire value is its history, the bet pays. For a CRUD record where "current value plus a `last_modified` column plus an audit table" suffices, it does not — and "an audit table" is the cheaper answer Fowler is implicitly pointing at when he says don't go down this path unless you really need to.

---

## 4. The myth this chapter exists to kill

### 4.1 The reconciliation row

From the doc-vs-reality reconciliation table in [[COLLECTION-PLAN]], the myth assigned to this chapter:

| Popular narrative | What the primary source actually says |
|---|---|
| "CQRS and Event Sourcing go together / are the same thing." | They are **independent** decisions that *compose*. Fowler: CQRS "adds risky complexity… use on specific portions… not the whole system." Neither requires the other. |

The conflation is everywhere in blog posts and conference talks: "CQRS/ES" written as a single hyphenated thing, as if adopting one obliges the other. It does not. The excerpt is explicit:

> "CQRS and Event Sourcing are **separate** decisions that *compose*… But **CQRS does not require Event Sourcing, and Event Sourcing does not require CQRS.** Treating them as a package deal is a common, costly conflation. Neither requires microservices either; both work inside a modular monolith." — [[young-cqrs-es]]

### 4.2 The four quadrants — proof they're independent

The cleanest way to see the independence is the 2×2. Each axis is its own decision; all four cells are real, sensible architectures.

|  | **State stored as current state (no ES)** | **State stored as event log (ES)** |
|---|---|---|
| **One model (no CQRS)** | Plain CRUD — the default for most of every system | Event-sourced single model: append events, fold to current state, no separate read model |
| **Separate read/write models (CQRS)** | CQRS over a state store: write to normalized tables, project to denormalized read tables | "CQRS/ES" — the famous combo: event store is the write side, projections build read models |

Why the bottom-right cell is *famous* but not *mandatory*: an event store is a naturally good write side (append-only, the events already exist), and projections that fold events into read views are a naturally good read side — so the two compose with almost no friction. The excerpt captures the gravitational pull:

> "CQRS naturally aligns with event-based architectures." — [[young-cqrs-es]]

But "aligns naturally" is not "requires." You can do CQRS with two relational schemas and zero events (top-right-of-CRUD cell). You can event-source a single model with no separate read side (bottom-left). The famous combo is one of four valid points in the design space, and you should choose your cell deliberately rather than inherit the bundled one because a talk said "CQRS/ES."

### 4.3 And neither requires microservices

The second half of the myth — that these are distributed-systems patterns that imply a service split — is also false, and it connects straight back to the course's topology spine ([[ch-04]]). Both patterns live happily inside a single process. CQRS is two models in one modular monolith; the projection updater can be an in-process event handler. Event Sourcing is an append-only table in your one database. Adopting either is *not* a reason to split a service, and splitting a service is *not* a reason to adopt either. Keep these decisions on independent axes from your deployment topology, exactly as [[ch-02]] kept bounded contexts independent from deployment.

---

## 5. The decision discipline: refuse by default, name the force

### 5.1 The default is CRUD, stated by the source

This is the only chapter in the consistency arc whose central message is *don't*. The reasoning chains cleanly from the First Law installed in [[ch-01]] ([[richards-ford-fundamentals]]) and re-stated in [[insights]]: every pattern is a bet about which changes you keep cheap; if you can't name the cost you haven't understood it. For CQRS and ES the costs are large, permanent, and specific (§2.3, §3.4), so the prior should be strongly against adoption. Fowler gives you the default outright:

> "Many systems do fit a CRUD mental model, and so should be done in that style." — Fowler ([[young-cqrs-es]])

CRUD on a well-drawn aggregate ([[ddd-aggregates-tactical]]) is not a fallback for the unsophisticated; it is the *correct* answer whenever the divergence/audit forces are absent, because it keeps everything cheap: one model to change, strongly-consistent reads of your own writes, no replay hazard, no event-schema museum to maintain.

### 5.2 The forces that flip each bet

Adopt only when a *named* force is present. Record the force in the ADR.

| Tool | Adopt only when this force is present | Stay CRUD when |
|---|---|---|
| **CQRS** | Read and write models genuinely diverge (complex domain whose query shapes ≠ command shapes) **or** asymmetric read/write scaling demands independent paths | The same model serves reads and writes acceptably; query and command shapes are close |
| **Event Sourcing** | A real need for audit / temporal queries / replay-debugging that a plain audit table cannot satisfy | "Current value + `last_modified` + an audit table" covers the requirement |

The excerpt's own list of legitimate triggers: "a complex domain better served by separate read/write models, or high-performance needs that demand independently scaled read and write paths" ([[young-cqrs-es]]). Note both are *forces in the requirements*, not preferences — which is the architecture-characteristics discipline from [[ch-01]] applied here: derive the critical few from requirements, don't maximize capability for its own sake.

### 5.3 Scope it to a portion, not the system

When you do adopt, the scope rule from §2.2 is non-negotiable: "use on specific portions of a system… not the system as a whole." Concretely, that means one bounded context gets CQRS while its neighbors stay CRUD; one aggregate type gets event-sourced while the rest of the model stores current state. The blast radius of the complexity is bounded to where the force lives. A team that event-sources its entire system has converted every trivial CRUD entity into a replay-hazard, schema-museum liability for the benefit of the handful of entities that actually needed audit — a net-negative trade across the system even where each local decision looked locally tempting.

### 5.4 Recording the refusal as an ADR

The decision to *not* adopt is as architecturally significant as the decision to adopt, and it needs the same durable record — the ADR from [[ch-01]] ([[nygard-release-it]]), which Nygard frames as "a conversation with a future developer." A future engineer staring at a plain-CRUD `Deal` table will eventually ask "why didn't we event-source this, given we obviously care about deal history?" — and without an ADR they may well "fix" it by adding ES and inheriting the replay hazard you deliberately avoided. The refusal ADR forecloses that. Its shape, using Nygard's Context / Decision / Consequences skeleton:

```
ADR-014: Deal aggregate uses CRUD + audit table, not Event Sourcing
Status:    Accepted
Context:   We need an audit trail of deal stage/amount changes for sales
           reporting. Event Sourcing would give this intrinsically. However,
           the Deal aggregate's handlers fire external SaaS side-effects
           (Salesforce writes, customer emails); replay-safety for those is
           a permanent, error-prone burden (see §3.3). Audit is the only ES
           force present — no temporal-query or replay-debugging requirement.
Decision:  Store current state (CRUD) plus an append-only deal_audit table
           written in the same transaction as each Deal change.
Conseq.:   (+) No replay hazard against live customer systems; no event-schema
               museum; strongly-consistent reads of our own writes.
           (-) Audit is a parallel artifact we must remember to write on every
               Deal mutation (mitigated: enforced in the aggregate, fitness-
               function-checked later — see ch-09). No free temporal queries;
               if that force later appears, revisit this ADR.
```

This is the trade-off spine made operational: the bet is named, the force that *would* flip it is recorded, and the future revisit condition is explicit. The ADR is how an expensive-to-reverse decision (storage model) stays revisable — which is exactly the evolution discipline ch-09 will formalize with fitness functions.

---

## 6. Applied to the sales agent (Lina TMR)

The learner's production sales agent — an LLM acting over many external SaaS tool APIs — is exactly the kind of system where the *refusal* discipline matters more than the patterns. Walk the two decisions through it.

### 6.1 Where CQRS might earn its place

The agent has bounded contexts (from the [[ch-02]] analysis): lead/pipeline, conversation, scheduling, CRM-sync. Most of these are CRUD: a lead is created, updated, read back; a meeting is scheduled. CRUD on the aggregate is correct — divergence is absent and you want read-your-writes so the agent never reasons over a stale view of a record it just changed.

The one plausible CQRS candidate is **conversation/analytics**. The write model is "append a turn to a conversation"; the read shapes the agent and dashboards want are wildly different — "all open deals with sentiment trending down," "rep-level pipeline velocity," cross-conversation aggregates. Those query shapes diverge sharply from the append-a-turn write shape, and read volume (dashboards, the agent's own retrieval) dwarfs write volume — asymmetric scaling. *That* is a named force. If it appears, scope CQRS to that one context: a normalized conversation write model, denormalized projections for the analytical reads, and explicit handling of the projection lag so the agent doesn't act on a view that's seconds behind the conversation. Everything else stays CRUD.

### 6.2 Why Event Sourcing is mostly the wrong bet here — and where its sharp edge is fatal

Event Sourcing on the agent's *own* state is tempting ("audit every decision the agent made!") and mostly wrong: a `last_modified` plus a decision-log table gives you the audit without the replay hazard or the event-schema museum. The audit force is real but a cheaper tool satisfies it.

And the agent makes §3.3's first sharp edge *acute*. The agent's event handlers don't send a harmless internal email — they **call external SaaS APIs**: create a Salesforce opportunity, send a real email, book a real calendar slot. Under Event Sourcing, replay re-runs `apply`. If those external calls aren't rigorously gateway-able and replay-suppressed, replaying the agent's log to reconstruct state or reproduce a bug would *re-fire real-world side-effects against live customer systems* — re-create opportunities, re-send emails. For an agent over external APIs, "external side-effects on replay" is not a footnote; it is potentially the most expensive failure mode in the system. That single force should push the agent's core state firmly toward CRUD-plus-audit-table and away from event-sourcing.

### 6.3 The inside/outside tie-back

This connects to [[ch-06]]: every external SaaS response is **outside data** — immutable, versioned, possibly stale ([[helland-data-outside-inside]]). The agent should *never* treat a SaaS payload as authoritative live state to event-source over. If you ever event-source anything in the agent, source the agent's *own* domain decisions (inside data you own), and treat ingested SaaS snapshots as the versioned, replay-inert outside data they are. Folding outside data into your event log as if it were your own truth would couple your reconstructable state to a vendor's mutable reality — a coupling the inside/outside distinction exists precisely to prevent. Record whichever choice you make as an ADR ([[ch-01]]) naming the force, so a future engineer reading it knows why the agent is CRUD-with-audit and not event-sourced.

---

## 7. The aggregate seed, closed

The chapter opened by noting the in-process seed: Vernon's rule 4 — one transaction per aggregate, domain events across them ([[ddd-aggregates-tactical]]). Close the loop. That single rule is the kernel from which the entire consistency arc grew:

- An aggregate emitting a **domain event** to coordinate with another aggregate (rule 4) is, stretched across services, a **saga** ([[ch-06]], [[richardson-saga]]).
- Persisting that domain event reliably with the state change is the **transactional outbox** ([[ch-06]], [[transactional-outbox]]).
- Letting those same domain events *drive a separate read model* is **CQRS** (§2).
- Letting those same domain events *be the storage itself* is **Event Sourcing** (§3).

Four patterns, one seed. This is why [[ddd-aggregates-tactical]] called the aggregate "a design-altitude idea, not a coding tip": get the aggregate boundary right in the modular monolith and you have the correct consistency boundaries for every pattern above, for free. Get it wrong (too big) and you bake in contention that every downstream pattern inherits. The aggregate is where you pay or save on consistency, once, early.

---

## Where this goes

The consistency arc is complete: [[ch-06]] gave the mandatory mechanics (inside/outside data, sagas, outbox); this chapter capped it with the optional power tools (CQRS, ES) and the discipline of refusing them by default. From here the course turns to **evolution — keeping the bet revisable**. Ch-08 takes the resilience view: every place your architecture crosses a boundary (every saga step, every external SaaS call the agent makes) is an *integration point* where failure enters and spreads, and the chapter installs the stability toolkit — timeout, circuit breaker, bulkhead, fail-fast — as *design-altitude* decisions about where to draw the blast radius, not as ops afterthoughts. The connective tissue: the same event-driven, async integration you reached for in [[ch-06]] to avoid distributed transactions is *itself* a resilience decision, because it removes the synchronous coupling those stability patterns would otherwise have to defend.
