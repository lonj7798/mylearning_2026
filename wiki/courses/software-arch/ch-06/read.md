<!-- chapter: ch-06
     track: consistency
     kind: content
     title: Inside vs Outside Data — Sagas and the Transactional Outbox
     deps: [[ch-03]], [[ch-05]]
     sources: [[helland-data-outside-inside]], [[richardson-saga]], [[transactional-outbox]], [[fowler-microservices]]
-->

# Chapter 06 — Inside vs Outside Data: Sagas and the Transactional Outbox

> **Core insight.** Every consistency pattern in distributed software descends from one physical fact: *the instant data crosses a service boundary you lose the lock and the shared transaction.* Pat Helland names the two regimes this creates — **inside data** (private, mutable, ACID, true "now") and **outside data** (the messages between services: immutable, versioned, identity-stamped, possibly stale). Once you accept that you cannot wrap one transaction around two services, the rest of the chapter is forced: replace the cross-service ACID transaction with a **saga** (a sequence of local transactions undone by *compensating* transactions, never a rollback), and make each step's event durable without a dual-write using the **transactional outbox**. None of these is a "best practice" — each is the priced consequence of refusing to pretend outside data is inside data.

> **Guideline.** Inside a boundary, use SQL and ACID freely; the moment a datum leaves, treat it as an immutable, versioned snapshot that may be stale, and design the application for that staleness rather than wishing it away. When one business operation must change state in several services, do not reach for 2PC — write a saga, price the **loss of isolation** it costs you, and add explicit countermeasures (semantic locks, commutative updates, re-reads) for the anomalies that loss creates. To emit each step's event reliably, persist it to an outbox table *in the same local transaction* as the business change and relay it asynchronously, and make every consumer idempotent because delivery is at-least-once.

---

## 1. The Root Distinction: Inside Data vs Outside Data

Chapters [[ch-02]] through [[ch-05]] were about *where* to put boundaries and *what contract* to expose across them. This chapter is about the bill that arrives the moment any boundary is real: what crossing it costs you in consistency. Pat Helland's "Data on the Outside versus Data on the Inside" (CIDR 2005) is the deepest cut in the whole library because it gives the *theoretical reason* every later pattern exists, not just the mechanics. [[helland-data-outside-inside]]

> **Sourcing note.** Helland's ACM Queue 2020 reprint returned **403** at fetch; the inside/outside/immutability claims below are corroborated via the CIDR-2005 PDF, Semantic Scholar, and "the morning paper" summary, per the gap log in [[COLLECTION-PLAN]]. The short phrases in quotes are taken from those corroborating sources; treat the longer characterizations as faithful paraphrase, not as deep verbatim from the reprint.

### 1.1 Two regimes, one boundary

Helland splits all data a service touches into two kinds:

| | **Inside data** | **Outside data** |
|---|---|---|
| Where it lives | private to one service | the messages *between* services |
| Mutability | mutable | immutable |
| Transactional? | ACID; you can lock it | no locks, no shared transaction |
| Time | "now" — current truth | a snapshot, true *as of when sent* |
| Schema | the service's own DDL | "each data item's schema is **versioned**" |
| Helland's phrase | "the realm of SQL and SQL's DDL" | "immutable… stable, such that a repeated request is unchanged" |

Inside the boundary, you live in the world [[ch-03]] built: an aggregate is the unit of immediate transactional consistency, and Vernon's rule 4 ("use eventual consistency outside the boundary") already told you the local-ACID world stops at the aggregate edge. Helland generalizes that edge to the *service* boundary and explains *why* it must stop there. [[ddd-aggregates-tactical]]

### 1.2 The three load-bearing claims

The whole chapter rests on three claims from [[helland-data-outside-inside]]:

1. **Services do not share transactions.** "You cannot wrap a transaction around two services." This is *the* reason distributed 2PC across services is a dead end — there is no global lock manager you are allowed to hold across an autonomous service's database.
2. **Outside data must be immutable.** "Messages themselves must also be immutable" — their content "should never change across retries." An immutable, identity-stamped message is safe to retry, cache, reorder, and replay; a mutable one is not. This is the precondition that makes the outbox and event-driven flow work at all.
3. **Outside data may be stale, and that's fine.** Because you cannot lock across the boundary, what you receive is a snapshot true as of when it was sent. You design for **eventual consistency at the application level**, reconciling staleness rather than pretending it isn't there.

Read in order, these three claims *generate* every other pattern in this chapter and the next. No-shared-transactions forces the saga (§3). Immutability is what lets the outbox relay a message safely (§5). Possible-staleness is the design constraint that turns eventual consistency from "an infra detail" into an application-design problem (§2).

It is worth being explicit about *why* this is the deepest cut in the library rather than just one more pattern. Aggregate boundaries ([[ddd-aggregates-tactical]]), database-per-service, sagas, the outbox, CQRS read models ([[young-cqrs-es]]), and immutable domain events are *not* independent inventions you choose à la carte — they are all consequences of one fact: the moment data crosses a service boundary, you lose locks and shared transactions, so the data must become immutable, versioned, and accepted-as-possibly-stale. Conversely, a [[distributed-monolith]] is precisely what you build when you *ignore* Helland and keep treating cross-boundary data as inside data — a shared database, synchronous "live" reads across the boundary, distributed locks. That anti-pattern (the subject of [[ch-04]]) is not a separate failure mode from this chapter's concerns; it is what this chapter's discipline exists to prevent.

### 1.3 Pricing the bet — the regime split itself

Even the inside/outside split is a trade, not a free win. Drawing the line **keeps cheap**: the ability to evolve each service's storage independently, to retry and replay messages safely, to scale services without a global lock. It **makes expensive**: you can no longer ask a simple cross-service question and get a transactionally-consistent answer — every cross-boundary read is potentially stale, and every cross-boundary write needs its own consistency machinery. If your system genuinely lives inside one consistency boundary, paying this price buys you nothing; that is exactly why the course defaults to a modular monolith ([[ch-04]], [[fowler-monolith-first]]) and keeps as much as possible *inside*.

---

## 2. Myth Check: Two False Beliefs This Chapter Kills

The reconciliation table in [[COLLECTION-PLAN]] flags two myths for this chapter. Both are seductive precisely because they let you *avoid* designing for the boundary.

### 2.1 "Use distributed transactions (2PC) across services"

**The narrative:** if you need atomicity across two services, run a two-phase commit and you get ACID back.

**What the primary sources say:** you cannot. Helland is categorical that **services do not share transactions** [[helland-data-outside-inside]]. Richardson is equally blunt about the adjacent case of writing to a DB *and* a broker:

> "It is not viable to use a traditional distributed transaction (2PC) that spans the database and the message broker." — Richardson [[transactional-outbox]]

2PC across autonomous services is not "advanced," it is the wrong tool: it requires a coordinator holding locks across systems that, by the database-per-service rule, you don't own. The resolution is to *replace* the cross-service ACID transaction with a **saga** (§3) and to make event emission safe with an **outbox** (§5) — accepting lost isolation rather than chasing a global lock you can't have.

### 2.2 "Eventual consistency is an infra problem"

**The narrative:** eventual consistency is something the message bus / database / platform sorts out; the application can pretend everything is strongly consistent.

**What the primary sources say:** at the design altitude this course cares about, eventual consistency is an **application-design** problem. Fowler is explicit in "Microservice Trade-Offs":

> "Business logic can end up making decisions on inconsistent information." — Lewis & Fowler [[fowler-microservices]]

Helland says the same from the data side: because outside data may be stale, your logic must be written to *expect* staleness [[helland-data-outside-inside]]. You cannot delegate this to infrastructure, because the infrastructure has no idea which stale read is harmless (a cached display name) and which is catastrophic (a balance you're about to debit). Deciding *where* staleness is tolerable and *where* it must be reconciled is an architectural decision — it belongs in your ADRs ([[ch-01]], [[nygard-release-it]]), not in a config file.

---

## 3. The Saga: Distributed Transactions Without 2PC

Once §1 and §2 land, the saga is not a clever option — it is the only shape left.

### 3.1 Why sagas exist

The precondition is data ownership. Database-per-service is the rule:

> "Keep each microservice's persistent data private to that service and accessible only via its API." — Richardson [[transactional-outbox]]

And that rule immediately creates the problem the saga solves:

> "The Database per Service pattern creates the need for this pattern." — Richardson [[richardson-saga]]

No shared database means no distributed ACID transaction. So the cross-service business operation is rebuilt as a chain:

> "A saga is a sequence of local transactions. Each local transaction updates the database and publishes a message or event to trigger the next local transaction in the saga." — Richardson [[richardson-saga]]

### 3.2 The 1987 origin (thesis, not verbatim)

Richardson did not invent the construct; he repurposed it. The idea comes from Hector Garcia-Molina and Kenneth Salem's "Sagas" (SIGMOD 1987).

> **Sourcing note.** The 1987 PDF was **image-only (no text layer)** at fetch, so the following is the paper's *thesis extracted from knowledge of the paper, not quoted verbatim* — honor this hedge, per [[COLLECTION-PLAN]] and [[richardson-saga]].

The original paper introduced sagas for **long-lived transactions (LLTs)** inside a *single* database. An LLT is split into sub-transactions T1…Tn, each paired with a **compensating transaction** C1…Cn that *semantically undoes* it. The guarantee is that the system ends in one of two clean shapes: either the full sequence T1…Tn ran, or a prefix T1…Tj ran and was followed by Cj…C1 — **never a stuck partial state.** Crucially, the 1987 construct already *relaxes isolation*: other transactions can observe intermediate sub-transaction results. Richardson's contribution was recognizing that this single-DB tool is exactly the right tool for cross-*service* consistency.

This lineage matters because it tells you what a saga *is not*. It is not a rollback (there is no global transaction to roll back). A compensation is a new forward transaction that produces a semantically-undoing effect: you don't un-charge a card by erasing the row, you `refund()` it.

### 3.3 Choreography vs orchestration

A saga needs a way to advance from step to step. Richardson gives two:

> "Choreography - each local transaction publishes domain events that trigger local transactions in other services." — Richardson [[richardson-saga]]
> "Orchestration - an orchestrator (object) tells the participants what local transactions to execute." — Richardson [[richardson-saga]]

| | Choreography | Orchestration |
|---|---|---|
| Control | decentralized; services react to events | a central orchestrator drives steps |
| Coupling | lower; fewer dependencies | higher, but the flow logic lives in one place |
| Visibility | hard to trace the whole flow | one place to see/debug the process |
| Best when | simple, naturally event-driven flow | complex, many interdependent steps |
| Risk | distributed logic, monitoring burden | the orchestrator becomes a bottleneck / single point of design gravity |

This is itself a priced bet, not a "which is better" question. **Choreography keeps cheap:** adding a new reaction to an existing event (just subscribe — no central code to touch) and avoiding a coordinator bottleneck. **It makes expensive:** *understanding* the flow — the business process exists only as an emergent property of who-listens-to-what, so debugging "why did this order stall?" means reconstructing the process from logs across services. **Orchestration keeps cheap:** comprehension and change of the *process* itself (it's one readable state machine) and observability (one place reports progress). **It makes expensive:** coupling — every participant now depends on the orchestrator, which accretes business logic and can quietly become the distributed system's new monolith. The Hard Parts rule from [[ch-04]] applies: size and centralization are *outputs* of trade-off analysis, not defaults.

> **Interactive — open [`figures/saga-choreography-vs-orchestration.html`](figures/saga-choreography-vs-orchestration.html).** Toggle between choreography and orchestration, set "Inject failure at Step 2 (Charge)" or "Step 3 (Confirm)," and hit Run. Watch the committed steps light green, the failed step go red, and the **compensations run in reverse** (purple). The amber banner marks the **lost-isolation window** — the interval during which a committed-but-not-yet-final step is already visible to every other transaction. That window is the entire point of §4; the figure lets you *see* it open and close.

### 3.4 The price: no isolation

This is the line the learner must internalize, because it is the cost almost everyone forgets to price:

> "Lack of isolation (the 'I' in ACID)… means there's risk that the concurrent execution of multiple sagas and transactions can [cause] data anomalies." — Richardson [[richardson-saga]]

A classic ACID transaction gives you A, C, I, and D. A saga gives you a form of **atomicity-of-outcome** (all steps, or a clean prefix-plus-compensation) and durability per local step — but it *throws away isolation*. Between the commit of step 1 and the final resolution of the saga, the intermediate state of step 1 is **visible to everyone else**. That is the amber window in the figure. Concrete anomalies:

- **Dirty reads:** another saga reads the `charged: $99` that you are about to `refund()`, and acts on a charge that won't survive.
- **Lost updates / non-repeatable reads:** two sagas reserve the last unit of inventory because neither sees the other's in-flight reservation.

Because a saga is not a free transaction, Richardson notes "a saga developer must typically use **countermeasures**":

| Countermeasure | What it does |
|---|---|
| **Semantic lock** | mark the in-flight row with a `PENDING` flag so other sagas know it is not final |
| **Commutative updates** | design operations whose order doesn't matter (e.g. `debit`/`credit` deltas) so concurrent sagas don't clobber each other |
| **Pessimistic view** | re-order saga steps so the riskiest-to-observe state is created last |
| **Re-read value** | re-read just before acting and abort if it changed since you read it (optimistic check) |
| **By-value tracking** | route saga handling by the *business risk* of the request so high-risk flows get stronger countermeasures |

The honest framing for the ADR: a saga buys you cross-service progress at the cost of isolation, and the countermeasures buy back *some* isolation at the cost of added per-step complexity. If you can't name which anomaly each countermeasure defends, you haven't finished pricing the bet (the First Law, [[ch-01]], [[richards-ford-fundamentals]]).

---

## 4. The Lost-Isolation Window, Concretely

It is worth slowing down on the window because it is where naive saga designs silently corrupt data.

Picture the 3-step order saga in the figure: `reserve(item)` → `charge(card)` → `confirm(order)`. Suppose step 3 fails. The compensations `refund(card)` then `release(item)` run in reverse, and the system returns to a consistent state. So far, so safe. But consider *when* the intermediate state was observable:

```
t0  Inventory: reserved=1   COMMITTED  (locally durable, globally non-final)
t1  Payment:   charged=$99  COMMITTED  (locally durable, globally non-final)
t2  Order:     confirm()    FAILS
t3  Payment:   refund()     compensation
t4  Inventory: release()    compensation
```

Between **t0 and t4**, any other transaction that reads inventory sees the unit as gone, and any fraud/ledger process that reads payment sees a $99 charge. Both observations are *true locally* and *wrong globally* — the saga had not yet decided to commit. A concurrent customer is told the item is out of stock; a downstream analytics job books revenue that gets refunded a second later. This is precisely the isolation a single ACID transaction would have hidden behind its locks.

The design move is not to eliminate the window (you can't — that would require the global lock Helland says you can't hold) but to **bound and label it**. A semantic lock turns the silent dirty read into an explicit "this is PENDING, don't rely on it yet" signal; choosing the step order so the most-sensitive state is written last (`pessimistic view`) shrinks the window for the worst anomaly. The architecture decision is *which* anomalies your domain cannot tolerate, and which countermeasures you'll pay for. Everything else you accept.

---

## 5. The Transactional Outbox: Emitting Events Without a Dual-Write

A saga step is "update the database **and** publish an event." That conjunction hides a trap.

### 5.1 The dual-write problem

> "How to atomically update the database and send messages to a message broker?" — Richardson [[transactional-outbox]]

The naive sequence — commit the DB row, then publish to the broker — can crash *between* the two operations:

- Crash after commit, before publish → the state changed but **no event fired**; the next saga step never runs. The saga stalls in a non-final state forever.
- Publish first, then the DB commit rolls back → an event fired for a write **that never happened**; downstream services act on a phantom.

And you cannot fix this with 2PC, because (§2.1) "it is not viable to use a traditional distributed transaction (2PC) that spans the database and the message broker." [[transactional-outbox]]

### 5.2 The outbox solution

The fix is to make event emission part of the *same single commit* as the business change:

> "The solution is for the service that sends the message to first store the message in the database as part of the transaction that updates the business entities." — Richardson [[transactional-outbox]]

You insert the outgoing event into an **outbox** table in the same local transaction that updates the business rows. One commit, one atomic unit. The guarantee:

> "Messages are guaranteed to be sent if and only if the database transaction commits." — Richardson [[transactional-outbox]]

Concretely, the saga step becomes one local transaction that touches two tables:

```
BEGIN;
  UPDATE opportunities SET stage = 'Closed Won' WHERE id = :id;   -- business change
  INSERT INTO outbox (event_type, payload, created_at)            -- the outside-data event
    VALUES ('OpportunityWon', :payload, now());
COMMIT;   -- one atomic commit: either both rows land, or neither does
```

There is exactly one commit, so the business change and the intent-to-publish share the row-level atomicity the database already gives you for free — no second system, no 2PC. Then a separate **relay** reads the outbox and publishes:

> "Two patterns for implementing the Message relay: The Transaction log tailing pattern [and] The Polling publisher pattern." — Richardson [[transactional-outbox]]

| Relay | How | Keeps cheap | Makes expensive |
|---|---|---|---|
| **Polling publisher** | a process polls the outbox table for unsent rows and publishes them | operational simplicity; no new infra | extra DB load + publish latency (you poll on an interval) |
| **Transaction log tailing (CDC)** | tail the DB commit log (e.g. Debezium) and publish committed outbox inserts | low latency; no polling load on the table | more moving infrastructure to run and reason about |

### 5.3 At-least-once and idempotency

Either relay delivers **at-least-once** — the relay can publish a row, crash before marking it sent, and republish on restart. Therefore:

> "Consumers must be **idempotent**." — Richardson (Guideline) [[transactional-outbox]]

This is exactly why the outbox (a *data* concern) and idempotency keys (an *API-contract* concern from [[ch-05]]) are the same conversation: the contract from ch-05 ("make writes idempotent so retries are safe under at-least-once delivery," [[fielding-rest]], [[transactional-outbox]]) is *required* by the delivery semantics the outbox produces here. Idempotency is the consumer-side mirror of the outbox's producer-side guarantee. And note the closing of the loop with Helland: the event the outbox emits *is* outside data — immutable, identity-stamped, safe to redeliver — which is the only reason at-least-once-plus-idempotency is sound at all. [[helland-data-outside-inside]]

### 5.4 Pricing the outbox bet

**Keeps cheap:** correctness under crashes — you never lose an event or fire a phantom, so the saga can't silently stall or fork. It also keeps your services genuinely decoupled (no shared DB, no 2PC). **Makes expensive:** an extra table, a relay process to operate and monitor, publish latency (polling) or infra complexity (CDC), and a hard *requirement* that every consumer be idempotent — which is itself design work, not free. If a write never needs to announce itself across a boundary, you owe none of this; the outbox is the toll for crossing, not a tax you pay everywhere.

---

## 6. Applied to the Sales Agent (Lina TMR)

The learner's production sales agent — an LLM acting over many external SaaS tool APIs (CRM, email, calendar, ticketing) — is, structurally, a distributed system whether or not it is deployed as one. Helland's distinction is the single highest-leverage boundary decision available to it.

### 6.1 Every external SaaS response is outside data

When the agent calls Salesforce and gets back an opportunity, or reads a Google Sheet row, **that response is outside data**: a versioned, possibly-stale snapshot, true only as of when Salesforce sent it — *never* authoritative live state inside the agent. [[helland-data-outside-inside]] The design consequence:

- The agent must keep its **inside model** (its own private, mutable working state for the conversation/deal it is driving) strictly separate from the **outside snapshots** it ingests. Treating a stale CRM read as live truth is the agent's version of "business logic making decisions on inconsistent information" [[fowler-microservices]] — e.g. acting on a deal stage that another rep changed thirty seconds ago.
- Because outside data may be stale, the agent should **re-read just before a consequential write** (the `re-read value` countermeasure from §3.4): confirm the opportunity is still in the stage it expects before it marks it Closed Won.

### 6.2 A multi-app action is a saga

Consider an agent action that must, in one logical operation: mark a Salesforce opportunity Closed Won, then send a routing email via Gmail, then create a follow-up task in the calendar. These are three **independent systems with no shared transaction** — exactly Helland's no-shared-transactions claim. So this action *is* a saga, and the agent must treat it as one:

- **Define compensations up front.** If the Gmail send fails after Salesforce is already marked Closed Won, what is the semantic undo? Often there is *no clean compensation* for an external side-effect (you can't un-send an email, and you may not want to un-win a deal) — which is itself the most important finding. Where compensation is impossible, the saga step must be ordered last (`pessimistic view`) or made a notification rather than a state change.
- **Price the lost-isolation window.** Between "deal marked won" and "task created," another agent run or a human in the CRM sees a won deal with no follow-up scheduled. Is that anomaly tolerable? If not, a semantic lock (a `PENDING_FOLLOWUP` flag) is the buy-back.
- **Choreography vs orchestration maps to agent design.** An LLM planner driving the steps and reacting to each tool result is, in effect, an **orchestrator** — readable and debuggable, but a single point where the whole process logic (and failure handling) concentrates. Pure event-reactive tool chains would be choreography — more decoupled, far harder to trace when a multi-app action stalls. For an agent whose failures must be auditable, the orchestration bet usually wins, and §3.3's cost (orchestrator gravity) is the price to accept consciously.

### 6.3 The outbox for the agent's own writes

If the agent persists its own decisions (e.g. "I marked deal X won and queued notification Y") and emits them to other internal components, it faces the dual-write problem directly: it must not record a decision without emitting the resulting action, nor emit an action whose record rolled back. The outbox pattern applies unchanged — and the at-least-once delivery it implies means every external tool call the relay drives must be **idempotent or guarded by an idempotency key** (§5.3, [[ch-05]]), so a relay retry doesn't send the routing email twice. This is the cross-application coordination skill the learner's prior benchmark work measured; here it is the thing being *designed*, not evaluated.

---

## Where this goes

This chapter installed the *mandatory* consistency mechanics: inside/outside data, the saga and its lost isolation, and the outbox that emits each step's event without a dual-write. These are the patterns you owe the moment a boundary is real. [[ch-07]] turns to the **optional power tools** layered on top — **CQRS** ("use a different model to update information than the model you use to read information") and **Event Sourcing** ("capture all changes to application state as a sequence of events"). The pivot is the spine again: where ch-06's patterns are forced by crossing a boundary, ch-07's are *refused by default* — Young's load-bearing caution is that both "add risky complexity" and belong only on the specific portion of a system that a concrete force (asymmetric read/write scaling, a true audit/replay need) demands. The in-process seed is already in your hands: the aggregate emitting domain events from [[ch-03]] is one immutable-event log away from event sourcing — but you should refuse that path until something specific forces it. [[young-cqrs-es]]
