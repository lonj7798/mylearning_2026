<!-- chapter: ch-10
     track: capstone
     kind: lab
     title: Lab: Architecting the Production Sales Agent (ADR + C4)
     deps: [[ch-09]], [[ch-07]]
     sources: [[ddd-bounded-context]], [[decompose-by-business-capability]], [[fowler-monolith-first]], [[richards-ford-hard-parts]], [[distributed-monolith]], [[helland-data-outside-inside]], [[transactional-outbox]], [[consumer-driven-contracts]], [[fielding-rest]], [[nygard-release-it]], [[c4-model]], [[richards-ford-fundamentals]], [[richardson-saga]], [[ddd-aggregates-tactical]], [[martin-clean-arch]], [[cockburn-hexagonal]], [[conway-team-topologies]], [[young-cqrs-es]], [[fowler-microservices]], [[newman-building-microservices]], [[martin-strangler-fig]]
-->

# Chapter 10 — Lab: Architecting the Production Sales Agent (ADR + C4)

> **Core insight.** This lab is not a new idea — it is the whole course run once, in order, against one real system. Nine chapters gave you a toolkit; the only thing left to learn is the *sequence* in which an architect reaches for it, and the discipline of pricing every reach as a bet. The sequence is fixed and it is not negotiable: find the boundaries first (they are the expensive-to-reverse decision, so you defer the *deployment* commitment but make the *modeling* commitment early), structure the inside of each boundary, choose a topology by trade-off analysis rather than fashion, design the contracts, decide what crossing a boundary costs, reach for power tools only under force, then engineer the edges to fail safely and the whole to stay revisable. The deliverable is two artifacts — an ADR set and a C4 sketch — and the test of whether you learned the course is whether every box on the C4 Container diagram has a one-line trade-off behind it.

> **Guideline.** Run the eight steps below in order against Lina TMR. At every step, do three things and only three things: (1) name the pattern, (2) state the bet in the form *"this keeps X cheap to change at the cost of making Y expensive,"* and (3) write it into an ADR if it is architecturally significant. Default every topology decision toward the modular monolith ([[fowler-monolith-first]]) and refuse every "best practice" framing — there are none ([[richards-ford-hard-parts]]). You are done when a stranger could read your ADRs and your C4 Container diagram and reconstruct not just *what* you decided but *what you were betting on*.

---

## 0. The brief

**Lina TMR** is a production sales agent: an LLM agent that operates autonomously over many external SaaS tool APIs (CRM, email, calendar, e-signature, enrichment, messaging, billing) to move deals through a pipeline on behalf of a sales team. It is triggered by events (a new lead, an inbound reply, a stage change) and by scheduled sweeps, and it acts — it sends mail, books meetings, updates CRM records, drafts contracts — with bounded human oversight. The learner previously built an *agent benchmark* (the automation-bench course); this lab is the inverse and the payoff: **stop evaluating an agent and design one.**

This is the right system for the capstone because it is saturated with the exact decisions the course is about:

- It is **boundary-rich**: "lead," "conversation," "meeting," and "contact" mean different things in different parts of the system — the polysemy signal from [[ddd-bounded-context]].
- It is **integration-heavy**: every external SaaS call is, by definition, [[helland-data-outside-inside|outside data]] — a versioned, possibly-stale snapshot, never authoritative live state.
- It is **failure-exposed**: dozens of third-party vendors, each of which can be slow or down, fan into one agent loop ([[nygard-release-it]]).
- It is **long-lived**: the domain policy ("how we qualify a lead," "when we escalate") must outlive whichever LLM API, vector DB, or framework is current this quarter ([[martin-clean-arch]]).

The deliverable, defined up front so every step serves it: an **ADR-style design memo** (a set of Title/Status/Context/Decision/Consequences records) plus a **C4 Context + Container sketch** in which a distributed monolith would become visible. Templates are in §10–11; the rest of the lab fills them.

### 0.1 The myth this lab exists to kill

The doc-vs-reality table in [[COLLECTION-PLAN]] lists ten myths, one per topic. The capstone's job is to kill the one that sits *above* all of them — the meta-myth the spine attacks from chapter one:

> "Microservices / DDD / CQRS / REST are best practices you should adopt."

They are not. *Software Architecture: The Hard Parts* is built on the opposite stance: it tackles "difficult problems… with no best practices that force you to choose among various compromises," and it teaches "how to think critically about the trade-offs involved with distributed architectures" ([[richards-ford-hard-parts]], book — thesis extracted). The First Law states the same thing positively: "Everything in software architecture is a trade-off. If you think you've found something that isn't a trade-off, you likely just haven't found the trade-off yet" (Richards & Ford, [[richards-ford-fundamentals]], book — thesis extracted, quoted as commonly published). So the lab's pass/fail criterion is brutal and simple: **if you cannot name the cost of a choice, you have not made the choice — you have copied it.**

---

## 1. Step 1 — Find the bounded contexts (the [[ch-02]] move)

Boundaries first, because they are the most expensive-to-reverse decision in the system, and the only kind worth committing to is one drawn on the slowest-changing structure. Evans' thesis, as Fowler quotes it:

> "Total unification of the domain model for a large system will not be feasible or cost-effective." — Eric Evans, via Fowler ([[ddd-bounded-context]])

The operational signal for where to cut is linguistic:

> "You need a different model when the language changes." — Fowler ([[ddd-bounded-context]])

So walk Lina's domain looking for where a word stops meaning the same thing. The polysemy is real and immediate:

| Word | In one context it means… | In another it means… | ⇒ boundary signal |
|------|---------------------------|----------------------|-------------------|
| **Lead / Contact** | a pipeline entity with a stage, score, owner | a CRM record with a vendor ID and sync status | Pipeline vs CRM-Sync |
| **Conversation** | a thread of inbound/outbound messages the agent reasons over | a sequence of provider-specific email/SMS payloads | Conversation vs (the messaging adapter) |
| **Meeting** | a commitment with attendees and an agenda | a calendar event with a provider event-ID and timezone | Scheduling vs (the calendar adapter) |
| **Deal won** | a pipeline outcome that triggers handoff | a CRM `stage_name` field update | Pipeline vs CRM-Sync |

Candidate bounded contexts for Lina, each a unified model with one [[ddd-bounded-context|ubiquitous language]]:

1. **Lead / Pipeline** — leads, scoring, stages, qualification *policy*. The slowest-changing, highest-value core.
2. **Conversation** — message threads, agent turns, the reasoning loop, drafted-vs-sent state.
3. **Scheduling** — meeting intents, availability, booking commitments.
4. **CRM-Sync** — reconciliation between Lina's inside model and the external CRM's outside model.

**The bet (price it).** Drawing the boundary on the *language/domain* (the slowest-changing structure, [[decompose-by-business-capability]]: "stable architecture since the business capabilities are relatively stable") keeps the model **cheap to evolve inside each context** at the cost of making **cross-context invariants expensive** — anything that must be true across, say, Pipeline and Scheduling now needs an explicit translation and eventual consistency (Step 5), not a shared object. A *wrong* boundary here is the most expensive mistake available: "model rot you pay for daily." This is why you commit the *modeling* now but defer the *deployment* commitment to Step 3.

**Myth killed (ch-02's):** "DDD requires microservices." False — DDD is a modeling discipline; these four contexts can be four *modules* in one deployable. No DDD source ties contexts to a topology ([[ddd-bounded-context]], [[COLLECTION-PLAN]]). That is exactly the door Step 3 walks through.

**Context mapping.** Between Lina's contexts and the *outside* world, the load-bearing relationship is the **Anticorruption Layer**: "translate the other context's model so it can't leak into yours" ([[ddd-bounded-context]], from Vernon's *DDD Distilled*). Every SaaS vendor's schema is a foreign model; the ACL is where you stop Salesforce's `Opportunity` or HubSpot's `Deal` from becoming Lina's vocabulary.

---

## 2. Step 2 — Structure the inside of each boundary (the [[ch-03]] move)

Inside each context, one rule governs everything. State it once, apply it four times:

> "Source code dependencies can only point inwards." — Robert C. Martin ([[martin-clean-arch]])

The same idea, phrased as a testability goal:

> "Allow an application to equally be driven by users, programs, automated test or batch scripts, and to be developed and tested in isolation from its eventual run-time devices and databases." — Alistair Cockburn ([[cockburn-hexagonal]], Intent quote; the canonical alistair.cockburn.us URL had an expired TLS cert on 2026-06-15, so this is corroborated via the alistaircockburn.com mirror — cite the mirror, not a clean fetch).

Clean = Hexagonal = Onion: a technology-free core, dependencies pointing inward, I/O at the edges through interfaces (ports) the core owns ([[martin-clean-arch]], [[cockburn-hexagonal]]). For Lina this is not optional polish — it is the bet that keeps the domain policy alive across vendor churn.

**Ports for Lina** (interfaces the core defines, in the core's own language):

- `ForQualifyingLeads` (primary/driving — an event or sweep drives the core)
- `ForStoringPipeline` (secondary/driven — the core calls out to persistence)
- `ForSendingMessages`, `ForBookingMeetings`, `ForSyncingCRM`, `ForReasoning` (secondary/driven — every SaaS vendor and the LLM itself sits behind one)

The LLM provider is a *driven adapter behind `ForReasoning`*. This is the single most important application of the dependency rule in Lina: the model you call this quarter (Claude, an open weights model, a fine-tune) is a "tool you call, not a base class you inherit your whole app from" ([[martin-clean-arch]]). Swapping it must be an adapter swap, never core surgery.

### 2.1 Tactical DDD — aggregates as in-process consistency boundaries

Inside the core, the **aggregate is the unit of immediate transactional consistency** ([[ddd-aggregates-tactical]]). Vernon's four rules (book — thesis extracted), applied to Lina:

1. **Model true invariants in consistency boundaries** — e.g. a `Lead` and its `Score` must be consistent together, immediately; a `Lead` and a booked `Meeting` need not be.
2. **Design small aggregates** — `Lead` is an aggregate; do not fold the whole conversation history into it.
3. **Reference other aggregates by identity** — a `Meeting` holds a `LeadId`, not a `Lead` object.
4. **Use eventual consistency outside the boundary** — when a won deal must trigger CRM-Sync, publish a domain event (`DealWon`), don't reach across and write CRM-Sync's data in the same transaction.

Rule 4 is load-bearing for the whole rest of the lab: it is "the in-process seed of the distributed saga" ([[ddd-aggregates-tactical]]). Get aggregate boundaries right *now*, inside the modular monolith, and a future service split inherits correct consistency boundaries for free.

**The bet (price it).** Ports + DTOs + mappers + small aggregates are **indirection and boilerplate** — "overkill for a small CRUD service" ([[martin-clean-arch]]). They keep the **expensive part (domain policy) cheap to preserve** across vendor and framework churn, at the cost of making **the simplest CRUD paths more verbose**. For Lina's long-lived core this is the right bet; for a throwaway script it would be waste. Name it as a bet, not a default.

---

## 3. Step 3 — Choose the topology: modular-monolith-first + the quantum (the [[ch-04]] move)

This is the spine of the whole lab, so price it most carefully.

**Default: a modular monolith.** Fowler's empirical claim is blunt:

> "Almost all the successful microservice stories have started with a monolith that got too big and was broken up." — Fowler ([[fowler-monolith-first]])

> "Almost all the cases where I've heard of a system that was built as a microservice system from scratch, it has ended up in serious trouble." — Fowler ([[fowler-monolith-first]])

The reason is boundary uncertainty, and it bites hardest exactly where Lina is — early, in an evolving domain:

> "Even experienced architects working in familiar domains have great difficulty getting boundaries right at the beginning." — Fowler ([[fowler-monolith-first]])

So the four contexts from Step 1 become four **modules** in one deployable: one schema/namespace per module, in-process interfaces between them, no module reaching into another's tables ([[fowler-monolith-first]]). The recommendation, verbatim:

> "Start a new application as a monolith initially, even if you think it's likely that it will benefit from a microservices architecture later on." — Fowler ([[fowler-monolith-first]])

**The central bet of this chapter, stated explicitly.** A modular monolith keeps your **boundaries cheap to redraw** — inside a monolith a bad boundary is a *refactor*; across services it is a *migration* ([[fowler-monolith-first]]) — at the cost of making **per-module deployment, scaling, and tech-choice independence expensive** (you cannot deploy or scale Conversation without redeploying Pipeline; you cannot run the LLM-heavy reasoning loop on different hardware from the CRUD pipeline without splitting). You are betting that, for Lina today, *the probability and cost of getting a boundary wrong* exceeds *the value of independent deployment of any one module*. That is almost certainly true at the start and may stop being true later — which is why Step 8 keeps the bet revisable.

### 3.1 The architecture quantum — the boundary test

Before extracting *anything*, test it with the quantum. An **architecture quantum** is "the smallest unit that is independently deployable, has high functional cohesion, and… includes its own data" ([[richards-ford-hard-parts]], book — thesis extracted). The detector: **count your quanta, not your services.** Two "services" that share a database are *one* quantum — and a [[distributed-monolith]].

### 3.2 Disintegrators vs integrators — size is an output, not a rule

The myth killed here (ch-04's): "pick service size by a rule of thumb." No — enumerate forces ([[richards-ford-hard-parts]]):

| For the Conversation/reasoning module, should we extract it? | |
|---|---|
| **Disintegrators (split smaller)** | divergent scalability (LLM calls are slow + bursty, unlike CRUD); fault isolation (a wedged reasoning loop shouldn't stall pipeline sweeps); distinct code-volatility (prompt/model logic changes weekly); possibly separate team ownership |
| **Integrators (keep together)** | tight data dependency on `Lead` and `Pipeline` state; chatty workflow between reason→act→update; shared domain types that change together; no DB transaction need *across* the seam if events are used |

You split **only when disintegrators outweigh integrators** ([[richards-ford-hard-parts]]). For Lina, the *reasoning/Conversation* module is the strongest extraction candidate (scalability + fault-isolation + volatility all pull to split); CRM-Sync is second (it is I/O-bound and fails independently). Pipeline is the *last* thing you'd extract — it's the cohesive core. **Decision for the memo:** start all four as modules; flag Conversation as the first extraction candidate *when* its scaling/fault-isolation forces actually bite, and extract it via Strangler Fig (Step 8).

### 3.3 The distributed-monolith trap and Conway

Myth killed: "split and get loose coupling for free." If the extracted modules must deploy together or share a DB, you built "all the pain of distributed systems without the independence" ([[distributed-monolith]]). The litmus test is independent deployability ([[newman-building-microservices]]: "the single most important principle," book — thesis extracted).

And the socio-technical constraint: "Any organization that designs a system… will produce a design whose structure is a copy of the organization's communication structure." — Melvin Conway, 1968 ([[conway-team-topologies]]). For a small team owning all of Lina, this *argues for the monolith*: there is no team boundary to mirror, so a service split would invent a coordination cost (cognitive load across deploy units) that buys nothing. The Inverse Conway Maneuver — "deliberately restructure teams to induce the architecture you want" ([[conway-team-topologies]]) — only becomes relevant if/when a dedicated team forms around, say, the reasoning module.

---

## 4. Step 4 — Design the tool-layer contracts (the [[ch-05]] move)

Lina's tool/integration layer — the adapters behind `ForSendingMessages`, `ForSyncingCRM`, etc. — is where contracts live, and the contract is "the most expensive-to-reverse artifact a service owns" ([[consumer-driven-contracts]]). Two contract surfaces matter:

**(a) The contracts Lina consumes (the SaaS vendors').** Be a **tolerant reader**:

> "An implementation must be conservative in its sending behaviour and liberal in its receiving behaviour… message receivers should implement 'just enough' validation: that is, they should only process data that contributes to the business functions they implement." — Ian Robinson ([[consumer-driven-contracts]])

Concretely: Lina's CRM adapter should read only the fields it needs from the CRM payload and ignore the rest, so a vendor adding a field never breaks Lina. Strict schema validation on the consumer side is precisely "what turns additive changes into breaking ones" ([[consumer-driven-contracts]]).

**(b) The contracts Lina exposes** (if/when Conversation is extracted, the API the rest of Lina depends on). Here you publish a contract and treat consumer expectations as obligations:

> "When a provider accepts and adopts the reasonable expectations expressed by a consumer, it enters into a consumer contract." — Robinson ([[consumer-driven-contracts]])

Consumer-driven contract tests in CI are "the only mechanism that preserves independent deployability" — the provider knows, before deploy, whether it breaks anyone ([[consumer-driven-contracts]], [[newman-building-microservices]]).

### 4.1 API style and the honest-REST myth

Pick style by the property you need, not by default ([[fielding-rest]]):

- **REST/HTTP** for cacheable, evolvable resources (Lina's own management API, if any).
- **gRPC** for low-latency internal service-to-service (a future Pipeline↔Conversation call).
- The external vendors are whatever they are; Lina adapts.

Myth killed (ch-05's): "our API is RESTful." Fowler, quoting Fielding: "Roy Fielding has made it clear that level 3 RMM is a pre-condition of REST" ([[fielding-rest]], from roy.gbiv.com — the ics.uci.edu mirror had a broken TLS chain, so quotes are from Fielding's own gbiv.com mirror). Almost everything the industry calls REST is Level-2 HTTP-RPC. **Be honest in the ADR about which you ship** — it costs nothing to be accurate and it sets correct expectations about evolvability.

### 4.2 Versioning and idempotency

Prefer **additive, backward/forward-compatible change** over versioned breakage ([[fielding-rest]]). And make every *write* Lina performs **idempotent** — "make writes idempotent (idempotency keys) so retries are safe under at-least-once delivery" ([[fielding-rest]], [[transactional-outbox]]). This is not optional for an agent: Lina *will* retry a "send email" or "create CRM record" call after a timeout, and without an idempotency key it will double-send. Idempotency (a contract concern) and the outbox (a data concern, Step 5) "are the same conversation" ([[transactional-outbox]]).

**The bet (price it).** A published, contract-tested interface keeps **the boundary cheap to evolve** (additive change can't break a tolerant reader; a breaking change is caught in CI) at the cost of **the discipline tax**: every change runs the contract suite, and breaking changes still cost coordinated migration. The cost is real; it is the price of independent deployability, which is the only thing that makes a future split worth doing.

---

## 5. Step 5 — Treat every SaaS response as outside data; saga + outbox (the [[ch-06]] move)

This is the deepest cut in the course and it applies to *every external call Lina makes*. Helland's root distinction:

> Inside data is "the realm of SQL and SQL's DDL" — private, mutable, transactional, "now." Outside data "is immutable and each data item's schema is versioned"; it "is stable, such that a repeated request is unchanged, and a reading of it results in the same interpretation." — Pat Helland ([[helland-data-outside-inside]]; the ACM Queue reprint returned 403 at fetch, so claims are corroborated via the CIDR-2005 PDF, Semantic Scholar, and "the morning paper" summary — cite as a summary-corroborated source, not a clean fetch).

The three load-bearing claims, applied to Lina:

1. **Services don't share transactions.** Lina cannot wrap a transaction around its own DB *and* the CRM. So 2PC across Lina+vendor is a dead end ([[helland-data-outside-inside]]).
2. **Outside data must be immutable.** Every CRM record, every enrichment result, every calendar event Lina ingests is a *snapshot*, stored immutably and identity-stamped — "safe to retry, cache, reorder, and replay" ([[helland-data-outside-inside]]).
3. **Outside data may be stale, and that's fine.** The CRM record Lina read 200ms ago may already be wrong. Design for it. Fowler's framing of the cost: "business logic can end up making decisions on inconsistent information" ([[fowler-microservices]], [[helland-data-outside-inside]]).

**The single highest-leverage decision in Lina:** separate the agent's **inside model** (its private Pipeline/Conversation state, ACID, "now") from **what it ingests and emits** (immutable, versioned, possibly-stale SaaS snapshots). Per [[helland-data-outside-inside]], this is "the single most leverage-rich boundary decision available" for an agent over many SaaS tools.

### 5.1 Sagas for multi-step external operations

When one Lina operation must update several external systems — e.g. *win a deal* = update CRM stage → send win-notice email → book a handoff meeting → notify Slack — you cannot do it in one transaction. Use a **saga**:

> "A saga is a sequence of local transactions. Each local transaction updates the database and publishes a message or event to trigger the next local transaction in the saga." — Chris Richardson ([[richardson-saga]])

(The construct originates in Garcia-Molina & Salem's 1987 "Sagas" paper for long-lived transactions — but that PDF is image-only with no text layer, so the LLT/compensation thesis is extracted from knowledge of the paper, **not** quoted verbatim; Richardson's microservices.io page is the verbatim-quotable source.)

**Choreography vs orchestration** — for Lina's "win a deal" flow, *orchestration* is the better bet: it is a "complex, many-interdependent-step" flow, and an orchestrator gives you "one place to see/debug the process" ([[richardson-saga]]) — which matters enormously for an autonomous agent you must audit. The cost is that "the orchestrator becomes a bottleneck" and a coupling point.

**The price of a saga — name it.** You trade away **isolation (the "I" in ACID)**:

> "Lack of isolation (the 'I' in ACID)… means there's risk that the concurrent execution of multiple sagas and transactions can [cause] data anomalies." — Richardson ([[richardson-saga]])

So Lina needs **countermeasures** — semantic locks (mark a deal "win-in-progress"), commutative updates, re-reads, by-value tracking ([[richardson-saga]]). A saga is *not* a free transaction. And every step needs a **compensating transaction**: if the win-notice email fails after the CRM update succeeded, you either retry forward or compensate (revert/flag the CRM change). Design the compensations, don't hope.

### 5.2 The transactional outbox

Lina cannot atomically write its DB *and* publish an event to trigger the next saga step:

> "How to atomically update the database and send messages to a message broker?" … "It is not viable to use a traditional distributed transaction (2PC) that spans the database and the message broker." — Richardson ([[transactional-outbox]])

The fix:

> "The solution is for the service that sends the message to first store the message in the database as part of the transaction that updates the business entities." — Richardson, with the guarantee "Messages are guaranteed to be sent if and only if the database transaction commits." ([[transactional-outbox]])

Relay it asynchronously — **polling publisher** (simple, more DB load/latency) or **transaction-log tailing / CDC** (lower latency, more infra) ([[transactional-outbox]]). Delivery is **at-least-once**, so every Lina consumer of these events **must be idempotent** — which closes the loop with the idempotency keys from Step 4.

**The bet (price it).** Inside/outside-data discipline + saga + outbox keeps **each external integration independently failable and replayable** (cheap to retry, cache, and reason about) at the cost of **giving up cross-system isolation and "live truth"** — Lina must everywhere accept staleness and design compensations. For an agent over a dozen flaky SaaS vendors, this is not a choice; it is the cost of operating at all. The only choice is whether you pay it deliberately (with a designed model) or accidentally (with a [[distributed-monolith]] that treats CRM rows as live shared state).

---

## 6. Step 6 — Reach for CQRS/ES only if a force demands it (the [[ch-07]] move)

Power tools, refused by default. CQRS:

> "CQRS stands for Command Query Responsibility Segregation. At its heart is the notion that you can use a different model to update information than the model you use to read information." — Fowler ([[young-cqrs-es]])

The load-bearing caution:

> "For some situations, this separation can be valuable, but beware that for most systems CQRS adds risky complexity." … "CQRS should only be used on specific portions of a system… and not the system as a whole." … "Many systems do fit a CRUD mental model, and so should be done in that style." — Fowler ([[young-cqrs-es]])

**For Lina:** the Pipeline core is mostly CRUD — refuse CQRS there. *One* portion plausibly earns it: a **reporting/analytics read model** over sales activity (pipeline velocity, agent-action dashboards) whose query shape diverges hard from the write model and whose read load scales independently. Scope CQRS to *that portion only*, fed by the same domain events the outbox already emits.

Event Sourcing:

> "Capture all changes to an application state as a sequence of events." — Fowler ([[young-cqrs-es]])

The caution:

> "Clearly this stuff can get very messy, don't go down this path unless you really need to." — Fowler ([[young-cqrs-es]])

**For Lina:** there is a *genuine* candidate force — an autonomous agent that takes consequential actions has a real **audit/replay need** ("why did Lina email this lead and book that meeting?"). Event Sourcing's free audit log and replay-debugging map directly onto agent accountability. But weigh the sharp edges: external side-effects on replay (Lina must gateway/suppress real sends when replaying) and historical-event schema evolution ([[young-cqrs-es]]). **Decision for the memo:** event-source the *Conversation/agent-action* context (where audit matters most) and leave Pipeline/Scheduling CRUD — *if and only if* the audit requirement is real, not aspirational.

**Myth killed (ch-07's):** "CQRS and Event Sourcing are the same / go together." They are independent decisions that compose; neither requires the other, and neither requires microservices ([[young-cqrs-es]]). The in-process seed of both is the aggregate emitting domain events (Step 2.1, [[ddd-aggregates-tactical]]) — which you already built.

**The bet (price it).** Each tool keeps **one specific property cheap** (CQRS: independent read scaling / query-shape freedom; ES: audit + temporal replay) at the cost of **risky complexity across the whole portion that adopts it**. Refuse both unless a named force (asymmetric read scaling; real audit/replay need) makes the cost worth paying. The default answer is "no."

---

## 7. Step 7 — Place timeouts/breakers/bulkheads at every integration point (the [[ch-08]] move)

Every external SaaS call is an integration point, and the integration point is where failure enters:

> "Integration Points without Timeouts is a surefire way to create Cascading Failures." — Michael Nygard ([[nygard-release-it]], *Release It!* — book, thesis extracted)

The toolkit, applied to Lina ([[nygard-release-it]]):

- **Timeout** — *every* remote call (CRM, email, calendar, LLM) gets a bound. "An unbounded wait is a held resource," and held resources are how one slow vendor hangs the whole agent loop.
- **Circuit Breaker** — wrap each flaky vendor; once failures cross a threshold, *open* and fail fast instead of piling requests on a dead service; *half-open* periodically to test recovery. If the enrichment API is down, Lina should skip enrichment, not stall every deal.
- **Bulkhead** — partition resource pools per vendor (and isolate the LLM-call pool) so one vendor's outage "does not sink the vessel." A storm of CRM timeouts must not exhaust the threads Lina needs to send mail.
- **Fail Fast** / **Steady State** — detect un-completable work and return immediately; ensure every accumulation (conversation logs, cached snapshots) has a matching cleanup ([[nygard-release-it]]).

**Why this is design altitude, not ops:** "where you place circuit breakers and bulkheads defines your system's blast radius. A [[distributed-monolith]] is precisely a system that skipped them" ([[nygard-release-it]]). The C4 Container diagram (Step 9) should make every integration point visible *as* a place a breaker lives.

**Async is itself a resilience decision.** Choosing event/outbox integration over synchronous call-chains "removes the synchronous coupling these patterns otherwise have to defend" ([[nygard-release-it]], [[richardson-saga]]). Lina's saga-over-outbox design from Step 5 already buys much of this: a downstream vendor outage delays an event, it doesn't synchronously stall the agent.

**The bet (price it).** Resilience patterns keep **the blast radius small** (one vendor down ≠ Lina down) at the cost of **added latency, complexity, and degraded-mode logic** — Lina must now have a defined behavior for "enrichment is open-circuited" (proceed without it? queue for later?). That degraded-mode design is work; skipping it is how a distributed system cascades.

---

## 8. Step 8 — Pick the first seam to strangle + fitness functions (the [[ch-09]] move)

The modular monolith is the *starting* bet, not the *final* one — keep it revisable. When a module's forces finally justify extraction (Step 3.2 flagged Conversation first), do not rewrite:

> "Replacing a serious IT system takes a long time, and the users can't wait for new features." — Fowler ([[martin-strangler-fig]])

Use the **Strangler Fig**: intercept via façade → extract one capability seam (and its data, together, so you don't create a shared-DB dependency mid-migration, per [[newman-building-microservices]]) → verify & shrink → repeat ([[martin-strangler-fig]]). "Investment and returns occur gradually and visibly," and "since these components are small, there isn't so much risk involved" ([[martin-strangler-fig]]). It is the operational counterpart to modular-monolith-first: both "refuse to make a single large irreversible bet on boundaries you can't yet trust."

**Lina's first strangle target:** the Conversation/reasoning module, because (from Step 3.2) it has the strongest disintegrator profile — bursty LLM-bound scaling, fault-isolation need, weekly volatility. Extract it behind the `ForReasoning` port that already exists (Step 2): the port *is* the interception seam. This is the payoff of doing Step 2 properly — the strangler has a clean place to cut.

### 8.1 Fitness functions — stop the bet from rotting

Keep the chosen characteristics enforced, not aspirational. A **fitness function** is "an objective integrity assessment of some architectural characteristic(s)" — an automated check that fails the build when a protected characteristic erodes ([[richards-ford-fundamentals]], from *Building Evolutionary Architectures*). For Lina, encode the lab's decisions as fitness functions:

| Decision to protect | Fitness function |
|---|---|
| Dependency rule holds (Step 2) | ArchUnit-style test: no `domain` package imports a vendor SDK or framework type |
| Modules don't reach into each other's tables (Step 3) | schema-access lint: module X's code touches only module X's schema |
| Inside ≠ outside data (Step 5) | test: no SaaS DTO type appears in a core aggregate's fields |
| Every remote call has a timeout (Step 7) | static check / test: no un-timed HTTP client in adapters |
| p99 agent-loop latency < target | monitor-as-fitness-function in CI/prod |

"This is how you keep the expensive-to-reverse decisions from rotting silently as the system evolves" ([[richards-ford-fundamentals]]).

**The bet (price it).** Evolutionary discipline (strangler + fitness functions) keeps **the architecture revisable and self-defending** at the cost of **ongoing enforcement and migration effort** — fitness functions are tests you must write and maintain; strangling is slower than a rewrite per-step. You pay continuously so that no single decision becomes silently unrevisable. That is the whole point of the course: architecture is the expensive-to-reverse set, and the master move is to *keep shrinking it* — Fowler's amendment, "a good architect makes change easier, thus reducing architecture" ([[richards-ford-fundamentals]]).

---

## 9. Worked partial example — the **CRM-Sync** bounded context

To show the shape, run one context end-to-end through the toolkit. (You complete the other three.)

**Context:** CRM-Sync — reconcile Lina's inside Pipeline model with an external CRM's outside model.

1. **Boundary (Step 1):** the language shifts — "Lead" (pipeline entity) ⇄ "Contact/Opportunity" (CRM record with vendor ID + sync status). Distinct model ⇒ its own context. **Bet:** isolating sync vocabulary keeps Pipeline's model clean (cheap to evolve) at the cost of an explicit translation layer (the ACL) that must be maintained.
2. **Structure (Step 2):** core defines `ForSyncingCRM` (port); a `SalesforceAdapter` / `HubSpotAdapter` implements it (driven adapter). Core depends on neither vendor's SDK. **Aggregate:** `SyncRecord { leadId, vendorId, lastSyncedSnapshot, status }` — small, references `Lead` by identity.
3. **Topology (Step 3):** a *module*, not a service, initially. Disintegrators (I/O-bound, fails independently) make it the *second* extraction candidate — but integrators (tight `Lead` dependency) say "not yet."
4. **Contract (Step 4):** tolerant reader on the CRM payload — read only `{id, stage, owner, parentAccount}`, ignore the rest, so vendor field additions never break Lina. Idempotency key on every CRM write.
5. **Consistency (Step 5):** every CRM read is **outside data** — store the snapshot immutably, stamped with fetch-time and vendor version; never treat it as live truth. A "win a deal" flow that touches CRM is one *step* of an orchestrated saga, with a compensating "revert stage" transaction; the trigger event comes via the outbox.
6. **Power tools (Step 6):** no CQRS, no ES here — it's CRUD-shaped reconciliation. Refused, by name.
7. **Resilience (Step 7):** timeout + circuit breaker + dedicated bulkhead pool around the CRM adapter. Degraded mode: if the breaker is open, queue the sync via outbox and proceed; don't stall the agent loop.
8. **Evolution (Step 8):** fitness function — "no Salesforce/HubSpot type appears in a core aggregate"; flagged as a future strangle target behind `ForSyncingCRM` if a dedicated integrations team forms (Conway).

The shape is identical for Conversation, Scheduling, and Pipeline — only the forces differ. **That sameness is the lesson:** the toolkit is a fixed sequence; the trade-off weights are what change.

---

## 10. Deliverable template A — ADR skeleton (fill in)

Use Nygard's structure: "Title, Status (proposed/accepted/deprecated/superseded), Context, Decision, Consequences"; "one or two pages"; "write each ADR as if it is a conversation with a future developer" — and record only "architecturally significant" decisions: "those that affect the structure, non-functional characteristics, dependencies, interfaces, or construction techniques" ([[nygard-release-it]]).

```
# ADR-00X: <short imperative title, e.g. "Default Lina to a modular monolith">

## Status
<proposed | accepted | deprecated | superseded by ADR-0YY>

## Context
<the forces in play: which architecture characteristics matter here, what
 the requirement is, what alternatives exist. State the uncertainty honestly.>

## Decision
<what we will do, in one or two sentences. Active voice.>

## Consequences
<THE BET, priced both ways:
   - This keeps CHEAP TO CHANGE: <...>
   - This makes EXPENSIVE: <...>
   - Risks / what would make us revisit this (the trigger to supersede): <...>>
```

**Minimum ADR set this lab must produce** (one each):

1. Bounded contexts of Lina (Step 1).
2. Modular-monolith-first topology + first extraction candidate (Step 3). *(the spine ADR)*
3. Inside/outside-data discipline for all SaaS responses (Step 5).
4. Saga style (orchestration) for multi-system flows + outbox (Step 5).
5. Resilience defaults at integration points (Step 7).
6. CQRS/ES scope decision — what's in, what's refused, and why (Step 6).
7. Fitness functions guarding the above (Step 8).

Each one's Consequences section is invalid if it does not name *both* sides of the bet. That is the grading rubric.

---

## 11. Deliverable template B — C4 Context + Container sketch (fill in)

C4 is "a set of hierarchical abstractions — software systems, containers, components, and code" ([[c4-model]]). You produce two levels.

**Myth to keep front-of-mind (ch-01's, restated):** a C4 **container** is "applications and data stores… a separately runnable/deployable unit" — "NOT a Docker container" ([[c4-model]], [[COLLECTION-PLAN]]). On Lina's diagram, a "container" is the deployable agent app, its database, its message bus — regardless of how they're packaged.

**Level 1 — System Context** (the system as one box; audience: everyone):

```
[Person: Sales rep] ──uses──> ( LINA TMR — autonomous sales agent )
                                        │
        ┌───────────────┬──────────────┼───────────────┬──────────────┐
        ▼               ▼              ▼               ▼              ▼
   [Ext: CRM]     [Ext: Email]   [Ext: Calendar]  [Ext: LLM API]  [Ext: Enrichment]
   <fill: which external systems Lina talks to, and the direction/purpose of each edge>
```

**Level 2 — Container** (zoom in: the deployable apps + data stores; audience: technical/ops). **This is the diagram where a distributed monolith becomes visible** — count the deploy-coupled containers and shared data stores:

```
┌────────────────────────── LINA TMR (system boundary) ──────────────────────────┐
│                                                                                  │
│   ( Agent App  — modular monolith )            [ Outbox / Message Bus ]          │
│     modules: Pipeline | Conversation |          <fill: relay style —             │
│              Scheduling | CRM-Sync               polling vs CDC>                  │
│        │                                                                         │
│        ▼                                                                         │
│   [ Lina DB — inside data, ACID ]   [ Snapshot store — outside data, immutable ] │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
        │ (every outbound edge below crosses a port + adapter with
        │  TIMEOUT + CIRCUIT BREAKER + BULKHEAD — mark each one)
        ▼
   <fill: edges to each external SaaS, labelled with the port name and the
    resilience pattern guarding it; mark which edges are synchronous vs
    event/outbox-driven>
```

**Completion checklist for your C4 sketch** (each is a course concept you must surface):

- [ ] Is **inside data** (`Lina DB`) drawn separately from **outside data** (snapshot store)? ([[helland-data-outside-inside]])
- [ ] Does any two modules share a schema? If so you've drawn one quantum, not two — and a [[distributed-monolith]] in waiting. ([[richards-ford-hard-parts]])
- [ ] Is every external edge marked with its **port name** (dependency rule, [[martin-clean-arch]]) and its **timeout/breaker/bulkhead** ([[nygard-release-it]])?
- [ ] Are saga steps **event/outbox-driven** (async) rather than synchronous call-chains? ([[richardson-saga]], [[transactional-outbox]])
- [ ] Could you point at one container and say "this is the first one I'd strangle out, here's the force that triggers it"? ([[martin-strangler-fig]])

If every box and edge has a one-line trade-off behind it, you have passed the lab — and the course.

---

## Where this goes

There is no ch-11; this is the capstone. But the forward pointer is into your own repository: take the seven ADRs and the two C4 levels you just drafted and put them under version control next to Lina's code, then wire the §8.1 fitness functions into CI so the bets you priced today fail the build the day they start to rot. The course's final claim is that an architecture is only as alive as its enforcement: the First Law tells you every decision is a trade-off, the ADR records which trade you made, the C4 diagram shows where a wrong trade would become visible, and the fitness function is what stops the trade from quietly reversing itself while you're not looking. Architecture is the expensive-to-reverse set — and the architect's whole job, restated one last time, is to keep that set small, named, and revisable.
