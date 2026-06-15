<!-- chapter: ch-03
     track: boundaries
     kind: content
     title: Structure Inside a Boundary: Hexagonal, Clean, and Aggregates
     deps: [[ch-02]]
     sources: [[martin-clean-arch]], [[cockburn-hexagonal]], [[ddd-aggregates-tactical]], [[richards-ford-fundamentals]], [[insights]], [[COLLECTION-PLAN]]
-->

# Chapter 03 — Structure Inside a Boundary: Hexagonal, Clean, and Aggregates

> **Core insight.** Once [[ch-02]] has drawn a bounded context, the question becomes how to arrange code *inside* it so the part that is expensive to reverse — domain policy — does not rot every time a vendor, framework, or database changes underneath it. Three named architectures (Hexagonal, Clean, Onion) give the same one-sentence answer: keep a technology-free core in the center, push all I/O to the edges, and let **source-code dependencies point only inward**. Tactical DDD then tells you how to shape that core: the **aggregate** is the unit of immediate transactional consistency, and the boundary you draw around it is itself an expensive-to-reverse decision — get it too big and you bake a contention point into the architecture; get it right and the eventual-consistency seam you will need later (the saga) is already there in miniature.

> **Guideline.** Define the interfaces (ports) inside the core in the core's own language; implement them outside as adapters; cross every boundary with plain data structures, never ORM rows or framework objects. Inside the core, model true invariants as small aggregates, one transaction per aggregate, reference other aggregates by identity, and let domain events carry change across boundaries. Pay the indirection tax (ports, DTOs, mappers) only where the core is long-lived enough to outlive its vendors — which is exactly the situation of the learner's sales agent.

---

## 1. The Dependency Rule Is the Whole Architecture in One Sentence

Robert Martin's "Clean Architecture" reduces a wall of diagrams to a single enforceable rule, and the excerpt records it verbatim ([[martin-clean-arch]]):

> "Source code dependencies can only point inwards." — Robert C. Martin

That sentence is the entire load-bearing claim. Everything else — concentric circles, port names, DTO discipline — is mechanism in service of it. The rule means: nothing in an inner circle may *name* anything in an outer circle. A detail (a framework, a DB driver, an LLM SDK) may depend on a policy; a policy may **never** depend on a detail.

The concentric circles, inner to outer, as the excerpt lays them out:

| Ring | Name | What lives here | Change rate |
|------|------|-----------------|-------------|
| 1 (center) | **Entities** | enterprise-wide business rules | "least likely to change" |
| 2 | **Use Cases** | application-specific rules; orchestrate data to/from entities | slow |
| 3 | **Interface Adapters** | controllers, presenters, gateways — convert between use-case form and external form | medium |
| 4 (rim) | **Frameworks & Drivers** | web framework, database, devices — "tools rather than constraints" | fast |

This ordering is not aesthetic. It is the [[ch-01]] spine applied to file layout: **the things hardest to change sit deepest, insulated from the things easiest to change.** The rule mechanically guarantees that a churn in the fast-moving rim (swap Postgres for DynamoDB, swap one model vendor for another) cannot force an edit to the slow-moving center.

### 1.1 The independence claims, and why they follow

Martin states three things the architecture buys you. The excerpt quotes them ([[martin-clean-arch]]):

- **"Independent of Frameworks"** — "the framework is a tool you call, not a base class you inherit your whole app from."
- **"Testable"** — "business rules can be tested without UI, database, or external elements."
- **"Independent of UI, Database, [and] any external agency"** — each is replaceable without touching core logic.

These are not three separate features; they are three consequences of the one rule. If no inner code names outer code, then by construction the inner code can be compiled, run, and tested with the outer code absent or faked. Testability is not an add-on you bolt on later — it is what "dependencies point inward" *is*, observed from the test harness.

### 1.2 How the type system tries to break the rule (and the fix)

The Dependency Rule is easy to state and easy to violate accidentally, because the violation rides in on a return type. The excerpt's second quote is the guard ([[martin-clean-arch]]):

> "The important thing is that isolated, simple, data structures are passed across boundaries." — Martin

If a use case calls a repository and the repository hands back an ORM row, the use case now transitively depends on the ORM — the inward arrow has been reversed through the type. The same happens if an inbound controller passes its framework `Request` object straight into a use case. The fix is mechanical: **cross every boundary with a plain DTO.** Map the ORM row to a domain object at the adapter; map the framework request to a command at the controller. The DTO is the airlock that stops an outer dependency from leaking inward.

This is the first concrete price of the pattern, and it is worth naming now: you write mappers. For a three-field CRUD endpoint, that mapper is pure ceremony. For the agent's domain core, it is the seam that lets the core survive a vendor it has not met yet.

### 1.3 The four rings, mapped onto Lina TMR

The ring table in §1 is abstract; pin it to the learner's system so the rule has teeth. The diagnostic that an architecture is correct is that you can fill the table in *and* every dependency still points inward:

| Ring | Lina TMR component | May depend on | May NOT depend on |
|------|--------------------|---------------|-------------------|
| Entities | `Lead`, `Deal`, qualification rules, escalation policy | nothing outside the core | the LLM SDK, Salesforce types, the web framework |
| Use cases | "qualify lead," "advance stage," "route notification" | entities + ports | concrete adapters or vendor types |
| Interface adapters | the LLM adapter, the Salesforce repo, the webhook controller | use-case ports + vendor SDKs | n/a (this is where SDKs are *allowed*) |
| Frameworks & drivers | Anthropic SDK, Postgres driver, the HTTP server | themselves | anything in the core |

The test of correctness is a grep: if any file under `entities/` or `use_cases/` imports `anthropic`, `salesforce`, or the web framework, the rule is broken and the column "may NOT depend on" was violated. Ch-09's fitness functions turn exactly that grep into an automated build-failing check, so the rule does not erode silently between commits.

---

## 2. Hexagonal: The Same Rule, Told from the Test Harness

Cockburn's Hexagonal (Ports & Adapters) architecture predates Martin's framing and motivates the identical rule from a different angle: not "policy must not depend on detail," but "I want to run my application with the screen and the database removed." The excerpt records the one line everyone quotes — corroborated via the alistaircockburn.com mirror because the canonical `.us` URL had an expired TLS certificate at fetch time on 2026-06-15 ([[cockburn-hexagonal]], gap logged in [[COLLECTION-PLAN]]):

> "Allow an application to equally be driven by users, programs, automated test or batch scripts, and to be developed and tested in isolation from its eventual run-time devices and databases." — Alistair Cockburn (via the .com mirror; the .us original is cert-blocked, see [[COLLECTION-PLAN]])

The motivation it fixes, as the excerpt frames it: conventional layering quietly entangles business logic with both the UI above and the database below, so you cannot test the core without a screen and a DB, and you cannot swap either without surgery. Hexagonal kills that asymmetry by treating *both* sides as equal "outside," each reached through a **port**.

### 2.1 Ports and adapters, defined precisely

The excerpt's definitions ([[cockburn-hexagonal]]):

- A **port** is an interface defined by the application, in the application's own language — e.g. `ForPlacingOrders`, `ForStoringOrders`. The excerpt's verbatim phrasing: "All input and output reaches/leaves the application through a port that isolates the application from external tools, technologies and delivery mechanisms."
- An **adapter** translates between a port and a concrete technology: a REST controller, a CLI, a test driver on the inbound side; a Postgres repo, an in-memory fake, a message-bus client on the outbound side.

The crucial property is *who owns the interface*. The application owns the port. The adapter obeys it. That is dependency inversion in one move: the outbound adapter (the database) depends on an interface the core declared, so the arrow points inward even though the data flows outward at runtime.

That last clause is where most people get confused, so it is worth slowing down. There are two different arrows in any architecture, and they point opposite directions on the driven side:

- **Runtime control flow** — *who calls whom at execution time.* On a driven port, the core calls out: the use case invokes `repository.save(lead)`. Flow points outward.
- **Source-code dependency** — *which file names which type at compile time.* The repository interface is declared in the core; the Postgres class in the outer ring implements it. So the outer ring names the inner ring's type. Dependency points inward.

The Dependency Rule governs the *second* arrow, not the first. You are allowed to call outward all day; you are not allowed to *name* outward. Dependency inversion is exactly the trick that lets runtime flow go one way while the compile-time dependency goes the other. If you ever feel the rule is impossible ("but the core has to use the database!"), you have conflated the two arrows — the core uses the *port*, the database implements it, and the SDK import lives only in the adapter file.

### 2.2 Primary vs secondary (driving vs driven)

Cockburn deliberately writes "pretending that all ports are fundamentally similar," but in practice they come in two flavors, and the distinction matters for where you put things ([[cockburn-hexagonal]]):

| | **Primary / driving** | **Secondary / driven** |
|---|---|---|
| Who initiates? | the adapter calls **into** the core | the core calls **out** through the adapter |
| Examples | UI, HTTP client, **test suite**, batch script | database, message broker, email gateway, **the LLM API** |
| Who owns the interface? | the core (the port is the core's API) | the core (the port is the core's requirement) |
| Direction of dependency | inward | inward (via dependency inversion) |

The symmetry is the whole point, and the excerpt states it: "the application can be equally driven by an automated, system-level regression test suite, by a human user, by a remote http application, or by another local application," and on the data side it can be "configured to run decoupled from external databases using an in-memory… database replacement."

**Use the figure now.** Open [`figures/hexagon-dependency-rule.html`](figures/hexagon-dependency-rule.html) and click each adapter to confirm it can be swapped without touching the core; then toggle "show a forbidden outward dependency" to see the one arrow the Dependency Rule forbids and what breaks the moment you draw it. Watching the LLM-API adapter swap with the core unchanged is the fastest way to feel why this pattern is the right bet for a long-lived agent.

### 2.3 Why a hexagon? (a myth worth retiring early)

The shape carries no significance of six. The excerpt is explicit ([[cockburn-hexagonal]]): Cockburn chose a polygon rather than the usual layered rectangle purely to leave visual room to draw several ports and their adapters around the core. **The number of sides is incidental.** If you ever hear someone argue about "the six concerns of hexagonal architecture," they have reverse-engineered meaning into a drawing convenience. The content is "core + ports + adapters + inward dependencies," not "six."

---

## 3. Same Idea, Three Names — and the One Myth This Chapter Kills

This is the doc-vs-reality reconciliation for ch-03. The popular framing treats Hexagonal, Clean, and Onion as three competing architectures you must choose between — a choice that spawns blog wars and "which is better" threads. The primary sources flatly disagree.

> **Myth:** "Hexagonal vs Clean vs Onion is an architecture decision you have to make."
> **Reality:** They are the same idea with different diagrams: a technology-free core, dependencies pointing inward, I/O at the edges via interfaces the core owns.

Both excerpts assert this directly, not by my paraphrase. [[martin-clean-arch]] says: "Clean = [[cockburn-hexagonal]] (Ports & Adapters) = Onion. All three: technology-free core, inward-pointing dependencies, I/O at the edges through inverted interfaces." And [[cockburn-hexagonal]] says: "Hexagonal, Onion, and Clean are the same idea with different diagrams."

The two differ only in *what they emphasize*, and that difference is genuinely useful:

| | Emphasizes | Best for explaining |
|---|---|---|
| **Clean** ([[martin-clean-arch]]) | the **rule**, stated crisply: "source code dependencies can only point inwards" | *why* the structure is correct; what is forbidden |
| **Hexagonal** ([[cockburn-hexagonal]]) | the **testability motivation**, vividly: drive the core with the DB and UI removed | *what you get* for following the rule; how to test |

So the resolution is not "pick one." It is: use the Clean vocabulary to state the constraint you are enforcing, and use the Hexagonal vocabulary to justify it to a teammate who asks "why all the interfaces?" The answer is "so the test suite can drive this with no database." There is no third architecture hiding behind the third name.

This matters for the [[ch-01]] spine because choosing between identical things is a pure waste of the only budget that counts: decisions that are expensive to reverse. Spend that budget on boundaries and contracts, not on diagram brand.

---

## 4. Tactical DDD Inside the Core: The Aggregate Is the Consistency Unit

Hexagonal/Clean tells you the *shape* of the core (technology-free, ports at the edge) but says nothing about how to organize the domain objects *within* it. That is tactical DDD's job, and it is merged into this chapter deliberately: [[cockburn-hexagonal]] and [[martin-clean-arch]] give you the wall; [[ddd-aggregates-tactical]] gives you the rooms.

A caution before the quotes: the tactical-DDD source is **book-thesis-extracted, not verbatim**. The excerpt header is explicit — Evans's *Domain-Driven Design* (2003) and Vernon's *Implementing DDD* / *DDD Distilled* are "books, theses extracted, not quoted verbatim" ([[ddd-aggregates-tactical]]). So everything in this section is an attributed paraphrase of the authors' position, never presented inside quotation marks as if it were a fetched line.

### 4.1 The vocabulary

As [[ddd-aggregates-tactical]] records the four building blocks:

- **Entity** — identity persists over time even as attributes change. An `Order` is the same order tomorrow even after every field on it changes.
- **Value object** — no identity; defined wholly by its attributes, immutable. `Money(amount, currency)` — two instances with the same fields are interchangeable.
- **Aggregate** — a cluster of entities and value objects treated as one unit for data changes, fronted by a single **aggregate root** that is the only external entry point.
- **Domain event** — a record that something meaningful happened in the domain (`OrderPlaced`), used to propagate change across aggregate and context boundaries.

The entity/value-object split is not pedantry; it changes how you store and compare things. An entity needs an identity column and a notion of "same one over time"; a value object needs neither and is safest stored inline and replaced wholesale rather than mutated. In Lina TMR's domain, a `Lead` is an entity (the same lead persists across stage changes), while the `FxRate` or the `AccountTier` it pulls from a SaaS sheet is a value object — defined entirely by its fields, immutable, and replaced rather than edited. Getting this wrong (treating a value object as an entity) is how you end up with mutable shared state that should have been a fresh copy.

The aggregate is the load-bearing concept. The excerpt's Core Insight states the thesis (paraphrased, per the book-extraction caveat): the aggregate is the unit of *transactional consistency*; its boundary decides what you can change atomically — and therefore where you must fall back to eventual consistency between units.

### 4.2 Vernon's four aggregate rules

These four rules are the actionable core, attributed to Vernon and recorded in [[ddd-aggregates-tactical]] as the "load-bearing claim." Presented as Vernon's position (paraphrase, not verbatim):

| # | Rule | What it buys you |
|---|------|------------------|
| 1 | **Model true invariants in consistency boundaries.** The aggregate = the set of objects that must be consistent *together, immediately*. | A crisp definition of "atomic": everything inside is one transaction; everything outside can be eventually consistent. |
| 2 | **Design small aggregates.** Prefer many small over few large. | Large aggregates create contention and slow loads; small ones reduce lock scope and concurrency conflicts. |
| 3 | **Reference other aggregates by identity.** Hold an `OrderId`, not an `Order` object. | Keeps the boundary crisp; prevents accidental large object graphs being loaded and mutated together. |
| 4 | **Use eventual consistency outside the boundary.** One transaction = one aggregate; when another aggregate must react, publish a domain event and let it update in a separate transaction. | This is the seam to everything in the consistency phase. |

### 4.3 Rule 4 is the in-process seed of the distributed saga

This is the single most important connection in the chapter and the reason the merge was made. The excerpt states it directly ([[ddd-aggregates-tactical]]): rule 4 is "the in-process seed of the distributed [[richardson-saga]]: a saga is 'one transaction per aggregate, coordinated by events' stretched across services."

Read that carefully. A saga (ch-06) is usually introduced as a *distributed-systems* pattern — what you reach for once you have multiple services and can no longer hold a transaction across them. The DDD claim is that the same shape exists *inside a single process*, right now, the moment you have two aggregates that cannot be updated in one transaction. "One local transaction per aggregate, glued by domain events, with compensating logic when something downstream fails" is a saga whether the aggregates live in the same module or across a network.

Make the seed concrete. Two aggregates, `Lead` and `Booking`, that rule 4 says cannot share a transaction. The in-process version reads:

```
# one local transaction: mutate exactly one aggregate
def qualify_lead(lead_id):
    lead = leads.load(lead_id)          # load one aggregate
    event = lead.qualify()              # mutate it; it emits LeadQualified
    leads.save(lead)                    # commit transaction #1
    publish(event)                      # hand off across the boundary

# a separate handler, a separate transaction, reacts to the event
def on_lead_qualified(event):
    booking = scheduling.create_for(event.lead_id)   # transaction #2
    bookings.save(booking)
    # if THIS fails, compensate: emit a LeadQualificationReverted event
```

Nothing here mentions a network. Yet the structure — local transaction, domain event, separately-committed reaction, compensation on failure — *is* a saga. The only thing a service split changes is that `publish(event)` crosses a wire and `on_lead_qualified` runs in another process. The consistency reasoning (what is atomic, what is eventual, what compensates) is identical and was decided here, in cheap-to-change in-process code.

The payoff is concrete and matches the spine: if you internalize correct aggregate boundaries inside a modular monolith, then a later service split inherits correct consistency boundaries **for free**. The expensive-to-reverse work (deciding what must be atomic together) is done once, at design time, in cheap-to-change in-process code — long before you pay the [[richards-ford-fundamentals]] microservice premium. Get the aggregate wrong — too big — and you have baked a contention point into the architecture that is expensive to reverse, exactly the kind of deep binding decision [[ch-01]] warned about. This is also why ch-04 treats aggregate boundaries as the *candidate* deployment seams: a boundary that is already a correct in-process consistency unit is the only kind that is safe to promote across a wire.

---

## 5. Pricing the Bet: Indirection Is Never Free, and Sometimes It's the Only Right Move

The trade-off spine demands that every pattern be priced as a bet — what it keeps cheap to change, and what it makes expensive. Hexagonal/Clean + aggregates is no exception, and the excerpt is unusually candid about the downside ([[martin-clean-arch]]):

> The cost is indirection and boilerplate (DTOs, ports, mappers). For a small CRUD service it's overkill.

Let me make the bet fully explicit, both columns.

### 5.1 What it keeps cheap to change

| Change you might face | Why it stays cheap |
|---|---|
| Swap the database (Postgres → DynamoDB) | New secondary adapter implementing the existing `ForStoringState` port; core untouched. |
| Swap the LLM vendor / add a local model | New secondary adapter behind `ForGeneratingReplies`; domain policy never imported the SDK. |
| Add a new way to drive the system (CLI, webhook, cron) | New primary adapter calling the existing inbound port; no core change. |
| Test domain policy without infrastructure | The test suite *is* a primary adapter; in-memory fakes stand in for driven adapters. |

### 5.2 What it makes expensive

| Cost you take on | When it bites |
|---|---|
| Mappers and DTOs at every boundary | Immediately, for every endpoint — pure ceremony on trivial CRUD. |
| Indirection: a call now hops through a port and an adapter | Reading the code requires following an interface to its implementation. |
| Discipline to keep it honest | One leaked ORM row or framework object silently reverses an inward arrow; without enforcement (a fitness function — ch-09), the rule erodes. |
| Up-front modeling cost for aggregates | Drawing the wrong consistency boundary is itself expensive to reverse. |

### 5.3 The decision criterion

The deciding question is not "is this clean?" but "**how long-lived is this core relative to its vendors?**" The excerpt's verdict ([[martin-clean-arch]]):

> For the learner's long-lived sales-agent core — where business policy must outlive whichever LLM API/vector DB/web framework is current — it's exactly the right bet: it keeps the *expensive-to-reverse* part (domain policy) insulated from the *cheap-to-swap* parts (vendors).

A throwaway script: skip it; the indirection costs more than the swaps you will never make. A long-lived domain core in a fast-moving vendor landscape: pay it; you *will* swap vendors, and you want each swap to be a new adapter, not a core rewrite. This is the [[insights]] formulation — "what does this keep cheap to change, and what does it make expensive?" — answered for this pattern.

---

## 6. Applied to the Sales Agent (Lina TMR)

The learner's production sales agent, Lina TMR, is an LLM agent acting over many external SaaS tool APIs. It is precisely the case the excerpt singles out as the right bet for this pattern, so the application is not a stretch — it is the worked example the source had in mind.

### 6.1 The core is the domain policy, not the model

The expensive-to-reverse part of Lina TMR is its **domain policy**: how a lead is qualified, when a deal advances a stage, when to escalate to a human, what a "stale" pipeline entry is. That policy should live in entities and use cases at the center and should not `import` the model SDK. The model is a **secondary / driven adapter** behind a port the core owns — call it `ForGeneratingReplies`. When the model market shifts (a cheaper model, a better one, a local fallback), you write a new adapter; the policy is untouched. The figure's LLM-API swap demonstrates exactly this move: click it, swap it, watch the core stay put.

Contrast the alternative — domain logic that calls the vendor SDK inline. Then "switch models" is a refactor that touches every file where a decision was made, and you cannot unit-test "when does Lina escalate?" without a live API key. That is the asymmetry [[cockburn-hexagonal]] exists to kill.

### 6.2 Every SaaS API is a driven adapter behind a port

Salesforce, Gmail, Google Sheets, the calendar — each is a secondary adapter implementing a port the core declares (`ForCallingTools`, or one port per capability). Two consequences:

1. **Their responses enter as plain DTOs, not live SDK objects.** Per §1.2, mapping the SDK response to a domain DTO at the adapter is what stops Salesforce's object model from leaking into the agent's core. This is also the natural home for the inside-vs-outside-data normalization that [[ch-01]] flagged and ch-06 makes central: an external SaaS response is *outside data* — a versioned, possibly-stale snapshot — and the adapter is where it gets stamped and frozen before the core ever sees it.
2. **Resilience has a place to live.** A driven adapter is where a timeout, a circuit breaker, and a bulkhead go (ch-08), so one slow vendor cannot stall the agent loop. The port makes "this dependency can fail" an explicit, wrappable seam rather than a buried inline call.

### 6.3 Aggregates inside the agent

Apply Vernon's rules to Lina TMR's domain. A reasonable aggregate is the `Lead` (or `PipelineEntry`): the cluster of entities and value objects that must stay consistent the instant the agent advances a stage or logs a qualification. Rule 2 says keep it small — do not fold the entire conversation history and every CRM-sync record into one giant `Lead` aggregate, or every update contends. Rule 3 says a `Lead` references the `Conversation` by identity, not by holding the whole conversation object. Rule 4 says when advancing a lead must trigger a calendar booking, that is a *separate* transaction reached by a domain event (`LeadQualified`), not a nested write — which is the in-process saga seed that, if Lina TMR ever splits scheduling into its own service, becomes a real distributed saga with the consistency boundary already correct.

This is the through-line stated in [[insights]]: default to a modular monolith with clean bounded contexts, get the in-process consistency boundaries right first, and only extract a service when the granularity disintegrators clearly win — a decision ch-04 makes operational.

---

## Where This Goes

This chapter structured the inside of a single boundary: a technology-free core, ports the core owns, adapters at the edges, inward-only dependencies, and small aggregates as the in-process consistency unit. Every one of those choices assumed the boundary itself stayed put — that the whole thing lives in one deployable.

Ch-04 challenges that assumption directly. It asks where the *runtime* boundaries should fall: monolith, modular monolith, or microservices — and prices that as the most expensive-to-reverse decision in the course. The aggregate boundaries you drew here become the candidate seams; the question ch-04 answers is which (if any) of them is worth promoting from an in-process module boundary to a deployment boundary, using the architecture quantum and granularity disintegrators-vs-integrators rather than a rule of thumb. The in-process saga seed from §4.3 is the bridge: cross a deployment boundary and that seed grows into the real distributed saga of the consistency phase.
