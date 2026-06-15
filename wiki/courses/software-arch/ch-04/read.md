<!-- chapter: ch-04
     track: topology
     kind: content
     title: Monolith, Modular Monolith, Microservices, and the Quantum
     deps: [[ch-03]]
     sources: [[fowler-microservices]], [[newman-building-microservices]], [[fowler-monolith-first]], [[decompose-by-business-capability]], [[conway-team-topologies]], [[richards-ford-hard-parts]], [[distributed-monolith]]
-->

# Chapter 04 — Monolith, Modular Monolith, Microservices, and the Quantum

> **Core insight.** Topology is the most expensive-to-reverse decision in the whole course, which is precisely why it comes *fourth*, not first: you only earn the right to draw deployment boundaries after you have the language boundaries ([[ch-02]]) and the inside-a-boundary structure ([[ch-03]]). Microservices are defined by *independent deployability*, not by size — and that property is bought, not given. You pay distribution, eventual consistency, and operational overhead to get it, and you only come out ahead above a complexity threshold the field calls the MicroservicePremium. The right default is the **modular monolith**: a single deployable unit with internally strict module boundaries, because inside a monolith a wrong boundary is a *refactor* and across services the same wrong boundary is a *migration*. Service size is the **output** of a trade-off analysis (granularity disintegrators vs integrators), never a rule of thumb — and the catastrophic failure mode, the **distributed monolith**, is what you get when you pay the distribution tax but keep the coupling.

> **Guideline.** Do not pick a topology; *derive* one. Start every new bounded context inside a modular monolith with clean internal seams. Before extracting any seam into its own service, run the architecture-quantum test (is this an independently deployable, cohesive unit *with its own data*?) and explicitly enumerate the disintegrators pushing it apart against the integrators holding it together — extract only when the disintegrators clearly win. Use Conway's Law as a lever (shape teams to induce the architecture you want), not as folklore to ignore. The litmus test for whether you actually achieved microservices is brutally simple: can you deploy service A without deploying service B? If not, you have a monolith with network latency.

---

## 1. The Decision That Comes Fourth on Purpose

Three chapters of this course have been about boundaries you can move cheaply. [[ch-02]] cut on *language* (bounded contexts), [[ch-03]] cut *inside* a boundary (hexagonal/clean core + aggregates). Both are reversible: a context that turns out wrong is a re-modeling exercise; a leaky port is a refactor. This chapter is the first one whose central decision is genuinely expensive to undo — and the course has been deliberately deferring it.

That deferral is the whole pedagogy. The course spine, stated in [[insights]], is that *architecture is the set of decisions that are expensive to reverse*. The most expensive-to-reverse decision in an application's life is where you cut a network boundary, because the cost of getting it wrong is not a code change but a coordinated, multi-team, data-migrating, contract-breaking re-deployment. So the entire boundaries phase ([[ch-02]]..[[ch-03]]) exists to let you get the *cheap* boundaries right first, defer the expensive one, and arrive here with a model good enough to bet on.

### 1.1 What this chapter is, and is not

This is a *topology* chapter, not a distributed-systems-internals chapter. We are choosing among three shapes — monolith, modular monolith, microservices — and learning the one test (the architecture quantum) and the one force-analysis (disintegrators vs integrators) that tells you which shape a given seam wants. How a saga preserves consistency once you *have* split, how the outbox guarantees a message is sent — those are [[ch-06]]. Here we decide *whether* and *where* to split at all.

---

## 2. Microservices Defined by Deployability, Not Size

The single most consequential correction this chapter makes is to the definition itself. The popular mental model is "microservices = small services." The primary sources agree on something else: the defining property is **independent deployability**.

### 2.1 The two definitions, side by side

Lewis & Fowler's canonical statement, quoted verbatim in [[fowler-microservices]]:

> "The microservice architectural style is an approach to developing a single application as a suite of small services, each running in its own process and communicating with lightweight mechanisms, often an HTTP resource API." — James Lewis & Martin Fowler, *Microservices* (martinfowler.com/articles/microservices.html, 2014)

The contrast they draw with a monolith — also from [[fowler-microservices]] — locates the real differentiator: a monolith is "a single unit" where "any changes to the system involve building and deploying a new version of the server-side application." The word *size* in the definition above is almost a distraction. The thing that makes a service a microservice is that it ships on its own clock.

Newman states this even more sharply. From [[newman-building-microservices]] (a book; thesis extracted, corroborated by O'Reilly excerpts and his talks — *not* a fetched URL), his definition is "independently releasable services modeled around a business domain," and the load-bearing paraphrase is:

> "Independent deployability is the single most important principle." — Sam Newman, paraphrased from *Building Microservices* 2e

Note the attribution discipline here: that line is rendered as an attributed *paraphrase* of a book thesis, not inside clean quotation marks as if it were a verbatim fetch, because the excerpt marks it that way. Newman's point is causal, not cosmetic: independent deployability is the *cause* of the good properties, not a side effect. Chasing it "forces loose coupling, well-defined contracts, and stable interfaces" ([[newman-building-microservices]]). You do not get loose coupling and then deployability; you commit to deployability and it *forces* the coupling discipline.

### 2.2 The two disciplines that buy deployability

[[newman-building-microservices]] names the two non-negotiables that make independent deployment real:

| Discipline | What it means | What breaks if you skip it |
|---|---|---|
| **Information hiding** | A service exposes behavior through APIs while hiding implementation details (database, tech choices, internal workflow) — Parnas's module-level rule lifted to the service level. | If internals leak (a shared table, an exposed schema), consumers couple to them and independent deployment dies. |
| **Data ownership** | "Each microservice must own its data. Shared databases create hidden coupling and destroy independent deployability." (book thesis, [[newman-building-microservices]]) Interaction happens through explicit APIs or events, never another service's database. | A shared DB is an *implicit, unversioned contract*; changing your table silently breaks the other service. |

These two are the same conclusion Richardson reaches from the transaction angle (Database per Service, previewed in [[ch-06]]) — Newman just arrives there from the *deployability* angle.

### 2.3 The nine characteristics — read as a cost ledger

[[fowler-microservices]] lists nine characteristics from Lewis & Fowler. Read at the trade-off altitude, several of them are not features — they are *prerequisites you must pay for*:

1. **Componentization via services** — out-of-process components; the payoff is independent deployment (the point of §2.1).
2. **Organized around business capabilities** — cross-functional teams own a "broad-stack implementation" (UI + storage + collaboration). This is a Conway's-Law consequence (§5).
3. **Products not projects** — "you build it, you run it"; you own the product over its full lifetime. (An *org* commitment, not a code one.)
4. **Smart endpoints and dumb pipes** — logic in services, plumbing kept simple; "as decoupled and as cohesive as possible."
5. **Decentralized governance** — pick the right tech per service ("Node.js for a reports page… C++ for a gnarly near-real-time component? Fine").
6. **Decentralized data management** — each service owns its database; polyglot persistence. This is the seed of every consistency cost in [[ch-06]].
7. **Infrastructure automation** — CI/CD, automated test and deploy are *prerequisites, not nice-to-haves*. This is a tax line, not a benefit.
8. **Design for failure** — "any service call could fail due to unavailability"; you must tolerate it (Netflix Simian Army). Another tax line — and the entire reason [[ch-08]] exists.
9. **Evolutionary design** — services are "replaceable rather than evolved"; refactor boundaries as understanding grows.

Characteristics 6, 7, and 8 are why microservices have a *premium*: you do not get to skip per-service data, automation, or failure tolerance. They are the entry fee.

---

## 3. Pricing the Bet: The MicroservicePremium

This is the chapter's central trade-off, so price it explicitly. From [[fowler-microservices]], the "Microservice Trade-Offs" article gives a benefit/cost ledger in Fowler's own words:

| Benefit | Fowler's words | The cost you pay for it | Fowler's words |
|---|---|---|---|
| Strong module boundaries | "reinforce modular structure… important for larger teams" | Distribution | "remote calls are slow and… always at risk of failure" |
| Independent deployment | "autonomous… less likely to cause system failures" | Eventual consistency | "everyone has to manage eventual consistency" |
| Technology diversity | "mix multiple languages… data-storage technologies" | Operational complexity | "need a mature operations team" |

The summarizing claim, verbatim from [[fowler-microservices]]:

> "There is a Microservice Premium: microservices impose a cost on productivity that can only be made up for in more complex systems." — Martin Fowler

And the eventual-consistency cost is specifically *application-level*, which is the part this design course cares about — also verbatim from [[fowler-microservices]]:

> "Business logic can end up making decisions on inconsistent information." — Martin Fowler

That single sentence is the bridge to [[ch-06]]: eventual consistency is not "an infra problem the platform team handles." It is a *design* problem that lands in your domain logic. Crossing a boundary means your code may now act on stale data, and you must design for that.

The authors' own hedge is itself the thesis. From [[fowler-microservices]]:

> "We write this with cautious optimism." — Lewis & Fowler

They explicitly decline to call microservices superior. A trade is not an upgrade.

### 3.1 The bet, stated as keep-cheap / make-expensive

To honor the trade-off spine, here is the microservices bet in the course's fixed form:

- **Keeps cheap to change:** deploying, scaling, and re-teching one capability *in isolation*; one team can ship without coordinating a global release. The blast radius of a change shrinks to one service.
- **Makes expensive:** anything that *crosses* a service boundary — a transaction (you lose ACID, [[ch-06]]), a synchronous call (you inherit latency + partial failure, [[ch-08]]), a schema change to a published contract (coordinated multi-team migration, [[ch-05]]). And it makes *every* change pay a standing operational tax (CI/CD, observability, failure tolerance) whether or not that change benefits from distribution.

You take this bet only when your *expected* changes are dominated by the first list and rarely touch the second.

---

## 4. MonolithFirst and the Modular-Monolith Default

Here is the chapter's most important doc-vs-reality correction.

### 4.1 The myth, named

> **Myth (from the reconciliation table in [[COLLECTION-PLAN]]):** "Microservices are the modern best practice; monoliths are legacy."

This is false, and the primary source is emphatic. From [[fowler-monolith-first]], verbatim:

> "Almost all the successful microservice stories have started with a monolith that got too big and was broken up." — Martin Fowler

> "Almost all the cases where I've heard of a system that was built as a microservice system from scratch, it has ended up in serious trouble." — Martin Fowler

The reasoning is the course thesis restated. Boundaries are the hard part, and they are hardest *early* — from [[fowler-monolith-first]]:

> "Even experienced architects working in familiar domains have great difficulty getting boundaries right at the beginning." — Martin Fowler

And once you have services, a wrong boundary is expensive: refactoring functionality between services is "much harder than it is in a monolith" ([[fowler-monolith-first]]). This is the single most important asymmetry in the chapter:

> **Inside a monolith a bad boundary is a refactor. Across services the same bad boundary is a migration.**

The explicit recommendation, verbatim from [[fowler-monolith-first]]:

> "Start a new application as a monolith initially, even if you think it's likely that it will benefit from a microservices architecture later on." — Martin Fowler

### 4.2 The modular monolith — the strictly-dominant default-when-unsure

A **modular monolith** ([[fowler-monolith-first]]) keeps the single deployable unit but enforces strict internal module boundaries: one schema/namespace per module, communication via in-process interfaces, no module reaching into another module's tables. It captures most of microservices' *organizational* benefit (clear ownership, enforced cohesion) with none of the *distribution* tax. And it is the ideal staging ground — once the module seams are clean, extraction to a real service becomes mechanical rather than archaeological.

The bet, in the course's form:

- **Keeps cheap to change:** the *boundary itself* (it's an in-process refactor); plus you pay zero distribution tax, zero eventual-consistency tax, zero standing ops tax for cross-service traffic.
- **Makes expensive:** independent deployment and independent scaling of a single module — you cannot scale just the conversation engine; you redeploy the whole unit. You also rely on *discipline* (not the network) to enforce the seams, and discipline rots without enforcement — which is exactly the job of fitness functions in [[ch-09]].

Compared against a distributed monolith (§7), the modular monolith *strictly dominates* when you are unsure: same coupling discipline required, but you avoid paying the distribution tax for nothing.

---

## 5. Where to Cut: Business Capability, Subdomain, and Conway

If you *do* split, the seam matters more than the count. Two of this chapter's concepts answer "where."

### 5.1 Cut on the slowest-changing structure

From [[decompose-by-business-capability]] (Chris Richardson, microservices.io), verbatim:

> "A business capability is a concept from business architecture modeling. It is something that a business does in order to generate value." — Chris Richardson

> "Define services corresponding to business capabilities." — Chris Richardson

And the reason this is the *right* seam, verbatim:

> "Stable architecture since the business capabilities are relatively stable." — Chris Richardson

This is the decomposition restatement of the course thesis. You commit your expensive-to-reverse boundary to the part of the system *least* likely to change. Orgs churn, tech churns, screens churn — what the business fundamentally *does* changes slowly. Cut there.

The capability path (outside-in, from business architecture) and the DDD-subdomain path (inside-out, from the domain model and its [[ch-02]] bounded contexts) converge in practice: a well-found bounded context usually *is* a business capability ([[decompose-by-business-capability]]).

### 5.2 The anti-decomposition: never split by technical layer

The classic mistake, named in [[decompose-by-business-capability]]: split by technical layer — "a UI service, a logic service, a data service." Then *every* business change touches every service, restoring full coupling over the network. That is the direct on-ramp to a distributed monolith (§7).

### 5.3 Conway's Law: the boundary is socio-technical

> **Myth (from [[COLLECTION-PLAN]]):** "Conway's Law is folklore."

It is not folklore; it is the thesis of the original 1968 paper. From [[conway-team-topologies]], verbatim:

> "Any organization that designs a system (defined broadly) will produce a design whose structure is a copy of the organization's communication structure." — Melvin Conway, 1968

Fowler's mechanism (in [[conway-team-topologies]]): "software coupling is enabled and encouraged by human communication," with the corollary that "the modular decomposition of a system and the decomposition of the development organization must be done together."

There are three responses ([[conway-team-topologies]]): **ignore** it (it happens anyway, by accident), **accept** it (align architecture to the communication paths you have), or the **Inverse Conway Maneuver** — deliberately restructure teams to *induce* the architecture you want, "particularly effective with microservices organized around business capabilities." The maneuver is the lever: org structure is not a constraint you suffer, it is a knob you can turn.

### 5.4 Team Topologies: cognitive load is the real constraint

Skelton & Pais (*Team Topologies*, 2019 — book, framework extracted; teamtopologies.com corroborates) supply the operational refinement in [[conway-team-topologies]]. Four team types — **stream-aligned** (the default; owns one valuable flow), **enabling** (temporarily upskills others), **complicated-subsystem** (owns deep specialist complexity), **platform** (internal self-service to "reduce cognitive load") — and three interaction modes (**collaboration**, **X-as-a-Service**, **facilitating**).

The load-bearing idea is **cognitive load**, not headcount ([[conway-team-topologies]]): draw a team boundary so the team can hold its whole responsibility in its head. A service boundary that splits a team's cognitive unit, or fuses two teams' units, "will be fought daily" — *regardless* of how clean the boundary looks on paper. This is why decomposition is a socio-technical trade-off, not a purely technical one.

---

## 6. The Architecture Quantum and the Granularity Forces

Now the test and the force-analysis — both from [[richards-ford-hard-parts]] (*Software Architecture: The Hard Parts*, 2021; book, thesis extracted, O'Reilly/Amazon descriptions corroborate).

### 6.1 The stance: no best practices

The book's entire premise, in its verbatim sense from [[richards-ford-hard-parts]], is that the hard parts of distributed architecture are "difficult problems… with no best practices that force you to choose among various compromises." It teaches "how to think critically about the trade-offs involved with distributed architectures" and gives "techniques to help you discover and weigh the trade-offs." This is the operationalization of the First Law ([[ch-01]]): if you cannot name the cost, you have not understood the pattern.

### 6.2 The architecture quantum — the boundary test

An **architecture quantum** ([[richards-ford-hard-parts]]) is the smallest unit that is **independently deployable**, has **high functional cohesion**, has synchronous connascence *within* it but not across it, and — crucially — includes its **own data**.

The "own data" clause is the whole power of the concept. Two "services" that share a database cannot deploy independently, so they are *one* quantum, not two. The quantum is therefore the formal distributed-monolith detector: **count your quanta, not your services.** If you drew three service boxes but they share one database, you have one quantum and three deployment headaches.

### 6.3 Granularity: disintegrators vs integrators

> **Myth (from [[COLLECTION-PLAN]]):** "Pick service size by a rule of thumb (e.g. fits in two pizzas / N lines of code)."

The primary source's correction, from [[richards-ford-hard-parts]]: do not argue about size, *enumerate forces*. Size is the **output** of the analysis, not an input.

| Granularity **disintegrators** (forces to split *smaller*) | Granularity **integrators** (forces to keep *together*) |
|---|---|
| Divergent **scalability / throughput** — one part needs 10× the capacity of another | A database **transaction** must span the two parts (the strongest integrator) |
| **Fault isolation** — one part failing must not take the other down | Tight **data dependencies** — they constantly read each other's data |
| Differing **security / access** requirements | Heavy **chatty workflow / orchestration** between them (network round-trips would dominate) |
| Distinct **code volatility** — one part changes far more often | **Shared code** that changes together |
| Separate **team ownership** (the Conway link, §5.3) | |

You split a quantum only when the disintegrators clearly outweigh the integrators. This single framework is the cure for *both* failure modes: premature microservices (disintegrators were weak, you split anyway) and the distributed monolith (integrators were strong — especially a shared transaction — and you split anyway).

> **Use the figure now.** Open [`figures/granularity-balance.html`](figures/granularity-balance.html) and toggle the forces on the two-pan scale. Watch the recommendation tip across monolith → modular monolith → microservice as you add disintegrators, and notice the explicit *distributed-monolith warning* that fires the moment you turn on the shared-database toggle while also splitting — that is the §7 trap rendered as a state you can trigger.

---

## 7. The Distributed Monolith: Paying the Tax, Buying Nothing

> **Myth (from [[COLLECTION-PLAN]]):** "Split into microservices and you get loose coupling for free."

This is the chapter's most dangerous myth because it fails *silently and late*. From [[distributed-monolith]], the definition: a set of services "so tightly coupled and interdependent that they behave like a monolithic application, defeating the core benefit of adopting microservices" — "a monolith that just happens to communicate over HTTP instead of function calls."

### 7.1 The four tells

From [[distributed-monolith]]:

1. **Deployment dependencies** — services must be released together. You have lost independent deployability, the *one* property that justified the split.
2. **Synchronous coupling** — a request fans out through a chain of real-time blocking calls instead of an async event. Any link's latency or outage stalls the whole chain.
3. **Shared database** — two services read/write the same schema, an *implicit, unversioned contract*. Changing one service's tables silently breaks the other.
4. **Cascading failures** — tight runtime coupling means one slow dependency drags the whole workflow down (the failure mode [[ch-08]] exists to stop).

### 7.2 Why it is "worst of both worlds"

The verbatim summary from [[distributed-monolith]]:

> "This network-based modularity gives you all the pain of distributed systems without the independence that makes microservices worthwhile."

The three-way comparison the learner should carry around (from [[distributed-monolith]]):

| Topology | Distribution tax | Deploy independence | Verdict |
|---|---|---|---|
| Monolith | **None** | None | Fine until it gets too big / too many teams |
| Well-cut microservices | High | **Bought** | Worth it above the MicroservicePremium |
| **Distributed monolith** | **High** | **None** | Pays the tax, buys nothing — the trap |
| Modular monolith | None | None (but seams are clean) | The strictly-dominant fallback when unsure |

The litmus test from [[distributed-monolith]] is the chapter's one-line summary: **if you cannot deploy service A without also deploying B, you do not have microservices — you have a monolith with network latency.** Fix the coupling before celebrating the decomposition.

---

## 8. Applied to the Sales Agent (Lina TMR)

The learner's production system is Lina TMR: an LLM agent acting over many external SaaS tool APIs (CRM, email, calendar, ticketing). Run this chapter's machinery on it.

### 8.1 Default topology: modular monolith, by the book

From [[ch-02]] the agent's bounded contexts plausibly are *lead/pipeline*, *conversation*, *scheduling*, and *CRM-sync*. The MonolithFirst verdict ([[fowler-monolith-first]]) applies directly: the domain is still being learned (the agent is in production but evolving), so the boundaries between those contexts are not yet trustworthy enough to freeze across a network. Make each context a **module** inside one deployable agent, with clean in-process interfaces and one schema per module. A wrong seam is then a refactor, not a migration.

### 8.2 Run the granularity forces on each seam

Apply §6.3 honestly, seam by seam:

| Seam | Disintegrators present? | Integrators present? | Verdict |
|---|---|---|---|
| **Conversation** engine | Different scalability (LLM-bound, bursty); fault isolation (a stuck conversation must not kill sync jobs); high code volatility (prompts change daily) | Few — it mostly emits events | Strongest *candidate* for first extraction if load demands it |
| **CRM-sync** | Fault isolation from a flaky vendor (Conway: maybe a different team) | Tight data dependency with lead/pipeline; chatty | Keep *with* pipeline unless the vendor flakiness forces fault isolation |
| **Scheduling** | Mild | Transactional ties to conversation outcomes | Keep together — an integrator (shared transaction) is the strongest hold |

The point is not the answer; it is that *size fell out of the force analysis*, exactly as [[richards-ford-hard-parts]] prescribes — not from a "split everything into microservices" reflex.

### 8.3 The trap to avoid, named for this system

The most likely way Lina TMR becomes a distributed monolith ([[distributed-monolith]]): someone "splits" the agent into a *conversation service* and a *CRM service* but lets both read the same Postgres tables (tell #3, shared database) and call each other synchronously inside the agent loop (tell #2). Now a slow CRM vendor stalls every conversation (tell #4, the cascading failure [[ch-08]] addresses), and you cannot ship a prompt change without redeploying CRM-sync (tell #1). You would be paying full distribution tax for zero independence. The architecture-quantum test catches this *on paper*: shared DB ⇒ one quantum ⇒ not actually two services.

### 8.4 Conway, for a small team

If Lina TMR is built by one small team, [[conway-team-topologies]] says the *cognitive-load* constraint, not the technology, sets the boundary. One team cannot independently operate four microservices without its cognitive load exceeding capacity — the architecture would be "fought daily." For one stream-aligned team, the modular monolith is not just the safe default; it is the Conway-correct one.

---

## Where This Goes

This chapter decided *whether* and *where* to cut a deployment boundary, and priced the cut as a bet. The moment you do cut — or even when a modular monolith talks to an external SaaS API — you create a **contract**, and a published contract is the single most expensive-to-reverse artifact a service owns: breaking it costs a coordinated, multi-team migration. [[ch-05]] takes up integration contracts: REST as a set of constraints (and the uncomfortable truth that most "RESTful" APIs are only Level-2 HTTP-RPC), API-style selection as a trade-off, additive versioning and idempotency, and consumer-driven contracts as the one mechanism that *preserves* the independent deployability this chapter taught you to value. Then [[ch-06]] prices what it actually costs to cross the boundary you just drew: the loss of locks and shared transactions, sagas, and the outbox.
