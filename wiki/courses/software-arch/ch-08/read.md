<!-- chapter: ch-08
     track: evolution
     kind: content
     title: Resilience as Architecture — Stability at the Integration Points
     deps: [[ch-06]]
     sources: [[nygard-release-it]], [[distributed-monolith]], [[richardson-saga]], [[richards-ford-fundamentals]]
-->

# Chapter 08 — Resilience as Architecture: Stability at the Integration Points

> **Core insight.** Every place your system reaches across a boundary — a network call, a broker, a shared pool — is a place a failure can *enter* and *spread*. Stability is not something you hope for, it is something you engineer by deciding *where to stop propagation*: bound every remote wait with a timeout, wrap flaky dependencies in a circuit breaker, partition resources into bulkheads, and fail fast. These are not operational afterthoughts bolted on at deploy time — *where you place a breaker or a bulkhead defines your system's blast radius*, and blast radius is an architectural property. A [[distributed-monolith]] is precisely the system that skipped these decisions: synchronous chains with no breakers, so one vendor outage cascades through everything.

> **Guideline.** Treat every integration point as guilty until proven safe. Never make a remote call without a **timeout**; once a dependency starts failing, **open a circuit breaker** and fail fast instead of piling requests on a corpse; **bulkhead** your pools so one drowning dependency can't drain the whole boat; and record *why* you placed each one where you did in an **ADR**, because the "why" is the first thing lost. Price each one as a bet: you spend a little capability (some false rejections, some partitioned-away throughput) to buy back the property that actually matters under load — *bounded* failure instead of *total* failure.

---

## 1. The Spine, Re-Applied: Resilience Is a Decision About Reversibility

The course's organizing claim, installed in ch-01, is that architecture is the set of decisions expensive to reverse, and the First Law tells you how to judge each one:

> "Everything in software architecture is a trade-off. If you think you've found something that isn't a trade-off, you likely just haven't found the trade-off yet." — Richards & Ford, *Fundamentals of Software Architecture* (book; thesis extracted, quoted as commonly published) — [[richards-ford-fundamentals]]

Resilience patterns are where this law bites hardest, because the naïve reading is "more resilience is always better — add timeouts everywhere, breakers everywhere." That is the First-Law trap. A timeout that fires too early turns a slow-but-correct call into a spurious failure. A breaker tuned too sensitive locks out a dependency that was only briefly twitchy. A bulkhead that partitions too finely strands capacity that a hot path needed. Each pattern is a *priced bet*, and the price is real.

The reason this chapter sits in the **evolution** phase rather than alongside the consistency mechanics of ch-06 is the through-line stated in [[richards-ford-fundamentals]]: evolution is about *keeping the bet revisable*. Stability patterns are how you keep an outage from being a one-way door. Without them, a single dependency failure can corrupt state, exhaust pools, and take down unrelated capabilities — and recovering from *that* is the most expensive reversal of all. With them, failure is *bounded*: it stays inside the compartment you assigned it, and the rest of the system keeps running while you fix the broken part. That is the same move ch-09 will make at a slower timescale with strangler-fig migrations and fitness functions — bound the blast radius of *change* the way these patterns bound the blast radius of *failure*.

### 1.1 Architecture characteristics pick which failures you must survive

You do not get resilience for free, and you do not need it uniformly. The architect's first task, per [[richards-ford-fundamentals]], is to *derive the critical few* architecture characteristics from the requirements rather than maximizing all of them. **Fault tolerance** and **availability** are the "-ilities" that put resilience patterns on the table; if a subsystem is a nightly batch job with no live caller, an aggressive circuit breaker buys you nothing and costs you complexity. Resilience is a characteristic you spend on *where the requirements demand it* — and the integration points that face an unreliable external world are exactly where they do.

### 1.2 The second Nygard contribution: record the why

[[nygard-release-it]] is two ideas, not one. The first is the stability patterns. The second, easy to forget because it is unglamorous, is the **Architecture Decision Record** — and the two belong together precisely *because resilience decisions are the ones whose rationale rots fastest*. A timeout budget of 2 seconds, a breaker cooldown of 30 seconds, a bulkhead sized to 20 connections: these look like arbitrary magic numbers six months later, and the temptation to "clean them up" or "tune them" without understanding why they were chosen is exactly how a carefully bounded blast radius silently un-bounds itself. Nygard's framing of the problem an ADR solves is the whole reason to pair them:

> "One of the hardest things to track during the life of a project is the motivation behind certain decisions." — Nygard, "Documenting Architecture Decisions" — [[nygard-release-it]]

> "An architecture decision record is a short text file in a format similar to an Alexandrian pattern." — Nygard — [[nygard-release-it]]

The structure — *Title, Status* (proposed/accepted/deprecated/superseded), *Context, Decision, Consequences* — is deliberately small ("one or two pages") and is meant to be "a conversation with a future developer." For a resilience decision the *Consequences* field is where the trade-off this chapter keeps insisting on actually gets written down: "we accept false rejections during the 30s cooldown in exchange for not draining the worker pool when vendor X is down." An undocumented resilience parameter is a bet whose price nobody can see — which is how it gets removed.

---

## 2. Cascading Failure: The Anti-Pattern That Motivates Everything

The reason to engineer stability at all is one failure mode, and it is worth naming precisely before naming any cure. From [[nygard-release-it]] (Michael Nygard, *Release It!* 2e — book, thesis extracted; corroborated via the csabapalfi/release-it notes and the Pragmatic Bookshelf material):

A failure at one integration point propagates "from subsystem to subsystem crashing each one." That is a **cascading failure**: a localized fault that does not stay local because the systems around it are coupled tightly enough to drag each other down. Nygard's blunt summary of the most common cause:

> "Integration Points without Timeouts is a surefire way to create Cascading Failures." — Nygard, *Release It!* (book; thesis extracted) — [[nygard-release-it]]

The mechanism is mundane and that is what makes it dangerous. Dependency D slows down — not crashes, just slows. Every caller of D now holds a thread (or a connection, or an event-loop slot) waiting for a response that is taking 30 seconds instead of 30 milliseconds. Those held resources are *finite*. The pool drains. New requests — even ones that have nothing to do with D — now block waiting for a free thread. The caller of the caller sees *its* dependency (you) slow down, and the same drainage happens one level up. A single slow vendor has now stalled an entire service graph, and nobody's database even went down. **An unbounded wait is a held resource, and held resources are how one slow dependency hangs the whole system.**

The trap has a second-order accelerant that every architect must price in: the *retry storm*. The well-meaning instinct when a call fails is to retry it. But if D is slow because it is *overloaded*, every retry adds load to an already-drowning dependency, pushing it further under and turning a recoverable brownout into a hard outage. Retries without a circuit breaker and without backoff are not resilience — they are an amplifier wired directly into the failure. This is exactly why [[nygard-release-it]] pairs the timeout with the breaker: the timeout bounds one wait, but only the breaker *stops generating load on a dependency you already know is failing*. The pattern that looks like it helps (retry harder) is the one that turns a cascade into a collapse, which is the First Law showing up inside the resilience toolkit itself — even "retry" has a cost, and the cost is the storm.

### 2.1 The distributed monolith is the cascading-failure machine

This is the exact failure that ties this chapter back to topology (ch-04). [[distributed-monolith]] lists four tells of the anti-pattern; the fourth is cascading failure itself:

> "Cascading failures — tight runtime coupling means one slow dependency drags down the entire workflow (the failure mode [[nygard-release-it]] exists to stop)." — [[distributed-monolith]] (community synthesis; not a single canonical author article)

And the second tell, *synchronous coupling*, is the structural precondition:

> "Synchronous coupling — a request fans out through a chain of real-time blocking calls instead of an async event or a message broker. Any link's latency or outage stalls the whole chain." — [[distributed-monolith]]

So the resilience toolkit and the distributed-monolith trap are two sides of one coin. A distributed monolith is, by definition, a system that decomposed into services but skipped the stability patterns: it kept synchronous blocking chains and added the network as a new failure source. [[distributed-monolith]]'s own trade-off framing makes the cost explicit — it is "all the pain of distributed systems without the independence that makes microservices worthwhile." The patterns below are the difference between a service graph that *contains* failure and one that *conducts* it.

---

## 3. The Stability Toolkit — Each Pattern Priced as a Bet

[[nygard-release-it]] gives a toolkit. The discipline this course demands is that none of them is a "best practice" — each is a trade you make when a specific force is present. Here is the toolkit, each entry with what it *keeps cheap to change* and what it *makes expensive*.

| Pattern | What it does (Nygard) | Keeps cheap | Makes expensive / costs |
|---|---|---|---|
| **Timeout** | "bound every remote wait" | caller's liveness — you never hold a thread forever | tuning: too short = false failures on slow-but-correct calls |
| **Circuit Breaker** | open and fail fast over a dead dependency; half-open to test recovery | caller latency & pool health during an outage | false rejections while OPEN even if the dep already recovered |
| **Bulkhead** | "partition resources… so a failure in one area can't drain the whole" | isolation — one drowning dependency can't sink unrelated work | utilization: stranded capacity in idle partitions |
| **Steady State** | every accumulation has a matching cleanup | unattended uptime — no human babysitting | up-front design of cleanup for every log/session/cache |
| **Fail Fast** | detect you can't succeed and return immediately | blast radius — "only the subsystem where the error occurred is affected" | gives up optimistic retries that might have worked |

### 3.1 Timeout — the non-negotiable floor

A timeout is the cheapest and most universal of the patterns, and the one with the least excuse for omission. Nygard's rule is unconditional: *bound every remote wait*. The bet is almost free — you give up the vanishing tail of "calls that would have succeeded if I'd just waited a bit longer" in exchange for guaranteeing that no single slow dependency can pin your resources indefinitely. The only real cost is tuning: a timeout shorter than the dependency's legitimate p99 will convert healthy-but-slow calls into spurious failures, which (combined with retries) can *itself* become a load-amplifying cascade. The architectural decision is not *whether* to time out but *what the budget is*, and that budget should be derived from the latency characteristics you committed to.

### 3.2 Circuit Breaker — fail fast over a corpse

The timeout bounds a *single* call. But a stream of calls to a dead dependency, each one dutifully waiting out its full timeout before failing, is still a slow-motion disaster: you are spending your entire timeout budget, per request, on a dependency you already know is down. The circuit breaker is the pattern that *remembers*. From [[nygard-release-it]]:

> "Circuit Breaker — track failures to a dependency; once over threshold, *open* the circuit and fail fast (skip the call) instead of piling up requests on a dead service. Periodically *half-open* to test recovery." — [[nygard-release-it]] (Nygard popularized this pattern)

The state machine is the whole pattern, and it is genuinely worth *operating* rather than reading. Open the companion below, fail the dependency, and send a burst of requests: watch the breaker trip to OPEN and start rejecting in ~0ms (the calls never touch the dead vendor), then heal the dependency and let the cooldown lapse to HALF-OPEN, where a single probe decides whether to close.

> **▶ Interactive:** [`figures/circuit-breaker.html`](figures/circuit-breaker.html) — click *Send request* / *Send 5 requests*, toggle the dependency *down* and *heal* it, and tune the trip threshold and cooldown. Watch the "Calls fast-failed (OPEN)" and "Latency last call" counters: while OPEN, latency is ~0ms because the breaker never reaches the corpse.

The three states price the bet exactly:

- **CLOSED** — calls pass through; the breaker counts failures. Cost: nothing yet; you pay full timeout on each failing call until the threshold trips.
- **OPEN** — fail fast, skip the call. *This is where the breaker earns its keep*: caller latency drops to ~0 and pools stop draining. Cost: **false rejections** — if the dependency recovers one second into a thirty-second cooldown, you reject perfectly serviceable requests for the rest of that window.
- **HALF-OPEN** — let exactly one probe through. If it succeeds, close; if it fails, re-open and restart the cooldown. Cost: the recovery is gated on a single probe's luck, so a dependency that recovers *intermittently* can flap.

The architectural decision is *where* the breaker lives (per-dependency, ideally — one breaker per external vendor, not one global breaker) and how the threshold and cooldown trade false-rejection against responsiveness. Those parameters are architecture characteristics in numeric form.

Two placement mistakes are worth naming because they quietly defeat the pattern. First, a **single global breaker** in front of many dependencies: when it trips, it fails fast for *everything*, so one sick vendor takes out calls to the eight healthy ones — you have re-created the cascade you were trying to prevent, just with the breaker as the conductor. The breaker must be scoped to the *unit of failure*, which is the individual dependency. Second, a **breaker with no timeout underneath it**: the breaker only counts failures, and "the call is hanging" only becomes a failure once *something* gives up — which is the timeout. A breaker on top of an unbounded wait never trips because the calls never finish failing; they just hang. The timeout and the breaker are a *pair*, not alternatives: the timeout converts a hang into a countable failure, and the breaker uses the count to stop trying. This is the precise sense in which [[nygard-release-it]] calls timeouts and circuit breakers "the two most effective counters" to cascading failure — they only work together.

### 3.3 Bulkhead — the ship's-hull metaphor, taken literally

A breaker protects you from one *named* dependency. A bulkhead protects you from one dependency's failure *spilling into the resources another dependency needs*. The metaphor in [[nygard-release-it]] is exact:

> "Just as a ship's hull is divided into watertight compartments so that a breach in one section does not sink the vessel." — [[nygard-release-it]]

Concretely: give each downstream dependency its *own* connection pool / thread pool / semaphore, sized independently. If vendor A goes slow and saturates *its* pool, callers of vendor B are untouched because B has its own compartment. Without bulkheads, A and B share one pool, and A's slowdown drains the shared pool, starving B even though B is perfectly healthy — the cascade again, now via resource contention rather than direct call chains. The bet: you trade *utilization* (idle capacity stranded in a quiet partition can't be borrowed by a busy one) for *isolation* (a flooded partition can't sink the rest). You pay this price deliberately when the cost of total failure dwarfs the cost of imperfect resource sharing.

### 3.4 Fail Fast and Steady State — the unglamorous two

**Fail Fast** is the principle the circuit breaker operationalizes, stated generally: detect that you cannot succeed and return *immediately*, so the caller can react (degrade, queue, return a friendly error) instead of hanging. Nygard's framing names the payoff directly:

> "the idea is to fail as fast as you can so that only the subsystem where the error occurred is affected." — [[nygard-release-it]]

That clause — *only the subsystem where the error occurred is affected* — is the blast-radius idea in one sentence, and it is why fail-fast is architectural rather than cosmetic.

**Steady State** is the slow-burn counterpart: every accumulation (logs, sessions, caches, temp files) must have a matching cleanup, so the system can run unattended without a human pruning disk or restarting a process at 3am. It is the resilience pattern that defends against *time* rather than against a *dependency*, and it is the one most often skipped because its failure mode is weeks away rather than seconds away. The bet here is inverted relative to the others: there is almost no runtime cost, only a *design-time* cost (you must think through the lifecycle of every resource you accumulate), and the failure of *not* paying it is the most embarrassing kind — a service that was "stable" for a month and then fell over because a table grew without bound.

### 3.5 The rest of the toolkit, and why size matters

[[nygard-release-it]] lists more patterns than the five above — Handshaking, Shed Load, Back Pressure, Governor, Let It Crash — and the family shares one shape: each one is a mechanism for a system to *say no in a controlled way* before an uncontrolled failure says it for you. **Back Pressure** lets a slow consumer signal upstream to slow down rather than silently building an unbounded queue (the Steady-State failure in a different costume). **Shed Load** drops requests at the edge when the system is past capacity, choosing *which* requests to fail rather than failing all of them randomly. **Governor** deliberately slows down automated actions so that an automated mistake (a runaway script, a misbehaving agent loop) cannot do unbounded damage before a human notices. The common trade-off across all of them: you spend *some* successful work — rejected requests, throttled throughput, slowed automation — to buy *bounded, predictable* degradation instead of an unpredictable collapse. None is a default to switch on everywhere; each is the right bet only where the requirements say "controlled partial failure beats uncontrolled total failure," which is precisely the availability/fault-tolerance characteristics doing their job.

---

## 4. The Myth This Chapter Kills: "Resilience Is an Ops Problem"

The popular narrative treats timeouts, breakers, and bulkheads as operational knobs — SRE territory, something you tune in the service mesh or the load balancer after the architecture is "done." The primary source disagrees, and the reconciliation is the spine of this chapter.

| Popular narrative | What the primary source says | Resolved in |
|---|---|---|
| "Resilience / stability is an ops concern you add after the design." | Stability patterns are *architectural*: where you place circuit breakers and bulkheads **defines your system's blast radius**, and blast radius is a structural property decided at design time. A [[distributed-monolith]] is precisely the system that skipped them. | [[nygard-release-it]], [[distributed-monolith]] |

[[nygard-release-it]] states it directly: "These are *architectural* decisions: where you place circuit breakers and bulkheads defines your system's blast radius." You cannot retrofit blast-radius control after you have already wired every service into a synchronous chain that shares a pool — by then the cascade paths are baked into the topology. The decision of *which* integration points get a breaker, *which* resources get their own bulkhead, and *whether* a call is synchronous at all is made when you draw the boundaries, not when you deploy. This is why the C4 *Container* diagram from ch-01 is where a distributed monolith becomes visible: the synchronous arrows between containers, with no breaker on them and a shared data store underneath, *are* the cascade paths drawn out.

The other half of the resolution is that the *cure* is partly a boundary decision, not just a pattern decision — which is the bridge to the next concept.

---

## 5. Async/Event Integration Is Itself a Resilience Decision

The most powerful stability move is not to add a breaker to a synchronous call — it is to make the call *not synchronous in the first place*. [[nygard-release-it]] makes this explicit:

> "Choosing async/event integration (→ [[richardson-saga]]) is itself a resilience decision: it removes the synchronous coupling these patterns otherwise have to defend." — [[nygard-release-it]]

This is where ch-08 reaches back into the consistency chapter it depends on (ch-06). A synchronous request-chain has a *temporal* coupling: A must wait for B which must wait for C, so C's latency is A's latency and C's outage is A's outage. That is the precise coupling the timeout/breaker/bulkhead toolkit exists to *defend* against. An event-driven integration *removes* the coupling instead of defending it: A emits an event and is done; B and C consume it when they can. If C is down, the event waits in the broker; nothing upstream stalls. This is exactly the saga's structure from [[richardson-saga]]:

> "A saga is a sequence of local transactions. Each local transaction updates the database and publishes a message or event to trigger the next local transaction in the saga." — Chris Richardson, microservices.io/patterns/data/saga.html — [[richardson-saga]]

Because each step is a local transaction glued by an event rather than a blocking call, a downstream outage *defers* a step rather than *failing* the whole operation. But — and this is the trade-off, never a free lunch — you have not eliminated the cost, you have *moved* it. The synchronous chain's cost was cascading latency; the saga's cost is **loss of isolation**:

> "Lack of isolation (the 'I' in ACID)… means there's risk that the concurrent execution of multiple sagas and transactions can [cause] data anomalies." — Richardson — [[richardson-saga]]

So the resilience decision and the consistency decision are the *same* decision viewed from two angles. Go synchronous and you keep ACID-like simplicity but must defend every link with timeout + breaker + bulkhead against cascades. Go async/event and you dissolve the cascade coupling but inherit the saga's countermeasure burden — semantic locks, commutative updates, re-reads, by-value tracking (the original 1987 long-lived-transaction idea behind sagas; the source PDF is image-only, so this lineage is thesis-extracted, not quoted verbatim — [[richardson-saga]]). There is no option that buys both for free. The architect's job is to know which bill they are choosing to pay.

### 5.1 Choreography vs orchestration is also a resilience choice

Even *within* the async/event decision, the saga's two coordination styles trade differently under failure. [[richardson-saga]] frames them:

> "Choreography - each local transaction publishes domain events that trigger local transactions in other services." — Richardson — [[richardson-saga]]
> "Orchestration - an orchestrator (object) tells the participants what local transactions to execute." — Richardson — [[richardson-saga]]

The resilience-relevant difference is *where the failure-handling logic lives*. **Choreography** has lower coupling and no single point of failure — but the compensation logic is scattered across services reacting to events, so when something goes wrong the recovery path is hard to *see*, which is itself an availability risk (you cannot fix fast what you cannot trace). **Orchestration** concentrates the saga's control flow — and therefore its timeout, retry, and compensation handling — in one place you can reason about and observe, at the cost of making that orchestrator a component whose own failure stalls the workflow (so it, too, needs its own resilience and its own bulkhead). For a long, multi-step agent workflow the orchestration style usually wins on *operability* precisely because the failure-handling is centralized and observable — which loops directly into §6.

---

## 6. Observability Is an Architectural Choice, Not a Dashboard

The final concept moves stability from *preventing* failure to *seeing* it. Deciding what must be observable is, per [[richards-ford-fundamentals]], an architectural decision tied to the critical architecture characteristics — not a thing you sprinkle on at the end.

The link is direct: a circuit breaker is only as good as your ability to know it tripped, and a breaker tripping is a first-class signal that a dependency is sick — far more actionable than a raw latency graph. If you decided that **availability** is a critical characteristic for a given path, then *the breaker state, the timeout-fire rate, and the bulkhead saturation on that path must be observable*, because those are the leading indicators of the characteristic you committed to protecting. Choosing them is choosing what you can react to.

There is a subtle inversion here that makes the point sharp: the resilience patterns *create the very signals you most want to observe*. A breaker transitioning to OPEN is a cleaner, earlier, more semantic alarm than a flood of timeout errors, because it is the system itself declaring "I have given up on dependency X." A bulkhead saturating tells you exactly which compartment is under water before the water reaches the others. So the patterns are not only mechanisms for *bounding* failure — they are *instruments* for *seeing* it. An architecture that decided to place breakers and bulkheads has, almost as a side effect, decided what its most important health signals are. That is why observability is not a dashboard you add later: the things worth watching were determined when you drew the integration points and chose how to defend them.

This is the seed of the **fitness function** idea that ch-09 develops in full: from [[richards-ford-fundamentals]], a fitness function is "an objective integrity assessment of some architectural characteristic(s)" — an automated check that fails the build (or fires an alert) when a protected characteristic erodes. "Breaker for vendor X must not stay OPEN longer than N minutes" or "p99 on the critical path < X" is a resilience characteristic turned from aspiration into *enforcement*. Observability is the substrate that makes such a fitness function measurable at all — you cannot enforce a characteristic you cannot see, and you cannot see one you did not decide to instrument.

And every one of these decisions — which integration points get breakers, what the timeout budgets are, which resources are bulkheaded, what must be observable — is exactly the kind of "architecturally significant" choice [[nygard-release-it]] says belongs in an ADR:

> "We will keep a collection of records for 'architecturally significant' decisions: those that affect the structure, non-functional characteristics, dependencies, interfaces, or construction techniques." — Nygard, "Documenting Architecture Decisions" (cognitect.com/blog/2011/11/15/documenting-architecture-decisions) — [[nygard-release-it]]

Resilience choices affect non-functional characteristics and dependencies by definition, so they are precisely what an ADR exists to preserve — "a conversation with a future developer" explaining *why* this dependency got a 2-second timeout and that one got a breaker with a 30-second cooldown.

---

## 7. Applied to the Sales Agent (Lina TMR): Every Tool Call Is an Integration Point

Lina TMR is an LLM agent acting over many external SaaS tool APIs — CRM, calendar, email, enrichment vendors, a dozen others. *Every single one of those tool calls is an integration point*, in Nygard's exact sense: a place where a failure can enter and spread. This is the chapter's most important application, because the agent's failure mode without these patterns is catastrophic and silent.

Picture the naïve agent loop: the planner decides to call the CRM, then the enrichment vendor, then the calendar, synchronously, inside one reasoning turn, with no timeouts. Now the enrichment vendor has a bad afternoon and starts responding in 40 seconds instead of 400ms. Without a timeout, the agent's turn *hangs* for 40 seconds per call. Without a circuit breaker, the next 200 conversations that need enrichment each independently wait out that 40 seconds — the agent's worker pool drains, and now conversations that *don't even use enrichment* can't get a worker. One slow vendor has stalled the entire agent fleet. This is the cascading failure of §2, and the agent-over-many-SaaS-tools topology is unusually prone to it precisely because it has *so many* integration points, each owned by a third party whose uptime you do not control.

The toolkit maps cleanly onto the agent:

| Agent concern | Pattern | The bet, priced |
|---|---|---|
| A vendor hangs and pins the reasoning turn | **Timeout** on every tool call, budgeted from the agent's per-turn latency target | give up the rare slow-but-correct response; never let one vendor freeze a turn |
| A vendor is down for an hour | **Circuit breaker per vendor** — trip after N failures, fail fast, let the agent route around it (degrade, defer, tell the user) | accept false rejections during cooldown; stop spending the whole timeout budget on a known-dead vendor every call |
| One sick vendor starves all others | **Bulkhead per vendor** — separate connection/worker pools so the enrichment outage can't drain the CRM pool | strand some idle capacity per partition; keep healthy vendors fully usable during one vendor's outage |
| A vendor's write succeeds but its response never returns | **Idempotent retries** + treat the response as **outside data** (immutable, possibly-stale snapshot, never authoritative live state — the ch-06 discipline from [[helland-data-outside-inside]] via [[richardson-saga]]) | accept eventual consistency; make retries safe under at-least-once delivery |
| A long multi-tool workflow shouldn't fail wholesale on one step | **Async/event steps with compensations** (saga shape) instead of one synchronous chain | inherit the saga's lost-isolation countermeasure burden; dissolve the cascade coupling between steps |

The architectural payoff — the research framing for this learner, who recently built an agent *benchmark* and is now designing the agent itself — is that resilience here is not a library you import, it is a *boundary decision*. Putting a circuit breaker between the agent core and each vendor adapter is the same inward-dependency move from ch-03: the technology-free agent core never blocks on, and never cascades from, the volatile outside world, because the adapter at the edge owns the timeout, the breaker, and the bulkhead. The agent's blast radius is decided when you draw that edge — and you record *why* each vendor got the budget it got in an ADR, so the next engineer doesn't quietly remove the breaker that is the only reason a Tuesday-afternoon enrichment outage didn't take down the whole sales agent.

---

## Where This Goes

This chapter bounded the blast radius of *failure* at runtime: timeouts, breakers, bulkheads, and the recognition that async integration is itself a resilience decision. Ch-09 bounds the blast radius of *change* over time. It takes the same "keep the bet revisable" idea up a level: the **strangler-fig** pattern makes a migration reversible by growing the new system around the old and retiring it piece by piece ("investment and returns occur gradually and visibly," with low per-step risk — [[martin-strangler-fig]]), and **fitness functions** turn the architecture characteristics you decided to protect here — the dependency rule, the latency budget, the breaker behavior — from aspiration into an automated check that fails the build when they rot. Resilience keeps an *outage* from being a one-way door; evolution keeps an *architecture* from being one.
