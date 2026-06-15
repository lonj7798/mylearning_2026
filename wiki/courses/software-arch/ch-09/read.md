<!-- chapter: ch-09
     track: evolution
     kind: content
     title: Evolution — Strangler-Fig Migration and Fitness Functions
     deps: [[ch-04]], [[ch-08]]
     sources: [[martin-strangler-fig]], [[newman-building-microservices]], [[fowler-monolith-first]], [[richards-ford-fundamentals]], [[martin-clean-arch]], [[distributed-monolith]], [[decompose-by-business-capability]]
-->

# Chapter 09 — Evolution: Strangler-Fig Migration and Fitness Functions

> **Core insight.** Every architectural bet you have priced across this course was made with incomplete information, and the domain will keep moving after you make it — so the last property an architecture needs is not correctness today but *revisability tomorrow*. There are two distinct threats to a bet you have already placed. The first is that *changing* it is a one-way door: the big-bang rewrite that freezes the old system, accumulates all risk into a single cutover, and delivers nothing until the end. The strangler-fig answers that — grow the new system *around* the old, route one capability seam at a time, and retire the legacy piece by piece so each step is small enough to undo. The second threat is quieter: a bet you chose deliberately can *rot silently* as the system evolves, the dependency rule violated one import at a time, the p99 latency creeping past the number you committed to, until the architecture you documented no longer matches the architecture you run. The fitness function answers that — an automated, objective check that fails the build the moment a protected characteristic erodes. Migration keeps the bet *changeable*; fitness functions keep it from *decaying* while unchanged. Both are the same discipline applied at different timescales: bound the blast radius of *change* the way ch-08's stability patterns bounded the blast radius of *failure*.

> **Guideline.** When you must replace something serious, refuse the rewrite. Put a façade or proxy in front of the legacy system so routing can change without callers noticing, then extract one business-capability seam at a time — *its behavior and its data together* — verify in parallel, delete the strangled path, and repeat. You are buying gradual, visible return and per-step reversibility at the cost of running two systems at once and carrying a coordination tax for the duration. And when you finally pick the few architecture characteristics that matter for *this* system, do not leave them as aspirations in a wiki: encode each one as a fitness function — an ArchUnit dependency rule, a contract test, a p99 monitor with a build-failing threshold — so the characteristic is *enforced*, not merely *intended*. Price evolution honestly: it costs ongoing migration discipline and continuous enforcement, but that recurring cost is exactly how the expensive-to-reverse decisions stay from quietly becoming the irreversible *wrong* ones.

---

## 1. The Spine, Re-Applied: Evolution Is the Insurance on Every Other Bet

The course's organizing claim, installed in ch-01, is that architecture is the set of decisions that are expensive to reverse, judged by one law:

> "Everything in software architecture is a trade-off. If you think you've found something that isn't a trade-off, you likely just haven't found the trade-off yet." — Richards & Ford, *Fundamentals of Software Architecture* (book; thesis extracted, quoted as commonly published) — [[richards-ford-fundamentals]]

Every chapter so far has placed a bet: ch-02 drew bounded contexts on the slowest-changing structure, ch-03 pointed dependencies inward, ch-04 defaulted to a modular monolith, ch-05 published contracts, ch-06 accepted lost isolation for sagas, ch-08 placed breakers to bound failure. Each was made under uncertainty, and each will be wrong *eventually* — not because the analysis was bad, but because the domain you analyzed will not hold still. This chapter is the insurance on all of them. It does not place a new structural bet; it makes the *act of changing* a bet cheap, and the act of *keeping* a bet honest automatic.

There is a precise way to state what this chapter does to ch-01's definition. Ch-01 defined architecture as the decisions that are expensive to reverse, and observed that the same decision can sit anywhere on a spectrum from trivially-reversible to one-way-door depending on *how you build it*. This chapter is the set of techniques that move a decision *down* that spectrum after you have made it — that take a service boundary, a vendor choice, a data-store commitment, and lower its reversal cost so that being wrong about it is survivable. That is a fundamentally different activity from making the decision well in the first place, and it is why evolution deserves its own phase rather than being folded into the chapters that placed each bet. You can place every bet in this course correctly and still ship a brittle architecture if you never invest in lowering the cost of being wrong; conversely, a merely-adequate set of bets, kept cheap to revise, will outlast a brilliant set of bets frozen in concrete.

The reason this sits in the **evolution** phase next to ch-08's resilience is the through-line stated in [[richards-ford-fundamentals]]:

> "An evolutionary architecture supports guided, incremental change across multiple dimensions." — Ford, Parsons & Kua, *Building Evolutionary Architectures* (book; thesis extracted) — [[richards-ford-fundamentals]]

Read the sentence as three load-bearing words. **Guided**: change is steered by something objective, not by hope — that is what fitness functions provide. **Incremental**: change arrives in small, reversible steps, never one big cutover — that is what the strangler-fig provides. **Across multiple dimensions**: not just code structure but performance, security, data, deployability — every architecture characteristic ch-01 told you to derive can rot, so every one can be guarded. Ch-08 bounded the blast radius of a *failure* in time (one outage, one breaker). This chapter bounds the blast radius of a *change* in time (one capability, one migration step; one violated rule, one failed build).

### 1.1 Fowler's amendment, finally cashed out

Ch-01 quoted Fowler's friendly amendment to the "architecture = hard-to-change decisions" definition: *"a good architect makes change easier — thus reducing architecture."* That was an aspiration when you first heard it. This chapter is where it becomes mechanism. The strangler-fig *shrinks the irreversible set* by converting one giant irreversible rewrite into a sequence of small reversible extractions. Fitness functions *shrink it differently* — by making the cost of an accidental architecture change (a violated dependency rule, a blown latency budget) immediate and cheap to detect, so it never compounds into an expensive-to-reverse fact on the ground. Both are Fowler's amendment as code: lowering the reversal cost of the decisions you already made.

### 1.2 Why evolution is a phase, not an afterthought

A common reading treats migration and "keeping things clean" as operational chores that happen *to* an architecture once the real design work is done. This course rejects that framing, and the rejection is the reason ch-08 and ch-09 form a phase of their own. The decision to make an architecture evolvable is itself an architecture decision with its own characteristics and its own cost — and it is one you can only buy *up front*. You cannot retrofit reversibility onto a system that hard-wired every caller to a concrete implementation; the strangler is only cheap because someone decided to invest in an interception seam before they needed it. You cannot retrofit "the dependency rule was never violated" onto a codebase that violated it for two years; the fitness function only works because it ran from the start. Evolvability is a characteristic you derive and pay for like any other "-ility" — the difference is that its payoff is entirely in the future, which is exactly why it is the one most often skipped and the one this phase exists to defend.

---

## 2. Why Big-Bang Rewrites Fail

The pattern only makes sense against the thing it refuses. From [[martin-strangler-fig]] (Martin Fowler, "StranglerFigApplication," martinfowler.com/bliki, 2004/renamed 2019):

> "Replacing a serious IT system takes a long time, and the users can't wait for new features." — Fowler, "StranglerFigApplication" — [[martin-strangler-fig]]

That one sentence contains the entire economic argument against the rewrite. A serious system represents years of accumulated business logic, edge cases, and hard-won bug fixes. A rewrite proposes to reproduce all of it before delivering *anything* — and during that window the old system is frozen (every new feature you add to it is throwaway work you must port) while the business keeps demanding the features users can't wait for. You are now in a race between a half-built replacement and a moving target, funding two efforts to ship one system.

The risk profile is the real killer. A rewrite concentrates *all* of its risk into one event: the cutover. Everything works on the new system or it doesn't, and you find out at the worst possible moment — the moment of switch, in production, with the old system already retired. There is no graceful partial state. The strangler inverts every one of these properties:

| Property | Big-bang rewrite | Strangler-fig migration |
|---|---|---|
| Value delivery | Nothing until the end | Gradual and visible per increment |
| Risk concentration | All in one cutover | Spread across many small steps |
| Reversibility | Effectively zero past the cutover | Each step undoable in isolation |
| Old-system work during migration | Frozen / throwaway | Keeps running and shipping |
| Failure cost | The whole rewrite | One capability's rework |
| Learning | Front-loaded guesses | Each step teaches the next |

Fowler is honest that this buys *manageability*, not *ease* — a hedge the chapter must reproduce rather than oversell:

> "Replacing a software system… is never going to be an easy task" — Fowler, "StranglerFigApplication" — [[martin-strangler-fig]]

The strangler does not make migration easy. It makes it *survivable*, which is a different and more honest claim.

### 2.1 The metaphor, and why it earns its name

Fowler watched strangler figs in Queensland: a vine germinates high in a host tree's canopy, grows roots down *around* the existing trunk, and over years becomes self-supporting as the original tree dies inside it. The new structure is built on and around the old one, never by clear-cutting first.

> "Like the fig, it begins with small additions, often new features, that are built on top of, yet separate to the legacy code base." — Fowler, "StranglerFigApplication" — [[martin-strangler-fig]]

The "on top of, yet separate to" is the design constraint hiding in the biology: the new code coexists with the legacy code, intercepting and replacing it incrementally, never requiring a moment where both are torn out at once.

### 2.2 The trap is *not finishing*

The strangler has a well-known failure mode that the metaphor itself warns about: the fig that *stops growing*. A migration that extracts two seams and then stalls — because the urgent feature work returned, or the team that owned the effort dissolved — leaves the system in the most expensive state of all: two codebases, a permanent façade, a split data estate, and the full distribution tax with none of the eventual payoff. The reversibility that makes each *step* safe also makes it psychologically easy to pause indefinitely, and a paused strangler is just a distributed monolith with good intentions. The discipline the pattern demands is therefore not only "go incrementally" but "*keep going* until the legacy core is gone or deliberately, explicitly kept" (step 4's "or small enough to keep" is a *decision*, not a place you drift to). This is the honest reading of Fowler's hedge that the pattern is manageable, not easy: the management never stops until the migration does.

This is also where the chapter's two halves first touch. A stalled strangler is detectable: the count of capabilities still in the legacy core stops decreasing, the façade routing table stops changing, the parallel-run comparisons stop being added. Each of those is a *measurable* signal, which means each can be a fitness function — a check that fails or alerts when a migration declared "in progress" has not moved a seam in N weeks. The same discipline that protects the dependency rule can protect *migration momentum itself*, turning "we'll finish it eventually" from a hope into a tracked, build-visible commitment. The tool that keeps an unchanged bet honest (§6) is also the tool that keeps a *changing* bet from quietly freezing halfway.

---

## 3. The Mechanism at Design Altitude

This is a *design-decision* course, so the strangler matters here as an architecture pattern — where you place the interception boundary and which seam you cut — not as a deployment recipe. [[martin-strangler-fig]] gives the four-step loop:

1. **Intercept.** Put an HTTP proxy, event interceptor, or façade between callers and the legacy system so routing can change *without callers noticing*. This is the single most important architectural move in the whole pattern: the façade is the indirection layer that makes every later step reversible, because the caller is decoupled from *which* implementation answers.
2. **Extract one seam.** Pick a business capability with a clean boundary — exactly the [[decompose-by-business-capability|capability seam]] discipline from ch-04 — reimplement it as new code, and route only *that* capability's traffic to the new implementation through the façade.
3. **Verify and shrink.** Run old and new in parallel where the risk warrants it, compare outputs, and once you trust the new path, delete the strangled legacy code so the monolith genuinely gets smaller.
4. **Repeat.** Until the legacy core is gone, or small enough that it is no longer worth strangling.

> **See it move:** open [`figures/strangler-fig-timeline.html`](figures/strangler-fig-timeline.html) and drag the time slider from "Day 0" to "Retired." Watch the façade stay fixed while capability seams migrate from the legacy monolith to new services one at a time — and use the **Roll back** button on any in-flight seam to confirm the property the chapter keeps insisting on: each step is reversible in isolation, so a wrong move costs one capability's rework, not the whole rewrite.

### 3.1 The façade is the load-bearing decision

Notice that steps 2–4 are only cheap *because* step 1 was done first. Without the interception layer, "route this capability to the new code" means editing every caller — which re-couples the migration to the very call sites you are trying to free, and makes rollback a second round of edits. The façade is what converts a migration into a *routing-table change*. That is why the strangler is fundamentally a statement about indirection, and why it composes so naturally with the dependency-inversion habit from ch-03: in both cases an interface in the middle is what lets you swap the implementation behind it without the caller participating.

There is a real decision *inside* step 1 that is easy to skip past: **where to put the interception point**. The façade can sit at the HTTP edge (a reverse proxy / API gateway routing by path), inside the process (a dispatch interface the monolith itself calls), or on the event bus (an interceptor that re-routes a class of events). Each placement prices differently. An HTTP-edge proxy is the most decoupled and the most reversible — flip a route, traffic moves — but it can only strangle capabilities that *are* HTTP endpoints. An in-process dispatch seam reaches capabilities that have no external URL, but it lives inside the artifact you are trying to retire, so the façade itself has to be carried until the end. The First Law is already biting at step 1: a more decoupled interception point costs more to retrofit onto a legacy system that wasn't built with one, while a cheaper-to-add in-process seam buys less reversibility. You choose the interception point the same way you choose any architecture decision — by which property you most need to keep cheap.

### 3.2 Why the advantages are not free

Fowler lists the upside verbatim, and the chapter prices each one:

> "Investment and returns occur gradually and visibly." — Fowler — [[martin-strangler-fig]]
> "Since these components are small, there isn't so much risk involved." — Fowler — [[martin-strangler-fig]]
> "The business can reap the value from these new components, allowing earlier return on the investment." — Fowler — [[martin-strangler-fig]]

Each increment also "helps us make better decisions as modernization continues" — the learning compounds, which is the opposite of the rewrite's front-loaded guessing. But the First Law forbids a free lunch, so name the cost explicitly. The strangler **keeps cheap to change**: the migration path (any single seam can be rerouted or rolled back), the risk per step (small), the funding model (incremental). It **makes expensive**: running two systems in parallel for the entire duration, maintaining the façade as a permanent piece of infrastructure for as long as the migration lasts, and tolerating a hybrid state where some capabilities live in the new world and some in the old — which means cross-capability transactions now straddle the boundary you have not finished moving, dragging in every inside-vs-outside-data concern from ch-06. A strangler migration that drags on for years is not reversible-and-cheap; it is a permanent distributed system you accidentally signed up for. The reversibility is real, but it is rented, not owned.

### 3.3 Verification is the step that is most often skipped

Step 3 says "verify and shrink," and the *verify* is where the per-step safety actually lives. The strongest form is **parallel run** (sometimes called dark launching or shadow traffic): route a copy of live traffic to both the legacy path and the new implementation, compare the outputs, and only switch the authoritative answer to the new path once they agree for long enough. This is what makes a step reversible *in practice* rather than just *in principle* — you discover the new implementation is wrong while legacy is still answering, so rollback is "stop comparing," not "restore from backup at 3am." Parallel run has its own cost (you are paying for two executions of the same work, and side-effecting capabilities — sending an email, charging a card — cannot be naively double-run, which ties straight back to ch-06's distinction between safe-to-replay outside data and effectful inside operations). Skipping verification is how a strangler *looks* incremental while actually betting each switch blind — you keep the small steps but throw away the reversibility that was the entire point of taking them.

---

## 4. Extract the Behavior *and* Its Data — Newman's Sharpest Rule

The strangler tells you to peel off a capability; Newman tells you the one way to do it that does not trade a rewrite for a [[distributed-monolith]]. From [[newman-building-microservices]] (Sam Newman, *Building Microservices* 2e / *Monolith to Microservices* — books, theses extracted; corroborated by O'Reilly excerpts and Newman's talks):

Newman's migration playbook is explicitly incremental — start from a monolith, often a modular one, find a seam, and use the strangler to peel off one capability at a time — but with a non-negotiable refinement: extract the *behavior* and its *data* **together**. The failure he is warning against is the half-migration where the new service has its own code but still reaches back into the monolith's database for its tables. That gives you a new deployable that *cannot deploy independently*, because it shares a schema with the thing you are trying to escape — which is the exact definition of the anti-pattern:

> "all the pain of distributed systems without the independence" — Newman, on the [[distributed-monolith]] (book; thesis extracted) — [[newman-building-microservices]]

The data-ownership rule is the same one ch-04 and ch-06 reached from other directions, restated for migration:

> "Each microservice must own its data. Shared databases create hidden coupling and destroy independent deployability." — Newman (book; thesis extracted) — [[newman-building-microservices]]

So the strangler step is not "move the code, point it at the old DB, move the data later." It is "move the code and its data as one atomic capability extraction," even though that is the harder version — because the easy version produces a distributed monolith *mid-migration*, which is strictly worse than the monolith you started with. This is where the chapter's two halves meet: the strangler gives you the *sequence*, Newman gives you the *unit*, and the unit is behavior-plus-data or nothing.

The reason data is the hard part deserves a name, because it is what makes the "move data later" shortcut so tempting and so wrong. Newman's whole framework is organized around **independent deployability** as the acid test:

> "Independent deployability is the single most important principle." — Newman (book; thesis extracted, paraphrased from *Building Microservices* 2e) — [[newman-building-microservices]]

and independent deployability is achieved through **information hiding** — a service "exposes behavior through APIs while hiding implementation details like databases, technology choices, and internal workflows." A shared database is the maximal *failure* of information hiding: the new service's most private implementation detail (its tables, its schema) is also the monolith's, so a schema change in either ripples into both and they can never deploy apart. That is why the data must move with the behavior — not as a tidiness preference, but because the data store *is* the thing whose hiddenness creates the independence. Deferring the data move is deferring the independence, which means the intermediate state of a "move data later" migration is, by Newman's own definition, not a partial microservice at all but a distributed monolith wearing a service's clothes.

### 4.1 The operational counterpart to MonolithFirst

The strangler is the second half of an argument ch-04 made with [[fowler-monolith-first]]. That chapter's claim:

> "Almost all the successful microservice stories have started with a monolith that got too big and was broken up." — Fowler, "MonolithFirst" — [[fowler-monolith-first]]

MonolithFirst is the *entry* discipline: don't place the irreversible boundary bet before you understand the domain. The strangler is the *exit* discipline: when you finally do split, do it incrementally and reversibly rather than in one cutover. They are the same refusal — to make a single large irreversible bet on boundaries you cannot yet trust — applied at the two ends of a system's life. Fowler's reason boundaries are the hard part is exactly why the exit must be incremental too:

> "Even experienced architects working in familiar domains have great difficulty getting boundaries right at the beginning." — Fowler, "MonolithFirst" — [[fowler-monolith-first]]

If you cannot get the boundary right at the beginning, you certainly cannot get *all* of them right in a single big-bang extraction. The strangler lets each boundary be discovered and corrected one seam at a time, with the cost of a wrong seam bounded to that seam's rework.

Seen this way, MonolithFirst and the strangler are not two patterns but one continuous policy on boundary commitment: *never make a boundary bet larger than you can afford to be wrong about*. At the start of a system that means "don't draw service boundaries at all yet" (modular monolith). Later it means "draw one service boundary, reversibly, and learn from it before drawing the next" (strangler). The policy is constant; only the affordable bet size changes as your understanding of the domain grows. This is the deepest sense in which evolution is the insurance on every other chapter: it is the discipline that keeps boundary commitments — the single most expensive-to-reverse decision the course identified — proportioned to how much you actually know.

---

## 5. Doc-vs-Reality: "A Rewrite Is Cleaner, So It's Faster"

The popular narrative this chapter must kill is the engineer's perennial temptation: *the legacy system is a mess; a clean rewrite will be faster and the result will be better.* The reconciliation table in [[COLLECTION-PLAN]] frames the whole evolution arc around primary sources beating folklore, and the strangler's primary source is blunt about this one.

| Popular narrative | What the primary source actually says |
|---|---|
| "A clean rewrite is faster than untangling the legacy system." | Fowler: replacing a serious system "takes a long time, and the users can't wait." The rewrite delivers nothing until the end and concentrates all risk in one cutover; the strangler delivers value gradually with bounded per-step risk. — [[martin-strangler-fig]] |
| "Once we split it into services, the migration is basically done." | Newman: a service that still shares the monolith's database is a [[distributed-monolith]] — "all the pain of distributed systems without the independence." The migration is done only when behavior *and data* are extracted together. — [[newman-building-microservices]] |
| "Microservices are the modern best practice; monoliths are legacy to be rewritten." | Fowler: start MonolithFirst; "almost all successful microservice stories started with a monolith." It's a trade with a MicroservicePremium, not an upgrade — and the migration is incremental, not a rewrite. — [[fowler-monolith-first]] |

The resolution is the same in every row: the rewrite *feels* faster because it ignores the parts that are hard precisely because they are valuable — the accumulated edge cases and the data. The strangler is slower per calendar quarter and faster in *expected risk-adjusted delivery*, because it never bets the whole system on a single event. The folklore optimizes for the wrong variable (perceived cleanliness) and the primary source optimizes for the right one (bounded, reversible delivery).

The second row deserves its own emphasis because it is the failure that masquerades as success. A team that "split into services" and declares victory while the new services still read from the monolith's tables has done the *visible* half of the migration (new deployables, new repos, new dashboards) and skipped the *load-bearing* half (data ownership). By Newman's definition they have not reached microservices at all; they have built the distributed monolith — and worse, they have done it while believing the hard part is over, so the actual hard part (untangling the shared schema) now has no political will behind it. The primary source's resolution is uncomfortable precisely because it refuses to let the org-chart change count as the architecture change: the migration is measured by data ownership, not by the number of services on the deployment diagram.

The deeper reason the rewrite myth is so durable is a cognitive bias the architect must name to resist: the legacy system's complexity is *visible* (you stare at the mess every day) while its *value* is invisible (the thousand edge cases it silently handles correctly are exactly the ones you forgot exist). A rewrite estimate is therefore built almost entirely from the visible part — "I could write the clean version of this in three months" — and the invisible part, the edge cases and the data migration, is precisely what blows the estimate and the cutover. The strangler does not make that hidden value cheaper to reproduce; it makes the *discovery* of it incremental, so each seam surfaces its own hidden complexity at the moment you reimplement it, with the legacy version still running next to it as the executable specification of what "correct" means. This is the same logic as MonolithFirst, ch-04's entry-side argument: you do not know enough to make the big bet, so you structure the work to *learn before you commit*. The rewrite asks you to commit before you learn; the strangler inverts the order.

---

## 6. Fitness Functions: Turning Characteristics from Aspiration into Enforcement

The strangler keeps a bet *changeable*. Fitness functions address the other failure mode — a bet you chose deliberately decaying while you weren't looking. The definition from [[richards-ford-fundamentals]]:

> "A fitness function is an objective integrity assessment of some architectural characteristic(s)." — Ford, Parsons & Kua, *Building Evolutionary Architectures* (book; thesis extracted) — [[richards-ford-fundamentals]]

Unpack each word. **Objective**: not a code-review opinion that depends on who reviewed; a check that returns pass/fail the same way every time. **Integrity assessment**: it asks "is this property still intact?" not "is this code good?" — it guards a *characteristic*, not aesthetics. **Architectural characteristic(s)**: the "-ilities" ch-01 told you to derive from requirements — the few that actually matter for *this* system. A fitness function is the mechanism that connects ch-01's "derive the critical few characteristics" to ch-09's "keep them from rotting": you derive the characteristic, then you write the function that fails the build when it erodes.

### 6.1 The dependency rule as a fitness function

The cleanest example ties straight back to ch-03. The Dependency Rule from [[martin-clean-arch]]:

> "Source code dependencies can only point inwards." — Robert C. Martin, "The Clean Architecture" — [[martin-clean-arch]]

In ch-03 this was a *design intent*. The problem with an intent is that it degrades one pull request at a time: a developer in a hurry imports a framework type into the domain core, the review misses it, and now policy depends on detail — the exact thing the rule forbids — and nobody notices until the framework needs replacing and the core won't come loose. A fitness function makes the intent *enforced*. An ArchUnit rule (or an import-linter check, or a custom dependency test) that asserts "no class in `domain.*` imports anything in `infrastructure.*`" runs in CI and *fails the build* on the offending PR. The violation is caught at the cheapest possible moment — before merge — instead of years later as an expensive-to-reverse fact. That is the difference the chapter is selling: "keep the dependency rule" becomes a line of test code, and the architecture you documented is the architecture you run.

The reason this particular fitness function is the canonical example is that the characteristic it guards — testability and framework-independence via the Dependency Rule — is the one whose erosion is *invisible until it is catastrophic*. A single forbidden import does nothing observable: the tests still pass, the feature still works, the violation is inert. It only becomes expensive at the exact moment you most need the property — when you try to swap the framework or the database the core was supposed to be independent of, and discover the core has been quietly fused to it across a hundred small violations nobody flagged. This is the silent-rot threat the chapter opened with, made concrete: the property that protects your most expensive-to-reverse decision (domain policy) is itself the one that degrades most silently, which is precisely why it must be *enforced by a machine* rather than *trusted to discipline*.

### 6.2 A fitness function is not a unit test

It is worth being precise about what makes a fitness function a distinct idea rather than a fancy name for testing, because the distinction is what makes it *architectural*. A unit test asserts that a *function* behaves correctly — given this input, expect that output; it guards *functionality*. A fitness function asserts that an *architecture characteristic* holds across the whole system — that the dependency graph has a shape, that p99 stays under a budget, that no module exceeds a coupling ceiling; it guards a *structural or operational property* that no single unit test could see because the property is emergent from how the parts fit together. A passing test suite tells you the features work; it tells you nothing about whether the system has quietly become unmaintainable. The fitness function is the check at the *architecture* altitude — which is exactly why it belongs in a design course and a unit test does not. It is also why fitness functions are the natural home for the trade-offs the whole course has been pricing: a characteristic you bet on (testability via the dependency rule, latency, deployability) is precisely the thing a fitness function is built to defend.

### 6.3 The taxonomy of fitness functions

Fitness functions are not only structure tests. [[richards-ford-fundamentals]] frames them across dimensions, and the practical forms are:

| Characteristic guarded | Fitness function form | Fails the build when… |
|---|---|---|
| Modularity / clean dependencies | ArchUnit / import-linter rule | a forbidden cross-layer import appears |
| Performance | p99 latency assertion in a load test | p99 crosses the committed threshold |
| Security | dependency-vuln scan / policy-as-code gate | a known-CVE dependency or open port is introduced |
| Reliability | chaos/contract test in CI | a circuit-breaker or timeout is removed (ties to ch-08) |
| Coupling | cyclic-dependency / fan-out metric | a module's afferent coupling exceeds a budget |

Some run *atomically* in CI (the dependency rule); some run *continuously* in production as monitors (the p99 latency check on live traffic). The unifying property is that the protected characteristic has a *number or a rule* attached and an *automated consequence* when it is violated. An undocumented, unchecked characteristic is an aspiration; a fitness function is a commitment.

The triggered/continual split matters more than it looks. A *triggered* fitness function runs at a discrete moment — a build, a deploy gate — and answers "is the property intact in the artifact we are about to ship?" A *continual* one runs against the live system forever and answers "is the property *still* intact under real load right now?" Structural properties (dependency shape, cyclomatic limits) are knowable from the code, so they belong in CI; operational properties (p99 latency, error budget, availability) are only knowable from production, so they belong in monitoring with an alerting threshold that is *also* build-failing on a load test. The architect's job is to put each protected characteristic on the side of the line where it can actually be measured — a continual property checked only at build time gives false confidence (it passed in staging and rotted in production), and a structural property checked only in production catches the violation far too late to be cheap to fix.

### 6.4 Pricing the fitness-function bet

The First Law applies here too. Fitness functions **keep cheap to change**: refactoring freely, because the guardrails catch the moment a refactor breaks a protected property; onboarding, because the rules document the architecture executably. They **make expensive**: writing and maintaining the checks (a fitness function is code that can itself rot or go stale), tuning thresholds so they neither flap nor sleep, and the discipline of *actually failing the build* rather than adding the violation to an ignore-list — the moment you start suppressing fitness-function failures, you have an aspiration again, wearing a CI badge. The honest price of evolutionary architecture is *ongoing enforcement*: it is not a one-time setup but a standing cost, and that standing cost is exactly what buys you a bet that stays honest as the system changes around it.

### 6.5 The two halves are one discipline

It is tempting to read the strangler and fitness functions as two unrelated tools that happen to share a chapter. They are not. They are the two faces of a single property — keeping a bet revisable — and they reinforce each other directly. Fitness functions are what *create and preserve* the seams the strangler later needs: a dependency-isolation rule is precisely what stops a module's boundary from filling in with cross-module calls, which is what keeps that module *extractable* if the disintegrator pressure ever justifies a strangle. Run the implication backward and the dependency is symmetric: a strangler migration in flight is a period of maximal structural churn — new services appearing, data moving, façade routing changing — which is exactly when an architecture is most likely to rot, and therefore exactly when fitness functions earn their cost. The strangler needs clean seams to grab; fitness functions keep the seams clean; the migration that uses those seams is the moment the fitness functions matter most. Evolution is one capability, measured at two timescales — the slow timescale of deliberate migration and the fast timescale of every commit — and an architecture that does one without the other is only half-protected.

---

## 7. Applied to the Sales Agent: Which Seam to Strangle First, and What to Guard

Bring it back to the learner's production sales agent (Lina TMR) — an LLM agent acting over many external SaaS tool APIs. The course defaulted this system to a **modular monolith** with clean bounded contexts (ch-04) and clean internal dependencies (ch-03). Evolution is how that default stays correct as the agent grows.

### 7.1 Choosing the first seam

The strangler's "pick one capability with a clean boundary" maps directly onto the bounded contexts ch-02 identified — lead/pipeline, conversation, scheduling, CRM-sync. The question evolution forces is *which seam to extract first*, and the answer is the same disintegrator analysis from ch-04: extract the seam where a specific deploy/scale/team pressure justifies the distributed tax, not the one that is merely annoying. For Lina TMR the strong candidate is the **CRM-sync** context: it is the part most coupled to external SaaS APIs (Salesforce, HubSpot, etc.), most likely to need independent scaling when a vendor is slow, and most likely to need independent deployment when a vendor changes its API. Crucially, by Newman's rule you extract CRM-sync's *behavior and its data together* — the synced snapshots of external records move with the code, so the new service owns its own store rather than reaching back into the monolith's database. And because every external SaaS response is **outside data** (ch-06) — an immutable, versioned, possibly-stale snapshot, never authoritative live state — the seam is naturally clean: the agent's *inside* model was already separated from what CRM-sync ingests, so strangling it does not have to untangle a shared mutable state. The inside-vs-outside discipline from ch-06 is what makes the strangler seam cheap here; a system that treated SaaS responses as live shared state would have no clean place to cut.

Note what the analysis explicitly does *not* say: it does not say to extract anything. The default remains the modular monolith from ch-04, and the honest answer to "which seam first" is frequently "none yet." A single-team agent serving moderate load has no deploy-coupling pain, no team-cognitive-load pain, and no scale asymmetry that a service split would relieve — so extracting CRM-sync would buy a distributed system whose only justification is a future that may not arrive. The strangler is the *mechanism* you reach for when the disintegrator pressure becomes real, not a goal to pursue on schedule. Recording that "we are staying a modular monolith, and here is the specific pressure that would trigger the first strangle" is itself a priced ADR — it names the seam, names the trigger, and pre-commits the team to the reversible path when the trigger fires, so the decision is made calmly in advance rather than in a panic when a vendor outage finally forces the issue.

If and when the trigger does fire, the strangle of CRM-sync runs the four-step loop concretely:

1. **Intercept.** The agent core already calls CRM-sync through an in-process port (ch-03's dependency inversion), so the "façade" mostly exists — the move is to make that port routable so a call can go to the in-monolith implementation *or* a new out-of-process service. Because the port was always there, step 1 is nearly free here, which is the payoff of having built the modular monolith with clean ports in the first place.
2. **Extract one seam.** Stand up the CRM-sync service owning its own snapshot store, and route only CRM-sync traffic to it through the port. The outside-data discipline means the snapshots it owns are immutable versioned copies, so there is no shared mutable state to coordinate during the move.
3. **Verify and shrink.** Parallel-run: have both implementations fetch and normalize the same vendor records, compare the normalized snapshots, and only switch authority to the new service once they agree — then delete the in-monolith CRM-sync code. Side-effecting calls (writing back to the CRM) are the ones that *cannot* be naively double-run, so those switch atomically rather than in parallel, exactly the ch-06 distinction surfacing again.
4. **Repeat or stop.** With CRM-sync out, re-run the disintegrator analysis on the remaining contexts. If none clears the bar, *stop deliberately* and record it — a two-component system (agent monolith + CRM-sync service) is a perfectly good resting state, not an incomplete migration, as long as stopping there was a decision rather than a stall.

The whole sequence is what the [`figures/strangler-fig-timeline.html`](figures/strangler-fig-timeline.html) companion animates with CRM-sync as the first seam — step through it and roll back step 2 to feel how the in-process port makes the agent's first extraction cheap and reversible.

### 7.2 Fitness functions for the agent

The characteristics worth guarding in this system, encoded as fitness functions:

- **The dependency rule on the agent core.** An ArchUnit-style check that the domain policy (lead scoring, routing rules, conversation logic) never imports a specific LLM-vendor SDK, vector-DB client, or web framework. This is the executable form of ch-03's central bet — keep the *expensive-to-reverse* part (domain policy that must outlive whichever LLM API is current) insulated from the *cheap-to-swap* parts. The day you migrate from one model vendor to another, this rule is what guarantees the migration is a swap at the edge, not surgery through the core.
- **Resilience characteristics never regress.** A CI check (ties to ch-08) that every external SaaS integration point still has a timeout and a circuit breaker — so a refactor can never silently remove the bulkhead that keeps one slow vendor from stalling the whole agent loop.
- **Contract tolerance.** A consumer-driven contract test (ch-05) for each SaaS integration, failing the build if the agent starts depending on a field a vendor does not guarantee — the tolerant-reader posture made enforceable.
- **Cost/latency budgets.** A p99 latency or per-task cost monitor on the agent loop with a build-or-alert-failing threshold — the agent's economic characteristics treated as first-class properties to defend, exactly as you would defend availability. For an LLM agent, per-task token cost is not a nice-to-have metric; it is the characteristic most likely to drift silently as prompts grow and tool calls multiply, so it is exactly the kind of property that demands a *continual* fitness function rather than a one-time review.
- **Bounded-context isolation.** An ArchUnit-style check that the agent's bounded contexts (lead/pipeline, conversation, scheduling, CRM-sync from ch-02) do not reach into each other's internals — only across published in-process interfaces. This is the fitness function that *keeps the modular monolith modular*: without it, the seams the strangler would later cut quietly fill in with cross-context calls, and the "extraction is mechanical" promise from ch-04 evaporates. The fitness function here is not protecting today's structure; it is protecting *tomorrow's option to extract* — evolvability guarding evolvability.

Notice the symmetry the agent example makes concrete: the fitness functions above are what keep the *strangler seams clean enough to use later*. A modular monolith whose module boundaries are enforced by a dependency-isolation fitness function is one where every bounded context stays extractable; a modular monolith without that enforcement degrades into a big ball of mud where no clean seam survives, and the strangler has nothing to grab onto. The two halves of this chapter are therefore not independent tools — fitness functions are what *preserve* the conditions under which the strangler stays cheap. Migration discipline and enforcement discipline are one system for keeping bets revisable.

The payoff framing: the previous course taught the learner to *benchmark* an agent; this course teaches them to *architect* one — and this chapter is the part that keeps the architecture from drifting away from the design memo the capstone (ch-10) will produce. A design memo without fitness functions is a snapshot; with them, it is a contract the running system is held to.

---

## Where This Goes

This chapter closed the loop on the spine: architecture is the expensive-to-reverse decisions, and evolution is how you keep "expensive-to-reverse" from collapsing into "irreversible." The strangler-fig makes *changing* a bet incremental and reversible; fitness functions keep an *unchanged* bet from rotting. Both shrink the irreversible set, finally cashing out Fowler's amendment from ch-01.

The single trade-off this chapter centered on is the one that governs the whole evolution phase: **you pay a continuous, ongoing cost — running two systems during a migration, maintaining a permanent façade, writing and tuning fitness functions, failing builds and refusing the ignore-list — to keep your already-placed bets revisable instead of frozen.** Every other pattern in the course bought you a property at the cost of some flexibility; evolution buys back the flexibility itself, and the price is that the bill never stops arriving. Whether that price is worth paying is, like everything else, a First-Law trade-off — for a throwaway script it is pure overhead, for the learner's long-lived production agent it is the difference between an architecture that can outlive its vendors and one that quietly becomes the irreversible wrong answer.

Ch-10 is the capstone lab: it takes the entire toolkit — bounded contexts (ch-02), clean internal structure (ch-03), modular-monolith-first topology (ch-04), integration contracts (ch-05), inside-vs-outside data and sagas (ch-06), CQRS/ES as optional power tools (ch-07), resilience at the integration points (ch-08), and this chapter's strangler-fig-and-fitness-function evolution discipline — and forces every one of them to become a *priced bet recorded as an ADR* plus a *C4 Context/Container sketch* for the production sales agent. The deliverable is a real design memo for Lina TMR: every decision named, every cost stated, and every protected characteristic carrying the fitness function that will keep it honest.
