<!-- chapter: ch-02
     track: boundaries
     kind: content
     title: Boundaries by Language: Strategic DDD and Bounded Contexts
     deps: [[ch-01]]
     sources: [[ddd-bounded-context]], [[decompose-by-business-capability]], [[conway-team-topologies]], [[fowler-monolith-first]], [[richards-ford-fundamentals]], [[COLLECTION-PLAN]], [[insights]]
-->

# Chapter 02 — Boundaries by Language: Strategic DDD and Bounded Contexts

> **Core insight.** A boundary is the single most expensive-to-reverse decision in a system, so you must place it on the *slowest-changing* structure you can find — and that structure is the language. Where a word like "Customer" stops meaning the same thing, the model has fractured whether you drew a line there or not. Domain-Driven Design's strategic half is the discipline of finding those linguistic fault lines *first* — before any deployment, framework, or service decision — because a boundary drawn on the language is a boundary you can commit to, while a boundary drawn on anything faster-moving (the org chart, the database schema, the current tech stack) is model rot you pay for every day.

> **Guideline.** Find your Bounded Contexts before you find your services. Walk the domain with the people who speak it, and listen for the seam where a noun changes meaning — that polysemy is the signal to split the model. Inside each context, enforce one Ubiquitous Language without contradiction; between contexts, make the relationship explicit (an Anticorruption Layer, a Shared Kernel, a Conformist mapping) and treat that choice as an organizational commitment, not a wiring detail. Do all of this at the modeling altitude: contexts are modules first, and only later — if a specific pressure justifies the distributed tax — services.

---

## 1. Boundaries Are the Irreversible Decision (Why This Chapter Comes Second)

[[ch-01]] installed the spine: architecture is the set of decisions that are expensive to reverse, and the First Law forces you to price every pattern as a bet ([[richards-ford-fundamentals]], [[insights]]). This chapter cashes that spine on the *first* concrete architectural artifact — the boundary — and it is deliberately the second chapter for one reason: of all the bets in this course, a boundary is the one with the highest reversal cost.

The course's organizing definition, quoted in [[richards-ford-fundamentals]], makes the stakes explicit:

> "Everything in software architecture is a trade-off. If you think you've found something that isn't a trade-off, you likely just haven't found the trade-off yet." — Richards & Ford (First Law, quoted as commonly published; book thesis, [[richards-ford-fundamentals]])

So what is the trade-off a boundary makes? A boundary *keeps cheap* every change that stays inside one side of it — you can rewrite a context's internals, swap its storage, restructure its objects, and nothing on the other side notices. A boundary *makes expensive* every change that crosses it — once two contexts depend on each other's model, moving behavior between them is no longer a refactor. Fowler states the asymmetry precisely in [[fowler-monolith-first]]: inside a single deployable a bad boundary is a refactor, but once you have distributed it, "refactoring functionality between services is much harder than it is in a monolith." Across services it is a *migration*.

That asymmetry is the whole reason strategic DDD exists. If boundaries were cheap to move, you would not need a discipline for placing them — you would just guess and adjust. They are not cheap, so you need a principled way to find the line that will *still be the right line in three years*. DDD's answer: draw it on the language, because the language is the slowest thing in the building.

### 1.1 The slowest-changing-structure principle

This principle threads the entire boundaries phase and is stated bluntly in the decomposition excerpt ([[decompose-by-business-capability]]):

> "Stable architecture since the business capabilities are relatively stable." — Chris Richardson

[[decompose-by-business-capability]] generalizes it: "Org structures churn, tech churns, screens churn — what the business fundamentally does changes slowly. Cut there and your expensive-to-reverse decision lands on solid ground." DDD reaches the same seam from the modeling side rather than the business-architecture side. The two are the same move described in two vocabularies — a point this chapter and ch-04 both lean on.

| Candidate structure to cut on | How fast it changes | Boundary quality |
|---|---|---|
| Current UI / screens | Every release | Worst — you re-cut constantly |
| Tech stack / framework | Every few years | Bad — couples boundary to a vendor bet |
| Org chart | Reorgs happen yearly | Risky alone — but Conway means it *will* shape the architecture anyway (§5) |
| Database schema / technical layer | With every feature | Worst kind — guarantees a [[distributed-monolith]] |
| **Business capability / ubiquitous language** | **Slowly — the business's identity** | **Best — the boundary worth committing to** |

The bet, stated as a bet: **drawing the boundary on the language keeps the cost of intra-context change near zero, at the price of forcing you to do slow, expensive domain conversation up front to find where the language actually breaks.** You pay analysis cost now to avoid migration cost forever.

---

## 2. Why One Unified Model Fails

The starting move of strategic DDD is a refusal. Evans refuses to build one model that covers the whole domain. The thesis, quoted by Fowler in [[ddd-bounded-context]]:

> "Total unification of the domain model for a large system will not be feasible or cost-effective." — Eric Evans

(Evans's *Domain-Driven Design* (2003) is a book; this line survives as a verbatim quote because Fowler reproduces it. Treat the rest of Evans's argument in this chapter as attributed thesis, not as fetched verbatim — see the gap-log hedges in [[COLLECTION-PLAN]] and the excerpt header in [[ddd-bounded-context]].)

The mechanism of failure is linguistic, not technical. As Fowler puts it in [[ddd-bounded-context]], "different groups of people will use subtly different vocabularies in different parts of a large organization," and a single model forced across them accumulates contradictions. The `Customer` class that must serve Sales *and* Support *and* Billing grows conditional fields, mode flags, and "if this came from the support side" branches until no one can reason about it. The model has not unified the domain; it has unified the *contradictions* of the domain into one object.

### 2.1 The failure is a coupling failure in disguise

This connects directly to the coupling/cohesion pillar in [[insights]]: a god-model is maximally coupled because every team that touches the domain touches the same class. A change requested by Support can break Sales, not through a deployment dependency but through a *semantic* one. The unified model looks cohesive (one `Customer`, how tidy) but is the opposite — it is a low-cohesion magnet that pulls unrelated reasons-to-change into one place. DDD's boundary restores cohesion by letting each context have a `Customer` that means exactly one thing.

### 2.2 What model rot actually looks like (worked example)

The abstraction is easy to nod along to; the mechanism is worth watching frame by frame, because it is gradual and each individual step looks reasonable. Start with an honest, single `Customer` for a young sales tool:

```
class Customer:
    id
    name
    email
    pipeline_stage        # Sales cares about this
```

Support ships. Support's `Customer` is an account with tickets and an SLA tier — but there is "already a `Customer` class," so a field is added rather than a model drawn:

```
class Customer:
    id
    name
    email
    pipeline_stage        # meaningless for a Support account
    sla_tier              # meaningless for a Sales lead
    open_ticket_count     # meaningless for a Sales lead
```

Now Billing arrives, and Billing's `Customer` is a legal entity with a tax ID and a payment method. Same move:

```
class Customer:
    ...
    pipeline_stage        # null for Billing
    sla_tier              # null for Sales
    tax_id                # null until they pay
    payment_method        # null for an unqualified lead
    # and: is_active means "in pipeline" to Sales,
    #      "has an open ticket" to Support,
    #      "has a valid card" to Billing
```

The decay is now visible. Half the fields are null for any given use of the object. `is_active` is the killer: three teams gave one boolean three incompatible meanings, so every read needs an out-of-band "which sense did you mean?" The invariants are gone too — Sales wants `pipeline_stage` required, Billing wants it forbidden, so the class can enforce *neither*. This is the contradiction-accumulation Evans predicted, made concrete: the object did not unify the domain, it absorbed every team's notion of `Customer` and lost the ability to be correct about any of them. The DDD fix is not "add more fields cleverly" — it is to recognize three Ubiquitous Languages collided in one class and split them into three contexts. Each then gets a `Customer` that is small, fully-required, and correct, with explicit translation where they meet (§4).

---

## 3. The Bounded Context and the Ubiquitous Language

A Bounded Context is the unit DDD offers in place of the doomed unified model. Fowler's definition, verbatim in [[ddd-bounded-context]]:

> "A Bounded Context is a central pattern in Domain-Driven Design." — Fowler

> "DDD divides up a large system into Bounded Contexts, each of which can have a unified model." — Fowler

The shift is subtle and total: unification does not disappear, it *shrinks*. You give up one model for the whole system and gain one rigorously consistent model *per context*. Inside the boundary, the model is allowed to be strict, complete, and contradiction-free — because it only has to be right about one slice of the domain.

The thing that holds a context together is its **Ubiquitous Language**: a single vocabulary shared by developers and domain experts, where every term has exactly one meaning. [[ddd-bounded-context]] puts the relationship sharply — "the model *is* a shared language between developers and domain experts." That phrasing is load-bearing. The language is not documentation *about* the model; the language and the model are the same artifact viewed from two seats. When a domain expert says "qualified lead" and a developer's class is named `Prospect`, the gap between those two words is a future bug — someone will eventually translate one into the other and translate it wrong. The Ubiquitous Language discipline forecloses that by insisting the code's nouns and verbs *are* the experts' nouns and verbs, with no glossary in between.

This is also why the signal to split is linguistic rather than technical, quoted in [[ddd-bounded-context]]:

> "You need a different model when the language changes." — Fowler

Two consequences follow that an architect actually acts on. First, the boundary is *discovered by listening*, not derived from a diagram — you find it in the room where the domain experts argue about what a word means. Second, enforcing "one term, one meaning" *inside* a context is what lets the model stay strict: because there is no second meaning of `Customer` to defend against, the class can carry required fields and real invariants instead of nullable mode-flags. The strictness inside is *bought* by the boundary outside.

### 3.1 The polysemy signal (the canonical teaching case)

The practical detector is **polysemy** — one word carrying two meanings. [[ddd-bounded-context]] uses the textbook offenders, `Customer` and `Product`, and Fowler's own utility example where "meter" meant "subtly different things to different parts of the organization." The pattern:

| Word | In context A | In context B | Verdict |
|---|---|---|---|
| `Customer` | Sales: a *lead* with a pipeline stage and a probability-to-close | Support: an *account* with open tickets and an SLA tier | Two models, two `Customer` types, explicit translation between |
| `Product` | Catalog: a marketing SKU with descriptions and images | Fulfillment: a physical item with weight and a warehouse bin | Same — do not force one class |

When you hear the same noun used with two different sets of attributes and rules, you have found a context boundary. The discipline is to *stop forcing them into one class* and instead draw two models with a deliberate translation between them. That translation is the subject of §4.

### 3.2 Strategic vs tactical — where this chapter sits

[[ddd-bounded-context]] is careful to scope itself: this is *strategic* DDD — the large-scale boundary and integration decisions. The building blocks *inside* a context (entities, value objects, aggregates, domain events) are *tactical* DDD and belong to ch-03 ([[ddd-aggregates-tactical]], previewed there). The split matters because conflating them is how teams "do DDD" by sprinkling aggregates everywhere while never drawing a single context boundary — getting the expensive part (the boundary) wrong while fussing over the cheap part (the object taxonomy).

---

## 4. Context Mapping: The Relationship Between Boundaries Is Itself a Decision

Drawing boundaries is half the job; the other half is deciding how the contexts on either side *relate*. Two contexts always have to exchange information eventually — Sales hands a closed deal to Fulfillment, Support reads account data Sales owns. The relationship you choose is a strategic decision with organizational consequences, not a wiring detail. [[ddd-bounded-context]] names Vernon's *DDD Distilled* catalog of context-mapping patterns (attributed thesis from the book, not verbatim):

> **Interactive companion:** [`figures/context-map-explorer.html`](figures/context-map-explorer.html) — click each bounded context to see how the same word `Customer` carries a different model in each (the polysemy of §3.1), then select two contexts and a mapping pattern to see the bet that mapping makes: what it keeps cheap and what it makes expensive. It is the spatial version of the table below.


| Mapping pattern | What it means | When you choose it |
|---|---|---|
| **Partnership** | Two contexts succeed or fail together; teams coordinate closely | Mutual dependency, aligned goals, willing to pay coordination cost |
| **Shared Kernel** | A small shared model both contexts depend on | Overlap is small, stable, and cheaper to share than to duplicate — but every change now needs both teams |
| **Customer-Supplier** | Downstream context's needs influence upstream's priorities | Upstream is willing to serve downstream as a customer |
| **Conformist** | Downstream simply adopts the upstream model as-is | You have no leverage over upstream (e.g. a vendor API) and translation isn't worth it |
| **Anticorruption Layer (ACL)** | A translation layer that converts the other context's model into yours so it can't leak in | You must integrate but refuse to let a foreign or messy model pollute your clean one |
| **Open Host Service** | A context publishes a well-defined protocol for many consumers | You are upstream of many and want one stable interface instead of N bespoke ones |
| **Published Language** | A shared, well-documented interchange format | Multiple contexts need a common lingua franca (e.g. a defined event schema) |

### 4.1 The Anticorruption Layer is the load-bearing one for an agent

Of these, the **Anticorruption Layer** is the pattern this learner will use most, and it is worth stating its bet precisely. [[ddd-bounded-context]] defines it as a layer that translates "the other context's model so it can't leak into yours." The bet: **an ACL keeps your core model cheap to evolve — its concepts never get contaminated by an external schema — at the price of writing and maintaining translation code at the boundary.** You pay mapper/DTO cost so that a vendor's data shape can never reach into your domain logic. (This is the strategic-altitude version of the inward-dependency rule that ch-03 makes tactical via [[martin-clean-arch]].)

### 4.2 Mapping choices are org choices

[[ddd-bounded-context]] flags that "choosing the relationship is a strategic decision with org consequences," cross-linking to [[conway-team-topologies]] — which is §5. A Shared Kernel means two teams are now coupled at the model level and must coordinate on every change to it; a Conformist relationship means one team has accepted permanent subordination to another team's model. These are people decisions wearing a diagram.

---

## 5. The Boundary Is Socio-Technical: Conway's Law

You can find the perfect linguistic boundary on a whiteboard and still have the system refuse to honor it — because the *org* gets a vote, whether you invite it or not. Conway's original 1968 thesis, verbatim via Fowler in [[conway-team-topologies]]:

> "Any organization that designs a system (defined broadly) will produce a design whose structure is a copy of the organization's communication structure." — Melvin Conway, 1968

The mechanism, per Fowler in [[conway-team-topologies]]: "software coupling is enabled and encouraged by human communication," with the corollary that "the modular decomposition of a system and the decomposition of the development organization must be done together." A bounded context that splits one team's cognitive unit down the middle, or that fuses two teams into one model, will be fought daily — the communication structure will quietly bend the code back toward its own shape.

### 5.1 Three responses and the lever

[[conway-team-topologies]] lays out Fowler's three responses:

1. **Ignore** it — it still happens, now by accident.
2. **Accept** it — align the architecture to the communication paths you actually have.
3. **Inverse Conway Maneuver** — deliberately restructure teams to *induce* the architecture you want; "particularly effective with microservices organized around business capabilities."

The reason this belongs in a *boundaries* chapter and not a management one: option 3 is an architectural lever. If you have found the right contexts but the org doesn't match them, you can move the org to match the boundary instead of letting the org corrupt the boundary. The constraint that decides whether a team can actually own a context is **cognitive load** — [[conway-team-topologies]] is emphatic that "cognitive load, not headcount, is the real constraint on a team's boundary." A context the team cannot hold in its head will rot regardless of how clean it looks on paper.

The trade-off framing: **aligning teams to contexts (the Inverse Conway Maneuver) keeps each context cheap to evolve in isolation, at the price of a reorg — and reorgs are themselves expensive and disruptive.** It is a real bet, not a free win.

---

## 6. Myth Killed: "DDD Requires Microservices"

This is the doc-vs-reality myth assigned to ch-02 in [[COLLECTION-PLAN]], and the primary source is unambiguous.

| Popular narrative | What the primary source actually says | Resolve in |
|---|---|---|
| "DDD requires microservices." | **False.** DDD is a modeling discipline; bounded contexts can be modules in a modular monolith. No DDD source ties it to a deployment topology. | [[ddd-bounded-context]] |

[[ddd-bounded-context]] states it directly: "DDD is a modeling discipline; it applies equally inside a modular monolith. Bounded contexts can be modules in one deployable." Nothing in Evans, Fowler's bliki entry, or Vernon's *DDD Distilled* ties a bounded context to a separately deployable unit. A context is a boundary *in the model*; whether that boundary is enforced by a module/namespace in one process or by a network hop between two services is a *separate, later, topology* decision — the subject of ch-04.

### 6.1 Why the conflation is dangerous

The myth is not harmless. If you believe "DDD requires microservices," then the moment you do the (correct, cheap, high-value) work of finding bounded contexts, you conclude you must *also* pay the distribution tax of turning each one into a service. That is exactly the premature-decomposition trap [[fowler-monolith-first]] warns against:

> "Almost all the cases where I've heard of a system that was built as a microservice system from scratch, it has ended up in serious trouble." — Fowler

> "Even experienced architects working in familiar domains have great difficulty getting boundaries right at the beginning." — Fowler

The correct sequence decouples the two decisions. **Find contexts now (cheap, reversible — they're just module boundaries); decide topology later (expensive, once usage has shown which seams are real).** [[fowler-monolith-first]] calls the staging ground a modular monolith: "the single deployable unit but enforced strict internal module boundaries… one schema/namespace per module, communication via in-process interfaces, no reaching into another module's tables." Bounded contexts ARE those modules. The myth collapses two decisions of wildly different reversal cost into one, and so leads teams to pay the irreversible cost before they have the information to pay it well.

### 6.2 The symmetric error: too many contexts

The First Law ([[richards-ford-fundamentals]]) forbids treating "draw contexts" as a free good, so name the cost on the *other* side too. Splitting is not costless: every boundary you draw is a translation you now owe (§4). Two contexts that turn out to share one Ubiquitous Language — where `Customer` really did mean the same thing on both sides — were split for nothing, and you now maintain mapper code, duplicated concepts, and cross-context calls that buy you no isolation. This is the over-decomposition failure, and it is the mirror image of the god-model: instead of one object absorbing every meaning, you have ten anemic contexts whose constant chatter reconstructs the coupling you were trying to remove.

The detector is the same one, run in reverse: if the language does *not* change across a candidate seam — if domain experts use the word identically on both sides and no invariant differs — then there is no boundary there, only an arbitrary line. The discipline is to split *exactly* where the language breaks and nowhere else. This is why §7's checklist leads with polysemy and ends with "draw it inside a modular monolith first" ([[fowler-monolith-first]]): a context that is merely a module is cheap to merge back if you over-split, whereas a context that became a service is a migration to undo. Keeping contexts as modules keeps *both* directions of the boundary error cheap to correct.

---

## 7. Pricing the Bet and Finding the Seams in Practice

The whole chapter reduces to one priced bet, stated in the outline's terms: **a bounded context drawn on the slowest-changing structure (the language/domain) is the boundary worth committing to; a wrong one is model rot you pay for daily.**

- **Keeps cheap:** every change confined to one context — its internal model, storage, and objects can churn freely; the Ubiquitous Language inside stays consistent; teams aligned to contexts move independently.
- **Makes expensive:** anything that has to cross a context boundary — you now need an explicit mapping (§4), and once two contexts genuinely depend on each other's model, moving behavior between them stops being a refactor.
- **Costs up front:** slow, human domain conversation to find where the language actually breaks. There is no static-analysis shortcut; polysemy lives in people's heads, not in the code.

### 7.1 The practitioner's seam-finding checklist

Synthesized from [[decompose-by-business-capability]] (which gives the how-to layer):

1. **Listen for polysemy.** Where does one noun carry two attribute sets / two rule sets? ([[ddd-bounded-context]]) That is the loudest signal.
2. **Find clusters of data and behavior that change together** and have few references outside the cluster — high cohesion, low coupling. ([[decompose-by-business-capability]])
3. **Cross-check against business capability.** Does the candidate context correspond to "something that a business does in order to generate value" (Richardson, [[decompose-by-business-capability]])? Capability and subdomain usually converge.
4. **Cross-check against Conway.** Will one team own it without exceeding cognitive load? A seam that cuts across a team's communication path will be fought. ([[conway-team-topologies]])
5. **Reject layer-based splits.** "A UI service, a logic service, a data service" is the anti-decomposition — every business change touches every layer, guaranteeing a [[distributed-monolith]]. ([[decompose-by-business-capability]])
6. **Draw it inside a modular monolith first.** Let usage reveal whether the seam is real before paying to distribute it. ([[fowler-monolith-first]])

---

## 8. Applied to the Sales Agent (Lina TMR)

Lina TMR is an LLM agent acting over many external SaaS tool APIs — the system [[insights]] pre-names as the course's through-line. Strategic DDD is where its architecture actually begins, and the move is to find the agent's bounded contexts *before* deciding anything about deployment.

Running the polysemy detector over the agent's domain surfaces candidate contexts where the language clearly breaks:

| Candidate context | Ubiquitous Language inside it | The polysemy signal |
|---|---|---|
| **Lead / Pipeline** | `Lead`, `Opportunity`, `Stage`, `probability-to-close` | A `Customer` here is a *prospect with a pipeline position* |
| **Conversation** | `Thread`, `Turn`, `Intent`, `Message` | A `Customer` here is *the entity on the other end of a dialogue* — no pipeline meaning at all |
| **Scheduling** | `Meeting`, `Slot`, `Availability`, `Invite` | A `Customer` here is *an attendee with a calendar*, nothing more |
| **CRM-Sync** | `Record`, `FieldMapping`, `SyncState`, `Conflict` | A `Customer` here is *a foreign row in Salesforce/HubSpot* — not even owned by the agent |

That last context is the load-bearing one for this learner. The CRM-Sync context is where every external SaaS API response enters the system, and [[insights]] names the discipline: every external API response is **outside data** — a versioned, possibly-stale snapshot, never authoritative live state. Strategically, that means **CRM-Sync must sit behind an Anticorruption Layer** (§4.1). Salesforce's notion of `Customer` — its field names, its picklist values, its quirks — must be *translated* at the boundary, never allowed to leak into the Lead/Pipeline or Conversation contexts. If a Salesforce schema change can reach into the agent's core domain logic, the boundary failed.

The priced bet for the agent, concretely: **defining these four contexts as modules now keeps the agent's core (how it reasons about a lead, how it conducts a conversation) cheap to evolve independently of whichever CRM, calendar API, or LLM provider is current — at the cost of writing ACL mappers between contexts and resisting the temptation to share one fat `Customer` object across all four.** Note what this chapter deliberately does *not* decide: whether CRM-Sync becomes a separate service. By the myth-killing of §6, that is a later topology decision. Right now CRM-Sync is a context — a module with a clean boundary and an ACL. The agent should default to a modular monolith and only extract a service when a specific pressure justifies the distributed tax ([[fowler-monolith-first]], [[insights]]).

There is a reason CRM-Sync is the strategically load-bearing context for *this* agent specifically, and not just one of four equals. An LLM agent's failure mode is silent contamination: if a Salesforce field's meaning bleeds into the prompt or the domain model, the agent will fluently reason over the wrong concept and produce confident, wrong actions — exactly the "false confidence in incorrect tool calls" failure the learner saw dominate in the agent-benchmark course. The ACL at the CRM-Sync boundary is therefore not bureaucratic indirection; it is the thing that keeps the upstream's churn (a renamed picklist, a new required field, a vendor API version bump) from reaching the part of the system that is expensive to get wrong. This is the strategic seed of two later chapters at once: the inward-dependency rule that makes the ACL a structural discipline (ch-03, [[martin-clean-arch]]) and the inside-vs-outside-data distinction that says every one of those external responses must be treated as an immutable, versioned, possibly-stale snapshot rather than live truth (the consistency phase). The boundary you draw here is the one those chapters build on top of.

---

## Where This Goes

This chapter found *where* to cut — the linguistic seam between contexts — and deferred every deployment question by establishing that a context is a module first, a service maybe-later. Ch-03 goes one zoom level deeper: having drawn the boundary, how do you structure the code *inside* it so the expensive part (domain policy) stays insulated from the cheap parts (vendors, frameworks, the current LLM API)? The answer is the Dependency Rule — "source code dependencies can only point inwards" ([[martin-clean-arch]]) — and the discovery, stated outright by the primary sources, that Hexagonal, Clean, and Onion are the same idea under three names. The Anticorruption Layer you placed at the CRM-Sync boundary in §8 is the strategic preview of that tactical inward-dependency discipline.
