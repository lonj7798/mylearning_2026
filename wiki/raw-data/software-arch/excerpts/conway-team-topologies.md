<!-- scope: Conway's Law + Inverse Conway Maneuver + Team Topologies (Skelton & Pais)
     deps: decompose-by-business-capability
     see-also: fowler-microservices, ddd-bounded-context
-->

# Conway's Law & Team Topologies — Org Structure IS Architecture

- **Core Insight:** Your architecture will mirror your org's communication structure whether you plan it or not (Conway's Law) — so team design is an architectural lever, and **cognitive load**, not headcount, is the real constraint on a team's boundary.
- **Guideline:** Pick the architecture you want, then shape teams to match it (the **Inverse Conway Maneuver**). Size each team's responsibility to its cognitive load; use the four team types and three interaction modes to make the mapping deliberate.
- **Source:** Melvin Conway (1968, original paper) via Martin Fowler "ConwaysLaw" (martinfowler.com/bliki/ConwaysLaw.html); Matthew Skelton & Manuel Pais, *Team Topologies* (2019, book — framework extracted; teamtopologies.com/key-concepts corroborates).
- **Relevant chapters:** service-decomposition, architectural-decisions, monolith-vs-microservices.

## Conway's Law (original, verbatim)

> "Any organization that designs a system (defined broadly) will produce a design whose structure is a copy of the organization's communication structure." — Melvin Conway, 1968

The mechanism (Fowler): "software coupling is enabled and encouraged by human communication." And the corollary: "the modular decomposition of a system and the decomposition of the development organization must be done together."

## Three responses (Fowler)

1. **Ignore** it — it still happens, now by accident.
2. **Accept** it — align architecture to the real communication paths you have.
3. **Inverse Conway Maneuver** — deliberately restructure teams to *induce* the architecture you want. "Particularly effective with microservices organized around business capabilities." This is the lever the learner should know exists.

## Team Topologies — four team types (Skelton & Pais)

1. **Stream-aligned** — owns a single valuable flow of work (a product, a user journey). The default; everyone else exists to support these.
2. **Enabling** — temporarily helps stream-aligned teams acquire a missing capability, then leaves.
3. **Complicated-subsystem** — owns an area of deep specialist complexity, sparing stream-aligned teams the cognitive tax.
4. **Platform** — provides an internal self-service platform that "reduce[s] cognitive load" so stream-aligned teams stay autonomous.

## Three interaction modes

- **Collaboration** — two teams work closely (high bandwidth, high cost; use briefly).
- **X-as-a-Service** — one consumes another with minimal coordination (the steady-state goal).
- **Facilitating** — one team helps another clear impediments (the enabling mode).

## The load-bearing idea: cognitive load

A team boundary should be drawn so the team can hold its whole responsibility in its head. Exceed that and quality and flow collapse — *regardless* of how clean the service boundary looks on paper. This reframes decomposition: it's a **socio-technical** trade-off, not a purely technical one. A service boundary that splits one team's cognitive unit, or fuses two teams', will be fought daily.

## Connections

- The technical decomposition this constrains → [[decompose-by-business-capability]].
- Bounded contexts often = team boundaries → [[ddd-bounded-context]].
- The "organized around business capabilities" characteristic restated socially → [[fowler-microservices]].
