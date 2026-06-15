<!-- scope: Sam Newman — independent deployability, information hiding, data ownership, migration
     deps: fowler-microservices, fowler-monolith-first
     see-also: decompose-by-business-capability, martin-strangler-fig, distributed-monolith
-->

# Building Microservices / Monolith to Microservices — Sam Newman

- **Core Insight:** The single defining property of a microservice is **independent deployability**; everything else (small size, own database, network APIs) is in service of being able to ship one service without coordinating the release of any other.
- **Guideline:** Make independent deployability the acid test for every boundary decision. Achieve it via aggressive **information hiding** — stable APIs over hidden internals — and **data ownership** — no shared databases, ever.
- **Source:** Sam Newman, *Building Microservices* 2e (2021) and *Monolith to Microservices* (2019) — **books, theses extracted; corroborated by O'Reilly excerpts and Newman's talks.**
- **Relevant chapters:** monolith-vs-microservices, service-decomposition, api-design-contracts.

## The definition (Newman's emphasis)

Microservices are "independently releasable services modeled around a business domain" that "encapsulate functionality and expose it via network endpoints." The phrase that does the work is *independently releasable*.

> "Independent deployability is the single most important principle." — Newman (paraphrased from *Building Microservices* 2e)

It is the *cause*, not a side effect: chasing it "forces loose coupling, well-defined contracts, and stable interfaces."

## Information hiding (the discipline that makes it possible)

A service must "expose behavior through APIs while hiding implementation details like databases, technology choices, and internal workflows." If internals leak (a shared table, an exposed schema), consumers couple to them and independent deployment dies. This is Parnas's information hiding lifted from the module to the service.

## Data ownership (the non-negotiable)

> "Each microservice must own its data. Shared databases create hidden coupling and destroy independent deployability."

Interaction happens "through explicit APIs or events," never through another service's database. (Same conclusion as Richardson's [[transactional-outbox|Database per Service]], reached from the deployability angle rather than the transaction angle.)

## Migration (Monolith to Microservices)

Newman's playbook is explicitly incremental: start from a monolith (often a modular one), find a seam, and use the **[[martin-strangler-fig|Strangler Fig]]** to peel off one capability at a time — extracting the *behavior* and its *data* together so you don't create a shared-DB dependency mid-migration. He is pointedly skeptical of microservices-for-their-own-sake, aligning with [[fowler-monolith-first]].

## The failure mode he names

When services lose independent deployability — must release together, call each other synchronously, share data — you have a [[distributed-monolith]]: "all the pain of distributed systems without the independence." Newman's whole framework is organized to prevent exactly this.

## Connections

- The 9-characteristics companion view → [[fowler-microservices]].
- Where to cut the seams → [[decompose-by-business-capability]].
- How to peel them off a legacy system → [[martin-strangler-fig]].
