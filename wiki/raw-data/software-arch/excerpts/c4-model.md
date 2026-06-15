<!-- scope: Simon Brown's C4 model — abstractions and the four levels of architecture diagrams
     see-also: richards-ford-fundamentals, nygard-release-it
-->

# The C4 Model — Communicating Architecture (Simon Brown)

- **Core Insight:** Architecture diagrams fail because they mix abstraction levels and invent ad-hoc notation; C4 fixes this by giving a fixed **hierarchy of abstractions** (system → container → component → code) so a diagram is a "map at one zoom level," not a tangle.
- **Guideline:** Diagram at exactly one level at a time. Start at Context (who/what touches the system), zoom to Containers (deployable apps + data stores), then Components inside a container; reach Code only when it earns its keep.
- **Source:** Simon Brown, c4model.com and c4model.com/abstractions; *Software Architecture for Developers* (book).
- **Relevant chapters:** architectural-decisions, service-decomposition, monolith-vs-microservices.

## The pitch (verbatim)

> "An easy to learn, developer friendly approach to software architecture diagramming." — Simon Brown

> "A set of hierarchical abstractions — software systems, containers, components, and code." — Brown

C4 is "notation independent" and "tooling independent" — it constrains *what* you draw at each level, not *how* you draw it.

## The abstraction hierarchy (verbatim)

> "A software system is made up of one or more containers (applications and data stores), each of which contains one or more components, which in turn are implemented by one or more code elements (classes, interfaces, objects, functions, etc)." — Brown

- **Person** — "people (actors, roles, personas, named individuals, etc) use the software systems that we build."
- **Software System** — the highest level; the thing that delivers value to its users.
- **Container** — "applications and data stores": a separately runnable/deployable unit (a server-side app, a SPA, a mobile app, a database, a file system, a message bus). **NOT a Docker container** — the most common misread; the C4 sense predates and is broader than OCI containers.
- **Component** — a grouping of related functionality behind an interface, *inside* a container; "implemented by one or more code elements."
- **Code** — classes/interfaces/functions; the lowest, usually auto-generated (e.g. UML), rarely hand-drawn.

## The four diagrams (zoom levels)

1. **Context** — the system as a box, surrounded by its users and the external systems it talks to. Audience: everyone.
2. **Container** — zoom in: the deployable apps and data stores that make up the system, and how they communicate. Audience: technical + ops.
3. **Component** — zoom into one container: its components and their responsibilities. Audience: developers of that container.
4. **Code** — zoom into one component: UML/class detail. Optional; let the IDE generate it.

## Why it serves the course

C4 is the *notation* for everything else in this library: a Context diagram shows where a [[ddd-bounded-context]] sits; a Container diagram is exactly where a [[distributed-monolith]] becomes visible (count the deploy-coupled containers); Component diagrams reveal whether [[martin-clean-arch|the dependency rule]] holds. It's the shared language for *talking about* the trade-offs the rest of the course makes.

## Connections

- The decisions the diagrams record → [[nygard-release-it]] (ADRs), [[richards-ford-fundamentals]].
- What a "container" boundary should follow → [[decompose-by-business-capability]].
