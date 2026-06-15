<!-- scope: Strangler Fig migration pattern (Fowler) + incremental modernization framing
     deps: fowler-monolith-first, decompose-by-business-capability
     see-also: newman-building-microservices
-->

# Strangler Fig Application — Incremental Migration (Fowler)

- **Core Insight:** A big-bang rewrite of a serious system almost always fails, because users can't wait years and risk compounds; instead, grow the new system *around* the old one and retire the old piece by piece.
- **Guideline:** Put a façade/interception layer in front of the legacy system; route one capability at a time to a new implementation; delete the old code once nothing calls it. Investment and value accrue gradually and reversibly.
- **Source:** Martin Fowler, "StranglerFigApplication" (martinfowler.com/bliki/StranglerFigApplication.html, 2004/renamed 2019); Sam Newman, *Monolith to Microservices* applies it (book).
- **Relevant chapters:** service-decomposition, monolith-vs-microservices, architectural-decisions.

## The metaphor

Fowler watched strangler figs in Queensland: vines that germinate in a tree's canopy, grow down around the host, and eventually become self-supporting as the original tree dies.

> "Like the fig, it begins with small additions, often new features, that are built on top of, yet separate to the legacy code base." — Fowler

## Why not a big-bang rewrite

> "Replacing a serious IT system takes a long time, and the users can't wait for new features." — Fowler

A rewrite freezes the old system, accumulates risk to one giant cutover, and delivers nothing until the end. The strangler inverts every property of that.

## The advantages (Fowler, verbatim)

- "Investment and returns occur gradually and visibly."
- "Since these components are small, there isn't so much risk involved."
- "The business can reap the value from these new components, allowing earlier return on the investment."
- Each increment teaches you something, "which helps us make better decisions as modernization continues."

Fowler is honest about the limit: "Replacing a software system… is never going to be an easy task" — the pattern makes it *manageable*, not *easy*.

## The mechanism (design altitude)

1. **Intercept** — an HTTP proxy / event interceptor / façade sits between callers and the legacy system so routing can change without callers noticing.
2. **Extract one seam** — pick a [[decompose-by-business-capability|business capability]] with a clean boundary; reimplement it; route its traffic to the new code.
3. **Verify & shrink** — run old and new in parallel where needed, compare, then delete the strangled legacy path.
4. **Repeat** — until the legacy core is gone or small enough to keep.

## Connection to the course thesis

The strangler is the *reversibility* discipline applied to migration: each step is small enough to undo, so a wrong move costs one capability's rework, not the whole rewrite. It is the operational counterpart to [[fowler-monolith-first]] — both refuse to make a single large irreversible bet on boundaries you can't yet trust.

## Connections

- The "don't start distributed" sibling argument → [[fowler-monolith-first]].
- Choosing which seam to strangle first → [[decompose-by-business-capability]].
- Newman's full migration playbook → [[newman-building-microservices]].
