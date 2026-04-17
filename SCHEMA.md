# Wiki Schema

Conventions for the wiki-maintainer agent and all contributors. Read before every operation.

## Directory Structure

```
wiki/
  index.md        — auto-maintained catalog (wiki-maintainer rebuilds this)
  log.md          — chronological session log (append-only)
  learner/        — learner profile pages (5 stubs; profiler agent updates)
  courses/        — one folder per course (created by new-course skill)
```

## Page Size Limit

Pages MUST stay under 120 lines (~4K tokens). No exceptions.

When a page exceeds 120 lines:
1. Create a hub page (`topic.md`) with header + summary + links to sub-pages.
2. Split content into sub-pages (`topic-overview.md`, `topic-detail.md`, etc.).
3. Each sub-page stays under 120 lines.
4. Update `index.md` with new sub-pages.

## Page Header (Required)

Every wiki page must begin with:

```markdown
# Page Title

<!-- scope: what this page covers (1 line)
     deps: [[prerequisite-page]]
     see-also: [[related-page]]
-->
```

Agents read these lines to decide relevance without consuming the full body.

## Page Naming

- Lowercase with hyphens: `learning-style.md`, `push-tactics.md`
- Descriptive names — the filename tells you what the page covers
- No numbering prefixes — pages are linked, not ordered

## Wikilinks

Use `[[page-name]]` syntax (Obsidian-compatible). The page name matches the
filename without `.md`:

- `[[session-log]]` links to `learner/session-log.md`
- `[[learning-style]]` links to `learner/learning-style.md`

Every page must have at least one outbound wikilink.

## Index Format

`index.md` is a categorized catalog rebuilt by wiki-maintainer:

```markdown
# Wiki Index

## Learner
- [[learning-style]] — preferred learning mode and pace
- [[push-tactics]] — tactic modes for the teaching agent

## Courses
- [[my-course]] — course summary

## Sessions
- [[session-log]] — session history
```

Categories emerge from content — wiki-maintainer decides grouping.

## Log Format

`log.md` is append-only, newest at bottom:

```markdown
# Wiki Log

## [2026-04-16] Session #1 — setup
Files updated: learning-style.md
New pages: none
Index rebuilt.
```

## Single Writer Rule

Only the `wiki-maintainer` agent writes to `wiki/`. All other agents and the
main session read only. Prevents conflicts and ensures consistent style.

Exception: `session-end.mjs` hook appends structured entries to `log.md`
directly (timestamp + counts; no LLM reasoning required).

## Content Guidelines

- **Be concise** — capture what matters, skip boilerplate
- **Link generously** — every concept that has a page gets a `[[wikilink]]`
- **Capture the why** — decisions and tradeoffs are more valuable than descriptions
- **Evolve, do not replace** — add evolution sections, preserving history
