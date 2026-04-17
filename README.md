# Wiki Template

A forkable wiki for use with the `jaewon-plugin-learning` Claude Code plugin.

## How to Fork and Use

1. Click **Use this template** on GitHub (or `git clone` and remove `.git/`).
2. Name your fork `<your-name>-learning-wiki` (or any name you prefer).
3. In the plugin repo, set `WIKI_ROOT` in `.jaewon-learning/settings.json` to point at your fork.
4. Run the `setup-learning-wiki` skill in Claude Code to initialize `.jaewon-learning/`.
5. Start a course with the `new-course` skill.

## What Lives Here

```
wiki/
  index.md          — auto-maintained catalog (wiki-maintainer agent)
  log.md            — append-only session log
  learner/          — your learning profile (5 pages, updated by profiler agent)
  courses/          — one folder per course (created by new-course skill)
dashboard/          — generated HTML dashboards (do not hand-edit)
.jaewon-learning/   — plugin runtime state (gitignored sessions/*.tmp)
```

## How the Plugin Uses This Repo

- **SessionStart hook** reads `wiki/learner/*.md` and injects a summary into Claude's system message.
- **Profiler agent** updates `wiki/learner/` after each discuss phase.
- **Wiki-maintainer agent** rebuilds `wiki/index.md` and lints links after every session.
- **Dashboard-builder agent** regenerates `dashboard/*.html`; push to `gh-pages` for a learning dashboard.

## What Never to Edit Manually

- `wiki/courses/*/verdict.json` — written by the evaluator agent only.
- `wiki/index.md` — rebuilt by wiki-maintainer; manual edits are overwritten.
- `dashboard/*.html` — generated output; edit `wiki/` content instead.

## Branching Convention

| Branch | Purpose |
|--------|---------|
| `main` | Canonical learning wiki |
| `course/<slug>` | Active course in progress |

Merge `course/<slug>` into `main` after verdict = `mastery`.

## See Also

- `SCHEMA.md` — wiki page conventions (120-line cap, header format, wikilinks)
- `CLAUDE.md` — workflow guide for the plugin teacher role
