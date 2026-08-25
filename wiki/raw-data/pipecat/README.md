# Raw-data library — `pipecat`

Pre-collected source library for the **Pipecat for a Production Voice Agent**
course (`wiki/courses/pipecat/`). Two source kinds, kept distinct:

1. **Pipecat itself** — official docs, the `pipecat-ai/pipecat` source, and
   provider documentation for STT/TTS/transport services.
2. **The migration target** — `boson-agent` at
   `/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent`. Read-only.
   Excerpts here quote its real modules so chapters can compare mechanisms
   against actual code rather than against a description of it.

## Layout

```
pipecat/
  README.md               ← this file
  COLLECTION-PLAN.md      ← target coverage + gap log
  insights.md             ← one-line core insight per excerpt (index)
  excerpts/
    <slug>.md             ← one source artifact each
```

## Excerpt header schema

Every `excerpts/<slug>.md` follows:

```markdown
# <Human Title>
<!-- slug: <slug> · type: doc|source|paper|module|boson · source: <url or repo path> -->

**Core Insight.** <the one idea this source contributes to the course>

**Guideline.** <the actionable rule a practitioner takes away>

## Technical Details
- <class/function names, real signatures, verbatim-ish key claims, numbers>
- **Migration angle:** <what this means for moving boson-agent onto Pipecat —
  which boson module it replaces, collides with, or leaves untouched>

## Citation
<authors/project, version or commit, date, url or repo path>
```

## Rules

- One artifact per file. Cite from a chapter's `read.md` via `[[slug]]`.
- Prefer **the Pipecat source tree** over prose docs when they disagree; this
  framework moves fast and the docs lag. Record the commit or version read.
- Every excerpt must carry the **Migration angle** line. This library exists to
  answer "what happens to boson-agent", not to document Pipecat in the abstract.
- `boson`-type excerpts quote the learner's own code. Quote real line ranges and
  name the file. Never modify the boson-agent repo from here — it is read-only
  for this course.
- New must-read material found during authoring → add a file here plus a row in
  `insights.md`; do not inline it into a chapter.

## Known constraints to carry into every chapter

- boson-agent today has **no server-side audio**. No STT, no TTS, no VAD. Voice
  is client-side and barge-in reasons over text partial transcripts
  (`gateway/interrupt/detector.py`). Server-side voice is the goal of this
  migration, so several chapters describe a capability being *added*, not ported.
- Lina TMR is a **telephony** product (Korean insurance tele-sales), so SIP /
  Twilio transport and telephony-codec audio quality are load-bearing, not
  optional extras.
- Korean STT accuracy on 8 kHz telephony audio is the single biggest unknown in
  the migration. Record every measured number found.
