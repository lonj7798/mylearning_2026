---
title: "Capstone: Keep or Replace, Subsystem by Subsystem"
chapter: ch-13
phase: decision
course: pipecat
sources:
  - rtv-vs-pipecat-gap
  - rtv-pipeline-session
  - rtv-webrtc-transport
  - rtv-vad-chunking
  - design-boson-rules-on-pipecat
  - deployment-scaling
  - boson-compact-session
  - pipecat-design-philosophy
  - theory-narrow-waist
  - stt-korean-providers
  - tts-korean-providers
  - flows-insurance-example
  - pipeline-task-runner
deps:
  - ch-03
  - ch-05
  - ch-09
  - ch-11
  - ch-12
figure: figures/migration-map.html
---

# Chapter 13 — Capstone: Keep or Replace, Subsystem by Subsystem

> **Scope, stated up front and enforced for the whole chapter.**
>
> **This is the only place in this course where a vote is cast, and you cast it.** Twelve chapters
> produced mechanism and evidence and deliberately withheld verdicts. [[ch-09/read]] §11 costed three
> resolutions and marked none recommended. [[ch-05/read]] §10.3 laid out two columns and stopped.
> [[ch-03/read]] §0 told you the two evidence classes and refused to score them. All of that was
> deferred to here.
>
> **The chapter is graded on being DECIDABLE.** That is a hard requirement, not a style note. An
> earlier draft of this capstone conditioned its answer on four numbers the course itself proves
> nobody has, and its honest output was *a plan to decide*. Nothing in a plan-to-decide can be
> graded. §1 gives you the three rules that fix that.
>
> **I do not answer any row for you.** §3 states seventeen rows and, for each, an evidence digest
> naming which chapter supplied which mechanism fact. The vote cell is empty in every one. There is
> no `KEEP` / `ADOPT` / `HYBRID-WRAP` label anywhere in §3 that you did not write yourself, and no
> comparative adjective — no *better*, no *wins*, no *should use*, no *the right choice*. This
> invariant was broken twice while the outline was under review. It is not broken here.
>
> **Some of your source material does contain verdicts, and I strip them.** [[rtv-vs-pipecat-gap]]
> says "Pipecat strictly better here" about VAD and prints a five-point Replace/Keep list in its
> migration angle. [[rtv-vad-chunking]] calls the two-frame blip "the sharpest correctness delta in
> the whole comparison." Those are the excerpt author's votes, not evidence, and they are not
> reproduced as this chapter's position. Where an excerpt states a *mechanism*, I use it and cite it.
> Where it states a *preference*, I drop it and say so.

---

## 왜 이 챕터인가

You have been asked twelve times to look at a mechanism and not decide. That is over.

The reason it was over-ruled for twelve chapters is that a build-vs-buy decision made on a
framework's reputation is worthless, and a build-vs-buy decision made subsystem by subsystem on
file-and-line evidence is the most valuable artifact this whole course can produce. You are not
choosing between "Pipecat" and "realtime_voice." You are choosing seventeen times, and the seventeen
answers do not have to agree with each other. [[rtv-vs-pipecat-gap]]'s own guideline says it in one
line: *"Decide per-layer, not per-framework."*

The reason it is over *now* is that you have run out of mechanism to learn. The remaining unknowns
are not readable — they are measurable, and there are exactly four of them, and three of them cannot
be measured on this machine. §1.4 names all four. §2 runs the one that can.

So this chapter has an unusual shape. It is mostly a **ledger**, not an exposition. The exposition
already happened. What is left is:

1. three rules that make a row answerable (§1),
2. one script you actually run, whose two printed numbers change one row from pending to decided (§2),
3. seventeen evidence digests with empty vote cells (§3),
4. the two collisions that cannot be resolved by reading anything (§4),
5. the things with no Pipecat home and the things Pipecat hands back, including compaction, routed
   here from [[ch-09/read]] (§5, §6),
6. deployment and process topology, which is row 17 and not a postscript (§7),
7. the deliverables and the watchlist (§9, §10).

Your strongest mode is framework extension — taking a mechanism and asking what it would do in a
place its author did not consider. Seventeen rows is seventeen opportunities to do exactly that, and
the assumption cell is where the extension goes. A vote with no assumption written next to it is a
guess wearing a vote's clothes.

---

## 0. How to read the evidence in this chapter

Same two classes as [[ch-03/read]] §0, with one addition.

| Class | Source of truth | How you check it |
|---|---|---|
| **Pipecat claims** — paths, line numbers, class names, counts | `wiki/raw-data/pipecat/pipecat-src` at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` | Open the file. Every number below was re-measured against that tree on 2026-08-25 and the command is printed next to it. |
| **boson-agent / realtime_voice claims** — LOC, class names, defaults | the `rtv-*` / `boson-*` / `design-*` excerpts under `wiki/raw-data/pipecat/excerpts/` | Check against your own repo. Not checkable from this wiki, and nothing here pretends otherwise. |
| **Measured claims** — the P50/P95 that §2 produces | your own endpoint, at the moment you run the probe | Re-run it. It is the only number in this course that did not exist before you made it. |

**Where an excerpt disagrees with the source, the source wins and I say so.** Three such
disagreements are carried in this chapter and each is flagged at the point of use:

1. **Deepgram TTFS P99.** [[design-boson-rules-on-pipecat]] §3 writes *"Pipecat's own STT TTFS P99
   reference is 0.45 s for Deepgram."* The file says `0.35`. Corrected in §3.2 and §2.5, because the
   comparison "Tier-2 roughly doubles the pre-LLM half of the budget" is sensitive to it.
2. **VAD chunk counts.** [[rtv-vad-chunking]] writes *"`start_secs = 0.2` → 7 chunks @16 kHz."*
   `round(0.2 / (512/16000))` is `round(6.25)` = **6**. Corrected in §3.1.
3. **realtime_voice WebRTC size.** [[rtv-webrtc-transport]] headlines *"~960 LOC"* and then lists
   files summing to **1,276**. [[ch-05/read]] §10.1 caught this first; row 6 carries 1,276 and marks
   ~960 unverified.

A fourth number is *soft* rather than wrong. [[rtv-vs-pipecat-gap]] counts **12** Pipecat transports;
`ls -d src/pipecat/transports/*/` returns **11** packages, which [[ch-05/read]] §2 resolved as
11 packages / 13 `BaseTransport` subclasses / one package (`whatsapp/`) with zero. Row 6 uses ch-05's
numbers.

**One caveat that applies to every boson row.** The boson snapshots are dated 2026-07-29
(`realtime_voice`, branch `voice-chat-dev`) and 2026-08-20 (`gateway`/`basement`, branch
`lina-new-dental-dev`). The Pipecat tree is 2026-08-25. Your repo is active. Every boson number in
this ledger is a floor, not a current reading. If a row's vote turns on a boson LOC count, re-measure
before you write the vote — that takes one `wc -l`.

---

## 1. The three rules

### 1.1 Why the rules exist: the failure mode they were written against

The draft this chapter replaces produced, for row 2 (ASR), something like:

> *"Whether to adopt a Pipecat streaming STT depends on Korean word-error rate on 8 kHz μ-law audio,
> which is unmeasured. Recommend benchmarking before deciding."*

Read that as a work product. It is true, it is honest, it is well-sourced — and it is **not a
decision**. It cannot be wrong, so it cannot be graded, so it cannot be acted on. Multiply it by
seventeen and the capstone's output is a list of benchmarks nobody has budget to run, which is the
same as no output.

The problem is not the unknown. The problem is treating an unknown as a **blocker** rather than as a
**falsifier**. Those are different objects:

| | Blocker | Falsifier |
|---|---|---|
| What it does to the decision | postpones it | records what would reverse it |
| What you ship this quarter | nothing | the decision, provisionally |
| What it costs to be wrong | unbounded (you never find out) | bounded (you wrote down the tripwire) |
| Who owns it | nobody, by construction | a named person, with a precondition |

Every real engineering decision you have ever shipped was made this way. You did not measure the
Korean STT before writing `OpenAICompatibleUnaryASR`; you assumed something and shipped it. The only
thing missing was writing the assumption down. That is all rules one and two are.

### 1.2 RULE ONE — every row gets a COMMITMENT, made under EXPLICITLY STATED assumptions

Not "it depends." A vote, plus the sentence **"I am assuming X."**

The form is fixed, and the second half is not optional:

```
Row 7 (telephony serializer):  <vote>
  I am assuming: the Korean carrier we sign with speaks a Twilio-shaped
  bidirectional-media WebSocket JSON protocol, not a raw SIP/RTP leg.
```

The assumption is doing real work. It is what makes the vote *reviewable* by someone who was not in
the room: a colleague who thinks the carrier will hand you SIP/RTP does not have to argue with your
vote, they argue with your assumption, which is a shorter and much more productive fight.

**What counts as an assumption.** A statement about the world that is currently unverified and that,
if it flipped, would change the vote. Three tests:

- It must be **about the world**, not about your preference. *"I am assuming we prefer fewer
  dependencies"* is not an assumption; it is a taste. *"I am assuming Lina ships on a carrier in the
  six-serializer set"* is an assumption.
- It must be **currently unverified**. If you can check it in five minutes, check it and delete the
  assumption. Assumptions are for what you cannot cheaply know, not for what you did not bother to
  look up.
- It must be **load-bearing**. If the vote survives the assumption flipping, the assumption is
  decoration. Delete it and find the real one.

**Rows may share assumptions, and several of these will.** "Lina goes to telephony within two
quarters" is load-bearing for rows 1, 2, 3, 6, 7 and 17 simultaneously. Write it once, reference it
six times, and notice what you have just learned: six of your seventeen votes are correlated, so
that one assumption is worth more verification effort than any other single thing in the ledger.

### 1.3 RULE TWO — every row gets a written FALSIFIER

The **specific observation** that would overturn that vote.

Here is the shape, using a deliberately unowned example so the illustration does not steer any of
your rows. A vote of the form *"adopt X for row N"* is falsified by:

> *on our own corpus, X produces more false turn-starts than what we run today*

and a vote of the form *"keep Y for row N"* is falsified by:

> *the measurement shows Y costs more than the budget allows*

Look at what both of those have in common. **Both name an OBSERVATION, not a preference.** Neither
says "if the team dislikes it" or "if it turns out to be ugly." Each one is a thing that either
happens or does not happen when you look, and looking is a defined act.

**The test that makes the rule bite: if a row cannot be falsified by any observation, the vote is not
a decision and the row is mis-stated.** This is not a stylistic complaint. It is a diagnosis, and the
cure is to go back and re-state the row. Two ways rows fail it:

1. **The row is a taste, not a decision.** "Adopt Pipecat's frame taxonomy because it is more
   elegant" has no falsifier because elegance is not observed. Re-state the row around what the
   taxonomy *does* — [[ch-02/read]] gave you the number: 577 `isinstance(frame, ...)` sites across
   136 files, which is an observable cost of adding a frame type, and now the row is falsifiable
   ("our port needs more than four new frame classes").
2. **The row is two rows.** If you cannot write one falsifier because there are two different
   observations that would each overturn it for different reasons, you have merged two subsystems.
   Split them. That is exactly why compaction is row 15 and the LLM loop is row 10 rather than one
   row: [[ch-09/read]]'s scope box says it in as many words — *"a different subsystem with different
   failure modes and a different trigger."*

**Each falsifier also names the observation TYPE and whether that observation is available today.**
Four types cover everything in this ledger:

| Type | What it is | Example in this ledger |
|---|---|---|
| **corpus** | replay recorded Lina audio or transcripts and count something | false turn-starts on lone Korean backchannels (row 1) |
| **probe** | a script you run against a live endpoint | §2's Tier-2 P50/P95 (row 13) |
| **benchmark** | a third-party or self-run accuracy/latency harness | Korean 8 kHz μ-law WER (row 2) |
| **code assertion** | a test that fails if the claim is false | exactly one inference-triggering frame per turn (rows 10, 12) |

The figure asks you for this type per row, and marks whether it is available today. That last flag is
the honest part: most of them are not, and a falsifier you cannot currently observe is still a
falsifier — it is a tripwire waiting for its precondition. §10 collects them.

### 1.4 RULE THREE — exactly ONE measurement is actually RUN

Not listed. Run.

The course produced four numbers nobody has. Here they are with what each would cost to obtain:

| # | The unknown | What it needs before it can be measured | Runnable today? |
|---|---|---|---|
| 1 | Korean STT accuracy (WER) on 8 kHz μ-law | a Korean STT contract, a labelled Lina corpus, telephony-band audio | **no** |
| 2 | interruption-broadcast → `TranscriptionFrame` gap | recorded 8 kHz telephony audio of lone Korean backchannels, plus a running Pipecat pipeline | **no** |
| 3 | **Tier-2 rule-evaluation latency against a real endpoint** | a model endpoint and two text prompts | **YES** |
| 4 | one-inference-per-turn under the transition-swallow design | a built `BosonRuleProcessor` prototype and a `FrameLogger` | **no** |

Number 3 is the one, and it was chosen for exactly one reason: **it needs no telephony carrier, no
Korean STT contract, and no audio at all.** It is two chat completions. Everything else about it —
that it decides row 13, that it feeds [[ch-11/read]]'s budget — is true but secondary. What makes it
the measurement is that it is the only one you can finish this afternoon.

**And it has a real executable home, which is a hard requirement.** The figure for this chapter,
[`figures/migration-map.html`](figures/migration-map.html), is a self-contained offline page loaded
from `file://`. It cannot call a model endpoint — not because of a design choice, but because the
course's figure contract makes every companion self-contained with no external requests, and because
a `file://` page has no origin to make a cross-origin POST from. **Never assign a job to an artifact
that structurally cannot do it.** So the probe is a sibling script:

```
wiki/courses/pipecat/ch-13/tier2-probe.py
```

and the figure does no network I/O of any kind. It shows the command, takes two pasted numbers, and
renders. That split is the correct one and it generalises: measurement lives where a process can run;
visualisation lives where a browser can render.

---

## 2. The one measurement: `tier2-probe.py`

### 2.1 What it measures, and why that is the right thing to measure

[[ch-12/read]] derived the seam and then presented the bill. From
[[design-boson-rules-on-pipecat]], verbatim:

> *"Tier 2 = the 2 LLM checks — **blocking, and this is the bill: ~250-400 ms of added pre-LLM
> latency on every turn** (one Qwen3.6-27B TTFT plus ~5 output tokens)."*

Two things about that sentence matter here. First, **it is an estimate.** It is arithmetic over a
model's expected TTFT, not a reading off a clock. Second, **row 13 hangs off it**, and so does the
"rule evaluation" slot in [[ch-11/read]]'s waterfall, which is empty today.

The two rules are `intent_rules` (priority 30) and `sentiment_tracker` (priority 10), both stamped
`@check(..., mode="parallel", check_type="llm")` in Lina's `03-orchestrator` layer. The mechanism
that decides what the probe must reproduce is in [[design-boson-rules-on-pipecat]] §1:

> *"parallel checks all run under one `asyncio.gather` (L74-80) and every non-continue result is
> kept"*

and §3:

> *"`sentiment_tracker` fires concurrently under the same `gather`, so wall clock ≈ max, not sum."*

**So the unit of measurement is one `asyncio.gather` of two completions, not two sequential
completions.** A probe that timed them one after another would report roughly double the truth. This
is the whole reason the probe is a script and not a stopwatch on a curl command.

### 2.2 The probe, and the four things it does deliberately

The file is `ch-13/tier2-probe.py`, 149 lines, standard library only. Four design points, each of
which is a decision you could have made differently:

**One — it uses `urllib.request` inside `asyncio.to_thread`, not an async HTTP client.** No
dependency, so the script runs anywhere Python 3.11 does, including a box where you have not created
a venv. The cost is one thread per request; at two concurrent requests that is free, and the timing
overhead is microseconds against a hundreds-of-milliseconds signal.

```python
# wiki/courses/pipecat/ch-13/tier2-probe.py L76-83
async def one_turn(endpoint, api_key, model, turn, timeout):
    """Wall clock of the parallel phase for one finished utterance, in ms."""
    started = time.perf_counter()
    await asyncio.gather(
        asyncio.to_thread(_post, endpoint, api_key, model, INTENT_SYSTEM, turn, timeout),
        asyncio.to_thread(_post, endpoint, api_key, model, SENTIMENT_SYSTEM, turn, timeout),
    )
    return (time.perf_counter() - started) * 1000.0
```

That is the `asyncio.gather` from `rules/engine.py` L74-80, reproduced. The clock starts before the
gather and stops after it, so what you get is exactly the wall-clock window boson's Phase-1 parallel
block occupies — the thing that lands on the pre-LLM critical path.

**Two — the prompts are shaped like the real ones, and `max_tokens=16`.** The intent prompt carries
the `"Most recent turn (PRIMARY SIGNAL — evaluate against THESE)"` anchor and the comma-separated-
index output contract from `intent_matcher.py` L205-271; `temperature=0.1` matches `llm_config.py`
L20,34. Output length matters more than you would expect: the bill is *TTFT plus ~5 output tokens*,
so a probe that let the model write a paragraph would measure the wrong thing.

**Three — the corpus is real Lina user turns, and you should replace it.** Eight Korean turns ship as
the default so the script runs with zero setup. They are representative in *shape* — a backchannel, a
price question, a deflection, a consent, a DNC — but they are not your traffic. `--corpus <file>`
takes one turn per line. **Use your own.** Token count drives TTFT, and if your real turns are
longer than these, your P95 is higher than what the default prints.

**Four — it degrades loudly and refuses to let you fake the number.**

```python
# wiki/courses/pipecat/ch-13/tier2-probe.py L109-115 — the message text as printed
tier2-probe: UNMEASURED. The endpoint did not answer.
  endpoint : ...
  model    : ...
  failed on: iteration 1/40, turn '네 말씀하세요'
  reason   : URLError: <urlopen error [Errno 61] Connection refused>
Bring the endpoint up, or point TIER2_ENDPOINT at a reachable one, and
re-run. Do NOT paste an estimate into figures/migration-map.html — row 13
stays pending until this prints two real numbers.
```

Exit code 1, nothing on stdout. The last sentence is the point of the whole rule-three exercise: the
value of this measurement comes entirely from it being an observation. A pasted estimate is worse
than an empty field, because an empty field is honest.

### 2.3 Percentiles on small N, worked before the formula

You are going to run this forty times, not forty thousand, and percentile estimators disagree at
small N in ways that matter. Work it once by hand.

Take nine samples, sorted, in milliseconds:

```
 rank:    1     2     3     4     5     6     7     8     9
 value: 240   258   261   270   288   301   319   355   612
```

**P50 by nearest rank.** Rank = ⌈9 × 50/100⌉ = ⌈4.5⌉ = 5. The 5th value: **288 ms**.

**P95 by nearest rank.** Rank = ⌈9 × 95/100⌉ = ⌈8.55⌉ = 9. The 9th value: **612 ms**.

Notice what just happened. With nine samples, **P95 is the maximum** — there is no ninth-and-a-half
sample to interpolate toward, so the estimator can only return the worst one you saw. That single
612 ms outlier (a cold cache, a scheduling hiccup, a retried connection) *is* your P95. Only the
formula:

$$\text{rank} = \left\lceil \frac{N \cdot q}{100} \right\rceil, \qquad P_q = x_{(\text{rank})}$$

with `x` the sorted samples. In the script:

```python
# wiki/courses/pipecat/ch-13/tier2-probe.py L86-90
def percentile(samples, q):
    """Nearest-rank percentile — honest for the small N this probe produces."""
    ordered = sorted(samples)
    rank = max(1, min(len(ordered), -(-len(ordered) * q // 100)))
    return ordered[int(rank) - 1]
```

`-(-a // b)` is integer ceiling division. Nearest rank was chosen over linear interpolation on
purpose: interpolation invents a value between two samples you actually saw, which reads as more
precision than forty samples support.

**Practical consequence: run at least 40 iterations.** At N=40, P95 is rank 38 — the third-worst
sample, not the worst — so one outlier no longer *is* your P95. At N=20, P95 is rank 19, the
second-worst. Below about 20 the number is a description of your worst luck rather than of your tail.

### 2.4 Running it

```bash
$ cd wiki/courses/pipecat/ch-13

$ export TIER2_ENDPOINT="http://localhost:8000/v1/chat/completions"
$ export TIER2_MODEL="Qwen3.6-27B-FP8"
$ export TIER2_API_KEY="..."           # omit if your endpoint is unauthenticated

$ python3 tier2-probe.py --iterations 40
tier2-probe: 40 turns against Qwen3.6-27B-FP8 at http://localhost:8000/v1/chat/completions   [min 231.4 / max 601.8 ms]
  P50     288.7 ms
  P95     441.2 ms
Paste P50 and P95 into the MEASUREMENT panel of figures/migration-map.html.
```

Positional form works too, for a one-liner in a notebook or a CI step:

```bash
$ python3 tier2-probe.py http://localhost:8000/v1/chat/completions Qwen3.6-27B-FP8 40
$ python3 tier2-probe.py --corpus ./lina-turns.txt --verbose --iterations 60
```

`--verbose` prints per-turn timings to **stderr**, so stdout stays exactly four lines and stays
pipeable. The numbers above are illustrative formatting, not a result — the result is whatever your
endpoint prints, and that is the whole point.

> **⚠️ The two numbers in the block above are made up to show the output shape.** Do not paste them
> into the figure. Row 13 stays pending until *your* endpoint prints *your* two numbers.

### 2.5 What the number decides, in both directions

Paste P50 and P95 into the MEASUREMENT panel of
[`figures/migration-map.html`](figures/migration-map.html). The panel is greyed and shows only the
run instruction until both fields are filled — an explicit UNMEASURED state, so an unrun probe cannot
be mistaken for a measured zero. Once filled it draws your measured bar with ch-12's 250–400 ms
estimate as a ghost bar behind it, and writes the measured value into the previously empty "rule
evaluation" slot of [[ch-11/read]]'s waterfall so the whole budget re-renders with a real number in
it. Both values persist in `localStorage`, so a reload does not discard the one measurement this
capstone is graded on.

Then read the result. There are three outcomes and all three are decisive:

**Inside 250–400 ms.** ch-12's estimate held. Row 13's latency term is what the design assumed, the
in-turn-veto-versus-next-turn-transition trade is priced as stated, and the row turns on the
*structural* question (how much of boson survives unedited) rather than on the budget. Your falsifier
for row 13 becomes a re-measurement trigger — "re-run when the model or the serving stack changes" —
not an open question.

**Below 250 ms.** The estimate was pessimistic. Note where the slack came from before you spend it:
a warmed KV cache, a quantised checkpoint, or a local endpoint with no network hop are all real and
all fragile in different ways. Ask specifically whether the production path has the same hop count as
the probe's, because the probe talks to whatever `TIER2_ENDPOINT` points at, and if that is
`localhost` you have measured a deployment you do not run.

**Above 400 ms.** The estimate was optimistic and one of [[ch-11/read]]'s budget lines is now
over-drawn by an amount you can name. This is where the correction from §0 earns its keep. The design
excerpt compared Tier-2 against *"Pipecat's own STT TTFS P99 reference is 0.45 s for Deepgram"* —
but the file says:

```python
# src/pipecat/services/stt_latency.py L38, L45, L61-62
DEFAULT_TTFS_P99: float = 1.0
...
DEEPGRAM_TTFS_P99: float = 0.35
...
SONIOX_TTFS_P99: float = 0.35
SPEECHMATICS_TTFS_P99: float = 0.74
```

`0.35`, not `0.45`. The whole table is 23 measured `*_TTFS_P99` constants spanning `0.35`
(Deepgram, Deepgram-SageMaker, Soniox) to `2.14` (xAI), plus `DEFAULT_TTFS_P99 = 1.0` and two
services (`NVIDIA_TTFS_P99`, `WHISPER_TTFS_P99`) that alias to it. The correction sharpens the
comparison rather than softening it: against a 0.35 s reference, a Tier-2 measurement of 400 ms is
larger than the entire STT P99 of the fastest provider in the tree, and the "Tier-2 roughly doubles
the pre-LLM half of the budget" framing understates it.

And note what `stt_latency.py` does **not** say, from [[stt-korean-providers]]: it *"records only
latency, is silent on the language and sample rate of the benchmark audio."* So 0.35 s is an
English-assumed number. Row 2's falsifier lives in that sentence.

---

## 3. The seventeen rows

**How to use this section.** Each row states what each side implements and which chapter established
it. **The vote cell is empty. The assumption cell is empty. The falsifier cell is empty.** All three
are yours, in the figure. The prose stops at the evidence — deliberately, because rule one demands
that you commit, and a chapter that commits for you has cancelled the only graded exercise in the
course.

Here is the ledger in one view. Seventeen rows, in the order the figure renders them.

| # | Subsystem | Evidence from | Vote | Assumption | Falsifier |
|---|---|---|---|---|---|
| 1 | VAD | [[ch-06/read]] | | | |
| 2 | ASR / streaming STT | [[ch-03/read]], [[ch-06/read]], [[ch-11/read]] | | | |
| 3 | TTS + Korean word timestamps | [[ch-07/read]] | | | |
| 4 | `KoreanPhraseChunker` | [[ch-03/read]], [[ch-07/read]] | | | |
| 5 | `AudioTextPlayoutLedger` | [[ch-03/read]], [[ch-07/read]], [[ch-08/read]] | | | |
| 6 | Transport | [[ch-05/read]] | | | |
| 7 | Telephony serializer | [[ch-05/read]], [[ch-06/read]] | | | |
| 8 | Session auth (`WebRTCSessionManager`) | [[ch-03/read]], [[ch-05/read]] | | | |
| 9 | Control protocol (`ControlEvent`) | [[ch-03/read]], [[ch-05/read]] | | | |
| 10 | The LLM loop | [[ch-09/read]] | | | |
| 11 | Tools | [[ch-09/read]] | | | |
| 12 | The stage machine | [[ch-10/read]] | | | |
| 13 | The rule layers | [[ch-12/read]] + §2's probe | | | |
| 14 | `ScriptEngine` | [[ch-12/read]] | | | |
| 15 | Compaction | [[boson-compact-session]] | | | |
| 16 | Observability | [[ch-11/read]] | | | |
| 17 | Deployment and process topology | [[ch-04/read]] + §7 | | | |

Seventeen. The figure renders seventeen. If you ever count sixteen or eighteen in either place, one
of them has drifted and the ledger is no longer a ledger.

---

### 3.1 Row 1 — VAD

[[ch-06/read]] supplied both mechanisms in full.

**Pipecat's analyzer.** Four states, and a confidence gate ANDed with a volume gate:

```python
# src/pipecat/audio/vad/vad_analyzer.py L25-28
VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6
```

```python
# src/pipecat/audio/vad/vad_analyzer.py L41-44
    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4
```

```python
# src/pipecat/audio/vad/vad_analyzer.py L206-232
            confidence = self.voice_confidence(audio_frames)

            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume

            if speaking:
                match self._vad_state:
                    case VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._vad_starting_count = 1
                    case VADState.STARTING:
                        self._vad_starting_count += 1
                    case VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
            else:
                match self._vad_state:
                    case VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._vad_starting_count = 0
                    case VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._vad_stopping_count = 1
                    case VADState.STOPPING:
                        self._vad_stopping_count += 1
```

The `STARTING → QUIET` arm on line 226 is the one to hold onto: a frame that entered `STARTING` and
does not sustain returns to `QUIET` without ever having emitted a speech-start.

**Native 8 kHz, with the same chunk counts at both rates.** The frame arithmetic:

```python
# src/pipecat/audio/vad/vad_analyzer.py L159-165
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        vad_frames_per_sec = self._vad_frames / self.sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
```

```python
# src/pipecat/audio/vad/silero.py L191-197
    def num_frames_required(self) -> int:
        """Get the number of audio frames required for VAD analysis.

        Returns:
            Number of frames required (512 for 16kHz, 256 for 8kHz).
        """
        return 512 if self.sample_rate == 16000 else 256
```

At 16 kHz: `512/16000 = 0.032` s per chunk, `round(0.2/0.032) = round(6.25) = 6`. At 8 kHz:
`256/8000 = 0.032` s per chunk — **identical**, so `6` again. Endpointing latency is invariant to the
sample rate by construction. **⚠️ Source correction:** [[rtv-vad-chunking]] states *"7 chunks
@16 kHz"*; `round(6.25)` in Python is `6` (banker's rounding is irrelevant here — 6.25 rounds down
under any convention). The value is **six**.

**Idle timeout.** `audio_idle_timeout: float = 1.0` (`llm_response_universal.py:170`, also
`processors/audio/vad_processor.py:46`) forces a speech stop when audio stops arriving at all.

**realtime_voice's analyzer**, per [[rtv-vad-chunking]]: `SileroVADConfig.threshold = 0.5`, one
`self._speaking` bool (two states), `min_speech_frames = 2` / `min_silence_frames = 6` counted in
*transport-sized frames* rather than seconds, no volume gate, no idle timeout, and
`SileroVAD.process` raising `ValueError("SileroVAD requires 16 kHz mono PCM")` at L58.
`EnergyVADConfig` is the sibling: `speech_rms = 500.0`, pure-Python RMS, documented as *"intended for
fallback and deterministic tests."*

**The two-frame blip, as a mechanism.** A 2-frame noise burst reaching `min_speech_frames = 2` in the
two-state machine emits `SPEECH_STARTED`, which in `VoiceSession._on_speech_started` (L284) advances
the generation and cancels the assistant. The same burst in the four-state machine enters `STARTING`,
fails `_vad_starting_count >= 6`, and is returned to `QUIET` by line 226 without emitting anything.
[[rtv-vad-chunking]] calls this "the sharpest correctness delta in the whole comparison" — **that is
the excerpt's verdict and it is not reproduced here.** The mechanism is: the two machines diverge on
short bursts, and the divergence direction is that one emits and the other does not.

**What the row turns on.** How often lone short bursts occur in Lina's actual audio, and what a
sample rate of 8 kHz does to each machine.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.2 Row 2 — ASR / streaming STT

Three chapters contributed and they contributed different kinds of fact.

**[[ch-03/read]] supplied the shape of what runs today.** Per [[rtv-vs-pipecat-gap]]:
`OpenAICompatibleUnaryASR` *"buffers the whole utterance into a WAV and does one
`audio.transcriptions.create` at `finalize()` (`openai_compat.py` L194-242,
`timeout_seconds=1.5`)"*, and `ASREventKind.INTERIM` / `END_OF_TURN` are *"declared in `types.py` but
never emitted by any real provider — only by a test fake."*

**[[ch-06/read]] supplied the Pipecat side**: a streaming `STTService` interface, ~20 STT providers,
the smart-turn default, and the latency table:

```python
# src/pipecat/services/stt_latency.py L38-46
DEFAULT_TTFS_P99: float = 1.0

# Measured P99 TTFS latency values (in seconds)
ASSEMBLYAI_TTFS_P99: float = 0.42
AWS_TRANSCRIBE_TTFS_P99: float = 1.90
AZURE_TTFS_P99: float = 1.80
CARTESIA_TTFS_P99: float = 0.81
DEEPGRAM_TTFS_P99: float = 0.35
DEEPGRAM_SAGEMAKER_TTFS_P99: float = 0.35
```

23 measured constants, `0.35` → `2.14`. **⚠️ Source correction, repeated from §2.5 because it lands
on this row too:** [[design-boson-rules-on-pipecat]] cites Deepgram at `0.45`; the file says `0.35`.

**[[ch-11/read]] supplied the arithmetic** that places a unary transcription RTT *wholly* after VAD
stop on the critical path — a serial term, not an overlapped one — against `CLAUDE.md`'s own line,
quoted in [[rtv-vs-pipecat-gap]]: *"P50 at or below 1.0 seconds and P95 at or below 1.5 seconds,"*
measured *"from the last voiced user sample to the first audible assistant sample, including
end-of-turn/VAD time."*

**The Korean-verified shortlist**, from [[stt-korean-providers]] — services with `Language.KO` in
their own map, ordered by the P99 above: Soniox `"ko"` (0.35), Speechmatics `"ko"` (0.74), Gladia
`"ko"` (1.49), Google `"ko-KR"` (1.57), Azure `"ko-KR"` (1.80), AWS Transcribe `"ko-KR"` (1.90),
ElevenLabs `"kor"` — three letters — (2.01), xAI `"ko"` (2.14), Fal `"ko"` (2.07), plus local
Whisper / Moonshine / FunASR. Deepgram, which sits at `0.35` in that table alongside Soniox, has
**no `LANGUAGE_MAP` at all**; `Language.KO` would be serialised as `"ko"` and the repo takes no position
on whether it is accepted. AssemblyAI's map lists eighteen languages and Korean is not among them.
Sarvam's only `KO*` entry is `Language.KOK_IN` — Konkani.

**The absence, stated plainly.** [[stt-korean-providers]], verified by grep across `src/`:

> *"No measured Korean accuracy number, and no Korean-on-8 kHz-telephony number of any kind, exists
> anywhere in this repository at this commit. … `WER` / `word error rate` / accuracy claims → zero
> hits, for any service, any language."*

The only `8000` values in the tree are telephony serializer defaults. **You are not going to get this
number by reading.** It is unknown #1 from §1.4, it is a *benchmark*-type falsifier, and its
precondition is a Korean STT contract plus a labelled Lina corpus.

**What the row turns on.** Whether a streaming interface removes a serial term from the budget by
enough to matter, and what the accuracy cost of that swap is in Korean on telephony-band audio.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.3 Row 3 — TTS and the six-service Korean word-timestamp intersection

[[ch-07/read]] supplied all of it, from [[tts-korean-providers]].

**Twelve services map `Language.KO`** — Cartesia, ElevenLabs, Azure (via `azure/common.py:200-201`),
Google, Inworld, Soniox, xAI, LMNT, MiniMax, AWS Polly, Camb, XTTS — verified by grep:

```bash
$ grep -rl "Language.KO\b" src/pipecat/services/*/tts.py src/pipecat/services/azure/common.py
```

**Six of the twelve also emit word timestamps.** The full `add_word_timestamps` caller list is
azure, cartesia, elevenlabs, elevenlabs/dialogue, gradium, hume, inworld, resembleai, rime, smallest,
soniox, speechify, xai. Intersect with the Korean twelve and you get **azure, cartesia, elevenlabs,
inworld, soniox, xai**. These are the services where a barge-in can truncate the assistant context at
the last *spoken* word via `TTSTextFrame.pts`, rather than at the last *generated* token.

**⚠️ The provenance caveat, which is the most important sentence in this row.** Those six are a
**grep over declared capability**, not a behavioural verification. Nobody in this course has watched
a Korean word timestamp arrive from any of them. The declaration says the service calls
`add_word_timestamps`; it does not say the provider returns useful boundaries for Korean 어절. Treat
"six" as an upper bound on candidates, not a verified set.

**`resolve_language` does not fail on an unmapped language:**

```python
# src/pipecat/transcriptions/language.py L614-629
    # Not in map - fall back with warning
    lang_str = str(language)

    if use_base_code:
        # Extract base code (e.g., "en" from "en-US")
        base_code = lang_str.split("-")[0].lower()
        logger.warning(f"Language {language} not verified. Using base code '{base_code}'.")
        return base_code
    else:
        logger.warning(f"Language {language} not verified. Using '{lang_str}'.")
        return lang_str
```

A `logger.warning` and a return. Rime — which *is* word-timestamp capable, which makes it the most
tempting wrong choice — maps exactly five languages (`ger/fra/eng/spa/hin`) and is called with
`use_base_code=False`, so `Language.KO` goes out as the literal `"ko"` to an API that does not know
it. Nothing raises. The downstream catch is `max_consecutive_zero_audio_contexts: int = 3`
(`tts_service.py:168`), which trips only after three consecutive contexts produce zero audio.

**No maintained self-hosted Korean TTS exists in the tree.** The only local service with a Korean
mapping is `XTTSService`, `@deprecated` since 1.7.0 with *"No replacement"*; the two maintained local
services, `KokoroTTSService` and `PiperTTSService`, map no Korean at all. There is also no Korean-
native vendor directory — no Typecast, no Supertone, no Naver Clova. **On-prem Korean TTS is an
absence, not an option**, unless someone writes a `TTSService` subclass.

**And realtime_voice's side:** [[rtv-vs-pipecat-gap]] records `OpenAICompatibleStreamingTTS`
(`with_streaming_response`, `chunk_size=1024`, PCM at 24 kHz,
`extra_body={"temperature":0.7,"num_timesteps":4}`) — one in-house provider, streaming, and the
excerpt's own note that boson has no TTS-side word timestamps.

**What the row turns on.** Whether a word-timestamp-emitting Korean provider is reachable on your
terms, because rows 5 and 3 are coupled through exactly that.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.4 Row 4 — `KoreanPhraseChunker`

**[[ch-03/read]] supplied the algorithm**, 283 lines per [[rtv-vad-chunking]]: a 1→2→tail adaptive
batching schedule (`_batch_phase` 0 → single sentence for time-to-first-audio, 1 → pairs, 2 → bounded
tail), `_is_safe_period` (L255) refusing to split `1.5`, `...`, or ASCII identifiers like `gpt-4.1`
and hostnames, `_is_numeric_separator` (L277) protecting `1,000`, and `_INTERNAL_TAG` stripping
Gateway control tags from spoken text while `start_char`/`end_char` keep the *source* span intact.
Defaults `min_chars=12, max_chars=60, batch_max_chars=320`, with `hard_max_chars` resolving to
`min(batch_max_chars, max_chars * 2)`.

**[[ch-07/read]] supplied what Pipecat offers in its place**: `SimpleTextAggregator` plus the
sentence matcher in `utils/string.py`.

```python
# src/pipecat/utils/string.py L118-127
# Latin punctuation that NLTK handles well — these need NLTK's disambiguation
# because "." can appear in abbreviations, decimals, etc.
_LATIN_SENTENCE_ENDING_PUNCTUATION: frozenset[str] = frozenset({".", "!", "?", ";", "…"})

# Non-Latin sentence-ending punctuation that is always unambiguous and never needs
# NLTK's disambiguation logic. Used as a fallback when NLTK doesn't support the
# language (e.g., Japanese, Chinese, Korean, Hindi, Arabic).
UNAMBIGUOUS_SENTENCE_ENDING_PUNCTUATION: frozenset[str] = (
    SENTENCE_ENDING_PUNCTUATION - _LATIN_SENTENCE_ENDING_PUNCTUATION
)
```

Korean is named in that comment, and here is the path it takes:

```python
# src/pipecat/utils/string.py L183-194
    if len(sentences) == 1 and first_sentence == text:
        if text and text[-1] in SENTENCE_ENDING_PUNCTUATION:
            return len(text)
        # Fallback for languages not supported by NLTK (e.g., Japanese, Chinese,
        # Korean, Hindi, Arabic). NLTK returned the entire text as a single
        # sentence, and the last character is not sentence-ending punctuation
        # (it's a lookahead character). Scan for unambiguous non-Latin sentence-
        # ending punctuation that doesn't need NLTK's disambiguation.
        for i, ch in enumerate(text):
            if ch in UNAMBIGUOUS_SENTENCE_ENDING_PUNCTUATION:
                return i + 1
        return 0
```

A linear scan for the first non-Latin terminator. No decimal guard, no identifier guard, no
thousands-separator guard, no batching schedule, no tag stripping. Note also that Korean is
*excluded* from the CJK word-grouping path — Cartesia tests `base_lang in {"zh", "ja"}`
(`cartesia/tts.py:454-455`) and ElevenLabs does the same at `tts_base.py:329`, so Korean falls
through to the ordinary space-separated branch. [[tts-korean-providers]] is careful about what that
means: *"That matches Korean 어절 spacing — but it is an untested assumption in this code, not a
verified Korean path."*

**What the row turns on.** Whether the guards encode knowledge that Lina's actual text needs, which
is a corpus question and one you can answer this week with your own transcript archive.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.5 Row 5 — `AudioTextPlayoutLedger`

**[[ch-03/read]] supplied the two methods.** From [[rtv-vad-chunking]]: `audible_text()` (L74) walks
phrases until the client cursor and, for the partially-played phrase, computes
`ratio = (cursor - sample_start) / (sample_end - sample_start)` then `text[:int(len(text) * ratio)]`
— *"a linear character-per-sample approximation, not a word-timestamp alignment."*
`playout_complete()` (L98) — all phrases `complete` **and** `played_sample >= queued_samples` — is
what lets `_cancel_generation` distinguish "customer interrupted me" from "I finished and they
replied," via the `semantic_interrupt` flag at `session.py` L502-507. `acknowledge()` moves the
cursor with `max(current, played_sample)`, so a late ack cannot rewind.

**[[ch-08/read]] supplied Pipecat's alternative and its gap.** Pipecat obtains the same guarantee
*positionally*: the assistant aggregator sits **after** `transport.output()`, so it only ever sees
text that was released, paced by word-timestamped `TTSTextFrame`s. [[rtv-vs-pipecat-gap]] records
what that costs: the alternative is *"Emergent, not explicit"* and *"No `[interrupted]` marker
written"* — the untagged-partial gap. History contains a truncated assistant turn with nothing
marking it as truncated.

**[[ch-07/read]] supplied the coupling.** The positional alternative is only available if the chosen
TTS emits word timestamps — which is row 3's six-service intersection, which is a grep and not a
behavioural test. If the provider emits none, the positional mechanism has nothing to pace on.
[[rtv-vad-chunking]] states the dependency directly: `AudioTextPlayoutLedger` *"would become
redundant only if the chosen TTS emits word timestamps."*

**The accuracy trade, from [[rtv-vs-pipecat-gap]], stated as mechanism:** the ledger is *"more
accurate on paper (works with timestamp-less TTS) and less accurate mid-word (linear char/sample
approximation)."* Both halves of that sentence are true simultaneously and they are about different
failure modes.

**This row is coupled to row 3 and the figure enforces it.** Dropping `AudioTextPlayoutLedger` while
choosing a TTS outside the six is flagged as an incompatible combination, with the reminder that the
six are a grep.

**What the row turns on.** Which TTS row 3 lands on, and whether an untagged truncated turn in
history is a problem for Lina's downstream consumers (the compaction summariser reads that history;
so does your evaluation set).

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.6 Row 6 — Transport

**[[ch-05/read]] supplied the whole row.** Both implementations wrap aiortc's `RTCPeerConnection`,
both resample with PyAV, both drive a data channel, both pace outbound audio against a wall clock.

**Sizes.** Pipecat's `transports/smallwebrtc/` is **2,176** lines across `transport.py` (1,085),
`connection.py` (825), `request_handler.py` (266) — checkable with `wc -l`. realtime_voice's
`transport/webrtc/` is **1,276** lines by its own per-file listing (`manager.py` 248, `control.py`
226, `peer.py` 231, `tracks.py` 216, `buffer.py` 123, `config.py` 64, `transport.py` 168).

**⚠️ Source correction.** [[rtv-webrtc-transport]] headlines *"~960 LOC"* and then lists files
summing to 1,276 — its own table contradicts its headline by 316 lines, about 33%. [[ch-05/read]]
§10.1 flagged it and could not resolve it (the repo is not on this machine and rule 3 forbids opening
it). **Use 1,276; treat ~960 as unverified.** If you are sizing a migration off this number, the
difference is a week.

**Breadth, counted.** `ls -d src/pipecat/transports/*/ | wc -l` → **11** packages;
`grep -rn "(BaseTransport)"` → **13** subclasses, because `local/` ships 2 and `websocket/` ships 3;
`whatsapp/` ships **0** and is a signalling adapter over `SmallWebRTCConnection`. [[ch-05/read]] §2
did that accounting. **⚠️** [[rtv-vs-pipecat-gap]] says 12; the tree says 11 packages.

**Two mechanisms present on one side only.** `SmallWebRTCConnection` has `renegotiate(sdp, type,
restart_pc=False)` (L443), `ask_to_renegotiate()` (L799), a `pc_id` (L302), and a hand-rolled
`"disconnected"` handler written because *"aiortc does not provide any way so we can be aware when we
are disconnected"* (L350). realtime_voice's only recovery path is a fresh
`accept_offer(reconnect=True)`. Also present on Pipecat's side only: video (`RawVideoTrack`) and
screen share, neither of which Lina TMR uses.

**Two mechanisms present on the other side only** — and they are rows 8 and 9, split out precisely
because they are separable from the transport plumbing.

**Granularity difference, from [[ch-05/read]] §10.3.** Pipecat's `RawAudioTrack` writes at 10 ms with
a per-write `Future` resolved when the chunk is consumed; realtime_voice's `OutboundAudioTrack` uses
20 ms packets and an `av.AudioFifo` that keeps the remainder across `recv()` calls, with no
write-completion future. Its `_silence_frame` (L201) carries a hard-won comment: *"PyAV does not
guarantee zero-initialized AudioFrame storage. Sending a fresh allocation as 'silence' can therefore
produce full-scale random PCM"* — and there is a pinned test for it
(`test_outbound_track_underflow_is_explicit_zero_pcm_silence`).

**What the row turns on.** Whether a second client type or a second transport is on the roadmap,
because 1,276 lines that pass an aiortc loopback test are already paid for and eleven transports you
do not use cost nothing and buy nothing.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.7 Row 7 — Telephony serializer

**[[ch-05/read]] supplied the structural fact first: there is no telephony transport at all.** A
phone call in Pipecat is `FastAPIWebsocketTransport` **plus** one of six `FrameSerializer`s:

```bash
$ ls src/pipecat/serializers/
__init__.py  base_serializer.py  exotel.py  genesys.py  plivo.py
protobuf.py  telnyx.py  twilio.py  vonage.py
```

Six telephony (`twilio`, `telnyx`, `plivo`, `exotel`, `genesys`, `vonage`) plus `protobuf.py`, which
is not telephony. Sizes: `exotel.py` 171, `vonage.py` 188, `plivo.py` 256, `telnyx.py` 292,
`twilio.py` 314, `genesys.py` **964** — the outlier, and [[ch-05/read]] §6.11 explains it as
session-management surface rather than codec work. The ABC itself, `base_serializer.py`, is 106 lines
and four methods.

**Neither existing stack has any telephony path.** [[rtv-vs-pipecat-gap]]: realtime_voice's
`SileroVAD` hard-rejects 8 kHz, there is no serializer layer, no μ-law, and `CLAUDE.md` names *"future
SIP/RTP or telephony adapters"* as an intent, not an implementation. So this row is the one row where
neither column contains a shipped answer; the comparison is between *adding a serializer to a
framework that has the slot* and *adding a telephony layer to a stack that has no slot*.

**The sizing, from [[ch-05/read]] §11.1 and §2.2.** A Korean carrier outside those six means writing
pattern-B code: *"250–300 lines for a Twilio-shaped protocol, up to ~1,000 if it has Genesys-shaped
session semantics."* A carrier that hands you a SIP/RTP leg directly is neither — it is a different
problem that lands below the serializer.

**[[ch-06/read]] supplied the two 8 kHz facts.** First, the 8 kHz side costs no VAD retuning:
`round(0.2 / (256/8000))` = `round(6.25)` = 6, identical to 16 kHz (§3.1). Second, and this is the
one that hurts, from [[stt-korean-providers]]: *"the only `8000` values in the tree are telephony
serializer defaults (`twilio.py:79`, `telnyx.py:60`, `plivo.py:54`, `exotel.py:49`,
`genesys.py:148`, `vonage.py:43`); no STT service documents behaviour at that rate."* **No 8 kHz
Korean STT number exists anywhere in the tree.**

**The figure flags one incompatible combination on this row**: keeping realtime_voice's Silero while
adding telephony, because it raises `ValueError` on 8 kHz.

**What the row turns on.** Which carrier, and when.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.8 Row 8 — Session auth (`WebRTCSessionManager`)

**[[ch-03/read]] and [[ch-05/read]] supplied both halves.**

From [[rtv-webrtc-transport]]: `WebRTCSessionManager` (`manager.py` L51), docstring *"Create
short-lived authorized sessions and enforce one live peer each."*
`create_session(customer_id, *, session_id=None, metadata=None) -> VoiceSessionTicket(session_id,
token, expires_at, customer_id)` mints `secrets.token_urlsafe(32)` and stores only
`hashlib.sha256(token).digest()`; `_authorize` (L227) checks expiry then `hmac.compare_digest`;
`session_token_ttl_seconds = 15 * 60`; `accept_offer(..., reconnect: bool = False)` raises
`SessionConflictError("this voice session already has a live peer")` unless `reconnect=True` —
explicit reconnect, no silent takeover.

**Pipecat ships no counterpart**, and [[ch-05/read]] §10.2 established that by grep rather than by
assertion:

```bash
$ grep -rn "token_urlsafe\|compare_digest" src/pipecat/
src/pipecat/runner/run.py:324:    if not hmac.compare_digest(expected, sig):
src/pipecat/transports/whatsapp/client.py:181:        if not hmac.compare_digest(expected_signature, received_signature):
```

Both hits are webhook-signature verification. Neither is voice-session authorization.
`SmallWebRTCConnection.__init__(ice_servers=...)` (`connection.py` L245) has no tokens, no TTL, no
customer binding; `request_handler.py` is a bare offer/answer endpoint.

**This is application code either way.** The row is not "does the framework have it" — the answer to
that is settled — it is whether the 248 lines you own move onto the Pipecat connection object
unchanged, get rewritten around it, or stay where they are with their transport.

**What the row turns on.** Whether row 6 moves, since this is a policy layer wrapped around whatever
connection object row 6 lands on.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.9 Row 9 — Control protocol (`ControlEvent`)

**[[ch-03/read]] and [[ch-05/read]] supplied both halves.**

From [[rtv-webrtc-transport]]: `ControlEvent` (`control.py` L25) is
`@dataclass(frozen=True, slots=True)` with `session_id, type, sequence, payload, turn_id,
generation_id, version=CONTROL_PROTOCOL_VERSION` where `CONTROL_PROTOCOL_VERSION = 1`. Docstring
L28-30: *"Audio bytes are intentionally prohibited. Microphone and assistant audio belong on RTP
tracks, never in JSON or base64."* `_reject_audio_payload` (L117) walks the payload **recursively**
and raises on any key normalising into `{"audio","audio_base64","audio_data","base64_audio","pcm",
"pcm16","wav"}`, any string starting `data:audio/`, and any `bytes/bytearray/memoryview`. `from_json`
(L64) rejects unknown top-level fields, non-object payloads, and version mismatches.
`OrderedControlChannel` (L136) refuses a partially-reliable channel *at construction* — `ordered=False`,
a non-`None` `maxRetransmits`, or a non-`None` `maxPacketLifeTime` each raise `SignalingError`
(*"Control events must not silently disappear"*) — and `receive()` enforces strict in-order delivery
with `SignalingError(f"out-of-order control event: expected {...}, received {...}")`. Outbound
sequence is a private counter, so the server owns ordering. Cap: `max_control_message_bytes = 64 KiB`.

**Pipecat ships no counterpart at the data channel.** `SmallWebRTCTransport._on_app_message(message,
sender)` → the `on_app_message` event handler: no schema, no sequence check, no size cap, no audio
ban. Pipecat does have a typed client protocol — RTVI, see [[rtvi-observability]] — but it rides a
different layer and is not enforced at the channel.

**The dotted-type mapping is where the closed union surfaces**, from [[ch-05/read]] §7 and
[[rtv-webrtc-transport]]: `_control_event()` (`transport.py` L118) maps `VoiceEvent →
event.kind.value`, `AgentTextDelta → "text_delta"`, `ASREvent → "transcript.interim" |
"transcript.final" | "asr.end_of_turn" | "asr.error"`, `VADEvent → "vad.speech_started" |
"vad.speech_stopped"`, and anything else raises
`TypeError(f"unsupported voice event: {type(event).__name__}")`. Under Pipecat that mapping becomes a
serializer's job — [[ch-05/read]] §7's finding is that *"the serializer is where the open sum type
gets closed."*

**What the row turns on.** Whether the wire contract your debug client and your ops tooling already
speak is worth keeping byte-identical across a transport change.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.10 Row 10 — The LLM loop

**[[ch-09/read]] supplied all of it.** Five mechanism facts, each with a file and line.

**One — `get_messages()` returns the live list.** `llm_context.py:245` returns `self._messages`
itself; only `truncate_large_values=True` copies. boson's `ContextManager.get_messages()` returns
`deepcopy(self._messages)` (`manager.py:47-51`). Each is right in its own home:
`_update_function_call_result` writes *through* the returned reference
(`llm_response_universal.py:2158-2165`), so identity is load-bearing on one side; a copy handed to a
provider adapter cannot be corrupted, so isolation is load-bearing on the other. [[ch-09/read]] §9.1
traced the eight steps by which porting the `deepcopy` defence makes tool results silently never
reach the model — no exception, no log line.

**Two — one inference per `LLMContextFrame`:**

```python
# src/pipecat/services/openai/base_llm.py L599-605
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
```

Line 601 **does not check direction**, which is why an upstream frame re-prompts the same service.

**Three — no `max_turns` anywhere.** `grep -rn "max_turns\|max_iterations\|max_tool_calls"
src/pipecat/` → zero. The only bounds are per-call timeouts. boson's guard is a two-line `while`
condition (`agent_loop.py:207-209`).

**Four — the loop is closed by topology, not by a loop header.** The last processor in the pipeline
pushes an `LLMContextFrame` **upstream** (`llm_response_universal.py:1889`), and the turn ends when
nobody does. [[ch-09/read]] §13 put it in one sentence per system: *"Pipecat: the turn ends when
nobody pushes an `LLMContextFrame` upstream. An absence. boson: the turn ends at
`agent_loop.py:363`, `break  # Done — text response means end of turn`. A statement. realtime_voice:
the turn ends when the agent's `AsyncIterator[AgentTextDelta]` stops. A delegation."*

**Five — three resolutions, costed, none recommended.** [[ch-09/read]] §11 developed **adopt**
(~30 L cap + gates + reminder processor; deletes `run_agent_loop`, 561 L; 22 tool signatures change),
**wrap** (~200–300 L in one processor; deletes nothing; 0 signatures change), and **bypass**
(~120–170 L bridge; deletes nothing; 0 signatures change), and explicitly left the choice open.
[[ch-09/read]] §10 also recorded the fact that **realtime_voice already implements the bypass shape**
— *"you are not reading a hypothetical: you are reading a description of code that exists on branch
`voice-chat-dev`"* — while making no claim about whether that was a good decision.

**Six — the third implementation, and why it is structurally a third answer.** From
[[rtv-pipeline-session]], realtime_voice's agent slot is one `Protocol` yielding exactly one type:
`StreamingConversationAgent.stream(request) -> AsyncIterator[AgentTextDelta]`. Not tool calls, not
context objects, not messages. There is **no `link()`, no `FrameDirection`, no upstream push** —
*"Where Pipecat gets a `LLMContextFrame` pushed upstream to close the tool loop, realtime_voice has
one direction only."* Map that onto the three ownership questions and the answer to all three is
"not the voice package": it holds no context, dispatches no tools, and learns the turn is over when
the iterator stops. The excerpt's own framing: *"`StreamingConversationAgent` is the one slot Pipecat
has no analogue for — Pipecat assumes it owns the LLM call, whereas boson deliberately delegates."*

**What the row turns on.** §4.1. This is half of the first irreducible collision.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.11 Row 11 — Tools

**[[ch-09/read]] supplied all of it.**

**Three registration routes, one dictionary.** `register_function(name, handler)`
(`llm_service.py:200-205`) with `name=None` installing a catch-all; `FunctionSchema(handler=...)`
carried in `ToolsSchema`; and direct functions whose schema is derived from the signature and
docstring (`adapters/schemas/direct_function.py:279-289`). `register_direct_function` is
`@deprecated` since 1.4.0 (`llm_service.py:982-984`).

**The `result_callback` contract: a handler never returns a value.** `llm_service.py:142-155`. It
settles by `await params.result_callback(result)`. boson's handlers are `handler(**arguments)`
returning a value (`tools/executor.py:67`), so all **22 tool signatures** under `agents/*/tools/` need
a change or a shim.

**`run_in_parallel` defaults to `True`** (`llm_service.py:303-308`). boson runs tools sequentially
inside `_execute_tool_uses` and its `_SYNC_HANDLER_LOCK` exists because, verbatim from
[[boson-tool-router]], *"production tools do read-modify-write on shared YAML/JSON files."* That
assumption must be re-asserted explicitly, not inherited.

**No `tool` role in boson's message schema.** `basement/schemas/message_schema.py:46` declares two
roles with blocks inside `content`; Pipecat's `LLMContext` is OpenAI-shaped with three-plus roles, a
`tool_calls` array, and a separate `tool` message. [[ch-09/read]] §8.4 has the side-by-side, and §9.2
prices it: a converter in each direction if history must survive the migration, plus a decision about
`is_error`, which has no Pipecat field. *"Not a rename. If someone on the team scopes this as
`s/user/tool/`, correct them."*

**Three gates, one slot.** boson separates exposure (what the model sees), availability (the
`_allowed_tools_var` `ContextVar` allowlist), and permission (`PermissionChecker.check_tool`).
`FunctionCallRegistryItem` has no permission field, no allowlist field, and `_run_function_call` has
no interception point between name resolution and invocation. [[ch-09/read]] §9.4 prices the two
ways out: 22 decorated handlers, or one catch-all plus your own dispatch table — *"which is
`ToolRouter` again, re-hosted."* Also: `ToolRegistry.discover_tools()` has no Pipecat equivalent and
stays as boson glue either way.

**What the row turns on.** §4.1. This is the other half of the first irreducible collision, and
[[ch-09/read]] §12's Move 1 (catch-all as a permission kernel) is the framework-extension sketch that
would make one of the votes cheap.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.12 Row 12 — The stage machine

**[[ch-10/read]] supplied all of it.**

**`FlowManager` is a plain class that lives outside the pipeline:**

```python
# src/pipecat/flows/manager.py L80, L91-101
class FlowManager:
    ...
    def __init__(
        self,
        *,
        llm: LLMService | LLMSwitcher,
        context_aggregator: Any,
        worker: PipelineWorker | None = None,
        task: PipelineWorker | None = None,
        context_strategy: ContextStrategyConfig | None = None,
        transport: BaseTransport | None = None,
        global_functions: list[FlowsFunctionSchema | FlowsDirectFunction] | None = None,
    ):
```

Not `class FlowManager(FrameProcessor)`. It drives from outside by queueing frames at the pipeline
**head** (`self._worker.queue_frames(frames)`, `manager.py:842`; `queue_frame` with `DOWNSTREAM`
enters at the head per [[pipeline-task-runner]]). [[theory-narrow-waist]] §4 recorded why that
matters structurally: `flows/` defined **2** frames, not 20, and node state — `NodeConfig`,
`FlowResult`, `ContextStrategyConfig` — *"is plain `TypedDict` held in `FlowManager._current_node`,
and never becomes a frame at all."*

**`FlowConfig` does not exist at this commit.** `grep -rn "FlowConfig" src/pipecat/` → zero hits.
There is no declarative whole-graph object; there are node configs produced at transition time.

**Two `NodeConfig` producers.** A function handler returning `tuple[Result, NodeConfig]` — the tool
edge, e.g. `insurance_quote.py` L114's `collect_age` — and a direct call to
`flow_manager.set_node_from_config(node_config)` (`manager.py:588`) from outside, which
[[flows-insurance-example]] found in-tree exactly once, at `warm_transfer.py:658`, guarded on
`flow_manager.current_node`.

**No transition-legality validation, and no node registry.** `_validate_node_config`
(`manager.py:867-898`) checks exactly two things: that `task_messages` is present, and that each
entry in `functions` is a `FlowsFunctionSchema` or a valid direct function. There is no from→to
check anywhere in the codebase. [[design-boson-rules-on-pipecat]] states it as a table row:
*"`StageMachine.transition()` legality → **NOTHING** … Flows has no from→to check anywhere in the
codebase."*

**boson's side**, from [[boson-stage-machine]]: a stage is a *context package* — prompt + visible
tools + visible skills + a whitelist of legal successors — and `StageMachine.transition()` returns
`TransitionResult(success=False, error="Transition 'a' -> 'b' not allowed")` on an illegal edge.
Nine Lina stages. **The LLM never chooses one.**

**`stage_config.py`'s direct mapping**, from [[flows-insurance-example]]: `stages[X]["tools"]` →
`NodeConfig["functions"]` is direct; `_GLOBAL_TOOLS` → `FlowManager(global_functions=[...])`, mixed
in at `manager.py:654` (`functions_list = self._global_functions + node_config.get("functions", [])`),
is an exact match. **`transitions` and `skills` are unmapped.** `transitions` *"stops being data"* —
in Flows the legal successors are implicit in which `NodeConfig` each function can return — and
`skills` has *"no Flows concept at all."*

**A named regression, which is the falsifier material for this row.** `stage_config.py` documents its
own bug twice: `v0.7.5 (#12)` — *"`purchase` added — `transition_detector.py:157` emits
`StageTransition("purchase")` … but the stage machine rejected it because this list omitted
`purchase`."* That class of bug is structurally impossible in Flows. The mechanism trade is: the
rejection path disappears, and so does the single-file readable transition table.

**What the row turns on.** Whether the edge whitelist is a safety property Lina depends on or a
config file that has caused more bugs than it caught. That is answerable from your own issue history.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.13 Row 13 — The rule layers

**[[ch-12/read]] supplied all of it, and §2 just measured its cost.**

**The three constraints and the seam derivation.** From [[design-boson-rules-on-pipecat]], the
position is a data dependency, not a convention:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py L856-873
    async def push_aggregation(self) -> str:
        """Push the current aggregation."""
        if len(self._aggregation) == 0:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()
        self._context.add_message(
            cast(LLMContextMessage, {"role": self.role, "content": aggregation})
        )
        await self.push_context_frame()

        message = UserTurnMessageAddedMessage(
            content=aggregation, timestamp=self._user_turn_start_timestamp
        )
        await self._call_event_handler("on_user_turn_message_added", message)

        return aggregation
```

`add_message` at 862-864 **writes**, `push_context_frame()` at 866 **pushes**, and
`base_llm.py:601` **consumes**. A processor between the user aggregator and the LLM therefore holds
the complete turn with inference not yet begun, and rollback is real via `context.set_messages(...)`
(`llm_context.py:377`). Option B — the `on_user_turn_message_added` event at line 871 — is *after*
the push, so **an event handler can never veto**. Option C — let generation start, then
`broadcast_interruption()` (`frame_processor.py:1017-1022`) — pays the first generation's TTFT twice.

**The collapse argument.** `push_frame` is irreversible, so cross-*processor* veto is unavailable and
all layers must collapse into **one** `FrameProcessor`. [[design-boson-rules-on-pipecat]] states the
loss explicitly in its mapping table: *"cross-`processor` veto — all layers must collapse into one
object."*

**The mapping table.** [[ch-12/read]] presents it as 11 rows. **⚠️ Count discrepancy worth
recording:** the underlying excerpt's table in [[design-boson-rules-on-pipecat]] §2 has **17** rows
(layer discovery, sequential `@check`, parallel `@check`, phase-1/2 commit, `Filter`, `Respond`,
`Inject`, `PreTool`, `StageTransition`, `StageMachine` legality, `StageDefinition.prompt/tools`,
`_GLOBAL_TOOLS`, `Compact()`, `SignalQueue`, `AgentStatusTracker`, `SharedLayerContext`,
`ScriptEngine`). Four of those seventeen are separately rows 12, 14, 15 and part of 11 in *this*
ledger, which is most of the difference. Use the mapping table for mechanism; do not use either count
as evidence of anything.

**The Tier-2 latency bill, now measured.** §2. Thirteen live `@check`s, exactly two of them
`check_type="llm"` (`intent_rules` prio 30, `sentiment_tracker` prio 10, both `mode="parallel"`).
Tier 1 is the other eleven — *"all pure Python, sub-millisecond, blocking, free."* Tier 2 is the two,
and it buys in-turn veto and in-turn transitions; the only way not to pay it is to run Tier 2
concurrently with the LLM's first tokens and call `set_node_from_config()` on completion, **which
makes every stage change land one turn late.** [[design-boson-rules-on-pipecat]] states the trade in
one sentence: *"veto and in-turn steering cost 250-400 ms; next-turn transitions cost 0 ms. boson
pays the 250-400 ms today."*

Write your measured P50 and P95 into this row, not just into the figure. The row now reads with a
number in it.

**What the row turns on.** §4.2, plus the number you just measured.

| Vote | Assumption | Falsifier | Measured Tier-2 P50 / P95 |
|---|---|---|---|
| | | | |

---

### 3.14 Row 14 — `ScriptEngine`

**[[ch-12/read]] supplied it in one line**, and it is the shortest row in the ledger because the
mechanism is the shortest. From [[design-boson-rules-on-pipecat]]'s mapping table:

> `ScriptEngine.process_turn(state, msg, registry)` → **runs unchanged inside the processor** —
> *"already stateless dict-in/`Action`-out"* — what is lost: *"nothing — cleanest port in the
> system."*

**Why it ports without edits**, from [[boson-script-engine]]: it is *"a stateless pure function over
a state dict that returns `(new_state, Action)"*, with zero gateway coupling — it imports only
`gateway.script.schema` and `gateway.schemas.actions`. 517 LOC (`engine.py` 284 + `schema.py` 233).

**The one structural fact that constrains where its output can go.** `Respond(step.text)` is the
*literal string spoken*, LLM excluded, because Korean insurance-consent script text is legally fixed.
[[boson-script-engine]] is explicit about what that rules out: *"Porting `purchase_pre_consent` onto
`NodeConfig.task_messages` would let the model paraphrase a regulated consent disclosure."* The
mapping that preserves the property is `Respond` → push a `TTSSpeakFrame` while suppressing the
downstream `LLMContextFrame`.

**Also recorded:** `FlowManager.state` — a plain `dict[str, Any]` (`manager.py:143`) — is the natural
home for `script_state`, since Pipecat has no `SessionState`. And `pause_for_interrupt` /
`resume_from_interrupt` must be re-driven from `InterruptionFrame` rather than from boson's
`AgentStatusTracker`, which has no frame equivalent.

**What the row turns on.** Whether "runs unchanged inside the processor" survives contact with the
interrupt re-driving, which is the only part of this row that is not free.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.15 Row 15 — Compaction

**Routed here from [[ch-09/read]], which excluded it by name.** Its scope box: *"Context compaction
is NOT in this chapter. `LLMContextSummarizer` versus boson's `gateway/compact/` is a different
subsystem with different failure modes and a different trigger, and it is on [[ch-13/read]]'s
give-back list."* That is rule two's "the row is two rows" test, applied in advance.

**The excerpt supplies the row.** From [[boson-compact-session]]: both systems solved the same
problem — summarise old history, keep a recent tail, splice `[system?] + [summary] + [tail]` — and
arrived at nearly identical designs. **The divergence is where the answer lands.** boson runs
summarisation as a detached `asyncio.create_task` that writes `session.pending_compact` and applies
it at the top of the *next* turn, so no turn ever waits on it (`bootstrap.py` L455-458: *"Apply
pending compact BEFORE layers"*). Pipecat's `LLMContextSummarizer` applies the result the moment an
`LLMContextSummaryResultFrame` arrives, mid-pipeline.

**The three-way parameter correspondence**, verified against the tree:

```python
# src/pipecat/utils/context/llm_context_summarization.py L146-148
    max_context_tokens: int | None = 8000
    max_unsummarized_messages: int | None = 20
    summary_config: LLMContextSummaryConfig = field(default_factory=LLMContextSummaryConfig)
```

```python
# src/pipecat/utils/context/llm_context_summarization.py L93-98
    target_context_tokens: int = 6000
    min_messages_after_summary: int = 4
    summarization_prompt: str | None = None
    summary_message_template: str = "Conversation summary: {summary}"
    llm: Optional["LLMService"] = None
    summarization_timeout: float = DEFAULT_SUMMARIZATION_TIMEOUT
```

| boson `CompactConfig` | Pipecat | Note |
|---|---|---|
| `threshold_messages = 30` (ge=5) | `max_unsummarized_messages = 20` | port as `30` |
| `keep_recent = 10` (ge=2) | `min_messages_after_summary = 4` | port as `10` |
| — | `max_context_tokens = 8000` | **a token trigger boson never had** |
| `provider` / `model` (`gpt-5.4-mini`) | `LLMContextSummaryConfig.llm: Optional[LLMService]` | dedicated cheap model, both sides |
| `temperature = 0.3` | — | not a field on the Pipecat config |

The token estimator is three constants, `llm_context_summarization.py:33-35`: `CHARS_PER_TOKEN = 4`,
`TOKEN_OVERHEAD_PER_MESSAGE = 10`, `IMAGE_TOKEN_ESTIMATE = 500`. boson has **no token estimate at
all** — [[boson-compact-session]]: *"Trigger is message-count only — there is no token-based trigger
anywhere in boson."*

**⚠️ A deprecation to route around.** The flat `LLMContextSummarizationConfig` (all four numbers on
one object) is `@deprecated` since 0.0.104, removed in 2.0.0
(`llm_context_summarization.py:170-173`). Write new code against
`LLMAutoContextSummarizationConfig` + a nested `LLMContextSummaryConfig`. Per
[[pipecat-design-philosophy]], the deprecation registry has **391 live deprecations, all with
`removed_in == "2.0.0"`** — so "adopt but on the deprecated class" is a real and avoidable mistake.

**The two named losses, which belong on this row's falsifier line.**

1. **Pre/post compact hooks have no equivalent.** boson has module-level `set_pre_compact_hook` /
   `set_post_compact_hook` with typed signatures that may *mutate* the input message list and the
   output summary string. Pipecat's closest thing is the `on_summary_applied` event carrying
   `SummaryAppliedEvent(original_message_count, new_message_count, summarized_message_count,
   preserved_message_count)` (`llm_context_summarizer.py:39`, fired at `:468`) — **observability
   only; it cannot mutate input or output.** Anything that strips noisy tool blocks pre-summary or
   appends extracted structured data post-summary must subclass `LLMContextSummarizer` or route
   through a dedicated summarisation `LLMService`.
2. **The `<system-reminder>Active stage: …</system-reminder>` re-injection.** `SharedHistory.
   swap_compact` (`session/history.py` L82) appends `<system-reminder>Active skill: …` and
   `<system-reminder>Active stage: …` after the summary when set. That is stage-machine coupling
   living inside the compactor. **It must be re-attached explicitly or stage identity is silently
   lost at every compaction** — silently, because nothing raises; the model simply stops being told
   which stage it is in, mid-call, after message 30.

**Two more mechanism facts worth carrying.** `LLMContextSummarizer` extends `BaseObject`, **not**
`FrameProcessor` — `process_frame(self, frame)` takes a single argument, no `direction`, and it is
driven by an aggregator rather than linked into the pipeline. And the tool-pair safety strategies
point in opposite directions: boson's `_safe_window_start` drops leading orphan `tool_result`s
**forward**; Pipecat's `_get_earliest_function_call_not_resolved_in_range` pulls `summary_end`
**back** before an unresolved call.

**What the row turns on.** Whether deferred-to-next-turn application is a property Lina depends on,
and whether either loss above is load-bearing for your compliance recording.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.16 Row 16 — Observability

**[[ch-11/read]] supplied the Pipecat side.**

**The observer plane.** `BaseObserver` is read-only over the frame graph — [[rtvi-observability]]:
*"You instrument by subscribing, not by editing processors."* Four observers ship:
`startup_timing_observer.py`, `turn_tracking_observer.py`, `user_bot_latency_observer.py`, plus a
`loggers/` package.

**`LatencyBreakdown` is the per-cycle object:**

```python
# src/pipecat/observers/user_bot_latency_observer.py L83-89, L107-111
class LatencyBreakdown(BaseModel):
    """Per-service latency breakdown for a single user-to-bot cycle.

    Collected between ``VADUserStoppedSpeakingFrame`` and
    ``BotStartedSpeakingFrame`` when ``enable_metrics=True`` in
    :class:`~pipecat.pipeline.worker.PipelineParams`.
    ...
    ttfb: list[TTFBBreakdownMetrics] = Field(default_factory=list)
    text_aggregation: TextAggregationBreakdownMetrics | None = None
    user_turn_start_time: float | None = None
    user_turn_secs: float | None = None
    function_calls: list[FunctionCallMetrics] = Field(default_factory=list)
```

Note the docstring's precondition: `enable_metrics=True` in `PipelineParams`, which defaults to
`False` ([[pipeline-task-runner]], `worker.py` L163-195).

**`can_generate_metrics()` is the gate:**

```python
# src/pipecat/processors/frame_processor.py L488-494
    def can_generate_metrics(self) -> bool:
        """Check if this processor can generate metrics.

        Returns:
            True if this processor can generate metrics.
        """
        return False
```

**Default `False` on the base class.** A processor you write emits nothing until you override it.
That is the "gating" fact: the plane exists, and your own processors are outside it by default.

**Aggregation is left to you.** There is no P50/P95 anywhere in the observer plane. `LatencyBreakdown`
gives you one cycle and `chronological_events()` formats it for a log line; turning a stream of those
into percentiles is application code. [[rtvi-observability]] records exporters — OpenTelemetry,
Sentry — and the RTVI wire protocol, but the percentile computation is not in the box.

**The boson side**, per [[rtvi-observability]] and [[rtv-vs-pipecat-gap]]: three things and no plane.
A trace decorator (`gateway/debug/log_decorator.py`, `time.perf_counter()` around a call printed as
`[TRACE …] EXIT (…ms)`), an ad-hoc `elapsed_ms` threaded into barge-in policy
(`core.py:166 should_interrupt(session_id, content, elapsed_ms)`), and
`BoundedAudioOutput.discarded_frames`, *"the only metric in the transport, and it is orphaned"* —
nothing reads it. Plus `provider_latency_ms` / `endpoint_latency_ms` fields on events. No OTel, no
spans, no aggregation.

**And the standard those two stacks are being held to is boson's own.** [[rtv-vs-pipecat-gap]]:
`CLAUDE.md` says *"Instrument before optimizing… Report P50/P95/P99."* Neither stack does that today,
and Pipecat's plane does not do the P-numbers either — it does the collection.

**What the row turns on.** Whether "aggregation is application code" is a small job or the same job
you would do from scratch, which you can settle by writing the aggregator once against
`LatencyBreakdown` and seeing how long it takes.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.17 Row 17 — Deployment and process topology

**This is a full row like any other and not a postscript.** [[ch-04/read]] supplied the host shape;
§7 below supplies the rest, because deployment is a decision *input* and not a latency term. Read §7,
then come back and fill these three cells.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

## 4. The two irreducible collisions

These two are stated as rows with their evidence and are **not resolved by the text**. They get a
vote cell, an assumption cell and a falsifier cell, all three filled by you.

### 4.1 Collision one — the agent boundary (rows 10 and 11)

**The statement.** Adopting Pipecat's pipeline means Pipecat owns the LLM call and the tool loop.
That stands against your own contract. From [[rtv-vs-pipecat-gap]], quoting `CLAUDE.md`: *"Keep
Basement and the dental business logic text-native"* / *"Basement and Gateway must not import
provider-specific audio code."* The excerpt's own characterisation: *"This is the deepest
architectural collision. Pipecat assumes it owns the LLM+tool loop; boson's whole `CLAUDE.md`
contract assumes it does not."*

**Why it is irreducible.** It is not a gap you can fill with 200 lines. It is two systems each
holding an invariant that the other's topology negates. `base_llm.py:601` consumes an
`LLMContextFrame` and starts a completion; nothing in Pipecat's design lets a completion be started
by something that is not an `LLMService` in the pipeline. Meanwhile boson's contract is that the LLM
call happens in `basement`, over text, with no audio-layer dependency. One of those has to give.

**What [[ch-09/read]] costed, and named none.** Three shapes, side by side (its §11.4 table):

| | **Adopt** | **Wrap** | **Bypass** |
|---|---|---|---|
| New Pipecat-side code | ~30 L cap + gates + reminder processor | ~200–300 L in one processor | ~120–170 L bridge |
| boson code deleted | `run_agent_loop` (561 L) + ~80 L cancellation repair + 6 provider shapers | none | none |
| Tool signatures changed | 22 | 0 | 0 |
| `max_turns` | rebuilt as a processor | kept | kept |
| Three gates | rebuilt as catch-all or decorators | kept | kept |
| Message schema | rewritten | kept | kept |
| Adapter layer used | yes (12 adapters) | no | no |
| Flows ([[ch-10/read]]) available | yes | expensive | no |
| Observer plane sees tool calls | yes | only what you emit | no |
| Already implemented in your repo | no | no | **yes, with realtime_voice as the voice layer** |

**The two priced consequences that attach to whichever you pick.**

**One — the missing turn ceiling.** No `max_turns` exists in Pipecat, and [[ch-09/read]] §9.3
rebuilt it as a counting processor over `FunctionCallsStartedFrame` / upstream `LLMContextFrame`,
placed **between `llm` and `tts`** because that is the only position that sees both the re-prompts
(travelling upstream) and the reset signal (`UserStartedSpeakingFrame`, travelling downstream). About
30 lines plus a test, against boson's two-line `while` condition — and with four named costs, the
sharpest being that **it counts re-prompts, not inferences**, so `max_turns=8` there means "8 tool
cycles," off by one from boson's "8 total iterations" by construction.

**Two — the `{"role": "tool"}` schema rewrite at 22 signatures.** §3.11. Two roles with blocks in
`content` versus three-plus roles with a `tool_calls` array and a separate `tool` message; dict
arguments versus JSON-string arguments; `is_error` versus no field. Plus the loss [[ch-09/read]] §9.2
names: `LLMContext` stores plain dicts with no schema, so *"a typo in a hand-built message dict is
discovered by the provider's 400, not by your type checker."*

**A third thing to weigh that is not a cost.** [[ch-09/read]] §10 recorded that realtime_voice
already implements the bypass shape, and was careful about what that does and does not mean:
*"it changes what 'adopt' and 'wrap' would cost — because both of them mean undoing something that
works today, and that undoing is a line item."*

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 4.2 Collision two — the rule layers (row 13)

**The statement.** [[ch-12/read]]'s derivation produced a **seam** (between `user_aggregator` and
`llm`, forced by the write-then-push data dependency at `llm_response_universal.py:856-866`) and a
**collapse** (all layers into one `FrameProcessor`, forced by `push_frame` being irreversible). The
open question this chapter poses without answering is: **how much of boson survives unedited under
each vote?**

**The candidate shape you are asked to evaluate, not told to adopt.** From
[[design-boson-rules-on-pipecat]]:

```python
# the proposed pipeline (design excerpt §4) — a candidate, not a recommendation
pipeline = Pipeline([
    transport.input(),
    stt,                       # Korean 8 kHz telephony STT
    BosonFillerGate(),         # boson layer 01
    user_aggregator,           # LLMContextAggregatorPair(context).user()
    BosonRuleProcessor(...),   # boson layers 02/03/04, Tier 1 + Tier 2
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])
# FlowManager(llm=llm, context_aggregator=pair, worker=worker, global_functions=[...])
# is NOT in this list — it drives from outside via worker.queue_frames (manager.py:841).
```

One `BosonRuleProcessor` between `user_aggregator` and `llm` holding **all 13 checks**, the
`RuleEngine`, the `SignalQueue` and a **real `SessionState`**; `StageMachine` kept as a pure legality
pre-check; `FlowManager` driving stages from outside via `set_node_from_config`.

**The claim attached to that shape, which is the thing to evaluate.** The excerpt's migration angle:
*"`BosonRuleProcessor` holds all 13 checks, the `RuleEngine`, the `SignalQueue`, the `StageMachine`
pre-check, and a real `SessionState` — so the rule files themselves need **zero edits**."* The
zero-edit claim rests entirely on keeping a real `SessionState` object and passing it as the
`session` argument, because `SharedLayerContext.__getattr__`/`__setattr__` proxy every unknown name
straight through to it. `flow_manager.state` is a `dict`, and rewriting 13 rules' `getattr(session,
…)` against a dict *"buys nothing."*

**Two positions in that list are load-bearing and the excerpt argues each from a data dependency, not
from taste.** `BosonFillerGate` must sit between `stt` and `user_aggregator`: one position earlier it
sees only audio and `_is_filler_text()` has no input; one position later the "네" is already
`add_message`d and a one-line string check has become a rollback. `BosonRuleProcessor` must sit
between `user_aggregator` and `llm`: one earlier and `kw in user_message.lower()` fires on fragments;
one later and `Inject` can no longer steer the generation it was written to steer.

**And one behaviour in it reverses the naive advice.** On a transition turn, `BosonRuleProcessor`
**swallows** the context frame and lets the Flows node be the sole inference trigger
(`respond_immediately=True` → `LLMRunFrame()` at `manager.py:707-709`). Ordering
`set_node_from_config()` before `push_frame()` does *not* fix the race, because the node's frames
enter at the head and must traverse `stt` → gate → aggregator before reaching `llm`, while your
pushed frame reaches `llm` immediately. Swallowing removes the race by construction. **The failure
modes to watch for are named**: zero generations (both paths swallowed) and two (node ran *and*
context pushed). That is unknown #4 from §1.4 and it is a *code-assertion*-type falsifier.

**The third open risk, which is a two-phase-commit blast radius.** boson rolls back by object
identity over `session.messages`; `LLMContext` offers only `set_messages(list)` with no identity
handle, and the aggregator has already written. The prototype the excerpt proposes:
snapshot/restore around the whole rule round, replay the Lina e2e suite
(`agents/test-lina-gateway/tests/`, `e2e_runner.py`), and count divergences — *"specifically turns
where a `PreTool` appended synthetic tool-call history before a later layer filtered."*

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

## 5. What has no Pipecat home and must stay boson code

Six items. This list is not a vote — it is the set of things that no vote can move, because the
destination does not exist. From [[design-boson-rules-on-pipecat]]'s migration angle plus the
per-subsystem excerpts:

1. **Cross-layer veto.** `push_frame` is irreversible; the only rollback surface is
   `LLMContext.set_messages()`. Veto across processor boundaries has no mechanism, which is why §4.2's
   candidate collapses four layers into one object.
2. **Transition legality.** `_validate_node_config` (`manager.py:867-898`) checks two things and
   neither is a from→to edge. `grep` for a legality check across `flows/` finds nothing. Keeping it
   means keeping boson's `StageMachine` class as a pre-check.
3. **`StageDefinition.skills`.** *"No Pipecat concept at all."* Lina's `product_manager` /
   `payment_manager` either flatten to functions or stay as boson meta-tools behind a `use_skill`
   direct function.
4. **Per-session attribute namespaces.** `SharedLayerContext`'s `__getattr__`/`__setattr__` proxy
   onto a live `SessionState`, so `session.sentiment_history`, `session.fired_rules`,
   `session.script_state` are dynamically attached and persist. `flow_manager.state` is a
   `dict[str, Any]` (`manager.py:143`). Pipecat has no `SessionState` at all —
   [[boson-compact-session]] calls it *"the most painful"* part of the port for exactly this reason.
5. **The `TOOL_PROCESSING` status.** boson's `AgentStatusTracker` has an enum value for it and
   *"`TOOL_PROCESSING` has no frame that means it."* `BotStartedSpeakingFrame` /
   `BotStoppedSpeakingFrame` cover the other states; the 500 ms `settling_ms` decay must be
   re-derived. This matters because `korean_fillers.py:66` reads `pre_turn_status` and the filler
   filter self-filters if it reads the wrong one.
6. **The `<system-reminder>` protocol.** `Inject` folds `<system-reminder>…</system-reminder>` into
   the *most recent user message* (`_merge_system_reminder`, `pipeline.py` L341-372). The design
   excerpt is precise about the loss: `context.add_message(...)` exists, but *"the option-β merge
   into the last user message has no frame equivalent,"* and *"the `\n---\n` separator +
   reminder-stacking convention is yours to reimplement."*

A seventh, adjacent, from [[boson-tool-router]]: **`ToolRegistry.discover_tools()`** — filesystem
discovery of `@tool`-decorated functions — has no Pipecat equivalent and stays as boson glue
producing the `tools=[...]` list.

---

## 6. What Pipecat gives back

Also not a vote. This is the other side of the same ledger: things that exist in the tree and would
arrive with an adoption, whatever you decide.

**1. Flows nodes for stages.** `NodeConfig(task_messages, role_message, functions, context_strategy,
respond_immediately)` against `StageDefinition.prompt/tools`. [[flows-insurance-example]] found
`stage_config.py` *"already is this graph, in declarative form."*

**2. `global_functions` for `_GLOBAL_TOOLS`.** An exact match, mixed in at every node:

```python
# src/pipecat/flows/manager.py L650-654
            # Build the node's function schemas (carrying handlers)
            new_functions: set[str] = set()

            # Mix in global functions that should be available at every node
            functions_list = self._global_functions + node_config.get("functions", [])
```

**3. `function` pre-actions for `PreTool`.** `actions.py:285` — `"function"` actions **always** wait,
which matches boson's synchronous-before-generation semantics. What is lost is the
preamble-as-first-stream-chunk; it becomes a separate `tts_say` action ordered before it.

**4. `ContextStrategy.RESET` as per-transition context truncation, which boson does not have
today.**

```python
# src/pipecat/flows/types.py L134-152
class ContextStrategy(Enum):
    """Strategy for managing context during node transitions.

    Parameters:
        APPEND: Append new messages to existing context (default).
        RESET: Reset context with new messages only.
        RESET_WITH_SUMMARY: Reset context but include an LLM-generated summary.

            .. deprecated:: 1.5.0
                Use :class:`LLMSummarizeContextFrame` instead — push it in a
                pre-action to trigger on-demand summarization during a node
                transition. See
                https://docs.pipecat.ai/guides/fundamentals/context-summarization.
                Will be removed in 2.0.0.
    """

    APPEND = "append"
    RESET = "reset"
    RESET_WITH_SUMMARY = "reset_with_summary"
```

**⚠️ Note which member is deprecated.** `RESET` and `APPEND` are live; **`RESET_WITH_SUMMARY` is
deprecated since 1.5.0** and removed in 2.0.0, replaced by pushing `LLMSummarizeContextFrame` in a
pre-action. [[flows-insurance-example]] cites `patient_intake.py` L306-313 and `warm_transfer.py`
L360-365 using `RESET_WITH_SUMMARY` — those examples are on the deprecated path. If you want the
warm-transfer briefing pattern for `escalate_to_human`, build it on the frame, not on the enum member.

**5. The [[ch-11/read]] observer plane, for a stack that is currently near-blind.** §3.16.

**6. COMPACTION** — split out of [[ch-09/read]] because it is a different subsystem with different
failure modes. `LLMContextSummarizer` plus `LLMAutoContextSummarizationConfig` covers what
`gateway/compact/` does, parameter for parameter: `threshold_messages=30 →
max_unsummarized_messages`, `keep_recent=10 → min_messages_after_summary`, plus a token trigger boson
never had. **With two real losses that belong on this give-back row's falsifier line**, restated from
§3.15 because they are the only part of the give-back that can bite silently:

- pre/post compact hooks have no equivalent — `on_summary_applied` is observability-only;
- the `<system-reminder>Active stage: …</system-reminder>` re-injection in `swap_compact` must be
  re-attached explicitly **or stage identity is silently lost at every compaction**.

**7. Two things from [[ch-09/read]] §12 worth listing here because they are capabilities boson has no
mechanism for.** `run_llm=False` on `FunctionCallResultProperties` — the only place in Pipecat where
a *tool* gets to say "the turn is over," which for `end_call` / `transfer_to_human` /
`schedule_callback` is a deterministic end-of-call that does not depend on model behaviour; boson's
turn ends at `agent_loop.py:363` on a text-only response, which the model must be *persuaded* to
produce. And the twelve provider adapters behind `BaseLLMAdapter.to_provider_tools_format`, against
the six shapings boson hand-writes.

---

## 7. Deployment and process topology (row 17)

Absorbed here from the old latency chapter because it is a **decision input, not a latency term**,
and building on the Lina host topology [[ch-04/read]] already delivered.

### 7.1 Pipecat ships a DEVELOPMENT runner, and says so

Its own banner:

```python
# src/pipecat/runner/run.py L392-400
def _print_dev_runner_banner():
    ...
        "ᓚᘏᗢ PIPECAT DEVELOPMENT RUNNER",
```

called at `run.py:1963`, immediately before `main()` starts the server. The module docstring calls it
*"This development runner executes Pipecat bots and provides the supporting infrastructure they
need."* And:

```bash
$ ls docs/
api
```

**There are no deployment docs in the repo** — `docs/` is Sphinx scaffolding only. This is not a
criticism; it is a scoping fact, and the guideline [[deployment-scaling]] draws from it is *"Do not
treat `pipecat.runner.run.main()` as your production entrypoint."*

### 7.2 `main()` ends at one `uvicorn.run(app, ...)` with no `workers=`

```python
# src/pipecat/runner/run.py L1999
    uvicorn.run(app, host=args.host, port=args.port)
```

No `workers=`, no reload. **One process.** Every session is an `asyncio.Task` on that one loop:

```python
# src/pipecat/runner/run.py L211-220
# Bot sessions started from a request handler outlive the response, and the event
# loop only holds a weak reference to a task, so one that nothing else references
# can be collected while it is still running.
_bot_sessions: set[asyncio.Task] = set()


def _start_bot_session(coro) -> asyncio.Task:
    """Run a bot in the background, holding a reference until it finishes."""
    task = asyncio.create_task(coro)
    _bot_sessions.add(task)
    task.add_done_callback(_bot_sessions.discard)
    return task
```

Each transport route mints `session_id = str(uuid.uuid4())` and calls `_start_bot_session(...)`
(`run.py:821, 845, 909, 1284, 1392`). WebRTC differs: `POST /start` only registers
`active_sessions[session_id] = body`, and the bot launches later from the offer handler via
`background_tasks.add_task(bot_module.bot, runner_args)` (`run.py:1002-1023`).

So: **concurrency = concurrent asyncio tasks on one loop. No process pool, no worker count, no
admission control, no per-session CPU isolation anywhere in `runner/`.**

For a Korean tele-sales agent this is the number that matters and it is worth stating in your own
units. If a call is one task and CPU-bound work inside a call blocks the loop — a resampler, a local
VAD forward pass, a JSON parse over a 64 KiB control message — it blocks *every other call in the
process*. The framework's answer to that is not in `runner/`.

### 7.3 The real runtime unit is the worker

`src/pipecat/workers/`. `BaseWorker` (1,565 L) owns activation, end/cancel, bus subscription and job
RPC. `PipelineWorker` (`pipeline/worker.py`, 1,506 L) wraps a user pipeline. `WorkerRunner`
(`workers/runner.py:83`, 550 L) owns the shared `WorkerBus` + `WorkerRegistry` and SIGINT/SIGTERM.
Per [[pipeline-task-runner]], `PipelineTask` and `PipelineRunner` are `@deprecated` 1.3.0 aliases
scheduled for removal in 2.0.0 — write against `PipelineWorker` + `WorkerRunner`.

The scaling knob is one keyword, and its docstring is the deployment advice:

```python
# src/pipecat/workers/runner.py L237-255
    async def run(
        self,
        worker: BaseWorker | None = None,
        *,
        auto_end: bool = True,
    ) -> None:
        """Run all added workers until the runner is stopped.

        By default (``auto_end=True``), the runner ends once every root
        worker has finished — so a single-pipeline bot naturally ends
        when its pipeline does. Multi-worker bots whose helpers run
        forever (e.g. waiting for bus messages) end by calling
        :meth:`end` / :meth:`cancel` from an event handler (typically on
        transport disconnect). For long-lived hosts that add and remove
        workers over many sessions (e.g. a FastAPI server), pass
        ``auto_end=False`` so the runner does not exit when no workers
        are left.
        """
```

`auto_end=True` is the container-per-call shape: one process, one call, exits when the pipeline does.
`auto_end=False` is the long-lived-host shape. **[[ch-04/read]] supplied this as the host shape for
Lina and it is the mechanism half of row 17.**

**The lifecycle safety valves** (`pipeline/worker.py` L91-100) are the framework's only cost
controls: `IDLE_TIMEOUT_SECS = 300`, `CANCEL_TIMEOUT_SECS = 20.0`, `SETUP_TIMEOUT_SECS = 20.0`,
`START_TIMEOUT_SECS = 20.0`, `HEARTBEAT_SECS = 1.0`, `HEARTBEAT_MONITOR_SECS = 10.0`, with
`idle_timeout_frames=(BotSpeakingFrame, UserSpeakingFrame)` and `cancel_on_idle_timeout=True`. An
abandoned call self-terminates after five minutes of no speech. That is the whole of it.

### 7.4 Cross-process scale-out is the bus, not the runner

`bus/local/async_queue.py` (`AsyncQueueBus`, in-process default) versus
`bus/network/{redis.py, pgmq.py}` (`RedisBus`, `PgmqBus`). From `examples/multi-worker/README.md`:
*"Distributed bus — Same patterns, but workers run in separate processes (or machines)."*
`workers/proxy/websocket/` gives point-to-point forwarding with *"No shared bus required."*

[[theory-narrow-waist]] §4 noticed what this is architecturally: **a second, parallel hourglass.**
`bus/messages.py` splits `BusMessage` into `BusDataMessage` / `BusSystemMessage` — same priority
split, different waist — and states the boundary verbatim: *"Bus messages are independent of pipeline
`Frame`s — if a worker needs to ship a frame between pipelines it wraps it in a `BusFrameMessage`."*
`BusFrameMessage(BusDataMessage)` is literally Frame-over-Bus, the IP-over-everything move inside one
codebase.

### 7.5 Cold start is `setup()`, not `StartFrame`

```python
# src/pipecat/observers/startup_timing_observer.py L95-107
class StartupTimingReport(BaseModel):
    """Report of startup timings for all measured processors.

    Parameters:
        start_time: Unix timestamp when the pipeline began setting up.
        total_duration_secs: Wall-clock time from the pipeline starting to set
            up until it had started. Processors are set up concurrently, so
            this is the span rather than the sum of what each cost.
        processor_timings: Per-processor timing data, in pipeline order.
    """
```

*"Processors are set up concurrently, so this is the span rather than the sum."* Read that as a
budgeting rule: your cold start is the **slowest** processor's `setup()`, not the total. One slow
model load dominates and eleven fast connects hide behind it. `ProcessorStartupTiming` splits
`setup_duration_secs` (connect, auth, model load) out of `duration_secs`, and `TransportTimingReport`
adds `bot_connected_secs` (SFU only) and `client_connected_secs`. End-to-end greeting cold start is
`UserBotLatencyObserver.on_first_bot_speech_latency`.

### 7.6 The entire scaling configuration surface is one line

`pipecat init` scaffolds from `cli/templates/server/`. Here is the complete generated deploy file:

```jinja
{# src/pipecat/cli/templates/server/pcc-deploy.toml.jinja2 — the whole file, 14 lines #}
agent_name = "{{ project_name }}"
secret_set = "{{ project_name }}-secrets"
{% if enable_video_input or enable_video_output %}
agent_profile = "agent-2x"
{% else %}
agent_profile = "agent-1x"
{% endif %}
{% if enable_krisp %}
[krisp_viva]
	audio_filter = "tel"
{% endif %}

[scaling]
	min_agents = 1
```

`min_agents = 1` is **the entirety of the repo's scaling configuration surface.** Sizing and
warm-pool floor are Pipecat Cloud concepts, not framework code. Expected-but-absent, per
[[deployment-scaling]]: no Kubernetes manifests, no autoscaling logic, no load-shedding, no
session-count limit, no graceful-drain helper beyond `WorkerRunner.end()`. Note also
`[krisp_viva] audio_filter = "tel"` — the template knows telephony audio exists, and it is a
commercial filter, not framework code.

### 7.7 The concrete choice, which is row 17

**One long-lived process** (`WorkerRunner(auto_end=False)`, N concurrent `PipelineWorker`s on one
loop) **versus container-per-call with a warm pool sized to concurrent-call peak.**

[[deployment-scaling]] establishes that boson-agent *is already* the first shape:
`packages/gateway/gateway/__main__.py` builds one `GatewayCore(config)` that *"owns process-scoped
resources (including MCP subprocesses),"* discovers rules/layers/stages once, and hands them to
`GatewayWebSocketServer`, whose `start()` wraps a single `websockets.serve(...)`; sessions are
multiplexed inside via `_handle_connection`, `_reserve_session_dispatch`, `_replace_active_task`,
`_cancel_session_dispatch`, `forget_session`. **So boson does not collide with the Pipecat runner —
it collides with `WorkerRunner`**, and the collision is a bookkeeping replacement:
`_cancel_active_task` → worker cancel, `_start_silence_timer` → `idle_timeout_secs` /
`idle_timeout_frames`.

**Two things that decide the row.**

**One — telephony arrives at a webhook.** Container-per-session needs a warm pool sized to
concurrent-call peak that a single-process model needs none of. That is a cost/architecture trade,
not a correctness one, and it is the reason row 17 cannot be settled by reading `runner/`.

**Two — `GatewayCore`'s process-scoped MCP subprocesses must be re-hosted above the worker so they
are not respawned per session.** This is the sharp edge. Under `auto_end=False` on one host, they
start once. Under container-per-call, per-agent MCP startup becomes cold-start cost paid on every
call, *on top of* the `min_agents` warm-pool floor. [[deployment-scaling]] flags it as *"a real risk
if Lina moves to container-per-call on Pipecat Cloud."*

**And one thing that is neutral.** Rules/layers/stage discovery is startup-time config loading and
ports as-is into `bot(runner_args)` — but then lands on the per-session cold-start path that
`StartupTimingObserver` measures. Under one long-lived process it is paid once; under
container-per-call it is paid per container. Same code, different bill.

**Now go back to §3.17 and fill in the three cells.**

---

## 8. Use the figure here

Open [`figures/migration-map.html`](figures/migration-map.html) and work the ledger in it rather than
in your head, because the figure enforces two things prose cannot.

**It makes "it depends" structurally unavailable.** Each row has three mandatory cells — vote,
free-text assumption, free-text falsifier. A row with an empty assumption or falsifier renders
incomplete in red, and the export button stays disabled until every row is complete. There are no
pre-selected votes, no defaults, no highlighted recommendations, and no `KEEP` / `ADOPT` /
`HYBRID-WRAP` label anywhere you did not click.

**It flags combinations that cannot both be true.** Three, with the reason attached:

- adopt Pipecat's LLM service while keeping text-native Gateway tools (§4.1);
- keep realtime_voice's Silero while adding telephony, which hard-rejects 8 kHz (§3.1, §3.7);
- drop `AudioTextPlayoutLedger` while choosing a TTS outside the six word-timestamp services (§3.3,
  §3.5) — with the reminder that those six are a grep, not a behavioural test.

**And it is where the one measurement lands.** Paste the two numbers from §2 into the MEASUREMENT
panel. Until you do, the panel sits greyed with the run command as its only content; after you do, it
draws your bar against ch-12's ghost estimate and re-renders [[ch-11/read]]'s waterfall with a real
number in the rule-evaluation slot. Both values persist in `localStorage`.

**One sentence on what to do with it:** cast all seventeen votes in one sitting, then walk the list a
second time and delete every assumption that would not change its vote if it flipped — what survives
the second pass is the real assumption set, and it will be much shorter than the first.

---

## 9. The deliverables

Five artifacts. This is what "done" means for this chapter.

**1. The completed seventeen-row vote table, with assumptions and falsifiers.** Exported from the
figure. Every row filled. No blanks, no "TBD," no row where the assumption restates the vote.

**2. The target architecture drawn end to end, with `FlowManager` outside the processor list.** Not
inside it — `FlowManager` is a plain class (§3.12) that drives from outside via
`worker.queue_frames`. If your diagram shows it in the pipeline, the diagram encodes a mechanism
error and every downstream conclusion about transition ordering inherits it.

**3. A migration sequence that keeps the agent shippable at every step, ordered by you and defended
in one sentence per step.** The constraint is the interesting part: *shippable at every step* means
no step may leave Lina unable to take a call. That constraint alone eliminates several orderings — it
is why "swap the LLM loop first" and "swap the transport first" are different in kind, not just in
order. One sentence per step, and the sentence should name what the step de-risks, not what it
accomplishes.

**4. `ch-13/tier2-probe.py`, run at least once, with its printed P50/P95 pasted into the figure — and
a note on what that did to the [[ch-11/read]] budget.** That last clause is the deliverable, not the
running. Did the pre-LLM half of the budget stay inside the P50 ≤ 1.0 s line with the measured value
in it, or did it not? Write the sentence.

**5. The remaining falsifiers as a watchlist with owners — not as a reason to postpone the
decision.** §10.

---

## 10. The watchlist: the three unrun measurements

These are falsifiers, not blockers. Each gets an owner and a precondition. Styled that way in the
figure too — beneath the measurement panel, explicitly as falsifiers.

| # | Observation | Type | Rows it would overturn | Precondition | Owner |
|---|---|---|---|---|---|
| 1 | Korean WER on 8 kHz μ-law, per candidate STT | benchmark | 2, and by coupling 1 and 7 | a Korean STT contract + a labelled Lina corpus + telephony-band audio | |
| 2 | interruption-broadcast → `TranscriptionFrame` gap, over lone Korean backchannels | corpus | 1, 5, and the filler-gate position in §4.2 | recorded 8 kHz telephony audio + a running Pipecat pipeline | |
| 3 | exactly one inference-triggering frame per turn under the transition-swallow design | code assertion | 10, 12, 13 | a built `BosonRuleProcessor` prototype + a `FrameLogger` at the `llm` input | |

**Notes that make each one actionable rather than aspirational.**

**#1.** The harness exists: `https://github.com/pipecat-ai/stt-benchmark`, named in
[[stt-korean-providers]]. What does not exist is your labelled corpus. That is the real precondition,
and it is a data-labelling task, not an engineering one. The shortlist to run it against is the
Korean-verified nine from §3.2, ordered by an **English-assumed** latency table — which is itself
part of why the benchmark matters.

**#2.** [[design-boson-rules-on-pipecat]] §5 states the expected sign in advance: *"If the gap is
positive (it will be, ~always)."* boson filters `"네"` by *content* and `pre_turn_status`; Pipecat
interrupts on VAD energy upstream of STT, so the bot is already interrupted before the gate sees
text. If confirmed, the consequence is named: *"a custom `BaseUserTurnStartStrategy` … that withholds
turn-start until a transcript exists is mandatory — and it costs the unmeasured Korean STT TTFS."*
Note the coupling: this falsifier's remedy costs the number #1 measures. Also note, from
[[boson-interrupt-subsystem]], that the exact analogue of boson's text-only detector already exists
upstream as `TranscriptionUserTurnStartStrategy`, so boson's current behaviour is *a supported
Pipecat configuration* — which changes what this measurement would cost you to act on.

**#3.** The assertion, verbatim from the excerpt: a `FrameLogger` at the `llm` input asserting that
exactly **one** inference-triggering frame arrives per turn and that `LLMSetToolsFrame` precedes it.
The two named failure modes are zero generations (both paths swallowed) and two (node ran *and*
context pushed). This is the cheapest of the three to obtain — it needs no audio and no vendor — but
it needs a prototype that does not exist yet, which is why it is on the watchlist rather than in §2.

**A discipline note.** An unowned falsifier is a wish. When you export the table, put a name in every
owner cell, including your own, and put a date next to the precondition. A falsifier with an owner
and no date is a wish with a name on it.

---

## 11. What to hold in your head

Twelve, in the shape the ledger uses.

1. **Seventeen rows.** The chapter, the figure, and your export all say seventeen. If any of them
   says sixteen or eighteen, one of them drifted.
2. **A vote without an assumption is a guess.** Rule one's second half is the half that does the work.
3. **A falsifier names an observation, not a preference.** If nothing you could see would overturn a
   row, the row is mis-stated — re-state it or split it.
4. **One measurement is run.** Tier-2 P50/P95, `ch-13/tier2-probe.py`, because it is the only one of
   the four that needs no carrier, no STT contract, and no audio.
5. **The offline figure cannot call an endpoint.** Measurement lives in a process; visualisation
   lives in a browser. Never assign a job to an artifact that structurally cannot do it.
6. **Nearest-rank P95 at N=9 is the maximum.** Run at least 40 iterations or you are reporting your
   worst luck.
7. **`round(0.2 / (512/16000)) = 6`, and `round(0.2 / (256/8000)) = 6`.** Pipecat's endpointing frame
   count is invariant to sample rate; the excerpt's "7" is wrong.
8. **Deepgram's in-tree TTFS P99 is `0.35`, not `0.45`.** The design excerpt's comparison understates
   the Tier-2 bill.
9. **The six Korean word-timestamp TTS services are a grep, not a behavioural test.** Upper bound on
   candidates, not a verified set.
10. **No Korean accuracy number and no 8 kHz number exist anywhere in the tree.** Zero grep hits for
    WER, any service, any language.
11. **`min_agents = 1` is the entire scaling configuration surface**, and `uvicorn.run(app, ...)` has
    no `workers=`. Concurrency is asyncio tasks on one loop until you build otherwise.
12. **Two collisions are irreducible** — the agent boundary (rows 10, 11) and the rule layers
    (row 13) — and this chapter stated both without resolving either, on purpose.

And one sentence about the whole exercise: **the ledger's value is not that the votes are right, it
is that the assumptions and falsifiers make it cheap to find out that one of them was wrong.** A
seventeen-row table with tripwires is an asset. A seventeen-row table of confident votes with nothing
written next to them is the same document with the useful half deleted.

---

## 다음 챕터로

There is no next chapter. This is the last one, and what it hands forward is not a chapter — it is
four files and a habit.

**The four files.** The exported vote table; the target architecture with `FlowManager` outside the
processor list; the migration sequence with one defended sentence per step; and `tier2-probe.py`
with a real number in its output and that number pasted into the figure. Put all four somewhere your
team can review, because rule one's whole purpose was to make them reviewable.

**The habit.** You spent twelve chapters being told not to decide, and one chapter deciding
seventeen times under stated assumptions with written tripwires. That asymmetry was the pedagogy: the
evidence had to be complete before the vote was allowed, and the vote had to be falsifiable before it
counted. Carry that shape into the next architecture decision you make at Lina, which will not be
about Pipecat.

**And re-run the probe.** Not once — whenever the model changes, whenever the serving stack changes,
whenever someone adds a third `check_type="llm"` rule. It takes forty seconds and it is the only
number in this entire course that you made yourself.

Two closing pointers, because they will come up the moment you start building.
[[design-boson-rules-on-pipecat]]'s own migration angle is one sentence long and worth re-reading
after you have voted: *"the port is one processor, not a framework port."* And
[[pipecat-design-philosophy]]'s deprecation registry — 391 live deprecations, all
`removed_in == "2.0.0"`, 97% with a named replacement — is the best migration backlog you will find,
because it tells you your exact future breakage before you write a line.
