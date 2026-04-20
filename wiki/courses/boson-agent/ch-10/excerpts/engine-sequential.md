# === CALLING SPEC ===
# PURPOSE: Deep walkthrough of the sequential phase of RuleEngine.evaluate
# CALLED BY: read.md (ch-10 index)
# CALLS: nothing
# PURE: yes
# DETERMINISTIC: yes

---
chapter: ch-10
course: boson-agent
phase: read
excerpt_of: engine-sequential
source_file: boson-agent/packages/gateway/gateway/rules/engine.py
created_at: "2026-04-19"
---

# Excerpt: Sequential Phase — `gateway/rules/engine.py`

> The sequential phase is a priority-sorted for-loop with a single break.
> That break is the entire policy: the first rule that has something to say
> owns this turn. Everything else in the sequential pipeline is silenced.

---

## Source: `RuleEngine.__init__` and sequential phase of `evaluate`

```python
# gateway/rules/engine.py, lines 22-60

class RuleEngine:
    """Runs rule checks and collects resulting actions.

    Sequential checks run in priority order; the first non-CONTINUE result
    short-circuits the rest of the sequential pipeline.
    Parallel checks all run concurrently; all non-CONTINUE results are collected.
    """

    def __init__(self, checks: list, fail_open: bool = True) -> None:
        self._fail_open = fail_open
        self._sequential: list = sorted(
            [c for c in checks if c.__check_mode__ == "sequential"],
            key=lambda c: c.__check_priority__,          # (A)
        )
        self._parallel: list = sorted(
            [c for c in checks if c.__check_mode__ == "parallel"],
            key=lambda c: c.__check_priority__,
        )

    async def evaluate(
        self,
        messages: list,
        user_message: Any,
        session: Any,
    ) -> list[Action]:
        """Run all checks and return collected actions.

        Returns [Continue()] when no check produces a non-CONTINUE action.
        """
        collected: list[Action] = []

        # --- sequential phase ---
        for check_fn in self._sequential:                # (B)
            actions = await self._run_check(check_fn, messages, user_message, session)
            non_continue = [a for a in actions if a.type not in ("continue", "pass")]  # (C)
            if non_continue:
                collected.extend(non_continue)
                # Short-circuit: stop sequential pipeline on first non-CONTINUE
                break                                    # (D)
```

---

## Annotation Key

**(A) Sorting at construction time, not at evaluation time**

```python
self._sequential: list = sorted(
    [c for c in checks if c.__check_mode__ == "sequential"],
    key=lambda c: c.__check_priority__,
)
```

The sort happens once in `__init__`, not on every call to `evaluate`. This is
a deliberate performance choice: for a gateway handling thousands of turns per
minute, eliminating a repeated sort (O(n log n) per turn) matters. The
tradeoff is that `_sequential` is frozen at construction time — you cannot
add checks to a live engine and have them sorted in. You must construct a new
`RuleEngine` to change the check set.

The `key=lambda c: c.__check_priority__` reads the `__check_priority__`
attribute stamped by `@check`. Python's `sorted()` is stable, so checks with
the same priority value retain their insertion order (which is the
alphabetical file-discovery order from the registry).

**(B) The sequential loop**

```python
for check_fn in self._sequential:
    actions = await self._run_check(check_fn, messages, user_message, session)
```

This is an `async for` pattern implemented as a regular `for` loop with
`await` inside. Each check is awaited sequentially — the next check does not
start until the current one completes. This is precisely what "sequential"
means: strict ordering with no concurrency within this phase.

`_run_check` handles both sync and async check functions transparently (see
[[engine-parallel.md]] for that detail). From the loop's perspective, every
check is awaitable.

**(C) Filtering out Continue and Pass**

```python
non_continue = [a for a in actions if a.type not in ("continue", "pass")]
```

Two action types are treated as "no opinion": `continue` (explicit
`Continue()`) and `pass` (explicit `Pass()`). Both mean "I have nothing to
say, keep going." The distinction between them is semantic for rule authors
(`Pass` is idiomatically used when the check deliberately chose not to act,
`Continue` when it actively approves), but the engine treats them identically.

This list comprehension is applied to every check's result, even if the check
returned a single `Action` (because `_run_check` normalises single actions
into `[action]`).

**(D) The break — the entire short-circuit policy**

```python
if non_continue:
    collected.extend(non_continue)
    break
```

This three-line block is the policy decision that defines sequential mode.
When a check returns a non-Continue action:

1. The action(s) are added to `collected`.
2. The `for` loop immediately exits.
3. No subsequent sequential check runs.

The implication for rule authors: **priority is load-bearing**. A
`priority=10` spam filter that returns `Filter()` will prevent a
`priority=20` greeting responder from ever seeing the message. This is
intentional — sequential checks are for ordered gatekeeping, not for
concurrent annotation.

---

## What Happens After the Sequential Phase

After the `break` (or after all sequential checks pass), execution falls
through to the parallel phase. If sequential produced a `collected` result,
the parallel phase still runs — the two phases are additive, not mutually
exclusive. This means a blocking sequential result (e.g., `Respond`) can
coexist with parallel annotations (e.g., `Inject`) in `collected`. The
`ActionExecutor` then decides which actions to honour and in what order.

In practice, a `Respond` action in `collected` will cause `ActionExecutor` to
set `result.should_continue = False`, and GatewayCore will short-circuit
before the agent loop (see [[../read.md]] §Overview for the full flow).

---

## Notice: Why "first non-CONTINUE wins" rather than "all are collected"?

Sequential checks are designed for **policy gates** — rules that are
mutually exclusive. If a spam filter blocks a message, it makes no sense for
a subsequent greeting detector to also fire on the same message. The first
non-CONTINUE wins semantics enforces exactly one policy decision per turn in
the sequential phase.

Contrast with parallel mode (see [[engine-parallel.md]]): parallel checks
are designed for **independent enrichment** — multiple checks annotate the
same message from different angles, and all results are wanted.

---

## Demo: Priority Ordering in `spam_filter.py`

```python
# agents/demo-gateway/layers/01-guard/rules/spam_filter.py, lines 12-28

@check("spam_filter", mode="sequential", priority=10)
def spam_filter(messages, user_message, session):
    for spam in SPAM_WORDS:
        if spam in lower:
            return Filter(reason=f"spam:{spam}")
    return Pass()

@check("greeting_responder", mode="sequential", priority=20)
def greeting_responder(messages, user_message, session):
    if lower in GREETING_WORDS and len(messages) == 0:
        return Respond(text="Hello! ...")
    return Pass()
```

With `priority=10`, `spam_filter` runs first. If it returns `Filter(...)`,
`greeting_responder` (priority=20) never executes. If `spam_filter` returns
`Pass()`, the loop continues and `greeting_responder` has its turn. The
priority gap of 10 leaves room to insert a new check at priority 15 without
renumbering.

See also: [[check-decorator.md]] for how `priority` is stamped onto the
function, [[registry.md]] for how checks are discovered and fed to the engine.
