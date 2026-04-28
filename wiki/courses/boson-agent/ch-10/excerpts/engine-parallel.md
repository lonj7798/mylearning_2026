# === CALLING SPEC ===
# PURPOSE: Deep walkthrough of the parallel phase and _run_check helper
# CALLED BY: read.md (ch-10 index)
# CALLS: nothing
# PURE: yes
# DETERMINISTIC: yes

---
chapter: ch-10
course: boson-agent
phase: read
excerpt_of: engine-parallel
source_file: boson-agent/packages/gateway/gateway/rules/engine.py
created_at: "2026-04-19"
---

# Excerpt: Parallel Phase and `_run_check` — `gateway/rules/engine.py`

> The parallel phase is a single `asyncio.gather` call wrapped in a result
> collector. `_run_check` is the normalisation adapter that makes sync and
> async checks indistinguishable to the caller and converts every return
> value into `list[Action]`.

---

## Source: Parallel phase of `evaluate` + `_run_check`

```python
# gateway/rules/engine.py, lines 62-106

        # --- parallel phase ---
        if self._parallel:
            results = await asyncio.gather(
                *[
                    self._run_check(fn, messages, user_message, session)
                    for fn in self._parallel
                ],
                return_exceptions=False,               # (A)
            )
            for actions in results:
                non_continue = [a for a in actions if a.type not in ("continue", "pass")]
                collected.extend(non_continue)         # (B)

        return collected if collected else [Continue()]  # (C)

    async def _run_check(
        self,
        check_fn,
        messages: list,
        user_message: Any,
        session: Any,
    ) -> list[Action]:
        """Invoke a single check function, handling sync/async and exceptions."""
        try:
            if inspect.iscoroutinefunction(check_fn):  # (D)
                result = await check_fn(messages, user_message, session)
            else:
                result = check_fn(messages, user_message, session)

            # Normalise: single Action or list[Action]
            if isinstance(result, Action):             # (E)
                return [result]
            if isinstance(result, list):
                return result
            return [Continue()]                        # (F)
        except Exception as exc:
            if self._fail_open:                        # (G)
                logger.warning(
                    "Check '%s' raised an exception (fail_open=True): %s",
                    getattr(check_fn, "__check_name__", repr(check_fn)),
                    exc,
                    exc_info=True,
                )
                return [Continue()]
            raise
```

---

## Annotation Key

**(A) `return_exceptions=False`**

```python
results = await asyncio.gather(
    *[self._run_check(fn, ...) for fn in self._parallel],
    return_exceptions=False,
)
```

`asyncio.gather(..., return_exceptions=False)` means: if any coroutine raises
an unhandled exception, `gather` immediately cancels the remaining coroutines
and re-raises the exception to the caller. This looks dangerous, but it is
safe here because `_run_check` already catches all exceptions internally
(via the `try/except` at line 97) and returns `[Continue()]` when
`fail_open=True`. So under normal operating conditions, no coroutine passed
to `gather` will ever raise — `_run_check` is a total function.

The `return_exceptions=False` choice means that if `fail_open=False` AND a
check raises, the exception propagates out of `evaluate` and up to
`GatewayCore.handle_message`, which will surface it as an error to the client.
This is the correct fail-hard behaviour when `fail_open` is disabled.

**(B) All non-CONTINUE results are collected**

```python
for actions in results:
    non_continue = [a for a in actions if a.type not in ("continue", "pass")]
    collected.extend(non_continue)
```

Unlike the sequential phase (which `break`s on the first non-Continue), the
parallel phase iterates all results and accumulates everything. If three
parallel checks each return `Inject(...)`, all three injections end up in
`collected`. The `ActionExecutor` receives them all and applies them in order.

This is the defining difference between sequential and parallel semantics:
sequential is a winner-takes-all gate; parallel is a union of all opinions.

**(C) `return collected if collected else [Continue()]`**

If both sequential and parallel phases produced only Continue/Pass results,
`collected` is empty. Rather than returning an empty list (which would force
every caller to handle the empty case), the engine returns `[Continue()]` as
a sentinel. This invariant — "evaluate always returns at least one action" —
simplifies the ActionExecutor and GatewayCore considerably.

**(D) `inspect.iscoroutinefunction(check_fn)`**

```python
if inspect.iscoroutinefunction(check_fn):
    result = await check_fn(messages, user_message, session)
else:
    result = check_fn(messages, user_message, session)
```

This is the sync/async bridge. A check decorated with `@check("foo")` and
defined as `def foo(...)` (sync) is called directly. One defined as
`async def foo(...)` is awaited. The `inspect.iscoroutinefunction` check is
resolved at call time, not at construction time, so there is no pre-sorting
of checks into sync vs async buckets.

For the sequential phase, this means sync checks do not block the event loop
— they run synchronously within the `await _run_check(...)` frame, which is
fine because the sequential phase is inherently serial. For the parallel phase,
sync checks wrapped in `_run_check` coroutines are also fine: when
`asyncio.gather` schedules them, the sync path executes immediately and
returns, effectively making the coroutine a completed future. This is correct
but means sync parallel checks offer no concurrency benefit — to get true
concurrency in the parallel phase you need `async def` checks.

**(E–F) Return value normalisation**

```python
if isinstance(result, Action):
    return [result]
if isinstance(result, list):
    return result
return [Continue()]
```

Rule authors can return either a single `Action` or a `list[Action]`. The
engine normalises both to `list[Action]` so all downstream code deals with
a uniform type. If a check returns `None` or anything unexpected, the engine
treats it as `Continue()` — a safe, non-breaking default.

**(G) Fail-open exception handling**

```python
except Exception as exc:
    if self._fail_open:
        logger.warning(
            "Check '%s' raised an exception (fail_open=True): %s",
            getattr(check_fn, "__check_name__", repr(check_fn)),
            exc,
            exc_info=True,
        )
        return [Continue()]
    raise
```

`fail_open=True` (the default) means any exception in any check is logged as
a warning and the check's result is treated as `Continue()`. The conversation
proceeds as if the check did not exist. This is a deliberate safety-over-
correctness tradeoff: in a live production conversation, a crashing rule
should not end the call. The operator sees the warning in logs and can fix the
rule in a subsequent deployment.

`fail_open=False` re-raises the exception, letting it propagate to
`handle_message` and surface as a client-facing error. Use this in test
environments where you want a crashing rule to be immediately visible.

The `getattr(check_fn, "__check_name__", repr(check_fn))` in the warning
ensures the log message names the failing check correctly even if the function
object is somehow missing its `__check_name__` stamp (defensive programming).

---

## Notice: `asyncio.gather` and the shared `session` object

All parallel checks receive the **same** `session` object. If two parallel
checks both write to `session.some_attribute`, the last write wins — Python
object mutation is not thread-safe, but asyncio is single-threaded, so races
can only occur at `await` boundaries. A parallel check that does:

```python
session.counter += 1        # read
await some_io()             # yield point
session.counter += 1        # write
```

...could see a stale `session.counter` if another parallel check also mutated
it between the read and the write. The framework does not protect against this.
Rule authors are expected to design parallel checks to be read-only on
`session`, or to use non-overlapping attributes.

The `transition_detector.py` in Lina avoids this by writing to dedicated
attributes (`session.checklist_state`, `session.escalate_count`) that no
other parallel check touches.

---

## Connection to the Universal Pattern

The parallel phase maps to step 3 of the universal pattern (concurrent
evaluation). `asyncio.gather` is the substrate mechanism that makes true
concurrency possible within a single Python event loop turn — it is not an
arbitrary design choice but a direct consequence of the asyncio programming
model. Without `gather`, concurrent LLM checks would have to be run
sequentially, multiplying latency by the number of LLM checks.

See also: [[engine-sequential.md]] for how sequential and parallel phases
compose, [[check-decorator.md]] for how `mode="parallel"` is stamped.
