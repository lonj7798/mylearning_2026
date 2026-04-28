# === CALLING SPEC ===
# PURPOSE: Deep walkthrough of the @check decorator source
# CALLED BY: read.md (ch-10 index)
# CALLS: nothing
# PURE: yes
# DETERMINISTIC: yes

---
chapter: ch-10
course: boson-agent
phase: read
excerpt_of: check-decorator
source_file: boson-agent/packages/gateway/gateway/rules/check.py
created_at: "2026-04-19"
---

# Excerpt: The `@check` Decorator — `gateway/rules/check.py`

> One 43-line file. Every `@check`-decorated function in the codebase carries
> metadata stamped here. Nothing else in the framework touches this module —
> it is the definition end of a two-sided contract whose enforcement end lives
> in `engine.py`.

---

## Full Source with Line-by-Line Commentary

```python
# gateway/rules/check.py, lines 1-43

# === CALLING SPEC ===
# PURPOSE: @check decorator and Check protocol
# CALLED BY: User rule files, rules/engine
# CALLS: schemas/actions
# PURE: yes
# DETERMINISTIC: yes (sealed)

"""@check decorator — annotates functions as rule checks."""

from typing import Callable, Literal

CheckMode = Literal["sequential", "parallel"]          # (A)
CheckType = Literal["deterministic", "llm"]            # (B)


def check(
    name: str,
    *,
    mode: CheckMode = "sequential",                    # (C)
    priority: int = 100,                               # (D)
    check_type: CheckType = "deterministic",           # (E)
) -> Callable:
    """Decorator that marks a function as a Gateway rule check.

    Args:
        name: Unique identifier for this check.
        mode: "sequential" checks run one-at-a-time; "parallel" checks run concurrently.
        priority: Lower number = higher priority (runs first within each mode).
        check_type: "deterministic" for pure functions; "llm" for async LLM-backed checks.

    The decorated function must accept (messages, user_message, session) and return
    a list[Action] or a single Action.
    """

    def decorator(fn: Callable) -> Callable:
        fn.__check_name__ = name                       # (F)
        fn.__check_mode__ = mode                       # (G)
        fn.__check_priority__ = priority               # (H)
        fn.__check_type__ = check_type                 # (I)
        return fn                                      # (J)

    return decorator
```

---

## Annotation Key

**(A) `CheckMode = Literal["sequential", "parallel"]`**

A `Literal` type alias, not a class or enum. This is intentional: the mode
value is stored as a plain string on the function object at (G), so registry
and engine code can inspect it with `c.__check_mode__ == "sequential"` without
importing any special type. Using a `Literal` gives the type checker
exhaustiveness guarantees without the runtime overhead of an enum.

**(B) `CheckType = Literal["deterministic", "llm"]`**

Same pattern. `check_type` is metadata only — the engine does not branch on
it at runtime. It exists for tooling, documentation, and future runtime
optimisations (e.g., a scheduler could refuse to run `llm` checks when a
token budget is exhausted). For now, it is purely informational.

**(C) `mode: CheckMode = "sequential"`**

The default is sequential. This is the safer default: a developer who omits
`mode=` gets a check that runs in the ordered, short-circuiting pipeline,
which is predictable. Opting into `parallel` is an explicit declaration that
the check has no side effects that depend on other checks' outcomes.

**(D) `priority: int = 100`**

Lower number = runs first. Default of `100` places an undecorated check in
the middle of the range. Safety checks that must run before anything else
use priorities like `1`, `5`, `10`. The numbers are arbitrary integers; the
engine sorts by them, so gaps are fine (you can insert a new check at `15`
without renumbering `10` and `20`).

**(E) `check_type: CheckType = "deterministic"`**

Again the safer default. An async function that happens to not call an LLM
could still be `deterministic`. The convention is: use `"llm"` when the check
issues a network call to an inference endpoint, so ops can trace or rate-limit
those calls separately.

**(F–I) The stamp block**

```python
fn.__check_name__ = name
fn.__check_mode__ = mode
fn.__check_priority__ = priority
fn.__check_type__ = check_type
```

These four lines are the entire implementation. The decorator does not wrap
the function, does not change its signature, does not add a `__call__` layer.
It stamps four dunder attributes directly onto the function object and returns
it unchanged. This is the lightest possible form of metadata attachment in
Python.

The consequence: a `@check`-decorated function **is still just a function**.
You can call it directly in tests with `spam_filter(messages, msg, session)` —
no unwrapping required. The metadata only matters when the registry discovers
it (`hasattr(obj, "__check_name__")`) and when the engine reads it
(`c.__check_mode__`, `c.__check_priority__`).

**(J) `return fn`**

The function object is returned unmodified. This contrasts with decorators
like `@functools.wraps` that wrap or replace the function. Because the
original function is returned, `inspect.iscoroutinefunction(check_fn)` in
the engine correctly detects whether the original was `async def`.

---

## Notice: Why Not a Class-Based Check?

The framework could have required checks to be classes with a `run()` method
and metadata as class attributes. It chose plain functions + dunder stamps
instead. This decision has three consequences:

1. **Zero boilerplate for authors.** A check is a one-liner `def` with a
   decorator. No `class`, no `super().__init__()`, no `self`.

2. **Direct callability in tests.** `spam_filter(messages, "buy now", session)`
   works without any test fixtures or factory methods.

3. **Registry detection is duck-typed.** `hasattr(obj, "__check_name__")` is
   all that is needed. Any callable with that attribute is a valid check,
   regardless of how it was created. This opens the door to programmatic
   check generation without touching the decorator at all.

---

## Connection to the Universal Pattern

The `@check` decorator is step 0 of the universal pattern: **stamp → sort →
evaluate → collect**. The stamp phase (this file) encodes two scheduling
decisions (`mode`, `priority`) and two metadata fields (`name`,
`check_type`) onto the function object. The sort and evaluate phases happen
in `engine.py`. The collect phase produces the action list that
`GatewayCore.handle_message` consumes.

Without this stamp, the registry has no way to distinguish a check function
from any other module-level callable. The `__check_name__` attribute is the
single marker that makes a function opt in to the rule system.

See also: [[../read.md]] for the full chapter, [[engine-sequential.md]] for
how `priority` drives execution order.
