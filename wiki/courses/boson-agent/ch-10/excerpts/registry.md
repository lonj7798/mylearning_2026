# === CALLING SPEC ===
# PURPOSE: Deep walkthrough of CheckRegistry — discovery and registration
# CALLED BY: read.md (ch-10 index)
# CALLS: nothing
# PURE: yes
# DETERMINISTIC: yes

---
chapter: ch-10
course: boson-agent
phase: read
excerpt_of: registry
source_file: boson-agent/packages/gateway/gateway/rules/registry.py
created_at: "2026-04-19"
---

# Excerpt: `CheckRegistry` — `gateway/rules/registry.py`

> The registry is the bridge between the filesystem and the engine. It turns
> `.py` files in a `rules/` directory into a sorted list of stamped callables
> that `RuleEngine` can iterate over. It knows nothing about what checks do —
> only that they carry `__check_name__`.

---

## Full Source with Line-by-Line Commentary

```python
# gateway/rules/registry.py, lines 1-64

class CheckRegistry:
    """Discover and store @check decorated rule functions."""

    def __init__(self) -> None:
        self._checks: list = []                             # (A)

    def discover_checks(self, checks_dir: Path) -> int:
        """Import all .py files in checks_dir and register @check functions.

        Returns count of discovered checks.
        """
        if not checks_dir.exists():
            return 0                                        # (B)

        count = 0
        for py_file in sorted(checks_dir.glob("*.py")):    # (C)
            if py_file.name.startswith("_"):
                continue                                    # (D)
            try:
                module = _import_module_from_path(py_file) # (E)
                for obj in vars(module).values():           # (F)
                    if hasattr(obj, "__check_name__"):
                        self.register(obj)
                        count += 1
            except Exception as exc:
                logger.error("Failed to load check from %s: %s", py_file, exc)
                continue                                    # (G)

        return count

    def register(self, fn) -> None:
        """Register a @check function."""
        self._checks.append(fn)
        logger.debug("Registered check: %s", getattr(fn, "__check_name__", repr(fn)))

    def get_all(self) -> list:
        """Return checks ordered by mode (sequential first) then priority."""
        return sorted(
            self._checks,
            key=lambda c: (
                0 if c.__check_mode__ == "sequential" else 1,  # (H)
                c.__check_priority__,                           # (I)
            ),
        )
```

---

## Annotation Key

**(A) `self._checks: list = []`**

A plain mutable list. The registry is stateful — it accumulates discovered
checks across multiple `discover_checks` calls (one per layer in a layered
gateway setup). There is no deduplication: if the same function object is
registered twice, it will appear twice in `get_all()`. In practice, each
layer has its own registry instance, so this is not a concern.

**(B) `if not checks_dir.exists(): return 0`**

Graceful handling of a missing `rules/` directory. An agent with no rules is
valid — the engine will have no checks, and `evaluate()` will always return
`[Continue()]`. This is a deliberate "works out of the box" affordance.

**(C) `sorted(checks_dir.glob("*.py"))`**

The glob returns files in filesystem order (which is undefined and
OS-dependent). The `sorted()` call normalises to lexicographic order. This
means `01_safety.py` loads before `02_context.py`, giving developers a
numbering convention to control load order. Note that load order does not
affect execution order (that is controlled by `priority`), but it does affect
which check "wins" a tie in `sorted()` when two checks share the same
priority value (Python sort is stable).

**(D) `if py_file.name.startswith("_"): continue`**

Files starting with `_` are skipped. This mirrors Python's convention for
private modules (`__init__.py`, `_helpers.py`). A rule author can place
shared utilities in `_utils.py` without accidentally registering them as
checks.

**(E) `_import_module_from_path(py_file)`**

This function is borrowed from `basement.tools.registry` — the same module
importer used for tool discovery. It dynamically imports a `.py` file by
path, using `importlib.util.spec_from_file_location` and
`importlib.util.module_from_spec`. The module is fully initialised (all
top-level code runs) before `vars(module)` is inspected.

This means a rule file's module-level code (e.g., defining `SPAM_WORDS = {...}`)
runs at discovery time, not at check-evaluation time. Large data structures
loaded at discovery time are effectively cached for the lifetime of the
gateway process.

**(F) `for obj in vars(module).values(): if hasattr(obj, "__check_name__")`**

`vars(module)` returns the module's `__dict__` — all names defined in the
module. The registry inspects every object and looks for the `__check_name__`
dunder attribute. Any callable with this attribute is a check. This is pure
duck typing: the registry does not import `check` from `gateway.rules.check`
and compare types. It only asks "does this object have the marker?"

This design means a programmatically created function with `fn.__check_name__ = "my_check"`
set manually would also be discovered and registered. The decorator is
convention, not enforcement.

**(G) `except Exception: logger.error(...); continue`**

If a rule file fails to import (syntax error, missing dependency, etc.), the
error is logged and the registry moves on to the next file. This fail-soft
behaviour is analogous to `fail_open` in the engine: a broken rule file
should not prevent the gateway from starting. Other rule files in the same
directory are unaffected.

**(H–I) Dual-key sort in `get_all()`**

```python
key=lambda c: (
    0 if c.__check_mode__ == "sequential" else 1,
    c.__check_priority__,
)
```

`get_all()` returns checks in the order: all sequential checks first (sorted
by priority), then all parallel checks (sorted by priority). This ordering is
used by `GatewayCore` or `__main__` when constructing the `RuleEngine`. The
engine will then re-sort `_sequential` and `_parallel` separately in its own
`__init__`, so `get_all()`'s ordering is not the final execution order —
but it provides a stable, human-readable view of all registered checks.

---

## Notice: Registry vs Engine Sort — Redundant But Deliberate

Both `CheckRegistry.get_all()` and `RuleEngine.__init__` sort the checks.
This looks redundant. It is redundant — but intentionally so. `get_all()` is
a public API consumed by any code that wants to inspect registered checks
(e.g., a CLI tool that prints the check list on startup). `RuleEngine.__init__`
sorts its own internal lists independently because it cannot trust the input
ordering. Separating concerns this way means neither class depends on the
other's sorting behaviour.

The `__main__.py` of the gateway package uses `CheckRegistry.get_all()` to
print a startup summary ("Checks: 4") before constructing the engine.

---

## Connection to the Universal Pattern

The registry is the **discover** step that precedes the universal
stamp → sort → evaluate → collect pattern. Without it, the engine would need
to know about every check at construction time (hardcoded imports). The
filesystem-scan approach means adding a new rule is as simple as dropping a
`.py` file into the `rules/` directory — no registration call, no import, no
factory. The framework discovers it automatically on the next startup.

See also: [[check-decorator.md]] for what `__check_name__` is,
[[engine-sequential.md]] for how the sorted list is consumed.
