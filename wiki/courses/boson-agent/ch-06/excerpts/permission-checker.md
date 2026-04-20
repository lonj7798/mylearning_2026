---
calling_spec:
  purpose: Deep walkthrough of PermissionChecker — deny-overrides-allow gate
  chapter: ch-06
  course: boson-agent
  phase: read
  excerpt_of: read.md
---

# PermissionChecker — Deep Walkthrough

The permission system spans three files: `config_schema.py` (data model), `loader.py`
(factory), and `checker.py` (enforcement). Only `checker.py` needs close study —
the others are thin wrappers.

---

## `checker.py` — The Enforcement Gate

```python
# packages/basement/basement/permissions/checker.py, lines 14-67

class PermissionChecker:
    """Evaluates allow/deny rules from PermissionConfig.

    Rules:
    - If denied_tools/denied_skills contains the name -> always denied.
    - If allowed_tools/allowed_skills is None -> no allow-list restriction.
    - If allowed_tools/allowed_skills is a list -> only listed names pass.
    - Deny always overrides allow when both match.
    """

    def __init__(self, config: PermissionConfig) -> None:
        self._denied_tools: frozenset[str] = frozenset(config.denied_tools)
        self._allowed_tools: frozenset[str] | None = (
            frozenset(config.allowed_tools) if config.allowed_tools is not None else None
        )
        self._denied_skills: frozenset[str] = frozenset(config.denied_skills)
        self._allowed_skills: frozenset[str] | None = (
            frozenset(config.allowed_skills) if config.allowed_skills is not None else None
        )

    def check_tool(self, name: str) -> bool:
        """Return True if the tool is permitted; raise PermissionDeniedError otherwise."""
        if name in self._denied_tools:
            raise PermissionDeniedError(f"Tool '{name}' is denied by permissions.")
        if self._allowed_tools is not None and name not in self._allowed_tools:
            raise PermissionDeniedError(
                f"Tool '{name}' is not in the allowed_tools list."
            )
        return True

    def check_skill(self, name: str) -> bool:
        """Return True if the skill is permitted; raise PermissionDeniedError otherwise."""
        if name in self._denied_skills:
            raise PermissionDeniedError(f"Skill '{name}' is denied by permissions.")
        if self._allowed_skills is not None and name not in self._allowed_skills:
            raise PermissionDeniedError(
                f"Skill '{name}' is not in the allowed_skills list."
            )
        return True

    def is_allowed(self, name: str, target_type: str = "tool") -> bool:
        """Non-raising permission check. Returns False instead of raising."""
        try:
            if target_type == "skill":
                self.check_skill(name)
            else:
                self.check_tool(name)
            return True
        except PermissionDeniedError:
            return False
```

**Mechanical explanation — line by line.**

`__init__` converts the lists from `PermissionConfig` into `frozenset` objects
immediately. `frozenset` gives O(1) membership tests and is immutable after
construction — the checker is sealed for its lifetime. The `_allowed_tools` field is
typed `frozenset[str] | None`. When `config.allowed_tools` is `None` (the default),
`_allowed_tools` stays `None`, meaning "no allowlist restriction." When it is a list,
it becomes a frozen set of exactly those names.

`check_tool` encodes the **deny-overrides-allow** rule in two sequential if-statements:

1. **Deny check first.** `if name in self._denied_tools` — if the name appears in the
   deny list, raise immediately. No further checks.
2. **Allowlist check second.** `if self._allowed_tools is not None and name not in
   self._allowed_tools` — if an allowlist exists AND the name is absent from it, raise.

Because deny runs first, a name that appears in both `denied_tools` and `allowed_tools`
is always blocked. This is the invariant the docstring calls "deny always overrides
allow when both match."

`is_allowed` is a non-raising wrapper. It delegates to `check_tool` / `check_skill`
and catches `PermissionDeniedError`, returning `False` instead. This is used by callers
that need a boolean (e.g., filtering a tool list for display) rather than an exception
(e.g., blocking execution).

---

## Interaction with ToolRouter

`PermissionChecker.check_tool` is called at two distinct points:

1. **Inside `ToolRouter.dispatch()`** (router.py line 104):

```python
# packages/basement/basement/metatool/router.py, lines 104-105
if self._permission_checker is not None:
    self._permission_checker.check_tool(tool_name)  # raises on denial
```

This is the **runtime enforcement** point. Every tool call — native or MCP — passes
through here. A `PermissionDeniedError` raised here propagates up to `run_agent_loop`,
which catches it via the `ON_ERROR` hook path.

2. **Inside the `use_skill` handler** (injector.py line 62):

```python
# packages/basement/basement/skills/injector.py, lines 62-63
if permissions is not None:
    permissions.check_skill(skill_name)
```

This is the **skill enforcement** point. The check runs inside the tool handler before
registry lookup — if the skill is denied, the handler raises before any injection
occurs.

---

## `loader.py` — Factory (trivial but worth seeing)

```python
# packages/basement/basement/permissions/loader.py, lines 14-23

def load_permissions(config: AgentConfig) -> PermissionChecker:
    """Create a PermissionChecker from the permissions block of an AgentConfig."""
    return PermissionChecker(config.permissions)
```

One line. `__main__.py` calls `load_permissions(config)` once at startup and passes the
resulting `PermissionChecker` to both `ToolRouter.__init__` and `create_use_skill`.
After that, the checker is read-only and shared between both sites.

---

## The `allowed_tools=None` Semantics (Carryforward from ch-02)

When `config.yaml` omits `permissions.allowed_tools` (the default), the YAML parser
sets it to `None`. `load_permissions` passes that through to `PermissionConfig`.
`PermissionChecker.__init__` leaves `_allowed_tools = None`. In `check_tool`, the
second condition short-circuits: `if self._allowed_tools is not None` is `False`, so
the branch never executes. Every tool name passes the allowlist check automatically.

This is also how `ToolRouter.set_allowed_tools(None)` works at the stage level: setting
`_allowed_tools = None` on the router opens the stage gate entirely, reverting to
permission-only enforcement.

The two mechanisms — `PermissionChecker._allowed_tools` and
`ToolRouter._allowed_tools` — are independent filters applied in sequence:

```
ToolRouter.dispatch(name)
  → PermissionChecker.check_tool(name)   [global policy]
  → router stage gate                    [per-stage policy]
  → dispatch_table lookup
  → handler()
```

A tool must clear both filters to execute.
