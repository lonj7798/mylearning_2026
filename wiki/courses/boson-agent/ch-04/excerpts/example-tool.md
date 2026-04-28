---
chapter: ch-04
course: boson-agent
phase: read
excerpt_of: boson-agent/agents/demo/tools/ and boson-agent/agents/test-lina/tools/
created_at: "2026-04-19"
---

# Excerpt: Example @tool Implementations

This page walks through four real tool files across two agents — from the minimal
one-liner to a production-grade async data loader — showing how the `@tool` contract
scales.

---

## 1. Minimal sync tool — `calculate.py`

```python
# boson-agent/agents/demo/tools/calculate.py, lines 1-15

from basement.tools.decorator import tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result. Example: '2 + 3 * 4' returns '14'."""
    try:
        # Safe eval for basic math
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

**Schema produced by `_generate_schema`:**

```json
{
  "type": "object",
  "properties": {
    "expression": {"type": "string"}
  },
  "required": ["expression"]
}
```

One parameter, one required field, one primitive type. The LLM must always supply
`expression`; there is no optional fallback.

Notice the function itself handles its own errors by returning an error string rather
than raising. This is a valid pattern — `execute_tool` would catch any exception anyway
and set `is_error=True`, but returning a descriptive string inside the happy path gives
the LLM richer context to self-correct ("Error: invalid characters in expression" is
more actionable than "Tool error: ValueError: ...").

---

## 2. Optional parameter / default value — `get_time.py`

```python
# boson-agent/agents/demo/tools/get_time.py, lines 1-17

"""Demo tool: get current time (simulated)."""
from basement.tools.decorator import tool


@tool
def get_time(timezone: str = "UTC") -> str:
    """Get current time for a timezone.

    Args:
        timezone: Timezone name (e.g., UTC, KST, EST, PST)
    """
    from datetime import datetime, timedelta

    offsets = {"UTC": 0, "KST": 9, "EST": -5, "PST": -8, "JST": 9, "CET": 1}
    offset = offsets.get(timezone.upper(), 0)
    now = datetime.utcnow() + timedelta(hours=offset)
    return f"{timezone.upper()}: {now.strftime('%Y-%m-%d %H:%M:%S')}"
```

**Schema produced:**

```json
{
  "type": "object",
  "properties": {
    "timezone": {"type": "string"}
  },
  "required": []
}
```

`timezone` has a default (`"UTC"`), so `param.default is inspect.Parameter.empty`
is `False` and it does **not** appear in `required`. The LLM may omit the parameter;
Python's default kicks in. This is the only mechanism for optional tool parameters —
give the Python parameter a default value.

The docstring's `Args:` section is included verbatim in `description` (the whole
`func.__doc__.strip()` is used). The LLM sees the inline documentation exactly as
written. Well-structured docstrings directly improve tool-call accuracy.

---

## 3. No-parameter tool — `check_dnc_status.py`

```python
# boson-agent/agents/test-lina/tools/check_dnc_status.py, lines 1-29

"""Check if a customer is on the Do-Not-Call list."""
import yaml
from pathlib import Path
from basement.tools.decorator import tool
from _session import get_active_customer

DATA_DIR = Path(__file__).parent.parent / "data"
CUSTOMER_DB = DATA_DIR / "customers" / "customer_db.yaml"


@tool
def check_dnc_status() -> str:
    """Check if a customer is on the Do-Not-Call list."""
    customer_id = get_active_customer()
    with open(CUSTOMER_DB) as f:
        data = yaml.safe_load(f)

    customers = data.get("customers", {})
    if customer_id not in customers:
        return f"Error: Customer '{customer_id}' not found."

    customer = customers[customer_id]
    status = customer.get("dnc_status", False)
    name = customer.get("name", customer_id)

    if status:
        return f"{name} ({customer_id}) is ON the Do-Not-Call list. Do not proceed with sales."
    return f"{name} ({customer_id}) is NOT on the Do-Not-Call list. OK to proceed."
```

**Schema produced:**

```json
{
  "type": "object",
  "properties": {},
  "required": []
}
```

Zero parameters. `_generate_schema` iterates zero parameters and returns an object with
empty `properties` and `required`. The LLM calls this tool with `{}` as input, which is
valid JSON Schema.

Notice the `from _session import get_active_customer` — this import works because
`_import_module_from_path` inserts the tool file's parent directory into `sys.path`
before executing the module (see [[./registry]]). The `_session.py` file is a private
shared module (underscore prefix → skipped by discovery) used by multiple tools in the
same agent.

This is the pattern for session-scoped state: keep it in a `_session.py` module,
import it from each tool that needs it. The registry never loads `_session.py` as a
tool file, but it is loadable as a regular module because it is on `sys.path`.

---

## 4. Production-scale tool — `escalate_to_human.py`

```python
# boson-agent/agents/test-lina/tools/escalate_to_human.py, lines 1-26

"""Register customer for transfer to a human agent."""
import hashlib
from basement.tools.decorator import tool
from _session import get_active_customer


@tool
def escalate_to_human(reason: str) -> str:
    """Register customer for transfer to a human agent.

    Use when the customer's request requires human handling
    (complex complaints, special requests, etc.).

    Args:
        reason: Reason for escalation
    """
    customer_id = get_active_customer()
    ref = hashlib.md5(f"{customer_id}-{reason}".encode()).hexdigest()[:8].upper()
    return (
        f"Escalation registered.\n"
        f"  Customer: {customer_id}\n"
        f"  Reason: {reason}\n"
        f"  Reference: ESC-{ref}\n"
        f"  A human agent will contact the customer within 24 hours."
    )
```

**Schema produced:**

```json
{
  "type": "object",
  "properties": {
    "reason": {"type": "string"}
  },
  "required": ["reason"]
}
```

The multiline docstring is used in full as the tool description. The first line serves
as a brief summary; subsequent lines provide usage guidance ("Use when..."). This richer
docstring pattern helps the LLM decide *when* to call the tool, not just *how*.

The return value is a pre-formatted multiline string. `execute_tool` calls `str()` on
it (a no-op for strings), and it lands in `ToolResultBlock.content` verbatim. The LLM
reads this formatted block and can incorporate the reference number into its response.

---

## Schema Generation Summary Table

| Tool | Params | Required | Optional | Schema shape |
|------|--------|----------|----------|--------------|
| `calculate` | `expression: str` | `expression` | — | `{string}` |
| `get_time` | `timezone: str = "UTC"` | — | `timezone` | `{string, default via Python}` |
| `check_dnc_status` | none | — | — | `{}` |
| `escalate_to_human` | `reason: str` | `reason` | — | `{string}` |
| `search_docs` | `query: str` | `query` | — | `{string}` |

The JSON Schema `required` array is the sole mechanism that marks a parameter as
mandatory to the LLM. Python defaults handle the optional case transparently.

---

## The Invariant Across All Examples

Every tool file follows the same three-step contract:

1. `from basement.tools.decorator import tool` — single import.
2. `@tool` on a function with a docstring and typed parameters.
3. Return any value with a meaningful `str()` representation.

Nothing else is required. No base class, no registration call, no config entry.
