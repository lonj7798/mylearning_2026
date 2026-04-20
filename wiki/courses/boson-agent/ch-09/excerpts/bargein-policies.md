---
chapter: ch-09
course: boson-agent
phase: read
excerpt_for: bargein-policies
created_at: "2026-04-19"
sources_cited:
  - "boson-agent/packages/gateway/gateway/interrupt/policy.py"
  - "boson-agent/docs/plan/v0_4/02-phase1-bargein-policies.md"
---

# Excerpt: Barge-in Policies

---

## The policy class hierarchy

```python
# boson-agent/packages/gateway/gateway/interrupt/policy.py, lines 13-116

class BargeInResult(Enum):
    ALLOW = "allow"
    IGNORE = "ignore"

    @property
    def is_bargein(self) -> bool:
        return self == BargeInResult.ALLOW


class BargeInPolicy:
    def evaluate(self, text: str, *, elapsed_ms: float = 0) -> BargeInResult:
        raise NotImplementedError


class AlwaysPolicy(BargeInPolicy):
    def evaluate(self, text: str, *, elapsed_ms: float = 0) -> BargeInResult:
        return BargeInResult.ALLOW


class DurationPolicy(BargeInPolicy):
    def __init__(self, min_ms: float = 500) -> None:
        self.min_ms = min_ms

    def evaluate(self, text: str, *, elapsed_ms: float = 0) -> BargeInResult:
        if elapsed_ms >= self.min_ms:
            return BargeInResult.ALLOW
        return BargeInResult.IGNORE


class WordFilterPolicy(BargeInPolicy):
    def __init__(
        self,
        ignore_words: list[str] | None = None,
        max_chars: int = 3,
    ) -> None:
        self.ignore_words = frozenset(
            w.lower() for w in (ignore_words or ["hmm", "uh", "um", "ah"])
        )
        self.max_chars = max_chars

    def evaluate(self, text: str, *, elapsed_ms: float = 0) -> BargeInResult:
        stripped = text.strip().lower()
        if stripped in self.ignore_words:
            return BargeInResult.IGNORE
        if len(stripped) <= self.max_chars:
            return BargeInResult.IGNORE
        return BargeInResult.ALLOW


class CompositePolicy(BargeInPolicy):
    def __init__(
        self,
        policies: list[BargeInPolicy],
        mode: str = "all",
    ) -> None:
        self.policies = policies
        self.mode = mode

    def evaluate(self, text: str, *, elapsed_ms: float = 0) -> BargeInResult:
        if not self.policies:
            return BargeInResult.IGNORE
        results = [p.evaluate(text, elapsed_ms=elapsed_ms) for p in self.policies]
        if self.mode == "any":
            if any(r.is_bargein for r in results):
                return BargeInResult.ALLOW
            return BargeInResult.IGNORE
        else:  # "all"
            if all(r.is_bargein for r in results):
                return BargeInResult.ALLOW
            return BargeInResult.IGNORE


def default_bargein_policy() -> CompositePolicy:
    """Sensible default: requires both duration AND meaningful content."""
    return CompositePolicy(
        policies=[
            DurationPolicy(min_ms=500),
            WordFilterPolicy(ignore_words=["hmm", "uh", "um", "ah"], max_chars=3),
        ],
        mode="all",
    )
```

---

## Mechanical walkthrough

**`BargeInResult.is_bargein`** is a property on the enum — a small but
important design choice. Callers write `result.is_bargein` not `result ==
BargeInResult.ALLOW`. This makes the predicate readable at the call site
(`InterruptHandler.check_barge_in` returns `policy.evaluate(...).is_bargein`)
and insulates callers from the raw enum value.

**`DurationPolicy`** solves the false-positive problem from voice interfaces:
when STT (speech-to-text) is streaming live, the first partial transcript
arrives within ~100ms of the agent beginning to speak. Without a duration
gate, every agent turn would immediately barge-in on itself. The 500ms default
is chosen to be longer than typical STT first-token latency but shorter than a
typical conversational pause.

**`WordFilterPolicy`** solves a different false-positive: filler sounds ("hmm",
"uh") that a voice interface sends as confirmed transcripts. These are real
messages from the protocol's perspective but are not meaningful interruptions.
The `frozenset` is computed once at construction — not on every `evaluate()`
call — which is correct for a hot path.

**`CompositePolicy`** is a strategy combinator. The `mode="all"` default
implements AND-logic: both duration AND content tests must pass. `mode="any"`
implements OR-logic. The empty-policy case returns `IGNORE` rather than
`ALLOW` — a safe default (fail closed: if no policies configured, do not
interrupt).

**`default_bargein_policy()`** hard-codes the sensible production default as a
function, not a module-level singleton. This matters: if it were a singleton,
`set_interrupt_tags()` called after import would not affect already-constructed
policies. As a factory function, each agent can call `default_bargein_policy()`
and get a fresh instance with whatever tag configuration is currently active.

**Notice — all policies are stateless after construction:** `evaluate()` takes
`elapsed_ms` as a parameter rather than tracking it internally. The caller
(`InterruptHandler.check_barge_in`, and ultimately `core.handle_message`) is
responsible for computing elapsed time from the session's turn-start timestamp.
This keeps the policies pure and easily testable — a policy instance can be
evaluated repeatedly with different elapsed times without side effects.

**Notice — wiring gap:** `GatewayCore.set_bargein_policy()` stores the policy
in `self._bargein_policy`. The field is used in zero places inside
`handle_message` in the current code. The policies are fully implemented and
tested in isolation, but the call to `check_barge_in()` that would use them
during an active stream does not yet exist. This connects to the broader
integration gap documented in [[excerpts/two-interrupt-handlers]].

Connection to universal pattern: the policy hierarchy is the textbook Strategy
pattern applied to a binary decision (ALLOW/IGNORE). The substrate — a
streaming WebSocket protocol that delivers messages asynchronously regardless
of agent state — forces any barge-in system to make this decision for every
incoming frame while a response is in flight. The Strategy pattern isolates
the decision logic from the delivery mechanism.
