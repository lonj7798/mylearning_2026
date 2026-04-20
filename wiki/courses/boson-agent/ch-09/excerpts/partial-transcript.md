---
chapter: ch-09
course: boson-agent
phase: read
excerpt_for: partial-transcript
created_at: "2026-04-19"
sources_cited:
  - "boson-agent/packages/gateway/gateway/interrupt/detector.py"
  - "boson-agent/packages/gateway/gateway/interrupt/handler.py"
  - "boson-agent/packages/gateway/gateway/schemas/session.py"
  - "boson-agent/packages/gateway/gateway/server/websocket.py"
---

# Excerpt: Partial Transcripts — PartialBuffer, PartialDetector, and Silence Timer

---

## PartialBuffer: the session field

```python
# boson-agent/packages/gateway/gateway/schemas/session.py, lines 21-27

@dataclass
class PartialBuffer:
    """Buffer for partial transcript updates."""

    text: str
    timestamp: float  # time.monotonic()
    message_index: int  # index in session.messages where partial is stored
```

`PartialBuffer` records three things: the current partial text, when it
arrived (monotonic clock), and where in `session.messages` the corresponding
partial `Message` lives. The `message_index` is the key field: it lets the
gateway find and replace the in-progress user message in-place without
appending a new one. This preserves history coherence — the LLM's view of the
conversation does not accumulate a stream of half-typed messages.

---

## PartialDetector: content overlap + timing heuristic

```python
# boson-agent/packages/gateway/gateway/interrupt/detector.py, lines 22-82

class PartialDetector:
    """Detects partial transcript updates using content + timing."""

    def __init__(
        self,
        overlap_chars: int = 10,
        timing_threshold_ms: float = 1000,
        silence_timeout_ms: float = 2000,
    ) -> None:
        self.overlap_chars = overlap_chars
        self.timing_threshold_ms = timing_threshold_ms
        self.silence_timeout_ms = silence_timeout_ms

    def is_partial(self, text: str, previous: str | None) -> bool:
        """Check if text is a partial update of previous via content overlap."""
        if not previous:
            return False
        compare_len = min(self.overlap_chars, len(previous))
        if compare_len == 0:
            return False
        return text[:compare_len] == previous[:compare_len]

    def is_likely_partial_by_timing(self, elapsed_ms: float) -> bool:
        """Check if timing suggests this is a partial update."""
        return elapsed_ms < self.timing_threshold_ms

    def detect(
        self,
        text: str,
        previous: str | None,
        elapsed_ms: float,
    ) -> DetectResult:
        """Combined detection: content overlap is primary, timing secondary."""
        if self.is_partial(text, previous):
            return DetectResult.PARTIAL
        if previous and self.is_likely_partial_by_timing(elapsed_ms):
            return DetectResult.PARTIAL
        return DetectResult.NEW_MESSAGE


def replace_partial_in_history(
    messages: list[Message],
    new_text: str,
    index: int,
) -> None:
    """Replace a partial message in the history list at given index."""
    if 0 <= index < len(messages):
        messages[index] = Message(role="user", content=new_text)
```

The detection logic uses two signals in sequence. Primary: content overlap —
if the first N characters of the new text match the first N characters of the
previous partial, it is an update. Secondary: timing — if the messages arrived
within `timing_threshold_ms` of each other and there is a previous partial,
treat as update even without character overlap (accounts for cases where the
user corrected the beginning of the sentence).

**Notice — `is_partial` uses `min(overlap_chars, len(previous))`:** If the
previous partial is shorter than `overlap_chars` (e.g., "hi" with
`overlap_chars=10`), it only compares the available 2 characters. This avoids
false negatives for very short partials. The downside: a one-character previous
partial matches almost anything that starts with the same letter, so short
partials have a high false-PARTIAL rate. The timing gate partially mitigates
this.

**`replace_partial_in_history`** is a module-level function (not a method)
because it operates on a plain list. It replaces `messages[index]` with a new
`Message` object — it does not mutate the existing object — which means any
code holding a reference to the old `Message` will not see the update. This is
correct because the only reference holder that matters is the
`ContextManager._messages` list, which is the same list object as
`session.messages` (see [[excerpts/two-interrupt-handlers]] for the shared
reference design in `SharedHistory`).

---

## Silence timer in the WebSocket server

```python
# boson-agent/packages/gateway/gateway/server/websocket.py, lines 162-277

# v0.4: Handle partial transcript
if msg.type == "partial_transcript":
    async for _ in self._message_handler(msg.session_id, msg.content):
        pass
    self._start_silence_timer(msg.session_id, websocket)
    continue

# ...

def _start_silence_timer(
    self, session_id: str, websocket: websockets.server.ServerConnection
) -> None:
    self._cancel_silence_timer(session_id)

    async def _on_silence() -> None:
        await asyncio.sleep(self._silence_timeout_ms / 1000)
        await self._finalize_partial(session_id, websocket)

    self._session_timers[session_id] = asyncio.create_task(_on_silence())


async def _finalize_partial(
    self, session_id: str, websocket: websockets.server.ServerConnection
) -> None:
    try:
        async for chunk in self._message_handler(session_id, ""):
            await websocket.send(
                serialize_server_message(
                    ServerMessage(
                        session_id=session_id, type="text_delta", content=chunk
                    )
                )
            )
        await websocket.send(
            serialize_server_message(
                ServerMessage(session_id=session_id, type="turn_end")
            )
        )
    except Exception:
        logger.exception("finalize_partial failed for session %s", session_id)
```

The silence timer implements the "end of utterance" detector for streaming
speech input. When a `partial_transcript` frame arrives, the handler is called
(which runs `detect_and_handle_partial` and replaces the message in history),
then a timer is (re)started for `silence_timeout_ms`. If no further
`partial_transcript` arrives before the timer fires, `_finalize_partial` is
called with an empty content string (`""`), which signals to `core.handle_message`
that the partial is done and the agent should respond.

**Notice — `_start_silence_timer` cancels the previous timer first:** Each
new partial transcript resets the silence window. This is correct: the user is
still speaking as long as STT keeps sending partials. The timer only fires
when speech pauses. Without this reset, the timer would fire at the original
deadline regardless of continued input.

**Notice — `_finalize_partial` calls `self._message_handler(session_id, "")`.
The empty string `""` is the signal:** `core.handle_message` receives `content=""`.
In the current implementation, if `content` is `""` and `partial_buffer` is set,
`detect_and_handle_partial` receives an empty string. `is_partial("", previous)`
returns `False` (empty string cannot have character overlap), and
`is_likely_partial_by_timing` returns `True` if within threshold. This edge
case means an empty finalization call within 1000ms of the last partial is
treated as another partial update (replacing the history entry with an empty
string), which is undesirable. This is a latent bug in the interaction between
the silence timer and the detector when the timeout is shorter than
`timing_threshold_ms`.

Connection to universal pattern: the partial transcript system is the gateway
equivalent of backpressure on a streaming input — instead of processing every
intermediate state as a full turn, the gateway accumulates updates in-place and
triggers a turn only on silence. This is forced by the substrate (STT systems
stream partials before finalizing) and the cost model (each agent turn calls
the LLM, so processing every partial would be prohibitively expensive).
