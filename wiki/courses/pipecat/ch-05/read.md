---
title: "Transports: WebRTC, WebSocket, and Telephony-as-a-Serializer"
chapter: ch-05
phase: voice-io
course: pipecat
sources:
  - transport-websocket
  - transport-telephony
  - transport-daily-webrtc
  - boson-gateway-server
  - rtv-webrtc-transport
deps:
  - ch-01
  - ch-02
  - ch-03
figure: figures/transport-comparison.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
verified: 2026-08-25
---

# Chapter 5 — Transports: WebRTC, WebSocket, and Telephony-as-a-Serializer

> **Scope, stated up front and enforced for the whole chapter.** This chapter describes what each
> design **does**. It does not rank them. There is no sentence in here saying that Pipecat's
> transport layer is better than `gateway/server/`, or that `realtime_voice/transport/webrtc/`
> should be kept or thrown away. Every "adopting X costs Y" line in §9, §10 and §12 is an
> accounting statement — lines you keep, lines you port, capabilities that appear, capabilities
> that disappear — not a recommendation. [[ch-13/read]] is the only chapter in this course that
> scores anything. If you find yourself wanting the verdict while reading §9, that is the chapter
> working: it is assembling the evidence that ch-13 spends.

---

## 왜 이 챕터인가

[[ch-04/read]] built the runtime and then deliberately left two holes in the canonical chain:

```
transport.input() → stt → user_aggregator → llm → tts → transport.output() → assistant_aggregator
        ▲                                                        ▲
     position 1                                             position 6
     (empty in ch-04)                                       (empty in ch-04)
```

This chapter fills them. And the first thing to internalise is that **filling them needs almost
nothing from ch-04**. A transport is not a runtime object. It is not scheduled, it does not own a
task, it does not know what a `PipelineWorker` is. `BaseTransport` is a 42-line class whose entire
public surface is two abstract methods that each return a `FrameProcessor`. The prerequisites for
this chapter are therefore [[ch-01/read]] — the uniform interface `process_frame` / `push_frame`,
and a `link()` that validates nothing — and [[ch-02/read]] — the narrow waist, `Frame` as an open
sum type. Not the runtime.

That matters because it means the transport question is a *composition* question, not an
*infrastructure* question, and the answer is small enough to hold in your head:

> A Pipecat transport is a pair of processors. Everything provider-specific either lives behind
> the connection object those processors wrap, or lives in a `FrameSerializer` you hand to them
> through a params field. There is no third place.

The second reason this chapter exists: it is **the first chapter where boson has a concrete
counterpart**. [[ch-03/read]] characterised `realtime_voice` as a whole. Here the comparison gets
file-and-line specific on two of your own subsystems at once — `packages/gateway/gateway/server/`
(1,404 lines, text protocol, no audio) and `packages/realtime_voice/realtime_voice/transport/webrtc/`
(aiortc, the same library Pipecat's `SmallWebRTCTransport` uses).

And the third reason, the one that actually decides things for Lina TMR: **there is no telephony
transport in Pipecat.** A phone call is a WebSocket transport plus a ~300-line serializer. Which
means the migration question for a Korean insurance tele-sales agent is not "which transport do I
pick" — it is "what does the serializer I have to write look like, and what does 8 kHz μ-law do to
Korean STT." §6 is the longest section in this chapter for that reason.

---

## 0. How to read the evidence in this chapter

Two classes of claim, as in [[ch-03/read]] §0, and they are not equally checkable.

**Class A — Pipecat.** Every path, line number, class name, count and LOC figure is checkable
against `wiki/raw-data/pipecat/pipecat-src` at commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`.
Where I print a command, I ran it. Where a curated excerpt disagrees with the tree, I say so and
the tree wins. There are **four** such corrections in this chapter and they are flagged inline
with **⚠️ CORRECTION**.

**Class B — boson.** `gateway/server/` and `realtime_voice/transport/webrtc/` live in private
repos that are not on this machine. Every number about them comes from [[boson-gateway-server]] and
[[rtv-webrtc-transport]], which were read directly from branches `lina-new-dental-dev` and
`voice-chat-dev`. Those figures are excerpt-attested, not clone-verified, and I mark boson code
blocks with a comment header saying so. One of them has an internal arithmetic problem that I
cannot resolve without the repo; §10.1 flags it rather than repeating it.

---

## 1. `BaseTransport` is 42 lines and declares exactly two methods

Start with the whole abstraction, because it fits on one screen.

**`src/pipecat/transports/base_transport.py` L96–137** (the file is 137 lines; this class is the
last thing in it, so `BaseTransport` is lines 96 through 137 — **42 lines**)

```python
class BaseTransport(BaseObject):
    """Base class for transport implementations.

    Provides the foundation for transport classes that handle media streaming,
    including input and output frame processors for audio and video data.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the base transport.

        Args:
            name: Optional name for the transport instance.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(name=name)
        self._input_name = input_name
        self._output_name = output_name

    @abstractmethod
    def input(self) -> FrameProcessor:
        """Get the input frame processor for this transport.

        Returns:
            The frame processor that handles incoming frames.
        """
        pass

    @abstractmethod
    def output(self) -> FrameProcessor:
        """Get the output frame processor for this transport.

        Returns:
            The frame processor that handles outgoing frames.
        """
        pass
```

Read the return types. `input()` and `output()` return `FrameProcessor` — the same type as every
other element of a `Pipeline`. Not a `Transport`-flavoured processor, not a subtype with extra
methods the pipeline knows about. The same type.

That single fact is the whole design. Recall [[ch-01/read]] §3: `link()` is two pointer assignments
and a `logger.debug`, and `Pipeline._link_processors` is a fold over that operator with no
validation. So when the canonical bot writes

```python
Pipeline([transport.input(), stt, context_aggregator.user(), llm, tts, transport.output(), ...])
```

the pipeline is not doing anything special at positions 1 and 6. It is folding `link` over a list
of `FrameProcessor`s, and two of them happen to have a socket behind them. `BaseTransport` itself
is not in the pipeline at all — it is a *factory* for the two processors that are, plus a holder
for the connection they share.

Consequences, in order of how load-bearing they are for you:

1. **Swapping a provider changes the connection object and the params class, nothing else.**
   Replacing `DailyTransport(room_url, token, bot_name, params=DailyParams(...))` with
   `SmallWebRTCTransport(webrtc_connection, params=TransportParams(...))` changes two lines of
   construction. The `Pipeline([...])` line is byte-identical. This is [[transport-daily-webrtc]]'s
   Core Insight and the tree bears it out.
2. **A transport is unit-testable the same way any processor is** — [[ch-01/read]] §6's
   substitutability argument applies unchanged, which is why `tests/test_websocket_server_transport.py`
   can define a nine-line `_RawAudioSerializer(FrameSerializer)` and drive the whole transport
   without a network.
3. **There is no place in `BaseTransport` to put policy.** No auth hook, no session concept, no
   protocol. If you want those, they go above (your ASGI route) or beside (a processor) or inside
   (the serializer). Hold that thought until §9 — it is the entire reason `access.py` survives a
   migration.

### 1.1 The other half of the file: `TransportParams`

Media configuration is a flat pydantic model, declarative, no methods.

**`src/pipecat/transports/base_transport.py` L66–93**

```python
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_out_enabled: bool = False
    audio_out_sample_rate: int | None = None
    audio_out_channels: int = 1
    audio_out_bitrate: int = 96000
    audio_out_10ms_chunks: int = 4
    audio_out_mixer: BaseAudioMixer | Mapping[str | None, BaseAudioMixer] | None = None
    audio_out_destinations: list[str] = Field(default_factory=list)
    audio_out_end_silence_secs: int = 2
    audio_out_auto_silence: bool = True
    audio_out_write_timeout_secs: float = 10.0
    audio_in_enabled: bool = False
    audio_in_sample_rate: int | None = None
    audio_in_channels: int = 1
    audio_in_filter: BaseAudioFilter | None = None
    audio_in_stream_on_start: bool = True
    audio_in_passthrough: bool = True
    video_in_enabled: bool = False
    video_out_enabled: bool = False
    video_out_is_live: bool = False
    video_out_width: int = 1024
    video_out_height: int = 768
    video_out_bitrate: int | None = None
    video_out_framerate: int = 30
    video_out_color_format: str = "RGB"
    video_out_codec: str | None = None
    video_out_destinations: list[str] = Field(default_factory=list)
```

Three defaults worth burning in for Lina:

- `audio_out_enabled: bool = False` and `audio_in_enabled: bool = False`. **Audio is off by
  default.** A transport constructed with a bare `TransportParams()` carries no audio in either
  direction and will look, from inside the pipeline, exactly like a working transport that happens
  to receive nothing. This is the single most common first-hour failure.
- `audio_out_10ms_chunks: int = 4`. This is the *N* from [[ch-04/read]] §5.1 (`control latency =
  queue depth / drain rate`) in its concrete form: the output side hands the wire 40 ms at a time.
  It reappears in §3.1 below, again in [[ch-08/read]] as interrupt granularity, and again in
  [[ch-11/read]] as a latency-budget term.
- Sample rates default to `None`. They are not "unset means 16000" — they resolve at setup, which
  is §3.

**⚠️ CORRECTION 1 (minor, but the outline states the listing as exhaustive).** The course outline
records `ls src/pipecat/transports/` as returning `base_input.py`, `base_output.py`,
`base_transport.py` and eleven directories. The tree also contains `__init__.py`:

```bash
$ ls src/pipecat/transports/
__init__.py  base_input.py  base_output.py  base_transport.py
daily/  heygen/  lemonslice/  livekit/  local/  moq/  smallwebrtc/
tavus/  vonage/  websocket/  whatsapp/
```

The load-bearing part of the outline's claim is untouched — **no `twilio/`, no `telnyx/`, no
`plivo/`, no `exotel/`, no `sip/`** — but if you are going to quote the listing, quote it whole.

---

## 2. Counting the inventory, and the place the counts stop being 1:1

[[ch-03/read]] §6.2 already gave you "11 transport packages" as a breadth number against
realtime_voice's one. Here is what that number is actually made of, because the composition is more
interesting than the count.

```bash
$ ls -d src/pipecat/transports/*/ | wc -l
11

$ grep -rn "(BaseTransport)" src/pipecat/ | sort
src/pipecat/transports/daily/transport.py:2279:class DailyTransport(BaseTransport):
src/pipecat/transports/heygen/transport.py:293:class HeyGenTransport(BaseTransport):
src/pipecat/transports/lemonslice/transport.py:711:class LemonSliceTransport(BaseTransport):
src/pipecat/transports/livekit/transport.py:1054:class LiveKitTransport(BaseTransport):
src/pipecat/transports/local/audio.py:203:class LocalAudioTransport(BaseTransport):
src/pipecat/transports/local/tk.py:246:class TkLocalTransport(BaseTransport):
src/pipecat/transports/moq/transport.py:1390:class MOQTransport(BaseTransport):
src/pipecat/transports/smallwebrtc/transport.py:951:class SmallWebRTCTransport(BaseTransport):
src/pipecat/transports/tavus/transport.py:883:class TavusTransport(BaseTransport):
src/pipecat/transports/vonage/video_connector.py:329:class VonageVideoConnectorTransport(BaseTransport):
src/pipecat/transports/websocket/client.py:479:class WebsocketClientTransport(BaseTransport):
src/pipecat/transports/websocket/fastapi.py:611:class FastAPIWebsocketTransport(BaseTransport):
src/pipecat/transports/websocket/server.py:518:class SingleClientWebsocketServerTransport(BaseTransport):
```

**11 packages, 13 subclasses.** The outline explains the gap as "some packages ship more than one",
which is true and incomplete. Do the per-package accounting:

| Package | `BaseTransport` subclasses | What it actually is |
|---|---|---|
| `daily/` | 1 — `DailyTransport` | SFU room, native client SDK |
| `heygen/` | 1 — `HeyGenTransport` | avatar vendor |
| `lemonslice/` | 1 — `LemonSliceTransport` | avatar vendor |
| `livekit/` | 1 — `LiveKitTransport` | SFU room |
| `local/` | **2** — `LocalAudioTransport`, `TkLocalTransport` | host sound card; Tk window |
| `moq/` | 1 — `MOQTransport` | Media over QUIC |
| `smallwebrtc/` | 1 — `SmallWebRTCTransport` | direct aiortc peer |
| `tavus/` | 1 — `TavusTransport` | avatar vendor |
| `vonage/` | 1 — `VonageVideoConnectorTransport` | **video**, unrelated to `serializers/vonage.py` |
| `websocket/` | **3** — `FastAPIWebsocketTransport`, `SingleClientWebsocketServerTransport`, `WebsocketClientTransport` | server-side socket, dev server, dial-out client |
| `whatsapp/` | **0** | — |
| | **13** | |

### 2.1 The zero is the interesting cell

`whatsapp/` is a transport *package* with no transport *class*:

```bash
$ ls src/pipecat/transports/whatsapp/
__init__.py  api.py  client.py

$ grep -rn "^class " src/pipecat/transports/whatsapp/client.py
src/pipecat/transports/whatsapp/client.py:35:class WhatsAppClient:
```

And what `WhatsAppClient` does is import somebody else's connection object:

**`src/pipecat/transports/whatsapp/client.py` L24, L70, L342**

```python
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
...
        self._ongoing_calls_map: dict[str, SmallWebRTCConnection] = {}
...
                pipecat_connection = SmallWebRTCConnection(self._ice_servers)
```

So WhatsApp calling is: a webhook API model layer (`api.py`, twelve pydantic models for
`WhatsAppConnectCall`, `WhatsAppTerminateCall`, webhook envelopes), an HMAC webhook-signature check,
and then **the existing SmallWebRTC connection and transport**. Meta's calling product ships as a
*signalling adapter*, not as a transport.

### 2.2 Three structural ways to add a provider — and this is the decision table for Lina

Stop and generalise, because this is the framework-extension shape you will need in §11:

| Pattern | What you write | Witness in the tree | When it applies |
|---|---|---|---|
| **A. New transport class** | `BaseTransport` subclass + input/output processors + a connection client | `DailyTransport` (3,065 L file), `LiveKitTransport`, `MOQTransport` | the provider has its own media SDK / its own wire media format |
| **B. New serializer** | `FrameSerializer` subclass; reuse `FastAPIWebsocketTransport` verbatim | `TwilioFrameSerializer` (314 L), `TelnyxFrameSerializer` (292 L) | the provider streams media **over a WebSocket** in its own JSON/binary envelope |
| **C. New signalling client** | a webhook/REST adapter that builds a `SmallWebRTCConnection` | `WhatsAppClient` (`client.py`) | the provider does SDP offer/answer over its own control plane but standard WebRTC media |

A Korean carrier or CPaaS that streams call audio to your endpoint over a WebSocket is **pattern B**,
and pattern B is roughly 250–300 lines. A carrier that hands you a SIP/RTP leg directly is neither —
Pipecat has no SIP stack at this commit, and you would be terminating SIP outside the pipeline and
feeding it in through pattern B anyway. Note that carefully: it is not that Pipecat makes SIP hard,
it is that Pipecat has nothing to say about SIP itself. Every SIP reference in the transport tree is
a *provider's* SIP interconnect surfaced through that provider's SDK:

```bash
$ grep -ril "sip" src/pipecat/transports/
src/pipecat/transports/daily/transport.py
src/pipecat/transports/daily/utils.py
src/pipecat/transports/livekit/transport.py
```

Daily contributes `DailyRoomSipParams` (`daily/utils.py:20`), `DailyRoom.sip_uri` /
`sip_endpoint()` (`:117`, `:122`), `DailySIPTransferFrame` (`daily/transport.py:117`) and
`on_dialin_ready(sip_endpoint)`; LiveKit contributes an inbound `sip_dtmf_received` handler
(`livekit/transport.py:246`). Two files, one vendor each, zero SIP stack. If your carrier hands you
a raw SIP leg, terminating it is your problem and pattern B is how the audio gets in.

---

## 3. Sample rates resolve at setup, and the chunk size falls out of them

This is a small mechanism with an outsized blast radius, and it is where the 8 kHz problem in §6
first becomes visible in code.

Neither transport side takes a sample rate at construction time. Both read it in `setup()`, from
`FrameProcessorSetup` — the same object [[ch-04/read]] §6 traced through the startup path.

**`src/pipecat/transports/base_input.py` L119, L126**

```python
    async def setup(self, setup: FrameProcessorSetup):
        ...
        await super().setup(setup)
        self._sample_rate = self._params.audio_in_sample_rate or setup.audio_in_sample_rate
```

**`src/pipecat/transports/base_output.py` L123, L130**

```python
    async def setup(self, setup: FrameProcessorSetup):
        ...
        await super().setup(setup)
        self._sample_rate = self._params.audio_out_sample_rate or setup.audio_out_sample_rate
```

and the fallbacks:

**`src/pipecat/processors/frame_processor.py` L106–107**

```python
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
```

So an unconfigured pipeline runs **16 kHz in, 24 kHz out**. Both numbers are asymmetric on purpose:
16 kHz is what streaming STT and Silero VAD want, 24 kHz is what most neural TTS emits.

### 3.1 The chunk arithmetic, with the numbers filled in

**`src/pipecat/transports/base_output.py` L132–136**

```python
        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
```

Worked, at the three rates you will actually meet:

| Rate | `audio_bytes_10ms` | `audio_chunk_size` (×4) | Wall-clock per chunk |
|---|---|---|---|
| 24 000 Hz (default out) | 240 × 1 × 2 = **480 B** | **1 920 B** | 40 ms |
| 16 000 Hz (STT/VAD rate) | 160 × 1 × 2 = **320 B** | **1 280 B** | 40 ms |
| 8 000 Hz (PSTN wire rate) | 80 × 1 × 2 = **160 B** | **640 B** | 40 ms |

The wall-clock column is constant by construction — that is the point of expressing the parameter
in 10 ms units rather than bytes. The comment tells you why it exists: *"This will help with
interruption handling."* [[ch-04/read]] §5.1 gave you the general law; this is its concrete
instance. With `audio_out_10ms_chunks = 4`, the output transport can abandon its playout at 40 ms
granularity. Set it to 1 and you get 10 ms granularity at 4× the send-loop overhead. [[ch-08/read]]
spends this.

Note what the table does *not* say: nothing here converts 8 kHz to 16 kHz. The transport's sample
rate is the *pipeline's* sample rate. Getting from a carrier's 8 kHz to the pipeline's 16 kHz is
somebody else's job, and §6.4 shows you exactly whose.

---

## 4. VAD is not attached to the transport any more

Before anything else in §5–§8, clear out a stale mental model, because it is the single thing most
likely to make you misread every Pipecat voice tutorial older than a few releases.

**`TransportParams` has no `vad_analyzer` field.** Look at the L66–93 block in §1.1 again. It is not
there. This is not an omission in my quote; it was removed:

**`CHANGELOG.md` L4402–4406**

```
- ⚠️ Removed `vad_analyzer` and `turn_analyzer` parameters from
  `TransportParams` and all transport input classes, along with all deprecated
  VAD/turn analysis logic in `BaseInputTransport`. VAD and turn detection are
  now handled entirely by `LLMUserAggregator`.
  (PR [#4229](https://github.com/pipecat-ai/pipecat/pull/4229))
```

The live mount points, by grep:

```bash
$ grep -rn "vad_analyzer" src/pipecat/ | grep -v "import\|docstring\|:.*#"
src/pipecat/processors/aggregators/llm_response_universal.py:175:    vad_analyzer: VADAnalyzer | None = None
src/pipecat/processors/audio/vad_processor.py:44:        vad_analyzer: VADAnalyzer,
src/pipecat/audio/vad/vad_controller.py:72:        vad_analyzer: VADAnalyzer,
```

- `LLMUserAggregatorParams.vad_analyzer` — `llm_response_universal.py:175`
- `VADProcessor(vad_analyzer=...)` — `processors/audio/vad_processor.py:41`
- `VADController` — `audio/vad/vad_controller.py:31`, the shared hysteresis machine both use

The canonical wiring is now at the aggregator:

```python
context_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
)
```

There is exactly one piece of VAD code *inside* a transport, and it is a `VADAnalyzer`
implementation rather than a consumer of one — `WebRTCVADAnalyzer(VADAnalyzer)` at
`daily/transport.py:203`, wrapping Daily's native VAD. Nothing in the transport calls it; you pass
it to the aggregator like any other analyzer.

**Why this belongs in the transport chapter even though VAD is [[ch-06/read]]'s subject:** it
relocates a boson port. [[boson-gateway-server]] records that boson's endpointing lives in the
server — `_start_silence_timer` (`websocket.py:616`) sleeps `silence_timeout_ms / 1000` (default
2000) and then calls `_finalize_partial` (`:661`). On Pipecat that logic has no home at the
transport layer at all; it belongs under `src/pipecat/turns/`. Do not plan that port against
`TransportParams`. [[ch-06/read]] owns it.

---

## 5. The WebSocket transport: a socket, a fabricated clock, and a serializer slot

Three modules, none of which share a base beyond `BaseTransport`:

```bash
$ wc -l src/pipecat/transports/websocket/*.py
       0 __init__.py
     559 client.py
     707 fastapi.py
     716 server.py
    1982 total
```

`__init__.py` is empty — import from the concrete module. `client.py` is Pipecat *dialing out* to
somebody else's WebSocket. `server.py` is a dev server. `fastapi.py` is the production path and the
only one that matters for Lina.

[[transport-websocket]] states the design in one sentence and the tree bears it out: the transport
owns *the socket, the audio clock, and connect/disconnect events* — and delegates **every byte on
the wire** to a pluggable `FrameSerializer`. Three responsibilities, one delegation. §5.3 and §5.4
are those three; §6 is the delegation.

### 5.1 `FastAPIWebsocketTransport` receives an already-accepted socket

**`src/pipecat/transports/websocket/fastapi.py` L630–673**

```python
    def __init__(
        self,
        websocket: WebSocket,
        params: FastAPIWebsocketParams,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the FastAPI WebSocket transport.

        Raises ``ValueError`` if ``params.allowed_origins`` is set and the
        connection's Origin header is missing or not in the allowed list. The
        caller is responsible for closing the WebSocket in that case.
        ...
        """
        if params.allowed_origins:
            origin = websocket.headers.get("origin", "")
            if not is_origin_allowed(origin, params.allowed_origins):
                raise ValueError(f"WebSocket connection rejected: origin '{origin}' not allowed")

        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params

        self._callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )

        self._client = FastAPIWebsocketClient(
            websocket, self._callbacks, ws_close_timeout=self._params.ws_close_timeout
        )

        self._input = FastAPIWebsocketInputTransport(
            self, self._client, self._params, name=self._input_name
        )
        self._output = FastAPIWebsocketOutputTransport(
            self, self._client, self._params, name=self._output_name
        )
```

Everything you need to know about the division of labour is in that signature and that docstring.
The transport takes a `WebSocket` that **you already accepted**. You own the route. You own the
handshake. You own any authentication. You own closing the socket when the constructor raises.
Pipecat's contribution starts one line after `await websocket.accept()`.

The one security control it does ship is an Origin allowlist, and it is 12 lines:

**`src/pipecat/utils/security/allowed_origins.py` L23–34**

```python
def is_origin_allowed(origin: str, allowed_origins: list[str]) -> bool:
    """Return whether ``origin`` is permitted by ``allowed_origins``.

    Args:
        origin: The value of the ``Origin`` header, or an empty string if absent.
        allowed_origins: List of allowed origin strings. An empty list allows
            all origins. When non-empty, a missing or disallowed origin is
            rejected.
    """
    if not allowed_origins:
        return True
    return origin.lower() in {o.lower() for o in allowed_origins}
```

Compare the *policy* (not the quality — §0, and the scope box) with what [[boson-gateway-server]]
records for `access.py`:

```python
# packages/gateway/gateway/server/access.py — shape as recorded in [[boson-gateway-server]]
# (boson-agent, private; excerpt-attested)
accepted_origins = (None, *allowed_origins)
```

boson's tuple deliberately admits a **missing** Origin, reserving that case for native clients,
and requires browsers to match the allowlist. Pipecat's `is_origin_allowed` rejects a missing
Origin whenever the allowlist is non-empty. Two different decisions about the same header, and if
you put boson's native clients behind Pipecat's check without noticing, they stop connecting. That
is the level of detail at which this migration bites.

### 5.2 Exactly six added fields

**`src/pipecat/transports/websocket/fastapi.py` L83–88**

```python
    add_wav_header: bool = False
    serializer: FrameSerializer | None = None
    session_timeout: int | None = None
    fixed_audio_packet_size: int | None = None
    allowed_origins: list[str] = Field(default_factory=default_allowed_origins)
    ws_close_timeout: float = _WS_CLOSE_TIMEOUT_DEFAULT
```

with `_WS_CLOSE_TIMEOUT_DEFAULT = 0.5` at L56. Six fields on top of the ~27 in `TransportParams`.
That is the entire configuration surface of the production WebSocket transport.

`session_timeout` deserves one sentence of demystification, because the name promises more than the
code delivers:

**`src/pipecat/transports/websocket/fastapi.py` L398–401**

```python
    async def _monitor_websocket(self, timeout: int):
        """Wait for ``timeout`` seconds, then trigger the client-timeout event if still open."""
        await asyncio.sleep(timeout)
        await self._client.trigger_client_timeout()
```

A bare sleep that fires one callback once. It is not an idle timer — it does not reset on activity,
and it does not close anything. If you want boson's `DEFAULT_SESSION_TTL_SECONDS = 1800.0`
idle-TTL behaviour ([[boson-gateway-server]]), this is not it.

### 5.3 With no serializer set, every inbound message is silently dropped

**`src/pipecat/transports/websocket/fastapi.py` L372–389**

```python
    async def _receive_messages(self):
        """Main message receiving loop for WebSocket messages."""
        try:
            async for message in self._client.receive():
                if not self._params.serializer:
                    continue

                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                elif isinstance(frame, InputTransportMessageFrame):
                    await self.broadcast_frame(InputTransportMessageFrame, message=frame.message)
                else:
                    await self.push_frame(frame)
```

and the outbound mirror:

**`src/pipecat/transports/websocket/fastapi.py` L559–570**

```python
    async def _write_frame(self, frame: Frame) -> bool:
        """Serialize and send a frame through the WebSocket.

        Returns:
            Whether the frame was sent.
        """
        if self._client.is_closing or not self._client.is_connected:
            return False

        if not self._params.serializer:
            return False
```

`continue`. `return False`. No log, no raise, no metric. A `FastAPIWebsocketTransport` with
`serializer=None` is a fully functional object that connects, fires `on_client_connected`, runs its
receive loop forever, and moves zero bytes in either direction. This is the same silent-drop
posture [[ch-01/read]] §7.3 traced through `process_frame`, and [[ch-03/read]] §2.3 traced through
the unhandled-frame path — it is a house style, and it is now at the wire boundary too.

Three routing branches in the inbound loop, worth naming because they map onto three different
mechanisms you already know:

| Deserialized frame is | Goes to | Why |
|---|---|---|
| `InputAudioRawFrame` | `push_audio_frame(frame)` | into the audio-in path (filters, passthrough) |
| `InputTransportMessageFrame` | `broadcast_frame(...)` | **both directions at once** |
| anything else | `push_frame(frame)` | ordinary downstream push |

`broadcast_frame` is the one to look at, because a transport message is not a media event and needs
to be visible to processors on both sides of the transport:

**`src/pipecat/processors/frame_processor.py` L1038–1054**

```python
    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Broadcasts a frame of the specified class upstream and downstream.
        ...
        """
        downstream_frame = frame_cls(**kwargs)
        upstream_frame = frame_cls(**kwargs)
        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)
```

Two instances, cross-referenced by `broadcast_sibling_id`, one each way. This is the mechanism
that a ported `ControlEvent` would ride on — hold it for §11.2.

### 5.4 A WebSocket has no playback clock, so the transport fabricates one

This is the non-obvious part of the whole module, and it is documented in a comment rather than a
design doc.

**`src/pipecat/transports/websocket/fastapi.py` L432–438**

```python
        # write_audio_frame() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed during setup.
        self._send_interval = 0
        self._next_send_time = 0
```

(The typo "to quickly" is in the source, verbatim, and it appears in four files —
`websocket/fastapi.py:434`, `websocket/server.py:349`, `websocket/client.py:348`, and
`services/heygen/client.py:174`. Copy-paste, which tells you the problem recurs anywhere Pipecat
writes media to a socket rather than to a device.)

Think about what is being solved. A sound card or an RTP track applies back-pressure by *time*: it
consumes 20 ms of audio every 20 ms and you cannot outrun it. A TCP socket applies back-pressure by
*buffer*, which for a few hundred KB of TTS is effectively no back-pressure at all. Without a
clock, the moment TTS produces a full sentence the transport would shove all of it onto the wire in
one burst, and then the carrier — not you — owns that audio. Barge-in becomes impossible: you can
stop producing, but the customer will keep hearing the sentence you already sent.

So the transport blocks:

**`src/pipecat/transports/websocket/fastapi.py` L456**

```python
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
```

**`src/pipecat/transports/websocket/fastapi.py` L599–608**

```python
    async def _write_audio_sleep(self):
        """Simulate audio playback timing with appropriate delays."""
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval
```

**Work the arithmetic before reading the formula as a formula.** With the defaults
(`audio_out_sample_rate = 24000`, `audio_out_10ms_chunks = 4`):

- `audio_chunk_size` = 1 920 bytes (§3.1)
- `audio_chunk_size / sample_rate` = 1920 / 24000 = **0.08**

That 0.08 is *not* seconds. `audio_chunk_size` is bytes and `sample_rate` is samples/second, so the
division is only dimensionally correct because 16-bit mono means 2 bytes per sample — the true
duration is 1920 bytes ÷ 2 = 960 samples ÷ 24000 = **40 ms**. The expression yields 80 ms, i.e.
exactly twice the real duration, and the `/ 2` cancels the factor of 2 that the missing
bytes-per-sample term introduced.

So `_send_interval` = 40 ms for a 40 ms chunk, which is what you want, and the `/ 2` is *not* a
deliberate 2× under-sleep for jitter headroom. It is the bytes-to-samples conversion written as a
constant. Two consequences you can act on:

1. **It is only correct for 16-bit mono.** Set `audio_out_channels = 2` and `audio_chunk_size`
   doubles (§3.1's formula multiplies by channels) while the divisor does not, so `_send_interval`
   doubles to 80 ms for a chunk that is still 40 ms of wall-clock stereo audio — the transport
   would sleep twice as long as the audio it just sent. Lina is mono; keep it mono on this path.
2. The line is *identical* in `server.py:379`, so both server-side WebSocket transports share the
   behaviour.

And the branch structure of `_write_audio_sleep` is a catch-up rule: if we are behind
(`sleep_duration == 0`), rebase the schedule off *now*; if we are on time, advance the schedule by
one interval without drift. Which is the same idea as realtime_voice's wall-clock pacing:

```python
# packages/realtime_voice/realtime_voice/transport/webrtc/tracks.py L168
# (boson-agent, private; excerpt-attested via [[rtv-webrtc-transport]])
#   _pace():  target = self._started_at + self._pts / self.sample_rate
```

boson computes the target from an absolute origin plus a sample counter; Pipecat accumulates
intervals and rebases on lateness. Both are wall-clock pacing; boson's cannot drift by construction,
Pipecat's rebases whenever it falls behind. State that as the difference and stop there — the
latency consequences are [[ch-11/read]]'s.

Finally, interruption resets the clock:

**`src/pipecat/transports/websocket/fastapi.py` L508–514**

```python
        if isinstance(frame, InterruptionFrame):
            # Drop any partially buffered audio to avoid replaying stale PCM
            if self._params.fixed_audio_packet_size:
                self._audio_send_buffer.clear()

            await self._write_frame(frame)
            self._next_send_time = 0
```

Three things in six lines, and all three matter for [[ch-08/read]]: drop the local packetization
remainder, **push the `InterruptionFrame` through the serializer** (which is how the carrier gets
told — §6.8), and zero the clock so the next audio chunk goes out immediately rather than waiting
out a stale schedule.

### 5.5 `fixed_audio_packet_size`, and why it exists

**`src/pipecat/transports/websocket/fastapi.py` L575–588**

```python
                # Optional protocol-level audio packetization:
                # If a downstream WebSocket media endpoint requires fixed-size PCM frames,
                # configure params.fixed_audio_packet_size (e.g. 640 for 20ms @ 16kHz PCM16 mono).
                packet_bytes = self._params.fixed_audio_packet_size

                if packet_bytes and isinstance(payload, (bytes, bytearray)):
                    self._audio_send_buffer.extend(bytes(payload))

                    # Send only full frames; keep remainder for the next call.
                    while len(self._audio_send_buffer) >= packet_bytes:
                        chunk = bytes(self._audio_send_buffer[:packet_bytes])
                        del self._audio_send_buffer[:packet_bytes]
                        await self._client.send(chunk)
                    return True
```

A carrier that demands exact-size media frames is common, and this is the hook for it: buffer,
emit whole packets, preserve the remainder. The in-code example (640 B = 20 ms @ 16 kHz PCM16 mono)
is exactly the kind of constraint a Korean CPaaS is likely to impose. Note that this path
`return True`s *before* the `success = True` assignment further down, so a fixed-packet send always
reports success even when nothing was flushed — a distinction that matters only if you build metrics
on the return value.

### 5.6 `ws_close_timeout` exists because of telephony

**`src/pipecat/transports/websocket/fastapi.py` L186–212**

```python
    async def disconnect(self):
        """Disconnect the WebSocket client.

        The close handshake is bounded by ``ws_close_timeout``. The close is
        initiated in a background task before we start waiting, so the close
        frame is sent to the peer in the common case; we then wait at most
        ``ws_close_timeout`` seconds for the peer to acknowledge it. If the peer
        never replies (e.g. a half-closed connection after the remote side
        already hung up), we stop waiting and let shutdown proceed instead of
        blocking on the ASGI server's close-handshake timeout. ...
        """
        self._leave_counter -= 1
        if self._leave_counter > 0:
            return

        if self.is_connected and not self.is_closing:
            self._closing = True
            self._close_task = asyncio.create_task(self._websocket.close(), name="fastapi-ws-close")
            self._close_task.add_done_callback(self._on_close_done)
            done, _ = await asyncio.wait({self._close_task}, timeout=self._ws_close_timeout)
            if not done:
                logger.debug(
                    f"{self} WebSocket close exceeded {self._ws_close_timeout}s; "
                    "proceeding with shutdown"
                )
```

The `FastAPIWebsocketParams` docstring names the motivating case explicitly: *"Prevents a dead or
half-closed peer (e.g. a telephony call already torn down on the provider's side) from stalling
pipeline shutdown."* This is [[ch-04/read]] §8's unbounded-`EndFrame` problem showing up at the
socket: `EndFrame` has no timeout, so if the transport's `disconnect()` blocks forever on a close
handshake with a carrier that already hung up, graceful shutdown never completes. 0.5 s is the
fence. Also note `_leave_counter` — input and output share one `FastAPIWebsocketClient`, so the
close is refcounted and the output side can flush a goodbye line after the input side is done.

### 5.7 `SingleClientWebsocketServerTransport` is dev-only, and is also your eval harness

**`src/pipecat/transports/websocket/server.py` L518–529**

```python
class SingleClientWebsocketServerTransport(BaseTransport):
    """WebSocket server transport that serves a single client at a time.
    ...
    Only one client can be connected at a time. While a client is connected, new
    connection attempts are rejected and the existing client is kept; once that
    client disconnects, the server accepts a new one. This makes it well suited
    for local development and single-session bots, but not for serving multiple
    concurrent clients.
```

Enforced, not documented:

**`src/pipecat/transports/websocket/server.py` L247–256**

```python
        # This transport only serves a single client at a time. If we already
        # have a live connection, reject the new one and keep the existing
        # client. The current connection's reference is cleared when it
        # disconnects (or something goes wrong), so the next client can connect.
        if self._websocket and self._websocket.state is State.OPEN:
            logger.warning(
                f"Rejecting client {websocket.remote_address}: a client is already connected"
            )
            await websocket.close(code=1013, reason="Server already has a connected client")
            return
```

Close code 1013 is "Try Again Later". The class was renamed in 1.4.0; the old names
(`WebsocketServerTransport`, `WebsocketServerParams`, `WebsocketServerCallbacks`,
`WebsocketServerInputTransport`, `WebsocketServerOutputTransport`) all carry `@deprecated` at
`server.py:643/658/673/688/703` with removal in 2.0.0. If you find a tutorial using
`WebsocketServerTransport`, it predates the rename.

The interesting thing is what the framework itself uses this dev-only transport *for*:

**`src/pipecat/evals/serializer.py` L87–93**

```python
class RTVIEvalSerializer(FrameSerializer):
    """Bridges JSON RTVI messages and pipeline frames for the eval harness.

    Use as the serializer of a ``SingleClientWebsocketServerTransport`` when running a bot
    under the eval harness. The bot pipeline must include an ``RTVIProcessor``
    and pass an ``RTVIObserver`` to the task.
    """
```

Pipecat's own evaluation harness is *a serializer plus the single-client dev transport*. That is a
directly reusable pattern for Lina's QA harness and §11.3 turns it into a concrete move.

---

## 6. Telephony: there is no telephony transport

### 6.1 The listing that settles it

```bash
$ ls src/pipecat/transports/
__init__.py  base_input.py  base_output.py  base_transport.py
daily/  heygen/  lemonslice/  livekit/  local/  moq/  smallwebrtc/
tavus/  vonage/  websocket/  whatsapp/

$ ls src/pipecat/serializers/
__init__.py  base_serializer.py  exotel.py  genesys.py  plivo.py
protobuf.py  telnyx.py  twilio.py  vonage.py
```

No `twilio/`. No `telnyx/`. No `plivo/`. No `exotel/`. No `sip/`. The entire telephony surface of
this framework is six files in a different directory:

```bash
$ wc -l src/pipecat/serializers/*.py
       0 __init__.py
     106 base_serializer.py
     171 exotel.py
     964 genesys.py
     256 plivo.py
     165 protobuf.py
     292 telnyx.py
     314 twilio.py
     188 vonage.py
    2456 total
```

Six telephony serializers totalling 2,185 lines. And note the trap: `transports/vonage/` **does**
exist, but it is `VonageVideoConnectorTransport` — video — and has nothing to do with
`serializers/vonage.py`. Same vendor, two unrelated products, one namespace collision.

### 6.2 The framework's own runner is the proof

You do not have to take the directory listing's word for it. Pipecat's development runner
constructs a phone bot, and here is what it constructs:

**`src/pipecat/runner/utils.py` L486–554** (elided in the middle; the four branches are structurally
identical)

```python
async def _create_telephony_transport(
    websocket: "WebSocket",
    params: Any,
    transport_type: str,
    call_data: CallData,
) -> BaseTransport:
    """Create a telephony transport with pre-parsed WebSocket data.
    ...
    Returns:
        Configured FastAPIWebsocketTransport ready for telephony use.
    """
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport

    # Always set add_wav_header to False for telephony
    params.add_wav_header = False

    logger.info(f"Using pre-detected telephony provider: {transport_type}")

    if transport_type == "twilio":
        from pipecat.serializers.twilio import TwilioFrameSerializer

        params.serializer = TwilioFrameSerializer(
            stream_sid=call_data["stream_id"],
            call_sid=call_data["call_id"],
            account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        )
    elif transport_type == "telnyx":
        ...
    elif transport_type == "plivo":
        ...
    elif transport_type == "exotel":
        ...
    else:
        raise ValueError(
            f"Unsupported telephony provider: {transport_type}. "
            f"Supported providers: twilio, telnyx, plivo, exotel"
        )

    return FastAPIWebsocketTransport(websocket=websocket, params=params)
```

The return type annotation says `BaseTransport`. The return statement says
`FastAPIWebsocketTransport`. **One transport, four serializers.** The provider difference reduces
entirely to which object gets assigned to `params.serializer`.

`create_transport()`'s docstring (`runner/utils.py:598`) makes the same point from the user's side —
the keys `"twilio"`, `"telnyx"`, `"plivo"`, `"exotel"` all map to `FastAPIWebsocketParams`
factories, with the comment `# add_wav_header and serializer will be set automatically`.

Provider detection happens by sniffing the first `start` message:

**`src/pipecat/runner/utils.py` L113 (signature), L135–143 (docstring)**

```python
async def parse_telephony_websocket(websocket: "WebSocket"):
    """Parse telephony WebSocket messages and return transport type and call data.
    ...
        - Telnyx::

            {
                "stream_id": str,
                "call_id": str,  # normalized from Telnyx's call_control_id
                "outbound_encoding": str,
                "from": str,
                "to": str,
            }
    """
```

Note `# normalized from Telnyx's call_control_id`. Every provider names its two identifiers
differently and `parse_telephony_websocket` flattens them to `stream_id` / `call_id`. §6.7 is about
why there are two.

### 6.3 The `FrameSerializer` ABC is four methods and 106 lines

**`src/pipecat/serializers/base_serializer.py` L23–44**

```python
class FrameSerializer(BaseObject):
    """Abstract base class for frame serialization implementations.

    Defines the interface for converting frames to/from serialized formats
    for transmission or storage. Subclasses must implement the core
    serialize/deserialize methods.
    """

    class InputParams(BaseModel):
        """Base configuration parameters for FrameSerializer.

        Parameters:
            ignore_rtvi_messages: Whether to ignore RTVI protocol messages during serialization.
                Defaults to True to prevent RTVI messages from being sent to external transports.
            resampler_clear_after_secs: Seconds of inactivity after which the stream resampler
                clears its internal history to avoid audio artefacts from stale state. Set to
                ``None`` to never clear — recommended for telephony providers (e.g. Genesys) that
                have irregular gaps between audio chunks. Defaults to ``0.2``.
        """

        ignore_rtvi_messages: bool = True
        resampler_clear_after_secs: float | None = 0.2
```

**`src/pipecat/serializers/base_serializer.py` L76–106**

```python
    async def setup(self, setup: FrameProcessorSetup):
        """Initialize the serializer with pipeline configuration.
        ...
        """
        pass

    @abstractmethod
    async def serialize(self, frame: Frame) -> str | bytes | None:
        ...

    @abstractmethod
    async def deserialize(self, data: str | bytes) -> Frame | None:
        ...
```

Plus `should_ignore_frame()` at L56 and `__init__` at L46. That is the whole contract: `setup`, `serialize`, `deserialize`,
`should_ignore_frame`. Two of the four are abstract.

`setup()` is how the serializer learns the pipeline's rate, and it is called from **both** ends —
`FastAPIWebsocketInputTransport.setup()` at `fastapi.py:306` and
`FastAPIWebsocketOutputTransport.setup()` at `fastapi.py:461`:

```python
        if self._params.serializer:
            await self._params.serializer.setup(setup)
```

Both sides call `setup()` on the *same* serializer instance, which is idempotent for the telephony
serializers (they only assign `self._sample_rate`) but is a real constraint if you write a stateful
one. `resampler_clear_after_secs: float | None = 0.2` is the small detail with a big field failure
behind it: a stream resampler carries filter history, and history that goes stale across a gap
produces audible artefacts. Genesys is named in the docstring as the provider whose chunk timing is
irregular enough that you should disable the clearing entirely. Write that down for any Korean
carrier: **if their media chunks arrive irregularly, `resampler_clear_after_secs=None`.**

### 6.4 One frame crossing the boundary, traced in both directions

This is the paragraph to reread. Everything about telephony that will hurt Lina is visible in these
two methods.

**Inbound — `src/pipecat/serializers/twilio.py` L276–314**

```python
    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Twilio WebSocket data to Pipecat frames.
        ...
        """
        message = json.loads(data)

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            # Input: Convert Twilio's 8kHz μ-law to PCM at pipeline input rate
            deserialized_data = await ulaw_to_pcm(
                payload, self._twilio_sample_rate, self._sample_rate, self._input_resampler
            )
            if deserialized_data is None or len(deserialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )
            return audio_frame
        elif message["event"] == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                return None
        else:
            return None
```

and the decode itself:

**`src/pipecat/audio/utils.py` L208–228**

```python
async def ulaw_to_pcm(
    ulaw_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler
):
    """Convert μ-law encoded audio to PCM and optionally resample.
    ...
    """
    # Convert μ-law to PCM
    in_pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)

    # Resample
    out_pcm_bytes = await resampler.resample(in_pcm_bytes, in_rate, out_rate)

    return out_pcm_bytes
```

Order matters: **decode companding first, then resample.** `audioop.ulaw2lin(x, 2)` expands each
8-bit μ-law byte into a 16-bit linear sample at the *same* rate (8 kHz), and only then does the
resampler take 8 kHz → 16 kHz. You now have 16-bit samples at 16 kHz that your STT will accept.

**Outbound — `src/pipecat/serializers/twilio.py` L166–214**

```python
    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Twilio WebSocket format.
        ...
        """
        if (
            self._params.auto_hang_up
            and not self._hangup_attempted
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._hangup_attempted = True
            await self._hang_up_call()
            return None
        elif isinstance(frame, InterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz μ-law for Twilio
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._twilio_sample_rate, self._output_resampler
            )
            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        # Return None for unhandled frames
        return None
```

Four responsibilities in one 48-line method, and this is the sentence to keep:

> The serializer is simultaneously **the codec boundary, the resample point, the call-control
> client, and the DTMF decoder.**

Codec: `pcm_to_ulaw` / `ulaw_to_pcm`. Resample: the two `create_stream_resampler` instances built in
`__init__` (L150–155). Call control: `_hang_up_call()` makes an HTTP request to Twilio's REST API
from inside `serialize()`. DTMF: the `"dtmf"` branch of `deserialize`. That is a lot of unrelated
concern in one class, and it is unrelated *because* the serializer is the only provider-specific
place the architecture provides. Pattern B in §2.2 has exactly one file in it.

### 6.5 The provider table, verified line by line

| Serializer | Wire rate (verified line) | Codec | Interruption event |
|---|---|---|---|
| `TwilioFrameSerializer` (314 L) | `twilio_sample_rate: int = 8000` — twilio.py:79 | μ-law | `{"event": "clear", "streamSid": ...}` — :187 |
| `TelnyxFrameSerializer` (292 L) | `telnyx_sample_rate: int = 8000` — telnyx.py:60 | `inbound_encoding`/`outbound_encoding`, `"PCMU"` default (:62–63), `"PCMA"` supported, else `raise ValueError(f"Unsupported encoding: ...")` (:166, :273) | `{"event": "clear"}` — :151 (no id) |
| `PlivoFrameSerializer` (256 L) | `plivo_sample_rate: int = 8000` — plivo.py:54 | μ-law | `{"event": "clearAudio", "streamId": ...}` — :139 |
| `ExotelFrameSerializer` (171 L) | `exotel_sample_rate: int = 8000` — exotel.py:49 | **raw PCM, resample only** — no companding | `{"event": "clear", "stream_sid": ...}` — :99 |
| `GenesysAudioHookSerializer` (964 L) | `genesys_sample_rate: int = 8000` — genesys.py:148 | `AudioHookMediaFormat.PCMU = "PCMU"  # μ-law, 8kHz` — :77, or L16 | (see §6.10) |
| `VonageFrameSerializer` (188 L) | `vonage_sample_rate: int = 16000` — vonage.py:48 | — | — |

**Five of six default to 8 kHz. Vonage is the only exception.** Exotel is the second exception on a
different axis: it is the only one that sends linear PCM rather than a companded codec, so its
serialize path is a bare `self._output_resampler.resample(...)` (exotel.py:104–106) with no
`pcm_to_ulaw`. It is also the only one with **no `auto_hang_up`** — its `__init__` accepts
`call_sid` and the docstring says *"optional, not used in this implementation"* (exotel.py:60).

### 6.6 The 4 kHz ceiling, and what it does to Korean

Work the example before the formula.

**A concrete run.** A customer says 있습니다 into a phone. The carrier's codec samples her voice
8,000 times per second and encodes each sample as one μ-law byte. Twilio base64s ~160 of those
bytes (20 ms) into a JSON message and sends it to your socket. `ulaw_to_pcm` expands each byte to a
16-bit sample and resamples 8 kHz → 16 kHz, producing 320 samples where there were 160. Your STT
receives a 16 kHz, 16-bit `InputAudioRawFrame` and is perfectly happy with it.

**Now: what is in those 320 samples?** Sampling at 8,000 Hz can represent frequency content up to
8000 / 2 = **4,000 Hz**, and content above that was removed by an anti-alias filter in the carrier's
equipment *before you ever saw a byte*. Interpolating to 16 kHz creates 320 numbers from 160
numbers. It does not create information. There is nothing in the 4–8 kHz band of that signal, because
there was nothing to carry it.

That is the whole formula: **Nyquist ceiling = wire rate / 2 = 4 kHz.** And on top of it, μ-law is
8-bit companded — logarithmic quantisation trading roughly 2 bits of effective dynamic range against
PCM16 for a perceptually reasonable curve on speech. So the signal is band-limited *and* coarsely
quantised.

**Why this is the governing constraint for Lina specifically**, rather than a generic audio caveat:

- Korean fricatives and affricates — ㅅ, ㅆ, ㅊ, and the ㅎ in many contexts — carry most of their
  discriminating energy in the high band. The energy that separates 사 from 싸, or 자 from 차, sits
  at and above 4 kHz.
- 받침 (final consonant) discrimination leans heavily on the release burst and on formant
  transitions whose distinguishing cues are also high-band. 갔습니다 / 같습니다 / 잤습니다 are
  spectrally much closer at 4 kHz than at 8 kHz.
- Insurance tele-sales runs on exactly the tokens this hurts: 보험료, 갱신, 특약, 해지, 청약 — and
  on numbers and dates, where 사 / 삼 and 십 / 시 confusions are catastrophic rather than cosmetic.

**The operational rule, stated so you can enforce it:** any Korean STT accuracy number you have
measured on 16 kHz microphone audio is **not transferable to this pipeline**. Not "somewhat
optimistic" — not transferable. Measure on real 8 kHz μ-law that has been through a carrier, or
record the number as unknown. If you must estimate before you have carrier access, the honest
proxy is to take your 16 kHz eval set, low-pass at 3.4–4 kHz, μ-law encode and decode it, upsample
to 16 kHz, and re-run. That is a simulation, not a measurement, and it will still be optimistic
because it omits packet loss, jitter and the carrier's own gain control. [[ch-07/read]] and
[[stt-korean-providers]] pick this up on the provider-selection side.

### 6.7 Use the figure here

**[figures/transport-comparison.html](figures/transport-comparison.html)** — open it now and go
straight to the telephony panel. It runs one frame across the serializer box (8 kHz μ-law in →
resample to 16 kHz → an `InterruptionFrame` out as a provider `clear` event) against a Nyquist band
chart with the Korean fricative and 받침 cue bands marked over the 4 kHz ceiling. Do one thing with
it: use the band chart to decide, before you write any code, whether Lina's number-and-date
confirmation flow needs a DTMF fallback or a spell-out-and-confirm turn design — the answer comes
from what is *missing* in the shaded region, not from anything in the pipeline. The other three
stacks (Daily/SmallWebRTC, raw WebSocket, your own aiortc transport) and the counting strip are
there to keep §2 and §10 honest while you read.

### 6.8 Two identifiers, and `auto_hang_up`

**`src/pipecat/serializers/twilio.py` L83–93**

```python
    def __init__(
        self,
        stream_sid: str,
        call_sid: str | None = None,
        account_sid: str | None = None,
        auth_token: str | None = None,
        region: str | None = None,
        edge: str | None = None,
        base_url: str | None = None,
        params: InputParams | None = None,
    ):
```

`stream_sid` is required; `call_sid` is optional. They are not two names for the same thing:

- **`stream_sid`** is the media stream. It is echoed on **every** outbound `media` message and on
  every `clear` message (see the `serialize` quote in §6.4 — both dicts carry it). Without it the
  carrier cannot route your audio to the right leg.
- **`call_sid`** is the call resource, and it is used for exactly one thing: REST call control.

Per provider: Telnyx uses `stream_id` + `call_control_id`; Plivo `stream_id` + `call_id`; Exotel
`stream_sid` + `call_sid` (unused). `parse_telephony_websocket` normalises all of them.

`auto_hang_up: bool = True` — **on by default** for Twilio, Telnyx and Plivo. Which makes the
constructor validation load-bearing:

**`src/pipecat/serializers/twilio.py` L113–127**

```python
        # Validate hangup-related parameters if auto_hang_up is enabled
        if self._params.auto_hang_up:
            # Validate required credentials
            missing_credentials = []
            if not call_sid:
                missing_credentials.append("call_sid")
            if not account_sid:
                missing_credentials.append("account_sid")
            if not auth_token:
                missing_credentials.append("auth_token")

            if missing_credentials:
                raise ValueError(
                    f"auto_hang_up is enabled but missing required parameters: {', '.join(missing_credentials)}"
                )
```

and the hang-up itself, which is an HTTP call issued from inside `serialize()`:

**`src/pipecat/serializers/twilio.py` L239–257**

```python
            # Parameters to set the call status to "completed" (hang up)
            params = {"Status": "completed"}

            # Make the POST request to update the call
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, auth=auth, data=params) as response:
                    if response.status == 200:
                        logger.info(f"Successfully terminated Twilio call {call_sid}")
                    elif response.status == 404:
                        # Handle the case where the call has already ended
                        # Error code 20404: "The requested resource was not found"
                        # Source: https://www.twilio.com/docs/errors/20404
                        try:
                            error_data = await response.json()
                            if error_data.get("code") == 20404:
                                logger.debug(f"Twilio call {call_sid} was already terminated")
                                return
                        except Exception:
                            pass  # Fall through to log the raw error
```

`POST /2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json` with `{"Status": "completed"}`,
treating 404/20404 as already-ended. Telnyx does the same shape against
`POST https://api.telnyx.com/v2/calls/{call_control_id}/actions/hangup`, treating 422/90018 as
already-ended.

**The collision to write down now.** `auto_hang_up` fires on `EndFrame` *or* `CancelFrame`.
[[ch-04/read]] §7 and §8 established that both of those are how a Pipecat pipeline ends — one
graceful, one immediate. [[boson-gateway-server]] records that boson's teardown path is
`_teardown_connection_sessions` (`websocket.py:381`) → `_on_disconnect(sid)` → `core.on_disconnect`
(`core.py:605`), which saves history and starts an idle-TTL clock. If you port boson's teardown as
written and leave `auto_hang_up=True`, **two independent subsystems will both try to end the call**,
and the one that loses will log an error against a call resource that no longer exists. Pick one
owner. The mechanical fact you need for that decision is that the serializer's version fires
synchronously inside frame serialization, i.e. earlier than anything your application-level teardown
can react to.

Also note `_hangup_attempted` (L156, set at L183): a one-shot latch, so `EndFrame` followed by
`CancelFrame` hangs up once.

### 6.9 Barge-in is a provider-specific flush, and it rides `InterruptionFrame`

Four providers, four different envelopes for the same idea:

| Provider | Flush message | Line |
|---|---|---|
| Twilio | `{"event": "clear", "streamSid": self._stream_sid}` | twilio.py:187 |
| Plivo | `{"event": "clearAudio", "streamId": self._stream_id}` | plivo.py:139 |
| Telnyx | `{"event": "clear"}` | telnyx.py:151 |
| Exotel | `{"event": "clear", "stream_sid": self._stream_sid}` | exotel.py:99 |

Why this exists at all is [[ch-04/read]] §5.1's arithmetic applied one hop further out than you
have been applying it. Your pipeline can flush its own output queue instantly. But the audio you
already handed the carrier is sitting in **the carrier's** playout buffer, and the customer is
still hearing it. There are three buffers between the LLM and the ear — the processor queue, the
transport's own packetization buffer, and the carrier's playout buffer — and an interruption has to
clear all three. §5.4's `process_frame` clears the second and forwards the frame to the serializer;
the serializer's `clear` message clears the third.

And here is the boson collision, from [[boson-gateway-server]] and [[boson-interrupt-subsystem]]:
boson's `gateway/interrupt/` currently decides to barge in by reasoning over **partial text
transcripts**. That decision logic is orthogonal to this — you can keep it — but the *effect* path
changes completely: whatever decides to interrupt must end up emitting an `InterruptionFrame` that
reaches the transport output, because that is the only thing that produces the carrier `clear`
event. An interruption that only stops the LLM will leave the customer listening to the rest of the
sentence. [[ch-08/read]] takes the whole cascade apart; this section is here so you know the last
link in it lives in a serializer.

### 6.10 DTMF is asymmetric, and the outbound half is a workaround

**Inbound works everywhere.** The `"dtmf"` branch quoted in §6.4 returns
`InputDTMFFrame(KeypadEntry(digit))` and swallows `ValueError` for unrecognised digits. The keypad
vocabulary is closed at twelve:

**`src/pipecat/audio/dtmf/types.py` L34–46**

```python
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    ZERO = "0"

    POUND = "#"
    STAR = "*"
```

0–9, `#`, `*`. No A/B/C/D (the fourth DTMF column, which almost nothing outside military and
legacy trunk signalling uses).

**Outbound is the gap.** No telephony serializer handles `OutputDTMFFrame` at all:

```bash
$ grep -rn "OutputDTMFFrame" src/pipecat/serializers/
$ echo $?
1
```

Zero hits. So the transport falls back:

**`src/pipecat/transports/base_output.py` L264–273, L287–293, L303–316**

```python
    async def write_dtmf(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Write a DTMF tone using the transport's preferred method.
        ...
        """
        if self._supports_native_dtmf():
            await self._write_dtmf_native(frame)
        else:
            await self._write_dtmf_audio(frame)
    ...
    def _supports_native_dtmf(self) -> bool:
        """Override in transport implementations that support native DTMF.

        Returns:
            True if the transport supports native DTMF, False otherwise.
        """
        return False
    ...
    async def _write_dtmf_audio(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Generate and send audio tones for DTMF.
        ...
        """
        if not frame.buttons:
            return
        for button in frame.buttons:
            dtmf_audio = await load_dtmf_audio(button, sample_rate=self._sample_rate)
            dtmf_audio_frame = OutputAudioRawFrame(
                audio=dtmf_audio, sample_rate=self._sample_rate, num_channels=1
            )
            await self.write_audio_frame(dtmf_audio_frame)
```

`load_dtmf_audio` reads one of twelve shipped WAV files:

```bash
$ ls src/pipecat/audio/dtmf/
__init__.py  dtmf-0.wav  dtmf-1.wav  dtmf-2.wav  dtmf-3.wav  dtmf-4.wav
dtmf-5.wav  dtmf-6.wav  dtmf-7.wav  dtmf-8.wav  dtmf-9.wav  dtmf-pound.wav
dtmf-star.wav  types.py  utils.py
```

So on a WebSocket telephony call, an outbound keypress is a **synthesized tone WAV pushed through
`write_audio_frame` as ordinary speech audio** — which means it then goes through §6.4's
`pcm_to_ulaw`, gets 8 kHz μ-law companded like a voice, and arrives at the far end as sound. Whether
the far-end IVR hears it depends on that IVR's DTMF detector, on the carrier's transcoding, and on
whether any voice-optimised processing in between mangles the dual-tone. There is no out-of-band
signalling path.

**⚠️ CORRECTION 2.** [[transport-telephony]] states that `_supports_native_dtmf()` "is only
overridden by the Daily transport." The tree disagrees — there are **two** overrides:

```bash
$ grep -rn "_supports_native_dtmf" src/pipecat/
src/pipecat/transports/base_output.py:270:        if self._supports_native_dtmf():
src/pipecat/transports/base_output.py:287:    def _supports_native_dtmf(self) -> bool:
src/pipecat/transports/daily/transport.py:2246:    def _supports_native_dtmf(self) -> bool:
src/pipecat/transports/livekit/transport.py:1018:    def _supports_native_dtmf(self) -> bool:
```

LiveKit implements it too, and with a documented limitation worth knowing:

**`src/pipecat/transports/livekit/transport.py` L1026–1038**

```python
    async def _write_dtmf_native(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Use LiveKit's native publish_dtmf method for telephone events.

        LiveKit's DTMF API sends a single tone per call, so when
        ``frame.buttons`` contains multiple entries only the first one is
        sent.
        ...
        """
        if not frame.buttons:
            return
        await self._client.send_dtmf(frame.buttons[0].value)
```

Daily's is richer — it forwards `session_id`, `digit_duration_ms` and `method` when the frame is a
`DailyOutputDTMFFrame` (`daily/transport.py:2254–2276`). The corrected statement: **native outbound
DTMF exists on exactly two transports, both of which are WebRTC/SFU transports; the WebSocket +
telephony-serializer path has none.**

**Why you should care.** Korean insurance tele-sales touches ARS/IVR trees constantly — transferring
to a human agent queue, navigating a carrier's authentication menu, confirming with a keypress
instead of a spoken number (which, per §6.6, is exactly the input most damaged by the 4 kHz
ceiling). If Lina ever needs to *press* a key rather than *hear* one, on a WebSocket telephony path
that is tone injection into the voice band, and you should validate it against your carrier before
designing a flow that depends on it.

### 6.11 Genesys is the outlier that shows how big a serializer can get

964 lines — three times Twilio — and it is not bloat. `GenesysAudioHookSerializer` implements a
stateful, sequence-numbered, resumable session protocol *inside a serializer*:

**`src/pipecat/serializers/genesys.py` L184–206**

```python
        # Protocol state
        self._client_seq = 0
        self._server_seq = 0
        self._is_open = False
        self._is_paused = False
        self._position = timedelta(0)

        # Session metadata
        self._conversation_id: str | None = None
        self._participant: dict[str, Any] | None = None
        self._custom_config: dict[str, Any] | None = None
        self._media_info: list[dict[str, Any]] | None = None
        self._input_variables: dict[str, Any] | None = None  # Custom input from Genesys
        self._output_variables: dict[str, Any] | None = None  # Custom output to Genesys

        # Event handlers
        self._register_event_handler("on_open")
        self._register_event_handler("on_close")
        self._register_event_handler("on_ping")
        self._register_event_handler("on_pause")
        self._register_event_handler("on_update")
        self._register_event_handler("on_error")
        self._register_event_handler("on_dtmf")
```

Read that list against §6.3's four-method ABC. Nothing in `FrameSerializer` suggests a serializer
may have seven event handlers, two sequence counters, a pause state and an ISO-8601 stream position
— but `FrameSerializer` extends `BaseObject`, which supplies the event machinery, and nothing stops
you. `_format_position` / `_parse_position` (L274–300) serialise `timedelta` as ISO 8601 durations
(`"PT1.5S"`) because Genesys's AudioHook protocol demands a stream position on most messages.

Two things to take from this:

1. **The serializer seam has no size ceiling.** If a Korean carrier's protocol has a handshake,
   heartbeats, resumption and per-message sequence numbers, all of that fits in a serializer, and
   Genesys is your reference implementation for the shape.
2. **Sequence discipline in Pipecat lives in a serializer**, not in a data-channel abstraction. That
   is precisely the observation §11.2 turns into a port plan for boson's `ControlEvent`.

---

## 7. The serializer is where the open sum type gets closed

Now the theory payoff, and it is the reason [[ch-02/read]] is a dependency of this chapter.

[[ch-02/read]] and [[ch-03/read]] §2.2 counted the waist: 133 classes declared in
`src/pipecat/frames/frames.py`, 131 ending in `Frame`, **123 transitive descendants of `Frame`**.
An open sum type that only grows.

A wire protocol cannot be open. Whatever crosses the socket must be a fixed, agreed vocabulary that
both ends compiled against. So somewhere, the open sum type must be projected onto a closed one —
and in Pipecat that somewhere is the serializer. Look at how brutal the projection is:

**`src/pipecat/serializers/protobuf.py` L48–63**

```python
    SERIALIZABLE_TYPES = {
        TextFrame: "text",
        OutputAudioRawFrame: "audio",
        TranscriptionFrame: "transcription",
        MessageFrame: "message",
        InterruptionFrame: "interruption",
    }
    SERIALIZABLE_FIELDS = {v: k for k, v in SERIALIZABLE_TYPES.items()}

    DESERIALIZABLE_TYPES = {
        TextFrame: "text",
        InputAudioRawFrame: "audio",
        TranscriptionFrame: "transcription",
        MessageFrame: "message",
        InterruptionFrame: "interruption",
    }
```

**Five.** 123 frame types in the pipeline, five on the protobuf wire. And the handling of everything
else:

**`src/pipecat/serializers/protobuf.py` L97–100**

```python
        proto_frame = frame_protos.Frame()  # type: ignore[attr-defined]
        if type(serializable) not in self.SERIALIZABLE_TYPES:
            logger.warning(f"Frame type {type(serializable)} is not serializable")
            return None
```

A `logger.warning` and a `None`. Twilio's version of the same decision, quoted in §6.4, is even
quieter — `# Return None for unhandled frames` with no log at all, and its `serialize` recognises
exactly four cases (`EndFrame`/`CancelFrame`, `InterruptionFrame`, `AudioRawFrame`,
`OutputTransportMessage*`) while `deserialize` recognises two (`media`, `dtmf`).

Note also `type(serializable) not in self.SERIALIZABLE_TYPES` — an **exact type check**, not
`isinstance`. A subclass of `TextFrame` is not serializable by `ProtobufFrameSerializer`. That is
the closed union enforced at its strictest.

**And realtime_voice pays the identical tax at the identical place.** From
[[rtv-webrtc-transport]], `WebRTCVoiceTransport._control_event()` (`transport.py` L118) is a
hand-written mapping from boson's closed union to wire type strings:

```python
# packages/realtime_voice/realtime_voice/transport/webrtc/transport.py L118
# (boson-agent, private; excerpt-attested via [[rtv-webrtc-transport]])
#   VoiceEvent      -> event.kind.value      # already dotted, e.g. "assistant.audio_committed"
#   AgentTextDelta  -> "text_delta"
#   ASREvent        -> "transcript.interim" | "transcript.final"
#                      | "asr.end_of_turn" | "asr.error"
#   VADEvent        -> "vad.speech_started" | "vad.speech_stopped"
#   else            -> raise TypeError(f"unsupported voice event: {type(event).__name__}")
```

Same shape. Same location in the architecture. **One difference worth naming precisely, without
grading it:** the unknown case. Pipecat returns `None` (drop, sometimes with a warning);
realtime_voice raises `TypeError`. Those are the two ends of the same design axis that
[[ch-01/read]] §7 and [[ch-03/read]] §2.3 established — Pipecat's transparency-by-default versus a
closed union's exhaustiveness. It shows up here for the third time, which is how you know it is the
real axis of the whole comparison rather than an incidental style choice.

So the honest statement of the narrow-waist cost, now that you can see both ends:

> The narrow waist is only narrow *inside* the pipeline. At every wire boundary, somebody writes an
> explicit, closed, hand-maintained mapping from the open sum type onto a fixed protocol vocabulary.
> Pipecat writes one per provider (7 files, 2,456 lines). realtime_voice writes one, once
> (`_control_event`). Both are hand-maintained; the cost scales with the number of protocols, not
> with the number of frame types.

**⚠️ CORRECTION 3.** The outline says there are "exactly 7 concrete `FrameSerializer`
implementations." That is right for `src/pipecat/serializers/`, and wrong repo-wide:

```bash
$ grep -rn "class .*(FrameSerializer)" src/pipecat/
src/pipecat/evals/serializer.py:87:class RTVIEvalSerializer(FrameSerializer):
src/pipecat/serializers/exotel.py:29:class ExotelFrameSerializer(FrameSerializer):
src/pipecat/serializers/genesys.py:81:class GenesysAudioHookSerializer(FrameSerializer):
src/pipecat/serializers/plivo.py:32:class PlivoFrameSerializer(FrameSerializer):
src/pipecat/serializers/protobuf.py:40:class ProtobufFrameSerializer(FrameSerializer):
src/pipecat/serializers/telnyx.py:37:class TelnyxFrameSerializer(FrameSerializer):
src/pipecat/serializers/twilio.py:57:class TwilioFrameSerializer(FrameSerializer):
src/pipecat/serializers/vonage.py:28:class VonageFrameSerializer(FrameSerializer):
```

**Eight** under `src/pipecat/`, seven under `serializers/`. The eighth is `RTVIEvalSerializer`
(`evals/serializer.py`, 254 L). A ninth lives in the test suite —
`tests/test_websocket_server_transport.py:49:class _RawAudioSerializer(FrameSerializer)`. State the
count with its scope attached, because the eighth one is the one that gives you §11.3.

---

## 8. WebRTC: `SmallWebRTCTransport` and `DailyTransport`

### 8.1 SmallWebRTC — the same aiortc you already use

```bash
$ wc -l src/pipecat/transports/smallwebrtc/*.py
       0 __init__.py
     825 connection.py
     266 request_handler.py
    1085 transport.py
    2176 total
```

**`src/pipecat/transports/smallwebrtc/transport.py` L970–1003**

```python
    def __init__(
        self,
        webrtc_connection: SmallWebRTCConnection,
        params: TransportParams,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the WebRTC transport.
        ...
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        self._callbacks = SmallWebRTCCallbacks(
            on_app_message=self._on_app_message,
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
        )

        self._client = SmallWebRTCClient(webrtc_connection, self._callbacks)

        self._input: SmallWebRTCInputTransport | None = None
        self._output: SmallWebRTCOutputTransport | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
```

Two things stand out.

**It takes plain `TransportParams`**, not a subclass. Daily has `DailyParams`, LiveKit has its own,
the WebSocket transports have `FastAPIWebsocketParams` — SmallWebRTC has nothing provider-specific
to configure, because everything provider-specific is in the `SmallWebRTCConnection` you hand it.
Pattern A in §2.2 at its cleanest.

**⚠️ CORRECTION 4 (a documentation bug in the tree, not in an excerpt).** The class docstring
advertises an event that does not exist:

**`src/pipecat/transports/smallwebrtc/transport.py` L957–961**

```python
    Event handlers available:

    - on_client_connected(transport, client): Client connected to WebRTC session
    - on_client_disconnected(transport, client): Client disconnected from WebRTC session
    - on_client_message(transport, message, client): Received a data channel message
```

```bash
$ grep -rn "on_client_message" src/pipecat/transports/smallwebrtc/
src/pipecat/transports/smallwebrtc/transport.py:961:    - on_client_message(transport, message, client): Received a data channel message
```

One hit, and it is the docstring line itself. The registered handler for data-channel messages is
`on_app_message`. `@transport.event_handler("on_client_message")` will not work — `BaseObject`
only permits handlers that were registered. Trust the three `_register_event_handler` calls, not
the docstring.

**Media clocking** is where the two aiortc implementations differ measurably:

**`src/pipecat/transports/smallwebrtc/transport.py` L83–99, L101–124**

```python
    def __init__(self, sample_rate: int, auto_silence: bool = True):
        """Initialize the raw audio track.
        ...
        """
        super().__init__()
        self._sample_rate = sample_rate
        self._auto_silence = auto_silence
        self._samples_per_10ms = sample_rate * 10 // 1000
        self._bytes_per_10ms = self._samples_per_10ms * 2  # 16-bit (2 bytes per sample)
        self._timestamp = 0
        self._start = time.time()
        # Queue of (bytes, future), broken into 10ms sub chunks as needed
        self._chunk_queue = deque()

    def add_audio_bytes(self, audio_bytes: bytes):
        """Add audio bytes to the buffer for transmission.
        ...
        Raises:
            ValueError: If audio bytes are not a multiple of 10ms size.
        """
        if len(audio_bytes) % self._bytes_per_10ms != 0:
            raise ValueError("Audio bytes must be a multiple of 10ms size.")
        future = asyncio.get_running_loop().create_future()

        # Break input into 10ms chunks
        for i in range(0, len(audio_bytes), self._bytes_per_10ms):
            chunk = audio_bytes[i : i + self._bytes_per_10ms]
            # Only the last chunk carries the future to be resolved once fully consumed
            fut = future if i + self._bytes_per_10ms >= len(audio_bytes) else None
            self._chunk_queue.append((chunk, fut))

        return future
```

Against [[rtv-webrtc-transport]]'s account of your own `OutboundAudioTrack` (`tracks.py` L111):

| | Pipecat `RawAudioTrack` | realtime_voice `OutboundAudioTrack` |
|---|---|---|
| Granularity | **10 ms** (`_samples_per_10ms`) | **20 ms** (`packet_duration_ms = 20`, 960 samples @ 48 kHz) |
| Rate | pipeline `sample_rate` | `output_sample_rate = 48_000` |
| Non-aligned input | `raise ValueError("Audio bytes must be a multiple of 10ms size.")` | `av.AudioFifo` keeps the remainder across `recv()` |
| Write completion | returns an `asyncio.Future` resolved when consumed | none |
| Underflow | `auto_silence: bool = True` emits silence | waits `packet_seconds` |
| Generation change | — (handled upstream) | **throws away the FIFO** (L142–146, L152–154) |
| Silence buffer | — | explicit `plane.update(b"\x00" * plane.buffer_size)` because *"PyAV does not guarantee zero-initialized AudioFrame storage"* |

Both are correct implementations of the same idea with different alignment contracts and different
places to put the interruption flush. The write-completion `Future` and the FIFO-discard are the two
mechanisms with no counterpart on the other side; note both and move on — [[ch-08/read]] is where
they matter.

**Connection recovery.** [[ch-03/read]] §6.4 already gave you this, so one line: `pc_id`
(`connection.py:302`), `renegotiate(sdp, type, restart_pc=False)` (`:443`), `ask_to_renegotiate()`
(`:799`), plus a hand-rolled disconnect detector because *"aiortc does not provide any way so we can
be aware when we are disconnected"* (`:350`). realtime_voice's only recovery path is a fresh
`accept_offer(reconnect=True)`.

And what `SmallWebRTCConnection` takes:

**`src/pipecat/transports/smallwebrtc/connection.py` L245–249**

```python
    def __init__(
        self,
        ice_servers: list[str] | list[IceServer] | None = None,
        connection_timeout_secs: int = 60,
    ):
```

Two arguments. No token, no TTL, no customer binding — §10 returns to this.

### 8.2 Daily — the SFU path, and the one place Pipecat *does* ship PSTN

**`src/pipecat/transports/daily/transport.py` L2279** is `DailyTransport(BaseTransport)`, and its
params class adds the provider surface:

**`src/pipecat/transports/daily/transport.py` L339–346 (selected fields)**

```python
    audio_in_user_tracks: bool = True
    ...
    dialin_settings: DailyDialinSettings | None = None
    ...
    transcription_enabled: bool = False
```

**`src/pipecat/transports/daily/transport.py` L246–256**

```python
class DailyDialinSettings(BaseModel):
    """Settings for Daily's dial-in functionality.

    Parameters:
        call_id: CallId is represented by UUID and represents the sessionId in the SIP Network.
        call_domain: Call Domain is represented by UUID and represents your Daily Domain on the SIP Network.
    """

    call_id: str = ""
    call_domain: str = ""
```

**State this precisely, because it is easy to over- or under-claim.** §6.1's finding is that
Pipecat has no telephony *transport*, and that remains exactly true. But Daily's SFU has its own
SIP interconnect, and `DailyTransport` exposes it — `dialin_settings`, an `on_dialin_ready(sip_endpoint)`
callback (`daily/transport.py:1537`), `on_dialout_answered`, and a runner helper
`_maybe_apply_daily_dialin` (`runner/utils.py`) that merges dial-in settings from the webhook body.
So there are **two** ways to get a phone call into a Pipecat pipeline: (a) WebSocket + provider
serializer, where you see 8 kHz μ-law and own the codec boundary; (b) Daily dial-in, where Daily
terminates SIP and you receive normal WebRTC media. Path (b) does not make §6.6 go away — the
audio still originated on a PSTN leg — but it moves the codec boundary out of your process and into
Daily's, and it takes the DTMF path with it (§6.10: Daily is one of the two transports with native
outbound DTMF).

Two more Daily details that are directly useful to you:

**Room join, and a performance default:**

**`src/pipecat/transports/daily/transport.py` L817–826**

```python
    async def join(self):
        """Join the Daily room with configured settings."""
        logger.info(f"Joining {self._room_url}")

        # For performance reasons, never subscribe to video streams (unless a
        # video renderer is registered).
        self._client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "screenVideo": "unsubscribed"}}
        )
```

**Per-participant audio capture** — `capture_participant_audio(participant_id, callback,
audio_source="microphone", sample_rate=16000, callback_interval_ms=20)` at `daily/transport.py:1157`.
For an eval/QA harness this is the feature that gives you separated speaker tracks rather than a
mixed-down recording, which is the difference between being able to score barge-in behaviour and
not. Worth noting even though [[ch-11/read]] owns observability.

---

## 9. The accounting against `gateway/server/` (1,404 lines)

Everything in this section is from [[boson-gateway-server]] — Class B evidence per §0.

### 9.1 What is actually in those 1,404 lines

| File | LOC | What it does |
|---|---|---|
| `websocket.py` | 734 | `GatewayWebSocketServer` (L35); socket lifecycle, ten per-session maps, dispatch generations, silence timer |
| `access.py` | 374 | bearer auth, subprotocol token, origin allowlist, principal-to-session binding, signed play cookie |
| `protocol.py` | 114 | the 3-field JSON envelope, type vocabularies, session-id validation |
| `interruption.py` | 95 | barge-in policy |
| `history.py` | 70 | `serialize_history()` debug projection |
| `__init__.py` | 17 | |
| **total** | **1 404** | |

### 9.2 The envelope maps onto a `FrameSerializer` almost exactly

```python
# packages/gateway/gateway/server/protocol.py L15-31
# (boson-agent, private; excerpt-attested via [[boson-gateway-server]])
@dataclass
class ClientMessage:  session_id: str; type: str; content: str
@dataclass
class ServerMessage:  session_id: str; type: str; content: str = ""
```

Three string fields, both directions. No frame id, no timestamp, no sequence number, no audio field,
no binary path. `VALID_CLIENT_TYPES` (L33) is four entries — `user_message`, `partial_transcript`,
`interrupt`, `get_history`. `VALID_SERVER_TYPES` (L39) declares six, but the excerpt records that
grepping every `ServerMessage(...)` construction site yields only **four** emitted types: `error`
(websocket.py:231, 240, 263, 500), `history` (:281), `text_delta` (:471), `turn_end` (:477).
`interrupted` and `stage_changed` are declared and never sent — and `agents/test-lina-gateway/client.py:113`
has a live `elif data["type"] == "stage_changed":` branch that is dead code today.

That is a small, closed vocabulary with a clean projection onto §7's picture: `protocol.py` becomes
a `FrameSerializer` subclass whose `serialize` has four branches and whose `deserialize` has four.
It is the single cleanest piece of this migration. **Do not** port `stage_changed` as if it exists;
per the excerpt it would be new work.

### 9.3 What has no Pipecat counterpart at the transport layer

Per [[boson-gateway-server]], `GatewayWebSocketServer.__init__` (`websocket.py:43–138`) keeps **ten**
per-session maps: `_session_connections`, `_session_owner`, `_session_timers`, `_partial_transcripts`,
`_partial_transcript_owners`, `_partial_finalize_claims`, `_active_tasks`, `_active_started_at`,
`_dispatch_locks`, `_dispatch_generations`. Group them by what a migration does to them:

| boson mechanism | Where it goes on Pipecat | Why |
|---|---|---|
| socket lifecycle: `start()`, `_handle_connection()`, teardown | **replaced** by `FastAPIWebsocketTransport` | §5.1 — you keep the route, it takes the accepted socket |
| `protocol.py` envelope | **ported** to a `FrameSerializer` subclass | §9.2 |
| `_start_silence_timer` / `_finalize_partial` | **relocated** to `src/pipecat/turns/` | §4 — endpointing is not transport-layer in this version |
| `access.py` (374 L) | **kept**, in your FastAPI route | §5.1 — Pipecat ships an Origin allowlist and nothing else |
| `history.py` + the `history_lock` per-turn serialization | **kept** | no Pipecat equivalent; it is a debug projection over *your* session store |
| `_reserve_session_dispatch` / `_replace_active_task` / `_cancel_session_dispatch` | **kept** | `PipelineTask` cancels a bot turn; it has no concept of two sockets racing for one session |
| session identity, reconnect-and-resume, 1800 s idle TTL | **kept**, in an outer layer | Pipecat pipelines are per-connection |

The generation protocol deserves a sentence of its own because it is the least replaceable thing
in the file. Per the excerpt, `_reserve_session_dispatch` (:515) bumps `_dispatch_generations[sid]`
**before any await** — *"event-loop tasks cannot interleave until an await, so incrementing the
generation here records ordering across connection reader loops"* — and `_cancel_active_task` (:584)
is a v0.7.4 hotfix that *awaits* the cancelled task so `except CancelledError` has already stashed
`session._pending_partial` before the successor reads session state. Both facts are about **two
connections competing for one logical session**. [[ch-04/read]] §7 showed you Pipecat's cancellation
model; it is per-pipeline, and a Pipecat pipeline is per-connection. There is nothing to map onto.

And `access.py` is where the largest single block of retained code lives:

- bearer via `Authorization` header **or** — because browser WebSocket APIs cannot set headers — a
  companion subprotocol `boson-bearer.<token>` (:296), with `select_subprotocol` (:43) negotiating
  `"boson-gateway"`
- constructor invariants (`websocket.py:56–66`): `auth_token` ≥ 32 chars; a non-loopback host
  **requires** a token; enabling `customer_db_path` requires `allowed_origins`
- `SessionAccess.authorize(websocket, session_id, operation)` (:338) binding a session to a
  *principal* — sha256 of the token, or a per-connection `secrets.token_urlsafe(32)` when tokenless
  — with `get_history` on an unbound session **refused** (:343–344)
- a per-process HMAC-SHA256 signed cookie `boson_gateway_play`, `PLAY_COOKIE_MAX_AGE_SECONDS = 300`,
  `HttpOnly; SameSite=Strict`

The Pipecat counterpart to all of that is the twelve-line `is_origin_allowed` in §5.1.

### 9.4 The number

[[boson-gateway-server]]'s conclusion, which the file inventory supports: **budget ~700 lines
retained and ~700 lines ported, not "delete 1,404 lines."** What you *gain* on the other side is
the whole audio path boson has never had — media in and out, VAD wiring, `InterruptionFrame`
propagation, six telephony serializers, and turn strategies under `src/pipecat/turns/`. What you
*spend* is rewriting the socket plumbing and the envelope, relocating endpointing, and keeping
everything above the frame layer where it already is. Both halves of that sentence are facts;
neither is a recommendation.

---

## 10. The accounting against `realtime_voice/transport/webrtc/`

Also Class B evidence, from [[rtv-webrtc-transport]].

### 10.1 ⚠️ An arithmetic problem in the excerpt, reported rather than repeated

[[rtv-webrtc-transport]] states the module is **"~960 LOC"** and then lists the files:

| File | LOC (per excerpt) |
|---|---|
| `manager.py` | 248 |
| `control.py` | 226 |
| `peer.py` | 231 |
| `tracks.py` | 216 |
| `buffer.py` | 123 |
| `config.py` | 64 |
| `transport.py` | 168 |
| **sum** | **1 276** |

248 + 226 + 231 + 216 + 123 + 64 + 168 = **1,276**, not ~960. The excerpt's own per-file table
contradicts its headline number by 316 lines — about 33%. [[ch-03/read]] §6.2 repeated the ~960
figure without catching this, and the course outline repeats it again for this chapter.

I cannot resolve it: the repo is not on this machine and rule 3 of this course forbids opening it.
Two plausible explanations — the 960 excludes blank lines and docstrings while the per-file numbers
are raw `wc -l`, or the headline is simply stale — and I have no evidence for either. **Use 1,276
as the file-listing total and treat "~960" as unverified until you can run `wc -l` yourself.** If
you are sizing a migration off this number, the difference between 960 and 1,276 is a week.

For scale on the other side, the Pipecat number *is* checkable: `src/pipecat/transports/smallwebrtc/`
is 2,176 lines across three files (§8.1).

### 10.2 The same ground, and the two things Pipecat does not ship

Both implementations wrap aiortc's `RTCPeerConnection`, both resample with PyAV, both drive a data
channel, both pace outbound audio against a wall clock. The divergences the excerpt records are
policy sitting inside the transport package:

**`WebRTCSessionManager`** (`manager.py` L51) — docstring *"Create short-lived authorized sessions
and enforce one live peer each."* `create_session(customer_id, ...) -> VoiceSessionTicket`, minting
`secrets.token_urlsafe(32)` and storing only `hashlib.sha256(token).digest()`; `_authorize` (L227)
checks expiry then `hmac.compare_digest`; `session_token_ttl_seconds = 15 * 60`;
`accept_offer(..., reconnect: bool = False)` raises `SessionConflictError("this voice session
already has a live peer")` unless `reconnect=True`.

Pipecat side, checked in the tree:

```bash
$ grep -rn "token_urlsafe\|compare_digest" src/pipecat/
src/pipecat/runner/run.py:324:    if not hmac.compare_digest(expected, sig):
src/pipecat/transports/whatsapp/client.py:181:        if not hmac.compare_digest(expected_signature, received_signature):
```

Both hits are webhook-signature verification. Neither is voice-session authorization. Combined with
§8.1's two-argument `SmallWebRTCConnection.__init__`, the finding is: **no tokens, no TTL, no
customer binding, no one-live-peer rule anywhere in Pipecat's WebRTC layer**, and
`request_handler.py` is a bare offer/answer endpoint.

**`ControlEvent`** (`control.py` L25) — versioned (`CONTROL_PROTOCOL_VERSION = 1`),
sequence-numbered, size-capped at 64 KiB, with `_reject_audio_payload` (L117) recursively refusing
anything that looks like audio, and `OrderedControlChannel` (L136) refusing a partially-reliable
data channel at construction. Pipecat's counterpart is `on_app_message(message: Any, sender: str)`
— §8.1's registered handler — with no schema, no sequence check, no size cap, no audio ban. Pipecat
*does* have a typed client protocol (RTVI, [[rtvi-observability]]), but it rides a different layer
and is not enforced at the data channel.

### 10.3 The two-column statement

| | Pipecat `smallwebrtc/` (2,176 L, verified) | realtime_voice `transport/webrtc/` (1,276 L per file listing, §10.1) |
|---|---|---|
| Library | aiortc | aiortc |
| Output granularity | 10 ms, `ValueError` on misalignment | 20 ms, `AudioFifo` keeps remainder |
| Renegotiation / ICE restart | `renegotiate(..., restart_pc)`, `ask_to_renegotiate()` | absent; only `accept_offer(reconnect=True)` |
| Video / screen share | present | absent by construction (`__post_init__` rejects `output_channels != 1`) |
| Session authorization | none | HMAC token, 15 min TTL, one live peer, explicit reconnect |
| Data-channel protocol | untyped `Any` passthrough | versioned, ordered, size-capped, audio-forbidden |
| Named error types | general exceptions | 7 (`webrtc/errors.py`, 33 L) |
| Other transports in the same package family | 10 more | none |

Adopting Pipecat's transport here means the left column arrives and the two right-column policy
rows have to be **re-implemented as application code around `SmallWebRTCConnection`**, because
Pipecat ships neither. That is the entire statement. [[ch-13/read]] decides what to do about it.

---

## 11. Four framework-extension moves for Lina

### 11.1 Spec `LinaCarrierSerializer(FrameSerializer)` before you talk to a carrier

You will write pattern B (§2.2). Here is the checklist, derived from what the six shipped
serializers actually do, so you can hand it to a carrier's integration team as a requirements list
rather than discovering each item during a call:

| # | Requirement | Where the shipped code does it |
|---|---|---|
| 1 | Learn the pipeline rate in `setup()` | `twilio.py:164` — `self._sample_rate = self._params.sample_rate or setup.audio_in_sample_rate` |
| 2 | Decode companding *then* resample inbound | `audio/utils.py:222–226` |
| 3 | Resample *then* encode companding outbound | `twilio.py:193–195` |
| 4 | Emit the provider's flush message on `InterruptionFrame` | `twilio.py:186–188` |
| 5 | Decide the `EndFrame`/`CancelFrame` behaviour explicitly (hang up, or not) | `twilio.py:178–185`; Exotel deliberately does not |
| 6 | Echo the stream identifier on every outbound message | `twilio.py:187, 203` |
| 7 | Return `InputDTMFFrame(KeypadEntry(digit))` for inbound keypresses | `twilio.py:305–312` |
| 8 | Set `resampler_clear_after_secs=None` if their chunk timing is irregular | `base_serializer.py:38–40` |
| 9 | Decide fixed packet size, if they demand one | `fastapi.py:575–586` |

Two questions to ask the carrier *first*, because they determine everything above: **what sample
rate and codec on the wire**, and **is there a flush/clear message**. If the answer to the second
is "no", barge-in on that carrier is bounded by their playout buffer depth and you should find that
number out before promising a barge-in latency (§6.9, [[ch-11/read]]).

Budget: 250–300 lines for a Twilio-shaped protocol, up to ~1,000 if it has Genesys-shaped session
state (§6.11).

### 11.2 Port `ControlEvent` as a serializer concern, not a data-channel concern

boson's `ControlEvent` and `OrderedControlChannel` (§10.2) enforce their invariants at the
**aiortc data channel**. Pipecat has no hook there — `on_app_message` takes `Any`. But §6.11
showed you that `GenesysAudioHookSerializer` maintains `_client_seq` / `_server_seq`, a session
open/close/pause state machine, and seven event handlers *inside a serializer*, and §5.3 showed you
that inbound non-audio frames arrive as `InputTransportMessageFrame` and get broadcast both
directions.

So the port is: **`ControlEvent` validation moves into `deserialize()`/`serialize()`.**

- `from_json`'s unknown-field rejection, version check and `_reject_audio_payload` → the top of
  `deserialize()`, raising or returning `None` per your chosen policy (note §7: Pipecat's house
  style is `None`, yours is `raise` — pick deliberately, do not inherit by accident)
- the strict in-order `sequence` check → serializer instance state, exactly like Genesys's
  `_client_seq`
- the outbound private counter → serializer instance state, exactly like Genesys's `_server_seq`
- the 64 KiB cap → a length check in `serialize()`
- `ordered=False` / `maxRetransmits` / `maxPacketLifeTime` refusal → **has no serializer analogue**
  on the WebSocket path, because TCP is already ordered and reliable. On a WebRTC path it stays at
  the data channel and must live in your `SmallWebRTCConnection` setup code.

That last bullet is the one to notice: five of the six invariants relocate cleanly, and the sixth
is only meaningful on the transport you would be replacing.

### 11.3 Make the serializer seam your eval seam

`RTVIEvalSerializer` (§5.7, §7's CORRECTION 3) is Pipecat's own harness: a serializer plus
`SingleClientWebsocketServerTransport`. The generalisation is that **a serializer is a
protocol-shaped test double for the entire I/O boundary**, and it costs nine lines when it needs to
(`tests/test_websocket_server_transport.py:49`).

The concrete move for Lina, and it is the highest-value thing in this section: build a
`ReplayCarrierSerializer` whose `deserialize()` reads recorded 8 kHz μ-law frames off disk on a
20 ms schedule and emits `InputAudioRawFrame` through the identical `ulaw_to_pcm` path, and whose
`serialize()` writes the bot's μ-law output to a file instead of a socket. Then:

- your STT accuracy numbers are measured on the **real** signal chain (§6.6), not on studio audio
- your barge-in timing is measured against a deterministic clock, so [[ch-11/read]]'s budget has
  reproducible inputs
- regression tests run with no carrier, no network, no cost, in CI
- and the pipeline under test is *byte-identical* to production, because the only thing you swapped
  is `params.serializer`

This is the framework-extension move that turns §6.6 from a warning into a measurement.

### 11.4 Decide the browser/console path separately from the call path

They are different transports and they should be different decisions:

- **the live PSTN call**: `FastAPIWebsocketTransport` + your carrier serializer (§6)
- **the agent console / supervisor listen-in / QA harness**: a WebRTC transport, where
  `capture_participant_audio(..., audio_source="microphone", sample_rate=16000)` (§8.2) gives you
  separated speaker tracks, and where you already have ~1,276 lines of working aiortc code (§10)

Nothing forces one answer for both, and treating them as one question is how the migration
estimate gets wrong. [[ch-13/read]] scores them separately for exactly this reason.

---

## 12. The transport-layer fact sheet

Verified against `pipecat-src` at `0cbf9c5b` unless marked ‹excerpt›.

| Fact | Value |
|---|---|
| `BaseTransport` | `base_transport.py:96`, 42 lines, in a 137-line file |
| Abstract methods | 2 — `input()` :122, `output()` :131, both `-> FrameProcessor` |
| Transport provider packages | **11** |
| `BaseTransport` subclasses | **13** (websocket 3, local 2, whatsapp **0**) |
| `TransportParams` audio defaults | `audio_in_enabled=False`, `audio_out_enabled=False`, `audio_out_10ms_chunks=4` |
| Sample-rate fallbacks | 16 000 in / 24 000 out — `frame_processor.py:106–107` |
| Output chunk math | `base_output.py:135–136`; 1 920 B @ 24 kHz = 40 ms |
| `vad_analyzer` on `TransportParams` | **removed** — CHANGELOG L4402 |
| Live VAD mount points | `LLMUserAggregatorParams.vad_analyzer`, `VADProcessor`, `VADController` |
| `FastAPIWebsocketTransport` | `websocket/fastapi.py:611`; takes an already-accepted socket |
| `FastAPIWebsocketParams` extra fields | exactly 6, incl. `serializer` and `ws_close_timeout: float = 0.5` |
| No serializer set | every inbound message dropped (`continue`, :376–377); every outbound dropped (`return False`, :568–569) |
| Fabricated audio clock | `_send_interval = (audio_chunk_size / sample_rate) / 2` :456; `/2` is the bytes→samples factor, correct only for 16-bit mono |
| WebSocket transport modules | fastapi 707 L, server 716 L, client 559 L |
| Dev server single-client | close code 1013, `server.py:255` |
| Telephony transports | **0** |
| Telephony serializers | **6** — twilio 314, telnyx 292, plivo 256, exotel 171, genesys 964, vonage 188 |
| Concrete `FrameSerializer` subclasses | **7** under `serializers/`, **8** under `src/pipecat/` (`RTVIEvalSerializer`) |
| Wire rate defaults | 8 000 Hz ×5; Vonage 16 000 Hz |
| Nyquist ceiling at 8 kHz | **4 kHz**; μ-law is 8-bit companded |
| Telephony wiring proof | `runner/utils.py:486` → `:554 return FastAPIWebsocketTransport(...)` |
| Barge-in flush | `clear` / `clearAudio`, provider-specific, via `serialize(InterruptionFrame)` |
| `auto_hang_up` | `True` by default on Twilio/Telnyx/Plivo; absent on Exotel |
| Outbound DTMF | no serializer support; `_write_dtmf_audio()` pushes tone WAVs as audio |
| Native outbound DTMF | **2 transports** — Daily `:2246`, LiveKit `:1018` |
| `KeypadEntry` | 12 entries, no A/B/C/D |
| `SmallWebRTCTransport` | `smallwebrtc/transport.py:951`; plain `TransportParams`; package 2 176 L |
| `SmallWebRTCConnection.__init__` | `ice_servers`, `connection_timeout_secs=60` — nothing else |
| Registered SmallWebRTC events | 3 — `on_app_message`, `on_client_connected`, `on_client_disconnected` (docstring's `on_client_message` does not exist) |
| Protobuf wire vocabulary | **5** frame types, exact-type match, out of 123 `Frame` descendants |
| boson `gateway/server/` ‹excerpt› | 1 404 L; ~700 keep / ~700 port |
| boson client message types ‹excerpt› | 4 valid; 6 server types declared, **4** ever emitted |
| realtime_voice `transport/webrtc/` ‹excerpt› | per-file listing sums to **1 276 L** (headline "~960" is unverified — §10.1) |

---

## 13. What this chapter does not settle

- **When the user's turn ends.** §4 moved endpointing off the transport and named
  `src/pipecat/turns/` as its home. boson's `_start_silence_timer` / `_finalize_partial` port is
  [[ch-06/read]]'s.
- **Which Korean STT survives 8 kHz μ-law.** §6.6 established the constraint and the measurement
  rule. Provider selection is [[ch-07/read]] with [[stt-korean-providers]].
- **The full interruption cascade.** §5.4, §6.9 and §8.1 each gave you one link. The chain —
  processor queue, transport buffer, carrier playout, and boson's text-transcript-based decision
  logic — is [[ch-08/read]].
- **Where the tool-call repair goes when a turn is cancelled mid-flight.** Parked in
  [[ch-04/read]]'s 다음 챕터로, owned by [[ch-09/read]].
- **Per-hop latency numbers.** §5.4's fabricated clock, §6's two resamples and §8.1's granularity
  difference are all latency terms with no measurements attached. [[ch-11/read]].
- **Keep or replace.** §9.4 and §10.3 are inventories, deliberately unscored. [[ch-13/read]].

---

## 다음 챕터로

What this chapter hands forward, named so later chapters cite it instead of re-deriving it:

- **A transport is a pair of processors** (§1). `BaseTransport` is 42 lines and two abstract methods
  returning `FrameProcessor`. Every later chapter that says "at the transport boundary" means one of
  those two objects, not a server.
- **The three provider patterns** (§2.2) — new transport class / new serializer / new signalling
  client, with `DailyTransport`, `TwilioFrameSerializer` and `WhatsAppClient` as the three witnesses.
  [[ch-13/read]] needs this to size the carrier-integration line item.
- **The serializer is the codec boundary, the resample point, the call-control client and the DTMF
  decoder** (§6.4). [[ch-07/read]] needs the resample point, [[ch-08/read]] needs the `clear` event,
  [[ch-11/read]] needs both resamples as latency terms.
- **The 4 kHz ceiling** (§6.6) and the measurement rule that follows from it: a Korean STT number
  measured on 16 kHz audio is not transferable to this pipeline. [[ch-07/read]] is built on this and
  [[ch-13/read]] scores against it.
- **The fabricated audio clock** (§5.4) — a WebSocket has no playout back-pressure, so the transport
  sleeps to emulate a device, and interruption zeroes the schedule. [[ch-08/read]] and
  [[ch-11/read]] both spend it.
- **VAD is not on the transport** (§4). Any port plan aimed at `TransportParams` is aimed at an
  older Pipecat. [[ch-06/read]] owns the relocation.
- **The wire is always a closed union** (§7). Pipecat closes it per provider in a serializer;
  realtime_voice closes it once in `_control_event`. [[ch-13/read]]'s open-versus-closed question
  now has a third data point.
- **Two migration inventories** (§9, §10) — ~700 keep / ~700 port on `gateway/server/`, and two
  named policy subsystems (`WebRTCSessionManager`, `ControlEvent`) that become application code on
  the WebRTC side. Both unscored, both waiting for [[ch-13/read]].
- **Four corrections** — the `ls` listing includes `__init__.py`; `_supports_native_dtmf` has two
  overrides, not one; there are 8 `FrameSerializer` subclasses repo-wide, not 7; and
  `SmallWebRTCTransport`'s docstring advertises an `on_client_message` event that is never
  registered. Plus one unresolved excerpt discrepancy: realtime_voice's WebRTC package is listed at
  1,276 lines and headlined at ~960 (§10.1).

[[ch-06/read]] takes the next question in the audio path, and it is the one §4 kept deferring:
**when does the user's turn end?** VAD as a hysteresis machine, streaming STT endpointing, and the
turn-strategy chain under `src/pipecat/turns/` — against boson's silence timer in the gateway and
realtime_voice's `SileroVAD` that raises on anything but 16 kHz mono. You now know that the audio
reaching that machinery came off an 8 kHz wire and was interpolated up. Go find out what the turn
detector does with it.
