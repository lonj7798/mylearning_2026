---
title: "Transports: WebRTC, WebSocket, 그리고 serializer로서의 telephony"
chapter: ch-05
phase: voice-io
course: pipecat
lang: ko
companion_of: read.md
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

# Chapter 5 — Transports: WebRTC, WebSocket, 그리고 serializer로서의 telephony

> 이 문서는 [[read]] ([read.md](read.md))의 한국어 companion입니다. section 번호는 원문과 1:1로 대응하므로
> 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, 줄 번호, 수치는 원문 그대로입니다.
> CS/ML 용어는 영어를 유지합니다 (frame, processor, pipeline, queue, aggregator, endpointing,
> back-pressure, interruption, timestamp, serializer, transport, companding 등).

> **Scope — 미리 명시하고, 이 chapter 전체에 대해 강제합니다.** 이 chapter는 각 design이 **무엇을 하는지**
> 기술합니다. 순위를 매기지 않습니다. Pipecat의 transport layer가 `gateway/server/`보다 낫다거나,
> `realtime_voice/transport/webrtc/`를 유지해야 한다/버려야 한다고 말하는 문장은 여기에 단 한 줄도
> 없습니다. §9, §10, §12에 나오는 모든 "X를 채택하면 Y를 지불한다" 류의 문장은 회계(accounting)
> 진술입니다 — 유지하는 줄 수, 포팅하는 줄 수, 새로 생기는 capability, 사라지는 capability — 권고가
> 아닙니다. [[ch-13/read]]가 이 course에서 무언가를 채점하는 유일한 chapter입니다. §9를 읽으면서
> 판정(verdict)을 원하게 된다면, 그건 이 chapter가 제대로 작동하고 있다는 뜻입니다: 이 chapter는
> ch-13이 소비할 evidence를 모으고 있는 중입니다.

---

## 왜 이 챕터인가

[[ch-04/read]]는 runtime을 세워 놓고 canonical chain에 의도적으로 구멍 두 개를 남겼습니다:

```
transport.input() → stt → user_aggregator → llm → tts → transport.output() → assistant_aggregator
        ▲                                                        ▲
     position 1                                             position 6
     (empty in ch-04)                                       (empty in ch-04)
```

이 chapter가 그 구멍을 메웁니다. 그리고 가장 먼저 내면화해야 할 사실은 **그 구멍을 메우는 데 ch-04로부터
필요한 것이 거의 없다**는 점입니다. transport는 runtime object가 아닙니다. schedule되지 않고, task를
소유하지 않으며, `PipelineWorker`가 무엇인지 알지 못합니다. `BaseTransport`는 42줄짜리 class이고, 그
public surface 전체는 각각 `FrameProcessor`를 반환하는 abstract method 두 개입니다. 따라서 이 chapter의
prerequisite은 [[ch-01/read]] — uniform interface인 `process_frame` / `push_frame`, 그리고 아무것도
validate하지 않는 `link()` — 와 [[ch-02/read]] — narrow waist, open sum type으로서의 `Frame` — 입니다.
runtime이 아닙니다.

이게 중요한 이유는, transport 문제가 *infrastructure* 문제가 아니라 *composition* 문제라는 뜻이기
때문이고, 그 답은 머릿속에 통째로 담을 수 있을 만큼 작기 때문입니다:

> Pipecat transport는 processor 한 쌍입니다. provider-specific한 모든 것은 그 두 processor가 감싸는
> connection object 뒤에 있거나, params field를 통해 그들에게 건네주는 `FrameSerializer` 안에 있습니다.
> 세 번째 장소는 없습니다.

이 chapter가 존재하는 두 번째 이유: **boson에 구체적인 counterpart가 등장하는 첫 chapter**입니다.
[[ch-03/read]]는 `realtime_voice`를 하나의 전체로서 특징지었습니다. 여기서는 그 비교가 당신 자신의
subsystem 두 개에 대해 동시에 파일-과-줄 단위로 구체화됩니다 — `packages/gateway/gateway/server/`
(1,404줄, text protocol, audio 없음)와 `packages/realtime_voice/realtime_voice/transport/webrtc/`
(aiortc, Pipecat의 `SmallWebRTCTransport`가 쓰는 것과 같은 라이브러리).

그리고 세 번째 이유, Lina TMR에 대해 실제로 무언가를 결정하는 이유: **Pipecat에는 telephony transport가
없습니다.** 전화 통화는 WebSocket transport 하나에 ~300줄짜리 serializer를 더한 것입니다. 즉 한국
보험 tele-sales agent를 위한 migration 질문은 "어떤 transport를 고를까"가 아니라 — "내가 작성해야 하는
serializer는 어떻게 생겼고, 8 kHz μ-law가 한국어 STT에 무슨 짓을 하는가"입니다. §6이 이 chapter에서
가장 긴 section인 이유가 그것입니다.

---

## 0. 이 chapter의 evidence를 읽는 법

[[ch-03/read]] §0에서와 같이 주장은 두 부류이고, 두 부류의 검증 가능성은 동등하지 않습니다.

**Class A — Pipecat.** 모든 경로, 줄 번호, class 이름, count, LOC 수치는 commit
`0cbf9c5b031eef06e53f0a193b9a67d60230e6be` 시점의 `wiki/raw-data/pipecat/pipecat-src`에 대해 검증
가능합니다. 명령어를 찍어 놓은 곳은 제가 실제로 실행한 것입니다. curated excerpt가 tree와 어긋나는
곳은 그렇다고 말하고, tree가 이깁니다. 이 chapter에는 그런 수정이 **네 개** 있고 전부 inline으로
**⚠️ CORRECTION**이라고 표시했습니다.

**Class B — boson.** `gateway/server/`와 `realtime_voice/transport/webrtc/`는 이 머신에 없는 private
repo에 있습니다. 그것들에 대한 모든 수치는 [[boson-gateway-server]]와 [[rtv-webrtc-transport]]에서
나오며, 이 둘은 branch `lina-new-dental-dev`와 `voice-chat-dev`에서 직접 읽은 것입니다. 그 수치들은
excerpt-attested이지 clone-verified가 아니며, boson code block에는 그렇다고 말하는 comment header를
붙여 두었습니다. 그중 하나에는 repo 없이는 해결할 수 없는 내부 산술 문제가 있어서, §10.1이 그것을
반복하는 대신 flag합니다.

---

## 1. `BaseTransport`는 42줄이고, method를 정확히 두 개 선언한다

abstraction 전체부터 시작합시다. 한 화면에 들어가니까요.

**`src/pipecat/transports/base_transport.py` L96–137** (파일은 137줄이고, 이 class가 파일의 마지막
요소이므로 `BaseTransport`는 96줄부터 137줄 — **42줄**입니다)

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

return type을 읽으십시오. `input()`과 `output()`은 `FrameProcessor`를 반환합니다 — `Pipeline`의 다른
모든 원소와 **같은 type**입니다. `Transport` 맛이 나는 processor도 아니고, pipeline이 알고 있는 추가
method를 가진 subtype도 아닙니다. 똑같은 type입니다.

그 사실 하나가 design 전부입니다. [[ch-01/read]] §3을 떠올리십시오: `link()`는 pointer 대입 두 개와
`logger.debug` 하나이고, `Pipeline._link_processors`는 validation 없이 그 operator를 fold한 것입니다.
그래서 canonical bot이 이렇게 쓸 때

```python
Pipeline([transport.input(), stt, context_aggregator.user(), llm, tts, transport.output(), ...])
```

pipeline은 position 1과 6에서 특별한 일을 하지 않습니다. `FrameProcessor`의 list에 대해 `link`를
fold할 뿐이고, 그중 두 개가 마침 뒤에 socket을 달고 있을 뿐입니다. `BaseTransport` 자체는 pipeline에
아예 들어가지 않습니다 — 그것은 pipeline에 들어가는 두 processor의 *factory*이자, 그 둘이 공유하는
connection의 holder입니다.

당신에게 load-bearing한 순서대로 정리한 결과들:

1. **provider를 바꾸는 것은 connection object와 params class를 바꾸는 것이고, 그 외에는 없습니다.**
   `DailyTransport(room_url, token, bot_name, params=DailyParams(...))`를
   `SmallWebRTCTransport(webrtc_connection, params=TransportParams(...))`로 교체하면 생성 코드 두 줄이
   바뀝니다. `Pipeline([...])` 줄은 byte 단위로 동일합니다. 이것이 [[transport-daily-webrtc]]의 Core
   Insight이고 tree가 그것을 뒷받침합니다.
2. **transport는 다른 어떤 processor와도 똑같은 방식으로 unit-testable합니다** — [[ch-01/read]] §6의
   substitutability 논증이 그대로 적용되며, 그래서 `tests/test_websocket_server_transport.py`가 아홉
   줄짜리 `_RawAudioSerializer(FrameSerializer)`를 정의해서 network 없이 transport 전체를 구동할 수
   있습니다.
3. **`BaseTransport`에는 policy를 놓을 자리가 없습니다.** auth hook도, session 개념도, protocol도
   없습니다. 그것들을 원한다면 위(당신의 ASGI route)에, 옆(processor)에, 또는 안(serializer)에
   들어갑니다. 이 생각을 §9까지 붙들고 계십시오 — 그것이 `access.py`가 migration에서 살아남는 이유
   전체입니다.

> 💡 **쉬운 설명 — "factory이자 holder"가 왜 중요한가요?**
> `Pipeline([...])`에 들어가는 건 `transport`가 아니라 `transport.input()`과 `transport.output()`이
> 반환한 **두 개의 별개 object**입니다. 그런데 그 둘은 같은 socket을 봐야 합니다 (들어온 오디오와
> 나가는 오디오가 같은 통화니까요). Pipecat은 그 공유를 상속이나 전역 상태가 아니라 "`BaseTransport`
> 인스턴스가 connection을 들고 있고, 두 processor에게 그 참조를 넘겨준다"로 해결합니다. 그래서
> pipeline은 transport를 전혀 모르는 채로 남고 (invariant 2 유지), 두 processor는 여전히 한 통화를
> 공유합니다. §5.6의 `_leave_counter`가 바로 이 공유의 부작용을 다루는 코드입니다.

### 1.1 파일의 나머지 절반: `TransportParams`

media configuration은 flat pydantic model이고, declarative하며, method가 없습니다.

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

Lina를 위해 머릿속에 새겨 둘 default 세 개:

- `audio_out_enabled: bool = False`와 `audio_in_enabled: bool = False`. **audio는 기본적으로 꺼져
  있습니다.** 맨 `TransportParams()`로 생성한 transport는 양방향 어디에도 audio를 나르지 않으며,
  pipeline 안에서 보기에는 마침 아무것도 수신하지 않는 정상 transport와 **정확히 똑같아 보입니다.**
  이것이 첫 한 시간 동안 가장 흔한 실패입니다.
- `audio_out_10ms_chunks: int = 4`. 이것이 [[ch-04/read]] §5.1의 *N*(`control latency = queue depth /
  drain rate`)이 구체화된 형태입니다: output 쪽은 wire에 한 번에 40 ms씩 건넵니다. 이 값은 아래 §3.1에
  다시 나오고, [[ch-08/read]]에서 interrupt granularity로 다시 나오며, [[ch-11/read]]에서 latency-budget
  항으로 또 나옵니다.
- sample rate는 default가 `None`입니다. "설정 안 하면 16000"이 아니라, setup 시점에 resolve됩니다.
  그게 §3입니다.

**⚠️ CORRECTION 1 (사소하지만, outline이 그 listing을 exhaustive하다고 진술함).** course outline은
`ls src/pipecat/transports/`가 `base_input.py`, `base_output.py`, `base_transport.py`와 디렉토리 11개를
반환한다고 기록합니다. tree에는 `__init__.py`도 있습니다:

```bash
$ ls src/pipecat/transports/
__init__.py  base_input.py  base_output.py  base_transport.py
daily/  heygen/  lemonslice/  livekit/  local/  moq/  smallwebrtc/
tavus/  vonage/  websocket/  whatsapp/
```

outline 주장에서 load-bearing한 부분은 그대로 유지됩니다 — **`twilio/` 없음, `telnyx/` 없음, `plivo/`
없음, `exotel/` 없음, `sip/` 없음** — 하지만 listing을 인용할 거라면 통째로 인용하십시오.

---

## 2. inventory 세기, 그리고 count가 1:1이기를 멈추는 지점

[[ch-03/read]] §6.2는 realtime_voice의 1개에 대한 breadth 수치로 이미 "11 transport packages"를
주었습니다. 그 숫자가 실제로 무엇으로 구성되어 있는지가 여기 있습니다. 구성이 count보다 흥미롭기
때문입니다.

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

**package 11개, subclass 13개.** outline은 그 격차를 "일부 package가 하나 이상을 ship한다"로
설명하는데, 맞지만 불완전합니다. package별로 회계를 해 보십시오:

| Package | `BaseTransport` subclass | 실제로 무엇인가 |
|---|---|---|
| `daily/` | 1 — `DailyTransport` | SFU room, native client SDK |
| `heygen/` | 1 — `HeyGenTransport` | avatar vendor |
| `lemonslice/` | 1 — `LemonSliceTransport` | avatar vendor |
| `livekit/` | 1 — `LiveKitTransport` | SFU room |
| `local/` | **2** — `LocalAudioTransport`, `TkLocalTransport` | host sound card; Tk window |
| `moq/` | 1 — `MOQTransport` | Media over QUIC |
| `smallwebrtc/` | 1 — `SmallWebRTCTransport` | direct aiortc peer |
| `tavus/` | 1 — `TavusTransport` | avatar vendor |
| `vonage/` | 1 — `VonageVideoConnectorTransport` | **video**, `serializers/vonage.py`와 무관 |
| `websocket/` | **3** — `FastAPIWebsocketTransport`, `SingleClientWebsocketServerTransport`, `WebsocketClientTransport` | server-side socket, dev server, dial-out client |
| `whatsapp/` | **0** | — |
| | **13** | |

### 2.1 0이 들어 있는 칸이 흥미로운 칸이다

`whatsapp/`은 transport *class*가 없는 transport *package*입니다:

```bash
$ ls src/pipecat/transports/whatsapp/
__init__.py  api.py  client.py

$ grep -rn "^class " src/pipecat/transports/whatsapp/client.py
src/pipecat/transports/whatsapp/client.py:35:class WhatsAppClient:
```

그리고 `WhatsAppClient`가 하는 일은 남의 connection object를 import하는 것입니다:

**`src/pipecat/transports/whatsapp/client.py` L24, L70, L342**

```python
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
...
        self._ongoing_calls_map: dict[str, SmallWebRTCConnection] = {}
...
                pipecat_connection = SmallWebRTCConnection(self._ice_servers)
```

즉 WhatsApp calling은: webhook API model layer(`api.py`, `WhatsAppConnectCall`,
`WhatsAppTerminateCall`, webhook envelope를 위한 pydantic model 12개), HMAC webhook-signature 검사,
그리고 **기존의 SmallWebRTC connection과 transport**입니다. Meta의 calling 제품은 transport가 아니라
*signalling adapter*로 ship됩니다.

### 2.2 provider를 추가하는 구조적 방법 세 가지 — 그리고 이것이 Lina를 위한 decision table이다

여기서 멈추고 일반화하십시오. 이것이 §11에서 필요할 framework-extension shape이기 때문입니다.

| Pattern | 무엇을 작성하는가 | tree 안의 witness | 언제 적용되는가 |
|---|---|---|---|
| **A. 새 transport class** | `BaseTransport` subclass + input/output processor + connection client | `DailyTransport` (3,065 L 파일), `LiveKitTransport`, `MOQTransport` | provider가 자기 media SDK / 자기 wire media format을 가진 경우 |
| **B. 새 serializer** | `FrameSerializer` subclass; `FastAPIWebsocketTransport`를 그대로 재사용 | `TwilioFrameSerializer` (314 L), `TelnyxFrameSerializer` (292 L) | provider가 media를 자기 JSON/binary envelope로 **WebSocket 위에** 흘리는 경우 |
| **C. 새 signalling client** | `SmallWebRTCConnection`을 만들어 주는 webhook/REST adapter | `WhatsAppClient` (`client.py`) | provider가 SDP offer/answer를 자기 control plane으로 하되 media는 표준 WebRTC인 경우 |

통화 audio를 WebSocket으로 당신의 endpoint에 흘려주는 한국 통신사나 CPaaS는 **pattern B**이고,
pattern B는 대략 250–300줄입니다. SIP/RTP leg를 직접 건네주는 통신사는 셋 다 아닙니다 — 이 commit의
Pipecat에는 SIP stack이 없고, 당신은 pipeline 바깥에서 SIP를 terminate한 다음 어차피 pattern B로
audio를 밀어 넣게 됩니다. 이건 조심해서 읽으십시오: Pipecat이 SIP를 어렵게 만드는 게 아니라, Pipecat이
SIP 자체에 대해 **할 말이 없는** 것입니다. transport tree의 모든 SIP 언급은 *provider의* SIP interconnect가
그 provider의 SDK를 통해 노출된 것입니다:

```bash
$ grep -ril "sip" src/pipecat/transports/
src/pipecat/transports/daily/transport.py
src/pipecat/transports/daily/utils.py
src/pipecat/transports/livekit/transport.py
```

Daily는 `DailyRoomSipParams`(`daily/utils.py:20`), `DailyRoom.sip_uri` / `sip_endpoint()`(`:117`,
`:122`), `DailySIPTransferFrame`(`daily/transport.py:117`), `on_dialin_ready(sip_endpoint)`를
기여합니다. LiveKit은 inbound `sip_dtmf_received` handler(`livekit/transport.py:246`)를 기여합니다.
파일 두 개, vendor 하나씩, SIP stack 0개. 통신사가 raw SIP leg를 건네준다면 그것을 terminate하는 건
당신 문제이고, audio가 안으로 들어오는 경로는 pattern B입니다.

---

## 3. sample rate는 setup에서 resolve되고, chunk size는 거기서 파생된다

작은 mechanism인데 폭발 반경(blast radius)이 큽니다. 그리고 §6의 8 kHz 문제가 코드에서 처음 보이는
지점이기도 합니다.

transport의 어느 쪽도 생성 시점에 sample rate를 받지 않습니다. 둘 다 `setup()`에서, `FrameProcessorSetup`
— [[ch-04/read]] §6이 startup path를 따라가며 추적한 바로 그 object — 으로부터 읽습니다.

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

그리고 fallback:

**`src/pipecat/processors/frame_processor.py` L106–107**

```python
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
```

즉 설정하지 않은 pipeline은 **입력 16 kHz, 출력 24 kHz**로 돕니다. 두 숫자가 비대칭인 것은 의도된
것입니다: 16 kHz는 streaming STT와 Silero VAD가 원하는 값이고, 24 kHz는 대부분의 neural TTS가
내보내는 값입니다.

### 3.1 chunk 산술, 숫자를 채워 넣고

**`src/pipecat/transports/base_output.py` L132–136**

```python
        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
```

실제로 만나게 될 세 rate에 대해 계산해 보면:

| Rate | `audio_bytes_10ms` | `audio_chunk_size` (×4) | chunk당 wall-clock |
|---|---|---|---|
| 24 000 Hz (default out) | 240 × 1 × 2 = **480 B** | **1 920 B** | 40 ms |
| 16 000 Hz (STT/VAD rate) | 160 × 1 × 2 = **320 B** | **1 280 B** | 40 ms |
| 8 000 Hz (PSTN wire rate) | 80 × 1 × 2 = **160 B** | **640 B** | 40 ms |

wall-clock 열은 구성상 상수입니다 — parameter를 byte가 아니라 10 ms 단위로 표현한 것의 요점이 바로
그것입니다. comment가 그것이 존재하는 이유를 말해 줍니다: *"This will help with interruption
handling."* [[ch-04/read]] §5.1이 일반 법칙을 주었고, 이것이 그 구체적 사례입니다.
`audio_out_10ms_chunks = 4`이면 output transport는 40 ms 단위로 playout을 포기할 수 있습니다. 1로
설정하면 send-loop overhead가 4배가 되는 대신 10 ms granularity를 얻습니다. [[ch-08/read]]가 이것을
소비합니다.

이 표가 말하지 *않는* 것에 주목하십시오: 여기 어디에도 8 kHz를 16 kHz로 변환하는 것은 없습니다.
transport의 sample rate는 *pipeline의* sample rate입니다. 통신사의 8 kHz에서 pipeline의 16 kHz로 가는
것은 다른 누군가의 일이고, §6.4가 그게 정확히 누구인지 보여줍니다.

> 💡 **쉬운 설명 — 왜 "40 ms"가 세 rate에서 모두 같은가?**
> `audio_bytes_10ms`는 정의상 "10 ms 분량의 byte 수"입니다. rate가 낮으면 그 byte 수도 같이 줍니다
> (8 kHz에서 160 B, 24 kHz에서 480 B). 그것을 항상 4배 하니 항상 "40 ms 분량"이 됩니다. 즉
> `audio_out_10ms_chunks`는 **시간 단위 parameter**이지 크기 단위 parameter가 아닙니다. 이게 왜
> 좋은 설계냐면: interruption granularity를 "40 ms"라고 한 번 정하면 8 kHz 전화든 24 kHz WebRTC든
> 같은 barge-in 반응성을 얻기 때문입니다. byte로 설정했다면 rate마다 다시 튜닝해야 했을 겁니다.

---

## 4. VAD는 이제 transport에 붙어 있지 않다

§5–§8의 다른 무엇보다 먼저, 낡은 mental model을 치워야 합니다. 이것이 몇 release 이전의 Pipecat voice
tutorial을 전부 오독하게 만들 가능성이 가장 큰 단일 요인이기 때문입니다.

**`TransportParams`에는 `vad_analyzer` field가 없습니다.** §1.1의 L66–93 block을 다시 보십시오. 없습니다.
제 인용에서 빠뜨린 게 아니라, 제거된 것입니다:

**`CHANGELOG.md` L4402–4406**

```
- ⚠️ Removed `vad_analyzer` and `turn_analyzer` parameters from
  `TransportParams` and all transport input classes, along with all deprecated
  VAD/turn analysis logic in `BaseInputTransport`. VAD and turn detection are
  now handled entirely by `LLMUserAggregator`.
  (PR [#4229](https://github.com/pipecat-ai/pipecat/pull/4229))
```

grep으로 확인한, 살아 있는 mount point:

```bash
$ grep -rn "vad_analyzer" src/pipecat/ | grep -v "import\|docstring\|:.*#"
src/pipecat/processors/aggregators/llm_response_universal.py:175:    vad_analyzer: VADAnalyzer | None = None
src/pipecat/processors/audio/vad_processor.py:44:        vad_analyzer: VADAnalyzer,
src/pipecat/audio/vad/vad_controller.py:72:        vad_analyzer: VADAnalyzer,
```

- `LLMUserAggregatorParams.vad_analyzer` — `llm_response_universal.py:175`
- `VADProcessor(vad_analyzer=...)` — `processors/audio/vad_processor.py:41`
- `VADController` — `audio/vad/vad_controller.py:31`, 둘 다 사용하는 공유 hysteresis machine

canonical wiring은 이제 aggregator에 있습니다:

```python
context_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
)
```

transport *안*에 있는 VAD 코드는 정확히 하나이고, 그것은 VAD의 소비자가 아니라 `VADAnalyzer`
implementation입니다 — `daily/transport.py:203`의 `WebRTCVADAnalyzer(VADAnalyzer)`로, Daily의 native
VAD를 감쌉니다. transport 안의 어떤 것도 그것을 호출하지 않습니다. 다른 analyzer와 똑같이 aggregator에
넘기는 것입니다.

**VAD가 [[ch-06/read]]의 주제인데도 왜 이게 transport chapter에 있는가:** boson의 port 위치를
옮기기 때문입니다. [[boson-gateway-server]]는 boson의 endpointing이 server에 살고 있다고 기록합니다 —
`_start_silence_timer`(`websocket.py:616`)가 `silence_timeout_ms / 1000`(default 2000)만큼 자고 나서
`_finalize_partial`(`:661`)을 호출합니다. Pipecat에서 그 logic은 transport layer에 있을 자리가 아예
없습니다. 그것은 `src/pipecat/turns/` 아래에 속합니다. 그 port를 `TransportParams`를 겨냥해서 계획하지
마십시오. [[ch-06/read]]가 그것을 소유합니다.

---

## 5. WebSocket transport: socket 하나, 조작된 clock 하나, serializer slot 하나

module 세 개이고, `BaseTransport` 말고는 공유하는 base가 없습니다:

```bash
$ wc -l src/pipecat/transports/websocket/*.py
       0 __init__.py
     559 client.py
     707 fastapi.py
     716 server.py
    1982 total
```

`__init__.py`는 비어 있습니다 — 구체 module에서 import하십시오. `client.py`는 Pipecat이 남의
WebSocket으로 *걸어 나가는* 것입니다. `server.py`는 dev server입니다. `fastapi.py`가 production path이고
Lina에게 유일하게 중요한 것입니다.

[[transport-websocket]]은 이 design을 한 문장으로 진술하고 tree가 그것을 뒷받침합니다: transport는
*socket, audio clock, connect/disconnect event*를 소유하고 — **wire 위의 모든 byte**를 pluggable한
`FrameSerializer`에 위임합니다. 책임 세 개, 위임 하나. §5.3과 §5.4가 그 셋이고, §6이 그 위임입니다.

### 5.1 `FastAPIWebsocketTransport`는 이미 accept된 socket을 받는다

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

역할 분담에 대해 알아야 할 모든 것이 저 signature와 저 docstring에 있습니다. transport는 **당신이 이미
accept한** `WebSocket`을 받습니다. route는 당신 것입니다. handshake도 당신 것입니다. authentication도
당신 것입니다. 생성자가 raise할 때 socket을 닫는 것도 당신 것입니다. Pipecat의 기여는
`await websocket.accept()` 다음 줄부터 시작합니다.

ship되는 유일한 security control은 Origin allowlist이고, 12줄입니다:

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

이 *policy*를 (품질이 아니라 — §0과 scope box를 보십시오) [[boson-gateway-server]]가 `access.py`에
대해 기록한 것과 비교하십시오:

```python
# packages/gateway/gateway/server/access.py — shape as recorded in [[boson-gateway-server]]
# (boson-agent, private; excerpt-attested)
accepted_origins = (None, *allowed_origins)
```

boson의 tuple은 **없는** Origin을 의도적으로 허용하고 그 경우를 native client용으로 예약하며, browser는
allowlist에 일치할 것을 요구합니다. Pipecat의 `is_origin_allowed`는 allowlist가 비어 있지 않은 한 없는
Origin을 거부합니다. 같은 header에 대한 두 개의 다른 결정이고, boson의 native client를 눈치채지 못한 채
Pipecat의 검사 뒤에 놓으면 그들은 접속을 멈춥니다. 이 migration이 무는 곳은 이 정도의 디테일입니다.

> 💡 **쉬운 설명 — Origin이 "없는" 경우가 왜 생기나요?**
> browser는 WebSocket handshake에 `Origin` header를 강제로 붙입니다 (CSRF 방어용). 하지만 native
> app이나 서버-대-서버 client는 붙일 의무가 없어서 header 자체가 없습니다. 그래서 정책이 갈립니다:
> boson은 "Origin이 없다 = browser가 아니다 = native client다 → 통과"이고, Pipecat은 "allowlist를
> 켰으면 없는 것도 불허"입니다. Pipecat 쪽이 더 엄격하지만, boson 쪽은 native client 지원이라는
> 요구사항이 있었던 겁니다. 어느 쪽이 옳다가 아니라, **당신의 client 구성이 무엇이냐**의 문제입니다.

### 5.2 정확히 여섯 개의 추가 field

**`src/pipecat/transports/websocket/fastapi.py` L83–88**

```python
    add_wav_header: bool = False
    serializer: FrameSerializer | None = None
    session_timeout: int | None = None
    fixed_audio_packet_size: int | None = None
    allowed_origins: list[str] = Field(default_factory=default_allowed_origins)
    ws_close_timeout: float = _WS_CLOSE_TIMEOUT_DEFAULT
```

L56에 `_WS_CLOSE_TIMEOUT_DEFAULT = 0.5`가 있습니다. `TransportParams`의 ~27개 위에 field 6개.
그것이 production WebSocket transport의 configuration surface 전부입니다.

`session_timeout`은 한 문장짜리 신비 해체가 필요합니다. 이름이 코드가 하는 것보다 많은 것을 약속하기
때문입니다:

**`src/pipecat/transports/websocket/fastapi.py` L398–401**

```python
    async def _monitor_websocket(self, timeout: int):
        """Wait for ``timeout`` seconds, then trigger the client-timeout event if still open."""
        await asyncio.sleep(timeout)
        await self._client.trigger_client_timeout()
```

callback 하나를 한 번 발화시키는 맨 sleep입니다. idle timer가 아닙니다 — activity에 reset되지 않고,
무엇을 닫지도 않습니다. boson의 `DEFAULT_SESSION_TTL_SECONDS = 1800.0` idle-TTL 동작
([[boson-gateway-server]])을 원한다면, 이건 그게 아닙니다.

### 5.3 serializer가 설정되지 않으면 모든 inbound message가 조용히 버려진다

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

그리고 outbound 쪽 거울:

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

`continue`. `return False`. log 없음, raise 없음, metric 없음. `serializer=None`인
`FastAPIWebsocketTransport`는 완전히 정상 동작하는 object로서 연결되고, `on_client_connected`를 발화하고,
receive loop를 영원히 돌리며, 양방향으로 byte 0개를 옮깁니다. 이것은 [[ch-01/read]] §7.3이
`process_frame`에서 추적하고 [[ch-03/read]] §2.3이 unhandled-frame path에서 추적한 것과 같은
silent-drop 자세입니다 — 그것은 house style이고, 이제 wire boundary에도 있습니다.

inbound loop의 세 routing branch는 이름을 붙여 둘 가치가 있습니다. 당신이 이미 아는 세 가지 서로 다른
mechanism에 대응하기 때문입니다:

| deserialize된 frame이 | 어디로 가는가 | 왜 |
|---|---|---|
| `InputAudioRawFrame` | `push_audio_frame(frame)` | audio-in path로 (filter, passthrough) |
| `InputTransportMessageFrame` | `broadcast_frame(...)` | **양방향 동시에** |
| 그 외 | `push_frame(frame)` | 평범한 downstream push |

`broadcast_frame`을 봐야 합니다. transport message는 media event가 아니고, transport 양쪽의 processor
모두에게 보여야 하기 때문입니다:

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

instance 두 개를 `broadcast_sibling_id`로 상호 참조시키고, 각 방향으로 하나씩. 포팅된 `ControlEvent`가
타고 갈 mechanism이 이것입니다 — §11.2를 위해 붙들어 두십시오.

> 💡 **쉬운 설명 — 왜 frame을 하나 만들어 두 번 push하지 않고, 두 개를 만드나요?**
> `Frame`은 `id`를 가지고, observer / metric / dedup 로직이 그 `id`로 frame을 식별합니다.
> 같은 object를 양쪽으로 밀면 "같은 frame이 두 번 흘렀다"로 보여서 [[ch-01/read]] §5.4의
> `ParallelPipeline` first-arrival dedup 같은 mechanism이 오작동합니다. 그래서 별개 instance 두 개를
> 만들되, `broadcast_sibling_id`로 서로를 가리켜 두어 "이 둘은 같은 사건의 두 면"이라는 정보를
> 잃지 않게 합니다.

### 5.4 WebSocket에는 playback clock이 없으므로, transport가 하나를 조작해 낸다

이것이 이 module 전체에서 가장 비자명한 부분이고, design doc이 아니라 comment로 문서화되어 있습니다.

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

("to quickly"라는 오타는 source에 그대로 있고, 파일 네 개에 등장합니다 — `websocket/fastapi.py:434`,
`websocket/server.py:349`, `websocket/client.py:348`, `services/heygen/client.py:174`. copy-paste이고,
그것은 Pipecat이 device가 아니라 socket에 media를 쓸 때마다 이 문제가 재발한다는 것을 말해 줍니다.)

무엇이 해결되고 있는지 생각해 보십시오. sound card나 RTP track은 *시간*으로 back-pressure를 겁니다:
20 ms마다 20 ms 분량의 audio를 소비하고, 당신은 그것을 앞지를 수 없습니다. TCP socket은 *buffer*로
back-pressure를 거는데, 수백 KB짜리 TTS에 대해서는 사실상 back-pressure가 없는 것과 같습니다. clock이
없으면 TTS가 문장 하나를 완성하는 순간 transport가 그 전부를 한 burst로 wire에 밀어 넣고, 그러면 그
audio는 당신이 아니라 **통신사가** 소유하게 됩니다. barge-in이 불가능해집니다: 생성을 멈출 수는 있지만,
고객은 이미 보낸 문장을 계속 듣게 됩니다.

그래서 transport는 block합니다:

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

**공식을 공식으로 읽기 전에 산술을 먼저 해 보십시오.** default(`audio_out_sample_rate = 24000`,
`audio_out_10ms_chunks = 4`)에서:

- `audio_chunk_size` = 1 920 bytes (§3.1)
- `audio_chunk_size / sample_rate` = 1920 / 24000 = **0.08**

그 0.08은 초가 *아닙니다*. `audio_chunk_size`는 byte이고 `sample_rate`는 samples/second이므로, 이
나눗셈이 차원적으로 말이 되는 것은 오직 16-bit mono가 sample당 2 byte를 뜻하기 때문입니다 — 진짜
duration은 1920 bytes ÷ 2 = 960 samples ÷ 24000 = **40 ms**입니다. 이 식은 80 ms를 내놓습니다. 즉 실제
duration의 정확히 두 배이고, `/ 2`는 빠진 bytes-per-sample 항이 도입한 인수 2를 상쇄합니다.

그래서 `_send_interval`은 40 ms chunk에 대해 40 ms이고, 그게 당신이 원하는 값입니다. 그리고 `/ 2`는
jitter 여유를 위한 의도적인 2배 under-sleep이 *아닙니다*. 그것은 상수로 쓰인 bytes→samples 변환입니다.
행동으로 옮길 수 있는 결과 두 가지:

1. **16-bit mono에서만 옳습니다.** `audio_out_channels = 2`로 설정하면 `audio_chunk_size`가 두 배가
   되고(§3.1의 공식이 channel 수를 곱합니다) 나눗수는 그대로이므로, `_send_interval`은 80 ms로 두 배가
   됩니다 — 실제로는 여전히 40 ms 분량인 stereo audio를 보내 놓고 두 배로 자는 것입니다. Lina는
   mono입니다. 이 경로에서는 mono를 유지하십시오.
2. 이 줄은 `server.py:379`에 *동일하게* 있으므로, server-side WebSocket transport 두 개가 이 동작을
   공유합니다.

그리고 `_write_audio_sleep`의 branch 구조는 catch-up rule입니다: 늦었으면(`sleep_duration == 0`)
schedule을 *지금* 기준으로 rebase하고, 제때면 drift 없이 schedule을 한 interval 전진시킵니다. 이것은
realtime_voice의 wall-clock pacing과 같은 아이디어입니다:

```python
# packages/realtime_voice/realtime_voice/transport/webrtc/tracks.py L168
# (boson-agent, private; excerpt-attested via [[rtv-webrtc-transport]])
#   _pace():  target = self._started_at + self._pts / self.sample_rate
```

boson은 절대 origin에 sample counter를 더해 target을 계산하고, Pipecat은 interval을 누적하다가 늦으면
rebase합니다. 둘 다 wall-clock pacing이고, boson 것은 구성상 drift할 수 없으며 Pipecat 것은 뒤처질
때마다 rebase합니다. 차이를 그렇게 진술하고 거기서 멈추십시오 — latency 결과는 [[ch-11/read]]의
것입니다.

> 💡 **쉬운 설명 — 두 pacing 방식의 차이를 숫자로**
> boson 방식: `target = 시작시각 + 보낸_샘플수 / rate`. 100번째 chunk의 target은 시작 후 정확히
> 4.000초입니다. 중간에 한 번 늦어도 다음 target은 여전히 4.000초라서 자동으로 따라잡습니다 —
> **drift 불가능**.
> Pipecat 방식: `next += interval`을 누적하되, 이미 늦었으면 `next = now + interval`로 **다시 기준을
> 잡습니다**. 한 번 크게 늦으면 그만큼의 지연이 영구히 schedule에 남습니다(따라잡지 않습니다).
> 대신 event loop가 밀렸을 때 "밀린 만큼 몰아서 보내기"를 하지 않아 burst가 생기지 않습니다.
> 통화 하나 안에서 몇 초 정도면 차이는 무시할 만하지만, 어느 쪽인지 아는 것과 모르는 것은
> [[ch-11/read]]에서 latency를 계상할 때 완전히 다릅니다.

마지막으로, interruption이 clock을 reset합니다:

**`src/pipecat/transports/websocket/fastapi.py` L508–514**

```python
        if isinstance(frame, InterruptionFrame):
            # Drop any partially buffered audio to avoid replaying stale PCM
            if self._params.fixed_audio_packet_size:
                self._audio_send_buffer.clear()

            await self._write_frame(frame)
            self._next_send_time = 0
```

여섯 줄에 세 가지가 있고, 셋 다 [[ch-08/read]]에 중요합니다: local packetization 나머지를 버리고,
**`InterruptionFrame`을 serializer로 통과시키고**(그것이 통신사에게 알려지는 경로입니다 — §6.8),
clock을 0으로 만들어 다음 audio chunk가 낡은 schedule을 기다리지 않고 즉시 나가게 합니다.

### 5.5 `fixed_audio_packet_size`와 그것이 존재하는 이유

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

정확한 크기의 media frame을 요구하는 통신사는 흔하고, 이것이 그것을 위한 hook입니다: buffer하고, 온전한
packet만 내보내고, 나머지를 보존합니다. 코드 안의 예시(640 B = 20 ms @ 16 kHz PCM16 mono)는 한국 CPaaS가
부과할 법한 제약의 전형입니다. 이 경로가 아래쪽의 `success = True` 대입보다 *먼저* `return True`한다는
점에 유의하십시오. 즉 fixed-packet 전송은 아무것도 flush되지 않았어도 항상 성공을 보고합니다 — 반환값
위에 metric을 쌓는 경우에만 문제가 되는 구분입니다.

### 5.6 `ws_close_timeout`은 telephony 때문에 존재한다

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

`FastAPIWebsocketParams` docstring이 동기가 된 사례를 명시적으로 이름 붙입니다: *"Prevents a dead or
half-closed peer (e.g. a telephony call already torn down on the provider's side) from stalling
pipeline shutdown."* 이것은 [[ch-04/read]] §8의 unbounded-`EndFrame` 문제가 socket에서 나타난
것입니다: `EndFrame`에는 timeout이 없으므로, transport의 `disconnect()`가 이미 끊은 통신사와의 close
handshake에서 영원히 block하면 graceful shutdown이 결코 완료되지 않습니다. 0.5초가 그 울타리입니다.
`_leave_counter`도 눈여겨보십시오 — input과 output이 하나의 `FastAPIWebsocketClient`를 공유하므로 close가
refcount되고, input 쪽이 끝난 뒤에도 output 쪽이 작별 인사 한 줄을 flush할 수 있습니다.

### 5.7 `SingleClientWebsocketServerTransport`는 dev 전용이고, 동시에 당신의 eval harness다

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

문서화가 아니라 강제됩니다:

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

close code 1013은 "Try Again Later"입니다. 이 class는 1.4.0에서 rename되었고, 옛 이름들
(`WebsocketServerTransport`, `WebsocketServerParams`, `WebsocketServerCallbacks`,
`WebsocketServerInputTransport`, `WebsocketServerOutputTransport`)은 모두 `server.py:643/658/673/688/703`에
`@deprecated`를 달고 2.0.0에서 제거 예정입니다. `WebsocketServerTransport`를 쓰는 tutorial을 발견했다면,
그것은 rename 이전 것입니다.

흥미로운 것은 framework 자신이 이 dev 전용 transport를 *무엇에* 쓰느냐입니다:

**`src/pipecat/evals/serializer.py` L87–93**

```python
class RTVIEvalSerializer(FrameSerializer):
    """Bridges JSON RTVI messages and pipeline frames for the eval harness.

    Use as the serializer of a ``SingleClientWebsocketServerTransport`` when running a bot
    under the eval harness. The bot pipeline must include an ``RTVIProcessor``
    and pass an ``RTVIObserver`` to the task.
    """
```

Pipecat 자신의 evaluation harness는 *serializer 하나 + single-client dev transport*입니다. Lina의 QA
harness에 직접 재사용 가능한 pattern이고, §11.3이 그것을 구체적인 수(move)로 바꿉니다.

---
## 6. Telephony: telephony transport는 존재하지 않는다

### 6.1 이것으로 결론이 나는 listing

```bash
$ ls src/pipecat/transports/
__init__.py  base_input.py  base_output.py  base_transport.py
daily/  heygen/  lemonslice/  livekit/  local/  moq/  smallwebrtc/
tavus/  vonage/  websocket/  whatsapp/

$ ls src/pipecat/serializers/
__init__.py  base_serializer.py  exotel.py  genesys.py  plivo.py
protobuf.py  telnyx.py  twilio.py  vonage.py
```

`twilio/` 없음. `telnyx/` 없음. `plivo/` 없음. `exotel/` 없음. `sip/` 없음. 이 framework의 telephony
surface 전체는 다른 디렉토리에 있는 파일 여섯 개입니다:

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

telephony serializer 여섯 개, 합계 2,185줄. 그리고 함정에 유의하십시오: `transports/vonage/`는 **실제로**
존재하지만 그것은 `VonageVideoConnectorTransport` — video — 이고 `serializers/vonage.py`와 아무 상관이
없습니다. 같은 vendor, 무관한 제품 두 개, namespace 충돌 하나.

### 6.2 framework 자신의 runner가 증거다

디렉토리 listing만 믿을 필요는 없습니다. Pipecat의 development runner가 phone bot을 만드는데, 그것이
무엇을 만드는지 보십시오:

**`src/pipecat/runner/utils.py` L486–554** (중간 생략; 네 branch는 구조적으로 동일합니다)

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

return type annotation은 `BaseTransport`라고 말합니다. return statement는 `FastAPIWebsocketTransport`라고
말합니다. **transport 하나, serializer 넷.** provider 차이는 전적으로 `params.serializer`에 어떤 object가
대입되느냐로 환원됩니다.

`create_transport()`의 docstring(`runner/utils.py:598`)은 사용자 쪽에서 같은 것을 말합니다 — key
`"twilio"`, `"telnyx"`, `"plivo"`, `"exotel"`이 전부 `FastAPIWebsocketParams` factory로 mapping되고,
comment는 `# add_wav_header and serializer will be set automatically`입니다.

provider 탐지는 첫 `start` message를 스니핑해서 이뤄집니다:

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

`# normalized from Telnyx's call_control_id`에 주목하십시오. 모든 provider가 자기 identifier 두 개를
다르게 이름 붙이고, `parse_telephony_websocket`이 그것들을 `stream_id` / `call_id`로 평탄화합니다.
왜 둘인지는 §6.7이 다룹니다. (원문의 참조 그대로입니다 — 실제 내용은 §6.8에 있습니다.)

### 6.3 `FrameSerializer` ABC는 method 네 개, 106줄이다

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

여기에 L56의 `should_ignore_frame()`과 L46의 `__init__`을 더하면 됩니다. contract 전체가 그것입니다:
`setup`, `serialize`, `deserialize`, `should_ignore_frame`. 넷 중 둘이 abstract입니다.

`setup()`은 serializer가 pipeline의 rate를 배우는 방법이고, **양쪽 끝 모두**에서 호출됩니다 —
`fastapi.py:306`의 `FastAPIWebsocketInputTransport.setup()`과 `fastapi.py:461`의
`FastAPIWebsocketOutputTransport.setup()`:

```python
        if self._params.serializer:
            await self._params.serializer.setup(setup)
```

양쪽이 *같은* serializer instance에 대해 `setup()`을 호출하는데, telephony serializer들에게는 이것이
idempotent하지만(그들은 `self._sample_rate`만 대입합니다) stateful한 것을 작성한다면 진짜 제약이
됩니다. `resampler_clear_after_secs: float | None = 0.2`는 뒤에 큰 현장 장애가 있는 작은 디테일입니다:
stream resampler는 filter history를 들고 있고, 공백을 사이에 두고 낡아 버린 history는 들리는
artefact를 만듭니다. Genesys가 docstring에서 chunk timing이 충분히 불규칙하니 clearing을 아예 끄라고
이름 붙여진 provider입니다. 한국 통신사에 대해서도 적어 두십시오: **media chunk가 불규칙하게
도착하면 `resampler_clear_after_secs=None`.**

> 💡 **쉬운 설명 — resampler에 "history"가 왜 있나요?**
> 8 kHz → 16 kHz resampling은 단순히 샘플을 복제하는 게 아니라 low-pass filter를 통과시키는
> 연산이고, FIR/IIR filter는 **직전 몇 개 샘플**을 기억해야 출력이 이어집니다. chunk 경계마다
> history를 유지해야 이음매에서 클릭 소리가 안 납니다. 그런데 통화가 3초 멈췄다가 재개되면 그
> 낡은 history는 이미 무의미한 신호의 잔향이라, 그대로 이어붙이면 오히려 artefact가 됩니다.
> `resampler_clear_after_secs=0.2`는 "0.2초 이상 조용하면 history를 버려라"입니다. 그런데
> Genesys처럼 원래 chunk가 뚝뚝 끊겨 오는 provider에서는 정상 통화 중에도 이 조건이 계속 걸려서
> 매번 history가 날아가고 — 그게 오히려 매 chunk 경계마다 artefact를 만듭니다. 그래서 `None`.

### 6.4 경계를 넘는 frame 하나를 양방향으로 추적하기

이 문단은 다시 읽어야 할 문단입니다. Lina를 아프게 할 telephony의 모든 것이 이 두 method에 보입니다.

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

그리고 decode 자체:

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

순서가 중요합니다: **companding을 먼저 decode하고, 그 다음에 resample.** `audioop.ulaw2lin(x, 2)`는
8-bit μ-law byte 하나를 *같은* rate(8 kHz)에서 16-bit linear sample로 확장하고, 그 다음에야 resampler가
8 kHz → 16 kHz를 합니다. 이제 당신은 16 kHz에서 16-bit sample을 갖게 되고, STT가 그것을 받아들입니다.

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

48줄짜리 method 하나에 책임 네 개가 있고, 이것이 붙들고 있어야 할 문장입니다:

> serializer는 동시에 **codec boundary이자, resample point이자, call-control client이자, DTMF
> decoder**입니다.

Codec: `pcm_to_ulaw` / `ulaw_to_pcm`. Resample: `__init__`(L150–155)에서 만들어지는 두 개의
`create_stream_resampler` instance. Call control: `_hang_up_call()`이 `serialize()` 안에서 Twilio의 REST
API로 HTTP request를 보냅니다. DTMF: `deserialize`의 `"dtmf"` branch. 하나의 class에 들어 있기에는 서로
무관한 관심사가 너무 많고, 무관한 *이유는* serializer가 architecture가 제공하는 유일한
provider-specific 장소이기 때문입니다. §2.2의 pattern B에는 파일이 정확히 하나 들어 있습니다.

### 6.5 provider table, 줄 단위로 검증됨

| Serializer | Wire rate (검증된 줄) | Codec | Interruption event |
|---|---|---|---|
| `TwilioFrameSerializer` (314 L) | `twilio_sample_rate: int = 8000` — twilio.py:79 | μ-law | `{"event": "clear", "streamSid": ...}` — :187 |
| `TelnyxFrameSerializer` (292 L) | `telnyx_sample_rate: int = 8000` — telnyx.py:60 | `inbound_encoding`/`outbound_encoding`, `"PCMU"` default (:62–63), `"PCMA"` 지원, 그 외 `raise ValueError(f"Unsupported encoding: ...")` (:166, :273) | `{"event": "clear"}` — :151 (id 없음) |
| `PlivoFrameSerializer` (256 L) | `plivo_sample_rate: int = 8000` — plivo.py:54 | μ-law | `{"event": "clearAudio", "streamId": ...}` — :139 |
| `ExotelFrameSerializer` (171 L) | `exotel_sample_rate: int = 8000` — exotel.py:49 | **raw PCM, resample만** — companding 없음 | `{"event": "clear", "stream_sid": ...}` — :99 |
| `GenesysAudioHookSerializer` (964 L) | `genesys_sample_rate: int = 8000` — genesys.py:148 | `AudioHookMediaFormat.PCMU = "PCMU"  # μ-law, 8kHz` — :77, 또는 L16 | (§6.10 참조) |
| `VonageFrameSerializer` (188 L) | `vonage_sample_rate: int = 16000` — vonage.py:48 | — | — |

**여섯 중 다섯이 8 kHz를 default로 합니다. Vonage만 예외입니다.** Exotel은 다른 축에서의 두 번째
예외입니다: companded codec이 아니라 linear PCM을 보내는 유일한 것이라, serialize path가
`pcm_to_ulaw` 없이 맨 `self._output_resampler.resample(...)`(exotel.py:104–106)입니다. 또한 **`auto_hang_up`이
없는** 유일한 것이기도 합니다 — `__init__`이 `call_sid`를 받긴 하고 docstring은 *"optional, not used in
this implementation"*이라고 말합니다(exotel.py:60).

### 6.6 4 kHz 천장, 그리고 그것이 한국어에 하는 일

공식보다 예제를 먼저 계산하십시오.

**구체적인 실행 하나.** 고객이 전화에 대고 있습니다라고 말합니다. 통신사의 codec이 그 목소리를 초당
8,000번 sampling하고 각 sample을 μ-law byte 하나로 encode합니다. Twilio가 그 byte 중 ~160개(20 ms)를
base64로 JSON message에 담아 당신의 socket으로 보냅니다. `ulaw_to_pcm`이 각 byte를 16-bit sample로
확장하고 8 kHz → 16 kHz로 resample해서, 160개였던 것에서 320개 sample을 만들어 냅니다. STT는 16 kHz,
16-bit `InputAudioRawFrame`을 받고 아주 만족합니다.

**이제: 그 320개 sample 안에는 무엇이 있는가?** 8,000 Hz로 sampling하면 8000 / 2 = **4,000 Hz**까지의
주파수 성분을 표현할 수 있고, 그 위의 성분은 *당신이 byte 하나 보기도 전에* 통신사 장비의 anti-alias
filter가 제거했습니다. 16 kHz로 interpolate하는 것은 160개 숫자에서 320개 숫자를 만드는 것입니다.
정보를 만들어 내지는 않습니다. 그 신호의 4–8 kHz 대역에는 아무것도 없습니다. 애초에 그것을 나를 것이
없었기 때문입니다.

공식 전체가 그것입니다: **Nyquist ceiling = wire rate / 2 = 4 kHz.** 그리고 그 위에, μ-law는 8-bit
companded입니다 — 음성에 대해 지각적으로 합리적인 곡선을 얻는 대가로 PCM16 대비 대략 2비트의 유효
dynamic range를 내주는 logarithmic quantisation입니다. 즉 신호는 대역이 제한되어 *있고* 동시에 거칠게
quantise되어 있습니다.

**왜 이것이 일반적인 audio 주의사항이 아니라 Lina에 대해 특별히 지배적인 제약인가:**

- 한국어의 마찰음과 파찰음 — ㅅ, ㅆ, ㅊ, 그리고 많은 문맥에서의 ㅎ — 은 변별 에너지의 대부분을 고역대에
  싣습니다. 사와 싸를, 또는 자와 차를 가르는 에너지는 4 kHz와 그 위에 있습니다.
- 받침(final consonant) 변별은 release burst와 formant transition에 크게 기대는데, 그 변별 단서 또한
  고역대입니다. 갔습니다 / 같습니다 / 잤습니다는 8 kHz에서보다 4 kHz에서 스펙트럼상 훨씬 가깝습니다.
- 보험 tele-sales는 정확히 이것이 해치는 token들로 굴러갑니다: 보험료, 갱신, 특약, 해지, 청약 — 그리고
  숫자와 날짜, 여기서 사 / 삼과 십 / 시 혼동은 미용상의 문제가 아니라 치명적입니다.

**강제할 수 있게 진술한 운영 규칙:** 16 kHz 마이크 audio에서 측정한 한국어 STT 정확도 수치는 이
pipeline에 **전이되지 않습니다**. "다소 낙관적"이 아니라 — 전이되지 않습니다. 통신사를 통과한 진짜
8 kHz μ-law에서 측정하거나, 그 수치를 unknown으로 기록하십시오. 통신사 접근 권한을 얻기 전에 추정해야
한다면, 정직한 proxy는 16 kHz eval set을 가져다 3.4–4 kHz에서 low-pass하고, μ-law로 encode했다가
decode하고, 16 kHz로 upsample한 뒤 다시 돌리는 것입니다. 그것은 측정이 아니라 시뮬레이션이고, packet
loss와 jitter와 통신사 자체의 gain control을 빠뜨리므로 여전히 낙관적일 것입니다. [[ch-07/read]]와
[[stt-korean-providers]]가 provider 선택 쪽에서 이것을 이어받습니다.

> 💡 **쉬운 설명 — "320개를 만들었는데 정보가 안 늘어난다"를 눈으로**
> 사진을 100×100으로 찍은 다음 200×200으로 늘리면 픽셀 수는 4배가 되지만 새로 보이는 디테일은
> 없습니다. resampling도 정확히 같습니다. 그리고 여기서 더 나쁜 점은, 통신사 장비가 4 kHz 위를
> **찍기 전에 잘라 버렸다**는 것입니다 — 원본 사진 자체가 흐린 렌즈로 찍힌 셈입니다.
> 그래서 "우리 STT는 16 kHz를 받으니 괜찮다"는 안심은 틀렸습니다. STT는 16 kHz 배열을 받지만,
> 그 배열의 절반 대역은 정의상 비어 있습니다.

### 6.7 여기서 figure를 사용하십시오

**[figures/transport-comparison.html](figures/transport-comparison.html)** — 지금 열고 곧장 telephony
panel로 가십시오. frame 하나를 serializer box를 가로질러 실행합니다(8 kHz μ-law 입력 → 16 kHz로
resample → provider `clear` event로 나가는 `InterruptionFrame`). 그리고 그것을 4 kHz 천장 위에 한국어
마찰음과 받침 단서 대역이 표시된 Nyquist band chart에 대비시킵니다. 이것으로 딱 한 가지를 하십시오:
band chart를 보고, 코드를 한 줄이라도 쓰기 **전에**, Lina의 숫자·날짜 확인 flow에 DTMF fallback이
필요한지 아니면 철자를 불러 확인하는 turn design이 필요한지 결정하십시오 — 답은 pipeline 안의
무엇에서가 아니라 음영 영역에서 *빠져 있는* 것에서 나옵니다. 나머지 세 stack(Daily/SmallWebRTC, raw
WebSocket, 당신 자신의 aiortc transport)과 counting strip은 읽는 동안 §2와 §10을 정직하게 유지하기 위해
거기 있습니다.

### 6.8 identifier 두 개, 그리고 `auto_hang_up`

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

`stream_sid`는 필수이고 `call_sid`는 optional입니다. 둘은 같은 것의 다른 이름이 아닙니다:

- **`stream_sid`**는 media stream입니다. **모든** outbound `media` message와 모든 `clear` message에
  echo됩니다(§6.4의 `serialize` 인용을 보십시오 — dict 둘 다 그것을 담고 있습니다). 이것이 없으면
  통신사가 당신의 audio를 올바른 leg로 routing할 수 없습니다.
- **`call_sid`**는 call resource이고, 정확히 한 가지에만 쓰입니다: REST call control.

provider별로: Telnyx는 `stream_id` + `call_control_id`; Plivo는 `stream_id` + `call_id`; Exotel은
`stream_sid` + `call_sid`(미사용). `parse_telephony_websocket`이 이 전부를 정규화합니다.

`auto_hang_up: bool = True` — Twilio, Telnyx, Plivo에서 **기본값이 켜짐**입니다. 그래서 생성자 validation이
load-bearing이 됩니다:

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

그리고 hang-up 자체는 `serialize()` 안에서 발행되는 HTTP call입니다:

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

`{"Status": "completed"}`로 `POST /2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json`을 보내고,
404/20404를 이미-종료로 취급합니다. Telnyx는
`POST https://api.telnyx.com/v2/calls/{call_control_id}/actions/hangup`에 대해 같은 shape을 하고,
422/90018을 이미-종료로 취급합니다.

**지금 적어 두어야 할 충돌.** `auto_hang_up`은 `EndFrame` *또는* `CancelFrame`에서 발화합니다.
[[ch-04/read]] §7과 §8은 그 둘이 Pipecat pipeline이 끝나는 방식임을 확립했습니다 — 하나는 graceful,
하나는 즉시. [[boson-gateway-server]]는 boson의 teardown path가
`_teardown_connection_sessions`(`websocket.py:381`) → `_on_disconnect(sid)` → `core.on_disconnect`
(`core.py:605`)이고, 이것이 history를 저장하고 idle-TTL clock을 시작한다고 기록합니다. boson의 teardown을
있는 그대로 포팅하면서 `auto_hang_up=True`를 그대로 두면, **독립적인 두 subsystem이 둘 다 통화를 끝내려
시도합니다.** 그리고 지는 쪽은 이미 존재하지 않는 call resource에 대해 error를 로깅합니다. 소유자를
하나 정하십시오. 그 결정에 필요한 기계적 사실은, serializer 쪽 버전이 frame serialization 안에서
동기적으로 발화한다는 것 — 즉 당신의 application-level teardown이 반응할 수 있는 어떤 것보다도 이르다는
것입니다.

`_hangup_attempted`(L156, L183에서 set)도 유의하십시오: one-shot latch라서, `EndFrame` 다음에
`CancelFrame`이 와도 hang up은 한 번만 일어납니다.

### 6.9 barge-in은 provider-specific한 flush이고, `InterruptionFrame`을 타고 간다

provider 넷, 같은 아이디어에 대한 서로 다른 envelope 넷:

| Provider | Flush message | 줄 |
|---|---|---|
| Twilio | `{"event": "clear", "streamSid": self._stream_sid}` | twilio.py:187 |
| Plivo | `{"event": "clearAudio", "streamId": self._stream_id}` | plivo.py:139 |
| Telnyx | `{"event": "clear"}` | telnyx.py:151 |
| Exotel | `{"event": "clear", "stream_sid": self._stream_sid}` | exotel.py:99 |

이것이 왜 존재하는가는 [[ch-04/read]] §5.1의 산술을 지금까지 적용해 온 것보다 한 hop 더 바깥에 적용한
것입니다. 당신의 pipeline은 자기 output queue를 즉시 flush할 수 있습니다. 하지만 이미 통신사에게 건넨
audio는 **통신사의** playout buffer에 앉아 있고, 고객은 아직 그것을 듣고 있습니다. LLM과 귀 사이에는
buffer가 셋 있고 — processor queue, transport 자신의 packetization buffer, 통신사의 playout buffer —
interruption은 셋 전부를 비워야 합니다. §5.4의 `process_frame`이 두 번째를 비우고 frame을 serializer로
전달합니다. serializer의 `clear` message가 세 번째를 비웁니다.

그리고 여기 boson 충돌이 있습니다. [[boson-gateway-server]]와 [[boson-interrupt-subsystem]]으로부터:
boson의 `gateway/interrupt/`는 현재 **partial text transcript**를 추론해서 barge-in 여부를 결정합니다.
그 결정 logic은 이것과 직교합니다 — 그대로 유지해도 됩니다 — 하지만 *효과(effect)* 경로는 완전히
바뀝니다: 무엇이 interrupt를 결정하든, 결국 transport output에 도달하는 `InterruptionFrame`을 emit해야
합니다. 그것만이 통신사 `clear` event를 만들어 내기 때문입니다. LLM만 멈추는 interruption은 고객이
문장의 나머지를 계속 듣게 만듭니다. [[ch-08/read]]가 그 cascade 전체를 해부합니다. 이 section은 그
사슬의 마지막 고리가 serializer에 산다는 것을 당신이 알게 하려고 여기 있습니다.

### 6.10 DTMF는 비대칭이고, outbound 절반은 workaround다

**Inbound는 어디서나 동작합니다.** §6.4에서 인용한 `"dtmf"` branch는
`InputDTMFFrame(KeypadEntry(digit))`을 반환하고 인식되지 않는 digit에 대해서는 `ValueError`를 삼킵니다.
keypad vocabulary는 열둘로 닫혀 있습니다:

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

0–9, `#`, `*`. A/B/C/D는 없습니다(네 번째 DTMF column으로, 군용과 legacy trunk signalling 바깥에서는
거의 아무도 쓰지 않습니다).

**Outbound가 구멍입니다.** 어떤 telephony serializer도 `OutputDTMFFrame`을 전혀 처리하지 않습니다:

```bash
$ grep -rn "OutputDTMFFrame" src/pipecat/serializers/
$ echo $?
1
```

hit 0개. 그래서 transport가 fallback합니다:

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

`load_dtmf_audio`는 ship되는 WAV 파일 열두 개 중 하나를 읽습니다:

```bash
$ ls src/pipecat/audio/dtmf/
__init__.py  dtmf-0.wav  dtmf-1.wav  dtmf-2.wav  dtmf-3.wav  dtmf-4.wav
dtmf-5.wav  dtmf-6.wav  dtmf-7.wav  dtmf-8.wav  dtmf-9.wav  dtmf-pound.wav
dtmf-star.wav  types.py  utils.py
```

그러니까 WebSocket telephony 통화에서 outbound 키 입력은 **`write_audio_frame`을 통해 평범한 음성
audio로 밀려 나가는 합성 tone WAV**입니다 — 즉 그것은 §6.4의 `pcm_to_ulaw`를 거쳐 음성처럼 8 kHz μ-law로
companding되고, 반대편에 소리로 도착합니다. 반대편 IVR이 그것을 듣느냐는 그 IVR의 DTMF detector, 통신사의
transcoding, 그리고 중간의 voice-optimised 처리가 dual-tone을 망가뜨리는지에 달려 있습니다. out-of-band
signalling 경로는 없습니다.

**⚠️ CORRECTION 2.** [[transport-telephony]]는 `_supports_native_dtmf()`가 "Daily transport에서만
override된다"고 진술합니다. tree는 동의하지 않습니다 — override는 **두 개**입니다:

```bash
$ grep -rn "_supports_native_dtmf" src/pipecat/
src/pipecat/transports/base_output.py:270:        if self._supports_native_dtmf():
src/pipecat/transports/base_output.py:287:    def _supports_native_dtmf(self) -> bool:
src/pipecat/transports/daily/transport.py:2246:    def _supports_native_dtmf(self) -> bool:
src/pipecat/transports/livekit/transport.py:1018:    def _supports_native_dtmf(self) -> bool:
```

LiveKit도 구현하고 있고, 알아 둘 가치가 있는 문서화된 한계가 딸려 있습니다:

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

Daily 쪽이 더 풍부합니다 — frame이 `DailyOutputDTMFFrame`일 때 `session_id`, `digit_duration_ms`,
`method`를 전달합니다(`daily/transport.py:2254–2276`). 수정된 진술: **native outbound DTMF는 정확히 두
transport에 존재하고, 둘 다 WebRTC/SFU transport이며, WebSocket + telephony-serializer 경로에는 없습니다.**

**왜 신경 써야 하는가.** 한국 보험 tele-sales는 ARS/IVR 트리를 끊임없이 건드립니다 — 상담원 큐로 전환,
통신사 인증 메뉴 탐색, 숫자를 말하는 대신 키 입력으로 확인(§6.6에 따르면 정확히 4 kHz 천장이 가장 크게
망가뜨리는 입력입니다). Lina가 키를 *듣는* 게 아니라 *누를* 필요가 생긴다면, WebSocket telephony
경로에서 그것은 음성 대역으로의 tone injection이고, 그것에 의존하는 flow를 설계하기 전에 통신사에 대고
검증해야 합니다.

### 6.11 Genesys는 serializer가 얼마나 커질 수 있는지 보여주는 outlier다

964줄 — Twilio의 세 배 — 이고, bloat가 아닙니다. `GenesysAudioHookSerializer`는 stateful하고,
sequence number가 붙고, 재개 가능한 session protocol을 *serializer 안에서* 구현합니다:

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

저 목록을 §6.3의 네-method ABC에 대고 읽으십시오. `FrameSerializer`의 어떤 것도 serializer가 event
handler 일곱 개, sequence counter 둘, pause state, ISO-8601 stream position을 가질 수 있다고 시사하지
않습니다 — 하지만 `FrameSerializer`는 `BaseObject`를 확장하고, `BaseObject`가 event machinery를
제공하며, 아무것도 당신을 막지 않습니다. `_format_position` / `_parse_position`(L274–300)은
`timedelta`를 ISO 8601 duration(`"PT1.5S"`)으로 serialize합니다. Genesys의 AudioHook protocol이 대부분의
message에 stream position을 요구하기 때문입니다.

여기서 가져갈 것 두 가지:

1. **serializer seam에는 크기 상한이 없습니다.** 한국 통신사의 protocol에 handshake, heartbeat, 재개,
   message별 sequence number가 있다면 그 전부가 serializer에 들어가고, Genesys가 그 shape에 대한 당신의
   reference implementation입니다.
2. **Pipecat에서 sequence discipline은 serializer에 삽니다.** data-channel abstraction이 아닙니다. 그것이
   정확히 §11.2가 boson의 `ControlEvent`에 대한 port plan으로 바꾸는 관찰입니다.

---
## 7. serializer는 open sum type이 닫히는 곳이다

이제 이론적 보상이고, [[ch-02/read]]가 이 chapter의 dependency인 이유입니다.

[[ch-02/read]]와 [[ch-03/read]] §2.2는 waist를 세었습니다: `src/pipecat/frames/frames.py`에 선언된
class 133개, 그중 131개가 `Frame`으로 끝나고, **123개가 `Frame`의 transitive descendant**입니다.
자라기만 하는 open sum type입니다.

wire protocol은 open일 수 없습니다. socket을 건너는 것은 양쪽 끝이 컴파일 시점에 합의한 고정된
vocabulary여야 합니다. 그래서 어딘가에서 open sum type이 closed one 위로 projection되어야 하고,
Pipecat에서 그 어딘가는 serializer입니다. 그 projection이 얼마나 잔인한지 보십시오:

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

**다섯 개.** pipeline에는 frame type 123개, protobuf wire 위에는 다섯 개. 그리고 나머지 전부에 대한
처리는:

**`src/pipecat/serializers/protobuf.py` L97–100**

```python
        proto_frame = frame_protos.Frame()  # type: ignore[attr-defined]
        if type(serializable) not in self.SERIALIZABLE_TYPES:
            logger.warning(f"Frame type {type(serializable)} is not serializable")
            return None
```

`logger.warning` 하나와 `None` 하나. §6.4에서 인용한 Twilio의 같은 결정은 더 조용합니다 — log 없는
`# Return None for unhandled frames`이고, 그 `serialize`는 정확히 네 경우
(`EndFrame`/`CancelFrame`, `InterruptionFrame`, `AudioRawFrame`, `OutputTransportMessage*`)를 인식하고
`deserialize`는 두 경우(`media`, `dtmf`)를 인식합니다.

`type(serializable) not in self.SERIALIZABLE_TYPES`도 유의하십시오 — `isinstance`가 아니라 **정확한 type
검사**입니다. `TextFrame`의 subclass는 `ProtobufFrameSerializer`가 serialize하지 않습니다. closed union을
가장 엄격하게 강제한 형태입니다.

> 💡 **쉬운 설명 — 왜 `isinstance`가 아니라 `type(...) in ...`인가요?**
> `isinstance`를 쓰면 `TextFrame`을 상속한 `KoreanHonorificTextFrame` 같은 것이 조용히 통과해서
> `"text"`로 인코딩되고, subclass가 들고 있던 추가 field는 wire에서 **소리 없이 사라집니다.**
> exact-type 검사는 그런 조용한 정보 손실 대신 "이건 serialize 못 한다"는 명시적 실패를 택한
> 겁니다. 대가는: `Frame`을 상속해서 만든 당신의 커스텀 frame은 protobuf serializer의 dict에
> 직접 등록하지 않는 한 절대 wire를 건너지 못한다는 것.

**그리고 realtime_voice는 동일한 위치에서 동일한 세금을 냅니다.** [[rtv-webrtc-transport]]로부터,
`WebRTCVoiceTransport._control_event()`(`transport.py` L118)는 boson의 closed union으로부터 wire type
문자열로 가는 손으로 작성된 mapping입니다:

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

같은 shape. architecture 안의 같은 위치. **채점하지 않고 정확히 이름 붙일 가치가 있는 차이 하나:**
unknown case입니다. Pipecat은 `None`을 반환합니다(드롭, 때때로 warning과 함께). realtime_voice는
`TypeError`를 raise합니다. 이 둘은 [[ch-01/read]] §7과 [[ch-03/read]] §2.3이 확립한 같은 design 축의 양
끝입니다 — Pipecat의 transparency-by-default 대 closed union의 exhaustiveness. 여기서 세 번째로
등장하는데, 그것이 이것이 우연한 style 선택이 아니라 이 비교 전체의 진짜 축임을 알려 주는 방식입니다.

이제 양 끝을 다 봤으니, narrow-waist 비용에 대한 정직한 진술은 이렇습니다:

> narrow waist는 오직 pipeline *안에서만* narrow합니다. 모든 wire boundary에서 누군가는 open sum type을
> 고정된 protocol vocabulary 위로 mapping하는 명시적이고, 닫혀 있고, 손으로 유지되는 매핑을 작성합니다.
> Pipecat은 provider마다 하나씩 작성합니다(파일 7개, 2,456줄). realtime_voice는 한 번에 하나를
> 작성합니다(`_control_event`). 둘 다 손으로 유지되며, 비용은 frame type의 개수가 아니라 protocol의
> 개수에 비례합니다.

**⚠️ CORRECTION 3.** outline은 "정확히 7개의 concrete `FrameSerializer` implementation"이 있다고
말합니다. `src/pipecat/serializers/`에 대해서는 맞고, repo 전체에 대해서는 틀렸습니다:

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

`src/pipecat/` 아래 **여덟 개**, `serializers/` 아래 일곱 개. 여덟 번째가
`RTVIEvalSerializer`(`evals/serializer.py`, 254 L)입니다. 아홉 번째는 test suite에 있습니다 —
`tests/test_websocket_server_transport.py:49:class _RawAudioSerializer(FrameSerializer)`. count는 항상
scope를 붙여서 진술하십시오. 여덟 번째가 §11.3을 당신에게 주는 것이기 때문입니다.

---

## 8. WebRTC: `SmallWebRTCTransport`와 `DailyTransport`

### 8.1 SmallWebRTC — 당신이 이미 쓰고 있는 그 aiortc

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

두 가지가 눈에 띕니다.

**subclass가 아니라 맨 `TransportParams`를 받습니다.** Daily에는 `DailyParams`가, LiveKit에는 자기
것이, WebSocket transport에는 `FastAPIWebsocketParams`가 있습니다 — SmallWebRTC에는 provider-specific하게
설정할 것이 없습니다. provider-specific한 모든 것이 당신이 건네주는 `SmallWebRTCConnection` 안에 있기
때문입니다. §2.2의 pattern A가 가장 깨끗하게 나타난 모습입니다.

**⚠️ CORRECTION 4 (excerpt가 아니라 tree 안의 문서 버그).** class docstring이 존재하지 않는 event를
광고합니다:

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

hit 하나, 그리고 그것은 docstring 줄 자신입니다. data-channel message에 대해 등록된 handler는
`on_app_message`입니다. `@transport.event_handler("on_client_message")`는 동작하지 않습니다 —
`BaseObject`는 등록된 handler만 허용합니다. docstring이 아니라 세 개의 `_register_event_handler`
호출을 믿으십시오.

**Media clocking**이 두 aiortc 구현이 측정 가능하게 갈라지는 곳입니다:

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

[[rtv-webrtc-transport]]가 기록한 당신 자신의 `OutboundAudioTrack`(`tracks.py` L111)과 대비하면:

| | Pipecat `RawAudioTrack` | realtime_voice `OutboundAudioTrack` |
|---|---|---|
| Granularity | **10 ms** (`_samples_per_10ms`) | **20 ms** (`packet_duration_ms = 20`, 48 kHz에서 960 samples) |
| Rate | pipeline `sample_rate` | `output_sample_rate = 48_000` |
| 정렬되지 않은 입력 | `raise ValueError("Audio bytes must be a multiple of 10ms size.")` | `av.AudioFifo`가 나머지를 `recv()` 사이에 유지 |
| Write completion | 소비되면 resolve되는 `asyncio.Future` 반환 | 없음 |
| Underflow | `auto_silence: bool = True`가 무음을 내보냄 | `packet_seconds`만큼 대기 |
| Generation 변경 | — (upstream에서 처리) | **FIFO를 버림** (L142–146, L152–154) |
| Silence buffer | — | 명시적 `plane.update(b"\x00" * plane.buffer_size)` — *"PyAV does not guarantee zero-initialized AudioFrame storage"*이기 때문 |

둘 다 같은 아이디어에 대한 올바른 구현이며, 정렬 계약(alignment contract)이 다르고 interruption flush를
놓을 자리가 다릅니다. write-completion `Future`와 FIFO-discard가 반대편에 counterpart가 없는 두
mechanism입니다. 둘 다 기록하고 넘어가십시오 — 그것들이 중요해지는 곳은 [[ch-08/read]]입니다.

> 💡 **쉬운 설명 — 10 ms vs 20 ms 정렬 계약이 실무에서 뭘 바꾸나요?**
> Pipecat 쪽은 "10 ms 배수가 아닌 byte를 주면 예외를 던진다"입니다 — 계약 위반을 **즉시, 시끄럽게**
> 알려 주지만, 호출자가 항상 정렬을 맞춰야 합니다. boson 쪽은 `AudioFifo`가 나머지를 들고 있다가
> 다음 호출에 이어 붙입니다 — 호출자가 편하지만, 나머지가 buffer에 남아 있다는 뜻이라
> **interruption 시점에 그 나머지를 명시적으로 버려야** 합니다(그래서 generation이 바뀌면 FIFO를
> 통째로 버리는 코드가 있는 겁니다). 즉 "어디서 잔여 audio를 flush하는가"라는 같은 문제를 한쪽은
> 호출자에게, 한쪽은 track에게 밀어 놓은 것입니다.

**Connection recovery.** [[ch-03/read]] §6.4가 이미 이것을 주었으니 한 줄로: `pc_id`(`connection.py:302`),
`renegotiate(sdp, type, restart_pc=False)`(`:443`), `ask_to_renegotiate()`(`:799`), 그리고
*"aiortc does not provide any way so we can be aware when we are disconnected"*(`:350`)이기 때문에 손으로
만든 disconnect detector. realtime_voice의 유일한 recovery 경로는 새로운 `accept_offer(reconnect=True)`
입니다.

그리고 `SmallWebRTCConnection`이 받는 것:

**`src/pipecat/transports/smallwebrtc/connection.py` L245–249**

```python
    def __init__(
        self,
        ice_servers: list[str] | list[IceServer] | None = None,
        connection_timeout_secs: int = 60,
    ):
```

argument 두 개. token 없음, TTL 없음, customer binding 없음 — §10이 이것으로 돌아옵니다.

### 8.2 Daily — SFU 경로, 그리고 Pipecat이 PSTN을 *실제로* ship하는 유일한 곳

**`src/pipecat/transports/daily/transport.py` L2279**가 `DailyTransport(BaseTransport)`이고, 그 params
class가 provider surface를 추가합니다:

**`src/pipecat/transports/daily/transport.py` L339–346 (선택된 field)**

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

**이것은 정확하게 진술하십시오. 과장하거나 축소하기 쉬운 지점이기 때문입니다.** §6.1의 발견은 Pipecat에
telephony *transport*가 없다는 것이고, 그것은 여전히 정확히 참입니다. 하지만 Daily의 SFU는 자기 SIP
interconnect를 가지고 있고 `DailyTransport`가 그것을 노출합니다 — `dialin_settings`,
`on_dialin_ready(sip_endpoint)` callback(`daily/transport.py:1537`), `on_dialout_answered`, 그리고 webhook
body에서 dial-in 설정을 병합하는 runner helper `_maybe_apply_daily_dialin`(`runner/utils.py`). 그래서
전화 통화를 Pipecat pipeline에 넣는 방법은 **두 가지**입니다: (a) WebSocket + provider serializer — 여기서
당신은 8 kHz μ-law를 보고 codec boundary를 소유합니다. (b) Daily dial-in — 여기서 Daily가 SIP를
terminate하고 당신은 평범한 WebRTC media를 받습니다. 경로 (b)가 §6.6을 사라지게 하지는 않습니다 —
audio는 여전히 PSTN leg에서 시작했습니다 — 하지만 codec boundary를 당신의 process 바깥, Daily 안으로
옮기고, DTMF 경로도 같이 가져갑니다(§6.10: Daily는 native outbound DTMF를 가진 두 transport 중 하나입니다).

당신에게 직접 유용한 Daily 디테일 두 개 더:

**Room join, 그리고 성능 관련 default:**

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

**참가자별 audio capture** — `daily/transport.py:1157`의
`capture_participant_audio(participant_id, callback, audio_source="microphone", sample_rate=16000,
callback_interval_ms=20)`. eval/QA harness에 대해서는, mixed-down 녹음이 아니라 화자별로 분리된 track을
주는 기능이고, 그것이 barge-in 동작을 채점할 수 있느냐 없느냐의 차이입니다. observability는
[[ch-11/read]]가 소유하지만 그래도 기록해 둘 가치가 있습니다.

---

## 9. `gateway/server/`(1,404줄)에 대한 회계

이 section의 모든 것은 [[boson-gateway-server]]에서 나옵니다 — §0에 따르면 Class B evidence입니다.

### 9.1 그 1,404줄에 실제로 무엇이 들어 있는가

| File | LOC | 하는 일 |
|---|---|---|
| `websocket.py` | 734 | `GatewayWebSocketServer`(L35); socket lifecycle, session별 map 열 개, dispatch generation, silence timer |
| `access.py` | 374 | bearer auth, subprotocol token, origin allowlist, principal-to-session binding, signed play cookie |
| `protocol.py` | 114 | 3-field JSON envelope, type vocabulary, session-id validation |
| `interruption.py` | 95 | barge-in policy |
| `history.py` | 70 | `serialize_history()` debug projection |
| `__init__.py` | 17 | |
| **합계** | **1 404** | |

### 9.2 envelope는 `FrameSerializer`에 거의 정확히 대응된다

```python
# packages/gateway/gateway/server/protocol.py L15-31
# (boson-agent, private; excerpt-attested via [[boson-gateway-server]])
@dataclass
class ClientMessage:  session_id: str; type: str; content: str
@dataclass
class ServerMessage:  session_id: str; type: str; content: str = ""
```

양방향 모두 문자열 field 세 개. frame id 없음, timestamp 없음, sequence number 없음, audio field 없음,
binary 경로 없음. `VALID_CLIENT_TYPES`(L33)는 네 항목입니다 — `user_message`, `partial_transcript`,
`interrupt`, `get_history`. `VALID_SERVER_TYPES`(L39)는 여섯 개를 선언하지만, excerpt는 모든
`ServerMessage(...)` 생성 지점을 grep하면 실제로 emit되는 type은 **네 개**뿐이라고 기록합니다:
`error`(websocket.py:231, 240, 263, 500), `history`(:281), `text_delta`(:471), `turn_end`(:477).
`interrupted`와 `stage_changed`는 선언되고 결코 보내지지 않습니다 — 그리고
`agents/test-lina-gateway/client.py:113`에는 오늘 기준 dead code인 살아 있는
`elif data["type"] == "stage_changed":` branch가 있습니다.

이것은 §7의 그림에 깨끗하게 projection되는 작고 닫힌 vocabulary입니다: `protocol.py`는 `serialize`가
branch 네 개를 갖고 `deserialize`가 branch 네 개를 갖는 `FrameSerializer` subclass가 됩니다. 이
migration에서 가장 깨끗한 조각 하나입니다. `stage_changed`는 존재하는 것처럼 포팅하지 **마십시오**.
excerpt에 따르면 그것은 신규 작업이 됩니다.

### 9.3 transport layer에서 Pipecat counterpart가 없는 것들

[[boson-gateway-server]]에 따르면, `GatewayWebSocketServer.__init__`(`websocket.py:43–138`)은 session별
map을 **열 개** 유지합니다: `_session_connections`, `_session_owner`, `_session_timers`,
`_partial_transcripts`, `_partial_transcript_owners`, `_partial_finalize_claims`, `_active_tasks`,
`_active_started_at`, `_dispatch_locks`, `_dispatch_generations`. migration이 그것들에 무엇을 하는지에
따라 묶으면:

| boson mechanism | Pipecat에서 어디로 가는가 | 왜 |
|---|---|---|
| socket lifecycle: `start()`, `_handle_connection()`, teardown | `FastAPIWebsocketTransport`로 **대체됨** | §5.1 — route는 당신이 유지하고, transport는 accept된 socket을 받음 |
| `protocol.py` envelope | `FrameSerializer` subclass로 **포팅됨** | §9.2 |
| `_start_silence_timer` / `_finalize_partial` | `src/pipecat/turns/`로 **재배치됨** | §4 — 이 버전에서 endpointing은 transport layer가 아님 |
| `access.py` (374 L) | 당신의 FastAPI route에 **유지됨** | §5.1 — Pipecat은 Origin allowlist 하나만 ship함 |
| `history.py` + turn별 직렬화를 위한 `history_lock` | **유지됨** | Pipecat 동등물 없음; *당신의* session store 위의 debug projection임 |
| `_reserve_session_dispatch` / `_replace_active_task` / `_cancel_session_dispatch` | **유지됨** | `PipelineTask`는 bot turn을 취소함; socket 두 개가 한 session을 두고 경쟁한다는 개념이 없음 |
| session identity, reconnect-and-resume, 1800 s idle TTL | 바깥 layer에 **유지됨** | Pipecat pipeline은 connection 단위임 |

generation protocol은 파일에서 가장 대체 불가능한 것이므로 자기 문장을 하나 가질 자격이 있습니다.
excerpt에 따르면, `_reserve_session_dispatch`(:515)는 **어떤 await보다 먼저**
`_dispatch_generations[sid]`를 증가시킵니다 — *"event-loop tasks cannot interleave until an await, so
incrementing the generation here records ordering across connection reader loops"* — 그리고
`_cancel_active_task`(:584)는 취소된 task를 *await*하는 v0.7.4 hotfix라서, 후속 코드가 session state를
읽기 전에 `except CancelledError`가 이미 `session._pending_partial`을 저장해 둡니다. 두 사실 모두
**connection 두 개가 논리적 session 하나를 두고 경쟁하는 것**에 관한 것입니다. [[ch-04/read]] §7이
Pipecat의 cancellation model을 보여주었습니다. 그것은 pipeline 단위이고, Pipecat pipeline은 connection
단위입니다. 대응시킬 것이 없습니다.

그리고 `access.py`가 유지되는 코드 중 가장 큰 단일 블록이 사는 곳입니다:

- `Authorization` header를 통한 bearer **또는** — browser WebSocket API가 header를 설정할 수 없으므로 —
  동반 subprotocol `boson-bearer.<token>`(:296), 그리고 `"boson-gateway"`를 negotiate하는
  `select_subprotocol`(:43)
- 생성자 invariant(`websocket.py:56–66`): `auth_token`은 32자 이상; non-loopback host는 token을
  **요구함**; `customer_db_path`를 켜면 `allowed_origins`가 필요함
- session을 *principal*에 binding하는 `SessionAccess.authorize(websocket, session_id, operation)`(:338)
  — token의 sha256, 또는 token이 없을 때는 connection별 `secrets.token_urlsafe(32)` — 그리고 binding되지
  않은 session에 대한 `get_history`는 **거부됨**(:343–344)
- process별 HMAC-SHA256 signed cookie `boson_gateway_play`, `PLAY_COOKIE_MAX_AGE_SECONDS = 300`,
  `HttpOnly; SameSite=Strict`

그 전부에 대한 Pipecat counterpart는 §5.1의 12줄짜리 `is_origin_allowed`입니다.

### 9.4 그 숫자

[[boson-gateway-server]]의 결론이고, 파일 inventory가 그것을 뒷받침합니다: **"1,404줄을 지운다"가 아니라
~700줄 유지, ~700줄 포팅으로 예산을 잡으십시오.** 반대편에서 *얻는* 것은 boson이 한 번도 가져 본 적
없는 audio path 전체입니다 — media 입출력, VAD wiring, `InterruptionFrame` 전파, telephony serializer
여섯 개, 그리고 `src/pipecat/turns/` 아래의 turn strategy들. *지불하는* 것은 socket plumbing과 envelope의
재작성, endpointing의 재배치, 그리고 frame layer 위의 모든 것을 지금 있는 자리에 그대로 유지하는
것입니다. 이 문장의 양쪽 절반 모두 사실이고, 어느 쪽도 권고가 아닙니다.

---

## 10. `realtime_voice/transport/webrtc/`에 대한 회계

이것도 Class B evidence이고, [[rtv-webrtc-transport]]에서 나옵니다.

### 10.1 ⚠️ excerpt의 산술 문제 — 반복하지 않고 보고함

[[rtv-webrtc-transport]]는 이 module이 **"~960 LOC"**라고 진술한 다음 파일들을 나열합니다:

| File | LOC (excerpt 기준) |
|---|---|
| `manager.py` | 248 |
| `control.py` | 226 |
| `peer.py` | 231 |
| `tracks.py` | 216 |
| `buffer.py` | 123 |
| `config.py` | 64 |
| `transport.py` | 168 |
| **합** | **1 276** |

248 + 226 + 231 + 216 + 123 + 64 + 168 = **1,276**이지 ~960이 아닙니다. excerpt 자신의 파일별 표가 자기
headline 수치와 316줄 — 약 33% — 어긋납니다. [[ch-03/read]] §6.2는 이것을 잡아내지 못한 채 ~960 수치를
반복했고, course outline은 이 chapter에 대해 또 반복합니다.

저는 이것을 해결할 수 없습니다: repo가 이 머신에 없고, 이 course의 rule 3이 그것을 여는 것을 금지합니다.
그럴듯한 설명 두 가지 — 960이 빈 줄과 docstring을 제외한 값이고 파일별 숫자는 raw `wc -l`이거나, 아니면
headline이 그냥 stale이거나 — 이고, 둘 중 어느 쪽에 대한 증거도 없습니다. **1,276을 파일 listing 합계로
쓰고, 직접 `wc -l`을 돌릴 수 있을 때까지 "~960"은 unverified로 취급하십시오.** 이 숫자로 migration
규모를 잡는다면, 960과 1,276의 차이는 일주일입니다.

반대편의 규모를 위해, Pipecat 수치는 검증 *가능*합니다: `src/pipecat/transports/smallwebrtc/`는 파일
세 개에 걸쳐 2,176줄입니다(§8.1).

### 10.2 같은 땅, 그리고 Pipecat이 ship하지 않는 두 가지

두 구현 모두 aiortc의 `RTCPeerConnection`을 감싸고, 둘 다 PyAV로 resample하고, 둘 다 data channel을
구동하고, 둘 다 outbound audio를 wall clock에 맞춰 pacing합니다. excerpt가 기록하는 갈라짐은 transport
package 안에 앉아 있는 policy입니다:

**`WebRTCSessionManager`**(`manager.py` L51) — docstring이 *"Create short-lived authorized sessions and
enforce one live peer each."*라고 말합니다. `create_session(customer_id, ...) -> VoiceSessionTicket`이
`secrets.token_urlsafe(32)`를 발행하고 `hashlib.sha256(token).digest()`만 저장합니다. `_authorize`(L227)가
만료를 확인한 뒤 `hmac.compare_digest`를 합니다. `session_token_ttl_seconds = 15 * 60`.
`accept_offer(..., reconnect: bool = False)`는 `reconnect=True`가 아니면
`SessionConflictError("this voice session already has a live peer")`를 raise합니다.

Pipecat 쪽, tree에서 확인:

```bash
$ grep -rn "token_urlsafe\|compare_digest" src/pipecat/
src/pipecat/runner/run.py:324:    if not hmac.compare_digest(expected, sig):
src/pipecat/transports/whatsapp/client.py:181:        if not hmac.compare_digest(expected_signature, received_signature):
```

두 hit 모두 webhook-signature 검증입니다. 어느 쪽도 voice-session authorization이 아닙니다. §8.1의
두-argument `SmallWebRTCConnection.__init__`과 합치면 결론은: **Pipecat의 WebRTC layer 어디에도 token
없음, TTL 없음, customer binding 없음, one-live-peer rule 없음**이고, `request_handler.py`는 맨
offer/answer endpoint입니다.

**`ControlEvent`**(`control.py` L25) — 버전이 있고(`CONTROL_PROTOCOL_VERSION = 1`), sequence number가
붙고, 64 KiB로 크기 상한이 있으며, `_reject_audio_payload`(L117)가 audio처럼 보이는 것을 재귀적으로
거부하고, `OrderedControlChannel`(L136)이 생성 시점에 partially-reliable data channel을 거부합니다.
Pipecat의 counterpart는 `on_app_message(message: Any, sender: str)` — §8.1의 등록된 handler — 이고,
schema 없음, sequence 검사 없음, 크기 상한 없음, audio 금지 없음입니다. Pipecat에는 typed client
protocol이 *있긴* 합니다(RTVI, [[rtvi-observability]]). 하지만 그것은 다른 layer를 타고 가며 data
channel에서 강제되지 않습니다.

### 10.3 두 열짜리 진술

| | Pipecat `smallwebrtc/` (2,176 L, 검증됨) | realtime_voice `transport/webrtc/` (파일 listing 기준 1,276 L, §10.1) |
|---|---|---|
| Library | aiortc | aiortc |
| Output granularity | 10 ms, 정렬 어긋나면 `ValueError` | 20 ms, `AudioFifo`가 나머지 유지 |
| Renegotiation / ICE restart | `renegotiate(..., restart_pc)`, `ask_to_renegotiate()` | 없음; `accept_offer(reconnect=True)`만 |
| Video / screen share | 있음 | 구성상 없음 (`__post_init__`이 `output_channels != 1`을 거부) |
| Session authorization | 없음 | HMAC token, 15분 TTL, one live peer, 명시적 reconnect |
| Data-channel protocol | untyped `Any` passthrough | 버전 있음, 순서 보장, 크기 상한, audio 금지 |
| Named error type | 일반 exception | 7개 (`webrtc/errors.py`, 33 L) |
| 같은 package 계열의 다른 transport | 10개 더 | 없음 |

여기서 Pipecat의 transport를 채택한다는 것은 왼쪽 열이 도착하고 오른쪽 열의 policy 행 두 개를
**`SmallWebRTCConnection` 주변의 application code로 재구현해야 한다**는 뜻입니다. Pipecat이 그 둘을 다
ship하지 않기 때문입니다. 그것이 진술 전부입니다. 그것에 대해 무엇을 할지는 [[ch-13/read]]가 결정합니다.

---
## 11. Lina를 위한 네 개의 framework-extension move

### 11.1 통신사와 이야기하기 *전에* `LinaCarrierSerializer(FrameSerializer)`를 명세하십시오

당신은 pattern B(§2.2)를 작성하게 됩니다. 여기 checklist가 있습니다. ship되는 serializer 여섯 개가
실제로 무엇을 하는지에서 도출한 것이라, 통화 중에 항목을 하나씩 발견하는 대신 통신사 integration 팀에
요구사항 목록으로 건넬 수 있습니다:

| # | 요구사항 | ship되는 코드가 그것을 하는 곳 |
|---|---|---|
| 1 | `setup()`에서 pipeline rate를 학습 | `twilio.py:164` — `self._sample_rate = self._params.sample_rate or setup.audio_in_sample_rate` |
| 2 | inbound에서 companding decode *후* resample | `audio/utils.py:222–226` |
| 3 | outbound에서 resample *후* companding encode | `twilio.py:193–195` |
| 4 | `InterruptionFrame`에서 provider의 flush message emit | `twilio.py:186–188` |
| 5 | `EndFrame`/`CancelFrame` 동작을 명시적으로 결정(끊는다, 또는 안 끊는다) | `twilio.py:178–185`; Exotel은 의도적으로 안 함 |
| 6 | 모든 outbound message에 stream identifier echo | `twilio.py:187, 203` |
| 7 | inbound 키 입력에 대해 `InputDTMFFrame(KeypadEntry(digit))` 반환 | `twilio.py:305–312` |
| 8 | chunk timing이 불규칙하면 `resampler_clear_after_secs=None` 설정 | `base_serializer.py:38–40` |
| 9 | 고정 packet size를 요구하면 그것을 결정 | `fastapi.py:575–586` |

통신사에게 *먼저* 물어볼 질문 두 개. 위의 모든 것을 결정하기 때문입니다: **wire 위의 sample rate와
codec은 무엇인가**, 그리고 **flush/clear message가 있는가**. 두 번째 답이 "없다"라면 그 통신사에서
barge-in은 그들의 playout buffer 깊이로 제한되며, barge-in latency를 약속하기 전에 그 숫자를 알아내야
합니다(§6.9, [[ch-11/read]]).

예산: Twilio 모양의 protocol이면 250–300줄, Genesys 모양의 session state가 있으면 최대 ~1,000줄(§6.11).

### 11.2 `ControlEvent`를 data-channel 관심사가 아니라 serializer 관심사로 포팅하십시오

boson의 `ControlEvent`와 `OrderedControlChannel`(§10.2)은 invariant를 **aiortc data channel**에서
강제합니다. Pipecat에는 거기에 hook이 없습니다 — `on_app_message`는 `Any`를 받습니다. 하지만 §6.11이
`GenesysAudioHookSerializer`가 `_client_seq` / `_server_seq`, session open/close/pause state machine,
event handler 일곱 개를 *serializer 안에서* 유지한다는 것을 보여주었고, §5.3이 inbound non-audio frame이
`InputTransportMessageFrame`으로 도착해서 양방향으로 broadcast된다는 것을 보여주었습니다.

그래서 port는 이것입니다: **`ControlEvent` validation을 `deserialize()`/`serialize()` 안으로 옮긴다.**

- `from_json`의 unknown-field 거부, version 검사, `_reject_audio_payload` → `deserialize()`의 맨 위로.
  당신이 선택한 정책에 따라 raise하거나 `None`을 반환(§7에 유의: Pipecat의 house style은 `None`이고
  당신 것은 `raise`입니다 — 의도적으로 고르십시오. 실수로 상속받지 마십시오)
- 엄격한 in-order `sequence` 검사 → serializer instance state로, Genesys의 `_client_seq`와 정확히 같게
- outbound private counter → serializer instance state로, Genesys의 `_server_seq`와 정확히 같게
- 64 KiB 상한 → `serialize()`의 길이 검사
- `ordered=False` / `maxRetransmits` / `maxPacketLifeTime` 거부 → WebSocket 경로에서는 **serializer
  대응물이 없습니다.** TCP가 이미 ordered하고 reliable하기 때문입니다. WebRTC 경로에서는 data channel에
  그대로 남고, 당신의 `SmallWebRTCConnection` setup 코드에 살아야 합니다.

마지막 항목이 눈여겨볼 것입니다: 여섯 invariant 중 다섯이 깨끗하게 재배치되고, 여섯 번째는 당신이
교체하려는 바로 그 transport에서만 의미가 있습니다.

> 💡 **쉬운 설명 — 왜 다섯 개는 옮겨지고 하나는 못 옮기나요?**
> `ControlEvent`의 invariant를 두 부류로 나눠 보십시오. (a) **message 내용에 대한 규칙** — version이
> 맞나, 모르는 field가 있나, audio가 섞여 있나, sequence가 순서대로인가, 64 KiB 넘나. 이건 전부
> "byte를 보고 판단"이라서 serializer가 할 수 있습니다. (b) **transport 자체의 성질에 대한 규칙** —
> "이 channel은 순서 보장이 없으니 거부한다". 이건 byte를 보고 알 수 있는 게 아니라 channel을
> 만들 때의 설정입니다. WebSocket은 TCP 위라 순서·재전송이 이미 보장되므로 (b)는 애초에 검사할
> 대상이 사라집니다 — 사라진 게 아니라 **공짜로 만족된** 것입니다.

### 11.3 serializer seam을 당신의 eval seam으로 만드십시오

`RTVIEvalSerializer`(§5.7, §7의 CORRECTION 3)는 Pipecat 자신의 harness입니다: serializer 하나 +
`SingleClientWebsocketServerTransport`. 일반화하면 **serializer는 I/O boundary 전체에 대한
protocol-shaped test double**이고, 필요할 때 아홉 줄이면 됩니다
(`tests/test_websocket_server_transport.py:49`).

Lina를 위한 구체적인 수이고, 이 section에서 가장 가치가 높은 것입니다: `ReplayCarrierSerializer`를
만드십시오. 그 `deserialize()`는 녹음된 8 kHz μ-law frame을 20 ms 스케줄로 디스크에서 읽어 동일한
`ulaw_to_pcm` 경로를 통해 `InputAudioRawFrame`을 emit하고, 그 `serialize()`는 bot의 μ-law 출력을
socket이 아니라 파일에 씁니다. 그러면:

- 당신의 STT 정확도 수치가 studio audio가 아니라 **진짜** signal chain(§6.6) 위에서 측정됩니다
- barge-in timing이 결정론적인 clock에 대해 측정되므로, [[ch-11/read]]의 budget이 재현 가능한 입력을
  갖게 됩니다
- regression test가 통신사 없이, network 없이, 비용 없이, CI에서 돕니다
- 그리고 test 대상 pipeline이 production과 *byte 단위로 동일*합니다. 바꾼 것이 `params.serializer`
  하나뿐이기 때문입니다

이것이 §6.6을 경고에서 측정으로 바꾸는 framework-extension move입니다.

### 11.4 browser/console 경로를 통화 경로와 따로 결정하십시오

그것들은 서로 다른 transport이고, 서로 다른 결정이어야 합니다:

- **실제 PSTN 통화**: `FastAPIWebsocketTransport` + 당신의 통신사 serializer (§6)
- **agent console / supervisor listen-in / QA harness**: WebRTC transport. 여기서는
  `capture_participant_audio(..., audio_source="microphone", sample_rate=16000)`(§8.2)이 화자별로 분리된
  track을 주고, 당신은 이미 ~1,276줄의 동작하는 aiortc 코드를 갖고 있습니다(§10)

둘 다에 대해 하나의 답을 강요하는 것은 아무것도 없고, 그 둘을 하나의 질문으로 취급하는 것이 migration
추정이 틀어지는 방식입니다. [[ch-13/read]]가 정확히 이 이유로 둘을 따로 채점합니다.

---

## 12. transport layer fact sheet

‹excerpt› 표시가 없으면 `0cbf9c5b` 시점의 `pipecat-src`에 대해 검증되었습니다.

| Fact | Value |
|---|---|
| `BaseTransport` | `base_transport.py:96`, 42줄, 137줄짜리 파일 안 |
| Abstract method | 2개 — `input()` :122, `output()` :131, 둘 다 `-> FrameProcessor` |
| Transport provider package | **11** |
| `BaseTransport` subclass | **13** (websocket 3, local 2, whatsapp **0**) |
| `TransportParams` audio default | `audio_in_enabled=False`, `audio_out_enabled=False`, `audio_out_10ms_chunks=4` |
| Sample-rate fallback | 16 000 in / 24 000 out — `frame_processor.py:106–107` |
| Output chunk 산술 | `base_output.py:135–136`; 24 kHz에서 1 920 B = 40 ms |
| `TransportParams`의 `vad_analyzer` | **제거됨** — CHANGELOG L4402 |
| 살아 있는 VAD mount point | `LLMUserAggregatorParams.vad_analyzer`, `VADProcessor`, `VADController` |
| `FastAPIWebsocketTransport` | `websocket/fastapi.py:611`; 이미 accept된 socket을 받음 |
| `FastAPIWebsocketParams` 추가 field | 정확히 6개, `serializer`와 `ws_close_timeout: float = 0.5` 포함 |
| serializer 미설정 시 | 모든 inbound message 드롭 (`continue`, :376–377); 모든 outbound 드롭 (`return False`, :568–569) |
| 조작된 audio clock | `_send_interval = (audio_chunk_size / sample_rate) / 2` :456; `/2`는 bytes→samples 인수이고 16-bit mono에서만 옳음 |
| WebSocket transport module | fastapi 707 L, server 716 L, client 559 L |
| Dev server single-client | close code 1013, `server.py:255` |
| Telephony transport | **0** |
| Telephony serializer | **6** — twilio 314, telnyx 292, plivo 256, exotel 171, genesys 964, vonage 188 |
| concrete `FrameSerializer` subclass | `serializers/` 아래 **7**, `src/pipecat/` 아래 **8** (`RTVIEvalSerializer`) |
| Wire rate default | 8 000 Hz ×5; Vonage 16 000 Hz |
| 8 kHz에서의 Nyquist ceiling | **4 kHz**; μ-law는 8-bit companded |
| Telephony wiring 증거 | `runner/utils.py:486` → `:554 return FastAPIWebsocketTransport(...)` |
| Barge-in flush | `clear` / `clearAudio`, provider별, `serialize(InterruptionFrame)`을 통해 |
| `auto_hang_up` | Twilio/Telnyx/Plivo에서 기본 `True`; Exotel에는 없음 |
| Outbound DTMF | serializer 지원 없음; `_write_dtmf_audio()`가 tone WAV를 audio로 밀어냄 |
| Native outbound DTMF | **transport 2개** — Daily `:2246`, LiveKit `:1018` |
| `KeypadEntry` | 12개 항목, A/B/C/D 없음 |
| `SmallWebRTCTransport` | `smallwebrtc/transport.py:951`; 맨 `TransportParams`; package 2 176 L |
| `SmallWebRTCConnection.__init__` | `ice_servers`, `connection_timeout_secs=60` — 그 외 없음 |
| 등록된 SmallWebRTC event | 3개 — `on_app_message`, `on_client_connected`, `on_client_disconnected` (docstring의 `on_client_message`는 존재하지 않음) |
| Protobuf wire vocabulary | **5** frame type, exact-type match, 123개 `Frame` descendant 중에서 |
| boson `gateway/server/` ‹excerpt› | 1 404 L; ~700 유지 / ~700 포팅 |
| boson client message type ‹excerpt› | valid 4개; server type 6개 선언, **4개**만 emit됨 |
| realtime_voice `transport/webrtc/` ‹excerpt› | 파일별 listing 합계 **1 276 L** (headline "~960"은 unverified — §10.1) |

---

## 13. 이 chapter가 해결하지 않는 것

- **사용자의 turn이 언제 끝나는가.** §4는 endpointing을 transport에서 떼어내고 `src/pipecat/turns/`를
  그 집으로 지목했습니다. boson의 `_start_silence_timer` / `_finalize_partial` port는 [[ch-06/read]]의
  것입니다.
- **어떤 한국어 STT가 8 kHz μ-law에서 살아남는가.** §6.6이 제약과 측정 규칙을 확립했습니다. provider
  선택은 [[stt-korean-providers]]와 함께 [[ch-07/read]]입니다.
- **interruption cascade 전체.** §5.4, §6.9, §8.1이 각각 고리 하나씩을 주었습니다. 그 사슬 —
  processor queue, transport buffer, 통신사 playout, 그리고 boson의 text-transcript 기반 결정 logic —
  은 [[ch-08/read]]입니다.
- **turn이 중간에 취소될 때 tool-call repair가 어디로 가는가.** [[ch-04/read]]의 다음 챕터로에
  주차되어 있고, [[ch-09/read]]가 소유합니다.
- **hop별 latency 수치.** §5.4의 조작된 clock, §6의 두 번의 resample, §8.1의 granularity 차이는 전부
  측정치가 붙어 있지 않은 latency 항입니다. [[ch-11/read]].
- **유지할 것인가 교체할 것인가.** §9.4와 §10.3은 inventory이고, 의도적으로 채점되지 않았습니다.
  [[ch-13/read]].

---

## 다음 챕터로

이 chapter가 앞으로 넘기는 것들입니다. 뒤 chapter들이 다시 도출하는 대신 인용할 수 있도록 이름을
붙였습니다.

- **transport는 processor 한 쌍이다** (§1). `BaseTransport`는 42줄이고 `FrameProcessor`를 반환하는
  abstract method 두 개입니다. 이후 "transport boundary에서"라고 말하는 모든 chapter는 server가 아니라
  그 두 object 중 하나를 뜻합니다.
- **세 가지 provider pattern** (§2.2) — 새 transport class / 새 serializer / 새 signalling client, 그리고
  세 witness인 `DailyTransport`, `TwilioFrameSerializer`, `WhatsAppClient`. [[ch-13/read]]가
  carrier-integration 항목의 규모를 잡는 데 이것이 필요합니다.
- **serializer는 codec boundary이자 resample point이자 call-control client이자 DTMF decoder다** (§6.4).
  [[ch-07/read]]는 resample point가 필요하고, [[ch-08/read]]는 `clear` event가 필요하며,
  [[ch-11/read]]는 두 resample 모두를 latency 항으로 필요로 합니다.
- **4 kHz 천장** (§6.6)과 거기서 따라오는 측정 규칙: 16 kHz audio에서 측정한 한국어 STT 수치는 이
  pipeline에 전이되지 않습니다. [[ch-07/read]]가 이 위에 세워지고 [[ch-13/read]]가 이것에 대고
  채점합니다.
- **조작된 audio clock** (§5.4) — WebSocket에는 playout back-pressure가 없으므로 transport가 device를
  흉내 내기 위해 자고, interruption이 schedule을 0으로 만듭니다. [[ch-08/read]]와 [[ch-11/read]]가 둘 다
  이것을 소비합니다.
- **VAD는 transport에 없다** (§4). `TransportParams`를 겨냥한 port plan은 더 오래된 Pipecat을 겨냥한
  것입니다. [[ch-06/read]]가 그 재배치를 소유합니다.
- **wire는 언제나 closed union이다** (§7). Pipecat은 provider마다 serializer에서 그것을 닫고,
  realtime_voice는 `_control_event`에서 한 번에 닫습니다. [[ch-13/read]]의 open-versus-closed 질문에
  이제 세 번째 data point가 생겼습니다.
- **두 개의 migration inventory** (§9, §10) — `gateway/server/`에 대해 ~700 유지 / ~700 포팅, 그리고
  WebRTC 쪽에서 application code가 되는 이름 붙은 policy subsystem 두 개(`WebRTCSessionManager`,
  `ControlEvent`). 둘 다 채점되지 않았고, 둘 다 [[ch-13/read]]를 기다립니다.
- **네 개의 correction** — `ls` listing에는 `__init__.py`가 포함되고; `_supports_native_dtmf`의
  override는 하나가 아니라 둘이고; repo 전체에 `FrameSerializer` subclass는 7개가 아니라 8개이며;
  `SmallWebRTCTransport`의 docstring이 결코 등록되지 않는 `on_client_message` event를 광고합니다.
  더해서 해결되지 않은 excerpt 불일치 하나: realtime_voice의 WebRTC package가 1,276줄로 나열되고
  ~960으로 headline되어 있습니다(§10.1).

[[ch-06/read]]는 audio path의 다음 질문을 다루고, 그것은 §4가 계속 미뤄 온 그 질문입니다:
**사용자의 turn은 언제 끝나는가?** hysteresis machine으로서의 VAD, streaming STT endpointing, 그리고
`src/pipecat/turns/` 아래의 turn-strategy chain — gateway에 있는 boson의 silence timer, 그리고 16 kHz
mono가 아니면 raise하는 realtime_voice의 `SileroVAD`에 대비해서. 당신은 이제 그 machinery에 도달하는
audio가 8 kHz wire에서 나와 interpolate되어 올라온 것임을 압니다. turn detector가 그것으로 무엇을 하는지
알아보러 가십시오.

