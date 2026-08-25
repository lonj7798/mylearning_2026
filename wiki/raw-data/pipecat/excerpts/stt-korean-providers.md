# Korean-Capable STT Services in the Pipecat Tree

<!-- slug: stt-korean-providers · type: source · source: src/pipecat/services/*/stt.py + src/pipecat/transcriptions/language.py -->

**Core Insight.** "Does Pipecat support Korean STT?" has three different answers depending
on the service: a *verified* mapping (`Language.KO` in a hand-curated `LANGUAGE_MAP`), a
*passthrough* (no map at all — the enum's string goes straight to the provider), or a
*documented exclusion* (the map exists and Korean is deliberately not in it). Only the
first is evidence. The repo carries **latency** numbers per provider but **zero accuracy
numbers** for any language, and nothing at all about 8 kHz telephony audio.

**Guideline.** Shortlist by *verified* Korean mapping plus streaming plus low measured
TTFS, then measure Korean-on-8 kHz yourself. Never infer Korean support from a service's
presence in the tree — `AssemblyAISTTService` is a full streaming STT whose declared
language list has no Korean, and `SarvamSTTService` maps `Language.KOK_IN` (Konkani).

## Technical Details

`src/pipecat/transcriptions/language.py` L310-311 defines `KO = "ko"` and `KO_KR = "ko-KR"`.
27 STT service modules live under `src/pipecat/services/`. What each actually says:

**Verified Korean (`Language.KO` present in the service's own map):**

| service | file:line | code sent | base class | TTFS P99 (s) |
|---|---|---|---|---|
| `SonioxSTTService` | `soniox/stt.py:163` | `"ko"` | `WebsocketSTTService` | 0.35 |
| `SpeechmaticsSTTService` | `speechmatics/stt.py:1159` | `"ko"` | `STTService` (streaming) | 0.74 |
| `GladiaSTTService` | `gladia/stt.py:113` | `"ko"` | `WebsocketSTTService` (`solaria-1`) | 1.49 |
| `GoogleSTTService` | `google/stt.py:233-234` | `"ko-KR"` | `STTService` (streaming) | 1.57 |
| `AzureSTTService` | `azure/common.py:199-201` | `"ko-KR"` | `STTService` (streaming) | 1.80 |
| `AWSTranscribeSTTService` | `aws/stt.py:458-459` | `"ko-KR"` | `WebsocketSTTService` | 1.90 |
| `NvidiaSTTService` (Riva) | `nvidia/stt.py:92-93` | `"ko-KR"` | `STTService` | 1.0 (unmeasured) |
| `ElevenLabsSTTService` | `elevenlabs/stt.py:109` | `"kor"` (3-letter!) | `SegmentedSTTService` | 2.01 |
| `XAISTTService` | `xai/stt.py:67` | `"ko"` | `WebsocketSTTService` | 2.14 |
| `FalSTTService` | `fal/stt.py:89` | `"ko"` | `SegmentedSTTService` | 2.07 |
| `WhisperSTTService` (local) | `whisper/stt.py:154`, `base_stt.py:90` | `"ko"` | `SegmentedSTTService` | 1.0 (default) |
| `MoonshineSTTService` (local) | `moonshine/stt.py:67` | `"ko"` | `SegmentedSTTService` | — |
| `FunASRSTTService` (local) | `funasr/stt.py:41` | `"ko"` | `SegmentedSTTService` | — |

Two local services name Korean in prose, the strongest in-repo statement of support:
`funasr/stt.py` L41 `_FUNASR_LANGUAGES = {"zh", "en", "ja", "ko", "yue", "nospeech"}` with
the docstring calling SenseVoice *"multilingual (Chinese, Cantonese, English, Japanese,
Korean and more)"*; `moonshine/stt.py` L123 *"Moonshine publishes models for Arabic,
Chinese, English, Japanese, Korean, Spanish, Ukrainian, and Vietnamese"*.

**Passthrough — no map, therefore no in-repo verification:**

- `DeepgramSTTService` (`deepgram/stt.py:288`) has **no `LANGUAGE_MAP` and no
  `language_to_service_language` override**. It serialises whatever it holds:
  `kwargs["language"] = str(s.language)` (L581-582), default `Language.EN` (L350).
  `Language.KO` would go out as `"ko"`; the repo takes no position on whether Deepgram
  accepts it. Deepgram nonetheless has the joint-best measured TTFS (0.35 s).
- `DeepgramFluxSTTService` (`deepgram/flux/stt.py:40`) documents only
  `model="flux-general-multi"` plus `language_hints`; no per-language map, and
  `supports_ttfs` returns `False` (L249) because Flux defines turns server-side.
- `OpenAISTTService` (`openai/stt.py`) passes `self._settings.language` straight into the
  transcription request (L155-161), default `Language.EN`.
  `OpenAIRealtimeSTTService._language_to_code` (L383-393) just takes the ISO-639-1 base
  code. Korean rides on Whisper's own language set; nothing in-repo confirms it.

**Documented exclusions — read these before shortlisting:**

- `AssemblyAISTTService`: `language_to_assemblyai_language` `LANGUAGE_MAP`
  (`assemblyai/stt.py` L128-148) lists ar, da, de, en, es, fi, fr, he, hi, it, ja, nl, no,
  pt, sv, tr, vi, zh — **no Korean**. The `language_code` docstring (L196-204) enumerates
  tier-1 `"en"/"es"/"fr"/"de"/"it"/"pt"` plus `"tr","nl","sv","no","da","fi","hi","vi",
  "ar","he","ja","zh"` — again no `ko`. `resolve_language(..., use_base_code=True)` would
  fall through to `"ko"` *with a warning*: a fallback, not support.
- `CartesiaTurnsSTTService` (`cartesia/turns/stt.py`): comment L157 — *"ink-2 is
  English-only at launch; language on emitted frames is fixed"* — and `self._language =
  Language.EN` (L158). Excluded outright.
- `SarvamSTTService` (`sarvam/stt.py`): the only `KO*` entry is `Language.KOK_IN: "kok-IN"`
  (L849), which is **Konkani**. An easy misread; Sarvam is an Indian-language service.

**Server-side endpointing, relevant to a Korean telephony bot:** `SonioxSTTService` (L422),
`SpeechmaticsSTTService` (L561, with `TurnDetectionMode` = `FIXED` / `EXTERNAL` (default) /
`ADAPTIVE` / `SMART_TURN`), `GladiaSTTService` (L370, when `enable_vad`) and
`DeepgramFluxSTTService` (`flux/stt_base.py` L250) all set `frame.user_turn_strategies =
ExternalUserTurnStrategies(...)` on their `STTMetadataFrame`, handing turn detection to the
provider. See `[[endpointing-turn-boundary]]`.

### The open unknown, stated explicitly

**No measured Korean accuracy number, and no Korean-on-8 kHz-telephony number of any kind,
exists anywhere in this repository at this commit.** Verified by grep across `src/`:
`WER` / `word error rate` / accuracy claims → **zero hits**, for any service, any language.
`stt_latency.py` records only *latency*, is silent on the language and sample rate of the
benchmark audio, and states the values were measured at `VADParams.stop_secs=0.2` —
nothing more. The only `8000` values in the tree are telephony serializer defaults
(`twilio.py:79`, `telnyx.py:60`, `plivo.py:54`, `exotel.py:49`, `genesys.py:148`,
`vonage.py:43`); no STT service documents behaviour at that rate. Per the COLLECTION-PLAN
gap log this fact is recorded rather than guessed. Closing it requires an own-run benchmark
(<https://github.com/pipecat-ai/stt-benchmark>) on real Lina TMR μ-law 8 kHz Korean audio.

- **Migration angle:** boson-agent has no STT to replace — a grep for
  `deepgram|whisper|speech_to_text|vad` over `boson-agent/packages/**/*.py` returns zero
  hits, so this is greenfield selection, not a port. It does decide the fate of
  `gateway/interrupt/detector.py`: a provider with verified Korean **and** server-side
  endpointing (Soniox 0.35 s; Speechmatics 0.74 s + `TurnDetectionMode.SMART_TURN`) lets
  `ExternalUserTurnStrategies` delete boson's `PartialDetector` outright, whereas a
  passthrough provider (Deepgram) keeps local smart-turn in the loop. Nothing here touches
  `basement/` or the stage machine. Blocking item before committing to a provider: run the
  Korean 8 kHz μ-law benchmark, because the whole shortlist is currently ordered by an
  English-assumed latency table.

## Citation

pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be` (2026-08-25);
`src/pipecat/transcriptions/language.py`, the 27 `src/pipecat/services/*/stt.py` modules,
`src/pipecat/services/stt_latency.py`, `src/pipecat/serializers/*.py`. Read 2026-08-25.
