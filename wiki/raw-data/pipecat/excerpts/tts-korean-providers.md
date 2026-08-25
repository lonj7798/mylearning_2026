# Korean TTS in Pipecat — twelve services map `Language.KO`, six of them emit word timestamps

<!-- slug: tts-korean-providers · type: source · source: src/pipecat/services/*/tts.py -->

**Core Insight.** Korean support in Pipecat is a per-service `LANGUAGE_MAP` literal, not a capability negotiation — and `resolve_language()` **does not fail** on an unmapped language, it falls back to the base code and logs a warning. So a Korean bot pointed at a provider with no Korean silently configures fine and produces garbage or silence at runtime. The intersection that actually matters for a barge-in-capable Korean agent — Korean mapping **and** word timestamps — is six services: **Azure, Cartesia, ElevenLabs, Inworld, Soniox, xAI**.

**Guideline.** Choose from that six. Verify the mapping by opening the service's `LANGUAGE_MAP` rather than trusting the provider's marketing page, and pair the language with a Korean voice ID — every default voice in the tree is English (`en-US-SaraNeural`, `Bryce`, …) and passing `Language.KO` with an English voice is the exact failure mode `max_consecutive_zero_audio_contexts=3` was added to catch.

## Technical Details

- **Every TTS service in the tree with an explicit Korean mapping** (grep `Language.KO` across `src/pipecat/services/*/tts.py`, excluding STT):

| Service (class, base) | Korean code | Transport | Word ts |
|---|---|---|---|
| `CartesiaTTSService` (`cartesia/tts.py:219`, `WebsocketTTSService`) | `Language.KO: "ko"` (`:112`) | WS | **yes** |
| `CartesiaHttpTTSService` (`:786`) | same | HTTP | no |
| `ElevenLabsTTSService` (`elevenlabs/tts.py:200`, base `ElevenLabsTTSBase(WebsocketTTSService)` `tts_base.py:491`) | `Language.KO: "ko"` (`tts_base.py:259`) | WS | **yes** |
| `ElevenLabsHttpTTSService` (`elevenlabs/tts.py:637`) | same | HTTP | yes |
| `AzureTTSService` (`azure/tts.py:271`, `TTSService, AzureBaseTTSService`) | `Language.KO: "ko-KR"`, `Language.KO_KR: "ko-KR"` (`azure/common.py:200-201`) | SDK push-stream | **yes** |
| `AzureHttpTTSService` (`azure/tts.py:832`) | same | HTTP | no |
| `GoogleTTSService` (`google/tts.py:1023`, `GoogleBaseTTSService`) | `Language.KO: "ko-KR"` (`:146`, `:363`) | gRPC `streaming_synthesize` (`:996`) | no |
| `GoogleHttpTTSService` (`:550`), `GeminiTTSService` (`:1205`) | same | HTTP / streaming | no |
| `InworldTTSService` (`inworld/tts.py:554`, `WebsocketTTSService`) | `Language.KO: "ko-KR"` (`:78`) | WS | **yes** (`_timestamp_type = "WORD"`, `:719`) |
| `InworldHttpTTSService` (`:123`) | same | HTTP | yes (`:466`) |
| `SonioxTTSService` (`soniox/tts.py:143`, `WebsocketTTSService`) | `Language.KO: "ko"` (`:89`) | WS, ≤5 concurrent streams/conn (`:150`) | **yes** |
| `XAITTSService` (`xai/tts.py:323`, `WebsocketTTSService`) | `Language.KO: "ko"` (`:72`) | WS | **yes**, gated on `with_timestamps` (`:320`) |
| `XAIHttpTTSService` (`:150`) | same | HTTP | no |
| `LmntTTSService` (`lmnt/tts.py:78`, `InterruptibleTTSService`) | `Language.KO: "ko"` (`:54`) | WS | no |
| `MiniMaxHttpTTSService` (`minimax/tts.py:137`) | `Language.KO: "Korean"` (`:69`) → `payload["language_boost"]` (`:393`) | HTTP chunked | no |
| `AWSPollyTTSService` (`aws/tts.py:148`) | `Language.KO: "ko-KR"` (`:98`) | HTTP | no |
| `CambTTSService` (`camb/tts.py:157`) | `Language.KO: "ko-kr"` (`:82`) | HTTP (MARS SDK) | no |
| `XTTSService` (`xtts/tts.py:93`) | `Language.KO: "ko"` (`:69`) | local HTTP | no |

- **Services that do NOT support Korean — verified in source, not assumed**:
  - `RimeTTSService` / `RimeHttpTTSService` / `RimeNonJsonTTSService` — `language_to_rime_language()` (`rime/tts.py:46`) maps exactly five languages to three-letter codes: `DE:"ger", FR:"fra", EN:"eng", ES:"spa", HI:"hin"`. **No Korean.** (Rime *is* word-timestamp capable, which makes this the most tempting wrong choice.)
  - `NeuphonicTTSService` — `language_to_neuphonic_lang_code()` (`neuphonic/tts.py:40`): `de, en, es, nl, ar, fr, pt, ru, HI, zh`. **No Korean.**
  - `KokoroTTSService` — `language_to_kokoro_language()` (`kokoro/tts.py:66`) maps espeak-ng voice names: `en-us, en-gb, es, fr-*, hi, it, ja, pt, pt-br, cmn, yue, zh*`. **No Korean**, and the docstring warns "an unsupported name fails at synthesis time".
  - `PiperTTSService` / `PiperHttpTTSService` (`piper/tts.py:44`, `:207`) — no `language_to_service_language` override at all; `self.Settings(model=None, voice=None, language=None)` (`:91`, `:243`). Language is whatever `.onnx` voice you load; Pipecat has no Korean knowledge here.
  - `OpenAITTSService` (`openai/tts.py:81`) — **no language parameter exists.** No override, `language=None` in `default_settings` (`:181`).
  - `FishAudioTTSService` (`fish/tts.py:79`) — `Settings.language: Language | None = Language.EN` (`:109`) but `default_settings` sets `language=None` (`:159`) and there is no mapping function. Nothing converts `Language.KO` to anything.
  - `DeepgramTTSService`, `GradiumTTSService`, `HumeTTSService`, `ResembleAITTSService` — all `language=None` in default settings, no language map.
- **There is no maintained self-hosted Korean option.** The only local service with a Korean mapping is `XTTSService`, which is `@deprecated` since 1.7.0: *"No replacement. `KokoroTTSService` and `PiperTTSService` are the maintained local TTS services."* Neither maps Korean. On-prem Korean TTS is a gap, not a choice.
- **No Korean-native provider is integrated.** No Typecast, Supertone, or Naver Clova service directory exists under `src/pipecat/services/`.
- **The silent-failure mechanism**: `resolve_language(language, language_map, use_base_code)` (`transcriptions/language.py:583`) — *"Not in map — fall back with warning."* Rime is called with `use_base_code=False`, so `Language.KO` becomes the literal `"ko"` and is sent to an API that only knows `eng/ger/fra/spa/hin`. Nothing raises. The downstream catch is `max_consecutive_zero_audio_contexts: int = 3` (`tts_service.py:168`), which only trips after three consecutive contexts produce zero audio.
- **Korean is deliberately excluded from the CJK word-grouping paths — and that is probably correct.** Both Cartesia (`_is_chinese_or_japanese_language`, `cartesia/tts.py:445`) and ElevenLabs (`elevenlabs/tts_base.py:329`) test `base_lang in {"zh", "ja"}`. Korean falls through to the ordinary space-separated branch, so `includes_inter_frame_spaces` stays `False` and the aggregator inserts spaces between word frames. That matches Korean 어절 spacing — but it is an untested assumption in this code, not a verified Korean path. Cartesia's `_normalize_word_timestamps` (`:479`) collapses a whole zh/ja message into one word; Korean gets per-token frames instead.
- **Word timestamps are what interruption and text-truncation hook into.** Services calling `add_word_timestamps` (full list): `azure, cartesia, elevenlabs, elevenlabs/dialogue, gradium, hume, inworld, resembleai, rime, smallest, soniox, speechify, xai`. Intersect with Korean → **azure, cartesia, elevenlabs, inworld, soniox, xai**. These are the only Korean services where a barge-in can truncate the assistant context at the last *spoken* word (via `TTSTextFrame.pts`) rather than the last *generated* token.
- **Real config details worth carrying**: Azure needs a locale-matched voice — default is `voice="en-US-SaraNeural", language="en-US"` (`azure/tts.py:330-333`); its SSML hard-codes `<mstts:silence type='Sentenceboundary' value='20ms' />` (`:202`) and has a `force_locale` flag (`:95`) that wraps text in `<lang xml:lang>`. Cartesia defaults `model="sonic-3.5", voice=None, language=Language.EN` (`:328`) and streams with `add_timestamps: bool = True` in `_build_msg` (`:531`). ElevenLabs defaults `model="eleven_flash_v2_5"` (`:343`); `"ko"` is present in both `ELEVENLABS_V2_5_LANGUAGES` (32 codes, `tts_base.py:39`) and `ELEVENLABS_V3_LANGUAGES` (74 codes, `:78`), and `elevenlabs_language_code(model, language)` (`tts_base.py:173`) *drops* the language code with a warning if the chosen model doesn't list it. Soniox defaults `model="tts-rt-v2", voice="Bryce"` (`:190`). Inworld uses `pre_merge_tokens=True` (`:466`, `:504`, `:1084`) because it emits spaces and punctuation as separate tokens.
- **Migration angle:** boson-agent has no TTS, so this is a green-field pick with one hard constraint imported from `packages/gateway/gateway/interrupt/`. boson's barge-in today is `CancellationFlag` (`interrupt/cancellation.py`, 187 L) plus `InterruptionGate.allows(session_id, content)` (`server/interruption.py`, 95 L) — a *text-level* cancel that knows what the model generated but not what the user heard. Moving that to Pipecat only gains fidelity if the TTS provider emits word timestamps, which restricts the Korean choice to the six above and effectively rules out Rime, Neuphonic, Kokoro and OpenAI regardless of price. `interrupt/fillers.py` (40 L, `set_filler_check` / `is_filler(text, agent_status)`) is user-side filler *suppression* and has no Pipecat counterpart — it is not replaced by anything here and must be reimplemented as a custom `FrameProcessor` upstream of the interruption strategy. The no-maintained-local-Korean-TTS finding also closes off any on-prem deployment story for Lina without a custom `TTSService` subclass.

## Citation

pipecat-ai/pipecat, commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`, read 2026-08-25.
`src/pipecat/services/{cartesia,elevenlabs,azure,google,openai,rime,neuphonic,lmnt,minimax,fish,kokoro,piper,inworld,xai,soniox,aws,camb,xtts}/tts.py`,
`src/pipecat/services/elevenlabs/tts_base.py`, `src/pipecat/services/azure/common.py`,
`src/pipecat/transcriptions/language.py`, `src/pipecat/services/tts_service.py`.
boson-agent read-only at `/Users/jaewon/mywork_2026/Lina_2026/boson-agent-dev/boson-agent`.
