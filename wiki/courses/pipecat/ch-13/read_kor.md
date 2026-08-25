---
title: "Capstone: 유지할 것과 교체할 것, subsystem 단위로"
chapter: ch-13
phase: decision
course: pipecat
lang: ko
companion_of: read.md
sources:
  - rtv-vs-pipecat-gap
  - rtv-pipeline-session
  - rtv-webrtc-transport
  - rtv-vad-chunking
  - design-boson-rules-on-pipecat
  - deployment-scaling
  - boson-compact-session
  - pipecat-design-philosophy
  - theory-narrow-waist
  - stt-korean-providers
  - tts-korean-providers
  - flows-insurance-example
  - pipeline-task-runner
deps:
  - ch-03
  - ch-05
  - ch-09
  - ch-11
  - ch-12
figure: figures/migration-map.html
pipecat_commit: 0cbf9c5b031eef06e53f0a193b9a67d60230e6be
verified: 2026-08-25
---

# Chapter 13 — Capstone: 유지할 것과 교체할 것, subsystem 단위로

> 이 문서는 [[read]] ([read.md](read.md))의 한국어 companion입니다. section 번호는 원문과 1:1로
> 대응하므로 두 파일을 나란히 놓고 읽을 수 있습니다. code block, 파일 경로, 줄 번호, 수치는 원문
> 그대로입니다. CS/ML 용어는 영어를 유지합니다 (frame, processor, node, transition, action,
> aggregator, latency, observer, falsifier, blocker, serializer, corpus, probe, benchmark,
> code assertion 등).

> **Scope — 미리 명시하고, 이 chapter 전체에 걸쳐 강제됩니다.**
>
> **이 course에서 vote(표)를 던지는 유일한 장소가 여기이고, 던지는 사람은 당신입니다.** 열두 개의
> chapter가 mechanism과 evidence를 생산했고 판정(verdict)은 의도적으로 보류했습니다. [[ch-09/read]]
> §11은 세 개의 resolution에 값을 매겼고 어느 것도 recommended로 표시하지 않았습니다. [[ch-05/read]]
> §10.3은 두 개의 column을 펼쳐 놓고 멈췄습니다. [[ch-03/read]] §0은 두 개의 evidence class를 알려
> 주고 그것들에 점수 매기기를 거부했습니다. 그 전부가 여기로 미뤄진 것입니다.
>
> **이 chapter는 DECIDABLE(결정 가능)한가로 채점됩니다.** 이건 문체에 관한 note가 아니라 강한
> 요구조건입니다. 이 capstone의 이전 draft는 자기 답을 이 course 스스로가 "아무도 가지고 있지 않다"고
> 증명한 네 개의 숫자에 의존시켰고, 그 정직한 output은 *결정하기 위한 계획(a plan to decide)*이었습니다.
> 계획 안에 있는 어떤 것도 채점될 수 없습니다. §1이 그것을 고치는 세 개의 rule을 줍니다.
>
> **저는 어떤 row도 당신 대신 답하지 않습니다.** §3은 열일곱 개의 row를 진술하고, 각각에 대해 어느
> chapter가 어느 mechanism fact를 공급했는지 이름을 붙인 evidence digest를 답니다. vote cell은 전부
> 비어 있습니다. §3 어디에도 당신이 직접 쓰지 않은 `KEEP` / `ADOPT` / `HYBRID-WRAP` label은 없고,
> 비교급 형용사도 없습니다 — *better*도, *wins*도, *should use*도, *the right choice*도 없습니다.
> 이 invariant는 outline이 review 중일 때 두 번 깨졌습니다. 여기서는 깨지지 않습니다.
>
> **당신의 source material 일부는 판정을 담고 있으며, 저는 그것을 벗겨냅니다.**
> [[rtv-vs-pipecat-gap]]은 VAD에 대해 "Pipecat strictly better here"라고 말하고, migration angle에서
> 다섯 항목짜리 Replace/Keep 리스트를 출력합니다. [[rtv-vad-chunking]]은 two-frame blip을 "the sharpest
> correctness delta in the whole comparison"이라고 부릅니다. 그것들은 excerpt 저자의 vote이지 evidence가
> 아니며, 이 chapter의 입장으로 재현되지 않습니다. excerpt가 *mechanism*을 진술하는 곳에서는 그것을
> 사용하고 인용합니다. excerpt가 *선호(preference)*를 진술하는 곳에서는 그것을 버리고, 버렸다고
> 말합니다.

---

## 왜 이 챕터인가

당신은 열두 번에 걸쳐 mechanism을 보되 결정하지는 말라는 요구를 받았습니다. 그건 끝났습니다.

그것이 열두 chapter 동안 금지되었던 이유는, framework의 평판에 근거해 내린 build-vs-buy 결정은 무가치한
반면 file-and-line evidence에 근거해 subsystem 단위로 내린 build-vs-buy 결정은 이 course 전체가 생산할
수 있는 가장 값진 artifact이기 때문입니다. 당신은 "Pipecat"과 "realtime_voice" 사이에서 고르는 게
아닙니다. 당신은 **열일곱 번** 고르는 것이고, 그 열일곱 개의 답은 서로 일치할 필요가 없습니다.
[[rtv-vs-pipecat-gap]] 자신의 guideline이 한 줄로 말합니다: *"Decide per-layer, not per-framework."*

그리고 그것이 *지금* 끝난 이유는 당신이 배울 mechanism을 다 써버렸기 때문입니다. 남은 미지수는 읽을
수 있는 것이 아니라 **측정할 수 있는** 것이고, 정확히 네 개이며, 그중 세 개는 이 머신에서 측정될 수
없습니다. §1.4가 넷 전부의 이름을 댑니다. §2는 측정 가능한 하나를 실제로 돌립니다.

그래서 이 chapter는 특이한 모양을 갖습니다. 이것은 서술(exposition)이 아니라 대부분 **ledger(원장)**
입니다. 서술은 이미 일어났습니다. 남은 것은:

1. row를 답할 수 있게 만드는 세 개의 rule (§1),
2. 당신이 실제로 돌리는 script 하나 — 그것이 출력하는 두 개의 숫자가 한 row를 pending에서 decided로
   바꿉니다 (§2),
3. vote cell이 비어 있는 열일곱 개의 evidence digest (§3),
4. 무엇을 읽어도 해소되지 않는 두 개의 collision (§4),
5. Pipecat에 갈 곳이 없는 것들과 Pipecat이 되돌려 주는 것들 — [[ch-09/read]]에서 여기로 라우팅된
   compaction 포함 (§5, §6),
6. deployment와 process topology — 이것은 row 17이고 후기(postscript)가 아닙니다 (§7),
7. deliverable과 watchlist (§9, §10).

당신의 가장 강한 mode는 framework extension입니다 — mechanism을 가져다가 그 저자가 고려하지 않은
자리에서 그것이 무슨 일을 할지 묻는 것. 열일곱 개의 row는 정확히 그것을 할 열일곱 번의 기회이고,
assumption cell이 그 extension이 들어갈 자리입니다. **옆에 assumption이 쓰이지 않은 vote는 vote의
옷을 입은 추측입니다.**

---

## 0. 이 chapter의 evidence를 읽는 법

[[ch-03/read]] §0과 같은 두 class이고, 하나가 추가됩니다.

| Class | Source of truth | 검증 방법 |
|---|---|---|
| **Pipecat claims** — 경로, 줄 번호, class 이름, count | commit `0cbf9c5b031eef06e53f0a193b9a67d60230e6be`의 `wiki/raw-data/pipecat/pipecat-src` | 파일을 여십시오. 아래의 모든 숫자는 2026-08-25에 그 tree에 대해 재측정되었고, 명령어가 옆에 인쇄되어 있습니다. |
| **boson-agent / realtime_voice claims** — LOC, class 이름, default | `wiki/raw-data/pipecat/excerpts/` 아래의 `rtv-*` / `boson-*` / `design-*` excerpt | 당신 자신의 repo에 대해 확인하십시오. 이 wiki에서는 확인 불가능하고, 여기의 어떤 것도 그렇지 않은 척하지 않습니다. |
| **Measured claims** — §2가 생산하는 P50/P95 | 당신 자신의 endpoint, 당신이 probe를 돌리는 그 순간 | 다시 돌리십시오. 이 course에서 당신이 만들기 전에는 존재하지 않았던 유일한 숫자입니다. |

**excerpt가 source와 불일치하는 곳에서는 source가 이기고, 저는 그렇게 말합니다.** 그런 불일치가 이
chapter에 세 개 실려 있고, 각각은 사용 지점에서 표시됩니다.

1. **Deepgram TTFS P99.** [[design-boson-rules-on-pipecat]] §3은 *"Pipecat's own STT TTFS P99
   reference is 0.45 s for Deepgram."*이라고 씁니다. 파일은 `0.35`라고 말합니다. §3.2와 §2.5에서
   수정합니다 — "Tier-2가 pre-LLM 예산의 절반을 대략 두 배로 만든다"는 비교가 이 값에 민감하기
   때문입니다.
2. **VAD chunk count.** [[rtv-vad-chunking]]은 *"`start_secs = 0.2` → 7 chunks @16 kHz."*라고 씁니다.
   `round(0.2 / (512/16000))`은 `round(6.25)` = **6**입니다. §3.1에서 수정합니다.
3. **realtime_voice WebRTC 크기.** [[rtv-webrtc-transport]]는 *"~960 LOC"*를 헤드라인으로 걸고, 그
   다음에 합계 **1,276**이 되는 파일 목록을 나열합니다. [[ch-05/read]] §10.1이 이것을 처음 잡아냈고,
   row 6은 1,276을 싣고 ~960을 unverified로 표시합니다.

네 번째 숫자는 틀렸다기보다 *물렁합니다(soft)*. [[rtv-vs-pipecat-gap]]은 Pipecat transport를 **12**개로
셉니다. `ls -d src/pipecat/transports/*/`는 **11**개 package를 반환하고, [[ch-05/read]] §2가 이것을
11 package / 13개 `BaseTransport` subclass / 그중 한 package(`whatsapp/`)는 0개로 해소했습니다.
row 6은 ch-05의 숫자를 씁니다.

**모든 boson row에 적용되는 caveat 하나.** boson snapshot의 날짜는 2026-07-29 (`realtime_voice`,
branch `voice-chat-dev`)와 2026-08-20 (`gateway`/`basement`, branch `lina-new-dental-dev`)입니다.
Pipecat tree는 2026-08-25입니다. 당신의 repo는 활발히 움직이고 있습니다. 이 ledger에 있는 모든 boson
숫자는 현재값이 아니라 **하한(floor)**입니다. 어떤 row의 vote가 boson LOC count에 달려 있다면, vote를
쓰기 전에 재측정하십시오 — `wc -l` 한 번이면 됩니다.

---

## 1. 세 개의 rule

### 1.1 rule이 존재하는 이유: 그것이 겨냥해 쓰인 failure mode

이 chapter가 대체한 draft는 row 2 (ASR)에 대해 대략 이런 것을 생산했습니다:

> *"Pipecat streaming STT를 adopt할지 여부는 8 kHz μ-law audio에서의 Korean word-error rate에
> 달려 있고, 그것은 측정되지 않았다. 결정 전에 benchmark할 것을 권고한다."*

그것을 하나의 work product로 읽으십시오. 참이고, 정직하고, 출처도 좋습니다 — 그리고 그것은
**결정이 아닙니다**. 틀릴 수가 없으므로 채점될 수 없고, 채점될 수 없으므로 실행에 옮길 수 없습니다.
그것을 열일곱 배 하면 capstone의 output은 아무도 돌릴 예산이 없는 benchmark 목록이 되고, 그건 output이
없는 것과 같습니다.

문제는 미지수(unknown)가 아닙니다. 문제는 미지수를 **falsifier**가 아니라 **blocker**로 취급하는
것입니다. 이 둘은 서로 다른 객체입니다:

| | Blocker | Falsifier |
|---|---|---|
| 결정에 대해 하는 일 | 결정을 미룬다 | 무엇이 그것을 뒤집을지를 기록한다 |
| 이번 분기에 ship하는 것 | 없음 | 잠정적으로(provisionally), 그 결정 |
| 틀렸을 때의 비용 | 무한(당신은 끝내 알지 못한다) | 유한(당신이 tripwire를 적어 두었다) |
| 누가 소유하는가 | 구성상 아무도 | 지명된 사람, precondition과 함께 |

당신이 지금껏 ship한 모든 실제 engineering 결정은 이런 식으로 내려졌습니다. 당신은
`OpenAICompatibleUnaryASR`을 쓰기 전에 Korean STT를 측정하지 않았습니다. 뭔가를 가정하고 ship했습니다.
빠져 있던 건 오직 그 가정을 적어 두는 일뿐이었습니다. rule 1과 rule 2는 딱 그것입니다.

> 💡 **쉬운 설명 — blocker와 falsifier가 정말 그렇게 다른가요?**
> 같은 문장 "Korean WER를 모른다"를 두 가지로 쓸 수 있습니다.
> blocker 버전: "모르니까 결정할 수 없다." → 달력에 아무 일도 안 생깁니다.
> falsifier 버전: "모르지만 X라고 가정하고 adopt에 vote한다. 우리 corpus에서 WER가 현행보다 나쁘게
> 나오면 이 vote는 뒤집힌다." → 결정은 이번 분기에 나가고, 뒤집을 조건이 문서에 남습니다.
> 정보량은 동일한데 **한쪽만 실행 가능**합니다. 그 차이가 이 chapter 전체의 설계 원리입니다.

### 1.2 RULE ONE — 모든 row는 COMMITMENT를 받으며, 그것은 명시적으로 진술된 assumption 아래 이뤄진다

"it depends"가 아닙니다. vote, 그리고 **"I am assuming X."**라는 문장.

형식은 고정되어 있고, 뒷부분은 optional이 아닙니다:

```
Row 7 (telephony serializer):  <vote>
  I am assuming: the Korean carrier we sign with speaks a Twilio-shaped
  bidirectional-media WebSocket JSON protocol, not a raw SIP/RTP leg.
```

assumption은 실제로 일을 하고 있습니다. 그것이 vote를 그 자리에 없었던 사람에게도 *reviewable*하게
만드는 것입니다: carrier가 SIP/RTP를 넘겨줄 거라고 생각하는 동료는 당신의 vote와 싸울 필요가 없고,
당신의 assumption과 싸우면 됩니다. 그게 훨씬 짧고 훨씬 생산적인 싸움입니다.

**무엇이 assumption으로 인정되는가.** 현재 검증되지 않았고, 만약 뒤집히면 vote를 바꿀 세계에 대한
진술. 세 가지 test:

- **세계에 대한 것**이어야 하지, 당신의 선호에 대한 것이면 안 됩니다. *"우리는 dependency가 적은 걸
  선호한다고 가정한다"*는 assumption이 아니라 취향입니다. *"Lina가 여섯 개 serializer 집합 안에 있는
  carrier 위에서 ship된다고 가정한다"*는 assumption입니다.
- **현재 검증되지 않은 것**이어야 합니다. 5분 안에 확인할 수 있다면, 확인하고 assumption을
  지우십시오. assumption은 싸게 알 수 없는 것을 위한 것이지, 찾아보기 귀찮았던 것을 위한 게 아닙니다.
- **load-bearing(하중을 받는 것)**이어야 합니다. assumption이 뒤집혀도 vote가 살아남는다면, 그
  assumption은 장식입니다. 지우고 진짜를 찾으십시오.

**row들은 assumption을 공유할 수 있고, 실제로 여럿이 그럴 것입니다.** "Lina는 두 분기 안에 telephony로
간다"는 row 1, 2, 3, 6, 7, 17에 대해 동시에 load-bearing입니다. 한 번 쓰고 여섯 번 참조하십시오.
그리고 당신이 방금 무엇을 알게 되었는지 알아채십시오: 열일곱 개 vote 중 여섯 개가 상관되어 있으므로,
그 하나의 assumption은 ledger 안의 다른 어떤 단일 항목보다도 검증 노력을 더 쏟을 가치가 있습니다.

### 1.3 RULE TWO — 모든 row는 쓰여진 FALSIFIER를 받는다

그 vote를 뒤엎을 **구체적인 observation**.

여기 그 모양이 있습니다. 당신의 어느 row도 유도하지 않도록 일부러 소유자 없는(unowned) 예시를
씁니다. *"row N에 대해 X를 adopt한다"* 형태의 vote는 이렇게 falsify됩니다:

> *우리 자신의 corpus에서, X는 오늘 우리가 돌리는 것보다 더 많은 false turn-start를 만들어낸다*

그리고 *"row N에 대해 Y를 keep한다"* 형태의 vote는 이렇게 falsify됩니다:

> *측정 결과, Y는 예산이 허용하는 것보다 비싸다*

두 문장의 공통점을 보십시오. **둘 다 선호가 아니라 OBSERVATION을 지명합니다.** 어느 쪽도 "팀이
싫어하면"이라거나 "보기 흉한 것으로 판명되면"이라고 말하지 않습니다. 각각은 당신이 볼 때 일어나거나
일어나지 않는 것이고, "본다"는 것은 정의된 행위입니다.

**이 rule에 이빨을 달아 주는 test: 어떤 observation으로도 falsify될 수 없는 row라면, 그 vote는 결정이
아니고 그 row는 잘못 진술된 것이다.** 이건 문체에 대한 불평이 아니라 진단이고, 치료법은 돌아가서 row를
다시 진술하는 것입니다. row가 이 test에 실패하는 두 가지 방식:

1. **row가 결정이 아니라 취향이다.** "Pipecat의 frame taxonomy가 더 우아하니까 adopt한다"에는
   falsifier가 없습니다. 우아함은 관측되지 않으니까요. 그 taxonomy가 *무엇을 하는지*를 중심으로 row를
   다시 진술하십시오 — [[ch-02/read]]가 숫자를 줬습니다: 136개 파일에 걸친 577개의
   `isinstance(frame, ...)` 지점. 그것은 frame type을 추가하는 데 드는 관측 가능한 비용이고, 이제 row는
   falsifiable해집니다("우리 port에는 새 frame class가 네 개보다 많이 필요하다").
2. **row가 사실 두 개의 row다.** 서로 다른 이유로 각각 그것을 뒤엎을 두 개의 서로 다른 observation이
   있어서 falsifier를 하나로 쓸 수 없다면, 당신은 두 개의 subsystem을 합쳐 놓은 것입니다. 쪼개십시오.
   compaction이 row 15이고 LLM loop가 row 10으로 따로 있는 이유가 정확히 그것입니다 —
   [[ch-09/read]]의 scope box가 그대로 말합니다: *"a different subsystem with different failure modes
   and a different trigger."*

**각 falsifier는 observation의 TYPE도 지명하고, 그 observation이 오늘 이용 가능한지도 표시합니다.**
이 ledger 안의 모든 것을 네 개의 type이 덮습니다:

| Type | 무엇인가 | 이 ledger 안의 예시 |
|---|---|---|
| **corpus** | 녹음된 Lina audio나 transcript를 replay해서 무언가를 센다 | 단독으로 등장하는 한국어 backchannel에서의 false turn-start (row 1) |
| **probe** | 살아 있는 endpoint에 대해 돌리는 script | §2의 Tier-2 P50/P95 (row 13) |
| **benchmark** | 서드파티 또는 자체 accuracy/latency harness | Korean 8 kHz μ-law WER (row 2) |
| **code assertion** | 주장이 거짓이면 실패하는 test | turn당 정확히 하나의 inference-triggering frame (row 10, 12) |

figure는 row별로 이 type을 요구하고, 오늘 이용 가능한지를 표시합니다. 그 마지막 flag가 정직한
부분입니다: 대부분은 이용 불가능하고, **지금 관측할 수 없는 falsifier도 여전히 falsifier입니다** —
그것은 precondition을 기다리는 tripwire입니다. §10이 그것들을 모읍니다.

### 1.4 RULE THREE — 정확히 하나의 measurement은 실제로 RUN된다

목록에 올리는 게 아닙니다. 돌리는 겁니다.

이 course는 아무도 가지고 있지 않은 네 개의 숫자를 생산했습니다. 각각을 얻는 데 드는 비용과 함께
여기 있습니다:

| # | 미지수 | 측정되기 전에 필요한 것 | 오늘 돌릴 수 있나? |
|---|---|---|---|
| 1 | 8 kHz μ-law에서의 Korean STT accuracy (WER) | Korean STT 계약, labelled Lina corpus, telephony-band audio | **아니오** |
| 2 | interruption-broadcast → `TranscriptionFrame` 간격 | 단독 한국어 backchannel의 8 kHz telephony audio 녹음, 그리고 돌아가는 Pipecat pipeline | **아니오** |
| 3 | **실제 endpoint에 대한 Tier-2 rule-evaluation latency** | model endpoint 하나와 text prompt 두 개 | **예** |
| 4 | transition-swallow 설계 아래에서 turn당 inference 하나 | 만들어진 `BosonRuleProcessor` prototype과 `FrameLogger` | **아니오** |

3번이 바로 그것이고, 정확히 한 가지 이유로 선택되었습니다: **telephony carrier도, Korean STT 계약도,
audio도 전혀 필요 없기 때문입니다.** 그것은 두 번의 chat completion입니다. 그것에 관한 나머지 전부 —
그것이 row 13을 결정한다는 것, [[ch-11/read]]의 예산에 먹인다는 것 — 는 참이지만 부차적입니다. 그것을
*그 measurement*으로 만드는 것은, 그것이 당신이 오늘 오후에 끝낼 수 있는 유일한 것이라는 사실입니다.

**그리고 그것은 실행 가능한 진짜 집을 가지며, 이것은 강한 요구조건입니다.** 이 chapter의 figure인
[`figures/migration-map.html`](figures/migration-map.html)은 `file://`에서 로드되는 self-contained
offline 페이지입니다. 그것은 model endpoint를 호출할 수 없습니다 — 설계상의 선택 때문이 아니라, 이
course의 figure contract가 모든 companion을 외부 요청 없는 self-contained로 만들기 때문이고, `file://`
페이지에는 cross-origin POST를 보낼 origin이 없기 때문입니다. **구조적으로 할 수 없는 일을 artifact에
할당하지 마십시오.** 그래서 probe는 형제 script입니다:

```
wiki/courses/pipecat/ch-13/tier2-probe.py
```

그리고 figure는 어떤 종류의 network I/O도 하지 않습니다. figure는 명령어를 보여주고, 붙여 넣은 두 개의
숫자를 받아, 렌더링합니다. 그 분리가 올바른 분리이고 일반화됩니다: **measurement은 process가 돌 수 있는
곳에 살고, visualisation은 browser가 렌더할 수 있는 곳에 삽니다.**

---

## 2. 하나의 measurement: `tier2-probe.py`

### 2.1 무엇을 측정하는가, 그리고 왜 그것이 측정해야 할 옳은 것인가

[[ch-12/read]]는 seam을 도출한 다음 청구서를 제시했습니다. [[design-boson-rules-on-pipecat]]에서
그대로:

> *"Tier 2 = the 2 LLM checks — **blocking, and this is the bill: ~250-400 ms of added pre-LLM
> latency on every turn** (one Qwen3.6-27B TTFT plus ~5 output tokens)."*

그 문장에서 여기 중요한 것은 두 가지입니다. 첫째, **그것은 추정치입니다.** 시계에서 읽은 값이 아니라
model의 기대 TTFT 위에서 한 산술입니다. 둘째, **row 13이 거기에 매달려 있고**, [[ch-11/read]]
waterfall의 "rule evaluation" 칸도 마찬가지인데 그 칸은 오늘 비어 있습니다.

두 개의 rule은 `intent_rules` (priority 30)와 `sentiment_tracker` (priority 10)이고, 둘 다 Lina의
`03-orchestrator` layer에서 `@check(..., mode="parallel", check_type="llm")`으로 찍혀 있습니다.
probe가 무엇을 재현해야 하는지를 결정하는 mechanism은 [[design-boson-rules-on-pipecat]] §1에 있습니다:

> *"parallel checks all run under one `asyncio.gather` (L74-80) and every non-continue result is
> kept"*

그리고 §3:

> *"`sentiment_tracker` fires concurrently under the same `gather`, so wall clock ≈ max, not sum."*

**그러므로 측정의 단위는 두 개의 순차적 completion이 아니라, 두 completion의 `asyncio.gather` 하나
입니다.** 그것들을 하나씩 차례로 재는 probe는 진실의 대략 두 배를 보고할 것입니다. 이것이 probe가
curl 명령에 대고 재는 스톱워치가 아니라 script인 이유 전체입니다.

> 💡 **쉬운 설명 — "wall clock ≈ max, not sum"**
> 두 요청이 각각 280 ms, 310 ms 걸린다고 합시다. 순차로 돌리면 590 ms입니다. 그런데 실제 boson은
> `asyncio.gather`로 동시에 던지므로 벽시계 시간은 둘 중 **느린 쪽**, 즉 310 ms에 가깝습니다.
> 그러니 "두 번 재서 더한다"는 probe는 590이라는, 운영에서 절대 관측되지 않을 숫자를 보고합니다.
> probe의 유일한 존재 이유가 "관측"인데 관측 대상이 틀린 셈이죠.

### 2.2 probe, 그리고 그것이 의도적으로 하는 네 가지

파일은 `ch-13/tier2-probe.py`, 149줄, 표준 라이브러리만 씁니다. 네 개의 설계 포인트가 있고, 각각은
당신이 다르게 선택할 수도 있었던 결정입니다:

**하나 — async HTTP client가 아니라 `asyncio.to_thread` 안의 `urllib.request`를 씁니다.** dependency가
없으므로 script는 Python 3.11이 도는 곳이면 어디서나 — venv를 만들지 않은 박스를 포함해 — 돕니다.
비용은 요청당 thread 하나인데, 동시 요청이 두 개면 그건 공짜이고, timing overhead는 수백 밀리초짜리
신호에 대해 마이크로초 단위입니다.

```python
# wiki/courses/pipecat/ch-13/tier2-probe.py L76-83
async def one_turn(endpoint, api_key, model, turn, timeout):
    """Wall clock of the parallel phase for one finished utterance, in ms."""
    started = time.perf_counter()
    await asyncio.gather(
        asyncio.to_thread(_post, endpoint, api_key, model, INTENT_SYSTEM, turn, timeout),
        asyncio.to_thread(_post, endpoint, api_key, model, SENTIMENT_SYSTEM, turn, timeout),
    )
    return (time.perf_counter() - started) * 1000.0
```

그것이 `rules/engine.py` L74-80의 `asyncio.gather`를 재현한 것입니다. 시계는 gather 앞에서 시작해
gather 뒤에서 멈추므로, 당신이 얻는 것은 정확히 boson의 Phase-1 parallel block이 차지하는 벽시계
window — pre-LLM critical path 위에 떨어지는 그것 — 입니다.

**둘 — prompt는 실제 것과 같은 모양이고, `max_tokens=16`입니다.** intent prompt는
`"Most recent turn (PRIMARY SIGNAL — evaluate against THESE)"` anchor와 `intent_matcher.py`
L205-271의 콤마 구분 index output contract를 실어 나릅니다. `temperature=0.1`은 `llm_config.py`
L20,34와 일치합니다. output 길이는 생각보다 더 중요합니다: 청구서가 *TTFT plus ~5 output tokens*이므로,
model에게 문단을 쓰게 허용하는 probe는 엉뚱한 것을 재게 됩니다.

**셋 — corpus는 실제 Lina user turn이고, 당신은 그것을 교체해야 합니다.** 여덟 개의 한국어 turn이
default로 함께 배포되어 script가 설정 없이 돌아갑니다. 그것들은 *모양*에서 대표성이 있습니다 —
backchannel 하나, 가격 질문 하나, 회피 하나, 동의 하나, DNC 하나 — 그러나 당신의 트래픽은 아닙니다.
`--corpus <file>`은 한 줄에 turn 하나를 받습니다. **당신의 것을 쓰십시오.** token 수가 TTFT를
좌우하고, 당신의 실제 turn이 이것들보다 길다면 당신의 P95는 default가 인쇄하는 것보다 높습니다.

**넷 — 시끄럽게 degrade하고, 숫자를 위조하도록 놔두지 않습니다.**

```python
# wiki/courses/pipecat/ch-13/tier2-probe.py L109-115 — the message text as printed
tier2-probe: UNMEASURED. The endpoint did not answer.
  endpoint : ...
  model    : ...
  failed on: iteration 1/40, turn '네 말씀하세요'
  reason   : URLError: <urlopen error [Errno 61] Connection refused>
Bring the endpoint up, or point TIER2_ENDPOINT at a reachable one, and
re-run. Do NOT paste an estimate into figures/migration-map.html — row 13
stays pending until this prints two real numbers.
```

exit code 1, stdout에는 아무것도 없습니다. 마지막 문장이 rule 3 연습 전체의 요점입니다: 이
measurement의 가치는 전적으로 그것이 observation이라는 데서 나옵니다. **붙여 넣은 추정치는 빈 칸보다
나쁩니다. 빈 칸은 정직하니까요.**

### 2.3 작은 N에서의 percentile, 공식보다 먼저 손으로

당신은 이것을 사만 번이 아니라 마흔 번 돌릴 것이고, percentile estimator들은 작은 N에서 문제가 되는
방식으로 서로 다릅니다. 한 번 손으로 해 보십시오.

정렬된 아홉 개의 sample, 단위는 밀리초:

```
 rank:    1     2     3     4     5     6     7     8     9
 value: 240   258   261   270   288   301   319   355   612
```

**nearest rank로 P50.** Rank = ⌈9 × 50/100⌉ = ⌈4.5⌉ = 5. 5번째 값: **288 ms**.

**nearest rank로 P95.** Rank = ⌈9 × 95/100⌉ = ⌈8.55⌉ = 9. 9번째 값: **612 ms**.

방금 무슨 일이 일어났는지 보십시오. sample이 아홉 개일 때 **P95는 최댓값입니다** — 보간해 갈 9.5번째
sample 같은 건 없으므로, estimator는 당신이 본 최악의 것만 반환할 수 있습니다. 그 612 ms outlier 하나가
(cold cache, scheduling 딸꾹질, 재시도된 connection) *곧* 당신의 P95입니다. 공식만 남기면:

$$\text{rank} = \left\lceil \frac{N \cdot q}{100} \right\rceil, \qquad P_q = x_{(\text{rank})}$$

여기서 `x`는 정렬된 sample입니다. script에서는:

```python
# wiki/courses/pipecat/ch-13/tier2-probe.py L86-90
def percentile(samples, q):
    """Nearest-rank percentile — honest for the small N this probe produces."""
    ordered = sorted(samples)
    rank = max(1, min(len(ordered), -(-len(ordered) * q // 100)))
    return ordered[int(rank) - 1]
```

`-(-a // b)`는 정수 ceiling division입니다. 선형 보간 대신 nearest rank가 일부러 선택되었습니다.
보간은 당신이 실제로 본 두 sample 사이의 값을 발명하는데, 그건 마흔 개의 sample이 지지하는 것보다 더
높은 정밀도처럼 읽히기 때문입니다.

**실무적 귀결: 최소 40회 iteration을 돌리십시오.** N=40에서 P95는 rank 38 — 최악이 아니라 세 번째로
나쁜 sample — 이므로 outlier 하나가 더 이상 당신의 P95가 *되지* 않습니다. N=20에서 P95는 rank 19,
두 번째로 나쁜 것입니다. 대략 20 아래로 내려가면 그 숫자는 당신의 tail이 아니라 당신의 최악의 운에
대한 기술(description)입니다.

> 💡 **쉬운 설명 — nearest rank가 왜 "정직한" 선택인가요?**
> 보간(interpolation) 방식은 rank 8.55 같은 소수 위치를 계산해서 355와 612 사이 어딘가의 값,
> 예컨대 500 ms를 만들어 냅니다. 그런데 500 ms는 **당신이 한 번도 관측하지 않은 숫자**입니다.
> 이 chapter의 요점이 "vote는 observation에 근거해야 한다"인데, percentile 계산에서만 없는 값을
> 지어내는 건 앞뒤가 안 맞습니다. nearest rank는 항상 실제로 본 sample 중 하나를 반환합니다.

### 2.4 실행하기

```bash
$ cd wiki/courses/pipecat/ch-13

$ export TIER2_ENDPOINT="http://localhost:8000/v1/chat/completions"
$ export TIER2_MODEL="Qwen3.6-27B-FP8"
$ export TIER2_API_KEY="..."           # omit if your endpoint is unauthenticated

$ python3 tier2-probe.py --iterations 40
tier2-probe: 40 turns against Qwen3.6-27B-FP8 at http://localhost:8000/v1/chat/completions   [min 231.4 / max 601.8 ms]
  P50     288.7 ms
  P95     441.2 ms
Paste P50 and P95 into the MEASUREMENT panel of figures/migration-map.html.
```

positional 형식도 동작하므로 notebook의 한 줄이나 CI step에서 쓸 수 있습니다:

```bash
$ python3 tier2-probe.py http://localhost:8000/v1/chat/completions Qwen3.6-27B-FP8 40
$ python3 tier2-probe.py --corpus ./lina-turns.txt --verbose --iterations 60
```

`--verbose`는 turn별 timing을 **stderr**로 인쇄하므로 stdout은 정확히 네 줄로 유지되고 pipe 가능한
상태로 남습니다. 위의 숫자들은 결과가 아니라 형식을 보여주기 위한 예시입니다 — 결과는 당신의
endpoint가 인쇄하는 무엇이든이고, 그게 요점 전부입니다.

> **⚠️ 위 block의 두 숫자는 output 모양을 보여주려고 지어낸 것입니다.** figure에 붙여 넣지 마십시오.
> row 13은 *당신의* endpoint가 *당신의* 두 숫자를 인쇄할 때까지 pending으로 남습니다.

### 2.5 그 숫자가 무엇을 결정하는가, 양방향으로

P50과 P95를 [`figures/migration-map.html`](figures/migration-map.html)의 MEASUREMENT 패널에 붙여
넣으십시오. 두 필드가 모두 채워지기 전까지 패널은 회색으로 처리되고 실행 지시만 보여 줍니다 — 명시적인
UNMEASURED 상태여서, 돌리지 않은 probe가 측정된 0으로 오해될 수 없습니다. 채워지고 나면 ch-12의
250–400 ms 추정치를 뒤쪽의 ghost bar로 두고 당신의 측정 bar를 그리며, 측정값을 [[ch-11/read]]
waterfall의 비어 있던 "rule evaluation" 칸에 써 넣어 전체 예산이 진짜 숫자와 함께 다시 렌더링되게
합니다. 두 값 모두 `localStorage`에 남으므로, reload해도 이 capstone이 채점되는 그 하나의 measurement이
버려지지 않습니다.

그런 다음 결과를 읽으십시오. 결과는 셋이고 셋 다 결정적입니다:

**250–400 ms 안쪽.** ch-12의 추정치가 성립했습니다. row 13의 latency 항은 설계가 가정한 그대로이고,
in-turn-veto 대 next-turn-transition의 교환은 진술된 대로 가격이 매겨졌으며, 이 row는 예산이 아니라
*구조적* 질문(boson의 얼마나 많은 부분이 수정 없이 살아남는가)으로 넘어갑니다. row 13에 대한 당신의
falsifier는 열린 질문이 아니라 재측정 트리거 — "model이나 serving stack이 바뀌면 다시 돌린다" — 가
됩니다.

**250 ms 아래.** 추정치가 비관적이었습니다. 그 여유를 쓰기 전에 어디서 왔는지 기록하십시오: 데워진
KV cache, quantise된 checkpoint, network hop이 없는 로컬 endpoint 모두 실재하고 각각 다른 방식으로
취약합니다. production 경로가 probe와 같은 hop 수를 갖는지 구체적으로 물으십시오. probe는
`TIER2_ENDPOINT`가 가리키는 무엇에든 말을 거는데, 그게 `localhost`라면 당신은 당신이 운영하지 않는
deployment를 측정한 것입니다.

**400 ms 위.** 추정치가 낙관적이었고, [[ch-11/read]] 예산 항목 중 하나가 이제 당신이 이름 붙일 수
있는 양만큼 초과 인출되었습니다. §0의 수정이 값을 하는 지점이 여기입니다. 설계 excerpt는 Tier-2를
*"Pipecat's own STT TTFS P99 reference is 0.45 s for Deepgram"*과 비교했지만 — 파일은 이렇게
말합니다:

```python
# src/pipecat/services/stt_latency.py L38, L45, L61-62
DEFAULT_TTFS_P99: float = 1.0
...
DEEPGRAM_TTFS_P99: float = 0.35
...
SONIOX_TTFS_P99: float = 0.35
SPEECHMATICS_TTFS_P99: float = 0.74
```

`0.45`가 아니라 `0.35`입니다. 표 전체는 `0.35`(Deepgram, Deepgram-SageMaker, Soniox)부터 `2.14`(xAI)에
걸친 23개의 측정된 `*_TTFS_P99` 상수에, `DEFAULT_TTFS_P99 = 1.0`과 그것에 alias되는 두 개의
서비스(`NVIDIA_TTFS_P99`, `WHISPER_TTFS_P99`)를 더한 것입니다. 이 수정은 비교를 무디게 하는 게 아니라
날카롭게 합니다: 0.35 s 기준선에 대해 400 ms짜리 Tier-2 측정치는 tree에서 가장 빠른 provider의 STT P99
**전체**보다 크고, "Tier-2가 pre-LLM 예산의 절반을 대략 두 배로 만든다"는 표현은 오히려 과소평가입니다.

그리고 `stt_latency.py`가 **말하지 않는** 것도 주목하십시오. [[stt-korean-providers]]에서: 그것은
*"records only latency, is silent on the language and sample rate of the benchmark audio."* 그러므로
0.35 s는 영어를 가정한 숫자입니다. row 2의 falsifier는 그 문장 안에 삽니다.

---

## 3. 열일곱 개의 row

**이 section을 쓰는 법.** 각 row는 양쪽이 무엇을 구현하는지, 그리고 어느 chapter가 그것을 확립했는지를
진술합니다. **vote cell은 비어 있습니다. assumption cell도 비어 있습니다. falsifier cell도 비어
있습니다.** 셋 다 figure에서 당신의 몫입니다. 산문은 evidence에서 멈춥니다 — 의도적으로. rule 1이
당신이 commit할 것을 요구하는데, 당신 대신 commit하는 chapter는 이 course의 유일한 채점 대상 연습을
취소해 버린 것이기 때문입니다.

한눈에 보는 ledger입니다. 열일곱 개의 row, figure가 렌더링하는 순서대로.

| # | Subsystem | Evidence from | Vote | Assumption | Falsifier |
|---|---|---|---|---|---|
| 1 | VAD | [[ch-06/read]] | | | |
| 2 | ASR / streaming STT | [[ch-03/read]], [[ch-06/read]], [[ch-11/read]] | | | |
| 3 | TTS + Korean word timestamps | [[ch-07/read]] | | | |
| 4 | `KoreanPhraseChunker` | [[ch-03/read]], [[ch-07/read]] | | | |
| 5 | `AudioTextPlayoutLedger` | [[ch-03/read]], [[ch-07/read]], [[ch-08/read]] | | | |
| 6 | Transport | [[ch-05/read]] | | | |
| 7 | Telephony serializer | [[ch-05/read]], [[ch-06/read]] | | | |
| 8 | Session auth (`WebRTCSessionManager`) | [[ch-03/read]], [[ch-05/read]] | | | |
| 9 | Control protocol (`ControlEvent`) | [[ch-03/read]], [[ch-05/read]] | | | |
| 10 | The LLM loop | [[ch-09/read]] | | | |
| 11 | Tools | [[ch-09/read]] | | | |
| 12 | The stage machine | [[ch-10/read]] | | | |
| 13 | The rule layers | [[ch-12/read]] + §2's probe | | | |
| 14 | `ScriptEngine` | [[ch-12/read]] | | | |
| 15 | Compaction | [[boson-compact-session]] | | | |
| 16 | Observability | [[ch-11/read]] | | | |
| 17 | Deployment and process topology | [[ch-04/read]] + §7 | | | |

열일곱. figure도 열일곱을 렌더링합니다. 어느 쪽에서든 열여섯이나 열여덟을 세게 된다면, 둘 중 하나가
표류(drift)한 것이고 그 ledger는 더 이상 ledger가 아닙니다.

---

### 3.1 Row 1 — VAD

[[ch-06/read]]가 두 mechanism을 전부 공급했습니다.

**Pipecat의 analyzer.** 네 개의 state, 그리고 volume gate와 AND된 confidence gate:

```python
# src/pipecat/audio/vad/vad_analyzer.py L25-28
VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6
```

```python
# src/pipecat/audio/vad/vad_analyzer.py L41-44
    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4
```

```python
# src/pipecat/audio/vad/vad_analyzer.py L206-232
            confidence = self.voice_confidence(audio_frames)

            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume

            if speaking:
                match self._vad_state:
                    case VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._vad_starting_count = 1
                    case VADState.STARTING:
                        self._vad_starting_count += 1
                    case VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
            else:
                match self._vad_state:
                    case VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._vad_starting_count = 0
                    case VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._vad_stopping_count = 1
                    case VADState.STOPPING:
                        self._vad_stopping_count += 1
```

226번 줄의 `STARTING → QUIET` 갈래를 붙들고 계십시오: `STARTING`에 들어갔다가 지속되지 못한 frame은
speech-start를 한 번도 emit하지 않은 채 `QUIET`으로 되돌아갑니다.

**네이티브 8 kHz, 그리고 두 sample rate에서 동일한 chunk count.** frame 산술:

```python
# src/pipecat/audio/vad/vad_analyzer.py L159-165
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        vad_frames_per_sec = self._vad_frames / self.sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
```

```python
# src/pipecat/audio/vad/silero.py L191-197
    def num_frames_required(self) -> int:
        """Get the number of audio frames required for VAD analysis.

        Returns:
            Number of frames required (512 for 16kHz, 256 for 8kHz).
        """
        return 512 if self.sample_rate == 16000 else 256
```

16 kHz에서: chunk당 `512/16000 = 0.032` s, `round(0.2/0.032) = round(6.25) = 6`. 8 kHz에서:
chunk당 `256/8000 = 0.032` s — **동일**, 그래서 다시 `6`. **endpointing latency는 구성상 sample
rate에 불변입니다.** **⚠️ Source correction:** [[rtv-vad-chunking]]은 *"7 chunks @16 kHz"*라고
말합니다. Python에서 `round(6.25)`는 `6`입니다 (banker's rounding은 여기서 무관합니다 — 6.25는 어떤
관례로도 내림입니다). 값은 **여섯**입니다.

**Idle timeout.** `audio_idle_timeout: float = 1.0` (`llm_response_universal.py:170`,
`processors/audio/vad_processor.py:46`에도)이 audio가 아예 도착하지 않을 때 speech stop을 강제합니다.

**realtime_voice의 analyzer**, [[rtv-vad-chunking]]에 따르면: `SileroVADConfig.threshold = 0.5`,
`self._speaking` bool 하나(두 개의 state), 초가 아니라 *transport 크기의 frame* 단위로 세는
`min_speech_frames = 2` / `min_silence_frames = 6`, volume gate 없음, idle timeout 없음, 그리고
L58에서 `ValueError("SileroVAD requires 16 kHz mono PCM")`을 raise하는 `SileroVAD.process`.
`EnergyVADConfig`가 형제입니다: `speech_rms = 500.0`, 순수 Python RMS, *"intended for fallback and
deterministic tests."*라고 문서화되어 있습니다.

**two-frame blip, mechanism으로서.** 두-state machine에서 `min_speech_frames = 2`에 도달하는 2-frame
noise burst는 `SPEECH_STARTED`를 emit하고, 그것은 `VoiceSession._on_speech_started` (L284)에서
generation을 진전시키고 assistant를 취소합니다. 같은 burst가 네-state machine에서는 `STARTING`에
들어갔다가 `_vad_starting_count >= 6`에 실패하고, 226번 줄에 의해 아무것도 emit하지 않은 채 `QUIET`으로
되돌려집니다. [[rtv-vad-chunking]]은 이것을 "the sharpest correctness delta in the whole comparison"
이라고 부릅니다 — **그것은 excerpt의 판정이고 여기서 재현되지 않습니다.** mechanism은 이것입니다:
두 machine은 짧은 burst에서 갈라지고, 갈라지는 방향은 한쪽은 emit하고 다른 쪽은 하지 않는다는 것입니다.

**이 row가 무엇에 달려 있는가.** Lina의 실제 audio에서 단독 짧은 burst가 얼마나 자주 일어나는지, 그리고
8 kHz의 sample rate가 각 machine에 무슨 일을 하는지.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.2 Row 2 — ASR / streaming STT

세 개의 chapter가 기여했고, 서로 다른 종류의 fact를 기여했습니다.

**[[ch-03/read]]는 오늘 돌아가는 것의 모양을 공급했습니다.** [[rtv-vs-pipecat-gap]]에 따르면
`OpenAICompatibleUnaryASR`은 *"buffers the whole utterance into a WAV and does one
`audio.transcriptions.create` at `finalize()` (`openai_compat.py` L194-242,
`timeout_seconds=1.5`)"*이고, `ASREventKind.INTERIM` / `END_OF_TURN`은 *"declared in `types.py` but
never emitted by any real provider — only by a test fake."*입니다.

**[[ch-06/read]]는 Pipecat 쪽을 공급했습니다**: streaming `STTService` interface, 약 20개의 STT
provider, smart-turn default, 그리고 latency 표:

```python
# src/pipecat/services/stt_latency.py L38-46
DEFAULT_TTFS_P99: float = 1.0

# Measured P99 TTFS latency values (in seconds)
ASSEMBLYAI_TTFS_P99: float = 0.42
AWS_TRANSCRIBE_TTFS_P99: float = 1.90
AZURE_TTFS_P99: float = 1.80
CARTESIA_TTFS_P99: float = 0.81
DEEPGRAM_TTFS_P99: float = 0.35
DEEPGRAM_SAGEMAKER_TTFS_P99: float = 0.35
```

23개의 측정된 상수, `0.35` → `2.14`. **⚠️ Source correction, §2.5에서 반복 — 이 row에도 떨어지기
때문입니다:** [[design-boson-rules-on-pipecat]]은 Deepgram을 `0.45`로 인용합니다. 파일은 `0.35`라고
말합니다.

**[[ch-11/read]]는 산술을 공급했습니다.** 그것은 unary transcription RTT를 critical path 위, VAD stop
*뒤에 통째로* 놓습니다 — 겹쳐지는 항이 아니라 직렬 항입니다. 그것을 [[rtv-vs-pipecat-gap]]에 인용된
`CLAUDE.md` 자신의 문장에 대고 재면: *"P50 at or below 1.0 seconds and P95 at or below 1.5 seconds,"*
그리고 그것은 *"from the last voiced user sample to the first audible assistant sample, including
end-of-turn/VAD time."*으로 측정됩니다.

**Korean-verified shortlist**, [[stt-korean-providers]]에서 — 자신의 map에 `Language.KO`를 가진
서비스들을 위 P99 순서로: Soniox `"ko"` (0.35), Speechmatics `"ko"` (0.74), Gladia `"ko"` (1.49),
Google `"ko-KR"` (1.57), Azure `"ko-KR"` (1.80), AWS Transcribe `"ko-KR"` (1.90), ElevenLabs `"kor"`
— 세 글자 — (2.01), xAI `"ko"` (2.14), Fal `"ko"` (2.07), 그리고 로컬 Whisper / Moonshine / FunASR.
그 표에서 Soniox와 나란히 `0.35`에 앉아 있는 Deepgram은 **`LANGUAGE_MAP` 자체가 없습니다**.
`Language.KO`는 `"ko"`로 직렬화될 것이고, 그것이 받아들여지는지에 대해 repo는 아무 입장도 취하지
않습니다. AssemblyAI의 map은 열여덟 개 언어를 나열하고 한국어는 그 안에 없습니다. Sarvam의 유일한
`KO*` 항목은 `Language.KOK_IN` — Konkani입니다.

**부재(absence), 있는 그대로.** [[stt-korean-providers]], `src/` 전역 grep으로 검증:

> *"No measured Korean accuracy number, and no Korean-on-8 kHz-telephony number of any kind, exists
> anywhere in this repository at this commit. … `WER` / `word error rate` / accuracy claims → zero
> hits, for any service, any language."*

tree 안의 유일한 `8000` 값들은 telephony serializer의 default입니다. **당신은 이 숫자를 읽어서 얻지
못합니다.** 그것은 §1.4의 미지수 #1이고, *benchmark* type의 falsifier이며, precondition은 Korean STT
계약과 labelled Lina corpus입니다.

**이 row가 무엇에 달려 있는가.** streaming interface가 예산에서 직렬 항 하나를 의미 있을 만큼 제거해
주는지, 그리고 telephony-band audio의 한국어에서 그 교체의 accuracy 비용이 얼마인지.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.3 Row 3 — TTS, 그리고 여섯 서비스짜리 Korean word-timestamp 교집합

[[ch-07/read]]가 [[tts-korean-providers]]로부터 전부 공급했습니다.

**열두 개의 서비스가 `Language.KO`를 map합니다** — Cartesia, ElevenLabs, Azure
(`azure/common.py:200-201`을 통해), Google, Inworld, Soniox, xAI, LMNT, MiniMax, AWS Polly, Camb,
XTTS — grep으로 검증:

```bash
$ grep -rl "Language.KO\b" src/pipecat/services/*/tts.py src/pipecat/services/azure/common.py
```

**열둘 중 여섯은 word timestamp도 emit합니다.** `add_word_timestamps` 호출자 전체 목록은 azure,
cartesia, elevenlabs, elevenlabs/dialogue, gradium, hume, inworld, resembleai, rime, smallest,
soniox, speechify, xai입니다. 한국어 열둘과 교집합을 취하면 **azure, cartesia, elevenlabs, inworld,
soniox, xai**가 나옵니다. 이것들이 barge-in이 assistant context를 마지막 *생성된* token이 아니라
마지막 *발화된* 단어에서 `TTSTextFrame.pts`를 통해 잘라낼 수 있는 서비스들입니다.

**⚠️ provenance caveat, 이 row에서 가장 중요한 문장.** 그 여섯은 **선언된 capability에 대한 grep**
이지 행동 검증이 아닙니다. 이 course의 누구도 그 어느 것으로부터 한국어 word timestamp가 도착하는 것을
지켜본 적이 없습니다. 선언은 그 서비스가 `add_word_timestamps`를 호출한다고 말하지, provider가 한국어
어절에 대해 쓸 만한 경계를 반환한다고 말하지 않습니다. "여섯"을 검증된 집합이 아니라 후보의 상한으로
취급하십시오.

**`resolve_language`는 map되지 않은 언어에 대해 실패하지 않습니다:**

```python
# src/pipecat/transcriptions/language.py L614-629
    # Not in map - fall back with warning
    lang_str = str(language)

    if use_base_code:
        # Extract base code (e.g., "en" from "en-US")
        base_code = lang_str.split("-")[0].lower()
        logger.warning(f"Language {language} not verified. Using base code '{base_code}'.")
        return base_code
    else:
        logger.warning(f"Language {language} not verified. Using '{lang_str}'.")
        return lang_str
```

`logger.warning` 하나와 return 하나. Rime은 — word-timestamp 능력이 *있어서* 가장 매혹적인 오답이
되는데 — 정확히 다섯 개 언어(`ger/fra/eng/spa/hin`)만 map하고 `use_base_code=False`로 호출되므로,
`Language.KO`는 그것을 모르는 API로 리터럴 `"ko"`로 나갑니다. 아무것도 raise하지 않습니다. 하류의
포착 장치는 `max_consecutive_zero_audio_contexts: int = 3` (`tts_service.py:168`)인데, 이는 연속된 세
개의 context가 zero audio를 내놓은 뒤에야 걸립니다.

**tree에 유지 관리되는 self-hosted Korean TTS는 없습니다.** 한국어 mapping을 가진 유일한 로컬 서비스는
`XTTSService`인데 1.7.0부터 *"No replacement"*와 함께 `@deprecated`입니다. 유지되는 두 로컬 서비스인
`KokoroTTSService`와 `PiperTTSService`는 한국어를 전혀 map하지 않습니다. 한국어 네이티브 벤더 디렉터리도
없습니다 — Typecast도, Supertone도, Naver Clova도 없습니다. **on-prem Korean TTS는 선택지가 아니라
부재입니다**, 누군가 `TTSService` subclass를 쓰지 않는 한.

**그리고 realtime_voice 쪽:** [[rtv-vs-pipecat-gap]]은 `OpenAICompatibleStreamingTTS`
(`with_streaming_response`, `chunk_size=1024`, 24 kHz PCM,
`extra_body={"temperature":0.7,"num_timesteps":4}`)를 기록합니다 — 사내 provider 하나, streaming,
그리고 boson에는 TTS 쪽 word timestamp가 없다는 excerpt 자신의 note.

**이 row가 무엇에 달려 있는가.** word timestamp를 emit하는 한국어 provider에 당신의 조건으로 접근
가능한지 여부. row 5와 row 3이 정확히 그것을 통해 결합되어 있기 때문입니다.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.4 Row 4 — `KoreanPhraseChunker`

**[[ch-03/read]]가 알고리즘을 공급했습니다.** [[rtv-vad-chunking]]에 따르면 283줄: 1→2→tail 적응형
batching 스케줄(`_batch_phase` 0 → time-to-first-audio를 위한 단일 문장, 1 → 쌍, 2 → 경계 있는 tail),
`1.5`, `...`, 그리고 `gpt-4.1` 같은 ASCII identifier와 hostname을 쪼개기를 거부하는
`_is_safe_period` (L255), `1,000`을 보호하는 `_is_numeric_separator` (L277), 그리고 발화 텍스트에서
Gateway control tag를 벗겨 내면서 `start_char`/`end_char`는 *원본* span을 온전히 유지하는
`_INTERNAL_TAG`. default는 `min_chars=12, max_chars=60, batch_max_chars=320`이고,
`hard_max_chars`는 `min(batch_max_chars, max_chars * 2)`로 결정됩니다.

**[[ch-07/read]]는 Pipecat이 그 자리에 무엇을 제공하는지 공급했습니다**: `SimpleTextAggregator`와
`utils/string.py`의 문장 matcher.

```python
# src/pipecat/utils/string.py L118-127
# Latin punctuation that NLTK handles well — these need NLTK's disambiguation
# because "." can appear in abbreviations, decimals, etc.
_LATIN_SENTENCE_ENDING_PUNCTUATION: frozenset[str] = frozenset({".", "!", "?", ";", "…"})

# Non-Latin sentence-ending punctuation that is always unambiguous and never needs
# NLTK's disambiguation logic. Used as a fallback when NLTK doesn't support the
# language (e.g., Japanese, Chinese, Korean, Hindi, Arabic).
UNAMBIGUOUS_SENTENCE_ENDING_PUNCTUATION: frozenset[str] = (
    SENTENCE_ENDING_PUNCTUATION - _LATIN_SENTENCE_ENDING_PUNCTUATION
)
```

그 주석에 한국어가 이름으로 등장하고, 한국어가 지나가는 경로는 이렇습니다:

```python
# src/pipecat/utils/string.py L183-194
    if len(sentences) == 1 and first_sentence == text:
        if text and text[-1] in SENTENCE_ENDING_PUNCTUATION:
            return len(text)
        # Fallback for languages not supported by NLTK (e.g., Japanese, Chinese,
        # Korean, Hindi, Arabic). NLTK returned the entire text as a single
        # sentence, and the last character is not sentence-ending punctuation
        # (it's a lookahead character). Scan for unambiguous non-Latin sentence-
        # ending punctuation that doesn't need NLTK's disambiguation.
        for i, ch in enumerate(text):
            if ch in UNAMBIGUOUS_SENTENCE_ENDING_PUNCTUATION:
                return i + 1
        return 0
```

첫 번째 비-Latin 종결 문자를 찾는 선형 스캔입니다. 소수점 guard도, identifier guard도, 천 단위 구분자
guard도, batching 스케줄도, tag 제거도 없습니다. 그리고 한국어는 CJK word-grouping 경로에서
*제외*되어 있다는 점도 주목하십시오 — Cartesia는 `base_lang in {"zh", "ja"}`를 test하고
(`cartesia/tts.py:454-455`), ElevenLabs도 `tts_base.py:329`에서 같은 일을 하므로, 한국어는 일반적인
공백 구분 갈래로 떨어집니다. [[tts-korean-providers]]는 그것이 무슨 뜻인지에 대해 조심스럽습니다:
*"That matches Korean 어절 spacing — but it is an untested assumption in this code, not a verified
Korean path."*

**이 row가 무엇에 달려 있는가.** 그 guard들이 Lina의 실제 텍스트가 필요로 하는 지식을 encode하고
있는지 여부. 그것은 corpus 질문이고, 당신의 transcript 아카이브로 이번 주에 답할 수 있는 질문입니다.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.5 Row 5 — `AudioTextPlayoutLedger`

**[[ch-03/read]]가 두 method를 공급했습니다.** [[rtv-vad-chunking]]에서: `audible_text()` (L74)는
client cursor까지 phrase들을 훑고, 부분적으로 재생된 phrase에 대해서는
`ratio = (cursor - sample_start) / (sample_end - sample_start)`를 계산한 뒤
`text[:int(len(text) * ratio)]`를 취합니다 — *"a linear character-per-sample approximation, not a
word-timestamp alignment."* `playout_complete()` (L98) — 모든 phrase가 `complete`이고 **그리고**
`played_sample >= queued_samples` — 이 `_cancel_generation`으로 하여금 `session.py` L502-507의
`semantic_interrupt` flag를 통해 "고객이 나를 끊었다"와 "내가 끝냈고 그들이 응답했다"를 구별하게
해 주는 것입니다. `acknowledge()`는 `max(current, played_sample)`로 cursor를 옮기므로, 늦게 도착한
ack가 cursor를 되감을 수 없습니다.

**[[ch-08/read]]가 Pipecat의 대안과 그 gap을 공급했습니다.** Pipecat은 같은 보장을 *위치적으로*
얻습니다: assistant aggregator가 `transport.output()` **뒤에** 앉아 있으므로, 그것은 word-timestamp가
찍힌 `TTSTextFrame`에 의해 pacing되어 실제로 방출된 텍스트만 봅니다. [[rtv-vs-pipecat-gap]]은 그 비용을
기록합니다: 그 대안은 *"Emergent, not explicit"*이고 *"No `[interrupted]` marker written"* — untagged-
partial gap입니다. history에는 잘린 assistant turn이 들어 있는데, 그것이 잘렸다는 표시가 아무것도
없습니다.

**[[ch-07/read]]가 결합(coupling)을 공급했습니다.** 위치적 대안은 선택된 TTS가 word timestamp를
emit할 때에만 사용 가능한데, 그것이 row 3의 여섯 서비스 교집합이고, 그것은 행동 test가 아니라
grep입니다. provider가 아무것도 emit하지 않으면 위치적 mechanism은 pacing할 대상이 없습니다.
[[rtv-vad-chunking]]은 그 의존성을 직접 진술합니다: `AudioTextPlayoutLedger`는 *"would become
redundant only if the chosen TTS emits word timestamps."*

**accuracy 교환, [[rtv-vs-pipecat-gap]]에서, mechanism으로 진술하면:** ledger는 *"more accurate on
paper (works with timestamp-less TTS) and less accurate mid-word (linear char/sample
approximation)."* 그 문장의 두 절반은 동시에 참이고, 서로 다른 failure mode에 관한 것입니다.

**이 row는 row 3과 결합되어 있고 figure가 그것을 강제합니다.** 여섯 바깥의 TTS를 선택하면서
`AudioTextPlayoutLedger`를 버리는 것은 양립 불가능한 조합으로 표시되며, 그 여섯이 grep이라는 리마인더가
따라붙습니다.

**이 row가 무엇에 달려 있는가.** row 3이 어느 TTS에 안착하는지, 그리고 history에 표시 없이 잘린 turn이
Lina의 하류 소비자들에게 문제가 되는지 여부 (compaction summariser가 그 history를 읽고, 당신의 평가
세트도 그렇습니다).

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.6 Row 6 — Transport

**[[ch-05/read]]가 이 row 전체를 공급했습니다.** 두 구현 모두 aiortc의 `RTCPeerConnection`을 감싸고,
둘 다 PyAV로 resample하고, 둘 다 data channel을 구동하고, 둘 다 outbound audio를 벽시계에 맞춰
pacing합니다.

**크기.** Pipecat의 `transports/smallwebrtc/`는 `transport.py` (1,085), `connection.py` (825),
`request_handler.py` (266)에 걸쳐 **2,176**줄입니다 — `wc -l`로 확인 가능합니다. realtime_voice의
`transport/webrtc/`는 자신의 파일별 목록으로 **1,276**줄입니다 (`manager.py` 248, `control.py` 226,
`peer.py` 231, `tracks.py` 216, `buffer.py` 123, `config.py` 64, `transport.py` 168).

**⚠️ Source correction.** [[rtv-webrtc-transport]]는 *"~960 LOC"*를 헤드라인으로 걸고 나서 합계
1,276이 되는 파일들을 나열합니다 — 자기 표가 자기 헤드라인을 316줄, 약 33% 만큼 반박합니다.
[[ch-05/read]] §10.1이 이것을 표시했지만 해소할 수는 없었습니다 (repo가 이 머신에 없고 rule 3이 그것을
여는 것을 금지합니다). **1,276을 쓰고, ~960은 unverified로 취급하십시오.** 이 숫자로 migration 규모를
잡고 있다면, 그 차이는 일주일입니다.

**폭(breadth), 센 값.** `ls -d src/pipecat/transports/*/ | wc -l` → **11** package;
`grep -rn "(BaseTransport)"` → **13** subclass — `local/`이 2개, `websocket/`이 3개를 배포하기
때문입니다. `whatsapp/`은 **0**개를 배포하고 `SmallWebRTCConnection` 위의 signalling adapter입니다.
[[ch-05/read]] §2가 그 회계를 했습니다. **⚠️** [[rtv-vs-pipecat-gap]]은 12라고 말하고, tree는 11
package라고 말합니다.

**한쪽에만 있는 두 개의 mechanism.** `SmallWebRTCConnection`은 `renegotiate(sdp, type,
restart_pc=False)` (L443), `ask_to_renegotiate()` (L799), `pc_id` (L302), 그리고 *"aiortc does not
provide any way so we can be aware when we are disconnected"* (L350)라는 이유로 손수 작성된
`"disconnected"` handler를 가집니다. realtime_voice의 유일한 복구 경로는 새로운
`accept_offer(reconnect=True)`입니다. Pipecat 쪽에만 있는 것 또 하나: video (`RawVideoTrack`)와 화면
공유인데, 둘 다 Lina TMR은 쓰지 않습니다.

**반대쪽에만 있는 두 개의 mechanism** — 그리고 그것들은 row 8과 row 9이고, transport 배관과 분리
가능하다는 바로 그 이유로 따로 떼어졌습니다.

**입도(granularity) 차이, [[ch-05/read]] §10.3에서.** Pipecat의 `RawAudioTrack`은 chunk가 소비될 때
resolve되는 write별 `Future`와 함께 10 ms 단위로 씁니다. realtime_voice의 `OutboundAudioTrack`은 20 ms
packet과, `recv()` 호출들 사이에 나머지를 유지하는 `av.AudioFifo`를 쓰며, write-completion future는
없습니다. 그것의 `_silence_frame` (L201)은 힘들게 얻은 주석을 답니다: *"PyAV does not guarantee
zero-initialized AudioFrame storage. Sending a fresh allocation as 'silence' can therefore produce
full-scale random PCM"* — 그리고 그것에 대해 고정된 test가 있습니다
(`test_outbound_track_underflow_is_explicit_zero_pcm_silence`).

**이 row가 무엇에 달려 있는가.** 두 번째 client type이나 두 번째 transport가 로드맵에 있는지 여부.
aiortc loopback test를 통과하는 1,276줄은 이미 값을 치른 것이고, 당신이 쓰지 않는 열한 개의 transport는
비용도 0이고 사 주는 것도 0이기 때문입니다.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.7 Row 7 — Telephony serializer

**[[ch-05/read]]가 먼저 구조적 사실을 공급했습니다: telephony transport 자체가 없습니다.** Pipecat에서
전화 통화는 `FastAPIWebsocketTransport` **더하기** 여섯 개의 `FrameSerializer` 중 하나입니다:

```bash
$ ls src/pipecat/serializers/
__init__.py  base_serializer.py  exotel.py  genesys.py  plivo.py
protobuf.py  telnyx.py  twilio.py  vonage.py
```

telephony 여섯 개(`twilio`, `telnyx`, `plivo`, `exotel`, `genesys`, `vonage`) 더하기 telephony가 아닌
`protobuf.py`. 크기: `exotel.py` 171, `vonage.py` 188, `plivo.py` 256, `telnyx.py` 292,
`twilio.py` 314, `genesys.py` **964** — 이상치이고, [[ch-05/read]] §6.11은 그것을 codec 작업이 아니라
session 관리 표면으로 설명합니다. ABC 자체인 `base_serializer.py`는 106줄에 네 개의 method입니다.

**기존 두 stack 어느 쪽에도 telephony 경로는 없습니다.** [[rtv-vs-pipecat-gap]]: realtime_voice의
`SileroVAD`는 8 kHz를 하드 거부하고, serializer layer도 없고, μ-law도 없으며, `CLAUDE.md`는
*"future SIP/RTP or telephony adapters"*를 구현이 아니라 의도로 지명합니다. 그래서 이 row는 어느
column에도 ship된 답이 들어 있지 않은 유일한 row입니다. 비교는 *slot을 가진 framework에 serializer를
추가하는 것* 대 *slot이 없는 stack에 telephony layer를 추가하는 것* 사이에서 이뤄집니다.

**규모 산정, [[ch-05/read]] §11.1과 §2.2에서.** 그 여섯 바깥의 한국 carrier는 pattern-B 코드를
쓴다는 뜻입니다: *"250–300 lines for a Twilio-shaped protocol, up to ~1,000 if it has Genesys-shaped
session semantics."* SIP/RTP leg를 직접 넘겨주는 carrier는 그 둘 중 어느 것도 아닙니다 — 그것은
serializer 아래에 떨어지는 다른 문제입니다.

**[[ch-06/read]]가 두 개의 8 kHz fact를 공급했습니다.** 첫째, 8 kHz 쪽은 VAD 재튜닝 비용이 없습니다:
`round(0.2 / (256/8000))` = `round(6.25)` = 6, 16 kHz와 동일 (§3.1). 둘째, 그리고 이게 아픈
쪽인데, [[stt-korean-providers]]에서: *"the only `8000` values in the tree are telephony serializer
defaults (`twilio.py:79`, `telnyx.py:60`, `plivo.py:54`, `exotel.py:49`, `genesys.py:148`,
`vonage.py:43`); no STT service documents behaviour at that rate."* **8 kHz 한국어 STT 숫자는 tree
어디에도 존재하지 않습니다.**

**figure는 이 row에서 양립 불가능한 조합 하나를 표시합니다**: telephony를 추가하면서 realtime_voice의
Silero를 유지하는 것 — 그것이 8 kHz에서 `ValueError`를 raise하기 때문입니다.

**이 row가 무엇에 달려 있는가.** 어느 carrier인지, 그리고 언제인지.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.8 Row 8 — Session auth (`WebRTCSessionManager`)

**[[ch-03/read]]와 [[ch-05/read]]가 양쪽 절반을 공급했습니다.**

[[rtv-webrtc-transport]]에서: `WebRTCSessionManager` (`manager.py` L51), docstring은 *"Create
short-lived authorized sessions and enforce one live peer each."*
`create_session(customer_id, *, session_id=None, metadata=None) -> VoiceSessionTicket(session_id,
token, expires_at, customer_id)`는 `secrets.token_urlsafe(32)`를 발행하고 `hashlib.sha256(token)
.digest()`만 저장합니다. `_authorize` (L227)는 만료를 확인한 뒤 `hmac.compare_digest`를 씁니다.
`session_token_ttl_seconds = 15 * 60`. `accept_offer(..., reconnect: bool = False)`는 `reconnect=True`
가 아닌 한 `SessionConflictError("this voice session already has a live peer")`를 raise합니다 —
명시적 reconnect, 조용한 탈취 없음.

**Pipecat은 대응물을 배포하지 않으며**, [[ch-05/read]] §10.2는 그것을 주장이 아니라 grep으로
확립했습니다:

```bash
$ grep -rn "token_urlsafe\|compare_digest" src/pipecat/
src/pipecat/runner/run.py:324:    if not hmac.compare_digest(expected, sig):
src/pipecat/transports/whatsapp/client.py:181:        if not hmac.compare_digest(expected_signature, received_signature):
```

두 hit 모두 webhook 서명 검증입니다. 어느 것도 voice-session authorization이 아닙니다.
`SmallWebRTCConnection.__init__(ice_servers=...)` (`connection.py` L245)에는 token도, TTL도, customer
binding도 없습니다. `request_handler.py`는 헐벗은 offer/answer endpoint입니다.

**어느 쪽이든 이것은 application code입니다.** 이 row는 "framework가 그것을 가지고 있는가"가 아닙니다
— 그 질문의 답은 정해졌습니다 — 당신이 소유한 248줄이 Pipecat connection 객체 위로 수정 없이
옮겨가는지, 그것을 중심으로 다시 쓰이는지, 아니면 자기 transport와 함께 있던 자리에 남는지입니다.

**이 row가 무엇에 달려 있는가.** row 6이 움직이는지 여부. 이것은 row 6이 안착하는 connection 객체가
무엇이든 그것을 감싸는 policy layer이기 때문입니다.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.9 Row 9 — Control protocol (`ControlEvent`)

**[[ch-03/read]]와 [[ch-05/read]]가 양쪽 절반을 공급했습니다.**

[[rtv-webrtc-transport]]에서: `ControlEvent` (`control.py` L25)는
`@dataclass(frozen=True, slots=True)`이고 `session_id, type, sequence, payload, turn_id,
generation_id, version=CONTROL_PROTOCOL_VERSION`을 가지며 `CONTROL_PROTOCOL_VERSION = 1`입니다.
docstring L28-30: *"Audio bytes are intentionally prohibited. Microphone and assistant audio belong
on RTP tracks, never in JSON or base64."* `_reject_audio_payload` (L117)는 payload를 **재귀적으로**
훑으면서 `{"audio","audio_base64","audio_data","base64_audio","pcm","pcm16","wav"}`로 정규화되는 모든
key, `data:audio/`로 시작하는 모든 문자열, 그리고 모든 `bytes/bytearray/memoryview`에 대해
raise합니다. `from_json` (L64)은 알 수 없는 top-level field, object가 아닌 payload, version 불일치를
거부합니다. `OrderedControlChannel` (L136)은 부분 신뢰(partially-reliable) channel을 *생성 시점에*
거부합니다 — `ordered=False`, `None`이 아닌 `maxRetransmits`, `None`이 아닌 `maxPacketLifeTime` 각각이
`SignalingError` (*"Control events must not silently disappear"*)를 raise합니다 — 그리고 `receive()`는
`SignalingError(f"out-of-order control event: expected {...}, received {...}")`로 엄격한 순서 배달을
강제합니다. outbound sequence는 private counter이므로 순서는 서버가 소유합니다. 상한:
`max_control_message_bytes = 64 KiB`.

**Pipecat은 data channel에 대응물을 배포하지 않습니다.** `SmallWebRTCTransport._on_app_message(message,
sender)` → `on_app_message` event handler: schema 없음, sequence 검사 없음, 크기 상한 없음, audio
금지 없음. Pipecat에는 typed client protocol이 있긴 합니다 — RTVI, [[rtvi-observability]] 참조 —
그러나 그것은 다른 layer를 타고 다니며 channel에서 강제되지 않습니다.

**닫힌 union이 수면 위로 드러나는 곳은 dotted-type mapping입니다**, [[ch-05/read]] §7과
[[rtv-webrtc-transport]]에서: `_control_event()` (`transport.py` L118)는 `VoiceEvent →
event.kind.value`, `AgentTextDelta → "text_delta"`, `ASREvent → "transcript.interim" |
"transcript.final" | "asr.end_of_turn" | "asr.error"`, `VADEvent → "vad.speech_started" |
"vad.speech_stopped"`로 map하고, 그 밖의 어떤 것이든
`TypeError(f"unsupported voice event: {type(event).__name__}")`를 raise합니다. Pipecat 아래에서 그
mapping은 serializer의 일이 됩니다 — [[ch-05/read]] §7의 발견은 *"the serializer is where the open sum
type gets closed."*입니다.

**이 row가 무엇에 달려 있는가.** 당신의 debug client와 ops 도구가 이미 말하고 있는 wire contract를
transport 변경을 가로질러 바이트 단위로 동일하게 유지할 가치가 있는지 여부.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.10 Row 10 — The LLM loop

**[[ch-09/read]]가 전부 공급했습니다.** 다섯 개의 mechanism fact, 각각 파일과 줄 번호와 함께.

**하나 — `get_messages()`는 살아 있는 list를 반환합니다.** `llm_context.py:245`는 `self._messages`
자체를 반환합니다. `truncate_large_values=True`일 때만 복사합니다. boson의
`ContextManager.get_messages()`는 `deepcopy(self._messages)`를 반환합니다 (`manager.py:47-51`).
각각은 자기 집에서 옳습니다: `_update_function_call_result`는 반환된 참조를 *통해서* 씁니다
(`llm_response_universal.py:2158-2165`)이므로 한쪽에서는 identity가 load-bearing이고, provider
adapter에 건네진 복사본은 오염될 수 없으므로 다른 쪽에서는 격리(isolation)가 load-bearing입니다.
[[ch-09/read]] §9.1은 `deepcopy` 방어를 그대로 옮기면 tool 결과가 조용히 model에 절대 도달하지 않게
되는 여덟 단계를 추적했습니다 — exception도 없고, log 줄도 없습니다.

**둘 — `LLMContextFrame`당 inference 하나:**

```python
# src/pipecat/services/openai/base_llm.py L599-605
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
```

601번 줄은 **direction을 검사하지 않으며**, 그것이 upstream frame이 같은 service를 다시 prompt하는
이유입니다.

**셋 — `max_turns`가 어디에도 없습니다.** `grep -rn "max_turns\|max_iterations\|max_tool_calls"
src/pipecat/` → 0. 유일한 경계는 호출별 timeout입니다. boson의 가드는 두 줄짜리 `while` 조건입니다
(`agent_loop.py:207-209`).

**넷 — loop는 loop header가 아니라 topology에 의해 닫힙니다.** pipeline의 마지막 processor가
`LLMContextFrame`을 **upstream으로** push하고 (`llm_response_universal.py:1889`), turn은 아무도
그것을 하지 않을 때 끝납니다. [[ch-09/read]] §13은 시스템별로 한 문장씩 정리했습니다:
*"Pipecat: the turn ends when nobody pushes an `LLMContextFrame` upstream. An absence. boson: the
turn ends at `agent_loop.py:363`, `break  # Done — text response means end of turn`. A statement.
realtime_voice: the turn ends when the agent's `AsyncIterator[AgentTextDelta]` stops. A delegation."*

> 💡 **쉬운 설명 — "부재(absence)로 끝나는 loop"가 왜 무서운가요?**
> boson에서 turn이 끝나는 지점은 `break` 한 줄이라 grep으로 찾을 수 있습니다. Pipecat에서 turn이
> 끝나는 지점은 **아무 일도 일어나지 않는 것**입니다. "누가 upstream push를 안 했다"는 사건은
> stack trace에도, log에도 남지 않습니다. 그래서 "왜 turn이 안 끝나지?"와 "왜 turn이 벌써 끝났지?"가
> 둘 다 디버깅하기 어렵고, `max_turns`가 없다는 셋째 사실과 겹치면 무한 tool loop를 막는 것이
> 오직 호출별 timeout뿐이라는 결론이 나옵니다.

**다섯 — 세 개의 resolution, 가격이 매겨졌고, 아무것도 추천되지 않았습니다.** [[ch-09/read]] §11은
**adopt** (~30 L 상한 + gate + reminder processor; `run_agent_loop` 561 L 삭제; tool signature 22개
변경), **wrap** (processor 하나에 ~200–300 L; 아무것도 삭제하지 않음; signature 0개 변경), 그리고
**bypass** (~120–170 L bridge; 아무것도 삭제하지 않음; signature 0개 변경)를 전개하고, 선택을 명시적으로
열어 두었습니다. [[ch-09/read]] §10은 또한 **realtime_voice가 이미 bypass 모양을 구현하고 있다**는
사실을 기록했습니다 — *"you are not reading a hypothetical: you are reading a description of code
that exists on branch `voice-chat-dev`"* — 그것이 좋은 결정이었는지에 대해서는 아무 주장도 하지 않으면서.

**여섯 — 세 번째 구현, 그리고 그것이 왜 구조적으로 세 번째 답인가.** [[rtv-pipeline-session]]에서,
realtime_voice의 agent slot은 정확히 한 type을 yield하는 `Protocol` 하나입니다:
`StreamingConversationAgent.stream(request) -> AsyncIterator[AgentTextDelta]`. tool call도 아니고,
context object도 아니고, message도 아닙니다. **`link()`도, `FrameDirection`도, upstream push도
없습니다** — *"Where Pipecat gets a `LLMContextFrame` pushed upstream to close the tool loop,
realtime_voice has one direction only."* 그것을 세 개의 소유권 질문에 매핑하면 셋 다 답은 "voice
package가 아니다"입니다: context를 보유하지 않고, tool을 dispatch하지 않으며, iterator가 멈출 때 turn이
끝났음을 알게 됩니다. excerpt 자신의 표현: *"`StreamingConversationAgent` is the one slot Pipecat has
no analogue for — Pipecat assumes it owns the LLM call, whereas boson deliberately delegates."*

**이 row가 무엇에 달려 있는가.** §4.1. 이것이 첫 번째 환원 불가능한 collision의 절반입니다.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.11 Row 11 — Tools

**[[ch-09/read]]가 전부 공급했습니다.**

**세 개의 등록 경로, 하나의 dictionary.** `register_function(name, handler)`
(`llm_service.py:200-205`) — `name=None`이면 catch-all을 설치합니다; `ToolsSchema`에 실려 오는
`FunctionSchema(handler=...)`; 그리고 signature와 docstring에서 schema가 파생되는 direct function
(`adapters/schemas/direct_function.py:279-289`). `register_direct_function`은 1.4.0부터
`@deprecated`입니다 (`llm_service.py:982-984`).

**`result_callback` contract: handler는 절대 값을 반환하지 않습니다.** `llm_service.py:142-155`.
그것은 `await params.result_callback(result)`로 결말을 냅니다. boson의 handler는 값을 반환하는
`handler(**arguments)`이므로 (`tools/executor.py:67`), `agents/*/tools/` 아래의 **22개 tool signature**
전부가 변경 또는 shim을 필요로 합니다.

**`run_in_parallel`의 default는 `True`입니다** (`llm_service.py:303-308`). boson은 `_execute_tool_uses`
안에서 tool을 순차적으로 돌리고, 그것의 `_SYNC_HANDLER_LOCK`이 존재하는 이유는 [[boson-tool-router]]에서
그대로: *"production tools do read-modify-write on shared YAML/JSON files."* 그 가정은 상속되는 게
아니라 명시적으로 다시 주장되어야 합니다.

**boson의 message schema에는 `tool` role이 없습니다.** `basement/schemas/message_schema.py:46`은
`content` 안의 block과 함께 두 개의 role을 선언합니다. Pipecat의 `LLMContext`는 OpenAI 모양이고 세 개
이상의 role, `tool_calls` 배열, 그리고 별도의 `tool` message를 가집니다. [[ch-09/read]] §8.4에
나란히 놓은 비교가 있고, §9.2가 가격을 매깁니다: history가 migration에서 살아남아야 한다면 양방향 각각의
converter, 더하기 `is_error`에 대한 결정 — 그것에 대응하는 Pipecat field는 없습니다. *"Not a rename.
If someone on the team scopes this as `s/user/tool/`, correct them."*

**세 개의 gate, 하나의 slot.** boson은 노출(exposure, model이 무엇을 보는가), 가용성(availability,
`_allowed_tools_var` `ContextVar` allowlist), 권한(permission, `PermissionChecker.check_tool`)을
분리합니다. `FunctionCallRegistryItem`에는 permission field도 allowlist field도 없고,
`_run_function_call`에는 이름 해석과 호출 사이에 개입 지점이 없습니다. [[ch-09/read]] §9.4는 두 가지
탈출로에 가격을 매깁니다: 데코레이트된 handler 22개, 또는 catch-all 하나 더하기 당신 자신의 dispatch
table — *"which is `ToolRouter` again, re-hosted."* 또한: `ToolRegistry.discover_tools()`에는 Pipecat
대응물이 없고, 어느 쪽이든 boson glue로 남습니다.

**이 row가 무엇에 달려 있는가.** §4.1. 이것이 첫 번째 환원 불가능한 collision의 나머지 절반이고,
[[ch-09/read]] §12의 Move 1 (permission kernel로서의 catch-all)이 vote 중 하나를 싸게 만들어 줄
framework-extension 스케치입니다.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.12 Row 12 — The stage machine

**[[ch-10/read]]가 전부 공급했습니다.**

**`FlowManager`는 pipeline 바깥에 사는 평범한 class입니다:**

```python
# src/pipecat/flows/manager.py L80, L91-101
class FlowManager:
    ...
    def __init__(
        self,
        *,
        llm: LLMService | LLMSwitcher,
        context_aggregator: Any,
        worker: PipelineWorker | None = None,
        task: PipelineWorker | None = None,
        context_strategy: ContextStrategyConfig | None = None,
        transport: BaseTransport | None = None,
        global_functions: list[FlowsFunctionSchema | FlowsDirectFunction] | None = None,
    ):
```

`class FlowManager(FrameProcessor)`가 아닙니다. 그것은 pipeline **head**에 frame을 queue함으로써
바깥에서 구동합니다 (`self._worker.queue_frames(frames)`, `manager.py:842`; [[pipeline-task-runner]]에
따라 `DOWNSTREAM`으로 `queue_frame`하면 head로 들어갑니다). [[theory-narrow-waist]] §4는 그것이 왜
구조적으로 중요한지 기록했습니다: `flows/`는 20개가 아니라 **2**개의 frame을 정의했고, node state —
`NodeConfig`, `FlowResult`, `ContextStrategyConfig` — 는 *"plain `TypedDict` held in
`FlowManager._current_node`, and never becomes a frame at all."*

**이 commit에 `FlowConfig`는 존재하지 않습니다.** `grep -rn "FlowConfig" src/pipecat/` → hit 0.
선언적 전체 그래프 객체는 없습니다. transition 시점에 생산되는 node config가 있을 뿐입니다.

**두 개의 `NodeConfig` 생산자.** `tuple[Result, NodeConfig]`를 반환하는 function handler — tool edge,
예컨대 `insurance_quote.py` L114의 `collect_age` — 그리고 바깥에서
`flow_manager.set_node_from_config(node_config)` (`manager.py:588`)를 직접 호출하는 것.
[[flows-insurance-example]]은 후자를 tree 안에서 정확히 한 번, `warm_transfer.py:658`에서
`flow_manager.current_node`를 가드로 두고 사용하는 것을 찾았습니다.

**transition 합법성 검증도 없고, node registry도 없습니다.** `_validate_node_config`
(`manager.py:867-898`)는 정확히 두 가지를 검사합니다: `task_messages`가 존재하는지, 그리고
`functions`의 각 항목이 `FlowsFunctionSchema`이거나 유효한 direct function인지. codebase 어디에도
from→to 검사는 없습니다. [[design-boson-rules-on-pipecat]]은 그것을 표의 한 행으로 진술합니다:
*"`StageMachine.transition()` legality → **NOTHING** … Flows has no from→to check anywhere in the
codebase."*

**boson 쪽**, [[boson-stage-machine]]에서: stage는 *context 패키지*입니다 — prompt + 보이는 tool +
보이는 skill + 합법적 후속 stage의 whitelist — 그리고 `StageMachine.transition()`은 불법 edge에 대해
`TransitionResult(success=False, error="Transition 'a' -> 'b' not allowed")`를 반환합니다. Lina의
stage는 아홉 개입니다. **LLM은 절대 그중 하나를 고르지 않습니다.**

**`stage_config.py`의 직접 매핑**, [[flows-insurance-example]]에서: `stages[X]["tools"]` →
`NodeConfig["functions"]`는 직접 대응입니다. `_GLOBAL_TOOLS` → `FlowManager(global_functions=[...])`는
`manager.py:654`에서 섞여 들어가며 (`functions_list = self._global_functions +
node_config.get("functions", [])`), 정확한 일치입니다. **`transitions`와 `skills`는 매핑되지
않습니다.** `transitions`는 *"stops being data"* — Flows에서 합법적 후속은 각 function이 어떤
`NodeConfig`를 반환할 수 있는지에 암묵적으로 들어 있습니다 — 그리고 `skills`는 *"no Flows concept at
all."*입니다.

**이름 붙은 regression 하나, 이것이 이 row의 falsifier 재료입니다.** `stage_config.py`는 자기 자신의
버그를 두 번 문서화합니다: `v0.7.5 (#12)` — *"`purchase` added — `transition_detector.py:157` emits
`StageTransition("purchase")` … but the stage machine rejected it because this list omitted
`purchase`."* 그 부류의 버그는 Flows에서는 구조적으로 불가능합니다. mechanism 교환은 이것입니다:
거부 경로가 사라지고, 한 파일에서 읽을 수 있는 transition 표도 함께 사라집니다.

**이 row가 무엇에 달려 있는가.** edge whitelist가 Lina가 의존하는 safety property인지, 아니면 막은
버그보다 만든 버그가 더 많은 config 파일인지. 그것은 당신 자신의 issue 이력에서 답할 수 있습니다.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.13 Row 13 — The rule layers

**[[ch-12/read]]가 전부 공급했고, §2가 방금 그 비용을 측정했습니다.**

**세 개의 constraint와 seam 도출.** [[design-boson-rules-on-pipecat]]에서, 그 위치는 관례가 아니라
데이터 의존성입니다:

```python
# src/pipecat/processors/aggregators/llm_response_universal.py L856-873
    async def push_aggregation(self) -> str:
        """Push the current aggregation."""
        if len(self._aggregation) == 0:
            return ""

        aggregation = self.aggregation_string()
        await self.reset()
        self._context.add_message(
            cast(LLMContextMessage, {"role": self.role, "content": aggregation})
        )
        await self.push_context_frame()

        message = UserTurnMessageAddedMessage(
            content=aggregation, timestamp=self._user_turn_start_timestamp
        )
        await self._call_event_handler("on_user_turn_message_added", message)

        return aggregation
```

862-864의 `add_message`가 **쓰고**, 866의 `push_context_frame()`이 **push하고**,
`base_llm.py:601`이 **소비합니다**. 따라서 user aggregator와 LLM 사이의 processor는 inference가 아직
시작되지 않은 상태로 완전한 turn을 손에 쥐고 있으며, rollback은 `context.set_messages(...)`
(`llm_context.py:377`)를 통해 실재합니다. 옵션 B — 871번 줄의 `on_user_turn_message_added` event —
는 push *이후*이므로 **event handler는 절대 veto할 수 없습니다**. 옵션 C — generation이 시작되게 두고
`broadcast_interruption()` (`frame_processor.py:1017-1022`)을 호출 — 은 첫 generation의 TTFT를 두 번
지불합니다.

> 💡 **쉬운 설명 — 왜 "쓰기 → push → 소비" 순서가 seam을 결정하나요?**
> rule layer가 하고 싶은 일은 "이 turn을 model에 보내기 전에 검사하고, 필요하면 막는다"입니다.
> 그러려면 **완성된 turn 텍스트를 볼 수 있으면서 아직 inference는 시작되지 않은** 시점이 필요합니다.
> 862줄에서 텍스트가 context에 쓰이고, 866줄에서 frame이 push되고, `base_llm.py:601`에서 inference가
> 시작됩니다. 그 사이의 유일한 자리가 aggregator와 llm 사이입니다. 한 칸 앞이면 텍스트가 없고, 한 칸
> 뒤면 이미 model이 돌기 시작했습니다. 그래서 이건 취향이 아니라 데이터 의존성이 정한 자리입니다.

**collapse 논증.** `push_frame`은 되돌릴 수 없으므로 processor 간 veto는 사용 불가능하고, 모든 layer는
**하나의** `FrameProcessor`로 접혀야 합니다. [[design-boson-rules-on-pipecat]]은 그 손실을 매핑 표에
명시적으로 진술합니다: *"cross-`processor` veto — all layers must collapse into one object."*

**매핑 표.** [[ch-12/read]]는 그것을 11행으로 제시합니다. **⚠️ 기록해 둘 만한 count 불일치:**
[[design-boson-rules-on-pipecat]] §2에 있는 원본 excerpt의 표는 **17**행입니다 (layer discovery,
sequential `@check`, parallel `@check`, phase-1/2 commit, `Filter`, `Respond`, `Inject`, `PreTool`,
`StageTransition`, `StageMachine` legality, `StageDefinition.prompt/tools`, `_GLOBAL_TOOLS`,
`Compact()`, `SignalQueue`, `AgentStatusTracker`, `SharedLayerContext`, `ScriptEngine`). 그 열일곱 중
넷은 *이* ledger에서 각각 row 12, 14, 15, 그리고 11의 일부로 따로 서 있으며, 그것이 차이의 대부분입니다.
매핑 표는 mechanism 용도로 쓰고, 어느 count도 무언가의 evidence로 쓰지 마십시오.

**Tier-2 latency 청구서, 이제 측정됨.** §2. 살아 있는 `@check` 열세 개, 그중 정확히 두 개가
`check_type="llm"` (`intent_rules` prio 30, `sentiment_tracker` prio 10, 둘 다 `mode="parallel"`).
Tier 1은 나머지 열한 개 — *"all pure Python, sub-millisecond, blocking, free."* Tier 2는 그 둘이고,
in-turn veto와 in-turn transition을 사 줍니다. 그 값을 치르지 않는 유일한 방법은 Tier 2를 LLM의 첫
token들과 동시에 돌리고 완료 시 `set_node_from_config()`를 호출하는 것인데, **그러면 모든 stage 변경이
한 turn 늦게 착지합니다.** [[design-boson-rules-on-pipecat]]은 그 교환을 한 문장으로 진술합니다:
*"veto and in-turn steering cost 250-400 ms; next-turn transitions cost 0 ms. boson pays the
250-400 ms today."*

측정한 P50과 P95를 figure에만이 아니라 이 row에도 적어 넣으십시오. 이제 이 row는 안에 숫자가 든 채로
읽힙니다.

**이 row가 무엇에 달려 있는가.** §4.2, 그리고 당신이 방금 측정한 숫자.

| Vote | Assumption | Falsifier | Measured Tier-2 P50 / P95 |
|---|---|---|---|
| | | | |

---

### 3.14 Row 14 — `ScriptEngine`

**[[ch-12/read]]가 한 줄로 공급했고**, mechanism이 가장 짧기 때문에 이것이 ledger에서 가장 짧은
row입니다. [[design-boson-rules-on-pipecat]]의 매핑 표에서:

> `ScriptEngine.process_turn(state, msg, registry)` → **runs unchanged inside the processor** —
> *"already stateless dict-in/`Action`-out"* — what is lost: *"nothing — cleanest port in the
> system."*

**왜 수정 없이 port되는가**, [[boson-script-engine]]에서: 그것은 *"a stateless pure function over
a state dict that returns `(new_state, Action)"*이고, gateway 결합도가 0입니다 — 오직
`gateway.script.schema`와 `gateway.schemas.actions`만 import합니다. 517 LOC (`engine.py` 284 +
`schema.py` 233).

**그 output이 갈 수 있는 곳을 제약하는 구조적 사실 하나.** `Respond(step.text)`는 LLM을 배제한 채
*문자 그대로 발화되는 문자열*입니다. 한국어 보험 동의 script 텍스트가 법적으로 고정되어 있기
때문입니다. [[boson-script-engine]]은 그것이 무엇을 배제하는지에 대해 명시적입니다: *"Porting
`purchase_pre_consent` onto `NodeConfig.task_messages` would let the model paraphrase a regulated
consent disclosure."* 그 property를 보존하는 매핑은 `Respond` → 하류의 `LLMContextFrame`을 억제하면서
`TTSSpeakFrame`을 push하는 것입니다.

**또한 기록됨:** `FlowManager.state` — 평범한 `dict[str, Any]` (`manager.py:143`) — 는 `script_state`의
자연스러운 집입니다. Pipecat에는 `SessionState`가 없기 때문입니다. 그리고 `pause_for_interrupt` /
`resume_from_interrupt`는 boson의 `AgentStatusTracker`가 아니라 `InterruptionFrame`으로부터 다시
구동되어야 하는데, 전자에는 frame 대응물이 없습니다.

**이 row가 무엇에 달려 있는가.** "runs unchanged inside the processor"가 interrupt 재구동과의 접촉에서
살아남는지 여부. 그것이 이 row에서 공짜가 아닌 유일한 부분입니다.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.15 Row 15 — Compaction

**[[ch-09/read]]에서 여기로 라우팅되었고, 그 chapter는 이것을 이름으로 배제했습니다.** 그것의 scope
box: *"Context compaction is NOT in this chapter. `LLMContextSummarizer` versus boson's
`gateway/compact/` is a different subsystem with different failure modes and a different trigger, and
it is on [[ch-13/read]]'s give-back list."* 그것이 rule 2의 "row가 사실 두 개의 row다" test를 미리
적용한 것입니다.

**excerpt가 이 row를 공급합니다.** [[boson-compact-session]]에서: 두 시스템은 같은 문제를 풀었고 —
오래된 history를 요약하고, 최근 tail을 유지하고, `[system?] + [summary] + [tail]`을 이어 붙이기 —
거의 동일한 설계에 도달했습니다. **갈라지는 지점은 그 답이 언제 착지하는가입니다.** boson은 요약을
분리된 `asyncio.create_task`로 돌려 `session.pending_compact`에 쓰고 *다음* turn의 맨 앞에서 적용하므로
어떤 turn도 그것을 기다리지 않습니다 (`bootstrap.py` L455-458: *"Apply pending compact BEFORE
layers"*). Pipecat의 `LLMContextSummarizer`는 `LLMContextSummaryResultFrame`이 도착하는 그 순간,
pipeline 중간에서 결과를 적용합니다.

**세 갈래 parameter 대응**, tree에 대해 검증:

```python
# src/pipecat/utils/context/llm_context_summarization.py L146-148
    max_context_tokens: int | None = 8000
    max_unsummarized_messages: int | None = 20
    summary_config: LLMContextSummaryConfig = field(default_factory=LLMContextSummaryConfig)
```

```python
# src/pipecat/utils/context/llm_context_summarization.py L93-98
    target_context_tokens: int = 6000
    min_messages_after_summary: int = 4
    summarization_prompt: str | None = None
    summary_message_template: str = "Conversation summary: {summary}"
    llm: Optional["LLMService"] = None
    summarization_timeout: float = DEFAULT_SUMMARIZATION_TIMEOUT
```

| boson `CompactConfig` | Pipecat | Note |
|---|---|---|
| `threshold_messages = 30` (ge=5) | `max_unsummarized_messages = 20` | `30`으로 port |
| `keep_recent = 10` (ge=2) | `min_messages_after_summary = 4` | `10`으로 port |
| — | `max_context_tokens = 8000` | **boson에는 없던 token trigger** |
| `provider` / `model` (`gpt-5.4-mini`) | `LLMContextSummaryConfig.llm: Optional[LLMService]` | 양쪽 다 전용 저가 model |
| `temperature = 0.3` | — | Pipecat config에는 없는 field |

token estimator는 세 개의 상수입니다, `llm_context_summarization.py:33-35`: `CHARS_PER_TOKEN = 4`,
`TOKEN_OVERHEAD_PER_MESSAGE = 10`, `IMAGE_TOKEN_ESTIMATE = 500`. boson에는 **token 추정이 전혀
없습니다** — [[boson-compact-session]]: *"Trigger is message-count only — there is no token-based
trigger anywhere in boson."*

**⚠️ 우회해야 할 deprecation 하나.** 평평한 `LLMContextSummarizationConfig` (네 개의 숫자가 한 객체에
있는 것)는 0.0.104부터 `@deprecated`이고 2.0.0에서 제거됩니다 (`llm_context_summarization.py:170-173`).
새 코드는 `LLMAutoContextSummarizationConfig` + 중첩된 `LLMContextSummaryConfig`에 대고 쓰십시오.
[[pipecat-design-philosophy]]에 따르면 deprecation registry에는 **살아 있는 deprecation이 391개 있고,
전부 `removed_in == "2.0.0"`**입니다 — 그러므로 "adopt하되 deprecated class 위에서"는 실재하고 피할 수
있는 실수입니다.

**이름 붙은 두 개의 손실, 이 row의 falsifier 줄에 속하는 것들.**

1. **pre/post compact hook에는 대응물이 없습니다.** boson에는 module-level `set_pre_compact_hook` /
   `set_post_compact_hook`이 있고, typed signature를 가지며 입력 message list와 출력 summary 문자열을
   *변형(mutate)*할 수 있습니다. Pipecat의 가장 가까운 것은 `SummaryAppliedEvent(original_message_count,
   new_message_count, summarized_message_count, preserved_message_count)`를 실어 나르는
   `on_summary_applied` event입니다 (`llm_context_summarizer.py:39`, `:468`에서 발화) — **observability
   전용이고, 입력이나 출력을 변형할 수 없습니다.** 요약 전에 시끄러운 tool block을 벗겨내거나 요약 후에
   추출된 구조적 데이터를 덧붙이는 것은 무엇이든 `LLMContextSummarizer`를 subclass하거나 전용 요약
   `LLMService`를 통해 라우팅해야 합니다.
2. **`<system-reminder>Active stage: …</system-reminder>` 재주입.** `SharedHistory.swap_compact`
   (`session/history.py` L82)는 설정되어 있을 때 요약 뒤에 `<system-reminder>Active skill: …`과
   `<system-reminder>Active stage: …`를 덧붙입니다. 그것은 compactor 안에 살고 있는 stage-machine
   결합입니다. **명시적으로 다시 붙이지 않으면 매 compaction마다 stage 정체성이 조용히 사라집니다** —
   조용히, 왜냐하면 아무것도 raise하지 않기 때문입니다. model은 그저 자기가 어느 stage에 있는지를 통화
   중간에, 30번째 message 이후로 듣지 못하게 됩니다.

> 💡 **쉬운 설명 — 왜 이 두 손실이 특별히 위험한가요?**
> 둘 다 **실패가 예외를 던지지 않습니다.** compaction은 성공하고, 대화는 계속되고, log는 깨끗합니다.
> 다만 model이 "지금 purchase stage다"라는 사실을 더 이상 듣지 못할 뿐입니다. 증상은 30 message 이후
> 갑자기 model이 stage에 안 맞는 말을 하기 시작하는 것으로 나타나고, 원인은 compaction 코드에
> 있습니다 — 원인과 증상이 시간적으로도 코드상으로도 멀리 떨어진, 가장 비싼 종류의 버그입니다.

**가지고 갈 만한 mechanism fact 두 개 더.** `LLMContextSummarizer`는 `FrameProcessor`가 **아니라**
`BaseObject`를 확장합니다 — `process_frame(self, frame)`은 인자를 하나만 받고 `direction`이 없으며,
pipeline에 link되는 대신 aggregator에 의해 구동됩니다. 그리고 tool-pair 안전 전략은 반대 방향을
가리킵니다: boson의 `_safe_window_start`는 앞쪽의 고아 `tool_result`들을 **앞으로** 버립니다. Pipecat의
`_get_earliest_function_call_not_resolved_in_range`는 해소되지 않은 call 앞으로 `summary_end`를
**뒤로** 당깁니다.

**이 row가 무엇에 달려 있는가.** 다음 turn으로 미뤄서 적용하는 것이 Lina가 의존하는 property인지,
그리고 위의 두 손실 중 어느 것이든 당신의 compliance 녹취에 대해 load-bearing인지.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.16 Row 16 — Observability

**[[ch-11/read]]가 Pipecat 쪽을 공급했습니다.**

**observer plane.** `BaseObserver`는 frame graph에 대해 읽기 전용입니다 — [[rtvi-observability]]:
*"You instrument by subscribing, not by editing processors."* 네 개의 observer가 배포됩니다:
`startup_timing_observer.py`, `turn_tracking_observer.py`, `user_bot_latency_observer.py`, 그리고
`loggers/` package.

**`LatencyBreakdown`이 cycle당 객체입니다:**

```python
# src/pipecat/observers/user_bot_latency_observer.py L83-89, L107-111
class LatencyBreakdown(BaseModel):
    """Per-service latency breakdown for a single user-to-bot cycle.

    Collected between ``VADUserStoppedSpeakingFrame`` and
    ``BotStartedSpeakingFrame`` when ``enable_metrics=True`` in
    :class:`~pipecat.pipeline.worker.PipelineParams`.
    ...
    ttfb: list[TTFBBreakdownMetrics] = Field(default_factory=list)
    text_aggregation: TextAggregationBreakdownMetrics | None = None
    user_turn_start_time: float | None = None
    user_turn_secs: float | None = None
    function_calls: list[FunctionCallMetrics] = Field(default_factory=list)
```

docstring의 전제 조건을 주목하십시오: `PipelineParams`의 `enable_metrics=True`인데, 그 default는
`False`입니다 ([[pipeline-task-runner]], `worker.py` L163-195).

**`can_generate_metrics()`가 gate입니다:**

```python
# src/pipecat/processors/frame_processor.py L488-494
    def can_generate_metrics(self) -> bool:
        """Check if this processor can generate metrics.

        Returns:
            True if this processor can generate metrics.
        """
        return False
```

**base class에서 default가 `False`입니다.** 당신이 쓰는 processor는 당신이 override하기 전까지 아무것도
emit하지 않습니다. 그것이 "gating" fact입니다: plane은 존재하고, 당신 자신의 processor는 default로 그
바깥에 있습니다.

**집계(aggregation)는 당신 몫으로 남습니다.** observer plane 어디에도 P50/P95는 없습니다.
`LatencyBreakdown`은 cycle 하나를 주고 `chronological_events()`가 그것을 log 한 줄로 포맷합니다.
그런 것들의 stream을 percentile로 바꾸는 것은 application code입니다. [[rtvi-observability]]는
exporter — OpenTelemetry, Sentry — 와 RTVI wire protocol을 기록하지만, percentile 계산은 상자 안에
없습니다.

**boson 쪽**, [[rtvi-observability]]와 [[rtv-vs-pipecat-gap]]에 따르면: 세 가지가 있고 plane은
없습니다. trace decorator 하나 (`gateway/debug/log_decorator.py`, 호출 주위의 `time.perf_counter()`가
`[TRACE …] EXIT (…ms)`로 인쇄됨), barge-in policy로 임시로 꿰어진 `elapsed_ms`
(`core.py:166 should_interrupt(session_id, content, elapsed_ms)`), 그리고
`BoundedAudioOutput.discarded_frames` — *"the only metric in the transport, and it is orphaned"* —
아무것도 그것을 읽지 않습니다. 더하기 event 위의 `provider_latency_ms` / `endpoint_latency_ms` field.
OTel도, span도, 집계도 없습니다.

**그리고 그 두 stack이 견주어지는 기준은 boson 자신의 것입니다.** [[rtv-vs-pipecat-gap]]: `CLAUDE.md`는
*"Instrument before optimizing… Report P50/P95/P99."*라고 말합니다. 오늘 어느 stack도 그것을 하지
않으며, Pipecat의 plane도 P-숫자를 만들어 주지는 않습니다 — 그것은 수집을 합니다.

**이 row가 무엇에 달려 있는가.** "집계는 application code다"가 작은 일인지 아니면 어차피 밑바닥부터
해야 할 같은 일인지. 그것은 `LatencyBreakdown`에 대고 aggregator를 한 번 써 보고 얼마나 걸리는지
보면 정리됩니다.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 3.17 Row 17 — Deployment and process topology

**이것은 다른 것과 똑같은 온전한 row이고 후기가 아닙니다.** [[ch-04/read]]가 host 모양을 공급했고,
아래 §7이 나머지를 공급합니다. deployment는 latency 항이 아니라 결정의 *입력*이기 때문입니다. §7을
읽고, 돌아와서 이 세 칸을 채우십시오.

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

## 4. 환원 불가능한 두 개의 collision

이 둘은 evidence와 함께 row로 진술되며 **본문에 의해 해소되지 않습니다.** 각각 vote cell, assumption
cell, falsifier cell을 받고, 셋 다 당신이 채웁니다.

### 4.1 Collision 하나 — agent boundary (row 10과 11)

**진술.** Pipecat의 pipeline을 adopt한다는 것은 Pipecat이 LLM 호출과 tool loop를 소유한다는 뜻입니다.
그것은 당신 자신의 contract와 대립합니다. [[rtv-vs-pipecat-gap]]에서 `CLAUDE.md`를 인용하면: *"Keep
Basement and the dental business logic text-native"* / *"Basement and Gateway must not import
provider-specific audio code."* excerpt 자신의 성격 규정: *"This is the deepest architectural
collision. Pipecat assumes it owns the LLM+tool loop; boson's whole `CLAUDE.md` contract assumes it
does not."*

**왜 환원 불가능한가.** 이것은 200줄로 메울 수 있는 gap이 아닙니다. 이것은 상대방의 topology가
부정하는 invariant를 각자 붙들고 있는 두 시스템입니다. `base_llm.py:601`은 `LLMContextFrame`을
소비하고 completion을 시작합니다. Pipecat 설계의 어떤 것도 pipeline 안의 `LLMService`가 아닌 무언가에
의해 completion이 시작되도록 허용하지 않습니다. 한편 boson의 contract는 LLM 호출이 `basement`에서,
텍스트 위에서, audio layer 의존성 없이 일어난다는 것입니다. 둘 중 하나가 양보해야 합니다.

**[[ch-09/read]]가 가격을 매겼고, 아무것도 지명하지 않았습니다.** 세 개의 모양, 나란히 (그 §11.4의 표):

| | **Adopt** | **Wrap** | **Bypass** |
|---|---|---|---|
| 새 Pipecat 쪽 코드 | ~30 L 상한 + gate + reminder processor | processor 하나에 ~200–300 L | ~120–170 L bridge |
| 삭제되는 boson 코드 | `run_agent_loop` (561 L) + ~80 L cancellation 수리 + 6개 provider shaper | 없음 | 없음 |
| 변경되는 tool signature | 22 | 0 | 0 |
| `max_turns` | processor로 재구축 | 유지 | 유지 |
| 세 개의 gate | catch-all 또는 decorator로 재구축 | 유지 | 유지 |
| Message schema | 재작성 | 유지 | 유지 |
| Adapter layer 사용 | 예 (12개 adapter) | 아니오 | 아니오 |
| Flows ([[ch-10/read]]) 사용 가능 | 예 | 비쌈 | 아니오 |
| Observer plane이 tool call을 봄 | 예 | 당신이 emit하는 것만 | 아니오 |
| 당신 repo에 이미 구현되어 있음 | 아니오 | 아니오 | **예, realtime_voice를 voice layer로 해서** |

**어느 것을 고르든 따라붙는 두 개의 가격 매겨진 귀결.**

**하나 — 사라진 turn 상한.** Pipecat에 `max_turns`는 존재하지 않고, [[ch-09/read]] §9.3은 그것을
`FunctionCallsStartedFrame` / upstream `LLMContextFrame`을 세는 counting processor로 재구축했으며,
**`llm`과 `tts` 사이**에 놓았습니다 — 그것이 재prompt(upstream으로 이동)와 reset
신호(`UserStartedSpeakingFrame`, downstream으로 이동)를 둘 다 보는 유일한 위치이기 때문입니다.
boson의 두 줄짜리 `while` 조건에 대해 대략 30줄 더하기 test 하나이고, 이름 붙은 네 개의 비용이 따르며
그중 가장 날카로운 것은 **그것이 inference가 아니라 재prompt를 센다**는 것입니다. 그래서 거기서
`max_turns=8`은 "tool cycle 8회"를 뜻하고, boson의 "총 iteration 8회"와 구성상 하나가 어긋납니다.

**둘 — signature 22개에 걸친 `{"role": "tool"}` schema 재작성.** §3.11. `content` 안에 block을 둔
두 개의 role 대 `tool_calls` 배열과 별도의 `tool` message를 둔 세 개 이상의 role. dict 인자 대
JSON 문자열 인자. `is_error` 대 field 없음. 더하기 [[ch-09/read]] §9.2가 지명한 손실:
`LLMContext`는 schema 없는 평범한 dict를 저장하므로 *"a typo in a hand-built message dict is
discovered by the provider's 400, not by your type checker."*

**비용이 아닌, 저울에 올릴 세 번째 것.** [[ch-09/read]] §10은 realtime_voice가 이미 bypass 모양을
구현하고 있다고 기록했고, 그것이 무엇을 뜻하고 무엇을 뜻하지 않는지에 대해 조심스러웠습니다:
*"it changes what 'adopt' and 'wrap' would cost — because both of them mean undoing something that
works today, and that undoing is a line item."*

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

### 4.2 Collision 둘 — rule layer (row 13)

**진술.** [[ch-12/read]]의 도출은 **seam** (`user_aggregator`와 `llm` 사이,
`llm_response_universal.py:856-866`의 write-then-push 데이터 의존성에 의해 강제됨)과 **collapse**
(모든 layer가 하나의 `FrameProcessor`로, `push_frame`이 되돌릴 수 없다는 사실에 의해 강제됨)를
생산했습니다. 이 chapter가 답하지 않고 제기하는 열린 질문은: **각 vote 아래에서 boson의 얼마나 많은
부분이 수정 없이 살아남는가?**

**당신에게 평가하라고 주어진 후보 모양이지, adopt하라고 지시된 것이 아닙니다.**
[[design-boson-rules-on-pipecat]]에서:

```python
# the proposed pipeline (design excerpt §4) — a candidate, not a recommendation
pipeline = Pipeline([
    transport.input(),
    stt,                       # Korean 8 kHz telephony STT
    BosonFillerGate(),         # boson layer 01
    user_aggregator,           # LLMContextAggregatorPair(context).user()
    BosonRuleProcessor(...),   # boson layers 02/03/04, Tier 1 + Tier 2
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])
# FlowManager(llm=llm, context_aggregator=pair, worker=worker, global_functions=[...])
# is NOT in this list — it drives from outside via worker.queue_frames (manager.py:841).
```

`user_aggregator`와 `llm` 사이의 `BosonRuleProcessor` 하나가 **열세 개의 check 전부**, `RuleEngine`,
`SignalQueue`, 그리고 **진짜 `SessionState`**를 보유합니다. `StageMachine`은 순수한 합법성 사전 검사로
유지되고, `FlowManager`는 바깥에서 `set_node_from_config`를 통해 stage를 구동합니다.

**그 모양에 붙어 있는 주장, 이것이 평가할 대상입니다.** excerpt의 migration angle:
*"`BosonRuleProcessor` holds all 13 checks, the `RuleEngine`, the `SignalQueue`, the `StageMachine`
pre-check, and a real `SessionState` — so the rule files themselves need **zero edits**."* 그
zero-edit 주장은 전적으로 진짜 `SessionState` 객체를 유지하고 그것을 `session` 인자로 넘기는 데
달려 있습니다. `SharedLayerContext.__getattr__`/`__setattr__`가 알려지지 않은 모든 이름을 그것으로
그대로 proxy하기 때문입니다. `flow_manager.state`는 `dict`이고, 13개 rule의 `getattr(session, …)`을
dict에 대고 재작성하는 것은 *"buys nothing."*입니다.

> 💡 **쉬운 설명 — `__getattr__` proxy가 왜 zero-edit의 열쇠인가요?**
> boson rule은 `session.sentiment_history`, `session.fired_rules`처럼 **아무 이름이나** 붙여 씁니다.
> `SharedLayerContext`가 모르는 attribute 접근을 전부 실제 `SessionState` 객체로 넘겨주기 때문에
> rule 코드는 자기가 무엇에 대고 쓰는지 신경 쓸 필요가 없습니다. 이것을 `dict`로 바꾸면 모든
> `session.x`가 `state["x"]`가 되어야 하고, 그건 13개 rule 파일 전부를 만지는 일입니다 — 그리고
> 그 대가로 얻는 게 없습니다. 그래서 "진짜 `SessionState`를 유지한다"가 protocol의 핵심 조건입니다.

**그 목록에서 두 개의 위치가 load-bearing이고, excerpt는 각각을 취향이 아니라 데이터 의존성으로부터
논증합니다.** `BosonFillerGate`는 `stt`와 `user_aggregator` 사이에 앉아야 합니다: 한 칸 앞이면 audio만
보이고 `_is_filler_text()`에 입력이 없습니다. 한 칸 뒤면 "네"가 이미 `add_message`된 뒤이고, 한 줄짜리
문자열 검사가 rollback으로 변해 버립니다. `BosonRuleProcessor`는 `user_aggregator`와 `llm` 사이에
앉아야 합니다: 한 칸 앞이면 `kw in user_message.lower()`가 조각(fragment)에서 발화하고, 한 칸 뒤면
`Inject`가 자기가 steer하려고 쓰였던 그 generation을 더 이상 steer할 수 없습니다.

**그리고 그 안의 한 가지 동작은 순진한 조언을 뒤집습니다.** transition turn에서 `BosonRuleProcessor`는
context frame을 **삼키고(swallow)** Flows node가 유일한 inference trigger가 되게 합니다
(`respond_immediately=True` → `manager.py:707-709`의 `LLMRunFrame()`). `set_node_from_config()`를
`push_frame()`보다 먼저 순서 짓는 것은 race를 고치지 *못합니다*. node의 frame들은 head로 들어가서
`llm`에 도달하기 전에 `stt` → gate → aggregator를 거쳐야 하는 반면, 당신이 push한 frame은 `llm`에
즉시 도달하기 때문입니다. 삼키기는 구성상 race를 제거합니다. **주시해야 할 failure mode에 이름이
붙어 있습니다**: generation 0회(양쪽 경로 모두 삼켜짐)와 2회(node도 돌고 context도 push됨). 그것이
§1.4의 미지수 #4이고, *code-assertion* type의 falsifier입니다.

**세 번째 열린 risk, 이것은 two-phase-commit의 폭발 반경입니다.** boson은 `session.messages`에 대해
객체 identity로 rollback합니다. `LLMContext`는 identity handle 없이 `set_messages(list)`만 제공하고,
aggregator는 이미 써 버렸습니다. excerpt가 제안하는 prototype: rule round 전체를 둘러싼
snapshot/restore, Lina e2e suite (`agents/test-lina-gateway/tests/`, `e2e_runner.py`) replay, 그리고
divergence 세기 — *"specifically turns where a `PreTool` appended synthetic tool-call history before
a later layer filtered."*

| Vote | Assumption | Falsifier |
|---|---|---|
| | | |

---

## 5. Pipecat에 집이 없어서 boson 코드로 남아야 하는 것들

여섯 항목. 이 목록은 vote가 아닙니다 — 어떤 vote도 옮길 수 없는 것들의 집합입니다. 목적지가 존재하지
않기 때문입니다. [[design-boson-rules-on-pipecat]]의 migration angle과 subsystem별 excerpt로부터:

1. **Cross-layer veto.** `push_frame`은 되돌릴 수 없고, 유일한 rollback 표면은
   `LLMContext.set_messages()`입니다. processor 경계를 가로지르는 veto에는 mechanism이 없으며, 그것이
   §4.2의 후보가 네 개의 layer를 하나의 객체로 접는 이유입니다.
2. **Transition 합법성.** `_validate_node_config` (`manager.py:867-898`)는 두 가지를 검사하고 어느
   것도 from→to edge가 아닙니다. `flows/` 전역에서 합법성 검사를 `grep`하면 아무것도 나오지 않습니다.
   그것을 유지한다는 것은 boson의 `StageMachine` class를 사전 검사로 유지한다는 뜻입니다.
3. **`StageDefinition.skills`.** *"No Pipecat concept at all."* Lina의 `product_manager` /
   `payment_manager`는 function으로 평평해지거나, `use_skill` direct function 뒤의 boson meta-tool로
   남습니다.
4. **session별 attribute namespace.** `SharedLayerContext`의 `__getattr__`/`__setattr__`가 살아 있는
   `SessionState`로 proxy하므로 `session.sentiment_history`, `session.fired_rules`,
   `session.script_state`가 동적으로 붙고 지속됩니다. `flow_manager.state`는 `dict[str, Any]`입니다
   (`manager.py:143`). Pipecat에는 `SessionState`가 아예 없습니다 — [[boson-compact-session]]은 정확히
   이 이유로 그것을 port에서 *"the most painful"* 부분이라고 부릅니다.
5. **`TOOL_PROCESSING` status.** boson의 `AgentStatusTracker`에는 그것을 위한 enum 값이 있고
   *"`TOOL_PROCESSING` has no frame that means it."* `BotStartedSpeakingFrame` /
   `BotStoppedSpeakingFrame`이 다른 state들을 덮고, 500 ms짜리 `settling_ms` 감쇠는 다시 도출되어야
   합니다. 이것이 중요한 이유는 `korean_fillers.py:66`이 `pre_turn_status`를 읽고, filler filter가
   잘못된 것을 읽으면 자기 자신을 걸러내 버리기 때문입니다.
6. **`<system-reminder>` protocol.** `Inject`는 `<system-reminder>…</system-reminder>`를 *가장 최근
   user message* 안으로 접어 넣습니다 (`_merge_system_reminder`, `pipeline.py` L341-372). 설계
   excerpt는 그 손실에 대해 정확합니다: `context.add_message(...)`는 존재하지만 *"the option-β merge
   into the last user message has no frame equivalent,"*이고, *"the `\n---\n` separator +
   reminder-stacking convention is yours to reimplement."*

인접한 일곱 번째, [[boson-tool-router]]에서: **`ToolRegistry.discover_tools()`** — `@tool`로
데코레이트된 function의 파일시스템 발견 — 에는 Pipecat 대응물이 없고, `tools=[...]` 목록을 생산하는
boson glue로 남습니다.

---

## 6. Pipecat이 되돌려 주는 것

이것도 vote가 아닙니다. 같은 ledger의 반대쪽입니다: tree 안에 존재하며 당신이 무엇을 결정하든 adoption과
함께 도착할 것들.

**1. stage를 위한 Flows node.** `NodeConfig(task_messages, role_message, functions, context_strategy,
respond_immediately)` 대 `StageDefinition.prompt/tools`. [[flows-insurance-example]]은
`stage_config.py`가 *"already is this graph, in declarative form."*임을 발견했습니다.

**2. `_GLOBAL_TOOLS`를 위한 `global_functions`.** 정확한 일치이고, 모든 node에서 섞여 들어갑니다:

```python
# src/pipecat/flows/manager.py L650-654
            # Build the node's function schemas (carrying handlers)
            new_functions: set[str] = set()

            # Mix in global functions that should be available at every node
            functions_list = self._global_functions + node_config.get("functions", [])
```

**3. `PreTool`을 위한 `function` pre-action.** `actions.py:285` — `"function"` action은 **항상**
기다리며, 그것은 boson의 generation-전-동기(synchronous-before-generation) semantics와 일치합니다.
잃는 것은 preamble-as-first-stream-chunk이고, 그것은 앞에 순서 지어진 별도의 `tts_say` action이
됩니다.

**4. transition별 context 절단으로서의 `ContextStrategy.RESET` — boson에는 오늘 없는 것.**

```python
# src/pipecat/flows/types.py L134-152
class ContextStrategy(Enum):
    """Strategy for managing context during node transitions.

    Parameters:
        APPEND: Append new messages to existing context (default).
        RESET: Reset context with new messages only.
        RESET_WITH_SUMMARY: Reset context but include an LLM-generated summary.

            .. deprecated:: 1.5.0
                Use :class:`LLMSummarizeContextFrame` instead — push it in a
                pre-action to trigger on-demand summarization during a node
                transition. See
                https://docs.pipecat.ai/guides/fundamentals/context-summarization.
                Will be removed in 2.0.0.
    """

    APPEND = "append"
    RESET = "reset"
    RESET_WITH_SUMMARY = "reset_with_summary"
```

**⚠️ 어느 member가 deprecated인지 주목하십시오.** `RESET`과 `APPEND`는 살아 있고,
**`RESET_WITH_SUMMARY`는 1.5.0부터 deprecated**이며 2.0.0에서 제거되고, pre-action에서
`LLMSummarizeContextFrame`을 push하는 것으로 대체됩니다. [[flows-insurance-example]]은
`RESET_WITH_SUMMARY`를 쓰는 `patient_intake.py` L306-313과 `warm_transfer.py` L360-365를 인용합니다 —
그 예제들은 deprecated 경로 위에 있습니다. `escalate_to_human`을 위해 warm-transfer briefing 패턴을
원한다면 enum member가 아니라 frame 위에 지으십시오.

**5. 현재 거의 눈이 먼 stack을 위한 [[ch-11/read]]의 observer plane.** §3.16.

**6. COMPACTION** — 서로 다른 failure mode를 가진 다른 subsystem이라서 [[ch-09/read]]에서 떼어졌습니다.
`LLMContextSummarizer` 더하기 `LLMAutoContextSummarizationConfig`가 `gateway/compact/`가 하는 일을
parameter 단위로 덮습니다: `threshold_messages=30 → max_unsummarized_messages`,
`keep_recent=10 → min_messages_after_summary`, 더하기 boson에는 없던 token trigger. **이 give-back
row의 falsifier 줄에 속하는 두 개의 실제 손실과 함께**, §3.15에서 다시 진술합니다 — give-back 중에서
조용히 물 수 있는 유일한 부분이기 때문입니다:

- pre/post compact hook에는 대응물이 없습니다 — `on_summary_applied`는 observability 전용입니다.
- `swap_compact`의 `<system-reminder>Active stage: …</system-reminder>` 재주입은 명시적으로 다시
  붙여야 하며, **그렇지 않으면 매 compaction마다 stage 정체성이 조용히 사라집니다.**

**7. [[ch-09/read]] §12에서 온 두 가지 — boson에 mechanism이 없는 capability라서 여기 나열합니다.**
`FunctionCallResultProperties`의 `run_llm=False` — Pipecat에서 *tool*이 "이 turn은 끝났다"고 말할 수
있는 유일한 자리이고, `end_call` / `transfer_to_human` / `schedule_callback`에 대해 그것은 model의
행동에 의존하지 않는 결정론적 통화 종료입니다. boson의 turn은 `agent_loop.py:363`에서 텍스트 전용
응답에 의해 끝나는데, 그것은 model이 *설득되어* 생산해야 하는 것입니다. 그리고 boson이 손으로 쓰는
여섯 개의 shaping에 대해, `BaseLLMAdapter.to_provider_tools_format` 뒤의 열두 개 provider adapter.

---

## 7. Deployment과 process topology (row 17)

옛 latency chapter에서 여기로 흡수되었습니다. 그것이 **latency 항이 아니라 결정의 입력**이기
때문이고, [[ch-04/read]]가 이미 배달한 Lina host topology 위에 짓기 때문입니다.

### 7.1 Pipecat은 DEVELOPMENT runner를 배포하고, 그렇게 말합니다

자기 자신의 banner:

```python
# src/pipecat/runner/run.py L392-400
def _print_dev_runner_banner():
    ...
        "ᓚᘏᗢ PIPECAT DEVELOPMENT RUNNER",
```

`run.py:1963`에서, `main()`이 서버를 시작하기 직전에 호출됩니다. module docstring은 그것을
*"This development runner executes Pipecat bots and provides the supporting infrastructure they
need."*라고 부릅니다. 그리고:

```bash
$ ls docs/
api
```

**repo에 deployment 문서는 없습니다** — `docs/`는 Sphinx scaffolding뿐입니다. 이것은 비판이 아니라
scoping fact이고, [[deployment-scaling]]이 그로부터 끌어내는 guideline은 *"Do not treat
`pipecat.runner.run.main()` as your production entrypoint."*입니다.

### 7.2 `main()`은 `workers=` 없는 `uvicorn.run(app, ...)` 하나로 끝납니다

```python
# src/pipecat/runner/run.py L1999
    uvicorn.run(app, host=args.host, port=args.port)
```

`workers=`도 없고 reload도 없습니다. **프로세스 하나.** 모든 session은 그 하나의 loop 위의
`asyncio.Task`입니다:

```python
# src/pipecat/runner/run.py L211-220
# Bot sessions started from a request handler outlive the response, and the event
# loop only holds a weak reference to a task, so one that nothing else references
# can be collected while it is still running.
_bot_sessions: set[asyncio.Task] = set()


def _start_bot_session(coro) -> asyncio.Task:
    """Run a bot in the background, holding a reference until it finishes."""
    task = asyncio.create_task(coro)
    _bot_sessions.add(task)
    task.add_done_callback(_bot_sessions.discard)
    return task
```

각 transport route는 `session_id = str(uuid.uuid4())`를 발행하고 `_start_bot_session(...)`을 호출합니다
(`run.py:821, 845, 909, 1284, 1392`). WebRTC는 다릅니다: `POST /start`는
`active_sessions[session_id] = body`를 등록할 뿐이고, bot은 나중에 offer handler에서
`background_tasks.add_task(bot_module.bot, runner_args)`를 통해 시작됩니다 (`run.py:1002-1023`).

그러므로: **동시성 = 하나의 loop 위의 동시 asyncio task. process pool도, worker count도, admission
control도, session별 CPU 격리도 `runner/` 어디에도 없습니다.**

한국어 tele-sales agent에게 이것이 중요한 숫자이고, 당신 자신의 단위로 진술할 가치가 있습니다. 통화
하나가 task 하나이고 통화 안의 CPU-bound 작업이 loop를 막는다면 — resampler, 로컬 VAD forward pass,
64 KiB control message에 대한 JSON parse — 그것은 *그 프로세스 안의 다른 모든 통화*를 막습니다.
그것에 대한 framework의 답은 `runner/`에 없습니다.

> 💡 **쉬운 설명 — "loop를 막는다"가 통화에 어떻게 보이나요?**
> asyncio는 협조적(cooperative)입니다. 어떤 코루틴이 `await` 없이 40 ms 동안 CPU를 쓰면, 그동안
> 같은 프로세스의 다른 모든 통화는 audio frame을 하나도 처리하지 못합니다. 통화가 30개면 30개 전부가
> 동시에 40 ms 끊깁니다. 증상은 "특정 통화가 느리다"가 아니라 **"전부가 동시에 뚝뚝 끊긴다"**이고,
> 원인은 그중 한 통화 안의 한 줄입니다. 이것이 process pool 없는 단일 loop 모델의 대가입니다.

### 7.3 진짜 runtime 단위는 worker입니다

`src/pipecat/workers/`. `BaseWorker` (1,565 L)가 activation, end/cancel, bus subscription, job RPC를
소유합니다. `PipelineWorker` (`pipeline/worker.py`, 1,506 L)가 사용자 pipeline을 감쌉니다.
`WorkerRunner` (`workers/runner.py:83`, 550 L)가 공유 `WorkerBus` + `WorkerRegistry`와 SIGINT/SIGTERM을
소유합니다. [[pipeline-task-runner]]에 따르면 `PipelineTask`와 `PipelineRunner`는 2.0.0에서 제거될
예정인 1.3.0 `@deprecated` alias입니다 — `PipelineWorker` + `WorkerRunner`에 대고 쓰십시오.

scaling 손잡이는 keyword 하나이고, 그 docstring이 곧 deployment 조언입니다:

```python
# src/pipecat/workers/runner.py L237-255
    async def run(
        self,
        worker: BaseWorker | None = None,
        *,
        auto_end: bool = True,
    ) -> None:
        """Run all added workers until the runner is stopped.

        By default (``auto_end=True``), the runner ends once every root
        worker has finished — so a single-pipeline bot naturally ends
        when its pipeline does. Multi-worker bots whose helpers run
        forever (e.g. waiting for bus messages) end by calling
        :meth:`end` / :meth:`cancel` from an event handler (typically on
        transport disconnect). For long-lived hosts that add and remove
        workers over many sessions (e.g. a FastAPI server), pass
        ``auto_end=False`` so the runner does not exit when no workers
        are left.
        """
```

`auto_end=True`는 container-per-call 모양입니다: 프로세스 하나, 통화 하나, pipeline이 끝나면 종료.
`auto_end=False`는 long-lived-host 모양입니다. **[[ch-04/read]]가 이것을 Lina의 host 모양으로
공급했고, 그것이 row 17의 mechanism 절반입니다.**

**lifecycle 안전 밸브**(`pipeline/worker.py` L91-100)가 framework의 유일한 비용 통제 장치입니다:
`IDLE_TIMEOUT_SECS = 300`, `CANCEL_TIMEOUT_SECS = 20.0`, `SETUP_TIMEOUT_SECS = 20.0`,
`START_TIMEOUT_SECS = 20.0`, `HEARTBEAT_SECS = 1.0`, `HEARTBEAT_MONITOR_SECS = 10.0`, 그리고
`idle_timeout_frames=(BotSpeakingFrame, UserSpeakingFrame)`, `cancel_on_idle_timeout=True`. 버려진
통화는 발화 없이 5분이 지나면 스스로 종료합니다. 그게 전부입니다.

### 7.4 프로세스 간 scale-out은 runner가 아니라 bus입니다

`bus/local/async_queue.py` (`AsyncQueueBus`, 프로세스 내 default) 대
`bus/network/{redis.py, pgmq.py}` (`RedisBus`, `PgmqBus`). `examples/multi-worker/README.md`에서:
*"Distributed bus — Same patterns, but workers run in separate processes (or machines)."*
`workers/proxy/websocket/`은 *"No shared bus required."*와 함께 점대점 forwarding을 줍니다.

[[theory-narrow-waist]] §4는 이것이 구조적으로 무엇인지 알아챘습니다: **두 번째의, 평행한
hourglass(모래시계).** `bus/messages.py`는 `BusMessage`를 `BusDataMessage` / `BusSystemMessage`로
쪼갭니다 — 같은 priority 분할, 다른 waist — 그리고 경계를 그대로 진술합니다: *"Bus messages are
independent of pipeline `Frame`s — if a worker needs to ship a frame between pipelines it wraps it in
a `BusFrameMessage`."* `BusFrameMessage(BusDataMessage)`는 문자 그대로 Frame-over-Bus이고, 하나의
codebase 안에서의 IP-over-everything 수(move)입니다.

### 7.5 cold start는 `StartFrame`이 아니라 `setup()`입니다

```python
# src/pipecat/observers/startup_timing_observer.py L95-107
class StartupTimingReport(BaseModel):
    """Report of startup timings for all measured processors.

    Parameters:
        start_time: Unix timestamp when the pipeline began setting up.
        total_duration_secs: Wall-clock time from the pipeline starting to set
            up until it had started. Processors are set up concurrently, so
            this is the span rather than the sum of what each cost.
        processor_timings: Per-processor timing data, in pipeline order.
    """
```

*"Processors are set up concurrently, so this is the span rather than the sum."* 그것을 budgeting
rule로 읽으십시오: 당신의 cold start는 총합이 아니라 **가장 느린** processor의 `setup()`입니다. 느린
model load 하나가 지배하고, 빠른 connect 열한 개는 그 뒤에 숨습니다. `ProcessorStartupTiming`은
`setup_duration_secs` (connect, auth, model load)를 `duration_secs`에서 떼어 내고,
`TransportTimingReport`는 `bot_connected_secs` (SFU 전용)와 `client_connected_secs`를 더합니다.
end-to-end greeting cold start는 `UserBotLatencyObserver.on_first_bot_speech_latency`입니다.

### 7.6 scaling 설정 표면 전체가 한 줄입니다

`pipecat init`은 `cli/templates/server/`에서 scaffold합니다. 생성되는 deploy 파일 전체가 여기 있습니다:

```jinja
{# src/pipecat/cli/templates/server/pcc-deploy.toml.jinja2 — the whole file, 14 lines #}
agent_name = "{{ project_name }}"
secret_set = "{{ project_name }}-secrets"
{% if enable_video_input or enable_video_output %}
agent_profile = "agent-2x"
{% else %}
agent_profile = "agent-1x"
{% endif %}
{% if enable_krisp %}
[krisp_viva]
	audio_filter = "tel"
{% endif %}

[scaling]
	min_agents = 1
```

`min_agents = 1`이 **repo의 scaling 설정 표면 전체입니다.** sizing과 warm-pool 하한은 framework
코드가 아니라 Pipecat Cloud 개념입니다. [[deployment-scaling]]에 따르면 기대되지만 부재한 것들:
Kubernetes manifest 없음, autoscaling 로직 없음, load-shedding 없음, session 수 제한 없음,
`WorkerRunner.end()` 너머의 graceful-drain helper 없음. `[krisp_viva] audio_filter = "tel"`도
주목하십시오 — 템플릿은 telephony audio가 존재한다는 것을 알고 있고, 그것은 framework 코드가 아니라
상용 filter입니다.

### 7.7 구체적인 선택, 그것이 row 17입니다

**긴 수명의 프로세스 하나** (`WorkerRunner(auto_end=False)`, 하나의 loop 위에 N개의 동시
`PipelineWorker`) **대 동시 통화 피크에 맞춰 크기를 잡은 warm pool을 가진 container-per-call.**

[[deployment-scaling]]은 boson-agent가 *이미* 첫 번째 모양임을 확립합니다:
`packages/gateway/gateway/__main__.py`는 *"owns process-scoped resources (including MCP
subprocesses),"*하는 `GatewayCore(config)` 하나를 만들고, rule/layer/stage를 한 번 발견해서
`GatewayWebSocketServer`에 넘깁니다. 그것의 `start()`는 단일한 `websockets.serve(...)`를 감싸고,
session들은 `_handle_connection`, `_reserve_session_dispatch`, `_replace_active_task`,
`_cancel_session_dispatch`, `forget_session`을 통해 그 안에서 다중화됩니다. **그러므로 boson은 Pipecat
runner와 충돌하지 않습니다 — `WorkerRunner`와 충돌합니다**, 그리고 그 충돌은 장부 정리의 교체입니다:
`_cancel_active_task` → worker cancel, `_start_silence_timer` → `idle_timeout_secs` /
`idle_timeout_frames`.

**이 row를 결정하는 두 가지.**

**하나 — telephony는 webhook으로 도착합니다.** container-per-session은 동시 통화 피크에 맞춘 warm
pool을 필요로 하는데, 단일 프로세스 모델은 그것이 전혀 필요 없습니다. 그것은 정확성이 아니라
비용/아키텍처 교환이고, row 17이 `runner/`를 읽어서 정리될 수 없는 이유입니다.

**둘 — `GatewayCore`의 프로세스 범위 MCP subprocess는 session마다 다시 생기지 않도록 worker 위쪽에
다시 호스팅되어야 합니다.** 이것이 날카로운 모서리입니다. 한 host에서 `auto_end=False`이면 그것들은
한 번 시작합니다. container-per-call이면 agent별 MCP 기동이 매 통화마다 지불되는 cold-start 비용이
되고, 그것은 `min_agents` warm-pool 하한 *위에* 얹힙니다. [[deployment-scaling]]은 그것을 *"a real
risk if Lina moves to container-per-call on Pipecat Cloud."*로 표시합니다.

**그리고 중립인 것 하나.** rule/layer/stage 발견은 시작 시점의 config 로딩이고 `bot(runner_args)`
안으로 그대로 port됩니다 — 그러나 그러면 `StartupTimingObserver`가 측정하는 session별 cold-start 경로
위에 착지합니다. 긴 수명의 프로세스 하나에서는 한 번 지불되고, container-per-call에서는 container마다
지불됩니다. 같은 코드, 다른 청구서.

**이제 §3.17로 돌아가서 세 칸을 채우십시오.**

---

## 8. 여기서 figure를 쓰십시오

[`figures/migration-map.html`](figures/migration-map.html)을 열고 머릿속이 아니라 그 안에서 ledger를
작업하십시오. figure가 산문이 할 수 없는 두 가지를 강제하기 때문입니다.

**그것은 "it depends"를 구조적으로 사용 불가능하게 만듭니다.** 각 row에는 세 개의 필수 cell이
있습니다 — vote, 자유 텍스트 assumption, 자유 텍스트 falsifier. assumption이나 falsifier가 빈 row는
빨간색으로 incomplete하게 렌더링되고, 모든 row가 완성될 때까지 export 버튼은 비활성 상태로 남습니다.
미리 선택된 vote도, default도, 강조된 추천도 없고, 당신이 클릭하지 않은 `KEEP` / `ADOPT` /
`HYBRID-WRAP` label은 어디에도 없습니다.

**그것은 동시에 참일 수 없는 조합을 표시합니다.** 세 개이고, 각각 이유가 붙어 있습니다:

- text-native Gateway tool을 유지하면서 Pipecat의 LLM service를 adopt하기 (§4.1);
- telephony를 추가하면서 realtime_voice의 Silero를 유지하기 — 그것은 8 kHz를 하드 거부합니다
  (§3.1, §3.7);
- word-timestamp 서비스 여섯 개 바깥의 TTS를 선택하면서 `AudioTextPlayoutLedger`를 버리기 (§3.3,
  §3.5) — 그 여섯이 행동 test가 아니라 grep이라는 리마인더와 함께.

**그리고 그곳이 하나의 measurement이 착지하는 곳입니다.** §2의 두 숫자를 MEASUREMENT 패널에 붙여
넣으십시오. 그러기 전까지 패널은 실행 명령만을 내용으로 하여 회색으로 앉아 있습니다. 붙여 넣은 뒤에는
ch-12의 ghost 추정치에 대고 당신의 bar를 그리고, rule-evaluation 칸에 진짜 숫자를 넣어 [[ch-11/read]]의
waterfall을 다시 렌더링합니다. 두 값 모두 `localStorage`에 남습니다.

**그것으로 무엇을 할지 한 문장으로:** 열일곱 개의 vote를 한자리에서 전부 던지고, 그런 다음 목록을 두
번째로 훑으면서 뒤집혀도 자기 vote를 바꾸지 않을 assumption을 전부 지우십시오. 두 번째 통과에서
살아남는 것이 진짜 assumption 집합이고, 그것은 첫 번째 것보다 훨씬 짧을 것입니다.

---

## 9. Deliverable

다섯 개의 artifact. 이 chapter에서 "done"이 뜻하는 것이 이것입니다.

**1. assumption과 falsifier가 함께 채워진, 완성된 열일곱 행짜리 vote 표.** figure에서 export합니다.
모든 row가 채워질 것. 빈칸 없음, "TBD" 없음, assumption이 vote를 되풀이하는 row 없음.

**2. `FlowManager`를 processor 목록 바깥에 둔, 끝에서 끝까지 그려진 target architecture.** 목록 안이
아닙니다 — `FlowManager`는 평범한 class이고 (§3.12) `worker.queue_frames`를 통해 바깥에서 구동합니다.
당신의 다이어그램이 그것을 pipeline 안에 그린다면 그 다이어그램은 mechanism 오류를 encode한 것이고,
transition 순서에 관한 모든 하류 결론이 그 오류를 물려받습니다.

**3. 매 단계에서 agent가 ship 가능한 상태로 남는 migration 순서 — 당신이 순서를 정하고 단계마다 한
문장으로 방어할 것.** 흥미로운 부분은 제약입니다: *매 단계에서 ship 가능*은 어떤 단계도 Lina가 통화를
받지 못하는 상태로 남겨서는 안 된다는 뜻입니다. 그 제약 하나가 여러 순서를 탈락시킵니다 — 그것이
"LLM loop를 먼저 교체한다"와 "transport를 먼저 교체한다"가 순서만이 아니라 종류에서 다른 이유입니다.
단계마다 한 문장, 그리고 그 문장은 그 단계가 무엇을 *성취하는지*가 아니라 무엇을 *de-risk하는지*를
지명해야 합니다.

**4. 최소 한 번 실행된 `ch-13/tier2-probe.py`, 인쇄된 P50/P95를 figure에 붙여 넣은 것 — 그리고 그것이
[[ch-11/read]] 예산에 무슨 일을 했는지에 대한 note.** 그 마지막 절이 deliverable이지, 실행 자체가
아닙니다. 측정값을 넣었을 때 예산의 pre-LLM 절반이 P50 ≤ 1.0 s 선 안에 머물렀습니까, 아닙니까?
그 문장을 쓰십시오.

**5. 남은 falsifier들을 owner와 함께 watchlist로 — 결정을 미루는 이유로서가 아니라.** §10.

---

## 10. Watchlist: 실행되지 않은 세 개의 measurement

이것들은 blocker가 아니라 falsifier입니다. 각각은 owner와 precondition을 받습니다. figure에서도 그렇게
스타일링되어 있습니다 — measurement 패널 아래에, 명시적으로 falsifier로서.

| # | Observation | Type | 뒤엎을 row | Precondition | Owner |
|---|---|---|---|---|---|
| 1 | 후보 STT별, 8 kHz μ-law에서의 Korean WER | benchmark | 2, 그리고 결합에 의해 1과 7 | Korean STT 계약 + labelled Lina corpus + telephony-band audio | |
| 2 | 단독 한국어 backchannel에 대한 interruption-broadcast → `TranscriptionFrame` 간격 | corpus | 1, 5, 그리고 §4.2의 filler-gate 위치 | 녹음된 8 kHz telephony audio + 돌아가는 Pipecat pipeline | |
| 3 | transition-swallow 설계 아래 turn당 정확히 하나의 inference-triggering frame | code assertion | 10, 12, 13 | 만들어진 `BosonRuleProcessor` prototype + `llm` 입력의 `FrameLogger` | |

**각각을 소망이 아니라 실행 가능하게 만드는 note.**

**#1.** harness는 존재합니다: `https://github.com/pipecat-ai/stt-benchmark`,
[[stt-korean-providers]]에서 지명되었습니다. 존재하지 않는 것은 당신의 labelled corpus입니다. 그것이
진짜 precondition이고, engineering 작업이 아니라 데이터 라벨링 작업입니다. 그것을 돌릴 shortlist는
§3.2의 Korean-verified 아홉이고, **영어를 가정한** latency 표 순서로 정렬되어 있습니다 — 그 사실
자체가 benchmark가 중요한 이유의 일부입니다.

**#2.** [[design-boson-rules-on-pipecat]] §5는 예상되는 부호를 미리 진술합니다: *"If the gap is
positive (it will be, ~always)."* boson은 `"네"`를 *내용*과 `pre_turn_status`로 거릅니다. Pipecat은
STT 상류에서 VAD 에너지로 interrupt하므로, gate가 텍스트를 보기 전에 봇은 이미 interrupt되어 있습니다.
확인되면 그 귀결에 이름이 붙어 있습니다: *"a custom `BaseUserTurnStartStrategy` … that withholds
turn-start until a transcript exists is mandatory — and it costs the unmeasured Korean STT TTFS."*
결합을 주목하십시오: 이 falsifier의 치료법은 #1이 측정하는 그 숫자를 비용으로 치릅니다. 또한
[[boson-interrupt-subsystem]]에서, boson의 텍스트 전용 detector의 정확한 대응물이 상류에
`TranscriptionUserTurnStartStrategy`로 이미 존재한다는 점도 주목하십시오 — 그러므로 boson의 현재
동작은 *지원되는 Pipecat 구성*이고, 그것은 이 measurement에 따라 행동하는 비용을 바꿉니다.

**#3.** assertion은 excerpt에서 그대로: `llm` 입력의 `FrameLogger`가 turn당 정확히 **하나**의
inference-triggering frame이 도착하고 `LLMSetToolsFrame`이 그보다 앞선다고 assert하는 것. 이름 붙은 두
failure mode는 generation 0회(양쪽 경로 모두 삼켜짐)와 2회(node도 돌고 context도 push됨)입니다. 이것이
셋 중 얻기 가장 싼 것입니다 — audio도 벤더도 필요 없습니다 — 그러나 아직 존재하지 않는 prototype이
필요하고, 그래서 §2가 아니라 watchlist에 있습니다.

**규율에 관한 note.** owner 없는 falsifier는 소망입니다. 표를 export할 때 당신 자신을 포함해 모든
owner 칸에 이름을 넣고, precondition 옆에 날짜를 넣으십시오. **owner가 있고 날짜가 없는 falsifier는
이름이 붙은 소망입니다.**

---

## 11. 머릿속에 담아 둘 것

열두 개, ledger가 쓰는 모양으로.

1. **열일곱 개의 row.** chapter도, figure도, 당신의 export도 전부 열일곱이라고 말합니다. 그중 어느
   하나가 열여섯이나 열여덟이라고 말한다면, 하나가 표류한 것입니다.
2. **assumption 없는 vote는 추측입니다.** rule 1의 뒷부분이 일을 하는 부분입니다.
3. **falsifier는 선호가 아니라 observation을 지명합니다.** 당신이 볼 수 있는 무엇으로도 한 row를 뒤엎을
   수 없다면, 그 row는 잘못 진술된 것입니다 — 다시 진술하거나 쪼개십시오.
4. **하나의 measurement이 실행됩니다.** Tier-2 P50/P95, `ch-13/tier2-probe.py` — 넷 중 carrier도, STT
   계약도, audio도 필요 없는 유일한 것이기 때문입니다.
5. **offline figure는 endpoint를 호출할 수 없습니다.** measurement은 process에 살고, visualisation은
   browser에 삽니다. 구조적으로 할 수 없는 일을 artifact에 할당하지 마십시오.
6. **N=9에서 nearest-rank P95는 최댓값입니다.** 최소 40회 iteration을 돌리십시오, 그러지 않으면 당신은
   자기 최악의 운을 보고하는 것입니다.
7. **`round(0.2 / (512/16000)) = 6`, 그리고 `round(0.2 / (256/8000)) = 6`.** Pipecat의 endpointing
   frame count는 sample rate에 불변이고, excerpt의 "7"은 틀렸습니다.
8. **tree 안의 Deepgram TTFS P99는 `0.45`가 아니라 `0.35`입니다.** 설계 excerpt의 비교는 Tier-2
   청구서를 과소평가합니다.
9. **여섯 개의 Korean word-timestamp TTS 서비스는 행동 test가 아니라 grep입니다.** 검증된 집합이
   아니라 후보의 상한입니다.
10. **한국어 accuracy 숫자도, 8 kHz 숫자도 tree 어디에도 없습니다.** WER에 대한 grep hit은 어떤
    서비스, 어떤 언어에 대해서도 0입니다.
11. **`min_agents = 1`이 scaling 설정 표면 전체이고**, `uvicorn.run(app, ...)`에는 `workers=`가
    없습니다. 당신이 달리 만들기 전까지 동시성은 하나의 loop 위의 asyncio task입니다.
12. **두 개의 collision은 환원 불가능합니다** — agent boundary (row 10, 11)와 rule layer (row 13) —
    그리고 이 chapter는 둘 다 진술하되 어느 것도 해소하지 않았습니다, 의도적으로.

그리고 이 연습 전체에 관한 한 문장: **ledger의 가치는 vote가 옳다는 데 있는 게 아니라, assumption과
falsifier가 그중 하나가 틀렸다는 것을 알아내는 비용을 싸게 만든다는 데 있습니다.** tripwire가 달린
열일곱 행짜리 표는 자산입니다. 옆에 아무것도 쓰이지 않은 확신에 찬 vote들의 열일곱 행짜리 표는, 같은
문서에서 쓸모 있는 절반을 지운 것입니다.

---

## 다음 챕터로

다음 chapter는 없습니다. 이것이 마지막이고, 그것이 앞으로 넘기는 것은 chapter가 아닙니다 — 네 개의
파일과 하나의 습관입니다.

**네 개의 파일.** export된 vote 표; `FlowManager`를 processor 목록 바깥에 둔 target architecture;
단계마다 방어된 한 문장이 달린 migration 순서; 그리고 output에 진짜 숫자가 든 `tier2-probe.py`와 그
숫자를 붙여 넣은 figure. 네 개 전부를 당신의 팀이 review할 수 있는 곳에 두십시오. rule 1의 목적
전체가 그것들을 reviewable하게 만드는 것이었으니까요.

**습관.** 당신은 열두 chapter 동안 결정하지 말라는 말을 들었고, 한 chapter 동안 진술된 assumption 아래
쓰여진 tripwire와 함께 열일곱 번 결정했습니다. 그 비대칭이 곧 교수법이었습니다: vote가 허용되기 전에
evidence가 완결되어야 했고, vote가 세어지기 전에 falsifiable해야 했습니다. 그 모양을 Lina에서 당신이
내릴 다음 architecture 결정으로 가져가십시오 — 그것은 Pipecat에 관한 것이 아닐 겁니다.

**그리고 probe를 다시 돌리십시오.** 한 번이 아니라 — model이 바뀔 때마다, serving stack이 바뀔
때마다, 누군가 세 번째 `check_type="llm"` rule을 추가할 때마다. 40초가 걸리고, 이 course 전체에서
당신이 스스로 만든 유일한 숫자입니다.

마지막 두 개의 포인터를, 당신이 만들기 시작하는 순간 곧바로 나올 것들이라서 남깁니다.
[[design-boson-rules-on-pipecat]] 자신의 migration angle은 한 문장이고, vote를 던진 뒤에 다시 읽을
가치가 있습니다: *"the port is one processor, not a framework port."* 그리고
[[pipecat-design-philosophy]]의 deprecation registry — 살아 있는 deprecation 391개, 전부
`removed_in == "2.0.0"`, 97%가 지명된 대체물을 가짐 — 는 당신이 찾을 수 있는 최고의 migration
backlog입니다. 한 줄을 쓰기도 전에 당신의 정확한 미래 breakage를 알려 주기 때문입니다.
