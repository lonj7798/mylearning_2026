# 23d — 같은 상황의 평가 행렬과 오류 위치

<!-- scope: proposed implementation records, controlled comparisons, metrics, and failure repair for speech layers, interruptions, action commitments, and matched acoustic diagnostics
     deps: [[read_kor]]
     see-also: [[read_kor]], [[methods_kor]], [[long-tail-papers]]
     sources: [[audio-multichallenge]], [[sdiareward]], [[deepdialogue]]
-->

## 1. 녹음 전에 의미 상황을 정한다

제안 필드: family ID, 사실·수정, 현재 정책, 도구 schema·실제 상태, 화자/녹음 출처, 허용 다음 행동, 필수 값, 타이밍 사건입니다. 요청된 생성 조건과 관찰된 녹음 특성을 나눕니다. 설계에서 망설이는 수정을 요구해도 실제로 있었고 알아들을 수 있었는지 검토합니다.

예제는 correction이 commit보다 먼저이며 확인이 누락된 상태입니다. 이미 예약한 recovery는 도구 결과가 있는 별도 family로 구성합니다. 두 경우를 같은 gold call로 합치지 않습니다. 불확실할 때 다시 질문하는 것을 포함한 허용 행동 집합이 필요합니다.

## 2. Matched 진단 행렬

| 조건 | 고정 정보 | 바뀌는 입력 | 진단 목적 |
|---|---|---|---|
| 깨끗한 전사 | 상황·정책 | 사람이 확인한 단어 | 문맥/행동 기준 |
| 실제 ASR 전사 | 같은 녹음 의미 | 인식 출력 | ASR 손실의 이후 영향 |
| 통제 TTS | 같은 글·정책 | voice/속도/소음 | 반복 음향 probe |
| 사람 즉흥 | 같은 목표·제약 | 표현·타이밍 | 덜 대본적인 상호작용 |
| Live interruption | 같은 수정 사실 | 순서/commit 상태 | Controller·상태 통합 |

ASR pipeline은 같은 LLM에 clean/actual 전사를 넣습니다. Clean은 성공하고 actual은 실패하면 LLM target을 바꾸기 전 인식 차이를 조사합니다. 둘 다 실패하면 문맥·정책도 원인 후보입니다. Native audio에서는 글·음성 조건이 다르므로 진단이지 자동으로 공정한 modality 순위 비교는 아닙니다.

## 3. 범위가 있는 지표

WER=(substitution+deletion+insertion)/reference word count입니다. 낮은 WER도 핵심 날짜 하나를 틀릴 수 있고 높은 WER는 무해한 filler일 수 있습니다. 필수 값 복구를 별도로 보고 canonicalization·모호한 음성의 human 판단을 정합니다. 완벽한 전사도 지시 범위를 확인하지는 않습니다.

행동은 도구 선택·유효 no-call·필수 인자·근거·선행 조건입니다. 전달은 이해 가능성·중요 값 발음·속도·요청 말투, 타이밍은 응답 시작·인터럽트 감지 후 멈춤·commit 경계를 기록합니다. 허용 시간은 제품 요구로 정하며 보편적 latency 값을 만들지 않습니다.

인터럽트에 기준 timestamp와 감지/멈춤 정의가 필요합니다. 못 감지한 사건을 숨기고 감지 뒤 stop time만 보고하지 않습니다. Tool request와 confirmation을 나눕니다. 요청 출력이 완료 증거는 아닙니다.

## 4. 필요한 시점에 값을 다시 확인한다

현재 정책에 따라 예약 전에 ‘목요일 십이 일 오후 세 시’를 다시 확인할 수 있습니다. 오래된 날짜를 반복하거나 근거 없는 침묵을 동의로 보면 안 됩니다. 정책이 명시적으로 지원하는 경우만 그 해석을 사용합니다. 반복은 확인 기회이지 듣기의 대체가 아닙니다. 겹침으로 불확실하면 질문합니다.

마지막 값뿐 아니라 commitment 순서를 봅니다. 최종 문장은 맞아도 이미 옛 날짜로 예약했을 수 있습니다. 사건 기록·도구 상태를 적절한 개인정보 처리와 함께 보관합니다. 개별 마지막 문장에 오류가 없어도 전체 pipeline 실패입니다.

## 5. 학습 비교와 수정 선택

가능하면 text data·acoustic data·controller 변경을 각각 비교하고 모델·추론 조건을 유지합니다. 화자·상황 family를 hold-out하며 가까운 음성/글은 같은 split에 둡니다. Clean common·희소 수정·통제 stress를 수집 출처와 함께 평가합니다.

인식 실패면 text SFT가 못 고칠 수 있습니다. Clean text에서도 상태가 틀리면 검증된 correction family가 후보입니다. Commit timing이면 orchestration을 수정해 경계를 시험합니다. 정답은 맞고 전달만 나쁘면 같은 단어로 음성 component를 조정합니다. 여러 변경도 유용할 수 있지만 전체 효과를 보고하며 모든 원인을 selection에 돌리지 않습니다.

[본문](read_kor.md) · [방법](methods_kor.md) · [다음 장](../ch-24/read_kor.md) · [English](practice.md).

