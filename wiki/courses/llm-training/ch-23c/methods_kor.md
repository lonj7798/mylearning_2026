# 23c — 실제 표현, 페르소나 실험, 감정 합성

<!-- scope: primary-source setups, measured outcomes, and transfer limits for policy-conditioned expression, emotion, lexical metrics, and valid tone transformations
     deps: [[read_kor]]
     see-also: [[read_kor]], [[practice_kor]], [[long-tail-papers]]
     sources: [[realtalk]], [[deepdialogue]], [[persona-lexical-diversity]]
-->

## 1. REALTALK의 장기 메시지와 좁은 표본

**Result (single study).** [[realtalk]]은 미국 영어 모어 화자 18–25세 열 명을 21일 관찰합니다(§3.1). 실제 메시지와 합성 대화의 감정·페르소나를 비교하며 §5.2는 실제의 더 넓은 감정·말투 변화와 합성의 일관된 긍정성·과도한 공감을 관찰합니다.

생성이 너무 매끄럽고 안정적이고 동조적인지 묻는 기준입니다. 전체 고객을 대표하지는 않습니다. §8의 인구통계·일상 채팅 한계가 있고 음성 현실성도 입증하지 않습니다. §4.3 극성 entropy는 하나의 지표이며 감정 품질 전체가 아닙니다.

**Interpretation.** 세일즈에서는 태스크·조건을 맞춘 실제 표현과 합성을 비교할 수 있습니다. 차이는 조사 신호이지 인간 행동 전체를 복제하라는 결론이 아닙니다. 실제 대화도 현재 방침을 위반할 수 있으므로 응답은 정책 기반으로 검증합니다.

## 2. 세부 페르소나와 길이 교란

**Result (single study).** [[persona-lexical-diversity]]는 Dolly·PersonaHub와 Llama/DeepSeek 응답을 포함해 인간/합성 instruction·persona 생성을 분석합니다(§§4–5). 시험 조건에서 persona가 어휘를 늘리지만 상세 정보의 추가 효과는 길이 통제 대비 제한적이고 내용 중복도 높습니다.

긴 답은 다른 단어를 쓸 기회가 많습니다. 길이를 맞추거나 통제하고 지표를 해석해야 합니다. 도구 능력·정책 적응·음성 감정·억양을 측정하지 않습니다. 운영자 선호는 따라야 할 지시이고 인구통계 persona는 생성 조건이므로 서로 대체되지 않습니다.

**Interpretation.** 비슷한 길이와 태스크의 baseline으로 비교하고 새 어휘가 새로운 표현 패턴인지 같은 행동을 길게 쓰는 것인지 봅니다. 어휘 새로움만을 목표로 삼지 않고 행동은 별도 시험합니다.

## 3. DeepDialogue: 감정 계획과 필터된 텍스트·음성

**Result (single study).** [[deepdialogue]]는 41개 분야·20개 텍스트 감정·방향성 전환 그래프를 사용합니다(§§3.1–3.2). 이력·분야·현재 감정으로 생성하며 아홉 모델 설정에서 후보 65,600개 중 40,150개 대화를 남깁니다. 응답은 최대 25단어 제약이 있습니다.

App.B.2는 긴 대화가 품질 검사에서 더 자주 실패해 남은 길이가 짧게 치우친다고 보고합니다. App.B.3은 텍스트 감정 20개를 음성 범주 8개로 매핑합니다. 20개 라벨이 20개 audible class가 아닙니다. §4.2는 speech emotion recognition과 RAVDESS 전이를 시험하며 세일즈/자연스러운 턴 관리 전체는 아닙니다.

전환 계획·후보·남은 흐름을 나눠 검사합니다. 계획한 희소 경로도 필터가 제거할 수 있습니다. 표현적인 TTS는 제어에 유용하지만 자연스러운 겹침·억양·고객 빈도를 입증하지 못합니다.

## 비교와 남은 질문

| 근거 | 참고할 제어/기준 | 미검증 범위 |
|---|---|---|
| REALTALK | 실제 장기 변화 | 세일즈 모집단·음성 |
| Persona 연구 | 상세함·길이 통제 | 행동·감정 적절성 |
| DeepDialogue | 분야·감정 흐름 | 실제 희소 상호작용·영업 개선 |

**Open question.** 실제 표현·검증된 수정·persona/emotion 생성을 어떻게 섞어야 모델이 좋아지고 방침·사실도 유지되는가? 최적 혼합을 정해주는 결과는 없습니다. 제품의 rubric과 예산으로 시험해야 합니다.

[본문](read_kor.md) · [실습](practice_kor.md) · [English](methods.md).

