# 23d — 사람 음성 benchmark와 spoken reward

<!-- scope: primary-source setups, measured outcomes, and transfer limits for speech layers, interruptions, action commitments, and matched acoustic diagnostics
     deps: [[read_kor]]
     see-also: [[read_kor]], [[practice_kor]], [[long-tail-papers]]
     sources: [[audio-multichallenge]], [[sdiareward]], [[deepdialogue]]
-->

## 1. Audio MultiChallenge: 목표에 따른 즉흥 발화

**Result (single study).** [[audio-multichallenge]]는 47명의 452개 대화와 1,712개 rubric을 평가합니다. 상황 설계·목표에 따라 사람이 즉흥적으로 말하며 합성 대본을 그대로 읽지 않습니다(§3). 말실수·환경 변화가 남고 기억·지시·일관성과 말 도중 수정·되돌림을 다룹니다.

엄선된 어려운 대화는 실제 고객 통화의 무작위 표본이 아닙니다. 낮은 점수는 해당 조건의 실패이며 발생 빈도는 아닙니다. 태스크·사람 발화·명시적 판정 기준을 참고할 수 있으나 상품 사실·정책·도구 선행 조건은 직접 제공해야 합니다. 이 benchmark는 fixed-context 평가이며 실시간 controller나 barge-in을 닫힌 loop로 검증한 것은 아닙니다.

## 2. SDiaReward: 전체 episode preference

**Result (single study).** [[sdiareward]]는 modality·구어체를 이용해 spoken dialogue reward를 만들고 평가합니다. §3.1은 modality pairing과 colloquialness pairing을 나누며 고립된 글 답만이 아닌 전체 episode를 봅니다. 운율·감정·구어체도 선호에 영향을 줍니다.

§6은 도메인별 절대 score offset을 설명합니다. 절대값을 전 분야의 보정된 품질 척도로 보지 않으며 같은 조건의 상대 비교가 더 적절합니다. 사실·도구 검사는 별도로 유지합니다. Reward model이 실제 희소 빈도나 예약 database 정답을 자동으로 안다는 결과는 아닙니다.

**Interpretation.** 음성 preference로 전달을 조사하며 명시적 태스크 rubric을 함께 사용합니다. 최적화에 쓰면 독립 평가를 유지합니다. Reward 상승과 수정 날짜 정확도 하락은 불일치이지 전체 성공은 아닙니다.

## 3. 표현적인 TTS의 한계

**Result (single study).** [[deepdialogue]]는 분야·감정 텍스트를 음성으로 만들고 speech emotion recognition과 RAVDESS 전이를 시험합니다(§4.2). App.B.3의 텍스트 감정 20개는 음성 범주 8개로 매핑됩니다. 20개의 자연스럽게 독립적인 음성 정서를 입증하지 않습니다.

TTS는 대본과 반복 조건을 유지해 진단에 유용하지만 즉흥 겹침·실제 고객 억양·live action timing을 입증하지 않습니다. 감정 인식 개선을 세일즈 성공으로 바꾸어 부르지 않습니다. 긴 대화가 text filtering에서 더 자주 탈락하므로 음성화 전 남은 상호작용을 확인합니다.

## 4. 연구별 참고점

| 출처 | 참고할 평가 | 추가 세일즈 근거 |
|---|---|---|
| Audio MultiChallenge | 상황 설계·사람 음성·rubric | 상품/방침 정답·빈도 |
| SDiaReward | 전체 음성 preference | 독립 사실/도구/타이밍 |
| DeepDialogue | 통제된 감정 TTS | 자연 수정·겹침·화자 |

**Open question.** 특정 학습 개입이 새 화자·정책에서 전체 시스템을 개선하는가? 텍스트 점수·reward·TTS 감정은 각각 일부입니다. 실제 배포 설정에서 층과 통합을 시험합니다.

[본문](read_kor.md) · [실습](practice_kor.md) · [English](methods.md).

