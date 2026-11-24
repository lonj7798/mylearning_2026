# 23a — 구성 레코드, 혼합 비율, 공정한 비교

<!-- scope: proposed implementation records, controlled comparisons, metrics, and failure repair for rarity units, sampling support, retention bias, and source-grounded knowledge coverage
     deps: [[read_kor]]
     see-also: [[read_kor]], [[methods_kor]], [[long-tail-papers]]
     sources: [[long-tail-knowledge]], [[graphgen]], [[knowledge-density-synthesis]], [[data-value-scaling]], [[long-tail-knowledge-survey]]
-->

## 1. 실제로 쓸 구성 레코드

제안 필드: family ID, 태스크, 현재 정책 버전, 근거 자료, 올바른 행동, 생성 조건, 검증 결과, 탈락 이유, 희소성 근거입니다. 희소성 근거는 계산 단위가 있는 실제 사용 빈도, 담당자의 사례, 의도적인 스트레스 설계 중 무엇인지 표시합니다. 스트레스 시험도 가치가 있지만 실제 빈도라고 주장할 필요는 없습니다.

두 번 수정된 일정이면 수정 순서 전체, 최신 근거 날짜, 예약 동의 여부, 각 값의 출처를 기록합니다. 누락 필드는 별도 상태이며 가장 흔한 값으로 채우지 않습니다. 생성 후보와 유효하게 남긴 개수를 함께 셉니다. 후보 200개에서 정답 20개라면 quota만 늘리기보다 생성기·근거를 개선해야 할 수 있습니다.

## 2. 데이터 비율의 분모가 다르다

흔한 대화 100개가 각각 200 training tokens, 희소 대화 10개가 각각 2,000 tokens라면 대화 비율은 10/110=9.1%, 토큰 비율은 20,000/40,000=50%입니다. 대화 수만 보고하면 학습 노출 절반이 가려집니다. Response-only loss라면 supervised assistant tokens도 별도로 보고합니다. 입력 토큰은 문맥과 compute에 영향을 주지만 같은 loss를 받지 않을 수 있습니다.

이는 계산 예시이며 권장 혼합이 아닙니다. 원본 대화 수·샘플링/반복 뒤 실제 본 토큰·감독 토큰을 각각 남깁니다. 길이가 다른 데이터에 epoch만 맞추면 비용이 다릅니다. 큰 누적 pool의 1 epoch는 작은 replace pool보다 훨씬 비쌀 수 있습니다.

## 3. 통제 비교

같은 base checkpoint와 흔한 기준 세트에서 시작합니다. 일반 생성과 희소 조건 생성의 training-token 예산을 맞춰 비교합니다. 이어서 실제 기준 데이터를 포함하는 경우와 포함하지 않는 경우를 비교할 수 있습니다. Tokenizer·문맥 길이·masking·optimizer·schedule·추론 조건·시험 항목을 유지합니다. 생성·검증 비용은 학습 비용과 따로 보고합니다.

실제 사용형 시험과 명시적인 rare/stress 시험을 함께 사용합니다. 가까운 변형을 생성하기 전에 family 단위로 분리합니다. 혼합 결정용 validation과 마지막 reporting test를 나누며 선별 기준 항목은 tuning에 쓰인 자료로 봅니다. 생성기·정책·혼합을 한꺼번에 바꾸면 전체 pipeline 효과이며 원인을 분리할 수 없습니다.

## 4. 결과 해석

남은 정답률은 올랐는데 모델이 그대로라면 오라벨이 주된 문제가 아니었을 수 있습니다. 희소 성능 상승과 흔한 성능 하락은 제품 선택이 필요한 trade-off입니다. 생성기 같은 표현에서만 좋아지면 shortcut일 수 있습니다. 모든 비교군이 plateau하면 용량·비용·중복·시험 ceiling도 가능하며 그것만으로 collapse가 아닙니다.

다음 행동은 자동적인 합성 추가보다 사례별 개수·오류 조사입니다. 유효한 희소 family는 보존하고 잘못된 정답은 근거로 수정하며 같은 예산으로 재평가합니다. 미지의 새 스킬은 여전히 공백이므로 임시 다중 태그 분류를 실제 사례에 따라 갱신합니다.

[본문](read_kor.md) · [연구 방법](methods_kor.md) · [다음 장](../ch-23b/read_kor.md) · [English](practice.md).

