# Chapter 23 — Discussion Record, October 7, 2026

<!-- scope: learner answers, tutor clarifications, and unresolved questions from chapter-23 discussion
     deps: [[read]], [[qa-2026-10]]
     see-also: [[qa]], [[../ch-22/discuss]], [[../ch-23a/read]]
-->

[Reading Q&A](qa-2026-10.md) · [Chapter](read.md) · [Continue to 23a](../ch-23a/read.md).
이 기록은 대화의 의미를 압축한 기록이며 축어 전사나 mastery 판정이 아니다. 기존 역사적 Q&A는 보존한다.

## 1 — 정답인 쉬운 데이터만 있으면 충분한가?

**Learner:** 다양성이 부족하므로 충분하지 않다고 답했다. 설명은 주로 난이도 차이에 초점이 있었다.
**Tutor clarification:** rare≠hard. 기존 pretraining 능력도 있으므로 새 학습 데이터가 쉽다는 이유만으로 모델의 모든 능력이 쉬운 사례에 제한된다고 결론 내리지 않는다. 정확성과 coverage를 따로 본다.

## 2 — 빠진 상황은 narrowing인가, coverage gap인가?

**Learner:** 변경된 요구사항이나 필터 전후 조건이 없는 질문에서 어느 쪽으로도 단정하지 않고 coverage gap을 제안했다.
**Tutor clarification:** 질문에 필터 이전에 그 상황이 존재했다는 조건이 빠져 있었다. 필터가 제거했다면 narrowing, 처음부터 없었다면 gap이다. 이 응답을 잘못된 분류로 기록하지 않는다.

## 3 — 실패한 사례를 버릴 것인가?

**Learner:** 실패한 예제를 고쳐 쓰는 방법을 제안했다.
**Tutor clarification:** 희소한 상황을 유지하고 정답 응답·인자를 재작성한 뒤 검증할 수 있다. 틀린 답을 positive target으로 유지하는 것과 구분한다. 실제 개선 여부는 아직 실험하지 않았다.

## 4 — Accumulation이 안정적이면 충분한가?

**Learner:** 더 많은 benchmark로 확인하고 싶다고 답했다.
**Tutor clarification:** benchmark 범위 확대와 함께 세대 안정성, 데이터 양의 scaling, 원본-only 비교를 나눠야 한다. 자료 비율과 출처를 고정한 양 비교는 quantity 효과를 더 분명하게 볼 수 있다.
**Learner:** 고정 비율이 특정 태스크 품질을 확인하는 데 도움이 된다고 설명했다.
**Remaining uncertainty:** 실제 모델·예산·평가 세트를 정하지 않았으므로 scaling 실험 설계는 아직 제안 단계다.

## 5 — 두 baseline이 모두 plateau면 collapse인가?

**Learner:** 반드시 collapse는 아니지만 큰 문제라고 답했다.
**Tutor clarification:** capacity·compute·중복·benchmark ceiling도 가능하다. 혼합 모델만 plateau하고 original-only가 계속 좋아지면 synthetic mixture 문제를 뒷받침하는 증거가 더 강하다. 결과가 없는 상태에서 원인을 확정하지 않는다.

## 6 — Verifier가 정답만 남기면 모델도 좋아지는가?

**Learner:** 정확도와 coverage의 trade-off 및 희소 사례 오류를 지적했다.
**Tutor clarification:** retained-data accuracy와 trained-model accuracy는 다르다. 옳은 rare sample도 거절되면 long-tail coverage를 잃을 수 있다.
**Learner:** verifier 평가에 관련 benchmark를 쓰겠다고 제안했다.
**Tutor clarification:** 독립 gold 표본에서 correct/wrong keep rate를 각각 측정하고 common/rare를 나누는 것이 필요하다. 후보 정답률 하나로는 오탈락을 알 수 없다.

## 7 — 더 많이 생성하면 누락이 해결되는가?

**Learner:** 생성기의 확률 분포에 의존하므로 pipeline에서 희소 조건을 명시해야 한다고 답했다.
**Tutor clarification:** 양수 확률의 사례는 샘플 수 증가가 도움이 될 수 있으나 없는 support를 보장하지 못한다. coverage 부족과 coverage 감소를 구분한다.

## 8 — Verifier 선호에 맞춘 학습과 독립 평가

**Learner:** verifier의 선호에 overfit할 가능성을 제기했다.
**Tutor clarification:** 그 가능성 자체가 overfitting의 증거는 아니다. 독립적인 정확도·행동 평가가 필요하다.
**Learner:** 더 큰 LLM에 전체 이력과 system policy를 제공해 확인하는 방식을 제안했다.
**Tutor clarification:** 큰 모델이 독립적인 검증자라는 보장은 없다. 직접 도구 인자 검사와 사람의 task rubric을 보완할 수 있다.
**Transcript clarification:** ‘Forfeit’는 ‘Overfit’로 정정됐다. 이후 ‘not overfit’ 발화는 뜻이 불명확하므로 추가 오답으로 기록하지 않는다.

## 9 — 표현 다양성과 상황 다양성

**Learner:** tone 변형은 표현 다양성이고, 조건·스킬 변화는 상황·도구 호출 다양성이라고 구분했다.
**Tutor clarification:** 서로 다른 축이므로 같은 의미의 표현 변화에는 필요한 행동을 유지하고, 조건·정책 변화에는 필요한 행동을 바꾸는 평가를 설계할 수 있다. 이는 적용 제안이며 수행 결과는 없다.

## Discussion boundaries and next step

- Original-only와 synthetic 추가 비교 질문에서 learner가 명확한 설명을 요청했고 질문 표현이 도움이 되지 않는다고 지적했다. 이후 질문은 조건을 구체화하고 한 번에 하나씩 묻는다.
- 이 대화는 chapter-23 개념 확인 기록이며 최종 verdict를 만들지 않았다.
- 실제 세일즈 모델의 혼합 비율, verifier calibration, long-tail slice 정의와 개선 효과는 아직 측정하지 않았다.
- Learner는 chapter 23a 읽기로 이동했다. [[../ch-23a/read]]에서 rarity/knowledge, [[../ch-23b/read]]에서 대화·스킬 범위를 이어서 다룬다.
