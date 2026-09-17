# 제42a장 — 좁은 학습이 만드는 광범위한 행동 변화

좁은 데이터로 SFT나 RL을 해도 모델의 다른 작업 행동이 바뀔 수 있다. emergent misalignment, 아첨(sycophancy), 특성 전이 사례를 읽고, 학습 전후 동일한 held-out 프롬프트에서 거부율·정확도·아첨 비율을 측정한다. 원인 추정을 위해 데이터 범위, 학습 step, seed를 고정하고 행동 변화를 slice별로 비교한다. 보류된 capability와 안전 테스트를 통과하지 못하면 reward가 올라도 배포하지 않는다.
