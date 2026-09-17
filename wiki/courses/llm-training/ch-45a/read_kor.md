# 제45a장 — 선호 최적화와 RL 단계 레시피 비교

DPO·PPO·GRPO·RLVR 레시피를 데이터 종류, verifier, KL 위치, update 횟수, 평가 시점으로 나란히 비교한다. 각 단계 뒤에 reward와 held-out 능력을 측정하고, SFT checkpoint를 대조군으로 둔다. 단계가 많아질수록 개선이 누적된다고 가정하지 말고 각 단계의 delta와 실패 slice를 기록한다.
