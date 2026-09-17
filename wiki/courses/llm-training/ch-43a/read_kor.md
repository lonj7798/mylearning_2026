# 제43a장 — 부정 샘플과 부정 gradient

오답 샘플의 likelihood를 낮추면 확률 질량이 다른 토큰으로 이동한다. 그 이동으로 정답이 자동으로 늘어난다는 보장은 없다. negative advantage, likelihood displacement, squeezing을 분리해 측정하고, pass@1과 pass@k, entropy, 오답 길이를 함께 기록한다. group 내 오답만 있을 때는 학습 신호가 없거나 모두 하향 조정되므로 verifier·baseline·ablation으로 원인을 확인한다.
