# 제45b장 — 다중 턴 에이전트 RL

관찰 토큰을 policy loss에 포함할지 mask할지, 최종 보상을 어느 action에 배분할지가 credit assignment를 결정한다. 턴별 action·observation·tool 결과·실패 원인을 저장하고, 전체 trajectory reward와 턴별 ablation을 비교한다. 긴 trajectory에서는 stale rollout과 gradient variance를 함께 기록하며, 성공률만으로 안정성을 판단하지 않는다.
