# 제44a장 — RL에서 응답 길이 제어

긴 오답이 계속 생성되면 reward가 정체된 뒤에도 토큰 수가 증가할 수 있다. `mean(length_wrong)-mean(length_right)`, completion length 분포, truncation 비율을 step별로 기록한다. `1/|o|` 정규화와 token-level 평균이 긴 응답에 주는 gradient를 계산하고, Dr.GRPO·DAPO의 length-independent loss와 비교한다. max length를 줄이는 것만으로 문제를 해결했다고 판단하지 않는다.
