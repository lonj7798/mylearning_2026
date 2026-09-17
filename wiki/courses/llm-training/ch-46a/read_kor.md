# 제46a장 — 소규모 에이전트 SFT 후 RL 실험

작은 tool 환경에서 SFT checkpoint를 만든 뒤 verifier 기반 RL을 수행한다. 학습 prompt와 겹치지 않는 일반성 평가 세트를 준비하고, reward·성공률·tool 오류·응답 길이·KL을 기록한다. RL 뒤 held-out 능력이 사전 등록한 하한보다 떨어지면 generality gate에서 실패로 처리하고 배포하지 않는다.
