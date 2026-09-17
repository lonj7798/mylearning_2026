# 제44b장 — 여러 도메인의 RL과 일반 능력

수학처럼 verifier가 있는 도메인과 대화·코드처럼 연속 reward를 쓰는 도메인을 섞을 때 domain별 prompt 수, reward scale, group 크기를 명시한다. 전체 평균만 보면 한 도메인의 퇴행이 가려지므로 도메인별 정확도·길이·entropy와 held-out capability를 비교한다. mixture 비율을 바꾼 ablation으로 transfer와 interference를 구분한다.
