# 제47a장 — 벤치마크 과적합과 일반화 감사

훈련에 사용한 benchmark 점수만으로 일반화를 주장하지 않는다. fresh item, 표면을 바꾼 perturbation, 정답을 뒤집은 counterfactual, 시간이 지나 새로 수집한 live item을 별도 평가한다. prompt 형식·matcher·sampling seed를 고정하고, 각 slice의 delta와 confidence interval을 보고한다.
