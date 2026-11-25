<!-- qa: rl-pipeline ch-01 reading Q&A. Companion to [[read]]. Append-only across cycles. -->

# ch-01 Q&A — Why RL-from-Base Needs Its Own Infrastructure

## Q1. "staleness"는 한국어로 무엇인가? (2026-11-24)
**Kernel:** 사전적 의미는 "신선하지 않음/묵음"(stale bread). RL에서는 보통 영어 그대로 쓰며, 뜻은 "data를 만든 policy가 얼마나 오래되었는가". Rocket 정의: token 하나마다 (그 token을 **생성한** sampler의 weight version)과 (그 token으로 **학습하는** learner의 version) 사이의 차이, 단위는 learner step. 예: step 100 weight로 생성 → step 105에서 학습 = staleness 5. 문제는 policy gradient가 on-policy를 가정하므로 staleness가 크면 off-policy가 되고, Rocket Fig. 3에서 sampler–learner log-prob 차이가 staleness와 함께 커짐 ([[mai-rl-infrastructure]]). 자세한 분해는 ch-03.

## Q2. "a failed run can simply be restarted"는 어떤 regime에서 무슨 뜻인가? (2026-11-24)
**Kernel:** 두 regime 비교 (read.md L32–42). ① distilled start: RL이 짧은 refinement(수일, 수백 step)라서 실패(infra 장애든 collapse 같은 학습 실패든)하면 run을 버리고 다시 돌리면 됨 — 비용이 작아 복잡한 recovery가 불필요. ② climb from base: 수 주~수 개월이라 restart 비용 = 마지막 checkpoint 이후 시간 + requeue/container 기동 (Table 1: learner 장애 88회/2,490 h ≈ 28 h당 1회), 그리고 [[mai-thinking-1]] Sec. 3.1.4에 따르면 collapse 직전 checkpoint 재개는 "often not viable"(불안정성이 collapse보다 많은 step 앞서 내재). 결론: restart가 fallback이 될 수 없으므로 component 단위 recovery가 필요 → ch-20/21. 열린 질문: 왜 collapse 이전 checkpoint 재개도 실패하는가?

## Q3. Recipe 1단계의 "expected number of RL steps"와 "wall-clock duration"은 어떻게 정하는가? (2026-11-25)
**Kernel:** 이 단계에서는 정확한 값이 아니라 order of magnitude(수일 vs 수 주)만 있으면 된다. 그것만으로 "restart가 fallback이 되는가"를 판정할 수 있다 (read.md L240). ① **Steps:** 근거가 셋이다. (a) data: dataset prompts ÷ unique prompts per batch = 1 epoch의 step 수. Olmo 3 Table 49로 계산하면 7B Think는 104,869/64 ≈ 1,639이고 실제 1,400 step을 돌렸다. 32B는 104,869/128 ≈ 819이고 실제 750이다. 둘 다 1 epoch 미만이며, 이 계산은 teacher의 것이다(paper가 근거를 밝히지 않음). (b) compute budget: ScaleRL 8B는 100K GPU-h ≈ 7,400 steps. (c) stopping rule: held-out curve fit(ScaleRL의 sigmoid는 ~1.5K GPU-h 이후 fit; Rocket Fig. 8은 log-linear fit R² = 0.974). ② **Wall clock** = steps × per-step time + failure overhead. Per-step time은 pilot run으로 측정한다. Olmo 3의 representative step은 1,000 s이고 그중 training은 125 s다. 1,400 × 1,000 s ≈ 16 days이며, 보고된 ≈15 days와 대략 맞는다(teacher 계산). Overhead의 참고치로 Rocket Table 1에서 targeted recovery가 없었다면 2,490 h에 432.5 h(17.37%)가 더해졌다. 열린 질문: async에서 step time이 longest rollout에 묶이지 않는다면, 긴 rollout의 비용은 어디로 옮겨 가는가?
