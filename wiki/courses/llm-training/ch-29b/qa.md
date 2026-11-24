<!-- chapter: ch-29b
     companion to: [[read]]
     kind: qa
     track: synthetic
-->

# ch-29b — Reading Q&A

## Q-재독 1. MSC가 뭔가? (2026-10-14)

**Kernel.** MSC = Multi-Session Chat, "Beyond Goldfish Memory: Long-Term Open-Domain Conversation"(arXiv 2107.07567)의 데이터셋(풀네임은 excerpt 제목; read.md는 약어만 사용). 구성: crowdworker가 PersonaChat persona를 맡아 최대 5 session 진행, session 간 간격 1~7시간 또는 1~7일 무작위, 별도 작업으로 중요한 개인 정보 요약을 쓰고 이후 session은 전체 이력 대신 요약을 읽음, 학습 episode 평균 1,614 token(§3). 장 내 역할: shape A(plan 기반 multi-session) 기준점. 일관성 사람 평가 MSC 3.87 vs Conversation Chronicles 4.71(5점, 0.5K episode, 평가자 3명). BST 2.7B를 MSC로 fine-tune 시 validation ppl session 1만 10.5, session 1–4 8.77(Table 6; perplexity, 과제 성공률 아님). session 1–4 학습/session 5 테스트로 길이 일반화 사례. 4K window 초과는 LoCoMo·BEAM뿐. see [[read]] §2, §Length coverage, [[msc-beyond-goldfish-memory]].

## Q-재독 2. perplexity가 뭔가? (2026-10-14)

**Kernel.** perplexity = exp(평균 NLL) = exp(cross-entropy loss). 낮을수록 정답 텍스트를 잘 예측. Worked example: 정답 token 4개에 모델이 준 확률 1/2, 1/4, 1/2, 1/8 → −log₂ = 1, 2, 1, 3 bit, 평균 1.75 bit, 2^1.75 ≈ 3.36 → "매 token ~3.4개 후보 중 고르는 정도로 헷갈림"(해석); 확률 1이면 1. MSC: BST 2.7B validation ppl session 1만 10.5, session 1–4 8.77(Table 6) — 다음 token 예측이 나아졌다는 뜻일 뿐 과거 session 사실을 기억해 답했는지는 보여주지 않음. NLL은 [[ch-25]] qa 참조. see [[read]] §2.
