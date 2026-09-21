<!-- chapter: ch-07
     companion to: [[read]]
     kind: qa
     track: foundations
-->

# ch-07 — Reading Q&A

## Q1. Non-finite value가 뭔가?

**Kernel.** 부동소수점 표현 범위를 벗어나거나(Inf) 수학적으로 정의되지 않은 연산 결과(NaN, 예: 0/0, Inf−Inf, log of underflowed zero)를 통칭. §1 도입부가 "non-finite value alone은 정보가 안 된다"고 하는 이유: NaN/Inf는 여러 단계 중 어디서 시작됐든 이후 모든 계산에 전파되므로, loss가 NaN이라는 사실만으로는 attention logit/output logit/optimizer state 중 어디가 원인인지 구분이 안 됨 — 그래서 §1이 단계별 진단 순서를 둠. §1.5의 zero-variance advantage(정확히 0, NaN 아님)와 구분해야 함. see [[read]] §1, §1.4, §1.5.

## Q2. Attention-logit growth가 뭔가?

**Kernel.** `z_ij = ⟨q_i,k_j⟩/√d_h`의 최댓값이 학습 중 유계로 안 머물고 계속 커지는 현상. 두 경로로 학습을 망가뜨림: (1) softmax saturation → 해당 row의 gradient가 0으로 소실, (2) fp16 exp overflow → Inf/NaN. `κ=1e4`를 강제하면 zero-layer bigram보다 나쁨(인과관계 직접 확인). **메커니즘은 norm 증가이지 cosine similarity 증가가 아님** (`z = ‖q‖‖k‖cosθ/√d_h` 분해로 확인 — cosine만 규제해도 norm이 자유로우면 z는 여전히 커질 수 있음, discuss에서 검증됨). 처방은 qk-layernorm(q,k의 norm을 직접 누름). 2.4M~1.2B 모델을 LR 3e-4~3e-1에서 안정화. step 2000의 max logit을 model size 이차식으로 fit해 4.8B 발산을 사전 예측 — 적중. see [[read]] §1.1.

## Q3. Attention-logit growth와 entropy collapse(§5.1)는 같은 이슈인가?

**Kernel.** 수학적 골격(softmax가 one-hot에 가까워짐)은 같지만 서로 다른 현상. (1) 대상 분포: attention-logit growth는 **attention weight**(어느 position을 볼지) softmax, entropy collapse는 **다음 토큰 예측**(vocab) softmax. (2) 학습 단계: pretraining forward pass 내 vs RL의 수많은 gradient update에 걸쳐. (3) 실패 성격: numerical failure(발산/NaN, §1) vs capability-level failure(loss는 정상, accuracy는 오르다 정체, 다양성만 좁아짐, §5). (4) 처방: qk-layernorm(norm 억제) vs covariance 기반 update 제한(Clip-Cov/KL-Cov) — 서로 무관. see [[read]] §1.1, §5.1, Core insight.

## Q4. clipped_fraction이 뭔가?

**Kernel.** 전체 step 중 실제로 gradient clipping이 발동한(‖g‖>threshold) step의 비율. §3 plateau 진단표에서 lr/clipped_fraction/active tokens 세 지표 조합으로 flat loss의 네 가지 원인(schedule 소진, threshold 과소설정, masking 버그, update collapse)을 구분. clipped_fraction≈1.0이면 threshold가 자연스러운 gradient scale보다 너무 낮아서 매 step이 같은 크기로 강제로 잘림(gradient가 담은 정보 소실). see [[read]] §3 표.

## Q5. Small-proxy study는 spike를 무시해도 된다는 뜻인가?

**Kernel.** 아니다, 정반대. small-proxy study는 slow divergence만 범위에 두고 fast spike는 명시적으로 scope 밖(footnote 1)이라 증거가 얇다는 정직한 인정. Llama 3의 "few loss spikes, no interventions" 인용은 spike를 무시해도 된다는 증거가 아니라 batch-ramp/LR/warmup 설계 덕에 애초에 spike가 적었다는 단일 run 보고(ablation 아님). skip-step 같은 실무 관행은 "engineering practice without a controlled ablation in these sources"로 명시 — 쓸 때 그렇게 말해야 함. see [[read]] §3.

## Q6. "training objective가 개선되는 동안 breadth를 줄인다"에서 objective와 breadth가 뭔가?

**Kernel.** Objective = 지금 이 단계에서 optimizer가 직접 최적화하는, 코드에 명시된 좁은 수치(pretrain: next-token loss, SFT: target completion loss, RL: reward). Breadth = ch-00부터 쓰인 용어로 "학습이 겨냥하지 않은 task/domain에서의 성능" = 일반 능력. 이 장(§4, §5)이 실측한 세 사례에서 objective↑ breadth↓가 동시에 일어남: forgetting(target task 51.47→68.71인데 MMLU FG 18.37%), alignment tax(reward↑인데 BLEU/F1↓), reward over-optimization(proxy reward는 계속↑인데 gold reward는 피크 후 하락). Loss curve만 보면 objective만 보이므로, breadth는 별도 held-out 측정이 필수. see [[read]] §4, §5, Core insight.

## Discuss — Probe 1 (label shift 버그, §2.1)

**Kernel.** Shift 이후 masking(`labels_shift[:, :prompt_len]=-100`)은 shift된 배열의 앞 prompt_len개(`p1,p2,r0`)를 지움 — **딱 첫 response token(r0)만** 빠지고 r1, r2는 여전히 학습됨(초기 답변에서 "response token 전체가 사라진다"로 과잉 일반화했다가 정정). r0는 흔히 chat template에서 role/format marker 역할이라, 이 토큰을 못 배우면 malformed turn starts(§6과 연결)로 나타남 — "말을 못한다"보다 정확한 표현.

## Discuss — Probe 2 (cosine-only 규제로 attention-logit growth를 막을 수 있는가)

**Kernel.** ✓ 못 막는다. `z_ij = ‖q_i‖‖k_j‖cosθ/√d_h`에서 cosine만 규제해도 norm(‖q‖,‖k‖)이 자유로우면 z는 여전히 무한정 커질 수 있음 — growth의 원인이 애초에 norm이었으므로(Q2), cosine 규제는 원인을 안 건드림. qk-layernorm이 norm을 직접 누르는 이유가 여기서 나옴.

## Discuss — Probe 3 (FG% vs raw point 하락, §4.2)

**Kernel.** `FG_i = (1/|E_i|)Σ_e(1/N)Σ_m(R_e^0−R_e^m)/R_e^0×100%`. 같은 3점 하락도 초기 점수 80이면 FG=3.75%, 초기 점수 40이면 FG=7.5% — 2배 차이. Raw point는 두 checkpoint를 동등하게 취급하지만 FG는 순위를 가름. FG%를 report해야 하는 이유: 시작 capability가 다른 모델/checkpoint 간에도 공정하게 비교 가능(상대적 지표이므로).
