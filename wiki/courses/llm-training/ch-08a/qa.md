<!-- chapter: ch-08a
     companion to: [[read]]
     kind: qa
     track: pretraining
-->

# ch-08a — Reading Q&A

## Q1. §4(estimate 불일치와 published law의 정밀도)가 이해가 안 됨 — 천천히 설명

**Kernel.** 세 조사가 순서대로 §4를 구성. (1) Porian et al.: Kaplan setup을 재현하며 실수를 하나씩 고침(output head FLOPs 누락, 고정 warmup, 미조정 schedule, 크기별 미조정 hyperparameter) — exponent가 0.835→0.706→0.602→0.571→0.497로 이동, 작은 모델을 불리하게 측정한 실험적 실수가 원래 결과의 대부분을 만들었음이 드러남. (2) Besiroglu et al.: Chinchilla 자신의 Fig.4를 digitize해서 재fit — 반올림(β 0.2849→0.28)과 optimizer 버그(Huber loss를 평균으로 계산해 L-BFGS가 너무 일찍 멈춤, 논문의 신뢰구간이 60만 run이 필요할 정도로 비현실적으로 좁았음) 두 원인으로 a=0.5126을 얻어 Approach 1-2와 일치시킴. (3) DeepSeek: 같은 방법, 다른 데이터 품질로 a가 0.450→0.524→0.578까지 이동(품질 순서는 저자들 추측). 결론: tokens-per-parameter 비율은 데이터+counting convention+hyperparameter protocol의 조합에 종속. see [[read]] §4.

## Q2. N과 C가 뭔가?

**Kernel.** N=파라미터 수(단 Kaplan은 embedding 제외, Hoffmann은 포함 — 정의가 다름). C=학습에 쓴 총 FLOPs, `C≈6ND`(D=토큰 수). §1 worked example(n_layer=8, d_model=512 소형 모델)에서 6N₁/M=0.43, 6N₂/M=1.32로 작은 모델일수록 정의 차이가 크고(2.3배 vs 1.3배 반대 방향), 80-layer 대형 모델에서는 8% 이내로 수렴 — 이게 §4 Porian et al.에서 output head FLOPs 누락이 작은 모델 exponent를 왜곡시킨 이유와 직결. see [[read]] §1.

## Q3. 이 N/C/D 공식들이 text 전용인데 multimodal은?

**Kernel.** ch-08a의 모든 source는 text-only(Kaplan, Chinchilla, Porian, Besiroglu, DeepSeek, Llama 3, SmolLM2 등) — multimodal scaling 논문 없음. §8에 딱 한 번 "LLaVA multimodal tuning"이 언급되지만 이건 over-training이 post-training plasticity에 미치는 영향 이야기지 multimodal pretraining scaling과 무관. Multimodal에서 N/D/C를 어떻게 셀지(vision encoder FLOPs, modality별 token 수 등)는 이 장 검증 범위 밖 — COLLECTION-PLAN gap 후보.

## Q4. §5(compute-optimal vs 의도적 과학습)와 "파라미터당 10~10,000토큰"

**Kernel.** Token multiplier M=D/N. Training은 한 번, inference는 서비스 내내 반복되므로, loss가 같다는 제약 하에 더 작은 N + 더 큰 D(over-training)로 총비용(6ND + 2N·D_inf)을 낮출 수 있음. 13B→7B로 17% 절감, 30B→13.6B로 28% 절감 사례. 47개 MPT 모델(150M-6B)을 파라미터당 10~10,000토큰까지 학습시킨 결과 loss도 Gauntlet 평균도 plateau 없이 계속 개선 — 단, 이 극단 데이터로 다시 fit하면 곡선이 평평해져서(α 0.08→0.18) "일반 비율에서 fit한 법칙은 극단 비율의 이득을 과대평가한다"는 자체 경고 포함. Gadre et al. Eq.4로 over-training M배는 모든 budget에서 일정한 compute 배율 비용(worked example: M=320이 M=20보다 항상 8.8% 더 많은 reducible loss, 1.75배 compute 필요). see [[read]] §5.

## Q5. Base model의 downstream 성능이 계속 오른다는 건 cross-entropy loss가 계속 낮아진다는 뜻인가?

**Kernel.** 아니다, 별개의 두 주장. Sardana 실험에서는 우연히 loss와 Gauntlet 평균이 같이 개선됐을 뿐 — 이건 §5의 좁은 사례. §6-§7이 바로 이 conflation을 반박: (1) §6 loss는 many-task 평균은 잘 예측(17-task 오차 0.05%)하지만 개별 task는 못함(HellaSwag 25.73%). (2) §7 Failure 1(Schaeffer et al.): accuracy는 계단함수라 정답 확률이 안 바뀌어도(loss 불변) 오답에 확률이 어떻게 분배되는지에 따라 accuracy가 1↔0으로 뒤집힘(4지선다 worked example). (3) §7 Failure 2: 46개 task 중 18개만 loss와 매끄럽게 scaling. (4) §7 Failure 3: emergence — 일부 능력은 threshold 전까지 random level에 머묾. ch-07의 objective(=upstream loss) vs breadth(=downstream) 구분과 정확히 같은 구조. see [[read]] §6, §7.

## Q6. "Downstream"이 뭔가?

**Kernel.** Pretraining objective(next-token prediction, cross-entropy loss)를 "upstream"이라 부르고, 그렇게 학습된 모델을 평가하는 실제 관심 task(MMLU, GSM8K, HumanEval, ARC-Challenge, HellaSwag 등)를 "downstream"이라 부름 — 직접 최적화 대상이 아닌, 나중에(down-the-line) 측정하는 능력. ch-07의 objective/breadth 구분과 1:1 대응(objective=upstream loss, breadth=downstream task 성능).

## Q7. §8(over-training이 plasticity에 주는 비용)이 논쟁의 여지가 있나?

**Kernel.** ✓ 그렇다, §8 스스로 인정. (1) "Status: Result (single study)"로 명시. (2) 재현 실패가 자체 포함: OLMo-1B는 2.5T 토큰 넘으면 SFT 후 저하, 같은 방법을 OLMo-2-7B에 3T까지 적용해도 재현 안 됨(같은 run 안 checkpoint들의 서로 다른 final LR이라는 confound를 저자들 스스로 지적). (3) §5의 실전 사례(Llama 3.1 8B/70B, SmolLM2, DeepSeek LLM 7B)와 긴장 — 이들은 적극적으로 over-training하는데 그 결과 SFT 후 문제를 겪었는지 데이터가 없음(테스트 안 됨). (4) 이 장의 course Interpretation이 "over-training이 7B+ 규모에서 harmful하다는 증거가 아니다"라고 명시적으로 결론을 유보 — §9 objective 3의 절차적 권고(두 token budget에서 각각 SFT해서 paired 비교)로 이어짐. see [[read]] §8, §9.
