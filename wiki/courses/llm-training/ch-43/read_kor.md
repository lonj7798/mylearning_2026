# Chapter 43 — RL에서의 엔트로피, 출력 다양성, KL 제어

> **핵심 통찰.** 정책 엔트로피는 RL 동안 떨어진다. 업데이트가 토큰 로그확률과 상관하기 때문이다. 엔트로피의 1차 변화는 토큰 로그확률과 그 logit 변화 사이 음의 공분산이며, 정책 그래디언트 업데이트 아래에서는 그 변화가 어드밴티지에 비례한다([[entropy-mechanism-llm-rl]] §3, 정리 1–2). 같은 상관은 시퀀스 수준에서 다양성 하락으로 나타난다. OLMo 2에서는 post-training 단계마다 서로 다른 생성 수가 줄고([[noveltybench]] 그림 6), 긴 RL 실행의 일부에서는 큰 k의 pass@k가 떨어진다([[prorl]] §4.2). 공개 실행에서 작동한 개입은 모든 토큰을 한꺼번에 건드리지 않고 분포의 선택된 부분, 즉 covariance-outlier 집합, 고엔트로피 토큰 20%, clip 범위 한쪽, 전체 그룹의 어드밴티지에 작용한다([[entropy-mechanism-llm-rl]] 표 2, [[high-entropy-minority-tokens]] 표 2, [[clip-low-clip-high-entropy]] §3.2, [[pass-at-k-training]] 표 1). 참조 정책에 대한 KL penalty는 목적이 다른 별도 제어다. 이것은 정책이 얼마나 멀리 움직이는지 제한하며, 그 양이 RL 데이터 밖 태스크의 망각을 예측한다([[rls-razor]] §3).
>
> **지침.** 실행의 검증 정확도가 plateau인데 엔트로피가 계속 떨어진다면, 업데이트를 받을 토큰을 바꿔라. `ε_high`를 올리거나(clip-higher), 정책 그래디언트를 배치의 고엔트로피 토큰으로 제한하거나, covariance outlier에 벌을 주어라. Qwen2.5-32B에서 KL-Cov는 7-벤치마크 수학 평균을 45.8에서 52.2로, Qwen3-32B에서 forking-token DAPO는 6-벤치마크 평균을 66.59에서 70.69로 올렸다. 반면 같은 논문의 entropy-loss sweep에서 계수 0.0001, 0.001은 엔트로피에 작은 영향만 주고, 0.01은 엔트로피 폭발을 일으켰으며, 0.005는 기준선을 이기지 못했다([[entropy-mechanism-llm-rl]] §4.1, 표 2; [[high-entropy-minority-tokens]] 표 2). 목표가 single-sample 정확도가 아니라 coverage라면 regularizer가 아닌 목적함수를 바꿔라. pass@k training과 unlikeliness reward는 각각의 설정에서 큰 k의 pass@k를 올린다([[pass-at-k-training]] 표 1, [[unlikeliness-reward]] 표 3). RL 데이터 밖 능력이 중요하면 시작 정책에 대한 측정된 KL budget을 유지하고 retention을 직접 평가하라. `β = 0` 레시피는 도메인 내부 metric만 보고되며([[dapo]] §2.3, [[dr-grpo]] §3, [[allenai-olmo3-open-instruct-scripts-recipe]]), 100× sweep한 한 실행에서는 더 큰 KL 계수만으로 능력 퇴행이 없어지지 않았다([[rlhf-instructgpt]] 부록 E.6).

---

## 이 챕터가 범용 모델에 중요한 이유

RL은 파이프라인(사전 학습 → 중간 학습 → SFT → 선호 최적화 → RL → 평가)의 마지막 단계이며, 출력 분포를 의도적으로 좁히는 단계다. 목적함수는 학습 프롬프트에서 점수가 좋은 응답의 확률을 올리는 것이다. 이 좁아짐의 비용 세 가지는 측정할 수 있다.

1. **Coverage.** 큰 k의 pass@k는 정책이 여전히 도달할 수 있는 서로 다른 정답 해법 수의 측정이다. [[prorl]]의 Diminish regime, 즉 “특히 수학 도메인”에서 pass@1은 개선되지만 pass@128은 자주 떨어진다(§4.2, 그림 4). 대칭 clip 아래 GSM8K 학습 동안 pass@8은 떨어지는 반면 mean@8은 오른다([[clip-low-clip-high-entropy]] 그림 6).
2. **개방형 프롬프트에서의 출력 다양성.** OLMo 2 체크포인트에서 `distinct_10`(temperature 1.0에서 샘플 10개 중 서로 다른 동치류 수)은 모든 post-training 단계에서 떨어진다. SFT → DPO → RLVR에 대해 7B는 `7.46 → 5.96 → 5.72`, 32B는 `7.25 → 5.22 → 5.08`이다([[noveltybench]] §4.4, 그림 6).
3. **RL 데이터 밖 능력.** 망각은 *새* 태스크에서 측정한 fine-tuned 정책과 base 정책 사이 KL divergence로 예측된다. 통제된 ParityMNIST/FashionMNIST 설정에서는 `R² = 0.96`, LLM sweep에서는 `R² = 0.71`이다([[rls-razor]] §3). 따라서 KL budget은 안정성 요령일 뿐 아니라 retention의 제어 변수다. ch-38a는 이 법칙이 나온 SFT 대 RL 비교를 다룬다.

엔트로피는 이 세 가지 모두 아래에 있다. 학습 루프가 기록할 수 있는 스텝별 양이며, 그 동역학은 이제 주장된 것이 아니라 유도된다. 이 챕터는 유도, 보고된 설정을 가진 개입, KL estimator와 penalty의 적용 위치, 각 선택이 breadth에 하는 일을 제시한다.

---

## §1 자주 혼동하는 세 양

**정의.**

- **정책 엔트로피** `H(π | s) = −Σ_a π(a|s) log π(a|s)`는 한 위치에서의 양이며, 프롬프트 배치에 대해 샘플된 응답 토큰 전체에서 평균낸다([[entropy-mechanism-llm-rl]] 식 5; [[clip-low-clip-high-entropy]] 식 4). nats로 측정하는 하나의 조건부 분포 성질이다.
- **출력 다양성**은 완전한 샘플 집합의 성질이다. `distinct_k`는 k개 생성 가운데 동치류 수([[noveltybench]] 식 1)이고, pass@k는 k개 샘플 중 적어도 하나가 정답일 확률이다.
- **참조에 대한 KL** `KL(π_θ ‖ π_ref)`(또는 그 역방향)는 학습된 정책과 고정 정책 사이 거리이며, 방문한 상태 전반에서 집계한다.

**반드시 따로 측정해야 하는 이유.** 한 위치에서 5토큰 어휘와 `π = (0.70, 0.20, 0.06, 0.03, 0.01)`을 취하면 `H = 0.892` nats다. 모든 위치에서 같은 엔트로피를 가진 정책도 고엔트로피 선택이 모두 국소적 표현 선택이면 거의 동일한 응답 열 개를 낼 수 있다. 반대로 평균 엔트로피가 더 낮은 정책은 남은 엔트로피가 해법 경로를 고르는 지점에 있으면 더 서로 다른 응답을 낼 수 있다. 이것이 [[high-entropy-minority-tokens]]의 경험적 그림이다. Qwen3-8B chain-of-thought 토큰 10⁶개 초과에서 50.64%는 엔트로피가 `10⁻²` 미만이고, 0.672 nats를 넘는 것은 20%뿐이다. 이 고엔트로피 소수가 추론이 분기하는 위치다(§3).

**로깅의 결과.** 스텝당 scalar “entropy” 하나는 위치에 대한 분포를 압축한 보기다. 평균뿐 아니라 quantile도 기록하라. 예를 들어 [[high-entropy-minority-tokens]] §3이 forking-token threshold로 쓴 80th percentile이다. 고정 프롬프트 세트에서 시퀀스 수준 다양성 metric도 기록한다.

---

## §2 엔트로피가 떨어지는 이유: 공분산 결과

**측정으로 말한 문제.** 엔트로피 항도 참조 KL도 없는, 네 계열 base 모델 11개의 실행에서는 검증 정확도가 오르다 포화하는 동안 엔트로피가 0을 향해 단조 감소한다. 2,400개 그래디언트 스텝 중 첫 200스텝에서 엔트로피 소비의 73%, 성능 이득의 76%가 일어난다([[entropy-mechanism-llm-rl]] §2.3, 그림 2). 이 논문은 검증 정확도 `R`과 엔트로피 `H` 사이에 `R = −a·exp(H) + b`를 적합하고(식 6), 첫 36스텝으로 적합하면 이후 200스텝을 수학에서 RMSE 0.9%, 코드에서 1.2%로 예측한다고 보고한다(§2.4). 저자들은 다른 policy model과 off-policy 데이터가 다른 패턴을 보였으므로 이 예측 가능성이 “명백히 보편적인 것은 아니다”라고 말한다(§2.6). 이 적합은 자기 범위 진술이 붙은 **결과(단일 연구)**로 다뤄야 한다.

**메커니즘, 단계별** ([[entropy-mechanism-llm-rl]] §3).

1. 상태에서의 정책을 logits `z` 위의 softmax로 쓴다.
2. 보조정리 1(logit 변화에 대한 1차): `ΔH(s) ≈ −Cov_{a~π(·|s)}(log π(a|s), Δz_{s,a})`. 기호에서 `Δz`는 상태 `s`에서 행동 `a`의 logit 변화이고, 공분산은 해당 상태의 현재 정책에서 뽑은 행동들에 대해 취한다.
3. 정리 1(vanilla policy gradient): logit 변화는 `Δz_{s,a} = η·π(a|s)·A(s,a)`이므로 `ΔH ≈ −η·Cov(log π(a|s), π(a|s)·A(s,a))`다. `η`는 학습률, `A`는 어드밴티지다.
4. 정리 2(natural policy gradient): `ΔH ≈ −η·Cov(log π(a|s), A(s,a))`.
5. 경험적으로 Qwen2.5-7B에서 `−dH`는 측정한 공분산을 따라가고, 학습 내내 공분산은 양수로 남는다(§3.3, 그림 8).

부호 규칙은 공분산에서 따른다. 업데이트가 로그확률과 *양의* 상관일 때 엔트로피가 떨어진다. 즉 가능성 높은 토큰이 양의 어드밴티지를 받거나, 가능성 낮은 토큰이 음의 어드밴티지를 받을 때다.

**수치 예시(손으로 확인 가능).** 위의 5토큰 분포 `π = (0.70, 0.20, 0.06, 0.03, 0.01)`, `H = 0.8916` nats를 취하자. 어드밴티지 `A = ±1`을 가진 샘플 토큰 하나, 학습률 `η = 0.5`에서 정확한 policy-gradient 방향 `Δz_j = η·A·(1[j = a] − π_j)`으로 logits를 갱신한다.

| 샘플된 토큰 | 어드밴티지 | 새 `π_a` | 정확한 `ΔH` | 1차 `−Cov` 예측 |
|---|---|---|---|---|
| `π_a = 0.70`(평균 이상) | +1 | 0.745 | **−0.0775** | −0.0753 |
| `π_a = 0.70` | −1 | 0.651 | **+0.0721** | +0.0753 |
| `π_a = 0.03`(평균 이하) | +1 | 0.062 | **+0.1591** | +0.1511 |
| `π_a = 0.03` | −1 | 0.014 | **−0.1410** | −0.1511 |

경계선은 정책 자체의 평균 로그확률 `Σ_j π_j log π_j = −H = −0.892`다. `p = 0.70`의 샘플 토큰은 `log π_a = −0.357 > −H`이고, `p = 0.03` 토큰은 `log π_a = −3.507 < −H`다. 네 행은 다음 절의 네 사분면이다. [figures/entropy-dynamics.html](figures/entropy-dynamics.html)에서 분포, 샘플 토큰, 어드밴티지, `η`를 바꾸고 정확한 엔트로피 변화 및 각 대안으로 이동하는 질량을 읽을 수 있다. 이 그림은 §5의 estimator 비교도 재현한다.

---

## §3 어드밴티지 부호별 엔트로피: 네 사분면

배치의 모든 토큰은 네 칸 중 하나에 든다. [[entropy-mechanism-llm-rl]]이 토큰 선택에 쓰는 공분산의 중심화된 형태(식 10)를 쓰면, 토큰 `y_i`의 기여는 `(log π(y_i) − mean_j log π(y_j)) · (A(y_i) − mean_j A(y_j))`이며, 양의 곱은 엔트로피를 아래로 민다.

|  | 평균 이상 확률 | 평균 이하 확률 |
|---|---|---|
| **양의 어드밴티지** | 곱 > 0 → 엔트로피 하락(정책이 이미 선호하는 것에 커밋) | 곱 < 0 → 엔트로피 상승(희귀 선택지를 들어 올림) |
| **음의 어드밴티지** | 곱 < 0 → 엔트로피 상승(질량이 mode를 떠남) | 곱 > 0 → 엔트로피 하락(희귀 선택지를 제거하고 그 질량이 주로 mode로 돌아감) |

실무적으로 중요한 결과가 두 가지 있다.

**(a) `p·A`로 토큰의 순위를 매기면 질량의 절반을 찾지 못한다.** `p > 0`이므로 `p·A`의 가장 큰 값은 모두 양의 어드밴티지 토큰이다. 배치에서 네 토큰 `(π, A) = (0.90, +1.2), (0.50, −0.8), (0.05, +1.0), (0.01, −1.5)`을 두면, 배치 평균은 `mean log π = −2.100`, `mean A = −0.025`이고 중심화된 곱은 `+2.443`, `−1.090`, `−0.918`, `+3.695`다. 가장 큰 엔트로피 감소 기여는 `π = 0.01`인 *음의 어드밴티지* 토큰이다. 그러나 `p·A` 순위에서는 `−0.015`를 받아 절대 선택되지 않는다. 이것이 [[entropy-mechanism-llm-rl]]이 `p·A`가 아니라 중심화된 곱(식 10)을 선택에 쓰는 이유다. 이 선택 규칙은 구성상 부호에 무관하다.

**(b) 두 clip 경계는 다른 사분면에서 작동한다.** clipped surrogate에서 하한 `1 − ε_low`는 음의 어드밴티지 토큰에서 binding하고, 상한 `1 + ε_high`는 양의 어드밴티지 토큰에서 binding한다. [[clip-low-clip-high-entropy]]는(정리 1, 1차, tabular softmax, 무작위 대칭 어드밴티지) 엔트로피 변화가 엔트로피를 올리는 clip-low 항과 내리는 clip-high 항으로 분해됨을 증명한다. 가중치는 `p_k = P(clip-low)`, `q_k = P(clip-high)`다. 부호 주장은 저자들이 보편적이지는 않지만 자신의 측정에서는 성립한다고 말한 두 부등식을 필요로 한다(부등식 7, 그림 1–2). 대칭 기본값 `ε_low = ε_high = 0.2`에서 clip-high 항이 지배하며, 보상이 무작위 노이즈여도 엔트로피는 떨어진다(§2.3, 그림 3, Qwen2.5-1.5B-Instruct). 무작위 보상에서 같은 하락 드리프트가 그림 4의 Qwen, Llama, Olmo base 모델과 그림 8a의 Llama3.2-1B-Instruct에서도 나타난다.

**음의 어드밴티지는 squeezing 뒤의 메커니즘이다.** 같은 5토큰 분포에서 `π = 0.03` 토큰의 logit만 §2의 업데이트가 적용한 0.485만큼 내리자. 재정규화 뒤 분포는 `(0.7082, 0.2023, 0.0607, 0.0187, 0.0101)`이고 엔트로피는 0.8916에서 0.8586 nats로 떨어진다. 아래로 민 토큰에서 제거된 확률 `0.0113`은 나머지 확률에 비례하여 재분배된다. 그중 `0.0082`, 즉 72%가 이미 0.70인 토큰으로, `0.0023`이 다음으로 큰 토큰으로 간다. softmax에는 다른 선택지가 없다. ch-43a는 이를 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`에서 유도하고 offline 대응물을 다룬다.

---

## §4 *어떤* 토큰을 갱신할지 바꾸는 개입

### 4.1 균일한 엔트로피 bonus

형식은 목적함수에 `+ c_H·H(π)`를 더하는 것이다. 이 항은 A3C에서 왔으며, A3C는 이를 Williams & Peng(1991)에게 돌리고 Atari와 TORCS에 `β = 0.01`, MuJoCo 실행에 `10⁻⁴`를 사용한다. 절제는 없다([[entropy-regularization-ppo]] §4, 보충 §8–9). 따라서 이 bonus는 매 스텝 보상에 `α·H(π(·|s_t))`를 더하고 soft-Q critic을 가진 off-policy 방법인 Soft Actor-Critic의 maximum-entropy 목적함수보다 오래되었다([[maximum-entropy-rl]] 식 1, §4.2). 두 설정은 항은 공유하지만 학습 루프는 공유하지 않으며, SAC 자체의 민감도 결과는 토큰 수준 엔트로피 목표가 아니라 보상 스케일에 관한 것이다(§5.2, 그림 3b). 대규모 continuous-control sweep에서는 entropy penalty 및 entropy constraint를 포함한 시험한 어떤 regularizer도 HalfCheetah 외에서는 유의하게 도움이 되지 않았다([[entropy-collapse-ppo]] §3.8, 부록 K.1).

LLM 규모에서 보고된 결과는 다음과 같다. 계수 0.0001과 0.001은 작은 영향, 0.01은 엔트로피 폭발, 0.005는 다른 기준선을 이기지 못한 채 엔트로피 안정화를 보였다([[entropy-mechanism-llm-rl]] §4.1, 그림 9). Pass@k-training 비교에서 0.005는 “모델 붕괴를 초래할 수” 있고, 시험한 더 작은 값도 대안을 이기지 못했다([[pass-at-k-training]] §3.1). DeepSWE는 엔트로피 손실이 “지수적으로 증가하는 엔트로피와 붕괴로 이어졌다”고 하여 제거했고, base-model 토큰 엔트로피가 0.3–1 안이면 불필요하다고 말한다([[deepswe]] §2.3). [[high-entropy-minority-tokens]]의 해석(§6, Discussion 3, **해석**, 통제 비교 없음)은 균일 bonus가 추론이 분기하는 고엔트로피 소수가 아니라 진행 중인 언어 또는 수학 구조를 완성하는 위치인 저엔트로피 다수의 엔트로피를 올린다는 것이다.

**권고.** 엔트로피가 하락하고 정확도 곡선이 평평해졌다면 `c_H`부터 시작하지 마라. 아래 네 선택지는 같은 설정에서 보고된 이득이 있다. bonus를 쓴다면 §4.1 sweep이 엔트로피 폭발 또는 개선되지 않은 기준선을 관측한 구간인 0.005–0.01로 취급하라([[entropy-mechanism-llm-rl]] §4.1, 그림 9; 그 그림의 모델은 인쇄되지 않았다).

### 4.2 공분산 outlier 개입: Clip-Cov와 KL-Cov

둘 다 식 10의 중심화 공분산으로 선택한 배치 토큰의 작은 일부에 작동한다.

- **Clip-Cov**(식 11–12): 공분산이 `[ω_low, ω_high]`에 있는 토큰 중에서 `⌊r·N⌋`개 토큰 인덱스를 균일하게 샘플하고 policy gradient를 detach한다. 보고 값은 `r = 2×10⁻⁴`, `ω_low = 1`, `ω_high = 5`이며 두 경계는 평균 공분산의 500배를 넘는다(§4.3). 선택의 스케일은 Qwen2.5-7B 스텝 1에서 상위 0.02% 토큰 평균 공분산 5.654 대 전체 토큰 0.003이다(표 1).
- **KL-Cov**(식 13–14): `rank(Cov) ≤ k·N`인 토큰에 추가 손실 항 `−β·D_KL(π_θ_old ‖ π_θ)`를 준다. Listing 1 구현에서는 `|log π − log π_old|`다. 보고 값은 `k = 2×10⁻³`(7B), `2×10⁻⁴`(32B), `β = 1`이다. 이 KL은 고정 참조가 아니라 같은 스텝의 *이전* 정책에 대한 것임에 유의하라.

식 10–12에서 쓴 선택 규칙의 의사 코드다. 공개 구현은 [[entropy-logging-patterns]]에 따르면 저자 저장소에서 적응한 verl의 `clip_cov`, `kl_cov` loss mode에 있다.

```python
# N개 rollout 토큰의 배치마다
cov = (logp - logp.mean()) * (adv - adv.mean())          # 식 10, 두 항 모두 중심화
band = (cov > omega_low) & (cov < omega_high)            # omega_low = 1, omega_high = 5
idx  = band.nonzero().flatten()
sel  = idx[torch.randperm(idx.numel())[: int(r * cov.numel())]]   # r = 2e-4, band 안에서 균일
mask = torch.zeros_like(cov, dtype=torch.bool); mask[sel] = True
pg   = -(adv * logp)
pg   = pg.masked_fill(mask, 0.0)                         # Clip-Cov: 선택 토큰 detach
```

7개 수학 벤치마크 평균 결과([[entropy-mechanism-llm-rl]] 표 2; DAPO-MATH 프롬프트, `256` 프롬프트 × `8` 응답, temperature 1, rollout 스텝당 업데이트 8회, 생성 한도 8,192)는 다음과 같다. Qwen2.5-7B: GRPO 38.6, clip-higher 38.8, Clip-Cov 40.4, KL-Cov 40.6. Qwen2.5-32B: GRPO 45.8, clip-higher 47.2, Clip-Cov 50.3, KL-Cov 52.2. 기준선의 엔트로피 plateau에서 KL-Cov 실행은 엔트로피를 10배 넘게 더 높게 유지한다(§4.3). **조건과 한계:** 수학만, 셀당 실행 하나, pass@k 없음이며, 저자들은 제어한 엔트로피 수준과 최종 정확도 사이 관계를 보고하지 않고 최적 엔트로피를 열린 질문으로 부른다(§4.5).

### 4.3 Clip-higher와 clip-lower

[[dapo]]는 `ε_low = 0.2`와 함께 상한을 `ε_high = 0.28`로 올린다. `ε = 0.2`에서 `π_old = 0.01` 토큰은 최대 0.012까지 오를 수 있지만 `π_old = 0.9` 토큰은 binding하지 않는 1.08 경계를 갖기 때문이다. up-clipped 토큰의 측정 평균 확률은 0.2 아래다(§3.1, 그림 3a). DAPO의 progressive stack에 clip-higher를 더하면 Qwen2.5-32B의 AIME24 avg@32가 36에서 38로 이동했다(표 1, 행당 실행 하나). DAPO는 `ε_low`를 0.2로 남긴다. “이를 높이면 이 토큰들의 확률을 0으로 억제하여 sampling space 붕괴를 초래하기” 때문이다(§3.1).

[[clip-low-clip-high-entropy]]는 같은 레버의 반대편을 취한다. `ε_low`를 *낮추면* clip-low 사건이 더 자주 일어나고, 이는 엔트로피를 올린다. clip-high를 비활성화한(`ε_high = ∞`) 상태에서 `ε_low = 0.15`는 GSM8K의 Qwen2.5-3B-Instruct에서 붕괴와 엔트로피 폭발을 모두 피한 값이다(§3.2, 그림 5). 엔트로피 제어 실행은 유사한 mean@8에서 pass@8을 보존했고, 대칭 `0.2 / 0.2` 실행은 학습 전반에서 pass@8을 잃었다(그림 6–7). 논문은 표가 아니라 곡선을 보고한다.

### 4.4 업데이트를 고엔트로피 토큰으로 제한하기

[[high-entropy-minority-tokens]]는 DAPO의 토큰 수준 손실에 `1[H_t^i ≥ τ_ρ^B]`를 곱한다. `τ_ρ^B`는 배치에서 토큰 상위 `ρ` 비율을 선택하는 엔트로피 threshold이며, 선택 토큰 수로 정규화한다(식 6). `ρ = 20%`, KL 손실 없음, 엔트로피 손실 없음에서 6-벤치마크 평균은 Qwen3-32B base에서 66.59 → 70.69(AIME'24 `55.83 → 63.54`, AIME'25 `45.63 → 56.67`), Qwen3-14B에서 61.40 → 64.39, Qwen3-8B에서 53.71 → 54.23이다(표 2, temperature 1.0에서 Acc@16). 대신 하위 80% 저엔트로피 토큰으로 학습하면 상당한 하락이 나온다(§5.3). 저자들은 이득이 모델 크기와 함께 커지는 것을 capacity 탓으로 돌린다(**해석**, §5.3). 도메인 밖 LiveCodeBench에서 top-10% 및 top-20% 실행은 all-token 실행 위에 남는다(§5.4, 그림 9, 표 값 없음).

TRL은 이를 기본값 1.0(마스킹 없음)인 `top_entropy_quantile`로 노출하며, 도움말은 이 논문을 인용해 0.2를 권한다([[entropy-logging-patterns]]).

### 4.5 regularizer 대신 목적함수 바꾸기

**Pass@k training** ([[pass-at-k-training]])은 k개 샘플의 *그룹*에 구성원 중 하나라도 정답이면 1을 주고, 어드밴티지를 해석적으로 유도한다. `N`개 롤아웃 중 `N_neg`가 오답이면 `R̄ = 1 − C(N_neg, k)/C(N, k)`, `σ = sqrt(R̄(1 − R̄))`, `Â_pos = (1 − R̄)/σ`, `Â_neg = (1 − R̄ − C(N_neg−1, k−1)/C(N−1, k−1))/σ`다(식 11–15).

수치 예시: `N = 8`, `k = 4`, 이진 보상.

| 그룹 구성 | Pass@1(GRPO) `A_pos / A_neg` | Pass@k training `Â_pos / Â_neg` |
|---|---|---|
| 정답 7, 오답 1(쉬운 프롬프트) | +0.378 / **−2.646** | 0 / 0(4개 그룹마다 통과: `R̄ = 1`) |
| 정답 4, 오답 4 | +1.000 / −1.000 | +0.120 / −0.120 |
| 정답 1, 오답 7(어려운 프롬프트) | +2.646 / −0.378 | +1.000 / −0.143 |

다양성에서 중요한 행은 첫 번째다. pass@1 아래 이미 풀린 프롬프트의 유일하게 다른 샘플은 배치에서 가장 큰 음의 어드밴티지를 받는다. 이것은 엔트로피를 제거하는 “평균 이하 확률 × 음의 어드밴티지” 사분면이다. Pass@k training은 그 프롬프트에 그래디언트를 전혀 주지 않는다. 보고 효과는 엔트로피가 더 높은 수준에 남고 약 200스텝부터 상승한다는 것이다(§3.2, 그림 7b, 수치 없음). Qwen2.5-7B-Instruct에서 Pass@1/Pass@k 점수는 pass@1 training 대비 도메인 내부 Enigmata `12.9/21.3 → 17.9/29.8`, 도메인 밖 KORBench `37.7/45.6 → 47.7/63.5`, AIME 2025 `5.4/19.1 → 7.1/22.4`로 오른다(표 1). k가 커지면 모든 어드밴티지가 작아지므로, 저자들은 이를 보상하기 위해 학습률을 올린다(§3.4).

**Unlikeliness reward** ([[unlikeliness-reward]])는 형식 정리 증명에서 같은 사분면을 양의 쪽에서 공격한다. uplift rate, 즉 특정 확률 순위의 정답 샘플 중 학습 중 확률이 오르는 비율은 샘플의 초기 확률과 함께 증가한다. 저자들은 저확률 정답 샘플이 “거의 uplift되지 않는다”(“rank bias”, §3.5, 그림 4)고 보고한다. 따라서 정답 샘플 보상은 순위 0이 가장 가능성 높은 샘플이고 `β_rank = 0.25`일 때 `1 − β_rank·(G − rank)/G`로 스케일한다. `G = 8`, 정답 샘플의 순위 0, 3, 6에서 그룹 어드밴티지는 세 정답 모두 `+1.291`에서 `+1.055`, `+1.283`, `+1.511`로 이동한다. 이제 가장 희귀한 정답 증명이 가장 큰 push를 받는다. 보고 효과는 큰 N에서 pass@N 상승과 pass@1 및 pass@2의 작은 손실(그림 5), 다른 변형의 단조 하락과 달리 스텝당 unique proof가 줄었다가 회복(§5.3)이다. MiniF2F-test에서 최종 모델은 DeepSeek-Prover-V1.5-RL의 pass@32 `49.2 ± 0.6`, pass@128 `51.2 ± 0.3`에 대해 pass@32 `48.8 ± 0.7`, pass@128 `50.6 ± 0.5`에 도달한다(표 3). 이 실행의 KL 계수도 0.02에서 0.10으로 올렸으며, 저자들은 이것이 pass@N 악화를 막는 데 도움은 되었으나 그 자체로 충분하지는 않았다고 말한다(§5, 부록 D). 따라서 rank 항과 KL 변경은 완전히 분리되지 않는다.

---

## §5 KL estimator: k1, k2, k3

**문제.** penalty에는 학습할 정책 `q`와 참조 `p`에 대해 `KL(q ‖ p)`가 필요하지만, 이용 가능한 것은 샘플된 토큰과 그 로그확률뿐이다. `r = p(x)/q(x)`, 샘플 `x ~ q`이면([[john-schulman-kl-tricks]):

| Estimator | 식 | Bias | 부호 | 비고 |
|---|---|---|---|---|
| `k1` | `−log r` | 없음 | 임의 | 약 절반의 샘플에서 음수 |
| `k2` | `½(log r)²` | 편향됨 | ≥ 0 | 기대값은 `f''(1) = 1`인 f-divergence이므로 KL과 2차까지 일치 |
| `k3` | `(r − 1) − log r` | 없음 | ≥ 0 | `λ = 1`인 `−log r + λ(r−1)`; control variate `r − 1`은 `q` 아래 평균 0 |

종이 위에서 할 점검 두 가지가 있다. k3의 불편성: `E_q[r − 1 − log r] = E_q[r] − 1 + KL(q‖p) = 0 + KL(q‖p)`이며, `E_q[r] = ∫ q·(p/q) = 1`이기 때문이다. 비음수성: `f(r) = r − 1 − log r`는 `f(1) = 0`, `f'(r) = 1 − 1/r`, `f''(r) = 1/r² > 0`이므로 convex이고 `r = 1`에서 최솟값 0을 갖는다.

**블로그 자체 실험의 수치**(`q`에서 샘플; 코드는 `p = N(0,1)`, `q = N(0.1,1)`로 설정한 뒤 `q = N(1,1)`로 설정한다. 주변 산문은 두 분포를 반대 순서로 이름 붙이며, 아래 숫자는 코드가 만드는 값이다):

| 실제 KL | `k1` bias / stdev(상대) | `k2` | `k3` |
|---|---|---|---|
| 0.005 | 0 / 20 | 0.002 / 1.42 | 0 / 1.42 |
| 0.5 | 0 / 2 | 0.25 / 1.73 | 0 / 1.70 |

[figures/entropy-dynamics.html](figures/entropy-dynamics.html)의 패널 2는 브라우저에서 이 실험을 다시 실행하므로, 위 표에 대해 bias-variance trade-off를 점검할 수 있다.

**블로그가 주장하지 않는 것.** RLHF에서 어떤 estimator를 쓸지, 프레임워크 기본값, 또는 이 식들을 손실로 쓸 때 그래디언트에 관해서는 아무 말도 하지 않는다. 마지막 항목은 다음 절의 주제이며 답을 바꾼다.

---

## §6 KL이 들어가는 곳: 보상 흐름, 손실 항, 그리고 각 그래디언트가 계산하는 것

**두 배치 위치.**

- *보상 안의 KL* ([[rlhf-instructgpt]] 식 2; 계보와 closed-form optimum `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)`는 [[kl-control-rlhf]]와 Korbak et al. 식 5): 토큰별 보상은 `r̂_t = r_t − β·(log π_φ(y_t|y_<t,x) − log π_SFT(y_t|y_<t,x))`가 되고, PPO의 어드밴티지 estimator는 이 shaping된 보상을 본다. InstructGPT는 모든 RL 모델에서 `β = 0.02`를 썼으며(부록 C.4), 인간 Likert 점수에서 최적값을 “약 0.01과 0.02”라고 보고한다(부록 E.7).
- *손실 안의 KL* ([[grpo]] §4.1.1, 식 4): 토큰별 손실은 `D̂`가 k3 식인 `+ β·D̂_KL`을 받는다. DeepSeekMath-RL 7B는 `β = 0.04`를 사용했다(§4.2). 논문이 말한 이 배치의 이유는 “Â 계산을 복잡하게 하지 않기 위해서”다.

**그래디언트는 같지 않으며, 모두가 reverse KL을 추정하지는 않는다.** `q_θ`를 정책, `p`를 참조로 두고, 보통 구현처럼 샘플을 고정해 `a ~ q_θ`를 샘플링하자.

- 손실로서의 `k1`: `∇_θ (log q_θ(a) − log p(a)) = ∇_θ log q_θ(a)`, `E_q[∇ log q_θ] = 0`이다. **k1 손실 항은 기대 그래디언트가 0이다.** 올바른 값을 로그하지만 학습하는 것은 없다.
- 손실로서의 `k2`: `∇_θ ½(log q_θ(a) − log p(a))² = (log(q_θ(a)/p(a)))·∇_θ log q_θ(a)`이며, 기대값은 `∇_θ KL(q_θ ‖ p)`다. **k2는 reverse-KL 그래디언트를 준다.**
- 손실로서의 `k3`: `∇_θ (p/q_θ − 1 + log q_θ − log p) = (1 − p(a)/q_θ(a))·∇_θ log q_θ(a)`이며, 기대값은 `−E_p[∇_θ log q_θ] = ∇_θ KL(p ‖ q_θ)`다. **k3는 forward-KL 그래디언트**를 주며, 이는 mode-seeking이 아니라 mass-covering이다.
- *보상* 안의 `k1`: `E_q[−β·log(q_θ/p)]`의 score-function 그래디언트는 `−β·E_q[log(q_θ/p)·∇ log q_θ]`이고, 이는 `−β·∇ KL(q_θ ‖ p)`다. **보상 안의 k1은 reverse-KL 그래디언트를 준다.** policy-gradient estimator가 빠져 있던 항을 공급하기 때문이다.

verl 코드는 같은 결론을 말한다. k1과 k3는 KL의 기대값은 가지지만 기대 그래디언트는 없고, k2는 올바른 그래디언트를 주며, `k3+` 같은 `+` 접미사는 backward pass에서 k2를 쓴다. OpenRLHF는 손실에는 k2 또는 k3, 보상에는 k1을 권한다([[entropy-logging-patterns]]). 실무적으로 dashboard의 숫자와 실제 최적화하는 양은 다를 수 있으므로, 손실에 든 estimator와 로그에 든 estimator를 모두 명시해야 한다.

**프레임워크마다 기본값이 다르다** ([[entropy-logging-patterns]], 2026년 4월 고정 commit의 코드). verl은 두 위치 모두 참조 KL이 기본 off이며(켜면 `kl_coef` 0.001, 보상 측 estimator k1, 손실 측 k3), OpenRLHF는 `init_coef` 0.01로 보상에 k1을 둔다. TRL PPO는 `kl_coef` 0.05로 보상에 k1 또는 k3을 두며, TRL GRPO는 β = 0.0으로 손실에 k3을 둔다. β = 0일 때 참조 모델조차 load하지 않는다. 이 네 사실을 확인하지 않고 서로 다른 stack의 두 실행에서 “KL curve”를 비교하면 서로 다른 양을 비교하는 것이다.

**Adaptive controller는 측정 KL이 목표를 넘으면 β를 올린다.** verl 규칙은 `β ← β·(1 + clip(KL/target − 1, −0.2, 0.2)·n_steps/horizon)`이다. `β = 0.001`, `target_kl = 0.1`, `horizon = 10000`, 한 스텝, 측정 KL 0.3이면 비율은 `0.3/0.1 − 1 = 2`이고 `0.2`로 clip된다. 따라서 `β ← 0.001·(1 + 0.2·1/10000) = 1.00002×10⁻³`다. 방향은 위쪽이고 스텝당 변화는 설계상 작다. 참조에서 멀어지는 실행은 한 번의 스텝 안이 아니라 수천 스텝에 걸쳐 보정된다.

---

## §7 실무의 KL budget: β = 0 레시피, 참조 reset, retention

**β = 0의 입장.** [[dapo]] §2.3은 긴 chain-of-thought 모델이 초기 분포에서 멀리 이동할 것으로 예상되므로 이 제한은 “필요하지 않다”는 논리로 KL 항을 제거한다. 항의 유무 절제는 보고하지 않으며 평가는 AIME뿐이다. [[dr-grpo]]는 전반에 걸쳐 `β = 0`으로 둔다(§3, 부록 G 표 6). Olmo 3 공개 RL 스크립트는 7B와 32B에 `β = 0.0`을 둔다([[allenai-olmo3-open-instruct-scripts-recipe]]). DeepSWE는 `use_kl_loss=False`로 설정하고 in-reward KL 호출을 주석 처리해 둔다([[deepswe-recipe]]). [[simpletir]]은 `β = 0`, 엔트로피 계수 0을 쓴다. 이것이 2025–2026 추론 레시피의 공통 설정이며, 그 모두는 도메인 내부 벤치마크에서 결과를 보고한다.

**β > 0의 입장.** [[prorl]]은 손실에 KL 항을 유지하고(식 4), 검증이 stagnate하면 참조 정책과 optimizer를 hard-reset한다. KL 항이 “점점 더 손실을 지배할 수 있기” 때문이다(§2.3.1, §3.3). 저자들은 KL penalty가 더 높은 rollout temperature나 엔트로피를 위한 clip-higher보다 “더 강하고 더 안정적인 해법”을 주었다고 말한다(§2.2.1, §2.3.1). 수치 비교도 reset 절제도 없다. Tülu 3의 최종 RLVR 실행은 PPO 목적함수의 KL penalty에 `β = 0.05`(8B), `β = 0.07`(70B)를 썼으며([[tulu-3]], arXiv:2411.15124v5 표 21 캡션), §6.2.1은 “더 많은 KL budget을 쓰는 것이 검증 가능 보상의 개선으로 반드시 이어지지는 않는다”고 보고한다.

**KL penalty가 정확도를 희생할 수 있다는 근거.** [[entropy-mechanism-llm-rl]] §4.1(그림 10)에서 0.001~0.1의 참조 KL 계수는 엔트로피를 안정화했지만 그 설정에서는 정확도를 낮췄다.

**KL penalty만으로 retention을 살 수 없다는 근거.** InstructGPT는 사전 학습 항을 끈 채 KL 보상 계수를 기본값의 100배인 2.0까지 sweep했지만, 공개 NLP 데이터셋의 퇴행은 남았고 큰 계수는 검증 보상을 낮췄다. 이를 제거한 것은 penalty가 아니라 사전 학습 데이터 혼합이었다([[rlhf-instructgpt]] 부록 E.6). [[rls-razor]]는 보완적인 양성 결과를 준다. SFT와 RL 실행 전반에서 held-out 이전 태스크의 망각은 *새* 태스크에서 측정한 `E_{x~τ}[KL(π_0 ‖ π)]`로 예측된다. 통제된 ParityMNIST/FashionMNIST 설정에서 `R² = 0.96`, 수학·화학·도구 사용의 Qwen2.5-3B-Instruct LLM sweep에서 `R² = 0.71`이다. 두 결과는 일관된다. 망각을 예측하는 것은 실제로 이동한 거리이며, penalty 계수는 이를 제어하는 간접적이고 보정이 약한 방법일 뿐이다.

**이것으로 할 일.** 고정 held-out 프롬프트 세트에서 시작 정책에 대한 실현 KL을 기록하고, β 값이 budget을 뜻한다고 가정하는 대신 그 측정 KL에 대해 다중 도메인 held-out suite의 retention을 그려라. β = 0 레시피가 β > 0 레시피와 같은 측정 KL에서 같은 retention을 잃는지는 **열린 질문**이다. 이 라이브러리의 어느 출처도 그 비교를 실행하지 않는다.

---

## §8 엔트로피, 저확률 토큰, 도구 호출 경계

에이전트 RL은 §2의 가정을 깬다. 컨텍스트 일부가 정책이 아니라 환경에 의해 쓰인다. [[simpletir]]은 Qwen2.5-7B base 모델의 다중 턴 도구 통합 추론에서 그 결과를 측정한다.

1. 인터프리터 출력은 다음 턴 프롬프트에 연결되며 사전 학습 분포에서 멀 수 있다. 그 뒤 모델 자신의 continuation에 모델이 낮은 확률을 부여하는 토큰이 들어간다. trace한 롤아웃에서 턴 1의 확률은 높고, 턴 2·3의 모델 자체 텍스트에 저확률 segment가 나타나며, 턴 4는 붕괴한다(§3.1, 그림 3). 도구 토큰의 손실을 마스킹해도 이를 멈추지 못한다. 드리프트가 모델 자신의 continuation 안에 있기 때문이다.
2. 명제 3.1은 logits에 대한 토큰별 그래디언트 노름을 `(m_{i,t}/Σ_j m_{i,j})·ρ_{i,t}(θ)·g_{i,t}·|Â_i|·sqrt(1 − 2P(c) + Σ_j P(j)²)`로 준다. `m`은 피드백 마스크, `ρ`는 중요도 비율, `g`는 unclipped gate, `Â`는 궤적 어드밴티지, `P`는 그 위치의 정책 분포다. §2의 5토큰 분포(`Σ_j P(j)² = 0.5346`) 수치 예에서 `P(c) = 0.01` 토큰의 제곱근 항은 `sqrt(1 − 0.02 + 0.5346) = 1.231`, `P(c) = 0.70` 토큰은 `sqrt(1 − 1.40 + 0.5346) = 0.367`이다. 희귀 토큰은 중요도 비율 전에도 mode의 토큰별 그래디언트 노름에 약 3.4배 기여한다. 음의 어드밴티지 궤적에서는 그 비율이 위로 무한이고 `π_old`가 작았던 정확한 지점에서 가장 크다.
3. 사용한 수정은 trajectory filtering이다. 완전한 코드 블록도 최종 답도 내지 않는 턴은 “void turn”이며, 이를 하나라도 가진 궤적은 배치에서 제거한다. 최저 토큰 확률 또는 최고 중요도 비율로 필터링해도 학습은 안정화하지 못했다. 1,000스텝 내 최고 점수는 SimpleTIR AIME24 50.5 / Math500 88.4, 순진한 다중 턴 20.8 / 73.1, 저확률 필터링 23.3 / 72.8, 고비율 필터링 26.3 / 75.0, 필터링 없이 void turn에서 생성 중단 26.1 / 77.3이다(표 2). 설정은 rollout temperature 1.0, clip `0.2 / 0.28`, 엔트로피 계수 0, `β = 0`, 프롬프트당 롤아웃 16개, PPO epoch 4다(부록 표 6).

에이전트 실행의 관련 선택 두 가지가 있다. DeepSWE는 KL 손실과 엔트로피 손실 없이 학습하고 temperature 1.0 롤아웃에 의존한다([[deepswe-recipe]]). Kimi K2는 “학습 전반에서 탐색에서 활용으로 이동하기 위한 temperature decay schedule”을 쓰며, 보고서에는 값이 인쇄되지 않았다([[kimi-k2-recipe]], §3.2.3). 도구 피드백 직후 정책 엔트로피가 *spike*한다는 주장은 이 라이브러리의 검증된 출처가 측정하지 않았다. SimpleTIR은 엔트로피가 아니라 토큰 확률과 그래디언트 노름을 측정한다. 그러므로 이를 **열린 질문**으로 취급하고 자기 실행에서 측정하라. 측정할 때 엔트로피는 정책이 생성한 위치에서만 계산하라. 마스킹한 관측 토큰을 평균에 포함하면 환경 텍스트가 정책 통계에 섞인다. ch-45b는 관측 마스킹과 턴 수준 신용 할당을 다룬다.

---

## 음성 샘플과 음성 피드백

**어느 종류의 음성인가.** 이 챕터는 음의 어드밴티지로 업데이트가 우도를 낮추는 토큰과 시퀀스인 유형 (4), *그래디언트로서의 음성*을 다룬다. 또한 배치에서 완전히 제거하는 샘플인 유형 (1), *음의 한계 가치*도 건드린다([[simpletir]]의 void-turn filtering, [[prorl]] §2.3 및 ch-16의 분산 0 그룹 필터링). 내용 또는 conditioning으로서의 음성인 유형 (2), (3)은 ch-31a에 속한다. 전체 유도는 ch-43a에 있다.

**이 단계에서 음성이 오는 곳.** verifier 또는 judge가 rollout을 라벨링하고, group baseline이 평균 아래 보상을 음의 어드밴티지로 바꾼다. 이진 verifier와 `G`개 그룹에서 전부 정답도 전부 오답도 아닌 모든 그룹은 두 부호를 모두 만든다.

**메커니즘.** softmax에서 `∂ log p_y / ∂ z_j = 1[j = y] − p_j`이므로, 샘플 토큰 로그확률을 내리면 현재 확률에 비례해 다른 모든 토큰에 질량을 더한다. 엔트로피 결과는 §3 표의 두 번째 행이다. 평균 이상 확률 토큰을 내리면 엔트로피가 오르고, 평균 이하 확률 토큰을 내리면 해방 질량 대부분이 mode로 돌아가므로 엔트로피가 떨어진다. §3 수치 예에서는 해방한 `0.0113` 단위 중 `0.0082`가 `π = 0.70` 토큰으로 돌아간다.

**수치가 있는 근거.**

- 하위 clip을 완전히 제거(`ε_low = 1.0`)하면 엔트로피가 낮아졌고, 이를 강화(`ε_low = 0.15`)하면 동일 mean@8에서 pass@8을 보존할 만큼 엔트로피가 올랐다([[clip-low-clip-high-entropy]] §3.2, 그림 5–6). 하위 clip은 음의 어드밴티지 토큰의 업데이트당 억제를 제한하는 경계다.
- DAPO는 `ε_low`를 0.2로 유지한다. 이를 올리면 저확률 토큰을 0으로 억제하여 sampling space를 붕괴시킨다는 것이 명시한 이유다([[dapo]] §3.1).
- dual clip은 verl 기본 `c = 3.0`에서 음의 어드밴티지 토큰의 손실을 `c·|A|`로 제한한다. 이는 off-policy 데이터의 음의 어드밴티지에서 비율이 위로 무한하기 때문에 도입한 Honor of Kings dual-clip PPO 논문에서 왔다([[verl-ppo-loss]]).
- Pass@k training은 이미 풀린 프롬프트에서 유일하게 다른 샘플이 받는 큰 음의 어드밴티지를 제거한다(§4.5 수치 예: `−2.646 → 0`).
- 다중 턴 도구 사용에서는 음의 어드밴티지 궤적이 위로 무한한 중요도 비율과 저확률 인자가 곱해지는 곳이다([[simpletir]] 명제 3.1).

**이 단계에서 음성을 더 안전하게 하는 제어:** 데이터를 on-policy로 유지; lower clip과 dual clip으로 업데이트 제한; 어떤 토큰을 갱신할지 선택(고엔트로피 마스킹, covariance outlier); 쉬운 프롬프트가 큰 penalty를 더 이상 만들지 않도록 group objective 변경(pass@k training); 실패가 substantive가 아니라 structural인 궤적 제거(void-turn filtering).

**진단.** advantage 부호별로 나눈 엔트로피, 그래디언트 노름, clip fraction을 기록하라. verl은 dual clip 아래 음의 어드밴티지 토큰에 `actor/pg_clipfrac_lower`를 기록한다. TRL GRPO는 `clip_ratio/low_mean`과 `clip_ratio/high_mean`을 따로 기록한다([[entropy-logging-patterns]]). 큰 k의 pass@k와 고정 개방형 프롬프트 세트의 `distinct_k` 측정도 더하라.

**효과 크기에 대한 정직함.** 이 챕터가 인용한 어느 출처도 정확도 이득 중 음의 어드밴티지 토큰에서 온 비율을 측정하지 않았다. 측정한 분해는 ch-43a에 있다. 여기 출처가 뒷받침하는 주장은 더 좁다. 음성의 *엔트로피* 및 *coverage* 효과는 아래로 민 토큰이 어느 사분면에 있는지에 따라 다르며, 음성 쪽을 제한하거나 재가중하면 [[clip-low-clip-high-entropy]]와 [[pass-at-k-training]]에서 pass@k가 바뀌었다.

---

## 레시피

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| InstructGPT (PPO-ptx) | 1.3B-175B | RL | KL 배치; β | 보상의 토큰별 KL(식 2); 0.02 | arXiv:2203.02155v1 부록 C.4 | 2026-09-15 검증 | 부록 E.7: Likert 점수는 0과 2에서 낮고, 최적은 “약 0.01과 0.02” |
| InstructGPT (PPO, γ = 0) | 1.3B | RL | retention용 KL sweep | 최대 2.0 | arXiv:2203.02155v1 부록 E.6 | 2026-09-15 검증 | 공개 NLP 데이터셋 퇴행은 100× β에서도 지속; 대신 pretraining mix가 고침 |
| DeepSeekMath-RL | 7B | RL | KL 배치; β; estimator | 손실의 KL; 0.04; 식 4(k3 형식) | [[grpo]] §4.1.1, §4.2 | 2026-09-14 검증 | 절제 보고 없음 |
| Tülu 3 RLVR(최종) | 8B; 70B | RL | β; warmup ratio; 생성 temperature | 0.05(8B), 0.07(70B); 0.0 / 0.07; 1.0 | [[tulu-3]], arXiv:2411.15124v5 표 21 | 2026-09-15 검증 | β를 `[0.1, 0.05, 0.03, 0.01]`로 sweep; §6.2.1: KL budget 증가는 항상 검증 보상을 올리지 않음 |
| DAPO (Qwen2.5-32B base) | 32B | RL | KL; ε_low / ε_high; rollout | 없음; 0.2 / 0.28; 프롬프트 512 × 16 | [[dapo]] §2.3, §4.1 | 2026-09-15 검증 | 표 1: clip-higher AIME24 avg@32 `36 → 38`, 행당 실행 하나; KL 제거 절제 없음 |
| Dr. GRPO 실행 | 1.5B-7B | RL | KL loss coef; KL penalty coef; temperature; top-p / top-k | 0.0; 0.0; 1.0; 1.0 / −1 | [[dr-grpo]] §3, 부록 G 표 6 | 2026-09-14 검증 | 이 논문에 절제 없음 |
| Cui et al. 실행(Qwen2.5) | 7B, 32B | RL | Clip-Cov r; ω_low; ω_high | `2×10⁻⁴`; 1; 5 | [[entropy-mechanism-llm-rl]] §4.3 | 2026-09-14 검증 | 그림 12(7B): 더 많은 clipped token에서 엔트로피 상승; r 대 정확도는 미보고 |
| Cui et al. 실행(Qwen2.5) | 7B; 32B | RL | KL-Cov k; β | `2×10⁻³`(7B), `2×10⁻⁴`(32B); 1 | [[entropy-mechanism-llm-rl]] §4.3 | 2026-09-14 검증 | 표 2: 7-벤치마크 평균 7B `38.6 → 40.6`, 32B `45.8 → 52.2` |
| Cui et al. §4.1 sweep | 미명시(§4 실험은 Qwen2.5-7B 및 32B) | RL | entropy-loss 계수 sweep | `{0.0001, 0.001, 0.005, 0.01}` | [[entropy-mechanism-llm-rl]] §4.1, 그림 9 | 2026-09-15 검증(그림 9의 모델 미인쇄) | 그림 9: 0.01 엔트로피 폭발; 0.005 안정적이나 이득 없음 |
| Forking-token DAPO(Qwen3 base) | 8B, 14B, 32B | RL | ρ; ε_low / ε_high; KL; entropy loss; LR | 20%; 0.2 / 0.28; 없음; 없음; 상수 `1e-6` | [[high-entropy-minority-tokens]] §5.2 | 2026-09-15 검증 | 표 2: 32B 평균 `66.59 → 70.69`; §5.3: `ρ = 10/50/100%` 모두 20%보다 낮은 엔트로피 |
| Park et al. GSM8K 실행(Qwen2.5-3B-Instruct) | 3B | RL | ε_high; ε_low; KL; entropy loss; LR; temperature | ∞; 0.15; 없음; 없음; `5e-7`; 1.0 | [[clip-low-clip-high-entropy]] §3.2, 부록 C.1 표 1 | 2026-09-15 검증 | 그림 5: collapse와 explosion 균형; 그림 6: 동일 mean@8에서 pass@8 보존 |
| Nemotron-Research-Reasoning-Qwen-1.5B (ProRL) | 1.5B | RL | KL 배치; β; 참조 reset; temperature | 손실의 KL(식 4); 미보고; stagnation 시 π_ref와 optimizer hard reset; 1.2 | [[prorl]] §2.3.1, §3.2, §3.3 | β 미보고(body, 부록 D-F, model card 확인) | KL, reset, temperature 절제 없음 |
| Olmo 3 RL(공개 스크립트) | 7B, 32B | RL | KL 계수; temperature; clip | β 0.0(기본 0.05 미사용); 1.0; 0.2 / 0.272 | [[allenai-olmo3-open-instruct-scripts-recipe]] | 2026-09-14 검증 | 절제 보고 없음 |
| DeepSWE-Preview | 32B | RL | KL; entropy coefficient; temperature | `use_kl_loss=False`, in-reward KL 주석; 0.0; 1.0 | [[deepswe-recipe]] | 2026-09-14 검증 | 블로그 §2.3: entropy loss가 상승 엔트로피와 붕괴 초래; 곡선만 |
| SimpleTIR(Qwen2.5-7B base) | 7B | RL | β; entropy coefficient; clip; rollout; PPO epoch | 0; 0; 0.2 / 0.28; 프롬프트당 16; 4 | [[simpletir]] 부록 표 6 | 2026-09-15 검증 | 표 2: void-turn filtering AIME24 50.5, 대안 20.8–26.3 |
| Kimi-K2-Instruct | 활성 1.04T / 32B | RL | Rollout temperature | 높은 값에서 낮은 값으로 decay schedule(값 미인쇄) | [[kimi-k2-recipe]] §3.2.3 | 2026-09-14 검증(메커니즘) | 절제 보고 없음 |
| GRPO-Unlikeliness-2(DeepSeek-Prover-V1.5-SFT) | 7B | RL | β_KL; β_rank; PPO epoch | 0.10; 0.25; 2 | [[unlikeliness-reward]] 표 1 | 2026-09-15 검증 | 표 2: 9,600문제 중 8,065개 해결, GRPO-Default 7,860개; 표 3 pass@128 |
| verl 기본 config | 모든 크기 | RL | 참조 KL; estimator; entropy coefficient | 두 위치 모두 off(`kl_coef` 0.001, 보상 k1, 손실 k3); 0 | [[entropy-logging-patterns]] | 2026-09-14 검증 | 절제 보고 없음 |
| OpenRLHF CLI 기본 | 모든 크기 | RL | 참조 KL; estimator; entropy coefficient | 보상에 `init_coef` 0.01; k1; None | [[entropy-logging-patterns]] | 2026-09-14 검증 | 절제 보고 없음 |
| TRL `GRPOConfig` 기본 | 모든 크기 | RL | 참조 KL β; estimator; `top_entropy_quantile` | 0.0(참조 모델 미로드); k3; 1.0 | [[entropy-logging-patterns]] | 2026-09-14 검증 | 도움말은 [[high-entropy-minority-tokens]]를 인용해 entropy masking 0.2 권장 |

**작은 범용 실행의 시작점.** 여기의 모든 숫자는 조건과 함께 위 `검증` 행에서 왔다. group-baseline RL의 검증 가능 프롬프트에서 7B급 모델이라면 rollout temperature 1.0(Dr. GRPO 1.5B–7B, Olmo 3 7B, SimpleTIR 7B), top-p 1.0(Dr. GRPO 1.5B–7B), `ε_low = 0.2`, `ε_high = 0.28`(32B DAPO, forking-token 및 SimpleTIR 7B–32B에서 재사용), entropy 계수 0(verl 기본, DeepSWE 32B, SimpleTIR 7B)을 쓴다. β는 시작 정책에 대해 측정 KL을 기록하면서 0(Olmo 3 7B/32B, Dr. GRPO)으로 두거나, RL 데이터 밖 태스크 retention이 acceptance gate 일부이면 보상 안의 작은 β 0.02–0.05를 쓴다(InstructGPT 1.3B–175B, Tülu 3 8B). 엔트로피가 떨어지는데 정확도가 평평해지면, 이 규모에서 이득이 보고된 첫 변경은 배치 토큰의 최고 엔트로피 20%로 policy gradient를 제한(Qwen3-8B/14B/32B base, 수학 프롬프트)하거나 KL-Cov `k = 2×10⁻³`, `β = 1`로 covariance outlier에 penalty를 주는 것(Qwen2.5-7B, 수학 프롬프트)이다.

---

## 일반화 관점

**(a) 폭을 넓히는 것.**

- Coverage를 보상하는 목적함수: Pass@k training은 같은 프롬프트의 pass@1 training 대비 도메인 밖 KORBench Pass@1/Pass@k를 `37.7/45.6 → 47.7/63.5`, AIME 2025를 `5.4/19.1 → 7.1/22.4`로 올렸다([[pass-at-k-training]] 표 1, 셀당 실행 하나).
- Clipping을 통한 엔트로피 제어는 대칭 기본값이 잃었던 pass@8을 동일 mean@8에서 보존했다([[clip-low-clip-high-entropy]] 그림 5–6).
- 업데이트를 고엔트로피 토큰으로 제한하면 수학 학습 정책의 도메인 밖 코드 벤치마크가 개선됐다([[high-entropy-minority-tokens]] §5.4, 그림 9, 곡선만).
- KL 항과 참조 reset을 둔 긴 RL은 held-out Reasoning Gym 태스크에서 이득을 냈으며, 그중 하나에서는 시작 체크포인트가 k=256까지 모든 k에서 0.00이었다([[prorl]] §4.3, 표 3).

**(b) 좁아짐 또는 망각을 만드는 것.**

- 이미 가능성 높은 토큰에서 지속되는 양의 어드밴티지와 이미 가능성 낮은 토큰에서의 음의 어드밴티지(§2–§3). 대칭 clipping 아래 reward가 정보를 담지 않아도 clip-high bias가 이를 더한다([[clip-low-clip-high-entropy]] §2.3).
- Rank bias: GRPO 아래 정답 샘플 uplift rate는 초기 확률과 함께 오르며, 가장 낮은 확률의 정답 샘플은 “거의 uplift되지 않는다”([[unlikeliness-reward]] §3.5, 그림 4).
- 순차 post-training 단계는 모든 OLMo 2 크기에서 `distinct_10`을 낮췄다([[noveltybench]] 그림 6).
- base 정책에서 이동한 거리는 이전 태스크 망각을 예측하며([[rls-razor]] §3), 더 큰 KL 계수만으로는 퇴행을 고치지 못했다([[rlhf-instructgpt]] 부록 E.6).
- 긴 실행 일부에서 시작 모델이 이미 coverage를 가진 태스크에서 pass@1은 오르고 pass@128은 떨어진다([[prorl]] §4.2). [[rlvr-beyond-base-model]]은 [[prorl]]과 [[entropy-mechanism-llm-rl]] §2.6이 응답하는 pass@k 비판이다.

**(c) 이 단계에서의 측정 방법.** 고정 schedule로 다음을 보고한다. policy 생성 위치에서만 계산한 토큰별 엔트로피(평균과 상위 quantile); β 값이 아니라 held-out 프롬프트 세트의 시작 정책에 대한 측정 KL; held-out 프롬프트의 pass@1과 큰 k에서 pass@k; temperature 1.0의 개방형 프롬프트에서 `distinct_k` 또는 동등 다양성 metric([[noveltybench]] §3); RL 프롬프트 혼합에 없는 도메인을 포함하는 다중 도메인 retention suite. 알려진 측정 오류: 마스킹 관측 토큰 위에서 평균낸 엔트로피는 정책 통계가 아니다(§8). 적은 샘플에서 추정한 pass@k는 고분산이다. `distinct_k`는 partitioner와 sampling temperature에 의존한다([[noveltybench]] §3, §4.1).

---

## 흔한 실수와 감지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| “고공분산” 토큰 선택에 `p·A`로 순위 매김 | 선택 집합에 양의 어드밴티지 토큰만 있음; 엔트로피가 계속 하락 | 선택 집합의 어드밴티지 부호 histogram 출력; 식 10 중심화 곱과 비교([[entropy-mechanism-llm-rl]]) |
| dashboard KL 숫자를 최적화하는 양으로 읽음 | 같은 “kl” 값의 두 실행이 다르게 행동 | 각 stack에서 estimator, 비교하는 두 정책, 배치(보상/손실) 확인([[entropy-logging-patterns]]) |
| k1을 손실 항으로 사용 | KL curve는 제어된 듯하지만 정책은 regularization이 없는 듯 이동 | k1 손실의 기대 그래디언트는 0(§6); backward pass를 k2로 바꾸거나 k1을 보상으로 옮김 |
| 도메인 내부 점수가 올랐으니 β = 0은 비용 없다고 가정 | AIME식 점수는 오르는데 held-out/도메인 밖 점수는 하락 | 시작 정책에 대한 실현 KL 측정 및 retention suite 실행([[rls-razor]], [[rlhf-instructgpt]] 부록 E.6) |
| collapse의 첫 대응으로 entropy coefficient 올림 | 엔트로피가 오르다 폭발; 응답 길이와 보상 저하 | 보고 결과를 염두에 둔 sweep(0.01은 [[entropy-mechanism-llm-rl]] §4.1에서 폭발, 0.005는 [[pass-at-k-training]] §3.1에서 불안정화); 선택 개입 선호 |
| 엔트로피를 출력 다양성의 대리물로 취급 | 엔트로피는 안정적인데 모델이 거의 중복 답을 반환 | `distinct_k` 또는 pass@k 직접 측정([[noveltybench]], [[clip-low-clip-high-entropy]] 그림 6) |
| 에이전트 실행에서 도구 출력 위치의 엔트로피 평균 | 엔트로피 curve가 정책이 아니라 도구 장황함에 따라 움직임 | 평균 전에 feedback mask 적용(§8, [[simpletir]]) |
| sampling temperature가 다른 실행의 pass@k 비교 | temperature 고정 시 사라지는 coverage 차이 | 모든 평가의 temperature와 top-p 고정 및 보고 |

---

## 이해 점검

1. 공분산 결과는 로그확률과 어드밴티지가 양의 공분산일 때 엔트로피가 떨어진다고 말한다. 저확률 토큰의 *음의* 어드밴티지도 왜 엔트로피를 낮추며, 제거한 확률 질량은 어디로 가는지 설명하라.
2. 동료가 엔트로피 붕괴에 맞서 `p_t·A_t`로 순위가 높은 토큰 2%의 그래디언트를 0으로 하자고 한다. §3의 네 토큰 예를 사용해 이 규칙이 선택할 수 있는 것과 못 하는 것, 희귀 토큰 억제가 붕괴를 이끄는 실행에서 무엇을 놓치는지 설명하라.
3. 손실에 더한 k1 항은 제어된 듯한 KL curve와 regularization 없음으로 이어지고, 같은 식을 보상에 더하면 regularize하는 이유는 무엇인가? 각각의 그래디언트로 답하라.
4. DAPO는 `ε_high`를 올리고 Park et al.은 `ε_low`를 내린다. 둘 다 더 높은 엔트로피를 보고한다. clip 범위 양 끝의 두 변화가 같은 부호 효과를 낼 수 있는 이유를 설명하라.
5. Pass@k training은 `k = 4`에서 8샘플 그룹의 정답 7개에 어드밴티지 0을 준다. 산술적으로 왜 그런지와 큰 k의 pass@k에 왜 중요한지 설명하라.
6. InstructGPT는 KL 계수를 100× 올려도 공개 NLP 데이터셋 퇴행이 남았고, RL's Razor는 새 태스크의 KL이 LLM에서 `R² = 0.71`로 망각을 예측한다고 보고한다. 두 결과를 조화시켜라.
7. 다중 턴 도구 사용에서 도구 출력 토큰 손실 마스킹이 SimpleTIR의 그래디언트 노름 폭발을 막지 못하는 이유는 무엇인가? 명제 3.1을 사용해 답하라.
8. 평균 엔트로피 curve가 동일한 실행 두 개가 있고, 하나는 개방형 프롬프트에서 거의 중복 답을 내지만 다른 하나는 그렇지 않다. [[high-entropy-minority-tokens]]의 토큰 엔트로피 통계와 양립하는 메커니즘 하나와 이를 구분할 측정을 제시하라.

---

## 연결

- **이전:** **ch-16 — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse.** 난이도 필터링은 두 어드밴티지 부호가 모두 있는 그룹을 내는 프롬프트를 결정한다. 분산 0 그룹은 그래디언트도 엔트로피 변화도 없다.
- **다음:** **ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages.** §3과 음성 피드백 절에서 개괄한 softmax 재분배를 offline 대응물과 함께 유도한다.
- **ch-37 — Policy-Gradient Foundations for Language Models.** 이 챕터가 로그확률과의 공분산을 분석하는 estimator.
- **ch-38 — KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax.** §7이 사용하는 KL-in-reward 관례 및 alignment-tax 측정.
- **ch-38a — SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.** KL budget을 일반성 제어로 만드는 retention 근거(RL's Razor, alignment 아래 다양성 손실).
- **ch-40 — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO.** group baseline, clip-higher, β = 0 레시피를 알고리즘으로 소개하는 곳.
- **ch-44b — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing.** §7의 모든 것과 상호작용하는 보상 신호의 breadth.
- **ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability.** §8의 에이전트 설정 전체.
- **ch-55 — verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation.** 여기서 언급한 `kl_penalty`, `clip_cov`, `kl_cov` 코드 경로.

---

## 출처

- [[entropy-mechanism-llm-rl]] — entropy-performance 적합, 공분산 정리, 선택 규칙과 결과를 가진 Clip-Cov/KL-Cov, entropy-loss 및 reference-KL sweep의 보고 결과.
- [[high-entropy-minority-tokens]] — chain of thought의 토큰 엔트로피 통계, forking-token masking 규칙(식 6), Qwen3 크기별 결과, entropy-bonus 대 clip-higher 논증.
- [[clip-low-clip-high-entropy]] — clip-low가 엔트로피를 올리고 clip-high가 내림을 보이는 정리 및 실험, clipping 엔트로피 제어가 pass@k를 보존한다는 결과.
- [[dapo]] — KL 항 제거의 명시한 이유와 값이 있는 clip-higher 메커니즘.
- [[pass-at-k-training]] — group 수준 목적함수, 해석적 어드밴티지, entropy 행동, 도메인 내부/밖 결과.
- [[unlikeliness-reward]] — GRPO rank bias, rank-scaled 보상, 정리 증명의 pass@N 결과.
- [[noveltybench]] — `distinct_k`, `utility_k`, OLMo 2의 단계별 다양성 하락.
- [[rls-razor]] — 새 태스크에서 측정한 fine-tuned와 base 정책 사이 KL로 예측한 망각.
- [[john-schulman-kl-tricks]] — §5의 k1/k2/k3 estimator와 bias/variance 수치.
- [[kl-control-rlhf]] — KL-regularized 목적함수와 closed-form optimum(Korbak); §6의 InstructGPT 위치와 함께 사용.
- [[rlhf-instructgpt]] — 보상 안 β = 0.02, 최적 범위, 능력을 회복하지 못한 KL sweep.
- [[grpo]] — 식 4 estimator를 쓴 손실 안 KL과 DeepSeekMath-RL의 β = 0.04.
- [[dr-grpo]] — 전반에 β = 0을 둔 레시피.
- [[prorl]] — 참조 정책 reset을 둔 손실 안 KL, 긴 RL 실행의 pass@k regime.
- [[tulu-3]] — RLVR β 값과 KL budget 증가가 항상 검증 보상을 높이지는 않는다는 관찰.
- [[allenai-olmo3-open-instruct-scripts-recipe]] — β = 0.0, temperature 1.0의 공개 RL 스크립트.
- [[deepswe]], [[deepswe-recipe]] — KL 손실과 entropy 손실이 없는 에이전트 실행 및 명시한 이유.
- [[kimi-k2-recipe]] — 공개된 RL 메커니즘인 rollout-temperature decay.
- [[simpletir]] — 도구 피드백 뒤 저확률 토큰, 그래디언트 노름 명제, void-turn filtering.
- [[entropy-logging-patterns]] — adaptive-KL 식 및 어드밴티지 부호 진단을 포함해 verl, OpenRLHF, TRL이 계산·로그·기본 설정하는 것.
- [[verl-ppo-loss]] — 비대칭 clipping 및 음의 어드밴티지의 dual clip이 있는 clipped surrogate.
- [[entropy-regularization-ppo]] — entropy bonus(A3C, Williams & Peng 1991 인용)의 기원 및 계수.
- [[entropy-collapse-ppo]] — 한 태스크 외에서는 어떤 regularizer도 유의하게 도움이 되지 않은 continuous-control sweep.
- [[maximum-entropy-rl]] — SAC의 maximum-entropy 목적함수와 temperature 민감도, LLM-RL 유추가 제한되는 이유.
- [[rlvr-beyond-base-model]] — coverage 논의가 응답하는 pass@k 비판. 주장에 대해서만 인용하며 숫자는 인용하지 않는다(카드가 재검증되지 않음).
- [[nathan-lambert-entropy-rl]], [[openrlhf-entropy-debugging]] — 단일 검증 가능 artifact나 검증 기록이 없는 practitioner 종합. framing으로만 인용한다. 엔트로피 threshold(0.1, 0.2 nats)는 찾은 primary source가 없으며 이 챕터에서 사용하지 않는다.
