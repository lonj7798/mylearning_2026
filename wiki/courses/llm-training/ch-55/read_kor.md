<!-- chapter: ch-55
     track: infra
     kind: content
     title: verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation
     deps: [ch-54]
     sources: [[verl-ppo-loss]], [[verl-grpo]], [[verl-rollout]], [[entropy-logging-patterns]], [[verl-dapo-recipe]], [[gigpo-verl-agent]], [[deepspeed-ulysses]], [[prorl]], [[dr-grpo]], [[grpo]], [[ppo]], [[kl-control-rlhf]], [[async-rollout]], [[rlvr-beyond-base-model]]
     figures: figures/verl-structure.html, figures/verl-loss-branches.html
     revised: 2026-09 (generality revision)
-->

# 제55장 — verl 내부 구조: 손실, 롤아웃, 다중 도메인 보상, 루프 내 검증

> **핵심 통찰.** verl은 다른 코드베이스에서 흔히 하나로 결합되는 세 가지, 즉 이점 추정기(`@register_adv_est`), 정책 손실(`@register_policy_loss`), 롤아웃 엔진(별도 워커 프로세스)을 분리한다. 이 분리 때문에 PPO, GRPO, Dr. GRPO, DAPO 및 그 계열은 서로 다른 학습 루프가 아니라 설정으로 구분된다. 실행이 좁은 과제 전문 모델을 만들지, 넓은 능력을 유지할지도 설정이 결정한다. 각 `data_source`가 어느 보상 함수로 라우팅되는지, 기준 정책에 대한 KL 항을 손실에 둘지(`actor.use_kl_loss`) 보상에 둘지(`algorithm.use_kl_in_reward`), 그리고 어떤 검증 세트를 `trainer.test_freq`에서 실행할지가 그 설정이다.
>
> **지침.** 알고리즘을 이해하려고 verl을 읽을 때는 `verl/trainer/ppo/core_algos.py`와 레지스트리 데코레이터부터 시작하라. 저장소의 모든 알고리즘은 설정에서 이름으로 선택되는 등록 함수이기 때문이다(core_algos.py L53–67, L116–134). 둘 이상의 도메인으로 학습할 때는 `data.reward_fn_key`(기본값 `data_source`)를 설정하고 각 값이 자체 채점기로 가도록 라우팅하라. `NaiveRewardManager`가 샘플마다 그 필드를 읽고, 등록되지 않은 값에 대해 `default_compute_score`가 `NotImplementedError`를 발생시키기 때문이다(naive.py L121–136; reward_score/__init__.py L107). 실행이 학습 도메인 밖의 능력을 잃지 않아야 하면 `trainer.test_freq`를 0보다 크게 설정하고 비대상 검증 파일을 포함하라. 검증 경로는 `data_source`별 메트릭을 내보내며, 루프의 다른 어느 부분도 학습하지 않은 도메인을 측정하지 않기 때문이다(ray_trainer.py L718–740, L1727–1728).

---

## 이 장이 범용 모델에 중요한 이유

이 장은 파이프라인의 RL 단계(사전학습 → 중간학습 → SFT → 선호도 최적화 → RL → 평가)에 위치한다. 이 단계에서 학습 신호는 응답 하나당 스칼라 하나이며, 사전학습과 SFT에서 쓰는 토큰별 목표보다 담는 정보가 적다. 그 스칼라가 한 도메인의 검증기 하나에서 오면, 최적화기에는 다른 곳에서 능력이 사라지는 것을 막는 항이 없다. 프레임워크는 그에 저항하는 항, 즉 여러 보상 라우트, 기준 정책에 대한 KL 항, 학습하지 않는 도메인에서의 루프 내 평가를 얼마나 쉽게 추가할 수 있는지를 결정한다.

verl은 여러 범용 능력 RL 실행이 사용했다고 보고한 프레임워크다. 수학, 코드, STEM, 논리 퍼즐, 지시 따르기에서 하나의 1.5B 정책을 학습하는 ProRL 실행은 프레임워크로 verl을 명시한다([[prorl]], §3.2). 따라서 verl을 읽는 데에는 구현 세부를 넘는 두 번째 목적이 있다. verl이 노출하는 메커니즘 집합은 RL 실행이 범용성을 유지하기 위해 조작할 수 있는 레버의 집합에 가깝다.

별도 언급이 없으면 이 장의 사실은 커밋 `753aed3e1c286ba6825a74342b28669e72c083ea`(2026-09-14 커밋)에서 읽었다. 줄 번호는 이동한다. 함수명과 설정 키 이름은 줄 번호보다 오래 안정적으로 유지되어 왔다.

---

## §1 고정한 커밋에서의 저장소

**정의.** verl은 Ray가 조정하는 트레이너가 생성, 로그 확률 계산, 최적화를 위한 별도 워커 그룹을 구동하는 LLM 사후학습용 RL 라이브러리다.

**알고리즘이 존재하는 네 위치.**

1. `verl/trainer/ppo/core_algos.py` — 이름으로 각각 등록되는 이점 추정기와 정책 손실이다. `@register_policy_loss`(L53–67)와 `@register_adv_est`(L116–134)는 딕셔너리에 쓰는 평범한 데코레이터이며, `AdvantageEstimator` 열거형은 14개 이름을 나열한다(L88–110).
2. `verl/workers/utils/losses.py` — `ppo_loss`는 `get_policy_loss_fn(config.policy_loss.loss_mode)`를 호출하고(L101–112), 정책 손실이 반환된 뒤 엔트로피 및 KL 항을 더한다(L57–144).
3. `verl/trainer/ppo/ray_trainer.py` — 단계 순서, 즉 생성, 로그 확률 재계산, 채점, 이점 계산(L235–247), 업데이트, 검증이다(L596–770, L1727–1728).
4. `verl/workers/rollout/` — 자체 프로세스에서 동작하는 생성 엔진들이다.

기본 정책 손실은 `vanilla`다(`actor.yaml` L61). `gspo`, `sapo`, `clip_cov`, `kl_cov`, `cispo`, `bypass_mode`를 포함하여 그 밖에 열한 이름이 등록되어 있다(core_algos.py L1379–2414) ([[verl-ppo-loss]]).

**더 오래된 자료와 비교하면 눈에 띄는 두 구조 변화.**

- 워커 클래스는 PR #6067, “deprecate workers, migrate to engines”에서 엔진 클래스로 대체되어 제거되었다. 이 PR은 커밋 `044bbba`로 2026-04-20에 병합되었다. `verl/workers/actor/*` 형태의 경로는 더 이상 존재하지 않으며, 동등한 코드는 `verl/workers/engine/` 아래에 있다.
- 동기 SPMD vLLM 롤아웃은 폐기되었다. `ServerAdapter.generate_sequences`는 이제 “The vLLM SPMD mode was retired in PR #4411”이라는 메시지와 함께 `NotImplementedError`를 발생시키며(`vllm_rollout.py` L325–341), 롤아웃 설정의 기본값은 `mode: async`다(`rollout.yaml` L8) ([[verl-rollout]]). `ServerAdapter` 자체는 제거되지 않았다. 서버 모드용 어댑터이고 여전히 `update_weights`를 구현한다(`vllm_rollout.py` L209–253). PR #4411의 병합 날짜는 여기서 읽은 소스 파일만으로는 확정할 수 없다.

클릭 가능한 모듈 맵 [figures/verl-structure.html](figures/verl-structure.html)은 위 각 파일의 역할, 핵심 함수, 이를 선택하는 설정 키를 나열한다. 로그에서 관찰한 증상에 해당하는 파일을 찾을 때 사용하라.

## §2 정책 손실을 처음부터 끝까지 읽기

**다루는 문제.** 이전 정책이 뽑은 샘플로 계산한 정책 경사 단계는 편향되고 분산이 큰 추정치다. [[ppo]]의 클리핑된 대리 목적은 샘플링 정책에 비해 한 번의 업데이트가 샘플 토큰의 확률을 얼마나 멀리 움직일 수 있는지 제한한다.

**구현된 메커니즘.** `core_algos.py` L1336–1361을 인용한다([[verl-ppo-loss]]).

```python
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

**수식.** 응답 토큰 t에 대해 다음과 같다.

- `A_t ≥ 0`일 때: `L_t = max(−A_t·r_t, −A_t·clip(r_t, 1−ε_low, 1+ε_high))`
- `A_t < 0`일 때: `L_t = min( max(−A_t·r_t, −A_t·clip(r_t, 1−ε_low, 1+ε_high)), −A_t·c )`

`r_t = exp(clamp(log π_θ(t) − log π_old(t), −20, 20))`는 토큰 확률 비율이다. `π_θ`는 업데이트 중인 정책이고, `π_old`는 롤아웃과 함께 로그 확률이 저장된 정책이다. `A_t`는 토큰 이점이다. `ε_low = clip_ratio_low`, `ε_high = clip_ratio_high`이며, 둘 모두 기본값은 `clip_ratio = 0.2`다(`actor.yaml` L36, L39, L42). `c = clip_ratio_c`의 기본값은 3.0이고 1.0보다 커야 한다고 단언된다(`actor.yaml` L83; core_algos.py L1323–1334). 클리핑되지 않은 분기에서는 `∂L_t/∂ log π_θ = −A_t·r_t`이고, 클리핑된 분기에서는 그래디언트가 0이다.

**계산 예 1 — 음의 이점 토큰, `A = −1`, `ε_low = ε_high = 0.2`, `c = 3`.**

| `r` | `−A·r` | `−A·clip(r)` | max | 이중 클립 경계 `−A·c` | `L` | 그래디언트 |
|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 0.8 | 0.8 | 3.0 | 0.8 | 0 (하한 클립 활성) |
| 1.0 | 1.0 | 1.0 | 1.0 | 3.0 | 1.0 | 1.0 |
| 2.0 | 2.0 | 1.2 | 2.0 | 3.0 | 2.0 | 2.0 |
| 5.0 | 5.0 | 1.2 | 5.0 | 3.0 | 3.0 | 0 (이중 클립 활성) |

두 가지로 읽을 수 있다. 이미 `(1−ε_low)·π_old` 아래로 밀린 토큰은 이 업데이트에서 더 이상 확률을 낮추는 압력을 받지 않는다. 비율이 과도하게 커진 토큰도 더 이상 아래로 밀리지 않는다. 손실이 `c·|A|`에서 상한을 가지며, 그 상한은 `r`에 대해 상수이기 때문이다. Ye 등은 두 번째 경계의 이유를 제시한다. 오프폴리시 데이터에서는 “when Â_t < 0, such a large ratio r_t(θ) will introduce a big and unbounded variance”가 발생한다(arXiv:1912.09729v3, Algorithm Design, Eq. 5). 이들의 ε = 0.2 및 c = 3이 verl의 기본값이다.

**계산 예 2 — Clip-Higher, `A = +1`, `r = 1.25`.** `ε_high = 0.2`이면 `clip(1.25, 0.8, 1.2) = 1.2`이고, max는 클리핑된 항을 선택하므로 그래디언트는 0이다. `ε_high = 0.28`이면 비율이 밴드 안에 있고 그래디언트는 `−A·r = −1.25`다. 이 한 가지 변경이 DAPO의 Clip-Higher다. verl 문서는 설정을 `clip_ratio_low: 0.2`, `clip_ratio_high: 0.28`로 제시한다(docs/algo/dapo.md L52–57) ([[verl-dapo-recipe]]). DAPO는 `ε_low`를 0.2로 유지한다. “because increasing it will suppress the probability of these tokens to 0, resulting in the collapse of the sampling space”이기 때문이다(arXiv:2503.14476v2 §3.1). **결과(단일 연구):** Qwen2.5-32B base에서 점진적 스택에 Clip-Higher를 더하자 AIME24 avg@32가 36에서 38로 이동했다. 행마다 실행은 하나였다(arXiv:2503.14476v2 Table 1).

[figures/verl-loss-branches.html](figures/verl-loss-branches.html)의 인터랙티브에서는 `A`, `r`, `ε_low`, `ε_high`, `c`를 움직일 수 있고 위 네 분기 중 어느 것이 선택되는지와 그래디언트가 무엇인지 보여 준다. 두 표를 손으로 확인할 때 사용하라.

**집계.** `agg_loss`(L1140–1206)에는 다섯 모드가 있다. 기본값인 `token-mean`(`actor.yaml` L86)은 마스크된 합계를 전역 유효 토큰 수로 나눈다(L1170–1175). 독스트링은 전역 배치에 대해 집계하면 손실이 FSDP 또는 Megatron 병렬성 레이아웃에 불변이라고 설명한다(L1150). `seq-mean-token-sum-norm`은 시퀀스별로 합한 손실을 `loss_scale_factor`로 나누며, 그 기본값은 응답 마스크 폭이다(L1181–1193). 이 모드는 위 대수식의 어느 줄도 바꾸지 않고 그래디언트를 바꾼다. 그래서 같은 목적함수도 길이 편향을 가질 수도, 갖지 않을 수도 있다.

**롤아웃 중요도 가중치.** `rollout_is_weights`는 토큰별 손실에 곱해진다(L1364–1365). 이는 `algorithm.rollout_correction`이 구성되고, 배치에 롤아웃 로그 확률이 있으며, bypass 모드가 꺼져 있을 때에만 존재한다(ray_trainer.py L1648–1658). YAML 기본값은 `rollout_is: null`이고 선택지는 `"token"` 또는 `"sequence"`다(config/algorithm.py L85–89). 설정 주석은 오프폴리시성의 두 원인으로, 예를 들어 vLLM BF16과 FSDP FP32 사이의 롤아웃 엔진과 학습 엔진 정밀도 불일치, 그리고 오래된 롤아웃 체크포인트를 든다(L70–72). 정밀도는 여러 원인 중 하나다. 두 엔진이 같은 dtype으로 동작해도 커널 및 구현 차이가 불일치를 만들기 때문에, 이 보정은 dtype 수정이 아니라 일반적인 중요도 가중치로 작성되어 있다.

---

## §3 KL 항을 둘 위치

verl은 기준 정책에 대한 KL 항을 둘 수 있는 두 위치를 지원하며, 둘 다 기본값으로 꺼져 있다. 두 위치는 그래디언트와 실패 양상이 서로 다르므로 이를 정확히 구분하는 일이 중요하다.

**손실 안에 두기.** `actor.use_kl_loss`(기본값 `false`, `actor.yaml` L103)는 정책 손실에 `kl_loss * kl_loss_coef`를 더한다(losses.py L131–142). `kl_loss_coef`의 기본값은 0.001이고, `kl_loss_type`의 기본값은 k3 추정기인 `low_var_kl`이다(`actor.yaml` L113, L116). GRPO 문서는 이 배치를 권장하며 GRPO에는 `use_kl_loss: True`로 설정하라고 말한다(docs/algo/grpo.md L43–47).

**보상 안에 두기.** `algorithm.use_kl_in_reward`(기본값 `False`, `ppo_trainer.yaml` L98)는 토큰 보상을 `token_level_scores − β·kl_penalty(old_log_probs, ref_log_prob)`로 대체하고 `actor/reward_kl_penalty`를 기록한다(ray_trainer.py L78–117). 이 경로의 추정기는 `algorithm.kl_penalty`이며, 기본값은 `kl`(k1)이다(`ppo_trainer.yaml` L101). `kl_ctrl` 블록의 기본값은 `type: fixed`, `kl_coef: 0.001`, `horizon: 10000`, `target_kl: 0.1`이다(`ppo_trainer.yaml` L109–119). 적응형 제어기는 `β ← β·(1 + clip(KL/target − 1, −0.2, 0.2)·n_steps/horizon)`로 갱신한다(core_algos.py L153–174) ([[entropy-logging-patterns]]).

**추정기.** 샘플링된 토큰에 대해 `δ = log π_θ − log π_ref`라고 두면, k1 = δ, k2 = δ²/2, k3 = exp(−δ) − 1 + δ다. verl은 입력을 ±20으로, k3 출력을 ±10으로 클램프한다(core_algos.py L2239–2245). `k3+` 같은 `+` 접미사는 k3 값을 유지하지만 straight-through 추정기를 통해 k2 그래디언트를 사용한다(L2201–2213).

**계산 예.** `δ = +0.1`인 토큰 하나를 보면, k1 = 0.1이고 k3 = exp(−0.1) − 1 + 0.1 = 0.90484 − 1 + 0.1 = 0.00484다. `δ = −0.1`인 다른 토큰을 보면, k1 = −0.1이고 k3 = exp(0.1) − 1 − 0.1 = 1.10517 − 1 − 0.1 = 0.00517이다. 두 토큰에서 평균을 내면 k1은 0.0, k3는 0.005를 낸다. k1은 KL의 불편 추정량이지만 토큰별로 음수 값을 취하므로, k1 곡선이 0 근처라고 해서 정책이 기준 정책에 가깝다는 뜻은 아니다. k3는 토큰별로 음수가 아니므로 그 곡선을 드리프트 크기로 직접 읽을 수 있다([[kl-control-rlhf]]).

**범용 모델에 대한 함의.** 기준 정책은 목적함수 안에서 보상 도메인 밖의 행동을 참조하는 유일한 항이다. 두 배치가 모두 꺼져 있으면 목적함수 안에는 시작 체크포인트로부터 멀어지는 이동을 막는 항이 없다. [[prorl]]은 이를 유지해야 한다고 주장한다. 저자들은 KL 제거의 근거가 되는 견해가 “often applies to base models prior to any supervised fine-tuning”이라고 관찰하고, 이미 일관된 사고 연쇄를 생성하는 체크포인트에서는 KL 항이 안정성과 엔트로피에 도움이 된다고 주장한다(§2.3.1). 또한 학습이 진행될수록 KL 항이 “may increasingly dominate the loss”할 수 있기 때문에, 검증이 정체되거나 악화될 때 기준 정책과 옵티마이저 상태를 하드 리셋한다(§2.3.1, §3.3). 두 주장은 논증과 하나의 학습 이력이지 절제 실험이 아니다. 리셋 규칙은 측정된 효과가 아니라 하나의 레시피로 다뤄야 한다. verl에는 내장된 기준 정책 리셋 스케줄러가 없으므로, 드라이버 스크립트에서 기준 모델 경로를 최근 체크포인트로 다시 가리키게 하여 구현한다.

---

## §4 GRPO, Dr. GRPO, Pass@k 추정기

**정의.** GRPO는 가치 모델을 같은 프롬프트에 대해 샘플링한 G개 응답의 평균 보상으로 대체한다([[grpo]], §4.1.2).

**메커니즘.** `compute_grpo_outcome_advantage`는 각 응답의 토큰 보상을 하나의 점수로 합산한 다음 그룹 안에서 정규화한다(core_algos.py L311–329) ([[verl-grpo]]).

```python
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
```

`A_i = (r_i − μ_g)/(σ_g + ε)`는 응답 i의 모든 토큰에 복사된다. `r_i`는 합산한 토큰 보상이고, `μ_g`와 `σ_g`는 같은 프롬프트 id를 공유하는 G개 응답에 대한 평균과 `torch.std`(제수 G−1)이며, `ε = 1e-6`이다(L272).

**계산 예, 이진 보상, G = 4.**

1. 정답 하나: μ = 0.25, σ = sqrt((0.75² + 3·0.25²)/3) = 0.5다. 정답 응답은 +1.5, 각 오답 응답은 −0.5를 받는다.
2. 정답 셋: μ = 0.75, σ = 0.5다. 각 정답 응답은 +0.5, 오답 하나는 −1.5를 받는다.
3. 정답이 0개 또는 4개: σ = 0이므로 모든 이점은 0/(0 + 1e-6) = 0이며, 이 그룹은 정책 그래디언트에 기여하지 않는다.
4. `norm_adv_by_std_in_grpo=False`이면 사례 1은 +0.75와 −0.25를, 사례 2는 +0.25와 −0.75를 낸다.

크기가 1인 그룹은 별도 사례다. 평균 0과 표준편차 1을 대입하므로 이점은 원시 보상을 1 + 1e-6으로 나눈 값과 같다(L315–317). 기본 `rollout.n`은 1이다(`rollout.yaml` L127). 따라서 이를 올리는 것을 잊은 GRPO 설정은 기준선 없이 원시 보상으로 학습하며, 0/1 보상에서는 음의 이점을 전혀 생성하지 않는다.

**Dr. GRPO는 하나가 아니라 네 설정이다.** 문서는 조합을 `loss_agg_mode: seq-mean-token-sum-norm`, 최대 응답 길이 같은 상수로 설정한(선택 사항으로 표시된) `loss_scale_factor`, `use_kl_loss: False`, `algorithm.norm_adv_by_std_in_grpo: False`로 제시한다(docs/algo/grpo.md L59–62). 표준편차 항과 길이 항은 서로 다른 위치에 있는 별개의 수정이다. 표준편차 항은 이점 함수에 있고, 길이 항은 `agg_loss`에 있다. 두 `seq-mean-token-sum*` 모드는 같은 본문을 공유한다. 즉 각 시퀀스 토큰에 대한 손실을 합한 뒤 전역 배치 크기로 나눈다. 어느 쪽도 시퀀스를 자기 길이로 나누지 않으므로, 둘 다 [[dr-grpo]]가 길이 편향의 원인으로 지목한 1/|o_i| 항을 제거한다(§3.1). `-norm`은 전체 손실에 적용되는 하나의 상수인 `loss_scale_factor`로 한 번 더 나눈다. `loss_scale_factor`를 설정하지 않으면 그 상수는 `loss_mask.shape[-1]`, 즉 현재 배치의 패딩된 응답 폭이어서 배치마다 바뀌며 유효 스텝 크기도 함께 움직인다(core_algos.py L1181–1193). `loss_scale_factor`를 고정 정수로 설정해야 정규화가 학습 전체에서 상수가 된다. [[dr-grpo]]는 길이 항과 표준편차 항을 모두 제거하면 학습 중 응답 길이가 늘어나는 것을 막고, 해당 벤치마크에서 오답 응답의 길이를 낮춘다고 보고한다(§3.2, Fig. 5; Qwen2.5-1.5B, R1 템플릿, MATH 학습 질문).

**Pass@k 이점.** `grpo_passk`(L471–530)는 arXiv:2503.19595를 구현한다. 크기 2 미만의 그룹에는 오류를 발생시키고, 가장 높은 보상의 응답에만 `r_max − r_second_max`의 이점을 부여하며, 선택적으로 그룹 표준편차로 나눈다. 나머지 모든 응답에는 0을 준다(L516–527). 어떤 응답도 음의 이점을 받지 않는다. 0/1 보상에서는 그룹에 정답이 정확히 하나일 때만 이점이 0이 아니다. 보상이 [1, 0, 0, 0]이면 정답 응답은 1/0.5 = 2.0을 받는다. 이 추정기는 최빈 답이 아니라 커버리지를 최적화한다. [[rlvr-beyond-base-model]]은 RLVR이 흔히 개선하지 못하는 양이 바로 그것이라고 주장한다.

**다목적 보상.** `gdpo`(L361–468)는 보상 차원별로 정규화하고 가중합을 취한 후 배치에 걸쳐 whitening을 수행한다(L466; arXiv:2601.05242). 실행에 서로 다른 스케일의 보상 성분, 즉 검증기, 형식 검사, 길이 패널티가 여러 개 있을 때, 이 경로는 한 성분이 스케일만으로 지배하는 것을 막는 진입점이다.

---

## §5 다중 도메인 학습: 혼합 데이터셋, 보상 라우팅, 도메인별 메트릭

**측정 가능한 문제.** 하나의 `data_source`로 학습한 실행은 하나의 검증기를 최적화한다. 손실 안에는 “답이 맞다”와 “답이 이 검증기를 만족한다”를 구분하는 것이 없으므로, 폭넓음은 목적함수가 아니라 데이터와 라우팅에서 나와야 한다.

**메커니즘, 단계별로.**

1. `data.train_files`와 `data.val_files`는 parquet 파일 목록 또는 단일 파일을 받는다(`legacy_data.yaml` L7–14). 여러 도메인은 여러 파일을 하나의 데이터셋으로 이어 붙여 들어가며, `data.shuffle`의 기본값은 True다(L68). 이 설정에는 파일별 샘플링 가중치가 없으므로 혼합 비율은 파일에 기록한 행 수다.
2. 각 행에는 `data_source` 문자열이 있다. `data.reward_fn_key`가 읽을 필드를 선택하며, 기본값은 `data_source`다(`legacy_data.yaml` L29–30).
3. `NaiveRewardManager`는 샘플별로 그 필드를 읽어 채점기에 넘긴다(`naive.py` L121–136).

```python
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            ...
                    score = self.compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
```

4. `default_compute_score`는 `data_source` 값에 대한 디스패치 테이블이다(`reward_score/__init__.py` L44–107). `openai/gsm8k`는 `gsm8k`로, `lighteval/MATH`, `DigitalLearningGmbH/MATH-lighteval`, `HuggingFaceH4/MATH-500`은 `math_reward`로, `math_dapo`, `math`, `math_dapo_reasoning` 및 `aime`로 시작하는 모든 것은 `math_dapo`로 간다. 여섯 개의 `numina_*` 이름은 `prime_math`로, `codecontests`, `apps`, `codeforces`, `taco`는 `sandbox_fusion.url`이 설정되었으면 샌드박스 실행 채점기로, 그렇지 않으면 `prime_code`로 간다. `hiyouga/geometry3k`는 `geo3k`로, 일곱 개의 `searchR1_*` 이름은 정확 일치 QA 채점기로 간다. 그 밖의 값은 모두 `NotImplementedError`를 발생시킨다(L107).
5. 자체 도메인은 `reward.custom_reward_function.path` 및 `.name`(기본 함수명 `compute_score`, `reward.yaml`)을 통해 붙이거나, 등록된 보상 매니저(`reward.reward_manager.name`, 기본값 `naive`)를 통해 붙인다. 보상 계산은 `reward.num_workers: 8`로 실행된다.
6. 채점기가 딕셔너리를 반환하면 그 `score` 키가 보상이 되고, 나머지 모든 키는 `reward_extra_info`에 추가된다(`naive.py` L145–149). 이것이 도메인별 `acc` 필드를 §6에서 설명하는 메트릭으로 운반하는 채널이다.

**작동하는 설정.** 수학, 코드, 지시 따르기의 세 도메인 실행이라면, `data_source`를 각각 `math_dapo`, `codecontests`, `my_org/ifeval`로 설정한 parquet 파일 세 개를 작성한다. 세 파일 모두를 `data.train_files`에 전달하고, 코드가 프로세스 내 채점기로 폴백하지 않고 실행되도록 `reward.sandbox_fusion.url`을 설정한다. 그리고 `my_org/ifeval`을 처리하며 나머지는 `default_compute_score`에 위임하는 함수를 `reward.custom_reward_function.path`로 제공한다. 원하는 혼합 비율에 맞춰 행 수를 설정한다. [[prorl]]은 프롬프트 136K개를 수학 40k, 코드 24k, STEM 25k, 논리 37k, 지시 따르기 10k로 사용했으며(§3.1, Table 4), 혼합에 대한 절제 실험은 보고하지 않았다.

**조건과 한계.** 딕셔너리를 반환하는 채점기는 비교하려는 모든 도메인에 대해 같은 키를 반환해야 한다. 메트릭 이름은 그 키들로 만들어지기 때문이다. 한 도메인에서는 연속 점수를, 다른 도메인에서는 이진 점수를 반환하는 채점기는 도메인별 메트릭을 같은 척도 위에 두지 않는다. 그룹 상대 이점은 그룹 안에서는 이를 무해하게 만들지만, 기록된 곡선들 사이에서는 그렇지 않다.

---

## §6 루프 내 검증

**측정 가능한 문제.** 학습 보상은 구성상 학습 도메인에서 상승한다. 다른 무엇이 움직였는지는 학습 곡선으로 관찰할 수 없다.

**메커니즘.**

1. `trainer.val_before_train`의 기본값은 True이며, 1단계 전에 전체 검증 패스를 실행한다(`ppo_trainer.yaml` L195; ray_trainer.py L1438–1443). `trainer.val_only`는 검증을 실행한 뒤 종료한다(L198).
2. `trainer.test_freq`의 기본값은 **−1**이며, 주기적 검증을 완전히 비활성화한다(`ppo_trainer.yaml` L201). 게이트는 `if self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)`이다(ray_trainer.py L1727–1728). 기본값으로 둔 실행은 시작과 끝에서 각각 검증 수치 하나만 만든다.
3. `_validate`는 각 검증 프롬프트를 `actor_rollout_ref.rollout.val_kwargs.n`회 반복하고, 학습과 같은 보상 경로로 응답을 채점하며, 각 샘플의 `data_source`를 기록한다(ray_trainer.py L596–719).
4. `_val_metrics_update`는 `data_source`별로 그룹화하고 `"{val-core|val-aux}/{data_source}/{var_name}/{metric_name}"`를 내보낸다(L721–740). 변수가 핵심 변수, 즉 있으면 `acc`, 없으면 `reward`이고, 이름이 최대 샘플 수에서 `mean`, `maj`, `best`로 시작하면 메트릭은 `val-core`로 간다. 그 밖의 모든 것은 `val-aux`로 간다.
5. `process_validation_metrics`는 프롬프트별로 계산한 뒤 프롬프트 전체에서 평균을 내어 다음을 산출한다. `mean@N`, `std@N`, 그리고 2, 4, 8, …, N까지 각 n에 대해 `best@n/mean`, `best@n/std`, `worst@n/mean`, `worst@n/std`, 채점기가 `pred` 필드를 반환했을 때의 `maj@n/*`다. 최고/최저 및 다수결 통계는 크기 n의 부트스트랩 재표본 1000개에서 나온다(`metric_utils.py` L896–1040).
6. 에이전트 실행에서는 롤아웃이 턴 수를 기록했을 때 `val-aux/num_turns/{min,max,mean}`이 내보내진다(L742–744).

**커버리지 측정을 조용히 없애는 기본값.** `val_kwargs` 기본값은 `temperature: 0`, `n: 1`, `do_sample: False`다(`rollout.yaml` L151–170). N = 1이면 코드는 `mean@1`만 내보내고 다른 것은 내보내지 않는다. `std@N`과 모든 `best@n`은 `n_resps > 1` 분기 안에 있기 때문이다(`metric_utils.py` L989). 따라서 pass@k 스타일 메트릭은 존재하지만 `val_kwargs.do_sample: True`, 0이 아닌 `val_kwargs.temperature`, 1보다 큰 `val_kwargs.n`을 설정하기 전에는 아무것도 산출하지 않는다.

**계산 예.** `val_kwargs.n: 8`, `temperature: 1.0`, `do_sample: True`로 설정하고, `math_dapo`(학습함), `openai/gsm8k`(학습함), `hiyouga/geometry3k`(학습하지 않음)에 대한 검증 파일을 둔다. `gen_ns(8)`은 [2, 4, 8]을 반환하므로, 각 `data_source`에서 `mean@8`, `std@8`, 그리고 부트스트랩 표준편차를 수반하는 `best@{2,4,8}/mean`을 얻는다. `val-core/hiyouga/geometry3k/acc/best@8/mean`은 하락하면서 `val-core/math_dapo/acc/mean@8`이 상승한다면, 그것은 학습 뒤가 아니라 학습 루프 안에서 측정한 좁아짐이다. `best@8/mean`은 8개 샘플에서 pass@8을 부트스트랩으로 추정한 값이다. 따라서 n = N에서는 같은 8개 응답에서 복원 추출하여 재표본을 만들며, 추정치는 샘플로부터 독립적이지 않다. 절대값이 아니라 추세를 읽어야 한다.

**추가할 만한 메트릭.** 기본 집합에는 범용 프롬프트에서 시작 체크포인트로부터의 드리프트를 측정하는 것이 없다. 모든 학습 도메인 밖의 고정 프롬프트 집합을 추가하고, 그 위에서 현재 정책과 기준 정책 사이의 토큰별 k3 KL 평균을 계산하여 검증 메트릭 옆에 기록하라. `use_kl_loss`가 켜져 있으면 이것은 KL 계수가 제어해야 할 양이고, 꺼져 있으면 아무것도 제어하지 않는 양이다.

---

## §7 롤아웃: 기준 경로와 프로덕션 경로

**HFRollout** (`hf_rollout.py` L39–177)는 학습 모듈에서 Hugging Face `generate`를 호출한다. FSDP 모듈은 `summon_full_params(writeback=False, recurse=False)`로 샤딩을 해제하며(L108–110), 출력은 `prompt_length + response_length`까지 오른쪽 패딩된다(L132–139). 모듈 독스트링은 이 클래스가 FSDP HybridShard에서 멈춘다고 기록한다(L16–18). 수치적 발산이 의심될 때 비동기 엔진과 로그 확률을 비교하는 데 사용하라. 규모 있는 학습에 쓰는 것이 아니다.

**비동기 vLLM 서버.** `vLLMHttpServer.generate`는 프롬프트 토큰 id를 받고 토큰별 로그 확률이 붙은 토큰 id를 반환한다(`vllm_async_server.py` L556–759) ([[verl-rollout]]). 로그를 읽을 때 중요한 단계는 다음과 같다.

1. `max_possible_tokens = max_model_len − len(prompt_ids)`이며, 이것이 1 미만이면 `ValueError`를 발생시킨다(L591–597).
2. `max_tokens`는 요청에서 가져오고, 없으면 `max_new_tokens`, 그것도 없으면 `min(response_length, prompt_length + response_length − len(prompt_ids))`에서 가져온다(L600–611).
3. 값은 `[1, max_possible_tokens]`로 클램프된다(L615–619). 이 클램프는 경고 없이 `max_tokens`를 줄인다. 이어지는 assert는 클램프 뒤에는 실패할 수 없다. 긴 프롬프트는 로그 줄 없이 응답 예산을 줄인다.
4. 요청은 `_submission_paused`가 설정된 동안 대기한 뒤(L662–665), `priority`를 `engine.generate`에 전달한다(L668–674). `AgentLoopManager`는 청킹 전에 `priority = np.arange(len(prompts))`를 할당하므로, priority는 오래됨 신호가 아니라 샘플별 인덱스다(`agent_loop.py` L1210–1214).
5. 중단된 요청은 `stop_reason="aborted"`와 함께 빈 토큰 id를 반환한다(L691–701).

**가중치 동기화.** `abort_all_requests`는 제출 게이트를 닫고, 진행 중인 승인에 최대 60 s를 기다린 다음 `pause_generation`을 호출한다(L932–1003). 코드 주석은 진행 중인 모든 요청이 중단되고 이전 가중치 캐시가 비워지지 않는 한 가중치 업데이트가 진행되어서는 안 된다고 명시한다(L989–991). 진행 중인 요청은 디코딩 중간에 일시 정지되는 것이 아니라 중단된다.

**배치 모드.** HYBRID는 가중치 동기화와 함께 GPU를 공유하는 한 프로세스에 롤아웃 및 학습 엔진을 둔다. COLOCATED는 LLM 판정기를 위해, 가중치 동기화 없이 별도 프로세스에서 같은 배치 그룹을 사용한다. STANDALONE은 별도 GPU를 사용하므로 오프폴리시다(`replica.py` L54–66).

**한 단계 중 생성이 차지하는 양.** **결과(단일 연구):** verl 자체의 완전 비동기 보고에서, Qwen2.5-Math-7B와 DAPO를 사용한 H20 GPU 128개 환경의 colocated 동기 기준선은 356.30 s 단계 중 177.85 s를 `gen`에 쓴다(약 50%). 400단계는 colocated에서 1d 16h 48m, 롤아웃:학습 분할 64:64의 완전 비동기에서는 17h 22m가 걸려 벽시계 시간 비율은 2.35×다. 마지막으로 보고된 AIME-2024 acc/mean@1은 colocated 0.2958, 비동기 0.3094였고, 최대값은 각각 0.3573과 0.3521이었다(`fully_async_policy/README.md` L333–369). 이는 한 구성, 한 벤치마크의 결과다. 이 비율을 다른 모델 크기나 클러스터로 가져가면 안 된다.

---

## §8 다중 턴 에이전트 루프

**메커니즘.** 다중 턴 및 도구 호출 롤아웃은 `verl/experimental/agent_loop/`에서 실행된다. `AgentLoopBase.run`은 `AgentLoopOutput(prompt_ids, response_ids, response_mask)`를 반환하며, 마스크는 모델이 생성한 토큰에는 1, 도구 응답 토큰에는 0이다(`agent_loop.rst` L33–72). `align_response_metadata`는 빌더가 삽입한 경계 토큰에 마스크 0을, 어시스턴트 토큰에는 롤아웃 로그 확률과 함께 마스크 1을, 도구 또는 사용자 토큰에는 로그 확률 0.0과 함께 마스크 0을 준다(`continuous_token.py` L390–413). 정책 손실은 정확히 이 마스크를 소비한다(`losses.py` L93–108). 그러므로 도구 출력은 절대 그래디언트를 받지 않고 `token-mean`의 토큰 분모에도 들어가지 않는다.

`ToolAgentLoop`는 PENDING → GENERATING → PROCESSING_TOOLS → TERMINATED 상태 기계다(`tool_agent_loop.py` L48–52, L166–178). 응답 마스크가 `response_length`에 도달하거나 `max_assistant_turns`/`max_user_turns`에 도달하면 종료한다(L282–287). 기본값은 `multi_turn.enable: False`, `max_parallel_calls: 1`, `max_tool_response_length: 256`, `tool_response_truncate_side: middle`, `format: hermes`, `tokenization_sanity_check_mode: strict`다(`rollout.yaml` L183–230). `agent.num_workers: 8`이고 `default_agent_loop: single_turn_agent`다(L246–249).

**토큰을 넣고 토큰을 내보내는 일이 중요한 이유.** 문서는 최종 메시지 목록에 채팅 템플릿을 적용하여 얻은 토큰 id가 턴별 프롬프트와 응답 id의 연결 결과와 다를 수 있다고 보고한다. 도구 파서가 어시스턴트 내용을 다시 쓰고 decode-then-encode가 항상 가역적이지 않기 때문이다. 최종 채팅 이력에 `apply_chat_template`를 쓰면 “make PPO training not even converged in single-turn”이라고도 명시한다(`agent_loop.rst` L155–182). 이는 수치 없는 경험 보고지만 메커니즘은 검사할 수 있다. 불일치는 삽입되거나 제거된 토큰 수만큼 마스크를 이동시키므로, 그래디언트가 잘못된 위치에 걸린다.

**컨텍스트 증가.** 매 턴은 하나의 토큰 시퀀스에 추가된다(`agent_data.prompt_ids = merge_result.token_ids`, L273, L388). `ToolAgentLoop`는 이전 턴을 리셋하거나 요약하지 않는다. 따라서 30턴 에피소드는 매 턴 전체 이력을 prefill하며, §7의 3단계 `max_tokens` 클램프가 턴마다 응답 예산을 줄인다.

**에이전트 포크가 더하는 것.** [[gigpo-verl-agent]](GiGPO, arXiv:2505.10978v3)는 verl 위에 구축되었으며, 전체 이력을 연결하지 않는 단계별 상호작용 패러다임, 매 단계에 어느 이력이 보일지 선택하는 메모리 모듈, gym 스타일 병렬 그룹 환경을 추가한다(App. A). 알고리즘 변화는 두 번째 그룹화 수준이다. GRPO와 같은 에피소드 수준 이점 `A^E(τ_i) = (R(τ_i) − mean(R))/F_norm(R)`(Eq. 3)에 더해, 하나의 과제 N개 궤적 전체에서 동일하게 반복되는 환경 상태, 즉 “anchor state”에서 취한 모든 행동을 수집하고, 그 집합 안에서 할인 수익을 정규화해 단계 수준 이점을 만든다(Eqs. 4–7). 둘은 `A(a_t^(i)) = A^E(τ_i) + ω·A^S(a_t^(i))`로 결합된다(Eq. 8). 여기서 `ω ≥ 0`은 에피소드 수준 항에 대한 단계 수준 항의 가중치다. 논문은 “with no further tuning”으로 `ω = 1`을 설정하고, ALFWorld와 WebShop에는 롤아웃 그룹 크기 `N = 8`, 검색 증강 QA에는 `N = 5`를 사용한다(§5.1). **결과(단일 연구, Table 1의 결과는 무작위 시드 3개 평균):** Qwen2.5-1.5B-Instruct에서 표준편차 정규화 없는 GiGPO는 GRPO를 “by 13.3% on ALFWorld and 10.6% on WebShop at 1.5B, and by 12.6% and 9.1%, respectively, at 7B”만큼 앞섰다. 검색 증강 QA에서는 3B에서 42.1%, 7B에서 47.2%에 도달했다(§5.2–5.3, Tables 1–2). 이 백분율은 성공률의 차이이며 논문은 퍼센트 기호로 표기한다. 저자들은 추가 롤아웃을 뽑지 않으므로 추가 비용이 “only < 0.002% time cost”라고 보고한다(§1). 또한 `F_norm = std`는 더 어려운 하위 과제에서 해로웠고 `F_norm = 1`이 그곳에서 도움이 되었다고 보고한다. 이는 난이도 편향에 관한 Dr. GRPO의 논거와 맞는다(§5.2).

**범용 모델에 대한 함의.** 단계 수준 크레딧은 한 에피소드의 서로 다른 단계에 서로 다른 이점을 부여하는 메커니즘이다. 이것이 없으면 50단계 에피소드의 모든 토큰이 같은 에피소드 수준 이점을 가지므로, 그래디언트는 결과를 바꾼 단계와 바꾸지 않은 단계를 구별하지 않는다.

---

## §9 긴 시퀀스: 시퀀스 병렬성과 길이 클램프

**클램프만으로 충분한 경우.** 롤아웃 길이는 생성 전에 세 번 제한되며(§7 3단계), `rollout.response_length`의 기본값은 `data.max_response_length`다(`rollout.yaml` L38–39). 응답이 컨텍스트 창 안에 들어가고 패킹된 마이크로배치 하나의 GPU별 활성화 메모리도 들어간다면 병렬성 변경은 필요 없다. `response_length`를 올리고 `ppo_max_token_len_per_gpu`를 낮추면 된다.

**충분하지 않은 경우.** 매우 긴 시퀀스로 학습하는 일은 시퀀스 길이와 함께 증가하고 데이터 병렬성으로는 줄일 수 없는 GPU별 활성화 메모리의 제한을 받는다. 시퀀스 병렬성은 시퀀스 자체를 분할한다.

**메커니즘(DeepSpeed-Ulysses).** P개 GPU 각각은 시퀀스의 `N/P`개 토큰을 보유하여 Q, K, V로 투영한다. 어텐션 전에 all-to-all이 재분배하여, 각 GPU는 서로 겹치지 않는 어텐션 헤드 부분집합에 대해 전체 시퀀스를 보유하게 된다. 헤드별로 어텐션을 계산한 뒤 두 번째 all-to-all이 출력을 MLP와 정규화 층을 위한 시퀀스 분할 레이아웃으로 되돌린다([[deepspeed-ulysses]], arXiv:2309.14509v2 §1, §3.1).

**통신.** 은닉 크기 h, 시퀀스 길이 N, 장치 P개에서 링크당 볼륨은 `4Nh/P`, 즉 O(N/P)이다. 집계 크기 M인 all-to-all의 링크당 비용이 M/P이기 때문이다. Megatron-LM 시퀀스 병렬성은 층마다 크기 Nh의 all-gather 두 번과 reduce-scatter 두 번을 사용하므로, 링크당 `4Nh`, 즉 P와 무관한 O(N)의 비용이 든다(§3.2). **결과(단일 연구):** 저자들은 비교 시스템의 4× 시퀀스 길이, 시퀀스당 백만 개가 넘는 토큰, 10×가 넘는 통신량 감소, 최대 2.5× 처리량, 하드웨어 피크의 54%가 넘는 GPU당 175 TFlops 초과의 지속 처리량으로 학습했다고 보고한다(§1). 이는 저자들의 하드웨어에서 얻은 2023년 측정값이다.

**verl에서.** `ulysses_sequence_parallel_size`는 FSDP 엔진 설정 필드이며 기본값은 1이다(`engine/fsdp.yaml` L44–45; `workers/config/engine.py` L292). 위 메커니즘의 헤드 수 제약은 코드에서 강제된다(`verl/utils/ulysses.py` L337–341).

```python
def validate_ulysses_config(num_heads, ulysses_sequence_size):
    if ulysses_sequence_size > 1:
        assert num_heads % ulysses_sequence_size == 0, (
            f"num_heads ({num_heads}) must be divisible by ulysses sequence size({ulysses_sequence_size})"
        )
```

입력은 SP 크기의 배수로 패딩되고 forward pass 전에 rank별로 슬라이스된다(`ulysses_pad_and_slice_inputs`, L302–334). 출력은 이후 gather되고 패딩이 제거된다(L247–280). `pad_to_length`가 켜져 있으면 `pad_to_length_bucket`은 내부적으로 `ulysses_sequence_parallel_size`의 배수로 올림된다(`engine/fsdp.yaml` L69–72). 시퀀스 병렬성은 학습 엔진에만 적용된다. 롤아웃 엔진은 자체 병렬성을 가진 별도 프로세스이므로, SP 크기를 올려도 정책이 더 긴 응답을 생성할 수 있게 되지는 않는다.

**보상 항으로서의 길이.** DAPO 레시피는 자르기보다 shaping을 한다. `data.max_response_length: 20480`, `overlong_buffer.enable: True`, `len: 4096`, `penalty_factor: 1.0`으로 설정하여, 응답이 `max_response_length − 4096`을 0에서 4096토큰만큼 초과할 때 패널티가 0에서 1.0까지 선형으로 상승한다(docs/algo/dapo.md L137–165) ([[verl-dapo-recipe]]). 17,408토큰 응답은 (17408 − 16384)/4096 = 0.25의 패널티를 보상에 음의 항으로 더한다. verl FAQ는 논문의 Overlong Filtering이 이 shaping과 겹치므로 여기에는 구현하지 않았다고 밝힌다(L169–171).

---

## 부정 샘플과 부정 피드백

**어떤 종류의 부정인가.** verl의 RL 경로는 부정을 주로 **그래디언트**로 사용한다(부정 피드백 장의 분류에서 종류 4). 혼합 그룹의 오답 응답은 음의 이점을 받고 정책 손실은 그 토큰들의 로그 확률을 낮춘다. 메트릭 값이 모두 같은 그룹을 `filter_groups`가 버릴 때에는 **음의 한계가치** 결정(종류 1)도 만든다. verl의 어느 부분도 실패를 내용으로 또는 제어 토큰 아래에서 학습하지 않는다. 그것은 SFT 단계의 선택이다.

**부정이 오는 곳과 레이블되는 방식.** `data_source`가 선택한 검증기에서 나온다(§5). 수학 채점기, 테스트 스위트 실행, 정확 일치 QA 검사다. 코드에는 거짓 음성 원인이 두 가지 보인다. `compute_score`가 `compute_score_timeout`을 초과하면 `NaiveRewardManager`는 보상 0.0을 할당하고 계속 진행한다(`naive.py` L129–143). 따라서 느린 채점기는 맞을 수도 있는 답을 부정 예로 바꾼다. 샌드박스 경로는 컴파일 오류, 구문 오류, 타임아웃에 0을 반환하므로 사용할 수 없는 답과 틀린 답을 합쳐 버린다. 기본 메트릭은 두 사건 모두 세지 않는다. 카운터를 추가하라.

**메커니즘.** softmax에 대해 `∂ log p_y/∂z_j = 1[j = y] − p_j`다. `log p_y`를 낮추면 다른 로짓은 현재 확률에 비례해 올라가므로, 제거된 질량은 이미 확률이 높았던 토큰으로 주로 간다. 따라서 낮은 확률 토큰을 반복해 아래로 누르면 분포를 재분배하는 대신 집중시키며, 이것이 강한 음의 그래디언트 아래 엔트로피가 감소하는 메커니즘이다.

**코드가 이들을 처리하는 방식.**

- 하한 클립은 `r < 1 − ε_low`가 된 음의 이점 토큰에서 그래디언트를 제거한다(§2, 계산 예 1, 행 1). 이것이 DAPO가 완화하지 않는 경계다. `ε_low`를 크게 하면 이 확률들을 0으로 몰아 샘플링 공간을 붕괴시킨다는 논거 때문이다(arXiv:2503.14476v2 §3.1).
- 이중 클립은 `c = 3.0`인 `c·|A|`를 손실이 넘으려는 순간 그래디언트를 제거한다(행 4). 음의 이점 토큰만 이 분기에 도달한다(`torch.where(advantages < 0, …)`).
- 분산이 0인 그룹은 그래디언트에 기여하지 않지만 그 토큰은 `response_mask`에 남는다(L329). 그러므로 `token-mean` 아래에서는 여전히 전역 토큰 분모에 포함된다(L1170–1175). 전부 오답인 배치는 다른 모든 토큰 업데이트의 크기를 낮춘다.
- `algorithm.filter_groups`(기본값 `enable: False`)는 전부 같은 그룹을 제거하고, `train_batch_size`개의 자격 있는 그룹이 생길 때까지 `gen_batch_size`로 재샘플링한다. 단, `max_num_gen_batches`가 상한이다(docs/algo/dapo.md L69–86). **결과(단일 연구):** Qwen2.5-32B에서 verl의 DAPO 재현은 동적 샘플링 시 AIME 2024 정확도 52%, 없을 때 50%를 보고한다. 각각 한 번의 실행이다(docs/algo/dapo.md L36–37).
- Pass@k 추정기는 음의 이점을 전혀 부여하지 않는다(§4).

**verl이 이미 내보내는 진단값.** `actor/pg_clipfrac`은 클리핑된 분기에 있는 토큰의 비율이고, `actor/pg_clipfrac_lower`는 `A < 0`인 토큰 중 손실이 이중 클립 경계를 넘은 비율이다(core_algos.py L1353, L1357–1359). 이 쌍은 이점 부호로 클리핑을 분리한다. `pg_clipfrac`이 상승하는데 `pg_clipfrac_lower`가 0 근처에 머물면 양의 이점 쪽 및 `ε_high`를 가리킨다. 반대라면 음의 이점 토큰에서 과도하게 커진 비율을 가리킨다. `actor/entropy`는 샘플링된 토큰 프록시가 아니라 로짓에서 계산한 범주형 엔트로피다([[entropy-logging-patterns]]).

**효과의 크기.** verl은 음의 이점이 pass@k, 캘리브레이션, 홀드아웃 과제에 미치는 효과를 측정해 보고하지 않는다. 카드의 Verification 절은 이를 “not reported” 아래에 나열한다. 프레임워크의 증거만으로 개선 중 어느 몫을 부정 샘플에 돌리면 안 된다.

---

## 레시피

프레임워크 기본값은 모델 레시피가 아니다. 아래 각 행은 어느 쪽인지를 말한다.

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 소스 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| verl 기본값 | n/a | RL | `clip_ratio`; `clip_ratio_low`; `clip_ratio_high` | 0.2; 0.2; 0.2 | verl@753aed3 `config/actor/actor.yaml` L36, L39, L42 | 2026-09-14 검증 | 절제 실험 미보고 |
| verl 기본값 | n/a | RL | `clip_ratio_c`(이중 클립) | 3.0(> 1.0 단언) | actor.yaml L83; core_algos.py L1323–1334 | 2026-09-14 검증 | 절제 실험 미보고 |
| verl 기본값 | n/a | RL | `loss_agg_mode`; `entropy_coeff`; `ppo_epochs` | `token-mean`; 0; 1 | actor.yaml L86, L93, L119 | 2026-09-14 검증 | 절제 실험 미보고 |
| verl 기본값 | n/a | RL | `use_kl_loss`; `kl_loss_coef`; `kl_loss_type` | false; 0.001; `low_var_kl` (k3) | actor.yaml L103, L113, L116 | 2026-09-14 검증 | 절제 실험 미보고 |
| verl 기본값 | n/a | RL | `use_kl_in_reward`; `kl_penalty`; `kl_ctrl` | False; `kl` (k1); fixed, `kl_coef` 0.001, horizon 10000, target 0.1 | `ppo_trainer.yaml` L98, L101, L109–119 | 2026-09-14 검증 | 절제 실험 미보고 |
| verl 기본값 | n/a | RL | `adv_estimator`; `rollout.n`; `norm_adv_by_std_in_grpo` | `gae`; 1; True | ppo_trainer.yaml L74, L77; `rollout/rollout.yaml` L127 | 2026-09-14 검증 | 문서: GRPO에는 `grpo` 및 n > 1 설정(grpo.md L29, L39) |
| verl 기본값 | n/a | RL | `val_before_train`; `val_only`; `test_freq` | True; False; −1(비활성화) | ppo_trainer.yaml L195, L198, L201 | 2026-09-14 검증 | 절제 실험 미보고 |
| verl 기본값 | n/a | RL | `val_kwargs.temperature`; `.n`; `.do_sample` | 0; 1; False | rollout.yaml L164, L167, L170 | 2026-09-14 검증 | 절제 실험 미보고 |
| verl 기본값 | n/a | RL | `ulysses_sequence_parallel_size` | 1 | `config/engine/fsdp.yaml` L45; `workers/config/engine.py` L292 | 2026-09-14 검증 | 절제 실험 미보고 |
| verl 기본값 | n/a | RL | `reward_fn_key`; `reward_manager.name`; `reward.num_workers` | `data_source`; `naive`; 8 | `config/data/legacy_data.yaml` L29–30; `config/reward/reward.yaml` | 2026-09-14 검증 | 절제 실험 미보고 |
| verl 문서, DAPO 레시피 | n/a | RL | `clip_ratio_low`; `clip_ratio_high` | 0.2; 0.28 | docs/algo/dapo.md L52–57 | 2026-09-14 검증 | 문서에 없음 |
| verl 문서, DAPO 레시피 | n/a | RL | `gen_batch_size`; `train_batch_size`; `filter_groups.metric`; `max_num_gen_batches` | 1536; 512; `acc`; 10 | docs/algo/dapo.md L73–82 | 2026-09-14 검증 | 문서에 없음 |
| verl 문서, DAPO 레시피 | n/a | RL | `max_response_length`; `overlong_buffer.len`; `penalty_factor` | 20480; 4096; 1.0 | docs/algo/dapo.md L141–149 | 2026-09-14 검증 | 문서에 없음 |
| verl 문서, Dr. GRPO | n/a | RL | `loss_agg_mode`; `loss_scale_factor`; `use_kl_loss`; `norm_adv_by_std_in_grpo` | `seq-mean-token-sum-norm`; 최대 응답 길이 등의 상수 정수(선택 사항으로 표시); False; False | docs/algo/grpo.md L59–62 | 2026-09-14 검증 | 문서에 절제 실험 미보고 |
| verl에서 DAPO 재현, Qwen2.5-32B base | 32B | RL | 동적 샘플링 켬/끔 | AIME 2024 acc. 52% 대 50% | docs/algo/dapo.md L36–37 (커밋 4f80e4, 16×8×H800) | 2026-09-14 검증 | 행마다 실행 하나; W&B 기록 연결 |
| Qwen2.5-Math-7B(verl fully_async 실험) | 7B | RL | 알고리즘; 최대 응답; `rollout.n`; 하드웨어 | DAPO; 28K 토큰; 16; H20 128개(64:64) | `fully_async_policy/README.md` L333–340, L361 | 2026-09-14 검증 | 32/64/128-GPU 비교 L353–361 |
| Qwen2.5-Math-7B(verl fully_async 실험) | 7B | RL | `staleness_threshold`; `partial_rollout` | 0.5; True | README.md L351–352 | 2026-09-14 검증 | staleness 절제 0/0.1/0.3/0.5, L380–396 |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | 프레임워크; 알고리즘; ε_low/ε_high | verl; GRPO + 분리된 클립 + 동적 샘플링; 0.2/0.4 | arXiv:2505.24864v1 §3.2 | 2026-09-14 검증 | §2.3 “helps retain entropy”; 수치 없음 |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | 데이터(프롬프트, 다섯 도메인) | 136K: 수학 40k, 코드 24k, STEM 25k, 논리 37k, 지시 따르기 10k | arXiv:2505.24864v1 §3.1, Table 4 | 2026-09-14 검증 | 혼합 절제 실험 미보고 |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | 평가 게이트 | 검증 혼합 | AIME2024, Codeforces, GPQA-diamond, IFEval, graph_color | arXiv:2505.24864v1 App. E | 2026-09-14 검증 | 최종 체크포인트 선택 규칙 미보고 |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | KL 계수 β | 미보고 | arXiv:2505.24864v1 §2.3.1 | 미보고(본문, App. D–F, HF 모델 카드 검사) | 절제 실험 미보고 |
| GiGPO (verl-agent), Qwen2.5-7B-Instruct | 7B | RL | 롤아웃 그룹 크기 N; 가중치 ω; GRPO 대비 ALFWorld 이득 | 8; 1(튜닝 미보고); 12.6% | arXiv:2505.10978v3 §5.1–5.2, Table 1 | 2026-09-14 검증 | 시드 3개; ALFWorld와 WebShop만 |

**작은 범용 실행의 시작점.** 여기의 모든 수치는 위 `verified` 행에서 왔다. `algorithm.adv_estimator: grpo` 및 `actor_rollout_ref.rollout.n: 16`(1.5B에서 ProRL 값, §3.2)을 설정해 GRPO를 사용한다. `clip_ratio_low: 0.2`를 유지하고 `clip_ratio_high: 0.28`로 올린다(verl 문서 DAPO 레시피). `clip_ratio_c: 3.0`과 `loss_agg_mode: token-mean`을 유지한다(verl 기본값). `actor.use_kl_loss: True`, `kl_loss_coef: 0.001`, `kl_loss_type: low_var_kl`로 기준 항을 켠다(verl 기본값이며 GRPO 문서가 권장하는 배치). `metric: acc`, `gen_batch_size: 1536`, `train_batch_size: 512`, `max_num_gen_batches: 10`으로 `algorithm.filter_groups`를 켠다(verl 문서 DAPO 레시피). 그리고 양수 `trainer.test_freq`, `val_kwargs.n: 8`, `do_sample: True`, 학습 및 비학습 도메인별 검증 파일을 설정한다. 조건: DAPO 값은 수학에서 16×8×H800으로 32B에서 읽었고, ProRL 값은 다섯 도메인 136K 세트, 4×8 H100-80GB, 약 16k GPU 시간으로 1.5B에서 읽었다. 어느 것도 당신의 모델 크기에 맞춰 튜닝되지 않았으며, 위 어느 행도 전이되는 절제 실험을 담지 않는다.

---

## 일반화 관점

**(a) 무엇이 폭을 넓히는가.**

- 하나의 실행에서 여러 `data_source` 값을 여러 검증기로 라우팅하는 것은 하나보다 많은 도메인을 그래디언트에 넣는 메커니즘이다(§5). **결과(단일 연구):** 다섯 도메인에 걸친 ProRL 1.5B 실행은 시작 체크포인트와 비교해 보고한 모든 축에서 개선했다. 수학 pass@1은 44.45 → 60.14, 코드는 23.08 → 37.49, GPQA-Diamond은 15.86 → 41.78이었으며, IFEval의 평균 검증기 보상은 44.05 → 66.02, Reasoning Gym은 4.24 → 59.06이었다(§3.4, Tables 1–3). 코드 평균에서 DeepCoder-1.5B를, 수학 평균에서 DeepScaleR-1.5B를 넘었다([[prorl]]).
- 기준 항을 유지하고 검증이 정체될 때 기준 정책을 리셋하는 것은 2k 단계에 걸친 지속 이득에 보고가 연결한 레시피다(§2.3.1, §3.3). 리셋을 KL 항이나 과제 다양성과 분리한 절제 실험은 없다.
- 다중 턴 설정의 단계 수준 크레딧은 에피소드 수준 스칼라가 담지 않는 그래디언트 신호를 추가한다([[gigpo-verl-agent]], §5.2).

**(b) 무엇이 좁아짐을 일으키는가.**

- 단일 검증기 실행에는 범용 행동을 선호하는 항이 없다. verl에서 유일한 평형추는 KL 항이며, 두 배치 모두 기본으로 꺼져 있다(§3).
- 분산 0 그룹과 기본 `rollout.n: 1`은 음의 신호를 완전히 없애거나 모든 이점을 원시 보상으로 만든다(§4). 이것은 보고 있는 어느 곡선도 바꾸지 않은 채 실행이 최적화하는 것을 바꾼다.
- **결과(단일 연구):** 다중 도메인 장기 RL 실행에서도 “Diminish” 영역의 과제, 대부분 수학은 pass@1이 오르는데 pass@128은 흔히 하락하며, 경계 이득의 크기는 시작 체크포인트 pass@128과 음의 상관을 보인다([[prorl]] §4.1–4.2, Figs. 3–4). **방향으로 복제됨:** [[rlvr-beyond-base-model]]은 시험한 벤치마크에서 큰 k일 때 base 모델이 RLVR 모델과 같거나 더 낫다고 보고하며, RLVR을 base 모델이 이미 샘플링할 수 있었던 경로의 확률을 올리는 것으로 해석한다.
- 보상 shaping 항도 자체 방식으로 좁힌다. overlong 패널티(§9)는 긴 응답에 대한 그래디언트이며, 긴 컨텍스트 과제가 필요로 하는 행동에 대한 그래디언트다.

**(c) 이 단계에서 이를 측정하는 방법.**

- 양수 `test_freq`에서의 `data_source`별 검증 메트릭과, `train_files`에 없는 `data_source`를 하나 이상 둔다(§6).
- `mean@n`과 함께 커버리지 프록시인 `val_kwargs.n ≥ 8`에서의 `best@n`을 둔다. `mean@8`이 오르며 `best@8`이 떨어지는 것이 위에서 설명한 좁아짐 신호다. n = N에서 부트스트랩은 같은 N개 응답을 재표본화하므로, 외부 pass@k 수치와 수준을 비교하지 말고 단계별 추세를 비교한다.
- 고정된 범용 프롬프트 집합에서 기준 정책에 대한 평균 k3 KL을 검증 단계마다 기록한다(§6).
- 검증 메트릭이 움직이기 전에 도착하는 조기 지표로 `actor/entropy`, `pg_clipfrac` 대 `pg_clipfrac_lower`를 기록한다.
- 알려진 측정 오류: `train_files`에도 있는 검증 프롬프트는 모든 메트릭을 낙관적으로 만든다. `val_kwargs`의 `temperature: 0`은 `best@n`을 사용할 수 없게 하고 `mean@1`을 분산 없는 점 추정으로 만든다. 채점기 타임아웃은 학습과 검증 모두에서 0으로 등록된다(§5, 부정 샘플 절).

---

## 흔한 실수와 탐지 방법

| 실수 | 관찰 가능한 증상 | 점검 |
|---|---|---|
| `rollout.n`을 1로 둔 GRPO 설정 | `actor/pg_clipfrac_lower`가 0에 머무름; 0/1 보상에서 어떤 응답도 음의 이점을 받지 않음 | `actor_rollout_ref.rollout.n`의 해석된 설정을 출력하고 core_algos.py L315–317의 그룹 크기를 확인 |
| 어느 플래그도 설정하지 않고 KL 항을 기대함 | 메트릭 덤프에 `kl_loss`와 `actor/reward_kl_penalty`가 없음; 막는 것 없이 정책이 드리프트 | `actor.use_kl_loss`와 `algorithm.use_kl_in_reward`를 확인; 둘 다 기본으로 꺼짐 |
| `actor/ppo_kl`을 기준 정책으로부터의 드리프트로 읽음 | 일반 행동은 바뀌는데 곡선은 0 근처 | `actor/ppo_kl`은 기준이 아니라 이전 정책과 현재 정책 간 k1임(core_algos.py L1340) |
| 다른 세 설정은 기본값으로 둔 채 `loss_agg_mode`만 “Dr. GRPO”로 설정 | `norm_adv_by_std_in_grpo`가 여전히 True여서 학습 중 응답 길이가 증가 | 문서 조합은 상수 `loss_scale_factor`를 둔 `seq-mean-token-sum-norm` + `norm_adv_by_std_in_grpo: False` + `use_kl_loss: False` (grpo.md L59–62) |
| `loss_scale_factor` 없이 `seq-mean-token-sum-norm` 사용 | 데이터 변경 없이 배치의 최장 응답이 바뀔 때 손실/그래디언트-노름 곡선 이동 | 제수는 `loss_mask.shape[-1]`로 폴백함(core_algos.py L1190–1193); `loss_scale_factor`를 고정 정수로 설정 |
| 라우트 없는 새 `data_source` 값 | `NotImplementedError: Reward function is not implemented for data_source=…`로 실행 중단 | reward_score/__init__.py L107; 사용자 `compute_score`에 값을 등록 |
| 채점기 타임아웃이 조용히 0점 처리됨 | 모델 변화 없이 한 도메인의 정확도 하락 | `[NaiveRewardManager] compute_score exceeded` 줄을 카운터로 기록(naive.py L138–142) |
| `test_freq`를 −1로 둠 | 전체 실행에 검증 지점이 정확히 두 개 | `ppo_trainer.yaml` L201; ray_trainer.py L1727–1728의 게이트 |
| `val_kwargs`를 `temperature: 0, n: 1`로 둠 | 메트릭 덤프에 `best@n` 또는 `std@n` 키 없음 | `metric_utils.py` L989은 `n_resps > 1`일 때만 계산 |
| 긴 프롬프트가 응답을 조용히 줄임 | 경고 없는 잘린 완료; 긴 프롬프트 행의 보상 하락 | `vllm_async_server.py` L615–619의 `max_tokens` 클램프는 아무것도 로그하지 않음; 요청별 `max_tokens` 기록 |
| 헤드 수의 약수가 아닌 값으로 `ulysses_sequence_parallel_size`를 올림 | `AssertionError: num_heads (…) must be divisible by ulysses sequence size(…)` | `verl/utils/ulysses.py` L337–341 |
| 끝에서 채팅 템플릿을 다시 적용한 다중 턴 학습 | 다시 작성된 토큰 수만큼 마스크 어긋남; 학습이 수렴하지 않음 | token-in/token-out 경로 사용; `tokenization_sanity_check_mode: strict` (rollout.yaml L227) |

---

## 이해 점검

1. 한 실행에서 `actor/pg_clipfrac`이 1을 향해 상승하는데 `actor/pg_clipfrac_lower`는 0 근처에 머문다. 손실의 어느 분기가 선택되는지, 어느 이점 부호가 관련되는지, 어떤 단일 설정 값을 먼저 바꿀지 설명하라. 그 변경이 왜 이 메트릭에는 영향을 주고 다른 메트릭에는 주지 않는지도 설명하라.
2. 기본 `token-mean` 집계는 전역 유효 토큰 수로 나눈다. 이점이 0인 전부 오답 그룹이 같은 배치의 다른 올바르게 채점된 그룹에 적용되는 업데이트 크기를 어떻게 바꾸는지, 그리고 `filter_groups`가 왜 이를 해결하는지 설명하라.
3. 두 팀이 같은 `kl` 곡선을 0.0 근처로 보고한다. 한 팀은 `kl_penalty: kl`, 다른 팀은 `kl_loss_type: low_var_kl`을 썼다. δ = ±0.1의 토큰별 값을 사용해, 첫 팀은 왜 정책이 기준에 가깝다고 결론 낼 수 없고 둘째 팀은 왜 가능한지 설명하라.
4. 검증 덤프에 `val-core/math_dapo/acc/mean@8`은 있지만 어느 data source에도 `best@8` 키가 없다. 원인이 되는 설정과, 부재를 만드는 코드 경로를 제시하라.
5. `data.train_files`에 parquet 파일 하나를 이어 붙여 두 번째 도메인을 추가했다. 학습 보상은 오르지만 새 도메인 검증 정확도는 평평하다. 그 양상과 양립하는 서로 다른 원인 세 가지, 즉 보상 라우팅 하나, 이점 추정기 하나, 혼합 하나와 이들을 구분할 로그 줄을 제시하라.
6. `ulysses_sequence_parallel_size`를 올리면 더 긴 시퀀스를 학습할 수 있지만, 정책이 더 긴 응답을 생성할 수는 없는 이유를 학습 엔진과 롤아웃 엔진 사이 프로세스 경계를 언급하여 설명하라.
7. GiGPO의 단계 수준 이점은 환경 상태가 궤적 간에 되풀이되어야 한다. 이 요구가 ALFWorld와 WebShop에서는 만족되지만 단일 턴 수학 RL에서는 왜 만족되지 않는지, 그리고 어떤 verl 기반 스택이 이 방법을 채택할 수 있는지에 무엇을 뜻하는지 설명하라.
8. Pass@k 이점 추정기는 음의 이점을 부여하지 않는다. softmax 그래디언트 `∂ log p_y/∂z_j = 1[j = y] − p_j`를 사용하여, 이것이 확률 질량이 이동하는 위치를 어떻게 바꾸는지 설명하고 표준 GRPO 대비 엔트로피 효과를 예측하라.

---

## 연결

- **ch-54 — 롤아웃 인프라: 오프폴리시 데이터, 비동기 RL, 에이전트 환경** — §7과 §8이 한 저장소에서 구현한 것을 보여 주는 구조적 논증(오래됨, 부분 롤아웃, 환경-루프 내)을 다룬다.
- **ch-56 — OpenRLHF 내부 구조: PPO, DPO, Ray 오케스트레이션, 망각 제어** — §3과 다른 KL 배치 및 제어기 기본값을 가진 다른 코드베이스의 같은 대수다.
- **ch-38 — KL 제어 RLHF: PPO, InstructGPT, Alignment Tax** — §2와 §3이 구현하는 클리핑된 대리 목적과 기준 정책 KL 항을 다룬다.
- **ch-40 — 그룹 기준선 RL: RLOO, GRPO, Dr. GRPO, DAPO, GSPO** — §4가 레지스트리 항목과 설정 플래그로 보여 주는 알고리즘을 다룬다.
- **ch-43 — RL에서 엔트로피, 출력 다양성, KL 제어** — 부정 샘플 절의 진단값이 일찍 포착하려는 실패 양상을 다룬다.
- **ch-53 — 실습: 홀드아웃 스위트, 망각 보고서, 섭동 강건성을 갖춘 평가 하네스** — §6의 루프 내 검증에 대응하는 루프 밖 구성 요소다.
- **ch-57 — TRL 내부 구조: SFT, DPO, GRPO, 증류 트레이너** — 비교의 세 번째 스택이다.
- **ch-58 — 일반 능력을 측정하고 보호하기 위한 사후학습 스택 선택 및 계측** — 프레임워크별 읽기가 선택 및 계측 결정으로 이어지는 곳이다.

---

## 출처

- [[verl-ppo-loss]] — 커밋 753aed3의 `compute_policy_loss_vanilla` 및 `agg_loss`: §2의 손실 대수, clip 및 dual-clip 기본값, 집계 모드, `losses.py`의 엔트로피/KL 조합.
- [[verl-grpo]] — §4에서 쓴 `compute_grpo_outcome_advantage`, Dr. GRPO 스위치, Pass@k 추정기와 G = 4 계산 예.
- [[verl-rollout]] — `HFRollout`, 비동기 vLLM 서버, 가중치 동기화 동작, 배치 모드, 에이전트 루프 및 응답 마스크, §7과 §8의 완전 비동기 시간 표.
- [[entropy-logging-patterns]] — §3의 메트릭 이름과 KL 추정기 수식, 프레임워크 간 기본값 표.
- [[verl-dapo-recipe]] — verl의 DAPO 문서: Clip-Higher 값, 동적 샘플링 설정, 손실 집계 모드, §9의 overlong 보상 shaping, 재현 표.
- [[gigpo-verl-agent]] — 두 수준 이점, anchor-state 그룹화, verl-agent의 단계별 상호작용 패러다임, §8의 ALFWorld/WebShop/QA 수치.
- [[deepspeed-ulysses]] — all-to-all 시퀀스 병렬 메커니즘, O(N/P) 통신 분석, §9에서 보고한 처리량 수치.
- [[prorl]] — 다중 도메인 verl 실행: 다섯 도메인 데이터 혼합, 기준 리셋을 수반한 KL-in-loss, 검증 혼합, 일반화 관점의 폭 및 좁아짐 결과.
- [[dr-grpo]] — `loss_agg_mode` 및 `norm_adv_by_std_in_grpo` 설정이 다루는 길이 및 난이도 편향.
- [[grpo]] — §4가 구현하는 그룹 상대 이점.
- [[ppo]] — 대칭 ε를 쓰는 §2의 클리핑된 대리 목적.
- [[kl-control-rlhf]] — KL 정규화 목적과 §3의 추정기 배경.
- [[async-rollout]] — §7의 서버가 구현하는 비동기 롤아웃 설계.
- [[rlvr-beyond-base-model]] — §6의 커버리지 메트릭과 §4의 Pass@k 추정기 동기를 제공하는 pass@k 비판.
