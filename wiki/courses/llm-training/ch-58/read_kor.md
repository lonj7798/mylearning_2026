<!-- chapter: ch-58 -->

# 제58장 — 프레임워크 비교와 선택 기준

이 장은 TRL, OpenRLHF, verl을 같은 조건에서 비교하고, 실험 규모와 병목에 따라 선택하는 방법을 설명한다. 판단 기준은 기능 목록이 아니라 rollout 지연, learner 처리량, 메모리, 장애 복구, 로그 재현성이다.

## §1 공통 구조

세 프레임워크 모두 프롬프트로 응답을 생성하고, policy와 reference의 로그확률을 계산한 뒤 reward·advantage·policy loss를 계산해 업데이트한다. DPO는 선호/비선호 응답을 함께 forward하고, PPO·GRPO는 rollout을 다시 평가한다. 프레임워크를 바꿀 때 verifier, prompt split, held-out 평가도 다시 확인해야 한다.

## §2 기능과 적합한 상황

| 항목 | TRL | OpenRLHF | verl |
|---|---|---|---|
| 적합한 용도 | 소규모 실험, HF 통합 | Ray 기반 분산 PPO/DPO | actor와 learner를 분리한 대규모 RL |
| rollout | trainer generate/vLLM 연동 | Ray actor·inference worker | 별도 rollout worker, vLLM·HF 경로 |
| 시작 난이도 | 낮음 | 중간 | 높음 |
| 핵심 병목 | 단일 trainer | Ray worker 조정 | 큐, 통신, policy 지연 |

1~4 GPU에서 verifier와 loss를 검증할 때는 TRL이 가장 짧다. PPO critic과 Ray actor pool이 필요하면 OpenRLHF를 사용한다. 생성이 learner보다 느리거나 여러 inference worker를 독립적으로 늘려야 하면 verl을 선택한다.

## §3 반드시 남길 로그

`reward_mean`, `reward_std`, 정답률, group의 정답·오답 수, policy/reference 로그확률 차이, KL, entropy, clip fraction, advantage 통계, 응답 토큰 수를 prompt ID와 step에 연결해 저장한다. rollout 생성 시간, learner update 시간, 큐 대기 시간, GPU peak memory도 기록한다. checkpoint·데이터·verifier 버전과 seed가 없으면 같은 결과를 재현할 수 없다. reward가 올라도 pass@k가 내려가거나 오답 길이가 늘면 성공으로 기록하지 않는다.

## §4 성능 범위

처리량은 모델 크기뿐 아니라 응답 길이, group 크기 `G`, update epoch, reference 계산, 통신, verifier 실행 시간으로 결정된다. 짧은 응답과 작은 `G`에서는 단일 TRL 프로세스가 충분하다. 긴 reasoning trace에서는 생성이 병목이 되므로 rollout worker를 분리한다. learner만 늘리고 생성 큐가 비어 있으면 tokens/sec는 증가하지 않는다.

## §4.5 세 실행 프로필

* **검증 run:** TRL GRPO/DPO, 1~4 GPU, 작은 prompt set과 짧은 max length로 loss와 verifier를 확인한다.
* **분산 PPO:** OpenRLHF Ray actor pool에서 KL controller, reward shaping, worker 재시작과 checkpoint 복구를 확인한다.
* **대규모 reasoning:** verl에서 vLLM rollout worker와 learner를 분리하고 queue length, policy version, stale data 비율을 모니터링한다.

## §5 선택 순서

오프라인 선호 데이터만 있으면 TRL DPO로 시작한다. 온라인 verifier가 필요하지만 단일 노드면 TRL GRPO로 loss를 검증한다. critic·Ray actor가 필요하면 OpenRLHF를 사용하고, rollout을 learner와 독립적으로 확장해야 하면 verl로 간다. 전환 전후에 같은 checkpoint, prompt, verifier, seed로 reward·KL·entropy·길이·tokens/sec·peak memory를 비교한다.

## §6 단계적 전환 기준

TRL trainer가 rollout과 update를 번갈아 실행해 GPU를 놀리거나, actor를 늘려도 learner가 결과를 받지 못하거나, 긴 응답 때문에 generation과 learner 메모리를 동시에 확보할 수 없을 때 분산 프레임워크로 전환한다. OpenRLHF에서 verl로 옮길 때는 Ray 조정 비용보다 actor/learner 큐 분리로 얻는 처리량이 큰지 측정한다.

## §6.5 피해야 할 설정

검증하지 않은 verifier로 대규모 run을 시작하지 않는다. `max_prompt_length`, `max_completion_length`, `temperature`, `clip range`, `KL coefficient`, `num_generations`, batch, seed를 명시한다. rollout과 learner의 policy version이 오래 어긋나면 stale data가 되므로 queue age를 기록한다.

## §7–§8 적용 범위

동반 시각화에서 rollout 시간·learner 시간을 바꿔 빈 큐와 대기 구간을 찾는다. 특정 프레임워크가 항상 최고인 것은 아니다. backend, batch, 통신, verifier 비용이 다르므로 동일 조건의 작은 실험으로 최종 선택한다.

## 연결

verl은 ch-55, OpenRLHF는 ch-56, TRL은 ch-57, rollout/replay는 ch-54, entropy·KL은 ch-43, 평가·회귀 추적은 ch-47·ch-51을 참조한다.
