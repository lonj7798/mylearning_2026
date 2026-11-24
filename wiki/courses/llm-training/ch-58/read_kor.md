# 챕터 58 — 범용 능력을 측정하고 보호하기 위한 post-training stack 선택과 계측

> **핵심 통찰.** verl, OpenRLHF, TRL의 clipped policy-gradient 대수는 거의 동일하므로 stack 선택은 algorithm 선택이 아니다. 다른 것은 각 stack이 지원하는 operating range(GPU 수, rollout이 얼마나 stale해질 수 있는지, loop 안에서 environment를 실행하는지)와 계측이다. 즉 run 중 general capability가 떨어지고 있음을 stack이 알릴 수 있는가다. 계측 격차는 측정할 수 있다. decoupled objective 없이 maximum staleness `η = 4`로 한 AReaL ablation은 synchronous oracle의 AIME24 42.0, MATH-500 89.2에 비해 AIME24 23.3, MATH-500 66.9를 기록했다. decoupled objective를 쓰면 같은 staleness에서 42.2와 89.5였다([[areal-async-rl]] Table 2). training reward만 기록한 run은 두 구성을 구별하지 못했을 것이다.
>
> **지침.** run이 일반적으로 유능한 model을 만들어야 한다면 throughput 비교 전에 stack에 세 가지를 요구한다. reward가 target하지 않는 held-out task의 periodic in-loop evaluation, 기록되는 rollout-versus-training log-probability diagnostic, 둘 이상의 domain을 route할 수 있는 reward layer다. 이 세 가지를 두 후보 이상이 갖추면 operating range로 고른다. single node와 environment 없음이면 TRL, Ray-managed actor/critic split이 있는 multi-node면 OpenRLHF, Megatron parallelism·MoE routing replay·tool agent loop가 필요한 multi-node면 verl을 쓴다. 수천 concurrent episode의 environment를 loop에 넣는 run은 single-turn trainer를 확장하지 말고 해당 용도로 만든 agentic stack(SkyRL-Agent, verl-agent, AReaL, prime-rl)을 고른다. 이 system들은 scheduling gain을 측정으로 보고하기 때문이다([[skyrl-agent]] §3.2; [[areal-async-rl]] §7.2).

---

## 이 장이 범용 모델에 중요한 이유

이 장은 [[verl-ppo-loss]], [[openrlhf-ppo]], [[trl-grpo]]를 각각 읽은 뒤 RL-infrastructure track의 끝에 있다. run을 시작하기 전에 답해야 할 질문, 즉 run을 신뢰할 수 있도록 어떤 stack을 어떤 설정으로 쓸 것인가를 다룬다.

일반성 때문에 이 질문이 어려운 이유는 RL이 주어진 reward를 최적화하고, reward는 보존하려는 capability보다 거의 항상 좁기 때문이다. math verifier는 instruction following, tool use, refusal behaviour, output diversity에 관해 아무것도 말하지 않는다. stack이 training reward 외에는 평가할 수 없다면 reward domain 밖 capability loss는 run이 끝날 때까지 보이지 않는다. 세 가지 별도 failure source가 이 invisibility를 그럴듯하게 만든다.

1. **Reward-directed narrowing.** policy는 reward에서 개선하면서 다른 곳의 능력을 잃을 수 있다. 탐지하려면 reward가 target하지 않는 task를 평가해야 한다.
2. **Off-policy drift.** rollout은 inference engine이 만들고 gradient는 training backend가 만든다. shared parameter 아래에서도 둘은 token probability에 불일치하며, nominally on-policy RL을 off-policy RL로 바꾼다([[thinkingmachines-defeating-nondeterminism]], RL section). correction이 없으면 published run에서 reward collapse와 entropy collapse가 발생했다.
3. **신뢰할 수 없는 측정.** generation이 reproducible하지 않으면 breadth의 before/after 비교는 model change와 sampling noise를 섞는다. Qwen3-235B-A22B-Instruct-2507에 temperature 0으로 prompt 하나의 completion 1000개를 sample했을 때 default vLLM kernel에서는 서로 다른 completion 80개가 나왔고, batch-invariant kernel에서는 같은 completion 하나가 1000회 모두 반복됐다([[thinkingmachines-defeating-nondeterminism]], “Experiments”: completion 1000개, 각 1000 token).

따라서 이 장은 stack selection을 먼저 instrumentation 문제(§2), 다음 capability matrix(§3), 마지막 throughput 문제(§4)로 다룬다.

---

## §1 “stack”이란 무엇이며 모두에 공통인 것은 무엇인가

**정의.** post-training stack은 네 layer다. gradient를 계산하는 *trainer*, sample을 생성하는 *rollout engine*, 그것을 score하는 *reward 또는 environment layer*, training과 분리된 task에서 model을 측정하는 *evaluation layer*다. 앞의 세 layer는 framework README가 설명하는 대상이다. 넷째가 generality를 관측할 수 있는지 결정한다.

**공통인 것.** 이 장에서 읽은 pinned commit — verl 753aed3(2026-09-14), OpenRLHF 64c1cc4(2026-04-19, `main` b117b2b를 2026-09-14에 확인), TRL a08e713(2026-04-21, `main` a04ffd3를 2026-09-14에 확인) — 에서 surrogate objective는 셋 모두 같은 객체다. 각각 clipped ratio와 advantage의 곱을 최소화한다.

```
L = − min( r_t · A_t ,  clip(r_t, 1 − ε_low, 1 + ε_high) · A_t )
```

- `r_t` = exp(log π_θ(a_t) − log π_old(a_t)). current policy와 batch에 log-probability가 기록된 policy 사이의 token별 probability ratio다.
- `A_t` = token t에 배정한 advantage다.
- `ε_low`, `ε_high` = lower·upper clip bound다. 셋 모두에서 따로 설정할 수 있다([[verl-ppo-loss]]; [[openrlhf-ppo]] loss.py L136–148; [[trl-grpo]] grpo_trainer.py L2507–2515).

Group-relative advantage도 constant만 다를 뿐 같은 구성이다. verl은 `std + 1e-6`으로 나눈다(core_algos.py L272, L326). TRL은 `std + 1e-4`으로 나눈다(grpo_trainer.py L2127–2149). OpenRLHF `group_norm`은 `std + 1e-9`로 나눈다(experience_maker.py L264–270). reward가 모두 같은 group은 세 경우 모두 advantage 0을 얻는다.

**함의.** loss 때문에 stack을 고르지 말아야 한다. framework만 다른 두 run도 outcome은 달라질 수 있지만 원인은 surrogate가 아니라 rollout path, KL placement, aggregation denominator, evaluation schedule에 있다.

## §2 일반성 계측: 무엇보다 먼저 확인할 네 hook

### §2.1 reward가 target하지 않는 task의 in-loop evaluation

**측정 가능한 문제.** RL은 수백 step을 실행한다. 기록한 scalar가 reward·entropy·KL·clip fraction뿐이면 target하지 않은 capability 하락은 run 뒤에야 탐지된다. stack이 (a) schedule에 따라 held-out set에서 generation을 실행하고, (b) training reward와 다른 scorer로 score하며, (c) 그 score로 checkpoint를 선택할 수 있을 때 generality measurement를 지원한다.

**각 stack이 제공하는 것.**

- **verl.** `trainer.val_before_train` default는 `True`, `trainer.test_freq` default는 `-1`(periodic validation 없음), `trainer.log_val_generations` default는 `0`, `trainer.validation_data_dir` default는 `null`이다(`verl/trainer/config/ppo_trainer.yaml` L154–201, 753aed3). `trainer.checkpoint_callback_class` hook(default `null`, 같은 파일)은 checkpoint save 뒤 실행되어 external evaluation job을 붙일 자리다. 따라서 default verl은 training 전 한 번 측정하고 `test_freq`를 설정하지 않으면 다시 측정하지 않는다.
- **OpenRLHF.** `--eval.steps` default `-1`, `--eval.dataset` default `None`, `--eval.temperature` 0.6, `--eval.n_samples_per_prompt` 4다(`openrlhf/cli/train_ppo_ray.py` L286, L535–540, 64c1cc4). planning 전 알아야 할 constraint는 `if args.eval.dataset: assert args.reward.remote_url`(L672–673)이다. in-loop evaluation에는 remote reward-model service가 필요하다. `--ckpt.best_metric_key`는 `eval_default_pass1` 같은 eval metric으로 checkpoint를 고른다(L297–300).
- **TRL.** `GRPOConfig`는 `transformers.TrainingArguments`에서 `eval_strategy`, `per_device_eval_batch_size`를 상속하고, `num_generations_eval`을 추가한다(default `None`이면 `num_generations`으로 fallback). global eval batch size가 eval generation 수로 나누어떨어지는지도 검증한다(`grpo_config.py` L59, L399, L919–928, a08e713; `main`에서도 L1105–1115).

**실제로 쓰인 근거.** INTELLECT-3 report는 RL run 중 AIME25·AIME24·LiveCodeBench·HLE·GPQA의 model performance를 15-step 간격으로 그리고, run이 끝날 때도 score가 상승 중이었다고 말한다(§3.3, “Online Evaluation”, Figure 9; [[intellect-3]]). benchmark 다섯 개 중 적어도 HLE·GPQA 둘은 RL reward mixture 대부분의 domain 밖이다. **결과(단일 연구)**이며 run 하나, seed 보고 없음이다.

**조건과 한계.** 세 general framework 어느 것도 broad evaluation suite를 ship하지 않는다. 각각 hook만 주고 suite는 user가 제공한다. 비용은 실제로 든다. prompt당 유용한 sample 수로 다섯 benchmark suite를 generation하면 같은 inference capacity를 rollout과 경쟁한다.

**범용 모델에 대한 함의.** `test_freq` / `--eval.steps` / `eval_strategy`를 optional이 아닌 required configuration으로 보고 reward에 없는 task family를 하나 이상 포함한 set에 연결한다.

### §2.2 재현 가능한 평가: rollout-versus-training log-probability gap

**정의.** sampler(vLLM, SGLang)와 learner(FSDP, Megatron, DeepSpeed)는 같은 parameter에서 같은 token probability를 다른 kernel, 다른 parallelism, 때로 다른 precision으로 계산한다. 결과 차이가 rollout–training mismatch다.

**측정된 문제.** Qwen2.5-32B DAPO run에서 vLLM sampler와 FSDP learner의 token별 최대 probability 차이는 1.0까지 도달했다. sampler는 probability 1, learner는 0을 준 token이 있었다([[rollout-training-mismatch-tis]] Fig. 1). 더 작은 GSM8K 예시의 최대 차이는 약 0.4였다. sample에 실제 쓰인 probability 반환, fp32 `lm_head`라는 두 vLLM patch 뒤에도 gap은 지속됐다.

**메커니즘.** policy-gradient estimator는 sampling distribution과 differentiated distribution이 같다고 가정한다. 다르면 REINFORCE term은 off-policy인 `E_{a∼π_sampler}[R(a)·∇ log π_learner(a)]`가 된다. Truncated importance sampling(TIS)은 bounded weight로 correction을 복구한다.

```
E_{a∼π_sampler(θ)} [ min( π_learner(a,θ) / π_sampler(a,θ), C ) · R(a) · ∇_θ log π_learner(a,θ) ]
```

- `π_sampler(a,θ)`, `π_learner(a,θ)`는 같은 parameter θ 아래 inference engine과 training backend가 action `a`에 배정한 probability다.
- `R(a)`는 sample reward다.
- `C`는 truncation bound다.

**계산 예시(variance, blog callout).** learner/sampler ratio가 16인 token 하나를 잡는다. 이 token의 gradient noise는 적용한 weight의 제곱으로 scale된다. untruncated importance sampling은 `16² = 256`, `C = 2` TIS는 `2² = 4`, `C = 8` TIS는 `8² = 64`다. truncation은 bias(weight가 실제값보다 낮게 cap됨)를 대가로 `C = 2`에서 `256/4 = 64×` variance reduction을 얻는다.

**근거.** 유일한 차이가 truncated ratio인 DAPO-32B run은 실행한 250 step 전반에서 더 높은 downstream performance를 보였다([[rollout-training-mismatch-tis]] Fig. 1; OPT 2025 workshop version은 benchmark를 AIME 2024라고 이름 붙인다). Qwen2.5-0.5B GSM8K에서 INT8 rollout은 BF16 rollout보다 PPO를 떨어뜨렸고, TIS는 INT8 run을 비슷한 accuracy로 되돌렸다(Fig. 2). gap이 원래 작았던 DeepSeek-R1-Distill-Qwen-1.5B(BF16 양쪽)에서는 TIS gain이 없었다(Fig. 3). **결과(단일 연구)**, released script가 있는 practitioner evidence다.

Thinking Machines report도 kernel 관점에서 같은 결론에 도달한다. Qwen2.5-VL-Instruct-8B에서 초기화한 Bigmath RLVR, maximum rollout length 4096에서 off-policy correction이 없으면 training 중 reward가 collapse하고 step 318 부근에 loss spike가 생겼다. importance weighting을 쓰면 sampler–trainer KL은 occasional spike가 있어도 약 0.001에 머물렀고, sampler와 trainer kernel이 bitwise-identical하면 KL은 0에서 평평했다([[thinkingmachines-defeating-nondeterminism]] RL section). 효과 방향은 독립 source 둘에서 **반복 확인됨(Replicated)**이다.

**각 stack이 노출하는 기능.**

- **verl:** `algorithm.rollout_correction` config group에는 `rollout_is: null`(default disabled), `rollout_is_threshold: 2.0`, three-policy decoupled formulation과 two-policy formulation 사이의 `bypass_mode`, rejection-sampling variant가 있다(`verl/trainer/config/algorithm/rollout_correction.yaml`, 753aed3). `rollout.calculate_log_probs` default는 `True`다(`rollout.yaml` L237).
- **OpenRLHF:** `--algo.advantage.is_correction_enable`은 default off, band `[0.5, 5.0]`, type `tis`, `icepop`, `seq-mask-tis`이며 `vllm_kl`은 masked mean `(rollout log-prob − trainer log-prob)` metric이다([[openrlhf-ppo]] loss.py L150–173; `train_ppo_ray.py` L257–271). `main` b117b2b에서는 flag가 `is_correction_level {off, token, seq}`, `mode {mask, clip}`, `gating {ratio, binary_kl, tv}`가 된다.
- **TRL:** `GRPOTrainer`에서 `use_vllm=True`이면 IS correction이 default on이고 mode `sequence_mask`, cap 3.0이다(`grpo_config.py` L792–819, a08e713). experimental async trainer는 model `dtype` default가 `"float32"`이고 vLLM server dtype이 다르면 시작 때 warning을 log한다. docstring은 이유를 training–inference mismatch로 적는다(`trl/experimental/async_grpo/async_grpo_config.py` L37–44, a04ffd3).

**범용 모델에 대한 함의.** breadth에서 checkpoint 두 개를 비교하려면 generation path가 같고 mismatch를 correction했거나 측정했어야 한다. step 1부터 mismatch metric(`vllm_kl` 또는 verl rollout-correction ratio)을 log하고, factor analysis가 셋 모두 mismatch를 바꾼다고 보이므로 모든 evaluation number 옆에 sampler dtype, tensor-parallel size, sequence-parallel size를 기록한다([[rollout-training-mismatch-tis]] Figs. 7–8).

### §2.3 Multi-domain reward routing

범용 모델은 scorer 하나 이상을 상대로 학습한다. stack은 hand-written glue 없이 여럿을 붙이고 합칠 수 있어야 한다.

- **TRL**은 `reward_funcs` list, `reward_weights`(default `None`, 각각 weight 1.0), `multi_objective_aggregation`(default `"sum_then_normalize"`, `scale_rewards` 전에 weighted reward를 합침)을 받는다(`grpo_config.py` L205–212, L678–690, a08e713). 셋 중 가장 직접적인 multi-domain path다.
- **OpenRLHF**는 service boundary인 `--reward.remote_url`를 통해 route한다. §2.1 evaluation constraint 때문에 in-loop evaluation을 쓰면 remote reward model이 required path다.
- **verl**은 `reward` config group과 legacy reward implementation group을 가진다(`ppo_trainer.yaml` defaults list L34–39, 753aed3). routing detail은 ch-55 주제다.
- **prime-rl**(INTELLECT-3 stack)은 reward function 대신 environment를 unit으로 삼는다. orchestrator는 “verifiers environments를 사용해 multi-turn rollout generation과 scoring을 abstract하고 Environments Hub의 어떤 environment든 training loop에 plug할 수 있게 한다.” report는 같은 environment interface가 synthetic data generation, supervised fine-tuning, RL, evaluation을 덮는다고 한다([[intellect-3]] §2.1, §2.1.1). 이 property가 training mixture와 evaluation suite를 비교 가능한 object로 만든다.

**함의.** multi-domain reward mixture 계획이라면 domain identity가 logging까지 살아남는 stack을 선호하여 domain별 reward를 따로 그린다. 평균 reward 하나는 한 domain이 regression하고 다른 domain이 improve하는 상황을 숨긴다.

### §2.4 Forgetting control: KL placement, LoRA, checkpoint hook

framework가 capability loss에 맞서 실제로 제공하는 knob 셋은 reference policy KL term, low-rank adaptation, checkpoint selection이다. default가 다르고 대부분 run은 default를 쓴다.

| 제어 | verl @753aed3 | OpenRLHF @64c1cc4 | TRL @a08e713 |
|---|---|---|---|
| reference KL, reward 안 | `algorithm.use_kl_in_reward`, default off | default path, `FixedKLController` `init_coef` 0.01; `--algo.kl.target` 설정 시에만 `AdaptiveKLController`(default `None`) | PPO path만: reward에서 `kl_coef` 0.05를 뺌([[trl-ppo]]) |
| reference KL, loss 안 | `actor.use_kl_loss`(default `false`; docs는 GRPO에 `True`라고 함), `kl_loss_coef` 0.001, `kl_loss_type` `low_var_kl`(`actor.yaml` L103–116) | `--algo.kl.use_loss`, actor loss에 `kl_loss * kl_ctl` 추가(ppo_actor.py L296–314) | GRPO: loss에 β·k3 추가, β default **0.0**, β = 0이면 reference model도 만들지 않음(grpo_trainer.py L2493–2497, L650–653) |
| loss의 entropy term | `entropy_coeff: 0`(`actor.yaml` L93) | `--actor.entropy_coef` 미설정 → logging만 | entropy term 없음 |
| LoRA | rollout path에서 LoRA adapter 지원(`lora_as_adapter`, sleep level 1에서 adapter-only weight sync)([[verl-rollout]]) | `--ds.lora.rank` default 0, `--ds.lora.alpha` 16, `target_modules` `all-linear`(train_ppo_ray.py L363–368) | `transformers` trainer를 통한 PEFT integration |
| RL 중 pretraining-loss mixing | 읽은 PPO trainer config에 없음 | 64c1cc4 `cli/train_ppo_ray.py`에 그런 flag 없음 | `GRPOConfig`에 없음 |
| checkpoint selection / averaging / merging hook | `trainer.checkpoint_callback_class`(default `null`); trainer 안 merge utility 없음 | `--ckpt.best_metric_key`; `openrlhf/cli/lora_combiner.py`는 loop가 부르지 않는 별도 command | `transformers` callback, `load_best_model_at_end`; a04ffd3의 `trl/experimental/merge_model_callback.py`는 mergekit으로 checkpoint와 target model을 merge |

두 관찰이 따른다. 첫째, GRPO-style RL default configuration에는 TRL의 KL term이 **없고**(β = 0.0), verl에도 없다(`use_kl_loss`, `use_kl_in_reward` 둘 다 off). reference model은 optional이다. forgetting control로 가장 자주 언급되는 control은 켜지기 전에는 off다.

둘째, training loop가 호출하는 merging hook을 ship하는 것은 셋 중 정확히 하나다. TRL `MergeModelCallback`(`trl/experimental/merge_model_callback.py`, a04ffd3)은 `transformers.TrainerCallback`이다. saved checkpoint를 target model과 mergekit으로 merge한다. target model default는 trained model config에서 읽은 model id다. `MergeConfig` method는 `linear`(default; policy weight 0.5, target weight 0.5), `ties`, `dare_ties`, `slerp`다. `merge_at_every_checkpoint` default는 `False`이므로 default merge는 매 save가 아니라 `on_train_end`에서 한 번 실행된다. `mergekit` extra가 설치되지 않으면 callback은 construction에서 `ImportError`를 낸다. 읽은 file에는 verl 동등물이 없고 OpenRLHF `openrlhf/cli/lora_combiner.py`(b117b2b)는 별도 command-line utility다. merge가 무엇을 이루며 어떤 weight를 써야 하는지는 ch-58a 주제다.

---

## §3 날짜가 있는 commit에 고정한 capability matrix

아래 각 cell은 column header에 적힌 commit에서 읽었거나 “not checked”로 표시했다. “Not present”는 읽은 file에서 그 commit에 flag나 module이 없다는 뜻일 뿐, repository 전체나 이후 version에 대한 주장이 아니다.

| # | 차원 | verl @753aed3(2026-09-14) | OpenRLHF @64c1cc4(2026-04-19), `main` b117b2b | TRL @a08e713(2026-04-21), `main` a04ffd3 |
|---:|---|---|---|---|
| 1 | asymmetric bound가 있는 clipped surrogate | 예([[verl-ppo-loss]]) | 예, `clip_eps_low`/`clip_eps_high`, optional `dual_clip`(default `None`) | 예, `epsilon`/`epsilon_high`, optional `delta` |
| 2 | GRPO family | Pass@k·GDPO를 포함한 registered estimator 14개의 `@register_adv_est`([[verl-grpo]]) | `group_norm`, `dr_grpo`, `rloo`, `reinforce_baseline`, `gae`, `reinforce` | `loss_type` 8개, default `dapo`([[trl-grpo]]) |
| 3 | actor–critic PPO | 예(`adv_estimator=gae`, critic enabled) | 예, Ray actor/critic split | **mainline에서 제거됨**: PR #7020이 2026-09-04에 `trl/experimental/ppo/`를 삭제했고 v1.13.0으로 release([[trl-ppo]]) |
| 4 | mainline trainer set | RL trainer | b117b2b에서 `openrlhf/trainer/`는 `ppo_trainer.py`, `ppo_trainer_async.py`, `dpo_trainer.py`, `rm_trainer.py`, `sft_trainer.py` 보유 | a04ffd3에서 `trl/trainer/`는 DPO, GRPO, KTO, RLOO, SFT, reward, distillation 보유 |
| 5 | asynchronous rollout | partial rollout이 있는 `fully_async_policy`([[verl-rollout]]) | `PPOTrainerAsync`, queue size default 1, optional partial rollout([[async-rollout]]) | a04ffd3의 `trl/experimental/async_grpo/`, `max_staleness` default 4, `queue_maxsize` 1024, `weight_sync_steps` 1 |
| 6 | rollout–training IS correction | `algorithm.rollout_correction`, default off, threshold 2.0 | `is_correction_enable`, default off, band [0.5, 5.0] | `use_vllm=True`이면 default on, `sequence_mask`, cap 3.0 |
| 7 | KL placement option | reward 또는 loss, 둘 다 default off | reward(default) 또는 loss | GRPO: loss만, β default 0.0 |
| 8 | in-loop evaluation | `test_freq`(default −1), `val_before_train`(default True) | `--eval.steps`(default −1), `--reward.remote_url` 필요 | `eval_strategy`, `num_generations_eval` |
| 9 | multi-domain reward routing | reward config group(ch-55) | remote reward service | `reward_funcs` list + `reward_weights` + `multi_objective_aggregation` |
| 10 | LoRA | 예(adapter-only weight sync) | `--ds.lora.rank`, default 0 | PEFT |
| 11 | long-sequence parallelism | flash-rl DAPO script에서 Ulysses sequence parallelism(learner SP 8)([[rollout-training-mismatch-tis]] recipe ledger) | `--ds.ring_attn_size`, default 1, `--ds.packing_samples` 자동 활성화(L506, L638–641) | async trainer가 Σ Lᵢ²로 균형을 맞춘 micro-batch packing |
| 12 | in-tree tool / environment loop | response mask가 있는 `ToolAgentLoop`([[verl-rollout]]) | `--train.agent_func_path` example script | `tools`, experimental `environment_factory`, `rollout_func`; async trainer에 OpenEnv `HarnessAdapter` bridge |
| 13 | MoE 처리 | `enable_rollout_routing_replay`로 routed-expert replay(vLLM ≥ 0.22.0 필요) | DPO에서 MoE aux loss 보존([[openrlhf-dpo]]) | async GRPO config의 `router_aux_loss_coef` default 0.001 |
| 14 | in-tree distillation config | `ppo_trainer.yaml` defaults에 `distillation` config group | not checked | `trl/trainer/distillation_trainer.py`, experimental `gkd`, `async_distillation`, `server_distillation`, `minillm` |
| 15 | preference-optimization variant | RL trainer의 focus 아님 | cDPO smoothing·IPO·optional NLL term이 있는 DPO([[openrlhf-dpo]]) | mainline DPO·KTO; `trl/experimental/` 아래 `cpo`, `orpo`, `bco`, `nash_md`, `xpo`, `online_dpo`, `sdpo`, `tpo`([[trl-online-dpo]]) |
| 16 | entropy/KL metric 이름 | `actor/entropy`, `actor/ppo_kl`, `actor/reward_kl_penalty` | `PolicyLoss.forward`가 반환하는 `ppo_kl`, `vllm_kl`, `clip_ratio` | GRPO: entropy, sign-split clip metric, `frac_reward_zero_std`([[entropy-logging-patterns]]) |

세 행은 이 framework의 오래된 설명이 이끌 결정을 바꾼다. TRL actor-critic PPO는 released mainline에 더는 없으므로 TRL에서 PPO를 쓰려면 v1.13.0 이전 release를 pin해야 한다(행 3). TRL에는 explicit staleness bound가 있는 asynchronous GRPO trainer가 있다(행 5). 그리고 reference policy KL term은 verl·OpenRLHF에서 reward뿐 아니라 loss에도 놓을 수 있다(행 7).

**Interactive companion.** [framework-compare.html](figures/framework-compare.html)은 §7 decision procedure를 한 번에 한 question씩 실행하고, 지나간 path를 보여 주며, 도달한 leaf의 configuration note와 instrumentation checklist를 함께 출력한다. 따라서 throughput 비교 전에 후보 stack을 세 instrumentation question 모두에 대조할 수 있다.

---

## §4 source가 있는 측정만 쓴 performance와 staleness

어떤 framework도 algorithm, model size, response length, straggler distribution, hardware와 독립적인 throughput 수치를 갖지 않는다. 아래 모든 수치는 system 저자가 보고한 within-system 비교이며, matched condition에서 framework 둘을 head-to-head로 benchmark한 것은 하나도 아니다.

| system | 비교 | 결과 | 설정 |
|---|---|---|---|
| verl `fully_async_policy` | fully async 64:64 대 colocated sync | 400 step: 17 h 22 m 대 1 d 16 h 48 m(2.35×) | Qwen2.5-Math-7B, DAPO, max response 28K, H20 128장([[verl-rollout]], `fully_async_policy/README.md` L333–361) |
| AReaL | async 대 synchronous system | matched benchmark score에서 end-to-end training time 최대 2.77× 감소 | R1-Distilled-Qwen 1.5B–32B, GPU 8장 H800 node, 최소 16 node·32B에는 48 node([[areal-async-rl]] §7.1–7.2); 별도 scaling study는 512 GPU까지 linear scaling 보고(§7.3) |
| SkyRL-Agent | Async Pipeline 대 Async Batch(Bounded) | ≈1.55×, generation 중 GPU utilization ≈90% | batch 64 × rollout 8(trajectory 512), H100 2×8장([[skyrl-agent]] §3.2, Fig. 1b) |
| GiGPO / verl-agent | GiGPO addition 대 GRPO baseline | 362.83 s iteration 중 grouping 0.01 s, step-advantage 계산 0.53 s(paper는 iteration time의 <0.002%라 하지만 `0.54 / 362.83`은 0.15%) | ALFWorld에서 Qwen2.5-1.5B-Instruct([[gigpo-verl-agent]] Fig. 6) |
| OpenRLHF async | async 대 sync | **보고되지 않음** — repository나 framework paper arXiv 여섯 version 어디에도 throughput/convergence 측정 없음([[async-rollout]]) |

**staleness 차원을 계산해 보기.** async는 trainer가 더 오래된 policy가 만든 rollout을 소비하게 하여 throughput을 산다. 얼마나 오래됐는지는 configurable bound이며, 그 bound는 objective와 상호작용한다.

AReaL은 parameter η로 staleness를 bound하고, data를 sample한 behaviour policy와 trust-region anchor인 proximal policy를 분리하는 decoupled PPO objective를 쓴다. 따라서 batch 하나에 여러 policy version이 섞일 수 있다([[areal-async-rl]] §5.1). Table 2는 1.5B model에서 objective 유무로 η를 바꾼다. 일곱 행 중 다섯을 여기 재현한다. full sweep은 0/1/2/4/8/16/∞이며 AMC23과 AIME25도 보고한다.

| 최대 staleness η | objective 없이 AIME24 | objective 적용 AIME24 | objective 없이 MATH-500 | objective 적용 MATH-500 |
|---:|---:|---:|---:|---:|
| 0(oracle) | 42.0 | 42.0 | 89.2 | 89.2 |
| 1 | 41.8 | 42.1 | 89.9 | 89.8 |
| 4 | 23.3 | 42.2 | 66.9 | 89.5 |
| 8 | 35.7 | 41.0 | 87.8 | 89.2 |
| ∞ | 34.0 | 36.9 | 87.1 | 88.1 |

`η = 4`에서 decoupled objective가 없으면 oracle 대비 AIME24 18.7점, MATH-500 22.3점을 잃는다. objective가 있으면 둘 다 1점 안이다. objective가 있어도 `η = ∞`의 unbounded staleness는 AIME24 5.1점을 잃는다. η = 4 행은 η = 8 행과 monotone하지 않으므로, 이 sweep에서 특정 η의 하락 크기는 η의 smooth function이 아니다. paper default는 coding `η = 4`, math `η = 8`이다(§7.1). **결과(단일 연구)**, model size 하나와 math·code만 해당한다.

다른 system이 고른 bound와 비교한다. TRL async GRPO `max_staleness` default는 4(sample이 discard되기 전 뒤처질 수 있는 weight-update step), INTELLECT-3 `max_off_policy_steps = 8`은 excessively off-policy rollout을 제거하는 값([[intellect-3]] §3.3)이다. OpenRLHF `--train.async_queue_size` default는 1이며 README는 더 큰 queue와 partial rollout을 더 off-policy라고 설명하고 async가 “may affect convergence”한다고 하지만 수량화하지 않는다([[async-rollout]]).

**범용 모델에 대한 함의.** staleness는 stability parameter만이 아니다. AReaL 표는 infrastructure setting이 일으킨 general-capability regression을 직접 측정한 것이며, 각 η에서 benchmark 네 개를 평가했기에 보였다. async를 실행하는 stack은 configured bound만이 아니라 달성한 staleness distribution을 log해야 한다.

---

## §5 긴 sequence와 긴 rollout

long chain-of-thought RL과 long-context training은 generality와 관계되는 두 방식으로 stack에 부담을 준다.

**중단 없는 길이에 따라 mismatch가 커진다.** factor analysis에서 20K token cap response는 4K cap보다 sampler–learner maximum mismatch가 더 크며 mean mismatch는 비슷하다. 20K response의 첫 4K token은 독립적으로 생성한 4K response의 mismatch를 흔히 넘는다([[rollout-training-mismatch-tis]] Figs. 9–10). 이는 long-CoT run이 short-answer run보다 §2.2 correction을 더 필요로 하고, short response에서 검증한 correction은 다시 측정하지 않으면 transfer하지 않는다는 뜻이다.

**parallelism 선택이 mismatch를 바꾼다.** vLLM TP1 sampler와 FSDP SP1 learner에서 첫 512개 DAPO-Math-17k prompt 중 response 하나는 maximum mismatch > 0.5였다. TP2 sampler는 두 개를 냈고, learner에 Ulysses SP8을 추가하면 double-digit count가 나왔다(Fig. 7). sampler와 learner의 parallelism을 맞추면 count가 줄었다(Fig. 8). 따라서 long sequence의 표준 해법인 sequence parallelism은 evaluation을 reproducible하지 않게 만드는 numerical gap의 source이기도 하다.

**각 stack이 제공하는 것.** OpenRLHF에는 `--ds.ring_attn_size`(default 1)가 있고 1보다 크면 sample packing을 자동 활성화한다(`train_ppo_ray.py` L506–511, L638–641). published run에서 verl의 long-sequence path는 learner의 Ulysses sequence parallelism이며 sampler tensor-parallel size는 따로 정한다(flash-rl DAPO script: sampler TP 2, learner SP 8, GPU 8장 node 4개; [[rollout-training-mismatch-tis]] recipe ledger). INTELLECT-3는 tensor·context·expert parallelism을 쓰는 trainer로 65,536-token context에서 학습했고, in-flight weight updating을 쓸 때 그 길이의 step time 약 1500 s, 없으면 2× 이상으로 늘었다고 보고했다([[intellect-3]] §2.1.1, §3.3). TRL async GRPO는 token budget이 아니라 row length 제곱합으로 균형을 맞춘 micro-batch를 packing한다(`async_grpo_config.py` L113–116).

그보다 앞서 별도 clamp가 중요하다. verl vLLM server는 `max_possible_tokens = max_model_len − len(prompt_ids)`를 계산한 뒤 `max_tokens`를 warning 없이 `[1, max_possible_tokens]`로 clamp한다([[verl-rollout]], `vllm_async_server.py` L591–619). prompt가 자라는 long-context run은 실패하는 대신 더 짧은 response를 조용히 생성한다.

---

## §6 Agentic stack: loop 안의 environment

**environment가 loop에 들어올 때 달라지는 것.** single-turn RL step에는 generation이라는 latency distribution 하나가 있다. agentic step에는 environment setup, 여러 turn의 generation, reward calculation이라는 세 개가 있고 각각 long tail을 가지며 policy와 무관한 이유로 실패할 수 있다. §3 framework는 첫 사례를 중심으로 설계됐다.

**각각 측정된 주장이 있는 네 설계.**

1. **SkyRL-Agent**는 rollout을 runtime initialization, agent run, reward calculation으로 나누고 stage별 bounded queue를 두며 CPU-bound stage를 GPU generation과 overlap한다. 이 pipeline은 generation 중 GPU utilization ≈90%에서 bounded async batching보다 ≈1.55× 빠르다([[skyrl-agent]] §3.2). backend bridge는 모든 LLM call을 input token, output token, log-probability transition으로 기록한다. 이것이 inference–training mismatch correction을 가능하게 하고 re-tokenization drift를 피한다(§3.3).
2. **verl-agent / GiGPO**는 rollout을 step-wise로 만든다. 각 step prompt는 full history concatenation 대신 memory module이 만들므로 long episode에서도 context는 대체로 일정하고, environment는 gym-style interface로 parallel 실행된다([[gigpo-verl-agent]] App. A). training 결과에서 repeated environment state의 action을 trajectory 사이에서 사후 group화하고 그 group 안 normalize하는 anchor-state grouping은 Qwen2.5-1.5B-Instruct에서 ALFWorld overall success를 GRPO 72.8 ± 3.6에서 86.1 ± 4.7로, WebShop success rate를 56.8 ± 3.8에서 67.4 ± 4.5로 올렸다(3 seed 평균, Table 1). 이 둘은 standard-deviation normalization이 없는 GiGPO variant(`F_norm = 1`) 수치다. 그것이 있는 variant는 86.7 ± 1.7과 65.0 ± 3.2다. 설정은 group size 8, rollout당 16 group(concurrent environment 128), `ω = 1`, `γ = 0.95`, KL-loss coefficient 0.01(App. E.1), history length 2(App. E.2)다.
3. **AReaL**은 rollout worker를 interruptible하게 만든다. `update_weights` request가 진행 중 generation을 interrupt하면 worker는 KV cache를 버리고 new parameter를 load해 unfinished sequence decoding을 계속한다(§4.1). rollout controller는 staleness constraint를 위반할 generation request를 reject한다([[areal-async-rl]] §5.1). 이는 §4 η 표와 짝을 이루는 설계다.
4. **prime-rl**은 orchestrator(CPU), trainer(FSDP2), OpenAI-compatible vLLM inference pool을 분리하고 `/update_weights`, `/reload_weights` endpoint를 둔다. multi-turn rollout과 scoring을 verifiers environment 뒤로 abstract하여 어떤 Environments Hub entry도 loop에 plug한다([[intellect-3]] §2.1.1). INTELLECT-3는 H200 8장 node 60개를 약 1:3 training-to-inference node ratio(학습 16, inference 44)로 사용했다(§3.3).

**transition-level 대 trajectory-level.** [[agent-lightning]]은 반대 방향에서 같은 per-call transition 관점을 쓴다. LangChain, OpenAI Agents SDK, AutoGen으로 쓴 agent execution을 `(input, output, reward)` transition으로 바꾸고 episode의 모든 call에 episode return을 배정한다. paper는 Llama-3.2-3B-Instruct agent 세 개에서 train/test reward curve가 상승함을 보고하지만 numeric table과 general-capability evaluation이 없다. 그러므로 masked trajectory training 대비 이점은 거기서 측정되지 않았다.

**agentic RL의 generality 근거.** SkyRL-Agent SA-SWE-32B는 R2E-Gym SWE instance 4.5K개에서 RL만으로 학습한 뒤 domain 밖에서 평가됐다. Qwen3-32B 시작점 대비 Terminal-Bench 13.75 → 16.25, BrowseComp-Plus accuracy 18.1 → 19.4, WebArena 15.8 → 17.0이었다([[skyrl-agent]] Table 3). 각 model은 variance 없이 한 번만 평가했다. 같은 paper에 counter-case가 있다. computer-use agent는 training reward가 올랐지만 validation accuracy는 “little to no gain”이었고 training task는 benchmark의 subset이었다(§5.3). 양쪽 모두 **결과(단일 연구)**이며 single-domain agentic RL이 다른 tool domain으로 transfer하는지는 **열린 질문**이다.

**agentic stack이 추가해야 하는 계측.** environment crash rate를 task failure rate와 분리한다(crash episode는 negative example이 아니다). trajectory length와 turn-count는 mean이 아니라 distribution을 기록한다. environment 하나의 고장이 average를 움직이지 않게 environment별 reward를 기록한다. 그리고 staleness는 weight-update step으로 측정한다. SkyRL-Agent는 직접 이유를 보고한다. deep-research iteration 30 부근 tool-serving timeout이 rollout을 오염시켰고 recovery는 step 30–33이 걸렸다. 저자들이 abnormal trajectory mask 이유로 드는 사실이다(§5.1).

---

## §7 종료하는 decision procedure

procedure에는 question 일곱 개와 stack 네 개를 이름 붙이는 leaf node 여섯 개가 있다. TRL과 verl은 각각 environment 없는 run과 tools path에 한 번씩, 두 번 나온다. 모든 question은 두 답을 정의하고, 모든 답은 앞으로 가거나 leaf에 도달하며, 어느 답도 앞 question으로 되돌아가지 않는다. question은 순서대로 답한다.

```
Q1. run이 stateful environment를 loop에 넣는가? 즉 call 사이 state를 보존하는 external system에
    model call을 한 번 넘게 하는 episode가 있는가?
      NO  -> Q2
      YES -> Q5

Q2. node 하나에 training model, optimizer state, rollout engine이 모두 들어가는가?
      YES -> LEAF A
      NO  -> Q3

Q3. Megatron-style tensor/pipeline parallelism, MoE routed-expert replay,
    또는 weight sync 중 request별 abort가 있는 token-in/token-out rollout이 필요한가?
      YES -> LEAF C
      NO  -> Q4

Q4. 별도 value model을 가진 Ray-managed actor/critic split 또는 process boundary인 remote
    reward-model service가 필요한가?
      YES -> LEAF B
      NO  -> LEAF C

Q5. run에 concurrent environment instance가 약 100개보다 많거나,
    episode가 약 10 turn 또는 30K token보다 긴가?
      YES -> LEAF D
      NO  -> Q6

Q6. environment를 context에 text를 반환하는 function-call tool로 표현할 수 있는가?
      NO  -> LEAF D
      YES -> Q7

Q7. node 하나에 training model, optimizer state, rollout engine이 모두 들어가는가?
      YES -> LEAF A(tools path)
      NO  -> LEAF C(tools path)
```

Q5의 threshold 세 개는 single-turn trainer가 언제 작동을 멈추는지 측정한 값이 아니라 §6 published agentic run에서 읽은 값이다. GiGPO는 8개씩 16 group(concurrent environment 128)을 실행하고([[gigpo-verl-agent]] App. E.1), SkyRL-Agent SWE training은 episode를 50 turn과 32K token에서 cap한다([[skyrl-agent]] §4.2).

**Leaf.**

- **LEAF A — TRL.** critic-free RL에는 `GRPOTrainer`, preference optimization에는 `DPOTrainer`/`KTOTrainer`, rollout과 training을 overlap해야 하면 `trl/experimental/async_grpo/`를 쓴다. tools path는 `tools` / `environment_factory`, 또는 async trainer의 OpenEnv `HarnessAdapter` bridge를 쓴다. 사용할 수 없는 것은 mainline actor-critic PPO(PR #7020, 2026-09-04에 삭제)다.
- **LEAF B — OpenRLHF.** `PPOTrainerAsync` 또는 synchronous trainer, Ray actor/critic split이 있는 DeepSpeed ZeRO, reward service용 `--reward.remote_url`를 쓴다. §2.1 coupling에 주의한다. in-loop evaluation에는 그 remote service가 필요하다.
- **LEAF C — verl.** FSDP 또는 Megatron backend, request별 abort가 있는 async vLLM rollout, trainer를 건드리지 않고 algorithm을 바꾸는 `@register_adv_est`·`@register_policy_loss`, tools path용 `ToolAgentLoop`를 쓴다.
- **LEAF D — environment용 stack.** stage 간 long-tail episode latency가 bottleneck이면 SkyRL-Agent, history concatenation이 constraint가 될 만큼 episode가 길면 verl-agent, weight sync에서 rollout을 interrupt하고 staleness를 bound해야 하면 AReaL, common environment interface에서 environment를 가져오고 공유해야 하면 prime-rl을 쓴다.

**도달한 leaf에서 한 번 적용하는 instrumentation gate.** commit 전에 세 가지를 모두 확인한다.

1. reward가 target하지 않는 family를 하나 이상 포함한 held-out suite의 periodic in-loop evaluation과 benchmark별 series(§2.1).
2. 기록한 rollout-versus-training log-probability diagnostic과, 0이 아닐 때 쓸 수 있는 correction(§2.2).
3. 각 score의 domain identity를 유지하는 reward 또는 environment routing(§2.3).

leaf가 이 중 하나에서 실패하고 시간이 있는 동안 hook을 추가할 수 없으면, fixed order `A → B → C → D`에서 다음 candidate를 택해 gate를 한 번 더 적용한다. order가 고정되고 list가 네 entry이므로 procedure는 최대 네 번 gate evaluation 뒤 끝난다.

---

## 음성 샘플과 음성 피드백

**여기에 적용되는 의미.** 이 장은 instrumentation에 관한 것이므로 course standard 네 의미 중 두 가지를 다룬다. **gradient로서의 음성**(A < 0인 token 또는 sequence), **content로서의 음성**(agent context 안의 tool error·corrective hint)이다. stack 선택은 둘을 얼마나 잘 관측할 수 있는지를 정한다.

**메커니즘.** id가 y이고 logit이 z인 sampled token에서 logit j에 대한 log-probability gradient는 다음이다.

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

음의 advantage는 이 token update의 sign을 뒤집는다. y에서 probability mass를 빼고 현재 probability `p_j`에 비례해 다른 entry에 나누어 준다. 따라서 mass는 이미 가장 likely한 alternative에 집중하며, 그 답도 틀릴 수 있다. 이를 진단하려면 aggregate가 아니라 advantage sign별 statistic이 필요하다.

**각 stack이 보이게 하는 것.**

- **TRL GRPO**는 clip metric을 sign별로 나눈다. low-clip metric은 A < 0인 곳의 `r < 1 − ε_low`만, high-clip metric은 A > 0인 곳의 `r > 1 + ε_high`만 센다(grpo_trainer.py L2589–2590). 또 prompt group reward variance가 0이라 policy term에 기여하지 않은 sample 비율 `frac_reward_zero_std`를 log한다([[trl-grpo]]).
- **OpenRLHF**는 양 sign의 clipped token을 합쳐 세고 dual-clip activation은 세지 않는 `clip_ratio` 하나를 반환한다([[openrlhf-ppo]] loss.py L180). sign-split diagnostic은 추가해야 한다.
- **verl**은 `actor/pg_clipfrac`와 별도 `actor/pg_clipfrac_lower`를 log한다(core_algos.py L1372–1374). GRPO estimator는 zero-advantage group을 제거하지 않고 group-normalized score에 `response_mask`를 곱한다(L1165–1170). 따라서 default `loss_agg_mode: token-mean`(L329; actor.yaml L86)에서는 그 token도 denominator에 들어간다([[verl-grpo]]).

**계산 예시(zero-variance group의 효과).** binary reward, `G = 4`, 정답 response 하나인 group을 잡으면 verl estimator는 `μ = 0.25`, `σ = 0.5`, 정답에 +1.5, 각 오답에 −0.5를 준다([[verl-grpo]]). 넷 모두 실패하면 `σ = 0`이고 모든 advantage는 `0/(0 + 1e-6) = 0`이다. prompt는 gradient에 기여하지 않지만 token은 denominator에 들어가므로 batch 나머지의 effective learning rate가 떨어진다. `frac_reward_zero_std`나 동등 metric이 없으면 이유 없이 평평해지는 training curve가 observable symptom이다.

**stack이 주는 제어.** TRL `mask_truncated_completions`은 truncated completion을 벌주지 않고 loss에서 제거한다(`grpo_config.py` L259–262). SkyRL-Agent는 32K-token context limit이나 50-turn step limit에서 멈춘 trajectory를 reward·advantage estimation은 그대로 두고 gradient에서 제외한다([[skyrl-agent]] §4.2). verl agent loop은 tool·template token mask를 0으로 줘 학습하지 않는다([[verl-rollout]]). INTELLECT-3는 clip 대신 mask한다. `[α, β] = [0.5, 5]` 밖 importance ratio는 0으로, token ratio 하나가 `1e-5` 아래면 rollout 전체를 mask한다. report는 이를 trainer–inference mismatch에 critical하다고 부른다([[intellect-3]] §3.3, Eq. 1–2).

**agent loop에서 content로서의 음성.** TRL은 exception, unknown tool name, unsupported call type을 다음 turn에 model이 읽는 `{"error": ...}` tool message로 바꾸며 `tools/call_frequency`, `tools/failure_frequency`를 log한다([[trl-grpo]] L1502–1532, L1778–1786). SkyRL-Agent는 recoverable error와 tool failure, 남은 step/context budget, invalid function call, failed edit을 덮는 corrective feedback·hint를 inject한다(§3.4, §4.2). 이들은 model 자신의 다음 token에 ordinary cross-entropy로 학습한다. likelihood를 아래로 밀지 않으므로 gradient case의 displacement risk는 적용되지 않는다.

**일반성에 미치는 효과.** framework source 어느 것도 음성 sample handling이 breadth에 미치는 효과를 측정하지 않았다. 측정된 근거는 course의 다른 곳(ch-43, ch-44)에 있다. 이 장의 요구사항은 advantage sign과 domain별로 split 가능한 log를 주는 stack을 고르라는 것이다. 그렇지 않으면 domain 하나에 한정된 negative-gradient pathology가 평균에서 사라진다.

## 레시피

행은 pinned locus가 있는 published run setting과 별도로 기록한 framework default다. framework default와 paper value는 다른 사실이다.

| 모델(정확한 release) | 크기 | 단계 | 설정 | 값 | source 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| INTELLECT-3 (GLM-4.5-Air base) | 106B MoE | RL | prompts per step × rollouts per prompt; max context | 256 × 16; 65,536 tokens | arXiv:2512.16144 §3.3 | verified 2026-09-17 | no ablation reported |
| INTELLECT-3 | 106B MoE | RL | `max_off_policy_steps`; optimizer; LR | 8; Muon; 1e−6 | §3.3 | verified 2026-09-17 | stated as removing excessively off-policy rollouts; no sweep |
| INTELLECT-3 | 106B MoE | RL | IS masking band [α, β]; whole-rollout mask threshold | [0.5, 5]; 1e−5 | §3.3 Eq. 1–2 | verified 2026-09-17 | report states double-sided masking was critical against trainer–inference mismatch; no ablation table |
| INTELLECT-3 | 106B MoE | RL | nodes; train:inference split; step time | 60 × 8 H200; 16:44; ≈1500 s/step at 65,536 length | §3.3 | verified 2026-09-17 | without in-flight weight updating, step time rises > 2× |
| INTELLECT-3 | 106B MoE | eval-gate | in-loop eval interval and suite | every 15 steps on AIME24, AIME25, LiveCodeBench, HLE, GPQA | §3.3 "Online Evaluation", Fig. 9 | verified 2026-09-17 | scores still rising at end of run |
| AReaL (R1-Distilled-Qwen) | 1.5B–32B | RL | max staleness η | 4 for coding, 8 for math | arXiv:2505.24298 §7.1 | verified 2026-09-17 | Table 2 staleness sweep 0/1/2/4/8/16/∞ with and without decoupled PPO |
| verl `fully_async_policy` run (Qwen2.5-Math-7B) | 7B | RL | `staleness_threshold`; `partial_rollout`; rollout:train GPUs | 0.5; True; 64:64 of 128 H20 | verl@753aed3 `verl/experimental/fully_async_policy/README.md` L349–361 | verified 2026-09-14 ([[verl-rollout]]) | staleness ablation 0/0.1/0.3/0.5 in the same README |
| flash-rl DAPO run (Qwen2.5-32B) | 32B | RL | TIS cap C; sampler TP; learner Ulysses SP | 8; 2; 8 | github.com/yaof20/verl@fd02f7b `dapo_qwen32b_bf16.sh` L5, L41, L58, L63 | verified 2026-09-14 ([[rollout-training-mismatch-tis]]) | Fig. 1 TIS vs no TIS; no sweep over C |
| SA-SWE-32B (SkyRL-Agent, from Qwen3-32B) | 32B | RL | batch = mini-batch; rollouts per task; LR; KL / entropy | 64 = 64; 8; 1e−6; both disabled | arXiv:2511.16108 §4.2 | verified 2026-09-14 ([[skyrl-agent]]) | no ablation reported |
| SA-SWE-32B | 32B | RL | context / turn limits; truncated trajectories | 32K tokens, 50 turns; excluded from gradient | §4.2 | verified 2026-09-14 | no ablation reported |
| GiGPO on verl-agent (Qwen2.5-1.5B-Instruct) | 1.5B | RL | group size; groups per rollout; ω; γ; KL loss coef; history length | 8; 16 (128 environments); 1; 0.95; 0.01; 2 | arXiv:2505.10978v3 App. E.1 (history length: App. E.2) | verified 2026-09-17 | ω sensitivity in App. E.5 Table 5 (WebShop success rate peaks at 68.3 for ω = 0.8 and stays within 65.8–68.3 over ω ∈ [0.4, 1.2], against 56.6 at ω = 0 and 56.3 at ω = 1.4) |
| verl framework default | n/a | RL | `trainer.test_freq`; `val_before_train`; `log_val_generations` | −1; True; 0 | verl@753aed3 `ppo_trainer.yaml` L154–201 | verified 2026-09-17 | no ablation reported |
| verl framework default | n/a | RL | `rollout_is`; `rollout_is_threshold` | `null` (off); 2.0 | `algorithm/rollout_correction.yaml`@753aed3 | verified 2026-09-17 | no ablation reported |
| OpenRLHF framework default | n/a | RL | `--eval.steps`; `--eval.dataset`; eval temperature; eval samples/prompt | −1; None; 0.6; 4 | OpenRLHF@64c1cc4 `cli/train_ppo_ray.py` L286, L535–540 | verified 2026-09-17 | eval dataset requires `--reward.remote_url` (L672–673) |
| OpenRLHF framework default | n/a | RL | KL controller; `init_coef`; async queue size | `FixedKLController`; 0.01; 1 | `ppo_trainer.py` L171–176; `train_ppo_ray.py` L275, L417–419 | verified 2026-09-14 ([[openrlhf-ppo]], [[async-rollout]]) | no ablation reported |
| TRL framework default | n/a | RL | `loss_type`; β; `scale_rewards`; vLLM IS correction | `dapo`; 0.0; `group`; on with `use_vllm=True`, `sequence_mask`, cap 3.0 | trl@a08e713 `grpo_config.py` L709, L792–819 | verified 2026-09-14 ([[trl-grpo]]) | no ablation reported |
| TRL framework default | n/a | RL | async GRPO `max_staleness`; `queue_maxsize`; `weight_sync_steps`; `dtype` | 4; 1024; 1; `float32` | trl@a04ffd3 `trl/experimental/async_grpo/async_grpo_config.py` L37–44, L345–375 | verified 2026-09-17 | docstring cites the training–inference mismatch as the reason for the float32 default |


**소규모 범용 run의 시작점.** 수십억 parameter model의 single-node GRPO run에서는 여기 모든 수치가 위 `verified` 행과 원래 조건에서 왔다. TRL에서 `use_vllm=True`를 사용한다. 그러면 sequence-level IS correction이 cap 3.0으로 default on이 된다(TRL default 행: vLLM-sampled rollout용 default이며 특정 model에서 측정한 값은 아님). `loss_type`, `beta`는 `dapo`, 0.0 default를 받지 말고 명시한다. β = 0 run은 reference model을 만들지 않아 forgetting KL control이 전혀 없기 때문이다(TRL default 행). `eval_strategy`와 `num_generations_eval`도 명시해 schedule에 따라 실행하고 reward가 target하지 않는 held-out family를 하나 이상 넣는다. 이 source에서 그러한 규율의 published in-loop example은 INTELLECT-3의 15 step마다 5-benchmark plot뿐이며 106B MoE, H200 512장 조건이므로 interval은 transfer하지 않고 practice만 transfer한다. async를 켜면 `max_staleness` 4에서 시작한다. 이는 TRL default와 AReaL coding setting에 맞고(AReaL 1.5B–32B, H800), Table 2 η = 4 행은 missing decoupled objective가 가장 큰 손상을 낸 staleness이기도 하다.

## 일반화 관점

**(a) 범위를 넓히는 것.** per-domain logging과 함께 여러 reward source를 한 run에 route하여 run 중 mixture를 re-balance한다. TRL `reward_funcs` + `reward_weights`, prime-rl verifiers-environment abstraction이 두 in-tree design이다([[trl-grpo]]; [[intellect-3]] §2.1.1). agentic RL의 cross-domain transfer는 한 번 보고됐다. SWE-only RL이 Terminal-Bench 13.75 → 16.25, BrowseComp-Plus 18.1 → 19.4, WebArena 15.8 → 17.0을 올렸다([[skyrl-agent]] Table 3). cell마다 evaluation 하나다.

**(b) 좁아짐/망각을 일으키는 것.** default configuration에서 실행하는 것: TRL GRPO β = 0.0은 reference model·KL anchor가 없고, verl `use_kl_loss`, `use_kl_in_reward`도 default off다(§2.4). unbounded 또는 uncorrected staleness: decoupled objective 없는 AReaL η = 4는 AIME24 18.7점, MATH-500 22.3점을 잃는다([[areal-async-rl]] Table 2). uncorrected rollout–training mismatch: Thinking Machines Bigmath RLVR의 mid-run reward collapse와 INT8 DAPO-32B의 entropy 0.2 아래 하락·abnormally long response다([[thinkingmachines-defeating-nondeterminism]]; [[rollout-training-mismatch-tis]] Fig. 5). 좁은 training set: benchmark subset인 OSWorld task 32개에서 학습한 SkyRL-Agent computer-use agent는 training reward가 올라도 validation accuracy “little to no gain”이었다(§5.3).

**(c) 이 단계를 측정하는 법.** 빠지기 쉬운 순서대로 세 instrument다. 첫째 reward domain 밖 family 하나 이상을 포함한 fixed held-out suite의 in-loop evaluation을 step 0부터 schedule로 실행하고 benchmark별 series로 log한다(verl `test_freq`, OpenRLHF `--eval.steps`, TRL `eval_strategy`). 둘째 reward 옆에 mismatch metric(`vllm_kl` 또는 verl rollout-correction ratio)을 그린다. published failure가 먼저 거기에 나타난다. 셋째 evaluation path의 reproducibility discipline이다. sampler dtype·tensor/sequence-parallel size를 고정하고 default kernel은 batch-invariant가 아니므로 batch-invariant kernel 없이는 temperature-0 evaluation도 deterministic하지 않음을 안다([[thinkingmachines-defeating-nondeterminism]]). 알려진 measurement error는 evaluation generation이 rollout과 inference capacity를 경쟁하므로 짧은 `test_freq`가 stack throughput comparison에 bias를 준다는 것, small sample count benchmark score가 seed 간 수점 움직이지만 framework source는 이를 수량화하지 않는다는 것이다.

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 확인 |
|---|---|---|
| algorithm 이름으로 stack 선택 | 고른 framework가 “GRPO를 가진다”지만 default `loss_type`, β, normalizer가 재현할 paper와 다름 | trainer name 대신 default를 읽는다: TRL `loss_type="dapo"`, β = 0.0; verl `norm_adv_by_std_in_grpo=True`; OpenRLHF `group_norm`, ε = 1e-9 |
| in-loop evaluation 없이 실행 | dashboard에는 reward·KL·entropy·clip fraction만 있음 | launch 전 `test_freq` / `--eval.steps` / `eval_strategy`가 default off가 아닌지 확인 |
| 다른 sampler setting으로 생성한 checkpoint 비교 | training change 없이 breadth가 수점 improve/regress | eval number마다 sampler dtype·TP·SP size를 기록하고 결론 전 new setting에서 baseline 재실행 |
| 안정적으로 보인다고 IS correction off | entropy가 계속 하락, response length 증가, k1 KL 음수, logged reward 상승·benchmark accuracy 불변 | step 0부터 OpenRLHF `vllm_kl` 또는 verl correction ratio를 그린다. negative k1 KL은 mismatched engine 사이 log-ratio 신호([[rollout-training-mismatch-tis]] Fig. 5) |
| staleness를 bound하지 않고 throughput 때문에 async queue depth 증가 | throughput은 오르나 held-out score 하락·training reward 불변 | batch별 achieved staleness를 log해 configured bound와 비교; §4 AReaL η 표 참조 |
| zero-variance prompt group을 무해하다고 취급 | config 변화 없이 training curve 평탄, gradient norm 하락 | zero reward std group fraction을 log(TRL `frac_reward_zero_std`; 다른 곳은 manual 계산) |
| single-turn trainer를 agent loop으로 확장 | step wall time이 소수 episode에 지배, environment failure를 task failure로 셈 | environment-crash rate와 task-failure rate를 분리하고 turn-count mean 대신 distribution plot |
| framework stated capability가 현재라고 가정 | 6개월 전 작동한 configuration이 더는 import 안 됨 | commit을 pin하고 file을 읽는다: TRL PPO trainer는 2026-09-04 PR #7020으로 삭제됐고 같은 기간 async GRPO trainer가 생김 |

## 이해도 점검

1. clipped surrogate는 verl·OpenRLHF·TRL에서 거의 같다. 같은 algorithm run이 두 framework에서 breadth가 다른 model을 만들 수 있는 이유를 loss 밖 mechanism 세 개로 설명하라.
2. AReaL η = 4는 decoupled objective 없이 AIME24 23.3, 있으면 42.2다. “batch가 여러 policy version을 섞음”에서 held-out score 하락까지 causal chain과 decoupled objective가 끊는 고리를 설명하라.
3. OpenRLHF는 `--eval.dataset`에 앞서 `--reward.remote_url`을 요구한다. TRL run과 비교해 OpenRLHF generality evaluation을 구축하는 순서에 이 constraint가 뜻하는 바를 설명하라.
4. TRL GRPO default β = 0.0은 reference model을 만들지 않는다. optimization 측 default 찬성 논거와 generality 측 반대 논거, 이를 결정할 evidence를 말하라.
5. maximum sampler–learner mismatch는 uninterrupted response length에 따라 커지고 mean은 그렇지 않다. softmax-gradient와 importance-weight variance를 이용해 training stability에서 maximum이 mean보다 중요한 이유를 설명하라.
6. zero-variance prompt group은 policy-gradient term이 없지만 token-mean denominator에는 든다. batch 나머지 prompt effective step size에 미치는 효과를 도출하고 잡기 위해 무엇을 log할지 말하라.
7. GiGPO anchor-state grouping은 362.83 s iteration에 0.54 s를 더하고 ALFWorld success를 GRPO보다 13.3점 높인다. 이것이 compute가 아니라 credit assignment 논거인 이유와 single-turn/agentic trainer 선택에 주는 뜻을 설명하라.
8. throughput은 최고지만 instrumentation은 최저인 stack을 받았다. “generality가 speed보다 중요하다”는 주장에 기대지 않고 reject할 논거를 구성하라.

## 연결

- **ch-54 — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments.** 이 장이 고르는 staleness·partial rollout·environment integration mechanism.
- **ch-55 — verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation.** 여기 요약한 reward routing을 포함한 §3 verl column 전체.
- **ch-56 — OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls.** §2.4에서 인용한 KL controller behaviour를 포함한 OpenRLHF column.
- **ch-57 — TRL Internals: SFT, DPO, GRPO, and Distillation Trainers.** 이전 장. loss-type family와 tool loop를 포함한 TRL column.
- **ch-58a — Open General-Model Recipes End to End: Pretraining to Merge.** 다음 장. 여기서 고른 stack을 published recipe의 모든 stage에 따라가며 §2.4가 기록한 merge step도 포함.
- **ch-02 — Numerical Precision, Determinism, and Train–Inference Mismatch.** §2.2 gap의 numerical cause.
- **ch-43 — Entropy, Output Diversity, and KL Control in RL.** §2.4 KL-placement 행의 theory와 §3 행 16 entropy diagnostic.
- **ch-51a — Evaluating Agent Generality and Reliability.** §6의 evaluation 측면.
- **ch-59 — Capstone: Reproduce One Stage of an Open General-Model Recipe with a Generality Gate.** 이 장 decision procedure를 stack-selection input으로 사용.

## 출처

- [[verl-ppo-loss]] — verl registered policy loss·clip parameter, §1 shared surrogate.
- [[verl-grpo]] — verl GRPO estimator, ε = 1e-6 constant, zero-advantage group 동작, 음성 섹션의 token-denominator effect.
- [[verl-rollout]] — async vLLM server, `max_tokens` clamp, tool agent loop·response mask, `fully_async_policy` 2.35× timing 표.
- [[openrlhf-ppo]] — `PolicyLoss`, KL controller·default coefficient, `vllm_kl` metric이 있는 IS-correction mode 세 개.
- [[openrlhf-dpo]] — §3 행 15 DPO path와 MoE aux-loss 보존.
- [[async-rollout]] — OpenRLHF async trainer, queue semantics·partial rollout, 공개 throughput 측정 부재.
- [[trl-grpo]] — loss type 8개와 default, advantage scaling, vLLM IS correction, sign-split clip metric, error message를 content로 쓰는 tool loop.
- [[trl-ppo]] — TRL actor-critic PPO와 §3 행 3에서 쓴 PR #7020 제거.
- [[trl-online-dpo]] — §3 행 15의 experimental preference-optimization trainer.
- [[entropy-logging-patterns]] — framework 세 개의 metric name과 KL estimator 차이.
- [[rollout-training-mismatch-tis]] — 측정한 sampler–learner gap, TIS formula·variance 논거, parallelism·length factor analysis, flash-rl DAPO recipe 행.
- [[thinkingmachines-defeating-nondeterminism]] — batch invariance, 1000 completion 중 80개 결과, off-policy correction·sampler–trainer KL의 RL section.
- [[areal-async-rl]] — interruptible rollout, staleness bound η, decoupled PPO objective, §4 전반에서 쓴 Table 2 staleness sweep.
- [[skyrl-agent]] — three-stage dispatcher와 1.55× 측정, transition backend bridge, SA-SWE-32B recipe, domain 밖 evaluation 표.
- [[gigpo-verl-agent]] — memory module을 쓰는 verl-agent step-wise rollout, anchor-state grouping, ALFWorld/WebShop result, iteration overhead breakdown.
- [[agent-lightning]] — §6 design contrast를 위해 인용한 external framework agent의 transition-level training.
- [[intellect-3]] — prime-rl orchestrator/trainer/inference architecture, verifiers Environments Hub interface, RL setting·mask threshold, benchmark 다섯 개 15-step in-loop evaluation.
