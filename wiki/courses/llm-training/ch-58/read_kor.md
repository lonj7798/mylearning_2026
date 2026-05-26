<!-- chapter: ch-58
     track: infra
     kind: content
     title: Framework Comparison and When to Pick Which
     deps: [ch-57]
     sources: [[verl-ppo-loss]], [[verl-grpo]], [[verl-rollout]], [[openrlhf-ppo]], [[openrlhf-dpo]], [[trl-ppo]], [[trl-grpo]], [[trl-online-dpo]], [[entropy-logging-patterns]], [[openrlhf-entropy-debugging]], [[async-rollout]]
     figures: figures/framework-compare.html
-->

# 58장 — Framework 비교와 선택 기준

> **핵심 통찰.** verl, OpenRLHF, TRL은 세 개의 "PPO library"가 아니다. 같은 대수에 대한 세 가지 *architectural bet*이다. TRL은 단일 Python process, Accelerate, 교육적 명료성에 건다. OpenRLHF는 Ray actor, `nn.Module` loss, `AdaptiveKLController`에 건다. verl은 registry(`@register_policy_loss`, `@register_adv_est`), 요청별 priority가 있는 async vLLM engine, SPMD HybridFlow에 건다. PPO surrogate는 셋 모두에서 같다. 하지만 각 framework가 버틸 수 있는 *deployment envelope*은 같지 않다. 막 읽은 README가 아니라, 지금 launch하려는 run과 같은 bet을 한 framework를 골라라.
>
> **가이드라인.** feature matrix(§2)와 decision tree(§5)를 한 쌍으로 사용하라. matrix는 "이 framework가 X를 할 수 있는가?"에 답하고, tree는 "내 scale + algo + engineering budget에서 어떤 framework가 가장 싼 bet인가?"에 답한다. scale을 backend에 맞춰라. single-node experiment와 online DPO는 TRL에 산다. `AdaptiveKLController`와 깔끔한 async semantic이 있는 production Ray-based RLHF는 OpenRLHF에 산다. partial rollout과 MoE routing이 있는 128-GPU+ HybridFlow run은 verl에 산다. 졸업 경로(§6), 즉 TRL → OpenRLHF → verl은 team이 run을 키우며 실제로 지나가는 경로다.

---

## §1 세 framework가 동의하는 것

대조하기 전에 invariant를 명시하자. 2026년 시점에서 세 codebase를 모두 읽으면 다음이 보인다.

- **같은 PPO-clip algebra.** verl `compute_policy_loss_vanilla`([[verl-ppo-loss]] L1080–1140), OpenRLHF `PolicyLoss.forward`([[openrlhf-ppo]] L68–168), TRL의 inline PPO([[trl-ppo]] L820–870)는 모두 `−min(r·A, clip(r, 1−ε_low, 1+ε_high)·A)`를 minimize한다. 차이는 장식이다.
- **같은 GRPO advantage.** verl의 `compute_grpo_outcome_advantage`([[verl-grpo]] L290–335)와 TRL의 upstream advantage builder는 모두 prompt group 안에서 outcome reward를 z-score한다. OpenRLHF는 experience buffer preprocessing에서 group baseline을 계산한다. 대수적으로 동일하다.
- **세 canonical KL estimator**([[entropy-logging-patterns]] 기준 K1 / K2 / K3): K1 = `Δlogp`(biased, verl reward shaping, TRL `objective/kl`), K2 = `0.5·Δlogp²`(Schulman, TRL `approxkl`), K3 = `exp(−Δlogp)+Δlogp−1`(unbiased, verl `kl_penalty="k3"`, TRL GRPO loss term).
- **Entropy는 기본적으로 loss term이 아니다.** 셋 모두 logging은 하지만 default config에서 entropy bonus로 regularize하지 않는다([[openrlhf-entropy-debugging]] defaults).
- **Rollout에는 vLLM.** 셋 모두 vLLM을 integrate한다. verl은 *async* vLLM engine을 실행한다([[verl-rollout]]). OpenRLHF의 async mode는 Ray + `vllm_lock`을 사용한다([[async-rollout]]). TRL은 in-process `_generate_vllm_server`와 `_generate_vllm_colocate`를 제공한다.

이들을 다르게 만드는 것은 loss가 아니다. KL이 어디에 들어가는지, rollout이 어떻게 schedule되는지, parallelism이 어떻게 표현되는지, 어떤 algorithm zoo가 in-tree로 제공되는지가 차이다.

---

## §2 Feature matrix

16-row attested matrix다. 모든 cell은 raw-data source에서 온 사실이다. "n/a"는 2026-04-21에 fetch한 `main` 기준 해당 framework에 그 feature가 실제로 없다는 뜻이다.

| # | Dimension | verl | OpenRLHF | TRL |
|---|---|---|---|---|
| 1 | **Rollout backend** | async vLLM `AsyncLLMEngine`([[verl-rollout]] `vllm_async_server.py` L440–525); SPMD sync는 PR #4411에서 retired | vLLM을 감싼 Ray actor. `vllm_lock`이 weight broadcast를 gate([[async-rollout]]) | HF `.generate()` default; `_generate_vllm_server` + `_generate_vllm_colocate`([[trl-online-dpo]] L585–893) |
| 2 | **Rollout orchestration** | `DataProto` + worker group; tokens-in-tokens-out; 요청별 `priority` | Ray `rollout_queue` + `rollout_slots`; `strategy.args.train.partial_rollout_enable`을 통한 partial-rollout([[async-rollout]]) | 단일 Python process; DDP/FSDP coordination에 Accelerate `gather()` |
| 3 | **Parallelism** | FSDP + Megatron-LM tensor/pipeline parallel via HybridFlow dataflow([[verl-rollout]]); multi-node native | DeepSpeed ZeRO-1/2/3 + Ray distribution; actor/critic split across GPUs([[openrlhf-dpo]]) | Accelerate + DeepSpeed/FSDP; single worker-group pattern; mainline에 Megatron integration 없음 |
| 4 | **PPO(actor-critic)** | `@register_policy_loss("vanilla")` + value loss; asymmetric `clip_ratio_low/high`, `clip_ratio_c=3.0` dual-clip([[verl-ppo-loss]]) | `PolicyLoss` module with dual_clip + asymmetric clip + three IS modes([[openrlhf-ppo]]) | `trl/experimental/ppo/ppo_trainer.py`; symmetric `cliprange` only; value clipping; 현재 experimental([[trl-ppo]]) |
| 5 | **GRPO / Dr.GRPO** | `@register_adv_est(GRPO)` + `norm_adv_by_std_in_grpo` toggle(False = Dr.GRPO)([[verl-grpo]]); Pass@k variant | Experience-buffer preprocessing에서 group baseline 계산. `PolicyLoss` 재사용 | `GRPOTrainer._compute_loss` with `loss_type ∈ {grpo, dr_grpo, bnpo, dapo, cispo, sapo, luspo, vespo}`([[trl-grpo]]) |
| 6 | **DPO(offline)** | 중심 주제는 아님. preference loss plugin은 있으나 track은 RL | `DPOLoss` + `concatenated_forward` + MoE aux + NLL-mix + label smoothing + IPO([[openrlhf-dpo]] L231–257) | `DPOTrainer` with `loss_type ∈ {sigmoid, hinge, ipo, kto_pair, ...}` |
| 7 | **Online DPO / judge-driven** | mainline에는 n/a | mainline에는 n/a | `trl/experimental/online_dpo/online_dpo_trainer.py` + sibling `nash_md`, `xpo`, `self_distillation`([[trl-online-dpo]]) |
| 8 | **KL-to-ref location** | Reward shaping(GAE 전 token별로 subtract); `kl_penalty()`에 K1/K2/K3 switch | `ppo_trainer.py` L172 주변의 `AdaptiveKLController` / `FixedKLController`를 통한 reward shaping | PPO: `non_score_reward` + adaptive controller로 reward shaping. GRPO: K3와 함께 `β·per_token_kl`이 *loss*에 추가됨([[entropy-logging-patterns]]) |
| 9 | **Entropy logging** | `verl_F.entropy_from_logits`를 통한 `actor/entropy`(true H); optional registry entropy loss | step별 mean `−logp`(biased proxy) | PPO: 두 field(`objective/entropy` biased, `policy/entropy_avg` true H). GRPO: `_metrics[mode]["entropy"]`(true H) |
| 10 | **vLLM-vs-train IS correction** | `rollout_is_weights`가 `compute_policy_loss_vanilla`에 전달됨. per-token multiplier | `enable_vllm_is_correction` with three modes: `tis`(truncated IS), `seq-mask-tis`, `icepop`([[openrlhf-ppo]] L60–76); `vllm_kl` metric 노출 | `GRPOTrainer`의 `vllm_importance_sampling_correction`; `importance_sampling_ratio`가 per-token loss에 곱해짐([[trl-grpo]]) |
| 11 | **DAPO-style asymmetric clip** | Native: `clip_ratio_low`, `clip_ratio_high`, `delta` upper cap | Native: `clip_eps_low`, `clip_eps_high` | GRPO only: `epsilon_low`, `epsilon_high`, `delta` cap |
| 12 | **GSPO(sequence-level ratio)** | loss registry(custom)를 통해 | Native one-line branch: `policy_loss_type="gspo"`([[openrlhf-ppo]] L43–48) | `GRPOTrainer`의 `importance_sampling_level="sequence"` |
| 13 | **Distributed checkpointing(DCP)** | FSDP DCP + Megatron checkpoint; scale에서는 sharded-save mandatory | DeepSpeed ZeRO checkpoint; ref model용 CPU-offload | Accelerate / safetensors sharded save; native Megatron path 없음 |
| 14 | **Async rollout / partial rollout** | 예: `priority`가 in-flight를 reorder하고 broadcast 중 pause-state(~L628) | 예: `ppo_trainer_async.py` + partial-rollout flag; 1.6–2.0× throughput([[async-rollout]]) | mainline async trainer 없음. [[async-rollout]] 기준 "forthcoming" |
| 15 | **Multi-node / 128-GPU+** | Primary target. HybridFlow dataflow가 code change 없이 sync→async→partial-rollout으로 scale | Ray cluster로 지원. Hu 2024에 70B production run 보고 | 8×H100까지 실용적. 70B는 Accelerate+FSDP로 가능하지만 partial-rollout knob 없음 |
| 16 | **MoE support** | Megatron MoE routing + async-aware weight broadcast; expert selection capture | DPO: MoE aux loss preserved([[openrlhf-dpo]]); PPO: DeepSpeed-MoE로 지원 | Basic(HF `transformers` MoE forward); specialized router-capture path 없음 |
| 17 | **VLM / multimodal** | async server에 `multi_modal_data`가 plumbed됨(`image_data`, `video_data`)([[verl-rollout]] L73) | Experimental extension. mainline은 아님 | Mainline `VisionDPOTrainer` / `VisionGRPOTrainer`(transformers integration) |
| 18 | **Maintenance velocity(2026-04)** | Active(volcengine/ByteDance Seed production); monthly tagged releases | Active; monthly releases; Hu et al. community | HF 쪽에서 셋 중 가장 active. weekly merges. GRPO zoo가 가장 빠르게 확장 |

---

## §3 Logging-pattern crib sheet

dashboard가 깨졌을 때는 각 framework가 세 신호를 *정확히* 무엇이라 부르는지 알아야 한다. [[entropy-logging-patterns]] 기준:

| Signal | verl field | OpenRLHF field | TRL PPO field | TRL GRPO field |
|---|---|---|---|---|
| **KL(π‖π_old)** | `actor/ppo_kl`(K1, logged) | `PolicyLoss.forward`의 `ppo_kl` | `objective/kl`(K1) + `policy/approxkl_avg`(K2) | loss 안의 K3를 통해 implicit |
| **KL(π‖π_ref)** | reward에 shaped. `kl_penalty`로 monitor | token별 `kl_ctl.value · kl_t`; `status["kl"]` | `objective/kl`(PPO용); GRPO: `β·per_token_kl` loss term |
| **Rollout-vs-train KL** | token별 ratio `rollout_is_weights`를 통해 | `PolicyLoss.forward`가 반환하는 `vllm_kl` | PPO에는 n/a. GRPO: `importance_sampling_ratio` |
| **Entropy** | `actor/entropy`(logits에서 true H) | step별 mean `−logp`(biased) | `objective/entropy`(biased) + `policy/entropy_avg`(true H) | `_metrics[mode]["entropy"]`(true H) |
| **Clip fraction** | `actor/pg_clipfrac` + `actor/pg_clipfrac_lower`(dual-clip hits) | `PolicyLoss.forward`가 반환하는 `clip_ratio` | `policy/clipfrac_avg` | in-loop 계산. 단일 canonical key 없음 |

**Collapse signature는 셋 모두에서 동일하다**([[openrlhf-entropy-debugging]]): entropy가 <100 step 안에 ≥30% 하락하고, PPO-KL이 ≥0.1로 spike하며, clipfrac가 1에 붙는다. framework별 key가 무엇이든 이 세 신호에 매핑해 사용하라.

---

## §4 Performance envelope

[[async-rollout]]과 verl blog에서 확인된 throughput:

- **Sync(TRL default).** rollout 중 optimizer는 idle, optimizer step 중 rollout은 idle. 8×H100의 7B에서 rollout이 wall time의 ≥70%를 지배한다. straggler handling은 없다. batch에서 가장 느린 rollout이 step을 gate한다.
- **OpenRLHF async.** `rollout_queue`가 둘을 decouple한다. staleness는 `k = queue_depth + partial_rollout_depth`, 보통 1–2로 bounded된다. Hu 2024 Figure 5: 7B에서 1.9× throughput, 70B에서 1.6×. `vllm_lock`이 weight broadcast를 serialize한다. partial-rollout mode는 weight update 시 in-flight generation을 interrupt한다.
- **verl async HybridFlow.** vLLM internal scheduler + 요청별 `priority` + pause-state. "continuous batching RL" pattern은 newly-weighted request가 straggler보다 앞서 가게 한다. in-flight generation은 old weight 아래에서 마치고 이어진다. OpenRLHF의 queue-based model보다 GPU를 더 완전히 saturate한다고 주장된다. scheduling granularity가 Ray가 아니라 vLLM이기 때문이다.

**Straggler handling**이 세 framework가 가장 크게 갈라지는 지점이다. TRL에는 없다. batch에서 가장 느린 rollout이 step을 gate한다. OpenRLHF의 queue는 trainer를 rollout variance에서 isolate하지만 rollout batch 내부에서 reorder하지 않는다. verl의 `priority`는 vLLM engine *안에서* reorder하여 짧은 completion을 먼저 끝내고 긴 completion은 stale weight 아래에서 계속하게 한다(IS-correction이 bias를 처리한다).

**vLLM-vs-train logprob mismatch**는 숨은 tax다. bf16 vLLM inference와 fp32 actor forward는 systematic ratio bias를 만든다. IS correction이 없으면 long completion에서 PPO가 약 50 step 안에 destabilize한다([[openrlhf-ppo]]). verl과 OpenRLHF는 모두 IS correction을 제공한다. TRL의 GRPO path에는 `vllm_importance_sampling_correction`이 있지만 TRL PPO에는 없다.

---

## §4.5 세 가지 구체적인 run profile

§4의 abstraction을 구체화하기 위해 세 attested envelope을 보자. 각각 다른 framework가 §5의 다른 leaf에 도착한다.

**Profile A — "Qwen-2.5-3B RLVR on 1×A100, 5K math prompts."** 3B base, G=8 rollouts, max_completion_length=1024, β_KL=0.05, Dr.GRPO aggregation. rollout은 wall-time의 약 60%지만 single GPU에서는 의미 있게 async할 수 없다. TRL을 고른다. `GRPOTrainer(loss_type="dr_grpo", vllm_importance_sampling_correction=True)`. Ray도 DeepSpeed dance도 없고, `_compute_loss`가 debugging하기에 single screen에 들어온다. 이것은 [[openrlhf-entropy-debugging]]의 "resource-constrained path" shape다.

**Profile B — "Llama-3-70B PPO on 32×H100, 3-day run, ultrafeedback prompts."** rollout이 wall-time의 약 80%다. sync는 GPU-hours를 하루 이상 낭비한다. OpenRLHF를 고른다. `AdaptiveKLController(target=10.0)`와 함께 `PolicyLoss(clip_eps_low=0.2, clip_eps_high=0.28, dual_clip=3.0, enable_vllm_is_correction=True, vllm_is_correction_type="tis")`를 사용한다. Ray-managed actor/critic split, queue-level async가 [[async-rollout]] Figure 5에서 확인된 약 1.6× throughput을 회수한다. dashboard의 `vllm_kl`은 PPO가 destabilize하기 전에 bf16/fp32 mismatch를 잡는다.

**Profile C — "Mixtral-8x22B GRPO on 128×H100, 2-week run, multi-modal agent rollouts."** straggler variance가 지배하고, MoE routing + VLM `image_data`가 in-flight이며, partial-rollout이 필수다. verl을 고른다. `clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=3.0`의 `@register_policy_loss("vanilla")`, `norm_adv_by_std_in_grpo=False`(Dr.GRPO mode)의 `@register_adv_est(GRPO)`를 사용한다. Async vLLM with `priority` reordering, weight broadcast 중 pause-state, 8x22B용 Megatron TP=8/PP=2. 같은 run은 OpenRLHF의 queue-level scheduling에 *맞지 않는다*. 가장 느린 sample이 step을 gate한다.

세 profile은 모두 같은 PPO algebra를 공유한다. framework 선택은 loss가 아니라 전적으로 §4의 performance envelope과 §2의 feature matrix가 결정한다.

---

## §5 Decision tree — 어떤 run에 어떤 framework인가

위에서 아래로 읽고, 첫 matching point에서 멈춘다. 모든 leaf는 한 줄 justification이 붙은 concrete recommendation이다.

```
Q1. 이것이 <8-GPU experimental run, pedagogical exercise, 또는 offline DPO인가?
    YES → TRL.
         근거: 단일 Python process이고 `Accelerate`가 parallelism을 처리하며,
         `DPOTrainer` + `GRPOTrainer`가 single-node run의 90%를 커버한다.
         monolithic `_compute_loss`는 debugging할 때 읽기 가장 명료한 코드다.
    NO  → Q2.

Q2. ONLINE DPO, self-rewarding LM, SPIN, Nash-MD, 또는 XPO가 필요한가?
    YES → TRL(필수).
         근거: `trl/experimental/online_dpo/` + sibling `nash_md`, `xpo`,
         `self_distillation` -- OpenRLHF 또는 verl mainline에는 동등한 것이 없다
         ([[trl-online-dpo]]).
    NO  → Q3.

Q3. 8–64 GPU run이고, standard PPO/GRPO/DPO이며, ByteDance / Qwen-style
    asymmetric clip과 partial-rollout async를 쓰는가?
    Q3a. AdaptiveKLController와 7B–70B에서 검증된 track record를 갖춘
         깔끔한 Ray-based async pattern을 주로 원하는가?
         YES → OpenRLHF.
              근거: `PolicyLoss` nn.Module + `AdaptiveKLController` +
              `rollout_queue`가 가장 작은 production-grade surface다.
              1.9x/1.6x throughput이 확인되어 있으며, IS-correction(tis/icepop/
              seq-mask-tis)이 in-tree로 제공된다([[openrlhf-ppo]], [[async-rollout]]).
         NO  → Q3b.
    Q3b. router capture가 있는 MoE, VLM multi-modal rollout, GRPO
         Pass@k, 또는 Megatron tensor/pipeline parallel이 필요한가?
         YES → verl.
              근거: `vllm_async_server.py`가 `multi_modal_data`를 plumb하고,
              registry-pluggable adv estimator에 GRPO Pass@k가 포함되며,
              Megatron integration이 first-class다([[verl-rollout]], [[verl-grpo]]).

Q4. run이 128-GPU+, multi-node, per-request priority가 있는 partial-rollout이며,
    몇 주 동안 실행될 예정인가?
    YES → verl.
         근거: HybridFlow dataflow graph는 셋 중 유일하게
         "same code, sync→async→partial-rollout"을 목표로 설계되었다.
         per-request priority + pause-state는 유일한 in-vLLM scheduler이며,
         ByteDance Seed production reference가 있다([[async-rollout]]).
    NO  → Q3를 다시 검토.

Q5. TRL에만 있고 다른 framework에는 없는 feature(CISPO, SAPO, LUSPO,
    entropy-quantile masking, online judge-driven DPO, KTO, SimPO, ORPO)가 필요한가?
    YES → scale이 크더라도 TRL.
         근거: GRPO zoo(`loss_type` switch가 8 variant를 커버)와
         DPO zoo(`loss_type`이 6+ variant를 커버)는 TRL-exclusive다.
         scaling cost는 실제지만, one-off research run에서는 이를
         verl/OpenRLHF에 재구현하는 algorithm cost가 더 크다([[trl-grpo]]).

Q6. 이번 분기의 primary goal이 algorithm debugging / understanding인가?
    YES → TRL(읽기), 그다음 OpenRLHF 또는 verl로 port.
         근거: one-file implementation이고 Ray / dataflow indirection이 없다.
```

---

## §6 Graduation criteria — TRL → OpenRLHF → verl

team은 "하나를 고르는" 것이 아니라, run이 이전 bet을 넘어설 때마다 다음으로 *졸업*한다. 기준은 의견이 아니라 codebase에서 나오는 concrete signal이다.

**TRL에서 OpenRLHF로 졸업할 때:**
- `AdaptiveKLController`가 필요할 때. TRL에도 PPO용 controller가 있지만 minimal하다. OpenRLHF의 것은 InstructGPT-faithful implementation이다.
- vLLM-IS-correction wall에 부딪힐 때. long completion에서 PPO가 약 50 step 안에 destabilize하고, `tis` / `icepop` / `seq-mask-tis`가 필요해지는 경우([[openrlhf-ppo]] L60–76).
- rollout이 wall time의 ≥70%가 되어 async로 회수해야 할 때. TRL의 in-process vLLM은 rollout과 optimizer step을 overlap할 수 없고, OpenRLHF의 Ray-based async는 가능하다.
- Mixtral/DeepSeek-MoE에서 DPO 중 MoE aux-loss preservation이 필요할 때. OpenRLHF의 `DPOTrainer`는 이를 native로 처리한다. TRL mainline은 그렇지 않다.
- single-process FSDP 대신 GPU 전반에 걸친 clean actor/critic split(Ray-distributed)이 필요할 때.

**OpenRLHF에서 verl로 졸업할 때:**
- batch 안에서 가장 느린 sample이 step을 gate하는 straggler-dominated rollout에 부딪히고, engine 내부에서 reorder할 per-request `priority`가 필요할 때. OpenRLHF의 queue-level isolation만으로는 부족하다. vLLM-scheduler-level reordering이 필요하다([[verl-rollout]]).
- 64 GPU를 넘어서고 HybridFlow dataflow가 필요할 때. 같은 code가 8 GPU에서 sync, 32에서 async, 128에서 partial-rollout으로 동작해야 한다.
- Megatron tensor/pipeline parallelism이 필요할 때(activation recomputation pathology가 있는 70B+).
- rollout engine이 `multi_modal_data`를 받아야 하는 VLM modality(image/video token)를 추가할 때.
- plug-in algorithm research를 원할 때. trainer를 수정하지 않고 30줄짜리 `@register_*` function으로 새 advantage estimator나 policy loss를 작성하는 것이 verl idiom이다.
- token granularity에서 KL-as-reward-shaping을 디버깅할 때. verl의 `kl_penalty()`와 `k1/k2/k3` switch가 가장 깔끔한 instrumentation surface다.

**졸업하지 말아야 할 때:**
- Ray 때문에만 OpenRLHF로 졸업하지 말라. Ray는 debugging difficulty를 더한다. async와 distributed actor/critic이 동시에 필요할 때만 도입하라.
- "가장 새롭기 때문에" verl로 졸업하지 말라. registry pattern은 강력하지만 indirection을 추가한다. custom `@register_policy_loss`를 절대 작성하지 않을 것이라면 OpenRLHF의 `PolicyLoss` module이 더 단순하다.
- 앞으로 갔다가 되돌아가지 말라. run이 verl-scale이 되면 TRL에서 재현하려면 Megatron integration 작업을 하거나 throughput을 낮게 받아들여야 한다. migration은 one-way로 계획하라.

---

## §6.5 2025–2026 run에서 관찰된 anti-pattern

team이 잘못된 framework를 골라 비용을 낸 흔한 failure mode:

- **128 GPU에서 TRL.** 더 많은 DeepSpeed configuration을 얹어 `PPOTrainer`나 `GRPOTrainer`를 ~8 H100 이상으로 scale하려는 시도. failure mode: rollout이 wall time의 ≥70%로 남고, partial-rollout knob은 없으며, fix("add async")는 training loop 재작성을 요구한다. attested signal: 500+ step 동안 `rollout_time / total_time > 0.7`가 지속된다. §6에 따라 졸업하라.
- **Online DPO에 OpenRLHF.** fresh pair를 sample하기 위해 `DPOTrainer` 위에 "judge callback"을 만드는 경우. failure mode: pair harness, EOS penalty, KL-canary, judge-interface plumbing을 모두 다시 만들어야 한다. [[trl-online-dpo]]가 이미 이를 제공한다. attested signal: non-TRL repo의 첫 `self_rewarding` commit은 거의 항상 TRL의 `_calculate_rewards_from_functions`를 복사한다.
- **교육용 one-file read에 verl.** `trl/trainer/grpo_trainer.py`가 모든 variant를 담은 single-file `_compute_loss`를 제공하는데도 GRPO를 이해하려고 ByteDance HybridFlow stack을 끌어오는 경우. failure mode: learner가 backward pass 하나를 실행하기 전 Ray + Megatron debugging에 사흘을 쓴다. §5 Q6이 이런 reading을 TRL로 route하는 이유가 있다.
- **Framework 사이에서 KL estimator를 섞기.** K1으로 verl sweep을 실행한 뒤, TRL GRPO(K3 in loss)로 port하고 KL number를 직접 비교하는 경우. [[entropy-logging-patterns]]가 이를 경고한다. K1은 biased이고 음수가 될 수 있으며, K3는 unbiased이고 ≥0이다. renormalization 없이는 curve를 비교할 수 없다.
- **"TRL PPO baseline과 맞추려고" `vllm_importance_sampling_correction` 끄기.** TRL PPO에는 correction이 없다. OpenRLHF와 verl에는 있다. reference와 맞추려고 production framework에서 이를 끄는 것은 방향이 틀렸다. long completion에서 PPO가 약 50 step 안에 destabilize한다. baseline을 upgrade하라.

각 anti-pattern의 root cause는 framework를 architectural bet이 아니라 commodity substitution으로 취급하는 것이다. §2의 matrix와 §5의 tree는 substitution cost를 명시하기 위해 함께 설계되었다.

---

## §7 Companion interactive

[framework-compare.html](figures/framework-compare.html)을 보라. scale, algo, special requirement(MoE / VLM / online DPO / async)를 고르면 recommendation이 matrix row를 인용하는 justification과 함께 update된다. checklist로 사용하라. 모든 recommendation은 cross-reference할 수 있는 §2 cell에 닿는다.

---

## §8 이 장이 아닌 것

이 장은 benchmark chapter가 아니다. 어떤 framework도 algo / model size / rollout length / straggler distribution과 독립적인 단일 "throughput number"를 갖지 않는다. reproducible benchmark는 네 가지를 모두 지정해야 하며, raw-data library에서 확인된 유일한 숫자는 Hu 2024([[async-rollout]] Figure 5)의 수치다. 또한 이 장은 recommendation chapter도 아니다. recommendation은 항상 "이 run과 architectural bet이 맞는 framework"다. §5의 tree가 그 match를 만들지만, 자기 constraint를 아는 team은 tree 없이도 route할 수 있다.

앞으로 가져갈 synthesis는 이것이다. **algorithm은 paper 안에 있고, framework는 deployment envelope이다**. 59장(capstone)은 open recipe 하나를 end-to-end로 재현하라고 요구할 것이다. 이 장은 그 capstone의 "which framework" 질문에 대한 decision input이다.

---

## Connections

- **ch-55 (verl internals)** — `[[verl-ppo-loss]]`, `[[verl-grpo]]`, `[[verl-rollout]]`에 대한 framework-specific deep read.
- **ch-56 (OpenRLHF internals)** — `[[openrlhf-ppo]]`, `[[openrlhf-dpo]]`, `[[async-rollout]]`에 대한 deep read.
- **ch-57 (TRL internals)** — `[[trl-ppo]]`, `[[trl-grpo]]`, `[[trl-online-dpo]]`에 대한 deep read.
- **ch-43 (entropy/KL control)** — §3 logging crib sheet의 theory.
- **ch-59 (capstone)** — 이 장의 decision tree를 framework-selection input으로 사용하는 "pick one and reproduce" run.
- [[entropy-logging-patterns]], [[openrlhf-entropy-debugging]] — 이 장이 §3에서 인용하는 cross-framework table.
- [[async-rollout]] — §4와 §5의 scale threshold를 뒷받침하는 throughput + staleness-bound evidence.
