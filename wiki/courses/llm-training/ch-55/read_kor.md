<!-- chapter: ch-55
     track: infra
     kind: content
     title: verl Internals
     deps: [ch-54]
     sources: [[verl-ppo-loss]], [[verl-grpo]], [[verl-rollout]], [[entropy-logging-patterns]], [[async-rollout]], [[ppo]], [[grpo]], [[dr-grpo]], [[kl-control-rlhf]]
     figures: figures/verl-structure.html
-->

# 55장 — verl 내부 구조

> **핵심 통찰.** verl은 서로 직교하는 두 레지스트리, 즉 advantage estimator(`@register_adv_est`)와 policy loss(`@register_policy_loss`)를 중심으로 구성되어 있다. 그래서 PPO, GRPO, Dr.GRPO, GSPO와 그 후속 알고리즘들은 별도의 코드 경로가 아니라 *설정 선택지*가 된다. 핵심 루프는 작다. `core_algos.py`에는 clipped-surrogate 대수식 약 100줄이 있고, `ppo_trainer.py`는 rollout→logprob→advantage→update 흐름을 조율하며, `vllm_async_server.py`는 GPU 시간을 가장 많이 잡아먹는 부분(rollout)을 맡는다. 그 밖의 모든 것, 즉 FSDP sharding, weight broadcast, LoRA adapter, MoE routing capture는 이 뼈대에 꽂힌다. `compute_grpo_outcome_advantage`에서 나온 `advantage` 토큰 하나가 `compute_policy_loss_vanilla`를 거쳐 `backward()`로 들어가는 흐름을 추적할 수 있게 되면, 나머지 저장소는 오후 한 번이면 훑을 수 있다.
>
> **가이드라인.** verl은 *trainer에서 bottom-up으로 읽지 말고, registry에서 top-down으로 읽어라.* `verl/trainer/ppo/core_algos.py`의 `@register_policy_loss("vanilla")`와 `@register_adv_est(AdvantageEstimator.GRPO)`에서 시작하라. 이 두 decorated function이 대수의 100%다. 레지스트리를 이해한 뒤에야 `ppo_trainer.py`(순서), `fsdp_sft_trainer.py`(RL 없는 baseline trainer), async rollout server로 들어가라. 실행을 디버깅할 때는 KL이 어디에 있는지 기억하라. verl은 TRL-GRPO와 달리 KL을 loss가 아니라 *reward*에 넣는다.

---

## §1 저장소 둘러보기 — 실제로 실행되는 파일 트리

verl의 production RL 코드는 네 디렉터리에 산다. 나머지는 config, 문서, 테스트, 실험 코드다.

```
verl/
  trainer/
    ppo/
      core_algos.py       # ← registries: @register_policy_loss, @register_adv_est
      ppo_trainer.py      # ← the rollout→logprob→adv→loss sweep
      reward.py, config/  # reward function + hparam dataclasses
    fsdp_sft_trainer.py   # ← non-RL baseline trainer (read this before ppo_trainer)
  workers/
    actor/                # ← forward pass + entropy_from_logits
    critic/               # ← value head (unused in GRPO)
    rollout/
      hf_rollout.py       # ← reference rollout — FSDP + HF .generate
      vllm_rollout/
        vllm_rollout.py        # ← ServerAdapter — SPMD sync, NOW RETIRED (PR #4411)
        vllm_async_server.py   # ← production async vLLM engine
  models/                 # ← FSDP + TP wrappers for llama/qwen/mixtral
  utils/, tools/          # ← masked_mean, agg_loss, checkpointing
```

두 가지 설계 결정이 레이아웃을 지배한다.
- **상속이 아니라 레지스트리.** GSPO나 Dr.GRPO를 추가하는 데 trainer subclassing은 필요 없다. 새 `@register_adv_est(...)` 함수와 config flag 하나가 필요할 뿐이다. 새 대수를 넣을 때 편집하는 유일한 파일은 `core_algos.py`다.
- **Rollout은 trainer callback이 아니라 worker다.** 별도의 Ray worker group이 vLLM을 감싼다. trainer는 RPC로 이 worker와 대화한다. 이것이 async와 partial-rollout을 가능하게 만든다.

**SPMD-sync 퇴역(PR #4411).** 2026년 1분기까지 verl에는 세 번째 rollout backend가 있었다. trainer와 같은 위치에서 SPMD mode로 vLLM을 실행하던 `vllm_rollout.py`의 `ServerAdapter`다. PR #4411은 이것을 async server로 대체했다. 예전 `generate_sequences` 메서드는 이제 예외를 던진다([[verl-rollout]], `vllm_rollout.py` 약 198–214행).

```python
def generate_sequences(self, prompts):
    raise NotImplementedError(
        "SPMD vLLM mode was retired in PR #4411; use AsyncLLMEngine via vllm_async_server."
    )
```

이를 대체한 것은 다음과 같다. `vllm_async_server.py`는 vLLM의 `AsyncLLMEngine`을 사용하고, 요청별 `generate(..., priority=...)`를 노출하며, paused-state weight broadcast hook을 구현한다. 트레이드오프는 throughput과 partial-rollout 지원은 얻고, 단일 프로세스 trainer의 디버깅 용이성은 잃는 것이다. DeepSeek R1 재현을 포함한 production recipe는 이제 async를 요구한다.

모듈별 파일 경로와 주요 함수는 클릭 가능한 모듈 트리 [figures/verl-structure.html](figures/verl-structure.html)을 보라.

## §2 `compute_policy_loss_vanilla` — 한 줄씩 보기

이것은 교과서적인 PPO-clip objective에 세 가지 중요한 부가 요소(asymmetric clip, dual-clip, rollout IS)를 더한 것이다. [[verl-ppo-loss]]에서 가져온 전체 인용(`verl/trainer/ppo/core_algos.py`, 대략 1080–1140행):

```python
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob, log_prob, advantages, response_mask,
    loss_agg_mode="token-mean", config=None, rollout_is_weights=None,
):
    clip_ratio = config.clip_ratio
    clip_ratio_low  = config.clip_ratio_low  or clip_ratio
    clip_ratio_high = config.clip_ratio_high or clip_ratio
    clip_ratio_c    = config.get("clip_ratio_c", 3.0)

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, -20.0, 20.0)
    ratio  = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)    # K1 monitor

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)            # pessimistic max

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)           # dual-clip floor
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights                      # vLLM-vs-actor IS
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask,
                       loss_agg_mode=loss_agg_mode, **config.global_batch_info)
    return pg_loss, {"actor/pg_clipfrac": ..., "actor/ppo_kl": ppo_kl, ...}
```

넘어가기 전에 내재화해야 할 다섯 가지가 있다.

1. **Asymmetric clip.** `clip_ratio_low`와 `clip_ratio_high`는 분리되어 있다. DAPO는 0.2 / 0.28을 쓴다. 위쪽을 더 느슨하게 두면 드문 positive-advantage token에서도 exploration이 살아남는다. 논문 [[ppo]]는 symmetric ε=0.2를 쓰며, verl은 `clip_ratio_low is None`이면 그 형태로 접는다.
2. **K1은 regularizer가 아니라 monitor다.** `ppo_kl = mean(-Δlogp)`는 [[kl-control-rlhf]]의 K1 estimator다. 이것은 `actor/ppo_kl`로 기록될 뿐 loss에 더해지지 않는다. KL-to-ref는 다른 곳에 있다(§5).
3. **Dual-clip은 `advantages < 0`일 때만 발동한다.** negative-advantage token에서 ratio가 폭발할 때, 즉 나쁜 action에 대해 policy가 rollout에서 너무 멀리 drift했을 때 `pg_losses3 = -A · c`가 floor 역할을 하여 gradient blow-up을 막는다. Ye et al. 2020의 아이디어다.
4. **Rollout IS 보정.** `rollout_is_weights = exp(logπ_train − logπ_rollout)`가 token별로 계산되어 loss에 곱해진다. 이것은 bf16(vLLM)과 fp32(actor)의 logprob 불일치를 보정하는 TIS / iCEPO patch다. sync mode에서는 비어 있고, async에서는 핵심적이다([[async-rollout]]).
5. **`agg_loss`는 parameterized되어 있다.** `token-mean`(기본값), `seq-mean-token-sum`(Dr.GRPO), `seq-mean-token-mean`(length-normalized) 중 하나다. 위의 같은 대수식도 이 flag에 따라 다른 gradient가 된다. [[dr-grpo]]의 length-bias 이야기는 전부 이 함수 호출 하나 안에 숨어 있다.

이 loss에는 entropy term이 *없다*. Entropy는 `workers/actor/*`에서 `verl_F.entropy_from_logits`로 별도 계산되며, 선택적으로 `entropy_loss` registry hook을 통해 regularize된다([[entropy-logging-patterns]] 참조).

---

## §3 GRPO advantage — group baseline이 만들어지는 방식

`compute_grpo_outcome_advantage`([[verl-grpo]], `core_algos.py` 약 290–335행)는 *advantage 단계만* 담당한다. 위의 같은 `compute_policy_loss_vanilla`가 재사용된다. GRPO는 "critic이 없고 group-relative advantage를 쓰는 PPO"다.

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards, response_mask, index,
    epsilon=1e-6, norm_adv_by_std_in_grpo=True, config=None,
):
    scores = token_level_rewards.sum(dim=-1)                  # outcome reward per rollout
    id2score = defaultdict(list)
    for i in range(len(scores)):
        id2score[index[i]].append(scores[i])
    id2mean, id2std = {}, {}
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx], id2std[idx] = torch.tensor(0.0), torch.tensor(1.0)
        else:
            tens = torch.stack(id2score[idx])
            id2mean[idx], id2std[idx] = tens.mean(), tens.std()
    for i in range(len(scores)):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]          # Dr.GRPO
    scores = scores.unsqueeze(-1) * response_mask              # broadcast to all tokens
    return scores, scores                                      # (advantages, returns)
```

중요한 점 네 가지:

1. **Group identity는 `index`를 통해 들어온다.** 배치의 각 rollout은 prompt-id를 들고 있다. rollout i의 advantage는 `(r_i − mean_over_group) / std_over_group`다. G = 8이면 [[grpo]]의 DeepSeekMath Eq. 3 신호와 정확히 일치한다.
2. **`(advantages, returns)`는 같은 tensor다.** critic이 없다. `vf_coef`는 0이고, `compute_value_loss`는 실행되지 않는다. 같은 모델 크기에서 GRPO의 memory footprint가 PPO의 약 절반인 이유도 이것이다.
3. **Dr.GRPO는 boolean 하나다.** `norm_adv_by_std_in_grpo=False`는 std denominator를 제거한다. [[dr-grpo]]의 unbiased variant가 정확히 이것이다. 다른 Dr.GRPO 수정인 1/|o_i| length normalization은 직교하는 문제이며 `agg_loss`(`seq-mean-token-sum` vs `seq-mean-token-mean`) 안에 있다.
4. **KL이 들어오는 위치.** *여기가 아니다.* 논문 [[grpo]]의 GRPO는 loss 안에 β·KL을 둔다. verl은 의도적으로 그렇게 하지 않는다. 같은 `core_algos.py`의 `kl_penalty(...)`를 통해 `token_level_rewards`가 이 함수에 도달하기 *전에* token별 reward에서 β·KL을 뺀다. 이는 [[kl-control-rlhf]]의 reward-shaping 관례와 일치하며, K3를 loss에 직접 더하는 TRL-GRPO와 다르다.

**GRPO-Pass@k** variant(`compute_grpo_passk_outcome_advantage`, 498–550행)는 각 group에서 가장 좋은 response에만 `(r_max − r_second_max)/σ` credit을 준다. pass@1이 아니라 pass@k를 최적화할 때 유용하다. 같은 registry, 다른 entry다.

---

## §4 Rollout: HFRollout(reference) vs async vLLM

Rollout은 wall-clock을 지배한다(보통 ≥70%). verl은 두 backend를 제공하며, production recipe는 (2)를 사용한다.

### 4.1 HFRollout — debug 경로

`verl/workers/rollout/hf_rollout.py` 40–125행([[verl-rollout]]):

```python
class HFRollout(BaseRollout):
    @torch.no_grad()
    def _generate_minibatch(self, prompts):
        generation_config = GenerationConfig(
            do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature,
        )
        param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False) \
                    if isinstance(self.module, FSDP) else contextlib.nullcontext()
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx, attention_mask=attention_mask,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                generation_config=generation_config,
                use_cache=True, return_dict_in_generate=True,
            )
```

이것이 correctness-only인 이유는 HF `.generate`가 FSDP-aware하지 않기 때문이다. 그래서 `summon_full_params`가 모든 parameter를 모든 rank에 unshard한다. 한 머신에서 1B 모델 batch 4라면 괜찮지만, 64 GPU에서 70B 모델이라면 치명적이다. async engine이 diverge했다고 의심될 때 `HFRollout`을 사용하라. side-by-side logprob 비교를 실행해 bug를 국소화한다.

### 4.2 Async vLLM — production 경로

`verl/workers/rollout/vllm_rollout/vllm_async_server.py` 약 440–525행([[verl-rollout]]):

```python
async def generate(self, prompt_ids, sampling_params, request_id,
                   image_data=None, video_data=None, priority=0):
    max_tokens = min(self.config.response_length,
                     self.config.max_model_len - len(prompt_ids))
    sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
    prompt = TokensPrompt(prompt_token_ids=prompt_ids, ...)
    lora_request = LoRARequest(...) if self.lora_as_adapter else None
    generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params,
                                     request_id=request_id, priority=priority,
                                     lora_request=lora_request)
    async for output in generator:
        final_res = output
    return TokenOutput(token_ids=final_res.outputs[0].token_ids,
                       log_probs=[...], ...)
```

여기서 노출되는 다섯 lever는 각각 production requirement다.
- **tokens-in-tokens-out.** server 안에 tokenizer가 없다. multi-turn tool-use loop가 깔끔해진다.
- **`priority`.** 새로 re-weight된 request가 queue 앞쪽으로 이동한다. trainer가 straggler를 기다리지 않는 *partial-rollout RL*(verl의 "continuous batching RL" 글)을 가능하게 한다([[async-rollout]]).
- **`max_tokens` 3-layer clamp.** 사용자 override → 전역 `response_length` → context-window residual. silent truncation을 막는다.
- **LoRA as adapter.** trainer는 adapter weight만 broadcast한다(GB가 아니라 MB). engine은 vLLM의 `LoRARequest`로 이를 load한다.
- **Paused-state weight broadcast.** `engine to paused state` block(약 628행)은 weight update 전에 새 generate를 막고 in-flight request를 drain한다. 그래서 mixed weights로 decode하지 않는다.

### 4.3 Trade table

| 관심사 | HFRollout | async vLLM server |
|------------------------|----------------------------|--------------------------------------|
| Throughput (7B, 8×H100)| ~1× (baseline)             | ~5–8× (continuous batching)          |
| FSDP aware             | 아니오(`summon_full_params`) | 예(별도 worker, 자체 weights) |
| Partial rollout        | 아니오 | 예(`priority` 경유) |
| Weight broadcast       | In-place(sync 불필요) | Paused-state + IS correction 필요 |
| Train-rollout logprob  | 정확함(같은 forward) | Drift(`vllm_kl` metric, [[entropy-logging-patterns]]) |
| Multi-turn tool use    | 어색함(retokenize 필요) | 자연스러움(tokens-in-tokens-out) |
| 사용할 때 | numerical bug 디버깅 | 그 밖의 모든 경우 |

---

## §5 Entropy + KL logging — verl이 제공하는 것과 추가할 것

[[entropy-logging-patterns]]는 cross-framework 그림을 정리한다. verl의 기본값은 이름 기준으로 다음과 같다.

- `actor/ppo_kl` — `compute_policy_loss_vanilla` 안에서 계산되는 K1 estimator `mean(logπ − logπ_old)`. monitor이며 regularizer가 아니다.
- `actor/entropy` — `workers/actor/*`에서 `verl_F.entropy_from_logits`로 계산하는 true categorical entropy `logsumexp(logits) − Σ p·logp`. 저렴한 `(−logp).mean()` proxy가 아니다.
- `actor/pg_clipfrac` — clipped branch가 이긴 token 비율. 1까지 치솟으면 ratio가 trust region 밖으로 터졌다는 뜻이다.
- `actor/pg_clipfrac_lower` — dual-clip floor가 발동한 비율(`advantages < 0`일 때만 가능).
- `kl_coef` + reward-shaping KL — verl은 token별 reward에서 `β · kl_penalty(logp, ref_logp, mode)`를 뺀다. 여기서 `mode ∈ {k1, k2, k3}`다. [[entropy-logging-patterns]]의 penalty function:

```python
def kl_penalty(logprob, ref_logprob, kl_penalty):
    if kl_penalty == "k1":  return logprob - ref_logprob
    if kl_penalty == "k2":  return 0.5 * (logprob - ref_logprob) ** 2
    if kl_penalty == "k3":                                          # Schulman unbiased, ≥0
        diff = ref_logprob - logprob
        return torch.exp(diff) - diff - 1
```

기본 recipe([[kl-control-rlhf]], [[grpo]]): `β ≈ 0.04`와 `k3`. K3는 항상 non-negative이므로 `actor/kl_loss` 곡선을 sign check 없이 읽을 수 있다.

**verl이 제공하는 것 vs production RL team이 추가하는 것.** 기본 dashboard에는 없지만, `workers/actor/*`에서 기존 `metrics: dict` return channel로 모두 추가할 수 있는 metric 다섯 가지:

1. **`rollout_kl = mean(logπ_actor − logπ_rollout)`** — vLLM-vs-actor drift. OpenRLHF의 `vllm_kl`과 같다. 20 consecutive step 동안 0.1 nats를 넘으면 alert.
2. **Per-bucket entropy** — prompt difficulty별 stratification. global entropy는 hard prompt에서 mode collapse가 일어나는 신호를 숨긴다.
3. **`clipfrac_positive_only`** — `pg_clipfrac`를 advantage sign별로 분리한다. upside-clip frac은 ε_high가 너무 빡빡하다는 뜻이고, downside는 ratio가 터지고 있다는 뜻이다.
4. **Held-out set에서 reward over ref reward:** `Δreward = reward_π − reward_π_ref`. training reward가 오르는데 Δreward가 음수가 되면 reward hacking을 일찍 잡은 것이다(RL track의 [[reward-hacking-taxonomy]]와 연결).
5. **Sequence-level IS histogram** — rollout별 `exp(Σ_t logπ_actor − logπ_rollout)`. tail이 seq-mask-tis가 발동하는 곳이다.

---

## §6 처음 읽는 사람을 위한 순회 경로

verl은 이 순서로 읽어라. 각 단계는 하나의 excerpt에 대응한다. (1) `core_algos.py::compute_policy_loss_vanilla` → [[verl-ppo-loss]]; (2) `core_algos.py::compute_grpo_outcome_advantage` → [[verl-grpo]]; (3) `core_algos.py::kl_penalty` → [[entropy-logging-patterns]]; (4) `fsdp_sft_trainer.py` — RL detour 없는 FSDP plumbing, `ppo_trainer.py` 전에 읽을 것; (5) `ppo_trainer.py` — 전체 rollout→logprob→advantage→loss sweep; (6) `hf_rollout.py` → [[verl-rollout]]; (7) `vllm_async_server.py::generate` → [[verl-rollout]] + [[async-rollout]].

---

## Connections

- **ch-53** — PPO theory. verl의 `compute_policy_loss_vanilla`는 이를 직접 구현한 것이다.
- **ch-54** — framework landscape(verl / OpenRLHF / TRL). 이 장은 verl을 구체적으로 파고든다.
- **ch-56** — OpenRLHF Internals. 같은 대수를 쓰지만 KL을 reward controller로 통과시키고 async에 Ray queue를 사용한다.
- **ch-46** — RL-track lab은 단일 파일 `_compute_loss` 때문에 `trl.GRPOTrainer`를 사용했다. 70B scale에서는 verl을 쓸 것이다.
- **ch-40 / [[grpo]]** — Eq. 3을 `compute_grpo_outcome_advantage`가 구현한 논문.
- **ch-43 / [[entropy-mechanism-llm-rl]]** — `actor/entropy` collapse signal의 mechanism과 추가 per-bucket entropy metric이 중요한 이유.
- **[[async-rollout]]** — async vLLM server의 architectural grounding.
- **[[kl-control-rlhf]]** — verl이 in-loss KL 대신 reward-shaping KL을 제공하는 이유.

## Further reading

- [[verl-ppo-loss]] — `compute_policy_loss_vanilla` 전체 본문 + asymmetric/dual-clip/IS 대수.
- [[verl-grpo]] — group-advantage code + `norm_adv_by_std_in_grpo` Dr.GRPO toggle.
- [[verl-rollout]] — `HFRollout` + async server; PR #4411 retirement.
- [[entropy-logging-patterns]] — verl vs OpenRLHF vs TRL metric table + `kl_penalty` k1/k2/k3.
- [[async-rollout]] — HybridFlow design, priority / paused-state hooks, IS correction, V-trace lineage.
- [[ppo]] — Schulman 2017. verl이 구현하는 clipped surrogate.
- [[grpo]] — DeepSeekMath Eq. 3. `compute_grpo_outcome_advantage`와 정확히 일치한다.
- [[dr-grpo]] — `norm_adv_by_std_in_grpo=False` + Dr.GRPO aggregation이 length-unbiased default인 이유.
- [[kl-control-rlhf]] — Stiennon/Ouyang/Korbak framework; k3 estimator math.

## Companion visualization

**[figures/verl-structure.html](figures/verl-structure.html)** — verl을 위한 self-contained clickable module tree. 네 개의 top-level pane(`core` / `trainer` / `rollout` / `workers+models`)으로 구성되어 있다. module card를 클릭하면 파일 경로, 주요 함수, training step에서 맡는 역할을 볼 수 있다. 첫 실제 debug session 전에 사용해 file-tree lookup이 반사적으로 되게 하라.
