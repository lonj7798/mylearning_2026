<!-- chapter: ch-54
     track: infra
     kind: content
     title: Rollout, Replay, Async Infrastructure
     deps: [ch-53]
     sources: [[best-of-n]], [[on-off-policy-rlhf]], [[replay-buffer-rlhf]], [[async-rollout]],
              [[minibatch-sharing-rl]], [[verl-rollout]], [[openrlhf-ppo]], [[trl-ppo]],
              [[policy-coverage-loss]], [[rejection-sampling-finetuning]]
     figures: figures/async-pipeline.html
     opens_track: infra (ch-54..ch-60)
-->

# 54장 — Rollout, Replay, Async Infrastructure

> **핵심 통찰.** LLM을 위한 RL은 loss 문제가 아니라 pipeline 문제다. [[dpo]] / [[ppo]] / [[grpo]]가 다섯 줄짜리 objective가 되고(ch-37..ch-40), reward signal이 wired되면(ch-42..ch-44), 남는 wall-time의 70%는 **rollout**이고, 남는 engineering risk의 90%는 **gradient가 data를 볼 때 data가 얼마나 stale한가**다. infra track은 여기서 시작해 그 pipeline 위에 네 축의 map을 깐다. *누가 생성하는가*(policy vs frozen teacher), *얼마나 fresh한가*(on-policy ↔ off-policy staleness `k`), *얼마나 reuse하는가*(no replay, prompt replay, trajectory replay), *얼마나 parallel한가*(sync, async queue-1, async queue-2, partial rollout). 모든 production stack, TRL, OpenRLHF, verl은 이 cube의 한 corner이며, corner 사이를 움직이는 데는 KL, variance, clipfrac으로 표시되는 비용이 있다.
>
> **가이드라인.** 새 RL run에서 첫 move는 algorithm choice가 아니다. (1) RM에 대해 Best-of-N을 시도해 KL-matched baseline ceiling을 얻는다([[best-of-n]]). (2) train해야 한다면 on-policy(iterative DPO 또는 PPO/GRPO)에서 시작한다. [[on-off-policy-rlhf]]가 offline DPO가 가져가는 gap의 약 80%가 순수 distribution shift임을 보이기 때문이다. (3) trajectory를 replay하지 마라. group-reward variance로 keyed된 *prompt*를 replay하라([[replay-buffer-rlhf]]). (4) run이 ~30 min을 넘으면 queue depth 1–2의 async([[async-rollout]])로 가고 `vllm_kl`을 watch하라. (5) group baseline이 noise가 되지 않도록 prompt당 rollout 수 `n`을 ≥ 4로 설정하라([[minibatch-sharing-rl]]). 이후 infra chapter(ch-55 verl, ch-56 OpenRLHF, ch-57 distributed training)는 이 다섯 move 중 *하나*를 deep dive한다.

---

## 1. 네 축의 map

이 장의 모든 것은 rollout → train loop의 네 축 중 하나에 놓인다.

| Axis | Endpoint A | Endpoint B | Paid by |
|---|---|---|---|
| Generator | policy π_t | frozen teacher π_ref / stronger model | distribution shift (DPO vs iterative DPO) |
| Freshness | on-policy (k=0) | off-policy (k=queue_depth + partial) | IS correction, ratio blow-up, `vllm_kl` |
| Reuse | no replay | prompt replay / trajectory replay | bias, ratio explosion, stale KL term |
| Parallelism | sync (trainer idle during rollout) | async queue + partial rollout | weight-broadcast pause, engine lock |

[[trl-ppo]]는 (π_t, k=0, no replay, sync)에 있다. reference corner다. [[openrlhf-ppo]] async와 [[verl-rollout]] vllm_async_server.py는 parallelism axis를 따라 움직이며 freshness axis에서 correction을 적용한다. [[replay-buffer-rlhf]]는 reuse axis를 추가한다. Figure [[figures/async-pipeline.html]]에서는 parallelism + freshness를 slide하면서 throughput vs staleness를 실시간으로 볼 수 있다.

---

## 2. Best-of-N — PPO가 넘어야 하는 baseline

어떤 RL run 전에든 [[best-of-n]]을 실행하라. Stiennon 2020(OpenAI TL;DR summarization)은 이후 모든 BoN analysis를 지배하는 closed form을 확립했다.

```
KL(BoN || base) = log N − (N−1)/N        # analytic, tight for well-calibrated RM
                ≈ log N   for large N
```

따라서 BoN-64는 KL ≈ ln 64 − 63/64 ≈ 3.17 nats에 있다. β를 설정해 attained KL이 ~3이 되도록 할 때 PPO가 받는 budget과 정확히 같다. 그 matched budget에서 Stiennon Figure 6은 BoN-64가 well-tuned PPO와 약 2 human-preference point 이내임을 보여 주며, **training cost는 0**이다. 이것이 PPO run이 engineering을 정당화하려면 넘어야 하는 ceiling이다. 종종 넘지 못한다. BoN-64는 deployment strategy로 자주 선택된다(Anthropic의 test-time compute work, Cohere의 chat model; [[best-of-n]] §Connections 참고).

**The ceiling.** Stiennon Figure 4의 BoN gold-vs-proxy curve는 최초로 문서화된 **reward-model overoptimization**이다. critical KL 이후에는 RM score가 계속 오르지만 human preference는 떨어진다. 이는 [[reward-model-overoptimization]](ch-41 §2)가 formalize한 같은 법칙이다. RM이 faithful할 때만 BoN은 N에 대해 monotone하다. `d* = α_bon / (2 β_bon)` nats^0.5 위에서는 argmax-over-N조차 degrade한다. 그러므로 BoN의 ceiling은 RM의 ceiling이다.

**RSFT — 한 단계 위.** [[rejection-sampling-finetuning]]은 BoN을 inference가 아니라 *training data*에 적용한 것이다. prompt마다 N개를 sample하고, filter하고, survivor에 SFT하며, optional로 iterate한다. Llama 2 post-training이 canonical recipe다. per-query inference cost 없이 BoN gain의 일부를 capture하지만 같은 RM ceiling에서 cap된다. RSFT를 "BoN cached into weights"로 다뤄라. SFT와 full RLHF 사이의 유용한 bridge다.

**Decision rule.** RM을 신뢰할 수 있고 inference에서 N = 16–64를 감당할 수 있다면 BoN이 move다. RM이 shaky하다면(overopt peak at `d ≈ 3`), BoN-8 plus tighter RM이 여전히 full PPO보다 보통 더 싸고 안전하다.

---

## 3. On-policy vs off-policy — 중요한 유일한 숫자는 `k`

[[on-off-policy-rlhf]](Tang 2024, DeepMind)는 오래된 논쟁을 정리했다. **offline DPO가 PPO보다 못한 이유는 DPO가 algorithmically weaker해서가 아니라 training data가 current policy와 다른 distribution에서 뽑히기 때문이다**. TL;DR / HH / GSM8K(Gemma-2B 및 7B)에서 gap decomposition은 다음과 같다.

- ≈ 80% distribution-shift contribution
- ≈ 20% variance-reduction contribution

Iterative DPO, 즉 각 step에서 π_t로 chosen/rejected pair를 sample하고, frozen RM으로 label을 붙이며, β=0.1로 DPO-update하고, π_0을 frozen reference로 유지하는 방식은 세 task 모두에서 **PPO와 match**하며 seed variance는 더 낮다. 교훈은 일반화된다.

```
gap(algorithm) = gap(distribution_shift) + gap(variance) + gap(algebra)
              ≈         0.8               +      0.2      +      ~0
```

PG vs closed-form의 algebra는 이 scale에서 중요하지 않다. *어떤 distribution에서 sample하느냐*가 중요하다.

**[[policy-coverage-loss]]가 이를 날카롭게 만든다.** Distribution shift는 scalar가 아니라 shape이다. source policy / dataset은 그 **support가** target-optimal policy의 support를 cover할 때만 유용하다. 올바른 region에 도달하는 policy를 유도할 수 있다면 imperfect reward model을 감당할 수 있다. Anthropic-HH에 대한 offline DPO가 어떤 task에는 도움이 되고 어떤 task에는 그렇지 않은 이유다. HH pair-distribution이 target의 action region을 cover하는지에 달려 있다. production에서는 transfer 전에 held-out reference에 대한 source-policy **win rate**를 측정하라.

**Staleness as a scalar.** async RL(§5)에서 off-policyness는 `k = queue_depth + partial_rollout_depth`로 parameterize된다. Tang의 "on-policy bonus"는 `k`에 대해 monotone하게 collapse한다. `k=1`(1-step queue)에서는 gap이 negligible하다. `k=5`에서는 offline-DPO 영역에 접근한다. 이것이 [[async-rollout]]이 queue depth를 1–2로 유지하고 `vllm_kl`을 canary로 monitor하는 이유다.

---

## 4. Replay — trajectory는 왜 깨지고 prompt는 왜 되는가

classical RL(DQN, Ape-X, R2D2)은 experience replay에 의존한다. LLM RL은 그렇지 않으며, [[replay-buffer-rlhf]]는 그 이유를 정확히 설명한다.

**PPO/GRPO에서 trajectory replay가 실패하는 이유**(네 mechanism이 compounding된다):

1. **Ratio explosion.** IS weight `exp(Σ_t Δ log π_t)`는 token-wise로 누적된다. 100-token response와 update당 0.01-nat drift per token이면, 약 20 update 안에 ratio가 어떤 clip bound도 넘는다. PPO clipfrac이 1에 붙고 gradient는 거의 all-zero가 된다.
2. **Stale advantages.** GRPO의 group baseline `μ_i = mean_k R_{i,k}`는 rollout time의 policy에 대해 계산된다. 어제의 advantage를 replay하면 어제의 baseline에 대해 train하게 되며, old self에 대한 reward hacking의 한 형태가 된다.
3. **Compute asymmetry.** LLM RL에서는 rollout이 update cost에 비해 싸다(vLLM prefill은 bandwidth-bound, forward+backward는 compute-bound). trajectory reuse로 절약되는 것이 적다.
4. **KL-to-ref mismatch.** per-token KL reward(§6)는 rollout time의 `π_ref`에 대해 계산된다. `π_ref`를 freeze한다면 괜찮지만 adapt한다면 replayed reward는 이제 wrong reference에 대한 것이 된다.

DeepSeek-R1 §3.2는 trajectory replay를 시도했다가 fresh rollout + larger batch로 포기했다고 보고한다. 확인된 negative result다.

**작동하는 것은 prompt-level replay다.** TRL의 experimental `GRPOWithReplayBufferTrainer`(`trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py`)는 `(prompt_ids, rewards, variance)`의 FIFO buffer를 유지한다.

```
# TRL prompt-replay schema (simplified from the real trainer)
@dataclass
class BufferEntry:
    prompt_ids: torch.Tensor
    rewards:    torch.Tensor          # (n,) per-rollout outcome rewards
    variance:   float                  # rewards.var().item()
    seen_steps: int
```

Sampling probability는 `p_i ∝ var_i + ε`다. reward가 모두 같은 prompt(all pass 또는 all fail)는 group variance가 0이므로 GRPO gradient에 zero contribution을 하고, replay에서 downweighted된다. old *completion*은 버린다. old *prompt*만 replay하고, engine은 current π_t 아래에서 fresh response를 resample한다. gradient에 old-policy content가 없으므로 IS correction이 필요 없다.

**Replay buffer에 대한 verdict.** OpenRLHF는 buffer를 ship하지 않는다(`rollout → train → discard`). verl의 `experience_makers/experience_buffer.py`는 PPO epoch를 위한 intra-step scratch pad이지 cross-step replay가 아니다. TRL의 것은 experimental이고 `p_replay=0.25` 뒤에 gated되어 있다. prompt-replay를 curriculum diagnostic(어려운 prompt를 두 번 보기)과 variance reduction으로 취급하라. off-policy correction trick이 아니다.

---

## 5. Async actor-learner — queue가 architecture다

Synchronous RL([[trl-ppo]] inner loop)은 machine의 절반을 낭비한다. rollout GPU는 trainer가 `backward()`를 실행하는 동안 idle하고, trainer는 rollout 동안 idle한다. [[async-rollout]](OpenRLHF + verl HybridFlow)은 bounded queue를 사이에 두고 두 process를 decouple해 1.6×–2.0× throughput을 회복한다. 그 대가로 bounded staleness `k ≤ queue_depth + partial_rollout_depth`를 지불한다.

**Architecture in prose.**

```
  prompts ─▶ [RolloutActor]───generate(π_{t−k})──▶ ┌────────────┐
               x R workers                          │ rollout_   │
               (vLLM AsyncLLMEngine)                │   queue    │
                                                    │  depth d=1 │
                                                    └─────┬──────┘
                                                          │
                                                  (FIFO pop, k≤d+partial)
                                                          ▼
                                          ┌─────────────────────────┐
                                          │   TrainingActor         │
                                          │   score → GAE / GRPO    │
                                          │   → clipped PG step     │
                                          │   → optimizer.step()    │
                                          └─────────┬───────────────┘
                                                    │ broadcast_weights
                                                    │ (holds vllm_lock)
                                                    ▼
                                          [engine.pause() — in-flight가
                                           drain되기를 기다리거나
                                           partial-rollout interrupt
                                           hook으로 interrupt]
                                                    │
                                                    └──▶ back to RolloutActor
```

**OpenRLHF primitives** (`openrlhf/trainer/ppo_trainer_async.py`에서):

- `rollout_queue`: capacity 1–2의 `ray.util.queue.Queue`.
- `rollout_slots`: `global_step` token을 carrying하는 companion backpressure queue(allowed in-flight rollout마다 slot 하나).
- `vllm_lock`: weight-broadcast와 generate를 serialize하는 `ray` asyncio lock.
- `strategy.args.train.partial_rollout_enable`: weight update 시 in-flight generate를 interrupt할 수 있게 한다(continuous-batching RL).

**verl primitives** (`verl/workers/rollout/vllm_rollout/vllm_async_server.py`에서):

- `engine.generate(..., priority=…)`: newly-weighted request가 queue를 jump하도록 하는 per-request priority.
- `engine to paused state` block(약 line 628): weight broadcast 전 hard-pause, 이후 resume.
- SPMD sync vLLM은 **retired**됨(PR #4411). 새 deployment는 async를 써야 한다. sync path는 `HFRollout`(reference, debug)뿐이다.
- Tokens-in / tokens-out — async server는 tokenizer를 건드리지 않는다(cheap multi-turn tool-use).

**Staleness bound와 canary.** `k = queue_depth + partial_rollout_depth`이며, 보통 1–2다. correction은 [[openrlhf-ppo]]의 `PolicyLoss.forward`에서 실행된다.

- Per-token truncated-IS weight `exp(log π_train − log π_rollout)`를 `[low, high]`로 clip(mode `tis`).
- seq-IS가 `[low, high]`를 벗어나는 sequence를 drop하는 sequence-level mask(`seq-mask-tis`).
- range 밖 token에 대한 per-token mask(`icepop`).
- seq-IS가 threshold를 넘는 entire rollout hard-cap drop.

`PolicyLoss.forward`가 `vllm_kl`로 반환하는 canary metric은 `masked_mean(rollout_log_probs − old_log_probs, action_mask)`다. Spec: `vllm_kl > 0.1`이 PPO clipfrac pegged at 1과 함께 나타나면 staleness가 IS correction이 처리할 수 있는 범위를 넘었다는 확인된 signal이다. queue depth가 커졌거나(producer가 consumer보다 빠름), vLLM fp-precision이 trainer와 다르다(bf16 rollout vs fp32 train).

**Ancestry.** 두 stack은 모두 IMPALA V-trace(Espeholt 2018)와 Ape-X / R2D2에서 내려온다. IS-weight clip은 V-trace의 ρ̄와 c̄와 *같은 mathematical object*다. 바뀐 것은 policy parameterization뿐이다.

**Companion.** [figures/async-pipeline.html](figures/async-pipeline.html)은 pipeline을 animate한다. queue depth, rollout/learner speed ratio, partial-rollout on/off를 slide하면 throughput + staleness(`k`) + `vllm_kl` proxy가 어떻게 evolve하는지 볼 수 있다.

---

## 6. Mini-batch sharing — B가 아니라 B × n

critic-free objective(GRPO, RLOO, REINFORCE++)에서 advantage는 **group-relative** statistic이다. baseline을 위해서만도 prompt당 ≥ 2 rollout이 필요하다. [[minibatch-sharing-rl]]은 ablation을 종합한다.

```
A_{i,k}^{GRPO} = (R_{i,k} − μ_i) / (σ_i + ε),   μ_i = mean_k R_{i,k}   (need n ≥ 2; stable n ≥ 4)
A_{i,k}^{RLOO} = R_{i,k} − (1/(n−1)) · Σ_{j≠k} R_{i,j}                 (variance ∝ 1/(n−1))
```

**Framework defaults — 확인됨.**

| Framework | Config knob | Default `n` | Batch recipe |
|---|---|---|---|
| verl | `actor_rollout_ref.rollout.n` | 8 | B=128 prompts × n=8 = 1024 seqs |
| TRL (GRPO) | `num_generations` | 8 | B=64 × n=8 = 512 seqs |
| OpenRLHF (PPO) | `n_samples_per_prompt` | 4 | B=128 × n=4 = 512 seqs |
| TRL (PPO, classic) | N/A (critic available) | 1 | B=256 × n=1 = 256 seqs |

**왜 `n`은 linear보다 싸게 증가하는가.** vLLM은 same-prompt request를 함께 batch하고 prompt-prefix forward pass를 공유한다. decode half만 `n`에 따라 scale한다. 1024-token prompt와 512-token response에서 `n=1`에서 `n=8`로 가도 wall time은 8×가 아니라 약 1.4×다.

**Per-prompt vs pooled advantages.** 두 pattern이 확인되어 있다.

- *Per-prompt* (GRPO, RLOO): 각 prompt group 안에서 `μ_i`, `σ_i`를 계산한다. prompt-difficulty bias를 제거한다(easy prompt는 baseline이 높고, hard prompt는 낮다). defensible default다.
- *Pooled*: global batch mean을 뺀다. REINFORCE++는 prompt가 충분히 homogeneous해 prompt-level variance가 noise일 때 이것을 쓴다. reduction 하나를 절약하지만 "hard prompt"와 "good response"를 혼동한다.

**n=1 special case.** `n=1`이면 group baseline이 없다. REINFORCE의 high variance를 받아들이거나 critic(PPO의 value head; [[trl-ppo]] `AutoModelForCausalLMWithValueHead`)을 가져와야 한다. critic-free family 전체(GRPO/RLOO/REINFORCE++)는 그 critic을 피하기 위해 존재한다. `n ≥ 4`가 그 대가다.

**Curve의 knee.** DeepSeekMath appendix ablation은 GRPO pass@1 gain이 MATH에서는 `n=16` 이후 plateau하지만 AIME에서는 계속 오른다는 것을 보인다. 더 어려운 task는 더 많은 rollout을 원한다. Lambert의 Interconnects sweep은 `n=8`이 pareto point라는 쪽으로 수렴한다.

---

## 7. Corner를 합치기 — 각 stack은 어디에 있는가

| Stack | Generator | Freshness | Reuse | Parallelism |
|---|---|---|---|---|
| [[trl-ppo]] (actor-critic) | π_t | k=0 | none | sync |
| TRL GRPO (default) | π_t | k=0 | none | sync + colocate vLLM |
| TRL `GRPOWithReplayBufferTrainer` | π_t | k=0 | **prompt replay** (var-weighted) | sync |
| TRL Online DPO | π_t | k=0 | none | sync |
| [[openrlhf-ppo]] async | π_{t−k} | k∈[1,2], IS-corrected | none | **async queue + partial** |
| [[verl-rollout]] (vllm_async_server) | π_{t−k} | k∈[1,2], IS-corrected, priority | none | **async + priority + partial** |
| DeepSeek-R1 production | π_t | k=0 | none (replay rejected §3.2) | sync, huge batch |
| BoN deployment (Anthropic, Cohere) | π_t | N/A (no gradient) | N/A | trivial parallelism |

infra track의 일은 ch-55..ch-60에서 이 corner들을 하나씩 걷는 것이다. ch-55는 verl end-to-end(ppo_loss, grpo, rollout, entropy logging)를 읽는다. ch-56은 OpenRLHF에 대해 같은 일을 한다. ch-57은 distributed training(FSDP, Ray)을 다룬다. ch-58은 rollout의 KV-cache reuse + speculative decoding을 다룬다. ch-59는 partial-rollout / continuous-batching RL이다. ch-60은 cost accounting이다. ch-54는 map이고, ch-55+는 terrain이다.

---

## Connections

- **ch-37 / ch-38 ([[dpo]])** — [[on-off-policy-rlhf]]가 dissect하는 offline-DPO baseline.
- **ch-39 ([[ppo]])** — `PolicyLoss.forward`가 계산하는 loss. async는 gradient가 아니라 *wrapper*다.
- **ch-40 ([[grpo]])** — group baseline이 `n ≥ 4`를 중요하게 만드는 이유.
- **ch-41 ([[reward-model-overoptimization]])** — BoN의 ceiling은 이 inverted-U다.
- **ch-43 ([[entropy-mechanism-llm-rl]])** — entropy-collapse는 async stack에서 `vllm_kl` spike로 가장 먼저 나타난다.
- **ch-55** — verl internals([[verl-rollout]] + [[verl-ppo-loss]] + [[verl-grpo]] deep dive).
- **ch-56** — OpenRLHF internals([[openrlhf-ppo]] deep dive, Ray actor topology 포함).
- **ch-57..ch-60** — distributed training, KV reuse, partial rollout, cost accounting.

## Further reading

- [[best-of-n]] — Stiennon 2020; BoN KL closed form; first overoptimization curve.
- [[on-off-policy-rlhf]] — Tang 2024; DPO-vs-PPO gap의 80/20 decomposition.
- [[replay-buffer-rlhf]] — trajectory-replay failure mode; TRL prompt-replay schema.
- [[async-rollout]] — OpenRLHF async + verl HybridFlow; IS correction and `vllm_kl`.
- [[minibatch-sharing-rl]] — RLOO 1/(n−1) variance; `n=8` defaults.
- [[verl-rollout]] — `vllm_async_server.py` line-by-line; around line 628의 engine-pause.
- [[openrlhf-ppo]] — `PolicyLoss.forward` body; `tis` / `icepop` / `seq-mask-tis`.
- [[policy-coverage-loss]] — "distribution shift"를 보는 올바른 lens로서의 coverage.
- [[rejection-sampling-finetuning]] — BoN-in-weights; SFT-side alternative.

## Companion visualization

**[figures/async-pipeline.html](figures/async-pipeline.html)** — animated async pipeline. Controls: **queue depth** `d` (1–4), **rollout:learner speed ratio** `ρ` (0.5–4), **partial-rollout** toggle, **IS-correction mode** (off / tis / icepop). Readouts: instantaneous throughput(rollouts/s), effective staleness `k`, estimated `vllm_kl` proxy, PPO clipfrac. d=1, ρ=2, partial-off에서 시작해 d를 올리면 throughput이 오르는 것을 보라. 그런 다음 IS off에서 `k`가 ~3을 넘으면 clipfrac이 1에 붙는 것을 보라. prose의 canary story(§5)가 여기서 kinesthetic해진다.
