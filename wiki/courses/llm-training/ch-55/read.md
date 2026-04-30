<!-- chapter: ch-55
     track: infra
     kind: content
     title: verl Internals
     deps: [ch-54]
     sources: [[verl-ppo-loss]], [[verl-grpo]], [[verl-rollout]], [[entropy-logging-patterns]], [[async-rollout]], [[ppo]], [[grpo]], [[dr-grpo]], [[kl-control-rlhf]]
     figures: figures/verl-structure.html
-->

# Chapter 55 — verl Internals

> **Core insight.** verl is organized around two orthogonal registries — advantage estimators (`@register_adv_est`) and policy losses (`@register_policy_loss`) — so that PPO, GRPO, Dr.GRPO, GSPO and their descendants are *configuration choices*, not separate code paths. The hot loop is minimal: `core_algos.py` holds ~100 lines of clipped-surrogate algebra, `ppo_trainer.py` orchestrates the rollout→logprob→advantage→update sweep, and `vllm_async_server.py` owns the GPU-time hog (rollout). Everything else — FSDP sharding, weight broadcast, LoRA adapters, MoE routing capture — plugs into that spine. Once you can trace a single token of `advantage` from `compute_grpo_outcome_advantage` through `compute_policy_loss_vanilla` into `backward()`, the rest of the repo is skinnable in an afternoon.
>
> **Guideline.** Read verl *top-down from the registry, not bottom-up from the trainer.* Start at `@register_policy_loss("vanilla")` and `@register_adv_est(AdvantageEstimator.GRPO)` in `verl/trainer/ppo/core_algos.py`; those two decorated functions are 100% of the algebra. Only after you understand the registry should you dive into `ppo_trainer.py` (ordering), `fsdp_sft_trainer.py` (the no-RL baseline trainer), and the async rollout server. When debugging a run, remember where the KL lives — verl puts KL in the *reward*, never in the loss, unlike TRL-GRPO.

---

## §1 Repo tour — the file tree that actually runs

verl's production RL code lives in four directories. Everything else is configs, docs, tests, or experiments.

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

Two design decisions dominate the layout:
- **Registry, not inheritance.** Adding GSPO or Dr.GRPO doesn't require subclassing a trainer — it requires a new `@register_adv_est(...)` function + a config flag. `core_algos.py` is the only file you edit for new algebra.
- **Rollout is a worker, not a trainer callback.** A separate Ray worker-group wraps vLLM; the trainer talks to it over RPC — this is what makes async + partial-rollout possible.

**SPMD-sync retirement (PR #4411).** Until 2026-Q1, verl shipped a third rollout backend: `ServerAdapter` in `vllm_rollout.py`, which ran vLLM in SPMD mode colocated with the trainer. PR #4411 replaced it with the async server; the old `generate_sequences` method now raises ([[verl-rollout]], `vllm_rollout.py` ≈ lines 198–214):

```python
def generate_sequences(self, prompts):
    raise NotImplementedError(
        "SPMD vLLM mode was retired in PR #4411; use AsyncLLMEngine via vllm_async_server."
    )
```

What replaced it: `vllm_async_server.py` uses vLLM's `AsyncLLMEngine`, exposes per-request `generate(..., priority=...)`, and implements a paused-state weight broadcast hook. The trade is throughput + partial-rollout support (gained) vs debuggability of a single-process trainer (lost). Production recipes including DeepSeek R1 reproductions now require async.

See the clickable module tree at [figures/verl-structure.html](figures/verl-structure.html) for file paths and key functions per module.

## §2 `compute_policy_loss_vanilla` — line by line

This is the textbook PPO-clip objective, plus three load-bearing extras (asymmetric clip, dual-clip, rollout IS). Full quote from [[verl-ppo-loss]] (`verl/trainer/ppo/core_algos.py`, ~lines 1080–1140):

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

Five things to internalize before moving on:

1. **Asymmetric clip.** `clip_ratio_low` vs `clip_ratio_high` separate — DAPO uses 0.2 / 0.28. The upside looseness keeps exploration alive on rare positive-advantage tokens; the paper [[ppo]] uses the symmetric ε=0.2, and verl collapses to that when `clip_ratio_low is None`.
2. **K1 is a monitor, not a regularizer.** `ppo_kl = mean(-Δlogp)` is the K1 estimator from [[kl-control-rlhf]]; it is logged as `actor/ppo_kl` and never added to the loss. KL-to-ref lives elsewhere (§5).
3. **Dual-clip fires only when `advantages < 0`.** When the ratio explodes on a negative-advantage token (policy drifted too far from rollout on a bad action), `pg_losses3 = -A · c` is a floor that prevents gradient blow-up. This is Ye et al. 2020.
4. **Rollout IS correction.** `rollout_is_weights = exp(logπ_train − logπ_rollout)` per-token, multiplied into the loss — this is the TIS / iCEPO patch for the bf16(vLLM) vs fp32(actor) logprob mismatch. Empty in sync mode; load-bearing in async ([[async-rollout]]).
5. **`agg_loss` is parametric.** `token-mean` (default), `seq-mean-token-sum` (Dr.GRPO), or `seq-mean-token-mean` (length-normalized). The same algebra above becomes a different gradient depending on this flag — the length-bias story from [[dr-grpo]] hides entirely in this one function call.

The loss contains *no* entropy term. Entropy is computed separately in `workers/actor/*` via `verl_F.entropy_from_logits` and optionally regularized via the `entropy_loss` registry hook (see [[entropy-logging-patterns]]).

---

## §3 GRPO advantage — how the group baseline is built

`compute_grpo_outcome_advantage` ([[verl-grpo]], `core_algos.py` ~lines 290–335) is *only* the advantage step. The same `compute_policy_loss_vanilla` above is reused — GRPO is "PPO with no critic and a group-relative advantage":

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

Four things that matter:

1. **Group identity comes in via `index`.** Each rollout in the batch carries its prompt-id. The advantage of rollout i is `(r_i − mean_over_group) / std_over_group`. With G = 8 this gives exactly the DeepSeekMath Eq. 3 signal from [[grpo]].
2. **`(advantages, returns)` are the same tensor.** There is no critic. `vf_coef` is 0, and `compute_value_loss` never runs. This is also why the GRPO memory footprint is ~half PPO's at equal model size.
3. **Dr.GRPO is one boolean.** `norm_adv_by_std_in_grpo=False` drops the std denominator — exactly the unbiased variant from [[dr-grpo]]. The 1/|o_i| length normalization, which is the other Dr.GRPO fix, is orthogonal and lives inside `agg_loss` (`seq-mean-token-sum` vs `seq-mean-token-mean`).
4. **Where KL enters.** *Not here.* GRPO in the paper [[grpo]] has β·KL inside the loss. verl intentionally does *not*: it subtracts β·KL from the per-token reward *before* `token_level_rewards` reaches this function, via `kl_penalty(...)` in the same `core_algos.py`. This matches the reward-shaping convention of [[kl-control-rlhf]] and differs from TRL-GRPO, which adds K3 to the loss directly.

The **GRPO-Pass@k** variant (`compute_grpo_passk_outcome_advantage`, lines 498–550) credits only the best response in each group with `(r_max − r_second_max)/σ`; useful when you optimize pass@k rather than pass@1. Same registry, different entry.

---

## §4 Rollout: HFRollout (reference) vs async vLLM

Rollout dominates wall-clock (≥70% typically). verl ships two backends; production recipes use (2).

### 4.1 HFRollout — the debug path

`verl/workers/rollout/hf_rollout.py` lines 40–125 ([[verl-rollout]]):

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

Why this is correctness-only: HF `.generate` isn't FSDP-aware, so `summon_full_params` unshards every parameter to every rank — fine for a 1B model at batch 4 on one machine, catastrophic at 70B on 64 GPUs. Use `HFRollout` when you suspect the async engine has diverged; run a side-by-side logprob comparison to localize the bug.

### 4.2 Async vLLM — the production path

`verl/workers/rollout/vllm_rollout/vllm_async_server.py` ~lines 440–525 ([[verl-rollout]]):

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

The five levers this exposes, each a production requirement:
- **tokens-in-tokens-out.** No tokenizer in the server — makes multi-turn tool-use loops clean.
- **`priority`.** Newly re-weighted requests jump the queue; enables *partial-rollout RL* (verl's "continuous batching RL" post) where the trainer never waits on stragglers ([[async-rollout]]).
- **`max_tokens` 3-layer clamp.** User override → global `response_length` → context-window residual. Prevents silent truncation.
- **LoRA as adapter.** Trainer broadcasts only adapter weights (MB, not GB). Engine loads via vLLM's `LoRARequest`.
- **Paused-state weight broadcast.** `engine to paused state` block (≈ line 628) blocks new generates + drains in-flight before a weight update, so you never decode on mixed weights.

### 4.3 Trade table

| Concern                | HFRollout                  | async vLLM server                    |
|------------------------|----------------------------|--------------------------------------|
| Throughput (7B, 8×H100)| ~1× (baseline)             | ~5–8× (continuous batching)          |
| FSDP aware             | No (`summon_full_params`)  | Yes (separate worker, own weights)   |
| Partial rollout        | No                         | Yes (via `priority`)                 |
| Weight broadcast       | In-place (no sync needed)  | Paused-state + IS correction needed  |
| Train-rollout logprob  | Exact (same forward)       | Drift (`vllm_kl` metric, [[entropy-logging-patterns]]) |
| Multi-turn tool use    | Awkward (retokenize)       | Natural (tokens-in-tokens-out)       |
| Use it when            | Debugging a numerical bug  | Everything else                      |

---

## §5 Entropy + KL logging — what verl ships, what you'd add

[[entropy-logging-patterns]] lays out the cross-framework picture. verl's defaults, by name:

- `actor/ppo_kl` — the K1 estimator `mean(logπ − logπ_old)` computed inside `compute_policy_loss_vanilla`. Monitor, not a regularizer.
- `actor/entropy` — true categorical entropy `logsumexp(logits) − Σ p·logp` via `verl_F.entropy_from_logits` in `workers/actor/*`. Not the cheap `(−logp).mean()` proxy.
- `actor/pg_clipfrac` — fraction of tokens where the clipped branch won; spiking to 1 means ratio has blown out the trust region.
- `actor/pg_clipfrac_lower` — fraction where dual-clip floor fired (only possible when `advantages < 0`).
- `kl_coef` + reward-shaping KL — verl subtracts `β · kl_penalty(logp, ref_logp, mode)` from the per-token reward, where `mode ∈ {k1, k2, k3}`. The penalty function from [[entropy-logging-patterns]]:

```python
def kl_penalty(logprob, ref_logprob, kl_penalty):
    if kl_penalty == "k1":  return logprob - ref_logprob
    if kl_penalty == "k2":  return 0.5 * (logprob - ref_logprob) ** 2
    if kl_penalty == "k3":                                          # Schulman unbiased, ≥0
        diff = ref_logprob - logprob
        return torch.exp(diff) - diff - 1
```

Default recipe ([[kl-control-rlhf]], [[grpo]]): `k3` with `β ≈ 0.04`. K3 is always non-negative, which makes `actor/kl_loss` curves readable without a sign check.

**What verl ships vs what production RL teams add.** Five metrics not on the default dashboard, all addable in `workers/actor/*` via the existing `metrics: dict` return channel:

1. **`rollout_kl = mean(logπ_actor − logπ_rollout)`** — vLLM-vs-actor drift; equivalent to OpenRLHF's `vllm_kl`. Alert at > 0.1 nats for 20 consecutive steps.
2. **Per-bucket entropy** — stratified by prompt difficulty. Global entropy hides the mode-collapse-on-hard-prompts signature.
3. **`clipfrac_positive_only`** — `pg_clipfrac` split by sign of advantage. Upside-clip frac → ε_high too tight; downside → ratios blowing out.
4. **Reward over ref reward** on a held-out set: `Δreward = reward_π − reward_π_ref`. Rising training reward with Δreward going negative is reward hacking caught early (connects to [[reward-hacking-taxonomy]] from the RL track).
5. **Sequence-level IS histogram** — `exp(Σ_t logπ_actor − logπ_rollout)` across rollouts. The tail is where seq-mask-tis triggers.

---

## §6 A tour path for first-time readers

Read verl in this order — each step maps to one excerpt: (1) `core_algos.py::compute_policy_loss_vanilla` → [[verl-ppo-loss]]; (2) `core_algos.py::compute_grpo_outcome_advantage` → [[verl-grpo]]; (3) `core_algos.py::kl_penalty` → [[entropy-logging-patterns]]; (4) `fsdp_sft_trainer.py` — FSDP plumbing without the RL detour, read before `ppo_trainer.py`; (5) `ppo_trainer.py` — the full rollout→logprob→advantage→loss sweep; (6) `hf_rollout.py` → [[verl-rollout]]; (7) `vllm_async_server.py::generate` → [[verl-rollout]] + [[async-rollout]].

---

## Connections

- **ch-53** — PPO theory; verl's `compute_policy_loss_vanilla` is the direct implementation.
- **ch-54** — framework landscape (verl / OpenRLHF / TRL); this chapter drills into verl specifically.
- **ch-56** — OpenRLHF Internals; uses the same algebra but threads KL through a reward controller and uses Ray queues for async.
- **ch-46** — the RL-track lab used `trl.GRPOTrainer` because of its single-file `_compute_loss`; verl is what you'd use at 70B scale.
- **ch-40 / [[grpo]]** — the paper whose Eq. 3 `compute_grpo_outcome_advantage` implements.
- **ch-43 / [[entropy-mechanism-llm-rl]]** — mechanism for the `actor/entropy` collapse signal and why the extra per-bucket entropy metric matters.
- **[[async-rollout]]** — architectural grounding for the async vLLM server.
- **[[kl-control-rlhf]]** — why verl ships reward-shaping KL instead of in-loss KL.

## Further reading

- [[verl-ppo-loss]] — `compute_policy_loss_vanilla` full body + asymmetric/dual-clip/IS algebra.
- [[verl-grpo]] — group-advantage code + `norm_adv_by_std_in_grpo` Dr.GRPO toggle.
- [[verl-rollout]] — `HFRollout` + async server; PR #4411 retirement.
- [[entropy-logging-patterns]] — verl vs OpenRLHF vs TRL metric table + `kl_penalty` k1/k2/k3.
- [[async-rollout]] — HybridFlow design, priority / paused-state hooks, IS correction, V-trace lineage.
- [[ppo]] — Schulman 2017; the clipped surrogate verl implements.
- [[grpo]] — DeepSeekMath Eq. 3; matches `compute_grpo_outcome_advantage` exactly.
- [[dr-grpo]] — why `norm_adv_by_std_in_grpo=False` + Dr.GRPO aggregation is the length-unbiased default.
- [[kl-control-rlhf]] — Stiennon/Ouyang/Korbak framework; k3 estimator math.

## Companion visualization

**[figures/verl-structure.html](figures/verl-structure.html)** — self-contained clickable module tree for verl. Four top-level panes (`core` / `trainer` / `rollout` / `workers+models`); click any module card to see its file path, key functions, and the role it plays in a training step. Use before your first real debug session so the file-tree lookups become reflexes.
