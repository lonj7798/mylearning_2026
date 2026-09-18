<!-- chapter: ch-56
     track: infra
     kind: content
     title: OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls
     deps: [ch-55]
     sources: [[openrlhf-ppo]], [[openrlhf-ppo-recipe]], [[openrlhf-dpo]], [[openrlhf-dpo-recipe]], [[async-rollout]], [[openrlhf-entropy-debugging]], [[rlhf-instructgpt]], [[rls-razor]], [[ring-attention]], [[dpo]], [[tulu-3]], [[verl-ppo-loss]], [[verl-rollout]]
     figures: figures/openrlhf-ray.html, figures/ring-attention-sharding.html
     revised: 2026-09 (generality revision)
-->

# Chapter 56 — OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls

> **Core insight.** OpenRLHF separates three things that other stacks mix together: a small `nn.Module`
> loss that computes only the clipped surrogate, a reward-shaping step in the experience maker that adds
> the KL-to-reference term to per-token rewards, and a set of Ray actors that own the policy, critic,
> reference, reward model, and vLLM engines. At commit 64c1cc4, the KL term is added to the reward by
> default with a fixed coefficient of 0.01, and `--algo.kl.use_loss` moves it into the actor loss instead
> ([[openrlhf-ppo]], `train_ppo_ray.py` L419, L484–486; `ppo_actor.py` L296–314). The repository ships no
> mechanism for mixing pretraining or SFT loss into RL, so the KL term, the reference model, and in-loop
> evaluation are the only capability-retention controls it exposes.
>
> **Guideline.** When an RL run must not lose capability outside its reward's domain, keep the KL term on
> with a non-zero `--algo.kl.init_coef`, set `--eval.dataset` to a broad held-out set so `eval_*_pass1` is
> logged during training, and record KL to the reference alongside it, because KL to the base policy
> measured on the training task predicts how much prior-task performance is lost ([[rls-razor]] §4).
> When the reward's domain is the only thing that matters and a later stage will repair breadth, β may be
> lowered, but then in-loop broad evaluation is the only remaining detector. Do not expect the KL term to
> substitute for data mixing: in InstructGPT's 1.3B sweep, raising β to 2.0 — 100× the default — still did
> not recover the public-NLP regressions, while mixing pretraining gradients at γ ≥ 20 did
> ([[rlhf-instructgpt]] App. E.6, Fig. 33–34).

---

## Why this chapter matters for a general-purpose model

This chapter sits at the RL stage of the pipeline (pre-training → mid-training → SFT → preference
optimization → RL → evaluation), and it reads one specific implementation of that stage. A paper's
objective states intent; the trainer's code decides whether the KL term is inside the advantage or inside
the loss, whether negative-advantage tokens are bounded, which prompts survive to the optimizer, and what
is measured while the run is in progress. Three of those decisions change breadth of capability rather
than task score, and all three are visible in OpenRLHF's source:

1. **Where the KL to the reference is applied, and with what coefficient.** This is the only explicit
   anchor to the pre-RL policy in the loop.
2. **Which prompts reach the optimizer.** `--algo.dynamic_filtering_enable` discards prompt groups whose
   mean reward is outside a band, which makes the effective training distribution a moving subset of the
   dataset.
3. **What is evaluated during training, and on which metric the best checkpoint is chosen.** OpenRLHF
   computes per-datasource pass@1 and pass@n in the loop, and auto-selects a single `eval_*_pass1` key for
   best-checkpoint tracking.

Chapter ch-55 read verl with the same questions. The two stacks disagree on defaults more than on
mechanisms, so a run that passes no flag inherits a different configuration under each framework.

---

## §1 Repository layout and the unit of composition

The file tree below is the `openrlhf/` package at `main` commit b117b2b, read on 2026-09-14 through the
GitHub tree API. Line loci in the rest of this chapter refer to commit 64c1cc4 (2026-04-19), the commit
the source cards pin ([[openrlhf-ppo]] Verification).

```
openrlhf/
  cli/        lora_combiner.py  serve_rm.py  train_dpo.py  train_ppo_ray.py  train_rm.py  train_sft.py
  datasets/   prompts_dataset.py  reward_dataset.py  sft_dataset.py  utils.py
  models/     actor.py  loss.py  model.py  ring_attn_utils.py  utils.py
  trainer/    dpo_trainer.py  ppo_trainer.py  ppo_trainer_async.py  rm_trainer.py  sft_trainer.py
    ppo_utils/  experience.py  experience_maker.py  kl_controller.py  length_penalty.py
                replay_buffer.py  samples_generator.py
    ray/        launcher.py  ppo_actor.py  ppo_critic.py  utils.py  vllm_engine.py  vllm_worker_wrap.py
  utils/      agent.py  config.py  deepspeed/  distributed_sampler.py  distributed_util.py
              logging_utils.py  loss_utils.py  math_utils.py  seqlen_balancing.py  utils.py
              vlm_utils.py
```

(`__init__.py` files are omitted; every other `.py` file in the package is listed.)

Four properties of this layout matter for the rest of the chapter. **There are four training entry
points**, and only `train_ppo_ray.py` is an RL entry point: there is no non-Ray PPO script and no
knowledge-distillation trainer in the package. **`models/loss.py` holds loss modules only** —
`PolicyLoss`, `ValueLoss` and `DPOLoss` are `nn.Module`s that receive log-probabilities and advantages and
return scalars; they do not read the reference model or the configuration. **The KL controller lives in
`trainer/ppo_utils/kl_controller.py`**, next to the experience maker, and holds a scalar `value` that the
experience maker and the actor both read. **`trainer/ray/` contains the actor classes**: `ppo_actor.py`
(policy), `ppo_critic.py` (value), `launcher.py` (placement groups and the reference and reward-model
workers), `vllm_engine.py` and `vllm_worker_wrap.py` (generation). A PPO job is a set of Ray actor handles
plus a driver loop.

The interactive diagram at [figures/openrlhf-ray.html](figures/openrlhf-ray.html) lets you click each Ray
actor to see what it owns, which tensors it sends, and which configuration flag changes its behaviour; it
also has a switch for the two KL placements described in §3.

---

## §2 `PolicyLoss`: the clipped objective and what it returns

**Definition.** `PolicyLoss` computes the PPO clipped surrogate over response tokens, optionally with
separate lower and upper clip bounds, an optional dual clip for negative-advantage tokens, and optional
importance-sampling weights that correct for the vLLM sampler's log-probabilities differing from the
trainer's.

**The problem it addresses.** A policy-gradient step computed from samples of an older policy is only
valid for small changes. Without a bound, one token with a large probability ratio and a large negative
advantage contributes a loss term that grows linearly in the ratio and can dominate the batch.

**Code** ([[openrlhf-ppo]] Technical Details, `openrlhf/models/loss.py` L136–148 and L175–182 at 64c1cc4,
verbatim):

```python
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
if self.dual_clip is None:
    loss = -torch.min(surr1, surr2)
else:
    clip1 = torch.min(surr1, surr2)
    clip2 = torch.max(clip1, self.dual_clip * advantages)
    loss = -torch.where(advantages < 0, clip2, clip1)
...
loss = (masked_mean(loss, action_mask, dim=None) if self.token_level_loss
        else masked_mean(loss, action_mask, dim=-1).mean())
clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
return loss, clip_ratio, ppo_kl, vllm_kl
```

Symbols: `ratio` is r_t = exp(log π_θ(y_t) − log π_old(y_t)) per response token for `policy_loss_type="ppo"`
(L122–124); for `"gspo"` it is the exponential of the masked mean log-ratio over the response, broadcast to
every token (L125–132). `advantages` is A_t. `clip_eps_low`, `clip_eps_high` are ε_low and ε_high, both
0.2 by default (`train_ppo_ray.py` L387–388). `dual_clip` is c, default `None`, asserted > 1.0 when set
(L106–107). `action_mask` marks response tokens.

**Worked example (bounds by advantage sign).** Take ε_low = ε_high = 0.2 and A = −1
([[openrlhf-ppo]] "Findings relevant to negative feedback"):

| r | surr1 = r·A | surr2 = clip(r)·A | loss = −min | gradient in r |
|---|---|---|---|---|
| 0.7 | −0.70 | −0.80 | 0.80 | zero (clipped branch) |
| 1.0 | −1.00 | −1.00 | 1.00 | non-zero |
| 1.5 | −1.50 | −1.20 | 1.50 | non-zero, grows linearly |
| 5.0, c = 3 | −5.00 | −1.20 | 3.00 | zero (dual-clip branch) |

The table states two separate bounds. For A < 0 the loss is constant once r falls below 1 − ε_low, and
it is unbounded above unless `dual_clip` is set. For A > 0 the roles swap: ε_high bounds the update. OpenRLHF leaves
`dual_clip` off by default; verl applies the same bound on every call with c = 3.0
([[verl-ppo-loss]] `core_algos.py` L1323–1334, `actor.yaml` L83). Neither repository reports an ablation
of c.

**What the loss returns.** Four scalars: the loss, `clip_ratio`, `ppo_kl`, and `vllm_kl`. `clip_ratio` is
the fraction of tokens with `surr2 < surr1`, counted across both advantage signs together, and it does not
count dual-clip activations (L180). This matters for diagnosis: a rising `clip_ratio` does not tell you
whether positive-advantage or negative-advantage tokens are being clipped. `ppo_kl` is the masked mean of
−(log π_θ − log π_old); note that when IS correction is enabled, L154 reassigns `log_ratio` inside the
correction branch, so `ppo_kl` then reports the same quantity as `vllm_kl` rather than the policy-vs-old
ratio ([[openrlhf-ppo]] "`ppo_kl` at this commit"). That is a property of this commit, not a general fact.

---

## §3 Where the KL to the reference is applied

**Definition.** The KL-to-reference term measures how far the policy has moved from a frozen reference
model π_ref (in RLHF, usually the SFT checkpoint) on the tokens the policy itself produced. OpenRLHF can
apply it in one of two places, controlled by one flag.

**Mechanism, step by step (reward placement, the default).**

1. The reference actor returns `base_action_log_probs` for the rollout.
2. `compute_approx_kl` turns log π_θ − log π_ref into a per-token KL estimate and clamps it to [−10, 10]
   (`models/utils.py` L48–79).
3. `compute_reward` builds the per-token reward: `−β·kl_t` on every response token, plus the scalar reward
   (clipped to `--reward.clip_range`, default (−10, 10)) written onto the last response token
   (`models/utils.py` L82–110).
4. Advantages are computed from that shaped reward by the configured estimator, then, for `gae`,
   `reinforce` and `reinforce_baseline`, mean-centred over the batch and divided by the batch standard
   deviation unless `--algo.advantage.no_std_norm` is set (`experience_maker.py` L315–328).
5. `PolicyLoss` sees advantages that already contain the KL cost.

```python
# openrlhf/models/utils.py @64c1cc4, compute_approx_kl (verbatim, L62-79)
log_ratio = log_probs.float() - log_probs_base.float()
if kl_estimator == "k1":
    pass  # log_ratio is already p - q
elif kl_estimator == "k2":
    log_ratio = log_ratio**2 / 2.0
elif kl_estimator == "k3":
    log_ratio = (-log_ratio).exp() - 1 + log_ratio
return log_ratio.clamp(min=-10, max=10)
```

`compute_reward` then writes `reward = last_reward + kl_reward`, where `kl_reward = -kl_coef * kl` covers
every response token and `last_reward` is the scalar reward scattered onto the final response token found
by `action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)`
(`models/utils.py` L92–108).

**Mechanism (loss placement).** With `--algo.kl.use_loss`, the experience maker skips the KL computation
entirely and zeroes `kl` (`experience_maker.py` L205–214), and the actor recomputes it during the update:

```python
# openrlhf/trainer/ray/ppo_actor.py @64c1cc4, L296-314 (excerpt)
if self.args.algo.kl.use_loss:
    if self.args.algo.kl.init_coef > 0:
        kl = compute_approx_kl(action_log_probs, base_action_log_probs,
                               kl_estimator=self.args.algo.kl.estimator)
    kl_loss = masked_mean(kl, experience.action_mask)
else:
    kl_loss = 0
loss = actor_loss + kl_loss * kl_ctl
```

**Worked numeric example.** Take one response of four tokens with per-token log-ratios
log π_θ − log π_ref = (0.5, −0.2, 0.3, 0.4), a terminal scalar reward of 1.0, and β = 0.01.

- k1 estimates: (0.5, −0.2, 0.3, 0.4); mean 0.250.
- k3 estimates: e^{−0.5}−1+0.5 = 0.1065; e^{0.2}−1−0.2 = 0.0214; e^{−0.3}−1+0.3 = 0.0408;
  e^{−0.4}−1+0.4 = 0.0703; mean 0.0598.
- Reward placement with k1: per-token rewards (−0.005, +0.002, −0.003, 1.0 − 0.004). With γ = λ = 1 and a
  zero value function, the return at the first token is 0.99, so every token's advantage carries the KL
  cost. The whole response is penalised by 0.01 in return units.
- Loss placement with k1: the policy loss gains 0.01 × 0.250 = 0.0025, and its gradient flows through
  log π_θ directly rather than through the advantage estimator.

The two estimators differ by a factor of about four on the same tokens (0.250 versus 0.0598), so β is not
transferable between them. The CLI encodes that: it prints a recommendation of k1 for reward placement and
k2 or k3 for loss placement (`train_ppo_ray.py` L675–680).

**Controllers.** `AdaptiveKLController` updates β multiplicatively toward a target KL, and
`FixedKLController.update` is a no-op (`kl_controller.py` L4–29):
`proportional_error = clip(KL_obs / KL_target − 1, −0.2, 0.2)`, then
`β ← β · (1 + proportional_error · n_steps / horizon)`. KL_obs is the measured mean KL for the batch,
KL_target is `--algo.kl.target`, n_steps is the rollout batch size times samples per prompt, and horizon
is `--algo.kl.horizon` (default 10000). Two facts about this controller are easy to state wrongly. First,
**the adaptive controller is not the default**: `--algo.kl.target` defaults to `None`, which selects
`FixedKLController(init_coef=0.01)` (`ppo_trainer.py` L171–176; `train_ppo_ray.py` L417–419), and the call
site carries the comment "TODO: KL controller must be FixedKLController; AdaptiveKLController is
incompatible here" (`ppo_trainer.py` L252–253). Second, **the adaptive rule is not InstructGPT's**: the
class docstring cites arXiv:1909.08593 (Ziegler et al. 2019), and InstructGPT used a fixed β = 0.02 for
all RL models ([[rlhf-instructgpt]] App. C.4).

**Conditions and limits.** OpenRLHF reports no experiment for any of these settings; the repository
contains no training results ([[openrlhf-ppo]] Summary). Every number above is a default or a code
behaviour, not a measured recommendation.

---

## §4 Forgetting controls that exist, and one that does not

**The measurable problem.** An RL run optimizes one reward on one prompt distribution. The quantity at
risk is performance on everything else: the benchmarks the model passed before RL started.

**Evidence that KL to the base policy predicts the loss.** [[rls-razor]] fine-tuned
Qwen 2.5 3B-Instruct on three tasks (math from Open-Reasoner-Zero, the Chemistry L-3 subset of
SciKnowEval, and ToolAlpaca tool use) with SFT and with GRPO using a binary success reward and no explicit
KL regularization, then measured HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande and HumanEval (§3.1).
On the Pareto frontier of new-task score against prior-task score, RL runs kept prior-task scores nearly
unchanged while SFT runs traded them away. The paper's forgetting law is that
E_{x∼τ}[KL(π₀ ‖ π)] — the KL between base and fine-tuned policy, evaluated on the *new task*
distribution τ — predicts forgetting regardless of which algorithm produced the model: a quadratic fit
gives R² = 0.96 in a controlled ParityMNIST setting and R² = 0.71 in the LLM experiments (§4, Fig. 3, Fig. 11).
**Result (single study).** Conditions: one model family at 3B for the LLM results, six benchmarks, GRPO
without a KL term. The paper does not test whether adding an explicit KL penalty to RL moves the same
curve, and the KL direction it measures (π₀ ‖ π) is the reverse of the direction OpenRLHF logs
(log π_θ − log π_ref, which estimates KL(π_θ ‖ π_ref)). Treat the OpenRLHF `kl` metric as a correlated
proxy, not the paper's quantity.

**The control OpenRLHF does not have.** InstructGPT's objective has a second term besides the KL penalty:

```
objective(φ) = E_{(x,y)∼D_{π_φ^RL}}[ r_θ(x,y) − β log( π_φ^RL(y|x) / π^SFT(y|x) ) ]
               + γ · E_{x∼D_pretrain}[ log π_φ^RL(x) ]
```
([[rlhf-instructgpt]] §3, Eq. 2; π_φ^RL is the RL policy, π^SFT the frozen SFT model, D_pretrain the
pre-training distribution, β the KL reward coefficient, γ the pre-training loss coefficient.)

The implementation drew 8 times more pre-training examples than RL episodes, computed PPO and
pre-training gradients in consecutive steps into the same buffer, and scaled the pre-training gradients by
γ = 27.8 (App. C.4). In the sweep, γ ≥ 20 recovered the 1.3B model's regressions on SQuADv2 and DROP, at
the cost of some validation reward, and γ = 27.8 worked from 1.3B to 175B (App. E.6). Raising β instead
did not work: at β = 2.0, 100× the default of 0.02, the regressions persisted and validation reward fell
sharply (App. E.6, Fig. 34). With PPO-ptx, the reported regressions were mitigated on all datasets tested
and HellaSwag exceeded GPT-3, while DROP, SQuADv2 and translation still lagged GPT-3 (§4.2, Fig. 29).
**Result (single study),** at 1.3B–175B with a 6B reward model.

A grep for `ptx` and for a pre-training dataset argument over `cli/train_ppo_ray.py`,
`trainer/ray/ppo_actor.py` and `trainer/ppo_trainer.py` at 64c1cc4 returns nothing: OpenRLHF's PPO path
has no pre-training or SFT loss-mixing term. The consequences for a run that must stay broad:

1. β and the reference model are the only in-loop anchors. Set `--algo.kl.init_coef` above zero and keep
   the reference actor resident.
2. Mixing must happen outside the RL loop — a separate SFT pass on general data, or checkpoint averaging
   with the pre-RL model, neither of which OpenRLHF automates.
3. In-loop measurement is the only remaining detector. §5 describes the part of the loop that can observe
   the loss while it is happening.

---

## §5 In-loop evaluation, checkpoint selection, and dynamic filtering

**Evaluation options** (`train_ppo_ray.py` L286, L535–540): `--eval.steps` (default −1, off),
`--eval.dataset`, `--eval.split` (default `train`), `--eval.temperature` (default 0.6), and
`--eval.n_samples_per_prompt` (default 4). `--eval.dataset` is accepted only together with
`--reward.remote_url` (L672–673), so in-loop evaluation requires a remote reward endpoint or an agent
function that can score the held-out prompts.

**What is computed.** `compute_eval_metrics` groups eval samples by their `datasource` field and, per
datasource, logs `eval_<ds>_pass1`, `eval_<ds>_pass<n>` (the max reward within the group of n samples),
`eval_<ds>_response_length_mean` and `eval_<ds>_truncated_rate`, plus overall length, truncation rate and
sample count (`ppo_trainer.py` L82–144). This is the mechanism that makes broad-domain monitoring possible:
put several held-out domains into one eval file with distinct `datasource` values and each gets its own
pass@1 and pass@n curve, on the same axis as reward.

**Checkpoint selection is a narrowing risk.** `_detect_eval_metric_key` takes an explicit
`--ckpt.best_metric_key`, and otherwise scans the metric names in sorted order and takes the first key
ending in `_pass1` (`ppo_trainer.py` L321–335). With several datasources, that is the alphabetically first
one. A run that evaluates `code`, `math` and `safety` will select checkpoints on `code` unless the key is
set explicitly.

**Dynamic filtering.** `--algo.dynamic_filtering_enable` with `--algo.dynamic_filtering_range`
(default `(0, 1)`) drops a prompt's entire group when the group's mean score falls outside the open
interval, and dispatches a replacement prompt from the dataloader
(`ppo_utils/samples_generator.py` L169–195). It requires `--reward.remote_url` or an agent function and
`--rollout.n_samples_per_prompt > 1` (L693–701). The fraction of prompts that survive is logged as
`dynamic_filtering_pass_rate` (`ppo_trainer.py` L542–543).

**Worked example.** With n = 8 samples per prompt and a {0, 1} verifier reward, the group mean is k/8 for
k correct samples. The default range `(0, 1)` is a strict inequality, so k = 0 and k = 8 are both dropped.
If, at some point in training, 30% of prompts are solved by all 8 samples and 20% by none, the pass rate
is 50% and the generator consumes about twice as many prompts per step as it uses. As training proceeds,
the k = 8 share grows, so the surviving distribution shifts toward whatever the policy currently finds
hard. That is the intended effect — every group has non-zero advantage variance — and it is also a
curriculum that removes mastered prompts from the training mixture. If retention on the mastered slice
matters, it has to be checked by evaluation, because those prompts no longer produce gradients.

---

## §6 DPO: the loss, the concatenated forward, and the reference model

**Loss.** With h = (log π_θ(y_c|x) − log π_θ(y_r|x)) − (log π_ref(y_c|x) − log π_ref(y_r|x)), OpenRLHF's
`DPOLoss` computes ℓ = −(1 − ε)·log σ(β·h) − ε·log σ(−β·h), or ℓ = (h − 1/(2β))² when `ipo` is set
([[openrlhf-dpo]], `loss.py` L264–281 at 64c1cc4):

```python
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

Symbols: y_c and y_r are the chosen and rejected responses; log π(y|x) is the **sum** of response-token
log-probabilities with prompt tokens masked (`dpo_trainer.py` L380–386); β is `--model.beta`, default 0.1;
ε is `--model.label_smoothing`, default 0.0; σ is the logistic function. `chosen_rewards` and
`rejected_rewards` are the implicit rewards, logged separately (L191–192).

**The concatenated forward.** `concatenated_forward` right-pads chosen and rejected to a common length
and concatenates them along the batch dimension, so one call covers both
(`dpo_trainer.py` L305–309, L350–360). The docstring gives the reason as avoiding two forward passes
"because it's faster for FSDP"; it makes no memory claim, and none is implied: concatenating produces a
2× batch, so the activations retained for backward are those of two forwards either way. The benefit is
one collective-heavy forward instead of two, not halved activation memory
([[openrlhf-dpo]] Verification).

**Total loss and the NLL anchor.** The trainer adds two optional terms
(`dpo_trainer.py` L157–184): `preference_loss + aux_loss * aux_loss_coef + nll_loss * nll_loss_coef`. The
NLL term is `-all_logps_mean[: chosen_ids.shape[0]].mean()` (L328) — the negative per-token mean
log-probability of each chosen response, averaged over the batch — not a sum, and not a term on the
rejected side. Its CLI help text cites the Llama 3.1 technical report (`train_dpo.py` L246). The
auxiliary term is the MoE router balancing loss, default coefficient 0 (`train_dpo.py` L244). Evaluation computes only
the preference loss and accuracy; neither optional term is included (L273–285).

**Reference model.** `train_dpo.py` uses the policy path for the reference when no reference path is given
(L342–343). The reference is set to `eval()` and its forward runs under `torch.no_grad()`
(L147, L160–163). CPU parameter offload for the reference is `--ref.offload`, **off by default**
(train_dpo.py L213; `deepspeed.py` L483–495), and it uses ZeRO stage 3 in the reference eval config when
training uses stage 3 and stage 0 otherwise.

**Evidence that these choices change general-benchmark results.** The Tülu 3 report ablated the
preference-optimization objective and β on an early SFT checkpoint with UltraFeedback, reporting an
average score over the evaluations used during development (§5.4.1, Table 18):

| Algorithm | LR | γ−β ratio | β | Epochs | Batch | Average score |
|---|---|---|---|---|---|---|
| SFT base (no preference training) | — | — | — | — | — | 55.7 |
| SimPO | 5e-7 | 0.5 | 2 | 1 | 128 | 51.8 |
| SimPO | 5e-7 | 0.3 | 10 | 1 | 128 | 52.9 |
| DPO | 5e-7 | — | 0.1 | 3 | 32 | 55.2 |
| PPO | 1e-6 | — | 0.0325 | 1 | 64 | 54.5 |
| PPO | 1e-6 | — | 0.05 | 1 | 64 | 55.5 |
| length-normalized DPO | 1e-7 | — | 5 | 3 | 32 | 56.1 |
| length-normalized DPO | 5e-7 | — | 10 | 3 | 32 | 55.2 |
| length-normalized DPO | 5e-7 | — | 15 | 3 | 32 | 55.7 |
| length-normalized DPO | 5e-7 | — | 2 | 3 | 32 | 46.8 |
| length-normalized DPO | 5e-7 | — | 5 | 3 | 32 | 53.4 |
| length-normalized DPO | 5e-7 | — | 5 | 1 | 32 | 57.3 |

**Result (single study).** Two readings matter here. First, nine of the eleven configurations score at or
below the SFT checkpoint they started from, and only two exceed it: preference training is not monotone in
broad-average score. That is the report's own conclusion — "only length-normalized DPO outperformed our
base checkpoint overall" (§5.4.1). Second, the spread within one objective (46.8 to 57.3 for
length-normalized DPO) is larger than the spread between objectives, and it is driven by β and epochs: at
the same learning rate and β = 5, three epochs give 53.4 and one epoch gives 57.3. β in the
length-normalized objective is not comparable to β = 0.1 in the plain objective, because the log-ratios
are divided by response length
(report Eq. 6). Conditions: one base checkpoint, one dataset, one seed per cell, and the report does not
name the exact evaluation subset in the table caption.

---

## §7 Long sequences: ring attention in OpenRLHF

**Definition.** Ring attention shards one sequence across N devices arranged in a ring; each device holds
a block of queries and rotates key-value blocks around the ring, overlapping the transfer of the next
block with the attention computation on the current one ([[ring-attention]] §3).

**The measurable problem it addresses.** Activation memory for attention grows with sequence length. The
paper's Table 1 gives maximum activation bytes per layer in bfloat16: 2bhs² for a vanilla transformer,
8bsh for memory-efficient attention, and **6bch for ring attention**, where b is batch size, h is hidden
dimension, s is sequence length, and c is block size. The ring-attention figure contains no s:
its activation memory scales with the per-device block size, not with the full sequence length.

**The constraint that decides whether it helps.** Blockwise attention on a block of size c costs 4dc²
FLOPs and requires transferring 4cd bytes of key and value blocks. Overlap requires 4dc²/F ≥ 4cd/B, that
is **c ≥ F/B**, where F is per-device FLOPS and B is unidirectional interconnect bandwidth (§3.2). The
minimum sequence length per device is s = 6c. The paper's Table 2:

| Host | FLOPS (TF) | Interconnect (GB/s) | Minimal block size | Minimal sequence length |
|---|---|---|---|---|
| A100 NVLink | 312 | 300 | 1.0K | 6.2K |
| A100 InfiniBand | 312 | 12.5 | 24.5K | 149.5K |
| TPU v4 | 275 | 268 | 1.0K | 6.2K |

**Worked example.** Suppose a packed training batch of 32,768 tokens and `--ds.ring_attn_size 8`, so each
rank holds 4,096 tokens. On NVLink-connected A100s the per-rank block exceeds the 1.0K minimum and the
key-value transfers hide behind compute. On A100s connected only by InfiniBand at 12.5 GB/s, the minimum
block is 24.5K tokens, so a 4,096-token slice leaves the ring waiting on communication for most of each
step. The same flag is a throughput gain on one cluster and a throughput loss on another.
[figures/ring-attention-sharding.html](figures/ring-attention-sharding.html) lets you set the interconnect,
ring size and packed length and see which side of the c ≥ F/B threshold the configuration falls on, with
the activation figure 6bch alongside.

**What OpenRLHF implements** (`openrlhf/models/ring_attn_utils.py` at 64c1cc4, and
`cli/train_ppo_ray.py` L506–513):

- `--ds.ring_attn_size` (default 1, meaning off) is the ring group size; `--ds.ring_attn_head_stride`
  (default 1) is "the number of heads to do ring attention each time. It should be a divisor of the number
  of heads. A larger value may results in faster training but will consume more memory."
- Ring attention requires packing. If `--ds.ring_attn_size > 1` and `--ds.packing_samples` is unset, the
  CLI prints "[Warning] --ring_attn_size > 1 requires --packing_samples." and turns packing on (L638–641).
- `unpad_and_slice_tensor` removes padding, concatenates the batch into one `(1, total_seqs)` sequence,
  pads it to a multiple of the ring size, and gives each rank its contiguous slice. The pad length is
  `ring_attn_pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size`. Position ids are
  rebuilt per packed sequence so each packed example restarts at 0.
- `gather_and_pad_tensor` all-gathers per-rank logits or log-probabilities, strips the padding, and
  restores the original `(batch, seqlen)` shape, so the loss modules never see the sharding.
- Ring size multiplies into the effective process count: the CLI asserts that
  `n_samples_per_prompt × rollout.batch_size ÷ rollout.micro_batch_size` is at least
  `actor.num_nodes × actor.num_gpus_per_node ÷ ring_attn_size ÷ tensor_parallel_size` (L703–709), and
  placement uses `duplicate_actors = ring_attn_size × tensor_parallel_size` (L90–139).

**Conditions and limits.** This is a training-side context-parallel option. It does not change the rollout
length limit, which is set by vLLM's generation arguments and `--data.max_len` (default 2048 tokens for
prompt plus response, `train_ppo_ray.py` L379). Long-context RL needs both: a rollout budget that admits
long responses and a training-side layout that can hold them.

---

## §8 Asynchronous rollout, partial rollout, and sampler drift

`--train.async_enable` swaps `PPOTrainer` for `PPOTrainerAsync`, which runs `GenerateSamplesActor` and
`TrainingActor` concurrently ([[async-rollout]], `ppo_trainer_async.py` L37–350). `rollout_queue` has size
`--train.async_queue_size` (default 1) and a companion `rollout_slots` queue, pre-filled with the same
number of tokens, acts as a counting semaphore the generator must take from before generating (L287–297).
`VLLMLock`, a Ray actor wrapping `asyncio.Lock`, serializes generation against weight broadcast so one
batch is generated with one weight set (L19–34). `--train.partial_rollout_enable` (defined at
`train_ppo_ray.py` L277–283 and asserted there to require async mode, L669–670) drops the lock and
instead calls vLLM `pause_generation(mode="keep")` before the broadcast
and `resume_generation` after (`vllm_engine.py` L132–136), so an in-flight sample may contain tokens from
two weight sets. verl aborts in-flight requests, waits up to 60 s and clears caches instead
([[verl-rollout]] `vllm_async_server.py` L932–1003); aborting and pausing are different trade-offs.

The importance-sampling correction (`--algo.advantage.is_correction_enable`, default off, band [0.5, 5.0],
type `tis`) exists because the trainer's recomputed `old_log_probs` are not the log-probabilities the vLLM
sampler used. `tis` clamps w = exp(old_log_probs − rollout_log_probs) into the band and multiplies the
loss; `icepop` zeroes tokens outside the band; `seq-mask-tis` drops sequences whose geometric-mean w is
outside it (`loss.py` L150–173). `vllm_kl` is the diagnostic ([[openrlhf-ppo]] Technical Details).

**Conditions and limits.** The repository reports no throughput or convergence measurement for async mode,
and the six arXiv versions of the OpenRLHF paper contain no async-versus-sync result
([[async-rollout]] Verification). The README's statements that synchronous mode has "better stability" and
partial rollout is "most aggressive off-policy" (L671–679) have no experiment behind them.

---

## Negative samples and negative feedback

Both objectives in this chapter use negatives **as gradient** (type 4 of the course standard): an explicit
decrease in the likelihood of a specific sample. Neither uses negatives as content or as conditioning.

**Where negatives come from.** In PPO and its critic-free variants, a token has A_t < 0 when its return
falls below the baseline: the critic's value for `gae`, the leave-one-out group mean for `rloo`, the group
mean for `reinforce_baseline` and `dr_grpo`, or the group mean divided by the group standard deviation for
`group_norm` (`experience_maker.py` L264–270). The KL reward −β·kl_t is itself negative on every token
where the k1 estimate is positive. Batch-level mean-centring (L315–328) creates negative advantages even
when every raw reward was positive, for `gae`, `reinforce` and `reinforce_baseline`. In DPO, the negative
is the rejected response, labelled by whatever produced the preference pair; the trainer has no false-
negative estimate and the repository reports none.

**Mechanism.** For a softmax over logits z with target token y,
∂ log p_y / ∂z_j = 1[j = y] − p_j. Lowering the likelihood of a sampled token removes mass from that token
and redistributes it in proportion to the current probabilities of the alternatives. When the pushed-down
token already had low probability, most of the freed mass goes to whichever alternative is already most
likely, which concentrates rather than diversifies the distribution. This is why the bound on the negative
branch matters: for A < 0, `PolicyLoss` has zero gradient below r = 1 − ε_low and unbounded loss above,
and only `dual_clip` caps it (§2).

For DPO, differentiating ℓ gives ∂ℓ/∂log π_θ(y_r|x) = β(1 − ε)σ(−βh) − βε σ(βh)
([[openrlhf-dpo]], derived from L272–275). With ε = 0 this is positive for all h, so every step lowers the
rejected log-probability; with ε = 0.1 and β = 0.1 the sign flips only once h exceeds 10·ln 9 ≈ 22.0. For
IPO, ∂ℓ/∂log π_θ(y_r|x) = 2(1/(2β) − h), so the push-down stops and reverses once h > 1/(2β) — 5.0 at
β = 0.1. Label smoothing and IPO are therefore bounds on the negative gradient, not only alternative
objectives.

**Controls available in this repository.**

| Control | Flag / code | Effect on the negative branch |
|---|---|---|
| Dual clip | `--actor.dual_clip` (default off) | Caps per-token loss at c·\|A\| for A < 0; zero gradient above r = c |
| Lower clip bound | `--actor.eps_clip_low_high` | ε_low decides where the A < 0 gradient turns off |
| IS band | `--algo.advantage.is_correction_*` | Downweights or drops tokens whose sampler mismatch is large |
| KL reward | `--algo.kl.init_coef` | Adds a per-token cost for moving away from π_ref in either direction |
| Positive anchor (DPO) | `--model.nll_loss_coef` | Adds cross-entropy pressure on the chosen response only |
| Label smoothing / IPO | `--model.label_smoothing`, `--model.ipo_enable` | Bound the rejected-side gradient as above |

**Diagnostics and their limits.** DPO logs `chosen_reward` and `reject_reward` separately, so a run where
both implicit rewards fall together — the margin improving while both log-probabilities decrease — is
visible; it does **not** log raw chosen and rejected log-probabilities ([[openrlhf-dpo]]). PPO logs
`ppo_clip_ratio`, `ppo_kl`, `kl`, `logprobs_diff` and, when IS correction is on, `vllm_kl`
(`ppo_actor.py` L291–310). `clip_ratio` does not split by advantage sign and omits dual-clip activations,
so a sign-split clip fraction has to be added by hand if the question is which branch is saturating.

**Honesty about effect size.** OpenRLHF's repository contains no experiments, so it provides no evidence
about how much of any gain comes from negative-advantage tokens or from the rejected term. The claim that
negatives drive most of an improvement is not supported by anything in this chapter's sources.

---

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenRLHF default (no model) | — | RL | KL coefficient β (`--algo.kl.init_coef`) | 0.01 | `cli/train_ppo_ray.py` L419 @64c1cc4 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | KL placement (`--algo.kl.use_loss`) | `False` → KL added to per-token reward | `train_ppo_ray.py` L484–486; `experience_maker.py` L205–214 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | KL estimator (`--algo.kl.estimator`); clamp | `k1`; [−10, 10] | `train_ppo_ray.py` L421–429; `models/utils.py` L79 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | KL controller (`--algo.kl.target`; horizon) | `None` → `FixedKLController`; 10000 | `train_ppo_ray.py` L417–418; `ppo_trainer.py` L171–176 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | clip ε; dual clip | 0.2; `None` (off, must be > 1.0) | `train_ppo_ray.py` L387–389; `loss.py` L106–107 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | entropy coefficient | `None` (no entropy term) | `train_ppo_ray.py` L431–436 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | advantage estimator; batch normalization | `gae`; mean-centre and divide by std unless `--algo.advantage.no_std_norm` (gae, reinforce, reinforce_baseline) | `train_ppo_ray.py` L477–483; `experience_maker.py` L315–328 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | eval steps; eval temperature; eval samples per prompt | −1 (off); 0.6; 4 | `train_ppo_ray.py` L286, L535–540 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | dynamic filtering; range | off; (0, 1) | `train_ppo_ray.py` L562–567 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | ring attention size; head stride | 1 (off); 1 | `train_ppo_ray.py` L506–513 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | max total sequence length (`--data.max_len`) | 2048 tokens (prompt + response) | `train_ppo_ray.py` L379 | verified 2026-09-14 | no ablation reported |
| OpenRLHF default (no model) | — | RL | async enable; queue size; partial rollout | `False`; 1; `False` | `train_ppo_ray.py` L274–283 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | preference loss; β; label smoothing; NLL coefficient | sigmoid DPO; 0.1; 0.0; 0 | `train_dpo.py` L237–247 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | reference path; reference offload | policy path when unset; off | `train_dpo.py` L342–343, L213 | verified 2026-09-14 | no ablation reported |
| InstructGPT (PPO-ptx, all sizes) | 1.3B–175B | RL | KL reward coefficient β; placement | 0.02; per-token in reward | arXiv:2203.02155 App. C.4 | verified 2026-09-17 | App. E.7 Fig. 36: human Likert optimum around 0.01–0.02 |
| InstructGPT (PPO-ptx, all sizes) | 1.3B–175B | RL | pre-training loss coefficient γ; pre-training examples | 27.8; 8× the number of RL episodes | arXiv:2203.02155 App. C.4 | verified 2026-09-17 | App. E.6 Fig. 33: γ ≥ 20 recovers SQuADv2 and DROP at 1.3B; Fig. 34: raising β to 2.0 does not |
| InstructGPT (PPO-ptx, all sizes) | 1.3B–175B | RL | clip ε; rollout temperature; episodes; batch / minibatch; inner epochs | 0.2; 1.0; 256k episodes over ~31k unique prompts; 512 / 64; 1 | arXiv:2203.02155 App. C.4 | verified 2026-09-17 | no ablation reported |
| Tülu 3 ablation checkpoint (early Tülu 3 SFT model, UltraFeedback) | not stated at the locus | preference | length-normalized DPO: β; epochs; LR; batch | 5; 1; 5e-7; 32 | Tülu 3 report §5.4.1 Table 18 | verified 2026-09-17 | Table 18: 57.3 average vs 55.7 for the SFT base; β = 2 gives 46.8 |

**Starting point for a small general-purpose RL run.** Every number here comes from a `verified` row
above. Keep the OpenRLHF defaults for clip ε (0.2) and the KL estimator (k1 with reward placement), set
`--algo.kl.init_coef` at the default 0.01 rather than 0, set `--actor.dual_clip 3.0` to bound the
negative-advantage branch as verl does by default, and set `--eval.dataset` with several `datasource`
values plus an explicit `--ckpt.best_metric_key` so checkpoint selection is not decided alphabetically.
These are framework defaults from a repository that reports no experiments; the conditions under which
they were chosen are not documented, so they are a starting configuration to measure against, not a
recommendation supported by results. The InstructGPT β = 0.02 and γ = 27.8 values come from
GPT-3-family models of 1.3B–175B with a 6B reward model on a 31k-prompt instruction distribution, and γ
has no OpenRLHF equivalent.

---

## Generalization lens

**(a) What increases breadth.** Keeping the KL term active gives the loop an anchor to the pre-RL policy,
and the quantity it controls is the one that predicts forgetting in [[rls-razor]] (§4, R² = 0.71 on the LLM
experiments, R² = 0.96 in the controlled setting). Mixing pre-training or SFT gradients into the RL update
works better than a larger β where it has been measured: γ ≥ 20 recovered SQuADv2 and DROP at 1.3B while
β = 2.0 did not ([[rlhf-instructgpt]] App. E.6). OpenRLHF does not implement that mixing, so this control
must be applied outside the framework. In-loop evaluation with several `datasource` values makes breadth a
monitored quantity rather than an end-of-run discovery (`ppo_trainer.py` L82–144).

**(b) What causes narrowing.** Setting `--algo.kl.init_coef 0` removes the only in-loop anchor.
Dynamic filtering removes mastered and unsolved prompts from the training mixture, so the effective
distribution narrows toward the current frontier of the policy. Best-checkpoint auto-detection selects on
a single datasource's pass@1 unless overridden. On the DPO side, β and the objective variant move the
broad average by more than 10 points in the Tülu 3 ablation, and nine of eleven configurations landed at
or below the SFT checkpoint they started from (Table 18). An unanchored negative gradient concentrates
probability mass on the already-most-likely alternative (see the negatives section), which reduces output
diversity even when pass@1 rises.

**(c) How to measure it at this stage.** Log `eval_<ds>_pass1` and `eval_<ds>_pass<n>` for at least one
domain outside the reward's domain, with `--eval.n_samples_per_prompt` greater than 1 so pass@n is
meaningful. Log `kl` and `logprobs_diff` every step and plot held-out scores against KL rather than
against step count, because KL is the predictor with published support. For DPO, log `chosen_reward` and
`reject_reward` separately and watch for both falling together. Add `dynamic_filtering_pass_rate` if
filtering is on, since a falling pass rate means the training distribution is shifting.

**Known measurement errors.** `ppo_kl` reports the sampler-drift quantity rather than the policy-vs-old
ratio when IS correction is enabled at this commit. `clip_ratio` mixes both advantage signs. The
`eval.split` default is `train`, so an eval file that also contains a train split will be evaluated on the
wrong rows unless the flag is set. In-loop pass@1 at temperature 0.6 is not comparable to a benchmark
harness's greedy number.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Assuming the adaptive KL controller is active | β never moves although KL drifts far from target | `--algo.kl.target` is `None` by default, which selects `FixedKLController`; print `kl_ctl.value` per step |
| Carrying a β tuned for k1 over to `--algo.kl.use_loss` with k3 | KL penalty becomes roughly 4× weaker on the same rollouts (worked example, §3) | Log both estimators on one batch before switching; heed the CLI's k1-vs-k2/k3 recommendation |
| Expecting `concatenated_forward` to halve DPO activation memory | Out-of-memory at the batch size the halving would have allowed | The concatenation makes a 2× batch; reduce `micro_batch_size` or enable `--ref.offload` |
| Leaving `--ref.offload` off on a large DPO run | Reference model out-of-memory during the no-grad forward | `--ref.offload` is off by default (`train_dpo.py` L213) |
| Relying on `clip_ratio` to see which advantage sign is clipping | `clip_ratio` rises with no indication of cause; dual-clip activations invisible | Compute the clip fraction separately for A ≥ 0 and A < 0 |
| Enabling ring attention on a low-bandwidth interconnect | Step time rises with `ring_attn_size` instead of falling | Compare the per-rank slice length to c = F/B from [[ring-attention]] Table 2 |
| Enabling ring attention without packing | Silent configuration change | The CLI prints "[Warning] --ring_attn_size > 1 requires --packing_samples." and sets packing on |
| Letting best-checkpoint selection auto-detect with several eval datasources | The saved "best" checkpoint tracks one domain | Set `--ckpt.best_metric_key` explicitly |
| Reading dynamic filtering as a pure efficiency feature | `dynamic_filtering_pass_rate` falls over training while held-out breadth degrades | Evaluate on prompts the filter removes, not only on the surviving mixture |
| Treating README statements about async stability as measured | Decisions made on unmeasured claims | [[async-rollout]] Verification: no async-versus-sync result exists in the repository or in any of the paper's six versions |

---

## Check your understanding

1. The default configuration adds −β·kl_t to every response token's reward rather than adding a KL term to
   the loss. Explain what that changes about the gradient that reaches log π_θ, and why the CLI recommends
   a different KL estimator for each placement.
2. For A < 0, the PPO loss is constant below r = 1 − ε_low and grows linearly above r = 1. Derive both
   statements from the `surr1`/`surr2` code, and explain why `dual_clip` is the only bound on the upper
   side.
3. [[rls-razor]] measures KL(π₀ ‖ π) on the new-task distribution, while OpenRLHF logs an estimate of
   KL(π_θ ‖ π_ref) on rollouts. Explain why the logged metric is a proxy rather than the paper's quantity,
   and what would have to be computed to log the paper's quantity.
4. InstructGPT's sweep found that γ ≥ 20 recovered the public-NLP regressions but β = 2.0 did not. Explain
   the mechanistic difference between the two terms that accounts for this, in terms of which distribution
   each one constrains.
5. Dynamic filtering with the default range and a binary reward drops groups where all samples are correct
   and groups where none are. Explain how that changes the training distribution over the course of a run,
   and design a measurement that would reveal whether the dropped prompts are being forgotten.
6. In the Tülu 3 ablation, length-normalized DPO at β = 2 scored 46.8 while the same objective at β = 5
   with one epoch scored 57.3 and the SFT base scored 55.7. Explain why a lower β can produce a *worse*
   broad average, and what you would log during such a run to see it happening before the final evaluation.
7. A run on InfiniBand-connected A100s sets `--ds.ring_attn_size 8` with 32,768 packed tokens and sees
   step time increase. Explain the cause using c ≥ F/B, and state two configuration changes that would
   each resolve it.
8. `concatenated_forward` gives one forward pass over a 2× batch. Explain why this is a speed argument and
   not a memory argument, and name the distributed-training mechanism the docstring credits.

---

## Connections

- **Previous — ch-55: verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation.** verl's `use_kl_loss` versus `use_kl_in_reward` is the same switch as `--algo.kl.use_loss`; its dual-clip default of 3.0 contrasts with OpenRLHF's `None`.
- **Next — ch-57: TRL Internals: SFT, DPO, GRPO, and Distillation Trainers.** A third implementation of the same objectives, read against ch-55 and this chapter.
- **ch-58: Choosing and Instrumenting a Post-Training Stack to Measure and Protect General Capability.** §4 and §5 supply two rows of that chapter's feature matrix: loss mixing (absent here) and in-loop broad evaluation (present, with the checkpoint-selection caveat).
- **ch-38: KL-Controlled RLHF: PPO, InstructGPT, and the Alignment Tax.** The objective whose two terms §3 and §4 trace into code.
- **ch-38a: SFT versus RL Generalization: On-Policy Data, KL to the Base Model, and Output Diversity.** The retention evidence, including [[rls-razor]], treated in its own right.
- **ch-39: Offline Preference Optimization: DPO and Its Variants.** The derivations behind §6's loss, including likelihood displacement and the anchors that bound the rejected-side gradient.
- **ch-43: Entropy, Output Diversity, and KL Control in RL.** k1/k2/k3 bias and variance, and KL placement, as an algorithmic question rather than a flag.
- **ch-50: Slice Analysis, Forgetting Slices, and Failure Bucketing.** What to do with the per-datasource metrics §5 makes available.
- **ch-05: Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen.** Ring attention among the other context-parallel options, and the world-size arithmetic `ring_attn_size` enters.

---

## Sources

- [[openrlhf-ppo]] — `PolicyLoss` code, KL estimators and reward shaping, controller defaults, advantage estimators and batch normalization, IS-correction variants, at commit 64c1cc4.
- [[openrlhf-ppo-recipe]] — the framework-default rows quoted in the Recipe table.
- [[openrlhf-dpo]] — `DPOLoss`, the train step, `concatenated_forward`, the NLL and MoE auxiliary terms, reference-model handling, and the derived rejected-side gradients.
- [[openrlhf-dpo-recipe]] — DPO CLI defaults, `DPOTrainer` constructor defaults, and the example Llama-3-8B script values.
- [[async-rollout]] — the async trainer's two actors, the queue and slot pair, `VLLMLock`, partial rollout through vLLM pause/resume, and the absence of any throughput measurement.
- [[verl-ppo-loss]] — verl's dual-clip default of 3.0, the contrast in §2.
- [[verl-rollout]] — verl's abort-and-clear weight-sync path, the contrast in §8.
- [[rlhf-instructgpt]] — Equation 2, the per-token KL penalty at β = 0.02, PPO-ptx at γ = 27.8, and the γ and β sweeps in App. E.6–E.7.
- [[rls-razor]] — the empirical forgetting law and the SFT-versus-RL retention comparison used in §4 and in the Generalization lens.
- [[ring-attention]] — Table 1 activation figures, the c ≥ F/B overlap condition, and Table 2's minimal block and sequence lengths per host.
- [[dpo]] — the DPO objective, implicit reward and gradient that §6's code implements.
- [[tulu-3]] — the preference-tuning ablation (§5.4.1, Table 18), the evidence that β and the objective variant move a broad average.
- [[openrlhf-entropy-debugging]] — practitioner triage notes for entropy and KL problems across OpenRLHF, verl and TRL. Source reliability: anecdotal (framework READMEs, issue trackers, community digests); its quantitative claims are not used as evidence in this chapter.
