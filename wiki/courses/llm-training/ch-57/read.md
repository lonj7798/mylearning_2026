<!-- chapter: ch-57
     track: infra
     kind: content
     title: TRL Internals: SFT, DPO, GRPO, and Distillation Trainers
     deps: [ch-56]
     sources: [[trl-grpo]], [[trl-grpo-recipe]], [[trl-online-dpo]], [[trl-online-dpo-recipe]], [[trl-ppo]], [[trl-ppo-recipe]], [[gkd-on-policy-distillation]], [[lora-learns-less-forgets-less]], [[hf-alignment-handbook]], [[hf-dpo-zoo]], [[trl-repo-a04ffd3]], [[grpo]], [[dpo]], [[ipo]], [[ppo]], [[dr-grpo]], [[john-schulman-kl-tricks]], [[verl-grpo]], [[openrlhf-ppo]], [[openrlhf-dpo]], [[entropy-logging-patterns]], [[rlhf-instructgpt]]
     figures: figures/trl-stack.html, figures/multi-reward-aggregation.html
     excerpts: excerpts/trl-repo-a04ffd3.md, excerpts/lora-learns-less-forgets-less.md, excerpts/gkd-on-policy-distillation.md
     revised: 2026-09 (generality revision)
-->

# Chapter 57 — TRL Internals: SFT, DPO, GRPO, and Distillation Trainers

> **Core insight.** TRL's post-training behaviour is decided by configuration fields whose defaults are neither the published paper settings nor stable across releases. At commit `a04ffd3` (2026-09-14) the GRPO default is `loss_type="dapo"` with `beta=0.0`, not the DeepSeekMath objective ([[trl-repo-a04ffd3]], `grpo_config.py` L796, L676; loss code in [[trl-grpo]]); `SFTConfig` trains on the whole rendered conversation unless `assistant_only_loss=True` ([[trl-repo-a04ffd3]], `sft_config.py` L271); `DPOConfig.loss_type` is a **list** of 15 named losses that are combined with `loss_weights` ([[trl-repo-a04ffd3]], `dpo_config.py` L81-L88); and `PPOTrainer` no longer exists — PR #7020 deleted it on 2026-09-04, released in v1.13.0 ([[trl-ppo]] Verification). Each of these fields changes which tokens receive gradient and how much, so each one changes how much general capability a post-training run keeps.
>
> **Guideline.** When reproducing a published result in TRL, pin the commit and set `loss_type`, `beta`, and `scale_rewards` explicitly, because the defaults differ from the objectives in [[grpo]] and [[dpo]]. When fine-tuning a conversational dataset, set `assistant_only_loss=True` and verify that the template's end-of-turn token is inside the mask, because TRL logs a warning rather than raising when it is not ([[trl-repo-a04ffd3]], `sft_trainer.py` L1271-L1277). When several reward functions have different numeric scales, use `multi_objective_aggregation="normalize_then_sum"`, because `"sum_then_normalize"` (the default) lets the reward with the larger raw spread set the sign of the advantage (§4.3 worked example). When the priority is keeping capability outside the fine-tuning domain, use LoRA with rank 256 and α = 2r rather than full fine-tuning, because on code instruction fine-tuning of Llama-2-7B, full fine-tuning scored 0.414 on the forgetting metric against 0.509 for LoRA r = 64 ([[lora-learns-less-forgets-less]] §4.2, Table S6); when the priority is target-domain accuracy on code continued pretraining, use full fine-tuning, because LoRA underperformed it at every rank tested in that regime (§4.1).

---

## Why this chapter matters for a general-purpose model

Pipeline position: pre-training → mid-training → **SFT → preference optimization → RL** → evaluation. [[ch-55]] covered verl and [[ch-56]] covered OpenRLHF. TRL is the third framework in this group and the one whose trainers are subclasses of `transformers.Trainer`, so its distributed layer is `accelerate` rather than Ray.

The generality question for a framework chapter is not throughput. It is **which configuration fields decide, without raising an error, how much of the base model's breadth survives the run**. Three measurable problems follow.

1. **Loss-mask and template overfitting.** If the loss covers the user and system tokens of a rendered chat, the model is trained to produce turn headers it will never need to produce at inference. If the assistant turn's end-of-turn token is outside the mask, the model is never trained to stop. Both are silent: the training loss curve looks normal.
2. **Objective drift between paper and framework.** A run configured with TRL defaults optimizes a different objective from the paper whose name is on the trainer. `beta=0.0` means no reference model at all, so the KL anchor to the pre-RL policy that [[rlhf-instructgpt]] relies on is absent unless it is turned on.
3. **Narrowing under a single reward.** One verifiable reward on one domain narrows the policy. TRL's multi-reward and multi-domain surface (`reward_funcs`, `reward_weights`, `multi_objective_aggregation`) and its evaluation hooks (`eval_dataset`, `num_generations_eval`, `LogCompletionsCallback`) are the parts of the API that let breadth be measured during training instead of after it.

Sections: the repository at a pinned commit (§1); SFT masking, packing, and templates (§2); the DPO family and the PEFT reference trick (§3); GRPO loss aggregation, KL estimators, and multi-reward training (§4); online and asynchronous trainers (§5); distillation trainers (§6); LoRA versus full fine-tuning, measured (§7); in-loop evaluation and merging (§8).

**Everything in this chapter is read at one commit: `a04ffd337108285002df29d9ae73ef7e855e03f1`, `main` on 2026-09-14.** File and line references are to that tree. TRL's API has changed across every recent minor release, so a line reference without a commit is not checkable.

---

## §1 The repository at a pinned commit

### §1.1 What the tree contains

At `a04ffd3` the `trl/` package splits into a supported surface and an experimental one. The full listing is in [[trl-repo-a04ffd3]]; the parts that matter here are:

```
trl/trainer/          sft_trainer.py  dpo_trainer.py  grpo_trainer.py
                      rloo_trainer.py  kto_trainer.py  reward_trainer.py
                      distillation_trainer.py  callbacks.py  utils.py
trl/experimental/     async_grpo/  async_distillation/  server_distillation/
                      online_dpo/  nash_md/  xpo/  gkd/  minillm/  gold/
                      sdft/  sdpo/  ssd/  iw_opd/  bco/  cpo/  orpo/  kto/
                      prm/  a2po/  gmpo/  gspo_token/  tpo/  harbor/
                      merge_model_callback.py
trl/losses/           dpo_loss.py  grpo_loss.py  jsd_loss.py
                      fused_linear_distillation.py  fused_linear_ppo.py
trl/rewards/          accuracy_rewards.py  format_rewards.py  other_rewards.py
trl/generation/       vllm_client.py  vllm_generation.py
```

Three facts follow directly from the listing and are worth stating because older descriptions of TRL contradict them.

1. **There is no `ppo_trainer.py` and no `modeling_value_head.py`.** `trl/models/` contains only `activation_offloading.py` and `utils.py`. PR #7020 (commit `700b845`, 2026-09-04) deleted `PPOTrainer`, `PPOConfig`, and the value-head model; the deletion shipped in v1.13.0 ([[trl-ppo]] Verification). Code that needs the actor-critic trainer must pin a TRL release earlier than v1.13.0 and import from `trl.experimental.ppo`.
2. **A separate value model, not a shared value head.** While `PPOTrainer` existed, the value function was a separate `nn.Module` with a `.score` head, wrapped with the policy in `PolicyAndValueWrapper` so that one `accelerator.prepare` call covered both (`ppo_trainer.py` L283-L302 at `a08e713`, [[trl-ppo]]). It did not import `AutoModelForCausalLMWithValueHead`.
3. **`ORPO`, `CPO`, `BCO`, `KTO` (experimental copy), and `GKD` are under `trl/experimental/`.** They are separate trainers, not `loss_type` values of `DPOTrainer`. [[hf-dpo-zoo]] states that "all variants are exposed through `trl.DPOTrainer` with a `loss_type` parameter"; that is not true at this commit and is corrected in §3.1.

### §1.2 Stable versus experimental

The division is by import path, not by quality. `trl/experimental/` holds both new research trainers (`a2po`, `gmpo`, `tpo`, `iw_opd`) and older algorithms that were moved out of the supported surface (`online_dpo` in v0.26.0, PR #4473; `ppo` in v0.26.0, PR #4482, then deleted). Two practical consequences:

- An import path is part of a reproduction recipe. `from trl import OnlineDPOTrainer` fails at this commit; `from trl.experimental.online_dpo import OnlineDPOTrainer` is the path ([[trl-online-dpo]]).
- The repository publishes no compatibility guarantee text that the audits could locate for either surface, so "stable" here means "in `trl/trainer/`", not a versioning promise.

See **[figures/trl-stack.html](figures/trl-stack.html)** for a map of the trainers at this commit, with the removed and relocated ones marked; it lets you check which import path a given algorithm has today.

---

## §2 SFT: what receives gradient

### §2.1 Definition and the measurable problem

Supervised fine-tuning minimizes cross-entropy on target tokens. The design decision is **which** tokens are targets. The measurable problem: a model trained on the full rendered conversation assigns gradient to the user-turn header and body, so at inference it can continue past its own answer and generate a fabricated user turn. A model whose end-of-turn token is outside the mask never learns to emit the stop token that the serving configuration expects.

### §2.2 Mechanism: the three masking modes

`SFTConfig` at `a04ffd3` exposes two masking fields, and their behaviour depends on the dataset shape ([[trl-repo-a04ffd3]], `sft_config.py` L96-L105):

- `completion_only_loss` (default `None`): "If `None` (default), the behavior depends on the dataset: loss is computed on the completion for prompt-completion datasets, and on the full sequence for language modeling datasets."
- `assistant_only_loss` (default `False`, L271): "loss is computed only on the assistant responses, which is supported only for conversational datasets."

So for a `messages`-column conversational dataset with both fields left at their defaults, **every token of the rendered conversation is a target**. There is no field named `train_on_response_only` in TRL; that name belongs to an Unsloth helper. The field named `max_seq_length` was replaced by `max_length` (default 1024, L203).

`assistant_only_loss` is implemented through the chat template's `{% generation %}` markers, which `apply_chat_template(..., return_assistant_tokens_mask=True)` turns into an `assistant_masks` column (`sft_trainer.py` L1551). TRL swaps in a training template when the tokenizer's template has no markers:

```python
# trl/trainer/sft_trainer.py @a04ffd3 L1262-L1277
# When assistant_only_loss is enabled, swap in a training chat template with {% generation %} markers
# if the current template doesn't already have them.
if args.assistant_only_loss and not has_generation_markers(processing_class.chat_template):
    self.chat_template = get_training_chat_template(processing_class)
else:
    self.chat_template = None

# A template can define generation markers and still attribute the assistant's end-of-turn token to the next
# message, leaving it out of the assistant mask so the model is never trained to stop.
if args.assistant_only_loss and not is_chat_template_stop_token_trained(
    processing_class, chat_template=self.chat_template
):
    logger.warning(
        "The chat template does not include the assistant turn's end-of-turn token in the loss mask; "
        "the model may not learn to stop."
    )
```

This is a **warning**, not an error. A run with a mis-marked template completes normally and produces a model that does not stop.

### §2.3 Worked example: how much gradient goes to non-assistant tokens

Take one rendered two-round conversation with these token counts: system 20, user₁ 30, assistant₁ 60, user₂ 25, assistant₂ 80. Total 215 tokens.

| Setting | Target tokens | Share of the gradient on tokens the model never generates at inference |
|---|---|---|
| `assistant_only_loss=False` (default, conversational data) | 215 | 75 / 215 = 34.9% |
| `assistant_only_loss=True` | 140 | 0% |

Under the default, roughly a third of the optimizer's work on this example is spent raising the probability of turn headers and user text. The loss value is lower in the default case (template tokens are highly predictable), which is why the loss curve does not reveal the problem.

### §2.4 Packing

`packing` (default `False`, L223) groups sequences into fixed-length blocks of `max_length`. `packing_strategy` (default `"bfd"`, L230) offers three options, quoted from the docstring (L80-L82): `"bfd"` (best-fit decreasing, truncates overflow), `"bfd_split"` (best-fit decreasing, splits overflow sequences), `"wrapped"` (aggressive, cuts mid-sequence). There is no `ConstantLengthDataset` in the current path. When `packing` is on with strategy `"bfd"`, `padding_free` is enabled regardless of its own value (L83-L88), which means the batch is flattened into one continuous sequence and requires FlashAttention 2 or 3 to keep per-example attention boundaries.

The relevance to breadth: `"wrapped"` cuts examples mid-sequence, so an example's answer can begin in one training block and end in another with no prompt. For instruction data this produces targets with no instruction. `"bfd"` truncates instead, which loses the tail of long examples — the failure that matters for long-context SFT ([[ch-32c]]).

### §2.5 Conditions and limits

No source in this library reports an ablation of TRL's SFT defaults. The published Zephyr-7B SFT config in the Alignment Handbook sets `max_seq_length: 2048`, `learning_rate: 2.0e-05`, `lr_scheduler_type: cosine`, `warmup_ratio: 0.1`, `num_train_epochs: 1`, `per_device_train_batch_size: 16`, `gradient_accumulation_steps: 1`, `bf16: true` — and sets **neither** `packing` nor any response-only masking field ([[hf-alignment-handbook]]; config read at `recipes/zephyr-7b-beta/sft/config_full.yaml`, commit `1de1fc9`). The [[hf-alignment-handbook]] card's instruction to "keep `packing: true` and `train_on_response_only: true`" is not supported by that config and is corrected here.

---

## §3 The DPO family

### §3.1 `loss_type` is a list, and the list is not what older summaries say

```python
# trl/trainer/dpo_config.py @a04ffd3 L81-L88 (docstring)
loss_type (`list[str]`, *optional*, defaults to `["sigmoid"]`):
    Type of loss to use. Possible values are: `'sigmoid'`, `'hinge'`, `'ipo'`, `'exo_pair'`,
    `'nca_pair'`, `'robust'`, `'bco_pair'`, `'sppo_hard'`, `'aot'`, `'aot_unpaired'`, `'apo_zero'`,
    `'apo_down'`, `'discopop'`, `'sft'`, `'sigmoid_norm'`. If multiple loss types are provided, they
    will be combined using the weights specified in `loss_weights`.
loss_weights (`list[float]`, *optional*):
    List of loss weights for multi-loss combinations. ... Example: `[0.8, 0.2, 1.0]` for MPO.
```

`'kto'`, `'simpo'`, `'orpo'` and `'cpo'` are **not** in this list. SimPO is `CPOTrainer(loss_type="simpo")`; KTO, ORPO, CPO and BCO are separate trainers, and at this commit all of them except `kto_trainer.py` live under `trl/experimental/`. This corrects [[hf-dpo-zoo]], whose "TRL implementation consistency" section claims a single `DPOTrainer` exposes every variant.

The combination loop is a weighted sum over the named losses (`dpo_trainer.py` L1470-L1472):

```python
loss = 0.0
for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
    if loss_type == "sigmoid":
        per_sequence_loss = -F.logsigmoid(self.beta * delta_score)
```

with `delta_score = chosen_scores - rejected_scores` (L1468), the log-ratio difference of [[dpo]]. The `'sft'` member is an ordinary cross-entropy on the chosen completion only (L1581-L1587), so `loss_type=["sigmoid", "sft"]` with `loss_weights=[1.0, 0.2]` is a preference loss anchored by a positive NLL term. This is the mechanism §Negatives calls "anchor with a positive NLL term".

### §3.2 The PEFT reference-model trick, stated precisely

DPO needs log-probabilities under a frozen reference π_ref. With a PEFT adapter, TRL does not load a second model:

```python
# trl/trainer/dpo_trainer.py @a04ffd3 L919-L924
if ref_model is None:
    if is_peft_model(self.model) or args.precompute_ref_log_probs:
        # If PEFT is used, the reference model is not needed since the adapter can be disabled to revert to the
        # initial model. If precompute_ref_log_probs is True, the reference model does not need to be kept in
        # memory during training.
        self.ref_model = None
```

and at loss time (L1412-L1418):

```python
if is_peft_model(model) and self.ref_model is None:
    # When training a PEFT adapter, how we obtain the reference depends on the setup:
    # - New adapter: disabling adapters yields the base model.
    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
    model = self.accelerator.unwrap_model(model)
    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
        ref_outputs = self.model(**ref_model_kwargs)
```

The saving is one frozen copy of the base weights, not one forward pass: the reference forward still runs, under `torch.no_grad()`. The second branch, `precompute_ref_log_probs`, removes the reference forward from the training loop entirely by computing π_ref log-probabilities once in a preprocessing pass; it is incompatible with `sync_ref_model`, which the docstring also marks as not yet compatible with PEFT (`dpo_config.py` L119-L124).

Note what `sync_ref_model` does and does not do. It periodically mixes the current policy into the reference (TR-DPO, `ref_model_mixup_alpha` default 0.6, `ref_model_sync_steps`). Without it the reference is frozen for the whole run. In **online DPO** the reference is also frozen — what becomes on-policy is the sampled pair, not the reference ([[trl-online-dpo]] Corrections).

---

## §4 GRPO: aggregation, KL, and multiple rewards

### §4.1 The eight loss types differ in the normalizer

`GRPOTrainer._compute_loss` implements eight `loss_type` values that share a per-token surrogate and differ in how the masked per-token loss is reduced to a scalar ([[trl-grpo]], `grpo_trainer.py` L2548-L2562 at `a08e713`):

```python
if self.loss_type in ["grpo", "sapo"]:
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
elif self.loss_type == "bnpo":
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
elif self.loss_type == "dr_grpo":
    loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
elif self.loss_type in ["cispo", "dapo", "vespo"]:
    normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
    loss = (per_token_loss * mask).sum() / normalizer
```

**Worked example.** One group with G = 2 completions. Completion A has 4 completion tokens, each with per-token loss 1.0. Completion B has 8 completion tokens, each with per-token loss 0.0. `max_completion_length = 16`.

| `loss_type` | Arithmetic | Scalar loss |
|---|---|---|
| `"grpo"` | ((4·1.0)/4 + (8·0.0)/8) / 2 | 0.500 |
| `"bnpo"` | (4·1.0 + 8·0.0) / 12 | 0.333 |
| `"dr_grpo"` | (4·1.0 + 8·0.0) / (2 × 16) | 0.125 |

Same rollout batch, three scalars differing by a factor of four. Under `"grpo"` each token of A enters the sum with coefficient 1/(4·2) and each token of B with 1/(8·2), because the per-sequence mean divides by that sequence's own length before the mean over the group: per-sequence normalization gives each token of a short completion a larger share of the gradient. The config docstring states the consequence — `grpo` normalization "tends to prefer shorter completions with positive advantages and longer ones with negative advantages" (`grpo_config.py` L239-L241). `dr_grpo` removes the length term entirely ([[dr-grpo]]).

At `a04ffd3` the three normalizer expressions above are unchanged; each is followed by a division by the current gradient-accumulation step count, and the `dapo`/`cispo`/`vespo` normalizer is additionally rescaled by `gradient_accumulation_steps / steps_per_generation` (`grpo_trainer.py` L3246-L3266). Those factors are common to every completion in the batch, so the relative weights in the worked example hold at the pinned commit.

**The default is `"dapo"`, not `"grpo"`** (`grpo_config.py` L796, `default="dapo"`), and `beta` defaults to `0.0` (L676), under which no reference model is created. A run left at defaults therefore optimizes the DAPO normalizer with no KL anchor. That is a deliberate choice the docs justify by citing Open-Reasoner-Zero, Dr. GRPO and DAPO ([[trl-grpo]]), but it is not the objective of [[grpo]] Eq. 3.

### §4.2 The KL estimator, stated correctly

TRL's GRPO KL term is `exp(ref − logp) − (ref − logp) − 1` (`grpo_trainer.py` L3193-L3194 at `a04ffd3`; L2493-L2497 at `a08e713`, [[trl-grpo]]). With `logr = log π_ref − log π_θ` and `r = exp(logr)`, that is `r − logr − 1`, which is Schulman's **k3** estimator. The naming facts, from [[john-schulman-kl-tricks]]:

| Estimator | Formula | Bias | Variance | Sign |
|---|---|---|---|---|
| k1 | `−log r` | unbiased | high | can be negative |
| k2 | `0.5 · (log r)²` | **biased** | low | non-negative |
| k3 | `(r − 1) − log r` | unbiased | low | non-negative |

**Worked example.** Two tokens, both sampled from π_θ.

- Token 1: log π_θ = −2.0, log π_ref = −2.3 → `logr` = −0.3, `r` = 0.7408. k1 = 0.300, k3 = 0.7408 + 0.300 − 1 = **0.041**.
- Token 2: log π_θ = −2.5, log π_ref = −2.2 → `logr` = +0.3, `r` = 1.3499. k1 = **−0.300**, k3 = 1.3499 − 0.300 − 1 = **0.050**.

Mean k1 over the two tokens is 0.000; mean k3 is 0.045. A single-sample k1 can be negative even though KL cannot, and the two samples cancel; k3 is per-sample non-negative. With a small number of tokens per step, a k1-based penalty can transiently reward the policy for moving away from the reference.

The old PPO trainer selected between k1 and k3 with `kl_estimator` (default `"k1"`) and applied the result inside the reward, not the loss: `non_score_reward = −kl_coef · kl`, added before GAE ([[trl-ppo]], `ppo_trainer.py` L775-L781). Its separately logged `approxkl` is the k2 form and measures the current policy against the **rollout** policy; the clip fraction is the distinct metric `pg_clipfrac` ([[trl-ppo]] Corrections). The two are different diagnostics and should not be read as one.

### §4.3 Multiple rewards and multiple domains

`GRPOTrainer` takes a list of `reward_funcs`, a `reward_weights` list (default: all 1.0, `grpo_config.py` L213-L215), and `multi_objective_aggregation` with two orders (L216-L224). The code (`grpo_trainer.py` L2787-L2826):

```python
if self.multi_objective_aggregation == "sum_then_normalize":
    rewards = (rewards_per_func * self.reward_weights...).nansum(dim=1)
    mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)...
    advantages = rewards - mean_grouped_rewards
    if self.scale_rewards != "none":
        advantages = advantages / (std_rewards + 1e-4)
elif self.multi_objective_aggregation == "normalize_then_sum":
    grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
    mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
    std_k = nanstd(grouped, dim=1, keepdim=True) ...
    reward_k = (grouped - mean_k) / (std_k + 1e-4)
    rewards = (reward_k * self.reward_weights...).nansum(dim=1)
    advantages = (rewards - torch.nanmean(rewards)) / (std_rewards + 1e-4)
```

`nanstd` uses the unbiased correction `count / (count − 1)` (`trl/trainer/utils.py` L881-L885), so the divisor below is the sample standard deviation.

**Worked example.** One prompt, G = 4 completions. Reward 1 is binary correctness `[1, 0, 1, 0]`. Reward 2 is a length penalty in raw token units `[−120, −100, −60, −80]`. `reward_weights = [1.0, 1.0]`, `scale_rewards="group"`.

*`sum_then_normalize`*: summed rewards `[−119, −100, −59, −80]`, mean −89.5, sample std 25.826.
Advantages: **[−1.14, −0.41, +1.18, +0.37]**.

*`normalize_then_sum`*: reward 1 normalized → `[0.866, −0.866, 0.866, −0.866]`; reward 2 normalized → `[−1.162, −0.387, +1.162, +0.387]`; weighted sum `[−0.296, −1.253, +2.028, −0.479]`; normalized again → **[−0.21, −0.89, +1.43, −0.34]**.

Read completion 1 and completion 4. Completion 1 is **correct** and the longest; under `sum_then_normalize` it receives the most negative advantage in the group (−1.14), so the policy is pushed away from a correct answer for being long. Completion 4 is **wrong** and of middling length; under `sum_then_normalize` it receives a positive advantage (+0.37). Under `normalize_then_sum` completion 4's advantage changes sign to −0.34, and completion 1's stays negative (−0.21) but rises above it, so the ordering of the four advantages places both correct completions above both wrong ones (+1.43 and −0.21 against −0.34 and −0.89); under `sum_then_normalize` the order is +1.18, +0.37, −0.41, −1.14, which puts a wrong completion above a correct one. The cause is scale: reward 2's raw spread is 60 units against reward 1's 1 unit, so `reward_weights=[1.0, 1.0]` does not mean equal influence under the default aggregation. Use **[figures/multi-reward-aggregation.html](figures/multi-reward-aggregation.html)** to change the two reward vectors and the weights and watch both advantage vectors update; it is the fastest way to see when the two orders disagree on sign.

**Implication for a general-purpose model.** Multi-domain RL is the standard way to avoid narrowing to one verifiable task: mix math, code, and instruction-following prompts in one `train_dataset` and give each domain its own reward function. That only works if each domain's reward contributes on a comparable scale. With `sum_then_normalize` and heterogeneous scales, a single domain's reward dominates the advantage and the mixture silently degenerates into single-domain RL.

### §4.4 Built-in reward functions

`trl/rewards/` ships `accuracy_reward`, `reasoning_accuracy_reward`, `get_cosine_scaled_reward` (`accuracy_rewards.py`), `think_format_reward` (`format_rewards.py`), and `get_repetition_penalty_reward`, `get_soft_overlong_punishment` (`other_rewards.py`) ([[trl-repo-a04ffd3]]). These are the format and length controls that appear in the recipe literature, available without writing a verifier.

---

## §5 Online and asynchronous trainers

### §5.1 Online DPO

`trl.experimental.online_dpo.OnlineDPOTrainer` samples exactly 2 completions per prompt (`self.num_generations = 2`, L339 at `a08e713`), scores each with `reward_funcs`, labels the higher-scoring one chosen, and takes a DPO or IPO step on that pair inside the same `training_step` ([[trl-online-dpo]]). No pair is stored. Two facts that older descriptions get wrong:

- **There is no judge interface at this commit.** TRL v1.1.0 (PR #5485) removed judge support; scoring is reward-functions only.
- **The reference model stays frozen.** What is on-policy is the pair, not π_ref.

The tie behaviour is a defect in this trainer: `mask = first_half >= second_half` (L1197 at `a08e713`, [[trl-online-dpo]]), so when both completions score identically — the common case for a binary correctness reward — the first sample is labelled chosen and the sigmoid DPO loss still has a nonzero gradient. Filter ties before using this trainer with binary rewards.

### §5.2 Asynchronous GRPO

`trl/experimental/async_grpo/` decouples rollout generation from the optimizer step. The staleness bound is explicit (`async_grpo_config.py` L120-L133):

```
max_inflight_tasks (defaults to -1): ... auto, which sets it to
    max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes
max_staleness (defaults to 4): Maximum number of weight update steps a rollout sample can lag
    behind the current model version before being discarded.
weight_sync_steps (defaults to 1): Number of training steps between weight synchronizations.
```

So a rollout produced under policy version *v* is usable until the policy reaches *v* + 4, after which it is discarded rather than importance-corrected. The claim that TRL is synchronous does not hold at this commit, so a statement about TRL's concurrency model needs a commit attached to it.

The config also defaults `dtype="float32"` and explains why in the docstring (L36-L46): the trainer is measured against the training-inference numerical mismatch described in "Defeating the Training-Inference Mismatch via FP16" (arXiv:2510.26788), and closing the gap requires serving the vLLM side in the same dtype; a mismatch is logged as a warning at train start. `vllm_mode` remains `"server"` or `"colocate"` in `GRPOConfig` (L589, default `"colocate"`); neither is deprecated at this commit.

---

## §6 Distillation trainers

### §6.1 The stable on-policy trainer

`trl/trainer/distillation_trainer.py` is a supported trainer, and its docstring states the method:

```
Trainer for knowledge distillation. The student is trained on-policy — it generates the completions itself — to
match the teacher's next-token distribution under a generalized Jensen-Shannon divergence (interpolating forward
KL, reverse KL, and JSD via `beta`), as introduced in On-Policy Distillation of Language Models
(https://huggingface.co/papers/2306.13649).
```

That paper is [[gkd-on-policy-distillation]] (Agarwal et al., ICLR 2024). Its Eq. 1 defines the generalized JSD with mixture M = βP + (1−β)Q, P the teacher and Q the student:

  D_JSD(β)(P ‖ Q) = β · D_KL(P ‖ M) + (1 − β) · D_KL(Q ‖ M)

- β — interpolation coefficient in (0, 1). P — teacher next-token distribution at a position; Q — student distribution at the same position. M — the β-mixture of the two.
- The endpoints are limits, not values of the expression: the paper cites Huszár (2015) for lim_{β→0} D_JSD(β)(P ‖ Q)/β = D_KL(P ‖ Q), and states that "gradients of JSD(β) behave similarly to forward KL and reverse KL when β is close to 0 and 1 respectively" (§2). The expression itself goes to 0 at both ends.
- TRL does not evaluate the mixture at the endpoints. `beta=0.0` is dispatched to the exact forward KL D_KL(P ‖ Q) and `beta=1.0` to the exact reverse KL D_KL(Q ‖ P); only 0 < β < 1 uses the mixture form. The table below therefore reports what TRL computes at each setting.

The endpoint and mixture branches are at `distillation_trainer.py` L142-L153, and the default is `beta=1.0` (`distillation_config.py` L354) — reverse KL, the mode-seeking end.

**Worked example.** Three-token vocabulary, one position. Teacher P = (0.7, 0.2, 0.1); student Q = (0.4, 0.4, 0.2).

| β | Divergence | Value (nats) |
|---|---|---|
| 0.0 | forward KL, D_KL(P ‖ Q) | 0.7·ln1.75 + 0.2·ln0.5 + 0.1·ln0.5 = **0.184** |
| 0.5 | JSD with M = (0.55, 0.30, 0.15) | 0.5·0.047 + 0.5·0.045 = **0.046** |
| 1.0 | reverse KL, D_KL(Q ‖ P) | 0.4·ln(4/7) + 0.4·ln2 + 0.2·ln2 = **0.192** |

Two things to take from the arithmetic. First, the interior of the β range is not merely "between" the endpoints in direction — it is roughly four times smaller in magnitude here, so a learning rate tuned at β = 1.0 is not transferable to β = 0.5. Second, the direction matters for breadth: forward KL is mass-covering (it penalizes the student for putting near-zero probability where the teacher has mass), reverse KL is mode-seeking.

### §6.2 Evidence on the divergence choice

[[gkd-on-policy-distillation]] measures both.

- **Task-agnostic distillation.** FLAN T5-XL teacher, FLAN T5-Base student, FLAN2021 (5.36M examples across 62 tasks), 50K training steps, evaluated on held-out MMLU (57 tasks) and BBH (23 tasks) by few-shot prompting, averaged over tasks. §4.4 reports that on-policy GKD with **reverse KL** "substantially outperforms supervised KD and ImitKD" and that reverse KL performs better than forward KL in this setting; the size of the gain is given only in §1 as "2% and 1% absolute accuracy improvement on the held-out BBH and MMLU benchmark suites (Figure 10)", without naming the baseline it is measured against. Fig. 10's caption gives the starting points: teacher FLAN T5-XL at 52.4% MMLU and 41% BBH, the student at 35.6% and 31.25%. The authors' explanation — labelled **Interpretation** in the paper — is that reverse KL's mode-seeking behaviour makes the model "zero in on the main intent or behavior specified by the instruction".
- **Diversity cost.** Moving from forward KL through generalized JSD to reverse KL "leads to decreased diversity, attributed to the enhanced mode-seeking characteristic of the divergence", measured by Self-BLEU across sampling temperatures (Fig. 4 caption). Mode-seeking divergences gave better quality at high temperature.
- **Aggregate task-specific gains.** Averaged over T5 student sizes (77M, 250M, 800M), on-policy GKD improved on the initial student by 2.1× on summarization (XSum), 1.7× on translation (WMT), and 1.9× on arithmetic reasoning (GSM8K), relative to the improvements the baseline KD methods achieved over the same student (§1, Fig. 1).

**Result (single study)** in all three cases: one paper, T5-family students up to 800M, four task families. It was not tested at modern decoder-only scales.

### §6.3 GKD and the other distillation trainers

`trl/experimental/gkd/` implements Algorithm 1 of the paper with the student-data fraction exposed as `lmbda` (default 0.5, `gkd_config.py` L59): λ = 1 is fully on-policy, λ = 0 is supervised KD on a fixed dataset. `beta` defaults to 0.5, `temperature` to 0.9, `max_new_tokens` to 128, `seq_kd` to `False` (sequence-level KD, "supervised FT on teacher-generated output"). `GKDConfig` subclasses `SFTConfig`, so all of §2's masking and packing fields apply to it.

Alongside GKD the experimental tree holds `async_distillation/`, `server_distillation/`, `minillm/`, `gold/`, `sdft/`, `sdpo/`, `ssd/`, and `iw_opd/` — importance-weighted off-policy distillation. The on-policy-distillation course covers the algorithm family in its own ch-06 (`wiki/courses/on-policy-distillation/ch-06/`); this chapter covers only which trainer implements which variant at this commit.

**Why distillation belongs in a generality chapter.** Distillation is the one post-training stage whose target is a full distribution rather than a single token, so it transfers the teacher's behaviour on inputs that appear in no training label. The task-agnostic result above is the evidence for that: the gain is measured on held-out benchmark **suites**, not on the distillation data.

---

## §7 LoRA versus full fine-tuning, measured

### §7.1 The measurable trade-off

[[lora-learns-less-forgets-less]] compares LoRA and full fine-tuning on Llama-2-7B in two regimes — instruction fine-tuning (≈100K prompt-response pairs) and continued pretraining (≈20B tokens) — on two target domains, code and math. Two metrics:

- **Learning**: HumanEval pass@1 (code) and GSM8K strict match (math).
- **Forgetting**: the average of HellaSwag, ARC-Challenge, and WinoGrande scores (§3.2, Fig. 2 caption).

Reported numbers, each with its setting:

| Regime | Target metric | Forgetting metric |
|---|---|---|
| Code IFT | LoRA r=16: 0.358; r=64: 0.417; r=256: 0.498 (all at epoch 4); full FT: 0.497 (epoch 8) | at epoch 16: full FT 0.414 vs LoRA r=64 **0.509** (§4.2 prose; Table S6). At epoch 4: full FT 0.512, LoRA r=64 0.632, r=256 0.631 |
| Code CPT, 20B tokens | LoRA underperforms full FT at every rank (§4.1) | full FT 0.545 vs LoRA r=256 **0.617** (Table S2) |
| Math CPT | LoRA r=256 peaks 0.203 at 16B; full FT 0.293 at 20B | LoRA 0.616 (20B) vs full FT 0.613 (16B) — no gap |
| Math IFT | LoRA r=256: 0.634 at 8 epochs; full FT: 0.642 at 4 epochs | LoRA 0.567 vs full FT 0.559 at epoch 16 — no gap |

**Conditions and limits.** One base model (Llama-2-7B), two domains, one forgetting suite of three multiple-choice benchmarks. The forgetting advantage is large in **code** and absent in **math**; the paper attributes this to domain shift, since OpenWebMath is dominated by English sentences (§4.2). Do not carry the code numbers to a math or dialogue run.

### §7.2 Diversity

On HumanEval, counting unique output strings out of 50 generations, full fine-tuning produced fewer unique generations than the base model for both passing and failing solutions, with LoRA between the two (§4.5, Fig. 5). The authors state the metric's limit themselves: "exact string matching between generations is not a sensitive metric of predictive diversity". **Result (single study)**, one benchmark, string-level metric.

### §7.3 The paper's configuration recommendations

Quoted from §4.7: "(a) using LoRA for instruction finetuning and not continued pretraining; (b) if GPU memory allows, targeting 'All' transformer modules with a rank of 256, since ranks 16 − 64 tend not to suffice for code tasks; (c) using α = 2r, and (d) sweeping over learning rates between [1e−5, 5e−4], picking the highest value that enables stable training." The α = 2r point matters because PEFT scales the adapter by α/r, so a fixed α scales high ranks down. Targeting "Attention" alone underperformed "MLP" and "All" (Fig. 7).

### §7.4 How to measure forgetting in a TRL run

The paper's metric is reproducible inside a TRL run without extra infrastructure: hold out three multiple-choice benchmarks that the fine-tuning data does not target, score them at the base checkpoint, score them again at each saved checkpoint, and report the average. Because the metric is a mean over log-likelihood-scored multiple choice, it needs no generation and no sampling parameters (§3.2). Run it through an external harness (`lm-eval-harness`, `lighteval`) on the saved checkpoints rather than inside `compute_metrics`, so the numbers are comparable to published ones.

---

## §8 Evaluation and merging inside the loop

`trl/trainer/callbacks.py` at `a04ffd3` defines `SyncRefModelCallback`, `RichProgressCallback`, `LogCompletionsCallback`, `WeaveCallback`, and `BEMACallback`. For breadth measurement the relevant ones are:

- **`LogCompletionsCallback`** (`callbacks.py` L263; `freq` docstring L283-L284, resolved at L314) — generates completions for the prompts in the trainer's `eval_dataset` every `freq` steps, where `freq` defaults to the trainer's `eval_steps`, and writes a step / prompt / completion table to Weights & Biases or Comet. It requires an eval dataset with a `prompt` column. Qualitative, but it is what catches template breakage and non-stopping models early.
- **`eval_dataset` plus `num_generations_eval`** (`grpo_config.py` L67-L69 docstring, L486 field, default `None` → `num_generations`) — GRPO can evaluate with fewer samples per prompt than it trains with, so an out-of-domain eval prompt set is affordable at every eval step.
- **`MergeModelCallback`** (`trl/experimental/merge_model_callback.py` L294, constructor L322-L331) — merges the training checkpoint with a target model using `mergekit`, at every checkpoint when `merge_at_every_checkpoint=True` (default `False`) and otherwise at the end of training, with optional `push_to_hub`. `MergeConfig` (L48-L112) takes `method` in `{"linear", "ties", "dare_ties", "slerp"}`; defaults are `linear` with weights 0.5 / 0.5, `ties` and `dare_ties` with policy density `[1.0, 0.7, 0.1]`, `slerp` with `t_values` 0.5, all in float16. This is the in-framework hook for weight-averaging a fine-tuned checkpoint back toward the model it started from, the standard post-hoc recovery move for capability lost during fine-tuning. TRL supplies the mechanism; it reports no evaluation of it.

---

## Negative samples and negative feedback

Which sense of "negative" applies depends on the trainer. Using the four-way distinction:

| Trainer | Sense of negative | Mechanism |
|---|---|---|
| `SFTTrainer` | (1) negative marginal value | Failures are excluded from the dataset upstream; the trainer has no mechanism for them. `loss_type="dft"` reweights tokens but does not push any down. |
| `DPOTrainer` | (4) negative as gradient | The rejected term of `delta_score = chosen_scores − rejected_scores` (L1468) lowers log π_θ(y_r). |
| `OnlineDPOTrainer` | (4) | The lower-reward of two on-policy samples becomes y_r. |
| `GRPOTrainer` | (4) | Any completion with A < 0 has its tokens' log-probabilities pushed down. |
| `GRPOTrainer` tool loop | (2) negative as content | Tool exceptions become `{"error": ...}` messages the model reads on the next turn; those tokens are masked out of the loss by `tool_mask` ([[trl-grpo]]). |

**Mechanism.** For the softmax, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`. A negative-advantage step on token y subtracts that gradient, so probability mass leaves y and is redistributed in proportion to the current p_j — that is, it concentrates on whichever token is already most likely. When y was already unlikely, the mass moved is small but its destination is uncontrolled. [[ch-43a]] holds the derivation and the displacement evidence.

**Controls TRL exposes**, with their loci:

1. **Bound the ratio on negatives.** `delta` applies an upper clamp to the importance ratio; combined with `min` over the two clipped terms it binds only when A < 0. Worked example from [[trl-grpo]]: with ε = 0.2 and δ = 3, A = −1 and r = 5 gives `min(−3, −1.2) = −3` — loss 3 with zero gradient in r; without δ the loss is 5. With A = +1 and r = 5 the objective is 1.2 with or without δ.
2. **Drop stale negatives.** The off-policy sequence mask discards a sequence only if A < 0 **and** its mean KL between the sampling and current policy exceeds `off_policy_mask_threshold` (`grpo_trainer.py` L3038-L3062, applied at L3161).
3. **Do not penalize truncation.** `mask_truncated_completions` (default `False`) removes truncated completions from the loss so a completion is not pushed down for hitting the length budget (`grpo_config.py` L267-L270 docstring, L831 field; the docstring cites DAPO).
4. **Asymmetric sign handling.** `sapo` uses `sapo_temperature_neg` = 1.05 for non-positive advantages against `sapo_temperature_pos` = 1.0 for positive ones (`grpo_config.py` L188-L193 docstring, L709 and L717 fields).
5. **Anchor with a positive NLL term.** `DPOConfig(loss_type=["sigmoid", "sft"], loss_weights=[1.0, 0.2])` adds cross-entropy on the chosen completion (§3.1).

**Diagnostics TRL logs.** `DPOTrainer` logs `rewards/chosen` and `rewards/rejected` **separately**, plus `rewards/margins`, `rewards/accuracies`, `logps/chosen` and `logps/rejected` (`dpo_trainer.py` L1346-L1363). Watching the margin alone hides the failure mode where both chosen and rejected log-probabilities fall and the margin still rises. `GRPOTrainer` logs sign-split clipping metrics — the low-clip metric counts r < 1 − ε_low only where A < 0, the high-clip counts r > 1 + ε_high only where A > 0 (`grpo_trainer.py` L3367-L3368 at `a04ffd3`) — plus `frac_reward_zero_std`, the fraction of groups whose completions all received the same reward and therefore produced A = 0 ([[trl-grpo]]). Entropy is logged every step; see [[entropy-logging-patterns]] for the cross-framework conventions and what each framework's "entropy" actually sums over.

**False negatives.** No TRL source reports a false-negative rate for any reward function; that number belongs to the verifier, not the framework.

**Effect on generality.** The size-of-effect question is not answered by any TRL source. Do not claim that the negative term is the main driver of a DPO or GRPO gain unless a study measures that split for the setting in question.

---

## Recipe

All framework rows are **framework defaults at one commit**, which §5.3 of the authoring standard treats as a distinct class of fact from a paper value or a released run config. Loci are file and line at `a04ffd337108285002df29d9ae73ef7e855e03f1` unless stated.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| TRL `a04ffd3` (v1.13.0.dev) | — | SFT | max_length; packing; packing_strategy; completion_only_loss; assistant_only_loss; loss_type; learning_rate | 1024; False; "bfd"; None; False; "chunked_nll"; 2e-5 | `trl/trainer/sft_config.py` L203, L223, L230, L259, L271, L281+L332, L141 [[trl-repo-a04ffd3]] | verified 2026-09-17 | no ablation reported |
| TRL `a04ffd3` | — | preference | loss_type; loss_weights; beta; f_divergence_type; learning_rate | ["sigmoid"]; None (→ all 1.0); 0.1; "reverse_kl"; 1e-6 | `trl/trainer/dpo_config.py` L241, L250, L293, L267, L155 | verified 2026-09-17 | no ablation reported |
| TRL `a04ffd3` | — | RL | loss_type; beta; scale_rewards; num_generations; max_completion_length; epsilon; top_entropy_quantile; multi_objective_aggregation; use_vllm; vllm_mode | "dapo"; 0.0; "group"; 8; 512; 0.2; 1.0; "sum_then_normalize"; False; "colocate" | `trl/trainer/grpo_config.py` L796, L676, L784, L479, L493, L688, L861, L772, L582, L589 [[trl-repo-a04ffd3]]; loss code [[trl-grpo]] | verified 2026-09-17 | no ablation reported; docstring cites DAPO ε_high 0.28 and "Beyond the 80/20 Rule" ρ 0.2 as **paper** values |
| TRL `a04ffd3` | — | distill-SFT | beta; temperature; max_completion_length; learning_rate; gradient_checkpointing | 1.0 (reverse KL); 1.0; 512; 1e-6; True | `trl/trainer/distillation_config.py` L354, L245, L222, L171, L155-L160 | verified 2026-09-17 | no ablation reported |
| TRL `a04ffd3` | — | distill-SFT | GKD: temperature; lmbda; beta; max_new_tokens; seq_kd | 0.9; 0.5; 0.5; 128; False | `trl/experimental/gkd/gkd_config.py` L55, L59, L66, L74, L96 | verified 2026-09-17 | no ablation reported |
| TRL `a04ffd3` | — | RL | async GRPO: dtype; num_generations; max_completion_length; max_staleness; weight_sync_steps; queue_maxsize; lr_scheduler_type | "float32"; 8; 2048; 4; 1; 1024; constant | `trl/experimental/async_grpo/async_grpo_config.py` L177, L232, L236, L358, L369, L365, L143-L149 | verified 2026-09-17 | docstring cites arXiv:2510.26788 for the float32 default |
| TRL `a08e713` (deleted in v1.13.0) | — | RL | PPO: kl_coef; vf_coef; cliprange; num_ppo_epochs; gamma; lam; response_length; temperature; kl_estimator | 0.05; 0.1; 0.2; 4; 1.0; 0.95; 53; 0.7; "k1" | `trl/experimental/ppo/ppo_config.py` L232-L264 [[trl-ppo]], [[trl-ppo-recipe]] | verified 2026-09-14 | one documented benchmark run: Pythia-1B, TL;DR (docs L173-L202) |
| Zephyr-7B SFT (Alignment Handbook) | 7B | SFT | base; max_seq_length; LR; schedule; warmup; epochs; per-device batch; grad accum; precision | Mistral-7B-v0.1; 2048; 2.0e-05; cosine; 0.1; 1; 16; 1; bf16 | `recipes/zephyr-7b-beta/sft/config_full.yaml` @`1de1fc9` [[hf-alignment-handbook]] | verified 2026-09-17 | no ablation reported; config sets neither packing nor response-only masking |
| GKD (on-policy distillation) | T5-Base student | distill-SFT | teacher; data; steps; divergence; λ | FLAN T5-XL; FLAN2021 5.36M examples / 62 tasks; 50K steps; reverse KL; 1 (on-policy) | arXiv:2306.13649v3 §4.4, Fig. 10 [[gkd-on-policy-distillation]] | verified 2026-09-17 | Fig. 10: +2% BBH, +1% MMLU absolute over the initial student |
| LoRA IFT (paper recommendation) | Llama-2-7B | SFT | target modules; rank; α; LR sweep | "All"; 256; 2r = 512; [1e−5, 5e−4], highest stable | arXiv:2405.09673v2 §4.7 [[lora-learns-less-forgets-less]] | verified 2026-09-17 | Fig. 7 (module sweep); Fig. S3 (α × LR sweep at r = 256) |

**Starting point for a small general-purpose run.** For an 8B model on one node, the verified rows above support: SFT with `SFTConfig(max_length=2048, packing=True, packing_strategy="bfd", assistant_only_loss=True, learning_rate=2e-5, lr_scheduler_type="cosine", warmup_ratio=0.1, num_train_epochs=1)` — the length, learning rate, schedule, warmup and epoch count are the Zephyr-7B values for a 7B Mistral base on ≈200K UltraChat dialogues, and `packing` and `assistant_only_loss` are TRL fields the Zephyr config does not set, so treat them as this course's choice rather than a reproduced value. Then `DPOConfig(loss_type=["sigmoid"], beta=0.1, learning_rate=1e-6)` at TRL's own default. Then, if moving to RL, set `GRPOConfig(loss_type=..., beta=..., scale_rewards=...)` explicitly rather than accepting `"dapo"`/`0.0`/`"group"`, and set `multi_objective_aggregation="normalize_then_sum"` whenever more than one reward function is in use. None of these TRL defaults has a published ablation; every number above that does have supporting evidence carries it in the last column.

---

## Generalization lens

**(a) What increases breadth.**
- **On-policy distillation against a stronger teacher.** Held-out-suite evidence: on-policy GKD with reverse KL outperformed supervised KD and ImitKD on BBH and MMLU, with the gain reported as 2% and 1% absolute, where the training data was FLAN2021 and neither suite is in it ([[gkd-on-policy-distillation]] §1 and §4.4, Fig. 10; the paper does not state the baseline for the 2% / 1% figures).
- **Multiple reward functions over mixed-domain prompts**, aggregated on a common scale (`normalize_then_sum`), keeps more than one domain's gradient visible in the advantage (§4.3 worked example; mechanism, no published TRL ablation).
- **LoRA at rank 256 with α = 2r** matched full fine-tuning on code IFT target accuracy (0.498 at epoch 4 against 0.497 at epoch 8) while scoring higher on the forgetting metric at both of those points (0.631 against 0.446) ([[lora-learns-less-forgets-less]] §4.1-§4.2, Tables S5 and S6). **Result (single study)**, Llama-2-7B.
- **`loss_type=["sigmoid", "sft"]` with weights** keeps a positive NLL anchor in the preference objective (`dpo_trainer.py` L1581-L1587). Mechanism; TRL reports no evaluation.

**(b) What causes narrowing or forgetting.**
- **Full fine-tuning on a distant domain.** Code CPT at 20B tokens: forgetting metric 0.545 for full FT against 0.617 for LoRA r = 256 ([[lora-learns-less-forgets-less]] Table S2). Output diversity also fell: fewer unique HumanEval generations out of 50 than the base model, for both passing and failing solutions (§4.5, Fig. 5).
- **Mode-seeking distillation.** Moving from forward KL toward reverse KL decreased generation diversity as measured by Self-BLEU ([[gkd-on-policy-distillation]] Fig. 4). TRL's `DistillationConfig` default is `beta=1.0`, the reverse-KL end.
- **Training on the full rendered conversation.** Roughly a third of the gradient on a typical two-round example goes to template and user tokens (§2.3), which trains the model to produce turn structure rather than answers.
- **Scale-dominated multi-reward aggregation.** Under `sum_then_normalize` a reward with a large raw spread can flip the advantage sign on correct completions (§4.3).
- **A single verifiable reward on a single domain** is the general narrowing mechanism; TRL's contribution is the `reward_funcs` list that makes the alternative cheap.

**(c) How to measure it for this stage.**
- **Forgetting**: the [[lora-learns-less-forgets-less]] metric — average of HellaSwag, ARC-Challenge, WinoGrande, scored at the base checkpoint and at every saved checkpoint. No generation, so no sampling-parameter confound (§3.2).
- **Coverage**: pass@k at large k on a held-out task, not pass@1. The same paper's Appendix F found full fine-tuning superior to LoRA r = 256 for k < 64 and equal above it — a difference invisible at k = 1.
- **Diversity**: unique generations out of a fixed sample count, with the caveat the authors state (string matching is coarse), or Self-BLEU as in [[gkd-on-policy-distillation]].
- **In-loop breadth**: a held-out `eval_dataset` of out-of-domain prompts with `num_generations_eval` set below `num_generations`, plus `LogCompletionsCallback` to read actual outputs (§8).
- **Known measurement error**: TRL's own entropy and KL metrics are not uniformly masked across trainers. At `a08e713` the PPO trainer's `objective/entropy` summed `−logprobs` over padded positions where the value is the sentinel `INVALID_LOGPROB = 1.0`, adding −1.0 per padded token until commit `190c39c` ([[trl-ppo]]); `OnlineDPOTrainer`'s `objective/kl` and `objective/entropy` sum over all completion positions with no padding mask, and its `mean_entropy` is the negative sampled log-probability, not the entropy of the token distribution ([[trl-online-dpo]]). Compare these numbers only within one trainer at one commit.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Using `train_on_response_only=True` or `max_seq_length=` in `SFTConfig` | `TypeError` on unexpected keyword, or the field is silently accepted by a wrapper and ignored | Print `SFTConfig.__dataclass_fields__.keys()` at the pinned commit; the fields are `assistant_only_loss`, `completion_only_loss`, `max_length` |
| Leaving masking at defaults on a conversational dataset | Model generates a fabricated user turn after its answer; training loss lower than expected | Decode one batch's `labels` with `-100` positions removed and confirm only assistant text remains |
| Chat template without `{% generation %}`, or with the end-of-turn token outside the mask | Model does not stop; generation runs to `max_new_tokens` | Search the training log for "does not include the assistant turn's end-of-turn token in the loss mask" (`sft_trainer.py` L1271-L1277) |
| Assuming `loss_type="kto"` / `"simpo"` / `"orpo"` on `DPOTrainer` | `ValueError: Unknown loss type` listing the 15 valid values (`dpo_trainer.py` L1597-L1600) | Read `dpo_config.py` L81-L88; use `CPOTrainer(loss_type="simpo")` and the separate experimental trainers |
| Reproducing a GRPO paper with default config | Results differ from the paper; no reference model in memory | Log `args.loss_type`, `args.beta`, `args.scale_rewards` at train start and compare with the paper's stated objective |
| Reading `approxkl` as the policy-to-reference KL | KL looks small while the policy has drifted far from the SFT model | `approxkl` is policy-vs-rollout-policy (k2 form); the reference KL is `objective/kl` ([[trl-ppo]] Corrections) |
| Treating k1 as the biased estimator | Wrong reasoning about why a KL penalty is noisy | k1 unbiased/high variance, k2 biased/low variance, k3 unbiased/low variance/non-negative ([[john-schulman-kl-tricks]]) |
| Equal `reward_weights` with unequal reward scales | Reward curves for the small-scale reward flat; advantage tracks only the large-scale reward | Log each reward function separately; switch to `multi_objective_aggregation="normalize_then_sum"` and compare advantage vectors |
| Binary reward with `OnlineDPOTrainer` | `rewards/accuracies` near 0.5 and the loss does not fall | Ties are labelled chosen-first at L1197; filter zero-margin pairs before the step |
| Importing `PPOTrainer` from a pinned recent version | `ImportError` | Deleted by PR #7020 on 2026-09-04, released in v1.13.0; pin an earlier release or use `RLOOTrainer`/`GRPOTrainer` |
| Comparing entropy across trainers or commits | Entropy "collapse" or "growth" that is an artifact | Check each trainer's masking at the commit in use ([[entropy-logging-patterns]]) |

---

## Check your understanding

1. A conversational SFT run leaves `assistant_only_loss` and `completion_only_loss` at their defaults. Trace the mechanism from that configuration to a model that emits a fabricated user turn at inference, and say why the training loss curve does not reveal it.
2. The same rollout batch produces scalar losses 0.500, 0.333 and 0.125 under `"grpo"`, `"bnpo"` and `"dr_grpo"`. Explain, from the normalizer expressions, which completions gain relative weight under each, and predict which normalizer produces the strongest pressure toward short completions when advantages are positive.
3. Why can a single-sample k1 KL estimate be negative when KL divergence cannot be, and what does that imply for a run that applies a k1-based penalty with a small number of tokens per optimizer step?
4. Two reward functions with the same weight produce opposite advantage signs for the same completion under the two `multi_objective_aggregation` orders. Derive the condition on the two reward vectors under which the orders disagree in sign.
5. TRL's `DistillationConfig` defaults to `beta=1.0` and the GKD paper reports that reverse KL both wins on held-out instruction-tuning suites and reduces generation diversity. Explain how one mechanism produces both results, and state what you would measure before accepting the default for a general-purpose student.
6. LoRA forgets less than full fine-tuning on code but not on math in the Biderman et al. experiments. Give a causal account of that difference, and say what it predicts for a dialogue-domain fine-tune.
7. Online DPO keeps a frozen reference model while sampling fresh pairs each step. Explain precisely which quantity in the DPO loss becomes on-policy and which does not, and what that implies for how far the policy can drift before the loss stops constraining it.
8. You are handed a TRL training script with no commit pin and asked whether it reproduces a 2025 GRPO paper. List, in order, the four configuration facts you would check first and say what each one could silently change.

---

## Connections

- Previous: **ch-56** — OpenRLHF Internals: PPO, DPO, Ray Orchestration, and Forgetting Controls. Supplies the Ray-based comparison for §1 and §5.2.
- Next: **ch-58** — Choosing and Instrumenting a Post-Training Stack to Measure and Protect General Capability. Turns §7 and §8 into a selection and instrumentation procedure.
- **ch-55** — verl Internals: Losses, Rollouts, Multi-Domain Rewards, and In-Loop Validation. The registry-based split of advantage and loss computation that TRL fuses into one `loss_type` switch.
- **ch-43a** — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. Holds the logit-gradient derivation the negatives section applies.
- **ch-40** — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO. The algebra that §4.1's normalizers implement.
- **ch-43** — Entropy, Output Diversity, and KL Control in RL. The β and KL-estimator discussion in §4.2.
- **ch-32c** — Claimed versus Effective Context Length and Long-Context Evaluation. The truncation failure that §2.4's `"bfd"` packing strategy produces on long examples.
- **ch-58a** — Open General-Model Recipes End to End: Pretraining to Merge. Where the merging callback of §8 appears inside a full recipe.

---

## Sources

- [[trl-repo-a04ffd3]] — chapter excerpt: repository tree at the pinned commit, and the `SFTConfig`, `DPOConfig`, `DistillationConfig`, `GKDConfig`, `AsyncGRPOConfig` field quotes with line numbers.
- [[trl-grpo]] — `_compute_loss` loss-type normalizers, advantage and multi-reward code, k3 KL term, entropy mask, off-policy sequence mask, tool-token masking, sign-split clip metrics, and the default `loss_type="dapo"` / `beta=0.0`.
- [[trl-grpo-recipe]] — the framework-default ledger for `GRPOConfig`.
- [[trl-ppo]] — the deleted actor-critic trainer: separate value model, KL-in-reward, k1/k3 selection, `approxkl` versus `pg_clipfrac`, the entropy-masking defect, and the PR #7020 removal.
- [[trl-ppo-recipe]] — PPO framework defaults and the documented Pythia-1B TL;DR run.
- [[trl-online-dpo]] — two-samples-per-prompt pairing, tie handling at L1197, frozen reference, absence of a judge interface, and the unmasked KL and entropy metrics.
- [[trl-online-dpo-recipe]] — online-DPO framework defaults and the documented runs.
- [[gkd-on-policy-distillation]] — chapter excerpt: generalized JSD definition, Algorithm 1 and λ, task-agnostic FLAN result on held-out MMLU and BBH, and the divergence-versus-diversity trade-off.
- [[lora-learns-less-forgets-less]] — chapter excerpt: learning and forgetting metrics, the code and math IFT/CPT numbers, the diversity count, and the §4.7 configuration recommendations.
- [[hf-alignment-handbook]] — the Zephyr-7B SFT config values used in the Recipe table. The card's "keep `packing: true` and `train_on_response_only: true`" guidance and its `train_on_response_only` code snippet do not match the config or the current TRL API and are corrected in §2.5.
- [[hf-dpo-zoo]] — the DPO-variant survey. Its claim that all variants are exposed through `DPOTrainer`'s `loss_type` is corrected in §3.1 against `dpo_config.py` L81-L88 and the repository tree.
- [[grpo]] — the DeepSeekMath objective and KL estimator that `loss_type="grpo"` and the k3 term follow.
- [[dr-grpo]] — the length-free normalizer implemented as `loss_type="dr_grpo"`.
- [[dpo]] — the closed-form preference loss behind `delta_score`.
- [[ipo]] — the squared loss implemented as `loss_type="ipo"`, including TRL's length normalization note.
- [[ppo]] — the clipped surrogate the deleted PPO trainer implemented.
- [[john-schulman-kl-tricks]] — the k1/k2/k3 definitions and their bias and variance properties.
- [[rlhf-instructgpt]] — the reference-KL-anchored RLHF pipeline that `beta=0.0` removes.
- [[verl-grpo]], [[openrlhf-ppo]], [[openrlhf-dpo]] — the same losses in the two frameworks of [[ch-55]] and [[ch-56]].
- [[entropy-logging-patterns]] — what each framework's entropy metric sums over, needed before comparing TRL numbers with verl or OpenRLHF numbers.
