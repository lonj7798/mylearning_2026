<!-- scope: HuggingFace TRL OnlineDPOTrainer (trl/experimental/online_dpo), pinned at commit a08e713 (2026-04-21): on-policy pair sampling, reward-function scoring, DPO/IPO loss
     deps: [[dpo]], [[ipo]]
     see-also: [[trl-online-dpo-recipe]], [[self-play-preference]], [[on-off-policy-rlhf]], [[openrlhf-dpo]], [[trl-grpo]]
-->

# Online DPO Trainer — TRL `trl/experimental/online_dpo/online_dpo_trainer.py`
- **Core Insight:** In every `training_step`, TRL's `OnlineDPOTrainer` samples 2 completions per prompt from the current policy, scores each completion with reward functions, labels the higher-scoring completion `chosen` and the other `rejected`, and applies the sigmoid DPO loss (default β 0.1) or the IPO loss to those pairs in the same step (online_dpo_trainer.py L339, L1187–1230; online_dpo_config.py L239–254).
- **Guideline:** When the reward functions can give both completions the same score (for example a binary correctness reward), handle ties before relying on this trainer, because at this commit a tied pair is still trained with the first completion as `chosen` (L1197) and the sigmoid DPO loss has a nonzero gradient for every pair.
- **Authors:** Hugging Face TRL; the docs credit Michael Noukhovitch, Shengyi Costa Huang, Quentin Gallouédec, Edward Beeching (docs/source/online_dpo_trainer.md L13). Method: Guo et al., "Direct Language Model Alignment from Online AI Feedback", arXiv:2402.04792 (trainer L156–167; docs L7).
- **Year:** 2026 snapshot (commit a08e713, 2026-04-21, VERSION `1.3.0.dev0`); moved to `trl.experimental` in v0.26.0 (2025-12-09, PR #4473); still present at main commit 0929acf (2026-09-10)
- **URL:** https://github.com/huggingface/trl/blob/a08e7139f933b770177fc2abc0b43118e26b260b/trl/experimental/online_dpo/online_dpo_trainer.py
- **Source type:** released config/code
- **Relevant topics:** online DPO, on-policy preference pairs, reward models as labelers, IPO loss, missing-EOS penalty, vLLM generation

## Summary
`OnlineDPOTrainer` takes a prompt-only dataset (docs L68) and builds preference pairs during training. For each batch it
generates two completions per prompt, computes one scalar reward per completion, forms one pair per prompt, computes a
DPO or IPO loss on those pairs, and calls backward. No pair is stored for later steps: generation, scoring, pairing, and the loss are
all inside one `training_step` call (L1089–1288). The OAIF abstract quoted in the docs describes an LLM annotator that
chooses the preferred response of each pair (docs L11). At this commit the trainer has no annotator or judge interface:
TRL v1.1.0 (2026-04-12, PR #5485) removed judge support, leaving `OnlineDPOTrainer`, `NashMDTrainer`, and `XPOTrainer`
"unified on reward-model scoring only" (release notes).

## Key Contributions
- On-policy pair construction with a fixed 2 samples per prompt (`self.num_generations = 2`, L339).
- Reward functions given as a model id (loaded as `AutoModelForSequenceClassification` with `num_labels=1`), a `PreTrainedModel`, or a Python callable; several functions are combined by a weighted `nansum` (L207–217, L1005–1049).
- Two loss types, `sigmoid` (DPO) and `ipo` (L1223–1228; config L248–254).
- Three generation paths: transformers `generate` (L893–1003), vLLM server (L615–689), vLLM colocate (L691–730).
- Base class of `XPOTrainer` (xpo_trainer.py L47) and `NashMDTrainer` (nash_md_trainer.py L106) in the same commit.

## Key Figures/Tables to Study
- online_dpo_trainer.py L1187–1230: reward computation, pairing, and loss.
- online_dpo_trainer.py L1005–1049: how rewards from models and callables are computed and combined.
- docs L111–128: definitions of the logged metrics, including the implicit-reward accuracy.

## Technical Details
1. Generation. In the transformers path, prompt ids are repeated with `.repeat(2, 1)` (L962), so the batch is ordered [first sample of every prompt, then second sample of every prompt]. The colocate vLLM path samples `n=2` per prompt (L477) and produces the same block order (L725–726). Default sampling: temperature 0.9, top_p 1.0, top_k 0, `max_new_tokens` 64 (config L173–202).
2. Log-probs. `_forward` left-truncates the prompt so prompt plus completion fits `max_length` (L1053–1057) and reads completion log-probs from `log_softmax` of the unscaled logits (L1083–1086). Reference log-probs come from a `create_reference_model` copy, or from the same model with the PEFT adapter disabled (L302–306, L1155–1164).
3. Rewards. `_calculate_rewards_from_functions(prompts=2 * prompts, ...)` scores each completion independently (L1187–1189). A callable returning `None` gives NaN, which `nansum` skips (L1040–1047). If `missing_eos_penalty` is set, it is subtracted from completions without an EOS token (L1121, L1192–1193).
4. Pairing and loss (verbatim, L1196–1201 and L1221–1226):

```python
first_half, second_half = rewards.split(batch_size)
mask = first_half >= second_half
batch_range = torch.arange(batch_size, device=device)
chosen_indices = batch_range + (~mask * batch_size)
rejected_indices = batch_range + (mask * batch_size)
...
logits = pi_logratios - ref_logratios
if self.args.loss_type == "sigmoid":
    losses = -F.logsigmoid(self.beta * logits)
elif self.args.loss_type == "ipo":
    losses = (logits - 1 / (2 * self.beta)) ** 2
```
- `pi_logratios` = log π_θ(y_c|x) − log π_θ(y_r|x) and `ref_logratios` = the same difference under π_ref. Each log-probability is a sum over non-padding completion tokens (L1209–1219). `beta` = β; for IPO it is the τ of arXiv:2310.12036 (config L51–55). y_c = chosen completion, y_r = rejected completion.
- The loss is the mean over pairs (L1230), followed by one `accelerator.backward` call (L1286).
- The docs' example commands call `examples/scripts/dpo_online.py` (docs L96–101, L137); the file at this commit is `examples/scripts/online_dpo.py` (repository tree). TRL v1.10.0 fixed the docs path (PR #6598).
5. Logged metrics (L1233–1269): `objective/kl` = mean over completions of Σ_t (log π_θ − log π_ref), the k1 form (L1244–1246); `objective/non_score_reward` = −β·KL summed (L1247); `objective/rlhf_reward` = reward + non_score_reward (L1254); `rewards/chosen` and `rewards/rejected` = β·(policy − reference) summed log-prob; `rewards/accuracies` = fraction of pairs with a positive implicit-reward margin (L1259–1267).

## Recipe ledger
Framework defaults, the Qwen2-0.5B quick-start run, and the Pythia 1B/2.8B/6.9B TL;DR runs are in [[trl-online-dpo-recipe]].

## Findings relevant to negative feedback
- The rejected completion is the lower-reward sample of two on-policy samples; its quality depends only on the policy and the reward functions (L1187–1201). The trainer does not filter pairs by reward margin, so pairs with a margin of 0 are trained with the first sample as `chosen` (L1197).
- The only explicit penalty is `missing_eos_penalty`; the docs present it as a way to penalize completions that reach `max_new_tokens` without EOS (docs L72–78).
- Measurement: `objective/kl` and `objective/entropy` sum over all completion positions, and no padding mask is applied at L1244 or L1257. `mean_entropy` is the negative sampled log-probability summed over the sequence, not the entropy of the full token distribution (L1257).
- Known defect at this commit: with `use_vllm=True` and `vllm_mode="server"`, generation used only every second gathered prompt (L641) and kept completions and prompt copies in interleaved order (L663–664, L688), while the loss expects block order (`rewards.split(batch_size)`, L1196). Commit f444f85 (PR #6228, 2026-07-02, released in v1.8.0) generates for all prompts and reorders to block layout; its code comment states that block layout is "what the loss expects". The default `vllm_mode` is "colocate" (config L286–295).

## Connections
- [[dpo]] — the `sigmoid` loss (config L59).
- [[ipo]] — the `ipo` loss (config L60).
- [[self-play-preference]] — Nash-MD; TRL's `NashMDTrainer` subclasses `OnlineDPOTrainer` (nash_md_trainer.py L106).
- [[on-off-policy-rlhf]] — on-policy versus off-policy preference data, the variable this trainer changes relative to offline DPO.
- [[openrlhf-dpo]] — an offline DPO trainer that reads fixed chosen/rejected pairs.
- [[trl-grpo]] — another TRL trainer that generates on-policy samples and scores them with reward functions.
- [[spin]], [[self-rewarding-lm]] — iterative DPO methods with model-generated rejected responses; TRL has no SPIN or self-rewarding trainer at this commit (repository tree).

## Verification
- Checked on 2026-09-14 against: the URL above (commit a08e713, 2026-04-21), plus `online_dpo_config.py`, `docs/source/online_dpo_trainer.md`, `xpo_trainer.py`, `nash_md_trainer.py`, `base_self_distillation_trainer.py`, and the repository tree at the same commit; main commit 0929acf (`training_step` differs only in the tokenizer attribute used for `eos_token_id`); commit f444f85; TRL release notes v0.26.0, v1.1.0, v1.8.0, and v1.10.0.
- Corrections to the previous card version:
  - "Judge is pluggable ... a `Judge` interface that calls a frozen LLM"; "LLM-judge ... exact [OAIF] pattern" → no judge support at this commit (removed in v1.1.0, PR #5485); rewards come only from `reward_funcs`, scored per completion (L1005–1049).
  - "avoids optimizing against a stale π_ref" → the reference model stays frozen (L302–306); what is refreshed each step is the pair data, which is sampled from the current policy (L1116–1119, L1187–1201).
  - "extensions ... live in sibling files (`nash_md`, `xpo`, `self_distillation`)" → `XPOTrainer` and `NashMDTrainer` subclass `OnlineDPOTrainer`; `BaseSelfDistillationTrainer` does not (it derives from `OnlineRolloutMixin`, `SelfDistillationMixin`, `_BaseTrainer`; base_self_distillation_trainer.py L70).
  - Line ranges "≈1089–1275" and "≈585–893" → `training_step` L1089–1288; vLLM paths L585–730, weight sync L732–864. The old excerpt replaced `self.accelerator.gather_for_metrics` with `gather`.
- Removed as unsupported by the source: "Guo 2024 / Dong 2024 closes the gap between DPO and PPO"; "reference implementation used by many self-play / Nash-LM experiments"; "same pattern underlies SPIN, self-rewarding LM training, and XPO"; "equivalent to RLOO with n=2"; "KL tracking is the canary for online DPO collapse: the policy drifts ... much faster than offline DPO"; "missing-EOS penalty prevents truncation gaming"; "no equivalent online trainer in OpenRLHF mainline"; "the DPO loss algebra is identical [to offline DPO]" (offline trainer not checked).
- Not reported by the source: win-rate numbers for the Pythia runs (docs L195 gives only a trend); comparison with offline DPO; any study of tie handling or of reward-margin filtering.
