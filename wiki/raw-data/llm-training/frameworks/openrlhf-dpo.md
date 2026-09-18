<!-- scope: OpenRLHF offline DPO code at a pinned commit: DPOLoss (DPO, cDPO label smoothing, IPO), DPOTrainer step with optional NLL and MoE aux terms, train_dpo.py defaults
     deps: [[dpo]], [[ipo]]
     see-also: [[openrlhf-dpo-recipe]], [[trl-online-dpo]], [[hf-dpo-zoo]], [[openrlhf-ppo]], [[rpo]], [[llama-3]]
-->

# OpenRLHF DPO implementation: `DPOLoss`, `DPOTrainer`, `train_dpo.py` (commit 64c1cc4)
- **Core Insight:** OpenRLHF computes DPO on summed response log-probabilities of the policy and a frozen reference, offers cDPO label smoothing and IPO as options, and can add an NLL term on chosen responses; the CLI defaults are β = 0.1, label smoothing 0, and NLL coefficient 0 (train_dpo.py L237, L242, L246).
- **Guideline:** When reproducing an OpenRLHF DPO run, pass β, epochs, and gradient clipping explicitly, because the `DPOTrainer` class defaults (β = 0.01, max_norm = 0.5, 2 epochs; dpo_trainer.py L42–44) differ from the CLI defaults that `train_dpo.py` passes in (β = 0.1, max_norm = 1.0, 1 epoch; L236–237, L275).
- **Authors:** OpenRLHF project contributors (GitHub organization OpenRLHF); pinned commit authored by xiaoxigua999
- **Year:** 2026 (commit 64c1cc4, 2026-04-19); the NLL term was added in commit ef0bd5b (2024-07-24)
- **URL:** https://github.com/OpenRLHF/OpenRLHF/tree/64c1cc4f30d4c0dab772c8aa1dc11288c425e0da (files `openrlhf/models/loss.py`, `openrlhf/trainer/dpo_trainer.py`, `openrlhf/cli/train_dpo.py`, `examples/scripts/train_dpo_llama.sh`)
- **Source type:** released config/code
- **Relevant topics:** DPO, cDPO label smoothing, IPO, NLL regularization, reference model, MoE auxiliary loss, preference-training defaults

## Summary
At commit 64c1cc4, OpenRLHF implements offline DPO in three files. `DPOLoss` (loss.py L246–281) receives summed log-probabilities of the chosen and rejected responses under the policy and the reference model, and returns the loss and the implicit rewards of both responses. `DPOTrainer.fit` (dpo_trainer.py L106–220) runs the policy with gradients and the reference model under `torch.no_grad()` on one concatenated chosen+rejected batch, adds the optional MoE auxiliary and NLL terms, and logs accuracy and both implicit rewards. `train_dpo.py` sets the CLI defaults and uses the policy path for the reference model when no reference path is given (L342–343). At current main b117b2b (2026-09-14), `DPOLoss` and the train step are unchanged except for line numbers (`DPOLoss` at loss.py L329–364), and evaluation metrics are weighted by sample count (commit 7c899eb, #1336).

## Key Contributions
- One loss module covers three objectives: sigmoid DPO (`label_smoothing = 0`), cDPO (`label_smoothing > 0`), and IPO (`ipo = True`) (loss.py L251–275).
- Policy and reference log-probabilities come from one forward pass each over the concatenated chosen+rejected batch; the docstring states the reason as avoiding two forward passes, "because it's faster for FSDP" (dpo_trainer.py L305–309).
- Optional NLL term on chosen responses and optional MoE auxiliary loss, each enabled when its coefficient exceeds 1e-8 (dpo_trainer.py L66–70, L169–180).
- Separate logging of `loss`, `acc`, `chosen_reward`, `reject_reward`, and `nll_loss` (dpo_trainer.py L184–197).
- Optional CPU parameter offload for the reference model through `--ref.offload` (train_dpo.py L48–52, L213; deepspeed_utils.py L78–99).

## Key Figures/Tables to Study
- `DPOLoss.forward` (loss.py L257–281): the three loss formulas and the implicit-reward definitions.
- Train step in `DPOTrainer.fit` (dpo_trainer.py L157–197): loss composition and logged metrics.
- `concatenated_forward`, `concatenated_inputs`, `_get_batch_logps` (dpo_trainer.py L305–388): padding, prompt masking, and summed versus mean log-probabilities.
- `examples/scripts/train_dpo_llama.sh` (L1–37): the example 8B configuration; values are in [[openrlhf-dpo-recipe]].

## Technical Details
**Loss.** Let h = (log π_θ(y_c|x) − log π_θ(y_r|x)) − (log π_ref(y_c|x) − log π_ref(y_r|x)) (loss.py L264–266).
- DPO and cDPO: ℓ = −(1 − ε)·log σ(β·h) − ε·log σ(−β·h). With ε = 0 this is the original DPO loss; the code comment cites DPO Eq. 7 and Eq. 3 of the cDPO note (L271–275).
- IPO: ℓ = (h − 1/(2β))²; the code comment cites Eq. 17 of arXiv:2310.12036v2 (L268–269).
- The batch loss is the mean of ℓ over pairs (L277).

Symbols: π_θ is the trained policy; π_ref is the reference model; y_c and y_r are the chosen and rejected responses; log π(y|x) is the sum of response-token log-probabilities with prompt tokens masked (dpo_trainer.py L380–386); β is `model.beta`; ε is `model.label_smoothing`; σ is the logistic function.

```python
# openrlhf/models/loss.py @64c1cc4, L264-281 (verbatim)
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

        return loss, chosen_rewards, rejected_rewards
```

**Trainer step** (dpo_trainer.py L157–184). The total loss is `preference_loss + aux_loss * aux_loss_coef + nll_loss * nll_loss_coef` (L176–180). `acc` is the fraction of pairs with `chosen_reward > reject_reward` (L184). The reference model is set to `eval()` (L147) and its forward pass runs under `torch.no_grad()` (L160–163). Evaluation computes only the preference loss and accuracy; the NLL and auxiliary terms are not included (L273–285).

**NLL term.** `concatenated_forward` returns `-all_logps_mean[: chosen_ids.shape[0]].mean()` (L328): the negative per-token mean log-probability of each chosen response, averaged over the batch. The preference loss uses summed log-probabilities (L322–326). The CLI help for `--model.nll_loss_coef` reads "Regularization with NLL loss, see LLama 3.1 tech report." (train_dpo.py L246).

**Batching.** Chosen and rejected sequences are right-padded to a common length and concatenated along the batch dimension; `prompt_id_lens` is repeated twice (L350–360). Prompt tokens are masked by position before the one-token shift (L380–384).

**MoE auxiliary loss.** The coefficient `--model.aux_loss_coef` has help text "MoE balancing loss" and default 0 (train_dpo.py L244); the trainer comment reads `# Mixtral 8*7b` (dpo_trainer.py L66).

## Recipe ledger
Framework defaults and the example-script values are in [[openrlhf-dpo-recipe]] (21 rows, status verified 2026-09-14). The source reports no ablation for any default.

## Findings relevant to negative feedback
The source is code. It contains no measurement of training dynamics. It shows the following about negative signals (terms from the course standard §6.1):
- The rejected response is used as gradient (type 4): h contains −log π_θ(y_r|x), and with ε = 0 minimizing ℓ lowers the rejected log-probability (loss.py L264–275).
- Derived from the formula at L272–275 (not stated in the source): ∂ℓ/∂log π_θ(y_r|x) = β(1 − ε)σ(−βh) − βεσ(βh). A positive value means a gradient step lowers log π_θ(y_r|x). With ε = 0 the value is always positive. With ε > 0 it becomes negative once e^{βh} > (1 − ε)/ε; for β = 0.1 and ε = 0.1 this happens at h > 10·ln 9 ≈ 22.0.
- Derived from L269 (not stated in the source): for IPO, ∂ℓ/∂log π_θ(y_r|x) = 2(1/(2β) − h), so a gradient step lowers the rejected log-probability only while h < 1/(2β) (5.0 at β = 0.1) and raises it when h > 1/(2β).
- The positive anchor available in the code is the NLL term on chosen responses (L170–180, L328).
- The logs separate `chosen_reward` and `reject_reward` (L191–192), which allows a reader to see whether both implicit rewards fall together. The logs do not include raw chosen or rejected log-probabilities.

## Connections
- [[dpo]] — the paper whose Eq. 7 the loss comment cites.
- [[ipo]] — the paper whose Eq. 17 the IPO branch cites.
- [[llama-3]] — the report the `nll_loss_coef` help text cites.
- [[rpo]] — a separate paper that also adds an NLL term on chosen responses to DPO.
- [[trl-online-dpo]], [[hf-dpo-zoo]] — TRL preference-optimization implementations.
- [[openrlhf-ppo]] — the PPO loss in the same `loss.py` file.

## Verification
- Checked on 2026-09-14 against: https://github.com/OpenRLHF/OpenRLHF at commit 64c1cc4f30d4c0dab772c8aa1dc11288c425e0da (last commit on or before 2026-04-21, the previous card's fetch date); DPO files re-checked at main b117b2b5 (2026-09-14).
- Corrections to the previous card version:
  - "`loss.py` ≈ lines 231–257 (`DPOLoss`)" → L246–281 at 64c1cc4; L329–364 at b117b2b.
  - "optional NLL on the chosen (Pang et al. 2024 RPO-style)" → the code does not cite RPO; the CLI help cites the Llama 3.1 tech report (train_dpo.py L246).
  - "`concatenated_forward` ... cuts activation memory roughly in half" → the docstring gives speed under FSDP as the reason (dpo_trainer.py L306–308); the code makes no memory claim.
  - "NLL ... an extra cross-entropy term on the chosen response" → the term is the negative per-token mean log-probability of each chosen response, averaged over the batch (L328).
  - "`dpo_trainer.py`, ~lines 150–185 (training step body)" → the train step is at L157–184; the old excerpt matched the code in content but condensed its formatting, so it is replaced by exact loci and a verbatim `DPOLoss` excerpt.
  - "Reference model ... supports DeepSpeed ZeRO-3 with the ref offloaded to CPU" → `--ref.offload` sets CPU parameter offload in the reference model's eval config, which uses ZeRO stage 3 when training uses stage 3 and stage 0 otherwise (deepspeed.py L483–495).
- Removed as unsupported by the source: "one of the most-cited reference implementations"; "useful when ~10% of pairs are mislabeled"; "IPO ... preserves diversity"; "RPO/SimPO-Mix recipe to combat DPO's typical chosen-logp degradation"; "MoE aux loss ... necessary to keep router balance during DPO of Mixtral/DeepSeek-MoE"; "four production tricks" framing; the TRL `loss_type` list (belongs to a different artifact, see [[hf-dpo-zoo]]).
- Not reported by the source: any ablation of β, label smoothing, IPO, or the NLL coefficient; any statement about likelihood displacement.
