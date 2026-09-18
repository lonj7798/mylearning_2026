<!-- scope: recipe ledger split out of [[hh-rlhf]] (arXiv:2204.05862) — preference-model and RLHF training settings
     deps: [[hh-rlhf]]
     see-also: [[rlhf-instructgpt]], [[constitutional-ai]]
-->

# Recipe ledger — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)

Split out of [[hh-rlhf]] to keep that card under the line limit. All rows are from arXiv:2204.05862 v1.
The paper reports learning rates as multipliers on the language-model pretraining learning rate, and never
states the pretraining learning rate itself, so the absolute values cannot be recovered.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Anthropic HHH context-distilled LM | 52B | distill-SFT | batch size | 32 sequences | arXiv:2204.05862 v1 App. A | verified 2026-09-18 | no ablation reported |
| Anthropic HHH context-distilled LM | 52B | distill-SFT | learning rate | 0.05 × pretraining LR, linearly decayed to zero | App. A | verified 2026-09-18 | no ablation reported |
| Anthropic preference model | 13M–52B scan | reward-model (preference-model pretraining) | learning rate | 0.1 × pretraining LR | App. A | verified 2026-09-18 | no ablation reported |
| Anthropic preference model | 13M–52B scan | reward-model (human-feedback finetuning) | learning rate | 0.01 × pretraining LR | App. A | verified 2026-09-18 | no ablation reported |
| Anthropic preference model | 13M–52B scan | reward-model | epochs | 1 | §3.2 (Figure 7 caption: "we train for one epoch") | verified 2026-09-18 | the single epoch is what makes the learning curve double as a dataset-size scaling curve |
| Anthropic RLHF policy | up to 52B | RL | algorithm | PPO with a KL penalty in the reward, r_total = r_PM − λ_KL·D_KL(policy ‖ policy₀) | §4.1 Eq. 4.1 | verified 2026-09-18 | no ablation reported |
| Anthropic RLHF policy | up to 52B | RL | KL coefficient λ_KL | 0.001 | §4.1; App. B.1 | verified 2026-09-18 | chosen from "a variety of hyperparameter scans" (App. B.1); the scan itself is not shown |
| Anthropic RLHF policy | up to 52B | RL | learning rate | 0.01 × pretraining LR | App. B.1 | verified 2026-09-18 | same scan (App. B.1) |
| Anthropic RLHF policy | up to 52B | RL | PPO clip ε | 0.2 | App. B.1 | verified 2026-09-18 | same scan (App. B.1) |
| Anthropic RLHF policy | up to 52B | RL | discount γ | 1 | App. B.1 | verified 2026-09-18 | no ablation reported |
| Anthropic RLHF policy | up to 52B | RL | entropy bonus | none | App. B.1 | verified 2026-09-18 | no ablation reported |
| Anthropic RLHF policy (main RLHF scan) | up to 52B | RL | PPO inner iterations K | 1 | App. B.1 | verified 2026-09-18 | App. B.1 states higher K typically gave more stable results |
| Anthropic RLHF policy (robustness study, §4.2) | scan of sizes | RL | PPO inner iterations K | 2 | App. B.1 | verified 2026-09-18 | same |
| Anthropic RLHF policy (online RLHF, §4.5) | up to 52B | RL | PPO inner iterations K | 4 | App. B.1 | verified 2026-09-18 | same |
| Anthropic RLHF policy (robustness study, §4.2) | scan of sizes | RL | max response length | 32 tokens | App. B.1 | verified 2026-09-18 | no ablation reported |
| Anthropic RLHF policy (all other runs) | up to 52B | RL | max response length | 128 tokens | App. B.1 | verified 2026-09-18 | no ablation reported |
| Anthropic RLHF policy (online RLHF, §4.5) | up to 52B | RL | LR schedule | halved every 100,000 samples | App. B.1 | verified 2026-09-18 | no ablation reported |
| Anthropic RLHF policy (robustness study, §4.2) | scan of sizes | RL | warmup | linear over the first 25,000 samples | App. B.1 | verified 2026-09-18 | no ablation reported |
| Anthropic RLHF policy | up to 52B | RL | prompt source | training split of the PM comparison dataset with the responses removed; one response generated per prompt | App. B.1 | verified 2026-09-18 | the authors note multi-turn policy training would need a model of the human side |
| Anthropic RLHF policy | up to 52B | RL | robustness horizon | train and test PM scores stay together to about 150k training samples, then diverge | §4.2, Figure 4 | verified 2026-09-18 | measured by splitting the static dataset 50:50 and training separate train/test PMs |
| Anthropic RLHF policy | up to 52B | RL | global batch, total samples, optimizer, GPU count | not reported (checked §4.1, App. A, App. B.1, and the paper's figures) | — | not reported | — |
| Rejection-sampling data-collection model | 52B | data collection | samples per prompt k | most often 16, against a 52B preference model | §2.3 | verified 2026-09-18 | no ablation reported |

## Notes on units
- "Samples" in the robustness and online-RLHF rows means RL training samples (policy responses scored by a
  preference model), as used in §4.2 and App. B.1, not comparisons in the preference dataset.
- The preference-model size scan and the policy size scan are separate; the robustness study in Figure 4
  (right) trains and tests all policy sizes against 52B preference models (§4.2).

## Connections
- [[hh-rlhf]] — the card this ledger belongs to.
- [[rlhf-instructgpt]] — the contemporaneous RLHF recipe the paper contrasts with in §1.3.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2204.05862 (arXiv v1, 2022-04-12).
- Corrections to the previous card version: none (new file, split out of [[hh-rlhf]]).
- Removed as unsupported by the source: none.
- Not reported by the source: absolute learning rates, global batch size, total RL steps or samples,
  hardware, and wall-clock or GPU-hour cost for any stage.
