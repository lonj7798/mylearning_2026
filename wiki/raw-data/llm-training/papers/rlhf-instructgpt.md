<!-- scope: InstructGPT — the three-stage RLHF recipe (SFT → RM → PPO / PPO-ptx) and its measured alignment tax
     deps: [[ppo]], [[bradley-terry-rm]]
     see-also: [[dpo]], [[rlaif-scaling]], [[constitutional-ai]], [[llama-2]]
-->

# Training language models to follow instructions with human feedback
- **Core Insight:** Supervised fine-tuning on labeler demonstrations, a 6B Bradley-Terry reward model trained on labeler rankings, and PPO against that reward model with a per-token KL penalty to the SFT policy produce a 1.3B model whose outputs labelers prefer to those of 175B GPT-3 (Abstract; Figure 1).
- **Guideline:** When aligning a model to a real user prompt distribution, train the reward model on K = 4–9 ranked completions per prompt and treat all C(K,2) comparisons from one prompt as a single batch element, because shuffling them into one dataset caused the reward model to overfit within a single pass (§3.5, Eq. 1 and footnote 5). When PPO regresses on public NLP datasets, mix pretraining gradients into the PPO update (PPO-ptx), because this reduced the regressions on all datasets tested while the model still lagged GPT-3 on DROP, SQuADv2, and translation (§4.2, Figure 29).
- **Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, et al. (OpenAI)
- **Year:** 2022 (arXiv v1 2022-03; NeurIPS 2022)
- **URL:** https://arxiv.org/abs/2203.02155
- **Source type:** paper
- **Relevant topics:** RLHF pipeline, reward modeling, PPO-ptx, alignment tax, instruction following

## Abstract
Scaling a language model does not by itself make it better at following a user's intent. The authors fine-tune GPT-3 on labeler-written demonstrations, collect labeler rankings of model outputs, train a reward model on those rankings, and then fine-tune the supervised policy against the reward model with PPO. In human evaluations on the OpenAI API prompt distribution, outputs from the 1.3B InstructGPT model are preferred to outputs from 175B GPT-3 despite 100x fewer parameters. The models improve on truthfulness and reduce toxic generation while showing minimal performance regressions on public NLP datasets.

## Key Contributions
- The three-stage pipeline (SFT → reward model → PPO) applied to a real API prompt distribution rather than a single task (§3.5, Figure 2).
- **PPO-ptx**: pretraining gradients accumulated together with PPO gradients to reduce regressions on public NLP datasets (§3.5, Eq. 2; App. C.4).
- Per-token KL penalty from the SFT model folded into the reward, following Stiennon et al. (2020) (§3.5; App. C.4).
- A comparison against 175B GPT-3 fine-tuned on FLAN and on T0++, showing that public-task instruction collections do not transfer to the API prompt distribution (§1, §4.1, App. C.5).
- Documentation of the prompt use-case taxonomy (Table 1) and the full labeler instructions (App. B.2).

## Key Figures/Tables to Study
- **Figure 1:** Win rate against the 175B SFT model by model size; 1.3B PPO-ptx is preferred to 175B GPT-3.
- **Figure 2:** Diagram of the three steps of the method.
- **Figure 3:** Win rates against 175B SFT split by prompt source (GPT vs InstructGPT API prompts) and by held-out vs training labelers.
- **Figure 5:** Likert comparison of PPO-ptx against FLAN and T0 on the InstructGPT prompt distribution.
- **Figure 7 and Table 14 (App. D):** RealToxicityPrompts human and Perspective API evaluations, 1,729 prompts, three 175B models, with and without a "respectful" instruction.
- **Figure 29:** Few-shot public-NLP-dataset results showing the regressions and their mitigation by PPO-ptx.
- **Equations 1 and 2:** reward-model loss and the PPO-ptx objective.

## Technical Details

### Data
- SFT dataset: about 13k training prompts; RM dataset: about 33k training prompts; PPO dataset: about 31k training prompts, all API prompts with no human labels (§3.2; Table 6 gives the exact splits: 11,295 + 1,430 SFT train; 6,623 + 26,584 RM train; 31,144 PPO train).
- Dataset is over 96% English (§3.3).
- Inter-annotator agreement: training labelers agree 72.6 ± 1.5% of the time; held-out labelers 77.3 ± 1.3% (§3.4).

### Stage 1 — Supervised fine-tuning
- 16 epochs, cosine LR decay to 10% of the initial value, no warmup, residual dropout 0.2 (§3.5; App. C.1).
- SFT models overfit on validation loss after 1 epoch, but more epochs still improve RM score and human preference ratings; final checkpoint selected by RM score on validation, not by validation loss (§3.5; App. C.1).

### Stage 2 — Reward model (6B)
- A single 6B RM is used for all PPO policy sizes; 175B RM training was unstable and more expensive as a value-function initialization (App. C.2).
- Each prompt is shown with K ∈ {4,…,9} completions to rank, giving up to C(K,2) comparisons per prompt (§3.5).
- Loss (Eq. 1): `loss(θ) = − (1 / C(K,2)) · E_{(x,y_w,y_l)~D}[ log σ( r_θ(x,y_w) − r_θ(x,y_l) ) ]`, where `r_θ(x,y)` is the scalar RM output, `y_w` the preferred completion, `y_l` the dispreferred one, and `D` the comparison dataset.
- All C(K,2) comparisons from one prompt are treated as a single batch element; this requires one forward pass per completion instead of C(K,2) passes and removes the single-epoch overfitting (§3.5, footnote 5).
- The RM is normalized by a bias so that labeler demonstrations score a mean of 0 before RL (§3.5).

### Stage 3 — PPO and PPO-ptx (Eq. 2)
`objective(φ) = E_{(x,y)~D_{π_φ^RL}}[ r_θ(x,y) − β log( π_φ^RL(y|x) / π^SFT(y|x) ) ] + γ · E_{x~D_pretrain}[ log π_φ^RL(x) ]`
- `r_θ` is the reward-model score; `β` is the KL reward coefficient; `π^SFT` is the supervised policy used both as initialization and as the KL reference; `γ` is the pretraining loss coefficient, set to 0 for the "PPO" models (§3.5).
- The environment is a bandit: one prompt, one response, one reward, episode ends (§3.5).
- The value function is initialized from the reward model (§3.5).
- Pretraining and PPO gradients are computed in consecutive steps and accumulated into the same buffer; 8 times more pretraining examples than RL episodes are used (App. C.4).

### Results relevant to alignment
- 175B InstructGPT outputs are preferred to 175B GPT-3 outputs 85 ± 3% of the time and to few-shot GPT-3 71 ± 4% of the time (§4.1).
- Against the FLAN and T0++ baselines (175B GPT-3 fine-tuned on about 1M examples each), InstructGPT has a 73.4 ± 2% win rate against the SFT baseline, compared with 29.8 ± 2% for FLAN and 26.8 ± 2% for T0 (§1, §4.1).
- InstructGPT generates about 25% fewer toxic outputs than GPT-3 when prompted to be respectful, and does not improve over GPT-3 on Winogender or CrowS-Pairs (§1, §4.2).

## Recipe ledger
All rows verified 2026-09-18 against arXiv:2203.02155v1. Sizes are the policy sizes; the reward model is 6B for every policy size (App. C.2).

| Model | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| InstructGPT SFT baseline | 1.3B, 6B | SFT | peak LR / batch (prompts) / epochs | 9.65e-6 / 32 / 16 | App. C.1 | verified | geometric search over 7 LRs; epochs tuned by geometric search, selected on RM score |
| InstructGPT SFT baseline | 175B | SFT | peak LR / batch (prompts) / epochs | 5.03e-6 / 8 / 16 | App. C.1 | verified | geometric search over 5 LRs |
| InstructGPT RM | 6B | reward-model | LR / batch (prompts) / epochs | 9e-6 / 64 / 1 | App. C.2 | verified | ±50% LR changes gave similar performance; multiple epochs overfit |
| InstructGPT RLHF init | 1.3B / 6B / 175B | SFT | peak LR (2 epochs, 10% pretraining mix) | 5e-6 / 1.04e-5 / 2.45e-6 | App. C.3 | verified | log-linear sweep of 5 LRs (1.3B, 6B) and 3 LRs (175B) |
| InstructGPT PPO / PPO-ptx | all | RL | episodes / unique prompts | 256k / about 31k | App. C.4 | verified | no ablation reported |
| InstructGPT PPO / PPO-ptx | all | RL | batch / minibatch / inner epochs | 512 / 64 / 1 | App. C.4 | verified | no ablation reported |
| InstructGPT PPO / PPO-ptx | all | RL | KL coefficient β | 0.02 | App. C.4 (Eq. 2) | verified | no ablation reported at this locus |
| InstructGPT PPO-ptx | all | RL | pretraining coefficient γ | 27.8 | App. C.4 (Eq. 2) | verified | Figure 33: a value of the mix coefficient reverses SQuADv2 and DROP regressions with minimal validation-reward loss |
| InstructGPT PPO / PPO-ptx | all | RL | clip ratio / sampling temperature / GAE discount / EMA decay | 0.2 / 1.0 / no discount / 0.992 | App. C.4 | verified | no ablation reported |
| InstructGPT PPO / PPO-ptx | all | RL | policy peak LR | not reported | App. C.4, E.9 | not reported | E.9 sweeps 2.55e-6 to 2.55e-5; runs above 8.05e-6 diverged without the pretraining mix; final checkpoints picked by Likert score |
| InstructGPT | all | RL | context / max prompt / max response | 2k / 1k / 1k tokens | App. C (preamble) | verified | no ablation reported |

## Findings relevant to generality
- **Public NLP task collections do not equal generality on real prompts.** FLAN and T0++ fine-tunes of 175B GPT-3 perform slightly worse than the SFT baseline on the API prompt distribution (§1, §4.1, Figure 5). The authors give two reasons: public NLP datasets capture task types (classification, QA) that are a small share of API usage, and they are hard to fit to the open-ended generation that API users submit (§4.1).
- **Alignment tax, measured.** RLHF regressed against GPT-3 on SQuADv2, DROP, HellaSwag, and WMT 2015 French-to-English (§1, §4.2). PPO-ptx mitigates the regressions on all datasets tested and surpasses GPT-3 on HellaSwag, but still lags GPT-3 on DROP, SQuADv2, and translation (§4.2, Figure 29). Increasing the KL coefficient instead of mixing pretraining gradients decreases validation reward and never fully recovers DROP and SQuAD (§4.2, Figures 33 and 34).
- **Generalization to other labelers.** Held-out labelers, who produced no training data, prefer InstructGPT to GPT-3 at about the same rate as training labelers (§1, Figure 3). The authors call this preliminary.
- **Generalization outside the fine-tuning distribution.** Qualitatively, the 175B PPO-ptx model follows instructions for summarizing and answering questions about code and sometimes follows non-English instructions, despite the fine-tuning data being over 96% English (§4.3, Figure 8). The authors label this a qualitative probe.
- **Remaining failures.** The model still makes simple mistakes, such as accepting false premises in a question (§4.3, Figure 9).

## Connections
- RL optimizer: [[ppo]].
- Reward-model formulation: [[bradley-terry-rm]].
- Direct-optimization alternative: [[dpo]].
- Safety-focused successors: [[constitutional-ai]], [[rlaif-scaling]].
- Implementation walkthroughs: [[hf-rlhf-illustrated]], [[costa-huang-ppo-details]].
- Later industrial variants: [[llama-2]].
- Critic-free replacements that keep the same KL-shaped reward: [[rloo]].

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2203.02155 (arXiv v1, 2022-03-04).
- Corrections to the previous card version:
  - "prevents alignment tax" → PPO-ptx reduces the regressions and the model still lags GPT-3 on DROP, SQuADv2, and translation (§4.2, Figure 29).
  - "PPO LR 1.41e-5 (fixed)" → not reported by the paper; E.9 reports a sweep from 2.55e-6 to 2.55e-5 and divergence above 8.05e-6 without the pretraining mix.
  - "Epochs per rollout: 4" → a single inner epoch per batch, batch 512 split into 8 minibatches of 64 (App. C.4).
  - "PPO rollout length ≤ 2048 tokens" → context length 2k, prompts longer than 1k filtered out, max response length 1k (App. C preamble).
  - "SFT LR 9.65e-6" listed as a single value → 9.65e-6 for 1.3B and 6B, 5.03e-6 for 175B (App. C.1); the RLHF initialization models are a separate run at 2 epochs with LRs 5e-6 / 1.04e-5 / 2.45e-6 (App. C.3).
  - "KL coef β 0.02 (adaptive controller optional)" → β = 0.02, constant; no adaptive KL controller is described.
  - "Table 6: RealToxicityPrompts results" → Table 6 is dataset sizes (App. A.3); the toxicity results are Figure 7 and Table 14 (App. D).
  - "Figure 3: Labeler win-rate curves — 1.3B PPO > 175B base" → that result is Figure 1; Figure 3 splits win rates by prompt source and by held-out vs training labelers.
  - "Train all pairs from the same prompt in the same minibatch" → all C(K,2) comparisons from one prompt are one batch element (§3.5).
  - "Released the prompt taxonomy and labeler guidelines" → the paper documents the use-case taxonomy (Table 1) and labeler instructions (App. B.2); no release is claimed.
- Removed as unsupported by the source: the "Entropy handling" paragraph (no entropy bonus, β annealing, or entropy-collapse tracking is discussed in the paper); the claim that the KL term "serves the same regularizing role" as an entropy bonus.
- Not reported by the source: PPO policy learning rate for the released checkpoints; wall-clock or GPU-hour cost (only petaflop/s-days: 4.9 for SFT 175B and 60 for 175B PPO-ptx, §5.1); optimizer betas beyond Adam β1 = 0.9, β2 = 0.95 (App. C preamble).
