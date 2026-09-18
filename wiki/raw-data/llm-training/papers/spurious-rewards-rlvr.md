<!-- scope: RLVR gains under weak and spurious rewards are model-family dependent; GRPO clipping bias and pre-existing code reasoning explain the Qwen2.5-Math case
     deps: [[grpo]], [[reward-model-overoptimization]]
     see-also: [[echo-chamber-rl-post-training]], [[rlvr-beyond-base-model]], [[prorl]], [[reinforcement-learning-with-one-training-example]]
-->

# Spurious Rewards: Rethinking Training Signals in RLVR
- **Core Insight:** On Qwen2.5-Math-7B, GRPO with randomly assigned rewards gains 21.4 absolute points on MATH-500 against 29.1 points from ground-truth rewards, but the same spurious rewards give minimal or negative gains on Qwen2.5, OLMo2, and Llama3 models, so the effect is a property of the base model's pretraining prior rather than of RLVR (Abstract, §2.2, §3).
- **Guideline:** When reporting an RLVR method, evaluate it on models from more than one family and include random and format rewards as baselines, because every weak or spurious reward the authors tested improved Qwen2.5 while failing to help at least one non-Qwen model (§3, §7). Do not use spurious rewards as a training strategy; the authors state they are introduced only for controlled analysis (§2.2 note, §7).
- **Authors:** Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, et al. (University of Washington; Allen Institute for AI; UC Berkeley)
- **Year:** 2025 (arXiv v1 2025-06; text checked is arXiv v2, 2026-02-25)
- **URL:** https://arxiv.org/abs/2506.10947
- **Source type:** paper
- **Relevant topics:** RLVR, reward quality, GRPO clipping bias, pretraining priors, model-family generality, code reasoning

## Abstract
RLVR can elicit strong mathematical reasoning in certain language models even with spurious rewards that have little, no, or negative correlation with the correct answer. Training Qwen2.5-Math-7B with GRPO and randomly assigned rewards improves MATH-500 by 21.4 absolute points, near the 29.1 points gained with ground-truth rewards. The authors explain the effect through a clipping bias in GRPO: the clip term amplifies behaviors that already have high probability under the base model even when the reward carries no information. As a case study they identify code reasoning in Qwen2.5-Math models, meaning reasoning written as Python without any code execution, whose frequency rises from 65% to over 90% under spurious rewards. The presence of such amplifiable behaviors is model-dependent, and rewards that work for Qwen often fail on Llama3 or OLMo2. The authors conclude that RL methods must be validated across diverse models.

## Key Contributions
- A reward ladder from standard to spurious — ground truth, majority vote over 64 samples, format (non-empty `\boxed{}`), random with fixed probability γ, and majority-voted incorrect labels — run under identical training settings (§2.2).
- The cross-family result: spurious rewards help Qwen2.5 models and are flat or harmful on OLMo2 and Llama3, across eight models (§3, Figure 3).
- A mechanistic account: disabling clipping in three separate ways removes the random-reward gain, so the gain depends on the clip term (§4, Figure 4).
- The code-reasoning case study, including a causal intervention that induces or suppresses the behavior by prompt and by reward (§5).

## Key Figures/Tables to Study
- **Figure 1**: MATH-500 accuracy after 300 RLVR steps for each reward on Qwen2.5-Math-7B, Qwen2.5-7B, Llama3.1-8B-Instruct, and OLMo2-7B.
- **Figure 3**: per-model MATH-500 curves for eight models, the central generality evidence.
- **Figure 4**: the four clipping ablations; random rewards improve performance only in the panel with clipping enabled.
- **Table 1**: pre-RL code frequency and accuracy split by code versus natural language.
- **Table 2**: MATH-500 after forcing responses to begin with "Let's solve this using Python."

## Technical Details

### Reward definitions (§2.2)
1. **Ground truth** — verifiable match against the labeled answer.
2. **Majority vote** — pseudo-labels from the majority of 64 sampled responses per prompt, taken before RLVR.
3. **Format** — reward any response containing at least one non-empty `\boxed{}`, ignoring correctness. Qwen2.5-Math's system prompt already specifies `\boxed{}`.
4. **Random** — reward assigned with fixed probability γ. Values tested: γ ∈ {0.7, 0.5, 0.3, 0.001, 0}; only γ = 0 produces no learning (App. A, Figure 8).
5. **Incorrect** — majority-vote labels restricted to the subset that is verifiably wrong, rewarding responses matching those wrong answers. Designed to separate "labels are probably correct" from "labels are high-probability model outputs".

### Main empirical results
- Qwen2.5-Math-7B, MATH-500 absolute gain: ground truth +29.1, incorrect labels +24.1, random +21.4 (§2.2).
- Qwen2.5-Math-7B, AMC: format +13.8, incorrect +24.1, random +21.4, against roughly +27 to +29 from majority-vote and ground-truth labels (§2.2).
- AIME2024: format +10.3, incorrect +10.2, random +10.2, ground truth +15.3. On AIME2025, written after every tested model's knowledge cutoff, ground truth has a clear advantage and other rewards give −0.4 to +4.5 (§2.2, App. D).
- Most gains appear within the first 50 training steps; Qwen2.5-Math-1.5B is the exception, gaining from random rewards only after 100 steps and only 4.9 points on AMC (§2.2).

### Model-family dependence (§3)
- Eight models tested: Qwen2.5-Math-7B, Qwen2.5-Math-1.5B, Qwen2.5-7B, Qwen2.5-1.5B, OLMo2-7B, OLMo2-7B-SFT, Llama3.1-8B (and -Instruct), Llama3.2-3B (and -Instruct).
- Across Qwen2.5, all non-random rewards including incorrect labels improve MATH-500; OLMo2 stays flat under spurious rewards and gains mainly from ground truth. Each weak or spurious reward fails to help at least one non-Qwen model and can be flat or harmful.
- Smaller models benefit less from random rewards; the authors conjecture larger models preserve more of the pretraining prior that these rewards amplify (§3). Labeled by the authors as a conjecture, not a measured mechanism.
- Models that were already RL post-trained see minimal gains under nearly all rewards (App. J).

### Clipping mechanism (§4)
- The GRPO objective's clip term is the candidate cause. Without clipping, the expected training objective under random rewards is zero.
- Three no-clipping variants: disable clipping in the implementation; raise the mini-batch size to match the rollout size; reduce the rollout batch size. The latter two enforce π_θ = π_old during a rollout, so ρ_t = 1 and clipping has no effect. Random rewards give no consistent improvement in any of the three; the gain returns when clipping is enabled (Figure 4).
- The stated bias: with clip threshold ε_c = 0.2, a token at π_old = 0.9 cannot reach the upper clip bound 1.08 because probabilities are at most 1, so it receives non-negative gradient bias; a token at π_old = 0.02 is clipped above 0.024, so small increases trigger suppression. Clipping therefore reinforces high-probability tokens and suppresses low-probability ones (§4, worked example; App. B).
- KL regularization is disabled in these runs (§4, footnote).

### Code reasoning (§5)
- Definition: responses containing the string "python", reasoning in code form with no code interpreter available (§5.1, §5.2, Figure 5).
- Pre-RL frequency and split accuracy on MATH-500 (Table 1): Qwen2.5-Math-7B 65.0% code frequency, 60.9% accuracy with code versus 35.0% without; Qwen2.5-Math-1.5B 53.6%, 52.6% versus 17.2%; Qwen2.5-7B 92.2%, 39.9% versus 61.5%; OLMo2-7B-SFT 98.0%, 21.0% versus 40.0%. Qwen2.5-1.5B, OLMo2-7B, Llama3.1-8B-Instruct, and Llama3.2-3B-Instruct never generate code.
- Under weak or spurious rewards, code frequency rises to about 90% within 15 steps and up to 95.6% for random rewards, tracking accuracy. Under ground-truth rewards code frequency rises then declines as natural-language reasoning improves, which the authors read as a different improvement mechanism (§5.2, Figure 6).
- Intervention by prompt (Table 2): forcing the first sentence to be "Let's solve this using Python." changes MATH-500 by +24.2 for Qwen2.5-Math-1.5B, +15.0 for Qwen2.5-Math-7B, +10.0 for Qwen2.5-1.5B, −19.4 for Qwen2.5-7B, −28.6 for Llama3.2-3B-Instruct, −21.6 for Llama3.1-8B-Instruct, −1.2 for OLMo2-7B, −2.8 for OLMo2-7B-SFT.
- Intervention by reward: a "Python reward" paying only for the string "python" pushes Qwen2.5-Math-7B above 99% code reasoning within 20 steps and improves performance mainly for Qwen2.5-Math (§5.3, Figure 7).
- The authors state code reasoning is not a complete explanation; lexical repetition is another behavior with the same pattern (App. G).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B and seven others | 1.5B–8B | RL | Algorithm | GRPO | arXiv:2506.10947v2 §4, App. A.3 | verified (2026-09-18) | no ablation reported; clipping is ablated in §4 |
| same | — | RL | Training data | DeepScaleR | arXiv:2506.10947v2 §2.1 | verified (2026-09-18) | no ablation reported |
| same | — | RL | Learning rate | constant 5e-7 | arXiv:2506.10947v2 App. A.3 | verified (2026-09-18) | no ablation reported |
| same | — | RL | Mini batch (rollouts before a gradient update) | 128 | arXiv:2506.10947v2 App. A.3 | verified (2026-09-18) | varied in the §4 clipping ablation (raised to match rollout size) |
| same | — | RL | Rollout batch (prompts rolled out at once) | 64 | arXiv:2506.10947v2 App. A.3 | verified (2026-09-18) | reduced in one §4 clipping ablation |
| same | — | RL | Samples per prompt | 16 rollouts for the GRPO advantage | arXiv:2506.10947v2 App. A.3 | verified (2026-09-18) | reduced to 8 in one §4 ablation panel |
| same | — | RL | Sampling temperature | τ = 1 | arXiv:2506.10947v2 App. A.3 | verified (2026-09-18) | no ablation reported |
| same | — | RL | KL and entropy loss | not applied | arXiv:2506.10947v2 App. A.3, §4 footnote | verified (2026-09-18) | no ablation reported |
| same | — | RL | Clip threshold used in the worked example | ε_c = 0.2 | arXiv:2506.10947v2 §4 | verified (2026-09-18) | §4 no-clipping ablations |
| same | — | RL | Training length | 300 steps (Figure 1); curves reported to 200 steps in §5 | arXiv:2506.10947v2 Figure 1, Figure 6 | verified (2026-09-18) | most gains within the first 50 steps (§2.2) |
| same | — | eval-gate | Decoding | temperature 0.0 for pass@1, 0.6 for pass@k | arXiv:2506.10947v2 App. A.4 | verified (2026-09-18) | follows the Zuo et al. 2025 codebase defaults |
| same | — | RL | Compute | ~24 hours on 8 A100s per run | arXiv:2506.10947v2 App. A.5 | verified (2026-09-18) | no ablation reported |

## Findings relevant to generality
- **The central generality caveat:** rewards validated on one model family do not transfer. Spurious rewards that lift Qwen2.5 give minimal improvement or degradation on Qwen2.5 base, OLMo2, and Llama3 (Abstract, §1, §3). Reading the random-reward result as a general property of RLVR contradicts the paper.
- **Benchmark-date evidence:** the gap between ground-truth and spurious rewards is clearest on AIME2025, whose questions postdate every tested model's knowledge cutoff (§2.2). The paper reports this observation; it does not test contamination directly and does not claim contamination as the cause.
- **What the gains are:** the authors position their result as evidence for the existing hypothesis that RLVR at open-source post-training compute scales triggers latent behaviors rather than teaching new ones, citing prior work (§2.2, §6).
- **Ground truth works differently:** under ground-truth rewards code frequency rises and then falls while accuracy keeps rising, which the paper reads as improvement not reducible to amplifying one prior behavior (§5.2).
- **Negative feedback:** the clipping analysis is about the sign and size of the gradient bias per token, not about negative samples. Clipping is reported to give non-negative gradient bias to already-likely tokens and suppressive bias to rarely-sampled tokens, echoing Yu et al. 2025 on clipping reducing exploration (§4).
- **Long context, agentic training, distillation:** not addressed by this source.

## Connections
- [[grpo]] — the algorithm whose clip term is the analyzed mechanism.
- [[echo-chamber-rl-post-training]] — related claim that RL post-training amplifies existing structure.
- [[rlvr-beyond-base-model]] — pass@k evidence that RLVR gains can overstate capability expansion.
- [[reinforcement-learning-with-one-training-example]] — one-shot RLVR, cited in §2 as a comparison point in the reward ladder.
- [[prorl]] — argues longer and more diverse RL can widen the reasoning frontier; a counterweight to the amplification reading.
- [[reward-model-overoptimization]] — separate failure mode: optimizing a learned proxy reward, rather than gaining from an uninformative one.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2506.10947 (arXiv v2, 2026-02-25)
- Corrections to the previous card version:
  - Core Insight framed the random-reward gain as a property of RLVR. Corrected to state the model-family dependence the abstract makes central: the effect holds for Qwen2.5-Math and fails on Qwen2.5, OLMo2, and Llama3 (Abstract, §3).
  - "code reasoning ... roughly 65% to above 90% frequency" was unattributed to a model. The 65.0% baseline is Qwen2.5-Math-7B specifically (Table 1), and the post-training level is ~90% within 15 steps, up to 95.6% under random rewards (§5.2).
  - "Identifies code reasoning as a concrete pretrained behavior that gets amplified" kept, with the paper's own qualification added: code reasoning is stated not to be a complete explanation, and lexical repetition behaves similarly (§5.3, App. G).
  - Author list truncated to the first six plus "et al." per the card standard, with all three affiliations named.
  - Year line now records that the checked text is arXiv v2 (2026-02-25) while v1 is 2025-06.
  - The card had no Source type field and no Recipe ledger despite the paper disclosing a full RL configuration; both added from App. A.3–A.5.
- Removed as unsupported by the source:
  - "Attributes this to GRPO clipping bias, not to meaningful signal extraction from the reward" as a flat claim — the paper presents clipping as a hypothesis supported by three ablations and states that the model-dependence, not clipping alone, determines the outcome (§3, §4).
  - "Treat apparent RLVR gains with caution; verify whether your reward is actually teaching the model or merely triggering distributional sharpening through clipping bias" — replaced with the paper's own stated recommendation, which is to test across model families and to include format and random rewards as baselines (§7).
- Not reported by the source: a direct contamination experiment on Qwen2.5 (the AIME2025 result is observational); results at model scales above 8B; pass@k curves for the spurious-reward runs; any leakage-free benchmark constructed by these authors.
