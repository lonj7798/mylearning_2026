<!-- scope: ICLR 2022 Blog Track post enumerating the 37 implementation details of the openai/baselines ppo2 PPO implementation, with a matched single-file PyTorch reproduction
     deps: [[ppo]]
     see-also: [[hf-rlhf-illustrated]], [[john-schulman-kl-tricks]], [[trl-ppo]]
-->

# The 37 Implementation Details of Proximal Policy Optimization
- **Core Insight:** Reproducing PPO's published results requires matching 37 implementation details of the `openai/baselines` `ppo2` code (commit `ea25b9e`) that the PPO paper does not describe; the post reproduces the official results in classic control, Atari, MuJoCo, LSTM, and MultiDiscrete settings with single-file PyTorch implementations that match those details.
- **Guideline:** When a PPO run does not match a published baseline, first check the run against this enumerated checklist and the post's debugging tests (ratio equals 1 in the first epoch of the first minibatch; `approx_kl` staying below 0.02; ~400 episodic return on Breakout), before changing hyperparameters.
- **Authors:** Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, Weixun Wang
- **Year:** 2022 (published 25 March 2022, ICLR 2022 Blog Track)
- **URL:** https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- **Source type:** practitioner evidence (peer-reviewed blog track, with released code and tracked experiments)
- **Relevant topics:** PPO reproducibility, reference implementation, orthogonal init, advantage normalization, value-loss clipping, LR annealing, GAE

## Summary
The post defines `ppo2` at commit `ea25b9e` as the official PPO implementation, after tracing the revision history of `openai/baselines` from the original `pposgd` code (commit `da99706`, 20 July 2017) through the `ppo1`/`ppo2` split (commit `2dd7d30`, 16 November 2017) to the final commit (`ea25b9e`, 31 January 2020) (§"Background"). It then enumerates 37 implementation details grouped as 13 core details, 9 Atari-specific details, 9 details for continuous action domains, 5 LSTM details, and 1 MultiDiscrete detail, plus 4 auxiliary details that `ppo2` does not use by default (§"Implementation Checklist with References"). Each detail is given with a permanent link to the corresponding line(s) of `openai/baselines` and a pointer to the prior literature that studied it. The post states that it does not perform ablation studies of its own: "Instead of doing ablation studies and making recommendations on which details matter, this blog post takes a step back and focuses on reproductions of PPO's results in all accounts." Where it reports whether a detail helps, it attributes the finding to Engstrom, Ilyas, et al. (2020) or Andrychowicz, et al. (2021). Code is released at `github.com/vwxyzjn/ppo-implementation-details` and tracked runs at `wandb.ai/vwxyzjn/ppo-details`.

## Key Contributions
- Genealogy of `openai/baselines` PPO revisions, fixing `ppo2` (`ea25b9e`) as the reproduction target (§"Background").
- An enumerated checklist of 37 implementation details, each with a permanent code link (§"Implementation Checklist with References").
- Single-file PyTorch reproductions whose reported curves match the official implementation: `ppo.py` (322 lines), `ppo_atari.py` (339 lines), `ppo_continuous_action.py` (331 lines), `ppo_multidiscrete.py` (335 lines) (§§ per-domain sections).
- Four auxiliary details not used by default in `ppo2`: clip-range annealing, parallelized gradient update, early stopping of policy optimization on an approximate-KL threshold, and invalid action masking (§"4 Auxiliary implementation details").
- Debugging and reproducibility recommendations, including seeding everything, checking that the ratio equals 1 on the first update, and enumerating which details a paper's PPO baseline uses (§"Recommendations").

## Key Figures/Tables to Study
- The table of best-reported PPO performance across RL libraries on Atari and MuJoCo (§"Background"), which motivates the reproduction problem.
- The per-domain file-difference views showing how many lines each group of details adds to `ppo.py`.
- The tracked experiment panels for classic control, Atari, MuJoCo, and Gym-μRTS, which are the evidence that the reproduction matches.

## Technical Details
The 13 core details, in the post's order: (1) vectorized architecture; (2) orthogonal initialization of weights and constant initialization of biases; (3) the Adam epsilon parameter; (4) Adam learning-rate annealing; (5) generalized advantage estimation; (6) minibatch updates; (7) normalization of advantages; (8) clipped surrogate objective; (9) value-function loss clipping; (10) overall loss and entropy bonus; (11) global gradient clipping; (12) debug variables; (13) shared and separate MLP networks for policy and value functions.

- Hidden layers use orthogonal initialization with scale `sqrt(2)` and zero biases; the policy output layer uses scale 0.01 and the value output layer scale 1 (core detail 2, `common/policies.py#L49-L63`).
- Adam epsilon is set to 1e-5, against PyTorch's default 1e-8 and TensorFlow's 1e-7; it is neither mentioned in the PPO paper nor configurable in the implementation (core detail 3, `ppo2/model.py#L100`).
- The learning rate decays linearly to 0: from 2.5e-4 for Atari and from 3e-4 for MuJoCo (core detail 4, `ppo2/ppo2.py#L133-L135`). The post attributes the finding that annealing raises episodic return to Engstrom, Ilyas, et al. (2020) and to Andrychowicz, et al. (2021), who report gains in 4 of 5 tasks with relatively small effect size (decision C31, figure 65).
- Advantages are normalized at the minibatch level, not over the whole batch (core detail 7, `ppo2/model.py#L139`). Andrychowicz, et al. (2021) report this does not affect performance much (decision C67, figure 35).
- Value-function loss clipping takes the maximum of the unclipped and clipped squared errors around the previous value estimate (core detail 9, `ppo2/model.py#L68-L75`). The post implements it for fidelity while noting Engstrom, Ilyas, et al. (2020) find no evidence it helps and Andrychowicz, et al. (2021) suggest it hurts (decision C13, figure 43).
- The overall loss is `policy_loss - entropy * entropy_coefficient + value_loss * value_coefficient`, with policy and value parameters sharing one optimizer (core detail 10, `ppo2/model.py#L91`).
- Gradients of policy and value networks are rescaled so the global L2 norm does not exceed 0.5 (core detail 11, `ppo2/model.py#L102-L108`).
- `approxkl` is logged as `(-logratio).mean()`, the k1 estimator; the post notes the lower-variance unbiased alternative `((ratio - 1) - logratio).mean()` (core detail 12, `ppo2/model.py#L115-L116`).
- Observation normalization (running mean and variance) and observation clipping to [-10, 10] are two separate details, and both belong to the 9 continuous-action details, not to the core 13 (`common/vec_env/vec_normalize.py#L4`, `#L39`). Reward scaling divides rewards by the standard deviation of a rolling discounted sum, and reward clipping to [-10, 10] is again a separate detail (`#L28`, `#L32`).
- In classic control the post reports that the separate-networks architecture outperforms the default shared-network architecture, and it adopts separate networks in the video tutorial (core detail 13).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| `openai/baselines` `ppo2` (`ea25b9e`), Atari | n/a | RL | rollout length `nsteps`; minibatches; GAE λ; γ; epochs; entropy coef; LR; clip ε | 128; 4; 0.95; 0.99; 4; 0.01; 2.5e-4 annealed to 0; 0.1 | blog §"9 Atari-specific implementation details", quoting `baselines/ppo2/defaults.py` | verified 2026-09-18 | no ablation reported; values quoted from the official defaults |
| `openai/baselines` `ppo2` (`ea25b9e`), MuJoCo | n/a | RL | rollout length `nsteps`; minibatches; GAE λ; γ; epochs; entropy coef; LR; clip ε; value network | 2048; 32; 0.95; 0.99; 10; 0.0; 3e-4 annealed to 0; 0.2; `copy` (separate networks) | blog §"9 details for continuous action domains", quoting `baselines/ppo2/defaults.py` | verified 2026-09-18 | no ablation reported; values quoted from the official defaults |
| Blog reproduction, Atari | n/a | RL | number of environments N | 8 | blog §"9 Atari-specific implementation details" | verified 2026-09-18 | matched to the PPO paper's "number of actors, 8" rather than to `common/cmd_util.py#L167`, which sets N to the CPU count |
| Blog reproduction, MuJoCo | n/a | RL | number of environments N | 1 | blog §"9 details for continuous action domains" (`common/cmd_util.py#L167`) | verified 2026-09-18 | no ablation reported |
| Blog reproduction, all domains | n/a | RL | early stopping on target KL | `--target-kl 0.01`, off by default | blog §"4 Auxiliary implementation details" | verified 2026-09-18 | attributed to Dossa et al., who suggest early stopping as an alternative to tuning the number of update epochs |

## Findings relevant to generality
- The post's position on asynchronous PPO is that the evidence is insufficient: it reports that RLlib's APPO documentation carries no benchmark information, and that no APPO implementation it knows of covers Atari, MuJoCo/PyBullet, MultiDiscrete action spaces, and LSTM at the same time (§"Is asynchronous PPO better?"). Its alternative recommendation is to make the vectorized environments faster.
- The post argues that modular RL library design disperses implementation details across files, and recommends single-file implementations for algorithmic research, at the stated cost of duplicated and harder-to-refactor code (§"Does modularity help RL libraries?").

## Connections
- [[ppo]] — the paper whose implementation this post enumerates.
- [[hf-rlhf-illustrated]] — covers the RLHF loop at a higher level of abstraction.
- [[trl-ppo]], [[openrlhf-ppo]] — LLM PPO implementations whose defaults can be compared row by row against the ledger above.
- [[john-schulman-kl-tricks]] — the source of the k1/k3 KL estimators referenced in core detail 12.
- Follow-up by an overlapping author group: "The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization", Shengyi Huang, Michael Noukhovitch, Arian Hosseini, Kashif Rasul, Weixun Wang, Lewis Tunstall, arXiv:2403.17031 (v1, 24 March 2024), which enumerates over 20 RLHF-specific implementation details and reproduces the TL;DR summarization RLHF scaling behaviour on Pythia models.

## Verification
- Checked on 2026-09-18 against: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ (published 25 March 2022).
- Corrections to the previous card version:
  - "Quantitative ablation on the Atari + MuJoCo stacks where omission of each trick is measurable" → the post states it does not do ablation studies and focuses on reproduction; ablation findings it cites are attributed to Engstrom, Ilyas, et al. (2020) and Andrychowicz, et al. (2021) (§"Implementation Checklist with References").
  - "Code-location table: mapping each trick to OpenAI Baselines line number" → the post gives a permanent code link inline with each detail; there is no such table.
  - "Observation normalization ... clipped to [-10, 10]" listed as a core item → observation normalization and observation clipping are two separate details, both in the 9 continuous-action details.
  - "PPO clip epsilon 0.2 is the most common default" → `cliprange` is 0.2 in the MuJoCo defaults and 0.1 in the Atari defaults.
  - "Global gradient clipping at max-norm 0.5 (RL) or 1.0 (RLHF typical)" → the post gives 0.5; the RLHF value is not in this source.
  - "Author:" (one field, five names) → "Authors:", and a "Source type" field was added, as required by the card standard.
  - Follow-up attributed to "Huang, Liu, von Werra — HF" → the arXiv:2403.17031 author list is Shengyi Huang, Michael Noukhovitch, Arian Hosseini, Kashif Rasul, Weixun Wang, Lewis Tunstall.
- Removed as unsupported by the source: "In LLM RLHF the analog is reward whitening"; "Separate policy/value clip ranges — not always enabled by default" (not one of the 37 details); "Ablation bar chart: per-trick performance delta"; "Annotated pseudocode block: PPO loop with each of the 37 details labeled inline"; the four "RLHF-specific follow-up highlights" bullets (reward whitening, per-token vs per-sequence log-prob, KL added to the per-token reward, value-head initialization from the reward model), which describe arXiv:2403.17031 and not this post.
- Not reported by the source: per-detail performance deltas measured by the post itself; any LLM or RLHF hyperparameters.
