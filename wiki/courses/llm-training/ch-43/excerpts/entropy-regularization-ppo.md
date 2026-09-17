---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/entropy-regularization-ppo.md
source_url: https://arxiv.org/abs/1602.01783
primary_version: arXiv:1602.01783v2 (A3C; ICML 2016)
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the card verified on 2026-09-14)"
---

# Excerpt: the entropy term in A3C (Mnih et al. 2016), and where it came from

- The slug names the entropy-regularization idea; the artifact is the A3C paper, "Asynchronous Methods for
  Deep Reinforcement Learning".
- Policy gradient as printed in §4: `∇_θ' log π(a_t|s_t; θ')·(R_t − V(s_t; θ_v)) + β·∇_θ' H(π(s_t; θ'))`. The
  paper gives a gradient, not a loss function, and Algorithm S3 omits the entropy term although §4 and
  Supp. §8 say it was used.
- Stated purpose (§4): "adding the entropy of the policy π to the objective function improved exploration by
  discouraging premature convergence to suboptimal deterministic policies". The idea is credited to Williams &
  Peng (1991), which places entropy bonuses 25 years before Soft Actor-Critic.
- Coefficients: `β = 0.01` for all Atari and TORCS experiments (Supp. §8); `10⁻⁴` on the differential entropy
  of the Gaussian policy in the MuJoCo experiments (Supp. §9). The paper reports no ablation of `β`.
- PPO (arXiv:1707.06347v2) adds "an entropy bonus to ensure sufficient exploration, as suggested in past work
  [Wil92; Mni+16]" with coefficient `c_2 = 0.01` in its Atari runs (Table 5) and no entropy bonus in its
  MuJoCo comparison (§6.1).

## Removed in the 2026-09 revision
"The bonus is the small-α limit of SAC", "preserved in every LLM-RL framework", and the claim that
Andrychowicz et al. rank the entropy coefficient as a second-tier knob were in the previous version and are
not in the sources. [[entropy-collapse-ppo]] reports no significant gain from any tested regularizer except on
HalfCheetah; the LLM-scale evidence is in [[entropy-mechanism-llm-rl]] §4.1 and [[high-entropy-minority-tokens]].
