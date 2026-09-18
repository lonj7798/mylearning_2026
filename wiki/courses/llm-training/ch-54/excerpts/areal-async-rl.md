<!-- chapter excerpt for ch-54. Primary-source extract, read 2026-09-17.
     If a library card of this slug exists under wiki/raw-data/llm-training/, prefer the card. -->

# AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning
- **Artifact:** Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, et al. (IIIS Tsinghua; Ant Group; HKUST). arXiv:2505.24298 (v1 2025-05; text read from v5, 2026-03-02). NeurIPS 2025. Source type: paper. Code: github.com/inclusionAI/AReaL.
- **Core Insight:** Fully decoupling generation from training and bounding data staleness with a rate limit plus a decoupled PPO objective gives up to 2.77× lower end-to-end training time than synchronous RL at the same GPU count, with matched or higher final accuracy (Abstract; §7.2, Table 1).
- **Guideline:** When rollout workers run ahead of the trainer, cap staleness with an explicit maximum-staleness parameter and regularize the update against a recent proximal policy rather than the behavior policy, because with naive PPO the AIME24 score of a 1.5B model falls from 42.0 (η = 0) to 23.3 at η = 4, while the decoupled objective holds it at 42.2 (§7.4, Table 2).

## Technical details (with loci)
- **Staleness control (§5.1, Eq. 3).** With current policy version `i`, total generated trajectories `N_r`, and training batch size `B`, new generation requests are admitted only while `⌊(N_r − 1)/B⌋ ≤ i + η`. `η` is the maximum permitted staleness in training steps; `η = 0` degenerates to synchronous RL. Older trajectories in the data buffer are prioritized when forming a batch. The paper notes that a small `η` slows generation when a few very long trajectories are in flight, and therefore suggests a large `η` for throughput, which in turn motivates the algorithmic fix below (§5.1).
- **Decoupled PPO objective (§5.2, Eqs. 4-5).** The behavior policy `π_behav` (which sampled the trajectory) is separated from a proximal policy `π_prox` (the trust-region center). The objective is
  `J(θ) = E_{q∼D, a_t∼π_behav} Σ_t (π_prox/π_behav) · min( u_t^prox(θ)·Â_t , clip(u_t^prox(θ), 1−ε, 1+ε)·Â_t )`
  with `u_t^prox(θ) = π_θ(a_t|s_t) / π_prox(a_t|s_t)`, `Â_t` the advantage and `ε` the clip range. `π_prox` is set to the parameters before each model update, and Eq. 5 is implemented by recomputing token probabilities when the global batch arrives (§5.2, "Practical Remark").
- **Why it matters for interrupted generation (§5.2, Prop. 1).** A sequence whose tokens were produced by several policy versions `(π_θ, …, π_{θ+k})` is equivalent to sampling entirely from one behavior policy `π_behav`, so mixed-version trajectories stay algorithmically valid (proof in §D).
- **Default staleness (§7.1).** η = 4 for coding, η = 8 for math, on R1-Distilled-Qwen models from 1.5B to 32B; three quarters of devices are allocated to inference.
- **Staleness ablation (§7.4, Table 2; 1.5B, math).** Scores without / with the decoupled objective — AIME24: η=0 oracle 42.0; η=1 41.8 / 42.1; η=2 40.0 / 41.8; η=4 23.3 / 42.2; η=8 35.7 / 41.0; η=16 35.8 / 38.7; η=∞ 34.0 / 36.9. MATH-500 at η=4: 66.9 / 89.5. Figure 5c reports effective training throughput rising from 128.7k tokens/s at η=0 to 396.8k at η=∞.
- **System ablations (§7.5, Fig. 6).** Dynamic micro-batch allocation: about 30% average throughput gain across 1B/7B/32B. Interruptible generation: 12% (1.5B) and 17% (7B) generation-throughput gain on 4 nodes.
- **End-to-end (§7.2, Table 1).** 1.5B math, 16 nodes, 250 PPO steps: 42.2 AIME24 in 14.8 training hours vs 42.0 in 41.0 hours for synchronous AReaL and 43.1 in 33.6 hours for verl. 7B: 63.1 in 25.4 h vs 63.0 in 57.7 h synchronous. 14B code, LiveCodeBench: 58.1 in 21.9 h vs 56.7 in 48.8 h.
- **Hyperparameters (§B.1, Table 3).** 512 prompts per batch, 16 answers per prompt, temperature 1.0, top-p 1.0, clip ε 0.2, 4 PPO minibatches, Adam LR 2×10⁻⁵, constant schedule, max generation length 27,648 tokens, reward +5 / −5 at the final token, no critic and no reference model.
- **Implementation (§6).** SGLang v0.4.6 for generation, Megatron-Core v0.11.0 for training, SLURM scheduling, reward computation and data transfer pipelined off the GPU path, padding-free sequence packing with dynamic micro-batch allocation (Algorithm 1).

## Not reported
Effect of η on out-of-domain benchmarks, pass@k, or entropy; per-domain staleness tuning during training (named as future work, §8).
