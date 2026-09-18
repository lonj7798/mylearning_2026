<!-- chapter excerpt for ch-54. Primary-source extract, read 2026-09-17.
     If a library card of this slug exists under wiki/raw-data/llm-training/, prefer the card. -->

# Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models
- **Artifact:** Michael Noukhovitch, Shengyi Huang, Sophie Xhonneux, Arian Hosseini, Rishabh Agarwal, Aaron Courville. arXiv:2410.18252 (v1 2024-10); ICLR 2025. Source type: paper. Code: github.com/mnoukhov/async_rlhf.
- **Core Insight:** Separating generation and training onto different GPUs makes RLHF faster but requires learning on data from earlier policy versions; RLHF performance degrades as off-policyness grows, with a logarithmic dropoff such that one and two steps of staleness behave almost identically (§3.2-§3.3, Figs. 3-4).
- **Guideline:** When moving an RLHF run from synchronous to asynchronous, keep the number of minibatch updates per generated batch small (N = 1 or 2 changed the win rate little in the TL;DR study) and prefer a loss measured to be robust to staleness, because in the same study PPO and RLOO lost most of their advantage off-policy while Online DPO was the only method that still learned at N = 64 (§3.3, Fig. 4).

## Technical details (with loci)
- **Off-policyness knob (§3.2).** The on-policy setup generates one minibatch, labels it with the reward model, and updates. The study instead generates `N` minibatches and performs `N` minibatch updates per iteration; the first update is on-policy and each later one is further from the generating policy. `N` is varied from 1 to 64.
- **Setup (§3.1).** TL;DR summarization in the controlled gold-model protocol of Gao et al. and Tang et al.; Pythia 410M SFT policy and reward model; minibatch 512; 256 steps (about 130,000 episodes). Success is the gold-model win rate against human reference summaries; drift is measured as the SFT model's perplexity on the policy's summaries, used as a KL proxy.
- **PPO under staleness (§3.2, Fig. 3).** Win rate is highest at N = 1 and falls as N grows. All values of N lie on the same win-rate-vs-KL Pareto curve: staleness slowed progress along the frontier rather than changing the frontier.
- **Loss comparison (§3.3, Fig. 4).** PPO is best at N = 1 but degrades quickly; RLOO likewise; Online DPO reaches a higher win rate at lower KL at N = 4 and is the only method to learn at N = 64. A Best-of-2 SFT baseline is included to control for Online DPO sampling two completions per prompt.
- **Scale (§4, Fig. 1, Table 1).** On 4×A100, a 2.8B Pythia trains about 25% faster asynchronously at equal performance, and the speedup grows with model size. For LLaMA 3.1 8B on No Robots with one GPU reserved for vLLM generation and seven for training: SFT 31.80% win rate; synchronous Online DPO 57.20% at compute time 230; asynchronous Online DPO 57.20% at compute time 142 (38% faster), with average response lengths 198.40 / 286.21 / 290.55.
- **Further optimizations (§1, §5).** Sampling K = 4 completions per prompt in training-bound settings reaches the same final win rate in half the steps, at the cost of more KL drift; the most aggressive compute optimizations reach nearly the same performance about 250% faster at 2.8B but are described as a trade-off with a performance cost.
- **Reasoning extension (Abstract).** Rho 1B on GSM8K trains about 70% faster asynchronously while matching synchronous accuracy.

## Not reported
No measurement on knowledge or instruction-following benchmarks outside the target task; no MoE or long-horizon agentic setting.
