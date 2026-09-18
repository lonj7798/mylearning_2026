<!-- scope: former synthesis card on prompts-per-batch vs rollouts-per-prompt in LLM RL; no single primary artifact exists
     deps: [[ppo]], [[grpo]]
     see-also: [[rloo]], [[deepseekmath]], [[verl-grpo]], [[trl-grpo]], [[async-rollout]]
-->

# Minibatch Sharing Across Prompts in LLM RL — Framework Synthesis
- **Source type:** none (synthesis card without a primary artifact)
- **Status:** no verifiable primary source. Chapters must not cite this card; cite the works listed below instead.

> **No verifiable primary source.** The previous card described itself as "synthesized" from four works and named no
> paper, report, or config that defines "minibatch sharing". Searched: the card's own URLs (arXiv:2402.14740 v2,
> arXiv:2402.03300, verl-project/verl@753aed3, huggingface/trl@a04ffd3) and the library card [[nathan-lambert-grpo]].
> None of them defines the term or contains the ablations the card attributed to them. A web search for the term could not be run
> in this session (search budget exhausted).

## Verifiable pointers to related works
Each item below was read at the stated locus on 2026-09-14. Each item is a fact about that work only.

- **RLOO estimator.** Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from
  Human Feedback in LLMs", arXiv:2402.14740 (v2, 2024-02). For k samples per prompt, each sample's baseline is the
  mean reward of the other k − 1 samples (§2.3, equation after Eq. 8). The paper runs RLOO and RAFT with
  k = 2 and k = 4 only (§5.1–5.2, Fig. 3, Table 1). In Table 1, RLOO (k=4) has win-rate 77.9 on TL;DR, 43.7 on HH (Pythia), and
  64.1 on HH (Llama). RLOO (k=2) has 74.2, 47.6, and 62.2 on the same three columns. Card: [[rloo]].
- **GRPO group size in DeepSeekMath-RL 7B.** Shao et al., arXiv:2402.03300. "For each question, we sample 64
  outputs. The max length is set to 1024, and the training batch size is 1024"; policy LR 1e-6, KL coefficient
  0.04, one policy update per exploration stage (§4.2). Outcome advantage is
  `Â_{i,t} = (r_i − mean(r)) / std(r)` over the group of G outputs (§4.1.2). The paper reports no ablation over
  the number of outputs per question in §4. Card: [[deepseekmath]].
- **verl default.** verl-project/verl@753aed3 `verl/trainer/config/rollout/rollout.yaml` L126–127:
  `n: 1` with the comment "number of responses (i.e. num sample times). > 1 for grpo".
  `verl/trainer/ppo/core_algos.py` L268–331 (`compute_grpo_outcome_advantage`): a group with one response gets
  mean 0 and std 1; otherwise advantage = (score − group mean) / (group std + epsilon), epsilon = 1e-6. Card: [[verl-grpo]].
- **TRL default.** huggingface/trl@a04ffd3 `trl/trainer/grpo_config.py` L479–480: `num_generations` defaults to 8;
  the effective batch size must be evenly divisible by it (help string, L482–483). Card: [[trl-grpo]].

Derived from the two advantage definitions above (not a claim from any single source): if every response to a prompt
receives the same reward, then `r_i − mean(r) = 0` for every response, so that prompt's advantages are all zero under
both the GRPO and RLOO baselines.

## Connections
- [[rloo]], [[deepseekmath]], [[grpo]] — primary sources for the group-baseline estimators.
- [[verl-grpo]], [[trl-grpo]] — framework cards for how samples per prompt are configured.
- [[async-rollout]], [[replay-buffer-rlhf]] — cards that link here. They should cite the works above instead.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2402.14740 (v2); https://arxiv.org/abs/2402.03300 (PDF);
  github.com/verl-project/verl@753aed3; github.com/huggingface/trl@a04ffd3.
- Corrections to the previous card version:
  - "RLOO Figure 2: advantage estimator variance vs n — the 1/(n−1) curve" → Figure 2 of arXiv:2402.14740 plots test
    reward during training for RLOO, RAFT, PPO, REINFORCE with baseline, and vanilla PG. It does not plot variance
    against k (§5.1, Fig. 2).
  - "RLOO §4 ablation … n=8 vs n=4" → the paper tests k ∈ {2, 4} only. §4 is the experimental setup, not an ablation.
  - "verl `rollout.n=8`" → the default is `n: 1` (rollout.yaml L127 @753aed3).
  - "DeepSeekMath Figure 7 (appendix): GRPO pass@1 vs n — knee at n=16 for MATH, keeps rising for AIME" → Figure 7
    is in §5.2.2, not the appendix. It plots Maj@K and Pass@K of DeepSeekMath-Instruct 7B and -RL 7B on GSM8K and
    MATH (temperature 0.7) against the number of evaluation candidates K, not the training group size. The caption
    states "RL enhances Maj@K but not Pass@K". No AIME result appears in it.
  - "Sources: Ahmadian et al. 2024 RLOO (§4 hyperparameter ablations)" → no such ablation exists in §4.
- Removed as unsupported by any listed source:
  - "DeepSeekMath ablations in appendix show gains plateau past n=16 … 16 is unstable on harder OOD problems".
  - "Interconnects GRPO notes (Lambert 2024): empirical sweep n ∈ {4, 8, 16} — 8 wins on most tasks".
  - "8 rollouts per prompt is the sweet spot; dropping below 4 makes group baselines noisy"; "numerically stable at n ≥ 4".
  - "moving from n=2 to n=4 halves variance".
  - "OpenRLHF PPO `n_samples_per_prompt=4`" (not checked against an OpenRLHF commit; removed).
  - "Typical config: B=128, n=8, effective batch 1024, micro-batch 4–16, PPO epochs 1 or 2".
  - "same-prompt rollouts share the prompt prefix forward pass in vLLM — effectively free beyond the first rollout".
  - "higher n with fixed B·n increases GPU utilization".
  - "RLOO … lower variance than GRPO's σ-normalized form under small-n".
  - "Length-matched packing: OpenRLHF and verl pack same-prompt rollouts together".
  - The link to [[reinforce-plus-plus]] "learned group statistics" (that card was not the source of this claim).
