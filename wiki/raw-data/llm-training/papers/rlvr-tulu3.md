<!-- scope: the RLVR stage of the Tülu 3 report (§6): objective, verifiable prompt set, PPO settings, ablations
     deps: [[dpo]], [[reward-model-overoptimization]]
     see-also: [[tulu-3]], [[tulu-3-sft-mix]], [[allenai-tulu-blog]], [[deepseek-r1]], [[ifbench]], [[rlvr-beyond-base-model]]
-->

# Tülu 3: Pushing Frontiers in Open Language Model Post-Training
*(This card covers only §6, Reinforcement Learning with Verifiable Rewards. The full report is
[[tulu-3]]; the SFT mixture is [[tulu-3-sft-mix]].)*

- **Core Insight:** Replacing the learned reward model with a binary verifier and training with PPO on 29,946 verifiable prompts raised Tülu 3 8B over its DPO starting point on GSM8K (84.3 → 87.6), MATH (42.0 → 43.7) and IFEval strict (81.1 → 82.4), while the average over 13 benchmarks moved 64.4 → 64.8 (Table 23).
- **Guideline:** When a task has a programmatic correctness check, use a verifier reward with PPO and keep the KL penalty coefficient high enough that KL stays small, because the report observes average scores dropping as KL grows at lower β (§6.2.1, Figure 21). Otherwise keep the reward model.
- **Authors:** Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, et al. (Allen Institute for AI; University of Washington)
- **Year:** 2024 (arXiv v1 2024-11; v5 2025-04-14)
- **URL:** https://arxiv.org/abs/2411.15124
- **Source type:** official technical report
- **Relevant topics:** RLVR, verifiable rewards, PPO, value-model initialization, KL overoptimization, IFEval constraints, GSM8K, MATH

## Summary
Tülu 3 is a family of post-trained models built on Llama 3.1 base models at 8B, 70B and 405B. The
pipeline is SFT → DPO → RLVR. RLVR keeps the KL-constrained RLHF objective but replaces the learned
reward model with a verification function v(x, y) that returns α when the extracted answer or the
satisfied constraint is correct and 0 otherwise (Eq. 7–8). The report releases data, code, reward
models, verifiers and an evaluation toolkit, and separates a development evaluation suite from an
unseen suite.

## Key Contributions
- RLVR objective: max_πθ E_{y∼πθ(x)} [v(x, y) − β·KL(πθ(y|x) ‖ π_ref(y|x))], with v(x,y) = α if correct else 0; α = 10, set from pilot experiments and not tuned further (§6, Eq. 7–8).
- A verifiable prompt mixture of 29,946 prompts: GSM8K train 7,473 (exact match on the extracted answer), MATH train 7,500 (exact match under the `flex` MATH evaluation logic), and 14,973 IF-verifiable prompts built by pairing instructions sampled from the Tülu 2 SFT mix with constraints from the IFEval taxonomy (Table 22).
- Five PPO implementation details carried over from Huang et al. 2024a: value model initialized from a general RM, dropout disabled, multi-epoch training with shuffling, a −10 penalty for responses without an EOS token, and advantage whitening (§6.2).
- Ablations over β, value-model initialization, RM-plus-verifier reward, and starting checkpoint strength (§6.2, §6.2.1).
- Asynchronous PPO infrastructure with dedicated vLLM inference GPUs, scaled to a 405B policy with an 8B value model (§6.3, §8.1).

## Key Figures/Tables to Study
- **Table 21:** PPO hyperparameters for RM-optimization and for RLVR, with the 70B differences noted in the caption.
- **Table 22:** the verifiable prompt mixture and its verifiers.
- **Table 23:** Llama 3.1 Instruct vs Tülu 3 DPO vs Tülu 3 RLVR at 8B and 70B on 13 benchmarks.
- **Figure 19:** verifiable reward, KL, response length and downstream test score for GSM8K, MATH and constraint prompts.
- **Figure 21:** average score vs KL under two value-model initializations — the overoptimization evidence.
- **Figure 22:** verifier-only reward vs verifier plus RM score.
- **Appendix B.4, Figures 28–29:** overoptimized IFEval outputs from higher-KL runs.

## Technical Details
- PPO, not GRPO: γ = 1.0, GAE λ = 0.95, clip ε = 0.2, value coefficient c1 = 0.1, grad-norm clip 1.0, linear LR schedule, generation temperature 1.0, 1 mini-batch (Table 21).
- Ablation runs train roughly 100,000 / 7,473 ≈ 13 epochs over the GSM8K prompt set, shuffled between epochs (§6.2).
- Final checkpoint selection: evaluate every 100 training steps (40 for 70B) and pick the checkpoint with the best combined MATH and IFEval performance (§6.4).
- Some 8B runs reached GSM8K 89.4 and IFEval 84.8 but scored worse on other benchmarks, lowering the average (§6.4).
- At 70B, RLVR improved IFEval (82.6 → 83.2) and MATH (62.3 → 63.0) and left GSM8K unchanged at 93.5, which the report attributes to saturation; the 70B run kept KL below 1 throughout, attributed to the lower learning rate (§6.4, Table 23).
- At 405B, GSM8K and IFEval prompts were dropped and only the MATH train set was used; MATH improved by over 5 points within 25 RLVR steps and training stopped at 75 steps for compute reasons, without observed saturation (§8.1, Figure 25).
- Infrastructure: three models in memory (policy, reference, value); vLLM with PagedAttention on dedicated inference GPUs; asynchronous training that always trains on the second-latest inference batch for reproducibility (§6.3). 405B used 32 nodes / 256 GPUs, 16-way tensor-parallel vLLM inference on 16 GPUs and 240 training GPUs; inference ≈550 s, weight transfer ≈25 s, training ≈1,500 s per iteration (§8.1).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | algorithm | PPO | arXiv:2411.15124v5 §6 | verified 2026-09-18 | no ablation reported (PPO chosen for RLVR in §5.4.1) |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | correct-answer reward α | 10 | §6 Eq. 8 | verified 2026-09-18 | "based on pilot experiments and did not tune it further" (§6) |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | learning rate | 3 × 10⁻⁷ | Table 21 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | effective batch size (prompts) | 224 | Table 21 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | PPO update iterations K | 4 | Table 21 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | max response length | 2,048 tokens (1,024 for GSM8K-only runs) | Table 21 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | total episodes | 100,000 | Table 21 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | KL coefficient β (sweep) | [0.1, 0.05, 0.03, 0.01] | Table 21 | verified 2026-09-18 | Figure 21: lower β → larger KL → lower average score |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | KL coefficient β (final run) | 0.05, warmup ratio ω = 0.0 | Table 21 caption | verified 2026-09-18 | §6.4: 8B final runs also tested β up to 0.15 |
| Llama-3.1-Tulu-3-8B | 8B | RL (RLVR) | value model init | reward model trained on the Tülu 3 8B preference mixture from Tülu 3 SFT | §6.4 | verified 2026-09-18 | Figure 21: general-RM init gives the highest GSM8K and average scores |
| Llama-3.1-Tulu-3-70B | 70B | RL (RLVR) | learning rate | 1 × 10⁻⁷ | Table 21; §6.4 | verified 2026-09-18 | §6.4: higher LR made KL explode in early exploration |
| Llama-3.1-Tulu-3-70B | 70B | RL (RLVR) | effective batch size | 640 | Table 21; §6.4 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-70B | 70B | RL (RLVR) | total episodes | 400,000 | §6.4 | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3-70B | 70B | RL (RLVR) | KL coefficient β (final run) | 0.07 (Table 21 caption) / 0.7 (§6.4 prose) | Table 21 caption vs §6.4 | conflict | the table caption states the released-run value; §6.4 prose gives 0.7 for the same run |
| Llama-3.1-Tulu-3-70B | 70B | RL (RLVR) | warmup ratio ω | 0.07 (Table 21 caption) / 0.1 (§6.4 prose) | Table 21 caption vs §6.4 | conflict | as above |
| Llama-3.1-Tulu-3-405B | 405B | RL (RLVR) | learning rate / batch / episodes / β | 1 × 10⁻⁷ / 1,856 / 300,000 / 0.05, ω = 0.0 | Table 35 (referenced from §8.1) | verified 2026-09-18 | hyperparameter tuning was limited by compute (§8.1) |
| Llama-3.1-Tulu-3-405B | 405B | RL (RLVR) | prompts / steps run | MATH train only; 75 steps | §8.1 | verified 2026-09-18 | §8.1: +5 MATH points by step 25; stopped early for compute |

## Findings relevant to generality
- Overoptimization is reported, not excluded: as β is lowered the policy incurs more KL from the initial model, and higher KL typically gives lower average scores across the evaluation suite (§6.2.1 "Overoptimization Happens", Figure 21). The exception is Figure 22, where the largest KL run has the highest average score.
- Constraint-following overoptimization: outputs from higher-KL RLVR IFEval runs are shown as overoptimized in Appendix B.4 (Figures 28–29).
- Adding reward-model scores on top of verifiable rewards was worse and noisier on GSM8K than verifier-only reward (§6.2.1, Figure 22).
- Starting from a weaker SFT checkpoint reaches the same verifiable reward as starting from DPO but incurs larger KL at the same β, and the stronger starting model usually gives better test performance (§6.2.1, Figure 20).
- The report warns that RLVR trains on data similar to the target evaluations, so overfitting can occur, and checks generalization at intermediate checkpoints (§E.1, Figure 43). The report's evaluation framework separates a development suite from an unseen suite (§7).

## Findings relevant to negative feedback
- The only negative signal is the 0 reward for an unverified answer plus a −10 penalty for a response that does not end with an EOS token within the token budget (§6.2, Table 21). There is no rejected-sample likelihood term at this stage.

## Connections
- [[tulu-3]] is the full report; this card holds only the RLVR stage.
- [[deepseek-r1]] applies rule-based verifier rewards at a larger scale and without a value model.
- [[reward-model-overoptimization]] describes the proxy/gold gap for learned reward models; the Tülu 3 RLVR runs show KL-dependent score decline even without a learned reward model (§6.2.1).
- [[ifbench]] (Pyatkin et al. 2025) reports that models overfit the IFEval constraint set, which bears on the IFEval verifier used here.
- [[prm800k]] and [[math-shepherd]] give per-step signal; RLVR uses only the outcome.
- [[rlvr-beyond-base-model]] measures a related narrowing effect with pass@k rather than KL.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2411.15124 (arXiv v5, 2025-04-14).
- Corrections to the previous card version:
  - "Goodhart's gap … is mechanically zero on verifiable prompts" and "there is no OOD region where the reward spuriously rises" → the report states the opposite for its own runs: "Overoptimization Happens … more KL divergence typically results in lower average scores" (§6.2.1, Figure 21), with overoptimized IFEval outputs in App. B.4.
  - "keeping the no-hacking guarantee on verifiable tasks" → removed; no such guarantee is claimed in the report.
  - "r(x, y) = v(x, y) in {0,1}" → v(x,y) = α if correct else 0, with α = 10 (Eq. 8, §6).
  - "group/batch ~128 prompts, 4 rollouts each, lr ~1e-6, β_KL ~0.04, length 2k–4k" → effective batch 224, PPO update iterations K = 4, LR 3e-7, final 8B β = 0.05, response length 2,048 (Table 21).
  - "*Code:* run model-generated code against unit tests in a sandbox" and "code verifier requires sandbox (isolate-style, per-rollout timeout 5s)" → there is no code RLVR in Tülu 3; the three verifier sources are GSM8K, MATH and IF-verifiable prompts (§6.1, Table 22).
  - "Mixing: RLVR prompts trained alongside RLHF prompts with separate reward pipelines" → the final runs combined the verifiable prompts into one set and used verifiable rewards only; adding RM scores was tested and was worse (§6.4, §6.2.1 Figure 22).
  - "a family of open post-trained Llama-3.1 models (8B and 70B)" → 8B, 70B and 405B were released (Table on the title page, §8.1).
  - "Llama-3.1 8B Instruct GSM8K 84.7, MATH 41.5, IFEval 80.5" → Table 23 gives 83.4, 42.5, 80.6. The Tülu 3 8B RLVR figures 87.6 / 43.7 / 82.4 are correct, but the DPO checkpoint (84.3 / 42.0 / 81.1), not Llama 3.1 Instruct, is the correct comparison for isolating RLVR.
  - "Year: 2024" kept, with the v1 month and the v5 date added.
  - Source type field was missing; added as `official technical report`.
  - Prompt-set size 29,946 and its three-way breakdown added (Table 22); the previous card gave none.
- Removed as unsupported by the source:
  - "advantage estimation via GAE with λ = 0.95, γ = 1.0 (episodic)" — λ and γ are correct (Table 21), but "episodic" was not stated; the descriptor is removed.
  - "if the verifier has loopholes (string-match math graders that accept '42' inside prose), RLVR can hack those loopholes" — the report does not report this failure; only IFEval constraint overoptimization (App. B.4) is documented.
  - "RLVR ablation table (with vs without RLVR stage)" — no such table exists; the with/without comparison is Table 23's DPO vs RLVR columns.
  - "Verifier coverage table — which task suites have verifiers vs which are RLHF-only" — no such table exists; Table 22 lists the prompt sources only.
- Not reported by the source: GPU-hours or wall-clock for the 8B and 70B RLVR runs; pass@k for any RLVR checkpoint; per-domain false-positive rates of the verifiers.
