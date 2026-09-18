<!-- scope: online RLHF theory and method: transfer from imperfect source reward models, justified by the policy-coverage property of KL-regularized objectives (TPO and empirical TPO)
     deps: [[on-off-policy-rlhf]], [[dpo]]
     see-also: [[ipo]], [[generative-reward-models]]
-->

# Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective
- **Core Insight:** Under KL regularization, a policy's coverage coefficient for the optimal policy is bounded above by a function of its value gap (Lemma 3.1), so the authors use high policy value (and, in practice, high win rate) as the criterion for choosing a data-collection policy from imperfect reward models; on XSum with a T5-small (80M) policy, empirical TPO with DPO reaches a 54.0 ± 1.2% win rate against iterative-DPO without transfer after 3 iterations (Table 1).
- **Guideline:** When several imperfect reward models of unknown quality are available for online preference optimization, choose among them per iteration by estimated win rate against the current policy, and keep the current policy as a candidate so data collection returns to ordinary online learning once no source wins, because in the paper's XSum experiments this selection was close to always using the best source (49.1-50.6% win rate against pure T5-Large transfer) and beat always using the worst source (53.1-54.5% against pure ROUGE-Lsum transfer) without knowing source quality in advance (§6, Table 1; App. I.2, Fig. 2).
- **Authors:** Jiawei Huang, Bingcong Li, Christoph Dann, Niao He (ETH Zurich; Google Research)
- **Year:** 2025 (arXiv v1 2025-02; ICML 2025, PMLR 267)
- **URL:** https://arxiv.org/abs/2502.19255
- **Source type:** paper
- **Relevant topics:** online RLHF sample complexity, reward-model transfer, policy coverability, KL-regularized objective, DPO/IPO/XPO, win-rate-based selection

## Abstract
Online RLHF needs many human preference labels. The paper asks how existing but misspecified reward models can reduce that cost. It first shows a property of the KL-regularized RLHF objective: a policy's coverability of the optimal policy is controlled by its sub-optimality. From this it derives transfer principles and a theoretical algorithm, Transfer Policy Optimization (TPO), with provable regret benefits over standard online learning. Because TPO requires several minimax optimizations, the authors also propose an empirical version that selects the transfer policy by win rate, is modular, and can be combined with DPO, IPO, and XPO. Experiments are on summarization.

## Key Contributions
- Lemma 3.1: for policies in the convex hull of the policy class or induced by source rewards in [0, R], coverage of the optimal policy is bounded by 1 + κ(e^{2R/β}) · value gap / β (§3.1).
- Principle 1: select transfer policies by policy value, because under regularization high value implies good coverage (§3.2).
- Principle 2, "self-transfer learning": a policy distilled offline (with RPO) from data collected by a no-regret online algorithm converges to the optimum at Õ(T^{-1/2}) without dependence on |S|, |A|, or Cov∞(Π) (§3.2, Thm. 3.2).
- TPO (Alg. 1) with Transfer Policy Selection (Alg. 2) and a regret bound (Thm. 4.4; Coro. 4.1).
- Empirical TPO (Alg. 3): UCB selection over win rates, compatible with any policy-optimization step (§5).

## Key Figures/Tables to Study
- **Fig. 1:** standard online RLHF versus the transfer setting with candidate policies from imperfect reward models plus the distilled policy.
- **Alg. 3:** the empirical selection loop that practitioners can reuse.
- **Table 1 (DPO) and Table 2 (IPO, XPO):** win rates of empirical TPO against three baselines over 3 iterations.
- **Fig. 2 (App. I.2):** how the transfer budget is allocated across sources per iteration, and the win rates that drive it.

## Technical Details
- **Objective:** J_β(π; r) = E_{s∼ρ, a∼π}[r(s,a)] − β·KL(π‖π_ref), with closed-form optimum π*_r(a|s) ∝ π_ref(a|s)·exp(r(s,a)/β) (§2.1, Eq. 1-2). ρ is the prompt distribution, π_ref the pretrained reference policy, β > 0 the KL coefficient, r ∈ [0, R].
- **Setting:** contextual bandit (prompt = state, response = action); W source rewards r_w of unknown quality; value gap Δ(w) = J_β(π*_{r*}) − J_β(π*_{r_w}) (§2.2). An LLM policy can serve as a source reward through β·log(π/π_0) (§2.2).
- **Coverage coefficient:** Cov_{π̃|π} = E_{s∼ρ, a∼π̃}[π̃(a|s)/π(a|s)] (Def. 2.2). κ(x) = (x−1)²/(x−1−log x) = O(x) (Lemma 3.1).
- **Why regularization matters:** with β = 0 a deterministic sub-optimal policy can have a gap of 2ε and infinite coverage coefficient, so near-optimality does not imply coverage (§3.1 bandit example).
- **TPO structure:** T iterations split into K = T/N blocks; in each block αN steps of an online no-regret algorithm (for example XPO) and (1 − α)N steps with the transfer policy chosen by Alg. 2 (§4, Alg. 1). Alg. 2 uses an optimistic (UCB-style) value estimate for source policies and a pessimistic estimate for the distilled policy (§4.1, Lemmas 4.2-4.3).
- **Regret:** with XPO and α = e^{−R/β}, TPO has Õ(W√T) regret when T is small and Õ(√T) after T is large enough (Coro. 4.1). With W = 0, TPO reduces to standard online RLHF with Õ(√T) regret after finite time (Coro. 4.5).
- **Win rate as a proxy:** P_{r*}(π̃ ≻ π) = E_{s∼ρ, a∼π̃, a′∼π}[P_{r*}(y = 1|s, a, a′)]; for a fixed comparator the lower bound on Cov_{π*|π} grows as π's win rate falls toward 0, so high-win-rate policies are preferred (§5, Lemma 5.1).
- **Empirical TPO:** at each step the collection policy is the candidate (sources plus current online policy) with the highest UCB win-rate estimate against the current policy; the policy is updated once per iteration by Alg_PO (§5, Alg. 3).
- **Experiment setup:** policy T5-small (80M) on XSum; r* simulated by a reward model distilled from Llama-3-8B-Instruct; sources (a) ROUGE-Lsum, (b) BERTScore, (c) T5-base (250M) log-probability, (d) T5-large (770M) log-probability; each source policy approximated by Best-of-N over the online policy's samples (§6; App. I.1). Baselines: (I) iterative-DPO without transfer, (II) pure transfer from the worst source ROUGE-Lsum, (III) pure transfer from the best source T5-Large, where worst and best are ranked by final policy value (§6).
- **Results with DPO (Table 1), win rate (%) of empirical TPO, iterations 1-3:** against (I) 52.1 ± 1.2, 53.3 ± 1.6, 54.0 ± 1.2; against (II) 53.1 ± 1.1, 54.5 ± 1.3, 53.3 ± 1.5; against (III) 49.5 ± 0.9, 49.1 ± 0.4, 50.6 ± 0.3. Result (single study).
- **Results with other optimizers (Table 2, App. I.2):** iteration-3 win rate against no transfer is 55.3 ± 1.1 with IPO and 52.2 ± 1.6 with XPO.
- **Selection over time:** win rates of all sources against the online policy decrease across iterations, and in iteration 3 empirical TPO switches back to online learning (§6; App. I.2, Fig. 2).
- **Name note:** the slug says "coverage loss", but the paper defines no loss term; coverage is an analysis quantity used to justify the selection rule.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Empirical TPO, T5-small policy on XSum | 80M | preference | iterations K; prompts per iteration | 3; 10k | arXiv:2502.19255v3 §6, App. I.1 "Training Details" | verified 2026-09-14 | no ablation reported |
| same | 80M | preference | responses per prompt; pair construction | 8 (N_BoN + 4 generated; top 4 by source reward merged with the other 4); highest and lowest r* reward form the DPO pair | App. I.1 | verified 2026-09-14 | no ablation reported |
| same | 80M | preference | Best-of-N size approximating each source policy | N = 32 | App. I.1 | verified 2026-09-14 | BoN responses outperformed direct T5-base/large responses (App. I.1, no table) |
| same | 80M | preference | peak LR, schedule (DPO and XPO runs); IPO LR | 5e-5 cosine annealing; 1e-5 for IPO | App. I.1; App. I.2 | verified 2026-09-14 | no ablation reported |
| same | 80M | preference | total batch; hardware | 64; 4 H-100 GPUs | App. I.1 | verified 2026-09-14 | no ablation reported |
| same | 80M | preference | UCB constant; win-rate threshold for the online policy | c·sqrt(log 1/δ) = 1.0; 0.55 instead of 0.5 | App. I.1 | verified 2026-09-14 | authors "believe it enhances overall performance"; no ablation reported |
| same | 80M | preference | simulated human reward r* | sfairXC/FsfairX-LLaMA3-RM-v0.1 | App. I.1 | verified 2026-09-14 | not applicable |
| same | 80M | preference | β, epochs per iteration, max response length, sampling temperature | not reported | checked §6, App. I.1-I.2 | not reported | none |
| same | 80M | eval-gate | evaluation prompts; seeds | 10k XSum test prompts; 3 seeds, 95% confidence | App. I.1; Table 1 caption | verified 2026-09-14 | not applicable |

## Findings relevant to generality and negative feedback
- **Negative samples:** among 8 responses per prompt, the lowest-r* response is the DPO rejected response, so negatives enter as gradient (App. I.1).
- **Scope of the evidence:** only T5 policies were fine-tuned, human preferences were simulated by a Llama-3-8B-based reward model, and larger-scale LLMs are left to future work (§7; App. I.1). No held-out-task or cross-domain evaluation is reported.

## Connections
- [[on-off-policy-rlhf]]: studies why online preference data beats offline data; this paper adds reuse of imperfect reward models inside the online loop.
- [[dpo]]: the default Alg_PO; the experiment is a transfer layer on iterative-DPO.
- [[ipo]]: second Alg_PO instantiation (Table 2a).
- [[generative-reward-models]]: another source of imperfect reward signals that could serve as transfer candidates.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2502.19255 (v3, 2025-05-18; v1 2025-02-26)
- Corrections to the previous card version:
  - Authors "Jiajin Zhang, Renshuai Tao, Yuhao Zhang, Zhiqi Shen, Peng Dai, Liyuan Liu, Yali Du, Yan Wang, Han Liu, Weinan Zhang" → "Jiawei Huang, Bingcong Li, Christoph Dann, Niao He" (title page).
  - "reuse instead of retraining preference supervision for each new domain" → reduce human annotations in online RLHF by transferring from imperfect reward models (Abstract, §1).
  - "derives suboptimality bounds" → coverage bound (Lemma 3.1), distilled-policy rate (Thm. 3.2), and regret bounds (Thm. 4.4, Coro. 4.1).
  - "Method comparison table under DPO / IPO with and without coverage-aware initialization" → Tables 1-2 compare empirical TPO with DPO, IPO, and XPO against no transfer, pure ROUGE-Lsum, and pure T5-Large; no initialization variant exists.
  - "source-policy selection ablation" → Fig. 2 reports selection allocation and win rates; it is not an ablation.
- Removed as unsupported by the source: the "Risks + gotchas" list (claims about badly covered high-quality sources, offline coverage metrics, and weakly regularized settings are not stated in the paper); "Why this matters: reward models are expensive to calibrate".
- Not reported by the source: DPO β, epochs per iteration, response length, temperature; fine-tuning of policies larger than T5-small (80M); evaluation on tasks other than XSum summarization.
