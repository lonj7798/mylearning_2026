<!-- scope: Williams (1992) REINFORCE — the statistical gradient-following update and the reinforcement baseline
     deps: []
     see-also: [[ppo]], [[trpo]], [[rloo]], [[grpo]]
-->

# Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning
- **Core Insight:** For any update of the form `Δw_ij = α_ij (r − b_ij) ∂ln g_i/∂w_ij`, the expected update vector has a nonnegative inner product with `∇_W E{r | W}`, and when the rate factor is the same constant for every weight the expected update equals `α ∇_W E{r | W}` (Theorem 1, §4).
- **Guideline:** When the baseline `b_ij` is computed without using the current sampled output `y_i` or the current reinforcement `r`, the update is still a REINFORCE algorithm and Theorem 1 applies; the paper reports an adaptive reinforcement-comparison baseline improving convergence speed empirically, and states that its analysis gives no basis for choosing among baselines (§4, §8.3).
- **Authors:** Ronald J. Williams (Northeastern University)
- **Year:** 1992 (Machine Learning 8, 229–256)
- **URL:** https://link.springer.com/article/10.1007/BF00992696
- **Source type:** paper
- **Relevant topics:** policy gradient, score function estimator, reinforcement baseline, episodic credit assignment

## Abstract
The paper defines a class of associative reinforcement-learning algorithms for connectionist networks with
stochastic units, named REINFORCE. The algorithms adjust weights along the gradient of expected reinforcement, for
immediate-reinforcement tasks and certain limited forms of delayed-reinforcement task, without explicitly computing
or storing gradient estimates. It gives specific instances of the class, shows that some previously published
algorithms are members, and shows how the class integrates with backpropagation (Abstract, p. 229).

## Key Contributions
- Defines a REINFORCE algorithm by its update form `Δw_ij = α_ij (r − b_ij) e_ij`, with `e_ij = ∂ln g_i/∂w_ij` named
  the *characteristic eligibility* of `w_ij`, `α_ij ≥ 0` a rate factor, and `b_ij` a reinforcement baseline
  conditionally independent of `y_i` given `W` and `x^i`. The name is an acronym for "REward Increment = Nonnegative
  Factor × Offset Reinforcement × Characteristic Eligibility" (§4).
- Theorem 1: for any REINFORCE algorithm the inner product of `E{ΔW | W}` and `∇_W E{r | W}` is nonnegative; with all
  `α_ij > 0` it is zero only at `∇_W E{r | W} = 0`; and with a common constant `α` the expected update equals
  `α ∇_W E{r | W}`. The last statement is equivalent to `(r − b_ij) ∂ln g_i/∂w_ij` being an unbiased estimate of
  `∂E{r | W}/∂w_ij` (§4; proof in Appendix A).
- Episodic REINFORCE for `k`-step episodes with one terminal reinforcement:
  `Δw_ij = α_ij (r − b_ij) Σ_{t=1..k} e_ij(t)`, derived by unfolding in time; Theorem 2 gives the same three
  conclusions for this form (§5, Eq. 11; proof in Appendix A).
- Extends the framework to multiparameter output distributions, including a Gaussian unit whose mean and standard
  deviation are both computed and both adapted (§6), and shows the eligibility can be backpropagated so REINFORCE
  units can sit inside a network trained by backpropagation (§7).

## Key Figures/Tables to Study
- §4, Eq. (5) and §5, Eq. (11): the Bernoulli characteristic eligibility and the episodic update.
- §8.3: baseline choice, including the Dayan (1990) minimum-variance baseline.
- Appendix A: the proofs of Theorem 1 and Theorem 2.

## Technical Details

### Setting
The analysis is for *associative immediate-reinforcement* tasks: the learner maps an input to an output and receives
a scalar reinforcement `r` determined by the most recent input-output pair (§1, §3). The performance criterion is
`E{r | W}` and the objective is to find a weight matrix maximizing it (§3).

### REINFORCE update
`Δw_ij = α_ij (r − b_ij) e_ij`, `e_ij = ∂ln g_i/∂w_ij` (§4)
- `r`: the scalar reinforcement received at the end of the trial; `g_i`: the probability mass or density function
  from which unit `i` draws its output `y_i`.
- `b_ij`: the reinforcement baseline, required to be conditionally independent of `y_i` given `W` and `x^i`;
  `α_ij`: a nonnegative rate factor depending at most on `w_i` and the trial index.

### Bernoulli-logistic instance
For a Bernoulli unit with `p_i = Pr{y_i = 1}`, `∂ln g_i/∂p_i = (y_i − p_i)/(p_i(1 − p_i))` (§4, Eq. 5). With `b_i = 0`
and `α_i = ρ_i p_i(1 − p_i)` the semilinear-unit update is `Δw_ij = α(r − b_ij)(y_i − p_i)x_j` (§4).

### Episodic form
Over a `k`-step episode with a single terminal reinforcement `r`: `Δw_ij = α_ij (r − b_ij) Σ_{t=1..k} e_ij(t)`
(§5, Eq. 11); for Bernoulli-logistic units, `Δw_ij = α_ij (r − b_ij) Σ_t [y_i(t) − p_i(t)] x_j(t − 1)` (§5). The
eligibility sum fits in one accumulator per weight, because each term depends only on the network's operation and
not on the reinforcement eventually received (§5).

### Rewards at every step
When reinforcement arrives at every step the performance measure is `E{Σ_{t=1..k} r(t) | W}`. The paper substitutes
`Σ_t r(t)` for `r` in Eq. (11), and notes that when `r` is causal there is "a potentially better way to perform the
necessary credit assignment": treat the `k`-step interval as `k` overlapping episodic problems all starting at the
beginning of the episode. The details are omitted (§5). The discounted return-to-go form used in later work is not
derived here.

### Reinforcement comparison as a baseline
Reinforcement comparison (attributed to Sutton, 1984) maintains an adaptive estimate `r̄` of upcoming reinforcement,
for example by exponential averaging `r̄(t) = γ r(t − 1) + (1 − γ) r̄(t − 1)` with `0 < γ ≤ 1` (§4, Eq. 10). It is a
REINFORCE algorithm as long as `r̄` is never computed from the current `y_i` or the current `r` (§4).

### Baselines and variance
Theorem 1 applies equally to every admissible baseline. §8.3 states that the analysis "offers no basis for choosing
among various choices of reinforcement baseline", and that the claim that an adaptive baseline improves convergence
speed rests on "extensive empirical investigation" rather than on the theorems. One worked case is given: a single
Bernoulli semilinear unit with only a bias weight and always-positive `r` performs a biased random walk with
`b = 0` and has nonzero probability of converging to the inferior output, while any `b` strictly between the two
possible values of `r` moves it toward the better output (§8.3). A minimum-variance baseline, considered by
Williams (1986) and investigated by Dayan (1990), is not the mean reinforcement and is harder to estimate; the
paper reports Dayan's simulations as showing "a slight improvement in convergence speed" with "a more convincing
advantage" still to be demonstrated (§8.3). §8.1 reports Sutton (1984) finding REINFORCE with reinforcement
comparison to outperform all other algorithms in that study. §9 names the two stated disadvantages of the class:
no general convergence theory, and susceptibility to convergence to false optima.

## Connections
- [[trpo]] — constrains the update with a trust region instead of a small rate factor; [[ppo]] — first-order
  surrogate with ratio clipping, whose `L^PG` term is the likelihood-ratio estimator.
- [[rloo]] — baseline is the mean of the other k−1 samples for the same prompt; [[grpo]] — group mean and standard
  deviation, whose standard-deviation term is not action-independent in the sense of §4, as [[dr-grpo]] analyzes;
  [[rloo-vs-grpo]] compares the two; [[reinforce-plus-plus]] — a 2025 variant for LLM training.

## Verification
- Checked on 2026-09-18 against: https://link.springer.com/content/pdf/10.1007/BF00992696.pdf (Machine Learning 8,
  229–256, 1992), full text.
- Corrections to the previous card version:
  - "Proves that adding a state-dependent baseline b(s) is unbiased and reduces variance" → Theorem 1 holds for any
    baseline conditionally independent of the sampled output; the paper states its analysis gives no basis for
    choosing among baselines and presents the convergence-speed advantage as empirical (§4, §8.3).
  - "Theorem 1 (baseline invariance)" → Theorem 1 states the inner-product result for `E{ΔW | W}` and
    `∇_W E{r | W}` (§4); baseline choice is discussed in §8.3.
  - "Section 2: derivation of the score function gradient" → the update and eligibility are defined in §4, proofs in
    Appendix A; §2 describes the network model.
  - "Per-step variant (causal form) `G_t = Σ_{t'≥t} γ^{t'−t} r_{t'}`" → the paper describes the causal case only as
    `k` overlapping episodic problems and omits the details; no discount factor or return-to-go formula appears (§5).
  - Year/venue made explicit: Machine Learning 8, 229–256 (1992); `Source type` field added.
- Removed as unsupported by the source:
  - "Always subtract a baseline b(s) ...; never omit it in high-dimensional action spaces" — no such recommendation;
    the paper's `b = 0` examples are valid REINFORCE algorithms.
  - "Algorithmic template for every modern policy-gradient method (PPO, TRPO, RLOO, GRPO)" as a contribution of this
    paper; those are later works, now listed under Connections.
  - The baseline table rows for Actor-Critic, RLOO, GRPO and PPO, the "LLM-specific form" section, and "Variance
    issues" including "raw REINFORCE variance scales with |y|" — none of this is in Williams (1992).
- Not reported by the source: variance bounds for the estimator; convergence guarantees; discounting; state-value-function baselines.
