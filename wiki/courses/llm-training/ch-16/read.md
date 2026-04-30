<!-- chapter: ch-16
     track: data
     title: RL Prompt Curation and Replay Data
     sources: [[replay-buffer-rlhf]], [[kimi-k1-5]], [[kimi-k2]], [[tulu-3]], [[rlvr-tulu3]], [[reinforcement-learning-with-one-training-example]], [[on-off-policy-rlhf]], [[minibatch-sharing-rl]], [[policy-coverage-loss]], [[prorl]]
     figures: figures/rollout-passrate.html
-->

# Chapter 16 — RL Prompt Curation and Replay Data

> **Core insight.** An RL prompt is not an SFT prompt. An SFT prompt needs a target completion; an RL prompt needs (a) a verifier or judge that returns a scalar reward and (b) a **non-degenerate rollout distribution** — some rollouts must succeed and some fail under the current policy, or the gradient is zero. A 100%-pass prompt wastes rollouts; a 0%-pass prompt wastes rollouts *and* depresses loss; `p̂ ∈ [0.1, 0.9]` is where the gradient signal lives.
>
> **Guideline.** Curate by (1) pairing every prompt with a verifier/judge, (2) mining pass-rate at the current policy and keeping the middle band, (3) *replaying prompts, never replaying trajectories* (the IS ratio explodes), and (4) re-measuring pass-rates as the policy moves. [[kimi-k1-5]] is the reference recipe; [[tulu-3]] is the open-source instantiation; [[reinforcement-learning-with-one-training-example]] shows the extreme edge.

---

## Why this chapter exists

[[ch-15]] closed the human-annotation track: rubric design, adjudication, preference sampling. Those outputs feed DPO and reward-model training — not an RLVR or rubric-RL loop. RL-stage training consumes a different artifact: a *pool of prompts* that the policy rolls out against, with rewards produced by a verifier, a reward model, or the model itself. Three traditions converge: RLVR ([[rlvr-tulu3]], [[tulu-3]]) treats the prompt's only role as carrying a verifier `v: (x, y) → {0,1}`; long-CoT RL with partial rollouts ([[kimi-k1-5]]) keeps prompts in a buffer and drives selection by live pass-rate; agentic RL ([[kimi-k2]]) uses environment seeds and outcome-success verifiers.

Across all three, the same operational question dominates: **which prompts go into the next batch, and in what proportion?** The naive answer ("all of them, uniformly") is wrong — zero-variance prompts contribute zero gradient under GRPO/RLOO ([[minibatch-sharing-rl]]), and always-fail prompts waste rollouts. The corrected answer is pass-rate-filtered, variance-weighted, and periodically re-measured against the moving policy. §1 defines the RL prompt object. §2 covers difficulty mining. §3 derives the stale-policy replay bias. §4 presents curriculum schedules. §5 bridges to the synthetic-prompt track.

---

## 1. What an RL-stage prompt actually is

An **SFT prompt** is a tuple `(x, y*)` — input + target. Training loss is `−log π_θ(y* | x)`. The target supplies the gradient direction.

An **RL prompt** is a tuple `(x, R)` where `R` is a *mechanism* — a function or process that, given any completion `y` the policy produces, returns a scalar reward. There is no target completion. The policy itself produces `y ~ π_θ(·|x)`, and the gradient comes from `(r(x, y) − b) ∇ log π_θ(y|x)` where `b` is some baseline. The prompt carries no supervision signal of its own — it carries the *ability to grade*.

That distinction forces three prompt-pool invariants:

**(i) Every prompt must be paired with a grader.** [[rlvr-tulu3]] is emphatic: *"for every prompt, define a per-example verifier that returns binary {0,1}"*. The grader types observed in 2024–2025 production:

| Grader type | Examples | Where from |
|---|---|---|
| Exact-match / symbolic | GSM8K integer match, MATH SymPy equivalence | [[tulu-3]], [[rlvr-tulu3]] |
| Unit-test execution | LiveCodeBench, SWE-bench cases | [[tulu-3]], [[kimi-k2]] |
| Constraint checker | IFEval regex constraints, JSON-schema validators | [[tulu-3]] |
| Scalar reward model | Bradley–Terry RM on preferences | classical RLHF |
| CoT reward model | model generates reasoning trace, then JSON verdict | [[kimi-k1-5]] (98.5% val-acc vs 84.4% scalar head) |
| Self-critique rubric | model produces rubric + scores its own completion | [[kimi-k2]] |
| Environment outcome | tool-call returns success / test env state | [[kimi-k2]] agentic RL |

A prompt without a grader cannot enter the RL pool. This is why the RL pool is typically a **subset** of the SFT pool, not a superset — instruction-following chit-chat has no mechanical verifier, so it stays in SFT or goes to DPO/RLHF with a learned RM.

**(ii) The "prompt" is really `(x, R, metadata)`.** The metadata slot carries difficulty tag, reference answer for verifiers that need it, mix identifier (which curriculum stage), and *most importantly* the rolling pass-rate estimate described in §2. Without these fields the prompt cannot be routed through the curriculum.

**(iii) RL prompts have a shelf life tied to the policy.** A prompt that was "hard" at step 0 (pass-rate 0.2) may be "solved" by step 5000 (pass-rate 0.95). Its gradient contribution degrades smoothly with policy improvement. Unlike SFT examples — whose utility is near-stationary — RL prompts are *non-stationary assets* whose value depends on the current `π_θ`. This is the single most often-missed property and it drives every operational choice below.

---

## 2. Difficulty mining — the pass-rate filter

**The filter.** Let `p̂(x) = (1/K) Σ_k 1[v(x, y_k) = 1]` where `y_k ~ π_θ(·|x)`, `k = 1..K`. The empirical pass-rate is the policy's probability of solving the prompt under the current sampling temperature. The prompt-pool filter is a thresholded band `[p_lo, p_hi]` — keep prompts with `p̂ ∈ [p_lo, p_hi]`, drop the rest.

**Why a band and not just a lower bound.** Two boundary cases reveal why both thresholds matter.

- `p̂ = 1` (always solves): the group baseline `μ = mean_k R_k = 1` and advantage `A_k = R_k − μ = 0` for all rollouts. GRPO gradient is zero. Same for RLOO: leave-one-out mean is 1, advantage is 0. The prompt contributes nothing — and worse, it *consumes* `K` rollouts of compute for that nothing. [[minibatch-sharing-rl]] formalizes this: the group-relative estimator is well-defined only when `var_k R_k > 0`.
- `p̂ = 0` (never solves): same degeneracy — all rewards are 0, advantage is 0 everywhere, gradient is zero. Additionally, always-fail prompts bias the learned value function (if PPO is used) toward pessimism.

The informative band is `0 < p̂ < 1`, with the sweet spot near `p̂ ≈ 0.5` where reward variance is maximized (`Var(Bernoulli(p)) = p(1−p)` peaks at `p = 0.5`). A typical production band:

| Setting | `p_lo` | `p_hi` | Rationale |
|---|---|---|---|
| Stable-phase RL ([[tulu-3]] RLVR) | 0.1 | 0.9 | wide band; accept low-variance prompts for diversity |
| Hard-reasoning RL ([[kimi-k1-5]]) | 0.05 | 0.5 | skew harder; prioritize near-miss prompts |
| ProRL boundary-expansion ([[prorl]]) | 0.0 | 0.3 | accept zero-pass-rate prompts *if* any single rollout succeeds over K_big |
| Cold-start (fresh policy) | 0.2 | 0.8 | medium band; avoid 0-pass prompts that would destabilize early training |

**The Kimi K1.5 measurement protocol.** [[kimi-k1-5]] is explicit about how pass-rate is estimated: *"difficulty from pass-rate of 10 SFT samples at high temperature."* Not the current policy — the SFT-stage policy, `K = 10` rollouts at high temperature. This has two consequences:

1. The measurement is cheap (one-shot, ~10K tokens per prompt).
2. The measurement is *stale* the moment RL starts. K1.5's curriculum and prioritized-sampling loop implicitly re-ranks prompts as the RL policy improves — `p(sample prompt i) ∝ 1 − success_rate_i` is computed against the *current* RL policy, not the frozen SFT measurement.

**The Tülu 3 open-source instantiation.** [[tulu-3]] builds its RLVR prompt pool by:

1. Starting from the SFT mix (939K prompts).
2. Keeping only prompts whose domain has a verifier — GSM8K, MATH, IFEval-style, code-with-tests.
3. Filtering to a smaller pool (~10^5 prompts) after difficulty estimation.
4. Running PPO for 10M episodes against that pool.

The ablation is buried in the paper but the directional claim is clear: filtered-by-pass-rate beats unfiltered by a few points on downstream evals, and the gain is robustly attributable to removing easy prompts (not hard ones).

**The one-shot RLVR evidence.** [[reinforcement-learning-with-one-training-example]] pushes the logic to its extreme: on Qwen2.5-Math-1.5B, *one* well-chosen prompt lifts MATH500 from 36.0% → 73.6%. The selection heuristic is **historical-variance score** — across training epochs on the full dataset, rank prompts by variance of their per-epoch accuracy. This is the pass-rate filter's derivative form: instead of absolute pass-rate, rank by *time-variation of pass-rate*, because a prompt whose difficulty fluctuates across training is a prompt whose gradient signal fluctuates too — and that's where the learning happens. The paper also shows that the useful prompts are not unique; many high-variance prompts work. The signal is the band, not the single best example.

---

## 3. Replay buffers and the stale-policy bias derivation

This section owes a careful derivation, because the folk version ("replay is fine, just clip") is wrong.

### 3.1 Classical replay vs policy-gradient replay

DQN, Ape-X, R2D2 rely on replay buffers because their improvement is bootstrapped from learned Q-values — the Bellman target `r + γ max_a Q(s', a)` does not depend on the behavior policy. LLM RL is different. PPO and GRPO are **policy-gradient** methods with estimator `∇ J(θ) = E_{y ~ π_θ}[ ∇ log π_θ(y|x) · A(x,y) ]` — an on-policy expectation. Stored `y ~ π_old` needs importance-ratio correction `ρ(x,y) = π_θ(y|x) / π_old(y|x)`; PPO's clip is that correction bounded.

### 3.2 The IS-ratio explosion derivation

A completion is `y = (y_1, ..., y_T)`; under a causal LM the ratio factorizes token-wise: `ρ(x,y) = Π_t ρ_t`. Assume per-token `log ρ_t ~ N(0, σ²)` (reasonable for small PPO updates; `σ²` grows with policy drift from `π_old`). Then `log ρ ~ N(0, T·σ²)` and

```
E[ρ]   = exp(T · σ² / 2)
Var[ρ] = exp(2 T σ²) − exp(T σ²)
```

Concrete numbers from [[replay-buffer-rlhf]], with per-token drift `σ ≈ 0.01` (so `σ² = 1e-4`) accumulated over `K` gradient-update steps since storage (so effective `σ² = K · 1e-4`):

| `K` steps stored | `T = 100` | `T = 1000` |
|---|---|---|
| 1 | `E[ρ] ≈ 1.005` (safe) | `E[ρ] ≈ 1.05` (safe) |
| 20 | `E[ρ] ≈ 1.10` (clip-reachable) | `E[ρ] ≈ 2.7` (outside any clip) |
| 50 | `E[ρ] ≈ 1.28` (marginal) | `E[ρ] ≈ 12` (clipping discards ~all samples) |

Variance is worse — it grows like `exp(2 T σ²)`, so the estimator blows up before the mean does. **Conclusion.** For replay interval `K ≥ 10` and response length `T ≥ 100`, trajectory-level replay injects either biased (clipped) or unboundedly-variant (unclipped) gradient updates. This is the mechanical reason [[replay-buffer-rlhf]] and [[deepseek-r1]] §3.2 both abandon trajectory replay. It is also the formal version of [[on-off-policy-rlhf]]'s finding that ~80% of the offline-DPO vs PPO gap is distribution shift — same pathology, now inside a PPO loop.

### 3.3 Three compounding pathologies

Beyond the IS ratio ([[replay-buffer-rlhf]]): (1) **Stale advantages** — GRPO's `A_{i,k} = (R_{i,k} − μ_i) / (σ_i + ε)` is normalized at rollout time; re-normalizing against current stats helps but does not fix IS drift. (2) **KL drift** — the reward stream includes `−β · KL(π_θ || π_ref)` and `π_ref` may have moved (e.g. [[prorl]] resets it), making the stored reward inconsistent with the current gradient. (3) **Critic staleness** — for PPO with a learned critic, `V_φ` has moved; `A = R − V` uses a stale `V`.

### 3.4 What survives — prompt-level replay

What [[replay-buffer-rlhf]] identifies as the operative 2024–2026 pattern: **replay the prompt, never replay the completion.** The buffer stores `(x, group_rewards, group_variance, seen_steps)`. On the next step, sample prompts from the buffer weighted by variance, but regenerate `y ~ π_θ` fresh. No IS correction is needed because the completion is on-policy; only the *prompt* was sampled from a possibly-non-uniform distribution, and that non-uniformity is the feature (hard/high-variance prompts get seen more).

TRL's `GRPOWithReplayBufferTrainer` ([[replay-buffer-rlhf]]) is the reference:

```python
@dataclass
class BufferEntry:
    prompt_ids: torch.Tensor
    rewards:    torch.Tensor      # (n,) per-rollout outcome rewards
    variance:   float              # rewards.var().item()
    seen_steps: int

# Sampling step
p_replay = 0.25
for i in range(B):
    if random() < p_replay and len(buffer) > 0:
        # draw from buffer, weighted by variance
        probs = np.array([e.variance + eps for e in buffer])
        probs /= probs.sum()
        entry = np.random.choice(buffer, p=probs)
        batch.append(entry.prompt_ids)
    else:
        batch.append(sample_fresh_prompt())

# After rollouts: insert each prompt back with updated variance.
```

Two details that are load-bearing:

- **Zero-variance downweight.** Entries with `variance == 0` (all rollouts correct or all incorrect) get probability 0. This is the pass-rate filter from §2 expressed as a sampling weight. A prompt that has become "solved" is silently ejected from the effective buffer without an explicit eviction rule.
- **Fresh regeneration.** The completions produced in the previous step are *discarded*. This is the key difference from classical trajectory replay. Storage is `O(#prompts)` not `O(#prompts × avg_tokens)`.

### 3.5 Kimi K1.5's partial rollouts — the legitimate exception

[[kimi-k1-5]] describes **partial rollouts**: long responses are broken into segments across training iterations, with previous trajectory segments reused from a replay buffer. This looks like trajectory replay, but is not:

- The reused segment is a *prefix* (prompt + early reasoning). Token positions after the prefix are rolled out fresh under the current policy.
- Because the prefix is never used as a `∇ log π_θ` target — only as a *context* for fresh sampling — the IS-ratio argument does not apply. It's more like prompt-extension than trajectory replay.
- The infrastructure win: the fixed output-token budget per iteration caps GPU memory, letting K1.5 scale to 128K RL context without quadratic rollout cost.

This pattern is worth internalizing because it looks like a counterexample and isn't one. The rule "don't replay completions you'll take gradients through" still holds.

---

## 4. Curriculum in prompt space

Three concrete curricula observed in 2024–2025 production, tabulated.

| Schedule | Who uses it | Stage 1 (cold-start) | Stage 2 (bulk) | Stage 3 (anneal) |
|---|---|---|---|---|
| Fixed-band ([[tulu-3]] RLVR) | Tülu 3, OLMo 2 | pool-wide, pass-rate ∈ [0.1, 0.9] | same | same (no annealing) |
| Prioritized 1−p ([[kimi-k1-5]]) | Kimi K1.5 | sample ∝ 1 − p̂, SFT-policy measured | sample ∝ 1 − p̂, RL-policy re-measured every ~500 steps | optionally narrow to `p̂ < 0.3` for final push |
| Annealed difficulty ([[prorl]]) | ProRL | broad task suite at medium difficulty | gradually narrow to harder-only prompts | reset reference policy + broaden again |
| Agentic outcome-filtered ([[kimi-k2]]) | Kimi K2 | environment-graded easy tool-call tasks | 20K-tool mixed pool, outcome-verified | long-horizon multi-tool tasks |

Four design principles cut across all four schedules:

**(a) Cold-start prompts must be solvable.** A fresh SFT policy has degenerate pass-rates on genuinely hard prompts (stuck at 0); starting RL against only `p̂ < 0.1` prompts produces zero gradient, and the run never leaves its initial configuration. [[kimi-k1-5]] frames this as "curriculum sampling" — begin near current competence. [[tulu-3]]'s fixed band `[0.1, 0.9]` is the lazy-but-robust version.

**(b) Annealing requires re-measurement.** A schedule "week 1: easy, week 2: medium, week 3: hard" without re-measuring `p̂(x)` against the current policy assumes prompt difficulty is intrinsic. It is not — difficulty is relative to `π_θ`. [[kimi-k1-5]]'s "prioritized sampling ∝ 1 − success_rate" works because success_rate is tracked *per prompt per epoch* against the live policy.

**(c) Reference-policy resets ([[prorl]]) are curriculum in disguise.** The reset step effectively re-curates the pool: previously-memorized prompts get re-shuffled because the new reference changes the KL-penalty landscape and the pass-rate distribution widens. The boundary-expansion claim [[prorl]] makes is operationally about curriculum-plus-resets, not a novel algorithm.

**(d) Policy coverage, not reward fidelity.** [[policy-coverage-loss]] formalizes the point: a signal is useful only if its induced distribution overlaps the target's support. The pass-rate filter is the operational form — `p̂ > 0` is exactly "non-zero coverage of the correct-answer set."

---

## 5. Bridge to synthetic — why Track 3 inherits this chapter

Two 2025 data points force the hand-to-synthetic transition.

**Prompt-pool exhaustion.** [[tulu-3]]'s 939K SFT pool yields ~10^5 verifiable RLVR prompts after filtering. One RL run burns 10^7 episodes (Tülu 3: exactly 10M); at `K = 8` rollouts/prompt that's 1.25M prompt-visits, ~12× per prompt. Pass-rates stabilize; the pool is "done." The next run wants fresh prompts, and human curation at this volume is infeasible.

**The verifier is the bottleneck.** An RL prompt without a verifier is useless (§1). Generating a math problem is easy; generating `(problem, reference_answer, equivalence-grader_spec)` is harder — but synthetic-friendly. A strong model produces triples at scale; a second-stage filter keeps only those where independent verification agrees. [[kimi-k2]]'s agentic pipeline — 20K+ tools, real + simulated environments, outcome-success verifiers filtering trajectories — is this same recipe at the environment level.

Track 3 (synthetic data generation) builds exactly this pipeline. The prerequisites this chapter installs: "RL prompt = `(x, R, metadata)` with a grader" (§1) the generator must produce; pass-rate filter (§2) against the target policy; prompt-level replay (§3) — synthetic prompts are a replay buffer to the trainer, and generated *solutions* must not be replayed as trajectories; curriculum re-measurement (§4) — solved prompts become negative-yield and the generator must be re-queried. This chapter's role is to make clear what the prompts must look like before they enter the RL loop, regardless of who wrote them.

---

## 6. A drop-in reference — RL prompt-pool manager

Combining §2, §3, §4 into the canonical prompt-pool manager that sits between the data pipeline and the RL trainer. Naming follows TRL + verl conventions.

```python
@dataclass
class RLPrompt:
    prompt_ids: torch.Tensor
    verifier_id: str            # "math_exact", "code_tests_42", "ifeval_json"
    reference:   dict | None    # e.g. {"answer": "42"}
    mix_tag:     str
    pass_rate:   float | None = None     # EMA of per-prompt mean reward
    var:         float        = 0.0       # reward variance last seen
    seen_steps:  int          = 0

class RLPromptPool:
    """Band filter + variance-weighted replay + periodic re-measurement.
       Prompts, not trajectories — storage is O(#prompts)."""
    def __init__(self, prompts, p_lo=0.1, p_hi=0.9,
                 p_replay=0.25, ema_alpha=0.3, remeasure_every=500):
        self.prompts = list(prompts)
        self.p_lo, self.p_hi = p_lo, p_hi
        self.p_replay, self.ema_alpha = p_replay, ema_alpha
        self.remeasure_every, self.step = remeasure_every, 0

    def _in_band(self, p): return p is None or self.p_lo <= p <= self.p_hi

    def sample_batch(self, B):
        band   = [pr for pr in self.prompts if self._in_band(pr.pass_rate)]
        replay = [pr for pr in band if pr.var > 1e-6]   # non-degenerate
        out = []
        for _ in range(B):
            if replay and np.random.random() < self.p_replay:
                w = np.array([pr.var for pr in replay]); w /= w.sum()
                out.append(replay[np.random.choice(len(replay), p=w)])
            else:
                out.append(band[np.random.randint(len(band))])
        return out

    def update_after_rollouts(self, prompts, rollouts):
        for pr, rewards in zip(prompts, rollouts):
            p_hat, v_hat = float(rewards.mean()), float(rewards.var())
            pr.pass_rate = (p_hat if pr.pass_rate is None
                            else (1-self.ema_alpha)*pr.pass_rate
                                 + self.ema_alpha*p_hat)
            pr.var, pr.seen_steps = v_hat, pr.seen_steps + 1
        self.step += 1

    def remeasure_if_due(self, rollout_fn, K=8):
        """Re-roll K completions/prompt at the *current* policy to refresh
           stale pass_rate / var. Kimi K1.5 does this implicitly via its
           prioritized-sampling loop; we make it explicit."""
        if self.step % self.remeasure_every != 0: return
        for pr in self.prompts:
            r = rollout_fn(pr, K=K)
            pr.pass_rate, pr.var = float(r.mean()), float(r.var())
```

Three invariants: (i) **no trajectory storage** — only `(pass_rate, var, seen_steps)` per prompt; completions are discarded after the gradient step, so the IS-ratio pathology from §3.2 never arises; (ii) **band filter and variance weighting are orthogonal** — the band decides eligibility, variance decides over-sampling; skip either and you get a strictly worse recipe; (iii) **re-measurement is explicit** — this is the piece most open-source reference implementations omit. Without it the band filter ossifies on stale pass-rates and the replay buffer drifts out of calibration.

---

## Connections and what's next

- **[[replay-buffer-rlhf]] / §3** — the framework-synthesis page on prompt-level replay and the IS-ratio argument.
- **[[kimi-k1-5]] / §2, §3.5, §4** — partial rollouts + prioritized sampling + pass-rate-from-SFT-policy curriculum.
- **[[kimi-k2]] / §1, §4** — agentic prompt pool with environment-graded outcome verifiers.
- **[[tulu-3]] / [[rlvr-tulu3]] / §1, §2** — the open-source RLVR prompt-curation recipe; 10M-episode budget.
- **[[reinforcement-learning-with-one-training-example]] / §2** — the extreme edge of the pass-rate-filter thesis; historical-variance ranking.
- **[[on-off-policy-rlhf]] / §3** — the theoretical ground for why distribution shift (trajectory replay's failure mode) dominates.
- **[[prorl]] / §4** — reference-policy resets as curriculum, boundary-expansion claim.
- **[[policy-coverage-loss]] / §4** — coverage as the formal cousin of the pass-rate filter.
- **ch-17 (lab)** — closes the data track. **Track 3 (synthetic data, ch-18+)** — produces RL prompts at scale; this chapter's invariants are its acceptance tests.

## Further reading

[[replay-buffer-rlhf]] (TRL `GRPOWithReplayBufferTrainer`; DeepSeek-R1 §3.2 negative-result); [[kimi-k1-5]] (partial rollouts, prioritized sampling); [[kimi-k2]] (agentic RL, joint RLVR + rubric); [[tulu-3]] / [[rlvr-tulu3]] (open RLVR pipeline; verifier taxonomy); [[reinforcement-learning-with-one-training-example]] (one-shot RLVR; historical-variance ranking); [[on-off-policy-rlhf]] (distribution-shift decomposition); [[prorl]] (prolonged RL + reference resets); [[policy-coverage-loss]] (coverage-based transfer bound); [[minibatch-sharing-rl]] (group-baseline variance math).

## Companion visualization

**[figures/rollout-passrate.html](figures/rollout-passrate.html)** — interactive rollout-pass-rate explorer. 1000 simulated prompts sampled from a beta-mixture distribution (reflecting the bimodal easy/hard shape seen in real RL pools). Drag the `p_lo` and `p_hi` thresholds and the page updates: (i) kept-prompt count, (ii) expected per-prompt reward variance (`p(1−p)`) integrated over the kept band, (iii) effective-difficulty curve showing what fraction of kept prompts sit near the high-information `p = 0.5` line, and (iv) annotated "zero-gradient zones" (`p ≈ 0` and `p ≈ 1`) with the group-baseline degeneracy note from [[minibatch-sharing-rl]]. Use it to internalize why the band matters and why the sweet spot is wider than you'd expect.
