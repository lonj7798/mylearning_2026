<!-- scope: formal definition of reward hacking and the conditions under which an unhackable proxy reward can exist
     deps: [[reward-model-overoptimization]]
     see-also: [[lilianweng-reward-hacking]], [[rlvr-tulu3]]
-->

# Defining and Characterizing Reward Hacking
- **Core Insight:** On any policy set that contains an open subset — which includes the set of all stationary policies — a pair of reward functions that is unhackable and non-trivial must induce the same ordering over policies (Theorem 1, §5.1), so a proxy that differs from the true reward in any way that matters is hackable.
- **Guideline:** When a proxy reward is optimized over a policy set with volume in policy space, do not expect a better-designed or narrower proxy to remove hacking, because Theorem 1 rules it out; instead restrict the policy set or limit optimization, which is what the paper's conclusion (§7) recommends.
- **Authors:** Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, David Krueger
- **Year:** 2022 (arXiv v1 2022-09; NeurIPS 2022; arXiv v2 2025-03)
- **URL:** https://arxiv.org/abs/2209.13085
- **Source type:** paper
- **Relevant topics:** reward hacking, proxy reward, simplification, unhackability, policy classes, Goodhart's law

## Abstract
The paper gives a formal definition of reward hacking in Markov decision processes. A pair of reward
functions is *hackable* relative to a policy set Π if there exist π, π′ ∈ Π with J₁(π) < J₁(π′) and
J₂(π) > J₂(π′); otherwise the pair is *unhackable* (Definition 1, §4.2). The relation is symmetric and
is not transitive (§4.2). The paper introduces *simplification* as an asymmetric special case: R₂ simplifies
R₁ when strict inequalities in J₁ may only collapse into equalities in J₂, never reverse (Definition 2, §4.2).
The central results are that non-trivial unhackability is impossible on any policy set containing an open
set (Theorem 1, §5.1), that non-trivial unhackable pairs always exist on finite policy sets (Theorem 2, §5.2),
and that a decidable dimension condition determines when a non-trivial simplification of a given reward
exists on a finite policy set (Theorem 3, §5.2). The authors conclude that reward functions learned by
reward modeling and inverse RL are almost certainly hackable and should be treated as auxiliaries to
policy learning rather than as specifications to be optimized (§6.2).

## Key Contributions
- **Definition of hackability** as a binary relation between two reward functions relative to a policy set and
  an environment (Definition 1, §4.2), with no assumption that either policy is optimal.
- **Definition of simplification** and refinement as an asymmetric special case of unhackability (Definition 2, §4.2).
- **Impossibility on open policy sets (Theorem 1, §5.1):** if Π̂ contains an open set, any unhackable and non-trivial
  pair is equivalent on Π̂, i.e. induces the same policy ordering. Corollary 1 applies this to all stationary policies;
  Corollary 2 extends it to ε-suboptimal (ε > 0) and δ-deterministic (δ < 1) sets, so restricting attention to good or
  near-deterministic policies does not help.
- **Existence on finite policy sets (Theorem 2, §5.2):** for any finite Π̂ with at least two policies of distinct
  visit counts and any R₁, a non-trivial R₂ exists that is unhackable but not equivalent. This covers the set of
  all deterministic policies, since the MDPs considered are finite.
- **Characterization of simplification (Theorem 3, §5.2):** a non-trivial simplification of R exists on finite Π̂
  iff dim(Z₁ ∪ … ∪ Z_m) ≤ dim(F(Π̂)) − 2, where the Z_i are the visit-count sets of the equal-value partition
  classes translated to the origin. Corollary 3: if all policies in Π̂ have distinct values and |Π̂| ≥ 2, a
  non-trivial simplification always exists.
- **Software suite** released to enumerate realizable policy orderings and their non-trivial simplifications (§5.2).

## Key Figures/Tables to Study
- **Figure 1 (§2):** the three-room cleaning-robot example. With r_true = [1,1,1], the proxy [1,1,0] is unhackable
  but the proxy [1,0,0] is hackable, because the proxy ranks cleaning the attic (J = 1) above cleaning bedroom
  and kitchen (J = 0) while the true reward ranks them 1 against 2.
- **Figure 2 (§3):** true reward rising then collapsing while proxy reward keeps rising.
- **Figure 4 (§5.2):** occupancy space showing that rotating a reward to reverse one inequality must pass through
  a reward that sets another pair equal — the proof idea of Theorem 2.
- **Figure 5 (§5.3):** infinite policy sets without open subsets sometimes admit simplification (a), sometimes not (b).

## Technical Details
- **Setting:** finite MDP (S, A, T, I, R, γ) with |A| > 1, all states reachable, and R(s,a,s′) of finite mean (§4.1).
  Reward is marginalized to R(s,a); policy value is J(π) = ⟨R, F^π⟩ where F^π are discounted state-action
  visit counts (§4.1). Linearity of J in visit counts is what makes unhackability a strong condition (Abstract).
- **Why unhackability is weaker than ordinal equivalence:** an unhackable pair may have J₁(π) < J₁(π′) together
  with J₂(π) = J₂(π′) (§4.2); prior work on reward equivalence (Ng et al.) required the full ordering to be
  preserved (§3).
- **Trivial cases:** a constant reward is unhackable with respect to every reward, which is why the paper's
  results are stated for non-trivial pairs (§4.2).
- **Worked deterministic example (§5.2):** in a two-state, two-action MDP with T(s,a) = a, I uniform, γ = 0.5,
  there are four deterministic policies; of the 4! = 24 policy orderings, 12 are realizable by some reward
  function, and in each of those exactly two of the six policy pairs can be set equal without producing the
  trivial reward. Equating three policies always yields the trivial simplification.
- **Empirical context cited by the paper (§3):** Pan et al. (2022) observe reward hacking in 5 of 9 manually
  constructed proxy-reward settings; Ibarz et al. (2018) and Pan et al. (2022) report the phase-transition shape
  drawn in Figure 2.
- **Stated limitations (§6.1):** only finite MDPs and Markov reward functions; the characterization of infinite
  policy sets is incomplete; the definition is symmetric, so it counts pairs where the true reward is high and the
  proxy low as evidence of hackability even though such policies are unlikely to be reached; hackability is not a
  guarantee that hacking occurs.

## Findings relevant to generality
- The paper's §6.2 argument bears directly on breadth of capability: if a narrow-task reward function is treated as
  a proxy for a broad-values reward function, Theorem 1 says the two are invariably hackable, so the authors
  conclude that a reward function used as a specification must encode the broad objective or risk being hacked.
  They list imitation learning, constrained RL, quantilizers, and incentive management as alternatives (§6.2).
- The paper reports no experiments on language models, agentic or software-engineering environments, negative
  feedback, long context, or distillation; it is a theory paper about finite MDPs.

## Connections
- Formal counterpart to the empirical over-optimization scaling laws in **[[reward-model-overoptimization]]**,
  which measures the gold-reward decline that Theorem 1 says cannot be designed away.
- **[[lilianweng-reward-hacking]]** collects the language-model failure modes (sycophancy, length bias, format
  bias) that this paper does not itself enumerate.
- Supports the case for verifier-grounded rewards in **[[rlvr-tulu3]]** and **[[deepseek-r1]]** as a restriction of
  the optimized objective rather than a better proxy.
- Ensemble and generative-judge defenses in **[[reward-ensembling]]** and **[[generative-reward-models]]** are
  proxy-side mitigations, which Theorem 1 predicts cannot make a proxy unhackable on an open policy set.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2209.13085 (arXiv v2, 5 Mar 2025)
- Corrections to the previous card version:
  - "unhackable iff for all π, π′ ∈ Π, R̃(π) ≥ R̃(π′) ⇒ R(π) ≥ R(π′)" → hackable iff ∃ π, π′ with
    J₁(π) < J₁(π′) and J₂(π) > J₂(π′); unhackable is the negation, and it permits a strict inequality in one
    reward to be an equality in the other (Definition 1, §4.2). The stated implication is closer to Definition 2
    (simplification).
  - "R̃ must be a positive affine transformation of R (or one of them constant)" → the conclusion of Theorem 1
    (§5.1) is that the pair is *equivalent* on Π̂, meaning J₁ and J₂ induce the same ordering of Π̂; the paper does
    not state a positive affine form.
  - "Theorem 3.2 statement (the main impossibility theorem)" → the main impossibility theorem is Theorem 1 in
    §5.1; the paper has no Theorem 3.2.
  - "over the set of all stochastic policies" → the hypothesis of Theorem 1 is that the policy set contains an
    open set; Corollary 1 instantiates it for all stationary policies, and Corollary 2 for ε-suboptimal and
    δ-deterministic sets (§5.1).
  - "Fig. 1 (policy-space picture): two reward-level sets that cross" → Figure 1 is the three-room cleaning-robot
    example (§2); the occupancy-space picture is Figure 4 (§5.2).
  - "conditions are characterized via the simplicial geometry of the return vectors" → the condition is the
    dimension test of Theorem 3 on the translated visit-count sets Z_i (§5.2).
- Removed as unsupported by the source: the bullet list of failure modes attributed to the paper's related work —
  sycophancy, length bias, sentiment bias, formatting/bold-text bias, reward-model blind spots, sandbagging vs
  jailbreaks, and the Lego-stacking robot; §3 cites the boat race, an evolved radio circuit, university rankings, a
  robot occluding the camera, Atari reward-model hacking, and a Half-Cheetah cartwheel instead. Also removed:
  "Table summarizing which policy classes admit non-trivial unhackable pairs" (no such table exists); "Appendix
  enumeration of hacking examples from prior literature"; "Key lemma: unhackability implies that return vectors
  lie on a monotone curve".
- Not reported by the source: any language-model, agentic, or SWE-environment experiment; any training recipe,
  so no Recipe ledger applies.
