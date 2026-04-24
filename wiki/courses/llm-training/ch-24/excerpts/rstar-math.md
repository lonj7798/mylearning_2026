---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rstar-math.md
source_url: https://arxiv.org/abs/2501.04519
created_at: "2026-04-23"
---

# Excerpt: rStar-Math — code-augmented MCTS and the pairwise Process Preference Model

**Source library:** `wiki/raw-data/llm-training/papers/rstar-math.md`
**Paper:** Guan et al. 2025, "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking" (MSRA)

---

## Why this source anchors ch-24 §5

rStar-Math is the 2025 synthesis of three threads ch-24 tracks separately: (1) MCTS as a test-time reasoning procedure (from [[rstar]]), (2) step-level correctness signals (from [[omegaprm]] / [[let-verify]]), and (3) iterated self-evolution (from STaR-lineage). The paper ships **747K verified step-level trajectories** that take a 7B base from baseline to o1-preview-level math.

---

## The two novel moves

**Move 1: Code-augmented MCTS steps.** Each MCTS node corresponds to a single reasoning step, and a step has the shape `(natural-language thought, Python code block)`. The code is executed at node-expansion time. **Execution failure ⇒ node pruned from the tree.** This gives step-level correctness *without ever seeing the gold answer at the step level* — the step-level signal is whether the code ran, not whether the math is right. The trajectory-level signal (gold-answer match) still requires ground truth.

Ch-24 §5 presents the pseudocode. The key loop — restated from the source's description, written out in executable form:

```
def mcts_round(root, policy, gold):
    for _ in range(N_rollouts):
        node = select_by_ucb(root)         # UCB / PUCT descent
        for _ in range(K):
            (thought, code) = policy.sample(prefix=node.trace)
            try:
                exec(code)                 # step-level verification
            except Exception:
                continue                   # prune
            node.add_child(Node(thought, code))
        leaf = rollout_to_terminal(node.children[0], policy)
        r = int(extract_boxed(leaf) == gold)
        backprop_Q_N(leaf, r)
```

Two subtleties the source calls out. First, "code executed" is a broad filter: a step whose code produces numerical nonsense passes as long as no exception fires. The *content* is evaluated only through the downstream rollout's terminal reward. Second, the UCB formula is the PUCT variant `Q + c_puct · P · √N_parent / (1 + N_child)`, same as AlphaZero/MuZero.

**Move 2: Pairwise Process Preference Model (PPM).** From the source (§Synthesis pipeline):

> Within each problem, pairs of sibling MCTS steps with high vs low Q-value form step-preference pairs. PPM trained with pairwise ranking loss.

The PPM is **not a scalar regression** onto a step-value like OmegaPRM's MC targets. It is a pairwise Bradley-Terry ranker:

```
L_PPM = -log σ( r_φ(step_high, prefix) - r_φ(step_low, prefix) )
```

where (step_high, step_low) are MCTS siblings sharing the same prefix, with Q-gap > δ.

The authors' argument for pairwise-over-scalar, from the source (§Modality-specific):

> PPM is not a scalar PRM: authors argue pairwise training avoids the Goodhart-style issues of scalar reward regression observed in math-shepherd / prm800k.

Scalar PRMs fit an *absolute* step-value, which a policy can then exploit by producing steps whose surface features correlate with high predicted value but whose content is worse. Pairwise training only constrains *relative* preferences between siblings — the student learns "this step is better than that step in this prefix," not "this step has value 0.83."

---

## The four-round self-evolution

From the source (§Key Contributions):

> Four-round self-evolution: each round's top policy samples new trajectories, PPM retrains, generator retrains.

Concretely:
- **Round 0**: bootstrap with Qwen2.5-Math-7B-Instruct as the policy; seed problem pool = ~747K from [[numina-math]] + MATH + GSM8K + olympiad + AIME.
- **Round k → Round k+1**: (a) run MCTS with the round-k policy, (b) extract step-pairs with Q-gap > δ to retrain the PPM, (c) take top-K trajectories by PPM score to SFT the next policy.

The round-by-round MATH curve is the punchline: **58 → 78 → 85 → 88 → 90**. Compounding.

One risk the source flags ch-24 §5 also carries forward:

> Compounding distribution narrowing: four self-evolution rounds risk collapsing to a small region of the solution space; authors mitigate with temperature scheduling.

This is the ch-23 model-collapse warning applied to reasoning-trace loops — each round of self-evolution risks narrowing the policy's support. The mitigation is temperature scheduling (don't anneal too aggressively) and keeping the problem pool fixed across rounds.

---

## Why step-level beats trajectory-level (the key comparison)

From the source (§Quality evaluation):

- PPM ablation: replacing PPM with a scalar PRM loses **6 MATH points**.

This is the empirical cousin of the Step-DPO argument (§6 of ch-24): step-level signal concentrates gradient on actual disagreement, rather than diluting it across a long shared prefix. At 6 points absolute on MATH, the effect is as large as a round of self-evolution. For Track-4 RL consumers, this is the recommendation: if your PRM is scalar-regressed, swap to pairwise before scaling.

---

## Headline numbers

From the source (§Quality evaluation):

- Qwen2.5-Math-7B-rStarMath: **90.0 MATH, 53.3 AIME24, 58.5 Olympiad**.
- Beats o1-mini on MATH; matches o1-preview on several benchmarks.

Still gold-answer-dependent: the trajectory-level reward requires known final answers. The novelty is that the **step-level signal is executability**, which is effectively free.

---

## Connections

- [[excerpts/omegaprm]] — MC-value-regressed PRM, the scalar alternative the PPM's pairwise training is designed to beat.
- [[excerpts/step-dpo]] — pairwise *policy* preference (not a separate PPM head) — the DPO-flavoured cousin.
- [[excerpts/openmathinstruct]] — terminal-only CoT filter; rStar-Math's inner loop is the strict upgrade.
- [[ch-24]] §5 (MCTS synthesis), §6 (step-level supervision), §8 (practical guidance).
