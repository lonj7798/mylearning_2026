---
chapter: ch-15
course: llm-inference
phase: read
excerpt_of: "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (Li et al. 2024)"
source_url: https://arxiv.org/abs/2406.16858
created_at: "2026-05-21"
---

# Excerpt: EAGLE-2 — Dynamic Draft Trees

**Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
**Year:** 2024
**Venue:** EMNLP 2024
**URL:** https://arxiv.org/abs/2406.16858
**Raw-data source:** [[raw-data/eagle-2]]

---

## The key empirical observation

EAGLE-2 opens with a measurement: **token acceptance probability is highly context-dependent, not position-dependent.** Specifically:

- Tree position `(depth=3, branch=2)` has acceptance probability that varies from 0.1 to 0.9 across contexts.
- Static trees (the same shape every step) waste budget on low-acceptance branches in unfavorable contexts and undersample in favorable ones.

So: **make the tree adaptive to the drafter's per-context confidence.**

---

## Confidence as an acceptance-rate proxy

The paper validates that the EAGLE drafter's softmax confidence is well-calibrated with the true acceptance probability:

```
draft_confidence(x_path) ≈ P(target accepts path)
```

This calibration is the foundation: if it holds, we can use confidence as a cheap proxy for true acceptance and allocate tree expansion accordingly.

---

## The dynamic tree expansion algorithm

```
Input: max tree size M, top-K branching per node, max depth D
Initialize: tree = {root: (path=[], confidence=1.0)}
            frontier = [root]

while |tree| < M and frontier is non-empty:
    # 1. Pick the highest-confidence node in the frontier.
    node = frontier.pop_max_by(confidence)

    # 2. Expand it: drafter step produces top-K candidates.
    for x_candidate, p_candidate in drafter.top_k(node.path):
        if len(node.path) < D:
            new_node = TreeNode(
                path=node.path + [x_candidate],
                confidence=node.confidence * p_candidate,
            )
            tree.add(new_node, parent=node)
            frontier.add(new_node)

# Done: tree has M nodes, focused on high-confidence branches.
```

Two phases per step:

1. **Expansion**: greedily grow the tree, always extending the highest cumulative-confidence frontier node.
2. **Reranking**: after expansion, prune the lowest-confidence leaves to keep verified leaf set small (saves verification time).

---

## Speedup numbers (paper, Table 3)

| Model | EAGLE | EAGLE-2 | Δ |
|-------|-------|---------|---|
| Vicuna-7B | 2.72× | 3.52× | +0.80 |
| Vicuna-13B | 2.89× | 3.81× | +0.92 |
| LLaMA-2-Chat-7B | 2.66× | 3.32× | +0.66 |
| LLaMA-2-Chat-13B | 3.01× | 3.66× | +0.65 |
| LLaMA-2-Chat-70B | 3.05× | 4.26× | +1.21 |
| Mixtral-8x7B | — | 3.99× | — |

EAGLE-2 vs EAGLE-1: **20-40% improvement** across the board.

LLaMA-2-Chat-70B at 4.26× is the largest reported speedup on a 70B model with **zero quality loss** (lossless Leviathan-rule verification).

---

## Why dynamic allocation matters

Static trees with M=60 nodes typically allocate ~10-20 per depth level. If at the current step the drafter is highly confident in 3 candidates at depth 1 but uncertain at depth 2, the static tree wastes ~40 budget at depth 2 on candidates that won't be accepted.

Dynamic trees push budget deeper along the 3 high-confidence depth-1 branches. Each deep branch is more likely to commit a long-accepted sequence.

The 20-40% improvement is exactly this allocation effect — confirmed by ablation in the paper.

---

## Tree size M as the main knob

```
M=20:   light verification, fast per round, lower acceptance.
M=60:   default, balanced.
M=128:  heavier verification, higher acceptance, diminishing returns.
M=256+: usually slower (verification cost dominates).
```

At high batch (> 16), smaller M wins because target verification becomes compute-bound.

---

## Integration in serving stacks

vLLM ships EAGLE / EAGLE-2 as `speculative_config = {"method": "eagle", "model": "...EAGLE-...", "draft_tree_choices": "..."}`.

The `draft_tree_choices` parameter selects from pre-tuned tree shapes (e.g. `"mc_sim_7b_63"`, `"mc_sim_7b_64"`, `"mc_sim_7b_128"`); dynamic-tree variants are selected with `"dynamic"`.

SGLang ships EAGLE-2 with similar interface.

---

## Pitfalls

- **Calibration assumption holds only for in-distribution data.** If the drafter is OOD (e.g. trained on chat but serving code), confidence may not predict acceptance well. Validate per workload.
- **Top-K branching at root.** Choose K based on entropy at the root — high entropy (open-ended) → K=4-5; low entropy (code) → K=2-3.
- **Dynamic tree expansion is sequential per node.** Naive implementations bottleneck on the heap operations; well-tuned implementations batch the expansion across all frontier nodes.
- **Reranking saves verification time.** If you skip reranking, you verify M leaves but only commit the longest accepted path — keep the rerank.

---

## Connections

- [[excerpts/eagle]] — base EAGLE drafter; EAGLE-2 only changes the tree policy.
- [[excerpts/medusa]] — earlier tree-attention method; EAGLE-2's dynamic trees apply to Medusa too in principle.
- [[excerpts/leviathan-2023]] — the acceptance rule still applies per branch.
- [[ch-15]] — parent chapter.
