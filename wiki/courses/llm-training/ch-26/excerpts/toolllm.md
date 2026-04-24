---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/toolllm.md
source_url: https://arxiv.org/abs/2307.16789
created_at: "2026-04-23"
---

# Excerpt: ToolLLM — ToolBench and DFS-DT

**Source library:** `wiki/raw-data/llm-training/papers/toolllm.md`
**Paper:** Qin, Liang, Ye et al. 2023, "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs."

---

## Why this source anchors ch-26

ToolLLM is the chapter's pivot from *annotating one call* to *generating a whole trajectory*. Two contributions matter.

1. **ToolBench** — the first large-scale tool-use SFT corpus grounded in real APIs, not toy tools. 16,464 REST endpoints, 126,486 instances, 469,585 real API calls. This is the corpus every open function-calling model uses either directly or as ancestor (xLAM's 3,673 executable APIs is the curated subset of ToolBench, and APIGen's pipeline re-annotates on that subset).
2. **DFS-DT** — the trajectory-synthesis algorithm that made multi-tool annotation tractable. Table 3 is the reason the paper works: at matched API-call budget, DFS-DT produces pass rates 15–20 points above ReACT on the hardest setting (intra-collection multi-tool).

Ch-26 §2 uses ToolLLM as the archetype of the "real substrate, synthetic supervision" split. This excerpt pulls out the DFS-DT pseudocode and the ReACT-comparison table that justify the split's search cost.

---

## What is real vs what is synthetic

From the source (lines 32–42):

> **What is real versus synthetic in the pipeline:**
> - **Real:** the RapidAPI catalog, API documentation, parameter schemas, code snippets, and the actual API responses returned during annotation.
> - **Synthetic:** the generated user instructions, the reasoning/action traces proposed by ChatGPT, the final natural-language answers, and much of the automatic evaluation logic.
>
> **ToolBench generation pipeline:**
> 1. **API collection:** crawl **16,464 REST APIs** from RapidAPI and store their docs and invocation details.
> 2. **Instruction generation:** sample APIs and prompt **ChatGPT (`gpt-3.5-turbo-16k`) with function-calling capability** to write tool-use instructions.
> 3. **Scenario coverage:** generate three settings: `G1` / `I1` for single-tool instructions, `G2` / `I2` for intra-category multi-tool instructions, and `G3` / `I3` for intra-collection multi-tool instructions.
> 4. **Solution-path annotation:** use the teacher model to search for a valid trajectory containing reasoning, API selection, parameter filling, real-time API execution, observation consumption, and final answer generation.
> 5. **Training export:** retain successful traces and convert them into ChatGPT-like multi-round conversation data for ToolLLaMA SFT.

The split is the durable intellectual contribution. Every pipeline downstream (APIGen, APIGen-MT, ToolACE) operates on the same frame: the API catalog is real, the instructions and traces are synthetic, the executions during annotation are real. *The environment under the trace is always real; the trace is always synthetic.* This is why "fully synthetic data" is a misnomer in function calling — the execution loop grounds the synthesis.

---

## DFS-DT: the algorithm that makes multi-tool annotation tractable

Source lines 43–48 describe the algorithm narratively. The operative sentence (line 44):

> **DFSDT trajectory synthesis:** the paper's key point is that plain ReACT is too brittle for complex multi-tool instructions. DFSDT expands a decision tree over candidate actions, allows backtracking/retraction, and explores more than one reasoning path before committing. In the appendix, the authors note that they use a preorder-style DFS variant to cut sorting cost; if no retraction is needed, the method effectively degrades to ReACT.

The critical observation is in the last clause. **Without retraction, DFS-DT is ReACT.** The search structure adds zero value if every action is committed on the first try. The entire empirical lift — 15–20 points on I3 — comes from the lines `if obs.is_error: continue` plus "sample multiple candidate actions and rank."

Ch-26 §2.3 transcribes this as pseudocode:

```python
def dfs_dt(instruction, apis, model, max_depth, beam):
    root = Node(history=[], depth=0)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_terminal() or node.depth >= max_depth:
            if node.is_successful():
                return node.trajectory     # first accepted trajectory wins
            continue
        candidates = model.sample_actions(node.history, apis, k=beam)
        for (thought, action) in sort_by_score(candidates):   # preorder
            obs = execute(action)
            if obs.is_error:
                continue                    # retract — do NOT commit this child
            child = Node(history=node.history + [(thought, action, obs)],
                         depth=node.depth + 1)
            stack.append(child)
    return None     # all branches exhausted; reject this instance
```

The preorder detail in the appendix is a cost optimisation: rather than sorting all candidates before descending, use a cheap preorder scoring pass that terminates as soon as a good-enough branch is found. In practice this halves the teacher-model cost vs a full best-first search.

---

## Table 3 — why the search cost is worth it

From the source (line 43):

> Empirically, Table 3 reports higher pass rates than ReACT and cost-matched ReACT@N on all three settings, so the same annotation budget yields more usable training trajectories.

The numbers (reproduced in ch-26 §2.3):

| Setting | ReACT | ReACT@N | DFS-DT |
|---|---|---|---|
| I1 (single-tool) | 42.2 | 47.7 | **57.3** |
| I2 (intra-cat)  | 30.0 | 34.3 | **48.2** |
| I3 (intra-col)  | 21.7 | 26.0 | **43.2** |

The ReACT@N column is the honest comparison: if you give plain ReACT the same API-call budget as DFS-DT by running N independent rollouts and keeping the first success, does the gap close? Table 3's answer is no — DFS-DT still wins by ~10 points on I3. The structural retraction is doing work that parallel independent rollouts cannot replicate, because the failure case isn't "ReACT occasionally picks a wrong action" but "ReACT commits wrong early-action observations into the context of every subsequent action."

**Practical translation.** When you cannot afford to discard half your annotation runs — and at ~$0.10 per full trajectory and 100K target instances that's a ~$5K delta — you need search with retraction, not greedy rollout plus retries.

---

## What ToolLLM does not yet verify

The paper's weakness, and the gap APIGen fills:

- Verification is binary and teacher-reported. A trajectory counts as "successful" if the teacher model says so. This lets a class of errors through: the teacher calls the wrong API with the wrong argument, gets a lucky-looking result, and self-reports success.
- No relevance-detection data. Every ToolBench trajectory assumes the offered APIs are appropriate; the model trained on it has no experience refusing irrelevant tool use. Hammer ([[hammer]]) adds this explicitly.
- ToolEval's LLM-judge reports **87.1% pass-rate agreement with humans** — good, but the residual 13% is the exact class of errors APIGen's semantic check catches.

ToolLLM remains the clearest open-source *data* source for tool use. The verification bar moved after APIGen, but the 469K real API calls in ToolBench are still the broadest grounding any modern pipeline inherits.

---

## Connections

- Upstream: [[toolformer]] — the single-call annotation precursor.
- Contemporaneous: [[gorilla]] — narrower API set (ML libraries only) but stronger retriever story.
- Direct successor on verification: [[apigen]] — takes ToolBench's 16K, curates to 3,673 executable, re-annotates with a three-layer verifier.
- Multi-turn extension: [[apigen-mt]] — blueprint-then-rollout on a similar POMDP abstraction.
- Evaluation descendant: [[bfcl]] — supersedes ToolEval with an AST matcher and ability-axis decomposition.
