---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: "On the difficulty of training Recurrent Neural Networks (Pascanu, Mikolov & Bengio)"
source_url: https://arxiv.org/abs/1211.5063
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: gradient norm clipping (Pascanu, Mikolov, Bengio)

**Authors:** Razvan Pascanu, Tomas Mikolov, Yoshua Bengio. arXiv:1211.5063 (v1 2012-11; read v2, 2013-02-16). ICML 2013.
**Status:** rewritten on 2026-09-15 from the PDF. The library card `classics/gradient-clipping.md` had no Verification section on that date; its LLM defaults, RL "10× norm bursts", "100× spike" monitoring rule, and FSDP notes are not statements of this paper and are not used by ch-01.

## Algorithm 1 (§3.2 "Scaling down the gradients")
```
ĝ ← ∂E/∂θ
if ‖ĝ‖ ≥ threshold then
    ĝ ← (threshold / ‖ĝ‖) · ĝ
end if
```
- ĝ is the full gradient vector over all parameters; ‖·‖ is its L2 norm. The rescaling keeps the direction of ĝ and bounds its norm by the threshold.
- "This algorithm is very similar to the one proposed by Tomas Mikolov and we only diverged from the original proposal in an attempt to provide a better theoretical foundation (ensuring that we always move in a descent direction with respect to the current mini-batch), though in practice both variants behave similarly." (Mikolov's variant clips "the gradient's temporal components element-wise (clipping an entry when it exceeds in absolute value a fixed threshold)", §3.1.)

## Threshold choice (§3.2)
"One good heuristic for setting this threshold is to look at statistics on the average norm over a sufficiently large number of updates. In our experiments we have noticed that for a given task and model size, training is not very sensitive to this hyper-parameter and the algorithm behaves well even for rather small thresholds."

## Interpretation given by the authors (§3.2)
"The algorithm can also be thought of as adapting the learning rate based on the norm of the gradient. ... we rely on the instantaneous gradient. This means that we can handle very abrupt changes in norm, while the other methods would not be able to do so."

## Scope
The paper studies recurrent networks (synthetic long-term-dependency tasks, polyphonic music, character-level Penn Treebank). It has no Transformer or LLM experiments. Implementation details for mixed precision and sharded training come from other sources ([[pytorch-adamw-clip-amp]]).

## Verification
- Checked on 2026-09-15 against arXiv:1211.5063v2 (§3.2, Algorithm 1).
