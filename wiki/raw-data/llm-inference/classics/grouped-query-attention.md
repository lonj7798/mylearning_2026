<!-- scope: Grouped-query attention as the MHA/MQA compromise used in modern LLMs
     deps: [[multi-query-attention]], [[kv-cache-memory-formula]]
     see-also: [[batching-for-inference]], [[prefill-vs-decode]]
-->

# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- **Core Insight:** Use an intermediate number of KV heads: groups of query heads share each key/value head, recovering quality close to MHA at speed closer to MQA.
- **Guideline:** For LLM serving, reason about `n_kv_heads` separately from `n_heads`; it directly sets KV-cache memory and decode bandwidth.
- **Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, Sumit Sanghai
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2305.13245
- **Relevant topics:** grouped-query attention, GQA, MQA, KV cache, uptraining

## Abstract
Grouped-query attention (GQA) generalizes multi-query attention by using more than one key/value head but fewer KV heads than query heads. The paper also proposes converting existing multi-head checkpoints to MQA/GQA via uptraining with a small fraction of original pretraining compute. It is now the standard lens for reading model configs that expose separate query-head and key/value-head counts.

## Key Contributions
- Defined GQA as the continuum between MHA (`H_kv = H_q`) and MQA (`H_kv = 1`).
- Showed that GQA can approach MHA quality with MQA-like inference speed.
- Proposed checkpoint conversion by mean-pooling heads followed by additional pretraining.
- Reported that uptraining can use about 5% of original pretraining compute.
- Made `num_key_value_heads` a standard architectural parameter in open LLMs.

## Key Figures/Tables to Study
- **Attention variant diagrams:** MHA vs MQA vs GQA.
- **Quality/speed comparisons:** Locate the best middle ground for `H_kv`.
- **Uptraining recipe:** How to convert MHA checkpoints.
- **Ablations by group count:** How many KV heads are enough?
- **Configuration fields:** Identify how model configs expose `num_key_value_heads`.

## Technical Details
If a layer has `H_q = 32` query heads and `H_kv = 8` KV heads, each KV head serves a group of 4 query heads. KV-cache memory and bandwidth scale with `8`, while the attention output still has 32 query heads.

Many modern decoder-only models expose `num_attention_heads` and `num_key_value_heads`. This is not a minor config detail: it can be the difference between fitting a long-context batch and running out of memory.

## Connections
- [[multi-query-attention]]: GQA is the practical compromise when MQA quality is insufficient.
- [[kv-cache-memory-formula]]: replace attention heads with KV heads in the cache formula.
- [[batching-for-inference]]: lower KV memory raises the maximum continuous batch size.
- [[prefill-vs-decode]]: GQA is most visible during decode and long-context serving.
