# Megatron-LM: Tensor Parallelism + Sequence Parallelism
<!-- slug: megatron-tp-sp · type: paper · source: https://arxiv.org/abs/1909.08053 + https://arxiv.org/abs/2205.05198 -->

**Core Insight.** Tensor parallelism (TP) splits individual weight matrices across GPUs within a node; sequence parallelism (SP) extends that to the non-TP regions (layer norm, dropout) by sharding along the sequence dimension — together they reduce per-GPU activation memory from `sbh(10 + 24/t + 5as/ht)` to `(sbh/t)(34 + 5as/h)`, a t× reduction over TP alone.

**Guideline.** Apply TP degree t = number of GPUs within an NVLink node (typically t=8). Without SP, the 10sbh term (layer norm + dropout) stays replicated and dominates at long sequences. Enable SP alongside TP — it replaces the all-reduce with AllGather + ReduceScatter at zero extra communication bandwidth cost.

## Technical Details

- **TP column-then-row split for MLP:** First linear is split column-wise (no pre-comm needed); second linear is split row-wise; one AllReduce after second linear synchronizes partial sums. Two AllReduces per transformer layer total (MLP + attention).
- **TP for attention:** Q, K, V projections split by attention heads across ranks; output projection uses row split; same AllReduce pattern.
- **8.3B model training result (Shoeybi 2019):** "15.1 PetaFLOPs across the entire application" on 512 GPUs, 76% scaling efficiency vs. single GPU baseline.
- **Activation memory before SP (TP=t):** `sbh(10 + 24/t + 5as/ht)` bytes/layer — the `10sbh` term is layer-norm and dropout, **not** divided by t because TP leaves them replicated.
- **Activation memory after SP+TP:** `(sbh/t)(34 + 5as/h)` bytes/layer — SP shards the 10sbh replicated ops along sequence dimension, replacing all-reduce with AllGather+ReduceScatter (same bandwidth, lower activation footprint).
- **Selective recomputation (Korthikanti 2022):** Recompute only the `5as²b/ht` attention softmax/score region (low compute-density ops) rather than full layers — achieves ~5× activation memory reduction with <2% additional FLOPs vs. 30–40% FLOP overhead for full recomputation.
- **530B GPT-3 on 2240 A100s:** 54.2% Model FLOPs Utilization with SP+selective recompute vs. 42.1% without (29% speedup).
- **Training-memory angle:** TP cuts the weight+activation cost within a layer by 1/t — but only the linear-layer activations; SP removes the hidden replica of the normalization activations, delivering true t× activation savings across the entire layer.

## Citation
Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. https://arxiv.org/abs/1909.08053

Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., & Catanzaro, B. (2022). Reducing Activation Recomputation in Large Transformer Models. MLSys 2023. https://arxiv.org/abs/2205.05198
