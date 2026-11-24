<!-- scope: DeepSeek Open Infra Week (Feb 2025) training-side repos — DualPipe (bidirectional pipeline schedule with forward/backward overlap and zero-bubble weight-gradient split), DeepEP (MoE expert-parallel dispatch/combine kernels, incl. backward and redundant-expert gradient reduction), profile-data (PyTorch Profiler traces of one overlapped forward/backward chunk pair); README and code at pinned commits
     deps: [[deepseek-v3]]
     see-also: [[deepseek-v3-recipe]], [[deepseek-v3.1]], [[deepseek-v3-2]], [[mixed-precision]], [[fsdp-sft]], [[verl-rollout]], [[thinkingmachines-defeating-nondeterminism]]
-->

# DeepSeek Open Infra: DualPipe, DeepEP, profile-data
- **Core Insight:** DualPipe feeds micro-batches from both ends of the pipeline, so each rank holds two model chunks and can run one forward and one backward chunk concurrently; this reduces the bubble from 1F1B's (PP−1)(F+B) to (PP/2−1)(F&B+B−3W) at the cost of 2× parameters and PP+1 activations per device (`DualPipe/README.md` L26–36).
- **Guideline:** If you train an MoE with expert parallelism, budget communication as a first-class cost: DeepEP's original README measured internode dispatch at 43–46 GB/s over ~50 GB/s RDMA NICs in the DeepSeek-V3 pretraining setting, and its example reserves a fixed SM count (`Buffer.set_num_sms(24)`) for communication kernels, which is compute the model does not get (`DeepEP` README @84d3d6f L15–24, L123).
- **Authors:** DualPipe: Jiashi Li, Chengqi Deng, Wenfeng Liang (`README.md` L55). DeepEP: Chenggang Zhao, Shangyan Zhou, Liyue Zhang, Chengqi Deng, Zhean Xu, Yuxuan Liu, Kuai Yu, Jiashi Li, Liang Zhao (citation block). profile-data: DeepSeek-AI.
- **Year:** 2025 (Open Infra Week, February 2025; DeepEP README first version committed 2025-02-25)
- **URL:** https://github.com/deepseek-ai/DualPipe/tree/030ce4325f ; https://github.com/deepseek-ai/DeepEP/tree/f6aa38fe06 (original README: /blob/84d3d6fdee65d9795bc3598f02c50afa54dccfb8/README.md) ; https://github.com/deepseek-ai/profile-data/tree/449602428a
- **Source type:** released code
- **Relevant topics:** pipeline parallelism, pipeline bubbles, zero-bubble backward split, computation–communication overlap, expert parallelism, all-to-all dispatch/combine, FP8 communication, redundant experts, profiling

## Summary
DualPipe is the pipeline schedule described in the DeepSeek-V3 report; the repo is a ~440-line PyTorch reference (`dualpipe/dualpipe.py`) plus a V-shape variant (DualPipeV). DeepEP provides the MoE all-to-all kernels; the current repo (V2.5) has removed the Feb-2025 V1 API and the NVSHMEM dependency (`README.md` L18). profile-data publishes Chrome-trace JSON files for one training chunk pair, prefill, and decode.

## Key Contributions
- Bubble/memory comparison of 1F1B, ZB1P, DualPipe, DualPipeV at equal PP stages (`DualPipe/README.md` L26–36).
- Zero-bubble split of backward into input-gradient (B) and weight-gradient (W) work, deferred through `WeightGradStore` (`dualpipe.py` L97, L223; `utils.py`).
- MoE dispatch/combine kernels whose backward passes reuse the opposite primitive (`DeepEP` README L208 and @84d3d6f L179).
- Redundant-expert primitives: `lb_prefetch_weights` and `lb_reduce_grads` (FP32 gradient accumulation over NVLink) (`DeepEP/README.md` L334–335; section L330–380).

## Key Figures/Tables to Study
- `DualPipe/README.md` bubble table and `images/dualpipe.png` (8 PP ranks, 20 micro-batches).
- `dualpipe.py` `step()` L294–440: the 8-phase schedule.
- DeepEP README @84d3d6f "Performance" tables (normal and low-latency kernels).
- `profile-data` `train.json` (open in chrome://tracing).

## Technical Details
**DualPipe constraints and schedule (`dualpipe.py`).** The module takes two stage modules per rank (`modules: Tuple[nn.Module, nn.Module]`, L14). Overlap is enabled only if both modules share a type that defines `overlapped_forward_backward` (L23); otherwise forward and backward chunks run sequentially (L125–128). Requirements (L332–333):
```python
assert num_ranks % 2 == 0
assert num_chunks > 0 and num_chunks % 2 == 0 and num_chunks >= num_ranks * 2, f"{num_chunks=}, {num_ranks=}"
```
`step()` runs eight phases named in comments: nF0, nF0F1, nB1W1F1 (zero bubble), the main nF0B1F1B0 loop, nB1F1B0, nB1B0, nWB0, nW (L358–425). In the main loop, the middle ranks do not overlap the first pair "to further reduce bubble size" (L384–386).

**Zero-bubble W split (`dualpipe.py` L87–119, L216–223).** During a backward chunk with `enable_zb=True`, `WeightGradStore.enabled` is set so weight-gradient closures are cached, then `WeightGradStore.flush()` queues them; `_weight_chunk()` later runs `WeightGradStore.pop()` in FIFO order. Input gradients are sent upstream first; weight gradients fill otherwise idle slots.

**User hook.** The README states real use requires a custom `overlapped_forward_backward` for the user's module (L47); the example's signature takes (module0, inputs0, criterion0, labels0, module1, loss1, outputs1, output_grads1) and its body is a placeholder that runs forward then backward (`examples/example_dualpipe.py` L55–80).

**DeepEP in training (README @84d3d6f, Feb 2025).** Test setting: H800 (~160 GB/s NVLink) with one CX7 400 Gb/s InfiniBand NIC per GPU (~50 GB/s), DeepSeek-V3/R1 pretraining configuration: 4096 tokens per batch, hidden 7168, top-4 groups, top-8 experts, FP8 dispatch, BF16 combine (L15).
| Type | Dispatch #EP | Bandwidth | Combine #EP | Bandwidth |
|---|---|---|---|---|
| Intranode | 8 | 153 GB/s (NVLink) | 8 | 158 GB/s (NVLink) |
| Internode | 16 | 43 GB/s (RDMA) | 16 | 43 GB/s (RDMA) |
| Internode | 32 | 44 GB/s (RDMA) | 32 | 47 GB/s (RDMA) |
| Internode | 64 | 46 GB/s (RDMA) | 64 | 45 GB/s (RDMA) |
The training example notes "the adaptive routing configuration of the network **must be off**" for normal kernels (L137), and "The backward process of MoE dispatch is actually a combine" (L179). The current README (V2.5) instead recommends enabling adaptive routing "under all network load conditions" (L467) and estimates SM counts analytically (`get_theoretical_num_sms`, L139, L153) instead of a fixed count.

**DeepEP redundant experts (current README L330–380).** `lb_prefetch_weights` pushes expert weights to redundant slots before expert compute; `lb_reduce_grads` reads redundant experts' FP32 gradients over NVLink and adds them into the original experts' gradients in place. Both are collective within an NVLink domain. The README credits MoonEP and UltraEP for the approach.

**profile-data training trace (`README.md`).** One forward and one backward chunk overlapped under DualPipe; each chunk contains 4 MoE layers; EP64, TP1, 4K sequence length, matching DeepSeek-V3 pretraining; PP communication excluded; routing simulated as perfectly balanced. Inference traces: prefill EP32 with two micro-batches; decode EP128, 128 requests per GPU, all-to-all not occupying SMs.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3 pretraining (as profiled) | not stated in repo | pretrain-stable | EP; TP; seq length | EP64; TP1; 4K | profile-data `README.md` "Training" | verified 2026-11-24 | trace only; no throughput number |
| DeepSeek-V3/R1 pretraining (DeepEP test) | not stated | pretrain-stable | tokens per batch; hidden; routing; precisions | 4096; 7168; top-4 groups, top-8 experts; FP8 dispatch, BF16 combine | DeepEP README @84d3d6f L15 | verified 2026-11-24 | bandwidth table L17–22 |
| DeepEP V1 example | — | pretrain-stable | SMs reserved for comm kernels | 24 (example value) | DeepEP README @84d3d6f L123 | verified 2026-11-24 | no ablation in README |
| DualPipe | — | pretrain-stable | memory cost | 2× parameters, PP+1 activations per device | DualPipe `README.md` L30 | verified 2026-11-24 | analytical, not measured |
| DualPipe | — | pretrain-stable | micro-batch constraint | even, ≥ 2 × PP ranks | `dualpipe.py` L333 | verified 2026-11-24 | — |

## Findings relevant to generality and negative feedback
- These repos contain no capability or loss measurements; they set throughput and parallel layout, which fix how many tokens fit a compute budget ([[deepseek-v3]] §3.2 gives the system-level description).
- The profile assumes balanced routing; real imbalance is what the redundant-expert primitives address (profile-data README; DeepEP README L330–380).

## Connections
- [[deepseek-v3]], [[deepseek-v3-recipe]]: DualPipe and the cross-node all-to-all are described in the report (§3.2).
- [[deepseek-v3.1]] / [[deepseek-v3-2]]: Keep Routing in RL reuses inference-time expert routes, a separate MoE consistency issue.
- [[mixed-precision]]: FP8 dispatch / BF16 combine is a communication-precision choice.

## Verification
- Fetched on 2026-11-24: raw README.md of DualPipe (HEAD 030ce4325f, 2025-12-25), DeepEP (HEAD f6aa38fe06, 2026-10-21; and commit 84d3d6fdee, 2025-02-25), profile-data (HEAD 449602428a, 2025-03-21); `dualpipe/dualpipe.py`, `dualpipe/utils.py`, `examples/example_dualpipe.py` at DualPipe HEAD. DeepEP kernel source and `dualpipev.py` were not read beyond assert/comment lines.
- Not reported by these sources: end-to-end training MFU or step time; measured DualPipe bubble; the trace JSONs were not opened.
