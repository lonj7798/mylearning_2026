<!-- scope: Ablation references comparing packed vs unpacked SFT; quality-neutral when masks are correct
     deps: [[sequence-packing]], [[loss-masking-prompt]]
     see-also: [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]]
-->

# Packed vs Unpacked SFT — Ablation Reference
- **Core Insight:** When attention masks and position IDs are reset per sub-sequence, packed and unpacked SFT produce the same loss trajectory and downstream metrics; when masks are missing, packing *hurts* because tokens attend across document boundaries.
- **Guideline:** Treat packing as a pure throughput optimization (~2–3× at short sequence lengths, diminishing at long); verify equivalence by matching training loss curves against a small unpacked run.
- **Authors:** (Krell 2021, plus 2023–2024 ablations in Tülu 3 paper, HF Alignment Handbook, Axolotl docs, Megatron-LM SFT blog)
- **Year:** 2021–2024
- **URL:** https://arxiv.org/abs/2107.02027 (original); https://arxiv.org/abs/2411.15124 (Tülu 3 includes packing ablation)
- **Relevant topics:** SFT throughput, attention masking correctness, packing equivalence

## Abstract
Sequence packing is a throughput optimization. Its quality effect depends entirely on whether the attention mask and position-ID reset are correctly implemented. Krell et al. 2021 proved equivalence in the BERT setting. Modern decoder-only SFT ablations (Tülu 3, HF handbook, Axolotl) confirm the same result: packed SFT with FlashAttention varlen matches unpacked SFT on MT-Bench, AlpacaEval, and IFEval, at 2–3× higher throughput for instruction-mixture data.

## Key Claims / Evidence
- **Krell 2021 (BERT phase 2):** 2× throughput, identical pretraining loss and GLUE scores.
- **Tülu 3 (Llama-3.1-8B SFT):** packed vs unpacked same MT-Bench within 0.05 pts; 2.5× throughput on short-document mixtures.
- **HF Alignment Handbook v1 (Mistral-7B Zephyr):** packed default, explicit note that disabling packing changes only throughput.
- **Axolotl `sample_packing: true` vs `false`:** community benchmarks show ≤ 0.5 pt difference when masks correct.

## Failure modes (when packing hurts)
1. **Missing block-diagonal mask** — tokens in sub-sequence 2 attend to sub-sequence 1 → cross-document leakage → subtle quality drop on multi-turn evals.
2. **Un-reset position IDs** — sub-sequence 2 sees positions L_1..L_1+L_2 instead of 0..L_2 → RoPE is effectively position-shifted.
3. **Label mask not re-applied per sub-sequence** — prompt tokens of sub-sequence 2 contribute to loss → looks like packed is worse, really a masking bug.
4. **Using flash_attn_func (dense) instead of flash_attn_varlen_func** — no mask → silent contamination.

## Diagnostic procedure
1. Train a 100-step unpacked baseline; record train loss curve and first-batch logits.
2. Train a 100-step packed run with identical data and seed; compare loss curves.
3. Differences > 0.01 nats at matching step indicate a mask/pos-ID bug, not a "packing hurts" phenomenon.

## Throughput model (rough)
```
speedup ≈ L_max / avg(L_i)
where L_i are sub-sequence lengths in the dataset.
```
For an SFT mixture with mean length 600 and L_max = 4096, expected speedup ≈ 6× (often realized as ~3× due to FlashAttention overhead and GPU memory bandwidth).

## Connections
- Mechanics: [[sequence-packing]].
- Per-sub-sequence masking: [[loss-masking-prompt]].
- Recipes using packing: [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]].
- Related memory optimization: [[fsdp-sft]].
