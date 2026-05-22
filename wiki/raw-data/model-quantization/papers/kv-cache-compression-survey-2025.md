<!-- scope: 2025 survey of KV-cache compression (quantization + sparsity + eviction)
     deps: [[kivi]], [[kvquant]]
     see-also: [[coupled-quant-eviction]], [[gear]], [[skvq]]
-->

# KV-Cache Compression Survey (2025)
- **Core Insight:** KV-cache compression has fractured into three orthogonal axes — quantization (KIVI / KVQuant), sparsity / eviction (H2O / StreamingLLM / Quest), and reordering / merging (KV-Sharing, Cross-Layer KV) — and the 2025 surveys converge on the same operational rule: pick one technique from each axis and compose, rather than pick the single best technique on each axis alone.
- **Guideline:** For long-context serving, baseline = INT8 KV quant + sliding-window eviction + cross-layer KV sharing; only escalate to ≤ 4-bit KV when the savings actually matter and a quant-aware eviction policy (coupled quant + eviction) is available.
- **Authors:** various survey authors (Park 2024/2025, others)
- **Year:** 2024-2025
- **URL:** representative: arxiv KV-cache compression surveys, late 2024 / 2025
- **Relevant topics:** KV-cache quantization, eviction, sliding window, cross-layer sharing, long-context serving

## Abstract
At 128K-1M-token context, the KV cache is the dominant memory footprint and bandwidth cost of LLM serving. 2024-2025 has produced three roughly orthogonal lines of compression: (1) **quantization** — KIVI's per-channel-K + per-token-V INT2-4 path, KVQuant's non-uniform NUF4 with dense+sparse outliers, GEAR's quant + error-compensation low-rank residual; (2) **eviction / sparsity** — H2O's attention-score-based heavy-hitter eviction, StreamingLLM's sink+window pattern, Quest's query-aware top-k page selection; (3) **structural** — cross-layer KV sharing (only every k layers cache K/V), grouped query attention as a KV-reduction technique, KV-merging at sequence-level. The survey takeaway: these axes are largely *additive* in compression ratio but *interactive* in quality — naive composition (quant + evict + share) is brittle, but quant-aware eviction policies (only evict tokens whose quant error is small) preserve quality.

## Key Contributions
- **Three-axis taxonomy**: quantization × sparsity/eviction × structural — clarifies that prior papers were optimizing along one axis at a time.
- **Compression-vs-quality Pareto frontier**: shows that the best 2025 systems (e.g. Quest + KIVI + cross-layer) get to ~ 8-16× KV reduction at < 1 % quality loss on long-context benchmarks (NIAH, RULER, LongBench).
- **Quantization granularity guidance**: K and V have very different statistics; K is channel-heterogeneous (one channel per head can dominate) → per-channel K quant; V is token-level smooth → per-token V quant works.
- **Coupling guidance**: combine quant + evict by evicting tokens whose quantized representation is closest to a heavy-hitter (so eviction error and quant error don't reinforce).
- **Long-context evaluation harness**: surveys consistently use RULER / NIAH for KV-compression eval rather than perplexity, which doesn't surface long-context degradation.

## Key Figures/Tables to Study
- The three-axis taxonomy figure (typical of these surveys).
- The Pareto plot: compression ratio (x) vs RULER score (y) for the best method per axis vs the best compositions.
- The K vs V statistics histogram: per-channel range for K, per-token range for V — the asymmetry that justifies asymmetric quant treatment.
- The "quant-aware eviction" diagram showing the interaction between eviction and quantization noise.

## Technical Details

### Quantization axis
- **KIVI** (Liu 2024): per-channel scale for K (2 bits), per-token scale for V (2 bits); W16A16KV2 viable on Llama-2.
- **KVQuant** (Hooper 2024): non-uniform 4-bit quant (NUF4) for K, per-token V; outlier sparse path.
- **GEAR** (Kang 2024): quantize K/V to W4, store a low-rank residual to compensate.
- **SKVQ** (Duanmu 2024): sliding-window quant — older tokens at lower bits, recent tokens at higher bits.

### Sparsity / eviction axis
- **H2O** (Zhang 2023): "heavy hitter oracle" — keep tokens with high cumulative attention score.
- **StreamingLLM** (Xiao 2023): sink tokens (first 4) + sliding window — works without retraining.
- **Quest** (Tang 2024): query-aware page-level top-k retrieval from the KV cache; reads only the pages relevant to the current query.

### Structural axis
- **Cross-layer KV (CLA, YOCO)**: only every k-th layer computes K/V; other layers reuse.
- **GQA / MQA**: group-query / multi-query attention — fewer KV heads.
- **Snapkv / PyramidKV**: layer-wise budget allocation, more KV in early layers, less in late.

### Composition guidance
- **Quant + structural**: orthogonal; cross-layer + INT4 KV = ~ 16× reduction with little interaction.
- **Quant + eviction**: interactive; naive composition can compound errors. The 2025 fix: evict tokens whose quant error is small (sacrifice precision-redundant tokens, keep precision-critical ones).
- **Eviction + structural**: orthogonal in principle; in practice layer-wise eviction budgets interact with cross-layer sharing (skipped layers can't be evicted from).

### Long-context eval
- Perplexity is dominated by near-context tokens → doesn't surface long-context KV compression errors.
- NIAH (Needle In A Haystack), RULER, LongBench are the standard suite; show 2-bit KV without quant-aware eviction can lose 20+ points where perplexity moves by < 0.1.

## Connections
- [[kivi]] — per-channel K + per-token V quant; survey's quant-axis primary.
- [[kvquant]] — non-uniform KV quant + sparse outliers.
- [[gear]] — quant + low-rank residual compensation.
- [[skvq]] — sliding-window quant; bridges quant and eviction.
- [[coupled-quant-eviction]] — companion 2025 work on quant-aware eviction policies.
- [[wkvquant]] — joint W4 + KV4 calibration; cross-axis composition.
- [[qaq]] — quality-adaptive KV-cache quantization.
- [[per-channel-vs-per-token-kv]] — analytical study of the K vs V quant asymmetry.
