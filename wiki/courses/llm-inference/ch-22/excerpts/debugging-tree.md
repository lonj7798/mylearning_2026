---
chapter: ch-22
course: llm-inference
phase: read
excerpt_of: "Diagnostic tree for 'my reproduction numbers don't match the paper'"
source_url: internal
created_at: "2026-05-21"
---

# Excerpt: The "numbers don't match" debugging tree

**Source:** distilled from common reproduction failures across the five method options
**Raw-data:** synthesis of [[raw-data/pagedattention]], [[raw-data/sglang-radixattention]], [[raw-data/speculative-decoding]], [[raw-data/eagle]], [[raw-data/distserve]]

---

## When to consult this tree

Symptoms:
- Headline number is outside the tolerance from [[excerpts/reproduction-memo]].
- Correctness tests pass but speed/utilisation is weak.
- Correctness tests fail.

Walk the tree top-down. Each branch has a specific test to run; don't skip.

---

## 0. Did correctness pass?

Run the per-method correctness check from [[excerpts/pagedattention-reference]] (lossless generation), [[excerpts/specdec-acceptance]] (lossless distribution), or the analogous test for your method.

- **Yes →** go to §1 (performance debugging).
- **No →** stop. There is no point benchmarking a buggy implementation. Walk §A (correctness debugging) first.

---

## §A — Correctness failed

### A.1 — Off-by-one in cache writes / reads

**Symptom.** Paged or radix cache: generations are gibberish or differ from baseline starting at token 1 or 2.

**Test.** Generate one token with your implementation, log the K/V values written to the cache and read on the next step. Compare to the baseline model's `past_key_values` at the same step.

**Common cause.** Filled-token counter incremented at the wrong layer (PagedAttention) or wrong moment in the layer loop.

### A.2 — Acceptance math wrong (SpecDec / EAGLE)

**Symptom.** Lossless test fails at T=0: SpecDec generates different tokens than target-only.

**Test.** Print `p[x] / q[x]`, `u`, and the accept/reject decision for the first 10 positions. Verify rejected positions sample from the **normalised** positive residual.

**Common causes** (in order of frequency):
1. Compared `p_logit > q_logit` instead of using probabilities + ratio.
2. Forgot to re-normalise `max(0, p-q)` before sampling.
3. Sampled `u` once and reused for multiple positions.
4. Took the bonus token from the wrong position of the target output.

See [[excerpts/specdec-acceptance]] for the worked-correct version.

### A.3 — Tree corruption (RadixAttention)

**Symptom.** First request works; second request with similar prefix crashes or returns garbage.

**Test.** Run the prefix-match test from [[ch-22]] §6: insert 100 random token strings into your tree, then query each one. All should return their full-length match.

**Common cause.** `split()` operation when inserting a string that shares a prefix-of-prefix with an existing node — the split must preserve KV pointers on both sides.

### A.4 — KV transfer corruption (DistServe)

**Symptom.** Single-request reproduction generates different output than co-located baseline.

**Test.** Print the first 10 K and V values transferred from prefill to decode, on both sides. They should be bit-identical.

**Common cause.** Float dtype mismatch (one side BF16, other FP16), or layer ordering mismatch (transferring layer-major when receiver expects head-major).

---

## §1 — Correctness passed but the number is weak

### 1.1 — Did you use the same hyperparameters as the paper?

**Test.** Re-read the paper's experimental setup section. List every hyperparameter the paper specifies (block size, chunk size, K, temperature, draft model choice, ...). Verify each one in your config.

**Common omissions.** Paper specifies `BLOCK_SIZE = 16`; your code uses 32. Paper specifies `K = 4`; your code uses 2 (default is "obviously" smaller). Paper specifies `T = 0`; your code uses 0.7 (chat default).

The default in your framework is rarely the value the paper used. Always pin.

### 1.2 — Is your hardware comparable?

**Test.** Look up paper's hardware: A100 / H100 / V100. Compare HBM bandwidth and FLOPs.

| GPU | HBM bandwidth | FP16 TFLOPs | Note |
|-----|--------------:|------------:|------|
| V100 | 0.9 TB/s | 125 | 2017 |
| A100 | 2.0 TB/s | 312 | 2020; common paper baseline |
| H100 | 3.4 TB/s | 989 (Tensor) | 2022; ~2× A100 bandwidth, ~3× compute |
| RTX 4090 | 1.0 TB/s | 165 | 2022; consumer |
| RTX 3090 | 0.94 TB/s | 142 | 2020; consumer |

**Implication.** PagedAttention's memory metric is hardware-independent. *Speed* metrics (SpecDec speedup, RadixAttention throughput) depend on the bandwidth/compute ratio. If the paper measured on A100 and you measure on H100, your speedup numbers may be different because the underlying decode is already faster.

For the memo, **always report your hardware and acknowledge the discrepancy**.

### 1.3 — Is your model architecture comparable?

**Test.** Compare attention head counts, KV head counts, hidden dim, layer count between your model and the paper's.

**Common mismatch.** EAGLE paper uses Llama-2-7B (full MHA, 32 KV heads). You use Llama-3-8B (GQA-8, 8 KV heads). Decode-time KV-cache bandwidth cost is 4× lower on GQA-8 — speculative methods get less to amortise → less speedup.

Confirmatory experiment: re-run with the paper's exact model. If your number matches there, the gap is architectural. (This is exactly the worked example in [[excerpts/reproduction-memo]].)

### 1.4 — Is your workload comparable?

**Test.** Compare paper's eval workload to yours. ShareGPT is *not* a fixed workload; different filters give different prompt/output length distributions.

For RadixAttention specifically: the paper's 5× headline is on **shared-prefix** workloads. ShareGPT has minimal prefix sharing — you should expect ~1.5× there, and only see 5× on synthetic-shared-prefix workloads.

### 1.5 — Quantization or precision mismatch

**Test.** Paper uses FP16; you use BF16. Or paper uses FP32 KV cache; you use FP16.

**Implication.** Usually negligible for speedup numbers, but can shift acceptance rates by 1–3 % for SpecDec/EAGLE.

### 1.6 — Did you measure correctly?

**Test.** Common measurement bugs:
- TPOT computed including TTFT (`total_time / total_tokens` instead of `(total - ttft) / (output_tokens - 1)`).
- Warm-up not discarded — first 50 requests pay JIT / CUDA-graph cost.
- Network latency in TTFT (running benchmark client on different host).
- Concurrent processes on the GPU (browser, nvidia-smi loops, ...).

Re-run the headline number with a clean process tree and the first 50 requests discarded.

---

## §2 — Number is *better* than the paper

Yes, sometimes this happens. Reasons to be suspicious:

1. **You're measuring the wrong thing.** Verify the metric matches the paper's exact definition.
2. **You're not pessimising the baseline correctly.** The paper compared to FasterTransformer / Orca / vLLM-without-the-feature. Your "baseline" might be a stronger or weaker stack.
3. **You're hardware-advantaged.** H100 + paper's A100 measurement can show better absolute numbers; the *ratio* should match.

Document the gap honestly in the memo. "I got 4× speedup vs the paper's 3×" is interesting if you can name why.

---

## §3 — Last resort: read the author's source

Reading the author's repo is allowed *for verification*, not for copying. After you have a failing reproduction and a hypothesis, look at the author's code to confirm or refute the hypothesis.

If you find your hypothesis was wrong and your code matches theirs but produces different numbers, the gap is genuinely on the hardware / workload / config axis — that's exactly the memo's "where my numbers diverge" section.

If you find your code was wrong, fix it and re-run.

If you find the author's code does something the paper doesn't describe (a hyperparameter sweep, a calibration step, a kernel tweak), document that omission in Section 5 ("what the paper didn't tell me"). This is one of the most valuable contributions a reproduction can make.

---

## Decision flowchart

```
Headline matches within tolerance?
├── Yes → Write memo (Section 4: which hyperparameter mattered most?)
└── No
    ├── Correctness fails → §A (correctness debugging)
    └── Correctness passes
        ├── Hyperparams match paper? → No → fix and retest
        ├── Hardware comparable? → No → test on equivalent hardware if possible
        ├── Model architecture comparable? → No → run confirmatory experiment on paper's model
        ├── Workload comparable? → No → reproduce paper's workload exactly
        ├── Measurement methodology correct? → No → re-measure with warmup + clean process
        └── Still unmatched → §3 (consult author source as verification only)
```

---

## Connections

- [[excerpts/reproduction-memo]] — the document this debugging activity ultimately feeds into.
- [[excerpts/method-comparison]] — risk profile per method (predicts which §A branches you'll hit).
- [[ch-21]] — the lab's failure-mode section uses the same diagnostic discipline at smaller scale.
