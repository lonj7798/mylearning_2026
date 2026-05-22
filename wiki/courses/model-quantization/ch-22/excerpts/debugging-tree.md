---
chapter: ch-22
course: model-quantization
phase: read
excerpt_of: "Debugging tree for 'my reproduction numbers do not match the paper'"
created_at: "2026-05-21"
---

# Excerpt: Reproduction-Gap Debugging Tree

**Sources:** distilled from [[raw-data/kivi]], [[raw-data/kvquant]], [[raw-data/gear]], [[raw-data/turboquant]], [[raw-data/survey-low-bit-llm-2024]] reproducibility notes

---

## The diagnostic tree

Read top-to-bottom; stop at the first match. Every branch ends with a concrete action.

```
Gap observed: my <metric> differs from paper by >2%
├── Is the gap on PPL?
│   ├── YES → § branch A
│   └── NO  → §branch B
├── Is the gap on long-context (NIAH/RULER/LongBench)?
│   ├── YES → §branch C
│   └── NO  → next branch
├── Is the gap on throughput / VRAM?
│   ├── YES → §branch D
│   └── NO  → next branch
├── Is the gap on a knowledge benchmark (MMLU/GSM8K)?
│   └── YES → §branch E
```

---

## Branch A — PPL gap

**A1. Calibration set mismatch.** Most common cause. The paper says "128 sequences from Pile"; you used 128 from C4. PPL difference can be 0.05–0.15. Action: re-run with the paper's exact calibration corpus + size + seed. If the paper doesn't specify the seed (most don't), try 3 seeds and report the median.

**A2. Tokenizer version.** Subtle. HF transformers tokenizer updates can change BPE merges, which changes both calibration tokenization and eval tokenization. Action: pin the tokenizer to the model's published version (`tokenizer.from_pretrained(MODEL, revision=<exact_sha>)`).

**A3. Model checkpoint version.** Some quantization papers reproduce on a specific model checkpoint that has since been re-released. Action: pin the model to its exact HF revision SHA used in the paper (often in the paper's appendix or supplementary).

**A4. Mixed-precision compute dtype.** Paper uses BF16 compute on Hopper; you used FP16 on Ampere. Activation magnitudes differ, and per-channel scales (KIVI/AWQ/etc.) computed from those magnitudes differ. Action: match the paper's compute dtype, or run both and report.

**A5. Implementation bug.** Run on the same model the paper used (e.g., Llama-2-7B if the paper used it). If gap closes, the issue is model-specific (move to E1/E2). If gap persists on the paper's exact model, the bug is in your implementation. Action: line-by-line compare your math to the paper's equations.

**A6. Numerical-fusion mismatch.** Paper uses a custom CUDA kernel with fused dequant + GEMM in FP32 partial-sum; your pure-PyTorch reference accumulates in BF16 with separate dequant. Gap can be 0.05–0.2 PPL. Action: explicitly cast partial sums to FP32 in your reference path; if you have a custom kernel, verify the partial-sum precision matches the paper's.

---

## Branch B — Other task-eval gap (not PPL, not long-context)

**B1. Eval harness version.** lm-eval-harness has changed normalization between versions; MMLU 5-shot accuracy can differ by 1–2 pp across versions for the same checkpoint. Action: pin to the exact lm-eval-harness commit the paper used (often documented in their `evaluate.sh`).

**B2. Few-shot prompt template.** MMLU 5-shot accuracy depends on prompt template (`{question}\nA.\nB.\nC.\nD.\nAnswer:` vs `Question: {question}\nA. ...`). Action: verify the template matches; check both lm-eval-harness's templates and the paper's stated template.

**B3. Generation parameters.** Some evals require greedy decoding (temperature=0); others sample. GSM8K accuracy can differ by 5+ pp between greedy and temperature=0.7. Action: pin generation parameters.

**B4. Stop-token handling.** GSM8K answer extraction is regex-dependent on the format the model emits. A quantized model may emit slightly different formatting that breaks the answer extractor. Action: read the first 10 model outputs; check if the format is what your extractor expects.

---

## Branch C — Long-context gap (NIAH / RULER / LongBench)

**C1. Position encoding handling.** RoPE-based models have multiple position-encoding variants (theta, scaling, base). KV-quant interaction with each is different. Action: verify your model's RoPE config matches the paper's; check that scaling (NTK / YaRN / linear) is correctly applied through the quantized KV path.

**C2. Attention masking under truncation.** Some implementations apply a sliding-window attention or position interpolation that changes the long-context behavior. Action: read the attention forward path; verify no implicit windowing is engaged at long context.

**C3. KV cache layout mismatch.** PagedAttention block size, the paper's chunk-quantization group size (e.g., KIVI g=32), and the attention kernel's tile size may not align. Misalignment causes per-block scale staleness or boundary artifacts. Action: log effective block size at each level and verify alignment; pay the bookkeeping cost if needed.

**C4. GQA / MHA difference.** Llama-3 uses GQA (grouped-query attention); Llama-2 is MHA. KV-quant methods designed for MHA may break on GQA because per-channel K outlier concentration is higher when K heads serve multiple attention heads. Action: run on the paper's exact attention architecture as a control. If the gap closes on MHA but persists on GQA, the gap is real and worth documenting in the memo §5.

**C5. Needle placement / extraction.** NIAH is sensitive to needle phrasing and extraction regex. A different needle than the paper used (or a different extractor) can shift accuracy by 5–10 pp. Action: use the canonical needle from the gkamradt repo unless the paper specifies otherwise.

---

## Branch D — Throughput / VRAM gap

**D1. Hardware difference.** Paper measured on H100 SXM 80GB; you have H100 PCIe 80GB. Memory bandwidth differs (3 TB/s vs 2 TB/s); decode throughput will differ proportionally. Action: report your hardware exactly; do not claim throughput parity if hardware differs.

**D2. Kernel choice.** Paper uses Marlin; you fell back to AutoGPTQ's CUDA kernel. Throughput can differ 1.5–3×. Action: confirm vLLM selected the correct kernel (check the load-time log line per [[ch-21]] excerpts/serving-stack); explicitly pin via `quantization=gptq_marlin`.

**D3. Native vs emulated format.** Paper uses native NVFP4 on Blackwell; you emulated on Ampere with FP8 fallback. Throughput is not comparable. Action: state this in memo §9 (limitations); only claim quality reproduction, not throughput.

**D4. Batch size or context length.** Paper measured at batch=1; you measured at batch=16. Compute-bound vs memory-bound regime differs; quantization speedup varies by batch. Action: match the paper's batch + context, or report at multiple operating points.

**D5. Warm-up insufficient.** First few iterations include CUDA kernel autotune, weight pre-shuffle (Marlin), and other one-time costs. Throughput numbers reported without warm-up are off by 10–30%. Action: warm-up ≥2 iterations, measure ≥10 iterations, report mean ± stdev.

---

## Branch E — Knowledge benchmark gap (MMLU / GSM8K)

**E1. Calibration-source distribution shift.** Paper calibrated on Wikitext-2; you calibrated on C4. MMLU accuracy can shift by 0.5–2 pp depending on this choice ([[ch-20]] §5.2). Action: re-run with the paper's calibration corpus.

**E2. Model is instruction-tuned but quantized as base.** If the paper used the base model and you used the chat model (or vice versa), MMLU and GSM8K behaviors differ substantially. Action: verify model variant matches.

**E3. Per-layer exception policy.** Aggressive methods (NVFP4, sub-2-bit) have layer exception policies (per [[fp4-inference-diagnosis]]). Paper kept embedding + head + early blocks in BF16; you quantized everything. MMLU drops disproportionately because the LM head's logit precision matters for the multiple-choice answer extraction. Action: implement the paper's exception policy; document any deviation.

**E4. KV-quant impact on chain-of-thought.** GSM8K 8-shot CoT exercises a long context (~3K tokens) full of arithmetic. KV-quant noise compounds across the CoT chain; GSM8K can drop 5+ pp where MMLU drops <1 pp. Action: if this is the gap, it's an interesting reproducible finding — KV-quant's GSM8K cost is often understated in papers. Document it.

---

## The meta-pattern

Every gap in this tree has the same structural form:

```
"Paper claims X under conditions (C_paper)"
"I observe Y under conditions (C_mine)"
"C_paper - C_mine reveals: <variable Z>"
"Test by isolating Z: re-run with one variable changed at a time"
"Gap closes → Z is the cause; gap persists → continue down the tree"
```

This is the same Karpathy "one change, one prediction, one outcome" discipline from [[karpathy-training-neural-net-recipe]] applied to reproduction studies. Internalise it; the same pattern serves every empirical project.

---

## When you cannot close the gap

Three healthy responses:

1. **Document the gap with evidence.** "I cannot reproduce X; here is what I tried" is a publishable contribution if the diagnostic walk-through is solid.
2. **Report the conditional reproduction.** "Method X reproduces on model A but breaks on model B for reason Z" is often more useful to the field than confirming the paper.
3. **Email the authors.** Most quantization-paper authors are responsive. A specific, code-citing question often gets a useful response. Document the response in the memo if you go this route.

Three unhealthy responses (avoid):

1. **Quietly drop the gap from your report.** Don't.
2. **Tune hyperparameters until the gap closes.** This is HARKing and contaminates the reproduction.
3. **Conclude "the paper is wrong" without strong evidence.** Strong claims need strong evidence; "I couldn't reproduce" is not strong evidence by itself.

---

## Connections

- [[ch-22]] §step-5-iterate — chapter section.
- [[ch-22]] §the-reproduction-workflow — broader workflow.
- [[ch-20]] §5 — calibration-set discipline this tree assumes.
- [[karpathy-training-neural-net-recipe]] — the "one change, one prediction" pattern.
- [[ch-22]] §what-reproduction-means — the framing that makes documenting the gap a legitimate outcome.
