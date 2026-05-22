---
chapter: ch-22
course: model-quantization
phase: read
excerpt_of: "Pinned long-context evaluation harness for the ch-22 capstone"
created_at: "2026-05-21"
---

# Excerpt: Long-Context Harness for the Capstone

**Sources:** [[raw-data/kivi]], [[raw-data/kvquant]], [[raw-data/turboquant]], standard NIAH/RULER/LongBench tooling

---

## What you must run

Per [[ch-20]] §4, KV-quant reproductions are not credible without long-context evals. The capstone requires *all three* of the following:

1. **NIAH heatmap** at deployment context lengths × 11 depth points.
2. **RULER 4-task subset** at the longest deployment context.
3. **LongBench per-category average** at intermediate context.

Plus the [[ch-20]] §6 short-context battery (PPL + MMLU + GSM8K) for sanity.

---

## NIAH

Source: https://github.com/gkamradt/LLMTest_NeedleInAHaystack

### Setup

```python
# Paul Graham essays as the haystack
HAYSTACK_PATH = "paulgrahamessays/essays_concatenated.txt"
NEEDLE = "The magic password is 7392-PRIME-MOUNTAIN."
QUESTION = "What is the magic password?"

# Sweep: 11 depths × 6 context lengths × 1 needle = 66 runs per method
DEPTHS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

def make_prompt(haystack, needle, depth_frac, context_len, tokenizer):
    # Build context of exactly context_len tokens
    haystack_tokens = tokenizer.encode(haystack)
    if len(haystack_tokens) < context_len:
        # Repeat haystack to reach context_len
        haystack_tokens = (haystack_tokens * (context_len // len(haystack_tokens) + 1))[:context_len]
    else:
        haystack_tokens = haystack_tokens[:context_len]
    # Insert needle at depth_frac
    insert_pos = int(depth_frac * context_len)
    needle_tokens = tokenizer.encode(needle)
    haystack_tokens = haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:]
    # Re-trim
    haystack_tokens = haystack_tokens[:context_len]
    prompt_tokens = haystack_tokens + tokenizer.encode("\n\nQuestion: " + QUESTION + "\nAnswer:")
    return prompt_tokens
```

### Scoring

Binary: did the model emit "7392" (or "7392-PRIME-MOUNTAIN") anywhere in the first 50 generated tokens? Aggregate across (depth, context_len) for the heatmap.

### What good looks like

FP16 baseline at 32K context: ≥95% accuracy across all depths.
KIVI INT2 KV at 32K: paper reports ≥95%; a working reproduction should match within ±2 pp.
Naive per-token INT2 KV at 32K: <40% accuracy in the middle 30–60% depth range — the "lost in the middle" pathology.

---

## RULER

Source: https://github.com/NVIDIA/RULER

### Subset for the capstone

13 task variants is more than the capstone needs. Run 4:

1. **single-needle (multi-key)** — variant of NIAH with multiple keys; tests retrieval discrimination.
2. **multi-needle** — multiple needles; tests retrieval with multiple targets.
3. **multi-hop** — find fact A, use A to find fact B; tests retrieval chaining (KV-quant most sensitive).
4. **aggregation (common-words)** — count or aggregate across the context; tests V-cache fidelity.

Run all 4 at 32K context (16K for the resource-constrained path).

### Scoring

Per-task accuracy; report per-task + average. The multi-hop track is the KV-quant canary — degradation here predicts deployment failure on real chained-reasoning prompts.

---

## LongBench

Source: https://github.com/THUDM/LongBench

### Per-category averages

LongBench has 6 categories:

| Category | Tasks | What it tests |
|----------|-------|--------------|
| Single-doc QA | NarrativeQA, Qasper, MultiFieldQA | comprehension over a single long document |
| Multi-doc QA | HotpotQA, 2WikiMultihopQA, Musique | multi-source synthesis |
| Summarization | GovReport, QMSum, MultiNews | abstraction over long input |
| Few-shot learning | TREC, TriviaQA, SAMSum | in-context learning over long shots |
| Synthetic | PassageCount, PassageRetrieval | controlled long-context retrieval |
| Code completion | RepoBench-P, LCC | long-context code understanding |

Report per-category averages + overall average. KV-quant impact varies by category:
- Most sensitive: Multi-doc QA (compounding retrieval errors).
- Less sensitive: Summarization (averaged degradation).
- Mixed: Code completion (depends on KV outlier patterns in code tokens).

---

## Pinned run protocol

```bash
# Step 1: prepare base model
python prepare_model.py --model meta-llama/Llama-3-8B --output checkpoints/llama-3-8b-fp16

# Step 2: apply quantization (your implementation)
python apply_quantization.py --base checkpoints/llama-3-8b-fp16 \
    --method kivi --bits 2 --group-size 32 --output checkpoints/llama-3-8b-kivi-int2

# Step 3: run short-context evals
python evaluate_short.py --checkpoint checkpoints/llama-3-8b-kivi-int2 \
    --tasks wikitext c4 mmlu gsm8k --seeds 42,43,44,45,46

# Step 4: run NIAH heatmap
python evaluate_niah.py --checkpoint checkpoints/llama-3-8b-kivi-int2 \
    --context-lengths 4096,8192,16384,32768,65536 \
    --depths 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --output results/niah_kivi_int2.json

# Step 5: run RULER subset
python evaluate_ruler.py --checkpoint checkpoints/llama-3-8b-kivi-int2 \
    --tasks single_needle_multi_key,multi_needle,multi_hop,common_words_aggregation \
    --context-length 32768 --output results/ruler_kivi_int2.json

# Step 6: run LongBench
python evaluate_longbench.py --checkpoint checkpoints/llama-3-8b-kivi-int2 \
    --context-length 16384 --output results/longbench_kivi_int2.json

# Step 7: also run on FP16 baseline (same commands, --base checkpoint)
# Step 8: compose memo from results/*.json
```

All scripts must accept `--checkpoint` and `--seed` and write JSON with metric + value + context_length + depth (where applicable). The harness is reproducible from the JSON outputs.

---

## Compute budget reality check

For Llama-3-8B on 1 × H100:

| Eval | Per method | All 4 methods (FP16 + KIVI + comparison + ablation) |
|------|-----------|-----------------------------------------------------|
| Short-context (PPL + MMLU + GSM8K) | ~30 min | ~2 hr |
| NIAH heatmap × 66 points | ~3 hr | ~12 hr |
| RULER 4-task × 32K | ~2 hr | ~8 hr |
| LongBench × 16K | ~4 hr | ~16 hr |
| Total | ~9.5 hr | ~38 hr |

If you have only one method to evaluate (your reproduction + FP16 baseline), budget ~20 hours of H100 time for the full eval pass. If you also run a comparison method, double it.

For the resource-constrained path (TinyLlama on 1 × 4090 with 4K/8K contexts only), the eval budget drops to ~6 hours per method.

---

## What not to do

- Skip NIAH/RULER and just report PPL. KV-quant failure modes are invisible in PPL on short corpora ([[ch-20]] §1). Reviewers and your future self will not trust a KV-quant reproduction without long-context evals.
- Report only context-length-aggregated NIAH accuracy. The heatmap matters; "lost in the middle" is a *depth-position* phenomenon, not a context-length phenomenon.
- Use a different needle phrasing between baseline and quantized runs. The needle is part of the test; varying it conflates retrieval failures with prompt-template variance.

---

## Connections

- [[ch-22]] §long-context-eval-mandatory — chapter section.
- [[ch-20]] §4 — methodology source for these benchmarks.
- [[kivi]] / [[kvquant]] / [[turboquant]] — papers whose long-context numbers this harness reproduces.
- [[long-context-eval]] (ch-20 excerpt) — the failure-mode taxonomy this harness is designed to detect.
