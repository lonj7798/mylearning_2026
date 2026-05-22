<!-- chapter: ch-21
     phase: lab-capstone
     kind: lab
     title: Lab — Deploy + Benchmark vLLM vs SGLang Head-to-Head
     deps: [ch-16, ch-17, ch-19]
     sources: [[vllm]], [[sglang]], [[vllm-benchmarks]], [[sglang-benchmarks]], [[ttft-tpot-itl]], [[sharegpt-workload]]
-->

# Chapter 21 — Lab: Deploy + Benchmark vLLM vs SGLang Head-to-Head

> **Lab objective.** Stand up vLLM and SGLang side-by-side on the *same* hardware, serving the *same* Llama-3-8B-Instruct (or Qwen-1.8B on the constrained path), and run a ShareGPT request-rate sweep from 1 → 32 req/s. For three saturation points (under-loaded / near-knee / past-knee), report TTFT, TPOT, throughput, and goodput@p99 for both stacks. Then run a *required ablation*: toggle prefix caching, sweep `max_num_batched_tokens`, and sweep chunked-prefill `chunk_size`. The deliverable is a one-page memo with the side-by-side table and a sentence answering "which framework wins for which workload, and what would flip that conclusion?"
>
> **Guideline.** Two paths are offered: full-budget (H100 + Llama-3-8B) and resource-constrained (one 24 GB GPU + Qwen-1.8B). The methodology is identical. The deliverable is the side-by-side table and the failure mode you found — not the absolute throughput number.

---

## Goal — three artifacts

1. **A repo.** `serving-lab/` with a `Makefile` that brings up vLLM + SGLang servers, runs the ShareGPT sweep against each, and dumps per-run JSON.
2. **A results table.** `results.json` with one row per `(framework, request_rate, ablation_cell)` × p50/p95/p99 of TTFT, TPOT, throughput, goodput.
3. **A memo.** `lab-memo.md` — one page. Side-by-side table at 3 saturation points + 1 ablation result + 1 failure mode + recommendation.

Reproducibility: pin the vLLM version, the SGLang version, the Llama-3-8B-Instruct commit hash, the ShareGPT JSON path + seed, and the GPU model. Without those pins the memo is gossip.

---

## Full-budget path

**Target hardware.** 1 × H100 80 GB (or A100 80 GB). One GPU is enough — both frameworks happily serve an 8B model on a single device.
**Model.** `meta-llama/Meta-Llama-3-8B-Instruct`, BF16.
**Workload.** ShareGPT-v3 cleaned trace, ≥ 5000 requests, replayed at request rates {1, 2, 4, 8, 12, 16, 24, 32} req/s under open-loop Poisson arrival.
**Wall-clock budget.** ~6 hours end-to-end.
**SLOs (declare upfront).** TTFT p99 ≤ 2 000 ms, TPOT p99 ≤ 80 ms/token.

### Step 1 — install + sanity check

```bash
# fresh venv
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# vLLM
pip install "vllm==0.6.4"

# SGLang (in a separate venv to avoid CUDA-kernel conflict)
deactivate
python -m venv .venv-sgl && source .venv-sgl/bin/activate
pip install "sglang[all]==0.3.7"
```

Sanity check each:

```bash
# vLLM smoke test
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8001 --dtype bfloat16 &
sleep 60
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Meta-Llama-3-8B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
pkill -f "vllm serve"

# SGLang smoke test
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 8002 --dtype bfloat16 &
sleep 60
curl http://localhost:8002/v1/chat/completions ...   # same payload, port 8002
pkill -f "sglang.launch_server"
```

If either smoke test fails, fix that before touching benchmarks. Common cause: HF auth (`huggingface-cli login`).

### Step 2 — baseline server launches

Pin the configs so the comparison is meaningful. Both servers should have the same memory utilisation target, the same max-context, the same dtype.

**vLLM baseline (port 8001).**
```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

**SGLang baseline (port 8002).**
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8002 \
  --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --context-length 8192 \
  --max-running-requests 256 \
  --chunked-prefill-size 8192
  # RadixAttention is on by default; toggle with --disable-radix-cache
```

Notes on the knobs (see [[ch-16]], [[ch-17]] for the internals):
- `gpu-memory-utilization` / `mem-fraction-static` carve out the KV-cache pool. Match them so both stacks have equal cache budget.
- `max-num-batched-tokens` (vLLM) and `chunked-prefill-size` (SGLang) cap the per-step token budget. Match them so each stack is doing equivalent batch work per iteration.
- Prefix caching is on by default in both. For the ablation we toggle it.

### Step 3 — fetch the ShareGPT trace

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
  -O sharegpt.json
```

Filter to requests with reasonable prompt/output token counts (the benchmark scripts do this automatically with `--sharegpt-output-len`, but you can pre-filter for speed).

### Step 4 — run the ShareGPT sweep against vLLM

The vLLM repo ships `benchmarks/benchmark_serving.py`. It supports both vLLM and any OpenAI-compatible server, so we use it for both stacks.

```bash
# Sweep request rates: 1, 2, 4, 8, 12, 16, 24, 32 req/s
for rate in 1 2 4 8 12 16 24 32; do
  python benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://localhost:8001 \
    --endpoint /v1/chat/completions \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --num-prompts 1000 \
    --request-rate ${rate} \
    --seed 42 \
    --save-result \
    --result-dir results/vllm \
    --result-filename "vllm_rate${rate}.json"
done
```

What each output JSON contains:
- `request_throughput` (req/s actually achieved)
- `output_token_throughput` (tokens/s)
- `mean_ttft_ms`, `median_ttft_ms`, `p99_ttft_ms`
- `mean_tpot_ms`, `median_tpot_ms`, `p99_tpot_ms`
- `mean_itl_ms`, `p99_itl_ms`
- per-request raw data for goodput computation

### Step 5 — run the ShareGPT sweep against SGLang

The same script — just change the URL:

```bash
for rate in 1 2 4 8 12 16 24 32; do
  python benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://localhost:8002 \
    --endpoint /v1/chat/completions \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --num-prompts 1000 \
    --request-rate ${rate} \
    --seed 42 \
    --save-result \
    --result-dir results/sglang \
    --result-filename "sglang_rate${rate}.json"
done
```

Alternatively, SGLang ships its own benchmark in `python/sglang/bench_serving.py`. Use it as a cross-check — the numbers should agree within a few percent.

### Step 6 — compute goodput@p99

Goodput is not directly emitted; compute it from the raw per-request data. Define an SLO predicate per [[ch-19]]:

```python
import json, glob

SLO_TTFT_MS = 2000
SLO_TPOT_MS = 80

def goodput(result_file):
    d = json.load(open(result_file))
    raw = d["raw_request_data"]   # list of per-request records
    success = sum(1 for r in raw
                  if r["ttft_ms"] <= SLO_TTFT_MS
                  and r["mean_tpot_ms"] <= SLO_TPOT_MS)
    duration = d["benchmark_duration"]
    return success / duration  # SLO-met requests per second

for path in glob.glob("results/vllm/*.json"):
    print(path, goodput(path))
```

For a stricter version, require p99 of *each* request's per-token gaps to clear the TPOT SLO — this catches requests that "averaged" the SLO but had nasty p99 jitter.

---

## Resource-constrained path

**Target hardware.** 1 × consumer GPU with ≥ 24 GB (RTX 3090, 4090, A5000, L4). Even 16 GB works at the cost of tighter context.
**Model.** `Qwen/Qwen1.5-1.8B-Chat` (or `Qwen/Qwen2.5-1.5B-Instruct` as a more recent substitute).
**Workload.** Same ShareGPT, but at smaller scale: rates {1, 2, 4, 8, 16} req/s, 500 prompts per run.
**Wall-clock budget.** ~3 hours.

The launch commands change only in the model path + lowered memory util:

```bash
# vLLM
vllm serve Qwen/Qwen1.5-1.8B-Chat \
  --port 8001 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --enable-prefix-caching

# SGLang
python -m sglang.launch_server \
  --model-path Qwen/Qwen1.5-1.8B-Chat \
  --port 8002 --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --context-length 4096 --max-running-requests 128 \
  --chunked-prefill-size 4096
```

The benchmark loop is identical except for the smaller prompt count and rate range.

**Why this path is still useful.** The absolute throughput numbers will be smaller, but the *shape* of the curves — where the knee is, how prefix caching shifts it, what chunked-prefill costs in TTFT — is the methodology you actually need. The lab deliverable is the methodology, not the absolute tokens/sec.

---

## Required ablation

Pick **all three** ablations if time permits; **at minimum** run ablation (a) for both frameworks. Each is a single-variable sweep on top of the baseline.

### (a) Prefix caching on / off

Stop the servers and relaunch with prefix caching disabled:

```bash
# vLLM — drop --enable-prefix-caching
vllm serve ... --max-num-batched-tokens 8192 --enable-chunked-prefill

# SGLang — add --disable-radix-cache
python -m sglang.launch_server ... --chunked-prefill-size 8192 --disable-radix-cache
```

Re-run the sweep. Expected result on ShareGPT (which has *some* prefix overlap from chat templates): TTFT improves 20–40 % when prefix caching is on; throughput improves 5–15 %. SGLang's RadixAttention typically extracts more from prefix overlap than vLLM's hash-block APC, especially on longer shared prefixes.

### (b) `max_num_batched_tokens` sweep

vLLM (matching `chunked-prefill-size` on SGLang). Sweep over {2048, 4096, 8192, 16384, 32768} at a fixed request rate near the saturation point (e.g., 16 req/s).

Expected shape: TPOT drops as the value grows (more decode tokens per step) but TTFT grows (longer prefill chunks). The crossover defines the throughput-vs-latency tradeoff knob. The ShareGPT "sweet spot" on H100 + 8B is usually 8 192 — but this is workload-dependent.

### (c) Chunked-prefill `chunk_size` sweep

In vLLM, `chunk_size` is implicit in `max_num_batched_tokens` (it caps how many prompt tokens go into one step). In SGLang, set `--chunked-prefill-size` to {512, 1024, 2048, 4096, 8192}.

Expected shape: small chunks → low TTFT for the first chunk, but more steps per prefill → higher total prefill latency. Large chunks → faster prefill but more decode-step starvation. The Sarathi-Serve sweet spot (see [[ch-05]]) is usually around 1 024–4 096 for chat-style prompts.

---

## Memo deliverable template

The output of this lab is the file `lab-memo.md`. Below is the template — fill the cells, don't restructure them.

```
# Serving Framework Head-to-Head — vLLM vs SGLang on Llama-3-8B

Hardware: 1 × H100 80 GB, CUDA 12.4
Model:    meta-llama/Meta-Llama-3-8B-Instruct (commit ...)
Workload: ShareGPT V3 cleaned, 1000 prompts/run, seed 42
SLO:      TTFT p99 ≤ 2000 ms, TPOT p99 ≤ 80 ms
vLLM:     0.6.4
SGLang:   0.3.7

## Side-by-side at three saturation points

|         | Rate   | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | Tput (tok/s) | Goodput@p99 (req/s) |
|---------|-------:|--------------:|--------------:|--------------:|--------------:|-------------:|--------------------:|
| vLLM    |   4    |   ___         |   ___         |   ___         |   ___         |   ___        |   ___               |
| SGLang  |   4    |   ___         |   ___         |   ___         |   ___         |   ___        |   ___               |
| vLLM    |  16    |   ___         |   ___         |   ___         |   ___         |   ___        |   ___               |
| SGLang  |  16    |   ___         |   ___         |   ___         |   ___         |   ___        |   ___               |
| vLLM    |  32    |   ___         |   ___         |   ___         |   ___         |   ___        |   ___               |
| SGLang  |  32    |   ___         |   ___         |   ___         |   ___         |   ___        |   ___               |

(Rows: under-loaded / near-knee / past-knee)

## Ablation result

(a) Prefix caching off vs on, at rate 16:
- vLLM:   TTFT p99 ____ → ____  (Δ ____ %)
- SGLang: TTFT p99 ____ → ____  (Δ ____ %)

## One failure mode I observed

(Specific: which framework, which workload, which knob, what happened. Not "SGLang was slower." Something like "vLLM's TTFT p99 degraded by 3× at rate 24 because chunked-prefill chunks competed with decode tokens for the 8192-token step budget; raising max_num_batched_tokens to 16384 fixed it.")

## Recommendation

- Workload type X (e.g., long shared system prompts): pick ____ because ____.
- Workload type Y (e.g., independent short chats): pick ____ because ____.
- Tipping point: ____ would flip the conclusion.
```

---

## Reflection prompts

The memo is the deliverable, but the *reflection* — the bit you can't bullshit — is what makes the lab worth doing.

1. **Where is the knee on each framework?** The request rate at which TTFT p99 crosses your SLO is the maximum capacity. How far apart are vLLM's and SGLang's knees?
2. **Which framework wins for high-prefix-overlap workloads?** RadixAttention is SGLang's headline feature. Does your ShareGPT trace exhibit enough overlap for it to matter? (Hint: ShareGPT chat templates are short; this is *not* the workload that maximally favours SGLang.)
3. **Which framework wins on near-overload behaviour?** Past the knee, one stack typically degrades more gracefully than the other. Which one, and why? (Hint: preemption policy + cache-hit dynamics.)
4. **What's the worst single config knob to mis-set?** If you had to give a junior engineer one warning before they deployed either stack, what would it be?
5. **Under what circumstance would your conclusion flip?** Longer context? Higher prefix sharing? Tighter TPOT SLO? Name the specific axis and the magnitude.

The answer-rubric is severity: a memo that says "vLLM was faster" learns nothing; one that says "vLLM beat SGLang by 14 % goodput at rate 24, but the gap inverts above rate 32 because SGLang's preemption policy is gentler under cache pressure" demonstrates fluency.

---

## What this lab is not

- **Not a paper.** You're not trying to publish a "vLLM vs SGLang" benchmark. Both projects move weekly; your numbers will be stale in a month. The transferable skill is *running the experiment correctly*, not the verdict.
- **Not a kernel benchmark.** Differences at the kernel level (FlashAttention 2 vs 3, Marlin vs cutlass) confound the framework comparison. Pin both stacks to use the same attention backend if you want a clean comparison; default backends are fine if you want the realistic-deployment comparison.
- **Not the capstone.** [[ch-22]] asks you to reproduce a *paper* end-to-end. This lab is the warm-up: a known harness, two known frameworks, three known ablations.

The point of the lab is *fluency*. After it, you should be able to: spin up either stack on a new model in <1 hour, run a ShareGPT sweep against it, and write a one-page summary that an SRE can act on.

---

## Connections

- **Back to [[ch-16]]** — vLLM's scheduler internals; explains why `max_num_batched_tokens` is the central knob.
- **Back to [[ch-17]]** — SGLang's RadixAttention; the reason ablation (a) usually moves more in SGLang than in vLLM.
- **Back to [[ch-19]]** — TTFT/TPOT/ITL definitions + the goodput-under-SLO methodology this lab applies.
- **Forward to [[ch-22]]** — the capstone uses one of these frameworks as the host for a reproduced paper.

## Further reading

- [[vllm-benchmarks]] — `benchmarks/benchmark_serving.py` flag reference.
- [[sglang-benchmarks]] — `bench_serving.py` and the RadixAttention-specific extra metrics.
- [[sharegpt-workload]] — workload characteristics + filtering caveats.
- [[ttft-tpot-itl]] — what each metric measures and when each matters.
- [[goodput-slo]] — the SLO-met-throughput notion this lab's deliverable centres on.

## Excerpts

- [[excerpts/launch-cheatsheet]] — copy-pasteable vLLM and SGLang launch commands for the two paths.
- [[excerpts/benchmark-driver]] — the `benchmark_serving.py` invocation pattern + goodput computation Python.
- [[excerpts/ablation-cells]] — the three ablation sweeps with expected qualitative shapes.
- [[excerpts/memo-template]] — the side-by-side table template + reflection scaffolding.
