---
chapter: ch-21
course: llm-inference
phase: read
excerpt_of: "vLLM benchmark_serving.py invocation + goodput computation"
source_url: https://docs.vllm.ai/en/latest/contributing/benchmarks.html
created_at: "2026-05-21"
---

# Excerpt: ShareGPT benchmark driver + goodput script

**Source:** vLLM `benchmarks/benchmark_serving.py`
**Raw-data source:** [[raw-data/vllm-benchmarks]], [[raw-data/sharegpt-workload]], [[raw-data/ttft-tpot-itl]]

---

## Why one driver, two servers

`benchmark_serving.py` from vLLM speaks the OpenAI Chat Completions API. Both vLLM and SGLang expose OpenAI-compatible servers, so the *same* driver script measures both — making the comparison apples-to-apples.

Get the script:

```bash
# Cloning vLLM gets you the benchmark scripts + ShareGPT helpers
git clone --depth 1 https://github.com/vllm-project/vllm.git
cd vllm/benchmarks
```

---

## Fetch the ShareGPT trace

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
  -O sharegpt.json
```

The cleaned split is ~600 MB. The benchmark script auto-filters for valid conversation pairs.

---

## Single-rate run (template)

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --base-url http://localhost:8001 \
  --endpoint /v1/chat/completions \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset-name sharegpt \
  --dataset-path sharegpt.json \
  --num-prompts 1000 \
  --request-rate 8 \
  --seed 42 \
  --save-result \
  --result-dir results/vllm \
  --result-filename "vllm_rate8.json"
```

Flag notes:

| Flag | Lab value | Why |
|------|-----------|-----|
| `--backend openai-chat` | required | Targets `/v1/chat/completions`; works for both vLLM and SGLang |
| `--num-prompts` | 1000 (full) / 500 (constrained) | Enough samples for stable p99; too few → noisy |
| `--request-rate` | swept {1..32} | Open-loop Poisson arrival rate |
| `--seed` | 42 | Reproducibility |
| `--save-result` | required | Dumps the JSON we parse later |

A request-rate of `inf` (or omitting `--request-rate`) means closed-loop (each new request fires as soon as the previous returns) — that's a *different* benchmark. Stick to open-loop for the lab.

---

## Full sweep driver

`run_sweep.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

URL=${1:-http://localhost:8001}
TAG=${2:-vllm}
RATES=${3:-"1 2 4 8 12 16 24 32"}

mkdir -p results/${TAG}

for rate in ${RATES}; do
  echo "[$(date +%H:%M:%S)] ${TAG} rate=${rate}"
  python benchmark_serving.py \
    --backend openai-chat \
    --base-url ${URL} \
    --endpoint /v1/chat/completions \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --num-prompts 1000 \
    --request-rate ${rate} \
    --seed 42 \
    --save-result \
    --result-dir results/${TAG} \
    --result-filename "${TAG}_rate${rate}.json"
done
```

Invoke:

```bash
./run_sweep.sh http://localhost:8001 vllm
./run_sweep.sh http://localhost:8002 sglang
```

---

## What the result JSON contains

```json
{
  "duration": 312.5,
  "completed": 1000,
  "total_input_tokens": 421083,
  "total_output_tokens": 187302,
  "request_throughput": 3.20,
  "input_throughput": 1347.5,
  "output_throughput": 599.4,
  "mean_ttft_ms": 412.7,
  "median_ttft_ms": 198.2,
  "std_ttft_ms": 612.8,
  "p99_ttft_ms": 1842.5,
  "mean_tpot_ms": 31.2,
  "median_tpot_ms": 28.9,
  "p99_tpot_ms": 67.4,
  "mean_itl_ms": 30.8,
  "p99_itl_ms": 71.1,
  "input_lens": [...],
  "output_lens": [...],
  "ttfts": [...],
  "itls": [[...], [...], ...]
}
```

The per-request arrays at the end are what you need for goodput.

---

## Compute goodput@p99

`compute_goodput.py`:

```python
import json, glob, sys, statistics
from pathlib import Path

SLO_TTFT_MS = 2000     # tighten / loosen per the lab memo's declared SLO
SLO_TPOT_MS = 80

def goodput_from_file(p):
    d = json.loads(Path(p).read_text())
    ttfts = d["ttfts"]
    itls = d["itls"]   # list-of-list, per request
    duration = d["duration"]

    n_success = 0
    for ttft, gaps in zip(ttfts, itls):
        if not gaps:
            continue
        ttft_ms = ttft * 1000
        # P99-ITL flavour: each request's worst inter-token gap must clear SLO
        worst_gap_ms = max(gaps) * 1000
        if ttft_ms <= SLO_TTFT_MS and worst_gap_ms <= SLO_TPOT_MS:
            n_success += 1
    return n_success / duration   # SLO-met requests per second

if __name__ == "__main__":
    for path in sorted(glob.glob(f"{sys.argv[1]}/*.json")):
        gp = goodput_from_file(path)
        rate = int(path.split("rate")[-1].split(".")[0])
        print(f"{path}  rate={rate:>3d}  goodput@p99={gp:.2f} req/s")
```

Run it on each set of results:

```bash
python compute_goodput.py results/vllm
python compute_goodput.py results/sglang
```

You'll get something like:

```
results/vllm/vllm_rate8.json    rate=  8  goodput@p99=7.82 req/s
results/vllm/vllm_rate16.json   rate= 16  goodput@p99=14.31 req/s
results/vllm/vllm_rate24.json   rate= 24  goodput@p99=15.10 req/s  ← knee
results/vllm/vllm_rate32.json   rate= 32  goodput@p99=11.20 req/s  ← past-knee degradation
```

The "past-knee degradation" pattern (offered rate keeps rising, goodput collapses) is one of the most important things to *see* in the lab. It is exactly what [[goodput-slo]] warns about.

---

## Common pitfalls

- **Server not warm.** First few hundred requests may pay JIT / CUDA-graph capture cost. Either discard the first 100 results, or run a 50-request warmup before the real sweep.
- **GPU shared with other processes.** `nvidia-smi` should show only your server and the benchmark driver. Anything else (Jupyter, browser, gaming overlay) will perturb p99 ITL.
- **Both servers running simultaneously.** Each takes ~17 GB for the BF16 8B weights + KV pool; two servers on one 80 GB H100 with `--gpu-memory-utilization 0.90` each will OOM. Run them sequentially.
- **Network limit on remote benchmarking.** If the benchmark client is on a different machine, network RTT pollutes TTFT measurement. Run client and server on the same host.

---

## Connections

- [[excerpts/launch-cheatsheet]] — the server commands that this driver hits.
- [[excerpts/ablation-cells]] — three ablation sweeps that re-use this driver verbatim with different server flags.
- [[ch-19]] — the metrics definitions this driver implements.
