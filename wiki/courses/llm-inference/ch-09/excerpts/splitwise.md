---
chapter: ch-09
course: llm-inference
phase: read
excerpt_of: "Splitwise: Efficient Generative LLM Inference Using Phase Splitting"
source_url: https://arxiv.org/abs/2311.18677
created_at: "2026-05-21"
---

# Excerpt: Splitwise — characterize first, then split across hardware tiers

**Authors:** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Inigo Goiri, Saeed Maleki, Ricardo Bianchini
**Year:** 2023 / 2024
**Venue:** ISCA 2024 (arXiv Nov 2023)
**URL:** https://arxiv.org/abs/2311.18677
**Raw-data source:** [[raw-data/splitwise]]

---

## The workload characterization (the empirical foundation)

Splitwise profiles prefill vs decode on production Azure / Microsoft traces. The decisive measurements:

| Metric | Prefill | Decode |
|---|---|---|
| Compute utilization (A100, batch 1) | 80–95% (long prompt) | ~10% (memory-bound) |
| Cost per token | ~1 ms (batched) | ~20 ms (per token, per request) |
| Power draw | ~360 W | ~250 W |
| Latency target (SLO) | TTFT (seconds) | TPOT (tens of ms) |
| Ideal batch size | 1–4 (prompts) | 32–64 (sequences) |
| Ideal hardware | high compute (H100) | high memory bandwidth (A100 OK) |

The 10% decode utilization is the headline. On a single sequence, decode reads the full model weights from HBM per token and does almost no math relative to the bytes moved. **The compute lies idle.** Batching helps amortize the weight read across many sequences — but a colocated engine cannot batch 64 decodes without crowding out the 4-prompt prefill batch.

---

## Phase splitting as a hardware specialization argument

Splitwise's most distinctive claim: don't just split phases across machines, split them across **hardware tiers**. Use H100s where compute matters (prefill); use A100s or older GPUs with strong memory subsystems where bandwidth matters (decode). The cost saving is structural:

- A100 list cost is ~40% of H100; decode-bound work doesn't need the H100 compute.
- A mixed fleet (H100 prefill, A100 decode) at the same QPS uses ~25% less power and costs ~30% less to acquire.

The DistServe paper does the *scheduler* contribution; Splitwise does the *fleet design* contribution.

---

## The mixed-pool hedge

Pure disaggregation strands capacity: at any moment one side is saturated and the other has idle headroom. Splitwise proposes a third pool of "mixed" workers that can swing to prefill or decode duty based on real-time load.

```
       Prefill pool (H100×N)
              │
              ├──► Mixed pool (H100×M) ◄──┐
              │                            │
       Decode pool (A100×K) ───────────────┘
```

The mixed pool absorbs traffic bursts. In simulation it cuts strand rate from 40% to <10% with M = 0.15 · (N + K).

---

## Reported numbers

- 1.4× higher throughput at the same dollar cost vs Llama-2-70B colocated H100.
- 2.35× more requests/sec at the same cost using mixed H100/A100 tiers.
- 25% lower power at the same throughput.

---

## Connections

- [[excerpts/distserve]] — operationalizes Splitwise's phase split as a goodput-optimized scheduler.
- [[excerpts/mooncake]] — production extension with KV-cache storage tiers.
- [[ch-09]] — parent synthesis.
- [[ch-20]] — production-stack reports that follow Splitwise's hardware-mix philosophy.
