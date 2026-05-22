<!-- chapter: ch-10
     phase: serving-architecture
     title: Admission Control + Goodput + Fair Scheduling
     sources: [[vtc]], [[niyama]], [[admission-control-goodput]], [[serving-optimization-foundations-2026]], [[llm-serving-survey]]
     back: [[continuous-batching]] (ch-04), [[distserve]] (ch-09)
     forward: [[ttft-tpot-itl]] (ch-19), framework SLO knobs (ch-16/17)
-->

# Chapter 10 — Admission Control + Goodput + Fair Scheduling

> **Core insight.** Throughput counts tokens; **goodput counts only the tokens that arrived on time**. Once a serving system is at or past saturation, more throughput can *reduce* goodput — by admitting requests it cannot finish in their SLO, it degrades every in-flight request as well. The fix is a three-layer policy stack: **admission control** decides who enters, **fair scheduling** decides who runs next, and **graceful overload** decides who gets shed. None of these reduce to FIFO or LRU; LLM serving has too much structure for generic heuristics to hold.
>
> **Guideline.** Always measure goodput, not throughput. For multi-tenant serving, use VTC-style virtual-token-counter scheduling above continuous batching. For mixed interactive/batch traffic, use Niyama-style QoS tiers. For long-context production at the SLO frontier, use Mooncake-style predictive early rejection. By 2026 the consensus position ([[serving-optimization-foundations-2026]]) is that LLM serving has outgrown FIFO/JSQ/LRU — formal optimization (mixed-integer scheduling, cost-aware routing) is starting to replace heuristics in research stacks.

---

## Why this chapter exists

By chapter 9 you can build a serving fleet — continuous batching, chunked prefill, optional prefill/decode disaggregation. None of those tell you what to do when:

1. Two clients arrive at the same instant; whose request runs first?
2. The fleet is past its sustainable QPS; do you queue, reject, or degrade?
3. A user sends a 200k-token prompt that will alone occupy 40% of decode KV memory for 30 seconds; do you admit it?
4. A premium tier ("interactive chat, p99 TTFT < 500 ms") shares hardware with a batch tier ("nightly summarization, no latency SLO"); how do you keep both honest?

These are the policy questions that sit on top of every serving engine. They are also the most under-engineered layer in production stacks — most deployments still use FIFO + max-concurrent-requests + per-API-key rate limits, and discover the limitations the first time a single user submits a million-token prompt.

The literature crystallized between 2024 and 2026:

- **[[vtc]]** (Sheng et al., Berkeley, 2024) — fair scheduling for continuous batching: the LLM analog of fair queueing.
- **[[niyama]]** (Goel et al., MSR-India, 2025) — QoS-driven co-scheduling of interactive and batch tiers.
- **[[admission-control-goodput]]** — the design-pattern synthesis: goodput + SLO-aware admission.
- **[[serving-optimization-foundations-2026]]** (Zhou, 2026) — position paper: serving needs formal optimization, not heuristics.
- **[[llm-serving-survey]]** (Miao et al., 2023) — the broader taxonomy.

This chapter builds the three policy layers, then puts them together.

---

## 1. Throughput vs goodput — the metric that should drive everything

A throughput-first scheduler maximizes:

```
throughput = requests_completed / wallclock_seconds
```

A goodput-first scheduler maximizes:

```
goodput(λ, S) = λ · Pr[ request meets SLO S ]
              = (requests_meeting_SLO) / wallclock_seconds
```

where `S` is the SLO tuple `(TTFT_max, TPOT_max, optionally total_latency_max)`.

The two diverge under load. Consider a system at 110% of its SLO-feasible QPS:

- **Throughput-first** admits everything. Many requests miss their SLO, but throughput looks high. Goodput is ~60% of nominal because the SLO-violators don't count.
- **Goodput-first** rejects ~20% of arrivals at the door. Throughput is lower (you literally serve fewer requests), but admitted requests mostly meet SLO. Goodput is ~95% of nominal.

The business observation: **a request that arrives 3× over budget is often useless to the user** (they've given up, retried, or churned). Goodput, not throughput, is what corresponds to revenue. Every recent serving paper — DistServe, Mooncake, Niyama, the 2026 position paper — uses goodput as the headline metric.

**Defining your SLO.** A typical production SLO tuple:

| Tier | TTFT p95 | TPOT p95 | Notes |
|---|---|---|---|
| Interactive chat | 500 ms | 50 ms | streaming UX |
| Code completion (Copilot-style) | 200 ms | 30 ms | typing UX |
| Voice / real-time | 100 ms | 20 ms | sub-perceptual |
| Background / batch | 30 s | 1 s | overnight summarization |
| Search RAG | 800 ms | 80 ms | mixed |

The SLO defines what "goodput" means. Without it, "fast enough" is undefined and you have nothing to optimize.

See [[excerpts/admission-control-goodput]] for the cost-model variables and the failure modes of throughput-only scheduling.

---

## 2. Admission control — refuse what you cannot serve

The colocated-engine instinct is to queue every arriving request. Past saturation, that turns the queue into a black hole: queue depth grows monotonically, every admitted request now sits behind a longer queue, and everyone misses SLO.

The fix is **admission control**: a guard at the door that decides, per-arrival, whether the system can serve this request inside its SLO given current state.

The decision logic (from [[admission-control-goodput]]):

```text
On request R arriving with SLO S:
    state = (queue_depth, active_batch, KV_free_blocks, prefill_rate, decode_rate)
    est_TTFT = queue_delay(state) + prefill_cost(R.prompt_len)
    est_TPOT = decode_cost(active_batch + 1, KV_free_after_admit)
    if est_TTFT > S.TTFT_max OR est_TPOT > S.TPOT_max:
        return REJECT  # 429 Too Many Requests
    if KV_blocks_needed(R) > KV_free_blocks:
        return REJECT  # cache full
    return ADMIT
```

The inputs the controller needs:

- **Queue model**: current waiting requests + their (predicted) prefill costs.
- **Per-phase service rate**: prefill tokens/sec, decode tokens/sec, both measured.
- **KV memory budget**: free blocks vs blocks-needed-for-admit.
- **Per-tenant SLO**: pulled from request metadata.
- **Output-length distribution**: needed to predict KV occupancy through the request lifetime.

The output-length prediction is the hard part. You can't know `N_output` until you've generated it. Practical systems use:

1. **Conservative upper bound**: `min(max_tokens, learned_p95_output_length)`.
2. **Pseudo-online estimation**: track per-user / per-app distribution of historical output lengths.
3. **Pre-fork early eviction**: if a request runs past predicted length, evict mid-flight under explicit policy.

Mooncake (ch-09) uses a learned predictor calibrated on production traces; it tolerates some prediction error in exchange for higher admit rate.

**Policy variants beyond hard reject:**

- **Defer / queue with deadline**: admit to a delay queue; reject if the queue still has room when the deadline hits.
- **Downgrade SLO tier**: an interactive request that can't make 500 ms TTFT gets demoted to the standard tier (1 s TTFT) with the user notified.
- **Route to alternate pool**: a decode pool is full → route to a different region's pool.
- **Truncate**: clip `max_tokens` to a value the system can serve.

---

## 3. VTC — fair scheduling for continuous batching

API rate limits ("100 requests / minute per key") are a poor fairness mechanism for LLM serving because requests vary in cost by 1000×. A user who sends 100 chat messages with 50-token prompts consumes far less than a user who sends 1 RAG query with a 100k-token prompt. Counting requests treats them as equivalent.

[[vtc]] (Sheng et al., arXiv 2401.00588) introduces **Virtual Token Counter (VTC)**: track each client's cumulative *served token-work* and prioritize the client with the lowest counter.

### The service function

For each client `c`, maintain a virtual counter `V_c`. When the scheduler runs `n_in` input tokens (prefill) and `n_out` output tokens (decode) for client `c` in a step:

```math
V_c \;\mathrel{+}=\; w_{\text{in}} \cdot n_{\text{in}} \;+\; w_{\text{out}} \cdot n_{\text{out}}
```

with `w_in / w_out` reflecting the per-token cost ratio (prefill tokens are cheaper per token than decode tokens; typical ratio `w_in : w_out` = 1 : 4 to 1 : 8).

### The scheduling rule

```text
When admitting requests into the running batch:
    candidate_clients = {c : c has pending requests}
    pick client c* = argmin_{c in candidate_clients} V_c
    admit c*'s next request (subject to KV budget)
```

This is "smallest virtual finish time first" — the LLM analog of weighted fair queueing.

### Join/leave handling

The naive "lowest counter first" rule has a known bug: a client that goes idle for an hour accumulates "credit" (their counter doesn't grow while others' do). When they return, they monopolize the system for hours.

VTC fixes this by clamping the rejoining counter:

```text
On client c rejoining after idle:
    V_c = max(V_c, min_active_V_c)
```

i.e., a rejoining client cannot start with a counter lower than the lowest active client. This bounds the credit hoarding.

### The fairness guarantee

VTC proves a **2× upper bound** on service difference between any two continuously-backlogged clients: at any time, no client has received more than 2× the service of another in the same time window. This is tight; the constant 2 cannot be improved without giving up work-conservation.

### Why VTC works above continuous batching

VTC plugs into the same scheduler loop continuous batching uses (per-iteration request selection). It doesn't reserve fixed shares; it remains **work-conserving** — if there's capacity and any request to run, the system runs *something*. The token-counter mechanism just shifts *which* something.

This matters because reserving capacity (e.g., "tenant A gets 30% of the GPUs") under-utilizes the fleet whenever a tenant is below its share. VTC achieves fairness with full utilization.

See [[excerpts/vtc]] for the service-function formula and the 2× fairness theorem proof sketch.

---

## 4. Niyama — QoS tiers for mixed workloads

[[niyama]] (Goel et al., MSR-India, arXiv 2503.22562) addresses a different fairness problem: **multiple latency classes on shared hardware**. The status quo is to silo: deploy one fleet for interactive chat and a separate fleet for batch summarization. Niyama's measurement: silos waste 30–40% of capacity because each fleet is over-provisioned for its peak.

### The mechanism

Requests carry a **QoS class** (`interactive`, `standard`, `batch`, possibly more). The scheduler co-runs all classes on the same hardware but with class-aware policies:

1. **Interactive class gets priority** for admission into the running batch, but only up to a configurable ceiling (so interactive bursts don't starve everything else).
2. **Batch class accepts dynamic chunking aggressively** — its prefills can be split into very small chunks because TPOT-style SLOs don't apply.
3. **Standard class fills the gaps** — it runs whenever interactive and batch have spare capacity.
4. **Hybrid prioritization** — interactive requests preempt standard ones at iteration boundaries (cheap, no KV ship); batch never preempts.
5. **Selective relegation under overload** — interactive requests stay; batch requests get pushed to a slower pool or rejected.

### Reported numbers

Niyama reports **32% more capacity** at the same SLO-attainment vs siloed deployment, with SLO violations dropping by 50% at extreme load. The gain comes almost entirely from sharing the headroom that siloed fleets reserve separately.

### Relationship to chunked prefill

Niyama's dynamic chunking is a direct extension of [[sarathi-serve]] (ch-05). Sarathi uses one chunk size globally; Niyama uses per-class chunk sizes. Batch-class can run with 2k-token chunks (high throughput); interactive-class runs with 256-token chunks (low decode interference).

See [[excerpts/niyama]] for the QoS class definitions and the hybrid prioritization pseudocode.

---

## 5. The 2026 position — serving has outgrown heuristics

[[serving-optimization-foundations-2026]] (Zhou, 2026) is the field-shaping position paper. Its claim:

> LLM serving systems still use FIFO scheduling, join-shortest-queue routing, round-robin load balancing, and LRU cache eviction — all general-purpose heuristics that pre-date LLM-specific structure. These are now provably suboptimal in operating regimes that matter.

The four pieces of LLM-specific structure that break generic heuristics:

1. **Prefill/decode phase asymmetry** — one queue for both is wrong; one cache for both is wrong.
2. **KV cache grows during execution** — memory occupancy is a function of time, not arrival.
3. **Unknown output length** — service time is fundamentally uncertain at admission.
4. **Continuous-batching coupling** — every active request affects every other request's TPOT.

The paper calls for formal models:

- **Routing**: mixed-integer programming over (request, worker, prefix-cache-location) to minimize total weighted TTFT.
- **Scheduling**: stochastic-deadline scheduling with predicted output lengths.
- **Cache eviction**: Belady-style optimal eviction approximations using request-level reuse predictions, not block-level LRU.
- **Admission**: economic-utility framing — admit iff `E[goodput contribution] > E[goodput cost]`.

The position is that **the literature has had three years to crystallize and is starting to converge on optimization-theoretic formulations**. Niyama is one early instantiation; SCORPIO ([[admission-control-goodput]] references it) is another. Expect 2027-class production stacks to ship optimization-based schedulers.

For this chapter the takeaway is methodological: when designing a serving policy, ask "what would the optimal scheduler do given full state?" before reaching for a heuristic.

---

## 6. Putting the layers together

A complete policy stack for a 2026-class serving system:

```text
Request arrives
   │
   ▼
[ ADMISSION ] ─── reject if est SLO violation
   │
   ▼
[ QoS classification ] ─── interactive / standard / batch
   │
   ▼
[ FAIR / PRIORITY scheduling ] ─── VTC per-tenant + Niyama per-class
   │
   ▼
[ ROUTING ] ─── cache-aware: send to worker holding longest prefix
   │
   ▼
[ EXECUTION ] ─── continuous batching + chunked prefill + paged KV
   │
   ▼ token stream
[ MONITORING ] ─── goodput / SLO attainment / per-tenant fairness / queue depth
   │
   ▼ feedback
[ OVERLOAD RESPONSE ] ─── shed batch tier, demote standard, refuse interactive last
```

Each layer above the engine is independently configurable. Production stacks tend to ship them in this order: continuous batching first (always), then admission + per-tenant rate limit (week 1), then fair scheduling (when tenants complain), then QoS classes (when batch workload appears), then optimization-based routing (when you have a research team).

---

## 7. Common failure modes the policies prevent

| Failure mode | Symptom | Layer that fixes it |
|---|---|---|
| One huge prompt monopolizes KV | TPOT explodes for everyone | Admission control + KV-aware admit |
| One noisy tenant starves others | Per-tenant p99 catastrophic | VTC fair scheduling |
| Interactive SLO violated during batch run | TTFT spikes nightly | Niyama QoS tiers |
| Queue depth grows monotonically | Latency snowball | Admission with predicted SLO |
| Idle-then-return user grabs everything | Burst-then-starve | VTC rejoin clamp |
| Fleet at 95% but goodput at 60% | Throughput high, SLO miss high | Switch metric to goodput |
| Long-tail output explodes mid-stream | Mid-flight KV exhaustion | Output-length cap + preemption |

---

## 8. Practitioner's policy-tuning checklist

```python
# Minimum viable policy stack — copy this as your starting point.

policy = {
    # 1. SLO definition (define this FIRST, everything else derives)
    "slo": {
        "interactive":  {"ttft_p95_ms": 500, "tpot_p95_ms": 50},
        "standard":     {"ttft_p95_ms": 2000, "tpot_p95_ms": 100},
        "batch":        {"ttft_p95_ms": 30000, "tpot_p95_ms": 1000},
    },
    # 2. Admission control
    "admission": {
        "max_queue_depth": 64,
        "max_predicted_ttft_ratio": 0.8,   # reject if est_TTFT > 0.8 * SLO
        "kv_admit_threshold": 0.85,         # don't admit if it pushes KV > 85% used
        "output_length_estimator": "ema_p95_per_tenant",
    },
    # 3. Fair scheduling (VTC)
    "fairness": {
        "policy": "vtc",
        "w_in": 1.0,
        "w_out": 4.0,
        "rejoin_clamp": True,
    },
    # 4. QoS (Niyama)
    "qos": {
        "interactive_admit_ceiling": 0.5,   # at most 50% batch share to interactive
        "batch_chunk_tokens": 2048,
        "interactive_chunk_tokens": 256,
        "preempt_standard_for_interactive": True,
    },
    # 5. Overload behavior
    "overload": {
        "trigger_at_kv_used": 0.9,
        "shed_class_order": ["batch", "standard"],   # never shed interactive first
        "early_reject_threshold_ms": 100,           # reject if est slip > 100 ms over SLO
    },
    # 6. Metrics (goodput, not throughput, is the primary)
    "monitoring": {
        "primary_metric": "goodput_per_tier",
        "track": ["ttft_p95", "tpot_p95", "kv_used", "vtc_max_diff", "reject_rate"],
    },
}
```

Tune from real traces. The defaults above are the right shape; the exact numbers depend on your model size, hardware, and workload.

---

## Connections and what's next

- **Back to [[continuous-batching]] (ch-04)** — every policy layer sits *above* the continuous-batching loop. Admission decides what enters the loop, VTC decides what the loop picks next.
- **Back to [[sarathi-serve]] (ch-05)** — Niyama's per-class chunk sizes are a generalization of Sarathi's global chunk size.
- **Back to [[distserve]] / [[mooncake]] (ch-09)** — disaggregation gives you two independent admission controllers (one per pool); Mooncake's early rejection is the production version of the admission logic here.
- **Forward to [[ttft-tpot-itl]] (ch-19)** — how to actually measure goodput, including request-rate sweeps and percentile reporting.
- **Forward to ch-16 / ch-17** — vLLM and SGLang both expose admission knobs (`max_num_seqs`, queue-depth caps); the policy concepts here map onto framework-specific config.
- **Forward to ch-21 lab** — the head-to-head benchmark must report goodput, not just throughput.

## Further reading

- [[vtc]] — Sheng et al. 2024; the fair-scheduling paper for continuous batching.
- [[niyama]] — Goel et al. 2025; QoS-driven mixed-workload scheduling.
- [[admission-control-goodput]] — synthesis card across DistServe, Mooncake, VTC, Sarathi.
- [[serving-optimization-foundations-2026]] — Zhou 2026; position paper calling for formal optimization.
- [[llm-serving-survey]] — Miao et al. 2023; the broader taxonomy of optimization families.

## Companion visualization

**[figures/goodput-vs-throughput-slo.html](figures/goodput-vs-throughput-slo.html)** — interactive request-rate sweep. Slider for QPS shows throughput rising monotonically while goodput peaks and then collapses past the saturation point; an admission-control toggle keeps goodput flat past saturation.
