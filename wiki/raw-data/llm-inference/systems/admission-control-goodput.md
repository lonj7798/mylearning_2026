<!-- scope: synthesis source page for SLO-aware admission control and goodput in LLM serving
     deps: [[distserve]]
     see-also: [[mooncake]], [[vtc]], [[sarathi-serve]]
-->

# Admission Control and Goodput in LLM Serving
- **Core Insight:** Under overload, serving every request can reduce useful output; goodput counts only work that satisfies user-visible SLOs.
- **Guideline:** Admit, defer, downgrade, or reject requests based on predicted TTFT/TPOT feasibility and KV-memory impact, not just queue order.
- **Authors:** Synthesis around DistServe, Mooncake, VTC, Sarathi-Serve, and SLO/goodput metric work
- **Year:** 2024-present
- **URL:** https://arxiv.org/abs/2401.09670 ; https://arxiv.org/abs/2407.00079 ; https://arxiv.org/abs/2410.14257 ; https://arxiv.org/abs/2505.23022
- **Relevant topics:** goodput, SLOs, admission control, early rejection, overload, TTFT, TPOT, fairness

## Abstract
Throughput measures how much work a serving system performs, but overloaded LLM systems can perform work that arrives too late to be useful. Goodput reframes capacity as requests or tokens that meet SLOs such as TTFT and TPOT. Admission control uses estimates of queueing delay, prefill cost, decode cost, and KV-cache footprint to decide whether a request should enter the system.

## Key Contributions
- Defines the serving-system pattern rather than one artifact.
- Connects DistServe's SLO-constrained request-rate objective with Mooncake's early rejection.
- Highlights KV cache as an admission-control resource, not only a memory-allocation detail.
- Separates fairness scheduling from SLO admission: a fair request can still be impossible to serve on time.
- Points to newer SLO/goodput metric work and heterogeneous-SLO schedulers as follow-up sources.

## Key Figures/Tables to Study
- DistServe goodput curves under TTFT/TPOT targets.
- Mooncake overload and prediction-based early rejection design.
- Revisiting SLO and Goodput Metrics: metric definitions and smooth-goodput framing.
- SCORPIO: admission control, TTFT guard, and TPOT guard for heterogeneous SLOs.

## Technical Details
An admission controller needs a cost model. Inputs include prompt tokens, requested max output tokens, observed output-length distribution, current queue, active decode batch, free KV blocks, prefill/decode service rates, and SLO tier. The controller estimates whether admitting the request will meet its SLO and whether it will cause already-admitted work to miss SLO.

KV cache is central because admission is not just about compute. A long prompt can consume enough KV blocks to reduce batch capacity for many short decodes. In disaggregated systems, the controller also accounts for KV transfer time and cache placement.

Policies include hard rejection, early rejection under overload, priority tiers, queue reordering, chunk-size changes, and routing to different worker pools. The risk is inaccurate prediction: rejecting too aggressively wastes capacity, while admitting doomed requests hurts goodput and tail latency.

## Connections
- [[distserve]] uses goodput under TTFT/TPOT SLOs as the main objective.
- [[mooncake]] adds prediction-based early rejection in a production long-context service.
- [[vtc]] addresses fairness; admission control addresses feasibility under SLO/load.
- [[sarathi-serve]] changes admission feasibility by reducing decode stalls with chunked prefill.
