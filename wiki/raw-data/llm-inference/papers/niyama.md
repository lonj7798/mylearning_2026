<!-- scope: QoS-driven LLM inference serving system for co-scheduling mixed interactive and batch workloads
     deps: [[continuous-batching]], [[sarathi-serve]]
     see-also: [[serving-optimization-foundations-2026]], [[admission-control-goodput]], [[vtc]]
-->

# Niyama: Breaking the Silos of LLM Inference Serving
- **Core Insight:** Instead of separating interactive and batch traffic into siloed clusters, Niyama co-schedules them on shared infrastructure using fine-grained QoS classes, dynamic chunking, hybrid prioritization, and selective relegation.
- **Guideline:** If a serving platform has mixed latency classes, model QoS explicitly; do not rely only on coarse interactive-vs-batch cluster partitioning.
- **Authors:** Kanishk Goel, Jayashree Mohan, Nipun Kwatra, Ravi Shreyas Anupindi, Ramachandran Ramjee
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2503.22562
- **Relevant topics:** QoS, scheduling, dynamic chunking, overload handling, mixed workloads, goodput

## Abstract
Niyama targets operational inefficiency from siloing LLM workloads into separate interactive and batch deployments. The system lets applications specify finer latency requirements, adapts scheduling based on real-time state, chunks requests dynamically, and selectively relegates requests during overload. The paper reports a 32% capacity increase over siloed deployments while maintaining QoS guarantees and reducing SLO violations under extreme load.

## Key Contributions
- Introduces fine-grained QoS classification for LLM inference requests.
- Uses dynamic chunking to improve throughput while preserving latency requirements.
- Applies hybrid prioritization to balance fairness and efficiency.
- Adds selective request relegation for graceful overload behavior.
- Evaluates against siloed deployments and reports higher capacity under QoS constraints.

## Key Figures/Tables to Study
- System architecture: how QoS classes enter scheduler decisions.
- Dynamic chunking timeline: shows how long prefills can be split without starving decode.
- Capacity and SLO-violation tables: connect scheduling policy to operational goodput.

## Technical Details

### Scheduling problem
Niyama treats request classes differently based on latency expectations. The key is not just "priority queue first"; it is balancing:
- currently admitted requests,
- expected chunk cost,
- SLO budget remaining,
- queue pressure,
- overload policy.

### Relationship to chunked prefill
Dynamic chunking is a natural extension of [[sarathi-serve]]-style chunked prefill: break large prompt processing into scheduler-visible units so decode-heavy latency-sensitive traffic can continue making progress.

## Connections
- [[serving-optimization-foundations-2026]] — Niyama is a concrete system answering the position paper's call.
- [[admission-control-goodput]] — use goodput rather than raw throughput for Niyama-style QoS evaluation.
- [[vtc]] — adjacent fairness/scheduling work.
- [[distserve]] / [[splitwise]] — topology-level approach to prefill/decode interference; Niyama is scheduler/QoS-level.
