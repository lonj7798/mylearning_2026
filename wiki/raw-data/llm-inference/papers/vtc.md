<!-- scope: VTC fair scheduler for continuous-batched LLM serving
     deps: [[continuous-batching]]
     see-also: [[orca]], [[pagedattention]], [[admission-control-goodput]]
-->

# Fairness in Serving Large Language Models
- **Core Insight:** Fairness in LLM serving should be measured in served token work, not only requests per minute.
- **Guideline:** In multi-tenant serving, schedule by accumulated virtual token service so long prompts or outputs do not distort fairness.
- **Authors:** Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, Ion Stoica
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2401.00588
- **Relevant topics:** fairness, Virtual Token Counter, continuous batching, multi-tenancy, service functions, scheduling

## Abstract
The paper argues that API-style request rate limits are a poor fairness mechanism for LLM serving because requests can vary widely in input and output tokens. It defines fairness over a service function that accounts for token work, then proposes Virtual Token Counter (VTC), a scheduler compatible with continuous batching. VTC tracks each client's served virtual tokens and prioritizes clients with lower counters while remaining work-conserving.

## Key Contributions
- Defines LLM serving fairness using input/output-token service functions.
- Introduces VTC, a fair scheduler for continuous-batched serving.
- Handles clients leaving and rejoining without allowing unlimited credit hoarding.
- Proves a tight 2x upper bound on service difference between backlogged clients.
- Evaluates against FCFS, requests-per-minute, and least-counter baselines.

## Key Figures/Tables to Study
- Figure 1: VTC serving architecture with per-client counters.
- Algorithm section: counter updates and join/rejoin handling.
- Fairness-bound theorem.
- Synthetic and LMSYS trace evaluations.

## Technical Details
VTC keeps a queue of requests and a virtual counter per client. When the system can admit more work into the running batch, VTC chooses requests from clients with the lowest counters. Counters are updated according to a service function, such as a weighted sum of input tokens processed and output tokens generated.

Because VTC is built on continuous batching, it does not need to reserve a fixed GPU share for each tenant. It remains work-conserving: if requests exist and capacity is available, the server should not idle merely to preserve fairness. The policy interacts with KV-cache budget because a selected request still must fit in the memory pool.

## Connections
- Extends [[continuous-batching]] from throughput scheduling to fair multi-tenant scheduling.
- Can be combined with [[pagedattention]]-style block accounting.
- Differs from [[admission-control-goodput]], which may reject or defer work to protect SLO attainment.

## Notes
VTC's service function can be adapted when input and output tokens have different measured costs.
