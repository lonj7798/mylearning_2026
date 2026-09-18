<!-- chapter excerpt for ch-54. Primary-source extract, read 2026-09-17.
     If a library card of this slug exists under wiki/raw-data/llm-training/, prefer the card. -->

# Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL (ASearcher)
- **Artifact:** Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, Yi Wu (IIIS Tsinghua; Ant Group; University of Washington). arXiv:2508.07976 (v1 2025-08; text read from v4, 2025-10-26). Source type: paper. Code: github.com/inclusionAI/ASearcher.
- **Core Insight:** In batch-generation RL systems a few very long trajectories block the whole batch, which is why turn limits in earlier online-RL search agents are 10 or fewer; decoupling trajectory execution from model updates (built on AReaL) allowed a turn limit of 128 per trajectory, and during training the agent reached over 100 tool calls and over 400k generated tokens in a single trajectory (§1, Fig. 1).
- **Guideline:** When the agent's useful behavior requires more turns than the synchronous batch can afford, remove the turn limit from the system design by making rollout asynchronous rather than by shortening the task, because the turn limit in synchronous systems is a consequence of batch blocking rather than of the algorithm (§1).

## Technical details (with loci)
- **System (§1).** Fully asynchronous agentic RL built on AReaL; trajectory execution is decoupled from model updates so a long trajectory does not block training.
- **Agent design (§1, Fig. 2).** A prompt-based agent with two basic tools, search and browsing, and no external LLM in the loop; the comparison baseline Search-R1 has search only.
- **Data (§1).** An LLM agent autonomously synthesizes challenging, uncertain, grounded QA pairs that require multi-turn tool use.
- **Results (Abstract, Fig. 1).** ASearcher-Web-QwQ, trained from QwQ-32B, reaches Avg@4 of 51.1 on xBench and 58.7 on GAIA. RL training improves GAIA by +15.0, xBench by +22.4, and Frames by +15.6 over the starting model.
- **Transfer (Abstract).** The authors report that the model reaches the performance of commercial systems when given an external summary tool in a zero-shot transfer setting with test-time search.

## Not reported
No throughput comparison against a synchronous baseline in this paper; no staleness parameter values; no held-out non-search evaluation.
