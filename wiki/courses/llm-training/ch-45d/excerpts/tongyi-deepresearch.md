---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: arXiv:2510.24701 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2510.24701
created_at: "2026-09-15"
---

# Excerpt: Tongyi DeepResearch Technical Report

- **Authors:** Tongyi DeepResearch Team (Alibaba)
- **Year:** 2025 (arXiv v1 2025-10-28; revised 2026-05-18)
- **Source type:** official technical report
- **Used in:** [[read]] §1, §2, §3, §5, Recipe, Negative feedback

## Scale and pipeline
- 30.5B total parameters, 3.3B activated (§6 limitations and §7 state the activated count).
- Two named stages: "agentic mid-training and agentic post-training" (§1). The report calls its mid-training "Agentic Continual Pre-training (Agentic CPT)".

## Agentic mid-training (§3.3.1)
- Two stages: "We initiate with a 32K context length in the first stage, before expanding to 128K in the second. This expanded context window is specifically leveraged in the second stage, where we introduce a substantial corpus of long-sequence (64K-128K) agentic behavior data."
- Objective is plain next-token prediction. "Throughout both stages, a small proportion of general pre-training data is interleaved, ensuring the model acquires specialized agentic competence without sacrificing its foundational generalization capabilities."
- Stated purpose: "provide a base model endowed with a strong inductive bias for agentic behavior, while simultaneously preserving broad linguistic competence."
- Synthesized data spans "the complete lifecycle of agent workflows": question synthesis, planning actions, reasoning actions and decision-making actions (§3.3.2).

## Three environment types (§3.2, §3.4.3)
- Prior World: task elements and tools with no real responses. Simulated: "an offline environment based on the 2024 Wikipedia database and ... a suite of local RAG tools to simulate the web environment", described as "low-cost, high-efficiency, and fully controllable". Real-world: live tools (Search, Visit, Python Interpreter, Google Scholar, File Parser) behind a unified sandbox with "QPS rate constraints, result caching, automatic timeout-and-retry protocols, graceful service degradation ... and seamless failover to backup data sources".
- Stated reason for the sandbox: API volatility "makes it nearly impossible to diagnose performance issues, obscuring whether a poor outcome is caused by a weakness in the agent's policy or by the instability of the environment itself".

## SFT cold start (§3.4.2)
- Rejection-sampled trajectories from high-performing open-source models, in two formulations: ReAct Mode and a Context Management Mode whose input is the previous step's trajectory summary, tool call and tool response, and whose output includes a new summary.
- Two-stage context schedule: "In the first stage, the context length is set to 40K, and the training data consist of ReAct Mode samples with context lengths shorter than 40K, along with all Context Management Mode samples ... In the second stage, the context length is extended to 128K, and the training data include ReAct Mode samples with context lengths between 40K and 128K, as well as a small portion of 40K data for stability."

## Agentic RL (§3.4.3)
- GRPO variant (Eq. 4-5): token-level policy-gradient loss, clip-higher, advantage `Â_{i,j} = R_i − mean({R_i})`, leave-one-out variance reduction, strictly on-policy sampling so the importance ratio "remains 1.0".
- "The reward is a pure 0 or 1 signal of answer correctness. We do not include a format reward (e.g., 0.1 for format correctness) because the preceding cold start stage ensures the model is already familiar with the required output format."
- Negative-sample control: "we observed in preliminary experiments that directly optimizing on an unfiltered set of negative rollouts significantly degrade training stability and can lead to policy collapse after extended training. To mitigate this, we selectively exclude certain negative samples from the loss calculation, for instance, those that do not yield a final answer because they exceed a length limit."
- Automatic data curation: filter out problems the SFT policy always fails or always solves; monitor D′ during training; refresh it from a background pool of newly moderate-difficulty problems "When the training reaches a certain step count or the reward plateaus".
- Stated conclusion: "the success of agentic RL depends more on the quality of the data and the stability of the training environment than on the specific algorithm being used."
- Infrastructure: step-level asynchronous rollout on rLLM, with separate servers for model inference and tool invocation and a centralized interaction handler.
- §3.4.4 adds model merging as the last stage, averaging variants derived from the same pre-trained model.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2510.24701: §1, §3.2, §3.3.1, §3.3.2, §3.4.2, §3.4.3, §3.4.4, §6.
- Not reported by the source: mid-training token counts; SFT dataset size; RL learning rate, group size G, clip values, step count; number of environments or tasks; GPU hours.
