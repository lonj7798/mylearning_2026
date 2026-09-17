---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: Qwen blog "Qwen3-Coder: Agentic Coding in the World", 2025-07-22 (no library card at the time of writing; chapter-local verified extract)
source_url: https://qwenlm.github.io/blog/qwen3-coder/
created_at: "2026-09-15"
---

# Excerpt: Qwen3-Coder: Agentic Coding in the World

- **Authors:** Qwen Team, Alibaba
- **Year:** 2025 (blog dated 2025-07-22)
- **Source type:** official blog
- **Used in:** [[read]] §1, §2, §5, Recipe

## Model and pre-training
- "Qwen3-Coder-480B-A35B-Instruct — a 480B-parameter Mixture-of-Experts model with 35B active parameters which supports the context length of 256K tokens natively and 1M tokens with extrapolation methods".
- "Scaling Tokens: 7.5T tokens (70% code ratio), excelling in coding while preserving general and math abilities."
- "Scaling Context: Natively supports 256K context and can be extended up to 1M with YaRN, optimized for repo-scale and dynamic data (e.g., Pull Requests) to empower Agentic Coding."
- "Scaling Synthetic Data: Leveraged Qwen2.5-Coder to clean and rewrite noisy data".

## Post-training
- Code RL: "we believe all code tasks are naturally well-suited for execution-driven large-scale reinforcement learning ... By automatically scaling test cases of diversity coding tasks, we created high-quality training instances". Reported effect: "It not only significantly boosted code execution success rates, but also brought gains to other tasks." No benchmark numbers are given for this claim.
- Long-horizon RL: "we introduced long-horizon RL (Agent RL) to encourage the model to solve real-world tasks through multi-turn interactions using tools. The key challenge of Agent RL lies in environment scaling. To address this, we built a scalable system capable of running 20,000 independent environments in parallel, leveraging Alibaba Cloud's infrastructure."
- Claimed outcome: "Qwen3-Coder achieves state-of-the-art performance among open-source models on SWE-Bench Verified without test-time scaling." The blog text prints no SWE-Bench score.

## Verification
- Checked on 2026-09-15 against https://qwenlm.github.io/blog/qwen3-coder/ (post dated 2025-07-22).
- Not reported by the post: the SWE-Bench Verified number; RL algorithm, reward definition, KL settings, batch or group sizes, step counts; mid-training or SFT description; any ablation.
