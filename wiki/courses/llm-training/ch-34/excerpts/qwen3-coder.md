---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: "Qwen3-Coder: Agentic Coding in the World (Qwen Team blog)"
source_url: https://qwenlm.github.io/blog/qwen3-coder/
created_at: "2026-09-15"
---

# Excerpt: Qwen3-Coder — execution-driven code RL and long-horizon agent RL

This excerpt stands in for the library card `qwen3-coder`, which did not exist when ch-34 was written.
Source type: official blog (Qwen Team), dated July 22, 2025. Read on 2026-09-15.

## Model
- Qwen3-Coder-480B-A35B-Instruct: 480B-parameter MoE with 35B active parameters; "supports the context length of 256K
  tokens natively and 1M tokens with extrapolation methods" (introduction).

## Pre-training (section "Pre-Training")
- "7.5T tokens (70% code ratio), excelling in coding while preserving general and math abilities."
- "Natively supports 256K context and can be extended up to 1M with YaRN."
- Qwen2.5-Coder used "to clean and rewrite noisy data".

## Post-training (section "Post-Training")
- **Code RL:** RL on "a broader set of real-world coding tasks" with automatically scaled test cases; the blog states
  that it "significantly boosted code execution success rates" and "brought gains to other tasks". No numbers and no
  list of the other tasks are given.
- **Long-horizon RL (Agent RL):** multi-turn interaction with tools on real software-engineering tasks such as
  SWE-Bench. "The key challenge of Agent RL lies in environment scaling. To address this, we built a scalable system
  capable of running 20,000 independent environments in parallel, leveraging Alibaba Cloud's infrastructure."
- Result claim: "state-of-the-art performance among open-source models on SWE-Bench Verified without test-time
  scaling". The blog text does not print the score.

## Not reported
- RL algorithm, reward definition, prompts or environments per step, number of RL steps, KL or entropy settings,
  SFT data and settings, held-out evaluation of non-coding abilities after agent RL.

## Verification
- Read on 2026-09-15 from the cached page https://qwenlm.github.io/blog/qwen3-coder/ (text sections only; the
  benchmark images were not read).
