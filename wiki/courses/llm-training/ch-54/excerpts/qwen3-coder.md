<!-- chapter excerpt for ch-54. Primary-source extract, read 2026-09-17.
     If a library card of this slug exists under wiki/raw-data/llm-training/, prefer the card. -->

# Qwen3-Coder: Agentic Coding in the World
- **Artifact:** Qwen Team, official blog post, 2025-07-22 (qwenlm.github.io / qwen.ai blog). Source type: official blog. Read from the page text cached 2026-09-17.
- **Core Insight:** The stated bottleneck for long-horizon agent RL is environment scaling, and the team reports building a system that runs 20,000 independent environments in parallel on Alibaba Cloud infrastructure to supply feedback for RL and for evaluation at scale (§"Scaling Long-Horizon RL").
- **Guideline:** When the RL task is multi-turn interaction with a real environment, plan environment capacity as a first-class system requirement alongside GPU capacity, because the post attributes its SWE-Bench Verified result among open models to that environment scale rather than to test-time scaling (§"Scaling Long-Horizon RL").

## Technical details (with loci)
- **Model.** Qwen3-Coder-480B-A35B-Instruct: 480B-parameter mixture of experts with 35B active parameters, 256K native context, extended to 1M with YaRN (§"Qwen3-Coder").
- **Pre-training (§"Pre-Training").** 7.5T tokens with a 70% code ratio; Qwen2.5-Coder used to clean and rewrite noisy data; context optimized for repository-scale and dynamic data such as pull requests.
- **Code RL (§"Scaling Code RL").** The team states that all code tasks suit execution-driven RL, and that automatically scaling test cases across diverse coding tasks produced the training instances; they report that this raised code execution success rates and also gave gains on other tasks, and describe "hard to solve, easy to verify" as the property they look for.
- **Long-horizon RL (§"Scaling Long-Horizon RL").** In real software-engineering tasks the model must plan, call tools, receive feedback, and decide over many turns. The key challenge named is environment scaling; the reported answer is a system running 20,000 independent environments in parallel. The post claims state-of-the-art performance among open-source models on SWE-Bench Verified without test-time scaling.
- **Tooling.** Qwen Code, a CLI adapted from Gemini CLI with a modified parser and function-calling protocol.

## Not reported
No per-benchmark numbers in the section on long-horizon RL, no RL algorithm, no reward definition, no environment-failure handling, no throughput or cost figures, no ablation of environment count.
