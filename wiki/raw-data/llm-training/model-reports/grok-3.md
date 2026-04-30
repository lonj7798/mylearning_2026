<!-- scope: Grok 3 public launch post and docs; no standalone technical report
     deps: [[deepseek-r1]], [[llama-4]]
     see-also: [[deepseek-v3]], [[qwen-3]]
-->

# Grok 3
- **Core Insight:** xAI's public story for Grok 3 is not a detailed training recipe but a scaling claim: very large pretraining plus very large-scale RL for reasoning and tool-using agents.
- **Guideline:** Treat Grok 3 as a "blog-only frontier report": use it for high-level direction on test-time compute, RL-for-reasoning, and agent/tool integration, but not for reproducible hyperparameters.
- **Authors / Lab:** xAI
- **Year:** 2025
- **URL:** https://x.ai/news/grok-3
- **Relevant topics:** reasoning RL, test-time compute, long context, agents, blog-only disclosure

## Abstract
Grok 3 was introduced in February 2025 as xAI's next-generation model family, including standard Grok 3, Grok 3 mini, and reasoning variants exposed through `Think`. The launch post says the models were trained on the Colossus supercluster with roughly 10x the compute of the previous generation, use a 1M-token context window, and rely on large-scale reinforcement learning to improve chain-of-thought reasoning, backtracking, and tool use. The public disclosure is directional rather than reproducible.

## Key Contributions
- Publicly frames **large-scale RL** as the main lever for reasoning improvement.
- Pushes **test-time compute** explicitly into the product interface via **Think** mode.
- Couples reasoning with **agent/tool use**, especially **DeepSearch**, code execution, and internet access.
- Claims **1M-token context** and strong long-context performance.
- Shows a frontier-lab shift toward **blog-level disclosure** instead of a full technical report.

## Key Figures/Tables to Study
- The benchmark tables in the launch post matter mainly as positioning, not as methodology.
- The most useful parts are the product/training claims around **Think**, **DeepSearch**, and **large-scale RL**.

## Technical Details

### Family framing
- Includes **Grok 3**, **Grok 3 mini**, and reasoning variants branded as **Grok 3 (Think)** and **Grok 3 mini (Think)**.
- Public docs list **Grok 3** as an API model, but they do not expose a technical training recipe.

### Training and scale
- xAI says Grok 3 was trained on the **Colossus supercluster** with **10x the compute of the previous state of the art**.
- The launch post says reasoning capabilities were refined through **large-scale reinforcement learning**.
- It also says the models are **still in training**, which matters when interpreting benchmark claims from February 2025.

### Reasoning and test-time compute
- The public story emphasizes that Grok 3 (Think) can reason for **seconds to minutes**.
- The post attributes error correction, backtracking, and exploration of multiple approaches to the RL stage.
- Benchmark examples are presented with **cons@64** and other high test-time-compute settings, so evaluation depends materially on sampling budget.

### Context and agents
- Publicly advertised **1M-token context window**.
- xAI positions **DeepSearch** as the first reasoning agent layered on top of Grok 3.
- The post explicitly connects future API releases to **tool use, code execution, and advanced agent capabilities**.

### What is not disclosed
- No public RL algorithm name.
- No disclosed optimizer, learning rate, KL coefficient, reward design, rollout batch size, or post-training data mixture.
- No standalone technical report was available in the public sources I used.

## Connections
- [[deepseek-r1]] is the most obvious open comparison on RL-for-reasoning, but DeepSeek discloses far more of the recipe.
- [[llama-4]] is similar in that Meta also shifted to a blog-first disclosure style for its latest model wave.
- [[qwen-3]] and [[deepseek-v3]] are the better sources if you want a more reproducible 2025 public training artifact.
