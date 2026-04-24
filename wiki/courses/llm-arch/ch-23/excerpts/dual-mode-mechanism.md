# Excerpt: Dual-Mode Inference — How One Model Thinks and Doesn't Think

<!-- source: [[qwen-3|report]], [[weng-why-we-think|blog]], [[raschka-reasoning-llms|blog]] -->

## The Problem: Separate Models for Separate Behaviors

Before Qwen 3, the standard deployment pattern for reasoning-capable LLMs was to serve separate models:

- **DeepSeek-R1** for math, code, and complex reasoning (extended chain-of-thought)
- **DeepSeek-V3** for general chat, summarization, and low-latency tasks (direct response)
- Application-level routing to direct queries to the appropriate model

This required 2x GPU allocation and a routing classifier that added latency and error. Misrouting had real costs: sending a simple factual query to the reasoning model wastes compute on unnecessary thinking; sending a complex math problem to the chat model produces lower-quality answers.

## Qwen 3's Solution: Same Weights, Different Tokens

Qwen 3 eliminates the dual-model pattern. A single checkpoint operates in two modes:

### Thinking Mode
- Activated by including `<think>` in the prompt or system instruction
- Model generates an extended reasoning trace inside `<think>...</think>` delimiters
- Trace can span hundreds to thousands of tokens
- Final answer follows the closing `</think>` tag
- Higher latency, higher token cost, higher quality on complex tasks

### Non-Thinking Mode
- Default behavior (no `<think>` token)
- Model produces direct answers with no reasoning trace
- Comparable to standard chat models
- Lower latency, lower cost, adequate quality for simple tasks

### The Key Insight: No Weight Changes

The Transformer weights are identical in both modes. The model has learned, through the four-stage post-training pipeline, to recognize the `<think>` token as a behavioral switch. This is an extreme form of instruction-following — the model doesn't just follow different *content* instructions ("be formal" vs. "be casual"), it follows different *computational* instructions ("deliberate extensively" vs. "respond directly").

This works because the `<think>` token, present in the vocabulary and trained through mode fusion (post-training stage 3), acts as a contextual signal that shifts the model's generation distribution. In thinking mode, the model has high probability of generating intermediate reasoning steps, self-corrections, and structured analysis. In non-thinking mode, these tokens have near-zero probability.

## The Thinking Budget: User-Controlled Compute Allocation

Qwen 3 adds a thinking budget mechanism: users can set a maximum token count for the reasoning trace. If the model's thinking reaches this budget, it "gracefully transitions to generating a response with incomplete reasoning."

This is significant because it makes inference-time compute allocation an explicit user parameter, not just a model behavior. The budget creates a continuous spectrum:

| Budget | Behavior | Use Case |
|--------|----------|----------|
| 0 | Non-thinking mode (direct answer) | Simple Q&A, classification, low-latency |
| 128 | Brief reasoning check | Moderate confidence tasks |
| 512 | Standard reasoning | Math, code, analysis |
| 2048 | Extended deliberation | Hard problems, multi-step proofs |
| 4096+ | Maximum thinking | Research-level problems, competition math |

### Connection to Scaling Law Research

Weng's survey of test-time compute scaling ([[weng-why-we-think|blog]]) identifies a key asymmetry:

> "Easier questions benefit from purely sequential test-time compute, whereas harder questions often perform best with an optimal ratio of sequential to parallel compute."

The thinking budget maps onto this finding. For easy problems, any thinking budget is wasted compute. For moderate problems, a small budget suffices. For hard problems, longer budgets help — but with diminishing returns, and never substituting for fundamental model capability.

Weng also notes:

> "Test-time and pretraining compute are NOT 1:1 exchangeable."

This means the thinking budget cannot compensate for a weak base model. The 0.6B model in thinking mode with budget 4096 will not match the 32B model in non-thinking mode. Pre-training establishes the *ceiling*; thinking mode lets the model *approach* that ceiling more reliably.

## Comparison with Other Reasoning Approaches

### DeepSeek-R1: Separate Model, Reasoning Only
- Dedicated reasoning model, always produces chain-of-thought
- No non-thinking mode — reasoning cannot be disabled
- Requires separate V3 deployment for non-reasoning tasks
- Advantage: fully optimized for reasoning quality

### OpenAI o1: Hidden Reasoning
- Reasoning traces are not visible to the user
- Multiple internal iterations processed before final answer
- Inference-time compute is controlled by the provider, not the user
- Advantage: simpler user interface (no mode selection needed)

### Qwen 3: Unified, User-Controlled
- Single model supports both modes
- Reasoning traces are visible (thinking mode)
- User controls compute allocation via budget
- Advantage: deployment simplicity, operational flexibility

The Qwen 3 approach trades per-mode optimization for operational simplicity. The unified model is slightly worse at pure reasoning than a dedicated R1-style model, and slightly worse at direct response than a dedicated chat model. But the elimination of dual-model deployment, routing, and infrastructure complexity more than compensates for most production use cases.

## Mode Confusion: The Training Challenge

The hardest part of building a dual-mode model is preventing mode confusion during training. Failure modes include:

1. **Thinking leakage:** The model generates partial reasoning traces in non-thinking mode, adding unnecessary tokens that degrade latency without improving quality.

2. **Thinking suppression:** The model truncates reasoning in thinking mode, producing shallow analysis that fails to leverage extended computation.

3. **Mode blending:** The model generates reasoning-like content that doesn't use the `<think>` delimiters, confusing downstream systems.

4. **Quality collapse:** Both modes converge to mediocre — neither as good as dedicated thinking nor as efficient as dedicated non-thinking.

The four-stage post-training pipeline addresses these failure modes sequentially: stages 1-2 build strong thinking capability, stage 3 integrates the mode switch while preserving both behaviors, and stage 4 ensures general capability is maintained. The ordering matters — you cannot fuse modes before you have a strong thinking capability to fuse.
