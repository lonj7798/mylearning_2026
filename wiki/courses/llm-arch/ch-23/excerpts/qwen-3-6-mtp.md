# Excerpt: Qwen 3.6 — Multi-Token Prediction and Thinking Preservation

<!-- source: [[qwen-3-6|report]] -->

## Multi-Token Prediction: Beyond Speculative Decoding

Standard autoregressive language models are trained with a next-token prediction (NTP) objective: given prefix $x_1, \ldots, x_t$, predict $x_{t+1}$. Each hidden state at position $t$ is optimized to carry exactly the information needed to predict the single next token. Multi-Token Prediction (MTP) changes this: the model is trained to simultaneously predict $x_{t+1}, x_{t+2}, \ldots, x_{t+k}$ from the same prefix, using $k$ separate prediction heads attached to the final hidden states.

### The Representation Effect

The surface-level application of MTP is speculative decoding: at inference time, the model generates multiple candidate tokens in parallel, then verifies them, accepting those that match what the main prediction head would have produced sequentially. This yields wall-clock speedups proportional to the average acceptance rate.

But the deeper effect is on what the model learns to encode in its hidden states. Under NTP, the hidden state at position $t$ needs to encode information about $x_{t+1}$ — and only $x_{t+1}$. Under MTP, the same hidden state must simultaneously support predictions of $x_{t+1}$ through $x_{t+k}$. This forces the model to encode **richer, more forward-looking representations**.

Consider a concrete example. In the sequence "The function returns None if the input list is empty, otherwise it returns the sorted", the NTP-trained model at "sorted" needs to predict "list" (or "result" or "array"). The MTP-trained model at "sorted" needs to simultaneously predict the next several tokens — "list", "in", "ascending", "order" — which requires encoding not just the immediate next word but the broader semantic trajectory of the sentence.

The empirical evidence from Qwen 3.6 supports this. The 27B dense model trained with MTP achieves 94.1% on AIME'26 (vs. 92.7% for Qwen 3.5-27B without MTP) and 77.2% on SWE-bench — improvements that correlate with tasks requiring multi-step planning and sequential reasoning, exactly where richer forward-looking representations would help.

### MTP vs. NTP: The Training Tradeoff

MTP training is more expensive per step than NTP. Each training step requires computing $k$ prediction heads rather than 1, and the loss is aggregated across all $k$ predictions. The additional prediction heads add parameters and compute, though they are typically lightweight (linear projections from the final hidden state).

The tradeoff: higher per-step cost, but potentially fewer total steps needed to achieve equivalent representation quality. MTP provides a richer gradient signal per training example — each token contributes $k$ loss terms rather than 1, giving the optimizer more information to work with.

| Dimension | NTP Training | MTP Training |
|-----------|-------------|-------------|
| Loss terms per token | 1 | k |
| Gradient richness | Single-step signal | Multi-step signal |
| Hidden state requirements | Encode next token | Encode next k tokens |
| Per-step compute | 1x | ~1.3-1.5x (lightweight heads) |
| Inference mode | Sequential generation | Speculative decoding available |
| Representation quality | Optimized for local prediction | Optimized for trajectory prediction |

## Thinking Preservation: Compounding Reasoning Across Turns

### The Problem: Re-Derivation Waste

In a typical agentic coding workflow, the model operates in multi-turn mode: it reasons about the problem, issues a tool call (e.g., reads a file), receives the result, reasons about the next step, issues another tool call, and so on. Each turn involves:

1. The model generates `<think>...</think>` reasoning
2. The model generates a tool-call action
3. The tool returns a result
4. The conversation continues with a new turn

In Qwen 3 and 3.5, the thinking blocks from previous turns were discarded or compressed before the next turn. This means at turn $n$, the model must re-derive any reasoning it performed in turns $1$ through $n-1$. For a 10-turn agentic workflow, this is enormously wasteful — the model spends tokens reconstructing context it had already computed.

### The Solution: Preserve the Thinking Trace

Qwen 3.6 introduces a training objective and inference template that retains thinking blocks across conversation turns. The conversation history looks like:

```
Turn 1: <think>analysis of the bug report...</think>  [read file X]
Tool result: [contents of X]
Turn 2: <think>based on my earlier analysis AND the file contents...</think>  [read file Y]
Tool result: [contents of Y]
Turn 3: <think>combining all prior reasoning...</think>  [apply fix]
```

At turn 3, the model can attend to its reasoning from turns 1 and 2 without regenerating it. The thinking traces are part of the KV cache, already computed and stored.

### Why This Matters for Agentic Performance

The performance gap attributable to thinking preservation is visible in the benchmark data:

| Benchmark | Qwen3.6-27B | Qwen3.5-397B-A17B | Relative Gap |
|-----------|-------------|--------------------|----|
| SkillsBench | 48.2% | 30.0% | +60.7% |
| SWE-bench Pro | 53.5% | 50.9% | +5.1% |
| Terminal-Bench 2.0 | 59.3% | — | — |

SkillsBench, which tests multi-step agentic coding skills across many tool-call rounds, shows the largest improvement — 77% relative gain. This is the benchmark most sensitive to re-derivation waste, because each round builds on previous reasoning. SWE-bench Pro, which involves fewer rounds of tool interaction, shows a smaller but still meaningful gap.

The implication: for agentic applications (coding agents, research assistants, multi-step task execution), thinking preservation is not a minor optimization — it is a capability multiplier.

### Interaction with Inference Defaults

Qwen 3.6 also ships with tighter inference defaults: temperature 0.2 and top_p 0.9 (vs. Qwen 3.5's 0.6 / 0.95). These are not arbitrary choices — they address a specific failure mode observed in Qwen 3.5: **circular reasoning loops**.

With higher temperature and top_p, the model's thinking mode sometimes enters repetitive cycles — restating the same reasoning, exploring the same dead-end approaches, failing to converge on an action. Lower temperature makes the reasoning chain more deterministic, reducing the probability of looping back to already-explored reasoning paths.

The combination of thinking preservation (which eliminates the need to re-derive) and tighter sampling (which reduces circular exploration) produces more efficient reasoning chains. The model spends fewer tokens reaching the same or better conclusions.

## The Architecture-Training Separability Thesis

Qwen 3.6's results make a strong empirical case for a principle that has been theorized but rarely demonstrated this clearly: **architecture and training are independently tunable levers for model quality.**

Qwen 3.5 changed the architecture (full attention to DeltaNet hybrid) and gained context length, throughput, and multimodal capability. Qwen 3.6 kept the architecture identical and changed only the training (MTP objective, thinking preservation, refined RL), gaining agentic reasoning quality to the point where a 27B dense model beats the 397B MoE predecessor.

Neither change subsumes the other. The DeltaNet architecture enables long-context efficiency; MTP enables richer representations. They compose, suggesting that future model generations should invest in both axes simultaneously rather than treating architecture as the sole path to improvement.
