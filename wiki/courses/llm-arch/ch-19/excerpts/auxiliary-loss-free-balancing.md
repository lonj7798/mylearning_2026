# Excerpt: Auxiliary-Loss-Free Load Balancing

<!-- source: [[deepseek-v3|report]], Section 2.2 -->

## The Problem

Standard MoE load balancing uses auxiliary losses added to the training objective:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{LM} + \alpha \cdot \mathcal{L}_\text{balance}$$

The balance loss $\mathcal{L}_\text{balance}$ penalizes uneven expert utilization. DeepSeek-V2 used three such losses simultaneously:

| Loss | Coefficient ($\alpha$) | Target |
|------|----------------------|--------|
| Expert-level balance | 0.003 | Even token count across experts |
| Device-level balance | 0.05 | Even compute across GPUs |
| Communication balance | 0.02 | Even cross-node traffic |

The problem: these loss terms inject gradient signal that **conflicts with language modeling**. The gating network receives gradients telling it both "route to the expert that gives the best prediction" *and* "route to the underloaded expert." These objectives are fundamentally at odds. Higher $\alpha$ means more balanced routing but worse language modeling quality. Lower $\alpha$ risks routing collapse.

## The Bias-Based Solution

DeepSeek-V3 replaces auxiliary losses with a non-gradient control mechanism:

```
For each expert e:
    routing_score_e = gate(x)_e + bias_e      # bias shifts score, does NOT enter loss
    
After each training step:
    for each expert e:
        if actual_load_e > target_load:
            bias_e -= gamma                     # discourage routing to overloaded expert
        else:
            bias_e += gamma                     # encourage routing to underloaded expert
```

Key properties:

1. **No gradient contamination.** The bias is added to the routing score but excluded from the loss computation. The gating network's gradients come purely from $\mathcal{L}_\text{LM}$.

2. **Control-theoretic framing.** The bias acts as an integral controller. It accumulates error (deviation from target load) over time and nudges the system toward balance. The speed parameter $\gamma$ controls the controller's aggressiveness.

3. **Works at the margin.** For tokens where the gating network strongly prefers one expert, the bias has negligible effect (the score gap is larger than the bias). For tokens where multiple experts score similarly, the bias tips the balance toward the underloaded expert. This means the bias primarily redirects tokens that the model is indifferent about — exactly the right behavior.

## Results

- Zero irrecoverable loss spikes across 14.8 trillion tokens of training
- Zero training rollbacks
- No token dropping required (DeepSeek-V2 dropped tokens at overloaded experts)
- Model quality improvement over auxiliary-loss-based V2

## Why It Matters

The auxiliary-loss approach conflates two orthogonal concerns:
- **Learning** (predict the next token well) — a gradient-based optimization problem
- **Systems** (distribute compute evenly) — a scheduling/control problem

Treating load balancing as a control problem rather than a learning problem is a cleaner decomposition. It recognizes that balanced routing is a *constraint* the system must satisfy, not an *objective* the model should learn.

A complementary sequence-wise auxiliary loss with an extremely small coefficient remains as a safety mechanism, but it functions as a regularizer rather than the primary balancing tool.

## Connection to [[ch-14]]

Chapter 14 covers the general MoE load balancing problem: auxiliary losses, capacity factors, token dropping, and routing collapse. DeepSeek-V3's bias-based approach represents the state-of-the-art solution — it decouples the balancing mechanism from the training loss entirely, which is the direction the field is moving.
