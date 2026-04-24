# Excerpt: The Speculative Decoding Algorithm

<!-- source: [[speculative-decoding|paper]] — Leviathan, Kalman, Matias (2022) -->

## The Core Problem

Autoregressive decoding generates one token per forward pass through the target model. Each forward pass is memory-bandwidth-bound: the GPU loads hundreds of gigabytes of weights to perform a trivial amount of arithmetic. The arithmetic intensity during single-token decoding is approximately 0.5 FLOP/byte for FP16, versus the ~156 FLOP/byte needed to saturate an A100's compute capacity. Over 99% of the GPU's compute capability is wasted.

The key asymmetry: **generating** token $t+1$ requires knowing token $t$ (sequential), but **verifying** whether a proposed sequence of tokens is correct can be done in parallel. A forward pass that processes K tokens in parallel costs nearly the same wall-clock time as a forward pass that processes 1 token, because both are dominated by weight-loading latency.

## The Algorithm

Speculative decoding uses two models:
- **Target model** $M_p$ (large, expensive): the model whose output distribution you want to preserve
- **Draft model** $M_q$ (small, fast): proposes candidate tokens

Each iteration:

1. **Draft:** Run $M_q$ autoregressively for K steps, producing candidates $\tilde{x}_1, \ldots, \tilde{x}_K$ with draft probabilities $q(\tilde{x}_i | x_{<i})$.

2. **Verify:** Feed the entire draft sequence into $M_p$ in one forward pass, obtaining target distributions $p(x_i | x_{<i})$ at each position.

3. **Accept/Reject:** For each token in order:
   - Accept with probability $\min(1, p(\tilde{x}_i)/q(\tilde{x}_i))$
   - On rejection: sample a correction from $\text{norm}(\max(0, p(x) - q(x)))$ and discard remaining drafts

4. **Bonus:** If all K accepted, sample one additional token from $p(x_{K+1})$ for free.

## The Mathematical Guarantee

The acceptance-rejection scheme produces an output distribution **identical** to standard autoregressive decoding from $M_p$. This is exact — not an approximation.

The proof: When token $\tilde{x}$ drawn from $q$ is accepted with probability $\min(1, p(\tilde{x})/q(\tilde{x}))$, and rejected tokens are replaced by sampling from the residual distribution $\text{norm}(\max(0, p - q))$, the marginal distribution over output tokens is exactly $p$.

This means speculative decoding cannot degrade output quality. A poor draft model produces more rejections (lower speedup), but the final tokens always follow the target model's distribution.

## Expected Speedup

With constant acceptance rate $\alpha$ and draft cost ratio $c = t_\text{draft}/t_\text{target}$:

$$E[\text{tokens per iteration}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

$$\text{Speedup} \approx \frac{E[\text{tokens}]}{1 + K \cdot c}$$

Practical results from the paper: **2-3x speedup** on T5-XXL with no retraining, no architecture changes, and identical outputs.

## The CPU Analogy

The paper explicitly draws the connection to speculative execution in CPU design. Modern CPUs predict which branch an `if` statement will take and begin executing that path before the condition is evaluated. If the prediction is correct (>95% of the time for well-predicted branches), the CPU gains cycles. If wrong, it discards the speculative work and pays a small penalty.

Speculative decoding applies the same principle: the draft model "predicts" the target model's outputs. When correct, multiple tokens are generated per target-model pass. When wrong, only the draft model's cheap compute is wasted.

## Choosing the Speculation Length K

The speculation length K (number of draft tokens per iteration) involves a tradeoff:

- **Larger K** increases the ceiling for tokens per iteration ($K+1$ when all accepted) but later draft tokens have compounding error — the probability of reaching draft token $i$ without rejection is $\alpha^i$, which drops exponentially. At $\alpha = 0.7$, the probability of accepting all 8 tokens is $0.7^8 = 5.7\%$.
- **Smaller K** wastes less draft compute on rejection but limits maximum throughput.

The optimal K depends on the acceptance rate:
- High alpha (0.85+): K=6-10 is optimal — most drafts are accepted
- Medium alpha (0.65-0.85): K=4-6 balances throughput and waste
- Low alpha (<0.65): K=2-3 minimizes wasted draft computation

Production systems increasingly use **adaptive K**, adjusting speculation length dynamically based on recent acceptance rates within a single generation.

## The Correction Distribution

A subtle but critical detail: when a draft token is rejected, the algorithm does not simply sample from $p(x)$. It samples from the **residual** distribution:

$$p_\text{corrected}(x) = \text{norm}(\max(0,\; p(x) - q(x)))$$

This distribution concentrates probability mass on tokens where $p(x) > q(x)$ — tokens the target model thinks are more likely than the draft model does. This is what makes the overall procedure exact: the combination of sometimes-accepting-from-$q$ and sometimes-sampling-from-the-residual yields exactly $p$.

If you naively sampled from $p(x)$ on rejection instead, the output distribution would be biased toward tokens with high $q(x)$ (because those are accepted more often) without the correction term to compensate.

## Why This Matters

Before speculative decoding, the only options for faster inference were:
- Quantization (reduces quality)
- Smaller models (reduces quality)
- KV-cache optimization (reduces memory, not latency per token)
- Batching (reduces per-token cost but not latency)

Speculative decoding is the first technique to reduce per-token latency without any quality degradation. It is now integrated into vLLM, TensorRT-LLM, and every major serving framework.
