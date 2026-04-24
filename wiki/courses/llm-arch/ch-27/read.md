# Chapter 27: Speculative Decoding

<!-- scope: draft-verify decoding, Medusa heads, multi-token prediction, acceptance rate analysis
     deps: [[ch-25]]
     see-also: [[ch-19]], [[ch-26]]
-->

## Overview

Autoregressive decoding generates one token per forward pass. Each forward pass through a 70B model requires loading hundreds of gigabytes of weights from HBM, performing a matrix multiply that takes microseconds, and writing back one token's worth of KV cache. The arithmetic intensity is absurdly low — the GPU's tensor cores sit idle while the memory bus delivers parameters. As [[ch-25]] established, decoding is memory-bandwidth-bound: the bottleneck is bytes moved, not FLOPs computed.

This creates a fundamental problem. A single forward pass through Llama 3 70B takes roughly the same wall-clock time whether it processes 1 token or 1,000 tokens in a batch, because the cost is dominated by loading weights, not by the matrix arithmetic on those weights. Yet autoregressive decoding insists on running the model sequentially — one token at a time, each conditioned on all previous tokens. A 500-token response requires 500 serial forward passes. The hardware can verify a sequence of 10 candidate tokens in nearly the same time it takes to generate one, but standard decoding never exploits this.

Speculative decoding ([[speculative-decoding|paper]]) breaks this sequential bottleneck with a simple insight borrowed from CPU architecture: **speculate, then verify**. A small, fast draft model proposes multiple tokens; the large target model checks them all in a single parallel forward pass. When the draft model guesses correctly — which it often does for predictable tokens — you get multiple tokens for the cost of one target-model pass. When it guesses wrong, you pay only the cost of the draft model's wasted work, which is cheap by definition.

This chapter covers the original speculative decoding algorithm and its mathematical guarantee of identical output distributions, then examines three directions the field has taken: self-drafting architectures like Medusa that eliminate the separate draft model, DeepSeek's multi-token prediction (MTP) that bridges training-time objectives with inference-time speedup, and the empirical analysis of when speculative decoding actually helps — which turns out to depend critically on the entropy of the text being generated.

---

## 1. Why Autoregressive Decoding Wastes Hardware

To understand speculative decoding, you must first understand exactly *why* standard decoding is slow — and why the solution is not simply "use a faster GPU."

### The Arithmetic Intensity Problem

During the prefill phase (processing the prompt), the model processes all tokens simultaneously. The batch of tokens provides enough arithmetic work to keep the GPU's compute units busy while weights stream from HBM. This is a **compute-bound** operation — adding more memory bandwidth wouldn't help much.

During the decode phase (generating tokens one at a time), the situation reverses. Each step performs one matrix-vector multiply per layer — a tiny amount of arithmetic relative to the bytes loaded:

$$\text{Arithmetic intensity} = \frac{\text{FLOPs}}{\text{Bytes loaded}} = \frac{2 \times d_\text{model}}{2 \times d_\text{model} \times \text{precision}} = \frac{1}{\text{precision bytes}}$$

For FP16 weights, the arithmetic intensity is 0.5 FLOP/byte. An A100 has 312 TFLOPS of FP16 compute but only 2 TB/s of HBM bandwidth. To saturate the compute, you need an arithmetic intensity of 312/2 = 156 FLOPs/byte. Single-token decoding achieves 0.5 — it utilizes **0.3%** of the GPU's compute capability.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">GPU Utilization: Prefill vs Decode (A100, 70B model)</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Phase</th>
<th style="text-align:right; padding:8px;">Tokens/step</th>
<th style="text-align:right; padding:8px;">Arithmetic Intensity</th>
<th style="text-align:right; padding:8px;">Bottleneck</th>
<th style="text-align:right; padding:8px;">Compute Util</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">Prefill (prompt)</td>
<td style="text-align:right; padding:8px;">512-4096</td>
<td style="text-align:right; padding:8px;">~128 FLOP/byte</td>
<td style="text-align:right; padding:8px;">Compute</td>
<td style="text-align:right; padding:8px;">60-80%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Decode (generation)</td>
<td style="text-align:right; padding:8px;">1</td>
<td style="text-align:right; padding:8px;">0.5 FLOP/byte</td>
<td style="text-align:right; padding:8px;">Memory BW</td>
<td style="text-align:right; padding:8px;">~0.3%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">Spec. decode verify</td>
<td style="text-align:right; padding:8px;">5-10</td>
<td style="text-align:right; padding:8px;">~5 FLOP/byte</td>
<td style="text-align:right; padding:8px;">Memory BW</td>
<td style="text-align:right; padding:8px;">~2-3%</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
Verification of K draft tokens costs nearly the same wall-clock time as generating 1 token, because both are bandwidth-bound.<br>
The extra FLOPs from K tokens are absorbed by idle compute capacity.
</div>
</div>

### The Key Observation: Verification is Parallel, Generation is Sequential

This is the asymmetry that speculative decoding exploits. **Generating** the next token requires the previous token's output — it is inherently sequential. But **verifying** whether a sequence of K candidate tokens is correct can be done in a single forward pass: you feed all K candidates into the target model simultaneously (like a prefill), and compare the model's output distribution at each position against what the draft model proposed.

Since verification processes K tokens together, its arithmetic intensity scales with K. At K=8, verification is roughly 8x more compute-efficient per token than single-token generation — still bandwidth-bound, but far less wasteful. And because the model weights must be loaded from HBM regardless, verifying 8 tokens costs nearly the same wall-clock time as generating 1.

---

## 2. The Speculative Decoding Algorithm

Leviathan, Kalman, and Matias (2022) ([[speculative-decoding|paper]]) formalized this idea with a precise algorithm that guarantees **identical output distributions** to standard autoregressive decoding. This is not an approximation — it is an exact speedup technique.

### Setup

You need two models:
- **Target model** $M_p$: the large model whose output quality you want to preserve (e.g., 70B parameters)
- **Draft model** $M_q$: a smaller, faster model from the same family or a distilled variant (e.g., 7B parameters)

Both models share the same vocabulary. The draft model is typically 10-20x faster per token than the target model.

### The Draft-Verify Cycle

Each iteration of speculative decoding performs:

**Step 1 — Draft.** Run $M_q$ autoregressively for $K$ steps, generating candidate tokens $\tilde{x}_1, \tilde{x}_2, \ldots, \tilde{x}_K$ and recording the draft probabilities $q(\tilde{x}_i \mid x_{<i})$ at each step.

**Step 2 — Verify.** Feed the entire draft sequence into $M_p$ in a single forward pass. This produces the target model's probability distributions $p(x_i \mid x_{<i})$ at each of the $K$ positions.

**Step 3 — Accept/Reject.** For each draft token $\tilde{x}_i$ in sequence:
- Compute the acceptance probability: $\min\!\left(1, \;\frac{p(\tilde{x}_i \mid x_{<i})}{q(\tilde{x}_i \mid x_{<i})}\right)$
- Draw $r \sim \text{Uniform}(0, 1)$
- If $r < $ acceptance probability: **accept** $\tilde{x}_i$ and continue to the next token
- If $r \geq $ acceptance probability: **reject** $\tilde{x}_i$, sample a correction token from the adjusted distribution $\text{norm}\!\left(\max(0,\; p(x) - q(x))\right)$, and discard all subsequent draft tokens

**Step 4 — Bonus token.** If all $K$ tokens are accepted, sample one additional token from $p(x_{K+1} \mid x_{\leq K})$, which is available for free from the verification forward pass.

The result: each iteration produces between 1 (all rejected after the first) and $K+1$ (all accepted plus bonus) tokens, using exactly one target-model forward pass plus $K$ draft-model forward passes.

### The Mathematical Guarantee

The acceptance-rejection sampling scheme is carefully constructed so that the marginal distribution of each accepted token is exactly $p(x)$ — the target model's distribution. The proof relies on a standard result from rejection sampling:

When a draft token $\tilde{x}$ with probability $q(\tilde{x})$ is accepted with probability $\min(1, p(\tilde{x})/q(\tilde{x}))$, and rejected tokens are replaced by sampling from $\text{norm}(\max(0, p(x) - q(x)))$, the resulting distribution over output tokens is exactly $p(x)$.

**Why this matters:** Speculative decoding is not a quality-latency tradeoff. It produces *identical* outputs to standard decoding (for a given random seed). Any text that standard decoding would generate, speculative decoding generates too — just faster. This is unlike quantization ([[ch-26]]), which explicitly trades quality for speed.

[Interactive visualization: [Draft-Verify Pipeline Animation](figures/draft-verify-pipeline.html)]

### Why the Acceptance Rate Determines Everything

The expected number of tokens per iteration is:

$$E[\text{tokens}] = \sum_{i=1}^{K} \prod_{j=1}^{i} \alpha_j + 1$$

where $\alpha_j$ is the acceptance probability at position $j$. If we simplify by assuming a constant acceptance rate $\alpha$:

$$E[\text{tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

The wall-clock speedup depends on the ratio of draft cost to target cost. Let $c = t_\text{draft} / t_\text{target}$ (typically 0.05-0.1 for a 10-20x smaller draft model):

$$\text{Speedup} \approx \frac{E[\text{tokens}]}{1 + K \cdot c}$$

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Speedup vs Acceptance Rate (K=5, draft cost ratio c=0.1)</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Acceptance Rate (alpha)</th>
<th style="text-align:right; padding:8px;">E[tokens/iter]</th>
<th style="text-align:right; padding:8px;">Cost (target passes)</th>
<th style="text-align:right; padding:8px;">Speedup</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560;">0.5</td>
<td style="text-align:right; padding:8px;">1.97</td>
<td style="text-align:right; padding:8px;">1.5</td>
<td style="text-align:right; padding:8px;">1.31x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d;">0.7</td>
<td style="text-align:right; padding:8px;">3.08</td>
<td style="text-align:right; padding:8px;">1.5</td>
<td style="text-align:right; padding:8px;">2.05x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4;">0.85</td>
<td style="text-align:right; padding:8px;">4.74</td>
<td style="text-align:right; padding:8px;">1.5</td>
<td style="text-align:right; padding:8px;">3.16x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">0.95</td>
<td style="text-align:right; padding:8px;">5.69</td>
<td style="text-align:right; padding:8px;">1.5</td>
<td style="text-align:right; padding:8px;">3.79x</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
At alpha=0.5, speculative decoding barely helps. At alpha=0.85+, you approach 3x+ speedup.<br>
The crossover point where spec. decoding beats standard decoding is roughly alpha > 0.4 for K=5, c=0.1.
</div>
</div>

[Interactive visualization: [Acceptance Rate Explorer](figures/acceptance-rate-explorer.html)]

---

## 3. Draft Model Selection: The Quality-Speed Tradeoff

The draft model must be simultaneously *fast enough* to justify the speculation overhead and *accurate enough* to achieve high acceptance rates. These goals are in tension.

### Same-Family Drafting

The most common approach uses a smaller model from the same family as the target:

| Target | Draft | Size Ratio | Typical alpha |
|--------|-------|-----------|---------------|
| Llama 3 70B | Llama 3 8B | 8.75x | 0.7-0.85 |
| GPT-4 | GPT-3.5 | ~10x | 0.6-0.8 |
| Gemini Ultra | Gemini Flash | ~10x | 0.7-0.85 |

Same-family drafting works well because the models share training data and vocabulary, so their probability distributions tend to agree on predictable tokens. The acceptance rate is highest for tokens where the target model is confident (high $p(\tilde{x})$), which are precisely the tokens the draft model is also likely to get right.

### Independently Trained Drafts

You can use any model as the draft, as long as it shares the vocabulary. The mathematical guarantee holds regardless — the acceptance-rejection scheme corrects for any distributional mismatch. However, the acceptance rate suffers when the draft and target models disagree systematically, which happens more with independently trained models.

### The Speculation Length K

Choosing K involves a tradeoff:
- **Larger K** = more draft tokens per iteration = higher potential speedup, but later draft tokens have lower acceptance rates (errors compound) and more wasted draft compute on rejection
- **Smaller K** = less wasted work on rejection, but lower ceiling for speedup

In practice, K=4-8 works best. The optimal K depends on the acceptance rate: when alpha is high (predictable text), larger K helps; when alpha is low (creative/reasoning text), smaller K reduces waste. Some serving systems adapt K dynamically based on recent acceptance rates.

---

## 4. Medusa: Self-Drafting with Multiple Prediction Heads

Medusa (Cai et al., 2024) eliminates the separate draft model entirely by adding **multiple prediction heads** to the target model itself. Instead of maintaining and running two separate models, Medusa attaches lightweight MLP heads that predict tokens 2, 3, ..., K+1 positions ahead.

### Architecture

The base model's final hidden state $h_t$ at position $t$ feeds into K additional heads:

$$\text{Head}_k(h_t) = W_k^{(2)} \cdot \text{SiLU}(W_k^{(1)} \cdot h_t) \qquad k = 1, \ldots, K$$

Each head is a single-hidden-layer MLP with a residual connection from $h_t$. Head $k$ predicts the token at position $t + k + 1$ (the original LM head predicts position $t + 1$).

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Medusa: Multiple Prediction Heads</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:12px;">
<div style="background:#16213e; padding:10px 40px; border-radius:8px; color:#e0e0e0; font-size:13px; font-weight:bold;">
Transformer backbone (frozen or fine-tuned)
</div>
<div style="color:#e94560; font-size:18px;">&#8595;</div>
<div style="background:#0f3460; padding:10px 30px; border-radius:8px; color:#4ecdc4; font-size:13px; font-weight:bold;">
h_t (final hidden state at position t)
</div>
<div style="display:flex; gap:16px; margin-top:8px;">
<div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
<div style="color:#888; font-size:11px;">&#8595;</div>
<div style="background:#e94560; padding:6px 14px; border-radius:6px; color:#fff; font-size:11px; font-weight:bold;">LM Head</div>
<div style="color:#fff; font-size:10px;">t+1</div>
</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
<div style="color:#888; font-size:11px;">&#8595;</div>
<div style="background:#ffd93d; padding:6px 14px; border-radius:6px; color:#1a1a2e; font-size:11px; font-weight:bold;">Medusa 1</div>
<div style="color:#ffd93d; font-size:10px;">t+2</div>
</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
<div style="color:#888; font-size:11px;">&#8595;</div>
<div style="background:#ffd93d; padding:6px 14px; border-radius:6px; color:#1a1a2e; font-size:11px; font-weight:bold;">Medusa 2</div>
<div style="color:#ffd93d; font-size:10px;">t+3</div>
</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
<div style="color:#888; font-size:11px;">&#8595;</div>
<div style="background:#ffd93d; padding:6px 14px; border-radius:6px; color:#1a1a2e; font-size:11px; font-weight:bold;">Medusa 3</div>
<div style="color:#ffd93d; font-size:10px;">t+4</div>
</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
<div style="color:#888; font-size:11px;">&#8595;</div>
<div style="background:#ffd93d; padding:6px 14px; border-radius:6px; color:#1a1a2e; font-size:11px; font-weight:bold;">Medusa 4</div>
<div style="color:#ffd93d; font-size:10px;">t+5</div>
</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center;">
Each Medusa head is a small MLP (~0.5% of total params) predicting a future token position.<br>
The original LM head is unchanged. Heads share the same hidden state h_t as input.
</div>
</div>

### Tree-Structured Verification

Because each Medusa head produces a distribution over the vocabulary (not a single token), the top-$s$ candidates from each head can be combined into a **tree of candidate sequences**. For example, with $s = 3$ candidates per head and $K = 4$ heads, the tree contains up to $3^4 = 81$ candidate paths — but many can be verified simultaneously using a carefully constructed attention mask.

The tree attention mask ensures that each candidate token attends only to its ancestors in the tree, not to siblings. This lets the target model score all candidate paths in a single forward pass. The path with the longest prefix of accepted tokens is selected.

This tree structure is what gives Medusa its speedup advantage over linear draft sequences: instead of a single chain of K draft tokens (where one rejection kills the rest), you have multiple alternative continuations. If the first candidate for position $t+2$ is rejected, a second or third candidate may be accepted, salvaging the downstream predictions.

### Training Medusa Heads

Two training strategies:

**Medusa-1 (head-only training):** Freeze the backbone, train only the Medusa heads on the model's own output distributions. This is fast (hours on a few GPUs) and preserves the base model's quality perfectly. However, the heads are limited by the information in $h_t$ — they can only predict future tokens based on features the original model happened to learn, without any ability to reshape those features for multi-token prediction.

**Medusa-2 (joint fine-tuning):** Fine-tune the backbone along with the heads. This allows the model to reshape its hidden representations to support multi-token prediction, yielding higher acceptance rates. The cost is a full fine-tuning run and the risk of degrading the base model's quality if not done carefully.

### Tradeoffs vs Standard Speculative Decoding

| Dimension | Speculative Decoding | Medusa |
|-----------|---------------------|--------|
| Extra model needed | Yes (draft model) | No |
| Extra memory | Draft model weights + KV cache | K small MLP heads (~0.5% params) |
| Deployment complexity | Two models, orchestration | Single model, modified heads |
| Output distribution | Identical to target | Relaxed (top-k truncation) |
| Acceptance rates | Depends on draft quality | Depends on head training |
| Typical speedup | 2-3x | 1.5-2.5x |

The critical distinction: standard speculative decoding preserves the exact output distribution. Medusa, in its practical form with tree attention and top-k truncation, introduces a slight approximation. For many applications this is acceptable, but it means Medusa is not a drop-in replacement when distributional exactness matters.

---

## 5. Multi-Token Prediction: Training for Faster Inference

DeepSeek-V3 ([[deepseek-v3|report]]) introduced Multi-Token Prediction (MTP) as a *training-time* objective that directly bridges to inference-time speculative decoding. Instead of adding heads post-hoc (Medusa) or maintaining a separate draft model, MTP trains the model from the start to predict multiple future tokens.

### The MTP Architecture

DeepSeek-V3's MTP uses $D$ sequential modules, each predicting the next token in the sequence. Critically, these modules maintain a **causal chain** — each MTP module takes as input the embedding of the token predicted by the previous module, combined with the main model's hidden state:

$$h_t^{(k)} = \text{MTPModule}_k\!\left(\text{concat}(h_t^{(k-1)},\; \text{embed}(\hat{x}_{t+k}))\right)$$

where $h_t^{(0)}$ is the main model's hidden state at position $t$, and $\hat{x}_{t+k}$ is the ground-truth token at position $t+k$ (during training, via teacher forcing).

This causal chain is what distinguishes MTP from Medusa's independent heads. Medusa head $k$ predicts token $t+k$ using only $h_t$ — it has no information about what tokens $t+1$ through $t+k-1$ actually are. MTP module $k$ receives the embeddings of the intervening tokens, allowing it to make more informed predictions.

### The Combined Training Loss

The total loss combines the standard next-token prediction with MTP:

$$\mathcal{L} = \mathcal{L}_\text{NTP} + \lambda \sum_{k=1}^{D} \mathcal{L}_\text{MTP}^{(k)}$$

where $\lambda$ is a weighting factor (DeepSeek-V3 uses $\lambda = 0.3$) and each $\mathcal{L}_\text{MTP}^{(k)}$ is a cross-entropy loss for predicting the token $k$ positions ahead.

### Why MTP Improves Training Quality

A counterintuitive result: MTP doesn't just enable faster inference — it actually improves the model's performance on standard benchmarks, even when MTP modules are discarded at inference time. The DeepSeek-V3 ablations showed consistent improvements on reasoning and code tasks.

The explanation: predicting multiple future tokens forces the model to build richer internal representations. To predict token $t+3$ correctly, the model must develop latent features at position $t$ that capture not just the immediate next token but the trajectory of the sequence. This acts as a form of implicit planning — the model learns to "look ahead" even when only asked for one token.

### MTP for Speculative Decoding at Inference

At inference time, DeepSeek-V3's MTP modules serve as an integrated draft mechanism:

1. The main model generates token $t+1$ via the standard LM head
2. MTP module 1 proposes token $t+2$ using $h_t$ and the embedding of token $t+1$
3. MTP module 2 proposes token $t+3$ using the output of MTP module 1 and the embedding of the proposed $t+2$
4. All proposals are verified in the next forward pass

DeepSeek-V3 reports a **1.8x speedup** from MTP-based speculative decoding. This is more modest than the 2-3x from external draft models, but comes with zero additional memory for a separate model and zero deployment complexity.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Speculative Decoding Approaches Compared</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Approach</th>
<th style="text-align:right; padding:8px;">Extra Params</th>
<th style="text-align:right; padding:8px;">Training Cost</th>
<th style="text-align:right; padding:8px;">Exact Dist?</th>
<th style="text-align:right; padding:8px;">Typical Speedup</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">External draft model</td>
<td style="text-align:right; padding:8px;">Full small model</td>
<td style="text-align:right; padding:8px;">None (off-the-shelf)</td>
<td style="text-align:right; padding:8px;">Yes</td>
<td style="text-align:right; padding:8px;">2-3x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">Medusa heads</td>
<td style="text-align:right; padding:8px;">~0.5% of target</td>
<td style="text-align:right; padding:8px;">Hours (head-only)</td>
<td style="text-align:right; padding:8px;">Approx.</td>
<td style="text-align:right; padding:8px;">1.5-2.5x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">MTP (DeepSeek)</td>
<td style="text-align:right; padding:8px;">MTP modules</td>
<td style="text-align:right; padding:8px;">Integrated in pre-train</td>
<td style="text-align:right; padding:8px;">Approx.</td>
<td style="text-align:right; padding:8px;">1.8x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#888; font-weight:bold;">EAGLE (layer reuse)</td>
<td style="text-align:right; padding:8px;">~2% of target</td>
<td style="text-align:right; padding:8px;">Days</td>
<td style="text-align:right; padding:8px;">Approx.</td>
<td style="text-align:right; padding:8px;">2-3x</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
EAGLE reuses early layers of the target model as a draft mechanism. Included for completeness.
</div>
</div>

---

## 6. When Speculative Decoding Helps — and When It Doesn't

The acceptance rate $\alpha$ depends on how well the draft model's distribution matches the target model's distribution. This match varies dramatically across different types of text.

### High-Entropy vs Low-Entropy Tokens

**Low-entropy tokens** (the target model is confident): deterministic completions, boilerplate code, common phrases, syntactic structure. Example: after "The United States of", the next token is almost certainly "America". Both draft and target models agree, so $\alpha \approx 1.0$.

**High-entropy tokens** (the target model is uncertain): creative word choices, reasoning steps, code logic, rare factual knowledge. Example: choosing between semantically valid but different continuations. The draft and target models are more likely to disagree, so $\alpha$ drops.

### Domain-Dependent Speedups

| Domain | Typical Alpha | Speedup | Why |
|--------|--------------|---------|-----|
| Code completion | 0.8-0.9 | 2.5-3x | Syntax is highly predictable; variable names repeat |
| Translation | 0.75-0.85 | 2-2.5x | Grammar constrains structure |
| Summarization | 0.7-0.8 | 1.8-2.2x | Content is anchored by source text |
| Creative writing | 0.5-0.65 | 1.2-1.6x | Many valid continuations |
| Mathematical reasoning | 0.5-0.7 | 1.3-1.8x | Reasoning steps have high entropy |
| Chain-of-thought | 0.55-0.7 | 1.3-1.8x | Thinking tokens are unpredictable |

The pattern: speculative decoding helps most when the text is **predictable** — when a small model can accurately anticipate what the large model would generate. For tasks where the large model's superior capabilities are most needed (complex reasoning, nuanced generation), those are precisely the tokens where the draft model disagrees most, reducing the benefit.

### The Reasoning Model Tension

This creates an interesting tension with the trend toward reasoning models ([[raschka-reasoning-llms|blog]], [[weng-why-we-think|blog]]). Reasoning models like DeepSeek-R1 and OpenAI o1 generate long chain-of-thought traces where many tokens are "thinking" tokens — exploratory reasoning, backtracking, self-correction. These tokens tend to be high-entropy: the model is genuinely uncertain about the reasoning path. A draft model is unlikely to predict the same reasoning trajectory.

However, there's a counterpoint: much of the verbosity in chain-of-thought reasoning is structural ("Let me think about this differently...", "Wait, that can't be right because..."). These scaffolding tokens are predictable even if the substantive reasoning tokens aren't. Empirically, speculative decoding still provides 1.3-1.8x speedup on reasoning tasks — less than code completion, but still worthwhile given the extremely long outputs.

### Batch Size Interaction

Speculative decoding's benefit decreases with larger batch sizes. At batch size 1, decoding is maximally bandwidth-bound — the perfect scenario for speculation. As batch size increases, the arithmetic intensity of standard decoding rises (more tokens share the weight-loading cost), reducing the waste that speculative decoding exploits.

At batch sizes above 16-32 (depending on model size and hardware), standard decoding may already be approaching compute-bound territory, and the overhead of running a draft model can exceed the speedup. This is why speculative decoding is most valuable for **latency-sensitive, low-batch settings** — interactive chat, code completion, real-time applications — rather than high-throughput batch processing.

---

## 7. Integration with Serving Systems

Speculative decoding is now standard in vLLM, TensorRT-LLM, and every major serving framework. The main implementation challenge is **KV cache management**: draft tokens that are rejected waste memory because their KV entries were computed during verification but must be discarded. PagedAttention ([[ch-25]]) mitigates this — rejected blocks can be freed immediately from the paged memory pool. Production systems also implement **adaptive K**, dynamically adjusting speculation length based on recent acceptance rates: when rejections spike, K decreases to reduce wasted draft compute; when acceptance is high, K increases to maximize throughput.

---

## Core Insights from the Literature

### Insight 1: Verification is parallelizable even though generation is not
**Paper:** Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" ([[speculative-decoding|paper]])

The fundamental asymmetry of autoregressive models: generating token $t+1$ requires knowing token $t$ (sequential), but checking whether a proposed token $t+1$ is correct can be done alongside checking tokens $t+2, \ldots, t+K$ (parallel). This asymmetry exists because the target model's forward pass can process multiple positions simultaneously — the same property that makes prefill fast. Speculative decoding reframes decode-phase tokens as a mini-prefill of draft candidates. **Guideline:** When a sequential process has a cheap-to-run approximation, consider speculate-and-verify as an acceleration strategy. The principle extends beyond LLMs — any sequential computation where verification is cheaper than generation is a candidate.

### Insight 2: Distributional exactness comes from rejection sampling, not from draft quality
**Paper:** Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" ([[speculative-decoding|paper]])

The acceptance-rejection scheme guarantees that accepted tokens follow the target model's distribution regardless of how bad the draft model is. A terrible draft model simply means more rejections (lower $\alpha$), not different outputs. This separation of correctness from efficiency is what makes speculative decoding deployable in production — you can't accidentally degrade quality by choosing the wrong draft model. You can only fail to achieve a speedup. **Guideline:** Always verify that your speculative decoding implementation uses the full acceptance-rejection scheme with the correction distribution $\text{norm}(\max(0, p - q))$. Simplified schemes that skip the correction step *do* change the output distribution.

### Insight 3: Multi-token prediction during training improves single-token quality
**Paper:** DeepSeek AI, "DeepSeek-V3 Technical Report" ([[deepseek-v3|report]])

Training with an MTP objective forces the model to develop richer representations that capture not just the immediate next token but future trajectory. DeepSeek-V3's ablations show that MTP improves standard benchmark performance even when MTP modules are discarded at inference. This suggests that next-token prediction alone provides a suboptimal training signal — the model benefits from being forced to plan ahead. **Guideline:** If you're pre-training a model from scratch and plan to use speculative decoding at inference, integrate MTP into the training objective rather than bolting on draft mechanisms post-hoc. The training-quality improvement alone may justify the additional compute.

### Insight 4: Acceptance rates are determined by text entropy, not model quality
**Sources:** Leviathan et al. ([[speculative-decoding|paper]]), DeepSeek-V3 ([[deepseek-v3|report]])

The draft-target agreement depends primarily on how predictable the tokens are, not on how good either model is in absolute terms. Boilerplate code has high acceptance rates even with a weak draft model; novel reasoning has low acceptance rates even with a strong draft model. This means speculative decoding's benefit is domain-dependent and partially task-dependent. **Guideline:** Profile acceptance rates on your actual workload before committing to speculative decoding infrastructure. Code-heavy and structured-output workloads benefit most; open-ended reasoning benefits least.

---

## Key Takeaways

1. **Autoregressive decoding wastes >99% of GPU compute.** Single-token generation has an arithmetic intensity of 0.5 FLOP/byte on FP16, versus the ~156 FLOP/byte needed to saturate an A100. The GPU is a memory-bandwidth delivery truck being used to carry one item at a time.

2. **Speculative decoding achieves speedup without quality loss.** The acceptance-rejection sampling scheme produces *identical* output distributions to standard decoding. This is fundamentally different from approximation techniques like quantization — there is no quality-speed tradeoff.

3. **The acceptance rate is the single most important metric.** Expected tokens per iteration scales as $(1 - \alpha^{K+1})/(1 - \alpha)$. Below $\alpha \approx 0.4$, speculative decoding hurts. Above $\alpha \approx 0.8$, it provides 2.5-3x speedup. Everything in between depends on the draft model's cost ratio.

4. **Three architectural approaches, different tradeoffs.** External draft models (exact, 2-3x, requires separate model), Medusa heads (approximate, 1.5-2.5x, no extra model), MTP (approximate, 1.8x, improves training quality). The trend is toward integrated approaches that eliminate deployment complexity.

5. **Speculative decoding helps most for predictable text.** Code completion (0.8-0.9 alpha) benefits far more than creative writing (0.5-0.65 alpha). This creates a tension with reasoning models, which generate high-entropy thinking tokens where draft accuracy is lowest.

6. **Batch size is the enemy of speculation.** At high batch sizes, standard decoding already amortizes weight-loading costs across many tokens, reducing the bandwidth waste that speculative decoding exploits. Speculation is primarily a latency optimization for interactive, low-batch settings.

7. **MTP bridges training and inference optimization.** DeepSeek-V3 shows that training with a multi-token objective both improves model quality (richer representations) and enables integrated speculative decoding at inference (1.8x speedup with no extra model). This suggests the future of speculative decoding lies in training-time integration, not post-hoc bolt-ons.

---

## References

- [[speculative-decoding|Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" (2022) (paper)]] — original algorithm and mathematical proof
- [[deepseek-v3|DeepSeek AI, "DeepSeek-V3 Technical Report" (2024) (report)]] — multi-token prediction and MTP-based speculative decoding
- [[raschka-reasoning-llms|Raschka, "Understanding Reasoning LLMs" (2025) (blog)]] — reasoning model training approaches
- [[weng-why-we-think|Weng, "Why We Think" (2025) (blog)]] — test-time compute and chain-of-thought
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024) — Medusa architecture
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024) — feature-level draft reuse
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023) — concurrent independent formalization of speculative decoding
