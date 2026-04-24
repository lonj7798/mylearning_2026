# Tensor Parallelism: Column-Parallel and Row-Parallel Mechanics

<!-- excerpt for ch-13, deep-dive on Megatron-LM tensor parallelism implementation -->

## The Mathematical Foundation

Tensor parallelism exploits two properties of matrix multiplication to split $Y = XW$ across $N$ GPUs:

**Column-parallel (split W by columns):**

$$Y = X \cdot [W_1 | W_2 | \cdots | W_N] = [XW_1 | XW_2 | \cdots | XW_N]$$

Each GPU $i$ stores $W_i$ (a column slice of $W$) and computes $Y_i = XW_i$ independently. The results $Y_i$ are partial outputs that can be used directly if the next operation is element-wise (like GeLU), or must be gathered if the full output is needed.

**Row-parallel (split W by rows):**

$$Y = X \cdot \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix} = \sum_{i=1}^{N} X_i W_i$$

Each GPU $i$ stores $W_i$ (a row slice of $W$) and a corresponding slice $X_i$ of the input. The partial results must be **summed** (all-reduce) to get the correct output.

## The Megatron-LM Transformer Pattern

The insight: pair column-parallel with row-parallel within each sub-layer to cancel intermediate communication.

### MLP Block

A standard transformer MLP is:

$$Y = \text{Dropout}(\text{GeLU}(X W_1) \cdot W_2)$$

Megatron-LM splits this as:

1. $W_1$ is **column-parallel**: each GPU gets columns $W_1^{(i)}$, computes $Z_i = \text{GeLU}(X W_1^{(i)})$
   - GeLU is element-wise, so it applies correctly to each partial output independently
   - **No communication needed** between column-parallel and GeLU

2. $W_2$ is **row-parallel**: each GPU gets rows $W_2^{(i)}$, computes $Y_i = Z_i W_2^{(i)}$
   - This produces a partial sum; **all-reduce** across GPUs gives the correct $Y = \sum_i Z_i W_2^{(i)}$

**Total communication: one all-reduce per MLP** in the forward pass.

In the backward pass, the conjugate operations apply: the all-reduce in forward becomes a no-op in backward (gradients are already correct), and the no-op in forward becomes an all-reduce in backward.

### Attention Block

Multi-head attention naturally decomposes across heads:

1. $W_Q, W_K, W_V$ are **column-parallel**: split by head groups
   - GPU $i$ computes attention for heads $\{i \cdot H/N, \ldots, (i+1) \cdot H/N - 1\}$
   - Each head's attention is independent — no communication needed

2. $W_O$ (output projection) is **row-parallel**: each GPU contributes its heads' output
   - **All-reduce** combines the partial results

**Total communication: one all-reduce per attention block** in the forward pass.

**Constraint on TP degree:** TP must divide the number of attention heads evenly. For GQA models, TP should divide the number of KV heads (not just query heads), since each TP rank needs complete KV head groups. Llama 3 8B has 8 KV heads, so practical TP $\leq$ 8. Using TP=16 would require KV head duplication across ranks and additional synchronization.

## Communication Volume Analysis

Per transformer block (forward pass):
- MLP: one all-reduce of size $(b \times s \times h)$ in BF16 = $2bsh$ bytes
- Attention: one all-reduce of size $(b \times s \times h)$ in BF16 = $2bsh$ bytes
- **Total: $4bsh$ bytes per block** (two all-reduces)

For a model with $L$ layers, the total TP communication per forward pass:

$$\text{TP comm (fwd)} = L \times 4bsh \text{ bytes}$$

For Llama 3 70B ($L = 80, h = 8192$) with $b = 1, s = 4096$, BF16:

$$80 \times 4 \times 1 \times 4096 \times 8192 \times 2 = 21.5 \text{ GB per forward pass}$$

This is communicated in $2L = 160$ all-reduce operations, each on the critical path. At NVLink bandwidth of 900 GB/s (H100 SXM), each all-reduce of ~134 MB takes ~0.15 ms. Total TP communication time: ~24 ms per forward pass. For context, the compute time for a 70B forward pass at BF16 on 8 H100s is ~150 ms — so TP communication is ~16% of compute time. This matches the ~11% throughput loss observed in the Ultra-Scale Playbook benchmarks (some overlap is possible).

## Sequence Parallelism: Completing the Picture

Standard TP shards the hidden dimension for attention and MLP. But LayerNorm and dropout operate on the **full hidden dimension**:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu$ and $\sigma^2$ are computed across the full hidden dimension $h$. These operations cannot be split along $h$.

Sequence Parallelism (SP) shards these operations along the **sequence dimension** instead. The transitions between TP and SP regions use:

- **TP to SP** (after row-parallel): replace all-reduce with **reduce-scatter**
  - All-reduce = reduce-scatter + all-gather
  - We only need the reduce-scatter part; the all-gather is deferred
  - Each GPU keeps $1/N$ of the sequence tokens with the full hidden dimension

- **SP to TP** (before column-parallel): **all-gather** along sequence dimension
  - Reconstruct full sequence before entering TP region

**Communication equivalence:** Two reduce-scatters + two all-gathers per block = two all-reduces per block. Same total volume as vanilla TP. But the activation memory is reduced: maximum activation size drops from $(b, s, h)$ to $(b, s, h/N)$ (in TP region) or $(b, s/N, h)$ (in SP region). Either way, each GPU stores at most $1/N$ of the full activation tensor.

## The Cross-Node Cliff

The Ultra-Scale Playbook benchmarks reveal a sharp throughput cliff when TP crosses node boundaries:

- TP=8 within a node (NVLink, 900 GB/s per GPU): **~11% throughput loss**
- TP=16 across 2 nodes (InfiniBand, ~400 GB/s per link): **~43% throughput loss**
- TP=32 across 4 nodes: **~66% throughput loss**

The explanation: TP's all-reduce sits on the critical path. Unlike DP's gradient all-reduce (which overlaps with backward computation), TP's all-reduce must complete before the next layer can begin. The all-reduce time is:

$$t_{\text{all-reduce}} = \frac{2(N-1)}{N} \cdot \frac{\text{message size}}{\text{bandwidth}} + 2(N-1) \cdot \text{latency}$$

When crossing node boundaries, both bandwidth drops (NVLink to InfiniBand) and latency increases. The effect is multiplicative: lower bandwidth means each message takes longer, and it blocks the entire forward/backward pass until completion.

**Rule of thumb from the playbook:** Keep TP $\leq$ number of GPUs per node. Use PP for cross-node parallelism — PP's point-to-point communication is far more tolerant of low bandwidth and high latency.

## Partial Overlap Techniques

Recent work (Megatron-LM, Nanotron) implements partial overlap of TP communication with computation by splitting matrix multiplications into blocks:

1. Start the all-gather for the next block while computing the current block
2. Pipeline the column-parallel computation with the all-reduce

This turns the critical-path all-reduce into a partially overlapped operation, recovering some of the throughput loss. The Domino paper explores this further with novel techniques to maximize overlap. However, full overlap remains impossible because the final reduction must complete before the dependent computation can begin.
