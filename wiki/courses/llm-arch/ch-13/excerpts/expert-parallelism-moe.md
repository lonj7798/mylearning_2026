# Expert Parallelism: Why MoE Training Is a Distributed Systems Problem

<!-- excerpt for ch-13, deep-dive on expert parallelism and MoE communication patterns -->

## The Basic Setup

In a Mixture-of-Experts (MoE) model, each transformer block's FFN is replaced by $E$ parallel "expert" FFNs plus a router that assigns tokens to experts. During forward pass:

1. **Router** computes expert assignments for each token: $\text{gate}(x) \to \{e_1, e_2, \ldots\}$
2. **Dispatch:** tokens are sent to their assigned experts (potentially on different GPUs)
3. **Expert compute:** each expert processes its received tokens independently
4. **Combine:** results are sent back to the originating GPU and combined

Expert parallelism (EP) distributes experts across GPUs: GPU $i$ hosts experts $\{i \cdot E/N_{\text{EP}}, \ldots, (i+1) \cdot E/N_{\text{EP}} - 1\}$.

## The All-to-All Communication Pattern

The dispatch and combine steps require **all-to-all** communication: every GPU potentially sends tokens to every other GPU, and receives tokens from every other GPU. This is qualitatively different from the collectives used by other parallelism strategies:

| Collective | Pattern | Bandwidth Usage |
|-----------|---------|-----------------|
| All-reduce (DP, TP) | Every GPU contributes; every GPU gets same result | Ring-efficient: $2(N-1)/N$ of message size |
| All-gather (ZeRO-3) | Every GPU contributes shard; every GPU gets full tensor | Ring-efficient: $(N-1)/N$ of message size |
| Point-to-point (PP) | One GPU to one GPU | Minimal: one link saturated |
| **All-to-all (EP)** | **Every GPU sends unique data to every other GPU** | **Full bisection bandwidth** |

All-to-all saturates the network's **bisection bandwidth** — the total bandwidth available for cutting the cluster in half. On a cluster with $N$ GPUs connected via a fat-tree network with bisection bandwidth $B$, the all-to-all throughput is bounded by $B$, regardless of individual link speeds.

This is why MoE training scales differently from dense model training: adding more GPUs for EP does not proportionally increase training speed because the all-to-all communication grows with the number of experts.

## Load Balancing: The Distributed Systems Challenge

Even with perfect communication, EP has a fundamental load-balancing problem: the router may send different numbers of tokens to different experts. If expert $j$ receives twice as many tokens as expert $k$, GPU $j$ takes twice as long to compute while GPU $k$ sits idle.

**Capacity factor:** Most MoE implementations cap the number of tokens each expert can receive at $C \cdot T / E$, where $C$ is the capacity factor (typically 1.0-1.5), $T$ is total tokens, and $E$ is total experts. Tokens exceeding the capacity are **dropped** (not processed), which degrades quality.

**Auxiliary load-balancing loss:** A regularization term encourages the router to distribute tokens evenly:

$$\mathcal{L}_{\text{aux}} = \alpha \sum_{e=1}^{E} f_e \cdot p_e$$

where $f_e$ is the fraction of tokens routed to expert $e$, $p_e$ is the average router probability for expert $e$, and $\alpha$ is a balancing coefficient.

The tension: a perfect load-balancing loss would force uniform routing, destroying the specialization that makes MoE effective. Too little balancing causes severe idle time on underloaded GPUs. The coefficient $\alpha$ trades model quality for training efficiency — a distributed systems parameter dressed up as a model hyperparameter.

## DeepSeek V3: Node-Constrained Routing

DeepSeek V3 introduced **node-limited routing** to control all-to-all communication scope. The constraint: each token is routed to experts on at most $M$ nodes (in their case, $M = 4$ out of 32+ nodes).

**Why this matters:** Without node constraints, the all-to-all exchanges data between all $N_{\text{EP}}$ GPUs. With node constraints, each token's dispatch is limited to $M \cdot G$ GPUs (where $G$ is GPUs per node). This doesn't change the total computation, but dramatically reduces the communication pattern's complexity and bandwidth requirement.

**Implementation:** The router first selects which $M$ nodes a token should go to (coarse-grained routing), then selects which experts within those nodes (fine-grained routing). This two-level routing is optimized jointly but enforces the locality constraint.

**Architecture impact:** The node-locality constraint limits which experts can be combined for a given token. This is an explicit tradeoff between model expressiveness and communication efficiency — the model cannot learn arbitrary expert combinations if they span too many nodes.

## EP and DP Interaction

EP only affects MoE layers (FFN replacements). The attention blocks, LayerNorms, and embeddings are **not** partitioned by EP. Without additional parallelism, every GPU would redundantly compute attention on the same data.

The solution: combine EP with DP. The total GPU count decomposes as:

$$N = N_{\text{DP}} \times N_{\text{EP}}$$

Within each EP group, different GPUs hold different experts. Across EP groups (DP dimension), different GPUs process different micro-batches. For non-MoE layers, all GPUs within an EP group process the same data — effectively, EP reduces the DP degree for non-MoE layers.

**Example (DeepSeek V3 simplified):**
- 256 total GPUs
- 256 experts
- EP = 32 (32 GPUs per EP group, each holding 8 experts)
- DP = 8 (8 EP groups processing different data)
- TP = 1 (within-GPU only for dense layers)

Each expert receives tokens from all 8 DP replicas within its EP group, so the all-to-all is scoped to 32 GPUs.

## Communication Cost Calculation

For a single MoE layer with $E$ experts, $N_{\text{EP}}$ GPUs, batch of $B$ tokens (after DP split), top-$k$ routing:

**Dispatch (forward):**
- Each GPU sends $\frac{k \cdot B}{N_{\text{EP}}}$ tokens to each other GPU (on average, assuming balanced routing)
- Each token has hidden dimension $h$ in BF16: $2h$ bytes per token
- Total per-GPU send volume: $\frac{k \cdot B \cdot 2h \cdot (N_{\text{EP}} - 1)}{N_{\text{EP}}}$

**Combine (forward):** Same volume in the reverse direction.

**Backward:** Doubles the communication (gradients flow back through the same pattern).

For DeepSeek V3 ($h = 7168$, $B = 4096$, $k = 8$, $N_{\text{EP}} = 32$):
- Per-GPU dispatch volume: $\frac{8 \times 4096 \times 14336 \times 31}{32} \approx 455$ MB per MoE layer
- With 60 MoE layers: ~27 GB per forward pass from EP alone

This is comparable to TP's communication volume — but all-to-all is harder to overlap with computation than TP's all-reduce, because the token routing depends on the router's output (which is computed during the forward pass, not known in advance).

## Practical Considerations

1. **Expert parallelism should be combined with other strategies.** EP alone leaves attention layers redundantly computed. The standard combination is EP + DP + TP (intra-node) + PP (inter-node).

2. **The number of experts is a distributed systems parameter.** More experts = more EP parallelism possible, but also more all-to-all communication. DeepSeek V3's choice of 256 experts is partly driven by fitting one expert per GPU at their target model size.

3. **Granularity matching matters.** If experts are too small (few parameters each), the compute-to-communication ratio becomes unfavorable and all-to-all overhead dominates. If experts are too large, they limit the EP degree.

4. **All-to-all does not have efficient ring implementations.** Unlike all-reduce (which has O(N) bandwidth-optimal ring algorithms), all-to-all fundamentally requires O(N) simultaneous pairwise exchanges. Network topology (fat-tree vs torus vs NVSwitch) has a much larger impact on all-to-all performance than on all-reduce performance.
