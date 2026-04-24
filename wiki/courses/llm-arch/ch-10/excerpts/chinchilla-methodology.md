<!-- scope: Chinchilla three-approach methodology, why Kaplan was wrong, practical allocation tables
     parent: [[ch-10]]
-->

# Chinchilla Methodology: Three Approaches to Compute-Optimal Training

## Why This Matters

The strength of the Chinchilla result comes from methodological triangulation: three independent approaches to finding the compute-optimal frontier, all yielding the same answer. Understanding the methodology explains both why Chinchilla is trusted and why Kaplan's earlier result was flawed.

## Approach 1: IsoFLOP Profiles

**Method:** Fix the total compute budget $C$, train models of varying sizes $N$, and find the $N$ that minimizes loss.

For each compute budget level:
1. Choose a range of model sizes (e.g., 70M to 16B)
2. For each model size, compute the number of tokens $D = C / (6N)$ that can be trained within budget
3. Train each model to completion
4. Plot loss vs. model size; the minimum identifies $N_\text{opt}(C)$

**Key result:** The optimal model size scales as $N_\text{opt} \propto C^a$ where $a \approx 0.50$. This directly gives the parameter half of the allocation rule.

**Why Kaplan missed this:** Kaplan did not independently vary training duration at each model size. Instead, they used a fixed learning rate schedule across all sizes. This confound biased the result toward larger models, because larger models benefit more from Kaplan's particular schedule -- making it appear that model size was more important than it actually is.

## Approach 2: IsoLoss Contours

**Method:** For a target loss level $L^*$, find the minimum compute $C$ required to achieve it.

1. From the IsoFLOP experiments, identify all $(N, D, C)$ triples that achieve loss close to $L^*$
2. Among these, find the triple that minimizes $C$
3. This gives the compute-optimal $(N^*, D^*)$ for each target loss level
4. Plot $N^*$ and $D^*$ as functions of $C^*$

**Key result:** Both $N^*$ and $D^*$ scale as $C^{*0.5}$, confirming equal scaling from a different angle.

**Advantage over Approach 1:** This method is less sensitive to the exact shape of the loss landscape near the minimum, because it aggregates over multiple compute budgets.

## Approach 3: Parametric Loss Fit

**Method:** Fit a parametric function $L(N, D)$ to all training runs, then analytically compute the optimal allocation.

The parametric form:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where $E$ is the irreducible loss (entropy of natural language), and the two power-law terms capture the contributions of finite model size and finite data.

**Fitted parameters (Chinchilla):**
- $E = 1.69$ nats
- $A = 406.4$, $\alpha = 0.34$
- $B = 410.7$, $\beta = 0.28$

To find the compute-optimal allocation, minimize $L(N, D)$ subject to $C = 6ND$:

$$\frac{\partial L}{\partial N} + \lambda \cdot 6D = 0, \qquad \frac{\partial L}{\partial D} + \lambda \cdot 6N = 0$$

Solving yields:

$$\frac{N_\text{opt}}{D_\text{opt}} = \frac{\alpha \cdot A}{\beta \cdot B} \cdot \left(\frac{D_\text{opt}}{N_\text{opt}}\right)^{1 - \alpha/\beta + ...}$$

The numerical solution gives approximately equal scaling: $N_\text{opt} \propto C^{0.49}$, $D_\text{opt} \propto C^{0.51}$.

## The Kaplan-Chinchilla Disagreement, Quantified

| Quantity | Kaplan (2020) | Chinchilla (2022) |
|----------|--------------|-------------------|
| $N_\text{opt}$ scaling with $C$ | $C^{0.73}$ | $C^{0.50}$ |
| $D_\text{opt}$ scaling with $C$ | $C^{0.27}$ | $C^{0.50}$ |
| Per 10x compute increase: $N$ multiplier | 5.5x | 3.16x |
| Per 10x compute increase: $D$ multiplier | 1.8x | 3.16x |
| Optimal tokens-per-parameter ratio | ~7 | ~20 |

The practical difference is enormous. For a compute budget that Kaplan would have allocated to a 200B model on 200B tokens, Chinchilla says: train a ~60B model on ~1.2T tokens. The Chinchilla model will have lower loss, lower inference cost (3.3x fewer parameters to serve), and better downstream performance.

## Practical Allocation Table

Using Chinchilla scaling ($N_\text{opt} \propto C^{0.5}$, with calibration from the paper):

| Compute Budget (FLOPs) | Optimal N | Optimal D | Rough GPU-hours (A100) |
|-------------------------|-----------|-----------|------------------------|
| $10^{19}$ | ~70M | ~1.4B | ~50 |
| $10^{20}$ | ~220M | ~4.5B | ~500 |
| $10^{21}$ | ~700M | ~14B | ~5K |
| $10^{22}$ | ~2.2B | ~45B | ~50K |
| $10^{23}$ | ~7B | ~140B | ~500K |
| $10^{24}$ | ~22B | ~450B | ~5M |
| $10^{25}$ | ~70B | ~1.4T | ~50M |

**Note:** These are compute-optimal allocations for minimizing loss per FLOP. If your goal is to minimize *inference cost* (as LLaMA's was), you should train a smaller model on more data -- accepting higher training cost for cheaper serving.

## Beyond Chinchilla: When to Deviate

Chinchilla optimality minimizes loss per FLOP of *training*. But the correct objective depends on your deployment scenario:

1. **High query volume:** Train a smaller model on more data (LLaMA strategy). The extra training cost is amortized over millions of inference queries.

2. **Research/one-off use:** Chinchilla-optimal is correct. You minimize total cost.

3. **Data-constrained regime:** If you have less unique data than $D_\text{opt}$, you face a choice: train a smaller model (staying on the Chinchilla frontier) or repeat data (with diminishing returns per [[scaling-data-constrained|paper]]).

4. **Inference-time scaling:** If you plan to use chain-of-thought or best-of-N at inference time, the effective compute per query increases. This shifts the optimal training allocation toward smaller, cheaper-to-serve models that can be queried multiple times.

## References

- [[chinchilla|Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) (paper)]]
- [[scaling-laws-kaplan|Kaplan et al., "Scaling Laws for Neural Language Models" (2020) (paper)]]
- [[scaling-data-constrained|Muennighoff et al., "Scaling Data-Constrained Language Models" (2023) (paper)]]
