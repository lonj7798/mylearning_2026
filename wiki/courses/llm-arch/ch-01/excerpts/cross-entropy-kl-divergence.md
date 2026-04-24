<!-- scope: cross-entropy / KL divergence mathematical deep-dive, parent: [[ch-01]] -->

# Cross-Entropy and KL Divergence: The Full Derivation

This excerpt works through the complete mathematical chain from information content to the cross-entropy loss used in every autoregressive LLM. The goal is not just to state the equivalences but to derive them from first principles so you can verify each step yourself.

---

## 1. Information Content of a Single Event

Start with a probability distribution $P$ over discrete outcomes. When an event with probability $p$ occurs, the **information content** (or "surprisal") is:

$$I(x) = -\log p(x)$$

Why the negative log? Three requirements uniquely determine this form:

1. **Monotonicity:** Rarer events ($p \to 0$) carry more information. $-\log p$ is monotonically decreasing in $p$.
2. **Additivity:** For independent events $A, B$: $I(A \cap B) = I(A) + I(B)$. Since $p(A \cap B) = p(A) \cdot p(B)$, we need $f(p_1 \cdot p_2) = f(p_1) + f(p_2)$. The logarithm is the unique continuous function satisfying this (Cauchy's functional equation).
3. **Continuity:** Small changes in probability produce small changes in information.

The base of the logarithm determines the unit: base 2 gives bits, base $e$ gives nats. LLM training uses natural log (nats), so cross-entropy loss values are in nats.

**Concrete examples:**
- Fair coin heads ($p = 0.5$): $I = -\ln(0.5) = 0.693$ nats = 1 bit
- Token with probability 0.01: $I = -\ln(0.01) = 4.605$ nats
- Token with probability 0.9: $I = -\ln(0.9) = 0.105$ nats (low surprise)

---

## 2. Entropy: Expected Surprise

The **entropy** of distribution $P$ is the expected information content:

$$H(P) = \mathbb{E}_{x \sim P}[I(x)] = -\sum_{x} P(x) \log P(x)$$

This measures the irreducible uncertainty in the data. For language, $H(P)$ represents the inherent unpredictability of human text -- even a perfect model cannot reduce its loss below $H(P)$.

**Properties:**
- $H(P) \geq 0$ always (information is non-negative)
- $H(P) = 0$ iff $P$ is deterministic (one outcome has probability 1)
- $H(P)$ is maximized when $P$ is uniform: $H_{\max} = \log |V|$

For a vocabulary of $|V| = 32{,}000$ tokens, the maximum entropy is $\log(32000) \approx 10.37$ nats. A well-trained LLM achieves cross-entropy around 2-3 nats, far below the uniform baseline.

---

## 3. Cross-Entropy: Comparing Two Distributions

Given the true data distribution $P$ and the model's predicted distribution $Q_\theta$:

$$H(P, Q_\theta) = -\sum_{x} P(x) \log Q_\theta(x)$$

**Information-theoretic meaning:** The average number of nats needed to encode samples from $P$ using an encoding scheme optimized for $Q_\theta$. If $Q_\theta = P$, the encoding is optimal and $H(P, Q_\theta) = H(P)$. Any mismatch between $Q_\theta$ and $P$ increases the cost.

In LLM training, $P$ is the empirical distribution of the training data (which puts all mass on the observed next token), and $Q_\theta$ is the model's softmax output. For a single training example where the true next token is $x^*$:

$$P(x) = \mathbf{1}[x = x^*] \quad \Rightarrow \quad H(P, Q_\theta) = -\log Q_\theta(x^*)$$

This is exactly the negative log-likelihood of the correct token under the model.

---

## 4. KL Divergence: The Gap Between Distributions

The **Kullback-Leibler divergence** measures how much $Q_\theta$ differs from $P$:

$$D_{KL}(P \| Q_\theta) = \sum_{x} P(x) \log \frac{P(x)}{Q_\theta(x)} = H(P, Q_\theta) - H(P)$$

**Properties:**
- $D_{KL}(P \| Q) \geq 0$ (Gibbs' inequality, provable via Jensen's inequality)
- $D_{KL}(P \| Q) = 0$ iff $P = Q$ almost everywhere
- **Not symmetric:** $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ in general
- **Not a metric:** Violates triangle inequality

The asymmetry matters. In LLM training we minimize $D_{KL}(P \| Q_\theta)$ (forward KL), which is **mean-seeking**: it forces $Q_\theta$ to place mass wherever $P$ has mass. The reverse KL $D_{KL}(Q_\theta \| P)$ would be **mode-seeking**, concentrating $Q_\theta$ on the single highest-probability mode of $P$. Forward KL is the right choice for language modeling because we want the model to cover the full distribution of plausible next tokens, not collapse to a single prediction.

---

## 5. The Three-Way Equivalence (Proof)

**Claim:** Minimizing cross-entropy $\equiv$ minimizing KL divergence $\equiv$ maximum likelihood estimation.

**Proof:**

From the decomposition:
$$H(P, Q_\theta) = H(P) + D_{KL}(P \| Q_\theta)$$

Since $H(P)$ is constant with respect to $\theta$:
$$\arg\min_\theta H(P, Q_\theta) = \arg\min_\theta D_{KL}(P \| Q_\theta) \qquad \text{(1)}$$

For maximum likelihood, given training data $\{x_1, \ldots, x_N\}$:
$$\hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^{N} Q_\theta(x_i) = \arg\max_\theta \sum_{i=1}^{N} \log Q_\theta(x_i)$$

Dividing by $N$ and applying the law of large numbers as $N \to \infty$:
$$\frac{1}{N}\sum_{i=1}^{N} \log Q_\theta(x_i) \xrightarrow{N \to \infty} \mathbb{E}_{x \sim P}[\log Q_\theta(x)] = -H(P, Q_\theta)$$

Therefore:
$$\hat{\theta}_{MLE} = \arg\max_\theta \left[-H(P, Q_\theta)\right] = \arg\min_\theta H(P, Q_\theta) \qquad \text{(2)}$$

Combining (1) and (2): all three objectives have the same minimizer. Three frameworks, one gradient.

*Source: [[bendersky-cross-entropy|blog]]*

---

## 6. Why Cross-Entropy Is Not a Distance Metric

A common imprecision: "cross-entropy measures the distance between distributions." It does not satisfy any of the three metric axioms:

| Axiom | Requirement | Cross-entropy |
|-------|-------------|---------------|
| Identity of indiscernibles | $d(P, P) = 0$ | $H(P, P) = H(P) \neq 0$ in general |
| Symmetry | $d(P, Q) = d(Q, P)$ | $H(P, Q) \neq H(Q, P)$ |
| Triangle inequality | $d(P, R) \leq d(P, Q) + d(Q, R)$ | Violated |

KL divergence is closer (it is zero iff $P = Q$) but also asymmetric and violates the triangle inequality. If you need an actual metric on probability distributions, use total variation distance or the Wasserstein distance. But for gradient-based optimization, the non-metric properties of cross-entropy are irrelevant -- what matters is that it has a well-defined gradient and its minimum corresponds to $Q_\theta = P$.

---

## 7. Practical Implications for LLM Training

**The irreducible loss floor.** Cross-entropy loss can never reach zero on natural language because $H(P) > 0$ -- language has genuine uncertainty. When you see a model with cross-entropy loss of 2.5 nats, the actual modeling error (KL divergence) is $2.5 - H(P)$. Estimating $H(P)$ for natural language is an open problem, but Shannon's experiments suggest English text has entropy around 1.0-1.3 bits/character, which translates to roughly 1.5-2.0 nats/token for modern tokenizers.

**Label smoothing and cross-entropy.** The original Transformer uses label smoothing ($\epsilon = 0.1$), which replaces the hard target $P(x) = \mathbf{1}[x = x^*]$ with:

$$P_{smooth}(x) = (1 - \epsilon)\mathbf{1}[x = x^*] + \frac{\epsilon}{|V|}$$

This changes the cross-entropy target: instead of driving $Q_\theta(x^*) \to 1$ (which requires logits $\to \infty$), the model targets $Q_\theta(x^*) \to 0.9$. The result is better calibration and generalization at the cost of slightly worse perplexity -- because perplexity rewards confidence, while calibration rewards accuracy.

*Source: [[attention-is-all-you-need|paper]], [[bendersky-cross-entropy|blog]]*
