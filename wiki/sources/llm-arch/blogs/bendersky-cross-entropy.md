<!-- scope: cross-entropy loss mathematical foundations
     deps: [[ch-02]]
     see-also: [[raschka-next-token-prediction]], [[hf-perplexity]]
-->

# Cross-Entropy and KL Divergence

- **Core Insight:** Minimizing cross-entropy = minimizing KL divergence = maximum likelihood; three frameworks, one gradient.
- **Guideline:** Don't treat cross-entropy as arbitrary — it's the unique loss where three optimization frameworks converge.

- **Author:** Eli Bendersky
- **URL:** https://eli.thegreenplace.net/2025/cross-entropy-and-kl-divergence/
- **Relevant chapters:** Loss functions, training objectives, information theory fundamentals

## Summary
A rigorous yet accessible walkthrough of the mathematical foundations behind the most common LLM training loss function. Builds from information content of single events, through entropy and cross-entropy, to KL divergence, and proves that cross-entropy minimization equals maximum likelihood estimation.

## Key Content

### 1. Information Content of a Single Event

Measures the "degree of surprise" when an event with probability p occurs:

**I(E) = -log_2(p)**

- An event with probability 1 yields zero information (no surprise)
- Rare events (p -> 0) yield high information content
- For independent events, information contents are additive: log(p1 * p2) = log(p1) + log(p2)

**Examples:**
- Fair coin heads (p=1/2): I = 1 bit
- Fair die landing on 4 (p=1/6): I ~ 2.58 bits

### 2. Entropy

The expected value of information across all possible outcomes of a random variable X:

**H(X) = -sum(p_j * log_2(p_j)) for j=1 to n**

Properties:
- High entropy = high uncertainty
- Low entropy = low uncertainty
- Always non-negative
- Deterministic distribution (one outcome with p=1): H = 0
- Maximum entropy occurs with uniform distributions

### 3. Cross-Entropy

Extends entropy to compare two different probability distributions:

**H(P, Q) = -sum(p_j * log_2(q_j))**

Where P = actual data distribution, Q = predicted distribution.

Properties:
- Non-negative
- Collapses to H(P) when P = Q
- Information-theoretic meaning: "the average number of bits required to encode an actual probability distribution P, when we assumed the data follows Q instead"

### 4. KL Divergence

Measures the difference between two probability distributions:

**D_KL(P, Q) = H(P, Q) - H(P)**

Alternative formulations:
- D_KL(P, Q) = -sum(p_j * log_2(q_j / p_j))
- D_KL(P, Q) = sum(p_j * log_2(p_j / q_j))

Properties:
- Always non-negative
- D_KL(P, P) = 0 (zero only when distributions are identical)
- **Not symmetric:** D_KL(P, Q) != D_KL(Q, P) — not a true distance metric

### 5. The Critical Relationship

**H(P, Q) = D_KL(P, Q) + H(P)**

**ML implication:** Since H(P) is independent of the model, optimizing cross-entropy is mathematically equivalent to optimizing KL divergence. This makes cross-entropy an ideal loss function.

### 6. Connection to Maximum Likelihood Estimation

The derivation proving cross-entropy minimization = MLE:

1. Start with likelihood: L(theta) = prod(Q_theta(x_i))
2. Convert to log-likelihood: log L(theta) = sum(log Q_theta(x_i))
3. Average over n samples: (1/n) sum(log Q_theta(x_i))
4. Apply Law of Large Numbers: sum(P(x_i) * log Q_theta(x_i))
5. Negate to minimize: -sum(P(x_i) * log Q_theta(x_i))

**Result:** The minimized function is precisely the cross-entropy formula, proving that "maximum likelihood estimation is equivalent to minimizing the cross-entropy between the true and predicted data distributions."

### Conceptual Hierarchy

Information Content (single event) -> Entropy (expected information) -> Cross-Entropy (two distributions) -> KL Divergence (subtract inherent uncertainty)

### Uses in Machine Learning

**Cross-entropy as loss function:**
- Provides a single scalar metric
- Lower for similar distributions, higher for dissimilar
- Non-negative
- Used in logistic regression and softmax classification

**KL divergence applications:**
- Variational Autoencoders (evidence lower bound / ELBO)
- Advanced generative modeling

## Notable Insights
- Cross-entropy loss in LLM training is doing exactly the same thing as maximum likelihood estimation — they are mathematically identical, not just similar.
- KL divergence's asymmetry (D_KL(P,Q) != D_KL(Q,P)) is why it's not called a "distance" — this matters when choosing which distribution is P vs Q.
- The cross-entropy between P and Q always >= H(P), with equality only when Q=P. The "excess" bits are exactly the KL divergence.
- When people say an LLM is trained with "cross-entropy loss," they mean the model is learning to assign high probability to the actual next token — the same thing as maximizing the likelihood of the training data.
