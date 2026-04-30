# SGD — Mathematical Derivation from First Principles

> **Goal**: Derive SGD step-by-step from the optimization problem, prove convergence rates, and clarify what each hyperparameter mathematically represents. Foundation for the Adam / AdamW derivations in the sibling note.

---

## 0. Setup — the optimization problem

Training a neural network is fundamentally a minimization:

$$
\min_{\theta \in \mathbb{R}^d} \; L(\theta) = \mathbb{E}_{x \sim \mathcal{D}}[\ell(x, \theta)]
$$

- $\ell(x, \theta)$ — sample-level loss (e.g. cross-entropy on one token).
- $\theta \in \mathbb{R}^d$ — all model parameters flattened into a vector.
- $\mathcal{D}$ — true data distribution (unknown).

Since we don't have $\mathcal{D}$, we work with the **empirical risk** over $N$ samples:

$$
\hat{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(x_i, \theta)
$$

Every optimizer (SGD, Adam, AdamW) is an algorithm to minimize $\hat{L}$.

---

## Step 1 — Why descend in direction $-\nabla L$?

**Claim**: among all unit-length directions, $u = -\nabla L / \|\nabla L\|$ decreases $L$ the fastest.

### Proof

First-order Taylor expansion:

$$
L(\theta + \epsilon u) = L(\theta) + \epsilon \, \nabla L(\theta)^\top u + O(\epsilon^2)
$$

For a unit vector $u$ ($\|u\| = 1$), the leading change is:

$$
L(\theta + \epsilon u) - L(\theta) \approx \epsilon \, \nabla L(\theta)^\top u
$$

We want to **minimize** this — find $u$ that makes the inner product as negative as possible:

$$
\min_{\|u\| = 1} \nabla L(\theta)^\top u
$$

By **Cauchy–Schwarz**:

$$
|\nabla L(\theta)^\top u| \leq \|\nabla L(\theta)\| \cdot \|u\| = \|\nabla L(\theta)\|
$$

with equality iff $u$ is parallel to $\nabla L(\theta)$. Including sign, the **minimum** is achieved at:

$$
u^* = -\frac{\nabla L(\theta)}{\|\nabla L(\theta)\|}, \qquad \min = -\|\nabla L(\theta)\|
$$

$\blacksquare$

This is the mathematical justification for "go opposite to the gradient." Every other direction makes less progress per unit step length.

---

## Step 2 — How big a step? (L-smoothness and the descent lemma)

The direction is settled. Now: how far?

### Assumption — $L$-smoothness

We assume $\nabla L$ is **Lipschitz continuous** with constant $L > 0$:

$$
\|\nabla L(x) - \nabla L(y)\| \leq L \, \|x - y\| \quad \forall x, y
$$

Equivalently (when $L$ is twice-differentiable): the Hessian is bounded — $\|\nabla^2 L(x)\|_2 \leq L$. This caps the curvature.

### Descent lemma

**Claim**: any $L$-smooth function satisfies

$$
L(y) \leq L(x) + \nabla L(x)^\top (y - x) + \frac{L}{2} \|y - x\|^2
$$

### Proof

By the fundamental theorem of calculus along the line segment from $x$ to $y$:

$$
L(y) = L(x) + \int_0^1 \nabla L(x + t(y - x))^\top (y - x) \, dt
$$

Add and subtract $\nabla L(x)^\top (y - x)$:

$$
L(y) = L(x) + \nabla L(x)^\top (y - x) + \int_0^1 \big[\nabla L(x + t(y - x)) - \nabla L(x)\big]^\top (y - x) \, dt
$$

Bound the residual integral with **Cauchy–Schwarz** and **Lipschitz**:

$$
\int_0^1 \|\nabla L(x + t(y - x)) - \nabla L(x)\| \cdot \|y - x\| \, dt
\leq \int_0^1 L \cdot t \, \|y - x\|^2 \, dt = \frac{L}{2} \|y - x\|^2
$$

Hence the descent lemma. $\blacksquare$

### Apply to gradient descent

Let $y = x - \alpha \nabla L(x)$:

$$
L(x - \alpha \nabla L(x)) \leq L(x) - \alpha \|\nabla L(x)\|^2 + \frac{L \alpha^2}{2} \|\nabla L(x)\|^2
$$

$$
= L(x) - \alpha \left(1 - \frac{L \alpha}{2}\right) \|\nabla L(x)\|^2
$$

For loss to **decrease**, we need $1 - L\alpha/2 > 0$, i.e.

$$
\boxed{\alpha < \frac{2}{L}}
$$

Maximize the per-step decrease $\alpha (1 - L\alpha/2)$ over $\alpha$: derivative $1 - L\alpha = 0$ gives $\alpha^* = 1/L$. At this optimum:

$$
L(x_{\text{new}}) \leq L(x) - \frac{1}{2L} \|\nabla L(x)\|^2
$$

**Interpretation**: the learning rate's mathematical optimum is $\alpha = 1/L$. If you knew the smoothness constant, you could pick $\alpha$ exactly. Real-world tuning is essentially estimating $1/L$ for your model.

---

## Step 3 — Convergence rate (deterministic GD)

### Convex case — rate $O(1/T)$

**Assumption**: $L$ is convex — $L(y) \geq L(x) + \nabla L(x)^\top (y - x)$ for all $x, y$.

Track the squared distance to the optimum $\theta^*$:

$$
\|x_{k+1} - x^*\|^2 = \|x_k - \alpha \nabla L(x_k) - x^*\|^2
$$

Expand:

$$
= \|x_k - x^*\|^2 - 2\alpha \nabla L(x_k)^\top (x_k - x^*) + \alpha^2 \|\nabla L(x_k)\|^2
$$

Convexity gives $\nabla L(x_k)^\top (x_k - x^*) \geq L(x_k) - L^*$. For $L$-smooth + convex one can also show $\|\nabla L(x_k)\|^2 \leq 2L (L(x_k) - L^*)$. Plugging in and telescoping over $k = 0, \ldots, T-1$:

$$
L(\bar{x}_T) - L^* \leq \frac{\|x_0 - x^*\|^2}{2 \alpha T}, \quad \bar{x}_T = \frac{1}{T} \sum_{k=1}^T x_k
$$

**Rate**: $O(1/T)$ — to reach $\varepsilon$ accuracy takes $O(1/\varepsilon)$ steps.

### Strongly convex case — linear (geometric) rate

**Stronger assumption**: $L(y) \geq L(x) + \nabla L(x)^\top(y - x) + \tfrac{\mu}{2} \|y - x\|^2$ with $\mu > 0$.

Then:

$$
L(x_T) - L^* \leq \left(1 - \frac{\mu}{L}\right)^T \big(L(x_0) - L^*\big)
$$

**Rate**: $O\big(\kappa \log(1/\varepsilon)\big)$ where $\kappa = L / \mu$ is the **condition number**.

This is the theoretical ceiling for plain GD. Nesterov / momentum will improve this to $O(\sqrt{\kappa} \log(1/\varepsilon))$ in Step 5 — and Nesterov proved no first-order method can do better.

---

## Step 4 — Stochasticity: GD → SGD

### Why we need it

Computing $\nabla L = \tfrac{1}{N} \sum \nabla \ell(x_i, \theta)$ exactly costs $O(N)$ per step. For pretraining $N$ is trillions of tokens — infeasible.

### The estimator

Sample a mini-batch $\mathcal{B}_t$ of size $B$ each step:

$$
g_t = \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla \ell(x_i, \theta_t)
$$

**Two crucial properties**:

**(1) Unbiasedness** — $\mathbb{E}[g_t \mid \theta_t] = \nabla L(\theta_t)$.
Trivially follows from $\mathcal{B}_t$ being i.i.d. from $\mathcal{D}$.

**(2) Bounded variance** — $\mathrm{Var}(g_t) = \sigma^2 / B$ for some $\sigma^2 > 0$.
Larger batch → less noise.

### SGD update

$$
\theta_{t+1} = \theta_t - \alpha \, g_t
$$

### Convergence rate

For convex, $L$-smooth, bounded-variance objective:

$$
\mathbb{E}[L(\bar{\theta}_T) - L^*] \leq \frac{\|\theta_0 - \theta^*\|^2}{2 \alpha T} + \frac{\alpha L \sigma^2}{2}
$$

The first term is the deterministic GD bound; the second is the **noise floor**. With **fixed** $\alpha$, SGD does **not converge** — it oscillates around the optimum at scale $O(\alpha L \sigma^2)$.

With **decaying schedule** $\alpha_t = O(1/\sqrt{t})$:

$$
\mathbb{E}[L(\bar{\theta}_T) - L^*] = O(1/\sqrt{T})
$$

For strongly convex with $\alpha_t = O(1/t)$:

$$
\mathbb{E}[L(\bar{\theta}_T) - L^*] = O(\sigma^2 / (\mu T))
$$

**Key takeaway**: noise costs you one order of convergence — GD's $O(1/T)$ becomes SGD's $O(1/\sqrt{T})$ in the convex case. Larger batch ($B$) and smaller step ($\alpha$) suppress the noise term but slow progress on the deterministic term.

---

## Step 5 — Momentum (Polyak's heavy-ball, 1964)

### Motivation

SGD has two failure modes:

- **Noisy zigzag** — successive batches disagree on direction; raw $g_t$ swings between batches and the trajectory wastes work.
- **Narrow ravines** — when the loss surface has very different curvature in different directions (large $\kappa$), the gradient points mostly across the valley, not down it. Plain GD bounces wall-to-wall.

### Heavy-ball update

Maintain a velocity vector $m_t \in \mathbb{R}^d$:

$$
m_t = \rho \, m_{t-1} + g_t, \quad m_0 = 0
$$
$$
\theta_{t+1} = \theta_t - \alpha \, m_t
$$

with $\rho \in [0, 1)$.

### What $m_t$ looks like, unrolled

$$
m_t = \sum_{k=0}^{t-1} \rho^k g_{t-k}
$$

An exponentially-weighted sum of past gradients. The effective averaging window is $1/(1 - \rho)$. With $\rho = 0.9$, that's 10 steps.

### Why it's faster — quadratic analysis

Take $L(\theta) = \tfrac{1}{2} \theta^\top A \theta$ with PD matrix $A$ having eigenvalues $\mu \leq \lambda_i \leq L$. Diagonalizing $A$, each eigen-mode evolves independently:

$$
\theta_{i, t+1} = \theta_{i, t} - \alpha \lambda_i \theta_{i, t} - \alpha \, m_{i, t}
$$
$$
m_{i, t+1} = \rho \, m_{i, t} + \lambda_i \theta_{i, t}
$$

In matrix form:

$$
\begin{pmatrix} \theta_{i, t+1} \\ m_{i, t+1} \end{pmatrix} = \underbrace{\begin{pmatrix} 1 - \alpha \lambda_i & -\alpha \\ \lambda_i & \rho \end{pmatrix}}_{=: M_i} \begin{pmatrix} \theta_{i, t} \\ m_{i, t} \end{pmatrix}
$$

Convergence rate per mode = spectral radius of $M_i$.

**Polyak's result**: with optimal choice

$$
\rho^* = \left(\frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}\right)^2,
\qquad \alpha^* = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}
$$

the spectral radius is $\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$, giving:

$$
\|\theta_t - \theta^*\| = O\!\left(\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^t\right) = O\!\left((1 - 1/\sqrt{\kappa})^t\right)
$$

### The big payoff

| Method | Iterations to $\varepsilon$ accuracy |
|---|---|
| Plain GD | $O(\kappa \, \log(1/\varepsilon))$ |
| Heavy-ball (and Nesterov) | $O(\sqrt{\kappa} \, \log(1/\varepsilon))$ |

For $\kappa = 10^6$ (typical for deep nets), the speedup factor is $\kappa / \sqrt{\kappa} = \sqrt{\kappa} = 1000$. **Three orders of magnitude faster**, from one extra line maintaining $m_t$.

Nesterov (1983) showed $O(\sqrt{\kappa} \log(1/\varepsilon))$ is **optimal** among first-order methods. You cannot do better without computing second derivatives. Momentum is not a hack; it's the optimal-rate first-order algorithm.

$\blacksquare$

---

## Step 6 — Hyperparameter dictionary

Each SGD-family hyperparameter has a precise mathematical role:

| Hyperparameter | Appears in | Mathematical meaning | Tuning direction |
|---|---|---|---|
| $\alpha$ (learning rate) | $\theta_{t+1} = \theta_t - \alpha g_t$ | step size; stability requires $\alpha < 2/L$, optimum $\alpha \approx 1/L$ | smaller for larger $L$ (sharper curvature) |
| $\rho$ (momentum) | $m_t = \rho m_{t-1} + g_t$ | gradient memory length $1/(1-\rho)$; gives $\sqrt{\kappa}$ speedup | closer to 1 for ill-conditioned problems |
| $B$ (batch size) | $g_t = \tfrac{1}{B} \sum \nabla \ell$ | $\mathrm{Var}(g_t) = \sigma^2 / B$; sets noise floor | larger ⇒ closer to GD; smaller ⇒ faster but noisier |
| $T$ (steps) | denominator of bounds | total optimization budget; convex SGD needs $O(1/\varepsilon^2)$ | larger for deeper / harder problems |

### Three load-bearing trade-offs

**(A) $\alpha$ vs $B$ — the noise budget**

The SGD bound's noise term is $\alpha L \sigma^2 / 2$ but the variance in $g_t$ is $\sigma^2 / B$. Net "effective noise" scales with $\alpha / B$. The **linear scaling rule** ($\alpha \propto B$) keeps this constant when changing batch size — well-known empirical heuristic, falls directly out of the math.

**(B) $\alpha$ vs $L$ — the smoothness ceiling**

$\alpha$ must be below $2/L$ for stability. Different model architectures have different effective $L$:
- Transformers without LayerNorm — $L$ is huge (gradients explode), needs tiny $\alpha$.
- Transformers with LayerNorm + residual — $L$ is tamed, allowing $\alpha \sim 10^{-4}$.
- This is part of why architectural innovations like Pre-LN unlock larger learning rates.

**(C) $\rho$ vs noise — averaging vs lag**

Larger $\rho$ averages over more past gradients — better noise rejection — but also lags when the gradient signal genuinely changes (regime shifts, schedule transitions). The default $\rho = 0.9$ is the empirical compromise. RL fine-tuning often lowers it because rewards are non-stationary.

---

## What this proved (and what it didn't)

**Proved**:

- Steepest descent direction is $-\nabla L$ (Cauchy–Schwarz).
- $L$-smoothness gives the descent lemma, which yields $\alpha < 2/L$ for stability.
- Convex GD: $O(1/T)$. Strongly convex GD: $O(\kappa \log(1/\varepsilon))$.
- SGD: bounded variance noise floor; with decaying $\alpha$ get $O(1/\sqrt{T})$.
- Momentum (Polyak): $O(\sqrt{\kappa} \log(1/\varepsilon))$ — provably $\sqrt{\kappa}$ speedup over plain GD.

**Did not prove** (but acknowledged):

- Convergence on **non-convex** objectives. Transformers are non-convex. The best general result is convergence to a stationary point ($\|\nabla L\| \to 0$) at rate $O(1/\sqrt{T})$ for SGD with decaying step. There are no global-minimum guarantees.
- That SGD-with-momentum will outperform GD-with-momentum in practice (it usually does, but the proof is subtle and depends on noise structure).
- Saddle-point escape. SGD's noise actually **helps** here in non-convex landscapes — it perturbs out of saddles. This is empirical / heuristic theory.

---

## Next note: AdaGrad → RMSProp → Adam → AdamW

Topics for the sibling derivation:

1. **Preconditioning** — viewing adaptive methods as approximating $H^{-1} \nabla L$.
2. **AdaGrad** — accumulating $\sum g_t^2$, why it stalls.
3. **RMSProp** — replacing the cumulative sum with an EMA.
4. **Adam** = momentum + RMSProp + bias correction. Full derivation of $\hat{m}_t, \hat{v}_t$.
5. **AdamW** — proximal-gradient interpretation; why decoupled decay is the mathematically correct version of weight decay under adaptive scaling.
6. **Convergence proof** — Reddi et al.'s counterexample showing original Adam can diverge, and the AMSGrad fix.

These all build directly on the SGD foundation derived here — momentum (Step 5), convergence machinery (Step 3), and the noise / batch trade-off (Step 4) all reappear. Read this note first.
