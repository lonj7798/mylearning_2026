# Chapter 1: Language Modeling Fundamentals

<!-- scope: what a language model is, autoregressive vs masked objectives, cross-entropy loss, perplexity, teacher forcing
     deps: none
     see-also: [[ch-02]] (attention mechanism), [[ch-04]] (decoder-only LLMs)
-->

## Overview

A language model assigns probabilities to sequences of tokens. Everything that follows in this course — attention, Transformers, MoE, speculative decoding — is a design choice in service of making that probability assignment more accurate, more efficient, or both.

This chapter establishes the objective function that all LLM architectures optimize. You'll understand *why* autoregressive modeling dominates, what cross-entropy loss actually optimizes, and a fundamental limitation of sequential prediction that motivates much of the architecture research you'll study later.

---

## 1. What Is a Language Model?

A language model defines a probability distribution over sequences of tokens:

$$P(x_1, x_2, \ldots, x_T)$$

Given a vocabulary $V$ of size $|V|$ (typically 32K–128K tokens), the model assigns a probability to every possible sequence. The goal of training is to make the model assign high probability to sequences that look like real language — and low probability to everything else.

The practical question is: how do you parameterize a distribution over variable-length sequences drawn from a vocabulary of 100K+ tokens? The space is combinatorially enormous. You need a factorization.

---

## 2. The Autoregressive Factorization

The chain rule of probability gives us an exact decomposition — no approximations, no assumptions:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

This is always valid. The "autoregressive" part is not the factorization itself (which is a mathematical identity) but the decision to **model each conditional $P(x_t \mid x_{<t})$ with a neural network**. The network takes the preceding tokens as input and outputs a distribution over the vocabulary for the next token.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0; font-family:monospace;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Autoregressive Generation (Left to Right)</div>
<div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
<div style="background:#0f3460; padding:10px 16px; border-radius:8px; color:#e94560; font-weight:bold;">The</div>
<div style="color:#666; font-size:20px;">→</div>
<div style="background:#0f3460; padding:10px 16px; border-radius:8px; color:#e94560; font-weight:bold;">cat</div>
<div style="color:#666; font-size:20px;">→</div>
<div style="background:#0f3460; padding:10px 16px; border-radius:8px; color:#e94560; font-weight:bold;">sat</div>
<div style="color:#666; font-size:20px;">→</div>
<div style="background:#0f3460; padding:10px 16px; border-radius:8px; color:#e94560; font-weight:bold;">on</div>
<div style="color:#666; font-size:20px;">→</div>
<div style="background:#16213e; padding:10px 16px; border-radius:8px; border:2px dashed #e94560; color:#e94560;">???</div>
</div>
<div style="color:#888; font-size:12px; margin-top:12px; font-family:sans-serif;">
P("the") × P("cat"|"the") × P("sat"|"the cat") × P("on"|"the cat sat") × P(???|"the cat sat on")
</div>
</div>

**Why this factorization won:** Every token position provides a training signal. Given a sequence of T tokens, you get T gradient updates from a single example. Compare this to masked language modeling (BERT), which masks ~15% of tokens and only trains on those — wasting 85% of potential signal per sequence. At scale, this efficiency difference compounds dramatically.

---

## 3. Autoregressive vs. Masked Language Modeling

The two dominant paradigms make fundamentally different architectural bets:

<div style="background:#f8f9fa; border-radius:12px; padding:24px; margin:20px 0; border:1px solid #dee2e6;">
<table style="width:100%; border-collapse:collapse; font-size:14px;">
<thead>
<tr style="border-bottom:2px solid #333;">
<th style="text-align:left; padding:8px 12px;">Dimension</th>
<th style="text-align:left; padding:8px 12px;">Autoregressive (GPT)</th>
<th style="text-align:left; padding:8px 12px;">Masked (BERT)</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #eee;">
<td style="padding:8px 12px; font-weight:bold;">Factorization</td>
<td style="padding:8px 12px;">Left-to-right chain rule</td>
<td style="padding:8px 12px;">Predict masked tokens from bidirectional context</td>
</tr>
<tr style="border-bottom:1px solid #eee;">
<td style="padding:8px 12px; font-weight:bold;">Training signal</td>
<td style="padding:8px 12px;">100% of tokens</td>
<td style="padding:8px 12px;">~15% of tokens (masked only)</td>
</tr>
<tr style="border-bottom:1px solid #eee;">
<td style="padding:8px 12px; font-weight:bold;">Context direction</td>
<td style="padding:8px 12px;">Causal (past only)</td>
<td style="padding:8px 12px;">Bidirectional (full sequence)</td>
</tr>
<tr style="border-bottom:1px solid #eee;">
<td style="padding:8px 12px; font-weight:bold;">Generation</td>
<td style="padding:8px 12px;">Natural (sample token by token)</td>
<td style="padding:8px 12px;">Unnatural (iterative re-masking)</td>
</tr>
<tr style="border-bottom:1px solid #eee;">
<td style="padding:8px 12px; font-weight:bold;">Scaling behavior</td>
<td style="padding:8px 12px;">Clean power-law loss curves</td>
<td style="padding:8px 12px;">Less predictable at scale</td>
</tr>
<tr style="border-bottom:1px solid #eee;">
<td style="padding:8px 12px; font-weight:bold;">Diversity</td>
<td style="padding:8px 12px;">Prefix mode collapse (99.8% same first word in controlled experiments)</td>
<td style="padding:8px 12px;">Higher diversity (36.1% unique first words)</td>
</tr>
<tr>
<td style="padding:8px 12px; font-weight:bold;">Overfitting</td>
<td style="padding:8px 12px;">Overfits faster (optimal at ~14K steps in controlled comparison)</td>
<td style="padding:8px 12px;">Underfits longer (still improving at 20K steps)</td>
</tr>
</tbody>
</table>
<div style="font-size:12px; color:#666; margin-top:12px;">Source: Controlled comparison from "Autoregressive vs. Masked Diffusion Language Models" (arXiv 2603.22075, 2026)</div>
</div>

**The tradeoff that matters:** AR models are more training-efficient (per FLOP, per token) and generate text naturally. But they commit to tokens left-to-right without lookahead — no backtracking, no revision. This **sequential commitment** is the fundamental architectural limitation that drives research into chain-of-thought prompting, scratchpads, multi-token prediction, and reasoning-specific training (topics we'll cover in Ch 12, Ch 19, and Ch 23).

---

## 4. Cross-Entropy Loss

The standard training objective for autoregressive LLMs. Given a training sequence $x_1, \ldots, x_T$:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

This is the **average negative log-likelihood (NLL)** of the data under the model. At each position $t$, the model outputs a distribution over the vocabulary, and we penalize it by $-\log P_\theta(x_t \mid x_{<t})$ — the negative log probability assigned to the correct next token.

### Why Cross-Entropy? The Three-Way Equivalence

Cross-entropy is not an arbitrary choice. Three different frameworks converge on the same objective:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="display:flex; justify-content:center; gap:40px; flex-wrap:wrap; align-items:center;">
<div style="background:#16213e; padding:16px 20px; border-radius:10px; text-align:center; min-width:180px;">
<div style="color:#e94560; font-weight:bold; font-size:15px;">Cross-Entropy</div>
<div style="color:#aaa; font-size:12px; margin-top:6px;">H(P, Q) = -∑ P(x) log Q(x)</div>
<div style="color:#666; font-size:11px; margin-top:4px;">Information theory</div>
</div>
<div style="color:#e94560; font-size:24px; font-weight:bold;">=</div>
<div style="background:#16213e; padding:16px 20px; border-radius:10px; text-align:center; min-width:180px;">
<div style="color:#e94560; font-weight:bold; font-size:15px;">KL Divergence + const</div>
<div style="color:#aaa; font-size:12px; margin-top:6px;">H(P) + D_KL(P ∥ Q)</div>
<div style="color:#666; font-size:11px; margin-top:4px;">Divergence minimization</div>
</div>
<div style="color:#e94560; font-size:24px; font-weight:bold;">=</div>
<div style="background:#16213e; padding:16px 20px; border-radius:10px; text-align:center; min-width:180px;">
<div style="color:#e94560; font-weight:bold; font-size:15px;">Negative Log-Likelihood</div>
<div style="color:#aaa; font-size:12px; margin-top:6px;">-∑ log P_θ(x_t | x_&lt;t)</div>
<div style="color:#666; font-size:11px; margin-top:4px;">Maximum likelihood</div>
</div>
</div>
<div style="color:#888; font-size:12px; margin-top:16px; text-align:center; font-family:sans-serif;">
Since H(P) is constant w.r.t. model parameters θ, minimizing cross-entropy ≡ minimizing KL divergence ≡ maximum likelihood estimation.
</div>
</div>

The key decomposition:

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

where $H(P)$ is the entropy of the data distribution (a constant — the irreducible uncertainty in language). Since $H(P)$ doesn't depend on model parameters $\theta$, minimizing cross-entropy $H(P, Q_\theta)$ is **exactly equivalent** to minimizing KL divergence $D_{KL}(P \| Q_\theta)$, which is exactly equivalent to maximum likelihood estimation.

Three frameworks, one gradient. This convergence is why cross-entropy is the universal choice — you're not picking among competing objectives, they're all the same.

### A Subtlety Worth Noting

Cross-entropy is **not a metric** in the mathematical sense. It violates all three metric axioms: $H(P, P) = H(P) \neq 0$ in general (non-negativity at identity fails), $H(P, Q) \neq H(Q, P)$ (symmetry fails), and it doesn't satisfy the triangle inequality. It works as a loss function for gradient descent, but interpreting it as a "distance between distributions" is imprecise. KL divergence is closer (it's zero iff $P = Q$) but also asymmetric.

*Source: Eli Bendersky's derivation ([[bendersky-cross-entropy|blog]]) at eli.thegreenplace.net*

---

## 5. Perplexity

Perplexity is the standard evaluation metric for language models. It's the exponentiated cross-entropy:

$$\text{PPL}(X) = \exp\left\{-\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})\right\} = \exp(\mathcal{L})$$

**Intuition:** A perplexity of $k$ means the model is, on average, as uncertain as if it were choosing uniformly among $k$ tokens at each step. Lower is better:
- PPL = 1: perfect prediction (model assigns probability 1 to every correct next token)
- PPL = |V|: uniform random (model has learned nothing)
- GPT-2 on WikiText: ~19.44 (naive) or ~16.44 (sliding window evaluation)

### Perplexity Pitfalls

**1. Evaluation methodology matters more than you'd think.** For fixed-context models like GPT-2 (1024 token window), naive perplexity computation — chunking the test set into non-overlapping blocks — gives PPL = 19.44. But sliding-window evaluation with stride 512 gives 16.44. A 15% improvement from evaluation methodology alone, not from any model change. When comparing perplexity numbers across papers, check how they evaluated.

**2. Perplexity is only well-defined for autoregressive models.** BERT and other masked LMs don't define $P(x_t \mid x_{<t})$, so you can't compute perplexity for them. This makes cross-paradigm comparison difficult — a recurring theme in LLM evaluation.

**3. Lower perplexity doesn't always mean better downstream performance.** Perplexity measures how well the model predicts the training distribution. But the tasks we care about (reasoning, coding, following instructions) are downstream properties. A model with PPL 10 might outperform a model with PPL 8 on reasoning benchmarks if its training data or architecture better captures compositional structure. Perplexity is necessary but not sufficient.

*Source: Hugging Face Perplexity Documentation ([[hf-perplexity|blog]])*

---

## 6. Teacher Forcing

During training, autoregressive models use **teacher forcing**: at each position, the model receives the ground-truth previous tokens as input (not its own predictions). This enables massive parallelism — all positions can be computed simultaneously via causal masking.

<div style="background:#f8f9fa; border-radius:12px; padding:24px; margin:20px 0; border:1px solid #dee2e6;">
<div style="font-weight:bold; margin-bottom:16px;">Training (Teacher Forcing) — Parallel</div>
<div style="display:flex; gap:4px; align-items:center; margin-bottom:8px; font-family:monospace; font-size:13px;">
<span style="background:#d4edda; padding:4px 8px; border-radius:4px;">The</span>
<span style="background:#d4edda; padding:4px 8px; border-radius:4px;">cat</span>
<span style="background:#d4edda; padding:4px 8px; border-radius:4px;">sat</span>
<span style="background:#d4edda; padding:4px 8px; border-radius:4px;">on</span>
<span style="background:#d4edda; padding:4px 8px; border-radius:4px;">the</span>
<span style="color:#666;">← all ground-truth inputs, processed in parallel</span>
</div>
<div style="display:flex; gap:4px; align-items:center; margin-bottom:20px; font-family:monospace; font-size:13px;">
<span style="background:#cce5ff; padding:4px 8px; border-radius:4px;">cat</span>
<span style="background:#cce5ff; padding:4px 8px; border-radius:4px;">sat</span>
<span style="background:#cce5ff; padding:4px 8px; border-radius:4px;">on</span>
<span style="background:#cce5ff; padding:4px 8px; border-radius:4px;">the</span>
<span style="background:#cce5ff; padding:4px 8px; border-radius:4px;">mat</span>
<span style="color:#666;">← all targets, loss computed simultaneously</span>
</div>

<div style="font-weight:bold; margin-bottom:16px;">Inference (Autoregressive) — Sequential</div>
<div style="display:flex; gap:4px; align-items:center; font-family:monospace; font-size:13px;">
<span style="background:#d4edda; padding:4px 8px; border-radius:4px;">The</span>
<span>→</span>
<span style="background:#fff3cd; padding:4px 8px; border-radius:4px;">cat</span>
<span>→</span>
<span style="background:#fff3cd; padding:4px 8px; border-radius:4px;">sat</span>
<span>→</span>
<span style="background:#fff3cd; padding:4px 8px; border-radius:4px;">...</span>
<span style="color:#666;">← each token depends on model's own previous output</span>
</div>
</div>

### The Train-Test Mismatch (Exposure Bias)

During training, the model always sees correct previous tokens. During inference, it sees its own (potentially incorrect) previous predictions. Errors compound: one wrong token shifts the distribution for all subsequent tokens.

This mismatch is called **exposure bias**, and the conventional wisdom is that it causes error accumulation during generation. But the story is more nuanced than that:

**The conventional view:** "Teacher forcing is fine for training; problems only appear at inference due to exposure bias."

**The deeper problem (Bachmann & Nagarajan, ICML 2024 — [[pitfalls-next-token|paper]]):** Teacher forcing can cause failure **even in-distribution**. On planning tasks, teacher forcing lets the model "cheat" — it can fit surface statistics of the training data (e.g., predicting the next step of a plan by pattern matching) without learning the actual computational procedure that generates the plan. The model achieves low training loss but hasn't learned the intended function.

This is not just a theoretical concern. It's directly relevant to why chain-of-thought prompting works: it converts "predict the answer directly" (which may require implicit planning the AR model can't do) into "predict the next reasoning step" (which is more locally predictable).

*Source: "The Pitfalls of Next-Token Prediction" (Bachmann & Nagarajan, ICML 2024)*

---

## 7. Why Autoregressive Modeling Works So Well

For someone heading toward architecture research, here's the multi-level answer:

**Information-theoretic:** The chain rule factorization is exact. Any joint distribution can be decomposed into conditionals. An AR model with a sufficiently expressive conditional estimator can represent any sequence distribution — the approximation is only in the neural network's capacity, not in the factorization.

**Computational universality:** Schick et al. (2024) proved that autoregressive decoding of a Transformer constitutes a universal Turing machine via 2,027 production rules. AR LLMs are not "just" pattern matchers — they are provably general-purpose computers when the context window can grow. This is a theoretical ceiling result, not a practical claim, but it establishes that no expressiveness is lost in principle.

**Training efficiency:** Every token position provides gradient signal (vs 15% for MLM). Causal masking enables full parallelization during training even though generation is sequential. The ratio of useful gradient signal to compute is maximized.

**Scaling predictability:** AR models have shown the most consistent scaling laws (Kaplan et al. 2020, Hoffmann et al. 2022). The loss-compute relationship follows a clean power law, enabling efficient resource allocation. This predictability is a practical advantage for labs spending millions on training — you can forecast whether a run will hit your target loss before committing full resources.

### The Fundamental Limitation

AR models lack an explicit planning mechanism. They commit to tokens sequentially without lookahead — no backtracking, no revision. This creates two problems:

1. **Diversity collapse:** Left-to-right commitment causes prefix mode collapse. In controlled experiments, 99.8% of AR-generated samples began with the same word.

2. **Planning failure:** There exist task classes where sequential commitment provably prevents learning the correct solution, even with perfect training (Bachmann & Nagarajan, 2024). The model can achieve low loss on the training data by fitting surface statistics without learning the actual computation.

**This tension — "AR is the best objective we have for scaling, but it has a fundamental planning limitation" — is the central open question you should carry through the entire course.** It motivates:
- Chain-of-thought and scratchpads (Ch 12)
- Multi-token prediction (Ch 19, DeepSeek-V3)
- Dual-mode thinking/non-thinking (Ch 23, Qwen 3)
- The broader question of inference-time compute allocation

---

## Core Insights from the Literature

### Insight 1: Training efficiency wins at scale, not representational power
**Paper:** Comparison of AR vs MLM training paradigms

BERT's bidirectional context is strictly more informative per token than AR's left-only context. But AR models compute loss on 100% of tokens vs ~15% for MLM. At scale, this 6.7× difference in gradient signal per sequence overwhelms the representational advantage. **Guideline:** When choosing an objective for large-scale pre-training, training signal density per FLOP matters more than per-token information access.

### Insight 2: Low training loss ≠ learned the intended function
**Paper:** Bachmann & Nagarajan, "The Pitfalls of Next-Token Prediction" ([[pitfalls-next-token|paper]]) (ICML 2024)

Teacher forcing lets models achieve low loss by fitting surface statistics without learning the actual computation that generates the data. This isn't just "exposure bias at inference" — it's a training-time failure where the model finds a shortcut that works on the training distribution but doesn't generalize to the task's actual structure. **Guideline:** When evaluating whether a model has "learned" a capability, low loss is necessary but not sufficient. Probe the model's reasoning process, not just its outputs.

### Insight 3: AR's planning limitation is fundamental, not incidental
**Paper:** Multiple — the convergence of chain-of-thought, scratchpads, and multi-token prediction research

Sequential left-to-right commitment means AR models cannot natively plan ahead. This is not a bug to fix but a structural property of the factorization. Every technique that improves LLM reasoning (CoT, scratchpads, multi-token prediction, thinking modes) works by converting implicit planning into explicit sequential steps. **Guideline:** Architecture innovations that help LLMs "think before speaking" are addressing a fundamental limitation, not an implementation gap. This is the single most important open problem in LLM architecture.

---

## Key Takeaways

1. **Language modeling = assigning probabilities to sequences.** Everything else is engineering and architecture choices in service of this.

2. **The autoregressive factorization is exact.** The modeling assumption is that a neural network can approximate each conditional well — the factorization itself is the chain rule, not an approximation.

3. **Cross-entropy = KL divergence + constant = negative log-likelihood.** Three frameworks, one gradient. This is why it's the universal training objective.

4. **Perplexity = exp(cross-entropy).** Useful but imperfect — evaluation methodology affects it, it's undefined for masked LMs, and lower PPL doesn't guarantee better downstream performance.

5. **Teacher forcing enables parallel training** but creates exposure bias at inference and can cause in-distribution failure on planning tasks.

6. **AR's strength (training efficiency, scaling predictability) comes with a cost (sequential commitment, no planning).** This tradeoff drives most of the architecture innovations you'll study in this course.

---

## References

- Vaswani et al. "Attention Is All You Need" (2017) — Ch 3 deep dive
- Radford et al. "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018)
- Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- Brown et al. "Language Models are Few-Shot Learners" (GPT-3, 2020)
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- [[pitfalls-next-token|Bachmann & Nagarajan "The Pitfalls of Next-Token Prediction" (ICML 2024) — paper]]
- Schick et al. "Autoregressive LLMs are Computationally Universal" (2024)
- "Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison" (arXiv 2603.22075, 2026)
- [[bendersky-cross-entropy|Eli Bendersky: "Cross-Entropy and KL Divergence" (eli.thegreenplace.net) — blog]]
- [[hf-perplexity|Hugging Face: "Perplexity of Fixed-Length Models" (huggingface.co/docs) — blog]]
- Sebastian Raschka: "How Does Next-Token Prediction Train an LLM?" (sebastianraschka.com)
