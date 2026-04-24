# Chapter 10: Scaling Laws

<!-- scope: power-law scaling relationships, compute-optimal training, Chinchilla revision, emergent abilities debate, inference-time scaling
     deps: [[ch-04]]
     see-also: [[ch-11]], [[ch-12]], [[ch-13]]
-->

## Overview

Scaling laws are the empirical equations that govern how language model performance improves as you increase parameters, data, and compute. Before Kaplan et al. (2020), training large models was largely intuition-driven: researchers picked a model size, gathered what data they could, and trained until the loss curve flattened. Scaling laws replaced intuition with prediction. They showed that cross-entropy loss follows clean power laws across seven orders of magnitude, enabling teams to forecast the performance of a 100B-parameter run from experiments at 100M parameters -- before committing millions of dollars in compute.

But scaling laws are not a single, settled result. The field has undergone a major correction: Kaplan et al. recommended allocating most of a compute budget to model size (train large, stop early), while Hoffmann et al. (2022) -- the Chinchilla paper -- showed this was wrong by a factor of ~5x in the data dimension. Chinchilla demonstrated that parameters and data should scale equally, and that most frontier models were severely undertrained. This single revision reshaped the entire industry: LLaMA, Mistral, and every major open-weight model that followed are direct descendants of Chinchilla's insight.

This chapter covers the core scaling results, the Chinchilla correction, how the field responded (LLaMA as a case study), the emergent abilities debate (real phase transitions or measurement artifacts?), the relationship between loss and downstream capabilities, and the emerging frontier of inference-time scaling -- where you spend compute at test time rather than training time.

---

## 1. Power-Law Scaling: The Kaplan et al. Foundation

Kaplan et al. ([[scaling-laws-kaplan|paper]]) established three foundational power-law relationships for decoder-only Transformers trained on the cross-entropy loss:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

where $N$ is non-embedding parameter count, $D$ is dataset size in tokens, and $C$ is compute in FLOPs. The constants $N_c$, $D_c$, $C_c$ are fitted intercepts.

These are not merely curve fits. They hold across **seven orders of magnitude** -- from models with thousands of parameters to billions -- with remarkably little scatter. The smoothness of these relationships is what makes them useful: you can train a suite of small models, fit the power law, and extrapolate to predict the loss of a model 1000x larger.

### What the Exponents Mean

The exponents encode how efficiently each resource converts into loss reduction:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Diminishing Returns: 10x Each Resource</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Resource</th>
<th style="text-align:right; padding:8px;">Exponent</th>
<th style="text-align:right; padding:8px;">Loss reduction per 10x</th>
<th style="text-align:left; padding:8px 8px 8px 16px;">Interpretation</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Parameters (N)</td>
<td style="text-align:right; padding:8px;">0.076</td>
<td style="text-align:right; padding:8px; color:#e94560;">~16%</td>
<td style="padding:8px 8px 8px 16px;">10x more params -> 10^0.076 ~ 1.19x lower loss</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">Data (D)</td>
<td style="text-align:right; padding:8px;">0.095</td>
<td style="text-align:right; padding:8px; color:#4ecdc4;">~20%</td>
<td style="padding:8px 8px 8px 16px;">10x more data -> 10^0.095 ~ 1.24x lower loss</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">Compute (C)</td>
<td style="text-align:right; padding:8px;">0.050</td>
<td style="text-align:right; padding:8px; color:#ffd93d;">~11%</td>
<td style="padding:8px 8px 8px 16px;">10x more compute -> 10^0.050 ~ 1.12x lower loss</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
Compute exponent is smallest because C = 6ND (approximately), so increasing C means increasing N and D jointly, and the compound effect is weaker per unit of C than per unit of N or D individually.
</div>
</div>

### Architecture Does Not Matter (Much)

One of Kaplan's most provocative findings: **within a wide range, architectural details have minimal effect on loss**. Width-to-depth ratio, number of attention heads, and feed-forward dimension -- none of these significantly alter the scaling curve as long as the total parameter count is held fixed. The loss is determined overwhelmingly by three numbers: $N$, $D$, and $C$.

This is the "scaling hypothesis" in its strongest form: architecture is a second-order effect. What matters is scale. This finding was instrumental in justifying the massive compute investments of 2020-2023, and it holds approximately -- though not perfectly -- across the Transformer family. (It does not hold across fundamentally different architectures like state-space models, which have different scaling exponents.)

### The Kaplan Allocation Rule (Later Revised)

Given a fixed compute budget $C$, how should you split it between model size and training duration? Kaplan recommended:

$$\text{For a 10x increase in } C: \quad N \times 5.5, \quad D \times 1.8$$

In other words: **most of the budget should go to making the model bigger, not training it longer**. This led to the "train big, stop early" paradigm -- train very large models on relatively modest data and halt well before convergence. GPT-3 (175B parameters, 300B tokens) was designed under this philosophy.

This recommendation was wrong. Chinchilla would later show the correct split is approximately equal.

---

## 2. The Chinchilla Correction: Compute-Optimal Training

Hoffmann et al. ([[chinchilla|paper]]) ran over 400 models ranging from 70M to 16B parameters on 5B to 500B tokens, using three independent estimation methods to derive the compute-optimal frontier:

1. **IsoFLOP profiles:** Fix compute, vary model size, find the $N$ that minimizes loss for each budget
2. **IsoLoss contours:** Fix target loss, find the minimum compute required
3. **Parametric fit:** Fit the joint loss function $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$

All three methods converged on the same conclusion:

$$N_{\text{opt}} \propto C^{0.5}, \qquad D_{\text{opt}} \propto C^{0.5}$$

**Parameters and data should scale equally with compute.** Doubling your compute budget means you should use a model $\sqrt{2}\times$ larger trained on $\sqrt{2}\times$ more tokens.

### How Chinchilla Proved It

The proof was not merely theoretical. DeepMind trained **Chinchilla** (70B parameters, 1.4T tokens) using the same compute budget as their earlier Gopher model (280B parameters, 300B tokens). The result was unambiguous:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Chinchilla (70B) vs. Gopher (280B): Same Compute, Different Allocation</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Benchmark</th>
<th style="text-align:right; padding:8px;">Gopher (280B, 300B tok)</th>
<th style="text-align:right; padding:8px;">Chinchilla (70B, 1.4T tok)</th>
<th style="text-align:right; padding:8px;">GPT-3 (175B)</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">MMLU (5-shot)</td>
<td style="text-align:right; padding:8px;">60.0%</td>
<td style="text-align:right; padding:8px; color:#4ecdc4; font-weight:bold;">67.5%</td>
<td style="text-align:right; padding:8px;">43.9%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Inference cost</td>
<td style="text-align:right; padding:8px;">1.0x (baseline)</td>
<td style="text-align:right; padding:8px; color:#4ecdc4; font-weight:bold;">0.25x</td>
<td style="text-align:right; padding:8px;">0.63x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Model parameters</td>
<td style="text-align:right; padding:8px;">280B</td>
<td style="text-align:right; padding:8px; color:#4ecdc4; font-weight:bold;">70B</td>
<td style="text-align:right; padding:8px;">175B</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Training tokens</td>
<td style="text-align:right; padding:8px;">300B</td>
<td style="text-align:right; padding:8px; color:#4ecdc4; font-weight:bold;">1,400B</td>
<td style="text-align:right; padding:8px;">300B</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
Chinchilla is 4x smaller than Gopher but beats it on every benchmark, using the same training compute. The 4x parameter reduction translates to 4x cheaper inference permanently.
</div>
</div>

A 70B model trained on 1.4T tokens uniformly outperformed a 280B model trained on 300B tokens -- despite using identical compute. The 4x parameter reduction also means Chinchilla is 4x cheaper to serve at inference time. This is the practical power of compute-optimal training: you get a better model that is also cheaper to deploy.

### Where Kaplan Went Wrong

The disagreement between Kaplan and Chinchilla comes down to methodology. Kaplan fixed the learning rate schedule and varied model size, but did not independently vary training duration at each model size. This created a confound: larger models were implicitly being compared at more favorable points on their learning rate schedules. Chinchilla controlled for this by training each model size for multiple durations, revealing that Kaplan had systematically underestimated the value of more data.

The corrected allocation for a 10x compute increase:

$$\text{Kaplan: } N \times 5.5,\; D \times 1.8 \qquad \longrightarrow \qquad \text{Chinchilla: } N \times 3.16,\; D \times 3.16$$

[Interactive: Compute-Optimal Frontier Explorer](figures/compute-optimal-frontier.html)

### The Chinchilla Tax

Chinchilla's insight imposed a retroactive "tax" on every model trained before it. GPT-3 (175B params, 300B tokens) was undertrained by a factor of ~6x in data -- the Chinchilla-optimal token count for 175B parameters is approximately 3.5T tokens. Gopher (280B, 300B tokens) was undertrained by ~15x. Every GPU-hour spent training these models beyond the Chinchilla frontier was, in a precise sense, wasted.

---

## 3. LLaMA: The Chinchilla Response

Meta's LLaMA ([[llama-1|report]]) was the most direct practical response to the Chinchilla scaling laws. The design philosophy was explicit: **train smaller models on far more data than Chinchilla-optimal**, optimizing for inference cost rather than training compute.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">LLaMA: Beyond Chinchilla-Optimal Training</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Model</th>
<th style="text-align:right; padding:8px;">Params</th>
<th style="text-align:right; padding:8px;">Tokens</th>
<th style="text-align:right; padding:8px;">Chinchilla-optimal tokens</th>
<th style="text-align:right; padding:8px;">Overtrain factor</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">LLaMA-7B</td>
<td style="text-align:right; padding:8px;">7B</td>
<td style="text-align:right; padding:8px;">1.0T</td>
<td style="text-align:right; padding:8px;">~140B</td>
<td style="text-align:right; padding:8px; color:#ffd93d;">~7x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">LLaMA-13B</td>
<td style="text-align:right; padding:8px;">13B</td>
<td style="text-align:right; padding:8px;">1.0T</td>
<td style="text-align:right; padding:8px;">~260B</td>
<td style="text-align:right; padding:8px; color:#ffd93d;">~4x</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">LLaMA-65B</td>
<td style="text-align:right; padding:8px;">65B</td>
<td style="text-align:right; padding:8px;">1.4T</td>
<td style="text-align:right; padding:8px;">~1.3T</td>
<td style="text-align:right; padding:8px; color:#ffd93d;">~1.1x</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
LLaMA-7B was trained on ~7x its Chinchilla-optimal data. This deliberately "wastes" training compute to produce a smaller model that is cheaper to serve.
</div>
</div>

The result was striking: **LLaMA-13B outperformed GPT-3 (175B) on most benchmarks despite being 13x smaller.** The key benchmarks:

| Benchmark | LLaMA-13B | GPT-3 (175B) |
|-----------|-----------|--------------|
| MMLU (5-shot) | 46.9% | 43.9% |
| HellaSwag | 79.2% | 78.9% |
| NaturalQuestions | 27.1% | 29.9% |

LLaMA's design made a deliberate tradeoff: spend more training compute (overtraining relative to Chinchilla) to get a smaller, cheaper-to-serve model. This is the correct strategy when inference cost dominates your total compute bill -- which it does for any model that serves real traffic. A model that is 10x cheaper to serve but costs 3x more to train pays for itself within weeks of deployment.

### The Architecture Was Not Novel

LLaMA's architecture combined pre-existing techniques: RMSNorm (pre-normalization), SwiGLU activations, RoPE, and training exclusively on public data. None of these were new. The contribution was showing that **the right combination of known techniques, trained at Chinchilla-optimal or beyond on quality data, beats larger models with proprietary data**. This validated the scaling hypothesis from a different angle: architecture innovation matters less than data quality and scale.

---

## 4. Scaling in Data-Constrained Regimes

Chinchilla assumed unlimited data. Muennighoff et al. ([[scaling-data-constrained|paper]]) asked the natural follow-up: what happens when you run out of unique data?

The answer, from 400 training runs up to 900B tokens and 9B parameters: **data repetition has sharply diminishing returns**.

- **Up to 4 epochs:** Negligible loss degradation compared to unique data
- **4-16 epochs:** Measurable but manageable degradation
- **Beyond 16 epochs:** The value of additional compute decays toward zero

This matters because we are approaching real data limits. Current estimates suggest the total stock of high-quality text on the internet is on the order of 10-20T tokens (after deduplication and quality filtering). Frontier models are already training on 10T+ tokens. The implication is that naively scaling data further will hit diminishing returns from repetition, requiring new strategies: synthetic data generation, multimodal data, or more efficient data utilization.

The paper also showed that augmenting training data with **code** partially mitigates data scarcity -- code data has higher information density per token and introduces structured reasoning patterns that transfer to natural language tasks. Relaxing quality filters also helps, though with quality-diversity tradeoffs.

See [[ch-12]] for detailed treatment of data curation and [[ch-13]] for synthetic data approaches.

---

## 5. The Emergent Abilities Debate

Wei et al. ([[emergent-abilities|paper]]) (2022) introduced one of the most contested ideas in the scaling era: that certain capabilities **emerge** at scale -- they are absent in smaller models and appear suddenly in larger models, without being predictable from extrapolation.

### The Claim

The paper catalogued dozens of tasks where models below a certain scale performed at chance level, then jumped to high accuracy above that scale. Examples included multi-step arithmetic, chain-of-thought reasoning, and certain BIG-Bench tasks. The defining property: emergence cannot be predicted by extrapolating from smaller models. The loss curve is smooth, but capabilities appear discontinuously.

If true, emergence has profound implications:
- **You cannot predict what a larger model will be able to do** by studying smaller ones
- **Scaling is more valuable than it appears** from loss curves alone, because qualitative jumps may await at the next order of magnitude
- **Safety becomes harder** because dangerous capabilities might appear suddenly and without warning

### The Rebuttal

Schaeffer, Miranda, and Koyejo ([[emergent-mirage|paper]]) (2023) proposed a much simpler explanation: **emergence is a measurement artifact, not a model property**. The argument is elegant:

1. Nonlinear or discontinuous metrics (like exact-match accuracy) produce apparent sharp transitions
2. Linear or continuous metrics (like per-token accuracy or Brier score) show smooth, predictable improvement at all scales
3. The choice of metric alone can create or destroy the appearance of emergence

[Interactive: Emergence vs. Metric Choice](figures/emergence-metric-explorer.html)

The mathematical intuition: consider a model whose per-token probability of generating the correct answer improves smoothly with scale, say from 0.1 to 0.99 over a 100x increase in parameters. If you measure **exact-match accuracy** on a multi-token answer (all tokens must be correct), the measured accuracy is the product of per-token probabilities. For a 10-token answer:

$$\text{Exact-match} = p^{10}$$

When $p = 0.5$: exact-match $= 0.001$ (appears to fail). When $p = 0.95$: exact-match $= 0.60$ (appears to succeed). The underlying improvement is smooth ($p$ from 0.5 to 0.95), but the measured metric shows a sharp jump from ~0 to ~0.6. The "emergence" is an artifact of the nonlinear metric, not a property of the model.

Schaeffer et al. confirmed this empirically: for every task with claimed emergence, switching to a continuous metric eliminated the sharp transition. They also showed the reverse: by choosing sufficiently nonlinear metrics, they could manufacture apparent emergence in **vision tasks** where none had been reported.

### The Nuanced View

The debate is not fully resolved. Several considerations:

1. **Some capabilities may genuinely require threshold scale.** In-context learning with many exemplars, multi-step logical reasoning, and self-correction may require a minimum circuit complexity that only exists above certain model sizes. The metric-artifact argument explains away *measured* discontinuities, but it does not prove that all capabilities scale smoothly.

2. **The practical impact is the same either way.** Whether emergence is "real" or a measurement artifact, the practical fact remains: certain tasks are only solved well by models above a certain size. For resource allocation decisions, what matters is the empirical curve on your target benchmark, not the philosophical question of whether the jump is "real."

3. **Loss is always smooth; task performance may not be.** Cross-entropy loss scales as a smooth power law at every scale tested. But the mapping from loss to task performance is mediated by the evaluation metric and the task structure. A smooth decrease in loss can produce arbitrarily sharp transitions in downstream metrics depending on how those metrics aggregate errors.

See the detailed analysis in [[ch-10/excerpts/emergent-abilities-deep-dive]].

---

## 6. Scaling Laws for Downstream Tasks

The power-law equations describe cross-entropy loss. But practitioners care about downstream performance: MMLU accuracy, HumanEval pass@1, GSM8K solve rate. Does lower loss always mean better capabilities?

### Loss as a Proxy

Empirically, **loss is a reliable proxy for average downstream performance** across a broad benchmark suite. The Chinchilla results show this clearly: Chinchilla's lower loss (from compute-optimal training) translated to uniformly better downstream scores than Gopher's.

But the relationship between loss and any *specific* task is noisier. A model with 0.1 nats lower loss might improve by 5 percentage points on MMLU but 0 points on a particular reasoning task. The mapping depends on:

1. **How much of the loss reduction comes from the task-relevant distribution.** If the improved loss is entirely from better prediction of common web text, it may not help with mathematical reasoning.
2. **The difficulty threshold of the task.** Some tasks have a minimum capability threshold below which performance is at chance regardless of loss.
3. **The metric used to evaluate.** As the emergence debate shows, nonlinear metrics can create artificial discontinuities in the loss-to-performance mapping.

### The Implications for Model Selection

This means scaling laws alone cannot tell you whether a model will succeed on your target task. They tell you the *expected* loss, which is a useful lower bound on capability, but the loss-to-task-performance mapping must be calibrated empirically for each task of interest. The practical protocol: train a few models at small scale, evaluate on your target tasks, fit the scaling curve for *task performance* (not just loss), and extrapolate.

---

## 7. Inference-Time Scaling: A New Axis

The scaling laws discussed so far are all about **training-time compute**: how much to spend on pre-training. But a fundamentally different scaling axis has emerged: **inference-time compute** -- spending more computation when generating each answer.

### The Core Idea

Weng ([[weng-why-we-think|blog]]) frames inference-time scaling through the lens of dual-process theory. Standard autoregressive generation is "System 1" -- fast, pattern-matching, one forward pass per token. Inference-time scaling adds "System 2" -- deliberate, multi-step, using additional compute per answer to improve quality.

Mechanisms for inference-time scaling include:

- **Chain-of-thought prompting:** Generating intermediate reasoning steps before the final answer. Each reasoning token costs ~$2N$ FLOPs (where $N$ is parameter count), so a 100-token chain-of-thought on a 70B model costs ~14 TFLOPs of additional inference compute.
- **Best-of-N sampling:** Generate $N$ independent answers, select the best one via a reward model. Linear scaling in $N$.
- **Beam search with process reward models:** Maintain $B$ candidate reasoning paths, score each step with a learned verifier, prune low-scoring branches.
- **Budget forcing:** Deliberately lengthening reasoning by inserting "wait" tokens, which shows strong positive correlation between thinking tokens and accuracy.

### Scaling Laws for Test-Time Compute

Snell et al. (2024), as discussed by Weng, established initial scaling relationships for inference-time compute:

- **Test-time and training-time compute are NOT 1:1 exchangeable.** A 14x smaller model with test-time sampling roughly matches a base model with greedy decoding, but only when the inference token budget stays below the pre-training token budget.
- **Effectiveness depends on problem difficulty.** Easy and medium problems benefit substantially from inference-time compute. Hard problems -- those requiring capabilities the model fundamentally lacks -- show minimal benefit regardless of inference budget.
- **Sequential vs. parallel compute matter differently.** Easier questions benefit from purely sequential test-time compute (longer chains of thought). Harder questions often perform best with an optimal ratio of sequential to parallel compute (multiple shorter chains, then aggregation).

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Training-Time vs. Inference-Time Scaling: Two Orthogonal Axes</div>
<div style="display:flex; gap:24px; flex-wrap:wrap; justify-content:center;">
<div style="flex:1; min-width:220px; background:#16213e; border-radius:8px; padding:16px; border-left:4px solid #e94560;">
<div style="color:#e94560; font-weight:bold; font-size:13px; margin-bottom:10px;">Training-Time Scaling</div>
<div style="color:#e0e0e0; font-size:12px; line-height:1.6;">
- More parameters (N)<br>
- More data (D)<br>
- More FLOPs (C = 6ND)<br>
- <span style="color:#ffd93d;">Cost paid once</span><br>
- Affects all queries equally<br>
- Power-law returns: L ~ C^{-0.05}
</div>
</div>
<div style="flex:1; min-width:220px; background:#16213e; border-radius:8px; padding:16px; border-left:4px solid #4ecdc4;">
<div style="color:#4ecdc4; font-weight:bold; font-size:13px; margin-bottom:10px;">Inference-Time Scaling</div>
<div style="color:#e0e0e0; font-size:12px; line-height:1.6;">
- More reasoning tokens<br>
- Multiple candidate answers<br>
- Beam search / MCTS<br>
- <span style="color:#ffd93d;">Cost paid per query</span><br>
- Can adapt to difficulty<br>
- Returns plateau on hard problems
</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center;">
Modern reasoning models (o1, DeepSeek-R1) operate on both axes simultaneously: heavy training-time investment to learn reasoning patterns, plus variable inference-time compute per query.
</div>
</div>

### DeepSeek-R1: Reasoning Emerges from RL

Raschka ([[raschka-reasoning-llms|blog]]) documents how DeepSeek-R1-Zero demonstrated that **reasoning emerges as a behavior from pure reinforcement learning** -- no supervised fine-tuning, no chain-of-thought demonstrations. The base model (DeepSeek-V3, 671B parameters) was trained with only accuracy rewards (compiler correctness for code, deterministic checking for math) and format rewards (reasoning wrapped in XML tags). The model spontaneously began generating reasoning traces and self-correction behaviors.

This is scaling in a qualitatively different sense: rather than scaling parameters or data, you scale the *behavior space* through reward signals. The model learns to allocate variable inference compute per problem by generating longer reasoning chains for harder problems.

The scale-dependent findings are instructive:

| Model Scale | Best Reasoning Approach |
|-------------|------------------------|
| Large (671B) | Pure RL can induce reasoning |
| Medium (32B) | SFT outperforms pure RL; SFT + RL likely optimal |
| Small (<10B) | Distillation from larger models most practical |

Pure RL requires sufficient model capacity to discover reasoning strategies through exploration. At smaller scales, it is more efficient to distill reasoning behaviors from a larger model.

[Interactive: Training vs. Inference Compute Tradeoff](figures/train-vs-inference-scaling.html)

---

## Core Insights from the Literature

### Insight 1: Loss follows power laws, but the optimal allocation was initially miscalibrated
**Paper:** Kaplan et al., "Scaling Laws for Neural Language Models" ([[scaling-laws-kaplan|paper]])

Kaplan established that loss scales as clean power laws with parameters, data, and compute across seven orders of magnitude. But the compute-allocation recommendation (spend mostly on model size) was wrong by a factor of ~3x in the data dimension. The methodology confound -- not independently varying training duration at each model size -- led to systematically underestimating data's value. **Guideline:** When deriving scaling laws, ensure all axes (model size, data, training duration) are varied independently. Confounds in the experimental design propagate into allocation recommendations that can waste millions of dollars in compute.

### Insight 2: Parameters and data must scale equally -- and most models were undertrained
**Paper:** Hoffmann et al., "Training Compute-Optimal Large Language Models" ([[chinchilla|paper]])

Chinchilla's core result ($N_\text{opt} \propto C^{0.5}$, $D_\text{opt} \propto C^{0.5}$) was validated by three independent estimation methods and confirmed by the Chinchilla model itself outperforming 4x-larger Gopher. The implication was that GPT-3, Gopher, and Megatron-Turing NLG were all significantly undertrained relative to their parameter count. **Guideline:** For a fixed compute budget, allocate equally to parameters and data. If you plan to serve the model at scale, consider overtraining beyond Chinchilla-optimal (as LLaMA did) to trade training compute for inference efficiency.

### Insight 3: Data repetition has sharply diminishing returns after ~4 epochs
**Paper:** Muennighoff et al., "Scaling Data-Constrained Language Models" ([[scaling-data-constrained|paper]])

With the internet's stock of high-quality text estimated at 10-20T tokens, the Chinchilla-optimal data requirement for frontier models is approaching the supply limit. Repeating data beyond 4 epochs yields rapidly diminishing returns, and beyond 16 epochs the value of additional compute decays toward zero. Code augmentation partially mitigates scarcity. **Guideline:** Treat unique high-quality tokens as a depletable resource. Invest in data curation and deduplication before considering repetition. When data is exhausted, pivot to synthetic data or multimodal sources rather than adding more epochs.

### Insight 4: "Emergence" may be a property of metrics, not models
**Papers:** Wei et al. ([[emergent-abilities|paper]]) and Schaeffer et al. ([[emergent-mirage|paper]])

The claim that capabilities emerge suddenly at scale -- absent in small models, present in large ones, unpredictable from extrapolation -- was challenged by showing that nonlinear metrics (exact-match accuracy) produce artificial sharp transitions, while linear metrics (per-token accuracy, Brier score) reveal smooth, predictable improvement. This is a cautionary tale about evaluation methodology: the choice of how you *measure* a model's capability can create or destroy the appearance of qualitative phase transitions. **Guideline:** Always evaluate models at multiple scales using both continuous and discrete metrics. If you observe apparent emergence, check whether a more granular metric reveals smooth underlying improvement. Do not make scaling investment decisions based on discontinuities that may be measurement artifacts.

### Insight 5: Inference-time compute is a new scaling axis orthogonal to training
**Sources:** Weng, "Why We Think" ([[weng-why-we-think|blog]]) and Raschka, "Understanding Reasoning LLMs" ([[raschka-reasoning-llms|blog]])

Training-time scaling (more parameters, more data) improves the model's base capability. Inference-time scaling (chain-of-thought, best-of-N, beam search) improves per-query performance by spending more compute at generation time. These two axes are not interchangeable: a 14x smaller model with test-time compute can match a larger model on easy problems but not on hard ones. Reasoning can also emerge from pure RL at sufficient scale (671B), but requires distillation at smaller scales. **Guideline:** Consider the total compute budget across both training and inference. For tasks with variable difficulty, invest in inference-time scaling mechanisms (reward models, verifiers) that can allocate compute adaptively per query.

---

## Key Takeaways

1. **Loss follows power laws across seven orders of magnitude.** The relationships $L(N)$, $L(D)$, $L(C)$ are smooth, predictable, and enable forecasting large-scale performance from small experiments. Architecture details are second-order effects compared to total parameter count.

2. **Chinchilla corrected the compute allocation: parameters and data scale equally.** Kaplan's "train big, stop early" was wrong. For a 10x compute increase, both $N$ and $D$ should increase by $\sqrt{10} \approx 3.16\times$. Most pre-Chinchilla models (GPT-3, Gopher) were severely undertrained.

3. **Overtraining beyond Chinchilla-optimal is a valid strategy for inference efficiency.** LLaMA deliberately trained 7B models on 7x the Chinchilla-optimal data, producing models that are cheaper to serve and competitive with much larger alternatives. The right strategy depends on whether you are optimizing training cost or inference cost.

4. **Data repetition hits diminishing returns after ~4 epochs.** The finite supply of high-quality internet text is a binding constraint for scaling. Future scaling will require synthetic data, multimodal sources, or fundamentally new data-efficiency techniques.

5. **Apparent "emergence" of capabilities may be a metric artifact.** Nonlinear evaluation metrics create the illusion of sharp phase transitions; continuous metrics reveal smooth improvement. Evaluate at multiple scales with multiple metrics before concluding that a capability genuinely emerges.

6. **Inference-time scaling is orthogonal to training-time scaling.** Chain-of-thought, best-of-N, and RL-induced reasoning patterns add a second compute axis that can be allocated per-query. This axis is effective for easy/medium problems but does not substitute for training-time capability on hard problems.

7. **The practical implication of all scaling laws: measure first, then extrapolate.** Train a suite of small models, evaluate on your target tasks (not just loss), fit the scaling curve, and use it to decide whether the next 10x of compute is justified by the predicted improvement.

---

## References

- [[scaling-laws-kaplan|Kaplan et al., "Scaling Laws for Neural Language Models" (2020) (paper)]] -- power-law relationships for N, D, C
- [[chinchilla|Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) (paper)]] -- Chinchilla scaling laws, compute-optimal allocation
- [[scaling-data-constrained|Muennighoff et al., "Scaling Data-Constrained Language Models" (2023) (paper)]] -- data repetition, data-constrained scaling
- [[emergent-abilities|Wei et al., "Emergent Abilities of Large Language Models" (2022) (paper)]] -- emergence of capabilities at scale
- [[emergent-mirage|Schaeffer et al., "Are Emergent Abilities of Large Language Models a Mirage?" (2023) (paper)]] -- emergence as metric artifact
- [[llama-1|Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023) (report)]] -- Chinchilla-informed model design
- [[raschka-reasoning-llms|Raschka, "Understanding Reasoning LLMs" (2025) (blog)]] -- reasoning approaches and inference-time scaling
- [[weng-why-we-think|Weng, "Why We Think" (2025) (blog)]] -- test-time compute, chain-of-thought, continuous-space thinking
