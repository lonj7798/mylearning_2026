# Chapter 11: Pre-training

<!-- scope: data curation, training stability, curriculum learning, two-stage pre-training, synthetic data, catastrophic forgetting
     deps: [[ch-10]]
     see-also: [[ch-12]], [[ch-24]]
-->

## Overview

Scaling laws ([[ch-10]]) tell you *how much* data and compute to spend. Pre-training is *how you actually spend it* — and the gap between theory and practice is enormous. Two runs with identical architecture and token count can differ by 5+ points on downstream benchmarks depending on data quality, training stability interventions, and curriculum design.

This chapter covers the full pre-training pipeline: how raw web crawls become training data, how to keep a multi-week training run from diverging, whether the order you present data matters, and how synthetic data can substitute for (or complement) organic data. The core tension throughout is **data quality vs. data quantity** — a tradeoff that Chinchilla's equal-scaling law left unresolved because it treated all tokens as equally informative.

The evidence is drawn from four open training reports that collectively represent the state of the art in pre-training methodology: Llama 3 ([[llama-3|report]]) for industrial-scale data pipelines, OLMo 2 ([[olmo-2|report]]) for two-stage curriculum design, Phi-4 ([[phi-4|report]]) for synthetic data strategies, and the scaling-data-constrained paper ([[scaling-data-constrained|paper]]) for the economics of data repetition. Together, they paint a picture of pre-training as a data engineering problem at least as much as a modeling problem.

---

## 1. Data Curation: From Common Crawl to Training Tokens

The raw internet is not a training dataset. Common Crawl contains ~250 billion pages, but the vast majority is boilerplate, spam, duplicated content, or machine-generated text. Every frontier lab applies a multi-stage filtering pipeline, and the specifics of that pipeline matter more than most architectural choices.

### The Llama 3 Pipeline

Llama 3 ([[llama-3|report]]) trained on **15 trillion tokens** — the largest openly documented pre-training dataset at its time of release. The data pipeline is a five-stage funnel:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Llama 3 Data Pipeline: From Raw Web to Training Tokens</div>
<div style="display:flex; flex-direction:column; gap:8px; align-items:center;">
<div style="background:#0f3460; padding:10px 32px; border-radius:8px; color:#e94560; font-weight:bold; font-size:13px; width:80%; text-align:center;">
Stage 1: URL-level filtering (robots.txt, adult content, known spam domains)
</div>
<div style="color:#e94560; font-size:16px;">&#8595;</div>
<div style="background:#0f3460; padding:10px 32px; border-radius:8px; color:#4ecdc4; font-weight:bold; font-size:13px; width:70%; text-align:center;">
Stage 2: Text extraction + heuristic filters (language ID, line length, repetition ratio)
</div>
<div style="color:#4ecdc4; font-size:16px;">&#8595;</div>
<div style="background:#0f3460; padding:10px 32px; border-radius:8px; color:#ffd93d; font-weight:bold; font-size:13px; width:60%; text-align:center;">
Stage 3: Deduplication (MinHash + URL-level exact dedup)
</div>
<div style="color:#ffd93d; font-size:16px;">&#8595;</div>
<div style="background:#0f3460; padding:10px 32px; border-radius:8px; color:#e94560; font-weight:bold; font-size:13px; width:50%; text-align:center;">
Stage 4: Quality classifier (fastText trained on Wikipedia vs web)
</div>
<div style="color:#e94560; font-size:16px;">&#8595;</div>
<div style="background:#e94560; padding:12px 32px; border-radius:8px; color:#fff; font-weight:bold; font-size:13px; width:40%; text-align:center;">
15T tokens for training
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center;">
Each stage reduces volume by 30-80%. The quality classifier alone removes ~50% of documents that passed heuristic filters.
</div>
</div>

**Heuristic filters** are simple but effective. Llama 3 filters on: too many special characters, extreme line lengths (very short = boilerplate, very long = data dumps), high n-gram repetition ratio (machine-generated), proportion of non-alphabetic characters. These are cheap to compute and remove the most egregious garbage.

**Quality classifiers** are the key differentiator between labs. Llama 3 trains a fastText binary classifier with Wikipedia/curated-books as positive examples and random web samples as negatives. Documents are scored and thresholded. The threshold is a critical hyperparameter: too aggressive and you lose diversity; too permissive and you train on garbage. Meta reports experimenting extensively with this threshold on downstream benchmarks.

**Deduplication** matters more than most people expect. Training on duplicated data has two effects: (1) it wastes compute on repeated information, and (2) it causes the model to memorize specific sequences verbatim, increasing privacy risk and reducing generalization. Llama 3 applies both URL-level exact deduplication and document-level MinHash (locality-sensitive hashing) to catch near-duplicates.

### Data Mixing Ratios

Raw token counts tell an incomplete story. The *composition* of the training mix is arguably more important than the total volume:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Pre-training Data Composition: Llama 3 vs Phi-4</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Category</th>
<th style="text-align:right; padding:8px;">Llama 3 (405B)</th>
<th style="text-align:right; padding:8px;">Phi-4 (14B)</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">General web</td>
<td style="text-align:right; padding:8px;">~50%</td>
<td style="text-align:right; padding:8px;">~30%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">Code</td>
<td style="text-align:right; padding:8px;">~17%</td>
<td style="text-align:right; padding:8px;">~20%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">Math/reasoning</td>
<td style="text-align:right; padding:8px;">~25%</td>
<td style="text-align:right; padding:8px;">~10%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Synthetic data</td>
<td style="text-align:right; padding:8px;">~0%</td>
<td style="text-align:right; padding:8px;">~40%</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">Multilingual</td>
<td style="text-align:right; padding:8px;">~8%</td>
<td style="text-align:right; padding:8px;">—</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
Llama 3 favors organic data at massive scale. Phi-4 compensates for smaller model size with 40% synthetic data.
</div>
</div>

How do labs determine these ratios? Not from first principles — from extensive ablation experiments. Llama 3 reports using scaling-law experiments on smaller models to predict the optimal data mix for the 405B run. The key insight is that **the optimal mix depends on the model size and the target capability profile**: a model optimized for code generation needs different ratios than one optimized for multilingual dialogue.

For a deeper treatment of data mixing strategies and the deduplication algorithms, see [[ch-11-data-pipeline|excerpt]].

---

## 2. Training Stability: Keeping Multi-Week Runs Alive

A 405B-parameter training run on 16,384 H100 GPUs costs millions of dollars and runs for weeks. A single divergence event can waste days of compute. Training stability is not a nice-to-have — it is an existential concern for large-scale pre-training.

### Loss Spikes

Loss spikes are sudden, dramatic increases in training loss that can cause the model to diverge irreversibly. They have several known causes:

1. **Data quality outliers:** A batch containing unusually high-loss examples (very long documents, corrupted text, code in unexpected languages) can produce extreme gradients.
2. **Learning rate interactions:** If the learning rate is too high relative to the current loss landscape curvature, the optimizer takes steps that overshoot.
3. **Numerical instability in attention:** As training progresses, attention logits can grow unboundedly, causing softmax overflow. This is the problem QK-Norm solves ([[ch-09]]).
4. **Hardware failures:** A faulty GPU producing NaN gradients that propagate through all-reduce operations.

The standard mitigation stack: gradient clipping (cap gradient norm at 1.0), learning rate warmup (start with a tiny LR and ramp up), and z-loss regularization (penalize large logits before the softmax).

### OLMo 2's Stability Innovations

OLMo 2 ([[olmo-2|report]]) provides the most transparent account of training stability engineering in the open literature. Their stability stack includes:

- **QK-Norm:** Normalizing query and key vectors before the attention dot product prevents attention logit growth. This adds minimal compute overhead but eliminates a major source of loss spikes at scale.
- **Z-loss regularization:** An auxiliary loss term $\mathcal{L}_z = \alpha \cdot \log^2(\sum_i e^{z_i})$ penalizes the log-partition function of the final logits. This prevents the model from producing extremely large logit values, which can cause numerical instability in the cross-entropy loss computation.
- **Improved initialization:** Carefully setting initial weight scales to preserve activation and gradient magnitudes across layers. Poor initialization can cause gradients to vanish or explode within the first few hundred steps.
- **Switch from LayerNorm to RMSNorm:** OLMo 1 used non-parametric LayerNorm; OLMo 2 switched to RMSNorm for better stability and slightly lower compute cost ([[ch-09]]).

### Learning Rate Schedules

The learning rate schedule controls how aggressively the optimizer updates weights throughout training. Three schedules dominate modern pre-training:

$$\eta_{\text{cosine}}(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi \, t}{T}\right)\right)$$

**Cosine annealing** (used by Llama 3, Chinchilla, most models) smoothly decays the learning rate from $\eta_{\max}$ to $\eta_{\min}$ over $T$ total steps. The cosine shape provides aggressive early learning and gentle late-stage refinement.

**Warmup-Stable-Decay (WSD)** (used by some recent models) keeps the learning rate constant at $\eta_{\max}$ for the majority of training, then sharply decays in the final phase. The argument for WSD: cosine annealing reduces the learning rate too early, wasting capacity in the middle of training when the model could still benefit from large updates.

**Linear warmup** is universal: all schedules begin with a linear ramp from 0 to $\eta_{\max}$ over the first 1,000-2,000 steps. Without warmup, the randomly initialized model receives large gradients that cause immediate divergence.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Learning Rate Schedules Compared</div>
<div style="display:flex; gap:24px; flex-wrap:wrap; justify-content:center;">
<div style="text-align:center; flex:1; min-width:200px;">
<div style="color:#e94560; font-weight:bold; font-size:12px; margin-bottom:8px;">Cosine (Llama 3)</div>
<div style="height:80px; width:100%; background:#0f3460; border-radius:8px; position:relative; overflow:hidden;">
<svg viewBox="0 0 200 80" style="width:100%; height:100%;">
<path d="M10,70 L20,70 Q30,70 40,68 Q80,50 100,40 Q140,20 160,15 Q180,12 190,12" stroke="#e94560" fill="none" stroke-width="2"/>
<text x="10" y="15" fill="#888" font-size="8">eta_max</text>
<text x="160" y="75" fill="#888" font-size="8">eta_min</text>
</svg>
</div>
<div style="color:#888; font-size:10px; margin-top:4px;">Smooth decay from peak to minimum</div>
</div>
<div style="text-align:center; flex:1; min-width:200px;">
<div style="color:#4ecdc4; font-weight:bold; font-size:12px; margin-bottom:8px;">WSD</div>
<div style="height:80px; width:100%; background:#0f3460; border-radius:8px; position:relative; overflow:hidden;">
<svg viewBox="0 0 200 80" style="width:100%; height:100%;">
<path d="M10,70 L20,15 L140,15 Q160,15 170,40 Q180,60 190,70" stroke="#4ecdc4" fill="none" stroke-width="2"/>
<text x="40" y="12" fill="#888" font-size="8">stable phase</text>
<text x="155" y="75" fill="#888" font-size="8">decay</text>
</svg>
</div>
<div style="color:#888; font-size:10px; margin-top:4px;">Constant LR, sharp late decay</div>
</div>
<div style="text-align:center; flex:1; min-width:200px;">
<div style="color:#ffd93d; font-weight:bold; font-size:12px; margin-bottom:8px;">Linear Decay</div>
<div style="height:80px; width:100%; background:#0f3460; border-radius:8px; position:relative; overflow:hidden;">
<svg viewBox="0 0 200 80" style="width:100%; height:100%;">
<path d="M10,70 L20,15 L190,70" stroke="#ffd93d" fill="none" stroke-width="2"/>
</svg>
</div>
<div style="color:#888; font-size:10px; margin-top:4px;">Simple linear ramp down</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:12px; text-align:center;">
All schedules begin with a linear warmup phase (leftmost ramp). The choice between cosine and WSD depends on whether you know the total step count in advance.
</div>
</div>

**Gradient clipping** caps the global gradient norm to prevent any single batch from causing catastrophically large parameter updates:

$$\hat{g} = g \cdot \min\!\left(1, \frac{c}{\|g\|_2}\right)$$

where $c$ is the clipping threshold (typically 1.0). Llama 3 uses gradient clipping with $c = 1.0$; Phi-4 and OLMo 2 do the same. This is universally applied and essentially free — it only activates when gradients are anomalously large.

For an interactive visualization of how these schedules behave under different parameterizations, see [Learning Rate Schedule Explorer](figures/lr-schedules.html).

---

## 3. Curriculum Learning: Does Data Order Matter?

Curriculum learning — presenting training data in a deliberate order rather than random shuffling — is one of pre-training's most debated topics. The intuition is compelling: humans learn better when concepts are introduced in a structured sequence (basics before advanced topics). Do LLMs benefit from the same principle?

### The Evidence For

OLMo 2 ([[olmo-2|report]]) provides the strongest positive evidence through its **two-stage pre-training** approach (covered in detail in Section 4). The key result: saving high-quality data for the final stage of training produces measurably better downstream performance than mixing it uniformly throughout training. This is a form of curriculum — ordering data by quality, with high-quality data concentrated at the end.

Phi-4 ([[phi-4|report]]) also implicitly uses curriculum by placing synthetic data (which is higher information density per token) at specific points in the training schedule rather than distributing it uniformly.

### The Evidence Against

The strongest counterargument comes from the scaling-data-constrained literature ([[scaling-data-constrained|paper]]): when you control for *total tokens seen* and *data quality*, random shuffling performs comparably to most curriculum strategies. The gains from curriculum learning are often confounded with gains from simply *having better data in the mix at all*.

Furthermore, implementing curriculum at scale introduces engineering complexity: you need to maintain multiple data pools, manage transitions between phases, and handle the risk that switching data distributions mid-training causes instability.

### The Synthesis

The current consensus, supported by both OLMo 2 and Llama 3, is nuanced:

1. **Random ordering within a quality tier works fine.** Within a given data source (e.g., web text, code, math), shuffling order does not meaningfully affect final quality.
2. **Ordering across quality tiers matters.** Starting with broad, lower-quality data and finishing with curated, high-quality data (the "coarse-to-fine" curriculum) consistently outperforms uniform mixing.
3. **The mechanism is not "learning basics first."** It is more likely that the final-stage data has an outsized effect on the model's learned representations because of the low learning rate during annealing — the model makes small, precise updates that are less likely to be overwritten.

This last point is critical. The benefit of two-stage training may have less to do with curriculum *per se* and more to do with the interaction between data quality and learning rate schedule: high-quality data + low learning rate = precise, durable improvements.

---

## 4. Two-Stage Pre-training: OLMo 2's Approach

OLMo 2 ([[olmo-2|report]]) formalized what had been ad-hoc practice at other labs into a deliberate two-stage pre-training methodology. The approach is simple in concept but powerful in effect.

### Stage 1: Broad Web Data

The first stage trains on **OLMo-Mix-1124**, a broad web corpus of approximately 3.9 trillion tokens (for the 7B model). This data is:
- Filtered but not heavily curated
- Diverse in topic, style, and quality
- Trained at high learning rate with cosine schedule

The goal of Stage 1 is to build the model's general language understanding: grammar, world knowledge, common reasoning patterns. The data quality bar is "not garbage" rather than "excellent."

### Stage 2: Curated Annealing

Stage 2 introduces **Dolmino-Mix-1124**, a carefully curated high-quality dataset of 843 billion tokens. The composition is deliberately skewed toward knowledge-dense domains:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">OLMo 2 Two-Stage Training Pipeline</div>
<div style="display:flex; gap:16px; align-items:stretch; flex-wrap:wrap; justify-content:center;">
<div style="flex:2; min-width:250px; background:#0f3460; border-radius:8px; padding:16px; border-left:4px solid #e94560;">
<div style="color:#e94560; font-weight:bold; font-size:13px; margin-bottom:8px;">Stage 1: OLMo-Mix-1124</div>
<div style="color:#e0e0e0; font-size:12px; line-height:1.6;">
~3.9T tokens (7B model)<br>
Broad web data, filtered<br>
High learning rate<br>
Goal: general language capacity
</div>
</div>
<div style="display:flex; align-items:center; color:#ffd93d; font-size:24px; font-weight:bold;">&#8594;</div>
<div style="flex:1; min-width:200px; background:#0f3460; border-radius:8px; padding:16px; border-left:4px solid #4ecdc4;">
<div style="color:#4ecdc4; font-weight:bold; font-size:13px; margin-bottom:8px;">Stage 2: Dolmino-Mix-1124</div>
<div style="color:#e0e0e0; font-size:12px; line-height:1.6;">
843B tokens<br>
50% high-quality filtered docs<br>
+ academic, math, educational<br>
+ Q&A, instruction (synthetic + human)<br>
Low LR (annealing phase)
</div>
</div>
<div style="display:flex; align-items:center; color:#ffd93d; font-size:24px; font-weight:bold;">&#8594;</div>
<div style="flex:1; min-width:150px; background:#e94560; border-radius:8px; padding:16px; display:flex; flex-direction:column; justify-content:center; align-items:center;">
<div style="color:#fff; font-weight:bold; font-size:13px; margin-bottom:4px;">Model Souping</div>
<div style="color:#fff; font-size:11px; text-align:center;">
Merge multiple<br>annealing variants<br>(50B + 300B mixes)
</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center;">
Stage 2 runs during the cosine annealing phase when the learning rate is low, ensuring high-quality data imprints durably.
</div>
</div>

### Model Souping: Ensembling Without Ensemble Cost

A distinctive OLMo 2 innovation is **model souping** — training multiple Stage 2 variants with different annealing data mixes (50B token and 300B token variants), then merging the best checkpoints via weight averaging:

$$\theta_{\text{final}} = \frac{1}{K} \sum_{k=1}^{K} \theta_k^{(\text{best})}$$

This is remarkably cheap: weight averaging is a single linear operation on the parameter tensors. But it produces a model that is more robust than any single checkpoint because it averages away the idiosyncratic overfitting of each variant. The technique works because the annealing variants share the same Stage 1 checkpoint — they occupy nearby points in weight space, so averaging produces a valid interpolation rather than nonsense.

### Why Two-Stage Works

The critical question: why not just mix the high-quality data uniformly throughout training? The answer involves an interaction between learning rate and data influence:

1. **Early training is high-variance.** The model's parameters change dramatically with each step. Any knowledge imprinted early can be overwritten thousands of times before training ends.
2. **Late training is low-variance.** During annealing, the learning rate is small, so each step makes precise adjustments. Knowledge imprinted here is more likely to persist in the final model.
3. **High-quality data has more to teach per token.** A well-written textbook passage or a carefully constructed math problem conveys more learnable structure than a random web page. Wasting this data during the high-variance early phase means the model may not retain its lessons.

This creates a natural two-stage design: use cheap, abundant web data to build general capacity (Stage 1), then use expensive, curated data to refine specific capabilities (Stage 2).

OLMo 2 7B outperforms Llama-3.1-8B on downstream benchmarks despite using fewer total training FLOPs — evidence that the two-stage approach is more compute-efficient than uniform mixing.

---

## 5. Synthetic Data for Pre-training: The Phi Approach

Phi-4 ([[phi-4|report]]) represents the most aggressive use of synthetic data in pre-training: **40% of all training tokens are synthetically generated**, totaling approximately 290 billion unique synthetic tokens produced via 50 distinct generation approaches.

### Why Synthetic Data

The motivation is both philosophical and practical:

1. **Data scarcity.** The scaling-data-constrained paper ([[scaling-data-constrained|paper]]) showed that high-quality unique text on the internet is finite. At current scaling trends, frontier models will exhaust available web data within a few years. Synthetic data is one way to break this ceiling.
2. **Information density.** A single carefully constructed synthetic math problem can contain more learnable structure than hundreds of random web paragraphs. Phi-4 replays its synthetic data for **13.8 epochs** — a rate that would cause severe degradation with organic web data, but works because synthetic data is sufficiently information-dense.
3. **Targeted capability injection.** Synthetic data lets you *choose* what capabilities to strengthen. Want better STEM reasoning? Generate synthetic STEM problems. Want better code understanding? Generate synthetic code explanations. You cannot do this with organic web crawls.

### Phi-4's Generation Pipeline

Phi-4 uses 50 distinct synthetic data generation approaches, including:

- **Multi-agent prompting:** Multiple LLM agents collaborate to generate, critique, and refine content
- **Self-revision workflows:** An LLM generates a draft, then iteratively improves it
- **Instruction reversal:** Given an answer, generate the question that would produce it — creating high-quality Q&A pairs
- **Seed curation from web/code:** Use high-quality web snippets as seeds for synthetic expansion

The teacher model for synthetic generation is GPT-4. This creates a remarkable dynamic: **the student (Phi-4, 14B) surpasses its teacher (GPT-4) on STEM benchmarks** (GPQA: 56.1% vs 50.6%, MATH: 80.4% vs 74.6%). The synthetic data distills the teacher's knowledge but the training process learns to generalize beyond it.

### When Synthetic Data Helps vs. Hurts

Synthetic data is not universally beneficial. The evidence suggests clear conditions:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Synthetic Data: When It Helps vs. When It Hurts</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #4ecdc4;">
<th style="text-align:left; padding:8px; color:#4ecdc4;">Helps</th>
<th style="text-align:left; padding:8px; color:#e94560;">Hurts</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">STEM reasoning, math, logic</td>
<td style="padding:8px;">Open-ended creative writing</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Structured code generation</td>
<td style="padding:8px;">Factual world knowledge (hallucination risk)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">Instruction following</td>
<td style="padding:8px;">Low-resource language fluency</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">When teacher model is stronger than student</td>
<td style="padding:8px;">When teacher and student are similar quality</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px;">When diversity of generation methods is high</td>
<td style="padding:8px;">When all synthetic data comes from one pipeline</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
The central risk of synthetic data is mode collapse: if all synthetic examples share stylistic or reasoning patterns from the teacher, the student model loses the diversity of organic data.
</div>
</div>

The **hallucination amplification** risk is real: synthetic data generated by an LLM may confidently state incorrect facts. If the student model trains on these, it inherits the teacher's mistakes and may amplify them. Phi-4 mitigates this through diversity (50 generation approaches, not one) and by using web data for factual grounding alongside synthetic data for reasoning.

For a detailed analysis of the synthetic data tradeoffs and the data repetition economics, see [[ch-11-synthetic-data|excerpt]].

---

## 6. The Economics of Data Repetition

The scaling-data-constrained paper ([[scaling-data-constrained|paper]]) — based on 400 training runs — establishes the empirical laws of data repetition. The central finding modifies the Chinchilla scaling law ([[chinchilla|paper]]) for the real world where data is finite.

### The Core Result

$$\text{Up to } \sim\!4 \text{ epochs: negligible loss degradation vs. unique data}$$
$$\text{Beyond } \sim\!4 \text{ epochs: value of additional compute decays toward zero}$$

This means that if you have $D$ unique tokens and a compute budget that allows training on $4D$ tokens, you can train for 4 epochs with essentially no penalty. But if your compute budget allows $40D$ tokens, the extra 10x repetition provides rapidly diminishing returns — the model begins memorizing rather than generalizing.

The paper proposes a modified scaling law that accounts for diminishing returns from repeated data:

$$L(N, D, R) = L(N, D_{\text{unique}}) + f(R)$$

where $R = \text{total tokens} / D_{\text{unique}}$ is the repetition factor and $f(R)$ captures the decay in value. Below $R \approx 4$, $f(R) \approx 0$. Above $R \approx 15$, additional compute provides almost no loss improvement regardless of model size.

### Practical Implications

This result has directly shaped training strategy at frontier labs:

1. **Chinchilla ratios assumed unlimited data.** The equal-scaling law ($N \propto D$) breaks down when you cannot generate enough unique tokens. In practice, large models must either tolerate some data repetition or find new data sources.
2. **Code data as mitigation.** The paper shows that augmenting training data with code — even for a model not specifically targeting code tasks — partially mitigates data scarcity. Code is structurally rich and contains high information density per token.
3. **Relaxing quality filters.** Counter-intuitively, when data is scarce, keeping slightly lower-quality data in the mix can be better than aggressive filtering followed by heavy repetition. The model benefits more from seeing diverse (if slightly noisy) data than from memorizing clean data.
4. **Synthetic data as escape valve.** Phi-4's 40% synthetic data strategy is a direct response to this data wall. If organic unique data is capped, synthetic generation provides a way to produce effectively unlimited unique tokens — with the caveats discussed in Section 5.

See the interactive [Data Repetition Explorer](figures/data-repetition.html) for a visualization of how loss degrades with repetition at different compute budgets.

---

## 7. Catastrophic Forgetting During Continued Pre-training

Continued pre-training — taking an already-trained model and training it further on new data — is increasingly common. Use cases include domain adaptation (training a general model on medical text), language adaptation (training an English model on Japanese text), and temporal updates (training on newer data). But continued pre-training carries a fundamental risk: **catastrophic forgetting**.

### The Mechanism

When a model trained on distribution $P_1$ is subsequently trained on distribution $P_2$, the gradient updates optimized for $P_2$ can overwrite the representations learned for $P_1$. The severity depends on the distributional distance between $P_1$ and $P_2$: if the new data is similar to the original, forgetting is mild; if it is very different (e.g., pivoting from English web text to Chinese medical text), forgetting can be catastrophic.

Formally, the model's loss on $P_1$ increases as it trains on $P_2$:

$$\mathcal{L}_{P_1}(\theta_{t+\Delta t}) > \mathcal{L}_{P_1}(\theta_t) \quad \text{when training on } P_2$$

The rate of increase depends on the learning rate, the number of continued pre-training steps, and the overlap between $P_1$ and $P_2$.

### Mitigation Strategies

**Replay buffers.** The most effective mitigation: mix a fraction of the original training data into the continued pre-training data. If 10-20% of each batch comes from $P_1$, the model maintains its original capabilities while learning from $P_2$. This is conceptually similar to experience replay in reinforcement learning.

**Data mixing strategies.** Rather than a fixed replay ratio, some approaches dynamically adjust the $P_1/P_2$ ratio based on monitored metrics:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Continued Pre-training: Replay Buffer Strategies</div>
<div style="display:flex; gap:16px; flex-wrap:wrap; justify-content:center;">
<div style="flex:1; min-width:180px; background:#0f3460; border-radius:8px; padding:16px;">
<div style="color:#e94560; font-weight:bold; font-size:12px; margin-bottom:8px;">No Replay</div>
<div style="color:#e0e0e0; font-size:11px; line-height:1.5;">100% new data<br>Fast adaptation<br>Severe forgetting<br>Risk: model loses general capabilities</div>
</div>
<div style="flex:1; min-width:180px; background:#0f3460; border-radius:8px; padding:16px;">
<div style="color:#ffd93d; font-weight:bold; font-size:12px; margin-bottom:8px;">Fixed Replay (10-20%)</div>
<div style="color:#e0e0e0; font-size:11px; line-height:1.5;">80-90% new + 10-20% original<br>Good balance<br>Mild forgetting<br>Industry standard approach</div>
</div>
<div style="flex:1; min-width:180px; background:#0f3460; border-radius:8px; padding:16px;">
<div style="color:#4ecdc4; font-weight:bold; font-size:12px; margin-bottom:8px;">Dynamic Replay</div>
<div style="color:#e0e0e0; font-size:11px; line-height:1.5;">Adjust ratio based on eval metrics<br>Monitor original-domain loss<br>Increase replay if forgetting detected<br>Most complex but most effective</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:12px; text-align:center;">
The optimal replay ratio depends on how different the new domain is from the original training distribution.
</div>
</div>

**Learning rate reduction.** Using a lower learning rate for continued pre-training (typically 10-50x lower than the original peak LR) reduces forgetting because each gradient step makes smaller parameter changes. This is why the two-stage pre-training approach (Section 4) works: Stage 2 naturally occurs during the annealing phase when the learning rate is already low.

**Elastic Weight Consolidation (EWC).** A more principled approach: identify which parameters are most important for the original task (via Fisher information matrix) and penalize changes to those parameters during continued training. EWC adds a regularization term:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{P_2}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

where $F_i$ is the Fisher information for parameter $i$ and $\theta^*$ is the original checkpoint. In practice, computing the full Fisher matrix is prohibitive for LLMs, so approximations are used.

**The connection to two-stage pre-training.** OLMo 2's approach cleverly avoids catastrophic forgetting by design: Stage 2 data (Dolmino-Mix) is not a different domain — it is a *higher-quality subset* of the same distribution, enriched with educational, mathematical, and instruction-following content. The distributional shift is small enough that forgetting is minimal, while the quality improvement is large enough to meaningfully improve benchmarks.

For a detailed treatment of continued pre-training techniques and their relationship to post-training ([[ch-12]]), see [[ch-11-continued-pretraining|excerpt]].

---

## Core Insights from the Literature

### Insight 1: Data quality dominates data quantity for pre-training efficiency
**Report:** Phi-4 Technical Report ([[phi-4|report]])

Phi-4 (14B parameters) matches or exceeds Llama-3.1-70B on reasoning benchmarks despite being 5x smaller, because 40% of its training data is high-quality synthetic data generated via 50 distinct approaches. The model replays synthetic data for 13.8 epochs — a repetition rate that would devastate performance with organic web data. This proves that information density per token matters more than raw token count when compute is fixed. The student surpassing the teacher (Phi-4 > GPT-4 on GPQA/MATH) demonstrates that synthetic data can enable genuine generalization, not just distillation. **Guideline:** When designing a pre-training data mix, invest heavily in data quality engineering (filtering, synthetic generation, curation). A dollar spent on data quality delivers more benchmark improvement than a dollar spent on additional raw tokens.

### Insight 2: Save your best data for last — learning rate and data quality interact multiplicatively
**Report:** OLMo 2 Technical Report ([[olmo-2|report]])

OLMo 2's two-stage approach — broad web data (Stage 1) followed by curated data during annealing (Stage 2) — outperforms uniform mixing of the same data. The mechanism is an interaction effect: high-quality data presented during the low-learning-rate annealing phase makes small, precise updates that persist in the final model, whereas the same data presented during high-LR early training would be overwritten by subsequent updates. Model souping (averaging multiple Stage 2 variants) further improves robustness at negligible cost. **Guideline:** Structure pre-training as at least two phases. Use the annealing phase to introduce your highest-quality curated and domain-specific data.

### Insight 3: Data repetition has a hard ceiling — unique tokens are the real currency
**Paper:** Muennighoff et al., "Scaling Data-Constrained Language Models" ([[scaling-data-constrained|paper]])

Across 400 training runs, data repetition up to ~4 epochs is essentially free — loss is indistinguishable from unique data. But beyond ~4 epochs, the marginal value of additional compute decays rapidly toward zero regardless of model size. This modifies Chinchilla's equal-scaling law for the real world where data is finite: you cannot simply train longer to compensate for a data shortage. This finding is the economic foundation for synthetic data investments, code data augmentation, and the relaxation of quality filters when data is scarce. **Guideline:** Plan your pre-training token budget around the amount of unique data you can source. If your compute budget implies more than 4 epochs over your unique data, invest in expanding the data pool (synthetic generation, code augmentation, relaxed filtering) rather than in more training steps.

### Insight 4: Training stability is not a luxury — it is a prerequisite for realizing scaling law predictions
**Report:** OLMo 2 Technical Report ([[olmo-2|report]])

OLMo 2's stability stack (QK-Norm, z-loss regularization, improved initialization, RMSNorm) is not a performance optimization — it is what makes multi-trillion-token training runs *possible* without divergence. A single loss spike can waste days of compute on a 1,280-GPU cluster. The stability interventions add negligible compute overhead but prevent the numerical instabilities (attention logit growth, logit explosion, gradient spikes) that cause catastrophic divergence. **Guideline:** Before scaling up a training run, implement the full stability stack: QK-Norm, z-loss, gradient clipping, proper initialization, and RMSNorm. Diagnose and fix instabilities at small scale before committing large-scale compute.

---

## Key Takeaways

1. **Pre-training is a data engineering problem.** The architecture is largely settled (decoder-only Transformer with GQA, RoPE, SwiGLU, RMSNorm). The remaining differentiator between models of the same size is data quality, data mix, and training methodology.

2. **Data filtering is a multi-stage funnel.** URL filtering, heuristic filters, deduplication (MinHash), and quality classifiers each remove 30-80% of remaining data. The quality classifier threshold is the most sensitive hyperparameter in the entire pipeline.

3. **Training stability requires an explicit engineering stack.** QK-Norm, z-loss, gradient clipping, learning rate warmup, and proper initialization are not optional at scale. A loss spike on a 16,384-GPU run can cost hundreds of thousands of dollars in wasted compute.

4. **Two-stage pre-training outperforms uniform mixing.** Training on broad web data first, then annealing on curated high-quality data, produces better models than mixing all data uniformly. The mechanism is the interaction between data quality and learning rate magnitude.

5. **Synthetic data can substitute for scale when quality is high enough.** Phi-4 (14B) matches 70B models because 40% of its data is high-quality synthetic content. But synthetic data works best for structured reasoning tasks and carries hallucination amplification risk for factual knowledge.

6. **Data repetition beyond ~4 epochs has rapidly diminishing returns.** This hard ceiling on the value of data repetition is the economic driver behind synthetic data generation and aggressive data collection strategies at frontier labs.

7. **Catastrophic forgetting during continued pre-training is managed through replay buffers.** Mixing 10-20% of original training data into continued pre-training batches effectively preserves the model's original capabilities while adapting to new domains.

---

## References

- [[llama-3|Meta AI, "The Llama 3 Herd of Models" (2024) (report)]] — 15T-token data pipeline, scaling-law-driven data mixing
- [[olmo-2|AI2, "OLMo 2: 2 OLMo 2 Furious" (2024-2025) (report)]] — Two-stage pre-training, model souping, training stability
- [[phi-4|Microsoft Research, "Phi-4 Technical Report" (2024) (report)]] — 40% synthetic data, student surpasses teacher
- [[scaling-data-constrained|Muennighoff et al., "Scaling Data-Constrained Language Models" (2023) (paper)]] — Data repetition economics, modified scaling laws
- [[chinchilla|Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) (paper)]] — Chinchilla scaling law (equal-scaling of parameters and data)
- [[ultra-scale-playbook|Tazi et al., "The Ultra-Scale Playbook" (2025) (blog)]] — Distributed training infrastructure, stability engineering
