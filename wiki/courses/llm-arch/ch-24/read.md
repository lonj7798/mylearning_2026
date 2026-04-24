# Chapter 24: Case Study: OLMo 2

<!-- scope: OLMo 2 technical report — fully open models, two-stage pre-training, QK-norm, model souping, published ablations
     deps: [[ch-09]], [[ch-11]]
     see-also: [[ch-18]], [[ch-29]]
-->

## Overview

Most LLM technical reports are marketing documents with architecture tables. They tell you *what* choices were made but rarely *why*, and almost never give you the artifacts to verify the claims yourself. OLMo 2 is the exception.

AI2's OLMo 2 family (7B, 13B, 32B) is the first frontier-competitive model line where every artifact is public: model weights at every stage, the full training data (OLMo-Mix-1124 and Dolmino-Mix-1124), training code, hyperparameters, training logs, and thousands of intermediate checkpoints. The 32B model is the first fully open model to outperform GPT-3.5-Turbo and GPT-4o mini on a multi-skill benchmark suite. But the scientific value of OLMo 2 is not its benchmark scores — it is the *transparency* that lets you trace each architectural decision to its empirical justification.

This chapter treats OLMo 2 as a case study in evidence-based architecture design. We will dissect four decisions — QK-norm for training stability, two-stage pre-training with curriculum learning, removing bias terms from all linear layers, and model souping for checkpoint selection — and for each one, examine the published ablation evidence. The goal is not to memorize OLMo 2's spec sheet but to practice the skill of reading an architecture report critically: separating the choices that matter from the choices that are incidental, and understanding why.

OLMo 2 builds directly on the normalization principles from [[ch-09]] (QK-norm, RMSNorm, initialization) and the pre-training strategies from [[ch-11]] (curriculum learning, data mixing, training stability). It provides concrete, reproducible evidence for concepts those chapters introduced theoretically.

---

## 1. The Architecture Inventory

OLMo 2 is a decoder-only Transformer. There is nothing exotic in the component list — the contribution is in how the components are combined and the evidence behind each choice.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">OLMo 2 Architecture Specifications</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Component</th>
<th style="text-align:right; padding:8px;">7B</th>
<th style="text-align:right; padding:8px;">13B</th>
<th style="text-align:right; padding:8px;">32B</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Hidden Dimension</td>
<td style="text-align:right; padding:8px;">4,096</td>
<td style="text-align:right; padding:8px;">5,120</td>
<td style="text-align:right; padding:8px;">5,120</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Layers</td>
<td style="text-align:right; padding:8px;">32</td>
<td style="text-align:right; padding:8px;">40</td>
<td style="text-align:right; padding:8px;">64</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Attention Heads</td>
<td style="text-align:right; padding:8px;">32</td>
<td style="text-align:right; padding:8px;">40</td>
<td style="text-align:right; padding:8px;">40</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">KV Heads</td>
<td style="text-align:right; padding:8px; color:#ffd93d;">32 (MHA)</td>
<td style="text-align:right; padding:8px; color:#ffd93d;">40 (MHA)</td>
<td style="text-align:right; padding:8px; color:#4ecdc4;">8 (GQA)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Context Length</td>
<td style="text-align:right; padding:8px;">4,096</td>
<td style="text-align:right; padding:8px;">4,096</td>
<td style="text-align:right; padding:8px;">4,096</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Training Tokens</td>
<td style="text-align:right; padding:8px;">4T</td>
<td style="text-align:right; padding:8px;">5T</td>
<td style="text-align:right; padding:8px;">6T</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Normalization</td>
<td style="text-align:right; padding:8px;" colspan="3">RMSNorm (post-norm placement)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Positional Encoding</td>
<td style="text-align:right; padding:8px;" colspan="3">RoPE</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Activation</td>
<td style="text-align:right; padding:8px;" colspan="3">SwiGLU</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Stability</td>
<td style="text-align:right; padding:8px;" colspan="3">QK-norm + Z-loss</td>
</tr>
<tr>
<td style="padding:8px; font-weight:bold;">Bias Terms</td>
<td style="text-align:right; padding:8px;" colspan="3">None (removed from all linear layers)</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
MHA = Multi-Head Attention (every query head has its own KV head). GQA = Grouped-Query Attention (multiple query heads share KV heads). The 32B model switches to GQA to manage KV cache at scale — exactly the reasoning covered in [[ch-07]].
</div>
</div>

Several things to notice immediately:

**Conservative context length.** While competitors push to 128K+ tokens (Llama 4 Scout claims 10M), OLMo 2 stays at 4,096. This is a deliberate resource allocation decision: rather than spending compute on context extension techniques (RoPE scaling, position interpolation from [[ch-06]]), AI2 invested those FLOPs into more training tokens and better data quality. This is the kind of tradeoff that matters for research: a 4K context model with 6T training tokens and published ablations is more scientifically useful than a 128K model whose training recipe is secret.

**MHA at 7B/13B, GQA only at 32B.** Most contemporary models use GQA at all scales (Llama 3 uses GQA-8 even at 8B). AI2 kept full MHA at 7B and 13B, switching to GQA only at 32B where the KV cache cost becomes prohibitive. This reflects a research-first philosophy: MHA has strictly more representational capacity than GQA (as established in [[ch-07]]), and at 7B/13B the KV cache is manageable.

**Dense architecture.** No MoE. This is the most consequential efficiency decision. A 32B dense model uses all 32B parameters for every token. A comparable MoE model like Qwen 3 (30B total, 3B active) processes each token with 10x fewer FLOPs. AI2 chose density for simplicity and reproducibility — MoE adds routing complexity that makes ablations harder to interpret.

---

## 2. Training Stability: QK-Norm and Z-Loss

The single most important architectural lesson from OLMo 2 is how it solved training stability. Large-scale training runs are expensive (1,280 H100 GPUs for the 32B model), and a loss spike at 80% through training can waste millions of dollars of compute. OLMo 1 suffered from exactly these instabilities. OLMo 2 fixed them with two complementary mechanisms.

### The Problem: Attention Logit Growth

As [[ch-09]] explained in detail, the dot product $q_i^T k_j$ in attention can grow unboundedly during training. As the model trains on trillions of tokens, the norms of query and key vectors tend to drift upward. When $\|q\| \cdot \|k\|$ becomes large enough, the softmax saturates — one or two attention entries dominate — and gradients through the softmax vanish. This manifests as sudden loss spikes followed by slow recovery or permanent divergence.

The fundamental issue is that standard attention has no mechanism to bound the scale of its logits. The $1/\sqrt{d_k}$ scaling factor is a constant set at initialization — it does not adapt as training progresses.

### QK-Norm: The Fix

OLMo 2 applies RMSNorm to queries and keys *before* the RoPE rotation and the dot product:

$$\hat{q}_h = \text{RMSNorm}(q_h), \quad \hat{k}_h = \text{RMSNorm}(k_h)$$

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{\hat{Q} \cdot \hat{K}^T}{\sqrt{d_k}}\right) V$$

After normalization, $\|\hat{q}_h\|$ and $\|\hat{k}_h\|$ are bounded (approximately unit scale), so the dot product $\hat{q}_h^T \hat{k}_h$ is bounded by $d_k$, and after the $1/\sqrt{d_k}$ scaling, attention logits stay in $[-\sqrt{d_k}, \sqrt{d_k}]$. For OLMo 2's head dimension of 128, this means logits are bounded in roughly $[-11.3, 11.3]$ — well within softmax's numerically stable regime.

**Why RMSNorm before RoPE?** RoPE is a rotation that preserves vector norms ($\|R_\theta x\| = \|x\|$). So normalizing before or after RoPE is mathematically equivalent for the norm. But applying RMSNorm first means the normalization operates on the raw projection output, which is simpler to reason about and avoids any interaction with the position-dependent rotation.

### Z-Loss: Complementary Output Stabilization

QK-norm stabilizes the *attention* logits. Z-loss stabilizes the *output* logits — the pre-softmax scores over the vocabulary:

$$\mathcal{L}_Z = \alpha \cdot \log^2\!\left(\sum_j e^{z_j}\right)$$

where $z_j$ are the output logits and $\alpha$ is a small coefficient ($\sim 10^{-4}$). This adds a penalty that grows quadratically when the log-sum-exp of output logits becomes large. Unlike QK-norm (which is a structural change), Z-loss is a regularization term — it gently pushes the model away from extreme output distributions without constraining what the model can represent.

The combination matters: QK-norm prevents instability *within* the model (attention entropy collapse), while Z-loss prevents instability *at the output* (logit explosion). Together they cover the two primary failure modes of long training runs.

### What Changed from OLMo 1

OLMo 1 used nonparametric LayerNorm (no learnable $\gamma$, $\beta$) and absolute positional embeddings. It suffered from training instabilities that required restarts. The OLMo 2 changes were:

1. **Nonparametric LayerNorm to RMSNorm** — drops mean-centering (as [[ch-09]] showed, the mean is dispensable) and adds learnable scale parameters
2. **Absolute position embeddings to RoPE** — enables relative position awareness and better length behavior
3. **Added QK-norm** — bounds attention logits
4. **Added Z-loss** — bounds output logits
5. **Improved initialization** — preserves activation and gradient scale across layers, enabling post-norm placement

Each of these changes has published ablation evidence. That is the scientific contribution.

---

## 3. Two-Stage Pre-Training

OLMo 2's training curriculum is the clearest public demonstration of a principle [[ch-11]] introduced: **the order in which a model sees data matters, and saving high-quality data for last works better than mixing it uniformly**.

[See interactive diagram: [figures/two-stage-training.html](figures/two-stage-training.html)]

### Stage 1: Broad Web Data

Stage 1 trains on OLMo-Mix-1124, a large web-crawl dataset (~3.9T tokens for 7B). This is the bulk learning phase — the model acquires general language competence, world knowledge, and basic reasoning patterns from a diverse but noisy data distribution.

- **7B:** ~3.9T tokens, approximately 1 epoch
- **13B:** ~5T tokens, approximately 1.2 epochs
- **32B:** ~6T tokens, approximately 1.5 epochs

The data is standard web data: Common Crawl filtered, deduplicated, and quality-classified. Nothing unusual here — this is the same approach as Llama, Mistral, and other open models.

### Stage 2: Curated Annealing

Stage 2 introduces Dolmino-Mix-1124, a carefully curated dataset of 843B tokens, during the learning rate annealing phase (when the learning rate is being lowered toward zero). The composition is deliberately different from Stage 1:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Dolmino-Mix-1124 Composition (Stage 2)</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #4ecdc4;">
<th style="text-align:left; padding:8px;">Category</th>
<th style="text-align:right; padding:8px;">Approx. Share</th>
<th style="text-align:left; padding:8px;">Purpose</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">High-quality filtered web</td>
<td style="text-align:right; padding:8px;">~50%</td>
<td style="padding:8px;">Continued general language modeling</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">Academic papers</td>
<td style="text-align:right; padding:8px;">~12%</td>
<td style="padding:8px;">Technical reasoning, formal writing</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">Math</td>
<td style="text-align:right; padding:8px;">~10%</td>
<td style="padding:8px;">Mathematical reasoning</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ff6b81; font-weight:bold;">Educational content</td>
<td style="text-align:right; padding:8px;">~10%</td>
<td style="padding:8px;">Structured knowledge, explanations</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#a29bfe; font-weight:bold;">Q&A / instruction data</td>
<td style="text-align:right; padding:8px;">~10%</td>
<td style="padding:8px;">Instruction following, dialogue patterns</td>
</tr>
<tr>
<td style="padding:8px; color:#74b9ff; font-weight:bold;">Synthetic data</td>
<td style="text-align:right; padding:8px;">~8%</td>
<td style="padding:8px;">Generated examples for coverage gaps</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
Total: 843B tokens. Stage 2 uses 50B and 300B token annealing variants, merged via model souping.
</div>
</div>

### Why Two-Stage Works

The intuition is curriculum learning: you teach broad knowledge first (read everything on the internet), then refine with high-quality examples (read textbooks and papers). But why not just mix the high-quality data into Stage 1 from the beginning?

**Data efficiency argument.** High-quality data is scarce (843B tokens vs 3.9T tokens of web data). If you mix it uniformly into Stage 1, the model sees each high-quality example roughly once early in training when its capacity to absorb fine distinctions is low. By saving it for Stage 2, the model encounters high-quality data when it already has strong language modeling foundations and can extract more from each example.

**Learning rate interaction.** Stage 2 coincides with learning rate annealing. The combination of curated data + small learning rate acts like fine-grained polishing: the model makes small, targeted updates rather than large, noisy ones. This is analogous to how humans study — you read broadly first, then focus on the details.

**Empirical evidence.** The OLMo 2 report shows that two-stage training significantly outperforms single-stage training at the same total compute. The 7B model trained with two stages outperforms Llama-3.1-8B despite using fewer total FLOPs. This is the clearest public evidence that training curriculum — not just total data — drives quality.

---

## 4. Model Souping: Ensembling Without Inference Cost

After Stage 2, OLMo 2 does something unusual: it trains *multiple* annealing variants with different data mixes (50B and 300B token variants), then merges the best checkpoints by averaging their weights. This technique, called **model souping** (Wortsman et al., 2022), produces a single model that performs better than any individual variant.

### How It Works

Given $N$ models $\theta_1, \ldots, \theta_N$ trained from the same Stage 1 checkpoint but with different Stage 2 mixes:

$$\theta_{\text{soup}} = \frac{1}{N} \sum_{i=1}^{N} \theta_i$$

This is just element-wise weight averaging. At inference time, $\theta_{\text{soup}}$ has the same architecture and cost as any individual model — the ensembling is free.

### Why Weight Averaging Works

Model souping relies on a property called **linear mode connectivity**: models fine-tuned from the same pre-trained checkpoint tend to lie in the same loss basin. The linear interpolation between any two such models does not cross a loss barrier — performance varies smoothly, and the average often outperforms either endpoint.

This property holds because Stage 2 makes relatively small updates to the Stage 1 weights. The different annealing variants explore slightly different regions of the same basin. Averaging them finds a point closer to the center of the basin, which tends to generalize better than any single point on the periphery.

**When it would fail:** Model souping would not work if the models being averaged were trained from different random initializations (they would be in different loss basins) or if Stage 2 made updates large enough to escape the original basin. The key condition is that the models share a common pre-training trajectory.

### The Selection Process

AI2 did not blindly average all variants. They evaluated each annealing variant on a benchmark suite and selected the best-performing ones for the soup. This selective souping is important — averaging a strong variant with a weak one can degrade the result. The selection criterion is: include a variant in the soup only if adding it improves the average's benchmark performance.

---

## 5. No Bias Terms: The Simplification That Works

OLMo 2 removes bias terms from all linear layers. Every projection — $W_Q$, $W_K$, $W_V$, $W_O$, the FFN up/down/gate projections, the output head — is a pure matrix multiplication: $y = Wx$ with no additive $b$.

This might seem like a minor detail, but it reflects a broader pattern in modern LLM design: **many classical neural network components exist for historical reasons and can be removed without loss**.

### Why Removing Bias Works

The bias term in $y = Wx + b$ adds a learnable offset to each output dimension. In a Transformer with RMSNorm (or LayerNorm) before or after each sublayer, this offset is largely redundant because:

1. **RMSNorm rescales activations.** Any constant offset added by a bias gets absorbed into the normalization statistics. The bias would need to fight against normalization at every layer boundary.
2. **The residual stream provides a mean shift.** Information about the "average" activation is carried through the residual connection. The model does not need per-layer biases to represent offsets.
3. **Parameter efficiency.** Removing bias from all linear layers saves a trivial number of parameters (the hidden dimension $d$ per projection, vs $d \times d$ for the weight matrix), but the simplification makes mixed-precision training and quantization cleaner — there are fewer distinct tensor shapes to manage.

This design choice was already established by Llama ([[llama-2|report]]) and adopted by most subsequent models. OLMo 2 confirms it: removing bias terms causes no measurable quality degradation and simplifies the implementation.

---

## 6. Published Ablations: What Actually Matters

The greatest scientific value of the OLMo 2 report is its ablation studies. Most model reports present only the final configuration. OLMo 2 publishes the experiments that *led* to that configuration — showing which changes helped, which did not, and by how much.

[See interactive visualization: [figures/ablation-results.html](figures/ablation-results.html)]

### What the Ablations Show

The OLMo 2 team systematically evaluated their architectural modifications by running controlled experiments at smaller scale. The key findings:

**Changes that clearly helped:**

| Modification | Effect | Evidence Quality |
|---|---|---|
| QK-norm | Eliminated loss spikes during training | Direct: loss curves with/without |
| Z-loss | Prevented output logit explosion | Direct: logit magnitude tracking |
| RMSNorm (from nonparametric LN) | Slightly better convergence, simpler | Controlled ablation |
| RoPE (from absolute pos emb) | Better downstream performance | Controlled ablation |
| Two-stage curriculum | Significant downstream improvement | Stage 1 only vs Stage 1+2 |
| Model souping | Consistent improvement over best single ckpt | Multiple variant comparison |

**Changes where evidence is more nuanced:**

| Modification | Finding |
|---|---|
| Post-norm vs pre-norm | Post-norm worked *only* with QK-norm + Z-loss + scaled init. In isolation, pre-norm was more stable. |
| GQA vs MHA | MHA used at 7B/13B (no quality cost); GQA adopted at 32B for KV cache manageability. Not a quality-driven decision. |
| Bias removal | No measurable difference. Removed for simplicity. |

### How to Read These Ablations

The ablation methodology is what matters for [[ch-29]]. Notice the structure:

1. **Controlled comparison.** Each ablation changes exactly one variable from the baseline. The baseline is clearly defined (OLMo 1 architecture for the stability changes, Stage 1-only for the curriculum changes).
2. **Multiple evaluation axes.** The team evaluates on the OLMES benchmark suite (20 benchmarks spanning knowledge recall, commonsense, general reasoning, and mathematical reasoning), not a single metric.
3. **Scale-dependent results.** Some modifications matter more at 32B than at 7B. The ablations are run at multiple scales when feasible.
4. **Negative results reported.** When a modification does not help (bias removal has no measurable effect), this is stated explicitly rather than omitted.

This is the standard a well-designed ablation study should meet. When reading other model reports, ask: are these standards met? If an architecture paper presents only the final configuration without ablations, the design choices are unverified assertions, not evidence.

---

## 7. The Post-Training Pipeline

OLMo 2's post-training uses the Tulu 3 recipe:

1. **SFT (Supervised Fine-Tuning):** Instruction-following on curated datasets
2. **DPO (Direct Preference Optimization):** Preference alignment without a reward model
3. **PPO (Proximal Policy Optimization):** RL-based refinement with a preference mix

This is a standard three-stage post-training pipeline (covered in [[ch-12]]). What is notable is the *openness*: the post-training data, code, and intermediate checkpoints are all public. Most models treat post-training as proprietary — AI2's release means researchers can study how post-training transforms the base model's behavior at every step.

---

## 8. Infrastructure and Training Efficiency

The 32B model was trained on 160 nodes of 8 NVIDIA H100 GPUs each (1,280 GPUs total) on Google Cloud's AI Hypercomputer with GPUDirect-TCPXO interconnect.

**Key metrics:**
- Throughput: >1,800 tokens/sec/GPU
- Model FLOPs Utilization (MFU): ~38%
- Total training FLOPs: not publicly specified for 32B, but 1.8e23 for 7B and 4.6e23 for 13B

An MFU of 38% is in the normal range for dense Transformer training on H100 clusters (typical values are 30-45% depending on model size, batch size, and parallelism strategy). For comparison, DeepSeek-V3 reported higher MFU, but that model uses FP8 mixed-precision training and custom optimization — techniques OLMo 2 deliberately avoids for reproducibility.

The modest MFU reflects AI2's practical constraint: they used a cloud cluster with standard interconnect rather than a purpose-built supercomputer. The choice of cloud infrastructure over dedicated hardware is itself a statement about accessibility — it demonstrates that frontier-competitive models can be trained on infrastructure available to well-funded academic labs, not just Big Tech.

---

## 9. Dense vs MoE: A Deliberate Non-Choice

OLMo 2 is dense. This is the single most consequential efficiency decision in the architecture, and it was deliberate.

At 32B parameters, a dense model uses all 32B parameters for every token. An MoE model at the same total parameter count might activate only 3-5B parameters per token, achieving 7-10x the throughput at comparable quality. By any efficiency metric, MoE dominates at this scale.

AI2 chose density for three reasons:

1. **Reproducibility.** Dense models are simpler to analyze. There is no routing stochasticity, no load balancing auxiliary loss, no expert specialization to characterize. When you ablate a component in a dense model, the effect is deterministic.

2. **Baseline value.** The research community needs strong dense baselines to evaluate MoE improvements against. If OLMo 2 were MoE, the gains from its training recipe (two-stage, souping) would be confounded with MoE's sparsity gains.

3. **Serving simplicity.** Dense models work with standard inference stacks without expert parallelism. This lowers the barrier for other researchers to use OLMo 2 as a starting point.

The cost is real: OLMo 2 32B requires significantly more inference compute per token than a comparable MoE model. But for a research artifact, the interpretability of a dense model outweighs the serving efficiency of a sparse one.

---

## 10. What Full Openness Enables

OLMo 2's release includes artifacts that no other competitive model provides:

**Thousands of intermediate checkpoints.** These enable studying *training dynamics* — how capabilities emerge, how loss curves evolve, how internal representations change over time. You cannot do this with a single final checkpoint.

**Complete training data.** Both OLMo-Mix-1124 and Dolmino-Mix-1124 are public. This means you can study the relationship between specific training data and model behavior — a question that is unanswerable for closed models.

**Training logs.** Loss curves, gradient norms, learning rate schedules, and hardware utilization metrics. These let you diagnose training pathologies and understand the stability properties of the architecture.

**Training code and recipes.** Not just the model code, but the complete training configuration including hyperparameters, data loading, and distributed training setup. This is the difference between "we used AdamW" (which every report says) and "here is the exact config file we used" (which almost none provide).

For [[ch-29]] (Designing Architecture Experiments), this transparency is invaluable. OLMo 2 is the only frontier-scale model where you can practice the full research loop: read the report, form a hypothesis about an architectural choice, download the code and data, and run a controlled experiment to test your hypothesis.

---

## 11. Comparative Positioning

How does OLMo 2 compare to the models studied in other case study chapters?

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">OLMo 2 vs Contemporary Architectures</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:12px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Feature</th>
<th style="text-align:center; padding:8px; color:#4ecdc4;">OLMo 2 32B</th>
<th style="text-align:center; padding:8px; color:#ffd93d;">Llama 3.1 70B</th>
<th style="text-align:center; padding:8px; color:#e94560;">DeepSeek-V3</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Architecture</td>
<td style="text-align:center; padding:8px;">Dense</td>
<td style="text-align:center; padding:8px;">Dense</td>
<td style="text-align:center; padding:8px;">MoE (671B total)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Attention</td>
<td style="text-align:center; padding:8px;">GQA</td>
<td style="text-align:center; padding:8px;">GQA</td>
<td style="text-align:center; padding:8px;">MLA</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Training Stability</td>
<td style="text-align:center; padding:8px;">QK-norm + Z-loss</td>
<td style="text-align:center; padding:8px;">Not disclosed</td>
<td style="text-align:center; padding:8px;">FP8 + custom</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Training Curriculum</td>
<td style="text-align:center; padding:8px;">Two-stage + soup</td>
<td style="text-align:center; padding:8px;">Multi-stage (details limited)</td>
<td style="text-align:center; padding:8px;">Not disclosed</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Context Length</td>
<td style="text-align:center; padding:8px;">4K</td>
<td style="text-align:center; padding:8px;">128K</td>
<td style="text-align:center; padding:8px;">128K</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Weights Open</td>
<td style="text-align:center; padding:8px; color:#4ecdc4;">Yes</td>
<td style="text-align:center; padding:8px; color:#4ecdc4;">Yes</td>
<td style="text-align:center; padding:8px; color:#4ecdc4;">Yes</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Data Open</td>
<td style="text-align:center; padding:8px; color:#4ecdc4;">Yes</td>
<td style="text-align:center; padding:8px; color:#e94560;">No</td>
<td style="text-align:center; padding:8px; color:#e94560;">No</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; font-weight:bold;">Training Code Open</td>
<td style="text-align:center; padding:8px; color:#4ecdc4;">Yes</td>
<td style="text-align:center; padding:8px; color:#e94560;">No</td>
<td style="text-align:center; padding:8px; color:#e94560;">No</td>
</tr>
<tr>
<td style="padding:8px; font-weight:bold;">Intermediate Checkpoints</td>
<td style="text-align:center; padding:8px; color:#4ecdc4;">Thousands</td>
<td style="text-align:center; padding:8px; color:#e94560;">No</td>
<td style="text-align:center; padding:8px; color:#e94560;">No</td>
</tr>
</tbody>
</table>
</div>

The comparison reveals OLMo 2's niche: it is not the most capable model, nor the most efficient, nor the longest-context. It is the most *scientifically useful* — the only model where you can verify every claim, reproduce every result, and study every aspect of training.

---

## Core Insights from the Report

### Insight 1: Training curriculum matters as much as architecture

**Source:** OLMo 2 Technical Report ([[olmo-2|report]])

The two-stage training approach — broad web data followed by curated annealing — produced measurable improvements over single-stage training at the same total compute. OLMo 2 7B outperforms Llama-3.1-8B despite using fewer FLOPs, and the report attributes a significant portion of this advantage to the curriculum design rather than architectural differences (the models use similar components). This confirms the principle from [[ch-11]]: how you sequence training data can matter more than incremental architectural tweaks. **Guideline:** When designing training runs, treat the data curriculum as a first-class hyperparameter. Reserve 10-20% of training for a curated annealing stage with high-quality data.

### Insight 2: Model souping extracts value from training variance

**Source:** OLMo 2 Technical Report ([[olmo-2|report]]), Wortsman et al. (2022)

Training multiple Stage 2 variants and weight-averaging the best ones consistently outperforms selecting the single best checkpoint. This works because fine-tuned variants from the same pre-training base share a loss basin (linear mode connectivity), and averaging moves toward the basin's center. The additional compute cost is modest (training a few annealing variants is cheap relative to Stage 1), and the inference cost is zero — the souped model has identical architecture and speed. **Guideline:** When performing learning-rate annealing or fine-tuning, train 3-5 variants with different data mixes or hyperparameters and average the weights of the best performers. This is a near-free quality improvement.

### Insight 3: QK-norm and Z-loss together solve the training stability problem

**Source:** OLMo 2 Technical Report ([[olmo-2|report]]), [[ch-09]]

QK-norm bounds attention logits by normalizing queries and keys before the dot product. Z-loss bounds output logits via a regularization penalty. Together, they cover the two primary failure modes of large-scale training (attention entropy collapse and logit explosion) and enabled OLMo 2 to train stably on trillions of tokens without the loss spikes that plagued OLMo 1. The combination also enabled post-norm placement, which may offer representational advantages over the pre-norm default. **Guideline:** For any training run exceeding 1T tokens, use QK-norm + Z-loss as default stability mechanisms. The compute overhead is negligible relative to the cost of a loss spike.

### Insight 4: Full openness is itself a research contribution

**Source:** OLMo 2 Technical Report ([[olmo-2|report]])

The release of intermediate checkpoints, training data, training logs, and code makes OLMo 2 uniquely valuable for studying training dynamics, data-model interactions, and emergent capabilities. No other frontier-competitive model provides these artifacts. This level of transparency enables research that is impossible with closed models: studying when capabilities emerge during training, how specific training data affects model behavior, and whether architectural modifications matter at scale. **Guideline:** When evaluating model reports for research purposes, distinguish between "open weights" (inference possible) and "fully open" (research possible). Most "open" models are only the former.

---

## Key Takeaways

1. **OLMo 2 is a research instrument, not a product.** Its value is not in benchmark rankings but in the published ablations, open artifacts, and reproducibility that let you practice architecture analysis on a frontier-scale model.

2. **QK-norm + Z-loss is the modern solution to training instability.** Normalizing queries and keys bounds attention logits; penalizing large output logits prevents explosion at the vocabulary projection. Together they enable stable multi-trillion-token training.

3. **Two-stage pre-training outperforms single-stage at the same compute.** Save high-quality data for the annealing phase. The model absorbs more from curated data when it already has strong language modeling foundations.

4. **Model souping is a near-free quality improvement.** Train multiple annealing variants, evaluate, weight-average the best. Zero inference cost, consistent gains, requires only linear mode connectivity (shared pre-training base).

5. **Removing bias terms from all linear layers causes no measurable quality loss.** RMSNorm and the residual stream make per-layer biases redundant. This simplification is now standard.

6. **Dense architecture is a deliberate choice for research, not a limitation.** MoE would be more efficient, but density makes ablations interpretable and results reproducible.

7. **"Open weights" is not the same as "fully open."** OLMo 2 releases weights, data, code, training logs, and intermediate checkpoints. Most "open" models release only weights. The distinction matters for research.

---

## References

- [[olmo-2|AI2, "OLMo 2: 2 OLMo 2 Furious — Technical Report" (2025) (report)]] — primary source
- [[ch-09|Chapter 9: Normalization and Residual Connections]] — QK-norm theory, RMSNorm, pre-norm vs post-norm
- [[ch-11|Chapter 11: Pre-training]] — curriculum learning, training stability, data mixing
- [[ch-07|Chapter 7: Attention Variants]] — MHA vs GQA tradeoffs, KV cache considerations
- [[llama-2|Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023) (report)]] — bias-free linear layers precedent
- Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time" (2022) — model souping theory
