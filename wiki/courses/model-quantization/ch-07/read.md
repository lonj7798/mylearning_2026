<!-- chapter: ch-07
     phase: llm-ptq-2022
     title: LLM.int8() and the Outlier Discovery
     sources: [[llm-int8]], [[dettmers-llm-int8-blog]], [[bitsandbytes-int8]]
     forward: [[gptq]] (ch-08), [[smoothquant]] (ch-09), [[awq]] (ch-09), [[spqr]] (ch-11), [[quip]] (ch-13), [[quarot]] (ch-14)
-->

# Chapter 7 — LLM.int8() and the Outlier Discovery

> **Core insight.** Past ~6.7B parameters, transformers undergo a phase transition: a tiny handful of feature dimensions (≈6 in a 6.7B model out of ~14k hidden dims) start producing coordinated, large-magnitude activations across all layers. A single absmax INT8 scale per row gets dominated by these outliers, leaving the other 99.9% of values represented by 1–2 quantization levels. The fix is structural: split the GEMM into INT8 for normal columns (99.9% of FLOPS) and FP16 for the ~6 outlier columns, then add the two paths.
>
> **Guideline.** For any LLM ≥ 7B that you want to deploy at INT8 without retraining, use `bitsandbytes.nn.Linear8bitLt(threshold=6.0)` (HuggingFace `load_in_8bit=True`). It will detect outlier columns per forward pass, route them through FP16, and INT8-quantize the rest. Memory savings ~50%, accuracy within FP16 noise. For models < 6.7B, prefer NF4 weight-only (covered in ch-12) — outliers don't dominate there, and 4-bit weights give better accuracy at the same memory budget.

---

## Why this chapter exists

Before [[llm-int8]] (August 2022), the conventional wisdom from [[q8bert]] was that INT8 transformer inference works, you just need 1 epoch of QAT and per-channel weight scales. That story held cleanly up to ~2.7B parameters (the largest models tested in 2019–2021).

Then Dettmers, Lewis, Belkada & Zettlemoyer pushed past 6.7B. The whole framework collapsed: vanilla INT8 caused zero-shot accuracy to fall **off a cliff** — OPT-13B went from 70% MMLU to random-chance. The cliff wasn't a smooth degradation; it was a phase transition at a specific scale. Something qualitatively new was happening.

This chapter walks through:

1. **The empirical discovery.** What "the cliff" looks like and where it happens.
2. **The outlier feature structure.** Why ~6 dimensions out of 12k, why coordinated across layers, why magnitudes 20–100× normal.
3. **The mixed-precision decomposition.** The two-formula fix: vector-wise scale + FP16 outlier path.
4. **Why α = 6.0 specifically.** The threshold choice and what breaks at neighboring values.
5. **What LLM.int8() left unsolved.** Activation < 8-bit, weight ≤ 4-bit, weight-only PTQ — the openings that [[smoothquant]], [[awq]], [[gptq]], [[spqr]] each fill.

LLM.int8() is the first paper of the modern LLM PTQ era. Every paper from ch-08 onward is either fixing what LLM.int8 leaves unfixed (lower bits) or attacking the outlier problem from a different angle (migration in [[smoothquant]]/[[awq]], preservation in [[spqr]]/[[squeezellm]], rotation in [[quip]]/[[quarot]]).

---

## 1. The cliff — what the discovery actually looks like

The headline plot from the paper (Figure 1, reproduced verbatim from the abstract data):

| Model | Parameters | FP16 acc | Naive absmax INT8 acc | Vector-wise INT8 acc | LLM.int8() acc |
|---|---|---|---|---|---|
| OPT-125M | 125M | 38.5 | 38.0 | 38.4 | 38.5 |
| OPT-1.3B | 1.3B | 56.4 | 55.6 | 56.3 | 56.4 |
| OPT-2.7B | 2.7B | 60.7 | 60.0 | 60.5 | 60.7 |
| OPT-6.7B | 6.7B | 65.9 | **41.6 (crash)** | 65.5 | 65.9 |
| OPT-13B | 13B | 68.0 | **31.5 (crash)** | 67.4 | 68.0 |
| OPT-66B | 66B | 71.5 | **24.3 (crash)** | 70.8 | 71.4 |
| OPT-175B | 175B | 72.0 | (untested) | **collapses** | 71.9 |

Three things to notice:

1. **Naive absmax INT8** (the "obvious" PTQ baseline) tracks FP16 for 125M, 1.3B, 2.7B; then crashes at 6.7B.
2. **Vector-wise INT8** (per-token activation + per-channel weight scales) recovers most of the loss up to ~66B, then itself starts cracking at 175B.
3. **LLM.int8()** (vector-wise + FP16 outlier decomposition) holds flat all the way to 175B.

The cliff at 6.7B is the whole story. Below it, INT8 PTQ "works"; above it, you need the outlier handling.

---

## 2. The outlier feature structure

[[dettmers-llm-int8-blog]] gives the practitioner's reframing: outliers are not random noise but a **coordinated, structurally-sparse mechanism** the model uses for feature selection. Three quantitative findings:

### Finding 1 — gradual onset, then sudden coordination

Even 125M-class models have occasional outlier features. They are *sparse* (a few % of forward passes) and *layer-uncoordinated* (one layer's outlier dims disagree with the next). At ~6.7B the coordination snaps into place: all layers suddenly agree on the same outlier dimensions.

### Finding 2 — structural sparsity

At 6.7B with ~12k hidden dims, a typical 2k-token sequence contains ~150k outlier values. These live in only **6 distinct feature dimensions**. The remaining ~11994 dims behave normally.

This is the empirical fact that makes the mixed-precision split practical: you don't need 1000 FP16 columns, you need 6. The FP16 path is cheap.

### Finding 3 — sign coherence

Outlier dimensions maintain **consistent positive/negative signs across layers**, suggesting downstream layers "know where" to apply the feature-suppression operation. Before the phase shift, signs disagree between layers; after, they align.

### Finding 4 — rapid growth with scale

Outlier peak magnitude grows:

| Model | Peak outlier magnitude |
|---|---|
| 6B | ~15 |
| 13B | ~60 |
| 66B | ~95 |
| 175B | ~100+ |

By 175B, outliers are 30–100× larger than typical activations (which sit in `[−2, +2]`).

### Why simple clipping doesn't work

Naive instinct: "just clip activations > 6 to 6 and call it a day." This destroys the model. The blog's **dual-stream interpretation**:

- One stream learns explanatory features (the bulk of dimensions, normal magnitudes).
- A second stream uses large-magnitude outlier dimensions to *remove* noisy / context-irrelevant features via subtractive interaction.

Clipping the outliers cuts the subtraction signal, leaving the model with un-filtered features. Empirically clipping at threshold 6 on a 13B model drops zero-shot accuracy by ~10%. The outliers encode a routing decision; you can't throw them away.

### Emergence tracks perplexity, not parameter count

Models with different architectures emerge at different parameter scales but at similar **perplexity**. Two implications:

1. Emergence is a property of the learned function, not size per se.
2. Better-trained smaller models can pre-emerge. A heavily-trained 3B model might exhibit the outlier phase transition.

This means parameter count is a heuristic, not a guarantee. The right diagnostic for whether to use LLM.int8() vs simpler INT8 is: **measure per-channel activation amplitudes on your specific model**. If any 1% of channels are > 5× the rest, you're in outlier territory.

---

## 3. The vector-wise quantization rule

For input `X ∈ ℝ^{s × h}` and weight `W ∈ ℝ^{h × o}`, instead of a single scalar scale per tensor:

```math
c_{x,i} \;=\; 127 / \max_j |X_{ij}| \qquad \text{(per-token activation scale)}
```

```math
c_{w,j} \;=\; 127 / \max_i |W_{ij}| \qquad \text{(per-output-channel weight scale)}
```

```math
\hat{X} \;=\; \text{round}(c_x \odot X), \quad \hat{W} \;=\; \text{round}(c_w \odot W) \quad \in [-127, 127]
```

INT32 GEMM `X̂ · Ŵ`, then dequantize element-wise by `(c_x · c_w^⊤)^{−1}`. Each output entry has its own normalisation constant — equivalent to per-row × per-column rescaling.

**Why per-token (per-row activation) is better than per-tensor.** A single outlier token doesn't poison the entire matrix; only that row's scale shrinks. The other rows preserve their dynamic range.

**Why per-channel (per-column weight).** Weight magnitudes vary across output channels; per-tensor wastes resolution on small-magnitude channels.

Vector-wise alone recovers most of the gap up to ~66B (Table 1 of the paper). Past that, even per-token scale gets dominated by sequences where multiple tokens fall into the outlier dimensions simultaneously. That's where the FP16 path kicks in.

---

## 4. The mixed-precision decomposition (the load-bearing formula)

Define the **outlier set**:

```math
O \;=\; \bigl\{ i : \exists\, j,\; |X_{ji}| \ge \alpha \bigr\}, \qquad \alpha = 6.0
```

Let `R = {0, …, h−1} \ O` (the regular set). The decomposition:

```math
X \cdot W \;=\; \underbrace{\sum_{i \in O} X_{:,i} \cdot W_{i,:}}_{\text{FP16 GEMM}}
            \;+\; \underbrace{\sum_{i \in R} X_{:,i} \cdot W_{i,:}}_{\text{INT8 GEMM with vector-wise scales}}
```

Properties:

- **Typically |O| = 6–20 columns** out of 12k–14k hidden dims (<0.1% of dims, ~99.9% of values stay INT8).
- **The FP16 path is cheap** because |O| is tiny. The INT8 path carries the work.
- **No calibration, no retraining.** Outlier set is recomputed per forward pass from the activation absmax. The threshold α=6.0 is fixed.

### Why α = 6.0 specifically

Empirically, the outlier features at the emergent scale have magnitudes **15–100**; non-outlier dims stay below **6**. Setting α=6 captures all emergent dims at every scale tested (6.7B → 175B) while never picking up more than 0.1% of values.

Sensitivity at neighbouring thresholds:

| α | Captures all outlier dims? | False-positive rate | Effect |
|---|---|---|---|
| 3.0 | yes | ~1% of normal dims also flagged | FP16 path bloats, slight accuracy gain |
| **6.0** | **yes** | **< 0.1%** | **sweet spot** |
| 10.0 | yes at 6.7B; misses some at 13B+ | 0% | accuracy starts dropping at 13B |
| 20.0 | misses outliers at 6.7B-13B | 0% | model collapses |

Past 13B the safety margin between max-normal-magnitude (~6) and min-outlier-magnitude (~15) gets tight enough that 6.0 is the empirical sweet spot.

### Hyperparameter recipe

| Knob | Value |
|---|---|
| α (outlier threshold) | 6.0 |
| Activation scale | per-token (per-row), absmax |
| Weight scale | per-output-channel, absmax |
| Bit-width | INT8 (regular) + FP16 (outlier path) |
| Calibration data | none — purely runtime |
| Framework | `bitsandbytes.nn.Linear8bitLt`, HF `load_in_8bit=True` |

---

## 5. The bitsandbytes implementation

[[bitsandbytes-int8]] is the reference implementation. The `Linear8bitLt` module is a drop-in for `nn.Linear`. The forward pass:

```python
# Simplified bitsandbytes.nn.modules.Linear8bitLt.forward()

# 1. Per-row scale of input X (one scalar per token)
s_X = X.abs().max(dim=-1, keepdim=True).values / 127

# 2. Per-row scale of weight W (one scalar per output channel)
s_W = W.abs().max(dim=-1, keepdim=True).values / 127

# 3. Outlier mask: columns with any |X[i,j]| > threshold
outlier_cols = (X.abs().max(dim=0) > 6.0).nonzero().flatten()
regular_cols = torch.tensor([i for i in range(X.shape[-1]) if i not in outlier_cols])

# 4. Quantize the regular path
X_int8 = (X[:, regular_cols] / s_X).round().clamp(-128, 127).to(torch.int8)
W_int8 = (W[:, regular_cols] / s_W).round().clamp(-128, 127).to(torch.int8)

# 5. Mixed-precision GEMM
Y_int8 = (X_int8 @ W_int8.T) * (s_X * s_W.T)       # INT8 path
Y_fp16 = X[:, outlier_cols] @ W[:, outlier_cols].T  # FP16 fallback

Y = Y_int8 + Y_fp16
```

The actual CUDA kernel (`gemm_mixed_8bit_lt` in `csrc/kernels.cu`) is more aggressive — it overlaps the INT8 and FP16 GEMMs on separate streams and dequantizes with FP16 accumulation — but the algorithm above captures the essential structure.

### What the `Int8Params` wrapper does

`bitsandbytes.nn.Int8Params` packs (INT8 weight + FP16 scales + outlier index) into one PyTorch Parameter. This lets `model.to(device)` move everything atomically and lets the standard HF model-loading code work without modification.

### Usage in HuggingFace

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,           # the α from the paper
    llm_int8_has_fp16_weight=False,   # don't keep FP16 master (memory)
)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", quantization_config=bnb_config)
```

Memory footprint for OPT-13B:

| Format | Bytes |
|---|---|
| FP16 | 26 GB |
| **INT8 (LLM.int8)** | **~13 GB (50%)** |
| 4-bit NF4 | 7 GB (27%) |

LLM.int8() gets 2× compression at essentially zero accuracy loss. That's enough to fit OPT-13B on one A100-24GB; OPT-30B on one A100-40GB; OPT-175B on a single 8×A100 node.

---

## 6. What LLM.int8() left unsolved (the openings for ch-08, ch-09, ch-11, ch-13)

LLM.int8() is INT8 weights + INT8 activations + FP16 outliers. It does NOT solve:

| Problem | Solved by | Where |
|---|---|---|
| Weight-only PTQ at 4-bit (more memory savings) | [[gptq]] | ch-08 |
| Weight-only PTQ at 3-bit | [[gptq]] + [[quip]] | ch-08, ch-13 |
| Sub-2-bit weights | [[aqlm]], [[quip-sharp]] | ch-13, ch-14 |
| Outlier migration instead of isolation | [[smoothquant]], [[awq]] | ch-09 |
| Outlier preservation (keep outlier *weights*, not outlier activations) | [[spqr]], [[squeezellm]], [[owq]] | ch-11 |
| Outlier elimination via rotation | [[quip]], [[quip-sharp]], [[quarot]], [[spinquant]] | ch-13, ch-14 |
| Activation quant below 8-bit | [[smoothquant]] (W8A8 → W4A8), [[awq]], [[quarot]] (W4A4) | ch-09, ch-14 |
| Production INT8 GEMM kernel | [[machete-kernel]] for Hopper | ch-19 |

The five lineages above are not exclusive — most production stacks now combine GPTQ-W4 + AWQ-style scaling + Marlin kernel, and the rotation methods are competitive with all of them at W4A4.

But every one of them traces back to the LLM.int8() observation: **outliers are real, they're concentrated, and they require explicit handling**. The Aug 2022 emergence threshold paper is the watershed.

---

## 7. Practitioner's playbook

```text
For LLM deployment at INT8 in 2026:

□ Model ≤ 6.7B?            → use NF4 weight-only ([[bitsandbytes-nf4]], ch-12). Better accuracy/byte than INT8.
□ Model 7B-70B?            → LLM.int8() (load_in_8bit=True) for "just works" deployment.
                            → GPTQ-W4 + Marlin kernel (vLLM, AutoGPTQ) for production serving — 4× memory, 2-4× throughput.
                            → AWQ-W4 (TinyChat, AutoAWQ) if you need a single per-channel scale recipe.
□ Model 100B+?             → Same as 7B-70B; outlier handling is critical regardless of method choice.
□ Activation quant needed? → SmoothQuant for W8A8 or AWQ-style scaling combined with the weight quant method.
□ Sub-4-bit weights?       → GPTQ + QuIP rotation preprocessing, or AQLM additive quantization.
□ Fine-tuning quantized?   → QLoRA on top of NF4 base ([[qlora]], ch-12).
```

---

## 8. Why LLM.int8() is the right paper to read first

It's short (8 pages). The empirical observation is unambiguous. The fix is two formulas. The implementation is open-source. The blog companion makes the intuition vivid. And every subsequent paper either builds on or attacks the framework it establishes.

If you only read one quantization paper from 2022, read [[llm-int8]]. Pair it with [[dettmers-llm-int8-blog]] for the "why" of outliers.

---

## Connections and what's next

- **Forward to [[gptq]] (ch-08)** — weight-only PTQ at 3-4 bits. Solves the "make the weights smaller" problem LLM.int8() doesn't touch.
- **Forward to [[smoothquant]] (ch-09)** — *migrates* outliers from activations to weights via the DFQ equivalent-transformation trick, then quantizes both at INT8. No FP16 path needed.
- **Forward to [[awq]] (ch-09)** — per-channel scale `s_j` chosen by activation magnitude, applied to the weight matrix; salient-channel preservation. W4A16 default of 2023.
- **Forward to [[spqr]] (ch-11)** — preserves outlier *weights* (top 0.5%) in FP16, quantizes the rest aggressively to ~3 bits.
- **Forward to [[quip]] (ch-13)** — *rotates* the activations and weights with a random orthogonal U,V so outliers disperse evenly across all dimensions, eliminating the structure entirely.
- **Forward to [[quarot]] (ch-14)** — Hadamard rotations folded into weights for W4A4 PTQ. The mature 2024 form of the rotation idea.
- **Back to [[quantization-mapping]] (ch-05)** — per-tensor activation scale was Krishnamoorthi's sweet spot for CNNs. The "per-tensor activation cell" is what LLM.int8 shows to be empty at LLM scale.
- **Back to [[q8bert]] (ch-06)** — the BERT-scale precursor that didn't see outliers because BERT-Base is sub-cliff.

## Further reading

- [[llm-int8]] — Dettmers et al. 2022. 8-page paper, read first.
- [[dettmers-llm-int8-blog]] — the practitioner's reframing of outliers as emergent features.
- [[bitsandbytes-int8]] — reference implementation; read `csrc/kernels.cu` for the actual kernel.
- [[quarot]] / [[spinquant]] — read after ch-13/14 for the rotation-based alternative to outlier isolation.
- Wei et al. *"Outlier Suppression"* (2022) — independent contemporary work that handles outliers via clipping + LayerNorm gain scaling; less successful but worth reading for the contrast.
