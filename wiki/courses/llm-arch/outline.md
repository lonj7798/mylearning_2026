# LLM Architecture: Foundations to Frontier

**29 chapters · 7 phases · Target: model architecture research proficiency**

Learner: UC Berkeley EECS grad. Solid math/CS. Hybrid depth: explainers first, papers for precision.

---

## Phase 1: Foundations (Ch 1–4)

### Ch 1 — Language Modeling Fundamentals
Autoregressive vs masked modeling, cross-entropy loss, perplexity, teacher forcing.

### Ch 2 — The Attention Mechanism
From RNNs to attention, scaled dot-product QKV, O(n²) cost, multi-head attention.

### Ch 3 — The Original Transformer
Vaswani et al. deep dive: encoder-decoder, residual stream, positional encoding, training innovations.

### Ch 4 — Decoder-Only LLMs
Causal masking, GPT lineage, why decoder-only won, pre-train/prompt paradigm shift.

---

## Phase 2: Components Under the Microscope (Ch 5–9)

### Ch 5 — Tokenization
BPE, WordPiece, SentencePiece, vocabulary size tradeoffs, tokenization failures.

### Ch 6 — Positional Encoding
Sinusoidal → learned → RoPE → ALiBi → iRoPE. Long-context implications of each.

### Ch 7 — Attention Variants
MHA → MQA → GQA → MLA, Flash Attention, sliding window. The memory/quality/speed trilemma.

### Ch 8 — FFN, Activations, and Width
FFN as key-value memory, ReLU → GELU → SwiGLU, GLU variants, width vs depth tradeoffs.

### Ch 9 — Normalization and Residual Connections
LayerNorm vs RMSNorm, pre-norm vs post-norm, QK-norm, residual stream as information highway.

---

## Phase 3: Scaling and Training (Ch 10–13)

### Ch 10 — Scaling Laws
Kaplan, Chinchilla, compute-optimal training, emergent abilities debate, inference-time scaling.

### Ch 11 — Pre-training
Data curation, training stability, curriculum learning, two-stage training, synthetic data.

### Ch 12 — Post-training and Alignment
SFT, RLHF, DPO, Constitutional AI, reasoning-specific training (DeepSeek-R1, Qwen 3).

### Ch 13 — Distributed Training
Data/tensor/pipeline parallelism, ZeRO, FSDP, expert parallelism for MoE.

---

## Phase 4: Advanced Architectures (Ch 14–17)

### Ch 14 — Mixture of Experts
Routing strategies, load balancing, auxiliary-loss-free balancing, fine-grained vs coarse, shared experts.

### Ch 15 — State Space Models and Alternatives to Attention
S4, Mamba, Mamba-2/SSD, linear attention, hybrid architectures.

### Ch 16 — Long Context
RoPE scaling (PI, NTK-aware, YaRN), iRoPE, chunked attention, RAG as alternative, lost-in-the-middle.

### Ch 17 — Multimodal Architectures
ViT, early/late fusion, cross-modal attention, SigLIP, Pan-and-Scan.

---

## Phase 5: Architecture Case Studies (Ch 18–24)

Each chapter: **dissect → compare → critique**

### Ch 18 — LLaMA 3 and Llama 4
Modern baseline → extreme-context MoE. GQA+RoPE+SwiGLU, then iRoPE + 16/128-expert MoE + 10M tokens.

### Ch 19 — DeepSeek-V3
Most architecturally novel. MLA (93% KV-cache reduction), 256 routed + 1 shared expert, aux-loss-free balancing, multi-token prediction.

### Ch 20 — Gemma 3
5:1 local/global attention, SoftCap logit capping, SigLIP + Pan-and-Scan multimodal.

### Ch 21 — Jamba
Triple hybrid: SSM + Transformer + MoE. 256K context. Proves you can mix paradigms.

### Ch 22 — Mamba-2
Pure SSM, zero attention. SSD framework bridges SSMs and linear attention. The radical bet.

### Ch 23 — Qwen 3
Dual-mode thinking/non-thinking. 30B total, 3B active MoE. Inference-time compute allocation.

### Ch 24 — OLMo 2
Fully open ablations. QK-norm, two-stage training. The bridge from studying to researching.

---

## Phase 6: Inference and Systems (Ch 25–27)

### Ch 25 — KV-Cache and Serving
PagedAttention, continuous batching, prefix caching, architecture-serving co-design.

### Ch 26 — Quantization and Compression
INT8/INT4, GPTQ, AWQ, outlier features, distillation, Pareto frontier.

### Ch 27 — Speculative Decoding
Draft models, parallel verification, Medusa, multi-token prediction, acceptance rates.

---

## Phase 7: Research Practice (Ch 28–29)

### Ch 28 — Reading and Critiquing Architecture Papers
Paper anatomy, mental ablation, ablation table literacy, reproducibility, research taste.

### Ch 29 — Designing Architecture Experiments
Hypothesis → ablation, compute-efficient experimentation, common pitfalls, writing up results.
