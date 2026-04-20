<!-- scope: master index of core insights from all LLM architecture reference sources
     deps: [[outline]]
     see-also: [[index]]
-->

# LLM Architecture — Core Insights Index

Every source in this library distilled to its one-sentence contribution and practical guideline.
Browse by category or search for a specific topic.

---

## Papers — Foundational

| Source | Core Insight | Guideline |
|--------|-------------|-----------|
| [[word2vec]] | Words as dense vectors where arithmetic works | Learned embeddings > one-hot for all discrete tokens |
| [[seq2seq]] | Encoder-decoder maps sequences but fixed-length vector is a bottleneck | Consider whether your architecture has an information bottleneck |
| [[bahdanau-attention]] | Soft alignment eliminates the fixed-length bottleneck | Add attention when variable-length input must be compressed |
| [[attention-is-all-you-need]] | Self-attention alone suffices for sequence modeling | Start from Transformer blueprint; deviate only with evidence |
| [[resnet]] | Skip connections enable very deep networks | Always use residual connections when stacking many layers |
| [[layer-norm]] | Normalize across features, not batch — works for variable-length | Use LayerNorm (or RMSNorm) for all Transformer models |
| [[pre-norm-vs-post-norm]] | Pre-norm is more stable; post-norm needs warmup | Place normalization before each sub-layer, not after |
| [[gpt-1]] | Generative pre-training + fine-tuning transfers across tasks | Pre-train generatively first, then fine-tune |
| [[gpt-2]] | Scale enables zero-shot multitask transfer | Invest equally in data quality and model size |
| [[gpt-3]] | In-context learning emerges at scale | Try few-shot prompting before fine-tuning |
| [[bert]] | Bidirectional masking produces richer representations | Use encoders for understanding tasks, decoders for generation |
| [[pitfalls-next-token]] | Teacher forcing can fail even in-distribution on planning tasks | Consider multi-token prediction for planning/reasoning tasks |
| [[emergent-abilities]] | Some capabilities appear suddenly at scale | Evaluate at multiple scales; don't assume linear capability growth |
| [[emergent-mirage]] | "Emergence" may be a measurement artifact of non-linear metrics | Use continuous metrics when evaluating across scales |

## Papers — Components & Techniques

| Source | Core Insight | Guideline |
|--------|-------------|-----------|
| [[rope]] | Relative position via rotation in complex space | Use RoPE as default; pairs well with context extension |
| [[alibi]] | Linear distance penalty is simpler and extrapolates | Use when training-free extrapolation matters; prefer RoPE for flexibility |
| [[yarn]] | Temperature-scaling extends RoPE context with minimal fine-tuning | Apply NTK-aware interpolation + temperature scaling, fine-tune ~400 steps |
| [[flash-attention]] | Attention bottleneck is memory IO, not FLOPs | Always use FlashAttention; never materialize full N×N matrix in HBM |
| [[flash-attention-2]] | Better warp partitioning closes gap to optimized GEMM | Profile non-matmul overhead; shift work to tensor-core matmuls |
| [[mqa]] | Single KV head cuts cache H-fold with minor quality loss | Maximum KV-cache reduction if you can retrain |
| [[gqa]] | Grouped KV heads: sweet spot between MHA quality and MQA speed | Default to GQA with G = H/8 for new models |
| [[glu-variants]] | Gated linear units consistently outperform standard activations | Use SwiGLU as default FFN activation; set d_ff = 8/3 × d_model |
| [[rmsnorm]] | Drop mean-centering; only re-scaling is essential | Use RMSNorm instead of LayerNorm in all new models |
| [[switch-transformer]] | Top-1 routing suffices for MoE | Start with top-1 + load-balancing loss (α ~0.01) |
| [[mamba]] | Selective (input-dependent) SSM parameters make SSMs competitive | Consider Mamba for long-sequence tasks; watch for retrieval weaknesses |
| [[mamba-2]] | SSMs and linear attention are mathematically dual | Use SSD framework to pick faster path per hardware/length |
| [[paged-attention]] | Virtual memory for KV cache eliminates fragmentation | Use vLLM or PagedAttention-based serving as default |
| [[speculative-decoding]] | Draft-then-verify gives mathematically equivalent sampling 2-3× faster | Use draft model for inference speedup on autoregressive generation |

## Papers — Scaling & Training

| Source | Core Insight | Guideline |
|--------|-------------|-----------|
| [[scaling-laws-kaplan]] | Loss follows clean power laws spanning 7 orders of magnitude | Use power-law fits to forecast loss before committing resources |
| [[chinchilla]] | Most models are undertrained on data; scale tokens ≈ parameters | For fixed compute, allocate equally to model size and data |
| [[scaling-data-constrained]] | Repeating data has diminishing returns beyond ~4 epochs | Prioritize collecting unique data over repeating existing data |
| [[instructgpt-rlhf]] | RLHF makes small models outperform larger ones on human preference | Align via SFT then RLHF; small aligned > large unaligned |
| [[dpo]] | Skip the reward model; policy itself defines preferences | Use DPO over full RLHF when simplicity matters |
| [[constitutional-ai]] | AI self-critique guided by principles scales alignment supervision | Define constitutional principles; use model to critique its own outputs |
| [[megatron-lm]] | Tensor parallelism splits matrix ops across GPUs within a layer | Split attention heads and FFN columns for intra-layer parallelism |
| [[zero]] | Shard optimizer/gradients/parameters; each GPU stores 1/N | Progressively shard (ZeRO-1→2→3) as model size grows |
| [[ultra-scale-playbook]] | 5D parallelism (DP, TP, PP, SP, EP) must be optimized jointly | Benchmark thousands of configs; sweet spot depends on hardware |

## Papers — Vision & Multimodal

| Source | Core Insight | Guideline |
|--------|-------------|-----------|
| [[vit]] | Transformers work for images; patches as tokens | Tokenize non-text modalities naturally; architecture transfers |

## Model Reports

| Source | Core Insight | Guideline |
|--------|-------------|-----------|
| [[llama-1]] | Public data + proven techniques beats proprietary models | Data quality and scale matter more than architecture novelty |
| [[llama-2]] | GQA enables 70B at reasonable serving cost | Choose attention variant based on serving hardware |
| [[llama-3]] | 15T tokens for 405B validates Chinchilla | Invest in data pipeline proportional to param count |
| [[llama-4]] | MoE + iRoPE enables 10M context at feasible serving cost | Decouple capacity from per-token compute via MoE |
| [[deepseek-v2]] | MLA compresses KV cache 93% via low-rank latent | KV cache size is an architectural choice, not fixed cost |
| [[deepseek-v3]] | Aux-loss-free bias balancing > gradient penalties for MoE | Use adaptive bias for routing stability |
| [[deepseek-r1]] | Pure RL produces emergent chain-of-thought reasoning | RL reward design can teach what SFT cannot |
| [[gemma-3]] | 5:1 local/global interleaving captures most of global's benefit | Not every layer needs full attention |
| [[jamba]] | SSM + attention hybrid (1:7) balances efficiency and recall | Match layer type to role: SSM for flow, attention for recall |
| [[mistral-7b]] | Sliding window + rolling buffer = fixed-memory inference | Design attention for bounded memory when deployment-constrained |
| [[mixtral]] | Sparse MoE gives 6× quality-per-FLOP at inference | MoE buys quality without proportional inference cost |
| [[qwen-3]] | Single model operates in thinking and non-thinking modes | Inference-time compute allocation is an architecture decision |
| [[olmo-2]] | Two-stage training + model souping > either alone | Save high-quality data for the final training stage |
| [[phi-4]] | 40% synthetic data makes 14B competitive with 70B+ | Data quality substitutes for parameter count |
| [[dbrx]] | Fine-grained MoE (16 experts, top-4) = 65× more combinations | More smaller experts with higher top-k = more expressive routing |

## Blog Posts & Explainers

| Source | Core Insight | Guideline |
|--------|-------------|-----------|
| [[alammar-illustrated-transformer]] | Visual decomposition makes attention intuitive | Use visual step-by-step when learning/teaching attention |
| [[alammar-illustrated-gpt2]] | GPT-2 is the minimal viable LLM architecture | Understand GPT-2 as the baseline for all decoder-only models |
| [[bendersky-cross-entropy]] | Cross-entropy = KL divergence + const = MLE | Three frameworks, one gradient — not an arbitrary choice |
| [[hf-perplexity]] | Evaluation methodology affects PPL by 15%+ | Always report evaluation methodology with PPL numbers |
| [[eleutherai-rope]] | RoPE derives from first principles via complex rotation | Mathematical derivation reveals why it extrapolates |
| [[hf-positional-encoding-design]] | Positional encoding evolved iteratively: integer → binary → sinusoidal → RoPE | Study historical evolution to understand current solutions |
| [[flash-attention-explained]] | GPU SRAM vs HBM determines attention's real bottleneck | Optimize memory movement, not arithmetic |
| [[hf-mixture-of-experts]] | Load balancing determines whether expert specialization occurs | Focus on routing stability for MoE |
| [[raschka-llm-architecture-comparison]] | Modern LLMs converge on shared components; variation is in routing | Compare architectures by departures from baseline |
| [[raschka-attention-variants]] | Attention is a 7+ family design space | Pick variant based on deployment constraint |
| [[raschka-kv-cache]] | KV cache enables O(n) per-token but creates memory challenges | Design attention with serving in mind |
| [[raschka-reasoning-llms]] | Four reasoning approaches with different cost profiles | Match approach to compute budget |
| [[weng-transformer-family]] | Variants form taxonomy: position × attention × computation | Classify new architectures by which dimension they innovate |
| [[weng-why-we-think]] | Test-time compute is a new scaling axis | Consider both training and inference compute |

---

**Total: 75 sources indexed** — 38 papers, 15 model reports, 22 blog posts/explainers.
