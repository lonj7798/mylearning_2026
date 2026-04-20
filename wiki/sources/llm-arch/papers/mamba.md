<!-- scope: Mamba — selective SSMs as attention alternative
     deps: [[attention-is-all-you-need]]
     see-also: [[mamba-2]], [[flash-attention]]
-->

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Core Insight:** Making SSM parameters input-dependent (selective) gives state space models the content-based reasoning they lacked, making them competitive with attention.
- **Guideline:** Consider Mamba for long-sequence tasks where linear-time scaling matters; watch for weaknesses on tasks requiring exact retrieval from arbitrary positions.
- **Authors:** Albert Gu, Tri Dao
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2312.00752
- **Relevant chapters:** state space models, sequence modeling, attention alternatives, efficient architectures

## Abstract
Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast inference (5x higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.

## Key Contributions
- Identifies content-based reasoning as the key missing ingredient in prior subquadratic architectures (linear attention, SSMs, etc.)
- Introduces selective state spaces: SSM parameters (A, B, C, delta) become functions of the input, enabling the model to selectively remember or forget information based on content
- Designs a hardware-aware parallel algorithm that efficiently computes the selective SSM in recurrent mode on GPUs, compensating for the loss of convolutional efficiency
- Proposes a simplified architecture that eliminates both attention and MLP blocks, using only selective SSM layers with gating
- Achieves 5x inference throughput over Transformers with linear (not quadratic) sequence length scaling, while matching or exceeding Transformer quality

## Architecture Details
- **Classical SSM:** h'(t) = Ah(t) + Bx(t), y(t) = Ch(t). Discretized: h_t = A_bar * h_{t-1} + B_bar * x_t, y_t = C * h_t. Parameters A, B, C are fixed (input-independent), enabling computation via convolution
- **Selective SSM:** B, C, and the discretization step delta become functions of the input: B_t = s_B(x_t), C_t = s_C(x_t), delta_t = softplus(s_delta(x_t)). This input-dependence makes the system non-linear and prevents convolution-based computation
- **Selection mechanism:** By making parameters input-dependent, the model can: (1) selectively ignore irrelevant tokens (small delta), (2) propagate important information (large delta), (3) reset state when encountering a new context. This provides the content-based reasoning that fixed SSMs lack
- **Hardware-aware algorithm:** Since selective SSMs cannot use FFT-based convolution, Mamba uses a parallel scan algorithm. The key optimization is keeping the state in GPU SRAM (not HBM), similar to FlashAttention's IO-aware approach. The recurrence is computed in a fused kernel that avoids materializing the full state in HBM
- **Mamba block:** Each block consists of: linear projection expanding the input, 1D convolution, selective SSM, gating (element-wise multiply with a parallel projection through SiLU), and linear projection back down. No attention, no separate MLP
- **State expansion:** The state dimension N (typically 16) controls the model's memory capacity. Unlike Transformers whose "memory" (KV cache) grows linearly with sequence length, Mamba's state is fixed-size
- **Inference:** During generation, Mamba operates as a true RNN: constant time per step, constant memory. No KV cache that grows with sequence length. This gives the 5x throughput advantage
- **Scaling:** Mamba-3B outperforms Transformers of equal size and matches 2x larger Transformers on language modeling

## Tradeoffs Discussed
- Making SSM parameters input-dependent prevents the use of efficient convolution (FFT), necessitating a custom parallel scan algorithm that is more complex to implement
- The fixed-size state (dimension N) limits the model's ability to recall information from arbitrarily far in the past — unlike Transformers which can attend to any position in the KV cache. This is a fundamental capacity-vs-efficiency tradeoff
- The hardware-aware algorithm is specifically optimized for GPU memory hierarchies; performance on other hardware (TPUs, custom accelerators) may differ
- Mamba has not been as thoroughly validated at very large scales (100B+ parameters) as Transformers, leaving open questions about scaling behavior
- The lack of an explicit attention mechanism means Mamba cannot perform exact copying or retrieval from specific positions as easily as Transformers, which may matter for certain tasks (e.g., in-context learning with many examples)
