<!-- scope: curated LLM reading list
     deps: [[ch-01]]
     see-also: [[raschka-catalog]]
-->

# Understanding Large Language Models: A Cross-Section of the Most Relevant Literature

- **Core Insight:** Curated reading paths accelerate learning more than breadth.
- **Guideline:** Follow structured paper sequences, not random exploration.

- **Author:** Sebastian Raschka, PhD
- **URL:** https://magazine.sebastianraschka.com/p/understanding-large-language-models
- **Relevant chapters:** All chapters (foundational overview); especially architecture, training, alignment

## Summary
A curated, chronological reading list of the most important LLM research papers, organized into four sections: architecture and tasks, scaling and efficiency, alignment, and RLHF. Raschka walks through 19 foundational papers covering attention mechanisms, transformer architecture, BERT, GPT, scaling laws, parameter-efficient fine-tuning, and alignment techniques like InstructGPT and Constitutional AI.

## Key Content

### Section 1: Understanding the Main Architecture and Tasks

**Bahdanau Attention (2014)** — Introduces attention mechanisms for RNNs to improve long-range sequence modeling. The mechanism allows RNNs to "translate longer sentences more accurately," establishing foundational concepts later refined in transformers. Addresses the limitation of RNNs in handling long-range dependencies.

**Attention Is All You Need (2017)** — The original transformer paper by Vaswani et al. Core architecture components:
- Encoder-decoder architecture
- Scaled dot product attention mechanism
- Multi-head attention blocks
- Positional input encoding

These concepts "remain the foundation of modern transformers."

**On Layer Normalization in the Transformer Architecture (2020)** — The placement of LayerNorm in transformer architectures remains debated. Two primary variants:
1. **Post-LN Transformer:** LayerNorm positioned between residual blocks (as shown in original paper figure)
2. **Pre-LN Transformer:** Updated architecture placing LayerNorm differently (matching the official code implementation)

Pre-LN is suggested to work better by "addressing gradient problems," though it can result in representation collapse. A newer proposal called ResiDual combines advantages of both.

**Fast Weight Programmers (Schmidhuber, 1991)** — Conceptually similar to modern transformers approximately 25 years before "Attention Is All You Need." A slow-learning feedforward network programs fast weights of another network. In modern terminology: keys and values (analogous to FROM and TO), queries (INPUT). Later work established that linearized self-attention and transformers are mathematically equivalent to these 1990s fast weight programmers, now called "linear Transformers."

**ULMFiT (2018)** — Proposes pretraining language models and transfer learning for downstream tasks. Three-stage finetuning:
1. Train a language model on large text corpus
2. Finetune on task-specific data to adapt style and vocabulary
3. Finetune classifier on task data with gradual layer unfreezing to prevent catastrophic forgetting

Established the now-standard approach of "training a language model on a large corpus and then finetuning it on a downstream task."

**BERT (2018)** — Encoder-style transformer with masked language modeling and next-sentence prediction pretraining objectives. Represents a bifurcation toward encoder-style architectures for predictive tasks (text classification, information extraction). RoBERTa later simplified pretraining by removing next-sentence prediction.

**GPT-1 (2018)** — Decoder-style transformer using autoregressive training via next-word prediction. Contrasts with BERT's bidirectional nature. Forms basis for "today's most influential LLMs, such as ChatGPT." GPT-2 and GPT-3 demonstrated zero-shot and few-shot learning abilities and emergent capabilities at scale.

**BART (2019)** — Combines encoder and decoder components, integrating strengths of both BERT-style and GPT-style architectures. Achieves bidirectional understanding benefits while maintaining generative capabilities needed for tasks like summarization and translation.

### Section 2: Scaling Laws and Improving Efficiency

**FlashAttention (2022)** — Replaces the scaled dot product attention mechanism with a more efficient implementation. Described as "one mechanism I have seen most often referenced lately."

**Cramming (2022)** — Trained BERT-style model for 24 hours on single GPU compared to original BERT requiring 16 TPUs for four days. Critical finding: while smaller models show higher throughput, they learn less efficiently. "Larger models do not require more training time to reach a specific predictive performance threshold."

**LoRA (2021)** — Decomposes weight changes (delta-W) into lower-rank representations based on the observation that "pretrained large language models have a low intrinsic dimension when adapted to a new task." Rather than updating all parameters, LoRA updates low-rank matrix decompositions, dramatically reducing trainable parameters while maintaining performance. Described as "one of the most influential approaches for finetuning large language models in a parameter-efficient manner."

**Gopher (2022)** — 280 billion parameter model with 80 layers trained on 300 billion tokens. Architecture modifications:
- RMSNorm replaces LayerNorm (avoids batch-size dependency; "stabilizes the training in deeper architectures")

Key findings:
- Increasing model size benefits "comprehension, fact-checking, and identification of toxic language the most"
- Tasks involving "logical and mathematical reasoning benefit less from architecture scaling"
- Duplicated data doesn't benefit or hurt performance
- "Doubling the batch size halves the training time but doesn't hurt convergence"

**Chinchilla / Training Compute-Optimal LLMs (2022)** — Central thesis: contemporary LLMs are "significantly undertrained." Chinchilla (70B params) outperforms GPT-3 (175B params) using 1.4 trillion tokens vs GPT-3's 300 billion. "The number of training tokens is as vital as the model size." This is the linear scaling law for LLM training.

**Pythia (2023)** — Open-source suite of 70M to 12B parameter models. Features Flash Attention and Rotary Positional Embeddings. Key findings:
- Duplicated training data neither benefits nor harms performance
- Training order doesn't influence verbatim memorization
- "Pretrained term frequency influences task performance"
- "Doubling the batch size halves the training time"

### Section 3: Alignment

**InstructGPT (2022)** — Three-stage process:
1. **Supervised Finetuning:** Human-generated prompt-response pairs finetune GPT-3
2. **Reward Model Training:** Humans rank model outputs to train a separate reward model
3. **Reinforcement Learning:** Reward model guides optimization using Proximal Policy Optimization (PPO)

This describes "the idea behind ChatGPT—according to recent rumors, ChatGPT is a scaled-up version of InstructGPT."

**Constitutional AI (2022)** — Self-training mechanism for creating "harmless" AI without direct human supervision. Uses human-provided rules as constraints for RL-based self-training. Reduces human annotation burden compared to RLHF.

**Self-Instruct (2022)** — Bootstrapping: using an LLM's own generations to create training data. Four-step process:
1. Seed task pool with 175 human-written instruction examples
2. Use pretrained LLM to classify task category
3. Generate responses using the pretrained LLM
4. Collect, prune, and filter responses before adding to training pool

### Architecture Evolution Timeline
1. **Attention Mechanisms** (2014) — Improved RNN long-range modeling
2. **Transformers** (2017) — Encoder-decoder with self-attention
3. **BERT** (2018) — Bidirectional encoders for understanding
4. **GPT** (2018) — Unidirectional decoders for generation
5. **BART** (2019) — Combined encoder-decoder approach

### Training Paradigm Shifts
1. Pretraining on large corpora
2. Task-specific finetuning (transfer learning)
3. Instruction alignment (matching human preferences)
4. RLHF integration (preference-based optimization)
5. Parameter-efficient methods (LoRA, adapters)

## Notable Insights
- The reading list is designed to be consumed chronologically, building understanding progressively from attention through alignment.
- Pre-LN vs Post-LN transformer debate remains unresolved, with practical implications for training stability.
- The Chinchilla scaling law fundamentally changed how the community thinks about training budgets: model size and training data should be balanced, not just one maximized.
- LoRA's insight that pretrained models have "low intrinsic dimension" when adapted has broad implications beyond NLP.
- The progression from human-labeled data (InstructGPT) to self-generated data (Self-Instruct) to rule-based self-training (Constitutional AI) shows a clear trajectory toward reducing human annotation costs.
