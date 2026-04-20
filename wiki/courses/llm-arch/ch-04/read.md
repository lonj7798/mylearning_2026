# Chapter 4: Decoder-Only LLMs

<!-- scope: causal masking, GPT lineage (GPT-1/2/3), why decoder-only won, encoder-decoder tradeoffs, pre-train/prompt paradigm shift
     deps: [[ch-03]]
     see-also: [[ch-10]], [[ch-18]]
-->

## Overview

The original Transformer from [[ch-03]] was an encoder-decoder architecture designed for sequence-to-sequence tasks like machine translation. Within a year of its publication, two divergent bets were placed: BERT ([[bert|paper]]) took the encoder and discarded the decoder; GPT took the decoder and discarded the encoder. By 2020, the decoder-only lineage had won so decisively that every frontier model — GPT-4, Claude, Gemini, LLaMA, DeepSeek — uses a decoder-only Transformer.

This chapter traces the GPT lineage from GPT-1 ([[gpt-1|paper]]) (117M parameters, 2018) through GPT-3 ([[gpt-3|paper]]) (175B parameters, 2020), establishing how each generation shifted the paradigm for how we use language models. The central question is not just what decoder-only models look like architecturally — that is straightforward — but *why* this simpler architecture defeated the alternatives. The answer involves causal masking, scaling behavior, training efficiency, and a paradigm shift from fine-tuning to prompting that nobody fully anticipated.

Understanding this lineage is prerequisite for every architecture case study in Phase 5 ([[ch-18]] through [[ch-24]]), because every modern model is a variation on the decoder-only theme established here.

---

## 1. Causal Masking: The Mechanism That Makes It Work

The single architectural difference between an encoder Transformer block and a decoder-only Transformer block is the **causal mask** (also called the attention mask). In the encoder, every token attends to every other token — full bidirectional attention. In the decoder-only model, token $t$ can only attend to tokens $1, 2, \ldots, t$ — it cannot look at the future.

The causal mask is applied before the softmax in the attention computation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M$ is a matrix with $M_{ij} = 0$ if $j \leq i$ (allowed) and $M_{ij} = -\infty$ if $j > i$ (blocked). In practice, GPT-2 uses $-10^9$ rather than $-\infty$ to avoid numerical issues. After softmax, $e^{-10^9} \approx 0$, so future positions contribute zero attention weight.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0; font-family:monospace;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Causal Attention Mask (4x4 example)</div>
<div style="display:grid; grid-template-columns:60px repeat(4, 60px); gap:2px; max-width:360px;">
<div style="padding:8px; color:#888; font-size:12px;"></div>
<div style="padding:8px; color:#e94560; text-align:center; font-size:12px; font-weight:bold;">The</div>
<div style="padding:8px; color:#e94560; text-align:center; font-size:12px; font-weight:bold;">cat</div>
<div style="padding:8px; color:#e94560; text-align:center; font-size:12px; font-weight:bold;">sat</div>
<div style="padding:8px; color:#e94560; text-align:center; font-size:12px; font-weight:bold;">on</div>

<div style="padding:8px; color:#e94560; font-size:12px; font-weight:bold;">The</div>
<div style="padding:8px; background:#0f3460; color:#e0e0e0; text-align:center; border-radius:4px;">1.0</div>
<div style="padding:8px; background:#2a0a0a; color:#555; text-align:center; border-radius:4px;">-inf</div>
<div style="padding:8px; background:#2a0a0a; color:#555; text-align:center; border-radius:4px;">-inf</div>
<div style="padding:8px; background:#2a0a0a; color:#555; text-align:center; border-radius:4px;">-inf</div>

<div style="padding:8px; color:#e94560; font-size:12px; font-weight:bold;">cat</div>
<div style="padding:8px; background:#0f3460; color:#e0e0e0; text-align:center; border-radius:4px;">0.6</div>
<div style="padding:8px; background:#0f3460; color:#e0e0e0; text-align:center; border-radius:4px;">0.4</div>
<div style="padding:8px; background:#2a0a0a; color:#555; text-align:center; border-radius:4px;">-inf</div>
<div style="padding:8px; background:#2a0a0a; color:#555; text-align:center; border-radius:4px;">-inf</div>

<div style="padding:8px; color:#e94560; font-size:12px; font-weight:bold;">sat</div>
<div style="padding:8px; background:#0f3460; color:#e0e0e0; text-align:center; border-radius:4px;">0.2</div>
<div style="padding:8px; background:#0f3460; color:#e0e0e0; text-align:center; border-radius:4px;">0.5</div>
<div style="padding:8px; background:#0f3460; color:#e0e0e0; text-align:center; border-radius:4px;">0.3</div>
<div style="padding:8px; background:#2a0a0a; color:#555; text-align:center; border-radius:4px;">-inf</div>

<div style="padding:8px; color:#e94560; font-size:12px; font-weight:bold;">on</div>
<div style="padding:8px; background:#0f3460; color:#e0e0e0; text-align:center; border-radius:4px;">0.1</div>
<div style="padding:8px; background:#0f3460; color:#e0e0e0; text-align:center; border-radius:4px;">0.3</div>
<div style="padding:8px; background:#0f3460; color:#e0e0e0; text-align:center; border-radius:4px;">0.2</div>
<div style="padding:8px; background:#0f3460; color:#e0e0e0; text-align:center; border-radius:4px;">0.4</div>
</div>
<div style="color:#888; font-size:12px; margin-top:12px; font-family:sans-serif;">
Lower-triangular pattern after softmax. Blue cells: attended positions. Dark cells: masked (future tokens). Values shown are post-softmax attention weights (illustrative).
</div>
</div>

**Why causal masking matters beyond just "preventing cheating":** The mask enforces the autoregressive factorization from [[ch-01]] at the architectural level. Each position's representation is computed using only past context, which means the model's output at position $t$ is a valid estimate of $P(x_{t+1} \mid x_1, \ldots, x_t)$. This is what makes teacher forcing work — all $T$ positions can be trained in parallel because the causal mask ensures position $t$ never leaks information from position $t+1$ or beyond.

Without the causal mask, you'd have BERT — bidirectional but unable to generate text autoregressively, because each position's representation depends on the full sequence.

---

## 2. The GPT Lineage: Three Generations, Three Paradigms

The GPT series is not just a scaling story. Each generation introduced a fundamentally different paradigm for how language models are used. The architecture barely changed — the paradigm shifted completely.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:20px; font-family:sans-serif; font-weight:bold;">GPT Architecture Evolution: Same Design, Radical Paradigm Shifts</div>
<div style="display:flex; gap:12px; align-items:stretch; flex-wrap:wrap;">

<div style="flex:1; min-width:200px; background:#16213e; border-radius:10px; padding:16px; border-left:4px solid #e94560;">
<div style="color:#e94560; font-weight:bold; font-size:15px;">GPT-1 (2018)</div>
<div style="color:#aaa; font-size:13px; margin-top:8px;">117M params</div>
<div style="color:#aaa; font-size:13px;">12 layers, d=768</div>
<div style="color:#aaa; font-size:13px;">512 context</div>
<div style="color:#aaa; font-size:13px;">BooksCorpus (800M words)</div>
<div style="color:#e0e0e0; font-size:13px; margin-top:12px; padding-top:12px; border-top:1px solid #333;">
<strong style="color:#e94560;">Paradigm:</strong> Pre-train + Fine-tune<br/>
<span style="font-size:12px;">Task-specific heads, gradient updates per task</span>
</div>
</div>

<div style="flex:1; min-width:200px; background:#16213e; border-radius:10px; padding:16px; border-left:4px solid #e94560;">
<div style="color:#e94560; font-weight:bold; font-size:15px;">GPT-2 (2019)</div>
<div style="color:#aaa; font-size:13px; margin-top:8px;">1.5B params (13x)</div>
<div style="color:#aaa; font-size:13px;">48 layers, d=1600</div>
<div style="color:#aaa; font-size:13px;">1024 context (2x)</div>
<div style="color:#aaa; font-size:13px;">WebText (40GB, 8M pages)</div>
<div style="color:#e0e0e0; font-size:13px; margin-top:12px; padding-top:12px; border-top:1px solid #333;">
<strong style="color:#e94560;">Paradigm:</strong> Zero-shot Transfer<br/>
<span style="font-size:12px;">No fine-tuning. Task described in natural language.</span>
</div>
</div>

<div style="flex:1; min-width:200px; background:#16213e; border-radius:10px; padding:16px; border-left:4px solid #e94560;">
<div style="color:#e94560; font-weight:bold; font-size:15px;">GPT-3 (2020)</div>
<div style="color:#aaa; font-size:13px; margin-top:8px;">175B params (117x)</div>
<div style="color:#aaa; font-size:13px;">96 layers, d=12288</div>
<div style="color:#aaa; font-size:13px;">2048 context (2x)</div>
<div style="color:#aaa; font-size:13px;">300B tokens (filtered blend)</div>
<div style="color:#e0e0e0; font-size:13px; margin-top:12px; padding-top:12px; border-top:1px solid #333;">
<strong style="color:#e94560;">Paradigm:</strong> In-Context Learning<br/>
<span style="font-size:12px;">Few-shot examples in the prompt. No gradient updates.</span>
</div>
</div>

</div>
<div style="color:#888; font-size:12px; margin-top:16px; font-family:sans-serif; text-align:center;">
Architecture changes: pre-norm LayerNorm, byte-level BPE, residual scaling, sparse attention (GPT-3). The paradigm changes were far more consequential than the architectural ones.
</div>
</div>

### 2.1 GPT-1: Transfer Learning for NLP (2018)

**The bet:** A decoder-only Transformer, pre-trained with a language modeling objective on unlabeled text, can be fine-tuned for diverse NLP tasks with minimal architectural changes.

**Architecture:** 12 layers, 768-dimensional model, 12 attention heads, 512-token context window. 117M parameters total. Trained on BooksCorpus (~800M words, ~7,000 unpublished books). GELU activations, learned positional embeddings, BPE tokenization with ~40K merges.

**The key innovation was not the architecture — it was the recipe.** GPT-1 established the two-stage paradigm:

1. **Stage 1 (Pre-training):** Standard next-token prediction on BooksCorpus. The model learns general language representations.
2. **Stage 2 (Fine-tuning):** Add a task-specific linear head. Fine-tune the entire model on labeled data, using both the task loss and an auxiliary language modeling loss (weighted by $\lambda = 0.5$).

The auxiliary LM loss during fine-tuning is an underappreciated detail. It acts as a regularizer, preventing the model from forgetting its pre-trained representations during task adaptation. GPT-1's ablation (Table 5 in the paper) shows that removing the auxiliary LM loss hurts performance on most tasks.

**Task-aware input transformations:** Rather than designing task-specific architectures, GPT-1 reformats different task types into linear token sequences:
- **Classification:** `[start] text [delim] [extract]` — extract class from final token
- **Entailment:** `[start] premise [delim] hypothesis [extract]`
- **Similarity:** Run both orderings, add element-wise before classification
- **Multiple choice:** `[start] context [delim] answer_i [extract]` for each option

This approach — structuring diverse tasks as text sequences fed to a single architecture — foreshadowed the "prompt engineering" paradigm of GPT-3, even though GPT-1 still required gradient updates.

**Results:** SOTA on 9 of 12 NLP benchmarks. 8.9% absolute improvement on commonsense reasoning (Stories Cloze), 5.7% on question answering (RACE), 1.5% on textual entailment (MultiNLI).

### 2.2 GPT-2 ([[gpt-2|paper]]): Scale as Zero-Shot Transfer (2019)

**The bet:** If the model is large enough and trained on diverse enough data, fine-tuning becomes unnecessary. The model can perform tasks described in natural language without any gradient updates.

**Architecture changes from GPT-1:**
- **Layer normalization moved to input** of each sub-block (pre-norm), with an additional LayerNorm after the final self-attention block. This stabilizes training at depth.
- **Residual weight scaling:** Residual layer weights initialized to $1/\sqrt{N}$ where $N$ is the number of residual layers. Prevents signal explosion in deep networks.
- **Context window doubled:** 512 $\rightarrow$ 1024 tokens.
- **Byte-level BPE:** 50,257 token vocabulary, enabling open-vocabulary handling without unknown tokens.

The architecture is nearly identical to GPT-1 — the differences are all stability and scaling refinements, not new mechanisms.

**The data innovation was as important as the scaling.** WebText was curated by scraping all outbound links from Reddit posts with 3+ karma — a human-filtered quality signal. The result: 40GB of diverse, relatively high-quality text from 8 million web pages. This demonstrated that data curation matters as much as model size. GPT-2 still underfits WebText at 1.5B parameters, meaning there was more to learn from the data than the model could capture.

**Model sizes tested:** 117M (Small), 345M (Medium), 762M (Large), 1.5B (XL). Performance scaled log-linearly with parameter count across tasks — a critical empirical finding that pointed toward the scaling laws Kaplan et al. would formalize in 2020 ([[ch-10]]).

**Zero-shot results:** GPT-2 achieved SOTA on 7 of 8 language modeling benchmarks without any fine-tuning. On downstream tasks, zero-shot performance was competitive with baseline supervised systems on reading comprehension (CoQA: 55 F1, matching or exceeding 3 of 4 baselines). Translation and summarization results were weaker but non-trivial — the model had learned task patterns from the distribution of web text alone.

**The paradigm shift:** GPT-1 said "pre-train then fine-tune." GPT-2 said "maybe you don't need to fine-tune at all." This was a radical claim in 2019. BERT had just dominated NLP benchmarks with task-specific fine-tuning, and the field assumed fine-tuning was the path forward. GPT-2 suggested that scale could substitute for task-specific adaptation.

### 2.3 GPT-3: In-Context Learning Emerges (2020)

**The bet:** At sufficient scale, the model can learn new tasks at inference time from a few examples in the prompt, with no gradient updates.

**Scale:** 175 billion parameters. 96 layers, 12288 model dimension, 96 attention heads (128 dims per head). 2048 token context window. Trained on 300B tokens from a filtered blend:
- Common Crawl (410B tokens, 60% of training weight)
- WebText2 (19B tokens, 22% weight)
- Books1 (12B tokens, 8%)
- Books2 (55B tokens, 8%)
- Wikipedia (3B tokens, 3%)

Note the training mix weights do not match the dataset sizes — smaller, higher-quality datasets like WebText2 and Books are upsampled relative to Common Crawl. This reweighting is an early instance of data curriculum design.

**Training compute:** ~$3.64 \times 10^{23}$ FLOPs, estimated at thousands of V100 GPU-days. Batch size was gradually ramped from 32K tokens to 3.2M tokens during training — this batch size warmup stabilizes early training dynamics.

**The architectural novelty was minimal.** GPT-3 uses alternating dense and locally-banded sparse attention patterns (similar to Sparse Transformer), but the paper does not emphasize this. The architecture is fundamentally the same decoder-only Transformer as GPT-1 and GPT-2. The revolution was entirely about what happened when you scaled it 100x.

**In-context learning — the emergence story:** GPT-3 was evaluated across three settings:
- **Zero-shot:** Task description only, no examples
- **One-shot:** Task description + one example
- **Few-shot:** Task description + several examples (typically 10-100, fitting in the 2048-token context)

The defining finding (Figure 1.2 in the paper): **the gap between zero-shot and few-shot performance widens with model scale.** Small models (~125M) barely benefit from in-context examples. Large models (175B) show dramatic improvements from just a few demonstrations. This means in-context learning is an *emergent property of scale* — it does not exist at small scales and appears progressively as the model grows.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">In-Context Learning: Emergent with Scale</div>
<div style="display:flex; gap:16px; flex-wrap:wrap; justify-content:center;">

<div style="background:#16213e; border-radius:10px; padding:16px; min-width:180px; text-align:center;">
<div style="color:#888; font-size:12px;">125M params</div>
<div style="display:flex; gap:6px; justify-content:center; margin-top:8px;">
<div style="background:#0f3460; width:40px; height:30px; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#aaa; font-size:11px;">0s</div>
<div style="background:#0f3460; width:40px; height:32px; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#aaa; font-size:11px;">1s</div>
<div style="background:#0f3460; width:40px; height:33px; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#aaa; font-size:11px;">fs</div>
</div>
<div style="color:#666; font-size:11px; margin-top:6px;">Minimal gain from examples</div>
</div>

<div style="color:#e94560; font-size:24px; align-self:center;">...</div>

<div style="background:#16213e; border-radius:10px; padding:16px; min-width:180px; text-align:center;">
<div style="color:#888; font-size:12px;">13B params</div>
<div style="display:flex; gap:6px; justify-content:center; margin-top:8px;">
<div style="background:#0f3460; width:40px; height:30px; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#aaa; font-size:11px;">0s</div>
<div style="background:#0f3460; width:40px; height:42px; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#aaa; font-size:11px;">1s</div>
<div style="background:#0f3460; width:40px; height:50px; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#aaa; font-size:11px;">fs</div>
</div>
<div style="color:#666; font-size:11px; margin-top:6px;">Moderate gain</div>
</div>

<div style="color:#e94560; font-size:24px; align-self:center;">...</div>

<div style="background:#16213e; border-radius:10px; padding:16px; min-width:180px; text-align:center;">
<div style="color:#e94560; font-size:12px; font-weight:bold;">175B params</div>
<div style="display:flex; gap:6px; justify-content:center; margin-top:8px;">
<div style="background:#0f3460; width:40px; height:30px; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#aaa; font-size:11px;">0s</div>
<div style="background:#0f3460; width:40px; height:55px; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#aaa; font-size:11px;">1s</div>
<div style="background:#e94560; width:40px; height:75px; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:11px; font-weight:bold;">fs</div>
</div>
<div style="color:#e0e0e0; font-size:11px; margin-top:6px;">Dramatic gain from few-shot</div>
</div>

</div>
<div style="color:#888; font-size:12px; margin-top:16px; font-family:sans-serif; text-align:center;">
0s = zero-shot, 1s = one-shot, fs = few-shot. Bar heights illustrate the widening performance gap with scale.<br/>
The few-shot advantage is an emergent property — it appears only at large scale and increases with model size.
</div>
</div>

**Benchmark highlights:**
- Few-shot GPT-3 175B matched or exceeded fine-tuned SOTA on several benchmarks — without any gradient updates
- On LAMBADA (word prediction from context), few-shot 175B achieved 86.4% accuracy, surpassing the prior fine-tuned SOTA of 68.0%
- On TriviaQA, few-shot 175B achieved 71.2% accuracy vs. fine-tuned SOTA of 68.0%
- On 3-digit arithmetic (e.g., "What is 456 + 789?"), few-shot 175B achieved ~100% accuracy — a task requiring emergent multi-step reasoning

**What in-context learning is NOT:** ICL is not gradient-based learning. The model's weights do not update during inference. The mechanism is better understood as the model's attention layers dynamically routing information from the in-context examples to inform predictions. Recent work has shown that Transformer attention heads can implement mesa-optimization — effectively running a learning algorithm in the forward pass. This is a deep connection to [[ch-10]] (scaling and emergent abilities).

---

## 3. Why Decoder-Only Won

This is the most consequential architectural question of the LLM era. Between 2018 and 2020, three architectures competed: encoder-only (BERT), encoder-decoder (T5, BART), and decoder-only (GPT). By 2023, decoder-only had won so thoroughly that no frontier lab was training encoder-only or encoder-decoder models for general-purpose use. Why?

The answer is not a single factor but a convergence of five advantages:

### 3.1 Training Efficiency: 100% vs. 15% Signal

As discussed in [[ch-01]], autoregressive models compute loss on every token in the sequence. BERT's masked language modeling computes loss on ~15% of tokens (those randomly selected for masking). For the same compute budget, an autoregressive model gets approximately $6.7\times$ more gradient signal per sequence. At the scale of hundreds of billions of tokens, this efficiency difference is decisive.

T5 (encoder-decoder) tried to bridge this gap with a "span corruption" objective that masks contiguous spans rather than individual tokens, but it still wastes compute on the uncorrupted encoder input that provides no direct training signal.

### 3.2 Architectural Simplicity: One Stack, Not Two

An encoder-decoder model has two distinct stacks of Transformer layers plus cross-attention layers connecting them. A decoder-only model has one stack. This simplicity pays dividends in three ways:

1. **Parameter efficiency:** All parameters serve a single computation graph. In an encoder-decoder, the encoder parameters are idle during generation and the decoder parameters are idle during encoding.
2. **Implementation simplicity:** One forward pass, one set of attention patterns, one KV cache. Serving infrastructure is simpler ([[ch-25]]).
3. **Hyperparameter optimization:** One depth to tune, one width to tune. Fewer architectural degrees of freedom means the scaling law search space is smaller.

### 3.3 Scaling Predictability

The Kaplan scaling laws ([[ch-10]]) showed that decoder-only autoregressive models follow clean power-law relationships between compute, parameters, data, and loss. This predictability is not just an academic curiosity — it is an operational advantage. Training a 175B model costs millions of dollars. Before committing, you need to predict the final loss from smaller pilot runs. Decoder-only models make this prediction reliable.

Encoder-decoder models showed less predictable scaling behavior, partly because the interaction between encoder and decoder depth introduces additional complexity in the scaling relationship.

### 3.4 Unified Input-Output Format

A decoder-only model treats everything as a sequence of tokens: input and output are in the same space, processed by the same layers. This means:

- **Any task is a sequence completion task.** Translation, summarization, QA, code generation, reasoning — all are expressed as "given this prefix, continue." No task-specific heads, no separate encoder for the source.
- **Multi-task training is free.** You don't need to design separate input formats for different tasks (though you still benefit from prompt templates).
- **Prompting works.** The model's entire interface is "tokens in, tokens out." This enabled the in-context learning paradigm that made GPT-3 transformative.

Encoder-decoder models need to decide what goes in the encoder vs. the decoder. This is fine for well-defined tasks (source sentence in encoder, target in decoder) but becomes awkward for open-ended generation, dialogue, and reasoning where the boundary between "input" and "output" is fluid.

### 3.5 KV Cache Efficiency at Inference

During autoregressive generation, a decoder-only model caches the key-value pairs from all previous tokens (the KV cache, [[ch-25]]). Each new token only computes attention against its own query and the cached keys/values — no recomputation.

An encoder-decoder model must first run the full encoder on the input, cache those representations, then run the decoder autoregressively while attending to both the cached encoder outputs (via cross-attention) and the growing decoder KV cache. The cross-attention adds latency and memory overhead at every decoding step.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Encoder-Decoder vs. Decoder-Only: Structural Comparison</div>

<div style="display:flex; gap:24px; flex-wrap:wrap; justify-content:center;">

<div style="flex:1; min-width:240px; max-width:320px;">
<div style="color:#e94560; font-weight:bold; text-align:center; margin-bottom:12px;">Encoder-Decoder (T5)</div>
<div style="background:#16213e; border-radius:8px; padding:12px; margin-bottom:8px;">
<div style="color:#aaa; font-size:12px; text-align:center; margin-bottom:8px;">ENCODER</div>
<div style="display:flex; flex-direction:column; gap:4px;">
<div style="background:#0f3460; padding:6px; border-radius:4px; color:#888; font-size:11px; text-align:center;">Bidirectional Self-Attention</div>
<div style="background:#0f3460; padding:6px; border-radius:4px; color:#888; font-size:11px; text-align:center;">FFN</div>
<div style="color:#555; text-align:center; font-size:10px;">x N layers</div>
</div>
</div>
<div style="text-align:center; color:#e94560; font-size:18px; margin:4px 0;">|</div>
<div style="background:#16213e; border-radius:8px; padding:12px;">
<div style="color:#aaa; font-size:12px; text-align:center; margin-bottom:8px;">DECODER</div>
<div style="display:flex; flex-direction:column; gap:4px;">
<div style="background:#0f3460; padding:6px; border-radius:4px; color:#888; font-size:11px; text-align:center;">Causal Self-Attention</div>
<div style="background:#e94560; padding:6px; border-radius:4px; color:#fff; font-size:11px; text-align:center; font-weight:bold;">Cross-Attention (encoder output)</div>
<div style="background:#0f3460; padding:6px; border-radius:4px; color:#888; font-size:11px; text-align:center;">FFN</div>
<div style="color:#555; text-align:center; font-size:10px;">x N layers</div>
</div>
</div>
<div style="color:#888; font-size:11px; text-align:center; margin-top:8px;">3 attention types, 2 stacks, cross-attention overhead</div>
</div>

<div style="flex:1; min-width:240px; max-width:320px;">
<div style="color:#e94560; font-weight:bold; text-align:center; margin-bottom:12px;">Decoder-Only (GPT)</div>
<div style="background:#16213e; border-radius:8px; padding:12px;">
<div style="color:#aaa; font-size:12px; text-align:center; margin-bottom:8px;">SINGLE STACK</div>
<div style="display:flex; flex-direction:column; gap:4px;">
<div style="background:#0f3460; padding:6px; border-radius:4px; color:#888; font-size:11px; text-align:center;">Causal Self-Attention</div>
<div style="background:#0f3460; padding:6px; border-radius:4px; color:#888; font-size:11px; text-align:center;">FFN</div>
<div style="color:#555; text-align:center; font-size:10px;">x N layers</div>
</div>
</div>
<div style="color:#888; font-size:11px; text-align:center; margin-top:8px;">1 attention type, 1 stack, simpler serving</div>
</div>

</div>
</div>

### 3.6 When Encoder-Decoder Still Wins

The decoder-only dominance has exceptions. Encoder-decoder architectures retain advantages for tasks with a clear, fixed-length source that the model needs to deeply understand before generating output:

- **Machine translation** with long source sentences: The encoder can attend bidirectionally to the full source, building a richer representation than a decoder-only model that processes the source left-to-right as a prefix.
- **Extractive QA and information retrieval:** When the task is "find the answer in this passage" rather than "generate an answer," bidirectional encoding of the passage is more natural.
- **Speech-to-text and vision-to-text:** Modalities where the input (audio spectrogram, image patches) has a fundamentally different structure from the output (text). Cross-attention between modality-specific encoders and a text decoder is a natural fit (see [[ch-17]]).

But for **open-ended generation, reasoning, dialogue, and instruction-following** — the tasks that drive modern LLM development — the decoder-only architecture's unified format, prompting capability, and scaling behavior dominate.

---

## 4. The Paradigm Shift: Pre-Train/Fine-Tune to Prompt-Based

The GPT lineage maps directly onto a paradigm transition that reshaped the entire field:

**GPT-1 era (2018-2019): Pre-train + Fine-tune.**
Train a general model, then fine-tune on each downstream task with labeled data. Every task requires gradient updates, task-specific data, and (often) a task-specific output head. BERT refined this paradigm and dominated NLP benchmarks.

**GPT-2 era (2019-2020): Pre-train + Zero-shot.**
The model performs tasks by generating text conditioned on a natural-language description: "TL;DR:" for summarization, "Q: ... A:" for QA. No gradient updates required, but zero-shot performance was inconsistent — it worked on some tasks, failed on others. The paradigm was suggestive but not yet reliable.

**GPT-3 era (2020+): Pre-train + Prompt (few-shot).**
Provide a few examples of the desired input-output mapping in the prompt. The model infers the pattern and generalizes. No gradient updates, no task-specific architecture. This is **in-context learning**, and it changed how the field thinks about model capabilities.

**Architectural implications of this shift:**

1. **Context window becomes paramount.** If tasks are solved via in-context examples rather than fine-tuning, you need a long enough context to fit those examples. This drives the context window arms race: 512 (GPT-1) $\rightarrow$ 1024 (GPT-2) $\rightarrow$ 2048 (GPT-3) $\rightarrow$ 128K+ (modern models, [[ch-16]]).

2. **Model size directly determines capability.** In the fine-tuning paradigm, even small models can be task-specialized. In the prompting paradigm, the model must be large enough for in-context learning to emerge. This creates an economic moat: only organizations that can train models at 100B+ scale can access the prompting paradigm.

3. **The model-task interface simplifies radically.** Instead of designing task-specific heads and training pipelines for each task, the interface is just text. This enabled the "foundation model" concept: one model, deployed via API, adapted to thousands of tasks through prompting alone.

4. **Fine-tuning doesn't disappear — it shifts.** Post-GPT-3, fine-tuning evolved from task-specific adaptation (GPT-1/BERT style) into alignment training: SFT, RLHF, DPO ([[ch-12]]). The goal shifted from "teach the model to do task X" to "teach the model to follow instructions and be helpful."

---

## 5. Architectural Details That Compound: GPT-1 to GPT-3

Beyond the paradigm shifts, several architectural refinements accumulated across the GPT lineage. Individually minor, they compound to enable stable training at 175B scale.

| Detail | GPT-1 | GPT-2 | GPT-3 |
|--------|-------|-------|-------|
| **LayerNorm position** | Post-norm | Pre-norm (input of each block) | Pre-norm |
| **Residual init scaling** | Standard | $1/\sqrt{N}$ for residual layers | $1/\sqrt{N}$ |
| **Tokenization** | BPE (~40K) | Byte-level BPE (50,257) | Byte-level BPE (50,257) |
| **Context window** | 512 | 1024 | 2048 |
| **Positional encoding** | Learned | Learned | Learned |
| **Activation** | GELU | GELU | GELU |
| **Attention pattern** | Dense | Dense | Alternating dense + sparse |
| **Batch size schedule** | Fixed (64) | Fixed (512) | Ramped (32K $\rightarrow$ 3.2M tokens) |

**Pre-norm vs. post-norm:** GPT-1 used the original Transformer's post-norm placement (LayerNorm after the residual addition). GPT-2 moved LayerNorm to the input of each sub-block. Pre-norm enables more stable gradients in deep networks because the residual pathway is unobstructed — the signal flows directly through addition without passing through normalization. This change was essential for scaling to 48+ layers.

**Byte-level BPE:** GPT-1 used a standard BPE vocabulary. GPT-2 introduced byte-level BPE, which operates on raw bytes rather than Unicode characters. The advantage: any text can be encoded without "unknown token" failures. The 50,257 vocabulary size was retained through GPT-3.

**Batch size warmup (GPT-3):** Starting with small batches and gradually increasing is motivated by the observation that early in training, the loss landscape is noisy and large batches waste compute on unreliable gradient estimates. Small batches provide more parameter updates per token, while large batches are more compute-efficient once the loss landscape stabilizes.

---

## Core Insights from the Literature

### Insight 1: The fine-tuning paradigm was a local optimum
**Paper:** Radford et al., "Improving Language Understanding by Generative Pre-Training" ([[gpt-1|GPT-1, 2018 (paper)]])

GPT-1 established pre-train + fine-tune as the dominant NLP paradigm, and BERT perfected it. But this was a local optimum in methodology space. The input transformations in GPT-1's Figure 1 — restructuring classification, entailment, and QA as token sequences — already contained the seed of the prompting paradigm. The field spent 2019-2020 fine-tuning BERT variants, but the decoder-only lineage was converging on eliminating fine-tuning entirely. **Guideline:** When a dominant paradigm requires increasingly elaborate engineering (task-specific heads, data collection, hyperparameter tuning per task), look for whether the paradigm itself can be replaced by scale.

### Insight 2: Data curation is an architectural decision
**Paper:** Radford et al., "Language Models are Unsupervised Multitask Learners" ([[gpt-2|GPT-2, 2019 (paper)]])

WebText's curation strategy — filtering by Reddit karma — was as important as doubling the model size. GPT-2 at 1.5B still underfits WebText, meaning the data contained more learnable signal than the model could absorb. The dataset quality ceiling constrained GPT-1 (BooksCorpus is relatively narrow), and GPT-3's careful reweighting of Common Crawl vs. curated sources confirms the pattern. **Guideline:** Treat data curation as a first-class design decision, on par with architecture choices. A mediocre model on excellent data will outperform an excellent model on mediocre data.

### Insight 3: In-context learning is an emergent capability, not a designed feature
**Paper:** Brown et al., "Language Models are Few-Shot Learners" ([[gpt-3|GPT-3, 2020 (paper)]])

Nobody designed GPT-3 to do in-context learning. The architecture is the same decoder-only Transformer from GPT-1, just larger. ICL emerged from scale — it's a property that appears when models cross a threshold in parameter count and training data diversity. Figure 1.2 in the paper shows this clearly: the few-shot advantage over zero-shot *widens* with scale. This means ICL is not a capability you can add to a small model; it's a phase transition in what models can do. **Guideline:** When evaluating whether a capability will emerge at larger scale, look for tasks where performance already improves disproportionately with model size. If the scaling curve is superlinear, the capability may strengthen dramatically at the next order of magnitude.

### Insight 4: Bidirectional context is a representational advantage that loses to training efficiency at scale
**Paper:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" ([[bert|2018 (paper)]])

BERT's ablation (Table 5) proved that bidirectional context produces strictly richer representations than left-to-right language modeling. Removing bidirectionality degraded performance on every benchmark. But BERT's masked LM objective trains on only 15% of tokens per sequence, while autoregressive models train on 100%. At GPT-3 scale (300B training tokens), this efficiency gap means the autoregressive model sees $\sim$20x more effective training signal for the same compute. The representational advantage of bidirectional context is real but bounded; the training efficiency advantage of autoregressive modeling compounds with scale. **Guideline:** When two approaches have different per-token quality vs. per-sequence efficiency tradeoffs, bet on efficiency at scale. The approach that extracts more learning signal per FLOP will eventually win.

### Insight 5: The embedding-output weight tying trick reveals a deep structural symmetry
**Paper:** Alammar, "The Illustrated GPT-2" ([[alammar-illustrated-gpt2|blog]])

GPT-2 reuses the input embedding matrix (transposed) as the output projection that converts final-layer representations into vocabulary logits. This is not just a parameter-efficiency trick — it reflects a structural insight: the model's input space and output space are the same (both are distributions over the vocabulary). Weight tying ensures that the model's "understanding" of a token as input is consistent with its "prediction" of that token as output. This design persists through modern models and reduces parameter count significantly (the embedding matrix for a 50K vocabulary with 1600-dimensional embeddings is ~80M parameters). **Guideline:** When input and output spaces share structure, tying their representations is a strong prior that reduces parameters and improves generalization.

---

## Key Takeaways

1. **Causal masking is the architectural enforcement of autoregression.** It's a single matrix applied before softmax that makes every position's representation depend only on past context, enabling parallel training via teacher forcing.

2. **The GPT lineage is a paradigm story, not just a scaling story.** GPT-1 introduced pre-train + fine-tune, GPT-2 demonstrated zero-shot transfer, GPT-3 established in-context learning. The architecture barely changed; the paradigm transformed completely.

3. **Decoder-only won through a convergence of five advantages:** training signal efficiency (100% vs 15%), architectural simplicity (one stack), scaling predictability (clean power laws), unified input-output format (everything is text completion), and inference efficiency (single KV cache).

4. **In-context learning is emergent, not designed.** It appears at scale as a phase transition. The gap between zero-shot and few-shot widens with model size, meaning larger models extract more from in-context examples.

5. **Data curation was as transformative as scaling.** WebText's quality-filtered web data (GPT-2) and GPT-3's careful source reweighting were prerequisite to the capability gains, not just the parameter count.

6. **Encoder-decoder is not dead, just specialized.** For tasks with a clear source-target structure (translation, cross-modal tasks), encoder-decoder retains advantages. But for general-purpose generation and reasoning, decoder-only dominates.

7. **The paradigm shift from fine-tuning to prompting has deep architectural implications:** longer context windows, larger minimum viable model size, and a redefinition of "fine-tuning" from task adaptation to alignment training ([[ch-12]]).

---

## References

- Radford et al. "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018) — [[gpt-1|paper]]
- Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019) — [[gpt-2|paper]]
- Brown et al. "Language Models are Few-Shot Learners" (GPT-3, 2020) — [[gpt-3|paper]]
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018) — [[bert|paper]]
- Alammar, "The Illustrated GPT-2: Visualizing Transformer Language Models" — [[alammar-illustrated-gpt2|blog]]
- Raffel et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, 2020)
- Kaplan et al. "Scaling Laws for Neural Language Models" (2020) — see [[ch-10]]
- Vaswani et al. "Attention Is All You Need" (2017) — see [[ch-03]]
