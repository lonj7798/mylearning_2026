<!-- scope: when encoder-decoder beats decoder-only and vice versa, structural analysis, parent: [[ch-04]] -->

# Encoder-Decoder vs. Decoder-Only: When Each Architecture Wins

The decoder-only architecture dominates frontier LLMs, but the encoder-decoder design retains genuine advantages for specific task structures. This excerpt provides a rigorous analysis of when each architecture is the better choice, grounded in the structural properties of attention, training efficiency, and task geometry.

---

## The Structural Difference

An encoder-decoder model has two distinct computation stages:

1. **Encoder**: Processes the source sequence with **bidirectional** self-attention. Every token attends to every other token. The output is a set of contextualized representations.
2. **Decoder**: Generates the target sequence autoregressively with **causal** self-attention, plus **cross-attention** to the encoder's representations.

A decoder-only model has one computation stage:

1. **Single stack**: Processes both "source" and "target" as a single concatenated sequence with causal self-attention throughout. The source is a prefix; the target is generated token-by-token after it.

The key difference is how the model processes the source: **bidirectionally (encoder) vs. left-to-right (decoder-only prefix)**.

---

## When Encoder-Decoder Wins

### 1. Fixed Source, Variable Target (Translation, Summarization)

When the task has a clear, fixed-length source that must be deeply understood before generation begins, bidirectional encoding provides richer source representations. In translation, the encoder can resolve ambiguities that require full-sentence context:

- "The bank was flooded" -- is this a river bank or a financial institution? The encoder sees the entire sentence bidirectionally and can use downstream context to disambiguate. A decoder-only model processing this as a prefix sees "The bank was" before it sees "flooded."

The BERT ablation (Table 5 in [[bert|paper]]) proved that bidirectional context produces strictly richer representations than left-to-right context on every benchmark tested.

### 2. Cross-Modal Tasks (Vision-to-Text, Speech-to-Text)

When the input modality (images, audio) has fundamentally different structure from the output modality (text), a modality-specific encoder paired with a text decoder is a natural fit. The encoder can use architecture optimized for the input modality (ViT for images, Conformer for audio), while the decoder handles text generation. Cross-attention bridges the modality gap.

Examples: Whisper (speech-to-text), Flamingo (image-to-text), PaLI (vision-language).

### 3. Extractive Tasks (QA, Information Retrieval)

When the task is "find the answer span in this passage" rather than "generate an answer," bidirectional encoding of the passage allows each token's representation to incorporate full-document context. The decoder-only model must process the passage left-to-right as a prefix, meaning early tokens have limited context about what comes later.

---

## When Decoder-Only Wins

### 1. Open-Ended Generation and Reasoning

When the boundary between "input" and "output" is fluid -- dialogue, chain-of-thought reasoning, creative writing -- the encoder-decoder split becomes an awkward forced division. What goes in the encoder vs. the decoder? In a multi-turn conversation, each turn is both "source" (context for the next turn) and "target" (generated text). The decoder-only model handles this naturally: everything is a single sequence.

### 2. Training Efficiency at Scale

As discussed in [[ch-04]], autoregressive models compute loss on 100% of tokens. BERT-style masked language modeling computes loss on ~15%. T5's span corruption is better but still wastes encoder compute on uncorrupted tokens that provide no direct training signal. At 300B+ tokens of training, this efficiency gap is decisive.

### 3. Scaling Predictability

The Kaplan scaling laws ([[ch-10]]) were established on decoder-only autoregressive models. These models follow clean power-law relationships between compute, parameters, data, and loss. Encoder-decoder models have more complex scaling behavior because the interaction between encoder depth, decoder depth, and cross-attention creates additional degrees of freedom.

### 4. In-Context Learning

In-context learning ([[gpt-3|paper]]) works because the model treats few-shot examples as part of its input prefix. This is natural for decoder-only models where input and output share the same token stream. Encoder-decoder models have no natural way to process in-context examples -- do they go in the encoder? The decoder? Both?

### 5. Serving Simplicity

One forward pass, one KV cache, one attention pattern. The decoder-only model's inference stack is simpler, cheaper, and better optimized in production serving frameworks (vLLM, TensorRT-LLM, llama.cpp). Encoder-decoder models require managing encoder representations, cross-attention caches, and two distinct computation stages.

---

## The Unification Argument

The deeper reason decoder-only won is that **any encoder-decoder task can be reformulated as a decoder-only task** by concatenating source and target into a single sequence. The reverse is not true -- open-ended generation cannot be naturally split into an encoder and decoder stage.

Decoder-only is the more general architecture. Encoder-decoder is a specialization that trades generality for richer source representations. At sufficient scale, the decoder-only model's left-to-right processing of the source captures enough context that the bidirectional advantage shrinks below the efficiency advantage.

---

## Decision Framework

| Factor | Favors Encoder-Decoder | Favors Decoder-Only |
|--------|----------------------|-------------------|
| Source structure | Fixed, well-defined source | Fluid input-output boundary |
| Source understanding | Deep comprehension needed | Surface-level suffices |
| Cross-modal | Different input/output modalities | Same modality (text-to-text) |
| Task type | Extractive, structured output | Generative, open-ended |
| Scale | Sub-10B models | 10B+ models |
| In-context learning | Not needed | Critical capability |
| Training data | Limited, task-specific | Large-scale unsupervised |
| Serving | Latency-tolerant batch | Low-latency interactive |

---

## References

- [[bert|Devlin et al. "BERT" (2018) (paper)]] -- bidirectional representation advantage
- [[gpt-1|Radford et al. "GPT-1" (2018) (paper)]] -- decoder-only pre-train + fine-tune
- [[gpt-3|Brown et al. "GPT-3" (2020) (paper)]] -- in-context learning paradigm
- Raffel et al. "T5" (2020) -- encoder-decoder scaling
