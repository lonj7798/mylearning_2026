# Excerpt: Llama 4 Early Fusion and Native Multimodality

<!-- source: [[llama-4|report]] — Meta AI, 2025 -->

## The Shift from Llama 3 to Llama 4

The Llama family underwent a fundamental architectural shift in its multimodal approach between generations:

| Dimension | Llama 3 (2024) | Llama 4 (2025) |
|-----------|---------------|----------------|
| Fusion type | Late (adapter-based) | Early (unified backbone) |
| LLM backbone | Dense, 405B | MoE, 109B total / 17B active |
| Multimodal data in pretraining | No | Yes (30T+ tokens incl. image/video) |
| Vision encoder | Post-hoc adapter | MetaCLIP, integrated |
| Cross-modal depth | Adapter capacity only | Full backbone capacity |

Llama 3 described its approach as "compositional multimodal integration" — vision, speech, and tool-use capabilities "added post-hoc through adapter-based approaches, keeping the language backbone frozen or lightly fine-tuned." This was pragmatic: the 405B dense model was Meta's most expensive training run, and retraining it for multimodality was not economically justified.

Llama 4 committed to "early fusion multimodality" — text and vision tokens "processed jointly in a unified backbone from the start." Two factors enabled this shift:

1. **MoE architecture:** With only 17B active parameters per token (despite 109B total for Scout), multimodal pretraining cost per token is comparable to training a much smaller dense model. MoE made joint pretraining economically feasible.
2. **Data scale:** Over 30 trillion tokens of text, image, and video data. Native multimodal training is only worthwhile when you have enough multimodal data to shape the representations meaningfully.

## MetaCLIP Vision Encoder

Llama 4 uses a vision encoder based on MetaCLIP — Meta's open-source reimplementation of CLIP that emphasizes data curation methodology:

- Trained separately from the language backbone using contrastive learning
- Produces visual features aligned with language representations
- Adapted for use with the Llama 4 backbone during multimodal pretraining

The encoder is trained separately but the backbone learns to consume its features during joint pretraining — a middle ground between training the vision encoder from scratch (wasteful) and freezing the backbone during adaptation (limiting).

## Early Fusion Architecture

In Llama 4's early fusion design:

1. MetaCLIP encodes the image into visual feature vectors
2. Visual features are projected into the LLM's model dimension
3. Projected visual tokens are **interleaved with text tokens before the first Transformer layer**
4. The unified MoE backbone processes the combined sequence through all layers

Every layer — every expert — sees both modalities. This means:
- **Self-attention** operates over the joint visual+textual sequence in every layer
- **Expert routing** in MoE layers can specialize: some experts may become vision-dominant, others text-dominant, others cross-modal
- **No adapter bottleneck** — cross-modal reasoning uses the full 17B active parameters, not a small projection layer

## MoE and Multimodality Synergy

The MoE architecture has a specific synergy with multimodality that deserves attention:

**Expert specialization by modality:** With 16 experts (Scout) or 128 experts (Maverick), routing can naturally allocate different experts to different modalities. A text token describing spatial relationships might route to a different expert than a visual token representing the spatial layout. This specialization happens automatically through the routing mechanism — no explicit modality-based routing rules are needed.

**Maverick's 128 experts:** The extreme expert count (128 routed experts, each token uses 1 shared + 1 routed) provides $65\times$ more expert combinations than typical 8-16 expert models. For multimodal inputs, this means finer-grained specialization: instead of "vision expert" vs "text expert," the model can develop experts for "spatial reasoning about images," "reading text in images," "connecting visual attributes to linguistic descriptions," etc.

## Scout: 10M Token Context with Multimodality

Scout's 10M token context (enabled by iRoPE) has direct implications for multimodal processing. At 10M tokens, a single context window can hold:

- Hundreds of images (at ~256-512 visual tokens each)
- A full video (thousands of frames sampled at visual token resolution)
- Multi-page documents with embedded images

The combination of early fusion + extreme context + MoE means Scout can perform tasks that require reasoning across many images simultaneously — image comparison, multi-document visual QA, long video understanding — that are infeasible for models with shorter contexts or late-fusion architectures.

## Performance: What Early Fusion Buys

The Llama 4 report positions Scout as "best multimodal model in its class," outperforming Gemma 3 and Gemini 2.0 Flash-Lite on multimodal benchmarks. Maverick "beats GPT-4o and Gemini 2.0 Flash across broad benchmarks."

The multimodal performance improvement from Llama 3 to Llama 4 comes from three simultaneous changes (architecture, data, training), making it difficult to isolate the contribution of early fusion alone. But the report explicitly frames early fusion as enabling "richer cross-modal interaction" and "joint pre-training with unlabeled multimodal data" — capabilities that late fusion structurally cannot provide.

## Design Implications

The Llama 3-to-4 transition clarifies when each approach is appropriate:

- **Late fusion** is correct when: you have an existing, expensive LLM you want to augment; multimodal data is limited; text quality must not degrade; budget constraints prevent retraining
- **Early fusion** is correct when: multimodal reasoning is a first-class objective; sufficient multimodal pretraining data exists; the training budget permits joint pretraining; MoE makes per-token compute feasible

The shift from dense to MoE was arguably the *enabler* of early fusion. A dense 405B model is too expensive to retrain with multimodal data. A MoE model with 17B active parameters costs roughly the same per-token as a 17B dense model — making multimodal pretraining economically viable despite the larger total parameter count.
