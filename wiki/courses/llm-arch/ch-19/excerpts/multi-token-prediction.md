# Excerpt: Multi-Token Prediction (MTP) in DeepSeek-V3

<!-- source: [[deepseek-v3|report]], Section 2.3 -->

## Motivation: Denser Training Signal

Standard next-token prediction provides one gradient signal per position: given context $t_1, \ldots, t_i$, predict $t_{i+1}$. The hidden state at position $i$ encodes information about the entire past context, but this rich representation is "consumed" to predict only a single token. The remaining information about future tokens $t_{i+2}, t_{i+3}, \ldots$ goes unused during training.

Multi-token prediction (MTP) extracts additional signal from each forward pass by predicting $D$ future tokens simultaneously:

$$\mathcal{L} = \mathcal{L}_\text{main}(t_{i+1}) + \lambda \sum_{d=1}^{D} \mathcal{L}_\text{MTP}^{(d)}(t_{i+d+1})$$

The weight $\lambda$ controls the contribution of MTP losses. The main next-token prediction loss remains the primary objective.

## Architecture: Sequential MTP Modules

DeepSeek-V3 uses $D$ lightweight MTP modules, each predicting one additional future token. The modules form a **causal chain**:

```
Main model output h_i  ──► MTP Module 1 ──► predict t_{i+2}
                                │
                     (h_i, embed(t_{i+1})) ──► MTP Module 2 ──► predict t_{i+3}
                                                      │
                                           (output_1, embed(t_{i+2})) ──► ...
```

Each module $d$ receives:
1. The output of module $d-1$ (or the main model for $d=1$)
2. The embedding of the token that module $d-1$ was predicting

This causal dependency is critical. Without it, module $d$ would predict $t_{i+d+1}$ given only the original context — without knowing what the model expects tokens $t_{i+2}$ through $t_{i+d}$ to be. The causal chain gives each module the "planned" intermediate tokens, making the prediction task well-defined.

## Parameter Sharing

The MTP modules share two components with the main model:
- **Embedding matrix:** same token embeddings used everywhere
- **Output head:** same vocabulary projection for prediction

This sharing serves two purposes:
1. **Parameter efficiency:** no separate vocabulary projection per MTP module
2. **Representation consistency:** the same token representations are used for both producing and consuming token predictions across the chain

Each MTP module has its own small set of parameters (a lightweight Transformer layer or MLP), but these are far smaller than the main model's layers.

## Why MTP Improves Quality

The quality benefit of MTP is not just "more gradient signal." Predicting multiple future tokens forces the model to build representations that are **less myopic**:

- **Next-token prediction** can succeed by learning local statistical patterns (bigrams, trigrams, common continuations). The model can "get away with" shallow features.
- **Multi-token prediction** requires understanding the *trajectory* of the text — where the argument is heading, what structure is being built, what constraints will apply several tokens ahead. This forces deeper semantic representations.

This is analogous to how multi-step lookahead in chess forces a player to evaluate positions more deeply than single-move evaluation.

## MTP for Speculative Decoding at Inference

During inference, standard autoregressive decoding generates one token per forward pass. Speculative decoding uses a small "draft" model to propose multiple candidate tokens, which the main model then verifies in parallel.

DeepSeek-V3's MTP modules serve as **native draft models**: they were trained to predict the exact tokens the main model would generate, using the same representations. This makes them well-calibrated drafts without the distribution mismatch that afflicts external draft models.

The process:
1. MTP modules propose $D$ candidate tokens
2. The main model verifies all candidates in a single forward pass
3. Accept the longest prefix of correct candidates; regenerate from the first incorrect one

The report claims a **1.8x inference speedup** from MTP-based speculative decoding. This is significant because it comes "for free" — the MTP modules were already trained; they're simply repurposed at inference time.

## Design Choice: Training-Only vs Inference-Time MTP

The MTP modules can be **optionally discarded** after training. If inference latency is not a concern (e.g., batch processing), you can drop the MTP modules entirely and use only the main model for standard autoregressive decoding. The quality benefits of MTP are baked into the main model's representations — the modules were only needed to provide the additional training signal.

This optionality is a key design feature. It means MTP incurs zero inference cost if you don't want speculative decoding, while providing both quality improvement (via better representations) and speed improvement (via speculative decoding) when the modules are retained.

## Connection to Training Signal Density

MTP relates to a broader theme in efficient training: **extracting more learning per forward pass**. Other approaches to denser training signal include:
- Masked language modeling (BERT predicts ~15% of tokens vs. next-token's one)
- Auxiliary tasks (sentence order prediction, contrastive learning)
- Data augmentation and replay

MTP is arguably the cleanest approach for autoregressive models because it preserves the left-to-right causal structure while predicting multiple positions. It doesn't require masking, doesn't change the generation paradigm, and directly improves the representations used for the model's actual task.
