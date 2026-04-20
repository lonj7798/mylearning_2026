<!-- scope: next-token prediction training objective
     deps: [[ch-02]]
     see-also: [[bendersky-cross-entropy]]
-->

# How Does Next-Token Prediction Train an LLM?

- **Core Insight:** Next-token prediction creates self-supervised signal from raw text.
- **Guideline:** The training objective shapes what the model can learn; understand the objective before the architecture.

- **Author:** Sebastian Raschka, PhD
- **URL:** https://sebastianraschka.com/faq/docs/next-token-prediction.html
- **Relevant chapters:** Training fundamentals, loss functions, language modeling

## Summary
A concise explainer on how next-token prediction transforms ordinary text into a supervised learning problem. Raschka explains the input-target structure, loss computation via cross-entropy, and the critical distinction between training (parallel loss computation) and generation (sequential token production).

## Key Content

### Core Concept

Next-token prediction transforms ordinary text into a supervised learning problem. As Raschka explains, "the label for each position is simply the token that comes next," making this approach self-supervised since targets derive directly from the text rather than manual annotations.

### Training Mechanism

**Input-Target Structure:**
The training process operates on shifted sequences. For any text tokenized into token IDs, the model receives tokens up to position *t* as input, with the target being the token at position *t+1*. This creates parallel training across multiple positions within a single sequence, enabling efficient batch learning.

**Model Processing:**
A GPT-style model processes input tokens and generates logits vectors — one per position. Each vector contains scores for every vocabulary token, representing the model's prediction of the next token. Critically, causal attention restricts predictions: "the prediction at each position can only depend on the current token and earlier tokens."

**Loss Computation:**
1. Logits convert to probabilities through softmax
2. The model receives a penalty when assigning low probability to the actual next token
3. Cross-entropy loss (equivalent to negative log likelihood) quantifies this penalty
4. Backpropagation adjusts weights so correct continuations gain higher probability in similar future contexts

### Learning Outcomes

Through repeated exposure across batches and documents, the model captures syntax, semantics, style, factual associations, and long-range language patterns — without explicit rule programming.

### Training vs. Generation

A key distinction:
- **Training:** Computes losses for many positions simultaneously within sequences (teacher forcing — all ground truth tokens available)
- **Generation:** Produces one token iteratively, appending each prediction to context before repeating (autoregressive)

## Notable Insights
- The elegance of next-token prediction is that it requires zero human annotation — the text itself provides all supervision signals.
- Causal attention masking is what enforces the autoregressive property during training, ensuring the model cannot "cheat" by looking ahead.
- The same forward pass during training processes all positions in parallel, making it far more efficient than generating tokens one at a time.
