<!-- scope: perplexity evaluation methodology
     deps: [[ch-02]]
     see-also: [[bendersky-cross-entropy]], [[raschka-next-token-prediction]]
-->

# Perplexity of Fixed-Length Models

- **Core Insight:** Evaluation methodology (sliding window vs chunked) affects perplexity by 15%+.
- **Guideline:** Always report evaluation methodology alongside perplexity numbers.

- **Author:** Hugging Face Documentation
- **URL:** https://huggingface.co/docs/transformers/en/perplexity
- **Relevant chapters:** Evaluation metrics, language modeling, inference

## Summary
A practical guide to computing perplexity (PPL) for autoregressive language models, explaining why naive computation is flawed, how sliding-window strategies improve the metric, and providing working code for computing perplexity with GPT-2 using the Hugging Face Transformers library.

## Key Content

### What Is Perplexity?

Perplexity (PPL) is the most common metric for evaluating autoregressive/causal language models. It is **not well defined** for masked language models like BERT.

**Definition:** The exponentiated average negative log-likelihood of a sequence:

**PPL(X) = exp{ -(1/t) * sum_i(log p_theta(x_i | x_{<i})) }**

Where X = (x_0, x_1, ..., x_t) is a tokenized sequence.

Lower perplexity = better model (assigns higher probability to the actual text).

### The Fixed-Length Problem

Models like GPT-2 have a fixed context window (1024 tokens). For sequences longer than this window, we cannot compute the full conditional probability p(x_t | x_{<t}) for all tokens.

**Naive approach:** Split the sequence into non-overlapping segments and compute perplexity for each independently. This serves as a poor approximation because the model has less context at most prediction steps, yielding higher (worse) PPL.

### Sliding-Window Strategy

The proper approach involves repeatedly sliding the context window:
- The model gets maximum context for each prediction
- Closer approximation to the true sequence probability decomposition
- Typically yields a more favorable (lower) score

**Trade-off:** Requires a separate forward pass for each token in the corpus.

**Practical compromise:** Use a strided sliding window — move the context by larger strides rather than sliding by 1 token at a time. This balances computation speed with context quality.

### Implementation with GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from accelerate import Accelerator
import torch
from tqdm import tqdm
from datasets import load_dataset

device = Accelerator().device
model_id = "openai-community/gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# Load and encode dataset
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

max_length = model.config.n_positions  # 1024
stride = 512
seq_len = encodings.input_ids.size(1)

nll_sum = 0.0
n_tokens = 0
prev_end_loc = 0

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc

    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100  # Ignore context tokens in loss

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    # Accumulate totals
    num_valid_tokens = (target_ids != -100).sum().item()
    batch_size = target_ids.size(0)
    num_loss_tokens = num_valid_tokens - batch_size  # internal label shift
    nll_sum += neg_log_likelihood * num_loss_tokens
    n_tokens += num_loss_tokens

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

avg_nll = nll_sum / n_tokens
ppl = torch.exp(avg_nll)
```

### Key Implementation Details

- **Label masking:** Setting context tokens to -100 excludes them from loss computation, so only the "new" tokens in each window contribute
- **Internal label shift:** The model internally shifts labels left by 1 (predicting next token), so the actual number of loss tokens is num_valid - batch_size
- **Stride = max_length:** Equivalent to the naive non-overlapping approach
- **Stride = 1:** Maximum context but most expensive

### Empirical Results

| Strategy | GPT-2 PPL on WikiText-2 |
|----------|------------------------|
| stride=1024 (no overlap) | 19.44 |
| stride=512 (sliding window) | 16.44 |
| GPT-2 paper reported | 19.93 |

The striding approach produces a more favorable score that better approximates true autoregressive decomposition.

## Notable Insights
- Perplexity is undefined for masked LMs like BERT — it only makes sense for autoregressive models where you predict next tokens.
- The naive segment-by-segment approach gives artificially high (worse) perplexity because the model lacks context at segment boundaries.
- The sliding window stride is a practical knob: smaller stride = better approximation but more compute. stride=512 on a 1024-context model is a good default.
- The -100 label masking trick is a Hugging Face convention that tells CrossEntropyLoss to ignore those positions — essential for correct sliding-window perplexity computation.
