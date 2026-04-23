<!-- chapter: ch-04
     track: foundations
     title: Sequence Packing, Masking, and Chat Templates
     sources: [[sequence-packing]], [[loss-masking-prompt]], [[neftune]], [[hf-alignment-handbook]], [[flash-attention]]
     figures: figures/packed-attention.html
-->

# Chapter 4 — Sequence Packing, Masking, and Chat Templates

> **Core insight.** SFT batches are 50–89% padding by default. Packing many short sequences into a fixed-length block — with an attention mask that forbids cross-sequence attention and a loss mask that hides prompt tokens — recovers that throughput at zero accuracy cost. Most SFT bugs live in the gap between the pack, the mask, and the chat template.
>
> **Guideline.** Always pack SFT data with FlashAttention's varlen API (`flash_attn_varlen_func` + `cu_seqlens`). Mask all non-current-assistant tokens out of the loss with `ignore_index=-100`. Treat the chat template as model-specific data, not code — log a rendered sample every training run.

---

## Why this chapter exists

Three separate bugs look identical in the loss curve (slightly worse than expected; converges normally) and are individually responsible for more wasted SFT compute than any other class of error:

1. **Packing without a block-diagonal attention mask** — tokens in sequence 2 attend to sequence 1. Loss value is plausible, but gradients are wrong on every token of every pack.
2. **Packing without per-subsequence label masking** — prompt tokens in pack member 3 pollute the loss because the mask reset was forgotten when concatenating.
3. **Chat template mismatch** — training uses `<|im_start|>` + `<|im_end|>`, eval uses `### Human:` + `### Assistant:`. Model renders perfectly during training, hallucinates formats at eval.

All three fall out of properly implementing the pipeline described in [[sequence-packing]] + [[loss-masking-prompt]]. This chapter is the operational blueprint.

---

## 1. Why packing exists

SFT datasets have heavily right-skewed length distributions. A sample from the LIMA / Alpaca / UltraChat family averages ~500 tokens with a tail reaching 4K. Naive batching pads every sequence to the longest in the batch (or worse, to `max_seq_len`). Typical padding waste: 50% for chat mixtures, up to 89% for short-instruction sets. That's 2× wasted GPU time.

Packing treats the batch as a 1D bin-packing problem. Concatenate multiple short sequences end-to-end into blocks of length `max_seq_len`:

```
pack = [ s_1 | s_2 | s_3 | ... | s_n | PAD ]
```

Done naively, two things break. The **forward** breaks because tokens in `s_2` attend to tokens in `s_1` (softmax partition function leaks across documents). The **loss** breaks because the `(prompt, response)` split is per-subsequence, but the label tensor is per-pack.

Fix both and packing is mathematically equivalent to unpacked training at 2× the throughput.

---

## 2. The correct packed block — position IDs, `cu_seqlens`, attention mask

Source: [[sequence-packing]]. The runtime view of a packed block:

```
tokens        = [t_0 t_1 t_2 | t_3 t_4 | t_5 t_6 t_7 t_8 | PAD PAD]
position_ids  = [ 0   1   2  |  0   1  |  0   1   2   3  |  0   0]
cu_seqlens    = [0, 3, 5, 9, 11]                # cumulative boundaries
```

`position_ids` **reset to 0** at each sub-sequence boundary. RoPE and learned positions depend on this.

The attention mask is block-diagonal with causal structure *inside* each block:

```
       t_0 t_1 t_2 t_3 t_4 t_5 t_6 t_7 t_8
t_0 [   ✓   .   .   .   .   .   .   .   . ]
t_1 [   ✓   ✓   .   .   .   .   .   .   . ]
t_2 [   ✓   ✓   ✓   .   .   .   .   .   . ]    ← end of s_1
t_3 [   .   .   .   ✓   .   .   .   .   . ]
t_4 [   .   .   .   ✓   ✓   .   .   .   . ]    ← end of s_2
t_5 [   .   .   .   .   .   ✓   .   .   . ]
t_6 [   .   .   .   .   .   ✓   ✓   .   . ]
t_7 [   .   .   .   .   .   ✓   ✓   ✓   . ]
t_8 [   .   .   .   .   .   ✓   ✓   ✓   ✓ ]    ← end of s_3
```

See `figures/packed-attention.html` for an interactive visualisation.

**Never materialise this mask.** For `max_seq_len=4096`, the dense mask is 4096² = 16M entries. Instead, use FlashAttention's varlen interface which consumes `cu_seqlens` and computes block-diagonal causal attention without forming the mask:

```python
from flash_attn import flash_attn_varlen_func

# cu_seqlens_q, cu_seqlens_k: 1-D int32 tensors of cumulative offsets
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_len,
    max_seqlen_k=max_len,
    causal=True,
)
```

Memory is O(Σ Lᵢ), not O(max² · batch). This is why packing is viable for 128k+ context SFT.

---

## 3. Packing algorithms — SPFHP in practice

[[sequence-packing]] documents two algorithms. In production, SPFHP (Shortest-Pack-First Histogram-Packing) is the default because it's O(N log N) and achieves >99% fill:

1. Compute the dataset's length histogram.
2. For each bin of target size `max_seq_len`, greedily place the *longest remaining sequence that still fits*.
3. When no sequence fits, close the bin with padding and start a new one.

A minimal reference implementation (TRL's `DataCollatorWithPacking` does the equivalent):

```python
def pack_sequences(sequences, max_len):
    # sort by length descending for first-fit-decreasing
    seqs = sorted(sequences, key=len, reverse=True)
    bins = []       # list of lists
    for s in seqs:
        for b in bins:
            if sum(len(x) for x in b) + len(s) <= max_len:
                b.append(s); break
        else:
            bins.append([s])
    return bins
```

For reference: NNLSHP (Non-Negative Least Squares) achieves mathematically optimal fill given a histogram; it's overkill unless you care about the last 0.5%.

---

## 4. Loss masking — the other half of the bug surface

Source: [[loss-masking-prompt]]. Three rules govern which tokens contribute to the SFT loss:

1. **Pad tokens: label = -100.** Ignored by `F.cross_entropy(..., ignore_index=-100)`.
2. **Prompt / user tokens: label = -100.** Training on prompt tokens wastes capacity on a distribution the model never produces at inference.
3. **Prior-turn assistant tokens in multi-turn: label = -100.** When computing loss for turn *k*, turns 1..k−1 are part of the prompt.

The standard loss:

```
L_SFT(θ) = −(1 / T_y) Σ_{t=1..T_y} log π_θ(y_t | p, y_<t)
```

In code:

```python
labels = input_ids.clone()
labels[:prompt_len] = -100            # mask the prompt
loss = F.cross_entropy(
    logits[..., :-1, :].reshape(-1, V),
    labels[..., 1:].reshape(-1),
    ignore_index=-100,
)
```

**Inside a packed block** the mask is per-subsequence:

```python
labels = input_ids.clone()
for (start, end), prompt_len_i in zip(subseq_ranges, prompt_lens):
    labels[start : start + prompt_len_i] = -100          # mask prompt of sub-i
    labels[start + prompt_len_i : end]   = input_ids[start + prompt_len_i : end]
# PAD region stays -100 from clone + a final labels[-pad_len:] = -100
```

Miss this and loss is computed over "user input" tokens — which silently degrades SFT by 0.3–1.0% on helpfulness benchmarks. The Shi 2024 study ([[loss-masking-prompt]]) formalises this: response-only loss dominates across dataset sizes; the only regime where full-sequence loss helps is tiny-dataset + strong-base-model.

**Multi-turn masking** (turns = `[u_1, a_1, u_2, a_2, ..., u_k, a_k]`):

- Mask **all** user turns.
- Mask **all** prior assistant turns `a_1..a_{k-1}`.
- Train only on `a_k` tokens.

An alternative "per-turn unrolling" variant trains on each `a_i` in turn, giving k× training samples from one conversation.

### NEFTune — the free accuracy lift

[[neftune]]: add tiny uniform noise to the embedding layer during SFT (only during training). One line:

```python
# forward(self, input_ids):
embeds = self.embed_tokens(input_ids)
if self.training and self.neftune_alpha > 0:
    d = embeds.size(-1)
    mag = self.neftune_alpha / math.sqrt(d)
    embeds = embeds + torch.empty_like(embeds).uniform_(-mag, mag)
```

Typical `neftune_alpha = 5`. Reported AlpacaEval gains: ~10 points on Llama-2-7B. Composes with packing + masking — the noise is added after token lookup, before the transformer. Zero cost.

---

## 5. Chat templates — model-specific data

A chat template is the exact tokenised rendering of a conversation into a single string. It encodes who spoke when, where the assistant turn begins, and how to signal end-of-turn. Modern models ship a canonical template with the tokenizer; two reasonable-looking but *incompatible* templates produce two models that can't talk to each other.

Llama-3 template (simplified):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>
```

Qwen ChatML:

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

Anthropic (no ChatML-style markers; role prefixes inline): different again.

**Three operational rules.**

1. **Use the tokenizer's `apply_chat_template`.** Do not hand-roll the string. The tokenizer stores the template in `tokenizer.chat_template` (Jinja2 string). HF Alignment Handbook ([[hf-alignment-handbook]]) uses this with `train_on_response_only=True` to auto-generate the prompt mask.
2. **Log a rendered sample every training run.** The first line of every training log should be the rendered template for one random conversation, with loss-mask positions highlighted. This catches 90% of template bugs before step 100.
3. **Eval uses the exact same template.** Chat-template leakage — eval inputs rendered with a different template than training — is a top-3 silent regression mode. Bake the template into the eval harness.

**Special tokens: `<think>`, `<tool_call>`, reasoning tags.** When training a model to emit thinking tags (Qwen 3, Phi-4-reasoning, R1 distill variants), the special tokens must be in the tokenizer's vocabulary *before* SFT. Adding them post-hoc requires embedding resize + re-init of the new rows — a common bug that causes thinking-mode models to hallucinate noisy tokens inside the `<think>` block.

---

## 6. The full correct pipeline — copy-paste reference

```python
import torch, torch.nn.functional as F, math
from flash_attn import flash_attn_varlen_func

def build_packed_batch(conversations, tokenizer, max_len=4096):
    """Return input_ids, labels, cu_seqlens, position_ids for one pack."""
    ids, labels, positions, offsets = [], [], [], [0]
    for conv in conversations:
        # 1. Render via tokenizer's canonical chat template.
        text   = tokenizer.apply_chat_template(conv, tokenize=False,
                                               add_generation_prompt=False)
        # 2. Tokenize with completion mask (HF returns a response-only mask).
        enc    = tokenizer.apply_chat_template(
            conv, tokenize=True, return_assistant_tokens_mask=True,
        )
        tok    = enc["input_ids"]
        is_ass = enc["assistant_masks"]          # 1 where token is assistant
        if len(ids) + len(tok) > max_len:
            break                                # pack full; next pack
        ids.extend(tok)
        labels.extend([t if m else -100 for t, m in zip(tok, is_ass)])
        positions.extend(range(len(tok)))
        offsets.append(offsets[-1] + len(tok))
    # pad
    pad_len = max_len - len(ids)
    ids      += [tokenizer.pad_token_id] * pad_len
    labels   += [-100] * pad_len
    positions += [0] * pad_len
    return (torch.tensor(ids), torch.tensor(labels),
            torch.tensor(offsets, dtype=torch.int32),
            torch.tensor(positions))

def packed_forward(model, ids, labels, cu_seqlens, positions):
    # model's attention replaces the dense SDPA with flash_attn_varlen_func,
    # passing cu_seqlens and positions through.
    logits = model(input_ids=ids, position_ids=positions,
                   cu_seqlens=cu_seqlens).logits
    return F.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.size(-1)),
        labels[..., 1:].reshape(-1),
        ignore_index=-100,
    )
```

---

## Connections and what's next

- **[[sequence-packing]] / [[flash-attention]]** — the varlen API is the enabling primitive.
- **[[loss-masking-prompt]]** — Shi 2024's ablation of response-only vs full-sequence loss.
- **[[neftune]]** — noise injection; free accuracy on top of the above.
- **ch-05 (FSDP)** — packing reduces per-step memory, which is what lets FSDP's sharding be effective; the two are always paired.
- **ch-33 (Case Study C)** — distillation SFT (R1-distill, Bespoke-Stratos) ships long `<think>` traces; special-token pre-registration and template design are load-bearing there.
- **ch-34 (SFT lab)** — masking unit tests are a required deliverable for the SFT lab.

## Further reading

- [[sequence-packing]] — Krell 2021; the packing formalism and algorithms.
- [[loss-masking-prompt]] — Shi 2024 + HF Alignment Handbook conventions.
- [[hf-alignment-handbook]] — production `apply_chat_template` + `train_on_response_only=True` recipes.
- [[allenai-tulu-sft-recipe]] — packing + masking at the Tülu 3 scale.

## Companion visualization

**[figures/packed-attention.html](figures/packed-attention.html)** — interactive block-diagonal attention mask visualiser. Drop in sub-sequence lengths, watch the attention mask and `cu_seqlens` update in real time, and see the padding-fraction gauge shift as you add or remove sequences.
