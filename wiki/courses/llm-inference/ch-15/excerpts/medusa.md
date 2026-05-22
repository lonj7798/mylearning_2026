---
chapter: ch-15
course: llm-inference
phase: read
excerpt_of: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (Cai et al. 2024)"
source_url: https://arxiv.org/abs/2401.10774
created_at: "2026-05-21"
---

# Excerpt: Medusa — Multi-Head Speculative Decoding

**Authors:** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao
**Year:** 2024
**Venue:** ICML 2024
**URL:** https://arxiv.org/abs/2401.10774
**Raw-data source:** [[raw-data/medusa]]

---

## Architecture

Add K decoding "heads" on top of the target LLM's penultimate hidden state. Each head `k ∈ {1, ..., K}` predicts the token at offset `+k+1`:

```
Target trunk → h_t   ∈ ℝ^d
                ↓
        ┌──────┼──────┬──────┐
        ↓      ↓      ↓      ↓
       head_0 head_1 head_2 head_3
        ↓      ↓      ↓      ↓
    token_t+1 t+2   t+3    t+4
```

Each head is structured as:

```python
class MedusaHead(nn.Module):
    def __init__(self, d, vocab_size, n_layers=1):
        self.blocks = nn.Sequential(*[
            ResBlock(d) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        # Optional: share lm_head with the target's lm_head

    def forward(self, h):
        return self.lm_head(self.blocks(h))
```

One ResBlock per head, plus a Linear projection. For K=4 heads on Llama-2-7B: ~200M extra params (~3% of base).

---

## Training modes

**Medusa-1**: freeze the trunk, train heads only.

```python
target.requires_grad_(False)
for head in medusa_heads:
    head.requires_grad_(True)

# Loss: K-position next-token CE
loss = 0
for k in range(K):
    logits_k = medusa_heads[k](h_t)
    loss += F.cross_entropy(logits_k, x[t+k+1])
loss = loss / K
```

Fast (~few hours on a small calibration corpus), no quality regression on the trunk. Accepts ~2-3 tokens per round.

**Medusa-2**: jointly train heads + finetune trunk. Trunk gets gradient through head losses + the original next-token loss.

```python
total_loss = next_token_loss + lambda_medusa * medusa_head_loss
```

Better acceptance (~3-4 tokens per round); ~1-2% perplexity regression on base model.

---

## Tree attention — top-k per head

Each head can emit *top-k* candidates rather than just argmax. The combinatorics build a draft tree:

```
head 0 emits top-3:    [A, B, C]
head 1 emits top-3 conditional on each:
    A → [A1, A2, A3]
    B → [B1, B2, B3]
    C → [C1, C2, C3]
head 2 emits top-2 conditional on each:
    A1 → [A1a, A1b], etc.
```

Total candidate paths: 3 × 3 × 2 = 18 paths, each spanning depth K.

For verification, concatenate all tree nodes (deduplicated) into one sequence, run the target forward, and apply a **tree attention mask**:

```
Each tree node attends only to:
  - its ancestors in the tree
  - the original prompt tokens
NOT to:
  - sibling branches at the same depth
  - other branches' descendants
```

In matrix form, the tree mask is a lower-triangular mask with extra zero blocks for cross-branch positions. FlashAttention's `attn_mask` parameter supports this; FlashInfer ships a specialized "tree attention" kernel that handles it more efficiently.

After verification, walk the tree from root to find the longest accepted prefix (using the Leviathan acceptance rule per position).

---

## The verification cost

Tree size M (number of nodes) is the verification budget. Larger M = more candidates = more chances for long-accepted paths, but more target compute per round.

The paper uses M ≈ 64 for the published results. Empirically: M=60 gives a sweet spot for Llama-2-7B at K=4.

---

## Speedup numbers (paper, Table 1)

| Model | Method | Speedup |
|-------|--------|---------|
| Vicuna-7B | Medusa-1 | 2.05× |
| Vicuna-7B | Medusa-2 | 2.46× |
| Vicuna-13B | Medusa-1 | 2.18× |
| Vicuna-13B | Medusa-2 | 2.71× |
| LLaMA-2-Chat-7B | Medusa-1 | 1.85× |
| LLaMA-2-Chat-13B | Medusa-1 | 2.01× |
| Zephyr-7B | Medusa-1 | 2.06× |

Measured on MT-Bench at greedy decoding, batch=1, A100.

---

## Where Medusa is weaker than EAGLE

Heads are **independent of each other** — head_2 predicts t+3 from `h_t` without knowing what head_1 emitted for t+2. This is the core limitation that EAGLE addresses by making the drafter autoregressive at the feature level.

Concretely: head_3's prediction is conditioned on `h_t` (a single hidden state from `t`), not on the just-emitted candidates `[x'_{t+1}, x'_{t+2}, x'_{t+3}]`. So head_3 can't refine its prediction based on which candidates head_1 and head_2 chose.

EAGLE's feature-level autoregression closes this gap.

---

## Pitfalls

- **Head training data**. Use the *target's own outputs* as training data, not external corpora. Otherwise the heads predict tokens the target wouldn't have predicted, and acceptance suffers.
- **Tree pruning is workload-dependent**. Optimal tree shapes for chat vs code differ. Most implementations let you specify a tree topology directly (vLLM's `draft_tree_choices` parameter).
- **lm_head sharing**. Sharing the target's lm_head saves ~vocab_size × d params per head. Common practice.
- **Batch=1 vs batch=N**. Medusa shines at batch=1-8; past batch=8, target verification becomes compute-bound (each tree node × batch slots → memory bandwidth saturated). Disable spec-dec at high batch.

---

## Connections

- [[excerpts/eagle]] — autoregressive feature-level drafter; the natural successor.
- [[excerpts/eagle-2]] — dynamic draft tree on top of EAGLE.
- [[excerpts/multi-token-prediction]] — Medusa-style heads but as a pretraining objective (DeepSeek V3).
- [[ch-15]] — parent chapter.
