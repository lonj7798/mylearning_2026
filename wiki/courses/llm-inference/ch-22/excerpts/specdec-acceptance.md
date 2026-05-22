---
chapter: ch-22
course: llm-inference
phase: read
excerpt_of: "Lossless speculative decoding acceptance test, worked through (Option 3)"
source_url: https://proceedings.mlr.press/v202/leviathan23a.html
created_at: "2026-05-21"
---

# Excerpt: Speculative-decoding acceptance test — the math, in code

**Source:** [[fast-inference-from-transformers-via-speculative-decoding]] (Leviathan, Kalman, Matias 2023)
**Raw-data:** [[raw-data/speculative-decoding]], [[raw-data/fast-inference-from-transformers-via-speculative-decoding]]

---

## The lossless-sampling identity

The result that makes speculative decoding work: if you draft from distribution `q`, then accept each token `x` with probability `min(1, p(x) / q(x))`, and on rejection sample a new token from the **normalised positive residual** `max(0, p(x) - q(x)) / Z`, then the resulting sample distribution is **exactly** `p`.

This is the entire correctness claim of SpecDec. Get it right and your method is lossless; get it wrong and you've built a biased decoder.

Proof sketch (Leviathan et al. 2023, §2.3): for any token `x`,

```
P(emit x) = q(x) · min(1, p(x)/q(x))           ← accepted path
          + [Σ_{y} q(y) · (1 - min(1, p(y)/q(y)))] · [max(0, p(x)-q(x)) / Z]   ← rejected path
```

After algebra both terms collapse to `p(x)`. The proof requires the residual to be properly normalised.

---

## The reference implementation, end to end

```python
import torch
import torch.nn.functional as F

def softmax_with_T(logits: torch.Tensor, T: float) -> torch.Tensor:
    if T == 0.0:
        # Greedy: dirac at argmax
        idx = logits.argmax(dim=-1, keepdim=True)
        p = torch.zeros_like(logits)
        p.scatter_(-1, idx, 1.0)
        return p
    return F.softmax(logits / max(T, 1e-6), dim=-1)


def speculative_decode_step(target, draft, prefix_ids: torch.Tensor,
                             K: int, T: float):
    """
    Run one speculative-decoding step: draft K tokens, target-verify all K+1 positions,
    accept with the Leviathan rule, return accepted tokens.

    target, draft : HF transformers CausalLM models
    prefix_ids    : (1, L) int64 on cuda
    K             : draft length
    T             : sampling temperature; 0 = greedy

    Returns: list[int] of accepted token IDs (>=1, <=K+1)
    """
    # --- 1. DRAFT K tokens autoregressively ---
    drafts: list[int] = []
    draft_dists: list[torch.Tensor] = []
    cur = prefix_ids
    for _ in range(K):
        with torch.no_grad():
            logits = draft(cur).logits[:, -1, :]    # (1, vocab)
        q = softmax_with_T(logits, T).squeeze(0)    # (vocab,)
        x = torch.multinomial(q, 1).item() if T > 0 else q.argmax().item()
        drafts.append(x)
        draft_dists.append(q)
        cur = torch.cat([cur, torch.tensor([[x]], device=cur.device)], dim=1)

    # --- 2. TARGET verifies all positions in one forward pass ---
    with torch.no_grad():
        # cur now has L + K tokens; we want target's predictions at positions L-1, L, ..., L+K-1
        # which are the next-token distributions for our drafted positions
        full_logits = target(cur).logits                      # (1, L+K, vocab)
        # Slice the K+1 positions whose predictions we need
        # Position L-1: predicts token at index L (first drafted)
        # Position L+K-1: predicts token at index L+K (the bonus token)
        target_logits = full_logits[:, -(K+1):, :]            # (1, K+1, vocab)
    target_dists = softmax_with_T(target_logits.squeeze(0), T)  # (K+1, vocab)

    # --- 3. ACCEPTANCE LOOP ---
    accepted: list[int] = []
    for i in range(K):
        x = drafts[i]
        q = draft_dists[i]
        p = target_dists[i]
        # Leviathan acceptance test
        ratio = (p[x] / max(q[x].item(), 1e-10)).clamp(max=1.0).item()
        u = torch.rand(1).item()
        if u <= ratio:
            accepted.append(x)
        else:
            # Sample from positive residual
            residual = (p - q).clamp(min=0.0)
            Z = residual.sum().clamp(min=1e-10)
            residual = residual / Z
            if T == 0.0:
                new_x = residual.argmax().item()
            else:
                new_x = torch.multinomial(residual, 1).item()
            accepted.append(new_x)
            return accepted    # stop accepting after first rejection

    # --- 4. BONUS TOKEN ---
    # All K drafts accepted; sample one more from the target's position-K prediction
    bonus_p = target_dists[K]
    if T == 0.0:
        bonus = bonus_p.argmax().item()
    else:
        bonus = torch.multinomial(bonus_p, 1).item()
    accepted.append(bonus)
    return accepted
```

---

## Verify lossless-ness before benchmarking

This test fails if the acceptance math is wrong. **Run it before measuring speed.**

```python
import torch
torch.manual_seed(0)

# Same seed, T=0 (greedy): SpecDec should produce identical sequences to target-only
target = ...   # Llama-3-8B-Instruct
draft  = ...   # Llama-3.2-1B-Instruct (same tokenizer)
prompt = tokenizer("The first three primes are", return_tensors='pt').input_ids.cuda()

# Method A: greedy with target only
tgt_only = target.generate(prompt, max_new_tokens=50, do_sample=False)

# Method B: speculative greedy
ids = prompt.clone()
while ids.shape[1] - prompt.shape[1] < 50:
    accepted = speculative_decode_step(target, draft, ids, K=4, T=0.0)
    ids = torch.cat([ids, torch.tensor([accepted], device=ids.device)], dim=1)
spec = ids[:, :prompt.shape[1] + 50]

assert torch.equal(tgt_only, spec), \
    f"LOSSLESS TEST FAILED!\n  target-only: {tokenizer.decode(tgt_only[0])}\n  spec:        {tokenizer.decode(spec[0])}"
print("LOSSLESS at T=0:", tokenizer.decode(spec[0]))
```

For `T > 0` (sampling) the sequences won't match (different random draws) but the **distribution** must match. Use a longer-form test:

```python
# Frequency test for T > 0: generate many continuations of the same prompt,
# verify token-frequency distributions agree.
from collections import Counter

def freq_dist(generator, n=2000):
    counts = Counter()
    for _ in range(n):
        out = generator()
        counts[out] += 1
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}

# Compare target's one-step distribution to SpecDec's effective one-step output
# (uses spec_decode_step with K=1 since we only care about the first emitted token)
T = 0.7
target_freq = freq_dist(lambda: torch.multinomial(softmax_with_T(target(prompt).logits[0,-1,:], T), 1).item())
spec_freq   = freq_dist(lambda: speculative_decode_step(target, draft, prompt, K=1, T=T)[0])

# TV distance should be small (<5%)
all_keys = set(target_freq) | set(spec_freq)
tv = 0.5 * sum(abs(target_freq.get(k, 0) - spec_freq.get(k, 0)) for k in all_keys)
print(f"TV distance: {tv:.4f}")
assert tv < 0.05, "Distribution diverged — acceptance math is wrong"
```

---

## Common bugs in the acceptance math

| Bug | Symptom |
|-----|---------|
| `accept if p > q` (logit/prob comparison instead of ratio) | Lossless test fails; some tokens never accepted that should be |
| Forgetting `clamp(max=1.0)` on the ratio | Ratios > 1 lead to negative residuals after rejection (rare but causes NaN multinomial) |
| Residual not re-normalised | Distribution biases toward already-likely tokens |
| Bonus token taken from target's *first* prediction instead of position K | Loses the +1 bonus speedup |
| Accepting drafts after first rejection | Lossless property breaks (later tokens conditioned on wrong context) |
| Sampling `u` once and reusing for multiple positions | Correlation between accept/reject decisions; subtle distribution bias |

The first three are the most common. Verify with the lossless test before touching anything else.

---

## Speedup math

Expected speedup per Leviathan §3.2:

```
speedup ≈ (1 - α^{K+1}) / ((1 - α) · (c + 1))
```

where `α` = per-token acceptance rate, `K` = draft length, `c` = cost of one draft pass / cost of one target pass.

For Llama-3-8B target + Llama-3.2-1B draft on ShareGPT chat (T=0):
- α ≈ 0.7
- K = 4
- c ≈ 0.10 (1B model runs ~10× faster than 8B per token)

→ Predicted speedup: `(1 - 0.7^5) / ((1 - 0.7) · 1.10)` ≈ `0.832 / 0.330` ≈ **2.52×**

Your measured TPOT speedup should land between 2.0× and 3.0× on this configuration. If you're below 2.0×, check (a) the draft model's tokenizer matches the target's, (b) you're amortising the target forward correctly (one pass per K drafts, not K passes), (c) the `c` ratio is what you expect (use `nvidia-smi dmon` to confirm).

---

## Connections

- [[ch-14]] — speculative decoding chapter; re-read before implementing.
- [[ch-15]] — feature-level variants (Medusa, EAGLE) build on this acceptance rule.
- [[excerpts/debugging-tree]] — what to do when the lossless test fails or the speedup is short.
