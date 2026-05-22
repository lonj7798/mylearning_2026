<!-- chapter: ch-14
     phase: decoding-acceleration
     title: Speculative Decoding Foundations
     sources: [[speculative-decoding]], [[fast-inference-from-transformers-via-speculative-decoding]], [[hf-assisted-generation]]
     figures: figures/speculative-acceptance-rate.html
-->

# Chapter 14 — Speculative Decoding Foundations

> **Core insight.** A small draft model proposes `K` future tokens at the cost of `K` cheap forward passes; the target model verifies all `K` in *one* expensive forward pass. The Leviathan-Kalman-Matias acceptance rule (Google 2023) — *accept token `x` if `u ≤ p_target(x)/p_draft(x)`, else resample from the positive residual `(p_target − p_draft)_+`* — preserves the target distribution **exactly**, so speculative decoding is *lossless*. Production speedups are 1.5-3× TPOT with the right drafter.
>
> **Guideline.** Use speculative decoding whenever target-model decode is memory-bandwidth-bound (which it almost always is at batch ≤ 32 and long context) and you have a drafter ~10-100× cheaper than the target. The right drafter is typically a *smaller LLM from the same family* (Llama-3-8B drafts for Llama-3-70B). N-gram or distilled drafters are alternatives when you can't afford a second model.

---

## Why this chapter exists

The decode loop you saw in [[kv-cache-memory-formula]] from ch-03 has an inescapable shape: one forward pass per token, memory-bandwidth-bound, ~30-50 ms per step on a 70B model. Doubling the GPU's HBM bandwidth would roughly halve the latency — except HBM bandwidth doesn't double on demand.

Speculative decoding (Xia et al. 2022, [[speculative-decoding]]; Leviathan-Kalman-Matias 2023, [[fast-inference-from-transformers-via-speculative-decoding]]) sidesteps the bandwidth ceiling with an algebraic trick: a transformer forward pass over a `K+1`-token sequence produces next-token logits at *every* position in parallel. So if you can *guess* the next `K` tokens cheaply, the expensive model only has to *verify* them — and the verification is one forward pass instead of `K`.

Three things to walk away with:

1. The exact algorithm box and the **probability-ratio acceptance rule** that makes it lossless. This is non-negotiable; people who don't understand the rule miss why "lossless speculative decoding" is actually true.
2. The speedup math: `α · K_eff − overhead` is the right mental model. Knowing what `α` is (acceptance rate) and what bounds the K_eff is (drafter quality + target verification cost) tells you whether spec-dec will help your workload.
3. The drafter-choice landscape — smaller LLM (the original Leviathan formulation), n-gram lookup, distilled draft, multi-head — and when each pays off.

This chapter is the foundation; ch-15 builds on it with the modern multi-head / feature-level / no-drafter variants (Medusa, EAGLE, Lookahead, PLD, MTP).

---

## 1. The core idea — guess and verify

### 1.1 Why a single forward pass verifies K tokens

A transformer forward pass on input `[x_1, x_2, ..., x_n]` produces hidden states at every position. The LM head turns each position's hidden state into a distribution over the next token. So after one forward pass, the model has emitted:

```
position 1 → distribution over x_2
position 2 → distribution over x_3
...
position n → distribution over x_{n+1}
```

The decode loop only uses the *last* position's distribution; the others are "wasted" because we don't yet know `x_2, ..., x_n` (they're being generated). But if we **already have a guess** for `x_2, ..., x_n`, those middle-position distributions become free verification.

That's the entire speculative-decoding trick. The drafter produces a candidate `[x'_2, ..., x'_K]`; the target model takes `[prompt..., x'_2, ..., x'_K]` and emits target distributions at every position; we check whether each `x'_i` is consistent with the target's distribution at position `i-1`. Accepted tokens are kept; the first rejected token is replaced by the target's own sample.

### 1.2 The algorithm box (greedy version, for intuition)

For greedy decoding (always take argmax):

```
Initialize: prompt
Loop:
  1. Drafter generates K candidate tokens: x'_1, x'_2, ..., x'_K
  2. Target forward pass over [prompt, x'_1, ..., x'_K]
     → emits target argmax at positions corresponding to x'_1, ..., x'_{K}, x'_{K+1}
  3. Walk left-to-right:
     accept x'_i if target_argmax(i) == x'_i
     stop at first rejection j; replace x'_j with target_argmax(j); discard x'_{j+1..K}
  4. Commit accepted tokens + the corrected one (or all K + one bonus if no rejection)
  5. Append to prompt; go to 1
```

If the drafter is reasonably aligned with the target, most `x'_i` match the target argmax and we accept long runs. Bonus token in the all-accepted case: free — the target already computed the distribution at the `K`-th position, no extra forward.

---

## 2. The lossless acceptance rule (Leviathan-Kalman-Matias 2023)

The greedy version preserves greedy outputs trivially: accepted tokens are exactly what the target would have produced. The hard case is **sampling** (temperature > 0, top-p, top-k) — where the target distribution `p(·)` matters, not just the argmax.

### 2.1 The acceptance rule

Let `q(x)` = drafter probability of proposing token `x`. Let `p(x)` = target probability at the same position.

For each drafted token `x'_i`, sample `u ~ Uniform(0, 1)`:

```
accept x'_i  if  u ≤ p(x'_i) / q(x'_i)
reject       otherwise
```

When `p ≥ q` everywhere on `x'_i`, the ratio is ≥ 1 and we always accept. When `p < q` for the proposed token, we sometimes reject — proportional to how much the target dislikes the proposal relative to the drafter.

### 2.2 What to do on rejection — the residual distribution

When we reject at position `j`, we resample from the **positive residual**:

```math
p_{\text{residual}}(x) = \frac{\max(0, \, p(x) - q(x))}{\sum_y \max(0, \, p(y) - q(y))}
```

This is the part of `p` that `q` *underweighted*. Combining "accept with probability `min(1, p/q)`" with "resample residual on reject" yields a sample that is exactly distributed as `p`.

### 2.3 The proof (short version)

For any token `x`:

```
P(commit x at position j)
  = P(draft x) · P(accept | draft x)  +  P(reject anything) · P(residual sample = x)
  = q(x) · min(1, p(x)/q(x))           +  (1 - α) · p_residual(x)
```

Case `p(x) ≥ q(x)`: `min(1, p/q) = 1`, so first term is `q(x)`. Residual at `x` is `(p(x) − q(x)) / Z` where `Z = 1 − α` (algebra). Combining: `q(x) + (1-α) · (p(x)-q(x))/(1-α) = q(x) + p(x) − q(x) = p(x)`. ✓

Case `p(x) < q(x)`: first term is `q(x) · p(x)/q(x) = p(x)`. Residual at `x` is 0 (`max(0, p−q) = 0`). Sum is `p(x)`. ✓

So `P(commit x) = p(x)` regardless of which case. The committed-token distribution is exactly `p`. **No quality loss.**

### 2.4 What "lossless" actually means

It means: for any input prompt and any sampling configuration (temperature, top-p, top-k, etc.), the committed-token distribution is *identical* in distribution to what you'd get from naive autoregressive sampling on the target alone. Sequences are not bit-identical (different random seeds give different samples), but the *distribution over output sequences* is unchanged.

This is the property that makes speculative decoding production-deployable: you do not need to re-evaluate quality, re-do red-teaming, or worry about regression on benchmarks. The output distribution is unchanged.

---

## 3. The speedup math

### 3.1 The basic formula

Let:
- `K` = number of tokens drafted per round
- `α` = expected acceptance rate per token (a function of how well the drafter matches the target)
- `c` = cost ratio = (drafter forward pass time) / (target forward pass time)
- `K_eff` = expected number of accepted tokens per target forward pass

Then per target forward pass we generate `1 + K_eff` tokens (the `K_eff` accepted draft tokens + 1 either the corrected rejection or the bonus). The drafter cost is `K · c` target-equivalents per round.

**Speedup vs naive decoding:**

```math
\text{speedup} = \frac{1 + K_{\text{eff}}}{1 + K \cdot c}
```

For independent acceptance with rate `α` per draft token:

```math
K_{\text{eff}} = \sum_{i=0}^{K-1} \alpha^{i+1} = \frac{\alpha (1 - \alpha^K)}{1 - \alpha}
```

For `α=0.7, K=4`: `K_eff = 0.7 · (1 − 0.7⁴) / 0.3 ≈ 1.77`. With `c = 0.05` (drafter is 20× cheaper): `speedup = (1 + 1.77) / (1 + 4·0.05) = 2.77 / 1.20 ≈ 2.3×`.

For `α=0.9, K=8`: `K_eff ≈ 5.69`. With `c=0.05`: `speedup ≈ 4.78×`.

For `α=0.5, K=4`: `K_eff ≈ 0.94`. With `c=0.05`: `speedup ≈ 1.62×`.

The pattern: speedup is dominated by `α`. A "good drafter" (high α) matters far more than a "fast drafter" (low c). This is why people obsess over drafter alignment with the target.

### 3.2 The K tradeoff

For high `α`, more K helps (more chances for free verification). For low `α`, more K hurts (drafter cost grows linearly, accepted tokens grow geometrically with `α < 1`).

Optimal K (rough):

```math
K^* \approx \log_{1/\alpha}(1 / c) - 1
```

For `α=0.7, c=0.05`: `K* ≈ log_{1.43}(20) − 1 ≈ 8 − 1 = 7`. In practice K=4 or 5 is what frameworks default to because acceptance is correlated (if you miss at position 3, you're more likely to miss at position 4).

### 3.3 Typical production speedups

| Method | Drafter | α (typical) | K | Reported speedup |
|--------|---------|-------------|---|------------------|
| Leviathan (Google 2023) | T5-small drafting T5-XXL | 0.6-0.8 | 5-7 | 2.0-2.7× |
| HF assisted-generation | TinyLlama drafting Llama-2-7B | 0.6-0.7 | 5 | 1.5-2.5× |
| HF assisted-generation | Llama-3-8B drafting Llama-3-70B | 0.7-0.8 | 5 | 2.0-3.0× |
| Prompt lookup decoding | n-gram from prompt | 0.3-0.95 (workload-dep) | 10 | 1.5-3.0× on RAG/code |

Numbers above 3× usually involve drafter improvements beyond vanilla spec-dec (Medusa, EAGLE — see ch-15).

---

## 4. Choosing a draft model

### 4.1 The three drafter families

**Smaller LLM from the same family.** The original Leviathan formulation. Llama-3-8B drafts for Llama-3-70B; both share tokenizer, training data, and architecture style → high alignment. Typical α ≈ 0.7-0.8.

Cost: load + memory + compute of a second model. For a 70B target, an 8B drafter adds ~16 GB to memory budget + ~5% of the target's compute per drafter step.

**Distilled drafter.** Train a small model specifically to predict what the target would produce. Better alignment than a generic small model from the same family → α ≈ 0.8-0.9, but requires a distillation pipeline. Used by some commercial serving stacks; less common in OSS.

**N-gram lookup drafter.** No model at all. The "drafter" is a string-search procedure: find n-grams in the prompt that match the current suffix, propose the continuation. Cost: ~0 (pure CPU lookup). Acceptance highly workload-dependent — α ≈ 0.9 on summarization / code editing (text is largely copy), α ≈ 0.2 on creative generation. Implemented in HF Transformers as `prompt_lookup_num_tokens` ([[prompt-lookup-decoding]]) and in vLLM as `--speculative-config` with `method=prompt-lookup`.

### 4.2 Tokenizer compatibility

The drafter and target **must share the tokenizer** — otherwise position-by-position alignment of the draft sequence to the target distribution is broken. Llama-3-8B and Llama-3-70B share the same tiktoken-based BPE → compatible. Mistral-7B and Llama-3-8B do not share → incompatible.

HF assisted-generation enforces this; it errors if `assistant_model.tokenizer != model.tokenizer`. "Universal assisted decoding" (HF, 2024) introduces re-tokenization at the boundary; works but adds overhead and can fragment acceptance.

### 4.3 The serving-system fit

Speculative decoding interacts with continuous batching ([[continuous-batching]] from ch-04) in subtle ways:

- **Decode batch size shrinks** because each request now needs `K+1` tokens of forward work per "logical step". To keep target utilization high, you want more in-flight requests.
- **Variable acceptance** means committed tokens per step vary across requests. The scheduler must reconcile this in iteration-level scheduling (vLLM does it correctly; some early frameworks did not).
- **KV cache writes are speculative** — if a token is rejected, its KV must be discarded. Modern implementations write to a temp buffer and only commit on acceptance.

vLLM exposes this as `--speculative-config '{"method":"draft","model":"meta-llama/Llama-3.2-1B","num_speculative_tokens":5}'`. Default behavior: spec-dec disabled at high concurrency (when target is already compute-bound, drafter adds no value), enabled at low concurrency.

---

## 5. HuggingFace assisted generation

The canonical reference implementation. From the [[hf-assisted-generation]] blog (Joao Gante 2023):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", device_map="auto")
draft  = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",  device_map="auto")
tok    = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

inputs = tok("Explain RoPE positional encoding to me.", return_tensors="pt").to("cuda")
out = target.generate(
    **inputs,
    assistant_model=draft,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)
```

The `assistant_model` kwarg switches `generate()` into assisted decoding. Internally, the loop is:

```
1. draft.generate(K=5) → 5 candidate tokens
2. target forward pass over prompt + 5 candidates
3. For each position, run the acceptance rule with target/draft logits
4. Commit accepted prefix; redraft from new position
```

Limitations (per HF docs): no batched-input support yet (batch=1 only), tokenizers must match (or use universal-assisted-decoding), and `assistant_confidence_threshold` can early-stop drafting when confidence drops.

A simpler no-assistant variant: `prompt_lookup_num_tokens=10` uses prompt-n-gram drafting (see [[prompt-lookup-decoding]] in ch-15).

---

## 6. When speculative decoding *doesn't* help

Spec-dec is not free; it has a payoff envelope. It loses when:

- **Compute-bound target** (large batch, prefill, very short context) — the target is already saturated; the drafter adds overhead without freeing bandwidth. Default vLLM behavior disables spec-dec when scheduler is compute-bound.
- **Bad drafter alignment** (α < 0.4) — `K_eff < 1`; the cost of running the drafter exceeds the saved target passes.
- **Very long context** (>32k) where the target's per-step bandwidth is dominated by KV-cache scan, not weight load. Spec-dec still works but the speedup shrinks because verification cost grows with context.
- **High-temperature creative generation** (T ≥ 1.2) — acceptance drops because the target distribution is flat; the residual sampling diverges from the drafter frequently.
- **Mismatched drafter tokenizer** — re-tokenization fragments acceptance to single tokens.

Rule of thumb: spec-dec is most useful for *low-to-medium batch*, *moderate context*, *low temperature*, *long generations*. Exactly the chatbot SLA where TPOT matters most.

---

## 7. Pitfalls

- **Numerical mismatch between draft and target precision.** If drafter runs in fp16 and target in bf16, the residual `p − q` computation can underflow. Use the same precision for both probability vectors.
- **Drafter caching past states.** The drafter has its own KV cache; you must invalidate it on rejection (the drafter committed something the target rejected). Most frameworks handle this; verify yours does.
- **Greedy/sampling drift.** Some old implementations used greedy on the drafter + sampling on the target. The acceptance rule depends on `q(x) > 0` for any sampled `x`; greedy drafter has `q = 0` for everything except the argmax → all rejection → 0 speedup. Use sampling on the drafter at the same temperature.
- **Bonus token forgotten.** After accepting all K draft tokens, the target's `K`-th forward emit is a free bonus token. Forgetting to commit it costs ~10-15% of the speedup. Check your implementation.
- **Spec-dec + continuous batching.** Requires speculative KV writes to be reversible. PagedAttention block tables make this clean (allocate a temp block; free it on rejection); contiguous KV makes it ugly. Don't roll your own.
- **`α` measurement is per-workload.** Don't trust a paper's reported speedup unless its workload looks like yours. Code generation can hit α=0.9; creative chat tops out at α=0.6.

---

## 8. Practitioner's cheat-sheet

```python
# vLLM offline API — spec-dec with a draft model.
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=8,
    speculative_config={
        "method": "draft",
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_speculative_tokens": 5,
    },
)

# Or prompt-lookup spec-dec (no draft model, good for RAG/code)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=8,
    speculative_config={
        "method": "prompt-lookup",
        "num_speculative_tokens": 10,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    },
)

# Monitor acceptance rate via the engine's metrics endpoint:
# vllm_spec_decode_efficiency, vllm_spec_decode_num_accepted_tokens_total
```

---

## Connections and what's next

- **[[kv-cache-memory-formula]] / ch-03** — explains why decode is bandwidth-bound, which is *why* spec-dec saves time (you're amortizing the weight-load cost across K tokens).
- **[[continuous-batching]] / ch-04** — interacts with spec-dec; vLLM's V1 engine handles this correctly.
- **ch-15** — the modern variants: Medusa (multiple heads on the target, no separate drafter), EAGLE (feature-level drafting, dynamic tree), Lookahead (parallel n-gram, no drafter), PLD (prompt lookup), MTP (DeepSeek-V3's native speculative heads).
- **[[vllm-scheduler]] / ch-16** — vLLM's speculative-decoding integration into the scheduler loop.

## Further reading

- [[speculative-decoding]] — Xia et al. 2022, the original SpecDec paper.
- [[fast-inference-from-transformers-via-speculative-decoding]] — Leviathan-Kalman-Matias 2023, the lossless sampling formulation.
- [[hf-assisted-generation]] — Joao Gante 2023, the reference library implementation.

## Companion visualization

**[figures/speculative-acceptance-rate.html](figures/speculative-acceptance-rate.html)** — interactive plot of expected speedup vs `α` and `K` under varying drafter cost `c`. Use it to predict whether spec-dec will help your workload before deploying.

## Excerpts

- [excerpts/speculative-decoding.md](excerpts/speculative-decoding.md) — Xia 2022 draft-and-verify, the original SpecDec.
- [excerpts/leviathan-2023.md](excerpts/leviathan-2023.md) — the acceptance rule + residual sampling proof.
- [excerpts/hf-assisted-generation.md](excerpts/hf-assisted-generation.md) — the reference API + tokenizer compatibility.
- [excerpts/speculative-speedup-math.md](excerpts/speculative-speedup-math.md) — expected-acceptance derivation, optimal K, real numbers.
