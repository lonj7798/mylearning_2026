---
chapter: ch-15
course: llm-inference
phase: read
excerpt_of: "Lookahead Decoding (Fu et al. 2024) + Prompt Lookup Decoding (HF Transformers 2024)"
source_url: https://arxiv.org/abs/2402.02057
created_at: "2026-05-21"
---

# Excerpt: Lookahead Decoding + Prompt Lookup Decoding

**Authors:** Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang (Lookahead, LMSYS / hao-ai-lab); Apoorv Saxena et al. (PLD, HuggingFace)
**Year:** 2024
**Venue:** ICML 2024 (Lookahead); HF blog/docs (PLD)
**URLs:** https://arxiv.org/abs/2402.02057 ; https://huggingface.co/docs/transformers/assisted_decoding
**Raw-data sources:** [[raw-data/lookahead-decoding]] ; [[raw-data/prompt-lookup-decoding]]

---

## Part 1: Lookahead Decoding

### The Jacobi-iteration insight

Autoregressive decoding can be reformulated as finding the fixed point of a system:

```
x_1 = LM(prompt)
x_2 = LM(prompt, x_1)
x_3 = LM(prompt, x_1, x_2)
...
```

This is a sequential dependency: `x_i` depends on `x_{i-1}`. Naive iteration = the standard AR loop, one position per pass.

**Jacobi iteration** is the classical numerical method for fixed-point equations: start with a *guess* for all `K` future tokens; run the LM on the guess; the LM emits `K` next-token distributions (one per position); use those as the next iteration's guess; converge.

```
guess_0 = random_tokens(K)
for iter in 1..max_iter:
    logits = LM(prompt + guess_{iter-1})
    guess_iter = argmax(logits) at each position
    if guess_iter == guess_{iter-1}: break
final tokens = guess_iter
```

Empirically, Jacobi converges fast on "obvious" positions (~1-2 iterations for code) and slowly on creative ones (~5-10).

### The Lookahead architecture

Lookahead Decoding runs two branches in a single target forward pass:

```
Input sequence: [prompt, current_guess_tokens, candidate_ngrams]
                         └────────┬────────┘  └────────┬────────┘
                            Lookahead branch    Verification branch
                            (Jacobi update)      (test stored n-grams)
```

The attention mask is structured so the lookahead branch attends to the prompt (and to a small window), and the verification branch attends to the prompt + the n-gram being tested. Both branches share weights and one forward pass.

Each forward pass:
1. **Lookahead branch** runs Jacobi on a window of `W` future positions. New `(W-1)`-grams are added to a "pool" of candidate continuations.
2. **Verification branch** tests the longest n-gram from the pool against the actual target output at the next position. Accept if matches; advance multiple tokens at once.

### Speedup numbers

| Model | Workload | Speedup |
|-------|----------|---------|
| LLaMA-2-7B | MT-Bench chat | 1.5-1.8× |
| LLaMA-2-13B | MT-Bench chat | 1.6-1.9× |
| CodeLlama-7B | HumanEval | 2.3-2.5× |
| CodeLlama-34B | HumanEval | 2.4× |

Lower than EAGLE because no learned drafter. Win: **no extra weights, no training**, works on any AR LM as a drop-in.

### Knobs

- `W` (window size): typically 5-10. Larger W → more n-gram candidates but slower Jacobi convergence per window.
- `N` (n-gram size): typically 5-7.
- `G` (n-gram pool size): typically 10-15 candidates verified per step.

### Pitfalls

- **Convergence is workload-dependent.** Code converges fast (deterministic syntax); open-ended chat is slow. Measure before deploying.
- **Memory overhead.** The n-gram pool grows with generation length; cap it.
- **Doesn't compose well with continuous batching.** Lookahead's per-request window state complicates the iteration-level scheduler. Most serving frameworks don't ship it; vLLM has experimental support.

---

## Part 2: Prompt Lookup Decoding (PLD)

### The mechanism

The "drafter" is a string search over the prompt:

```python
def prompt_lookup_drafter(prompt_tokens, generated_tail, K, n_max, n_min):
    """
    prompt_tokens: list of token IDs (the original prompt)
    generated_tail: list of last few generated tokens
    K: number of speculative tokens to propose
    n_max, n_min: range of n-gram sizes to try

    Returns: list of K candidate tokens (or empty if no match)
    """
    for n in range(n_max, n_min - 1, -1):
        if len(generated_tail) < n:
            continue
        suffix = generated_tail[-n:]
        # Find suffix in prompt_tokens
        for i in range(len(prompt_tokens) - n):
            if prompt_tokens[i:i+n] == suffix:
                # Found! Propose the next K tokens from the prompt
                start = i + n
                end = min(start + K, len(prompt_tokens))
                return prompt_tokens[start:end]
    return []
```

Pure Python, microseconds per step. `c ≈ 0.001`.

### When PLD wins

The "drafter" is exact text from the prompt — so it has near-perfect alignment with the target *when the output is copying from the prompt*. Workloads where this is common:

- **RAG**: answer often quotes retrieved passages.
- **Summarization**: summary often reuses key spans from the source.
- **Code refactoring**: output ≈ input with localized edits.
- **Translation**: proper nouns, numbers, code blocks pass through verbatim.
- **Document Q&A**: short answers quote a span.

For these, `α ≈ 0.85-0.95`, K_eff ≈ 5-8, speedup 2-3×.

### When PLD loses

- **Creative writing**: output is novel; prompt has no useful n-grams to copy. `α ≈ 0.1-0.2`.
- **Open-ended chat**: same problem.
- **Math reasoning**: output is novel symbols, not in the prompt.

Rule: if your workload's `α < 0.4`, **disable PLD** — it's a slowdown.

### Production knobs

```python
# HF Transformers
out = model.generate(
    input_ids,
    prompt_lookup_num_tokens=10,
    max_new_tokens=512,
)

# vLLM V1
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    speculative_config={
        "method": "prompt-lookup",
        "num_speculative_tokens": 10,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    },
)
```

### Why PLD is the "free lunch" of spec-dec

- **No training.**
- **No extra weights.**
- **No tokenizer compatibility constraint** (drafter is just lookup).
- **Microsecond drafter cost** — even if α drops to 0.5, you still win.
- **Implementation is ~30 lines of Python.**

Every serving stack supports PLD; it's the first thing to enable for RAG/code workloads.

---

## Comparison

| Method | Training | Extra weights | α (chat) | α (RAG/code) | Speedup (chat) | Speedup (RAG/code) |
|--------|----------|---------------|----------|--------------|----------------|---------------------|
| Lookahead | none | none | 0.4-0.6 | 0.7-0.85 | 1.5-1.8× | 2.3-2.5× |
| PLD | none | none | 0.1-0.3 | 0.85-0.95 | <1× (disable) | 2-3× |

PLD wins by a lot when copying happens; Lookahead is the safer general-purpose no-training choice.

---

## Connections

- [[excerpts/medusa]], [[excerpts/eagle]] — the trained-drafter alternatives.
- [[excerpts/leviathan-2023]] — both methods use the standard acceptance rule.
- [[ch-15]] — parent chapter.
