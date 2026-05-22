<!-- chapter: ch-01
     track: generation-foundations
     title: Autoregressive Generation Loop + Sampling
     sources: [[attention-is-all-you-need]], [[language-models-are-unsupervised-multitask-learners]], [[neural-text-degeneration]], [[beam-search]], [[hf-generation-strategies]], [[openai-streaming-and-token-usage]], [[structured-generation-constrained-decoding]]
     figures: figures/sampling-distribution.html
-->

# Chapter 1 — Autoregressive Generation Loop + Sampling

> **Core insight.** Every LLM inference engine is a single loop: run the prompt through the model once (prefill), then run one forward pass per output token (decode), each step turning a `[vocab]` logit vector into a single sampled token id, until a stop condition fires. Everything else — PagedAttention, continuous batching, speculative decoding — is an optimization layered over this loop without changing its shape.
>
> **Guideline.** Memorize the loop. Pick the decoding mode by task: greedy / low-temperature for extractive or constrained tasks; `top_p ∈ [0.9, 0.95]` with `temperature ∈ [0.7, 1.0]` for open-ended chat; beam search **only** for short-answer constrained tasks (translation, formal summarization). Never reach for beam search in open-ended chat — Holtzman 2019 settled this.

---

## Why this chapter exists

You cannot reason about a serving stack without first internalizing what it is serving: a single, mechanical, token-by-token loop. The whole rest of this course optimizes pieces of this loop — `Attention(Q,K,V)` (ch-02), KV-cache reads (ch-03), batch composition (ch-04), kernels (ch-11), draft tokens (ch-14). If the loop is fuzzy, those optimizations are just words.

Three things you should walk away with:

1. The exact pseudocode for prompt → prefill → decode loop → stop, including where the KV cache appears and where sampling happens.
2. The four sampling knobs (`temperature`, `top_k`, `top_p`, `min_p`) as transformations applied **to logits before sampling**, plus the one historical mistake (beam search for chat) and why it fails.
3. Why "structured output" is a `logits[invalid] = -inf` mask, not a prompt instruction — and how that mask interacts with streaming and batching.

All of these come from [[attention-is-all-you-need]], [[neural-text-degeneration]], [[hf-generation-strategies]], [[openai-streaming-and-token-usage]], and [[structured-generation-constrained-decoding]] in the raw-data library.

---

## 1. The autoregressive loop in 25 lines of pseudocode

GPT-2 ([[language-models-are-unsupervised-multitask-learners]]) made it a default that *the same* causal decoder stack produces every kind of output — answers, completions, code, JSON — by repeatedly predicting one next token. The serving pipeline looks like:

```python
def generate(model, tokenizer, prompt, max_new_tokens, stop_ids, sampler):
    # 1. Tokenize
    input_ids = tokenizer.encode(prompt)                    # [S]
    output_ids = []

    # 2. PREFILL — one forward pass over the entire prompt.
    #    Returns logits for every position; we only keep position S-1.
    #    Side effect: each layer writes K, V for every prompt token into the KV cache.
    logits, kv_cache = model.forward(input_ids, kv_cache=None)
    next_logits = logits[-1]                                # [vocab]

    # 3. DECODE LOOP — one forward pass per output token.
    for step in range(max_new_tokens):
        # 3a. Logits -> token id (sampling).
        next_id = sampler(next_logits)
        output_ids.append(next_id)

        # 3b. Stop conditions.
        if next_id in stop_ids or step + 1 == max_new_tokens:
            finish_reason = "stop" if next_id in stop_ids else "length"
            break

        # 3c. ONE-TOKEN forward pass — uses cached K, V for all prior tokens.
        #     Appends one new K, V entry per layer to kv_cache.
        logits, kv_cache = model.forward([next_id], kv_cache=kv_cache)
        next_logits = logits[-1]                            # [vocab]

    return output_ids, finish_reason
```

Three things to notice immediately:

- **Prefill is one call**; decode is `O(max_new_tokens)` calls. The first call costs roughly `O(S²·d)` in attention work and `O(S·d²)` in FFN work; each subsequent decode call costs `O(S·d)` attention and `O(d²)` FFN per layer. The asymmetry is the entire reason later chapters split these phases. See [[prefill-vs-decode]] and ch-03 §3 for the arithmetic-intensity argument.
- **The KV cache is what lets decode skip recomputing past keys/values.** [[attention-is-all-you-need]]'s scaled dot-product attention `softmax(QK^T / √d_k) V` requires all past `K, V` to compute the current step's attention; we cache them rather than recompute them. The memory cost of that cache is ch-03's main formula.
- **Sampling is a pure function of the logit vector at position-1.** The model produces `[vocab]` floats; the sampler turns them into one integer. Nothing in the model architecture decides between greedy and top-p — it's all post-hoc on the logits.

---

## 2. Logits → probabilities → token: the four-knob recipe

Decoding strategies operate on a single `[vocab]` logit vector. The order of operations is universal:

```python
def sample(logits, temperature, top_k, top_p, min_p):
    # 1. Temperature: divide before softmax. T<1 sharpens, T>1 flattens.
    if temperature == 0.0:
        return int(logits.argmax())                          # greedy
    logits = logits / temperature

    # 2. Top-k: keep highest-k logits, set the rest to -inf.
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, -float("inf"), logits)

    # 3. Convert to probabilities.
    probs = torch.softmax(logits, dim=-1)

    # 4. Top-p (nucleus): keep smallest set whose CDF ≥ p, renormalize.
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        mask = cdf <= top_p
        mask[..., 0] = True                                  # always keep argmax
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs * mask)
        probs = probs / probs.sum()

    # 5. Min-p: drop everything below min_p * max_prob (more local than top-p).
    if min_p > 0.0:
        threshold = min_p * probs.max()
        probs = torch.where(probs < threshold, 0.0, probs)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, num_samples=1))
```

**Operational reading of each knob.** Temperature `T` rescales raw logits by `1/T`; the softmax exponential then becomes `e^(z/T)`, so `T<1` concentrates mass on the argmax and `T>1` spreads it out. Top-k truncates the vocabulary to a fixed size; the model's confidence is ignored. Top-p (nucleus sampling, [[neural-text-degeneration]]) keeps the smallest set `V_p` such that `Σ_{x∈V_p} P(x|context) ≥ p` and adapts to model confidence — a low-entropy step gets 1–2 candidates, a high-entropy step might get 50. Min-p (newer, ~2024) thresholds at `p · max(probs)` and is more robust to scale than top-p.

[[hf-generation-strategies]] maps these directly to `GenerationConfig`: `do_sample=False, num_beams=1` is greedy; `do_sample=True, top_p=0.95, temperature=0.8` is the chat default; `num_beams=4, early_stopping=True` is beam search.

**The combination convention.** Most modern serving stacks (vLLM, SGLang, TGI, the OpenAI Chat Completions API) apply: temperature → top-k → top-p → min-p → multinomial sample. Repetition / frequency / presence penalties enter as additive logit biases *before* temperature. Don't accidentally apply top-k after top-p — you'll either undo the nucleus or oversample.

---

## 3. Why beam search degenerates in chat — the Holtzman 2019 result

Beam search ([[beam-search]]) keeps the top-`B` partial sequences at each step, scored by cumulative log-probability. For translation, summarization, and grammar-constrained tasks it works fine: the highest-likelihood sequence really is close to the right one. For open-ended generation, Holtzman et al. 2019 ([[neural-text-degeneration]]) showed it catastrophically fails — beam outputs become "I don't know. I don't know. I don't know..." or "the the the the".

The diagnosis: human text does **not** sit on the maximum-likelihood path at every step. The probability distribution over human continuations is high-entropy and the unreliable tail has a lot of mass. Beam search greedily compounds tiny per-step likelihood advantages, eventually getting trapped in repetitive attractors that have high *local* likelihood (`P("the" | "the")` is huge) but vanishing global plausibility.

Holtzman's fix — nucleus sampling — is exactly the `top_p` block above. The cumulative-mass truncation:

```
V_p = smallest set such that Σ_{x∈V_p} P(x|context) ≥ p
sample uniformly from renormalized V_p
```

…matches human-text entropy at every step. When the model is confident (`P(next) = 0.95`), `V_p` is one token (effectively greedy). When the model is uncertain, `V_p` opens up. Beam search has no analogous mechanism — its diversity is bounded by `B` and concentrated on the most-likely paths, exactly the wrong place.

**Practical rule.** Use beam (`num_beams=4`, `length_penalty=1.0`) for: machine translation, abstractive summarization, code completion with a known target length. Use sampling (`top_p=0.95`, `temperature=0.8`) for everything else — chat, story, free-form Q&A, agent reasoning. The hosted APIs (OpenAI, Anthropic) don't even expose beam search; it's a deliberate choice.

---

## 4. Streaming + token accounting at the API boundary

From the engine's perspective, decode produces one token at a time anyway — streaming just exposes that to the client over Server-Sent Events (SSE) rather than blocking on the whole response. [[openai-streaming-and-token-usage]] documents the canonical contract:

```
event: response.created
event: response.output_text.delta    {"delta": "Hello"}
event: response.output_text.delta    {"delta": " world"}
...
event: response.completed            {"usage": {"input_tokens": 12,
                                                 "output_tokens": 47,
                                                 "total_tokens": 59}}
```

For the Chat Completions endpoint, `stream_options: {"include_usage": true}` makes the final chunk carry the `usage` object. Without it, you have to count tokens client-side or call `/usage` endpoints out-of-band.

**The three things that trip people up.**

1. **`finish_reason`.** Decoded loops have multiple terminating conditions — EOS token, max_tokens cap, stop-string match, content-filter trip. Always read `finish_reason` (`"stop"`, `"length"`, `"content_filter"`, `"tool_calls"`); the same partial text means different things depending on which fired.
2. **Interrupted streams may not deliver the final `usage` chunk** ([[openai-streaming-and-token-usage]] explicit warning). If a client disconnects mid-stream, persist tokens received separately from your billing record; reconcile later via request-id.
3. **Token usage ≠ character count.** The tokenizer rules govern the input/output counts. A 200-character English prompt is ~50 tokens; the same 200 characters of Korean might be 150 tokens. Always count with the model's own tokenizer.

The server side of streaming sits on a continuously-batched decode loop (ch-04). Each new request joins the batch at its next decode step; finished requests leave. Streaming and continuous batching are the matched pair that makes the SSE contract work at scale.

---

## 5. Constrained decoding: structure as a logit mask

"Output JSON" as a prompt instruction is unreliable — the model occasionally emits trailing commas, missing braces, or freeform prose around the JSON. The robust fix is to enforce structure at decode time: compile the schema into a finite-state machine, and at every step set `logits[invalid_tokens] = -inf` before sampling. [[structured-generation-constrained-decoding]] is the synthesis card.

The runtime loop:

```python
def constrained_decode(logits, state, grammar):
    valid_token_ids = grammar.valid_next_tokens(state)       # finite-state lookup
    mask = torch.full_like(logits, -float("inf"))
    mask[valid_token_ids] = 0.0
    masked_logits = logits + mask                            # invalid → -inf
    next_id = sample(masked_logits, ...)                     # normal sampler
    state = grammar.advance(state, next_id)
    return next_id, state
```

**Implementation taxonomy.**

- **Regex / JSON schema** → finite automaton; one-table lookup per step. Outlines, XGrammar, llama.cpp grammars.
- **EBNF / CFG** → pushdown automaton (stack state). Slower to advance per step, broader expressivity. XGrammar precompiles to mostly-FSA fast paths.
- **Cached precompile** — modern grammar backends (XGrammar 2024) precompile schemas and reuse the FSA across requests, dropping per-step overhead from ~10 ms to ~10 μs.

**Three pitfalls.**

1. **Tokenizer mismatch.** Grammars are character-level; logits are token-level. A grammar that accepts `"name":` may not accept the BPE token `"name":` (with both chars in one token). Mature backends pre-resolve this with a tokenizer-aware FSA.
2. **Quality drop on heavily constrained outputs.** Forcing the model down low-probability paths can collapse semantic quality even when syntax is perfect. Validate semantics separately.
3. **Streaming interaction.** Partial JSON streamed mid-decode is not parseable. Clients should buffer until `finish_reason` or use streaming-JSON parsers.

vLLM exposes this via `guided_json` / `guided_regex` / `guided_grammar` ([[vllm-structured-output]]); SGLang exposes it via its DSL's `regex=...` constraint ([[sglang-structured-output]]). Both run XGrammar by default in late 2024+.

---

## 6. Putting it all together: a complete inference event

A single user request `"What's 2+2?"` traces through the stack like this:

```
t=0     POST /v1/chat/completions with stream=true
t=2ms   Server tokenizes, admits request to engine
t=4ms   Scheduler adds request to next-step batch (ch-04)
t=12ms  PREFILL forward pass: K, V written to KV-cache blocks (ch-03, ch-06)
t=13ms  TTFT — first decode logits ready; sampler → "4"; SSE delta emitted
t=14ms  DECODE step 2: logits ready; sampler → "."; SSE delta emitted
t=15ms  DECODE step 3: logits ready; sampler → EOS; finish_reason="stop"
t=15ms  Final SSE chunk with usage; request leaves batch, KV blocks freed
```

Every box on that timeline is a separate later chapter:
- "Scheduler adds to batch" → ch-04 (continuous batching) + ch-16 (vLLM internals)
- "K, V written to KV-cache blocks" → ch-03 (formula) + ch-06 (PagedAttention)
- "PREFILL forward pass" → ch-02 (attention math) + ch-11 (FlashAttention)
- "Sampler → token" → this chapter

---

## 7. Cheat-sheet

```
PREFILL:  one fwd pass, S prompt tokens; writes KV for all S; logits[S-1] used.
DECODE:   one fwd pass per output token, 1 token query, reads all KV.
STOP:     EOS id  |  max_new_tokens  |  stop_strings  |  external cancel.

SAMPLERS:
  greedy        argmax(logits)                                 # T=0
  multinomial   sample ∝ softmax(logits / T)
  top-k         keep top k logits, sample
  top-p         keep smallest V s.t. Σ P ≥ p, renormalize, sample
  min-p         drop probs < min_p · max_prob, renormalize, sample
  beam (B>1)    keep B best partial seqs by cumulative log-prob

PIPELINE ORDER:  penalties → temperature → top-k → top-p → min-p → multinomial

DEFAULTS (2026):
  chat:    temperature=0.7, top_p=0.9 or 0.95
  code:    temperature=0.2, top_p=0.95
  trans:   num_beams=4, temperature=0
  extract: temperature=0 (greedy)

STREAMING:
  SSE deltas during decode; final chunk carries usage (if include_usage=true).
  Always check finish_reason; reconcile partial streams by request-id.

CONSTRAINED:
  logits[invalid] = -inf via FSA/PDA derived from schema/regex/CFG.
  XGrammar / Outlines / lmformatenforcer; vLLM `guided_json`, SGLang `regex=`.
```

---

## Connections and what's next

- **[[attention-complexity]] / ch-02** — the `O(L²·d)` prefill vs `O(L·d)` decode asymmetry is what makes this two-phase loop necessary; you can't run prefill 4096 times.
- **[[kv-cache-memory-formula]] / ch-03** — the cache referenced above has bytes `= 2·L·H_kv·d_head·T·B`, and is usually the binding constraint on batch size.
- **[[continuous-batching]] / ch-04** — the single-request loop becomes a multi-request loop where each decode step admits / evicts requests. Streaming is what continuous batching exposes to clients.
- **[[pagedattention]] / ch-06** — the KV cache becomes a paged virtual memory system, breaking the contiguous-allocation assumption of this chapter's pseudocode.
- **[[speculative-decoding]] / ch-14** — relaxes the "one fwd pass per token" invariant by drafting K tokens cheaply and verifying them in one target-model pass.
- **[[vllm-structured-output]] / ch-16** — production wiring of the logit-mask approach inside vLLM's scheduler.

## Further reading

- [[attention-is-all-you-need]] — scaled dot-product attention + causal masking; the architectural skeleton of the loop.
- [[language-models-are-unsupervised-multitask-learners]] — GPT-2; why "everything is next-token prediction" became the universal interface.
- [[neural-text-degeneration]] — Holtzman 2019; the empirical reason chat ≠ beam search.
- [[hf-generation-strategies]] — the `GenerationConfig` reference; pragmatic parameter recipes.
- [[openai-streaming-and-token-usage]] — the canonical SSE + usage contract.
- [[structured-generation-constrained-decoding]] — XGrammar / Outlines / Guidance; runtime mask synthesis.

## Companion visualization

**[figures/sampling-distribution.html](figures/sampling-distribution.html)** — interactive slider showing how `temperature ∈ {0.2, 0.7, 1.0, 1.5}` and `top_p ∈ {0.5, 0.9, 0.95, 1.0}` reshape an example next-token distribution. Use it to internalize why low-T greedy and high-T nucleus produce completely different texts from identical logits.
