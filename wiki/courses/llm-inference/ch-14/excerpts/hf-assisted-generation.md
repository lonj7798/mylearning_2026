---
chapter: ch-14
course: llm-inference
phase: read
excerpt_of: "Assisted Generation: A New Direction Toward Low-Latency Text Generation (HuggingFace 2023)"
source_url: https://huggingface.co/blog/assisted-generation
created_at: "2026-05-21"
---

# Excerpt: HuggingFace Assisted Generation — The Reference API

**Authors:** Joao Gante (HuggingFace)
**Year:** 2023 (blog), 2024+ (docs)
**Venue:** HF blog + Transformers documentation
**URLs:** https://huggingface.co/blog/assisted-generation ; https://huggingface.co/docs/transformers/assisted_decoding
**Raw-data source:** [[raw-data/hf-assisted-generation]]

---

## The user-facing API

The entire interface is one kwarg:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", device_map="auto")
draft  = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",  device_map="auto")
tok    = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

inputs = tok("Explain RoPE positional encoding.", return_tensors="pt").to("cuda")
out = target.generate(
    **inputs,
    assistant_model=draft,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)
```

That's it. No new loop, no custom code. The `assistant_model` kwarg routes `generate()` through the assisted-decoding code path, which implements the Leviathan acceptance rule under the hood.

---

## The internal loop (`assisted_decoding`)

From `transformers/generation/utils.py::_assisted_decoding`:

```
1. Until max_new_tokens:
2.   K = self.generation_config.num_assistant_tokens   # default 5
3.   candidate_input_ids = draft.generate(..., max_new_tokens=K)
4.   candidate_logits = draft logits at each candidate position
5.   target_outputs = target(input_ids + candidate_input_ids)
6.   For each position i in 1..K:
7.     accept_prob = min(1, p_target(x'_i) / p_draft(x'_i))
8.     if uniform() <= accept_prob:
9.       commit x'_i
10.    else:
11.      x_residual = sample_from((p_target - p_draft).clamp(min=0))
12.      commit x_residual
13.      break  # discard remaining candidates
14.  if all K accepted:
15.    commit bonus token from target_logits[K]
16. Return committed sequence
```

Pure functional implementation — no custom kernels, no in-place hacks. ~200 lines.

---

## Constraints (from the docs)

- **`batch_size == 1`** — assisted generation does not currently support batched inputs. This is the single biggest limitation for serving.
- **Same tokenizer required** — `assistant_model.tokenizer` must equal `model.tokenizer`. Otherwise positions don't align.
- **No streaming mid-draft** — streaming yields at the end of each round (every ~5 tokens), not per-token.
- **Greedy/sampling consistency** — when `do_sample=False`, falls back to the greedy rule (target argmax vs drafter token).

---

## Universal Assisted Decoding (2024 addition)

When tokenizers don't match (e.g. drafting Llama with a Mistral assistant), HF introduced **universal assisted decoding**: the drafter generates in its own tokenizer, the output is re-tokenized into the target's vocabulary at the boundary, and partial-token mismatches are handled by accepting only complete-token prefixes.

Cost: ~20-30% of the speedup is lost to re-tokenization fragmentation. Generally not worth it unless you really can't get a tokenizer-compatible drafter.

---

## Prompt Lookup Variant (no drafter)

The simplest drafter is no drafter — search the prompt for n-grams matching the current suffix:

```python
out = target.generate(
    **inputs,
    prompt_lookup_num_tokens=10,
    max_new_tokens=512,
)
```

The "drafter" is `transformers/generation/candidate_generator.py::PromptLookupCandidateGenerator`:

```
1. For ngram_size in range(max, min, -1):  # try longest match first
2.   Search input_ids for ngram_size-length suffix that matches current tail
3.   If found: return the following 10 tokens as candidate
4.   Else: continue
5. If no match: return []  # skip drafting this round
```

Acceptance rate is workload-dependent — high (~0.9) on summarization / code editing / RAG with quoted text; low (~0.2) on creative open-ended generation. See [[prompt-lookup-decoding]] (ch-15) for the deeper write-up.

---

## Reported speedups (from the blog)

- **codegen-mono-350M drafting codegen-mono-2B**: 1.5-2× on code-generation prompts.
- **opt-125M drafting opt-30B**: 2-3× on chat prompts at temperature 0.7.
- **Up to 10×** in cherry-picked greedy-decoding scenarios with high acceptance.

The 10× claim is for the regime where the drafter is essentially perfect (e.g. drafting a known continuation) — not typical production speedups.

---

## Connections

- [[excerpts/leviathan-2023]] — the acceptance rule HF implements.
- [[excerpts/speculative-decoding]] — the predecessor algorithmic skeleton.
- [[ch-15]] — Medusa, EAGLE, Lookahead, PLD, MTP all build on this skeleton.
- [[ch-14]] — parent chapter.
