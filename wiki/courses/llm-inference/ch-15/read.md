<!-- chapter: ch-15
     phase: decoding-acceleration
     title: Multi-Head + Feature-Level + N-Gram Draft Methods
     sources: [[medusa]], [[eagle]], [[eagle-2]], [[lookahead-decoding]], [[prompt-lookup-decoding]], [[self-speculative-decoding]], [[multi-token-prediction-inference]]
     figures: figures/draft-method-comparison.html
-->

# Chapter 15 — Multi-Head + Feature-Level + N-Gram Draft Methods (Medusa / EAGLE / Lookahead / PLD / MTP)

> **Core insight.** The 2024 speculative-decoding frontier eliminates the *separate draft model* of [[speculative-decoding]] (ch-14). The drafter becomes either (a) **extra heads on the target itself** (Medusa, MTP), (b) a **feature-level predictor on the target's hidden states** (EAGLE / EAGLE-2), (c) **the target run in parallel via Jacobi iteration** (Lookahead), or (d) **prompt n-gram lookup** (PLD). All four collapse the drafter cost ratio `c → ~0`, so the speedup formula `S = (1−α^{K+1})/((1−α)(1+Kc))` becomes dominated by `α` and `K` alone. Result: 3-4× TPOT speedup on chat, 2-3× on creative.
>
> **Guideline.** For self-hosted production: EAGLE-2 + dynamic draft tree is the current pareto winner (3-4× lossless TPOT speedup). For zero-effort speedup on RAG/code: PLD with `prompt_lookup_num_tokens=10`. For MoE models you control training of: native MTP heads (DeepSeek V3 ships this). Medusa is the simpler-to-train baseline; Lookahead is the no-extra-anything fallback.

---

## Why this chapter exists

[[speculative-decoding]] ch-14 established that lossless draft-and-verify is real, and gave you the algorithm box + the acceptance rule. But the *practical bottleneck* of vanilla spec-dec is the drafter: loading a second model takes memory, the drafter's tokenizer must match the target's, and the drafter forward pass adds wall-clock per round (~5-15% of target cost). At scale, those frictions make spec-dec a "sometimes worth it" feature.

2024's contribution is to make spec-dec almost-always worth it by killing the separate drafter. The five families covered here all do that, but they do it differently:

| Family | Drafter is | Trains? | Best for |
|--------|-----------|---------|----------|
| **Medusa** | extra LM heads on target trunk | yes (head-only or joint) | self-hosted, training-friendly |
| **EAGLE / EAGLE-2** | tiny feature predictor on top of target's penultimate layer | yes (small feature module) | best lossless speedup (3-4×) |
| **Lookahead** | the target itself, run in parallel via Jacobi iteration | no | when no training is possible |
| **PLD** (prompt lookup) | n-gram match in prompt | no | RAG / code / summarization |
| **MTP** (Multi-Token Prediction) | extra heads trained as a pretraining objective | yes, at pretrain | model-builders (DeepSeek V3) |

Three things to walk away with:

1. The mechanism of each — how the drafter is constructed and how verification works (especially **tree attention** which Medusa and EAGLE share).
2. The acceptance-rate / cost tradeoff for each: who wins on what workload.
3. When to pick which in production.

---

## 1. Medusa — multiple decoding heads on the target

[[medusa]] (Cai et al. 2024, Together AI) is the simplest "no drafter" approach. Take a pretrained target LLM; freeze the trunk; train K extra LM heads, where head `k` predicts the token at offset `+k+1` from the current position. At inference, the K heads emit K candidate tokens in parallel — they *are* the drafter.

### 1.1 Architecture

```
Target trunk → hidden_state h_t
                ↓
        ┌──────┼──────┬──────┐
        ↓      ↓      ↓      ↓
       head_0 head_1 head_2 head_3
        ↓      ↓      ↓      ↓
    token_t+1 t+2   t+3    t+4
```

Each head is a small residual block + LM-head-style projection. Total params added: ~5-15% of the target depending on K and head width.

The K heads run in **one forward pass** of the target trunk — no autoregression in the drafter — so the drafter cost is essentially zero (`c ≈ 0.02`).

### 1.2 Tree attention — handling top-k per head

Medusa doesn't just emit *one* candidate per head. It emits *top-k* candidates per head, producing a *draft tree*:

```
              h_t
               │
   ┌───┬───┐       (head 0 emits top-3 candidates for t+1)
   A   B   C
  ┌┴┐ ┌┴┐ ┌┴┐    (head 1 emits top-3 for t+2 conditional on each t+1)
  A1 A2 B1 B2 ... 
```

Each tree path is a candidate continuation. Verification runs the target on the *concatenated tree* with a **tree-attention mask** that ensures each path's tokens only attend to its own ancestors (not to other paths' siblings):

```
   tokens in tree:    [A, B, C, A1, A2, B1, B2, C1, C2, ...]
   tree mask: A1 attends to {A, prompt};   A2 attends to {A, prompt}
              B1 attends to {B, prompt};   ...
```

A single target forward pass over the tree validates all paths simultaneously. The accepted prefix is the longest path that survives the verification + acceptance rule (Leviathan-style).

### 1.3 Medusa-1 vs Medusa-2

- **Medusa-1**: train only the K heads, freeze the trunk. Cheap; works for any pretrained target. Accepts ~2-3 tokens per round on average.
- **Medusa-2**: jointly train heads + finetune trunk. ~1-2% perplexity regression on the base model, but accepts ~3-4 tokens per round. Better speedup but you must serve a slightly modified model.

### 1.4 Speedup numbers

From the Medusa paper (Llama-2-7B, Vicuna-7B/13B, MT-Bench):

| Setup | Speedup vs vanilla AR |
|-------|------------------------|
| Medusa-1 (Vicuna-7B) | 2.05× |
| Medusa-1 (Vicuna-13B) | 2.18× |
| Medusa-2 (Vicuna-7B) | 2.46× |
| Medusa-2 (Vicuna-13B) | 2.71× |

Speedup is roughly batch-size-invariant up to batch=8; degrades past that as target verification becomes compute-bound.

### 1.5 Where Medusa loses

Medusa heads are **independent** — head `k` predicts t+k from `h_t` without conditioning on head `k-1`'s output. This caps acceptance because head 3's prediction can't refine itself based on head 2's commit. EAGLE fixes this.

---

## 2. EAGLE — feature-level draft on hidden states

[[eagle]] (Li et al. 2024, Zhejiang University) addresses Medusa's independence bottleneck by making the drafter *autoregressive at the feature level*. Instead of predicting K independent tokens from one hidden state, EAGLE's drafter predicts the *next hidden feature* from the current one — and then uses the target's LM head to turn that feature into a token. Then it predicts the *next-next feature* conditioned on the just-committed one. And so on.

### 2.1 The two-tier insight

Why feature-level instead of token-level? Two reasons:

1. **Features are smoother than tokens.** Token sequences have hard, discrete distribution shifts (every sampled token jolts the distribution). Hidden features evolve more continuously, so a small predictor can learn the transitions accurately.
2. **The one-step-ahead token resolves uncertainty.** Predicting feature `f_{t+1}` purely from `f_t` is ambiguous because we don't know the sampled token `x_{t+1}`. EAGLE conditions on the just-sampled token (one step ahead at the token level), which collapses the entropy.

```
Drafter step k:
  input:  f_t, x_{t+1..t+k}
  output: f_{t+k+1}  (predicted hidden feature)
  then:   x_{t+k+1} = target_LM_head(f_{t+k+1})
```

### 2.2 What the drafter actually is

A small autoregressive transformer with ~1 decoder layer + a feature-projection MLP, total params ~0.2-0.5B for a 70B target. Trained on (target_features, target_tokens) pairs distilled from the target on a calibration corpus (~1B tokens).

The drafter shares the **target's LM head** — no separate head, no tokenizer alignment problem. Cost: `c ≈ 0.02` (similar to Medusa).

### 2.3 Speedup numbers

From the EAGLE paper (Vicuna, LLaMA-2-Chat):

| Setup | Speedup vs vanilla AR | vs Medusa |
|-------|------------------------|-----------|
| EAGLE on Vicuna-7B | 2.7× | 1.3× |
| EAGLE on Vicuna-13B | 2.9× | 1.3× |
| EAGLE on LLaMA-2-Chat-13B | 3.0× | 1.4× |
| EAGLE on LLaMA-2-Chat-70B | 3.0× | 1.3× |

EAGLE is the lossless-spec-dec champion when you can afford the drafter training.

---

## 3. EAGLE-2 — dynamic draft trees

[[eagle-2]] (Li et al. 2024, same team) replaces EAGLE's fixed-shape draft tree with a *context-adaptive* tree. The insight: token acceptance probabilities vary by *context*, not just by tree position. So allocate tree expansion toward branches the drafter is confident about, prune low-confidence branches.

### 3.1 The dynamic tree algorithm

```
1. Maintain a frontier of (path, cumulative_confidence) tuples.
2. At each expansion step:
   a. Pop the top-N highest-confidence frontier nodes.
   b. Expand each by drafter's top-K predictions.
   c. Score new children by drafter confidence.
   d. Push into frontier.
3. Stop when tree has M total nodes or expansion exhausted.
```

`M` is the verification budget (a parameter — usually ~60 nodes). Static trees waste budget on low-confidence branches; dynamic trees spend it where acceptance is likely.

### 3.2 Speedup numbers

From the EAGLE-2 paper (same models):

| Setup | Speedup vs vanilla AR | vs EAGLE-1 |
|-------|------------------------|------------|
| EAGLE-2 on Vicuna-7B | 3.5× | 1.30× |
| EAGLE-2 on Vicuna-13B | 3.8× | 1.31× |
| EAGLE-2 on LLaMA-2-Chat-70B | 4.26× | 1.42× |

The 3.05-4.26× range is the current published frontier for lossless spec-dec on Llama-style models.

### 3.3 Why dynamic trees help

Static trees with M nodes have fixed branching. If position 1 is highly uncertain (drafter low-confidence) and position 2 is highly certain (drafter very high-confidence given the right t+1), the static tree wastes most M on diverse t+1 candidates when it should spend on deep extensions of the few high-confidence t+1s.

Dynamic trees push budget where it matters. The 20-40% improvement over EAGLE-1 is exactly this allocation effect.

---

## 4. Lookahead Decoding — Jacobi-style parallel n-gram

[[lookahead-decoding]] (Fu et al. 2024, LMSYS) is the *no-extra-model, no-extra-training* alternative. It uses **Jacobi iteration** — a classical fixed-point method — to find the next K tokens in parallel from the target itself.

### 4.1 The Jacobi formulation

Autoregressive generation is computing the fixed point of `f(x_1, ..., x_K) = (LM(x_0), LM(x_0, x_1), ..., LM(x_0, ..., x_{K-1}))`. Each call to `f` improves the guess. Naive iteration is the sequential AR loop — one position per pass.

Jacobi iteration is: start with a *random* guess for K future tokens; run the target on the guess; the model emits K outputs (next-token distribution per position); use those as the next guess; repeat until convergence. Converged outputs are the AR decoding.

### 4.2 The N-gram pool

In practice, Jacobi converges fast on positions where the answer is "obvious" (the first 1-2 are usually right within 1-2 iterations). Lookahead Decoding maintains a *pool* of n-grams discovered during Jacobi iteration:

```
Each forward pass:
  - Compute lookahead branch: Jacobi step on a window of W future positions.
  - Compute verification branch: check accumulated n-gram candidates against actual model output.
  - Accept any n-gram that matches; refresh others.
```

Both branches share the **same target forward pass** — different attention masks across the input sequence.

### 4.3 Speedup numbers

From the Lookahead paper:

| Setup | Speedup vs vanilla AR |
|-------|------------------------|
| LLaMA-2-7B (chat) | 1.5-1.8× |
| LLaMA-2-13B (chat) | 1.6-1.9× |
| CodeLlama-7B (coding) | 2.3-2.5× |

Lower than Medusa/EAGLE because Jacobi doesn't have a *learned* drafter — the n-gram discovery is opportunistic. Win: no training, no extra weights, no tokenizer issues, works on any AR LM.

---

## 5. Prompt Lookup Decoding (PLD) — n-gram from the prompt

[[prompt-lookup-decoding]] (HF Transformers 2024, contributor Apoorv Saxena) is the most lightweight method in this chapter: the "drafter" is just `string.find(suffix)` on the prompt.

### 5.1 The mechanism

```
For each decode step:
  1. current_suffix = last n tokens of the generated sequence
  2. for ngram_size in [max_n, max_n - 1, ..., min_n]:
       match = find_ngram(prompt, current_suffix[-ngram_size:])
       if match found:
         return prompt[match_end : match_end + K]   # K candidates
   3. Otherwise: skip drafting this round (no candidates)
```

Cost: pure CPU string lookup, microseconds per step. `c ≈ 0.001`.

### 5.2 Why this works for specific workloads

Many production workloads have a *copy-heavy* output distribution:

- **RAG / summarization** — model often quotes the source verbatim.
- **Code editing / refactor** — output is mostly the input file with small changes.
- **Long-document Q&A** — answers reference exact spans.
- **Document translation** — proper nouns, numbers, code snippets copied through.

For these, the prompt *is* the drafter — `α ≈ 0.85-0.95`, speedup 2-3× with zero infrastructure.

### 5.3 vLLM and HF API

```python
# HF Transformers
out = model.generate(input_ids, prompt_lookup_num_tokens=10, max_new_tokens=512)

# vLLM (V1)
llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    speculative_config={
        "method": "prompt-lookup",
        "num_speculative_tokens": 10,
        "prompt_lookup_max": 5,    # max n-gram size to try
        "prompt_lookup_min": 2,    # min n-gram size
    },
)
```

**When PLD doesn't help**: creative chat, open-ended generation. α drops to 0.1-0.2. Switch to a drafter-based method (or disable spec-dec entirely).

---

## 6. Self-Speculative Decoding — drafter is the target with skipped layers

[[self-speculative-decoding]] (Zhang et al. 2023) makes the drafter the *same model* but with selected intermediate layers skipped. No extra weights, no extra training.

### 6.1 Layer-skipping draft

For a 70-layer Llama-2-70B, define a skip set `S ⊂ [1..70]` (typically ~half the layers). The drafter runs only `[1..70] \ S`; the verifier runs all 70.

```
Draft forward:   skip layers in S, get faster but lower-quality logits
Verify forward:  full model, all 70 layers, parallel over K positions
```

Drafter cost: `c ≈ 0.4-0.6` depending on how many layers are skipped. Much higher than Medusa/EAGLE — this is the cost of using the full-rank LM as drafter.

### 6.2 Where layers can be skipped

The paper analyzes which layers contribute least to next-token prediction (via gradient-based importance). For Llama-2, mid-network layers (15-50) tolerate skipping; early and final layers are critical.

### 6.3 Speedup

Reported up to **1.99×** on LLaMA-2 variants. Lower than EAGLE but unique selling point: no extra weights, no training, works as a drop-in for any AR model.

### 6.4 When to use

- Memory-constrained: can't fit a separate drafter.
- No training pipeline: can't add Medusa heads or EAGLE module.
- Acceptable to give up some speedup for zero infrastructure.

In practice, layer-skipping is rare in production today — EAGLE's training cost is modest enough that most people pay it.

---

## 7. Multi-Token Prediction (MTP) — native speculative heads

[[multi-token-prediction-inference]] (Gloeckle et al. 2024, FAIR) made MTP famous as a *pretraining objective* — train the model to predict `n` future tokens per position via `n` independent output heads. Reported gains: ~3× inference speedup *and* better code-generation quality.

### 7.1 The MTP objective

For each position `t`, compute loss over `n` future positions:

```math
\mathcal{L} = -\sum_{i=1}^{n} \log p(x_{t+i} \mid x_{\le t}; W_{\text{head}_i})
```

Heads `1..n` share the trunk; each is a small projection (one linear + softmax). The auxiliary heads improve sample efficiency, especially for code and algorithmic tasks (MBPP, HumanEval).

### 7.2 At inference

At each step, all `n` heads emit candidate tokens for `t+1..t+n` in parallel. Verification (like Medusa) is one target forward over the concatenated candidates, with the longest-accepted prefix committed.

The difference from Medusa: heads are trained as a **first-class pretraining objective**, not as a post-hoc add-on. This gives them more capacity-aware predictions and higher acceptance rates.

### 7.3 DeepSeek V3 — production-deployed MTP

DeepSeek V3 ([[ch-20]] preview) uses MTP natively. The 671B MoE has an MTP head trained alongside the main next-token objective. At inference, MTP enables ~3× TPOT speedup on serving stacks that integrate it (vLLM has MTP support for DeepSeek-V3 via `--speculative-config '{"method":"deepseek_mtp"}'`).

### 7.4 Why MTP > Medusa for serving builders

- **Heads are co-pretrained**, so they're already aligned with the trunk distribution. No separate finetuning step.
- **Better inductive biases** from the pretraining objective improve token-level accuracy on hard positions (code, structured output).
- **No quality regression** on the base model — both objectives are jointly optimized.

Cost: you have to choose this *at pretraining time*. Existing checkpoints can't be retrofitted (you'd be doing Medusa-style head training).

---

## 8. The decision tree

```
Need spec-dec speedup?
├── No training pipeline available?
│   ├── Prompt has copy-heavy content?    → PLD (zero infra, 2-3× on RAG)
│   ├── No extra memory?                  → Lookahead (Jacobi, 1.5-2×)
│   └── Some extra memory ok?             → Self-spec (layer-skip, ~1.5×)
├── Have training pipeline?
│   ├── Want max lossless speedup?        → EAGLE-2 (3-4×, dynamic tree)
│   ├── Want simpler training?            → Medusa-1 or Medusa-2 (2-2.7×)
│   └── Building a model from scratch?    → MTP as pretraining objective
```

In OSS serving practice (vLLM, SGLang) as of 2026:
- **EAGLE-2** integrations are stable for Llama, Qwen, Mistral.
- **PLD** is the default for any workload with copy-heavy outputs.
- **MTP** is supported for DeepSeek-V3 specifically.
- Medusa is supported but losing share to EAGLE-2.

---

## 9. Pitfalls

- **Tree attention complexity.** A naive tree-attention implementation has `O(K²)` overhead per node from the tree mask. Use FlashAttention's `attn_mask` variant or specialized kernels (FlashInfer ships a tree-attention kernel).
- **Medusa head quality varies wildly across positions.** Head_0 (next token) is easy; head_4 (4 tokens ahead) is hard. Position-aware tree pruning helps.
- **EAGLE feature alignment.** EAGLE assumes the drafter feature is in the *same* representation space as the target's penultimate hidden state. If you finetune the target after training EAGLE, you must retrain EAGLE.
- **Lookahead's Jacobi convergence is workload-dependent.** Code converges fast; creative writing barely converges. Don't deploy lookahead without measuring.
- **PLD on creative chat is a slowdown.** Always check `α`; disable PLD if it drops below 0.4.
- **MTP requires careful pretraining**. Just bolting on heads at finetune time gives you Medusa-1 (cheaper but lower α). MTP's value is the joint pretraining.
- **Dynamic tree budget M.** EAGLE-2's M=60 is the published default; tuning M down trades acceptance for less verification cost. At very high batch sizes, smaller M wins.

---

## 10. Practitioner's cheat-sheet

```python
# vLLM — EAGLE-2 (the modern default)
from vllm import LLM, SamplingParams
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=8,
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-70B",
        "num_speculative_tokens": 5,
        "draft_tree_choices": "mc_sim_7b_63",
    },
)

# vLLM — prompt lookup (zero training, great for RAG/code)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "method": "prompt-lookup",
        "num_speculative_tokens": 10,
    },
)

# vLLM — Medusa
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "method": "medusa",
        "model": "FasterDecoding/medusa-llama-3.1-70b",
        "num_speculative_tokens": 5,
    },
)

# vLLM — DeepSeek-V3 MTP
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    speculative_config={
        "method": "deepseek_mtp",
        "num_speculative_tokens": 1,
    },
)
```

---

## Connections and what's next

- **[[speculative-decoding]] / ch-14** — foundation: acceptance rule, residual sampling, speedup math.
- **[[continuous-batching]] / ch-04** — interacts with all spec-dec methods via per-step variable token commit.
- **[[pagedattention]] / ch-06** — speculative KV writes go into temp blocks; rejection frees them cleanly.
- **[[vllm-scheduler]] / ch-16** — vLLM's spec-dec integration via `speculative_config`.
- **[[ch-20]] (production reports)** — DeepSeek-V3 with native MTP, Qwen-3 with EAGLE-style serving.

## Further reading

- [[medusa]] — Cai et al. 2024, multi-head spec-dec.
- [[eagle]] — Li et al. 2024, feature-level drafter.
- [[eagle-2]] — Li et al. 2024, dynamic draft trees.
- [[lookahead-decoding]] — Fu et al. 2024, Jacobi parallel decoding.
- [[prompt-lookup-decoding]] — HF Transformers 2024, prompt n-gram.
- [[self-speculative-decoding]] — Zhang et al. 2023, layer-skipping.
- [[multi-token-prediction-inference]] — Gloeckle et al. 2024, native MTP heads.

## Companion visualization

**[figures/draft-method-comparison.html](figures/draft-method-comparison.html)** — interactive table comparing α, K, c, and speedup across Medusa / EAGLE / EAGLE-2 / Lookahead / PLD / MTP on chat vs code vs RAG workloads.

## Excerpts

- [excerpts/medusa.md](excerpts/medusa.md) — multi-head architecture, tree attention mask, Medusa-1 vs Medusa-2.
- [excerpts/eagle.md](excerpts/eagle.md) — feature-level draft, one-step-ahead uncertainty resolution.
- [excerpts/eagle-2.md](excerpts/eagle-2.md) — dynamic draft tree algorithm, context-aware allocation.
- [excerpts/lookahead-prompt-lookup.md](excerpts/lookahead-prompt-lookup.md) — Jacobi parallel decoding + PLD n-gram drafter.
- [excerpts/multi-token-prediction.md](excerpts/multi-token-prediction.md) — MTP as pretraining objective + DeepSeek V3 deployment.
