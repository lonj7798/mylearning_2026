<!-- chapter: ch-22
     phase: lab-capstone
     kind: capstone
     title: Capstone — Reproduce a Serving Optimization From Scratch
     deps: [ch-06, ch-07, ch-09, ch-14, ch-15]
     sources: [[pagedattention]], [[sglang-radixattention]], [[distserve]], [[speculative-decoding]], [[eagle]], [[mooncake]]
     capstone_for: llm-inference
-->

# Chapter 22 — Capstone: Reproduce a Serving Optimization From Scratch

> **Capstone objective.** Pick one serving-time optimization from the 2023–2025 frontier — **PagedAttention**, **RadixAttention**, **Speculative Decoding**, **EAGLE**, or **Disaggregated Prefill/Decode** — and *implement it from the paper, not from the author's repo*. Run it on Llama-3-8B (full budget) or Qwen-1.8B (constrained), reproduce the paper's headline number (e.g., PagedAttention 96 % utilisation, RadixAttention 5× shared-prefix throughput, SpecDec 2–3× TPOT, EAGLE 3× TPOT, DistServe goodput delta), and write a reproduction memo that names the gap between your numbers and theirs — or states that you matched within 10 %.
>
> **Guideline.** The deliverable is the memo. A clean reproduction proves you understand the recipe. A *failed* reproduction with a clean diagnosis is often more valuable, because it surfaces what the paper didn't tell you. Both outcomes pass; only "I couldn't reproduce, maybe the paper is wrong" fails.

---

## Why this capstone exists

The lab ([[ch-21]]) put you in front of two black-box frameworks and asked you to *measure* them. The capstone asks something harder: read a paper that defines a serving primitive, implement that primitive, and verify it does what the paper claims on real hardware.

This is the day-to-day shape of serving research. Every quantization, scheduling, caching, or speculation paper looks reproducible until you try. The gap between "the paper says 2.5× speedup" and "I see 1.4× on my hardware" is where the field's actual progress lives, because that gap is always traceable to a specific (hardware / batch shape / model / hyperparameter) interaction the paper didn't fully cover.

The five method options below span the three surface areas of the course: KV-cache engineering (PagedAttention, RadixAttention), decode acceleration (SpecDec, EAGLE), and serving architecture (DistServe). Pick the one whose surface area matches where you want to do research.

---

## Method options

Each option has: core mechanism (one paragraph), key reproduction risk (one paragraph), and the target headline number you're trying to match.

### Option 1 — PagedAttention (vLLM)

**Mechanism** (per [[pagedattention]]).
KV cache is split into **fixed-size blocks** of `B = 16` tokens. Each sequence carries a *block table* — a list of pointers into a global pool of physical blocks. The attention kernel reads K/V through this block-table indirection instead of assuming contiguous per-sequence storage. Memory waste drops from "one half-empty allocation per request" to "at most one half-empty block per request" — i.e. fragmentation is bounded by `B = 16` tokens regardless of request length. Beam search and parallel sampling get *copy-on-write* sharing: sibling sequences point to the same physical blocks until they diverge, at which point only the divergent block is copied.

**Key reproduction risk.** The block-table indirection lives **inside the attention kernel**. A pure-PyTorch reference implementation is correct but ~10× slower than dense attention, so you cannot report throughput numbers from the reference. You need either a Triton kernel (search the vLLM repo for `paged_attention_v2.cu` for the C++ reference; write a Triton equivalent) or you must report only the *memory* metric. The paper's 96 % utilisation is a memory claim, not a kernel-speed claim — it is the reproducible one without writing CUDA.

**Target headline numbers** (from [[pagedattention]]).
- KV-cache memory utilisation: ~96 % (i.e., < 4 % waste from internal fragmentation), vs ~60–80 % for naive contiguous allocators.
- 2–4× throughput improvement vs Orca / FasterTransformer on similar latency targets. (Throughput claim is kernel-quality dependent — see risk above.)

---

### Option 2 — RadixAttention (SGLang)

**Mechanism** (per [[sglang-radixattention]]).
KV-cache prefix reuse organised as a **radix tree** keyed by token sequences. Each cached prefix corresponds to a path from root to a node. A new request walks the tree against its token IDs, finds the longest matching prefix, and reuses those KV blocks — only the unmatched suffix is prefilled fresh. The tree supports `match()`, `insert()`, `evict()` (LRU), and `lock()` (pin while in-flight requests reference a node). Combined with cache-aware scheduling, the scheduler can group requests with overlapping prefixes into the same batch to maximise reuse.

**Key reproduction risk.** Two-layer correctness: (a) the radix-tree data structure must handle prefix-of-prefix and split correctly when two requests diverge mid-prefix, and (b) the cache-aware scheduling must not evict a node still referenced by an in-flight request. The data-structure bug shows up as "first request works, second request gets corrupted KV state." The scheduling bug shows up as silent crashes ~100 requests in. Test the data structure standalone (insert N random prefix strings, verify all match) before hooking into a real model.

**Target headline numbers** (from [[sglang-radixattention]] / SGLang paper).
- **5× throughput** on shared-prefix workloads (chained generation, multi-turn chat with a fixed system prompt) vs vLLM's APC.
- **2× throughput** on standard ShareGPT chat where prefix overlap is modest.
- **20–90 % cache hit rate** depending on workload class.

---

### Option 3 — Speculative Decoding (lossless sampling variant)

**Mechanism** (per [[speculative-decoding]] / [[fast-inference-from-transformers-via-speculative-decoding]]).
A small **draft model** proposes K tokens autoregressively (cheap). The **target model** runs *one* parallel forward pass over the K positions, producing target distributions `p_i` at each position. For each drafted token `x_i` with draft prob `q_i(x_i)` and target prob `p_i(x_i)`:

```
u ~ Uniform(0, 1)
if u <= p_i(x_i) / q_i(x_i):
    accept x_i
else:
    reject; sample new token from the residual max(0, p_i - q_i) / Z; stop accepting
```

If all K drafts accept, the target's *bonus* token (the next prediction at position K+1) is also kept. Expected speedup is `(1 + accepted_run_length) / target_forward_cost`. Typical numbers: K=4, acceptance rate ≈ 0.7 per token, ~2× speedup.

**Key reproduction risk.** The acceptance test must use *probabilities*, not logits. A common bug is `accept if p_logit > q_logit` (wrong: doesn't preserve target distribution). The residual sampler must be `max(0, p - q) / sum(max(0, p - q))`, with explicit re-normalisation; without it you bias toward already-likely tokens. Verify lossless property: with the same seed, generated sequences from speculative decoding should match those from target-only decoding *exactly* (modulo the bonus-token tie-breaking).

**Target headline numbers** (from [[fast-inference-from-transformers-via-speculative-decoding]]).
- **2–3× wall-clock speedup** at T=0 (greedy) for Llama-3-8B as target + a smaller draft (Llama-3.2-1B is the natural pair).
- **Sample-quality identical** to target-only.

---

### Option 4 — EAGLE (feature-level speculative decoding)

**Mechanism** (per [[eagle]]).
Instead of drafting at the *token* level with a separate small LM, EAGLE drafts at the **feature** (penultimate-layer hidden state) level using a single-layer transformer-style draft head trained on top of the target's frozen weights. The draft head takes `(current_token_embedding, previous_hidden_state)` and predicts the *next hidden state* — which the target's LM head then turns into a next-token distribution. Drafting in feature space is easier because the feature distribution is smoother than the token distribution. EAGLE also constructs a **draft tree** (multiple candidate continuations per step) and verifies the whole tree in one target forward pass, increasing the expected accepted-tokens-per-pass.

**Key reproduction risk.** Training the draft head is *not* optional — EAGLE requires a brief training run on the target model's own outputs (~1k steps on a small dataset). Skipping this and using random init gives ~0 % acceptance. The "one-step-ahead token" input is also load-bearing: the draft head conditions on `(token_t, hidden_t-1)` to predict `hidden_t`, which then maps to `token_t+1`. Getting the offsets wrong by one position gives 30–40 % acceptance instead of 80 %.

**Target headline numbers** (from [[eagle]]).
- **3× TPOT speedup** on Llama-2-7B / Llama-2-13B targets.
- ~80 % per-token acceptance rate (vs ~70 % for vanilla SpecDec at similar K).
- **Lossless** sampling.

---

### Option 5 — Disaggregated Prefill / Decode (DistServe-style)

**Mechanism** (per [[distserve]]).
Standard serving co-locates prefill and decode on the same GPU pool. They interfere: prefill is compute-bound (long, batchable), decode is memory-bandwidth-bound (one token at a time, sensitive to per-step latency). DistServe *separates* them: one GPU pool runs only prefill, another runs only decode. After prefill, the KV cache is **shipped** from prefill GPU to decode GPU over the interconnect. Each pool can be sized and parallelised independently for its phase, and SLOs for TTFT (prefill-dominated) and TPOT (decode-dominated) are optimised separately. The catch: KV transfer cost is `2 · layers · kv_heads · head_dim · prompt_len · dtype_bytes` — for Llama-3-8B at 8 k context this is ~1 GB, which is real interconnect time.

**Key reproduction risk.** This is the most *systems-complex* option. You need:
1. Two server processes (or two GPUs in one process) — prefill server and decode server.
2. A KV-transport channel (NCCL p2p, or a TCP fallback for correctness testing).
3. Synchronisation: decode server must wait for the full KV cache before starting decode.
4. The goodput metric, not raw throughput.

Common bug: KV transfer happens but pipelining is wrong — decode server waits for *all* layers' KV before starting, instead of streaming layer-by-layer. The paper's headline goodput is unreachable without overlapping transfer and decode.

**Target headline numbers** (from [[distserve]]).
- **4–8× higher request rate** at fixed TTFT-and-TPOT SLO, vs co-located continuous batching.
- Goodput delta is workload-dependent; tightest SLO + most heterogeneous prompt/output lengths → biggest gain.

---

## Recommendation matrix

If you have 3 days and want clean reproduction: pick **PagedAttention** (memory metric only).
If you want a data-structures challenge: pick **RadixAttention**.
If you want a probability-theory + decoding challenge: pick **Speculative Decoding**.
If you want a small-scale training challenge: pick **EAGLE**.
If you want the systems-engineering challenge: pick **Disaggregated Prefill/Decode**.

The capstone is graded on the reproduction memo, not on the specific method picked.

---

## Full-budget path

**Hardware.** 1 × H100 80 GB (or 2 × H100 for DistServe). Mostly enough for Llama-3-8B; for the EAGLE training step you also want ~50 GB free for activations.
**Model.** `meta-llama/Meta-Llama-3-8B-Instruct` for everything except SpecDec (where target = Llama-3-8B, draft = `meta-llama/Llama-3.2-1B-Instruct`).
**Wall-clock.** 3–5 days end-to-end.

**Per-method workload.**

| Method | Eval workload | Headline metric |
|--------|---------------|-----------------|
| PagedAttention | Mixed-length ShareGPT, 1000 requests | KV-cache memory utilisation (%) |
| RadixAttention | Shared-prefix synthetic: 100 system prompts × 50 user follow-ups each | Throughput (tok/s) vs no-cache baseline |
| SpecDec | ShareGPT chat, T=0 greedy | TPOT speedup vs target-only |
| EAGLE | ShareGPT chat, T=0 greedy | TPOT speedup + acceptance rate |
| DistServe | ShareGPT chat, request-rate sweep, TTFT/TPOT SLO | Goodput@SLO vs co-located baseline |

---

## Resource-constrained path

**Hardware.** 1 × consumer 24 GB GPU (RTX 3090 / 4090 / A5000).
**Model.** `Qwen/Qwen1.5-1.8B-Chat` for everything except SpecDec (target = Qwen1.5-1.8B, draft = `Qwen/Qwen2.5-0.5B-Instruct`).
**Wall-clock.** 2–3 days.

**Caveats.** Qualitative reproduction only. The *shape* of the result should match the paper (e.g., RadixAttention should still beat no-cache by a clear factor); the *absolute* number won't (1.8B has different KV-cost-to-compute ratio than 8B). The memo must say so explicitly.

**The DistServe option needs 2 GPUs** — even at small scale, the prefill/decode split is the entire point. If you only have one GPU, pick a different option.

---

## The reproduction workflow

A six-step skeleton that works for all five method options.

### Step 1 — Read the paper twice

**First pass:** mechanism, one paragraph summary in your own words. If you can't write it from the paper alone, read it again.

**Second pass:** translate every equation in the methods section to a line of pseudocode in a `notes.md` file. The paper-to-pseudocode mapping is what you'll lean on during debugging.

For PagedAttention this means writing out: block-table layout, allocation policy, gather-into-attention. For SpecDec this means writing the acceptance rule + residual sampling out longhand, *with* the normalisation. For EAGLE this means specifying the draft head's input/output dims and the offsets.

### Step 2 — Pure-PyTorch reference (no kernel optimisation)

Write a slow but correct implementation. Use Python loops, `torch.einsum`, and `torch.no_grad()` for everything. Avoid CUDA kernels at this stage; you are testing whether the *algorithm* is right.

For PagedAttention, the reference might be:

```python
import torch
BLOCK = 16

class PagedKVCache:
    def __init__(self, n_layers, n_kv_heads, head_dim, n_blocks, dtype=torch.bfloat16):
        # Global block pool: (n_blocks, BLOCK, n_kv_heads, head_dim)
        self.K_pool = torch.zeros(n_layers, n_blocks, BLOCK, n_kv_heads, head_dim, dtype=dtype, device='cuda')
        self.V_pool = torch.zeros_like(self.K_pool)
        self.free_blocks = list(range(n_blocks))
        self.block_tables = {}   # seq_id -> list of physical block IDs

    def allocate(self, seq_id, n_tokens):
        n_blocks_needed = (n_tokens + BLOCK - 1) // BLOCK
        assigned = [self.free_blocks.pop() for _ in range(n_blocks_needed)]
        self.block_tables[seq_id] = assigned
        return assigned

    def gather_kv(self, seq_id, layer):
        # Return contiguous K, V for attention computation
        blocks = self.block_tables[seq_id]
        K_seq = torch.cat([self.K_pool[layer, b] for b in blocks], dim=0)  # (n_tokens_padded, n_kv_heads, head_dim)
        V_seq = torch.cat([self.V_pool[layer, b] for b in blocks], dim=0)
        return K_seq, V_seq

    def utilisation(self, active_seqs):
        used = sum(len(self.block_tables[s]) for s in active_seqs) * BLOCK
        # Token count actually filled (last block may be partial — bounded by BLOCK-1 waste per seq)
        # ...
```

Slow, but it verifies the indirection works and gives you a memory-utilisation number.

### Step 3 — Hook into the target model

For KV-cache methods (PagedAttention, RadixAttention), monkey-patch the model's KV-cache class. With HF transformers 4.36+, subclass `DynamicCache`:

```python
from transformers import DynamicCache, AutoModelForCausalLM

class PagedDynamicCache(DynamicCache):
    def __init__(self, paged_kv_cache, seq_id):
        super().__init__()
        self.paged = paged_kv_cache
        self.seq_id = seq_id

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Write to paged storage
        self.paged.append(self.seq_id, layer_idx, key_states, value_states)
        # Return all-K, all-V for the attention kernel
        return self.paged.gather_kv(self.seq_id, layer_idx)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                              torch_dtype=torch.bfloat16).cuda()
pkv = PagedKVCache(n_layers=32, n_kv_heads=8, head_dim=128, n_blocks=4000)
out = model.generate(input_ids, past_key_values=PagedDynamicCache(pkv, seq_id=0), max_new_tokens=100)
```

For SpecDec, wrap the generation loop:

```python
def speculative_generate(target, draft, prompt_ids, K=4, max_new=200, T=0.0):
    tokens = prompt_ids.tolist()
    while len(tokens) < len(prompt_ids) + max_new:
        # 1. Draft K tokens autoregressively
        drafts, draft_probs = draft_propose(draft, tokens, K, T)
        # 2. Target one pass over all K+1 positions
        target_logits = target_verify(target, tokens + drafts)
        target_probs = softmax(target_logits, T)
        # 3. Acceptance loop
        accepted = []
        for i, (tok, q, p) in enumerate(zip(drafts, draft_probs, target_probs)):
            u = torch.rand(1).item()
            if u <= p[tok].item() / max(q[tok].item(), 1e-10):
                accepted.append(tok)
            else:
                # Residual sample
                residual = (p - q).clamp(min=0)
                residual = residual / residual.sum()
                new_tok = torch.multinomial(residual, 1).item()
                accepted.append(new_tok)
                break
        else:
            # All accepted; take bonus token from target
            accepted.append(target_probs[K].argmax().item())
        tokens.extend(accepted)
    return tokens
```

For DistServe, wrap the model's prefill and decode paths in two separate processes communicating over `torch.distributed.send` / `recv`.

### Step 4 — Verify correctness

Before benchmarking, verify the method does what it claims:

- **PagedAttention.** Generate 100 tokens from a paged-cache model and a baseline contiguous-cache model on the same prompt with the same seed. Token sequences must match exactly.
- **RadixAttention.** Send two requests with the same 100-token prefix; verify the second one's prefill spends ≤ 1 step on the matched prefix. Then send a third with a *different* prefix; verify the cache splits the tree node and no corruption occurs.
- **SpecDec.** Run greedy generation with target-only and with SpecDec; the two should produce identical token sequences. If not, the acceptance test is buggy.
- **EAGLE.** Same lossless test as SpecDec.
- **DistServe.** Generate one request end-to-end; verify the output matches what a co-located server produces.

If correctness fails, do not proceed to benchmarking. Fix the bug first.

### Step 5 — Run the eval

Apply the [[ch-21]] benchmark harness. Same `benchmark_serving.py` driver, but pointed at your reproduction server. Compare your headline number to the paper's:

| Method | Paper number | Tolerance |
|--------|--------------|-----------|
| PagedAttention | 96 % memory util | ± 2 % |
| RadixAttention | 5× throughput on shared-prefix | ± 20 % |
| SpecDec | 2–3× TPOT speedup | ± 30 % |
| EAGLE | 3× TPOT speedup | ± 30 % |
| DistServe | Goodput delta (workload-specific) | ± 30 % |

KV-cache methods have tight tolerances (the metric is a clean memory measurement). Decode-acceleration methods have looser tolerances (acceptance rate is dataset-dependent). DistServe is the loosest because the result depends heavily on hardware interconnect and request-shape distribution.

### Step 6 — Iterate or diagnose, then write the memo

If your numbers match within the tolerance: write the memo, document the recipe that worked.

If not: walk down the diagnosis tree —

1. Implementation bug? Compare your reference to the paper equations line-by-line.
2. Hyperparameter mismatch? Check the paper's exact `K`, `block_size`, `chunk_size`, `temperature`, etc.
3. Hardware mismatch? Paper used A100; you have H100 or RTX 4090. Interconnect, HBM bandwidth, FlashAttention version all matter.
4. Workload mismatch? Paper's "ShareGPT" filter is not the same as yours. Replicate the exact filter (cleaned, length-filtered, seeded).
5. Quantization mismatch? Paper FP16 vs your BF16 — usually negligible, but worth noting.

State your hypothesis, test it, document the result. The diagnostic process *is* the learning outcome.

---

## Reproduction memo

The memo is the deliverable. Template:

```markdown
# Reproduction Memo — <method name> on <model>

**Author:**           ____
**Method:**           PagedAttention / RadixAttention / SpecDec / EAGLE / DistServe
**Paper:**            <citation + arxiv link>
**Hardware:**         GPU model, count
**Model:**            <model id + commit hash>
**Workload:**         <eval workload, dataset, seed>
**Date:**             ____

## 1. Implementation summary

(One paragraph: what you implemented from the paper, in your own words.
Include the line count of your reference implementation and the key
files. Cite the paper's equations / section numbers you mapped from.)

## 2. Correctness verification

(How did you confirm the method does what it claims, before benchmarking?
For lossless methods, the lossless test. For paged caches, the
identical-generation test. For RadixAttention, the prefix-match test.
For DistServe, the equivalent-to-co-located test.)

## 3. Headline number

| Metric          | Paper number | My number | Δ %  | Within tolerance? |
|-----------------|-------------:|----------:|-----:|:-----------------:|
| (e.g.) TPOT speedup  |        2.5× |     2.1× | -16% |       Yes (±30%)  |

## 4. Where my number diverges (or matches) the paper

(Required even on a successful reproduction. If matched: which paper
hyperparameter mattered most? If not matched: which axis is the gap
on — hardware, batch size, hyperparameter, workload? State the hypothesis,
the experiment that tested it, and the conclusion.)

## 5. What the paper didn't tell me

(The most valuable section. Every reproduction surfaces at least one
implementation decision the paper glossed over. Name yours.)

## 6. What I would do differently next time

(Concrete: which step took longest? Which step was riskiest? What would
you check earlier?)

## 7. Optional stretch goal status

(If attempted: did the PR to vLLM / SGLang land? What was the
maintainers' feedback?)
```

---

## What "reproduction" means

**Reproduction is not "I cloned the author's repo and ran their script."** That's verification.

**Reproduction is "I read the paper, implemented the method, ran it on comparable infrastructure, and got numbers within the paper's claimed bounds — or I can name precisely where my numbers diverge and offer a hypothesis for why."**

Both outcomes pass. The memo for a successful reproduction documents the recipe that worked. The memo for an unmatched reproduction documents the gap and the diagnosis — often the more valuable artifact, because it surfaces what the paper omitted.

Example of a bad memo:
> "I implemented EAGLE but only got 1.5× speedup instead of 3×. Not sure why. Maybe the paper is wrong."

Example of a good memo:
> "I implemented EAGLE on Llama-3-8B and measured 1.6× TPOT speedup (paper claimed 3× on Llama-2-7B). The gap reproduces when I run my implementation on Llama-2-7B (2.7×, within 10 % of paper). I attribute the Llama-3-8B gap to its GQA-8 attention having lower memory-bandwidth pressure than Llama-2-7B's MHA-32 — EAGLE's speedup is bounded by the ratio of target-pass cost to per-token decode cost, and GQA already makes decode cheaper, leaving less to amortise. Recommendation: EAGLE's reported speedups should be retested per attention architecture."

The good memo is publishable as a workshop note (MLSys, NeurIPS Efficient LLM Inference, ICLR Tiny Papers). The bad memo is gossip.

---

## Optional stretch goal

Pick at most **one** stretch goal after the core capstone is done.

### Stretch A — PR your implementation to vLLM or SGLang

Both projects accept implementation contributions for new optimizations *if* they meet the framework's contribution rubric (passing tests, benchmark scripts, an integration test). The implementation surface differs:
- **vLLM** is C++/CUDA heavy. New attention algorithms register through the `AttentionBackend` interface; new schedulers extend `SchedulerInterface`.
- **SGLang** is more Python-forward. Custom KV caches subclass `MemoryPool`; custom schedulers extend `Scheduler`.

Read CONTRIBUTING.md of the chosen project before starting. A clean "draft PR" with the algorithm + microbenchmark is the typical entry point; full integration tests come later.

### Stretch B — Run your implementation on a second model

If you reproduced on Llama-3-8B, also run on Qwen-3-8B or DeepSeek-R1-Distill-Llama-8B. Different attention geometries (MHA / GQA / MLA) often shift the optimization's payoff dramatically. The cross-model comparison is what turns a "reproduction" into "evidence about how the method generalises."

---

## Connections

- **Back to [[ch-06]]** — PagedAttention background; this capstone's Option 1.
- **Back to [[ch-07]]** — Prefix caching + RadixAttention background; this capstone's Option 2.
- **Back to [[ch-09]]** — Disaggregation background (DistServe, Splitwise, Mooncake); this capstone's Option 5.
- **Back to [[ch-14]]** — Speculative decoding foundations; this capstone's Option 3.
- **Back to [[ch-15]]** — Multi-head / feature-level draft methods (Medusa, EAGLE, Lookahead); this capstone's Option 4.
- **Back to [[ch-21]]** — the benchmark harness this capstone reuses.

## Further reading

- [[pagedattention]] — the canonical paper.
- [[sglang-radixattention]] — the canonical paper + source.
- [[speculative-decoding]], [[fast-inference-from-transformers-via-speculative-decoding]] — the two SpecDec foundations.
- [[eagle]] — the feature-level variant.
- [[distserve]], [[mooncake]] — the disaggregation lineage.

## Excerpts

- [[excerpts/method-comparison]] — side-by-side breakdown of the five options with reproduction-risk scoring.
- [[excerpts/pagedattention-reference]] — pseudocode-level reference implementation skeleton for Option 1.
- [[excerpts/specdec-acceptance]] — the acceptance test with full residual-sampling math worked through for Option 3.
- [[excerpts/reproduction-memo]] — the memo template + worked-example "good vs bad memo" pair.
- [[excerpts/debugging-tree]] — the diagnostic tree for "my numbers don't match the paper."
