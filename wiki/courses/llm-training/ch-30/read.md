<!-- chapter: ch-30
     track: sft
     kind: content
     title: SFT Design Axes
     deps: [ch-29]
     sources: [[loss-masking-prompt]], [[sequence-packing]], [[neftune]], [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]], [[tulu-3-sft-mix]], [[packed-vs-unpacked-ablation]], [[rlhf-instructgpt]], [[llama-3]], [[qwen-3]], [[deepseek-v3]], [[deepseek-r1]]
     figures: figures/sft-axes.html
     opens_track: sft (ch-30..ch-36)
-->

# Chapter 30 — SFT Design Axes

> **Core insight.** SFT is not one algorithm; it is a point in a five-dimensional design space — regime, loss mask, chat template, packing strategy, embedding noise — and every downstream RL / DPO / RLVR run inherits the surface that these axes carved into the base model. The template you picked, the tokens you masked, and the turns you trained on pre-determine what RL can later optimise and what it cannot. RL moves the policy *along* the SFT manifold; it does not rebuild it.
>
> **Guideline.** Decide each axis explicitly before you run: (1) regime (instruction / chat / tool / reasoning / agentic) dictates which tokens even exist; (2) loss mask (response-only is the default, per [[loss-masking-prompt]] and [[hf-alignment-handbook]]); (3) chat template (verbatim from your target base model — Llama-3, Qwen, DeepSeek — never edited ad-hoc); (4) packing (on by default with `flash_attn_varlen_func` + `cu_seqlens`, per [[sequence-packing]] and [[packed-vs-unpacked-ablation]]); (5) NEFTune (on for datasets ≤ 100K, off at ≥ 500K — [[neftune]] and [[allenai-tulu-sft-recipe]]). Document the five-tuple in a one-line config header and validate by decoding one packed batch back to text before the real run.

---

## Why this chapter exists

The synthetic track (ch-18..ch-29) ended with a pool of instruction rows on disk. The RL track (ch-37..ch-46) will start from a policy checkpoint. Every chapter between the two is SFT — and the SFT track (ch-30..ch-36) is where the policy is born. This opening chapter names the axes. Ch-31 (iterative SFT↔RL bridges), ch-32 (reasoning SFT and long-CoT cold-start), ch-33 (tool-call SFT), ch-34 (agentic SFT), ch-35 (distillation), and ch-36 (the SFT lab) will each vary one or more of these axes — and ch-35's distillation recipe only makes sense once you know which axis the teacher fixed for you.

The `pre-determines` framing matters because the SFT track is the last place in the pipeline where you have cheap, dense gradient on behaviours you care about. RL has sparse reward and a KL tether to π_SFT (see [[rlhf-instructgpt]] Equation 2: `r(x,y) − β · log π_RL/π_SFT`). If the behaviour is not even a plausible completion under π_SFT, the KL penalty will snap the policy back before reward can pull it forward. SFT is the generator of the manifold RL explores.

---

## §1. Regime — the five shapes SFT can take

One "SFT" run can implement any of five regimes. They differ in which tokens exist, which get loss, and what the template has to express.

| Regime | Example input | Example target | Tokens introduced |
|--------|---------------|----------------|-------------------|
| Single-turn instruction | "Translate to French: hello" | "bonjour" | user, assistant |
| Multi-turn chat | alternating `user` / `assistant` | final (or per-turn) `assistant` | system, user, assistant |
| Tool-call | user prompt | `<tool_call>{"name":..., "arguments":...}</tool_call>` + assistant reply | `tool_call`, `tool_response` (or role `tool`) |
| Reasoning | math / code prompt | `<think>…</think><answer>…</answer>` per [[deepseek-r1]] template | `<think>`, `</think>` |
| Agentic | user task + tool catalog | interleaved thought / tool-call / observation trace | all of the above + loop structure |

The regime chosen determines everything downstream: a model trained with single-turn instruction SFT cannot emit a `<tool_call>` block because the token distribution over `<tool_call>` never saw gradient. Ch-31 will show that SFT→RL bridges work by regime: rejection-sampling SFT (Llama-2 appendix, carried forward in [[llama-3]]) only adds mass to behaviours the SFT regime already supports.

---

## §2. Loss mask — what contributes to cross-entropy

Per-regime mask table. `Y` = included in loss; `N` = masked to `-100`; `opt` = toggleable but the attested default is the starred value.

| Regime | system | user | assistant (current turn) | assistant (prior turns) | tool_call | tool_response | thought (`<think>`) |
|--------|--------|------|--------------------------|-------------------------|-----------|---------------|----------------------|
| Single-turn instruction | N | N | **Y** | — | — | — | — |
| Multi-turn chat | N | N | **Y** | N (mask) | — | — | — |
| Tool-call | N | N | **Y** | N | **Y** (the call) | N (observation) | — |
| Reasoning | N | N | **Y** | N | — | — | **Y** (the CoT is signal) |
| Agentic | N | N | **Y** | N | **Y** | N | **Y** |

The invariants: system and user are always masked; tool-response / observation tokens are always masked (they come from the environment, the model never emits them); the *emitted* artefacts (assistant text, tool-call JSON, thought content) always carry loss. The one sharp ablation in the literature is [[loss-masking-prompt]] / Shi 2024: response-only SFT strictly beats full-sequence SFT on MT-Bench and AlpacaEval, except in the tiny-dataset / strong-base regime where full-sequence acts as mild continued-pretraining. That is not a contradiction; it is a sign that when the SFT objective is too narrow, leaking pretrain-style loss helps — which is why [[rlhf-instructgpt]] later adds `γ · L_ptx` to PPO.

Minimal implementation, directly from [[loss-masking-prompt]]:

```python
# mask.py — the one primitive every SFT framework hides from you
import torch
import torch.nn.functional as F

IGNORE = -100

def build_labels(input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Single-turn: mask first prompt_len positions; loss on the rest."""
    labels = input_ids.clone()
    labels[:prompt_len] = IGNORE
    return labels

def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # shift-by-one + flatten + ignore_index
    return F.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.size(-1)),
        labels[..., 1:].reshape(-1),
        ignore_index=IGNORE,
    )
```

Multi-turn extension (per [[loss-masking-prompt]] §Multi-turn chat masking): for a conversation `[u_1, a_1, u_2, a_2, ..., u_k, a_k]`, mask every `u_i`, mask `a_1..a_{k-1}`, train only on `a_k`. The per-turn-training variant unrolls the conversation k times, masking through `a_{i-1}` and training on `a_i` each time — same loss value, k× more data.

---

## §3. Chat template — the surface the base model already speaks

Chat templates are not cosmetic. They are the set of literal strings (tokens, not text) that the tokenizer maps to special-token IDs, and that the base model's post-training already conditioned on. Using the wrong template is a silent-bug class: decoding looks fine, training loss looks fine, downstream win-rate collapses 5-20 pts. [[hf-alignment-handbook]] names it the "#1 silent bug" and instructs: decode a packed batch back to text and eyeball the delimiters before launching.

Chat-template matrix across four shipping families. All strings quoted verbatim from the respective model reports / tokenizer configs.

| Family | BOS | Role delimiter | Turn-end | Thinking tokens | Tool-call tokens |
|--------|-----|----------------|----------|-----------------|-------------------|
| ChatML (OpenAI / early Mistral) | none / `<s>` | `<\|im_start\|>role\n` | `<\|im_end\|>\n` | none native | convention only |
| Llama-3 ([[llama-3]]) | `<\|begin_of_text\|>` | `<\|start_header_id\|>role<\|end_header_id\|>\n\n` | `<\|eot_id\|>` | none native | `<\|python_tag\|>` (Llama-3.1 extension) |
| Qwen-3 ([[qwen-3]]) | none | `<\|im_start\|>role\n` | `<\|im_end\|>\n` | `<think>...</think>` scoped inside assistant | `<tool_call>{...}</tool_call>` inside assistant |
| DeepSeek-V3 ([[deepseek-v3]]) | `<｜begin▁of▁sentence｜>` | `<｜User｜>...<｜Assistant｜>` | `<｜end▁of▁sentence｜>` | DeepSeek-R1 adds `<think>...</think>` in assistant | convention inside assistant |

Concrete rendering of one user/assistant exchange in each family, verbatim delimiters:

```
# ChatML
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
Hello.<|im_end|>

# Llama-3  (per the Llama-3 tokenizer's chat_template)
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hello.<|eot_id|>

# Qwen-3 with hybrid thinking
<|im_start|>user
Solve 2+2<|im_end|>
<|im_start|>assistant
<think>2 plus 2 is 4.</think>4<|im_end|>

# DeepSeek-V3
<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜>Hello.<｜end▁of▁sentence｜>
```

Three consequences you must respect:

1. **Every delimiter is a real token ID.** `<|eot_id|>` in Llama-3 tokenises to one token; in ChatML `<|im_end|>` is one token; they are not interchangeable substrings.
2. **The base model's generation halts on its native end-of-turn.** If you train Llama-3 base with ChatML, the post-SFT model will generate forever because `<|im_end|>` is not a stop token in its config.
3. **Tool-call and thinking tokens live *inside* the assistant turn.** They are not their own role. The assistant emits `<think>...</think>` and `<tool_call>{...}</tool_call>` as part of one continuous span terminated by the turn-end token.

The attested practitioner move is: always call `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=...)` rather than building the string yourself. The template is shipped in the tokenizer's `tokenizer_config.json` for a reason — it is the contract with the base model.

---

## §4. Packing — throughput axis that is also a correctness axis

Raw instruction batches are 50–89% padding ([[sequence-packing]] abstract; measured on BERT/GLUE and carried forward to SFT). Packing concatenates short sequences into fixed-length blocks with a block-diagonal attention mask and per-sub-sequence position-ID reset. The attested speedup:

```
speedup ≈ L_max / avg(L_i)
# [[packed-vs-unpacked-ablation]] §Throughput model
```

For an SFT mixture with `avg(L_i) = 600`, `L_max = 4096` → expected 6×, realised 2.5–3× after FlashAttention overhead.

The correctness contract, from [[sequence-packing]] §Mechanics:

```python
# packed_batch.py — the three fields every SFT trainer must produce together
cu_seqlens   = [0, L_1, L_1 + L_2, ..., sum(L_i)]  # int32, length n+1
position_ids = [0,1,..L_1-1, 0,1,..L_2-1, ...]    # reset per sub-sequence
labels       = build_labels_per_subseq(input_ids, prompt_lens, cu_seqlens)

# attention delegated to FlashAttention varlen — no L_max^2 mask materialised
from flash_attn import flash_attn_varlen_func
out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens,
                             max_seqlen=max(L_i), causal=True)
```

Four failure modes, each enumerated in [[packed-vs-unpacked-ablation]] §Failure modes:

1. Missing block-diagonal mask → tokens in sub-sequence 2 attend to sub-sequence 1 → softmax partition leaks across documents.
2. Un-reset position IDs → sub-sequence 2 sees positions L_1..L_1+L_2; RoPE is position-shifted.
3. Label mask not re-applied per sub-sequence → prompt tokens of sub-sequence 2 contribute to loss.
4. Using `flash_attn_func` (dense) instead of `flash_attn_varlen_func` → no mask at all, silent contamination.

Diagnostic: train 100 steps unpacked, 100 steps packed, same seed and data; loss curves must agree within 0.01 nats at matched step. Larger gap means bug, not quality drop.

---

## §5. NEFTune and the prompt-masked-vs-full-sequence axis

Two small-but-real tunable knobs:

**NEFTune** ([[neftune]]). One line, added pre-transformer:

```python
# neftune.py
import torch

def neftune(embeds: torch.Tensor, alpha: float = 5.0) -> torch.Tensor:
    L, d = embeds.shape[-2], embeds.shape[-1]
    eps = (torch.rand_like(embeds) * 2 - 1) * (alpha / (L * d) ** 0.5)
    return embeds + eps
```

Attested numbers: LLaMA-2-7B on plain Alpaca, AlpacaEval 29.79 → 64.69 with `α=5`. On Evol-Instruct / ShareGPT / OpenPlatypus, +8–10 pts. Gain saturates at scale: [[allenai-tulu-sft-recipe]] finds NEFTune neutral at 939K SFT prompts. The dataset-size rule: on by default when `|D| ≤ 100K`, off at `|D| ≥ 500K`, uncertain in between — sweep `α ∈ {5, 10, 15}` on a held-out 500-prompt probe if you care.

**Loss weighting**. Full-sequence loss is `L_full = −(1/(T_p+T_y)) · Σ log π(x_t|x_<t)`; response-only is `L_SFT = −(1/T_y) · Σ log π(y_t|p, y_<t)`. Shi 2024 (cited in [[loss-masking-prompt]]) ablated three variants — response-only, full-sequence, prompt-weighted (`α · L_prompt + L_response` with `α < 1`) — across Alpaca / ShareGPT / LIMA and multiple scales. Response-only dominates, prompt-weighted gives modest gains in some slices, full-sequence loses broadly. The default is response-only; the exception is tiny-dataset (≤ 1K) SFT on a strong base, where full-sequence acts as a mild continued-pretrain regulariser.

---

## §6. Two production recipes — what "on by default" actually means

[[hf-alignment-handbook]] Zephyr-7B-β SFT (Mistral-7B base, UltraChat-200K):

| Knob | Value |
|------|-------|
| Template | ChatML (Mistral default) |
| Max seq length | 2048 |
| Packing | true |
| Train on response only | true |
| NEFTune | α=5 (toggled per run) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Learning rate | 2e-5 |
| Schedule | cosine, 10% warmup |
| Epochs | 1 |
| Global batch | 128 |
| Precision | BF16 |
| Distributed | FSDP FULL_SHARD |

[[allenai-tulu-sft-recipe]] / [[tulu-3-sft-mix]] Tülu-3 SFT (Llama-3.1 base, 939K mix):

| Knob | 8B | 70B |
|------|-----|-----|
| Template | Llama-3 (native) | Llama-3 (native) |
| Max seq length | 4096 | 4096 |
| Packing | yes | yes |
| Train on response only | yes | yes |
| NEFTune | **off** (neutral at 939K) | off |
| Optimizer | AdamW (0.9, 0.95) | same |
| Learning rate | 5e-6 | 2e-6 |
| Schedule | linear, 3% warmup | same |
| Epochs | 2 | 2 |
| Global batch (prompts) | 128 | 128 |
| Precision | BF16 | BF16 |
| Distributed | FSDP FULL_SHARD | FSDP HYBRID_SHARD |

Notice how the axes map to the same labels with different values. That is the point of this chapter: the axes are stable, the defaults are not. Data scale moves NEFTune from "on" to "off"; base-model family moves template from ChatML to Llama-3; 70B parameter count halves the learning rate. Ch-36 (the SFT lab) will force you to fill this table for your own run and justify every deviation.

---

## §7. What SFT pre-determines — and what RL can fix later

SFT assigns positive probability mass to every behaviour it trains on, and leaves every other behaviour on the base model's prior. RL (PPO with KL tether per [[rlhf-instructgpt]]; DPO / GRPO in the RL track) moves probability *along* this manifold:

- **SFT-fixable**: surface format adherence (turn delimiters, JSON structure, `<think>` tags), style, verbosity, mandatory refusals, vocabulary-level token choices, language coverage that exists in SFT data.
- **RL-fixable (that SFT struggles with)**: preference between two well-formed completions both inside π_SFT's support, reward-verifiable correctness (math / code execution), helpfulness tradeoffs that depend on pairwise ranking rather than a single ground-truth.
- **Neither fixes well**: behaviours absent from both SFT and the base model's prior. If SFT never emitted a tool-call, PPO cannot find one at reasonable temperature because its rollout log-prob is effectively zero.

The pre-determines framing closes the chapter: pick the regime and template in ch-30, populate the mix in ch-31..ch-35, run the lab in ch-36 — then in ch-37..ch-46 you inherit whatever manifold you shaped here.

---

## Connections

- **ch-29** — the synthetic-set lab feeds this chapter its "single-turn instruction" pool; regime and template are where that pool meets the base model.
- **ch-31** — iterative SFT↔RL bridges (RSFT, Best-of-N SFT, STaR) vary the "regime" axis by generating new synthetic data from the current policy.
- **ch-32** — reasoning SFT and long-CoT cold-start deepen the "thought-token mask" row of the table ([[deepseek-r1]] template; [[qwen-3]] hybrid-thinking).
- **ch-33** — tool-call SFT makes the `tool_call` / `tool_response` columns load-bearing; chat-template matrix gets a sixth column.
- **ch-34** — agentic SFT chains multiple tool-call turns; the loss mask has to survive interleaved observations and thoughts.
- **ch-35** — distillation inherits the teacher's regime and template; SFT design axes become "which teacher axes do I adopt verbatim".
- **ch-36** — the SFT lab is this chapter written as runnable code.
- **ch-37..ch-46 (RL track)** — operates on π_SFT. Every axis fixed here bounds what RL can move.

## Further reading

- [[loss-masking-prompt]] — Shi 2024; response-only vs full-sequence ablation; multi-turn masking rule.
- [[sequence-packing]] — Krell 2021; SPFHP, cu_seqlens, position-ID reset, varlen kernel contract.
- [[packed-vs-unpacked-ablation]] — compilation of 2021–2024 ablations; four packing failure modes; diagnostic procedure.
- [[neftune]] — Jain 2023; one-line embedding noise; α=5 default; saturation at scale.
- [[hf-alignment-handbook]] — Zephyr recipe; `SFTConfig(packing=True, train_on_response_only=True)`; decode-a-packed-batch lesson.
- [[allenai-tulu-sft-recipe]] / [[tulu-3-sft-mix]] — 939K mix; skill-level ablations; NEFTune saturation.
- [[rlhf-instructgpt]] — the β · KL term that makes SFT the manifold RL walks on.
- [[llama-3]] — Llama-3 chat template; iterative SFT/RS/DPO rounds.
- [[qwen-3]] — hybrid-thinking template; `<|im_start|>` / `<think>` scoping.
- [[deepseek-v3]] / [[deepseek-r1]] — DeepSeek delimiter family; `<think>`/`<answer>` template chosen for rule-based reward parseability.

## Companion visualization

**[figures/sft-axes.html](figures/sft-axes.html)** — self-contained interactive config builder. Toggle the five axes (regime / mask / template / packing / NEFTune) and see the resulting training config summary, the per-regime loss-mask row highlighted, the template's verbatim delimiters, and the expected packing speedup recomputed from your `L_max / avg(L_i)` entry. Use it as a pre-flight checklist before ch-36's lab: if any axis on the summary reads "unset" or "inferred", you are not yet ready to launch SFT.
