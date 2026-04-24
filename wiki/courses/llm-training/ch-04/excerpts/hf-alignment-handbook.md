---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: HuggingFace Alignment Handbook — SFT recipe, TRL DataCollatorWithPacking, apply_chat_template
source_url: https://github.com/huggingface/alignment-handbook
created_at: "2026-04-23"
---

# Excerpt: HF Alignment Handbook — TRL, Packing, and Chat Templates

**Source:** `wiki/raw-data/llm-training/blogs/hf-alignment-handbook.md`
**Repository:** https://github.com/huggingface/alignment-handbook
**Authors (Zephyr lineage):** Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, Thomas Wolf

---

## Bibliographic header

The Alignment Handbook is not a paper — it is HuggingFace's reference stack. Its importance is that it canonises a *recipe*: the YAML config + shell one-liner that produces Zephyr-7B, StarChat2, and Llama-3.1-Tülu. Every downstream SFT project either extends this handbook or explains how it diverges. The core recipe:

> *"Chat-template-aware SFT with response-only loss + packing, then DPO — no custom RL code, all on TRL + Accelerate + DeepSpeed/FSDP."*

This single sentence encodes the industry consensus post-2023: use TRL's `SFTTrainer`, turn on `packing=True` and `train_on_response_only=True`, let Accelerate handle distribution, chain DPO after SFT.

---

## The recipe guideline

From the raw-data notes:

> *"Use the handbook's `run_sft.py` as a starting template; override `dataset_mixer`, `chat_template`, and `max_seq_length`; keep `packing: true` and `train_on_response_only: true` unless you have a specific reason to deviate."*

Three things to override, two things to leave alone. The handbook's value is precisely that it removes decisions — by making packing and response-only masking *defaults*, it eliminates the ability for a practitioner to accidentally turn them off.

---

## The chat-template rendering step — `apply_chat_template`

Every tokeniser in `transformers ≥ 4.34` carries a Jinja-style template in `tokenizer.chat_template`. The handbook's first step on every dataset is:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4."},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

The result is a single string with the model's canonical chat formatting — for Mistral/Zephyr it uses `<|system|>`, `<|user|>`, `<|assistant|>` role tokens; for Llama-3 it uses `<|start_header_id|>...<|end_header_id|>` spans; for Qwen it uses `<|im_start|>` / `<|im_end|>`. Each family expects a *different* format, and the pretrained/instruction-tuned checkpoint was trained on its specific one.

**Notice:** this is where the most common silent SFT bug lives. If you format messages with the *wrong* template (e.g., using Llama-2's `[INST] ... [/INST]` on a Mistral model, or forgetting to use the template at all and passing raw dicts to `SFTTrainer`), the model trains on a chat format it will never see at inference. The handbook's "verify chat template by decoding a packed batch" advice is a direct response to this bug class.

---

## `apply_chat_template` pitfalls

### Pitfall 1: the `add_generation_prompt` flag

`apply_chat_template(messages, add_generation_prompt=True)` appends the assistant-role prefix (e.g., `<|assistant|>\n`) to signal "generate assistant response next". At **training** time, you want `add_generation_prompt=False` — the assistant response is already in `messages`. At **inference** time you want `add_generation_prompt=True` — there is no assistant response yet. Getting this backwards means training on a template that has the assistant prefix twice, or inferring from a template that never cues the model to start generating.

### Pitfall 2: system prompts that the base model wasn't trained with

Some models (older Llama-2 variants) do not support a system-role message in their template. Passing `{"role": "system", ...}` either raises or silently gets inlined into the first user turn. Check the template source before assuming a system slot exists.

### Pitfall 3: template mutation across tokeniser versions

Templates are strings embedded in `tokenizer_config.json`. When a model author updates their template (common for bug fixes), downstream checkpoints using the old template can diverge from the new one. Pin the tokeniser version alongside the model checkpoint.

### Pitfall 4: the handbook's advice — decode and eyeball

From the raw-data notes:

> *"Always verify chat template by decoding a packed batch — template mismatch is the #1 silent bug."*

The concrete verification is:

```python
batch = next(iter(trainer.get_train_dataloader()))
print(tokenizer.decode(batch["input_ids"][0]))
```

If the decoded string does not look like the model's native chat format (role tokens in the right places, prompt/response alternation correct), stop the run and fix the template.

---

## TRL `SFTTrainer` — packing + response-only masking defaults

From the raw-data notes:

```python
from trl import SFTTrainer, SFTConfig
config = SFTConfig(
    packing=True,
    max_seq_length=2048,
    train_on_response_only=True,  # masks user tokens automatically
    dataset_kwargs={"add_special_tokens": False},
)
```

Each flag maps to a specific mechanism:

### `packing=True` → `DataCollatorWithPacking`

When `packing=True`, `SFTTrainer` installs `DataCollatorWithPacking` (or the newer `DataCollatorForCompletionOnlyLM` with packing) as the data collator. The collator:

1. Tokenises each example once.
2. Concatenates tokenised examples into packs of length ≤ `max_seq_length`, using a greedy first-fit (a simplification of SPFHP — see [[excerpts/sequence-packing]]).
3. Builds `position_ids` that reset to 0 at each sub-sequence boundary.
4. Builds `cu_seqlens` for FlashAttention's varlen kernel (FlashAttention-2 varlen API, https://arxiv.org/abs/2307.08691; original FlashAttention, https://arxiv.org/abs/2205.14135).
5. Pads the final slack with `[PAD]` tokens whose labels are `-100`.

**Notice:** TRL's packing implementation currently uses a *streaming* first-fit rather than the offline SPFHP described in the Krell et al. paper. The packing ratio is slightly worse (~95% vs ~99%) but the implementation does not need to materialise the full length histogram, which matters for streamed datasets.

### `train_on_response_only=True` → automatic user-token masking

This flag activates a post-processing step that, after tokenising the chat-templated string, identifies the assistant-role spans and sets labels to `-100` for everything else:

```python
# conceptual — the real code pattern-matches against tokeniser-specific
# role tokens emitted by apply_chat_template
labels = input_ids.clone()
for span_start, span_end in user_and_system_spans:
    labels[span_start:span_end] = -100
```

The span identification relies on the specific role tokens emitted by `apply_chat_template` — yet another reason template correctness is load-bearing. If the template is wrong, the span detection fails silently and masking is incorrect. See [[excerpts/loss-masking-prompt]] for the formal loss definition this flag implements.

### `dataset_kwargs={"add_special_tokens": False}`

`add_special_tokens=False` prevents the tokeniser from automatically prepending `<bos>` / appending `<eos>` to every sub-sequence. The chat template already inserts the correct special tokens at the right positions; letting the tokeniser add more on top produces doubled `<bos>` tokens that the model was not pretrained to handle.

---

## Axolotl `sample_packing` — the parallel implementation

Axolotl (`axolotl-ai-cloud/axolotl`) exposes the same packing concept under a different flag:

```yaml
# axolotl config excerpt
sample_packing: true
pad_to_sequence_len: true
sequence_len: 2048
```

Under the hood Axolotl implements offline SPFHP more faithfully than TRL — it precomputes the packing plan over the full dataset before training starts, yielding a higher packing ratio at the cost of a slower `prepare_dataset` phase. The choice between TRL (streaming, lower overhead) and Axolotl (offline, higher density) is a throughput-vs-latency tradeoff typically made at the dataset-size crossover of ~1M examples.

---

## Zephyr-7B-β reference hyperparameters

From the raw-data notes:

| Knob | Value |
|------|-------|
| Model | Mistral-7B-v0.1 |
| Max seq length | 2048 |
| Packing | true |
| Train on response only | true |
| Optimizer | AdamW (β = 0.9, 0.95) |
| Learning rate | 2e-5 |
| LR schedule | cosine, 10% warmup |
| Epochs | 1 |
| Global batch size | 128 |
| Precision | BF16 |
| FSDP / ZeRO | FSDP FULL_SHARD |
| Gradient checkpointing | true |

Observations on these choices:

- **1 epoch.** The handbook canonises single-epoch SFT. Multiple epochs on 200K UltraChat caused measurable overfitting in the Zephyr v1 experiments — the loss curve keeps dropping but MT-Bench plateaus or regresses. NEFTune (α=5, stacked cleanly) partially mitigates this, but the handbook's conservative default is to just not run more than one epoch.
- **LR 2e-5 with cosine + 10% warmup.** Standard for full-finetune SFT on 7B. For LoRA the handbook uses 2e-4 (an order of magnitude higher), reflecting LoRA's smaller effective parameter count.
- **Global batch 128 @ 2048 length.** Each step processes 128 × 2048 = 262K tokens. With packing at ~95% density on UltraChat, effective useful tokens per step is ~250K — a reasonable SFT batch for 7B-scale.
- **FSDP FULL_SHARD.** The handbook prefers FSDP for 7–13B; ZeRO-3 for ≥ 13B. In 2024 this crossover shifted (FSDP's `use_orig_params=True` closed most of the gap) but the handbook's default is still FSDP for the reference 7B config.

---

## DPO recipe — the handoff from SFT

From the raw-data notes:

| Knob | Value |
|------|-------|
| β | 0.1 |
| Learning rate | 5e-7 |
| LR schedule | cosine |
| Epochs | 1 |
| Global batch size | 32 pairs |
| π_ref | SFT checkpoint (loaded separately, frozen) |

DPO follows SFT directly — the SFT checkpoint is both the initialisation for the policy and the reference `π_ref`. The learning rate drops two orders of magnitude (2e-5 → 5e-7) because DPO's gradient magnitudes are larger per-step (log-ratio differences rather than cross-entropy), and the batch is halved in example count (32 pairs ≈ 64 sequences). One epoch again; the raw-data note about "longer causes chosen-side collapse" is the motivation for later methods (IPO, SimPO) that are less sensitive to over-training.

DPO is out of scope for ch-04 (that chapter is packing / masking / chat templates), but the handoff matters: SFT done right is the precondition for DPO to work. Misformatted chat templates at SFT time produce a broken SFT checkpoint that DPO then amplifies.

---

## Lessons table — practitioner wisdom from the handbook

From the raw-data notes:

> *"- Always verify chat template by decoding a packed batch — template mismatch is the #1 silent bug.*
> *- DPO runs for < 1 epoch; longer causes chosen-side collapse (motivating [[ipo]], [[simpo]]).*
> *- Use ZeRO-3 for ≥ 13B; FSDP for ≤ 13B (in 2024 the throughput crossover shifted; re-benchmark).*
> *- NEFTune (α=5) stacks cleanly; toggled via SFT config."*

Each of these is a bug story compressed to one line. The chat-template-decode instruction in particular has prevented more silent SFT failures than any other single piece of handbook advice — it is the cheapest sanity check available and catches the most expensive failure mode.

---

## Connections

- Packing mechanism implemented by `DataCollatorWithPacking`: [[excerpts/sequence-packing]]
- Response-only masking implemented by `train_on_response_only`: [[excerpts/loss-masking-prompt]]
- Regulariser toggled via SFT config: [[excerpts/neftune]]
- Chapter synthesis: [[ch-04]]
- FlashAttention varlen backing TRL's packed kernels: https://arxiv.org/abs/2307.08691 (v2), https://arxiv.org/abs/2205.14135 (v1)
