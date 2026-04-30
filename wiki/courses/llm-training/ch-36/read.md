<!-- chapter: ch-36
     track: sft
     kind: lab
     title: Lab — Packed SFT Run with Masking Tests
     deps: [ch-35]
     sources: [[sequence-packing]], [[packed-vs-unpacked-ablation]], [[loss-masking-prompt]], [[neftune]], [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]], [[tulu-3-sft-mix]], [[lima]], [[karpathy-training-neural-net-recipe]], [[fsdp-sft]]
     figures: figures/ablation-matrix.html
     capstone_for: sft-track (ch-30..ch-36)
-->

# Chapter 36 — Lab: Packed SFT Run with Masking Tests

> **Core insight.** The SFT stack is a silent-failure machine. Packing without per-sub-sequence masks, masking without per-sub-sequence reset, or NEFTune on top of a broken mask all *train* — loss curve fine, throughput fine, only downstream eval notices the model has learned to regurgitate prompts. The capstone of this track is not `Llama-3.2-3B-sft.safetensors`; it is the `tests/` directory that failed first, caught the regression, and would catch it again next sprint.
>
> **Guideline.** Build inside-out: unit-test the three invariants ([[sequence-packing]] varlen mask, [[loss-masking-prompt]] per-sub-sequence reset, chat-template rendering) *before* you launch a training step. Run three ablations — packed vs unpacked ([[packed-vs-unpacked-ablation]]), prompt-masked vs full-loss, NEFTune on/off ([[neftune]]) — each as a single-axis change per [[karpathy-training-neural-net-recipe]]. Data mix is [[lima]]-plus-synthetic against a decontaminated eval suite ([[tulu-3-sft-mix]]). Deliverable: `sft-run-memo.md` with ablation table and one regression your masking tests caught.

---

## Goal

Three artifacts, each reproducible by a peer:

1. **A tested SFT run.** `checkpoint-final/` plus a green `pytest` report on `tests/test_chat_template.py`, `tests/test_loss_mask.py`, `tests/test_packed_attention.py`. Every test must have failed at least once before it passed — commit the failing output as a fixture.
2. **An ablation grid.** `ablations.json` with eight rows (2×2×2) for full-budget, or two rows (packed × unpacked) for resource-constrained. Columns: `train_loss_final`, `throughput_tok_per_gpu_per_s`, `mtbench_mean`, plus per-slice breakdown across the six [[tulu-3-sft-mix]] buckets.
3. **A memo.** `sft-run-memo.md`, one page: ablation table, regression spec, one surprise.

Predict every ablation's delta *before* training per [[karpathy-training-neural-net-recipe]]. Surprises are what you learned.

---

## Full-budget path

Target: 8×H100 (or 8×A100-80GB), Llama-3.2-3B base, 100K mix, ~8 h wall-clock for all 8 runs.

- **Base.** `meta-llama/Llama-3.2-3B` (not 3B-Instruct — you apply the chat template). FSDP `FULL_SHARD` per [[fsdp-sft]]; `auto_wrap_policy = transformer_auto_wrap_policy({LlamaDecoderLayer})`; BF16 mixed precision.
- **Mix.** 100K = 1K [[lima]] + 79K ch-29 synthetic + 20K decontaminated No-Robots / OpenAssistant-2. Skill balance per [[tulu-3-sft-mix]]: chat 27%, math 21%, code 14%, IF 11%, safety 10%, multilingual/reasoning 17%. Decontaminate against MT-Bench by 8-gram overlap ≥ 50% + embedding sim > 0.9.
- **Training.** [[hf-alignment-handbook]] defaults: LR 2e-5 cosine 10% warmup, 1 epoch, global batch 128, `max_seq_length=2048`, packing true, response-only true, `AdamW(0.9, 0.95)`. NEFTune α=5 per [[neftune]] when axis is on.
- **Eval.** MT-Bench single-turn (80 prompts), sliced by the six [[tulu-3-sft-mix]] buckets. Judge `gpt-4o-mini`; paired win-rate vs `packed+masked+neftune-off` baseline.

## Resource-constrained path

Target: 1 GPU (≥24 GB), Llama-3.2-1B or Qwen2.5-1.5B, 20K mix, ~3 h for 2 runs.

- Mix = 1K LIMA + 15K synthetic + 4K chat; skill balance unchanged.
- One axis: **packed vs unpacked**. NEFTune on, masking on. This is the axis whose correctness matters most and whose throughput delta is largest — exactly what [[packed-vs-unpacked-ablation]] §Diagnostic targets.
- All three masking unit tests still required. The lab's integrity is in the tests.

---

## §1 Data mix assembly

Write `build_mix(spec) -> Dataset` returning HF `Dataset` with `{messages, slice, source}`. The `slice` column is what makes the per-slice regression table possible — forget it now and skill-level signal is unrecoverable.

Decontamination is not optional. Per [[tulu-3-sft-mix]], 8-gram overlap ≥ 50% vs any MT-Bench prompt is an automatic drop. Log drops; if > 5% drop, your synthetic set is contaminated — regenerate before training.

---

## §2 Chat-template unit tests

The #1 silent bug per [[hf-alignment-handbook]]: template mismatch. Test it like code.

```python
# tests/test_chat_template.py
import pytest
from transformers import AutoTokenizer

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

def test_single_turn_template_has_bos(tok):
    msgs = [{"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}]
    s = tok.apply_chat_template(msgs, tokenize=False)
    assert s.startswith(tok.bos_token), f"missing BOS: {s!r}"

def test_assistant_mask_recovers_only_response(tok):
    msgs = [{"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"}]
    out = tok.apply_chat_template(msgs, tokenize=True,
                                  return_assistant_tokens_mask=True,
                                  return_dict=True)
    recovered = tok.decode([t for t, m in zip(out["input_ids"], out["assistant_masks"]) if m])
    assert "4" in recovered and "2+2" not in recovered, \
        f"assistant mask leaked prompt: {recovered!r}"

def test_empty_system_prompt_does_not_duplicate_bos(tok):
    msgs = [{"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"}]
    s = tok.apply_chat_template(msgs, tokenize=False)
    assert s.count(tok.bos_token) == 1, f"BOS duplicated: {s!r}"
```

Fail modes you will hit: `Llama-3.2` vs `Llama-3.1` templates differ; `Qwen2.5`'s `<|im_start|>` tokenizes as literal tag tokens if special tokens aren't added. Decode a random packed batch verbatim once per debug session per [[hf-alignment-handbook]].

---

## §3 Loss-mask unit tests

From [[loss-masking-prompt]] §Implementation: `labels[:prompt_len] = -100`. Tests must check (a) prompt ignored, (b) completion included, (c) multi-turn assistant-k only, (d) per-sub-sequence reset inside a pack.

```python
# tests/test_loss_mask.py
import torch, pytest
from transformers import AutoTokenizer
from your_sft_lib.masking import build_labels, build_labels_packed

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

def test_prompt_tokens_are_ignored(tok):
    ids = tok("USER: solve 2+2  ASSISTANT: 4", return_tensors="pt").input_ids[0]
    prompt_len = len(tok("USER: solve 2+2  ASSISTANT:").input_ids)
    labels = build_labels(ids, prompt_len=prompt_len)
    assert (labels[:prompt_len] == -100).all(), "prompt tokens leaked into loss"
    assert (labels[prompt_len:] != -100).any(), "completion lost its labels"

def test_packed_labels_reset_per_subsequence(tok):
    # three short conversations packed into one block
    convs = [("u1", "a1_token"), ("u22", "a2_token"), ("u333", "a3_token")]
    packed_ids, cu_seqlens, prompt_lens = pack(convs, tok)
    labels = build_labels_packed(packed_ids, cu_seqlens, prompt_lens)
    for i, (start, plen) in enumerate(zip(cu_seqlens[:-1], prompt_lens)):
        end = cu_seqlens[i+1]
        assert (labels[start:start+plen] == -100).all(), f"sub-seq {i} prompt not masked"
        assert (labels[start+plen:end] != -100).any(), f"sub-seq {i} completion dropped"

def test_pad_tokens_always_ignored(tok):
    ids = torch.tensor([tok.pad_token_id] * 5 + [10, 11, 12])
    labels = build_labels(ids, prompt_len=0, pad_id=tok.pad_token_id)
    assert (labels[:5] == -100).all(), "pad tokens contribute to loss"

def pack(convs, tok):
    # helper — your packing routine under test
    from your_sft_lib.packing import pack_batch
    return pack_batch(convs, tok, max_len=128)
```

The first test to actually fail in a fresh pipeline is almost always `test_packed_labels_reset_per_subsequence` — packing functions tend to mask the *first* prompt correctly and silently share the mask for the rest. This is Failure Mode 3 from [[packed-vs-unpacked-ablation]].

---

## §4 Packed-attention unit tests

[[sequence-packing]] gives you two invariants: (a) `cu_seqlens` offsets sum to `Σ L_i`, (b) token `t` in sub-sequence `i` has zero attention weight onto any token outside `[cu_seqlens[i], cu_seqlens[i+1])`. Both are testable without training.

```python
# tests/test_packed_attention.py
import torch, pytest
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
    HAS_FA = True
except ImportError:
    HAS_FA = False

def test_cu_seqlens_sum_matches_total(make_pack):
    seqs, packed, cu = make_pack([5, 7, 3, 11])
    assert cu.tolist() == [0, 5, 12, 15, 26]
    assert cu[-1].item() == packed.shape[0]
    assert cu[-1].item() == sum(len(s) for s in seqs)

def test_position_ids_reset_per_subsequence(make_pack):
    _, packed, cu = make_pack([4, 6, 3])
    from your_sft_lib.packing import build_position_ids
    pos = build_position_ids(cu)
    assert pos.tolist() == [0,1,2,3, 0,1,2,3,4,5, 0,1,2], "RoPE positions not reset"

@pytest.mark.skipif(not HAS_FA, reason="flash-attn not installed")
def test_varlen_attention_zeros_cross_sample_leakage(make_pack):
    torch.manual_seed(0)
    seqs, packed, cu = make_pack([4, 6])
    B, H, D = 1, 1, 16
    L = packed.shape[0]
    q = torch.randn(L, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(L, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(L, H, D, device="cuda", dtype=torch.bfloat16)
    out_varlen = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu.to("cuda", torch.int32),
        cu_seqlens_k=cu.to("cuda", torch.int32),
        max_seqlen_q=6, max_seqlen_k=6, causal=True,
    )
    # Sub-seq 1 output must equal standalone sub-seq 1 output.
    q1, k1, v1 = q[:4], k[:4], v[:4]
    out_solo = flash_attn_varlen_func(
        q1, k1, v1,
        cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        cu_seqlens_k=torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        max_seqlen_q=4, max_seqlen_k=4, causal=True,
    )
    assert torch.allclose(out_varlen[:4], out_solo, atol=1e-2), \
        "packed sub-sequence 1 output contaminated by sub-sequence 2"

@pytest.fixture
def make_pack():
    def _make(lens):
        seqs = [torch.arange(L) + 1000 * i for i, L in enumerate(lens)]
        packed = torch.cat(seqs)
        cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)))
        return seqs, packed, cu
    return _make
```

Add a negative control: compute dense causal attention over the packed tensor without `cu_seqlens`; its output on sub-seq 2 must *differ* from standalone sub-seq 2 attention. A suite with only positive cases cannot distinguish a correct packer from a no-op one (Failure Mode 4 from [[packed-vs-unpacked-ablation]]).

---

## §5 Ablation grid

2×2×2 cube (full-budget); packed-only axis (resource-constrained).

| Run | Packed | Masked | NEFTune | Purpose |
|-----|--------|--------|---------|---------|
| `full` | yes | yes | yes | headline claim |
| `no-pack` | no | yes | yes | isolate throughput cost; [[packed-vs-unpacked-ablation]] predicts parity |
| `full-loss` | yes | no | yes | isolate masking; [[loss-masking-prompt]] predicts MT-Bench drop |
| `no-neft` | yes | yes | no | NEFTune saturation at scale ([[allenai-tulu-sft-recipe]] reports neutral at 939K) |
| `no-pack-no-mask`, `no-pack-no-neft`, `no-mask-no-neft` | mixed | mixed | mixed | interaction controls |
| `none` | no | no | no | worst-case anchor |

Predict each delta before launch. Log predictions in `predictions.txt`, `git add` it, treat as contract: any row that disagrees beyond MT-Bench standard error is a paragraph in the memo.

---

## §6 Eval harness

- **Harness.** A 120-line script running 80 MT-Bench prompts with one judge call per response. `lm-eval-harness` is overkill here.
- **Slices.** Six [[tulu-3-sft-mix]] buckets. Hand-tag each of the 80 prompts (budget 20 minutes).
- **Judge.** `gpt-4o-mini`; paired with `packed+masked+neftune-off` baseline. Report paired win-rate + bootstrapped 95% CI.
- **Throughput.** Log `tokens_per_gpu_per_second` *and* `samples_per_second` separately. Packing's win only shows in the former — reporting samples/s makes packing look like a no-op. [[packed-vs-unpacked-ablation]] predicts speedup ≈ L_max / avg(L_i); for 100K mix with avg(L_i)≈600, L_max=2048, expect ~2.5–3× realized.

---

## §7 Memo template

`sft-run-memo.md`, one page, five sections:

1. **Setup.** Base, mix composition, epochs, GPU count, wall-clock.
2. **Ablation table.** 8 (or 2) rows with MT-Bench mean, per-slice breakdown, throughput, paired win-rate vs `full`. Tag every delta "predicted" or "surprising."
3. **Test-caught regression.** Test name, commit SHA of buggy code, observed failure (mask overlap diff, cross-sample leakage L2, etc.), and the fix. This is the section the memo exists for.
4. **One surprise.** A specific per-slice row where the ablation delta disagreed with your prediction. Not "numbers were noisy." Example: "`no-neft` dropped 4 pts on IF slice but tied on chat — predicted the opposite."
5. **Next instrumentation.** One test you'd add to catch this regression *before* it reaches a training step. The test directory grows monotonically.

---

## Acceptance criteria

Hard gates, in order.

1. `pytest tests/ -v` passes on final code; a prior commit in `git log` shows at least one of the three test files failed first.
2. `ablations.json` has the full cube (or 2-row packed-axis cube). Throughput column is `tokens_per_gpu_per_second`, not `samples_per_second`.
3. Step-1 loss within 20% of `ln(|V|)` on every run ([[karpathy-training-neural-net-recipe]] overfit-a-batch sanity); otherwise chat template is wrong.
4. `pre_clip_grad_norm < 2` for first 100 steps of `full`; a spike means prompt tokens are leaking into loss.
5. `full` vs `no-pack` train-loss curves differ by < 0.01 nats at matching steps per [[packed-vs-unpacked-ablation]] §Diagnostic; larger gap = bug, not packing property.
6. Memo §3 names a specific regression with commit SHA, failing test, and fix diff. No SHA = capstone not earned.
7. At least one memo §2 row tagged "surprising" — otherwise the lab taught you nothing.

---

## Connections

- **ch-30 / ch-31 / ch-32 / ch-33 / ch-34 / ch-35** — this lab is the end-to-end instantiation: §1 lifts ch-34's data-mix discipline; §2 enforces ch-31's chat-template invariants; §3 enforces ch-32's loss-mask rules; §4 enforces ch-33's packed-attention contract; §5's NEFTune axis tests ch-35 at scale.
- **ch-29 (synthetic lab)** — the 79K synthetic slice comes from `instructions.jsonl` there.
- **Track 4 (RL)** — `checkpoint-final` from `full` becomes `π_ref` for every DPO / PPO / GRPO chapter that follows.

## Further reading

- [[sequence-packing]], [[packed-vs-unpacked-ablation]] — varlen API, cu_seqlens, four failure modes, 100-step diagnostic, throughput model.
- [[loss-masking-prompt]] — Shi 2024; response-only dominates; multi-turn masking rule.
- [[neftune]] — uniform α/√(Ld) noise; saturates on large mixes.
- [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]], [[tulu-3-sft-mix]] — Zephyr + Tülu 3 hparams and decontamination discipline.
- [[lima]] — 1K baseline slice; Superficial Alignment Hypothesis.
- [[karpathy-training-neural-net-recipe]] — predict-before-run; overfit-one-batch; monitor clip-norm.
- [[fsdp-sft]] — FULL_SHARD memory math; `auto_wrap_policy` on `LlamaDecoderLayer`.

## Companion visualization

**[figures/ablation-matrix.html](figures/ablation-matrix.html)** — interactive 2×2×2 ablation cube. Click any corner to surface an illustrative MT-Bench score and a justification grounded in [[packed-vs-unpacked-ablation]], [[loss-masking-prompt]], or [[neftune]]. Use as a prediction aid: click through the cube, read each justification, *then* write `predictions.txt`. Scores are illustrative; arrow directions are attested.
