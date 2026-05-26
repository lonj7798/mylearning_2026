<!-- chapter: ch-36
     track: sft
     kind: lab
     title: Lab — Packed SFT Run with Masking Tests
     deps: [ch-35]
     sources: [[sequence-packing]], [[packed-vs-unpacked-ablation]], [[loss-masking-prompt]], [[neftune]], [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]], [[tulu-3-sft-mix]], [[lima]], [[karpathy-training-neural-net-recipe]], [[fsdp-sft]]
     figures: figures/ablation-matrix.html
     capstone_for: sft-track (ch-30..ch-36)
-->

# 36장 — Lab: Masking Test가 있는 Packed SFT Run

> **핵심 통찰.** SFT stack은 silent-failure machine이다. Per-sub-sequence mask 없는 packing, per-sub-sequence reset 없는 masking, broken mask 위의 NEFTune은 모두 *train*된다. Loss curve도 정상이고 throughput도 정상이며, downstream eval만 모델이 prompt를 regurgitate하는 법을 배웠다는 것을 알아차린다. 이 track의 capstone은 `Llama-3.2-3B-sft.safetensors`가 아니다. 먼저 실패했고, regression을 잡았고, 다음 sprint에도 다시 잡을 `tests/` directory이다.
>
> **가이드라인.** Inside-out으로 만들어라. Training step을 launch하기 *전에* 세 invariant([[sequence-packing]] varlen mask, [[loss-masking-prompt]] per-sub-sequence reset, chat-template rendering)를 unit-test하라. 세 ablation을 실행하라. Packed vs unpacked([[packed-vs-unpacked-ablation]]), prompt-masked vs full-loss, NEFTune on/off([[neftune]])이며, 각각은 [[karpathy-training-neural-net-recipe]] 기준 single-axis change여야 한다. Data mix는 [[lima]]-plus-synthetic이며 decontaminated eval suite([[tulu-3-sft-mix]])를 기준으로 한다. Deliverable: ablation table과 masking test가 잡은 regression 하나를 담은 `sft-run-memo.md`.

---

## Goal

Peer가 reproducible하게 만들 수 있는 artifact 세 가지:

1. **Tested SFT run.** `checkpoint-final/`과 `pytest` green report on `tests/test_chat_template.py`, `tests/test_loss_mask.py`, `tests/test_packed_attention.py`. 모든 test는 pass하기 전에 적어도 한 번은 fail했어야 한다. Failing output을 fixture로 commit하라.
2. **Ablation grid.** Full-budget이면 eight rows(2×2×2), resource-constrained이면 two rows(packed × unpacked)의 `ablations.json`. Columns: `train_loss_final`, `throughput_tok_per_gpu_per_s`, `mtbench_mean`, 그리고 여섯 [[tulu-3-sft-mix]] bucket별 per-slice breakdown.
3. **Memo.** `sft-run-memo.md`, 한 페이지: ablation table, regression spec, one surprise.

[[karpathy-training-neural-net-recipe]]에 따라 training 전에 모든 ablation delta를 *예측*하라. Surprise가 배운 것이다.

---

## Full-budget path

Target: 8×H100(또는 8×A100-80GB), Llama-3.2-3B base, 100K mix, 8 run 전체에 대해 ~8 h wall-clock.

- **Base.** `meta-llama/Llama-3.2-3B`(3B-Instruct 아님 — chat template는 직접 적용한다). [[fsdp-sft]] 기준 FSDP `FULL_SHARD`; `auto_wrap_policy = transformer_auto_wrap_policy({LlamaDecoderLayer})`; BF16 mixed precision.
- **Mix.** 100K = 1K [[lima]] + 79K ch-29 synthetic + 20K decontaminated No-Robots / OpenAssistant-2. [[tulu-3-sft-mix]] 기준 skill balance: chat 27%, math 21%, code 14%, IF 11%, safety 10%, multilingual/reasoning 17%. MT-Bench에 대해 8-gram overlap ≥ 50% + embedding sim > 0.9로 decontaminate하라.
- **Training.** [[hf-alignment-handbook]] defaults: LR 2e-5 cosine 10% warmup, 1 epoch, global batch 128, `max_seq_length=2048`, packing true, response-only true, `AdamW(0.9, 0.95)`. NEFTune axis가 on일 때 [[neftune]] 기준 NEFTune α=5.
- **Eval.** MT-Bench single-turn(80 prompts), 여섯 [[tulu-3-sft-mix]] bucket별 slice. Judge `gpt-4o-mini`; `packed+masked+neftune-off` baseline 대비 paired win-rate.

## Resource-constrained path

Target: 1 GPU(≥24 GB), Llama-3.2-1B 또는 Qwen2.5-1.5B, 20K mix, 2 run에 ~3 h.

- Mix = 1K LIMA + 15K synthetic + 4K chat; skill balance는 그대로.
- 한 axis: **packed vs unpacked**. NEFTune on, masking on. 이것이 correctness가 가장 중요하고 throughput delta가 가장 큰 axis이다. 정확히 [[packed-vs-unpacked-ablation]] §Diagnostic targets가 말하는 axis이다.
- 세 masking unit test는 여전히 required이다. Lab의 integrity는 test에 있다.

---

## §1 Data mix assembly

`build_mix(spec) -> Dataset`을 작성하라. HF `Dataset`을 반환하며 `{messages, slice, source}`를 가진다. `slice` column이 per-slice regression table을 가능하게 한다. 지금 잊으면 skill-level signal은 복구할 수 없다.

Decontamination은 optional이 아니다. [[tulu-3-sft-mix]] 기준 MT-Bench prompt와 8-gram overlap ≥ 50%인 것은 automatic drop이다. Drop을 log하라. Drop이 > 5%이면 synthetic set이 contaminated된 것이다. Training 전에 regenerate하라.

---

## §2 Chat-template unit tests

[[hf-alignment-handbook]] 기준 #1 silent bug는 template mismatch이다. 코드처럼 test하라.

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

맞닥뜨릴 fail mode: `Llama-3.2`와 `Llama-3.1` template는 다르다. `Qwen2.5`의 `<|im_start|>`는 special token이 추가되어 있지 않으면 literal tag token으로 tokenize된다. [[hf-alignment-handbook]] 기준 debug session마다 한 번은 random packed batch를 verbatim으로 decode하라.

---

## §3 Loss-mask unit tests

[[loss-masking-prompt]] §Implementation 기준: `labels[:prompt_len] = -100`. Test는 (a) prompt ignored, (b) completion included, (c) multi-turn assistant-k only, (d) pack 내부의 per-sub-sequence reset을 확인해야 한다.

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

Fresh pipeline에서 실제로 처음 fail하는 test는 거의 항상 `test_packed_labels_reset_per_subsequence`이다. Packing function은 *첫 번째* prompt를 올바르게 mask하고 나머지에는 mask를 조용히 공유하는 경향이 있다. 이것이 [[packed-vs-unpacked-ablation]]의 Failure Mode 3이다.

---

## §4 Packed-attention unit tests

[[sequence-packing]]은 두 invariant를 준다. (a) `cu_seqlens` offset이 `Σ L_i`로 합산된다. (b) token `t` in sub-sequence `i`는 `[cu_seqlens[i], cu_seqlens[i+1])` 밖의 token에 attention weight가 0이다. 둘 다 training 없이 test할 수 있다.

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

Negative control을 추가하라. `cu_seqlens` 없이 packed tensor 전체에 dense causal attention을 계산하면 sub-seq 2의 output은 standalone sub-seq 2 attention과 *달라야* 한다. Positive case만 있는 suite는 correct packer와 no-op packer를 구분할 수 없다([[packed-vs-unpacked-ablation]]의 Failure Mode 4).

---

## §5 Ablation grid

2×2×2 cube(full-budget); packed-only axis(resource-constrained).

| Run | Packed | Masked | NEFTune | Purpose |
|-----|--------|--------|---------|---------|
| `full` | yes | yes | yes | headline claim |
| `no-pack` | no | yes | yes | isolate throughput cost; [[packed-vs-unpacked-ablation]] predicts parity |
| `full-loss` | yes | no | yes | isolate masking; [[loss-masking-prompt]] predicts MT-Bench drop |
| `no-neft` | yes | yes | no | NEFTune saturation at scale ([[allenai-tulu-sft-recipe]] reports neutral at 939K) |
| `no-pack-no-mask`, `no-pack-no-neft`, `no-mask-no-neft` | mixed | mixed | mixed | interaction controls |
| `none` | no | no | no | worst-case anchor |

Launch 전에 각 delta를 predict하라. Prediction을 `predictions.txt`에 log하고 `git add`한 뒤 contract로 취급하라. MT-Bench standard error를 넘어 prediction과 disagree하는 row는 memo의 한 paragraph가 된다.

---

## §6 Eval harness

- **Harness.** Prompt당 judge call 하나로 80개 MT-Bench prompt를 실행하는 120-line script. 여기서는 `lm-eval-harness`가 overkill이다.
- **Slices.** 여섯 [[tulu-3-sft-mix]] bucket. 80 prompt를 직접 tag하라(budget 20 minutes).
- **Judge.** `gpt-4o-mini`; `packed+masked+neftune-off` baseline과 paired. Paired win-rate + bootstrapped 95% CI를 보고하라.
- **Throughput.** `tokens_per_gpu_per_second`와 `samples_per_second`를 별도로 log하라. Packing의 win은 전자에만 나타난다. Samples/s를 보고하면 packing이 no-op처럼 보인다. [[packed-vs-unpacked-ablation]]은 speedup ≈ L_max / avg(L_i)를 예측한다. Avg(L_i)≈600, L_max=2048인 100K mix에서는 ~2.5–3× realized를 기대하라.

---

## §7 Memo template

`sft-run-memo.md`, 한 페이지, 다섯 section:

1. **Setup.** Base, mix composition, epochs, GPU count, wall-clock.
2. **Ablation table.** 8개(또는 2개) row와 MT-Bench mean, per-slice breakdown, throughput, paired win-rate vs `full`. 모든 delta를 "predicted" 또는 "surprising"으로 tag하라.
3. **Test-caught regression.** Test name, buggy code의 commit SHA, observed failure(mask overlap diff, cross-sample leakage L2 등), fix. Memo가 존재하는 이유인 section이다.
4. **One surprise.** Ablation delta가 prediction과 disagree한 specific per-slice row. "Numbers were noisy"가 아니다. 예: "`no-neft` dropped 4 pts on IF slice but tied on chat — predicted the opposite."
5. **Next instrumentation.** Training step에 도달하기 *전에* 이 regression을 잡기 위해 추가할 test 하나. Test directory는 monotonically grow한다.

---

## Acceptance criteria

Hard gate, 순서대로.

1. Final code에서 `pytest tests/ -v`가 pass하고, `git log`의 prior commit 하나가 세 test file 중 적어도 하나가 먼저 fail했음을 보여준다.
2. `ablations.json`은 full cube(또는 2-row packed-axis cube)를 가진다. Throughput column은 `tokens_per_gpu_per_second`이고 `samples_per_second`가 아니다.
3. 모든 run에서 step-1 loss가 `ln(|V|)`의 20% 이내이다([[karpathy-training-neural-net-recipe]] overfit-a-batch sanity). 그렇지 않으면 chat template가 잘못된 것이다.
4. 첫 100 step에서 `pre_clip_grad_norm < 2`이어야 하며, 이는 `full`에 적용된다. Spike는 prompt token이 loss로 leak된다는 뜻이다.
5. [[packed-vs-unpacked-ablation]] §Diagnostic 기준 `full` vs `no-pack` train-loss curve가 matching step에서 < 0.01 nats 차이여야 한다. 더 큰 gap은 packing property가 아니라 bug이다.
6. Memo §3는 specific regression, commit SHA, failing test, fix diff를 이름 붙인다. SHA가 없으면 capstone을 earned하지 못한 것이다.
7. Memo §2 row 중 적어도 하나가 "surprising"으로 tag되어야 한다. 그렇지 않으면 lab이 아무것도 가르치지 않은 것이다.

---

## Connections

- **ch-30 / ch-31 / ch-32 / ch-33 / ch-34 / ch-35** — 이 lab은 end-to-end instantiation이다. §1은 ch-34의 data-mix discipline을 들어 올리고, §2는 ch-31의 chat-template invariant를 enforce하고, §3은 ch-32의 loss-mask rule을 enforce하고, §4는 ch-33의 packed-attention contract를 enforce하며, §5의 NEFTune axis는 ch-35를 scale에서 test한다.
- **ch-29 (synthetic lab)** — 79K synthetic slice는 그곳의 `instructions.jsonl`에서 온다.
- **Track 4 (RL)** — `checkpoint-final` from `full`은 이어지는 모든 DPO / PPO / GRPO chapter의 `π_ref`가 된다.

## Further reading

- [[sequence-packing]], [[packed-vs-unpacked-ablation]] — varlen API, cu_seqlens, four failure modes, 100-step diagnostic, throughput model.
- [[loss-masking-prompt]] — Shi 2024; response-only dominates; multi-turn masking rule.
- [[neftune]] — uniform α/√(Ld) noise; large mix에서 saturate.
- [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]], [[tulu-3-sft-mix]] — Zephyr + Tülu 3 hparams and decontamination discipline.
- [[lima]] — 1K baseline slice; Superficial Alignment Hypothesis.
- [[karpathy-training-neural-net-recipe]] — predict-before-run; overfit-one-batch; monitor clip-norm.
- [[fsdp-sft]] — FULL_SHARD memory math; `auto_wrap_policy` on `LlamaDecoderLayer`.

## Companion visualization

**[figures/ablation-matrix.html](figures/ablation-matrix.html)** — interactive 2×2×2 ablation cube. 어떤 corner든 클릭하면 illustrative MT-Bench score와 [[packed-vs-unpacked-ablation]], [[loss-masking-prompt]], 또는 [[neftune]]에 grounded된 justification이 나타난다. Prediction aid로 사용하라. Cube를 클릭해 보고 각 justification을 읽은 *다음* `predictions.txt`를 작성하라. Score는 illustrative이고, arrow direction은 attested이다.
