<!-- chapter: ch-04
     track: foundations
     title: Sequence Packing, Masking, and Chat Templates
     sources: [[sequence-packing]], [[loss-masking-prompt]], [[neftune]], [[hf-alignment-handbook]], [[flash-attention]]
     figures: figures/packed-attention.html
-->

# 4장 — Sequence Packing, Masking, Chat Template

> **핵심 통찰.** SFT batch는 기본적으로 50–89%가 padding이다. 여러 짧은 sequence를 fixed-length block으로 packing하고, cross-sequence attention을 금지하는 attention mask와 prompt token을 숨기는 loss mask를 붙이면 정확도 손실 없이 그 throughput을 회복한다. 대부분의 SFT bug는 pack, mask, chat template 사이의 틈에 산다.
>
> **지침.** SFT data는 항상 FlashAttention의 varlen API(`flash_attn_varlen_func` + `cu_seqlens`)로 packing하라. `ignore_index=-100`으로 current assistant가 아닌 모든 token을 loss에서 mask하라. chat template은 code가 아니라 model-specific data로 취급하라. 모든 training run에서 rendered sample을 log하라.

---

## 이 장이 필요한 이유

서로 다른 세 버그가 loss curve에서는 똑같이 보인다(기대보다 약간 나쁘지만 정상적으로 수렴). 그리고 각각은 다른 어떤 오류 부류보다 더 많은 SFT compute 낭비를 만든다.

1. **Block-diagonal attention mask 없이 packing** — sequence 2의 token이 sequence 1을 attend한다. loss 값은 그럴듯하지만 모든 pack의 모든 token에서 gradient가 틀린다.
2. **Per-subsequence label masking 없이 packing** — concatenate할 때 mask reset을 잊어서 pack member 3의 prompt token이 loss를 오염시킨다.
3. **Chat template mismatch** — training은 `<|im_start|>` + `<|im_end|>`를 쓰고 eval은 `### Human:` + `### Assistant:`를 쓴다. Model은 training 중에는 완벽하게 render하지만 eval에서 format을 hallucinate한다.

세 문제 모두 [[sequence-packing]] + [[loss-masking-prompt]]에 설명된 pipeline을 제대로 구현하면 사라진다. 이 장은 그 운영 blueprint다.

---

## 1. Packing이 존재하는 이유

SFT dataset은 길이 분포가 심하게 right-skewed하다. LIMA / Alpaca / UltraChat 계열 sample은 평균 약 500 token이지만 tail은 4K까지 닿는다. Naive batching은 모든 sequence를 batch에서 가장 긴 sequence까지, 더 나쁘게는 `max_seq_len`까지 padding한다. 일반적인 padding waste는 chat mixture에서 50%, short-instruction set에서는 최대 89%다. GPU 시간이 2배 낭비된다.

Packing은 batch를 1D bin-packing 문제로 다룬다. 여러 짧은 sequence를 끝에서 끝으로 concatenate하여 길이 `max_seq_len`의 block으로 만든다.

```
pack = [ s_1 | s_2 | s_3 | ... | s_n | PAD ]
```

순진하게 하면 두 가지가 깨진다. **Forward**는 `s_2`의 token이 `s_1`의 token을 attend하기 때문에 깨진다(softmax partition function이 document 경계를 넘어 leak된다). **Loss**는 `(prompt, response)` split이 per-subsequence인데 label tensor는 per-pack이기 때문에 깨진다.

둘 다 고치면 packing은 unpacked training과 수학적으로 같고 throughput은 2배가 된다.

---

## 2. 올바른 packed block — position ID, `cu_seqlens`, attention mask

출처: [[sequence-packing]]. packed block의 runtime view:

```
tokens        = [t_0 t_1 t_2 | t_3 t_4 | t_5 t_6 t_7 t_8 | PAD PAD]
position_ids  = [ 0   1   2  |  0   1  |  0   1   2   3  |  0   0]
cu_seqlens    = [0, 3, 5, 9, 11]                # cumulative boundaries
```

`position_ids`는 각 sub-sequence boundary에서 **0으로 reset**된다. RoPE와 learned position은 이것에 의존한다.

attention mask는 각 block *안*에서는 causal structure를 갖는 block-diagonal 형태다.

```
       t_0 t_1 t_2 t_3 t_4 t_5 t_6 t_7 t_8
t_0 [   ✓   .   .   .   .   .   .   .   . ]
t_1 [   ✓   ✓   .   .   .   .   .   .   . ]
t_2 [   ✓   ✓   ✓   .   .   .   .   .   . ]    ← end of s_1
t_3 [   .   .   .   ✓   .   .   .   .   . ]
t_4 [   .   .   .   ✓   ✓   .   .   .   . ]    ← end of s_2
t_5 [   .   .   .   .   .   ✓   .   .   . ]
t_6 [   .   .   .   .   .   ✓   ✓   .   . ]
t_7 [   .   .   .   .   .   ✓   ✓   ✓   . ]
t_8 [   .   .   .   .   .   ✓   ✓   ✓   ✓ ]    ← end of s_3
```

대화형 시각화는 `figures/packed-attention.html`을 보라.

**이 mask를 절대 materialise하지 마라.** `max_seq_len=4096`이면 dense mask는 4096² = 16M entry다. 대신 `cu_seqlens`를 받아 mask를 만들지 않고 block-diagonal causal attention을 계산하는 FlashAttention의 varlen interface를 사용하라.

```python
from flash_attn import flash_attn_varlen_func

# cu_seqlens_q, cu_seqlens_k: 1-D int32 tensors of cumulative offsets
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_len,
    max_seqlen_k=max_len,
    causal=True,
)
```

Memory는 O(Σ Lᵢ)이지 O(max² · batch)가 아니다. 이것이 128k+ context SFT에서 packing이 가능한 이유다.

---

## 3. Packing algorithm — 실무의 SPFHP

[[sequence-packing]]은 두 algorithm을 문서화한다. production에서는 SPFHP(Shortest-Pack-First Histogram-Packing)가 기본값이다. O(N log N)이고 >99% fill을 달성하기 때문이다.

1. Dataset의 length histogram을 계산한다.
2. target size `max_seq_len`의 각 bin에 대해, *아직 남아 있으면서 들어갈 수 있는 가장 긴 sequence*를 greedily 배치한다.
3. 들어갈 수 있는 sequence가 없으면 padding으로 bin을 닫고 새 bin을 시작한다.

최소 reference implementation(TRL의 `DataCollatorWithPacking`이 동등한 일을 한다):

```python
def pack_sequences(sequences, max_len):
    # sort by length descending for first-fit-decreasing
    seqs = sorted(sequences, key=len, reverse=True)
    bins = []       # list of lists
    for s in seqs:
        for b in bins:
            if sum(len(x) for x in b) + len(s) <= max_len:
                b.append(s); break
        else:
            bins.append([s])
    return bins
```

참고로 NNLSHP(Non-Negative Least Squares)는 histogram이 주어졌을 때 수학적으로 최적인 fill을 달성한다. 마지막 0.5%가 중요하지 않다면 과하다.

---

## 4. Loss masking — bug surface의 나머지 절반

출처: [[loss-masking-prompt]]. SFT loss에 어떤 token이 기여하는지는 세 규칙이 지배한다.

1. **Pad token: label = -100.** `F.cross_entropy(..., ignore_index=-100)`가 무시한다.
2. **Prompt / user token: label = -100.** prompt token을 학습하는 것은 inference에서 model이 생성하지 않는 distribution에 capacity를 낭비하는 것이다.
3. **Multi-turn의 prior-turn assistant token: label = -100.** turn *k*의 loss를 계산할 때 1..k−1 turn은 prompt의 일부다.

표준 loss:

```
L_SFT(θ) = −(1 / T_y) Σ_{t=1..T_y} log π_θ(y_t | p, y_<t)
```

코드에서는 다음과 같다.

```python
labels = input_ids.clone()
labels[:prompt_len] = -100            # mask the prompt
loss = F.cross_entropy(
    logits[..., :-1, :].reshape(-1, V),
    labels[..., 1:].reshape(-1),
    ignore_index=-100,
)
```

**Packed block 내부**에서 mask는 per-subsequence다.

```python
labels = input_ids.clone()
for (start, end), prompt_len_i in zip(subseq_ranges, prompt_lens):
    labels[start : start + prompt_len_i] = -100          # mask prompt of sub-i
    labels[start + prompt_len_i : end]   = input_ids[start + prompt_len_i : end]
# PAD region stays -100 from clone + a final labels[-pad_len:] = -100
```

이것을 놓치면 "user input" token 위에서 loss가 계산된다. 이는 helpfulness benchmark에서 SFT를 조용히 0.3–1.0% degrade한다. Shi 2024 연구([[loss-masking-prompt]])는 이를 formalize한다. response-only loss가 dataset size 전반에서 우세하며, full-sequence loss가 도움이 되는 유일한 regime은 tiny-dataset + strong-base-model이다.

**Multi-turn masking**(turns = `[u_1, a_1, u_2, a_2, ..., u_k, a_k]`):

- user turn은 **모두** mask한다.
- prior assistant turn `a_1..a_{k-1}`은 **모두** mask한다.
- `a_k` token만 학습한다.

대안인 "per-turn unrolling" 변형은 각 `a_i`를 차례로 학습하여 하나의 conversation에서 k배의 training sample을 만든다.

### NEFTune — 공짜 accuracy lift

[[neftune]]: SFT 중 embedding layer에 작은 uniform noise를 더한다(학습 중에만). 한 줄이다.

```python
# forward(self, input_ids):
embeds = self.embed_tokens(input_ids)
if self.training and self.neftune_alpha > 0:
    d = embeds.size(-1)
    mag = self.neftune_alpha / math.sqrt(d)
    embeds = embeds + torch.empty_like(embeds).uniform_(-mag, mag)
```

일반적인 `neftune_alpha = 5`. 보고된 AlpacaEval gain은 Llama-2-7B에서 약 10포인트다. packing + masking과 함께 쓸 수 있다. noise는 token lookup 뒤, transformer 앞에 더해진다. 비용은 0이다.

---

## 5. Chat template — model-specific data

Chat template은 conversation을 하나의 string으로 rendering하는 정확한 tokenised 형식이다. 누가 언제 말했는지, assistant turn이 어디서 시작하는지, end-of-turn을 어떻게 알릴지를 encode한다. 현대 model은 tokenizer와 함께 canonical template을 제공한다. 합리적으로 보이지만 *호환되지 않는* 두 template은 서로 대화할 수 없는 두 model을 만든다.

Llama-3 template(단순화):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>
```

Qwen ChatML:

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

Anthropic(ChatML-style marker 없음; role prefix inline)는 또 다르다.

**운영 규칙 세 가지.**

1. **tokenizer의 `apply_chat_template`를 사용하라.** string을 손으로 만들지 마라. tokenizer는 template을 `tokenizer.chat_template`(Jinja2 string)에 저장한다. HF Alignment Handbook([[hf-alignment-handbook]])은 `train_on_response_only=True`와 함께 이것을 사용해 prompt mask를 자동 생성한다.
2. **모든 training run에서 rendered sample을 log하라.** 모든 training log의 첫 줄은 random conversation 하나에 대한 rendered template이어야 하며, loss-mask position이 highlight되어야 한다. step 100 전에 template bug의 90%를 잡는다.
3. **Eval은 정확히 같은 template을 사용한다.** Chat-template leakage, 즉 eval input을 training과 다른 template으로 rendering하는 것은 top-3 silent regression mode다. template을 eval harness에 포함하라.

**Special token: `<think>`, `<tool_call>`, reasoning tag.** model이 thinking tag(Qwen 3, Phi-4-reasoning, R1 distill variant)를 emit하도록 학습할 때 special token은 SFT *전에* tokenizer vocabulary에 있어야 한다. 나중에 추가하면 embedding resize + new row re-init이 필요하다. 이는 thinking-mode model이 `<think>` block 안에 noisy token을 hallucinate하게 만드는 흔한 버그다.

---

## 6. 완전한 올바른 pipeline — copy-paste reference

```python
import torch, torch.nn.functional as F, math
from flash_attn import flash_attn_varlen_func

def build_packed_batch(conversations, tokenizer, max_len=4096):
    """Return input_ids, labels, cu_seqlens, position_ids for one pack."""
    ids, labels, positions, offsets = [], [], [], [0]
    for conv in conversations:
        # 1. Render via tokenizer's canonical chat template.
        text   = tokenizer.apply_chat_template(conv, tokenize=False,
                                               add_generation_prompt=False)
        # 2. Tokenize with completion mask (HF returns a response-only mask).
        enc    = tokenizer.apply_chat_template(
            conv, tokenize=True, return_assistant_tokens_mask=True,
        )
        tok    = enc["input_ids"]
        is_ass = enc["assistant_masks"]          # 1 where token is assistant
        if len(ids) + len(tok) > max_len:
            break                                # pack full; next pack
        ids.extend(tok)
        labels.extend([t if m else -100 for t, m in zip(tok, is_ass)])
        positions.extend(range(len(tok)))
        offsets.append(offsets[-1] + len(tok))
    # pad
    pad_len = max_len - len(ids)
    ids      += [tokenizer.pad_token_id] * pad_len
    labels   += [-100] * pad_len
    positions += [0] * pad_len
    return (torch.tensor(ids), torch.tensor(labels),
            torch.tensor(offsets, dtype=torch.int32),
            torch.tensor(positions))

def packed_forward(model, ids, labels, cu_seqlens, positions):
    # model's attention replaces the dense SDPA with flash_attn_varlen_func,
    # passing cu_seqlens and positions through.
    logits = model(input_ids=ids, position_ids=positions,
                   cu_seqlens=cu_seqlens).logits
    return F.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.size(-1)),
        labels[..., 1:].reshape(-1),
        ignore_index=-100,
    )
```

---

## 연결과 다음 내용

- **[[sequence-packing]] / [[flash-attention]]** — varlen API가 enabling primitive다.
- **[[loss-masking-prompt]]** — Shi 2024의 response-only vs full-sequence loss ablation.
- **[[neftune]]** — noise injection. 위 요소들 위에 공짜 accuracy를 얹는다.
- **ch-05 (FSDP)** — packing은 per-step memory를 줄이고, 그것이 FSDP sharding을 효과적으로 만든다. 둘은 항상 짝을 이룬다.
- **ch-33 (Case Study C)** — distillation SFT(R1-distill, Bespoke-Stratos)는 긴 `<think>` trace를 제공한다. special-token pre-registration과 template design은 거기서 load-bearing이다.
- **ch-34 (SFT lab)** — masking unit test는 SFT lab의 필수 deliverable이다.

## 더 읽을거리

- [[sequence-packing]] — Krell 2021; packing formalism과 algorithm.
- [[loss-masking-prompt]] — Shi 2024 + HF Alignment Handbook convention.
- [[hf-alignment-handbook]] — production `apply_chat_template` + `train_on_response_only=True` recipe.
- [[allenai-tulu-sft-recipe]] — Tülu 3 scale의 packing + masking.

## 함께 보는 시각화

**[figures/packed-attention.html](figures/packed-attention.html)** — interactive block-diagonal attention mask visualiser. sub-sequence length를 넣으면 attention mask와 `cu_seqlens`가 실시간으로 업데이트되고, sequence를 추가하거나 제거할 때 padding-fraction gauge가 어떻게 움직이는지 볼 수 있다.
