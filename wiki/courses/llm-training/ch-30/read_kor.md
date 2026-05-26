<!-- chapter: ch-30
     track: sft
     kind: content
     title: SFT Design Axes
     deps: [ch-29]
     sources: [[loss-masking-prompt]], [[sequence-packing]], [[neftune]], [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]], [[tulu-3-sft-mix]], [[packed-vs-unpacked-ablation]], [[rlhf-instructgpt]], [[llama-3]], [[qwen-3]], [[deepseek-v3]], [[deepseek-r1]]
     figures: figures/sft-axes.html
     opens_track: sft (ch-30..ch-36)
-->

# 30장 — SFT Design Axes

> **핵심 통찰.** SFT는 하나의 algorithm이 아니다. Regime, loss mask, chat template, packing strategy, embedding noise라는 five-dimensional design space의 한 점이다. 그리고 downstream RL / DPO / RLVR run은 이 axis들이 base model에 새긴 surface를 모두 상속한다. 당신이 고른 template, mask한 token, 학습한 turn은 RL이 나중에 무엇을 optimise할 수 있고 무엇을 할 수 없는지 미리 결정한다. RL은 SFT manifold를 따라 policy를 *이동*시킬 뿐, 새로 만들지 않는다.
>
> **가이드라인.** Run 전에 각 axis를 명시적으로 결정하라. (1) regime(instruction / chat / tool / reasoning / agentic)은 어떤 token이 존재하는지 결정한다. (2) loss mask(response-only가 [[loss-masking-prompt]]와 [[hf-alignment-handbook]] 기준 default). (3) chat template(target base model, 즉 Llama-3, Qwen, DeepSeek의 것을 verbatim 사용하고 ad-hoc으로 edit하지 말 것). (4) packing(default on, `flash_attn_varlen_func` + `cu_seqlens`, [[sequence-packing]] 및 [[packed-vs-unpacked-ablation]] 기준). (5) NEFTune(dataset ≤ 100K에서는 on, ≥ 500K에서는 off — [[neftune]] 및 [[allenai-tulu-sft-recipe]]). Five-tuple을 one-line config header에 문서화하고, 실제 run 전에 packed batch 하나를 text로 decode해 validate하라.

---

## 이 장이 필요한 이유

Synthetic track(ch-18..ch-29)은 disk 위 instruction rows pool로 끝났다. RL track(ch-37..ch-46)은 policy checkpoint에서 시작한다. 둘 사이의 모든 chapter가 SFT이며, SFT track(ch-30..ch-36)은 policy가 태어나는 곳이다. 이 opening chapter는 axis의 이름을 붙인다. Ch-31(iterative SFT↔RL bridges), ch-32(reasoning SFT and long-CoT cold-start), ch-33(tool-call SFT), ch-34(agentic SFT), ch-35(distillation), ch-36(SFT lab)은 각각 이 axis 중 하나 이상을 vary할 것이다. 그리고 ch-35의 distillation recipe는 teacher가 어떤 axis를 고정했는지 알고 나서야 의미가 있다.

`pre-determines` framing이 중요한 이유는 SFT track이 care하는 behaviour에 cheap, dense gradient를 얻을 수 있는 pipeline의 마지막 지점이기 때문이다. RL에는 sparse reward와 π_SFT에 대한 KL tether가 있다([[rlhf-instructgpt]] Equation 2: `r(x,y) − β · log π_RL/π_SFT`). Behaviour가 π_SFT 아래서 plausible completion도 아니라면, reward가 앞으로 당기기 전에 KL penalty가 policy를 되돌린다. SFT는 RL이 탐색하는 manifold의 generator다.

---

## §1. Regime — SFT가 취할 수 있는 다섯 형태

하나의 "SFT" run은 다섯 regime 중 어떤 것도 구현할 수 있다. 차이는 어떤 token이 존재하는지, 어디에 loss가 걸리는지, template이 무엇을 표현해야 하는지다.

| Regime | Example input | Example target | Tokens introduced |
|--------|---------------|----------------|-------------------|
| Single-turn instruction | "Translate to French: hello" | "bonjour" | user, assistant |
| Multi-turn chat | alternating `user` / `assistant` | final (or per-turn) `assistant` | system, user, assistant |
| Tool-call | user prompt | `<tool_call>{"name":..., "arguments":...}</tool_call>` + assistant reply | `tool_call`, `tool_response` (or role `tool`) |
| Reasoning | math / code prompt | `<think>…</think><answer>…</answer>` per [[deepseek-r1]] template | `<think>`, `</think>` |
| Agentic | user task + tool catalog | interleaved thought / tool-call / observation trace | all of the above + loop structure |

선택한 regime는 downstream의 모든 것을 결정한다. Single-turn instruction SFT로 학습한 model은 `<tool_call>` block을 emit할 수 없다. `<tool_call>`에 대한 token distribution이 gradient를 본 적이 없기 때문이다. Ch-31은 SFT→RL bridge가 regime별로 작동한다는 것을 보여 준다. Rejection-sampling SFT(Llama-2 appendix, [[llama-3]]에서 이어짐)는 SFT regime가 이미 support하는 behaviour에만 mass를 더한다.

---

## §2. Loss mask — 무엇이 cross-entropy에 기여하는가

Regime별 mask table. `Y` = loss에 포함, `N` = `-100`으로 mask, `opt` = toggle 가능하지만 attested default가 starred value.

| Regime | system | user | assistant (current turn) | assistant (prior turns) | tool_call | tool_response | thought (`<think>`) |
|--------|--------|------|--------------------------|-------------------------|-----------|---------------|----------------------|
| Single-turn instruction | N | N | **Y** | — | — | — | — |
| Multi-turn chat | N | N | **Y** | N (mask) | — | — | — |
| Tool-call | N | N | **Y** | N | **Y** (the call) | N (observation) | — |
| Reasoning | N | N | **Y** | N | — | — | **Y** (the CoT is signal) |
| Agentic | N | N | **Y** | N | **Y** | N | **Y** |

Invariant: system과 user는 항상 mask된다. Tool-response / observation tokens는 항상 mask된다(environment에서 오며 model이 emit하지 않음). *Emitted* artefacts(assistant text, tool-call JSON, thought content)는 항상 loss를 갖는다. 문헌의 sharp ablation 하나는 [[loss-masking-prompt]] / Shi 2024다. Tiny-dataset / strong-base regime에서 full-sequence가 mild continued-pretraining처럼 작동하는 예외를 제외하면 response-only SFT가 MT-Bench와 AlpacaEval에서 full-sequence SFT를 엄격히 이긴다. 이것은 모순이 아니다. SFT objective가 너무 좁으면 pretrain-style loss를 leak하는 것이 도움이 된다는 신호다. 그래서 [[rlhf-instructgpt]]는 이후 PPO에 `γ · L_ptx`를 추가한다.

[[loss-masking-prompt]]에서 바로 온 minimal implementation:

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

Multi-turn extension([[loss-masking-prompt]] §Multi-turn chat masking 기준): conversation `[u_1, a_1, u_2, a_2, ..., u_k, a_k]`에서 모든 `u_i`를 mask하고, `a_1..a_{k-1}`을 mask하며, `a_k`만 학습한다. Per-turn-training variant는 conversation을 k번 unroll하고, `a_{i-1}`까지 mask한 뒤 `a_i`를 학습한다. Loss value는 같고 data는 k× 더 많다.

---

## §3. Chat template — base model이 이미 말하는 surface

Chat template는 cosmetic하지 않다. Tokenizer가 special-token ID로 mapping하는 literal strings(tokens, not text)의 set이며, base model의 post-training이 이미 condition된 surface다. 잘못된 template 사용은 silent-bug class다. Decoding은 괜찮아 보이고 training loss도 괜찮아 보이지만 downstream win-rate가 5–20 pts collapse한다. [[hf-alignment-handbook]]은 이를 "#1 silent bug"라고 부르며, launch 전에 packed batch를 text로 decode해 delimiter를 눈으로 확인하라고 지시한다.

Shipping family 네 곳의 chat-template matrix. 모든 string은 해당 model report / tokenizer config에서 verbatim으로 인용된다.

| Family | BOS | Role delimiter | Turn-end | Thinking tokens | Tool-call tokens |
|--------|-----|----------------|----------|-----------------|-------------------|
| ChatML (OpenAI / early Mistral) | none / `<s>` | `<\|im_start\|>role\n` | `<\|im_end\|>\n` | none native | convention only |
| Llama-3 ([[llama-3]]) | `<\|begin_of_text\|>` | `<\|start_header_id\|>role<\|end_header_id\|>\n\n` | `<\|eot_id\|>` | none native | `<\|python_tag\|>` (Llama-3.1 extension) |
| Qwen-3 ([[qwen-3]]) | none | `<\|im_start\|>role\n` | `<\|im_end\|>\n` | `<think>...</think>` scoped inside assistant | `<tool_call>{...}</tool_call>` inside assistant |
| DeepSeek-V3 ([[deepseek-v3]]) | `<｜begin▁of▁sentence｜>` | `<｜User｜>...<｜Assistant｜>` | `<｜end▁of▁sentence｜>` | DeepSeek-R1 adds `<think>...</think>` in assistant | convention inside assistant |

각 family에서 user/assistant exchange 하나를 rendering한 예. Delimiter는 verbatim이다.

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

반드시 지켜야 할 세 consequence:

1. **모든 delimiter는 real token ID다.** Llama-3의 `<|eot_id|>`는 one token이다. ChatML의 `<|im_end|>`도 one token이다. 서로 교체 가능한 substring이 아니다.
2. **Base model의 generation은 native end-of-turn에서 멈춘다.** Llama-3 base를 ChatML로 학습하면 post-SFT model은 `<|im_end|>`가 config의 stop token이 아니기 때문에 끝없이 생성한다.
3. **Tool-call과 thinking token은 assistant turn *안에* 산다.** 별도 role이 아니다. Assistant가 `<think>...</think>`와 `<tool_call>{...}</tool_call>`를 하나의 continuous span 일부로 emit하고, turn-end token으로 종료한다.

Attested practitioner move: string을 직접 만들지 말고 항상 `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=...)`를 호출하라. Template는 이유가 있어서 tokenizer의 `tokenizer_config.json`에 ship된다. 그것이 base model과의 contract다.

---

## §4. Packing — throughput axis이면서 correctness axis

Raw instruction batch는 50–89%가 padding이다([[sequence-packing]] abstract; BERT/GLUE에서 측정되어 SFT로 이어짐). Packing은 짧은 sequence를 fixed-length block으로 concatenate하고 block-diagonal attention mask 및 sub-sequence별 position-ID reset을 사용한다. Attested speedup:

```
speedup ≈ L_max / avg(L_i)
# [[packed-vs-unpacked-ablation]] §Throughput model
```

`avg(L_i) = 600`, `L_max = 4096`인 SFT mixture라면 expected 6×, FlashAttention overhead 후 realised 2.5–3×다.

Correctness contract는 [[sequence-packing]] §Mechanics에서 온다.

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

[[packed-vs-unpacked-ablation]] §Failure modes에 enumerated된 네 failure mode:

1. Missing block-diagonal mask → sub-sequence 2의 token이 sub-sequence 1에 attend → softmax partition이 documents 사이에서 leak.
2. Un-reset position IDs → sub-sequence 2가 positions L_1..L_1+L_2를 봄. RoPE가 position-shifted.
3. Label mask not re-applied per sub-sequence → sub-sequence 2의 prompt tokens가 loss에 기여.
4. `flash_attn_varlen_func` 대신 `flash_attn_func`(dense)를 사용 → mask가 전혀 없음. Silent contamination.

Diagnostic: 같은 seed와 data로 unpacked 100 steps, packed 100 steps를 train한다. Loss curve는 matched step에서 0.01 nats 이내로 agree해야 한다. 더 큰 gap은 quality drop이 아니라 bug다.

---

## §5. NEFTune과 prompt-masked-vs-full-sequence axis

작지만 실제로 조정 가능한 knob 두 가지:

**NEFTune**([[neftune]]). Pre-transformer에 추가되는 one line:

```python
# neftune.py
import torch

def neftune(embeds: torch.Tensor, alpha: float = 5.0) -> torch.Tensor:
    L, d = embeds.shape[-2], embeds.shape[-1]
    eps = (torch.rand_like(embeds) * 2 - 1) * (alpha / (L * d) ** 0.5)
    return embeds + eps
```

Attested numbers: plain Alpaca 위 LLaMA-2-7B, AlpacaEval 29.79 → 64.69 with `α=5`. Evol-Instruct / ShareGPT / OpenPlatypus에서는 +8–10 pts. Gain은 scale에서 saturate한다. [[allenai-tulu-sft-recipe]]는 939K SFT prompts에서 NEFTune이 neutral하다고 본다. Dataset-size rule: `|D| ≤ 100K`이면 default on, `|D| ≥ 500K`이면 off, 그 사이는 불확실하다. 중요하다면 held-out 500-prompt probe에서 `α ∈ {5, 10, 15}`를 sweep하라.

**Loss weighting**. Full-sequence loss는 `L_full = −(1/(T_p+T_y)) · Σ log π(x_t|x_<t)`이고 response-only는 `L_SFT = −(1/T_y) · Σ log π(y_t|p, y_<t)`다. Shi 2024([[loss-masking-prompt]]에서 cite)는 Alpaca / ShareGPT / LIMA 및 여러 scale에서 response-only, full-sequence, prompt-weighted(`α · L_prompt + L_response`, `α < 1`) 세 variant를 ablate했다. Response-only가 우세하고, prompt-weighted는 일부 slice에서 modest gain을 주며, full-sequence는 대체로 진다. Default는 response-only다. 예외는 strong base 위 tiny-dataset(≤ 1K) SFT로, full-sequence가 mild continued-pretrain regulariser처럼 작동한다.

---

## §6. Two production recipes — "on by default"의 실제 의미

[[hf-alignment-handbook]] Zephyr-7B-β SFT(Mistral-7B base, UltraChat-200K):

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

[[allenai-tulu-sft-recipe]] / [[tulu-3-sft-mix]] Tülu-3 SFT(Llama-3.1 base, 939K mix):

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

Axis가 같은 label에 mapping되지만 value가 다른 것을 보라. 이것이 이 장의 point다. Axis는 stable하지만 default는 그렇지 않다. Data scale은 NEFTune을 "on"에서 "off"로 옮긴다. Base-model family는 template를 ChatML에서 Llama-3로 옮긴다. 70B parameter count는 learning rate를 절반 이하로 낮춘다. Ch-36(SFT lab)은 당신 자신의 run에 대해 이 table을 채우고 모든 deviation을 justify하게 할 것이다.

---

## §7. SFT가 미리 결정하는 것 — 그리고 RL이 나중에 고칠 수 있는 것

SFT는 학습한 모든 behaviour에 positive probability mass를 할당하고, 나머지 behaviour는 base model prior에 남긴다. RL([[rlhf-instructgpt]]의 KL tether가 있는 PPO, RL track의 DPO / GRPO)은 이 manifold를 따라 probability를 움직인다.

- **SFT-fixable**: surface format adherence(turn delimiters, JSON structure, `<think>` tags), style, verbosity, mandatory refusals, vocabulary-level token choices, SFT data에 존재하는 language coverage.
- **RL-fixable(SFT가 어려워하는 것)**: π_SFT support 안의 well-formed completion 둘 사이의 preference, reward-verifiable correctness(math / code execution), single ground-truth가 아니라 pairwise ranking에 의존하는 helpfulness tradeoffs.
- **둘 다 잘 못 고침**: SFT와 base model prior 모두에 없는 behaviour. SFT가 tool-call을 emit한 적이 없다면 PPO는 reasonable temperature에서 찾을 수 없다. Rollout log-prob가 사실상 0이기 때문이다.

Pre-determines framing으로 장을 닫는다. Ch-30에서 regime과 template을 고르고, ch-31..ch-35에서 mix를 채우며, ch-36에서 lab을 실행한다. 그러면 ch-37..ch-46에서는 여기서 shaping한 manifold를 상속한다.

---

## Connections

- **ch-29** — synthetic-set lab은 이 장에 "single-turn instruction" pool을 제공한다. Regime과 template은 그 pool이 base model과 만나는 지점이다.
- **ch-31** — iterative SFT↔RL bridges(RSFT, Best-of-N SFT, STaR)는 current policy에서 새 synthetic data를 생성해 "regime" axis를 vary한다.
- **ch-32** — reasoning SFT와 long-CoT cold-start는 table의 "thought-token mask" row를 심화한다([[deepseek-r1]] template; [[qwen-3]] hybrid-thinking).
- **ch-33** — tool-call SFT는 `tool_call` / `tool_response` columns를 load-bearing하게 만든다. Chat-template matrix에는 여섯 번째 column이 붙는다.
- **ch-34** — agentic SFT는 여러 tool-call turn을 chain한다. Loss mask는 interleaved observations와 thoughts를 견뎌야 한다.
- **ch-35** — distillation은 teacher의 regime과 template을 상속한다. SFT design axes는 "which teacher axes do I adopt verbatim"가 된다.
- **ch-36** — SFT lab은 이 장을 runnable code로 쓴 것이다.
- **ch-37..ch-46 (RL track)** — π_SFT 위에서 작동한다. 여기서 고정한 모든 axis가 RL이 움직일 수 있는 범위를 bound한다.

## Further reading

- [[loss-masking-prompt]] — Shi 2024; response-only vs full-sequence ablation; multi-turn masking rule.
- [[sequence-packing]] — Krell 2021; SPFHP, cu_seqlens, position-ID reset, varlen kernel contract.
- [[packed-vs-unpacked-ablation]] — 2021–2024 ablations compilation; 네 packing failure modes; diagnostic procedure.
- [[neftune]] — Jain 2023; one-line embedding noise; α=5 default; saturation at scale.
- [[hf-alignment-handbook]] — Zephyr recipe; `SFTConfig(packing=True, train_on_response_only=True)`; decode-a-packed-batch lesson.
- [[allenai-tulu-sft-recipe]] / [[tulu-3-sft-mix]] — 939K mix; skill-level ablations; NEFTune saturation.
- [[rlhf-instructgpt]] — SFT를 RL이 걷는 manifold로 만드는 β · KL term.
- [[llama-3]] — Llama-3 chat template; iterative SFT/RS/DPO rounds.
- [[qwen-3]] — hybrid-thinking template; `<|im_start|>` / `<think>` scoping.
- [[deepseek-v3]] / [[deepseek-r1]] — DeepSeek delimiter family; rule-based reward parseability를 위해 선택된 `<think>`/`<answer>` template.

## Companion visualization

**[figures/sft-axes.html](figures/sft-axes.html)** — self-contained interactive config builder. 다섯 axis(regime / mask / template / packing / NEFTune)를 toggle하면 resulting training config summary, highlighted per-regime loss-mask row, template의 verbatim delimiters, `L_max / avg(L_i)` entry에서 recompute된 expected packing speedup을 볼 수 있다. Ch-36 lab 전에 pre-flight checklist로 사용하라. Summary의 axis 중 하나라도 "unset" 또는 "inferred"라면 아직 SFT를 launch할 준비가 되지 않았다.
