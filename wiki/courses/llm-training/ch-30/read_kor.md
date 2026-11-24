<!-- chapter: ch-30
     track: sft
     kind: content
     title: SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate
     deps: [ch-29e, ch-04]
     sources: [[loss-masking-prompt]], [[loss-masking-regimes]], [[neftune]], [[sequence-packing]], [[sequence-packing-contract]], [[hf-alignment-handbook]], [[tulu-3]], [[tulu-3-sft-recipe]], [[open-instruct-allenai-recipes]], [[chat-template-matrix]], [[smol-training-playbook]], [[lora-learns-less-forgets-less]], [[lora-without-regret]], [[minimax-m2-interleaved-thinking]], [[glm-5]], [[orpo]], [[qwen-3]], [[llama-3]], [[llama-3-recipe]], [[qwen-2.5-recipe]], [[deepseek-r1]], [[deepseek-r1-recipe]], [[in2-film]]
     figures: figures/sft-axes.html
     revised: 2026-09 (generality revision)
-->

# Chapter 30 — SFT 설계 선택과 일반화 효과: Masking, Packing, Template, Epoch, Learning Rate

> **핵심 통찰.** 구현 세부사항처럼 보이는 supervised fine-tuning(SFT) 선택은 model이 instruction data에 과적합하는 정도를 바꾼다. 약 1K–70K 예시의 single-turn instruction set에서 instruction token에도 loss를 주면 7개 dataset 중 6개에서 AlpacaEval 1.0이 올랐고, embedding noise를 더하면 4개 dataset 모두에서 AlpacaEval이 올랐다. 각 group이 측정한 dataset에서는 두 방법 모두 더 높은 training loss, 더 낮은 held-out loss, training response와의 더 낮은 overlap을 보였다([[loss-masking-prompt]] Table 1, Fig. 3, Table 2; [[neftune]] Table 1, Figs. 4-5). Trainer 설정도 결과를 바꾼다. Tülu 3는 training setup 간 gap을 loss aggregation까지 추적한 뒤 sum loss를 선택했고, 서로 다른 seed의 SFT run은 8B에서 평균 0.3점, 70B에서 2.6점을 포괄한다([[tulu-3]] §4.3.2, Table 14).
>
> **지침.** SFT set이 작거나(약 1K-14K 예시), output에 비해 instruction이 길다면 instruction에 대한 loss와 response-only loss를 비교하라. Shi et al.이 이 regime에서 가장 큰 gain을 측정했기 때문이다. 그 외에는 모든 assistant turn에 assistant-only loss를 유지하라. 이것이 open-instruct default이고 SmolLM3는 user turn masking이 평가 대부분을 개선했다고 보고하기 때문이다. 공개된 learning rate를 복사할 때는 loss aggregation(sum 또는 mean)도 같이 복사하라. Tülu 3는 sum loss에서 \(5\times10^{-6}\)을 선택했다. 작은 dataset을 packing할 때는 같은 optimizer step 수의 unpacked run과 비교하라. Packing은 epoch당 step 수를 줄이기 때문이다. 모든 설정은 development suite에서 고르고, development 중 보지 않은 held-out suite에 보고하라.

## 학습한 버전에 대한 정정

1. “RL은 SFT가 만든 것을 *따라* policy를 움직이며 다시 만들지 않는다”, “SFT가 tool-call을 한 번도 내지 않았다면 PPO는 찾을 수 없다” → DeepSeek-R1-Zero는 SFT 없이 DeepSeek-V3-Base에 policy-gradient RL algorithm인 GRPO를 직접 적용하여 AIME 2024 pass@1을 15.6%에서 77.9%로 올렸다. SFT와 RL이 각각 무엇을 더하는지는 후속 장에서 측정하며 SFT만으로 확정되지 않는다([[deepseek-r1]], v2 §2, §2.3).
2. “response-only SFT가 MT-Bench와 AlpacaEval에서 full-sequence SFT를 엄격히 이긴다”, “Response-only가 지배하며 full-sequence는 널리 진다” → Shi et al.은 instruction loss가 “many scenarios”에서 결과를 개선하며, 긴 instructionㆍ짧은 output 및 적은 예시에서 gain이 가장 크다고 보고한다. Less MMLU Chat에서 AlpacaEval 1.0은 4.42에서 9.78로 올랐다([[loss-masking-prompt]], arXiv:2405.14394v2 Abstract, Table 1).
3. “Shi 2024가 response-only, full-sequence, prompt-weighted 세 variant를 ablate했다” → prompt-loss-weight 연구는 Huerta-Enochian and Ko(arXiv:2401.13586)다. Shi et al.은 instruction modelling과 instruction tuning 및 NEFTune을 비교한다([[loss-masking-regimes]]; arXiv:2405.14394v2 §4.1).
4. `u_i`를 모두 mask하고 `a_1..a_{k-1}`을 mask해 `a_k`만 학습하는 것을 default로 둔 것 → open-instruct default SFT transform은 모든 assistant turn을 학습한다(`last_turn_only=False`). TRL의 `assistant_only_loss`도 모든 assistant response에서 loss를 계산한다. last-turn-only는 별도 option이다([[loss-masking-regimes]]: open-instruct@098424c `dataset_transformation.py` L1214-1218, L1554-1575; TRL v0.23.0 `sft_config.py` L85-94).
5. “conversation을 \(k\)번 unroll해 같은 loss value로 \(k\)배 더 많은 data” → turn별 expansion은 새 target token을 더하지 않는다. 같은 history rendering이면 모든 assistant turn의 single-pass loss와 같은 (context, target) pair를 만들며 prefix만 재처리한다. Template이 earlier turn을 current turn과 다르게 render할 때만 달라진다(§1.2 유도; [[chat-template-matrix]], Qwen3-8B model card, Best Practices).
6. “ChatML (Mistral default)”, “ChatML (OpenAI / early Mistral)” → Zephyr-7B-β SFT config는 `<|system|>`, `<|user|>`, `<|assistant|>` 뒤 content와 `eos_token`을 render한다. Mistral-7B-Instruct-v0.2에서 fine-tune된 FILM-7B는 Mistral `[INST]` template을 썼다([[hf-alignment-handbook]], alignment-handbook@1de1fc9 `recipes/zephyr-7b-beta/sft/config_full.yaml`; [[in2-film]] §2.2, App. D Example 9).
7. Zephyr의 “Packing true”, “Train on response only true”, “NEFTune α=5”, “AdamW (β₁=0.9, β₂=0.95)”, “FSDP FULL_SHARD” 행 → config는 이 key들을 어느 것도 설정하지 않는다. `learning_rate: 2.0e-05`, cosine schedule, `warmup_ratio: 0.1`, 1 epoch, `max_seq_length: 2048`, device당 batch 16, gradient accumulation 1을 설정한다. Hub model `alignment-handbook/zephyr-7b-sft-full`의 Trainer-generated card는 Adam beta (0.9, 0.999)와 8 device total batch 128을 보고하므로 옛 “Global batch 128”은 model card와 맞지만 β₂=0.95는 맞지 않는다([[hf-alignment-handbook]], 같은 file; huggingface.co/alignment-handbook/zephyr-7b-sft-full README).
8. ``SFTConfig(packing=True, train_on_response_only=True)`` → TRL `SFTConfig`에는 prompt-completion dataset의 `completion_only_loss`, conversational dataset의 `assistant_only_loss`(default `False`)가 있으며 `train_on_response_only` field는 없다([[loss-masking-regimes]], TRL v0.23.0 `sft_config.py` L85-94).
9. Tülu 3 “Template | Llama-3 (native)” → Tülu 3는 자체 template을 썼다. Table 13에서 Llama 3 template은 51.6으로 가장 낮고 Tülu 3 template은 52.8이다([[tulu-3]], arXiv:2411.15124v5 §4.3.1, Table 13, App. B.3).
10. “Tülu recipe가 939K SFT prompt에서 NEFTune neutral”, “\(|D|\le100K\)면 on, \(|D|\ge500K\)면 off” → Tülu 3 report는 NEFTune을 언급하지 않고 source도 이런 size rule을 주지 않는다([[tulu-3]] §4.3; [[neftune]] Verification).
11. Tülu 3 “Packing yes”, “AdamW (0.9, 0.95)”, “FSDP FULL_SHARD / HYBRID_SHARD” → Table 11이나 §4.3 어디에도 없다. Report는 LR, linear schedule, effective batch 128, max length 4,096, warmup ratio 0.03, 2 epoch, sum loss를 주지만 옛 표는 이를 빠뜨렸다. Reproduction command는 DeepSpeed ZeRO-3를 쓴다([[tulu-3]] Table 11, §4.3.2; [[open-instruct-allenai-recipes]] Technical Details).
12. “70B parameter count가 learning rate를 반으로 한다” → \(2\times10^{-6}\) 대 \(5\times10^{-6}\)는 2.5배이며, 각 값은 hyperparameter search로 골랐다([[tulu-3]] §4.3).
13. “Raw instruction batch는 50–89% padding” → Krell et al.은 BERT pre-training data(Wikipedia, 약 50% padding)와 GLUE(length 128의 CoLA 89%)를 측정했으며 instruction data는 측정하지 않았다([[sequence-packing]], arXiv:2107.02027 Abstract).
14. “speedup ≈ \(L_{max}/\operatorname{avg}(L_i)\), 기대 6×, 실제 2.5–3×”, “loss curve는 0.01 nat 내에서 일치해야 한다” → 어떤 source에도 없다. Kundu et al.은 올바른 position ID에도 offline packing validation loss가 padding보다 높았고(1.284 대 1.129), 한 epoch의 optimizer step이 훨씬 적기 때문이라고 본다([[sequence-packing-contract]], arXiv:2407.09105 §4.1, Table 2).
15. “packed-vs-unpacked ablation에 네 failure mode”, “position ID를 reset하지 않으면 RoPE가 position-shift됨” → 해당 card에 primary source가 없다. Correct block-diagonal mask에서는 rotary attention이 relative offset에만 의존하므로 한 document 안의 constant offset은 변화를 만들지 않는다. Position ID는 kernel이 그것으로 boundary를 정하거나(Kundu §3.3), learned absolute embedding일 때 중요하다(BERT: MLM accuracy 71.8에서 stall, 72.1과 비교)([[sequence-packing-contract]]; arXiv:2107.02027 §4.2.1).
16. “downstream win-rate 5-20점 붕괴”, handbook이 “#1 silent bug”라고 함 → source에는 5-20점이 없고 handbook의 `config_full.yaml`, `scripts/sft.py`(1de1fc9)에서 이 문장을 찾지 못했다. Tülu 3의 측정 template effect는 다섯 template 간 1.4점 spread이며, training-serving mismatch가 아니라 template choice를 측정한다([[tulu-3]] Table 13).
17. “base model이 post-training에서 이미 conditioned한 것” → base checkpoint에는 post-training이 없다. Llama 3는 header와 termination token을 post-training용으로 설계한 chat protocol로 설명한다([[llama-3]], arXiv:2407.21783v3 §4.1.1).
18. NEFTune가 Evol-Instruct, ShareGPT, OpenPlatypus에서 “+8–10점” → GPT-4 judge AlpacaEval point는 +9.26, +7.54, +8.61이다([[neftune]] Table 1).
19. “SFT는 모든 다른 behavior를 base model prior에 남긴다” → softmax cross-entropy gradient는 probability에 비례하여 모든 non-target logit을 낮추며, ORPO experiment에서는 chosen response SFT가 rejected response log-probability를 올렸다([[orpo]] §3, Fig. 3).
20. “ch-32 reasoning SFT와 long-CoT cold-start, ch-33 tool-call SFT, ch-34 agentic SFT, ch-31..35에서 mix 구성” → ch-32는 mid-training, ch-33과 ch-34는 case study, tool 및 agentic data는 ch-26과 ch-27, SFT mixture 구성은 ch-30b다(outline.json).

## 범용 모델에 이 장이 중요한 이유

Pipeline 순서는 pre-training → mid-training → SFT → preference optimization → RL → evaluation이다. SFT는 chat format으로 render한 curated demonstration으로 학습한다. 이는 이후 preference optimization과 RL이 sample할 output format을 정하며, 작은 set 학습은 그 set 밖 task에서 base model이 이미 가진 점수를 낮출 수 있다(아래 첫 문제).

세 측정 가능한 문제가 이 장의 동기다. 첫째 instruction-tuned model은 SFT data를 과적합할 수 있다. Shi et al.은 Dolly 9,229개로 학습한 LLaMA-2-7B가 18개 NLP task 평균에서 49.32에서 45.54로 떨어졌다고 보고한다([[loss-masking-prompt]] Table 3). 둘째 trainer 세부사항은 결과를 바꾼다. Tülu 3는 open-instruct와 다른 setting에서 학습한 SFT model의 gap을 발견하고 loss aggregation까지 추적했다([[tulu-3]] §4.3.2). 셋째, 선택 하나가 development score는 올리지만 그에 맞는 held-out score는 올리지 않을 수 있다([[tulu-3]] Table 32; 아래 §9).

[[ch-29e]]는 unseen-task 일반화를 주는 data property를 다루었고 [[ch-04]]는 packing, mask, template의 mechanism을 다루었다. 이 장은 그 mechanism의 어떤 setting이 일반화하는지와 어떻게 측정하는지를 묻는다.

## §1 SFT regime과 loss-mask 규칙

### 1.1 어떤 token이 loss를 받는가

**정의.** Loss mask는 `-100`인 position을 cross-entropy가 무시하는 token별 label vector다. SFT regime은 한 example의 형태다. single-turn, multi-turn, tool call, reasoning(thinking span 포함), agentic(tool call과 tool output이 있는 multi-step trajectory)이다.

**문제.** Inference에서 model이 전혀 내지 않는 token(user text, tool output)을 target으로 학습하거나, 반드시 내야 하는 token(earlier assistant turn, tool-call argument)을 제외할 수 있다. 두 변화 모두 error 없이 model이 배우는 것을 바꾼다.

| Segment | inference에서 생성자 | open-instruct와 TRL assistant-only loss의 default label | 근거가 있는 variant |
|---|---|---|---|
| System prompt, tool schema | client | masked | 이 장 source에서 loss로 검사하지 않음 |
| User turn | user | masked | instruction modelling에서 single-turn instruction token 포함(template token 제외), [[loss-masking-prompt]] Eq. 4 |
| 모든 assistant turn | model | trained | last turn만: open-instruct `last_turn_tulu_tokenize_and_truncate_v1` |
| Assistant message 안의 tool-call text | model | trained(assistant content임) | — |
| Tool output(`tool` role) | environment | masked(`role == "assistant"` span만 trainable) | — |
| Assistant turn의 thinking span | model | render되면 trained | earlier-turn thinking은 아예 render되지 않을 수 있음(§7) |
| Agentic trajectory의 erroneous step | model | 제거하지 않으면 trained | GLM-5는 context에 유지하고 loss에서 mask([[glm-5]] §3.1) |

Trainable span을 선택하는 open-instruct code([[loss-masking-regimes]], open-instruct@098424c `open_instruct/dataset_transformation.py` L1214-1218):

```python
def _trainable_assistant_indices(messages: list[dict[str, Any]], last_turn_only: bool) -> list[int]:
    assistant_indices = [idx for idx, m in enumerate(messages) if m["role"] == "assistant"]
    if last_turn_only:
        return assistant_indices[-1:]
    return assistant_indices
```

Default SFT transform은 `sft_tulu_tokenize_and_truncate_v1`(`finetune.py` L132-133)이며 이를 `last_turn_only=False`로 호출한다. TRL v0.23.0 default는 `assistant_only_loss`가 “defaults to `False`”이고 이 경우 “loss is computed on the entire sequence”이며, `completion_only_loss=None`은 prompt-completion dataset의 completion에만 loss를 계산한다고 문서화한다(`trl/trainer/sft_config.py` L85-94). 따라서 단일 `messages` list로 저장한 conversational dataset을 TRL default로 학습하면 user turn도 학습한다.

### 1.2 모든 turn의 single pass 대 turn별 expansion

**메커니즘.**

1. Single pass: conversation 전체를 한 번 render하고 모든 assistant span을 학습한다. Assistant turn \(i\)는 그 앞 render된 history에서 예측된다.
2. Last turn only: 같은 rendering이지만 label은 final assistant turn에만 둔다.
3. Per-turn expansion: assistant turn \(i\)마다 inference client가 turn \(i\)에서 render할 history가 들어 있는 example 하나를 만들고, label은 turn \(i\)에만 둔다.

Template이 매 turn 같은 방식으로 history를 render하면 option 1과 3은 같은 (context, target) pair set을 만들므로 합한 gradient도 같다. Qwen3 template처럼 earlier thinking content를 history에서 제거할 때는 달라진다([[chat-template-matrix]], Qwen3-8B model card, Best Practices).

**Worked example.** Conversation은 \(u_1=50\) token, \(a_1=200\) thinking + 50 answer token, \(u_2=40\) token, \(a_2=300\) thinking + 60 answer token이다(template token 무시).

- History에 thinking을 둔 single pass: \(50+250+40+360=700\) token 처리, \(250+360=610\) token 학습. Turn 2는 340-token history에 condition된다.
- Last turn only: 700 token 처리, 360 token 학습.
- Earlier thinking을 제거하는 inference client는 turn 2에 \(50+50+40=140\) token history를 보이며 single-pass example에는 이 history가 없다. Client와 일치하는 per-turn expansion: copy 1은 300 token(250 trained), copy 2는 \(140+360=500\) token(360 trained). 합계 800 token 처리, 610 학습으로 single pass보다 compute 14% 증가.
- Sequence 내부 평균을 내고 두 sequence에 다시 평균 loss를 내면 copy 1의 token weight는 \(1/(2\cdot250)=0.0020\), copy 2는 \(1/(2\cdot360)\approx0.0014\)다. Single pass는 모든 trained token에 \(1/610\approx0.0016\)을 준다. Sum loss라면 두 layout 모두 trained token weight가 1이다(§3).
- Open-instruct `olmo_thinker_remove_intermediate_thinking`처럼 rendering 중 earlier thinking을 제거하는 template은 non-final assistant message에 `content.split('</think>')[-1]`을 쓴다. Single pass에서 \(a_1\)의 thinking 200 token은 input과 label 모두에서 사라져 500 token 처리, 410 token 학습이 된다([[chat-template-matrix]], open-instruct@098424c `dataset_transformation.py` L427-471).

Companion figure [figures/sft-axes.html](figures/sft-axes.html)는 이 conversation을 token별로 render한다. Reader는 loss layout, training template history rule, serving client history rule을 각각 설정하고 label을 받는 token, token별 loss weight, training-serving history가 다른 위치를 확인할 수 있다.

**범용 모델에 대한 함의.** 모든 assistant turn 학습은 모든 demonstration token을 쓴다. Deployed client가 training template과 같은 방식으로 history를 render하면 single pass로 충분하다. Client가 earlier thinking을 제거하면, 위 세 layout 중 per-turn expansion만이 inference에서 받는 history 아래에서 thinking을 포함한 모든 turn을 학습한다(위 rendering rule에서 도출한 해석).

## §2 Regularizer로서 instruction에 걸리는 loss

**정의.** Shi et al.의 instruction tuning(IT)은 completion token에만 loss를 준다. Instruction modelling(IM)은 instruction token에도 loss를 추가하고 template token은 제외한다([[loss-masking-prompt]] §3).

```
IT:  L = − Σ_{j=1..n} log P(C_j | I_1..I_m, C_1..C_{j−1})                         (Eq. 2)
IM:  L = − Σ_{t=1..m+n} log P(x_t | x_1..x_{t−1}) · 1(x_t ∉ T)                    (Eq. 4)
```

여기서 \(I\)는 \(m\) token instruction, \(C\)는 \(n\) token completion, \(x\)는 이어 붙인 sequence, \(T\)는 `<|user|>` 같은 template token 집합이다. \(\mathbf1(\cdot)\)은 token이 template token이 아닐 때 1이다.

**문제.** Shi et al.은 작은 dataset의 output, 특히 짧은 output에만 학습하면 model이 그 output을 암기할 수 있다고 가정한다(§1). Training prompt에 대한 greedy output과 training output 사이 BLEU(n-gram overlap score, 0-100), training loss, held-out loss로 암기를 측정한다.

**근거**(LLaMA-2-7B base; LR \(2\times10^{-5}\), batch 128, 보통 2 epoch; App. C Table 6). NLP mean은 benchmark task 18개의 평균이다. AlpacaEval 1.0은 GPT-4가 심사한 Text-Davinci-003 대비 win rate이며, MT-Bench는 80개 two-turn question에서 GPT-4가 심사한 1-10 score다. 결과(단일 연구):

| Dataset(예시 수; instruction/output length ratio) | 18개 task NLP mean, IT → IM | AlpacaEval 1.0, IT → IM | MT-Bench, IT → IM |
|---|---|---|---|
| Less MMLU Chat (13,533; 26.3) | 47.18 → 47.84 | 4.42 → 9.78 | 3.86 → 4.54 |
| Less Tydiqa (13,533; 5.9) | 48.21 → 48.70 | 5.12 → 10.10 | 4.08 → 4.36 |
| Alpagasus Dolly 9k (9,229; 0.30) | 45.54 → 48.00 | 21.54 → 30.77 | 4.33 → 4.55 |
| Alpagasus Alpaca 5k (5,305; 0.57) | 45.29 → 47.47 | 16.29 → 19.52 | 3.62 → 3.48 |
| LIMA (1,030; 0.09) | 48.79 → 49.60 | 33.06 → 32.94 | 4.77 → 4.83 |

출처: arXiv:2405.14394v2 Table 1, Table 5. IM은 Alpaca 5k와 Dolly 3k에서 MT-Bench를, LIMA에서 AlpacaEval 1.0을 낮췄으므로 효과가 균일하지 않다. Mechanism 근거로 LIMA에서 output token mean training loss는 IT 1.37, IM 1.45였지만 Tülu V2의 10% sample test loss는 IT 1.32, IM 1.17이었다(Fig. 3). Training output과 greedy output 사이 BLEU는 7개 dataset 모두에서 IM이 낮췄다. 예를 들어 Less BBH ICL은 60.96 → 53.94였다(Table 2). Base model KL penalty는 대체물이 아니다. LIMA에서 AlpacaEval 2.0을 2.58에서 0.06으로 낮췄다(Table 3). 저자들은 IM을 current practice의 대체물로 제안하지 않는다고 말한다(Abstract). Tülu V2 subset의 AlpacaEval 1.0 gain은 instruction/output ratio를 약 10으로 고정했을 때 example 수가 35,000에서 1,000 쪽으로 줄수록 커졌다(Fig. 2 right, App. C).

**Prompt loss weight.** Huerta-Enochian and Ko는 prompt token의 weight를 masked prompt인 0부터 full loss인 1까지의 PLW로 정의한다. LLaMA-1 7B와 LLaMA-2 7B, PLW value 10개, Alpaca variant 3개(60 run, benchmark 13개)에서 short-completion data 성능은 PLW와 통계적으로 유의한 negative quadratic relationship을 보였다. Multiple-choice 및 short-generation benchmark에서는 0.01-0.5가 가장 좋았고 long-generation benchmark에서는 1.0 근처가 가장 좋았다. Long-completion data에서는 PLW를 “safely ignored”할 수 있다([[loss-masking-regimes]], arXiv:2401.13586v4 Abstract, §1 contributions, §4 setup). 결과(단일 연구).

**Worked example.** 200 token prompt의 mean token loss가 2.0(sum 400), 20 token completion의 mean loss가 1.5(sum 30)라고 하자. Weighted sum은 PLW·400 + 30이다. PLW = 0에서는 loss 30이다. PLW = 0.1에서는 70이며 prompt는 그중 \(40/70=57\%\)를 차지한다. PLW = 1에서는 430이며 prompt는 93%를 차지한다. Short completion에서는 작은 weight도 prompt term을 summed loss의 더 큰 부분으로 만든다.

**조건과 한계.** 두 연구 모두 6.7B–13B parameter model(LLaMA family, Shi et al. Fig. 5의 OPT-6.7B)을 쓰고 LLM-judged win rate를 사용한다. Shi et al. Table 1 dataset과 Huerta-Enochian and Ko dataset은 single-turn이다. Shi et al. Fig. 2에는 multi-turn conversation을 포함하는 ShareGPT(50,000 예시)와 Tülu V2(326,181)도 있으나, 논문은 IM이 earlier turn을 어떻게 처리하는지 설명하지 않는다. Hybrid reasoning SFT mixture인 SmolLM3(3B)는 user turn masking이 평가 대부분에서 “a few points of improvement”를 냈고 IFEval에서 가장 큰 효과였다고 보고하지만 benchmark별 수치는 주지 않는다([[smol-training-playbook]], “Masking User Turns”). Library의 어떤 연구도 백만 예시 multi-turn mix에서 IM을 시험하지 않는다. **함의(해석):** 증거는 특히 긴 instructionㆍ짧은 output의 작은 single-turn set에서 instruction loss를 regularizer로 쓰는 것을 지지하지만, 대규모 multi-turn chat mix의 default로 지지하지는 않는다.

## §3 Epoch, learning rate, loss aggregation

**정의.** Loss aggregation은 token별 loss를 differentiation할 scalar로 결합하는 방식이다. 한 forward pass의 non-padding token 평균, accumulation step에 걸쳐 평균한 micro-batch별 평균, 또는 sum이다.

**문제.** Tülu 3는 open-instruct에서 학습한 SFT model과 TPU 같은 다른 setting에서 학습한 model 사이 performance gap을 발견하고, “gradient accumulation 또는 distributed training setup을 고려하지 않고” 평균을 낸 것에 원인을 돌렸다([[tulu-3]] §4.3.2).

**수식**([[tulu-3]] §4.3.2, Eqs. 1-2). Summed token loss \(l_1,l_2\), non-padding token \(n_1,n_2\)를 가진 두 sample에 대해:

```
one forward pass:        L = (l₁ + l₂) / (n₁ + n₂)          (모든 token이 같은 weight)
gradient accumulation:   L = (l₁/n₁ + l₂/n₂) / 2            (모든 sample이 같은 weight)
sum loss:                L = l₁ + l₂                          (모든 token weight 1; LR 재조정 필요)
```

**Worked example.** Sample 1은 mean loss 2.0인 \(n_1=10\) token(\(l_1=20\)), sample 2는 mean loss 1.0인 \(n_2=90\) token(\(l_2=90\))이라고 하자. One forward pass는 \(L=110/100=1.10\)이고 각 token weight는 0.01이다. 두 micro-batch에 걸친 accumulation은 \(L=(2.0+1.0)/2=1.50\)이다. Sample 1 token weight는 \(1/20=0.05\), sample 2 token weight는 \(1/180\approx0.0056\)으로 9:1 ratio다. Sum은 token당 weight 1로 \(L=110\)이므로 gradient scale은 batch당 token 수에 따라 증가한다. GPU 수 또는 accumulation step을 바꾸면 앞의 두 loss는 바뀌지만 세 번째는 바뀌지 않는다. [figures/sft-axes.html](figures/sft-axes.html)의 aggregation panel은 임의 \(n_1,n_2\) 및 micro-batch split에 이 weight를 다시 계산한다.

**근거.** Tülu 2 mixture로 Llama 3.0을 fine-tuning했을 때 LR \(5\times10^{-6}\)인 sum loss가 최고 average를 냈고 2 epoch가 더 긴 학습을 이겼다(Figs. 5-6). Tülu 3는 sum loss, \(5\times10^{-6}\)(8B) 또는 \(2\times10^{-6}\)(70B), 2 epoch를 사용했다([[tulu-3]] §4.3). 릴리스된 open-instruct command는 commit 8781471에서 `--reduce_loss sum`을 전달했지만, 098424c에서는 flag가 없어지고 default가 mean이다([[open-instruct-allenai-recipes]] Technical Details). 상태: 결과(단일 연구), 문서화된 reproduction conflict 있음.

**Epoch와 learning rate, 다른 근거.** Alpaca로 학습한 LLaMA-2 7B는(ChatGPT judge의 AlpacaEval win rate) NEFTune 없이 1, 3, 5, 7, 9 epoch에서 40.50, 48.26, 48.94, 48.63, 47.45였고, NEFTune을 적용하면 55.09, 62.55, 62.24, 60.50, 58.14였다([[neftune]] Table 14). Shi et al.은 다섯 dataset에서 epoch 2-10에 대한 18-task NLP mean을 그리며 IM이 IT보다 “generally has a lower instruction tuning tax”라고 보고한다(Fig. 4). Instruction tuning tax는 base model 대비 이 NLP task에서의 하락이다. SmolLM3의 LR scan은 \(3\times10^{-6}\) 또는 \(1\times10^{-5}\)가 더 큰 value보다 평균적으로 낫고, \(1\times10^{-5}\) 초과 LR에서는 AIME25가 하락한다고 발견했다. 5 epoch는 평균적으로 1 epoch보다 “a few more percentage points” 더했다([[smol-training-playbook]], “Tuning the Learning Rate”, “Scaling the Number of Epochs”). Best epoch 수는 setting별로 달랐다. Alpaca에서는 NEFTune 없이 5, NEFTune을 두면 3(1, 3, 5, 7, 9 중), Tülu 2 mix의 Tülu 3에서는 2(2–7 중), SmolLM3 baseline mixture에서는 5가 1보다 좋았다(다른 count는 playbook figure에만 나타남). 각 결과는 단일 연구이며 일반 규칙은 열린 질문이다.

**Seed variance와 soup.** Model soup은 같은 base에서 fine-tune한 여러 model weight의 element-wise average다. Tülu 3 SFT seed별 average는 8B에서 59.9, 60.1, 59.8, 59.8, 59.8(spread 0.3)였고 70B에서 71.8, 70.0, 72.6(spread 2.6)이었다. 두 seed best soup은 60.2와 72.5였고 best single run이 release됐다([[tulu-3]] Table 14). 해당 크기의 seed spread보다 작은 두 SFT configuration 차이는 증거가 아니다. Soup은 [[ch-30c]]가 다룬다.

**함의.** Learning rate는 aggregation, batch size, epoch count와 함께 정의된다. 공개된 learning rate를 재사용할 때 다른 세 setting도 함께 재사용하라. Tülu 3는 sum loss와 2 epoch를 함께 둔 상태에서 LR을 골랐기 때문이다(Figs. 5-6). 그렇지 않으면 learning rate를 다시 sweep하라.

## §4 Packing과 optimization에 미치는 효과

**정의.** Packing은 여러 example을 고정 길이 row 하나에 이어 붙인다. Variable-length attention(`cu_seqlens`)에서는 token이 자기 example 안에서만 attention하며, mechanics는 [[ch-04]]에 있다.

**문제.** Packing은 speed뿐 아니라 optimizer step 수와 step당 token 수를 바꾼다.

**Worked example.** 평균 500 token인 example 100,000개, batch 128을 생각하자. Unpacked는 epoch당 \(\lceil100{,}000/128\rceil=782\) step이고, step당 평균 non-padding token은 약 \(128\times500=64{,}000\)이다. 완전히 채운 4,096-token row로 packing하면 \(50{,}000{,}000/4{,}096=12{,}208\) row, epoch당 \(\lceil12{,}208/128\rceil=96\) step, step당 524,288 token이다. 같은 epoch 수는 이제 optimizer update 782회가 아니라 96회로, 8.1배 차이다.

**근거.**

- Kundu et al.(Mistral-7B, FLAN 20K subset, 1 epoch): padding은 742 token/s, validation loss 1.129였다. Position ID를 써서 585 row로 offline packing하면 3,010 token/s, 1.284였고, step count를 유지하는 mini-batch packing은 1,408 token/s, 1.127이었다([[sequence-packing-contract]], arXiv:2407.09105 §4.1, Table 2).
- Wang et al.(LLaMA-3 8B와 70B): WildChat(GPT-4) 69K에서 padding / random packing / greedy packing average는 8B에서 49.58 / 49.46 / 50.6, 70B에서 61.50 / 65.97 / 65.92였다. 1M open-source set에서는 8B 54.3 / 54.95 / 55.05, 70B 66.12 / 67.26 / 67.54였다. WildChat 70B training은 padded 9,533 s, random packing 3,749 s였다(arXiv:2410.08081 Tables 3, 5).
- SmolLM3: packing은 throughput을 3-5× 올렸다. 같은 effective batch 128에서 packing은 IFEval을 낮췄고, 해당 model과 dataset에서는 effective batch 32 초과에서 average가 하락했다([[smol-training-playbook]], “To Pack or Not to Pack?”).

독립된 두 group은 작은 SFT set을 같은 epoch count로 packing할 때 더 나쁜 결과를 보고한다. Kundu et al.의 더 높은 validation loss(20K example), SmolLM3의 더 낮은 IFEval이다(복제됨). 70B와 1M-example set에서 packing gain은 한 연구(Wang et al.; 결과, 단일 연구)에서 온다.

**Position ID.** Rotary position embedding(RoPE)은 attention score가 query와 key position 사이 offset에 의존하게 하므로, 올바르게 mask된 한 document 안의 constant offset은 attention을 바꾸지 않는다. `position_ids` reset은 attention kernel이 Kundu et al.(§3.3)처럼 그것에서 `cu_seq_len`(example boundary)을 계산할 때와, learned absolute position(Krell et al., BERT, §4.2.1)에서 여전히 중요하다. Reset하지 않은 position이 trained range를 넘을 때도 중요하다(해석). Kundu et al. Table 4(Mistral-7B, FLAN 20K)에서 position ID 없이 packing하면 example이 서로 attention하게 되며, 같은 optimizer step 35에서 position ID 사용 1.221 대비 validation loss 1.294, 281 step에서 1.170 대비 1.252였다. Wang et al.은 packed conversation 사이 attention을 reset했는지 밝히지 않는다.

**함의.** SFT set이 크거나(약 1M example) model이 크면(70B) packing을 사용하라. Wang et al.에서 이 run은 2.5–6.7배 더 빨랐고 random 또는 greedy packing은 average score를 −0.19에서 +4.47점 바꿨기 때문이다(Tables 3, 5, 70B의 네 dataset 전부 및 8B의 1M set). Model이 3B–8B이고 set이 작다면(Kundu et al. 20K, Wang et al. 69K; SmolLM3 baseline subset 크기는 text에 없음), matched optimizer step에서 packing과 padding을 비교하거나, Kundu et al.에서 padded run과 같은 step count 및 validation loss를 유지한 mini-batch packing을 사용하라.

## §5 NEFTune

**정의와 수식**([[neftune]] Algorithm 1). \(X\in\mathbb R^{B\times L\times d}\)인 embedding에 대해 \(X'=X+(\alpha/\sqrt{L\cdot d})\cdot\epsilon\), \(\epsilon\sim\operatorname{Uniform}(-1,1)^{B\times L\times d}\)다. \(B\)는 batch size, \(L\)은 sequence length, \(d\)는 embedding width, \(\alpha\)는 noise scale이다. Noise는 training 중에만 추가한다.

**근거.** LLaMA-2 7B, Text-Davinci-003 대비 GPT-4 judge에서 Alpaca는 29.79 → 64.69, Evol-Instruct는 70.34 → 79.60, ShareGPT는 68.74 → 76.28, OpenPlatypus는 62.00 → 70.61이었다(Table 1). ARC, HellaSwag, MMLU, TruthfulQA는 안정적으로 유지됐다(Fig. 3). NEFT는 더 높은 training loss, 약간 더 낮은 held-out loss, training response 대비 더 낮은 ROUGE-L/BLEU를 보였다(Figs. 4-5). Alpaca output length도 375에서 1,062 character로 길어졌다(Table 4). 이는 headline metric이 LLM judge이고, 저자들이 score가 length와 상관된다고 보고했기 때문에 중요하다. LLaMA-1 7B(Alpaca)에 더 긴 answer를 내도록 prompting하면 GPT-4 win rate는 32.36에서 48.01로 올랐지만 NEFT model의 61.99보다는 낮았다(§5.2-5.3, Table 5).

**§2와의 관계.** Shi et al.에서 NEFTune과 IM은 각각 7개 dataset 중 6개에서 IT보다 AlpacaEval 1.0을 올렸다. IM에 NEFTune을 추가하면 7개 중 6개에서 IM 단독보다 AlpacaEval 1.0을 올렸지만, LIMA, Less MMLU Chat, Less BBH ICL에서는 NLP mean을 낮췄다(Tables 1, 4). 독립된 두 group은 서로 다른 intervention에서 같은 pattern, 즉 더 높은 training loss, 더 낮은 held-out loss, training output과의 더 낮은 overlap을 보고한다. 각 method pattern은 한 group만 측정했으므로, 이는 어느 method의 replication이라기보다 overfitting 설명과 일관된 증거다(해석).

**한계.** Single-turn set(Evol-Instruct 70k, ShareGPT 70K, Open-Platypus 25k; Alpaca size는 명시되지 않음)에서 시험했고, \(\alpha\)는 AlpacaEval 자체에서 선택했으며 safety는 평가하지 않았다([[neftune]] §3.2, App. A.1, §7). Library의 어떤 source도 백만 예시 multi-turn mix에서 NEFTune을 시험하지 않는다.

## §6 Training과 inference 사이 template 일관성

**정의.** Chat template은 message를 token ID로 render하는 함수다. Role header, turn terminator, thinking 및 tool content 규칙을 포함한다.

**문제.** SFT, 이후 stage, serving 사이 template 차이는 error를 내지 않는다. Tülu 3는 Llama 3.0의 intermediate SFT mix에서 template choice 효과를 측정했다. Newline을 EOS로 바꾼 Tülu 53.0, Zephyr 52.9, trailing newline 없는 Tülu 3 52.8, Tülu 2 52.6, Llama 3 51.6이었다. 최고 option은 “후속 post-training pipeline step과 generation inconsistency를 피하기 위해” 사용하지 않았다([[tulu-3]] §4.3.1, Table 13). 결과(단일 연구; template당 run 수는 보고되지 않음).

**Mixed thinking mode의 template 설계.** Qwen3는 thinking data와 non-thinking data를 한 SFT stage에서 합쳤다. Non-thinking sample은 빈 `<think></think>` block을 유지하며, 저자들은 이것이 “internal format consistency”를 보장한다고 설명한다. Multi-turn dialog에는 여러 `/think`, `/no_think` flag가 있고 각 response는 마지막 flag를 따른다([[qwen-3]], arXiv:2505.09388v1 §4.3, Table 9; card가 아직 검증되지 않아 paper에서 읽음).

**릴리스 pipeline에서 보고된 template 및 formatting-token 문제.**

- OLMo 3 7B Think DPO와 RL은 “a minor miscommunication” 때문에 SFT template과 약간 다른 chat template을 썼다. OLMo 3.2+의 Think SFT data는 첫 `<think>` token을 prompt로 mask한 bug를 피하려 instruct template으로 tokenize한다([[open-instruct-allenai-recipes]], `docs/olmo3.md` L22-28).
- SmolLM3의 첫 hybrid baseline은 system message의 persona를 무시했다. Processing code bug가 template `custom_instructions` argument를 `None`으로 설정해 모든 training sample에서 system message를 제거했기 때문이다. 이를 고쳐도 evaluation score는 바뀌지 않았다([[smol-training-playbook]], “Vibe-Test Your Baselines”).
- Llama 3는 DPO에서 chosen 및 rejected response 모두의 header와 termination token을 mask한다. 그것을 포함하면 “tail repetition 또는 abrupt termination token 생성 같은 바람직하지 않은 model behavior”가 생길 수 있기 때문이다. 저자들은 shared token이 opposing gradient를 받는다는 가설을 세운다([[llama-3]] §4.1.4).

**점검.** Training template과 serving stack으로 conversation 하나를 render하여 token ID를 비교하고 role별 label mask를 출력한다. TRL은 `return_assistant_tokens_mask=True`와 template의 `{% generation %}` block에서 assistant mask를 얻는다([[smol-training-playbook]], “Masking User Turns”). **함의.** 다섯 template 간 선택은 Tülu 3 average를 1.4점 움직였다. 이 장 source는 SFT-serving template mismatch의 score effect를 측정하지 않는다. 보고된 사례는 pipeline bug(OLMo 3, SmolLM3)와 DPO formatting-token behavior(Llama 3)이므로, 위 rendered-token check가 사용 가능한 control이다.

## §7 Agentic trajectory: reasoning history, erroneous step, truncation, token ID

**Reasoning history 규칙은 model마다 다르다.**

- Qwen3: “historical model output should only include the final output part and does not need to include the thinking content”(Qwen3-8B model card, Best Practices; [[chat-template-matrix]]).
- DeepSeek-V3.2: “Historical reasoning content is discarded only when a new user message is introduced.” Tool message만 append하면 reasoning은 유지되고, tool call과 result는 항상 유지된다(arXiv:2512.02556v1 §3.2.1; [[chat-template-matrix]]).
- MiniMax-M2(official blog): 이전 round thinking을 유지하는 것 대 버리는 것은 SWE-Bench Verified 69.4 대 67.2, Tau² 87 대 64, BrowseComp 44.0 대 31.4, GAIA 75.7 대 67.9, xBench 72.0 대 66.0이었다. Evaluation setting은 제공되지 않는다([[minimax-m2-interleaved-thinking]]).
- GLM-5는 coding-agent scenario를 위해 turn 간 모든 thinking block을 유지하는 “Preserved Thinking”을 학습한다([[glm-5]] §3.1). SmolLM3는 model을 “appropriately condition”하는 데 필요하다고 판단해 training 중 모든 turn의 reasoning token을 유지했다([[smol-training-playbook]], “Picking a Good Chat Template”).

SFT consequence는 §1.2에서 따른다. Deployed client가 earlier thinking을 유지하거나 제거할 때 SFT history도 같은 규칙으로 render해야 한다. 그렇지 않으면 학습된 각 turn은 inference에서 model이 받지 않는 history에 condition된다(해석). Thinking을 유지해 학습하고 이를 버리는 API로 serving하는 model은 training보다 더 짧은 history를 받는다. MiniMax는 prior-round thinking을 보존하지 않은 M2 community deployment를 보고하고 API와 client가 reasoning content를 되돌려 전달하지 않은 탓으로 돌린다. 위 benchmark 비교는 그 thinking을 버리는 비용을 측정한다(official source 하나, evaluation setting 미보고).

**Erroneous step.** GLM-5는 agent trajectory의 erroneous segment를 context에는 유지하지만 “loss function에서 masked out”하여 “incorrect action을 강화하지 않으면서 error correction behavior를 학습”하게 한다([[glm-5]] §3.1). 실패 tool call이 training target이었다면 cross-entropy는 그 확률을 올린다. GLM-5는 이 trajectory를 버리는 방식과 masking을 비교한 ablation을 보고하지 않는다.

**Truncation.** Open-instruct는 “right-sided truncation drops the trailing EOS, so a cut inside an assistant turn leaves trainable text with no terminator”라고 문서화한다. `keep`(default)는 그대로 학습하고, `terminate`는 마지막 token을 trainable EOS로 만들며, `drop`은 row를 mask한다([[loss-masking-regimes]], `dataset_transformation.py` L1427-1488). Trajectory가 `max_seq_length`에 비해 길면 truncation은 더 큰 row 비율에 영향을 준다(해석; 이 장 source는 agentic data rate를 보고하지 않음).

**Token-in-token-out.** GLM-5는 inference engine이 생성한 token ID에서 RL trajectory를 구축한다. Text를 다시 tokenize하면 “token boundary, whitespace/normalization 처리, truncation, special-token 위치에서 미묘한 mismatch를 도입할 수” 있기 때문이다([[glm-5]] §4.1.2). SFT data가 RL rollout 또는 rejection sampling에서 왔다면 token ID 저장은 같은 mismatch를 피한다. Open-instruct는 assistant label span을 도출할 수 없는 conversation을 mask한 뒤 drop하며, 이는 `tulu-3-sft-olmo-2-mixture`의 약 0.005%다(`dataset_transformation.py` L1499-1504). Token-in-token-out의 SFT 적용은 이 course의 해석이며 GLM-5는 RL에 대해 설명한다.

## §8 LoRA 대 full fine-tuning: forgetting에 맞선 학습

**정의.** LoRA(low-rank adaptation)는 pretrained weight matrix \(W\in\mathbb R^{d\times k}\)를 freeze하고 low-rank update만 학습한다. 따라서 fine-tuned matrix는 trainable \(A\in\mathbb R^{d\times r}\), \(B\in\mathbb R^{r\times k}\)에 대해 \(W+\gamma_r\cdot AB\)다([[lora-learns-less-forgets-less]] §2). 여기서 \(r\)은 rank(\(r\)은 \(d,k\)보다 훨씬 작음), \(\gamma_r=\alpha/r\)은 scale factor, \(\alpha\)는 hyperparameter다. Biderman et al.은 \(\alpha=2r\)로 설정하고 모든 attention 및 MLP matrix에 LoRA를 적용한다.

**문제.** 좁은 domain의 full fine-tuning은 그 domain 밖 task 점수(forgetting)를 낮출 수 있다. LoRA는 update를 rank \(r\)로 제한한다. Biderman et al.은 이것이 forgetting을 얼마나 줄이고 target-domain learning에 얼마의 비용을 내는지 측정한다.

**근거.** Magicoder-Evol-Instruct-110K(72.97M token)의 Llama-2-7B, epoch 4에서 full fine-tuning은 HumanEval pass@1 0.470, forgetting average 0.512였고, LoRA \(r=256\)은 0.498, 0.631, LoRA \(r=16\)은 0.358, 0.652였다. Forgetting average는 HellaSwag, ARC-Challenge, WinoGrande accuracy 평균이므로 높을수록 forgetting이 적다(Tables S5-S6, §3.3). Full fine-tuning은 50 sample 중 unique HumanEval solution도 LoRA보다 적게 만들었으며 base model이 가장 높았다(§4.5, Fig. 5). 저자들은 exact string matching이 coarse diversity measure라고 지적한다. General Tülu-v2-mix에서는 모든 LoRA rank(16, 64, 256)가 MT-Bench에서 full fine-tuning의 one standard error 안에 있었고, 2 epoch에서는 full fine-tuning이 LoRA보다 forgetting이 적었다. Main-paper ordering은 epoch 6에서 다시 나타났다(App. C.2). Thinking Machines(practitioner-evidence)는 Tulu3 및 OpenThoughts3 subset에서 adapter capacity가 소진될 때까지 LoRA가 log loss에서 full fine-tuning과 일치하며, optimal LR은 LoRA가 10× 높고 attention-only LoRA는 MLP-only LoRA보다 나쁘다고 발견했다([[lora-without-regret]]). Biderman et al.도 full fine-tuning보다 한 order 높은 optimal LoRA LR을 보고하므로(§4.7, App. B Fig. S1), 이 ratio는 복제됨이다.

**조건과 한계.** Biderman et al.은 base model 하나(Llama-2-7B), target domain 두 개(code, math), general mix 하나를 시험한다. Thinking Machines는 forgetting이 아니라 training log loss를 측정한다.

**함의(해석).** Code data에서는 rank가 learning과 forgetting을 trade-off했지만, general Tülu-v2-mix의 2 epoch에서는 forgetting ordering이 뒤집혔다. 따라서 forgetting은 training mix 밖 suite에서, 사용하는 data와 epoch count마다 측정해야 한다. [[ch-30a]]가 protocol과 다른 control을 다룬다.

## §9 각 설계 축의 평가 protocol

Tülu 3는 결정을 위해 쓰는 development suite와 development 중 보지 않는 unseen suite로 평가를 나눈다. 예를 들어 IFEval은 development, IFEval-OOD는 unseen이다([[tulu-3]] §2.2, Table 3). SFT ablation에서 Persona data를 제거하면 IFEval은 72.8에서 53.6으로 떨어졌지만 IFEval-OOD는 17.6에서 18.0으로 움직였다. Math data를 제거하면 MATH는 31.5에서 23.5, Deepmind Mathematics는 32.3에서 23.3으로 떨어졌다(Table 32). 저자들은 data choice가 “Precise Instruction Following에서 development evaluation에 과적합하고, 어느 정도 Knowledge Recall과 Reasoning에서도 그러하다”고 결론낸다(§7.4.1). Math의 경우 Table 32는 math data 제거 때 development와 unseen score가 함께 하락함을 보인다.

| 축 | 여기서 결정 | 채택 전 여기서 확인 | 넘어야 할 noise |
|---|---|---|---|
| Instruction loss / PLW | short-answer와 long-generation으로 나눈 development task | unseen suite; training target에 대한 output BLEU | seed spread(Tülu 3 Table 14) |
| Epoch, LR, aggregation | development average 및 skill별 score | base model 대비 forgetting average; unseen suite | 동일 |
| Packing | matched optimizer step의 development score | unseen instruction-following task | 동일 |
| Template와 history rendering | training과 serving의 rendered token ID 일치 | serving client 아래 agentic held-out task | — |
| LoRA rank | target-domain score | forgetting average; prompt당 unique generation | 동일 |

## Negative sample과 negative feedback

**어느 의미가 적용되는가.** Course는 negative 네 의미를 쓴다. (1) training 전 제거하는 negative marginal value sample, (2) input content 또는 corrected target에 failure를 남기고 ordinary cross-entropy로 학습하는 것, (3) `incorrect` 같은 control token 아래 failure를 학습하는 것, (4) sample likelihood를 명시적으로 낮추는 것(gradient로서 negative)이다. Ordinary SFT는 sample을 버릴 때 (1), failed step을 context에 zero loss로 둘 때 (2)를 쓴다. 이 절은 SFT cross-entropy가 암묵적으로 억제하는 것도 다룬다. (4)는 [[ch-31a]], [[ch-39]]가 다룬다.

**SFT에서 negative의 출처.** Rejection sampling은 prompt마다 여러 output을 만들고 최고 또는 정답만 남기며 나머지는 버린다. Llama 3는 prompt당 \(K\)(보통 10–30) output을 sample하고 reward model이 최고로 score한 하나를 남긴다([[llama-3-recipe]], v3 §4.2.2 row). Agentic trajectory에는 실패 tool call이 있으며 GLM-5는 이를 context에 유지하고 loss를 mask한다([[glm-5]] §3.1). 두 source 모두 label의 false-negative rate를 보고하지 않는다.

**메커니즘.** Logit \(z\), target token \(y\)에서 softmax cross-entropy는 다음을 준다.

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

\(z_j\)는 token \(j\) logit, \(p_j=\operatorname{softmax}(z)_j\)는 model probability, \(y\)는 target token이며 \(\mathbf1[j=y]\)는 target일 때 1, 아니면 0이다. Learning rate \(\eta\)로 \(\log p_y\)에 gradient-ascent step을 하면 \(z_y\)는 \(\eta(1-p_y)\)만큼 오르고 다른 모든 logit은 \(\eta\cdot p_j\)만큼 내려가므로, 그 position에서 가장 확률 높은 wrong token이 가장 크게 내려간다. **Worked example:** \(p=(0.6,0.3,0.1)\), target token 2, \(\eta=1\)이면 logit 변화는 \((-0.6,+0.7,-0.1)\)이다. \(z=\log p\)에서 시작하면 새 확률은 \((0.322,0.590,0.088)\)이다. 0.6 competitor는 0.278, 0.1 competitor는 0.012를 잃는다.

**할 수 없는 것.** 이 억제는 teacher forcing 아래 dataset target position에만 작용하며, 각 position은 model 자기 sample이 아니라 dataset의 이전 token에 condition된다. Model이 자기 이전 token에서 생성할 특정 wrong continuation을 낮추지 못하며, good response와 token을 공유하는 bad response 확률은 올릴 수도 있다. ORPO의 HH-RLHF OPT-350M experiment에서 chosen response SFT는 chosen과 함께 rejected response log-probability도 올렸고 rejected가 더 높은 경우도 있었다([[orpo]] §3, Fig. 3). Rejected response를 명시적으로 낮추는 \(\lambda\cdot L_{OR}\)를 더하면 \(\lambda=1.0\)에서 이를 뒤집었다(Fig. 7).

**Control과 진단.** Context에 남긴 failed step은 mask한다. Known-bad response set을 두고 SFT 전후 mean log-probability를 log한다. Positive-only full fine-tuning은 unique HumanEval solution을 줄였으므로([[lora-learns-less-forgets-less]] §4.5), prompt당 unique generation과 pass@k(\(k\) sample 중 하나 이상이 맞을 확률)를 보고한다. **일반성 효과:** 좁은 set의 positive-only SFT는 demonstrated output에 확률을 집중시킨다. 이 장에서 측정한 cost는 더 낮은 diversity와 더 높은 forgetting이며 negative에서 온 gain의 비중은 측정하지 않았다.

## 레시피

| 모델(정확한 릴리스) | 크기 | 단계 | 설정 | 값 | 출처 위치 | 상태 | 이 값의 근거 |
|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak LR; schedule; warmup ratio | \(5\times10^{-6}\); linear; 0.03 | arXiv:2411.15124v5 Table 11, §4.3 [[tulu-3]] | verified 2026-09-15 | §4.3.2 Fig. 5: sum loss 및 5e-6 최고(Llama 3.0, Tülu 2 mix); hyperparameter search 뒤 발견 |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | epoch; max length; effective batch(sequence) | 2; 4,096 token; 128 | Table 11 | verified 2026-09-15 | Fig. 6: 2 epoch 최고(Llama 3.0, Tülu 2 mix) |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | data; compute | prompt 939,344개; GPU 32개, 6시간 | Table 7; §4.3 | verified 2026-09-15 | Fig. 4: stratified subsample 중 full mix 최고 |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | loss aggregation | paper: sum loss; open-instruct command는 8781471에서 `--reduce_loss sum`, 098424c에서는 flag 없음(default mean) | §4.3.2; [[open-instruct-allenai-recipes]] | conflict | Figs. 5-6; paper value가 LR 선택 |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | chat template; seed | trailing newline 없는 Tülu 3 template(App. B.3); 5 seed best single run, seed 123(60.1); open-instruct command도 seed 123 | Table 13; Table 14; [[open-instruct-allenai-recipes]] (docs/tulu3.md L53) | verified 2026-09-15 | Table 13(template당 run 수 미보고); Table 14 seed 59.8-60.1 |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | peak LR; epoch; batch; length; warmup; seed | \(2\times10^{-6}\); 2; 128; 4,096; 0.03; 3 seed best, seed 456(72.6) | Table 11; §4.3; Table 14 | verified 2026-09-15 | hyperparameter search; seed 70.0-72.6 |
| Llama-3.1-Tulu-3-8B/70B-SFT | 8B, 70B | SFT | packing; NEFTune; optimizer beta | not reported | checked §4.3, Table 11, App. B | not reported | — |
| open-instruct SFT default(released model 아님) | any | SFT | loss mask | 모든 assistant turn(`last_turn_only=False`) | open-instruct@098424c `finetune.py` L132-133, `dataset_transformation.py` L1214-1218 | verified 2026-09-15 (code only) | ablation 없음 |
| zephyr-7b-sft-full(recipe file) | 7B | SFT | LR; schedule; warmup; epoch; max length | 2.0e-05; cosine; 0.1; 1; 2048 | alignment-handbook@1de1fc9 `recipes/zephyr-7b-beta/sft/config_full.yaml` [[hf-alignment-handbook]] | verified 2026-09-15 | ablation 없음; 이 commit file은 2023 run과 다를 수 있음 |
| zephyr-7b-sft-full | 7B | SFT | device batch; accumulation; data; packing/mask key | 16; 1; ultrachat_200k `train_sft`, `test_sft`, 각 weight 1.0; packing/assistant-only/NEFTune key 없음 | same file | verified 2026-09-15 | ablation 없음 |
| zephyr-7b-sft-full(Hub model card, Trainer 생성; Transformers 4.36.2) | 7B | SFT | LR; device; total batch(sequence); optimizer; schedule; epoch; step | 2e-05; 8; 128(16 × 8); Adam \(\beta\) (0.9,0.999), \(\epsilon\) 1e-08; cosine, warmup 0.1; 1; 1,090 | huggingface.co/alignment-handbook/zephyr-7b-sft-full README | verified 2026-09-15 | ablation 없음; config와 LR/schedule/warmup/epoch/device batch 일치 |
| Llama 3 “largest models”(이 locus에서 크기 미출력) | not printed | SFT | LR; step; loss | \(10^{-5}\); 8.5K-9K step; target token cross-entropy, prompt token mask | arXiv:2407.21783v3 §4.1.3 [[llama-3-recipe]] | verified 2026-09-14 | 여러 round/data mix에서 잘 작동; 수치 없음 |
| Qwen2.5 Instruct, open-weight | 0.5B-72B | SFT | epoch; length; LR; weight decay; clipping | 2; 32,768; \(7\times10^{-6}\to7\times10^{-7}\); 0.1; 1.0 | arXiv:2412.15115v2 §4.1 [[qwen-2.5-recipe]] | verified 2026-09-14 | ablation 없음 |
| DeepSeek-R1(Dev1/Dev3 SFT) | 671B MoE | SFT | base; epoch; LR; context; batch | DeepSeek-V3-Base; 2-3; cosine \(5\times10^{-5}\to5\times10^{-6}\); 32,768; 128 | arXiv:2501.12948 v2 B.4.2 [[deepseek-r1-recipe]] | verified 2026-09-14 | ablation 없음 |
| SmolLM3-3B SFT ablation baseline | 3B | SFT | LR; effective batch; epoch; packing; max length | \(1\times10^{-5}\); 128; 1; off; 8,192(Instruct) 또는 32,768 | Smol Training Playbook, “Baby Baselines” [[smol-training-playbook]] | verified 2026-09-15 | LR scan의 3e-6/1e-5가 큰 LR보다 높음; value는 figure에만 |
| LLaMA-2-7B IT/IM(Shi et al.) | 7B | SFT | LR; batch; epoch; length; AdamW \(\beta,\epsilon\); schedule; weight decay | \(2\times10^{-5}\); 128; 보통 2(2,3,10); 2,048; (0.9,0.98), 1e-6; linear, warmup 0.03; 0 | arXiv:2405.14394v2 App. C, Table 6 [[loss-masking-prompt]] | verified 2026-09-15 | ablation 없음 |
| LLaMA-2 7B NEFTune | 7B | SFT | LR; epoch; batch; length; \(\alpha\) | 5e-5(Adam); 3; 128; 512; \(\alpha\) 5,5,10,15(Alpaca/Evol/ShareGPT/OpenPlatypus) | arXiv:2310.05914v2 App. A.1, Table 7 [[neftune]] | verified 2026-09-14 | LR sweep는 LLaMA-1 7B Alpaca; \(\alpha\)는 ChatGPT-judge AlpacaEval에서 {5,10,15} best; epoch 선택 절차 없음; A.2 Table 14는 NEFT 3에서 peak |
| Llama-2-7B Tülu-v2-mix(Biderman et al.) | 7B | SFT(full, LoRA) | LR; optimizer; batch; length | full 5e-6, LoRA 1e-4(\(r=16,64,256,\alpha=2r\)); decoupled LionW(0.9,0.95); 192; 4,096 | arXiv:2405.09673v2 App. C.1 [[lora-learns-less-forgets-less]] | verified 2026-09-15 | initial LR sweep 뒤; App. B Fig. S1 |

소규모 범용 실행의 시작점: 약 8B base model과 약 백만 example general mix에는 검증된 Tülu 3 8B 행이 LR \(5\times10^{-6}\), linear schedule, warmup 0.03, 2 epoch, effective batch 128 sequence, max length 4,096 token, sum loss를 준다. Tülu 3는 Llama 3.1 8B의 prompt 939,344개에 H100 GPU 32개로 6시간 사용했다. Token-mean loss에는 LR이 전이되지 않으므로 sweep해야 한다. 3B와 더 작은 ablation mix에는 SmolLM3 baseline이 LR \(1\times10^{-5}\), effective batch 128, 1 epoch, packing 없음으로 one node H100 8개에서 subset당 30-90분을 썼다. Tülu 3 SFT seed spread가 8B 0.3 average point, 70B 2.6 point였으므로 가까운 비교는 seed 간 반복한다.

## 일반화 관점

**(a) 무엇이 breadth를 넓히는가.** Instruction loss는 Table 1의 작은 dataset 7개 모두에서 18-task NLP mean을, 6개에서 AlpacaEval 1.0을 올렸다([[loss-masking-prompt]] Table 1). NEFTune은 AlpacaEval을 올렸고 ARC, HellaSwag, MMLU, TruthfulQA는 안정적이었다([[neftune]] Fig. 3). Tülu 3 math data는 development 및 unseen math score 모두를 올렸다([[tulu-3]] Table 32). Packing은 WildChat(GPT-4) 69K 70B average를 61.50에서 65.97(random), 65.92(greedy)로 올렸고, 1M-example set에서 8B는 54.3에서 54.95/55.05, 70B는 66.12에서 67.26/67.54로 올렸다([[sequence-packing-contract]], Wang et al. Table 3; 단일 연구, seed 미보고).

**(b) 무엇이 narrowing 또는 forgetting을 일으키는가.** Dolly example 9,229개 instruction tuning은 18-task mean을 base 49.32에서 45.54로 낮췄고([[loss-masking-prompt]] Table 3), IT는 epoch 2-10에 걸쳐 IM보다 “generally” 더 높은 instruction tuning tax였다(Fig. 4). Code full fine-tuning은 LoRA보다 더 잊고 distinct solution도 적게 만들었다([[lora-learns-less-forgets-less]] Tables S5-S6, Fig. 5). Tülu 3 Persona data는 IFEval을 19.2점 올렸지만 IFEval-OOD는 이를 둘 때 0.4점 낮았다([[tulu-3]] Table 32). SmolLM3 baseline set을 effective batch 128로 packing하면 IFEval이 낮아졌다([[smol-training-playbook]], “To Pack or Not to Pack?”; text의 수치 없음). Base model KL penalty는 NLP task를 보호했지만 AlpacaEval 2.0을 base model 0.01 근처까지 낮췄다([[loss-masking-prompt]] Table 3).

**(c) 이 단계에서 어떻게 측정하는가.** Decision에는 development suite, report에는 unseen suite를 쓴다([[tulu-3]] §2.2). Mix에 없는 task에서 base model 대비 forgetting average를 보고한다([[lora-learns-less-forgets-less]] §3.3). Output token의 training/held-out loss와 output-training target BLEU를 overfitting probe로 보고한다([[loss-masking-prompt]] Fig. 3, Table 2). Prompt당 unique generation을 보고하고 seed spread보다 큰 차이일 때만 configuration을 비교한다([[tulu-3]] Table 14). 알려진 오류: LLM-judge win rate는 output length와 함께 오르며, 긴 output만 prompting해도 AlpacaEval win rate 하나가 32.36에서 48.01로 올랐다([[neftune]] §5.2-5.3, Table 5). Development benchmark 하나는 과적합될 수 있고(위 IFEval), exact-string uniqueness는 coarse diversity measure다([[lora-learns-less-forgets-less]] §4.5).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 점검 |
|---|---|---|
| Conversational dataset이 TRL에서 default로 assistant-only라고 가정 | user token label이 −100이 아님; model이 user query를 이어 학습(SmolLM3 “Masking User Turns”) | role별 label 출력; `assistant_only_loss=True` 설정 후 mask 확인 |
| 실수로 last turn만 학습 | assistant token보다 trained token이 적음 | label ≠ −100 수와 assistant token 수 비교 |
| Sum-loss LR을 mean-loss trainer에 복사 | GPU 수 또는 accumulation 변경 시 결과 변화 | fixed total batch에서 accumulation 두 설정 실행 후 비교 |
| Serving client가 버리는 thinking history 학습 | real client에서 agentic held-out score 낮음(§7 MiniMax-M2 비교; training rendering 미보고) | step \(k\)의 training prefix와 client request render diff |
| 작은 set을 unpacked epoch/LR로 packing | epoch당 step이 pack factor만큼 감소; IFEval 하락 | step과 step당 token log; matched step에서 비교 |
| boundary를 position ID에서 정하는 kernel에서 `position_ids` reset 없는 padding-free collation | example이 boundary 간 attention; FLAN 20K Mistral-7B는 같은 35 optimizer step에서 position ID 1.221 대비 validation loss 1.294; Kundu Table 2의 10 model에서는(두 row가 row count도 다름) position ID 없음−있음 loss가 Phi-2 −0.089부터 Falcon +0.170 | position ID에서 도출한 boundary가 example length와 같은지 assert |
| 실패 tool call을 trainable target으로 둠 | error span 안 trained token이 label audit에 보임; source는 downstream effect 미측정 | error turn label audit; erroneous segment mask(GLM-5 §3.1) |
| 긴 trajectory를 `keep`으로 truncate | unterminated response 학습 | truncated row 수를 세고 `terminate` 또는 `drop` 선택 |
| Reporting benchmark에서 epoch/data 선택 | development gain, held-out gain 없음 | 끝까지 보지 않는 unseen suite 유지 |
| Full fine-tuning LR을 LoRA에 사용 | LoRA가 full fine-tuning보다 낮음 | sweep; 보고된 optimum은 LoRA가 약 10× 높음 |
| 한 seed로 결정 | 재실행에서 ranking 전환 | 여러 seed 학습, Table 14 spread와 비교 |

## 이해 점검

1. Completion이 아닌 instruction token에도 loss를 더하면 completion held-out loss가 왜 낮아질 수 있는가? Shi et al.의 어떤 측정이 설명을 지지하며, 어떤 dataset이 균일한 이득을 반박하는가?
2. Prompt-loss-weight worked example로, 최적 PLW가 completion/prompt length ratio 및 benchmark의 long/short generation 요구에 의존하는 이유를 설명하라.
3. 같은 LR/data의 두 run이 gradient accumulation만 다르다. Tülu 3 식으로 각 run이 어떤 sample을 더 weight하는지와 sum loss가 차이를 없애는 이유를 설명하라.
4. Team이 20K-example SFT set을 packing하고 padded run보다 validation loss가 높다. 서로 다른 fix를 예측하는 원인 두 가지와 이를 구별할 측정을 제시하라.
5. Qwen3 및 DeepSeek-V3.2 history rule을 써서, 모든 assistant turn의 single-pass loss와 per-turn expansion이 언제 다른 gradient를 내는지 설명하라.
6. Cross-entropy가 모든 non-target logit을 낮추는데도 ordinary SFT가 model 자기 생성 wrong tool call의 확률을 낮추지 못하는 이유는 무엇인가?
7. Tülu 3 Persona data는 IFEval을 19.2점 올렸고 IFEval-OOD는 올리지 않았다. SFT setting을 어떻게 선택ㆍ보고해야 함을 뜻하는가?
8. LoRA는 code instruction data에서는 full fine-tuning보다 덜 잊었지만 Tülu-v2-mix 2 epoch에서는 full fine-tuning이 덜 잊었다. 이를 설명할 data/training length 성질과 시험 방법은 무엇인가?

## 연결

- 이전: **ch-29e — Unseen Task 일반화를 위한 Instruction Tuning**([[ch-29e]]): unseen-task 일반화를 주는 data property; 이 장은 그런 data의 training setting을 바꾼다.
- 의존성: **ch-04 — Sequence Packing, Loss Masking, Chat Template**([[ch-04]]): `cu_seqlens`, mask, template mechanism.
- 다음: **ch-30a — Fine-Tuning의 Forgetting과 Alignment Tax: 측정과 제어**([[ch-30a]]): §8이 참조한 forgetting report와 control.
- **ch-30b — Multi-Skill SFT Mixture: Interference, Transfer, Agentic/Long-Context Share**([[ch-30b]]); **ch-30c — Generalist Model의 Weight Averaging과 Model Merging**([[ch-30c]]).
- **ch-31a — Supervised Training의 Negative Sample: Correction, Failure Conditioning, Critique, Unlikelihood**([[ch-31a]]); **ch-39 — Offline Preference Optimization: DPO와 variant**([[ch-39]]).
- **ch-26 — Tool과 Function-Calling Data**([[ch-26]]); **ch-27 — Agentic Trajectory Data**([[ch-27]]).
- **ch-36 — Lab: Masking Test, Forgetting Report, Held-Out Evaluation Split가 있는 SFT Run**([[ch-36]]): 이 장의 축을 실행한다.

## 출처

- [[loss-masking-prompt]] — Shi et al. 2024(arXiv:2405.14394v2): IM 정의, Tables 1-5, Figs. 2-4, App. C setting. Library card는 여전히 반대 결과를 말하며 정정이 필요하다.
- [[loss-masking-regimes]] — excerpt: Huerta-Enochian and Ko 2024 PLW result; open-instruct@098424c masking/truncation code; TRL v0.23.0 loss option.
- [[neftune]] — Jain et al. 2023: noise formula, Table 1, overfitting/length analysis, setting.
- [[sequence-packing]] — Krell et al. 2021: padding fraction, BERT position-ID ablation.
- [[sequence-packing-contract]] — excerpt: Krell, Kundu et al. 2024, Wang et al. 2024 packing result.
- [[hf-alignment-handbook]] — alignment-handbook@1de1fc9의 Zephyr-7B-β SFT config와 `alignment-handbook/zephyr-7b-sft-full` Trainer-generated card(card 자체는 unverified).
- [[tulu-3]] — Tülu 3 report v5: Tables 3, 7, 11, 13, 14, 32; §4.3.2 aggregation(card 자체 unverified, value는 paper에서 읽음).
- [[tulu-3-sft-recipe]] — paper에 출력된 Tülu 3 SFT setting/ablation excerpt.
- [[open-instruct-allenai-recipes]] — reproduction command, loss-reduction drift, OLMo 3 template note.
- [[chat-template-matrix]] — excerpt: Tülu 3/Zephyr/Llama 3 template fact; Qwen3 card, DeepSeek-V3.2 history rule.
- [[smol-training-playbook]] — excerpt: SmolLM3 SFT baseline, masking, packing, LR, epoch, template design.
- [[lora-learns-less-forgets-less]] — excerpt: Biderman et al. 2024 learning/forgetting table.
- [[lora-without-regret]] — excerpt: Thinking Machines 2025 LoRA capacity/LR result.
- [[minimax-m2-interleaved-thinking]] — excerpt: retained 대 discarded thinking result.
- [[glm-5]] — excerpt: preserved thinking, masked erroneous segment, token-in-token-out.
- [[orpo]] — SFT가 rejected log-probability를 올림(Fig. 3); odds-ratio term(Fig. 7).
- [[qwen-3]] — Qwen3 report arXiv:2505.09388v1 §4.3, Table 9: thinking-mode-fusion SFT template의 empty think block과 mode flag(card unverified, paper에서 읽음).
- [[llama-3]], [[llama-3-recipe]] — chat protocol, SFT loss/LR, DPO formatting-token mask.
- [[qwen-2.5-recipe]], [[deepseek-r1]], [[deepseek-r1-recipe]] — SFT recipe 행; SFT 없는 R1-Zero RL.
- [[in2-film]] — Mistral-based SFT run의 Mistral `[INST]` template.
- [[allenai-tulu-sft-recipe]] — Corrections에서만 인용; NEFTune, packing, optimizer 행은 Tülu 3 report에 없다.
