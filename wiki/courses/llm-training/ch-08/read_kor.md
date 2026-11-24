# Chapter 8 — 실습: Target 대 General Capability 측정을 포함한 Minimal Trainer

> **핵심 통찰.** supervised fine-tuning run은 동시에 두 quantity를 바꾸며, 그중 하나만 보고하는 lab은 good run과 bad run을 구별할 수 없다. Alpagasus Alpaca 5k(5,305 example)로 fine-tune한 LLaMA-2-7B에서 response-only SFT는 AlpacaEval1.0을0.01에서16.29로 올렸지만, NLP task18개의 mean은49.32에서45.29로4.03 point 내렸다([[loss-masking-prompt]], Table1). 그런 number가 meaningful한지를 결정하는 correctness test는 correct run이 통과하는 test여야 한다. 이 lab의2026-04 version에 있던 six gate 중 둘은 step-1 loss를 `ln(V)`와 비교하고 prompt-token embedding gradient가 정확히 zero여야 한다고 요구하기 때문에 correct code에서도 fail한다.
>
> **지침.** fine-tuning run을 evaluate할 때에는 target gain과 general regression이 같은 optimisation에서 나오고 둘 중 하나만으로는 해석할 수 없으므로, 같은 target task와 같은 broad suite에서 base checkpoint와 fine-tuned checkpoint를 측정한다. proposed acceptance gate가 correct run에서도 fail할 수 있다면, failure에 원인이 하나인 test로 바꾼다. attention masking에는 packed-versus-unpacked equivalence, loss masking에는 `labels != -100`인 position의 decoded inspection, input pipeline에는 first batch에서의 base model 자체 loss를 사용한다. benchmark difference가 한 run의 final checkpoint들 사이 spread보다 작다면—1B model의 ARC Challenge에서는 accuracy1.7 point([[signal-and-noise-eval]] §3.1)—improvement라고 하지 말고 unresolved로 보고한다.

---

## 이 chapter가 general-purpose model에 중요한 이유

이 lab은 foundations phase(ch-00부터 ch-08까지)를 끝내고 pretraining phase 앞에 놓인다. supervised fine-tuning은 reader가 통제하는 한 amount의 data만 달리하는 base model과 trained model, 즉 checkpoint *pair*를 만드는 가장 저렴한 stage이고, generality measurement에는 이 pair가 필요하기 때문에 여기 둔다. pipeline의 모든 이후 stage(mid-training, scale의 SFT, preference optimisation, RL)는 세 operation을 다시 사용한다. 어느 token이 attend되는지, 어느 token이 loss를 갖는지, update를 어떻게 만들고 clip하는지다. 이 세 operation의 error는 exception을 일으키지 않는다. model이 train되는 distribution을 바꾸며, 예상보다 나쁜 number로만 나타난다.

이 stage의 generality question은 좁고 답할 수 있다. target task를 개선하는 fine-tuning run이 base model이 이미 가졌던 ability를 보존하는가? [[loss-masking-prompt]]는 보존하지 못한 case를 측정하고 그 drop을 instruction tuning tax라고 부른다(§4.3 “#3”). 해당 논문의 Fig.4는 LLaMA-2-7B-BASE에서 다섯 dataset의 epoch2부터10에 대해18-task mean을 plot하며, paper가 그 figure를 읽는 방식은 instruction modelling이 response-only loss보다 더 낮은 tax를 갖는다는 것이다. 이 lab의 deliverable은 reader가 통제하는 run에 대한 그 measurement와 run이 correct였다는 evidence를 함께 내는 것이다.

---

## §1 Lab이 만드는 산출물

wiki가 아니라 reader 자신의 run directory에 다음 네 artifact를 만든다.

1. **`symbol-map.md`** — §2의 every operation을 한 pinned commit에 있는 one trainer의 file, symbol, line range에 mapping하고, commit hash를 기록한다.
2. **세 test script** — `packing_equivalence.py`, `decode_labels.py`, `step1_loss.py`(§3)다. 각각 output을 run과 함께 보관하는 file에 쓴다.
3. **한 run과 작은 ablation grid** — reader의 compute scale에서 training run 하나를 수행하고, budget이 허용하는 subset에서 §5의 lever를 적용한다. 각각 seed를 최소 두 개 쓰거나 seed를 돌리지 않았다는 명시적 statement를 남긴다.
4. **`target-vs-general.md`** — checkpoint별로 one row를 갖는 table 하나다. column은 target-task metric, broad-suite score(MMLU, GSM8K, HumanEval, IFEval), 적어도 두 domain의 held-out bits-per-byte, second prompt format에서의 same target metric이다. base checkpoint가 first row다.

**Acceptance gate.** (1) `symbol-map.md`는 그 commit에 실제로 있는 commit hash와 line number를 cite한다. (2) packed와 unpacked forward pass는 §3.1에 기록한 tolerance 안에서 same token에 대해 agree한다. (3) `labels != -100`인 position의 decoded text는 end-of-turn token을 포함해 정확히 assistant turn이어야 한다. (4) step-1 training loss는 same batch의 base model loss와 logging precision 안에서 같아야 한다. (5) `target-vs-general.md`에는 base-model row와 second-format column이 있다. gate2,3,4에 fail한 run은 원인을 찾기 전에는 evaluate하지 않는다. 그 evaluation은 bug를 측정하게 되기 때문이다.

---

## §2 Pinned commit에서 trainer 하나를 처음부터 끝까지 읽기

**정의.** pinned-commit reading은 framework에 관한 every claim을 source의 named commit과 대조하고 claim 옆에 commit hash를 적는다는 뜻이다.

**이것이 다루는 문제를 측정 가능하게 말하면.** framework symbol은 release 사이에서 이동하거나 사라진다. 아래에서 읽는 trainer와 같은 repository에서 `GRPOTrainer._compute_loss`는 commit `a08e713`(2026-04-21)에서는 L2418에 있고 commit `a04ffd3`(2026-09-14)에서는 L3113에 있으며, 그 두 commit 사이에 default `max_completion_length`가512로 바뀌었다([[trl-grpo]], Verification). 따라서 blog post에서 copy한 chapter나 configuration은 조용히 낡는다. 더 이상 없는 argument는 construction에서 reject되거나, class가 arbitrary keyword argument를 받는 경우에는 ignore된다.

**메커니즘 — commit `aa89588`에 있는 TRL `SFTTrainer`의 data path** ([[trl-sft-trainer]]):

1. `_prepare_dataset`(L1438)은 필요하면 dataset을 ChatML로 convert하고 non-conversational example에는 EOS token을 append하며 chat template을 통해 tokenize한다. prompt-completion data에는 template을 두 번 적용한다. 먼저 `add_generation_prompt=True`인 prompt, 다음으로 prompt + completion을 적용하며, second tokenization이 first tokenization으로 시작하지 않으면 warn한다(L1536-1556).
2. `assistant_only_loss=True`이면 tokenizer를 `return_assistant_tokens_mask=True`로 호출한다(L1543). `{% generation %}` marker가 없는 template은 `get_training_chat_template`로 replace되고(L1251-1258), assistant token을 전혀 내지 않는 example은 raise한다(L1585-1591).
3. `build_labels`(L1606-1632)는 applicable mask bit가0인 곳마다 `-100`을 써서 `labels = input_ids`를 만든다.
4. truncation은 packing이 off일 때에만 수행하며, fully masked로 남은 example은 drop한다(L1634-1660).
5. `pack_dataset`(L1678)은 row를 `max_length`까지 pack하고 `seq_lengths` column을 추가한다.
6. `DataCollatorForLanguageModeling`(L403)은 label을 `-100`으로 pad하고, `seq_lengths`에서 `position_ids`를 derive한다(L528-552). padding-free일 때 batch를 one row로 flatten하고 `labels[position_ids == 0] = -100`으로 설정한다(L516). 따라서 packed document 각각의 first token은 loss를 갖지 않는다.

update 자체는 `transformers.Trainer`에서 inherit된다([[hf-trainer-loop]]). `_inner_training_loop` 안에서 이 순서로 L2697-2718에서 clipping, L2740에서 `optimizer.step()`, L2750에서 `lr_scheduler.step()`을 수행한다. `training_step`(L3981)은 forward pass, gradient-accumulation division, `accelerator.backward`(L4059-4070)만 수행한다.

**concept-to-symbol map**(위 locus에서 verified했다. 같은 map을 click 가능한 path와 각 stage를 덮는 test로 보려면 [figures/trainer-map.html](figures/trainer-map.html)을 참고):

| Concept(chapter) | `trl@aa89588` / `transformers@v4.57.1`의 symbol | Locus |
|---|---|---|
| Chat template(ch-04) | `tokenize_fn` 안의 `_tokenize(..., chat_template=self.chat_template)` | `sft_trainer.py` L1519-1603 |
| Loss masking(ch-04) | `assistant_only_loss` / `completion_only_loss` → `build_labels` | `sft_config.py` L259-280; `sft_trainer.py` L1606-1632 |
| Packing(ch-04) | `packing`, `packing_strategy` → `pack_dataset`, column `seq_lengths` | `sft_config.py` L223-238; `sft_trainer.py` L1662-1678 |
| Block-diagonal attention(ch-04) | `padding_free` → `seq_lengths`에서 `position_ids`; FlashAttention variant 필요 | `sft_trainer.py` L1169, L394-398, L528-552 |
| Mixed precision(ch-02) | `TrainingArguments`의 `bf16`; FSDP는 optimizer step을 full precision으로 유지 | [[fsdp-sft]] §4.4 |
| Gradient clipping(ch-01) | `args.max_grad_norm` → `accelerator.clip_grad_norm_` | `trainer.py` L2697-2718 |
| Optimizer와 scheduler 순서(ch-03) | `optimizer.step()` 다음 `lr_scheduler.step()` | `trainer.py` L2740, L2750 |
| Schedule horizon(ch-03) | packed dataloader에서 `max_steps = ceil(num_train_epochs × num_update_steps_per_epoch)` | `trainer.py` L5675-5689 |
| In-loop metric(ch-06) | `entropy`, `mean_token_accuracy`, `num_tokens` | `sft_trainer.py` L1815-1876 |

**이 commit에 존재하지 않는 이름.** 그러므로 configuration을 읽지 않고 copy했다는 sign은 `train_on_response_only`, `max_seq_length`(현재는 `max_length`), `DataCollatorWithPacking`, `DataCollatorForCompletionOnlyLM`, `ConstantLengthDataset`이다([[trl-sft-trainer]]). library card [[hf-alignment-handbook]]에는 그런 snippet이 있었지만 repository 자체의 Zephyr configuration에는 `packing` key, masking flag, FSDP block이 없고, DeepSpeed ZeRO-3 accelerate config로 launch한다([[hf-alignment-handbook]] excerpt, `config_full.yaml` 및 `scripts/sft.py` L20-35).

**대조를 위한 두 번째 trainer.** nanochat의 `scripts/chat_sft.py`는 같은 operation을 one file에서 수행하지만 다른 convention을 쓴다. ignore index는 `-100`이 아니라 `-1`이고, loss mask는 template marker가 아니라 `tokenizer.render_conversation`에서 오며, packing은 “no tokens are ever discarded”가 되도록 truncate가 아니라 pad한다([[nanochat]], L180-187, L285-296). 두 trainer는 같은 세 concept에 서로 다른 spelling을 부여한다. 따라서 한 trainer에 맞춰 쓴 script를 다른 trainer에서 실행하면 error를 내지 않은 채 model을 wrong token으로 train한다.

**general-purpose model에 대한 함의.** 이 stage 각각은 gradient가 어느 token을 볼지 선택한다. prior assistant turn을 포함하는 mask, document들이 서로 attend하는 packed row, end-of-turn token이 loss 밖에 있는 template은 모두 loss curve를 smooth하게 유지하면서 trained distribution을 바꾼다.

---

## §3 Correct run이 통과하는 세 test

### 3.1 Packed 대 unpacked equivalence

**정의.** 같은 example을 두 번 run한다. 한 번은 document별 position ID와 함께 one row로 pack하고, 다른 한 번은 document당 one row로 unpack한다. 같은 token에서 token별 log-probability를 비교한다.

**메커니즘.** packing은 document를 fixed-length row에 concatenate한다. correct packing은 attention을 within-document pair로 restrict하고 document마다 position ID를0부터 다시 시작한다. row 전체에 dense causal mask를 쓰면 earlier position 모두에 attend할 수 있다.

**Worked example.** 3 token과2 token인 document 둘을5 token row 하나에 pack한다고 하자. 각 document로 제한한 causal mask 아래 allowed query-key pair 수는 첫 번째에서3·4/2 = 6, 두 번째에서2·3/2 = 3이므로 total9개다. row 전체의 dense causal mask 아래에서는5·6/2 = 15 pair가 allowed다. 15개 중6개, 즉40%가 document boundary를 cross한다. 그6 pair는 forward pass와 gradient에 들어가며 loss value는 이들을 within-document pair와 분리하지 않는다.

**근거.** 20K FLAN subset에서 Mistral-7B를 one epoch fine-tune했을 때 validation loss는 padding(no packing)에서1.129, position ID 없는 packing에서1.306, position ID *있는* offline packing에서1.284, position ID 있는 online minibatch packing에서1.127이었다([[packing-with-flash-attention]] Table2; 8×A100, max sequence length4096, one seed). Llama-2-7B의 같은 row는1.266 / 1.579 / — / 1.262였고 Falcon-7B는1.842 / 2.585였다. **결과(single study).** author는 offline packing의 residual gap을 attention이 아니라 one epoch의 optimisation step 수가 더 적기 때문이라고 본다(§4.1, §4.3). 그러므로 이 lab의 test는 end-of-run loss를 비교하지 않고 fixed token에서 forward pass를 비교한다.

**Test.** batch 하나를 두 방식으로 build하고 CPU의 `float32` 또는 deterministic kernel로 model을 run한 뒤, response position에서 `log p(token)`을 비교한다. correct position ID와 FlashAttention variant를 사용하면 difference는 floating-point reassociation 수준이다. 관측한 maximum absolute difference를 기록하고 이후 run의 tolerance로 삼는다. difference가0.1 nat 정도이면 attention이 document boundary를 cross하고 있다. TRL은 BFD packing을 non-Flash attention implementation과 결합하면 raise 대신 warning을 낸다([[trl-sft-trainer]] L1236-1244). 그러므로 warning은 이 test의 대체물이 아니다.

### 3.2 `labels != -100`인 position의 decoded inspection

**Test.**

```python
row = batch["labels"][0].tolist()
kept = [i for i, lab in enumerate(row) if lab != -100]
print(tokenizer.decode([batch["input_ids"][0][i] for i in kept]))
```

print된 text는 assistant turn만이어야 하고 다른 것은 없어야 하며 end-of-turn token으로 끝나야 한다. chat template이 그 token을 next message에 attribute하면 TRL은 “the model may not learn to stop”이라고 warn한다([[trl-sft-trainer]] L1260-1266). stop token이 loss에 없는 채 train된 model은 inference에서 turn 끝을 지나 generate한다. 이는 training-time symptom 없이 생기는 capability failure다.

**2026-04 version의 masking test가 invalid인 이유.** 그 version은 `model.get_input_embeddings().weight.grad[prompt_ids].abs().sum() == 0.0`를 assert했다. correct code에서도 다음 두 mechanism 때문에 false다.

1. response position은 prompt position에 attend한다. prompt position *j*의 key와 value vector는 prompt token의 input embedding row의 function이므로, prompt position이 loss를 carry하지 않더라도 attention path를 통한 `∂L/∂E[prompt token]`은 non-zero다.
2. input embedding과 output embedding이 tied이면 같은 tensor가 output projection이다. softmax probability `p`와 target `y`에 대해 logit `z_j`에 대한 gradient는 `p_j − 1[j = y]`다. 여기서 `p_j`는 vocabulary item *j*에 대해 model이 주는 probability이고, `1[j = y]`는 target이면1, 아니면0이다. `p = (0.5, 0.3, 0.2)`와 target0이면 gradient는 `(−0.5, +0.3, +0.2)`다. token 어느 것도 batch에 나타나지 않았는데도 row1과2는 gradient를 받는다. every vocabulary row는 every supervised position에서 gradient를 받는다.

correct masking 아래 zero인 quantity는 position별 loss contribution이며, 이것이3.2가 직접 inspect하는 것이다.

### 3.3 Base model 자체 loss에 대한 step-1 loss

**Test.** training 전에 same masking으로 first prepared batch에서 base model loss를 compute한다. 이어 training을 시작해 step-1 logged loss를 읽는다. 둘은 same forward pass이므로 logging precision 안에서 agree해야 한다.

**`ln(V)`가 wrong reference인 이유.** `ln(V)`는 vocabulary size `V`의 uniform predictor loss이며 randomly initialised model을 나타낸다. `V = 128,000`이면 `ln V = 11.76`이다. instruction data에서 pretrained base model은 token당1-2 nat 근처에 있다. [[packing-with-flash-attention]] Table2가 측정한 one-epoch SFT validation loss는1.129(Mistral-7B),1.266(Llama-2-7B),1.169(Granite-8B-code),1.842(Falcon-7B)다. 따라서 “step-1 loss가 `ln(V)`의20% 안”이라는 gate는 every correct run을 fail시키고 label이 random인 run을 pass시킨다. pretrained checkpoint에서11.76 근처의 step-1 loss 자체가 diagnostic이다. label shift, template, tokenizer 중 하나가 wrong이다.

**loss를 comparable하게 만들기 — worked example.** token당 nat 단위 cross-entropy는 tokenizer에 depend한다. bit per byte는 그 dependence를 제거한다. `bpb = ℓ / (B · ln 2)`이고, `ℓ`는 nat 단위 summed negative log-likelihood, `B`는 text의 UTF-8 byte 수다([[paloma]] App.B). token당 평균3.8 byte인 text에서 loss가 token당1.30 nat이면 `bpb = 1.30 / (0.6931 × 3.8) = 0.49`다. 같은 conversion으로 lab은135M run을7B run 및 base model과 one axis에서 비교할 수 있다.

---

## §4 General capability에 대비해 target capability 측정하기

**명시한 측정.** checkpoint마다 네 quantity를 낸다. (a) target-task metric, (b) training data가 target하지 않는 broad suite—MMLU, GSM8K, HumanEval, IFEval, (c) one domain보다 많은 domain에서 bits-per-byte로 낸 held-out loss, (d) second prompt format에서의 target metric이다. base checkpoint도 identical harness, same day, same decoding parameter로 측정한다.

**held-out set이 하나보다 많아야 하는 이유.** single held-out set은 composition을 밝히지 않은 mixture다. pretraining corpus만 다른1B model 여섯 개에서 C4-only model의 RedPajama arXiv domain per-domain perplexity는391,171까지 갔다. C4와 mC4-en model은 약20B-token checkpoint에서 약150B-token checkpoint로 갈 때 각각65개와43개 domain에서 *나빠졌지만*, aggregate number 하나는 계속 좋아졌을 것이다([[paloma]] §4.1, App.D.1.1). **결과(single study)**이며1B scale과 English plus code에만 해당한다. 같은 source에서 lab으로 가져오는 procedural detail 두 개는 다음이다. document를 concatenate하지 말고 BOS token 뒤에서 각각 score한다(Pythia-1.4B, 2B token: concatenated92.23 ± 17.33 대 separate42.57 ± 0.29, Table17). 그리고 number를 믿기 전에 evaluation text를 training set against decontaminate한다(§3 G1).

**Noise와 difference로 볼 기준.** 1B model의 final checkpoint30개는 ARC Challenge에서 accuracy1.7 point span을 보인다([[signal-and-noise-eval]] §3.1). benchmark30개 전체에서 primary metric 대신 bits-per-byte로 score하면 average signal-to-noise ratio가10.0에서31.5로, small-scale decision accuracy가77.0%에서83.7%로 올랐다(Fig.6). 이 lab에 대한 consequence는 두 가지다. every ablation에 accuracy와 함께 bits-per-byte를 보고하고, seed 하나의1 point accuracy difference는 unresolved로 취급한다.

**Format robustness.** task53개에 대해 meaning-preserving prompt format10개를 sample했을 때, best format과 worst format 사이 median spread는 accuracy7.5 point였다. task20%는 every LLaMA-2 setting에서 적어도15 point spread였고, LLaMA-2-13B의 observed maximum은76 point였다. 더 큰 model, 더 많은 few-shot example, instruction tuning도 이 sensitivity를 제거하지 못했다([[prompt-format-sensitivity-formatspread]] §4.2, abstract). d = 0.02에서 LLaMA-2-13B와-70B의 model comparison은 format change에 따라0.141 확률로 reverse된다(§4.2, Fig.4). SFT checkpoint의 single-format score는 checkpoint와 format을 함께 측정한다. 이것이 gate5가 two format을 요구하는 이유다.

**이 측정이 드러내는 trade-off.** [[loss-masking-prompt]] Table1은 이 pattern을 one table에 제시한다. LLaMA-2-7B base는18-task mean에서49.32, AlpacaEval1.0에서0.01을 score한다. Alpagasus Alpaca5k에서 response-only SFT를 한 뒤에는45.29와16.29, Alpagasus Dolly9k에서 same SFT를 한 뒤에는45.54와21.54를 score한다. open-ended metric은16-21 point 오르는 동안 broad-suite mean은 약4 point 내린다. Table1의 every dataset과 method를 두 axis에 plot한 interactive version은 [figures/target-vs-general.html](figures/target-vs-general.html)에 있다. §1의 memo는 reader 자신의 checkpoint 두 개에 대한 같은 plot이다.

---

## §5 Ablation grid

아래 lever 각각은 적어도 한 setting에서 measured evidence를 갖는다. lab은 budget이 허용하는 subset을 seed와 함께 run하고, 각 cell에 대해 §4의 두 axis를 모두 보고한다.

| Lever | 비교할 setting | source가 측정한 것 | Locus |
|---|---|---|---|
| Epoch | 1 / 2 / 3 | Tülu2 mixture에서 Llama3.0, sum loss, LR5e-6: average performance는2 epoch에서 peak이며7까지 개선되지 않음 | [[tulu-3]] §4.3.2, Fig.6 |
| Epoch(second setting) | 1 / 3 / 5 / 7 / 9 | NEFTune을 쓴 LLaMA-2-7B + Alpaca, AlpacaEval(ChatGPT judge):55.09 / 62.55 / 62.24 / 60.50 / 58.14 | [[neftune]] Table14 |
| Epoch, general side | 2부터10 | 다섯 dataset의 LLaMA-2-7B-BASE: response-only와 instruction-modelling loss에 대해18-task NLP mean을 epoch2부터10까지 plot; paper는 drop을 instruction tuning tax라고 부르고 instruction modelling의 tax가 더 낮다고 보고 | [[loss-masking-prompt]] §4.3 “#3”, Fig.4 |
| Learning rate와 loss reduction | 2e-6, 5e-6, 1e-5, 2e-5에서 sum 대 mean loss | Tülu2 mixture의 Llama3.0: 5e-6에서 sum loss가 best average를 냄 | [[tulu-3]] §4.3.2, Fig.5 |
| Response-only 대 instruction-modelling loss | prompt token을 mask하거나 하지 않음 | LLaMA-2-7B, Alpagasus Alpaca5k: instruction modelling에서18-task mean45.29 → 47.47, AlpacaEval1.0 16.29 → 19.52 | [[loss-masking-prompt]] Table1 |
| NEFTune | off / α = 5 | LLaMA-2-7B + Alpaca, AlpacaEval(GPT-4 judge)29.79 → 64.69; ARC, HellaSwag, MMLU, TruthfulQA는 “remain stable”; α는 AlpacaEval 자체에서 선택 | [[neftune]] Table1, §4, App.A.1 |
| evaluation 시 prompt format | 적어도 두 format | format10개, task53개에서 median spread7.5 point | [[prompt-format-sensitivity-formatspread]] §4.2 |
| Seed | seed ≥ 2, 또는 none을 run했다는 statement | 1B에서 ARC Challenge final-checkpoint span1.7 point | [[signal-and-noise-eval]] §3.1 |

**Worked example: loss reduction이 detail이 아니라 lever인 이유.** micro-batch 둘을 하나의 update로 accumulate한다. micro-batch A는 loss-bearing token10개와 summed token loss20 nat, micro-batch B는 token90개와 summed loss90 nat를 갖는다. 둘을 하나의 forward pass로 처리하면 `(20 + 90) / (10 + 90) = 1.10`을 report한다. micro-batch mean을 average하면 `(2.0 + 1.0) / 2 = 1.50`을 report하며 A의 token 각각에는 `1/(2 × 10) = 0.05`, B의 token 각각에는 `1/(2 × 90) = 0.0056`을 weight로 준다. equal해야 하는 token 사이 factor9 차이다. Tülu3는 이 effect를 Eq.1-2로 제시하며, 이를 제거하기 위해 adjusted learning rate와 sum loss를 사용한다([[tulu-3]] §4.3.2). lab은 trainer가 쓴 reduction을 보고한다. TRL의 `chunked_nll`은 trainer가 해당 count를 supply하면 loss를 `sum / num_items_in_batch`로 reduce하고, 그러지 않으면 local valid token에 대한 mean으로 reduce한다(`sft_trainer.py` L161-164, L225-231; [[trl-sft-trainer]]).

**Training-format diversification.** 이 chapter에서 읽은 source에서는 여러 chat template 아래 training하는 것이 이 scale에서 robustness를 개선하는지는 **open question**이다. [[prompt-format-sensitivity-formatspread]]는 evaluation-time sensitivity를 측정하고 instruction tuning이 이를 제거하지 않는다고 말하지만 fine-tuning 중 template diversification은 test하지 않는다. 그러므로 lab은 diversification을 sign을 모르는 experiment로 취급하고 two format evaluation은 requirement로 둔다. ch-30이 SFT-specific literature로 training-side question을 다룬다.

---

## §6 두 compute path

**Full-budget path.** H100 8개 node 하나, 7-8B base model, prompt 수만 개의 single-source instruction set을 쓴다. published reference point 두 개가 비용의 bound를 준다. Tülu3 8B model은939K-prompt mixture에서 effective batch128, maximum sequence length4,096, 2 epoch로 GPU32개에서6시간 학습했고, 70B model은 GPU64개에서50시간 학습했다([[tulu-3]] §4.3). nanochat의 전체 pipeline—tokenizer, depth-24 model의 pretraining, SFT, evaluation—은 H100 8개 node 하나에서 약1.5시간으로 document되어 있고 README는 그 node의 가격을 시간당 약$24로 책정한다([[nanochat]] `speedrun.sh` L3-4, README L341). 이 lab은 reader의 throughput을 predict하지 않는다. `SFTTrainer`는 step마다 `num_tokens`를 log하며([[trl-sft-trainer]] L1854-1866), measured token per second를 memo에 넣는다.

**Resource-constrained path.** memory16GB 이하의 GPU 하나, 125M-500M base model, prompt 수천 개를 쓴다. §1의 every gate와 §3의 every test는 변함없이 적용되며, before/after generality table도 그렇다. broad suite는 same harness와 same two format에서 score한다. 바뀌는 것은 expected result다. 이 scale에서는 target gain이 작고 general regression이 지배적일 수 있다. 이를 정직하게 보고하는 것이 deliverable이다. nanochat은 SFT script를 처음부터 끝까지 읽을 수 있고 held-out bits-per-byte 및 composite chat metric을200 step마다 이미 evaluate하므로 이 path의 reference다([[nanochat]] L59-62).

**Distributed setting.** model이 optimizer state와 함께 one device memory에 fit하지 않을 때 FSDP full sharding을 쓴다. parameter와 gradient는 low precision에서 compute하지만 optimizer step은 full precision에서 수행한다. 따라서 parameter peak memory는 `K_full·Σψ_i/F + K_full·max_i ψ_i`에서 `K_full·Σψ_i/F + K_low·max_i ψ_i` byte로 낮아진다. 여기서 `ψ_i`는 FSDP unit *i*의 element count, `F`는 sharding factor, `K_low`, `K_full`은 각 precision에서 element당 byte 수다([[fsdp-sft]] §4.4). 그렇지 않으면 single-device training을 쓴다. lab의 question은 sharding을 요구하지 않고 sharding은 every anomaly에 두 번째 explanation을 추가한다.

---

## 레시피

Lab value는 이 chapter가 내린 choice이며, 아래 every source value는 밝힌 locus에서 읽었다.

| Model(정확한 release) | Size | Stage | Setting | Value | Source 위치 | Status | 이 value의 근거 | Lab value | 차이의 이유 |
|---|---|---|---|---|---|---|---|---|---|
| zephyr-7b-sft-full | 7B | SFT | base model; precision; attention | `mistralai/Mistral-7B-v0.1`; bf16; `flash_attention_2` | alignment-handbook@1de1fc9 `recipes/zephyr-7b-beta/sft/config_full.yaml` | verified 2026-09-17 | ablation 보고 없음 | SmolLM2-135M 또는 Qwen2.5-0.5B; bf16; 가능하면 FlashAttention variant | GPU 하나에 fit; packing test에는 Flash variant 필요 |
| zephyr-7b-sft-full | 7B | SFT | peak LR; schedule; warmup; epoch | 2.0e-5; cosine; `warmup_ratio: 0.1`; 1 | same file | verified 2026-09-17 | ablation 보고 없음 | same LR와 schedule; ablation으로 epoch1/2/3 | epoch은 §5의 lever |
| zephyr-7b-sft-full | 7B | SFT | sequence length; per-device batch; accumulation | `max_seq_length: 2048`; 16; 1 | same file | verified 2026-09-17 | ablation 보고 없음 | `max_length: 1024`; per-device4; sequence128에 맞추는 accumulation | 16GB memory |
| zephyr-7b-sft-full | 7B | SFT | global batch(sequence) | 128 | derived: 16 × 1 × 8 device, device count는 `scripts/sft.py` L20-21 | derived | file에 print되지 않음 | sequence128 | batch를 유지하고 split을 변경 |
| zephyr-7b-sft-full | 7B | SFT | packing; loss masking; grad clip | `packing` key 없음; masking flag 없음; `max_grad_norm` 없음 | same file | not reported(YAML, `scripts/sft.py` 확인) | — | `packing=True`, `assistant_only_loss=True`, `max_grad_norm=1.0` | lab은 packing과 masking을 명시적으로 test |
| Tülu3 8B | 8B | SFT | epoch; peak LR; loss reduction | 2; 5e-6; sum loss | arXiv:2411.15124 §4.3 및 Table11(epoch, LR); §4.3.2(sum loss) | verified 2026-09-17 | §4.3.2 Fig.5(sum loss가4 LR × 2 reduction 중5e-6에서 best) 및 Fig.6(2-7 중2 epoch best), Tülu2 mixture의 Llama3.0, seed1개 | 2 epoch; handbook row의 LR | 939K prompt 대 수천 개; LR은 data size를 넘어 transfer하지 않음 |
| Tülu3 8B | 8B | SFT | effective batch; max sequence length; compute | 128; 4,096; GPU32개에서6시간 | arXiv:2411.15124 §4.3 | verified 2026-09-17 | ablation 보고 없음 | sequence128; token1,024 | scale |
| Tülu3 70B | 70B | SFT | peak LR; compute | 2e-6; GPU64개에서50시간 | arXiv:2411.15124 §4.3 | verified 2026-09-17 | “found after a hyperparameter search”; search는 보고되지 않음 | 사용하지 않음 | lab scope 밖 |
| LLaMA-2-7B(IT 및 IM run) | 7B | SFT | LR; batch; epoch; schedule; weight decay | 2e-5; sequence128; 2; linear, warmup ratio0.03; 0 | arXiv:2405.14394v2 App.C Table6 | verified 2026-09-17 | Table1이 dataset7개의 outcome을 보고; seed1개 | same LR와 batch | Table1 baseline과 direct comparability |
| LLaMA-2-7B + Alpaca(NEFTune) | 7B | SFT | LR; epoch; batch; max length; α | 5e-5; 3; 128; 512; 5 | arXiv:2310.05914v2 App.A.1, Table7 | verified 2026-09-14([[neftune]]) | Table14 epoch sweep; α는 AlpacaEval 자체에서 선택 | on/off ablation으로 α = 5 | target metric에서 α를 선택하는 것은 §5가 warn하는 failure mode |
| nanochat SFT(`chat_sft.py`) | depth-24 | SFT | initial LR fraction; warmup; warmdown; eval cadence | base LR의0.8; 0.0; 0.5; 200 step마다 bits-per-byte와 ChatCORE | github.com/karpathy/nanochat@f527f76 `scripts/chat_sft.py` L54-62 | verified 2026-09-17 | ablation 보고 없음 | eval cadence copy | in-loop general measurement은 option이 아니라 gate |
| TRL `SFTConfig`(framework default; trained model 아님) | — | SFT | `max_length`; `packing`; `packing_strategy`; `assistant_only_loss`; `loss_type` | 1024; `False`; `"bfd"`; `False`; field default `None`, `"chunked_nll"`로 resolve | trl@aa89588 `sft_config.py` L203-293 | verified 2026-09-17 | framework default; ablation 없음 | lab config에서 every one을 explicit하게 설정 | default는 commit 사이에 변함([[trl-grpo]] Verification) |

**작은 general-purpose run의 시작점.** GPU 하나에서 instruction example 수천 개로0.5B base model을 run한다면 bf16, `warmup_ratio = 0.1`인 cosine schedule, peak LR2.0e-5, first epoch1과 ablation epoch2,3, `max_length = 1024`, accumulation으로 도달하는 global batch sequence128, BFD strategy와 FlashAttention variant를 사용하는 `packing=True`, `assistant_only_loss=True`, `max_grad_norm = 1.0`에서 시작한다. LR, schedule, warmup, epoch count는 verified Zephyr row(Mistral-7B, UltraChat-200K, 8×H100 node 하나)와 Tülu3 row(8B, prompt939K, GPU32개)에서 왔다. 어느 source도0.5B나 example 수천 개에서 run하지 않았으므로 이 value는 이 scale에서 근거가 뒷받침하는 recommendation이 아니라 ablate할 starting point다.

---

## 일반화 관점

**(a) breadth를 늘리는 것.** 두 axis를 모두 측정하는 것 자체가 breadth를 시야에 두는 intervention이다. 18-task mean을 protect하는 lever는 그것을 측정할 때만 보인다(instruction-modelling loss는 Alpagasus Alpaca5k에서 그 mean을2.18 point recover한다, [[loss-masking-prompt]] Table1). target metric이 아니라 broad average에서 선택한 epoch count에서 stop하는 것이 두 번째다. Tülu2 mixture에서2 epoch는 evaluation suite average로 선택됐다([[tulu-3]] Fig.6). bits-per-byte로 score하면 그 뒤의 every small-scale decision reliability가 올라간다(30-task average SNR10.0 → 31.5, [[signal-and-noise-eval]] Fig.6).

**(b) narrowing을 일으키는 것.** 하나의 narrow distribution에서 더 오래 train하는 것: LLaMA-2-7B-BASE의18-task mean은 epoch2-10에 대해 plot되며 paper는 drop을 instruction tuning tax라고 부른다([[loss-masking-prompt]] §4.3 “#3”, Fig.4). target metric만으로 hyperparameter를 select하는 것: NEFTune α value는 AlpacaEval에서 선택됐고 safety는 evaluate되지 않았다([[neftune]] App.A.1, §7). one prompt format에서 training하고 evaluate하는 것: format sensitivity는 instruction tuning 뒤에도 남으며 single format은 median7.5 point spread를 숨긴다([[prompt-format-sensitivity-formatspread]] §4.2). silent input-pipeline error도 narrow한다. position ID 없는 packing은 Mistral-7B validation loss를1.129에서1.306으로 올렸다([[packing-with-flash-attention]] Table2).

**(c) 이 stage에서 측정하는 방법.** training set against decontaminate하고 BOS 뒤에서 document를 separately score한 domain별 held-out bits-per-byte([[paloma]] §3, App.B, Table17), base와 trained checkpoint에 identically run한 broad accuracy suite(MMLU, GSM8K, HumanEval, IFEval), 두 prompt format 아래 target metric, 그리고 seed 또는 one run final checkpoint 사이 spread에서 얻어 result와 함께 명시한 noise floor를 낸다([[signal-and-noise-eval]] §3.1).

---

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 검사 |
|---|---|---|
| “step-1 loss ≈ `ln(V)`”로 gate | every correct run이 gate fail; shuffled label run은 pass | step-1 loss를 same batch base model loss와 비교(§3.3); pretrained base는1-2 nat 근처([[packing-with-flash-attention]] Table2) |
| prompt-token embedding gradient가 정확히 zero라고 assert | correct masking에서도 assertion fail | 대신 decoded `labels != -100` position을 inspect(§3.2); attention과 tied embedding 모두 unsupervised row로 gradient route |
| 존재하지 않는 argument(`train_on_response_only`, `max_seq_length`)가 있는 config copy | argument가 reject되거나 silently ignore; masking이 never apply | run 전 pinned commit에서 symbol을 grep([[trl-sft-trainer]]) |
| `training_step` 또는 `_maybe_log_save_evaluate`에 clipping instrumentation | hook이 never fire; `grad_norm`가 `None`으로 log | clipping은 `_inner_training_loop`의 `trainer.py` L2697-2718에 있음([[hf-trainer-loop]]) |
| packing이 cosine horizon을 깨뜨린다고 가정 | 존재하지 않는 bug에 시간 낭비 | `max_steps`는 packed dataloader에서 compute(`trainer.py` L5675-5689); pre-packing step count로 hand-built scheduler를 만들 때가 실제로 깨지는 경우 |
| non-FlashAttention implementation으로 packing | log에 warning, 약간 나쁜 loss, error 없음 | packed-versus-unpacked test(§3.1); TRL은 warning만 냄([[trl-sft-trainer]] L1236-1244) |
| `{% generation %}` 없는 template 또는 stop token을 제외한 template에서 `assistant_only_loss=True` | inference에서 model이 generate를 stop하지 않음 | L1251-1266 trainer warning; kept position을 decode해 end-of-turn token이 있는지 확인 |
| target task만 evaluate | broad-suite mean4 point를 잃은 run을 success라고 보고 | `target-vs-general.md`의 base-model row([[loss-masking-prompt]] Table1이 reference pattern) |
| prompt format 하나만 score | 다른 방식으로 prompt를 쓰는 reader에게 result가 reproduce되지 않음 | 두 format을 모두 report([[prompt-format-sensitivity-formatspread]] §4.2) |
| held-out perplexity에 document concatenate | run 사이 held-out loss variance가 큼 | BOS 뒤 document 각각을 score(92.23 ± 17.33 대42.57 ± 0.29, [[paloma]] Table17) |
| seed 하나에서1 point benchmark change 보고 | “improvement”가 rerun에서 survive하지 않음 | 1B ARC Challenge의 final-checkpoint spread1.7 point와 비교([[signal-and-noise-eval]] §3.1) |
| target metric으로 α, LR, epoch select | target metric은 오르나 broad suite는 내리고 selection이 이를 숨김 | broad suite 또는 held-out average에서 select하고 무엇을 썼는지 명시 |

---

## 이해 점검

1. §3.2의 masking test는 gradient가 아니라 decoded token을 inspect한다. prompt-token embedding row에 대한 zero-gradient assertion이 false가 되는 두 mechanism을 제시하고, input/output embedding이 tied되지 않은 model에도 여전히 적용되는 mechanism을 말하라.
2. pretrained7B base와128,000-token vocabulary에서 run이 step-1 loss11.8을 report한다. test할 candidate cause를 순서대로 나열하고, 각각이 decoded `labels != -100` output에 무엇을 할지 말하라.
3. one epoch에서 position ID 없는 packing은 Mistral-7B validation loss를1.129에서1.306으로 올렸지만, position ID *있는* offline packing도1.284로 올렸다. 두 increase의 cause가 다른 이유와, lab test가 end-of-run loss가 아니라 forward pass를 비교하는 이유를 설명하라.
4. checkpoint가 target task에서6 point 얻고18-task mean에서3 point 잃었다. 그3 point loss가 real regression인지 판단하게 해 주는 additional measurement와 unresolved로 report하게 하는 result를 말하라.
5. bits-per-byte가135M과7B run의 ablation을 nat-per-token으로는 할 수 없는 방식으로 comparable하게 만드는 이유와, comparison이 valid하려면 evaluation text가 가져야 하는 property를 설명하라.
6. Tülu3는 evaluation suite average에서2 epoch를 select한 반면 NEFTune α value는 AlpacaEval에서 select했다. §4의 two axis 관점에서 두 selection procedure의 risk가 다른 이유와 NEFTune paper 자체가 report한 number가 license하는 것과 license하지 않는 것을 설명하라.
7. reader가 이 lab을 reproduce하고 자기 prompt format에서 target gain7 point를 report한다. second reader가 그 gain이 model property인지 prompt property인지 결정하게 해 주는 smallest additional number set은 무엇인가?

---

## 연결

- **ch-07 — Training Failure Modes: Numerical, Masking, and Capability-Level Failures**(previous chapter이자 이 chapter의 dependency): failure taxonomy를 제공하며 이 lab은 masking과 capability-level entry를 executable test로 구현한다.
- **ch-04 — Sequence Packing, Loss Masking, and Chat Templates**: §2와 §3이 source에서 verify하는 mechanism.
- **ch-05 — Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen**: recipe table이 per-device batch, accumulation, global batch를 분리하는 이유.
- **ch-06 — Checkpointing, In-Loop Evaluation, and Checkpoint Selection**: 이 lab이 nanochat에서 채택하는 in-loop measurement cadence와 epoch ablation 뒤 checkpoint-selection rule.
- **ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability**(next chapter): run 하나를 측정하는 데서 compute budget이 무엇을 사는지 predict하는 쪽으로 이동.
- **ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate**: training-side template diversification을 포함해 이 lab이 ablate하는 lever의 full treatment.
- **ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split**: 이 lab의 production-scale version.
- **ch-47 — Evaluation Harness and Suite Design for General Capability**: 이 lab이 suite 하나와 format 두 개로 provisional하게 답하는 harness question.

---

## 출처

- [[trl-sft-trainer]] — chapter excerpt: commit `aa89588`의 TRL `SFTTrainer`와 `SFTConfig`; §2, §3, recipe table에서 쓴 every symbol, default, guard.
- [[hf-trainer-loop]] — chapter excerpt: clipping, `optimizer.step()`, `lr_scheduler.step()`, `max_steps` computation의 `transformers` v4.57.1 locus.
- [[nanochat]] — chapter excerpt: §2에서 읽은 second trainer(ignore index `-1`, pad-not-crop packing) 및 §6의 costed8×H100 reference run.
- [[loss-masking-prompt]] — Shi et al.2024의 chapter excerpt: Table1 target-versus-general number, instruction tuning tax, response-only 대 instruction-modelling lever.
- [[packing-with-flash-attention]] — Kundu et al.2024의 chapter excerpt: §3.1과 §3.3에 쓴 Table2 packed-versus-padded validation loss 및 throughput.
- [[prompt-format-sensitivity-formatspread]] — Sclar et al.2023의 chapter excerpt: two-format requirement의 바탕인 format spread number.
- [[hf-alignment-handbook]] — chapter excerpt: repository에 작성된 Zephyr SFT configuration 및 library card snippet의 correction.
- [[paloma]] — domain별 held-out loss, bits-per-byte definition, decontamination, separate-document scoring variance.
- [[neftune]] — NEFTune lever, α selection procedure, broad benchmark에 보고된 effect.
- [[signal-and-noise-eval]] — benchmark difference noise floor 및 decision metric으로서 bits-per-byte의 근거.
- [[tulu-3]] — 8B와70B에서 SFT epoch, learning rate, loss reduction, batch, compute.
- [[fsdp-sft]] — §6에서 인용한 mixed-precision memory contract.
- [[trl-grpo]] — 이 repository의 symbol과 default가 commit 사이에서 움직인다는 verified evidence이며, §2가 commit 하나를 pin하는 이유.
