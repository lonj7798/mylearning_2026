# Chapter 36 — Lab: Masking Test, Forgetting Report, Held-Out Evaluation Split을 포함한 SFT Run

> **핵심 통찰.** SFT run은 target score를 올리면서 train하지 않은 ability를 낮출 수 있다. LLaMA13B에서 Code-Alpaca set20,022 example은 Codex-Eval(HumanEval pass@10)을28.6에서34.2로 올렸으나 BBH는39.3에서35.6, TyDiQA는43.2에서38.9로 낮췄다. 더 큰 seven-dataset mixture(490,694 example; dataset Table1 count의 sum이며 size-match되지 않음)는 six evaluation 모두에서 Code-Alpaca보다 높았다([[tulu-1-how-far-can-camels-go]] Table1,3). development suite gain도 transfer하지 못할 수 있다. math, coding, IFEval constraint type으로 만든 Precise IF set인 Tülu3 Persona dataset을 제거하면 IFEval은72.8에서53.6으로 낮아졌지만 IFEval-OOD는17.6에서18.0으로 갔다([[tulu-3-lab-evidence]], Table32). 이 lab은 loss가 무엇에 대해 compute되는지 unit test로 training을 gate하고, mixture가 target하지 않는 suite 및 어떤 decision에도 쓰지 않는 held-out suite에서 paired base-versus-SFT difference로 각 run을 판단하며 every interval에 seed variance를 포함한다.
>
> **지침.** SFT run을 base model 또는 다른 run과 비교할 때에는 first optimizer step 전에 template, label, packing unit test를 run한다. `{% generation %}` block 없는 chat template은 `apply_chat_template`이 all-zero assistant mask를 return하게 만들고 warning만 log하기 때문이다([[hf-chat-template-assistant-mask]], transformers v4.46.0 L1805-1808). data, epoch, LoRA setting을 고를 때 development suite를 정하고 skill별 다른 benchmark를 가진 held-out suite를 decision을 written down할 때까지 unexamined로 둔다. Tülu3 instruction-following data choice는 IFEval에 overfit했기 때문이다([[tulu-3-lab-evidence]], §7.4.1). difference를 report할 때 item과 적어도 SFT seed3개에 대해 paired95% interval을 compute한다. Tülu3 70B SFT seed average는70.0에서72.6 범위였고(same source, Table14), 60% 근처의 IFEval prompt541개 score는 [[benchmark-variance-quantified]] §3.1 formula로 derive한 item-level half-width4.1 point를 갖기 때문이다. compute가1B model만 허용하면 data-arm axis, epoch axis, full forgetting report는 유지하고 loss와 LoRA axis는 drop한다. §3, §4.1 study에서 data composition과 training duration이 data가 target하지 않은 ability score를 바꿨기 때문이다([[tulu-1-how-far-can-camels-go]] Table3; [[lora-learns-less-forgets-less]] TableS6).

## 이 chapter가 general-purpose model에 중요한 이유

general-purpose model의 training pipeline은 pre-training → mid-training → SFT → preference optimization → RL → evaluation으로 진행된다. 이 lab은 SFT phase의 last chapter다. 이어지는 preference 및 RL phase([[ch-15]]부터)는 이 lab이 만드는 종류의 SFT checkpoint에서 시작하며, evaluation lab [[ch-53]]는 이 lab의 run과 generation을 reuse하고, forgetting report는 [[ch-30a]] §6이 정의한 format을 쓴다.

lab은 측정 가능한 problem 세 가지를 다룬다.

1. **무엇을 train하는지의 error.** wrong template, label mask, packed attention pattern은 exception을 일으키지 않으면서 decreasing loss curve를 만든다. 공개 pipeline에도 이런 error가 있다. OLMo3 7B Think DPO와 RL은 “a minor miscommunication” 때문에 SFT template과 “slightly different”한 chat template을 사용했고(L22), open-instruct tokenization code는 first `<think>` token을 “incorrectly masks … as part of the prompt”한다. Think SFT tokenization은 `<think>` 없는 template을 사용해 이를 피한다(L28,L39)([[open-instruct-allenai-recipes]], commit098424c의 `docs/olmo3.md`).
2. **Forgetting.** instruction dataset 하나의 SFT는 data 밖 ability score를 낮출 수 있다. [[loss-masking-prompt]] Table1에서 instruction set7개 각각의 response-only LLaMA-2-7B run은 base model18-task mean49.32보다 낮았다(range45.29-48.79).
3. **Noise에 대한 selection.** small 또는 reused evaluation set에서 한 choice는 ability change가 아니라 item sampling, seed variance, 해당 benchmark format에 대한 overfitting을 reflect할 수 있다.

[[ch-30]]은 SFT design axis를 정의했고 [[ch-30a]]는 forgetting measurement를 정의했다. 이 lab은 controlled configuration set에서 둘 다 run하고 claim에 interval이 붙은 decision memo를 만든다.

## §1 Lab question, hypothesis, deliverable, budget path

**Question.** verified masking을 쓴 SFT run이 held-out split의 forgetting report를 pass하면서 target을 improve하는가?

**Run 전 작성하는 hypothesis.** 각 row는 source가 측정한 direction, 그 measurement setting, 그리고 이 lab에서 이를 contradict할 observation을 밝힌다.

| Axis | Predicted direction | Source와 setting | Contradicting observation |
|---|---|---|---|
| Narrow single-domain data 대 diverse mix(equal size) | narrow: target score는 higher 또는 equal, non-target score는 lower | [[tulu-1-how-far-can-camels-go]] Table3(LLaMA13B, size-match되지 않음); [[lora-learns-less-forgets-less]] §4.2(Llama-2-7B code IFT) | seed3개에서 forgetting suite의 narrow arm paired interval이 zero를 포함 |
| Epoch1 →2 →3 | epoch가 늘수록 non-target score decrease | [[lora-learns-less-forgets-less]] TableS6(full fine-tuning, code, epoch1/2/4에서0.595,0.579,0.512) | epoch-3 forgetting interval이 epoch1과 overlap |
| Instruction modelling(IM) 대 response-only(IT), example1,030개 | 두 subset에서 IM ≥ IT on non-target mean; 어느 subset이 이 suite에서 더 gain하는지는 source prediction 없음 | [[loss-masking-prompt]] Table1(LLaMA-2-7B,18-task mean, dataset7개에서 IM − IT +0.49부터+2.46; seed 보고 없음); Fig.2의 short-output data larger gap은 AlpacaEval1.0에서 측정 | 어느 subset에서도 non-target mean의 IM − IT interval이 zero 아래 |
| LoRA 대 full fine-tuning | narrow arm: LoRA가 덜 forget; diverse arm2 epoch: LoRA advantage 없음 | [[lora-learns-less-forgets-less]] TableS6,S13 | narrow arm에서 LoRA가 더 forget |

**LoRA**(low-rank adaptation)는 pretrained weight를 freeze하고 targeted weight matrix마다 rank r의 matrix pair를 train한다. α/r로 scale한 product를 그 matrix에 add한다. [[lora-learns-less-forgets-less]]의 **forgetting average**는 HellaSwag, WinoGrande, ARC-Challenge mean accuracy다(§3.3). lower value는 more forgetting을 뜻한다. 그 table은 base model value를 print하지 않으므로 number는 base에서 drop이 아니라 training duration과 method 사이 change를 보인다.

**Deliverable.**

1. `tests/`: §2 test와 corresponding bug를 inject하면 fail하는 negative control을 각각 둔다.
2. `runs.jsonl`: run당 row 하나로 base checkpoint, data file hash, arm, epoch, loss type, LoRA setting, seed, code commit, step-0 loss를 기록한다.
3. `eval/items/<run>/<suite>.jsonl`: base model과 every checkpoint의 item별 score.
4. `decontamination.md`: every arm이 every suite와 overlap한 count(§3).
5. `sft-lab-memo.md`: prediction과 non-inferiority margin(suite별 largest acceptable drop, §6.4; training 전에 commit), development result, decision list(held-out evaluation 전에 commit), held-out result, [[ch-30a]] §6 format의 run별 forgetting report.

**Budget path.**

| Item | Full path | Resource-constrained path |
|---|---|---|
| Base model | Llama-3.1-Tulu-3-8B-SFT의 base인 Llama-3.1-8B | OLMo-2-0425-1B-SFT에 사용한 OLMo2 1B base |
| Arm당 example | 20,000 | 10,000 |
| Axis | data arm; epoch1/2/3; IM 대 IT; LoRA 대 full | data arm × epoch{1,3} |
| Configuration당 seed | 3 | 3 |
| Unit test, decontamination, forgetting report, held-out split | required | required |

**Compute arithmetic(derived).** Tülu3는 final8B SFT model을 prompt939,344개에서2 epoch, GPU32개,6시간으로 train했고192 GPU-hour다([[tulu-3-lab-evidence]], §4.3). subsample이 example당 same mean token과 throughput을 가진다는 assumption 아래, example20,000개2-epoch run은192 × 20,000/939,344 = 4.09 GPU-hour다. §4.4의 full-path matrix는 같은 assumption에서 training total76 GPU-hour이며 evaluation time은 estimate하지 않는다. 1B path의 verified reference run(OLMo-2-0425-1B-SFT, `tulu-3-sft-olmo-2-mixture-0225` 2 epoch)은 H1008개에서 “approximately”9시간 걸렸다([[open-instruct-allenai-recipes]], `docs/olmo2.md` L10-13). 해당 document는 mixture example count를 주지 않으므로 per-example rate를 derive하지 않는다.

## §2 Gate1: first optimizer step 전 unit test

**정의.** gate는 rendered token id, label vector, packed attention pattern을 independent computation과 대조하는 test suite다. training commit에서 pass한 뒤에만 training을 시작한다.

**문제.** template choice만으로 Tülu3의 intermediate SFT average는 Llama3.0에서 template5개에 걸쳐51.6에서53.0으로 움직였다([[tulu-3-lab-evidence]], Table13; template당 run 수는 보고되지 않음). template 또는 mask error는 어느 token id가 turn boundary를 mark하고 어느 position을 train하는지 바꾸지만 training loop는 어느 변화도 report하지 않는다.

### 2.1 Chat-template rendering과 assistant mask

**메커니즘.** transformers v4.46.0에서 `apply_chat_template(..., return_assistant_tokens_mask=True, return_dict=True)`는 every mask를 zero로 시작하고 `{% generation %}` block에서 record한 character span 내부 position만1로 set한다. template에 그런 block이 없으면 code는 “return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword.”를 log하고 continue한다([[hf-chat-template-assistant-mask]], L1805-1808,L1895-1910). 2026-09-15에 fetch한 `allenai/OLMo-2-0425-1B-Instruct` template에는 그런 block이 없다. non-final assistant turn은 `'<|assistant|>\n' + content + eos_token + '\n'`으로 render하고 final turn은 trailing newline 없이 render한다.

아래 test는 prefix를 render해 assistant span을 찾으므로 `{% generation %}`에 depend하지 않는다(course code이며 source에서 가져온 것이 아님).

```python
# tests/test_chat_template.py
import pytest
from transformers import AutoTokenizer

CONVS = [
    [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
    [{"role": "system", "content": "Answer briefly."},
     {"role": "user", "content": "Capital of France?"}, {"role": "assistant", "content": "Paris."},
     {"role": "user", "content": "And of Italy?"}, {"role": "assistant", "content": "Rome."}],
]

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("path/to/training-tokenizer")

def assistant_spans(tok, messages):
    full = tok.apply_chat_template(messages, tokenize=True)
    spans = []
    for k, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(messages[:k], tokenize=True, add_generation_prompt=True)
        upto = tok.apply_chat_template(messages[:k + 1], tokenize=True)
        assert full[:len(prefix)] == prefix and full[:len(upto)] == upto, "rendering is not prefix-stable"
        spans.append((len(prefix), len(upto)))          # content + eos, excluding template text after eos
    return full, spans

@pytest.mark.parametrize("messages", CONVS)
def test_span_is_content_plus_eos(tok, messages):
    full, spans = assistant_spans(tok, messages)
    for (s, e), m in zip(spans, [m for m in messages if m["role"] == "assistant"]):
        assert tok.decode(full[s:e]).strip() == m["content"] + tok.eos_token

@pytest.mark.parametrize("messages", CONVS)
def test_training_ids_equal_serving_ids(tok, messages):
    from serving import render_ids                      # renderer of the inference stack
    assert tok.apply_chat_template(messages, tokenize=True) == render_ids(messages)

def test_hf_mask_not_empty_if_used(tok):                # needed only when training reads assistant_masks
    out = tok.apply_chat_template(CONVS[0], tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    assert sum(out["assistant_masks"]) > 0
```

**Negative control.** (1) serving renderer를 `eos_token` 뒤 trailing newline을 drop하는 renderer로 replace한다. `test_training_ids_equal_serving_ids`는 fail해야 한다. (2) assistant turn 하나 뒤 `eos_token`을 omit하는 training template을 쓴다. `test_span_is_content_plus_eos`는 fail해야 한다.

### 2.2 Response-only loss와 instruction modelling의 label position

**정의.** Shi et al.은 instruction tuning(IT)을 completion token만의 loss, instruction modelling(IM)을 template token을 제외한 instruction과 completion token의 loss로 정의한다([[loss-masking-prompt]] §3).

```
IT:  L = − Σ_{j=1..n} log P(C_j | I_1..I_m, C_1..C_{j−1})                    (Eq. 2)
IM:  L = − Σ_{t=1..m+n} log P(x_t | x_1..x_{t−1}) · 1(x_t ∉ T)               (Eq. 4)
```

I는 m token의 instruction, C는 n token의 completion, x는 그 concatenation, T는 `<|user|>` 같은 template token set이고 1(·)은 non-template token이면1이다.

**Test.** (a) IT 아래 −100 이외 label을 갖는 position set은 every assistant turn에 대한 §2.1 span union과 같다. (b) IM 아래 그 set은 IT set의 superset이며 role header 안 position을 포함하지 않는다. (c) batch에서 training loop가 log한 trained token count는 same row에서(a) 또는(b)로 compute한 count와 같다.

**Worked example.** Shi et al. Table5는 average length를 준다(unit stated 안 됨). LIMA(total484.47, output442.75)에서 IT는 position442.75/484.47 = 91.4%를 train하고 IM은 최대100%다. Less MMLU Chat(total225.19, output8.24)에서 IT는3.7%, IM은 최대225.19/8.24 = 27.3배 많은 position을 train한다. 따라서 loss로의 prompt leak은 두 번째 dataset에서 logged trained-token count를 최대27.3배 multiply하지만 첫 번째에서는 최대484.47/442.75 = 1.094다. test(c)는 exact count를 compare하므로 둘 다 detect한다. logged count visual check는27.3배 change는 보지만9.4% change는 보지 못한다.

### 2.3 Packed row: boundary label과 cross-example attention

**정의.** packed row는 example 여러 개를 concatenate한다. correct packing은 example마다 `position_ids`를0에서 restart하고 attention을 그 example token으로 restrict하며 example 하나의 last token으로 다음 example first token을 predict하도록 train하지 않는다.

**메커니즘**([[packing-position-ids]], Kundu et al. §3.3-3.5).

1. collator는 mini-batch example을 `input_ids` row 하나로 concatenate한다.
2. example마다 first label을 −100으로 convert한다.
3. example별 `position_ids`를 concatenate한다.
4. `attention_mask=None`이면 attention code는 `position_ids`에서 `cu_seq_len`을 compute하고 `flash_attn_varlen_func()`를 call한다.

**Worked example**(Kundu et al. §3.5). length4,8,5,11의 example 네 개는 `position_ids` [0,1,2,3, 0,…,7, 0,…,4, 0,…,10]을 준다. value가0인 position은0,4,12,17이므로 `cu_seq_len` = [0,4,12,17,28]이다. label28개 중4개가 −100이므로 prompt masking 전에는 trained position24개가 남는다.

```python
# tests/test_packing.py  (course code, not taken from a source)
import pytest, torch
from sft.packing import pack_examples                   # function under test

def cu_from_position_ids(pos):
    return (pos == 0).nonzero().flatten().tolist() + [len(pos)]

def test_boundaries_and_first_labels():
    exs = [list(range(10, 14)), list(range(20, 28)), list(range(30, 35)), list(range(40, 51))]
    ids, labels, pos = pack_examples(exs, prompt_lens=[0, 0, 0, 0])
    assert cu_from_position_ids(pos) == [0, 4, 12, 17, 28]
    assert all(labels[s] == -100 for s in (0, 4, 12, 17))
    assert int((labels != -100).sum()) == 24

@pytest.mark.skipif(not torch.cuda.is_available(), reason="uses the training attention kernel")
def test_packed_logprobs_equal_separate(model, example_tensors, tol):
    ids = torch.cat(example_tensors)[None].cuda()
    pos = torch.cat([torch.arange(len(e)) for e in example_tensors])[None].cuda()
    with torch.no_grad():
        packed = model(input_ids=ids, position_ids=pos).logits.float().log_softmax(-1)[0]
        start = 0
        for e in example_tensors:
            alone = model(input_ids=e[None].cuda()).logits.float().log_softmax(-1)[0]
            assert (packed[start:start + len(e)] - alone).abs().max().item() < tol
            start += len(e)
```

**Negative control.** `position_ids` 없이 packed row에서 model을 call한다. 그러면 second example log-probability는 separate pass와 `tol`보다 크게 differ해야 한다. **Tolerance.** calibration에서 `tol`을 정한다. example 하나의 separate pass를 two different batch composition 안에서 두 번 run하고 observed maximum difference의 multiple을 취한다. value를 record한다. source는 threshold를 주지 않는다.

### 2.4 Run-level check와 작동하지 않는 test 세 가지

1. **Step-0 loss.** separate evaluation script로 first training batch labeled token에서 base model mean negative log-likelihood를 compute한다. training loop step-0 loss는 memo에 기록한 tolerance 안에서 이것과 match해야 한다. match는 same weight, label, loss aggregation을 쓴다는 것을 confirm한다.
2. **ln|V|가 reference가 아닌 이유.** Karpathy check “verify loss @ init”은 “you initialize your final layer correctly”일 때 −log(1/n_classes)를 expect한다([[karpathy-training-neural-net-recipe]], section2). 이는 uniform prediction loss다. vocabulary100,000 token이면 ln100,000 = 11.51 nat(derived)다. pretrained model은 uniform보다 text를 잘 predict하므로 correct SFT run은 이 value below에서 시작한다. ln|V| 근처 step-0 loss는 weight가 load되지 않았거나 output layer가 re-initialized됐음을 뜻한다(Interpretation).
3. **Example 두 개 overfit.** loss가 더 decrease하지 않을 때까지 example 두 개로 train하고 labeled token loss가 zero 쪽으로 fall하며 greedy decoding이 completion 둘을 reproduce하는지 confirm한다([[karpathy-training-neural-net-recipe]], “overfit one batch”). 이는 label shift와 optimizer update를 함께 check한다.
4. **Model이 받는 batch decode.** label을 text로 render하고 −100 position을 blank로 보이는 packed row 하나를 print한다. same section의 “exactly what goes into your network” decode advice를 따른다.
5. **Test 아님: packed와 unpacked loss curve match.** correct position ID에서도20K FLAN subset의 Mistral-7B validation loss는 unpacked(19,961 row)1.129, offline packing(585 row)1.284였다. author는19,961 example을585 row로 pack하면 one epoch optimizer step이 적어서 gap이 난다고 본다(§4.1). row count를 유지한 mini-batch packing은1.127이었다([[packing-position-ids]], Table2). correct implementation도 curve가 differ하므로 §2.3 forward pass로 equivalence를 test한다.
6. **Test 아님: prompt leak의 gradient-norm threshold.** library의 어떤 source도 threshold를 주지 않고 norm spike에는 other cause가 있다. leak은 §2.2(a)-(c)로 directly test한다. pre-clipping norm은 diagnosis용으로만 log한다.
7. **Loss aggregation은 run 사이 fixed.** Tülu3는 mean loss에서 gradient accumulation이 each micro-batch에 equal weight를 주므로 token weight가 그 micro-batch를 share한 token 수에 depend함을 보였고(Eq.1-2), sum loss로 switch했다([[tulu-3-lab-evidence]], §4.3.2). packing도 same way로 micro-batch당 token을 바꾼다. lab은 all arm에 aggregation 하나를 사용하고 기록한다.

## §3 Data arm: narrow single-domain set과 equal-size diverse mix

**정의.** **narrow arm**은 skill domain 하나의 example을 담는다. **diverse arm**은 general mixture의 every category를 그 size에 proportionate하게 sample한다. 두 arm은 same number의 example을 갖는다.

**문제.** 여기서 쓰는 published comparison은 size를 match하지 않는다. [[tulu-1-how-far-can-camels-go]] Table3에서 code set은 example20,022개, mixture는490,694개(Table1의 dataset7개 count sum)여서 mixture advantage에는 composition과 quantity가 섞여 있다. lab은 quantity difference를 제거한다.

**Direction에 대한 근거.**

- LLaMA13B, Code-Alpaca 대 base: Codex-Eval28.6 →34.2, MMLU42.3 →42.5, GSM14.5 →13.5, BBH39.3 →35.6, TyDiQA43.2 →38.9. Human+GPT mixture는 MMLU,GSM,BBH,TyDiQA에서49.3,40.5,43.3,45.6, Codex-Eval에서35.9였다([[tulu-1-how-far-can-camels-go]] Table3). GSM은 test item1,319개 중200개를 사용하므로(App.E),14.5%에서 derived analytic half-width4.9 point 안에 −1.0 GSM change가 있다. Result(single study).
- example1,030-13,533개의 single source에서 response-only SFT한 LLaMA-2-7B: base49.32 대비18-task mean45.29-48.79. Alpagasus Alpaca5k는 Commonsense Reasoning을75.86에서66.06으로 낮췄다([[loss-masking-prompt]] Table1). Result(single study).
- Llama-2-7B full fine-tuning: “programming induces more forgetting than math”. epoch2 뒤 forgetting average는 Magicoder-Evol-Instruct-110K에서0.579, MetaMathQA에서0.599였고 code set에서는 epoch16에0.414로 낮아졌다([[lora-learns-less-forgets-less]] §4.2, TableS6,S8). Result(single study). Tülu1은 BBH와 TyDiQA에서 base model 대비 drop을 measure하고 Biderman et al.은 printed base value 없이 training duration에 따른 decrease를 measure한다. 둘 다 다른 setting과 metric에서 code instruction data 뒤 lower non-code score를 가리킨다.
- Tülu3 8B: math data를 remove하면 MATH31.5 →23.5, unseen DeepMind Mathematics32.3 →23.3이므로 skill-specific data가 unseen benchmark로 transfer했다([[tulu-3-lab-evidence]], Table32).

**Construction.**

1. **Diverse arm(full path).** Table7 category에 stratify하여 Tülu3 SFT mixture에서 example20,000개를 sample한다([[tulu-3-sft-mix]]; category sum은 [[tulu-3-lab-evidence]]). resulting count는 아래 table에 있다.
2. **Narrow arm.** Tülu3 Persona Python에서 example20,000개를 sample한다(example34,999개; persona로 condition한 GPT-4o problem과 claude-3-5-sonnet program, [[tulu-3-lab-evidence]] §3.1.2). learner의 [[ch-29]] pool을 category 하나로 restrict한 것이 row20,000개일 때 alternative다.
3. **Token accounting.** arm별 trained token과 optimizer step을 record한다. arm별 trained token이 differ하면 둘 다 report하고 token-matched diverse run 하나를 add한다.
4. **Decontamination.** every development, held-out, forgetting suite에 대해 Tülu3 rule을 both arm에 apply한다(worked example 참고).
5. **Representation label.** §5의 suite별로 arm이 해당 skill data를 contain하는지 record한다.

**Worked example: stratified count**(Table7 category sum에서 derived, largest-remainder rounding).

| Category | SFT prompt | Share | 20,000-example diverse arm의 example |
|---|---|---|---|
| Math reasoning | 334,252 | 35.58% | 7,117 |
| Coding | 142,275 | 15.15% | 3,029 |
| General | 116,872 | 12.44% | 2,489 |
| Safety 및 non-compliance | 110,983 | 11.81% | 2,363 |
| Knowledge recall | 104,982 | 11.18% | 2,235 |
| Multilingual | 100,000 | 10.65% | 2,129 |
| Precise IF | 29,980 | 3.19% | 638 |
| Total | 939,344 | 100% | 20,000 |

**Worked example: decontamination rule**([[tulu-3-lab-evidence]], §3.2). GSM8K test question 하나에 token40개가 있다고 하자. 그중22개가 training example 하나와 share하는8-gram 내부에 있으면22/40 = 55% >50%이므로 question은 그 example과 overlap한다. GSM8K test question은1,319개이며1,319의2%는26.38이므로 question27개 이상과 overlap하는 training source는 GSM8K에 contaminated다. held-out suite에서는 source를 remove한다. development suite에서는 performance를 significantly affect하지 않으면 whole source를 remove하고, 그렇지 않으면 matching example만 remove한다.

**General-purpose model에 대한 함의.** narrow arm은 fixed size에서 single-domain synthetic set이 its domain 밖에서 무엇을 cost하는지 measure한다. diverse arm은 general-purpose recipe가 사용할 control이다.

## §4 Generality를 바꾸는 training axis

### 4.1 Epoch1,2,3

**문제.** [[lora-learns-less-forgets-less]]에서 forgetting은 training duration과 함께 증가했다(§4.2). code의 full fine-tuning forgetting average는 epoch1,2,4,8,16에서0.595,0.579,0.512,0.446,0.414였고 HumanEval은 epoch8에서0.497로 peak했다(TableS5-S6). diverse Tülu-v2-mix에서 full fine-tuning forgetting average는 epoch2,4,6에서0.660,0.652,0.621이었다(TableS13). Tülu3는 Tülu2 mixture에서2 epoch보다 길게 train해도 average가 improve하지 않는다고 찾았다(§4.3.2, numeric label 없는 curve).

**메커니즘: one run의 checkpoint가 아닌 separate run.** 위 code IFT run은 “separate models for1,2,4,8,and16 epochs”였다(§4.1). Tülu-v2-mix number는 duration별 cooldown 없이6-epoch run 하나의 “hot” checkpoint에서 왔다(App.C). 두 design은 equivalent하지 않다.

**Worked example(derived).** Tülu3 schedule은3% warmup 뒤 linear decay다([[tulu-3-lab-evidence]], Table11). T step의3-epoch run에서 stepT/3 learning rate는(1 − 1/3)/(1 − 0.03) = peak의0.687, peak5×10⁻⁶이면3.44×10⁻⁶이다. same schedule의1-epoch run은0에서 end한다. 그러므로3-epoch run epoch-1 checkpoint는 learning rate가 아직 peak69%일 때 take한다. lab은 epoch1,2,3을 separate run으로 train한다.

### 4.2 Example1,030 subset에서 instruction modelling 대 response-only loss

**근거.** LIMA(example1,030; instruction/output length ratio0.0942)에서 IM은18-task mean48.79를49.60으로, MT-Bench4.77을4.83으로 바꿨고 AlpacaEval1.0은33.06에서32.94로 바꿨다([[loss-masking-prompt]] Table1,5). AlpacaEval1.0에서 IM advantage는 Less MMLU Chat(ratio26.3;4.42 →9.78)처럼 long instruction과 short output dataset에서 더 컸고, ratio10 근처 TüluV2 subset이1,000 example 쪽으로 shrink할수록 증가했다(Fig.2; 두 panel 모두 AlpacaEval1.0 improvement plot). 18-task mean에는 Table1이 그런 ordering을 보이지 않는다. LIMA의 IM − IT는+0.81, Less MMLU Chat은+0.66이었고 largest gain 세 개(+2.18부터+2.46)는 ratio0.30-0.64의 Alpagasus set에서 나왔다(Table1,5). mechanism evidence로 LIMA에서 IM은 output training loss가 더 높았고(1.45 대1.37) TüluV2 test loss는 더 낮았다(1.17 대1.32)(Fig.3). seed나 interval은 report되지 않았다. Result(single study).

**Design.** example1,030개 subset 둘을 쓴다. long output의 subsetL(LIMA 또는 instruction/output ratio0.1 below인 diverse-arm example)과 short output subsetS(ratio at least10, Fig.2 right의 fixed ratio)다. 각각 IT와 IM으로2 epoch, seed3개 train한다. lab non-target suite에서 source는 두 subset 모두 IM ≥ IT를 support한다. 이 suite에서 subsetS difference가 더 큰지는 Open question이다. source의 ratio dependence는 lab이 run하지 않는 AlpacaEval1.0에서 측정됐다. loss type만 바꾸며 LR, batch, epoch은 model size의 lab row를 따른다.

**한계.** Shi et al.은 “we are not proposing IM as a replacement for current fine-tuning processes”라고 썼다(Abstract). axis는 small set에서 overfitting mechanism을 test하며20,000-example arm의 loss를 choose하지 않는다.

### 4.3 LoRA 대 full fine-tuning

**근거.** narrow code data, Llama-2-7B, epoch4에서 full fine-tuning은 HumanEval0.470, forgetting average0.512였다. LoRA r = 256은0.498,0.631이고 LoRA r = 16은0.358,0.652였다([[lora-learns-less-forgets-less]] TableS5-S6). similar target accuracy(r = 256 대 epoch4 full fine-tuning)에서 LoRA는 HumanEval2.8 point, forgetting average11.9 point 더 높았다(derived). diverse Tülu-v2-mix epoch2에서 forgetting average는 full fine-tuning0.660, LoRA r = 16 0.650이었고, MT-Bench에서는 “all LoRA models are within one standard error” of full fine-tuning이었다. epoch6에는 full fine-tuning이 가장 많이 forgot했다(TableS10-S13, App.C.2). 각 condition은 한 번 train했다(§4.1). Result(single study).

**Lab setting.** App.A처럼 every block projection matrix7개(W_q,W_k,W_v,W_o,W_gate,W_up,W_down)에 LoRA를 적용하고 r = 256, α = 512를 쓴다. 이는 TableS5에서 HumanEval이 full fine-tuning에 도달한 유일한 rank다(epoch4에서0.498 대0.470). LoRA LR은 App.A가 r = 256과 r = 16/64에 사용한 value인 {1×10⁻⁴,2×10⁻⁴}에서 development suite로 choose한다. App.B는 LoRA best LR이 full fine-tuning보다 order of magnitude 하나 높음을 찾았다.

### 4.4 Run matrix(full path)

| Configuration | Arm | Epoch | Loss | Adapter | Seed | GPU-hour(derived, §1) |
|---|---|---|---|---|---|---|
| F1 baseline | diverse | 2 | IT | full | 3 | 12.3 |
| F2 | narrow | 2 | IT | full | 3 | 12.3 |
| F3,F4 | diverse | 1;3 | IT | full | 3 +3 | 6.1 +18.4 |
| F5,F6 | diverse;narrow | 2 | IT | LoRA r = 256 | 3 +3 | 12.3 +12.3 |
| F7 | subsetL과S(each1,030) | 2 | IT;IM | full | run12개 | 2.5 |

equal-throughput assumption 아래 total은 run30개와 training76 GPU-hour다. assumption은 F5,F6에서 가장 weak하다. [[lora-learns-less-forgets-less]]는 “for standard implementations and a fixed batch size, LoRA tends to train slower than full finetuning”이라고 report한다(§4.7, App.I). constrained path는 full fine-tuning과 IT에서 {narrow,diverse} × epoch{1,3}을 run한다. example10,000개에서 configuration4개, run12개다.

## §5 Evaluation split: development suite, held-out suite, forgetting suite

### 5.1 Development와 held-out suite가 separate benchmark인 이유

**정의.** **development suite**는 LoRA learning rate, run acceptance, each axis interpretation의 every choice에 사용한다. **held-out suite**는 skill마다 different benchmark를 담고 decision list를 commit한 뒤 한 번 evaluate한다. Tülu3는 이 design을 사용했고 “did not examine scores on our unseen set when developing our models”라고 했다([[tulu-3-lab-evidence]], §2.2, Table3).

**근거.** Tülu3 SFT ablation에서 Persona dataset이 없으면 IFEval은72.8에서53.6으로 drop했다(removal은 Persona math, coding, Precise IF set을 cover, §4.2 및 Table10). 그러나 IFEval-OOD는 drop하지 않았다(17.6 대18.0). author는 choice가 “overfit to the development evaluations in Precise Instruction Following”했다고 conclude한다(§7.4.1). Persona IF는 “25 different constraint types defined in IFEval”을 cover하도록 generate됐다(§3.1.2). 두 번째 Ai2 study에서는 model 다수가 IFEval25 constraint template에서80% above지만 GPT-4.1과 Claude3.7 Sonnet은 IFBench unseen constraint58개에서50% below다([[ifbench]] §1, Fig.1). 두 result는 IFEval score가25 type 밖 constraint following을 overstate한다는 데 agree한다. author group이 overlap하므로 independent replication이 아니라 각각 Result(single study)다.

benchmark item 하나를 random으로 development/held-out half로 split하면 item noise에 대한 selection은 remove하지만 benchmark format에 대한 overfitting은 remove하지 못한다. lab은 separate benchmark를 쓴다(Tülu3 design의 Interpretation).

### 5.2 Suite role

| Suite | Skill | Item | Role | Narrow(code) arm | Diverse arm |
|---|---|---|---|---|---|
| MMLU | knowledge | 14,042([[benchmark-variance-quantified]] Table1) | forgetting report; development | not represented | knowledge-recall category 있음 |
| ARC-Challenge | science reasoning | 1,165(same table에서 evaluate) | forgetting report | not represented | dedicated category 없음 |
| GSM8K | grade-school math | 1,319(same table) | forgetting report; development | not represented | represented(math reasoning) |
| TriviaQA | closed-book factual recall | 11,313(same table) | forgetting report | not represented | dedicated category 없음 |
| IFEval | precise instruction following | prompt541개([[ifeval]] §1) | forgetting report; development | not represented | represented(Persona IF는25 type 사용) |
| HumanEval | code | 164(same table) | target; development | represented | represented |
| MMLU-Pro, AGIEval English, DeepMind Mathematics, BigCodeBench-Hard(1,140개 중148개), IFEval-OOD(constraint52개) 또는 IFBench(constraint58개,prompt300개) | skill당 하나 | listed와 같음 | held-out | — | — |

“not represented”라 mark한 suite decrease는 forgetting이다. represented suite decrease는 target에 reach하지 못한 것이다. diverse arm에서는 GSM8K와 IFEval이 target을 measure하며 held-out partner DeepMind Mathematics와 IFEval-OOD 또는 IFBench는 target이 transfer하는지 measure한다.

### 5.3 Protocol

1. suite마다 prompt template, shot count, decoding, answer extraction을 fix하고 base model과 every checkpoint에 사용한다. Tülu3처럼 IFEval prompt-level loose accuracy(§7.2)와 prompt-level strict accuracy를 report한다. loose criterion은 first line을 remove한 뒤 word count에 fail하는 response를 accept할 수 있기 때문이다([[ifeval]] §2.2).
2. base model이 chat task로 follow하지 못하는 suite에서는 base와 SFT checkpoint에 completion-style formulation 하나를 쓴다. Tülu1은 vanilla LLaMA가 “little instruction-following ability without further prompt engineering”이어서 AlpacaEval을 evaluate하지 않았다([[tulu-1-how-far-can-camels-go]] §4.2). base-versus-SFT IFEval difference에는 format과 ability가 섞이므로 IFEval entry는 base 대 SFT가 아니라 same chat format의 arm(narrow 대 diverse)을 compare한다(Interpretation). diverse arm에는 IFEval constraint type으로 만든 Precise IF example638개가 있으므로(§3), 이 entry는 forgetting이 아니라 arm difference로 label하여 report한다.
3. base model이 standard multiple-choice MMLU에서 chance 근처 score하면 cloze formulation을 쓴다. [[benchmark-variance-quantified]]의7B seed model에서는 standard MMLU가 monotonicity0.09에서25.86에 머문 반면 MMLU-Cloze는 monotonicity0.95에서37.47에 도달했다(Table1, §3.3). monotonicity는 discrete metric이면 increasing sequence, continuous metric이면 decreasing sequence와 seed model training checkpoint score 사이 Kendall rank correlation이다(§3.1).
4. item별 score를 store한다. §6의 every interval은 이로부터 compute한다.
5. held-out suite를 run하기 전에 decision list를 commit하고 every configuration의 held-out result를 report한다.

## §6 Statistics: paired interval, bootstrap, seed

### 6.1 Comparison 하나의 item-level interval

**Formula.** N item에서 accuracyS인0/1 score의 analytic95% half-width는1.96 · sqrt(S(1 − S)/N)다([[benchmark-variance-quantified]] §3.1). same item에서 score한 checkpoint A,B에 대해서는([[adding-error-bars-evals]] §4.1-4.2):

```
SE_unpaired = sqrt(SE_A² + SE_B²)
SE_paired   = sqrt( (1/(n−1)) Σ_i (d_i − d̄)² / n )          (Eq. 7)
CI_95%      = d̄ ± 1.96 × SE                                  (Eq. 5)
```

SE_A,SE_B는 checkpoint별 mean score standard error(0/1 score라면 sqrt(S(1 − S)/n))이고 d_i는 item i에서 checkpointA score − checkpointB score, d̄는 n item의 d_i mean, SE는 SE_unpaired 또는 SE_paired다.

**Worked example: suite size.** accuracy60%에서 IFEval prompt541개는 half-width1.96 · sqrt(0.6 ×0.4/541) =4.1 point, ARC-Challenge item1,165개를40%에서 score하면2.8 point, HumanEval problem164개를30%에서 score하면7.0 point다(each problem을0/1로 score할 때; derived. sample에 average한 pass@k는0/1 score가 아니므로 formula는 approximation일 뿐이다). MT-Bench는 category8개 각각 question10개로 question80개다([[judge-llm-bias]] §2.2). Tülu-v2-mix에서 all LoRA run은 MT-Bench에서 full fine-tuning의 one standard error 안이었다([[lora-learns-less-forgets-less]] App.C.2). lab은10-question category slice로 decision하지 않는다.

**Worked example: paired forgetting delta.** base model이 GSM8K question1,319개 중660개(50.0%)를 answer한다. narrow-arm SFT 뒤 그중110개가 wrong이 되고 이전 wrong70개가 right가 되어 SFT checkpoint는620개(47.0%)를 score한다.

- d̄ = (70 −110)/1,319 = −0.0303.
- Σ d_i² =180이므로 Σ(d_i −d̄)² =180 −1,319 ×0.0303² =178.79, sample variance는178.79/1,318 =0.1357, SE_paired =sqrt(0.1357/1,319) =0.0101.
- Paired CI: −0.0303 ±0.0199 = [−0.050,−0.010]이며 zero를 exclude한다.
- SE_unpaired =sqrt(0.500 ×0.500/1,319 +0.470 ×0.530/1,319) =0.0195; CI [−0.068,+0.008]은 zero를 include한다.
- same net change가 loss220개, gain180개에서 왔다면 SE_paired =0.0151, paired CI는[−0.060,−0.001]이다.

[figures/forgetting-ci-planner.html](figures/forgetting-ci-planner.html)은 suite size, base score, loss/gain count, seed count, seed standard deviation, comparison 수를 고르고 zero에 대한 unpaired, paired, seed-inclusive, Bonferroni-adjusted interval을 보게 한다.

### 6.2 Paired bootstrap

**정의.** paired bootstrap은 replacement와 함께 item index를 resample하고 each resample에서 mean difference를 recompute하며 resampled mean의2.5th,97.5th percentile을95% interval로 취한다.1.96 multiplier 뒤 normal approximation을 쓰지 않으며 F1이나 judge rating 같은 non-binary score에도 apply된다.

```python
import numpy as np   # course code, not taken from a source

def paired_bootstrap_ci(ref, new, n_boot=10_000, alpha=0.05, seed=0, chunk=500):
    d = np.asarray(new, float) - np.asarray(ref, float)      # same items, same order
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(0, len(d), (min(chunk, n_boot - b), len(d)))].mean(1)
                            for b in range(0, n_boot, chunk)])  # chunks bound memory for 14,042-item suites
    return d.mean(), np.quantile(means, [alpha / 2, 1 - alpha / 2])
```

**Worked example(손으로 검산 가능).** item5개의 difference는 d = [1,0,0,−1,1]이므로 d̄ =0.2이고 Σ(d_i −d̄)² =0.64 +0.04 +0.04 +1.44 +0.64 =2.8이다. equiprobable resample5⁵ =3,125개 전체에서 resampled mean variance는(2.8/5)/5 =0.112, standard error0.335, percentile interval[−0.4,0.8]이다(enumerated). Eq.7은sqrt((2.8/4)/5) =0.374를 준다. bootstrap standard error는 factor sqrt((n −1)/n) =0.894만큼 smaller이며 small n에서만 중요하다.

### 6.3 Seed variance

**근거.** Tülu3 8B SFT seed average는59.9,60.1,59.8,59.8,59.8(standard deviation0.13, derived)이고70B average는71.8,70.0,72.6(standard deviation1.33, derived)였다. author는 variation을 “multiple training runs”의 reason이라 부른다([[tulu-3-lab-evidence]], Table14). different seed로 pretrain한7B model10개에서 seed standard deviation은 Table1 every benchmark의 item-level95% half-width보다 below였다. 예를 들어 ARC-C0.80 대2.74, MMLU0.57 대0.72다([[benchmark-variance-quantified]] Table1). 이는 pretraining seed이며 여기 어떤 source도 benchmark별 SFT seed variance를 measure하지 않는다. 각 measurement는 its stage의 Result(single study)다.

**Formula(course approximation, derived; source에서 온 것이 아님).** armA,B가 k_A,k_B seed, per-seed mean score와 sample standard deviation s_A,s_B를 갖고 seed-averaged per-item score에서 paired item-level standard error SE_items를 compute할 때:

```
SE_total ≈ sqrt( SE_items² + s_A²/k_A + s_B²/k_B )
```

base-versus-SFT comparison에서는 base term s_B가0이다. approximation은 fixed item set의 training randomness와 item sampling을 independent로 treat한다.

**Worked example(illustrative score).** narrow arm GSM8K seed별:47.0,45.8,48.1(mean46.97,s_A =1.15). diverse arm:49.5,50.2,48.9(mean49.53,s_B =0.65). difference는−2.57 point다. SE_items =1.0 point이면 seed term은1.15²/3 +0.65²/3 =0.58, SE_total =sqrt(1.00 +0.58) =1.26,95% interval은[−5.03,−0.10]이다. arm당 seed 하나면 seed term을 estimate할 수 없고 interval은 SE_items만 써[−4.53,−0.61]가 되며, 이는 data support보다 narrow하다.

**Two-level bootstrap.** arm 안 seed와 item을 jointly resample한다.

```python
def two_level_bootstrap_ci(A, B, n_boot=5_000, alpha=0.05, seed=0):
    """A: [seeds_A, items], B: [seeds_B, items]; B has one row for the base model."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        it = rng.integers(0, A.shape[1], A.shape[1])
        a = A[rng.integers(0, A.shape[0], A.shape[0])][:, it].mean()
        c = B[rng.integers(0, B.shape[0], B.shape[0])][:, it].mean()
        out[b] = a - c
    return A.mean() - B.mean(), np.quantile(out, [alpha / 2, 1 - alpha / 2])
```

**한계.** seed3개면 arm당 distinct seed multiset은10개이므로 bootstrap distribution의 seed part는 arm당 최대10 value만 취한다. every interval 옆에 per-seed score를 report한다.

### 6.4 Decision rule과 multiple comparison

1. training 전 forgetting suite마다 non-inferiority margin δ(largest acceptable drop)를, target suite마다 minimum gain of interest를 쓴다([[ch-30a]] §6, item12).
2. suite에서 SFT −base의 seed-inclusive95% interval lower end가 −δ above일 때 run은 forgetting report를 pass한다.
3. configuration difference interval이 correction 뒤 zero를 exclude하면 axis effect를 detected로 report한다.
4. **Bonferroni correction.** comparison m개이면 each를 α/m에서 test한다. m =20(forgetting suite5개 × axis4개)이면 α/m =0.0025이고 two-sided multiplier는1.96 대신3.02다(derived). worked seed example은−2.57 ±3.02 ×1.26 = [−6.37,+1.24]가 되어 zero를 include한다. [[ch-51]]은 interval-based go/no-go decision을 더 깊게 다룬다.
5. benchmark score가 lab scale에서 decide하기에 too noisy하면 comparison을 더하는 것보다 lower-noise formulation 또는 metric을 prefer한다.60M-1B의 pretraining ablation에서 question1,000개 ARC-Easy subset은 instance90% fewer로 MMLU보다 higher decision accuracy를 냈다([[signal-and-noise-eval]] App.B.2). Result(single study; SFT에서는 test되지 않음).

## §7 Acceptance criteria와 memo

run과 lab은 아래 모두를 satisfy할 때 accepted된다.

1. training commit에서 `pytest tests/`가 pass하고 §2의 each negative control은 corresponding bug를 inject했을 때 fail한다. output 둘 다 commit한다.
2. each run step-0 loss는 recorded tolerance 안에서 same labeled token의 independent base-model loss와 같다(§2.4 item1). example 두 개 overfit check는 completion 둘 다 reproduce한다.
3. `runs.jsonl`에는 data hash와 commit을 포함한 configuration별 seed3개가 있으며 evaluation 뒤 drop한 run이 없다.
4. `decontamination.md`는 every arm,suite의 Tülu3 overlap count와 취한 action을 report한다.
5. every run에는 MMLU,ARC-Challenge,GSM8K,TriviaQA,IFEval에서 paired,seed-inclusive95% interval을 갖는 [[ch-30a]] §6 format forgetting report가 있다. IFEval은 arm-versus-arm comparison으로 label한다(§5.3). each suite를 represented 또는 not represented로 label한다. safety pair,recovery test처럼 lab이 measure하지 않는 §6 format item은 not measured로 list한다.
6. prediction과 margin은 first training run 전에, decision list는 held-out evaluation 전에 commit했고 `git log`는 order를 보인다.
7. memo는 interval이 zero를 include한 configuration도 포함해 all configuration을 report하며 held-out score로 seed 하나를 select하지 않는다.

**Memo section.** (1) Setup: base model, example/token count가 있는 arm, run, hardware. (2) Development table: suite와 configuration별 seed mean, per-seed score, interval. (3) development table에서 내린 decision. (4) development-minus-held-out skill별 gap과 함께 한 번 compute한 held-out table. (5) run별 forgetting report. (6) §1 hypothesis를 supported, contradicted, undecided로 mark하고 각각을 결정한 interval을 적는다.

## 레시피

모든 row는 stated locus에서 읽었으며 date는 Status column에 있다.

| Model(정확한 release) | Size | Stage | Setting | Value | Source 위치 | Status | 이 value의 근거 | Lab value | 차이의 이유 |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak LR; schedule; warmup; effective batch; max length; epoch | 5×10⁻⁶;linear;0.03;128;4,096;2 | arXiv:2411.15124v5 §4.3, Table11([[tulu-3-lab-evidence]]) | verified 2026-09-15 | “found after a hyperparameter search”(§4.3); Llama3.0 + Tülu2 mix의 §4.3.2 Fig.5-6, numeric label 없음 | full path: same, epoch axis에서1/2/3 | epoch은 tested axis |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | loss aggregation | sum(paper); open-instruct8781471의 `--reduce_loss sum`;098424c에서 flag absent, mean default | §4.3.2; `docs/tulu3.md`@8781471 L49([[open-instruct-allenai-recipes]]) | conflict(paper와2025-06 command 대 current default) | §4.3.2 Fig.5-6 | every run에 sum | packing 또는 GPU count로 per-token weight가 변하면 안 됨(§2.4) |
| Llama-3.1-Tulu-3-8B-SFT;-70B-SFT | 8B;70B | SFT | seed; checkpoint selection | 42,123,456,789,1011(8B);42,123,456(70B); best single run release | Table14, §4.3.1 | verified 2026-09-15 | Table14 average59.8-60.1(8B),70.0-72.6(70B) | seed3; all report; selection 없음 | reported score로 selection하면 그 score에 upward bias(Interpretation) |
| Tülu3(all size) | — | eval-gate | decontamination | user turn8-gram; one training instance와 match한 test token >50%; evaluation instance 중 >2% overlap이면 source contaminated | §3.2 | verified 2026-09-15 | full-string,n-gram,embedding matching 비교 §3.2 | development,held-out,forgetting suite against same rule | none |
| OLMo-2-0425-1B-SFT | 1B | SFT | LR; effective batch; epoch; max length; warmup; data | 3×10⁻⁵;8 ×2 ×8 =128;2;4,096;0.03;`tulu-3-sft-olmo-2-mixture-0225` | open-instruct@098424c `docs/olmo2.md` L45-53([[open-instruct-allenai-recipes-recipe]]) | verified 2026-09-14 | ablation 보고 없음 | constrained path: same LR,batch,length,warmup; epoch1,3; arm당 example10,000개 | epoch은 tested axis; example count는 budget |
| OLMo-2-0425-1B-SFT | 1B | SFT | loss reduction |5e5c93d(L50)의 `--reduce_loss sum`;098424c에서는 absent | `docs/olmo2.md`@5e5c93d L50 | conflict | PR#1024가 default를 mean으로 변경 | sum | 위와 같음 |
| Llama-2-7B IT 및 IM run(Shi et al.) | 7B | SFT | LR; total batch; epoch; max length; optimizer; schedule; weight decay |2×10⁻⁵;128;2,3,10(“typically”2);2,048;AdamW(0.9,0.98),ε1e-6;linear,warmup0.03;0 | arXiv:2405.14394v2 App.C,Table6([[loss-masking-prompt]]) | verified 2026-09-15 | 해당 value ablation 보고 없음 | F7은 model size lab LR,batch,2 epoch | F7 안에서 loss type만 change |
| Llama-2-7B → Magicoder-Evol-Instruct-110K,LoRA | 7B | SFT | rank;α; LoRA LR; global batch;length;optimizer;schedule;epoch |r=16,64,256;α=2r;2×10⁻⁴(r=16,64),1×10⁻⁴(r=256);192;4,096;decoupled LionW(0.9,0.95);cosine,warmup0.1;1,2,4,8,16 separate run | arXiv:2405.09673v2 §4.1,App.A([[lora-learns-less-forgets-less]]) | verified 2026-09-15 | TableS5-S6; App.B2-epoch LR sweep |r=256,α=512, all module; development suite에서{1×10⁻⁴,2×10⁻⁴} LR;2 epoch | source와 base model,optimizer differ |
| Llama-2-7B → Magicoder; →MetaMathQA,full FT |7B|SFT|2-epoch sweep best full fine-tuning LR|5×10⁻⁵(code);1×10⁻⁵(math)|App.B|verified 2026-09-15|Fig.S1|사용 안 함|lab base model에는 Tülu3,OLMo2 value 사용|
| Llama-2-7B → Tülu-v2-mix |7B|SFT|LR;global batch;checkpoint|full5×10⁻⁶,LoRA1×10⁻⁴;192;duration별 cooldown 없는 one run에서 epoch2,4,6 evaluate|App.C,C.1|verified 2026-09-15|TableS10-S13|epoch count별 separate run|hot checkpoint는 nonzero LR에서 take(§4.1)|
| Tülu(LLaMA7B,13B) |7B;13B|SFT|epoch;LR;schedule;max length;loss mask|2;2×10⁻⁵;linear warmup3%,linear decay;2,048;assistant token만|arXiv:2306.04751v2 App.D,§3.2([[tulu-1-how-far-can-camels-go]])|verified 2026-09-14|ablation 보고 없음|setting에 사용 안 함|Table3 data comparison에 cite|

**작은 general-purpose run의 시작점.** Tülu3 SFT mixture stratified subsample로8B base model을 train한다면 peak LR5×10⁻⁶,3% warmup의 linear decay, effective batch128,max length4,096,2 epoch,sum loss를 쓴다. Tülu3는 Llama3.1 8B에서 prompt939,344개, GPU32개로 이 value를 썼으며 example20,000개에 re-tune하지 않았다. 1B base model에는 OLMo-2-0425-1B-SFT value를 쓴다. LR3×10⁻⁵,effective batch128,max length4,096,warmup0.03,2 epoch, H1008개에서 `tulu-3-sft-olmo-2-mixture-0225` data다. LoRA는 all module에서r=256,α=512,LR1×10⁻⁴이110K code example과 LionW의 Tülu-v2-mix에서 Llama-2-7B에 사용한 setting이다. AdamW에서는 LR이 verified되지 않아 development suite에서 choose한다.

## 일반화 관점

**(a) breadth를 늘리는 것.**

- Skill category를 mix: Tülu1 mixture는 size-match 없이13B의 every evaluation에서 code-only set보다 높았다([[tulu-1-how-far-can-camels-go]] Table3). lab equal-size arm은 그 result의 composition part를 test한다.
- gain이 unseen benchmark로 transfer하는 skill data: Tülu3 math data는 MATH를8.0 point, unseen DeepMind Mathematics를9.0 point 올렸다([[tulu-3-lab-evidence]], Table32).
- narrow data에서 update size limit: epoch4에 LoRA r=256은 full fine-tuning HumanEval0.470 대비0.498, forgetting average0.512 대비0.631에 도달했다([[lora-learns-less-forgets-less]] TableS5-S6).
- small instruction set의 instruction token loss: IM은 [[loss-masking-prompt]] Table1 dataset7개 모두에서 IT보다18-task mean을 올렸고(+0.49부터+2.46; seed 보고 없음), AlpacaEval1.0 advantage는 short-output data와 fewer example에서 더 컸다(Fig.2).

**(b) narrowing 또는 forgetting을 일으키는 것.**

- single-domain SFT data: code-only data가13B에서 BBH,TyDiQA를 낮췄고([[tulu-1-how-far-can-camels-go]]), [[loss-masking-prompt]] Table1의 every single-source IT run은 base18-task mean below를 score했다.
- more epoch: code full fine-tuning에서 forgetting average는 epoch와 함께 fall했고(separate run), Tülu-v2-mix에서는 epoch2에서6으로(0.660,0.652,0.621; one run checkpoint) fall했다([[lora-learns-less-forgets-less]] TableS6,S13).
- benchmark constraint list로 만든 training data: IFEval25 constraint type으로 만든 Precise IF set을 포함한 Persona dataset은 IFEval을19.2 point, IFEval-OOD를−0.4만큼 move했다([[tulu-3-lab-evidence]], Table32). model 다수는 IFEval80% above지만 GPT-4.1과 Claude3.7 Sonnet은 IFBench50% below다([[ifbench]] §1).
- metric 하나에서 more SFT data: Tülu3 stratified subsample에서 “TruthfulQA performance actually drops as the amount of data in the mix increases”(§4.2,Fig.4,numeric label 없음).

**(c) 이 stage에서 측정하는 방법.**

- suite별 paired base-versus-SFT difference에 represented/not represented label과 seed-inclusive interval을 붙인다(§6; [[adding-error-bars-evals]] §4.2).
- skill마다 different benchmark를 써 development-minus-held-out gap을 낸다([[tulu-3-lab-evidence]] Table3,32).
- interest effect에 충분히 큰 suite를 쓴다. item164개 code benchmark는30%에서7.0-point half-width이고 question80개 chat benchmark는 LoRA와 full fine-tuning을 separate하지 못했다([[lora-learns-less-forgets-less]] App.C.2).
- standard MMLU가 chance 근처일 때 cloze MMLU처럼 model scale에서 measurable signal이 있는 formulation을 쓴다(7B seed model은 token210B 뒤 standard MMLU25.86; [[benchmark-variance-quantified]] Table1,§3.3).

## 흔한 실수와 탐지 방법

| 실수 | 관측 가능한 증상 | 검사 |
|---|---|---|
| pretrained model step-0 loss를 ln\|V\|와 compare | every correct run이 initial-loss gate “fail” | same labeled token에서 independently computed base model loss와 compare(§2.4) |
| packed/unpacked training curve가 match해야 한다고 요구 | correct packing reject; step count에 따라 validation loss differ([[packing-position-ids]] Table2에서1.284 대1.129) | forward-pass log-probability equality test(§2.3) |
| gradient-norm threshold를 prompt-leakage test로 사용 | unrelated spike false alarm; spike 없는 leakage | trained-token count test(§2.2c) |
| `{% generation %}` 없는 template에 `return_assistant_tokens_mask=True` | log warning; `assistant_masks` all zero | `test_hf_mask_not_empty_if_used`; template keyword search |
| packed example each first label을 mask하지 않음 | example당 one씩 trained-token count 초과 | boundary label test(§2.3) |
| mix에 IFEval-constraint data가 있는데 IFEval을 held-out으로 취급 | IFEval gain이 IFEval-OOD 또는 IFBench gain 없이 발생 | suite-role table(§5.2); skill별 held-out partner |
| arm example 또는 token count differ | token match에서 rerun하면 narrow-vs-diverse difference change | arm별 trained token,step report(§3) |
| 3-epoch run epoch-1 checkpoint를1-epoch run으로 report | epoch1 score가 separately trained1-epoch run과 differ | `runs.jsonl`에 schedule horizon,stop step record(§4.1) |
| configuration당 seed 하나 | seed spread보다 작은 difference를 effect로 report | seed3개; seed-inclusive interval(§6.3) |
| question10개 MT-Bench category로 decide | item-level half-width보다 작은 category delta를 effect로 report | use 전 half-width compute(§6.1) |
| held-out suite에서 best seed 또는 epoch choose | held-out score가 development trend above; committed decision list 없음 | `git log`의 commit order(§7 item6) |
| arm 사이 packing/GPU count가 다른 mean loss aggregation | same data가 batch layout에 따라 different result | `runs.jsonl`에 fixed sum aggregation record(§2.4 item7) |
| base-vs-SFT IFEval change를 forgetting으로 interpret | ability와 무관하게 base model이 chat format fail | same format의 arm compare(§5.3) |

## 이해 점검

1. narrow-arm run이 HumanEval을5 point improve하고 GSM8K paired interval이[−4.1,−0.3]이다. diverse-arm GSM8K interval은[−1.0,+1.5]다. 어느 result가 forgetting인지와 diverse arm GSM8K entry가 forgetting measurement가 아닌 이유를 설명하라.
2. 3-epoch run epoch-1 checkpoint가 separate1-epoch run보다 덜 forget한다. §4.1 learning-rate arithmetic으로 두 checkpoint가 differ하는 reason 두 가지와 epoch count claim을 support하는 design을 설명하라.
3. correct position ID의 packed run이 one epoch 뒤 unpacked run보다 higher validation loss를 가질 수 있는 이유와 이것이 cross-example attention을 indicate하지 않는 이유를 설명하라.
4. IFEval constraint type으로 만든 Precise IF set을 포함한 Persona dataset removal은 IFEval을19.2 point 낮췄지만 IFEval-OOD는 낮추지 않았다. Tülu3 author의 causal account를 제시하고 diverse arm이 IFEval에서 narrow arm보다 이기지만 IFBench에서는 이기지 않을 때 lab conclusion을 설명하라.
5. §6.3에서 interval은 correction 전 zero를 exclude하지만 comparison20개 Bonferroni correction 뒤 include한다. 무엇이 바뀌었고 memo가 axis에 대해 무엇을 report해야 하는지 설명하라.
6. Shi et al.은 Less MMLU Chat에서 LIMA보다 AlpacaEval1.0 IM advantage가 크지만18-task mean에서는 그렇지 않음을 찾았다. §2.2 token-share arithmetic으로 instruction/output ratio가 gradient 중 instruction token fraction을 어떻게 바꾸며 그 fraction이 non-target suite change를 predict할 필요가 없는 이유를 설명하라.
7. narrow code data epoch4에서 LoRA r=256은 similar HumanEval(0.498 대0.470)에서 full fine-tuning보다 덜 forget했지만, Tülu-v2-mix epoch2에서는 full fine-tuning이 덜 forget했다(forgetting average0.660 대 LoRA r=16의0.650). 둘을 account하는 mechanism 하나와 이를 contradict할 lab observation을 제안하라.
8. pretrained model에서 ln|V| 근처 step-0 loss가 loading error evidence인 이유와 randomly initialized output layer에서는 같은 value가 expected result인 이유를 설명하라.

## 연결

- Previous chapter: [[ch-35a]] — Distillation in Practice B: Prompt Selection, Teacher Sampling, and Quality Filters.
- Next chapter: [[ch-15]] — Human Preference and Instruction Data: Annotation Protocols, Agreement, and Prompt Coverage.
- Dependency: [[ch-30a]] — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control(paired forgetting delta, report format §6).
- [[ch-04]] — Sequence Packing, Loss Masking, and Chat Templates(§2에서 test하는 mechanics).
- [[ch-30]] — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate(§4에서 run하는 axis).
- [[ch-30b]] — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares(two arm beyond mixture composition).
- [[ch-29]] — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification(narrow synthetic arm optional source).
- [[ch-31a]] — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood(this lab은 positive target만 train; decontamination은 negative marginal value로 example remove).
- [[ch-47]] — Evaluation Harness and Suite Design for General Capability(this lab에 depend); [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions; [[ch-53]] — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness(this lab per-item file reuse).
- [[ch-46]] — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention(preference/RL training 뒤 same forgetting report apply).

## 출처

- [[tulu-3]] — Tülu3 report(arXiv:2411.15124v5); report library card.
- [[tulu-3-lab-evidence]] — suite split,decontamination rule,seed,recipe row에 쓴 Tülu3 Table3,7,11,13,14,32 및 §3.2,§4.3.2,§7.4.1 excerpt.
- [[tulu-3-sft-mix]] — diverse,narrow arm build에 쓴 Tülu3 SFT mixture component.
- [[loss-masking-prompt]] — Shi et al., IT/IM objective(Eq.2,4),Table1,5,6,Fig.2-3; same-name chapter excerpt에는 library card summary correction이 있다.
- [[lora-learns-less-forgets-less]] — Biderman et al., TableS5-S13,§4.1-4.2,App.A-C: epoch,domain,LoRA rank별 forgetting.
- [[tulu-1-how-far-can-camels-go]] — Wang et al., Table1 dataset size, Table3 13B single-dataset 대 mixture comparison, §4.2 base-model AlpacaEval note,App.D setting,App.E GSM subsample.
- [[benchmark-variance-quantified]] — Madaan et al., analytic CI formula,Table1 seed standard deviation/CI half-width,MMLU cloze result.
- [[adding-error-bars-evals]] — Miller, unpaired/paired standard error(Eq.5-7).
- [[ifeval]] — Zhou et al., instruction type25개,prompt541개,strict/loose metric.
- [[ifbench]] — Pyatkin et al., IFEval 대 unseen constraint58개.
- [[open-instruct-allenai-recipes]] — open-instruct doc: OLMo3 template error,loss-reduction drift,OLMo2 1B stage time.
- [[open-instruct-allenai-recipes-recipe]] — verified OLMo-2-0425-1B-SFT recipe row.
- [[packing-position-ids]] — Kundu et al., padding-free collator mechanism 및 Table2 step-count effect.
- [[hf-chat-template-assistant-mask]] — transformers v4.46.0 assistant-mask code 및 OLMo2 1B Instruct template.
- [[karpathy-training-neural-net-recipe]] — “verify loss @ init”, “overfit one batch”, network input tensor decoding(section2).
- [[signal-and-noise-eval]] — Heineman et al., pretraining ablation의 benchmark signal-to-noise와 small-subset decision accuracy.
- [[judge-llm-bias]] — Zheng et al., MT-Bench construction: question80개, category8개 각각10개(§2.2).
